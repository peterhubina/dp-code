"""
PAM50 WSI + tabular fusion inference on CPTAC-BRCA
==================================================
Runs a 10-fold multimodal ensemble trained on TCGA-BRCA over CPTAC slides, with
frozen weights and no fine-tuning. The tabular modality is whatever the run was
trained on -- RNA expression (`.scratch/rna-gdc/...`) or arm-level CNV
(`.scratch/cnv-tabular/...`); the script only cares that the fold transforms and
the CPTAC table share feature *names*.

Each fold ships its own tabular transform (s_<fold>_tabular_transform.json): the
features it selected on the training fold, plus that fold's per-feature mean and
std. Those are applied here by *feature name*, not by the stored column index,
because the CPTAC table has a different column order. Features the fold selected
that CPTAC does not carry are imputed at the training mean, i.e. z=0, and the
count is reported per fold -- under the GDC-derived RNA tables and the 39-arm CNV
tables that count should be zero, and a non-zero value means the two tables are
not on a common feature axis.

Architecture is read from the checkpoint directory's `experiment_*.txt` (the dict
CLAM's main.py writes next to the checkpoints), so every fusion operator in the
ladder -- concat, gated, cross_attention, film_attention, coattn -- loads with
strict=True without the caller having to remember film_rank, modality_dropout or
the co-attention group spec. An explicit CLI flag that contradicts the txt is a
hard error (exit 2); a flag left unset takes the txt value, and when no txt
exists the legacy defaults in LEGACY_ARCH apply.

--rna_ablate replaces every tabular input with the training mean (z=0), which
turns the fusion model into its WSI branch plus a constant. Comparing a normal
run to an ablated one shows how much the tabular branch actually contributes on
CPTAC.
"""

import argparse
import ast
import glob
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "project", "CLAM"))

from models.model_multimodal import CLAMRNAFusion
from utils.tabular_groups import build_tabular_groups

LABEL_MAP = {0: "LumA", 1: "LumB", 2: "Basal", 3: "Her2"}

REPO_ROOT = Path(__file__).resolve().parents[2]

# What this script assumed before it learned to read experiment_*.txt. Applied only
# when the checkpoint dir carries no txt, so pre-existing invocations keep behaving.
LEGACY_ARCH = {
    "fusion_mode": "gated",
    "tabular_hidden_dim": 256,
    "tabular_num_layers": 2,
    "fusion_hidden_dim": 32,
    "film_rank": 32,
    "modality_dropout": 0.0,
    "tabular_group_spec": None,
    "model_size": "big",
    "dropout": 0.5,
    "k_sample": 4,
    "wsi_model_type": "clam_mb",
}

# resolved-arch key -> key as CLAM's experiment_*.txt spells it
TXT_KEYS = {
    "fusion_mode": "fusion_mode",
    "tabular_hidden_dim": "tabular_hidden_dim",
    "tabular_num_layers": "tabular_num_layers",
    "fusion_hidden_dim": "fusion_hidden_dim",
    "film_rank": "film_rank",
    "modality_dropout": "modality_dropout",
    "tabular_group_spec": "tabular_group_spec",
    "model_size": "model_size",
    "dropout": "use_drop_out",
    "k_sample": "B",
    "wsi_model_type": "model_type",
}

# resolved-arch keys the caller can override on the command line
CLI_ARCH_KEYS = (
    "fusion_mode", "tabular_hidden_dim", "tabular_num_layers", "fusion_hidden_dim",
    "film_rank", "modality_dropout", "tabular_group_spec", "model_size", "dropout",
)


def parse_args():
    parser = argparse.ArgumentParser(description="PAM50 fusion inference on CPTAC-BRCA")
    parser.add_argument("--feature_dir", default=".datasets/cptac-brca/embeddings")
    parser.add_argument("--dataset_csv", default=".datasets/cptac-brca/cptac_brca_pam50_dataset.csv")
    parser.add_argument("--tabular_csv", default=".scratch/rna-gdc/CPTAC_BRCA_RNA_gdc_4class_clam.csv.gz")
    parser.add_argument("--ckpt_dir", default=".scratch/results/pam50_wsi_rna_gatedfusion_gdc_s1")
    parser.add_argument("--output_dir", default=".scratch/cptac_validation/results/predictions_fusion")
    parser.add_argument("--rna_ablate", action="store_true",
                        help="Replace the tabular modality with the training mean (z=0) "
                             "to isolate the WSI branch")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an output dir that already holds "
                             "ensemble_predictions.csv")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                        help="Resolve the architecture, build fold 0 and load its checkpoint "
                             "with strict=True, print the resolved dict and exit without inference")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=1536)

    # Architecture flags default to None: unset means "take it from the checkpoint dir's
    # experiment_*.txt", and disagreeing with that txt is an error rather than a silent win.
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--fusion_mode", default=None)
    parser.add_argument("--fusion_hidden_dim", type=int, default=None)
    parser.add_argument("--tabular_hidden_dim", type=int, default=None)
    parser.add_argument("--tabular_num_layers", type=int, default=None)
    parser.add_argument("--film_rank", type=int, default=None,
                        help="FiLM bottleneck rank; >0 creates film_bottleneck/gamma/beta")
    parser.add_argument("--modality_dropout", type=float, default=None,
                        help=">0 creates the tabular_absent_embedding parameter")
    parser.add_argument("--tabular_group_spec", default=None,
                        help="'prefix' or a signature CSV; required by fusion_mode=coattn")
    return parser.parse_args()


def read_experiment_txt(ckpt_dir):
    """Return the settings dict CLAM's main.py wrote beside the checkpoints, or None."""
    matches = sorted(glob.glob(os.path.join(ckpt_dir, "experiment_*.txt")))
    if not matches:
        return None, None
    if len(matches) > 1:
        sys.exit("ERROR: {} holds {} experiment_*.txt files ({}); cannot tell which run "
                 "produced the checkpoints.".format(ckpt_dir, len(matches),
                                                    ", ".join(os.path.basename(m) for m in matches)))
    return ast.literal_eval(Path(matches[0]).read_text()), matches[0]


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return a == b
    return a == b


def resolve_architecture(args):
    """Merge CLI flags with the checkpoint dir's experiment_*.txt; conflicts exit 2."""
    settings, txt_path = read_experiment_txt(args.ckpt_dir)
    arch, source = {}, {}
    conflicts = []

    for key, default in LEGACY_ARCH.items():
        cli = getattr(args, key, None) if key in CLI_ARCH_KEYS else None
        txt_key = TXT_KEYS[key]
        has_txt = settings is not None and txt_key in settings
        txt = settings[txt_key] if has_txt else None

        if cli is not None and has_txt and not _same(cli, txt):
            conflicts.append("  {}: --{}={!r} but {} says {!r}".format(
                key, key, cli, os.path.basename(txt_path), txt))
        if cli is not None:
            arch[key], source[key] = cli, "cli"
        elif has_txt:
            arch[key], source[key] = txt, "txt"
        else:
            arch[key], source[key] = default, "legacy default"

    if conflicts:
        print("ERROR: architecture flags contradict {}:".format(txt_path), file=sys.stderr)
        print("\n".join(conflicts), file=sys.stderr)
        print("Drop the flag to use the checkpoint's own value.", file=sys.stderr)
        sys.exit(2)

    spec = arch["tabular_group_spec"]
    if spec is not None and spec != "prefix" and not os.path.isabs(spec):
        arch["tabular_group_spec"] = str(REPO_ROOT / spec)

    # Not recorded in the txt: fixed by the CPTAC feature extraction and the label set.
    arch["embed_dim"] = args.embed_dim
    arch["n_classes"] = args.n_classes
    arch["subtyping"] = True

    print("Resolved architecture ({}):".format(
        txt_path if txt_path else "no experiment_*.txt in --ckpt_dir; legacy defaults"))
    for key in list(LEGACY_ARCH) + ["embed_dim", "n_classes", "subtyping"]:
        print("  {:22s} {!r:<28s} [{}]".format(key, arch[key], source.get(key, "cli/const")))
    return arch


def resolve_group_indices(arch, selected_feature_names):
    """Co-attention token grouping, built the way core_utils.py builds it at train time."""
    if arch["fusion_mode"] != "coattn":
        return None
    spec = arch["tabular_group_spec"]
    if not spec:
        sys.exit("ERROR: fusion_mode 'coattn' needs --tabular_group_spec (or a "
                 "tabular_group_spec entry in the checkpoint's experiment_*.txt).")
    group_names, group_indices = build_tabular_groups(list(selected_feature_names), spec)
    print("  co-attention tokens: {}".format(
        ", ".join("{}({})".format(n, len(i)) for n, i in zip(group_names, group_indices))))
    return group_indices


def index_features(feature_dir):
    paths = sorted(Path(feature_dir).rglob("*.h5"), key=lambda p: len(p.parts))
    index = {}
    for path in paths:
        if path.is_file() and path.stem not in index:
            index[path.stem] = str(path)
    return index


def load_model(ckpt_path, tabular_input_dim, arch, device, tabular_group_indices=None):
    # Mirrors project/CLAM/utils/core_utils.py:348-363, the call that built these weights.
    model = CLAMRNAFusion(
        wsi_model_type=arch["wsi_model_type"],
        gate=True,
        size_arg=arch["model_size"],
        dropout=arch["dropout"],
        k_sample=arch["k_sample"],
        n_classes=arch["n_classes"],
        subtyping=arch["subtyping"],
        embed_dim=arch["embed_dim"],
        tabular_input_dim=tabular_input_dim,
        tabular_hidden_dim=arch["tabular_hidden_dim"],
        tabular_num_layers=arch["tabular_num_layers"],
        fusion_hidden_dim=arch["fusion_hidden_dim"],
        fusion_mode=arch["fusion_mode"],
        film_rank=arch["film_rank"],
        modality_dropout=arch["modality_dropout"],
        tabular_group_indices=tabular_group_indices,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    clean = {k.replace(".module", ""): v for k, v in ckpt.items() if "instance_loss_fn" not in k}
    model.load_state_dict(clean, strict=True)
    model.to(device)
    model.eval()
    return model


def apply_transform(rna_table, transform, ablate=False):
    """Select the fold's genes by name, z-score them, impute absent genes at z=0."""
    names = list(transform["selected_feature_names"])
    mean = np.asarray(transform["mean"], dtype=np.float32)
    std = np.asarray(transform["std"], dtype=np.float32)

    present = rna_table.columns.intersection(names)
    missing = [n for n in names if n not in present]

    # start every gene at its training mean, then overwrite the ones CPTAC carries
    matrix = np.tile(mean, (len(rna_table), 1))
    if not ablate and len(present):
        pos = {name: i for i, name in enumerate(names)}
        cols = [pos[n] for n in present]
        matrix[:, cols] = rna_table[present].to_numpy(dtype=np.float32)

    return ((matrix - mean) / std).astype(np.float32), len(missing)


def fold_paths(ckpt_dir, fold):
    return (os.path.join(ckpt_dir, f"s_{fold}_checkpoint.pt"),
            os.path.join(ckpt_dir, f"s_{fold}_tabular_transform.json"))


def dry_run(args, arch, device):
    """Build fold 0 and load its weights with strict=True; no data, no inference."""
    ckpt_path, transform_path = fold_paths(args.ckpt_dir, 0)
    if not os.path.exists(ckpt_path):
        sys.exit(f"ERROR: no fold-0 checkpoint at {ckpt_path}")
    transform = json.loads(Path(transform_path).read_text())
    names = list(transform["selected_feature_names"])
    group_indices = resolve_group_indices(arch, names)
    load_model(ckpt_path, len(names), arch, device, tabular_group_indices=group_indices)
    print(f"DRY RUN: fold 0 loaded with strict=True from {ckpt_path} "
          f"({len(names)} tabular features)")
    print(json.dumps(arch, indent=2, sort_keys=True))
    return 0


def main():
    args = parse_args()
    arch = resolve_architecture(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dry_run:
        return dry_run(args, arch, device)

    existing = os.path.join(args.output_dir, "ensemble_predictions.csv")
    if os.path.exists(existing) and not args.overwrite:
        sys.exit(f"ERROR: {existing} already exists. Pass --overwrite to replace it, "
                 "or point --output_dir somewhere else.")
    os.makedirs(args.output_dir, exist_ok=True)
    Path(args.output_dir, "inference_args.json").write_text(json.dumps({
        "architecture": arch,
        "ckpt_dir": os.path.abspath(args.ckpt_dir),
        "tabular_csv": os.path.abspath(args.tabular_csv),
        "feature_dir": os.path.abspath(args.feature_dir),
        "dataset_csv": os.path.abspath(args.dataset_csv),
        "n_folds": args.n_folds,
        "rna_ablate": bool(args.rna_ablate),
    }, indent=2, sort_keys=True) + "\n")

    if args.rna_ablate:
        print("TABULAR ABLATION: all tabular inputs held at the training mean (z=0)")

    dataset = pd.read_csv(args.dataset_csv)
    feature_paths = index_features(args.feature_dir)
    missing_wsi = [s for s in dataset["slide_id"] if s not in feature_paths]
    if missing_wsi:
        print(f"WARNING: {len(missing_wsi)} slides missing WSI features, skipping them")
        dataset = dataset[~dataset["slide_id"].isin(missing_wsi)].reset_index(drop=True)

    rna = pd.read_csv(args.tabular_csv)
    rna_by_case = rna.set_index("case_id").drop(
        columns=[c for c in ("sample", "label", "sample_type_code") if c in rna.columns])
    missing_rna = sorted(set(dataset["case_id"]) - set(rna_by_case.index))
    if missing_rna:
        print(f"WARNING: {len(missing_rna)} cases missing tabular data, "
              f"dropping their slides: {missing_rna}")
        dataset = dataset[~dataset["case_id"].isin(missing_rna)].reset_index(drop=True)

    print(f"Dataset: {len(dataset)} slides / {dataset['case_id'].nunique()} cases")
    print(f"Tabular table: {rna_by_case.shape[0]} cases x {rna_by_case.shape[1]} features")

    all_fold_probs = []
    imputed_counts = []

    for fold in range(args.n_folds):
        ckpt_path, transform_path = fold_paths(args.ckpt_dir, fold)
        if not os.path.exists(ckpt_path):
            print(f"Fold {fold}: checkpoint not found at {ckpt_path}, skipping")
            continue

        transform = json.loads(Path(transform_path).read_text())
        rna_matrix, n_missing = apply_transform(rna_by_case, transform, ablate=args.rna_ablate)
        rna_lookup = {case: rna_matrix[i] for i, case in enumerate(rna_by_case.index)}
        imputed_counts.append(n_missing)

        names = list(transform["selected_feature_names"])
        print(f"\nFold {fold}: {len(names)} tabular features, "
              f"{n_missing} absent from CPTAC and imputed at training mean")
        # Recomputed per fold: the grouping follows the fold's own selected feature names.
        group_indices = resolve_group_indices(arch, names)
        model = load_model(ckpt_path, len(names), arch, device,
                           tabular_group_indices=group_indices)

        fold_results = []
        for _, row in dataset.iterrows():
            with h5py.File(feature_paths[row["slide_id"]], "r") as f:
                wsi = torch.from_numpy(f["features"][:]).float().to(device)
            tab = torch.from_numpy(rna_lookup[row["case_id"]]).float().unsqueeze(0).to(device)

            with torch.inference_mode():
                _, Y_prob, Y_hat, _, _ = model((wsi, tab))

            probs = Y_prob.cpu().numpy().squeeze()
            fold_results.append({
                "slide_id": row["slide_id"],
                "case_id": row["case_id"],
                "true_label": int(row["label"]),
                "true_name": LABEL_MAP[int(row["label"])],
                "pred_label": Y_hat.item(),
                "pred_name": LABEL_MAP[Y_hat.item()],
                "p_LumA": probs[0], "p_LumB": probs[1],
                "p_Basal": probs[2], "p_Her2": probs[3],
            })

        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(os.path.join(args.output_dir, f"fold_{fold}_predictions.csv"), index=False)
        print(f"  Fold {fold} accuracy: {(fold_df['true_label'] == fold_df['pred_label']).mean():.4f}")
        all_fold_probs.append(fold_df[["p_LumA", "p_LumB", "p_Basal", "p_Her2"]].values)

        del model
        torch.cuda.empty_cache()

    if all_fold_probs:
        mean_probs = np.mean(all_fold_probs, axis=0)
        ensemble = dataset[["slide_id", "case_id", "label"]].rename(columns={"label": "true_label"}).copy()
        ensemble["true_name"] = ensemble["true_label"].map(LABEL_MAP)
        ensemble["pred_label"] = mean_probs.argmax(axis=1)
        ensemble["pred_name"] = ensemble["pred_label"].map(LABEL_MAP)
        ensemble["p_LumA"] = mean_probs[:, 0]
        ensemble["p_LumB"] = mean_probs[:, 1]
        ensemble["p_Basal"] = mean_probs[:, 2]
        ensemble["p_Her2"] = mean_probs[:, 3]
        ensemble["max_prob"] = mean_probs.max(axis=1)

        path = os.path.join(args.output_dir, "ensemble_predictions.csv")
        ensemble.to_csv(path, index=False)

        print(f"\n=== Ensemble Results (slide-level) ===")
        print(f"Slides: {len(ensemble)}  Cases: {ensemble['case_id'].nunique()}")
        print(f"Accuracy: {(ensemble['true_label'] == ensemble['pred_label']).mean():.4f}")
        print(f"Tabular features imputed per fold: min {min(imputed_counts)}, max {max(imputed_counts)}")
        print(f"\nPredicted distribution:")
        print(ensemble["pred_name"].value_counts().to_string())
        print(f"\nTrue distribution:")
        print(ensemble["true_name"].value_counts().to_string())
        print(f"\nSaved ensemble predictions: {path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
