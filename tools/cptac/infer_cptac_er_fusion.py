"""
CPTAC-BRCA external validation: ER, WSI + clinicopath gated fusion
==================================================================
Applies the TCGA-trained `er_wsi_clinpath_gated` folds to CPTAC without
retraining. The per-fold standardisation is reloaded from
s_{fold}_tabular_transform.json, so each fold sees CPTAC through exactly the
transform it was fitted with on its own training split.

Model construction and checkpoint loading follow tools/diagnostics/gate_probe.py,
which validated this same reconstruction against the saved per-fold pickles to
max |dprob| = 1.8e-07. Columns are matched to the transform by NAME rather than
position, so a reordered table cannot silently misalign features.

Tabular features are per-case; every slide of a case is scored with that case's
vector, and case-level probabilities are the mean over the case's slides -- the
same convention as infer_cptac_er.py, so the two arms are directly comparable.

    python tools/cptac/infer_cptac_er_fusion.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "project" / "CLAM"))

from models.model_multimodal import CLAMRNAFusion  # noqa: E402

LABEL_MAP = {0: "ER-negative", 1: "ER-positive"}
POS_INDEX = 1
CLAM_KWARGS = dict(gate=True, size_arg="big", dropout=0.5, k_sample=4,
                   n_classes=2, subtyping=False, embed_dim=1536)


def parse_args():
    parser = argparse.ArgumentParser(description="ER fusion external validation on CPTAC")
    parser.add_argument("--feature_dir", type=str,
                        default=".datasets/cptac-brca/embeddings")
    parser.add_argument("--dataset_csv", type=str,
                        default=".datasets/cptac-brca/cptac_brca_er_dataset.csv")
    parser.add_argument("--tabular_csv", type=str,
                        default=".datasets/cptac-brca/cptac_brca_er_clinicopath_clam.csv")
    parser.add_argument("--ckpt_dir", type=str,
                        default=".scratch/results/er/er_wsi_clinpath_gated_s1")
    parser.add_argument("--output_dir", type=str,
                        default=".scratch/cptac_validation/results/er_clinpath")
    parser.add_argument("--fusion_mode", type=str, default="gated")
    parser.add_argument("--fusion_hidden_dim", type=int, default=32)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--ablate_table", action="store_true",
                        help="replace the standardised tabular vector with all-zeros, i.e. the "
                             "fold's own training mean -- the 'table absent' condition of "
                             "tools/diagnostics/gate_probe.py. Output dir gets a _table_absent "
                             "suffix unless --output_dir is given explicitly.")
    args = parser.parse_args()
    if args.ablate_table and "--output_dir" not in sys.argv:
        args.output_dir = args.output_dir.rstrip("/") + "_table_absent"
    return args


def index_features(feature_dir):
    index = {}
    for path in sorted(Path(feature_dir).rglob("*.h5"), key=lambda p: len(p.parts)):
        if path.is_file() and path.stem not in index:
            index[path.stem] = str(path)
    return index


def load_transform(path):
    payload = json.loads(Path(path).read_text())
    return (list(payload["selected_feature_names"]),
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["std"], dtype=np.float32))


def load_model(ckpt_path, tabular_dim, args):
    model = CLAMRNAFusion(
        wsi_model_type="clam_mb",
        tabular_input_dim=tabular_dim,
        tabular_hidden_dim=256,
        tabular_num_layers=2,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_mode=args.fusion_mode,
        **CLAM_KWARGS,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical = [k for k in missing if "instance_loss_fn" not in k]
    if critical or unexpected:
        raise RuntimeError(f"{ckpt_path}: missing={critical[:4]} unexpected={unexpected[:4]}")
    return model.eval()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = pd.read_csv(args.dataset_csv)
    tabular = pd.read_csv(args.tabular_csv).set_index("case_id")
    features = index_features(args.feature_dir)

    missing_tab = sorted(set(dataset["case_id"]) - set(tabular.index))
    if missing_tab:
        print(f"WARNING: {len(missing_tab)} cases lack a tabular row, dropping their slides")
        dataset = dataset[~dataset["case_id"].isin(missing_tab)].reset_index(drop=True)
    missing_feat = sorted(set(dataset["slide_id"]) - set(features))
    if missing_feat:
        print(f"WARNING: {len(missing_feat)} slides lack features, dropping")
        dataset = dataset[~dataset["slide_id"].isin(missing_feat)].reset_index(drop=True)
    print(f"dataset: {len(dataset)} slides / {dataset['case_id'].nunique()} cases")

    # Cache each slide's features once; every fold reuses them.
    fold_probs = []
    for fold in range(args.n_folds):
        ckpt = os.path.join(args.ckpt_dir, f"s_{fold}_checkpoint.pt")
        tpath = os.path.join(args.ckpt_dir, f"s_{fold}_tabular_transform.json")
        if not (os.path.exists(ckpt) and os.path.exists(tpath)):
            print(f"fold {fold}: checkpoint or transform missing, skipping")
            continue

        names, mean, std = load_transform(tpath)
        absent = [n for n in names if n not in tabular.columns]
        if absent:
            raise SystemExit(f"fold {fold}: tabular table lacks {absent[:5]}")
        raw = tabular[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        scaled = np.nan_to_num((raw - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        if args.ablate_table:
            # Post-standardisation zeros ARE the fold's training mean, so this feeds the
            # head a table carrying no case-specific information rather than a null vector.
            scaled = np.zeros_like(scaled)
        by_case = {case: scaled[i] for i, case in enumerate(tabular.index)}

        model = load_model(ckpt, len(names), args).to(device)
        probs = np.zeros(len(dataset))
        for i, row in enumerate(dataset.itertuples(index=False)):
            with h5py.File(features[row.slide_id], "r") as handle:
                wsi = torch.from_numpy(handle["features"][:]).float().to(device)
            tab = torch.from_numpy(by_case[row.case_id]).float().unsqueeze(0).to(device)
            with torch.inference_mode():
                _, Y_prob, _, _, _ = model((wsi, tab))
            probs[i] = Y_prob.cpu().numpy().reshape(-1)[POS_INDEX]

        frame = dataset[["case_id", "slide_id", "label"]].rename(
            columns={"label": "true_label"}).copy()
        frame["p_ER_positive"] = probs
        frame.to_csv(os.path.join(args.output_dir, f"fold_{fold}_predictions.csv"), index=False)

        case = frame.groupby("case_id").agg(true_label=("true_label", "first"),
                                            p_ER_positive=("p_ER_positive", "mean"))
        from sklearn.metrics import roc_auc_score
        print(f"fold {fold}: slide AUROC {roc_auc_score(frame.true_label, frame.p_ER_positive):.4f}  "
              f"case AUROC {roc_auc_score(case.true_label, case.p_ER_positive):.4f}")
        fold_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    if not fold_probs:
        raise SystemExit(f"no usable checkpoints under {args.ckpt_dir}")

    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, roc_auc_score)
    mean_probs = np.mean(fold_probs, axis=0)
    slide_df = dataset[["case_id", "slide_id", "label", "label_name"]].rename(
        columns={"label": "true_label", "label_name": "true_name"}).copy()
    slide_df["p_ER_positive"] = mean_probs
    slide_df.to_csv(os.path.join(args.output_dir, "ensemble_slide_predictions.csv"), index=False)

    case_df = slide_df.groupby("case_id").agg(
        true_label=("true_label", "first"),
        p_ER_positive=("p_ER_positive", "mean"),
        n_slides=("slide_id", "count")).reset_index()
    case_df["true_name"] = case_df["true_label"].map(LABEL_MAP)
    case_df.to_csv(os.path.join(args.output_dir, "ensemble_case_predictions.csv"), index=False)

    def block(y, p):
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {"n": int(len(y)), "n_pos": int(y.sum()),
                "auroc": float(roc_auc_score(y, p)),
                "auprc_pos": float(average_precision_score(y, p)),
                "balanced_acc": float(balanced_accuracy_score(y, pred)),
                "sensitivity_ERpos": float(tp / (tp + fn)),
                "specificity_ERneg": float(tn / (tn + fp)),
                "confusion_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}

    slide_m = block(slide_df.true_label.values, slide_df.p_ER_positive.values)
    case_m = block(case_df.true_label.values, case_df.p_ER_positive.values)
    summary = {"arm": f"er_wsi_clinpath_{args.fusion_mode} (TCGA-trained, frozen)",
               "cohort": "CPTAC-BRCA", "n_folds_used": len(fold_probs),
               "ensemble": {"slide_level": slide_m, "case_level": case_m}}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Ensemble, case level ===")
    for key, value in case_m.items():
        print(f"  {key:20s} {value}")
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
