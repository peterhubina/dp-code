"""Forward-pass-only probe of the gated WSI+tabular ER fusion checkpoints.

Answers: did the sigmoid fusion gate collapse onto one modality?

Nothing is trained or written back into the experiment directories. The script
re-runs the frozen CLAM branch once per (slide, fold) to cache the pooled WSI
vector, then replays the trained fusion head three ways per test slide:

  intact          -- exactly the training-time forward pass
  table absent    -- tabular input := fold train mean (= zeros post-standardisation)
  image absent    -- pooled WSI := fold train-split mean pooled WSI vector

Usage (from repo root):
    python tools/diagnostics/gate_probe.py
    python tools/diagnostics/gate_probe.py --skip-cache   # reuse pooled WSI cache
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
CLAM_DIR = REPO / "project" / "CLAM"
sys.path.insert(0, str(CLAM_DIR))

from dataset_modules.multimodal_dataset import (  # noqa: E402
    TabularFeatureStore,
    read_tabular_feature_table,
)
from dataset_modules.rna_dataset import RNAFeatureTransform  # noqa: E402
from models.model_clam import CLAM_MB  # noqa: E402
from models.model_multimodal import CLAMRNAFusion  # noqa: E402

RESULTS = REPO / ".scratch" / "results" / "er"
OUT_DIR = RESULTS / "diagnostics"
EMBED_DIR = REPO / ".datasets" / "tcga-brca" / "embeddings"
SPLIT_DIR = CLAM_DIR / "splits" / "tcga_brca_er_100"
MANIFEST = CLAM_DIR / "dataset_csv" / "tcga_brca_er.csv"
CACHE = OUT_DIR / "pooled_wsi_cache.npz"

LABEL_DICT = {"ER-negative": 0, "ER-positive": 1}
N_FOLDS = 10

# Recorded in .scratch/results/er/<exp>_s1/experiment_<exp>.txt
CLAM_KWARGS = dict(
    gate=True, size_arg="big", dropout=0.5, k_sample=4, n_classes=2,
    subtyping=False, embed_dim=1536,
)
ARMS = {
    "er_wsi_rna_gated": REPO / ".scratch" / "TCGA-BRCA-rna" / "tcga_brca_er_rna_clam.csv.gz",
    "er_wsi_clinpath_gated": REPO / "tools" / "data" / "tcga_brca_clinicopath_clam.csv",
}


def fold_split(fold: int, key: str) -> list[str]:
    df = pd.read_csv(SPLIT_DIR / f"splits_{fold}.csv")
    return df[key].dropna().astype(str).tolist()


def build_pooled_cache(device: torch.device) -> None:
    """Run the frozen CLAM branch of every fold over every manifest slide once."""
    manifest = pd.read_csv(MANIFEST)
    slides = manifest["slide_id"].astype(str).tolist()
    missing = [s for s in slides if not (EMBED_DIR / f"{s}.h5").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} slides lack .h5 features, e.g. {missing[:3]}")

    branches = []
    for fold in range(N_FOLDS):
        ckpt = torch.load(RESULTS / "er_wsi_rna_gated_s1" / f"s_{fold}_checkpoint.pt", map_location="cpu")
        state = {k[len("wsi."):]: v for k, v in ckpt.items() if k.startswith("wsi.")}
        branch = CLAM_MB(**CLAM_KWARGS)
        missing_k, unexpected_k = branch.load_state_dict(state, strict=False)
        critical = [k for k in missing_k if "instance_loss_fn" not in k]
        if critical or unexpected_k:
            raise RuntimeError(f"fold {fold}: missing={critical[:4]} unexpected={unexpected_k[:4]}")
        branches.append(branch.to(device).eval())

    pooled = np.zeros((len(slides), N_FOLDS, 512), dtype=np.float32)

    def read(slide: str) -> np.ndarray:
        with h5py.File(EMBED_DIR / f"{slide}.h5", "r") as fh:
            return np.asarray(fh["features"][:], dtype=np.float32)

    # Bounded prefetch: Executor.map would queue every 100 MB slide at once.
    depth = 4
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = deque(pool.submit(read, s) for s in slides[:depth])
        for idx, slide in enumerate(slides):
            if idx + depth < len(slides):
                pending.append(pool.submit(read, slides[idx + depth]))
            feats = pending.popleft().result()
            h = torch.from_numpy(feats).to(device)
            del feats
            with torch.inference_mode():
                for fold, branch in enumerate(branches):
                    _, _, _, _, res = branch(h, return_features=True)
                    pooled[idx, fold] = (
                        CLAMRNAFusion._pool_wsi_features(res["features"]).squeeze(0).float().cpu().numpy()
                    )
            del h
            if (idx + 1) % 100 == 0:
                print(f"  pooled {idx + 1}/{len(slides)} slides", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, slide_ids=np.array(slides, dtype=object), pooled=pooled)
    print(f"wrote {CACHE}")


def load_cache() -> tuple[dict[str, int], np.ndarray]:
    blob = np.load(CACHE, allow_pickle=True)
    slides = [str(s) for s in blob["slide_ids"]]
    return {s: i for i, s in enumerate(slides)}, blob["pooled"]


def load_fusion_model(arm: str, fold: int, tabular_dim: int) -> CLAMRNAFusion:
    model = CLAMRNAFusion(
        wsi_model_type="clam_mb",
        tabular_input_dim=tabular_dim,
        tabular_hidden_dim=256,
        tabular_num_layers=2,
        fusion_hidden_dim=32,
        fusion_mode="gated",
        **CLAM_KWARGS,
    )
    state = torch.load(RESULTS / f"{arm}_s1" / f"s_{fold}_checkpoint.pt", map_location="cpu")
    missing_k, unexpected_k = model.load_state_dict(state, strict=False)
    critical = [k for k in missing_k if "instance_loss_fn" not in k]
    if critical or unexpected_k:
        raise RuntimeError(f"{arm} fold {fold}: missing={critical[:4]} unexpected={unexpected_k[:4]}")
    return model.eval()


def transform_from_json(path: Path) -> RNAFeatureTransform:
    payload = json.loads(path.read_text())
    return RNAFeatureTransform(
        selected_idx=np.asarray(payload["selected_idx"], dtype=np.int64),
        selected_feature_names=list(payload["selected_feature_names"]),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def case_level(records: pd.DataFrame, col: str) -> pd.DataFrame:
    return records.groupby("case_id", as_index=False)[col].mean()


def summarise(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def probe_arm(arm: str, tabular_csv: Path, slide_index: dict[str, int], pooled: np.ndarray) -> dict:
    manifest = pd.read_csv(MANIFEST)
    manifest["case_id"] = manifest["case_id"].astype(str)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest["label_idx"] = manifest["label"].map(LABEL_DICT).astype(int)

    metadata, features, feature_names = read_tabular_feature_table(tabular_csv, label_dict=LABEL_DICT)
    store = TabularFeatureStore(metadata, features, feature_names)
    matched = manifest[manifest["case_id"].isin(store.available_cases)].reset_index(drop=True)

    rows: list[dict] = []
    gate_rows: list[dict] = []
    per_fold: list[dict] = []

    for fold in range(N_FOLDS):
        transform = transform_from_json(RESULTS / f"{arm}_s1" / f"s_{fold}_tabular_transform.json")
        store.set_transform(transform)
        model = load_fusion_model(arm, fold, tabular_dim=len(transform.selected_feature_names))

        test_slides = [s for s in fold_split(fold, "test") if s in set(matched["slide_id"])]
        train_slides = [s for s in fold_split(fold, "train") if s in set(matched["slide_id"])]
        if not test_slides or not train_slides:
            raise RuntimeError(f"{arm} fold {fold}: empty split after tabular matching")

        # "image absent" reference: mean pooled WSI vector over this fold's TRAIN slides.
        train_mean_wsi = torch.from_numpy(
            pooled[[slide_index[s] for s in train_slides], fold].mean(axis=0, keepdims=True)
        )
        zero_tab = torch.zeros(1, len(transform.selected_feature_names), dtype=torch.float32)

        saved = pickle.load(open(RESULTS / f"{arm}_s1" / f"split_{fold}_results.pkl", "rb"))
        case_by_slide = dict(zip(matched["slide_id"], matched["case_id"]))
        label_by_slide = dict(zip(matched["slide_id"], matched["label_idx"]))

        fold_gates = []
        for slide in test_slides:
            case_id = case_by_slide[slide]
            wsi_vec = torch.from_numpy(pooled[slide_index[slide], fold][None, :])
            tab_vec = store.get(case_id).unsqueeze(0)

            with torch.inference_mode():
                enc_tab = model.tabular_encoder(tab_vec)
                logits, _, gate = model._gated_fusion(wsi_vec, enc_tab)
                prob = torch.softmax(logits, dim=1)[0, 1].item()

                enc_zero = model.tabular_encoder(zero_tab)
                logits_nt, _, _ = model._gated_fusion(wsi_vec, enc_zero)
                prob_no_table = torch.softmax(logits_nt, dim=1)[0, 1].item()

                logits_ni, _, _ = model._gated_fusion(train_mean_wsi, enc_tab)
                prob_no_image = torch.softmax(logits_ni, dim=1)[0, 1].item()

            gate_np = gate.squeeze(0).numpy()
            fold_gates.append(gate_np)
            rows.append({
                "fold": fold, "slide_id": slide, "case_id": case_id,
                "label": label_by_slide[slide],
                "prob_intact": prob,
                "prob_no_table": prob_no_table,
                "prob_no_image": prob_no_image,
                "prob_saved": float(saved[slide]["prob"].reshape(-1)[1]),
                "gate_mean": float(gate_np.mean()),
                "gate_saved": float(saved[slide].get("fusion_wsi_gate_mean", np.nan)),
            })
            gate_rows.append(gate_np)

        fold_gates = np.stack(fold_gates)
        per_fold.append({
            "fold": fold,
            "n_test_slides": len(test_slides),
            "n_train_slides": len(train_slides),
            "gate_mean": float(fold_gates.mean()),
            "gate_std_across_cases": float(fold_gates.mean(axis=1).std(ddof=0)),
            "gate_std_within_case_dims": float(fold_gates.std(axis=1, ddof=0).mean()),
        })

    df = pd.DataFrame(rows)
    gates = np.stack(gate_rows)  # [n_test_slides, 32]

    max_disc = float(np.abs(df["prob_intact"] - df["prob_saved"]).max())
    max_gate_disc = float(np.abs(df["gate_mean"] - df["gate_saved"]).max())

    per_case = df.groupby("case_id", as_index=False).agg(
        label=("label", "first"),
        prob_intact=("prob_intact", "mean"),
        prob_no_table=("prob_no_table", "mean"),
        prob_no_image=("prob_no_image", "mean"),
        gate_mean=("gate_mean", "mean"),
    )
    auc = {
        key: float(roc_auc_score(per_case["label"], per_case[key]))
        for key in ("prob_intact", "prob_no_table", "prob_no_image")
    }

    return {
        "arm": arm,
        "tabular_csv": str(tabular_csv),
        "tabular_dim": len(transform.selected_feature_names),
        "n_test_slides": int(len(df)),
        "n_test_cases": int(len(per_case)),
        "sanity": {
            "max_abs_prob_discrepancy_vs_saved_pkl": max_disc,
            "max_abs_gate_mean_discrepancy_vs_saved_pkl": max_gate_disc,
            "passed": bool(max_disc < 1e-3),
        },
        "gate_per_case_mean": summarise(gates.mean(axis=1)),
        "gate_all_dims_pooled": summarise(gates.reshape(-1)),
        "gate_spread_across_32_dims": {
            "mean_within_case_std": float(gates.std(axis=1, ddof=0).mean()),
            "per_dim_mean_min": float(gates.mean(axis=0).min()),
            "per_dim_mean_max": float(gates.mean(axis=0).max()),
            "per_dim_mean": [float(v) for v in gates.mean(axis=0)],
        },
        "per_fold": per_fold,
        "auroc_case_level_pooled": auc,
        "auroc_drop": {
            "table_removed": auc["prob_intact"] - auc["prob_no_table"],
            "image_removed": auc["prob_intact"] - auc["prob_no_image"],
        },
        "_per_case_frame": per_case,
    }


def wsi_alone_case_probs() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    case_by_slide = dict(zip(manifest["slide_id"].astype(str), manifest["case_id"].astype(str)))
    rows = []
    for fold in range(N_FOLDS):
        saved = pickle.load(open(RESULTS / "er_wsi_alone_s1" / f"split_{fold}_results.pkl", "rb"))
        for slide, rec in saved.items():
            rows.append({
                "case_id": case_by_slide[str(slide)],
                "prob_wsi_alone": float(np.asarray(rec["prob"]).reshape(-1)[1]),
            })
    return pd.DataFrame(rows).groupby("case_id", as_index=False)["prob_wsi_alone"].mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cache", action="store_true", help="reuse an existing pooled WSI cache")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not (args.skip_cache and CACHE.is_file()):
        print(f"Building pooled WSI cache on {device} ...", flush=True)
        build_pooled_cache(device)

    slide_index, pooled = load_cache()
    wsi_alone = wsi_alone_case_probs()

    report = {
        "generated_by": "tools/diagnostics/gate_probe.py",
        "image_absent_definition": (
            "pooled WSI vector (mean over the CLAM_MB per-class attention-pooled features, "
            "512-dim) replaced by the mean of that same vector over the fold's TRAIN-split "
            "slides, computed with the same frozen fold-specific CLAM branch. The attention "
            "network is bypassed for the ablated pass, so no per-slide image information "
            "reaches the fusion head."
        ),
        "table_absent_definition": (
            "standardised tabular vector replaced by all-zeros, i.e. the fold's training-set "
            "mean after the fold's own fitted feature-selection + standardisation transform."
        ),
        "arms": {},
    }

    for arm, csv_path in ARMS.items():
        print(f"\n=== {arm} ===", flush=True)
        res = probe_arm(arm, csv_path, slide_index, pooled)
        per_case = res.pop("_per_case_frame")
        merged = per_case.merge(wsi_alone, on="case_id", how="inner")
        res["agreement_with_wsi_alone"] = {
            "n_matched_cases": int(len(merged)),
            "pearson_r": float(pearsonr(merged["prob_intact"], merged["prob_wsi_alone"])[0]),
            "pearson_p": float(pearsonr(merged["prob_intact"], merged["prob_wsi_alone"])[1]),
            "spearman_rho": float(spearmanr(merged["prob_intact"], merged["prob_wsi_alone"])[0]),
            "spearman_p": float(spearmanr(merged["prob_intact"], merged["prob_wsi_alone"])[1]),
        }
        res["auroc_case_level_pooled"]["wsi_alone_on_same_cases"] = float(
            roc_auc_score(merged["label"], merged["prob_wsi_alone"])
        )
        report["arms"][arm] = res
        print(json.dumps({k: v for k, v in res.items() if k != "gate_spread_across_32_dims"}, indent=2)[:2000])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gate_and_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"\nwrote {OUT_DIR / 'gate_and_ablation.json'}")
    print(f"wrote {OUT_DIR / 'gate_and_ablation.md'}")


def write_markdown(report: dict) -> None:
    lines = [
        "# Gate collapse and modality-ablation probe (ER fusion arms)",
        "",
        "Forward passes only, on the released per-fold checkpoints. No retraining.",
        "Evaluated on the 10 held-out test splits of `splits/tcga_brca_er_100`, which",
        "partition the cohort, so pooled case-level metrics use each case exactly once.",
        "",
        "The gate is the weight on the **image**: `fused = g * proj_wsi + (1-g) * proj_tab`,",
        "`g = sigmoid(Linear([proj_wsi ; proj_tab]))`, a 32-dim vector.",
        "",
        "## Ablation definitions",
        "",
        f"- **Table absent** — {report['table_absent_definition']}",
        f"- **Image absent** — {report['image_absent_definition']}",
        "",
    ]
    for arm, res in report["arms"].items():
        s, g, auc = res["sanity"], res["gate_per_case_mean"], res["auroc_case_level_pooled"]
        agree = res["agreement_with_wsi_alone"]
        lines += [
            f"## {arm}",
            "",
            f"- tabular input dim: {res['tabular_dim']}; "
            f"{res['n_test_slides']} test slides / {res['n_test_cases']} test cases",
            f"- sanity check vs saved `split_i_results.pkl`: max |Δprob| = "
            f"{s['max_abs_prob_discrepancy_vs_saved_pkl']:.3e} "
            f"({'PASS' if s['passed'] else 'FAIL'}); max |Δgate mean| = "
            f"{s['max_abs_gate_mean_discrepancy_vs_saved_pkl']:.3e}",
            "",
            "### Image gate (per-case mean over the 32 gate dimensions)",
            "",
            "| mean | std | min | p10 | p50 | p90 | max |",
            "|---|---|---|---|---|---|---|",
            f"| {g['mean']:.4f} | {g['std']:.4f} | {g['min']:.4f} | {g['p10']:.4f} | "
            f"{g['p50']:.4f} | {g['p90']:.4f} | {g['max']:.4f} |",
            "",
            f"Pooling all 32 dimensions of all cases: mean {res['gate_all_dims_pooled']['mean']:.4f}, "
            f"p10 {res['gate_all_dims_pooled']['p10']:.4f}, p90 {res['gate_all_dims_pooled']['p90']:.4f}, "
            f"range [{res['gate_all_dims_pooled']['min']:.4f}, {res['gate_all_dims_pooled']['max']:.4f}].",
            f"Mean within-case spread across the 32 dimensions (std): "
            f"{res['gate_spread_across_32_dims']['mean_within_case_std']:.4f}; "
            f"per-dimension means span "
            f"[{res['gate_spread_across_32_dims']['per_dim_mean_min']:.4f}, "
            f"{res['gate_spread_across_32_dims']['per_dim_mean_max']:.4f}].",
            "",
            "### Per-fold image gate",
            "",
            "| fold | n test slides | mean gate | std across cases | mean std across dims |",
            "|---|---|---|---|---|",
        ]
        for f in res["per_fold"]:
            lines.append(
                f"| {f['fold']} | {f['n_test_slides']} | {f['gate_mean']:.4f} | "
                f"{f['gate_std_across_cases']:.4f} | {f['gate_std_within_case_dims']:.4f} |"
            )
        lines += [
            "",
            "### Case-level pooled AUROC under inference-time ablation",
            "",
            "| condition | AUROC | drop vs intact |",
            "|---|---|---|",
            f"| intact | {auc['prob_intact']:.4f} | — |",
            f"| table absent | {auc['prob_no_table']:.4f} | {res['auroc_drop']['table_removed']:+.4f} |",
            f"| image absent | {auc['prob_no_image']:.4f} | {res['auroc_drop']['image_removed']:+.4f} |",
            f"| WSI-alone arm, same cases | {auc['wsi_alone_on_same_cases']:.4f} | — |",
            "",
            "### Agreement with the WSI-alone arm (case-level ER-positive probability)",
            "",
            f"- matched cases: {agree['n_matched_cases']}",
            f"- Pearson r = {agree['pearson_r']:.4f} (p = {agree['pearson_p']:.3e})",
            f"- Spearman rho = {agree['spearman_rho']:.4f} (p = {agree['spearman_p']:.3e})",
            "",
        ]

    rna, clin = report["arms"]["er_wsi_rna_gated"], report["arms"]["er_wsi_clinpath_gated"]
    lines += [
        "## Reading",
        "",
        f"The gate never collapses *numerically*: its mean sits at "
        f"{rna['gate_per_case_mean']['mean']:.3f} (RNA arm) and "
        f"{clin['gate_per_case_mean']['mean']:.3f} (clinicopath arm), both near the 0.5 "
        "midpoint, and the individual dimensions spread widely "
        f"(pooled range [{rna['gate_all_dims_pooled']['min']:.3f}, "
        f"{rna['gate_all_dims_pooled']['max']:.3f}] and "
        f"[{clin['gate_all_dims_pooled']['min']:.3f}, "
        f"{clin['gate_all_dims_pooled']['max']:.3f}]). The gate mean is therefore not "
        "diagnostic on its own: a mixing weight of one half says nothing about how much "
        "*discriminative* variance each projected branch carries. The ablations do.",
        "",
        f"**RNA arm — functionally an RNA classifier.** Deleting the table costs "
        f"{rna['auroc_drop']['table_removed']:+.4f} AUROC "
        f"({rna['auroc_case_level_pooled']['prob_intact']:.4f} -> "
        f"{rna['auroc_case_level_pooled']['prob_no_table']:.4f}); deleting the image costs "
        f"only {rna['auroc_drop']['image_removed']:+.4f} "
        f"({rna['auroc_case_level_pooled']['prob_intact']:.4f} -> "
        f"{rna['auroc_case_level_pooled']['prob_no_image']:.4f}). With the image gone the "
        f"model still beats the WSI-alone arm on the same cases "
        f"({rna['auroc_case_level_pooled']['prob_no_image']:.4f} vs "
        f"{rna['auroc_case_level_pooled']['wsi_alone_on_same_cases']:.4f}), while with the "
        "table gone it falls *below* it. The frozen image branch contributes roughly "
        f"{rna['auroc_drop']['image_removed']:.3f} AUROC of the reported gain.",
        "",
        f"**Clinicopath arm — functionally an image classifier.** Deleting the table costs "
        f"{clin['auroc_drop']['table_removed']:+.4f} AUROC, i.e. nothing measurable; "
        f"deleting the image costs {clin['auroc_drop']['image_removed']:+.4f} and drops the "
        f"model to {clin['auroc_case_level_pooled']['prob_no_image']:.4f}, near chance. Its "
        f"case-level predictions correlate r = {clin['agreement_with_wsi_alone']['pearson_r']:.3f} "
        "with the WSI-alone arm. The null result is mechanistic: the fusion head learned to "
        "route around the 24-dim table entirely.",
        "",
        "**Caveat.** Both ablations are off-manifold — neither branch ever saw a constant "
        "input during training — so the absolute ablated AUROCs are lower bounds on what a "
        "single-modality model retrained from scratch would achieve. The *relative* "
        "comparison between the two ablations within one arm is the load-bearing result.",
        "",
    ]
    (OUT_DIR / "gate_and_ablation.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
