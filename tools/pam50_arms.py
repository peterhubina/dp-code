"""Shared plumbing for the two PAM50 prediction arms and the metrics used to compare them.

``evaluate_cnv_wsi_fusion.py`` and ``stack_wsi_cnv.py`` report numbers that have to agree with
each other, so the pieces they both depend on live here rather than being defined twice. The CNV
arm in particular: it is one logistic regression with one regularisation setting, and if that
setting drifts between the two scripts their tables silently stop describing the same model.

Nothing here reads a slide. The WSI arm is consumed as probabilities already written to disk by
the CLAM training and external-validation runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent

#: Sorted, and every probability matrix in both scripts is in this column order.
CLASSES = np.array(["Basal", "Her2", "LumA", "LumB"])

TCGA_ARMS = REPO / ".datasets/cnv/tcga_brca_cna_arm.csv"
CPTAC_ARMS = REPO / ".datasets/cnv/cptac_brca_cna_arm.csv"
TCGA_LABELS = REPO / "tools/data/tcga_brca_pam50_labels.csv"
CPTAC_WSI_PROBS = REPO / ".scratch/cptac_validation/results/predictions/ensemble_predictions.csv"
CLAM_OOF = REPO / ".scratch/results/pam50_final_s1"
CLAM_SPLITS = REPO / "project/CLAM/splits/tcga_brca_subtyping_100"


def case_of(slide_id: str) -> str:
    """TCGA slide barcode -> the 3-field case id the rest of the repo keys on."""
    return "-".join(slide_id.split("-")[:3])


def renormalise(p: np.ndarray) -> np.ndarray:
    return p / p.sum(1, keepdims=True)


def macro_auroc(y, probs) -> float:
    return roc_auc_score(y, probs, multi_class="ovr", average="macro")


def balanced_acc(y, probs) -> float:
    return balanced_accuracy_score(y, CLASSES[probs.argmax(1)])


def cnv_arm():
    """The CNV arm, defined once. Balanced because Her2 is 8% of TCGA."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced"))


def load_tcga_arms():
    """Arm-level CNV and PAM50 labels for TCGA, Normal-like dropped to match CPTAC."""
    x = pd.read_csv(TCGA_ARMS, index_col=0)
    y = pd.read_csv(TCGA_LABELS).drop_duplicates("case_id").set_index("case_id")["label"]
    shared = x.index.intersection(y.index)
    x, y = x.loc[shared], y.loc[shared]
    labelled = y != "Normal"
    return x[labelled], y[labelled]


def load_cptac_arms():
    return pd.read_csv(CPTAC_ARMS, index_col=0)


def load_cptac_wsi_probs():
    """TCGA-trained CLAM applied to CPTAC, mean-pooled from slides to cases."""
    slides = pd.read_csv(CPTAC_WSI_PROBS)
    return slides.groupby("case_id").agg(
        {**{f"p_{c}": "mean" for c in CLASSES}, "true_name": "first"}), len(slides)


def load_clam_oof(with_folds: bool = False) -> pd.DataFrame:
    """CLAM out-of-fold probabilities per case, optionally tagged with the fold that produced them.

    The 10 splits are drawn independently rather than partitioned, so a case can be held out by
    several of them; ``fold`` is therefore the *first* fold that tested a case, and ``p0..p3`` are
    averaged over however many models saw it. See the ``stack_wsi_cnv`` docstring for what that
    costs.
    """
    import pickle

    rows = []
    for path in sorted(CLAM_OOF.glob("split_*_results.pkl")):
        fold = int(path.stem.split("_")[1])
        for slide, pred in pickle.load(open(path, "rb")).items():
            rows.append({"case_id": case_of(slide), "fold": fold, "label": int(pred["label"]),
                         **{f"p{i}": p for i, p in enumerate(np.asarray(pred["prob"]).ravel())}})
    how = {**{f"p{i}": "mean" for i in range(4)}, "label": "first"}
    if with_folds:
        how["fold"] = "first"
    return pd.DataFrame(rows).groupby("case_id").agg(how)


def clam_column_order(oof: pd.DataFrame, labels: pd.Series) -> list[str]:
    """Recover CLAM's integer->class mapping so its probabilities can be put in CLASSES order.

    ``label`` is CLAM's stored ground truth rather than a prediction, so the mode is exact. It is
    still asserted, because a non-bijective result would make the caller's ``.index()`` lookup
    reuse one column and silently drop a class.
    """
    order = [labels[oof["label"] == i].mode()[0] for i in range(len(CLASSES))]
    assert len(set(order)) == len(CLASSES), f"class-order recovery is not a permutation: {order}"
    return order


def fold_train_cases(fold: int) -> set:
    """Case ids in a CLAM fold's train column."""
    split = pd.read_csv(CLAM_SPLITS / f"splits_{fold}.csv")
    return {case_of(s) for s in split["train"].dropna()}


def bootstrap_indices(y, n_boot: int, seed: int) -> list:
    """Resample positions, rejecting any draw that loses a class so AUROC stays defined."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_boot:
        draw = rng.integers(0, len(y), len(y))
        if len(np.unique(np.asarray(y)[draw])) == len(CLASSES):
            out.append(draw)
    return out


def delta_ci(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple:
    """Paired difference over shared resamples: (mean, lo, hi, 'sig'|'ns')."""
    d = scores_a - scores_b
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi, "sig" if (lo > 0 or hi < 0) else "ns"
