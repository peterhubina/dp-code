#!/bin/bash
# Multi-seed repeat of the claim-bearing ER arms.
#
# WHY: every Part C difference is under 0.01 AUROC, so single-seed variance is the biggest
# live objection to both the primary null (FiLM does not beat gated) and the positive
# graceful-degradation finding (co-attention harms on clinicopathology, FiLM does not).
# Seed 1 is already done; this adds seeds 2 and 3 for a mean +/- sd over three seeds.
#
# WHAT VARIES: only --seed, which drives weight init, dropout and the weighted sampler
# (main.py seed_torch at line 50, called per fold). The SPLITS ARE FIXED CSV FILES, so the
# folds are byte-identical across seeds and every comparison stays paired. The mechanism
# configuration does NOT change: film_rank stays 64, selected once on seed-1 validation
# folds. Re-selecting per seed would be re-tuning, not replication.
#
# WSI-ALONE IS RE-TRAINED PER SEED, and each seed's fusion arms load and freeze that seed's
# own checkpoints (er_wsi_alone_s<seed>). This tests whether the conclusion survives a full
# pipeline re-initialisation rather than only a fusion-head re-initialisation.
#
# Usage (from repo root):
#   bash tools/train_er_multiseed.sh            # seeds 2 and 3, ~9 h total
#   bash tools/train_er_multiseed.sh 2          # just seed 2, ~4.5 h
#   SEEDS="2 3 4" bash tools/train_er_multiseed.sh
#
# Roughly 33 min per 10-fold arm, 8 arms per seed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-${*:-2 3}}"

# Arms carrying a Part C claim. Deliberately EXCLUDED, and this is a stated limitation:
#   *_gatedcap  -- capacity controls; seed 1 already showed capacity explains nothing
#                  (gated dim-96 0.9447 vs FiLM 0.9462, p = 0.739), and they carry no claim.
#   clinpath_delta -- was numerically identical to clinpath_film at seed 1 (p = 0.974).
# Both therefore remain single-seed results.
CHAPTER1_ARMS="wsi rna clinpath"                                   # tools/train_er_ablation.sh
CHAPTER2_ARMS="film_rna film_clinpath delta_rna coattn_rna coattn_clinpath"

for seed in ${SEEDS}; do
    echo "################################################################"
    echo "### SEED ${seed}  --  8 arms, WSI-alone first (it writes the frozen checkpoints)"
    echo "################################################################"

    for arm in ${CHAPTER1_ARMS}; do
        echo ">>> seed ${seed}: chapter-1 arm '${arm}'"
        SEED="${seed}" bash tools/train_er_ablation.sh "${arm}"
    done

    for arm in ${CHAPTER2_ARMS}; do
        echo ">>> seed ${seed}: chapter-2 arm '${arm}'"
        SEED="${seed}" bash tools/train_er_novel_fusion.sh "${arm}"
    done

    echo ">>> seed ${seed} complete."
done

echo
echo "All seeds done. Analyse with:"
echo "  python tools/evaluate_er_ablation.py --seeds 1 ${SEEDS} --unit case"
