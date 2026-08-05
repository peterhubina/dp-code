#!/bin/bash
# Multi-seed repeat of the claim-bearing ER arms.
#
# WHY: every Part C difference is under 0.01 AUROC, so single-seed variance is the biggest
# live objection to both the primary null (FiLM does not beat gated) and the positive
# graceful-degradation finding (co-attention harms on clinicopathology, FiLM does not).
# Seed 1 is already done; this adds seeds 2 and 3 for a mean +/- sd over three seeds.
#
# WHAT VARIES: only --seed (`clam.seed`), which drives weight init, dropout and the weighted
# sampler (main.py seed_torch at line 50, called per fold). The SPLITS ARE FIXED CSV FILES, so the
# folds are byte-identical across seeds and every comparison stays paired. The mechanism
# configuration does NOT change: film_rank stays 64, selected once on seed-1 validation folds.
# Re-selecting per seed would be re-tuning, not replication.
#
# WSI-ALONE IS RE-TRAINED PER SEED, and each seed's fusion arms load and freeze that seed's own
# checkpoints (er_wsi_alone_s<seed>). The experiment configs interpolate `clam.seed` into that path,
# so this happens without any argument juggling here.
#
# THIS IS NOW A SHIM, and it is an ORCHESTRATOR of two other shims -- it calls no entry point
# itself. It is also PARTIALLY BLOCKED: four of its five chapter-2 arms are not ported to dp-train
# (see tools/train_er_novel_fusion.sh, and addendum A13 of the reproducibility refactor). Rather
# than fail four hours into a run, it refuses up front and names them.
#
# Nothing here has ever been run: only _s1 directories exist under .scratch/results/er, so the
# "mean +/- sd over three seeds" this script exists to produce does not exist on disk.
#
# Usage (from repo root):
#   bash tools/train_er_multiseed.sh            # seeds 2 and 3, ~9 h total
#   bash tools/train_er_multiseed.sh 2          # just seed 2, ~4.5 h
#   SEEDS="2 3 4" bash tools/train_er_multiseed.sh
#
# Roughly 33 min per 10-fold arm.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-${*:-2 3}}"

# Arms carrying a Part C claim. Deliberately EXCLUDED, and this is a stated limitation:
#   *_gatedcap  -- capacity controls; seed 1 already showed capacity explains nothing
#                  (gated dim-96 0.9447 vs FiLM 0.9462, p = 0.739), and they carry no claim.
#   delta_clinpath -- was numerically identical to clinpath_film at seed 1 (p = 0.974).
# Both therefore remain single-seed results.
CHAPTER1_ARMS="wsi rna clinpath"                                   # tools/train_er_ablation.sh
CHAPTER2_ARMS="film_rna film_clinpath delta_rna coattn_rna coattn_clinpath"

# What the two shims can actually run today.
PORTED_CHAPTER1="wsi rna clinpath"
PORTED_CHAPTER2="film_rna"

blocked=""
for arm in ${CHAPTER1_ARMS}; do
    [[ " ${PORTED_CHAPTER1} " == *" ${arm} "* ]] || blocked+=" ${arm}"
done
for arm in ${CHAPTER2_ARMS}; do
    [[ " ${PORTED_CHAPTER2} " == *" ${arm} "* ]] || blocked+=" ${arm}"
done

if [[ -n "${blocked}" ]]; then
    cat >&2 <<EOF
This multi-seed matrix cannot run as written: the following arms have no dp-train
experiment config and would fail partway through --${blocked}

Only each ER wrapper's default invocation was ported (reproducibility refactor,
addendum A13). What is available:
  chapter 1  wsi rna clinpath      (tools/train_er_ablation.sh, all three)
  chapter 2  film_rna              (tools/train_er_novel_fusion.sh)

To repeat the ported arms at another seed, which is what this script is for:
  for seed in ${SEEDS}; do
      for arm in ${PORTED_CHAPTER1}; do SEED=\$seed bash tools/train_er_ablation.sh \$arm; done
      for arm in ${PORTED_CHAPTER2}; do SEED=\$seed bash tools/train_er_novel_fusion.sh \$arm; done
  done

tools/train_er_novel_fusion.sh lists what each unported arm was and what it would take to add one.
EOF
    exit 2
fi

for seed in ${SEEDS}; do
    echo "################################################################"
    echo "### SEED ${seed}  --  WSI-alone first (it writes the frozen checkpoints)"
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
