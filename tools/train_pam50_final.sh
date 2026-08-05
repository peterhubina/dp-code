#!/bin/bash
# Full 10-fold cross-validation for PAM50 molecular subtype classification.
# Config frozen from sweep candidate "baseline-best" (selected by val_auc).
#
# THIS IS NOW A SHIM. The configuration lives in
# `dpcode/conf/experiment/pam50_wsi_final.yaml`, and the equivalent command is:
#
#     dp-train experiment=pam50_wsi_final
#
# The hyperparameters are unchanged — the rendered argv parses to exactly the
# namespace this script used to pass, field for field, including the
# full-precision `lr`, `reg` and `bag_weight` and the fact that instance
# clustering is ON here and off in every fusion experiment.
#
# What you gain by calling `dp-train` directly: the run directory becomes
# self-describing (`config.resolved.yaml`, `run_metadata.json`, `clam_argv.json`,
# `metrics.json`, `.hydra/`), a completed run cannot be overwritten by accident,
# and every path comes from `dpcode/conf/paths/default.yaml` rather than from
# `../../` relative to `project/CLAM`.
#
# Usage (unchanged, and now runnable from any directory):
#   bash tools/train_pam50_final.sh
#   bash tools/train_pam50_final.sh clam.seed=2      # extra Hydra overrides pass through
#   bash tools/train_pam50_final.sh --dry-run

set -euo pipefail

DP_TRAIN=(dp-train)
command -v dp-train >/dev/null 2>&1 || DP_TRAIN=(python -m dpcode.cli.train)

echo "tools/train_pam50_final.sh now runs:" >&2
echo "    ${DP_TRAIN[*]} experiment=pam50_wsi_final $*" >&2
echo >&2

exec "${DP_TRAIN[@]}" experiment=pam50_wsi_final "$@"
