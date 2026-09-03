# Independent recomputation of the PAM50 WSI+CNV headline numbers

Session date 2026-09-01. Repo `/workspace/dp-code`, branch `repro-hydra-refactor`,
git sha `1dc5c9f2fd671010b9d0b2ceabb3dd70024dcb2b` (working tree dirty; every dirty path is
documentation or notebook, none of them a script this evaluation ran).

**Result up front: everything reproduced. Zero discrepancies > 0.001** between (a) the fresh
`dp-analysis` runs made in this session, (b) the prior CSVs in `.scratch/wsi_cnv_report/`, and
(c) the standing numbers quoted in `CLAUDE.md`. The full comparison, 108 paired checks, is the
output of `check_discrepancies.py` in this directory; the list is at the bottom of this file.

Nothing was trained, downloaded or overwritten. Only CPU `dp-analysis` actions ran — six
invocations of four actions — each into a fresh self-describing dir under
`.scratch/analysis/<action>/<timestamp>/`.

---

## 1. Entry point: the console scripts are not on PATH

`dp-analysis` (and every other `dp-*` script) is **not installed as an executable** in this
environment. `dp-code 0.1.0` is present as a legacy `dp_code.egg-info` develop install at
`/workspace/dp-code`, so `import dpcode` works and the console-script entry points are declared
in the metadata, but no wrapper exists in `/opt/venv/bin`. `pip install -e '.[dev]'` (the fix
`CLAUDE.md` prescribes) was **not** run — that would modify the environment.

Every command below was therefore invoked as the identical module entry point:

```
/opt/venv/bin/python -m dpcode.cli.analysis <action> [overrides]
```

`dpcode/cli/analysis.py` ends in `if __name__ == "__main__": sys.exit(main())`, so this is the
same `main()` the `dp-analysis` console script would call. Each run's `run_metadata.json`
records `command_line: dp-analysis <action> ...` — the runs are indistinguishable from
console-script invocations.

**Flag correction to `CLAUDE.md`:** `dp-analysis` has **no `--dry-run` flag**. `dp-analysis
--help` shows `--show-config` (compose and print, run nothing) and `--no-run-dir`. The dry-run
step of this task was done with `--show-config`, which exited 0 for all four actions.

## 2. Commands run, and the run directories they created

| # | command | run directory | exit | wall |
|---|---|---|---|---|
| 0 | `dp-analysis list` | (no run dir) | 0 | — |
| 0 | `dp-analysis {cnv_wsi_fusion,stack_wsi_cnv,compare_fusion_ladder,cnv_controls} --show-config` | (no run dir) | 0 | — |
| 1 | `dp-analysis cnv_wsi_fusion` | `/workspace/dp-code/.scratch/analysis/cnv_wsi_fusion/2026-09-01_18-58-28` | 0 | 41.9 s |
| 2 | `dp-analysis cnv_wsi_fusion analysis.internal=true` | `/workspace/dp-code/.scratch/analysis/cnv_wsi_fusion/2026-09-01_18-59-16` | 0 | 69.5 s |
| 3 | `dp-analysis stack_wsi_cnv` | `/workspace/dp-code/.scratch/analysis/stack_wsi_cnv/2026-09-01_19-00-36` | 0 | 43.3 s |
| 4 | `dp-analysis compare_fusion_ladder` | `/workspace/dp-code/.scratch/analysis/compare_fusion_ladder/2026-09-01_19-01-21` | 0 | 52.5 s |
| 5 | `dp-analysis cnv_controls` | `/workspace/dp-code/.scratch/analysis/cnv_controls/2026-09-01_19-02-20` | 0 | 2.5 s |
| 6 | `dp-analysis cnv_controls analysis.burden_definition=frac_altered` | `/workspace/dp-code/.scratch/analysis/cnv_controls/2026-09-01_19-04-48` | 0 | 2.8 s |

Run 6 was extra: it checks the second reading of "aneuploidy burden" that `CLAUDE.md` quotes
(0.673). No action refused for a missing input; `make_cnv_tabular` was deliberately not run
(it writes into `.scratch/cnv-tabular`, which the ladder was trained against).

Each dir holds `output.txt`, `config.resolved.yaml`, `run_metadata.json`; `cnv_controls` also
writes `controls.json` with full-precision values.

Each command internally dispatches the underlying script, recorded on line 1 of `output.txt`:

```
python tools/evaluate_cnv_wsi_fusion.py --n-boot 4000 --bootstrap-seed 7 --cv-folds 10 --cv-seed 0
python tools/evaluate_cnv_wsi_fusion.py --internal --n-boot 4000 --bootstrap-seed 7 --cv-folds 10 --cv-seed 0
python tools/stack_wsi_cnv.py --n-boot 2000 --bootstrap-seed 11 --stacker-C 1.0 --stacker-max-iter 4000 --nm-xatol 0.0001 --nm-fatol 1e-06 --nm-maxiter 2000 --clip-floor 1e-09
/opt/venv/bin/python /workspace/dp-code/tools/compare_fusion_ladder.py --n-boot 2000
```

## 3. Bootstrap seeds and N behind each number (file:line)

The three bootstrap seeds are **deliberately different** and are not unified.

| quantity | N_BOOT | seed | defined at |
|---|---|---|---|
| external CPTAC table, its AUROC/balAcc CIs and all its pairwise deltas | 4000 | 7 | `tools/evaluate_cnv_wsi_fusion.py:57` (`N_BOOT = 4000`), `:58` (`BOOTSTRAP_SEED = 7`); used at `:77-78`, `:149`, `:169` |
| internal §5 head-to-head (`--internal`) | 4000 | 7 | same constants; the CNV arm's CV is `CV_FOLDS = 10` `:59`, `CV_SEED = 0` `:60`, applied at `:164` |
| stacking rules, internal + external (§7) | 2000 | 11 | `tools/stack_wsi_cnv.py:73` (`N_BOOT = 2000`), `:74` (`BOOTSTRAP_SEED = 11`); used at `:210` and `:239` |
| fusion-operator ladder, all 8 pooled arms + model-count control | 2000 | 13 | `tools/compare_fusion_ladder.py:97` — `bootstrap_indices(y.values, args.n_boot, seed=13)`; the seed is hard-coded inside the script, only `--n-boot` is a flag |
| prior report CSVs in `.scratch/wsi_cnv_report/` | 4000 / 7 (external), 2000 / 13 (internal ladder) | | `.scratch/wsi_cnv_report/build_csvs.py:46` (`EXT_N_BOOT, EXT_SEED = 4000, 7`), `:48` (`LADDER_N_BOOT, LADDER_SEED = 2000, 13`) |

Resampling scheme: `tools/pam50_arms.py:163-171` — `np.random.default_rng(seed)`, positions
resampled with replacement, **draws that lose a class are rejected** so macro AUROC stays
defined. CI = 2.5/97.5 percentiles of the paired difference, verdict `sig` iff the interval
excludes 0 (`tools/pam50_arms.py:174-178`).

The CNV arm itself, defined once: `tools/pam50_arms.py:90-99` —
`make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.1, class_weight='balanced'))`
(`CNV_C = 0.1` at `:90`, `CNV_MAX_ITER = 4000` at `:91`, `CNV_CLASS_WEIGHT = "balanced"` at `:92`).
Class order `CLASSES = ["Basal","Her2","LumA","LumB"]` at `tools/pam50_arms.py:46`; CLAM's own
order on disk is `['LumA','LumB','Basal','Her2']` (printed by both internal runs) and is bridged
by `clam_column_order()`. All tables here are in the sorted `pam50_arms` order.

## 4. The two internal CNV protocols — they are NOT interchangeable

This is the single most important thing for the LaTeX table. The internal 599-case set is scored
under two CNV cross-validation regimes, and `CLAUDE.md`'s "CNV 0.862–0.872, mean 0.922–0.926" is
exactly the span between them:

| regime | CNV alone | equal-weight mean | phi(WSI, CNV) | produced by |
|---|---|---|---|---|
| **CNV refit per CLAM fold** (report's nominated regime) | **0.8721** | **0.9259** | **0.193** | `compare_fusion_ladder`, `stack_wsi_cnv`, `internal_ladder_pooled.csv` |
| CNV via `StratifiedKFold(10, shuffle, seed 0)` | 0.8624 | 0.922 | 0.269 | `cnv_wsi_fusion analysis.internal=true`, `cnv_controls` |

`results_consolidated.csv` uses the **per-CLAM-fold** regime for the baseline and every ladder
contrast (that is the only regime in which the five trained operators can be compared to the
mean on one bootstrap), and carries the StratifiedKFold CNV and its own mean as two clearly
labelled extra rows. Deltas are never mixed across regimes.

Reporting-rule 4 consequence: never write "internal φ = 0.269" without naming
StratifiedKFold(10, seed 0), and never write "0.193" without naming the per-CLAM-fold refit.

## 5. Error-correlation phi, with sources

| regime | phi | fresh source | prior CSV |
|---|---|---|---|
| internal TCGA, CNV refit per CLAM fold | **0.193** (full precision 0.1925222699476296) | `compare_fusion_ladder/.../output.txt:38` | `error_correlation.csv:2` |
| internal TCGA, CNV `StratifiedKFold(10, seed 0)` — the *published* value | **+0.269** (0.2691667570881939) | `cnv_controls/.../output.txt:18`, `controls.json` `internal_error_independence.phi` | `error_correlation.csv:3` |
| among the 5 jointly trained operators (10 pairs) | **0.656** (0.6557280302040964; min 0.582, max 0.706) | `compare_fusion_ladder/.../output.txt:37` | `error_correlation.csv:4` |
| external CPTAC, raw WSI vs CNV | **−0.006** (−0.0058823529411764705) | `cnv_controls/.../output.txt:14`, `controls.json` `external_error_independence.phi` | `external_error_correlation.csv:2` |

Contingency behind the external −0.006 (`cnv_controls/.../output.txt:15-17`): both right 56,
WSI only 24, CNV only 24, both wrong 10; WSI acc 0.702, CNV acc 0.702, either-right 0.912.
Internal (`:19-21`): both right 337, WSI only 103, CNV only 77, both wrong 82; WSI acc 0.735,
CNV acc 0.691, either-right 0.863.

## 6. Aneuploidy burden — the exact printed lines (reporting rule, non-negotiable)

From `/workspace/dp-code/.scratch/analysis/cnv_controls/2026-09-01_19-02-20/output.txt:32-38`:

```
=== 5. Aneuploidy burden alone (1 feature) — report this beside the 39-arm model ===
  CLAUDE.md makes this non-negotiable: at 0.685 it is high enough that Basal ~0.97
  reads as genome instability unless the arm *pattern* is shown to add to it.
  definition: mean_abs_log2
  5-fold, seed 0      0.6854   published 0.685  match  <- the published protocol
  5-fold x 10 seeds  0.6893 +- 0.0033
  39 arms - burden = +0.1764 macro AUROC on the 10-seed protocol, so the arm *pattern* carries signal beyond total instability
```

And under the alternative definition
(`.../cnv_controls/2026-09-01_19-04-48/output.txt`, `analysis.burden_definition=frac_altered`):

```
  definition: frac_altered (threshold 0.2)
  5-fold, seed 0      0.6733   published 0.685  DIFFERS by -0.0120  <- the published protocol
  5-fold x 10 seeds  0.6784 +- 0.0051
```

The `DIFFERS` banner is the script comparing `frac_altered` against the *`mean_abs_log2`*
published value; 0.6733 is itself the number `CLAUDE.md` quotes as 0.673 for this definition, so
this is a definition switch and **not** a failure to reproduce.

## 7. Cohort counts (all verified fresh against `cohort_counts.csv`)

- External CPTAC: **114 cases / 378 slides** — LumA 56, Basal 27, LumB 17, Her2 14
  (`cnv_wsi_fusion/.../output.txt:4-5`; `cohort_counts.csv:6`).
- Internal pooled OOF: **599 cases** — LumA 318, LumB 124, Basal 106, Her2 51
  (`compare_fusion_ladder/.../output.txt:3`, `cohort_counts.csv:5`). This is 599 of 910, because
  CLAM's 10 splits are drawn independently rather than partitioned.
- CNV fitting set (5-fold reseed protocol and the burden control): **945** TCGA non-Normal cases
  × 39 arms — LumA 499, LumB 197, Basal 171, Her2 78 (`cnv_controls/.../output.txt:1`;
  `cohort_counts.csv:4`).
- `stack_wsi_cnv/.../output.txt:4`: the per-CLAM-fold CNV refit trains on 726 cases per fold,
  "no eval case appears in its own fold's train split".

## 8. What could NOT be recomputed fresh, and the on-disk source used instead

Three quantities the LaTeX table needs are **not printed by any `dp-analysis` action**. They
come from `.scratch/wsi_cnv_report/*.csv`, produced by `build_csvs.py` on 2026-08-06. That
script was **not re-run** (it overwrites the CSVs in place, which this task forbids). Its point
estimates were nonetheless re-derived from the fresh runs and agree to display precision, and
its bootstrap constants are the same ones the fresh runs used (`build_csvs.py:46,48`).

1. **Balanced-accuracy CIs, external** — the fresh `cnv_wsi_fusion` prints `balAcc` with no
   interval. Taken from `external_aggregate.csv` (N=4000, seed 7). `stack_wsi_cnv` *does* print
   external balAcc CIs, but at N=2000 / seed 11 — a different bootstrap; see §9.
2. **Balanced-accuracy CIs, internal** — `compare_fusion_ladder` prints `balAcc` only. Taken
   from `internal_ladder_pooled.csv` (N=2000, seed 13).
3. **Full-precision external per-class AUROC** — `cnv_controls` prints 3 dp. Taken from
   `external_per_class.csv`; all 20 cells agree with the fresh 3 dp values (§B of the check).

Additionally:

4. **Per-class recall / Her2 recall for the five trained fusion operators does not exist
   anywhere on disk.** No script computes it, so `her2_recall` is blank for those five rows.
   Internal Her2 recall is available only for WSI only (26/51), CNV only (25/51 per-CLAM-fold,
   26/51 under StratifiedKFold) and the probability mean (30/51 per-CLAM-fold, 32/51 under
   StratifiedKFold) — `internal_per_class.csv:2-4` and `cnv_wsi_fusion/.../output.txt:39-41`.
5. **Balanced-accuracy CIs for the two protocol-variant rows and the SLD-EM row** are not
   produced by anything; those cells are blank and the point estimates are 3 dp.
6. `film_attention` and `coattn` checkpoints still cannot be evaluated by `dp-evaluate`
   (`evaluate_multimodal.py:70`), which is why the whole ladder is scored from per-fold
   `split_*_results.pkl` by `compare_fusion_ladder`, not by an evaluator. Unchanged, expected.

## 9. Same number, two bootstraps — a protocol difference, not a discrepancy

`stack_wsi_cnv` (N=2000, seed 11) and `cnv_wsi_fusion` (N=4000, seed 7) score the *same* three
external arms. The **point estimates are identical** (0.847 / 0.888 / 0.909 and balAcc
0.513 / 0.716 / 0.646); only the intervals move:

| arm | AUROC CI, seed 7 / N=4000 | AUROC CI, seed 11 / N=2000 | largest bound shift |
|---|---|---|---|
| WSI alone | [0.793, 0.897] | [0.794, 0.898] | 0.001 |
| CNV alone | [0.833, 0.935] | [0.840, 0.934] | **0.007** |
| equal-weight mean | [0.861, 0.949] | [0.866, 0.949] | **0.005** |

Same for the deltas vs the mean: WSI alone is −0.0628 [−0.1036, −0.0236] at seed 7 and
−0.0632 [−0.1049, −0.0221] at seed 11; CNV alone is −0.0211 [−0.0475, +0.0034] at seed 7 and
−0.0211 [−0.0480, +0.0027] at seed 11. Verdicts are identical in every case
(WSI sig, CNV ns). **`results_consolidated.csv` and `contrasts.csv` use the seed-7 / N=4000
bootstrap throughout for the external cohort**, because that is the one behind the published
headline table; the seed-11 numbers exist only in `stack_wsi_cnv/.../output.txt:24-38`. Quote
one or the other, never a mix.

## 10. Column semantics in `results_consolidated.csv`

- `delta_auroc_vs_mean` / `delta_ci_lo` / `delta_ci_hi`: **arm minus the equal-weight probability
  mean**, so a negative value means the arm is worse than the untrained baseline. This matches
  the sign printed by `compare_fusion_ladder` and `stack_wsi_cnv`. Note that
  `external_contrasts.csv` on disk stores the *opposite* sign (`model_b − model_a`); the values
  were negated and the CI bounds swapped when copied into `contrasts.csv`. Verified against the
  printed lines of `cnv_wsi_fusion/.../output.txt:22-34`.
- `contrasts.csv` uses `d = model_a − model_b` for every row, both cohorts.
- `trained`: true only for a **jointly trained multimodal fusion operator** (the five ladder
  arms). The unimodal arms and every probability-space combination rule (equal-weight mean,
  prior-balanced mean) are `false` — they are untrained *combinations*, though the underlying
  WSI and CNV models are of course themselves trained.
- `post_hoc`: true for anything computed on CPTAC after seeing CPTAC — the prior-balanced WSI
  arm, the prior-balanced fusion, and the SLD-EM arm. Reporting rule 3.
- `bal_acc_ci_lo/hi` blank means no script produces that interval (§8).
- Values are full precision where a CSV carries them and 3 dp where only the fresh `output.txt`
  does; the `protocol` field says which rows are 3 dp.

## 11. Reporting rules, as they apply to this table

1. **CNV alone is present in both cohorts and must be quoted whenever fusion is.** Externally the
   prior-balanced fusion beats CNV alone by ΔAUROC **+0.0240, CI [+0.0005, +0.0499], `sig` — a
   lower bound that prints as +0.000** (`external_contrasts.csv:10`; the fresh run prints the
   mirrored `CNV (39 arms) − Fusion balanced dAUROC −0.024 [−0.050,−0.000]` at
   `cnv_wsi_fusion/.../output.txt:33`). The *equal-weight* fusion beats CNV alone by only
   **+0.0211 [−0.0034, +0.0475], `ns`** (`:32`, `external_contrasts.csv:9`).
2. **The equal-weight mean is the baseline row in both cohorts**, not the WSI-only model.
3. Two external rows and the SLD-EM row are marked `post_hoc=true`.
4. Every row carries its protocol string, including which CNV CV regime and which bootstrap.

## 12. Discrepancy list

**None.** 108 paired comparisons at tolerance 0.001 (`check_discrepancies.py`, section-by-section
output reproduced when the script is re-run):

- **A** — 20 checks. Fresh `cnv_wsi_fusion` external table vs `external_aggregate.csv`
  (macro AUROC, both CI bounds, balAcc for all 5 arms). Max |diff| **0.000499**, all attributable
  to the fresh run printing 3 dp.
- **B** — 20 checks. Fresh `cnv_controls` external per-class AUROC vs `external_per_class.csv`
  (5 arms × 4 classes). Max |diff| **0.000470**. The script's own banner agrees:
  `vs the published table: all 12 cells match` (`cnv_controls/.../output.txt:11`).
- **C** — 32 checks. Fresh `compare_fusion_ladder` vs `internal_ladder_pooled.csv`
  (8 arms × macro AUROC + 2 CI bounds + balAcc). Max |diff| **0.000403**.
- **D** — 15 checks. Fresh ladder deltas vs the mean vs the same CSV. Max |diff| **0.000048**.
- **E** — 2 checks. Both internal CNV protocols vs `internal_cnv_protocols.csv`. Max |diff|
  **0.000426**.
- **F** — 4 checks. All four phi values vs `error_correlation.csv` /
  `external_error_correlation.csv`. Max |diff| **0.000478**.
- **G** — 15 checks. Fresh values vs the standing numbers in `CLAUDE.md` ("What this project is"
  and "Reporting rules"). Max |diff| **0.000427** (aneuploidy burden 0.68543 vs the quoted
  0.685). Every standing number holds: external 0.847 / 0.888 / 0.909 / 0.912, internal WSI
  0.887, CNV 0.862–0.872, mean 0.9259, ladder span 0.8818–0.8992, CNV headline 0.8657 ± 0.0034
  vs "0.866 ± 0.003", phi 0.656 / 0.269 / −0.006.
- **H** — cohort counts, exact integer match on all three sets.

`cnv_controls` additionally self-checks against the published document and printed `match` on
every one of its own assertions: the 12 external per-class cells, both phi values, both
accuracies, the 5-fold × 10-seed headline, the burden control, all four rows of the
regularisation sweep (0.879 / 0.870 / 0.860 / 0.856 at C = 0.01 / 0.1 / 1 / 10) and the
leave-one-site-out mean (0.8779 ± 0.0346 over 13 sites vs the published 0.878 ± 0.035).

The one *apparent* mismatch it prints — `DIFFERS by -0.0120` — is run 6's deliberate
`frac_altered` burden definition being compared against the `mean_abs_log2` published value; see
§6.

## 13. Files written by this evaluation

All under
`/tmp/claude-0/-workspace-dp-code/9063f4f7-e1aa-4e0e-a0ca-2e9f04a95f2b/scratchpad/eval/`:
`results_consolidated.csv`, `results_consolidated.json`, `contrasts.csv`,
`external_per_class.csv`, `eval_notes.md`, plus the two scripts that produced and checked them
(`build_consolidated.py`, `check_discrepancies.py`). Nothing was written inside the repository
except the six `dp-analysis` run directories listed in §2.
