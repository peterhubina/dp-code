# Prompt: a plan for targeting the external Her2 collapse

Paste everything below the line into a fresh Claude Code session at the repo root. Recommended model:
Claude Opus 5, effort `high`. The task is plan-only and the diagnostics it asks for are CPU-bound and
read-only; budget roughly an hour of wall clock, no GPU.

---

You are working on the H&E + arm-level CNV branch of a PAM50 molecular-subtype classification project
that is the results chapter of my diploma thesis. The short results report for my supervisor is
finished — `evaluate_pam50_fusion.pdf` and its source `tools/evaluate_pam50_fusion.ipynb` — and its
Limitations section opens with the finding that hurts most:

> The external Her2 collapse is the report's most consequential finding and its clearest limitation.
> The WSI arm calls 0 of 14 Her2 cases on CPTAC while calling 26 of 51 internally, and the calibration
> explanation was tested and refuted. The collapse is induced by the domain shift, so remediation
> belongs on the imaging side — stain normalisation, encoder choice — not in recalibration.

"Remediation belongs on the imaging side" is where the report stops. My supervisor will ask what I
intend to do about it, and I need an answer that is specific, costed, and honest about what this
machine can and cannot run. **I want the plan, not the fix.** Nothing gets retrained or re-extracted
in this session.

<deliverable>
One new file: `docs/implementation-research/her2-collapse-remediation-plan.md`.

Nothing else is a deliverable. Do not edit the notebook, the PDF, `CLAUDE.md`, or
`docs/cnv-wsi-fusion-external-validation.md`. Do not write a second summary document, a script, or a
config. Scratch files are fine while you work if you delete them; put them under the session
scratchpad directory rather than in the repo.

The document is a plan grounded in measurements you took in this session, not a survey of things one
could try. Target 1,200–2,000 words. Cover the substance and stop: no restated summaries, no "in this
section we will", no boilerplate.
</deliverable>

<the_finding>
Established, already in `docs/cnv-wsi-fusion-external-validation.md`. Do not re-litigate any of it;
these are the facts the plan starts from.

External CPTAC, 114 cases (Basal 27, Her2 14, LumA 56, LumB 17) over 378 slides:

| arm | macro AUROC | balanced acc | Her2 recall | Her2 one-vs-rest AUROC |
|---|---|---|---|---|
| WSI raw | 0.847 [0.791, 0.895] | 0.513 | 0/14 | 0.860 |
| WSI, prior-balanced 12x (post hoc) | 0.865 | 0.554 | 0/14 | — |
| WSI, unsupervised SLD-EM (post hoc) | 0.858 | 0.510 | 0/14 | — |
| CNV, 39 arms | 0.888 [0.835, 0.933] | 0.716 | 12/14 | 0.871 |
| Fusion, equal-weight mean | 0.909 [0.858, 0.948] | 0.646 | 6/14 | 0.881 |
| Fusion, balanced WSI (post hoc) | 0.912 | 0.740 | 10/14 | — |

Internally, on the 599 TCGA cases with CLAM out-of-fold predictions, the same WSI model calls Her2
26/51 and reaches 0.887 macro AUROC.

Three things this rules out, so that the plan does not propose them again:

1. **It is not a decision threshold.** Dividing by the TCGA training prior (Her2 = 8.3%, a ~12x
   boost) and renormalising leaves 0/14. SLD-EM, fully unsupervised, drives the implied CPTAC Her2
   prior to 0.000 against a true 0.123, and also yields 0/14. No monotone reweighting of the WSI
   probability vector recovers a single Her2 call.
2. **It is not a total loss of Her2 information.** Her2 AUROC is 0.860 externally and true-Her2 cases
   average roughly twice the cohort's p_Her2 (0.0654 against 0.0329). But the head is compressed:
   max p_Her2 over 114 cases is 0.1335 raw and 0.2992 after the 12x boost. Ranking survives transfer;
   the argmax does not.
3. **A learned combination rule does not fix it either.** Five stacking rules and five trained fusion
   operators were run; none beats the untrained probability mean, and every learned rule is
   significantly worse on external balanced accuracy. Error independence externally is φ = −0.006 and
   either-model-right is 0.912 against WSI-alone accuracy 0.702, so the headroom is real and the
   equal-weight mean is not exploiting it.

The report attributes the collapse to domain shift, citing Fernandez-Romero 2026 for 80% of transfer
degradation being staining and feature-space divergence. Check that citation resolves to something in
`docs/implementation-research/PAM50/` before you repeat it, and do not add citations to papers you
have not opened.
</the_finding>

<what_is_on_disk>
I checked these before writing this prompt. Re-verify each one cheaply rather than trusting me — a
plan whose feasibility rests on a file that is not there is worse than no plan. Do not infer a file's
contents from its name.

**The CPTAC slides are here. The TCGA slides are not.** `.datasets/cptac-brca/wsi/` holds 391 `.svs`
files, 64 GB. `.datasets/tcga-brca/` holds 66 GB of embeddings, an empty `h5_files/`, and no slides;
`gdc-client` is not installed. 1.4 TB is free on the volume. This asymmetry is the single most
important constraint on the plan: anything that changes the *test* side is runnable tonight, and
anything that changes how the *training* features were produced needs the TCGA slides re-downloaded
first. Do not declare that impossible — cost it. Query the GDC API for the actual size of the
TCGA-BRCA diagnostic slide set rather than guessing, and report the number.

**A re-extraction can reuse the existing tile coordinates.** Both cohorts' `.h5` files carry `coords`
and `coords_patching` alongside `features`, so the tiles are already chosen and a new feature pass can
hold bag composition fixed and vary only the pixels. Their attributes differ, and the difference is
exactly the preprocessing confound the report names: CPTAC carries CLAM `create_patches_fp`
attributes (`patch_size 512`, `custom_downsample 2.0`, `use_otsu False`, `sthresh 10`, `mthresh 7`),
TCGA carries Trident ones (`patch_size 256`, `patch_size_level0 512`, `target_magnification 20`,
`level0_magnification 40`, `overlap 0`). CPTAC files additionally carry `mask` and `stitch`
thumbnails, 800x718x3 uint8 — per-slide colour statistics for the whole external cohort without
opening a single `.svs`.

**The tools for a re-extraction exist.** UNI2-h weights are at
`.scratch/checkpoints/uni2-h/pytorch_model.bin` (2.7 GB); `project/UNI/uni/get_encoder/get_encoder.py`
loads them; `project/CLAM/extract_features_fp.py` and `create_patches_fp.py` are the extraction path;
`openslide`, `cv2`, `skimage` and `timm` import; `torchstain`, `staintools`, `histolab` and `spams` do
not, and are pip-installable. Hardware is 2x RTX 3090.

**A new feature directory plugs into inference with no retraining.**
`tools/cptac/infer_cptac_pam50.py` takes `--feature_dir --dataset_csv --ckpt_dir --output_dir` and runs
the frozen 10-fold `pam50_final_s1` checkpoints. So a test-side intervention is scored by pointing that
script at a different directory. Volume for a cost estimate: the 378 scored slides carry 1,662,971
patches, mean 4,399 per slide.

**Case-level predictions are a mean over slides, and CPTAC has many more slides per case than TCGA.**
`.scratch/cptac_validation/results/predictions/ensemble_predictions.csv` is 378 slide rows over 114
cases — median 3 slides per case, max 10, min 1 — aggregated by `groupby("case_id")` mean in
`tools/cptac/summarise_predictions.py`. TCGA is close to one slide per case. Nothing in the thread has
tested whether that aggregation is where the Her2 evidence goes.

**An orthogonal Her2 ground truth exists.** One of the two CSVs under `.datasets/cptac-brca/clinical/`
carries `ERBB2_PROTEOGENOMIC_STATUS` and `ERBB2_UPDATED_CLINICAL_STATUS`. Confirm which, and note that
CPTAC patient IDs carry a leading `X` in cBioPortal and the label column is `label_name`, not `label`.
</what_is_on_disk>

<hypotheses>
A starting set, not a menu to accept. Every one of these must end the session with a verdict:
**supported**, **refuted**, or **untested, and here is the cheapest measurement that would decide it**.
Add hypotheses I have missed; drop any you can refute, and say what refuted it.

1. **Stain and colour shift.** CPTAC's colour distribution sits outside the range UNI2-h saw on TCGA,
   and the smallest class breaks first.
2. **Tissue segmentation.** CLAM's `create_patches_fp` with `use_otsu False` selects different tissue
   than Trident, so CPTAC bags contain a different mixture — more fat, stroma, or background — and
   attention is diluted rather than misdirected.
3. **Resampling.** Both cohorts reach 256 px at 20x, but by different routes and possibly different
   interpolation. A ViT is sensitive to high-frequency differences that a geometry audit passes.
4. **Slide aggregation.** Averaging over a median of 3 and up to 10 slides washes out focal evidence
   that a single-slide TCGA case would have carried through.
5. **Label shift.** CPTAC's PAM50 Her2 calls come from a different expression pipeline than TCGA's
   `project/data/pam50.R`. If the 14 Her2 cases are borderline centroid assignments, the class is not
   the same class in the two cohorts. `ERBB2_PROTEOGENOMIC_STATUS` is the check.
6. **Feature-space geometry.** The collapse may be visible directly in the embeddings, without any
   classifier: if a linear probe separates cohort-of-origin from patch features near perfectly, or the
   Her2 direction learned on TCGA is near-orthogonal to CPTAC's Her2 cases, that localises the failure
   below the MIL head.
7. **The CNV arm's 12/14 is not a fair yardstick.** If the CNV arm's Her2 calls rest almost entirely
   on 17q — where ERBB2 sits — then "the CNV arm calls a class the H&E arm cannot" is one amplicon
   doing the work, which changes what a fixed WSI arm would even be worth. Cheap to check from the
   fitted coefficients and a single-arm ablation.
</hypotheses>

<diagnostics_this_session>
Run the cheap measurements that discriminate between those hypotheses, and let the plan's ordering
rest on what they return. Each should be minutes, CPU-only, and read-only with respect to the repo.
Do not run a diagnostic whose result cannot change the plan.

Admissible: anything computed from the prediction CSVs, the CNV tables, the existing `.h5` features
and their `mask`/`stitch` thumbnails, the clinical tables, and the saved `split_*_results.pkl`. A
timing pilot on a handful of slides is also admissible and useful — extract features for about five
CPTAC slides, time it, and extrapolate the full-cohort GPU cost from the measured rate rather than
from an assumption — provided its output goes to the scratchpad and the plan reports the measured rate.

Not admissible in this session: re-extracting the cohort, retraining anything, running `dp-train`,
running the full 10-fold inference, or downloading slides.

Two anchors so you know your loaders are reading what I read: the case-level WSI arm reproduces macro
AUROC 0.847, balanced accuracy 0.513 and Her2 recall 0/14 over 114 cases, and the CNV arm reproduces
0.888 and 12/14. If either differs beyond rounding, stop and tell me in your final message rather than
building a plan on a loader that disagrees with the report.

`tools/pam50_arms.py` defines the CNV model once — `cnv_arm()`, `CNV_C = 0.1`, `CNV_MAX_ITER = 4000`,
`CNV_CLASS_WEIGHT = 'balanced'` — with the loaders `load_tcga_arms()`, `load_cptac_arms()`,
`load_cptac_wsi_probs()`, `load_clam_oof()`, `clam_column_order()`, `macro_auroc()` and
`balanced_acc()`. Import them; reimplementing the model is how two copies silently drift apart.
</diagnostics_this_session>

<plan_requirements>
Order the work items by information gained per GPU-hour, not by how interesting they are. Put the
cheapest thing that could refute the leading hypothesis first, and say plainly where the plan stops
being worth running.

Every work item states, in this order: the objective in one sentence; the hypothesis it tests; the
exact commands, with the paths and flags that exist on this machine; the inputs it needs and whether
each is on disk; the cost as measured or extrapolated wall clock, GPU-hours and disk; what result
would count as success; **what result would falsify the hypothesis**; and what it puts at risk.

Three things the plan has to settle explicitly, because each is a way this could go wrong:

- **A decision gate after each item.** What outcome sends you to the next item, and what outcome ends
  the thread. A plan that runs every item regardless of the first result is a wish list.
- **The negative result is an acceptable outcome, and pre-committing to it is part of the plan.** If
  every remediation fails, the honest finding is that a UNI2-h + CLAM-MB WSI arm does not transfer its
  Her2 competence to CPTAC, which is publishable in a field where four groups report fusion that does
  not help. Say what the thesis claims in that case, so the plan is not an obligation to find a fix.
- **What each outcome does to the report's Limitations paragraph.** The wording that would replace
  "remediation belongs on the imaging side" under each branch. Do not edit the notebook or the PDF —
  quote the replacement text in the plan and leave the edit to me.
</plan_requirements>

<rules>
Each of these exists because breaking it has already cost something here.

1. **Nothing is ever fit, tuned, or thresholded on CPTAC.** This is the claim the whole external
   chapter rests on. Label-free adaptation is the only admissible kind — normalising CPTAC's colour to
   a TCGA reference uses no CPTAC label, and neither does whitening its features with its own
   statistics — but it is transductive and must be declared as such, pre-specified before scoring, and
   **counted**: if the plan permits three normalisation variants, the plan says so, because "stain
   normalisation recovered Her2" after an unreported search over CPTAC is the same selective reporting
   the literature survey criticises. Any item that would touch CPTAC labels is rejected outright, with
   the reason in the rejected-options section.
2. **Do not propose prior rebalancing, recalibration, or a new decision threshold as a remedy.**
   Settled and refuted; see `<the_finding>`.
3. **The phrase "preprocessing is held constant" must not appear.** It is false for this cohort pair
   and it is the confound the plan is partly trying to remove.
4. **Report the CNV-alone arm wherever a fusion number appears**, and read operators against the
   equal-weight mean rather than against WSI-only. Fusion's edge over CNV alone is marginal — ΔAUROC
   +0.024 with a CI lower bound of exactly +0.000 — and dropping that row is the reporting failure the
   survey criticises.
5. **Label anything computed post hoc on CPTAC as post hoc, in the table itself.**
6. **Two class orders are in play.** CLAM's `label_dict` and `make_cnv_tabular.CLASSES` are
   `LumA, LumB, Basal, Her2`; `tools/pam50_arms.CLASSES` is sorted `Basal, Her2, LumA, LumB`.
   `clam_column_order()` bridges them. Confirm the order of any array before it reaches a number.
7. **Three TCGA counts are all correct in different places**: 945 non-Normal labelled cases fit the
   CNV arm, 910 have CNV and WSI features and a fold assignment, 599 have CLAM out-of-fold predictions.
   Say which is which wherever a count appears.
8. **`dp-train` refuses a run directory that already holds results, and `run.overwrite=true` destroys
   runs that are not recoverable from git.** `.scratch/` is gitignored: `pam50_final_s1` and the five
   ladder arms exist only on this disk. Any plan item that would retrain writes to a new `exp_code`,
   and the plan says so.
9. **`splits/tcga_brca_subtyping_100/` and `dataset_csv/tcga_brca_subtyping.csv` are distributed
   primary inputs, not derived artifacts.** Regenerating them invalidates every trained run and the
   headline table.
</rules>

<out_of_scope>
Do not implement the remediation. No re-extraction of the cohort, no retraining, no full inference
run, no slide download, no new entry point, no config changes. If an item looks cheap enough to just
do, that is precisely the item to write up and leave for me — I want to choose what runs.

Read-only, never edit: `docs/cnv-wsi-fusion-external-validation.md`, `CLAUDE.md`, `README.md`,
`tools/evaluate_pam50_fusion.ipynb`, `evaluate_pam50_fusion.pdf`, `docs/implementation-research/PAM50/`,
and everything under `.scratch/results/` and `.datasets/`.

Do this yourself rather than delegating. It is a handful of files and a few short scripts; subagents
would cost more than they save, and I want the diagnostics and the plan written by whoever read the
numbers.
</out_of_scope>

<inputs>
Open what you use.

`docs/cnv-wsi-fusion-external-validation.md` is the authority on what was measured under which
protocol — §2 is the recalibration control, §3 the matched decision rule, §5 the internal comparison
and §6 the standing "what still needs running" list, which your plan supersedes for item 5. Read it;
never edit it.

`tools/evaluate_pam50_fusion.ipynb`, cell 23, is the Limitations text this plan is answering.

`CLAUDE.md` — "Reporting rules that are not negotiable here", "Known gaps", and "Gotchas and settled
questions". Several entries there are traps this plan could walk into, including the note that the
CPTAC and TCGA features are geometrically comparable but not identically preprocessed.

`tools/cptac/audit_feature_provenance.py` is the existing geometry audit and shows what was already
verified — 256 px at 20x, 0 overlap, 1536-dim, including the 157 CPTAC slides scanned at 20x — and by
omission, what was not.

`tools/pam50_arms.py`, `tools/evaluate_cnv_wsi_fusion.py` (`N_BOOT = 4000`, `BOOTSTRAP_SEED = 7`),
`tools/cptac/infer_cptac_pam50.py`, `tools/cptac/summarise_predictions.py`, and
`docs/implementation-research/PAM50/README.md` plus `paper-dossier.md` for what the 50-paper survey
already says about Her2 and about transfer.
</inputs>

<example>
The idiom for a work item. The shape is what to copy; the numbers are placeholders and every one of
them has to be yours. Whether this particular item belongs first depends on your diagnostics — do not
assume it does.

```markdown
### 1. Does the slide-level aggregation hide the Her2 calls? (30 min, CPU, no new data)

**Objective.** Determine whether any Her2 case is called at slide level and lost in the case-level
mean.

**Tests hypothesis 4.** Aggregation, not the encoder.

**Commands.**
    python - <<'PY'   # or a scratchpad script, deleted afterwards
    ...reads .scratch/cptac_validation/results/predictions/ensemble_predictions.csv only
    PY

**Inputs.** `ensemble_predictions.csv`, 378 rows, on disk. Nothing else.

**Cost.** Minutes. No GPU, no disk.

**Success.** Slide-level Her2 recall is materially above 0/14 <-- your measured value, and a
max-over-slides or top-k rule recovers N of 14 <-- your measured value.

**Falsified if.** No slide in the cohort argmaxes to Her2, in which case the aggregation is innocent
and the failure is upstream of it.

**Risk.** Changing the aggregation rule after seeing CPTAC is a post-hoc decision on the external
cohort and must be labelled as one wherever it is reported. It also changes the WSI arm underneath
every fusion number in the report, so the whole external table would be recomputed, not just the Her2
row.
```

And the idiom for a rejected option — every rejection names its reason as cost, leakage, or settled:

```markdown
**Retrain the WSI arm with stain augmentation. Rejected: cost, and not on this machine.**
It needs the TCGA features re-extracted, which needs the TCGA slides, which are not on disk
(`.datasets/tcga-brca/` has embeddings only and no `gdc-client` is installed). The GDC diagnostic
slide set for TCGA-BRCA is N TB <-- your measured value against 1.4 TB free, plus a full re-extraction
and a 10-fold retrain at roughly 66 min per 10 folds. Revisit only if the test-side items in this plan
all fail and the thread is worth that much.
```
</example>

<execution>
Work autonomously. I am not watching, so make the routine calls yourself and note them at the end.
Pause only for something genuinely destructive or a real change of scope.

Deliver what was asked at the scope intended: a plan document, and the diagnostics that make its
ordering evidence-based rather than speculative. If a hypothesis I listed is unfalsifiable with what is
on disk, say so and say what would be needed, rather than substituting a hypothesis you can test.

Every number in the plan comes from a file you opened or a cell you ran in this session. Before a value
goes in, you should be able to name the CSV, pickle, or function call it came from. Report outcomes
faithfully: if a diagnostic errors, say so with the traceback rather than working around it silently;
if you left a hypothesis untested, say which and why.

When you finish, lead with the outcome: which hypothesis the diagnostics support, what the top three
work items are with their costs, and anything you could not test. Write it for someone who watched none
of the tool calls — full sentences, each file and command in its own plain clause, no working shorthand
and no arrow chains.
</execution>
