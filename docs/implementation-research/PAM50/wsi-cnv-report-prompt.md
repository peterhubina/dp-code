# Prompt: build the WSI + arm-level CNV evaluation report

Paste everything below the line into a fresh Claude Code session at the repo root. Recommended
effort: `high` (or `xhigh` if you want the SOTA-placement prose worked harder).

---

You are producing the evaluation report for the H&E + arm-level CNV branch of a master's thesis on
PAM50 molecular-subtype classification. It goes to a thesis committee and later to a journal
reviewer, so every number in it has to be traceable to something on disk or to a script that
regenerates it. A plausible-looking number that nobody can reproduce is worse than a missing one,
because it survives my review and fails theirs.

There is already a report in exactly the register I want, for the sibling RNA thread:
`docs/rna_unimodal_evaluation.pdf`, built from `.scratch/rna_unimodal_report/`. Read its LaTeX
source before writing anything. The new report is its counterpart for the CNV thread and should be
recognisably the same document family — same preamble, same table idiom, same tone.

<deliverable>
A build directory `.scratch/wsi_cnv_report/` containing:

- `wsi_cnv_evaluation.tex` — the report source
- `figures/` — every figure as both `.pdf` and `.png`
- `regenerate_figures.py` — regenerates every figure from CSVs, no on-figure titles (LaTeX captions
  carry the description), following `.scratch/rna_unimodal_report/regenerate_figures.py`
- the CSVs each figure and table was built from, so a reader can recompute any cell
- `wsi_cnv_evaluation.pdf` — the compiled report

Then copy the compiled PDF to `docs/wsi_cnv_evaluation.pdf` and the figure images to
`docs/wsi_cnv_evaluation_images/`, mirroring how the RNA report is published.
</deliverable>

<why_this_shape>
The RNA report is the template because the committee has already seen it and because its structure
survives review: problem definition and cohort first, then pipeline, then protocol, then results
from aggregate down to per-class, then the SOTA comparison, then limitations. Matching it means the
two chapters read as one body of work rather than two write-ups.
</why_this_shape>

<reference_report>
`.scratch/rna_unimodal_report/rna_unimodal_evaluation.tex` (464 lines). Reuse its preamble verbatim,
including these macros, which give the arrows and bolding in every results table:

```latex
\newcommand{\metricup}{\ensuremath{\uparrow}}
\newcommand{\metricdown}{\ensuremath{\downarrow}}
\newcommand{\best}[1]{\textbf{#1}}
```

Its section order is: Problem Definition (with a class table and a cohort table) → Pipeline (one
subsection per stage) → Experimental Setup (hyperparameter optimisation, evaluation protocol) →
Results (aggregate, fold-level stability, per-class) → Comparison to the unimodal branch →
Fusion evaluation → Comparison to SOTA (one single-modality table, one multimodal table) →
Discussion, Limitations, and Next Steps.

Follow that order, adapted to this thread's actual content. Its prose style is plain declarative
sentences in full paragraphs with tables carrying the numbers — not bulleted fragments. Match it.
</reference_report>

<inputs>
Every path below was confirmed to exist. Open what you use; do not assume a file's contents from its
name.

**The frozen results document — read it in full first, and never edit it.**
`docs/cnv-wsi-fusion-external-validation.md` (325 lines). This is the authority for what was
measured and under which protocol. The report is a presentation of these results, not a new
analysis of them.

**Project rules.** `CLAUDE.md`, especially the sections "What this project is doing right now",
"Reporting rules that are not negotiable here", "Known gaps" and "Gotchas and settled questions".
Read it; several of its entries are traps that this report could walk into.

**Internal TCGA training runs** under `.scratch/results/`:
- `pam50_final_s1/` — the WSI-only baseline. 10 folds, `split_{0..9}_results.pkl`,
  `s_{0..9}_checkpoint.pt`, `experiment_pam50_final.txt`.
- `pam50_wsi_cnv_{concat,gated,cross_attention,film_attention,coattn}_s1/` — the five trained fusion
  operators. Each has `summary.csv` (columns `folds,test_auc,val_auc,test_acc,val_acc`),
  `split_{0..9}_results.pkl`, `fold_{0..9}_history.csv` (per-epoch training curves),
  `experiment_*.txt` and `s_*_tabular_transform.json`.

**External CPTAC inference:**
`.scratch/cptac_validation/results/predictions/ensemble_predictions.csv` — 378 slides → 114 cases,
columns `slide_id,case_id,true_label,…,p_LumA,p_LumB,p_Basal,p_Her2`. Plus per-fold
`fold_{0..9}_predictions.csv`. **This is the WSI arm only.**

**CNV features:** `.scratch/cnv-tabular/{TCGA,CPTAC}_BRCA_CNV_arm_4class_clam.csv` (910 and 114 data
rows) and `.datasets/cnv/{tcga,cptac}_brca_cna_arm.csv` (981×39 and 114×39). The CNV model is
`StandardScaler → LogisticRegression(max_iter=4000, C=0.1, class_weight='balanced')`, defined once in
`tools/pam50_arms.py`; it is refit in-process by the analysis scripts rather than stored as a
checkpoint.

**The analysis scripts that produce the numbers.** All CPU, all seconds-to-a-minute, all read-only
with respect to the trained models. Run them; do not hand-transcribe from the results document when
a script will print the value:
```
dp-analysis cnv_wsi_fusion                        # external TCGA -> CPTAC, the headline table
dp-analysis cnv_wsi_fusion analysis.internal=true # adds the internal TCGA head-to-head
dp-analysis stack_wsi_cnv                         # five learned combination rules vs the mean
dp-analysis cnv_controls                          # per-class AUROC tables, error independence, controls
dp-analysis compare_fusion_ladder                 # the five operators vs WSI-only, CNV-only and the mean
```
Each writes a self-describing run directory under `.scratch/analysis/<action>/<timestamp>/`
containing `output.txt`, `config.resolved.yaml`, `run_metadata.json` and a results JSON. Cite the run
directory in the report's reproduction notes so a reader can find the exact run.

**SOTA material — already researched, use it rather than searching again.**
- `docs/implementation-research/PAM50/sota-comparison-cnv-fusion.md` — the comparison document. Its
  §3 table and §4 subsections are the source for every comparator number and every comparability
  flag. Its §6 lists claims that must be softened, with replacement wording.
- `docs/implementation-research/PAM50/sota-comparison-new-rows.json` — 20 verified papers in the
  survey schema (pids P51–P70).
- `docs/implementation-research/PAM50/README.md` and `survey-data.json` — the original 50-paper
  survey (pids P01–P50).
</inputs>

<what_does_not_exist>
This section is the most important one in the prompt. These are the holes a report like this falls
into, and each one has to be represented as a hole rather than filled.

1. **The five trained fusion operators have no external CPTAC evaluation, and cannot easily get
   one.** `project/CLAM/evaluate_multimodal.py` accepts only `auto|concat|gated|residual|cross_attention`,
   and under `auto` a `film_attention` checkpoint raises and a `coattn` checkpoint is misidentified
   as `cross_attention` and then dies in `load_state_dict(strict=True)`. `dp-evaluate` also defaults
   to the TCGA test split, not CPTAC. So the external results table has exactly **four** model rows —
   WSI-only, CNV-only, the equal-weight probability mean, and the post-hoc prior-balanced mean — and
   the operator ladder is an **internal-only** result. Say so explicitly in the report where a reader
   would otherwise expect an external ladder column.

2. **Run-to-run variance has never been measured.** Seeds and folds are fixed, but no tolerance has
   been established, and bitwise reproducibility is neither claimed nor achievable because
   `cross_attention` and `coattn` use `nn.MultiheadAttention` with no deterministic-algorithm flags
   set. Do not quote a ± that came from anywhere other than an actual multi-seed or multi-fold
   computation, and label which it is.

3. **Calibration is not measured for any arm in this thread.** The RNA report has an ECE column; this
   one does not, unless you compute it yourself from saved predictions. If you compute it, say so and
   say how. Otherwise write `--`, as the RNA report does for metrics it lacks.

4. **`residual` is not in the ladder** and cannot be trained — it needs a matched tabular-only
   checkpoint via `--pretrained_rna_ckpt` and no supported trainer produces one. The ladder is five
   operators, not six.

If a number you want does not exist, the report says it does not exist. An honest gap is a finding.
</what_does_not_exist>

<report_content>
Cover, in the reference report's section order:

**Problem definition and cohort.** Four-class PAM50 (LumA / LumB / Basal / Her2), Normal-like
dropped. State both cohorts and be precise about which count applies where, because three different
TCGA counts are all correct in different places: 981 labelled cases, of which **945** are non-Normal
and form the CNV arm's fitting set; **910** have CNV *and* WSI features *and* a fold assignment and
form the CLAM tabular table; **599** have CLAM out-of-fold predictions and are the set every internal
head-to-head is computed on. CPTAC is 114 cases / 378 slides (Basal 27, Her2 14, LumA 56, LumB 17).
Getting these confused is the single easiest way to make the report wrong.

**Pipeline.** One subsection each for: the WSI branch (UNI2-h 1536-dim patch features → CLAM-MB,
`--model_size big`); the CNV branch (39 chromosome arms, each the median gene-level log2 over that
arm, acrocentric p-arms excluded by construction); and the fusion strategies (the untrained
equal-weight probability mean, and the five trained operators with their `--tabular_hidden_dim 64
--tabular_top_n_features 0 --fusion_hidden_dim 32` sizing and the warm start from `pam50_final_s1`).
State *why* arm-level and not focal: shallow WGS resolves arm/segment scale, so the assay is
clinically reachable, whereas a model over 19,755 focal GISTIC calls would not transfer to an sWGS
setting. State *why* CNV and not RNA: PAM50 labels are computed from the expression matrix itself
(`project/data/pam50.R`), so an RNA branch leaks the target by construction.

**Experimental setup.** The split protocol (`splits/tcga_brca_subtyping_100`, k=10, seed 1), and the
fact — stated plainly, not buried — that CLAM's 10 splits are drawn **independently rather than
partitioned**, so 599 of 910 cases land in at least one test fold and 242 in two to five. That makes
"WSI alone" a small ensemble, flattering it by roughly +0.01 AUROC. Also state that the ladder is a
*near*-baseline rather than matched comparison: `pam50_final_s1` used sweep-selected
`lr 1.008e-4 / reg 2.446e-6 / bag_weight 0.553 / inst_loss svm` with instance clustering on, while
the five ladder arms use rounded `lr 1e-4 / reg 2.5e-6`, CLAM's default `bag_weight`, no instance
loss and `--no_inst_cluster`.

**Results**, in this order:
- Internal TCGA aggregate over the 599 shared cases: WSI-only, CNV-only, the probability mean, and
  the five operators.
- Fold-level stability from each arm's `summary.csv`, reported as mean ± SD across the ten folds,
  with a sentence noting that pooled out-of-fold metrics differ from fold means because they are
  computed after concatenating held-out predictions.
- Per-class internal AUROC and per-class recall.
- External CPTAC: the four available rows, per-class AUROC, and per-class recall at argmax.
- Error independence, with the regime named on every value (see reporting rules).
- The operator ladder against the untrained mean, including the model-count control (five operators
  ensembled, best pair, worst pair) and the error-correlation mechanism.

**Comparison to SOTA.** Two tables, mirroring the RNA report's single-modality and multimodal split.
Rules in the next block.

**Discussion, limitations and next steps.** The HER2-enriched external collapse (0/14 raw, 0/14 after
a 12× prior boost, 0/14 under unsupervised SLD-EM, against 26/51 in-domain) and the fact that the
calibration explanation was tested and refuted. The open warm-start confound on the ladder mechanism
claim. Power: external Her2 n=14 and LumB n=17, so per-class external estimates are indicative. The
preprocessing confound: CPTAC `.h5` files carry CLAM `create_patches_fp` attributes while TCGA's
carry Trident ones, so tissue segmentation differs and the phrase "preprocessing is held constant"
must not appear.
</report_content>

<reporting_rules>
Non-negotiable, and each exists because breaking it has already caused a problem.

1. **Report the CNV-alone arm every time a fusion number appears.** Fusion's edge over CNV alone is
   marginal — ΔAUROC +0.024 with a CI lower bound of exactly +0.000, and no significant
   balanced-accuracy difference. Omitting it reproduces the selective reporting the literature survey
   criticises.

2. **The equal-weight probability mean is the baseline that operators must clear, not the WSI-only
   model.** Every ladder table is read against the mean.

3. **Label anything computed post hoc on CPTAC as post hoc.** The prior-balanced WSI variant was run
   after the raw fusion underperformed. In particular, do **not** repeat the defect that
   `docs/implementation-research/PAM50/sota-comparison-cnv-fusion.md` §6 item 5 identifies: §9 of the
   results document quotes ΔAUROC +0.066 and Δbalanced-accuracy +0.226 "against raw WSI", but those
   are arithmetically *prior-balanced fusion minus raw WSI* — a post-hoc-corrected model against a
   pre-registered one, a contrast that appears in no table. The pre-specified contrast is
   **+0.063 [+0.023, +0.106]** AUROC and **+0.134** balanced accuracy. Report the pre-specified
   contrast as the headline and the mixed one, if at all, explicitly labelled.

4. **Say which protocol produced each control.** The internal CNV headline 0.866 ± 0.003 is 5-fold ×
   10 reseeds; the aneuploidy-burden number, the C sweep and the site holdout are single 5-fold runs
   at seed 0.

5. **Nominate one internal CNV figure and say why.** Three exist under three protocols and all are
   defensible: 0.862 (StratifiedKFold(10, seed 0) on the 599 cases), 0.872 (CNV refit per CLAM fold so
   both arms are out-of-fold on the same fold) and 0.866 ± 0.003 (the published headline on the full
   945-case set). The 0.010 spread is larger than several contrasts the report calls significant, so
   leaving the reader to pick is not acceptable.

6. **Every error-correlation φ carries its regime.** External CPTAC WSI-vs-CNV is −0.006; internal is
   **0.269** under `StratifiedKFold(10, seed 0)` (this is the value `dp-analysis cnv_controls` checks
   against as published, recorded at `dpcode/cli/analysis.py:336`) and **0.193** under the
   per-CLAM-fold refit used in §8 of the results document; among the five jointly trained operators it
   is 0.656. The mechanism claim is a contrast between two of these, so name which two.

7. **The two class orders.** CLAM's `label_dict` and `make_cnv_tabular.CLASSES` are
   `LumA, LumB, Basal, Her2`; `tools/pam50_arms.CLASSES` is sorted `Basal, Her2, LumA, LumB`. Confirm
   which order any array you touch is in before you put it in a table.
</reporting_rules>

<sota_comparison>
Build both SOTA tables **only** from
`docs/implementation-research/PAM50/sota-comparison-cnv-fusion.md` and
`sota-comparison-new-rows.json`. Do not search for new papers and do not add a comparator that is not
in those files.

Every comparator row must carry, as columns or as a compact flag string: class count (4 / 5 / 2 /
NR), prediction unit (patch / slide / case), label basis (RNA-derived PAM50 / IHC surrogate / mixed /
NR), evaluation regime (internal CV / external cohort / pooled into training), and refereed versus
unrefereed preprint. Those five flags are what make the comparison honest; without them the numbers
are uncalibrated by an unknown amount.

Specific things that are easy to get wrong and are already settled in the comparison document:

- **Amer et al. 2025's like-for-like arm is macro-AUC 0.8835 / 0.8836, not 0.8604 / 0.8616.** The
  latter pair belongs to their Image + Graph row (two WSI representations, no CNV). Their CNV-alone
  arm is 0.8284 and their four-modality fusion is 0.9153, both internal 10-fold on TCGA n=977. Their
  clinical branch is their strongest single modality (accuracy 70.43% / macro-AUC 0.8522) but the EHR
  vector's contents are unspecified, so if it encodes ER/PR/HER2 the comparison is circular — state
  that wherever their fusion number is quoted.
- **Never difference an internal number against an external one.** Amer's 0.8284 is internal 10-fold;
  our 0.888 is a held-out cohort. They sit on opposite sides of a transfer boundary. The defensible
  sentence compares their internal CNV arm to our internal 0.862–0.872 and states that our external
  0.888 has no counterpart in their work.
- **Never convert, round, average or harmonise a metric across papers.** A macro-F1 is not an AUROC,
  a slide-level number is not a case-level number, and an accuracy is neither. Where a comparator
  reports a metric we do not, leave the cell `--`.
- Mark unrefereed preprints as such in the table. Six of the twenty new rows are preprints.

Follow the RNA report's caption convention of stating up front that cohort sizes, splits, feature
extractors and reporting protocols differ, so absolute numbers are not a direct ranking.
</sota_comparison>

<figures>
Generate figures only where the data supports one. Candidates, in rough order of value:
per-fold test AUC across the ladder arms with the probability mean as a reference line; the external
CPTAC confusion matrix for the equal-weight mean; one-vs-rest ROC curves on the external cohort;
per-class AUROC comparison across WSI-only, CNV-only and fusion; and a bar chart of the operator
ladder against the untrained mean.

Use the reference script's conventions: serif font, no on-figure titles, `savefig.dpi = 200`,
`bbox_inches="tight"`, both `.pdf` and `.png`, and the same class colour mapping
(`LumA #1f77ff`, `LumB #d62728`, `Basal #2ca02c`, `Her2 #ff7f0e`). matplotlib, pandas and
scikit-learn are installed and available.
</figures>

<build>
No LaTeX toolchain is currently installed — `pdflatex`, `xelatex`, `lualatex`, `tectonic` and
`latexmk` are all absent. TeX Live is installable from the configured apt sources; a dry run of
`apt-get install -y --no-install-recommends texlive-latex-base` resolves cleanly, and the reference
preamble also needs `booktabs` and `xcolor`, which live in `texlive-latex-recommended` and
`texlive-latex-extra`. PyPI is reachable if you prefer another route.

Establish a working build before writing report content: compile a two-line document, confirm a PDF
comes out, then write. Discovering at the end that nothing compiles wastes the whole run. If no route
to a PDF works, deliver the `.tex` and the figures, state plainly in your final message that the PDF
could not be built and what blocked it, and do not silently substitute a Markdown file for the
deliverable.
</build>

<integrity>
Every number in the report must come from a file you opened or a command you ran in this session.
Before you write a value into a table, you should be able to name the run directory, CSV, pickle or
script output it came from. If you cannot, do not write it.

Do not copy numbers from `docs/cnv-wsi-fusion-external-validation.md` into the report without
regenerating them, except where no producing script exists — and where that is the case, say so in
the reproduction notes rather than presenting the value as freshly computed. Two figures in the
results document genuinely have no producing command: the `mean` and `mean on true-Her2 cases`
columns of its §2, and the cross-cohort platform correlation (per-arm r = 0.960) of its §4. Both were
computed interactively.

If a regenerated value disagrees with the published one, that is a finding, not something to
reconcile silently. Report the discrepancy in your final message and put the recomputed value in the
report with a footnote, leaving the results document untouched.

Before reporting progress or stating that a section is done, check the claim against an actual tool
result from this session rather than against your expectation of what should have happened.
</integrity>

<boundaries>
- Read-only, never edit: `docs/cnv-wsi-fusion-external-validation.md`, `README.md`, `CLAUDE.md`,
  `docs/implementation-research/PAM50/*`.
- Do not retrain, re-tune or re-run any experiment. `dp-train` is not part of this task. Running the
  `dp-analysis` actions is expected and correct — they only recompute metrics from saved predictions.
- `dp-train` refuses a run directory that already holds `summary.csv` or an `s_*_checkpoint.pt`, and
  `run.overwrite=true` would destroy results that are not recoverable from git. Do not go near it.
- New files belong in `.scratch/wsi_cnv_report/`, plus the two published copies under `docs/`. Clean
  up any scratch or scratch-test files you create along the way.
</boundaries>

<execution>
Work autonomously. I am not watching, so do not ask whether to proceed on anything that follows from
this brief; make the routine calls yourself and note them in the final message. Pause only for
something genuinely destructive or a real change of scope.

Delegate to subagents where the work is genuinely parallel and sizeable — for example, one agent
regenerating the internal-results CSVs from the ladder pickles while another extracts the SOTA
comparator rows and their flags from the comparison document. Do not delegate work you can finish in
a handful of tool calls, and keep the total count modest.

Establish a check on your own work as you go rather than at the end: after each results table is
written, re-derive one cell of it from the underlying CSV or pickle and confirm it matches.

Match the report's length to its substance. The RNA report is about 460 lines of LaTeX; this one has
more arms to cover and a real SOTA section, so somewhat longer is fine, but do not pad with filler
sections, restated summaries or boilerplate. Prose in full paragraphs, tables for the numbers.

When you finish, lead with the outcome: what the report contains, what could not be included and
why, and any discrepancy you found between a regenerated number and its published counterpart. Write
that summary for someone who did not watch any of the tool calls — spell out identifiers, drop the
working shorthand, and give each file or command its own plain clause.
</execution>

<example>
The table idiom to match, taken verbatim from the reference report. Note the `\metricup` arrows in
the header, `\best{}` on the winning cell per column, `\scriptsize` plus `\resizebox` for wide
tables, and `booktabs` rules with no vertical lines.

```latex
\begin{table}[H]
\centering
\caption{Aggregate held-out test results on the matched 986-slide cohort.}
\label{tab:aggregate_results}
\scriptsize
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
Method & n & Macro AUC~\metricup & Acc.~\metricup & Bal. Acc.~\metricup & Macro F1~\metricup & Weighted F1~\metricup & ECE~\metricdown \\
\midrule
WSI only & 986 & 0.8879 & 0.7181 & -- & 0.65 & 0.7179 & -- \\
Matched RNA only & 986 & 0.9785 & 0.8783 & 0.8809 & 0.8644 & 0.8799 & -- \\
\textbf{Selective ensemble} & 986 & \best{0.9796} & \best{0.8996} & \best{0.8856} & \best{0.8804} & \best{0.8999} & 0.0256 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

And the SOTA table idiom, with the `\emph{Reference}` / `\emph{Our work}` block split that keeps
published numbers visually separate from ours:

```latex
\midrule
\multicolumn{6}{l}{\emph{Reference}} \\
Hezil et al.~\cite{hezil_repo} & RNA-seq only & 1022 patients & \best{0.91} & \best{0.9} & \best{0.91} \\
\midrule
\multicolumn{6}{l}{\emph{Our work}} \\
Matched RNA only & RNA MLP & 986 slides & 0.8783 & 0.8644 & 0.8799 \\
```
</example>
