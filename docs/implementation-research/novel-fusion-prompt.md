# Novel-fusion arm — ready-to-paste orchestration prompt

**How to run it.** Paste the block below as the first user turn. Recommended: orchestrator at
`xhigh` effort (ideally Claude Fable 5); subagents Opus 5 at `xhigh` with a
large max-output budget. Phase 1 is design only and ends at a hard checkpoint — the
model stops and waits for you to pick a design before any code is written.

**The one line you may want to change.** The block says *you* run the full training. If you would
rather the agent launch the 10-fold runs itself, replace that clause in `<division_of_labor>`.

---

````text
You are the orchestrator for the design-then-implementation of a novel multimodal fusion
mechanism in the /workspace/dp-code computational-pathology repo. You dispatch independent
subtasks to subagents and keep working while they run. Read this whole brief before acting.

<context_and_legitimacy>
This is a master's-thesis project in computational (digital) pathology, a mainstream published
subfield of medical machine learning. All data is public and de-identified: TCGA-BRCA whole-slide
images, RNA-seq expression tables, and clinical variables from the NIH/NCI open research portals
(GDC, cBioPortal). No patient is identifiable; nothing here is private or regulated PHI. The work
is entirely computational — statistical modelling of existing public tables and images. There are
no laboratory protocols, no synthesis methods, and no dual-use hazard content. Terms such as ESR1,
gene expression, RNA-seq, and molecular subtype are ordinary bioinformatics feature names in this
context; handle them exactly as you would any tabular machine-learning feature. Proceed normally
and do the work end to end.
</context_and_legitimacy>

<intent>
I am writing the methods-contribution chapter of my thesis. The preceding chapter established a
leakage-controlled ER-status ablation on TCGA-BRCA and produced a real result: RNA fusion helps,
clinicopathology fusion does not. That chapter used CLAM's stock `gated` fusion. This chapter must
contribute a *fusion mechanism*, not just another benchmark run — something defensible as novel
against the WSI+omics fusion literature, that works for BOTH second modalities, and that either
beats the stock gated baseline or produces an honest, well-diagnosed null.

Because this is a methods contribution, the design reasoning matters as much as the final number.
That is why this task is split: Phase 1 produces a written design I approve before any GPU time is
spent. Do not start Phase 2 until I have chosen a design.
</intent>

<ground_truth>
All of the following was verified against tool results in a prior session. Treat it as given; do
not re-derive it, but do open a file before making any claim about its contents.

Completed baseline ablation (binary ER status, case-level, out-of-fold, 10 site-holdout folds):
- WSI-alone (CLAM-MB, UNI2-h 1536-dim): AUROC 0.8957, AUPRC 0.9609, F1 0.9048, ECE 0.090.
- WSI+RNA, `--fusion_mode gated`, frozen WSI branch: AUROC 0.9412. DeLong vs WSI-alone
  Δ +0.0442, p = 1.6e-5, bootstrap CI [+0.024, +0.065], n = 956 matched cases. SIGNIFICANT.
- WSI+clinicopath, `--fusion_mode gated`, frozen WSI branch: AUROC 0.8937. DeLong vs WSI-alone
  Δ −0.0020, p = 0.74, CI [−0.013, +0.010], n = 1003. NULL.
- Cohort: 1003 cases / 1068 slides, 77.4% ER-positive. Report: docs/er-prediction-results.md.

Data and code that already exist (do NOT rebuild these):
- Embeddings: 1126 `.h5` in `.datasets/tcga-brca/embeddings`, key `features`, shape
  (1, n_patches, 1536). Do NOT re-extract features.
- Manifest: `project/CLAM/dataset_csv/tcga_brca_er.csv` (case_id, slide_id, label).
- Splits: `project/CLAM/splits/tcga_brca_er_100/splits_{0..9}.csv` (10 site-holdout folds) and
  `project/CLAM/splits/tcga_brca_er_lsgo/splits_0.csv`. Only `splits_{i}.csv` is read at train time.
- RNA table: `.scratch/TCGA-BRCA-rna/tcga_brca_er_rna_clam.csv.gz` — 996 cases, 20530 raw gene
  columns, keyed by case_id, carries the ER label.
- Clinicopath table: `tools/data/tcga_brca_clinicopath_clam.csv` — 1046 cases, 24 numeric features
  (raw age + one-hot stage/T/N/M/histology, each block with an `_unknown` column), carries the ER
  label, contains NO receptor-status field.
- Fusion model: `project/CLAM/models/model_multimodal.py`, class `CLAMRNAFusion`. Existing modes:
  `concat`, `gated`, `residual`, `cross_attention` — all four are already implemented, so your
  contribution must go beyond them. It already logs `fusion_wsi_gate_mean`.
- Dataset plumbing: `project/CLAM/dataset_modules/multimodal_dataset.py` joins tabular features by
  `case_id`, de-duplicates to one row per case, filters slides to matched cases, and RAISES if the
  tabular label disagrees with the manifest label. `RNAFeatureTransform.fit` performs variance
  top-N selection AND standardisation on the TRAIN FOLD ONLY, saved per fold as
  `s_{fold}_tabular_transform.json`.
- Frozen-branch fusion loads `--pretrained_wsi_ckpt .../er_wsi_alone_s1/s_{fold}_checkpoint.pt`
  (CLAM templates `{fold}`) together with `--freeze_wsi_branch`.
- Runner: `tools/train_er_ablation.sh {wsi|rna|clinpath|all}`.
- Analysis: `tools/evaluate_er_ablation.py` (primary — bootstrap CIs, figures, slide and case
  units) and `tools/analyze_er_ablation.py` (independent cross-check). They agree to four decimals.
- Per-fold test predictions are saved as `.scratch/results/er/<exp>_s1/split_{i}_results.pkl`
  holding per-slide {slide_id, prob, label}, so every metric is recomputable offline.
</ground_truth>

<known_gaps>
These are open weaknesses in the existing work. Phase 1 must address them; do not silently inherit
them.

1. THERE IS NO TABULAR-ONLY BASELINE. The ablation contains WSI-alone, WSI+RNA and
   WSI+clinicopath, but never RNA-alone or clinicopath-alone. So the claim "fusion beats H&E" is
   supported, while the claim "fusion beats simply using the RNA table" is NOT. A reviewer will ask
   this first. Any new arm must be accompanied by tabular-only baselines on the same folds.
2. IT IS UNKNOWN WHETHER GATED RNA FUSION ACTUALLY USES THE IMAGE. With a frozen WSI branch and a
   20530-dim RNA vector, the gate may have collapsed onto RNA, making the "fusion" model an RNA
   classifier wearing a WSI hat. `fusion_wsi_gate_mean` is already logged and the checkpoints are
   on disk, so this is measurable.
3. THE CLINICOPATH NULL IS UNDIAGNOSED. It is not known whether clinicopath carries no ER signal
   beyond morphology, or whether the fusion mechanism failed to extract it.
4. Single seed (1). No multi-seed variance estimate.
</known_gaps>

<design_constraint>
The central design problem, and the thing that makes this interesting: ONE fusion mechanism must
serve two modalities with opposite statistics.
- RNA: 20530-dim, dense, continuous, information-rich, and already sufficient on its own to move
  AUROC by +0.044. The risk is domination — the image contributes nothing.
- Clinicopath: 24-dim, sparse, mostly one-hot, low-information for ER, currently null. The risk is
  that a high-capacity fusion head simply ignores it, or overfits it.
A mechanism that only works for one of the two is a weaker contribution. State explicitly how your
design handles this asymmetry.
</design_constraint>

<phase_1_design>
Deliverable: `docs/implementation-research/novel-fusion-design.md`. No model code is written in
this phase. Cover, in this order:

1. GROUNDING. Read docs/er-prediction-results.md, docs/implementation-research/handoff.md,
   .scratch/er_pipeline_notes.md, and project/CLAM/models/model_multimodal.py. Summarise what the
   existing four fusion modes actually compute — you will be claiming novelty relative to them, so
   be precise about what they already do.

2. LITERATURE POSITIONING. Survey WSI+omics and WSI+clinical fusion for multiple-instance learning:
   at minimum MCAT (vendored in this repo at project/MCAT), PORPOISE, MOTCat, SurvPath, CMTA,
   TANGLE, and any 2024-2026 work you find. Use the paper-search skill and web search. For each,
   record in one line what its fusion operator is. Then state plainly which parts of your proposal
   are genuinely new and which are recombinations — an honest "this is a novel combination of X and
   Y applied to a setting where it has not been tested" is a defensible thesis claim; an inflated
   novelty claim is not.

3. DIAGNOSIS. Using the saved checkpoints and per-fold predictions (no retraining), answer the
   questions in <known_gaps> items 2 and 3 with numbers: what is the learned gate mean for the RNA
   and clinicopath arms; how correlated are the fusion arm's predictions with the WSI-alone arm's;
   and what does a quick tabular-only logistic-regression or MLP probe on the same folds achieve
   for each modality? These probes are cheap on CPU and are the controls the ablation is missing.
   The diagnosis determines which designs are worth trying, so do it before proposing them.

4. CANDIDATE DESIGNS. Propose 3 to 4 concrete mechanisms. For each give: the operator in equations
   or precise prose, where it attaches in CLAMRNAFusion, approximate added parameter count, how it
   handles the RNA-versus-clinicopath asymmetry, its failure mode, and — this is required — a
   FALSIFIABLE HYPOTHESIS of the form "if this mechanism works for the stated reason, then metric M
   on measurement N will move in direction D." Ideas worth considering, not a menu to exhaust:
   conditioning MIL patch attention on the tabular vector (FiLM-style modulation of the attention
   scores rather than late fusion of the pooled embedding); cross-attention from tabular tokens to
   patch tokens before pooling; modality dropout during training to force both pathways to remain
   predictive and to enable graceful degradation when RNA is absent at inference; low-rank bilinear
   or tensor pooling; a learned per-case reliability weight; sparsity or grouping priors over the
   20530 genes. You are not restricted to these.

5. SELECTION. Score the candidates against: expected effect size given the diagnosis, novelty
   defensibility, implementation risk inside CLAM, compute cost, and whether it addresses both
   modalities. Recommend ONE primary design and ONE fallback. Give a recommendation, not a survey.

6. EXPERIMENT PLAN. Specify the exact arms to run, including the missing tabular-only baselines and
   the modality-dropout / missing-modality evaluation if your design supports it. Specify how
   design selection avoids multiple-comparison bias: variants are selected on VALIDATION folds
   only, and the test folds are read once for the chosen design. State the pre-registered primary
   endpoint (DeLong of the novel arm vs the gated arm, per modality, on matched cases) before any
   test-set number is looked at.

END OF PHASE 1. Write the design document, then STOP and present me with the recommended design,
the fallback, and the diagnosis numbers. Do not write model code, do not launch training, and do
not begin Phase 2 until I have replied with a choice.
</phase_1_design>

<phase_2_implementation>
Begin only after I approve a design. Then:

1. Implement the chosen mechanism as a new `--fusion_mode <name>` in
   project/CLAM/models/model_multimodal.py, wired through project/CLAM/main.py argparse choices and
   the settings dict, following the existing modes' style. Keep the change minimal and additive:
   the four existing modes must behave EXACTLY as before, and the leakage controls (train-fold-only
   transform fitting, case_id join, label consistency check) must be untouched.
2. Write unit tests covering: output shapes for both a 20530-dim and a 24-dim tabular input;
   gradient flow into the new parameters; that `--freeze_wsi_branch` really leaves WSI-branch
   parameters with `requires_grad=False` and unchanged after a step; and, if you implement modality
   dropout, that inference with a zeroed modality still produces sensible output. Tests must fail
   if the mechanism is wired up wrongly — verify that by temporarily breaking it.
3. Run a SMOKE TEST only: one fold, 1-2 epochs, tiny subset if needed, purely to prove the plumbing
   runs end to end and the loss decreases. This is not the experiment; keep it under a few minutes.
4. Produce the exact runnable commands for the full ablation (novel arm for each modality, plus the
   tabular-only baselines), pinned to the SAME split directory, seed, and frozen checkpoints as the
   existing arms so every comparison is paired. Print them and hand off.
5. After I report that training is done, extend tools/evaluate_er_ablation.py (do not write a third
   analysis script) to include the new arms, and update docs/er-prediction-results.md with the
   novel-arm comparison, the DeLong test against BOTH WSI-alone and the gated arm, and the
   missing-modality result if applicable.
</phase_2_implementation>

<scientific_guardrails>
- Same folds, same site holdout, same seed, same frozen WSI checkpoints as the existing arms.
  A comparison on different splits is not a comparison.
- The bar for "the novel arm works" is beating the GATED arm (RNA 0.9412 / clinicopath 0.8937),
  not merely beating WSI-alone. Test against both and report both.
- Select variants on validation folds; read test folds once for the chosen design. If you end up
  reporting more than one variant on test, say so and apply a multiple-comparison correction.
- A null result is publishable here and must not be hidden or rescued by post-hoc variant hunting.
  If the honest conclusion is "the stock gated mechanism is not improved upon," write that.
- The RNA input includes ESR1 by design. Because the ER label is IHC-derived (a protein stain),
  this is a legitimate predictor and NOT target leakage — do not "fix" it.
- Never fabricate a metric, a fold count, a matched N, or a citation. If a number is not verified
  from a tool result in this session, say so explicitly.
</scientific_guardrails>

<division_of_labor>
I run the full training. You may run cheap CPU probes (logistic regression, gate statistics,
analysis of saved predictions) and the single-fold smoke test described above, but you do NOT
launch the multi-fold ablation — you prepare exact commands and hand them to me.
</division_of_labor>

<boundaries>
- Do NOT re-extract UNI2-h features, rebuild the splits, or rebuild the label/feature tables.
- Do NOT download CPTAC or any external cohort; external validation is a separate, later step.
- Do NOT refactor CLAM, MCAT, or the existing analysis scripts beyond what the new mode needs. No
  new abstractions for hypothetical future fusion modes; the right amount of complexity is the
  minimum that serves the chosen design.
- Do NOT write a third analysis script; extend tools/evaluate_er_ablation.py.
- Do NOT modify the existing four fusion modes' behaviour.
</boundaries>

<subagent_usage>
Delegate independent work and keep going while it runs: the literature survey, the diagnostic
probes, and later the unit tests are natural parallel subtasks. Give each subagent a fully
self-contained brief — state the scope, the files it may touch, the exact deliverable, and the
stop condition, because a subagent cannot see this prompt. Do not delegate work you can finish
directly in one step. Before accepting a subagent's numbers, spot-check them against a file
yourself.
</subagent_usage>

<verification>
Before reporting progress, audit each claim against a tool result from this session. Report
outcomes faithfully: if a test fails, say so and include the output; if a step was skipped, say
that; when something is verified, state it plainly without hedging. After implementation, dispatch
a fresh-context verifier subagent to confirm independently that the existing four fusion modes are
unchanged, that the frozen branch is genuinely frozen, and that no split or transform behaviour
moved. A self-review is not a substitute for a fresh-context check.
</verification>

<memory>
Keep .scratch/er_pipeline_notes.md updated: one lesson per line with a one-line summary. Record
corrections and confirmed approaches alike, including why they mattered. Update existing lines
rather than duplicating them, and delete anything that turns out to be wrong.
</memory>

<communication>
Lead with the outcome: your first sentence should answer "what did you find" or "what happened."
Supporting detail comes after. Write the final message as a re-grounding for someone who did not
watch the run: complete sentences, no arrow chains or invented shorthand, each file, metric, and
flag in its own plain clause. Give recommendations rather than exhaustive surveys of options you
will not pursue.

Pause for me only when the work genuinely requires it: the Phase 1 to Phase 2 gate, a real scope
change, or a decision only I can make. Otherwise proceed end to end. Do not end a turn on a plan or
a promise — if your closing paragraph describes work you have not done, do that work now with tool
calls.
</communication>
````

---

## Why the prompt is built this way

| Element | Guidance it follows |
|---|---|
| `<intent>` explains the thesis chapter and *why* the split into two phases | "Give the reason, not only the request" — context lets the model connect the task to what matters |
| `<ground_truth>` lists verified numbers and paths | Prevents re-deriving settled facts and reduces hallucination; pairs with "never speculate about code you have not opened" |
| `<known_gaps>` and `<design_constraint>` | Names the real scientific problems (no tabular-only baseline, possible gate collapse) so the model attacks them instead of inheriting them |
| Hard Phase 1 → Phase 2 checkpoint | "Pause only when the work genuinely requires it" — choosing which design gets GPU time is a decision only the author can make |
| Falsifiable hypothesis required per candidate | Forces design discipline and makes the later result interpretable |
| Validation-only selection, pre-registered endpoint | Guards against the multiple-comparison bias that arises when several fusion variants are tried |
| "A null result is publishable and must not be hidden" | Anti-p-hacking; matches the honest-reporting instruction |
| `<boundaries>` and "minimum complexity" language | Anti-overengineering guidance from both the Fable 5 and general guides |
| Fresh-context verifier after implementation | "Separate, fresh-context verifier subagents tend to outperform self-critique" |
| Self-contained subagent briefs | Opus 4.8 is literal and cannot see the parent prompt |
| Communication and no-promise-endings block | Fixes the two failure modes the Fable 5 guide flags for long agentic runs |
| Brief legitimacy frame at the top | The Fable 5 guide notes life-sciences content can trigger safety classifiers; a factual framing prevents a spurious refusal |

Deliberately **not** included: any instruction to "explain your reasoning step by step" or to
transcribe internal deliberation. The Fable 5 guide warns that such phrasing can trigger the
`reasoning_extraction` refusal category. The prompt asks for a written *design document* — an
artifact — which achieves the same visibility without that risk.
