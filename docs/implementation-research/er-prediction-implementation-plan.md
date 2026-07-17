# ER-status prediction on TCGA-BRCA — implementation plan + agent-orchestration prompt

**Task:** Binary ER (estrogen-receptor) status prediction from H&E WSIs on TCGA-BRCA, reusing the
existing **UNI2-h → CLAM attention-MIL** pipeline, with a **mandatory WSI-alone vs. fusion ablation**.
This is the first implementation step of the ER→ODX arc laid out in
`next-steps-action-plan.md`, and it operationalises the reliability verdict in
`tcga-brca-reliable-fusion-task-report.md` and the clinical framing in
`er-prediction-clinical-need-grounded.md`.

**Decisions locked with the author (2026-07-17):**

| Decision | Choice |
|---|---|
| Scope of this step | **Internal TCGA first; CPTAC external validation staged as Phase 2** |
| ER label source | **cBioPortal / GDC clinical**; **1% cutoff primary, 10% cutoff as sensitivity analysis** |
| Fusion modality | **Three-way ablation: WSI-alone vs. WSI+RNA vs. WSI+clinicopath** |
| Deliverable of this session | **This document** (the plan + the orchestration prompt). No code executed. |

**Framing (why this task, for the reviewer):** the contribution is *not* "predict ER" — ER IHC is
cheap and mandated. The contribution is a **leakage-controlled test of whether a second modality
beats H&E-alone for a clinically-anchored label**, on a pipeline with a clean external-validation
path (CPTAC, Phase 2). ER is chosen as the proof-of-concept because its label is IHC-derived (so RNA
fusion is *not* target leakage, unlike PAM50) and it carries the most reproducible H&E signal in the
literature (AUROC 0.80–0.82 across three independent groups).

---

## 1. Verified starting state (checked against the repo, 2026-07-17)

- **UNI2-h embeddings are already extracted:** 1,127 `.h5` files in `.datasets/tcga-brca/embeddings`.
  The WSI branch needs no feature re-extraction for Phase 1.
- **No ER label exists on disk.** `tools/data/` holds only PAM50, OS, and DFI label CSVs. ER-status
  sourcing is a genuine prerequisite, not a label swap.
- **CLAM already supports fusion:** `project/CLAM/main.py` exposes
  `--fusion_mode {concat,gated,residual,cross_attention}`, `--tabular_csv`, `--tabular_top_n_features`,
  `--fusion_hidden_dim`; the model lives in `project/CLAM/models/model_multimodal.py`. The wrapper
  `tools/train_pam50_multimodal.sh` freezes the WSI branch and loads a pretrained checkpoint by default.
- **RNA table exists:** `.scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz`
  (CLAM-format) plus the `download-rna.py` / `prepare-rna-wsi-classification.py` pipeline.
- **Known infra bug (from the memo):** `tools/train_pam50_final.sh` points `--data_root_dir` at the
  empty `.datasets/embeddings`; real data is `.datasets/tcga-brca/embeddings`. Fix before any run.
- **Task registry:** CLAM maps `--task <name>` → `dataset_csv/<name>.csv`. There is **no
  `tcga_brca_er` task yet**; it must be added.

---

## 2. Phased technical plan

### Phase 0 — Infra fixes (fast, do first)
- Repoint `tools/train_pam50_final.sh` `--data_root_dir` to `.datasets/tcga-brca/embeddings`.
- Confirm every embedding `.h5` opens and exposes the expected `[N,1536]` feature key.

### Phase 1 — ER + clinicopath label engineering
- Pull TCGA-BRCA clinical from **cBioPortal** (`brca_tcga_pan_can_atlas_2018`; cross-check `brca_tcga`).
  Fields: `ER_STATUS_BY_IHC`, and — for the cutoff sensitivity — any percent-positive / ER-level
  category field available (`er_level_cell_percentage_category` in the GDC/BCR clinical XML).
- Emit **`tools/data/tcga_brca_er_labels.csv`** (`case_id,label`) with `ER-negative=0, ER-positive=1`.
  - **Primary label = 1% cutoff** (ASCO/CAP 2020: ER-low 1–10% → positive).
  - **Sensitivity label = 10% cutoff** (ER-low → negative), emitted as a second column/file.
  - **Honest caveat to record:** if cBioPortal only exposes the pre-binarised `Positive/Negative`
    call (no percentage), the two cutoffs collapse. In that case, source the percentage from the GDC
    BCR biotab/XML; if still unavailable, document that the 1%/10% sensitivity could not be run and
    treat the provided call as 1%-equivalent. **Do not fabricate a percentage.**
- Emit **`tools/data/tcga_brca_clinicopath.csv`** (`case_id`, age, grade, AJCC stage, T/N/M,
  histological type) for the clinicopath fusion arm, from the same clinical pull.
- Drop `Indeterminate`/missing ER; report the retained N and class balance (expect ~78% ER+).

### Phase 2 — CLAM task registration + site-aware splits
- Add a **`tcga_brca_er`** branch to `project/CLAM/main.py` (`n_classes=2`,
  `label_dict={'ER-negative':0,'ER-positive':1}`, `csv_path='dataset_csv/tcga_brca_er.csv'`).
- Build **`project/CLAM/dataset_csv/tcga_brca_er.csv`** (`case_id,slide_id,label`) by joining the ER
  labels to the embedding `slide_id`s. Keep **all slides of a case in the same fold** (patient-level),
  **and additionally hold out by tissue-submitting site (TSS = barcode chars 6–7)** per Howard 2021 —
  CLAM's stock `create_splits_seq.py` is patient-level only, so this needs a **site-grouped split
  generator** (a small new script). Produce k=10 folds; also emit one **leave-site-groups-out** split
  for the per-site generalisation report.

### Phase 3 — WSI-alone baseline
- Train `clam_mb` (headline) on `tcga_brca_er`: `--embed_dim 1536 --weighted_sample --early_stopping
  --bag_loss ce --inst_loss svm`, on the site-aware folds. Save per-fold checkpoints — they are the
  frozen WSI branch for fusion **and** the model shipped to Phase-2 CPTAC.
- This is the number every fusion arm must beat.

### Phase 4 — Fusion feature tables
- **RNA arm:** produce an ER-matched CLAM-format RNA feature table (features keyed by `slide_id`;
  the label comes from the dataset_csv, not the RNA file). **Leakage check to state explicitly:** the
  ER label is IHC-derived, so feeding the transcriptome (incl. `ESR1`) is a legitimate predictor, not
  target leakage — this is exactly the property that makes ER (unlike PAM50) a clean fusion task.
- **Clinicopath arm:** convert `tcga_brca_clinicopath.csv` to a CLAM-format tabular table (one-hot /
  standardised), keyed by `slide_id`. **Fit encoders on the train fold only.**

### Phase 5 — Three-way ablation (the headline result)
Same folds, same site holdout, frozen pretrained WSI branch (`--freeze_wsi_branch`):
1. **WSI-alone** (Phase 3).
2. **WSI + RNA**, `--fusion_mode gated`.
3. **WSI + clinicopath**, `--fusion_mode gated`.
Report **AUROC + AUPRC + F1 (mean±std over folds)**, **per-site** breakdown, and a **DeLong test**
of each fusion arm vs. WSI-alone. Run the whole ablation at **both** ER cutoffs (1% headline, 10%
sensitivity) or document why the sensitivity arm could not run (Phase-1 caveat).

### Phase 6 — Aggregation, verification, reporting
- Aggregate metrics; add a calibration plot and confusion matrices.
- **A separate, fresh-context verifier subagent** re-derives: (a) no case/slide crosses folds,
  (b) no site crosses train/test in the leave-site-groups-out split, (c) the RNA/clinicopath encoders
  saw only train-fold data, (d) the fusion N matches the matched-modality intersection.
- Write `docs/er-prediction-results.md` with the ablation table, per-site table, cutoff sensitivity,
  and an explicit statement of whether fusion beat H&E-alone (a *null* is a publishable result here —
  it motivates the ODX chapter; **do not hide it**).

### Phase 2-external (staged, not this step)
Download CPTAC-BRCA WSIs + RNA → extract UNI2-h → run the frozen Phase-3/Phase-5 models →
report WSI-alone + fusion externally, stain-normalised, label-cutoff-harmonised. Kept out of scope
now per the author's decision.

---

## 3. Orchestration model (Fable 5 orchestrator, Opus 4.8 subagents)

Rationale from the model-specific prompting guides:
- **Fable 5 orchestrator** — dispatches and sustains parallel subagents reliably, navigates ambiguity,
  and self-verifies with fresh-context verifier subagents. Run at **`high`/`xhigh`** effort. Keep it
  from over-planning ("when you have enough info to act, act") and ground every progress claim against
  a tool result.
- **Opus 4.8 subagents** — literal, precise instruction-followers; give each a **fully-specified,
  self-contained brief** with explicit scope, run at **`xhigh`** effort for the coding tasks.

**Dependency graph (what parallelises):**

```
Phase 0 (infra fix) ─┐
Phase 1 (ER + clinicopath labels)  ──┐        [A, B, C run in parallel]
Phase 4-RNA (RNA table)            ──┤
Phase 2-code (task reg + split gen)──┘
        │ (labels + splits ready)
        ▼
Phase 3 (WSI-alone train)  ──►  Phase 5 (fusion ablation, gpu-serial)
        │
        ▼
Phase 6 (verify + report)  ◄── fresh-context verifier subagent
```

GPU-bound training (Phases 3, 5) is serial; everything upstream of it (label/RNA/clinicopath tables,
code plumbing) is independent and fans out.

---

## 4. Ready-to-paste orchestration prompt

Paste the block below to the **Fable 5 orchestrator**. It is written to the best-practices guides:
explicit intent, XML-structured, hard boundaries, self-verification, checkpoints only where real, and
a memory file for lessons. Subagent briefs inside it are the units the orchestrator dispatches to
**Opus 4.8 at `xhigh`**.

````text
You are the orchestrator (Claude Fable 5, effort=high) for an ML implementation task in the
/workspace/dp-code computational-pathology repo. You dispatch independent subtasks to Opus 4.8
subagents at xhigh effort and keep working while they run. Read this whole brief before acting.

<intent>
I'm building the first implementation step of a master's thesis on multimodal breast-cancer
biomarker prediction. This step must produce a leakage-controlled, site-split ablation showing
whether a second modality (RNA-seq OR clinicopathology) beats H&E-alone for binary ER status on
TCGA-BRCA, reusing the existing UNI2-h → CLAM attention-MIL pipeline. The result — including a null
result — feeds a thesis chapter and a later CPTAC external-validation step. Correctness of the
splits and the leakage argument matters more than headline AUROC.
</intent>

<ground_truth>
- UNI2-h embeddings already exist: 1,127 .h5 files in .datasets/tcga-brca/embeddings (1536-dim). Do
  NOT re-extract WSI features.
- No ER label exists on disk; you must source it from cBioPortal/GDC clinical.
- CLAM (project/CLAM/main.py) already supports --fusion_mode {concat,gated,...}, --tabular_csv,
  --tabular_top_n_features, --fusion_hidden_dim; fusion model is models/model_multimodal.py; the
  wrapper tools/train_pam50_multimodal.sh freezes the WSI branch and loads a pretrained checkpoint.
- CLAM maps --task <name> to dataset_csv/<name>.csv. Task tcga_brca_er does NOT exist yet; add it.
- RNA CLAM-format table: .scratch/TCGA-BRCA-rna/TCGA_BRCA_RNA_primary_tumor_4class_clam.csv.gz.
- Bug to fix: tools/train_pam50_final.sh --data_root_dir points at empty .datasets/embeddings; real
  path is .datasets/tcga-brca/embeddings.
</ground_truth>

<locked_decisions>
- Scope: TCGA-internal only this run. Do NOT download CPTAC or any external cohort.
- ER label: cBioPortal/GDC. Primary cutoff = 1% (ER-low positive); also emit a 10% cutoff
  (ER-low negative) as a sensitivity label. If no percent-positive field is available anywhere,
  document that the sensitivity arm cannot run and treat the provided call as 1%-equivalent —
  never fabricate a percentage.
- Fusion: three-way ablation — WSI-alone vs WSI+RNA vs WSI+clinicopath — same folds, same site
  holdout, frozen WSI branch.
- Splits: patient-level AND tissue-submitting-site holdout (TSS = barcode chars 6-7), per Howard 2021.
</locked_decisions>

<plan>
Execute the six-phase plan in docs/er-prediction-implementation-plan.md §2. Fan out the independent
work first, then train, then verify:
  Parallel wave (dispatch together): Subagent-LABELS, Subagent-RNA, Subagent-CODE.
  Then: Subagent-TRAIN-WSI (WSI-alone baseline).
  Then: Subagent-ABLATION (WSI+RNA and WSI+clinicopath, reusing the frozen baseline checkpoints).
  Finally: Subagent-VERIFY (fresh context) then Subagent-REPORT.
</plan>

<subagent_briefs>
Give each subagent a self-contained brief (Opus 4.8 is literal — state scope and stop-conditions
explicitly). Templates:

  Subagent-LABELS (xhigh): "Fetch TCGA-BRCA clinical from cBioPortal (brca_tcga_pan_can_atlas_2018,
  cross-check brca_tcga). Produce tools/data/tcga_brca_er_labels.csv (case_id,label; ER-negative=0,
  ER-positive=1) at the 1% cutoff, plus a 10%-cutoff column/file. Produce
  tools/data/tcga_brca_clinicopath.csv (case_id, age, grade, AJCC stage, T/N/M, histology). Drop
  Indeterminate/missing ER. Print retained N and class balance. Do not touch model code. If no
  percent-positive field exists, say so explicitly and stop — do not invent one."

  Subagent-RNA (xhigh): "Produce an ER-matched CLAM-format RNA feature table keyed by slide_id from
  .scratch/TCGA-BRCA-rna/, reusing prepare-rna-wsi-classification.py conventions. Features only —
  labels come from the dataset_csv. Fit any feature selection on train folds only (leave the
  top_n_features selection to CLAM's --tabular_top_n_features). Do not modify CLAM."

  Subagent-CODE (xhigh): "In project/CLAM/main.py add a tcga_brca_er task (n_classes=2,
  label_dict={'ER-negative':0,'ER-positive':1}, csv_path='dataset_csv/tcga_brca_er.csv'). Write a
  site-grouped split generator that keeps all slides of a case in one fold AND holds out by TSS
  (barcode chars 6-7); emit k=10 folds and one leave-site-groups-out split. Also fix the
  train_pam50_final.sh --data_root_dir bug. Do not train anything. Keep changes minimal — no
  refactors beyond what the task needs."

  Subagent-TRAIN-WSI (xhigh): "Once dataset_csv/tcga_brca_er.csv and the splits exist, train clam_mb
  on tcga_brca_er (--embed_dim 1536 --weighted_sample --early_stopping --bag_loss ce --inst_loss svm)
  over the site-aware folds. Save per-fold checkpoints. Report per-fold and mean±std AUROC/AUPRC/F1."

  Subagent-ABLATION (xhigh): "Using the frozen WSI checkpoints, run WSI+RNA (--fusion_mode gated,
  --tabular_csv <rna>) and WSI+clinicopath (--fusion_mode gated, --tabular_csv <clinicopath>) on the
  SAME folds. Run at both ER cutoffs if available. Report AUROC/AUPRC/F1 mean±std, per-site, and a
  DeLong test of each fusion arm vs WSI-alone."

  Subagent-VERIFY (fresh context, xhigh): "Independently re-derive and confirm: no case or slide
  crosses folds; no site crosses train/test in the leave-site-groups-out split; RNA and clinicopath
  encoders saw only train-fold data; fusion N equals the matched-modality intersection. Report any
  violation with the offending IDs. Report coverage, not just pass/fail — surface anything uncertain."

  Subagent-REPORT (xhigh): "Write docs/er-prediction-results.md: ablation table, per-site table,
  cutoff sensitivity, calibration, and one plain sentence stating whether fusion beat WSI-alone."
</subagent_briefs>

<boundaries>
- Do NOT download CPTAC or any external cohort; do NOT re-extract UNI2-h features.
- Do NOT add features, refactor, or introduce abstractions beyond what each phase needs.
- The RNA input includes ESR1 by design and is NOT target leakage for an IHC-derived ER label — do
  not "fix" this. PAM50-style leakage does not apply here; state this reasoning in the report.
- Never fabricate a clinical value (ER percentage, a metric, a fold count). If a number is not
  verified from a tool result this session, say so.
- Report a null ablation result honestly — it is a valid outcome, not a failure to hide.
</boundaries>

<verification>
Before reporting progress, audit each claim against a tool result from this session. Establish a
verification checkpoint after Phase 5: dispatch Subagent-VERIFY with fresh context against the split
and leakage spec above; a self-critique is not a substitute. If tests or runs fail, report the
failure with its output.
</verification>

<checkpoints>
Pause for me only when the work genuinely requires it: (a) if the ER percent-positive field is
unavailable and the 10% sensitivity arm therefore cannot run, tell me and proceed with the
1%-only result; (b) if the matched fusion N drops below ~400 cases (underpowered). Otherwise proceed
end to end without asking. Do not end a turn on a plan or a promise — do the work with tool calls.
</checkpoints>

<memory>
Keep a lessons file at .scratch/er_pipeline_notes.md: one lesson per line with a one-line summary
(e.g. the exact cBioPortal field that carried ER%, the matched fusion N, any split gotcha). Record
corrections and confirmed approaches; update rather than duplicate.
</memory>

<communication>
When you report to me, lead with the outcome (did fusion beat H&E-alone, at what N, with what
split), then the supporting detail. Write the final summary as a re-grounding for someone who did
not watch the run: complete sentences, no arrow-chains, each file/metric in its own clause.
</communication>
````

---

## 5. Open items deferred (not blockers for Phase 1)

- **ER percent-positive availability** — the 1% vs 10% sensitivity depends on a percentage field
  existing in cBioPortal/GDC clinical; the orchestrator prompt handles the fallback honestly.
- **Matched fusion N** — the RNA-matched, ER-labelled intersection sets the power ceiling; the
  checkpoint rule flags it if it falls below ~400.
- **CPTAC Phase 2** — WSI download + UNI2-h extraction + frozen-model external run; its own plan once
  Phase 1 lands a result.
- **DeLong vs. bootstrap** for the significance test — either is defensible; DeLong specified for
  paired AUROC comparison on the same folds.
