# Prompt — SOTA comparison for the H&E + arm-level CNV thread

Paste everything below the line into a fresh Claude Code session opened in `/workspace/dp-code`.

Run configuration to set **before** pasting:

- `/model` → Claude Fable 5 (this session is the orchestrator)
- `/effort` → `xhigh`

Subagent model and effort are set inside the prompt itself (`model: 'opus'`, `effort: 'xhigh'` on
every `agent()` call), so no further harness configuration is needed.

---

I am finishing a master's thesis chapter on multimodal PAM50 molecular-subtype classification, and
the chapter needs a state-of-the-art comparison section that a thesis committee and, later, a
journal reviewer will attack. The experiments are finished and frozen — nothing about the pipeline
is going to change in response to what you find. What I need is an accurate, verifiable answer to
one question: **against the most recent published work, where do our numbers actually stand, which
of our claims survive contact with the literature, and which have to be softened or dropped?**

The comparison has to be built on papers I can cite and a reviewer can check. A plausible-sounding
comparison built on a hallucinated metric would be worse than no comparison at all, because it would
survive my own review and fail theirs.

<what_we_built>
Four-class PAM50 classification (LumA / LumB / Basal / Her2; Normal-like dropped), trained on
TCGA-BRCA and externally validated on CPTAC-BRCA, fusing two modalities:

- **H&E whole-slide images** — UNI2-h patch features (1536-dim) → CLAM-MB multiple-instance
  learning, 10-fold CV on TCGA-BRCA (910 non-Normal cases, 1009 slides).
- **Arm-level copy-number variation** — 39 chromosome arms, each the median gene-level log2 over
  that arm, from cBioPortal. Deliberately coarse: shallow WGS resolves arm/segment scale, so the
  assay is cheap and clinically reachable, unlike a model over ~19,755 focal GISTIC calls. CNV was
  chosen over RNA because PAM50 labels are computed from the expression matrix itself, so an RNA
  branch leaks the target by construction; copy number is a different assay and carries no such
  circularity.
- **External cohort** — CPTAC-BRCA, 114 cases / 378 slides (Basal 27, Her2 14, LumA 56, LumB 17).
  Nothing was ever refit, tuned or thresholded on CPTAC. The fusion rule was fixed on TCGA before
  the external set was scored.
</what_we_built>

<our_results>
These are the numbers the comparison has to place. Read them from
`docs/cnv-wsi-fusion-external-validation.md` yourself rather than trusting this summary — it is
here so you know what the target is, not as a substitute for the source.

**External, TCGA-trained → CPTAC-tested, n = 114 cases, case-level macro AUROC:**

| Model | macro AUROC [95% CI] | balanced acc | Her2 recall |
|---|---|---|---|
| WSI only (CLAM-MB + UNI2-h) | 0.847 [0.791, 0.895] | 0.513 | 0/14 |
| CNV only (39 arms, logistic regression) | 0.888 [0.835, 0.933] | 0.716 | 12/14 |
| Fusion — equal-weight probability mean | 0.909 [0.858, 0.948] | 0.646 | 6/14 |
| Fusion — mean of prior-balanced WSI + CNV (post hoc) | 0.912 | 0.740 | 10/14 |

**Internal TCGA**, 599 cases with out-of-fold predictions: WSI 0.887, CNV 0.862–0.872, mean-fusion
0.922–0.926. The CNV arm's published internal headline is 0.866 ± 0.003 (5-fold × 10 reseeds).

**The fusion-operator ladder** — five trained fusion operators, 10 folds each, pooled out-of-fold
over the same 599 cases:

| arm | pooled macro AUROC | balanced acc |
|---|---|---|
| WSI only | 0.8872 | 0.6772 |
| CNV only | 0.8721 | 0.6784 |
| **probability mean (untrained)** | **0.9259** | **0.7513** |
| concat / gated / cross-attention / FiLM-attention / co-attention | 0.8827 / 0.8947 / 0.8917 / 0.8818 / 0.8992 | 0.665–0.685 |

**Every trained operator loses to the untrained probability mean**, and five operators ensembled
together still lose to two independently trained unimodal models. The proposed mechanism is error
correlation: φ = 0.656 among the jointly trained operators against φ = 0.193 between the two
unimodal arms. One confound is open and stated in the results document: all five operators
warm-start from the same WSI checkpoint, so "joint training collapses diversity" and "shared
initialisation collapses diversity" are not yet separated.

**Two further findings that need literature placement:**

- The WSI arm calls Her2-enriched 0/14 on CPTAC and stays at 0/14 after a 12× prior boost and under
  unsupervised SLD-EM prior estimation, while calling 26/51 in-domain. The calibration explanation
  was tested and refuted; we attribute it to domain shift.
- Aneuploidy burden alone (`mean_abs_log2`, one scalar) reaches 0.685 macro AUROC, which bounds how
  much of the 39-arm result is genome instability rather than an arm *pattern*.
</our_results>

<what_already_exists_on_disk>
`docs/implementation-research/PAM50/` holds a 50-paper survey completed **2026-07-31**:

- `README.md` (144 KB) — three tables (core / adjacent / background), a landscape section, §6 "Where
  this project sits", §7 "Open gaps", §8 "Checked and deliberately excluded", §9 "Where this search
  is weak".
- `paper-dossier.md` (293 KB) — per-paper long-form extractions.
- `survey-data.json` — 50 rows with fields `title, authors, year, venue, url, doi, modalities,
  datasets, endpoint, wsi_pipeline, fusion_method, metrics, external_validation, code,
  evidence_level, caveats, relevance, pid`, plus `critic_missing` (8 recovered papers),
  `critic_assessment`, `search_notes`, `gapfill_resolved`, `gapfill_triage`.

**That survey was built for a different thread.** It was assembled while the second modality was
RNA, and its six search routes were biomedical registries, CS preprints, Google Scholar, unimodal
baselines, expression-inference/radiogenomics, and citation-chaining. **Copy number was never a
search axis** — ten of the fifty rows mention CNV/CNA/GISTIC incidentally, none at arm level, none
involving shallow WGS, and the search notes mention copy-number terms twice in 32 KB. Exactly one
row is a WSI+CNV PAM50 paper: Amer et al. 2025 (arXiv:2509.03408), 10-fold CV on TCGA only,
CNV-alone 0.8284 and four-modality fusion 0.9153, both internal.

So the survey is a strong, trustworthy foundation for the WSI-only comparison and for the general
"fusion does not beat the strong unimodal arm" claim, and it is close to empty on the axes this
chapter now turns on. §8 lists papers already checked and ruled out, and §9 lists the search's own
known weak spots including unverified leads worth chasing. Both are load-bearing: §8 exists
specifically so the same papers are not re-searched.
</what_already_exists_on_disk>

<task>
Produce a SOTA comparison document for the H&E + arm-level CNV thread, in five phases.

**Phase 1 — Ground yourself in what we measured and what we already have.**
Read `docs/cnv-wsi-fusion-external-validation.md` in full, `CLAUDE.md`, and §6, §7, §8 and §9 of the
survey README. Establish the exact numbers, the exact protocol each was measured under, and the
reporting rules the project has committed to (the CNV-alone arm is reported every time fusion is
reported; the equal-weight mean is the baseline, not the WSI-only model; anything computed post hoc
on CPTAC is labelled post hoc).

**Phase 2 — Triage the 50 existing rows against the CNV thread.**
Go through `survey-data.json` row by row and classify each one for this comparison, not for the RNA
one. For each row record: is it a fair comparator, a near comparator needing a caveat, context only,
or irrelevant here — and the specific reason, referencing its actual protocol (class count,
prediction unit, internal vs external, label basis). Produce an explicit **do-not-re-search list**
so Phase 3 does not spend tokens rediscovering what §8 already ruled out.

**Phase 3 — Independent search, only where the survey is thin.**
Use the `paper-search` skill. Search these axes, which the existing survey does not cover:

1. **WSI + copy-number fusion for molecular subtype**, breast first, other cancers as precedent.
   Amer 2025 is the only known instance; establish whether it is genuinely the only one.
2. **Copy number → breast subtype without imaging**: arm-level, chromosome-arm, broad vs focal CNA,
   aneuploidy signatures, and clinically-deployed **shallow / low-pass WGS** classifiers. This is
   where the claim "a 39-feature sWGS-reachable assay is statistically indistinguishable from a
   UNI2-h + CLAM pipeline" lives or dies.
3. **Simple averaging vs learned fusion operators.** Evidence from any domain that late fusion or
   probability averaging beats trained multimodal operators, plus the mechanism literature —
   ensemble diversity and error correlation, modality collapse, greedy modality learning, unimodal
   ensembles outperforming joint training. Our ladder result is a negative result; a reviewer will
   ask whether it is already known.
4. **The specific operators we ran** — gated, cross-attention, FiLM conditioning, co-attention — as
   used in computational pathology multimodal work (the MCAT / PORPOISE / SurvPath / MOTCat family
   and successors), even where the endpoint is survival rather than subtype. We need operator
   precedent and any published head-to-head against a trivial baseline.
5. **CPTAC-BRCA as an external cohort** for breast subtype or PAM50 models, 2025–2026.
6. **HER2-enriched failure under cohort transfer**, and stain normalisation / foundation-encoder
   choice as the proposed remediation.
7. **A recency sweep**: anything relevant published or preprinted after 2026-07-31, the survey's
   cutoff. Today is 2026-08-06, so this window is narrow but must be swept.

**Phase 4 — Extract and verify.**
For every candidate that survives selection, extract the same field set the existing survey uses so
new rows are drop-in compatible with `survey-data.json`. Read full text where it is reachable; where
it is not, say so and mark the evidence level. Then verify: a separate check that each extracted
number really appears in the source with that metric name and that split type, and that each URL and
DOI resolves to the paper claimed.

**Phase 5 — Synthesise the comparison.**
Write the deliverable described below. It must answer, for each of our numbers, what the closest
published comparator is, whether we are above or below it, and — the part that matters most —
whether the comparison is legitimate given differences in class count, prediction unit (patch /
slide / case), label basis (RNA-derived PAM50 vs IHC surrogate), and evaluation regime (internal CV
vs external cohort). Where no legitimate comparator exists, say that plainly; an honest "nothing
published is comparable" is a finding, not a failure.
</task>

<orchestration>
Use a workflow for this. Author a workflow script and run it with the `Workflow` tool. Every
`agent()` call takes `{model: 'opus', effort: 'xhigh'}` — Opus 5 at xhigh for every subagent, with
this session (Fable 5, xhigh) as the orchestrator. Budget roughly 16–20 agents in total, which is
above the default per-workflow size guideline; that is authorised for this task.

Shape it roughly as: two grounding agents (results/claims, survey triage) → six parallel searchers,
one per axis in Phase 3 → extraction, pipelined so each searcher's candidates move to extraction as
soon as that searcher returns rather than waiting for the slowest → a verification stage → a final
synthesis agent. Use `pipeline()` rather than a barrier unless a stage genuinely needs every prior
result at once; deduplicating candidates across searchers before extraction is a real barrier and is
the exception.

Two things about how the searchers should report. **Have every searcher report every candidate it
finds with a relevance score and a one-line rationale, and make selection a separate stage.** Do not
tell a searcher to return only the most relevant papers — an instruction to be selective gets
followed literally at the point of discovery, and papers get dropped before anything has seen them
side by side. The narrowing to a citable shortlist happens in the selection stage, with the full
candidate list visible.

**Do not ask any agent to double-check its own work.** Verification belongs to the dedicated
verification stage, run by agents with fresh context that did not do the extraction.

Keep working while subagents run, and step in if one drifts off the axis it was given. Before you
report progress or state a finding to me, check it against an actual tool result from this session
rather than against your expectation of what the searcher should have found — if a number is
unverified, say it is unverified.
</orchestration>

<integrity_rules>
These are the rules the existing survey was built under, and the new material has to match them or
it cannot be merged with it.

- Never invent a metric, cohort size, venue, DOI or URL. Write `NR` wherever a value was not seen in
  the source.
- Copy every number verbatim with its metric name and split type. Never convert, round, average or
  harmonise across papers — a macro-F1 is not an AUROC, a slide-level number is not a case-level
  number, and an accuracy is not either of them.
- Record an evidence level per paper: full text read, abstract only, or title only. An abstract-only
  row means the details in it are as thin as the abstract was.
- Verify that every link resolves to the paper claimed.
- Mark unrefereed preprints as unrefereed.
- Flag class count (4 vs 5), prediction unit (patch / slide / case), and label basis (RNA-derived
  PAM50 vs IHC surrogate) on every comparator, because most cross-paper comparisons in this field
  are uncalibrated by an unknown amount without them.
- Where a paywall blocks verification, say which paper, which paywall, and what remains unverified,
  rather than filling the gap from a search-engine snippet.
</integrity_rules>

<boundaries>
- `docs/cnv-wsi-fusion-external-validation.md` is mine. Read it; do not edit it. Same for
  `README.md` and `CLAUDE.md`.
- Do not re-run, re-tune or re-analyse any experiment, and do not modify any of our numbers. If you
  believe one of our numbers is wrong or one of our claims is unsupportable, write that in the
  findings section of the deliverable and leave the number alone.
- Write exactly two new files: the comparison document and the new-rows JSON named below. Clean up
  any scratch files you create along the way.
- Pause and ask me only if something genuinely blocks the work. A paywalled paper is not a blocker —
  record it as unverified and move on.
</boundaries>

<deliverable>
**File 1 — `docs/implementation-research/PAM50/sota-comparison-cnv-fusion.md`**, structured as:

1. **What this is and how far to trust it** — what was searched, which registries worked and which
   did not, how many candidates were found and how many were extracted, the full-text vs
   abstract-only split, and the cutoff date.
2. **Reused from the existing survey** — which of the 50 rows are fair comparators for the CNV
   thread and why, and which were re-checked and set aside.
3. **New papers** — one table, with the comparability flags from the integrity rules on every row.
4. **The comparison, number by number.** One subsection per result of ours: WSI-only external,
   CNV-only internal and external, the equal-weight-mean fusion, and the operator ladder. Each gives
   the closest published comparator, the direction and size of the gap, and an explicit statement of
   whether the comparison is legitimate.
5. **Claims that survive** — the novelty claims that hold after this search, each with the specific
   evidence that supports it and the specific paper that would have pre-empted it if it existed.
6. **Claims that must be softened or dropped**, with what to say instead.
7. **Still unverified** — paywalls, abstract-only rows, leads not chased, and what it would take.

Match the register of the existing survey README: plain declarative prose, tables where a table
genuinely helps, no padding. Cover the substance and stop; a long document is not a thorough one.

**File 2 — `docs/implementation-research/PAM50/sota-comparison-new-rows.json`** — the new papers
only, as objects using the exact field names of `survey-data.json`'s `rows`, so they can be merged
into the survey later.

An example of the level of specificity a comparison entry needs — this one is drawn from the
existing survey and is the standard to hit, not a paper to re-verify:

<example>
**Amer et al. 2025 (arXiv:2509.03408), against our CNV-only arm.**
Modalities H&E WSI (as patches and as a cell graph) + CNV + clinical, TCGA-BRCA only, 977 patients
after exclusions, 4-class PAM50 with labels from Netanely et al. 2016, patient-level, 10-fold CV.
CNV-alone macro-AUC 0.8284; four-modality fusion 0.9153. Both internal.

Comparison is legitimate on class count, label basis and prediction unit, and illegitimate on
regime: their 0.8284 is internal 10-fold and our 0.888 is a held-out cohort, so the numbers sit on
opposite sides of a transfer boundary and cannot be differenced. The defensible statement is that
their CNV arm is 0.8284 internal against our 0.862–0.872 internal, with our external 0.888 having no
counterpart in their work. Their clinical modality is the strongest single arm at accuracy 70.43% /
macro-AUC 0.8522, but the EHR vector's contents are unspecified; if it encodes ER/PR/HER2 the
comparison is circular, and that has to be stated whenever their fusion number is quoted.
</example>
</deliverable>

<success_criteria>
The comparison is done when:

- Every claim in `docs/cnv-wsi-fusion-external-validation.md` §9 (Positioning) has either a named
  published comparator or an explicit, evidenced statement that none exists.
- The specific claim "no published multimodal PAM50 model has an external, never-trained-on,
  PAM50-specific evaluation" has been re-tested against papers published since 2026-07-31 and
  against the copy-number axis the original survey never searched — this claim is the thesis's
  central novelty argument and is the single most important thing to try to break.
- The ladder result (every trained operator loses to the untrained mean) is placed against whatever
  prior work exists on averaging-beats-fusion, so I know whether it is novel, known-but-unreported
  in this field, or already published elsewhere.
- Every number attributed to another paper has been seen in that paper's text by an agent in this
  run, or is explicitly marked unverified.

Open with what you found: the headline is which of our claims survived and which did not, not a
description of the process that got there. Write the final summary for someone who did not watch any
of the tool calls.
</success_criteria>

Build the workflow and run it end to end.
