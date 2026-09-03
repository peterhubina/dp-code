# WSI + CNV evaluation summary (short form)

A 4-to-6 page summary of the PAM50 four-class H&E + arm-level CNV work: internal evaluation on
TCGA-BRCA (599 pooled out-of-fold cases) and external validation on CPTAC-BRCA (114 cases, never
trained on). The headline is that the untrained equal-weight probability mean of the two
independently trained arms beats every jointly trained fusion operator; the CNV-alone arm is
reported wherever fusion is.

Rebuild with `./build.sh` (runs `pdflatex` twice) or build `wsi_cnv_summary.tex` in any TeX
distribution; `results_table.tex` is standalone and can be `\input` into another document.

Every number comes from `data/`: `results_consolidated.csv`, `contrasts.csv`,
`external_per_class.csv` and `eval_notes.md` are the verified 2026-09-01 recompute (zero
discrepancies above 0.001 against the 2026-08-06 runs), and `cohort_counts.csv` is copied from
`.scratch/wsi_cnv_report/`. `eval_notes.md` names the protocol, bootstrap seed and source line
behind each value; Section 4 of the PDF lists every producing command and path.

The long-form report, with figures and the full control set, is `docs/wsi_cnv_evaluation.pdf`.
