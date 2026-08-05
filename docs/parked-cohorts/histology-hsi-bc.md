# HistologyHSI-BC Recurrence — parked cohort

**Status in this project: parked.** Nothing in the current pipeline reads it, no reported number
depends on it, and the numbers it once produced are void (see *Why it is parked*). This page exists
for two reasons: the repository's root `README.md` used to *be* this dataset's README, and the
attribution below is a licence obligation that must survive that file being rewritten.

---

## The dataset (upstream description)

> Metastasis occurs in nearly 1 out of 3 breast cancer (BC) patients and significantly reduces
> survival rates, particularly in cases of distant metastases. As most distant metastases develop
> after diagnosis (i.e. recurrence) and remain incurable, there is a critical need for prognostic
> biomarkers to assess recurrence risk. Multimodal data analysis has emerged as a promising approach
> to integrate diverse information, offering a more comprehensive perspective. This study introduces
> the Histology HSI-BC (hyperspectral imaging — breast cancer) Recurrence Database, the first
> publicly accessible multimodal database designed to advance BC distant recurrence prediction.

The database comprises **47 histopathological whole-slide images**, **677 hyperspectral images**,
and clinical and demographic data from **47 breast-cancer patients**, of whom **22 (47%)**
experienced distant recurrence over a 12-year follow-up. Slides were digitised with a whole-slide
scanner and annotated by expert pathologists; the HS images were acquired with a hyperspectral
camera coupled to a bright-field microscope.

The original README's "More information about the dataset can be found on:" sentence ended with a
dangling colon and no URL, so **no upstream landing page is recorded anywhere in this repository**.
That URL is one of the things the author has to supply.

## Where it lives here

| item | location |
|---|---|
| Data root | `${paths.hsi_bc_root}` — `.datasets/HistologyHSI-BC-Recurrence/` by default, `DP_HSI_BC_ROOT` to move it |
| WSIs (`.mrxs`) | `${paths.hsi_bc_root}/01_01_Histological_Images` |
| Tissue annotations (GeoJSON) | `${paths.hsi_bc_root}/01_02_Tissue_Annotations` |
| Clinical table | `${paths.hsi_bc_root}/Histology_HSI_BRCA_Recurrence.xlsx` |
| Pipeline | `tools/hsi_bc/run_pipeline.sh` → `prepare_manifest.py`, `infer_pam50.py` |
| Exploration notebooks | `tools/preprocessing/main.ipynb`, `tools/preprocessing/overlay_tissue_areas.ipynb` |

The two notebooks are the ones the upstream README described as `main.ipynb` and
`overlay_tissue_areas.ipynb`; they are **not** at the repository root, and both use working-directory
relative paths, so they only run with Jupyter's cwd set to their own directory.

`tools/hsi_bc/run_pipeline.sh` is the only path in this repository that still needs the **gated**
`MahmoodLab/UNI2-h` encoder weights (`UNI2H_CKPT_PATH`, default
`.scratch/checkpoints/uni2-h/pytorch_model.bin`) and the untracked `project/UNI/` clone: it tiles
raw slides rather than consuming pre-extracted features. Nothing on the PAM50 + CNV reproduction
path does.

## Why it is parked

**The tiling geometry does not match TCGA's.** HSI-BC slides were tiled at a 62.3 µm field of view
against the 128 µm used for the TCGA training features — a 2.055x scale difference. A CLAM model
trained on 128 µm tiles and applied to 62.3 µm tiles is being shown objects at the wrong physical
scale, so **every HSI-BC external number produced before this was noticed is void** and stays void
until the cohort is re-tiled at matched geometry. Re-tiling is possible (the pipeline above does the
tiling) but has not been done, and no result from this cohort is reported anywhere in the thesis.

A separate, earlier belief — that the cohort has no usable recurrence labels — was wrong: the labels
exist (47 cases, 22 relapses, ~12.5-year follow-up). The blocker is the magnification mismatch, not
the labels.

## Dependencies the upstream README cited

These are the citations the dataset's own README asked users to carry, and they belong in the thesis
bibliography if any HSI-BC material is used:

- **OpenSlide** — Python module for reading whole-slide image formats. <https://openslide.org/>
- **Spectral Python (SPy)** — hyperspectral image processing. <https://www.spectralpython.net>
- Harris, C.R., Millman, K.J., van der Walt, S.J. et al. *Array programming with NumPy.* Nature 585,
  357–362 (2020). <https://doi.org/10.1038/s41586-020-2649-2>
- J. D. Hunter, *Matplotlib: A 2D Graphics Environment*, Computing in Science & Engineering, vol. 9,
  no. 3, pp. 90–95, 2007. <https://doi.org/10.1109/MCSE.2007.55>
- Virtanen, P., Gommers, R., Oliphant, T.E. et al. *SciPy 1.0: fundamental algorithms for scientific
  computing in Python.* Nat Methods 17, 261–272 (2020). <https://doi.org/10.1038/s41592-019-0686-2>

## Attribution and licence (preserved verbatim)

The following notice was carried in this repository's root `README.md` until that file was rewritten
as the project README. It is the **upstream dataset's** notice and names the dataset's authors, not
this repository's author:

> Copyright 2025 Laura Quintana-Quintana, Esther Sauras-Colón, Javier Santana-Nunez, Alessio Fiorin
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
> > http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software distributed under the License
> is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
> or implied. See the License for the specific language governing permissions and limitations under
> the License.

**Unresolved, and only the author can resolve it.** The repository's root `LICENCE` is the unmodified
Apache-2.0 text with its `Copyright [yyyy] [name of copyright owner]` placeholder never filled, and
the notice above — sitting in the root README — was the only thing filling that role. As published,
the licence therefore attributed *this repository's code* to the HistologyHSI-BC dataset authors.
Moving the notice here removes that specific confusion but does not decide the repository's own
licence; see the licensing note in `CITATION.cff` and `REPRODUCING.md`.
