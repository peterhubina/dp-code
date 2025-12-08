# Histological Hyperspectral Breast Cancer Recurrence Database (HistologyHSI-BC Recurrence)
Metastasis occurs in nearly 1 out of 3 breast cancer (BC) patients and significantly reduces survival rates, particularly in cases of distant metastases. As most distant metastases develop after diagnosis (i.e., recurrence) and remain incurable, there is a critical need for prognostic biomarkers to assess recurrence risk. Multimodal data analysis has emerged as a promising approach to integrate diverse information, offering a more comprehensive perspective. This study introduces the Histology HSI-BC (hyperspectral imaging - breast cancer) Recurrence Database, the first publicly accessible multimodal database designed to advance BC distant recurrence prediction. This database provides a promising resource for studying BC recurrence prediction and personalized treatment strategies by integrating the aforementioned multimodal data.

## Dataset 

The database comprises 47 histopathological whole-slide images (WSI), 677 hyperspectral (HS) images, and clinical and demographic data from 47 BC patients, of whom 22 (47%) experienced distant recurrence over a 12-year follow-up. Histopathological slides were digitized using a whole-slide scanner and annotated by expert pathologists, while HS images were acquired with an HS camera coupled to a bright-field microscope.

More information about the dataset can be found on:

## Usage

This repository contains the following scripts:
* `main.ipynb`: provide a basic example of how to load and perform some basic preprocessing to WSI (.mrxs) and HS (ENVI format) data using Python.
* `overlay_tissue_areas.ipynb`: provide a tutorial on manipulating annotations in GeoJSON format to overlay tissue compartments on a WSI using Python.
## Dependencies

Python script requires:
   - Openslide. Python module for reading whole-slide image formats. https://openslide.org/  
   - Spectral Python (SPy). Python module for hyperspectral image processing. https://www.spectralpython.net
   - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2
   - J. D. Hunter, "Matplotlib: A 2D Graphics Environment," in Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, May-June 2007, doi: 10.1109/MCSE.2007.55.
   - Virtanen, P., Gommers, R., Oliphant, T.E. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17, 261–272 (2020). https://doi.org/10.1038/s41592-019-0686-2


## License

Copyright 2025 Laura Quintana-Quintana, Esther Sauras-Colón, Javier Santana-Nunez, Alessio Fiorin

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
