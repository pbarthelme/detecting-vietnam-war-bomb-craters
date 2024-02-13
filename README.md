# Detecting Vietnam War bomb craters in declassified KH-9 satellite imagery

This is the code repository for the manuscript *Detecting Vietnam War bomb craters in declassified KH-9 satellite imagery* (in submission).

## Setup
This implementation uses Python 3.11 in combination with multiple other packages specified in the `./environment.yml` file. A more detailed environment file specifying the exact package versions used during the analysis is provided as `./environment_detailed.yml`. A conda environment with the corresponding packages can be created using the command `conda env create -f environment.yml`.

## Data Availability
All data used for this analysis is freely available. The original, scanned KH-9 images can be downloaded from [EarthExplorer](https://earthexplorer.usgs.gov/). The georeferenced KH-9 images area available [here](https://doi.org/10.7488/ds/7682) (Quang Tri) and [here](https://doi.org/10.7488/ds/7683) (Tri-border area). The detected craters, trained models and other data is available [here](10.5281/zenodo.10629987). The THOR bombing records can be downloaded [here](https://data.world/datamil/vietnam-war-thor-data) (`thor_data_vietnam.csv`). The GADM boundaries used to create some of the Figures are available [here](https://gadm.org) ([Vietnam](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_0.json), [Lao PDR](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_LAO_0.json), [Cambodia](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KHM_0.json)).

## Structure
The code is structured in Jupyter notebooks available in the `code/` folder. The notebooks need to be run in the correct order, indicated by the numbers in their filenames, as they rely on outputs from previous notebooks. Some functions are provided in separate files (`code/utils.py`, `code/evaluation.py` and `code/analysis.py`) to help with code organisation and reuse of functions across multiple notebooks.

Each notebook includes a short summary and descriptions of inputs and outputs. The corresponding parameters and file paths are specified in the `./config.yml` file. Any statistics, metrics or Figures created in the notebook that are used in the paper are also pointed out in the description. 

### [0_extract_imagery_footprints.ipynb](code/0_extract_imagery_footprints.ipynb)
This notebook extracts the imagery footprints of the KH-9 images (excluding no data values) as polygons. The polygons are used to avoid duplication of crater counts and as extents for plots. The notebook also creates grids of varying grid sizes for each study area that are used for the comparison of aggregated crater counts with the THOR bombing data (see analysis notebooks).

### [0_gen_training_tiles.ipynb](code/0_gen_training_tiles.ipynb)
This notebook splits up each KH-9 image raster file into tiles of 256 $\times$ 256 pixels. A random subset of these tiles is selected for each study area to be used for manual labelling and model training.

### [1_labels.ipynb](code/1_labels.ipynb)
The image tiles generated in the 0_gen_training_tiles notebook are used to manually create crater labels in QGIS. This notebook is a placeholder to visualise this extra step and provide a short summary of labelled craters.

### [2_create_training_data.ipynb](code/2_create_training_data.ipynb)
This notebook takes the selected image tiles (geotiff files) and the corresponding labels (geojson files with crater polygons manually created in QGIS) and transforms them into the training data for the neural network. The geojson files are rasterized and an additional boundary class for instance segmentation is added. The final data, consisting of matching image pairs (256 $\times$ 256 pixels) of KH-9 image and label, are split into training, validation and test set and saved in a .h5 file. 

### [3_train_model.ipynb](code/3_train_model.ipynb)
This notebook trains a UNet model using the Pytorch and segmentation-models-pytorch packages. In a first step multiple models are trained based on different settings of the alpha value in the focal loss function. During this step data from both study areas is used together. In a second step the alpha parameter with the best performance on the validation data of both study areas is selected. This model is then fine-tuned independently for each study area using only the training data of the specific study area. This results in two final models, one for each study area.

### [4_evaluate_model.ipynb](code/4_evaluate_model.ipynb)
Evaluates the final models on the test sets for each study area, calculating the model accuracy metrics presented in the paper.

### [5_predict_full_rasters.ipynb](code/5_predict_full_rasters.ipynb)
1) Applies the trained models to all KH-9 images across both study areas (raster files for each KH-9 image)
2) Post-processing of model predictions, removing boundary class pixels and identifying individual crater instances (raster files for each KH-9 image)
3) Extract individual crater polygons and centroids from post-processed raster files (geojson files for each KH-9 image)
4) Create final set of crater polygons and centroids for each study area by removing craters that were counted twice due to overlapping KH-9 images (only relevant for tri-border area as Quang Tri images were mosaicked before) 

### [6_figure_1a.ipynb](code/6_figure_1a.ipynb)
This notebook creates Figure 1a of the paper, showing an overview of the THOR bombing data.

### [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb)
This notebook analyses the crater predictions and compares them with the THOR bombing records for the Quang Tri study area.
1) Main summary statistics and counts for detected craters and subsets of THOR bombing records
2) Calculate correlation statistics across grid cells of different sizes
3) Create multiple Figures used in the paper

### [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)
This notebook analyses the crater predictions and compares them with the THOR bombing records for the Tri-border area.
1) Main summary statistics and counts for detected craters and subsets of THOR bombing records
2) Calculate correlation statistics across grid cells of different sizes
3) Create multiple Figures used in the paper

## Locations where outputs used in the paper were created:
### Key statistics
* **Total craters labelled**: [1_labels.ipynb](code/1_labels.ipynb)
* **Crater prevalence**: [2_create_training_data.ipynb](code/2_create_training_data.ipynb)
* **Alpha best**: [3_train_model.ipynb](code/3_train_model.ipynb)
* **Combined F1-score**: [4_evaluate_model.ipynb](code/4_evaluate_model.ipynb)
* **Number of predicted craters**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb), [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)
* **Number of predicted craters by crater class**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb), [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)
* **Number of bombs dropped for different subsets of the bombing records**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb), [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)
* **Number and percentage of bombing vs craters in grid cells of Quang Tri with >90% of bombing in previous year**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb)

### Tables
* **Table 1**: [4_evaluate_model.ipynb](code/4_evaluate_model.ipynb)
* **Table 2**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb), [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)

### Figures
* **Figure 1**: [6_figure_1a.ipynb](code/6_figure_1a.ipynb), [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb), [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)
* **Figure 4**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb)
* **Figure 5**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb)
* **Figure 6**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb), [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)
* **Figure 7**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb), [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)
* **Figure 8**: [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)

### Supplementary Materials
* **Supplementary Materials - Table S4**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb)
* **Supplementary Materials - Table S5**: [6_analysis_tba.ipynb](code/6_analysis_tba.ipynb)
* **Supplementary Materials - Figure S2**: [6_analysis_quang_tri.ipynb](code/6_analysis_quang_tri.ipynb)
