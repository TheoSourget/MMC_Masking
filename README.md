MMC_Masking
==============================

Study on the effect of masking the ROI in medical images to evaluate potential bias/shortcuts in datasets

In our study, we apply 5-different masking strategies to chest x-ray images and train a densenet121 model for each type using a 5-fold cross-validation protocol.

After training, models are evaluated on each types of images using the AUC, additional analysis with: SHAP, t-SNE and the cosine similarity are performed to better understand the behavior of the model.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

Install and run
---
**Clone the repo:**
```
git clone https://github.com/TheoSourget/MMC_Masking.git
cd MMC_Masking
```

**Create the environment and install dependencies:**
```
make create_environment
make requirements
```

Get and process the data:
```
Get the PadChest dataset from: https://bimcv.cipf.es/bimcv-projects/padchest/
Unzip every folder in the data/raw folder (let the images in the subfolder)

Get the CheXmask data from: https://physionet.org/content/chexmask-cxr-segmentation-data/0.4/
Put OriginalResolution/Padchest.csv in data/processed

make classes="Cardiomegaly,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion" data
```
After this step the data/processed folder should contain:
* an "images" folder containing the resized images
* an "rois" folder containing all the masks of lungs generated with cheXmask data
* a file "processed_label.csv" containing the metadata of all kept images and the one hot encoding label
* The Padchest.csv file used previously

**Train models**
In the .env file specify the parameter of your training like below
```
MODEL_NAME=NormalDataset

#Training parameters
NB_EPOCHS=250
NB_FOLDS=5
BATCH_SIZE=2
LEARNING_RATE=0.0001   
CLASSES=cardiomegaly,pneumonia,atelectasis,pneumothorax,effusion

#Early stopping parameters
ES_DELTA=0.001
ES_PATIENCE=25

#Masking parameters
MASKING_SPREAD=0
INVERSE_ROI=False
BOUNDING_BOX=False
```

Launch the training
```
make train
```

Change the masking parameters to apply other masking

**Evaluate models**

In src/models/eval_model.py:
change the following lines at the beginning of the main() function to put your models' name and the masking parameters to use
```
models_names=["NormalDataset","NoLungDataset_0","OnlyLungDataset_0","NoLungBBDataset_0","OnlyLungBBDataset_0"]
...
valid_params={
        "Normal":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        "NoLung":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        "NoLungBB":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        "OnlyLung":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        "OnlyLungBB":{"masking_spread":0,"inverse_roi":True,"bounding_box":True}
    }
```
The results will be printed in the terminal and in the generated data/interim/valid_resuls.csv file

**Visualisation**:

To reproduce the same visualisations as in the study run:
```
make visualization
```
The figures are saved in the reports/figures folder

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
