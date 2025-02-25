import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.pytorch_dataset import MaskingDataset
from src.data.utils import get_splits

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold
import numpy as np

def main():
    NB_FOLDS = 5
    CLASSES = ["cardiomegaly","pneumonia","atelectasis","pneumothorax","effusion"]
    #Load the base dataset
    training_data, testing_data = get_splits(NB_FOLDS,data_dir="./data/processed")

    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)


    with open("./data/interim/test_results_tabular.csv", "a") as csv_file:
        csv_file.write(f"model_name,class,fold,auc")

    for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):
        #Split with fold
        print(f"\nStart FOLD {i}:")
        train_data = MaskingDataset(data_dir="./data/processed")
        train_data.img_labels = training_data.img_labels.iloc[train_index].reset_index(drop=True)
        train_data.img_paths = np.array(training_data.img_paths)[train_index]
        train_data.roi_paths = np.array(training_data.roi_paths)[train_index]
        
        val_data = MaskingDataset(data_dir="./data/processed")
        val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
        val_data.img_paths = np.array(training_data.img_paths)[val_index]
        val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

        #Drop rows with nan values for the features of interest
        train_data.img_labels = train_data.img_labels.dropna(subset=["PatientBirth","PatientSex_DICOM","Projection"])
        val_data.img_labels = val_data.img_labels.dropna(subset=["PatientBirth","PatientSex_DICOM","Projection"])
        testing_data.img_labels = testing_data.img_labels.dropna(subset=["PatientBirth","PatientSex_DICOM","Projection"])

        #get min and max year of train data to normalize birth feature
        min_year = train_data.img_labels["PatientBirth"].min()
        max_year = train_data.img_labels["PatientBirth"].max()

        #Get the features from train
        birth_feature = [(year-min_year)/(max_year-min_year) for year in train_data.img_labels["PatientBirth"]]
        sex_feature = [0 if s=="M" else 1 for s in train_data.img_labels["PatientSex_DICOM"]]
        projection_feature = [0 if "AP" in p else 1 for p in train_data.img_labels["Projection"]]
        X_train= np.stack([birth_feature,sex_feature,projection_feature],axis=-1)

        #Get the features from val
        birth_feature = [(year-min_year)/(max_year-min_year) for year in val_data.img_labels["PatientBirth"]]
        sex_feature = [0 if s=="M" else 1 for s in val_data.img_labels["PatientSex_DICOM"]]
        projection_feature = [0 if "AP" in p else 1 for p in val_data.img_labels["Projection"]]
        X_val= np.stack([birth_feature,sex_feature,projection_feature],axis=-1)

        #Get the features from test data
        birth_feature = [(year-min_year)/(max_year-min_year) for year in testing_data.img_labels["PatientBirth"]]
        sex_feature = [0 if s=="M" else 1 for s in testing_data.img_labels["PatientSex_DICOM"]]
        projection_feature = [0 if "AP" in p else 1 for p in testing_data.img_labels["Projection"]]
        X_test= np.stack([birth_feature,sex_feature,projection_feature],axis=-1)
        
        #One classifier for each class as it's a multilabel task
        lst_probas = [[] for l in testing_data.img_labels["Onehot"].tolist()]
        lst_auc = []
        for j,c in enumerate(CLASSES):
            #Get the labels for class c
            y_train = np.array([l[j] for l in train_data.img_labels["Onehot"].tolist()])
            y_val = np.array([l[j] for l in val_data.img_labels["Onehot"].tolist()])
            y_test = np.array([l[j] for l in testing_data.img_labels["Onehot"].tolist()])

            #Fit the classifier
            clf = LogisticRegression(random_state=1907).fit(X_train, y_train)

            #Predict on the test set
            probas = clf.predict_proba(X_test)[:, 1]
            for k,p in enumerate(probas):
                lst_probas[k].append(p)
            auc = roc_auc_score(y_test, probas)
            lst_auc.append(auc)

        #Save the probas and the auc
        with open(f"./data/interim/test_probas_Tabular_Fold{i}.csv", "a") as csv_file:
            csv_file.write(f"labels,probas")
            for label,proba in zip(testing_data.img_labels["Onehot"].tolist(),lst_probas):
                csv_file.write(f"\n{np.array(label)},{np.array(proba)}")
       
        with open("./data/interim/test_results_tabular.csv", "a") as csv_file:
            for j,c in enumerate(CLASSES):
                csv_file.write(f"\nTabular,{c},{i},{lst_auc[j]}")
                print(c,lst_auc[j])

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
