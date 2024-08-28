from fileinput import filename
from operator import index
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os.path


import torch
from torchvision.models import resnet50,densenet121
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.functional import sigmoid
from sklearn.metrics import roc_auc_score,f1_score

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from src.data.pytorch_dataset import MaskingDataset

import shap
from scipy.linalg import sqrtm
from scipy.spatial.distance import cosine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_auc_per_label():
    models_valid_results = pd.read_csv("./data/interim/valid_results.csv")
    mean_auc_per_class = models_valid_results.groupby(["class","training_set","valid_set"])["auc"].mean()
    for class_label in mean_auc_per_class.index.get_level_values('class').unique():
        result_class = mean_auc_per_class[mean_auc_per_class.index.get_level_values('class').isin([class_label])].droplevel(0)
        result_class = result_class.reset_index().pivot(columns='valid_set',index='training_set',values='auc')
        plt.figure()
        plt.title(f"Mean AUC for {class_label} across different masking strategies")
        sns.heatmap(result_class, annot=True,cmap="RdYlGn")
        plt.savefig(f"./reports/figures/mean_auc_{class_label}.png",format='png')

def generate_explainability_map():
    #Get hyperparameters 
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    model_name="NormalDataset"

    #Load the base dataset
    training_data = MaskingDataset(data_dir="./data/processed")
    testing_data = MaskingDataset(data_dir="./data/processed")

    #Split the dataset into training/testing splits
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=2, random_state = 1907)
    train_eval_split = splitter.split(training_data.img_labels, groups=training_data.img_labels['PatientID'])
    train_idx, test_idx = next(train_eval_split)
    training_data.img_labels = training_data.img_labels.iloc[train_idx].reset_index(drop=True)
    training_data.img_paths = np.array(training_data.img_paths)[train_idx]
    training_data.roi_paths = np.array(training_data.roi_paths)[train_idx]

    testing_data.img_labels = testing_data.img_labels.iloc[test_idx].reset_index(drop=True)
    testing_data.img_paths = np.array(testing_data.img_paths)[test_idx]
    testing_data.roi_paths = np.array(testing_data.roi_paths)[test_idx]
    

    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)
    
    valid_params={
        "Normal":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        # "NoLung":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        # "NoLungBB":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        # "OnlyLung":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        # "OnlyLungBB":{"masking_spread":0,"inverse_roi":True,"bounding_box":True}
    }

    for param_config_name in valid_params:
        print(model_name,param_config_name)
        for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):        
            print("\nFOLD",i)
            val_data = MaskingDataset(data_dir="./data/processed",**valid_params[param_config_name])
            val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
            val_data.img_paths = np.array(training_data.img_paths)[val_index]
            val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

            valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
            
            
            #Define model, loss and optimizer
            model = densenet121(weights='DEFAULT')#Weights pretrained on imagenet_1k
            
            # Freeze every layer except last denseblock and classifier
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features.denseblock4.denselayer16.parameters():
                param.requires_grad = True
           
            kernel_count = model.classifier.in_features
            model.classifier = torch.nn.Sequential(
             torch.nn.Flatten(),
             torch.nn.Linear(kernel_count, len(CLASSES))
            )
            
            for module in model.modules():
                if isinstance(module, torch.nn.ReLU):
                    module.inplace = False
                    
            try:
                model.load_state_dict(torch.load(f"./models/{model_name}/{model_name}_Fold{i}.pt"))
                model.to(DEVICE)
            except FileNotFoundError as e:
                print("No model saved for fold",i)
                continue

            images, _ = next(iter(valid_dataloader))
            images = images.to(DEVICE)
            background = images[:1]
            test_images= images[1:]
            e = shap.DeepExplainer(model, images)
            shap_values = e.shap_values(test_images)

def get_embedding(model_name,valid_params):
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    
    #Load the base dataset
    training_data = MaskingDataset(data_dir="./data/processed")
    testing_data = MaskingDataset(data_dir="./data/processed")

    #Split the dataset into training/testing splits
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=2, random_state = 1907)
    train_eval_split = splitter.split(training_data.img_labels, groups=training_data.img_labels['PatientID'])
    train_idx, test_idx = next(train_eval_split)
    training_data.img_labels = training_data.img_labels.iloc[train_idx].reset_index(drop=True)
    training_data.img_paths = np.array(training_data.img_paths)[train_idx]
    training_data.roi_paths = np.array(training_data.roi_paths)[train_idx]

    testing_data.img_labels = testing_data.img_labels.iloc[test_idx].reset_index(drop=True)
    testing_data.img_paths = np.array(testing_data.img_paths)[test_idx]
    testing_data.roi_paths = np.array(testing_data.roi_paths)[test_idx]
    

    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)
    
    
    models_flatten_output = {
        masking_param:[] for masking_param in valid_params
    }
    
    for masking_param in valid_params:
        print(f"\n{masking_param}")
        for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):        
            val_data = MaskingDataset(data_dir="./data/processed",**valid_params[masking_param])
            val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
            val_data.img_paths = np.array(training_data.img_paths)[val_index]
            val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

            valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
            
            
            #Define model, loss and optimizer
            model = densenet121(weights='DEFAULT')#Weights pretrained on imagenet_1k
            
            # Freeze every layer except last denseblock and classifier
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features.denseblock4.denselayer16.parameters():
                param.requires_grad = True
           
            kernel_count = model.classifier.in_features
            model.classifier = torch.nn.Sequential(
             torch.nn.Flatten(),
             torch.nn.Linear(kernel_count, len(CLASSES))
            )
            
            try:
                model.load_state_dict(torch.load(f"./models/{model_name}/{model_name}_Fold{i}.pt"))
                model.to(DEVICE)
            except FileNotFoundError as e:
                print("No model saved for fold",i)
                continue
            
            return_nodes = {
                "classifier.0": "flatten"
            }
            model = create_feature_extractor(model,return_nodes)
            model.eval()
            with torch.no_grad():
                labels_dataset = []
                for i, data in enumerate(valid_dataloader, 0):
                    inputs, labels = data
                    inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                    outputs = model(inputs)
                    models_flatten_output[masking_param].extend(outputs["flatten"].detach().cpu().tolist())
                    labels_dataset += labels
            models_flatten_output[masking_param] = np.array(models_flatten_output[masking_param])
            break
    return models_flatten_output, labels_dataset

def get_cosine():
    model_name="NormalDataset"
    valid_params={
        "NormalDataset":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        "NoLungDataset_0":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        "OnlyLungDataset_0":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        "NoLungDatasetBB_0":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        "OnlyLungDatasetBB_0":{"masking_spread":0,"inverse_roi":True,"bounding_box":True},
    }
    CLASSES = os.environ.get("CLASSES").split(",")
    
    models_flatten_output,labels_dataset = get_embedding(model_name,valid_params)

    no_lung_similarities = []
    only_lung_similarities = []
    no_lungbb_similarities = []
    only_lungbb_similarities = []
    for j in range(len(models_flatten_output["NormalDataset"])):
        normal = models_flatten_output["NormalDataset"][j]
        nolung = models_flatten_output["NoLungDataset_0"][j]
        nolungbb = models_flatten_output["NoLungDatasetBB_0"][j]
        onlylung = models_flatten_output["OnlyLungDataset_0"][j]
        onlylungbb = models_flatten_output["OnlyLungDatasetBB_0"][j]
        no_lung_similarities.append(1- cosine(normal,nolung))
        only_lung_similarities.append(1- cosine(normal,onlylung))
        no_lung_similarities.append(1- cosine(normal,nolung))
        no_lungbb_similarities.append(1- cosine(normal,nolungbb))
        only_lung_similarities.append(1- cosine(normal,onlylung))
        only_lungbb_similarities.append(1- cosine(normal,onlylungbb))
    print("all",np.mean(no_lung_similarities),np.mean(no_lungbb_similarities), np.mean(only_lung_similarities),np.mean(only_lungbb_similarities))

    #Per class
    for i,c in enumerate(CLASSES):
        no_lung_similarities = []
        only_lung_similarities = []
        no_lungbb_similarities = []
        only_lungbb_similarities = []
        class_indices = [j for j, l in enumerate(labels_dataset) if l[i] == 1 ]
        for j in class_indices:
            normal = models_flatten_output["NormalDataset"][j]
            nolung = models_flatten_output["NoLungDataset_0"][j]
            nolungbb = models_flatten_output["NoLungDatasetBB_0"][j]
            onlylung = models_flatten_output["OnlyLungDataset_0"][j]
            onlylungbb = models_flatten_output["OnlyLungDatasetBB_0"][j]

            no_lung_similarities.append(1- cosine(normal,nolung))
            no_lungbb_similarities.append(1- cosine(normal,nolungbb))
            only_lung_similarities.append(1- cosine(normal,onlylung))
            only_lungbb_similarities.append(1- cosine(normal,onlylungbb))
        print(c,np.mean(no_lung_similarities),np.mean(no_lungbb_similarities), np.mean(only_lung_similarities),np.mean(only_lungbb_similarities))

def main():
    # generate_auc_per_label()
    #generate_explainability_map()
    get_cosine()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
