import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import os.path
import glob
import torch
from torch.nn.functional import sigmoid
from sklearn.metrics import roc_auc_score
from src.models.utils import get_model,make_single_pred
import pandas as pd
import ast
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_model(model,data_info):
    model.to(DEVICE)
    model.eval()
    lst_labels = []
    lst_preds = []
    lst_probas = []
    auc_scores = []
    with torch.no_grad():
        for i,img_info in tqdm(data_info.iterrows(),total=data_info.shape[0]):
            img_path = f"./data/processed/CXR14/images/{img_info['Image Index']}"
            labels = [img_info["Onehot"]]
            probas = make_single_pred(model,img_path)
            lst_labels.extend(labels)
            lst_probas.extend(probas.cpu().detach().numpy())
            lst_preds.extend(probas.cpu().detach().numpy()>0.5)
        
        lst_labels = np.array(lst_labels)
        lst_preds = np.array(lst_preds)
        lst_probas = np.array(lst_probas)
        for i in range(lst_labels.shape[1]):
            labels = lst_labels[:,i]
            probas = lst_probas[:,i]
            auc_score=roc_auc_score(labels,probas)
            auc_scores.append(auc_score)
    return auc_scores,lst_labels,lst_probas

def main():
    #Get hyperparameters 
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    models_names=["NormalDataset","NoLungDataset_0","OnlyLungDataset_0","NoLungBBDataset_0","OnlyLungBBDataset_0"]
    
    with open("./data/interim/cxr14_results.csv", "w") as csv_file:
        csv_file.write("training_set,class,fold,auc")
    for model_name in models_names:
        print(model_name)
        for i in range(NB_FOLDS):        
            print(f"\n {model_name} FOLD {i}")            
            
            #Define model
            weights = {
                "name":model_name,
                "fold":i
            }
            model = get_model(CLASSES,weights)

            data_info = pd.read_csv("./data/processed/CXR14/processed_labels.csv")
            data_info["Onehot"] = data_info["Onehot"].apply(lambda x: ast.literal_eval(x))
            auc_scores,lst_labels,lst_probas = eval_model(model,data_info)
            
            with open(f"./data/interim/cxr14_probas_{model_name}_Fold{i}.csv", "a") as csv_file:
                csv_file.write(f"label,proba")
                for label,proba in zip(lst_labels,lst_probas):
                    csv_file.write(f"\n{label},{proba}")
            with open("./data/interim/cxr14_results.csv", "a") as csv_file:
                for j,c in enumerate(CLASSES):
                    csv_file.write(f"\n{model_name},{c},{i},{auc_scores[j]}")
                    print(c,auc_scores[j])
        
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
