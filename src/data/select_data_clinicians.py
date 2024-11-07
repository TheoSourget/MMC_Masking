from operator import index
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import os.path

import torch
import torch.optim as optim
from torchvision.models import densenet121
from torchvision.transforms import v2
from torch.nn.functional import sigmoid


from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from src.data.pytorch_dataset import MaskingDataset
import pandas as pd
from torchvision.utils import save_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_image(model,valid_dataloader):
    model.to(DEVICE)
    model.eval()
    lst_labels = []
    lst_probas = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
            outputs = model(inputs)
            output_sigmoid = sigmoid(outputs)
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())
            
        lst_labels = np.array(lst_labels)
        lst_probas = np.array(lst_probas)

    return lst_labels,lst_probas

def main():
    #Get hyperparameters 
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    models_names=["NormalDataset","NoLungDataset_0","OnlyLungDataset_0","NoLungBBDataset_0","OnlyLungBBDataset_0"]
        
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
        
    valid_params={
        "NormalDataset":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        "NoLungDataset_0":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        "NoLungBBDataset_0":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        "OnlyLungDataset_0":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        "OnlyLungBBDataset_0":{"masking_spread":0,"inverse_roi":True,"bounding_box":True}
    }

    
    for model_name in models_names:
            print(model_name)               
            testing_data = MaskingDataset(data_dir="./data/processed",**valid_params[model_name])
            testing_data.img_labels = testing_data.img_labels.iloc[test_idx].reset_index(drop=True)
            testing_data.img_paths = np.array(testing_data.img_paths)[test_idx]
            testing_data.roi_paths = np.array(testing_data.roi_paths)[test_idx]
            valid_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE)

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
                model.load_state_dict(torch.load(f"./models/{model_name}/{model_name}_Fold1.pt"))
                model.to(DEVICE)
            except FileNotFoundError as e:
                print("No model saved for fold")
                continue
            
            img_paths = valid_dataloader.dataset.img_paths
            lst_labels,lst_probas = select_image(model,valid_dataloader)
            for i in range(lst_labels.shape[1]):
                lst_img_names = []
                labels = lst_labels[:,i]
                probas = lst_probas[:,i]
                dict_val = {"imageID":img_paths,"labels":labels,"probas":probas,}
                df = pd.DataFrame(dict_val).sort_values(by=['probas'])
                # df.to_csv(f"./preds_{model_name}_{CLASSES[i]}.csv")
                df_positives = df[df["labels"]==1]
                idx_25 = int(len(df_positives) * 25/100)
                idx_50 = int(len(df_positives) * 50/100)
                idx_75 = int(len(df_positives) * 75/100)
                imgs_to_save = df_positives.iloc[[idx_25,idx_50,idx_75]][["imageID","probas"]]
                imgs_to_save["labels"]=CLASSES[i]
                imgs_to_save["masking"]=model_name
                for img_path in imgs_to_save["imageID"]:
                    img, label = valid_dataloader.dataset.get_image_by_id(img_path)
                    img_name = f"{img_path.split('/')[-1].removesuffix('.png')}_{model_name}.png"
                    lst_img_names.append(img_name)
                    save_image(img, f"./data/interim/selected_images_test/{img_name}")
                imgs_to_save["imageID"] = lst_img_names
                imgs_to_save.to_csv("./data/interim/selected_images_test/selected_images.csv",mode='a',header=not os.path.exists("./data/interim/selected_images_test/selected_images.csv"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
