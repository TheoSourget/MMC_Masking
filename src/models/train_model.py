from operator import index
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import os.path
from tqdm import tqdm
from datetime import datetime
from carbontracker.tracker import CarbonTracker

import torch
import torch.optim as optim
from torchvision.models import resnet50
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score,f1_score

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from src.data.pytorch_dataset import MaskingDataset

def training_epoch(model,criterion,optimizer,train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    train_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs,labels = inputs.float().to(device), torch.Tensor(np.array(labels).T).float().to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        output_softmax = softmax(outputs)
        if i == 0:
            print(softmax(outputs)>0.5,labels)
            print(f1_score(labels.cpu().detach().numpy(),(output_softmax.cpu().detach().numpy()>0.5),average=None))
    return train_loss

def valid_epoch(model,valid_data):
    return 0,0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Get hyperparameters 
    NB_EPOCHS = int(os.environ.get("NB_EPOCHS"))
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
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
    

    #Define model, loss and optimizer
    model = resnet50(weights='DEFAULT') #Weights pretrained on imagenet_1k
    kernel_count = model.fc.in_features
    model.fc = torch.nn.Linear(kernel_count, len(CLASSES))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)
    for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):
        train_data = MaskingDataset(data_dir="./data/processed")
        train_data.img_labels = training_data.img_labels.iloc[train_index].reset_index(drop=True)
        train_data.img_paths = np.array(training_data.img_paths)[train_index]
        train_data.roi_paths = np.array(training_data.roi_paths)[train_index]
        
        val_data = MaskingDataset(data_dir="./data/processed")
        val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
        val_data.img_paths = np.array(training_data.img_paths)[val_index]
        val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
        valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

        #Training epoch process
        for epoch in tqdm(range(NB_EPOCHS)):
            #Training
            loss = training_epoch(model,criterion,optimizer,train_dataloader)
            print(loss)
            #Validation
            loss,metric = valid_epoch(model,valid_dataloader)
            
            #Logging + saving model
            pass

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
