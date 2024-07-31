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
from torchvision.models import resnet50,densenet121
from torchvision.transforms import v2
from torch.nn.functional import sigmoid
from sklearn.metrics import roc_auc_score,f1_score

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from src.data.pytorch_dataset import MaskingDataset

torch.manual_seed(1907)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training_epoch(model,criterion,optimizer,train_dataloader):
    model.to(DEVICE)
    model.train()
    train_loss = 0.0
    lst_labels = []
    lst_preds = []
    lst_probas = []
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        output_sigmoid = sigmoid(outputs)
        
        lst_labels.extend(labels.cpu().detach().numpy())
        lst_probas.extend(output_sigmoid.cpu().detach().numpy())
        lst_preds.extend(output_sigmoid.cpu().detach().numpy()>0.5)
        
    lst_labels = np.array(lst_labels)
    lst_preds = np.array(lst_preds)
    lst_probas = np.array(lst_probas)
    auc_scores=roc_auc_score(lst_labels,lst_probas,average=None)
    print(f"train ({len(lst_labels)} images)",auc_scores,flush=True)

    return train_loss/lst_labels.shape[0],auc_scores

def valid_epoch(model,criterion,valid_dataloader):
    model.to(DEVICE)
    model.eval()
    val_loss = 0.0
    lst_labels = []
    lst_preds = []
    lst_probas = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            output_sigmoid = sigmoid(outputs)
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())
            lst_preds.extend(output_sigmoid.cpu().detach().numpy()>0.5)

        lst_labels = np.array(lst_labels)
        lst_preds = np.array(lst_preds)
        auc_scores=roc_auc_score(lst_labels,lst_probas,average=None)
        print(f"val ({len(lst_labels)} images)",auc_scores,flush=True)
    return val_loss/lst_labels.shape[0],auc_scores

def main():
    #Get hyperparameters 
    NB_EPOCHS = int(os.environ.get("NB_EPOCHS"))
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    MODEL_NAME = os.environ.get("MODEL_NAME")
    
    ES_PATIENCE = int(os.environ.get("ES_PATIENCE"))
    ES_DELTA = float(os.environ.get("ES_DELTA"))
    
    masking_spread = 0
    inverse_roi = False
    bounding_box = True

    base_run_name = f'runs/{datetime.now().strftime("%b_%d_%Y_%H%M%S")}'

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
    
    #Define data augmentation
    transforms = v2.Compose([
        v2.RandomRotation(degrees=45),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=(0.7,1.1))
    ])

    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)
    
    for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):
        writer = SummaryWriter(f'{base_run_name}/Fold{i}')
        train_data = MaskingDataset(data_dir="./data/processed",transform=transforms,masking_spread=masking_spread,inverse_roi=inverse_roi,bounding_box=bounding_box)
        train_data.img_labels = training_data.img_labels.iloc[train_index].reset_index(drop=True)
        train_data.img_paths = np.array(training_data.img_paths)[train_index]
        train_data.roi_paths = np.array(training_data.roi_paths)[train_index]
        
        val_data = MaskingDataset(data_dir="./data/processed",masking_spread=masking_spread,inverse_roi=inverse_roi,bounding_box=bounding_box)
        val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
        val_data.img_paths = np.array(training_data.img_paths)[val_index]
        val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
        valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
        
        
        #Compute class weights to counter data imbalenced
        y = np.array(train_data.img_labels["Onehot"].tolist())
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
        
        model.to(DEVICE)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        # weights = torch.as_tensor(np.sum(y==0,axis=0)/np.sum(y,axis=0),dtype=torch.float).to(DEVICE)
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        criterion.requires_grad = True
        
        optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)


        earlystopping_count = 0
        best_loss = np.inf
        #Training epoch process
        for epoch in tqdm(range(NB_EPOCHS)):
            # if i == 0 and epoch == 0:
            #     tracker = CarbonTracker(epochs=NB_EPOCHS*NB_FOLDS,stop_and_confirm=False,components="gpu")
            #     tracker.epoch_start()
           
            #Training
            train_loss,train_metric = training_epoch(model,criterion,optimizer,train_dataloader)
            
            #Validation
            val_loss,val_metric = valid_epoch(model,criterion,valid_dataloader)
            
            #Logging + saving model
            writer.add_scalar('Training loss (CE)',
                            train_loss,
                            epoch)
            writer.add_scalar('Validation loss (CE)',
                            val_loss,
                            epoch)


            for j,c in enumerate(CLASSES):
                writer.add_scalar(f'Train AUC Scores {c}',
                                    train_metric[j],
                                    epoch)
                                    
            for j,c in enumerate(CLASSES):
                writer.add_scalar(f'Validation AUC Scores {c}',
                                    val_metric[j],
                                    epoch)



            print(f"\nTraining Loss: {train_loss} \tValid Loss: {val_loss}",flush=True)
            if val_loss < best_loss:
                print(f"Model saved epoch {epoch}")
                torch.save(model.state_dict(),f'./models/{MODEL_NAME}_Fold{i}.pt')
                best_loss = val_loss
                earlystopping_count = 0
            elif val_loss + ES_DELTA > best_loss :
                earlystopping_count += 1
            else:
                earlystopping_count = 0

            if earlystopping_count >= ES_PATIENCE:
                print(f"Early Stopping at epoch {epoch}")
                break
            # if i==0:
            #     tracker.epoch_end()
            #     tracker.stop()
        writer.close() 
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
