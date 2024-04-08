from model.Unet import Unet

from tqdm import tqdm
import pandas as pd
import torch
torch.manual_seed(1907)
import torch.nn as nn

from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.model_selection import train_test_split

import numpy as np
from LungSegmentationDataset import LungSegmentationDataset
from carbontracker.tracker import CarbonTracker
from torch.utils.tensorboard import SummaryWriter
import os.path

def dice_coeff(input,target,reduce_batch_first=False,epsilon=1e-6):
    # Calculate the Dice score of every image in a batch, from: https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/lung_segmentation_unet')
    NB_EPOCHS = 2
    VALID_SIZE = 0.2
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    #Use a "light" version if True (3.7M params) or the paper version if False (31M params)
    lightUnet = True

    #Load camus dataset
    train_data =LungSegmentationDataset(
        transform=Compose([ToPILImage(),Resize((512,512)),ToTensor()]),
        target_transform=Compose([ToPILImage(),Resize((512,512)),ToTensor()]),
    )

    valid_data =LungSegmentationDataset(
        transform=Compose([ToPILImage(),Resize((512,512)),ToTensor()]),
        target_transform=Compose([ToPILImage(),Resize((512,512)),ToTensor()]),
    )
    
    #Split with validation set
    train_indices, val_indices = train_test_split(np.arange(0,len(train_data),1),test_size=VALID_SIZE,random_state=1907)
    valid_data =torch.utils.data.Subset(train_data,val_indices)
    train_data = torch.utils.data.Subset(train_data,train_indices)
    
    #Turn the dataset into DataLoader
    if os.path.isfile("./data/dataloader/train_dataloader.pth") and os.path.isfile("./data/dataloader/valid_dataloader.pth"):
        train_dataloader = torch.load('./data/dataloader/train_dataloader.pth')
        valid_dataloader = torch.load('./data/dataloader/valid_dataloader.pth')
    else:
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
        valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE)
        torch.save(train_dataloader,"./data/dataloader/train_dataloader.pth")
        torch.save(valid_dataloader,"./data/dataloader/valid_dataloader.pth")

    net = Unet(1,1,light=lightUnet).to(device)
    net.load_state_dict(torch.load('./weights/Unet.pt'))
    optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE)
    
    criterion = nn.BCEWithLogitsLoss()
    criterion.requires_grad = True
    lossEvolve = []
    valEvolve = []
    diceEvolve = []
    
    tracker = CarbonTracker(epochs=NB_EPOCHS,stop_and_confirm=False,components="gpu")
    
    lst_images = []
    for j, data in enumerate(valid_dataloader, 0):
        inputs, labels, img_names = data
        lst_images += img_names
    df_res_val = pd.DataFrame(lst_images,columns=["img_name"])

    for epoch in tqdm(range(NB_EPOCHS)): 
        print("################# EPOCH:",epoch+1,"#################")
        tracker.epoch_start()

        
        #Train
        net.train()
        train_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, img_names = data
            inputs,labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float().to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        #Validation
        net.eval()
        val_loss = 0.0
        dice_curr = 0.0
        lst_dice = []
        lst_imgnames = []
        with torch.no_grad():
            for j, data in enumerate(valid_dataloader, 0):
                inputs, labels, img_names = data
                inputs,labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels.float().to(device))
                val_loss += loss.item()

                softmax = torch.sigmoid(outputs)
                pred = softmax>=0.5
                lst_dice.append(dice_coeff(pred,labels.int()))
                lst_imgnames.append(img_names)
            all_dice = torch.cat(lst_dice)
            df_res_val[f"dice_epoch_{epoch}"] = all_dice.cpu()

        lossEvolve.append(train_loss/(i+1))
        valEvolve.append(val_loss/(j+1))
        diceEvolve.append(all_dice.mean())
        print(f"Training Loss: {train_loss/(i+1)} \tValid Loss: {val_loss/(j+1)} \tDice: {all_dice.mean()}+/-{all_dice.std()}")
        writer.add_scalar('Training loss (BCE)',
                            train_loss/(i+1),
                            epoch)
        writer.add_scalar('Validation loss (BCE)',
                            val_loss/(j+1),
                            epoch)
        
        writer.add_scalar('Validation Dice Score',
                            all_dice.mean(),
                            epoch)

        torch.save(net.state_dict(),'./weights/Unet2.pt')
        df_res_val.to_csv("./results/validation_dice.csv")
        tracker.epoch_end()
    tracker.stop()
    writer.close()
