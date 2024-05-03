import os
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import glob
import ast
import numpy as np
import cv2

class MaskingDataset(Dataset):
    def __init__(self, data_dir, masking_spread=None, inverse_roi=False, transform=None):
        self.img_paths = glob.glob(f'{data_dir.removesuffix("/")}/images/*.png')
        self.roi_paths = glob.glob(f'{data_dir.removesuffix("/")}/rois/*.png')
        self.img_labels = pd.read_csv(f'{data_dir.removesuffix("/")}/processed_labels.csv',index_col=0)
        self.img_labels = self.img_labels[self.img_labels["ImageID"].isin([p.split("/")[-1] for p in glob.glob(f'{data_dir.removesuffix("/")}/images/*.png')])]
        self.img_labels["Onehot"] = self.img_labels["Onehot"].apply(lambda x: ast.literal_eval(x))
        
        self.img_paths = [f"{data_dir.removesuffix('/')}/images/{img_id}" for img_id in self.img_labels["ImageID"]]
        self.roi_paths = [f"{data_dir.removesuffix('/')}/rois/{img_id}" for img_id in self.img_labels["ImageID"]]

        self.masking_spread = masking_spread
        self.inverse_roi = inverse_roi
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        roi_path = self.roi_paths[idx]
        image = read_image(img_path,ImageReadMode.RGB)
        image = image / image.max()
        if self.masking_spread != None:
            roi = plt.imread(roi_path).astype(np.uint8)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            if self.masking_spread < 0:
                roi = cv2.erode(roi,kernel,iterations=abs(self.masking_spread))   
            elif self.masking_spread > 0:
                roi = cv2.dilate(roi,kernel,iterations=abs(self.masking_spread))
            roi = torch.Tensor(roi).bool()
            if self.inverse_roi:
                image *= roi
            else:
                image *= ~roi

        label = self.img_labels.iloc[idx]["Onehot"]
        if self.transform:
            image = self.transform(image)
        return torch.Tensor(image), torch.Tensor(label)