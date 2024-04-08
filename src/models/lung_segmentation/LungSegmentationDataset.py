from torch.utils.data import Dataset
import skimage.io as io
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np

class LungSegmentationDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        img_present = [p.split("/")[-1].removesuffix(".png") for p in sorted(glob.glob("./data/imgs/*"))]
        gt_present = [p.split("/")[-1].removesuffix("_mask.png") for p in sorted(glob.glob("./data/masks/*"))]
        self.data = sorted(list(set(img_present) & set(gt_present)))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        image = io.imread(f"./data/imgs/{img_name}.png")[:,:,0]
        gt_mask = io.imread(f"./data/masks/{img_name}_mask.png")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt_mask = self.target_transform(gt_mask)
        return image,gt_mask,f"./data/imgs/{img_name}.png"