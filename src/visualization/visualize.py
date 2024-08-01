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
    pass

def main():
    generate_auc_per_label()
    generate_explainability_map()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
