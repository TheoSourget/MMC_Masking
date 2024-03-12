# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import glob
import shutil
import os
import numpy as np
import torch
import skimage.io as io
from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
from src.models.lung_unet import PretrainedUNet


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    os.makedirs(os.path.dirname(f"./{output_filepath}/images/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./{output_filepath}/rois/"), exist_ok=True)
    assert os.path.isdir(f"./{output_filepath}/images")
    assert os.path.isdir(f"./{output_filepath}/rois")

    logger.info(f'Creating image dataset in ./{output_filepath}/images')
    create_images(input_filepath,output_filepath)
    logger.info(f'Creating ROIs of images in ./{output_filepath}/rois')
    create_rois(output_filepath)
    assert os.listdir(f"./{output_filepath}/images")
    assert os.listdir(f"./{output_filepath}/rois")
    
    logger.info(f'Dataset is ready to be used!')


def create_images(input_filepath,output_filepath): 
    #Load labels
    labels = pd.read_csv(f'{input_filepath}/labels.csv')
    #Filter images to remove lateral views
    images_filtered = labels[labels["Projection"] != "L"]
    #Get images present at input_filepath
    images_path = glob.glob(f"./{input_filepath}/**/*.png",recursive=True)
    image_names = [path.split('/')[-1] for path in images_path]
    for idx,i_name in enumerate(tqdm(image_names)):
        if i_name in images_filtered["ImageID"].unique():
            shutil.copyfile(images_path[idx], f"./{output_filepath}/images/{i_name}")


def create_rois(output_filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Load lungs segmentation model
    lung_segmentation_model = PretrainedUNet(
        in_channels=1,
        out_channels=2, 
        batch_norm=True, 
        upscale_mode="bilinear"
    )
    lung_segmentation_model.load_state_dict(torch.load("./models/unet-6v.pt"))
    lung_segmentation_model.to(device)
    lung_segmentation_model.eval()
    
    #Apply segmentation on files created during create_images function
    for img_path in tqdm(glob.glob(f"./{output_filepath}/images/*")):
        #Load image and resize to match segmentation model input
        img = io.imread(img_path)
        img = resize(img,(512,512))

        #Transform into tensor
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        #Apply model on image and get mask
        outs = lung_segmentation_model(img_tensor)
        softmax = torch.nn.functional.log_softmax(outs, dim=1)
        seg_map = torch.argmax(softmax, dim=1).cpu()[0].numpy().astype(np.uint8)
        
        #Convert Tensor to image and save
        Image.fromarray(seg_map).save(f"./{output_filepath}/rois/{img_path.split('/')[-1]}")
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
