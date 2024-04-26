# -*- coding: utf-8 -*-
from operator import index
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import ast
import glob
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
import torch
from src.models.lung_segmentation.model.Unet import Unet


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('classes', default=None,type=str)
def main(input_filepath, output_filepath, classes):
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    os.makedirs(os.path.dirname(f"./{output_filepath}/images/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./{output_filepath}/rois/"), exist_ok=True)
    assert os.path.isdir(f"./{output_filepath}/images")
    assert os.path.isdir(f"./{output_filepath}/rois")

    logger.info(f'Processing labels and store the result in ./{output_filepath}/processed_labels.csv')
    filter_and_process_labels(input_filepath,output_filepath,classes)
    assert os.path.exists(f"{output_filepath}/processed_labels.csv")
    
    if os.listdir(f"./{output_filepath}/images") == []:
        logger.info(f'Creating image dataset in ./{output_filepath}/images')
        create_images(input_filepath,output_filepath)
        assert os.listdir(f"./{output_filepath}/images")
    else:
        logger.info(f'./{output_filepath}/images not empty, skipping creation of images')
    
    if os.listdir(f"./{output_filepath}/rois") == []:
        logger.info(f'Creating ROIs of images in ./{output_filepath}/rois')
        create_rois(output_filepath)
        assert os.listdir(f"./{output_filepath}/rois")
    else:
        logger.info(f'./{output_filepath}/rois not empty, skipping creation of rois')

    logger.info(f'Dataset is ready to be used!')


def filter_and_process_labels(input_filepath,output_filepath,classes):
    
    base_df = pd.read_csv(f'{input_filepath}/labels.csv',index_col=0)
    invalid_images = pd.read_csv(f'{input_filepath}/Invalid_images.csv', header=None, index_col=0)
    
    # Excluding NaNs in the labels
    df_no_nan = base_df[~base_df["Labels"].isna()]
    # Excluding labels including the 'suboptimal study' label
    df_no_clear_label = df_no_nan[~df_no_nan["Labels"].str.contains('suboptimal study')]
    df_no_clear_label = df_no_clear_label[~df_no_nan["Labels"].str.contains('exclude')]
    df_no_clear_label = df_no_clear_label[~df_no_nan["Labels"].str.contains('Unchanged')]

    # Keeping only the PA, AP and AP_horizontal projections
    df_view = df_no_clear_label[(df_no_clear_label['Projection'] == 'PA') | (df_no_clear_label['Projection'] == 'AP') | (df_no_clear_label['Projection'] == 'AP_horizontal')]

    # Stripping and lowercasing all individual labels
    stripped_lowercased_labels = []

    for label_list in list(df_view['Labels']):
        label_list = ast.literal_eval(label_list)
        prepped_labels = []
        
        for label in label_list:
            if label != '':
                new_label = label.strip(' ').lower()   # Stripping and lowercasing
                prepped_labels.append(new_label)
        
        # Removing label duplicates in this appending
        stripped_lowercased_labels.append(list(set(prepped_labels)))

    # Applying it to the preprocessed dataframe
    df_view['Labels'] = stripped_lowercased_labels
    invalid_images.columns = list(df_view.columns ) +["path"]
    df_no_invalid = df_view[~df_view['ImageID'].isin(invalid_images['ImageID'])]
    if classes:
        accepted_classes = [c.strip(' ').lower() for c in classes.split(",")]
        all_new_labels = []
        all_onehot_labels = []
        for label_list in df_no_invalid['Labels']:
            new_labels = list(set(label_list) & set(accepted_classes))
            if len(new_labels) == 0:
                new_labels = ['no finding']
            all_new_labels.append(new_labels)
            all_onehot_labels.append([1 if l in new_labels else 0 for l in accepted_classes])     
        df_no_invalid['Processed_Labels'] =  all_new_labels
        df_no_invalid['Onehot'] = all_onehot_labels

    df_to_save = df_no_invalid.reset_index(drop=True)
    df_to_save.to_csv(f"{output_filepath}/processed_labels.csv",sep=",")

def create_images(input_filepath,output_filepath): 
    #Load labels
    labels = pd.read_csv(f'{output_filepath}/processed_labels.csv')
    #Filter images to remove lateral views
    
    #Get images present at input_filepath
    images_path = glob.glob(f"./{input_filepath}/**/*.png",recursive=True)
    image_names = [path.split('/')[-1] for path in images_path]
    for idx,i_name in enumerate(tqdm(image_names)):
        #Resize the image and save it in the processed folder
        if i_name in labels["ImageID"].unique():
            img = io.imread(images_path[idx])
            img = resize(img,(512,512))
            img = (img*255).astype(np.uint8)
            io.imsave(f"./{output_filepath}/images/{i_name}",img)
    
def rle2mask(mask_rle: str, label=1, shape=(3520,4280)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)# Needed to align to RLE direction

def decode_both_lungs(row, label=1):

    right = rle2mask(
        mask_rle=row["Right Lung"],
        label=label,
        shape=(int(row["Height"]),int(row["Width"]))
    )

    left = rle2mask(
        mask_rle=row["Left Lung"],
        label=label,
        shape=(int(row["Height"]),int(row["Width"]))
    )

    return right + left

def create_rois(output_filepath):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # #Load lungs segmentation model
    # lung_segmentation_model = Unet(1,1,light=False)
    # lung_segmentation_model.load_state_dict(torch.load("./models/UnetBig.pt"))
    # lung_segmentation_model.to(device)
    # lung_segmentation_model.eval()
    
    # #Apply segmentation on files created during create_images function
    # for img_path in tqdm(glob.glob(f"./{output_filepath}/images/*")):
    #     #Load image and resize to match segmentation model input
    #     img = plt.imread(img_path)
        
    #     #Transform into tensor
    #     img_tensor = torch.from_numpy(img).float()
    #     img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    #     img_tensor = img_tensor.to(device)
        
    #     #Apply model on image and get mask
    #     outs = lung_segmentation_model(img_tensor)
    #     pred = torch.sigmoid(outs) >= 0.5
    #     seg_map = pred.cpu()[0][0].numpy().astype(np.uint8)
        
    #     #Convert Tensor to image and save
    #     Image.fromarray(seg_map).save(f"./{output_filepath}/rois/{img_path.split('/')[-1]}")

    masks_df = pd.read_csv(f"./{output_filepath}/Padchest.csv")
    images_present = [img_path.split('/')[-1] for img_path in glob.glob(f"./{output_filepath}/images/*")]
    mask_present = []
    for idx,row in tqdm(masks_df.dropna().iterrows()):
        if row['ImageID'] not in images_present:
            continue
        mask = decode_both_lungs(row)
        mask_resize = resize(mask,(512,512))
        mask_resize = mask_resize > 0
        mask_img = Image.fromarray(mask_resize)
        mask_img.save(f"./{output_filepath}/rois/{row['ImageID']}")
        mask_present.append(row['ImageID'])
    for imageID in (set(images_present)-set(mask_present)):
        os.remove(f"./{output_filepath}/images/{imageID}")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
