import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os.path


import torch
from torch.nn.functional import sigmoid

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from src.data.pytorch_dataset import MaskingDataset
from src.models.utils import get_model,make_single_pred
from src.data.utils import get_splits

import shap
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_auc_per_label(path_to_results="./data/interim/valid_results.csv",base_img_name="mean_auc",show=False):
    models_valid_results = pd.read_csv(path_to_results)
    mean_auc_per_class = models_valid_results.groupby(["class","training_set","valid_set"])["auc"].mean()
    for class_label in mean_auc_per_class.index.get_level_values('class').unique():
        result_class = mean_auc_per_class[mean_auc_per_class.index.get_level_values('class').isin([class_label])].droplevel(0)
        result_class = result_class.reset_index().pivot(columns='valid_set',index='training_set',values='auc')
        result_class = result_class[["NoLungBB","NoLung","Normal","OnlyLungBB","OnlyLung"]]
        plt.figure(figsize=(9,7))
        plt.title(class_label,size=20)
        heatmap = sns.heatmap(result_class, annot=True,cmap="RdYlGn",annot_kws={"size": 15},xticklabels=["NoLungsBB","NoLungs","Full","OnlyLungsBB","OnlyLungs"],yticklabels=["NoLungsBB","NoLungs","Full","OnlyLungsBB","OnlyLungs"])
        heatmap.yaxis.set_tick_params(labelsize = 14,rotation=45)
        heatmap.xaxis.set_tick_params(labelsize = 14)
        plt.xlabel('Test set masking', fontsize=16)
        plt.ylabel('Training set masking', fontsize=16)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        plt.tight_layout()
        plt.savefig(f"./reports/figures/{base_img_name}_{class_label}.png",format='png')
        if show:
            plt.show()

def generate_explainability_map(show=False):
    #Get hyperparameters 
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    model_name="NormalDataset"

    training_data,testing_data = get_splits(NB_FOLDS)
    

    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)
    
    valid_params={
        "Normal":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        # "NoLung":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        # "NoLungBB":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        # "OnlyLung":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        # "OnlyLungBB":{"masking_spread":0,"inverse_roi":True,"bounding_box":True}
    }

    for param_config_name in valid_params:
        print(model_name,param_config_name)
        for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):        
            print("\nFOLD",i)
            val_data = MaskingDataset(data_dir="./data/processed",**valid_params[param_config_name])
            val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
            val_data.img_paths = np.array(training_data.img_paths)[val_index]
            val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

            valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
                        
            weights = {
                "name":model_name,
                "fold":i
            }

            model = get_model(CLASSES,weights)
            model.eval()

            images, _ = next(iter(valid_dataloader))
            images = images.to(DEVICE)
            background = images[:1]
            test_images= images[1:]
            e = shap.DeepExplainer(model, images)
            shap_values = e.shap_values(test_images)

def get_embedding(model_name,valid_params):
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    CLASSES = os.environ.get("CLASSES").split(",")
    
    
    #Get train and test splits
    
    training_data,testing_data = get_splits(NB_FOLDS)
    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)
    
    
    models_flatten_output = {
        masking_param:[] for masking_param in valid_params
    }
    
    for masking_param in valid_params:
        print(f"\n{masking_param}")
        for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):        
            val_data = MaskingDataset(data_dir="./data/processed",**valid_params[masking_param])
            val_data.img_labels = training_data.img_labels.iloc[val_index].reset_index(drop=True)
            val_data.img_paths = np.array(training_data.img_paths)[val_index]
            val_data.roi_paths = np.array(training_data.roi_paths)[val_index]

            valid_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
            
            weights = {
                "name":model_name,
                "fold":i
            }

            return_nodes = {
                "classifier.0": "flatten"
            }
            model = get_model(CLASSES,weights,return_nodes=return_nodes)
            model.eval()

            with torch.no_grad():
                labels_dataset = []
                for i, data in enumerate(valid_dataloader, 0):
                    inputs, labels = data
                    inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                    outputs = model(inputs)
                    models_flatten_output[masking_param].extend(outputs["flatten"].detach().cpu().tolist())
                    labels_dataset += labels
            models_flatten_output[masking_param] = np.array(models_flatten_output[masking_param])
            break
    return models_flatten_output, labels_dataset

def scatter_hist(x, y, ax, ax_histx, ax_histy,label,visualisation_param):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y,alpha=0.5,label=label,**visualisation_param)

    bins = np.arange(0, 1, 0.05)
    ax_histx.hist(x, bins=bins, color=visualisation_param["c"], alpha = 0.5)
    ax_histy.hist(y, bins=bins, color=visualisation_param["c"], alpha = 0.5, orientation='horizontal')
    
    ax_histx.axis('off')
    ax_histy.axis('off')

def plot_tsne(show=False):
    model_name="NormalDataset"
      
    valid_params={
        "Full":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        "NoLung":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        "NoLungBB":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        "OnlyLung":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        "OnlyLungBB":{"masking_spread":0,"inverse_roi":True,"bounding_box":True},
    }
    
    visualisation_param = {
        "Full":{"c":"tab:blue","marker":"o"},
        "NoLung":{"c":"tab:orange","marker":"^"},
        "NoLungBB":{"c":"tab:green","marker":"s"},
        "OnlyLung":{"c":"purple","marker":"*"},
        "OnlyLungBB":{"c":"red","marker":"X"}
    }
    
    models_embeddings,labels_dataset = get_embedding(model_name,valid_params)
    
    #We convert the dict from get_embedding to regroup them all (not group by masking anymore) in an array to perform the t-SNE
    models_flatten_output = []
    
    #This array will keep the info on the masking used to produce this embedding, useful for visualisation later
    labels_masking = []
    for masking in models_embeddings:
        models_flatten_output.extend(models_embeddings[masking])
        labels_masking += [masking] * len(models_embeddings[masking])    
    models_flatten_output = np.array(models_flatten_output)
    
    #Taken from https://learnopencv.com/t-sne-for-feature-visualization/
    tsne = TSNE(n_components=2,perplexity=10).fit_transform(np.array(models_flatten_output))
    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))
     
        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)
     
        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range
     
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
     
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)


    #Plot divided by masking strategy
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    
    for masking_param in valid_params:
        indices = [j for j, l in enumerate(labels_masking) if l == masking_param]
        # extract the coordinates of the points of the current masking
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        scatter_hist(current_tx, current_ty, ax, ax_histx, ax_histy,masking_param,visualisation_param[masking_param])
        # plt.scatter(current_tx,current_ty,label=masking_param,alpha=0.5,**visualisation_param[masking_param])
    
    ax.legend(bbox_to_anchor=(2,1),fontsize=15)
    plt.savefig(f"./reports/figures/tsne.png",format='png',bbox_inches='tight')
    if show:
        plt.show()

def get_cosine():
    model_name="NormalDataset"
    valid_params={
        "NormalDataset":{"masking_spread":None,"inverse_roi":False,"bounding_box":False},
        "NoLungDataset_0":{"masking_spread":0,"inverse_roi":False,"bounding_box":False},
        "OnlyLungDataset_0":{"masking_spread":0,"inverse_roi":True,"bounding_box":False},
        "NoLungDatasetBB_0":{"masking_spread":0,"inverse_roi":False,"bounding_box":True},
        "OnlyLungDatasetBB_0":{"masking_spread":0,"inverse_roi":True,"bounding_box":True},
    }
    CLASSES = os.environ.get("CLASSES").split(",")
    
    models_flatten_output,labels_dataset = get_embedding(model_name,valid_params)

    no_lung_similarities = []
    only_lung_similarities = []
    no_lungbb_similarities = []
    only_lungbb_similarities = []
    for j in range(len(models_flatten_output["NormalDataset"])):
        normal = models_flatten_output["NormalDataset"][j]
        nolung = models_flatten_output["NoLungDataset_0"][j]
        nolungbb = models_flatten_output["NoLungDatasetBB_0"][j]
        onlylung = models_flatten_output["OnlyLungDataset_0"][j]
        onlylungbb = models_flatten_output["OnlyLungDatasetBB_0"][j]

        no_lung_similarities.append(1- cosine(normal,nolung))
        no_lungbb_similarities.append(1- cosine(normal,nolungbb))
        only_lung_similarities.append(1- cosine(normal,onlylung))
        only_lungbb_similarities.append(1- cosine(normal,onlylungbb))
    with open("./data/interim/cosine_similarities.csv","w") as csv_file:
        csv_file.write(f"class,no_lung,no_lungbb,only_lung,only_lungbb\n")
        csv_file.write(f"all,{np.mean(no_lung_similarities)}+/-{np.std(no_lung_similarities)},\
            {np.mean(no_lungbb_similarities)}+/-{np.std(no_lungbb_similarities)},\
            {np.mean(only_lung_similarities)}+/-{np.std(only_lung_similarities)},\
            {np.mean(only_lungbb_similarities)}+/-{np.std(only_lungbb_similarities)}\n")
    print(f"all,{np.mean(no_lung_similarities)}+/-{np.std(no_lung_similarities)},\
            {np.mean(no_lungbb_similarities)}+/-{np.std(no_lungbb_similarities)},\
            {np.mean(only_lung_similarities)}+/-{np.std(only_lung_similarities)},\
            {np.mean(only_lungbb_similarities)}+/-{np.std(only_lungbb_similarities)}")
    #Per class
    for i,c in enumerate(CLASSES):
        no_lung_similarities = []
        only_lung_similarities = []
        no_lungbb_similarities = []
        only_lungbb_similarities = []
        class_indices = [j for j, l in enumerate(labels_dataset) if l[i] == 1 ]
        for j in class_indices:
            normal = models_flatten_output["NormalDataset"][j]
            nolung = models_flatten_output["NoLungDataset_0"][j]
            nolungbb = models_flatten_output["NoLungDatasetBB_0"][j]
            onlylung = models_flatten_output["OnlyLungDataset_0"][j]
            onlylungbb = models_flatten_output["OnlyLungDatasetBB_0"][j]

            no_lung_similarities.append(1- cosine(normal,nolung))
            no_lungbb_similarities.append(1- cosine(normal,nolungbb))
            only_lung_similarities.append(1- cosine(normal,onlylung))
            only_lungbb_similarities.append(1- cosine(normal,onlylungbb))
        with open("./data/interim/cosine_similarities.csv","a") as csv_file:
            csv_file.write(f"{c},{np.mean(no_lung_similarities)}+/-{np.std(no_lung_similarities)},\
                {np.mean(no_lungbb_similarities)}+/-{np.std(no_lungbb_similarities)},\
                {np.mean(only_lung_similarities)}+/-{np.std(only_lung_similarities)},\
                {np.mean(only_lungbb_similarities)}+/-{np.std(only_lungbb_similarities)}\n")
            
        print(f"{c},{np.mean(no_lung_similarities)}+/-{np.std(no_lung_similarities)},\
                {np.mean(no_lungbb_similarities)}+/-{np.std(no_lungbb_similarities)},\
                {np.mean(only_lung_similarities)}+/-{np.std(only_lung_similarities)},\
                {np.mean(only_lungbb_similarities)}+/-{np.std(only_lungbb_similarities)}")


def dilation_impact_auc(model_name,masking_param,dilation_factors):
    """
    Apply validation set and compute AUC for images of the validation sets with increasing dilation factor
    @param:
        -model_name: name of the weights to load
        -masking_param: dict with the masking parameters {"inverse_roi":False,"bounding_box":False}
        -dilation_factors: list of the dilation factors (masking spread) to apply
    @return:
        -The list of AUCs for each dilation factors
    """
    NB_FOLDS = int(os.environ.get("NB_FOLDS"))
    BATCH_SIZE = 1
    CLASSES = os.environ.get("CLASSES").split(",")
    
    #Load the base dataset
    training_data,testing_data = get_splits(NB_FOLDS)
    

    #Create k-fold for train/val
    group_kfold = GroupKFold(n_splits=NB_FOLDS)

    lst_auc = [[] for i in range(len(dilation_factors))]

    for k,dilation_factor in enumerate(dilation_factors):
        print("\nDilation factor",k)
        for i, (train_index,val_index) in enumerate(group_kfold.split(training_data.img_labels, groups= training_data.img_labels['PatientID'])):
            test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE)

            #Define model
            weights = {
                "name":model_name,
                "fold":i
            }
            model = get_model(CLASSES,weights)
            model.eval()
        
            lst_labels = []
            lst_probas = []
            auc_scores = []
            test_dataloader.dataset.masking_spread = dilation_factor
            test_dataloader.dataset.inverse_roi = masking_param["inverse_roi"]
            test_dataloader.dataset.bounding_box = masking_param["bounding_box"]
            
            with torch.no_grad():
                for j,data in enumerate(test_dataloader):
                    inputs,labels = data
                    inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
                    outputs = model(inputs)
                    output_sigmoid = sigmoid(outputs)
                    lst_labels.extend(labels.cpu().detach().numpy())
                    lst_probas.extend(output_sigmoid.cpu().detach().numpy())
                
                lst_labels = np.array(lst_labels)
                lst_probas = np.array(lst_probas)
                for j in range(lst_labels.shape[1]):
                    labels = lst_labels[:,j]
                    probas = lst_probas[:,j]
                    auc_score=roc_auc_score(labels,probas)
                    auc_scores.append(auc_score)
                
                lst_auc[k].append(auc_scores)
    return lst_auc

def plot_impact_auc(model_name,masking_param,savefile_name,show=False):
    """
    @param:
        -model_name: name of the weights to load
        -masking_param: dict with the masking parameters (ex: {"inverse_roi":False,"bounding_box":False})
        -savefile_name: Name of the file used to save the plot
        -show: bool, default False. Save and show the plot if True, only save it otherwise
    """
    CLASSES = os.environ.get("CLASSES").split(",")
    dilation_factors = [0,5,10,25,50,100,150,200,300,400,500]

    lst_auc_all_disease_dilation = dilation_impact_auc(model_name,masking_param,dilation_factors)
    
    plt.figure()
    for c in range(len(CLASSES)):
        lst_auc_disease_dilation = np.array(lst_auc_all_disease_dilation)[:,:,c]
        mean_auc_disease = np.array([np.mean(lst_auc_disease_dilation[k]) for k in range(len(lst_auc_disease_dilation))])
        std_auc_disease = np.array([np.std(lst_auc_disease_dilation[k]) for k in range(len(lst_auc_disease_dilation))])
        plt.plot(dilation_factors,mean_auc_disease,marker="o",label=CLASSES[c])
        plt.fill_between(dilation_factors, mean_auc_disease-std_auc_disease, mean_auc_disease+std_auc_disease,alpha=0.2)


    plt.title("Evolution of AUC while expanding mask's size")
    plt.xlabel("Dilation factor")
    plt.ylabel("Mean AUC across 5-fold")
    plt.legend()

    plt.savefig(f"./reports/figures/{savefile_name}.png",bbox_inches='tight')
    if show:
        plt.show()        

def main():
    # print("GENERATING AUC MATRICES")
    # generate_auc_per_label("./data/interim/test_results.csv","test_mean_auc")
    
    # print("\nCOMPUTING COSINE SIMILARITIES")
    # get_cosine()
    
    # print("\nCREATING t-SNE PLOT")
    # plot_tsne()

    # # print("\nGENERATING EXPLAINABILITY MAPS")
    # # generate_explainability_map()

    
    print("Dilation impact")
    #Impact of dilation for the model trained on Full images and evaluated on images with only the lungs
    #Increasing the dilation factor will INCREASE the proportion of visible pixels in the image
    masking_param = {"inverse_roi":True,"bounding_box":False}
    plot_impact_auc("NormalDataset",masking_param,"auc_dilation_OnlyLungs",False)


    #Impact of dilation for the model trained on Full images and evaluated on images without the lungs
    #Increasing the dilation factor will DECREASE the proportion of visible pixels in the image
    masking_param = {"inverse_roi":False,"bounding_box":False}
    plot_impact_auc("NormalDataset",masking_param,"auc_dilation_NoLungs",False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
