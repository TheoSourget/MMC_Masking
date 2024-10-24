from sympy import im
import torch
from torchvision.models import densenet121
from torchvision.models.feature_extraction import create_feature_extractor
import tensorflow as tf
import numpy as np
import skimage.io as io
from torch.nn.functional import sigmoid

def get_model(classes,weights=None,inplace_relu=True,return_nodes=None):
        
        """
        Instanciate the model (densenet121) used in our study
        @param:
            -CLASSES: array containing the CLASSES to classify (useful for the dimension of last layer)
            -weights: dict of form {"name":model_name,"fold":fold_number} use to load the weights after training. Default: None
            -inplace_relu: bool indicating if relu operation is done inplace or not, should be False to use SHAP. Default: True
            -return_nodes: dict like {"classifier.0": "flatten"} use to return output from specific layer of the model. Default: None
        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Load architecture with default weights (pretrained on imagenet_1k)
        model = densenet121(weights='DEFAULT')
        
        # Freeze every layer except last denseblock and classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features.denseblock4.denselayer16.parameters():
            param.requires_grad = True
       
        kernel_count = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
         torch.nn.Flatten(),
         torch.nn.Linear(kernel_count, len(classes))
        )

        #Put relu inplace to false, useful for SHAP
        if not inplace_relu:
            for module in model.modules():
                if isinstance(module, torch.nn.ReLU):
                    module.inplace = False

        #Load weights after training 
        if weights:
            try:
                model.load_state_dict(torch.load(f"./models/{weights['name']}/{weights['name']}_Fold{weights['fold']}.pt"))
            except FileNotFoundError as e:
                print("Error loading model")
                return None

        #Return output from a particular layer instead of the output from classfier head.        
        if return_nodes:
            model = create_feature_extractor(model,return_nodes)

        model.to(DEVICE)
        return model


def make_single_pred(model,image_path):
        """
        Function use to make prediction without Pytorch Dataloader, useful for external dataset
        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = io.imread(image_path)
        if len(img.shape) == 3:
            img = img[:,:,0]             
        
        img = np.expand_dims(img,-1)
        max_value = np.max(img) 
        img = tf.image.resize_with_pad(img, 512, 512)
        img = (img/max_value).numpy()
        img = np.concatenate((img,)*3, axis=-1)
        img = torch.Tensor(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
        outputs = model(img)
        proba = sigmoid(outputs)
        return proba

