from model.Unet import Unet
import torch
torch.manual_seed(1907)
import torch.nn as nn
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Unet(1,1,light=True)
    net.load_state_dict(torch.load('./weights/Unet.pt'))
    net.eval()
    
    assert os.path.isfile("./data/dataloader/valid_dataloader.pth")
    valid_dataloader = torch.load('./data/dataloader/valid_dataloader.pth')

    idxImgTest = 0
    inputs, labels, img_name = valid_dataloader.dataset[idxImgTest]
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)
    print(img_name)
    new_labels = torch.squeeze(labels)
    
    outputs = net(inputs)
    outputs = torch.sigmoid(outputs)
    pred = (outputs > 0.5)[0]
  
    plt.figure(figsize=(20,20))
    plt.subplot(131)
    plt.title(img_name.split("/")[-1])
    plt.axis("off")
    plt.imshow(inputs[0][0],cmap="gray")

    plt.subplot(132)
    plt.title("Ground Truth")
    plt.imshow(new_labels.detach().cpu().numpy())
    plt.axis("off")

    plt.subplot(133)
    plt.title("Model Segmentation")
    plt.imshow(pred.detach().cpu().numpy()[0],vmax=3,vmin=0)
    plt.axis("off")

    plt.show()