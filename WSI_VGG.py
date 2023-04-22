import torchvision.models as models
import torch
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch import nn
import torchvision
import cv2
import numpy as np
from PIL import Image

# https://github.com/uta-smile/DeepAttnMISL/blob/master/DeepAttnMISL_model.py

# Downloading the pretrained VGG-16 Model
model = models.vgg16(pretrained=True)

# Loading the Patches
data_path = "T:\CV\EXtracted_Patches\TCGA-A1-A0SF-01A\patch_1084.png"

transform = transforms.Compose([
    transforms.Resize(224),  # resize the image to 224x224
    transforms.ToTensor(),   # convert the image to a PyTorch tensor
])

data = Image.open(data_path)
data = transform(data)
data = data.unsqueeze(0)
print(data.shape)
model.eval()
with torch.no_grad():
    forward = model(data)
    print(forward.shape)