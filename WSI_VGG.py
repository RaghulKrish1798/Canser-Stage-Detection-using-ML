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

def VGG16_feature_extractor(data):
    # Downloading the pretrained VGG-16 Model
    model = models.vgg16(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(224),  # resize the image to 224x224
        transforms.ToTensor(),   # convert the image to a PyTorch tensor
    ])
    
    data = [transform(image) for image in data]
    data = torch.stack(data)
    # data = data.unsqueeze(0)
    print(data.shape)
    model.eval()
    with torch.no_grad():
        forward = model(data)
        print(forward.shape)
    return forward

def K_means_clustering(feature):
    feature = np.array(feature)
    k = 10
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(feature)
    cluster_assignments = kmeans.labels_
    print(cluster_assignments.shape)
    print(cluster_assignments)
    return cluster_assignments

def load_images():
    # Loading the Patches
    data_path = "T:\CV\EXtracted_Patches\Test"
    img_name = os.listdir(data_path)
    images = []
    for file_name in img_name:
        image = Image.open(os.path.join(data_path, file_name))
        images.append(image)
    return images

patches = load_images() # Loading the Extracted Patches
features = VGG16_feature_extractor(patches) # Extracting Features
cluster = K_means_clustering(features)  # Performing K-means Clustering on the extracted features