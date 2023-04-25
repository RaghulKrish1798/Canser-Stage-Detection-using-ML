import torchvision.models as models
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# https://github.com/uta-smile/DeepAttnMISL/blob/master/DeepAttnMISL_model.py
device = torch.device('cuda')
torch.cuda.empty_cache()
def VGG16_feature_extractor(data):
    # Downloading the pretrained VGG-16 Model
    model = models.vgg16(pretrained=True).to(device)
    transform = transforms.Compose([
        transforms.Resize(224),  # resize the image to 224x224
        transforms.ToTensor(),   # convert the image to a PyTorch tensor
    ])
    
    data = [transform(image) for image in data]
    # Splitting Data into two to accomodate the GPU memeory
    temp_1, temp_2 = train_test_split(data, test_size=0.5, random_state=42, shuffle=True)
    temp_1 = torch.stack(temp_1)
    temp_2 = torch.stack(temp_2)
    model.eval() # Making model to evaluation mode

    with torch.no_grad():
        # Passing the data through the model in two halves
        temp_1 = temp_1.to(device)
        f_1 = model(temp_1)
        del temp_1

        temp_2 = temp_2.to(device)
        f_2 = model(temp_2)
        del temp_2
        forward = torch.cat((f_1, f_2), dim = 0) # Saving all the extracted features into one variable

    return forward


# def K_means_clustering(feature):
#     feature = np.array(feature)
#     k = 10
#     kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(feature)
#     cluster_assignments = kmeans.labels_
#     print(cluster_assignments.shape)
#     print(cluster_assignments)
#     return cluster_assignments


def load_images():
    # Loading the Patches
    data_path = "/kaggle/input/test-data/TCGA-A1-A0SF-01A"
    img_name = os.listdir(data_path)
    images = []
    for file_name in img_name:
        image = Image.open(os.path.join(data_path, file_name))
        images.append(image)
    return images

patches = load_images() # Loading the Extracted Patches
features = VGG16_feature_extractor(patches) # Extracting Features
print(features.shape)
features_df = pd.DataFrame(features.cpu().numpy())
features_df.to_csv("TCGA-A1-A0SF-01A.csv")

# cluster = K_means_clustering(features)  # Performing K-means Clustering on the extracted features