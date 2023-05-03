import torchvision.models as models
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import gc

# https://github.com/uta-smile/DeepAttnMISL/blob/master/DeepAttnMISL_model.py
device = torch.device('cuda')
gc.collect()
torch.cuda.empty_cache()
def VGG16_feature_extractor(data):
    # Downloading the pretrained VGG-16 Model
    model = models.vgg16(pretrained=True).to(device)
    transform = transforms.Compose([
        transforms.Resize(224),  # resize the image to 224x224
        transforms.ToTensor(),   # convert the image to a PyTorch tensor
    ])
    
    data = [transform(image) for image in data]
    t_1, t_2 = train_test_split(data, test_size=0.5, random_state=42, shuffle=True)
    temp_1, temp_2 = train_test_split(t_1, test_size=0.5, random_state=42, shuffle=True)
    temp_3, temp_4 = train_test_split(t_2, test_size=0.5, random_state=42, shuffle=True)
    
    temp_1 = torch.stack(temp_1)
    temp_2 = torch.stack(temp_2)
    temp_3 = torch.stack(temp_3)
    temp_4 = torch.stack(temp_4)
    model.eval()

    with torch.no_grad():
        temp_1 = temp_1.to(device)
        f_1 = model(temp_1)
        del temp_1
        
        temp_2 = temp_2.to(device)
        f_2 = model(temp_2)
        del temp_2
        
        temp_3 = temp_3.to(device)
        f_3 = model(temp_3)
        del temp_3
        
        temp_4 = temp_4.to(device)
        f_4 = model(temp_4)
        del temp_4
        
        forward = torch.cat((f_1, f_2, f_3, f_4), dim = 0)
    return forward

def load_images(data_path):
    # Loading the Patches
    img_name = os.listdir(data_path)
    images = []
    for file_name in img_name:
        image = Image.open(os.path.join(data_path, file_name))
        images.append(image)
    return images

folder_path = "/kaggle/input/train-data"

folders = next(os.walk(folder_path))[1]
# print(folders)
def clear_gpu_mem():
    gc.collect()
    torch.cuda.empty_cache()

for file_name in folders:
    data_path = f"/kaggle/input/train-data/{file_name}"
    patches = load_images(data_path) # Loading the Extracted Patches
    features = VGG16_feature_extractor(patches) # Extracting Features
    print(features.shape)
    features_df = pd.DataFrame(features.cpu().numpy())
    features_df.to_csv(f"{file_name}.csv")
    print(file_name)
    clear_gpu_mem()