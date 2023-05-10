import torchvision.models as models
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
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
    model.eval()
    if len(data) < 25:
        data = torch.stack(data)

        with torch.no_grad():
            data = data.to(device)
            forward = model(data)
    else:
#     temp1_1, t_2 = train_test_split(data, test_size=float(7/8), random_state=42, shuffle=True)
#     temp_1, temp_2 = train_test_split(t_1, test_size=0.5, random_state=42, shuffle=True)
#     t_3, t_4 = train_test_split(t_2, test_size=float(6/7), random_state=42, shuffle=True)
#     temp_3, temp_4 = train_test_split(t_2, test_size=0.5, random_state=42, shuffle=True)

        t_1, t_2 = train_test_split(data, test_size=float(10/11), random_state=42, shuffle=True)
        temp_1, temp_2 = train_test_split(t_1, test_size=0.5, random_state=42, shuffle=True)
        t_3, t_4 = train_test_split(t_2, test_size=float(9/10), random_state=42, shuffle=True)
        temp_3, temp_4 = train_test_split(t_3, test_size=0.5, random_state=42, shuffle=True)
        t_5, t_6 = train_test_split(t_4, test_size=float(8/9), random_state=42, shuffle=True)
        temp_5, temp_6 = train_test_split(t_5, test_size=0.5, random_state=42, shuffle=True)
        t_7, t_8 = train_test_split(t_6, test_size=float(7/8), random_state=42, shuffle=True)
        temp_7, temp_8 = train_test_split(t_7, test_size=0.5, random_state=42, shuffle=True)
        t_9, t_10 = train_test_split(t_8, test_size=float(6/7), random_state=42, shuffle=True)
        temp_9, temp_10 = train_test_split(t_9, test_size=0.5, random_state=42, shuffle=True)
        t_11, t_12 = train_test_split(t_10, test_size=float(5/6), random_state=42, shuffle=True)
        temp_11, temp_12 = train_test_split(t_11, test_size=0.5, random_state=42, shuffle=True)
        t_13, t_14 = train_test_split(t_12, test_size=float(4/5), random_state=42, shuffle=True)
        temp_13, temp_14 = train_test_split(t_13, test_size=0.5, random_state=42, shuffle=True)
        t_15, t_16 = train_test_split(t_14, test_size=float(3/4), random_state=42, shuffle=True)
        temp_15, temp_16 = train_test_split(t_15, test_size=0.5, random_state=42, shuffle=True)
        t_17, t_18 = train_test_split(t_16, test_size=float(2/3), random_state=42, shuffle=True)
        temp_17, temp_18 = train_test_split(t_17, test_size=0.5, random_state=42, shuffle=True)
        t_19, t_20 = train_test_split(t_18, test_size=float(1/2), random_state=42, shuffle=True)
        temp_19, temp_20 = train_test_split(t_19, test_size=0.5, random_state=42, shuffle=True)
        temp_21, temp_22 = train_test_split(t_20, test_size=0.5, random_state=42, shuffle=True)
        
        temp_1 = torch.stack(temp_1)
        temp_2 = torch.stack(temp_2)
        temp_3 = torch.stack(temp_3)
        temp_4 = torch.stack(temp_4)
        temp_5 = torch.stack(temp_5)
        temp_6 = torch.stack(temp_6)
        temp_7 = torch.stack(temp_7)
        temp_8 = torch.stack(temp_8)
        temp_9 = torch.stack(temp_9)
        temp_10 = torch.stack(temp_10)
        temp_11 = torch.stack(temp_11)
        temp_12 = torch.stack(temp_12)
        temp_13 = torch.stack(temp_13)
        temp_14 = torch.stack(temp_14)
        temp_15 = torch.stack(temp_15)
        temp_16 = torch.stack(temp_16)
        temp_17 = torch.stack(temp_17)
        temp_18 = torch.stack(temp_18)
        temp_19 = torch.stack(temp_19)
        temp_20 = torch.stack(temp_20)
        temp_21 = torch.stack(temp_21)
        temp_22 = torch.stack(temp_22)

        with torch.no_grad():    
    
            temp_1 = temp_1.to(device)
            f_1 = model(temp_1)
            del temp_1
            clear_gpu_mem()
            
            temp_2 = temp_2.to(device)
            f_2 = model(temp_2)
            del temp_2
            clear_gpu_mem()
            
            temp_3 = temp_3.to(device)
            f_3 = model(temp_3)
            del temp_3
            clear_gpu_mem()
            
            temp_4 = temp_4.to(device)
            f_4 = model(temp_4)
            del temp_4
            clear_gpu_mem()
            
            temp_5 = temp_5.to(device)
            f_5 = model(temp_5)
            del temp_5
            clear_gpu_mem()
            
            temp_6 = temp_6.to(device)
            f_6 = model(temp_6)
            del temp_6
            clear_gpu_mem()
            
            temp_7 = temp_7.to(device)
            f_7 = model(temp_7)
            del temp_7
            clear_gpu_mem()
            
            temp_8 = temp_8.to(device)
            f_8 = model(temp_8)
            del temp_8
            clear_gpu_mem()
            
            temp_9 = temp_9.to(device)
            f_9 = model(temp_9)
            del temp_9
            clear_gpu_mem()
            
            temp_10 = temp_10.to(device)
            f_10 = model(temp_10)
            del temp_10
            clear_gpu_mem()
            
            temp_11 = temp_11.to(device)
            f_11 = model(temp_11)
            del temp_11
            clear_gpu_mem()
            
            temp_12 = temp_12.to(device)
            f_12 = model(temp_12)
            del temp_12
            clear_gpu_mem()
            
            temp_13 = temp_13.to(device)
            f_13 = model(temp_13)
            del temp_13
            clear_gpu_mem()
            
            temp_14 = temp_14.to(device)
            f_14 = model(temp_14)
            del temp_14
            clear_gpu_mem()
            
            temp_15 = temp_15.to(device)
            f_15 = model(temp_15)
            del temp_15
            clear_gpu_mem()
            
            temp_16 = temp_16.to(device)
            f_16 = model(temp_16)
            del temp_16
            clear_gpu_mem()
            
            temp_17 = temp_17.to(device)
            f_17 = model(temp_17)
            del temp_17
            clear_gpu_mem()
            
            temp_18 = temp_18.to(device)
            f_18 = model(temp_18)
            del temp_18
            clear_gpu_mem()
            
            temp_19 = temp_19.to(device)
            f_19 = model(temp_19)
            del temp_19
            clear_gpu_mem()
            
            temp_20 = temp_20.to(device)
            f_20 = model(temp_20)
            del temp_20
            clear_gpu_mem()
            
            temp_21 = temp_21.to(device)
            f_21 = model(temp_21)
            del temp_21
            clear_gpu_mem()
            
            temp_22 = temp_22.to(device)
            f_22 = model(temp_22)
            del temp_22
            clear_gpu_mem()
            
            forward = torch.cat((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22), dim = 0)
            # forward = torch.cat((f_1, f_2, f_3, f_4), dim=0)
    return forward

def load_images(data_path):
    # Loading the Patches
    img_name = os.listdir(data_path)
    images = []
    for file_name in img_name:
        image = Image.open(os.path.join(data_path, file_name))
        images.append(image)
    return images

# folder_path = "/kaggle/input/train-data"
folder_path = "/extracted_patches"

folders = next(os.walk(folder_path))[1]
# print(folders)
def clear_gpu_mem():
    gc.collect()
    torch.cuda.empty_cache()

# bad_patients = ["TCGA-D8-A1JB-01A", "TCGA-E2-A570-01A"]
# bad_patients = ['TCGA-BH-A0B9-01A-01-TSA', 'TCGA-BH-A0BW-01A-01-TSA', 'TCGA-A2-A04N-01A-01-TSA', 'TCGA-A7-A0DC-01A-01-TS1', 'TCGA-AR-A24P-01A-01-TSA', 'TCGA-BH-A0EB-01A-01-TSA', 'TCGA-AR-A1AP-01A-01-TSA', 'TCGA-BH-A0E6-01A-01-TSA', 'TCGA-AO-A0J2-01A-01-TSA', 'TCGA-BH-A0HA-01A-01-TSA', 'TCGA-AR-A2LR-01A-01-TSA', 'TCGA-AO-A03U-01B-02-TSB', 'TCGA-BH-A0H0-01A-01-TSA', 'TCGA-BH-A18P-01A-01-TSA', 'TCGA-AN-A0FF-01A-01-TSA', 'TCGA-AR-A1AX-01A-01-TSA']
bad_patients = []
extracted_patients = []
list_of_dirs = os.listdir(folder_path)
dirs_to_extract = set(list_of_dirs) - set(bad_patients)
clear_gpu_mem()
for file_name in bad_patients:
    
    print(file_name)
    
    if os.path.exists(os.path.join(f"{file_name}.csv")):
        continue
    
    else:
        try:
            data_path = os.path.join(folder_path, file_name)
            print(len(os.listdir(data_path)))
            patches = load_images(data_path) # Loading the Extracted Patches
            features = VGG16_feature_extractor(patches) # Extracting Features
            print(features.shape)
            features_df = pd.DataFrame(features.cpu().numpy())
            features_df.to_csv(f"{file_name}.csv")
            extracted_patients.append(file_name)
            clear_gpu_mem()
        
        except:
            bad_patients.append(file_name)
            clear_gpu_mem()
            pass
print(f"These patients were not processed: {bad_patients}")
print(f'These patients were processed: {extracted_patients}')