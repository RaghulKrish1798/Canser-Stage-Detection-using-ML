'''
Test the FWD Pass of ResNet50 network pre-trained on ImageNet data with our own patches
'''

import torch
from torchvision.models import resnet50, ResNet50_Weights
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt

## UNCOMMENT THIS BEFORE TEST
sample_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# torch.save(sample_model.state_dict(), '../torch_start_model.pt')

dirpaths = os.listdir('../extracted_patches/')
# print(dirpaths)

patches_dir = glob(f'../extracted_patches/{dirpaths[0]}/*.png')
patch_info = cv2.imread(patches_dir[0])
patch_info = patch_info.reshape(3, 224, 224)
patch_info = torch.Tensor(patch_info)
patch_info = patch_info.unsqueeze(0)

feat_vector = sample_model(patch_info)
print(feat_vector.shape)



