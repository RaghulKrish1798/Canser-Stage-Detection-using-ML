'''
Test the FWD Pass of ResNet50 network pre-trained on ImageNet data with our own patches
'''

import torch
from torchvision.models import resnet50, ResNet50_Weights

sample_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

print(ResNet50_Weights.IMAGENET1K_V2.transforms())

