import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

sample_net = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
