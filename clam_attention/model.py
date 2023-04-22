import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import sys
from utils import initialize_weights

sample_net = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)

class AttnBackbone(nn.Module):
    def __init__(self, L=1024, D=256, n_classes=1, dropout=False) -> None:
        super(AttnBackbone, self).__init__()

        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        
        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x



class ClamAttention(nn.Module):

    def __init__(self, num_classes=2, dropout=False, k_sample=8, instance_loss_fn=nn.CrossEntropyLoss()) -> None:
        super(ClamAttention, self).__init__()
        self.size = [1000, 512, 256]

        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = AttnBackbone(L=self.size[1], D=self.size[2], n_classes=1, dropout=dropout)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(self.size[1], num_classes)
        instance_classifiers = nn.Linear(self.size[1], num_classes)
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.num_classes = num_classes

        initialize_weights(self)
    
    def relocate(self):

        if (sys.platform in ["win32", "linux"]):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif (sys.platform == "darwin"):
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            print("system not supported for GPU optimization")
            return
        


        
        

    