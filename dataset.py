import torch
from torch.utils.data import Dataset
from glob import glob
import os
import pandas as pd

class CustomImageDataset(Dataset):

    def __init__(self, dataset_dir) -> None:
        super(CustomImageDataset, self).__init__()
        # self.dataset_dir = dataset_dir
        self.client_img_dirs = os.listdir(os.path.join(dataset_dir, "images"))
        self.client_stage_dir = os.path.join(dataset_dir, "BRCA_stages.csv")
        stages_df = 
