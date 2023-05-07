import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# clustered_data = pd.read_csv("clustered_data.csv", delimiter=",")

# for patient in clustered_data['pid'].unique():
#     patient_rows = clustered_data.loc[clustered_data['pid'] == patient]
#     cluster_data = {}
#     for cluster_number in patient_rows['cluster_num'].unique():
#         cluster_data[cluster_number] = patient_rows.loc[patient_rows['cluster_num']==cluster_number].drop(columns=["pid", "cluster_num"]).values
    
class CustomDataset(Dataset):

    def __init__(self, data_path, label_path, cluster_num=10) -> None:
        super(CustomDataset, self).__init__()
        data = pd.read_csv(data_path, delimiter=",")
        labels = pd.read_csv(label_path, delimiter=",")

        data = data.loc[data['pid'].isin(labels.pid.values)]
        labels = labels.loc[labels['pid'].isin(data.pid.values)]

        self.data = data
        self.labels = labels
        self.labels['stage'] = self.labels.stage.map(lambda x: one_hot_binary(x))
        self.patients = data.pid.values
        self.cluster_num = cluster_num

    def len(self):
        return len(self.labels)
    
    def __getitem__(self, index):

        patient = self.patients[index]
        patient_rows = self.data.loc[self.data['pid'] == patient]
        
        cluster_data = {}
        mask = np.zeros((1, self.cluster_num), dtype=np.float16)
        
        for cluster_number in patient_rows['cluster_num'].unique():
            mask[cluster_number] = 1
            cluster_data[cluster_number] = torch.Tensor(patient_rows.loc[patient_rows['cluster_num']==cluster_number].drop(columns=["pid", "cluster_num"]).transpose().values, dtype=torch.float32)
        
        for i in range(self.cluster_num):
            if i not in cluster_data.keys():
                cluster_data[i] = torch.Tensor(np.zeros(1, 1000), dtype=np.float32)
        
        mask = torch.Tensor(mask, dtype=torch.float32)

        label = self.labels['stage'].loc[self.labels['pid'] == patient].values
        label = torch.Tensor(label, dtype=torch.float32)

        return cluster_data, mask, label




def one_hot_binary(label):
    if(label in ["stage i", "stage ia", "stage ib"]):
        return [1,0]
    elif(label in ["stage ii", "stage iia", "stage iib"]):
        return [0,1]
    else:
        raise ValueError
