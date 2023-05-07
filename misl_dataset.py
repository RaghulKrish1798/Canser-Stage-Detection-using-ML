import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
# import defaultdict

# clustered_data = pd.read_csv("clustered_data.csv", delimiter=",")

# for patient in clustered_data['pid'].unique():
#     patient_rows = clustered_data.loc[clustered_data['pid'] == patient]
#     cluster_data = {}
#     for cluster_number in patient_rows['cluster_num'].unique():
#         cluster_data[cluster_number] = patient_rows.loc[patient_rows['cluster_num']==cluster_number].drop(columns=["pid", "cluster_num"]).values
    
class CustomDataset(Dataset):

    def __init__(self, features, labels, cluster_num=10) -> None:
        super(CustomDataset, self).__init__()

        self.data = features
        self.labels = labels
        self.labels['stage'] = self.labels.stage.map(lambda x: one_hot_binary(x))
        self.patients = self.data.pid.values
        self.cluster_num = cluster_num

    def __len__(self):
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

def stratified_split(data, labels, fraction, random_state=None):

    if random_state:
        random.seed(random_state)

    indices_per_label = {}
    
    for stage in labels['stage'].unique():
        indices_per_label[stage] = labels[labels['stage']==stage].index.values
    
    first_set_indices, second_set_indices = list(), list()

    first_set_indices, second_set_indices = list(), list()

    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices.tolist(), n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices.tolist()) - set(random_indices_sample))
    
    first_set_inputs = data.loc[first_set_indices]
    first_set_labels = labels.loc[first_set_indices]
    second_set_inputs = data.loc[second_set_indices]
    second_set_labels = labels.loc[second_set_indices]

    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels

def train_valid_test_split(data_path, label_path, test_fraction, valid_fraction, random_state=None):

    data = pd.read_csv(data_path, delimiter=",")
    labels = pd.read_csv(label_path, delimiter=",")
    labels = remove_unwanted_labels(labels)

    data = data.loc[data['pid'].isin(labels.pid.values)]
    labels = labels.loc[labels['pid'].isin(data.pid.values)]

    print(len(data))
    print(len(labels))


    x_test, y_test, x_train_val, y_train_val = stratified_split(data, labels, test_fraction, random_state)
    x_val, y_val, x_train, y_train = stratified_split(x_train_val, y_train_val, valid_fraction, random_state)
    

    test_dataset = CustomDataset(x_test, y_test)
    valid_dataset = CustomDataset(x_val, y_val)
    train_dataset = CustomDataset(x_train, y_train)


    return train_dataset, valid_dataset, test_dataset

def remove_unwanted_labels(stage_idx:pd.DataFrame) -> pd.DataFrame:
    final_df = stage_idx.loc[(stage_idx['stage'] == "stage i") | (stage_idx['stage'] == "stage ia") | (stage_idx['stage'] == "stage ib") | (stage_idx['stage'] == "stage ii") | (stage_idx['stage'] == "stage iia") | (stage_idx['stage'] == "stage iib")]
    return final_df