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
        self.patients = self.data.pid.unique()
        self.cluster_num = cluster_num

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index):


        patient = self.patients[index]
        # print(patient)
        patient_rows = self.data.loc[self.data['pid'] == patient]

        cluster_data = [torch.Tensor(np.zeros((1, 1000)))] * self.cluster_num
        mask = np.zeros(self.cluster_num)
        
        for cluster_number in patient_rows['cluster_num'].unique():
            mask[cluster_number] = 1
            cluster_data[cluster_number] = torch.Tensor(patient_rows.loc[patient_rows['cluster_num']==cluster_number].drop(columns=["pid", "cluster_num"]).values)
        
        mask = torch.tensor(mask)

        # print(patient in self.labels.pid.values)
        
        label = self.labels[self.labels['pid'] == patient]['stage'].values[0]
        # print(self.labels.loc[self.labels['pid'] == patient]['stage'])

        # print(label)
        # raise ValueError
        label = torch.Tensor(label)

        return (cluster_data, mask), label




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

    patients_per_label = {}
    
    for stage in labels['stage'].unique():
        patients_per_label[stage] = set(labels[labels['stage']==stage].pid.values)
    
    first_set_patients, second_set_patients = list(), list()

    for label, patients in patients_per_label.items():
        n_patients_for_label = round(len(patients) * fraction)
        random_patients_sample = random.sample(patients, n_patients_for_label)
        if label == "stage ii":
            appending_patients_sample = random.sample(random_patients_sample, round(len(random_patients_sample) * 0.65))
        else:
            appending_patients_sample = random_patients_sample
        first_set_patients.extend(appending_patients_sample)
        second_set_patients.extend(patients - set(random_patients_sample))
    
    first_set_inputs = data.loc[data['pid'].isin(first_set_patients)]
    first_set_labels = labels.loc[labels['pid'].isin(first_set_patients)]
    second_set_inputs = data.loc[data['pid'].isin(second_set_patients)]
    second_set_labels = labels.loc[labels['pid'].isin(second_set_patients)]
    # print(first_set_inputs.pid.values)
    # print(first_set_labels.pid.values)

    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels

def train_valid_test_split(data_path, label_path, test_fraction=0.2, valid_fraction=0.1, random_state=None, test_mode=False):

    data = pd.read_csv(data_path, delimiter=",")
    labels = pd.read_csv(label_path, delimiter=",")
    labels = remove_unwanted_labels(labels)
    labels.drop(columns=["Unnamed: 0"], inplace=True)

    print(len)
    data = data[data['pid'].isin(labels.pid.values)]
    labels = labels[labels['pid'].isin(data.pid.values)]


    data.sort_values(by="pid", inplace=True)
    labels.sort_values(by="pid", inplace=True)

    # print(len(data))
    # print(len(labels))

    labels['stage'] = labels.stage.map(lambda x: map_to_one_hot_binary_logits(x))

    if test_mode:
         return(CustomDataset(data, labels, cluster_num=10))

    x_test, y_test, x_train_val, y_train_val = stratified_split(data, labels, test_fraction, random_state)
    x_val, y_val, x_train_raw, y_train_raw = stratified_split(x_train_val, y_train_val, valid_fraction, random_state)
    x_train, y_train, _, _ = stratified_split(data, labels, 1.0, random_state)


    print(f"test stage values: {y_test.stage.value_counts()}")
    print(f"train stage values: {y_train.stage.value_counts()}")
    print(f"validation stage values: {y_val.stage.value_counts()}")

    

    test_dataset = CustomDataset(x_test, y_test)
    valid_dataset = CustomDataset(x_val, y_val)
    train_dataset = CustomDataset(x_train, y_train)


    return train_dataset, valid_dataset, test_dataset

def remove_unwanted_labels(stage_idx:pd.DataFrame) -> pd.DataFrame:
    final_df = stage_idx.loc[(stage_idx['stage'] == "stage i") | (stage_idx['stage'] == "stage ia") | (stage_idx['stage'] == "stage ib") | (stage_idx['stage'] == "stage ii") | (stage_idx['stage'] == "stage iia") | (stage_idx['stage'] == "stage iib")]
    return final_df

def map_to_one_hot_binary_logits(stage:str) -> str:
    if ((stage=="stage i") | (stage=="stage ia") | (stage=="stage ib")):
        return "stage i"
    elif (stage=="stage ii") | (stage=="stage iia") | (stage=="stage iib"):
        return "stage ii"
    else:
        raise(ValueError)