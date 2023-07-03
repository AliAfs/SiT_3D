import os
import numpy as np
import torch
from torch.utils.data import Dataset

"""
def load_data_list(folder_path = "../cohort_new/numpy arrays compressed new"):
    file_list = os.listdir(folder_path)

    data_list = []
    for file_name in file_list:
        if file_name.endswith('.npz'):
            file_path = os.path.join(folder_path, file_name)  
            data = np.load(file_path)
            array = data['volume'].astype(float)
            data_list.append(array)

    # Squeeze the arrays to remove the redundant dimension
    data_list = [arr.squeeze() for arr in data_list]

    return data_list 
"""

class NumpyArrayDataset(Dataset):
    def __init__(self, directory="../cohort_new/numpy arrays compressed new", transform=None):
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load and preprocess the data at the given index        
        array = np.load(os.path.join(self.directory, self.file_list[idx]))
        data = array['volume'].astype(float)
        
        # Perform any necessary transformations or preprocessing on the data
        if self.transform is not None:
            clean_crops, corrupted_crops, masks_crops = self.transform(data.squeeze())
        return clean_crops, corrupted_crops, masks_crops