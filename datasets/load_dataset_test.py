import os
import numpy as np
import torch
from torch.utils.data import Dataset

def load_data_list(folder_path = "../cohort_new/numpy arrays compressed new"):
    file_list = os.listdir(folder_path)

    data_list = []
    for file_name in file_list:
        if file_name.endswith('.npz'):
            file_path = os.path.join(folder_path, file_name)  
            data = np.load(file_path)
            array = data['arr_0'].astype(float)
            data_list.append(array)

    # Squeeze the arrays to remove the redundant dimension
    data_list = [arr.squeeze() for arr in data_list]
    
    """
    data_list_resized = []
    output_size = (147, 224, 224)
    for sample in data_list:
        sample_tensor = torch.from_numpy(sample).float()
        dz, dy, dx = np.subtract(output_size, sample_tensor.shape)

        dx_left = dx // 2
        dx_right = dx - dx_left
        dy_left = dy // 2
        dy_right = dy - dy_left
        dz_left = dz // 2
        dz_right = dz - dz_left

        sample_resized = torch.nn.functional.pad(sample_tensor, (dx_left, dx_right, dy_left, dy_right, dz_left, dz_right), "constant")
        data_list_resized.append(sample_resized.unsqueeze(0))
    """

    return data_list 

class NumpyArrayDataset(Dataset):
    def __init__(self, directory="../cohort_new/numpy arrays compressed new", transform=None):
        self.directory = directory
        self.file_list = os.listdir(directory)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load and preprocess the data at the given index
        #data = self.file_list[index]
        file_name = self.file_list[idx]
        file_path = os.path.join(self.directory, file_name)
        
        array = np.load(file_path)
        array_file0 = array.files[0]
        data = array[array_file0].astype(float)
        # Squeeze the arrays to remove the redundant dimension
        data = data.squeeze()
        
        # Perform any necessary transformations or preprocessing on the data
        if self.transform is not None:
            clean_crops, corrupted_crops, masks_crops = self.transform(data)
        return clean_crops, corrupted_crops, masks_crops