import os
import numpy as np
import torch

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