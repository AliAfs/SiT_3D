import argparse
import os
import numpy as np

import pydicom

from scipy.ndimage import zoom


def get_args_parser():
    parser = argparse.ArgumentParser('Dicom_to_Numpy', add_help=False)

    parser.add_argument('--clean_folder_dir', default='/path/to/sorted_folder', type=str, help='Path to the sorted cohort dicom files')
    parser.add_argument('--voxel_size', default='2,2,2', type=str, help='Voxel size in the format i,j,k')
    parser.add_argument('--output_dir', default="/path/to/output", type=str, help='Path to save volumes as npz files.')

    return parser

def dicom_to_npz(args):
    "Function to unify voxel sizes and save volume as npz files."

    root_directory = args.clean_folder_dir
    output_directory = args.output_dir
    target_voxel_size = tuple(float(x) for x in args.voxel_size.split(','))

    for root, dirs, files in os.walk(root_directory):
        # Iterate over the subdirectories
        for subdirectory in dirs:
            subdirectory_path = os.path.join(root, subdirectory)
            dicom_files = [f for f in os.listdir(subdirectory_path) if f.endswith('.dcm')]

            # Check if the folder contains more than one DICOM file
            if len(dicom_files) > 1:
                try:
                    # Sort the DICOM files based on slice position or number
                    dicom_files.sort(key=lambda x: pydicom.dcmread(os.path.join(subdirectory_path, x)).SliceLocation)

                    slices = []
                    for file_name in dicom_files:
                        dicom_path = os.path.join(subdirectory_path, file_name)
                        ds = pydicom.dcmread(dicom_path)
                        slices.append(ds.pixel_array)

                    # Convert slices into a 3D NumPy array
                    volume = np.stack(slices)

                    # Get the current voxel spacing
                    current_voxel_size = (float(ds.SliceThickness), float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))

                    # Calculate the zoom factors for resampling
                    zoom_factors = (current_voxel_size[0] / target_voxel_size[0],
                                    current_voxel_size[1] / target_voxel_size[1],
                                    current_voxel_size[2] / target_voxel_size[2])

                    # Resample the volume to the target voxel size
                    volume_resampled = zoom(volume, zoom_factors, order=1)
                    

                    # Normalize the volume to [0, 1]
                    #min_value = np.min(volume_resampled)
                    #max_value = np.max(volume_resampled)
                    #volume_normalized = (volume_resampled - min_value) / (max_value - min_value)
                    
                    # Save the volume as compressed numpy array npz
                    last_folder_name = os.path.basename(subdirectory_path)
                    save_path = os.path.join(output_directory, f'{last_folder_name}.npz')
                    np.savez_compressed(save_path, volume=volume_resampled)
                    print(f'Processed Volume: {last_folder_name}')
                except:
                    pass
                
        # Move to the next subdirectory
        for subdirectory in dirs:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dicom_to_Numpy', parents=[get_args_parser()])
    args = parser.parse_args()
    dicom_to_npz(args)
