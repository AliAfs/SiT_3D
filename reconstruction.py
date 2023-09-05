import vision_transformer_3D as vits
from vision_transformer_3D import CLSHead, RECHead_3D

from datasets import datasets_utils_3D, load_dataset
from datasets.load_dataset_3D import NumpyArrayDataset
import utils

import torch.nn as nn
import torch
import torchvision

import matplotlib.pyplot as plt

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Reconstruction', add_help=False)

    # Reconstruction Parameters
    parser.add_argument('--drop_perc', type=float, default=0.7, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0, help='Drop X percentage of the input image')
    
    parser.add_argument('--drop_align', type=str, default="1,1,1", help='Align drop with patches; Set to patch size to align corruption with patches; Possible format 7,16,16')
    parser.add_argument('--drop_type', type=str, default='zeros', help='Drop Type.')
    

    # Model parameters
    parser.add_argument('--model', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_custom'], help="Name of architecture")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    

    # Dataset
    parser.add_argument('--data_location', default='../npz/original/', type=str, help='Dataset location.')
    parser.add_argument('--volume_size', type=str, default="21,64,64", help='Volume size to randomly crop from the whole volume; Possible format 21,64,64')
    
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')


    # Checkpoint
    parser.add_argument('--checkpoint_path', default='../checkpoints/checkpoint0460.pth', type=str, help='Path to the checkpoint to load.')
    
    return parser

# replace from other images
class collate_batch(object): 
    def __init__(self, drop_replace=0.3, drop_align=(1,1,1)):
        self.drop_replace = drop_replace
        self.drop_align = drop_align
        
    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        
        if self.drop_replace > 0:
            batch[1][0], batch[2][0] = datasets_utils_3D.GMML_replace_list(batch[0][0], batch[1][0], batch[2][0],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
            batch[1][1], batch[2][1] = datasets_utils_3D.GMML_replace_list(batch[0][1], batch[1][1], batch[2][1],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
        
        return batch

        
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def plot_slices(img1, img2, img3, cmap='gray'):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Create a figure with two subplots

    # Plot the first image on the left subplot
    axes[0].imshow(img1, cmap=cmap)
    axes[0].axis('off')  

    # Plot the second image on the right subplot
    axes[1].imshow(img2, cmap=cmap)
    axes[1].axis('off')  

    # Plot the second image on the right subplot
    axes[2].imshow(img3, cmap=cmap)
    axes[2].axis('off') 

    plt.show()  # Display the figure

@torch.no_grad()
def visualize_center_slice(args):
    #utils.init_distributed_mode(args)
    utils.fix_random_seeds()

    # prepare dataset
    transform = datasets_utils_3D.DataAugmentationSiT(args)

    # Create an instance of custom dataset
    dataset = NumpyArrayDataset(args.data_location, transform=transform)

    student = vits.__dict__["vit_base"](drop_path_rate=0.1)
    student = FullPipline(student, CLSHead(768, 256), RECHead_3D(768))
    #student = nn.DataParallel(student)

    #utils.restart_from_checkpoint(
    #    "../checkpoint0460.pth",
    #    student=student)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    checkpoint['student'] = remove_module_prefix(checkpoint['student'])
    student.load_state_dict(checkpoint['student'])

    clean_crops, corrupted_crops, masks_crops = dataset[0]
    clean_crops = [clean_crop.unsqueeze(0) for clean_crop in clean_crops]
    corrupted_crops = [corrupted_crop.unsqueeze(0) for corrupted_crop in corrupted_crops]

    s_cls, s_recons = student(torch.cat(corrupted_crops[0:]))
    plot_slices(clean_crops[0][0, 0, 0].numpy(), s_recons[0, 0, 0].numpy(), corrupted_crops[0][0, 0, 0].numpy())

class FullPipline(nn.Module):
    def __init__(self, backbone, head, head_recons):
        super(FullPipline, self).__init__()
        
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.head_recons = head_recons

    def forward(self, x, recons=True):
        _out = self.backbone(x)
        
        if recons:
            return self.head(_out[:, 0]), self.head_recons(_out[:, 1:])
        else:
            return self.head(_out[:, 0]), None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Reconstruction', parents=[get_args_parser()])
    args = parser.parse_args()
    try:
        # Try to parse the argument as a tuple
        args.drop_align = tuple(map(int, args.drop_align.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid tuple format for --drop_align. Valid format 1,1,1")
    try:
        # Try to parse the argument as a tuple
        args.volume_size = tuple(map(int, args.volume_size.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid tuple format for --volume_size. Valid format 21,64,64")
    
    visualize_center_slice(args)

