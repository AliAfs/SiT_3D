import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('..')


import argparse
import os
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from datasets import datasets_utils_3D, load_dataset

import utils
import vision_transformer_3d as vits
from vision_transformer_3d import CLSHead, RECHead_3D
import torchvision

from torch.utils.data import Dataset, DataLoader

from datasets.load_dataset_3D import NumpyArrayDataset

def get_args_parser():
    parser = argparse.ArgumentParser('SiT', add_help=False)

    def tuple_arg(arg):
      if arg is None:
          return None
      try:
          # Try to parse the argument as a tuple
          return tuple(map(int, arg.split(',')))
      except ValueError:
          raise argparse.ArgumentTypeError("Invalid tuple format.")

    # Reconstruction Parameters
    parser.add_argument('--drop_perc', type=float, default=0.5, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.5, help='Drop X percentage of the input image')
    
    parser.add_argument('--drop_align', type=str, default="1,1,1", help='Align drop with patches; Set to patch size to align corruption with patches')
    parser.add_argument('--drop_type', type=str, default='zeros', help='Drop Type.')
    parser.add_argument('--lmbda', type=int, default=1, help='Scaling factor for the reconstruction loss')

    # SimCLR Parameters
    parser.add_argument('--out_dim', default=192, type=int, help="Dimensionality of output features")
    parser.add_argument('--simclr_temp', default=0.2, type=float, help="tempreture for SimCLR.")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="EMA parameter for teacher update.")

    # Model parameters
    parser.add_argument('--model', default='vit_tiny', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_custom'], help="Name of architecture")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    
    # Training parameters
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--volume_size', type=str, default='21,64,64', help='Volume size that selected randomly from the whole input volume')
    parser.add_argument('--patch_size', type=str, default='7,16,16', help='Size of patches that model divide the volume into')
    
    # Dataset
    parser.add_argument('--data_location', default='../../npz/original', type=str, help='Dataset location.')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')

    return parser

def train_SiT(args):
    # prepare dataset
    transform = datasets_utils_3D.DataAugmentationSiT(args)

    # Create an instance of custom dataset
    dataset = NumpyArrayDataset(args.data_location, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    

    # building networks 
    if args.model=='vit_custom':
        try:
            input_embed_dim = int(input("embed_dim: "))
            input_depth = int(input("depth: "))
            input_num_heads = int(input("num_heads: "))
            if input_embed_dim%input_num_heads != 0:
                raise ValueError("Invalid input. embed_dim must be dividable by num_heads!")
        except ValueError:
            raise ValueError("Invalid input. Please enter integers.")
        student = vits.__dict__[args.model](input_embed_dim=input_embed_dim, input_depth=input_depth, input_num_heads=input_num_heads,
                                            drop_path_rate=args.drop_path_rate, patch_size=args.patch_size, volume_size = args.volume_size)
    else:
        student = vits.__dict__[args.model](drop_path_rate=args.drop_path_rate, patch_size=args.patch_size, volume_size = args.volume_size)

    embed_dim = student.embed_dim
    depth = student.depth
    num_heads = student.num_heads

    print(f"Model {args.model} ...")
    print(f"        Embed dimension: ", embed_dim)
    print(f"        Depth: ", depth)
    print(f"        Number of heads: ", num_heads)


    student = FullPipline(student, CLSHead(embed_dim, args.out_dim), RECHead_3D(in_dim=embed_dim, patch_size=args.patch_size, volume_size = args.volume_size))

    for clean_crops, corrupted_crops, masks_crops in data_loader:
        s_cls, s_recons = student(torch.cat(corrupted_crops[0:]))
        
        break

class FullPipline(nn.Module):
    def __init__(self, backbone, head, head_recons):
        super(FullPipline, self).__init__()
        
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.head_recons = head_recons

    def forward(self, x, recons=True):
        _out = self.backbone(x)
        
        if recons==True:
            return self.head(_out[:, 0]), self.head_recons(_out[:, 1:])
        else:
            return self.head(_out[:, 0]), None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT', parents=[get_args_parser()])
    args = parser.parse_args()

    try:
      # Try to parse the argument as a tuple
      args.drop_align = tuple(map(int, args.drop_align.split(',')))
      args.volume_size = tuple(map(int, args.volume_size.split(',')))
      args.patch_size = tuple(map(int, args.patch_size.split(',')))
    except ValueError:
      raise argparse.ArgumentTypeError("Invalid tuple format. Valid format 1,1,1")
    
    for i, j in zip(args.volume_size, args.patch_size):
        assert i % j == 0, "volume_size must be divisible by patch_size"
        
    train_SiT(args)