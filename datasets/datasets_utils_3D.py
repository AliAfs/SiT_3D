from PIL import Image
import numpy as np
import random

from numpy.random import randint
import torch

from PIL import ImageFilter, ImageOps
from torchvision import transforms as tf

from torchvision import transforms


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def getItem(idx, X, target = None, transform=None, training_mode = 'SSL'):
    if transform is not None:
        X = transform(X)

    return X, target



class myRandCrop(tf.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super(myRandCrop, self).__init__(size, scale, ratio, interpolation)
        
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return tf.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)
   
class myRandomHorizontalFlip(tf.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(myRandomHorizontalFlip, self).__init__(p=p)
        
    def forward(self, img):
        if torch.rand(1) < self.p:
            return tf.functional.hflip(img), 1
        return img, 0
    
class PadAndCrop(object):
    """
    Apply Padding and Cropping to the 3D tensor.
    """
    def __init__(self, output_size=(147, 224, 224)):
        self.output_size = output_size

    def __call__(self, tensor):
        #tensor = torch.from_numpy(sample).float()
        dz, dy, dx = np.subtract(self.output_size, tensor.shape)

        dx_left = dx // 2
        dx_right = dx - dx_left
        dy_left = dy // 2
        dy_right = dy - dy_left
        dz_left = dz // 2
        dz_right = dz - dz_left

        tensor_resized = torch.nn.functional.pad(tensor, (dx_left, dx_right, dy_left, dy_right, dz_left, dz_right), "constant")
        
        return tensor_resized.unsqueeze(0)
    
class RandomVolumePatch(object):
    """
    Select a random volume patch from 3D tensor.
    """
    def __init__(self, threshold=0.1, volume_size=(21, 64, 64), required_percentage=0.7):
        self.threshold = threshold
        self.volume_size = volume_size
        self.required_percentage = required_percentage

    def __call__(self, tensor):
        tensor_shape = tensor.shape
        
        if len(tensor_shape) != 3:
            raise ValueError("The input tensor must be a 3D tensor.")

        if any(dim < size for dim, size in zip(tensor_shape, self.volume_size)):
            raise ValueError("The volume size cannot be larger than the tensor dimensions.")

        while True:
            # Generate random indices for the volume
            start_indices = [np.random.randint(dim - size + 1) for dim, size in zip(tensor_shape, self.volume_size)]
            end_indices = [start + size for start, size in zip(start_indices, self.volume_size)]

            # Extract the random volume from the tensor
            volume = tensor[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
            return volume.unsqueeze(0) # FIXME for the moment return volume unconditionally.
            # Calculate the percentage of values above the threshold
            above_threshold = torch.sum(volume > self.threshold).item()

            percentage_above_threshold = above_threshold / torch.numel(volume)

            if percentage_above_threshold >= self.required_percentage:
                return volume.unsqueeze(0)

class GrayValueMirror:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, tensor):
        tensor = tensor.squeeze(0)
        if random.random() < self.probability:
            min_value = torch.min(tensor)
            max_value = torch.max(tensor)
            mirrored_tensor = max_value - tensor + min_value
            return mirrored_tensor.unsqueeze(0)
        else:
            return tensor.unsqueeze(0)

def GMML_replace_list(samples, corrup_prev, masks_prev, drop_type='noise', max_replace=0.35, align=(1,1,1)):
        
    rep_drop = 1 if drop_type == '' else ( 1 / ( len(drop_type.split('-')) + 1 ) )
    
    n_imgs = samples.size()[0] #this is batch size, but in case bad inistance happened while loading
    samples_aug = samples.detach().clone()
    masks = torch.zeros_like(samples_aug)
    for i in range(n_imgs):
        idx_rnd = randint(0, n_imgs)
        if random.random() < rep_drop: 
            samples_aug[i], masks[i] = GMML_drop_rand_patches_3d(samples_aug[i], samples[idx_rnd], max_replace=max_replace, align=align)
        else:
            samples_aug[i], masks[i] = corrup_prev[i], masks_prev[i]

    return samples_aug, masks

def GMML_drop_rand_patches_3d(X, X_rep=None, drop_type='noise', max_replace=0.7, align=(1,1,1), max_block_sz=0.4):
    import torch
    import numpy as np
    from random import randint

    #######################
    # max_replace: percentage of image to be replaced
    # align: align corruption with the patch sizes
    # max_block_sz: percentage of the maximum block to be dropped
    #######################

    torch.manual_seed(0)
    np.random.seed(0)

    C, D, H, W = X.size()
    n_drop_pix = np.random.uniform(min(0.5, max_replace), max_replace) * D * H * W
    mx_blk_depth = int(D * max_block_sz)
    mx_blk_height = int(H * max_block_sz)
    mx_blk_width = int(W * max_block_sz)

    align_d, align_h, align_w = align
    
    align_d = max(1, align_d)
    align_h = max(1, align_d)
    align_w = max(1, align_d)

    mask = torch.zeros_like(X)
    drop_t = np.random.choice(drop_type.split('-'))

    while mask[0].sum() < n_drop_pix:
        
        ####### get a random block to replace
        rnd_e = (randint(0, D - align_d) // align_d) * align_d
        rnd_r = (randint(0, H - align_h) // align_h) * align_h
        rnd_c = (randint(0, W - align_w) // align_w) * align_w

        rnd_d = min(randint(align_d, mx_blk_depth), D - rnd_e)
        rnd_d = round(rnd_d / align_d) * align_d
        rnd_h = min(randint(align_h, mx_blk_height), H - rnd_r)
        rnd_h = round(rnd_h / align_h) * align_h
        rnd_w = min(randint(align_w, mx_blk_width), W - rnd_c)
        rnd_w = round(rnd_w / align_w) * align_w

        if X_rep is not None:
            X[:, rnd_e:rnd_e + rnd_d, rnd_r:rnd_r + rnd_h, rnd_c:rnd_c + rnd_w] = X_rep[:,
                                                                                  rnd_e:rnd_e + rnd_d,
                                                                                  rnd_r:rnd_r + rnd_h,
                                                                                  rnd_c:rnd_c + rnd_w].detach().clone()
        else:
            if drop_t == 'noise':
                X[:, rnd_e:rnd_e + rnd_d, rnd_r:rnd_r + rnd_h, rnd_c:rnd_c + rnd_w] = torch.empty(
                    (C, rnd_d, rnd_h, rnd_w), dtype=X.dtype, device=X.device).normal_()
            elif drop_t == 'zeros':
                X[:, rnd_e:rnd_e + rnd_d, rnd_r:rnd_r + rnd_h, rnd_c:rnd_c + rnd_w] = torch.zeros(
                    (C, rnd_d, rnd_h, rnd_w), dtype=X.dtype, device=X.device)
            else:
                ####### get a random block to replace from
                rnd_e2 = (randint(0, D - rnd_d) // align_d) * align_d
                rnd_r2 = (randint(0, H - rnd_h) // align_h) * align_h
                rnd_c2 = (randint(0, W - rnd_w) // align_w) * align_w

                X[:, rnd_e:rnd_e + rnd_d, rnd_r:rnd_r + rnd_h, rnd_c:rnd_c + rnd_w] = X[:,
                                                                                      rnd_e2:rnd_e2 + rnd_d,
                                                                                      rnd_r2:rnd_r2 + rnd_h,
                                                                                      rnd_c2:rnd_c2 + rnd_w].detach().clone()

        mask[:, rnd_e:rnd_e + rnd_d, rnd_r:rnd_r + rnd_h, rnd_c:rnd_c + rnd_w] = 1

    return X, mask

def random_selected_block(X, rand_block_perc=0.1):
    
    torch.manual_seed(0)
    np.random.seed(0)

    # Calculate the size of the selected block based on the proportion
    C, D, H, W = X.size()
    blk_depth = int(D * rand_block_perc)
    blk_height = int(H * rand_block_perc)
    blk_width = int(W * rand_block_perc)

    # Randomly select the starting position for the selected block
    start_depth = torch.randint(0, D - blk_depth + 1, (1,))
    start_height = torch.randint(0, H - blk_height + 1, (1,))
    start_width = torch.randint(0, W - blk_width + 1, (1,))

    # Calculate the ending position
    end_depth = start_depth + blk_depth
    end_height = start_height + blk_height
    end_width = start_width + blk_width

    block = torch.zeros_like(X)

    # Set the selected block to 1
    block[:, start_depth:end_depth, start_height:end_height, start_width:end_width] = 1

    return block

class DataAugmentationSiT(object):
    def __init__(self, args):
        
        # for corruption
        self.drop_perc = args.drop_perc
        self.drop_type = args.drop_type
        self.drop_align = args.drop_align
        self.rand_block_perc = args.rand_block_perc
        
        
        # first global crop
        self.global_transfo1 = transforms.Compose([
            torch.from_numpy,
            transforms.ConvertImageDtype(torch.float32),
            #PadAndCrop(output_size=(147, 224, 224)),
            RandomVolumePatch(volume_size=args.volume_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10)
        ])

        # second global crop
        self.global_transfo2 = transforms.Compose([
            torch.from_numpy,
            transforms.ConvertImageDtype(torch.float32),
            #PadAndCrop(output_size=(147, 224, 224)),
            RandomVolumePatch(volume_size=args.volume_size),
            #GrayValueMirror(probability=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10)
        ])

    def __call__(self, image):
        
        clean_crops = []
        corrupted_crops = []
        masks_crops = []
        rand_block_crops = []

        ## augmented 1
        im_orig = self.global_transfo1(image)
        
        im_corrupted = im_orig.detach().clone()
        im_mask = torch.zeros_like(im_corrupted)
        im_block = torch.zeros_like(im_corrupted)
        if self.drop_perc > 0:
            im_corrupted, im_mask = GMML_drop_rand_patches_3d(im_corrupted, 
                                                           max_replace=self.drop_perc, drop_type=self.drop_type, align=self.drop_align)
        im_block = random_selected_block(im_corrupted, rand_block_perc = self.rand_block_perc)

        clean_crops.append(im_orig)
        corrupted_crops.append(im_corrupted)
        masks_crops.append(im_mask)
        rand_block_crops.append(im_block)

        ## augmented 2
        im_orig = self.global_transfo2(image)
        
        im_corrupted = im_orig.detach().clone()
        im_mask = torch.zeros_like(im_corrupted)
        if self.drop_perc > 0:
            im_corrupted, im_mask = GMML_drop_rand_patches_3d(im_corrupted, 
                                                           max_replace=self.drop_perc, drop_type=self.drop_type, align=self.drop_align)
        im_block = random_selected_block(im_corrupted, rand_block_perc = self.rand_block_perc)
        
        clean_crops.append(im_orig)
        corrupted_crops.append(im_corrupted)
        masks_crops.append(im_mask)
        rand_block_crops.append(im_block)

        return clean_crops, corrupted_crops, masks_crops, rand_block_crops