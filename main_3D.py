import warnings
warnings.filterwarnings("ignore")


import argparse
import os
import sys
import datetime
import time
import math
import json
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from datasets import datasets_utils_3D

import utils
import vision_transformer_3D as vits
from vision_transformer_3D import CLSHead, RECHead_3D
import torchvision

from datasets.load_dataset_3D import NumpyArrayDataset

import wandb

# Log in to wandb with API key on the command line!

# Read config yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

def get_args_parser():
    parser = argparse.ArgumentParser('SiT', add_help=False)

    # Reconstruction Parameters
    parser.add_argument('--drop_perc', type=float, default=config['drop_perc'], help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=config['drop_replace'], help='Drop X percentage of the input image')
    
    parser.add_argument('--drop_align', type=str, default=config['drop_align'], help='Align drop with patches; Set to patch size to align corruption with patches; Possible format 7,16,16')
    parser.add_argument('--drop_type', type=str, default=config['drop_type'], help='Drop Type.')
    
    parser.add_argument('--lmbda', type=int, default=config['lmbda'], help='Scaling factor for the reconstruction loss')
    
    # SimCLR Parameters
    parser.add_argument('--out_dim', default=config['out_dim'], type=int, help="Dimensionality of output features")
    parser.add_argument('--simclr_temp', default=config['simclr_temp'], type=float, help="temperature for SimCLR.")
    parser.add_argument('--momentum_teacher', default=config['momentum_teacher'], type=float, help="EMA parameter for teacher update.")
    

    # Model parameters
    parser.add_argument('--model', default=config['model'], type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_custom'], help="Name of architecture")
    parser.add_argument('--drop_path_rate', type=float, default=config['drop_path_rate'], help="stochastic depth rate")
    

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=config['use_fp16'])
    parser.add_argument('--weight_decay', type=float, default=config['weight_decay'])
    parser.add_argument('--weight_decay_end', type=float, default=config['weight_decay_end'])
    parser.add_argument('--clip_grad', type=float, default=config['clip_grad'])
    
    parser.add_argument('--batch_size', default=config['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['epochs'], type=int, help='Number of epochs of training.')

    parser.add_argument("--lr", default=config['lr'], type=float, help="Learning rate.")
    parser.add_argument("--warmup_epochs", default=config['warmup_epochs'], type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=config['min_lr'], help="Target LR at the end of optimization.")
    

    # Dataset
    parser.add_argument('--data_location', default=config['data_location'], type=str, help='Dataset location.')
    parser.add_argument('--volume_size', type=str, default=config['volume_size'], help='Volume size to randomly crop from the whole volume; Possible format 21,64,64')

    parser.add_argument('--output_dir', default=config['output_dir'], type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=config['saveckp_freq'], type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=config['seed'], type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=config['num_workers'], type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default=config['dist_url'], type=str, help="set up distributed training")
    parser.add_argument("--local_rank", default=config['local_rank'], type=int)
    
    parser.add_argument('--save_recon', default=config['save_recon'], type=utils.bool_flag, help='Save reconstructions.')
    
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
    
    
def train_SiT(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = True

    # prepare dataset
    transform = datasets_utils_3D.DataAugmentationSiT(args)
    
    # Create an instance of custom dataset
    dataset = NumpyArrayDataset(args.data_location, transform=transform)
    
    print(f"Data loaded: there are {len(dataset.file_list)} images.")

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True, 
        collate_fn=collate_batch(args.drop_replace, args.drop_align))

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
                                            drop_path_rate=args.drop_path_rate, volume_size=args.volume_size, patch_size=args.patch_size)
        teacher = vits.__dict__[args.model](input_embed_dim=input_embed_dim, input_depth=input_depth, input_num_heads=input_num_heads,
                                            volume_size=args.volume_size, patch_size=args.patch_size)
    else:
        student = vits.__dict__[args.model](drop_path_rate=args.drop_path_rate, volume_size=args.volume_size, patch_size=args.patch_size)
        teacher = vits.__dict__[args.model](volume_size=args.volume_size, patch_size=args.patch_size)
    
    embed_dim = student.embed_dim
    depth = student.depth
    num_heads = student.num_heads

    student = FullPipline(student, CLSHead(embed_dim, args.out_dim), RECHead_3D(embed_dim, volume_size=args.volume_size, patch_size=args.patch_size))
    teacher = FullPipline(teacher, CLSHead(embed_dim, args.out_dim), RECHead_3D(embed_dim, volume_size=args.volume_size, patch_size=args.patch_size))
    student, teacher = student.cuda(), teacher.cuda()
    
    # synchronize batch norms
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    # we need DDP wrapper to have synchro batch norms working...
    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    #teacher = nn.DataParallel(teacher)
    teacher_without_ddp = teacher.module

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    #student = nn.DataParallel(student)
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    #teacher_without_ddp.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.model} network.")
    
    # preparing SimCLR loss
    simclr_loss = SimCLR(args.simclr_temp).cuda()

    # preparing optimizer 
    optimizer = torch.optim.AdamW(utils.get_params_groups(student))  # to use with ViTs

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # init schedulers 
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256., 
        args.min_lr, args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs)
    
    wd_schedule = utils.cosine_scheduler( args.weight_decay,
        args.weight_decay_end, args.epochs, len(data_loader))
    
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))

    # Resume training 
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student, teacher=teacher,
        optimizer=optimizer, fp16_scaler=fp16_scaler)
    
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Training ..")
    wandb.init(project='SiT-Collaboration',
                # track hyperparameters and run metadata
                config={
                "model": args.model,
                "learning_rate": args.lr,
                "depth": depth,
                "num_heads": num_heads,
                "out_dim": args.out_dim,
                "volume_size": args.volume_size,
                "patch_size": args.patch_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "drop_perc": args.drop_perc,
                "drop_replace": args.drop_replace,
                })
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        
        # Training
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, simclr_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler=None, args=args)
        
        # logs
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1, 'args': args}
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, simclr_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    
    save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    bz = args.batch_size
    plot_ = args.save_recon
    
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, ((clean_crops, corrupted_crops, masks_crops, rand_block_crops)) in enumerate(metric_logger.log_every(data_loader, 5, header)):

        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  
                param_group["weight_decay"] = wd_schedule[it]

        wandb.log({'lr': lr_schedule[it], 'wd': wd_schedule[it]})

        # move images to gpu
        clean_crops = [im.cuda(non_blocking=True) for im in clean_crops]
        corrupted_crops = [im.cuda(non_blocking=True) for im in corrupted_crops]
        masks_crops = [im.cuda(non_blocking=True) for im in masks_crops]
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            t_cls, _ = teacher(torch.cat(clean_crops[0:]), recons=False)             
            s_cls, s_recons = student(torch.cat(corrupted_crops[0:]))
            
            c_loss = simclr_loss(s_cls, t_cls, epoch)
            
            #-------------------------------------------------
            recloss = F.l1_loss(s_recons, torch.cat(clean_crops[0:]), reduction='none')
            
            r_loss = recloss[torch.cat(masks_crops[0:2])==1].mean()
            r_loss_new = recloss[torch.cat(rand_block_crops[0:2])==1].mean()  

            if plot_==True and utils.is_main_process():
                plot_ = False
                #validating: check the reconstructed images
                print_out = save_recon + '/epoch_' + str(epoch).zfill(5)  + '.jpg' 
                imagesToPrint = torch.cat([clean_crops[0][0: min(15, bz)].cpu(),  corrupted_crops[0][0: min(15, bz)].cpu(),
                                       s_recons[0: min(15, bz)].cpu(), masks_crops[0][0: min(15, bz)].cpu()], dim=0)
                # Only save the first slice
                imagesToPrint = imagesToPrint[:, :, 0, :, :]
                torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, bz), normalize=True, range=(-1, 1))
            
            
            loss = c_loss + args.lmbda * r_loss + args.lmbda2 * r_loss_new
            
            wandb.log({'loss': loss, 'c_loss': c_loss, 'r_loss': r_loss, 'r_loss_new': r_loss_new})

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)

            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  
                param_norms = utils.clip_gradients(student, args.clip_grad)

            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(c_loss=c_loss.item())
        metric_logger.update(r_loss=r_loss.item())
        metric_logger.update(r_loss_new=r_loss_new.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class SimCLR(nn.Module):
    def __init__(self, temp=0.2):
        super().__init__()
        
        self.temp = temp
        
    def contrastive_loss(self, q, k):
        
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        # gather all targets
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        N = logits.shape[0] 
        
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)

    def forward(self, student_output, teacher_output, epoch):

        student_out = student_output
        student_out = student_out.chunk(2)

        teacher_out = teacher_output 
        teacher_out = teacher_out.detach().chunk(2)

        return self.contrastive_loss(student_out[0], teacher_out[1]) + self.contrastive_loss(student_out[1], teacher_out[0])


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


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
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid tuple format for --drop_align. Valid format 1,1,1")
    try:
        # Try to parse the argument as a tuple
        args.volume_size = tuple(map(int, args.volume_size.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid tuple format for --volume_size. Valid format 21,64,64")
    try:
        # Try to parse the argument as a tuple
        args.patch_size = tuple(map(int, args.patch_size.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid tuple format for --patch_size. Valid format 7,16,16")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_SiT(args)
