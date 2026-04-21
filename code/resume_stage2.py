"""Resume Stage 2 training from a checkpoint."""
import os, sys, torch
from model import CausalSpatiotemporalModel
from train import train_stage2
from torch.utils.tensorboard import SummaryWriter

# Config
resume_weight = './weight/stage2_0029.weight'
start_epoch = 30
remaining_epochs = 10

# Patch STAGE2_EPOCHS to only run remaining
import train as train_module
train_module.STAGE2_EPOCHS = remaining_epochs

net = CausalSpatiotemporalModel()
net.load_state_dict(torch.load(resume_weight, map_location='cpu'))
print(f'Resumed from {resume_weight} (epoch 29)')

writer = SummaryWriter('./runs/resume/')

data_path_roots = ['../data/sphere_dataset/']
ss_paths = [
    '../data/character_dataset/michelle/cross_jumps',
    '../data/character_dataset/kaya/zombie_scream',
    '../data/character_dataset/big_vegas/cross_jumps',
    '../data/character_dataset/mousey/dancing_1',
    '../data/character_dataset/ortiz/jazz_dancing',
]

net = train_stage2(net, writer, data_path_roots, './weight/',
                   train_seq_num=7, test_seq_num=7,
                   self_supervised_paths=ss_paths)

writer.close()
print('Stage 2 resume complete.')
