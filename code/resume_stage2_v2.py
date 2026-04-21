import torch
from model import CausalSpatiotemporalModel
from train import train_stage2
from torch.utils.tensorboard import SummaryWriter

resume_weight = './weight_v2/stage1_0013.weight'

net = CausalSpatiotemporalModel()
net.load_state_dict(torch.load(resume_weight, map_location='cpu'))
print(f'Resumed from {resume_weight}')

writer = SummaryWriter('./runs/v2_stage2/')

data_path_roots = ['../data/sphere_dataset/']
ss_paths = [
    '../data/character_dataset/michelle/cross_jumps',
    '../data/character_dataset/kaya/zombie_scream',
    '../data/character_dataset/big_vegas/cross_jumps',
    '../data/character_dataset/mousey/dancing_1',
    '../data/character_dataset/ortiz/jazz_dancing',
]

net = train_stage2(net, writer, data_path_roots, './weight_v2/',
                   train_seq_num=7, test_seq_num=7,
                   self_supervised_paths=ss_paths)

writer.close()
print('Stage 2 v2 complete.')
