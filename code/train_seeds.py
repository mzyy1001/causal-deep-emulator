"""Train V9 cone with different random seeds."""
import os, sys, torch
import numpy as np

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print('Seed: %d' % seed)

# Import after setting seed
import config
config.NUM_SCALES = 1
config.USE_CAUSAL_CONE = True

import importlib
import model as mm
importlib.reload(mm)

weight_dir = './weight_v9_seed%d/' % seed
os.makedirs(weight_dir, exist_ok=True)

# Use train_v3's training function
from train_v3 import train_stage1_v3
from model import CausalSpatiotemporalModel
from torch.utils.tensorboard import SummaryWriter

net = CausalSpatiotemporalModel()
writer = SummaryWriter(os.path.join(weight_dir, 'runs/'))

net = train_stage1_v3(
    net, writer, ['../data/sphere_all/'], weight_dir,
    train_seq_num=7, test_seq_num=7,
    epochs=60, lr=1e-3, batch_size=64, num_scales=1,
    grad_accum=1, warmup_epochs=5, grad_clip=1.0)

writer.close()
print('Seed %d training complete.' % seed)
