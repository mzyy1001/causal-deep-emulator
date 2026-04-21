"""Evaluate V9 cone on all 10 character motions at original stiffness."""
import torch, numpy as np, os
import config
config.NUM_SCALES = 1
config.USE_CAUSAL_CONE = True
import importlib, model as mm
importlib.reload(mm)
from model import CausalSpatiotemporalModel, build_multiscale_edges
from model_baseline import Graph_MLP
from config import TEMPORAL_WINDOW
import data_loader

device = torch.device('cuda')
CR = '../data/character_dataset'

cone = CausalSpatiotemporalModel()
cone.load_state_dict(torch.load('./weight_v9_cone/best_stage1.weight', map_location='cpu'))
cone.eval().to(device)

bl = Graph_MLP()
bl.load_state_dict(torch.load('./weight_v7_baseline/best.weight', map_location='cpu'))
bl.eval().to(device)

print('Evaluating ALL 10 motions')
ALL = [('mousey','dancing_1'),('mousey','swing_dancing_1'),
       ('michelle','cross_jumps'),('michelle','gangnam_style'),
       ('big_vegas','cross_jumps'),('big_vegas','cross_jumps_rotation'),
       ('kaya','dancing_running_man'),('kaya','zombie_scream'),
       ('ortiz','cross_jumps_rotation'),('ortiz','jazz_dancing')]

print('%-30s %10s %10s %8s' % ('Motion','V9 Cone','Baseline','Ratio'))
print('-' * 62)

for ch, mo in ALL:
    path = os.path.join(CR, ch, mo)
    if not os.path.isdir(path):
        print('%-30s SKIP' % ('%s/%s' % (ch, mo)))
        continue
    c = data_loader.loadData_Int(os.path.join(path, 'c'))
    ct = torch.from_numpy(c).long().to(device)
    fr = (ct == 0)
    V = len(c)
    ad = torch.from_numpy(data_loader.loadData_Int(os.path.join(path, 'adj')).reshape(V, -1)).long().to(device)
    ms = build_multiscale_edges(ad, 1)
    kr = data_loader.loadData_Float(os.path.join(path, 'k'))
    k = torch.from_numpy(np.expand_dims(kr, 1) * 0.000001).float().to(device)
    mr = data_loader.loadData_Float(os.path.join(path, 'm'))
    mr = np.expand_dims(mr, 1) * 1000
    mr[0] = 1.0
    ma = torch.from_numpy(mr).float().to(device)
    fn = len([f for f in os.listdir(path) if f.startswith('x_')])
    cm, bm = [], []
    with torch.no_grad():
        for i in range(min(fn - 1, 50)):
            uf, xf = [], []
            for t in range(TEMPORAL_WINDOW + 1):
                ix = max(0, min(i - t, fn - 1))
                u = data_loader.loadData_Float(os.path.join(path, 'u_%d' % ix)).reshape(-1, 3)
                u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
                uf.append(torch.from_numpy(u).float().to(device))
            for t in range(-1, TEMPORAL_WINDOW):
                ix = max(0, min(i - t, fn - 1))
                x = data_loader.loadData_Float(os.path.join(path, 'x_%d' % ix)).reshape(-1, 3)
                x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
                xf.append(torch.from_numpy(x).float().to(device))
            dy = torch.stack(uf, 0)
            rf = torch.stack(xf, 0)
            un = data_loader.loadData_Float(os.path.join(path, 'u_%d' % (i + 1))).reshape(-1, 3)
            un = np.concatenate([np.zeros((1, 3), dtype=np.float64), un], axis=0)
            un = torch.from_numpy(un).float().to(device)
            uc = data_loader.loadData_Float(os.path.join(path, 'u_%d' % i)).reshape(-1, 3)
            uc = np.concatenate([np.zeros((1, 3), dtype=np.float64), uc], axis=0)
            uc = torch.from_numpy(uc).float().to(device)
            gt = (un - uc)[fr]
            pc = cone(ct, dy, rf, ad, k, ma, ms)
            cm.append(((pc - gt) ** 2).mean().item())
            dp = torch.cat([dy[0], dy[1], dy[min(2, dy.shape[0] - 1)]], dim=-1).unsqueeze(0)
            rp = torch.cat([rf[0], rf[1], rf[min(2, rf.shape[0] - 1)]], dim=-1).unsqueeze(0)
            pb = bl(ct, dp, rp, ad, k.unsqueeze(0), ma.unsqueeze(0)).squeeze(0)
            bm.append(((pb - gt) ** 2).mean().item())
    ca = float(np.mean(cm))
    ba = float(np.mean(bm))
    print('%-30s %10.6f %10.6f %8.2fx' % ('%s/%s' % (ch, mo), ca, ba, ca / ba))
    torch.cuda.empty_cache()
print('\nDONE')
