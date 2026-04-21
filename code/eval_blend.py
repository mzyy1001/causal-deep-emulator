"""Evaluate all blend ratios."""
import torch, numpy as np, os, sys
import config; config.NUM_SCALES = 1; config.USE_CAUSAL_CONE = True
import importlib, model as mm; importlib.reload(mm)
from model import CausalSpatiotemporalModel, build_multiscale_edges
from model_baseline import Graph_MLP
from config import TEMPORAL_WINDOW
import data_loader

device = torch.device('cuda')
GT = '../data/scaled_stiffness_dense'
CR = '../data/character_dataset'
SM = {str(i+1): s for i, s in enumerate([50000,100000,250000,500000,1000000,2500000,5000000,10000,25000,75000,150000,300000,750000,1500000,2000000,3000000,4000000,6000000,8000000,10000000])}
MS = [('mousey','dancing_1'),('mousey','swing_dancing_1'),('big_vegas','cross_jumps'),('big_vegas','cross_jumps_rotation'),('ortiz','cross_jumps_rotation'),('ortiz','jazz_dancing')]

bl = Graph_MLP()
bl.load_state_dict(torch.load('./weight_v7_baseline/best.weight', map_location='cpu'))
bl.eval().to(device)

blend = sys.argv[1]  # e.g. "60"
weight_path = './weight_blend_%s/best_stage1.weight' % blend

cone = CausalSpatiotemporalModel()
cone.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
cone.eval().to(device)

cw = 0; tot = 0
for ch, mo in MS:
    gb = os.path.join(GT, ch, mo); orig = os.path.join(CR, ch, mo)
    c = data_loader.loadData_Int(os.path.join(orig, 'c'))
    ct = torch.from_numpy(c).long().to(device); fr = (ct == 0); V = len(c)
    ad = torch.from_numpy(data_loader.loadData_Int(os.path.join(orig, 'adj')).reshape(V, -1)).long().to(device)
    ms = build_multiscale_edges(ad, 1)
    mr = data_loader.loadData_Float(os.path.join(orig, 'm'))
    mr = np.expand_dims(mr, 1) * 1000; mr[0] = 1.0
    ma = torch.from_numpy(mr).float().to(device)
    w = 0
    for sq in sorted([d for d in os.listdir(gb) if d.isdigit()]):
        gp = os.path.join(gb, sq)
        kr = data_loader.loadData_Float(os.path.join(gp, 'k'))
        k = torch.from_numpy(np.expand_dims(kr, 1) * 0.000001).float().to(device)
        fn = len([f for f in os.listdir(gp) if f.startswith('x_')])
        cm, bm = [], []
        with torch.no_grad():
            for i in range(min(fn - 1, 30)):
                uf, xf = [], []
                for t in range(TEMPORAL_WINDOW + 1):
                    ix = max(0, min(i - t, fn - 1))
                    u = data_loader.loadData_Float(os.path.join(gp, 'u_%d' % ix)).reshape(-1, 3)
                    u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
                    uf.append(torch.from_numpy(u).float().to(device))
                for t in range(-1, TEMPORAL_WINDOW):
                    ix = max(0, min(i - t, fn - 1))
                    x = data_loader.loadData_Float(os.path.join(gp, 'x_%d' % ix)).reshape(-1, 3)
                    x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
                    xf.append(torch.from_numpy(x).float().to(device))
                dy = torch.stack(uf, 0); rf = torch.stack(xf, 0)
                un = data_loader.loadData_Float(os.path.join(gp, 'u_%d' % (i + 1))).reshape(-1, 3)
                un = np.concatenate([np.zeros((1, 3), dtype=np.float64), un], axis=0)
                un = torch.from_numpy(un).float().to(device)
                uc = data_loader.loadData_Float(os.path.join(gp, 'u_%d' % i)).reshape(-1, 3)
                uc = np.concatenate([np.zeros((1, 3), dtype=np.float64), uc], axis=0)
                uc = torch.from_numpy(uc).float().to(device)
                gt = (un - uc)[fr]
                pc = cone(ct, dy, rf, ad, k, ma, ms)
                cm.append(((pc - gt) ** 2).mean().item())
                dp = torch.cat([dy[0], dy[1], dy[min(2, dy.shape[0]-1)]], dim=-1).unsqueeze(0)
                rp = torch.cat([rf[0], rf[1], rf[min(2, rf.shape[0]-1)]], dim=-1).unsqueeze(0)
                pb = bl(ct, dp, rp, ad, k.unsqueeze(0), ma.unsqueeze(0)).squeeze(0)
                bm.append(((pb - gt) ** 2).mean().item())
        if float(np.mean(cm)) < float(np.mean(bm)):
            w += 1; cw += 1
        tot += 1
    print('  %s/%s: %d/20' % (ch, mo, w))
print('Blend 0.%s: %d/%d (%.1f%%)' % (blend, cw, tot, cw / tot * 100))
