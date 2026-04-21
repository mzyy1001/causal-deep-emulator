"""
Evaluate stiffness generalization on character FEM ground truth.

Models trained on sphere (stiffness 50K-5M) are tested on characters
at 6 different stiffness values (10K-5M). This tests whether the model
can generalize to:
1. Unseen geometry (characters vs sphere)
2. Unseen stiffness on that geometry
3. Stiffness outside training range (10K < 50K minimum training value)

Usage:
    python eval_fem_stiffness.py
"""
import os
import torch
import numpy as np
import json

import config
# Try scales=1 first, fall back to 3
NUM_SCALES_EVAL = int(os.environ.get('NUM_SCALES_EVAL', '1'))
config.NUM_SCALES = NUM_SCALES_EVAL

import importlib
import model as model_module
importlib.reload(model_module)
from model import CausalSpatiotemporalModel, build_multiscale_edges
from model_baseline import Graph_MLP
from config import TEMPORAL_WINDOW
import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEM_ROOT = '../data/vega_stiffness'
STIFFNESS_MAP = {
    '1': 50000, '2': 200000, '3': 500000, '4': 2000000, '5': 5000000,
    '6': 25000, '7': 75000, '8': 150000, '9': 300000, '10': 750000,
    '11': 1000000, '12': 3000000, '13': 8000000,
}
TRAINING_STIFFNESS = {50000, 200000, 500000, 2000000, 5000000}

CHARACTERS = [
    ('mousey', 'dancing_1'),
    ('mousey', 'swing_dancing_1'),
    ('big_vegas', 'cross_jumps'),
    ('big_vegas', 'cross_jumps_rotation'),
    ('ortiz', 'cross_jumps_rotation'),
    ('ortiz', 'jazz_dancing'),
]
# Note: michelle and kaya excluded due to VegaFEM segfault on their meshes

CHAR_ROOT = '../data/character_dataset'


def evaluate_on_fem(net, data_path, orig_motion_path, model_type='causal', num_scales=1):
    """Evaluate single-step MSE against FEM ground truth.

    Uses topology (c, adj) from original motion data but stiffness (k)
    and displacements (u_*, x_*) from FEM data.
    """
    frame_num = len([f for f in os.listdir(data_path) if f.startswith('x_')])
    if frame_num < 3:
        return None

    # Use original topology (what model was trained with)
    c = data_loader.loadData_Int(os.path.join(orig_motion_path, 'c'))
    constraint = torch.from_numpy(c).long().to(device)
    free = (constraint == 0)
    V = len(c)

    adj = torch.from_numpy(
        data_loader.loadData_Int(os.path.join(orig_motion_path, 'adj')).reshape(V, -1)
    ).long().to(device)
    ms_edges = build_multiscale_edges(adj, num_scales) if model_type != 'baseline' else None

    # Use FEM stiffness (the test variable) — pad to match V if needed
    k_raw = data_loader.loadData_Float(os.path.join(data_path, 'k'))
    stiffness_val = k_raw[1] if len(k_raw) > 1 else k_raw[0]
    if len(k_raw) < V:
        k_raw = np.concatenate([k_raw[:1], k_raw])  # prepend padding
    k = torch.from_numpy(np.expand_dims(k_raw[:V], 1) * 0.000001).float().to(device)

    # Use original mass
    m_raw = data_loader.loadData_Float(os.path.join(orig_motion_path, 'm'))
    m_raw = np.expand_dims(m_raw, 1) * 1000
    m_raw[0] = 1.0
    mass = torch.from_numpy(m_raw).float().to(device)

    mse_list = []
    with torch.no_grad():
        for i in range(min(frame_num - 1, 40)):
            # Build input frames
            uf = []
            for tau in range(TEMPORAL_WINDOW + 1):
                idx = max(0, min(i - tau, frame_num - 1))
                u = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % idx)).reshape(-1, 3)
                # No padding row needed — FEM data already has V vertices
                if len(u) == V - 1:
                    u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
                uf.append(torch.from_numpy(u).float().to(device))

            xf = []
            for tau in range(-1, TEMPORAL_WINDOW):
                idx = max(0, min(i - tau, frame_num - 1))
                x = data_loader.loadData_Float(os.path.join(data_path, 'x_%d' % idx)).reshape(-1, 3)
                if len(x) == V - 1:
                    x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
                xf.append(torch.from_numpy(x).float().to(device))

            dyn = torch.stack(uf, 0)
            ref = torch.stack(xf, 0)

            # GT delta
            u_next = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % (i + 1))).reshape(-1, 3)
            u_curr = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % i)).reshape(-1, 3)
            if len(u_next) == V - 1:
                u_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_next], axis=0)
                u_curr = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_curr], axis=0)
            u_next = torch.from_numpy(u_next).float().to(device)
            u_curr_t = torch.from_numpy(u_curr).float().to(device)
            delta_gt = (u_next - u_curr_t)[free]

            # Model prediction
            if model_type == 'baseline':
                dp = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0] - 1)]], dim=-1).unsqueeze(0)
                rp = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0] - 1)]], dim=-1).unsqueeze(0)
                delta_pred = net(constraint, dp, rp, adj,
                                 k.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)
            else:
                delta_pred = net(constraint, dyn, ref, adj, k, mass, ms_edges)

            mse = ((delta_pred - delta_gt) ** 2).mean().item()
            mse_list.append(mse)

    if not mse_list:
        return None
    return {
        'stiffness': float(stiffness_val),
        'mse': float(np.mean(mse_list)),
        'frames': len(mse_list),
    }


def main():
    # Load models
    models = {}

    # Try to load causal models
    for name, path, scales in [
        ('Ours', './weight_final/best_stage1.weight', 1),
    ]:
        if os.path.exists(path):
            config.NUM_SCALES = scales
            importlib.reload(model_module)
            from model import CausalSpatiotemporalModel as CSM
            net = CSM()
            net.load_state_dict(torch.load(path, map_location='cpu'))
            net.eval().to(device)
            models[name] = (net, 'causal', scales)
            print('Loaded %s: %s (scales=%d)' % (name, path, scales))

    # Baseline
    bl_path = './weight_baseline_retrained/best.weight'
    if os.path.exists(bl_path):
        bl = Graph_MLP()
        bl.load_state_dict(torch.load(bl_path, map_location='cpu'))
        bl.eval().to(device)
        models['Baseline'] = (bl, 'baseline', 0)
        print('Loaded Baseline: %s' % bl_path)

    print('\nModels: %s' % list(models.keys()))

    # Evaluate
    all_results = {}
    for char, motion in CHARACTERS:
        fem_path = os.path.join(FEM_ROOT, char, motion)
        if not os.path.isdir(fem_path):
            print('SKIP: %s not found' % fem_path)
            continue

        key = '%s/%s' % (char, motion)
        orig_motion_path = os.path.join(CHAR_ROOT, char, motion)
        all_results[key] = {}
        print('\n' + '=' * 70)
        print('  %s' % key)
        print('=' * 70)

        # Header
        model_names = list(models.keys())
        header = '%12s %8s' % ('Stiffness', 'InTrain?')
        for mn in model_names:
            header += ' %12s' % mn
        print(header)
        print('-' * (22 + 14 * len(model_names)))

        seq_dirs = sorted([d for d in os.listdir(fem_path) if d.isdigit()])
        for seq_dir in seq_dirs:
            seq_path = os.path.join(fem_path, seq_dir)
            stiff = STIFFNESS_MAP.get(seq_dir, 0)
            in_train = 'YES' if stiff in TRAINING_STIFFNESS else 'NO'

            row = '%12.0f %8s' % (stiff, in_train)
            for mn in model_names:
                net, mtype, scales = models[mn]
                config.NUM_SCALES = scales if scales > 0 else 1
                importlib.reload(model_module)

                r = evaluate_on_fem(net, seq_path, orig_motion_path, model_type=mtype, num_scales=scales if scales > 0 else 1)
                if r:
                    row += ' %12.6f' % r['mse']
                    if mn not in all_results[key]:
                        all_results[key][mn] = []
                    all_results[key][mn].append(r)
                else:
                    row += ' %12s' % 'ERR'
            print(row)

        torch.cuda.empty_cache()

    # Summary
    print('\n' + '=' * 70)
    print('  SUMMARY: Average MSE by stiffness category')
    print('=' * 70)

    for key in all_results:
        print('\n  %s:' % key)
        for mn in all_results[key]:
            results = all_results[key][mn]
            in_train = [r for r in results if r['stiffness'] in TRAINING_STIFFNESS]
            out_train = [r for r in results if r['stiffness'] not in TRAINING_STIFFNESS]

            avg_in = np.mean([r['mse'] for r in in_train]) if in_train else 0
            avg_out = np.mean([r['mse'] for r in out_train]) if out_train else 0
            avg_all = np.mean([r['mse'] for r in results])

            print('    %s: in-train=%.6f  out-train=%.6f  all=%.6f' % (
                mn, avg_in, avg_out, avg_all))

    # Save
    os.makedirs('./eval_results/', exist_ok=True)
    with open('./eval_results/fem_stiffness_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nSaved to ./eval_results/fem_stiffness_results.json')


if __name__ == '__main__':
    main()
