"""
Stiffness robustness on characters: feed different stiffness values to the model
for the same character motion and measure how predictions change.

Without FEM ground truth at different stiffness, we measure:
1. Prediction stability: how much does MSE change as stiffness varies?
2. Physical plausibility: stiffer material should produce smaller displacements
3. Causal vs baseline comparison: which model responds more physically to stiffness?

Usage:
    python eval_stiffness_char.py --weight ./weight_v2/stage2_0009.weight
"""
import argparse
import os
import json
import torch
import numpy as np

from config import NUM_SCALES, TEMPORAL_WINDOW
from model import CausalSpatiotemporalModel, build_multiscale_edges
from model_baseline import Graph_MLP
import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHAR_ROOT = '../data/character_dataset'

# Stiffness values to test (original training range + extrapolations)
STIFFNESS_VALUES = [10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]

CHARACTERS = [
    ('michelle', 'cross_jumps'),
    ('mousey', 'dancing_1'),
    ('mousey', 'swing_dancing_1'),
    ('kaya', 'zombie_scream'),
    ('big_vegas', 'cross_jumps'),
    ('ortiz', 'jazz_dancing'),
]


def evaluate_with_stiffness(net, data_path, stiffness_override, model_type='causal'):
    """Run single-step evaluation with overridden stiffness value."""
    frame_num = len([f for f in os.listdir(data_path) if f.startswith('x_')])
    constraint_np = data_loader.loadData_Int(os.path.join(data_path, 'c'))
    constraint = torch.from_numpy(constraint_np).long().to(device)
    free_mask = (constraint == 0)
    V = constraint.shape[0]

    adj_raw = data_loader.loadData_Int(os.path.join(data_path, 'adj'))
    adj_matrix = torch.from_numpy(adj_raw.reshape(V, -1)).long().to(device)
    ms_edges = build_multiscale_edges(adj_matrix, NUM_SCALES) if model_type != 'baseline' else None

    # Override stiffness with the test value
    k_data = np.full((V, 1), stiffness_override * 0.000001, dtype=np.float64)
    m_data = data_loader.loadData_Float(os.path.join(data_path, 'm'))
    m_data = np.expand_dims(m_data, axis=1) * 1000
    m_data[0] = 1.0
    stiffness = torch.from_numpy(k_data).float().to(device)
    mass = torch.from_numpy(m_data).float().to(device)

    pred_magnitudes = []
    mse_vs_original = []

    with torch.no_grad():
        for k in range(min(frame_num - 1, 50)):
            u_frames = []
            for tau in range(TEMPORAL_WINDOW + 1):
                idx = max(0, min(k - tau, frame_num - 1))
                u = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % idx)).reshape(-1, 3)
                u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
                u_frames.append(torch.from_numpy(u).float().to(device))
            x_frames = []
            for tau in range(-1, TEMPORAL_WINDOW):
                idx = max(0, min(k - tau, frame_num - 1))
                x = data_loader.loadData_Float(os.path.join(data_path, 'x_%d' % idx)).reshape(-1, 3)
                x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
                x_frames.append(torch.from_numpy(x).float().to(device))

            dyn = torch.stack(u_frames, dim=0)
            ref = torch.stack(x_frames, dim=0)

            if model_type == 'baseline':
                dp = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
                rp = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
                delta_u = net(constraint, dp, rp, adj_matrix,
                              stiffness.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)
            else:
                delta_u = net(constraint, dyn, ref, adj_matrix, stiffness, mass, ms_edges)

            # Prediction magnitude (how much the model predicts to move)
            pred_mag = delta_u.norm(dim=-1).mean().item()
            pred_magnitudes.append(pred_mag)

            # MSE vs original GT (at original stiffness=50000)
            u_next = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % (k+1))).reshape(-1, 3)
            u_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_next], axis=0)
            u_next = torch.from_numpy(u_next).float().to(device)
            u_curr = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % k)).reshape(-1, 3)
            u_curr = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_curr], axis=0)
            u_curr = torch.from_numpy(u_curr).float().to(device)
            delta_gt = (u_next - u_curr)[free_mask]
            mse = ((delta_u - delta_gt) ** 2).mean().item()
            mse_vs_original.append(mse)

    return {
        'avg_pred_magnitude': float(np.mean(pred_magnitudes)),
        'mse_vs_gt50k': float(np.mean(mse_vs_original)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', required=True)
    parser.add_argument('--weight_baseline', default='./weight/_0000100.weight')
    parser.add_argument('--output', default='./eval_results/stiffness_char.json')
    args = parser.parse_args()

    # Load models
    models = {}
    print('Loading causal: %s' % args.weight)
    causal = CausalSpatiotemporalModel()
    causal.load_state_dict(torch.load(args.weight, map_location='cpu'))
    causal.eval().to(device)
    models['causal'] = (causal, 'causal')

    if os.path.exists(args.weight_baseline):
        print('Loading baseline: %s' % args.weight_baseline)
        baseline = Graph_MLP()
        baseline.load_state_dict(torch.load(args.weight_baseline, map_location='cpu'))
        baseline.eval().to(device)
        models['baseline'] = (baseline, 'baseline')

    results = {}

    for char, motion in CHARACTERS:
        data_path = os.path.join(CHAR_ROOT, char, motion)
        if not os.path.isdir(data_path):
            continue
        key = '%s/%s' % (char, motion)
        print('\n=== %s ===' % key)
        results[key] = {}

        for model_name, (net, mtype) in models.items():
            print('  %s:' % model_name)
            stiff_results = []
            for stiff_val in STIFFNESS_VALUES:
                r = evaluate_with_stiffness(net, data_path, stiff_val, mtype)
                stiff_results.append({
                    'stiffness': stiff_val,
                    'pred_magnitude': r['avg_pred_magnitude'],
                    'mse_vs_gt50k': r['mse_vs_gt50k'],
                })
                print('    K=%8d: pred_mag=%.6f  mse=%.6f' % (
                    stiff_val, r['avg_pred_magnitude'], r['mse_vs_gt50k']))

            results[key][model_name] = stiff_results
            torch.cuda.empty_cache()

    # Summary: physical plausibility check
    # Stiffer material → smaller displacement predicted?
    print('\n' + '='*70)
    print('  PHYSICAL PLAUSIBILITY CHECK')
    print('  (stiffer material should produce smaller displacements)')
    print('='*70)

    for key in results:
        print('\n  %s:' % key)
        for model_name in results[key]:
            mags = [r['pred_magnitude'] for r in results[key][model_name]]
            stiffs = [r['stiffness'] for r in results[key][model_name]]
            # Check if magnitude decreases as stiffness increases
            monotonic = all(mags[i] >= mags[i+1] for i in range(len(mags)-1))
            ratio = mags[0] / max(mags[-1], 1e-10)  # soft/stiff ratio
            corr = float(np.corrcoef(np.log10(stiffs), mags)[0, 1])
            print('    %s: soft/stiff ratio=%.2f  monotonic=%s  corr=%.3f' % (
                model_name, ratio, monotonic, corr))
            print('      K=10K: %.6f  K=50K: %.6f  K=5M: %.6f' % (
                mags[0], mags[2], mags[-2]))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResults saved to %s' % args.output)


if __name__ == '__main__':
    main()
