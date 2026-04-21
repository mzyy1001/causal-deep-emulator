"""
Comprehensive evaluation: stiffness robustness, multi-motion, resource usage.
Compares: causal model vs ablation (K=3 checkpoint) vs original baseline.
"""
import os
import sys
import time
import json

import torch
import numpy as np

from config import NUM_SCALES, TEMPORAL_WINDOW
from model import CausalSpatiotemporalModel, PhysicsLoss, build_multiscale_edges
from model_baseline import Graph_MLP
import data_loader

# --- Paths ---
CAUSAL_WEIGHT = './weight_v2/stage2_0029.weight'
ABLATION_WEIGHT = './weight/stage2_0009.weight'
BASELINE_WEIGHT = './weight/_0000100.weight'
SPHERE_TEST = '../data/sphere_dataset/test/motion_1'
CHAR_ROOT = '../data/character_dataset'
OUTPUT_DIR = './eval_results/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_CHARACTERS = {
    'michelle':  ['cross_jumps', 'gangnam_style'],
    'big_vegas': ['cross_jumps', 'cross_jumps_rotation'],
    'kaya':      ['dancing_running_man', 'zombie_scream'],
    'mousey':    ['dancing_1', 'swing_dancing_1'],
    'ortiz':     ['cross_jumps_rotation', 'jazz_dancing'],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_causal_model(path):
    net = CausalSpatiotemporalModel()
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()
    return net.to(device)


def load_baseline_model(path):
    net = Graph_MLP()
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()
    return net.to(device)


def measure_inference_time(net, data_path, num_frames=50, model_type='causal'):
    """Measure per-frame inference time and peak GPU memory."""
    frame_num = len([f for f in os.listdir(data_path) if f.startswith('x_')])
    num_frames = min(num_frames, frame_num - 1)

    constraint_np = data_loader.loadData_Int(os.path.join(data_path, 'c'))
    constraint = torch.from_numpy(constraint_np).long().to(device)
    adj_raw = data_loader.loadData_Int(os.path.join(data_path, 'adj'))
    V = constraint.shape[0]
    adj_matrix = torch.from_numpy(adj_raw.reshape(V, -1)).long().to(device)

    k_data = data_loader.loadData_Float(os.path.join(data_path, 'k'))
    k_data = np.expand_dims(k_data, axis=1) * 0.000001
    m_data = data_loader.loadData_Float(os.path.join(data_path, 'm'))
    m_data = np.expand_dims(m_data, axis=1) * 1000
    m_data[0] = 1.0
    stiffness = torch.from_numpy(k_data).float().to(device)
    mass = torch.from_numpy(m_data).float().to(device)

    ms_edges = build_multiscale_edges(adj_matrix, NUM_SCALES) if model_type != 'baseline' else None

    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for k in range(num_frames):
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

            torch.cuda.synchronize()
            t0 = time.time()

            if model_type == 'baseline':
                dyn_packed = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
                ref_packed = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
                _ = net(constraint, dyn_packed, ref_packed, adj_matrix,
                        stiffness.unsqueeze(0), mass.unsqueeze(0))
            else:
                _ = net(constraint, dyn, ref, adj_matrix, stiffness, mass, ms_edges)

            torch.cuda.synchronize()
            times.append(time.time() - t0)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = np.mean(times[5:]) if len(times) > 5 else np.mean(times)
    return {
        'avg_ms': float(avg_time * 1000),
        'fps': float(1.0 / avg_time),
        'peak_gpu_mb': float(peak_mem),
        'num_frames': num_frames,
    }


def evaluate_rollout_mse(net, data_path, model_type='causal'):
    """Compute single-step MSE across frames."""
    frame_num = len([f for f in os.listdir(data_path) if f.startswith('x_')])
    constraint_np = data_loader.loadData_Int(os.path.join(data_path, 'c'))
    constraint = torch.from_numpy(constraint_np).long().to(device)
    free_mask = (constraint == 0)
    V = constraint.shape[0]

    adj_raw = data_loader.loadData_Int(os.path.join(data_path, 'adj'))
    adj_matrix = torch.from_numpy(adj_raw.reshape(V, -1)).long().to(device)
    ms_edges = build_multiscale_edges(adj_matrix, NUM_SCALES) if model_type != 'baseline' else None

    k_data = data_loader.loadData_Float(os.path.join(data_path, 'k'))
    stiffness_val = float(k_data[1])
    k_data = np.expand_dims(k_data, axis=1) * 0.000001
    m_data = data_loader.loadData_Float(os.path.join(data_path, 'm'))
    m_data = np.expand_dims(m_data, axis=1) * 1000
    m_data[0] = 1.0
    stiffness = torch.from_numpy(k_data).float().to(device)
    mass = torch.from_numpy(m_data).float().to(device)

    mse_list = []
    with torch.no_grad():
        for k in range(min(frame_num - 1, 100)):
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
                dyn_packed = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
                ref_packed = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
                delta_u = net(constraint, dyn_packed, ref_packed, adj_matrix,
                              stiffness.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)
            else:
                delta_u = net(constraint, dyn, ref, adj_matrix, stiffness, mass, ms_edges)

            # GT
            u_gt_next = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % (k+1))).reshape(-1, 3)
            u_gt_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_gt_next], axis=0)
            u_gt_next = torch.from_numpy(u_gt_next).float().to(device)
            u_gt_curr = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % k)).reshape(-1, 3)
            u_gt_curr = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_gt_curr], axis=0)
            u_gt_curr = torch.from_numpy(u_gt_curr).float().to(device)
            delta_gt = (u_gt_next - u_gt_curr)[free_mask]

            mse = float(((delta_u - delta_gt) ** 2).mean().item())
            mse_list.append(mse)

    return {
        'stiffness': stiffness_val,
        'single_step_mse': float(np.mean(mse_list)),
        'num_frames': len(mse_list),
    }


# ============================================================================
# MAIN
# ============================================================================

print('=' * 70)
print('  COMPREHENSIVE EVALUATION')
print('=' * 70)

results = {'models': {}, 'stiffness': {}, 'characters': {}, 'resource': {}}

# --- Load models ---
models = {}
for name, path, mtype in [
    ('causal', CAUSAL_WEIGHT, 'causal'),
    ('ablation', ABLATION_WEIGHT, 'causal'),
]:
    if os.path.exists(path):
        print('\nLoading %s: %s' % (name, path))
        models[name] = (load_causal_model(path), mtype)
    else:
        print('\nSKIP %s: %s not found' % (name, path))

if os.path.exists(BASELINE_WEIGHT):
    print('\nLoading baseline: %s' % BASELINE_WEIGHT)
    models['baseline'] = (load_baseline_model(BASELINE_WEIGHT), 'baseline')

print('\nModels loaded: %s' % list(models.keys()))

# --- 1. Resource usage ---
print('\n' + '=' * 70)
print('  1. RESOURCE USAGE')
print('=' * 70)

test_seq_path = os.path.join(SPHERE_TEST, '1')
for model_name, (net, mtype) in models.items():
    print('\n  %s:' % model_name)
    r = measure_inference_time(net, test_seq_path, num_frames=50, model_type=mtype)
    results['resource'][model_name] = r
    print('    Avg inference: %.2f ms/frame' % r['avg_ms'])
    print('    FPS: %.1f' % r['fps'])
    print('    Peak GPU memory: %.0f MB' % r['peak_gpu_mb'])
    torch.cuda.empty_cache()

# --- 2. Stiffness robustness ---
print('\n' + '=' * 70)
print('  2. STIFFNESS ROBUSTNESS')
print('=' * 70)

seq_dirs = sorted([d for d in os.listdir(SPHERE_TEST) if d.isdigit()])
for model_name, (net, mtype) in models.items():
    print('\n  %s:' % model_name)
    stiffness_results = []
    for seq_dir in seq_dirs:
        seq_path = os.path.join(SPHERE_TEST, seq_dir)
        r = evaluate_rollout_mse(net, seq_path, model_type=mtype)
        stiffness_results.append(r)
        print('    Seq %s: stiffness=%.0f  MSE=%.8f' % (seq_dir, r['stiffness'], r['single_step_mse']))
    results['stiffness'][model_name] = stiffness_results
    torch.cuda.empty_cache()

# --- 3. Character generalization ---
print('\n' + '=' * 70)
print('  3. CHARACTER GENERALIZATION')
print('=' * 70)

for model_name, (net, mtype) in models.items():
    if mtype == 'baseline':
        continue
    print('\n  %s:' % model_name)
    char_results = []
    for char, motions in ALL_CHARACTERS.items():
        for motion in motions:
            motion_path = os.path.join(CHAR_ROOT, char, motion)
            if not os.path.isdir(motion_path):
                continue
            frame_num = len([f for f in os.listdir(motion_path) if f.startswith('x_')])
            if frame_num < 10:
                continue
            r = evaluate_rollout_mse(net, motion_path, model_type=mtype)
            char_results.append({
                'character': char,
                'motion': motion,
                'frames': frame_num,
                'single_step_mse': r['single_step_mse'],
            })
            print('    %s/%s: MSE=%.8f (%d frames)' % (char, motion, r['single_step_mse'], frame_num))
    results['characters'][model_name] = char_results
    torch.cuda.empty_cache()

# --- 4. Model sizes ---
print('\n' + '=' * 70)
print('  4. MODEL SIZES')
print('=' * 70)

for model_name, (net, mtype) in models.items():
    params = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('  %s: %s params (%s trainable)' % (model_name, '{:,}'.format(params), '{:,}'.format(trainable)))
    results['models'][model_name] = {'params': params, 'trainable': trainable}

# --- Save JSON ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

json_path = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print('\nResults saved to %s' % json_path)

# --- Summary ---
print('\n' + '=' * 70)
print('  SUMMARY')
print('=' * 70)

print('\n  Resource Usage:')
print('  %-12s %10s %8s %8s %12s' % ('Model', 'ms/frame', 'FPS', 'GPU MB', 'Params'))
print('  ' + '-'*12 + ' ' + '-'*10 + ' ' + '-'*8 + ' ' + '-'*8 + ' ' + '-'*12)
for name in models:
    res = results['resource'].get(name, {})
    mod = results['models'].get(name, {})
    print('  %-12s %10.2f %8.1f %8.0f %12s' % (
        name, res.get('avg_ms', 0), res.get('fps', 0),
        res.get('peak_gpu_mb', 0), '{:,}'.format(mod.get('params', 0))))

print('\n  Stiffness Robustness (single-step MSE):')
model_names = list(models.keys())
header = '  %12s' % 'Stiffness'
for n in model_names:
    header += ' %12s' % n
print(header)
print('  ' + '-'*12 + (' ' + '-'*12) * len(model_names))
if results['stiffness']:
    first_model = list(results['stiffness'].keys())[0]
    for i, r in enumerate(results['stiffness'][first_model]):
        row = '  %12.0f' % r['stiffness']
        for n in model_names:
            sr = results['stiffness'].get(n, [])
            if i < len(sr):
                row += ' %12.8f' % sr[i]['single_step_mse']
            else:
                row += ' %12s' % 'N/A'
        print(row)

print('\n  Character Generalization (single-step MSE):')
for name in ['causal', 'ablation']:
    cr = results['characters'].get(name, [])
    if cr:
        avg = sum(c['single_step_mse'] for c in cr) / len(cr)
        print('  %s: avg MSE = %.8f across %d motions' % (name, avg, len(cr)))

print('\nDone!')
