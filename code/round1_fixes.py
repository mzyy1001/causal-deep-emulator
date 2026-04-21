"""
Round 1 fixes: diagnostic analysis, rollout stability, causal speed validation.
"""
import os
import json
import time
import torch
import numpy as np

from config import NUM_SCALES, TEMPORAL_WINDOW
from model import CausalSpatiotemporalModel, build_multiscale_edges
from model_baseline import Graph_MLP
import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = './eval_results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPHERE_TEST = '../data/sphere_dataset/test/motion_1'
CHAR_ROOT = '../data/character_dataset'

# Load models
print("Loading models...")
causal = CausalSpatiotemporalModel()
causal.load_state_dict(torch.load('./weight/stage2_0009.weight', map_location='cpu'))
causal.eval().to(device)

# Also load Stage 1 only checkpoint for comparison
causal_s1 = CausalSpatiotemporalModel()
causal_s1.load_state_dict(torch.load('./weight/stage1_0059.weight', map_location='cpu'))
causal_s1.eval().to(device)

baseline = Graph_MLP()
baseline.load_state_dict(torch.load('./weight/_0000100.weight', map_location='cpu'))
baseline.eval().to(device)


def load_sequence_data(data_path):
    """Load topology and material for a sequence."""
    constraint_np = data_loader.loadData_Int(os.path.join(data_path, 'c'))
    constraint = torch.from_numpy(constraint_np).long().to(device)
    V = constraint.shape[0]

    adj_raw = data_loader.loadData_Int(os.path.join(data_path, 'adj'))
    adj_matrix = torch.from_numpy(adj_raw.reshape(V, -1)).long().to(device)
    ms_edges = build_multiscale_edges(adj_matrix, NUM_SCALES)

    k_data = data_loader.loadData_Float(os.path.join(data_path, 'k'))
    stiffness_val = k_data[1]
    k_data = np.expand_dims(k_data, axis=1) * 0.000001
    m_data = data_loader.loadData_Float(os.path.join(data_path, 'm'))
    m_data = np.expand_dims(m_data, axis=1) * 1000
    m_data[0] = 1.0
    stiffness = torch.from_numpy(k_data).float().to(device)
    mass = torch.from_numpy(m_data).float().to(device)

    frame_num = len([f for f in os.listdir(data_path) if f.startswith('x_')])
    return constraint, adj_matrix, ms_edges, stiffness, mass, stiffness_val, frame_num


def load_frame(data_path, k, frame_num):
    """Load input frames at position k."""
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
    return torch.stack(u_frames, dim=0), torch.stack(x_frames, dim=0)


# ============================================================================
# 1. DIAGNOSTIC: Stage 1 vs Stage 2 on sphere
# ============================================================================
print('\n' + '='*70)
print('  1. DIAGNOSTIC: Stage1 vs Stage2 vs Baseline on Sphere')
print('='*70)

diag_results = {}
for seq_id in ['1', '4', '7']:
    data_path = os.path.join(SPHERE_TEST, seq_id)
    constraint, adj_matrix, ms_edges, stiffness, mass, stiff_val, frame_num = load_sequence_data(data_path)
    free_mask = (constraint == 0)
    V = constraint.shape[0]

    mse_s1, mse_s2, mse_bl = [], [], []
    with torch.no_grad():
        for k in range(min(frame_num - 1, 100)):
            dyn, ref = load_frame(data_path, k, frame_num)

            # GT
            u_next = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % (k+1))).reshape(-1, 3)
            u_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_next], axis=0)
            u_next = torch.from_numpy(u_next).float().to(device)
            u_curr = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % k)).reshape(-1, 3)
            u_curr = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_curr], axis=0)
            u_curr = torch.from_numpy(u_curr).float().to(device)
            delta_gt = (u_next - u_curr)[free_mask]

            # Stage 1 causal
            d1 = causal_s1(constraint, dyn, ref, adj_matrix, stiffness, mass, ms_edges)
            mse_s1.append(((d1 - delta_gt)**2).mean().item())

            # Stage 2 causal
            d2 = causal(constraint, dyn, ref, adj_matrix, stiffness, mass, ms_edges)
            mse_s2.append(((d2 - delta_gt)**2).mean().item())

            # Baseline
            dyn_packed = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
            ref_packed = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
            db = baseline(constraint, dyn_packed, ref_packed, adj_matrix,
                          stiffness.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)
            mse_bl.append(((db - delta_gt)**2).mean().item())

    r = {
        'stiffness': stiff_val,
        'stage1_mse': float(np.mean(mse_s1)),
        'stage2_mse': float(np.mean(mse_s2)),
        'baseline_mse': float(np.mean(mse_bl)),
    }
    diag_results[seq_id] = r
    print('  Seq %s (K=%.0f): Stage1=%.6f  Stage2=%.6f  Baseline=%.6f' % (
        seq_id, stiff_val, r['stage1_mse'], r['stage2_mse'], r['baseline_mse']))

torch.cuda.empty_cache()

# ============================================================================
# 2. ROLLOUT STABILITY (autoregressive, 100 frames)
# ============================================================================
print('\n' + '='*70)
print('  2. ROLLOUT STABILITY (100-frame autoregressive)')
print('='*70)

rollout_results = {}
for seq_id in ['4']:
    data_path = os.path.join(SPHERE_TEST, seq_id)
    constraint, adj_matrix, ms_edges, stiffness, mass, stiff_val, frame_num = load_sequence_data(data_path)
    free_mask = (constraint == 0)
    V = constraint.shape[0]

    rollout_len = min(100, frame_num - TEMPORAL_WINDOW - 2)
    start_frame = TEMPORAL_WINDOW + 1

    for model_name, net, mtype in [('causal', causal, 'causal'), ('causal_s1', causal_s1, 'causal'), ('baseline', baseline, 'baseline')]:
        dyn, ref = load_frame(data_path, start_frame, frame_num)
        per_step_mse = []

        with torch.no_grad():
            for step in range(rollout_len):
                gt_frame = start_frame + step + 1
                if gt_frame >= frame_num:
                    break

                if mtype == 'baseline':
                    dyn_packed = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
                    ref_packed = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
                    delta_u = net(constraint, dyn_packed, ref_packed, adj_matrix,
                                  stiffness.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)
                else:
                    delta_u = net(constraint, dyn, ref, adj_matrix, stiffness, mass, ms_edges)

                # Predicted displacement
                u_pred = torch.zeros(V, 3, device=device)
                u_pred[free_mask] = dyn[0, free_mask] + delta_u
                u_pred[~free_mask] = 0.0

                # GT displacement
                u_gt = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % gt_frame)).reshape(-1, 3)
                u_gt = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_gt], axis=0)
                u_gt = torch.from_numpy(u_gt).float().to(device)

                mse = ((u_pred[free_mask] - u_gt[free_mask])**2).mean().item()
                per_step_mse.append(mse)

                # Load next reference
                x_next = data_loader.loadData_Float(os.path.join(data_path, 'x_%d' % gt_frame)).reshape(-1, 3)
                x_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), x_next], axis=0)
                x_next = torch.from_numpy(x_next).float().to(device)

                # Shift history
                new_dyn = torch.zeros_like(dyn)
                new_dyn[0] = u_pred
                new_dyn[1:] = dyn[:-1]
                new_ref = torch.zeros_like(ref)
                new_ref[0] = x_next
                new_ref[1:] = ref[:-1]
                dyn = new_dyn
                ref = new_ref

        rollout_results[model_name] = {
            'per_step_mse': [float(m) for m in per_step_mse],
            'avg_mse': float(np.mean(per_step_mse)),
            'final_mse': float(per_step_mse[-1]) if per_step_mse else 0,
            'max_mse': float(np.max(per_step_mse)) if per_step_mse else 0,
        }
        print('  %s: avg=%.6f  final=%.6f  max=%.6f' % (
            model_name, rollout_results[model_name]['avg_mse'],
            rollout_results[model_name]['final_mse'],
            rollout_results[model_name]['max_mse']))

torch.cuda.empty_cache()

# ============================================================================
# 3. CAUSAL SPEED ANALYSIS
# ============================================================================
print('\n' + '='*70)
print('  3. CAUSAL SPEED ANALYSIS')
print('='*70)

speed_results = {}

# Analyze learned velocities across different stiffness values
for seq_id in ['1', '4', '7']:
    data_path = os.path.join(SPHERE_TEST, seq_id)
    constraint, adj_matrix, ms_edges, stiffness, mass, stiff_val, frame_num = load_sequence_data(data_path)

    with torch.no_grad():
        theta = causal.form_vertex_properties(constraint, stiffness, mass)
        v = causal.causal_cone.velocity_mlp(theta)  # (V, 1)

        # Causal mask at different delays
        tau = torch.arange(1, TEMPORAL_WINDOW + 1, device=device, dtype=torch.float32)
        s_max = causal.causal_cone(theta, tau)  # (V, T)

        free_mask = (constraint == 0)
        v_free = v[free_mask].cpu().numpy().flatten()
        s_max_free = s_max[free_mask].cpu().numpy()

        speed_results[seq_id] = {
            'stiffness': float(stiff_val),
            'v_mean': float(np.mean(v_free)),
            'v_std': float(np.std(v_free)),
            'v_min': float(np.min(v_free)),
            'v_max': float(np.max(v_free)),
            's_max_tau1_mean': float(np.mean(s_max_free[:, 0])),
            's_max_tau5_mean': float(np.mean(s_max_free[:, -1])),
        }
        print('  Seq %s (K=%.0f): v=%.4f +/- %.4f [%.4f, %.4f]  s_max(1)=%.2f  s_max(5)=%.2f' % (
            seq_id, stiff_val,
            speed_results[seq_id]['v_mean'], speed_results[seq_id]['v_std'],
            speed_results[seq_id]['v_min'], speed_results[seq_id]['v_max'],
            speed_results[seq_id]['s_max_tau1_mean'],
            speed_results[seq_id]['s_max_tau5_mean']))

# Correlation: does velocity correlate with stiffness?
stiffnesses = [speed_results[s]['stiffness'] for s in ['1', '4', '7']]
velocities = [speed_results[s]['v_mean'] for s in ['1', '4', '7']]
if len(set(velocities)) > 1:
    corr = np.corrcoef(stiffnesses, velocities)[0, 1]
    print('\n  Stiffness-velocity correlation: %.4f' % corr)
else:
    print('\n  All velocities identical (stiffness has no effect on velocity)')

# Character speeds
print('\n  Character speeds:')
char_speeds = {}
for char in ['michelle', 'kaya', 'mousey']:
    motion = {'michelle': 'cross_jumps', 'kaya': 'zombie_scream', 'mousey': 'dancing_1'}[char]
    data_path = os.path.join(CHAR_ROOT, char, motion)
    if not os.path.isdir(data_path):
        continue
    constraint, adj_matrix, ms_edges, stiffness, mass, stiff_val, frame_num = load_sequence_data(data_path)
    with torch.no_grad():
        theta = causal.form_vertex_properties(constraint, stiffness, mass)
        v = causal.causal_cone.velocity_mlp(theta)
        free_mask = (constraint == 0)
        v_free = v[free_mask].cpu().numpy().flatten()
        char_speeds[char] = {
            'v_mean': float(np.mean(v_free)),
            'v_std': float(np.std(v_free)),
            'constrained_pct': float((constraint == 1).sum().item() / constraint.shape[0] * 100),
        }
        print('    %s: v=%.4f +/- %.4f  constrained=%.1f%%' % (
            char, char_speeds[char]['v_mean'], char_speeds[char]['v_std'],
            char_speeds[char]['constrained_pct']))

# ============================================================================
# 4. BASELINE ON CHARACTERS
# ============================================================================
print('\n' + '='*70)
print('  4. BASELINE ON CHARACTERS')
print('='*70)

ALL_CHARACTERS = {
    'michelle':  ['cross_jumps', 'gangnam_style'],
    'big_vegas': ['cross_jumps', 'cross_jumps_rotation'],
    'kaya':      ['dancing_running_man', 'zombie_scream'],
    'mousey':    ['dancing_1', 'swing_dancing_1'],
    'ortiz':     ['cross_jumps_rotation', 'jazz_dancing'],
}

baseline_char_results = []
for char, motions in ALL_CHARACTERS.items():
    for motion in motions:
        data_path = os.path.join(CHAR_ROOT, char, motion)
        if not os.path.isdir(data_path):
            continue
        constraint, adj_matrix, ms_edges, stiffness, mass, stiff_val, frame_num = load_sequence_data(data_path)
        free_mask = (constraint == 0)

        mse_list = []
        with torch.no_grad():
            for k in range(min(frame_num - 1, 100)):
                dyn, ref = load_frame(data_path, k, frame_num)

                u_next = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % (k+1))).reshape(-1, 3)
                u_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_next], axis=0)
                u_next = torch.from_numpy(u_next).float().to(device)
                u_curr = data_loader.loadData_Float(os.path.join(data_path, 'u_%d' % k)).reshape(-1, 3)
                u_curr = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_curr], axis=0)
                u_curr = torch.from_numpy(u_curr).float().to(device)
                delta_gt = (u_next - u_curr)[free_mask]

                dyn_packed = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
                ref_packed = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
                delta_b = baseline(constraint, dyn_packed, ref_packed, adj_matrix,
                                   stiffness.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)
                mse_list.append(((delta_b - delta_gt)**2).mean().item())

        avg_mse = float(np.mean(mse_list))
        baseline_char_results.append({
            'character': char, 'motion': motion, 'baseline_mse': avg_mse
        })
        print('  %s/%s: baseline MSE=%.8f' % (char, motion, avg_mse))
        torch.cuda.empty_cache()

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================
all_results = {
    'diagnostic': diag_results,
    'rollout': rollout_results,
    'causal_speeds': speed_results,
    'char_speeds': char_speeds,
    'baseline_characters': baseline_char_results,
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open(os.path.join(OUTPUT_DIR, 'round1_analysis.json'), 'w') as f:
    json.dump(all_results, f, indent=2, cls=NpEncoder)

print('\n' + '='*70)
print('  ALL ANALYSES COMPLETE')
print('='*70)
print('Results saved to %s' % os.path.join(OUTPUT_DIR, 'round1_analysis.json'))
