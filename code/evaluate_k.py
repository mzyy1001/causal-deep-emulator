"""
Stiffness (elastic K) robustness evaluation for the causal spatiotemporal deep emulator.

Evaluates model performance across different elastic stiffness values.
Each test sequence has a uniform stiffness; this script runs rollout on each
and reports per-stiffness MSE and physics energy.
Supports comparing two checkpoints (e.g., causal vs no-causal ablation).

Usage:
    # Evaluate across all stiffness sequences in test data:
    python evaluate_k.py --weight ./weight/stage2_0004.weight \
                         --data ../data/sphere_dataset/test/motion_1

    # Compare causal model vs original baseline:
    python evaluate_k.py --weight ./weight/stage2_0004.weight \
                         --weight2 ./weight/_0000100.weight \
                         --model_type causal --model_type2 baseline \
                         --data ../data/sphere_dataset/test/motion_1

    # Also sweep rollout horizon:
    python evaluate_k.py --weight ./weight/stage2_0004.weight \
                         --data ../data/sphere_dataset/test/motion_1 \
                         --rollout_steps 1,4,8
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from config import NUM_SCALES, TEMPORAL_WINDOW
from model import CausalSpatiotemporalModel, PhysicsLoss, build_multiscale_edges
from model_baseline import Graph_MLP
import data_loader


def load_model(weight_path, model_type='causal'):
    """Load a trained model checkpoint.

    Args:
        weight_path: path to .weight file
        model_type: 'causal' for CausalSpatiotemporalModel,
                    'baseline' for original Graph_MLP (Zheng et al.)
    """
    if model_type == 'baseline':
        net = Graph_MLP()
    else:
        net = CausalSpatiotemporalModel()
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    return net, model_type


def evaluate_sequence(net, seq_path, rollout_K, physics_loss_fn, model_type='causal'):
    """Run rollout on a single sequence and compute metrics.

    Args:
        net: trained model
        seq_path: path to sequence data (containing u_*, x_*, c, adj, k, m)
        rollout_K: autoregressive rollout horizon
        physics_loss_fn: PhysicsLoss instance
        model_type: 'causal' or 'baseline'

    Returns:
        dict with metrics
    """
    frame_num = len([f for f in os.listdir(seq_path) if f.startswith("x_")])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load topology and material
    constraint_np = data_loader.loadData_Int(os.path.join(seq_path, "c"))
    constraint = torch.from_numpy(constraint_np).long().to(device)
    free_mask = (constraint == 0)

    adj_raw = data_loader.loadData_Int(os.path.join(seq_path, "adj"))
    V = constraint.shape[0]
    adj_matrix = torch.from_numpy(adj_raw.reshape(V, -1)).long().to(device)
    ms_edges = build_multiscale_edges(adj_matrix, NUM_SCALES) if model_type == 'causal' else None

    k_raw = data_loader.loadData_Float(os.path.join(seq_path, "k"))
    stiffness_value = k_raw[1]  # raw stiffness (uniform per sequence)
    k_data = np.expand_dims(k_raw, axis=1) * 0.000001
    m_data = data_loader.loadData_Float(os.path.join(seq_path, "m"))
    m_data = np.expand_dims(m_data, axis=1) * 1000
    m_data[0] = 1.0
    stiffness = torch.from_numpy(k_data).float().to(device)
    mass = torch.from_numpy(m_data).float().to(device)

    per_step_mse = []
    per_step_physics = []
    total_mse = 0.0
    total_physics = 0.0
    num_windows = 0

    t_start = time.time()

    max_start = frame_num - rollout_K - TEMPORAL_WINDOW - 1
    if max_start < 0:
        max_start = 0

    with torch.no_grad():
        for start_frame in range(0, min(max_start + 1, frame_num - rollout_K - 1)):
            # Build input history at start_frame
            u_frames = []
            for tau in range(TEMPORAL_WINDOW + 1):
                idx = max(0, min(start_frame - tau, frame_num - 1))
                u = data_loader.loadData_Float(os.path.join(seq_path, f"u_{idx}")).reshape(-1, 3)
                u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
                u_frames.append(torch.from_numpy(u).float().to(device))

            x_frames = []
            for tau in range(-1, TEMPORAL_WINDOW):
                idx = max(0, min(start_frame - tau, frame_num - 1))
                x = data_loader.loadData_Float(os.path.join(seq_path, f"x_{idx}")).reshape(-1, 3)
                x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
                x_frames.append(torch.from_numpy(x).float().to(device))

            dyn = torch.stack(u_frames, dim=0)  # (T+1, V, 3)
            ref = torch.stack(x_frames, dim=0)  # (T+1, V, 3)

            step_mse_list = []
            step_phys_list = []

            pos_prev = ref[2] + dyn[1]
            pos_curr = ref[1] + dyn[0]

            cur_dyn = dyn.clone()
            cur_ref = ref.clone()

            for step in range(rollout_K):
                gt_frame = start_frame + step + 1
                if gt_frame >= frame_num:
                    break

                if model_type == 'baseline':
                    # Baseline expects (B, V, 9) dynamic and reference
                    # dynamic_f = [u(t), u(t-1), u(t-2)]
                    # reference_f = [x(t+1), x(t), x(t-1)]
                    dyn_packed = torch.cat([
                        cur_dyn[0], cur_dyn[1],
                        cur_dyn[min(2, cur_dyn.shape[0]-1)]
                    ], dim=-1).unsqueeze(0)  # (1, V, 9)
                    ref_packed = torch.cat([
                        cur_ref[0], cur_ref[1],
                        cur_ref[min(2, cur_ref.shape[0]-1)]
                    ], dim=-1).unsqueeze(0)  # (1, V, 9)
                    delta_u = net(constraint, dyn_packed, ref_packed,
                                  adj_matrix, stiffness.unsqueeze(0),
                                  mass.unsqueeze(0))
                    delta_u = delta_u.squeeze(0)  # (V_free, 3)
                else:
                    delta_u = net(constraint, cur_dyn, cur_ref, adj_matrix,
                                  stiffness, mass, ms_edges)

                u_pred = torch.zeros(V, 3, device=device)
                u_pred[free_mask] = cur_dyn[0, free_mask] + delta_u
                u_pred[~free_mask] = 0.0

                # Ground truth
                u_gt = data_loader.loadData_Float(
                    os.path.join(seq_path, f"u_{gt_frame}")).reshape(-1, 3)
                u_gt = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_gt], axis=0)
                u_gt = torch.from_numpy(u_gt).float().to(device)

                mse = ((u_pred[free_mask] - u_gt[free_mask]) ** 2).mean().item()
                step_mse_list.append(mse)

                x_ref_next = data_loader.loadData_Float(
                    os.path.join(seq_path, f"x_{gt_frame}")).reshape(-1, 3)
                x_ref_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), x_ref_next], axis=0)
                x_ref_next = torch.from_numpy(x_ref_next).float().to(device)

                pos_next = x_ref_next + u_pred
                phys = physics_loss_fn(pos_next, pos_curr, pos_prev, mass).item()
                step_phys_list.append(phys)

                pos_prev = pos_curr
                pos_curr = pos_next

                new_dyn = torch.zeros_like(cur_dyn)
                new_dyn[0] = u_pred
                new_dyn[1:] = cur_dyn[:-1]

                new_ref = torch.zeros_like(cur_ref)
                new_ref[0] = x_ref_next
                new_ref[1:] = cur_ref[:-1]

                cur_dyn = new_dyn
                cur_ref = new_ref

            if step_mse_list:
                while len(per_step_mse) < len(step_mse_list):
                    per_step_mse.append([])
                    per_step_physics.append([])
                for i, (m, p) in enumerate(zip(step_mse_list, step_phys_list)):
                    per_step_mse[i].append(m)
                    per_step_physics[i].append(p)
                total_mse += sum(step_mse_list)
                total_physics += sum(step_phys_list)
                num_windows += 1

    wall_time = time.time() - t_start

    avg_per_step_mse = [np.mean(s) if s else 0.0 for s in per_step_mse]
    avg_per_step_physics = [np.mean(s) if s else 0.0 for s in per_step_physics]

    return {
        'stiffness': stiffness_value,
        'rollout_K': rollout_K,
        'rollout_mse': total_mse / max(num_windows, 1),
        'physics_energy': total_physics / max(num_windows, 1),
        'per_step_mse': avg_per_step_mse,
        'per_step_physics': avg_per_step_physics,
        'num_windows': num_windows,
        'wall_time': wall_time,
    }


def discover_sequences(data_root):
    """Find all numbered sequence directories under data_root."""
    seqs = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and name.isdigit():
            seqs.append(path)
    return seqs


def evaluate_model(net, data_root, rollout_steps, physics_loss_fn, model_type='causal'):
    """Evaluate a model across all sequences and rollout horizons."""
    seq_paths = discover_sequences(data_root)
    if not seq_paths:
        print(f"  No sequences found in {data_root}")
        return []

    results = []
    for seq_path in seq_paths:
        seq_name = os.path.basename(seq_path)
        for K in rollout_steps:
            print(f"  Seq {seq_name}, rollout K={K}...", end=" ", flush=True)
            r = evaluate_sequence(net, seq_path, K, physics_loss_fn, model_type)
            print(f"stiffness={r['stiffness']:.0f}  MSE={r['rollout_mse']:.8f}  "
                  f"PhysE={r['physics_energy']:.4f}  ({r['wall_time']:.1f}s)")
            results.append(r)
    return results


def print_results(results, label=""):
    """Pretty-print evaluation results grouped by stiffness."""
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    print(f"\n{'Stiffness':>12} {'K':>4} | {'Rollout MSE':>12} | {'Physics E':>12} | "
          f"{'Time(s)':>8} | Per-step MSE")
    print(f"{'-'*12} {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*30}")
    for r in results:
        step_str = ', '.join(f'{m:.6f}' for m in r['per_step_mse'][:6])
        print(f"{r['stiffness']:>12.0f} {r['rollout_K']:>4} | {r['rollout_mse']:>12.8f} | "
              f"{r['physics_energy']:>12.4f} | {r['wall_time']:>8.2f} | [{step_str}]")


def save_csv(results, path, label=""):
    """Save results to CSV (append mode for comparison)."""
    max_steps = max((len(r['per_step_mse']) for r in results), default=0)
    fieldnames = ['label', 'stiffness', 'rollout_K', 'rollout_mse',
                  'physics_energy', 'num_windows', 'wall_time']
    fieldnames += [f'step_{i+1}_mse' for i in range(max_steps)]
    fieldnames += [f'step_{i+1}_physics' for i in range(max_steps)]

    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            row = {
                'label': label,
                'stiffness': r['stiffness'],
                'rollout_K': r['rollout_K'],
                'rollout_mse': r['rollout_mse'],
                'physics_energy': r['physics_energy'],
                'num_windows': r['num_windows'],
                'wall_time': r['wall_time'],
            }
            for i, m in enumerate(r['per_step_mse']):
                row[f'step_{i+1}_mse'] = m
            for i, p in enumerate(r['per_step_physics']):
                row[f'step_{i+1}_physics'] = p
            writer.writerow(row)
    print(f"  Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Stiffness robustness evaluation for the deep emulator')
    parser.add_argument('--weight', required=True, help='Path to model checkpoint')
    parser.add_argument('--weight2', default=None,
                        help='Optional second checkpoint for comparison')
    parser.add_argument('--model_type', default='causal', choices=['causal', 'baseline'],
                        help='Model type for first checkpoint (default: causal)')
    parser.add_argument('--model_type2', default='baseline', choices=['causal', 'baseline'],
                        help='Model type for second checkpoint (default: baseline)')
    parser.add_argument('--data', required=True,
                        help='Path to test motion dir containing numbered sequence subdirs')
    parser.add_argument('--rollout_steps', default='1,4,8',
                        help='Comma-separated rollout horizons to evaluate (default: 1,4,8)')
    parser.add_argument('--output', default='stiffness_robustness.csv',
                        help='Output CSV path')
    parser.add_argument('--label', default='causal', help='Label for first checkpoint')
    parser.add_argument('--label2', default='baseline', help='Label for second checkpoint')
    args = parser.parse_args()

    rollout_steps = [int(k) for k in args.rollout_steps.split(',')]
    physics_loss_fn = PhysicsLoss()

    # Evaluate first checkpoint
    print(f"Loading model ({args.model_type}): {args.weight}")
    net1, mtype1 = load_model(args.weight, args.model_type)
    results1 = evaluate_model(net1, args.data, rollout_steps, physics_loss_fn, mtype1)
    print_results(results1, label=args.label)
    save_csv(results1, args.output, label=args.label)

    # Evaluate second checkpoint if provided
    if args.weight2:
        print(f"\nLoading model ({args.model_type2}): {args.weight2}")
        net2, mtype2 = load_model(args.weight2, args.model_type2)
        results2 = evaluate_model(net2, args.data, rollout_steps, physics_loss_fn, mtype2)
        print_results(results2, label=args.label2)
        save_csv(results2, args.output, label=args.label2)

        # Print comparison for each (stiffness, K) pair
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON: {args.label} vs {args.label2}")
        print(f"{'=' * 70}")
        print(f"{'Stiffness':>12} {'K':>4} | {'MSE ratio':>10} | {'Physics ratio':>14}")
        print(f"{'-'*12} {'-'*4}-+-{'-'*10}-+-{'-'*14}")

        # Match by (stiffness, rollout_K)
        r2_map = {(r['stiffness'], r['rollout_K']): r for r in results2}
        for r1 in results1:
            key = (r1['stiffness'], r1['rollout_K'])
            if key in r2_map:
                r2 = r2_map[key]
                mse_ratio = r1['rollout_mse'] / max(r2['rollout_mse'], 1e-12)
                phys_ratio = r1['physics_energy'] / max(r2['physics_energy'], 1e-12)
                print(f"{r1['stiffness']:>12.0f} {r1['rollout_K']:>4} | "
                      f"{mse_ratio:>10.4f} | {phys_ratio:>14.4f}")
        print(f"\n  (ratio < 1 means {args.label} is better)")


if __name__ == '__main__':
    main()
