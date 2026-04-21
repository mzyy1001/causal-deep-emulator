"""
Testing / rollout prediction for the causal spatiotemporal deep emulator.

Usage:
    # Single motion (default):
    python test.py --character michelle --motion cross_jumps --weight ./weight/stage2_0004.weight

    # Batch evaluation across all characters/motions:
    python test.py --all --weight ./weight/stage2_0004.weight

    # Skip rendering (metrics only):
    python test.py --all --weight ./weight/stage2_0004.weight --metrics-only
"""

import argparse
import csv
import os
import time
import subprocess

import torch
import numpy as np
from config import TEMPORAL_WINDOW, NUM_SCALES
from model import CausalSpatiotemporalModel, PhysicsLoss, build_multiscale_edges
import data_loader
import render
import trimesh
import cv2 as cv
from os import listdir
import animationTet2Surface


def loadData_Float(filename):
    data = np.fromfile(filename, dtype=np.float64)
    return data[1:].reshape(-1, 3)


def predict_rollout(net, frame_num, mesh_path_root, data_path_root, eval_path_root,
                    flag, character_name=None, camera_set=[6.0, 2.0, 5.0, 0.3],
                    metrics_only=False):
    if not os.path.exists(eval_path_root):
        os.makedirs(eval_path_root)

    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()

    offset_filename = os.path.join(data_path_root, "offset")
    offset = data_loader.loadData_Float(offset_filename).reshape(-1, 3)

    # History buffer for autoregressive rollout
    u_history = None  # will be (T+1, V, 3) buffer of displacements
    ms_edges = None
    inference_times = []
    per_frame_mse = []

    for k in range(frame_num - 1):
        constraint, dynamic_frames, reference_frames, adj_matrix, stiffness, mass = \
            data_loader.loadTestInputData(data_path_root, k, frame_num)
        output_f = data_loader.loadTestOutputData(data_path_root, k, frame_num)

        # Unbatch
        constraint_vec = constraint
        dyn = dynamic_frames[0]   # (T+1, V, 3)
        ref = reference_frames[0]  # (T+1, V, 3)
        stiff = stiffness[0]      # (V, 1)
        mass_v = mass[0]          # (V, 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        constraint_vec = constraint_vec.to(device)
        dyn = dyn.to(device)
        ref = ref.to(device)
        stiff = stiff.to(device)
        mass_v = mass_v.to(device)
        adj_matrix = adj_matrix.to(device)
        output_f = output_f.to(device)

        # Build multi-scale edges once
        if ms_edges is None:
            ms_edges = build_multiscale_edges(adj_matrix, NUM_SCALES)

        free_mask = (constraint_vec == 0)

        # Use autoregressive history after first frame
        if k == 0:
            u_history = dyn.clone()
        else:
            dyn[0, free_mask] = u_history[0, free_mask]
            dyn[1:, free_mask] = u_history[:-1, free_mask]

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.time()
            delta_u = net(constraint_vec, dyn, ref, adj_matrix, stiff, mass_v, ms_edges)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - t_start)

        # Build full prediction
        V = dyn.shape[1]
        dis_pred = torch.zeros(1, V, 3, device=device)
        dis_pred[0, free_mask] = dyn[0, free_mask] + delta_u
        dis_pred[0, ~free_mask] = ref[0, ~free_mask] + output_f[0, ~free_mask]

        # Update history buffer
        new_history = torch.zeros_like(u_history)
        new_history[0, free_mask] = dis_pred[0, free_mask]
        new_history[0, ~free_mask] = u_history[0, ~free_mask]
        new_history[1:] = u_history[:-1]
        u_history = new_history

        # Load actual GT displacement u(t+1) directly (not from model state + delta)
        gt_u_next = data_loader.loadTestOutputData(data_path_root, k, frame_num)
        gt_u_next = gt_u_next.to(device)
        # Actual GT: u(k) + delta_u_gt = u(k+1), but load u(k) from file too
        gt_u_curr_raw = data_loader.loadData_Float(
            os.path.join(data_path_root, 'u_%d' % k)).reshape(-1, 3)
        gt_u_curr_raw = np.concatenate([np.zeros((1, 3), dtype=np.float64), gt_u_curr_raw], axis=0)
        gt_u_curr_t = torch.from_numpy(gt_u_curr_raw).float().unsqueeze(0).to(device)
        gt_disp = gt_u_curr_t + gt_u_next  # true u(t+1) from GT data

        # Per-frame MSE (free vertices only)
        frame_mse = ((dis_pred[0, free_mask] - gt_disp[0, free_mask]) ** 2).mean().item()
        per_frame_mse.append(frame_mse)

        # Ground truth — move to CPU for numpy/rendering
        dis_true = gt_disp.cpu()
        dis_input = ref[0:1, :, :].cpu()
        dis_pred = dis_pred.cpu()

        if metrics_only:
            continue

        # ── Render tet mesh ──
        if "tet" in flag:
            mesh_filename = os.path.join(mesh_path_root, "rest.ply")
            mesh_true = trimesh.load(mesh_filename, process=False)
            mesh_pred = trimesh.load(mesh_filename, process=False)
            mesh_input = trimesh.load(mesh_filename, process=False)

            for v, vertex in enumerate(mesh_input.vertices):
                mesh_input.vertices[v] = vertex + np.array(dis_input[:, v + 1, :]) - offset[v, :]
            for v, vertex in enumerate(mesh_true.vertices):
                mesh_true.vertices[v] = vertex + np.array(dis_true[:, v + 1, :]) - offset[v, :]
            for v, vertex in enumerate(mesh_pred.vertices):
                mesh_pred.vertices[v] = vertex + np.array(dis_pred[:, v + 1, :]) - offset[v, :]

            stiff_cpu = stiffness[0].numpy()
            for v, vertex in enumerate(mesh_true.vertices):
                color = [0.0, 0.0, 0.0, 255.0]
                sv = stiff_cpu[v + 1, 0]
                if sv == 5.0:
                    color = [64, 64, 64, 255.0]
                elif sv == 1.0:
                    color = [255, 51, 153, 255.0]
                elif sv == 0.5:
                    color = [153, 82, 255, 255.0]
                elif sv == 0.1:
                    color = [51, 51, 255, 255.0]
                elif sv == 0.05:
                    color = [59, 76, 192, 255.0]
                mesh_input.visual.vertex_colors[v] = color
                mesh_true.visual.vertex_colors[v] = color
                mesh_pred.visual.vertex_colors[v] = color

            for i, c in enumerate(constraint.numpy()):
                if c == 1 and i > 0:
                    mesh_input.visual.vertex_colors[i - 1] = [255, 0, 0, 255.0]
                    mesh_true.visual.vertex_colors[i - 1] = [255, 0, 0, 255.0]
                    mesh_pred.visual.vertex_colors[i - 1] = [255, 0, 0, 255.0]

            image_input = render.render_single_mesh(mesh_input, camera_set=camera_set, resolution=[500, 800])
            image_true = render.render_single_mesh(mesh_true, camera_set=camera_set, resolution=[500, 800])
            image_pred = render.render_single_mesh(mesh_pred, camera_set=camera_set, resolution=[500, 800])
            image_tet = np.concatenate((image_input, image_true, image_pred), axis=1)
            if "surface" not in flag:
                img_filename = os.path.join(eval_path_root, str(k) + ".png")
                print(img_filename)
                cv.imwrite(img_filename, image_tet)

        # ── Render surface mesh ──
        if "surface" in flag:
            surfaceMesh_filename = os.path.join(mesh_path_root, "surface_render.ply")
            surfaceMesh_input = trimesh.load(surfaceMesh_filename, process=False)
            surfaceMesh_true = trimesh.load(surfaceMesh_filename, process=False)
            surfaceMesh_pred = trimesh.load(surfaceMesh_filename, process=False)

            pred_dis = dis_pred[:, 1:, :] - offset
            pred_dis = np.reshape(pred_dis.numpy(), (-1, 1))
            np.savetxt(os.path.join(eval_path_root, "pre_u"), pred_dis)

            true_dis = dis_true[:, 1:, :] - offset
            true_dis = np.reshape(true_dis.numpy(), (-1, 1))
            np.savetxt(os.path.join(eval_path_root, "true_u"), true_dis)

            input_dis = dis_input[:, 1:, :] - offset
            input_dis = np.reshape(input_dis.numpy(), (-1, 1))
            np.savetxt(os.path.join(eval_path_root, "input_u"), input_dis)

            animationTet2Surface.animationTet2Surface(
                mesh_path_root=mesh_path_root, eval_path_root=eval_path_root,
                character_name=character_name, prefix="pre_")
            animationTet2Surface.animationTet2Surface(
                mesh_path_root=mesh_path_root, eval_path_root=eval_path_root,
                character_name=character_name, prefix="input_")
            animationTet2Surface.animationTet2Surface(
                mesh_path_root=mesh_path_root, eval_path_root=eval_path_root,
                character_name=character_name, prefix="true_")

            surface_input_dis = loadData_Float(os.path.join(eval_path_root, "input_SurfaceDis.u"))
            surface_true_dis = loadData_Float(os.path.join(eval_path_root, "true_SurfaceDis.u"))
            surface_pre_dis = loadData_Float(os.path.join(eval_path_root, "pre_SurfaceDis.u"))

            for v, vertex in enumerate(surfaceMesh_input.vertices):
                surfaceMesh_input.vertices[v] = vertex + surface_input_dis[v, :]
            for v, vertex in enumerate(surfaceMesh_true.vertices):
                surfaceMesh_true.vertices[v] = vertex + surface_true_dis[v, :]
            for v, vertex in enumerate(surfaceMesh_pred.vertices):
                surfaceMesh_pred.vertices[v] = vertex + surface_pre_dis[v, :]

            image_input = render.render_single_mesh(surfaceMesh_input, camera_set=camera_set, resolution=[500, 800])
            image_true = render.render_single_mesh(surfaceMesh_true, camera_set=camera_set, resolution=[500, 800])
            image_pred = render.render_single_mesh(surfaceMesh_pred, camera_set=camera_set, resolution=[500, 800])
            image_surface = np.concatenate((image_input, image_true, image_pred), axis=1)
            if "tet" in flag:
                image = np.concatenate((image_tet, image_surface), axis=0)
            else:
                image = image_surface
            img_filename = os.path.join(eval_path_root, str(k) + ".png")
            print(img_filename)
            cv.imwrite(img_filename, image)

    # Timing and metrics summary
    metrics = {}
    if inference_times:
        total_t = sum(inference_times)
        avg_t = total_t / len(inference_times)
        fps = 1.0 / avg_t if avg_t > 0 else 0
        avg_mse = np.mean(per_frame_mse) if per_frame_mse else 0.0
        print(f"\n── Inference Summary ──")
        print(f"  Frames: {len(inference_times)}")
        print(f"  Total:  {total_t:.3f}s")
        print(f"  Avg:    {avg_t*1000:.2f}ms/frame")
        print(f"  FPS:    {fps:.1f}")
        print(f"  Rollout MSE: {avg_mse:.8f}")
        metrics = {
            'frames': len(inference_times),
            'total_time': total_t,
            'avg_ms': avg_t * 1000,
            'fps': fps,
            'rollout_mse': avg_mse,
        }
    return metrics


# ─── Available character/motion combos ────────────────────────────────────────

ALL_CHARACTERS = {
    'michelle':  ['cross_jumps', 'gangnam_style'],
    'big_vegas': ['cross_jumps', 'cross_jumps_rotation'],
    'kaya':      ['dancing_running_man', 'zombie_scream'],
    'mousey':    ['dancing_1', 'swing_dancing_1'],
    'ortiz':     ['cross_jumps_rotation', 'jazz_dancing'],
}


def run_single(net, character_name, motion_name, mesh_path_root, eval_path_root,
               flag="tet_surface", camera_set=None, metrics_only=False):
    """Run evaluation for a single character/motion pair."""
    if camera_set is None:
        camera_set = [6.0, 2.0, 5.0, 0.3]

    input_path = os.path.join(mesh_path_root, character_name, motion_name)
    mesh_path = os.path.join(mesh_path_root, character_name)

    if not os.path.isdir(input_path):
        print(f"  SKIP: {input_path} not found")
        return None

    frames = len([f for f in listdir(input_path) if f.startswith("x_")])
    if frames < 2:
        print(f"  SKIP: {character_name}/{motion_name} has only {frames} frames")
        return None

    eval_path = os.path.join(eval_path_root, character_name, motion_name)
    print(f"\n{'─' * 60}")
    print(f"  {character_name} / {motion_name}  ({frames} frames)")
    print(f"{'─' * 60}")

    metrics = predict_rollout(
        net, frame_num=frames, mesh_path_root=mesh_path,
        data_path_root=input_path, eval_path_root=eval_path,
        flag=flag, character_name=character_name, camera_set=camera_set,
        metrics_only=metrics_only)

    # Combine frames into video (unless metrics-only)
    if not metrics_only and os.path.isdir(eval_path):
        png_files = [f for f in os.listdir(eval_path) if f.endswith('.png')]
        if png_files:
            video_path = os.path.join(eval_path, "animation.mp4")
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", os.path.join(eval_path, "%d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                video_path
            ], check=False)
            print(f"  Video saved to {video_path}")

    if metrics:
        metrics['character'] = character_name
        metrics['motion'] = motion_name
    return metrics


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test / evaluate the deep emulator')
    parser.add_argument('--character', default='michelle', help='Character name')
    parser.add_argument('--motion', default='cross_jumps', help='Motion name')
    parser.add_argument('--weight', default='./weight/stage2_0004.weight',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', default='../data/character_dataset/',
                        help='Root path to character dataset')
    parser.add_argument('--eval_root', default='./weight/eval/',
                        help='Root path for evaluation output')
    parser.add_argument('--flag', default='tet_surface',
                        help='Render mode: tet, surface, or tet_surface')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all characters and motions')
    parser.add_argument('--metrics-only', action='store_true',
                        help='Skip rendering, compute metrics only')
    parser.add_argument('--output', default=None,
                        help='Path to save summary CSV (default: eval_root/summary.csv)')
    args = parser.parse_args()

    net = CausalSpatiotemporalModel()
    net.load_state_dict(torch.load(args.weight, map_location='cpu'))

    all_metrics = []

    if args.all:
        for char, motions in ALL_CHARACTERS.items():
            for motion in motions:
                m = run_single(net, char, motion, args.data_root, args.eval_root,
                               flag=args.flag, metrics_only=args.metrics_only)
                if m:
                    all_metrics.append(m)
    else:
        m = run_single(net, args.character, args.motion, args.data_root, args.eval_root,
                       flag=args.flag, metrics_only=args.metrics_only)
        if m:
            all_metrics.append(m)

    # Save summary CSV
    if all_metrics:
        csv_path = args.output or os.path.join(args.eval_root, "summary.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = ['character', 'motion', 'frames', 'rollout_mse',
                      'total_time', 'avg_ms', 'fps']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow({k: m.get(k, '') for k in fieldnames})
        print(f"\nSummary saved to {csv_path}")

        # Print summary table
        print(f"\n{'Character':<12} {'Motion':<22} {'Frames':>6} {'MSE':>12} {'ms/frame':>9} {'FPS':>6}")
        print(f"{'-'*12} {'-'*22} {'-'*6} {'-'*12} {'-'*9} {'-'*6}")
        for m in all_metrics:
            print(f"{m['character']:<12} {m['motion']:<22} {m['frames']:>6} "
                  f"{m['rollout_mse']:>12.8f} {m['avg_ms']:>9.2f} {m['fps']:>6.1f}")
