"""
Render 4-way comparison videos: Reference | VegaFEM GT | Baseline | Ours
for each character/motion at stiffness=50000.
"""
import os
import sys
import numpy as np
import torch
import trimesh
import cv2 as cv
import time

import config
config.NUM_SCALES = 1
import importlib
import model as model_module
importlib.reload(model_module)
from model import CausalSpatiotemporalModel, build_multiscale_edges
from model_baseline import Graph_MLP
from config import TEMPORAL_WINDOW
import data_loader
import render

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHAR_ROOT = '../data/character_dataset'
VEGA_ROOT = '../data/vega_stiffness'
OURS_WEIGHT = './weight_final/best_stage1.weight'
BL_WEIGHT = './weight_baseline_retrained/best.weight'
EVAL_ROOT = './weight/eval_compare'
CAMERA = [5.0, 2.5, 6.0, 0.3]  # close camera for visible meshes

MOTIONS = [
    ('mousey', 'dancing_1'),
    ('mousey', 'swing_dancing_1'),
    ('big_vegas', 'cross_jumps'),
    ('big_vegas', 'cross_jumps_rotation'),
    ('ortiz', 'cross_jumps_rotation'),
    ('ortiz', 'jazz_dancing'),
]

# Stiffness seq 1 = 50000
STIFF_SEQ = '1'


def load_models():
    print('Loading models...')
    net = CausalSpatiotemporalModel()
    net.load_state_dict(torch.load(OURS_WEIGHT, map_location='cpu'))
    net.eval().to(device)

    bl = Graph_MLP()
    bl.load_state_dict(torch.load(BL_WEIGHT, map_location='cpu'))
    bl.eval().to(device)
    return net, bl


def render_mesh(mesh, camera_set, resolution=[400, 600]):
    """Render a single mesh and return image."""
    return render.render_single_mesh(mesh, camera_set=camera_set, resolution=resolution,
                                     bg_color=[0.15, 0.15, 0.2, 1.0])


def render_motion(net, bl, char, motion):
    """Render 4-way comparison for one character/motion."""
    char_path = os.path.join(CHAR_ROOT, char, motion)
    vega_path = os.path.join(VEGA_ROOT, char, motion, STIFF_SEQ)
    mesh_path = os.path.join(CHAR_ROOT, char, 'rest.ply')
    out_path = os.path.join(EVAL_ROOT, char, motion)
    os.makedirs(out_path, exist_ok=True)

    if not os.path.exists(mesh_path):
        print('  SKIP: no rest.ply for %s' % char)
        return

    # Load topology
    c_np = data_loader.loadData_Int(os.path.join(char_path, 'c'))
    constraint = torch.from_numpy(c_np).long().to(device)
    free = (constraint == 0)
    V = len(c_np)

    adj = torch.from_numpy(
        data_loader.loadData_Int(os.path.join(char_path, 'adj')).reshape(V, -1)
    ).long().to(device)
    ms_edges = build_multiscale_edges(adj, 1)

    # Stiffness = 50000 for this visualization
    k_raw = data_loader.loadData_Float(os.path.join(char_path, 'k'))
    k = torch.from_numpy(np.expand_dims(k_raw, 1) * 0.000001).float().to(device)
    m_raw = data_loader.loadData_Float(os.path.join(char_path, 'm'))
    m_raw = np.expand_dims(m_raw, 1) * 1000
    m_raw[0] = 1.0
    mass = torch.from_numpy(m_raw).float().to(device)

    # Offset
    offset_file = os.path.join(char_path, 'offset')
    if os.path.exists(offset_file):
        offset = data_loader.loadData_Float(offset_file).reshape(-1, 3)
    else:
        offset = np.zeros((V, 3))

    # Number of frames (use min of char and vega)
    char_frames = len([f for f in os.listdir(char_path) if f.startswith('x_')])
    vega_frames = len([f for f in os.listdir(vega_path) if f.startswith('u_')])
    num_frames = min(char_frames - 1, vega_frames - 1, 80)

    # Load rest mesh
    rest_mesh = trimesh.load(mesh_path, process=False)
    V_mesh = len(rest_mesh.vertices)
    rest_verts = rest_mesh.vertices.copy()

    # Pre-scan all frames to find bounding box for camera
    print('  Pre-scanning %d frames for camera...' % num_frames)
    all_min = np.array([1e10, 1e10, 1e10])
    all_max = np.array([-1e10, -1e10, -1e10])
    for f in range(num_frames):
        x = data_loader.loadData_Float(os.path.join(char_path, 'x_%d' % f)).reshape(-1, 3)
        u = np.fromfile(os.path.join(vega_path, 'u_%d' % f), dtype=np.float64).reshape(-1, 3)
        # World position = rest + x + u - offset
        total_disp = x[:V_mesh] + u[:V_mesh]
        world = rest_verts + total_disp - offset[:V_mesh]
        all_min = np.minimum(all_min, world.min(axis=0))
        all_max = np.maximum(all_max, world.max(axis=0))

    center = (all_min + all_max) / 2
    size = all_max - all_min
    max_extent = max(size) * 1.5  # padding
    # Camera: position at center + offset, looking at center
    cam_dist = max_extent * 1.2
    cam = [cam_dist * 0.5, center[1], center[2] + cam_dist, 0.15]
    print('  Camera: center=(%.1f,%.1f,%.1f) extent=%.1f cam=%s' % (
        *center, max_extent, cam))

    print('  Rendering %d frames...' % num_frames)

    # Autoregressive history for both models
    u_history_ours = None
    u_history_bl = None

    for k_frame in range(num_frames):
        # Load input frames for models
        u_frames = []
        for tau in range(TEMPORAL_WINDOW + 1):
            idx = max(0, min(k_frame - tau, char_frames - 1))
            u = data_loader.loadData_Float(os.path.join(char_path, 'u_%d' % idx)).reshape(-1, 3)
            u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
            u_frames.append(torch.from_numpy(u).float().to(device))
        x_frames = []
        for tau in range(-1, TEMPORAL_WINDOW):
            idx = max(0, min(k_frame - tau, char_frames - 1))
            x = data_loader.loadData_Float(os.path.join(char_path, 'x_%d' % idx)).reshape(-1, 3)
            x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
            x_frames.append(torch.from_numpy(x).float().to(device))

        dyn = torch.stack(u_frames, 0)
        ref = torch.stack(x_frames, 0)

        # Model predictions (single-step, not autoregressive for simplicity)
        with torch.no_grad():
            delta_ours = net(constraint, dyn, ref, adj, k, mass, ms_edges)
            dp = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
            rp = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
            delta_bl = bl(constraint, dp, rp, adj, k.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)

        # Build displacements
        # Reference motion (skeletal): x_frame is displacement from rest
        ref_disp = ref[0, 1:V_mesh+1, :].cpu().numpy()  # skip padding

        # GT secondary from VegaFEM
        vega_u = np.fromfile(os.path.join(vega_path, 'u_%d' % k_frame), dtype=np.float64).reshape(-1, 3)
        gt_secondary = vega_u[:V_mesh]

        # Our prediction: u(t) + delta = predicted secondary displacement
        ours_full = np.zeros((V, 3))
        ours_full[free.cpu().numpy()] = (dyn[0, free] + delta_ours).cpu().numpy()
        ours_secondary = ours_full[1:V_mesh+1]

        bl_full = np.zeros((V, 3))
        bl_full[free.cpu().numpy()] = (dyn[0, free] + delta_bl).cpu().numpy()
        bl_secondary = bl_full[1:V_mesh+1]

        # Total displacement = reference motion + secondary motion
        # Reference view: only skeletal motion (no secondary)
        # GT/Baseline/Ours: skeletal + secondary
        disps = [
            ref_disp,                      # Reference: skeletal only
            ref_disp + gt_secondary,       # GT: skeletal + VegaFEM secondary
            ref_disp + bl_secondary,       # Baseline: skeletal + predicted secondary
            ref_disp + ours_secondary,     # Ours: skeletal + predicted secondary
        ]

        # Error vs GT for coloring (per-vertex L2 distance from GT secondary)
        bl_error = np.linalg.norm(bl_secondary - gt_secondary, axis=-1)   # (V_mesh,)
        ours_error = np.linalg.norm(ours_secondary - gt_secondary, axis=-1)
        gt_mag = np.linalg.norm(gt_secondary, axis=-1)  # GT displacement magnitude

        # Per-vertex data for coloring: [ref=None, GT=displacement, BL=error, Ours=error]
        color_data = [
            None,         # Reference: constraint coloring
            gt_mag,       # GT: displacement magnitude (blue->yellow)
            bl_error,     # Baseline: error vs GT (green->red)
            ours_error,   # Ours: error vs GT (green->red)
        ]

        # Render 4 meshes
        images = []
        labels = ['Reference', 'VegaFEM GT', 'Baseline', 'Ours']
        # Color scheme: blue=small disp, red=large disp
        # Reference gets gray, others get heatmap
        colors_base = [
            [180, 180, 180],  # Reference: gray
            [100, 200, 100],  # GT: green tint
            [100, 150, 255],  # Baseline: blue tint
            [255, 130, 100],  # Ours: orange tint
        ]

        # Color scales
        max_gt_mag = max(gt_mag.max(), 0.001)
        max_error = max(bl_error.max(), ours_error.max(), 0.001)

        for idx, (label, disp) in enumerate(zip(labels, disps)):
            m = trimesh.load(mesh_path, process=False)
            nv = len(m.vertices)

            # Apply displacement
            for v in range(nv):
                if v < len(disp):
                    if v < len(offset):
                        m.vertices[v] = m.vertices[v] + disp[v] - offset[v]
                    else:
                        m.vertices[v] = m.vertices[v] + disp[v]

            c_cpu = constraint.cpu().numpy()
            cdata = color_data[idx]

            if idx == 0:
                # Reference: constraint coloring (gray=constrained, blue=free)
                for v in range(nv):
                    ci = v + 1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [180, 180, 180, 255]
                    else:
                        m.visual.vertex_colors[v] = [100, 180, 255, 255]
            elif idx == 1:
                # GT: displacement magnitude (blue -> cyan -> yellow)
                for v in range(nv):
                    ci = v + 1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [120, 120, 120, 255]
                    elif v < len(cdata):
                        t = min(cdata[v] / max_gt_mag, 1.0)
                        r = int(50 + 205 * t)
                        g = int(100 + 155 * t)
                        b = int(255 * (1 - t))
                        m.visual.vertex_colors[v] = [r, g, b, 255]
            else:
                # Baseline/Ours: error vs GT (green=accurate -> yellow -> red=large error)
                for v in range(nv):
                    ci = v + 1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [120, 120, 120, 255]
                    elif v < len(cdata):
                        t = min(cdata[v] / max_error, 1.0)
                        if t < 0.5:
                            s = t * 2
                            r = int(50 + 205 * s)
                            g = 220
                            b = 50
                        else:
                            s = (t - 0.5) * 2
                            r = 255
                            g = int(220 * (1 - s))
                            b = int(50 * (1 - s))
                        m.visual.vertex_colors[v] = [r, g, b, 255]

            img = render_mesh(m, cam, resolution=[400, 600])
            # Label with background
            label_colors = [[100,180,255], [255,255,100], [100,220,50], [255,150,100]]
            sublabels = ['', '(displacement)', '(error vs GT)', '(error vs GT)']
            cv.rectangle(img, (3, 3), (220, 55), (0, 0, 0), -1)
            cv.putText(img, label, (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       tuple(label_colors[idx]), 2)
            cv.putText(img, sublabels[idx], (8, 48), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                       (200, 200, 200), 1)
            images.append(img)

        # Concatenate 4 images horizontally
        combined = np.concatenate(images, axis=1)

        # Add frame number
        cv.putText(combined, 'Frame %d' % k_frame, (10, combined.shape[0] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        img_path = os.path.join(out_path, '%d.png' % k_frame)
        cv.imwrite(img_path, combined)

        if k_frame % 20 == 0:
            print('    Frame %d/%d' % (k_frame, num_frames))

    # Make video
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = 'ffmpeg'

    # Save video in flat folder with descriptive name
    video_dir = os.path.join(EVAL_ROOT, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_name = '%s_%s.mp4' % (char, motion)
    video_path = os.path.join(video_dir, video_name)
    import subprocess
    subprocess.run([ffmpeg, '-y', '-framerate', '30',
                    '-i', os.path.join(out_path, '%d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    video_path], check=False, capture_output=True)
    size = os.path.getsize(video_path) // 1024 if os.path.exists(video_path) else 0
    print('  Video: %s (%d KB)' % (video_path, size))


def main():
    net, bl = load_models()

    for char, motion in MOTIONS:
        print('\n=== %s/%s ===' % (char, motion))
        try:
            render_motion(net, bl, char, motion)
        except Exception as e:
            print('  ERROR: %s' % str(e))


if __name__ == '__main__':
    main()
