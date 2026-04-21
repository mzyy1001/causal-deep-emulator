"""
Render 4-way surface comparison videos: Reference | VegaFEM GT | Baseline | Ours
Uses local vega_FEM utilities for tet->surface interpolation.
"""
import os
import sys
import numpy as np
import torch
import trimesh
import cv2 as cv
import subprocess
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
OURS_WEIGHT = './weight_v3a/best_stage1.weight'  # local weights
BL_WEIGHT = './weight/_0000100.weight'
OUT_DIR = './weight/eval_surface'
STIFF_SEQ = '1'  # stiffness=50K

MOTIONS = [
    ('mousey', 'dancing_1'),
    ('mousey', 'swing_dancing_1'),
    ('big_vegas', 'cross_jumps'),
    ('big_vegas', 'cross_jumps_rotation'),
    ('ortiz', 'cross_jumps_rotation'),
    ('ortiz', 'jazz_dancing'),
]

TEXT2MAT = './vega_FEM/utilities/bin/textMatrix2Matrix'
INTERP = './vega_FEM/utilities/bin/interpolateData'


def tet_to_surface(tet_disp, char, tmp_dir):
    """Interpolate tet displacement to surface using vega_FEM utilities."""
    os.makedirs(tmp_dir, exist_ok=True)
    char_path = os.path.join(CHAR_ROOT, char)

    # Write tet displacement as text
    txt_path = os.path.join(tmp_dir, 'tet.txt')
    bin_path = os.path.join(tmp_dir, 'tet.dis')
    surf_path = os.path.join(tmp_dir, 'surf.u')

    flat = tet_disp.reshape(-1, 1)
    np.savetxt(txt_path, flat)

    # Convert to binary
    subprocess.run([TEXT2MAT, txt_path, bin_path, str(len(flat)), '1', '1.0'],
                   capture_output=True, check=True)

    # Interpolate
    subprocess.run([INTERP,
                    os.path.join(char_path, '%s.veg' % char),
                    os.path.join(char_path, '%s.obj' % char),
                    bin_path, surf_path,
                    '-i', os.path.join(char_path, '%s.interp' % char)],
                   capture_output=True, check=True)

    # Read surface displacement
    raw = np.fromfile(surf_path, dtype=np.float64)
    return raw[1:].reshape(-1, 3)  # skip header


def render_motion(net, bl, char, motion):
    char_path = os.path.join(CHAR_ROOT, char, motion)
    vega_path = os.path.join(VEGA_ROOT, char, motion, STIFF_SEQ)
    surf_mesh_path = os.path.join(CHAR_ROOT, char, 'surface_render.ply')
    out_path = os.path.join(OUT_DIR, char, motion)
    tmp_dir = os.path.join(out_path, 'tmp')
    os.makedirs(out_path, exist_ok=True)

    if not os.path.exists(surf_mesh_path):
        print('  SKIP: no surface_render.ply for %s' % char)
        return
    if not os.path.exists(vega_path):
        print('  SKIP: no VegaFEM data at %s' % vega_path)
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

    k_data = torch.from_numpy(
        np.expand_dims(data_loader.loadData_Float(os.path.join(char_path, 'k')), 1) * 0.000001
    ).float().to(device)
    m_raw = data_loader.loadData_Float(os.path.join(char_path, 'm'))
    m_raw = np.expand_dims(m_raw, 1) * 1000
    m_raw[0] = 1.0
    mass = torch.from_numpy(m_raw).float().to(device)

    offset = data_loader.loadData_Float(os.path.join(char_path, 'offset')).reshape(-1, 3)

    char_frames = len([f for f in os.listdir(char_path) if f.startswith('x_')])
    vega_frames = len([f for f in os.listdir(vega_path) if f.startswith('u_')])
    num_frames = min(char_frames - 1, vega_frames - 1, 80)

    surf_mesh = trimesh.load(surf_mesh_path, process=False)
    V_mesh = V - 1  # tet vertices (V includes padding)
    V_surf = len(surf_mesh.vertices)

    # Pre-scan for camera bounding box
    print('  Pre-scanning %d frames for camera...' % num_frames)
    all_min = np.array([1e10, 1e10, 1e10])
    all_max = np.array([-1e10, -1e10, -1e10])
    for f in range(0, num_frames, 5):  # sample every 5 frames
        x = data_loader.loadData_Float(os.path.join(char_path, 'x_%d' % f)).reshape(-1, 3)
        u = np.fromfile(os.path.join(vega_path, 'u_%d' % f), dtype=np.float64).reshape(-1, 3)
        # Approximate world pos from tet (before surface interp)
        total = x[:V_mesh] + u[:V_mesh]
        rest_verts = trimesh.load(os.path.join(CHAR_ROOT, char, 'rest.ply'), process=False).vertices
        world = rest_verts + total - offset[:V_mesh]
        all_min = np.minimum(all_min, world.min(axis=0))
        all_max = np.maximum(all_max, world.max(axis=0))

    center = (all_min + all_max) / 2
    extent = max(all_max - all_min) * 1.5
    cam = [extent * 0.5, center[1], center[2] + extent, 0.15]
    print('  Camera: %s' % cam)
    print('  Rendering %d frames...' % num_frames)

    for k_frame in range(num_frames):
        # Load input for models
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

        with torch.no_grad():
            delta_ours = net(constraint, dyn, ref, adj, k_data, mass, ms_edges)
            dp = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
            rp = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
            delta_bl = bl(constraint, dp, rp, adj, k_data.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)

        # Build tet displacements (V_mesh vertices, no padding)
        ref_x = ref[0, 1:V_mesh+1, :].cpu().numpy()  # reference displacement

        vega_u = np.fromfile(os.path.join(vega_path, 'u_%d' % k_frame), dtype=np.float64).reshape(-1, 3)[:V_mesh]

        ours_full = np.zeros((V, 3))
        ours_full[free.cpu().numpy()] = (dyn[0, free] + delta_ours).cpu().numpy()
        ours_sec = ours_full[1:V_mesh+1]

        bl_full = np.zeros((V, 3))
        bl_full[free.cpu().numpy()] = (dyn[0, free] + delta_bl).cpu().numpy()
        bl_sec = bl_full[1:V_mesh+1]

        # Total tet displacement for each view (ref + secondary - offset)
        tet_ref = ref_x - offset[:V_mesh]
        tet_gt = ref_x + vega_u - offset[:V_mesh]
        tet_bl = ref_x + bl_sec - offset[:V_mesh]
        tet_ours = ref_x + ours_sec - offset[:V_mesh]

        # Interpolate each to surface
        try:
            surf_ref = tet_to_surface(tet_ref, char, tmp_dir + '/ref')
            surf_gt = tet_to_surface(tet_gt, char, tmp_dir + '/gt')
            surf_bl = tet_to_surface(tet_bl, char, tmp_dir + '/bl')
            surf_ours = tet_to_surface(tet_ours, char, tmp_dir + '/ours')
        except Exception as e:
            print('    Frame %d interpolation error: %s' % (k_frame, e))
            continue

        # Error for coloring
        gt_sec_surf = surf_gt - surf_ref
        bl_sec_surf = surf_bl - surf_ref
        ours_sec_surf = surf_ours - surf_ref
        bl_error = np.linalg.norm(bl_sec_surf - gt_sec_surf, axis=-1)
        ours_error = np.linalg.norm(ours_sec_surf - gt_sec_surf, axis=-1)
        max_err = max(bl_error.max(), ours_error.max(), 0.001)

        # Render 4 surface meshes
        images = []
        labels = ['Reference', 'VegaFEM GT', 'Baseline', 'Ours']
        surf_disps = [surf_ref, surf_gt, surf_bl, surf_ours]
        error_data = [None, None, bl_error, ours_error]
        label_colors = [[100, 180, 255], [255, 255, 100], [100, 220, 50], [255, 150, 100]]

        for idx, (label, sd) in enumerate(zip(labels, surf_disps)):
            m = trimesh.load(surf_mesh_path, process=False)
            for v in range(min(V_surf, len(sd))):
                m.vertices[v] = m.vertices[v] + sd[v]

            # Convert texture to vertex colors if needed
            if hasattr(m.visual, 'kind') and m.visual.kind == 'texture':
                m.visual = m.visual.to_color()

            # Coloring
            if idx <= 1:
                # Reference/GT: keep mesh color
                pass
            else:
                # Baseline/Ours: green->red error heatmap
                err = error_data[idx]
                colors = np.array(m.visual.vertex_colors).copy()
                for v in range(min(V_surf, len(err))):
                    t = min(err[v] / max_err, 1.0)
                    if t < 0.5:
                        s = t * 2
                        r, g, b = int(50 + 205*s), 220, 50
                    else:
                        s = (t - 0.5) * 2
                        r, g, b = 255, int(220*(1-s)), int(50*(1-s))
                    colors[v] = [r, g, b, 255]
                m.visual.vertex_colors = colors

            img = render.render_single_mesh(m, camera_set=cam, resolution=[400, 600],
                                            bg_color=[0.15, 0.15, 0.2, 1.0])
            cv.rectangle(img, (3, 3), (220, 55), (0, 0, 0), -1)
            sublabel = '' if idx <= 1 else '(error vs GT)'
            cv.putText(img, label, (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       tuple(label_colors[idx]), 2)
            cv.putText(img, sublabel, (8, 48), cv.FONT_HERSHEY_SIMPLEX, 0.4,
                       (200, 200, 200), 1)
            images.append(img)

        combined = np.concatenate(images, axis=1)
        cv.putText(combined, 'Frame %d' % k_frame, (10, combined.shape[0] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv.imwrite(os.path.join(out_path, '%d.png' % k_frame), combined)
        if k_frame % 10 == 0:
            print('    Frame %d/%d' % (k_frame, num_frames))

    # Make video
    video_dir = os.path.join(OUT_DIR, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, '%s_%s.mp4' % (char, motion))
    subprocess.run(['ffmpeg', '-y', '-framerate', '30',
                    '-i', os.path.join(out_path, '%d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    video_path], check=False, capture_output=True)
    size = os.path.getsize(video_path) // 1024 if os.path.exists(video_path) else 0
    print('  Video: %s (%d KB)' % (video_path, size))

    # Cleanup tmp
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    print('Loading models...')
    net = CausalSpatiotemporalModel()
    net.load_state_dict(torch.load(OURS_WEIGHT, map_location='cpu'))
    net.eval().to(device)

    bl = Graph_MLP()
    bl.load_state_dict(torch.load(BL_WEIGHT, map_location='cpu'))
    bl.eval().to(device)

    for char, motion in MOTIONS:
        print('\n=== %s/%s ===' % (char, motion))
        try:
            render_motion(net, bl, char, motion)
        except Exception as e:
            import traceback
            print('  ERROR: %s' % e)
            traceback.print_exc()


if __name__ == '__main__':
    main()
