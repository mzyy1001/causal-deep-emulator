"""
Render 5-way comparison: Reference | GT | Cone | NoCone | Baseline
Uses scaled stiffness GT at E=50K (seq 1).
"""
import os, sys, numpy as np, torch, trimesh, cv2 as cv, subprocess

import config
config.NUM_SCALES = 1
import importlib, model as mm
importlib.reload(mm)
from model import CausalSpatiotemporalModel, build_multiscale_edges
from model_baseline import Graph_MLP
from config import TEMPORAL_WINDOW
import data_loader, render

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHAR_ROOT = '../data/character_dataset'
GT_ROOT = '../data/scaled_stiffness_dense'
CONE_WEIGHT = './weight_v7_cone/best_stage1.weight'
NOCONE_WEIGHT = './weight_v7_nocone/best_stage1.weight'
BL_WEIGHT = './weight_v7_baseline/best.weight'
OUT_DIR = './weight/eval_5way/videos'
STIFF_SEQ = '1'  # E=50K

MOTIONS = [
    ('mousey', 'dancing_1'),
    ('mousey', 'swing_dancing_1'),
    ('big_vegas', 'cross_jumps'),
    ('big_vegas', 'cross_jumps_rotation'),
    ('ortiz', 'cross_jumps_rotation'),
    ('ortiz', 'jazz_dancing'),
]

def load_models():
    # Cone
    config.USE_CAUSAL_CONE = True
    importlib.reload(mm)
    from model import CausalSpatiotemporalModel as CSM
    cone = CSM()
    cone.load_state_dict(torch.load(CONE_WEIGHT, map_location='cpu'))
    cone.eval().to(device)

    # NoCone
    config.USE_CAUSAL_CONE = False
    importlib.reload(mm)
    from model import CausalSpatiotemporalModel as CSM2
    nocone = CSM2()
    nocone.load_state_dict(torch.load(NOCONE_WEIGHT, map_location='cpu'))
    nocone.eval().to(device)
    config.USE_CAUSAL_CONE = True

    # Baseline
    bl = Graph_MLP()
    bl.load_state_dict(torch.load(BL_WEIGHT, map_location='cpu'))
    bl.eval().to(device)

    return cone, nocone, bl


def render_motion(cone, nocone, bl, char, motion):
    char_path = os.path.join(CHAR_ROOT, char, motion)
    gt_path = os.path.join(GT_ROOT, char, motion, STIFF_SEQ)
    mesh_path = os.path.join(CHAR_ROOT, char, 'rest.ply')
    out_path = os.path.join(OUT_DIR, '..', char, motion)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(mesh_path):
        print('  SKIP: no rest.ply'); return

    c_np = data_loader.loadData_Int(os.path.join(char_path, 'c'))
    constraint = torch.from_numpy(c_np).long().to(device)
    free = (constraint == 0); V = len(c_np)
    adj = torch.from_numpy(data_loader.loadData_Int(os.path.join(char_path, 'adj')).reshape(V, -1)).long().to(device)
    ms = build_multiscale_edges(adj, 1)
    k_data = torch.from_numpy(np.expand_dims(data_loader.loadData_Float(os.path.join(char_path, 'k')), 1) * 0.000001).float().to(device)
    m_raw = data_loader.loadData_Float(os.path.join(char_path, 'm'))
    m_raw = np.expand_dims(m_raw, 1) * 1000; m_raw[0] = 1.0
    mass = torch.from_numpy(m_raw).float().to(device)
    offset = data_loader.loadData_Float(os.path.join(char_path, 'offset')).reshape(-1, 3)

    char_frames = len([f for f in os.listdir(char_path) if f.startswith('x_')])
    gt_frames = len([f for f in os.listdir(gt_path) if f.startswith('u_')])
    num_frames = min(char_frames - 1, gt_frames - 1, 80)

    rest_mesh = trimesh.load(mesh_path, process=False)
    V_mesh = len(rest_mesh.vertices)

    # Camera: scan bounding box
    all_min, all_max = np.full(3, 1e10), np.full(3, -1e10)
    for f in range(0, num_frames, 5):
        u = data_loader.loadData_Float(os.path.join(char_path, 'u_%d' % f)).reshape(-1, 3)[:V_mesh]
        world = rest_mesh.vertices + u - offset[:V_mesh]
        all_min = np.minimum(all_min, world.min(axis=0))
        all_max = np.maximum(all_max, world.max(axis=0))
    center = (all_min + all_max) / 2
    extent = max(all_max - all_min) * 1.3
    cam = [extent * 0.4, center[1], center[2] + extent * 0.8, 0.2]

    print('  Rendering %d frames, cam=%s' % (num_frames, [round(x,1) for x in cam]))

    for k in range(num_frames):
        uf, xf = [], []
        for tau in range(TEMPORAL_WINDOW + 1):
            idx = max(0, min(k - tau, char_frames - 1))
            u = data_loader.loadData_Float(os.path.join(gt_path, 'u_%d' % idx)).reshape(-1, 3)
            u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
            uf.append(torch.from_numpy(u).float().to(device))
        for tau in range(-1, TEMPORAL_WINDOW):
            idx = max(0, min(k - tau, char_frames - 1))
            x = data_loader.loadData_Float(os.path.join(gt_path, 'x_%d' % idx)).reshape(-1, 3)
            x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
            xf.append(torch.from_numpy(x).float().to(device))
        dyn = torch.stack(uf, 0); ref = torch.stack(xf, 0)

        with torch.no_grad():
            d_cone = cone(constraint, dyn, ref, adj, k_data, mass, ms)
            d_nocone = nocone(constraint, dyn, ref, adj, k_data, mass, ms)
            dp = torch.cat([dyn[0], dyn[1], dyn[min(2, dyn.shape[0]-1)]], dim=-1).unsqueeze(0)
            rp = torch.cat([ref[0], ref[1], ref[min(2, ref.shape[0]-1)]], dim=-1).unsqueeze(0)
            d_bl = bl(constraint, dp, rp, adj, k_data.unsqueeze(0), mass.unsqueeze(0)).squeeze(0)

        # Displacements
        x_ref = ref[0, 1:V_mesh+1, :].cpu().numpy()
        gt_u = data_loader.loadData_Float(os.path.join(gt_path, 'u_%d' % k)).reshape(-1, 3)[:V_mesh]
        gt_secondary = gt_u - x_ref

        def build_secondary(delta):
            full = np.zeros((V, 3))
            full[free.cpu().numpy()] = (dyn[0, free] + delta).cpu().numpy()
            return full[1:V_mesh+1] - x_ref

        cone_sec = build_secondary(d_cone)
        nocone_sec = build_secondary(d_nocone)
        bl_sec = build_secondary(d_bl)

        # Errors for coloring
        cone_err = np.linalg.norm(cone_sec - gt_secondary, axis=-1)
        nocone_err = np.linalg.norm(nocone_sec - gt_secondary, axis=-1)
        bl_err = np.linalg.norm(bl_sec - gt_secondary, axis=-1)
        gt_mag = np.linalg.norm(gt_secondary, axis=-1)
        max_err = max(cone_err.max(), nocone_err.max(), bl_err.max(), 0.001)
        max_gt = max(gt_mag.max(), 0.001)

        labels = ['Reference', 'GT', 'Cone', 'NoCone', 'Baseline']
        disps = [x_ref, gt_u, x_ref + cone_sec, x_ref + nocone_sec, x_ref + bl_sec]
        color_data = [None, gt_mag, cone_err, nocone_err, bl_err]
        label_colors = [(100,180,255), (255,255,100), (50,255,100), (255,180,50), (180,130,255)]

        images = []
        c_cpu = constraint.cpu().numpy()
        for idx, (label, disp) in enumerate(zip(labels, disps)):
            m = trimesh.load(mesh_path, process=False)
            for v in range(V_mesh):
                if v < len(disp) and v < len(offset):
                    m.vertices[v] = m.vertices[v] + disp[v] - offset[v]

            cd = color_data[idx]
            if cd is None:
                for v in range(V_mesh):
                    ci = v + 1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [180,180,180,255]
                    else:
                        m.visual.vertex_colors[v] = [100,180,255,255]
            elif idx == 1:
                for v in range(min(V_mesh, len(cd))):
                    ci = v + 1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [120,120,120,255]
                    else:
                        t = min(cd[v] / max_gt, 1.0)
                        m.visual.vertex_colors[v] = [int(50+205*t), int(100+155*t), int(255*(1-t)), 255]
            else:
                for v in range(min(V_mesh, len(cd))):
                    ci = v + 1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [120,120,120,255]
                    else:
                        t = min(cd[v] / max_err, 1.0)
                        if t < 0.5:
                            s = t * 2
                            r, g, b = int(50+205*s), 220, 50
                        else:
                            s = (t-0.5)*2
                            r, g, b = 255, int(220*(1-s)), int(50*(1-s))
                        m.visual.vertex_colors[v] = [r, g, b, 255]

            img = render.render_single_mesh(m, camera_set=cam, resolution=[320, 480],
                                            bg_color=[0.12,0.12,0.18,1.0])
            cv.rectangle(img, (2,2), (130,30), (0,0,0), -1)
            cv.putText(img, label, (5,22), cv.FONT_HERSHEY_SIMPLEX, 0.55, label_colors[idx], 2)
            images.append(img)

        combined = np.concatenate(images, axis=1)
        cv.putText(combined, 'Frame %d' % k, (5, combined.shape[0]-8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv.imwrite(os.path.join(out_path, '%d.png' % k), combined)
        if k % 20 == 0:
            print('    Frame %d/%d' % (k, num_frames))

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except:
        ffmpeg = 'ffmpeg'
    video_path = os.path.join(OUT_DIR, '%s_%s.mp4' % (char, motion))
    subprocess.run([ffmpeg, '-y', '-framerate', '30',
                    '-i', os.path.join(out_path, '%d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    video_path], check=False, capture_output=True)
    sz = os.path.getsize(video_path) // 1024 if os.path.exists(video_path) else 0
    print('  Video: %s (%d KB)' % (video_path, sz))


if __name__ == '__main__':
    cone, nocone, bl = load_models()
    char = sys.argv[1] if len(sys.argv) > 1 else None
    motion = sys.argv[2] if len(sys.argv) > 2 else None
    if char and motion:
        print('=== %s/%s ===' % (char, motion))
        render_motion(cone, nocone, bl, char, motion)
    else:
        for c, m in MOTIONS:
            print('\n=== %s/%s ===' % (c, m))
            render_motion(cone, nocone, bl, c, m)
