"""
Render 5-way comparison for ALL stiffness values.
Usage: python render_5way_allstiff.py <char> <motion>
Generates one video per stiffness value.
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
OUT_DIR = './weight/eval_allstiff'

STIFF_TRAIN = [50000, 100000, 250000, 500000, 1000000, 2500000, 5000000]
STIFF_TEST = [10000, 25000, 75000, 150000, 300000, 750000, 1500000, 2000000, 3000000, 4000000, 6000000, 8000000, 10000000]
ALL_STIFF = STIFF_TRAIN + STIFF_TEST
STIFF_SEQ = {s: str(i+1) for i, s in enumerate(ALL_STIFF)}


def load_models():
    config.USE_CAUSAL_CONE = True; importlib.reload(mm)
    from model import CausalSpatiotemporalModel as C1
    cone = C1(); cone.load_state_dict(torch.load(CONE_WEIGHT, map_location='cpu')); cone.eval().to(device)
    config.USE_CAUSAL_CONE = False; importlib.reload(mm)
    from model import CausalSpatiotemporalModel as C2
    nocone = C2(); nocone.load_state_dict(torch.load(NOCONE_WEIGHT, map_location='cpu')); nocone.eval().to(device)
    config.USE_CAUSAL_CONE = True
    bl = Graph_MLP(); bl.load_state_dict(torch.load(BL_WEIGHT, map_location='cpu')); bl.eval().to(device)
    return cone, nocone, bl


def render_one_stiffness(cone, nocone, bl, char, motion, stiff, cam, preloaded):
    """Render one video for a specific stiffness value."""
    constraint, free, V, adj, ms, mass, offset, V_mesh, rest_verts = preloaded
    seq = STIFF_SEQ[stiff]
    gt_path = os.path.join(GT_ROOT, char, motion, seq)
    char_path = os.path.join(CHAR_ROOT, char, motion)
    mesh_path = os.path.join(CHAR_ROOT, char, 'rest.ply')
    seen = 'train' if stiff in STIFF_TRAIN else 'test'
    out_path = os.path.join(OUT_DIR, char, motion, 'E%d' % stiff)
    os.makedirs(out_path, exist_ok=True)

    # Stiffness input
    k_raw = np.full(V, stiff, dtype=np.float64)
    k_data = torch.from_numpy(np.expand_dims(k_raw, 1) * 0.000001).float().to(device)

    char_frames = len([f for f in os.listdir(char_path) if f.startswith('x_')])
    gt_frames = len([f for f in os.listdir(gt_path) if f.startswith('u_')])
    num_frames = min(char_frames - 1, gt_frames - 1, 60)

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

        x_ref = ref[0, 1:V_mesh+1, :].cpu().numpy()
        gt_u = data_loader.loadData_Float(os.path.join(gt_path, 'u_%d' % k)).reshape(-1, 3)[:V_mesh]
        gt_sec = gt_u - x_ref

        def build_sec(delta):
            full = np.zeros((V, 3))
            full[free.cpu().numpy()] = (dyn[0, free] + delta).cpu().numpy()
            return full[1:V_mesh+1] - x_ref

        cone_sec = build_sec(d_cone); nocone_sec = build_sec(d_nocone); bl_sec = build_sec(d_bl)
        cone_err = np.linalg.norm(cone_sec - gt_sec, axis=-1)
        nocone_err = np.linalg.norm(nocone_sec - gt_sec, axis=-1)
        bl_err = np.linalg.norm(bl_sec - gt_sec, axis=-1)
        gt_mag = np.linalg.norm(gt_sec, axis=-1)
        max_err = max(cone_err.max(), nocone_err.max(), bl_err.max(), 0.001)
        max_gt = max(gt_mag.max(), 0.001)

        labels = ['Reference', 'GT', 'Cone', 'NoCone', 'Baseline']
        disps = [x_ref, gt_u, x_ref+cone_sec, x_ref+nocone_sec, x_ref+bl_sec]
        cdata = [None, gt_mag, cone_err, nocone_err, bl_err]
        lcols = [(100,180,255),(255,255,100),(50,255,100),(255,180,50),(180,130,255)]
        c_cpu = constraint.cpu().numpy()

        images = []
        for idx2, (lab, disp) in enumerate(zip(labels, disps)):
            m = trimesh.load(mesh_path, process=False)
            for v in range(V_mesh):
                if v < len(disp) and v < len(offset):
                    m.vertices[v] = rest_verts[v] + disp[v] - offset[v]
            cd = cdata[idx2]
            if cd is None:
                for v in range(V_mesh):
                    ci = v+1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [180,180,180,255]
                    else:
                        m.visual.vertex_colors[v] = [100,180,255,255]
            elif idx2 == 1:
                for v in range(min(V_mesh, len(cd))):
                    ci = v+1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [120,120,120,255]
                    else:
                        t = min(cd[v]/max_gt, 1.0)
                        m.visual.vertex_colors[v] = [int(50+205*t),int(100+155*t),int(255*(1-t)),255]
            else:
                for v in range(min(V_mesh, len(cd))):
                    ci = v+1
                    if ci < len(c_cpu) and c_cpu[ci] == 1:
                        m.visual.vertex_colors[v] = [120,120,120,255]
                    else:
                        t = min(cd[v]/max_err, 1.0)
                        if t < 0.5:
                            s2 = t*2; r,g,b = int(50+205*s2),220,50
                        else:
                            s2 = (t-0.5)*2; r,g,b = 255,int(220*(1-s2)),int(50*(1-s2))
                        m.visual.vertex_colors[v] = [r,g,b,255]
            img = render.render_single_mesh(m, camera_set=cam, resolution=[280,420],
                                            bg_color=[0.12,0.12,0.18,1.0])
            cv.rectangle(img, (2,2), (110,26), (0,0,0), -1)
            cv.putText(img, lab, (4,20), cv.FONT_HERSHEY_SIMPLEX, 0.45, lcols[idx2], 1)
            images.append(img)

        combined = np.concatenate(images, axis=1)
        # Add stiffness label
        cv.rectangle(combined, (0, combined.shape[0]-25), (350, combined.shape[0]), (0,0,0), -1)
        cv.putText(combined, 'E=%d (%s) Frame %d' % (stiff, seen, k),
                   (5, combined.shape[0]-8), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv.imwrite(os.path.join(out_path, '%d.png' % k), combined)

    # Video
    vid_dir = os.path.join(OUT_DIR, 'videos')
    os.makedirs(vid_dir, exist_ok=True)
    vid = os.path.join(vid_dir, '%s_%s_E%d.mp4' % (char, motion, stiff))
    try:
        import imageio_ffmpeg; ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except: ffmpeg = 'ffmpeg'
    subprocess.run([ffmpeg,'-y','-framerate','30','-i',os.path.join(out_path,'%d.png'),
                    '-c:v','libx264','-pix_fmt','yuv420p',vid], check=False, capture_output=True)
    sz = os.path.getsize(vid)//1024 if os.path.exists(vid) else 0
    return sz


if __name__ == '__main__':
    char = sys.argv[1]; motion = sys.argv[2]
    print('Loading models...')
    cone, nocone, bl = load_models()

    # Preload topology
    char_path = os.path.join(CHAR_ROOT, char, motion)
    mesh_path = os.path.join(CHAR_ROOT, char, 'rest.ply')
    c_np = data_loader.loadData_Int(os.path.join(char_path, 'c'))
    constraint = torch.from_numpy(c_np).long().to(device)
    free = (constraint == 0); V = len(c_np)
    adj = torch.from_numpy(data_loader.loadData_Int(os.path.join(char_path, 'adj')).reshape(V,-1)).long().to(device)
    ms = build_multiscale_edges(adj, 1)
    m_raw = data_loader.loadData_Float(os.path.join(char_path, 'm'))
    m_raw = np.expand_dims(m_raw, 1)*1000; m_raw[0] = 1.0
    mass = torch.from_numpy(m_raw).float().to(device)
    offset = data_loader.loadData_Float(os.path.join(char_path, 'offset')).reshape(-1, 3)
    rest_mesh = trimesh.load(mesh_path, process=False)
    V_mesh = len(rest_mesh.vertices)
    rest_verts = rest_mesh.vertices.copy()

    # Camera
    all_min, all_max = np.full(3,1e10), np.full(3,-1e10)
    char_frames = len([f for f in os.listdir(char_path) if f.startswith('x_')])
    for f in range(0, min(char_frames, 80), 5):
        u = data_loader.loadData_Float(os.path.join(char_path, 'u_%d' % f)).reshape(-1,3)[:V_mesh]
        w = rest_verts + u - offset[:V_mesh]
        all_min = np.minimum(all_min, w.min(axis=0)); all_max = np.maximum(all_max, w.max(axis=0))
    center = (all_min+all_max)/2; extent = max(all_max-all_min)*1.3
    cam = [extent*0.4, center[1], center[2]+extent*0.8, 0.2]

    preloaded = (constraint, free, V, adj, ms, mass, offset, V_mesh, rest_verts)

    print('=== %s/%s === (20 stiffness, cam=%s)' % (char, motion, [round(x,1) for x in cam]))
    for stiff in sorted(ALL_STIFF):
        sz = render_one_stiffness(cone, nocone, bl, char, motion, stiff, cam, preloaded)
        seen = 'train' if stiff in STIFF_TRAIN else 'test'
        print('  E=%10d (%s): %d KB' % (stiff, seen, sz))
    print('Done!')
