"""
Render original GT animation for a character: both tet mesh and surface mesh.
Uses the original render.py and animationTet2Surface.py.
"""
import os
import numpy as np
import trimesh
import cv2 as cv
import subprocess
from data_loader import loadData_Float, loadData_Int
import render
import animationTet2Surface

CHAR = 'mousey'
MOTION = 'dancing_1'
CHAR_ROOT = '../data/character_dataset'
OUT_DIR = './weight/eval_gt'

char_path = os.path.join(CHAR_ROOT, CHAR, MOTION)
mesh_path = os.path.join(CHAR_ROOT, CHAR)
out_path = os.path.join(OUT_DIR, CHAR, MOTION)
os.makedirs(out_path, exist_ok=True)

# Load data
offset = loadData_Float(os.path.join(char_path, 'offset')).reshape(-1, 3)
c = loadData_Int(os.path.join(char_path, 'c'))
stiffness = loadData_Float(os.path.join(char_path, 'k'))
frame_num = len([f for f in os.listdir(char_path) if f.startswith('x_')])
print('Frames: %d' % frame_num)

# Camera — scan bounding box
rest_mesh = trimesh.load(os.path.join(mesh_path, 'rest.ply'), process=False)
V_mesh = len(rest_mesh.vertices)
all_min = np.array([1e10, 1e10, 1e10])
all_max = np.array([-1e10, -1e10, -1e10])
for f in range(0, frame_num, 5):
    u = loadData_Float(os.path.join(char_path, 'u_%d' % f)).reshape(-1, 3)
    world = rest_mesh.vertices + u[:V_mesh] - offset[:V_mesh]
    all_min = np.minimum(all_min, world.min(axis=0))
    all_max = np.maximum(all_max, world.max(axis=0))
center = (all_min + all_max) / 2
extent = max(all_max - all_min) * 1.3
cam = [extent * 0.4, center[1], center[2] + extent * 0.8, 0.2]
print('Camera:', cam)

for k in range(frame_num):
    u = loadData_Float(os.path.join(char_path, 'u_%d' % k)).reshape(-1, 3)
    x_ref = loadData_Float(os.path.join(char_path, 'x_%d' % k)).reshape(-1, 3)

    # --- Tet mesh ---
    mesh_gt = trimesh.load(os.path.join(mesh_path, 'rest.ply'), process=False)
    mesh_ref = trimesh.load(os.path.join(mesh_path, 'rest.ply'), process=False)

    for v in range(V_mesh):
        # GT: rest + u - offset (u includes ref + secondary for original data)
        mesh_gt.vertices[v] = mesh_gt.vertices[v] + u[v] - offset[v]
        # Reference: rest + x_ref - offset
        mesh_ref.vertices[v] = mesh_ref.vertices[v] + x_ref[v] - offset[v]

    # Color tet: constrained=gray, free by stiffness
    for v in range(V_mesh):
        sv = stiffness[v + 1] if v + 1 < len(stiffness) else 50000
        ci = v + 1
        if ci < len(c) and c[ci] == 1:
            mesh_gt.visual.vertex_colors[v] = [200, 200, 200, 255]
            mesh_ref.visual.vertex_colors[v] = [200, 200, 200, 255]
        else:
            mesh_gt.visual.vertex_colors[v] = [100, 200, 255, 255]
            mesh_ref.visual.vertex_colors[v] = [100, 180, 255, 255]

    img_ref = render.render_single_mesh(mesh_ref, camera_set=cam, resolution=[500, 700],
                                         bg_color=[0.12, 0.12, 0.18, 1.0])
    img_gt = render.render_single_mesh(mesh_gt, camera_set=cam, resolution=[500, 700],
                                        bg_color=[0.12, 0.12, 0.18, 1.0])

    # --- Surface mesh ---
    # Write displacement for surface interpolation
    dis = u[:V_mesh] - offset[:V_mesh]
    dis_ref = x_ref[:V_mesh] - offset[:V_mesh]

    np.savetxt(os.path.join(out_path, 'gt_u'), dis.reshape(-1, 1))
    np.savetxt(os.path.join(out_path, 'ref_u'), dis_ref.reshape(-1, 1))

    # Tet -> surface interpolation
    for prefix in ['gt_', 'ref_']:
        animationTet2Surface.animationTet2Surface(
            mesh_path_root=mesh_path, eval_path_root=out_path,
            character_name=CHAR, prefix=prefix)

    # Load surface displacements
    def load_surf(path):
        data = np.fromfile(path, dtype=np.float64)
        return data[1:].reshape(-1, 3)

    surf_gt_dis = load_surf(os.path.join(out_path, 'gt_SurfaceDis.u'))
    surf_ref_dis = load_surf(os.path.join(out_path, 'ref_SurfaceDis.u'))

    surf_gt = trimesh.load(os.path.join(mesh_path, 'surface_render.ply'), process=False)
    surf_ref = trimesh.load(os.path.join(mesh_path, 'surface_render.ply'), process=False)

    for v in range(min(len(surf_gt.vertices), len(surf_gt_dis))):
        surf_gt.vertices[v] = surf_gt.vertices[v] + surf_gt_dis[v]
        surf_ref.vertices[v] = surf_ref.vertices[v] + surf_ref_dis[v]

    img_surf_ref = render.render_single_mesh(surf_ref, camera_set=cam, resolution=[500, 700],
                                              bg_color=[0.12, 0.12, 0.18, 1.0])
    img_surf_gt = render.render_single_mesh(surf_gt, camera_set=cam, resolution=[500, 700],
                                             bg_color=[0.12, 0.12, 0.18, 1.0])

    # Labels
    for img, label in [(img_ref, 'Reference (tet)'), (img_gt, 'GT (tet)'),
                        (img_surf_ref, 'Reference (surface)'), (img_surf_gt, 'GT (surface)')]:
        cv.rectangle(img, (3, 3), (250, 35), (0, 0, 0), -1)
        cv.putText(img, label, (8, 28), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Combine: top row = tet (ref, gt), bottom row = surface (ref, gt)
    top = np.concatenate([img_ref, img_gt], axis=1)
    bottom = np.concatenate([img_surf_ref, img_surf_gt], axis=1)
    combined = np.concatenate([top, bottom], axis=0)

    cv.putText(combined, 'Frame %d' % k, (10, combined.shape[0] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv.imwrite(os.path.join(out_path, '%d.png' % k), combined)
    if k % 20 == 0:
        print('  Frame %d/%d' % (k, frame_num))

# Make video
video_path = os.path.join(OUT_DIR, 'mousey_dancing_1_GT.mp4')
subprocess.run(['ffmpeg', '-y', '-framerate', '30',
                '-i', os.path.join(out_path, '%d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                video_path], check=False, capture_output=True)
size = os.path.getsize(video_path) // 1024 if os.path.exists(video_path) else 0
print('Video: %s (%d KB)' % (video_path, size))
