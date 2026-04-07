"""
Testing / rollout prediction for the causal spatiotemporal deep emulator.
"""

import os
import torch
import numpy as np
from config import TEMPORAL_WINDOW, NUM_SCALES
from model import CausalSpatiotemporalModel, build_multiscale_edges
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
                    flag, character_name=None, camera_set=[6.0, 2.0, 5.0, 0.3]):
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
            delta_u = net(constraint_vec, dyn, ref, adj_matrix, stiff, mass_v, ms_edges)

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

        # Ground truth — move to CPU for numpy/rendering
        dis_true = (dyn[0:1] + output_f).cpu()
        dis_input = ref[0:1, :, :].cpu()
        dis_pred = dis_pred.cpu()

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


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    character_name = "michelle"
    motion_name = "cross_jumps"
    weight_path = "./weight/stage2_0004.weight"
    mesh_path_root = "../data/character_dataset/"
    eval_path_root = "./weight/eval/"
    camera_set = [6.0, 2.0, 5.0, 0.3]

    input_path = os.path.join(mesh_path_root, character_name, motion_name)
    mesh_path = os.path.join(mesh_path_root, character_name)
    frames = len([f for f in listdir(input_path) if f.startswith("x_")])
    eval_path = os.path.join(eval_path_root, character_name, motion_name)

    net = CausalSpatiotemporalModel()
    net.load_state_dict(torch.load(weight_path))
    predict_rollout(net, frame_num=frames, mesh_path_root=mesh_path,
                    data_path_root=input_path, eval_path_root=eval_path,
                    flag="tet_surface", character_name=character_name, camera_set=camera_set)

    # ── Combine frames into video ──
    import subprocess
    video_path = os.path.join(eval_path, "animation.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", os.path.join(eval_path, "%d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ], check=True)
    print(f"Video saved to {video_path}")
