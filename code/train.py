"""
Two-stage hybrid training for the causal spatiotemporal deep emulator.

Stage 1: Supervised learning with ground-truth displacement increments.
Stage 2: Self-supervised fine-tuning with physics energy minimization
         and progressively lengthened K-step rollout.
"""

import os
from os import listdir
import torch
import numpy as np
from config import NUM_SCALES
from model import CausalSpatiotemporalModel, PhysicsLoss, build_multiscale_edges
import data_loader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from random import shuffle


# ─── Training Hyperparameters ──────────────────────────────────────────────────
STAGE1_EPOCHS = 60
STAGE1_LR = 1e-4
STAGE1_BATCH = 64
STAGE1_DECAY = 0.96

STAGE2_EPOCHS = 40
STAGE2_LR = 1e-5
STAGE2_BATCH = 4
STAGE2_DECAY = 0.98
STAGE2_MAX_ROLLOUT = 8   # maximum rollout horizon K
STAGE2_ROLLOUT_WARMUP = 10  # epochs before increasing K


# ─── Stage 1: Supervised ──────────────────────────────────────────────────────

def train_stage1(net, writer, data_path_root, out_weight_folder,
                 train_seq_num=10, test_seq_num=10):
    print("=" * 60)
    print("Stage 1: Supervised Training")
    print("=" * 60)

    if torch.cuda.is_available():
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=STAGE1_LR)
    net.train()

    train_data_path = os.path.join(data_path_root, "train")
    test_data_path = os.path.join(data_path_root, "test")
    train_files = sorted([f for f in listdir(train_data_path) if f.startswith("motion_")])
    test_files = sorted([f for f in listdir(test_data_path) if f.startswith("motion_")])

    for epoch in range(STAGE1_EPOCHS):
        lr = STAGE1_LR * (STAGE1_DECAY ** epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        print(f"\n[Stage 1] Epoch {epoch}  lr={lr:.6f}")
        writer.add_scalar('Stage1/LearningRate', lr, epoch)

        # ── Train ──
        net.train()
        train_loss_sum, train_count = 0.0, 0
        shuffle(train_files)
        for train_file in train_files:
            mesh_path = os.path.join(train_data_path, train_file)
            frame_num = len([f for f in listdir(mesh_path + "/1/") if f.startswith("x_")]) - 1
            dataset = data_loader.MeshDataset(mesh_path, train_seq_num, frame_num)
            loader = DataLoader(dataset, batch_size=STAGE1_BATCH, shuffle=True, num_workers=2)

            # Precompute multi-scale edges once per mesh topology
            adj = dataset.adj_matrix
            if torch.cuda.is_available():
                adj = adj.cuda()
            ms_edges = build_multiscale_edges(adj, NUM_SCALES)

            for constraint, dyn, ref, adj_mat, stiff, mass, gt in loader:
                if torch.cuda.is_available():
                    dyn, ref, stiff, mass, gt = (
                        dyn.cuda(), ref.cuda(), stiff.cuda(), mass.cuda(), gt.cuda())
                    constraint = constraint[0].cuda()
                else:
                    constraint = constraint[0]

                pred = net.forward_batch(
                    constraint, dyn, ref, adj_mat[0], stiff, mass, ms_edges)
                target = gt[:, constraint == 0, :]
                loss = net.compute_supervised_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
                train_count += 1

        avg_train = train_loss_sum / max(train_count, 1)
        print(f"  train loss: {avg_train:.6f}")
        writer.add_scalar('Stage1/Loss/train', avg_train, epoch)

        # ── Test ──
        net.eval()
        test_loss_sum, test_count = 0.0, 0
        with torch.no_grad():
            for test_file in test_files:
                mesh_path = os.path.join(test_data_path, test_file)
                frame_num = len([f for f in listdir(mesh_path + "/1/") if f.startswith("x_")]) - 1
                dataset = data_loader.MeshDataset(mesh_path, test_seq_num, frame_num)
                loader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=2)

                adj = dataset.adj_matrix
                if torch.cuda.is_available():
                    adj = adj.cuda()
                ms_edges = build_multiscale_edges(adj, NUM_SCALES)

                for constraint, dyn, ref, adj_mat, stiff, mass, gt in loader:
                    if torch.cuda.is_available():
                        dyn, ref, stiff, mass, gt = (
                            dyn.cuda(), ref.cuda(), stiff.cuda(), mass.cuda(), gt.cuda())
                        constraint = constraint[0].cuda()
                    else:
                        constraint = constraint[0]

                    pred = net.forward_batch(
                        constraint, dyn, ref, adj_mat[0], stiff, mass, ms_edges)
                    target = gt[:, constraint == 0, :]
                    loss = net.compute_supervised_loss(pred, target)
                    test_loss_sum += loss.item()
                    test_count += 1

        avg_test = test_loss_sum / max(test_count, 1)
        print(f"  test loss:  {avg_test:.6f}")
        writer.add_scalar('Stage1/Loss/test', avg_test, epoch)

        # Save checkpoint
        path = os.path.join(out_weight_folder, f"stage1_{epoch:04d}.weight")
        torch.save(net.state_dict(), path)

    return net


# ─── Stage 2: Self-Supervised with Physics Energy ─────────────────────────────

def rollout_k_steps(net, constraint, dynamic_frames, reference_frames,
                    adj_matrix, stiffness, mass, ms_edges, K):
    """Run K-step autoregressive rollout, returning predicted positions.

    Args:
        dynamic_frames: (T+1, V, 3) initial history
        reference_frames: (T+1, V, 3) reference history
        K: rollout steps

    Returns:
        predictions: list of K (V, 3) predicted position tensors
    """
    V = dynamic_frames.shape[1]
    free_mask = (constraint == 0)

    # Working copies of frame history
    dyn = dynamic_frames.clone()
    ref = reference_frames.clone()

    predictions = []
    for step in range(K):
        delta_u = net(constraint, dyn, ref, adj_matrix, stiffness, mass, ms_edges)

        # Compute full predicted displacement
        x_pred = torch.zeros(V, 3, device=dyn.device)
        x_pred[free_mask] = dyn[0, free_mask] + delta_u  # u(t) + delta_u
        x_pred[~free_mask] = ref[0, ~free_mask]  # constrained: use reference

        predictions.append(x_pred)

        # Shift history: new frame becomes current
        new_dyn = torch.zeros_like(dyn)
        new_dyn[0] = x_pred
        new_dyn[1:] = dyn[:-1]

        new_ref = torch.zeros_like(ref)
        # Shift reference forward (approximate: keep current reference pattern)
        new_ref[0] = ref[0]  # next reference stays same (would need future ref)
        new_ref[1:] = ref[:-1]

        dyn = new_dyn
        ref = new_ref

    return predictions


def train_stage2(net, writer, data_path_root, out_weight_folder,
                 train_seq_num=10, test_seq_num=10):
    print("=" * 60)
    print("Stage 2: Self-Supervised Physics Training")
    print("=" * 60)

    physics_loss_fn = PhysicsLoss()
    if torch.cuda.is_available():
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=STAGE2_LR)
    net.train()

    train_data_path = os.path.join(data_path_root, "train")
    train_files = sorted([f for f in listdir(train_data_path) if f.startswith("motion_")])

    for epoch in range(STAGE2_EPOCHS):
        lr = STAGE2_LR * (STAGE2_DECAY ** epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Progressive rollout: start K=1, increase over epochs
        K = min(1 + epoch // STAGE2_ROLLOUT_WARMUP, STAGE2_MAX_ROLLOUT)
        print(f"\n[Stage 2] Epoch {epoch}  lr={lr:.8f}  K={K}")
        writer.add_scalar('Stage2/LearningRate', lr, epoch)
        writer.add_scalar('Stage2/RolloutK', K, epoch)

        train_loss_sum, train_count = 0.0, 0
        shuffle(train_files)
        for train_file in train_files:
            mesh_path = os.path.join(train_data_path, train_file)
            frame_num = len([f for f in listdir(mesh_path + "/1/") if f.startswith("x_")]) - 1
            dataset = data_loader.MeshDataset(mesh_path, train_seq_num, frame_num)
            loader = DataLoader(dataset, batch_size=STAGE2_BATCH, shuffle=True, num_workers=2)

            adj = dataset.adj_matrix
            if torch.cuda.is_available():
                adj = adj.cuda()
            ms_edges = build_multiscale_edges(adj, NUM_SCALES)

            for constraint, dyn_batch, ref_batch, adj_mat, stiff_batch, mass_batch, gt in loader:
                B = dyn_batch.shape[0]
                optimizer.zero_grad()
                batch_loss = 0.0

                for b in range(B):
                    constraint_b = constraint[0]
                    dyn = dyn_batch[b]       # (T+1, V, 3)
                    ref = ref_batch[b]       # (T+1, V, 3)
                    stiff = stiff_batch[b]   # (V, 1)
                    mass_val = mass_batch[b]  # (V, 1)

                    if torch.cuda.is_available():
                        constraint_b = constraint_b.cuda()
                        dyn, ref = dyn.cuda(), ref.cuda()
                        stiff, mass_val = stiff.cuda(), mass_val.cuda()

                    # K-step rollout
                    preds = rollout_k_steps(
                        net, constraint_b, dyn, ref,
                        adj_mat[0] if adj_mat.dim() > 2 else adj_mat,
                        stiff, mass_val, ms_edges, K)

                    # Accumulate physics loss over K steps
                    total_phys_loss = 0.0
                    x_prev = dyn[1, :, :]  # t-1 positions
                    x_curr = dyn[0, :, :]  # t positions
                    for x_next in preds:
                        step_loss = physics_loss_fn(x_next, x_curr, x_prev, mass_val)
                        total_phys_loss = total_phys_loss + step_loss
                        x_prev = x_curr
                        x_curr = x_next

                    # Accumulate gradients across batch
                    (total_phys_loss / B).backward()
                    batch_loss += total_phys_loss.item()

                optimizer.step()
                train_loss_sum += batch_loss / B
                train_count += 1

        avg_loss = train_loss_sum / max(train_count, 1)
        print(f"  physics loss: {avg_loss:.6f}")
        writer.add_scalar('Stage2/Loss/physics', avg_loss, epoch)

        path = os.path.join(out_weight_folder, f"stage2_{epoch:04d}.weight")
        torch.save(net.state_dict(), path)

    return net


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("################ Training #####################")

    out_weight_folder = "./weight/"
    data_path_root = "../data/sphere_dataset/"

    if not os.path.exists(out_weight_folder):
        os.makedirs(out_weight_folder)

    net = CausalSpatiotemporalModel()
    writer = SummaryWriter('./runs/')

    # Stage 1: Supervised
    net = train_stage1(net, writer, data_path_root, out_weight_folder,
                       train_seq_num=7, test_seq_num=7)

    # Stage 2: Self-Supervised
    net = train_stage2(net, writer, data_path_root, out_weight_folder,
                       train_seq_num=7, test_seq_num=7)

    writer.close()
    print("Training complete.")
