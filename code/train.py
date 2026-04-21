"""
Two-stage hybrid training for the causal spatiotemporal deep emulator.

Stage 1: Supervised learning with ground-truth displacement increments.
Stage 2: Self-supervised fine-tuning with physics energy minimization
         and progressively lengthened K-step rollout.
"""

import os
import time
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

def collect_motion_files(data_path_roots, split):
    """Collect all motion directories across multiple dataset roots.

    Args:
        data_path_roots: list of dataset root paths (e.g., ['../data/sphere_dataset/'])
        split: 'train' or 'test'

    Returns:
        list of (motion_dir_path, root_path) tuples
    """
    all_files = []
    for root in data_path_roots:
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            continue
        for f in sorted(listdir(split_path)):
            if f.startswith("motion_"):
                all_files.append(os.path.join(split_path, f))
    return all_files


def train_stage1(net, writer, data_path_root, out_weight_folder,
                 train_seq_num=10, test_seq_num=10):
    print("=" * 60)
    print("Stage 1: Supervised Training")
    print("=" * 60)

    if torch.cuda.is_available():
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=STAGE1_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS, eta_min=1e-6)
    net.train()

    # Support single path (string) or multiple paths (list)
    if isinstance(data_path_root, str):
        data_path_roots = [data_path_root]
    else:
        data_path_roots = data_path_root

    train_files = collect_motion_files(data_path_roots, "train")
    test_files = collect_motion_files(data_path_roots, "test")
    print(f"  Training motions: {len(train_files)} from {len(data_path_roots)} dataset(s)")
    print(f"  Test motions: {len(test_files)}")

    for epoch in range(STAGE1_EPOCHS):
        lr = optimizer.param_groups[0]['lr']
        print(f"\n[Stage 1] Epoch {epoch}  lr={lr:.6f}")
        writer.add_scalar('Stage1/LearningRate', lr, epoch)

        # ── Train ──
        net.train()
        train_loss_sum, train_count = 0.0, 0
        epoch_start = time.time()
        forward_time_sum = 0.0
        shuffle(train_files)
        for mesh_path in train_files:
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

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd_start = time.time()

                pred = net.forward_batch(
                    constraint, dyn, ref, adj_mat[0], stiff, mass, ms_edges)
                target = gt[:, constraint == 0, :]
                loss = net.compute_supervised_loss(pred, target)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_time_sum += time.time() - t_fwd_start

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
                train_count += 1

        epoch_wall = time.time() - epoch_start
        avg_train = train_loss_sum / max(train_count, 1)
        avg_fwd = forward_time_sum / max(train_count, 1)
        print(f"  train loss: {avg_train:.6f}  epoch: {epoch_wall:.1f}s  avg_fwd: {avg_fwd*1000:.1f}ms")
        writer.add_scalar('Stage1/Loss/train', avg_train, epoch)
        writer.add_scalar('Stage1/Time/epoch_sec', epoch_wall, epoch)
        writer.add_scalar('Stage1/Time/avg_forward_ms', avg_fwd * 1000, epoch)

        # ── Test ──
        net.eval()
        test_loss_sum, test_count = 0.0, 0
        with torch.no_grad():
            for mesh_path in test_files:
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
        scheduler.step()

    return net


# ─── Stage 2: Self-Supervised with Physics Energy ─────────────────────────────

def rollout_k_steps(net, constraint, dynamic_frames, reference_frames,
                    adj_matrix, stiffness, mass, ms_edges, K):
    """Run K-step autoregressive rollout, returning predicted displacements.

    Args:
        dynamic_frames: (T+1, V, 3) displacement history [u(t), u(t-1), ...]
        reference_frames: (T+1, V, 3) reference history [x(t+1), x(t), ...]
        K: rollout steps

    Returns:
        pred_displacements: list of K (V, 3) predicted displacement tensors
        ref_positions: list of K (V, 3) corresponding reference positions
            so that actual_position = ref_positions[i] + pred_displacements[i]
    """
    V = dynamic_frames.shape[1]
    free_mask = (constraint == 0)

    # Working copies of frame history
    dyn = dynamic_frames.clone()
    ref = reference_frames.clone()

    pred_displacements = []
    ref_positions = []
    for step in range(K):
        delta_u = net(constraint, dyn, ref, adj_matrix, stiffness, mass, ms_edges)

        # Compute full predicted displacement u(t+1)
        u_pred = torch.zeros(V, 3, device=dyn.device)
        u_pred[free_mask] = dyn[0, free_mask] + delta_u  # u(t) + delta_u
        # For constrained vertices: displacement from reference
        # ref[0] is x(t+1), ref[1] is x(t); constrained displacement stays 0
        u_pred[~free_mask] = 0.0

        pred_displacements.append(u_pred)
        ref_positions.append(ref[0].clone())  # x(t+1) reference position

        # Shift history: new displacement becomes current
        new_dyn = torch.zeros_like(dyn)
        new_dyn[0] = u_pred
        new_dyn[1:] = dyn[:-1]

        new_ref = torch.zeros_like(ref)
        # Shift reference forward (approximate: keep current reference pattern)
        new_ref[0] = ref[0]  # next reference stays same (would need future ref)
        new_ref[1:] = ref[:-1]

        dyn = new_dyn
        ref = new_ref

    return pred_displacements, ref_positions


def train_stage2(net, writer, data_path_root, out_weight_folder,
                 train_seq_num=10, test_seq_num=10,
                 self_supervised_paths=None):
    """Stage 2: Self-supervised physics fine-tuning.

    Args:
        data_path_root: geometry dataset paths (sphere etc.) — uses MeshDataset
        self_supervised_paths: optional list of character motion paths for
            self-supervised training (no GT needed, uses SelfSupervisedDataset)
    """
    print("=" * 60)
    print("Stage 2: Self-Supervised Physics Training")
    print("=" * 60)

    physics_loss_fn = PhysicsLoss()
    if torch.cuda.is_available():
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=STAGE2_LR)
    net.train()

    # Collect training data sources
    if isinstance(data_path_root, str):
        data_path_roots = [data_path_root]
    else:
        data_path_roots = data_path_root

    # Geometry datasets (with GT displacements from FEM)
    train_files = collect_motion_files(data_path_roots, "train")

    # Self-supervised datasets (character motions, no GT needed)
    ss_paths = self_supervised_paths or []
    print(f"  Geometry motions: {len(train_files)} from {len(data_path_roots)} dataset(s)")
    print(f"  Self-supervised motions: {len(ss_paths)} character paths")

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
        epoch_start = time.time()
        forward_time_sum = 0.0

        # Build dataset list: geometry + self-supervised
        all_datasets = []
        shuffle(train_files)
        for mesh_path in train_files:
            frame_num = len([f for f in listdir(mesh_path + "/1/") if f.startswith("x_")]) - 1
            ds = data_loader.MeshDataset(mesh_path, train_seq_num, frame_num)
            all_datasets.append(ds)

        for ss_path in ss_paths:
            # Count frames (flat directory)
            frame_num = len([f for f in listdir(ss_path) if f.startswith("x_")])
            if frame_num < 2:
                continue
            ds = data_loader.SelfSupervisedDataset(ss_path, frame_num)
            all_datasets.append(ds)

        shuffle(all_datasets)
        for dataset in all_datasets:
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

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_fwd_start = time.time()

                    # K-step rollout
                    pred_disps, ref_poss = rollout_k_steps(
                        net, constraint_b, dyn, ref,
                        adj_mat[0] if adj_mat.dim() > 2 else adj_mat,
                        stiff, mass_val, ms_edges, K)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    forward_time_sum += time.time() - t_fwd_start

                    # Accumulate physics loss over K steps
                    # Convert displacements to actual positions: pos = ref + u
                    # dyn = [u(t), u(t-1), ...], ref = [x(t+1), x(t), x(t-1), ...]
                    total_phys_loss = 0.0
                    pos_prev = ref[2, :, :] + dyn[1, :, :]  # x(t-1) + u(t-1)
                    pos_curr = ref[1, :, :] + dyn[0, :, :]  # x(t) + u(t)
                    for u_next, x_ref in zip(pred_disps, ref_poss):
                        pos_next = x_ref + u_next  # actual position
                        step_loss = physics_loss_fn(pos_next, pos_curr, pos_prev, mass_val)
                        total_phys_loss = total_phys_loss + step_loss
                        pos_prev = pos_curr
                        pos_curr = pos_next

                    # Accumulate gradients across batch
                    (total_phys_loss / B).backward()
                    batch_loss += total_phys_loss.item()

                optimizer.step()
                train_loss_sum += batch_loss / B
                train_count += 1

        epoch_wall = time.time() - epoch_start
        avg_loss = train_loss_sum / max(train_count, 1)
        avg_fwd = forward_time_sum / max(train_count, 1)
        print(f"  physics loss: {avg_loss:.6f}  epoch: {epoch_wall:.1f}s  avg_fwd: {avg_fwd*1000:.1f}ms")
        writer.add_scalar('Stage2/Loss/physics', avg_loss, epoch)
        writer.add_scalar('Stage2/Time/epoch_sec', epoch_wall, epoch)
        writer.add_scalar('Stage2/Time/avg_forward_ms', avg_fwd * 1000, epoch)

        path = os.path.join(out_weight_folder, f"stage2_{epoch:04d}.weight")
        torch.save(net.state_dict(), path)

    return net


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train the causal spatiotemporal deep emulator')
    parser.add_argument('--data_paths', default='../data/sphere_dataset/',
                        help='Comma-separated geometry dataset root paths (default: sphere only)')
    parser.add_argument('--stage2_data', default=None,
                        help='Comma-separated character motion paths for Stage 2 '
                             'self-supervised training (e.g., '
                             '../data/character_dataset/michelle/cross_jumps,'
                             '../data/character_dataset/kaya/zombie_scream)')
    parser.add_argument('--weight_dir', default='./weight/', help='Output weight directory')
    parser.add_argument('--train_seq', type=int, default=7, help='Sequences per motion for training')
    parser.add_argument('--test_seq', type=int, default=7, help='Sequences per motion for testing')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    print("################ Training #####################")

    out_weight_folder = args.weight_dir
    data_path_roots = [p.strip() for p in args.data_paths.split(',')]
    print(f"Dataset roots: {data_path_roots}")

    if not os.path.exists(out_weight_folder):
        os.makedirs(out_weight_folder)

    net = CausalSpatiotemporalModel()
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print(f"Resumed from {args.resume}")

    writer = SummaryWriter('./runs/')

    # Stage 1: Supervised
    net = train_stage1(net, writer, data_path_roots, out_weight_folder,
                       train_seq_num=args.train_seq, test_seq_num=args.test_seq)

    # Stage 2: Self-Supervised
    ss_paths = None
    if args.stage2_data:
        ss_paths = [p.strip() for p in args.stage2_data.split(',')]
        print(f"Stage 2 self-supervised paths: {ss_paths}")

    net = train_stage2(net, writer, data_path_roots, out_weight_folder,
                       train_seq_num=args.train_seq, test_seq_num=args.test_seq,
                       self_supervised_paths=ss_paths)

    writer.close()
    print("Training complete.")
