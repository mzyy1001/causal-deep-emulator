"""
V3 training: Multiple convergence strategies tested.

Key changes from V2:
- Gradient accumulation to simulate larger effective batch
- LR warmup + cosine annealing
- Gradient clipping
- Configurable NUM_SCALES override
"""
import os
import time
import argparse
from os import listdir

import torch
import numpy as np
from model import CausalSpatiotemporalModel, PhysicsLoss, build_multiscale_edges
import data_loader
from data_loader import SelfSupervisedDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from random import shuffle


def train_stage1_v3(net, writer, data_path_roots, out_weight_folder,
                    train_seq_num=7, test_seq_num=7,
                    epochs=60, lr=1e-3, batch_size=16, num_scales=1,
                    grad_accum=1, warmup_epochs=5, grad_clip=1.0):
    """Improved Stage 1 with warmup, grad accum, clipping."""
    print("=" * 60)
    print("Stage 1 V3: epochs=%d lr=%.1e batch=%d scales=%d accum=%d" % (
        epochs, lr, batch_size, num_scales, grad_accum))
    print("=" * 60)

    if torch.cuda.is_available():
        net = net.cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)

    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    net.train()

    # Collect data
    all_train_files = []
    all_test_files = []
    for root in data_path_roots:
        train_path = os.path.join(root, "train")
        test_path = os.path.join(root, "test")
        if os.path.isdir(train_path):
            for f in sorted(listdir(train_path)):
                if f.startswith("motion_"):
                    all_train_files.append(os.path.join(train_path, f))
        if os.path.isdir(test_path):
            for f in sorted(listdir(test_path)):
                if f.startswith("motion_"):
                    all_test_files.append(os.path.join(test_path, f))

    print("  Train motions: %d, Test motions: %d" % (len(all_train_files), len(all_test_files)))
    best_test = float('inf')

    for epoch in range(epochs):
        cur_lr = optimizer.param_groups[0]['lr']
        print("\n[Stage 1] Epoch %d  lr=%.6f" % (epoch, cur_lr))
        writer.add_scalar('Stage1/LearningRate', cur_lr, epoch)

        # Train
        net.train()
        train_loss_sum, train_count = 0.0, 0
        epoch_start = time.time()
        shuffle(all_train_files)

        for mesh_path in all_train_files:
            frame_num = len([f for f in listdir(mesh_path + "/1/") if f.startswith("x_")]) - 1
            dataset = data_loader.MeshDataset(mesh_path, train_seq_num, frame_num)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

            adj = dataset.adj_matrix
            if torch.cuda.is_available():
                adj = adj.cuda()
            ms_edges = build_multiscale_edges(adj, num_scales)

            optimizer.zero_grad()
            accum_count = 0

            for constraint, dyn, ref, adj_mat, stiff, mass, gt in loader:
                if torch.cuda.is_available():
                    dyn, ref, stiff, mass, gt = (
                        dyn.cuda(), ref.cuda(), stiff.cuda(), mass.cuda(), gt.cuda())
                    constraint_b = constraint[0].cuda()
                else:
                    constraint_b = constraint[0]

                pred = net.forward_batch(
                    constraint_b, dyn, ref, adj_mat[0], stiff, mass, ms_edges)
                target = gt[:, constraint_b == 0, :]
                loss = torch.mean((pred - target) ** 2) / grad_accum

                loss.backward()
                accum_count += 1

                if accum_count >= grad_accum:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0

                train_loss_sum += loss.item() * grad_accum
                train_count += 1

            # Handle remaining gradients
            if accum_count > 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        epoch_time = time.time() - epoch_start
        avg_train = train_loss_sum / max(train_count, 1)
        print("  train: %.6f  time: %.1fs" % (avg_train, epoch_time))
        writer.add_scalar('Stage1/Loss/train', avg_train, epoch)

        # Test
        net.eval()
        test_loss_sum, test_count = 0.0, 0
        with torch.no_grad():
            for mesh_path in all_test_files:
                frame_num = len([f for f in listdir(mesh_path + "/1/") if f.startswith("x_")]) - 1
                dataset = data_loader.MeshDataset(mesh_path, test_seq_num, frame_num)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

                adj = dataset.adj_matrix
                if torch.cuda.is_available():
                    adj = adj.cuda()
                ms_edges = build_multiscale_edges(adj, num_scales)

                for constraint, dyn, ref, adj_mat, stiff, mass, gt in loader:
                    if torch.cuda.is_available():
                        dyn, ref, stiff, mass, gt = (
                            dyn.cuda(), ref.cuda(), stiff.cuda(), mass.cuda(), gt.cuda())
                        constraint_b = constraint[0].cuda()
                    else:
                        constraint_b = constraint[0]

                    pred = net.forward_batch(
                        constraint_b, dyn, ref, adj_mat[0], stiff, mass, ms_edges)
                    target = gt[:, constraint_b == 0, :]
                    loss = torch.mean((pred - target) ** 2)
                    test_loss_sum += loss.item()
                    test_count += 1

        avg_test = test_loss_sum / max(test_count, 1)
        print("  test:  %.6f%s" % (avg_test, " *best*" if avg_test < best_test else ""))
        writer.add_scalar('Stage1/Loss/test', avg_test, epoch)

        if avg_test < best_test:
            best_test = avg_test
            torch.save(net.state_dict(), os.path.join(out_weight_folder, "best_stage1.weight"))

        torch.save(net.state_dict(), os.path.join(out_weight_folder, "stage1_%04d.weight" % epoch))
        scheduler.step()

    print("\nBest test loss: %.6f" % best_test)
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', default='../data/sphere_dataset/')
    parser.add_argument('--weight_dir', default='./weight_v3/')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--num_scales', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    print("################ V3 Training #####################")
    os.makedirs(args.weight_dir, exist_ok=True)

    # Override NUM_SCALES in config at runtime
    import config
    config.NUM_SCALES = args.num_scales
    # Reload model with new config
    import importlib
    import model as model_module
    importlib.reload(model_module)
    from model import CausalSpatiotemporalModel

    net = CausalSpatiotemporalModel()
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print("Resumed from %s" % args.resume)

    data_roots = [p.strip() for p in args.data_paths.split(',')]
    writer = SummaryWriter(os.path.join(args.weight_dir, 'runs/'))

    net = train_stage1_v3(
        net, writer, data_roots, args.weight_dir,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch,
        num_scales=args.num_scales, grad_accum=args.grad_accum,
        warmup_epochs=args.warmup, grad_clip=args.grad_clip)

    writer.close()
    print("V3 Stage 1 complete.")
