"""
Overfit test: Can the causal model memorize a single sphere sequence?
If not, there's likely an architecture or optimization bug.

Also tests: causal cone as soft prior (additive penalty) vs hard gate.
"""
import os
import torch
import numpy as np
from model import CausalSpatiotemporalModel, build_multiscale_edges
from config import NUM_SCALES, TEMPORAL_WINDOW
import data_loader
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '../data/sphere_dataset/train/motion_1'
SEQ_NUM = 1  # single sequence
EPOCHS = 200
LR = 1e-3
BATCH = 4

print('='*70)
print('  OVERFIT TEST: Single sphere sequence')
print('='*70)

# Load dataset (1 sequence only)
frame_num = len([f for f in os.listdir(DATA_PATH + '/1/') if f.startswith('x_')]) - 1
dataset = data_loader.MeshDataset(DATA_PATH, SEQ_NUM, frame_num)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
print('Samples: %d, Frames: %d' % (len(dataset), frame_num))

adj = dataset.adj_matrix.to(device)
ms_edges = build_multiscale_edges(adj, NUM_SCALES)

# Train fresh model
net = CausalSpatiotemporalModel()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

best_loss = float('inf')
for epoch in range(EPOCHS):
    net.train()
    epoch_loss = 0.0
    count = 0
    for constraint, dyn, ref, adj_mat, stiff, mass, gt in loader:
        dyn, ref, stiff, mass, gt = dyn.to(device), ref.to(device), stiff.to(device), mass.to(device), gt.to(device)
        constraint_b = constraint[0].to(device)

        pred = net.forward_batch(constraint_b, dyn, ref, adj_mat[0], stiff, mass, ms_edges)
        target = gt[:, constraint_b == 0, :]
        loss = torch.mean((pred - target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        count += 1

    avg = epoch_loss / max(count, 1)
    if avg < best_loss:
        best_loss = avg
    if epoch % 10 == 0 or epoch < 5:
        print('  Epoch %3d: loss=%.8f  best=%.8f' % (epoch, avg, best_loss))

print('\nFinal overfit loss: %.8f' % best_loss)
if best_loss < 0.001:
    print('PASS: Model can memorize the sequence.')
elif best_loss < 0.01:
    print('PARTIAL: Model learns but doesn\'t fully converge.')
else:
    print('FAIL: Model cannot memorize even one sequence. Architecture or optimization bug likely.')

# Now test: what if we increase learning rate and train longer?
print('\n' + '='*70)
print('  OVERFIT TEST 2: Higher LR (1e-2), no weight decay')
print('='*70)

net2 = CausalSpatiotemporalModel().to(device)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-2)

best_loss2 = float('inf')
for epoch in range(100):
    net2.train()
    epoch_loss = 0.0
    count = 0
    for constraint, dyn, ref, adj_mat, stiff, mass, gt in loader:
        dyn, ref, stiff, mass, gt = dyn.to(device), ref.to(device), stiff.to(device), mass.to(device), gt.to(device)
        constraint_b = constraint[0].to(device)

        pred = net2.forward_batch(constraint_b, dyn, ref, adj_mat[0], stiff, mass, ms_edges)
        target = gt[:, constraint_b == 0, :]
        loss = torch.mean((pred - target) ** 2)

        optimizer2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net2.parameters(), 1.0)
        optimizer2.step()
        epoch_loss += loss.item()
        count += 1

    avg = epoch_loss / max(count, 1)
    if avg < best_loss2:
        best_loss2 = avg
    if epoch % 10 == 0:
        print('  Epoch %3d: loss=%.8f  best=%.8f' % (epoch, avg, best_loss2))

print('\nFinal overfit loss (high LR): %.8f' % best_loss2)

# Test 3: Check if the problem is the Tanh activations in dynamics MLP
# Tanh squashes output to [-1,1] but GT deltas can be large
print('\n' + '='*70)
print('  DIAGNOSTIC: GT displacement increment statistics')
print('='*70)

all_gt = []
for constraint, dyn, ref, adj_mat, stiff, mass, gt in loader:
    constraint_b = constraint[0]
    target = gt[:, constraint_b == 0, :]
    all_gt.append(target.numpy())

all_gt = np.concatenate(all_gt, axis=0)
print('GT delta_u shape: %s' % str(all_gt.shape))
print('GT delta_u mean: %.8f' % np.mean(np.abs(all_gt)))
print('GT delta_u std: %.8f' % np.std(all_gt))
print('GT delta_u max: %.8f' % np.max(np.abs(all_gt)))
print('GT delta_u min nonzero: %.8f' % np.min(np.abs(all_gt[np.abs(all_gt) > 0])) if np.any(all_gt != 0) else 'all zero')
print('GT delta_u fraction zero: %.4f' % (np.sum(np.abs(all_gt) < 1e-10) / all_gt.size))

# Check what the model actually predicts
net.eval()
with torch.no_grad():
    constraint, dyn, ref, adj_mat, stiff, mass, gt = dataset[len(dataset)//2]
    dyn, ref, stiff, mass = dyn.unsqueeze(0).to(device), ref.unsqueeze(0).to(device), stiff.unsqueeze(0).to(device), mass.unsqueeze(0).to(device)
    constraint = constraint.to(device)
    pred = net.forward_batch(constraint, dyn, ref, adj_mat, stiff, mass, ms_edges)
    pred_np = pred.cpu().numpy().flatten()
    gt_np = gt[constraint.cpu() == 0, :].numpy().flatten()
    print('\nPred stats: mean=%.8f std=%.8f max=%.8f' % (np.mean(np.abs(pred_np)), np.std(pred_np), np.max(np.abs(pred_np))))
    print('GT   stats: mean=%.8f std=%.8f max=%.8f' % (np.mean(np.abs(gt_np)), np.std(gt_np), np.max(np.abs(gt_np))))
    if np.max(np.abs(gt_np)) > 1.0:
        print('\nWARNING: GT deltas exceed 1.0 — Tanh output layer caps predictions at [-1,1]!')
        print('This is likely the ROOT CAUSE of the performance gap.')

print('\nDone!')
