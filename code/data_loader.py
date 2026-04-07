"""
Data loader for the causal spatiotemporal deep emulator.
Supports extended temporal history (T frames) and multi-scale edge construction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from config import TEMPORAL_WINDOW


def loadData_Float(filename):
    data = np.fromfile(filename, dtype=np.float64)
    return np.array(data)


def loadData_Int(filename):
    data = np.fromfile(filename, dtype=np.int32)
    return np.array(data)


class MeshDataset(Dataset):
    """Dataset for training with extended temporal window."""

    def __init__(self, mesh_path_root, seq_num, frame_num):
        """
        Args:
            mesh_path_root: path containing sequence folders
            seq_num: number of sequences to load
            frame_num: number of frames per sequence
        """
        temporal_window = TEMPORAL_WINDOW
        self.constraint, self.adj_matrix = load_topology(mesh_path_root)

        self.dynamic_frames = []   # list of (T+1, V, 3) tensors
        self.reference_frames = [] # list of (T+1, V, 3) tensors
        self.stiffness = []        # list of (V, 1) tensors
        self.mass = []             # list of (V, 1) tensors
        self.output_f = []         # list of (V, 3) tensors (delta_u ground truth)

        for i in range(seq_num):
            seq_path = os.path.join(mesh_path_root, str(i + 1))

            # Load stiffness and mass (constant per sequence)
            k = loadData_Float(os.path.join(seq_path, "k"))
            k = np.expand_dims(k, axis=1) * 0.000001
            m = loadData_Float(os.path.join(seq_path, "m"))
            m = np.expand_dims(m, axis=1) * 1000
            m[0] = 1.0

            for j in range(frame_num - 1):
                # Load T+1 frames of dynamic displacements: [t, t-1, ..., t-T]
                u_frames = []
                for tau in range(temporal_window + 1):
                    idx = max(0, min(j - tau, frame_num - 1))
                    u = loadData_Float(os.path.join(seq_path, f"u_{idx}")).reshape(-1, 3)
                    u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
                    u_frames.append(u)

                # Load T+1 frames of reference positions: [t+1, t, t-1, ..., t-T+1]
                x_frames = []
                for tau in range(-1, temporal_window):
                    idx = max(0, min(j - tau, frame_num - 1))
                    x = loadData_Float(os.path.join(seq_path, f"x_{idx}")).reshape(-1, 3)
                    x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
                    x_frames.append(x)

                # Ground truth: delta_u = u(t+1) - u(t)
                u_next = loadData_Float(os.path.join(seq_path, f"u_{min(j + 1, frame_num - 1)}")).reshape(-1, 3)
                u_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_next], axis=0)
                u_curr = loadData_Float(os.path.join(seq_path, f"u_{j}")).reshape(-1, 3)
                u_curr = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_curr], axis=0)
                delta_u = u_next - u_curr

                self.dynamic_frames.append(np.stack(u_frames, axis=0))    # (T+1, V, 3)
                self.reference_frames.append(np.stack(x_frames, axis=0))  # (T+1, V, 3)
                self.stiffness.append(k)
                self.mass.append(m)
                self.output_f.append(delta_u)

        self.dynamic_frames = torch.from_numpy(np.array(self.dynamic_frames)).float()
        self.reference_frames = torch.from_numpy(np.array(self.reference_frames)).float()
        self.stiffness = torch.from_numpy(np.array(self.stiffness)).float()
        self.mass = torch.from_numpy(np.array(self.mass)).float()
        self.output_f = torch.from_numpy(np.array(self.output_f)).float()

    def __len__(self):
        return self.dynamic_frames.shape[0]

    def __getitem__(self, idx):
        return (self.constraint, self.dynamic_frames[idx], self.reference_frames[idx],
                self.adj_matrix, self.stiffness[idx], self.mass[idx], self.output_f[idx])


def load_topology(mesh_path_root):
    """Load constraint and adjacency (shared across sequences for same mesh)."""
    c_filename = os.path.join(mesh_path_root, "1", "c")
    constraint = loadData_Int(c_filename)
    constraint = torch.from_numpy(constraint).long()

    adj_filename = os.path.join(mesh_path_root, "1", "adj")
    adj_raw = loadData_Int(adj_filename)
    # Infer vertex count from constraint
    V = constraint.shape[0]
    adj_matrix = adj_raw.reshape(V, -1)
    adj_matrix = torch.from_numpy(adj_matrix).long()

    return constraint, adj_matrix


def loadTestInputData(file_path, curr_frame, frame_num):
    """Load test input data with extended temporal window.

    Returns:
        constraint: (V,)
        dynamic_frames: (1, T+1, V, 3)
        reference_frames: (1, T+1, V, 3)
        adj_matrix: (V, max_neighbors)
        stiffness: (1, V, 1)
        mass: (1, V, 1)
    """
    temporal_window = TEMPORAL_WINDOW
    # Dynamic frames: [t, t-1, ..., t-T]
    u_frames = []
    for tau in range(temporal_window + 1):
        idx = max(0, min(curr_frame - tau, frame_num - 1))
        u = loadData_Float(os.path.join(file_path, f"u_{idx}")).reshape(-1, 3)
        u = np.concatenate([np.zeros((1, 3), dtype=np.float64), u], axis=0)
        u_frames.append(u)

    # Reference frames: [t+1, t, t-1, ..., t-T+1]
    x_frames = []
    for tau in range(-1, temporal_window):
        idx = max(0, min(curr_frame - tau, frame_num - 1))
        x = loadData_Float(os.path.join(file_path, f"x_{idx}")).reshape(-1, 3)
        x = np.concatenate([np.zeros((1, 3), dtype=np.float64), x], axis=0)
        x_frames.append(x)

    c_filename = os.path.join(file_path, "c")
    constraint = loadData_Int(c_filename)

    adj_filename = os.path.join(file_path, "adj")
    adj_raw = loadData_Int(adj_filename)
    V = constraint.shape[0]
    adj_matrix = adj_raw.reshape(V, -1)

    k = loadData_Float(os.path.join(file_path, "k"))
    k = np.expand_dims(k, axis=1) * 0.000001
    m = loadData_Float(os.path.join(file_path, "m"))
    m = np.expand_dims(m, axis=1) * 1000
    m[0] = 1.0

    dynamic_frames = np.stack(u_frames, axis=0)[np.newaxis]    # (1, T+1, V, 3)
    reference_frames = np.stack(x_frames, axis=0)[np.newaxis]  # (1, T+1, V, 3)

    constraint = torch.from_numpy(constraint).long()
    dynamic_frames = torch.from_numpy(dynamic_frames).float()
    reference_frames = torch.from_numpy(reference_frames).float()
    adj_matrix = torch.from_numpy(adj_matrix).long()
    stiffness = torch.from_numpy(k[np.newaxis]).float()
    mass = torch.from_numpy(m[np.newaxis]).float()

    return constraint, dynamic_frames, reference_frames, adj_matrix, stiffness, mass


def loadTestOutputData(file_path, curr_frame, frame_num):
    """Load ground truth displacement increment for testing."""
    u_next = loadData_Float(os.path.join(file_path, f"u_{curr_frame + 1}")).reshape(-1, 3)
    u_next = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_next], axis=0)
    u_curr = loadData_Float(os.path.join(file_path, f"u_{curr_frame}")).reshape(-1, 3)
    u_curr = np.concatenate([np.zeros((1, 3), dtype=np.float64), u_curr], axis=0)
    delta_u = u_next - u_curr
    return torch.from_numpy(delta_u[np.newaxis]).float()
