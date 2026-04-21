"""
Original Deep Emulator baseline model (Zheng et al., CVPR 2021).

This module provides the original Graph_MLP architecture for comparison
against the causal spatiotemporal model. It uses 3 frames of history
(u, u-1, u-2) and single-scale 1-ring message passing.

Weights can be loaded from the original checkpoint format (e.g., _0000100.weight).
"""

import torch
import torch.nn as nn
import numpy as np


class Graph_MLP(nn.Module):
    """Original Deep Emulator model (Zheng et al., CVPR 2021)."""

    def __init__(self):
        super(Graph_MLP, self).__init__()

        # mlp beta: edge features (37 = 1 constraint + 18 neighbor + 18 center)
        self.edge_mlp = nn.Sequential(
            nn.Linear(37, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )

        # mlp alpha: point features (18 = 5*3 positions + 1 k + 1 m + 1 k/m)
        self.point_mlp = nn.Sequential(
            nn.Linear(18, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64)
        )

        # mlp gamma: instance prediction (192 = 64 point + 128 edge -> 3 displacement)
        self.instance_mlp = nn.Sequential(
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 192),
            nn.Tanh(),
            nn.Linear(192, 3)
        )

    def form_feature(self, constraint, dynamic_f, reference_f, adj_matrix,
                     stiffness, mass):
        """Form edge and point features from input data.

        Args:
            constraint: (V,) int tensor
            dynamic_f: (B, V, 9) — [u(t), u(t-1), u(t-2)] concatenated
            reference_f: (B, V, 9) — [x(t+1), x(t), x(t-1)] concatenated
            adj_matrix: (V, max_neighbors) 0-indexed adjacency
            stiffness: (B, V, 1)
            mass: (B, V, 1)
        """
        # Ensure all inputs on same device
        dev = dynamic_f.device
        adj_matrix = adj_matrix.to(dev)
        constraint = constraint.to(dev)

        mask = torch.ones(adj_matrix.shape[0], adj_matrix.shape[1], device=dev)
        mask[adj_matrix == 0] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(3)
        mask = mask.expand(dynamic_f.shape[0], -1, -1, -1)

        # Center point features (relative to x(t))
        point_f_u_curr = dynamic_f[:, :, 0:3] - reference_f[:, :, 3:6]
        point_f_u_pre = dynamic_f[:, :, 3:6] - reference_f[:, :, 3:6]
        point_f_u_ppre = dynamic_f[:, :, 6:9] - reference_f[:, :, 3:6]
        point_f_x_next = reference_f[:, :, 0:3] - reference_f[:, :, 3:6]
        point_f_x_pre = reference_f[:, :, 6:9] - reference_f[:, :, 3:6]
        point_f_k = stiffness
        point_f_k_m = stiffness / mass
        point_f = torch.cat((point_f_u_curr, point_f_u_pre, point_f_u_ppre,
                             point_f_x_next, point_f_x_pre,
                             point_f_k, mass, point_f_k_m), 2)

        # Neighbor features
        edge_f_c = constraint[adj_matrix].unsqueeze(0).unsqueeze(3)
        edge_f_c = edge_f_c.expand(dynamic_f.shape[0], -1, -1, -1)

        ref_center = reference_f[:, :, 3:6].unsqueeze(2).expand(
            -1, -1, adj_matrix.shape[1], -1)
        neighbor_f_u_curr = dynamic_f[:, adj_matrix, 0:3] - ref_center
        neighbor_f_u_pre = dynamic_f[:, adj_matrix, 3:6] - ref_center
        neighbor_f_u_ppre = dynamic_f[:, adj_matrix, 6:9] - ref_center
        neighbor_f_x_next = reference_f[:, adj_matrix, 0:3] - ref_center
        neighbor_f_x_curr = reference_f[:, adj_matrix, 3:6] - ref_center
        neighbor_f_x_pre = reference_f[:, adj_matrix, 6:9] - ref_center
        neighbor_f = torch.cat((neighbor_f_u_curr, neighbor_f_u_pre,
                                neighbor_f_u_ppre, neighbor_f_x_next,
                                neighbor_f_x_curr, neighbor_f_x_pre), 3)

        edge_f = torch.cat((edge_f_c.float(), neighbor_f,
                            point_f.unsqueeze(2).expand(
                                -1, -1, adj_matrix.shape[1], -1)), 3)

        point_f = point_f[:, constraint == 0, :]
        edge_f = edge_f[:, constraint == 0, :, :]
        mask = mask[:, constraint == 0, :, :]

        return edge_f, point_f, mask

    def forward(self, constraint, dynamic_f, reference_f, adj_matrix,
                stiffness, mass):
        """Forward pass.

        Args:
            constraint: (V,) int tensor
            dynamic_f: (B, V, 9) — [u(t), u(t-1), u(t-2)]
            reference_f: (B, V, 9) — [x(t+1), x(t), x(t-1)]
            adj_matrix: (V, max_neighbors)
            stiffness: (B, V, 1)
            mass: (B, V, 1)

        Returns:
            output: (B, V_free, 3) predicted displacement increment
        """
        edge_f, point_f, mask = self.form_feature(
            constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass)

        output_edge = self.edge_mlp(edge_f)
        output_edge = output_edge * mask
        output_edge = torch.sum(output_edge, 2)

        output_point = self.point_mlp(point_f)

        input_instance = torch.cat((output_point, output_edge), 2)
        output_instance = self.instance_mlp(input_instance)

        return output_instance

    def compute_graph_loss(self, output_pred, output_f, constraint):
        output_f = output_f[:, constraint == 0, :]
        loss = torch.pow(output_pred - output_f, 2).mean()
        return loss
