import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import TEMPORAL_WINDOW, NUM_SCALES, MSG_DIM, PROP_DIM, MASK_SHARPNESS


def build_multiscale_edges(adj_matrix, num_scales):
    """Build multi-scale edge sets from the original adjacency matrix.

    Args:
        adj_matrix: (V, max_neighbors) 1-indexed adjacency, 0 = padding.
        num_scales: number of coarsening levels L (total scales = L+1).

    Returns:
        List of edge index tensors [(2, E_s) for s in 0..L], 0-indexed.
    """
    V = adj_matrix.shape[0]
    device = adj_matrix.device

    # Build sparse adjacency set per vertex (0-indexed, excluding padding)
    neighbors = [set() for _ in range(V)]
    for i in range(V):
        for j in range(adj_matrix.shape[1]):
            nb = adj_matrix[i, j].item()
            if nb > 0:
                neighbors[i].add(nb - 1)  # convert to 0-indexed

    # Scale 0: original 1-ring edges
    edge_sets = []
    src_list, dst_list = [], []
    for i in range(V):
        for nb in neighbors[i]:
            src_list.append(i)
            dst_list.append(nb)
    if len(src_list) > 0:
        edge_sets.append(torch.tensor([src_list, dst_list], dtype=torch.long, device=device))
    else:
        edge_sets.append(torch.zeros(2, 0, dtype=torch.long, device=device))

    # Higher scales: expand by one hop each level
    reachable = [set(neighbors[i]) for i in range(V)]
    for s in range(1, num_scales + 1):
        new_reachable = [set(r) for r in reachable]
        for i in range(V):
            for nb in list(reachable[i]):
                new_reachable[i].update(neighbors[nb])
            new_reachable[i].discard(i)
        # New edges = edges in this scale but not in the previous
        src_list, dst_list = [], []
        for i in range(V):
            new_edges = new_reachable[i] - reachable[i]
            for nb in new_edges:
                src_list.append(i)
                dst_list.append(nb)
        if len(src_list) > 0:
            edge_sets.append(torch.tensor([src_list, dst_list], dtype=torch.long, device=device))
        else:
            edge_sets.append(torch.zeros(2, 0, dtype=torch.long, device=device))
        reachable = new_reachable

    return edge_sets


class MSEAAggregation(nn.Module):
    """Single-scale message passing: edge encoding + node update."""

    def __init__(self, node_dim, edge_hidden=128, node_hidden=128):
        super().__init__()
        # Edge MLP: takes [src_feat, dst_feat, rel_pos] -> message
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, node_hidden),
        )
        # Node update MLP: takes [node_feat, aggregated_msg] -> updated_feat
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + node_hidden, node_hidden),
            nn.ReLU(),
            nn.Linear(node_hidden, node_hidden),
        )
        self.node_hidden = node_hidden

    def forward(self, h, edge_index):
        """
        Args:
            h: (V, D) node features
            edge_index: (2, E) source-target edge indices, 0-indexed

        Returns:
            m: (V, node_hidden) aggregated per-node representation
        """
        V, D = h.shape
        device = h.device

        if edge_index.shape[1] == 0:
            return torch.zeros(V, self.node_hidden, device=device)

        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)  # (E, 2D)
        messages = self.edge_mlp(edge_feat)  # (E, node_hidden)

        # Scatter-add messages to destination nodes
        agg = torch.zeros(V, self.node_hidden, device=device)
        agg.index_add_(0, dst, messages)

        out = self.node_mlp(torch.cat([h, agg], dim=-1))  # (V, node_hidden)
        return out


class CausalConeModule(nn.Module):
    """Learn per-vertex propagation velocity and map to max scale index."""

    def __init__(self, prop_dim, num_scales, scale_radii=None):
        """
        Args:
            prop_dim: dimension of per-vertex property features (theta_i).
            num_scales: L, number of coarsening levels.
            scale_radii: optional list of characteristic radii per scale.
                         If None, uses [1, 2, ..., L+1].
        """
        super().__init__()
        self.num_scales = num_scales
        self.velocity_mlp = nn.Sequential(
            nn.Linear(prop_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # ensure positive velocity
        )
        if scale_radii is not None:
            self.register_buffer('scale_radii', torch.tensor(scale_radii, dtype=torch.float32))
        else:
            self.register_buffer('scale_radii',
                                 torch.arange(1, num_scales + 2, dtype=torch.float32))

    def forward(self, theta, tau):
        """
        Args:
            theta: (V, prop_dim) per-vertex properties
            tau: scalar or (T,) temporal delays

        Returns:
            s_max: (V, T) continuous max admissible scale (float, for soft masking)
        """
        v = self.velocity_mlp(theta)  # (V, 1)
        if isinstance(tau, (int, float)):
            tau = torch.tensor([tau], device=theta.device, dtype=torch.float32)
        tau = tau.float()  # (T,)
        radius = v * tau.unsqueeze(0)  # (V, T)

        # Continuous scale index: interpolate between discrete scale radii.
        # For each (vertex, tau), find the continuous position along the scale_radii axis.
        # scale_radii: (L+1,) e.g. [1, 2, 3, 4]
        # Result is a float in [0, L] indicating how far the radius reaches.
        # We use a sum-of-sigmoids formulation for smooth differentiability:
        #   s_max = sum_s sigmoid(sharpness * (radius - scale_radii[s]))
        # This smoothly counts how many scale thresholds the radius exceeds.
        diffs = radius.unsqueeze(-1) - self.scale_radii.unsqueeze(0).unsqueeze(0)  # (V, T, L+1)
        s_max = torch.sigmoid(MASK_SHARPNESS * diffs).sum(dim=-1) - 1.0  # (V, T)
        s_max = s_max.clamp(min=0.0)
        return s_max


class SpatiotemporalAttention(nn.Module):
    """Lightweight attention for weighting multi-scale spatiotemporal messages."""

    def __init__(self, node_dim, msg_dim, prop_dim):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(node_dim + msg_dim + 1 + prop_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, h_t, messages, tau_values, theta):
        """
        Args:
            h_t: (V, node_dim) current node features
            messages: (V, T, S, msg_dim) multi-scale temporal messages
            tau_values: (T,) temporal delay values
            theta: (V, prop_dim) per-vertex properties

        Returns:
            weights: (V, T, S) attention weights (softmax over valid entries)
        """
        V, T, S, D = messages.shape
        h_exp = h_t.unsqueeze(1).unsqueeze(2).expand(V, T, S, -1)
        tau_exp = tau_values.unsqueeze(0).unsqueeze(2).expand(V, T, S).unsqueeze(-1)
        theta_exp = theta.unsqueeze(1).unsqueeze(2).expand(V, T, S, -1)

        attn_input = torch.cat([h_exp, messages, tau_exp, theta_exp], dim=-1)
        logits = self.attn_mlp(attn_input).squeeze(-1)  # (V, T, S)
        return logits  # raw logits; masking + softmax applied externally


class CausalSpatiotemporalModel(nn.Module):
    """
    Full model implementing:
    - Multi-Scale Edge Aggregation (MSEA)
    - Causal Cone Masking
    - Spatiotemporal Region Aggregation
    - Dynamics Update
    """

    def __init__(self, node_input_dim=3, ref_input_dim=3):
        """
        Args:
            node_input_dim: dimension of per-vertex dynamic features per frame
            ref_input_dim: dimension of per-vertex reference features per frame
        """
        super().__init__()
        num_scales = NUM_SCALES
        prop_dim = PROP_DIM
        msg_dim = MSG_DIM
        self.num_scales = num_scales
        self.msg_dim = msg_dim

        # Feature encoder: per-frame vertex features -> hidden dim
        self.feat_encoder = nn.Sequential(
            nn.Linear(node_input_dim + ref_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, msg_dim),
        )

        # Property encoder: stiffness, mass, constraint -> theta
        self.prop_encoder = nn.Sequential(
            nn.Linear(3, 64),  # [stiffness, mass, constraint]
            nn.ReLU(),
            nn.Linear(64, prop_dim),
        )

        # MSEA aggregators: one per scale
        self.msea_layers = nn.ModuleList([
            MSEAAggregation(msg_dim, edge_hidden=128, node_hidden=msg_dim)
            for _ in range(num_scales + 1)
        ])

        # Causal cone
        self.causal_cone = CausalConeModule(prop_dim, num_scales)

        # Spatiotemporal attention
        self.st_attention = SpatiotemporalAttention(msg_dim, msg_dim, prop_dim)

        # Current state encoder
        self.current_encoder = nn.Sequential(
            nn.Linear(node_input_dim + ref_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, msg_dim),
        )

        # Dynamics update: predicts displacement increment
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(msg_dim + msg_dim, 256),  # [current_state, context]
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
        )

    def encode_frame(self, dynamic_f, reference_f):
        """Encode a single frame's features.

        Args:
            dynamic_f: (V, 3) displacement at this frame
            reference_f: (V, 3) reference position at this frame

        Returns:
            h: (V, msg_dim)
        """
        feat = torch.cat([dynamic_f, reference_f], dim=-1)
        return self.feat_encoder(feat)

    def form_vertex_properties(self, constraint, stiffness, mass):
        """Form per-vertex property features theta.

        Args:
            constraint: (V,) binary constraint mask
            stiffness: (V, 1) stiffness values
            mass: (V, 1) mass values

        Returns:
            theta: (V, prop_dim)
        """
        props = torch.cat([
            stiffness,
            mass,
            constraint.float().unsqueeze(-1)
        ], dim=-1)  # (V, 3)
        return self.prop_encoder(props)

    def forward(self, constraint, dynamic_frames, reference_frames, adj_matrix,
                stiffness, mass, multiscale_edges=None):
        """
        Args:
            constraint: (V,) binary constraint mask
            dynamic_frames: (T+1, V, 3) displacements at frames [t, t-1, ..., t-T]
                            index 0 = current frame t
            reference_frames: (T+1, V, 3) reference positions at same frames
            adj_matrix: (V, max_neighbors) original 1-indexed adjacency
            stiffness: (V, 1) per-vertex stiffness
            mass: (V, 1) per-vertex mass
            multiscale_edges: optional precomputed list of edge index tensors

        Returns:
            delta_u: (V_free, 3) predicted displacement increment for unconstrained vertices
        """
        device = dynamic_frames.device
        T = TEMPORAL_WINDOW
        num_frames = dynamic_frames.shape[0]
        V = dynamic_frames.shape[1]

        # Build multi-scale edges if not precomputed
        if multiscale_edges is None:
            multiscale_edges = build_multiscale_edges(adj_matrix, self.num_scales)

        # Vertex properties
        theta = self.form_vertex_properties(constraint, stiffness, mass)  # (V, prop_dim)

        # Encode historical frames and run MSEA at each scale
        # messages[tau][s] = (V, msg_dim)
        all_messages = torch.zeros(V, T, self.num_scales + 1, self.msg_dim, device=device)

        for tau_idx in range(T):
            frame_idx = tau_idx + 1  # tau=1 corresponds to frame t-1, which is index 1
            if frame_idx >= num_frames:
                break
            h_frame = self.encode_frame(
                dynamic_frames[frame_idx], reference_frames[frame_idx]
            )  # (V, msg_dim)

            for s in range(self.num_scales + 1):
                m_s = self.msea_layers[s](h_frame, multiscale_edges[s])  # (V, msg_dim)
                all_messages[:, tau_idx, s, :] = m_s

        # Causal cone masking (smooth sigmoid mask for gradient flow)
        tau_values = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        s_max = self.causal_cone(theta, tau_values)  # (V, T), continuous float

        # Smooth mask: M(i,tau,s) = sigmoid(sharpness * (s_max(i,tau) - s))
        # Approaches 1 when s << s_max, approaches 0 when s >> s_max.
        # Fully differentiable — no vanishing gradient from hard cutoff.
        scale_indices = torch.arange(self.num_scales + 1, device=device).float()  # (S,)
        mask = torch.sigmoid(
            MASK_SHARPNESS * (s_max.unsqueeze(-1) - scale_indices.unsqueeze(0).unsqueeze(0))
        )  # (V, T, S)

        # Attention weights
        attn_logits = self.st_attention(
            self.encode_frame(dynamic_frames[0], reference_frames[0]),
            all_messages, tau_values, theta
        )  # (V, T, S)

        # Multiply logits by soft mask, then softmax over all (tau, scale) entries
        masked_logits = attn_logits + torch.log(mask + 1e-8)
        V_dim, T_dim, S_dim = masked_logits.shape
        weights = F.softmax(masked_logits.reshape(V_dim, -1), dim=-1).reshape(V_dim, T_dim, S_dim)

        # Weighted aggregation
        # (V, T, S, 1) * (V, T, S, msg_dim) -> sum -> (V, msg_dim)
        context = (weights.unsqueeze(-1) * all_messages).sum(dim=1).sum(dim=1)  # (V, msg_dim)

        # Current state encoding
        current_feat = torch.cat([dynamic_frames[0], reference_frames[0]], dim=-1)
        h_current = self.current_encoder(current_feat)  # (V, msg_dim)

        # Dynamics update
        update_input = torch.cat([h_current, context], dim=-1)
        delta_u = self.dynamics_mlp(update_input)  # (V, 3)

        # Only return for unconstrained vertices
        free_mask = (constraint == 0)
        delta_u_free = delta_u[free_mask]

        return delta_u_free

    def forward_batch(self, constraint, dynamic_frames_batch, reference_frames_batch,
                      adj_matrix, stiffness_batch, mass_batch, multiscale_edges=None):
        """Batched forward pass.

        Args:
            constraint: (V,) binary constraint mask (shared across batch)
            dynamic_frames_batch: (B, T+1, V, 3)
            reference_frames_batch: (B, T+1, V, 3)
            adj_matrix: (V, max_neighbors)
            stiffness_batch: (B, V, 1)
            mass_batch: (B, V, 1)
            multiscale_edges: optional precomputed

        Returns:
            delta_u: (B, V_free, 3)
        """
        B = dynamic_frames_batch.shape[0]
        results = []
        for b in range(B):
            out = self.forward(
                constraint,
                dynamic_frames_batch[b],
                reference_frames_batch[b],
                adj_matrix,
                stiffness_batch[b],
                mass_batch[b],
                multiscale_edges
            )
            results.append(out)
        return torch.stack(results, dim=0)

    def compute_supervised_loss(self, pred, target):
        """MSE loss for supervised training (Stage 1).

        Args:
            pred: (B, V_free, 3) predicted displacement increments
            target: (B, V_free, 3) ground truth displacement increments

        Returns:
            loss: scalar
        """
        return torch.mean((pred - target) ** 2)


class PhysicsLoss(nn.Module):
    """Physics-based self-supervised energy terms for Stage 2 training.

    All energy terms are normalized by vertex/element count to keep
    gradients on a consistent scale regardless of mesh resolution.
    Configurable weights allow balancing the relative contribution
    of each term.
    """

    def __init__(self, gravity=9.81, w_inertia=1.0, w_gravity=0.01, w_strain=1.0):
        super().__init__()
        self.gravity = gravity
        self.w_inertia = w_inertia
        self.w_gravity = w_gravity
        self.w_strain = w_strain

    def inertia_loss(self, pos_next, pos_curr, pos_prev, mass):
        """Inertial potential energy (mean over vertices).

        Args:
            pos_next: (V, 3) predicted next positions
            pos_curr: (V, 3) current positions
            pos_prev: (V, 3) previous positions
            mass: (V, 1) per-vertex mass

        Returns:
            scalar loss
        """
        accel = pos_next + pos_prev - 2 * pos_curr  # (V, 3)
        energy = 0.5 * (mass * accel * accel).mean()
        return energy

    def gravity_loss(self, pos_curr, mass):
        """Gravitational potential energy (mean over vertices, y-axis up).

        Args:
            pos_curr: (V, 3) current positions (actual world positions)
            mass: (V, 1) per-vertex mass

        Returns:
            scalar loss
        """
        return -(mass * self.gravity * pos_curr[:, 1:2]).mean()

    def strain_loss(self, x, elements, rest_volumes, rest_inv, lam, mu):
        """Neo-Hookean strain energy (mean over elements).

        Args:
            x: (V, 3) deformed vertex positions
            elements: (E, 4) tetrahedral element indices
            rest_volumes: (E,) rest volume per element
            rest_inv: (E, 3, 3) inverse of rest shape matrix per element
            lam: Lame's first parameter
            mu: Lame's second parameter (shear modulus)

        Returns:
            scalar loss
        """
        # Compute deformation gradient per element
        v0 = x[elements[:, 0]]  # (E, 3)
        v1 = x[elements[:, 1]]
        v2 = x[elements[:, 2]]
        v3 = x[elements[:, 3]]

        # Deformed shape matrix
        Ds = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)  # (E, 3, 3)
        F_def = torch.bmm(Ds, rest_inv)  # (E, 3, 3)

        # Neo-Hookean energy
        det_F = torch.det(F_def)  # (E,)
        det_F = det_F.clamp(min=1e-8)
        log_det = torch.log(det_F)

        FtF = torch.bmm(F_def.transpose(1, 2), F_def)
        tr_FtF = FtF[:, 0, 0] + FtF[:, 1, 1] + FtF[:, 2, 2]

        psi = 0.5 * lam * log_det ** 2 - mu * log_det + 0.5 * mu * (tr_FtF - 3)
        energy = (psi * rest_volumes).mean()
        return energy

    def forward(self, pos_next, pos_curr, pos_prev, mass, elements=None,
                rest_volumes=None, rest_inv=None, lam=1.0, mu=1.0):
        """Combined physics loss on actual world positions.

        Args:
            pos_next: (V, 3) predicted next world positions
            pos_curr: (V, 3) current world positions
            pos_prev: (V, 3) previous world positions
            mass: (V, 1) per-vertex mass
            elements: optional (E, 4) tet indices for strain
            rest_volumes: optional (E,) rest volumes
            rest_inv: optional (E, 3, 3) inverse rest shape matrices
            lam: Lame parameter lambda
            mu: Lame parameter mu

        Returns:
            total_loss: scalar
        """
        loss = self.w_inertia * self.inertia_loss(pos_next, pos_curr, pos_prev, mass)
        loss = loss + self.w_gravity * self.gravity_loss(pos_curr, mass)

        if elements is not None and rest_volumes is not None and rest_inv is not None:
            loss = loss + self.w_strain * self.strain_loss(
                pos_next, elements, rest_volumes, rest_inv, lam, mu)

        return loss
