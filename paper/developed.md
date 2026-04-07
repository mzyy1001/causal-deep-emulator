# Developed Deep Emulator

## SOTA

**论文**：Wang & Liu, 2024（CAVW）— *Multi-scale edge aggregation mesh-graph-network for character secondary motion*（后文简称 **MSEA-MGN**）

### 训练方式："依赖 GT 形变序列"到"基于能量的自监督/物理监督"
MSEA-MGN 将二次运动动力学写成**能量最小化**形式，并把下列能量项直接作为训练目标（监督信号）：
- 惯性能
- 重力势能
- 应变/弹性能

### 模型结构：
**Multi-scale Edge Aggregation** 模块:
- 构造**嵌套的多层 edge sets**（共享节点，但边的尺度不同，长attention）
- 在不同尺度之间传播信息
- 避免显式 pooling 破坏连通结构


### 表格对比
| 方法 | MSEA-MGN| ASAR | VegaFEM | Deep Emulator（Zheng et al.） | MGN | CFD-GCN | GNS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strain energy (J) | **9323.187** | 12016.274 | 11577.143 | 13347.024 | 10225.146 | 1.6e17 | 2.3e8 |
| Inertia energy (J) | 136.260 | 134.637 | 136.107 | 136.201 | 136.55 | - | - |
| Gravity energy (J) | 189.909 | 187.757 | 189.304 | 189.257 | 189.83 | - | - |
| Rollout-RMSE（ASAR） | **3.097** | - | 3.180 | 3.248 | 3.180 | 110.07 | 43.712 |
| Rollout-RMSE（VegaFEM） | 3.335 | 3.265 | - | **2.918** | 3.346 | 105.12 | 48.028 |

### 相比 Deep Emulator 的提升幅度


- **Strain energy（应变能）**：  
  13347.024 → 9323.187  
  **下降 4023.837 J（约 ↓30.15%）**

- **Rollout-RMSE（ASAR ground truth）**：  
  3.248 → 3.097  
  **下降 0.151（约 ↓4.65%）**

- **Rollout-RMSE（VegaFEM ground truth）**：  
  2.918 → 3.335  
  **上升 0.417（约 ↑14.29%）**  

- **Strain energy**：10225.146 → 9323.187（**↓8.82%**）  
- **Rollout-RMSE（ASAR）**：3.180 → 3.097（**↓2.61%**）

## Notation

To avoid ambiguity, we distinguish three positional quantities throughout:

- $x^{\mathrm{ref}}_t(i) \in \mathbb{R}^3$: the **reference** (primary/skeletal) position of vertex $i$ at frame $t$, prescribed by the driving animation.
- $u_t(i) \in \mathbb{R}^3$: the **secondary displacement** of vertex $i$ at frame $t$, relative to the reference.
- $y_t(i) = x^{\mathrm{ref}}_t(i) + u_t(i)$: the **deformed** (world-space) position.

We further use:

- $\kappa_i$: per-vertex stiffness (to avoid confusion with the rollout horizon $K$).
- $m_i$: per-vertex mass.
- $c_i \in \{0,1\}$: boundary constraint flag ($c_i=1$ if vertex $i$ is constrained).
- $T$: the **input temporal window**, i.e. the number of past frames the model observes in a single forward pass. We set $T=5$.
- $K$: the **autoregressive rollout horizon** used exclusively during Stage 2 self-supervised training. $K$ controls how many future steps the model predicts autoregressively before computing the physics loss; it does *not* affect the temporal discretisation or step size.

**Discrete formulation.** All temporal indices in this work refer to *frame indices*: the temporal delay $\tau \in \{1,\dots,T\}$ counts frames, not physical time. We fix the temporal discretisation to one frame and the spatial scale spacing to one graph-hop unit. Under this discrete formulation, any alternative temporal or spatial step size can be absorbed into the learned propagation speed $v_i$ and/or the scale thresholds $R_s$. Therefore, introducing an additional learnable step-size parameter is unnecessary.

---

## Method

Secondary motion is fundamentally a problem of spatiotemporal influence propagation: the state of a vertex at time $t$ is not determined solely by its local neighborhood at the same instant, but rather by a causally consistent spatiotemporal region extending into the past. Physically, a local perturbation diffuses through the spatial domain over time, so long-range spatial interactions should only become relevant after a sufficient temporal delay. We encode this inductive bias through two complementary mechanisms:

1. **Multi-Scale Edge Aggregation (MSEA):** expands the spatial receptive field at each time step without altering the node set or mesh topology.
2. **Causal Cone Masking:** gates which spatial scales are admissible at each temporal delay, explicitly enforcing the principle that influence propagates outward with finite speed.

The graph-based multi-scale aggregation and causal masking serve as a *learned surrogate for physical influence propagation*: the discrete graph-hop scales approximate the spatial reach of mechanical waves, while the causal cone constrains this reach to grow with temporal delay just as physical wavefronts do. The continuous physics energy in Stage 2 then *regularises* this surrogate so that its predictions remain physically plausible. In this sense, the graph module provides an efficient, differentiable feature extractor whose outputs are disciplined by the energy-based loss.

---

### Multi-Scale Edge Construction

Given the original mesh with vertex set $V$ and adjacency defined by 1-ring neighbors, we construct a hierarchy of $L+1$ edge sets on the same node set:
$$
\mathcal{E} = \{E^{(0)}, E^{(1)}, \dots, E^{(L)}\}, \quad E^{(0)} := E^{\mathrm{fine}}.
$$

Scale $E^{(0)}$ contains the original 1-ring edges. For each subsequent level $\ell \geq 1$, we expand the reachable set by one additional hop: for every vertex $i$, the neighbors of its current reachable set at level $\ell-1$ are added to level $\ell$, excluding self-loops and edges already present at coarser levels. Formally, let $\mathcal{N}^{(\ell)}(i)$ denote the reachable set of vertex $i$ at scale $\ell$:
$$
\mathcal{N}^{(0)}(i) = \{j : (i,j) \in E^{\mathrm{fine}}\}, \quad \mathcal{N}^{(\ell)}(i) = \bigcup_{j \in \mathcal{N}^{(\ell-1)}(i)} \mathcal{N}^{(0)}(j) \setminus \{i\}.
$$
The new edges at scale $\ell$ are $E^{(\ell)} = \{(i,j) : j \in \mathcal{N}^{(\ell)}(i) \setminus \mathcal{N}^{(\ell-1)}(i)\}$.

In our implementation we use $L=3$ coarsening levels ($L+1=4$ total scales). Each scale level corresponds to one additional graph hop, so scale $\ell$ captures interactions at a topological distance of $\ell+1$ edges from the source vertex.

---

### Per-Scale Message Passing

For each historical frame at delay $\tau$ ($\tau = 1, \dots, T$), we first encode the per-vertex state by concatenating the secondary displacement $u_{t-\tau}(i)$ and the reference position $x^{\mathrm{ref}}_{t-\tau}(i)$:
$$
h_{t-\tau}(i) = \mathrm{MLP}_{\mathrm{enc}}\big([u_{t-\tau}(i);\; x^{\mathrm{ref}}_{t-\tau}(i)]\big) \in \mathbb{R}^{D},
$$
where $D = 128$ is the hidden dimension.

On each edge set $E^{(s)}$, we perform one round of message passing with a dedicated aggregation module $\mathrm{MSEA}^{(s)}$:
$$
m^{(s)}_{t-\tau}(i) = \mathrm{MSEA}^{(s)}\big(h_{t-\tau},\; E^{(s)}\big)_i.
$$

Each $\mathrm{MSEA}^{(s)}$ module consists of an edge MLP that computes messages from concatenated source-destination features, a scatter-add aggregation onto destination nodes, and a node update MLP that combines the original features with the aggregated messages:
$$
\mathbf{e}_{j \to i} = \mathrm{MLP}_{\mathrm{edge}}([h_j;\; h_i]), \quad \bar{m}_i = \sum_{j:(j,i)\in E^{(s)}} \mathbf{e}_{j \to i}, \quad m^{(s)}_i = \mathrm{MLP}_{\mathrm{node}}([h_i;\; \bar{m}_i]).
$$

This yields a spatiotemporal message tensor $\{m^{(s)}_{t-\tau}(i)\}$ for $\tau \in \{1,\dots,T\}$ and $s \in \{0,\dots,L\}$.

---

### Causal Cone Masking

We model the causal constraint that information from distant spatial scales should only be accessible after sufficient temporal delay. For each vertex $i$, we learn a propagation speed from its material properties:
$$
v_i = \mathrm{Softplus}\!\big(\mathrm{MLP}_{\mathrm{vel}}(\theta_i)\big),
$$
where $\theta_i = \mathrm{MLP}_{\mathrm{prop}}([\kappa_i;\; m_i;\; c_i])$ encodes the per-vertex stiffness $\kappa_i$, mass $m_i$, and constraint flag $c_i$ into a property embedding of dimension $d_\theta = 3$.

The causal radius at delay $\tau$ is $r_i(\tau) = v_i \cdot \tau$. Because both $\tau$ and the scale thresholds $R_s$ are measured in discrete units (frames and graph-hop levels, respectively), the learned quantity $v_i$ has units of *graph-scale levels per frame*: it represents the rate at which the admissible spatial scale expands with each additional frame of temporal delay, not a physical velocity in metres per second.

We map the causal radius to a continuous maximum admissible scale index via a sum-of-sigmoids formulation that is everywhere differentiable:
$$
s_{\max}(i,\tau) = \left[\sum_{s=0}^{L} \sigma\!\big(\beta(r_i(\tau) - R_s)\big)\right] - 1,
$$
where $R_s = s+1$ are the characteristic thresholds per scale, $\sigma$ is the sigmoid function, and $\beta = 5$ controls the transition sharpness.

The soft causal mask is then:
$$
M(i,\tau,s) = \sigma\!\big(\beta\,(s_{\max}(i,\tau) - s)\big),
$$
which smoothly approaches 1 when $s \ll s_{\max}$ and 0 when $s \gg s_{\max}$, preserving gradient flow unlike a hard binary mask.

---

### Spatiotemporal Attention Aggregation

The masked multi-scale messages are combined via a lightweight attention mechanism. The attention logits are computed by:
$$
\alpha_{i,\tau,s} = \mathrm{MLP}_{\mathrm{attn}}\!\big([h_t(i);\; m^{(s)}_{t-\tau}(i);\; \tau;\; \theta_i]\big).
$$

To enforce the causal structure while maintaining differentiability, we add the log-mask to the logits before applying softmax over all $(\tau, s)$ entries:
$$
w_{i,\tau,s} = \mathrm{Softmax}_{(\tau,s)}\!\big(\alpha_{i,\tau,s} + \log(M(i,\tau,s) + \epsilon)\big),
$$
where $\epsilon = 10^{-8}$ prevents numerical issues.

The spatiotemporal context vector is the weighted sum:
$$
c_t(i) = \sum_{\tau=1}^{T} \sum_{s=0}^{L} w_{i,\tau,s} \cdot m^{(s)}_{t-\tau}(i).
$$

---

### Dynamics Update

The current frame state is encoded independently:
$$
h_t^{\mathrm{cur}}(i) = \mathrm{MLP}_{\mathrm{cur}}\!\big([u_t(i);\; x^{\mathrm{ref}}_t(i)]\big).
$$

The displacement increment for unconstrained vertices ($c_i = 0$) is predicted by:
$$
\Delta u_i(t) = F_\theta\!\big([h_t^{\mathrm{cur}}(i);\; c_t(i)]\big),
$$
where $F_\theta$ is a 5-layer MLP with Tanh activations (dimensions $256 \to 256 \to 256 \to 128 \to 3$). The secondary displacement and deformed position are then updated as:
$$
u_i(t+1) = u_i(t) + \Delta u_i(t), \qquad y_i(t+1) = x^{\mathrm{ref}}_{t+1}(i) + u_i(t+1).
$$

For constrained vertices ($c_i = 1$), the secondary displacement is not predicted; instead, we set $u_i(t+1) = 0$ so that $y_i(t+1) = x^{\mathrm{ref}}_{t+1}(i)$.

Compared to the original Deep Emulator which uses:
$$
\Delta u_i^{\mathrm{DE}}(t) = F_{\mathrm{DE}}\!\big(u(t), \dot{u}(t), \ddot{u}(t),\; x^{\mathrm{ref}}(t), \dot{x}^{\mathrm{ref}}(t), \ddot{x}^{\mathrm{ref}}(t),\; m, \kappa\big),
$$
our formulation augments the dynamics function with the causally-gated spatiotemporal context $c_t(i)$, enabling the model to capture long-range influence propagation that the original single-frame local architecture cannot represent.

---

## Hybrid Training

We adopt a two-stage hybrid training strategy: Stage 1 pre-trains the model with supervised displacement data, and Stage 2 fine-tunes with physics-based self-supervised energy minimisation. Note that the two stages employ losses of fundamentally different nature and scale: Stage 1 uses a normalised per-vertex MSE in displacement space, while Stage 2 uses a sum of physical energies (in Joules). The two losses are therefore not directly numerically comparable; the transition from Stage 1 to Stage 2 should be understood as a change of objective, not a continuation of the same loss curve.

---

### Stage 1: Supervised Pre-training

**Data.** We generate simulation sequences on sphere primitives under varying external excitations, boundary conditions, and material parameters using a FEM solver. Each frame provides a ground-truth displacement increment $\Delta u_i^{\mathrm{gt}}(t) = u_i(t+1) - u_i(t)$.

**Objective.** The model is trained to minimise the mean squared error between predicted and ground-truth displacement increments, evaluated only over unconstrained vertices ($c_i = 0$):
$$
\mathcal{L}_{\mathrm{sup}} = \frac{1}{N_{\mathrm{free}}} \sum_{\{i:\,c_i=0\}} \left\| \Delta u_i^{\mathrm{pred}}(t) - \Delta u_i^{\mathrm{gt}}(t) \right\|_2^2.
$$

We train for 60 epochs with the Adam optimiser (initial learning rate $10^{-4}$, exponential decay factor 0.96 per epoch, batch size 48).

---

### Stage 2: Self-Supervised Physics Fine-tuning

#### Position Recovery

Given the predicted displacement increment $\Delta \hat{u}_i(t)$, the predicted deformed position at the next frame is recovered as:
$$
\hat{u}_i(t+1) = u_i(t) + \Delta\hat{u}_i(t), \qquad \hat{y}_i(t+1) = x^{\mathrm{ref}}_{t+1}(i) + \hat{u}_i(t+1),
$$
where $x^{\mathrm{ref}}_{t+1}(i)$ is the known reference position at frame $t+1$. For constrained vertices, $\hat{u}_i(t+1) = 0$ and $\hat{y}_i(t+1) = x^{\mathrm{ref}}_{t+1}(i)$.

#### Physics Energy Loss

The self-supervised loss combines three physical energy terms, evaluated over all vertices (with constrained vertices clamped to their reference positions as Dirichlet boundary conditions):
$$
\mathcal{L}_{\mathrm{phys}} = \mathcal{L}_{\mathrm{inertia}} + \mathcal{L}_{\mathrm{gravity}} + \mathcal{L}_{\mathrm{strain}}.
$$

**Inertial potential energy:**
$$
\mathcal{L}_{\mathrm{inertia}} = \frac{1}{2} (\hat{y}_{t+1} + y_{t-1} - 2\,y_t)^\top \mathbf{M}\, (\hat{y}_{t+1} + y_{t-1} - 2\,y_t),
$$
where $\mathbf{M}$ is the diagonal mass matrix, and $y_t$, $y_{t-1}$ are the deformed positions at frames $t$ and $t-1$.

**Gravitational potential energy:**
$$
\mathcal{L}_{\mathrm{gravity}} = -\mathbf{M}\, g\, y_t^{(y)},
$$
where $g = 9.81\;\mathrm{m/s^2}$ and $y_t^{(y)}$ denotes the vertical component.

**Neo-Hookean strain energy:**
$$
\mathcal{L}_{\mathrm{strain}} = \sum_{e=1}^{n_{\mathrm{el}}} \Psi(\mathbf{F}^e)\, v^e, \quad \Psi = \tfrac{1}{2}\lambda(\log|\det\mathbf{F}|)^2 - \mu\log|\det\mathbf{F}| + \tfrac{1}{2}\mu(\mathrm{tr}(\mathbf{F}^\top\mathbf{F}) - 3),
$$
where $\mathbf{F}^e = \mathbf{D}_s^e (\mathbf{D}_m^e)^{-1}$ is the deformation gradient of element $e$, $v^e$ is the rest volume, and $\lambda, \mu$ are the Lame parameters.

#### Progressive $K$-Step Rollout

Recall that $K$ denotes the autoregressive rollout horizon, which is distinct from the input temporal window $T$: the model always observes $T$ past frames as input, while $K$ controls how many future frames are predicted autoregressively during training. Increasing $K$ does not change the temporal discretisation or step size.

To improve long-horizon stability, we employ a curriculum strategy that progressively increases $K$ during training. Starting from $K=1$, the rollout horizon increases by one step every 10 epochs up to a maximum of $K=8$:
$$
K(\mathrm{epoch}) = \min\!\Big(1 + \Big\lfloor \frac{\mathrm{epoch}}{10} \Big\rfloor,\; 8\Big).
$$

The physics loss is accumulated over the full rollout:
$$
\mathcal{L}_{\mathrm{phys}}^{(K)} = \sum_{k=1}^{K} \mathcal{L}_{\mathrm{phys}}(\hat{y}_{t+k},\; y_{t+k-1},\; y_{t+k-2}).
$$

The overall Stage 2 objective is:
$$
\min_\theta\; \mathbb{E}\!\left[\mathcal{L}_{\mathrm{phys}}^{(K)}\right].
$$

Stage 2 runs for 40 epochs with the Adam optimiser (initial learning rate $10^{-5}$, decay factor 0.98, batch size 1 due to the rollout memory cost).

---


## Literature Review

### 长程传播 / 多尺度结构

**Physics-informed graph neural network emulation of soft-tissue mechanics**  
coarse graph + clustering nodes + multi-layer graph 用于解决长距离传播/耦合；通过 physics-informed 约束提升稳定性与泛化。  

---

### 残差场与分解
**Towards High-Quality 3D Motion Transfer with Realistic Apparel Animation**  
依赖历史 $K$ 帧；时序非线性顶点残差场；区分身体与衣物（分支/分类处理）。  

---

### 结构先验 + 细节补偿
**A Neural-Network-Based Approach for Loose-Fitting Clothing**  
rope chain 取代 mass-spring（更稳定/可控的结构先验）；skinning + QNN 做细节补偿。  


---

### 物理约束

**Hierarchical Neural Skinning Deformation with Self-supervised Training for Character Animation**  
分物理层 + 物理约束：Inertia、Muscle Strain Energy、Gravity、Damping、Soft-tissue Strain Energy 等；强调层级结构与约束设计。  

---

### 其他技术路径
**Real-Time Secondary Animation with Spring Decomposed Skinning**  
在骨骼上添加弹簧实现 skin 的次级运动；偏工程/实时管线的另一条路线。  

**Fast Complementary Dynamics via Skinning Eigenmodes**  
用 Linear Blend Skinning 子空间（eigenmodes）替代传统位移空间训练，提高效率与稳定性。  
