# Causal Spatiotemporal Deep Emulator for Character Secondary Motion

This repository implements a causal spatiotemporal deep emulator for predicting secondary motion of 3D characters. The method extends the [Deep Emulator (CVPR 2021)](https://github.com/PeterZhouSZ/deep_emulator) with:

- **Multi-Scale Edge Aggregation (MSEA):** constructs a hierarchy of edge sets on the mesh to expand the spatial receptive field without pooling.
- **Causal Cone Masking:** a learnable, differentiable mask that gates spatial scales by temporal delay, encoding finite-speed mechanical propagation.
- **Hybrid Two-Stage Training:** Stage 1 uses supervised displacement data; Stage 2 fine-tunes with physics-based energy minimisation (inertia, gravity, Neo-Hookean strain) and progressive K-step rollout.

## Requirements

- Python 3.12
- PyTorch
- NumPy, trimesh, OpenCV, pyrender
- (Optional) TensorBoard for training visualisation
- (Optional) FFmpeg for video generation
- (For surface rendering) `libosmesa6`, `libglu1-mesa`, `freeglut3`

## Repository Structure

```
code/
  config.py          # Global hyperparameters (temporal window, scales, etc.)
  model.py           # CausalSpatiotemporalModel, MSEA, CausalCone, PhysicsLoss
  data_loader.py     # Dataset and data loading with extended temporal history
  train.py           # Two-stage hybrid training (supervised + physics)
  test.py            # Rollout prediction with tet/surface rendering
  render.py          # Mesh rendering utilities
  animationTet2Surface.py  # Volumetric-to-surface displacement interpolation
  vega_FEM/          # Vega FEM utilities for surface interpolation
data/
  sphere_dataset/    # Training/test data (sphere primitives)
  character_dataset/ # Character mesh data for evaluation
paper/
  method.tex         # Method section (LaTeX)
  literature.tex     # Related work section (LaTeX)
  developed.md       # Development notes
```

## Training

```bash
cd code
python train.py
```

This runs Stage 1 (60 epochs supervised) followed by Stage 2 (40 epochs physics-based self-supervised with progressive rollout). Weights are saved to `code/weight/`. Training logs can be viewed with:

```bash
tensorboard --logdir=code/runs/
```

## Testing

```bash
cd code
python test.py
```

Edit `test.py` to set the weight path, character, motion, and rendering flag (`"tet"`, `"surface"`, or `"tet_surface"`). Output frames are saved to `code/weight/eval/`. To generate a video:

```bash
ffmpeg -framerate 30 -i ./weight/eval/michelle/cross_jumps/%d.png -c:v libx264 -pix_fmt yuv420p animation.mp4
```

## Acknowledgements

This work builds upon:
- [Deep Emulator](https://github.com/PeterZhouSZ/deep_emulator) (Zheng et al., CVPR 2021)
- [MSEA-MGN](https://doi.org/10.1002/cav.2245) (Wang & Liu, CAVW 2024)
