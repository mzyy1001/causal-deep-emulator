"""
FEM data generation for the causal spatiotemporal deep emulator.

Two modes:
  1. Training data: Generate FEM simulations on simple geometries (sphere, cube,
     cylinder, etc.) with varied stiffness/mass. These augment the existing sphere
     training data to improve generalization.

  2. Test data: Run FEM on character meshes with NEW stiffness values (different
     from training) to produce ground truth for stiffness-robustness evaluation.

Prerequisites:
  - Vega FEM library installed (http://barbic.usc.edu/vega/)
  - Libraries and Makefile-headers copied to vega_FEM/
  - Or: any FEM solver that outputs per-vertex displacement sequences

Usage:
    # Generate test data for michelle with new stiffness values:
    python generate_fem_data.py --mode test \
        --veg ../data/character_dataset/michelle/michelle.veg \
        --constraint ../data/character_dataset/michelle/cross_jumps/c \
        --motion ../data/character_dataset/michelle/cross_jumps \
        --stiffness 10000,25000,75000,150000,500000,2000000,10000000 \
        --output ../data/character_test_stiffness/michelle/cross_jumps

    # Generate training data for a new geometry:
    python generate_fem_data.py --mode train \
        --veg ../data/geometries/cube.veg \
        --stiffness 50000,100000,250000,500000,1000000,2500000,5000000 \
        --output ../data/cube_dataset/train/motion_1

Data format (compatible with data_loader.py):
    output_dir/
      seq_num/              # one subdirectory per stiffness value
        c                   # (V,) int32 constraint flags
        adj                 # (V * max_neighbors,) int32 adjacency, 1-indexed
        k                   # (V,) float64 stiffness per vertex
        m                   # (V,) float64 mass per vertex
        u_0, u_1, ...       # (V*3,) float64 displacement per frame
        x_0, x_1, ...       # (V*3,) float64 reference position per frame
"""

import argparse
import os
import shutil
import numpy as np


def parse_veg_file(veg_path):
    """Parse a Vega FEM .veg file to extract vertices and elements.

    Returns:
        vertices: (V, 3) float64 array
        elements: (E, 4) int array (0-indexed vertex indices)
    """
    vertices = []
    elements = []
    section = None
    num_vertices = 0
    num_elements = 0

    with open(veg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('*VERTICES'):
                section = 'vertices'
                continue
            elif line.startswith('*ELEMENTS'):
                section = 'elements'
                continue

            if section == 'vertices':
                parts = line.split()
                if len(parts) >= 4 and num_vertices == 0:
                    # Header line: num_verts dim ...
                    try:
                        num_vertices = int(parts[0])
                    except ValueError:
                        pass
                    continue
                if len(parts) >= 4:
                    # vertex_id x y z
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif section == 'elements':
                parts = line.split()
                if len(parts) >= 2 and num_elements == 0:
                    try:
                        num_elements = int(parts[0])
                    except ValueError:
                        pass
                    continue
                if len(parts) >= 5:
                    # element_id v0 v1 v2 v3 (1-indexed in veg)
                    elements.append([int(parts[1])-1, int(parts[2])-1,
                                     int(parts[3])-1, int(parts[4])-1])

    return np.array(vertices, dtype=np.float64), np.array(elements, dtype=np.int32)


def build_adjacency_from_elements(num_vertices, elements, max_neighbors=20):
    """Build 1-indexed adjacency matrix from tet elements.

    Returns:
        adj: (V, max_neighbors) int32 array, 1-indexed, 0-padded
    """
    neighbors = [set() for _ in range(num_vertices)]
    for elem in elements:
        for i in range(4):
            for j in range(4):
                if i != j:
                    neighbors[elem[i]].add(elem[j])

    adj = np.zeros((num_vertices, max_neighbors), dtype=np.int32)
    for i in range(num_vertices):
        nb_list = sorted(neighbors[i])
        for j, nb in enumerate(nb_list[:max_neighbors]):
            adj[i, j] = nb + 1  # 1-indexed
    return adj


def compute_vertex_mass(vertices, elements, density=1000.0):
    """Compute per-vertex lumped mass from tet volumes.

    Returns:
        mass: (V,) float64
    """
    V = len(vertices)
    mass = np.zeros(V, dtype=np.float64)

    for elem in elements:
        v0, v1, v2, v3 = vertices[elem[0]], vertices[elem[1]], vertices[elem[2]], vertices[elem[3]]
        # Tet volume = |det([v1-v0, v2-v0, v3-v0])| / 6
        mat = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=0)
        vol = abs(np.linalg.det(mat)) / 6.0
        tet_mass = density * vol / 4.0  # distribute equally to 4 vertices
        for vi in elem:
            mass[vi] += tet_mass

    return mass


def copy_reference_motion(motion_path, output_seq_path, num_frames=None):
    """Copy reference positions (x_*) from an existing motion to the output directory.

    This provides the driving animation. Displacements (u_*) need to be generated
    by running FEM with the new stiffness.
    """
    frame_files = sorted([f for f in os.listdir(motion_path) if f.startswith("x_")])
    if num_frames is not None:
        frame_files = frame_files[:num_frames]

    for fname in frame_files:
        src = os.path.join(motion_path, fname)
        dst = os.path.join(output_seq_path, fname)
        shutil.copy2(src, dst)

    return len(frame_files)


def write_sequence_metadata(output_path, constraint, adj, stiffness_value,
                            mass, num_vertices):
    """Write c, adj, k, m files for a sequence."""
    # Constraint flags
    constraint.astype(np.int32).tofile(os.path.join(output_path, "c"))

    # Adjacency
    adj.flatten().astype(np.int32).tofile(os.path.join(output_path, "adj"))

    # Stiffness (uniform value for all vertices)
    k = np.full(num_vertices, stiffness_value, dtype=np.float64)
    k.tofile(os.path.join(output_path, "k"))

    # Mass
    mass.astype(np.float64).tofile(os.path.join(output_path, "m"))


def generate_zero_displacement_placeholder(output_path, num_frames, num_vertices):
    """Write zero-displacement files as placeholders for FEM output.

    These should be replaced with actual FEM simulation results.
    """
    zero_u = np.zeros(num_vertices * 3, dtype=np.float64)
    for i in range(num_frames):
        zero_u.tofile(os.path.join(output_path, f"u_{i}"))

    print(f"    WARNING: Wrote zero displacements as placeholders.")
    print(f"    Run FEM simulation and replace u_* files with actual results.")


def generate_test_data(args):
    """Generate test data for character meshes with new stiffness values."""
    print(f"Parsing mesh: {args.veg}")
    vertices, elements = parse_veg_file(args.veg)
    V = len(vertices)
    print(f"  Vertices: {V}, Elements: {len(elements)}")

    # Build adjacency
    adj = build_adjacency_from_elements(V, elements)
    print(f"  Adjacency: {adj.shape}")

    # Compute mass
    mass = compute_vertex_mass(vertices, elements)
    print(f"  Mass range: [{mass.min():.6f}, {mass.max():.6f}]")

    # Load constraint from existing motion data
    constraint_file = args.constraint
    if constraint_file and os.path.exists(constraint_file):
        constraint = np.fromfile(constraint_file, dtype=np.int32)
        print(f"  Loaded constraint from {constraint_file}: "
              f"{np.sum(constraint == 1)} constrained / {V} total")
    else:
        # Default: no constraints
        constraint = np.zeros(V, dtype=np.int32)
        print(f"  No constraint file; all vertices free")

    stiffness_values = [float(s) for s in args.stiffness.split(',')]
    print(f"  Stiffness values to generate: {stiffness_values}")

    for seq_idx, stiff_val in enumerate(stiffness_values, start=1):
        seq_path = os.path.join(args.output, str(seq_idx))
        os.makedirs(seq_path, exist_ok=True)
        print(f"\n  Sequence {seq_idx}: stiffness = {stiff_val:.0f}")

        # Write metadata
        write_sequence_metadata(seq_path, constraint, adj, stiff_val, mass, V)

        # Copy reference motion
        if args.motion and os.path.isdir(args.motion):
            num_frames = copy_reference_motion(args.motion, seq_path)
            print(f"    Copied {num_frames} reference frames from {args.motion}")

            # Write placeholder displacements
            generate_zero_displacement_placeholder(seq_path, num_frames, V)

            # Generate FEM runner script
            fem_script = os.path.join(seq_path, "run_fem.sh")
            with open(fem_script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Run FEM simulation for stiffness={stiff_val:.0f}\n")
                f.write(f"# Input: {args.veg}\n")
                f.write(f"# Stiffness: {stiff_val}\n")
                f.write(f"# Frames: {num_frames}\n")
                f.write(f"# Output: u_0 ... u_{num_frames-1} in {seq_path}\n")
                f.write(f"#\n")
                f.write(f"# Replace this script with your FEM solver invocation.\n")
                f.write(f"# The solver should output per-frame displacement files (u_N)\n")
                f.write(f"# in the same binary format: V*3 float64 values.\n")
                f.write(f"echo 'FEM simulation not yet configured. Edit this script.'\n")
            os.chmod(fem_script, 0o755)
            print(f"    Created FEM runner: {fem_script}")
        else:
            print(f"    No motion path provided; skipping frame copy")

    # Write offset file (vertex centering)
    offset = np.zeros((V, 3), dtype=np.float64)
    offset.tofile(os.path.join(args.output, "offset"))

    print(f"\nTest data structure created at {args.output}")
    print(f"Next steps:")
    print(f"  1. Run FEM simulation for each stiffness to generate u_* files")
    print(f"  2. Replace placeholder u_* files with FEM output")
    print(f"  3. Evaluate with: python evaluate_k.py --weight <model> --data {args.output}")


def generate_train_data(args):
    """Generate training data for simple geometries."""
    print(f"Parsing mesh: {args.veg}")
    vertices, elements = parse_veg_file(args.veg)
    V = len(vertices)
    print(f"  Vertices: {V}, Elements: {len(elements)}")

    adj = build_adjacency_from_elements(V, elements)
    mass = compute_vertex_mass(vertices, elements)

    # For training geometries, constraint is typically the bottom vertices
    if args.constraint and os.path.exists(args.constraint):
        constraint = np.fromfile(args.constraint, dtype=np.int32)
    else:
        # Default: constrain bottom 10% of vertices (by y-coordinate)
        y_coords = vertices[:, 1]
        threshold = np.percentile(y_coords, 10)
        constraint = np.zeros(V, dtype=np.int32)
        constraint[y_coords <= threshold] = 1
        print(f"  Auto-constrained {np.sum(constraint)} bottom vertices (y <= {threshold:.4f})")

    stiffness_values = [float(s) for s in args.stiffness.split(',')]
    print(f"  Stiffness values: {stiffness_values}")

    for seq_idx, stiff_val in enumerate(stiffness_values, start=1):
        seq_path = os.path.join(args.output, str(seq_idx))
        os.makedirs(seq_path, exist_ok=True)
        print(f"\n  Sequence {seq_idx}: stiffness = {stiff_val:.0f}")

        write_sequence_metadata(seq_path, constraint, adj, stiff_val, mass, V)

        # Write reference positions (rest pose repeated = static reference)
        # For training, the FEM solver should provide both x_* and u_*
        num_frames = args.num_frames or 200
        rest_pos = vertices.flatten()
        for i in range(num_frames):
            rest_pos.astype(np.float64).tofile(os.path.join(seq_path, f"x_{i}"))

        generate_zero_displacement_placeholder(seq_path, num_frames, V)

        fem_script = os.path.join(seq_path, "run_fem.sh")
        with open(fem_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Run FEM simulation for training geometry\n")
            f.write(f"# Mesh: {args.veg}\n")
            f.write(f"# Stiffness: {stiff_val}\n")
            f.write(f"# Frames: {num_frames}\n")
            f.write(f"# Apply external forces (gravity, impulse, etc.) and record\n")
            f.write(f"# both reference positions (x_*) and displacements (u_*).\n")
            f.write(f"echo 'FEM simulation not yet configured. Edit this script.'\n")
        os.chmod(fem_script, 0o755)

    offset = np.zeros((V, 3), dtype=np.float64)
    offset.tofile(os.path.join(args.output, "offset"))

    print(f"\nTraining data structure created at {args.output}")
    print(f"Next steps:")
    print(f"  1. Run FEM simulation for each stiffness")
    print(f"  2. Replace placeholder x_* and u_* files with FEM output")
    print(f"  3. Train with: python train.py --data_paths {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate FEM data for training/testing the deep emulator')
    parser.add_argument('--mode', required=True, choices=['train', 'test'],
                        help='train: simple geometry data; test: character with new stiffness')
    parser.add_argument('--veg', required=True, help='Path to .veg volumetric mesh file')
    parser.add_argument('--constraint', default=None,
                        help='Path to constraint file (binary int32). '
                             'If not provided, auto-generates for training.')
    parser.add_argument('--motion', default=None,
                        help='Path to existing motion data (for test mode: copies x_* frames)')
    parser.add_argument('--stiffness', required=True,
                        help='Comma-separated stiffness values to generate')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Number of frames to generate (train mode only)')
    args = parser.parse_args()

    if args.mode == 'test':
        generate_test_data(args)
    else:
        generate_train_data(args)


if __name__ == '__main__':
    main()
