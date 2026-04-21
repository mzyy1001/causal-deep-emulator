"""
FEM simulation using SfePy to generate ground truth displacements
for characters at different stiffness values.

Runs implicit dynamic elasticity on tetrahedral meshes with:
- Neo-Hookean (or linear) material
- Gravity loading
- Fixed vertex boundary conditions
- Prescribed reference motion as boundary displacement

Outputs displacement sequences compatible with data_loader.py format.

Usage:
    python run_fem_sfepy.py \
        --veg ../data/character_dataset/mousey/mousey.veg \
        --motion ../data/character_dataset/mousey/dancing_1 \
        --stiffness 50000,100000,500000,1000000,5000000 \
        --output ../data/fem_test/mousey/dancing_1 \
        --frames 50
"""
import argparse
import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time


def parse_veg_file(veg_path):
    """Parse Vega FEM .veg file."""
    vertices = []
    elements = []
    section = None
    num_verts_read = False
    num_elems_read = False

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
            elif line.startswith('*'):
                section = None
                continue

            if section == 'vertices':
                parts = line.split()
                if not num_verts_read:
                    num_verts_read = True
                    continue
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif section == 'elements':
                parts = line.split()
                if not num_elems_read:
                    num_elems_read = True
                    continue
                if len(parts) >= 5:
                    elements.append([int(parts[1])-1, int(parts[2])-1,
                                     int(parts[3])-1, int(parts[4])-1])

    return np.array(vertices, dtype=np.float64), np.array(elements, dtype=np.int32)


def compute_tet_stiffness_matrix(vertices, elements, young_modulus, poisson_ratio):
    """Compute global stiffness matrix for linear elasticity on tet mesh."""
    V = len(vertices)
    ndof = V * 3

    # Lame parameters
    lam = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    mu = young_modulus / (2 * (1 + poisson_ratio))

    # Material matrix (6x6 for 3D)
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu

    rows, cols, vals = [], [], []

    for elem in elements:
        v0, v1, v2, v3 = vertices[elem[0]], vertices[elem[1]], vertices[elem[2]], vertices[elem[3]]

        # Shape function gradients for linear tet
        J = np.array([v1 - v0, v2 - v0, v3 - v0]).T  # 3x3
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-15:
            continue
        vol = abs(det_J) / 6.0
        J_inv = np.linalg.inv(J)

        # Gradients of shape functions (4 nodes, 3 components)
        dN = np.zeros((4, 3))
        dN[0] = -J_inv.sum(axis=1)
        dN[1] = J_inv[:, 0]
        dN[2] = J_inv[:, 1]
        dN[3] = J_inv[:, 2]

        # B matrix (6x12) for this element
        B = np.zeros((6, 12))
        for i in range(4):
            B[0, 3*i] = dN[i, 0]
            B[1, 3*i+1] = dN[i, 1]
            B[2, 3*i+2] = dN[i, 2]
            B[3, 3*i] = dN[i, 1]
            B[3, 3*i+1] = dN[i, 0]
            B[4, 3*i+1] = dN[i, 2]
            B[4, 3*i+2] = dN[i, 1]
            B[5, 3*i] = dN[i, 2]
            B[5, 3*i+2] = dN[i, 0]

        Ke = vol * B.T @ D @ B  # 12x12

        # Assemble into global
        dofs = []
        for n in elem:
            dofs.extend([3*n, 3*n+1, 3*n+2])
        for i in range(12):
            for j in range(12):
                if abs(Ke[i, j]) > 1e-20:
                    rows.append(dofs[i])
                    cols.append(dofs[j])
                    vals.append(Ke[i, j])

    K = csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
    return K


def compute_mass_matrix_lumped(vertices, elements, density=1000.0):
    """Compute lumped mass matrix."""
    V = len(vertices)
    mass = np.zeros(V)

    for elem in elements:
        v0, v1, v2, v3 = vertices[elem[0]], vertices[elem[1]], vertices[elem[2]], vertices[elem[3]]
        J = np.array([v1 - v0, v2 - v0, v3 - v0]).T
        vol = abs(np.linalg.det(J)) / 6.0
        m = density * vol / 4.0
        for n in elem:
            mass[n] += m

    # Expand to 3*V (x, y, z per vertex)
    M = np.repeat(mass, 3)
    return M


def run_dynamic_simulation(vertices, elements, constraint_mask, ref_positions,
                           young_modulus, poisson_ratio=0.45, density=1000.0,
                           gravity=-9.81, dt=1.0/30.0, damping=0.1, num_frames=50):
    """Run implicit Newmark time integration.

    Args:
        vertices: (V, 3) rest positions
        elements: (E, 4) tet element indices
        constraint_mask: (V,) bool, True = fixed
        ref_positions: list of (V, 3) reference positions per frame
        young_modulus: Young's modulus (stiffness)
        num_frames: number of frames to simulate

    Returns:
        displacements: list of (V, 3) displacement arrays per frame
    """
    V = len(vertices)
    ndof = V * 3

    print('    Computing stiffness matrix (E=%.0f)...' % young_modulus)
    K = compute_tet_stiffness_matrix(vertices, elements, young_modulus, poisson_ratio)

    print('    Computing mass matrix...')
    M_diag = compute_mass_matrix_lumped(vertices, elements, density)

    # Damping: C = damping * M
    C_diag = damping * M_diag

    # Gravity force
    f_gravity = np.zeros(ndof)
    for i in range(V):
        f_gravity[3*i + 1] = M_diag[3*i] * gravity  # y-direction

    # Newmark parameters (average acceleration)
    beta = 0.25
    gamma = 0.5

    # Effective stiffness: K_eff = K + (1/(beta*dt^2))*M + (gamma/(beta*dt))*C
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)

    M_sparse = csr_matrix((M_diag, (range(ndof), range(ndof))), shape=(ndof, ndof))
    C_sparse = csr_matrix((C_diag, (range(ndof), range(ndof))), shape=(ndof, ndof))
    K_eff = K + a0 * M_sparse + a1 * C_sparse

    # Apply boundary conditions (zero rows/cols for constrained DOFs)
    constrained_dofs = []
    for i in range(V):
        if constraint_mask[i]:
            constrained_dofs.extend([3*i, 3*i+1, 3*i+2])
    constrained_dofs = np.array(constrained_dofs)

    # Modify K_eff for BCs
    K_eff_lil = K_eff.tolil()
    for dof in constrained_dofs:
        K_eff_lil[dof, :] = 0
        K_eff_lil[:, dof] = 0
        K_eff_lil[dof, dof] = 1.0
    K_eff = K_eff_lil.tocsr()

    # Initial conditions
    u = np.zeros(ndof)  # displacement
    v = np.zeros(ndof)  # velocity
    a = np.zeros(ndof)  # acceleration

    displacements = [np.zeros((V, 3))]  # frame 0: zero displacement

    print('    Simulating %d frames...' % num_frames)
    t_start = time.time()

    for frame in range(1, num_frames):
        # External force: gravity + reference motion forcing
        f_ext = f_gravity.copy()

        # If we have reference positions, add forcing from skeletal motion
        if frame < len(ref_positions):
            # Force from reference motion change
            ref_disp = (ref_positions[frame] - vertices).flatten()
            # Add a spring-like force pulling constrained vertices to reference
            for i in range(V):
                if constraint_mask[i]:
                    for d in range(3):
                        dof = 3*i + d
                        u[dof] = ref_disp[dof]

        # Effective force
        f_eff = f_ext + M_sparse @ (a0 * u + a0 * dt * v + (0.5/beta - 1) * a)
        f_eff += C_sparse @ (a1 * u + (a1*dt - 1) * v + dt * (gamma/(2*beta) - 1) * a)

        # Apply BCs to RHS
        for dof in constrained_dofs:
            if frame < len(ref_positions):
                ref_disp = (ref_positions[frame] - vertices).flatten()
                f_eff[dof] = ref_disp[dof]
            else:
                f_eff[dof] = u[dof]

        # Solve
        u_new = spsolve(K_eff, f_eff)

        # Update acceleration and velocity
        a_new = a0 * (u_new - u) - a0 * dt * v - (0.5/beta - 1) * a
        v_new = v + dt * ((1 - gamma) * a + gamma * a_new)

        u = u_new
        v = v_new
        a = a_new

        disp = u.reshape(V, 3)
        displacements.append(disp.copy())

        if frame % 10 == 0:
            max_disp = np.abs(disp).max()
            print('      Frame %d/%d: max_disp=%.4f' % (frame, num_frames, max_disp))

    elapsed = time.time() - t_start
    print('    Simulation done in %.1fs' % elapsed)
    return displacements


def save_sequence(output_dir, vertices, elements, displacements, ref_positions,
                  constraint_mask, stiffness_value, mass_values):
    """Save in data_loader.py compatible format."""
    os.makedirs(output_dir, exist_ok=True)
    V = len(vertices)

    # c: constraint flags
    constraint_mask.astype(np.int32).tofile(os.path.join(output_dir, 'c'))

    # adj: build from elements
    from generate_fem_data import build_adjacency_from_elements
    adj = build_adjacency_from_elements(V, elements)
    adj.flatten().astype(np.int32).tofile(os.path.join(output_dir, 'adj'))

    # k: stiffness
    k = np.full(V, stiffness_value, dtype=np.float64)
    k.tofile(os.path.join(output_dir, 'k'))

    # m: mass
    mass_values.astype(np.float64).tofile(os.path.join(output_dir, 'm'))

    # offset
    offset = np.zeros((V, 3), dtype=np.float64)
    offset.tofile(os.path.join(output_dir, 'offset'))

    # u_N and x_N
    num_frames = len(displacements)
    for i in range(num_frames):
        displacements[i].astype(np.float64).tofile(os.path.join(output_dir, 'u_%d' % i))
        if i < len(ref_positions):
            ref_positions[i].astype(np.float64).tofile(os.path.join(output_dir, 'x_%d' % i))
        else:
            ref_positions[-1].astype(np.float64).tofile(os.path.join(output_dir, 'x_%d' % i))

    print('    Saved %d frames to %s' % (num_frames, output_dir))


def main():
    parser = argparse.ArgumentParser(description='Run FEM simulation for character stiffness test')
    parser.add_argument('--veg', required=True, help='Path to .veg mesh file')
    parser.add_argument('--motion', required=True, help='Path to motion data (with x_* reference files)')
    parser.add_argument('--stiffness', default='50000,100000,500000,1000000,5000000',
                        help='Comma-separated Young modulus values')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to simulate')
    parser.add_argument('--density', type=float, default=1000.0)
    parser.add_argument('--poisson', type=float, default=0.45)
    parser.add_argument('--damping', type=float, default=0.1)
    args = parser.parse_args()

    print('Parsing mesh: %s' % args.veg)
    vertices, elements = parse_veg_file(args.veg)
    V = len(vertices)
    print('  Vertices: %d, Elements: %d' % (V, len(elements)))

    # Load constraint from motion data
    from data_loader import loadData_Int, loadData_Float
    constraint = loadData_Int(os.path.join(args.motion, 'c'))
    constraint_mask = (constraint == 1)
    print('  Constrained: %d / %d' % (constraint_mask.sum(), V))

    # Load reference positions
    frame_files = sorted([f for f in os.listdir(args.motion) if f.startswith('x_')])
    num_frames = min(args.frames, len(frame_files))
    print('  Loading %d reference frames...' % num_frames)

    ref_positions = []
    for i in range(num_frames):
        x = loadData_Float(os.path.join(args.motion, 'x_%d' % i)).reshape(-1, 3)
        ref_positions.append(x)

    # Compute mass
    mass_values = np.zeros(V, dtype=np.float64)
    for elem in elements:
        v0, v1, v2, v3 = vertices[elem[0]], vertices[elem[1]], vertices[elem[2]], vertices[elem[3]]
        J = np.array([v1 - v0, v2 - v0, v3 - v0]).T
        vol = abs(np.linalg.det(J)) / 6.0
        m = args.density * vol / 4.0
        for n in elem:
            mass_values[n] += m

    stiffness_values = [float(s) for s in args.stiffness.split(',')]

    for seq_idx, E_val in enumerate(stiffness_values, start=1):
        print('\n=== Sequence %d: E = %.0f ===' % (seq_idx, E_val))
        seq_dir = os.path.join(args.output, str(seq_idx))

        displacements = run_dynamic_simulation(
            vertices, elements, constraint_mask, ref_positions,
            young_modulus=E_val,
            poisson_ratio=args.poisson,
            density=args.density,
            damping=args.damping,
            num_frames=num_frames)

        save_sequence(seq_dir, vertices, elements, displacements,
                      ref_positions, constraint, E_val, mass_values)

    print('\nAll simulations complete. Output: %s' % args.output)


if __name__ == '__main__':
    main()
