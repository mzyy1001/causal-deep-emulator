/*
  VegaFEM dynamic simulation driver with prescribed motion.

  Loads a .veg mesh and reference motion (x_* files), applies reference
  positions to constrained vertices as Dirichlet BCs, simulates free
  vertices with specified stiffness, outputs displacement files.

  This generates ground truth at different stiffness values using the
  SAME driving motion — for stiffness robustness evaluation.

  Usage:
    ./run_sim mesh.veg motion_dir output_dir stiffness num_frames [damping]

  Where motion_dir contains: c, x_0, x_1, ..., x_{N-1}
  Output: output_dir/u_0, ..., u_{N-1}, x_0, ..., c, k, m, adj
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "volumetricMesh.h"
#include "volumetricMeshENuMaterial.h"
#include "volumetricMeshLoader.h"
#include "tetMesh.h"
#include "corotationalLinearFEM.h"
#include "corotationalLinearFEMForceModel.h"
#include "generateMassMatrix.h"
#include "implicitNewmarkSparse.h"
#include "sparseMatrix.h"

// Read binary float64 array
std::vector<double> readBinaryF64(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Cannot read %s\n", filename); return {}; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = sz / sizeof(double);
    std::vector<double> data(n);
    size_t r = fread(data.data(), sizeof(double), n, f);
    (void)r;
    fclose(f);
    return data;
}

// Read binary int32 array
std::vector<int> readBinaryI32(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Cannot read %s\n", filename); return {}; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = sz / sizeof(int);
    std::vector<int> data(n);
    size_t r = fread(data.data(), sizeof(int), n, f);
    (void)r;
    fclose(f);
    return data;
}

void writeBinaryF64(const char* filename, const double* data, int n) {
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("Cannot write %s\n", filename); return; }
    fwrite(data, sizeof(double), n, f);
    fclose(f);
}

void copyFile(const char* src, const char* dst) {
    FILE* fin = fopen(src, "rb");
    FILE* fout = fopen(dst, "wb");
    if (!fin || !fout) { printf("Cannot copy %s -> %s\n", src, dst); return; }
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fin)) > 0) fwrite(buf, 1, n, fout);
    fclose(fin); fclose(fout);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s mesh.veg motion_dir output_dir stiffness num_frames [damping]\n", argv[0]);
        printf("\n  motion_dir should contain: c, x_0, x_1, ...\n");
        printf("  Output: displacement + reference files compatible with data_loader.py\n");
        return 1;
    }

    const char* meshFile = argv[1];
    const char* motionDir = argv[2];
    const char* outputDir = argv[3];
    double stiffness = atof(argv[4]);
    int numFrames = atoi(argv[5]);
    double dampingMass = (argc > 6) ? atof(argv[6]) : 0.01;
    double dt = 1.0 / 30.0;

    // Load mesh
    printf("Loading mesh: %s\n", meshFile);
    VolumetricMesh* volumetricMesh = VolumetricMeshLoader::load(meshFile);
    if (!volumetricMesh) { printf("Error loading mesh\n"); return 1; }
    TetMesh* tetMesh = dynamic_cast<TetMesh*>(volumetricMesh);
    if (!tetMesh) { printf("Not a tet mesh\n"); return 1; }

    int n = tetMesh->getNumVertices();
    int ndof = 3 * n;
    printf("Vertices: %d, Elements: %d\n", n, tetMesh->getNumElements());

    // Override stiffness
    for (int i = 0; i < tetMesh->getNumMaterials(); i++) {
        VolumetricMesh::Material* mat = tetMesh->getMaterial(i);
        VolumetricMesh::ENuMaterial* enuMat = dynamic_cast<VolumetricMesh::ENuMaterial*>(mat);
        if (enuMat) {
            printf("Material %d: E=%.0f -> %.0f\n", i, enuMat->getE(), stiffness);
            enuMat->setE(stiffness);
        }
    }

    // Load constraints (c file has n+1 ints: padding + n flags)
    char path[2048];
    snprintf(path, sizeof(path), "%s/c", motionDir);
    std::vector<int> cFlags = readBinaryI32(path);
    int offset = ((int)cFlags.size() == n + 1) ? 1 : 0;

    std::vector<int> constrainedDOFs;
    std::vector<bool> isConstrained(n, false);
    for (int i = 0; i < n; i++) {
        if (cFlags[i + offset] == 1) {
            isConstrained[i] = true;
            constrainedDOFs.push_back(3*i);
            constrainedDOFs.push_back(3*i+1);
            constrainedDOFs.push_back(3*i+2);
        }
    }
    printf("Constrained: %d / %d vertices\n", (int)constrainedDOFs.size()/3, n);

    // Load reference positions for all frames
    printf("Loading %d reference frames from %s...\n", numFrames, motionDir);
    std::vector<std::vector<double>> refFrames(numFrames);
    for (int f = 0; f < numFrames; f++) {
        snprintf(path, sizeof(path), "%s/x_%d", motionDir, f);
        refFrames[f] = readBinaryF64(path);
        if (refFrames[f].empty()) {
            printf("Warning: cannot read frame %d, using frame 0\n", f);
            refFrames[f] = refFrames[0];
        }
    }
    // Reference has n*3 values (no padding row in .veg-generated data)
    // But motion_dir x_* may have (n+1)*3 from data_loader format
    int refSize = refFrames[0].size();
    int refOffset = (refSize == (n+1)*3) ? 3 : 0; // skip padding if present
    printf("Reference frame size: %d (offset=%d)\n", refSize, refOffset);

    // Get rest positions
    std::vector<double> restPos(n * 3);
    for (int i = 0; i < n; i++) {
        Vec3d v = tetMesh->getVertex(i);
        restPos[3*i] = v[0]; restPos[3*i+1] = v[1]; restPos[3*i+2] = v[2];
    }

    // Build FEM
    CorotationalLinearFEM* fem = new CorotationalLinearFEM(tetMesh);
    CorotationalLinearFEMForceModel* forceModel = new CorotationalLinearFEMForceModel(fem);

    // Mass matrix
    SparseMatrix* massMatrix;
    GenerateMassMatrix::computeMassMatrix(tetMesh, &massMatrix, true);

    // Gravity force
    double gravity = -9.81;
    std::vector<double> gravityForce(ndof, 0.0);
    for (int i = 0; i < n; i++) {
        double mass_i = massMatrix->GetEntry(3*i, 3*i);
        gravityForce[3*i+1] = mass_i * gravity;
    }

    // Integrator
    // Use Newmark (beta=0.25, gamma=0.5) instead of backward Euler
    // This preserves momentum and produces natural oscillations
    ImplicitNewmarkSparse* integrator = new ImplicitNewmarkSparse(
        ndof, dt, massMatrix, forceModel,
        (int)constrainedDOFs.size(), constrainedDOFs.data(),
        dampingMass, 0.0,  // dampingMass, dampingStiffness
        1,                 // maxIterations (1 = single Newton step)
        1e-6,              // epsilon
        0.25, 0.5,         // Newmark beta, gamma (trapezoidal rule)
        0                  // numSolverThreads
    );

    // Create output directory
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", outputDir);
    int ret = system(cmd);
    (void)ret;

    // Displacement buffer
    std::vector<double> u(ndof, 0.0);
    std::vector<double> uOutput(n * 3, 0.0);

    printf("Simulating %d frames (E=%.0f, damping=%.3f)...\n", numFrames, stiffness, dampingMass);

    for (int frame = 0; frame < numFrames; frame++) {
        if (frame > 0) {
            // KEY INSIGHT: x_* contains LBS/skinning displacement for ALL vertices.
            // The FEM should simulate the SECONDARY perturbation on top of x_ref.
            //
            // Strategy:
            // 1. Set ALL vertices to x_ref (the skeletal position)
            // 2. Add the FEM-computed secondary perturbation for free vertices
            // 3. Constrained vertices: u = x_ref exactly (no secondary)
            // 4. Free vertices: u = x_ref + secondary_perturbation

            // Get current secondary perturbation (u_prev - x_ref_prev)
            // Then apply it as initial condition for the new frame

            // Only update constrained DOFs — free vertices keep their velocity!
            for (int i = 0; i < n; i++) {
                if (isConstrained[i]) {
                    for (int d = 0; d < 3; d++) {
                        integrator->SetQ(3*i+d, refFrames[frame][refOffset + 3*i+d]);
                    }
                }
            }

            // Apply gravity only — inertial forces arise naturally from
            // constrained vertex motion through elastic coupling
            integrator->SetExternalForces(gravityForce.data());

            // Time step
            int code = integrator->DoTimestep();
            if (code != 0 && frame < 5) {
                printf("Warning: integrator returned %d at frame %d\n", code, frame);
            }

            // Get updated displacement (x_ref + secondary perturbation)
            integrator->GetqState(u.data());

            // Re-enforce constrained vertices exactly to reference
            for (int i = 0; i < n; i++) {
                if (isConstrained[i]) {
                    for (int d = 0; d < 3; d++) {
                        u[3*i+d] = refFrames[frame][refOffset + 3*i+d];
                    }
                }
            }
        }

        // Write displacement (n*3, no padding — matching .veg vertex count)
        for (int i = 0; i < n; i++) {
            uOutput[3*i] = u[3*i];
            uOutput[3*i+1] = u[3*i+1];
            uOutput[3*i+2] = u[3*i+2];
        }
        snprintf(path, sizeof(path), "%s/u_%d", outputDir, frame);
        writeBinaryF64(path, uOutput.data(), n * 3);

        // Copy reference position file
        snprintf(cmd, sizeof(cmd), "%s/x_%d", motionDir, std::min(frame, numFrames-1));
        snprintf(path, sizeof(path), "%s/x_%d", outputDir, frame);
        copyFile(cmd, path);

        if (frame % 20 == 0 || frame == numFrames - 1) {
            double maxU = 0;
            for (int i = 0; i < n; i++) {
                if (!isConstrained[i]) {
                    for (int d = 0; d < 3; d++)
                        if (fabs(u[3*i+d]) > maxU) maxU = fabs(u[3*i+d]);
                }
            }
            printf("  Frame %d/%d: max_free_disp=%.6f\n", frame, numFrames, maxU);
        }
    }

    // Copy constraint, adjacency, and write stiffness/mass files
    snprintf(cmd, sizeof(cmd), "%s/c", motionDir);
    snprintf(path, sizeof(path), "%s/c", outputDir);
    copyFile(cmd, path);

    snprintf(cmd, sizeof(cmd), "%s/adj", motionDir);
    snprintf(path, sizeof(path), "%s/adj", outputDir);
    copyFile(cmd, path);

    // Write stiffness (all vertices same value)
    {
        std::vector<double> k(n + 1, stiffness); // +1 for padding
        snprintf(path, sizeof(path), "%s/k", outputDir);
        writeBinaryF64(path, k.data(), n + 1);
    }

    // Write mass
    {
        std::vector<double> m(n + 1, 0.0);
        for (int i = 0; i < n; i++) {
            m[i + 1] = massMatrix->GetEntry(3*i, 3*i); // skip padding index 0
        }
        snprintf(path, sizeof(path), "%s/m", outputDir);
        writeBinaryF64(path, m.data(), n + 1);
    }

    // Write offset (zeros)
    {
        std::vector<double> off((n + 1) * 3, 0.0);
        snprintf(path, sizeof(path), "%s/offset", outputDir);
        writeBinaryF64(path, off.data(), (n + 1) * 3);
    }

    printf("Done. Output: %s (%d frames)\n", outputDir, numFrames);

    delete integrator;
    delete forceModel;
    delete fem;
    delete massMatrix;
    delete tetMesh;
    return 0;
}
