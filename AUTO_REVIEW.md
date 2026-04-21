# Auto Review Log — Causal Spatiotemporal Deep Emulator

## Round 1 (2026-04-12)

### Assessment (Summary)
- Score: 3/10
- Verdict: Not ready
- Key criticisms: No MSEA-MGN comparison, sphere failure (0.134 vs 0.0004), no rollout eval, no baseline on characters, causal speeds not validated

### Actions Taken (Round 1 → Round 2)
1. Diagnostic: Stage1 vs Stage2 vs Baseline — showed Stage1 already fails (0.135 MSE)
2. Rollout stability: causal diverges to MSE=273 at frame 100; baseline stays at 2.8
3. Causal speed analysis: learned v≈0.51 constant everywhere; mechanism collapsed
4. Baseline on characters: baseline 10.8x better than causal on all 10 motions

---

## Round 2 (2026-04-13)

### Assessment (Summary)
- Score: 2/10
- Verdict: No — model fundamentally doesn't work
- Key findings:
  - Causal model loses to Deep Emulator on ALL benchmarks (sphere + characters)
  - Previous "66% improvement over ablation" compared two broken models
  - Causal speed collapsed to constant ~0.51
  - Rollout catastrophically unstable
  - 33x slower with 10x worse accuracy

<details>
<summary>Click to expand full reviewer response</summary>

Re-score: 2/10 for a top venue.

The new diagnostics invalidate the central empirical claims. The proposed model is slower, less accurate, less stable, not material-adaptive, and not competitive with the original 2021 baseline.

Key recommendations:
1. Can the model overfit one sphere sequence? (Debug test)
2. Use Deep Emulator as backbone instead of MSEA
3. Add causal cone as regularizer, not restrictive gate
4. Add temporal history minimally to Deep Emulator
5. Train supervised until matching baseline before adding physics loss
6. Validate learned propagation on controlled synthetic data

</details>

### Actions Taken (Round 2 → Round 3)
- Running overfit test on single sphere sequence
- Diagnosing architecture/training bugs
- Testing causal cone as soft prior instead of hard gate

### Status
- Continuing to Round 3
- Score trajectory: 3 → 2
