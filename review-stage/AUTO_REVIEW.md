# Auto Review Log — V9 Causal Spatiotemporal Deep Emulator

## Round 1 (2026-04-20)

### Assessment (Summary)
- Score: 4/10 (top venue), 6/10 (CGF/Pacific Graphics)
- Verdict: Almost for mid-tier graphics venue. No for top venue.
- Key criticisms:
  1. Scaled GT threatens stiffness generalization claim
  2. Full-range avg still favors baseline
  3. Low-stiffness failure is important (larger deformations)
  4. Baseline fairness: T=5 vs T=3 history
  5. 6x slower inference
  6. Causal cone interpretability needs evidence
  7. Small test setup (6 motions)
  8. Residual bypass weakens causal purity
  9. FEM validation doesn't resolve per-stiffness accuracy

### Reviewer Recommendation
- Target: Pacific Graphics or CGF
- Frame as: "causal-bias graph emulator improves mid-high stiffness extrapolation"
- Critical experiment: independent FEM simulation at unseen stiffness

### Status
- Continuing to Round 2

---

## Round 2 (2026-04-20)

### Assessment (Summary)
- Score: 4.5/10 (top venue), **7/10 (Pacific Graphics)**, 6.5/10 (CGF)
- Verdict: **Almost, leaning Yes for Pacific Graphics**
- Key remaining:
  1. Scaled GT remains main risk (but defensible)
  2. Cone interpretability needed
  3. Visual evidence at different stiffness
  4. T=5 baseline fairness argument is partial

### Reviewer Recommendation
- **Target: Pacific Graphics** (best fit for current evidence)
- Frame as: "improved stiffness extrapolation in validated small-deformation regime"
- Don't overclaim physical generalization

### Score Trajectory
- Round 1: 4/10 (top), 6/10 (mid-tier)
- Round 2: 4.5/10 (top), **7/10 (Pacific Graphics)**

### Status
- **STOP CONDITION MET**: Score ≥ 6 for target venue
- Recommended: Submit to Pacific Graphics with honest framing
