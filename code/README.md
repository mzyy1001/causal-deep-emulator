# Code

Run `train.py` to train the model (Stage 1 supervised + Stage 2 physics self-supervised).

Run `test.py` to generate rollout predictions. Set the `flag` variable to control rendering:
- `"tet"` — volumetric mesh rendering
- `"surface"` — surface mesh rendering (requires Vega FEM utilities)
- `"tet_surface"` — both, stacked vertically

For surface rendering, compile the Vega FEM utilities in `vega_FEM/` and ensure `libosmesa6`, `libglu1-mesa`, and `freeglut3` are installed.
