"""
Global configuration for the causal spatiotemporal deep emulator.
Edit the constants below to change model and training settings.
"""

# ─── Temporal Window ───────────────────────────────────────────────────────────
# T: number of past frames used for spatiotemporal aggregation.
# Changing this value affects data loading, model architecture, and inference.
TEMPORAL_WINDOW = 5

# ─── Model Architecture ───────────────────────────────────────────────────────
NUM_SCALES = 3       # L: number of coarsening levels (total scales = L+1)
MSG_DIM = 128        # hidden dimension for message passing
PROP_DIM = 3         # dimension of per-vertex property features

# ─── Causal Cone ─────────────────────────────────────────────────────────────
# Controls how sharply the soft causal mask transitions from 1 to 0.
# Higher = closer to binary; lower = smoother gradient flow.
MASK_SHARPNESS = 5.0

# Set to False to disable causal cone masking (ablation baseline).
# When False, all spatial scales are equally accessible at all temporal delays.
USE_CAUSAL_CONE = True
