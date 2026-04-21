#!/bin/bash
# Generate VegaFEM ground truth for stiffness robustness experiments.
#
# Train stiffness (5 values): 50000, 200000, 500000, 2000000, 5000000
# Test stiffness (8 values): 25000, 75000, 150000, 300000, 750000, 1000000, 3000000, 8000000
#
# Total: 13 stiffness values, more test than train.

VEGA_SIM="$(dirname $0)/run_sim"
DATA_ROOT="$(dirname $0)/../data"
CHAR_ROOT="$DATA_ROOT/character_dataset"
OUT_ROOT="$DATA_ROOT/vega_stiffness"
FRAMES=100
DAMPING=0.01

TRAIN_STIFF="50000 200000 500000 2000000 5000000"
TEST_STIFF="25000 75000 150000 300000 750000 1000000 3000000 8000000"
ALL_STIFF="$TRAIN_STIFF $TEST_STIFF"

# Characters to generate
CHARACTERS="mousey michelle kaya big_vegas ortiz"
# Motion per character
declare -A MOTIONS
MOTIONS[mousey]="dancing_1"
MOTIONS[michelle]="cross_jumps"
MOTIONS[kaya]="zombie_scream"
MOTIONS[big_vegas]="cross_jumps"
MOTIONS[ortiz]="jazz_dancing"

echo "=========================================="
echo " VegaFEM Stiffness Data Generation"
echo "=========================================="
echo "Train stiffness: $TRAIN_STIFF"
echo "Test stiffness:  $TEST_STIFF"
echo "Characters: $CHARACTERS"
echo "Frames: $FRAMES"

SEQ=1
for CHAR in $CHARACTERS; do
    MOTION=${MOTIONS[$CHAR]}
    VEG="$CHAR_ROOT/$CHAR/${CHAR}.veg"
    MOTION_DIR="$CHAR_ROOT/$CHAR/$MOTION"

    if [ ! -f "$VEG" ]; then
        echo "SKIP: $VEG not found"
        continue
    fi

    echo ""
    echo "=== $CHAR/$MOTION ==="

    for STIFF in $ALL_STIFF; do
        OUT_DIR="$OUT_ROOT/$CHAR/$MOTION/$SEQ"
        echo "  Stiffness=$STIFF -> $OUT_DIR"
        $VEGA_SIM "$VEG" "$MOTION_DIR" "$OUT_DIR" $STIFF $FRAMES $DAMPING
        SEQ=$((SEQ + 1))
    done
    SEQ=1  # reset per character
done

echo ""
echo "=========================================="
echo " Generation complete: $OUT_ROOT"
echo "=========================================="

# Write metadata
cat > "$OUT_ROOT/stiffness_config.txt" << EOF
train_stiffness: $TRAIN_STIFF
test_stiffness: $TEST_STIFF
frames: $FRAMES
damping: $DAMPING
characters: $CHARACTERS
EOF
