#!/bin/bash
# Script to run evaluation for all available models

# Set the directory paths
EVAL_DIR="/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/eval_ci"
PREDS_DIR="/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/preds"

# Change to evaluation directory
cd "$EVAL_DIR" || exit 1

echo "=========================================="
echo "Running evaluation for all models"
echo "=========================================="
echo ""

# Find all .nc files in predictions directory
for pred_file in "$PREDS_DIR"/*.nc; do
    # Extract model name (filename without .nc extension)
    model_name=$(basename "$pred_file" .nc)
    
    echo ""
    echo "=========================================="
    echo "Processing model: $model_name"
    echo "=========================================="
    echo ""
    
    # Run evaluation for this model
    export MODEL_NAME="$model_name"
    python run_all_eval.py
    
    # Check if evaluation succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully for $model_name"
    else
        echo "✗ Evaluation failed for $model_name"
    fi
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo ""
echo "Results are in: $EVAL_DIR/figs/"
ls -lh "$EVAL_DIR/figs/"
