#!/bin/bash
# Run evaluation for selected predictions (each basename matches MODEL_NAME / *.nc in PREDS_DIR).

# Set the directory paths
EVAL_DIR="/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/eval_neighbour"
PREDS_DIR="/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/data/preds"

# ---------------------------------------------------------------------------
# Choose what to evaluate (exact basename without .nc):
#
# 1) Edit the MODEL_NAMES array below — comment out lines or reorder as needed.
# 2) Or override when invoking:
#      EVAL_MODEL_NAMES="vit_CRPS_n0 vit_CRPS_spectral_n3" ./run_all_models.sh
# 3) Or evaluate every NetCDF in PREDS_DIR:
#      EVAL_ALL_PREDICTIONS=1 ./run_all_models.sh
# ---------------------------------------------------------------------------
MODEL_NAMES=(
  vit_CRPS_n0
  vit_CRPS_n1
  vit_CRPS_n3
  vit_CRPS_spectral_n0
  vit_CRPS_spectral_n1
  vit_CRPS_spectral_n3
)

if [ "${EVAL_ALL_PREDICTIONS:-0}" = "1" ]; then
  MODEL_NAMES=()
  shopt -s nullglob
  for pred_file in "$PREDS_DIR"/*.nc; do
    MODEL_NAMES+=("$(basename "$pred_file" .nc)")
  done
  shopt -u nullglob
elif [ -n "${EVAL_MODEL_NAMES:-}" ]; then
  read -r -a MODEL_NAMES <<< "$EVAL_MODEL_NAMES"
fi

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
  echo "No models to evaluate."
  echo "Set MODEL_NAMES in this script, or EVAL_MODEL_NAMES=\"...\", or EVAL_ALL_PREDICTIONS=1."
  exit 1
fi

cd "$EVAL_DIR" || exit 1

echo "=========================================="
echo "Running evaluation for ${#MODEL_NAMES[@]} model(s)"
echo "=========================================="
printf ' %s\n' "${MODEL_NAMES[@]}"
echo ""

for model_name in "${MODEL_NAMES[@]}"; do
  pred_file="${PREDS_DIR}/${model_name}.nc"
  if [ ! -f "$pred_file" ]; then
    echo ""
    echo "=========================================="
    echo "Skipping missing predictions: $model_name"
    echo "(expected $pred_file)"
    echo "=========================================="
    continue
  fi

  echo ""
  echo "=========================================="
  echo "Processing model: $model_name"
  echo "=========================================="
  echo ""

  export MODEL_NAME="$model_name"
  if python run_all_eval.py; then
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
