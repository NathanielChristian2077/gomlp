#!/usr/bin/env bash
set -euo pipefail

# Runs the frozen top-3 MLP candidates in the Go implementation, then evaluates
# dense vs DSA on validation/test and benchmarks inference on the test split.
#
# Usage:
#   DATASET=./dataset bash scripts/run_top3_go_dsa.sh
#
# Optional environment variables:
#   RUNS_ROOT=runs/go_top3_dsa
#   EPOCHS=500
#   THRESHOLDS=0,0.025,0.05,0.075,0.1
#   BENCH_REPEAT=300
#   BENCH_WARMUP=20
#   GOMAXPROCS_OVERRIDE=0
#   SKIP_BENCH=0

DATASET="${DATASET:-./dataset}"
RUNS_ROOT="${RUNS_ROOT:-runs/go_top3_dsa}"
EPOCHS="${EPOCHS:-500}"
THRESHOLDS="${THRESHOLDS:-0,0.025,0.05,0.075,0.1}"
BENCH_REPEAT="${BENCH_REPEAT:-300}"
BENCH_WARMUP="${BENCH_WARMUP:-20}"
GOMAXPROCS_OVERRIDE="${GOMAXPROCS_OVERRIDE:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"

if [[ ! -d "$DATASET" ]]; then
  echo "dataset directory not found: $DATASET" >&2
  exit 1
fi

mkdir -p "$RUNS_ROOT"

MANIFEST="$RUNS_ROOT/top3_manifest.csv"
cat > "$MANIFEST" <<'CSV'
rank,label,hidden,learning_rate,batch_size,seed,gpu_score,gpu_acc_mean,gpu_f1_mean,gpu_best_seed_acc,gpu_best_seed_f1
1,top1_32x64x512_lr0p01_b16_s24,32x64x512,0.01,16,24,0.881118,0.584524,0.569265,0.64,0.660377
2,top2_64x32x512_lr0p003_b16_s25,64x32x512,0.003,16,25,0.872363,0.578095,0.575089,0.62,0.641509
3,top3_128x32x512_lr0p003_b32_s9,128x32x512,0.003,32,9,0.871975,0.576667,0.570688,0.63,0.704000
CSV

echo "dataset=$DATASET"
echo "runs_root=$RUNS_ROOT"
echo "epochs=$EPOCHS thresholds=$THRESHOLDS"
echo "manifest=$MANIFEST"

run_candidate() {
  local rank="$1"
  local label="$2"
  local hidden="$3"
  local lr="$4"
  local batch="$5"
  local seed="$6"

  local candidate_dir="$RUNS_ROOT/$label"
  local go_runs="$RUNS_ROOT/go_runs"
  mkdir -p "$candidate_dir" "$go_runs"

  echo
  echo "=== rank=$rank label=$label hidden=$hidden lr=$lr batch=$batch seed=$seed ==="

  for split in validation test; do
    echo "--- compare split=$split ---"
    go run ./cmd/compare \
      -dataset "$DATASET" \
      -runs "$go_runs" \
      -name "$label" \
      -hidden "$hidden" \
      -lr "$lr" \
      -batch "$batch" \
      -seed "$seed" \
      -epochs "$EPOCHS" \
      -split "$split" \
      -thresholds "$THRESHOLDS" \
      -out "$candidate_dir/compare_${split}.csv"
  done

  if [[ "$SKIP_BENCH" != "1" ]]; then
    echo "--- bench split=test ---"
    go run ./cmd/bench \
      -dataset "$DATASET" \
      -runs "$go_runs" \
      -name "$label" \
      -hidden "$hidden" \
      -lr "$lr" \
      -batch "$batch" \
      -seed "$seed" \
      -epochs "$EPOCHS" \
      -split test \
      -thresholds "$THRESHOLDS" \
      -repeat "$BENCH_REPEAT" \
      -warmup "$BENCH_WARMUP" \
      -gomaxprocs "$GOMAXPROCS_OVERRIDE" \
      -out "$candidate_dir/bench_test.csv"
  fi
}

run_candidate 1 top1_32x64x512_lr0p01_b16_s24 32x64x512 0.01 16 24
run_candidate 2 top2_64x32x512_lr0p003_b16_s25 64x32x512 0.003 16 25
run_candidate 3 top3_128x32x512_lr0p003_b32_s9 128x32x512 0.003 32 9

python scripts/summarize_top3_go_dsa.py --root "$RUNS_ROOT"

echo
echo "done"
echo "combined_compare=$RUNS_ROOT/top3_compare_combined.csv"
echo "combined_bench=$RUNS_ROOT/top3_bench_combined.csv"
echo "report_snippet=$RUNS_ROOT/top3_report_snippet.md"
