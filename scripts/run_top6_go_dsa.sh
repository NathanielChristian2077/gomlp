#!/usr/bin/env bash
set -Eeuo pipefail

# Runs the six frozen output-head candidates in the Go implementation.
# Each candidate is evaluated with dense inference, DSA exact and thresholded DSA.
#
# Usage:
#   DATASET=./dataset bash scripts/run_top6_go_dsa.sh
#
# Debug helpers:
#   ONLY_RANK=1 SKIP_BENCH=1 DATASET=./dataset bash scripts/run_top6_go_dsa.sh
#   LOG_DIR=runs/go_top6_output_heads_dsa/logs DATASET=./dataset bash scripts/run_top6_go_dsa.sh

DATASET="${DATASET:-./dataset}"
RUNS_ROOT="${RUNS_ROOT:-runs/go_top6_output_heads_dsa}"
EPOCHS="${EPOCHS:-500}"
THRESHOLDS="${THRESHOLDS:-0,0.025,0.05,0.075,0.1}"
BENCH_REPEAT="${BENCH_REPEAT:-300}"
BENCH_WARMUP="${BENCH_WARMUP:-20}"
GOMAXPROCS_OVERRIDE="${GOMAXPROCS_OVERRIDE:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
ONLY_RANK="${ONLY_RANK:-}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

mkdir -p "$RUNS_ROOT"
LOG_DIR="${LOG_DIR:-$RUNS_ROOT/logs}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$MASTER_LOG") 2>&1
trap 'status=$?; echo; echo "FAILED status=$status line=$LINENO command=$BASH_COMMAND"; echo "log=$MASTER_LOG"; exit $status' ERR

echo "log=$MASTER_LOG"
echo "started_at=$(date -Is)"

if [[ ! -d "$DATASET" ]]; then
  echo "dataset directory not found: $DATASET" >&2
  exit 1
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo
  echo "--- preflight: go test ./... ---"
  go test ./...
fi

MANIFEST="$RUNS_ROOT/top6_manifest.csv"
cat > "$MANIFEST" <<'CSV'
rank,label,hidden,learning_rate,batch_size,head,seed,parameter_count,score,acc_mean,acc_std,acc_min,acc_max,f1_mean,gap_abs_mean,best_seed_acc,best_seed_f1
1,rank1_32x64x512_softmax2_lr0p01_b16_s7,32x64x512,0.01,16,softmax2,7,167522,0.895363,0.589048,0.023177,0.54,0.64,0.574793,0.093968,0.64,0.600000
2,rank2_32x64x512_sigmoid1_lr0p01_b16_s24,32x64x512,0.01,16,sigmoid1,24,167009,0.881118,0.584524,0.024417,0.53,0.64,0.569265,0.115317,0.64,0.660377
3,rank3_64x32x512_sigmoid1_lr0p003_b16_s25,64x32x512,0.003,16,sigmoid1,25,281697,0.872363,0.578095,0.023728,0.52,0.62,0.575089,0.133651,0.62,0.641509
4,rank4_128x32x512_sigmoid1_lr0p003_b32_s9,128x32x512,0.003,32,sigmoid1,9,545953,0.871975,0.576667,0.024365,0.53,0.63,0.570688,0.123413,0.63,0.704000
5,rank5_64x32x512_softmax2_lr0p003_b16_s11,64x32x512,0.003,16,softmax2,11,282210,0.868193,0.578571,0.031965,0.51,0.69,0.580320,0.142222,0.69,0.715596
6,rank6_128x32x512_softmax2_lr0p003_b32_s12,128x32x512,0.003,32,softmax2,12,546466,0.857485,0.574048,0.026100,0.51,0.63,0.556492,0.131032,0.63,0.660550
CSV

echo "dataset=$DATASET"
echo "runs_root=$RUNS_ROOT"
echo "epochs=$EPOCHS thresholds=$THRESHOLDS"
echo "skip_bench=$SKIP_BENCH only_rank=${ONLY_RANK:-all}"
echo "manifest=$MANIFEST"

run_candidate() {
  local rank="$1"
  local label="$2"
  local hidden="$3"
  local lr="$4"
  local batch="$5"
  local head="$6"
  local seed="$7"

  if [[ -n "$ONLY_RANK" && "$ONLY_RANK" != "$rank" ]]; then
    echo "skip rank=$rank because ONLY_RANK=$ONLY_RANK"
    return 0
  fi

  local candidate_dir="$RUNS_ROOT/$label"
  local go_runs="$RUNS_ROOT/go_runs"
  mkdir -p "$candidate_dir" "$go_runs"

  echo
  echo "=== rank=$rank label=$label hidden=$hidden head=$head lr=$lr batch=$batch seed=$seed ==="

  for split in validation test; do
    echo "--- compare split=$split ---"
    go run ./cmd/compare \
      -dataset "$DATASET" \
      -runs "$go_runs" \
      -name "$label" \
      -hidden "$hidden" \
      -head "$head" \
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
      -head "$head" \
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

run_candidate 1 rank1_32x64x512_softmax2_lr0p01_b16_s7 32x64x512 0.01 16 softmax2 7
run_candidate 2 rank2_32x64x512_sigmoid1_lr0p01_b16_s24 32x64x512 0.01 16 sigmoid1 24
run_candidate 3 rank3_64x32x512_sigmoid1_lr0p003_b16_s25 64x32x512 0.003 16 sigmoid1 25
run_candidate 4 rank4_128x32x512_sigmoid1_lr0p003_b32_s9 128x32x512 0.003 32 sigmoid1 9
run_candidate 5 rank5_64x32x512_softmax2_lr0p003_b16_s11 64x32x512 0.003 16 softmax2 11
run_candidate 6 rank6_128x32x512_softmax2_lr0p003_b32_s12 128x32x512 0.003 32 softmax2 12

echo
echo "done"
echo "finished_at=$(date -Is)"
echo "manifest=$MANIFEST"
echo "log=$MASTER_LOG"
echo "per-candidate outputs are under $RUNS_ROOT/<label>/"
