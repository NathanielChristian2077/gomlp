#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${1:-./dataset}"
RESULT_DIR="results/baseline_dense_h128_b16_lr001"
EPOCHS=100
HIDDEN=128
BATCH=16
LR=0.001
LOG_EVERY=10
SEEDS=(1 2 3 4 5 42)

mkdir -p "${RESULT_DIR}"

SUMMARY="${RESULT_DIR}/summary.csv"
printf "seed,selected_epoch,selected_val_loss,selected_val_acc,test_loss,test_acc,precision,recall,f1\n" > "${SUMMARY}"

extract_value() {
	local key="$1"
	local file="$2"
	grep -Eo "${key}=[0-9]+(\.[0-9]+)?" "${file}" | tail -n 1 | cut -d= -f2
}

for seed in "${SEEDS[@]}"; do
	csv_path="${RESULT_DIR}/seed_${seed}.csv"
	log_path="${RESULT_DIR}/seed_${seed}.log"

	echo "running seed=${seed}"

	go run ./cmd/train \
		--dataset "${DATASET_PATH}" \
		--epochs "${EPOCHS}" \
		--hidden "${HIDDEN}" \
		--batch "${BATCH}" \
		--lr "${LR}" \
		--seed "${seed}" \
		--log-every "${LOG_EVERY}" \
		--out "${csv_path}" | tee "${log_path}"

	selected_epoch="$(extract_value selected_epoch "${log_path}")"
	selected_val_loss="$(extract_value selected_val_loss "${log_path}")"
	selected_val_acc="$(extract_value selected_val_acc "${log_path}")"
	test_loss="$(extract_value test_loss "${log_path}")"
	test_acc="$(extract_value test_acc "${log_path}")"
	precision="$(extract_value precision "${log_path}")"
	recall="$(extract_value recall "${log_path}")"
	f1="$(extract_value f1 "${log_path}")"

	printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
		"${seed}" \
		"${selected_epoch}" \
		"${selected_val_loss}" \
		"${selected_val_acc}" \
		"${test_loss}" \
		"${test_acc}" \
		"${precision}" \
		"${recall}" \
		"${f1}" >> "${SUMMARY}"
done

echo "summary written to ${SUMMARY}"
