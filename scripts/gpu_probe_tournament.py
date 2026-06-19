#!/usr/bin/env python3
"""Targeted tournament for selecting the best individual MLP.

The broad GPU search is useful for discovering candidates. This script is for the
next step: run a small, explicit list of candidate configurations across many
seeds, summarize robustness, and rank them without touching the test split.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

from gpu_search_pytorch import (
    GroupConfig,
    RunResult,
    Stage,
    configure_torch,
    load_existing_results,
    load_split,
    open_summary_writer,
    print_progress,
    resolve_device,
    result_to_row,
    run_or_load,
)


DEFAULT_CANDIDATES = (
    "256x64x128:0.01:16",
    "128x64x32:0.01:32",
    "32x64x512:0.01:16",
    "512x32x512:0.001:16",
    "512x32x512:0.01:16",
    "256x256x32:0.01:32",
    "64x32x512:0.003:16",
    "256x32:0.003:16",
    "128x32x512:0.002:32",
    "128x32x512:0.003:32",
)


@dataclass(frozen=True)
class Aggregate:
    hidden: str
    learning_rate: float
    batch_size: int
    parameter_count: int
    runs: int
    completed: int
    failed: int
    score: float
    acc_mean: float
    acc_std: float
    acc_min: float
    acc_max: float
    f1_mean: float
    f1_std: float
    loss_mean: float
    gap_abs_mean: float
    epochs_mean: float
    train_time_ms_mean: float
    best_seed: int
    best_seed_acc: float
    best_seed_f1: float
    best_seed_loss: float
    best_seed_gap: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed candidate tournament for the best individual MLP.")
    parser.add_argument("--dataset", default="./dataset", help="dataset root with train/cat, train/dog, validation/cat and validation/dog")
    parser.add_argument("--runs", default="runs/mlp_tournament_v1", help="output directory")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="CPU thread count used by PyTorch")
    parser.add_argument("--seeds", default="1-42", help="seed list/range, for example 1-42 or 1,3,5")
    parser.add_argument("--max-epochs", type=int, default=500, help="maximum epochs per run")
    parser.add_argument("--stage-name", default="tournament", help="stage name stored in summary.csv and run ids")
    parser.add_argument("--candidate", action="append", default=[], help="candidate spec hidden:lr:batch; can be repeated")
    parser.add_argument("--candidates-file", default="", help="optional CSV with hidden,learning_rate,batch_size columns")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="reuse completed rows from summary.csv")
    parser.add_argument("--deterministic", action="store_true", help="request deterministic PyTorch algorithms when possible")
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_epochs <= 0:
        raise SystemExit("--max-epochs must be positive")

    seeds = tuple(parse_seeds(args.seeds))
    if not seeds:
        raise SystemExit("--seeds produced no seed values")

    candidate_specs = list(args.candidate) if args.candidate else list(DEFAULT_CANDIDATES)
    groups = [parse_candidate(spec) for spec in candidate_specs]
    if args.candidates_file:
        groups.extend(read_candidate_file(Path(args.candidates_file)))
    groups = unique_groups(groups)
    if not groups:
        raise SystemExit("no candidates to run")

    runs_dir = Path(args.runs)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / "summary.csv"
    ranking_path = runs_dir / "ranking.csv"
    best_path = runs_dir / "best_individual.csv"

    device = resolve_device(args.device)
    configure_torch(args.workers, args.deterministic, device)

    print(f"device={device}")
    print(f"runs={runs_dir}")
    print(f"candidates={len(groups)} seeds={len(seeds)} total_runs={len(groups) * len(seeds)} max_epochs={args.max_epochs}")
    for group in groups:
        print(f"candidate hidden={group.hidden_label} lr={group.learning_rate:g} batch={group.batch_size}")

    train_x, train_y = load_split(Path(args.dataset), "train", device)
    val_x, val_y = load_split(Path(args.dataset), "validation", device)
    print(f"train={train_x.shape[0]} validation={val_x.shape[0]} input={train_x.shape[1]}")

    stage = Stage(args.stage_name, seeds, args.max_epochs, None, True)
    existing = load_existing_results(summary_path) if args.resume else {}
    writer, summary_file = open_summary_writer(summary_path, append=args.resume and summary_path.exists())

    all_results: List[RunResult] = []
    try:
        total_groups = len(groups)
        for group_index, group in enumerate(groups, start=1):
            print(f"\n[{group_index}/{total_groups}] hidden={group.hidden_label} lr={group.learning_rate:g} batch={group.batch_size}")
            for seed_index, seed in enumerate(seeds, start=1):
                result = run_or_load(
                    group=group,
                    seed=seed,
                    stage=stage,
                    runs_dir=runs_dir,
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                    existing=existing,
                )
                all_results.append(result)
                existing[result.run_id] = result
                if not result.cached:
                    writer.writerow(result_to_row(result))
                    summary_file.flush()
                print_progress(group_index, total_groups, result)
                if seed_index == len(seeds):
                    pass
    finally:
        summary_file.close()

    aggregates = aggregate_results(load_existing_results(summary_path).values(), stage_name=args.stage_name)
    write_ranking(ranking_path, aggregates)
    write_ranking(best_path, aggregates[:1])
    print_ranking(aggregates)
    print(f"summary={summary_path}")
    print(f"ranking={ranking_path}")
    print(f"best_individual={best_path}")


def parse_candidate(spec: str) -> GroupConfig:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 3:
        raise SystemExit(f"invalid candidate spec {spec!r}; expected hidden:lr:batch")
    hidden = parse_hidden(parts[0])
    try:
        lr = float(parts[1])
        batch = int(parts[2])
    except ValueError as exc:
        raise SystemExit(f"invalid candidate spec {spec!r}") from exc
    if batch < 0:
        raise SystemExit("batch size must be >= 0")
    return GroupConfig(hidden=hidden, learning_rate=lr, batch_size=batch)


def parse_hidden(value: str) -> Tuple[int, ...]:
    parts = [part.strip() for part in value.lower().replace(",", "x").split("x") if part.strip()]
    if not parts:
        raise SystemExit("hidden architecture cannot be empty")
    try:
        hidden = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise SystemExit(f"invalid hidden architecture: {value}") from exc
    if any(size <= 0 for size in hidden):
        raise SystemExit("all hidden sizes must be positive")
    return hidden


def parse_seeds(value: str) -> List[int]:
    seeds = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise SystemExit(f"invalid seed range: {token}")
            seeds.update(range(start, end + 1))
        else:
            seeds.add(int(token))
    return sorted(seeds)


def read_candidate_file(path: Path) -> List[GroupConfig]:
    out: List[GroupConfig] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"hidden", "learning_rate", "batch_size"}
        if not required.issubset(reader.fieldnames or set()):
            raise SystemExit(f"{path} must contain columns: hidden, learning_rate, batch_size")
        for row in reader:
            hidden = row["hidden"].strip()
            lr = row["learning_rate"].strip()
            batch = row["batch_size"].strip()
            out.append(parse_candidate(f"{hidden}:{lr}:{batch}"))
    return out


def unique_groups(groups: Sequence[GroupConfig]) -> List[GroupConfig]:
    seen = set()
    out = []
    for group in groups:
        key = group.key
        if key in seen:
            continue
        seen.add(key)
        out.append(group)
    return out


def aggregate_results(results: Iterable[RunResult], stage_name: str) -> List[Aggregate]:
    grouped: Dict[Tuple[str, float, int], List[RunResult]] = {}
    for result in results:
        if result.stage != stage_name:
            continue
        key = (result.hidden, result.learning_rate, result.batch_size)
        grouped.setdefault(key, []).append(result)

    aggregates: List[Aggregate] = []
    for (hidden, lr, batch), values in grouped.items():
        completed = [r for r in values if r.completed]
        failed = [r for r in values if not r.completed]
        if not completed:
            continue

        acc = [r.best_val_accuracy for r in completed]
        f1 = [r.best_val_f1 for r in completed]
        loss = [r.best_val_loss for r in completed]
        gap_abs = [abs(r.generalization_gap) for r in completed]
        epochs = [r.epochs_run for r in completed]
        train_ms = [r.train_time_ms for r in completed]
        best = sorted(completed, key=lambda r: (-r.best_val_accuracy, -r.best_val_f1, r.best_val_loss))[0]
        acc_mean = mean(acc)
        f1_mean = mean(f1)
        acc_std = pstdev(acc) if len(acc) > 1 else 0.0
        f1_std = pstdev(f1) if len(f1) > 1 else 0.0
        gap_mean = mean(gap_abs)
        score = acc_mean + 0.50 * f1_mean - 0.50 * acc_std - 0.25 * gap_mean + 0.10 * min(acc)

        aggregates.append(Aggregate(
            hidden=hidden,
            learning_rate=lr,
            batch_size=batch,
            parameter_count=completed[0].parameter_count,
            runs=len(values),
            completed=len(completed),
            failed=len(failed),
            score=score,
            acc_mean=acc_mean,
            acc_std=acc_std,
            acc_min=min(acc),
            acc_max=max(acc),
            f1_mean=f1_mean,
            f1_std=f1_std,
            loss_mean=mean(loss),
            gap_abs_mean=gap_mean,
            epochs_mean=mean(epochs),
            train_time_ms_mean=mean(train_ms),
            best_seed=best.seed,
            best_seed_acc=best.best_val_accuracy,
            best_seed_f1=best.best_val_f1,
            best_seed_loss=best.best_val_loss,
            best_seed_gap=best.generalization_gap,
        ))

    aggregates.sort(key=lambda r: (-r.score, -r.acc_mean, -r.f1_mean, r.acc_std, r.gap_abs_mean))
    return aggregates


def write_ranking(path: Path, aggregates: Sequence[Aggregate]) -> None:
    fields = [
        "rank", "hidden", "learning_rate", "batch_size", "parameter_count", "runs", "completed", "failed",
        "score", "acc_mean", "acc_std", "acc_min", "acc_max", "f1_mean", "f1_std", "loss_mean",
        "gap_abs_mean", "epochs_mean", "train_time_ms_mean", "best_seed", "best_seed_acc", "best_seed_f1",
        "best_seed_loss", "best_seed_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(aggregates, start=1):
            payload = {field: getattr(row, field) for field in fields if field != "rank"}
            payload["rank"] = rank
            writer.writerow(payload)


def print_ranking(aggregates: Sequence[Aggregate]) -> None:
    print("\nTOP candidates")
    print("rank,hidden,lr,batch,score,acc_mean,acc_std,acc_min,acc_max,f1_mean,gap_abs_mean,best_seed,best_seed_acc,best_seed_f1")
    for rank, row in enumerate(aggregates[:20], start=1):
        print(
            f"{rank},{row.hidden},{row.learning_rate:g},{row.batch_size},{row.score:.6f},"
            f"{row.acc_mean:.6f},{row.acc_std:.6f},{row.acc_min:.6f},{row.acc_max:.6f},"
            f"{row.f1_mean:.6f},{row.gap_abs_mean:.6f},{row.best_seed},{row.best_seed_acc:.6f},{row.best_seed_f1:.6f}"
        )


if __name__ == "__main__":
    main()
