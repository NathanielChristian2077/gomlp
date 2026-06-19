#!/usr/bin/env python3
"""Targeted GPU probe for one MLP candidate configuration.

This script is meant for follow-up checks after the broad GPU search. It runs one
architecture/lr/batch configuration across a chosen seed set using train and
validation only. The test split remains untouched.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from gpu_search_pytorch import (
    GroupConfig,
    Stage,
    configure_torch,
    load_existing_results,
    load_split,
    open_summary_writer,
    print_progress,
    resolve_device,
    result_to_row,
    run_or_load,
    select_top_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe one MLP candidate across many seeds using train/validation only.")
    parser.add_argument("--dataset", default="./dataset", help="dataset root with train/cat, train/dog, validation/cat and validation/dog")
    parser.add_argument("--runs", default="runs/probe_128x32x512_lr003_b32", help="output directory")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="CPU thread count used by PyTorch")
    parser.add_argument("--hidden", default="128x32x512", help="hidden architecture, for example 128x32x512")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--batch", type=int, default=32, help="batch size; 0 means full-batch")
    parser.add_argument("--seeds", default="1-42", help="seed list/range, for example 1-42 or 1,3,5")
    parser.add_argument("--max-epochs", type=int, default=500, help="maximum epochs before safety stop")
    parser.add_argument("--stage-name", default="probe", help="stage name stored in summary.csv")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="reuse completed rows from summary.csv")
    parser.add_argument("--deterministic", action="store_true", help="request deterministic PyTorch algorithms when possible")
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_epochs <= 0:
        raise SystemExit("--max-epochs must be positive")

    hidden = parse_hidden(args.hidden)
    seeds = tuple(parse_seeds(args.seeds))
    if not seeds:
        raise SystemExit("--seeds produced no seed values")

    runs_dir = Path(args.runs)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / "summary.csv"

    device = resolve_device(args.device)
    configure_torch(args.workers, args.deterministic, device)

    print(f"device={device}")
    print(f"candidate hidden={format_hidden(hidden)} lr={args.lr:g} batch={args.batch} seeds={seeds[0]}..{seeds[-1]} n={len(seeds)}")

    train_x, train_y = load_split(Path(args.dataset), "train", device)
    val_x, val_y = load_split(Path(args.dataset), "validation", device)
    print(f"train={train_x.shape[0]} validation={val_x.shape[0]} input={train_x.shape[1]} runs={runs_dir}")

    group = GroupConfig(hidden=hidden, learning_rate=args.lr, batch_size=args.batch)
    stage = Stage(args.stage_name, seeds, args.max_epochs, None, True)
    existing = load_existing_results(summary_path) if args.resume else {}
    writer, summary_file = open_summary_writer(summary_path, append=args.resume and summary_path.exists())

    results = []
    try:
        for index, seed in enumerate(seeds, start=1):
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
            results.append(result)
            if not result.cached:
                writer.writerow(result_to_row(result))
                summary_file.flush()
            print_progress(index, len(seeds), result)
    finally:
        summary_file.close()

    print_summary(results)
    print(f"summary={summary_path}")


def parse_hidden(value: str) -> Tuple[int, ...]:
    parts = [part.strip() for part in value.lower().replace(",", "x").split("x") if part.strip()]
    if not parts:
        raise SystemExit("--hidden cannot be empty")
    try:
        hidden = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise SystemExit(f"invalid --hidden value: {value}") from exc
    if any(size <= 0 for size in hidden):
        raise SystemExit("all hidden sizes must be positive")
    return hidden


def format_hidden(hidden: Sequence[int]) -> str:
    return "x".join(str(size) for size in hidden)


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


def print_summary(results: Sequence[object]) -> None:
    completed = [r for r in results if getattr(r, "completed", False)]
    failed = [r for r in results if not getattr(r, "completed", False)]
    if not completed:
        print("no completed runs")
        return

    acc = [float(r.best_val_accuracy) for r in completed]
    f1 = [float(r.best_val_f1) for r in completed]
    loss = [float(r.best_val_loss) for r in completed]
    gap = [abs(float(r.generalization_gap)) for r in completed]

    print("\nAggregate probe summary")
    print(f"completed={len(completed)} failed={len(failed)}")
    print(f"val_acc_mean={mean(acc):.6f} val_acc_min={min(acc):.6f} val_acc_max={max(acc):.6f}")
    print(f"val_f1_mean={mean(f1):.6f} val_loss_mean={mean(loss):.6f} gap_abs_mean={mean(gap):.6f}")

    print("\nTop seeds")
    ranked = sorted(completed, key=lambda r: (-float(r.best_val_accuracy), -float(r.best_val_f1), float(r.best_val_loss)))
    for result in ranked[:10]:
        print(
            f"seed={result.seed} acc={result.best_val_accuracy:.6f} f1={result.best_val_f1:.6f} "
            f"loss={result.best_val_loss:.6f} gap={result.generalization_gap:.6f} "
            f"epoch={result.best_epoch} stop={result.stop_reason}"
        )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
