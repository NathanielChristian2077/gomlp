#!/usr/bin/env python3
"""Compare the two valid binary-output formulations for the frozen top MLPs.

Modes:
  sigmoid1: one output logit trained with BCEWithLogitsLoss.
  softmax2: two output logits trained with CrossEntropyLoss.

A one-neuron softmax is intentionally not implemented because it is degenerate:
softmax over a single logit is always 1, so it cannot represent cat vs dog.
This script uses only train and validation. Keep the test split untouched until
one output formulation is selected.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as torch_nn
import torch.nn.functional as F

from gpu_search_pytorch import configure_torch, load_split, resolve_device


SEARCH_VERSION = "gpu-output-head-tournament-v2"
INPUT_SIZE = 64 * 64
MIN_EPOCHS = 30
PATIENCE = 35
LOW_LEARNING_WINDOW = 15
MIN_VALIDATION_DELTA = 1e-4
LOW_LEARNING_DELTA = 1e-4
DIVERGENT_LOSS_LIMIT = 10.0

DEFAULT_CANDIDATES = (
    "32x64x512:0.01:16",
    "64x32x512:0.003:16",
    "128x32x512:0.003:32",
)
DEFAULT_MODES = ("sigmoid1", "softmax2")
VALID_MODES = set(DEFAULT_MODES)


@dataclass(frozen=True)
class CandidateConfig:
    hidden: Tuple[int, ...]
    learning_rate: float
    batch_size: int
    head_mode: str

    @property
    def hidden_label(self) -> str:
        return "x".join(str(size) for size in self.hidden)

    @property
    def key(self) -> str:
        return f"h={self.hidden_label}|lr={self.learning_rate:g}|bs={self.batch_size}|head={self.head_mode}"


@dataclass(frozen=True)
class Stage:
    name: str
    seeds: Tuple[int, ...]
    max_epochs: int


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass
class RunResult:
    run_id: str
    stage: str
    completed: bool
    cached: bool
    hidden: str
    depth: int
    parameter_count: int
    learning_rate: float
    batch_size: int
    effective_batch_size: int
    head_mode: str
    output_dim: int
    seed: int
    max_epochs: int
    epochs_run: int
    stop_reason: str
    best_epoch: int
    best_train_loss: float
    best_train_accuracy: float
    best_train_precision: float
    best_train_recall: float
    best_train_f1: float
    best_val_loss: float
    best_val_accuracy: float
    best_val_precision: float
    best_val_recall: float
    best_val_f1: float
    final_train_loss: float
    final_train_accuracy: float
    final_val_loss: float
    final_val_accuracy: float
    generalization_gap: float
    val_true_negative: int
    val_false_positive: int
    val_false_negative: int
    val_true_positive: int
    train_time_ms: int
    run_directory: str
    error: str = ""


@dataclass(frozen=True)
class Aggregate:
    hidden: str
    learning_rate: float
    batch_size: int
    head_mode: str
    parameter_count: int
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
    best_seed: int
    best_seed_acc: float
    best_seed_f1: float
    best_seed_loss: float
    best_seed_gap: float


class HeadMLP(torch_nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int], output_dim: int) -> None:
        super().__init__()
        layers: List[torch_nn.Module] = []
        previous = input_size
        for size in hidden_sizes:
            linear = torch_nn.Linear(previous, size)
            torch_nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            torch_nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(torch_nn.ReLU())
            previous = size
        self.hidden = torch_nn.Sequential(*layers)
        self.output = torch_nn.Linear(previous, output_dim)
        torch_nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
        torch_nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sigmoid1 vs softmax2 on frozen top MLP candidates.")
    parser.add_argument("--dataset", default="./dataset", help="dataset root with train/cat, train/dog, validation/cat and validation/dog")
    parser.add_argument("--runs", default="runs/output_head_tournament_v2", help="output directory")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="CPU thread count used by PyTorch")
    parser.add_argument("--seeds", default="1-42", help="seed list/range, for example 1-42 or 1,3,5")
    parser.add_argument("--max-epochs", type=int, default=500, help="maximum epochs per run")
    parser.add_argument("--candidate", action="append", default=[], help="candidate spec hidden:lr:batch; can be repeated")
    parser.add_argument("--mode", action="append", default=[], help="head mode: sigmoid1 or softmax2; can be repeated")
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

    candidate_specs = args.candidate if args.candidate else list(DEFAULT_CANDIDATES)
    modes = validate_modes(args.mode if args.mode else list(DEFAULT_MODES))
    configs = unique_configs(
        CandidateConfig(hidden=hidden, learning_rate=lr, batch_size=batch, head_mode=mode)
        for hidden, lr, batch in (parse_candidate(spec) for spec in candidate_specs)
        for mode in modes
    )

    runs_dir = Path(args.runs)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / "summary.csv"
    ranking_path = runs_dir / "ranking.csv"
    best_path = runs_dir / "best_head.csv"

    device = resolve_device(args.device)
    configure_torch(args.workers, args.deterministic, device)
    print(f"device={device}")
    print(f"configs={len(configs)} seeds={len(seeds)} total_runs={len(configs) * len(seeds)} max_epochs={args.max_epochs}")
    for config in configs:
        print(f"candidate hidden={config.hidden_label} lr={config.learning_rate:g} batch={config.batch_size} head={config.head_mode}")

    train_x, train_y = load_split(Path(args.dataset), "train", device)
    val_x, val_y = load_split(Path(args.dataset), "validation", device)
    print(f"train={train_x.shape[0]} validation={val_x.shape[0]} input={train_x.shape[1]}")

    stage = Stage("output_head", seeds, args.max_epochs)
    existing = load_existing_results(summary_path) if args.resume else {}
    writer, summary_file = open_summary_writer(summary_path, append=args.resume and summary_path.exists())

    try:
        for group_index, config in enumerate(configs, start=1):
            print(f"\n[{group_index}/{len(configs)}] hidden={config.hidden_label} lr={config.learning_rate:g} batch={config.batch_size} head={config.head_mode}")
            for seed in seeds:
                result = run_or_load(config, seed, stage, runs_dir, train_x, train_y, val_x, val_y, existing)
                existing[result.run_id] = result
                if not result.cached:
                    writer.writerow(dataclasses.asdict(result))
                    summary_file.flush()
                print_progress(group_index, len(configs), result)
    finally:
        summary_file.close()

    aggregates = aggregate_results(load_existing_results(summary_path).values())
    write_ranking(ranking_path, aggregates)
    write_ranking(best_path, aggregates[:1])
    print_ranking(aggregates)
    print(f"summary={summary_path}")
    print(f"ranking={ranking_path}")
    print(f"best_head={best_path}")


def run_or_load(config: CandidateConfig, seed: int, stage: Stage, runs_dir: Path, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, existing: Dict[str, RunResult]) -> RunResult:
    run_id = make_run_id(config, seed, stage)
    if run_id in existing:
        return dataclasses.replace(existing[run_id], cached=True)
    run_dir = runs_dir / stage.name / f"{run_id}_{sanitize(config.hidden_label)}_lr{lr_label(config.learning_rate)}_bs{config.batch_size}_{config.head_mode}_seed{seed}"
    try:
        return train_one(config, seed, stage, run_id, run_dir, train_x, train_y, val_x, val_y)
    except Exception as exc:
        return failed_result(config, seed, stage, run_id, run_dir, train_x.shape[0], str(exc))


def train_one(config: CandidateConfig, seed: int, stage: Stage, run_id: str, run_dir: Path, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor) -> RunResult:
    set_run_seed(seed, train_x.device)
    output_dim = output_dim_for_mode(config.head_mode)
    model = HeadMLP(INPUT_SIZE, config.hidden, output_dim).to(train_x.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    batch = effective_batch_size(config.batch_size, train_x.shape[0])
    generator = torch.Generator(device=train_x.device)
    generator.manual_seed(seed + 1000)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_train: Optional[Metrics] = None
    best_val: Optional[Metrics] = None
    no_improvement = 0
    stop_reason = "max_epochs"
    train_losses: List[float] = []
    val_losses: List[float] = []
    metrics_rows: List[List[str]] = []
    started = time.perf_counter()

    for epoch in range(1, stage.max_epochs + 1):
        model.train()
        permutation = torch.randperm(train_x.shape[0], device=train_x.device, generator=generator)
        for start in range(0, train_x.shape[0], batch):
            idx = permutation[start : start + batch]
            xb = train_x.index_select(0, idx)
            yb = train_y.index_select(0, idx)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_for_mode(logits, yb, config.head_mode)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_x, train_y, config.head_mode)
        val_metrics = evaluate(model, val_x, val_y, config.head_mode)
        train_losses.append(train_metrics.loss)
        val_losses.append(val_metrics.loss)
        metrics_rows.append(metrics_to_row(epoch, train_metrics, val_metrics))

        if is_better_validation(val_metrics, best_val):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_train = train_metrics
            best_val = val_metrics
            no_improvement = 0
        else:
            no_improvement += 1

        reason = should_stop(epoch, train_losses, val_losses, train_metrics, val_metrics, no_improvement)
        if reason is not None:
            stop_reason = reason
            break

    final_train = train_metrics
    final_val = val_metrics
    if best_state is None or best_train is None or best_val is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_train = final_train
        best_val = final_val
        best_epoch = len(train_losses)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir / "config.json", config, seed, stage, run_id)
    write_metrics_csv(run_dir / "metrics.csv", metrics_rows)
    torch.save({
        "search_version": SEARCH_VERSION,
        "run_id": run_id,
        "stage": stage.name,
        "hidden_sizes": list(config.hidden),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "seed": seed,
        "head_mode": config.head_mode,
        "output_dim": output_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val.loss,
        "best_validation_accuracy": best_val.accuracy,
        "state_dict": best_state,
    }, run_dir / "best.pt")

    return RunResult(
        run_id=run_id,
        stage=stage.name,
        completed=True,
        cached=False,
        hidden=config.hidden_label,
        depth=len(config.hidden),
        parameter_count=parameter_count(INPUT_SIZE, config.hidden, output_dim),
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        effective_batch_size=batch,
        head_mode=config.head_mode,
        output_dim=output_dim,
        seed=seed,
        max_epochs=stage.max_epochs,
        epochs_run=len(train_losses),
        stop_reason=stop_reason,
        best_epoch=best_epoch,
        best_train_loss=best_train.loss,
        best_train_accuracy=best_train.accuracy,
        best_train_precision=best_train.precision,
        best_train_recall=best_train.recall,
        best_train_f1=best_train.f1,
        best_val_loss=best_val.loss,
        best_val_accuracy=best_val.accuracy,
        best_val_precision=best_val.precision,
        best_val_recall=best_val.recall,
        best_val_f1=best_val.f1,
        final_train_loss=final_train.loss,
        final_train_accuracy=final_train.accuracy,
        final_val_loss=final_val.loss,
        final_val_accuracy=final_val.accuracy,
        generalization_gap=best_train.accuracy - best_val.accuracy,
        val_true_negative=best_val.true_negative,
        val_false_positive=best_val.false_positive,
        val_false_negative=best_val.false_negative,
        val_true_positive=best_val.true_positive,
        train_time_ms=elapsed_ms,
        run_directory=str(run_dir),
    )


def evaluate(model: HeadMLP, x: torch.Tensor, y: torch.Tensor, mode: str) -> Metrics:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(loss_for_mode(logits, y, mode).detach().cpu())
        predicted = predict_class(logits, mode)
        actual = y.long()
        tp = int(((predicted == 1) & (actual == 1)).sum().item())
        tn = int(((predicted == 0) & (actual == 0)).sum().item())
        fp = int(((predicted == 1) & (actual == 0)).sum().item())
        fn = int(((predicted == 0) & (actual == 1)).sum().item())
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(loss, accuracy, precision, recall, f1, tn, fp, fn, tp)


def loss_for_mode(logits: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "sigmoid1":
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), y)
    if mode == "softmax2":
        return F.cross_entropy(logits, y.long())
    raise ValueError(f"unknown mode: {mode}")


def predict_class(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "sigmoid1":
        return (torch.sigmoid(logits.squeeze(1)) >= 0.5).long()
    if mode == "softmax2":
        return torch.argmax(logits, dim=1).long()
    raise ValueError(f"unknown mode: {mode}")


def output_dim_for_mode(mode: str) -> int:
    if mode == "sigmoid1":
        return 1
    if mode == "softmax2":
        return 2
    raise ValueError(f"unknown mode: {mode}")


def is_better_validation(candidate: Metrics, current: Optional[Metrics]) -> bool:
    if current is None:
        return True
    if candidate.accuracy != current.accuracy:
        return candidate.accuracy > current.accuracy
    return candidate.loss < current.loss - MIN_VALIDATION_DELTA


def should_stop(epoch: int, train_losses: Sequence[float], val_losses: Sequence[float], train: Metrics, val: Metrics, no_improvement: int) -> Optional[str]:
    if not math.isfinite(train.loss) or not math.isfinite(val.loss):
        return "divergent_or_non_finite"
    if epoch >= MIN_EPOCHS and (train.loss > DIVERGENT_LOSS_LIMIT or val.loss > DIVERGENT_LOSS_LIMIT):
        return "divergent_or_non_finite"
    if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
        return "validation_patience"
    if epoch >= MIN_EPOCHS and has_low_learning(train_losses, val_losses) and no_improvement >= LOW_LEARNING_WINDOW:
        return "low_learning"
    return None


def has_low_learning(train_losses: Sequence[float], val_losses: Sequence[float]) -> bool:
    if len(train_losses) <= LOW_LEARNING_WINDOW or len(val_losses) <= LOW_LEARNING_WINDOW:
        return False
    previous = len(train_losses) - 1 - LOW_LEARNING_WINDOW
    last = len(train_losses) - 1
    return train_losses[previous] - train_losses[last] < LOW_LEARNING_DELTA and val_losses[previous] - val_losses[last] < LOW_LEARNING_DELTA


def aggregate_results(results: Iterable[RunResult]) -> List[Aggregate]:
    grouped: Dict[Tuple[str, float, int, str], List[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.hidden, result.learning_rate, result.batch_size, result.head_mode), []).append(result)

    aggregates: List[Aggregate] = []
    for (hidden, lr, batch, mode), values in grouped.items():
        completed = [r for r in values if r.completed]
        failed = [r for r in values if not r.completed]
        if not completed:
            continue
        acc = [r.best_val_accuracy for r in completed]
        f1 = [r.best_val_f1 for r in completed]
        loss = [r.best_val_loss for r in completed]
        gap_abs = [abs(r.generalization_gap) for r in completed]
        epochs = [r.epochs_run for r in completed]
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
            head_mode=mode,
            parameter_count=completed[0].parameter_count,
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
            best_seed=best.seed,
            best_seed_acc=best.best_val_accuracy,
            best_seed_f1=best.best_val_f1,
            best_seed_loss=best.best_val_loss,
            best_seed_gap=best.generalization_gap,
        ))
    aggregates.sort(key=lambda r: (-r.score, -r.acc_mean, -r.f1_mean, r.acc_std, r.gap_abs_mean))
    return aggregates


def parse_candidate(spec: str) -> Tuple[Tuple[int, ...], float, int]:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 3:
        raise SystemExit(f"invalid candidate spec {spec!r}; expected hidden:lr:batch")
    return parse_hidden(parts[0]), float(parts[1]), int(parts[2])


def parse_hidden(value: str) -> Tuple[int, ...]:
    parts = [part.strip() for part in value.lower().replace(",", "x").split("x") if part.strip()]
    if not parts:
        raise SystemExit("hidden architecture cannot be empty")
    hidden = tuple(int(part) for part in parts)
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


def validate_modes(modes: Sequence[str]) -> List[str]:
    out = []
    for mode in modes:
        mode = mode.strip().lower()
        if mode == "softmax1":
            raise SystemExit("softmax1 is invalid: softmax over one logit is always 1")
        if mode not in VALID_MODES:
            raise SystemExit(f"invalid mode {mode!r}; expected one of {sorted(VALID_MODES)}")
        if mode not in out:
            out.append(mode)
    return out


def unique_configs(configs: Iterable[CandidateConfig]) -> List[CandidateConfig]:
    seen = set()
    out = []
    for config in configs:
        if config.key in seen:
            continue
        seen.add(config.key)
        out.append(config)
    return out


def set_run_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def effective_batch_size(batch_size: int, train_size: int) -> int:
    return train_size if batch_size <= 0 or batch_size > train_size else batch_size


def parameter_count(input_size: int, hidden: Sequence[int], output_dim: int) -> int:
    total = 0
    previous = input_size
    for size in hidden:
        total += previous * size + size
        previous = size
    total += previous * output_dim + output_dim
    return total


def make_run_id(config: CandidateConfig, seed: int, stage: Stage) -> str:
    payload = {
        "version": SEARCH_VERSION,
        "stage": dataclasses.asdict(stage),
        "hidden": list(config.hidden),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "head_mode": config.head_mode,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def failed_result(config: CandidateConfig, seed: int, stage: Stage, run_id: str, run_dir: Path, train_size: int, error: str) -> RunResult:
    output_dim = output_dim_for_mode(config.head_mode)
    return RunResult(
        run_id=run_id,
        stage=stage.name,
        completed=False,
        cached=False,
        hidden=config.hidden_label,
        depth=len(config.hidden),
        parameter_count=parameter_count(INPUT_SIZE, config.hidden, output_dim),
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        effective_batch_size=effective_batch_size(config.batch_size, train_size),
        head_mode=config.head_mode,
        output_dim=output_dim,
        seed=seed,
        max_epochs=stage.max_epochs,
        epochs_run=0,
        stop_reason="failed",
        best_epoch=0,
        best_train_loss=0,
        best_train_accuracy=0,
        best_train_precision=0,
        best_train_recall=0,
        best_train_f1=0,
        best_val_loss=0,
        best_val_accuracy=0,
        best_val_precision=0,
        best_val_recall=0,
        best_val_f1=0,
        final_train_loss=0,
        final_train_accuracy=0,
        final_val_loss=0,
        final_val_accuracy=0,
        generalization_gap=0,
        val_true_negative=0,
        val_false_positive=0,
        val_false_negative=0,
        val_true_positive=0,
        train_time_ms=0,
        run_directory=str(run_dir),
        error=error,
    )


def summary_fields() -> List[str]:
    return list(dataclasses.asdict(failed_result(CandidateConfig((1,), 0.1, 1, "sigmoid1"), 1, Stage("x", (1,), 1), "x", Path("x"), 1, "")).keys())


def open_summary_writer(path: Path, append: bool) -> Tuple[csv.DictWriter, object]:
    mode = "a" if append else "w"
    file = path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(file, fieldnames=summary_fields())
    if not append:
        writer.writeheader()
        file.flush()
    return writer, file


def load_existing_results(path: Path) -> Dict[str, RunResult]:
    if not path.exists():
        return {}
    out: Dict[str, RunResult] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                result = row_to_result(row)
            except Exception:
                continue
            if result.completed:
                out[result.run_id] = result
    return out


def row_to_result(row: Dict[str, str]) -> RunResult:
    return RunResult(
        run_id=row["run_id"],
        stage=row["stage"],
        completed=parse_bool(row["completed"]),
        cached=parse_bool(row.get("cached", "false")),
        hidden=row["hidden"],
        depth=int(row["depth"]),
        parameter_count=int(row["parameter_count"]),
        learning_rate=float(row["learning_rate"]),
        batch_size=int(row["batch_size"]),
        effective_batch_size=int(row["effective_batch_size"]),
        head_mode=row["head_mode"],
        output_dim=int(row["output_dim"]),
        seed=int(row["seed"]),
        max_epochs=int(row["max_epochs"]),
        epochs_run=int(row["epochs_run"]),
        stop_reason=row["stop_reason"],
        best_epoch=int(row["best_epoch"]),
        best_train_loss=float(row["best_train_loss"]),
        best_train_accuracy=float(row["best_train_accuracy"]),
        best_train_precision=float(row["best_train_precision"]),
        best_train_recall=float(row["best_train_recall"]),
        best_train_f1=float(row["best_train_f1"]),
        best_val_loss=float(row["best_val_loss"]),
        best_val_accuracy=float(row["best_val_accuracy"]),
        best_val_precision=float(row["best_val_precision"]),
        best_val_recall=float(row["best_val_recall"]),
        best_val_f1=float(row["best_val_f1"]),
        final_train_loss=float(row["final_train_loss"]),
        final_train_accuracy=float(row["final_train_accuracy"]),
        final_val_loss=float(row["final_val_loss"]),
        final_val_accuracy=float(row["final_val_accuracy"]),
        generalization_gap=float(row["generalization_gap"]),
        val_true_negative=int(row["val_true_negative"]),
        val_false_positive=int(row["val_false_positive"]),
        val_false_negative=int(row["val_false_negative"]),
        val_true_positive=int(row["val_true_positive"]),
        train_time_ms=int(row["train_time_ms"]),
        run_directory=row["run_directory"],
        error=row.get("error", ""),
    )


def write_ranking(path: Path, aggregates: Sequence[Aggregate]) -> None:
    fields = ["rank"] + (list(dataclasses.asdict(aggregates[0]).keys()) if aggregates else [])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, aggregate in enumerate(aggregates, start=1):
            row = dataclasses.asdict(aggregate)
            row["rank"] = rank
            writer.writerow(row)


def print_ranking(aggregates: Sequence[Aggregate]) -> None:
    print("\nTOP output heads")
    print("rank,hidden,lr,batch,head,score,acc_mean,acc_std,acc_min,acc_max,f1_mean,gap_abs_mean,best_seed,best_seed_acc,best_seed_f1")
    for rank, row in enumerate(aggregates[:20], start=1):
        print(f"{rank},{row.hidden},{row.learning_rate:g},{row.batch_size},{row.head_mode},{row.score:.6f},{row.acc_mean:.6f},{row.acc_std:.6f},{row.acc_min:.6f},{row.acc_max:.6f},{row.f1_mean:.6f},{row.gap_abs_mean:.6f},{row.best_seed},{row.best_seed_acc:.6f},{row.best_seed_f1:.6f}")


def metrics_to_row(epoch: int, train: Metrics, val: Metrics) -> List[str]:
    return [str(epoch), f"{train.loss:.8f}", f"{train.accuracy:.8f}", f"{train.precision:.8f}", f"{train.recall:.8f}", f"{train.f1:.8f}", f"{val.loss:.8f}", f"{val.accuracy:.8f}", f"{val.precision:.8f}", f"{val.recall:.8f}", f"{val.f1:.8f}"]


def write_metrics_csv(path: Path, rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"])
        writer.writerows(rows)


def write_run_config(path: Path, config: CandidateConfig, seed: int, stage: Stage, run_id: str) -> None:
    payload = {"search_version": SEARCH_VERSION, "run_id": run_id, "stage": dataclasses.asdict(stage), "hidden_sizes": list(config.hidden), "learning_rate": config.learning_rate, "batch_size": config.batch_size, "head_mode": config.head_mode, "seed": seed}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_progress(group_index: int, group_total: int, result: RunResult) -> None:
    status = "cached" if result.cached else "trained"
    if not result.completed:
        status = "failed"
    print(f"{status} group={group_index}/{group_total} hidden={result.hidden} head={result.head_mode} lr={result.learning_rate:g} batch={result.batch_size} seed={result.seed} best_val_acc={result.best_val_accuracy:.4f} best_val_f1={result.best_val_f1:.4f} epoch={result.best_epoch} stop={result.stop_reason}")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def lr_label(lr: float) -> str:
    return f"{lr:g}".replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
