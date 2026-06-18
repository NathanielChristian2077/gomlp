#!/usr/bin/env python3
"""GPU-assisted MLP architecture search for the cat/dog dataset.

This script intentionally does not evaluate the test split. It uses only train and
validation so the final test set remains untouched until a small set of candidates
has been selected.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("Pillow is required. Install with: python -m pip install pillow") from exc

try:
    import torch
    import torch.nn as torch_nn
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("PyTorch is required. Install a CUDA build of torch before running this script.") from exc


IMAGE_SIZE = 64
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
CLASS_TO_LABEL = {"cat": 0.0, "dog": 1.0}
HIDDEN_CANDIDATES = (16, 32, 64, 128, 256, 512)
LEARNING_RATES = (0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01)
BATCH_SIZES = (0, 16, 32, 64)
SEEDS = tuple(range(1, 43))
SEARCH_VERSION = "gpu-search-pytorch-v1"


@dataclass(frozen=True)
class GroupConfig:
    hidden: Tuple[int, ...]
    learning_rate: float
    batch_size: int

    @property
    def hidden_label(self) -> str:
        return "x".join(str(v) for v in self.hidden)

    @property
    def key(self) -> str:
        return f"h={self.hidden_label}|lr={self.learning_rate:g}|bs={self.batch_size}"


@dataclass(frozen=True)
class Stage:
    name: str
    seeds: Tuple[int, ...]
    max_epochs: int
    keep_top: Optional[int]
    save_artifacts: bool


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


class MLP(torch_nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int]) -> None:
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
        self.output = torch_nn.Linear(previous, 1)
        torch_nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
        torch_nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return self.output(x).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch GPU search for MLP architectures using train/validation only.")
    parser.add_argument("--dataset", default="./dataset", help="dataset root with train/cat, train/dog, validation/cat and validation/dog")
    parser.add_argument("--runs", default="runs/gpu_search_pytorch_v1", help="output directory")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--strategy", choices=("halving", "exhaustive"), default="halving", help="halving is recommended; exhaustive runs the full 260064 config/seed grid")
    parser.add_argument("--workers", type=int, default=1, help="CPU thread count used by PyTorch; GPU execution remains sequential for stability")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="reuse completed rows from summary.csv")
    parser.add_argument("--deterministic", action="store_true", help="request deterministic PyTorch algorithms when possible")
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    runs_dir = Path(args.runs)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / "summary.csv"
    selected_path = runs_dir / "selected_groups.json"

    device = resolve_device(args.device)
    configure_torch(args.workers, args.deterministic, device)

    print(f"device={device} torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    train_x, train_y = load_split(Path(args.dataset), "train", device)
    val_x, val_y = load_split(Path(args.dataset), "validation", device)
    print(f"train={train_x.shape[0]} validation={val_x.shape[0]} input={train_x.shape[1]} runs={runs_dir}")

    groups = generate_groups()
    stages = build_stages(args.strategy)
    existing = load_existing_results(summary_path) if args.resume else {}
    writer, summary_file = open_summary_writer(summary_path, append=args.resume and summary_path.exists())

    try:
        current_groups = groups
        all_stage_selection: Dict[str, List[str]] = {}
        for stage in stages:
            expected_runs = len(current_groups) * len(stage.seeds)
            print(f"stage={stage.name} groups={len(current_groups)} seeds={len(stage.seeds)} runs={expected_runs} max_epochs={stage.max_epochs} keep_top={stage.keep_top}")
            stage_results: List[RunResult] = []

            for group_index, group in enumerate(current_groups, start=1):
                for seed in stage.seeds:
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
                    stage_results.append(result)
                    if not result.cached:
                        writer.writerow(result_to_row(result))
                        summary_file.flush()
                    print_progress(group_index, len(current_groups), result)

            if stage.keep_top is not None:
                current_groups = select_top_groups(stage_results, current_groups, stage.keep_top)
            all_stage_selection[stage.name] = [group.key for group in current_groups]
            selected_path.write_text(json.dumps(all_stage_selection, indent=2), encoding="utf-8")
    finally:
        summary_file.close()

    print(f"summary={summary_path}")
    print(f"selected_groups={selected_path}")


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested, but torch.cuda.is_available() is false")
    return device


def configure_torch(workers: int, deterministic: bool, device: torch.device) -> None:
    torch.set_num_threads(workers)
    random.seed(0)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_split(dataset_root: Path, split: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[float] = []
    for class_name, label in CLASS_TO_LABEL.items():
        folder = dataset_root / split / class_name
        if not folder.is_dir():
            raise SystemExit(f"missing dataset folder: {folder}")
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if not paths:
            raise SystemExit(f"empty dataset folder: {folder}")
        for path in paths:
            xs.append(load_image(path))
            ys.append(label)
    x = torch.stack(xs).to(device=device, dtype=torch.float32, non_blocking=True)
    y = torch.tensor(ys, device=device, dtype=torch.float32)
    return x, y


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        data = torch.tensor(list(image.getdata()), dtype=torch.float32) / 255.0
        return data.view(INPUT_SIZE)


def generate_groups() -> List[GroupConfig]:
    architectures: List[Tuple[int, ...]] = []
    for a in HIDDEN_CANDIDATES:
        architectures.append((a,))
    for a in HIDDEN_CANDIDATES:
        for b in HIDDEN_CANDIDATES:
            architectures.append((a, b))
    for a in HIDDEN_CANDIDATES:
        for b in HIDDEN_CANDIDATES:
            for c in HIDDEN_CANDIDATES:
                architectures.append((a, b, c))

    groups = [GroupConfig(hidden=arch, learning_rate=lr, batch_size=batch) for arch in architectures for lr in LEARNING_RATES for batch in BATCH_SIZES]
    groups.sort(key=lambda g: (len(g.hidden), g.hidden, g.learning_rate, g.batch_size))
    return groups


def build_stages(strategy: str) -> List[Stage]:
    if strategy == "exhaustive":
        return [Stage("exhaustive", SEEDS, 500, None, True)]
    return [
        Stage("screen_s1", (1,), 20, 1024, False),
        Stage("screen_s2", tuple(range(1, 6)), 60, 256, False),
        Stage("screen_s3", tuple(range(1, 15)), 150, 64, False),
        Stage("final_s4", SEEDS, 500, None, True),
    ]


def run_or_load(
    group: GroupConfig,
    seed: int,
    stage: Stage,
    runs_dir: Path,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    existing: Dict[str, RunResult],
) -> RunResult:
    run_id = make_run_id(group, seed, stage)
    if run_id in existing:
        cached = dataclasses.replace(existing[run_id], cached=True)
        return cached

    run_dir = runs_dir / stage.name / f"{run_id}_{sanitize_name(group.hidden_label)}_lr{lr_label(group.learning_rate)}_bs{group.batch_size}_seed{seed}"
    try:
        return train_one(group, seed, stage, run_id, run_dir, train_x, train_y, val_x, val_y)
    except Exception as exc:  # keep the full search alive after a bad config
        return RunResult(
            run_id=run_id,
            stage=stage.name,
            completed=False,
            cached=False,
            hidden=group.hidden_label,
            depth=len(group.hidden),
            parameter_count=parameter_count(INPUT_SIZE, group.hidden),
            learning_rate=group.learning_rate,
            batch_size=group.batch_size,
            effective_batch_size=effective_batch_size(group.batch_size, train_x.shape[0]),
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
            error=str(exc),
        )


def train_one(
    group: GroupConfig,
    seed: int,
    stage: Stage,
    run_id: str,
    run_dir: Path,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> RunResult:
    set_run_seed(seed, train_x.device)
    model = MLP(INPUT_SIZE, group.hidden).to(train_x.device)
    criterion = torch_nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=group.learning_rate)
    batch = effective_batch_size(group.batch_size, train_x.shape[0])
    generator = torch.Generator(device=train_x.device)
    generator.manual_seed(seed + 1000)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_train: Optional[Metrics] = None
    best_val: Optional[Metrics] = None
    best_set = False
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
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, criterion, train_x, train_y)
        val_metrics = evaluate(model, criterion, val_x, val_y)
        train_losses.append(train_metrics.loss)
        val_losses.append(val_metrics.loss)
        metrics_rows.append(metrics_to_row(epoch, train_metrics, val_metrics))

        if is_better_validation(val_metrics, best_val, best_set):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_train = train_metrics
            best_val = val_metrics
            best_set = True
            no_improvement = 0
        else:
            no_improvement += 1

        if should_stop_divergent(epoch, train_metrics, val_metrics):
            stop_reason = "divergent_or_non_finite"
            break
        if epoch >= 30 and no_improvement >= 35:
            stop_reason = "validation_patience"
            break
        if epoch >= 30 and has_low_learning(train_losses, val_losses, 15, 1e-4) and no_improvement >= 15:
            stop_reason = "low_learning"
            break

    final_train = train_metrics
    final_val = val_metrics
    if best_train is None or best_val is None or best_state is None:
        best_train = final_train
        best_val = final_val
        best_epoch = len(train_losses)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if stage.save_artifacts:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_run_config(run_dir / "config.json", group, seed, stage, run_id)
        write_metrics_csv(run_dir / "metrics.csv", metrics_rows)
        torch.save(
            {
                "search_version": SEARCH_VERSION,
                "run_id": run_id,
                "stage": stage.name,
                "hidden_sizes": list(group.hidden),
                "learning_rate": group.learning_rate,
                "batch_size": group.batch_size,
                "effective_batch_size": batch,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val.loss,
                "best_validation_accuracy": best_val.accuracy,
                "state_dict": best_state,
            },
            run_dir / "best.pt",
        )

    return RunResult(
        run_id=run_id,
        stage=stage.name,
        completed=True,
        cached=False,
        hidden=group.hidden_label,
        depth=len(group.hidden),
        parameter_count=parameter_count(INPUT_SIZE, group.hidden),
        learning_rate=group.learning_rate,
        batch_size=group.batch_size,
        effective_batch_size=batch,
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
        run_directory=str(run_dir if stage.save_artifacts else ""),
    )


def set_run_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: MLP, criterion: torch_nn.Module, x: torch.Tensor, y: torch.Tensor) -> Metrics:
    model.eval()
    logits = model(x)
    loss = float(criterion(logits, y).detach().cpu())
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= 0.5
    actual = y >= 0.5
    tp = int((predicted & actual).sum().item())
    tn = int((~predicted & ~actual).sum().item())
    fp = int((predicted & ~actual).sum().item())
    fn = int((~predicted & actual).sum().item())
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(loss, accuracy, precision, recall, f1, tn, fp, fn, tp)


def is_better_validation(candidate: Metrics, current: Optional[Metrics], current_set: bool) -> bool:
    if not current_set or current is None:
        return True
    if candidate.accuracy != current.accuracy:
        return candidate.accuracy > current.accuracy
    return candidate.loss < current.loss - 1e-4


def should_stop_divergent(epoch: int, train_metrics: Metrics, val_metrics: Metrics) -> bool:
    if not math.isfinite(train_metrics.loss) or not math.isfinite(val_metrics.loss):
        return True
    return epoch >= 30 and (train_metrics.loss > 10.0 or val_metrics.loss > 10.0)


def has_low_learning(train_losses: Sequence[float], val_losses: Sequence[float], window: int, min_delta: float) -> bool:
    if len(train_losses) <= window or len(val_losses) <= window:
        return False
    previous = len(train_losses) - 1 - window
    last = len(train_losses) - 1
    train_improvement = train_losses[previous] - train_losses[last]
    val_improvement = val_losses[previous] - val_losses[last]
    return train_improvement < min_delta and val_improvement < min_delta


def select_top_groups(results: Sequence[RunResult], previous_groups: Sequence[GroupConfig], keep_top: int) -> List[GroupConfig]:
    grouped: Dict[str, List[RunResult]] = {}
    by_key = {group.key: group for group in previous_groups}
    for result in results:
        if result.completed:
            key = f"h={result.hidden}|lr={result.learning_rate:g}|bs={result.batch_size}"
            grouped.setdefault(key, []).append(result)

    scored: List[Tuple[float, float, float, int, str]] = []
    for key, values in grouped.items():
        val_acc = mean(v.best_val_accuracy for v in values)
        val_loss = mean(v.best_val_loss for v in values)
        gap = mean(abs(v.generalization_gap) for v in values)
        epochs = int(mean(v.epochs_run for v in values))
        scored.append((-val_acc, val_loss, gap, epochs, key))

    scored.sort()
    selected = [by_key[key] for *_, key in scored[:keep_top] if key in by_key]
    print(f"selected_groups={len(selected)} keep_top={keep_top}")
    return selected


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def effective_batch_size(batch_size: int, train_size: int) -> int:
    if batch_size <= 0 or batch_size > train_size:
        return train_size
    return batch_size


def parameter_count(input_size: int, hidden: Sequence[int]) -> int:
    total = 0
    previous = input_size
    for size in hidden:
        total += previous * size + size
        previous = size
    total += previous + 1
    return total


def make_run_id(group: GroupConfig, seed: int, stage: Stage) -> str:
    payload = {
        "version": SEARCH_VERSION,
        "stage": dataclasses.asdict(stage),
        "hidden": list(group.hidden),
        "learning_rate": group.learning_rate,
        "batch_size": group.batch_size,
        "seed": seed,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def lr_label(lr: float) -> str:
    return f"{lr:g}".replace(".", "p").replace("-", "m")


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
        reader = csv.DictReader(file)
        for row in reader:
            try:
                result = row_to_result(row)
            except Exception:
                continue
            if result.completed:
                out[result.run_id] = result
    return out


def summary_fields() -> List[str]:
    return [
        "run_id", "stage", "completed", "cached", "hidden", "depth", "parameter_count",
        "learning_rate", "batch_size", "effective_batch_size", "seed", "max_epochs",
        "epochs_run", "stop_reason", "best_epoch", "best_train_loss", "best_train_accuracy",
        "best_train_precision", "best_train_recall", "best_train_f1", "best_val_loss",
        "best_val_accuracy", "best_val_precision", "best_val_recall", "best_val_f1",
        "final_train_loss", "final_train_accuracy", "final_val_loss", "final_val_accuracy",
        "generalization_gap", "val_true_negative", "val_false_positive", "val_false_negative",
        "val_true_positive", "train_time_ms", "run_directory", "error",
    ]


def result_to_row(result: RunResult) -> Dict[str, object]:
    return dataclasses.asdict(result)


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
        run_directory=row.get("run_directory", ""),
        error=row.get("error", ""),
    )


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def metrics_to_row(epoch: int, train: Metrics, val: Metrics) -> List[str]:
    return [
        str(epoch), f"{train.loss:.8f}", f"{train.accuracy:.8f}", f"{train.precision:.8f}",
        f"{train.recall:.8f}", f"{train.f1:.8f}", f"{val.loss:.8f}", f"{val.accuracy:.8f}",
        f"{val.precision:.8f}", f"{val.recall:.8f}", f"{val.f1:.8f}",
    ]


def write_metrics_csv(path: Path, rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"])
        writer.writerows(rows)


def write_run_config(path: Path, group: GroupConfig, seed: int, stage: Stage, run_id: str) -> None:
    payload = {
        "search_version": SEARCH_VERSION,
        "run_id": run_id,
        "stage": dataclasses.asdict(stage),
        "hidden_sizes": list(group.hidden),
        "learning_rate": group.learning_rate,
        "batch_size": group.batch_size,
        "seed": seed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_progress(group_index: int, group_total: int, result: RunResult) -> None:
    status = "cached" if result.cached else "trained"
    if not result.completed:
        status = "failed"
    print(
        f"{status} stage={result.stage} group={group_index}/{group_total} run={result.run_id} "
        f"hidden={result.hidden} lr={result.learning_rate:g} batch={result.batch_size} seed={result.seed} "
        f"epochs={result.epochs_run} stop={result.stop_reason} val_acc={result.best_val_accuracy:.4f} "
        f"val_loss={result.best_val_loss:.6f} gap={result.generalization_gap:.4f} time_ms={result.train_time_ms} error={result.error}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Partial summary.csv was preserved.", file=sys.stderr)
        raise
