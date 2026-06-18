#!/usr/bin/env python3
"""Batched-seed PyTorch GPU search for the cat/dog MLP problem.

This script keeps the same summary.csv schema and run-id convention as
scripts/gpu_search_pytorch.py, but trains several seeds of the same
architecture/lr/batch group in parallel on the GPU.

It intentionally uses only train and validation. The test split remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import random
import sys
import time
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
    import torch.nn.functional as F
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

MIN_EPOCHS = 30
PATIENCE = 35
LOW_LEARNING_WINDOW = 15
MIN_VALIDATION_DELTA = 1e-4
LOW_LEARNING_DELTA = 1e-4
DIVERGENT_LOSS_LIMIT = 10.0


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


SeedSnapshot = List[Tuple[torch.Tensor, torch.Tensor]]


class BatchedMLP(torch_nn.Module):
    """A group of same-shape MLPs trained in parallel across the seed dimension."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], seeds: Sequence[int], device: torch.device) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("at least one hidden layer is required")
        if not seeds:
            raise ValueError("at least one seed is required")

        self.hidden_sizes = tuple(hidden_sizes)
        self.seeds = tuple(seeds)
        self.weights = torch_nn.ParameterList()
        self.biases = torch_nn.ParameterList()

        previous = input_size
        layer_shapes = list(hidden_sizes) + [1]
        for layer_index, out_features in enumerate(layer_shapes):
            nonlinearity = "relu" if layer_index < len(hidden_sizes) else "linear"
            weight, bias = init_layer_for_seeds(previous, out_features, seeds, layer_index, nonlinearity, device)
            self.weights.append(torch_nn.Parameter(weight))
            self.biases.append(torch_nn.Parameter(bias))
            previous = out_features

    def forward_active(self, x: torch.Tensor, active_idx: torch.Tensor) -> torch.Tensor:
        if active_idx.numel() == 0:
            raise ValueError("active_idx cannot be empty")

        if x.dim() == 2:
            hidden = torch.einsum("bi,sio->sbo", x, self.weights[0].index_select(0, active_idx))
        elif x.dim() == 3:
            hidden = torch.einsum("sbi,sio->sbo", x, self.weights[0].index_select(0, active_idx))
        else:
            raise ValueError(f"invalid input rank: expected 2 or 3, got {x.dim()}")
        hidden = hidden + self.biases[0].index_select(0, active_idx).unsqueeze(1)
        hidden = torch.relu(hidden)

        for layer_index in range(1, len(self.hidden_sizes)):
            hidden = torch.einsum("sbi,sio->sbo", hidden, self.weights[layer_index].index_select(0, active_idx))
            hidden = hidden + self.biases[layer_index].index_select(0, active_idx).unsqueeze(1)
            hidden = torch.relu(hidden)

        out_index = len(self.hidden_sizes)
        logits = torch.einsum("sbi,sio->sbo", hidden, self.weights[out_index].index_select(0, active_idx))
        logits = logits + self.biases[out_index].index_select(0, active_idx).unsqueeze(1)
        return logits.squeeze(-1)

    def snapshot_seed(self, local_seed_index: int) -> SeedSnapshot:
        snapshot: SeedSnapshot = []
        for weight, bias in zip(self.weights, self.biases):
            snapshot.append((weight[local_seed_index].detach().cpu().clone(), bias[local_seed_index].detach().cpu().clone()))
        return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched-seed PyTorch GPU search using train/validation only.")
    parser.add_argument("--dataset", default="./dataset", help="dataset root with train/cat, train/dog, validation/cat and validation/dog")
    parser.add_argument("--runs", default="runs/gpu_search_pytorch_v1", help="output directory; compatible with gpu_search_pytorch.py")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--strategy", choices=("halving", "exhaustive"), default="halving", help="halving is recommended; exhaustive runs the full grid")
    parser.add_argument("--seed-batch", type=int, default=42, help="maximum number of seeds trained together for the same group")
    parser.add_argument("--workers", type=int, default=1, help="CPU thread count used by PyTorch")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="reuse completed rows from summary.csv")
    parser.add_argument("--deterministic", action="store_true", help="request deterministic PyTorch algorithms when possible")
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.seed_batch <= 0:
        raise SystemExit("--seed-batch must be positive")

    runs_dir = Path(args.runs)
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / "summary.csv"
    selected_path = runs_dir / "selected_groups.json"

    device = resolve_device(args.device)
    configure_torch(args.workers, args.deterministic, device)

    print(f"device={device} torch={torch.__version__} cuda_available={torch.cuda.is_available()} seed_batch={args.seed_batch}")
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
                cached, pending = split_cached_and_pending(group, stage, existing)
                for result in cached:
                    stage_results.append(result)
                    print_progress(group_index, len(current_groups), result)

                for seed_chunk in chunks(pending, args.seed_batch):
                    results = train_seed_chunk(group, seed_chunk, stage, runs_dir, train_x, train_y, val_x, val_y)
                    for result in results:
                        stage_results.append(result)
                        existing[result.run_id] = result
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


def split_cached_and_pending(group: GroupConfig, stage: Stage, existing: Dict[str, RunResult]) -> Tuple[List[RunResult], List[int]]:
    cached: List[RunResult] = []
    pending: List[int] = []
    for seed in stage.seeds:
        run_id = make_run_id(group, seed, stage)
        if run_id in existing and existing[run_id].completed:
            cached.append(dataclasses.replace(existing[run_id], cached=True))
        else:
            pending.append(seed)
    return cached, pending


def train_seed_chunk(
    group: GroupConfig,
    seeds: Sequence[int],
    stage: Stage,
    runs_dir: Path,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> List[RunResult]:
    if not seeds:
        return []
    try:
        return train_seed_chunk_checked(group, tuple(seeds), stage, runs_dir, train_x, train_y, val_x, val_y)
    except Exception as exc:
        return [failed_result(group, seed, stage, runs_dir, exc, train_x.shape[0]) for seed in seeds]


def train_seed_chunk_checked(
    group: GroupConfig,
    seeds: Tuple[int, ...],
    stage: Stage,
    runs_dir: Path,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> List[RunResult]:
    device = train_x.device
    set_run_seed(min(seeds), device)
    model = BatchedMLP(INPUT_SIZE, group.hidden, seeds, device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=group.learning_rate)
    batch = effective_batch_size(group.batch_size, train_x.shape[0])
    generators = make_generators(seeds, device)

    seed_count = len(seeds)
    active: List[int] = list(range(seed_count))
    best_epoch = [0 for _ in seeds]
    best_train: List[Optional[Metrics]] = [None for _ in seeds]
    best_val: List[Optional[Metrics]] = [None for _ in seeds]
    best_snapshot: List[Optional[SeedSnapshot]] = [None for _ in seeds]
    no_improvement = [0 for _ in seeds]
    stop_reason = ["max_epochs" for _ in seeds]
    epochs_run = [0 for _ in seeds]
    train_histories: List[List[float]] = [[] for _ in seeds]
    val_histories: List[List[float]] = [[] for _ in seeds]
    final_train: List[Optional[Metrics]] = [None for _ in seeds]
    final_val: List[Optional[Metrics]] = [None for _ in seeds]
    metrics_rows: Optional[List[List[List[str]]]] = [[[] for _ in range(0)] for _ in seeds] if stage.save_artifacts else None

    synchronize_if_cuda(device)
    started = time.perf_counter()

    for epoch in range(1, stage.max_epochs + 1):
        if not active:
            break
        model.train()
        active_idx = torch.tensor(active, device=device, dtype=torch.long)
        permutations = torch.stack([torch.randperm(train_x.shape[0], device=device, generator=generators[local]) for local in active], dim=0)

        for start in range(0, train_x.shape[0], batch):
            indices = permutations[:, start : start + batch]
            xb = train_x[indices]
            yb = train_y[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_active(xb, active_idx)
            per_seed_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none").mean(dim=1)
            loss = per_seed_loss.mean()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_batched(model, train_x, train_y, active_idx)
        val_metrics = evaluate_batched(model, val_x, val_y, active_idx)

        still_active: List[int] = []
        for position, local in enumerate(active):
            train_m = train_metrics[position]
            val_m = val_metrics[position]
            final_train[local] = train_m
            final_val[local] = val_m
            train_histories[local].append(train_m.loss)
            val_histories[local].append(val_m.loss)
            epochs_run[local] = epoch
            if metrics_rows is not None:
                metrics_rows[local].append(metrics_to_row(epoch, train_m, val_m))

            if is_better_validation(val_m, best_val[local]):
                best_epoch[local] = epoch
                best_train[local] = train_m
                best_val[local] = val_m
                best_snapshot[local] = model.snapshot_seed(local)
                no_improvement[local] = 0
            else:
                no_improvement[local] += 1

            reason = should_stop_seed(epoch, train_histories[local], val_histories[local], train_m, val_m, no_improvement[local])
            if reason is None:
                still_active.append(local)
            else:
                stop_reason[local] = reason

        active = still_active

    synchronize_if_cuda(device)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    results: List[RunResult] = []
    for local, seed in enumerate(seeds):
        if final_train[local] is None or final_val[local] is None:
            raise RuntimeError(f"seed {seed} did not produce metrics")
        if best_train[local] is None or best_val[local] is None or best_snapshot[local] is None:
            best_train[local] = final_train[local]
            best_val[local] = final_val[local]
            best_epoch[local] = epochs_run[local]
            best_snapshot[local] = model.snapshot_seed(local)

        run_id = make_run_id(group, seed, stage)
        run_dir = make_run_dir(runs_dir, group, seed, stage, run_id)
        if stage.save_artifacts:
            save_seed_artifacts(run_dir, group, seed, stage, run_id, batch, best_epoch[local], best_val[local], best_snapshot[local], metrics_rows[local] if metrics_rows else [])

        best_train_m = best_train[local]
        best_val_m = best_val[local]
        final_train_m = final_train[local]
        final_val_m = final_val[local]
        results.append(RunResult(
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
            epochs_run=epochs_run[local],
            stop_reason=stop_reason[local],
            best_epoch=best_epoch[local],
            best_train_loss=best_train_m.loss,
            best_train_accuracy=best_train_m.accuracy,
            best_train_precision=best_train_m.precision,
            best_train_recall=best_train_m.recall,
            best_train_f1=best_train_m.f1,
            best_val_loss=best_val_m.loss,
            best_val_accuracy=best_val_m.accuracy,
            best_val_precision=best_val_m.precision,
            best_val_recall=best_val_m.recall,
            best_val_f1=best_val_m.f1,
            final_train_loss=final_train_m.loss,
            final_train_accuracy=final_train_m.accuracy,
            final_val_loss=final_val_m.loss,
            final_val_accuracy=final_val_m.accuracy,
            generalization_gap=best_train_m.accuracy - best_val_m.accuracy,
            val_true_negative=best_val_m.true_negative,
            val_false_positive=best_val_m.false_positive,
            val_false_negative=best_val_m.false_negative,
            val_true_positive=best_val_m.true_positive,
            train_time_ms=elapsed_ms,
            run_directory=str(run_dir if stage.save_artifacts else ""),
        ))
    return results


def init_layer_for_seeds(in_features: int, out_features: int, seeds: Sequence[int], layer_index: int, nonlinearity: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    weights: List[torch.Tensor] = []
    biases: List[torch.Tensor] = []
    scale = math.sqrt((2.0 if nonlinearity == "relu" else 1.0) / float(in_features))
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for previous_layer in range(layer_index):
            fan_in, fan_out = layer_shape_for_index(in_features=None, hidden_placeholder=None, previous_layer=previous_layer)
            _ = fan_in
            _ = fan_out
        weight = torch.randn((in_features, out_features), generator=generator, dtype=torch.float32) * scale
        bias = torch.zeros((out_features,), dtype=torch.float32)
        weights.append(weight)
        biases.append(bias)
    return torch.stack(weights, dim=0).to(device), torch.stack(biases, dim=0).to(device)


def layer_shape_for_index(in_features: object, hidden_placeholder: object, previous_layer: int) -> Tuple[int, int]:
    return 1, 1


def make_generators(seeds: Sequence[int], device: torch.device) -> List[torch.Generator]:
    generators: List[torch.Generator] = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 1000)
        generators.append(generator)
    return generators


def set_run_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_batched(model: BatchedMLP, x: torch.Tensor, y: torch.Tensor, active_idx: torch.Tensor) -> List[Metrics]:
    model.eval()
    logits = model.forward_active(x, active_idx)
    y_expanded = y.unsqueeze(0).expand(logits.shape[0], -1)
    losses = F.binary_cross_entropy_with_logits(logits, y_expanded, reduction="none").mean(dim=1)
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= 0.5
    actual = y_expanded >= 0.5
    tp = (predicted & actual).sum(dim=1)
    tn = (~predicted & ~actual).sum(dim=1)
    fp = (predicted & ~actual).sum(dim=1)
    fn = (~predicted & actual).sum(dim=1)
    total = tp + tn + fp + fn
    accuracy = torch.where(total > 0, (tp + tn).float() / total.float(), torch.zeros_like(total, dtype=torch.float32))
    precision = torch.where(tp + fp > 0, tp.float() / (tp + fp).float(), torch.zeros_like(tp, dtype=torch.float32))
    recall = torch.where(tp + fn > 0, tp.float() / (tp + fn).float(), torch.zeros_like(tp, dtype=torch.float32))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))

    out: List[Metrics] = []
    for i in range(logits.shape[0]):
        out.append(Metrics(
            loss=float(losses[i].detach().cpu()),
            accuracy=float(accuracy[i].detach().cpu()),
            precision=float(precision[i].detach().cpu()),
            recall=float(recall[i].detach().cpu()),
            f1=float(f1[i].detach().cpu()),
            true_negative=int(tn[i].detach().cpu()),
            false_positive=int(fp[i].detach().cpu()),
            false_negative=int(fn[i].detach().cpu()),
            true_positive=int(tp[i].detach().cpu()),
        ))
    return out


def is_better_validation(candidate: Metrics, current: Optional[Metrics]) -> bool:
    if current is None:
        return True
    if candidate.accuracy != current.accuracy:
        return candidate.accuracy > current.accuracy
    return candidate.loss < current.loss - MIN_VALIDATION_DELTA


def should_stop_seed(epoch: int, train_losses: Sequence[float], val_losses: Sequence[float], train_metrics: Metrics, val_metrics: Metrics, no_improvement: int) -> Optional[str]:
    if not math.isfinite(train_metrics.loss) or not math.isfinite(val_metrics.loss):
        return "divergent_or_non_finite"
    if epoch >= MIN_EPOCHS and (train_metrics.loss > DIVERGENT_LOSS_LIMIT or val_metrics.loss > DIVERGENT_LOSS_LIMIT):
        return "divergent_or_non_finite"
    if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
        return "validation_patience"
    if epoch >= MIN_EPOCHS and has_low_learning(train_losses, val_losses, LOW_LEARNING_WINDOW, LOW_LEARNING_DELTA) and no_improvement >= LOW_LEARNING_WINDOW:
        return "low_learning"
    return None


def has_low_learning(train_losses: Sequence[float], val_losses: Sequence[float], window: int, min_delta: float) -> bool:
    if len(train_losses) <= window or len(val_losses) <= window:
        return False
    previous = len(train_losses) - 1 - window
    last = len(train_losses) - 1
    train_improvement = train_losses[previous] - train_losses[last]
    val_improvement = val_losses[previous] - val_losses[last]
    return train_improvement < min_delta and val_improvement < min_delta


def save_seed_artifacts(run_dir: Path, group: GroupConfig, seed: int, stage: Stage, run_id: str, effective_batch: int, best_epoch: int, best_val: Metrics, snapshot: SeedSnapshot, metrics_rows: Sequence[Sequence[str]]) -> None:
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
            "effective_batch_size": effective_batch,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val.loss,
            "best_validation_accuracy": best_val.accuracy,
            "state_dict": snapshot_to_single_state_dict(snapshot),
        },
        run_dir / "best.pt",
    )


def snapshot_to_single_state_dict(snapshot: SeedSnapshot) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    hidden_count = len(snapshot) - 1
    for layer_index, (weight, bias) in enumerate(snapshot[:hidden_count]):
        sequential_index = layer_index * 2
        state[f"hidden.{sequential_index}.weight"] = weight.t().contiguous()
        state[f"hidden.{sequential_index}.bias"] = bias.contiguous()
    out_weight, out_bias = snapshot[-1]
    state["output.weight"] = out_weight.t().contiguous()
    state["output.bias"] = out_bias.contiguous()
    return state


def failed_result(group: GroupConfig, seed: int, stage: Stage, runs_dir: Path, exc: Exception, train_size: int) -> RunResult:
    run_id = make_run_id(group, seed, stage)
    run_dir = make_run_dir(runs_dir, group, seed, stage, run_id)
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
        effective_batch_size=effective_batch_size(group.batch_size, train_size),
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


def chunks(values: Sequence[int], size: int) -> Iterable[Tuple[int, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


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


def make_run_dir(runs_dir: Path, group: GroupConfig, seed: int, stage: Stage, run_id: str) -> Path:
    return runs_dir / stage.name / f"{run_id}_{sanitize_name(group.hidden_label)}_lr{lr_label(group.learning_rate)}_bs{group.batch_size}_seed{seed}"


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


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Partial summary.csv was preserved.", file=sys.stderr)
        raise
