#!/usr/bin/env python3
"""Validation-only ensemble probe for saved PyTorch MLP checkpoints.

This is intentionally a side experiment. It reads completed runs from one or more
summary.csv files, loads their best.pt checkpoints, and evaluates probability
averaging on the validation split by default. Do not use --split test until the
individual model selection is frozen.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from gpu_search_pytorch import MLP, INPUT_SIZE, Metrics, configure_torch, load_split, resolve_device


@dataclass(frozen=True)
class Candidate:
    run_id: str
    hidden: str
    learning_rate: float
    batch_size: int
    seed: int
    best_val_accuracy: float
    best_val_f1: float
    best_val_loss: float
    generalization_gap: float
    run_directory: Path
    checkpoint_path: Path

    @property
    def group_key(self) -> Tuple[str, float, int]:
        return (self.hidden, self.learning_rate, self.batch_size)

    @property
    def individual_score(self) -> float:
        return self.best_val_accuracy + 0.50 * self.best_val_f1 - 0.25 * abs(self.generalization_gap) - 0.10 * self.best_val_loss


@dataclass(frozen=True)
class EnsembleResult:
    name: str
    source: str
    size: int
    members: str
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    mean_member_acc: float
    mean_member_f1: float
    best_member_acc: float
    best_member_f1: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe simple probability ensembles from saved MLP checkpoints.")
    parser.add_argument("--dataset", default="./dataset", help="dataset root")
    parser.add_argument("--summary", action="append", default=[], help="summary.csv path; can be repeated")
    parser.add_argument("--runs", default="runs/mlp_tournament_v1", help="default run root used when --summary is omitted")
    parser.add_argument("--output", default="", help="output CSV path; default is <runs>/ensemble_summary.csv")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="CPU thread count used by PyTorch")
    parser.add_argument("--split", default="validation", choices=("validation", "test"), help="split to evaluate; keep validation until final selection is frozen")
    parser.add_argument("--top-k", default="3,5,7,11", help="ensemble sizes for global and per-group top-k")
    parser.add_argument("--max-groups", type=int, default=8, help="number of groups to try for per-group ensembles")
    parser.add_argument("--deterministic", action="store_true", help="request deterministic PyTorch algorithms when possible")
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_groups <= 0:
        raise SystemExit("--max-groups must be positive")

    runs_dir = Path(args.runs)
    summary_paths = [Path(p) for p in args.summary] if args.summary else [runs_dir / "summary.csv"]
    output_path = Path(args.output) if args.output else runs_dir / "ensemble_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_k_values = parse_top_k(args.top_k)

    device = resolve_device(args.device)
    configure_torch(args.workers, args.deterministic, device)
    print(f"device={device}")
    print(f"split={args.split}")
    print("summaries=" + ",".join(str(p) for p in summary_paths))

    x, y = load_split(Path(args.dataset), args.split, device)
    print(f"{args.split}={x.shape[0]} input={x.shape[1]}")

    candidates = load_candidates(summary_paths)
    if not candidates:
        raise SystemExit("no completed candidates with existing best.pt checkpoints found")
    print(f"checkpoints={len(candidates)}")

    ensembles = build_ensemble_specs(candidates, top_k_values, args.max_groups)
    results: List[EnsembleResult] = []
    for index, (name, source, members) in enumerate(ensembles, start=1):
        print(f"[{index}/{len(ensembles)}] {name} size={len(members)}")
        results.append(evaluate_ensemble(name, source, members, x, y, device))

    results.sort(key=lambda r: (-r.accuracy, -r.f1, r.loss, r.size))
    write_results(output_path, results)
    print_results(results)
    print(f"ensemble_summary={output_path}")


def load_candidates(summary_paths: Sequence[Path]) -> List[Candidate]:
    by_run_id: Dict[str, Candidate] = {}
    for summary_path in summary_paths:
        if not summary_path.exists():
            print(f"warning: missing summary {summary_path}")
            continue
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if not parse_bool(row.get("completed", "false")):
                    continue
                run_directory = Path(row.get("run_directory", ""))
                if not run_directory:
                    continue
                checkpoint_path = run_directory / "best.pt"
                if not checkpoint_path.exists():
                    continue
                run_id = row["run_id"]
                by_run_id[run_id] = Candidate(
                    run_id=run_id,
                    hidden=row["hidden"],
                    learning_rate=float(row["learning_rate"]),
                    batch_size=int(row["batch_size"]),
                    seed=int(row["seed"]),
                    best_val_accuracy=float(row["best_val_accuracy"]),
                    best_val_f1=float(row["best_val_f1"]),
                    best_val_loss=float(row["best_val_loss"]),
                    generalization_gap=float(row["generalization_gap"]),
                    run_directory=run_directory,
                    checkpoint_path=checkpoint_path,
                )
    return list(by_run_id.values())


def build_ensemble_specs(candidates: Sequence[Candidate], top_k_values: Sequence[int], max_groups: int) -> List[Tuple[str, str, List[Candidate]]]:
    specs: List[Tuple[str, str, List[Candidate]]] = []
    global_by_score = sorted(candidates, key=lambda c: (-c.individual_score, -c.best_val_accuracy, -c.best_val_f1, c.best_val_loss))
    global_by_acc = sorted(candidates, key=lambda c: (-c.best_val_accuracy, -c.best_val_f1, c.best_val_loss))

    for k in top_k_values:
        if len(global_by_score) >= k:
            specs.append((f"global_score_top{k}", "global_score", global_by_score[:k]))
        if len(global_by_acc) >= k:
            specs.append((f"global_acc_top{k}", "global_acc", global_by_acc[:k]))

    grouped: Dict[Tuple[str, float, int], List[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.group_key, []).append(candidate)

    group_rank = []
    for key, members in grouped.items():
        if len(members) < min(top_k_values):
            continue
        score = mean(c.individual_score for c in members)
        acc = mean(c.best_val_accuracy for c in members)
        f1 = mean(c.best_val_f1 for c in members)
        group_rank.append((score, acc, f1, key, members))
    group_rank.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))

    for _, _, _, key, members in group_rank[:max_groups]:
        hidden, lr, batch = key
        ranked = sorted(members, key=lambda c: (-c.individual_score, -c.best_val_accuracy, -c.best_val_f1, c.best_val_loss))
        for k in top_k_values:
            if len(ranked) >= k:
                name = f"group_{sanitize(hidden)}_lr{lr_label(lr)}_b{batch}_top{k}"
                specs.append((name, "per_group_score", ranked[:k]))

    deduped: List[Tuple[str, str, List[Candidate]]] = []
    seen = set()
    for name, source, members in specs:
        ids = tuple(c.run_id for c in members)
        if ids in seen:
            continue
        seen.add(ids)
        deduped.append((name, source, members))
    return deduped


def evaluate_ensemble(name: str, source: str, members: Sequence[Candidate], x: torch.Tensor, y: torch.Tensor, device: torch.device) -> EnsembleResult:
    probabilities = []
    for member in members:
        checkpoint = torch.load(member.checkpoint_path, map_location=device)
        hidden_sizes = tuple(int(v) for v in checkpoint["hidden_sizes"])
        model = MLP(INPUT_SIZE, hidden_sizes).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probabilities.append(torch.sigmoid(logits))
    mean_probability = torch.stack(probabilities, dim=0).mean(dim=0)
    metrics = metrics_from_probability(mean_probability, y)
    return EnsembleResult(
        name=name,
        source=source,
        size=len(members),
        members=";".join(f"{m.hidden}|lr={m.learning_rate:g}|b={m.batch_size}|seed={m.seed}|run={m.run_id}" for m in members),
        loss=metrics.loss,
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        true_negative=metrics.true_negative,
        false_positive=metrics.false_positive,
        false_negative=metrics.false_negative,
        true_positive=metrics.true_positive,
        mean_member_acc=mean(m.best_val_accuracy for m in members),
        mean_member_f1=mean(m.best_val_f1 for m in members),
        best_member_acc=max(m.best_val_accuracy for m in members),
        best_member_f1=max(m.best_val_f1 for m in members),
    )


def metrics_from_probability(probability: torch.Tensor, y: torch.Tensor) -> Metrics:
    eps = 1e-7
    p = probability.clamp(eps, 1.0 - eps)
    loss = float((-(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))).mean().detach().cpu())
    predicted = p >= 0.5
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


def parse_top_k(value: str) -> List[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k <= 0:
            raise SystemExit("top-k values must be positive")
        out.append(k)
    return sorted(set(out))


def write_results(path: Path, results: Sequence[EnsembleResult]) -> None:
    fields = [
        "rank", "name", "source", "size", "loss", "accuracy", "precision", "recall", "f1",
        "true_negative", "false_positive", "false_negative", "true_positive", "mean_member_acc",
        "mean_member_f1", "best_member_acc", "best_member_f1", "members",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = {field: getattr(result, field) for field in fields if field != "rank"}
            row["rank"] = rank
            writer.writerow(row)


def print_results(results: Sequence[EnsembleResult]) -> None:
    print("\nTOP ensembles")
    print("rank,name,size,accuracy,f1,loss,tn,fp,fn,tp,mean_member_acc,best_member_acc")
    for rank, result in enumerate(results[:20], start=1):
        print(
            f"{rank},{result.name},{result.size},{result.accuracy:.6f},{result.f1:.6f},{result.loss:.6f},"
            f"{result.true_negative},{result.false_positive},{result.false_negative},{result.true_positive},"
            f"{result.mean_member_acc:.6f},{result.best_member_acc:.6f}"
        )


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def lr_label(lr: float) -> str:
    return f"{lr:g}".replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
