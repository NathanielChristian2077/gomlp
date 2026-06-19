#!/usr/bin/env python3
"""Summarize top-3 Go dense/DSA comparison outputs.

Input layout is produced by scripts/run_top3_go_dsa.sh. The script combines raw
compare/bench CSV files and writes small tables plus a Markdown report snippet.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ManifestRow:
    rank: int
    label: str
    hidden: str
    learning_rate: str
    batch_size: str
    seed: str
    gpu_score: str
    gpu_acc_mean: str
    gpu_f1_mean: str
    gpu_best_seed_acc: str
    gpu_best_seed_f1: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Go top-3 dense/DSA outputs.")
    parser.add_argument("--root", default="runs/go_top3_dsa", help="root produced by run_top3_go_dsa.sh")
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = root / "top3_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")

    manifest = read_manifest(manifest_path)
    compare_rows = collect_compare_rows(root, manifest)
    bench_rows = collect_bench_rows(root, manifest)

    compare_combined = root / "top3_compare_combined.csv"
    bench_combined = root / "top3_bench_combined.csv"
    dense_summary = root / "top3_dense_summary.csv"
    dsa_summary = root / "top3_dsa_summary.csv"
    bench_summary = root / "top3_bench_summary.csv"
    report_path = root / "top3_report_snippet.md"

    write_rows(compare_combined, compare_rows)
    write_rows(bench_combined, bench_rows)

    dense_rows = build_dense_summary(compare_rows)
    dsa_rows = build_dsa_summary(compare_rows)
    bench_summary_rows = build_bench_summary(bench_rows)

    write_rows(dense_summary, dense_rows)
    write_rows(dsa_summary, dsa_rows)
    write_rows(bench_summary, bench_summary_rows)
    report_path.write_text(build_report(manifest, dense_rows, dsa_rows, bench_summary_rows), encoding="utf-8")

    print(f"compare_rows={len(compare_rows)} -> {compare_combined}")
    print(f"bench_rows={len(bench_rows)} -> {bench_combined}")
    print(f"dense_summary={dense_summary}")
    print(f"dsa_summary={dsa_summary}")
    print(f"bench_summary={bench_summary}")
    print(f"report_snippet={report_path}")


def read_manifest(path: Path) -> List[ManifestRow]:
    out: List[ManifestRow] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            out.append(ManifestRow(
                rank=int(row["rank"]),
                label=row["label"],
                hidden=row["hidden"],
                learning_rate=row["learning_rate"],
                batch_size=row["batch_size"],
                seed=row["seed"],
                gpu_score=row["gpu_score"],
                gpu_acc_mean=row["gpu_acc_mean"],
                gpu_f1_mean=row["gpu_f1_mean"],
                gpu_best_seed_acc=row["gpu_best_seed_acc"],
                gpu_best_seed_f1=row["gpu_best_seed_f1"],
            ))
    return out


def collect_compare_rows(root: Path, manifest: Sequence[ManifestRow]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in manifest:
        candidate_dir = root / item.label
        for split in ("validation", "test"):
            path = candidate_dir / f"compare_{split}.csv"
            if not path.exists():
                print(f"warning: missing compare file {path}")
                continue
            for row in read_csv_dicts(path):
                rows.append(with_manifest(row, item))
    return rows


def collect_bench_rows(root: Path, manifest: Sequence[ManifestRow]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in manifest:
        path = root / item.label / "bench_test.csv"
        if not path.exists():
            print(f"warning: missing bench file {path}")
            continue
        for row in read_csv_dicts(path):
            rows.append(with_manifest(row, item))
    return rows


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def with_manifest(row: Dict[str, str], item: ManifestRow) -> Dict[str, str]:
    out = {
        "rank": str(item.rank),
        "label": item.label,
        "candidate_hidden": item.hidden,
        "candidate_learning_rate": item.learning_rate,
        "candidate_batch_size": item.batch_size,
        "candidate_seed": item.seed,
        "gpu_score": item.gpu_score,
        "gpu_acc_mean": item.gpu_acc_mean,
        "gpu_f1_mean": item.gpu_f1_mean,
        "gpu_best_seed_acc": item.gpu_best_seed_acc,
        "gpu_best_seed_f1": item.gpu_best_seed_f1,
    }
    out.update(row)
    return out


def build_dense_summary(compare_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in compare_rows:
        if row.get("mode") != "dense":
            continue
        out.append(select_fields(row, [
            "rank", "label", "candidate_hidden", "candidate_learning_rate", "candidate_batch_size", "candidate_seed",
            "split", "best_epoch", "loss", "accuracy", "precision", "recall", "f1",
            "true_negative", "false_positive", "false_negative", "true_positive",
        ]))
    return sort_rows(out, ("rank", "split"))


def build_dsa_summary(compare_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    dense_by_candidate_split: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in compare_rows:
        if row.get("mode") == "dense":
            dense_by_candidate_split[(row["label"], row["split"])] = row

    out: List[Dict[str, str]] = []
    for row in compare_rows:
        if row.get("mode") == "dense":
            continue
        dense = dense_by_candidate_split.get((row["label"], row["split"]))
        dense_accuracy = as_float(dense.get("accuracy")) if dense else 0.0
        dense_f1 = as_float(dense.get("f1")) if dense else 0.0
        sparse_ops = as_float(row.get("sparse_ops_total"))
        dense_ops = as_float(row.get("dense_ops_total"))
        ops_saved = 0.0 if dense_ops <= 0 else 1.0 - sparse_ops / dense_ops
        out.append({
            "rank": row["rank"],
            "label": row["label"],
            "candidate_hidden": row["candidate_hidden"],
            "candidate_learning_rate": row["candidate_learning_rate"],
            "candidate_batch_size": row["candidate_batch_size"],
            "candidate_seed": row["candidate_seed"],
            "split": row["split"],
            "mode": row["mode"],
            "threshold": row["threshold"],
            "accuracy": row["accuracy"],
            "accuracy_delta_from_dense": fmt(as_float(row.get("accuracy")) - dense_accuracy),
            "f1": row["f1"],
            "f1_delta_from_dense": fmt(as_float(row.get("f1")) - dense_f1),
            "avg_sparsity": row["avg_sparsity"],
            "avg_active_ratio": row["avg_active_ratio"],
            "estimated_speedup": row["estimated_speedup"],
            "ops_saved_ratio": fmt(ops_saved),
            "max_abs_diff_from_dense": row["max_abs_diff_from_dense"],
            "mismatch_count_from_dense": row["mismatch_count_from_dense"],
            "avg_active_by_layer": row["avg_active_by_layer"],
        })
    return sort_rows(out, ("rank", "split", "threshold"))


def build_bench_summary(bench_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    dense_by_label: Dict[str, Dict[str, str]] = {}
    for row in bench_rows:
        if row.get("mode") == "dense":
            dense_by_label[row["label"]] = row

    out: List[Dict[str, str]] = []
    for row in bench_rows:
        dense = dense_by_label.get(row["label"])
        dense_ns = as_float(dense.get("ns_per_forward")) if dense else 0.0
        row_ns = as_float(row.get("ns_per_forward"))
        real_speedup = 0.0 if row_ns <= 0 else dense_ns / row_ns
        out.append({
            "rank": row["rank"],
            "label": row["label"],
            "candidate_hidden": row["candidate_hidden"],
            "candidate_learning_rate": row["candidate_learning_rate"],
            "candidate_batch_size": row["candidate_batch_size"],
            "candidate_seed": row["candidate_seed"],
            "split": row["split"],
            "mode": row["mode"],
            "threshold": row["threshold"],
            "repeats": row["repeats"],
            "ns_per_forward": row["ns_per_forward"],
            "real_speedup_vs_dense": fmt(real_speedup),
            "estimated_speedup": row["estimated_speedup"],
            "ops_saved_ratio": row["ops_saved_ratio"],
            "avg_sparsity": row["avg_sparsity"],
            "avg_active_by_layer": row["avg_active_by_layer"],
        })
    return sort_rows(out, ("rank", "mode", "threshold"))


def build_report(manifest: Sequence[ManifestRow], dense_rows: Sequence[Dict[str, str]], dsa_rows: Sequence[Dict[str, str]], bench_rows: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("# Resultados Top-3 MLP Go + DSA")
    lines.append("")
    lines.append("## Candidatos congelados")
    lines.append("")
    lines.append("| Rank | Arquitetura | LR | Batch | Seed | Score GPU | Val acc média GPU | Val F1 médio GPU |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for item in manifest:
        lines.append(f"| {item.rank} | `{item.hidden}` | {item.learning_rate} | {item.batch_size} | {item.seed} | {item.gpu_score} | {item.gpu_acc_mean} | {item.gpu_f1_mean} |")

    lines.append("")
    lines.append("## Métricas densas no Go")
    lines.append("")
    lines.append("| Rank | Arquitetura | Split | Acc | F1 | Loss | Matriz TN/FP/FN/TP |")
    lines.append("|---:|---|---|---:|---:|---:|---|")
    for row in dense_rows:
        matrix = f"{row['true_negative']}/{row['false_positive']}/{row['false_negative']}/{row['true_positive']}"
        lines.append(f"| {row['rank']} | `{row['candidate_hidden']}` | {row['split']} | {short(row['accuracy'])} | {short(row['f1'])} | {short(row['loss'])} | {matrix} |")

    exact_rows = [row for row in dsa_rows if row.get("threshold") in {"0", "0.00000000"}]
    if exact_rows:
        lines.append("")
        lines.append("## DSA exacta, threshold = 0")
        lines.append("")
        lines.append("| Rank | Split | Acc delta | F1 delta | Sparsity | Speedup estimado | Ops salvas | Mismatch | Max diff |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in exact_rows:
            lines.append(
                f"| {row['rank']} | {row['split']} | {short(row['accuracy_delta_from_dense'])} | {short(row['f1_delta_from_dense'])} | "
                f"{short(row['avg_sparsity'])} | {short(row['estimated_speedup'])} | {short(row['ops_saved_ratio'])} | "
                f"{row['mismatch_count_from_dense']} | {short(row['max_abs_diff_from_dense'])} |"
            )

    best_threshold_rows = best_threshold_by_candidate(dsa_rows)
    if best_threshold_rows:
        lines.append("")
        lines.append("## Melhor threshold por candidato no teste")
        lines.append("")
        lines.append("| Rank | Threshold | Acc | F1 | Sparsity | Mismatch | Speedup estimado |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in best_threshold_rows:
            lines.append(f"| {row['rank']} | {short(row['threshold'])} | {short(row['accuracy'])} | {short(row['f1'])} | {short(row['avg_sparsity'])} | {row['mismatch_count_from_dense']} | {short(row['estimated_speedup'])} |")

    if bench_rows:
        exact_bench = [row for row in bench_rows if row.get("threshold") in {"0", "0.00000000"}]
        lines.append("")
        lines.append("## Benchmark de inferência, teste")
        lines.append("")
        lines.append("| Rank | Modo | Threshold | ns/forward | Speedup real vs dense | Sparsity |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for row in exact_bench:
            lines.append(f"| {row['rank']} | {row['mode']} | {short(row['threshold'])} | {short(row['ns_per_forward'])} | {short(row['real_speedup_vs_dense'])} | {short(row['avg_sparsity'])} |")

    lines.append("")
    lines.append("## Observações para o relatório")
    lines.append("")
    lines.append("A DSA exata usa threshold zero e, portanto, deve preservar as decisões do modelo denso. A evidência esperada é `mismatch_count_from_dense = 0` e `max_abs_diff_from_dense` próximo de zero. Thresholds maiores que zero passam a ser aproximações: podem aumentar a esparsidade, mas precisam ser discutidos separadamente porque podem alterar predições.")
    lines.append("")
    return "\n".join(lines)


def best_threshold_by_candidate(dsa_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in dsa_rows:
        if row.get("split") != "test":
            continue
        grouped.setdefault(row["label"], []).append(row)
    out: List[Dict[str, str]] = []
    for rows in grouped.values():
        rows = sorted(rows, key=lambda r: (-as_float(r.get("f1")), -as_float(r.get("accuracy")), as_float(r.get("threshold"))))
        out.append(rows[0])
    return sort_rows(out, ("rank",))


def select_fields(row: Dict[str, str], fields: Sequence[str]) -> Dict[str, str]:
    return {field: row.get(field, "") for field in fields}


def sort_rows(rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> List[Dict[str, str]]:
    def key(row: Dict[str, str]) -> Tuple[object, ...]:
        out: List[object] = []
        for field in fields:
            value = row.get(field, "")
            try:
                out.append(float(value))
            except ValueError:
                out.append(value)
        return tuple(out)
    return sorted(rows, key=key)


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for field in row.keys():
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value))
    except ValueError:
        return 0.0


def fmt(value: float) -> str:
    return f"{value:.8f}"


def short(value: object) -> str:
    value_f = as_float(value)
    return f"{value_f:.4f}"


if __name__ == "__main__":
    main()
