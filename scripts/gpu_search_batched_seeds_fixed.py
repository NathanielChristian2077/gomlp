#!/usr/bin/env python3
"""Fixed batched-seed GPU search runner.

The first batched runner averaged losses across independent seed models. Because
each seed owns a disjoint slice of the parameter tensors, averaging scaled each
seed gradient by 1/active_seed_count. This wrapper preserves the intended
learning rate by summing per-seed losses instead.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import gpu_search_batched_seeds as base


def train_seed_chunk_checked_fixed(group, seeds: Tuple[int, ...], stage, runs_dir, train_x, train_y, val_x, val_y):
    device = train_x.device
    base.set_run_seed(min(seeds), device)
    model = base.BatchedMLP(base.INPUT_SIZE, group.hidden, seeds, device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=group.learning_rate)
    batch = base.effective_batch_size(group.batch_size, train_x.shape[0])
    generators = base.make_generators(seeds, device)

    seed_count = len(seeds)
    active: List[int] = list(range(seed_count))
    best_epoch = [0 for _ in seeds]
    best_train: List[Optional[base.Metrics]] = [None for _ in seeds]
    best_val: List[Optional[base.Metrics]] = [None for _ in seeds]
    best_snapshot = [None for _ in seeds]
    no_improvement = [0 for _ in seeds]
    stop_reason = ["max_epochs" for _ in seeds]
    epochs_run = [0 for _ in seeds]
    train_histories: List[List[float]] = [[] for _ in seeds]
    val_histories: List[List[float]] = [[] for _ in seeds]
    final_train: List[Optional[base.Metrics]] = [None for _ in seeds]
    final_val: List[Optional[base.Metrics]] = [None for _ in seeds]
    metrics_rows = [[] for _ in seeds] if stage.save_artifacts else None

    base.synchronize_if_cuda(device)
    import time
    started = time.perf_counter()

    for epoch in range(1, stage.max_epochs + 1):
        if not active:
            break
        model.train()
        active_idx = torch.tensor(active, device=device, dtype=torch.long)
        permutations = torch.stack([
            torch.randperm(train_x.shape[0], device=device, generator=generators[local])
            for local in active
        ], dim=0)

        for start in range(0, train_x.shape[0], batch):
            indices = permutations[:, start : start + batch]
            xb = train_x[indices]
            yb = train_y[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_active(xb, active_idx)
            per_seed_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none").mean(dim=1)
            # Critical difference from gpu_search_batched_seeds.py: each seed has
            # independent parameters, so summing preserves the per-seed gradient
            # scale of standalone training. Averaging would silently turn lr into
            # lr / active_seed_count.
            per_seed_loss.sum().backward()
            optimizer.step()

        train_metrics = base.evaluate_batched(model, train_x, train_y, active_idx)
        val_metrics = base.evaluate_batched(model, val_x, val_y, active_idx)

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
                metrics_rows[local].append(base.metrics_to_row(epoch, train_m, val_m))

            if base.is_better_validation(val_m, best_val[local]):
                best_epoch[local] = epoch
                best_train[local] = train_m
                best_val[local] = val_m
                best_snapshot[local] = model.snapshot_seed(local)
                no_improvement[local] = 0
            else:
                no_improvement[local] += 1

            reason = base.should_stop_seed(epoch, train_histories[local], val_histories[local], train_m, val_m, no_improvement[local])
            if reason is None:
                still_active.append(local)
            else:
                stop_reason[local] = reason
        active = still_active

    base.synchronize_if_cuda(device)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    amortized_ms = max(1, elapsed_ms // max(1, seed_count))

    results = []
    for local, seed in enumerate(seeds):
        if final_train[local] is None or final_val[local] is None:
            raise RuntimeError(f"seed {seed} did not produce metrics")
        if best_train[local] is None or best_val[local] is None or best_snapshot[local] is None:
            best_train[local] = final_train[local]
            best_val[local] = final_val[local]
            best_epoch[local] = epochs_run[local]
            best_snapshot[local] = model.snapshot_seed(local)

        run_id = base.make_run_id(group, seed, stage)
        run_dir = base.make_run_dir(runs_dir, group, seed, stage, run_id)
        if stage.save_artifacts:
            base.save_seed_artifacts(
                run_dir, group, seed, stage, run_id, batch, best_epoch[local],
                best_val[local], best_snapshot[local], metrics_rows[local] if metrics_rows else [],
            )

        best_train_m = best_train[local]
        best_val_m = best_val[local]
        final_train_m = final_train[local]
        final_val_m = final_val[local]
        results.append(base.RunResult(
            run_id=run_id,
            stage=stage.name,
            completed=True,
            cached=False,
            hidden=group.hidden_label,
            depth=len(group.hidden),
            parameter_count=base.parameter_count(base.INPUT_SIZE, group.hidden),
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
            train_time_ms=amortized_ms,
            run_directory=str(run_dir if stage.save_artifacts else ""),
        ))
    return results


base.train_seed_chunk_checked = train_seed_chunk_checked_fixed


if __name__ == "__main__":
    base.main()
