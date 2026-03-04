#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import itertools
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None


METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]

DEFAULT_CLASS_NAMES = {
    4: ["up", "down", "left", "right"],
    8: ["up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare model performance across experiments.")
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("/work/SSR/share/code/MultiView_DriverAction_PyTorch/logs/train"),
        help="Root folder that contains experiment folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/work/SSR/share/code/MultiView_DriverAction_PyTorch/logs/eval_results"),
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--target-folds",
        type=int,
        default=5,
        help="Prefer runs with at least this many valid folds.",
    )
    parser.add_argument(
        "--allow-incomplete-folds",
        action="store_true",
        help="Fallback to best available run if no run reaches target-folds.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment names to evaluate. Default: evaluate all under train-root.",
    )
    parser.add_argument(
        "--reference-experiment",
        type=str,
        default=None,
        help="Reference experiment for significance test. Default: best f1_macro_mean experiment.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names, e.g. --class-names safe normal risky",
    )
    return parser.parse_args()


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0, mean, mean
    std = float(arr.std(ddof=1))
    margin = 1.96 * std / np.sqrt(arr.size)
    return mean, std, float(mean - margin), float(mean + margin)


def paired_permutation_pvalue(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired test requires equal-length vectors")

    diff = a_arr - b_arr
    obs = abs(float(diff.mean()))
    n = diff.size
    if n == 0:
        return 1.0

    count = 0
    total = 0
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        s = np.asarray(signs, dtype=np.float64)
        val = abs(float((diff * s).mean()))
        if val >= obs - 1e-12:
            count += 1
        total += 1
    return float((count + 1) / (total + 1))


def count_valid_pairs(best_preds_dir: Path) -> int:
    pred_files = sorted(best_preds_dir.glob("*_pred.pt"))
    valid = 0
    for pred_file in pred_files:
        fold = pred_file.stem.replace("_pred", "")
        if (best_preds_dir / f"{fold}_label.pt").exists():
            valid += 1
    return valid


def find_preferred_run(experiment_dir: Path, target_folds: int, allow_incomplete_folds: bool) -> Path | None:
    full_runs: list[Path] = []
    partial_runs: list[tuple[int, Path]] = []

    for date_dir in sorted([p for p in experiment_dir.iterdir() if p.is_dir()]):
        for time_dir in sorted([p for p in date_dir.iterdir() if p.is_dir()]):
            best_preds = time_dir / "best_preds"
            if not best_preds.exists():
                continue
            n_valid = count_valid_pairs(best_preds)
            if n_valid <= 0:
                continue
            if n_valid >= target_folds:
                full_runs.append(time_dir)
            else:
                partial_runs.append((n_valid, time_dir))

    if full_runs:
        return sorted(full_runs)[-1]

    if not allow_incomplete_folds:
        return None

    if not partial_runs:
        return None

    partial_runs.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return partial_runs[0][1]


def load_fold_predictions(best_preds_dir: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    fold_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_class_idx = -1

    for pred_file in sorted(best_preds_dir.glob("*_pred.pt")):
        fold = pred_file.stem.replace("_pred", "")
        label_file = best_preds_dir / f"{fold}_label.pt"
        if not label_file.exists():
            continue

        pred = torch.load(pred_file, map_location="cpu")
        label = torch.load(label_file, map_location="cpu")

        if isinstance(pred, list):
            pred = torch.cat(pred, dim=0)
        if isinstance(label, list):
            label = torch.cat(label, dim=0)

        pred = pred.detach().cpu()
        label = label.detach().cpu().long().view(-1)

        if pred.ndim > 1:
            pred_cls = pred.argmax(dim=1).long().view(-1)
            n_class = int(pred.shape[1])
        else:
            pred_cls = pred.long().view(-1)
            n_class = int(max(pred_cls.max().item(), label.max().item()) + 1)

        max_class_idx = max(max_class_idx, n_class - 1)
        fold_data[fold] = (label.numpy(), pred_cls.numpy())

    if not fold_data:
        raise ValueError(f"No valid pred/label fold pairs found in {best_preds_dir}")

    all_labels = np.concatenate([v[0] for v in fold_data.values()])
    all_preds = np.concatenate([v[1] for v in fold_data.values()])
    num_classes = max(max_class_idx + 1, int(max(all_labels.max(), all_preds.max()) + 1))
    return fold_data, num_classes


def compute_scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_weight),
        "recall_weighted": float(r_weight),
        "f1_weighted": float(f_weight),
    }


def save_confusion_png(cm: np.ndarray, class_names: list[str], title: str, save_path: Path, fmt: str) -> None:
    if plt is None or sns is None:
        return
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def resolve_class_names(num_classes: int, class_names_arg: list[str] | None) -> list[str]:
    if class_names_arg:
        if len(class_names_arg) == num_classes:
            return class_names_arg
        print(
            f"[warn] --class-names length ({len(class_names_arg)}) != num_classes ({num_classes}), "
            "falling back to defaults."
        )

    if num_classes in DEFAULT_CLASS_NAMES:
        return DEFAULT_CLASS_NAMES[num_classes]

    return [f"class_{i}" for i in range(num_classes)]


def evaluate_experiment(experiment_name: str, run_dir: Path, output_root: Path, class_names_arg: list[str] | None) -> dict:
    best_preds_dir = run_dir / "best_preds"
    fold_data, num_classes = load_fold_predictions(best_preds_dir)
    fold_ids = sorted(fold_data.keys(), key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)

    y_true = np.concatenate([fold_data[f][0] for f in fold_ids], axis=0)
    y_pred = np.concatenate([fold_data[f][1] for f in fold_ids], axis=0)

    class_names = resolve_class_names(num_classes, class_names_arg)

    # overall metrics
    overall = compute_scalar_metrics(y_true, y_pred)

    # per-class metrics
    labels = list(range(num_classes))
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    # fold metrics
    fold_metric_rows: list[list] = []
    fold_metric_by_key: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for fold in fold_ids:
        fy, fp = fold_data[fold]
        fm = compute_scalar_metrics(fy, fp)
        fold_metric_rows.append([
            fold,
            len(fy),
            fm["accuracy"],
            fm["balanced_accuracy"],
            fm["precision_macro"],
            fm["recall_macro"],
            fm["f1_macro"],
            fm["precision_weighted"],
            fm["recall_weighted"],
            fm["f1_weighted"],
        ])
        for k in METRIC_KEYS:
            fold_metric_by_key[k].append(float(fm[k]))

    exp_out = output_root / experiment_name
    exp_out.mkdir(parents=True, exist_ok=True)

    # save per-class
    per_class_rows = []
    for i in range(num_classes):
        per_class_rows.append([class_names[i], p_cls[i], r_cls[i], f_cls[i], int(support[i])])
    write_csv(exp_out / "per_class_metrics.csv", ["class", "precision", "recall", "f1", "support"], per_class_rows)

    # save confusion matrix csv
    write_csv(
        exp_out / "confusion_matrix_counts.csv",
        ["actual\\pred", *class_names],
        [[class_names[i], *cm[i].tolist()] for i in range(num_classes)],
    )
    write_csv(
        exp_out / "confusion_matrix_norm.csv",
        ["actual\\pred", *class_names],
        [[class_names[i], *cm_norm[i].tolist()] for i in range(num_classes)],
    )

    save_confusion_png(cm, class_names, f"{experiment_name} Confusion Matrix (Count)", exp_out / "confusion_matrix_counts.png", "d")
    save_confusion_png(cm_norm * 100.0, class_names, f"{experiment_name} Confusion Matrix (%)", exp_out / "confusion_matrix_percent.png", ".2f")

    # save fold metrics
    write_csv(
        exp_out / "per_fold_metrics.csv",
        [
            "fold",
            "num_samples",
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ],
        fold_metric_rows,
    )

    # fold ci summary
    ci_rows = []
    fold_metric_by_id: dict[str, dict[str, float]] = {}
    summary = {
        "experiment": experiment_name,
        "run_dir": str(run_dir),
        "num_classes": int(num_classes),
        "num_samples": int(len(y_true)),
        "folds": fold_ids,
        **overall,
    }
    for k in METRIC_KEYS:
        mean, std, lo, hi = ci95(fold_metric_by_key[k])
        ci_rows.append([k, mean, std, lo, hi])
        summary[f"{k}_mean"] = mean
        summary[f"{k}_std"] = std
        summary[f"{k}_ci95_low"] = lo
        summary[f"{k}_ci95_high"] = hi

    for row in fold_metric_rows:
        fold_id = str(row[0])
        fold_metric_by_id[fold_id] = {
            "accuracy": float(row[2]),
            "balanced_accuracy": float(row[3]),
            "precision_macro": float(row[4]),
            "recall_macro": float(row[5]),
            "f1_macro": float(row[6]),
            "precision_weighted": float(row[7]),
            "recall_weighted": float(row[8]),
            "f1_weighted": float(row[9]),
        }
    summary["fold_metric_by_id"] = fold_metric_by_id

    write_csv(exp_out / "fold_ci95_metrics.csv", ["metric", "mean", "std", "ci95_low", "ci95_high"], ci_rows)

    with (exp_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "per_class": {
                    "class_names": class_names,
                    "precision": p_cls.tolist(),
                    "recall": r_cls.tolist(),
                    "f1": f_cls.tolist(),
                    "support": support.tolist(),
                },
                "confusion_matrix": cm.tolist(),
                "confusion_matrix_norm": cm_norm.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return summary


def main() -> None:
    args = parse_args()
    train_root = args.train_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if not train_root.exists():
        raise FileNotFoundError(f"train root not found: {train_root}")

    if args.experiments:
        experiment_dirs = [train_root / name for name in args.experiments]
    else:
        experiment_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])

    all_summaries: list[dict] = []

    for exp_dir in experiment_dirs:
        if not exp_dir.exists() or not exp_dir.is_dir():
            continue
        run_dir = find_preferred_run(exp_dir, args.target_folds, args.allow_incomplete_folds)
        if run_dir is None:
            print(f"[skip] {exp_dir.name}: no run with >= {args.target_folds} valid folds")
            continue

        print(f"[eval] {exp_dir.name} -> {run_dir}")
        summary = evaluate_experiment(exp_dir.name, run_dir, output_root, args.class_names)
        all_summaries.append(summary)

    if not all_summaries:
        raise RuntimeError("No experiments evaluated. Check train-root and fold availability.")

    all_summaries = sorted(all_summaries, key=lambda x: (x["f1_macro_mean"], x["accuracy_mean"]), reverse=True)

    reference_name = args.reference_experiment if args.reference_experiment else all_summaries[0]["experiment"]
    reference = None
    for row in all_summaries:
        if row["experiment"] == reference_name:
            reference = row
            break
    if reference is None:
        raise ValueError(f"reference experiment not found: {reference_name}")

    sig_rows: list[list] = []
    sig_json: list[dict] = []
    for row in all_summaries:
        if row["experiment"] == reference_name:
            sig_rows.append([
                row["experiment"],
                reference_name,
                len(row["folds"]),
                0.0,
                0.0,
                1.0,
                1.0,
            ])
            sig_json.append(
                {
                    "experiment": row["experiment"],
                    "reference": reference_name,
                    "n_common_folds": len(row["folds"]),
                    "delta_f1_macro_mean": 0.0,
                    "delta_accuracy_mean": 0.0,
                    "pvalue_f1_macro": 1.0,
                    "pvalue_accuracy": 1.0,
                }
            )
            continue

        common_folds = sorted(set(row["folds"]) & set(reference["folds"]))
        if not common_folds:
            delta_f1 = float("nan")
            delta_acc = float("nan")
            p_f1 = float("nan")
            p_acc = float("nan")
        else:
            row_f1 = [row["fold_metric_by_id"][f]["f1_macro"] for f in common_folds]
            ref_f1 = [reference["fold_metric_by_id"][f]["f1_macro"] for f in common_folds]
            row_acc = [row["fold_metric_by_id"][f]["accuracy"] for f in common_folds]
            ref_acc = [reference["fold_metric_by_id"][f]["accuracy"] for f in common_folds]
            p_f1 = paired_permutation_pvalue(row_f1, ref_f1)
            p_acc = paired_permutation_pvalue(row_acc, ref_acc)
            delta_f1 = float(np.mean(np.asarray(row_f1) - np.asarray(ref_f1)))
            delta_acc = float(np.mean(np.asarray(row_acc) - np.asarray(ref_acc)))

        sig_rows.append([
            row["experiment"],
            reference_name,
            len(common_folds),
            delta_f1,
            delta_acc,
            p_f1,
            p_acc,
        ])
        sig_json.append(
            {
                "experiment": row["experiment"],
                "reference": reference_name,
                "n_common_folds": len(common_folds),
                "delta_f1_macro_mean": delta_f1,
                "delta_accuracy_mean": delta_acc,
                "pvalue_f1_macro": p_f1,
                "pvalue_accuracy": p_acc,
            }
        )

    write_csv(
        output_root / "significance_vs_reference.csv",
        [
            "experiment",
            "reference",
            "n_common_folds",
            "delta_f1_macro_mean",
            "delta_accuracy_mean",
            "pvalue_f1_macro",
            "pvalue_accuracy",
        ],
        sig_rows,
    )
    with (output_root / "significance_vs_reference.json").open("w", encoding="utf-8") as f:
        json.dump(sig_json, f, indent=2, ensure_ascii=False)

    header = [
        "experiment",
        "run_dir",
        "num_samples",
        "num_classes",
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_ci95_low",
        "accuracy_ci95_high",
        "f1_macro_mean",
        "f1_macro_std",
        "f1_macro_ci95_low",
        "f1_macro_ci95_high",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "balanced_accuracy_ci95_low",
        "balanced_accuracy_ci95_high",
    ]

    rows = [[s.get(k, "") for k in header] for s in all_summaries]
    write_csv(output_root / "experiment_comparison_summary.csv", header, rows)

    with (output_root / "experiment_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("=" * 96)
    print("experiment\tnum_samples\tacc_mean±std\tf1_macro_mean±std\tbalanced_acc_mean±std")
    for s in all_summaries:
        print(
            f"{s['experiment']}\t{s['num_samples']}\t"
            f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}\t"
            f"{s['f1_macro_mean']:.4f}±{s['f1_macro_std']:.4f}\t"
            f"{s['balanced_accuracy_mean']:.4f}±{s['balanced_accuracy_std']:.4f}"
        )
    print("=" * 96)
    if plt is None or sns is None:
        print("matplotlib/seaborn not available -> saved numeric results only (CSV/JSON).")
    print(f"reference for significance test: {reference_name}")
    print(f"saved evaluation outputs to: {output_root}")


if __name__ == "__main__":
    main()
