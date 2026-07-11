"""Minimal reproducible analysis for Figure 5h-i."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


LABEL_ORDER = ["Simple", "Multi", "Ridge", "Lasso", "Causal ML"]
TASK_ORDER = ["CH", "TG"]
TARGET_MAP = {"CH": "Total cholesterol (mg/dL)", "TG": "TG (mg/dL)"}
PANEL_MAP = {"CH": "h", "TG": "i"}
AXIS_MAP = {"MAE": "left", "R2": "right"}
YLABEL_MAP = {"MAE": "MAE (mg dl^-1)", "R2": "R2"}
REQUIRED_COLUMNS = [
    "PatientID",
    "Sweat Rate (uL/min)",
    "Sweat CH (uM)",
    "Sweat TG (uM)",
    "Total cholesterol (mg/dL)",
    "TG (mg/dL)",
    "CALCULATED BMI",
]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    feature_set: str
    model_name: str


MODEL_SPECS = [
    ModelSpec("Simple", "marker_only", "Linear"),
    ModelSpec("Multi", "marker_plus_rate", "Linear"),
    ModelSpec("Ridge", "full_model", "Ridge"),
    ModelSpec("Lasso", "full_model", "Lasso"),
    ModelSpec("Causal ML", "full_model", "RandomForest"),
]


def validate_input(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))


def prepare_task_df(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "CH":
        cols = ["PatientID", "Sweat CH (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI", "Total cholesterol (mg/dL)"]
        rename_map = {
            "Sweat CH (uM)": "sweat_marker",
            "Sweat Rate (uL/min)": "sweat_rate",
            "CALCULATED BMI": "bmi",
            "Total cholesterol (mg/dL)": "blood_target",
        }
    elif task == "TG":
        cols = ["PatientID", "Sweat TG (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI", "TG (mg/dL)"]
        rename_map = {
            "Sweat TG (uM)": "sweat_marker",
            "Sweat Rate (uL/min)": "sweat_rate",
            "CALCULATED BMI": "bmi",
            "TG (mg/dL)": "blood_target",
        }
    else:
        raise ValueError("task must be 'CH' or 'TG'")

    return (
        df[cols]
        .dropna()
        .reset_index(drop=False)
        .rename(columns={"index": "row_id"})
        .rename(columns=rename_map)
    )


def feature_columns(feature_set: str) -> list[str]:
    feature_sets = {
        "marker_only": ["sweat_marker"],
        "marker_plus_rate": ["sweat_marker", "sweat_rate"],
        "full_model": ["sweat_marker", "sweat_rate", "bmi"],
    }
    return feature_sets[feature_set]


def make_model(model_name: str, lasso_alpha: float, rf_trees: int, random_state: int):
    if model_name == "Linear":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge(alpha=1.0)
    if model_name == "Lasso":
        return Lasso(alpha=lasso_alpha, max_iter=10000)
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=rf_trees,
            random_state=random_state,
            max_depth=None,
            min_samples_leaf=1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def compute_fold_metrics(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        task_df = prepare_task_df(df, task)
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

        for spec in MODEL_SPECS:
            feat_cols = feature_columns(spec.feature_set)
            x = task_df[feat_cols].to_numpy()
            y = task_df["blood_target"].to_numpy()
            patient_ids = task_df["PatientID"].to_numpy()
            row_ids = task_df["row_id"].to_numpy()
            base_model = make_model(spec.model_name, args.lasso_alpha, args.rf_trees, args.random_state)

            for fold, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
                model = clone(base_model)
                model.fit(x[train_idx], y[train_idx])
                pred = model.predict(x[test_idx])
                rows.append(
                    {
                        "task": task,
                        "target": TARGET_MAP[task],
                        "model": spec.label,
                        "model_order": LABEL_ORDER.index(spec.label) + 1,
                        "feature_set": spec.feature_set,
                        "features": ", ".join(feat_cols),
                        "estimator": spec.model_name,
                        "fold": fold,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "train_patient_ids": ";".join(map(str, sorted(set(patient_ids[train_idx])))),
                        "test_patient_ids": ";".join(map(str, sorted(set(patient_ids[test_idx])))),
                        "test_row_ids": ";".join(map(str, row_ids[test_idx])),
                        "MAE": float(mean_absolute_error(y[test_idx], pred)),
                        "R2": float(r2_score(y[test_idx], pred)),
                    }
                )
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], LABEL_ORDER, ordered=True)
    return out.sort_values(["task", "model", "fold"]).reset_index(drop=True)


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_df.groupby(["task", "target", "model", "model_order", "feature_set", "features", "estimator"], observed=True)
        .agg(
            n_folds=("fold", "count"),
            n_test_min=("n_test", "min"),
            n_test_max=("n_test", "max"),
            MAE_mean=("MAE", "mean"),
            MAE_sd=("MAE", lambda x: x.std(ddof=1)),
            MAE_min=("MAE", "min"),
            MAE_max=("MAE", "max"),
            R2_mean=("R2", "mean"),
            R2_sd=("R2", lambda x: x.std(ddof=1)),
            R2_min=("R2", "min"),
            R2_max=("R2", "max"),
        )
        .reset_index()
    )
    summary["model"] = pd.Categorical(summary["model"], LABEL_ORDER, ordered=True)
    return summary.sort_values(["task", "model"]).reset_index(drop=True)


def read_reported_summary(path: str | None) -> dict[tuple[str, str, str], float]:
    if not path:
        return {}
    df = pd.read_csv(path)
    values = {}
    for _, row in df.iterrows():
        for metric in ["MAE", "R2"]:
            values[(row["task"], row["model"], metric)] = float(row[metric])
    return values


def build_bar_tidy(summary_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    reported = read_reported_summary(args.reported_summary)
    rows = []
    for _, row in summary_df.iterrows():
        task = str(row["task"])
        model = str(row["model"])
        for metric in ["MAE", "R2"]:
            key = (task, model, metric)
            if args.bar_source == "reported_summary":
                if key not in reported:
                    raise ValueError(
                        "Reported summary is missing a required value for "
                        f"task={task}, model={model}, metric={metric}"
                    )
                value = reported[key]
                bar_source = "reported_summary"
            else:
                value = float(row[f"{metric}_mean"])
                bar_source = "5fold_cv_mean"

            if args.whisker == "sd":
                sd = float(row[f"{metric}_sd"])
                low, high = value - sd, value + sd
                whisker_source = "5fold_cv_sample_sd"
            elif args.whisker == "minmax":
                fold_low, fold_high = float(row[f"{metric}_min"]), float(row[f"{metric}_max"])
                low, high = min(value, fold_low), max(value, fold_high)
                whisker_source = "5fold_cv_min_max_enclosing_bar"
            else:
                raise ValueError(f"Unknown whisker mode: {args.whisker}")

            rows.append(
                {
                    "panel": PANEL_MAP[task],
                    "task": task,
                    "target": TARGET_MAP[task],
                    "model": model,
                    "model_order": int(row["model_order"]),
                    "metric": metric,
                    "axis": AXIS_MAP[metric],
                    "ylabel": YLABEL_MAP[metric],
                    "bar_value": value,
                    "whisker_low": low,
                    "whisker_high": high,
                    "n_folds": int(row["n_folds"]),
                    "bar_source": bar_source,
                    "whisker_source": whisker_source,
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "model_order", "metric"]).reset_index(drop=True)


def build_points_tidy(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in fold_df.iterrows():
        for metric in ["MAE", "R2"]:
            rows.append(
                {
                    "panel": PANEL_MAP[row["task"]],
                    "task": row["task"],
                    "target": row["target"],
                    "model": row["model"],
                    "model_order": int(row["model_order"]),
                    "metric": metric,
                    "axis": AXIS_MAP[metric],
                    "ylabel": YLABEL_MAP[metric],
                    "fold": int(row["fold"]),
                    "fold_value": float(row[metric]),
                    "n_train": int(row["n_train"]),
                    "n_test": int(row["n_test"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "model_order", "fold", "metric"]).reset_index(drop=True)


PANEL_SPECS = {
    "CH": {"label": "h", "mae_ylim": (-10, 40), "r2_ylim": (-0.2, 0.8), "mae_color": "#f5d9af"},
    "TG": {"label": "i", "mae_ylim": (-20, 100), "r2_ylim": (-0.2, 1.0), "mae_color": "#f2a67f"},
}


def draw_task_panel(
    ax: plt.Axes,
    task: str,
    bar_tidy: pd.DataFrame,
    points_tidy: pd.DataFrame,
    random_state: int,
) -> None:
    rng = np.random.default_rng(random_state)
    spec = PANEL_SPECS[task]
    data = bar_tidy[bar_tidy["task"] == task]
    points = points_tidy[points_tidy["task"] == task]
    x = np.arange(len(LABEL_ORDER))
    width = 0.36
    ax2 = ax.twinx()
    mae = data[data["metric"] == "MAE"].set_index("model").loc[LABEL_ORDER]
    r2 = data[data["metric"] == "R2"].set_index("model").loc[LABEL_ORDER]
    ax.bar(
        x - width / 2,
        mae["bar_value"],
        yerr=np.vstack([mae["bar_value"] - mae["whisker_low"], mae["whisker_high"] - mae["bar_value"]]),
        width=width,
        color=spec["mae_color"],
        edgecolor="#6e6e6e",
        linewidth=0.5,
        capsize=2,
    )
    ax2.bar(
        x + width / 2,
        r2["bar_value"],
        yerr=np.vstack([r2["bar_value"] - r2["whisker_low"], r2["whisker_high"] - r2["bar_value"]]),
        width=width,
        color="#9fc3d4",
        edgecolor="#6e6e6e",
        linewidth=0.5,
        capsize=2,
    )
    for i, label in enumerate(LABEL_ORDER):
        mae_points = points[(points["model"] == label) & (points["metric"] == "MAE")]
        r2_points = points[(points["model"] == label) & (points["metric"] == "R2")]
        ax.scatter(
            np.full(len(mae_points), x[i] - width / 2) + rng.uniform(-0.045, 0.045, len(mae_points)),
            mae_points["fold_value"],
            s=9,
            color="#2d2d2d",
            alpha=0.85,
            linewidths=0,
            zorder=5,
        )
        ax2.scatter(
            np.full(len(r2_points), x[i] + width / 2) + rng.uniform(-0.045, 0.045, len(r2_points)),
            r2_points["fold_value"],
            s=12,
            facecolors="white",
            edgecolors="#5b7f99",
            linewidths=0.8,
            zorder=6,
        )
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax2.axhline(0, color="#777777", linewidth=0.6)
    ax.set_ylim(*spec["mae_ylim"])
    ax2.set_ylim(*spec["r2_ylim"])
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"MAE (mg dl$^{-1}$)", color="#5c4635")
    ax2.set_ylabel(r"$R^2$", color="#5b7f99")
    ax.tick_params(axis="y", colors="#5c4635")
    ax2.tick_params(axis="y", colors="#5b7f99")
    ax.text(-0.25, 1.08, spec["label"], transform=ax.transAxes, fontsize=11, fontweight="bold")


def figure_legend_handles() -> list:
    handles = [
        Patch(facecolor="#f5d9af", edgecolor="#6e6e6e", label="MAE bar"),
        Patch(facecolor="#9fc3d4", edgecolor="#6e6e6e", label="R2 bar"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2d2d2d", markersize=3.5, label="Fold-level MAE"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#5b7f99", markersize=4, label="Fold-level R2"),
    ]
    return handles


def plot_figure(bar_tidy: pd.DataFrame, points_tidy: pd.DataFrame, out_dir: Path, random_state: int) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.55), dpi=300)
    for ax, task in zip(axes, TASK_ORDER):
        draw_task_panel(ax, task, bar_tidy, points_tidy, random_state)
    fig.legend(
        handles=figure_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
        fontsize=6.8,
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.7)
    fig.savefig(out_dir / "figure5hi.png", bbox_inches="tight")
    plt.close(fig)

    for task in TASK_ORDER:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(2.95, 2.55), dpi=300)
        draw_task_panel(ax_single, task, bar_tidy, points_tidy, random_state)
        fig_single.legend(
            handles=figure_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            frameon=False,
            fontsize=6.8,
            ncol=2,
        )
        fig_single.tight_layout(rect=(0, 0.16, 1, 1))
        fig_single.savefig(out_dir / f"figure5{PANEL_SPECS[task]['label']}.png", bbox_inches="tight")
        plt.close(fig_single)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/merged_data.csv")
    parser.add_argument("--out", default="results/figure5hi")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lasso-alpha", type=float, default=0.1)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--bar-source", choices=["cv", "reported_summary"], default="cv")
    parser.add_argument("--reported-summary", default=None)
    parser.add_argument("--whisker", choices=["sd", "minmax"], default="sd")
    args = parser.parse_args()
    if args.bar_source == "reported_summary" and not args.reported_summary:
        raise ValueError("--reported-summary is required when --bar-source reported_summary")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    validate_input(df)
    fold_df = compute_fold_metrics(df, args)
    summary_df = summarize_fold_metrics(fold_df)
    bar_tidy = build_bar_tidy(summary_df, args)
    points_tidy = build_points_tidy(fold_df)
    keys = ["panel", "task", "target", "model", "model_order", "metric", "axis", "ylabel"]
    plot_tidy = points_tidy.merge(bar_tidy, on=keys, how="left")

    fold_df.to_csv(out_dir / "figure5hi_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "figure5hi_summary_metrics.csv", index=False)
    bar_tidy.to_csv(out_dir / "figure5hi_bar_tidy.csv", index=False)
    points_tidy.to_csv(out_dir / "figure5hi_points_tidy.csv", index=False)
    plot_tidy.to_csv(out_dir / "figure5hi_plot_tidy.csv", index=False)
    plot_figure(bar_tidy, points_tidy, out_dir, args.random_state)
    print(f"Saved results to: {out_dir}")
    print(summary_df[["task", "model", "MAE_mean", "MAE_sd", "R2_mean", "R2_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
