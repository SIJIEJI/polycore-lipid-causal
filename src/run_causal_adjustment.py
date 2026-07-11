"""Causal adjustment diagnostics for the sweat-to-blood lipid analysis.

The diagnostics estimate adjusted associations implied by the causal diagram:

    Confounders -> blood lipid
    Confounders -> sweat rate
    Confounders -> sweat marker
    blood lipid -> sweat marker
    sweat rate -> sweat marker

They are not used as the Figure 5h-i predictive estimator. Instead, they make
the causal adjustment assumptions auditable and generate CSV outputs that show
which variables are adjusted for before the predictive model is fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from causal_specification import TASK_SPECS, adjustment_sets, write_causal_specification


def encode_covariates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    available = [c for c in columns if c in df.columns]
    cov = df[available].copy()
    cov = pd.get_dummies(cov, drop_first=True)
    for col in cov.columns:
        cov[col] = pd.to_numeric(cov[col], errors="coerce")
        cov[col] = cov[col].fillna(cov[col].median())
    return cov


def residualize(values: pd.Series, covariates: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if covariates.empty:
        return y - np.nanmean(y)
    model = LinearRegression()
    x = covariates.to_numpy(dtype=float)
    model.fit(x, y)
    return y - model.predict(x)


def slope_and_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.isclose(np.var(x), 0):
        return float("nan"), float("nan")
    slope = float(LinearRegression().fit(x.reshape(-1, 1), y).coef_[0])
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, r


def adjusted_association_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, spec in TASK_SPECS.items():
        needed = [spec.blood_target, spec.sweat_marker]
        task_df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
        if any(c not in task_df.columns for c in needed):
            continue

        for set_name, columns in adjustment_sets(task).items():
            cov = encode_covariates(task_df, columns)
            x_resid = residualize(task_df[spec.blood_target], cov)
            y_resid = residualize(task_df[spec.sweat_marker], cov)
            slope, r = slope_and_r(x_resid, y_resid)
            used_columns = [c for c in columns if c in df.columns]
            missing_columns = [c for c in columns if c not in df.columns]
            rows.append(
                {
                    "task": task,
                    "blood_target": spec.blood_target,
                    "sweat_marker": spec.sweat_marker,
                    "adjustment_set": set_name,
                    "adjusted_for": "; ".join(used_columns),
                    "missing_adjustment_columns": "; ".join(missing_columns),
                    "n": int(len(task_df)),
                    "slope_sweat_marker_per_blood_unit": slope,
                    "partial_r": r,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/merged_data.csv")
    parser.add_argument("--out", default="results/causal_adjustment")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    write_causal_specification(out)
    adjusted_association_rows(df).to_csv(out / "causal_adjustment_summary.csv", index=False)
    print(f"Saved causal adjustment outputs to: {out}")


if __name__ == "__main__":
    main()
