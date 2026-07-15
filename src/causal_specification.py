"""Causal assumptions used to define the Figure 5h-i model inputs.

This module intentionally separates causal assumptions from estimator choice.
The manuscript-facing "Causal ML" model is implemented as a predictive model
whose input set is selected from the causal diagram and adjustment rationale
below.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SWEAT_RATE = "Sweat Rate (uL/min)"
BMI = "CALCULATED BMI"
SEX = "Gender"

CORE_CONFOUNDER_CANDIDATES = [
    "Age (18>)",
    "Gender",
    "HgA1C",
    "Glucose",
    "Blood Pressure H",
    "Blood Pressure L",
    "Pulse",
    "Weight (lb)",
    BMI,
    "BMR (kcal)",
    "Fat%",
    "Fat mass (lb)",
    "FFM (lb)",
    "Predicted muscle mass (lb)",
    "TBW (lb)",
]


@dataclass(frozen=True)
class CausalTaskSpec:
    task: str
    blood_target: str
    sweat_marker: str
    sweat_marker_node: str
    blood_node: str


TASK_SPECS = {
    "CH": CausalTaskSpec(
        task="CH",
        blood_target="Total cholesterol (mg/dL)",
        sweat_marker="Sweat CH (uM)",
        sweat_marker_node="Sweat cholesterol",
        blood_node="Blood total cholesterol",
    ),
    "TG": CausalTaskSpec(
        task="TG",
        blood_target="TG (mg/dL)",
        sweat_marker="Sweat TG (uM)",
        sweat_marker_node="Sweat triglyceride",
        blood_node="Blood triglyceride",
    ),
}


def get_task_spec(task: str) -> CausalTaskSpec:
    if task not in TASK_SPECS:
        raise ValueError("task must be one of: " + ", ".join(TASK_SPECS))
    return TASK_SPECS[task]


def dag_edges(task: str) -> list[tuple[str, str, str]]:
    spec = get_task_spec(task)
    return [
        ("Confounders", spec.blood_node, "participant physiology can influence blood lipid level"),
        ("Confounders", "Sweat rate", "participant physiology can influence sweat secretion"),
        ("Confounders", spec.sweat_marker_node, "participant physiology can influence sweat matrix/transport"),
        (spec.blood_node, spec.sweat_marker_node, "blood lipid is the biological source signal"),
        ("Sweat rate", spec.sweat_marker_node, "sweat secretion affects measured sweat concentration"),
    ]


def prediction_feature_sets(task: str) -> dict[str, list[str]]:
    spec = get_task_spec(task)
    causal_guided = [spec.sweat_marker, SWEAT_RATE, BMI]
    if task == "CH":
        causal_guided = [spec.sweat_marker, SWEAT_RATE, BMI, SEX]
    return {
        "marker_only": [spec.sweat_marker],
        "marker_plus_rate": [spec.sweat_marker, SWEAT_RATE],
        "full_model": [spec.sweat_marker, SWEAT_RATE, BMI],
        "causal_guided_minimal": causal_guided,
    }


def model_feature_sets(task: str) -> dict[str, list[str]]:
    """Feature names after task-specific preprocessing in reproduce_figure5hi.py."""
    causal_guided = ["sweat_marker", "sweat_rate", "bmi"]
    if task == "CH":
        causal_guided = ["sweat_marker", "sweat_rate", "bmi", "sex"]
    return {
        "marker_only": ["sweat_marker"],
        "marker_plus_rate": ["sweat_marker", "sweat_rate"],
        "full_model": ["sweat_marker", "sweat_rate", "bmi"],
        "causal_guided_minimal": causal_guided,
    }


def adjustment_sets(task: str) -> dict[str, list[str]]:
    spec = get_task_spec(task)
    causal_guided = [SWEAT_RATE, BMI]
    if task == "CH":
        causal_guided = [SWEAT_RATE, BMI, SEX]
    return {
        "unadjusted": [],
        "sweat_rate_adjusted": [SWEAT_RATE],
        "causal_guided_minimal": causal_guided,
        "candidate_confounders": [c for c in CORE_CONFOUNDER_CANDIDATES if c != spec.blood_target],
    }


def build_dag_edges_table() -> pd.DataFrame:
    rows = []
    for task in TASK_SPECS:
        for source, target, rationale in dag_edges(task):
            rows.append({"task": task, "source": source, "target": target, "rationale": rationale})
    return pd.DataFrame(rows)


def build_feature_sets_table() -> pd.DataFrame:
    rows = []
    for task in TASK_SPECS:
        for feature_set, columns in prediction_feature_sets(task).items():
            rows.append(
                {
                    "task": task,
                    "feature_set": feature_set,
                    "columns": "; ".join(columns),
                    "role": "prediction_input",
                }
            )
        for set_name, columns in adjustment_sets(task).items():
            rows.append(
                {
                    "task": task,
                    "feature_set": set_name,
                    "columns": "; ".join(columns),
                    "role": "causal_adjustment_set",
                }
            )
    return pd.DataFrame(rows)


def write_causal_specification(out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_dag_edges_table().to_csv(out / "causal_dag_edges.csv", index=False)
    build_feature_sets_table().to_csv(out / "causal_feature_sets.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/causal_specification")
    args = parser.parse_args()
    write_causal_specification(args.out)
    print(f"Saved causal specification tables to: {args.out}")


if __name__ == "__main__":
    main()
