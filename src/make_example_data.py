"""Generate synthetic schema-compatible data for smoke testing.

The generated data are not real study data and do not reproduce manuscript
metrics. They only verify that the analysis code runs end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_example_data(n_subjects: int, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for patient_id in range(1, n_subjects + 1):
        gender = int(rng.integers(0, 2))
        age = int(rng.integers(21, 66))
        bmi = float(rng.uniform(19, 35))
        weight = float(rng.normal(175, 35))
        fat_pct = float(np.clip(12 + 1.1 * (bmi - 20) + rng.normal(0, 4), 8, 50))
        fat_mass = weight * fat_pct / 100
        ffm = weight - fat_mass
        muscle = ffm * float(rng.uniform(0.82, 0.96))
        tbw = ffm * float(rng.uniform(0.68, 0.75))
        bmr = 900 + 8.0 * weight + 4.0 * muscle - 2.5 * age + float(rng.normal(0, 80))
        hba1c = float(rng.normal(5.6 + 0.025 * (bmi - 25), 0.25))
        glucose_screen = float(rng.normal(94 + 1.4 * (bmi - 25), 9))
        bp_h = float(rng.normal(112 + 1.0 * (bmi - 25), 9))
        bp_l = float(rng.normal(72 + 0.55 * (bmi - 25), 6))
        pulse = float(rng.normal(70, 8))

        for _ in range(repeats):
            sweat_rate = float(np.clip(rng.normal(1.8 + 0.03 * (bmi - 25), 0.55), 0.2, 4.0))
            sweat_ch = float(np.clip(rng.normal(1.8 + 0.04 * (bmi - 25) - 0.10 * gender, 0.55), 0.1, 5))
            sweat_tg = float(np.clip(rng.normal(80 + 1.5 * (bmi - 25), 30), 5, 180))
            blood_ch = 150 + 10.5 * sweat_ch - 4.5 * sweat_rate + 1.5 * bmi + 6 * gender + float(rng.normal(0, 18))
            blood_tg = 55 + 0.85 * sweat_tg - 8.0 * sweat_rate + 2.0 * bmi + float(rng.normal(0, 22))

            rows.append(
                {
                    "PatientID": patient_id,
                    "Glucose (mg/dL)": float(rng.normal(glucose_screen, 12)),
                    "HDL C (mg/dL)": float(rng.normal(55 - 0.5 * (bmi - 25), 8)),
                    "TG (mg/dL)": blood_tg,
                    "Total cholesterol (mg/dL)": blood_ch,
                    "LDL Chol (mg/dL)": float(rng.normal(105 + 1.0 * (bmi - 25), 16)),
                    "Sweat Rate (uL/min)": sweat_rate,
                    "Sweat CH (uM)": sweat_ch,
                    "Sweat TG (uM)": sweat_tg,
                    "Age (18>)": age,
                    "Gender": gender,
                    "HgA1C": hba1c,
                    "Glucose": glucose_screen,
                    "Blood Pressure H": bp_h,
                    "Blood Pressure L": bp_l,
                    "Pulse": pulse,
                    "Weight (lb)": weight,
                    "CALCULATED BMI": bmi,
                    "BMR (kcal)": bmr,
                    "Fat%": fat_pct,
                    "Fat mass (lb)": fat_mass,
                    "FFM (lb)": ffm,
                    "Predicted muscle mass (lb)": muscle,
                    "TBW (lb)": tbw,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/example_merged_data.csv")
    parser.add_argument("--n-subjects", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_example_data(args.n_subjects, args.repeats, args.seed)
    df.to_csv(out, index=False)
    print(f"Saved synthetic example data: {out} ({df.shape[0]} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()

