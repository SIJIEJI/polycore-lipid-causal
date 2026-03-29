import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# =========================
# 1. User settings
# =========================
FILE_PATH = "merged_data.csv"   # change this if needed
OUT_DIR = "negative_control_results"
TEST_SIZE = 0.30
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# 2. Load data
# =========================
df = pd.read_csv(FILE_PATH)
print(f"Loaded data shape: {df.shape}")

strict_non_sensor_features = [
        "Age (18>)",
        "Gender",
        "Blood Pressure H",
        "Blood Pressure L",
        "Pulse",
        "Weight (lb)",
        "CALCULATED BMI",
        "BMR (kcal)",
        "Fat%",
        "Fat mass (lb)",
        "FFM (lb)",
        "Predicted muscle mass (lb)",
        "TBW (lb)",
    ]


# =========================
# 3. Helper functions
# =========================
def prepare_task_df(df: pd.DataFrame, task: str = "TG") -> pd.DataFrame:
    """
    Prepare cleaned dataframe for TG or CH task.
    """
    strict_non_sensor_features = [
        "Age (18>)",
        "Gender",
        "Blood Pressure H",
        "Blood Pressure L",
        "Pulse",
        "Weight (lb)",
        "CALCULATED BMI",
        "BMR (kcal)",
        "Fat%",
        "Fat mass (lb)",
        "FFM (lb)",
        "Predicted muscle mass (lb)",
        "TBW (lb)",
    ]

    base_cols = ["PatientID"] + strict_non_sensor_features

    if task == "TG":
        cols = base_cols + [
            "Sweat TG (uM)",
            "Sweat Rate (uL/min)",
            "TG (mg/dL)",
        ]
        rename_map = {
            "Sweat TG (uM)": "sweat_marker",
            "Sweat Rate (uL/min)": "sweat_rate",
            "CALCULATED BMI": "bmi",
            "TG (mg/dL)": "blood_target",
        }
    elif task == "CH":
        cols = base_cols + [
            "Sweat CH (uM)",
            "Sweat Rate (uL/min)",
            "Total cholesterol (mg/dL)",
        ]
        rename_map = {
            "Sweat CH (uM)": "sweat_marker",
            "Sweat Rate (uL/min)": "sweat_rate",
            "CALCULATED BMI": "bmi",
            "Total cholesterol (mg/dL)": "blood_target",
        }
    else:
        raise ValueError("task must be 'TG' or 'CH'")

    task_df = (
        df[cols]
        .dropna()
        .reset_index(drop=False)
        .rename(columns={"index": "row_id"})
        .rename(columns=rename_map)
    )

    return task_df


def get_feature_sets() -> dict:
    """
    Define the two feature sets used in the negative-control table.
    """
    return {
        "full_model": ["sweat_marker", "sweat_rate", "bmi"],
        "confounders_only": ["strict_non_sensor_features"],
    }


def evaluate_model(
    task_df: pd.DataFrame,
    feat_cols: list[str],
    model,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[float, float]:
    """
    Fit one model on one feature set and return test MAE and test R2.
    """
    X = task_df[feat_cols].values
    y = task_df["blood_target"].values

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    return test_mae, test_r2


def build_negative_control_rows(task_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """
    Build only the four rows needed for the rebuttal table:
    - TG + all confounder (Causal on full_model)
    - confounder only 
    - CH + all confounder (Causal on full_model)
    - confounder only 
    """
    feature_sets = get_feature_sets()

    all_idx = np.arange(len(task_df))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    full_model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        max_depth=None,
        min_samples_leaf=1,
    )
    linear_baseline = LinearRegression()

    full_mae, full_r2 = evaluate_model(
        task_df,
        feature_sets["full_model"],
        full_model,
        train_idx,
        test_idx,
    )
    conf_mae, conf_r2 = evaluate_model(
        task_df,
        feature_sets["confounders_only"],
        linear_baseline,
        train_idx,
        test_idx,
    )

    if task_name == "TG":
        row_label_full = "TG + all confounder"
    elif task_name == "CH":
        row_label_full = "CH + all confounder"
    else:
        raise ValueError("task_name must be 'TG' or 'CH'")

    rows = pd.DataFrame(
        [
            {
                "row_label": row_label_full,
                "MAE": full_mae,
                "R2": full_r2,
            },
            {
                "row_label": "confounder only",
                "MAE": conf_mae,
                "R2": conf_r2,
            },
        ]
    )
    return rows


# =========================
# 4. Run TG and CH tasks
# =========================
tg_df = prepare_task_df(df, task="TG")
ch_df = prepare_task_df(df, task="CH")

print(f"TG usable samples: {len(tg_df)}")
print(f"CH usable samples: {len(ch_df)}")

tg_rows = build_negative_control_rows(tg_df, task_name="TG")
ch_rows = build_negative_control_rows(ch_df, task_name="CH")

final_table = pd.concat([tg_rows, ch_rows], axis=0).reset_index(drop=True)
final_table = final_table[["row_label", "MAE", "R2"]]

# Optional formatting for clean manuscript-style output
final_table["MAE"] = final_table["MAE"].map(lambda x: round(x, 6))
final_table["R2"] = final_table["R2"].map(lambda x: round(x, 6))

print("\n===== Negative-control table =====")
print(final_table.to_string(index=False))

csv_path = os.path.join(OUT_DIR, "negative_control_table.csv")
final_table.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")
