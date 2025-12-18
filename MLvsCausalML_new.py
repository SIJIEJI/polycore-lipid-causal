import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("merged_data.csv")

# Show columns with positional indices to identify "V column onward"
cols_with_idx = list(enumerate(df.columns.tolist()))
cols_with_idx[:40]


# Helper to build and evaluate three model settings for a given target
def evaluate_models(target_col, core_feature, sweat_rate_col="Sweat Rate (uL/min)", bmi_col="CALCULATED BMI"):
    # Drop rows with missing target
    sub = df.dropna(subset=[target_col]).copy()
    
    # Baseline: core sweat only
    X_base = sub[[core_feature]].copy()
    y = sub[target_col].values
    
    # Minimal causal: core sweat + sweat rate + BMI
    X_causal_min = sub[[core_feature, sweat_rate_col, bmi_col]].copy()
    
    # Expanded confounders: all columns from position >= 21 (V column onward), excluding target and ID-like fields
    start_idx = 21  # 'Age (18>)' position; corresponds to Excel's V
    expand_cols = df.columns[start_idx:]
    # Remove target and leakage columns if present in expand set
    exclude = {target_col, core_feature, sweat_rate_col, bmi_col}
    expand_features = [c for c in expand_cols if c not in exclude]
    
    # Compose expanded features: core + sweat rate + BMI + (V onward others)
    X_expanded = sub[[core_feature, sweat_rate_col, bmi_col] + expand_features].copy()
    
    # Identify numeric columns (drop non-numeric like categorical 'Gender' for linear models)
    numeric_cols = X_expanded.select_dtypes(include=[np.number]).columns.tolist()
    # For baseline/causal_min we ensure numeric as well
    base_numeric = X_base.select_dtypes(include=[np.number]).columns.tolist()
    causal_numeric = X_causal_min.select_dtypes(include=[np.number]).columns.tolist()
    
    # Train-test split using same indices for fairness
    idx = np.arange(len(sub))
    X_train_b, X_test_b, y_train, y_test, idx_train, idx_test = train_test_split(
        X_base[base_numeric], y, idx, test_size=0.3, random_state=42
    )
    X_train_c, X_test_c = X_causal_min.iloc[idx_train][causal_numeric], X_causal_min.iloc[idx_test][causal_numeric]
    X_train_e, X_test_e = X_expanded.iloc[idx_train][numeric_cols], X_expanded.iloc[idx_test][numeric_cols]
    
    # Pipelines
    def pipe(model):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),  # keep sparse-friendly; though data dense
            ("model", model)
        ])
    
    # Models: baseline LinearRegression; causal_min RandomForest; expanded RidgeCV (robust to multicollinearity) + RF
    baseline = pipe(LinearRegression()).fit(X_train_b, y_train)
    causal = pipe(RandomForestRegressor(n_estimators=300, random_state=42)).fit(X_train_c, y_train)
    expanded_ridge = pipe(RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))).fit(X_train_e, y_train)
    expanded_rf = pipe(RandomForestRegressor(n_estimators=400, random_state=42)).fit(X_train_e, y_train)
    
    # Predict
    preds = {
        "baseline_lin": baseline.predict(X_test_b),
        "causal_rf": causal.predict(X_test_c),
        "expanded_ridge": expanded_ridge.predict(X_test_e),
        "expanded_rf": expanded_rf.predict(X_test_e),
    }
    
    # Metrics
    results = {}
    for k, yhat in preds.items():
        results[k] = {
            "MAE": float(mean_absolute_error(y_test, yhat)),
            "R2": float(r2_score(y_test, yhat))
        }
    return results, {
        "n_test": len(y_test),
        "n_features_expanded": len(numeric_cols),
        "expanded_feature_names": numeric_cols[:10] + (["..."] if len(numeric_cols) > 10 else [])
    }

# Evaluate TG
tg_results, tg_info = evaluate_models(
    target_col="TG (mg/dL)",
    core_feature="Sweat TG (uM)"
)

# Evaluate CH
ch_results, ch_info = evaluate_models(
    target_col="Total cholesterol (mg/dL)",
    core_feature="Sweat CH (uM)"
)

print(tg_results, tg_info, ch_results, ch_info)

# Let's produce CSV and comparison plots for expanded confounder models (RF, since it performed best).
tg_data_expanded = df.dropna(subset=["TG (mg/dL)"]).copy()
ch_data_expanded = df.dropna(subset=["Total cholesterol (mg/dL)"]).copy()

# TG expanded
start_idx = 21  # V col index
tg_expand_cols = df.columns[start_idx:]
tg_exclude = {"TG (mg/dL)", "Sweat TG (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI"}
tg_features_expanded = ["Sweat TG (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI"] + \
                       [c for c in tg_expand_cols if c not in tg_exclude]
tg_features_expanded_num = tg_data_expanded[tg_features_expanded].select_dtypes(include=[np.number]).columns.tolist()

X_tg = tg_data_expanded[tg_features_expanded_num]
y_tg = tg_data_expanded["TG (mg/dL)"]
X_train_tg, X_test_tg, y_train_tg, y_test_tg, idx_train_tg, idx_test_tg = train_test_split(
    X_tg, y_tg, tg_data_expanded.index, test_size=0.3, random_state=42
)
rf_tg_expanded = RandomForestRegressor(n_estimators=400, random_state=42).fit(X_train_tg, y_train_tg)
tg_preds = rf_tg_expanded.predict(X_tg)

tg_out_expanded = tg_data_expanded[["PatientID", "Sweat TG (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI", "TG (mg/dL)"]].copy()
tg_out_expanded = tg_out_expanded.rename(columns={
    "Sweat TG (uM)": "sweat_tg",
    "Sweat Rate (uL/min)": "sweat_rate",
    "CALCULATED BMI": "bmi",
    "TG (mg/dL)": "blood_tg"
})
tg_out_expanded["set"] = np.where(tg_data_expanded.index.isin(idx_test_tg), "test", "train")
tg_out_expanded["pred_expanded_causal"] = tg_preds

# CH expanded
ch_expand_cols = df.columns[start_idx:]
ch_exclude = {"Total cholesterol (mg/dL)", "Sweat CH (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI"}
ch_features_expanded = ["Sweat CH (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI"] + \
                       [c for c in ch_expand_cols if c not in ch_exclude]
ch_features_expanded_num = ch_data_expanded[ch_features_expanded].select_dtypes(include=[np.number]).columns.tolist()

X_ch = ch_data_expanded[ch_features_expanded_num]
y_ch = ch_data_expanded["Total cholesterol (mg/dL)"]
X_train_ch, X_test_ch, y_train_ch, y_test_ch, idx_train_ch, idx_test_ch = train_test_split(
    X_ch, y_ch, ch_data_expanded.index, test_size=0.3, random_state=42
)
rf_ch_expanded = RandomForestRegressor(n_estimators=400, random_state=42).fit(X_train_ch, y_train_ch)
ch_preds = rf_ch_expanded.predict(X_ch)

ch_out_expanded = ch_data_expanded[["PatientID", "Sweat CH (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI", "Total cholesterol (mg/dL)"]].copy()
ch_out_expanded = ch_out_expanded.rename(columns={
    "Sweat CH (uM)": "sweat_ch",
    "Sweat Rate (uL/min)": "sweat_rate",
    "CALCULATED BMI": "bmi",
    "Total cholesterol (mg/dL)": "blood_ch"
})
ch_out_expanded["set"] = np.where(ch_data_expanded.index.isin(idx_test_ch), "test", "train")
ch_out_expanded["pred_expanded_causal"] = ch_preds

# Save CSVs
tg_exp_path = "./tg_predictions_expanded_conf.csv"
ch_exp_path = "./ch_predictions_expanded_conf.csv"
tg_out_expanded.to_csv(tg_exp_path, index=False)
ch_out_expanded.to_csv(ch_exp_path, index=False)
# Let's produce baseline ML predictions (traditional) for TG and CH for comparison
def baseline_lin_predictions(data, target_col, core_feature):
    X = data[[core_feature]].select_dtypes(include=[np.number])
    y = data[target_col]
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("lin", LinearRegression())
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    return preds

# TG baseline
tg_preds_base = baseline_lin_predictions(tg_data_expanded, "TG (mg/dL)", "Sweat TG (uM)")
tg_out["pred_baseline"] = tg_preds_base

# CH baseline
ch_preds_base = baseline_lin_predictions(ch_data_expanded, "Total cholesterol (mg/dL)", "Sweat CH (uM)")
ch_out["pred_baseline"] = ch_preds_base

# Plot both baseline and expanded in same plot for TG
def plot_baseline_vs_expanded(df, true_col, base_col, exp_col, title, unit):
    plt.figure(figsize=(6,6))
    plt.scatter(df[true_col], df[base_col], alpha=0.6, label="Baseline ML", color="orange")
    plt.scatter(df[true_col], df[exp_col], alpha=0.6, label="Expanded Causal ML", color="blue")
    min_val, max_val = df[true_col].min(), df[true_col].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel(f"True {title} ({unit})")
    plt.ylabel(f"Predicted {title} ({unit})")
    plt.title(f"Baseline vs Expanded Causal ML: {title}")
    plt.legend()
    plt.show()

plot_baseline_vs_expanded(tg_out, "blood_tg", "pred_baseline", "pred_expanded_causal", "Triglycerides", "mg/dL")
plot_baseline_vs_expanded(ch_out, "blood_ch", "pred_baseline", "pred_expanded_causal", "Total Cholesterol", "mg/dL")
