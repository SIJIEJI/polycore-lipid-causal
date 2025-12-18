import pandas as pd

# Load the uploaded CSV file
file_path = "merged_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
df.head()

# First, let's prepare the dataset for causal modeling by including confounders
# We will use: Sweat TG (uM), Sweat Rate (uL/min), BMI, and predict Blood TG (mg/dL)

# Select and clean relevant columns
causal_cols = ["Sweat TG (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI", "TG (mg/dL)"]
df_causal = df[causal_cols].dropna()

# Rename columns for simplicity
df_causal = df_causal.rename(columns={
    "Sweat TG (uM)": "sweat_tg",
    "Sweat Rate (uL/min)": "sweat_rate",
    "CALCULATED BMI": "bmi",
    "TG (mg/dL)": "blood_tg"
})

# Show cleaned data
df_causal.head()


import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Define predictors and target
X_linear = df_causal[["sweat_tg"]]  # baseline model
X_causal = df_causal[["sweat_tg", "sweat_rate", "bmi"]]  # adjusted with confounders
y = df_causal["blood_tg"]

# Split into train and test
X_train_lin, X_test_lin, y_train, y_test = train_test_split(X_linear, y, test_size=0.3, random_state=42)
X_train_causal, X_test_causal, _, _ = train_test_split(X_causal, y, test_size=0.3, random_state=42)

# Train models
baseline_model = LinearRegression().fit(X_train_lin, y_train)
causal_model = RandomForestRegressor(random_state=42, n_estimators=100).fit(X_train_causal, y_train)

# Predict
y_pred_baseline = baseline_model.predict(X_test_lin)
y_pred_causal = causal_model.predict(X_test_causal)

# Evaluate
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

mae_causal = mean_absolute_error(y_test, y_pred_causal)
r2_causal = r2_score(y_test, y_pred_causal)


print("基线模型 MAE:", mae_baseline, "R²:", r2_baseline)
print("因果模型 MAE:", mae_causal, "R²:", r2_causal)


# Prepare data for causal modeling for cholesterol
causal_cols_ch = ["Sweat CH (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI", "Total cholesterol (mg/dL)"]
df_causal_ch = df[causal_cols_ch].dropna()

# Rename columns
df_causal_ch = df_causal_ch.rename(columns={
    "Sweat CH (uM)": "sweat_ch",
    "Sweat Rate (uL/min)": "sweat_rate",
    "CALCULATED BMI": "bmi",
    "Total cholesterol (mg/dL)": "blood_ch"
})

# Define predictors and target
X_linear_ch = df_causal_ch[["sweat_ch"]]  # baseline model
X_causal_ch = df_causal_ch[["sweat_ch", "sweat_rate", "bmi"]]  # with confounders
y_ch = df_causal_ch["blood_ch"]

# Train-test split
X_train_lin_ch, X_test_lin_ch, y_train_ch, y_test_ch = train_test_split(X_linear_ch, y_ch, test_size=0.3, random_state=42)
X_train_causal_ch, X_test_causal_ch, _, _ = train_test_split(X_causal_ch, y_ch, test_size=0.3, random_state=42)

# Fit models
baseline_model_ch = LinearRegression().fit(X_train_lin_ch, y_train_ch)
causal_model_ch = RandomForestRegressor(random_state=42, n_estimators=100).fit(X_train_causal_ch, y_train_ch)

# Predictions
y_pred_baseline_ch = baseline_model_ch.predict(X_test_lin_ch)
y_pred_causal_ch = causal_model_ch.predict(X_test_causal_ch)

# Evaluation
mae_baseline_ch = mean_absolute_error(y_test_ch, y_pred_baseline_ch)
r2_baseline_ch = r2_score(y_test_ch, y_pred_baseline_ch)

mae_causal_ch = mean_absolute_error(y_test_ch, y_pred_causal_ch)
r2_causal_ch = r2_score(y_test_ch, y_pred_causal_ch)

print("基线模型 MAE:", mae_baseline_ch, "R²:", r2_baseline_ch)
print("因果模型 MAE:", mae_causal_ch, "R²:", r2_causal_ch)




# Plot prediction results: Baseline vs Causal model
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Baseline: Sweat TG only
sns.scatterplot(x=y_test, y=y_pred_baseline, ax=ax[0])
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[0].set_title(f"Baseline Model (MAE: {mae_baseline:.1f}, R²: {r2_baseline:.2f})")
ax[0].set_xlabel("True Blood TG (mg/dL)")
ax[0].set_ylabel("Predicted Blood TG")

# Causal model: with confounder adjustment
sns.scatterplot(x=y_test, y=y_pred_causal, ax=ax[1])
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[1].set_title(f"Causal Model (MAE: {mae_causal:.1f}, R²: {r2_causal:.2f})")
ax[1].set_xlabel("True Blood TG (mg/dL)")
ax[1].set_ylabel("Predicted Blood TG")

plt.suptitle("Sweat-to-Blood Triglyceride Inference: Model Comparison", fontsize=14)
plt.tight_layout()
plt.show()




# tg_ML = "./tg_predictions_baseline_vs_causal.csv"
# ch_ML = "./ch_predictions_baseline_vs_causal.csv"
# tg_CausalML = "./tg_predictions_causal_vs_causal.csv"
# ch_CausalML = "./ch_predictions_causal_vs_causal.csv"

# tg_ML = pd.read_csv(tg_ML)
# ch_ML = pd.read_csv(ch_ML)
# tg_CausalML = pd.read_csv(tg_CausalML)
# ch_CausalML = pd.read_csv(ch_CausalML)
# tg_out.to_csv(tg_path, index=False)
# ch_out.to_csv(ch_path, index=False)

# Plot cholesterol inference results: Baseline vs Causal
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Baseline model
sns.scatterplot(x=y_test_ch, y=y_pred_baseline_ch, ax=ax[0])
ax[0].plot([y_test_ch.min(), y_test_ch.max()], [y_test_ch.min(), y_test_ch.max()], 'r--')
ax[0].set_title(f"Baseline CH Model (MAE: {mae_baseline_ch:.1f}, R²: {r2_baseline_ch:.2f})")
ax[0].set_xlabel("True Total CH (mg/dL)")
ax[0].set_ylabel("Predicted Total CH")

# Causal model
sns.scatterplot(x=y_test_ch, y=y_pred_causal_ch, ax=ax[1])
ax[1].plot([y_test_ch.min(), y_test_ch.max()], [y_test_ch.min(), y_test_ch.max()], 'r--')
ax[1].set_title(f"Causal CH Model (MAE: {mae_causal_ch:.1f}, R²: {r2_causal_ch:.2f})")
ax[1].set_xlabel("True Total CH (mg/dL)")
ax[1].set_ylabel("Predicted Total CH")

plt.suptitle("Sweat-to-Blood Cholesterol Inference: Model Comparison", fontsize=14)
plt.tight_layout()
plt.show()

# save plots data to csv
# tg_ML_data = pd.DataFrame({
#     "true": y_test,
#     "pred_baseline": y_pred_baseline,
#     "pred_causal": y_pred_causal
# })
# ch_ML_data = pd.DataFrame({
#     "true": y_test_ch,
#     "pred_baseline": y_pred_baseline_ch,
#     "pred_causal": y_pred_causal_ch
# })

# tg_ML_data.to_csv("tg_ML_data.csv", index=False)
# ch_ML_data.to_csv("ch_ML_data.csv", index=False)


# # Reload the uploaded CSV and generate baseline vs causal prediction CSVs for TG and CH
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # Load uploaded file
# #df = pd.read_csv("/mnt/data/merged_data.csv")

# # ---------- TG ----------
# tg_cols = ["PatientID", "Sweat TG (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI", "TG (mg/dL)"]
# tg_df = df[tg_cols].dropna().rename(columns={
#     "Sweat TG (uM)": "sweat_tg",
#     "Sweat Rate (uL/min)": "sweat_rate",
#     "CALCULATED BMI": "bmi",
#     "TG (mg/dL)": "blood_tg"
# }).reset_index(drop=False).rename(columns={"index": "row_id"})

# X_lin = tg_df[["sweat_tg"]]
# X_causal = tg_df[["sweat_tg", "sweat_rate", "bmi"]]
# y = tg_df["blood_tg"]

# X_train_lin, X_test_lin, y_train, y_test, idx_train, idx_test = train_test_split(
#     X_lin, y, tg_df.index, test_size=0.3, random_state=42
# )
# X_train_causal = X_causal.loc[idx_train]
# X_test_causal = X_causal.loc[idx_test]

# lin_tg = LinearRegression().fit(X_train_lin, y_train)
# rf_tg = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_causal, y_train)

# pred_lin = lin_tg.predict(X_lin)
# pred_rf = rf_tg.predict(X_causal)

# tg_out = tg_df.copy()
# tg_out["set"] = np.where(tg_df.index.isin(idx_test), "test", "train")
# tg_out["pred_baseline"] = pred_lin
# tg_out["pred_causal"] = pred_rf

# # ---------- CH ----------
# ch_cols = ["PatientID", "Sweat CH (uM)", "Sweat Rate (uL/min)", "CALCULATED BMI", "Total cholesterol (mg/dL)"]
# ch_df = df[ch_cols].dropna().rename(columns={
#     "Sweat CH (uM)": "sweat_ch",
#     "Sweat Rate (uL/min)": "sweat_rate",
#     "CALCULATED BMI": "bmi",
#     "Total cholesterol (mg/dL)": "blood_ch"
# }).reset_index(drop=False).rename(columns={"index": "row_id"})

# X_lin_ch = ch_df[["sweat_ch"]]
# X_causal_ch = ch_df[["sweat_ch", "sweat_rate", "bmi"]]
# y_ch = ch_df["blood_ch"]

# X_train_lin_ch, X_test_lin_ch, y_train_ch, y_test_ch, idx_train_ch, idx_test_ch = train_test_split(
#     X_lin_ch, y_ch, ch_df.index, test_size=0.3, random_state=42
# )
# X_train_causal_ch = X_causal_ch.loc[idx_train_ch]
# X_test_causal_ch = X_causal_ch.loc[idx_test_ch]

# lin_ch = LinearRegression().fit(X_train_lin_ch, y_train_ch)
# rf_ch = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_causal_ch, y_train_ch)

# pred_lin_ch = lin_ch.predict(X_lin_ch)
# pred_rf_ch = rf_ch.predict(X_causal_ch)

# ch_out = ch_df.copy()
# ch_out["set"] = np.where(ch_df.index.isin(idx_test_ch), "test", "train")
# ch_out["pred_baseline"] = pred_lin_ch
# ch_out["pred_causal"] = pred_rf_ch

# # Save files
# tg_path = "./tg_predictions_baseline_vs_causal.csv"
# ch_path = "./ch_predictions_baseline_vs_causal.csv"
# tg_out.to_csv(tg_path, index=False)
# ch_out.to_csv(ch_path, index=False)

# tg_path, ch_path
