import pandas as pd

# Load the uploaded CSV file
file_path = "merged_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
df.head()


import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# TG dataset
tg_data = df[["Sweat TG (uM)", "TG (mg/dL)"]].dropna().copy()
tg_data.rename(columns={"Sweat TG (uM)": "Sweat_TG_uM", "TG (mg/dL)": "Blood_TG_mg_dL"}, inplace=True)

# Fit TG model
X_tg = sm.add_constant(tg_data["Sweat_TG_uM"])
y_tg = tg_data["Blood_TG_mg_dL"]
model_tg = sm.OLS(y_tg, X_tg).fit()
tg_data["Fitted"] = model_tg.predict(X_tg)
tg_data["Residual"] = y_tg - tg_data["Fitted"]

# CH dataset
ch_data = df[["Sweat CH (uM)", "Total cholesterol (mg/dL)"]].dropna().copy()
ch_data.rename(columns={"Sweat CH (uM)": "Sweat_CH_uM", "Total cholesterol (mg/dL)": "Blood_Total_CH_mg_dL"}, inplace=True)

# Fit CH model
X_ch = sm.add_constant(ch_data["Sweat_CH_uM"])
y_ch = ch_data["Blood_Total_CH_mg_dL"]
model_ch = sm.OLS(y_ch, X_ch).fit()
ch_data["Fitted"] = model_ch.predict(X_ch)
ch_data["Residual"] = y_ch - ch_data["Fitted"]

# Plot TG regression and residuals
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.regplot(data=tg_data, x="Sweat_TG_uM", y="Blood_TG_mg_dL", ci=95, line_kws={"color": "red"}, ax=ax[0])
ax[0].set_title("Sweat TG vs Blood TG with 95% CI")
sns.scatterplot(data=tg_data, x="Fitted", y="Residual", ax=ax[1])
ax[1].axhline(0, linestyle="--", color="gray")
ax[1].set_title("Residuals vs Fitted (TG)")
plt.tight_layout()
plt.show()

# Plot CH regression and residuals
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.regplot(data=ch_data, x="Sweat_CH_uM", y="Blood_Total_CH_mg_dL", ci=95, line_kws={"color": "red"}, ax=ax[0])
ax[0].set_title("Sweat CH vs Blood Total CH with 95% CI")
sns.scatterplot(data=ch_data, x="Fitted", y="Residual", ax=ax[1])
ax[1].axhline(0, linestyle="--", color="gray")
ax[1].set_title("Residuals vs Fitted (CH)")
plt.tight_layout()
plt.show()

# Save to CSV
tg_file_path = "./sweat_blood_TG_with_fit.csv"
ch_file_path = "./sweat_blood_CH_with_fit.csv"
tg_data.to_csv(tg_file_path, index=False)
ch_data.to_csv(ch_file_path, index=False)

(tg_file_path, ch_file_path)
