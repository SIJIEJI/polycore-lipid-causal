import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

# Load the uploaded CSV file
file_path = "merged_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
df.head()
# Target columns for TG analysis
sweat_tg_col = "Sweat TG (uM)"
blood_tg_col = "TG (mg/dL)"

start_col_index = df.columns.get_loc("V") if "V" in df.columns else 21  # fallback index if "V" not literal colname
cols_from_v = df.columns[start_col_index:]

# Ensure numeric
df_v = df[cols_from_v].apply(pd.to_numeric, errors="coerce")

# Drop columns with all NaN
df_v = df_v.dropna(axis=1, how="all")
# Combine Sweat/Blood TG with all factors from V onwards
df_corr_analysis_tg = pd.concat([df[[sweat_tg_col, blood_tg_col]], df_v], axis=1)

# Ensure numeric
df_corr_analysis_tg = df_corr_analysis_tg.apply(pd.to_numeric, errors="coerce")

# Compute correlations with Blood TG
correlations_with_blood_tg = df_corr_analysis_tg.corr()[blood_tg_col].drop(blood_tg_col)

# Sort by absolute correlation descending
correlations_sorted_tg = correlations_with_blood_tg.reindex(correlations_with_blood_tg.abs().sort_values(ascending=False).index)

# Highlight where Sweat TG is in this ranking
sweat_tg_corr = correlations_sorted_tg[sweat_tg_col]

# Plot bar chart
plt.figure(figsize=(8,6))
correlations_sorted_tg.plot(kind="bar", color=["red" if idx == sweat_tg_col else "skyblue" for idx in correlations_sorted_tg.index])
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Pearson r with Blood TG")
plt.title(f"Correlation of Factors with Blood TG (Sweat TG r={sweat_tg_corr:.2f})")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

sweat_tg_corr, correlations_sorted_tg.head(10)

# Target columns
sweat_ch_col = "Sweat CH (uM)"
blood_ch_col = "Total cholesterol (mg/dL)"

# Combine Sweat/Blood CH with all factors from V onwards
df_corr_analysis = pd.concat([df[[sweat_ch_col, blood_ch_col]], df_v], axis=1)

# Ensure numeric
df_corr_analysis = df_corr_analysis.apply(pd.to_numeric, errors="coerce")

# Compute correlations with Blood CH
correlations_with_blood = df_corr_analysis.corr()[blood_ch_col].drop(blood_ch_col)

# Sort by absolute correlation descending
correlations_sorted = correlations_with_blood.reindex(correlations_with_blood.abs().sort_values(ascending=False).index)

# Highlight where Sweat CH is in this ranking
sweat_ch_corr = correlations_sorted[sweat_ch_col]

# Plot bar chart
plt.figure(figsize=(8,6))
correlations_sorted.plot(kind="bar", color=["red" if idx == sweat_ch_col else "skyblue" for idx in correlations_sorted.index])
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Pearson r with Blood CH")
plt.title(f"Correlation of Factors with Blood CH (Sweat CH r={sweat_ch_corr:.2f})")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

sweat_ch_corr, correlations_sorted.head(10)

# Refined Nature-style version
#import matplotlib as mpl
# Prepare data for CH
corr_abs_ch = correlations_with_blood.abs()
corr_abs_ch_sorted = corr_abs_ch.reindex(corr_abs_ch.sort_values(ascending=False).index)

# Prepare data for TG
corr_abs_tg = correlations_with_blood_tg.abs()
corr_abs_tg_sorted = corr_abs_tg.reindex(corr_abs_tg.sort_values(ascending=False).index)

# Nature-style tweaks
mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False
})

fig, axes = plt.subplots(1, 2, figsize=(6.8,3), sharey=True)  # width ~ Nature single column

# Panel A - CH
colors_ch = ["#d62728" if idx == sweat_ch_col else "#1f77b4" for idx in corr_abs_ch_sorted.index]
axes[0].bar(range(len(corr_abs_ch_sorted)), corr_abs_ch_sorted, color=colors_ch)
axes[0].set_xticks(range(len(corr_abs_ch_sorted)))
axes[0].set_xticklabels(corr_abs_ch_sorted.index, rotation=90)
axes[0].set_ylim(0, 0.6)
axes[0].set_ylabel("|Pearson r| with Blood CH")
axes[0].text(-0.4, 0.62, "(a)", fontsize=10, fontweight="bold")
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

# Panel B - TG
colors_tg = ["#d62728" if idx == sweat_tg_col else "#1f77b4" for idx in corr_abs_tg_sorted.index]
axes[1].bar(range(len(corr_abs_tg_sorted)), corr_abs_tg_sorted, color=colors_tg)
axes[1].set_xticks(range(len(corr_abs_tg_sorted)))
axes[1].set_xticklabels(corr_abs_tg_sorted.index, rotation=90)
axes[1].set_ylim(0, 0.6)
axes[1].set_ylabel("|Pearson r| with Blood TG")
axes[1].text(-0.4, 0.62, "(b)", fontsize=10, fontweight="bold")
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout(w_pad=2)
plt.savefig("./Sweat_Blood_Correlation_Factors_NatureStyle.png", bbox_inches="tight")
plt.show()
