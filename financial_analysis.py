import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("2018_Financial_Data.csv")

# Prepare data
sector_counts = df["Sector"].value_counts()

sector_class_counts = pd.crosstab(
    df["Sector"],
    df["Class"]
)

sector_class_percent = pd.crosstab(
    df["Sector"],
    df["Class"],
    normalize="index"
) * 100

# Correlation analysis
correlations = df.corr(numeric_only=True)["2019 PRICE VAR [%]"]

correlations = correlations.drop(
    labels=["2019 PRICE VAR [%]", "Class"],
    errors="ignore"
)

top_correlations = correlations.loc[
    correlations.abs().nlargest(10).index
].sort_values()

# Create dashboard
fig, axes = plt.subplots(2, 2, figsize=(22, 18))

# Chart 1: Companies by Sector
sector_counts.plot(kind="bar", ax=axes[0, 0])

axes[0, 0].set_title("Companies by Sector")
axes[0, 0].set_xlabel("Sector")
axes[0, 0].set_ylabel("Number of Companies")
axes[0, 0].tick_params(axis="x", rotation=45)

# Chart 2: Class Distribution
sector_class_counts.plot(
    kind="bar",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Class Distribution by Sector")
axes[0, 1].set_xlabel("Sector")
axes[0, 1].set_ylabel("Number of Companies")
axes[0, 1].tick_params(axis="x", rotation=45)

# Chart 3: Class Distribution (%)
sector_class_percent.plot(
    kind="bar",
    stacked=True,
    ax=axes[1, 0]
)

axes[1, 0].set_title("Class Distribution by Sector (%)")
axes[1, 0].set_xlabel("Sector")
axes[1, 0].set_ylabel("Percentage")
axes[1, 0].tick_params(axis="x", rotation=45)

# Chart 4: Top Correlations
top_correlations.plot(
    kind="barh",
    ax=axes[1, 1]
)

axes[1, 1].set_title("Top 10 Financial Indicators Correlated with Stock Returns")
axes[1, 1].set_xlabel("Correlation")

# Final layout
plt.suptitle(
    "Financial Data Analysis Dashboard",
    fontsize=18
)
for ax in axes.flat:
    ax.tick_params(axis="x", labelsize=8)

plt.subplots_adjust(
    hspace=0.45,
    wspace=0.35,
    bottom=0.15
)   
# Improve labels readability
for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.tick_params(axis="x", labelsize=8)

    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")

axes[1, 1].tick_params(axis="y", labelsize=8.5, pad=3)
fig.subplots_adjust(
    left=0.07,
    right=0.98,
    bottom=0.15,
    top=0.90,
    hspace=1.0,
    wspace=0.65
)
axes[0, 0].set_xlabel("")
axes[0, 1].set_xlabel("")
axes[1, 0].set_xlabel("")
plt.savefig("eda_dashboard.png", dpi=300, bbox_inches="tight")
plt.show()
