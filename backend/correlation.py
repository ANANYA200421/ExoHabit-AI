import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/exoplanet_ranking.csv")

# Select only numeric columns
numeric_df = df.select_dtypes(include=["float64", "int64"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Star–Planet Parameter Correlation")
plt.tight_layout()

plt.savefig("../frontend/static/correlation_heatmap.png")
plt.show()
