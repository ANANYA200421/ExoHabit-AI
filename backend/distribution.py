import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/exoplanet_ranking.csv")

plt.figure(figsize=(8,6))
sns.histplot(df["Habitability_Probability"], bins=20, kde=True)

plt.title("Habitability Probability Distribution")
plt.xlabel("Habitability Probability")
plt.ylabel("Number of Planets")

plt.tight_layout()
plt.savefig("../frontend/static/score_distribution.png")
plt.show()
