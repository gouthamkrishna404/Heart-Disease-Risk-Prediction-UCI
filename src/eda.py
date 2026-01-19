import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("output/preprocessing", exist_ok=True)

df = pd.read_csv("data/heart.csv")
df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

# Missing values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values")
plt.savefig("output/preprocessing/missing_values.png")
plt.close()

# Target distribution
sns.countplot(x="num", data=df)
plt.title("Target Distribution")
plt.savefig("output/preprocessing/target_distribution.png")
plt.close()

num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f"{col} Boxplot")
    plt.savefig(f"output/preprocessing/{col}_boxplot.png")
    plt.close()

    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.savefig(f"output/preprocessing/{col}_hist.png")
    plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols + ["num"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("output/preprocessing/correlation.png")
plt.close()
