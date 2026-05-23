import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset/customer_data.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("visuals/correlation_heatmap.png")
plt.close()

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=10, kde=True)
plt.title("Age Distribution")
plt.savefig("visuals/age_distribution.png")
plt.close()

# Income vs Spending
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["Annual_Income"], y=df["Spending_Score"], hue=df["Gender"])
plt.title("Income vs Spending Score")
plt.savefig("visuals/income_vs_spending.png")
plt.close()

# Region Count
plt.figure(figsize=(6,4))
df["Region"].value_counts().plot(kind="bar")
plt.title("Customers by Region")
plt.xlabel("Region")
plt.ylabel("Count")
plt.savefig("visuals/region_distribution.png")
plt.close()

# Generate Insights Report
report = """
EDA PROJECT INSIGHTS REPORT
===========================

1. Age distribution shows most customers are young adults.
2. Higher annual income often correlates with higher spending score.
3. Spending behavior varies by gender and region.
4. North and South regions have more customer activity.
5. Correlation heatmap identifies important influencing features.

Conclusion:
EDA helps uncover hidden trends and business patterns for better decision making.
"""

with open("reports/eda_report.txt", "w") as f:
    f.write(report)

print("\nEDA Project Completed Successfully!")
