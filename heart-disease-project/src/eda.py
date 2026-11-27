
import pandas as pd
from pathlib import Path

# Locate the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Locate the Cleveland dataset
DATASET_PATH = PROJECT_DIR / "datasets" / "processed" / "cleveland.csv"

# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Display basic information
print("\nFIRST FIVE RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())



import matplotlib.pyplot as plt

# Create a histogram of patient ages
plt.figure(figsize=(8, 5))

plt.hist(
    df["age"],
    bins=15,
    edgecolor="black"
)

plt.title("Age Distribution of Cleveland Patients")
plt.xlabel("Age (years)")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()



# Compare age distributions by heart disease status

plt.figure(figsize=(8, 5))

df[df["num"] == 0]["age"].hist(
    bins=15,
    alpha=0.6,
    label="No Heart Disease"
)

df[df["num"] == 1]["age"].hist(
    bins=15,
    alpha=0.6,
    label="Heart Disease"
)

plt.title("Age Distribution by Heart Disease Status")
plt.xlabel("Age (years)")
plt.ylabel("Number of Patients")
plt.legend()

plt.tight_layout()
plt.show()



# Compare heart disease prevalence by sex

sex_comparison = pd.crosstab(
    df["sex"],
    df["num"],
    normalize="index"
) * 100

sex_comparison.index = ["Female", "Male"]
sex_comparison.columns = ["No Heart Disease", "Heart Disease"]

print("\nHEART DISEASE DISTRIBUTION BY SEX (%)")
print(sex_comparison.round(2))

# Create a bar chart
sex_comparison.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Heart Disease Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Percentage of Patients")
plt.xticks(rotation=0)
plt.legend(title="Diagnosis")
plt.tight_layout()
plt.show()



# Compare chest pain type with heart disease status

cp_comparison = pd.crosstab(
    df["cp"],
    df["num"],
    normalize="index"
) * 100

cp_comparison.columns = [
    "No Heart Disease",
    "Heart Disease"
]

cp_comparison.index = [
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
]

print("\nHEART DISEASE DISTRIBUTION BY CHEST PAIN TYPE (%)")
print(cp_comparison.round(2))

# Visualise the results
cp_comparison.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Heart Disease Distribution by Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Percentage of Patients")
plt.xticks(rotation=20)
plt.legend(title="Diagnosis")
plt.tight_layout()
plt.show()


# Compare resting blood pressure by heart disease status

plt.figure(figsize=(8, 5))

df[df["num"] == 0]["trestbps"].hist(
    bins=15,
    alpha=0.6,
    label="No Heart Disease"
)

df[df["num"] == 1]["trestbps"].hist(
    bins=15,
    alpha=0.6,
    label="Heart Disease"
)

plt.title("Resting Blood Pressure by Heart Disease Status")
plt.xlabel("Resting Blood Pressure (mm Hg)")
plt.ylabel("Number of Patients")
plt.legend()
plt.tight_layout()
plt.show()

# Compare resting blood pressure statistics

bp_comparison = df.groupby("num")["trestbps"].agg(
    ["count", "mean", "median", "std"]
)

bp_comparison.index = [
    "No Heart Disease",
    "Heart Disease"
]

print("\nRESTING BLOOD PRESSURE STATISTICS")
print(bp_comparison.round(2))