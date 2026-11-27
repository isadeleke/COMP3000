import pandas as pd
from pathlib import Path

# To find the project direectory 
PROJECT_DIR = Path(__file__).resolve().parent.parent

#Define the oiginal and processed dataset folders i have created
ORIGINAL_DATA_DIR = PROJECT_DIR / "datasets" / "original"
PROCESSED_DATA_DIR = PROJECT_DIR / "datasets" / "processed"

#create the processed dataset folder if it does not exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

#Start to prepare the dataset
#Define the 14 columns in the UCI Heart Disease dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]   

#define the four datasets in the UCI Heart Disease dataset
datasets = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data"
}

#load, clean and save each dataset as a csv file 
for dataset_name, dataset_file in datasets.items():

    file_path = ORIGINAL_DATA_DIR / dataset_file

    #Read the orignial data and recognise ? as missing values
    df = pd.read_csv(file_path, header=None, names=columns, na_values="?")

    #convert measurements recorded as zero into missing values 
    #These corrections apply only to the affected datasets.

    if dataset_name in ["switzerland", "va"]:
        df.loc[df["chol"] == 0, "chol"] = pd.NA

    if dataset_name == "va":
        df.loc[df["trestbps"] == 0, "trestbps"] = pd.NA

    #convert the original target into binary classification 
    # 0 = no heart disease
    # 1, 2, 3, 4 = heart disease present 
    df["num"] = (df["num"] > 0).astype(int)

    #saving the prepared dataset as a csv file in the processed dataset folder
    processed_file_path = PROCESSED_DATA_DIR / f"{dataset_name}.csv"
    df.to_csv(processed_file_path, index=False)

    #display a summary of the dataset after processing
    print(f"\n{dataset_name.upper()} DATASET")
    print("Records:", len(df))
    print("Missing Values:", df.isna().sum().sum())
    print("Target distribution:")
    print(df["num"].value_counts().sort_index())
    print("Saved to :", processed_file_path)

print("\nAll datasets have been processed and saved successfully.")