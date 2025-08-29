import pandas as pd
import numpy as np
import os

# List of all dataset paths
DATASET_FILES = [
    "archive/Base.csv",
    "archive/Variant I.csv",
    "archive/Variant II.csv",
    "archive/Variant III.csv",
    "archive/Variant IV.csv",
    "archive/Variant V.csv"
]

def preprocess_dataset(path):
    df = pd.read_csv(path)
    # 1. Remove rows with missing values
    df = df.dropna()
    # 2. Remove outliers (IQR method) for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    # 3. Group rare categories in categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        counts = df[col].value_counts()
        rare = counts[counts < 5].index
        df[col] = df[col].apply(lambda x: 'OTROS' if x in rare else x)
    # 4. Convert numeric columns to float
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    # 5. Encode categorical columns (simple integer encoding)
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    # 6. Normalize numeric columns (z-score)
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
        else:
            df[col] = 0.0
    return df

def main():
    dfs = []
    for file in DATASET_FILES:
        if os.path.exists(file):
            print(f"Processing {file} ...")
            dfs.append(preprocess_dataset(file))
        else:
            print(f"Warning: {file} not found.")
    df_all = pd.concat(dfs, ignore_index=True)
    # Identify target column (fraud/fraude)
    target_col = [col for col in df_all.columns if 'fraud' in col.lower() or 'fraude' in col.lower()]
    if not target_col:
        raise Exception("No fraud column found!")
    target_col = target_col[0]
    # Split features and target
    X = df_all.drop(columns=[target_col]).values.tolist()
    y = df_all[target_col].values.tolist()
    print(f"Total samples: {len(X)}")
    # Optionally: Save processed data
    pd.DataFrame(X).to_csv("processed_features.csv", index=False)
    pd.DataFrame(y, columns=[target_col]).to_csv("processed_target.csv", index=False)
    print("Processed data saved as processed_features.csv and processed_target.csv.")

if __name__ == "__main__":
    main()
