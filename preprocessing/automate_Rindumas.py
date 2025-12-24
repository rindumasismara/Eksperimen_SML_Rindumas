import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocessing(df):
    df = df.copy()

    # 1. Handle missing value
    df = df.dropna()

    # 2. Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # 3. Select numeric columns
    num_cols = df.columns.drop("isFraud")

    # 4. Scaling
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df[num_cols] = df[num_cols].astype(float)

    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

def main():
    input_path = "onlinepaymentfraud_dataset.csv"
    output_path = "preprocessing/onlinepaymentfraud_preprocessing.csv"

    df = load_data(input_path)
    df_clean = preprocessing(df)
    save_data(df_clean, output_path)

if __name__ == "__main__":
    main()