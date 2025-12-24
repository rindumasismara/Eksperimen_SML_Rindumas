import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocessing(df):
    df = df.copy()

    # 1. Drop missing value
    df.dropna(inplace=True)

    # 2. Encoding kolom type
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"])

    # 3. Scaling kolom numerik
    scaler = StandardScaler()
    num_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

def main():
    input_path = "../onlinepaymentfraud_dataset.csv"
    output_path = "onlinepaymentfraud_preprocessing.csv"

    df = load_data(input_path)
    df_clean = preprocessing(df)
    save_data(df_clean, output_path)

if __name__ == "__main__":
    main()