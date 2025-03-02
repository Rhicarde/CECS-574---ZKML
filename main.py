import kagglehub
import numpy as np
import pandas as pd

def download_dataset():
    # Downloading Dataset
    path = kagglehub.dataset_download("ankushpanday2/heart-attack-risk-and-prediction-dataset-in-india")
    print("Path to dataset files:", path)

    return path

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = download_dataset() + '\\heart_attack_prediction_india.csv'
    df = pd.read_csv(path)

    print(df.head())
    print(df.describe())
    print(df.columns)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
