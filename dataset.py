import kagglehub
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder


class Dataset:
    def __init__(self):
        self.path = self.download_dataset() + '\\heart_attack_prediction_india.csv'
        self.dataset = pd.read_csv(self.path)

    def download_dataset(self):
        # Downloading Dataset
        path = kagglehub.dataset_download("ankushpanday2/heart-attack-risk-and-prediction-dataset-in-india")
        print("Path to dataset files:", path)

        return path

    def split_dataset(self):
        x = self.dataset.drop(columns=['Heart_Attack_Risk', 'Patient_ID'])
        y = self.dataset['Heart_Attack_Risk']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

        ct = ColumnTransformer(transformers=
        [
            ('onehot', OneHotEncoder(drop='first'), ['State_Name', 'Gender']),
            ('normal', StandardScaler(),
             ['Diastolic_BP', 'Annual_Income', 'Emergency_Response_Time', 'Systolic_BP', 'Cholesterol_Level',
              'Triglyceride_Level', 'LDL_Level', 'HDL_Level'])
        ], remainder='passthrough'
        )

        x_train_ct = ct.fit_transform(x_train)
        x_test_ct = ct.transform(x_test)

        return x_train_ct, x_test_ct, y_train, y_test