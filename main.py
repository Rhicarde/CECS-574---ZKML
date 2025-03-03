import kagglehub
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def download_dataset():
    # Downloading Dataset
    path = kagglehub.dataset_download("ankushpanday2/heart-attack-risk-and-prediction-dataset-in-india")
    print("Path to dataset files:", path)

    return path

def train_test(df, epoch=20):
    x = df.drop(columns=['Heart_Attack_Risk', 'Patient_ID'])
    y = df['Heart_Attack_Risk']

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

    model = LogisticRegression()
    model.fit(x_train_ct, y_train)
    y_pred = model.predict(x_test_ct)

    print('accuracy_score', accuracy_score(y_test, ypred))
    print('Confusion_matrix', confusion_matrix(y_test, ypred))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = download_dataset() + '\\heart_attack_prediction_india.csv'
    df = pd.read_csv(path)

    train_test(df)


