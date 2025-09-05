""" """

import pandas as pd
import time
import typer
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

def create_pandas_df_from_csv(path_to_csv):
    if path_to_csv is None:
        print(f'No path to csv provided')
        return None
    else:    
        print(f"path to csv is {path_to_csv}")
        data_df = pd.read_csv(path_to_csv, header=0, index_col=[0, 1])
        data_df = data_df.fillna(value=0)
        print(f'dataset created')
        return data_df

def split_dataset(data_df):
    """Split the dataset into features and target."""
    X = data_df.drop(columns=['Metadata_Treatment'])
    y = data_df["Metadata_Treatment"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42, shuffle=True)
    return x_train, x_test, x_val, y_train, y_test, y_val

def standardize_data(x_train, x_test, x_val):
    """Standardize the feature data."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)
    X_val = scaler.transform(x_val)
    return X_train, X_test, X_val

def train_gradient_boosting(X_train, y_train):
    """Train a gradient boosting model."""
    #timestamp
    t0 = time.time()
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    clf.score(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    report_train = classification_report(y_train, y_pred_train)
    print("Training time:", time.time() - t0)
    return clf, report_train, y_train, y_pred_train

def evaluate_model(model, X_val,y_val):
    y_pred = model.predict(X_val)
    report_val = classification_report(y_val, y_pred)
    return report_val, y_val, y_pred

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def main(
        DAPI_alone: bool = False, 
        A488_alone: bool = False, 
        A568_alone: bool = False,
        joined_data: bool = False):
    
    dataset_list =[]

    if DAPI_alone == True:
        path_to_csv = "/mnt/efs/aimbl_2025/student_data/S-DD/DAPI_alone.csv"
        dataset_list.append("DAPI_alone")

    if A488_alone == True:
        path_to_csv = "/mnt/efs/aimbl_2025/student_data/S-DD/A488_alone.csv"
        dataset_list.append("A488_alone")

    if A568_alone == True:
        path_to_csv = "/mnt/efs/aimbl_2025/student_data/S-DD/A568_alone.csv"
        dataset_list.append("A568_alone")

    if joined_data == True:
        path_to_csv = "/mnt/efs/aimbl_2025/student_data/S-DD/joined_data.csv"
        dataset_list.append("joined_dataset")


    data_df = create_pandas_df_from_csv(path_to_csv)
    print(data_df.head())
    print(data_df.shape)

    x_train, x_test, x_val, y_train, y_test, y_val = split_dataset(data_df)
    X_train, X_test, X_val = standardize_data(x_train, x_test, x_val)

    clf, report_train, y_train, y_pred_train = train_gradient_boosting(X_train, y_train)
    print("Training complete.")
    print(f"Training report for Gradient Boosting Classifier:")
    print(report_train)
    print(f"Confusion matrix for training:")
    print(confusion_matrix(y_train, y_pred_train))
    # Generate a time stamp with YYYYMMDD-HHMMSS
    tstamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = (f'Test_gradientboostingclassifier_{'-'.join(dataset_list)}-{tstamp}.pkl')
    save_model(clf, file_path)

    report_val, y_val, y_pred=evaluate_model(clf, X_val, y_val)
    print(f"Validation report for Gradient Boosting Classifier:")
    print(report_val)
    print(f"Confusion matrix for validation:")
    print(confusion_matrix(y_val, y_pred))

    
    TREATMENTS = ['8nMActD', 'DMSO', '1uMdoxo', 'CX5461', '5uMflavo', '800nMActD', '10uMmg132', '10uMwort']
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=TREATMENTS)
    disp.plot(xticks_rotation = 45)
    disp.figure_.savefig(f'/mnt/efs/aimbl_2025/student_data/S-DD/nucleoli_classifier_validation_cm_{'-'.join(dataset_list)}-{tstamp}.png')

if __name__ == "__main__":
    typer.run(main)