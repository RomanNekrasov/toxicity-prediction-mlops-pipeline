import logging
import shutil
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from joblib import load
from google.cloud import storage
import pandas as pd
from pathlib import Path
import json
import os
import argparse
import numpy as np


# create a function to add features
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def load_model_from_model_repo(model_repo, model_name):
    local_file = f'{model_repo}/{model_name}'
    model = load(local_file)
    return model


def calculate_metrics_and_save(y, y_pred, average_var, metrics_path):
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average=average_var),
        'recall': recall_score(y, y_pred, average=average_var),
        'f1': f1_score(y, y_pred, average=average_var)
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    file = f'metrics_{average_var}'
    with open(file, 'w') as outfile:
        json.dump(metrics, outfile)
    shutil.copy(file, metrics_path)
    os.remove(file)


def load_model_from_gcs(project_id, model_repo, model_name):
    local_file = f'/tmp/{model_name}'
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(model_repo)
    blob = bucket.blob(model_name)
    blob.download_to_filename(local_file)
    model = load(local_file)
    os.remove(local_file)
    return model


def predict_multilabel_classifier(project_id, predict_data, model_repo, metrics_path, validation_data):
    """Takes models from model repo and makes predictions on the predict data.
    If validation data is True, then it also performs validation on the predict data.
    """
    model_names = ['0_toxic_model.joblib',
                   '1_severe_toxic_model.joblib',
                   '2_obscene_model.joblib',
                   '3_threat_model.joblib',
                   '4_insult_model.joblib',
                   '5_identity_hate_model.joblib',
                   'vectorizer_model.joblib']
    if validation_data:
        # read in the data
        df_predict_data = pd.read_csv(predict_data)
        df_predict_data.drop(['id'], axis=1, inplace=True)
        y_all = df_predict_data.drop('comment_text', axis=1)
        X = df_predict_data['comment_text']
    else:
        X = predict_data

    # load the vectorizer from GCS
    vectorizer = load_model_from_gcs(project_id, model_repo, model_names[-1])
    # load the vectorizer from model repo
    # vectorizer = load_model_from_model_repo(model_repo, model_names[-1])  # This line is for testing locally
    # vectorize the text data
    X_dtm = vectorizer.transform([X])
    logging.info('Vectorized data!')

    # load the models from GCS
    cols_target = model_names[:-1]
    models = []
    for model_name in cols_target:
        model = load_model_from_gcs(project_id, model_repo, model_name)
        # model = load_model_from_model_repo(model_repo, model_name)  # This line is for testing locally
        models.append(model)

    # make predictions with predict data
    y_pred = []
    y_pred_proba = []
    for model in models:
        y_pred.append(model.predict(X_dtm))
        y_pred_proba.append(model.predict_proba(X_dtm)[:, -1])
        X_dtm = add_feature(X_dtm, y_pred[-1])

    # if validation data is True, then compute the evalution metrics for the predict data
    if validation_data:
        y_pred = np.array(y_pred).T
        micro_metrics = calculate_metrics_and_save(y_all, y_pred, 'micro', metrics_path)
        macro_metrics = calculate_metrics_and_save(y_all, y_pred, 'macro', metrics_path)
    else:
        labels = ['toxic',
                  'severe_toxic',
                  'obscene',
                  'threat',
                  'insult',
                  'identity_hate']
        instance_predictions = {}
        for label, predict_p in zip(labels, y_pred_proba):
            instance_predictions[label] = float(predict_p)  # need to cast to float since predict_p is np.array

        return instance_predictions


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, help="GCP project id")
    parser.add_argument('--predict_data', type=str, help="Dataframe with training features")
    parser.add_argument('--model_repo', type=str, help="Name of the model bucket")
    parser.add_argument('--metrics_path', type=str, help="Name of the file to be used for saving evaluation metrics")
    parser.add_argument('--validation_data', action='store_true',
                        help="Weather to perform validation on the data. Now this is just a flag (no value needed). If not given, it sets arguments to False")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    predict_multilabel_classifier(**parse_command_line_arguments())
    # The *args and **kwargs is a common idiom to allow arbitrary number of arguments to functions
