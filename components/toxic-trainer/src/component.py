from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
from google.cloud import storage
import pandas as pd
from pathlib import Path
import json
import os
import argparse


# create a function to add features
def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def train_multilabel_classifier(project_id, X_dtm, y_all, model_repo, metrics_path):

    # set up classifier
    logreg = LogisticRegression(C=12.0)
    cols_target = y_all.columns
    
    for labelnum, label in enumerate(cols_target):
        print('... Processing {}'.format(label))
        y = y_all[label]
        # train the model using X_dtm & y
        logreg.fit(X_dtm,y)

        # save the model
        local_file = f'/tmp/{labelnum}_{label}_model.joblib'
        dump(logreg, local_file)

        # Save to GCS as f'{labelnum}_{label}_model.joblib'
        client = storage.Client(project=project_id)
        bucket = client.get_bucket(model_repo)
        blob = bucket.blob(f'{labelnum}_{label}_model.joblib')

        # Upload the locally saved model
        blob.upload_from_filename(local_file)

        # Cleaning up by deleting the local file
        os.remove(local_file)

        # make predictions with training data
        y_pred_X = logreg.predict(X_dtm)
        # compute the evalution metrics for the training data
        metrics = {
            'accuracy': accuracy_score(y,y_pred_X),
            'precision': 'TODO',
            'recall': 'TODO',
            'f1': 'TODO'
        }
        print(metrics)

        # save the evaluation metrics
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as outfile:
            json.dump(metrics, outfile)

        # chain current label to X_dtm
        X_dtm = add_feature(X_dtm, y)
        print('Shape of X_dtm is now {}'.format(X_dtm.shape))
        # chain current label predictions to test_X_dtm
        # test_X_dtm = add_feature(test_X_dtm, test_y)
        # print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))

# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, help="GCP project id")
    parser.add_argument('--X_dtm', type=str, help="Dataframe with training features")
    parser.add_argument('--y_all', type=str, help="Dataframe with test features")
    parser.add_argument('--model_repo', type=str, help="Name of the model bucket")
    parser.add_argument('--metrics_path', type=str, help="Name of the file to be used for saving evaluation metrics")
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    train_multilabel_classifier(**parse_command_line_arguments())
    # The *args and **kwargs is a common idiom to allow arbitrary number of arguments to functions