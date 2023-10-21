import logging
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump
from google.cloud import storage
import pandas as pd
from pathlib import Path
import os
import argparse


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def vectorize_data(X):
    vect = TfidfVectorizer(max_features=5000, stop_words='english')
    vect.fit(X)
    X_dtm = vect.transform(X)
    return X_dtm, vect


def save_model_to_gcs(project_id, model_repo, model_name, model):
    local_file = f'/tmp/{model_name}_model.joblib'
    dump(model, local_file)
    # Save to GCS as f'{labelnum}_{label}_model.joblib'
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(model_repo)
    blob = bucket.blob(f'{model_name}_model.joblib')
    # Upload the locally saved model to the bucket
    blob.upload_from_filename(local_file)
    # Cleaning up by deleting the local file
    os.remove(local_file)

def save_model_to_model_repo(model_repo, model_name, model):
    local_file = f'{model_repo}/{model_name}_model.joblib'
    Path(local_file).parent.mkdir(parents=True, exist_ok=True)
    dump(model, local_file)


def train_multilabel_classifier(project_id, train_path, model_repo, metrics_path):
    # read in the data
    df_train_data = pd.read_csv(train_path)
    df_train_data.drop(['id'], axis=1, inplace=True)
    X = df_train_data['comment_text']
    y_all = df_train_data.drop('comment_text', axis=1)

    # vectorize the text data
    X_dtm, vectorizer = vectorize_data(X)
    logging.info('Vectorized data!')

    # save the vectorizer to GCS
    #save_model_to_gcs(project_id, model_repo, 'vectorizer', vectorizer)
    save_model_to_model_repo(model_repo, 'vectorizer', vectorizer)

    # set up classifier
    logreg = LogisticRegression(C=12.0)

    cols_target = y_all.columns
    for labelnum, label in enumerate(cols_target):
        print('... Processing {}'.format(label))
        y = y_all[label]
        # train the model using X_dtm & y
        logreg.fit(X_dtm, y)

        # save the model to GCS
        #save_model_to_gcs(project_id, model_repo, f'{labelnum}_{label}', logreg)
        save_model_to_model_repo(model_repo, f'{labelnum}_{label}', logreg)

        # make predictions with training data
        y_pred = logreg.predict(X_dtm)

        # chain current label to X_dtm
        X_dtm = add_feature(X_dtm, y)
        print('Shape of X_dtm is now {}'.format(X_dtm.shape))
        # chain current label predictions to test_X_dtm
        # test_X_dtm = add_feature(test_X_dtm, test_y)


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, help="GCP project id")
    parser.add_argument('--train_path', type=str, help="Dataframe with training features")
    parser.add_argument('--model_repo', type=str, help="Name of the model bucket")
    parser.add_argument('--metrics_path', type=str, help="Name of the file to be used for saving evaluation metrics")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    train_multilabel_classifier(**parse_command_line_arguments())
    # The *args and **kwargs is a common idiom to allow arbitrary number of arguments to functions
