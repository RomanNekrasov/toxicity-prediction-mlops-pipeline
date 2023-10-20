import pandas as pd
import re
import logging
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_data(X):
  vect = TfidfVectorizer(max_features=5000,stop_words='english')
  X_dtm = vect.fit_transform(X)
  return X_dtm

def clean_text(text):
  text = text.lower()
  text = re.sub(r"what's", "what is ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", "cannot ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r"\'scuse", " excuse ", text)
  text = re.sub('\W', ' ', text)
  text = re.sub('\s+', ' ', text)
  text = text.strip(' ')
  return text

# function has to be used for both the training and the prediction data
def clean_data(dataframe):
  dataframe['comment_text'] = dataframe['comment_text'].map(lambda com : clean_text(com))
  logging.info('Cleaned text!')
  dataframe.drop(['id'], axis=1, inplace=True)
  X = dataframe['comment_text']
  y_all = dataframe.drop('comment_text', axis=1)
  # vectorize the text data
  X_dtm = vectorize_data(X)

# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="The ingested dataframe")
    args = parser.parse_args()
    return vars(args)