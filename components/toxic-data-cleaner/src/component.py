from google.cloud import storage
import pandas as pd
import os
from pathlib import Path
import re
import logging

# function has to be used for both the training and the prediction data
def clean_data(dataframe):

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
  dataframe['comment_text'] = dataframe['comment_text'].map(lambda com : clean_text(com))

  dataframe.drop(['id'], axis=1, inplace=True)

  logging.info('Cleaned text!')

