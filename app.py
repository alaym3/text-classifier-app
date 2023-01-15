import streamlit as st
import numpy as np
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from google.cloud import storage

#### 1st step always: make layout wider #####
st.set_page_config(layout="wide", page_title="text-classification")

st.title('Sentiment classification app')
st.markdown('### This app allows you to type in a phrase and see how much the phrase is classified to be positive or negative!')

# # load models
# def read(bucket_name):
#     """read a blob from GCS using file-like IO"""
#     # The ID of your GCS bucket
#     # bucket_name = "your-bucket-name"

#     # The ID of your new GCS object
#     # blob_name = "storage-object-name"

#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blobs = storage_client.list_blobs(bucket_name)

#     # blob = bucket.blob(blob_name)

#     # Mode can be specified as wb/rb for bytes mode.
#     # See: https://docs.python.org/3/library/io.html
#     # with blob.open("w") as f:
#     #     f.write("Hello world")
#     for blob in blobs:
#         with blob.open("r") as f:
#             st.markdown(f.read())

# read("model-bucket-streamlit-text-classification")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("models")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Prediction for a new phrase
# text = test_dataset[0]['text'] # or text = "I love this movie"
text = st.text_area('Text to perform sentiment analysis', '''
    This is a very happy, positive-sounding text sample! Type your own.
    ''')




inputs = tokenizer(text, return_tensors="pt") # , padding = True, truncation = True, return_tensors='pt').to('cuda')
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()
# print('Predictions for the phrase "' + text + '" : ', predictions)
st.write('Predictions for the phrase "' + text + '" : ', predictions)


# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])

#     chart_data