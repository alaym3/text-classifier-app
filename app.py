import streamlit as st
import numpy as np
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from google.cloud import storage

#### 1st step always: make layout wider #####
st.set_page_config(layout="wide", page_title="text-classification")

st.title('Sentiment classification app')
st.markdown('### This app allows you to type in a phrase and see how much the phrase is classified to be positive or negative. \
    Try it out on your own, and even compare the results for two different phrases!')

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./models/")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


left, right = st.columns(2)
with left:
    text_left = st.text_area('Write some text to perform sentiment analysis!', \
        'This is a very happy, positive-sounding text sample! Type your own.')

    inputs_left = tokenizer(text_left, return_tensors="pt") # , padding = True, truncation = True, return_tensors='pt').to('cuda')
    labels_left = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs_left = model(**inputs_left, labels=labels_left)
    predictions_left = torch.nn.functional.softmax(outputs_left.logits, dim=-1)
    predictions_left = predictions_left.cpu().detach().numpy()

    # store the positive and negative predictions
    neg_prediction_left = predictions_left[0][0]*100
    pos_prediction_left = predictions_left[0][1]*100

    # print results
    st.markdown(f'**Probability of the phrase being negative: {neg_prediction_left}%**')
    st.markdown(f'**Probability of the phrase being positive: {pos_prediction_left}%**')

with right:
    text_right = st.text_area('Write another text to perform sentiment analysis!',\
        'This is a very sad, upset-sounding text sample.. Type your own.')

    inputs_right = tokenizer(text_right, return_tensors="pt") # , padding = True, truncation = True, return_tensors='pt').to('cuda')
    labels_right = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs_right = model(**inputs_right, labels=labels_right)
    predictions_right = torch.nn.functional.softmax(outputs_right.logits, dim=-1)
    predictions_right = predictions_right.cpu().detach().numpy()

    # store the positive and negative predictions
    neg_prediction_right = predictions_right[0][0]*100
    pos_prediction_right = predictions_right[0][1]*100

    # print results
    st.markdown(f'**Probability of the phrase being negative: {neg_prediction_right}%**')
    st.markdown(f'**Probability of the phrase being positive: {pos_prediction_right}%**')


st.markdown('#### About the model:')
st.markdown('- We started with the pre-trained transformer [bert-base-uncased](https://huggingface.co/bert-base-uncased) since it is the top used model \
    for performing Natural Language Processing tasks on English text, including classification and \
        question-answering. BERT consists of a bidirectional transformer that looks back and forward when \
            analysing the tokens to learn the context of words.')
st.markdown('#### About the training:')
st.markdown('- The model has been trained on a [Rotten Tomatoes review dataset](https://huggingface.co/datasets/rotten_tomatoes). The dataset includes two columns: the text from Rotten Tomatoes \
    reviews for movies, along with a column indicating if the review is positive or negative. [Rotten Tomatoes](https://www.rottentomatoes.com/) is a platform \
        where movie reviews are submitted by expert audiences and regular people.')
