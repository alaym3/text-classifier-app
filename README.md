# Text-classifier-app

This streamlit app has deployed a model from the [DTU MLOps final project repo](https://github.com/alaym3/DTU_MLOps_FinalProject).

### Overall goal of the project
The goal of the project is to use natural language processing in order to perform sentiment classification on text, in order to predict whether a certain movie review from [Rotten Tomatoes](https://www.rottentomatoes.com/) is positive or negative.

### What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
We will use the [Transformers](https://huggingface.co/) framework since we are working with Natural Language Processing, specifically for sentiment classification of text.

### How to you intend to include the framework into your project
We will work on sentiment classification of text. The Transformers framework is highly flexible and allows many customizations. Many pretrained models for various types of Natural Language Processing tasks exist. They also provide datasets that can be combined with the pretrained models they offer, which makes the framework perfect for our task.

### What data are you going to run on (initially, may change)
We plan to use datasets provided by [HuggingFace](https://huggingface.co/datasets) - we will use the [Rotten Tomatoes review dataset](https://huggingface.co/datasets/rotten_tomatoes). The dataset includes two columns: the text from Rotten Tomatoes reviews for movies, along with a column indicating if the review is positive or negative. [Rotten Tomatoes](https://www.rottentomatoes.com/) is a platform where movie reviews are submitted by expert audiences and regular people.

We may look into other datasets from [HuggingFace](https://huggingface.co/datasets) or [Kaggle](https://www.kaggle.com/datasets) related to reviews of content or services, as we continue.

### What deep learning models do you expect to use
We expect to start by using the pre-trained transformer [bert-base-uncased](https://huggingface.co/bert-base-uncased) since it is the top used model for performing Natural Language Processing tasks on English text, including classification and question-answering. BERT consists of a bidirectional transformer that looks back and forward when analysing the tokens to learn the context of words. Since we want to perform sentiment classification on movie reviews, BERT is a natural model to begin with.
