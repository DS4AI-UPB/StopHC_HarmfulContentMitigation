# Harmful Speech Detection

## Introduction
This project provides  a comprehensive approach to harmful speech detection using various natural language processing (NLP) techniques and embeddings. The notebook includes preprocessing steps, model training, hyperparameter tuning, and evaluation of models using different embedding techniques such as GloVe, BERT, RoBERTa, and Node2Vec.

## Data Processing
The dataset used in this notebook is extracted from the EXIST2023 dataset. The dataset contains tweets labeled with annotations for harmful speech and it also mentions a list of annotators with characteristics.

## GloVe Embeddings
The notebook starts with training a model using GloVe embeddings. It tokenizes tweets, pads sequences, and constructs the embedding matrix. Then, it builds a Bidirectional GRU model with an Embedding layer using GloVe embeddings. The model is trained using k-fold cross-validation.

## Hyperparameter Tuning
Hyperparameter tuning is performed using grid search to find the optimal hyperparameters for the GloVe-based model. This includes tuning parameters such as the number of units in the GRU layer, batch size, and optimizer.

## BERT and RoBERTa Embeddings
The notebook also explores the use of BERT and RoBERTa embeddings for harmful speech detection. It demonstrates how to tokenize tweets and obtain BERT/RoBERTa vectors. These embeddings are then used to train models and evaluate their performance.

## Node2Vec Embeddings
Additionally, the notebook incorporates graph embeddings using Node2Vec. It constructs a graph of tweets based on common annotators and defines a custom weighted walk function for generating Node2Vec embeddings. These embeddings are combined with other embeddings for training a hybrid model.

## Model Training
Finally, the notebook provides a function to train models using different embedding techniques. It loads the specified model and concatenates embeddings with graph embeddings. The model architecture includes Bidirectional GRU layers followed by GlobalMaxPooling1D and Dense layers. The model is trained using k-fold cross-validation.



