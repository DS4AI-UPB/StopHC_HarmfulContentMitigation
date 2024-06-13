from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from transformers import BertTokenizer, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel
import numpy as np
import tensorflow as tf
from typing import List
from sklearn.model_selection import KFold
from node2vec import Node2Vec
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

class WordEmbeddings:
    def __init__(self, max_len):
        self.max_len = max_len

    
    def word2VecEmbeddings(self, word2vec_embedding_dict, max_length):
        vocab_size = len(word2vec_embedding_dict.index_to_key)

        # Initialize the embedding matrix
        embedding_matrix = np.zeros((vocab_size, max_length))

        # Get the word vectors and fill the embedding matrix
        words = word2vec_embedding_dict.index_to_key
        for i in range(len(words)):
            embedding_vector = word2vec_embedding_dict[words[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_matrix[vocab_size - 1]= np.random.normal(scale=0.6, size=(max_length, ))
        
        return embedding_matrix


    def gloveEmbeddings(self, tokenizer, glove_embedding_dict):
        # Get word index
        word_index = tokenizer.word_index
        
        # Build the embedding matrix
        embedding_matrix = np.zeros((len(word_index) + 1, 50))
         
        for word, i in word_index.items():
            # Check if word is in GloVe embeddings
            if word in glove_embedding_dict:
                embedding_matrix[i] = glove_embedding_dict[word]
        
        return embedding_matrix
    
        
    # Function to get BERT vectors for a dataset
    def bertEmbeddings(self, dataset):
        # Initialize BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = TFBertModel.from_pretrained('bert-base-uncased', trainable=False)

        bert_embeddings = []
        tweets = dataset  # Extract tweets from dataset
        
        for text in tweets:
            # Tokenize text using BERT tokenizer
            encoded_text = tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='tf')
            
            # Get BERT model outputs
            outputs = bert_model(encoded_text)
        
            # Extract last hidden states from BERT outputs
            last_hidden_states = outputs.last_hidden_state
            
            cls_embedding = outputs[1]
            bert_embeddings.append(cls_embedding)

        bert_vectors = tf.concat(bert_embeddings, axis=0)
        bert_vectors_reshaped = tf.expand_dims(bert_vectors, axis=1)

        return bert_vectors_reshaped
    


    # Function to get RoBERTa vectors for a dataset
    def RoBertaEmbeddings(self, dataset: List[dict]) -> tf.Tensor:
        # Initialize RoBERTa tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta_model = TFRobertaModel.from_pretrained('roberta-base', trainable=False)

        embeddings = []
        tweets = dataset # Extract tweets from dataset
        
        for text in tweets:
            # Tokenize text using RoBERTa tokenizer
            encoded_text = tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='tf')
            
            # Get RoBERTa model outputs
            outputs = roberta_model(encoded_text)

            # Extract last hidden states from RoBERTa outputs
            last_hidden_states = outputs.last_hidden_state

            # Calculate mean embedding over sequence length
            mean_embedding = tf.reduce_mean(last_hidden_states, axis=1)
        
            embeddings.append(mean_embedding)

        # Concatenate RoBERTa embeddings along the first axis
        vectors = tf.concat(embeddings, axis=0)
        # Add an additional dimension to RoBERTa vectors
        vectors_reshaped = tf.expand_dims(vectors, axis=1)

        return vectors_reshaped
    

    def node2vecEmbeddings(self, graph, dataset, graph_embedding_size=50, walk_length=30, num_walks=200, workers=4, p=1, q=1):

        # Define custom weighted walk function
        def weighted_walk(node, prev_node, p, q):
            weights = []
            neighbors = list(graph.neighbors(node))

            for neighbor in neighbors:
                weight = graph.edges[node, neighbor].get('cost', 1)
                if neighbor == prev_node:
                    weights.append(weight / p)
                else:
                    weights.append(weight / q)

            weight_sum = sum(weights)
            probabilities = [weight / weight_sum for weight in weights]
            return np.random.choice(neighbors, p=probabilities)[0]

        # Create Node2Vec model with custom walk function
        node2vec = Node2Vec(graph, dimensions=graph_embedding_size, walk_length=walk_length, num_walks=num_walks, workers=workers)
        node2vec._walk = weighted_walk  # Override default walk function

        # Train the Node2Vec model
        node2vec_model = node2vec.fit(window=10, min_count=1)

        # Generate embeddings for tweets
        node2vec_embeddings = np.array([node2vec_model.wv[f"tweet_{tweet_id}"] for tweet_id in dataset.index])
        node2vec_embeddings = pad_sequences(node2vec_embeddings, maxlen=graph_embedding_size, dtype="float32", value=0, truncating="post", padding="post")

        # expand  embeddings structure for later use
        node2vec_embeddings = tf.concat(node2vec_embeddings, axis=0)
        node2vec_embeddings = tf.expand_dims(node2vec_embeddings, axis=1)
        return node2vec_embeddings