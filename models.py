from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, GRU, Dense, GlobalMaxPooling1D, LeakyReLU, Dropout, Input, LSTM, Conv1D
from tensorflow import keras
from keras.optimizers.legacy import Adam, RMSprop
import tensorflow as tf


def build_classif_gru_model_with_embedding_3(word_index, embedding_matrix, max_sequence_length, gru_units, optimizator, lr):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 50, weights=[embedding_matrix], input_length=max_sequence_length, trainable=True))
    model.add(Bidirectional(GRU(units=gru_units, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
    else: 
        opt_final = RMSprop(learning_rate=lr)

    model.compile(optimizer=opt_final, loss='binary_crossentropy', metrics=['accuracy'])


    return model

def build_classif_gru_model_with_embedding(word_index, embedding_matrix, max_sequence_length, gru_units, optimizator, lr):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 50, weights=[embedding_matrix], input_length=max_sequence_length, trainable=True))
    model.add(Bidirectional(GRU(units=gru_units, return_sequences=True)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
    else: 
        opt_final = RMSprop(learning_rate=lr)

    model.compile(optimizer=opt_final, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_classif_lstm_model_with_embedding_2(word_index, embedding_matrix, max_sequence_length, optimizator, hidden_units, lr):
  
  model = Sequential()
  model.add(Embedding(len(word_index) + 1, max_sequence_length, weights=[embedding_matrix], input_length=max_sequence_length))
  model.add(Bidirectional(LSTM(hidden_units, dropout=0.3, recurrent_dropout=0.3)))
  model.add(Dense(16, activation="relu"))
  model.add(Dense(2, activation="softmax"))

  if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
  else: 
        opt_final = RMSprop(learning_rate=lr)

  model.compile(
    loss="categorical_crossentropy", 
    optimizer=opt_final, 
    metrics=["acc"])
  
  return model


def build_classif_lstm_model_with_embedding(word_index, embedding_matrix, max_sequence_length):
  
  model = Sequential()
  model.add(Embedding(len(word_index) + 1, 50, weights=[embedding_matrix], input_length=max_sequence_length))
  model.add(Bidirectional(LSTM(units=50, return_sequences=True, dropout=0.2, recurrent_dropout=0.4)))  # Replace GRU with LSTM
  model.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)))
  model.add(Dense(16, activation="relu"))
  model.add(Dense(2, activation="softmax"))
  model.compile(
    loss="categorical_crossentropy", 
    optimizer=RMSprop(learning_rate=0.0001), 
    metrics=["acc"]
)
  return model

def build_classif_gru_model_3(dimension_lenght, gru_units, optimizator, lr):
  
    second_gru_units = int(gru_units / 2)
    model = Sequential()
    model.add(Bidirectional(GRU(units=gru_units, return_sequences=True), input_shape=(None, dimension_lenght)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(units=second_gru_units, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
    else: 
        opt_final = RMSprop(learning_rate=lr)

    model.compile(optimizer=opt_final, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_classif_gru_model(dimension_lenght, gru_units, optimizator, lr):
  
    model = Sequential()
    model.add(Bidirectional(GRU(units=gru_units, return_sequences=True), input_shape=(None, dimension_lenght)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))

    if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
    else: 
        opt_final = RMSprop(learning_rate=lr)

    model.compile(optimizer=opt_final, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_classif_gru_model_2(dimension_lenght, gru_units, optimizator, lr):
  
    model = Sequential()
    model.add(Bidirectional(GRU(units=gru_units, return_sequences=True), input_shape=(None, dimension_lenght)))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
    else: 
        opt_final = RMSprop(learning_rate=lr)

    model.compile(
    loss="binary_crossentropy", 
    optimizer=opt_final, 
    metrics=["acc"])
    return model

def build_classif_lstm_model(dimension_lenght, lstm_units, optimizator, lr):

    model = Sequential()
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True), input_shape=(None, dimension_lenght)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))

    if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
    else: 
        opt_final = RMSprop(learning_rate=lr)

    model.compile(optimizer=opt_final, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_classif_lstm_model_2(dimension_lenght, lstm_units, optimizator, lr):

    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3), input_shape=(None, dimension_lenght)))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
    else: 
        opt_final = RMSprop(learning_rate=lr)

    model.compile(
        loss="categorical_crossentropy", 
        optimizer=opt_final, 
        metrics=["acc"])
    
    return model

def build_classif_lstm_model_3(dimension_lenght, lstm_units, optimizator, lr):

    second_lstm_units = int(lstm_units / 2)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True), input_shape=(None, dimension_lenght)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=second_lstm_units, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    if optimizator == 'adam':
        opt_final = Adam(learning_rate=lr)
    else: 
        opt_final = RMSprop(learning_rate=lr)

    model.compile(optimizer=opt_final, loss='binary_crossentropy', metrics=['accuracy'])

    return model