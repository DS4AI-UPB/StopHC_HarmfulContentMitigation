
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt


def process_dataset_exist2023(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # convert JSON 
    df = pd.DataFrame.from_dict(data, orient='index')

    # extract the english tweets
    df_en = df[df['lang'] == 'en']

    # define one label per tweet
    y_labels = df_en['labels_task1'].apply(lambda labels: int(labels.count('YES') > len(labels) / 2))

    return df_en, y_labels


def hate_speech_dataset(filename):
    data = pd.read_csv(filename)
    df_tweets = data['tweet']
    (data
     .groupby("class")
     .agg(
        hate_speech_count=("hate_speech_count", "mean"),
        offensive_language_count=("offensive_language_count", "mean"),
        neither_count=("neither_count", "mean"),
     )
     .round(1)
)

    data = (data
    .drop(columns=["count", "hate_speech_count", "offensive_language_count", "neither_count"])
    )
    data = (data
    .assign(
        class_=data["class"].map({
            0:1,
            1:1,
            2:0
        })
    )
    .drop(columns=["class"])
    .rename(columns={"class_": "class"})
)
    return df_tweets, data["class"]


def process_dataset_final_copy(filename):
    df = pd.read_csv(filename)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the 'Class' column to binary labels
    y_labels = label_encoder.fit_transform(df['Class'])

    # Extract the 'Tweets' column into a new dataframe
    tweets_df = df['Tweets']

    # Display the resulting dataframe
    return tweets_df, y_labels

def process_dataset_final(filename):
    df = pd.read_csv(filename)

    # Define a function to map labels to 0 or 1
    def label_to_binary(label):
        if label.lower() in ['sexism', 'racism']:
            return 1
        else:
            return 0

    # Apply the function to the 'Class' column to create a new column 'Label'
    y_labels = df['Class'].apply(label_to_binary)

    # Extract the 'Tweets' and 'Label' columns into a new dataframe
    tweets_df = df['Tweets']

    # Display the resulting dataframe
    return tweets_df, y_labels

def process_dataset_olid(filename):
    df = pd.read_csv(filename)

    # Define a function to map labels to 0 or 1
    def label_to_binary(label):
        if label == "OFF":
            return 1
        else:
            return 0

    # Apply the function to the 'Class' column to create a new column 'Label'
    y_labels = df['subtask_a'].apply(label_to_binary)

    # Extract the 'Tweets' and 'Label' columns into a new dataframe
    tweets_df = df['tweet']

    # Display the resulting dataframe
    return tweets_df, y_labels


def plot_loss_curves(histories, start_fold=5, end_fold=10):
    plt.figure(figsize=(12, 8))
    for fold in range(start_fold - 1, end_fold):
        plt.plot(histories[fold]['loss'], label=f'Fold {fold + 1} Train Loss')
        plt.plot(histories[fold]['val_loss'], label=f'Fold {fold + 1} Val Loss')
    
    plt.title('Training and Validation Loss Curves for Folds 5 to 10')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def kfold_train_evaluate(model, X, y, nr_splits, callback, epoch_num):
    # Initialize K-Fold
    kfold = StratifiedKFold(n_splits=nr_splits, shuffle=True, random_state=42)
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    histories = []

    X = tf.constant(X)
    y_labels_tensor = tf.constant(y)
    model.save_weights('weight_model.h5')

    # Iterate over folds
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}/{nr_splits}")

        X_train_fold, X_val_fold = tf.gather(X, train_index), tf.gather(X, val_index)
        y_train_fold, y_val_fold = tf.gather(y_labels_tensor, train_index), tf.gather(y_labels_tensor, val_index)

        # Train the model
        model.load_weights('weight_model.h5')

        history = model.fit(X_train_fold, y_train_fold, epochs=epoch_num, batch_size=16, callbacks=[callback], validation_data=(X_val_fold, y_val_fold))
        
        # Collect the history
        print('History:')
        print(history)
        print(history.history)
        histories.append(history.history)

        # Evaluate the model on the validation set
        y_pred_fold = model.predict(X_val_fold)
        y_pred_binary_fold = (y_pred_fold > 0.5).astype(int)

        # Calculate metrics for this fold
        accuracy_fold = accuracy_score(y_val_fold, y_pred_binary_fold.flatten())
        precision_fold = precision_score(y_val_fold, y_pred_binary_fold.flatten())
        recall_fold = recall_score(y_val_fold, y_pred_binary_fold.flatten())
        f1_fold = f1_score(y_val_fold, y_pred_binary_fold.flatten())

        # Append metrics to lists
        accuracy_list.append(accuracy_fold)
        precision_list.append(precision_fold)
        recall_list.append(recall_fold)
        f1_list.append(f1_fold)

    # Calculate average metrics across all folds
    average_accuracy = np.mean(accuracy_list)
    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)
    average_f1 = np.mean(f1_list)

    print(f'Average Accuracy: {average_accuracy:.4f}')
    print(f'Average Precision: {average_precision:.4f}')
    print(f'Average Recall: {average_recall:.4f}')
    print(f'Avearage F1: {average_f1:.4f}')

    # Plot the loss curves
    return histories


def train_evaluate(model, X_train, X_val, y_train, y_val, callback, epoch_num, batch_size, class_weights_dict):
    # Split the dataset into training and validation sets
    X_train = tf.constant(X_train)
    y_train_tensor = tf.constant(y_train)
    X_val = tf.constant(X_val)
    y_val_tensor = tf.constant(y_val)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epoch_num, batch_size=batch_size, class_weight=class_weights_dict, callbacks=callback, validation_split=0.2)

    # Evaluate the model on the validation set
    y_pred_val = model.predict(X_val)
    y_pred_binary_val = (y_pred_val > 0.5).astype(int)

    # Calculate metrics for this fold
    accuracy_val = accuracy_score(y_val_tensor, y_pred_binary_val.flatten())
    precision_val = precision_score(y_val_tensor, y_pred_binary_val.flatten())
    recall_val = recall_score(y_val_tensor, y_pred_binary_val.flatten())
    f1_val = f1_score(y_val_tensor, y_pred_binary_val.flatten())
    conf_matrix = confusion_matrix(y_val_tensor, y_pred_binary_val.flatten())

    print(f' Accuracy: {accuracy_val:.4f}')
    print(f' Precision: {precision_val:.4f}')
    print(f' Recall: {recall_val:.4f}')
    print(f' F1: {f1_val:.4f}')

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot the loss curves
    return history.history

def train_evaluate_2(model, X_train, X_val, y_train, y_val, callback, epoch_num, batch_size):
    
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # X_train = tf.constant(X_train)
    # y_train_tensor = tf.constant(y_train)
    # X_val = tf.constant(X_val)
    # y_val_tensor = tf.constant(y_val)
    # print(y_train_tensor)
    # Train the model

    history = model.fit(X_train, y_train, epochs=epoch_num, batch_size=batch_size, callbacks=callback, validation_split=0.2)

    # Evaluate the model on the validation set
    y_pred_val = model.predict(X_val)
    print(y_pred_val)
    y_pred_val = np.argmax(y_pred_val, axis=1)
    y_val = np.argmax(y_val, axis=1)
    # Calculate metrics for this fold
    accuracy_val = accuracy_score(y_val, y_pred_val)
    precision_val = precision_score(y_val, y_pred_val)
    recall_val = recall_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val)
    conf_matrix = confusion_matrix(y_val, y_pred_val)


    print(f' Accuracy: {accuracy_val:.4f}')
    print(f' Precision: {precision_val:.4f}')
    print(f' Recall: {recall_val:.4f}')
    print(f' F1: {f1_val:.4f}')

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


    # Plot the loss curves
    return history.history