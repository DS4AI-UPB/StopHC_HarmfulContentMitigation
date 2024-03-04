
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

def process_dataset(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # convert JSON 
    df = pd.DataFrame.from_dict(data, orient='index')

    # extract the english tweets
    df_en = df[df['lang'] == 'en']

    # define one label per tweet
    y_labels = df_en['labels_task1'].apply(lambda labels: int(labels.count('YES') > len(labels) / 2))

    return df_en, y_labels



def kfold_train_evaluate(model, X, y, nr_splits, callback, epoch_num):
    # Initialize K-Fold
    kfold = StratifiedKFold(n_splits=nr_splits, shuffle=True, random_state=42)
    
    accuracy_list = []
    precision_list = []
    recall_list = []

    X_tensor = tf.constant(X)
    y_tensor = tf.constant(y)

    # Iterate over folds
    for fold, (train_index_tensorX_tensor, val_index_tensorX_tensor) in enumerate(kfold.split(X_tensor, y_tensor)):
        print(f"Fold {fold + 1}/{nr_splits}")

        X_tensor_train_fold, X_tensor_val_fold = tf.gather(X_tensor, train_index_tensorX_tensor), tf.gather(X_tensor, val_index_tensorX_tensor)
        y_train_fold, y_val_fold = tf.gather(y_tensor, train_index_tensorX_tensor), tf.gather(y_tensor, val_index_tensorX_tensor)

        # Train the model
        model.fit(X_tensor_train_fold, y_train_fold, epochs=epoch_num, batch_size=32, callbacks=[callback])

        # evaluate the model
        y_pred_fold = model.predict(X_tensor_val_fold)
        y_pred_binary_fold = (y_pred_fold > 0.5).astype(int)

        accuracy_fold = accuracy_score(y_val_fold, y_pred_binary_fold.flatten())
        precision_fold = precision_score(y_val_fold, y_pred_binary_fold.flatten())
        recall_fold = recall_score(y_val_fold, y_pred_binary_fold.flatten())

        accuracy_list.append(accuracy_fold)
        precision_list.append(precision_fold)
        recall_list.append(recall_fold)

    # Calculate average metrics across all folds
    average_accuracy = np.mean(accuracy_list)
    average_precision = np.mean(precision_list)
    average_recall = np.mean(recall_list)

    print(f'Average Accuracy: {average_accuracy:.4f}')
    print(f'Average Precision: {average_precision:.4f}')
    print(f'Average Recall: {average_recall:.4f}')