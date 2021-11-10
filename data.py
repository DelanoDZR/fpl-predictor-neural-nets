import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_data():
    dataset = pd.read_csv(
        "data/by_gameweek/mids\\gw1.csv", sep=",", index_col=0)
    return dataset

def split_data(dataset):



    train, test = train_test_split(dataset.copy(), test_size=0.2)
    train.pop('name')
    test.pop('name')

    train_labels = train.pop('Target_Output')
    train_features = np.array([train.to_numpy()], dtype=np.float)
    train_labels = np.array([train_labels], dtype=np.float)

    test_labels = test.pop('Target_Output')
    test_features = np.array([test.to_numpy()], dtype=np.float)
    test_labels = np.array([test_labels], dtype=np.float)

    return train_features, train_labels, test_features, test_labels


def get_data_sets():
    return split_data(read_data())
