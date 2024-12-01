import numpy as np
import pandas as pd
from pandas import read_csv
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_file(filepath):
    """Load a file as a numpy array."""
    return read_csv(filepath, header=None, delim_whitespace=True).values

def load_data(group, prefix=''):
    """Load data for a given group (train or test)."""
    signals = [
        f'{signal}_{group}.txt' for signal in 
        ['total_acc_x', 'total_acc_y', 'total_acc_z',
         'body_acc_x', 'body_acc_y', 'body_acc_z',
         'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    ]
    # Stack the signal data as features
    X = np.dstack([load_file(f"{prefix}{group}/Inertial Signals/{signal}") for signal in signals])
    y = load_file(f"{prefix}{group}/y_{group}.txt") - 1  # Convert labels to zero-based
    return X, to_categorical(y)

def load_timestepped_data(prefix='UCI HAR Dataset/'):
    """Load train and test datasets."""
    X_train, y_train = load_data('train', prefix)
    X_test, y_test = load_data('test', prefix)

    # get the labels
    base_path = "UCI HAR Dataset/"
    train_labels = base_path + "train/y_train.txt"
    test_labels = base_path + "test/y_test.txt"
    y_train_encode = pd.read_csv(train_labels, header=None).values.ravel()
    y_test_encode = pd.read_csv(test_labels, header=None).values.ravel()

    # encoding the activity labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_encode)
    y_test_encoded = label_encoder.transform(y_test_encode)
    
    return X_train, y_train, X_test, y_test, y_train_encoded, y_test_encoded