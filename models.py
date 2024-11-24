import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_dataset

# # data preparation from project structure
base_path = "UCI HAR Dataset/"
# train_features = base_path + "train/X_train.txt"
train_labels = base_path + "train/y_train.txt"
# test_features = base_path + "test/X_test.txt"
test_labels = base_path + "test/y_test.txt"
# activity_labels_path = base_path + "activity_labels.txt"

# # loading the data
# X_train = pd.read_csv(train_features, delim_whitespace=True, header=None).values
y_train = pd.read_csv(train_labels, header=None).values.ravel()
# X_test = pd.read_csv(test_features, delim_whitespace=True, header=None).values
y_test = pd.read_csv(test_labels, header=None).values.ravel()

# print(f"X_train shape: {X_train.shape}")

# encoding the activity labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# # one-hot encoding for the labels
# y_train_one_hot = to_categorical(y_train_encoded)
# y_test_one_hot = to_categorical(y_test_encoded)

# # reshaping data for LSTM amd CNN (samples, timesteps, features)
# X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
# X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

##### 1) LSTM MODEL #####
def lstm_model(X_train, y_train_one_hot, X_test, y_test_one_hot, y_test_encoded):

    # defining the LSTM Model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])), 
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(y_train_one_hot.shape[1], activation='softmax')  # number of classes as output layer
    ])

    # plotting the model
    # plot_model(model, show_shapes=True, show_layer_names=True)

    # compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # training the model
    history = model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_test, y_test_one_hot))

    # evaluating the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # printing classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)

    target_names = [str(cls) for cls in label_encoder.classes_] # converting integer class names to strings
    print(classification_report(y_test_encoded, y_pred, target_names=target_names))

    # plotting training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss for LSTM Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plotting confusion matrix 
    cm = confusion_matrix(y_test_encoded, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix for LSTM Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

##### 2) CNN MODEL #####

def cnn_model():
    return

##### 3) CNN-LSTM MODEL #####
def cnn_lstm_model(X_train, y_train, X_test, y_test):
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    print(n_timesteps, n_features, n_outputs)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], n_features))

    # Define the CNN-LSTM model
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        # CNN layers
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Dropout(0.5),
        MaxPooling1D(pool_size=2),
        # LSTM layer
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')  # number of classes as output layer
    ])

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Evaluating the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Printing classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    target_names = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(y_test_encoded, y_pred, target_names=target_names))

    # Plotting training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss for CNN-LSTM Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix for CNN-LSTM Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# lstm_model(X_train, y_train_one_hot, X_test, y_test_one_hot, y_test_encoded)

# Needed the raw data as it had the timestepped data
X_train, y_train, X_test, y_test = load_dataset()
cnn_lstm_model(X_train, y_train, X_test, y_test)

