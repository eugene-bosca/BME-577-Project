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

# loading the data
X_train, y_train, X_test, y_test = load_dataset()
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

##### 1) LSTM MODEL #####
def lstm_model(X_train, y_train, X_test, y_test):
    # Define the LSTM Model
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(n_outputs, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Classification Report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    target_names = [str(cls) for cls in range(y_train.shape[1])]
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))

    # Plotting training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss for LSTM Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(y_train.shape[1]), yticklabels=range(y_train.shape[1]))
    plt.title('Confusion Matrix for LSTM Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

##### 2) CNN MODEL #####

def cnn_model():
    return

##### 3) CNN-LSTM MODEL #####
def cnn_lstm_model(X_train, y_train, X_test, y_test):

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
    target_names = [str(cls) for cls in range(y_train.shape[1])]
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))

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
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(y_train.shape[1]), yticklabels=range(y_train.shape[1]))
    plt.title('Confusion Matrix for CNN-LSTM Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# RUNNING THE MODELS
# lstm_model(X_train, y_train, X_test, y_test)

cnn_lstm_model(X_train, y_train, X_test, y_test)

