import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_timestepped_data
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

label_encoder = LabelEncoder()

def cnn_lstm_model(X_train, y_train, X_test, y_test):
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    print(n_timesteps, n_features, n_outputs)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_train.shape[1], n_features))

    # Define the CNN-LSTM model
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        # CNN layers
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.24714392468145407),
        MaxPooling1D(pool_size=2),
        # LSTM layer
        LSTM(256, return_sequences=True),
        Dropout(0.24714392468145407),
        LSTM(256//2, return_sequences=False),
        Dropout(0.24714392468145407),
        Dense(128, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')  # number of classes as output layer
    ])

    optimizer = Adam(learning_rate=0.0009094598593981077)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Training the model
    history = model.fit(
        X_train, y_train,
        epochs=50,  # early stopping shoudl take effect before 50 epochs
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stopping]
    )
    # Evaluating the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Printing classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    model.save('cnn_lstm_model.h5')
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

X_train, y_train, X_test, y_test, y_train_encoded, y_test_encoded = load_timestepped_data()
cnn_lstm_model(X_train, y_train, X_test, y_test)