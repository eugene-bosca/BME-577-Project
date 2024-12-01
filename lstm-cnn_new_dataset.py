import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('Time_domain_subsamples/Time_domain_subsamples/KU-HAR_v1.0_raw_samples.csv', header=None)

# Extract features and labels
X = data.iloc[:, :-3].values  # Exclude last 3 columns (Class ID, Length, Serial No.)
y = data.iloc[:, -3].values   # Class ID (column 9001)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Reshape features for time-series input
n_samples, n_features = X.shape
n_axes = 6  # AccX, AccY, AccZ, GyroX, GyroY, GyroZ
n_timesteps = n_features // n_axes  # Time steps per axis

X = X.reshape((n_samples, n_timesteps, n_axes))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN-LSTM model
model = Sequential([
    Input(shape=(n_timesteps, n_axes)),
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
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()