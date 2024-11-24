import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# data preparation from project structure
base_path = "UCI HAR Dataset/"
train_features = base_path + "train/X_train.txt"
train_labels = base_path + "train/y_train.txt"
test_features = base_path + "test/X_test.txt"
test_labels = base_path + "test/y_test.txt"
activity_labels_path = base_path + "activity_labels.txt"

# loading the data
X_train = pd.read_csv(train_features, delim_whitespace=True, header=None).values
y_train = pd.read_csv(train_labels, header=None).values.ravel()
X_test = pd.read_csv(test_features, delim_whitespace=True, header=None).values
y_test = pd.read_csv(test_labels, header=None).values.ravel()

# encoding the activity labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# one-hot encoding for the labels
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

# Reshape the data for the spatial stream (1, 1, features)
X_train_spatial = X_train.reshape((X_train.shape[0], 1, 1, X_train.shape[1]))  # (samples, 1, 1, features)
X_test_spatial = X_test.reshape((X_test.shape[0], 1, 1, X_test.shape[1]))  # (samples, 1, 1, features)

# Reshape the data for the temporal stream (samples, timesteps, features)
X_train_temporal = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # (samples, timesteps, features)
X_test_temporal = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))  # (samples, timesteps, features)

# Define the spatial CNN stream
spatial_input = Input(shape=(1, 1, X_train.shape[1]))  # (1, 1, features)
spatial_conv = Conv2D(32, (1, 1), activation='relu')(spatial_input)
spatial_pool = MaxPooling2D(pool_size=(1, 1))(spatial_conv)
spatial_flatten = Flatten()(spatial_pool)

# Define the temporal LSTM stream
temporal_input = Input(shape=(1, X_train.shape[1]))  # (samples, timesteps, features)
temporal_lstm = LSTM(128, return_sequences=False)(temporal_input)
temporal_dense = Dense(64, activation='relu')(temporal_lstm)

# Merge the two streams
merged = Concatenate()([spatial_flatten, temporal_dense])

# Fully connected layers for classification
fc = Dense(64, activation='relu')(merged)
dropout = Dropout(0.5)(fc)
output = Dense(y_train_one_hot.shape[1], activation='softmax')(dropout)  # number of classes

# Create the model
model = Model(inputs=[spatial_input, temporal_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    [X_train_spatial, X_train_temporal], y_train_one_hot,
    epochs=10, batch_size=32, validation_data=([X_test_spatial, X_test_temporal], y_test_one_hot)
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_test_spatial, X_test_temporal], y_test_one_hot, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Printing confusion matrix
y_pred = np.argmax(model.predict([X_test_spatial, X_test_temporal]), axis=1)
from sklearn.metrics import classification_report, confusion_matrix
target_names = [str(cls) for cls in label_encoder.classes_]  # converting integer class names to strings
print(classification_report(y_test_encoded, y_pred, target_names=target_names))

# Plotting training and validation loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss for CNN Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting confusion matrix
import seaborn as sns

cm = confusion_matrix(y_test_encoded, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for CNN Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()