import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_timestepped_data

label_encoder = LabelEncoder()

def cnn_2d_model(X_train, y_train, X_test, y_test, y_test_encoded):
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    print(n_timesteps, n_features, n_outputs)

    # Reshape data for 2D convolutions
    X_train = X_train.reshape((X_train.shape[0], n_timesteps, n_features, 1))
    X_test = X_test.reshape((X_test.shape[0], n_timesteps, n_features, 1))

    # Define the CNN model with 2D convolutions
    model = Sequential([
        Input(shape=(n_timesteps, n_features, 1)),
        # CNN layers
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')  # Number of classes as output layer
    ])

    # Compiling the model
    model.compile(optimizer= Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    # Training the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), callbacks=[reduce_lr])

    # Evaluating the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Predict and generate classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    target_names = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(y_test_encoded, y_pred, target_names=target_names))

    # Plotting training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss for 2D CNN Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix for 2D CNN Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Specify the folder where figures will be saved
    output_folder = "cnn_figures"
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Save the training and validation loss figure
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss for 2D CNN Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_fig_path = os.path.join(output_folder, 'training_validation_loss.png')
    plt.savefig(loss_fig_path, dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory

    # Save the confusion matrix figure
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix for 2D CNN Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_fig_path = os.path.join(output_folder, 'confusion_matrix.png')
    plt.savefig(cm_fig_path, dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory

    print(f"Figures saved in folder: {output_folder}")


# Needed the raw data as it had the timestepped data
X_train, y_train, X_test, y_test, y_train_encoded, y_test_encoded = load_timestepped_data()
cnn_2d_model(X_train, y_train, X_test, y_test, y_test_encoded)
