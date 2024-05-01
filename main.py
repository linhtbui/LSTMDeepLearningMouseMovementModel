import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns

def load_session_data(base_path, labels=None, is_training=True):
    data_list = []
    for user_folder in [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]:
        user_path = os.path.join(base_path, user_folder)
        for session_file in os.listdir(user_path):
            file_path = os.path.join(user_path, session_file)
            # Extract session ID from filename (assuming no file extension)
            session_id = session_file.replace('session_', '').split('.')[0]
            if os.path.isfile(file_path):
                try:
                    session_data = pd.read_csv(file_path, header=None,
                                               names=['record timestamp', 'client timestamp', 'button', 'state', 'x',
                                                      'y'])
                    session_data['session_id'] = session_id
                    data_list.append(session_data)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
    all_data = pd.concat(data_list, ignore_index=True)
    if not is_training:
        all_data = all_data.merge(labels, on='session_id', how='left')
    else:
        all_data['is_illegal'] = 0
    return all_data

def preprocess_data(data):
    scaler = StandardScaler()
    try:
        # Convert 'client timestamp' to numeric
        data['client timestamp'] = pd.to_numeric(data['client timestamp'], errors='coerce')
        data['button'] = pd.to_numeric(data['button'], errors='coerce')
        data['state'] = pd.to_numeric(data['state'], errors='coerce')

        # Convert other columns to numeric and handle possible conversion errors
        numeric_cols = ['record timestamp', 'x', 'y']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Check if required columns are present
        required_cols = ['session_id', 'record timestamp', 'x', 'y']
        if not set(required_cols).issubset(data.columns):
            raise ValueError("Required columns are missing in the DataFrame.")

        # Sorting by session ID and timestamp
        data.sort_values(by=['session_id', 'record timestamp'], inplace=True)

        # Calculate derivatives and distances if the required columns are present
        if 'x_diff' not in data.columns:
            data['x_diff'] = data.groupby('session_id')['x'].diff().fillna(0).astype(float)
        if 'y_diff' not in data.columns:
            data['y_diff'] = data.groupby('session_id')['y'].diff().fillna(0).astype(float)
        if 'time_diff' not in data.columns:
            data['time_diff'] = data.groupby('session_id')['record timestamp'].diff().fillna(0).astype(float)

        data['dx_dt'] = data['x_diff'] / data['time_diff'].replace(0, np.nan)
        data['dy_dt'] = data['y_diff'] / data['time_diff'].replace(0, np.nan)

        # Calculate other features
        data['distance'] = np.sqrt(data['x_diff'] ** 2 + data['y_diff'] ** 2)
        data['velocity'] = (data['distance'] / data['time_diff'].replace(0, np.nan)).fillna(0)
        data['acceleration'] = (data['velocity'].diff().fillna(0) / data['time_diff'].replace(0, np.nan)).fillna(0)
        data['angle'] = np.arctan2(data['y_diff'], data['x_diff']).fillna(0)

        # Standardize features
        features = ['distance', 'velocity', 'acceleration', 'angle', 'dx_dt', 'dy_dt']
        data[features] = scaler.fit_transform(data[features])

        # Remove any remaining non-finite values
        data.fillna(0, inplace=True)
    except Exception as e:
        print("Error in preprocessing data:", e)

    return data

def plot_loss(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_accuracy(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def build_model(input_shape, N=1, hidden_size=50):
    model = Sequential()
    optimizer = Adam(clipvalue=1.0)

    # Add N LSTM layers with ReLU activation
    for _ in range(N):
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=input_shape))
        model.add(Activation('relu'))

    # Add a fully connected layer to reduce dimensionality to 1
    model.add(Dense(1))

    # Add a sigmoid activation layer for binary classification
    model.add(Activation('sigmoid'))

    # Compile the model with the new optimizer
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def check_data(train_features, test_features):
    # Check data types
    train_types = np.unique([str(type(item)) for sublist in train_features for item in sublist])
    test_types = np.unique([str(type(item)) for sublist in test_features for item in sublist])
    print("Unique data types in train_features:", train_types)
    print("Unique data types in test_features:", test_types)

    # Check for finite values and print results
    if np.issubdtype(train_features.dtype, np.floating) and np.issubdtype(test_features.dtype, np.floating):
        print("Train features finite:", np.all(np.isfinite(train_features)))
        print("Test features finite:", np.all(np.isfinite(test_features)))
    else:
        print("Data types:", train_features.dtype, test_features.dtype)
        print("Data contains non-floating types or conversion was unsuccessful.")

    # Check for any NaNs
    print("Train features NaNs:", np.any(np.isnan(train_features)))
    print("Test features NaNs:", np.any(np.isnan(test_features)))

def visualize_features(train_data, test_data):
    features = train_data.columns

    # Plot histograms for each feature in the training dataset
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(train_data[feature], kde=True, color='blue', label='Train')
        sns.histplot(test_data[feature], kde=True, color='red', label='Test')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
def main():
    labels = pd.read_csv('Mouse-Dynamics-Challenge-master/public_labels.csv')
    # Preprocess the labels DataFrame
    labels['session_id'] = labels['filename'].str.replace('session_', '').str.split('.').str[0]
    labels.drop('filename', axis=1, inplace=True)

    train_path = 'Mouse-Dynamics-Challenge-master/training_files'
    test_path = 'Mouse-Dynamics-Challenge-master/test_files'
    train_data = load_session_data(train_path, is_training=True)
    test_data = load_session_data(test_path, labels, is_training=False)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    visualize_features(train_data, test_data)

    train_features = np.array(train_data.drop(['session_id', 'is_illegal'], axis=1))
    train_labels = train_data['is_illegal'].values
    test_features = np.array(test_data.drop(['session_id', 'is_illegal'], axis=1))
    test_labels = test_data['is_illegal'].values

    check_data(train_features, test_features)  # Add this line to check data types

    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels,
                                                                              test_size=0.2, random_state=42)

    train_features = train_features.reshape((train_features.shape[0], 1, train_features.shape[1]))
    val_features = val_features.reshape((val_features.shape[0], 1, val_features.shape[1]))
    test_features = test_features.reshape((test_features.shape[0], 1, test_features.shape[1]))

    train_labels = train_labels.reshape(-1, 1)
    val_labels = val_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)

    # Define the parameters for the modified model
    N = 2  # Number of LSTM layers
    hidden_size = 50  # Hidden size for LSTM layers

    # Build the model with the specified parameters
    model = build_model((train_features.shape[1], train_features.shape[2]), N=N, hidden_size=hidden_size)

    plot_model(model, to_file='model_architecture.png', show_shapes=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    history = model.fit(train_features, train_labels, epochs=10, validation_data=(val_features, val_labels),
              batch_size=32, callbacks=[reduce_lr])


    test_loss, test_accuracy = model.evaluate(test_features, test_labels)
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
