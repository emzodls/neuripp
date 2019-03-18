import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

def create_model_conv():
    model = Sequential([
        keras.layers.Input(shape=(120, 20)),
        keras.layers.Conv1D(filters=75,
               kernel_size=5,
               padding='valid',
               activation='relu',
               kernel_initializer='glorot_normal',
               strides=1),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(filters=75,
                            kernel_size=3,
                            padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            strides=1),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(filters=150,
                            kernel_size=3,
                            padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            strides=1),
        keras.layers.Dropout(0.1),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_model_conv_lstm():
    model = Sequential([
        keras.layers.Input(shape=(120, 20)),
        keras.layers.Conv1D(filters=75,
               kernel_size=5,
               padding='valid',
               activation='relu',
               kernel_initializer='glorot_normal',
               strides=1),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(filters=75,
                            kernel_size=3,
                            padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            strides=1),
        keras.layers.Dropout(0.1),
        keras.layers.Conv1D(filters=150,
                            kernel_size=3,
                            padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal',
                            strides=1),
        keras.layers.Dropout(0.1),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False,
                                                     dropout=0.15, recurrent_dropout=0.15, implementation=0)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_model_lstm():
    #### needs uniform hot vector matrix
    model = Sequential([
        keras.layers.Input(shape=(120, 20)),
        keras.layers.Masking(mask_value=0., input_shape=(120, 20)),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False,
                                                     dropout=0.15, recurrent_dropout=0.15, implementation=0)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_model_embedded_lstm():
    #### needs uniform hot vector matrix
    model = Sequential([
        keras.layers.Embedding(21,128),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False, dropout=0.15,
                                                     recurrent_dropout=0.15, implementation=0)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model