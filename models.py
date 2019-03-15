import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

def create_model_lstm():
    #### needs uniform hot vector matrix
    model = Sequential([
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_model_embedded_lstm():
    #### needs uniform hot vector matrix
    model = Sequential([
        keras.layers.Embedding(21,128),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model