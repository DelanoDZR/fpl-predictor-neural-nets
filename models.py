import tensorflow as tf


def mlp(shape):
    return tf.keras.models.Sequential([
        tf.keras.Input(shape=shape),
        tf.keras.layers.Dense(36, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])


def lstm(shape):
    return tf.keras.models.Sequential([
        tf.keras.Input(shape=shape),
        tf.keras.layers.LSTM(36, return_sequences=True),
        tf.keras.layers.Dense(units=1, activation='relu')
    ])


def cnn(shape):
    return tf.keras.Sequential([
        tf.keras.Input(shape=shape),
        tf.keras.layers.Conv1D(filters=23,
                               kernel_size=3,
                               input_shape=shape),
        tf.keras.layers.MaxPool1D(input_shape=shape),
        tf.keras.layers.Dense(units=12, activation='relu'),
        tf.keras.layers.Dense(units=1,input_shape=shape),
    ])
