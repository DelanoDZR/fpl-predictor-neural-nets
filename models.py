import tensorflow as tf


def mlp(shape):
    return tf.keras.models.Sequential([
        tf.keras.Input(shape=shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])


def lstm(shape):
    return tf.keras.models.Sequential([
        tf.keras.Input(shape=shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='relu')
    ])


def cnn(shape, filters, kernel_size):
    return tf.keras.Sequential([
        tf.keras.Input(shape=shape),
        tf.keras.layers.Conv1D(filters=filters,
                               kernel_size=kernel_size,
                               input_shape=shape),
        tf.keras.layers.MaxPool1D(input_shape=shape),
        tf.keras.layers.Dense(units=64,input_shape=shape),
        tf.keras.layers.Dense(units=1,input_shape=shape),
    ])
