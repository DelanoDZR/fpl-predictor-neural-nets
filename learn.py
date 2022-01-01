import tensorflow as tf


def train_and_evaluate(model, train_features, train_labels, test_features, test_labels, epochs):
    print("Training")
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam())

    history = model.fit(train_features, train_labels, epochs=epochs, callbacks=[callback])

    print("Evaluation")
    return model.evaluate(test_features, test_labels, verbose=2), history
