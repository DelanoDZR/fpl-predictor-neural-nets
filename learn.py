import tensorflow as tf


def train_and_evaluate(model, train_features, train_labels, test_features, test_labels, epochs):
    print("Training")
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam())

    model.fit(train_features, train_labels, epochs=epochs)

    print("Evaluation")
    return model.evaluate(test_features, test_labels, verbose=2)
