import data
import learn
import models


def main():
    train_features, train_labels, test_features, test_labels = data.get_data_sets("mids")

    model = models.mlp((None, 23))
    learn.train_and_evaluate(model, train_features, train_labels, test_features, test_labels, 100)

    print(train_features.shape)
    print(train_labels.shape)


if __name__ == "__main__":
    main()
