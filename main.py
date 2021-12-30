import data
import learn
import models
import plot
import result


def main():

    position = 'fwds'   # gks, defs, mids, fwds
    window = 9          # 3, 6, 9

    train_features, train_labels, test_features, test_labels = data.get_data_sets(position, window)

    results = []
    number_of_filters = [32, 64, 128, 256]
    kernel_sizes = []
    if window == 3:
        kernel_sizes = [1,2]
    elif window == 6:
        kernel_sizes = [1,2,3,4,5]
    elif window == 9:
        kernel_sizes = [1,2,3,4,5,6,7,8]

    for filter_count in number_of_filters:
        for kernel_size in kernel_sizes:
            print("Beginning processing of network with " +
                  str(filter_count) + " filters of kernel size " + str(kernel_size))
            model = models.cnn((None, 23), filter_count, kernel_size, position)
            mse, history = learn.train_and_evaluate(model, train_features, train_labels, test_features, test_labels, 100)
            results.append(result.Result(filter_count, kernel_size, mse))
            plot.plot_training(history, filter_count, kernel_size, position)

    plot.plot_evaluation(results, position)


if __name__ == "__main__":
    main()
