import data
import learn
import models
import plot


class Result:
    def __init__(self, fil, ker, mse):
        self.filters =  fil
        self.kernel_size = ker
        self. mse = mse


def main():

    #positions = {'GK': 'gks'} #, 'DEF': 'defs', 'MID': 'mids', 'FWD': 'fwds'}
    arr = []
    arr2 = []
    train_features, train_labels, test_features, test_labels = data.get_data_sets('gks')

    filters = [32, 64, 128, 256]
    kernel_size = [1,2]
    for i in filters:
        for j in kernel_size:

            model = models.cnn((None, 23), i, j)
            mse, history = learn.train_and_evaluate(model, train_features, train_labels, test_features, test_labels, 100)
            arr.append("Filters = " + str(i) + "\n" +
                       "Kernel_Size = " + str(j) + "\n" +
                       "MSE = " + str(mse) + "\n\n")
            arr2.append(Result(i, j, mse))
            plot.plotTraining(history)


    for z in arr:
        print(z)

    print()
    print(arr2[0].filters)



    #print(train_features.shape)
    #print(train_labels.shape)


if __name__ == "__main__":
    main()
