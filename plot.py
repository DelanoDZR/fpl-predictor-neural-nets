import matplotlib.pyplot as plt
import numpy as np

positions = {'gks': 'goalkeepers', 'defs': 'defenders', 'mids': 'midfielders', 'fwds': 'forwards'}


def plot_training(history, filter_count, kernel_size, position):
    plt.plot(history.history['loss'])
    plt.title('Training loss for ' + positions[position] + ' CNN with ' +
              str(filter_count) + ' filters ' +
              'and kernel size ' + str(kernel_size),
              loc='center', wrap=True)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('TLoss F ' + str(filter_count) + ' KS ' + str(kernel_size) + ' ' + position + '.png')
    plt.show()


def plot_evaluation(arr, position):
    labels = ['32', '64', '128', '256']
    size1 = []
    size2 = []
    size3 = []
    size4 = []
    size5 = []
    size6 = []
    size7 = []
    size8 = []


    for i in arr:
        if i.kernel_size == 1:
            size1.append(round(i.mse,3))
        elif i.kernel_size == 2:
            size2.append(round(i.mse,3))
        elif i.kernel_size == 3:
            size3.append(round(i.mse,3))
        elif i.kernel_size == 4:
            size4.append(round(i.mse,3))
        elif i.kernel_size == 5:
            size5.append(round(i.mse,3))
        elif i.kernel_size == 6:
            size6.append(round(i.mse,3))
        elif i.kernel_size == 7:
            size7.append(round(i.mse,3))
        elif i.kernel_size == 8:
            size8.append(round(i.mse,3))

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 4*(width), size1, width, label='1')
    rects2 = ax.bar(x - 3*(width), size2, width, label='2')
    rects3 = ax.bar(x - 2*(width), size3, width, label='3')
    rects4 = ax.bar(x - 1*(width), size4, width, label='4')
    rects5 = ax.bar(x , size5, width, label='5')
    rects6 = ax.bar(x + 1*(width), size6, width, label='6')
    rects7 = ax.bar(x + 2*(width), size7, width, label='7')
    rects8 = ax.bar(x + 3*(width), size8, width, label='8')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MSE')
    ax.set_xlabel('Number of Filters')
    ax.set_xticks(x, labels)
    ax.set_title("Mean squared error per combination of hyperparameter configurations for " + positions[position],
                 loc='center', wrap=True)
    ax.legend(title="Kernel Size")

    ax.bar_label(rects1, padding=2, rotation=90)
    ax.bar_label(rects2, padding=2, rotation=90)
    ax.bar_label(rects3, padding=2, rotation=90)
    ax.bar_label(rects4, padding=2, rotation=90)
    ax.bar_label(rects5, padding=2, rotation=90)
    ax.bar_label(rects6, padding=2, rotation=90)
    ax.bar_label(rects7, padding=2, rotation=90)
    ax.bar_label(rects8, padding=2, rotation=90)


    fig.tight_layout()

    plt.savefig('MSE Cluster ' + position +'.png')
    plt.show()

