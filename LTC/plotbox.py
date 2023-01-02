import matplotlib.pyplot as plot

import numpy as np

import json


def render(data, param_name='h'):
    # Generate the data
    train_sequence_list = []
    predict_sequence_list = []
    params = []
    for test_set in data:
        # testing q (LTC)
        tr_seq =[]
        pr_seq =[]
        param = test_set['params'][param_name]
        for run in test_set['data']:
            tr_seq.append(run['LTC']['MeanTrErr'])
            pr_seq.append(run['LTC']['MeanPrErr'])

        train_sequence_list.append(tr_seq)
        predict_sequence_list.append(pr_seq)
        params.append(param)


    #subplot
    fig, ax = plot.subplots(1,2)

    #left
    x_lab = param_name
    x = list(range(1, len(train_sequence_list)+1))
    y = [np.mean(seq) for seq in train_sequence_list]
    ax[0].plot(x,y)
    ax[0].boxplot(train_sequence_list, showmeans=True)

    #right
    x = list(range(1, len(predict_sequence_list)+1))
    y = [np.mean(seq) for seq in predict_sequence_list]
    ax[1].plot(x,y)

    ax[1].boxplot(predict_sequence_list, showmeans=True)
    y_lab = "Error in Sampling"
    x_ticks = [round(param, 3) for param in params]

    plot.setp(ax, xlabel=x_lab)
    plot.setp(ax[0], ylabel=y_lab)
    plot.setp(ax[0], xticklabels=x_ticks)

    y_lab = "Error in Prediction"
    plot.setp(ax[1], ylabel=y_lab)
    plot.setp(ax[1], xticklabels=x_ticks)

    # tiks
    ticks = list(range(len(train_sequence_list)))

    # plot.setp(ax, xticks=params)

    plot.show()


if __name__ == "__main__":

    path = '../data/'
    with open(path + "printAll.json") as infile:
        data = json.load(infile)
        render(data)
