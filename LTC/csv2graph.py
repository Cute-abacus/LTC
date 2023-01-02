import matplotlib.pyplot as plot

import numpy as np

import csv

from scipy import optimize


def extract_db(data):
    """Extract h, q ,alpha parameter tuples from datasheet
    Distance mode in another item
    """
    db = {}
    dis_mode = None
    for row in data:

        if not dis_mode: # get if dis_mode have value
            dis_mode = row[0]
        # strip first one
        row = [ round(float(cell), 8) for cell in row[1:]]
        params_tuple = (row[0], row[1], row[2])
        if params_tuple in db:
            if type(params_tuple) is type(()):
                db[params_tuple].append(row[3:])
            else:
                print("[Err] db table cell is not tuple")
                return
        else:
            db[params_tuple] = [row[3:], ]

    if dis_mode == 'eu':
        dis_mode = 'Euclidean'
    elif dis_mode == 'an':
        dis_mode = 'Angular'
    else:
        dis_mode = 'Unknown'

    return db, dis_mode


def render_h(data, low, high, def_params):
    """
    Render csv data into box graph
    csv format
    h, q, alpha, CP_tr, CP_pr, LTC_tr, LTC_pr
    """


    # 1. map same params result into same buckets

    db, dis_mode = extract_db(data)

    # 2. list sequences by params

    # h in [.1, 1.] , q is 30, alpha is .5
    params = np.arange(low, high, .1)
    params = np.round(params, 8)

    train_sequence_list = []
    predict_sequence_list = []

    for param in params:
        h = param
        q = def_params[1]
        alpha = def_params[2]

        train_sequence = [res[2] for res in db[(h, q, alpha)]]
        train_sequence_list.append(train_sequence)
        predict_sequence = [res[3] for res in db[(h, q, alpha)]]
        predict_sequence_list.append(predict_sequence)


    # subplot
    fig, ax = plot.subplots(1, 2)

    # left
    x_lab = 'h'
    x = list(range(1, len(train_sequence_list) + 1))
    y = [np.mean(seq) for seq in train_sequence_list]

    color = 'green' if dis_mode == 'Euclidean' else 'red'
    ax[0].plot(x, y, color=color, marker=">")
    ax[0].legend(['LTC({})'.format(dis_mode),], loc='upper center', bbox_to_anchor=(.25,1.02,.5,.1))
    ax[0].boxplot(train_sequence_list, showmeans=False)

    # right
    x = list(range(1, len(predict_sequence_list) + 1))
    y = [np.mean(seq) for seq in predict_sequence_list]

    ax[1].plot(x, y, color=color, marker='>')
    ax[1].legend(['LTC({})'.format(dis_mode),], loc='upper center', bbox_to_anchor=(.25,1.02,.5,.1))
    ax[1].boxplot(predict_sequence_list, showmeans=False)

    y_lab = "Error in Sampling"
    x_ticks = [round(param, 3) for param in params]
    x_ticks_fix = ['']*len(x_ticks)
    for i in range(0, len(x_ticks_fix), 10):
        x_ticks_fix[i] = x_ticks[i] -1


    plot.setp(ax, xlabel=x_lab)
    plot.setp(ax[0], ylabel=y_lab)
    plot.setp(ax[0], xticklabels=x_ticks)

    y_lab = "Error in Prediction"
    plot.setp(ax[1], ylabel=y_lab)
    plot.setp(ax[1], xticklabels=x_ticks)

    plot.show()

def render_alpha(data, low, high, def_params):
    """
    Render csv data into box graph
    csv format
    h, q, alpha, CP_tr, CP_pr, LTC_tr, LTC_pr
    """

    # 1. map same params result into same buckets
    db, dis_mode = extract_db(data)

    # 2. list sequences by params

    # alpha in [.2, .9] , h is .5, alpha is .5
    params = np.arange(.2, .9, .1)
    params = np.round(params, 8)

    train_sequence_list = []
    predict_sequence_list = []

    for param in params:
        h = def_params[0]
        q = def_params[1]
        alpha = param

        train_sequence = [res[2] for res in db[(h, q, alpha)]]
        train_sequence_list.append(train_sequence)
        predict_sequence = [res[3] for res in db[(h, q, alpha)]]
        predict_sequence_list.append(predict_sequence)


    # subplot
    fig, ax = plot.subplots(1, 2)

    # left
    x_lab = r'$\alpha$'
    x = list(range(1, len(train_sequence_list) + 1))
    y = [np.mean(seq) for seq in train_sequence_list]

    def fitting(x, a, b, c):
        return a*x*x + b*x + c

    fit_params, fit_params_cov = optimize.curve_fit(fitting, x, y)
    fitting_y = [ fitting(i, fit_params[0], fit_params[1], fit_params[2]) for i in x]

    ax[0].plot(x, fitting_y, color='gray', linestyle='--', marker='*')
    ax[0].legend(['LTC(fitting)',], loc='upper center', bbox_to_anchor=(.25,1.02,.5,.1))
    ax[0].boxplot(train_sequence_list, showmeans=True)

    # right
    x = list(range(1, len(predict_sequence_list) + 1))
    y = [np.mean(seq) for seq in predict_sequence_list]
    fit_params, fit_params_cov = optimize.curve_fit(fitting, x, y)
    fitting_y = [ fitting(i, fit_params[0], fit_params[1], fit_params[2]) for i in x]
    ax[1].plot(x, fitting_y, color='gray', linestyle='--', marker='*')
    ax[1].legend(['LTC(fitting)',], loc='upper center', bbox_to_anchor=(.25,1.02,.5,.1))
    ax[1].boxplot(predict_sequence_list, showmeans=True)

    y_lab = "Error in Sampling"
    x_ticks = [round(param, 3) for param in params]
    x_ticks_fix = ['']*len(x_ticks)
    for i in range(0, len(x_ticks_fix), 10):
        x_ticks_fix[i] = x_ticks[i] -1


    plot.setp(ax, xlabel=x_lab)
    plot.setp(ax[0], ylabel=y_lab)
    plot.setp(ax[0], xticklabels=x_ticks)

    y_lab = "Error in Prediction"
    plot.setp(ax[1], ylabel=y_lab)
    plot.setp(ax[1], xticklabels=x_ticks)

    plot.show()

def render_q(data, low, high, def_params):
    """
    Render csv data into box graph
    csv format
    h, q, alpha, CP_tr, CP_pr, LTC_tr, LTC_pr
    """

    # 1. map same params result into same buckets

    db, dis_mode = extract_db(data)

    # 2. list sequences by params

    # q in [1, 49] , h is .5, alpha is .5
    params = range(low, high)

    train_sequence_list = []
    predict_sequence_list = []

    cp_train_sequence_list = []
    cp_predict_sequence_list = []

    for param in params:
        h = def_params[0]
        q = param
        alpha = def_params[2]

        # seq of LTC
        train_sequence = [res[2] for res in db[(h, q, alpha)]]
        train_sequence_list.append(train_sequence)
        predict_sequence = [res[3] for res in db[(h, q, alpha)]]
        predict_sequence_list.append(predict_sequence)

        # seq of CP
        cp_train_sequence = [res[0] for res in db[(h, q, alpha)]]
        cp_train_sequence_list.append(cp_train_sequence)
        cp_predict_sequence = [res[1] for res in db[(h, q, alpha)]]
        cp_predict_sequence_list.append(cp_predict_sequence)

    # 3. subplots
    fig, ax = plot.subplots(1, 2)

    # left plot
    x_lab = "q"
    x = list(range(1, len(train_sequence_list) + 1))
    y = [np.mean(seq) for seq in train_sequence_list]
    ax[0].plot(x, y, color='red', marker=">", label='LTC-CP') # line of LTC

    y = [np.mean(seq) for seq in cp_train_sequence_list]
    ax[0].plot(x, y, color='green', marker="*", label='CP') # line of CP
    ax[0].legend(loc="upper center", bbox_to_anchor=(.15, 1.02, .7, 0.1),
                 mode='expand', borderaxespad=0, ncol=3)

    # right
    x = list(range(1, len(predict_sequence_list) + 1))
    y = [np.mean(seq) for seq in predict_sequence_list]
    ax[1].plot(x, y, color='red', marker=">", label='LTC-CP') # line of LTC

    y = [np.mean(seq) for seq in cp_predict_sequence_list]
    ax[1].plot(x, y, color='green', marker="*", label='CP') # line of CP
    ax[1].legend(loc="upper center", bbox_to_anchor=(.15, 1.02, .7, 0.1),
                 mode='expand', borderaxespad=0, ncol=3)


    # axis
    y_lab = "Error in Sampling"
    x_ticks = [round(param, 3) for param in params]
    x_ticks_fix = ['']*len(x_ticks)
    for i in range(0, len(x_ticks_fix), 10):
        x_ticks_fix[i] = x_ticks[i] -1


    plot.setp(ax, xlabel=x_lab)
    plot.setp(ax[0], ylabel=y_lab)

    y_lab = "Error in Prediction"
    plot.setp(ax[1], ylabel=y_lab)

    plot.show()

if __name__ == "__main__":
    path = '../data2/data.csv'
    with open(path) as infile:
        reader = csv.reader(infile, delimiter=',')
        data = []
        for row in reader:
            # round str to 8 precision float
            new_row = [ str(cell) for cell in row]
            # new_row = [ round(float(cell), 8) for cell in row]
            data.append(new_row)

        params = [
            .6, # h
            20, # q
            .7, # alpha
        ]
        # render_h(data, low=.3, high=1.01, def_params=params)
        render_alpha(data, low=.2, high=.9, def_params=params)
        # render_q(data, low=1, high=31, def_params=params)
