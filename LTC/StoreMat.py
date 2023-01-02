import scipy.io as scio


def load(datapath, label):

    data_origin = scio.loadmat(datapath)

    data = data_origin[label]

    return data


def store(path, data):

    scio.savemat(path, data)