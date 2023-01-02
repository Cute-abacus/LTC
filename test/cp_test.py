import tensorly as tl
import numpy as np
import numpy.ma as ma

from tensorly.decomposition import parafac

import os
import scipy.io as sio
import math
import time


def calc_err(measured, predicted, missed):
    # count non-zeros to filter all missing scenarios
    if np.count_nonzero(missed) == 0:
        return 0.
    # tensor with indices for training
    training_before = ma.masked_array(measured, mask=missed)
    training_after = ma.masked_array(predicted, mask=missed)

    # tensor with indices for inferring
    predicting_before = ma.masked_array(measured, mask=1 - missed)
    predicting_after = ma.masked_array(predicted, mask=1 - missed)

    # error measure function
    meas_func = lambda before, after: math.sqrt(np.sum((before - after) ** 2)) \
                                      / math.sqrt(np.sum(before ** 2))

    # calculate training error

    return meas_func(training_before, training_after), \
           meas_func(predicting_before, predicting_after)


def measure_ave_var(obj, times):
    if times <= 0:
        return 0., 0.

    sum_train_err, sum_predict_err = 0., 0.
    errs = []
    for i in range(times):
        errs.append(obj.run())

    ave_train_err1 = sum( e1[0] for e1,e2 in errs) / times
    ave_predict_err1 = sum( e2[0] for e1,e2 in errs) / times

    var_train_err1 = sum( (ave_train_err1 - e1[0]) ** 2 for e1,e2 in errs) / times
    var_predict_err1 = sum( (ave_predict_err1 - e2[0]) ** 2 for e1,e2 in errs) / times

    ave_train_err2 = sum( e1[1] for e1,e2 in errs) / times
    ave_predict_err2 = sum( e2[1] for e1,e2 in errs) / times

    var_train_err2 = sum( (ave_train_err2 - e1[1]) ** 2 for e1,e2 in errs) / times
    var_predict_err2 = sum( (ave_predict_err2 - e2[1]) ** 2 for e1,e2 in errs) / times
    return ([ave_train_err1, ave_train_err2], [var_train_err1, var_train_err2]), \
           ([ave_predict_err1, ave_predict_err2], [var_predict_err1, var_predict_err2])

def load_Abilene():
    # import Abilene dataset
    data_dir = os.path.join(os.getcwd(), '../data')
    mat_fname = os.path.join(data_dir, 'Abilene.mat')
    mat_contents = sio.loadmat(mat_fname)

    # data matrix Abilene as 'A'
    # A[i, j, k]
    # i: 288, 5-minutes time intervals of a day, 12*24 = 288
    # j: 169, 169 days of collected data
    # k: 144,  12 - 12 nodes ping matrix

    A = mat_contents['A']
    return A


class TensorDecompCP(object):
    """Tensor Decomposition class in CP framework with missing data"""
    def __init__(self, tensor, mask):
        self.tensor = tensor
        self.mask = mask

    def run(self):
        # Rank of the CP decomposition
        cp_rank = 2

        # Perform the CP decomposition
        # missed
        # missed entries:
        #           0 -- value missing
        #           1 -- value measured

        missed = self.mask  # Pr(0) = 0.4
        missed_A = self.tensor * missed
        w1, f1 = parafac(self.tensor, rank=cp_rank, init='random', n_iter_max=100, tol=10e-6, mask=missed)
        w2, f2 = parafac(missed_A, rank=cp_rank, init='random', n_iter_max=100, tol=10e-6)
        # Reconstruct the image from the factors
        cp_reconstruction1 = tl.kruskal_to_tensor((w1, f1))
        cp_reconstruction2 = tl.kruskal_to_tensor((w2, f2))

        train_error1, predict_error1 = calc_err(self.tensor, cp_reconstruction1, missed)
        train_error2, predict_error2 = calc_err(self.tensor, cp_reconstruction2, missed)

        return [train_error1, train_error2], [predict_error1, predict_error2]

    @classmethod
    def random_mask(cls, n, p):
        """
        Given an array, random fixed size 0s into 1s.
        :param n: Shape of array(Integer or Tuple)
        :param p: Probability of 0
        :return Mask: Shape n N-D array Mask
        """
        size = math.prod(n)
        mask = np.ones(size)
        K = int(size * p) # K 0s
        mask[:K] = 0
        np.random.shuffle(mask)
        mask = mask.reshape(n)
        return mask


if __name__ == "__main__":

    # load data
    A = load_Abilene()

    # config mask
    mask = TensorDecompCP.random_mask(A.shape, .4)

    TD_obj = TensorDecompCP(A, mask)

    start = time.perf_counter()
    times = 2
    train_res, predict_res = measure_ave_var(TD_obj, times)
    end = time.perf_counter()

    print(f'CP Decomposition in {times} times, '
          f'Mean Training Error(origin) is: {train_res[0][0]},'
          f' Variance of Training Error is:{train_res[1][0]}\n',
          f'Mean Training Error(masked) is: {train_res[0][1]},'
          f' Variance of Training Error is:{train_res[1][1]}')

    print(f'CP Decomposition in {times} times, '
          f'Mean Predicting Error(origin) is: {predict_res[0][0]},'
          f' Variance of Predicting Error is:{predict_res[1][0]}\n',
          f'Mean Predicting Error(masked) is: {predict_res[0][1]},'
          f' Variance of Predicting Error is:{predict_res[1][1]}')

    print(f"Time elapsed in {end - start :0.4f} seconds")
