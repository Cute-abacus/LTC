"""Localized Tensor Completion helping functionality"""
import tensorly as tl
import numpy as np
import numpy.ma as ma

from tensorly.decomposition import parafac

import os
import scipy.io as sio
import math
import time


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

    def __init__(self, tensor, mask, rank=1):
        self.tensor = tensor
        self.mask = mask

        # Rank of the CP decomposition
        self.rank = rank

    def run(self):

        # Perform the CP decomposition
        # missed
        # missed entries:
        #           0 -- value missing
        #           1 -- value measured

        missed_A = self.tensor * self.mask
        self.weights, self.factors = parafac(
            self.tensor, rank=self.rank, init='random', n_iter_max=100, tol=10e-6, mask=self.mask)

        # Reconstruct the image from the factors
        self.cp_reconstruction = tl.kruskal_to_tensor((self.weights, self.factors))

    def calc_err(self):
        """Measure performance of a tensor decomposition"""

        # perform CP
        self.run()

        measured = self.tensor
        predicted = self.cp_reconstruction
        missed = self.mask

        # count non-zeros to filter all missing scenarios
        if np.count_nonzero(missed) == 0:
            return 0.
        # tensor with indices for training
        training_before = ma.masked_array(measured, mask=1 - missed)
        training_after = ma.masked_array(predicted, mask=1 - missed)

        # tensor with indices for inferring
        predicting_before = ma.masked_array(measured, mask=missed)
        predicting_after = ma.masked_array(predicted, mask=missed)

        # error measure function
        meas_func = lambda before, after: math.sqrt(np.sum((before - after) ** 2)) \
                                          / math.sqrt(np.sum(before ** 2))

        # calculate training error

        return meas_func(training_before, training_after), \
               meas_func(predicting_before, predicting_after)

    def measure_ave_var(self, times):
        if times <= 0:
            return 0., 0.

        sum_train_err, sum_predict_err = 0., 0.
        errs = []
        for i in range(times):
            errs.append(self.calc_err())

        ave_train_err = sum(e1 for e1, e2 in errs) / times
        ave_predict_err = sum(e2 for e1, e2 in errs) / times

        var_train_err = sum((ave_train_err - e1) ** 2 for e1, e2 in errs) / times
        var_predict_err = sum((ave_predict_err - e2) ** 2 for e1, e2 in errs) / times
        return (ave_train_err, var_train_err), \
               (ave_predict_err, var_predict_err)

    @classmethod
    def random_mask(cls, n, p):
        """
        Given an array, random fixed size 1s into 0s.
        :param n: Shape of array(Integer or Tuple)
        :param p: Probability of 0
        :return mask: Shape n N-D array Mask
        """
        size = math.prod(n)
        mask = np.zeros(size)
        K = int(size * p)  # K 0s
        mask[:K] = 1
        np.random.shuffle(mask)
        mask = mask.reshape(n)
        return mask


if __name__ == "__main__":
    # load data
    A = load_Abilene()

    # config mask
    mask = TensorDecompCP.random_mask(A.shape, .4) #Pr(0) =0.6

    TD_obj = TensorDecompCP(A, mask, rank=2)

    start = time.perf_counter()
    times = 40
    train_res, predict_res = TD_obj.measure_ave_var(times)
    end = time.perf_counter()

    print(f'CP Decomposition in {times} times, '
          f'Mean Training Error is: {train_res[0]:.5f}, Variance of Training Error is:{train_res[1]:.5f}')
    print(f'CP Decomposition in {times} times,'
          f' Mean Predicting Error is: {predict_res[0]:.5f}, Variance of Predicting Error is:{predict_res[1]:.5f}')
    print(f"Time elapsed in {end - start :0.4f} seconds")
