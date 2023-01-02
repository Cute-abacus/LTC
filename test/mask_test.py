import numpy as np
from tensorly.decomposition import parafac
import tensorly as tl
import math
import numpy.ma as ma


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

    return meas_func(training_before, training_after),\
           meas_func(predicting_before, predicting_after)


def run():
    A = tl.tensor(np.arange(24).reshape((3, 4, 2)), dtype=tl.float32)
    # Rank of the CP decomposition
    cp_rank = 1

    # Perform the CP decomposition
    # missed
    # missed entries:
    #           0 -- value missing
    #           1 -- value measured

    missed = np.random.binomial(1, 0.4, A.shape)  # Pr(1) = 0.4, i.e Pr(0) = 0.6
    missed_A = A * missed
    weights, factors = parafac(missed_A, rank=cp_rank, init='random', tol=10e-6, mask=missed)
    # Reconstruct the image from the factors
    cp_reconstruction = tl.kruskal_to_tensor((weights, factors))

    train_error, predict_error = calc_err(A, cp_reconstruction, missed)

    return train_error, predict_error


def measure_ave_var(times):
    if times <= 0:
        return 0., 0.

    sum_train_err, sum_predict_err = 0., 0.
    errs = []
    for i in range(times):
        errs.append(run())

    ave_train_err = sum( e1 ** 2 for e1,e2 in errs) / times
    ave_predict_err = sum( e2 ** 2 for e1,e2 in errs) / times

    var_train_err = sum( (ave_train_err - e1) ** 2 for e1,e2 in errs) / times
    var_predict_err = sum( (ave_predict_err - e2) ** 2 for e1,e2 in errs) / times
    return (ave_train_err, var_train_err),\
           (ave_predict_err, var_predict_err)


times = 1
train_res, predict_res = measure_ave_var(times)

print(f'CP Decomposition in {times} times, '
      f'Mean Training Error is: {train_res[0]}, Variance of Training Error is:{train_res[1]}')
print(f'CP Decomposition in {times} times,'
      f' Mean Training Error is: {predict_res[0]}, Variance of Training Error is:{predict_res[1]}')
