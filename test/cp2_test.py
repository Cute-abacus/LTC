"""Implementation of Localized Sensitive Hashing for tensors"""
import numpy as np
import time

import multiprocessing as mp
import csv

from LTC import LTC2
from functools import wraps


def time_it(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Module {func.__name__} took {end - start :0.4f} seconds")
        return result
    return wrapped_func


@time_it
def LTC_test(queue, h, q, alpha, sample_rate, rank, times, ap_cap, sub_rank, return_dict, it):
    """LTC test main function"""

    ##################################
    # Step 1                         #
    # Load data                      #
    # Compute Tensor Factor Matrices #
    ##################################

    # tensor decomposition result
    tensor_D_result_dict = {}

    # load data
    T = LTC2.load_abilene()
    # T = LTC2.load_synthesis()

    # config mask
    mask = LTC2.TensorDecompCP.random_mask(T.shape, sample_rate)

    # define cp rank
    rank = rank

    # link data to CP decomposition object
    TD_obj = LTC2.TensorDecompCP(T, mask, rank)

    TD_obj.run()  # perform CP decomposition

    # tensor factor matrices: [A, B, C]
    factors = TD_obj.factors

    times = times
    tensor_D_result_dict['CP'] = LTC2.calc_err(T, TD_obj.cp_recovered, mask)
    tensor_D_result_dict['LTC'] = 0., 0.

    print("[Debug] Process {} done.".format(it))

    print(f'CP Decomposition at the {times}th time, '
          f'Mean Training Error is: {tensor_D_result_dict["CP"][0]}, '
          f'Mean Prediction Error is: {tensor_D_result_dict["CP"][1]}')
    return_dict[it] = tensor_D_result_dict

    # message result to message q
    # csv format
    # h, q, alpha, CP_tr, CP_pr, LTC_tr, LTC_pr

    result = [
        h,
        q,
        alpha,
        tensor_D_result_dict['CP'][0],
        tensor_D_result_dict['CP'][1],
        tensor_D_result_dict['LTC'][0],
        tensor_D_result_dict['LTC'][1],
    ]

    queue.put(result)

    return tensor_D_result_dict


def result_save(json_str):
    import json

    path = '../data/'

    file_dict_list = []
    for test_tuple in json_str:
        params, result_dicts = test_tuple
        file_dict = {}
        file_dict['params'] = params

        tensorD_results = []
        for run in result_dicts:
            tensorD_result = {}
            for test in run:
                tensorD_result[test] = {
                    'MeanTrErr': run[test][0],
                    'MeanPrErr': run[test][1],
                }
            tensorD_results.append(tensorD_result)
        file_dict['data'] = tensorD_results

        file_dict_list.append(file_dict)

    # serializing json
    with open(path + "data.json", "w") as outfile:
        json.dump(file_dict_list, outfile)


def writer_func(q):
    """ writer listen to message on queue, and write into a csv file"""

    path = '../data/data.csv'
    while 1:
        m = q.get()
        if m == 'done': # message after joined
            break
        # write
        with open(path, 'a', newline='') as f:
            wr = csv.writer(f, delimiter=',')
            wr.writerow(m)


if __name__ == "__main__":

    default_parameters= {
        'h': .5,
        'q': 30,
        'alpha': .5,
        'sample_rate': .4,
        'rank': 2, # rank must >= 2
        'times': 1,
        'ap_cap': 10,
        'sub_rank': 1,
        'runs': 5,
    }

    count = 1

    # Manage a list of processes, one writer, many workers
    # Workers sent results to writer, writer write into file
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count()-1)

    # Init writer 1st
    writer = pool.apply_async(writer_func, (q,))

    jobs = []

    # h
    for i, v in np.ndenumerate(np.arange(.3, .9, .1)):
        params = default_parameters.copy()
        runs = default_parameters['runs']
        del params['runs']
        params['h'] = v
        params['times'] = count
        count += 1
        result_dicts = [None] * runs
        for run in range(runs):
            # result_dict = LTC_test(**params)
            # result_dicts.append(result_dict)
            params_copy = params.copy()
            params_copy['return_dict'] = result_dicts
            params_copy['it'] = run
            job = pool.apply_async(LTC_test, (q,), kwds=params_copy)
            jobs.append(job)

    # q
    # for j, v1 in np.ndenumerate(range(1, 11, 1)):
    #     params = default_parameters.copy()
    #     runs = default_parameters['runs']
    #     del params['runs']
    #     params['q'] = int(v1)  # JSON complain when int32
    #     params['times'] = count
    #     count += 1
    #     result_dicts = [None] * runs
    #     for run in range(runs):
    #         # result_dict = LTC_test(**params)
    #         # result_dicts.append(result_dict)
    #         params_copy = params.copy()
    #         params_copy['return_dict'] = result_dicts
    #         params_copy['it'] = run
    #         job = pool.apply_async(LTC_test, (q,), kwds=params_copy)
    #         jobs.append(job)


    # alpha
    # for k, v2 in np.ndenumerate(np.arange(.2, .9, .1)):
    #     params = default_parameters.copy()
    #     runs = default_parameters['runs']
    #     del params['runs']
    #     params['alpha'] = v2
    #     params['times'] = count
    #     count += 1
    #     result_dicts = [None] * runs
    #     for run in range(runs):
    #         # result_dict = LTC_test(**params)
    #         # result_dicts.append(result_dict)
    #         params_copy = params.copy()
    #         params_copy['return_dict'] = result_dicts
    #         params_copy['it'] = run
    #         job = pool.apply_async(LTC_test, (q,), kwds=params_copy)
    #         jobs.append(job)


    for job in jobs:
        job.get()

    q.put('done')
    pool.close()
    pool.join()
