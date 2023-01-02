"""Implementation of Localized Sensitive Hashing for tensors"""

# Fix local import issue
import sys
from pathlib import Path
cwd =str(Path(__file__).parent.parent)
sys.path.insert(0,cwd)

import tensorly as tl
import numpy as np
import numpy.ma as ma

from tensorly.decomposition import parafac

import os
import scipy.io as sio
import math
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


def CP_test(sample_rate, rank):
    ##################################
    # Step 1                         #
    # Load data                      #
    # Compute Tensor Factor Matrices #
    ##################################

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
    return TD_obj

@time_it
def LTC_test(queue,TD_obj, h, q, alpha, sample_rate,
             rank, times, ap_cap, sub_rank, return_dict, it, dis_mode):
    """LTC test main function"""

    # tensor decomposition result
    tensor_D_result_dict = {}

    T = TD_obj.tensor

    # config mask
    mask = TD_obj.mask

    # define cp rank
    rank = rank

    # tensor factor matrices: [A, B, C]
    factors = TD_obj.factors
    if factors == None:
        print("[Debug] Error factor in TD_obj")

    times = times
    tensor_D_result_dict['CP'] = LTC2.calc_err(T, TD_obj.cp_recovered, mask)

    ##################################
    # Step  2                        #
    # Create LSH Hash Table          #
    ##################################


    # fill hash tables with super indices
    hash_tables_objs = [] # create hash tables for [A,B,C]
    for factor in factors:
        hash_tables_objs.append(LTC2.LSH_Hash_Table(factor))

    # adjust parameter h
    # adjust parameter alpha
    # adjust parameter q

    ##################################
    # Step   3                       #
    # Define distance of two         #
    # entries from tensor            #
    ##################################

    # distance between entry (1,2,3) and entry (4,5,6)

    tensor_dis_calc_obj = LTC2.tensor_distance_calculation(TD_obj,hash_tables_objs)
    # distance = tensor_dis_calc_obj.calc_dis_entry((1,2,3),(4,5,6))

    # init ap candidate list
    ap_list = LTC2.calc_AP_candiate_list(hash_tables_objs, bucket_number=5)

    # create sub tensor (objs) from each ap in ap list
    sub_tensors = [LTC2.sub_tensor(TD_obj, hash_tables_objs, h, item, sub_rank)
                   for item in ap_list]

    # apply ap selecting algos to confirm the final ap_confirmed_list
    confirmed_sub_tensor_list = LTC2.confirm_sub_tensor_list(
        sub_tensors,
        alpha,
        tensor_dis_calc_obj,
        q,
        dis_mode
    )

    sub_tensors_recovered = []
    for sub_tensor in confirmed_sub_tensor_list:
        sub_tensors_recovered.append(sub_tensor.recover_sub_tensor(rank))


    LTC_recovered = LTC2.fusion_sub_tensors_to_recovered_tensor(
        confirmed_sub_tensor_list, tensor_dis_calc_obj, dis_mode)


    tensor_D_result_dict['LTC'] = LTC2.calc_err(T, LTC_recovered, mask)

    print("[Debug] Process {} done.".format(it))

    print(f'CP Decomposition at the {times}th time, '
          f'Mean Training Error is: {tensor_D_result_dict["CP"][0]}, '
          f'Mean Prediction Error is: {tensor_D_result_dict["CP"][1]}')
    print(f'LTC Decomposition at the {times}th time, '
          f'Mean Training Error is: {tensor_D_result_dict["LTC"][0]}, '
          f'Mean Prediction Error is: {tensor_D_result_dict["LTC"][1]}')
    return_dict[it] = tensor_D_result_dict

    # message result to message q
    # csv format
    # h, q, alpha, CP_tr, CP_pr, LTC_tr, LTC_pr

    result = [
        dis_mode,
        h,
        q,
        alpha,
        tensor_D_result_dict['CP'][0],
        tensor_D_result_dict['CP'][1],
        tensor_D_result_dict['LTC'][0],
        tensor_D_result_dict['LTC'][1],
    ]

    queue.put(result)

    # TODO clear memory of dicts before going next iteration


    return tensor_D_result_dict


def result_print(params, result_dict):
    for run in result_dict:
        for test in run:
            print(f'{test} Decomposition at the {params["times"]}th time, '
              f'Mean Training Error is: {run[test][0]}, '
              f'Mean Prediction Error is: {run[test][1]}'
              )


def result_save(json_str):
    import json

    path = './data/'

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

    path = '../data2/data.csv'
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
        'h': .6,
        'q': 20,
        'alpha': .7,
        'sample_rate': .4,
        'rank': 2, # rank must >= 2
        'times': 1,
        'ap_cap': 10,
        'sub_rank': 1,
        'runs': 5,
        'dis_mode': 'eu', # euclidean -> eu; angular -> an
    }

    count = 1

    # CP decomposition
    TD_obj = CP_test(default_parameters['sample_rate'], default_parameters['rank'])

    # Manage a list of processes, one writer, many workers
    # Workers sent results to writer, writer write into file
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count()-1)

    # Init writer 1st
    writer = pool.apply_async(writer_func, (q,))

    jobs = []

    # h
    # for i, v in np.ndenumerate(np.arange(.3, 1.01, .1)):
    #     params = default_parameters.copy()
    #     runs = default_parameters['runs']
    #     del params['runs']
    #     params['h'] = v
    #     params['times'] = count
    #     count += 1
    #     result_dicts = [None] * runs
    #     for run in range(runs):
    #         # result_dict = LTC_test(**params)
    #         # result_dicts.append(result_dict)
    #         params_copy = params.copy()
    #         params_copy['return_dict'] = result_dicts
    #         params_copy['it'] = run
    #         job = pool.apply_async(LTC_test, (q,TD_obj,), kwds=params_copy)
    #         jobs.append(job)

    # q
    # for j, v1 in np.ndenumerate(range(1, 31, 1)):
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
    #         job = pool.apply_async(LTC_test, (q,TD_obj,), kwds=params_copy)
    #         jobs.append(job)


    # alpha
    for k, v2 in np.ndenumerate(np.arange(.2, .9, .1)):
        params = default_parameters.copy()
        runs = default_parameters['runs']
        del params['runs']
        params['alpha'] = v2
        params['times'] = count
        count += 1
        result_dicts = [None] * runs
        for run in range(runs):
            # result_dict = LTC_test(**params)
            # result_dicts.append(result_dict)
            params_copy = params.copy()
            params_copy['return_dict'] = result_dicts
            params_copy['it'] = run
            job = pool.apply_async(LTC_test, (q,TD_obj,), kwds=params_copy)
            jobs.append(job)


    for job in jobs:
        job.get()

    q.put('done')
    pool.close()
    pool.join()
