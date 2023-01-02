"""Implementation of Localized Sensitive Hashing for tensors"""
import tensorly as tl
import numpy as np
import numpy.ma as ma

from tensorly.decomposition import parafac

import os
import scipy.io as sio
import math
import time

from LTC import LTC
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
def LTC_test(h, q, alpha, sample_rate, rank, times, ap_cap, sub_rank):
    """LTC test main function"""

    ##################################
    # Step 1                         #
    # Load data                      #
    # Compute Tensor Factor Matrices #
    ##################################

    # tensor decomposition result
    tensor_D_result_dict = {}

    # load data
    A = LTC.load_abilene()

    # config mask
    mask = LTC.TensorDecompCP.random_mask(A.shape, sample_rate)

    # define cp rank
    rank = rank

    # link data to CP decomposition object
    TD_obj = LTC.TensorDecompCP(A, mask, rank)

    TD_obj.run()  # perform CP decomposition

    # tensor factor matrices: [A, B, C]
    factors = TD_obj.factors

    times = times
    tensor_D_result_dict['CP'] = LTC.calc_err(A, TD_obj.cp_recovered, mask)

    ##################################
    # Step  2                        #
    # Create LSH Hash Table          #
    ##################################


    # fill hash tables with super indices
    hash_tables_objs = [] # create hash tables for [A,B,C]
    for factor in factors:
        hash_tables_objs.append(LTC.LSH_Hash_Table(factor))

    # adjust parameter h
    # adjust parameter alpha
    # adjust parameter q

    ##################################
    # Step   3                       #
    # Define distance of two         #
    # entries from tensor            #
    ##################################

    # distance between entry (1,2,3) and entry (4,5,6)

    tensor_dis_calc_obj = LTC.tensor_distance_calculation(TD_obj,hash_tables_objs)
    # distance = tensor_dis_calc_obj.calc_dis_entry((1,2,3),(4,5,6))


    ap_list = LTC.calc_AP_candiate_list(TD_obj.tensor, ap_cap)

    sub_tensors = [LTC.sub_tensor(TD_obj, hash_tables_objs, h, item, sub_rank) for item in ap_list]
    # ap_objs_list = []
    # for idx in range(len(ap_list)):
    #     ap_objs_list[idx] = [
    #         ap_list[idx],
    #         sub_tensors[idx],
    #     ]

    confirmed_sub_tensor_list = LTC.confirm_sub_tensor_list(
                        sub_tensors,
                        alpha,
                        tensor_dis_calc_obj,
                        q)

    sub_tensors_recovered = []
    for sub_tensor in confirmed_sub_tensor_list:
        sub_tensors_recovered.append(sub_tensor.recover_sub_tensor(rank))


    LTC_recovered = LTC.fusion_sub_tensors_to_recovered_tensor(sub_tensors,tensor_dis_calc_obj)


    tensor_D_result_dict['LTC'] = LTC.calc_err(A, LTC_recovered, mask)

    return tensor_D_result_dict


@time_it
def result_print(json_str):
    for test_tuple in json_str:
        params, result_dict = test_tuple
        for test in result_dict:
            print(f'{test} Decomposition at the {params["times"]}th time, '
                  f'Mean Training Error is: {result_dict[test][0]}, '
                  f'Mean Prediction Error is: {result_dict[test][1]}'
                  )

@time_it
def result_save(json_str):
    import json

    path = '../data/'

    file_dict_list = []
    for test_tuple in json_str:
        params, result_dict = test_tuple
        file_dict = {}
        file_dict['params'] = params

        tensorD_result = {}
        for test in result_dict:
            tensorD_result[test] = {
                'MeanTrErr': result_dict[test][0],
                'MeanPrErr': result_dict[test][1],
            }
        file_dict['data'] = tensorD_result

        file_dict_list.append(file_dict)

    # serializing json
    with open(path + "data.json", "w") as outfile:
        json.dump(file_dict_list, outfile)


if __name__ == "__main__":

    default_parameters= {
        'h': .05,
        'q': 5,
        'alpha': .3,
        'sample_rate': .4,
        'rank': 2, # rank must >= 2
        'times': 1,
        'ap_cap': 10,
        'sub_rank': 1,
    }

    params_result_dicts = []
    # h from .05 to .2, increase step by .05
    for i,v in np.ndenumerate(np.arange(.05, .06, .05)):
        params = default_parameters.copy()
        params['h'] = v
        params['times'] = list(i)[0] + 1
        result_dict = LTC_test(**params)
        params_result_dicts.append((params, result_dict))

    result_print(params_result_dicts)
    result_save(params_result_dicts)




