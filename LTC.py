"""LTC tensor decomposition implementation"""

import tensorly as tl
import numpy as np
import numpy.ma as ma

from tensorly.decomposition import parafac

import os
import scipy.io as sio
import math


def load_abilene():
    # import Abilene dataset
    data_dir = os.path.join(os.getcwd(), '../data')
    mat_fname = os.path.join(data_dir, 'oldData.mat')
    mat_contents = sio.loadmat(mat_fname)

    # data matrix Abilene as 'A'
    # A[i, j, k]
    # i: 288, 5-minutes time intervals of a day, 12*24 = 288
    # j: 169, 169 days of collected data
    # k: 144,  12 - 12 nodes ping matrix

    A = mat_contents['oldData']
    return A


class TensorDecompCP(object):
    """Tensor Decomposition class in CP framework with missing data"""

    def __init__(self, tensor, mask, rank=1, iters=10):
        self.tensor = tensor
        self.mask = mask

        # Rank of the CP decomposition
        self.rank = rank
        self.iters = iters

        self.weights = None
        self.factors = None
        self.cp_recovered = None


    def run(self):
        # Perform the CP decomposition
        # missed
        # missed entries:
        #           0 -- value missing
        #           1 -- value measured

        missed = self.mask
        missed_A = self.tensor * missed
        weights, factors = parafac(
            self.tensor, rank=self.rank, init='random',
            n_iter_max=self.iters, tol=10e-6, mask=missed)
        self.weights, self.factors = weights, factors  # store weights & factors in object

        # Reconstruct the image from the factors
        cp_reconstruction = tl.kruskal_to_tensor((weights, factors))
        self.cp_recovered = cp_reconstruction

        train_error, predict_error = calc_err(self.tensor, cp_reconstruction, missed)

        return train_error, predict_error

    def getFactorMatrixSlice(self, dim, idx):
        """
        Get factor matrix slice by slicing direction and index
        :param dim: slicing dimension/direction e.g. frontal
        horizontal, vertical
        :param idx: slicing index
        :return: A vector that consists __idx__ indexed
        slice of factor matrix, vector length is the rank of
        the CP decomposition.
        """
        if self.factors is not None:
            return self.factors[dim][idx]
        else:
            return []

    @classmethod
    def random_mask(cls, n, p):
        """
        Given an array, random fixed size 0s into 1s.
        missed entries:
                   0 -- value missing
                   1 -- value measured
        :param n: Shape of array(Integer or Tuple)
        :param p: Probability of 1
        :return Mask: Shape n N-D array Mask
        """
        size = math.prod(n)
        mask = np.zeros(size)
        K = int(size * p)  # K 1s
        mask[:K] = 1
        np.random.shuffle(mask)
        mask = mask.reshape(n)
        return mask


def calc_err(measured, predicted, missed):
    # count non-zeros to filter all missing scenarios
    if np.count_nonzero(missed) == 0:
        return 0.
    # tensor with indices for training
    # noted mask in ma:
    #               1 -> missing
    #               0 -> sampled
    training_before = ma.masked_array(measured, mask=1-missed)
    training_after = ma.masked_array(predicted, mask=1-missed)

    # tensor with indices for inferring
    predicting_before = ma.masked_array(measured, mask=missed)
    predicting_after = ma.masked_array(predicted, mask=missed)

    # error measure function
    meas_func = lambda before, after: math.sqrt(np.sum((before - after) ** 2)) \
                                      / math.sqrt(np.sum(before ** 2))

    # calculate training error

    return meas_func(training_before, training_after), \
           meas_func(predicting_before, predicting_after)


'''step 1.2 '''
# 1. encoding tensor factors
# 2. hash table initialization
# 3. tensor re-organize

##########################################
# 利用factor matrices对tensor slice进行编码 #
# obtain frontal slice k, i.e X::k       #
##########################################

# a1 ... aR = A
# b1 ... bR = B
# c1 ... cR = C


class LSH_Hash_Table(object):
    """ Super LSH hash table indexing class"""

    def __init__(self, factorMatrix, SuperIndex_N=10):

        #  inti hashing parameters
        self.SuperIndex_N = SuperIndex_N
        self.FactorMatrix = factorMatrix
        self.init_parmeters()  # setting up random seeds for vector to be projected

        # hashing
        self.factor_matrix_to_hash_table()  # compute hash table "run()"
        self.calc_hash_table_max_distance()
        self.calc_hash_table_min_distance()
        self.calc_ordered_list_from_hash()

        # helping functions
        # index 2 code:
        #           (0,0) -> factor[0][0]
        # code 2 hash
        #           code -> self.Super_LSH_Hash(code)
        # hash 2 ordered index
        #           hash -> self.hash_list_sorted.index(hash)

    def init_parmeters(self):
        # random functions
        vector_len = len(self.FactorMatrix[0])  # length of a slice
        self.rnd_a_func = \
            lambda length, times: [
                np.random.normal(np.arange(1, length+1), 1) for i in range(times)]
           # lambda length, times: [np.random.uniform(-1, 1, length) for i in range(times)]
        self.rnd_int = \
            lambda times: np.random.randint(0, 1000, times)

        # provide N(0,1) random projected vectors {a0 ... an}
        self.vectors_a = self.rnd_a_func(vector_len, self.SuperIndex_N)
        self.integers_r = self.rnd_int(self.SuperIndex_N)

    def Super_LSH_Hash(self, code):
        """
        Super LSH hash index calculation
        $H(code) = /Sigma_{i=1}^{N} r_i hash_{a_i}(code)$
        :param code: code to be Super HASH
        :param SuperIndex_N: times of small hash
         calculations to be performed
        :return: Super LSH Hash index
        """
        # tensor slice coefficients = CODE
        # frontal slice k = slice on C dimension of tensor
        # e.g slice 0 of Code = factors[2][0]
        #                            ↑  ↑
        #                            C  k

        # Super indexed LSH hash of code
        SuperHash = 0.
        for i in range(self.SuperIndex_N):
            SuperHash += self.integers_r[i] \
                         * LSH_Hash(code, self.vectors_a[i])

        return SuperHash

    def get_hash_table(self):
        return self.hash_table

    def factor_matrix_to_hash_table(self):
        """
        create super index hashing table from factor matrix
        :param factor: factor matrix from tensor reordering
        :return table: created hash table
        """
        table = {}
        for slice in self.FactorMatrix:
            index = self.Super_LSH_Hash(slice)
            table[index] = slice

        self.hash_table = table

    def calc_hash_table_max_distance(self):
        """calculate max key distance of hash table"""
        cur_max = 0.
        list_hash_table = list(self.hash_table)
        cur_begin = list_hash_table[0]
        cur_end = list_hash_table[0]
        for begin in self.hash_table:
            for end in self.hash_table:
                if begin is not end:
                    if np.abs(begin - end) > cur_max:
                        cur_max = begin - end
                        cur_begin = begin
                        cur_end = end
        self.hash_table_max_distance_tuple =\
            (self.hash_table[cur_begin],
             self.hash_table[cur_end],
             cur_max)

    def calc_ordered_list_from_hash(self):

        hash_list = list(self.hash_table)
        hash_list_sorted = hash_list.copy()
        hash_list_sorted.sort()

        self.hash_list = hash_list
        self.hash_list_sorted = hash_list_sorted

        self.hash_list_max = self.hash_list_sorted[-1]
        self.hash_list_min = self.hash_list_sorted[0]

    def calc_hash_table_min_distance(self):
        """calculate min key distance of hash table"""

        # list keys of hash table
        keys = list(self.hash_table.keys())
        # default min equals distance between first and second
        cur_min = np.abs(keys[0] - keys[1])

        cur_begin = keys[0]
        cur_end = keys[1]

        for begin in self.hash_table:
            for end in self.hash_table:
                if begin is not end:
                    if np.abs(begin - end) < cur_min:
                        cur_min = np.abs(begin - end)
                        cur_begin = begin
                        cur_end = end
        self.hash_table_min_distance_tuple = \
            (self.hash_table[cur_begin],
             self.hash_table[cur_end],
             cur_min)



def LSH_Hash(coef, vector=None):
    """Hash function is coef's projection on vector
    :param vector: vector to be projected to
    :param coef: coefficients to be hashed(projected)
    """

    # 1. generate vector
    # vector need to be a random vector in N(0,1) distribution
    # vector need to have same elements as coef's
    if vector is None:
        vector = np.random.normal(np.arange(1, len(coef)+ 1), 1)
        # vector = np.random.uniform(-1, 1, len(coef))

    # 2. normalize coef

    coef_normalized = coef # if coef len is 1 (rank 1)
    if len(coef) > 1:
        coef_norm = np.linalg.norm(coef)
        coef_normalized = coef / coef_norm

    return np.dot(vector, coef_normalized)


'''step 1.3 '''

##########################################
# 1. random anchor point list            #
# 2. choose latter q-1 anchor points     #
# 3. calculate K of anchor point         #
# 4. fusion all sub-tensors into global  #
#    tensor                              #
##########################################


class tensor_distance_calculation(object):
    """create tensor entry distance calculation class"""
    def __init__(self, tensor_obj, hash_table_objs):
        self.tensor_obj = tensor_obj
        self.hash_table_objs = hash_table_objs

    def calc_dis_entry(self, entry1, entry2):
        # dis_a
        dis_a = self.calc_dis_slices_index(
            0, entry1, entry2)

        # dis_b
        dis_b = self.calc_dis_slices_index(
            1, entry1, entry2)

        # dis_c
        dis_c = self.calc_dis_slices_index(
            2, entry1, entry2)

        return dis_a * dis_b * dis_c

    def _calc_dis_slices(self, slice1, slice2):
        up = np.dot(slice1, slice2)
        down = np.linalg.norm(slice1) * np.linalg.norm(slice2)
        return math.acos(round((up / down), 8))

    def calc_dis_slices_index(self, dim, entry1, entry2):
        # dis_a
        slice_1 = self.tensor_obj.getFactorMatrixSlice(
            dim, entry1[dim])
        slice_2 = self.tensor_obj.getFactorMatrixSlice(
            dim, entry2[dim])

        slice_dis = self._calc_dis_slices(slice_1, slice_2)

        # normalize slice dis into [0,1] range
        max_dis_tuple = self.hash_table_objs[dim].hash_table_max_distance_tuple
        min_dis_tuple = self.hash_table_objs[dim].hash_table_min_distance_tuple

        # calc min & max dis in slice distance
        min_dis = self._calc_dis_slices(min_dis_tuple[0], min_dis_tuple[1])
        max_dis = self._calc_dis_slices(max_dis_tuple[0], max_dis_tuple[1])

        slice_dis_normalized = \
            (slice_dis - min_dis) / (max_dis - min_dis)
        return slice_dis_normalized


def calc_AP_candiate_list(tensor, size):
    """random a list of AP_candidate from reordered tensor factor matrices"""
    # AP should not sit on side of tensor

    list_0 = np.arange(tensor.shape[0])[1:-1]
    list_1 = np.arange(tensor.shape[1])[1:-1]
    list_2 = np.arange(tensor.shape[2])[1:-1]
    dim_0s = np.random.choice(list_0, size, replace=False)
    dim_1s = np.random.choice(list_1, size, replace=False)
    dim_2s = np.random.choice(list_2, size, replace=False)

    AP_candidate_list = [[dim_0s[i],dim_1s[i],dim_2s[i]] for i in range(len(dim_0s))]
    # create sub-tensors based on AP candidate list

    return AP_candidate_list


class sub_tensor(object):
    """
    calculate sub-tensor's coverage of global tensor(i.e. covered indices)
    mark sub-tensor's size
    """
    def __init__(self, tensor_obj, hash_objs, h, AP, sub_rank):
        self.tensor_obj = tensor_obj
        self.hash_objs = hash_objs
        self.h = h
        self.AP = AP
        self.sub_rank = sub_rank
        self.sub_tensor = None
        self.sub_tensor_mask = None
        self.sub_tensor_recovered = None
        self.create_sub_tensor_from_AP()

    def get_global_entry_from_sorted_index(self, entry_sorted_idx):
        """trace back tensor's global index from sorted
        index of an giving entry"""
        global_entry = []
        for i in range(len(entry_sorted_idx)):
            sorted_idx = entry_sorted_idx[i]
            global_idx = \
                self.hash_objs[i].hash_list.index(
                    self.hash_objs[i].hash_list_sorted[sorted_idx]
                )
            global_entry.append(global_idx)

        return global_entry

    def calc_density(self):
        self.covered_global_indices = [
            self.get_global_entry_from_sorted_index(
                idx) for idx in self.covered_indices]

        covered_num = 0
        # mask 0 missing
        # mask 1 has data
        for entry in self.covered_global_indices:
            #test is an entry is missing
            if self.tensor_obj.mask[
                entry[0],
                entry[1],
                entry[2]
            ] > .5:
                covered_num += 1

        return covered_num/len(self.covered_global_indices)


    def create_sub_tensor_from_AP(self):
        entry_dis_calc_obj = tensor_distance_calculation(
            self.tensor_obj, self.hash_objs)

        # since h is in range[0,1]
        # in each dim search range is in: AP[0] +/- dim.size * h

        search_range = []
        for i in range(len(self.AP)):
            # get slice
            slice = self.hash_objs[i].FactorMatrix[self.AP[i]]
            key = self.hash_objs[i].Super_LSH_Hash(slice)
            index = self.hash_objs[i].hash_list_sorted.index(key)

            low = index - math.ceil(
                self.tensor_obj.tensor.shape[i] * self.h)
            low = low if low >0 else 0

            high = index + math.ceil(
                self.tensor_obj.tensor.shape[i] * self.h)
            high = high if high < self.tensor_obj.tensor.shape[i] \
                else self.tensor_obj.tensor.shape[i]

            #keys = self.hash_objs[i].hash_list_sorted[low:high]
            keys = list(range(low,high))
            search_range.append(keys)

        covered_indices =[]
        # search limit
        # for i in search_range[0]:
        #     for j in search_range[1]:
        #         for k in search_range[2]:
        #             if self.h >= entry_dis_calc_obj.calc_dis_entry([i,j,k], self.AP):
        #                 covered_indices.append([i,j,k])

        for i in search_range[0]:
            for j in search_range[1]:
                for k in search_range[2]:
                    covered_indices.append([i,j,k])

        self.covered_ranges = search_range
        self.covered_indices = covered_indices
        self.sub_tensor_size = len(covered_indices)

    def get_sub_tensor_cells(self):
        """calculate sub tensor's cells from global tensor"""
        sub_tensor = np.zeros(
            (len(self.covered_ranges[0]),
             len(self.covered_ranges[1]),
             len(self.covered_ranges[2]))
        )

        sub_tensor_mask = np.zeros(
            (len(self.covered_ranges[0]),
             len(self.covered_ranges[1]),
             len(self.covered_ranges[2]))
        )

        for (i,j,k),value in np.ndenumerate(sub_tensor):
            global_entry = self.get_global_entry_from_sorted_index([i,j,k])
            sub_tensor[i,j,k] = self.tensor_obj.tensor[
                global_entry[0],
                global_entry[1],
                global_entry[2]
            ]
            sub_tensor_mask[i,j,k] = self.tensor_obj.mask[
                global_entry[0],
                global_entry[1],
                global_entry[2]
            ]

        self.sub_tensor = sub_tensor
        self.sub_tensor_mask = sub_tensor_mask

        return sub_tensor, sub_tensor_mask

    def recover_sub_tensor(self, rank=3):
        if type(self.sub_tensor_recovered) != np.ndarray:
            st, st_mask = self.get_sub_tensor_cells()
            st_cp_obj = TensorDecompCP(st, st_mask, rank, iters=100)
            st_cp_obj.run()
            self.sub_tensor_recovered = st_cp_obj.cp_recovered
        return self.sub_tensor_recovered

    def global_entry_to_sub_tensor_entry(self, entry):

        sub_entry = [0, 0, 0]
        for i in range(len(entry)):
            sub_entry[i] = self.covered_ranges[i].index(entry[i])
        return sub_entry


def confirm_sub_tensor_list(sub_tensors, alpha, tensor_distance_calc_obj, q=30):
    """choose ap from ap candidates
     alpha : [0,1], blending factor
     q: 30, total ap to confirm
     l: number of ap already chose
     :type tensor_distance_calc_obj: tensor_distance_calculation
     :return confirm ap list, ap as sub-tensor
     """

    # 1. choose first ap as random

    first_ap = np.random.choice(sub_tensors)

    l = 0
    confirmed = [first_ap]
    unconfirmed = sub_tensors.copy()
    unconfirmed.remove(first_ap)

    # 2. blending

    while l < q:

        # create current sub-tensor

        density = sub_tensors[l].calc_density()

        # 3. blending right part

        blended_max = 0.
        for cur_uncon in unconfirmed:
            sum_d = 0.
            for cur_con in confirmed:
                sum_d += tensor_distance_calc_obj.calc_dis_entry(
                    cur_con.AP, cur_uncon.AP)
            blended = density * alpha + (sum_d/(l+1)) * (1-alpha)
            if blended > blended_max:
                blended_max = blended
                blended_max_tensor = cur_uncon
        confirmed.append(blended_max_tensor)
        unconfirmed.remove(blended_max_tensor)
        l += 1
    return confirmed


def calc_K_correlation(sub_tensor, entry, entry_distance_calc_obj):
    """measure correlation between an entry
    to sub_tensor's anchor point"""

    ap = sub_tensor.AP
    distance = entry_distance_calc_obj.calc_dis_entry(ap, entry)
    K = 0
    if distance< sub_tensor.h:
        K = 1 - distance**2
    return K


def fusion_sub_tensors_to_recovered_tensor(sub_tensors, calc_dis_obj):

    # 1. recovered tensor entry is CP recovered tensor entry
    recovered_tensor = sub_tensors[0].tensor_obj.cp_recovered.copy()
    if type(recovered_tensor) != np.ndarray:
        print("[FusionErr] CP recovered is None")
        return None

    # 2. calculate K map of all entries covered by sub_tensors

    K_map = {} # dictionary map of np.array(entry)--K pair
    for st in sub_tensors:
        # covered_indices of global tensor
        for entry in st.covered_indices:
            K_map[str(entry)] = 0.

    # loop all entries from sub tensors
    for st in sub_tensors:
        # covered_indices of global tensor
        for entry in st.covered_indices:
            for st2 in sub_tensors:
                if is_entry_covered_by_sub_tensor(entry, st2):
                    K_map[str(entry)] +=\
                        calc_K_correlation(st2,entry,calc_dis_obj)

    # 3. loop all sub tensors entries to accumulate

    LTC_entry_map = {}
    for st in sub_tensors:
        # covered_indices of global tensor
        for entry in st.covered_indices:
            LTC_entry_map[str(entry)] = 0.

    for st in sub_tensors:
        # covered_indices of global tensor
        for entry in st.covered_indices:
            if False:
                # debug K calculation
                recovered_st = st.recover_sub_tensor(st.sub_rank)
                sub_indices = st.global_entry_to_sub_tensor_entry(entry)
                recovered_cell = recovered_st[
                    sub_indices[0],
                    sub_indices[1],
                    sub_indices[2]
                ]
                LTC_entry_map[str(entry)] = recovered_cell

            else:

                for st2 in sub_tensors:
                    if is_entry_covered_by_sub_tensor(entry, st2):
                        recovered_st = st2.recover_sub_tensor(st2.sub_rank)
                        sub_indices = st2.global_entry_to_sub_tensor_entry(entry)
                        recovered_cell = recovered_st[
                            sub_indices[0],
                            sub_indices[1],
                            sub_indices[2]
                            ]
                        top = calc_K_correlation(st2,entry,calc_dis_obj) \
                              * recovered_cell

                        if(K_map[str(entry)] != 0.):
                            LTC_entry_map[str(entry)] += top / K_map[str(entry)]

    # 4. weighted recovered data
    for entry_str in LTC_entry_map:
        entry = eval(entry_str)
        old = recovered_tensor[
            entry[0],
            entry[1],
            entry[2]
        ]
        new = LTC_entry_map[entry_str]
        if new == 0.0:
            continue
        recovered_tensor[
            entry[0],
            entry[1],
            entry[2]
        ] = LTC_entry_map[entry_str]


    return recovered_tensor


def is_entry_covered_by_sub_tensor(entry, sub_tensor):
    ranges = sub_tensor.covered_ranges
    if ranges[0][0] <= entry[0] <= ranges[0][-1]:
        if ranges[1][0] <= entry[1] <= ranges[1][-1]:
            if ranges[2][0] <= entry[2] <= ranges[2][-1]:
                return True
    return False
