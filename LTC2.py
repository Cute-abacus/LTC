"""LTC tensor decomposition implementation"""

import tensorly as tl
import numpy as np
import numpy.ma as ma
from tensorly.decomposition import parafac

import os
import scipy.io as sio
import math
import timeit

from LTC import makeSynthesis

Debug_flag = True
Debug_flag1 = False
Debug_reduce_A = True

def load_abilene():
    # import Abilene dataset
    data_dir = os.path.join(os.getcwd(), '../data')
    mat_fname = os.path.join(data_dir, 'p2p1953.mat')
    mat_contents = sio.loadmat(mat_fname)

    # data matrix Abilene as 'A'
    # A[i, j, k]
    # i: 288, 5-minutes time intervals of a day, 12*24 = 288
    # j: 169, 169 days of collected data
    # k: 144,  12 - 12 nodes ping matrix

    A = mat_contents['X']
    if Debug_reduce_A:
        A = A[0:200, 0:200, :]
    return A


def load_synthesis():
    """
     import synthesis dataset
     data matrix synthesis as 'S'
     S[i, j, k]
     i: 100
     j: 100
     k: 100

    """
    params = {
        'dims': [20, 20, 20],
        'rank': 1,
    }
    S = makeSynthesis.makeSynthesis(**params)
    return S


class TensorDecompCP(object):
    """Tensor Decomposition class in CP framework with missing data"""

    def __init__(self, tensor, mask, rank=1, iters=100):
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
        try:
            missed_A = self.tensor * missed
        except:
            pass

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
        Given an array, random fixed size 1s into 0s.
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
        return 0., 0.
    # tensor with indices for training
    # noted mask in ma:
    #               1 -> missing
    #               0 -> sampled
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

    def __init__(self, factorMatrix, SuperIndex_N=1):

        #  inti hashing parameters
        self.SuperIndex_N = SuperIndex_N
        self.FactorMatrix = factorMatrix
        self.init_parmeters()  # setting up random seeds for vector to be projected

        # hashing
        self.factor_matrix_to_hash_table()  # compute hash table "run()"
        self.calc_hash_table_max_distance()
        self.calc_hash_table_min_distance()
        self.calc_ordered_list_from_hash()

    def init_parmeters(self):
        vector_len = len(self.FactorMatrix[0])  # length of a slice

        if self.SuperIndex_N == 1:
            # fixed parameters for N=1 instance
            self.vectors_a = [np.arange(1, vector_len+1), ]
            self.integers_r = np.array([13,])
        else:
            # random functions
            self.rnd_a_func = \
                lambda length, times: [
                    np.random.normal(np.arange(1, length + 1), 1) for i in range(times)]
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
                        cur_max = np.abs(begin - end) # enforce distance >= 0
                        cur_begin = begin
                        cur_end = end
        self.hash_table_max_distance_tuple = \
            (self.hash_table[cur_begin],
             self.hash_table[cur_end],
             cur_max)

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

    def calc_ordered_list_from_hash(self):

        hash_list = list(self.hash_table)
        hash_list_sorted = hash_list.copy()
        hash_list_sorted.sort()

        self.hash_list = hash_list
        self.hash_list_sorted = hash_list_sorted

        self.hash_list_max = self.hash_list_sorted[-1]
        self.hash_list_min = self.hash_list_sorted[0]

    # helping functions
    def get_hash_table(self):
        return self.hash_table

    def Idx_2_Ordered_Idx(self, idx):
        """
        index 2 code:
                  index -> self.FactorMatrix[index]
        code 2 hash
                  code -> self.Super_LSH_Hash(code)
        hash 2 ordered index
                  hash -> self.hash_list_sorted.index(hash)
        """

        return self.hash_list_sorted.index(self.Super_LSH_Hash(self.FactorMatrix[idx]))

    def Ordered_Idx_2_Idx(self, ordered_idx):
        """
        reversed of Idx_2_Ordered_Idx()
        """
        _hash = self.hash_list_sorted[ordered_idx]
        code = self.hash_table[_hash]
        idx = np.argwhere(self.FactorMatrix == code)
        return idx[0][0]


def LSH_Hash(coef, vector=None):
    """Hash function is coef's projection on vector
    :param vector: vector to be projected to
    :param coef: coefficients to be hashed(projected)
    """

    # 1. generate vector
    # vector need to be a random vector in N(0,1) distribution
    # vector need to have same elements as coef's
    if vector is None:
        vector = np.random.normal(np.arange(1, len(coef) + 1), 1)
        # vector = np.random.uniform(-1, 1, len(coef))

    # 2. normalize coef

    coef_normalized = coef  # if coef len is 1 (rank 1)
    if len(coef) > 1:
        # np.linalg.norm(x) -> np.sqrt(x.dot(x))
        # coef_norm = np.linalg.norm(coef)
        # coef_norm = np.sqrt(coef.dot(coef))
        coef_norm = math.sqrt(coef.dot(coef))
        coef_normalized = coef / coef_norm


    # res = np.dot(vector, coef_normalized)
    res = vector.dot(coef_normalized)
    return res


'''step 1.3 '''


class tensor_distance_calculation(object):
    """create tensor entry distance calculation class"""

    def __init__(self, tensor_obj, hash_table_objs):
        self.tensor_obj = tensor_obj
        self.hash_table_objs = hash_table_objs

    def calc_dis_entry_hash(self, ap1, ap2, mode="eu"):
        """RE-DEFINE Distance function
        Since the distance defined in paper is distance between entry cells,
        we need to implement a distance between hashes(ap is represented in hashes)

        d(h1, h2) =
            (euclidean):
                    l2 distance of Hash
                    a = (h1 - h2)/ (max - min)
                    d = sqrt(a^2 + b^2 + c^2)

            (angular):
                    product of code projection (Hash)
                    norm_projection = projection/max_projection
                    a = arccos((h1 - h2)/ (max - min))
                    a [0, pi/2] -> [0, 1]
        """

        dis = 1.
        if mode == "eu":  # euclidean distance
            dis = 0.

        for i in range(len(ap1)):
            top = np.abs(ap1[i] - ap2[i])
            bot = self.hash_table_objs[i].hash_table_max_distance_tuple[-1] # fix max distance

            if mode == "eu":  # euclidean distance
                dis += (top / bot)**2
            else:  # angular distance
                rad = math.acos(round(1. - top/bot, 8))
                rad /= math.pi * .5
                dis *= rad

        if mode == "eu":  # euclidean distance
            return math.sqrt(dis)
        return dis

    def calc_dis_entry(self, entry1, entry2):

        product = 1.
        for i in range(len(self.tensor_obj.shape)):
            product *= self.calc_dis_slices_index(i, entry1, entry2)

        return product

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


def calc_AP_candiate_list(hash_objs, bucket_number=10):
    """ AP candidates as permutations of idx of reordered tensor factor matrices"""
    # AP should not sit on side of tensor

    # bucket_number in one dimension
    # AP is not a cell, but a hash tuple like (hash_A, hash_B, hash_C)
    # AP sits in hash space, instead of tensor index space

    ap_idx_pool = []
    for i in range(len(hash_objs)):
        # A
        max_hash = hash_objs[i].hash_list_max
        min_hash = hash_objs[i].hash_list_min
        bucket_width = (max_hash - min_hash) / bucket_number
        bucket_centers = list(np.arange(min_hash + bucket_width / 2., max_hash, bucket_width))
        ap_idx_pool.append(bucket_centers)

    ap_tensor = np.zeros(
        (bucket_number, bucket_number, bucket_number),
        dtype=(float, len(hash_objs)))

    for i in range(bucket_number):
        for j in range(bucket_number):
            for k in range(bucket_number):
                ap_tensor[i, j, k] = (
                    ap_idx_pool[0][i],
                    ap_idx_pool[1][j],
                    ap_idx_pool[2][k])

    # ap candidate list
    ap_list = []
    for ap in np.ndindex(ap_tensor.shape[:-1]):
        # ap is (hash1, hash2, hash3)
        # get index by hash

        ap_list.append(ap_tensor[ap])
    return ap_list


class sub_tensor(object):
    """
    calculate sub-tensor's coverage of global tensor(i.e. covered indices)
    mark sub-tensor's size
    """

    def __init__(self, tensor_obj, hash_objs, h, AP, sub_rank, bucket_number=10):
        self.tensor_obj = tensor_obj
        self.hash_objs = hash_objs
        self.h = h
        self.AP = AP  # ap is hash now
        self.sub_rank = sub_rank
        self.bucket_num = bucket_number
        self.sub_tensor = None
        self.sub_tensor_mask = None
        self.sub_tensor_recovered = None
        self._create_sub_tensor_from_AP()

    def _create_sub_tensor_from_AP(self):

        # since h is in range[0,1]
        # in each dim search range is in: AP[0] +/- dim.size * h
        search_range = []
        for i in range(len(self.AP)):
            # get slice
            # AP is hash now, need to get its covered indices

            # fixing param h issues
            # width is associate with h param
            width = (self.hash_objs[i].hash_list_max \
                     - self.hash_objs[i].hash_list_min) * self.h

            # width = (self.hash_objs[i].hash_list_max \
            #          - self.hash_objs[i].hash_list_min) / self.bucket_num
            # width *= self.h

            st_indices_ordered = []
            for idx in range(len(self.hash_objs[i].hash_list)):
                if self.AP[i] - width<= self.hash_objs[i].hash_list[idx] \
                        < self.AP[i] + width:
                    st_indices_ordered.append(idx)

            # transfer ordered idx into original idx
            st_indices = [self.hash_objs[i].Ordered_Idx_2_Idx(idx)
                         for idx in st_indices_ordered]

            search_range.append(st_indices)

        def covered_indices_gen(s_range=search_range):
            for i in s_range[0]:
                for j in s_range[1]:
                    for k in s_range[2]:
                        yield [i, j, k]
        # covered_indices = []
        # for i in search_range[0]:
        #     for j in search_range[1]:
        #         for k in search_range[2]:
        #             covered_indices.append([i, j, k])

        self.covered_ranges = search_range
        # covered_indices memory -> generator to save memory
        # self.covered_indices = covered_indices
        self.covered_indices_gen = covered_indices_gen
        self.sub_tensor_size = np.prod([len(i) for i in search_range])

        # fill sub-tensor cells with data from original tensor
        sub_tensor = None
        sub_tensor_mask = None
        if self.sub_tensor_size:
            # hard code 3 way tensor
            a_mode = self.tensor_obj.tensor[search_range[0], :, :]
            b_mode = a_mode[:, search_range[1], :]
            c_mode = b_mode[:, :, search_range[2]]
            sub_tensor = c_mode

            # mask filling
            a_mode_m = self.tensor_obj.mask[search_range[0], :, :]
            b_mode_m = a_mode_m[:, search_range[1], :]
            c_mode_m = b_mode_m[:, :, search_range[2]]
            sub_tensor_mask = c_mode_m

        self.sub_tensor = sub_tensor
        self.sub_tensor_mask = sub_tensor_mask

        # compute density of sub_tensor
        if self.sub_tensor_size:
            self.density = np.sum(self.sub_tensor_mask) / self.sub_tensor_mask.size
        else:
            self.density = 0

    def recover_sub_tensor(self, rank=3):
        if type(self.sub_tensor_recovered) != np.ndarray:
            # lazy computation, check if decomposition is performed
            if type(self.sub_tensor) != np.ndarray:
                print("[Err] Sub-tensor is not ndarray")
            st_cp_obj = TensorDecompCP(
                self.sub_tensor,
                self.sub_tensor_mask,
                rank, iters=100)
            st_cp_obj.run()
            self.sub_tensor_recovered = st_cp_obj.cp_recovered
        return self.sub_tensor_recovered

    def global_ordered_idx_2_recovered_idx(self, gl_entry):

        # A
        idx = [None] * len(gl_entry)
        for i in range(len(gl_entry)):
            idx[i] = \
                self.covered_ranges[i].index(gl_entry[i])
        return idx


def confirm_sub_tensor_list(sub_tensors, alpha, tensor_distance_calc_obj, q, dis_mode):
    """choose ap from ap candidates
     alpha : [0,1], blending factor
     q: 30, total ap to confirm
     l: number of ap already chose
     :type tensor_distance_calc_obj: tensor_distance_calculation
     :return confirm ap list, ap as sub-tensor
     """

    # 1. choose largest ap as first
    largest_ap = sub_tensors[0]
    for st in sub_tensors:
        if st.sub_tensor_size > largest_ap.sub_tensor_size:
            largest_ap = st

    first_ap = largest_ap

    # settings

    l = 1
    confirmed = [first_ap]
    unconfirmed = sub_tensors.copy()
    unconfirmed.remove(first_ap)

    # 2. blending

    while l < q:

        # create current sub-tensor

        density = sub_tensors[l].density

        # 3. blending right part

        blended_max = 0.
        blended_max_tensor_flag = False
        for cur_uncon in unconfirmed:
            if cur_uncon.sub_tensor_size < 1:  # skip empty sub tensor
                continue
            sum_d = 0.
            for cur_con in confirmed:
                sum_d += tensor_distance_calc_obj.calc_dis_entry_hash(
                    cur_con.AP, cur_uncon.AP, dis_mode)
            blended = density * alpha + (sum_d / l) * (1 - alpha)
            if blended > blended_max:
                blended_max = blended
                blended_max_tensor = cur_uncon
                blended_max_tensor_flag = True

        if blended_max_tensor_flag:
            confirmed.append(blended_max_tensor)
            unconfirmed.remove(blended_max_tensor)
            l += 1
        else:  # l<q, but can't find candidate
            break
    return confirmed


def calc_K_correlation(sub_tensor, entry, entry_distance_calc_obj, dis_mode):
    """measure correlation between an entry
    to sub_tensor's anchor point"""
    ap = sub_tensor.AP
    entry_hash = []

    if Debug_flag1:
        t0 = timeit.default_timer()
    for i in range(len(entry)):
        if Debug_flag1:
            t11 = timeit.default_timer()
        code = entry_distance_calc_obj.hash_table_objs[i].FactorMatrix[entry[i]]
        entry_hash.append(
            entry_distance_calc_obj.hash_table_objs[i].Super_LSH_Hash(code))
        if Debug_flag1:
            t12 = timeit.default_timer()
            print(f"Hashing Once took {t12-t11}")
    if Debug_flag1:
        t1 = timeit.default_timer()
    distance = entry_distance_calc_obj.calc_dis_entry_hash(ap, entry_hash, dis_mode)
    if Debug_flag:
        t2 = timeit.default_timer()
    K = 0.
    if distance < sub_tensor.h:
        K = 1 - distance ** 2
    if Debug_flag1:
        t3 = timeit.default_timer()
        t_dict ={
            'SuperHash': t1 - t0,
            'distancing': t2 - t1,
            'other': t3 - t2,
        }
        print(t_dict)
    return K


def fusion_sub_tensors_to_recovered_tensor(sub_tensors, calc_dis_obj, dis_mode):
    # 1. recovered tensor entry is CP recovered tensor entry
    recovered_tensor = sub_tensors[0].tensor_obj.cp_recovered.copy()
    tensor_size = recovered_tensor.size
    if type(recovered_tensor) != np.ndarray:
        print("[FusionErr] CP recovered is None")
        return None

    # 2. calculate K map of all entries covered by sub_tensors

    K_map = {}  # dictionary map of np.array(entry)--K pair

    # loop all entries from sub tensors
    if Debug_flag:
        start = timeit.default_timer()
    for st in sub_tensors:
        # covered_indices of global tensor
        for entry in st.covered_indices_gen():
            key = str(entry)
            K = calc_K_correlation(st, entry, calc_dis_obj, dis_mode)
            if K > 0.:
                if key in K_map:
                    K_map[key] += K
                else:
                    K_map[key] = K

    if Debug_flag:
        end = timeit.default_timer()
        print(f"--------------------------------------K_map init time is {end - start}")
    # 3. loop all sub tensors entries to accumulate

    LTC_entry_map = {}

    if Debug_flag:
        start = timeit.default_timer()
    for st in sub_tensors:
        # 3.1 recover tensor
        recovered_st = st.recover_sub_tensor(st.sub_rank)

        for entry in st.covered_indices_gen():
            # pre-requisites, K does have a non-zero value
            key = str(entry)
            if key in K_map:
                # 3.2 apply K
                sub_indices = st.global_ordered_idx_2_recovered_idx(entry)
                recovered_cell = recovered_st[
                    sub_indices[0],
                    sub_indices[1],
                    sub_indices[2]
                ]
                top = calc_K_correlation(st, entry, calc_dis_obj, dis_mode) \
                      * recovered_cell
                # 3.3 update LTC_entry_map
                if key in LTC_entry_map:
                    LTC_entry_map[key] += top / K_map[key]
                else:
                    LTC_entry_map[key] = top / K_map[key]

    if Debug_flag:
        end = timeit.default_timer()
        print(f"--------------------------------LTC_entry_map init time is {end - start}")

    # 4. weighted recovered data
    if Debug_flag:
        print(f"LTC updated {LTC_entry_map.__len__()/tensor_size *100:.4f}% entries in tensor.")
    for entry_str in LTC_entry_map:
        entry = eval(entry_str)
        old = recovered_tensor[
            entry[0],
            entry[1],
            entry[2]
        ]
        new = LTC_entry_map[entry_str]
        # if new == 0.0:
        #     print("LTC cell is 0.0")
        #     continue
        recovered_tensor[
            entry[0],
            entry[1],
            entry[2]
        ] = LTC_entry_map[entry_str]

    return recovered_tensor


def is_entry_covered_by_sub_tensor(entry, sub_tensor):
    ranges = sub_tensor.covered_ranges
    if entry[0] in ranges[0] and entry[1] in ranges[1] and entry[2] in ranges[2]:
        return True
    return False
