import numpy as np
from LTC import LTC

"""LSH Hashing Table creating """

class LSH_Hash_Table(object):
    """ Super LSH hash table indexing class"""
    def __init__(self, factorMatrix, SuperIndex_N=10):

        #  inti hashing parameters
        self.SuperIndex_N = SuperIndex_N
        self.FactorMatrix = factorMatrix
        self.vector_len = len(factorMatrix[0]) # length of a slice
        self.init_parmeters()
        self.factor_matrix_to_hash_table() # compute hash table "run()"
        self.calc_hash_table_max_distance()
        self.calc_hash_table_min_distance()

    def init_parmeters(self):
        # random functions
        self.rnd_a_func = \
            lambda length, times: [np.random.uniform(-1, 1, length) for i in range(times)]
        self.rnd_int = \
            lambda times: np.random.randint(0, 1000, times)

        # provide N(0,1) random projected vectors {a0 ... an}
        self.vectors_a = self.rnd_a_func(self.vector_len, self.SuperIndex_N)
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
                         * LTC.LSH_Hash(code, self.vectors_a[i])

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
        for begin in self.hash_table:
            for end in self.hash_table:
                if begin is not end:
                    if np.abs(begin-end) > cur_max:
                        cur_max = begin - end
        self.hash_table_max_distance = cur_max

    def calc_hash_table_min_distance(self):
        """calculate min key distance of hash table"""

        # list keys of hash table
        keys = list(self.hash_table.keys())
        # default min equals distance between first and second
        cur_min = keys[0] - keys[1]
        for begin in self.hash_table:
            for end in self.hash_table:
                if begin is not end:
                    if np.abs(begin-end) < cur_min:
                        cur_min = begin - end

        self.hash_table_min_distance = cur_min


