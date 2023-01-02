"""Tensor completion based on re-ordered tensor's factor matrices"""

import numpy as np
import tensorly as tl
from LTC import LTC


def calc_dis_anchor_points(anchor_point_1, anchor_point_2):
    """
    Define distance/similarity between m_{i,j,k} and m_{i',j',k'}
    :param anchor_point_1: Anchor Point 1
    :param anchor_point_2: Anchor Point 2
    :return: Distance/Similarity between anchor points (in Radius)
    """
    d1 = anchor_point_1

def calc_dis_slices(slice_1, slice_2):
    """
    Define distance between a_i and a_i'
    :param slice_1: tensor slice 1 (in vector, coefficents of factor matrices)
    :param slice_2: tensor slice 2 (in vector, coefficents of factor matrices)
    :return: Distance/Similarity between slices (in Radius)
    """
    pass


