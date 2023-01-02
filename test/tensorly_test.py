import tensorly as tl

import numpy as np

# create a random 10x10x10 tensor
tensor = np.random.random((10, 10, 10))

# mode-1 unfolding (i.e. zeroth mode)
unfolded = tl.unfold(tensor, mode=0)
# refold the unfolded tensor
folded = tl.fold(unfolded, mode=0, shape=tensor.shape)



diff3D = np.ones((10,10,10))
for i in range(tensor.shape[0]):
    for j in range(tensor.shape[1]):
        for k in range(tensor.shape[2]):
            diff3D[i,j,k] = tensor[i,j,k] - folded[i,j,k]

print(diff3D.shape)

#import Abilene dataset

import os
import scipy.io as sio

data_dir = os.path.join(os.getcwd(), '../data')
mat_fname = os.path.join(data_dir, 'Abilene.mat')
mat_contents = sio.loadmat(mat_fname)

# data matrix Abilene as 'A'
# A[i, j, k]
# i: 288, 5-minutes time intervals of a day, 12*24 = 288
# j: 169, 169 days of collected data
# k: 144,  12 - 12 nodes ping matrix

A = mat_contents['A']

# visualization of a layer of A
import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.heatmap(A[:,:,1])
plt.show()

