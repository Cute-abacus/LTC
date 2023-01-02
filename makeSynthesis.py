import numpy as np
import tensorly as tl


def makeSynthesis(dims, rank):
    """make synthesis low rank tensor
    tensor = Σ_r (a_r ○ b_r ○ c_r)
    100 * 100 * 100 , rank = 3
    """

    tensors = []
    for k in range(2):
        tensors_lvl1 = []
        for i in range(2):
            tensors_lvl2 = []
            for j in range(2):
                # 1. random [A, B, C]
                factors = []
                for dim in dims:
                    matrix = np.random.rand(dim, rank)
                    factors.append(matrix)

                # 2. Kruskal's tensor product
                weights = np.ones((rank,))

                sub_tensor = tl.kruskal_to_tensor((weights, factors))
                tensors_lvl2.append(sub_tensor)
            sub_tensor_lvl1 = np.concatenate(tensors_lvl2, axis=0)
            tensors_lvl1.append(sub_tensor_lvl1)
        tensor = np.concatenate(tensors_lvl1, axis=2)
        tensors.append(tensor)
    Concat_tensor = np.concatenate(tensors,axis=1)

    return Concat_tensor

def bench(tensor):

    import LTC2

    # config mask
    mask = LTC2.TensorDecompCP.random_mask(tensor.shape, p=.4)

    # define cp rank
    rank = 2

    # link data to CP decomposition object
    TD_obj = LTC2.TensorDecompCP(tensor, mask, rank, iters=10)

    train_err, predit_err = TD_obj.run()  # perform CP decomposition
    print('train error is {}\n'
          'predi error is {}'.format(train_err, predit_err))


if __name__ == "__main__":

    params = {
        'dims': [20, 20, 20],
        'rank': 1,
    }
    bench(makeSynthesis(**params))

