import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
from torch import nn
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.hash import HashTable

import torch

a = torch.tensor([[0, 2.], [3, 0]])
print(a)
print(a.to_sparse())   # only values and indices of non-zero elements are stored

i = [[0, 1, 1],
     [2, 0, 2]]

v = [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 3))
print(s)

print(s.to_dense())

features = torch.randn(4, 2)
print(features.shape)
indices = torch.tensor([[0, 0, 0, 1],
                        [0, 1, 1, 1],
                        [1, 0, 1, 1],
                        [1, 2, 2, 2]], dtype=torch.int32)
print(indices)
print(indices[:, 1].shape)
print(indices[:, 1])
spatial_shape = torch.tensor([4, 4, 4])
batch_size = 2
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x_dense_NCHW = x.dense()
print(x_dense_NCHW.shape)