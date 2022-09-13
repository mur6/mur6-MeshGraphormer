

from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import json
import code
from manopth.manolayer import ManoLayer
import scipy.sparse
import src.modeling.data.config as cfg



class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []
    
    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse.FloatTensor(i, v, u.shape))
    
    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape)) 

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i,i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat

def get_graph_params(filename, nsize=1):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    A = data['A']
    U = data['U']
    D = data['D']
    U, D = scipy_to_pytorch(A, U, D)
    A = [adjmat_sparse(a, nsize=nsize) for a in A]
    return A, U, D

filename = "src/modeling/data/mano_downsampling.npz"
#aaa = get_graph_params(filename=filename, nsize=1)
#torch.save(aaa, "aaa.pt")



a, u, d = torch.load("experiments/aaa.pt")
#print(type(a[0]))
#print(type(u))
print(type(d))

class Mesh(object):
    """Mesh object that is used for handling certain graph operations."""
    def __init__(self, a: list[torch.Tensor], u: list[torch.Tensor], d: list[torch.Tensor], num_downsampling:int=1, nsize:int=1):
        #print(filename)
        #filename = "src/modeling/data/mano_downsampling.npz"
        self._A = a
        self._U = u
        self._D = d
        self.num_downsampling = num_downsampling

    def downsample(self, x: torch.Tensor, n1:int=0, n2:int=1):
        """Downsample mesh."""
        if n2 is None:
            n2 = self.num_downsampling
        if x.ndim < 3:
            for i in range(n1, n2):
                x = torch.sparse.mm(self._D[i], x)
        elif x.ndim == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in range(n1, n2):
                    y = torch.sparse.mm(self._D[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def upsample(self, x: torch.Tensor, n1:int=1, n2:int=0):
        """Upsample mesh."""
        r = list(range(n2, n1))
        r.reverse()
        if x.ndim < 3:
            for i in r:
                x = torch.sparse.mm(self._U[i], x)
        elif x.ndim == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in r:
                    y = torch.sparse.mm(self._U[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

mesh = Mesh(a, u, d)
#torch.script
x = torch.jit.script(mesh)
#print(x.downsample.graph)
for k in d:
    print(k)
    print(type(k))
