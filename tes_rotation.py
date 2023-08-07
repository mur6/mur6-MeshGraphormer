import trimesh
from trimesh import transformations
from trimesh.transformations import rotation_matrix, concatenate_matrices, euler_from_matrix, euler_matrix
import numpy as np


alpha, beta, gamma = 0, 0, np.pi
origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
# I = identity_matrix()
Rx = rotation_matrix(alpha, xaxis)
Ry = rotation_matrix(beta, yaxis)
Rz = rotation_matrix(gamma, zaxis)
R = concatenate_matrices(Rx, Ry, Rz)
euler = euler_from_matrix(R, 'rxyz')

Re = euler_matrix(alpha, beta, gamma, 'rxyz')

print(Re)
