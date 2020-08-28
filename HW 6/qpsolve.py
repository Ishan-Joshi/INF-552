#   Sample code from package tutorial
from qpsolvers import dense_solvers
from qpsolvers import solve_qp
import numpy as np
from scipy.sparse import csc_matrix
from cvxopt import matrix, solvers

Q = 2 * matrix([[2, .5], [.5, 1]])
p = matrix([1.0, 1.0])
G = matrix([[-1.0, 0.0], [0.0, -1.0]])
h = matrix([0.0, 0.0])
A = matrix([[1.0], [1.0]])
b = matrix(1.0)
#sol = solvers.qp(Q, p, G, h, A, b)

#print(sol['x'])
