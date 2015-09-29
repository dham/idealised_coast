from firedrake import *
import numpy as np

m = RectangleMesh(50, 10, 100, 20)

# Parameters for Gaussian.
a = 6
sigma = 5
c = 50.


def gaussian(x):
    return a * np.exp(-((x-50.)**2)/(2*sigma**2))

x = m.coordinates.dat.data

x[:, 1] = 20. + x[:, 1] * (gaussian(x[:, 0]) - 20.)/20.

V1 = FunctionSpace(m, "RT", 1)
V2 = FunctionSpace(m, "DG", 0)

W = V1 * V2

bcs = [DirichletBC(W[0], (1., 0.), 1),
       DirichletBC(W[0], (1., 0.), 2),
       DirichletBC(W[0], (0., 0.), 3),
       DirichletBC(W[0], (0., 0.), 4)]

solution = Function(W)
u, p = split(solution)
v, q = TestFunctions(W)

solve((inner(u, v) + p * div(v) + div(u) * q) * dx == 0, solution, bcs=bcs,
      nullspace=VectorSpaceBasis(constant=True))

u, p = solution.split()

File("u.pvd") << u
File("p.pvd") << p
