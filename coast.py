from firedrake import *
import numpy as np

degree = 2
quadrilateral = False

# Parameters for Gaussian.
a = 6
sigma = 5
c = 50.


def gaussian(x):
    return a * np.exp(-((x-50.)**2)/(2*sigma**2))


def function_space(mesh, degree, quadrilateral):
    """Create the required mixed function space."""

    if quadrilateral:
        raise NotImplementedError
    else:
        if degree == 1:
            V1 = FunctionSpace(m, "RT", 1)
            V2 = FunctionSpace(m, "DG", 0)

        elif degree == 2:
            V1 = FunctionSpace(m, "BDFM", 2)
            V2 = FunctionSpace(m, "DG", 1)

        else:
            raise ValueError("Degree must be 1 or 2")

        return V1 * V2


# Create the mesh and insert the gaussian bump.
m = RectangleMesh(50, 10, 100, 20,
                  quadrilateral=quadrilateral)
x = m.coordinates.dat.data
x[:, 1] = 20. + x[:, 1] * (gaussian(x[:, 0]) - 20.)/20.

W = function_space(mesh, 1, quadrilateral)

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
