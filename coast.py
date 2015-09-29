from firedrake import *
import numpy as np

degree = 1
quadrilateral = True

# Parameters for Gaussian.
a = 6
sigma = 5
c = 50.


def gaussian(x):
    return a * np.exp(-((x-50.)**2)/(2*sigma**2))


def function_space(mesh, degree, quadrilateral):
    """Create the required mixed function space."""

    if quadrilateral:
        if degree == 1:
            u0 = FiniteElement("CG", interval, 1)
            u1 = FiniteElement("DG", interval, 0)
            RT1 = HDiv(OuterProductElement(u0, u1)) + HDiv(OuterProductElement(u1, u0))
            V1 = FunctionSpace(m, RT1)
            V2 = FunctionSpace(m, "DG", 0)
        else:
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


def mesh(quadrilateral):
    if not quadrilateral:
        m = RectangleMesh(50, 10, 100, 20,
                          quadrilateral=False)
    else:
        m_0 = IntervalMesh(50, 100)
        m = ExtrudedMesh(m_0, 10, 20./10.)

    return m


def boundaries(quadrilateral):
    if not quadrilateral:
        bcs = [DirichletBC(W[0], (1., 0.), 1),
               DirichletBC(W[0], (1., 0.), 2),
               DirichletBC(W[0], (0., 0.), 3),
               DirichletBC(W[0], (0., 0.), 4)]
    else:
        bcs = [DirichletBC(W[0], (1., 0.), 1),
               DirichletBC(W[0], (1., 0.), 2),
               DirichletBC(W[0], (0., 0.), "top"),
               DirichletBC(W[0], (0., 0.), "bottom")]
    return bcs

# Create the mesh and insert the gaussian bump.
m = mesh(quadrilateral)
x = m.coordinates.dat.data
x[:, 1] = 20. + x[:, 1] * (gaussian(x[:, 0]) - 20.)/20.

W = function_space(mesh, 1, quadrilateral)

bcs = boundaries(quadrilateral)

solution = Function(W)
u, p = split(solution)
v, q = TestFunctions(W)

solve((inner(u, v) + p * div(v) + div(u) * q) * dx == 0, solution, bcs=bcs,
      nullspace=VectorSpaceBasis(constant=True))

u, p = solution.split()

File("u.pvd") << u
File("p.pvd") << p
