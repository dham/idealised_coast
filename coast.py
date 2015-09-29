from firedrake import *

m = RectangleMesh(100, 20, 50, 10)

V1 = FunctionSpace(m, "RT", 1)
V2 = Functionspace(m, "DG", 0)

W = V1 * V2

v, q = TestFunctions(V1, V2)

solution = Function(W)
u, p = split(solution)

solve((inner(u, v) + p * div(v) + div(q) * u) * dx == 0, solution, bcs=bcs)

File("out.pvd") << solution
