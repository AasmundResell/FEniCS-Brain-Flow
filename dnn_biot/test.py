from fenics import *
from dolfin_adjoint import *
from ufl_dnn.neural_network import ANN
from matplotlib.pyplot import show

# Manufature solution

import sympy as sym

x, y = sym.symbols("x[0], x[1]")

K = 1.0
u = sym.sin(sym.pi * x) * sym.sin(sym.pi * y)
f_ex = sym.diff(K * sym.diff(u, x), x, 1) + sym.diff(K * sym.diff(u, y), y, 1)

variables = [
    u,
    K,
    f_ex,
]

variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

UFLvariables = [Expression(var, degree=2) for var in variables]

u_ex, K, f_ex = UFLvariables

Nx, Ny = 20, 20
mesh = UnitSquareMesh(Nx, Ny)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
layers = [4, 10, 1]
f_ex = project(f_ex, V)
obs = project(u_ex, V)
plot(f_ex)

show()

bias = [True, True]
x, y = SpatialCoordinate(mesh)

net = ANN(layers, bias=bias, mesh=mesh)
E = K * inner(grad(u), grad(v)) * dx + net(x, y) * v * dx
bcs = DirichletBC(V, Constant(0.0), "on_boundary")
hat_u = Function(V)

# Solve PDE
solve(lhs(E) == rhs(E), hat_u, bcs)

plot(hat_u)

show()


# L ^ 2 error as loss
loss = assemble((hat_u - obs) ** 2 * dx)  # Loss funtction

# Define reduced formulation of problem
hat_loss = ReducedFunctional(loss, net.weights_ctrls())

# Use scipy L - BFGS optimiser
opt_theta = minimize(
    hat_loss, options={"disp": True, "gtol": 1e-12, "ftol": 1e-12, "maxiter": 20}
)
print(opt_theta)
net.set_weights(opt_theta)


#assert assemble(net(x, y) ** 2 * dx) < 1e-6

u_test = Function(V)
E_test = K * inner(grad(u), grad(v)) * dx + net(x, y) * v * dx

solve(lhs(E_test) == rhs(E_test),u_test,bcs)

f_pred = project(net(x,y),V)
plot(f_pred)


show()


plot(u_test)

show()
