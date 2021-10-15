from fenics import *
from matplotlib.pyplot import show
from dolfin_adjoint import *
from ufl_dnn.neural_network import ANN

class BoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
def a_u(u, v):
    return my * (inner(grad(u), grad(v)) + inner(grad(u), nabla_grad(v))) * dx

def a_p(K, p, q):
    return K * dot(grad(p), grad(q)) * dx

def b(s, v):
    return s * div(v) * dx

def c(alpha, p, q):
    return alpha / Lambda * dot(p, q) * dx

def F(f, v):
    return dot(f, v) * dx

def test_SteadyBiot_dnn():
    
    import sympy as sym

    x, y = sym.symbols("x[0], x[1]")
    my = 1 / 3
    Lambda = 16666
    alpha = 1.0
    c = 1.0
    K = 1.0
    u =  (
        sym.sin(2 * sym.pi * y) * (-1 + sym.cos(2 * sym.pi * x))
        + 1 / (my + Lambda) * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)
    )
    v =  (
        sym.sin(2 * sym.pi * x) * (1 - sym.cos(2 * sym.pi * y))
        + 1 / (my + Lambda) * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)
    )
    p1 = -1 * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)  # p Network1
    p0 = Lambda * (sym.diff(u, x, 1) + sym.diff(v, y, 1)) - alpha * p1

    fx,fy = 0.0,0.0 #force term

    g_ex = -K * (sym.diff(p1, x, 2) + sym.diff(p1, y, 2))
    variables = [
        u,
        v,
        p0,
        p1,
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        g_ex,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [Expression(var, degree=2) for var in variables]

    (
        u,
        v,
        p0,
        p1,
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        g_ex,
    ) = UFLvariables
    f = as_vector((fx, fy))
    mesh = UnitSquareMesh(10, 10)
    
    g = [g_ex]
    alpha = [1, alpha]
    c = [c]
    K = [K]
    
# Generate function space
    V = VectorElement("CG", triangle, 2)  # Displacement
    Q_0 = FiniteElement("CG", triangle, 1)  # Total pressure
    Q_1 = FiniteElement("CG", triangle, 1) #Network 1
    mixedElement = []
    mixedElement.append(V)
    mixedElement.append(Q_0)
    mixedElement.append(Q_1)
    
    W_element = MixedElement(mixedElement)
    W = FunctionSpace(mesh, W_element)

    test = TestFunction(W)
    q = split(test)  # q[0] = v, q[1],q[2],... = q_0,q_1,...

    trial = TrialFunction(W)
    p_ = split(trial)  # p_[0] = u_, p_[1],p_[2],... = p_0,p_1,...

    up_n = Function(W)
    p_n = split(up_n)  # p_n[0] = u_n, p_n[1],p_n[2],... = p0_n,p1_n,...

    # variational formulation
    sources = []  # Contains the source term for each network
    innerProdP = (
        []
    )  # Contains the inner product of the gradient of p_j for each network
    dotProdP = []  # Contains the dot product of alpha_j & p_j,
    bcs_D = []  # Contains the terms for the Dirichlet boundaries
    integrals_N = []  # Contains the integrals for the Neumann boundaries

    x, y = SpatialCoordinate(mesh)

    layers = [4, 10, 1]
    g_ex = project(g_ex, V)
    obs = project(u_ex, V)

    plot(g_ex)

    show()
    
    bias = [True, True]
    x, y = SpatialCoordinate(mesh)
    
    net = ANN(layers, bias=bias, mesh=mesh)
    E = K * inner(grad(u), grad(v)) * dx + net(x, y) * v * dx
    bcs = DirichletBC(V, Constant(0.0), "on_boundary")
    hat_u = Function(V)
    
    # Solve PDE
    solve(lhs(E) == rhs(E), hat_u, bcs)
    

    # L ^ 2 error as loss
    loss = assemble((hat_u - obs) ** 2 * dx)  # Loss funtction

    # Define reduced formulation of problem
    hat_loss = ReducedFunctional(loss, net.weights_ctrls())
    
    # Use scipy L - BFGS optimiser
    opt_theta = minimize(
        hat_loss, options={"disp": True, "gtol": 1e-12, "ftol": 1e-12, "maxiter": 80}
    )
    net.set_weights(opt_theta)
    

    #assert assemble(net(x, y) ** 2 * dx) < 1e-6
    
#    u_test = Function(V)
#    E_test = K * inner(grad(u), grad(v)) * dx + net(x, y) * v * dx
    
#    solve(lhs(E_test) == rhs(E_test),u_test,bcs)
    
#    f_pred = project(net(x,y),W.sub(2))
#    plot(f_pred)
    
    
#    show()

    
#    plot(u_test)

#    show()

    u_e = Expression((variables[0], variables[1]), degree=2)
    V_e = VectorFunctionSpace(mesh, "P", 2)
    Q_e = FunctionSpace(mesh, "P", 1)
    u_e = project(u_e, V_e)
    p_e1 = project(UFLvariables[3], Q_e)

    vtkUfile = File("solution_steady/u.pvd")
    vtkPfile = File("solution_steady/p1.pvd")
    vtkUfile << u
    vtkPfile << p[1]
    vtkUEfile = File("solution_steady/u_e.pvd")
    vtkPEfile = File("solution_steady/p_e1.pvd")
    vtkUEfile << u_e
    vtkPEfile << p_e1
    er2U = errornorm(u_e, u, "L2")
    print("Error L2 for velocity = ", er2U)
    er2P = errornorm(p_e1, p[1], "L2")
    print("Error L2 for pressure = ", er2P)
    plot(p[1])

    show()

if __name__ == "__main__":
    test_SteadyBiot_dnn()
