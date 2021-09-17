from fenics import *
from mshr import *
from matplotlib.pyplot import show


def biotMPET_improved(
    mesh, T, numTsteps, numPnetworks, f, g, alpha, K, cj, my, Lambda, typeS=False
):
    if not typeS:
        dt = 0
        numTsteps = 1
        progress = 0
    elif typeS:  # Will need to be saved during calculation
        dt = T / numTsteps
        print(dt)
        vtkU = File("solution_transient/u.pvd")
        vtkP = File("solution_transient/p.pvd")
        # Add progress bar
        progress = Progress("Time-stepping", numTsteps)
        set_log_level(LogLevel.PROGRESS)

    # Generate function space
    V = VectorElement("P", triangle, 2)  # Displacement
    Q_0 = FiniteElement("P", triangle, 1)  # Total pressure
    mixedElement = []
    mixedElement.append(V)
    mixedElement.append(Q_0)
    for i in range(numPnetworks):
        Q = FiniteElement("P", triangle, 1)
        mixedElement.append(Q)

    W_element = MixedElement(mixedElement)
    W = FunctionSpace(mesh, W_element)

    test = TestFunction(W)
    q = split(test)  # q[0] = v, q[1],q[2],... = q_0,q_1,...

    trial = TrialFunction(W)
    p_ = split(trial)  # p_[0] = u_, p_[1],p_[2],... = p_0,p_1,...

    up_n = Function(W)
    p_n = split(up_n)  # p_n[0] = u_n, p_n[1],p_n[2],... = p0_n,p1_n,...

    bcu = DirichletBC(W.sub(0).sub(0), Constant(0.0), "on_boundary")
    bcv = DirichletBC(W.sub(0).sub(1), Constant(0.0), "on_boundary")
    bcp0 = DirichletBC(W.sub(1), Constant(0.0), "on_boundary")

    bcs = [bcu, bcv, bcp0]

    # variational formulation

    sources = []  # Contains the source term for each network
    innerProdP = (
        []
    )  # Contains the inner product of the gradient of p_j for each network
    dotProdP = []  # Contains the dot product of alpha_j & p_j,
    timeD_ = []  # Time derivative for the current step
    timeD_n = []  # Time derivative for the previous step

    def a_u(u, v):
        return 2 * my * inner(epsilon(u), epsilon(v)) * dx

    def a_p(K, p, q):
        return K * inner(grad(p), grad(q)) * dx

    def b(s, v):
        return s * div(v) * dx

    def c(alpha, p, q):
        return alpha / Lambda * dot(p, q) * dx

    def F(f, v):
        return dot(f, v) * dx

    for i in range(numPnetworks):  # apply for each network
        print(i)

        bcp = DirichletBC(W.sub(i + 2), Constant(0.0), "on_boundary")
        bcs.append(bcp)
        dotProdP.append(c(alpha[i + 1], p_[i + 2], q[1]))
        sources.append(F(g[i], q[i + 2]))
        innerProdP.append(a_p(K[i], p_[i + 2], q[i + 2]))
        if typeS:  # implicit euler
            timeD_.append(
                (
                    (1 / dt)
                    * (
                        cj[i] * p_[i + 2]
                        + alpha[i] / Lambda * sum(a * b for a, b in zip(alpha, p_[1:]))
                    )
                )
                * q[i + 2]
                * dx
            )
            timeD_n.append(
                (
                    (1 / dt)
                    * (
                        cj[i] * p_n[i + 2]
                        + alpha[i] / Lambda * sum(a * b for a, b in zip(alpha, p_n[1:]))
                    )
                )
                * q[i + 2]
                * dx
            )

    dotProdP.append(c(alpha[0], p_[1], q[1]))

    lhs = (
        a_u(p_[0], q[0])
        + b(p_[1], q[0])
        + b(q[1], p_[0])
        - sum(dotProdP)
        + sum(innerProdP)
        + sum(timeD_)
    )

    rhs = F(f, q[0]) + sum(sources) + sum(timeD_n)

    A = assemble(lhs)
    [bc.apply(A) for bc in bcs]

    up = Function(W)

    t = 0
    print(dt)
    for i in range(numTsteps):
        t += dt
        f[0].t = t
        f[1].t = t
        g[0].t = t
        g[1].t = t
        b = assemble(rhs)
        [bc.apply(b) for bc in bcs]
        solve(A, up.vector(), b)

        if typeS:
            vtkU << (up.sub(0), t)
            vtkP << (up.sub(1), t)
        up_n.assign(up)
        progress += 1
    res = []
    res = split(up)

    u = project(res[0], W.sub(0).collapse())
    p0 = project(res[1], W.sub(1).collapse())
    p1 = project(res[2], W.sub(1).collapse())
    p2 = project(res[3], W.sub(1).collapse())

    return u, p0, p1, p2


def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)


def test_SteadyBiot():

    import sympy as sym

    x, y = sym.symbols("x[0], x[1]")  # needed by UFL
    my = 1 / 3
    Lambda = 16666
    alpha = 1.0
    c = 1.0
    K = 1.0
    t = 1.0
    u = t * (
        sym.sin(2 * sym.pi * y) * (-1 + sym.cos(2 * sym.pi * x))
        + 1 / (my + Lambda) * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)
    )
    v = t * (
        sym.sin(2 * sym.pi * x) * (1 - sym.cos(2 * sym.pi * y))
        + 1 / (my + Lambda) * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)
    )
    p1 = -t * 1 * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)  # p Network1
    p2 = -t * 2 * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)  # p Network2
    p0 = Lambda * (sym.diff(u, x, 1) + sym.diff(v, y, 1)) - alpha * (p1 + p2)
    epsilonxx = sym.diff(u, x, 1)
    epsilonyy = sym.diff(v, y, 1)
    epsilonxy = 1 / 2 * (sym.diff(u, y, 1) + sym.diff(v, x, 1))
    fx = -2 * my * (sym.diff(epsilonxx, x, 1) + sym.diff(epsilonxy, y, 1)) - sym.diff(
        p0, x, 1
    )  # calculate force term x
    fy = -2 * my * (sym.diff(epsilonxy, x, 1) + sym.diff(epsilonyy, y, 1)) - sym.diff(
        p0, y, 1
    )  # calculate force term y
    g1 = -K * (sym.diff(p1, x, 2) + sym.diff(p1, y, 2))  # calculate source term 1
    g2 = -K * (sym.diff(p2, x, 2) + sym.diff(p2, y, 2))  # calculate source term 1
    variables = [
        u,
        v,
        p0,
        p1,
        p2,
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        g1,
        g2,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [Expression(var, degree=2) for var in variables]

    (
        u,
        v,
        p0,
        p1,
        p2,
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        g1,
        g2,
    ) = UFLvariables
    f = as_vector((fx, fy))
    mesh = UnitSquareMesh(10, 10)

    # exact solution, need to make copies from c-strings
    # instead of assinging directly from UFLvariables?
    g = [g1, g2]
    alpha = [1, alpha, alpha]
    c = [c, c]
    K = [K, K]
    u, p0, p1, p2 = biotMPET_improved(
        mesh, 0, 0, 2, f, g, alpha, K, c, my, Lambda, False
    )

    # Post processing
    u_e = Expression((variables[0], variables[1]), degree=2)
    V_e = VectorFunctionSpace(mesh, "P", 2)
    Q_e = FunctionSpace(mesh, "P", 1)
    u_e = project(u_e, V_e)
    p_e1 = project(UFLvariables[3], Q_e)

    vtkUfile = File("solution_steady/u.pvd")
    vtkPfile = File("solution_steady/p1.pvd")
    vtkUfile << u
    vtkPfile << p1
    vtkUEfile = File("solution_steady/u_e.pvd")
    vtkPEfile = File("solution_steady/p_e1.pvd")
    vtkUEfile << u_e
    vtkPEfile << p_e1
    er2U = errornorm(u_e, u, "L2")
    print("Error L2 for velocity = ", er2U)
    er2P = errornorm(p_e1, p1, "L2")
    print("Error L2 for pressure = ", er2P)
    plot(p1)

    show()


def test_TransientBiot():

    import sympy as sym

    x, y = sym.symbols("x[0], x[1]")  # needed by UFL
    my = 1 / 3
    Lambda = 16666
    alpha = 1.0
    c = 1.0
    K = 1.0
    t = sym.symbols("t")
    u = t * (
        sym.sin(2 * sym.pi * y) * (-1 + sym.cos(2 * sym.pi * x))
        + 1 / (my + Lambda) * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)
    )
    v = t * (
        sym.sin(2 * sym.pi * x) * (1 - sym.cos(2 * sym.pi * y))
        + 1 / (my + Lambda) * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)
    )
    p1 = -t * 1 * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)  # p Network1
    p2 = -t * 2 * sym.sin(sym.pi * x) * sym.sin(sym.pi * y)  # p Network2
    p0 = Lambda * (sym.diff(u, x, 1) + sym.diff(v, y, 1)) - alpha * (p1 + p2)
    epsilonxx = sym.diff(u, x, 1)
    epsilonyy = sym.diff(v, y, 1)
    epsilonxy = 1 / 2 * (sym.diff(u, y, 1) + sym.diff(v, x, 1))
    fx = -2 * my * (sym.diff(epsilonxx, x, 1) + sym.diff(epsilonxy, y, 1)) - sym.diff(
        p0, x, 1
    )  # calculate force term x
    fy = -2 * my * (sym.diff(epsilonxy, x, 1) + sym.diff(epsilonyy, y, 1)) - sym.diff(
        p0, y, 1
    )  # calculate force term y
    g1 = (
        +c * sym.diff(p1, t, 1)
        + alpha
        / Lambda
        * (
            1 * sym.diff(p0, t, 1)
            + alpha * sym.diff(p1, t, 1)
            + alpha * sym.diff(p2, t, 1)
        )
        - K * (sym.diff(p1, x, 2) + sym.diff(p1, y, 2))
    )  # calculate source term 1
    g2 = (
        +c * sym.diff(p2, t, 1)
        + alpha
        / Lambda
        * (
            1 * sym.diff(p0, t, 1)
            + alpha * sym.diff(p1, t, 1)
            + alpha * sym.diff(p2, t, 1)
        )
        - K * (sym.diff(p2, x, 2) + sym.diff(p2, y, 2))
    )  # calculate source term 1
    print(g1)
    print(g2)
    variables = [
        u,
        v,
        p0,
        p1,
        p2,
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        g1,
        g2,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [Expression(var, degree=2, t=0) for var in variables]

    (
        u,
        v,
        p0,
        p1,
        p2,
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        g1,
        g2,
    ) = UFLvariables
    f = as_vector((fx, fy))
    mesh = UnitSquareMesh(15, 15)

    # exact solution, need to make copies from c-strings
    # instead of assinging directly from UFLvariables?
    g = [g1, g2]
    alpha = [1, alpha, alpha]
    cj = [c, c]
    K = [K, K]
    T = 0.5
    numTsteps = 40
    u, p0, p1, p2 = biotMPET_improved(
        mesh, T, numTsteps, 2, f, g, alpha, K, cj, my, Lambda, True
    )

    # Post processing
    u_e = Expression((variables[0], variables[1]), degree=2, t=T)
    p1_e = Expression(variables[3], degree=2, t=T)
    V_e = VectorFunctionSpace(mesh, "P", 2)
    Q_e = FunctionSpace(mesh, "P", 1)
    u_e = project(u_e, V_e)
    p1_e = project(p1_e, Q_e)
    vtkUEfile = File("solution_transient/u_e.pvd")
    vtkPEfile = File("solution_transient/p1_e.pvd")
    vtkUEfile << u_e
    vtkPEfile << p1_e
    er2U = errornorm(u_e, u, "L2")
    print("Error L2 for velocity = ", er2U)
    er2P = errornorm(p1_e, p1, "L2")
    print("Error L2 for pressure = ", er2P)

    plot(u)

    show()


test_TransientBiot()
