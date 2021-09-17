from biot import *
from fenics import *
from mshr import *
from matplotlib.pyplot import show


def test_SteadyBiot():

    import sympy as sym

    x, y = sym.symbols("x[0], x[1]")
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
