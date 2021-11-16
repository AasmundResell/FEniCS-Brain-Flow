from biot import biotMPET
from fenics import *
from mshr import *
from matplotlib import pyplot
from math import log
#import biot
#import imp
#imp.reload(biot)

class BoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


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
    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)

    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries

    U = as_vector((u, v))
    # Define boundary conditions
    boundary_conditionsU = {1: {"Dirichlet": U}}
    # First key denotes the network, the second denotes the boundary marker
    boundary_conditionsP = {
        (1, 1): {"Dirichlet": p1},
        (2, 1): {"Dirichlet": p2},
    }

    g = [g1, g2]
    alpha = [1, alpha, alpha]
    c = [c, c]
    K = [K, K]
    u, p = biotMPET(
        mesh,
        0,
        0,
        2,
        f,
        alpha,
        K,
        c,
        my,
        Lambda,
        boundary_conditionsU,
        boundary_conditionsP,
        boundary_markers,
        g=g,
        boundaryNum=1,
        transient=False,
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
    vtkPfile << p[1]
    vtkUEfile = File("solution_steady/u_e.pvd")
    vtkPEfile = File("solution_steady/p_e1.pvd")
    vtkUEfile << u_e
    vtkPEfile << p_e1
    er2U = errornorm(u_e, u, "L2")
    print("Error L2 for displacements = ", er2U)
    er2P = errornorm(p_e1, p[1], "L2")
    print("Error L2 for pressure = ", er2P)
    plot(p[1])

    show()


def test_TransientBiot():

    import sympy as sym

    x, y = sym.symbols("x[0], x[1]")
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

    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)

    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries

    U = as_vector((u, v))
    # Define boundary conditions
    boundary_conditionsU = {1: {"Dirichlet": U}}
    # First key denotes the network, the second denotes the boundary marker
    boundary_conditionsP = {
        (1, 1): {"Dirichlet": p1},
        (2, 1): {"Dirichlet": p2},
    }
    g = [g1, g2]
    alpha = [1, alpha, alpha]
    cj = [c, c]
    K = [K, K]
    p_initial = [p0,p1,p2]
    T = 0.5
    numTsteps = 40
    u, p = biotMPET(
        mesh,
        T,
        numTsteps,
        2,
        f,
        alpha,
        K,
        cj,
        my,
        Lambda,
        boundary_conditionsU,
        boundary_conditionsP,
        boundary_markers,
        g=g,
        p_initial = p_initial,
        boundaryNum=1,
        transient=True,
    )

    # Post processing
    u_e = Expression((variables[0], variables[1]), degree=2, t=T)
    p1_e = Expression(variables[3], degree=2, t=T)
    p2_e = Expression(variables[4], degree=2, t=T)
    V_e = VectorFunctionSpace(mesh, "P", 2)
    Q1_e = FunctionSpace(mesh, "P", 1)
    Q2_e = FunctionSpace(mesh, "P", 1)
    u_e = project(u_e, V_e)
    p1_e = project(p1_e, Q1_e)
    p2_e = project(p2_e, Q2_e)
    vtkUEfile = File("solution_transient/u_e.pvd")
    vtkPEfile = File("solution_transient/p1_e.pvd")
    vtkUEfile << u_e
    vtkPEfile << p1_e
    er2U = errornorm(u_e, u, "L2")
    print("Error L2 for displacements = ", er2U)
    er2P = errornorm(p1_e, p[1], "L2")
    print("Error L2 for pressure = ", er2P)

    pu_e = plot(u_e)

    # set colormap
    pu_e.set_cmap("viridis")
    #pu_e.set_clim( 0.0 , 1.0 )

    # add a title to the plot
    #pyplot.title("u_e")

    # add a colorbar
    pyplot.colorbar( pu_e );
    pyplot.show()

    pu = plot(u)

    # set colormap
    pu.set_cmap("viridis")
    #pu.set_clim( 0.0 , 1.0 )

    # add a title to the plot
    #pyplot.title("u")

    # add a colorbar
    pyplot.colorbar( pu );
    pyplot.show()

    pp_e = plot(p1_e)

    # set colormap
    pp_e.set_cmap("viridis")

    # add a colorbar
    pyplot.colorbar( pp_e );
    pyplot.show()

    pp = plot(p[1])

    # set colormap
    pp.set_cmap("viridis")
    #pp.set_clim( 0.0 , 1.0 )

    # add a colorbar
    pyplot.colorbar( pp );
    pyplot.show()

    pp2_e = plot(p2_e)

    # set colormap
    pp2_e.set_cmap("viridis")

    # add a colorbar
    pyplot.colorbar( pp2_e );
    pyplot.show()

    pp2 = plot(p[2])

    # set colormap
    pp2.set_cmap("viridis")

    # add a colorbar
    pyplot.colorbar( pp2 );
    pyplot.show()

def test_meshRefineBiot():

    import sympy as sym

    x, y = sym.symbols("x[0], x[1]")
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
    U = as_vector((u, v))

    # Define boundary conditions
    boundary_conditionsU = {1: {"Dirichlet": U}}
    # First key denotes the network, the second denotes the boundary marker
    boundary_conditionsP = {
        (1, 1): {"Dirichlet": p1},
        (2, 1): {"Dirichlet": p2},
    }
    
    g = [g1, g2]
    alpha = [1, alpha, alpha]
    cj = [c, c]
    K = [K, K]
    p_initial = [p0,p1,p2]
    T = 0.5
    numTsteps = 4


    mesh = UnitSquareMesh(4, 4)
    erroruL2 = []
    errorpL2 = []
    erroruH1 = []
    errorpH1 = []
    h = []
    for n in range(5):
        
        plot(mesh)
        show()
        # Define boundary
        boundary_markers = MeshFunction("size_t", mesh, 1)
        boundary_markers.set_all(9999)

        bx0 = BoundaryOuter()
        bx0.mark(boundary_markers, 1)  # Applies for all boundaries
        
    
        u, p = biotMPET(
            mesh,
            T,
            numTsteps,
            2,
            f,
            alpha,
            K,
            cj,
            my,
            Lambda,
            boundary_conditionsU,
            boundary_conditionsP,
            boundary_markers,
            g = g,
            p_initial = p_initial,
            boundaryNum=1,
            transient=True,
        )

        # Post processing
        u_e = Expression((variables[0], variables[1]), degree=4, t=T)
        p1_e = Expression(variables[3], degree=3, t=T)
        V_e = VectorFunctionSpace(mesh, "P", 4)
        Q_e = FunctionSpace(mesh, "P", 3)
        u_e = project(u_e, V_e)
        p1_e = project(p1_e, Q_e)

        er2U = errornorm(u_e, u, "L2",mesh=mesh)
        print("Error L2 for displacements = ", er2U)
        er2P = errornorm(p1_e, p[1], "L2",mesh=mesh)
        print("Error L2 for pressure = ", er2P)
        er1U = errornorm(u_e, u, "H1",mesh=mesh)
        print("Error H1 for displacements = ", er1U)
        er1P = errornorm(p1_e, p[1], "H1",mesh=mesh)
        print("Error H1 for pressure = ", er1P)
        h.append(mesh.hmin())
        erroruL2.append(er2U)
        errorpL2.append(er2P)
        erroruH1.append(er1U)
        errorpH1.append(er1P)
        del p
        mesh = refine(mesh)

    print("h = ",h)
    print("Convergence rate for displacement using L2: ",convergenceRate(erroruL2, h))
    print("Convergence rate for pressure 1 using L2: ",convergenceRate(errorpL2, h))
    print("Convergence rate for displacement using H1: ",convergenceRate(erroruH1, h))
    print("Convergence rate for pressure 1 using H1: ",convergenceRate(errorpH1, h))
    
def convergenceRate(error,h):
    rate = [log(error[i+1]/error[i])/log(h[i+1]/h[i]) for i in range(len(error)-1)]
    return rate

if __name__ == "__main__":
#    test_SteadyBiot()
    test_TransientBiot()
#    test_meshRefineBiot()
