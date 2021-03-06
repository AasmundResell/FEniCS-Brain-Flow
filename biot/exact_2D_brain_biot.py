from biot import biotMPET
#import biot
from meshes.brain_mesh_2D import generate_2D_brain_mesh_exact, generate_2D_brain_mesh
#import pylab as plt
from fenics import *
from mshr import *
from test_exact_biot import convergenceRate
#from vedo.dolfin import plot,show, Latex
from matplotlib import pyplot
#import imp
#imp.reload(biot)

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def exact_2D_brain_biot_test_steady():
    import sympy as sym

    x, y = sym.symbols("x[0], x[1]")
    my = 1 / 3
    Lambda = 16666
    alpha = 1.0
    c = 1.0
    K = 1.0
    t = 1.0
    u = t * (
        sym.sin(2 * sym.pi * y*1e-3) * (-1 + sym.cos(2 * sym.pi * x*1e-3))
        + 1 / (my + Lambda) * sym.sin(sym.pi * x*1e-3) * sym.sin(sym.pi * y*1e-3)
    )
    v = t * (
        sym.sin(2 * sym.pi * x*1e-3) * (1 - sym.cos(2 * sym.pi * y*1e-3))
        + 1 / (my + Lambda) * sym.sin(sym.pi * x*1e-3) * sym.sin(sym.pi * y*1e-3)
    )
    p1 = -t * 1 * sym.sin(sym.pi * x*1e-3) * sym.sin(sym.pi * y*1e-3)  # p Network1
    p0 = Lambda * (sym.diff(u, x, 1) + sym.diff(v, y, 1)) - alpha * p1
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
        g1,
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
        g1,
    ) = UFLvariables
    f = as_vector((fx, fy))
    U = as_vector((u, v))

    g = [g1]
    alpha = [1, alpha]
    c = [c]
    K = [K]

    mesh = generate_2D_brain_mesh(15)

    n = FacetNormal(mesh)  # normal vector on the boundary
    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)

    bx0 = Boundary()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries

    # Define boundary conditions
    boundary_conditionsU = {1: {"Dirichlet": U}}

    # Define boundary conditions
    boundary_conditionsP = {
        (1, 1): {"Dirichlet": p1},  # Network 1, boundarymarker 1
    }

    u, p = biotMPET(
        mesh,
        0,
        0,
        1,
        f,
        alpha,
        K,
        c,
        my,
        Lambda,
        boundary_conditionsU,
        boundary_conditionsP,
        boundary_markers,
        g = g,
        boundaryNum=1,
        transient=False,
    )

    # Post processing
    V_e = VectorFunctionSpace(mesh, "P", 2)
    Q_e = FunctionSpace(mesh, "P", 1)

    u_e = project(Expression((variables[0], variables[1]), degree=2), V_e)
    p1_e = project(Expression(variables[3], degree=2), Q_e)

    vtkUEfile = File("solution_exact2D_brain/u_e.pvd")
    vtkPEfile = File("solution_exact2D_brain/p1_e.pvd")
    vtkUEfile << u_e
    vtkPEfile << p1_e

    vtkUEfile = File("solution_exact2D_brain/u.pvd")
    vtkPEfile = File("solution_exact2D_brain/p1.pvd")
    vtkUEfile << u
    vtkPEfile << p[1]
    er2U = errornorm(u_e, u, "L2")
    print("Error L2 for displacements = ", er2U)
    er2P = errornorm(p1_e, p[1], "L2")
    print("Error L2 for pressure = ", er2P)
    
    
    pu_e = plot(u_e)

    # set colormap
    pu_e.set_cmap("viridis")
    #pu_e.set_clim( 0.0 , 1.0 )

    # add a title to the plot
    pyplot.title("u_e")

    # add a colorbar
    pyplot.colorbar( pu_e );
    pyplot.show()

    pu = plot(u)

    # set colormap
    pu.set_cmap("viridis")
    #pu.set_clim( 0.0 , 1.0 )

    # add a title to the plot
    pyplot.title("u")

    # add a colorbar
    pyplot.colorbar( pu );
    pyplot.show()

    pp_e = plot(p1_e)

    # set colormap
    pp_e.set_cmap("viridis")

    # add a title to the plot
    pyplot.title("p1_e")

    # add a colorbar
    pyplot.colorbar( pp_e );
    pyplot.show()

    pp = plot(p[1])

    # set colormap
    pp.set_cmap("viridis")
    #pp.set_clim( 0.0 , 1.0 )

    # add a title to the plot
    pyplot.title("p1")

    # add a colorbar
    pyplot.colorbar( pp );
    pyplot.show()


def exact_2D_brain_biot_test_transient():
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
    p0 = Lambda * (sym.diff(u, x, 1) + sym.diff(v, y, 1)) - alpha * p1
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
        + alpha / Lambda * (1 * sym.diff(p0, t, 1) + alpha * sym.diff(p1, t, 1))
        - K * (sym.diff(p1, x, 2) + sym.diff(p1, y, 2))
    )

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
        g1,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [Expression(var, degree=2,t=0) for var in variables]

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
        g1,
    ) = UFLvariables
    f = as_vector((fx, fy))
    U = as_vector((u, v))

    g = [g1]
    alpha = [1, alpha]
    c = [c]
    K = [K]
    p_initial = [p0,p1]
    T = 10
    numTsteps = 80

    mesh = generate_2D_brain_mesh_exact(12)

    #plot(mesh,mode="mesh")
    #show()

    n = FacetNormal(mesh)  # normal vector on the boundary
    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)

    bx0 = Boundary()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries

    # Define boundary conditions for the displacements
    boundary_conditionsU = {1: {"Dirichlet": U}}

    # Define boundary conditions for the pressure
    boundary_conditionsP = {
        (0, 1): {"Dirichlet": p0},  # Total pressure, boundarymarker 0
        (1, 1): {"Dirichlet": p1},  # Network 1, boundarymarker 0
    }

    u, p = biotMPET(
        mesh,
        T,
        numTsteps,
        1,
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
        p_initial = p_initial,
        boundaryNum=1,
        transient=True,
    )

    # Post processing
    V_e = VectorFunctionSpace(mesh, "P", 2)
    Q_e = FunctionSpace(mesh, "P", 1)

    u_e = project(Expression((variables[0], variables[1]), degree=2,t=T), V_e)
    p1_e = project(Expression(variables[3], degree=2,t=T), Q_e)

    vtkUEfile = File("solution_exact2D_brain/u_e.pvd")
    vtkPEfile = File("solution_exact2D_brain/p1_e.pvd")
    vtkUEfile << u_e
    vtkPEfile << p1_e

    vtkUEfile = File("solution_exact2D_brain/u.pvd")
    vtkPEfile = File("solution_exact2D_brain/p1.pvd")
    vtkUEfile << u
    vtkPEfile << p[1]
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

    pp1_e = plot(p1_e)

    # set colormap
    pp1_e.set_cmap("viridis")

    # add a title to the plot
    #pyplot.title("p1_e")

    # add a colorbar
    pyplot.colorbar( pp1_e );
    pyplot.show()

    pp1 = plot(p[1])

    # set colormap
    pp1.set_cmap("viridis")
    #pp.set_clim( 0.0 , 1.0 )

    # add a colorbar
    pyplot.colorbar( pp1 );
    pyplot.show()

    
def test_2Dbrain_Biot_Convergence():
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
    p0 = Lambda * (sym.diff(u, x, 1) + sym.diff(v, y, 1)) - alpha * p1

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
        + alpha / Lambda * (1 * sym.diff(p0, t, 1) + alpha * sym.diff(p1, t, 1))
        - K * (sym.diff(p1, x, 2) + sym.diff(p1, y, 2))
    )

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
        g1,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [Expression(var, degree=2,t=0) for var in variables]

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
        g1,
    ) = UFLvariables
    f = as_vector((fx, fy))
    U = as_vector((u, v))

    g = [g1]
    alpha = [1, alpha]
    c = [c]
    K = [K]
    p_initial = [p0,p1]
    T = 0.5
    numTsteps = 4

    # Define boundary conditions
    boundary_conditionsU = {1: {"Dirichlet": U}}
    # Define boundary conditions
    boundary_conditionsP = {
        (0, 1): {"Dirichlet": p0},  # Total pressure, boundarymarker 0
        (1, 1): {"Dirichlet": p1},  # Network 1, boundarymarker 0
    }
    
    mesh = generate_2D_brain_mesh_exact(6)

    erroruL2 = []
    errorpL2 = []
    erroruH1 = []
    errorpH1 = []
    h = []

    #For each mesh refinement 
    for n in range(4):
        
        plot(mesh)
        show()

        # Define boundary
        boundary_markers = MeshFunction("size_t", mesh, 1)
        boundary_markers.set_all(9999)

        bx0 = Boundary()
        bx0.mark(boundary_markers, 1)  # Applies for all boundaries
        
    
        u, p = biotMPET(
            mesh,
            T,
            numTsteps,
            1,
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

        mesh = refine(mesh)

    print(h) 
    print("Convergence rate for displacement using L2: ",convergenceRate(erroruL2, h))
    print("Convergence rate for pressure 1 using L2: ",convergenceRate(errorpL2, h))
    print("Convergence rate for displacement using H1: ",convergenceRate(erroruH1, h))
    print("Convergence rate for pressure 1 using H1: ",convergenceRate(errorpH1, h))


if __name__ == "__main__":
    exact_2D_brain_biot_test_steady()
    #exact_2D_brain_biot_test_transient()
    #test_2Dbrain_Biot_Convergence()
