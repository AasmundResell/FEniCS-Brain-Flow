from biot import biotMPET
from meshes.brain_mesh_2D import generate_2D_brain_mesh
from matplotlib.pyplot import show
from ufl import FacetNormal,as_vector
from dolfin import SubDomain,MeshFunction,near,Expression,Constant,File,plot
import sympy as sym

if __name__ == "__main__":
    tol = 1e-14

    class BoundaryOuter(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class BoundaryInner(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[0] ** 2 + x[1] ** 2) ** (1 / 2) <= 0.2

    class BoundaryChannel(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[0], -0.05, tol) or near(x[0], 0.05, tol))

    mesh = generate_2D_brain_mesh()

    n = FacetNormal(mesh)  # normal vector on the boundary
    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)

    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 0)  # Applies for all boundaries
    bx1 = BoundaryInner()
    bx1.mark(boundary_markers, 1)  # Overwrites the inner ventricles boundary
    bx2 = BoundaryChannel()
    bx2.mark(boundary_markers, 2)  # Overwrites the channel ventricles boundary

    
    for x in mesh.coordinates():
        if bx0.inside(x, True):
            print("%s is on x = 0" % x)
        if bx1.inside(x, True):
            print("%s is on x = 1" % x)
        if bx2.inside(x, True):
            print("%s is on x = 2" % x)

    my = 1 / 3
    Lambda = 16666
    alpha = 0.49
    c = 3.9e-4
    K = 1.57e-5
    t = sym.symbols("t")
    fx = 0  # force term x-direction
    fy = 0  # force term y-direction
    g1 = c * sym.sin(4 * sym.pi * t)  # source term
    pSkull = 5 + 2 * sym.sin(2 * sym.pi * t)
    pVentricles = 5 + (2 + 0.012) * sym.sin( 2 * sym.pi * t)  # 0.012 is the transmantle pressure difference

    neumannBC = -alpha * 10000 * pVentricles

    variables = [
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        g1,
        pSkull,
        pVentricles,
        neumannBC,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [
        Expression(var, degree=2, t=0) for var in variables
    ]  # Generate ufl varibles

    (
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        g1,
        pSkull,
        pVentricles,
        neumannBC,
    ) = UFLvariables
    f = as_vector((fx, fy))

    g = [g1]
    alpha = [1, alpha]
    cj = [c]
    K = [K]
    T = 10
    numTsteps = 80

    # Define boundary conditions
    boundary_conditionsU = {
        0: {"Dirichlet": Constant(0.0)},
        1: {"Neumann": n * neumannBC},
        2: {"Neumann": n * neumannBC},
    }
    
    # Define boundary conditions
    boundary_conditionsP = {
        0: {"Dirichlet": pSkull},
        1: {"Dirichlet": pVentricles},
        2: {"Dirichlet": pVentricles},
    }
    u, p = biotMPET(
        mesh,
        T,
        numTsteps,
        1,
        f,
        g,
        alpha,
        K,
        cj,
        my,
        Lambda,
        boundary_conditionsU,
        boundary_markers,
        True,
       # boundary_conditionsP,
        #boundary_markers
    )

    # Post processing
    vtkUfile = File("solution_brain_2D/u.pvd")
    vtkPfile = File("solution_brain_2D/p_1.pvd")
    vtkUfile << u
    vtkPfile << p[1]

    plot(u)

    show()
