from biot import biotMPET_improved
from meshes.brain_mesh_2D import generate_2D_brain_mesh
from fenics import *
from matplotlib.pyplot import show
import sympy as sym

if __name__ == "__main__":
    tol = 1E-14
    
    class BoundaryOuter(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near((x[0]**2+x[1]**2)**(1/2), 1, tol)
    class BoundaryInner(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near((x[0]**2+x[1]**2)**(1/2),0.2 , tol)
    class BoundaryChannel(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[0],-0.05 , tol) or near(x[0],0.05 , tol))
        
    mesh = generate_2D_brain_mesh()
    
    #Define boundary
    boundary_markers = MeshFunction('size_t', mesh,1)
    boundary_markers.set_all(9999)
    
    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 0)
    bx1 = BoundaryInner()
    bx1.mark(boundary_markers, 1)
    bx2 = BoundaryChannel()
    bx2.mark(boundary_markers,2)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    
    for x in mesh.coordinates():
        if bx0.inside(x, True): print('%s is on x = 0' % x)
        if bx1.inside(x, True): print('%s is on x = 1' % x)
        if bx2.inside(x, True): print('%s is on x = 2' % x)
    
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
    p0 = Lambda * (sym.diff(u, x, 1) + sym.diff(v, y, 1)) - alpha * (p1)
    epsilonxx = sym.diff(u, x, 1)
    epsilonyy = sym.diff(v, y, 1)
    epsilonxy = 1 / 2 * (sym.diff(u, y, 1) + sym.diff(v, x, 1))
    fx = 0 #force term x-direction
    fy = 0 #force term y-direction
    g1 = c * sym.diff(p1, t, 1) #source term 

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

    UFLvariables = [Expression(var, degree=2, t=0) for var in variables]

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


    g = [g1]
    alpha = [1, alpha]
    cj = [c]
    K = [K]
    T = 0.5
    numTsteps = 4
    u, p = biotMPET_improved(
        mesh, T, numTsteps, 1, f, g, alpha, K, cj, my, Lambda, True
    )

    # Post processing
    u_e = Expression((variables[0], variables[1]), degree=2, t=T)
    p1_e = Expression(variables[3], degree=2, t=T)
    V_e = VectorFunctionSpace(mesh, "P", 2)
    Q_e = FunctionSpace(mesh, "P", 1)
    u_e = project(u_e, V_e)
    p1_e = project(p1_e, Q_e)
    vtkUEfile = File("solution_brain_2D/u_e.pvd")
    vtkPEfile = File("solution_brain_2D/p1_e.pvd")
    vtkUEfile << u_e
    vtkPEfile << p1_e
    er2U = errornorm(u_e, u, "L2")
    print("Error L2 for velocity = ", er2U)
    er2P = errornorm(p1_e, p[1], "L2")
    print("Error L2 for pressure = ", er2P)
    
    plot(u)
    
    show()
    

