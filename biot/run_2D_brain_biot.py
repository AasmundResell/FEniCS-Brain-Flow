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
    
    my = 1 / 3
    Lambda = 16666
    alpha = 1.0
    c = 1.0
    K = 1.0
    t = sym.symbols("t")
    fx = 0 #force term x-direction
    fy = 0 #force term y-direction
    g1 = c * sym.sin(t) #source term 

    variables = [
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
    vtkUEfile = File("solution_brain_2D/u.pvd")
    vtkPEfile = File("solution_brain_2D/p_1.pvd")
    vtkUEfile << u
    vtkPEfile << p[1]
    
    plot(u)
    
    show()
    

