import biot
from biot import biotMPET
from meshes.brain_mesh_2D import generate_2D_brain_mesh_mm, generate_2D_brain_mesh_m
from meshes.read_brain_mesh_3D import read_brain_mesh_3D,read_brain_scale
from matplotlib.pyplot import show
from ufl import FacetNormal, as_vector
from dolfin import *
import sympy as sym
import csv
import imp
#imp.reload(biot)



    
def run_physical_2D_brain_WindkesselBC(n=20):
    mesh = generate_2D_brain_mesh_mm(n)

    n = FacetNormal(mesh)  # normal vector on the boundary
    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)

    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries
    bx1 = BoundaryInner()
    bx1.mark(boundary_markers, 2)  # Overwrites the inner ventricles boundary
    bx2 = BoundaryChannel()
    bx2.mark(boundary_markers, 3)  # Overwrites the channel ventricles boundary
    #Defining material parameters and expressions
    E = 1500
    nu = 0.49
    my = E/(2*(1+nu))
    Lambda = nu*E/((1+nu)*(1-2*nu))
    conversionP = 133.32#Pressure conversion: mmHg to Pa
    
    alpha = 0.49
    c = 10e-6
    kappa = 4e-15*1e6 #[m²  to mm²]
    mu_f = 0.7e-3
    K = kappa/mu_f
    t = sym.symbols("t")
    p_VEN = sym.symbols("p_VEN")
    p_SAS = sym.symbols("p_SAS")
    fx = 0  # force term x-direction
    fy = 0  # force term y-direction
    p_initial1 = 0.0
    p_initial0 = -alpha*p_initial1
    pSkull = p_SAS
    pVentricles = p_VEN

    Compliance = 2.5e3/conversionP # [microL/mmHg] to [mm^3/Pa]
    Resistance = 10.81*conversionP*60e-3 # [mmHg*min/microL] to [Pa*s/mm^3]
    beta_VEN = 0.1
    beta_SAS = 0.1
    
    print("C=",Compliance)
    print("R=",Resistance)
    
    variables = [
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        pSkull,
        pVentricles,
        p_initial0,
        p_initial1,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [
        Expression(var, degree=2, t=0.0, p_VEN= 0.009,p_SAS = 0.018) for var in variables
    ]  # Generate ufl varibles

    (
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        pSkull,
        pVentricles,
        p_initial0,
        p_initial1,
        
    ) = UFLvariables
    f = as_vector((fx, fy))
    U = as_vector((Expression("0.0", degree=2), Expression("0.0", degree=2)))
    alpha = [1, alpha]
    cj = [c]
    K = [K]
    p_init = [p_initial0, p_initial1]
    T = 3
    numTsteps = 300

    source_scale = 1/1173670.5408281302 
    g = [ReadSourceTerm(mesh,source_scale)]
    
    boundary_conditionsU = {
        1: {"Dirichlet": U},
        2: {"NeumannWK": pVentricles},
        3: {"NeumannWK": pVentricles},
    }

   
    boundary_conditionsP = { #Applying windkessel bc
        (1, 1): {"RobinWK": (beta_SAS,pSkull)},
        (1, 2): {"RobinWK": (beta_VEN,pVentricles)},
        (1, 3): {"RobinWK": (beta_VEN,pVentricles)},
    }

#    boundary_conditionsP = { #Applying windkessel bc
#        (1, 1): {"DirichletWK": pSkull},
#        (1, 2): {"DirichletWK": pVentricles},
#        (1, 3): {"DirichletWK": pVentricles},
#    }


    
     
    u, p = biotMPET(
        mesh,
        T,
        numTsteps,
        1,
        f,
        alpha,
        K,
        cj,
        my,
        Lambda,
        boundary_conditionsU,
        boundary_conditionsP,
        boundary_markers,
        boundaryNum=3,
        p_initial = p_init,
        transient=True,
        g = g,
        WindkesselBC = True,
        Resistance = Resistance,
        Compliance = Compliance,
        filesave = "results/2D_brain_windkesselBC"
    )


def run_physical_2D_brain_periodicBC(n = 20):
    mesh = generate_2D_brain_mesh_mm(n)


    n = FacetNormal(mesh)  # normal vector on the boundary
    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)

    bx0 = BoundaryOuter()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries
    bx1 = BoundaryInner()
    bx1.mark(boundary_markers, 2)  # Overwrites the inner ventricles boundary
    bx2 = BoundaryChannel()
    bx2.mark(boundary_markers, 3)  # Overwrites the channel ventricles boundary

    
    for x in mesh.coordinates():
        if bx0.inside(x, True):
            print("%s is on x = 0" % x)
        if bx1.inside(x, True):
            print("%s is on x = 1" % x)
        if bx2.inside(x, True):
            print("%s is on x = 2" % x)
    
    E = 1500
    nu = 0.49
    my = E/(2*(1+nu))
    Lambda = nu*E/((1+nu)*(1-2*nu))
    conversionP = 133.32#Pressure conversion: mmHg to Pa
    
    p_initial1 = 0.0
    alpha = 0.49
    c = 10e-6
    kappa = 4e-15*1e6 #[m^2 to mm^2]
    mu_f = 0.7e-3
    tune = 100 #For suppressing instabilities
    K = kappa/mu_f*tune
    t = sym.symbols("t")
    fx = 0  # force term x-direction
    fy = 0  # force term y-direction
    p0 = 0.075*conversionP
    p_VEN = p0*sym.sin(2*sym.pi*t)
    p_initial0 = -alpha*p_initial1
    beta_SAS = 0.2
    beta_VEN = 0.2

    
    
    variables = [
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        p_VEN,
        p_initial0,
        p_initial1,
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
        p_VEN,
        p_initial0,
        p_initial1,
    ) = UFLvariables
    f = as_vector((fx, fy))
    U = as_vector((Expression("0.0", degree=2), Expression("0.0", degree=2)))
    alpha = [1, alpha]
    cj = [c]
    K = [K]
    p_init = [p_initial0, p_initial1]
    T = 3
    numTsteps = 300

    
    boundary_conditionsU = {
        1: {"Dirichlet": U},
        2: {"Neumann": p_VEN},
        3: {"Neumann": p_VEN},
    }

   
    boundary_conditionsP = { #Applying windkessel bc
        (1, 1): {"Robin": (beta_SAS,None)},
        (1, 2): {"Robin": (beta_VEN,p_VEN)},
        (1, 3): {"Robin": (beta_VEN,p_VEN)},
    }
    
    u, p = biotMPET(
        mesh,
        T,
        numTsteps,
        1,
        f,
        alpha,
        K,
        cj,
        my,
        Lambda,
        boundary_conditionsU,
        boundary_conditionsP,
        boundary_markers,
        boundaryNum=3,
        g = [None, None],
        p_initial = p_init,
        transient=True,
        filesave = "results/2D_brain_periodicBC",
    )
    
def run_physical_2D_brain_periodicBC_meter(n = 20):
    mesh = generate_2D_brain_mesh_m(n)

    n = FacetNormal(mesh)  # normal vector on the boundary
    class BoundaryOuter_m(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
        
    class BoundaryInner_m(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-10
            return on_boundary and  (x[0] ** 2 + x[1] ** 2) ** (1 / 2) <= 0.032
            
    class BoundaryChannel_m(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-10
            return on_boundary and (near(x[0], -0.001, tol) or near(x[0], 0.001, tol))

    # Define boundary
    boundary_markers = MeshFunction("size_t", mesh, 1)
    boundary_markers.set_all(9999)
    bx0 = BoundaryOuter_m()
    bx0.mark(boundary_markers, 1)  # Applies for all boundaries
    bx1 = BoundaryInner_m()
    bx1.mark(boundary_markers, 2)  # Overwrites the inner ventricles boundary
    bx2 = BoundaryChannel_m()
    bx2.mark(boundary_markers, 3)  # Overwrites the channel ventricles boundary

    
    for x in mesh.coordinates():
        if bx0.inside(x, True):
            print("%s is on x = 0" % x)
        if bx1.inside(x, True):
            print("%s is on x = 1" % x)
        if bx2.inside(x, True):
            print("%s is on x = 2" % x)
    
    E = 1500
    nu = 0.49
    my = E/(2*(1+nu))
    Lambda = nu*E/((1+nu)*(1-2*nu))
    conversionP = 133.32#Pressure conversion: mmHg to Pa
    
    p_initial1 = 0.0
    alpha = 0.49
    c = 10e-6
    kappa = 4e-15
    mu_f = 0.7e-3
    tune = 1000 #For suppressing instabilities
    K = kappa/mu_f*tune
    t = sym.symbols("t")
    fx = 0  # force term x-direction
    fy = 0  # force term y-direction
    p0 = 0.075*conversionP
    p_VEN = p0*sym.sin(2*sym.pi*t)
    p_initial0 = -alpha*p_initial1
    beta_SAS = 1.0
    beta_VEN = 1.0
    
    
    variables = [
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        p_VEN,
        p_initial0,
        p_initial1,
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
        p_VEN,
        p_initial0,
        p_initial1,
    ) = UFLvariables
    f = as_vector((fx, fy))
    U = as_vector((Expression("0.0", degree=2), Expression("0.0", degree=2)))
    alpha = [1, alpha]
    cj = [c]
    K = [K]
    p_init = [p_initial0, p_initial1]
    T = 3
    numTsteps = 300

    
    boundary_conditionsU = {
        1: {"Dirichlet": U},
        2: {"Neumann": p_VEN},
        3: {"Neumann": p_VEN},
    }

   
    boundary_conditionsP = { #Applying windkessel bc
        (1, 1): {"Robin": (beta_SAS,None)},
        (1, 2): {"Robin": (beta_VEN,p_VEN)},
        (1, 3): {"Robin": (beta_VEN,p_VEN)},
    }
    
    u, p = biotMPET(
        mesh,
        T,
        numTsteps,
        1,
        f,
        alpha,
        K,
        cj,
        my,
        Lambda,
        boundary_conditionsU,
        boundary_conditionsP,
        boundary_markers,
        boundaryNum=3,
        g = [None, None],
        p_initial = p_init,
        transient=True,
        filesave = "results/2D_brain_periodicBC_meter",
    )
    

def ReadSourceTerm(mesh,source_scale=1.0,periods = 3):
    filestr = 'data/baladont_tot_inflow_series_shifted.csv'
    g = TimeSeries("source_term")
    Q = FunctionSpace(mesh,"CG",1)
    time_period = 0.0
    for i in range(periods):
        #Read in the source term data
        with open(filestr) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                   print(f'Column names are {", ".join(row)}')
                   line_count += 1
                else:
                   print(float(row[1]))
                   source = Constant(float(row[1])*source_scale) #Uniform source on the domain
                   source = project(source, Q)
                   g.store(source.vector(),float(row[0]) + i*time_period)
                   print(f"\t timestep {row[0]} adding {row[1]} to the source.")
                   line_count += 1
            if i == 0:
                time_period = float(row[0])
                print("t =", time_period)
            print(f'Processed {line_count} lines.')
    return g


class BoundaryOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class BoundaryInner(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and  (x[0] ** 2 + x[1] ** 2) ** (1 / 2) <= 32

class BoundaryChannel(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and (near(x[0], -1, tol) or near(x[0], 1, tol))
    
if __name__ == "__main__":
    n = 30
#    run_physical_2D_brain_WindkesselBC(n)
    run_physical_2D_brain_periodicBC(n)
    #run_physical_2D_brain_periodicBC_meter(n)
