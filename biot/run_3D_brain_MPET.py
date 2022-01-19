from biot import biotMPET
from run_2D_brain_biot import ReadSourceTerm
from meshes.read_brain_mesh_3D import read_brain_mesh_3D, read_brain_scale
from matplotlib.pyplot import show
from ufl import FacetNormal, as_vector
from dolfin import *
import sympy as sym
import csv
#import biot 
#import imp
#imp.reload(biot)


def run_3D_brain_steady():

    #Mesh is in mm!
    mesh,subdomains,boundary_markers = read_brain_mesh_3D()
    n = FacetNormal(mesh)  # normal vector on the boundary
 
    
    E = 1500
    nu = 0.49
    my = E/(2*(1+nu))
    Lambda = nu*E/((1+nu)*(1-2*nu))

    #Pressure conversion: mmHg to Pa
    conversionP = 133.32
    
    p_initial1 = 4.5*conversionP
    alpha = 0.49
    c = 10e-6
    kappa = 1e-16*1e6 # [m² to mm²]
    mu_f = 0.8e-3
    K = kappa/mu_f
    t = 0.5
    p_csf = sym.symbols("p_csf")
    fx = 0  # force term x-direction
    fy = 0  # force term y-direction
    fz = 0  # force term z-direction
    p0 = 0.075*conversionP
    pSkull = p0
    pVentricles = p0*sym.sin(2*sym.pi*t)
    p_initial0 = -alpha*p_initial1
    
    variables = [
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        fz,
        pSkull,
        pVentricles,
        p_initial0,
        p_initial1,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [
        Expression(var, degree=1) for var in variables
    ]  # Generate ufl varibles

    (
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        fz,
        pSkull,
        pVentricles,
        p_initial0,
        p_initial1,
    ) = UFLvariables
    f = as_vector((fx, fy, fz))
    U = as_vector((Expression("0.0", degree=1), Expression("0.0", degree=1), Expression("0.0", degree=1)))
    alpha = [1, alpha]
    cj = [c]
    K = [K]
    p_init = [p_initial0, p_initial1]
    T = 0.0
    numTsteps = 0
    g = [None,None]

    
    boundary_conditionsU = {
        1: {"Dirichlet": U},
        2: {"Neumann": pVentricles},
        3: {"Neumann": pVentricles},
    }

   
    boundary_conditionsP = {
        (1, 1): {"Dirichlet": pSkull},
        (1, 2): {"Dirichlet": pVentricles},
        (1, 3): {"Dirichlet": pVentricles},
    }
    solver_method, precondition = 'bicgstap','hypre_amg'

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
        g=g,
        boundaryNum=3,
        p_initial = p_init,
        transient=False,
        filesave = "results/3D_brain_steady",
        )


def run_3D_brain_periodicBC():

    #Mesh is in mm!
    mesh,subdomains,boundary_markers = read_brain_mesh_3D()
    n = FacetNormal(mesh)  # normal vector on the boundary
 
    
    E = 1500
    nu = 0.49
    my = E/(2*(1+nu))
    Lambda = nu*E/((1+nu)*(1-2*nu))

    #Pressure conversion: mmHg to Pa
    conversionP = 133.32
    
    p_initial1 = 4.5*conversionP
    alpha = 0.49
    c = 10e-6
    kappa = 4e-15*1e6 #[m² to mm²]
    mu_f = 0.7e-3
    K = kappa/mu_f
    t = sym.symbols("t")
    fx = 0  # force term x-direction
    fy = 0  # force term y-direction
    fz = 0  # force term z-direction
    p0 = 0.075*conversionP
    p_VEN = p0*sym.sin(2*sym.pi*t)
    p_initial0 = -alpha*p_initial1
    
    variables = [
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        fz,
        p_VEN,
        p_initial0,
        p_initial1,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [
        Expression(var, degree=1,t=0) for var in variables
    ]  # Generate ufl varibles

    (
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        fz,
        p_VEN,
        p_initial0,
        p_initial1,
    ) = UFLvariables
    f = as_vector((fx, fy, fz))
    U = as_vector((Expression("0.0", degree=1), Expression("0.0", degree=1), Expression("0.0", degree=1)))
    alpha = [1, alpha]
    cj = [c]
    K = [K]
    p_init = [p_initial0, p_initial1]
    T = 3.0
    numTsteps = 60
    g = [None,None]
    beta_SAS = 1.0
    beta_VEN = 1.0

    #print("Reading and scaling source term...")
    #brain_mesh_3D = read_brain_mesh_3D()
    #source_scale = read_brain_scale(mesh) 
    #g = [ReadSourceTerm(mesh,source_scale)]
    
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
    solver_method, precondition = 'bicgstap','hypre_amg'

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
        g=g,
        boundaryNum=3,
        p_initial = p_init,
        transient=True,
        filesave = "results/3D_brain_periodicBC",
        )

def run_3D_brain_WindkesselBC():

    #Mesh is in mm!
    mesh,subdomains,boundary_markers = read_brain_mesh_3D()
    n = FacetNormal(mesh)  # normal vector on the boundary
 
    
    E = 1500
    nu = 0.49
    my = E/(2*(1+nu))
    Lambda = nu*E/((1+nu)*(1-2*nu))

    #Pressure conversion: mmHg to Pa
    conversionP = 133.32
    
    alpha = 0.49
    c = 10e-6
    kappa = 4e-15*1e6 #[m² to mm²]
    mu_f = 0.7e-3
    K = kappa/mu_f
    t = sym.symbols("t")
    p_VEN = sym.symbols("p_VEN")
    p_SAS = sym.symbols("p_SAS")
    fx = 0  # force term x-direction
    fy = 0  # force term y-direction
    fz = 0  # force term z-direction
    p0 = 0.075*conversionP
    p_initial1 = 0.0
    p_initial0 = -alpha*p_initial1
    p_Skull = p_SAS
    p_Ventricles = p0*sym.sin(2*sym.pi*t)

    Compliance = 2.5e3/conversionP # [microL/mmHg] to [mm³/Pa]
    Resistance = 10.81*conversionP*60e-3 # [mmHg*min/microL] to [Pa*s/mm³]
    
    variables = [
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        fz,
        p_Skull,
        p_Ventricles,
        p_initial0,
        p_initial1,
    ]

    variables = [sym.printing.ccode(var) for var in variables]  # Generate C++ code

    UFLvariables = [
        Expression(var, degree=1,t=0.0, p_VEN= 0.009,p_SAS = 0.018) for var in variables
    ]  # Generate ufl varibles

    (
        my,
        Lambda,
        alpha,
        c,
        K,
        fx,
        fy,
        fz,
        p_Skull,
        p_Ventricles,
        p_initial0,
        p_initial1,
    ) = UFLvariables
    f = as_vector((fx, fy, fz))
    U = as_vector((Expression("0.0", degree=1), Expression("0.0", degree=1), Expression("0.0", degree=1)))
    alpha = [1, alpha]
    cj = [c]
    K = [K]
    p_init = [p_initial0, p_initial1]
    T = 3.0
    numTsteps = 60
    beta_SAS = 1.0
    beta_VEN = 1.0

    #print("Reading and scaling source term...")
<<<<<<< HEAD
    #source_scale = read_brain_scale(mesh) 
    #g = [ReadSourceTerm(mesh,source_scale)]
=======
    source_scale = read_brain_scale(mesh) 
    g = [ReadSourceTerm(mesh,source_scale)]
>>>>>>> ebda251770d6636daf8accdc2348268cfabadb31
    
    boundary_conditionsU = {
        1: {"Dirichlet": U},
        2: {"NeumannWK": p_Ventricles},
        3: {"NeumannWK": p_Ventricles},
    }

   
    boundary_conditionsP = { #Applying windkessel bc
        (1, 1): {"RobinWK": (beta_SAS,p_Skull)},
        (1, 2): {"RobinWK": (beta_VEN,p_Ventricles)},
        (1, 3): {"RobinWK": (beta_VEN,p_Ventricles)},
    }
    solver_method, precondition = 'bicgstap','hypre_amg'

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
<<<<<<< HEAD
        g = [None,None],
=======
        g = g,
>>>>>>> ebda251770d6636daf8accdc2348268cfabadb31
        WindkesselBC = True,
        Resistance = Resistance,
        Compliance = Compliance,
        filesave = "results/3D_brain_windkesselBC"
    )

if __name__ == "__main__":
#    run_3D_brain_periodicBC()
    run_3D_brain_WindkesselBC()
