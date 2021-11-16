from biot import biotMPET
from physical_2D_brain_biot import ReadSourceTerm
from meshes.read_brain_mesh_3D import read_brain_mesh_3D, read_brain_scale
from matplotlib.pyplot import show
from ufl import FacetNormal, as_vector
from dolfin import *
import sympy as sym
import csv


def run_3D_brain_periodicBC_steady():
    mesh,subdomains,boundary_markers = read_brain_mesh_3D()
    n = FacetNormal(mesh)  # normal vector on the boundary
    
    
    E = 1500
    nu = 0.479
    my = E/(2*(1+nu))
    Lambda = nu*E/((1+nu)*(1-2*nu))

    #Pressure conversion: mmHg to Pa
    conversionP = 133.32
    
    p_initial1 = 4.5*conversionP
    alpha = 0.49
    c = 10e-6
    kappa = 1e-16
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
        Expression(var, degree=2) for var in variables
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

    print("Reading and scaling source term...")
    brain_mesh_3D = read_brain_mesh_3D()
    source_scale = read_brain_scale(mesh) 
    g = [ReadSourceTerm(mesh,source_scale)]
    
    boundary_conditionsU = {
        0: {"Neumann": U},
        1: {"Dirichlet": U},
        2: {"Neumann": pVentricles},
        3: {"Neumann": pVentricles},
    }

   
    boundary_conditionsP = { #Applying windkessel bc
        (1, 0): {"Neumann": Expression("0.0", degree=1)},
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
        g,
        alpha,
        K,
        cj,
        my,
        Lambda,
        boundary_conditionsU,
        boundary_conditionsP,
        boundary_markers,
        boundaryNum=4,
        p_initial = p_init,
        transient=False,
        )

if __name__ == "__main__":
    run_3D_brain_periodicBC_steady()
