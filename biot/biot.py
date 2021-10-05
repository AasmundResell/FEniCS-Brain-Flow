from fenics import *
from mshr import *
import ufl


def biotMPET_improved(
    mesh,
    T,
    numTsteps,
    numPnetworks,
    f,
    g,
    alpha,
    K,
    cj,
    my,
    Lambda,
    boundary_conditionsU,
    boundary_markersU,
    #    boundary_conditionsP,
    #    boundary_markersP,
    typeS=False,
):
    dsU = Measure("ds", domain=mesh, subdomain_data=boundary_markersU)
    # dsP = Measure("ds", domain=mesh, subdomain_data=boundary_markersP)

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

    # variational formulation
    sources = []  # Contains the source term for each network
    innerProdP = (
        []
    )  # Contains the inner product of the gradient of p_j for each network
    dotProdP = []  # Contains the dot product of alpha_j & p_j,
    timeD_ = []  # Time derivative for the current step
    timeD_n = []  # Time derivative for the previous step
    bcs_D = []  # Contains the terms for the Dirichlet boundaries
    integrals_N = []  # Contains the integrals for the Neumann boundaries
    time_expr = []  # Terms that needs to be updated at each timestep

    def a_u(u, v):
        return my * (inner(grad(u), grad(v)) + inner(grad(u), nabla_grad(v))) * dx

    def a_p(K, p, q):
        return K * dot(grad(p), grad(q)) * dx

    def b(s, v):
        return s * div(v) * dx

    def c(alpha, p, q):
        return alpha / Lambda * dot(p, q) * dx

    def F(f, v):
        return dot(f, v) * dx

    for i in range(numPnetworks):  # apply for each network
        # Applying boundary terms
        bcp = DirichletBC(W.sub(i + 2), Constant(0.0), "on_boundary")
        bcs_D.append(bcp)

        dotProdP.append(c(alpha[i + 1], p_[i + 2], q[1]))  # Applying time derivative
        sources.append(F(g[i], q[i + 2]))  # Applying source term
        innerProdP.append(a_p(K[i], p_[i + 2], q[i + 2]))  # Applying diffusive term
        time_expr.append(g[i])

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

    # Defining boundary conditions for displacements
    for i in boundary_conditionsU:
        if "Dirichlet" in boundary_conditionsU[i]:
            print("Applying Dirichlet BC.")
            bcs_D.append(
                DirichletBC(
                    W.sub(0).sub(0),
                    boundary_conditionsU[i]["Dirichlet"],
                    boundary_markersU,
                    i,
                )
            )
            bcs_D.append(
                DirichletBC(
                    W.sub(0).sub(1),
                    boundary_conditionsU[i]["Dirichlet"],
                    boundary_markersU,
                    i,
                )
            )
        elif "Neumann" in boundary_conditionsU[i]:
            if boundary_conditionsU[i]["Neumann"] != 0:
                print("Applying Neumann BC.")
                N = boundary_conditionsU[i]["Neumann"]
                integrals_N.append(inner(N, q[0]) * ds(i))
                time_expr.append(N)
    bcp0 = DirichletBC(W.sub(1), Constant(0.0), "on_boundary")

    bcs_D.append(bcp0)

    lhs = (
        a_u(p_[0], q[0])
        + b(p_[1], q[0])
        + b(q[1], p_[0])
        - sum(dotProdP)
        + sum(innerProdP)
        + sum(timeD_)
    )

    rhs = F(f, q[0]) + sum(sources) + sum(timeD_n) + sum(integrals_N)
    time_expr.append(f[0])
    time_expr.append(f[1])
    A = assemble(lhs)
    [bc.apply(A) for bc in bcs_D]

    up = Function(W)

    t = 0

    for i in range(numTsteps):
        t += dt
        for expr in time_expr:  # Update all time dependent terms
            if isinstance(expr, ufl.tensors.ComponentTensor):
                for dimexpr in expr.ufl_operands:
                    for op in dimexpr.ufl_operands:
                        try:
                            op.t = t
                        except:
                            pass
            else:
                update_operator(expr, t)

        b = assemble(rhs)
        [bc.apply(b) for bc in bcs_D]
        solve(A, up.vector(), b)

        if typeS:
            vtkU << (up.sub(0), t)
            vtkP << (up.sub(1), t)
        up_n.assign(up)
        progress += 1
    res = []
    res = split(up)
    u = project(res[0], W.sub(0).collapse())
    p = []

    for i in range(1, numPnetworks + 2):
        p.append(project(res[i], W.sub(1).collapse()))

    return u, p


def update_operator(expr, time):
    if isinstance(expr, ufl.algebra.Operator):
        for op in expr.ufl_operands:
            update_operator(op, time)
    elif isinstance(expr, ufl.Coefficient):
        expr.t = time
