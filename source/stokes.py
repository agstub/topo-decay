# This file contains the functions needed for solving the nonlinear Stokes problem.
from bdry_conds import mark_boundary
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         locate_dofs_topological)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.log import LogLevel, set_log_level
from dolfinx.mesh import locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from params import g, rho_i, rho_w, sea_level, eta0
from petsc4py import PETSc
from ufl import (FacetNormal, FiniteElement, Measure, MixedElement,
                 SpatialCoordinate, TestFunctions, div, dx, grad, inner, split,
                 sym)

def eta(z):
      # ice viscosity 
      return eta0

def weak_form(u,p,v,q,f,g_base,ds,nu,dt,x):
    z = x[1]
    # Weak form residual of the ice-shelf problem
    F = 2*eta(z)*inner(sym(grad(u)),sym(grad(v)))*dx
    F += (- div(v)*p + q*div(u))*dx - inner(f, v)*dx
    F += (g_base+rho_w*g*dt*inner(u,nu))*inner(v,nu)*ds(3)
    return F

def stokes_solve(domain,dt):
        # Stokes solver for the ice-shelf problem using Taylor-Hood elements

        # Define function spaces
        P1 = FiniteElement('P',domain.ufl_cell(),1)     # Pressure p
        P2 = FiniteElement('P',domain.ufl_cell(),2)     # Velocity u
        element = MixedElement([[P2,P2],P1])
        W = FunctionSpace(domain,element)  # Function space for (u,p)

        #---------------------Define variational problem------------------------
        w = Function(W)
        (u,p) = split(w)
        (v,q) = TestFunctions(W)
      
        # Neumann condition at ice-water boundary
        x = SpatialCoordinate(domain)
        g_0 = rho_w*g*(sea_level-x[1])
        g_base = 0.5*(g_0+abs(g_0))

        # Body force
        f = Constant(domain,PETSc.ScalarType((0,-rho_i*g)))      

        # Outward-pointing unit normal to the boundary  
        nu = FacetNormal(domain)           

        # Mark bounadries of mesh and define a measure for integration
        facet_tag,LeftBoundary,RightBoundary = mark_boundary(domain)
        ds = Measure('ds', domain=domain, subdomain_data=facet_tag)

        # Define boundary conditions on the inflow/outflow boundary
        facets_1 = locate_entities_boundary(domain, domain.topology.dim-1, LeftBoundary)        
        facets_2 = locate_entities_boundary(domain, domain.topology.dim-1, RightBoundary)
        dofs_1 = locate_dofs_topological(W.sub(0).sub(0), domain.topology.dim-1, facets_1)
        dofs_2 = locate_dofs_topological(W.sub(0).sub(0), domain.topology.dim-1, facets_2)
        bc1 = dirichletbc(PETSc.ScalarType(0), dofs_1,W.sub(0).sub(0))
        bc2 = dirichletbc(PETSc.ScalarType(0), dofs_2,W.sub(0).sub(0))
        bcs = [bc1,bc2]

        # Define weak form
        F = weak_form(u,p,v,q,f,g_base,ds,nu,dt,x)

        # Solve for (u,p)
        problem = NonlinearProblem(F, w, bcs=bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        set_log_level(LogLevel.WARNING)
        n, converged = solver.solve(w)
        assert(converged)

        return w