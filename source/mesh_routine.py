#-------------------------------------------------------------------------------------
# These functions are used to:
# (1) move the mesh at each timestep according the solution and forcings (move_mesh), 
# (2) retrieve numpy arrays of the upper and lower surface elevations (get_surfaces)
# (3) perturb the mesh to create an initial condition (deform_mesh)
#-------------------------------------------------------------------------------------

import numpy as np
from bdry_conds import TopBoundary, WaterBoundary
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from petsc4py.PETSc import ScalarType
from ufl import (FacetNormal, SpatialCoordinate, TestFunction,
                 TrialFunction, ds, dx, grad, inner)

#-------------------------------------------------------------------------------------
def move_mesh(w,domain,dt):
    # this function computes the surface displacements and moves the mesh
    # by solving Laplace's equation for a smooth displacement function
    # defined for all mesh vertices

    V = FunctionSpace(domain, ("CG", 1))
 
    # solve for slope at surfaces (first component of normal vector)
    # and interpolate onto enitire domain
    n0 = FacetNormal(domain)[0]
    nu = n0
    u_ = TrialFunction(V)
    v_ = TestFunction(V)
    a = inner(u_,v_)*dx+inner(u_,v_)*ds
    l = inner(nu, v_)*ds
    prob0 = LinearProblem(a,l, bcs=[])
    n0 = prob0.solve()
    slope = n0/((1-n0**2)**0.5) # -d(surface)/dx and +d(base)/dx

    # displacement at upper and lower boundaries
    disp_h = dt*(w.sub(0).sub(1) + w.sub(0).sub(0)*slope) 
    disp_s = dt*(w.sub(0).sub(1) - w.sub(0).sub(0)*slope) 

    disp_h_fcn = Function(V)
    disp_s_fcn = Function(V)
    
    disp_h_fcn.interpolate(Expression(disp_h, V.element.interpolation_points()))
    disp_s_fcn.interpolate(Expression(disp_s, V.element.interpolation_points()))

    facets_1 = locate_entities_boundary(domain, domain.topology.dim-1, WaterBoundary)        
    facets_2 = locate_entities_boundary(domain, domain.topology.dim-1, TopBoundary)
    dofs_1 = locate_dofs_topological(V, domain.topology.dim-1, facets_1)
    dofs_2 = locate_dofs_topological(V, domain.topology.dim-1, facets_2)

    # # define displacement boundary conditions on upper and lower surfaces
    bc1 = dirichletbc(disp_s_fcn, dofs_1)
    bc2 = dirichletbc(disp_h_fcn, dofs_2)

    bcs = [bc1,bc2]

    # # solve Laplace's equation for a smooth displacement field on all vertices
    disp = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(disp), grad(v))*dx 
    f = Constant(domain, ScalarType(0.0))
    L = f*v*dx

    problem = LinearProblem(a,L, bcs=bcs)
    sol = problem.solve()

    disp_vv = sol.x.array

    X = domain.geometry.x

    X[:,1] += disp_vv

    return domain


def get_surfaces(domain):
# retrieve numpy arrays of the upper and lower surface elevations
    X = domain.geometry.x
    x = np.sort(X[:,0])
    z = X[:,1][np.argsort(X[:,0])]
    x_u = np.unique(X[:,0])
    h = np.zeros(x_u.size)      # upper surface elevation
    s = np.zeros(x_u.size)      # lower surface elevation

    for i in range(h.size):
        h[i] = np.max(z[np.where(np.isclose(x_u[i],x))])
        s[i] = np.min(z[np.where(np.isclose(x_u[i],x))])

    return h,s,x_u


def deform_mesh(domain,disp):
    # this function deforms the meesh to create a 
    # perturbed initial condition
    V = FunctionSpace(domain, ("CG", 1))
    x = SpatialCoordinate(domain)

    # displacement at upper and lower boundaries
    disp_h =  disp(x[0])
    disp_s = lambda x: 0*x[0] + 0

    disp_h_fcn = Function(V)
    disp_s_fcn = Function(V)
    
    disp_h_fcn.interpolate(Expression(disp_h, V.element.interpolation_points()))
    disp_s_fcn.interpolate(disp_s)

    facets_1 = locate_entities_boundary(domain, domain.topology.dim-1, WaterBoundary)        
    facets_2 = locate_entities_boundary(domain, domain.topology.dim-1, TopBoundary)
    dofs_1 = locate_dofs_topological(V, domain.topology.dim-1, facets_1)
    dofs_2 = locate_dofs_topological(V, domain.topology.dim-1, facets_2)

    # # define displacement boundary conditions on upper and lower surfaces
    bc1 = dirichletbc(disp_s_fcn, dofs_1)
    bc2 = dirichletbc(disp_h_fcn, dofs_2)

    bcs = [bc1,bc2]

    # # solve Laplace's equation for a smooth displacement field on all vertices,
    # # given the boundary displacement disp_bdry
    disp = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(disp), grad(v))*dx 
    f = Constant(domain, ScalarType(0.0))
    L = f*v*dx

    problem = LinearProblem(a,L, bcs=bcs)
    sol = problem.solve()

    disp_vv = sol.x.array

    X = domain.geometry.x

    X[:,1] += disp_vv

    return domain