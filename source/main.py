#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing ice-shelf response to 
# sub-ice-shelf melting or freezing anomalies. The code relies on FEniCS--see README
#------------------------------------------------------------------------------------

import numpy as np
from dolfinx.mesh import create_rectangle
from mesh_routine import get_surfaces, move_mesh
from mpi4py import MPI
from params import H, L, Nx, Nz, nt, t_f
from stokes import stokes_solve


def solve(a,m):

    # generate mesh
    p0 = [-L/2.0,0.0]
    p1 = [L/2.0,H]
    domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [Nx, Nz])

    # Define arrays for saving surfaces
    h_i,s_i,x = get_surfaces(domain)
    nx = x.size
    h = np.zeros((nx,nt))
    s = np.zeros((nx,nt))

    t = np.linspace(0,t_f, nt)
    # # Begin time stepping
    for i in range(nt):

        print('Iteration '+str(i+1)+' out of '+str(nt)+' \r',end='')

        t_i = t[i]

        # Solve the Stoke problem for w = (u,p)
        sol = stokes_solve(domain)

        # Move the mesh 
        domain = move_mesh(sol,domain,t_i,a,m)
       
        # save surfaces
        h_i,s_i,x = get_surfaces(domain)
  
        h[:,i] = h_i
        s[:,i] = s_i


    return h,s,x