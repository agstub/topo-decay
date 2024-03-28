#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing ice-shelf response to 
# sub-ice-shelf melting or freezing anomalies. The code relies on FEniCS--see README
#------------------------------------------------------------------------------------

import numpy as np
from mesh_routine import get_surfaces, move_mesh
from stokes import stokes_solve


def solve(domain,timesteps):

    # Define arrays for saving surfaces
    h_i,s_i,x = get_surfaces(domain)
    nx = x.size
    nt = timesteps.size
    h = np.zeros((nx,nt))
    s = np.zeros((nx,nt))

    # # Begin time stepping
    dt = timesteps[1] - timesteps[0]
    for i in range(nt):

        print('Iteration '+str(i+1)+' out of '+str(nt)+' \r',end='')

        if i>0:
            dt = timesteps[i] - timesteps[i-1]

        # Solve the Stoke problem for w = (u,p)
        sol = stokes_solve(domain,dt)

        # Move the mesh 
        domain = move_mesh(sol,domain,dt)
       
        # save surfaces
        h_i,s_i,x = get_surfaces(domain)
  
        h[:,i] = h_i
        s[:,i] = s_i


    return h,s,x