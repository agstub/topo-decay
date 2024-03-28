## topo-decay
Nonlinear Stokes ice-flow solver in FEniCS configured for exploring topographic decay on floating ice shelves/shells. See agstub/linear-shelf-melt for more description for now... 

This code requires FEniCS (https://fenicsproject.org) and can be run through a Docker (https://www.docker.com) container via

`docker run --init -ti -p 8888:8888 -v $(pwd):/home/fenics/shared -w /home/fenics/shared dolfinx/lab:stable`

