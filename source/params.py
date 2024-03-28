# All model/numerical parameters are set here.

# Model parameters
rho_i = 917.0                                    # Density of ice [kg/m^3]
rho_w = 1020.0                                   # Density of seawater [kg/m^3]
g = 9.81                                         # Gravitational acceleration [m/s^2]
eta0 = 1e14                                      # viscosity [Pa s]
delta = rho_w/rho_i-1                            # flotation factor
eps = 2.529e-14                                  # regularization for the relaxation("R") 
                                                 # and buoyancy ("B") functions

H = 500.0                                        # Thickness (height of the domain) [m]
sea_level = H*(rho_i/rho_w)                      # Sea level elevation [m]
t_r = 2*eta0/(rho_i*g*H)                         # viscous relaxation time scale [s]
t_e = (4*eta0/((rho_w-rho_i)*g*H))*(rho_w/rho_i) # intrinsic time scale [s]

ratio = t_r/t_e