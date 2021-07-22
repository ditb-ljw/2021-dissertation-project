import sys
sys.path.append(r'C:\Users\ljw_l\anaconda3\Lib\site-packages')
from docplex.mp.model import Model
import numpy as np
from GA.data_processing import W, C, CAB_data_processing


# Data

# set of nodes
N = list(range(15))
# set of potential locations for the hubs
P = list(range(10))
# set of time periods
T = [0, 1, 2]

# capacity of a module
gamma = 0.075
# maximum number of modules that can be installed in a hub at location k
Q = [5]*10

# flow to be sent from node i to node j in period t in scenario s
w = 

N_P = [len(N), len(P)]
gamma_alpha = [gamma, 0.2]

distance, coefficients, demand_dict, highest_originate, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_data_processing(W, C, N_P, gamma_alpha)
test_data = {'distance': distance, 'hub_locations': hub_locations, 'coefficients': coefficients, 'demand_dict': demand_dict, \
'highest_originate': highest_originate, 'module_capacity': module_capacity, 'install_hub_cost_matrix': install_hub_cost_matrix, \
'initial_capacity_cost_dict': initial_capacity_cost_dict, 'additional_capacity_cost_dict': additional_capacity_cost_dict}


# Create Model
M_DE = Model('Hub_location_problem')

# Decision variables
u = M_DE.binary_var_matrix(install_hub_cost_matrix.shape[0], install_hub_cost_matrix.shape[1], name = 'hub_location')
z =
r =

R = range(1, 10)
idx = [(i, j, k) for i in R for j in R for k in R]

x = model.binary_var_dict(idx, name = "X")
