import sys
sys.path.append(r'C:\Users\ljw_l\anaconda3\Lib\site-packages')
from docplex.mp.model import Model
import numpy as np
from GA.data_processing import W, C, CAB_model_data_processing


# Data

# Data with different combinations
# set of nodes
N = list(range(15))
# set of potential locations for the hubs
P = list(range(10))
# set of time periods
T = [0, 1, 2]
# set of scenarios
S = list(range(5))
# capacity of a module
gamma = 0.075
# transfer cost in period t
a = 0.2
alpha = [a, a, a]

# Fixed data
# maximum number of modules that can be installed in a hub at location k
Q = {p: 5 for p in P}
# collection cost in period t
X = [1, 1, 1]
# distribution cost in period t
delta = [1, 1, 1]

# Calculate the rest of data
N_P = [len(N), len(P)]
gamma_alpha = [gamma, a]
distance, demand_dict, originated_flow_dict, destined_flow_dict, install_hub_cost_matrix, initial_f_k_q_t, initial_m_k_q_t, additional_f_k_q_t, additional_m_k_q_t, initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_model_data_processing(W, C, N_P, gamma_alpha)

# flow to be sent from node i to node j in period t in scenario s
w = demand_dict
# total flow originated at node i in period t and scenario s
O = originated_flow_dict
# total flow destined to node i in period t and scenario s
D = destined_flow_dict
# fixed cost for installing a hub in location k in period t
f = install_hub_cost_matrix
# fixed cost for installing q modules in a hub at location k in period t
ff = initial_f_k_q_t
# operating cost incurred when q modules are installed in a hub at location k in period t
mm = initial_m_k_q_t
# fixed cost for installing q additional modules in a hub at location k in period t
fff = additional_f_k_q_t
# operating cost incurred when q additional modules are installed in a hub at location k in period t
mmm = additional_m_k_q_t
# the total cost resulting from setting up the initial capacity of a hub with q modules to be located at k in period t
g = initial_capacity_cost_dict
# the cost for installing q additional modules in an existing hub at location k in period t
h = additional_capacity_cost_dict
# the distance between nodes i and j(distances are symmetric and satisfy the triangle inequality)
d = distance


# Create Model
M_DE = Model('Hub_location_problem')


# Decision variables

# if a hub is installed at k in period t: u_t_k = 1
idx_u = [(t, k) for t in T for k in P]
u = M_DE.binary_var_dict(idx_u, name = 'u')
# if hub k receives q modules in period t: z_t_k_q = 1
idx_z = [(t, k, q) for t in T for k in P for q in range(1, Q[k]+1)]
z = M_DE.binary_var_dict(idx_z, name = 'z')
# if hub k receives q additional modules in period t and scenario s: r_s_t_k_q = 1
idx_r = [(s, t, k, q) for s in S for t in [1,2] for k in P for q in range(1, Q[k])]
r = M_DE.binary_var_dict(idx_r, name = 'r')

# amount of flow with origin at i that is collected at hub k in period t and scenario s: x_s_t_i_k
idx_x = [(s, t, i, k) for s in S for t in T for i in N for k in P]
x = M_DE.continuous_var_dict(idx_x, lb=0, name = 'x')
# amount of flow with origin at i that is collected at hub k and is distributed by hub l in period t and scenario s: y_s_t_i_k_l
idx_y = [(s, t, i, k, l) for s in S for t in T for i in N for k in P for l in P]
y = M_DE.continuous_var_dict(idx_y, lb=0, name = 'y')
# amount of flow with origin at i destined to node j that is distributed by hub l in period t and scenario s: v_s_t_i_l_j
idx_v = [(s, t, i, l, j) for s in S for t in T for i in N for l in P for j in N]
v = M_DE.continuous_var_dict(idx_v, lb=0, name = 'v')


# Constraints
