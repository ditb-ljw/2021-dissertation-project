import numpy as np


# read all the contents in the text file
file = open(r'C:\Users\ljw_l\github\2021-dissertation-project\GA\CAB.txt','r')
contents = file.readlines()
file.close()

# extract the data and save in two matrices
data = contents[28:652]
W = np.zeros((25,25))
C = np.zeros((25,25))
for i in data:
    row_content = list(map(float, i.split()))
    row_i = int(row_content[0]) - 1
    col_j = int(row_content[1]) - 1
    W[row_i, col_j] = row_content[2]
    C[row_i, col_j] = row_content[3]

#
#print(W[:10,:10])
#print(C[:10,:10])


def CAB_data_processing(W, C, N_P_case, gamma_alpha_case):

    N_P_test_cases = {0: [15, 10], 1: [15, 15], 2: [20, 10], 3: [20, 15], 4: [20, 20], 5: [25, 10], 6: [25, 15], 7: [25, 20], 8: [25, 25]}
    gamma_alpha_test_cases = {0: [0.075, 0.2], 1: [0.075, 0.4], 2: [0.075, 0.6], 3: [0.075, 0.8], 4: [0.1, 0.2], 5: [0.1, 0.4], 6: [0.1, 0.6], 7: [0.1, 0.8], \
    8: [0.125, 0.2], 9: [0.125, 0.4], 10: [0.125, 0.6], 11: [0.125, 0.8]}
    X = 1
    delta = 1
    T = 3
    S = 5
    Q = 5
    beta = 0.2

    N_P = N_P_test_cases[N_P_case]
    N = N_P[0]
    P = N_P[1]
    gamma_alpha = gamma_alpha_test_cases[gamma_alpha_case]
    gamma = gamma_alpha[0]
    alpha = gamma_alpha[1]

    # Total flow originated at node k
    O_k = np.sum(W[:N, :N], axis = 1)
    f_k_1 = T*15*np.log(O_k)

    f_k_t = np.zeros([T, N])
    f_k_t[0,:] = f_k_1
    for i in range(T-1):
        f_k_t[i+1,:] = 1.02*f_k_t[i,:]

    initial_f_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        initial_f_k_q_1[:,q] = beta*f_k_1*q
    initial_f_k_q_t = {0: initial_f_k_q_1}
    for t in range(T-1):
        initial_f_k_q_t[t+1] = 1.02*initial_f_k_q_t[t]

    initial_m_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        initial_m_k_q_1[:,q] = 0.6*beta*f_k_1*q
    initial_m_k_q_t = {0: initial_m_k_q_1}
    for t in range(T-1):
        initial_m_k_q_t[t+1] = 1.02*initial_m_k_q_t[t]

    additional_f_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        additional_f_k_q_1[:,q] = 2*beta*f_k_1*q
    additional_f_k_q_t = {0: additional_f_k_q_1}
    for t in range(T-1):
        additional_f_k_q_t[t+1] = 1.02*additional_f_k_q_t[t]

    additional_m_k_q_t = initial_m_k_q_t.copy()

    g_k_q_t = {}
    sum_initial_operating = sum(initial_m_k_q_t.values())
    for t in range(T):
        g_k_q_t[t] = initial_f_k_q_t[t] + sum_initial_operating
        sum_initial_operating -= initial_m_k_q_t[t]

    h_k_q_t = {}
    sum_additional_operating = sum(additional_m_k_q_t.values())
    for t in range(T):
        h_k_q_t[t] = additional_f_k_q_t[t] + sum_additional_operating
        sum_additional_operating -= additional_m_k_q_t[t]

    # demands
    scaled_W = W[:N, :N]/np.sum(W[:N, :N])
    demand_dict = {s: [] for s in range(S)}
    for s in range(S):
        demand_dict[s].append(scaled_W)
        for t in range(1,T):
            W_ij_s_t = demand_dict[s][t-1]*(1 + 6/100*(s+1))
            demand_dict[s].append(W_ij_s_t)

    return P, [Q]*N, [X, alpha, delta], demand_dict, list(range(S)), gamma, f_k_t, g_k_q_t, h_k_q_t




def test_data():

    distance = C

    max_capacity = [5]*25
    coefficients = [1,1,1]
    demand_dict = {0: [W[:5,:5], W[:5,:5], W[:5,:5]]}
    scenarios = [0, 1, 2, 3, 4]

    probabilities = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}

    module_capacity = 100000
    install_hub_cost_matrix = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    capacity_cost = np.array([np.arange(31), np.arange(31), np.arange(31), np.arange(31), np.arange(31)])
    initial_capacity_cost_dict = {0: capacity_cost, 1: capacity_cost, 2: capacity_cost}
    additional_capacity_cost_dict = {0: capacity_cost, 1: capacity_cost, 2: capacity_cost}
    return distance, max_capacity, coefficients, demand_dict, scenarios, probabilities, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict
