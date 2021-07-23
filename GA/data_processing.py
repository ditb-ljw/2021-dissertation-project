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


def CAB_data_processing(W, C, N_P, gamma_alpha):
    '''
    Generate test data for different values of nodes, hubs, gamma and alpha with CAB dataset.

    Input:
        W (ndarray): Demand matrix from CAB dataset
        C (ndarray): Distance(cost) matrix from CAB dataset
        N_P (list, length: 2): Combination of number of nodes and number of potential locations for the hubs
        gamma_alpha (list, length: 2): Combination of value of gamma and alpha

    Output:
        C[:N, :N] (ndarray): Distance(cost) matrix of nodes in N
        [X, alpha, delta] (list): Collection cost, transfer cost and distribution cost
        demand_dict (dictionary: {scenario: [ndarray, ...]}): Dictionary of demand matrices in each time period for each scenario
        highest_originate (ndarray): Highest total amount of flow originated at each node in each time period
        gamma (float): Capacity of a module
        f_k_t (ndarray): Cost for installing each hub in each time period
        g_k_q_t (dictionary: {time_period: nparray, ...}): Cost for building and operating different numbers of initial modules for each node in different time periods
        h_k_q_t (dictionary: {time_period: nparray, ...}): Cost for building and operating different numbers of additional modules for each node in different time periods
    '''

#    N_P_test_cases = {0: [15, 10], 1: [15, 15], 2: [20, 10], 3: [20, 15], 4: [20, 20], 5: [25, 10], 6: [25, 15], 7: [25, 20], 8: [25, 25]}
#    gamma_alpha_test_cases = {0: [0.075, 0.2], 1: [0.075, 0.4], 2: [0.075, 0.6], 3: [0.075, 0.8], 4: [0.1, 0.2], 5: [0.1, 0.4], 6: [0.1, 0.6], 7: [0.1, 0.8], \
#    8: [0.125, 0.2], 9: [0.125, 0.4], 10: [0.125, 0.6], 11: [0.125, 0.8]}
    X = 1
    delta = 1
    T = 3
    S = 5
    Q = 5
    beta = 0.2

#    N_P = N_P_test_cases[N_P_case]
    N = N_P[0]
    P = N_P[1]
#    gamma_alpha = gamma_alpha_test_cases[gamma_alpha_case]
    gamma = gamma_alpha[0]
    alpha = gamma_alpha[1]

    # Total flow originated at node k
    O_k = np.sum(W[:N, :N], axis = 1)

    # Cost for installing each hub in time period 1(0)
    f_k_1 = T*15*np.log(O_k)
    # Cost for installing each hub in each time period
    f_k_t = np.zeros([T, N])
    f_k_t[0,:] = f_k_1
    for i in range(T-1):
        f_k_t[i+1,:] = 1.02*f_k_t[i,:]

    # Cost for installing q initial modules in a hub at location k in period t
    initial_f_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        initial_f_k_q_1[:,q] = beta*f_k_1*q
    initial_f_k_q_t = {0: initial_f_k_q_1}
    for t in range(T-1):
        initial_f_k_q_t[t+1] = 1.02*initial_f_k_q_t[t]
    # Operating cost incurred when q initial modules are installed in a hub at location k in period t
    initial_m_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        initial_m_k_q_1[:,q] = 0.6*beta*f_k_1*q
    initial_m_k_q_t = {0: initial_m_k_q_1}
    for t in range(T-1):
        initial_m_k_q_t[t+1] = 1.02*initial_m_k_q_t[t]

    # Cost for installing q additional modules in a hub at location k in period t
    additional_f_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        additional_f_k_q_1[:,q] = 2*beta*f_k_1*q
    additional_f_k_q_t = {0: additional_f_k_q_1}
    for t in range(T-1):
        additional_f_k_q_t[t+1] = 1.02*additional_f_k_q_t[t]
    # Operating cost incurred when q additional modules are installed in a hub at location k in period t
    additional_m_k_q_t = initial_m_k_q_t.copy()

    # Total cost resulting from setting up the initial capacity of a hub with q modules to be located at k in period t
    # (setup cost and operating cost)
    g_k_q_t = {}
    sum_initial_operating = sum(initial_m_k_q_t.values())
    for t in range(T):
        g_k_q_t[t] = initial_f_k_q_t[t] + sum_initial_operating
        sum_initial_operating -= initial_m_k_q_t[t]
    # The cost for installing q additional modules in an existing hub at location k in period t
    # (setup cost and operating cost)
    h_k_q_t = {}
    sum_additional_operating = sum(additional_m_k_q_t.values())
    for t in range(T):
        h_k_q_t[t] = additional_f_k_q_t[t] + sum_additional_operating
        sum_additional_operating -= additional_m_k_q_t[t]

    # Demand in each time period for each scenario
    scaled_W = W[:N, :N]/np.sum(W[:N, :N])
    demand_dict = {s: [] for s in range(S)}
    for s in range(S):
        demand_dict[s].append(scaled_W)
        for t in range(1,T):
            W_ij_s_t = demand_dict[s][t-1]*(1 + 6/100*(s+1))
            demand_dict[s].append(W_ij_s_t)
    # Highest total amount of flow originated at node k in time t
    highest_originate = np.zeros([T, N], dtype=int)
    for t in range(T):
        for i in range(N):
            highest_originate_t_i = max(np.sum(demand_dict[s][t][i,:]) for s in range(S))
            highest_originate[t,i] = np.ceil(highest_originate_t_i/gamma)

    return C[:N, :N], [X, alpha, delta], demand_dict, highest_originate, gamma, f_k_t, g_k_q_t, h_k_q_t


def CAB_model_data_processing(W, C, N_P, gamma_alpha):
    '''
    Generate test data for the the full extensive form of the deterministic equivalent
    with different values of nodes, hubs, gamma and alpha based on CAB dataset.

    Input:
        W (ndarray): Demand matrix from CAB dataset
        C (ndarray): Distance(cost) matrix from CAB dataset
        N_P (list, length: 2): Combination of number of nodes and number of potential locations for the hubs
        gamma_alpha (list, length: 2): Combination of value of gamma and alpha

    Output:
        , g_k_q_t, h_k_q_t

        C[:N, :N] (ndarray): Distance(cost) matrix of nodes in N
        demand_dict (dictionary: {scenario: [ndarray, ...]}): Dictionary of demand matrices in each time period for each scenario
        originated_flow_dict (dictionary: {scenario: [ndarray, ...]}): Total flow originated at each node in each period and each scenario
        destined_flow_dict (dictionary: {scenario: [ndarray, ...]}): Total flow destined to each node in each period and each scenario
        f_k_t (ndarray): Cost for installing each hub in each time period
        initial_f_k_q_t (dictionary: {time_period: nparray, ...}): Fixed cost for installing modules in a hub in each period
        initial_m_k_q_t (dictionary: {time_period: nparray, ...}): Operating cost incurred when modules are installed in a hub in each period
        additional_f_k_q_t (dictionary: {time_period: nparray, ...}): Fixed cost for installing additional modules in a hub in each period
        additional_m_k_q_t (dictionary: {time_period: nparray, ...}): Operating cost incurred when additional modules are installed in a hub in each period
        g_k_q_t (dictionary: {time_period: nparray, ...}): Cost for building and operating different numbers of initial modules for each node in different time periods
        h_k_q_t (dictionary: {time_period: nparray, ...}): Cost for building and operating different numbers of additional modules for each node in different time periods
    '''

#    N_P_test_cases = {0: [15, 10], 1: [15, 15], 2: [20, 10], 3: [20, 15], 4: [20, 20], 5: [25, 10], 6: [25, 15], 7: [25, 20], 8: [25, 25]}
#    gamma_alpha_test_cases = {0: [0.075, 0.2], 1: [0.075, 0.4], 2: [0.075, 0.6], 3: [0.075, 0.8], 4: [0.1, 0.2], 5: [0.1, 0.4], 6: [0.1, 0.6], 7: [0.1, 0.8], \
#    8: [0.125, 0.2], 9: [0.125, 0.4], 10: [0.125, 0.6], 11: [0.125, 0.8]}
    X = 1
    delta = 1
    T = 3
    S = 5
    Q = 5
    beta = 0.2

#    N_P = N_P_test_cases[N_P_case]
    N = N_P[0]
    P = N_P[1]
#    gamma_alpha = gamma_alpha_test_cases[gamma_alpha_case]
    gamma = gamma_alpha[0]
    alpha = gamma_alpha[1]

    # Total flow originated at node k
    O_k = np.sum(W[:N, :N], axis = 1)

    # Cost for installing each hub in time period 1(0)
    f_k_1 = T*15*np.log(O_k)
    # Cost for installing each hub in each time period
    f_k_t = np.zeros([T, N])
    f_k_t[0,:] = f_k_1
    for i in range(T-1):
        f_k_t[i+1,:] = 1.02*f_k_t[i,:]

    # Cost for installing q initial modules in a hub at location k in period t
    initial_f_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        initial_f_k_q_1[:,q] = beta*f_k_1*q
    initial_f_k_q_t = {0: initial_f_k_q_1}
    for t in range(T-1):
        initial_f_k_q_t[t+1] = 1.02*initial_f_k_q_t[t]
    # Operating cost incurred when q initial modules are installed in a hub at location k in period t
    initial_m_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        initial_m_k_q_1[:,q] = 0.6*beta*f_k_1*q
    initial_m_k_q_t = {0: initial_m_k_q_1}
    for t in range(T-1):
        initial_m_k_q_t[t+1] = 1.02*initial_m_k_q_t[t]

    # Cost for installing q additional modules in a hub at location k in period t
    additional_f_k_q_1 = np.zeros([N, Q+1])
    for q in range(Q+1):
        additional_f_k_q_1[:,q] = 2*beta*f_k_1*q
    additional_f_k_q_t = {0: additional_f_k_q_1}
    for t in range(T-1):
        additional_f_k_q_t[t+1] = 1.02*additional_f_k_q_t[t]
    # Operating cost incurred when q additional modules are installed in a hub at location k in period t
    additional_m_k_q_t = initial_m_k_q_t.copy()

    # Total cost resulting from setting up the initial capacity of a hub with q modules to be located at k in period t
    # (setup cost and operating cost)
    g_k_q_t = {}
    sum_initial_operating = sum(initial_m_k_q_t.values())
    for t in range(T):
        g_k_q_t[t] = initial_f_k_q_t[t] + sum_initial_operating
        sum_initial_operating -= initial_m_k_q_t[t]
    # The cost for installing q additional modules in an existing hub at location k in period t
    # (setup cost and operating cost)
    h_k_q_t = {}
    sum_additional_operating = sum(additional_m_k_q_t.values())
    for t in range(T):
        h_k_q_t[t] = additional_f_k_q_t[t] + sum_additional_operating
        sum_additional_operating -= additional_m_k_q_t[t]

    # Demand in each time period for each scenario
    scaled_W = W[:N, :N]/np.sum(W[:N, :N])
    demand_dict = {s: [] for s in range(S)}
    for s in range(S):
        demand_dict[s].append(scaled_W)
        for t in range(1,T):
            W_ij_s_t = demand_dict[s][t-1]*(1 + 6/100*(s+1))
            demand_dict[s].append(W_ij_s_t)

    # total flow originated at node i in period t for each scenario
    originated_flow_dict = {}
    for s in range(S):
        originated_flow_matrix = np.zeros([T, N])
        for t in range(T):
            originated_flow = np.sum(demand_dict[s][t], axis = 1)
            originated_flow_matrix[t,:] = originated_flow
        originated_flow_dict[s] = originated_flow_matrix

    # total flow destined to node i in period t for each scenario
    destined_flow_dict = {}
    for s in range(S):
        destined_flow_matrix = np.zeros([T, N])
        for t in range(T):
            destined_flow = np.sum(demand_dict[s][t], axis = 0)
            destined_flow_matrix[t,:] = destined_flow
        destined_flow_dict[s] = destined_flow_matrix

    del additional_f_k_q_t[0]
    del additional_m_k_q_t[0]
    del h_k_q_t[0]
    return C[:N, :N], demand_dict, originated_flow_dict, destined_flow_dict, f_k_t, initial_f_k_q_t, initial_m_k_q_t, additional_f_k_q_t, additional_m_k_q_t, g_k_q_t, h_k_q_t
