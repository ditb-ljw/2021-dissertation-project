import sys
sys.path.append(r'C:\Users\ljw_l\anaconda3\Lib\site-packages')
from docplex.mp.model import Model
import numpy as np
import time
from GA.data_processing import W, C, CAB_model_data_processing

for gam in [0.075, 0.1, 0.125]:
    for alp in [0.2, 0.4, 0.6, 0.8]:
        # Data

        # Data with different combinations
        # set of nodes
        N = list(range(25))
        # set of potential locations for the hubs
        P = list(range(25))
        # set of time periods
        T = [0, 1, 2]
        # set of scenarios
        S = list(range(5))
        # probability for each scenario
        prob = {s: 0.2 for s in S}
        # capacity of a module
        gamma = gam
        # transfer cost in period t
        a = alp
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
        distance, demand_dict, originated_flow_dict, destined_flow_dict, \
        install_hub_cost_matrix, initial_f_k_q_t, initial_m_k_q_t, additional_f_k_q_t, additional_m_k_q_t, \
        initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_model_data_processing(W, C, N_P, gamma_alpha)

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
        start = time.time()
        M_DE = Model('Hub_location_problem')


        # Decision variables

        # if a hub is installed at k in period t: u_t_k = 1
        #idx_u = [(t, k) for t in T for k in P]
        #u = M_DE.binary_var_dict(idx_u, name = 'u')
        # if hub k receives q modules in period t: z_t_k_q = 1
        #idx_z = [(t, k, q) for t in T for k in P for q in range(1, Q[k]+1)]
        #z = M_DE.binary_var_dict(idx_z, name = 'z')
        # if hub k receives q additional modules in period t and scenario s: r_s_t_k_q = 1
        #idx_r = [(s, t, k, q) for s in S for t in [1,2] for k in P for q in range(1, Q[k])]
        #r = M_DE.binary_var_dict(idx_r, name = 'r')

        # relaxation
        # if a hub is installed at k in period t: u_t_k = 1
        idx_u = [(t, k) for t in T for k in P]
        u = M_DE.continuous_var_dict(idx_u, lb=0, ub = 1, name = 'u')
        # if hub k receives q modules in period t: z_t_k_q = 1
        idx_z = [(t, k, q) for t in T for k in P for q in range(1, Q[k]+1)]
        z = M_DE.continuous_var_dict(idx_z, lb=0, ub = 1, name = 'z')
        # if hub k receives q additional modules in period t and scenario s: r_s_t_k_q = 1
        idx_r = [(s, t, k, q) for s in S for t in [1,2] for k in P for q in range(1, Q[k])]
        r = M_DE.continuous_var_dict(idx_r, lb=0, ub = 1, name = 'r')

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

        # Constraints (4) impose that in each time period and for each potential location for the hubs,
        # if a hub is installed then it must have a certain number of modules which define its initial capacity.
        M_DE.add_constraints((sum(z[(t, k, q)] for q in range(1, Q[k]+1)) == u[(t, k)] \
                              for k in P for t in T), \
                              names = '(4)')
        # Constraints (5) state that one hub can be installed at most once during the planning horizon.
        M_DE.add_constraints((sum(u[(t, k)] for t in T) <= 1 \
                              for k in P), \
                              names = '(5)')
        # Constraints (43) assure that in each time period and scenario the flow originated in each node is routed via at least one hub.
        M_DE.add_constraints((sum(x[(s, t, i, k)] for k in P) == O[s][t, i] \
                              for i in N for t in T for s in S), \
                              names = '(43)')
        # Constraints (44) assure that in each time period and scenario the flow between each pair of nodes is distributed via at least one hub.
        M_DE.add_constraints((sum(v[(s, t, i, l, j)] for l in P) == w[s][t][i, j] \
                              for i in N for j in N for t in T for s in S), \
                              names = '(44)')
        # Constraints (45) state that the expansion of the hub’s capacity in a time period and scenario is only allowed if that hub was installed in a previous period.
        M_DE.add_constraints((sum(r[(s, t, k, q)] for q in range(1, Q[k])) <= sum(u[(tau, k)] for tau in range(t)) \
                              for k in P for t in [1,2] for s in S), \
                              names = '(45)')
        # Inequalities (46) assure that, for each hub, in each time period and scenario, the incoming flow from non-hubs as well as the flow originated at the hub cannot exceed the hub’s capacity.
        M_DE.add_constraints((sum(x[(s, t, i, k)] for i in N) <= \
                              gamma*sum(q*z[(tau, k, q)] for q in range(1, Q[k]+1) for tau in range(t+1)) + \
                              gamma*sum(q*r[(s, tau, k, q)] for q in range(1, Q[k]) for tau in range(1, t+1)) \
                              for k in P for t in T for s in S), \
                              names = '(46)')
        # In each potential location for the hubs there is a maximum number of modules that can be installed as constraints (47) indicate.
        M_DE.add_constraints((sum(q*z[(t, k, q)] for t in T for q in range(1, Q[k]+1)) + \
                              sum(q*r[(s, t, k, q)] for t in [1,2] for q in range(1, Q[k])) <= Q[k] \
                              for k in P for s in S), \
                              names = '(47)')
        # The flow conservation and divergence constraints are described by (48).
        M_DE.add_constraints((sum(y[(s, t, i, k, l)] for l in P if l != k) - sum(y[(s, t, i, l, k)] for l in P if l != k) == \
                              x[(s, t, i, k)] - sum(v[(s, t, i, k, j)] for j in N) \
                              for i in N for k in P for t in T for s in S), \
                              names = '(48)')
        # The condition defined by (49) ensures that consistent values are obtained for y-variables.
        M_DE.add_constraints((sum(y[(s, t, i, k, l)] for l in P if l != k) <= x[(s, t, i, k)] \
                              for i in N for k in P for t in T for s in S), \
                              names = '(49)')
        # Inequalities (50) impose that a hub must collect its own outbound flow.
        M_DE.add_constraints((x[(s, t, i, k)] <= O[s][t, i]*(1 - sum(u[(tau, i)] for tau in range(t+1))) \
                              for s in S for t in T for i in P for k in P if k != i), \
                              names = '(50)')
        # Constraints (51) state that the flow destined to any node can be distributed only by an open hub.
        M_DE.add_constraints((sum(v[(s, t, i, l, j)] for i in N) <= D[s][t, j]*sum(u[(tau, l)] for tau in range(t+1)) \
                              for j in N for l in P for t in T for s in S), \
                              names = '(51)')
        # A hub needs to distribute its own inbound flow as stated by constraints (52).
        M_DE.add_constraints((v[(s, t, i, l, j)] <= D[s][t, j]*(1 - sum(u[(tau, j)] for tau in range(t+1))) \
                              for s in S for t in T for i in N for l in P for j in P if j != l), \
                              names = '(52)')

        # valid inequalities
        # (36) (enhanced (47))
        #M_DE.add_constraints((sum(q*z[(t, k, q)] for t in T for q in range(1, Q[k]+1)) + sum(q*r[(s, t, k, q)] for t in [1,2] for q in range(1, Q[k])) <= \
        #                      Q[k]*sum(u[(t, k)] for t in T) \
        #                      for k in P for s in S), \
        #                      names = '(36)')
        # (37) (enhanced (36))
        M_DE.add_constraints((sum(np.floor(q/l)*z[(t, k, q)] for t in T for q in range(1, Q[k]+1)) + sum(np.floor(q/l)*r[(s, t, k, q)] for t in [1,2] for q in range(1, Q[k])) <= \
                              np.floor(Q[k]/l)*sum(u[(t, k)] for t in T) \
                              for k in P for s in S for l in range(1, Q[k]+1)), \
                              names = '(37)')
        # (38)
        #M_DE.add_constraints((sum(q*z[(t, k, q)] for q in range(1, Q[k]+1)) + sum(q*r[(s, tau, k, q)] for tau in range(t+1, len(T)) for q in range(1, Q[k])) >= \
        #                      np.ceil(max(O[s][t_au, k] for t_au in range(t, len(T)))/gamma)*u[(t, k)] \
        #                      for k in P for t in T for s in S), \
        #                      names = '(38)')
        # (39) (enhanced(38))
        M_DE.add_constraints((sum(np.ceil(q/l)*z[(t, k, q)] for q in range(1, Q[k]+1)) + sum(np.ceil(q/l)*r[(s, tau, k, q)] for tau in range(t+1, len(T)) for q in range(1, Q[k])) >= \
                              np.ceil(np.ceil(max(O[s][t_au, k] for t_au in range(t, len(T)))/gamma)/l)*u[(t, k)] \
                              for k in P for t in T for s in S for l in range(1, int(np.ceil(max(O[s][t_au, k] for t_au in range(t, len(T)))/gamma) + 1))), \
                              names = '(39)')
        # (40)
        M_DE.add_constraints((sum(np.ceil(q/l)*z[(tau, k, q)] for k in P for q in range(1, Q[k]+1) for tau in range(t+1)) + \
                              sum(np.ceil(q/l)*r[(s, tau, k, q)] for k in P for q in range(1, Q[k]) for tau in range(1, t+1)) >= \
                              np.ceil(np.ceil(np.sum(w[s][t])/gamma)/l) \
                              for t in T for s in S for l in range(1, 5+1)), \
                              names = '(40)')
        # (41)
        M_DE.add_constraints((x[(s, t, i, k)] <= O[s][t, i]*sum(u[(tau, k)] for tau in range(t+1)) \
                              for i in N for k in P for s in S for t in T), \
                              names = '(41)')


        # Objective function (42)
        obj_f = sum((f[t, k]*u[(t, k)] + sum(g[t][k, q]*z[(t, k, q)] for q in range(1, Q[k]+1))) for k in P for t in T) + \
                sum(prob[s]*(sum(X[t]*d[i, k]*x[(s, t, i, k)] for i in N for k in P for t in T) + \
                sum(alpha[t]*d[k, l]*y[(s, t, i, k, l)] for i in N for k in P for l in P for t in T) + \
                sum(delta[t]*d[l, j]*v[(s, t, i, l, j)] for i in N for l in P for j in N for t in T) + \
                sum(h[t][k, q]*r[(s, t, k, q)] for k in P for q in range(1, Q[k]) for t in [1, 2])) for s in S)
        M_DE.set_objective('min', obj_f)
        stop = time.time()
        construct_time = round(stop-start, 2)

        # M_DE.print_information()


        # Solution
        start = time.time()
        M_DE.solve()
        stop = time.time()
        solve_time = round(stop-start, 2)
#        print(str(len(N))+' '+str(len(P))+' '+str(gam)+'  '+str(alp))
        print(str(construct_time)+'   '+str(solve_time)+'   '+str(round(M_DE.objective_value, 2)))
