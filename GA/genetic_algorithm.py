import numpy as np
from GA.Chromosomes import chromosome
from GA.data_processing import W, C, CAB_data_processing


N_P = [15, 10]
gamma_alpha = [0.075, 0.2]

distance, coefficients, demand_dict, highest_originate, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_data_processing(W, C, N_P, gamma_alpha)

# test
hub_locations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
initial_capacity_matrix = np.array([[5, 1, 1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
test_c = chromosome(initial_capacity_matrix, distance, hub_locations, coefficients, demand_dict, highest_originate, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict)
print(test_c.is_feasible())
print(test_c.fitness())
