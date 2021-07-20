import numpy as np
from GA.Chromosomes import chromosome
from GA.data_processing import W, C, CAB_data_processing
from GA.local_search import rand_chromosome_matrix, generate_initial_chromosome, find_neighbourhood, local_optimum

N_P = [15, 10]
gamma_alpha = [0.075, 0.2]
hub_locations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

distance, coefficients, demand_dict, highest_originate, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_data_processing(W, C, N_P, gamma_alpha)
test_data = {'distance': distance, 'hub_locations': hub_locations, 'coefficients': coefficients, 'demand_dict': demand_dict, \
'highest_originate': highest_originate, 'module_capacity': module_capacity, 'install_hub_cost_matrix': install_hub_cost_matrix, \
'initial_capacity_cost_dict': initial_capacity_cost_dict, 'additional_capacity_cost_dict': additional_capacity_cost_dict}

# test

initial_capacity_matrix = rand_chromosome_matrix(N_P[0], hub_locations, highest_originate)
neighbourhood_matrix = rand_neighbourhood(initial_capacity_matrix, hub_locations, highest_originate)

#initial_capacity_matrix = np.array([[5, 1, 1, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
test_a = chromosome(initial_capacity_matrix, test_data)
print(test_a.is_feasible())
test_a.calculate_fitness()
test_a.fitness

test_b = chromosome(neighbourhood_matrix, test_data)
print(test_b.is_feasible())
print(test_b.fitness())



initial_chromosome_list = generate_initial_chromosome(1, test_data)
initial_chromosome = initial_chromosome_list[0]
initial_capacity_matrix = initial_chromosome.initial_capacity_matrix
print(initial_capacity_matrix)
print(initial_chromosome.fitness)
# a = find_neighbourhood(initial_chromosome_list[0], 10)
b = local_optimum(initial_chromosome, 30, 30)
print(b.initial_capacity_matrix)
print(b.fitness)