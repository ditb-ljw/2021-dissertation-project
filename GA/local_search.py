import numpy as np
import random
from GA.Chromosomes import chromosome
from GA.data_processing import test_data


def generate_initial_population():
#    pop_size
    max_capacity = test_data()[1]
    demand_dict = test_data()[3]
    scenarios = test_data()[4]
    module_capacity = test_data()[6]

    population = {}
    node_number = len(max_capacity)
    time_periods = len(demand_dict[0])

#    time_0_max_demand = max(np.sum(demand_dict[s][0]) for s in scenarios)
#    total_num_module = np.ceil(time_0_max_demand / module_capacity)
#    average_module_hub = int(total_num_module / node_number)

    chromosome_matrix = np.zeros([time_periods,node_number])
    random_chromosome = chromosome(chromosome_matrix)
    while random_chromosome.if_feasible() == False:
        chromosome_matrix = np.zeros([time_periods,node_number])
        old_hubs = []
        for i in range(time_periods):
            for j in range(node_number):
                if j not in old_hubs:
                    module_number = random.choice(range(max_capacity[j]+1))
                    if module_number > 0:
                        chromosome_matrix[i,j] = module_number
                        old_hubs.append(j)
                else:
                    continue
        random_chromosome = chromosome(chromosome_matrix)

    return random_chromosome


generate_initial_population().initial_capacity_matrix
