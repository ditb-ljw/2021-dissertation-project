import numpy as np
import time
import pandas as pd
from GA.Chromosomes import chromosome
from GA.data_processing import W, C, CAB_data_processing
from GA.local_search import rand_chromosome_matrix, rand_neighbourhood, generate_initial_chromosome, find_neighbourhood, local_optimum
from GA.genetic_algorithm import one_pt_col_crossover, uniform_col_crossover, one_pt_row_crossover, mutation, new_generation, GA

N_P = [15, 10]
gamma_alpha = [0.075, 0.2]
hub_locations = list(range(10))

distance, coefficients, demand_dict, highest_originate, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_data_processing(W, C, N_P, gamma_alpha)
test_data = {'distance': distance, 'hub_locations': hub_locations, 'coefficients': coefficients, 'demand_dict': demand_dict, \
'highest_originate': highest_originate, 'module_capacity': module_capacity, 'install_hub_cost_matrix': install_hub_cost_matrix, \
'initial_capacity_cost_dict': initial_capacity_cost_dict, 'additional_capacity_cost_dict': additional_capacity_cost_dict}

start = time.time()
initial_chromosome_list = generate_initial_chromosome(50, test_data, True)
stop = time.time()
rand_initial_time = round(stop-start, 2)

start = time.time()
initial_chromosomes = [local_optimum(initial_chromosome, 10, 10, True) for initial_chromosome in initial_chromosome_list[:int((1-0)*50)]] \
+ initial_chromosome_list[int((1-0)*50):]
stop = time.time()
ls_time = round(stop-start, 2)

start = time.time()
res = GA(initial_chromosomes, 50, 10, 0.3, 0.3, 0.1, 0.01, True)
stop = time.time()
ga_time = round(stop-start, 2)

cost =  round(1/res.fitness, 2)
print('='+str(rand_initial_time)+'+'+str(ls_time)+'+'+str(ga_time))
print(cost)

# test
initial_chromosome_list = generate_initial_chromosome(100, test_data, False)
b = [local_optimum(initial_chromosome, 10, 20, False) for initial_chromosome in initial_chromosome_list]
res = GA(b, 100, 10, 0.4, 0.4, 0.1, 0.04, False)
1/res.fitness

res = GA(initial_chromosome_list, 100, 10, 0.4, 0.3, 0.1, 0.01, False)
res = local_optimum(res, 20, 20, False)
1/res.fitness

initial_chromosome_list = generate_initial_chromosome(100, test_data, True)
b = [local_optimum(initial_chromosome, 10, 10, True) for initial_chromosome in initial_chromosome_list]
res = GA(b, 100, 10, 0.4, 0.4, 0.1, 0.04, True)
1/res.fitness

res = GA(initial_chromosome_list, 100, 10, 0.4, 0.3, 0.1, 0.01, True)
res_1 = local_optimum(res, 50, 50, True)
1/res_1.fitness



# Test data
N_P = [[15, 10], [25, 25]]
gamma_alpha = [[0.075, 0.2], [0.125, 0.8]]
hub_locations = [list(range(10)), list(range(25))]
reroute_prefer_expand = [True, False]

# Genetic Algorithm
initial_random_portion_list = [0, 0.5]
pop_size_list = [50, 100]
num_generation_list = [50, 100]
num_unchange_list = [10, 20]
one_pt_col_crossover_probability_list = [0.3, 0.4]
uniform_col_crossover_probability_list = [0.3, 0.4]
one_pt_row_crossover_probability_list = [0.1, 0.15]
mutation_probability_list = [0.01, 0.04]

# Local Search
move_times_list = [10, 20]
neighbourhood_numbers_list = [10, 20]



# experiments
N_P = [15, 10]
gamma_alpha = [0.075, 0.2]
hub_locations = list(range(10))

distance, coefficients, demand_dict, highest_originate, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_data_processing(W, C, N_P, gamma_alpha)
test_data = {'distance': distance, 'hub_locations': hub_locations, 'coefficients': coefficients, 'demand_dict': demand_dict, \
'highest_originate': highest_originate, 'module_capacity': module_capacity, 'install_hub_cost_matrix': install_hub_cost_matrix, \
'initial_capacity_cost_dict': initial_capacity_cost_dict, 'additional_capacity_cost_dict': additional_capacity_cost_dict}

reroute_prefer_expand = False
print(reroute_prefer_expand)

experiments = {1:[0,0,0,0,0,0,0,0,0,0], 2:[0,0,0,0,0,1,1,1,1,1], 3:[0,0,1,1,1,0,0,0,1,1], 4:[0,1,0,1,1,0,1,1,0,0], 5:[0,1,1,0,1,1,0,1,0,1], 6:[0,1,1,1,0,1,1,0,1,0], 7:[1,0,1,1,0,0,1,1,0,1], 8:[1,0,1,0,1,1,1,0,0,0], 9:[1,0,0,1,1,1,0,1,1,0], 10:[1,1,1,0,0,0,0,1,1,0], 11:[1,1,0,1,0,1,0,0,0,1], 12:[1,1,0,0,1,0,1,0,1,1]}

for i in range(1,13):
    print(i)

    data_index = experiments[i]

    initial_random_portion = initial_random_portion_list[data_index[0]]
    pop_size = pop_size_list[data_index[1]]
    num_generation = num_generation_list[data_index[2]]
    num_unchange = num_unchange_list[data_index[3]]
    one_pt_col_crossover_probability = one_pt_col_crossover_probability_list[data_index[4]]
    uniform_col_crossover_probability = uniform_col_crossover_probability_list[data_index[5]]
    one_pt_row_crossover_probability = one_pt_row_crossover_probability_list[data_index[6]]
    mutation_probability = mutation_probability_list[data_index[7]]

    move_times = move_times_list[data_index[8]]
    neighbourhood_numbers = neighbourhood_numbers_list[data_index[9]]


    rand_initial_time_list = []
    ls_time_list = []
    ga_time_list = []
    cost_list = []

    for j in range(5):
        print(j)
        start = time.time()
        initial_chromosome_list = generate_initial_chromosome(pop_size, test_data, reroute_prefer_expand)
        stop = time.time()
        rand_initial_time = round(stop-start, 2)

        start = time.time()
        initial_chromosomes = [local_optimum(initial_chromosome, move_times, neighbourhood_numbers, reroute_prefer_expand) for initial_chromosome in initial_chromosome_list[:int((1-initial_random_portion)*pop_size)]] \
        + initial_chromosome_list[int((1-initial_random_portion)*pop_size):]
        stop = time.time()
        ls_time = round(stop-start, 2)

        start = time.time()
        res = GA(initial_chromosomes, num_generation, num_unchange, one_pt_col_crossover_probability, uniform_col_crossover_probability, one_pt_row_crossover_probability, mutation_probability, reroute_prefer_expand)
        stop = time.time()
        ga_time = round(stop-start, 2)

        cost =  round(1/res.fitness, 2)

        rand_initial_time_list.append(rand_initial_time)
        ls_time_list.append(ls_time)
        ga_time_list.append(ga_time)
        cost_list.append(cost)

    time_str = ''
    cost_str = ''
    for k in range(5):
        time_str += '='+str(rand_initial_time_list[k])+'+'+str(ls_time_list[k])+'+'+str(ga_time_list[k])+','
        cost_str += str(cost_list[k])+','

    print(time_str)
    print(cost_str)


# GALS
# Test data
N_P = [[15, 10], [25, 25]]
gamma_alpha = [[0.075, 0.2], [0.125, 0.8]]
hub_locations = [list(range(10)), list(range(25))]
reroute_prefer_expand = [True, False]

# Genetic Algorithm
pop_size_list = [50, 100]
num_generation_list = [50, 100]
num_unchange_list = [10, 20]
one_pt_col_crossover_probability_list = [0.3, 0.4]
uniform_col_crossover_probability_list = [0.3, 0.4]
one_pt_row_crossover_probability_list = [0.1, 0.15]
mutation_probability_list = [0.01, 0.04]

# Local Search
move_times_list = [50, 100]
neighbourhood_numbers_list = [50, 100]



# experiments
N_P = [25, 25]
gamma_alpha = [0.125, 0.8]
hub_locations = list(range(25))

distance, coefficients, demand_dict, highest_originate, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_data_processing(W, C, N_P, gamma_alpha)
test_data = {'distance': distance, 'hub_locations': hub_locations, 'coefficients': coefficients, 'demand_dict': demand_dict, \
'highest_originate': highest_originate, 'module_capacity': module_capacity, 'install_hub_cost_matrix': install_hub_cost_matrix, \
'initial_capacity_cost_dict': initial_capacity_cost_dict, 'additional_capacity_cost_dict': additional_capacity_cost_dict}

reroute_prefer_expand_list = [True]

experiments = {1:[0,0,0,0,0,0,0,0,0], 2:[0,0,0,0,0,1,1,1,1], 3:[0,0,1,1,1,0,0,0,1], 4:[0,1,0,1,1,0,1,1,0], 5:[0,1,1,0,1,1,0,1,0], 6:[0,1,1,1,0,1,1,0,1], 7:[1,0,1,1,0,0,1,1,0], 8:[1,0,1,0,1,1,1,0,0], 9:[1,0,0,1,1,1,0,1,1], 10:[1,1,1,0,0,0,0,1,1], 11:[1,1,0,1,0,1,0,0,0], 12:[1,1,0,0,1,0,1,0,1]}

for reroute_prefer_expand in reroute_prefer_expand_list:
    print(reroute_prefer_expand)
    for i in range(6,13):
        print(i)

        data_index = experiments[i]

        pop_size = pop_size_list[data_index[0]]
        num_generation = num_generation_list[data_index[1]]
        num_unchange = num_unchange_list[data_index[2]]
        one_pt_col_crossover_probability = one_pt_col_crossover_probability_list[data_index[3]]
        uniform_col_crossover_probability = uniform_col_crossover_probability_list[data_index[4]]
        one_pt_row_crossover_probability = one_pt_row_crossover_probability_list[data_index[5]]
        mutation_probability = mutation_probability_list[data_index[6]]

        move_times = move_times_list[data_index[7]]
        neighbourhood_numbers = neighbourhood_numbers_list[data_index[8]]

        rand_initial_time_list = []
        ga_time_list = []
        ls_time_list = []
        cost_list = []

        for j in range(5):
            print(j)
            start = time.time()
            initial_chromosome_list = generate_initial_chromosome(pop_size, test_data, reroute_prefer_expand)
            stop = time.time()
            rand_initial_time = round(stop-start, 2)

            start = time.time()
            after_ga = GA(initial_chromosome_list, num_generation, num_unchange, one_pt_col_crossover_probability, uniform_col_crossover_probability, one_pt_row_crossover_probability, mutation_probability, reroute_prefer_expand)
            stop = time.time()
            ga_time = round(stop-start, 2)

            start = time.time()
            print(after_ga.fitness)
            after_ls = local_optimum(after_ga, move_times, neighbourhood_numbers, reroute_prefer_expand)
            stop = time.time()
            ls_time = round(stop-start, 2)

            cost =  round(1/after_ls.fitness, 2)

            rand_initial_time_list.append(rand_initial_time)
            ga_time_list.append(ga_time)
            ls_time_list.append(ls_time)
            cost_list.append(cost)

        time_str = ''
        cost_str = ''
        for k in range(5):
            time_str += '='+str(rand_initial_time_list[k])+'+'+str(ga_time_list[k])+'+'+str(ls_time_list[k])+','
            cost_str += str(cost_list[k])+','

        print(time_str)
        print(cost_str)






# LS
# Test data
N_P = [[15, 10], [25, 25]]
gamma_alpha = [[0.075, 0.2], [0.125, 0.8]]
hub_locations = [list(range(10)), list(range(25))]
reroute_prefer_expand = [True, False]

# Local Search
pop_size_list = [1, 25, 50]
move_times_list = [50, 100, 200]
neighbourhood_numbers_list = [50, 100, 200]



# experiments
N_P = [25, 25]
gamma_alpha = [0.125, 0.8]
hub_locations = list(range(25))

distance, coefficients, demand_dict, highest_originate, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict = CAB_data_processing(W, C, N_P, gamma_alpha)
test_data = {'distance': distance, 'hub_locations': hub_locations, 'coefficients': coefficients, 'demand_dict': demand_dict, \
'highest_originate': highest_originate, 'module_capacity': module_capacity, 'install_hub_cost_matrix': install_hub_cost_matrix, \
'initial_capacity_cost_dict': initial_capacity_cost_dict, 'additional_capacity_cost_dict': additional_capacity_cost_dict}

reroute_prefer_expand_list = [True, False]

experiments = {1:[0, 0, 0], 2:[0, 1, 1], 3:[0, 2, 2], 4:[1, 0, 1], 5:[1, 1, 2], 6:[1, 2, 0], 7:[2, 0, 2], 8:[2, 1, 0], 9:[2, 2, 1]}

for reroute_prefer_expand in reroute_prefer_expand_list:
    print(reroute_prefer_expand)
    for i in range(1,10):
        print(i)

        data_index = experiments[i]

        pop_size = pop_size_list[data_index[0]]
        move_times = move_times_list[data_index[1]]
        neighbourhood_numbers = neighbourhood_numbers_list[data_index[2]]


        rand_initial_time_list = []
        ls_time_list = []
        cost_list = []

        for j in range(5):
            print(j)
            start = time.time()
            initial_chromosome_list = generate_initial_chromosome(pop_size, test_data, reroute_prefer_expand)
            stop = time.time()
            rand_initial_time = round(stop-start, 2)

            start = time.time()
            after_ls = [local_optimum(initial_chromosome, move_times, neighbourhood_numbers, reroute_prefer_expand) for initial_chromosome in initial_chromosome_list]
            after_ls_fitness = [c.fitness for c in after_ls]
            zipped_lists = zip(after_ls_fitness, after_ls)
            sorted_pairs = sorted(zipped_lists, reverse = True)
            sorted_after_ls = [pair[1] for pair in sorted_pairs]
            best_one = sorted_after_ls[0]
            stop = time.time()
            ls_time = round(stop-start, 2)
            print(best_one.fitness)

            cost =  round(1/best_one.fitness, 2)

            rand_initial_time_list.append(rand_initial_time)
            ls_time_list.append(ls_time)
            cost_list.append(cost)

        time_str = ''
        cost_str = ''
        for k in range(5):
            time_str += '='+str(rand_initial_time_list[k])+'+'+str(ls_time_list[k])+','
            cost_str += str(cost_list[k])+','

        print(time_str)
        print(cost_str)
