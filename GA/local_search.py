import numpy as np
import random


def rand_chromosome_matrix(N_num, hub_locations, highest_originate):
    '''
    Generate a random initial capacity(module) matrix for chromosome(may be infeasible).

    Input:
        N_num (int): Number of nodes
        hub_locations (list): Indices of potential locations for the hubs
        highest_originate (ndarray): Highest total amount of flow originated at each node in each time period

    Output:
        chromosome_matrix (ndarray): A random initial capacity(module) matrix for chromosome(may be infeasible)
    '''

    chromosome_matrix = np.zeros([3,N_num], dtype=int)
    old_hubs = []
    for t in range(3):
        for k in hub_locations:
            # Hubs can only be built at the location where no hubs have been built
            if k not in old_hubs:
                build_hub = random.choice([0,1])    # build a hub with probability 0.5
            else:
                continue
            if build_hub == 1:
                at_least_modules = highest_originate[t, k]
                module_number = random.choice(range(at_least_modules, 6))   # Number of modules should be in a feasible range
                chromosome_matrix[t,k] = module_number
                old_hubs.append(k)

    return chromosome_matrix


def rand_neighbourhood(chromosome_matrix, hub_locations, highest_originate):

    neighbourhood_matrix = chromosome_matrix.copy()

    rand_time = random.choice([0,1,2])
    rand_hub = random.choice(hub_locations)

    plus_minus = random.choice([0,1])

    if plus_minus == 0:
        neighbourhood_matrix[rand_time, rand_hub] += 1
    else:
        neighbourhood_matrix[rand_time, rand_hub] -= 1

    return neighbourhood_matrix
