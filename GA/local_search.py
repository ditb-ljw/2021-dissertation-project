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
            if k not in old_hubs:
                build_hub = random.choice([0,1])
            else:
                continue
            if build_hub == 1:
                at_least_modules = highest_originate[t, k]
                module_number = random.choice(range(at_least_modules, 6))
                chromosome_matrix[t,k] = module_number
                old_hubs.append(k)

    return chromosome_matrix
