import numpy as np
import random
from GA.Chromosomes import chromosome


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
    '''
    Randomly generate a neighbourhood matrix(may be infeasible) of the input matrix.

    Input:
        chromosome_matrix (ndarray): A random initial capacity(module) matrix for chromosome
        hub_locations (list): Indices of potential locations for the hubs
        highest_originate (ndarray): Highest total amount of flow originated at each node in each time period

    Output:
        neighbourhood_matrix (ndarray): A neighbourhood matrix(may be infeasible) of the input matrix
    '''

    neighbourhood_matrix = chromosome_matrix.copy()

    # Randomly choose an entry to make change
    rand_entry = []
    rand_time = random.choice([0,1,2])
    rand_hub = random.choice(hub_locations)
    rand_entry.append((rand_time, rand_hub))


    while True:

        # Do hub-based move or capacity-based move with 0.5 probability respectively
        hub_or_capacity = random.choice([0,1])

        if hub_or_capacity == 0:
        # Hub-based move

            if neighbourhood_matrix[rand_time, rand_hub] > 0:
            # Close this hub in this time period
                neighbourhood_matrix[rand_time, rand_hub] = 0
                # Close this hub or relocate the hub with 0.5 probability respectively
                close_or_relocate = random.choice([0,1])

                if close_or_relocate == 0:
                # Close the hub without relocating
                    return neighbourhood_matrix
                else:
                # Relocate the closed hub
                    # Relocate the closed hub in the same time period at a different location
                    # or in a different time period but at the same location with 0.5 probability respectively
                    relocate_same_time = random.choice([0,1])

                    if relocate_same_time == 0:
                    # Relocate the closed hub in the same time period at a different location
                        available_hubs = hub_locations.copy()
                        available_hubs.remove(rand_hub)

                        while len(available_hubs) > 0:
                            # Randomly choose a location to relocate the hub
                            relocate_location = random.choice(available_hubs)
                            if sum(chromosome_matrix[:, relocate_location]) == 0:
                            # No hub is built at this location in all time periods, so the hub can be relocate at this location
                                module_number = max(chromosome_matrix[rand_time, rand_hub], highest_originate[rand_time, relocate_location])
                                neighbourhood_matrix[rand_time, relocate_location] = module_number
                                return neighbourhood_matrix
                            else:
                            # There is a hub built at this location in a time period, so the hub cannot be relocate at this location
                                available_hubs.remove(relocate_location)

                    # Relocate the closed hub in a different time period but at the same location
                    available_time_periods = [0,1,2]
                    available_time_periods.remove(rand_time)
                    # Randomly choose a time period to relocate the hub
                    relocate_time_period = random.choice(available_time_periods)
                    module_number = max(chromosome_matrix[rand_time, rand_hub], highest_originate[relocate_time_period, rand_hub])
                    neighbourhood_matrix[relocate_time_period, rand_hub] = module_number
                    return neighbourhood_matrix

            else:
            # Build a hub at this location in this time period
                if sum(chromosome_matrix[:, rand_hub]) == 0:
                # No hub is built at this location in all time periods, so a hub can be built at this location
                    # Build a hub or relocate the hub with 0.5 probability respectively
                    build_or_relocate = random.choice([0,1])

                    if build_or_relocate == 0:
                    # Build a hub without relocating
                        module_number = highest_originate[rand_time, rand_hub]
                        neighbourhood_matrix[rand_time, rand_hub] = module_number
                        return neighbourhood_matrix
                    else:
                    # Relocate a hub in this time period to the chosen location
                        available_hubs = hub_locations.copy()
                        available_hubs.remove(rand_hub)

                        while len(available_hubs) > 0:
                            # Randomly choose a location to relocate the hub
                            relocate_location = random.choice(available_hubs)
                            chosen_hub_capacity = chromosome_matrix[rand_time, relocate_location]
                            if chosen_hub_capacity > 0:
                            # There is a hub built at the chosen location
                                neighbourhood_matrix[rand_time, relocate_location] = 0
                                module_number = max(chosen_hub_capacity, highest_originate[rand_time, rand_hub])
                                neighbourhood_matrix[rand_time, rand_time] = module_number
                                return neighbourhood_matrix
                            else:
                            # There is no hub built at the chosen location
                                available_hubs.remove(relocate_location)

                        # There is no hub in available_hubs
                        # Build a hub
                        module_number = highest_originate[rand_time, rand_hub]
                        neighbourhood_matrix[rand_time, rand_hub] = module_number
                        return neighbourhood_matrix

                else:
                # There is a hub built at this location in a time period, so adjust the set up time of the hub to this time period
                    available_time_periods = [0,1,2]
                    available_time_periods.remove(rand_time)
                    # Randomly choose a time period when a hub was built
                    chosen_time_period = random.choice(available_time_periods)
                    chosen_hub_capacity = chromosome_matrix[chosen_time_period, rand_hub]
                    while chosen_hub_capacity == 0:
                        available_time_periods.remove(chosen_time_period)
                        chosen_time_period = random.choice(available_time_periods)
                        chosen_hub_capacity = chromosome_matrix[chosen_time_period, rand_hub]

                    neighbourhood_matrix[chosen_time_period, rand_hub] = 0
                    module_number = max(chromosome_matrix[chosen_time_period, rand_hub], highest_originate[rand_time, rand_hub])
                    neighbourhood_matrix[rand_time, rand_hub] = module_number
                    return neighbourhood_matrix


        else:
        # Capacity-based move
            chosen_capacity = neighbourhood_matrix[rand_time, rand_hub]

            if chosen_capacity == 5:
            # The capacity has reached the maximum limit
                # Reduce the capacity by 1 or reinstall 1 module to another hub in the same time period with 0.5 probability respectively
                minus_or_reinstall = random.choice([0,1])
                neighbourhood_matrix[rand_time, rand_hub] -= 1

                if minus_or_reinstall == 0:
                # Reduce the capacity by 1
                    return neighbourhood_matrix
                else:
                # Reinstall 1 module to another hub in the same time period
                    available_hubs = hub_locations.copy()
                    available_hubs.remove(rand_hub)

                    while len(available_hubs) > 0:
                        # Randomly choose a hub to install the module
                        reinstall_hub = random.choice(available_hubs)
                        if chromosome_matrix[rand_time, reinstall_hub] > 0 and chromosome_matrix[rand_time, reinstall_hub] < 5:
                        # The capacity of the chosen hub has not reached the maximum limit
                            neighbourhood_matrix[rand_time, reinstall_hub] += 1
                            return neighbourhood_matrix
                        else:
                        # The capacity of the chosen hub has reached the maximum limit
                            available_hubs.remove(reinstall_hub)

                    # There is no hub in available_hubs
                    # Cannot reinstall module
                    return neighbourhood_matrix

            elif chosen_capacity < 5 and chosen_capacity > highest_originate[rand_time, rand_hub]:
            # The capacity has not reached the maximum limit and is greater than the minimum limit
                # Change the capacity by 1 or reinstall 1 module to another hub in the same time period with 0.5 probability respectively
                change_or_reinstall = random.choice([0,1])

                if change_or_reinstall == 0:
                # Change the capacity by 1
                    # Increase or reduce the capacity with 0.5 probability respectively
                    plus_minus = random.choice([0,1])

                    if plus_minus == 0:
                        # Increase the capacity by 1
                        neighbourhood_matrix[rand_time, rand_hub] += 1
                    else:
                        # Reduce the capacity by 1
                        neighbourhood_matrix[rand_time, rand_hub] -= 1
                    return neighbourhood_matrix
                else:
                # Reinstall 1 module to another hub in the same time period
                    # Increase or reduce the capacity of the chosen hub with 0.5 probability respectively
                    plus_minus = random.choice([0,1])
                    if plus_minus == 0:
                        # Increase the capacity by 1
                        neighbourhood_matrix[rand_time, rand_hub] += 1

                        available_hubs = hub_locations.copy()
                        available_hubs.remove(rand_hub)

                        while len(available_hubs) > 0:
                            # Randomly choose a hub to remove a module
                            reinstall_hub = random.choice(available_hubs)
                            if chromosome_matrix[rand_time, reinstall_hub] > highest_originate[rand_time, reinstall_hub]:
                            # The capacity of the chosen hub is greater than the minimum limit
                                neighbourhood_matrix[rand_time, reinstall_hub] -= 1
                                return neighbourhood_matrix
                            else:
                            # The capacity of the chosen hub is 0
                                available_hubs.remove(reinstall_hub)

                        # There is no hub in available_hubs
                        # Cannot reinstall module
                        return neighbourhood_matrix

                    else:
                        # Reduce the capacity by 1
                        neighbourhood_matrix[rand_time, rand_hub] -= 1

                        available_hubs = hub_locations.copy()
                        available_hubs.remove(rand_hub)

                        while len(available_hubs) > 0:
                            # Randomly choose a hub to install a module
                            reinstall_hub = random.choice(available_hubs)
                            if chromosome_matrix[rand_time, reinstall_hub] < 5 and chromosome_matrix[rand_time, reinstall_hub] > 0:
                            # The capacity of the chosen hub has not reached the maximum limit
                                neighbourhood_matrix[rand_time, reinstall_hub] += 1
                                return neighbourhood_matrix
                            else:
                            # The capacity of the chosen hub has reached the maximum limit
                                available_hubs.remove(reinstall_hub)

                        # There is no hub in available_hubs
                        # Cannot reinstall module
                        return neighbourhood_matrix

            elif chosen_capacity == highest_originate[rand_time, rand_hub]:
            # The capacity is equal to the minimum limit
                # Increase the capacity by 1 or reinstall 1 module from another hub in the same time period with 0.5 probability respectively
                plus_or_reinstall = random.choice([0,1])
                neighbourhood_matrix[rand_time, rand_hub] += 1

                if plus_or_reinstall == 0:
                # Increase the capacity by 1
                    return neighbourhood_matrix
                else:
                # Reinstall 1 module from another hub in the same time period
                    available_hubs = hub_locations.copy()
                    available_hubs.remove(rand_hub)

                    while len(available_hubs) > 0:
                        # Randomly choose a hub to remove the module
                        reinstall_hub = random.choice(available_hubs)
                        if chromosome_matrix[rand_time, reinstall_hub] > highest_originate[rand_time, reinstall_hub]:
                        # The capacity of the chosen hub is greater than the minimum limit
                            neighbourhood_matrix[rand_time, reinstall_hub] -= 1
                            return neighbourhood_matrix
                        else:
                        # The capacity of the chosen hub is 0
                            available_hubs.remove(reinstall_hub)

                    # There is no hub in available_hubs
                    # Cannot reinstall module
                    return neighbourhood_matrix

            else:
            # The capacity is 0
                # Choose an entry again
                while (rand_time, rand_hub) in rand_entry:
                # Choose a different entry
                    rand_time = random.choice([0,1,2])
                    rand_hub = random.choice(hub_locations)
                rand_entry.append((rand_time, rand_hub))


def generate_initial_chromosome(pop_size, test_data, prefer_expand):
    '''
    Randomly generate feasible initial population.

    Input:
        pop_size (int): Number of initial chromosomes
        test_data (dictionary): {'distance': ndarray, 'hub_locations': list, 'coefficients': list, 'demand_dict': dictionary,
                                'highest_originate': ndarray, 'module_capacity': float, 'install_hub_cost_matrix': ndarray,
                                'initial_capacity_cost_dict': dictionary, 'additional_capacity_cost_dict': dictionary}
        prefer_expand (boolean): True: Tend to add modules to hubs. False: Tend to reroute flows to hubs that have not yet reached their capacity

    Output:
        initial_chromosome_list (list, length:pop_size): List of initial population

    '''

    hub_locations = test_data['hub_locations']
    highest_originate = test_data['highest_originate']
    N_num = highest_originate.shape[1]

    initial_matrix_list = []
    initial_chromosome_list = []

    for i in range(pop_size):
        feasiblity = False
        while feasiblity == False:
        # Randomly generate matrices until getting a feasible one
            initial_capacity_matrix = rand_chromosome_matrix(N_num, hub_locations, highest_originate)

            repeating_matrix = True
            while repeating_matrix == True:
            # Matrices should be different from each other
                for j in initial_matrix_list:
                    if (initial_capacity_matrix == j).all():
                        initial_capacity_matrix = rand_chromosome_matrix(N_num, hub_locations, highest_originate)
                        break
                repeating_matrix = False

            initial_chromosome = chromosome(initial_capacity_matrix, test_data)
            feasiblity = initial_chromosome.is_feasible()
        initial_chromosome.calculate_fitness(prefer_expand)
        initial_chromosome_list.append(initial_chromosome)
        initial_matrix_list.append(initial_capacity_matrix)

    return initial_chromosome_list


def find_neighbourhood(input_chromosome, iter_times, prefer_expand):
    '''
    Find a chromosome with higher fitness value in the neighbourhood of the input chromosome within limited iteration times.
    If cannot find such chromosome, consider the input_chromosome as the local optimum.

    Input:
        input_chromosome (object: chromosome): A chromosome
        iter_times (int): Iteration times
        prefer_expand (boolean): True: Tend to add modules to hubs. False: Tend to reroute flows to hubs that have not yet reached their capacity

    Output:
        neighbourhood_chromosome (object: chromosome): A chromosome with higher fitness value in the neighbourhood of the input chromosome
        (input_chromosome (object: chromosome): The input chromosome)
    '''

    input_matrix = input_chromosome.initial_capacity_matrix
    input_fitness = input_chromosome.fitness
    test_data = input_chromosome.test_data
    hub_locations = test_data['hub_locations']
    highest_originate = test_data['highest_originate']

    neighbourhood_matrix_list = []

    for i in range(iter_times):
        feasiblity = False
        while feasiblity == False:
        # Randomly generate neighbourhood matrices until getting a feasible one
            neighbourhood_matrix = rand_neighbourhood(input_matrix, hub_locations, highest_originate)

            repeating_matrix = True
            while repeating_matrix == True:
            # Matrices should be different from each other
                for j in neighbourhood_matrix_list:
                    if (neighbourhood_matrix == j).all():
                        neighbourhood_matrix = rand_neighbourhood(input_matrix, hub_locations, highest_originate)
                        break
                repeating_matrix = False

            neighbourhood_chromosome = chromosome(neighbourhood_matrix, test_data)
            feasiblity = neighbourhood_chromosome.is_feasible()

        neighbourhood_matrix_list.append(neighbourhood_matrix)
        neighbourhood_chromosome.calculate_fitness(prefer_expand)
        neighbourhood_matrix_fitness = neighbourhood_chromosome.fitness

        if neighbourhood_matrix_fitness > input_fitness:
            return neighbourhood_chromosome

    return input_chromosome


def local_optimum(input_chromosome, move_times, neighbourhood_numbers, prefer_expand):
    '''
    Within limited move times, find a local optimum by keeping searching and moving to a neighbourhood chromosome
    with higher fitness value.

    Input:
        input_chromosome (object: chromosome): A chromosome
        move_times (int): Maximal move times
        neighbourhood_numbers (int): Maximal number of neighbourhood chromosomes which can be searched for one with higher fitness
        prefer_expand (boolean): True: Tend to add modules to hubs. False: Tend to reroute flows to hubs that have not yet reached their capacity

    Output:
        improved_chromosome (object: chromosome): A local optimum or improved chromosome with higher fitness value than that of the input chromosome
    '''

    initial_chromosome = input_chromosome
    initial_fitness = initial_chromosome.fitness
#    print('initial fitness')
#    print(initial_fitness)

    improved_chromosome = find_neighbourhood(input_chromosome, neighbourhood_numbers, prefer_expand)
    improved_fitness = improved_chromosome.fitness
#    print('improved fitness')
#    print(improved_fitness)

    move = 0
    while improved_fitness != initial_fitness and move < move_times:
    # Stop if local optimum has been found or move time limit has been reached
#        print(move)
        # Move to the neighbourhood chromosome with higher fitness value than that of the initial one
        initial_chromosome = improved_chromosome
        initial_fitness = improved_fitness
#        print('initial fitness')
#        print(initial_fitness)

        improved_chromosome = find_neighbourhood(initial_chromosome, neighbourhood_numbers, prefer_expand)
        improved_fitness = improved_chromosome.fitness
#        print('improved fitness')
#        print(improved_fitness)

        move += 1

    return improved_chromosome
