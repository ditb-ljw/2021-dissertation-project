import numpy as np
import random
import bisect
from GA.Chromosomes import chromosome


def calculate_cumulative_probability(fitness_list):
    '''
    Calculate the cumulative probability for each chromosome.

    Input:
        fitness_list (list): Fitness value for each chromosome

    Output:
        cumulative_probability (list): Cumulative probability for each chromosome
    '''

    total_sum = sum(fitness_list)
    norm_fitness_list = [i/total_sum for i in fitness_list]
    cumulative_probability = np.cumsum(norm_fitness_list)

    return cumulative_probability


def roulette_wheel(cumulative_probability):
    '''
    Select a chromosome by roulette wheel selection.

    Input:
        cumulative_probability (list): Cumulative probability for each chromosome

    Output:
        chromosome_index (int): Index of selected chromosome
    '''

    rand_num = random.uniform(0, 1)
    chromosome_index = bisect.bisect_left(cumulative_probability, rand_num)

    return chromosome_index


def one_pt_col_crossover(chromosome_1, chromosome_2):
    '''
    One-point crossover(column): Choose a random point on the two parents, split parents at this crossover point and create children by exchanging tails (columns).

    Input:
        chromosome_1 (object: chromosome): One of the parent chromosomes
        chromosome_2 (object: chromosome): One of the parent chromosomes

    Output:
        child_chromosome_1 (object: chromosome): One of the child chromosomes
        child_chromosome_2 (object: chromosome): One of the child chromosomes
    '''

    parent_matrix_1 = chromosome_1.initial_capacity_matrix
    parent_matrix_2 = chromosome_2.initial_capacity_matrix
    node_number = parent_matrix_1.shape[1]

    child_matrix_1 = np.zeros([3, node_number], dtype=int)
    child_matrix_2 = np.zeros([3, node_number], dtype=int)

    # Randomly choose a crossover point
    split_pt = random.choice(range(1, node_number))

    child_matrix_1[:, :split_pt] = parent_matrix_1[:, :split_pt]
    child_matrix_1[:, split_pt:] = parent_matrix_2[:, split_pt:]
    child_matrix_2[:, :split_pt] = parent_matrix_2[:, :split_pt]
    child_matrix_2[:, split_pt:] = parent_matrix_1[:, split_pt:]

    test_data = chromosome_1.test_data
    child_chromosome_1 = chromosome(child_matrix_1, test_data)
    child_chromosome_2 = chromosome(child_matrix_2, test_data)

    return child_chromosome_1, child_chromosome_2


def uniform_col_crossover(chromosome_1, chromosome_2):
    '''
    Uniform crossover(column): Assign 'heads' to one parent, 'tails' to the other,
    Flip a coin for each column of the first child,
    Make an inverse copy of the column for the second child.

    Input:
        chromosome_1 (object: chromosome): One of the parent chromosomes
        chromosome_2 (object: chromosome): One of the parent chromosomes

    Output:
        child_chromosome_1 (object: chromosome): One of the child chromosomes
        child_chromosome_2 (object: chromosome): One of the child chromosomes
    '''

    parent_matrix_1 = chromosome_1.initial_capacity_matrix
    parent_matrix_2 = chromosome_2.initial_capacity_matrix
    node_number = parent_matrix_1.shape[1]

    child_matrix_1 = np.zeros([3, node_number], dtype=int)
    child_matrix_2 = np.zeros([3, node_number], dtype=int)

    for i in range(node_number):
        head_or_tail = random.choice([0,1])
        if head_or_tail == 0:
            child_matrix_1[:, i] = parent_matrix_1[:, i]
            child_matrix_2[:, i] = parent_matrix_2[:, i]
        else:
            child_matrix_1[:, i] = parent_matrix_2[:, i]
            child_matrix_2[:, i] = parent_matrix_1[:, i]

    test_data = chromosome_1.test_data
    child_chromosome_1 = chromosome(child_matrix_1, test_data)
    child_chromosome_2 = chromosome(child_matrix_2, test_data)

    return child_chromosome_1, child_chromosome_2


def one_pt_row_crossover(chromosome_1, chromosome_2):
    '''
    One-point crossover(row): Choose a random point on the two parents, split parents at this crossover point and create children by exchanging tails (rows).
    If there are more than one positive numbers on one column, set the last non-zero number as the maximal number among them and set other numbers to zero.

    Input:
        chromosome_1 (object: chromosome): One of the parent chromosomes
        chromosome_2 (object: chromosome): One of the parent chromosomes

    Output:
        child_chromosome_1 (object: chromosome): One of the child chromosomes
        child_chromosome_2 (object: chromosome): One of the child chromosomes
    '''

    parent_matrix_1 = chromosome_1.initial_capacity_matrix
    parent_matrix_2 = chromosome_2.initial_capacity_matrix
    node_number = parent_matrix_1.shape[1]

    child_matrix_1 = np.zeros([3, node_number], dtype=int)
    child_matrix_2 = np.zeros([3, node_number], dtype=int)

    # Randomly choose a crossover point
    split_pt = random.choice(range(1, 3))

    child_matrix_1[:split_pt, :] = parent_matrix_1[:split_pt, :]
    child_matrix_1[split_pt:, :] = parent_matrix_2[split_pt:, :]
    child_matrix_2[:split_pt, :] = parent_matrix_2[:split_pt, :]
    child_matrix_2[split_pt:, :] = parent_matrix_1[split_pt:, :]

    # Adjust child chromosome 1 if there are more than one positive numbers in one column
    for i in range(node_number):
        positive_values = []
        for t in [2,1,0]:
            if child_matrix_1[t, i] > 0:
                positive_values.append((child_matrix_1[t, i], t))

        if len(positive_values) > 1:
            time = positive_values[0][1]
            modules = max(positive_values)[0]
            for j in positive_values:
                if j[1] == time:
                    child_matrix_1[time, i] = modules
                else:
                    child_matrix_1[j[1], i] = 0

    # Adjust child chromosome 2 if there are more than one positive numbers in one column
    for i in range(node_number):
        positive_values = []
        for t in [2,1,0]:
            if child_matrix_2[t, i] > 0:
                positive_values.append((child_matrix_2[t, i], t))

        if len(positive_values) > 1:
            time = positive_values[0][1]
            modules = max(positive_values)[0]
            for j in positive_values:
                if j[1] == time:
                    child_matrix_2[time, i] = modules
                else:
                    child_matrix_2[j[1], i] = 0


    test_data = chromosome_1.test_data
    child_chromosome_1 = chromosome(child_matrix_1, test_data)
    child_chromosome_2 = chromosome(child_matrix_2, test_data)

    return child_chromosome_1, child_chromosome_2


def mutation(input_chromosome):
    '''
    Randomly adjust an entry of the matrix of the input chromosome.

    Input:
        input_chromosome (object: chromosome): A chromosome chosen to be mutated

    Output:
        mutated_chromosome (object: chromosome): The chromosome after being mutated
    '''

    input_matrix = input_chromosome.initial_capacity_matrix
    node_number = input_matrix.shape[1]
    test_data = input_chromosome.test_data
    hub_locations = test_data['hub_locations']
    highest_originate = test_data['highest_originate']

    feasible = False
    while feasible == False:
    # Generate a mutated matrix until it is feasible
        mutated_matrix = input_matrix.copy()

        # Randomly choose an entry to make change
        rand_time = random.choice([0,1,2])
        rand_hub = random.choice(hub_locations)

        chosen_value = input_matrix[rand_time, rand_hub]
        if chosen_value == 5:
        # The value has reached its maximum so it should be reduced
            rand_modules = random.choice(range(highest_originate[rand_time, rand_hub], 5))
            mutated_matrix[rand_time, rand_hub] = rand_modules
        elif chosen_value == highest_originate[rand_time, rand_hub]:
        # The value has reached its minimum so it should be increased
            rand_modules = random.choice(range(highest_originate[rand_time, rand_hub] + 1, 6))
            mutated_matrix[rand_time, rand_hub] = rand_modules
        else:
        # The value is 0 or between its maximum and minimum
            module_list = [x for x in range(highest_originate[rand_time, rand_hub], 6) if x != chosen_value]
            rand_modules = random.choice(module_list)
            mutated_matrix[rand_time, rand_hub] = rand_modules

        mutated_chromosome = chromosome(mutated_matrix, test_data)
        feasible = mutated_chromosome.is_feasible()

    return mutated_chromosome


def new_generation(old_generation, one_pt_col_crossover_probability, uniform_col_crossover_probability, one_pt_row_crossover_probability, mutation_probability, prefer_expand):
    '''
    Generate new generation by crossover and mutation based on old generation.

    Input:
        old_generation (list): List of chromosomes of old generation(sorted by their fitness value in decending order)
        one_pt_col_crossover_probability (float): Probability of 1-point crossover(column)
        uniform_col_crossover_probability (float): Probability of uniform crossover(column)
        one_pt_row_crossover_probability (float): Probability of 1-point crossover(row)
        mutation_probability (float): Probability of mutation
        prefer_expand (boolean): True: Tend to add modules to hubs. False: Tend to reroute flows to hubs that have not yet reached their capacity

    Output:
        new_generation (list): List of chromosomes of new generation(sorted by their fitness value in decending order)
    '''

    population_number = len(old_generation)
    fitness_list = [c.fitness for c in old_generation]
    cumulative_probability = calculate_cumulative_probability(fitness_list)
    new_generation_list = []
    new_generation_fitness = []
    new_generation_matrices = []

    new_generation_length = 0
    while new_generation_length < population_number:
        # Select different parents
        parent_index_1 = roulette_wheel(cumulative_probability)
        parent_index_2 = roulette_wheel(cumulative_probability)
        while parent_index_2 == parent_index_1:
            parent_index_2 = roulette_wheel(cumulative_probability)

        parent_chromosome_1 = old_generation[parent_index_1]
        parent_chromosome_2 = old_generation[parent_index_2]

        # Crossover
        rand_num = random.uniform(0, 1)
        if rand_num <= one_pt_col_crossover_probability:
            # 1-point crossover(column)
            child_chromosome_1, child_chromosome_2 = one_pt_col_crossover(parent_chromosome_1, parent_chromosome_2)
        elif rand_num <= one_pt_col_crossover_probability + uniform_col_crossover_probability:
            # Uniform crossover(column)
            child_chromosome_1, child_chromosome_2 = uniform_col_crossover(parent_chromosome_1, parent_chromosome_2)
        elif rand_num <= one_pt_col_crossover_probability + uniform_col_crossover_probability + one_pt_row_crossover_probability:
            # 1-point crossover(row)
            child_chromosome_1, child_chromosome_2 = one_pt_row_crossover(parent_chromosome_1, parent_chromosome_2)
        else:
            child_chromosome_1, child_chromosome_2 = parent_chromosome_1, parent_chromosome_2
        # Check
        for child_chromosome in [child_chromosome_1, child_chromosome_2]:
            repeating_matrix = False
            # New chromosomes should be different from each other
            for j in new_generation_matrices:
                if (child_chromosome.initial_capacity_matrix == j).all():
                    repeating_matrix = True
                    break
            # Check feasibility
            if child_chromosome.is_feasible() == True and repeating_matrix == False:
                # Mutation
                rand_num = random.uniform(0, 1)
                if rand_num <= mutation_probability:
                    child_chromosome = mutation(child_chromosome)

                child_chromosome.calculate_fitness(prefer_expand)
                new_generation_list.append(child_chromosome)
                new_generation_fitness.append(child_chromosome.fitness)
                new_generation_matrices.append(child_chromosome.initial_capacity_matrix)
                new_generation_length += 1


    # Elitism
    best_matrix = old_generation[0].initial_capacity_matrix
    best_matrix_kept = False
    for j in new_generation_matrices:
        if (best_matrix == j).all():
            best_matrix_kept = True
            break
    if best_matrix_kept == False:
        new_generation_list.append(old_generation[0])
        new_generation_fitness.append(old_generation[0].fitness)
        new_generation_matrices.append(best_matrix)

    # Sort in decending order
    zipped_lists = zip(new_generation_fitness, new_generation_list)
    sorted_pairs = sorted(zipped_lists, reverse = True)
    # Choose a fixed number of best chromosomes
    new_generation = [pair[1] for pair in sorted_pairs[:population_number]]

    return new_generation


def GA(initial_population, num_generation, num_unchange, one_pt_col_crossover_probability, uniform_col_crossover_probability, one_pt_row_crossover_probability, mutation_probability, prefer_expand):
    '''
    Genetic algorithm.

    Input:
        initial_population (list): List of chromosomes of initial generation(unsorted)
        num_generation (int): Total number of generations
        num_unchange (int): Successive max number of generations where the best fitness value keeps (almost) unchanged
        one_pt_col_crossover_probability (float): Probability of 1-point crossover(column)
        uniform_col_crossover_probability (float): Probability of uniform crossover(column)
        one_pt_row_crossover_probability (float): Probability of 1-point crossover(row)
        mutation_probability (float): Probability of mutation
        prefer_expand (boolean): True: Tend to add modules to hubs. False: Tend to reroute flows to hubs that have not yet reached their capacity

    Output:
        new_generation_list[0] (object: chromosome): The chromosome with best fitness value
    '''

    old_generation_list = initial_population
    # Sort in decending order
    old_generation_fitness = [c.fitness for c in old_generation_list]
    zipped_lists = zip(old_generation_fitness, old_generation_list)
    sorted_pairs = sorted(zipped_lists, reverse = True)

    old_generation = [pair[1] for pair in sorted_pairs]
    print(old_generation[0].fitness)

    last_best_fitness = 0
    unchange = 0
    for g in range(num_generation):
        new_generation_list = new_generation(old_generation, one_pt_col_crossover_probability, uniform_col_crossover_probability, one_pt_row_crossover_probability, mutation_probability, prefer_expand)
        best_fitness = new_generation_list[0].fitness
        if abs(best_fitness - last_best_fitness) < 1e-5:
            unchange += 1
        else:
            unchange = 0

        if unchange == num_unchange:
            return new_generation_list[0]

        last_best_fitness = best_fitness
        old_generation = new_generation_list

    return new_generation_list[0]
