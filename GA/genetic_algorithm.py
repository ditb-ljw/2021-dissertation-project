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
