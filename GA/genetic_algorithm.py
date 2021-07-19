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
