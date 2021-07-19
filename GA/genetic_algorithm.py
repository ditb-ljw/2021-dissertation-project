import numpy as np
import random
from GA.Chromosomes import chromosome


def calculate_cumulative_probability(fitness_list):

    total_sum = sum(fitness_list)
    norm_fitness_list = [i/total_sum for i in fitness_list]
    cumulative_probability = np.cumsum(norm_fitness_list)

    return cumulative_probability


def roulette_wheel(cumulative_probability):

    rand_num = random.uniform(0, 1)
