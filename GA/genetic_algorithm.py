import numpy as np
from GA.Chromosomes import chromosome

initial_capacity_matrix = np.array([[0, 3, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
test_c = chromosome(initial_capacity_matrix)
test_c.if_feasible()
test_c.fitness()
