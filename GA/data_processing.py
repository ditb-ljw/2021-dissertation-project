import numpy as np


# read all the contents in the text file
file = open(r'C:\Users\ljw_l\github\2021-dissertation-project\GA\CAB.txt','r')
contents = file.readlines()
file.close()

# extract the data and save in two matrices
data = contents[28:652]
W = np.zeros((25,25))
C = np.zeros((25,25))
for i in data:
    row_content = list(map(float, i.split()))
    row_i = int(row_content[0]) - 1
    col_j = int(row_content[1]) - 1
    W[row_i, col_j] = row_content[2]
    C[row_i, col_j] = row_content[3]

#
#print(W[:10,:10])
#print(C[:10,:10])



def test_data():
    distance = C[:5,:5]
    max_capacity = [30, 30, 30, 30, 30]
    coefficients = [1,1,1]
    demand_dict = {0: [W[:5,:5], W[:5,:5], W[:5,:5]]}
    scenarios = [0]
    probabilities = {0:1}
    module_capacity = 100000
    install_hub_cost_matrix = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    capacity_cost = np.array([np.arange(31), np.arange(31), np.arange(31), np.arange(31), np.arange(31)])
    initial_capacity_cost_dict = {0: capacity_cost, 1: capacity_cost, 2: capacity_cost}
    additional_capacity_cost_dict = {0: capacity_cost, 1: capacity_cost, 2: capacity_cost}
    return distance, max_capacity, coefficients, demand_dict, scenarios, probabilities, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict
