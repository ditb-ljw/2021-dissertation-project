import numpy as np
# import bisect

# read all the contents in the text file
file = open('CAB.txt','r')
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
print(W[:10,:10])
print(C[:10,:10])



def shortest_paths_allocation(node_list, distance, coefficients):

    X = coefficients[0]
    alpha = coefficients[1]
    delta = coefficients[2]

    hubs = []
    non_hubs = []
    for i in range(len(node_list)):
        if node_list[i] > 0:
            hubs.append(i)
        else:
            non_hubs.append(i)

    hub_node_cost = np.zeros([len(hubs), len(non_hubs)])
    hub_node = np.zeros([len(hubs), len(non_hubs)])
    for k_i, k in enumerate(hubs):
        for j_i, j in enumerate(non_hubs):
                cost_node = min((alpha*distance[k,l] + delta*distance[l,j], l) for l in hubs)
                hub_node_cost[k_i,j_i] = cost_node[0]
                hub_node[k_i,j_i] = cost_node[1]

    node_node_cost = np.zeros([len(non_hubs), len(non_hubs)])
    node_node = np.zeros([len(non_hubs), len(non_hubs)])
    for i_i, i in enumerate(non_hubs):
        for j_i, j in enumerate(non_hubs):
            cost_node = min((X*distance[i,k] + hub_node_cost[k_i,j_i], k) for k_i, k in enumerate(hubs))
            node_node_cost[i_i,j_i] = cost_node[0]
            node_node[i_i,j_i] = cost_node[1]


    return node_node_cost, node_node, hub_node_cost, hub_node


shortest_paths_allocation([0,0,1,0,1,1, 0, 0, 0, 1], C[:10,:10], [1,1,1])
