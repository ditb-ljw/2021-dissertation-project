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


hubs = []
non_hubs = []
for i in range(len(node_list)):
    if node_list[i] > 0:
        hubs.append(i)
    else:
        non_hubs.append(i)

def shortest_paths_allocation(hubs, non_hubs, distance, coefficients):

    X = coefficients[0]
    alpha = coefficients[1]
    delta = coefficients[2]

    node_number = len(hubs) + len(non_hubs)

    hub_node_cost = np.zeros([len(hubs), node_number])
    hub_node = np.zeros([len(hubs), node_number])
    for k_i, k in enumerate(hubs):
        for j in range(node_number):
            cost_node = min((alpha*distance[k,l] + delta*distance[l,j], l) for l in hubs)
            hub_node_cost[k_i,j] = cost_node[0]
            hub_node[k_i,j] = cost_node[1]

    node_node_cost = np.zeros([node_number, node_number])
    first_hub = np.zeros([node_number, node_number])
    second_hub = np.zeros([node_number, node_number])
    for i in range(node_number):
        for j in range(node_number):
            cost_node = min((X*distance[i,k] + hub_node_cost[k_i,j], k) for k_i, k in enumerate(hubs))
            node_node_cost[i,j] = cost_node[0]
            first_hub[i,j] = cost_node[1]
            second_hub[i,j] = hub_node[hubs.index(cost_node[1]), j]

    return hub_node_cost, node_node_cost, first_hub, second_hub


def allocate_capacity(hubs, non_hubs, demand, first_hub):

    node_number = len(hubs) + len(non_hubs)
    hub_capacity = np.zeros(node_number)

    for i in in range(node_number):
        for j in range(node_number):
            hub_capacity[first_hub[i,j]] += demand[i,j]

    return hub_capacity


def additional_capacity_reroute(old_hubs, new_hubs, non_hubs, hub_capacity, max_hub_capacity, hub_capacity_now, hub_node_cost, node_node_cost, first_hub, second_hub):

    node_number = len(old_hubs) + len(new_hubs) + len(non_hubs)
    new_capacity = np.zeros(node_number)

    for i in old_hubs:
        if hub_capacity[i] > hub_capacity_now[i]:
            new_capacity[i] = hub_capacity[i]
        else:
            new_capacity[i] = hub_capacity_now[i]

    



    return
