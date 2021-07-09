import numpy as np
import random
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
            if i in hubs:
                cost_node = (X*distance[i,i] + hub_node_cost[i,j], i)
            else:
                cost_node = min((X*distance[i,k] + hub_node_cost[k_i,j], k) for k_i, k in enumerate(hubs))
            node_node_cost[i,j] = cost_node[0]
            first_hub[i,j] = cost_node[1]
            second_hub[i,j] = hub_node[hubs.index(cost_node[1]), j]

    return hub_node_cost, node_node_cost, hub_node, first_hub, second_hub



def reroute(hubs, non_hubs, demand, first_hub, second_hub, max_hub_capacity, hub_node_cost, hub_node, distance, coefficients):

    X = coefficients[0]
    alpha = coefficients[1]
    delta = coefficients[2]

    node_number = len(hubs) + len(non_hubs)
    hub_flow = np.zeros(node_number)
    flow = {i:[] for i in hubs}

    for i in range(node_number):
        for j in range(node_number):
            hub_flow[first_hub[i,j]] += demand[i,j]
            route = (i,first_hub[i,j],second_hub[i,j],j)
#            route_str = ''.join(str(x) for x in route)
#            flow[route_str] = [route, demand[i,j]]
            flow[first_hub[i,j]].append([route, demand[i,j]])

    available_hubs = hubs.copy()
    full_hubs = []
    exceed = np.zeros(node_number)
    for k in hubs:
        exceed[k] = hub_flow[k] - max_hub_capacity[k]
        if exceed[k] >= 0:
            full_hubs.append(k)
            available_hubs.remove(k)

    for k in full_hubs:
        while exceed[k] > 0:
            reroute_flow_i = random.choice(range(len(flow[k])))
            reroute_flow = flow[k][reroute_flow_i]
            origin = reroute_flow[0][0]
            if origin == k:
                continue
            destination = reroute_flow[0][3]
            reroute_cost_hub = min((X*distance[origin,k] + hub_node_cost[hubs.index(k),destination], k) for k in available_hubs)
            new_route = (origin,reroute_cost_hub[1],hub_node[hubs.index(reroute_cost_hub[1]), destination],destination)
            reroute_flow_value = min(reroute_flow[1], exceed[k])
            if reroute_flow_value == reroute_flow[1]:
                flow[k].pop(reroute_flow_i)
            else:
                reroute_flow[1] -= reroute_flow_value
            flow[reroute_cost_hub[1]].append([new_route, reroute_flow_value])
#            hub_flow[k] -= reroute_flow_value
#            hub_flow[reroute_cost_hub[1]] += reroute_flow_value
            exceed[k] -= reroute_flow_value
            exceed[reroute_cost_hub[1]] += reroute_flow_value
            if exceed[reroute_cost_hub[1]] >= 0:
                full_hubs.append(reroute_cost_hub[1])
                available_hubs.remove(reroute_cost_hub[1])

    return flow, exceed


def flow_cost(hubs, non_hubs, flow, hub_node_cost, distance, coefficients):

    X = coefficients[0]

    total_cost_flow = 0
    for k in hubs:
        for flows in flow[k]:
            origin = flow[0][0]
            hub_1 = flow[0][1]
            destination = flow[0][3]
            total_cost_flow += flows[1]*(X*distance[origin,hub_1] + hub_node_cost[hub_1,destination])

    return total_cost_flow


def additional_capacity(capacity_now, max_hub_capacity, exceed):

#    max_capacity_now = max_hub_capacity.copy()
#    for i in new_hubs:
#        max_capacity_now[i] = capacity_now[i]

    add_capacity = []
    for i in range(len(exceed)):
        capacity = max_hub_capacity[i] + exceed[i]
        additional = capacity - capacity_now[i]
        if additional > 0:
            add_capacity.append(additional)
        else:
            add_capacity.append(0)

    return add_capacity


def hub_capacity_cost(new_hubs, initial_capacity, add_capacity, install_hub_cost, initial_capacity_cost, additional_capacity_cost):

    install_cost = 0
    initial_cost = 0
    for i in range(len(new_hubs)):
        install_cost += install_hub_cost[new_hubs[i]]
        initial_cost += initial_capacity_cost[new_hubs[i], initial_capacity[i]]

    additional_cost = 0
    for j in range(len(add_capacity)):
        additional_cost += additional_capacity_cost[j, add_capacity[j]]

    total_hub_cost = install_cost + initial_cost + additional_cost

    return total_hub_cost
