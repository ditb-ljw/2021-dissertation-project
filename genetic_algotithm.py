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


def shortest_paths_allocation(hubs, non_hubs, distance, coefficients):
    '''
    Allocate the collection hub and distribution hub for each node following the shortest paths(regardless of capacities).
    Hubs are allocated to themselves as collection hub.

    Input:
        hubs (list): Indices of hub nodes
        non_hubs (list): Indices of non_hub nodes
        distance (ndarray): A matrix containing distance from each node to another node
        coefficients (list, length 3): Collection cost, transfer cost and distribution cost

    Output:
        hub_node_cost (ndarray): The minimal distance from each hub to each node
        hub_node (ndarray): The distribution hub on the shortest path from each hub to each node
        first_hub (ndarray): Collection hub for each flow
        second_hub (ndarray): Distribution hub for each flow
    '''

    X = coefficients[0]
    alpha = coefficients[1]
    delta = coefficients[2]

    # All nodes
    node_number = len(hubs) + len(non_hubs)

    # Calculate the distance from each hub to each node via each distribution hub, and find out the shortest path and the corresponding distribution hub
    hub_node_cost = np.zeros([len(hubs), node_number])
    hub_node = np.zeros([len(hubs), node_number])
    for k_i, k in enumerate(hubs):
        for j in range(node_number):
            cost_node = min((alpha*distance[k,l] + delta*distance[l,j], l) for l in hubs)
            hub_node_cost[k_i,j] = cost_node[0]
            hub_node[k_i,j] = cost_node[1]

    # Calculate the distance from each node to each node via each collection hub, and find out the shortest path and the corresponding collection hub
    # Record the collection hub and distribution hub for each flow in two arraies
    first_hub = np.zeros([node_number, node_number])
    second_hub = np.zeros([node_number, node_number])
    for i in range(node_number):
        for j in range(node_number):
            if i in hubs:
                cost_node = (X*distance[i,i] + hub_node_cost[hubs.index(i),j], i)
            else:
                cost_node = min((X*distance[i,k] + hub_node_cost[k_i,j], k) for k_i, k in enumerate(hubs))
            first_hub[i,j] = cost_node[1]
            second_hub[i,j] = hub_node[hubs.index(cost_node[1]), j]

    return hub_node_cost, hub_node, first_hub, second_hub



def reroute(hubs, non_hubs, demand, first_hub, second_hub, max_hub_capacity, hub_node_cost, hub_node, distance, coefficients):
    '''
    Reroute exceeded flows (randomly chosen) to hubs that have not yet reached their capacity, so the capacity of each hub does not exceed the limit.

    Input:
        hubs (list): Indices of hub nodes
        non_hubs (list): Indices of non_hub nodes
        demand (ndarray): A matrix containing demand from each node to another node
        first_hub (ndarray): Collection hub for each flow(via shortest paths)
        second_hub (ndarray): Distribution hub for each flow(via shortest paths)
        max_hub_capacity (list): Maximal capacity that can be installed in a hub (initial capacity for a new hub)
        hub_node_cost (ndarray): The minimal distance from each hub to each node
        hub_node (ndarray): The distribution hub on the shortest path from each hub to each node
        distance (ndarray): A matrix containing distance from each node to another node
        coefficients (list, length 3): Collection cost, transfer cost and distribution cost

    Output:
        flow (dictionary: {collection hub: [[(origin, collection hub, distribution hub, destination), amount of flow], ... ]}):
            Flow via each collection hub. Including indices for all nodes on the route and amount of flow.
        exceed (list): Amount of flow that exceed the maximal capacity for each hub (negative if below the maximal capacity)
    '''

    X = coefficients[0]
    alpha = coefficients[1]
    delta = coefficients[2]

    # All nodes
    node_number = len(hubs) + len(non_hubs)
    hub_flow = np.zeros(node_number)
    flow = {i:[] for i in hubs}

    # Calculate the total amount of flow via each collection hub, record the route and amount of flow via each collection hub
    for i in range(node_number):
        for j in range(node_number):
            hub_flow[int(first_hub[i,j])] += demand[i,j]
            if demand[i,j] > 0:
                route = (i,int(first_hub[i,j]),int(second_hub[i,j]),j)
                flow[int(first_hub[i,j])].append([route, demand[i,j]])

    # Calculate the amount of flow that exceed the maximal capacity for each hub
    # Get the list of hubs that have not yet reached their capacity and the list of hubs that have reached their capacity
    available_hubs = hubs.copy()
    full_hubs = []
    exceed = np.zeros(node_number)
    for k in hubs:
        exceed[k] = hub_flow[k] - max_hub_capacity[k]
        if exceed[k] >= 0:
            full_hubs.append(k)
            available_hubs.remove(k)

    # For each hubs that have reached their capacity, arbitrarily reroute an excess flow(except the flow originated at the hub)
    # via its cheapest alternative hub that have not yet reached the maximal capacity, until the exceed amount of flow is 0
    for k in full_hubs:
        while exceed[k] > 0:
            # Infeasible
            if available_hubs == []:
                return 0
            reroute_flow_i = random.choice(range(len(flow[k])))
            reroute_flow = flow[k][reroute_flow_i]
            origin = reroute_flow[0][0]
            if origin == k:
                continue
            destination = reroute_flow[0][3]
            reroute_cost_hub = min((X*distance[origin,k] + hub_node_cost[hubs.index(k),destination], k) for k in available_hubs)
            new_route = (origin,reroute_cost_hub[1],int(hub_node[hubs.index(reroute_cost_hub[1]), destination]),destination)
            reroute_flow_value = min(reroute_flow[1], exceed[k])
            if reroute_flow_value == reroute_flow[1]:
                flow[k].pop(reroute_flow_i)
            else:
                reroute_flow[1] -= reroute_flow_value
            flow[reroute_cost_hub[1]].append([new_route, reroute_flow_value])
            exceed[k] -= reroute_flow_value
            exceed[reroute_cost_hub[1]] += reroute_flow_value
            if exceed[reroute_cost_hub[1]] >= 0:
                full_hubs.append(reroute_cost_hub[1])
                available_hubs.remove(reroute_cost_hub[1])

    return flow, exceed


def flow_cost(hubs, non_hubs, flow, hub_node_cost, distance, coefficients):
    '''
    Calculate the total cost for collecting, transferring, and distributing the traffic.

    Input:
        hubs (list): Indices of hub nodes
        non_hubs (list): Indices of non_hub nodes
        flow (dictionary: {collection hub: [[(origin, collection hub, distribution hub, destination), amount of flow], ... ]}):
            Flow via each collection hub. Including indices for all nodes on the route and amount of flow.
        hub_node_cost (ndarray): The minimal distance from each hub to each node
        distance (ndarray): A matrix containing distance from each node to another node
        coefficients (list, length 3): Collection cost, transfer cost and distribution cost

    Output:
        total_cost_flow (float): total cost for collecting, transferring, and distributing the traffic
    '''

    X = coefficients[0]

    total_cost_flow = 0
    for k in hubs:
        for flows in flow[k]:
            origin = flows[0][0]
            hub_1 = flows[0][1]
            destination = flows[0][3]
            total_cost_flow += flows[1]*(X*distance[origin,hub_1] + hub_node_cost[hubs.index(hub_1),destination])

    return total_cost_flow


def additional_capacity(capacity_now, max_hub_capacity, exceed):
    '''
    Calculate the additional capacity for each hub.

    Input:
        capacity_now (list): Capacity that each hub has already installed until now
        max_hub_capacity (list): Maximal capacity that can be installed in a hub (initial capacity for a new hub)
        exceed (list): Amount of flow that exceed the maximal capacity for each hub (negative if below the maximal capacity)

    Output:
        add_capacity (list): Additional capacity needed for each hub
    '''

    # For already built hubs, after allocating flows to each hub,
    # if the total amount of flow is greater than the capacity that has already installed, add the exceeded amount of capacity
    # if the total amount of flow is less than the capacity that has already installed, then additional capacity is not needed.
    add_capacity = []
    for i in range(len(exceed)):
        capacity = max_hub_capacity[i] + exceed[i]
        additional = capacity - capacity_now[i]
        if additional > 0 and capacity_now[i] > 0:
            add_capacity.append(additional)
        else:
            add_capacity.append(0)

    return add_capacity


def hub_capacity_cost(new_hubs, initial_capacity, add_capacity, install_hub_cost, initial_capacity_cost, additional_capacity_cost):
    '''
    Calculate the cost for establishing hubs with the corresponding initial capacities and
    the total cost for installing additional modules at existing hubs

    Input:
        new_hubs (list): Indices for new hubs built in this time period
        initial_capacity (list): Initial capacity for each new hub
        add_capacity (list): Additional capacity needed for each hub
        install_hub_cost (list): Cost for installing each hub
        initial_capacity_cost (ndarray): Cost for building different numbers of initial modules for each hub
        additional_capacity_cost (ndarray): Cost for building different numbers of additional modules for each hub

    Output:
        total_hub_cost (float): The cost for establishing hubs with the corresponding initial capacities and for installing additional modules at existing hubs
    '''

    install_cost = 0
    initial_cost = 0
    for i in new_hubs:
        install_cost += install_hub_cost[i]
        initial_cost += initial_capacity_cost[i, initial_capacity[i]]

    additional_cost = 0
    for j in range(len(add_capacity)):
        additional_cost += additional_capacity_cost[j, int(add_capacity[j])]

    total_hub_cost = install_cost + initial_cost + additional_cost

    return total_hub_cost


def fitness(initial_capacity_matrix, distance, max_capacity, coefficients, demand_dict, scenarios, probabilities, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict):
    '''
    Calculate the fitness value(1/objective value) for a chromosome

    Input:
        initial_capacity_matrix (ndarray): Chromosome: initial capacity(module) of each hub in each period
        distance (ndarray): A matrix containing distance from each node to another node
        max_capacity (list): Maximum number of modules that can be installed in a hub
        coefficients (list, length 3): Collection cost, transfer cost and distribution cost
        demand_dict (dictionary: {scenario: [ndarray, ...]}): Dictionary of demand matrices in each time period for each scenario
        scenarios (list): Different scenarios
        probabilities (dictionary: {scenario: probability}): Probability that each scenario occurs
        module_capacity (float): Capacity of a module
        install_hub_cost_matrix (ndarray): Cost for installing each hub in each time period
        initial_capacity_cost_dict (dictionary: {time_period: nparray, ...}): Cost for building different numbers of initial modules for each node in different time periods
        additional_capacity_cost_dict (dictionary: {time_period: nparray, ...}): Cost for building different numbers of additional modules for each node in different time periods

    Output:
        fitness_value (float): The fitness value(1/objective value) for a chromosome
    '''

    total_cost = 0
    max_capacity = np.array(max_capacity)*module_capacity
    flow_routing = {}
    capacity_expansion = {}

    for s in scenarios:
        flow_routing[s] = []
        capacity_expansion[s] = np.zeros([initial_capacity_matrix.shape[0], initial_capacity_matrix.shape[1]])

        old_hubs = []
        cost_periods = 0
        capacity_now = np.zeros(initial_capacity_matrix.shape[1])

        for t in range(initial_capacity_matrix.shape[0]):
            demand = demand_dict[s][t]
            initial_capacity = initial_capacity_matrix[t,:]*module_capacity
            new_hubs = [k for k in range(len(initial_capacity)) if initial_capacity[k] > 0]

            hubs = old_hubs + new_hubs
            # Infeasible
            if hubs == []:
                return 0

            non_hubs = [i for i in range(initial_capacity_matrix.shape[1]) if i not in hubs]
            max_hub_capacity = max_capacity.copy()
            # The capacity limit for each new hub is its initial capacity because hubs cannot be built and expanded in the same time period
            for k in range(len(new_hubs)):
                max_hub_capacity[new_hubs[k]] = initial_capacity[new_hubs[k]]

            hub_node_cost, hub_node, first_hub, second_hub = shortest_paths_allocation(hubs, non_hubs, distance, coefficients)
            try:
                flow, exceed = reroute(hubs, non_hubs, demand, first_hub, second_hub, max_hub_capacity, hub_node_cost, hub_node, distance, coefficients)
                flow_routing[s].append(flow)
            # Infeasible
            except TypeError:
                return 0
            cost_flow = flow_cost(hubs, non_hubs, flow, hub_node_cost, distance, coefficients)

            for j in new_hubs:
                capacity_now[j] += initial_capacity[j]
            add_capacity = np.ceil(np.array(additional_capacity(capacity_now, max_hub_capacity, exceed))/module_capacity)
            capacity_expansion[s][t,:] = add_capacity

            install_hub_cost = install_hub_cost_matrix[t,:]
            initial_capacity_cost = initial_capacity_cost_dict[t]
            additional_capacity_cost = additional_capacity_cost_dict[t]
            total_hub_cost = hub_capacity_cost(new_hubs, initial_capacity_matrix[t,:], add_capacity, install_hub_cost, initial_capacity_cost, additional_capacity_cost)

            for i in old_hubs:
                capacity_now[i] += add_capacity[i]*module_capacity

            old_hubs += new_hubs
            cost_periods += total_hub_cost + cost_flow

        cost_scenario = probabilities[s]*cost_periods
        total_cost += cost_scenario

    fitness_value = 1/total_cost

    return fitness_value, flow_routing, capacity_expansion



# test
initial_capacity_matrix = np.array([[0, 3, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
#distance = np.array([[0, 5, 8, 6, 10],[6, 0, 7, 8, 12],[8, 6, 0, 4, 9],[9, 6, 5, 0, 6],[11, 10, 8, 7, 0]])
distance = C[:5,:5]
max_capacity = [30, 30, 30, 30, 30]
coefficients = [1,1,1]
#demand_dict = {0: [np.array([[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]), np.array([[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]]), np.array([[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]])], 1:[np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])]}
#scenarios = [0,1]
#probabilities = {0:0.5, 1:0.5}
demand_dict = {0: [W[:5,:5], W[:5,:5], W[:5,:5]]}
scenarios = [0]
probabilities = {0:1}
module_capacity = 100000
install_hub_cost_matrix = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
capacity_cost = np.array([np.arange(31), np.arange(31), np.arange(31), np.arange(31), np.arange(31)])
initial_capacity_cost_dict = {0: capacity_cost, 1: capacity_cost, 2: capacity_cost}
additional_capacity_cost_dict = {0: capacity_cost, 1: capacity_cost, 2: capacity_cost}

fitness(initial_capacity_matrix, distance, max_capacity, coefficients, demand_dict, scenarios, probabilities, module_capacity, install_hub_cost_matrix, initial_capacity_cost_dict, additional_capacity_cost_dict)
