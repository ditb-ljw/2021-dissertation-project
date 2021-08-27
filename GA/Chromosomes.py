import numpy as np
import random
from operator import itemgetter


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
    Reroute exceeded flows (randomly chosen) to hubs that have not yet reached their capacity,
    then expand hubs so the capacity of each hub does not exceed the limit.(tend to expand)

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
            reroute_flow_value = min(reroute_flow[1], exceed[k], -exceed[reroute_cost_hub[1]])
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

    exceed = [e if abs(e) > 1e-5 else 0 for e in exceed]

    return flow, exceed


def reroute_min_expand(hubs, non_hubs, demand, first_hub, second_hub, capacity_now, max_hub_capacity, hub_node_cost, hub_node, distance, coefficients, module_capacity):
    '''
    Reroute exceeded flows (randomly chosen) to hubs that have not yet reached their current capacity,
    then expand hubs so the capacity of each hub does not exceed the limit.(tend to reroute)

    Input:
        hubs (list): Indices of hub nodes
        non_hubs (list): Indices of non_hub nodes
        demand (ndarray): A matrix containing demand from each node to another node
        first_hub (ndarray): Collection hub for each flow(via shortest paths)
        second_hub (ndarray): Distribution hub for each flow(via shortest paths)
        capacity_now (list): Capacity that each hub has already installed until now
        max_hub_capacity (list): Maximal capacity that can be installed in a hub (initial capacity for a new hub)
        hub_node_cost (ndarray): The minimal distance from each hub to each node
        hub_node (ndarray): The distribution hub on the shortest path from each hub to each node
        distance (ndarray): A matrix containing distance from each node to another node
        coefficients (list, length 3): Collection cost, transfer cost and distribution cost
        module_capacity (float): Capacity of a module

    Output:
        flow (dictionary: {collection hub: [[(origin, collection hub, distribution hub, destination), amount of flow], ... ]}):
            Flow via each collection hub. Including indices for all nodes on the route and amount of flow.
        exceed_limit (list): Amount of flow that exceed the maximal capacity for each hub (negative if below the maximal capacity)
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

    # Calculate the amount of flow that exceed the current capacity for each hub
    # Get the list of hubs that have not yet reached their capacity and the list of hubs that have reached their capacity
    available_hubs = hubs.copy()
    full_hubs = []
    exceed = np.zeros(node_number)
    for k in hubs:
        exceed[k] = round((hub_flow[k] - capacity_now[k]), 10)
        if exceed[k] >= 0:
            full_hubs.append(k)
            available_hubs.remove(k)

    # For each hubs that have reached their current capacity, arbitrarily reroute an excess flow(except the flow originated at the hub)
    # via its cheapest alternative hub that have not yet reached the current capacity, until all hubs are full
    for k in full_hubs:
        flow_to_hub = flow[k]
        flow_origin = {i:[] for i in range(node_number)}
        for f in flow_to_hub:
            origin_node = f[0][0]
            flow_origin[origin_node].append(f)

        while available_hubs != [] and exceed[k] > 0:
            origin_node_list = [o for o in range(node_number) if o != k and flow_origin[o] != []]
            if origin_node_list == []:
                break
            reroute_origin = random.choice(origin_node_list)
            reroute_flow_i = random.choice(range(len(flow_origin[reroute_origin])))
            reroute_flow = flow_origin[reroute_origin][reroute_flow_i]

            destination = reroute_flow[0][3]
            reroute_cost_hub = min((X*distance[reroute_origin,k] + hub_node_cost[hubs.index(k),destination], k) for k in available_hubs)
            new_route = (reroute_origin, reroute_cost_hub[1], int(hub_node[hubs.index(reroute_cost_hub[1]), destination]), destination)
            reroute_flow_value = min(reroute_flow[1], exceed[k], -exceed[reroute_cost_hub[1]])
            if reroute_flow_value == reroute_flow[1]:
                flow_origin[reroute_origin].pop(reroute_flow_i)
                flow[k].remove(reroute_flow)
            else:
                reroute_flow[1] -= reroute_flow_value
            flow[reroute_cost_hub[1]].append([new_route, reroute_flow_value])
#            flow_origin[reroute_origin].append([new_route, reroute_flow_value])
            exceed[k] -= reroute_flow_value
            exceed[reroute_cost_hub[1]] += reroute_flow_value
            hub_flow[k] -= reroute_flow_value
            hub_flow[reroute_cost_hub[1]] += reroute_flow_value
            if exceed[reroute_cost_hub[1]] >= 0:
                full_hubs.append(reroute_cost_hub[1])
                available_hubs.remove(reroute_cost_hub[1])

    exceed = [round(e,10) if abs(e) > 1e-5 else 0 for e in exceed]
    # There is no need to add modules if current capacity is enough
    if all([i <= 0 for i in exceed]):
        exceed_limit = capacity_now + exceed - max_hub_capacity
        return flow, exceed_limit

#    print('expand')
    # Expand hubs if current capacity is not enough
    # Calculate minimal number of modules needed to cover all the excess flow
    total_exceed = np.sum([round(e,5) for e in exceed if e > 0])
    total_add_module = np.ceil(total_exceed/module_capacity)
#    print('total_add_module')
#    print(total_add_module)

    # Record hubs that do not have enough capacity to deal with their outbound flow
    outbound = np.sum(demand, axis = 1)
    lower_originate =[]
    for i, c in enumerate([np.ceil(capacity/module_capacity) for capacity in capacity_now]):
        if c > 0 and c < outbound[i]:
            lower_originate.append((exceed[i], i))
    # Sort the amount of excess flow for the rest of the hubs in descending order
    sorted_exceed = sorted([(exceed[i], i) for i in range(len(exceed)) if (i not in [j[1] for j in lower_originate]) and (exceed[i] > 0)], reverse = True, key=itemgetter(0))
    sorted_exceed = lower_originate + sorted_exceed
#    print('sorted_exceed')
#    print(sorted_exceed)
    expand_capacity_list = capacity_now.copy()
    rest_add_module = total_add_module

    exceed_left = []
    available_hubs = hubs.copy()
    full_hubs = []
    # Expand hubs
    for e, i in sorted_exceed:
#        print('rest')
#        print(rest_add_module)
        if rest_add_module == 0:
            break
        max_expand = np.ceil(round((max_hub_capacity[i] - capacity_now[i]), 5)/module_capacity)
#        print('max_expand')
#        print(max_expand)
        if max_expand > 0:
            expand_module = min(np.ceil(e/module_capacity), max_expand)
#            print('expand_module')
#            print(expand_module)
#            print(i)
#            print(expand_module)
            expand_capacity = expand_module*module_capacity
            expand_capacity_list[i] += expand_capacity
            rest_add_module -= expand_module
#            print('rest_add_module')
#            print(rest_add_module)
        else:
            expand_capacity = 0
#            print('expand_module')
#            print('0')

        if round(e - expand_capacity, 5) > 0:
            exceed_left.append((round(e - expand_capacity, 5),i))
            full_hubs.append(i)
            available_hubs.remove(i)

#    print(full_hubs)
#    print(available_hubs)
    sorted_exceed_left = sorted(exceed_left, reverse = True, key=itemgetter(0))
#    print('sorted_exceed_left')
#    print(sorted_exceed_left)
    while rest_add_module > 0:
#        print('rest')
#        print(rest_add_module)
        e, i = sorted_exceed_left[0]
        expand_hub = min((distance[i,k], k) for k in available_hubs)[1]
#        print('expand_hub')
#        print(expand_hub)
        max_expand = np.ceil(round((max_hub_capacity[expand_hub] - expand_capacity_list[expand_hub]), 5)/module_capacity)
        expand_module = min(np.ceil(e/module_capacity), max_expand, rest_add_module)
#        print('expand_module')
#        print(expand_module)
        expand_capacity = expand_module*module_capacity

        if round(e - expand_capacity, 5) > 0:
            sorted_exceed_left.append((round(e - expand_capacity, 5),i))
            full_hubs.append(expand_hub)
            available_hubs.remove(expand_hub)
#            print(full_hubs)
#            print(available_hubs)
        expand_capacity_list[expand_hub] += expand_capacity
        rest_add_module -= expand_module
        sorted_exceed_left.pop(0)
        sorted_exceed_left = sorted(sorted_exceed_left, reverse = True, key=itemgetter(0))
#        print('sorted_exceed_left')
#        print(sorted_exceed_left)


    # Reroute after expand
    available_hubs = hubs.copy()
    full_hubs = []
    exceed = np.zeros(node_number)
    for k in hubs:
        exceed[k] = hub_flow[k] - expand_capacity_list[k]
        if exceed[k] >= 0:
            full_hubs.append(k)
            available_hubs.remove(k)
    exceed = [round(e,10) if abs(e) > 1e-5 else 0 for e in exceed]

#    print('exceed')
#    print(exceed)
#    print('full_hubs')
#    print(full_hubs)
#    print('available_hubs')
#    print(available_hubs)
    for k in full_hubs:
        flow_to_hub = flow[k]
        flow_origin = {i:[] for i in range(node_number)}
        for f in flow_to_hub:
            origin_node = f[0][0]
            flow_origin[origin_node].append(f)

        while exceed[k] > 0:
            origin_node_list = [o for o in range(node_number) if o != k and flow_origin[o] != []]
            if origin_node_list == []:
                break
            reroute_origin = random.choice(origin_node_list)
            reroute_flow_i = random.choice(range(len(flow_origin[reroute_origin])))
            reroute_flow = flow_origin[reroute_origin][reroute_flow_i]
#            print('reroute_flow')
#            print(reroute_flow)

            destination = reroute_flow[0][3]
            reroute_cost_hub = min((X*distance[reroute_origin,k] + hub_node_cost[hubs.index(k),destination], k) for k in available_hubs)
            new_route = (reroute_origin, reroute_cost_hub[1], int(hub_node[hubs.index(reroute_cost_hub[1]), destination]), destination)
            reroute_flow_value = min(reroute_flow[1], exceed[k], -exceed[reroute_cost_hub[1]])
#            print('reroute_flow_value')
#            print(reroute_flow_value)
            if reroute_flow_value == reroute_flow[1]:
#                print('1')
                flow_origin[reroute_origin].pop(reroute_flow_i)
                flow[k].remove(reroute_flow)
            else:
#                print('2')
                reroute_flow[1] -= reroute_flow_value
            flow[reroute_cost_hub[1]].append([new_route, reroute_flow_value])
#            flow_origin[reroute_origin].append([new_route, reroute_flow_value])
            exceed[k] -= reroute_flow_value
            exceed[reroute_cost_hub[1]] += reroute_flow_value
            hub_flow[k] -= reroute_flow_value
            hub_flow[reroute_cost_hub[1]] += reroute_flow_value
            if exceed[reroute_cost_hub[1]] >= 0:
                full_hubs.append(reroute_cost_hub[1])
                available_hubs.remove(reroute_cost_hub[1])

            exceed = [round(e,10) if abs(e) > 1e-5 else 0 for e in exceed]
#            print('exceed')
#            print(exceed)

    exceed_limit = expand_capacity_list + exceed - max_hub_capacity
    exceed_limit = [e if abs(e) > 1e-5 else 0 for e in exceed_limit]
    return flow, exceed_limit

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
        additional = round((capacity - capacity_now[i]), 7)
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




class chromosome():
    '''
    Chromosome: a matrix of initial capacity(module) of each hub in each period
    '''

    def __init__(self, initial_capacity_matrix, test_data):
        '''
        Constructor of chromosome.

        Input:
        initial_capacity_matrix (ndarray): Initial capacity(module) of each hub in each period
        test_data (dictionary): {'distance': ndarray, 'hub_locations': list, 'coefficients': list, 'demand_dict': dictionary,
                                'highest_originate': ndarray, 'module_capacity': float, 'install_hub_cost_matrix': ndarray,
                                'initial_capacity_cost_dict': dictionary, 'additional_capacity_cost_dict': dictionary}
            distance (ndarray): A matrix containing distance from each node to another node
            hub_locations (list): Indices of potential locations for the hubs
            coefficients (list, length 3): Collection cost, transfer cost and distribution cost
            demand_dict (dictionary: {scenario: [ndarray, ...]}): Dictionary of demand matrices in each time period for each scenario
            highest_originate (ndarray): Highest total amount of flow originated at each node in each time period
            module_capacity (float): Capacity of a module
            install_hub_cost_matrix (ndarray): Cost for installing each hub in each time period
            initial_capacity_cost_dict (dictionary: {time_period: nparray, ...}): Cost for building and operating different numbers of initial modules for each node in different time periods
            additional_capacity_cost_dict (dictionary: {time_period: nparray, ...}): Cost for building different numbers of additional modules for each node in different time periods

        Attributes:
        initial_capacity_matrix (ndarray): Initial capacity(module) of each hub in each period
        test_data (dictionary): {'distance': ndarray, 'hub_locations': list, 'coefficients': list, 'demand_dict': dictionary,
                                'highest_originate': ndarray, 'module_capacity': float, 'install_hub_cost_matrix': ndarray,
                                'initial_capacity_cost_dict': dictionary, 'additional_capacity_cost_dict': dictionary}
        fitness (float): The fitness value(1/objective value) for a chromosome
        flow_routing (dictionary: {scenario:[flow in period 0, ...], ...}): The route and amount of flow in each time period for each scenario
        capacity_expansion (dictionary: {scenario: ndarray, ...}): Additional modules for each hub in each time period for each scenario
        '''

        self.initial_capacity_matrix = initial_capacity_matrix
        self.test_data = test_data

        self.fitness = 0
        self.flow_routing = {}
        self.capacity_expansion = {}


    def calculate_fitness(self, prefer_expand):
        '''
        Calculate the fitness value(1/objective value) and get the corresponding flow routing and additional modules for a chromosome

        Input:
            prefer_expand (boolean): True: Tend to add modules to hubs. False: Tend to reroute flows to hubs that have not yet reached their capacity.

        Data:
            initial_capacity_matrix (ndarray): Initial capacity(module) of each hub in each period
            distance (ndarray): A matrix containing distance from each node to another node
            max_capacity (list): Maximum number of modules that can be installed in a hub
            coefficients (list, length 3): Collection cost, transfer cost and distribution cost
            demand_dict (dictionary: {scenario: [ndarray, ...]}): Dictionary of demand matrices in each time period for each scenario
            scenarios (list): Different scenarios
            probabilities (dictionary: {scenario: probability}): Probability that each scenario occurs
            module_capacity (float): Capacity of a module
            install_hub_cost_matrix (ndarray): Cost for installing each hub in each time period
            initial_capacity_cost_dict (dictionary: {time_period: nparray, ...}): Cost for building and operating different numbers of initial modules for each node in different time periods
            additional_capacity_cost_dict (dictionary: {time_period: nparray, ...}): Cost for building different numbers of additional modules for each node in different time periods
        '''

        test_data = self.test_data

        if self.is_feasible() == False:
            self.fitness = 0
            return None

        distance = test_data['distance']
        hub_locations = test_data['hub_locations']
        node_number = self.initial_capacity_matrix.shape[1]
        max_capacity = [5 if i in hub_locations else 0 for i in range(node_number)]
        coefficients = test_data['coefficients']
        demand_dict = test_data['demand_dict']
        highest_originate = test_data['highest_originate']
        scenarios = [0, 1, 2, 3, 4]
        probabilities = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
        module_capacity = test_data['module_capacity']
        install_hub_cost_matrix = test_data['install_hub_cost_matrix']
        initial_capacity_cost_dict = test_data['initial_capacity_cost_dict']
        additional_capacity_cost_dict = test_data['additional_capacity_cost_dict']

        total_cost = 0
        maximal_capacity = max_capacity.copy()
        maximal_capacity = np.array(maximal_capacity)*module_capacity
        flow_routing = {}
        capacity_expansion = {}

        for s in scenarios:
            flow_routing[s] = []
            capacity_expansion[s] = np.zeros([self.initial_capacity_matrix.shape[0], self.initial_capacity_matrix.shape[1]])

            old_hubs = []
            cost_periods = 0
            capacity_now = np.zeros(self.initial_capacity_matrix.shape[1])

            for t in range(self.initial_capacity_matrix.shape[0]):
                demand = demand_dict[s][t]
                initial_capacity = self.initial_capacity_matrix[t,:]*module_capacity
                new_hubs = [k for k in range(len(initial_capacity)) if initial_capacity[k] > 0]

                hubs = old_hubs + new_hubs
                # Infeasible
                if hubs == []:
                    self.fitness = 0
                    return None

                non_hubs = [i for i in range(self.initial_capacity_matrix.shape[1]) if i not in hubs]
                max_hub_capacity = np.zeros(self.initial_capacity_matrix.shape[1])
                # The capacity limit for each new hub is its initial capacity because hubs cannot be built and expanded in the same time period
                for k in hubs:
                    if k in new_hubs:
                        max_hub_capacity[k] = initial_capacity[k]
                    else:
                        max_hub_capacity[k] = maximal_capacity[k]

                hub_node_cost, hub_node, first_hub, second_hub = shortest_paths_allocation(hubs, non_hubs, distance, coefficients)
                for j in new_hubs:
                    capacity_now[j] += initial_capacity[j]
                try:
                    if prefer_expand == True:
                        flow, exceed = reroute(hubs, non_hubs, demand, first_hub, second_hub, max_hub_capacity, hub_node_cost, hub_node, distance, coefficients)
                    else:
                        flow, exceed = reroute_min_expand(hubs, non_hubs, demand, first_hub, second_hub, capacity_now, max_hub_capacity, hub_node_cost, hub_node, distance, coefficients, module_capacity)

                    flow_routing[s].append(flow)
                # Infeasible
                except TypeError:
                    self.fitness = 0
                    return None
                cost_flow = flow_cost(hubs, non_hubs, flow, hub_node_cost, distance, coefficients)

                add_capacity = np.ceil(np.array(additional_capacity(capacity_now, max_hub_capacity, exceed))/module_capacity)
                capacity_expansion[s][t,:] = add_capacity

                install_hub_cost = install_hub_cost_matrix[t,:]
                initial_capacity_cost = initial_capacity_cost_dict[t]
                additional_capacity_cost = additional_capacity_cost_dict[t]
                total_hub_cost = hub_capacity_cost(new_hubs, self.initial_capacity_matrix[t,:], add_capacity, install_hub_cost, initial_capacity_cost, additional_capacity_cost)

                for i in old_hubs:
                    capacity_now[i] += add_capacity[i]*module_capacity

                old_hubs += new_hubs
                cost_periods += total_hub_cost + cost_flow

            cost_scenario = probabilities[s]*cost_periods
            total_cost += cost_scenario

        fitness_value = 1/total_cost

        self.fitness = fitness_value
        self.flow_routing = flow_routing
        self.capacity_expansion = capacity_expansion

        return None


    def is_feasible(self):
        '''
        Check if the chromosome is feasible.

        Data:
            initial_capacity_matrix (ndarray): initial capacity(module) of each hub in each period
            hub_locations (list): Indices of potential locations for the hubs
            demand_dict (dictionary: {scenario: [ndarray, ...]}): Dictionary of demand matrices in each time period for each scenario
            highest_originate (ndarray): Highest total amount of flow originated at each node in each time period
            scenarios (list): Different scenarios
            max_capacity (list): Maximum number of modules that can be installed in a hub
            module_capacity (float): Capacity of a module

        Output:
            boolean: True if feasible, False if infeasible
        '''

        test_data = self.test_data

        hub_locations = test_data['hub_locations']
        node_number = self.initial_capacity_matrix.shape[1]
        max_capacity = [5 if i in hub_locations else 0 for i in range(node_number)]
        demand_dict = test_data['demand_dict']
        highest_originate = test_data['highest_originate']
        scenarios = [0, 1, 2, 3, 4]
        module_capacity = test_data['module_capacity']

        for j in range(self.initial_capacity_matrix.shape[1]):
            positive_entries = 0
            for i in range(self.initial_capacity_matrix.shape[0]):
                if self.initial_capacity_matrix[i, j] > 0:
                    positive_entries += 1
                # There must be at most one positive number in each column
                if positive_entries > 1:
                    return False
                # Only nodes in the list of potential locations for hubs can become hubs
                if j not in hub_locations and positive_entries > 0:
                    return False

        # The total initial capacity in each period(0 if no new hub is built) should be greater than or equal to
        # the highest total demand in that period – the total maximal capacity of already built hubs
        max_demand = []
        total_maximal_capacity = []
        total_initial_capacity = []
        total_max_capacity = 0
        for t in range(self.initial_capacity_matrix.shape[0]):
            max_demand_period = max(np.sum(demand_dict[s][t]) for s in scenarios)
            max_demand.append(max_demand_period)

            total_initial_capacity.append(np.sum(self.initial_capacity_matrix[t,:]))

            new_hubs = [k for k in range(self.initial_capacity_matrix.shape[1]) if self.initial_capacity_matrix[t,k] > 0]
            total_max_capacity += sum([max_capacity[c] for c in new_hubs])
            total_maximal_capacity.append(total_max_capacity)

        total_maximal_capacity.pop()
        total_maximal_capacity = [0] + total_maximal_capacity

        for i in range(len(total_initial_capacity)):
            if total_initial_capacity[i]*module_capacity < max_demand[i] - total_maximal_capacity[i]*module_capacity:
                return False

        # The initial capacity for each hub in each period should be greater or equal to the highest total amount of flow originated at the hub
        for t in range(self.initial_capacity_matrix.shape[0]):
            for k in hub_locations:
                initial_capacity_k_t = self.initial_capacity_matrix[t, k]
                if initial_capacity_k_t == 0:
                    continue
                elif initial_capacity_k_t > 5 or initial_capacity_k_t < highest_originate[t, k]:
                    return False

        return True
