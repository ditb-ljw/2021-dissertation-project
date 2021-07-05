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

def greedy_allocation(capacities, demands, costs):

    hubs = []
    non_hubs = []
    for i in range(len(capacities)):
        if capacities[i] > 0:
            hubs.append(i)
        else:
            non_hubs.append(i)


    # send to hubs
    allocation = {}
    nodes_to_hubs = {}
    hub_flow = {}
    for i in hubs:
        nodes_to_hubs[i] = []
        hub_flow[i] = -capacities[i]

    # initial allocation
    for i in non_hubs:
        cost_i_hub = costs[i, hubs]
        i_hub = np.argmin(cost_i_hub)
        allocation[i] = {hubs[i_hub]: sum(demands[i,:])}
        nodes_to_hubs[hubs[i_hub]].append(i)

    # adjust to satisfy capacities
    hubs_available = hubs.copy()
    for i in hubs:
        if nodes_to_hubs[i]:
            for j in nodes_to_hubs[i]:
                hub_flow[i] += allocation[j][i]

    for i in hubs:
        hub_unavailable = 1
        while hub_flow[i] > 0:
            if hub_unavailable == 1:
                hubs_available.remove(i)
                hub_unavailable = 0
            if len(hubs_available) == 0:
                print('Infeasible!')
                return 0
            else:
                cost_i_hub_left = costs[nodes_to_hubs[i], :][:, hubs_available] - costs[nodes_to_hubs[i], i][np.newaxis].T
            reallocate = np.argwhere(cost_i_hub_left == np.min(cost_i_hub_left))
            reallocate_node = nodes_to_hubs[i][reallocate[0][0]]
            reallocate_hub = hubs_available[reallocate[0][1]]
#            bisect.insort(nodes_to_hubs[reallocate_hub], reallocate_node)
            nodes_to_hubs[reallocate_hub].append(reallocate_node)

            reallocate_flow = min(hub_flow[i], allocation[reallocate_node][i])
            hub_flow[reallocate_hub] += reallocate_flow
            hub_flow[i] -= reallocate_flow
            allocation[reallocate_node][i] -= reallocate_flow
            allocation[reallocate_node][reallocate_hub] = reallocate_flow

            if allocation[reallocate_node][i] == 0:
                nodes_to_hubs[i].remove(reallocate_node)



    # receive from hubs

    print(allocation)
    print(nodes_to_hubs)
    print(hub_flow)



greedy_allocation([0,0,100000000,0,50000,100000, 0, 0, 0, 20000], W[:10,:10], C[:10,:10])
