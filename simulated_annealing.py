import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graph_search_algo
import math
import random

inf = np.inf

# This definition is to compute the cut size of the partition.
def cut_size_compute(adjacency_matrix_upd_kl, partition_1, partition_2, node_list):
    cut_size = 0
    for node_u in partition_1:
        node_u_index = node_list.index(node_u)
        for node_v in partition_2:
            node_v_index = node_list.index(node_v)
            cut_size = cut_size + adjacency_matrix_upd_kl[node_u_index][node_v_index]
    
    return cut_size

# To plot the partitions on the same graph.
def partition_graph_plotter(adjacency_matrix, partition_1, partition_2, node_list):

    G = nx.from_numpy_matrix(adjacency_matrix, create_using = nx.Graph)

    # calculating the angle for positioning the nodes
    count = len(partition_1)
    angle = (2*math.pi)/count
    pos = {}
    for node in partition_1:
        value = partition_1.index(node) + 1
        pos[node_list.index(node)] = (-count - count + (count* math.cos(value*angle)), count*(math.sin(value*angle)))
    for node in partition_2:
        value = partition_2.index(node) + 1
        pos[node_list.index(node)] = (count + count + (count* math.cos(value*angle)), count*(math.sin(value*angle)))
    # print(pos)
    # creating a node list for each partition separately and also removing dmy
    nodes_partition_1 = []
    nodes_partition_2 = []
    nodes_dict = {}

    for i in range(len(partition_1)):
        if partition_1[i] != 'DMY':
            nodes_partition_1.append(node_list.index(partition_1[i]))
            nodes_dict[node_list.index(partition_1[i])] = partition_1[i]
    for i in range(len(partition_2)):
        if partition_2[i] != 'DMY':
            nodes_partition_2.append(node_list.index(partition_2[i]))
            nodes_dict[node_list.index(partition_2[i])] = partition_2[i]
    
    # Adding all the nodes, edges and labels to the graph. Each partition is of different color.
    nx.draw_networkx_nodes(G, pos, nodelist = nodes_partition_1, node_color = 'yellow', node_size = 500)
    nx.draw_networkx_nodes(G, pos, nodelist = nodes_partition_2, node_color = 'skyblue', node_size = 500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, nodes_dict, font_size = 8)
    plt.show()


def init_partition(node_list):
    partition_1 = []
    partition_2 = []
    for node in range(0, len(node_list)):
        if node%2 == 0:
            partition_1.append(node_list[node])
        else:   
            partition_2.append(node_list[node])
    
    return partition_1, partition_2

def partitioning_sim_ann(adjacency_matrix_inf, node_list):
    adjacency_matrix = np.array(adjacency_matrix_inf)
    adjacency_matrix = graph_search_algo.convert_inf_zero(adjacency_matrix)

    partition_1, partition_2 = init_partition(node_list)
    temp = 1000
    rate = 0.95
    frozen_temp = 0.5
    total_attempts = len(node_list) * 100
    cost_list =[]

    cut_size = cut_size_compute(adjacency_matrix, partition_1, partition_2, node_list)
    present_cost = cut_size + ((len(partition_1) - len(partition_2)) ** 2)
    cost_list.append(present_cost)
    best_cost = present_cost
    best_partition_1 = partition_1
    best_partition_2 = partition_2
    print("Partitions and cost before partitioning: ")
    print("Partition 1: ")
    print(best_partition_1)
    print("Partition 2: ")
    print(best_partition_2)
    print("Cut size: ", cut_size)
    print("Cost of the partition: ", best_cost)
    print("No. of nodes in Partition 1: ", len(best_partition_1))
    print("No. of nodes in Partition 2: ", len(best_partition_2))
    partition_graph_plotter(adjacency_matrix, best_partition_1, best_partition_2, node_list)

    temp_failed = 0

    while temp > frozen_temp:
        state_change = 0
        no_of_state_changes = 0
        for i in range(0, total_attempts):
            swap_node = random.choice(node_list)
            if swap_node in partition_1:
                partition_1.remove(swap_node)
                partition_2.append(swap_node)
            else:
                partition_2.remove(swap_node)
                partition_1.append(swap_node)
            cut_size = cut_size_compute(adjacency_matrix, partition_1, partition_2, node_list)
            new_cost = cut_size + ((len(partition_1) - len(partition_2)) ** 2)
            
            delta = new_cost - present_cost
            if delta < 0:
                present_cost = new_cost
                state_change = 1
                no_of_state_changes = no_of_state_changes + 1
                if best_cost > present_cost:
                    best_cost = present_cost
                    best_partition_1 = partition_1
                    best_partition_2 = partition_2
                    cost_list.append(best_cost)
            
            else:
                prob_of_acceptance = random.random()
                prob_of_state_change = (math.e) ** (-delta / temp)
                if prob_of_state_change > prob_of_acceptance:
                    present_cost = new_cost
                    state_change = 1
                    no_of_state_changes = no_of_state_changes + 1
                    if best_cost > present_cost:
                        best_cost = present_cost
                        best_partition_1 = partition_1
                        best_partition_2 = partition_2 
                        cost_list.append(best_cost)
                else:
                    if swap_node in partition_1:
                        partition_1.remove(swap_node)
                        partition_2.append(swap_node)
                    else:
                        partition_2.remove(swap_node)
                        partition_1.append(swap_node)

            if no_of_state_changes == 10:
                break

        if state_change == 1:
            temp_failed = 0
        else:
            temp_failed = temp_failed + 1

        if temp_failed == 3:
            break
        else:
            temp = temp * rate
    
    print("Partitions and cost after partitioning: ")
    print("Partition 1: ")
    print(best_partition_1)
    print("Partition 2: ")
    print(best_partition_2)
    print("Cut size: ", cut_size)
    print("Cost of the partition: ", best_cost)
    print("No. of nodes in Partition 1: ", len(best_partition_1))
    print("No. of nodes in Partition 2: ", len(best_partition_2))
    partition_graph_plotter(adjacency_matrix, best_partition_1, best_partition_2, node_list)





