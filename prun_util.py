import math
import networkx as nx 
import numpy as np

def logistic(x, x_a):
    # Here, since the output is a prob, we let L = 1 and k=1
    return 1 / (1 + math.exp(x_a-x))

def calc_prob_pairwise(G, u, v, time):
    gamma = 1.0
    nbrs = G.predecessors(v)
    intensity = 0.
    f_predecessors = 0
    m_predecessors = 0
    # print(nbrs)
    for nbr in nbrs:
        # print(nbr)
        
        if len(G.nodes[nbr]['lscstate']) > 0:
            # print(G.nodes[nbr]['lscstate'][-1])
            # print(G.nodes[nbr]['lsctime'][-1])
            if G.nodes[nbr]['lscstate'][-1] == 'I':
                f_predecessors += 1
                intensity -= G.nodes[nbr]['intensity_weight'] * math.exp(gamma*(G.nodes[nbr]['lsctime'][-1] - time))
            elif G.nodes[nbr]['lscstate'][-1] == 'R':
                m_predecessors += 1
                intensity += G.nodes[nbr]['intensity_weight'] * math.exp(gamma*(G.nodes[nbr]['lsctime'][-1] - time))
        # print(intensity)
    # print("intensity: ", intensity)
    if intensity < 0:
        return logistic(-intensity, G.nodes[v]['logistic_weight']), False
    elif intensity > 0:
        return logistic(intensity, G.nodes[v]['logistic_weight']), True
    else:
        return 0, None
        
        
def calc_prob(G, adjacency_mat, time):
    gamma = 1.0
    # nbrs = G.predecessors(v)
    
    # adjacency_mat = adjacency_mat.toarray()
    # print(adjacency_mat.shape)
    # print(adjacency_mat.dtype)
    # print(adjacency_mat.flag)
    # print(type(adjacency_mat))

    intensity = np.zeros(len(G.nodes()), dtype=np.float32)
    true_indicator = np.zeros(len(G.nodes()))
    prob = np.zeros(len(G.nodes()))
    for node in G.nodes(): 
        # logistic_weights[node] = G.nodes[node]['logistic_weight']
        if len(G.nodes[node]['lscstate']) > 0:
            if G.nodes[node]['lscstate'][-1] == 'I':
                intensity[node] = -G.nodes[node]['intensity_weight'] * math.exp(gamma*(G.nodes[node]['lsctime'][-1] - time))
            elif G.nodes[node]['lscstate'][-1] == 'R':
                intensity[node] = G.nodes[node]['intensity_weight'] * math.exp(gamma*(G.nodes[node]['lsctime'][-1] - time))
    # print(intensity.flags)
    # default adj mat is propagating, Transpose to receiving
    # received_intensity = np.matmul(adjacency_mat.T, intensity)
    
    # received_intensity = np.matmul(adjacency_mat, intensity)
    received_intensity = adjacency_mat.dot(intensity)
    # print(received_intensity.shape)
    # received_intensity = adjacency_mat.T @ intensity
    
    for node in G.nodes(): 
        if received_intensity[node] < 0:
            prob[node] = logistic(-received_intensity[node], G.nodes[node]['logistic_weight'])
            # true_indicator[node] = 0
        elif received_intensity[node] > 0:
            prob[node] = logistic(received_intensity[node], G.nodes[node]['logistic_weight'])
            true_indicator[node] = 1
        else:
            prob[node] = 0.
            true_indicator[node] = -1
    return prob, true_indicator
    
    # intensity = 0.
    # f_predecessors = 0
    # m_predecessors = 0
    # # print(nbrs)
    # for nbr in nbrs:
        # # print(nbr)
        
        # if len(G.nodes[nbr]['lscstate']) > 0:
            # # print(G.nodes[nbr]['lscstate'][-1])
            # # print(G.nodes[nbr]['lsctime'][-1])
            # if G.nodes[nbr]['lscstate'][-1] == 'I':
                # f_predecessors += 1
                # intensity -= G.nodes[nbr]['intensity_weight'] * math.exp(gamma*(G.nodes[nbr]['lsctime'][-1] - time))
            # elif G.nodes[nbr]['lscstate'][-1] == 'R':
                # m_predecessors += 1
                # intensity += G.nodes[nbr]['intensity_weight'] * math.exp(gamma*(G.nodes[nbr]['lsctime'][-1] - time))
        # print(intensity)
    # print("intensity: ", intensity)
    # if intensity < 0:
        # return logistic(-intensity, G.nodes[v]['logistic_weight']), False
    # elif intensity > 0:
        # return logistic(intensity, G.nodes[v]['logistic_weight']), True
    # else:
        # return 0, None
