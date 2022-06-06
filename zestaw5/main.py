import random
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
# Zad1


def random_flow_network(n: int):
    nodes_in_layer = [1]
    nodes = {0: 0}
    for i in range(n):
        rand = random.randint(2, n)
        nodes_in_layer.append(rand)
        for x in range(rand):
            nodes[max(nodes.keys()) + 1] = len(nodes_in_layer) - 1
    nodes_in_layer.append(1)
    nodes[max(nodes.keys()) + 1] = len(nodes_in_layer) - 1
    all_nodes = sum(nodes_in_layer)

    possibles = [[] for _ in range(n + 1)]
    for i in range(1, all_nodes):
        possibles[nodes[i] - 1].append(i)
    poss = deepcopy(possibles)
    graph = {i: {} for i in range(all_nodes)}
    for i in range(len(possibles[0])):
        choice = random.choice(possibles[0])
        weight = random.randint(1, 10)
        graph[0][choice] = {}
        graph[0][choice]['weight'] = weight
        possibles[0].remove(choice)

    possibles = deepcopy(poss)

    for i in range(n - 1):
        for j in poss[i]:
            weight = random.randint(1, 10)
            choice = random.choice(poss[i + 1])
            graph[j][choice] = {}
            graph[j][choice]['weight'] = weight
            if nodes_in_layer[i + 1] <= nodes_in_layer[i + 2]:
                poss[i + 1].remove(choice)
        poss = deepcopy(possibles)

    for i in possibles[-2]:
        weight = random.randint(1, 10)
        graph[i][all_nodes - 1] = {}
        graph[i][all_nodes - 1]['weight'] = weight

    for i in range(n - 1, 0, -1):
        for j in poss[i]:
            if j not in poss[i - 1]:
                weight = random.randint(1, 10)
                choice = random.choice(poss[i - 1])
                graph[choice][j] = {}
                graph[choice][j]['weight'] = weight

    print(graph)
    for _ in range(2 * n):
        layers = []
        for i in range(n - 1):
            layers.append(i)
        random_layer = random.choice(layers)
        available_nodes = []
        for i in poss[random_layer] + poss[random_layer + 1]:
            available_nodes.append(i)
        choices = random.sample(available_nodes, 2)
        while graph[choices[0]] == choices[1]:
            choices = random.sample(available_nodes, 2)

        weight = random.randint(1, 10)
        graph[choices[0]][choices[1]] = {}
        print("ADDING EDGE FROM ", choices[0], " TO ", choices[1], " WEIGHT OF ", weight)
        graph[choices[0]][choices[1]]['weight'] = weight
        available_nodes.clear()

    print(graph)
    return graph, nodes


def draw_flow_network(graph, nodes, layers):
    G = nx.DiGraph(graph)
    pos = nx.spring_layout(G, iterations=100)
    i = 0
    prev_level = []
    for node in pos:
        level = nodes[node]
        prev_level.append(level)
        if len(prev_level) > 2 and prev_level[-1] != prev_level[-2]:
            i = 0
        pos[node] = (10 * level, i)
        i += 5

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos=pos, node_size=650, node_color='#ffaaaa', with_labels=True)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    plt.show()


if __name__ == '__main__':
    g, n = random_flow_network(4)
    draw_flow_network(g, n, 6)
