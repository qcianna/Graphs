import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#Z poprzednich zestawów
##########################################################
def draw_adjacency_matrix(adj_mat, weights, index_start=1):

    labels = dict(
        list(enumerate(range(index_start,
                             len(adj_mat) + index_start))))

    g = nx.from_numpy_array(np.array(adj_mat))
    pos = nx.circular_layout(g)
    nx.draw_networkx(g, pos=pos, with_labels=True, labels=labels)
    w = nx.from_numpy_array(np.array(weights))
    labels = nx.get_edge_attributes(w, 'weight')
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=labels)
    plt.show()


def random_n_l(n, l):
    if l > n * (n - 1) / 2:
        print("Za duza liczba krawedzi!")
        return False

    adj_mat = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        adj_mat[i][i] = 2
    edges = 0

    while edges < l and min(map(min, adj_mat)) == 0:
        i, j = np.random.choice(n, size=2, replace=False)
        if adj_mat[i][j] == 0:
            adj_mat[i][j] = 1
            adj_mat[j][i] = 1
            edges += 1

    for i in range(n):
        adj_mat[i][i] = 0
    return adj_mat


def adjacency_matrix_to_adjacency_list(matrix):

    n = len(matrix)
    adj_list = [[] for i in range(n)]
    for i in range(0, n):
        mat_row = matrix[i]
        list_row = adj_list[i]
        for j in range(0, n):
            if mat_row[j] == 1:
                list_row.append(j + 1)
    return adj_list


def cohesive_component(G):
    nr = 0
    comp = [-1 for i in range(len(G))]

    for v in range(len(G)):
        if comp[v] == -1:
            nr = nr + 1
            comp[v] = nr
            cohesive_component_r(nr, v, G, comp)
    return comp


def cohesive_component_r(nr, v, G, comp):
    adj = []
    for i in range(len(G)):
        if G[v][i] == 1:
            adj.append(i)

    for u in adj:
        if comp[u] == -1:
            comp[u] = nr
            cohesive_component_r(nr, u, G, comp)
##########################################################

#Zad.1 Monika Kidawska
          
def generate_cohesive_graph(n, l):
    graph = random_n_l(n, l)

    while max(cohesive_component(graph)) != 1:
        graph = random_n_l(n, l)

    w = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if graph[i][j] == 1 and i<=j:
                w[i][j] = random.randint(0, 10)
            if i>j:
              w[i][j] = w[j][i]

    return graph, w

def print_matrix(matrix, rows):
  print("[")
  for i in range(rows):
      print(matrix[i])
  print("]")

if __name__ == '__main__':
    graph_data = generate_cohesive_graph(12, 18)
    print("Macierz sąsiedztwa:")
    print_matrix(graph_data[0], 12)
    print("Macierz wag:")
    print_matrix(graph_data[1], 12)
    draw_adjacency_matrix(graph_data[0], graph_data[1])
