import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import math

# Zestaw 4


#pomocnicze
def read_file(filename):
    df = pd.read_csv(filename, header=None)
    array_values = df.values
    array = np.array([[int(value) for value in row[0].split()] for row in array_values])
    return array


def check_input(data):

    if check_if_adj_mat(data):
        return "Adjacency matrix"

    if max(map(max, data)) != 1:
        return "Adjacency list"

    return "Incidence matrix"


def check_if_adj_mat(data):

    size = len(data)
    # sprawdzam czy n x n
    for row in data:
        if size != len(row):
            return False

    # sprawdzam czy 0 na diagonali
    for i in range(size):
        if data[i][i] != 0:
            return False

    return True


def print_in_rows(matrix):
    for row in matrix:
        print(row)


def di_adjacency_matrix_to_adjacency_list(adj_mat):
    n = len(adj_mat)
    adj_list = [[] for i in range(n)]
    for i in range(0, n):
        mat_row = adj_mat[i]
        list_row = adj_list[i]
        for j in range(0, n):
            if mat_row[j] == 1:
                list_row.append(j+1)
    return adj_list


def di_adjacency_list_to_adjacency_matrix(adj_list):
    n = len(adj_list)
    adj_mat = [[0 for i in range(n)] for j in range(n)]
    for i in range(0, n):
        for j in adj_list[i]:
            adj_mat[i][j-1] = 1
    return adj_mat


def di_adjacency_matrix_to_incidence_matrix(matrix):
    number_of_edges = sum(1 if matrix[i][j] == 1 else 0 for i in range(len(matrix)) for j in range(len(matrix)))
    incidence_matrix = [[0 for i in range(number_of_edges)] for j in range(len(matrix))]

    current_edge = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 1:
                incidence_matrix[i][current_edge] = -1
                incidence_matrix[j][current_edge] = 1
                current_edge += 1

    return incidence_matrix


def di_incidence_matrix_to_adjacency_matrix(matrix):
    adjacency_matrix = [[0 for i in range(len(matrix))] for j in range(len(matrix))]

    for j in range(len(matrix[0])):
        w = 0
        h = 0
        for i in range(len(matrix)):
            if matrix[i][j] == 1:
                w = i
            if matrix[i][j] == -1:
                h = i
        adjacency_matrix[h][w] = 1

    return adjacency_matrix


def di_incidence_matrix_to_adjacency_list(matrix):
    tmp = di_incidence_matrix_to_adjacency_matrix(matrix)
    return di_adjacency_matrix_to_adjacency_list(tmp)


def di_adjacency_list_to_incidence_matrix(matrix):
    tmp = di_adjacency_list_to_adjacency_matrix(matrix)
    return di_adjacency_matrix_to_incidence_matrix(tmp)


#zadanie 1
def z1_random_digraph(n, p):
    # zwraca macierz sasiedztwa n x n
    if p > 1 or p < 0:
        print('Niepoprawne prawdopodobienstwo! p >= 0 & p <= 1')
        return False

    adj_mat = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(n):
            if random.uniform(0, 1) < p and i != j:
                adj_mat[i][j] = 1
    return adj_mat


def di_draw_adjacency_matrix(adj_mat, title='Graph', weights_mat=None):
    G = nx.DiGraph()

    [G.add_node(i) for i in range(len(adj_mat))]
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat[i])):
            if adj_mat[i][j] != 0:
                if weights_mat:
                    G.add_edge(i, j, weight=weights_mat[i][j])
                else:
                    G.add_edge(i, j, weight='')

    pos = nx.circular_layout(G)
    plt.title(title)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx(G, pos=pos, arrows=True, connectionstyle='arc3, rad = 0.05')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.2,
                                 font_color='black', font_size=8, font_weight='bold')

    plt.show()


def di_draw_adjacency_list(matrix, title='Graph', weights_mat=None):
    tmp = di_adjacency_list_to_adjacency_matrix(matrix)
    di_draw_adjacency_matrix(tmp, title, weights_mat)


def di_draw_incidence_matrix(matrix, title='Graph', weights_mat=None):
    tmp = di_incidence_matrix_to_adjacency_matrix(matrix)
    di_draw_adjacency_matrix(tmp, title, weights_mat)


def di_draw(matrix, title='Graph', weights_mat=None):
    type_check = check_input(matrix)
    if type_check == "Adjacency list":
        di_draw_adjacency_list(matrix, title, weights_mat)
    elif type_check == "Adjacency matrix":
        di_draw_adjacency_matrix(matrix, title, weights_mat)
    else:
        di_draw_incidence_matrix(matrix, title, weights_mat)

#zadanie 2


def kosaraju(g):
    #zwraca tablice n elementow wskazujacych w ktorym scc jest wierzcholek
    d = [-1 for _ in range(len(g))] #2
    f = [-1 for _ in range(len(g))] #3
    t = 0

    for v in range(len(g)):
        if d[v] == -1:
            d, f, t = dfs_visit(v, g, d, f, t)

    g2 = np.array(g).transpose() # 8
    nr = 0 #9

    comp = [-1 for _ in range(len(g))] #10 #11

    keys = range(len(g))

    d = {keys[i]: f[i] for i in range(len(keys))}
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

    for v in d:
        if comp[v] == -1:
            nr = nr + 1
            comp[v] = nr
            comp = components_r(nr, v, g2, comp)

    return comp


def dfs_visit(v, g, d, f, t):
    t = t + 1
    d[v] = t
    for u in range(len(g[v])):
        if g[v][u] == 1:
            if d[u] == -1:
                d, f, t = dfs_visit(u, g, d, f, t)
    t = t+1
    f[v] = t
    return d, f, t


def components_r(nr, v, g2, comp):
    for u in range(len(g2[v])):
        if g2[v][u] == 1:
            if comp[u] == -1:
                comp[u] = nr
                comp = components_r(nr, u, g2, comp)
    return comp


#zadanie 3

def weights(adj_mat, min=-5, max=40):
    #zwraca tablice n xn, w miejscach gdzie w adj_mat bylo 1 jest waga
    wei_mat = [[0 if adj_mat[j][i] == 0 else random.randint(min, max) for i in range(len(adj_mat))] for j in range(len(adj_mat))]
    return wei_mat


def sc_digraph(n=10, p=0.4):
    #losowanie spojnego digrafu
    scdigraph = z1_random_digraph(n, p)
    scc = kosaraju(scdigraph)
    while len(np.unique(np.array(scc))) != 1:
        scdigraph = z1_random_digraph(n, p)
        scc = kosaraju(scdigraph)
    return scdigraph


def bellman_ford(g, w):
    n = len(g)
    d = [math.inf for _ in range(n)]
    p = [-1 for _ in range(n)]

    d[0] = 0
    for i in range(1, n-1):
        for u in range(n):
            for v in range(n):
                if g[u][v] != 0:
                    if d[v] > d[u] + w[u][v]:
                        d[v] = d[u] + w[u][v]
                        p[v] = u
    for u in range(n):
        for v in range(n):
            if g[u][v] != 0:
                if d[v] > d[u] + w[u][v]:
                    return False

    print("Dystanse: ")
    print(d)
    print("Poprzednicy")
    print(p)
    return True


if __name__ == '__main__':
    print("Zestaw 4!")

    # #zadanie 1
    # graph = z1_random_digraph(7, 1)
    # print_in_rows(graph)
    # di_draw(graph)

    # #zadanie 2
    # graph2 = z1_random_digraph(8, 0.1)
    # print(kosaraju(graph2))
    # di_draw(graph2)

    #zadanie 3
    graph = sc_digraph()
    res = False
    while res == False:
        w = weights(graph)
        res = bellman_ford(graph, w)
    di_draw(graph, weights_mat=w)
