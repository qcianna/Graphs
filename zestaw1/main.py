import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# Zestaw 1

# Zadanie 1


def adjacency_matrix_to_incidence_matrix(matrix):

    number_of_edges = sum(1 if matrix[i][j] == 1 else 0 for i in range(len(matrix) - 1) for j in range(i + 1, len(matrix)))

    incidence_matrix = [[0 for i in range(number_of_edges)] for j in range(len(matrix))]

    current_edge = 0
    for i in range(len(matrix) - 1):
        for j in range(i + 1, len(matrix)):
            if matrix[i][j] == 1:
                incidence_matrix[i][current_edge] = 1
                incidence_matrix[j][current_edge] = 1
                current_edge += 1

    return incidence_matrix


def incidence_matrix_to_adjacency_matrix(matrix):

    adjacency_matrix = [[0 for i in range(len(matrix))] for j in range(len(matrix))]

    w = -1
    h = -1
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            if matrix[i][j] == 1:
                if w == -1:
                    w = i
                else:
                    h = i
        adjacency_matrix[w][h] = 1
        adjacency_matrix[h][w] = 1
        w = -1
        h = -1

    return adjacency_matrix


def adjacency_matrix_to_adjacency_list(matrix):

    n = len(matrix)
    adj_list = [[] for i in range(n)]
    for i in range(0, n):
        mat_row = matrix[i]
        list_row = adj_list[i]
        for j in range(0, n):
            if mat_row[j] == 1:
                list_row.append(j+1)
    return adj_list


def adjacency_list_to_adjacency_matrix(matrix):

    n = len(matrix)
    adj_mat = [[0 for i in range(n)] for j in range(n)]
    for i in range(0, n):
        for j in matrix[i]:
            adj_mat[i][j-1] = 1
    return adj_mat


def incidence_matrix_to_adjacency_list(matrix):
    tmp = incidence_matrix_to_adjacency_matrix(matrix)
    return adjacency_matrix_to_adjacency_list(tmp)


def adjacency_list_to_incidence_matrix(matrix):
    tmp = adjacency_list_to_adjacency_matrix(matrix)
    return adjacency_matrix_to_incidence_matrix(tmp)


def check_input(data):

    if max(map(max, data)) != 1:
        return "Adjacency list"

    if check_if_adj_mat(data):
        return "Adjacency matrix"

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

# zadanie 2


def draw_adjacency_matrix(adj_mat, index_start=0):

    labels = dict(list(enumerate(range(index_start, len(adj_mat) + index_start))))

    g = nx.from_numpy_array(np.array(adj_mat))
    pos = nx.circular_layout(g)
    nx.draw_networkx(g, pos=pos, with_labels=True, labels=labels)
    plt.show()


def draw_adjacency_list(matrix, index_start=0):
    tmp = adjacency_list_to_adjacency_matrix(matrix)
    draw_adjacency_matrix(tmp, index_start)


def draw_incidence_matrix(matrix, index_start=0):
    tmp = incidence_matrix_to_adjacency_matrix(matrix)
    draw_adjacency_matrix(tmp, index_start)


# zadanie 3

def random_n_l(n, l):
    #tworzymy macierz sasiedztwa n na n
    adj_mat = [[0 for i in range(n)] for j in range(n)]
    #wypelniam przekatna 'wartownikami'
    for i in range(n):
        adj_mat[i][i] = n
    edges = 0

    #warunek - dopoki liczba krawedzi i sa jeszcze 0ra
    while edges < l and max(map(min, adj_mat)) == 0:
        #2 rozne liczby w przedziale n
        i, j = np.random.choice(n, size=2, replace=False)
        if adj_mat[i][j] == 0:
            adj_mat[i][j] = 1
            adj_mat[j][i] = 1
            edges += 1

    #usuwam wartownikow
    for i in range(n):
        adj_mat[i][i] = 0
    return adj_mat


def random_n_p(n, p):
    adj_mat = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(i+1, n):
            if random.uniform(0, 1) < p:
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1
    return adj_mat

# pomocnicze


def check_input_val(matrix):
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            if matrix[i][j] != 1 and matrix[i][j] != 0:
                print('Macierz nie powinna zawierać wartości różnych od 0 i 1')


def print_in_rows(matrix):
    for row in matrix:
        print(row)


if __name__ == '__main__':
    graph_data = np.loadtxt('matrix.txt', dtype='int')

    print("This graph data is : " + check_input(graph_data))
    draw_adjacency_matrix(graph_data)

    incidence_data = adjacency_matrix_to_incidence_matrix(graph_data)
    print("Now it's : " + check_input(incidence_data))
    draw_incidence_matrix(incidence_data)







