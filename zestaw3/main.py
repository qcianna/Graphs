import numpy as np

# zad2 Anna Kucia

d = []
p = []


def init(graph, s):
    # czyscimy tablice d i p
    d.clear()
    p.clear()
    # petla po ilosci wierzcholkow
    for _ in range(len(graph)):
        d.append(float('inf'))
        p.append(None)
    d[s] = 0


# u, v - wierzchołki polaczone krawedzia
# we - wagi
def relax(u, v, we):
    weight = w[u][v]
    if w[u][v] is None:
        weight = we[v][u]
    if d[v] > (d[u] + weight):
        d[v] = d[u] + weight
        p[v] = u


# na wejsciu lista sasiedztwa jako graph
# to_print - jesli True to drukujemy wyniki
def dijkstra(graph, we, s, to_print=True):
    init(graph, s)
    S = []
    while len(S) != len(graph):
        u = find_min(S, d)
        S.append(u)
        for v in (graph[u]):
            relax(u, v, we)
    distances = pretty_print(S, d, p, to_print)
    return distances


def find_min(S, dis):
    v_min = 0
    min_distance = float('inf')
    for v in range(len(dis)):
        if v not in S:
            if dis[v] < min_distance:
                v_min = v
                min_distance = dis[v]
    return v_min


# to_print - jesli True to drukujemy wyniki
def pretty_print(S, dis, pp, to_print):
    distances = np.zeros(len(S))

    # sortujemy zeby zawsze miec taka sama strukture tablicy
    for i in sorted(S):
        path = [i]
        next = pp[i]
        while next is not None:
            path.append(next)
            index = next
            next = p[index]
        if to_print:
            print("d(", i, ") = ", dis[i], " --> ", [x for x in reversed(path)])
        distances[i] = dis[i]
    if to_print:
        print("\n")

    return distances


# zad3 Mariusz Marszałek

# funkcja pomocnicza do tworzenia macierzy symetrycznej z macierzy trojkatnej
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


# zadanie 3, przyjmuje graf i tablice wag, zwraca macierz odleglosci
def distance_matrix(graph, we, to_print=True):
    length = len(graph)
    matrix = np.zeros([length, length])
    for i in range(length):
        distances = dijkstra(graph, we, i, False)
        for j in range(i+1, length):
            matrix[i, j] = distances[j]
    matrix = symmetrize(matrix)
    if to_print:
        print("Macierz odleglosci w grafie")
        print(matrix)

    return matrix


# zad4 Mariusz Marszałek

# zadanie 4, przyjmuje graf i tablice wag, zwraca tablice z indeksami centrum (indeks 0) i centrum minimax (indeks 1)
def graph_center(graph, we):
    length = len(graph)
    matrix = distance_matrix(graph, we, False)
    sum_of_weights = np.zeros(length)
    min_weight = float('inf')
    center = -1
    farthest_vertex_distances = np.zeros(length)
    min_minimax = float('inf')
    minimax_center = -1
    print("\nObliczanie centrum")
    for i in range(length):
        sum_of_weights[i] = sum(j for j in matrix[i])
        if sum_of_weights[i] < min_weight:
            min_weight = sum_of_weights[i]
            center = i
    print(sum_of_weights)
    print("\nObliczanie centrum minimax")
    for i in range(length):
        farthest_vertex_distances[i] = max(j for j in matrix[i])
        if farthest_vertex_distances[i] < min_minimax:
            min_minimax = farthest_vertex_distances[i]
            minimax_center = i
    print(farthest_vertex_distances)
    print("\n")
    print("Centrum =", center, "Suma odleglosci:", int(min_weight), "\n")
    print("Centrum minimax =", minimax_center, "Odleglosc od najdalszego:", int(min_minimax))

    return [center, minimax_center]

########################################


if __name__ == "__main__":

    graph = [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2]]
    w = []
    for x in range(len(graph)):
        w.append([None for _ in range(len(graph))])
    w[0][1] = 5
    w[0][2] = 8
    w[1][2] = 9
    w[1][3] = 2
    w[2][3] = 6
    print(w)
    distance_matrix(graph, w)
    graph_center(graph, w)

    # graph2 = [[1, 2], [0, 3], [0, 3],
    #           [1, 2, 4, 5], [3, 5, 6],
    #           [3, 4, 6], [4, 5]]
    #
    # w = []
    # for x in range(len(graph2)):
    #     w.append([None for _ in range(len(graph2))])
    # w[0][1] = 2
    # w[0][2] = 6
    # w[1][3] = 5
    # w[2][3] = 8
    # w[3][4] = 10
    # w[3][5] = 15
    # w[4][5] = 6
    # w[4][6] = 2
    # w[5][6] = 6
    # distance_matrix(graph2, w)
    # print("\n")
    # graph_center(graph2, w)

