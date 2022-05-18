# zad2 Anna Kucia

d = []
p = []


def init(graph, s):
    # petla po ilosci wierzcholkow
    for _ in range(len(graph)):
        d.append(float('inf'))
        p.append(None)
    d[s] = 0


#u, v - wierzchołki polaczone krawedzia
#w - wagi
def relax(u, v, w):
    weight = w[u][v]
    if w[u][v] is None:
        weight = w[v][u]
    if d[v] > (d[u] + weight):
        d[v] = d[u] + weight
        p[v] = u

#na wejsciu lista sasiedztwa jako graph
def dijkstra(graph, w, s):
    init(graph, s)
    S = []
    while len(S) != len(graph):
        u = find_min(S, d)
        S.append(u)
        for v in (graph[u]):
            relax(u, v, w)
    pretty_print(S, d, p)


def find_min(S, d):
    v_min = 0
    min_distance = float('inf')
    for v in range(len(d)):
        if v not in S:
            if d[v] < min_distance:
                v_min = v
                min_distance = d[v]
    return v_min


def pretty_print(S, d, p):
    for i in S:
        path = [i]
        next = p[i]
        while next is not None:
            path.append(next)
            index = next
            next = p[index]
        print("d(", i, ") = ", d[i], " --> ", [x for x in reversed(path)])



graph = [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2]]
w = []
for i in range(len(graph)):
    w.append([0 for _ in range(len(graph))])
w[0][1] = 5
w[0][2] = 8
w[1][2] = 9
w[1][3] = 2
w[2][3] = 6
dijkstra(graph, w, 0)

graph2 = [[1, 2], [0, 3], [0, 3],
          [1, 2, 4, 5], [3, 5, 6],
          [3, 4, 6], [4, 5]]

w = []
for i in range(len(graph2)):
    w.append([None for _ in range(len(graph2))])
w[0][1] = 2
w[0][2] = 6
w[1][3] = 5
w[2][3] = 8
w[3][4] = 10
w[3][5] = 15
w[4][5] = 6
w[4][6] = 2
w[5][6] = 6
dijkstra(graph2, w, 0)

