import numpy as np
import pandas as pd
import random
from copy import deepcopy
from typing import List, Tuple, DefaultDict
from collections import defaultdict, OrderedDict

from lab01 import *

# zadania 1, 2 - Mariusz Marszałek


def zad1_graph_series(entry: List[int]) -> bool:
    sequence = deepcopy(entry)
    size = len(sequence)
    sequence.sort(reverse=True)
    while sequence:
        if all(item == 0 for item in sequence):
            return True
        if sequence[0] < 0 or sequence[0] >= size or any(item < 0 for item in sequence):
            return False
        for i in range(1, sequence[0]):
            sequence[i] -= 1
        sequence[0] = 0
        sequence.sort(reverse=True)


def zad1_adjacency_list(entry: List[int]) -> DefaultDict:
    graph = defaultdict(list)
    sequence = deepcopy(entry)
    sequence.sort(reverse=True)
    sequence = [[sequence[i], i] for i in range(len(sequence))]  # pary postaci stopień, indeks

    for i in range(len(sequence)):
        start_idx = sequence[0][1]
        for j in range(sequence[0][0] + 1):  # 0,0 bo to jest wierzchołek o największym stopniu, bo sortujemy w każdej iteracji
            end_idx = sequence[j][1]
            if start_idx == end_idx:
                continue
            graph[start_idx+1].append(end_idx+1)
            graph[end_idx+1].append(start_idx+1)
            sequence[j][0] -= 1  # zmniejszamy pozostały stopień o 1
        sequence[0][0] = 0
        sequence.sort(reverse=True)
    return graph


def zad2_randomise(randomisations_number, entry: DefaultDict) -> OrderedDict:
    # wybieramy wierzchołek startowy krawędzi 1
    # wybieramy wierzchołek końcowy krawędzi 1
    # usuwamy krawędź 1 z listy
    # tak samo dla krawędzi 2
    # dodajemy do grafu nowe, zrandomizowane krawędzie
    graph = deepcopy(entry)
    i = 0
    while i < randomisations_number:  # n randomizacji postaci ab-cd -> ad-bc
        try:
            [e1_start, e1_end] = zad2_random_edge(graph)
            [e2_start, e2_end] = zad2_random_edge(graph)
            if e2_end not in graph[e1_start] and e2_start not in graph[e1_end] and e1_end not in graph[e2_start] and e1_start not in graph[e2_end] and e2_end != e1_start and e2_end != e1_end and e1_end != e2_start:
                graph[e1_start].append(e2_end)
                graph[e2_end].append(e1_start)
                graph[e2_start].append(e1_end)
                graph[e1_end].append(e2_start)
                i += 1
                print(graph)
            else:
                graph[e1_start].append(e1_end)
                graph[e1_end].append(e1_start)
                graph[e2_start].append(e2_end)
                graph[e2_end].append(e2_start)
        except ValueError as e:
            print(str(e))

    return OrderedDict(sorted(graph.items()))


def zad2_random_edge(graph):
    while True:
        start = random.sample(graph.items(), 1)
        if start[0][1]:
            end = random.sample(start[0][1], 1)
            start = start[0]
            graph[start[0]].remove(end[0])
            graph[end[0]].remove(start[0])

            return [start[0], end[0]]

#zadanie 3 - Monika Kidawska


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


def zad3_cohesive_component(G, silent=False):
    comp = cohesive_component(G)
    if len(comp) == 0:
        print("Z3 : Pusta macierz!")
        return None

    comp_number = max(comp)
    max_length = 0
    max_length_comp = 0

    if not silent:
        for i in range(1, comp_number + 1):
            tmp = []
            print(i, end="")
            for v in range(len(comp)):
                if comp[v] == i:
                    tmp.append(v + 1)
            print(")", *tmp)
            if (len(tmp) > max_length):
                max_length = len(tmp)
                max_length_comp = i

        print("Największa składowa ma numer", max_length_comp)
    return comp_number


#zadanie 5 - Mariusz Marszałek

def dict_to_list(graph):
    list = []
    for key in graph:
        list.append(graph[int(key)])
    return list


def list_to_dict(lst):
    res_dct = {i: lst[i] for i in range(0, len(lst))}
    return res_dct


def zad5_k_regular(n, k):
    graph_sequence = n * [k]
    graph = zad1_adjacency_list(graph_sequence)

    return dict_to_list(zad2_randomise(20, graph))

#zadanie 5 - Tomasz Maczek



def random_k_regular(n):
    if n % 2 == 0:
        k = random.randrange(0, n)
    else:
        k = random.randrange(0, n, 2)

    current_edges = [0 for i in range(n)]
    adjacency_matrix = [[0 for i in range(n)] for j in range(n)]

    j = 0
    for i in range(n):
        while current_edges[i] < k:
            if i != j and adjacency_matrix[j][i] == 0 and current_edges[j] < k:
                adjacency_matrix[j][i] = 1
                adjacency_matrix[i][j] = 1
                current_edges[i] = current_edges[i] + 1
                current_edges[j] = current_edges[j] + 1
            j = j + 1
            if j == n:
                j = 0
    return adjacency_matrix


#zadanie 6 - Anna Kucia

#graph = adj_list


def hamilton_R(graph, v, visited, stack):
    #sprawdzic czy spojny jesli tak-> kontynuowac,
    # jesli nie to nie jest to cykl hamiltona
    if zad3_cohesive_component(adjacency_list_to_adjacency_matrix(graph), silent=True) != 1:
        print('Graf nie jest spojny!')
        return None
    stack.append(v)
    if len(stack) < len(graph):
        visited[v] = True
        for u in graph[v]:
            if visited[u-1] == False:
                cycle = hamilton_R(graph, u-1, visited, stack)
                if cycle:
                    return cycle
        visited[v] = False
        stack.pop()
    else:
        test = False
        for u in graph[v]:
            if u-1 == 0:
                test = True
        cycle = stack
        if test == True:
            print("Cykl Hamiltona:")
            cycle.append(0)
        else:
            print("Ścieżka Hamiltona:")
        return cycle
    return None


if __name__ == '__main__':
    print ("Zestaw 2!")

    #zadania 1, 2 - przyklad
    A = [4,4,4,4,4,4]
    exists = zad1_graph_series(A)
    print(exists)
    # if exists:
    #     graf = zad1_adjacency_list(A)
    #     zad2_randomise(5, graf)

    # B = [1, 3, 2, 3, 2, 1, 1] # przedostatania 1 <- 4 na zajeciach
    # print("Czy B to ciag graficzny: " + str(zad1_graph_series(B)))
    #
    # C = []
    # print("Dla C zwraca : " + str(zad1_graph_series(C)))


    #zadanie 3
    # graph_data = read_file('matrix.txt')
    # draw(graph_data)
    # G = graph_data
    # zad3_cohesive_component(G)
    #
    # graph_data = read_file('matrix0.txt')
    # draw(graph_data)
    # zad3_cohesive_component(graph_data)
    #
    # graph_data = []
    # zad3_cohesive_component(graph_data)



    #zadanie 5

    #wersja Tomasza
    # ans = random_k_regular(9)
    # tmp = list_to_dict(adjacency_matrix_to_adjacency_list(ans))
    # tmp = zad2_randomise(20, tmp)
    # tmp = dict_to_list(tmp)
    #
    # print("In rows:")
    # print_in_rows(tmp)
    # print("Input:")
    # print(tmp)
    # draw(tmp)

    #wersja Mariusza
    z5 = zad5_k_regular(9, 4)
    print(z5)
    draw(z5)
    #zadanie 6

    # list = [[2, 4, 5], [1, 3, 5, 6], [2, 7, 4], [1, 3, 6, 7], [1, 8, 2], [8, 2, 4], [8, 3, 4], [7, 6, 5]]
    # print(hamilton_R(list, 0, [False for _ in range(len(list))], []))

    #przetestowac na losowym