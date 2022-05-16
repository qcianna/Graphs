import numpy as np
import pandas as pd
import random
from copy import deepcopy
from typing import List, Tuple, DefaultDict
from collections import defaultdict

from lab01 import *

# zadania 1, 2 - Mariusz Marszałek


def zad1_graph_series(entry: List[int]) -> bool:
    sequence = deepcopy(entry)
    size = len(sequence)
    sequence.sort(reverse=True)
    while sequence:
        if all(item == 0 for item in sequence):
            return True
        if sequence[0] < 0 or sequence[0] >= size or all(item < 0 for item in sequence[1:size - 1]):
            return False
        for i in range(1, sequence[0]):
            sequence[i] -= 1
        sequence.pop(0)
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
            graph[start_idx].append(end_idx)
            graph[end_idx].append(start_idx)
            sequence[j][0] -= 1  # zmniejszamy pozostały stopień o 1
        sequence[0][0] = 0
        sequence.sort(reverse=True)
    return graph


def zad2_randomise(randomisations_number, entry: DefaultDict) -> DefaultDict:
    # wybieramy wierzchołek startowy krawędzi 1
    # wybieramy wierzchołek końcowy krawędzi 1
    # usuwamy krawędź 1 z listy
    # tak samo dla krawędzi 2
    # dodajemy do grafu nowe, zrandomizowane krawędzie
    graph = deepcopy(entry)
    for i in range(randomisations_number):  # n randomizacji postaci ab-cd -> ad-bc
        try:
            [e1_start, e1_end] = zad2_random_edge(graph)
            [e2_start, e2_end] = zad2_random_edge(graph)

            graph[e1_start[0]].append(e2_end)
            graph[e2_start[0]].append(e1_end)
        except ValueError as e:
            print(str(e))

        #print(graph)
    return graph


def zad2_random_edge(graph):
    start = random.sample(sorted(graph.keys()), 1)
    while len(graph[start[0]]) < 1:  # o ile ma on jakiegoś sąsiada, jeśli nie, wybieramy dalej
        start = random.sample(sorted(graph.keys()), 1)

    end = random.choice(graph[start[0]])
    if all(item == start for item in graph[start[0]]):
        raise ValueError("nie mozna przeprowadzic randomizacji")
    else:
        while end == start:
            end = random.choice(graph[start[0]])
    graph[start[0]].remove(end)

    return [start, end]

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

def zad5_k_regular(n, k):
    graph_sequence = n * [k]
    graph = zad1_adjacency_list(graph_sequence)

    return zad2_randomise(20, graph)


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
    # A = [4, 2, 2, 3, 2, 1, 4, 2, 2, 2, 2]
    # exists = zad1_graph_series(A)
    # if exists:
    #     graf = zad1_adjacency_list(A)
    #     zad2_randomise(5, graf)
    #
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
    # ans = random_k_regular(9)
    # print_in_rows(ans)
    # draw_adjacency_matrix(ans)

    # z5 = zad5_k_regular(9, 4)
    # print(z5)

    #zadanie 6

    # list = [[2, 4, 5], [1, 3, 5, 6], [2, 7, 4], [1, 3, 6, 7], [1, 8, 2], [8, 2, 4], [8, 3, 4], [7, 6, 5]]
    # print(hamilton_R(list, 0, [False for _ in range(len(list))], []))

    #przetestowac na losowym