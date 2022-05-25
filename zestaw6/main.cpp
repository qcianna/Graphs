#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#define N 1000000

bool contains(std::vector<std::vector<int>> graph, int current, int node) {

    for(int j=0; j<graph[current].size(); j++) {
            if(graph[current][j] == node) {
            return true;
        }
    }
    return false;
}

void prettyPrint(std::vector<int> tab) {

    for(int i=0; i<tab.size(); i++) {
        std::cout << i << " --> " << (double)tab[i]/N << std::endl;
    }
}

void pageRankRandomWalk(std::vector<std::vector<int>> graph, int d) {

    srand(time(0));

    int size = graph.size();
    std::vector<int> visited;
    for(int i=0; i<size; i++) {
        visited.push_back(0);
    }

    int current = 0;

    int counter = 0;
    for(int i=0; i<N; i++) {
        double randomChoice = (double)rand()/RAND_MAX;
        if(randomChoice > d) {
            while(true) {
                int newNode = rand() % size;
                if(contains(graph, current, newNode)) {
                    counter++;
                    current = newNode;
                    visited[current]++;
                    break;
                }
            }
        } else {
            current = rand() % size;
            visited[current]++;
        }
    }

    prettyPrint(visited);
}

int main() {

    int A = 0, B = 1, C = 2, D = 3, E = 4, 
    F = 5, G = 6, H = 7, I = 8, J = 9, K = 10, L = 11;

    std::vector<std::vector<int>> graph;
    for(int i=0; i<12; i++) {
        std::vector<int> vec;
        graph.push_back(vec);
    }

    graph[0].push_back(E);
    graph[0].push_back(F);
    graph[0].push_back(I);
    graph[1].push_back(A);
    graph[1].push_back(C);
    graph[1].push_back(F);
    graph[2].push_back(B);
    graph[2].push_back(D);
    graph[2].push_back(E);
    graph[2].push_back(L);
    graph[3].push_back(C);
    graph[3].push_back(E);
    graph[3].push_back(H);
    graph[3].push_back(I);
    graph[3].push_back(K);
    graph[4].push_back(C);
    graph[4].push_back(G);
    graph[4].push_back(H);
    graph[4].push_back(I);
    graph[5].push_back(B);
    graph[5].push_back(G);
    graph[6].push_back(E);
    graph[6].push_back(F);
    graph[6].push_back(H);
    graph[7].push_back(D);
    graph[7].push_back(G);
    graph[7].push_back(I);
    graph[7].push_back(L);
    graph[8].push_back(D);
    graph[8].push_back(E);
    graph[8].push_back(H);
    graph[8].push_back(J);
    graph[9].push_back(I);
    graph[10].push_back(D);
    graph[10].push_back(I);
    graph[11].push_back(A);
    graph[11].push_back(H);

    pageRankRandomWalk(graph, 0.15);
}