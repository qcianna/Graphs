def create_tree_prim(weights):
    length = len(weights)
    t = [0]
    w = [i for i in range(1, length)]
    precursor = [None]

    while len(t) < length:
        min_weight = float('inf')
        idx = -1
        prev = None
        for i in t:
            for j in w:
                if weights[i][j] is not None:
                    if weights[i][j] < min_weight:
                        min_weight = weights[i][j]
                        idx = j
                        prev = i
        t.append(idx)
        precursor.append(prev)
        w.remove(idx)
        # print(t, "\n", precursor)
    return dict(zip(t, precursor))


# Zad.5 - Monika Kidawska
  
if __name__ == '__main__':
  weights_mat = [[None, 3, None, 9, 10, None], [3, None, None, 1, 8, 2], [None, None, None, 4, 7, None],
                [9, 1, 4, None, None, 3], [10, 8, 7, None, None, None], [None, 2, None, 3, None, None]]
  min_spanning_tree = create_tree_prim(weights_mat)
  print(min_spanning_tree)

  weights_mat_2 = [[None, None, 8, 7, 1], [None, None, 2, 3, 5], [8, 2, None, None, 8], [7, 3, None, None, 6], [1, 5, 8, 6, None]]
  min_spanning_tree = create_tree_prim(weights_mat_2)
  print(min_spanning_tree)