#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
#from gurobipy import *
import numpy as np
import random
import numba
from numba import jit
# import six
import sys
# import dimod
#import networkx as nx
# from python_tsp.heuristics import solve_tsp_local_search
# import dwave_networkx as dnx
import matplotlib.pyplot as plt
# sys.modules['sklearn.externals.six'] = six
# import mlrose

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def solve_naive_greedy(points, node_count):
    """repeatedly fetch the closest one to the last visited point"""
    V = [points[0]]
    solution = [0]

    D = [[0 for i in range(node_count)] for j in range(node_count)]
    S = points[1:]

    for r, point_row in enumerate(points):
        for c, point_col in enumerate(points):
            D[r][c] = length(point_row, point_col)

    count = 1
    obj = 0
    visited = []

    while count < node_count:
        min_d, index = findMinDistance(V[count - 1], S, visited)
        obj += min_d
        V.append(S[index])
        solution.append(index + 1)
        count += 1
    obj += length(V[-1], V[0])
    return obj, solution


def findMinDistance(point, S, visited):
    """returning the min distance and index"""
    min_d = np.Inf
    ret_index = 0
    for index, target in enumerate(S):
        if length(point, target) < min_d and index not in visited:
            min_d = length(point, target)
            ret_index = index
    visited.append(ret_index)
    return min_d, ret_index


def find_all_solution(cur_solution):
    index = 0
    ret = []
    used = []
    swap_index = []
    while index < 20 * len(cur_solution):
        i = np.random.randint(0, len(cur_solution))
        j = np.random.randint(0, len(cur_solution))
        while (i, j) in used or i == j:
            i = np.random.randint(0, len(cur_solution))
            j = np.random.randint(0, len(cur_solution))
        used.append((i, j))
        new_solution = []
        if i < j:
            new_solution = cur_solution.copy()
            new_solution[i:j] = reversed(new_solution[i:j])
        else:
            new_solution = cur_solution.copy()
            new_solution[j:i] = reversed(new_solution[j:i])
        ret.append(new_solution)
        swap_index.append(tuple((i, j)))
        index += 1
    return ret, swap_index


def evaluate(points, all_solutions, tabu_list, swap_index):
    best = np.Inf
    best_permutation = []
    best_index = 0
    for i, p in enumerate(all_solutions):
        cur = compute_length(points, p)
        if cur < best and swap_index[i] not in tabu_list and tuple(reversed(swap_index[i])) not in tabu_list:
            best_permutation = p
            best = cur
            best_index = i
    return best, best_permutation, best_index


def solve_2_opt(points, node_count):
    if node_count == 33810:
        a = [i for i in range(node_count)]
        np.random.shuffle(a)
        ini_solution = a
        ini_obj = compute_length(points, ini_solution)
        return ini_obj, ini_solution
    else:
        ini_obj, ini_solution = solve_naive_greedy(points, node_count)
    max_iter = 300
    opt_obj = ini_obj
    opt_solution = ini_solution
    cur_obj = ini_obj
    cur_solution = ini_solution
    iter = 0
    X= []
    y = []
    tabu_list = []
    tabu_len = 50
    while iter <= max_iter:
        all_solutions, swap_index = find_all_solution(cur_solution)
        nei_obj, nei_solution, nei_index = evaluate(points, all_solutions, tabu_list, swap_index)
        if iter % 10 == 0:
            print("====iter{}====".format(iter))
        if nei_obj > cur_obj:
            # whether to accept the worse solution
            r = np.random.rand()
            if r < 0.5:
                cur_obj = nei_obj
                cur_solution = nei_solution
                if len(tabu_list) >= tabu_len:
                    tabu_list.remove(tabu_list[0])
                tabu_list.append(tuple(reversed(swap_index[nei_index])))
            else:
                break
        else:
            cur_obj = nei_obj
            cur_solution = nei_solution
            if len(tabu_list) >= tabu_len:
                tabu_list.remove(tabu_list[0])
            tabu_list.append(tuple(reversed(swap_index[nei_index])))
        if cur_obj < opt_obj:
            opt_obj = cur_obj
            opt_solution = cur_solution
        print(tabu_list)
        X.append(opt_obj)

        iter += 1
        y.append(iter - 1)
    # plt.plot(y, X)
    X = [i.x for i in points]
    y = [i.y for i in points]
    for i in opt_solution:
        plt.plot(X[i:i-2], y[i:i-2], 'bo-')
    plt.show()

    return opt_obj, opt_solution


def solve_k_opt(points, node_count):
    if node_count == 33810:
        a = [i for i in range(node_count)]
        np.random.shuffle(a)
        ini_solution = a
        ini_obj = compute_length(points, ini_solution)
        return ini_obj, ini_solution
    else:
        # ini_obj, ini_solution = solve_naive_greedy(points, node_count)
        a = [i for i in range(node_count)]
        np.random.shuffle(a)
        ini_solution = a
        ini_obj = compute_length(points, ini_solution)
    max_iter = 300
    opt_obj = ini_obj
    opt_solution = ini_solution
    cur_obj = ini_obj
    cur_solution = ini_solution
    iter = 0
    max_k = 3
    X= []
    y = []
    tabu_list = []
    tabu_len = 50
    while iter <= max_iter:
        all_solutions, swap_index = find_all_solutions_3_opt(points, cur_solution, max_k)
        if len(all_solutions) == 0:
            break
        nei_obj, nei_solution, nei_index = evaluate(points, all_solutions, tabu_list, swap_index)
        if iter % 10 == 0:
            print("====iter{}====".format(iter))
        if nei_obj > cur_obj:
            # whether to accept the worse solution
            r = np.random.rand()
            if r < 0.5:
                cur_obj = nei_obj
                cur_solution = nei_solution
                if len(tabu_list) >= tabu_len:
                    tabu_list.remove(tabu_list[0])
                tabu_list.append(tuple(reversed(swap_index[nei_index])))
            else:
                break
        else:
            cur_obj = nei_obj
            cur_solution = nei_solution
            if len(tabu_list) >= tabu_len:
                tabu_list.remove(tabu_list[0])
            tabu_list.append(tuple(reversed(swap_index[nei_index])))
        if cur_obj < opt_obj:
            opt_obj = cur_obj
            opt_solution = cur_solution
        print(tabu_list)
        X.append(opt_obj)

        iter += 1
    return opt_obj, opt_solution


def find_all_solutions_3_opt(points, cur_solution, max_k):
    index = 0
    ret = []
    used = []
    swap_index = []
    for i in range(200 * len(cur_solution)):
        new_solution = cur_solution.copy()
        ks = random.sample([i for i in range(0, len(cur_solution))], 3)
        ks = sorted(ks)
        while ks in used:
            ks = random.sample([i for i in range(len(cur_solution))], 2)
        A, B, C, D, E, F = cur_solution[ks[0] - 1], cur_solution[ks[0]], \
                           cur_solution[ks[1] - 1], cur_solution[ks[1]],\
                           cur_solution[ks[2] - 1], cur_solution[ks[2] - 1]
        # d0 = length(points[A], points[B]) + length(points[C], points[D]) + length(points[E], points[F])
        # d1 = length(points[A], points[C]) + length(points[B], points[D]) + length(points[E], points[F])
        # d2 = length(points[A], points[B]) + length(points[C], points[E]) + length(points[D], points[F])
        # d3 = length(points[A], points[D]) + length(points[E], points[B]) + length(points[C], points[F])
        # d4 = length(points[F], points[B]) + length(points[C], points[D]) + length(points[E], points[A])
        # exchange i and j
        new_solution[ks[0]: ks[1]] = reversed(new_solution[ks[0]: ks[1]])
        ret.append(new_solution)
        swap_index.append(tuple((ks[0], ks[1])))

        # exchange j and k
        new_solution[ks[1]: ks[2]] = reversed(new_solution[ks[1]: ks[2]])
        ret.append(new_solution)
        swap_index.append(tuple((ks[1], ks[2])))

        # exchange i and j
        new_solution[ks[0]: ks[2]] = reversed(new_solution[ks[0]: ks[2]])
        ret.append(new_solution)
        swap_index.append(tuple((ks[0], ks[2])))

        # if d0 > d1:
        #     new_solution[ks[0]: ks[1]] = reversed(new_solution[ks[0]: ks[1]])
        #     ret.append(new_solution)
        #     swap_index.append(tuple((ks[0], ks[1])))
        #     continue
        # elif d0 > d2:
        #     new_solution[ks[1]: ks[2]] = reversed(new_solution[ks[1]: ks[2]])
        #     ret.append(new_solution)
        #     swap_index.append(tuple((ks[1], ks[2])))
        #     continue
        # elif d0 > d4:
        #     new_solution[ks[0]: ks[2]] = reversed(new_solution[ks[0]: ks[2]])
        #     ret.append(new_solution)
        #     swap_index.append(tuple((ks[0], ks[2])))
        #     continue
        # elif d0 > d3:
        #     tmp = new_solution[ks[1]: ks[2]] + new_solution[ks[0]: ks[1]]
        #     new_solution[ks[0]: ks[2]] = tmp
        #     print(len(new_solution))
        #     ret.append(new_solution)
        #     swap_index.append(tuple((ks[0], ks[1], ks[2])))
        #     continue
    return ret, swap_index
# def solve_tsp(points, nodeCount):
#     m = [[0 for i in range(nodeCount)] for j in range(nodeCount)]
#     for i in range(nodeCount):
#         for j in range(nodeCount):
#             m[i][j] = length(points[i], points[j])
#     m = np.array(m)
#     permutation, distance = solve_tsp_local_search(m)
#     return distance, permutation
#
#
# # bad quality
# def solve_mlrose(points, nodeCount):
#     fitness_coords = mlrose.TravellingSales(coords=points)
#     problem_fit = mlrose.TSPOpt(length=nodeCount, fitness_fn=fitness_coords, maximize=False)
#     best_state, best_fitness = mlrose.simulated_annealing(problem_fit, random_state=3)
#     obj = compute_length(points, best_state)
#     return best_fitness, best_state
#
#
# # bad quality
# def solve_nx(points):
#     G = nx.Graph()
#     m = set()
#     for indexi, i in enumerate(points):
#         for indexj, j in enumerate(points):
#             if i != j:
#                 m.add(tuple((indexi, indexj, length(i, j))))
#     G.add_weighted_edges_from(m)
#     ret = dnx.traveling_salesperson(G, dimod.SimulatedAnnealingSampler())
#     obj = compute_length(points, ret)
#     return obj, ret


def compute_length(points, permutation):
    """given a permutation, compute its length"""
    obj = 0
    for index, i in enumerate(permutation):
        obj += length(points[i], points[permutation[index - 1]])
    return obj


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # obj, solution = solve_naive_greedy(points, nodeCount)
    obj, solution = solve_k_opt(points, nodeCount)
    # print(compute_length(points, solution), obj)
    # obj, solution = solve_tsp(points, nodeCount)
    # obj, solution = solve_nx(points)
    # obj, solution = solve_mlrose(points, nodeCount)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
