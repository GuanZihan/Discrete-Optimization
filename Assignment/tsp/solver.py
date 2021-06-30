#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
#from gurobipy import *
import numpy as np
# import six
import sys
# import dimod
#import networkx as nx
# from python_tsp.heuristics import solve_tsp_local_search
# import dwave_networkx as dnx
#import matplotlib.pyplot as plt
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
    min_d = 99999
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
    while index < 200:
        i = np.random.randint(0, len(cur_solution))
        j = np.random.randint(0, len(cur_solution))
        while (i, j) in used:
            i = np.random.randint(0, len(cur_solution))
            j = np.random.randint(0, len(cur_solution))
        used.append((i, j))
        new_solution = []
        if i <= j:
            new_solution = cur_solution[:i]
            temp = []
            for p in range(i, j + 1):
                temp.append(cur_solution[p])
            temp.reverse()
            new_solution += temp
            new_solution += cur_solution[j + 1:]
        if i > j:
            new_solution = cur_solution[:j]
            temp = []
            for p in range(j, i + 1):
                temp.append(cur_solution[p])
            temp.reverse()
            new_solution += temp
            new_solution += cur_solution[i + 1:]
        ret.append(new_solution)
        index += 1
    return ret


def evaluate(points, all_solutions):
    best = 99999
    best_permutation = []
    for p in all_solutions:
        cur = compute_length(points, p)
        if cur < best:
            best_permutation = p
            best = cur
    return best, best_permutation


def solve_2_opt(points, node_count):
    ini_obj, ini_solution = solve_naive_greedy(points, node_count)
    opt_obj = ini_obj
    opt_solution = ini_solution
    cur_obj = ini_obj
    cur_solution = ini_solution
    iter = 0
    max_iter = 500
    X= []
    y = []
    while iter <= max_iter:
        all_solutions = find_all_solution(cur_solution)
        nei_obj, nei_solution = evaluate(points, all_solutions)
        if nei_obj > cur_obj:
            # whether to accept the worse solution
            r = np.random.rand()
            if r < 0.4:
                cur_obj = nei_obj
                cur_solution = nei_solution
            else:
                break
        else:
            cur_obj = nei_obj
            cur_solution = nei_solution
        if cur_obj < opt_obj:
            opt_obj = cur_obj
            opt_solution = cur_solution
        X.append(opt_obj)

        iter += 1
        y.append(iter - 1)
    #plt.plot(y, X)
    #plt.show()
    return opt_obj, opt_solution


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
    obj, solution = solve_2_opt(points, nodeCount)
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
