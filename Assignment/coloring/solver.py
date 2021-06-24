#!/usr/bin/python
# -*- coding: utf-8 -*-
from multiprocessing import cpu_count

import cvxpy as cp
import time
import numpy as np
import random
import networkx as nx


def solve_MIP(edges, vertex_num):
    global color_num
    build_start = time.time()
    dic = {50: 8, 70: 20, 100: 16, 250: 99, 500: 18, 1000: 124}  # change
    if vertex_num in dic.keys():
        color_num = dic[vertex_num]
    else:
        color_num = vertex_num
    X = cp.Variable((vertex_num, color_num), boolean=True)
    W = cp.Variable(color_num, boolean=True)
    constraints = []
    for item in edges:
        for col in range(color_num):
            constraints += [X[item[0], col] + X[item[1], col] <= W[col]]

    for node in range(vertex_num):
        exp = 0
        for col in range(color_num):
            exp += X[node, col]
        constraints += [exp == 1]

    obj = 0
    for i in range(color_num):
        obj += W[i]
    obj = cp.Minimize(obj)
    problem = cp.Problem(obj, constraints)
    build_finish = time.time()
    print(cp.installed_solvers())
    print("build time is ", build_finish - build_start)
    problem.solve(solver=cp.GUROBI)
    print("solve time is ", time.time() - build_finish)

    return obj.value, W.value, X.value


def color_node(edge_matrix, colored_dic, node_index, color_index):
    for index, val in enumerate(edge_matrix[node_index]):
        if val == 1 and colored_dic.get(index, -1) == color_index:
            return False
    for row_index, row in enumerate(edge_matrix):
        val = row[node_index]
        if val == 1 and colored_dic.get(row_index, -1) == color_index:
            return False
    return True


def solve_greedy(edges, vertex_num):
    edge_matrix = [[0 for i in range(vertex_num)] for j in range(vertex_num)]
    dic = {}
    for item in edges:
        edge_matrix[item[0]][item[1]] = 1
        edge_matrix[item[1]][item[0]] = 1
        # dic[item[0]] = dic.get(item[0], 0) + 1
        # dic[item[1]] = dic.get(item[1], 0) + 1
    # dic_ = sorted(dic.items(), key=lambda x:x[1], reverse=True)
    #
    # sorted_dic = [k[0] for k in dic_]

    color_index = 0
    uncolored_set = [i for i in range(vertex_num)]

    colored_dic = {}
    while uncolored_set:
        for i in reversed(uncolored_set):
            if color_node(edge_matrix, colored_dic, i, color_index):
                colored_dic[i] = color_index
                uncolored_set.remove(i)
            else:
                for j in range(color_index):
                    if color_node(edge_matrix, colored_dic, i, j):
                        colored_dic[i] = j
                        uncolored_set.remove(i)
        color_index += 1
    a = list(colored_dic.items())
    ret = [k[1] for k in sorted(a)]
    # print(colored_dic)
    return ret, colored_dic, edge_matrix, color_index


def tabucol(new_solution, edges, vertex_num, k):
    violations, vio_edge = evaluate(new_solution, edges)
    iter = 0
    tabu_list = []
    while violations != 0 and iter <= 5:
        new_solution, violations, vio_edge = swap_color(new_solution, violations, vio_edge, tabu_list, k, edges)
        iter += 1
        # tabu_list.extend(violations[1])
    if violations == 0:
        # print(new_solution)
        return True
    return False


def swap_color(new_solution, violations, vio_edge, tabu_list, k, edges):
    cur = new_solution
    min_vio_edge = []
    ret = []
    min_vio = 999
    for i in vio_edge:
        for j in range(k):
            cur[i[0]] = j
            violations, vio_edge = evaluate(cur, edges)
            if violations < min_vio:
                min_vio = violations
                ret = cur
                min_vio_edge = vio_edge
    return ret, min_vio, min_vio_edge


def evaluate(solution, edges):
    violation = 0
    vio_edge = []
    for edge in edges:
        if solution[edge[0]] == solution[edge[1]]:
            violation += 1
            vio_edge.append(edge)
    return violation, vio_edge


def random_init(vertex_num, k):
    colored_dic = {}
    random.seed(1)
    for i in range(vertex_num):
        random_col = random.randint(0, k-1)
        colored_dic[i] = random_col
    return colored_dic


def swap(edges, colored_dic, color_index_1, color_index_2):
    ret_dic = {}
    class_1 = []
    class_2 = []
    edges = np.array(edges)

    for item in colored_dic.items():
        if item[1] == color_index_1:
            class_1.append(item[0])
        if item[1] == color_index_2:
            class_2.append(item[0])
    neighbor_matrix = edges[np.ix_(class_1 + class_2, class_1 + class_2)]
    random_row = np.random.randint(0, len(class_1))
    neighbor_matrix = np.delete(neighbor_matrix, random_row, 1)
    neighbor_matrix = np.delete(neighbor_matrix, random_row, 0)
    return ret_dic


def greedy(node_count, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges)

    strategies = [nx.coloring.strategy_largest_first,
                  ]

    best_color_count, best_coloring = node_count, {i: i for i in range(node_count)}
    for strategy in strategies:
        print(graph.degree)
        curr_coloring = nx.coloring.greedy_color(G=graph, strategy=strategy)
        curr_color_count = max(curr_coloring.values()) + 1
        if curr_color_count < best_color_count:
            best_color_count = curr_color_count
            best_coloring = curr_coloring
    return best_color_count, 0, [best_coloring[i] for i in range(node_count)]

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append([int(parts[0]), int(parts[1])])

    # build a trivial solution
    # every node has its own color
    res = greedy(node_count, edges)
    # res = solve_gurobi(edges, node_count)
    # ret = solve_greedy(edges, node_count)
    # ret = []
    # eng = matlab.engine.start_matlab()
    # eng.cd(r"matlab")
    # eng.solve(edges, node_count)

    solution = []
    col_index = []
    ret = []
    # for row, row_val in enumerate(res[2]):
    #     for col, col_val in enumerate(res[2][row]):
    #         if res[2][row][col] == 1:
    #             if col not in col_index:
    #                 col_index.append(col)
    #             ret.append(col_index.index(col))

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, res))

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
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
