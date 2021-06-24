#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from gurobipy import *

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def solve_MIP(points, node_count):
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
        # S.remove(S[index])
        count += 1
    return obj, solution


def findMinDistance(point, S, visited):
    min_d = 999
    ret_index = 0
    for index, target in enumerate(S):
        if length(point, target) < min_d and index not in visited:
            min_d = length(point, target)
            ret_index = index
    visited.append(ret_index)
    return min_d, ret_index

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    obj, solution = solve_MIP(points, nodeCount)

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

