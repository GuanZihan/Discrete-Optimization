#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import time
Item = namedtuple("Item", ['index', 'value', 'weight'])

taken = []


def Oracle2(dp, items, item_count, weights):
    for i in range(item_count):
        for j in range(weights):
            if i == 0: dp[i][j] = 0
            elif items[i].weight <= j:
                dp[i][j] = max(dp[i - 1][j - items[i].weight] + items[i].value, dp[i - 1][j])
            else: dp[i][j] = dp[i-1][j]


def Oracle(items, k, j):
    if j == 0:
        return 0
    elif items[j].weight <= k:
        left = Oracle(items, k, j - 1)
        right = items[j].value + Oracle(items, k - items[j].weight, j - 1)
        if left > right:
            max_item = left
        else:
            max_item = right
        return max_item
    else:
        return Oracle(items, k, j - 1)

def traceback(items, w, i, taken):
    if i == 0:
        return taken
    if Oracle(items, w, i) > Oracle(items, w, i - 1):
        return traceback(items, w - items[i].weight, i-1, taken +[i])
    else:
        # print("else", i, )
        return traceback(items, w, i-1, taken)

def traceback_dp(dp, items, w, i, taken):
    if i == 0:
        return taken
    if dp[i][w] > dp[i-1][w]:
        return traceback_dp(dp, items, w - items[i].weight, i-1, taken +[i])
    else:
        # print("else", i, )
        return traceback_dp(dp, items, w, i-1, taken)

def process(taken, capacity):
    ret = []
    for i in range(capacity):
        if i in taken:
            ret.append(1)
        else:
            ret.append(0)
    return ret


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    add = [0] * item_count

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    # recursion version
    # start = time.time()
    # value = Oracle(items, capacity - 1, item_count - 1)
    # print("recursion time: ", time.time() - start)

    # dynamic programming version
    dp = [[0 for i in range(capacity)] for j in range(item_count)]
    start = time.time()
    Oracle2(dp, items, item_count, capacity)
    value = dp[-1][-1]
    # print("dynamic programming time: ", time.time() - start)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    taken = []
    start = time.time()
    taken = traceback_dp(dp, items, capacity - 1, item_count - 1, taken)
    taken = process(taken, item_count)
    # print("process: ", time.time() - start)
    output_data += ' '.join(map(str, taken))

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

