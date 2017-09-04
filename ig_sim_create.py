# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:08:36 2017

@author: scram
"""
import itertools as it
import igraph as ig
import random as ran
import time 

def find_all_paths(graph, start, end, mode = 'OUT', maxlen = None):
    def find_all_paths_aux(adjlist, start, end, path, maxlen = None):
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        if maxlen is None or len(path) <= maxlen:
            for node in adjlist[start] - set(path):
                paths.extend(find_all_paths_aux(adjlist, node, end, path, maxlen))
        return paths
    adjlist = [set(graph.neighbors(node, mode = mode)) \
        for node in range(graph.vcount())]
    all_paths = []
    start = start if type(start) is list else [start]
    end = end if type(end) is list else [end]
    for s in start:
        for e in end:
            all_paths.extend(find_all_paths_aux(adjlist, s, e, [], maxlen))
    return all_paths

def create_random(n,m):
    g = ig.Graph()
    g.add_vertices(n)
    node_mat = list(it.permutations(range(n),2))
    edges = []
    for i in range(m):
        newedge = ran.choice(node_mat)
        edges.append(newedge)
    g.add_edges(edges)
    return g
#
#def tune_transitivity(G,t):
#    for 

G = create_random(10,100)
#cuts = G.all_st_cuts(0,300)
start = time.time()
#paths = G.get_all_simple_paths(0, to=1)
paths = find_all_paths(G,0, 1)
end = time.time()
print(paths)
print(time.time())



