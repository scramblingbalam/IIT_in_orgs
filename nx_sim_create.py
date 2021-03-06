# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:35:16 2017

@author: Colin A Drayton cdrayton@umich.edu
"""
import itertools as it
import networkx as nx
import random
import time
import numpy as np
n = 10 # number of nodes
m = 50 # number of edges
p = 0.3#probability of triad closer
r = 0.3
m = int(float(m)/float(n))+1
print(m)
G = nx.powerlaw_cluster_graph(n, m, p)
#print(nx.triangles(G))
#print(nx.clustering(G))
print("Average Clustering",nx.average_clustering(G))
print("edge_num",G.size())
GD = G.to_directed()
print("edge_num",GD.size())
print(GD.nodes(),"Node list")

    
def reciprocity_ratio(gd):
    reciprocal = 0.0 
#    for i in it.combinations(gd.nodes(),2):
    for i in it.permutations(gd.nodes(),2):
        if i in gd.edges() and i[::-1] in gd.edges():
            reciprocal += 1.0
    return reciprocal/gd.size()

print("reciprocity ratio",reciprocity_ratio(GD))

def change_reciprocity_ratio(gd,r):
    if r >= 0.0 and r <= 1.0:
        r = r *(-1)
        del_times = int(r*gd.size())+1
        print(del_times)
        for i in range(del_times):
            ran_edge = random.choice(gd.edges())
            gd.remove_edge(ran_edge[0],ran_edge[1])
#        return gd
    else:
        raise ValueError("value of 'r' must be between 1 and 0")

def radom_weights(Graph,percision=3):
    """
    Adds random weights between 1 and 0 with a given percision
    """
    percision = percision + 2
    for i in Graph.edges_iter():
        ran_str = str(random.random())
        if len(ran_str) > percision:
            ran = float(ran_str[:percision])
        else:
            ran = float(ran_str)
        Graph[i[0]][i[1]]['weight'] = ran

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def conductance(g,i,j):
    """ calculates the conductance between two nodes in a graph
        ARGS:
            g a graph
            i index of starting node
            j index of ending node
        returns:
            float: conductance mesasure
    """
    C = []
    paths = nx.all_simple_paths(g,0,1)
#    print(len(list(paths)))
#    print(len(list(paths)),"LENGHT")
    for path in list(paths):
        path_calc = []
        for k,l in pairwise(path):
            Wkl = g[k][l]['weight']
            Dk = g.out_degree(k)
            path_calc.append(Wkl/Dk)
        C.append(np.prod(path_calc))
    return sum(C)




change_reciprocity_ratio(GD,r)
radom_weights(GD)
print("edge_num",GD.size())
print("Node Num", GD.order())
print("reciprocity ratio",reciprocity_ratio(GD))
path_start = time.time()

paths = nx.all_simple_paths(GD,0,1)
path_end = time.time()
print(len(list(paths)))
print("ALL path time",path_end-path_start)

con_start = time.time()
con_metric = conductance(GD,0,1)
con_end = time.time()
print(con_metric)
print("Conductance time",con_end-con_start)
