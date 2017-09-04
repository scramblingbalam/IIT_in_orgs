# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:44:53 2017

@author: scram
"""

import itertools as it
import networkx as nx
import random
import time
import timeit
import numpy as np
import nested
import igraph as ig

########### NetworkX functions ###########

def reciprocity_ratio(gd):
    reciprocal = 0.0 
#    for i in it.combinations(gd.nodes(),2):
    for i in it.permutations(gd.nodes(),2):
        if i in gd.edges() and i[::-1] in gd.edges():
            reciprocal += 1.0
    return reciprocal/gd.size()


def change_reciprocity_ratio(gd,r):
    if r >= 0.0 and r <= 1.0:
        del_times = int(r*gd.size())+1
        print(del_times)
        for i in range(del_times):
            ran_edge = random.choice(gd.edges())
            gd.remove_edge(ran_edge[0],ran_edge[1])
#        return gd
    else:
        raise ValueError("value of 'r' must be between 1 and 0")

############# iGRRaph Functions

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
        newedge = random.choice(node_mat)
        edges.append(newedge)
    g.add_edges(edges)
    return g

###### START SCRIPT ###############
G = None
GD = None
Glast = None
n = 20# number of nodes
m = 200 # number of edges
p = 0.3#probability of triad closer
r = 0.5
m = int(float(m)/float(n))#+1
print(m)


#
#try:
#    GD = nx.read_gml("test_GML.gml")
#except:    
#    G = nx.powerlaw_cluster_graph(n, m, p)
#    #print(nx.triangles(G))
#    #print(nx.clustering(G))
#    print("Average Clustering",nx.average_clustering(G))
#    GD = G.to_directed()
#    change_reciprocity_ratio(GD,r)
#    nx.write_gml(GD,"test_GML.gml")

#### TO create new network comment above and uncomment below
G = nx.powerlaw_cluster_graph(n, m, p)
print(nx.triangles(G))
print(nx.clustering(G))
mean_cluster = nx.average_clustering(G)
print("Average Clustering",mean_cluster)
GD = G.to_directed()
change_reciprocity_ratio(GD,r)
## save graph as edge_list
nx.write_gml(GD,"test_GML.gml")
GD_gen = GD
######  END Graph creation  #################################

print(GD.nodes(),"Node list")
print("edge_num",GD.size())
print("Node Num", GD.order())
rr = reciprocity_ratio(GD)
print("reciprocity ratio",rr)


startNX = time.time()
#start = timeit.timeit()
pathsNX = list(nx.all_simple_paths(GD,0,1))
endNX = time.time()
timeNX = endNX-startNX


############### START iGraph evaluation  ##################
G = ig.Graph(directed=True)
G = ig.load("test_GML.gml")
#G = ig.load(G_gml)
print(G,"IG graph")
print("num edges",len(G.get_edgelist()))
#print("iGraph the same as last?", G == Glast)
G.write_graphml("test.graphml")

startIG = time.time()
pathsIG = find_all_paths(G,0, 1,mode = 'OUT')
endIG = time.time()
timeIG = endIG-startIG

print("\n___________________\n")
print("Num Nodes",n)
print("Num Edges",len(G.get_edgelist()),GD.size())
print("Reciprocity Ratio",rr)
print("Average Clustering",mean_cluster)
print("ALL path time NetworkX",timeNX)
print("ALL path time iGraph",timeIG)
print("\n___________________")
with open("IG_time.txt",'w') as filetime:
    filetime.write(str(timeIG))

#### Visualization 
import matplotlib.pyplot as plt
fig_size = plt.rcParams["figure.figsize"]
#print("Current size:", fig_size)
fig_size[0] = 16
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
#ig.plot(G, layout = G.layout('kk'))
nx.draw_spring(GD, with_labels=True,font_size = 30,node_size =700)

#print("edges teh same for NX and IG?",G.get_edgelist() == GD.edges())
print("PATHS the same for NX and IG?",set(map(tuple,pathsIG))==set(map(tuple,pathsNX)))
print("Number of paths from NetworkX",len(pathsNX))
print("Number of paths from iGraph",len(pathsIG))


#print(timeit.timeit("f_test()", setup="from __main__ import f_test"))
print(G.get_edgelist())
print("iGraph edges")
print("DONE!!!!!!!!!!!")
#print(GD.edges())
