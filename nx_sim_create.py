# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:35:16 2017

@author: Colin A Drayton cdrayton@umich.edu
"""
import itertools as it
import networkx as nx
import random
n = 100 # number of nodes
m = 1000 # number of edges
p = 0.3#probability of triad closer
m = int(float(m)/float(n))+2
print(m)
G = nx.powerlaw_cluster_graph(n, m, p)
#print(nx.triangles(G))
#print(nx.clustering(G))
print(nx.average_clustering(G))
print(len(G.edges()))
GD = G.to_directed()
print(len(GD.edges()))
print(GD.nodes())

    
def reciprocity_ratio(gd):
    reciprocal = 0.0 
#    for i in it.combinations(gd.nodes(),2):
    for i in it.permutations(gd.nodes(),2):
        if i in gd.edges() and i[::-1] in gd.edges():
            reciprocal += 1.0
    return reciprocal/gd.size()

print(reciprocity_ratio(GD))

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
    
change_reciprocity_ratio(GD,0.5)
print(reciprocity_ratio(GD))

print(GD.size())