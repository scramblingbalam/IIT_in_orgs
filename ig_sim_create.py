# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:08:36 2017

@author: scram
"""
import itertools as it
import igraph as ig
import random as ran
import time 



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


#print(G)
#cluster = G.transitivity_undirected()
#print("clustering coefficient", cluster)
#print("num edges",len(G.get_edgelist()))
##G.to_directed(mutual=False)
#G.to_directed(mutual=True)
#print("num edges",len(G.get_edgelist()))
#tc = G.triad_census()
#print(tc["300"])
#start = time.time()
#paths = G.get_all_shortest_paths(0,to=1)
#end = time.time()
#print(paths)
#print(end-start)
#
#start =time.time()
#cuts = G.all_st_cuts(0,1)
#end = time.time()
#print(cuts)
#print(end-start)
#
#start = time.time()
#paths = G.get_all_simple_paths(0, to=1)
#end = time.time()
#print(paths)
#print(time.time())
#
#tris = [
#"120D",
#"120U",
#"120C",
##"210C",
#"300"]
##print(type(tris[3]))
#
#tot_tris = [tc[i] for i in tris]
#
#print(len(G.get_edgelist())*cluster)
#print(tot_tris)

#from optparse import OptionParser
#import inspect
#inspect.getmembers(OptionParser, predicate=inspect.ismethod)
