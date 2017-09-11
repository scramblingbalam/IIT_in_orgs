# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:08:36 2017

@author: scram
"""
import itertools as it
import igraph as ig
import random
import numpy as np
import time 
import dask.array as da
from bson.son import SON
from pymongo import MongoClient

def find_all_paths(graph, start, end, mode = 'OUT', maxlen = None):
    def find_all_paths_aux(adjlist, start, end, path, maxlen = None):
        path = path + [start]
#        print("Start",start)
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

def find_all_paths_Mongo(graph,graphColl, start, end, mode = 'OUT', maxlen = None):
    def find_all_paths_Mongo_aux(adjlist,graphColl, start, end, path, maxlen = None):
#        path = list(pathColl.find({'_id':'{}->{}'.format(start,end)}))
        path = path + [start]
#        print("start",start)
        if start == end:
            return [path]
        paths = []
        if maxlen is None or len(path) <= maxlen:
            neighbors = list(graphColl.find({'_id':start}))[0]['neighbors_{}'.format(mode)]
#            print([node for node in set(neighbors) - set(path)])
#            print([node for node in adjlist[start] - set(path)])
            for node in set(neighbors) - set(path):
                paths.extend(find_all_paths_Mongo_aux(adjlist,graphColl, node, end, path, maxlen))
        return paths
    adjlist = [set(graph.neighbors(node, mode = mode)) \
        for node in range(graph.vcount())]
    all_paths = []
    start = start if type(start) is list else [start]
    end = end if type(end) is list else [end]
    for s in start:
        for e in end:
            empty_path = []
            all_paths.extend(find_all_paths_Mongo_aux(adjlist,graphColl, s, e, empty_path, maxlen))
    return all_paths

def create_random(n,m):
    g = ig.Graph(directed =True)
    g.add_vertices(n)
    node_mat = list(it.permutations(range(n),2))
#    edges = []
#    for i in range(m):
#        newedge = np.random.choice(node_mat,replace = False)
#        edges.append(newedge)
    edges = random.sample(node_mat,m)
#    for edge in edges:
#        g.add_edge(edge[0],edge[1])
    g.add_edges(sorted(edges))
    return g

def igraph2mongo(graph,collection,mode='OUT',overwrite = False):
    """
    Takes an iGraph graph object and turns it into a local Monogo edge_list
    """
    for i in graph.vs:
        if not list(collection.find({'_id':i.index})):
            post = {"_id": i.index,
                    "neighbors_{}".format(mode):list(set(graph.neighbors(i.index,mode=mode)))}
            post_id = collection.insert_one(post).inserted_id
            print( "node ",post_id," added")
        elif overwrite == True:
            post = {"_id": i.index,
                    "neighbors_{}".format(mode):list(set(graph.neighbors(i.index,mode=mode)))}
            collection.replace_one({'_id':i.index},post)
            print("node ",i.index," replaced")
        else:
#            print("THIS object has the _id",i.index,list(collection.find({'_id':i.index})))
            pass
    if overwrite == True:
        print(collection, "has been changed")


def mongo2igraph(collection,directed = True,mode='OUT'):
    """
    Takes a mongo collection and out puts an iGraph network 
    """
    g = ig.Graph(directed =directed)
    node_ids = collection.distinct("_id")
#    print(node_ids)
    g.add_vertices(len(node_ids))
#    print(g)
    edges = []
    for i in node_ids:
        for j in collection.find_one({'_id':i})["neighbors_{}".format(mode)]:
#            print(i,j)
            edges.append((i,j)) 
    g.add_edges(sorted(edges))
    return g


if __name__ == "__main__":
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    db = client.IIT_in_orgs
    
    n = 10 # number of nodes
    m = 100 # number of edges
    p = 0.3 # probability of triad closer
    r = 0.5
    
    
    coll_graph = db["node{}_edge{}_p{}%_r{}%_graph".format(n,m,int(p*100),int(r*100))]
    coll_paths = db["node{}_edge{}_p{}%_r{}%_paths".format(n,m,int(p*100),int(r*100))]
    if not coll_graph.find_one():
        G = create_random(n,m)
        print("Graph Put To MongoDB")
        igraph2mongo(G,coll_graph)
    else:
        print("Graph pulled from MongoDB")
        G = mongo2igraph(coll_graph, directed = True)
#    print("THE TWO GRAPHS ARE THE SAME?",Gmongo.get_edgelist() == G.get_edgelist())
    start = time.time()
    paths = find_all_paths(G,0, 1)
    end = time.time()
    print("iGraph time",end-start)
    print("Number of paths",len(paths))
    
    startMg = time.time()
    pathsMg = find_all_paths_Mongo(G,coll_graph,0, 1)
    endMg = time.time()
    
    print("memory and mongo mehtods same",pathsMg == paths)
    print("Number of paths Mongo",len(pathsMg))
    print("Mongo time",endMg-startMg)
    
    paths_s = set(map(tuple,paths))
    pathsMg_s = set(map(tuple,pathsMg))
    dif_pathsMg = paths_s - pathsMg_s
    dif_paths = pathsMg_s - paths_s
    print(dif_paths)
    print(dif_pathsMg)