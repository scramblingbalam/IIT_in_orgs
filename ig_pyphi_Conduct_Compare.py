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
import sys
import gate2tpm as g2t
import pyphi
import csv 
import nested

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


def find_all_paths_Mongo(paths_coll_str,graphColl, start,DB ,end, mode = 'OUT', maxlen = None,deliminator = ','):
    path_coll = DB[paths_coll_str+"start{}_end{}".format(start,end)]
    def find_all_paths_Mongo_aux(path_coll,adjlist,graphColl, start, end, path, maxlen = None):
        path = path + [start]
        if start == end:
            try:
                path_str = deliminator.join(map(str,path))
                path_coll.insert_one({'_id':path_str})
#                return [path]
            except Exception as err:
                if err.code == 11000:
                    pass
                else:
                    print(err)
                    sys.exit()            

        paths = []
        if maxlen is None or len(path) <= maxlen:
            for node in set(adjlist[start]) - set(path):
                paths.extend(find_all_paths_Mongo_aux(path_coll,adjlist,graphColl, node, end, path, maxlen))
        return paths
    adjlist = [set(list(graphColl.find({'_id':start}))[0]['neighbors_{}'.format(mode)])
                for start in graphColl.distinct('_id')]
    start = start if type(start) is list else [start]
    end = end if type(end) is list else [end]
    for s in start:
        for e in end:
            find_all_paths_Mongo_aux(path_coll,adjlist,graphColl, s, e, [], maxlen)
    def destringize(string):
        return list(map(int,string.split(deliminator)))
    return list(map(destringize,path_coll.distinct('_id')))

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

def t2eid():
    return
def create_random_weight_range(n,m,Range=(0,10,1)):
    R = Range
    g = ig.Graph(directed = True)
    g.add_vertices(n)
    node_mat = list(it.permutations(range(n),2))
    edges = sorted(random.sample(node_mat,m))
    g.add_edges(edges)
    weights = [random.randrange(R[0],R[1],R[2]) for i in edges]
    g.es["weight"]=weights
    return g

def random_weights(Graph,percision=3):
    """
    Adds random weights between 1 and 0 with a given percision
    """
    percisionStr = "{"+"0:.{}f".format(percision)+"}"
    float(percisionStr.format(random.random()))
    Graph.es['weight'] = [float(percisionStr.format(random.random())) for i in range(Graph.ecount())]    
    return Graph


def create_random_weighted(n,m,percision = 3,weighted = True,self_loops=False):
    g = ig.Graph(directed = True)
    g.add_vertices(n)
    edges = list(it.permutations(range(n),2))
    if not self_loops:
        edges = list(filter(lambda x:x[0]!=x[1],edges))
    g.add_edges(random.sample(list(edges) ,m))
    if weighted:
        g = random_weights(g,percision=percision)
    else:
        g.es['weight'] = [1]*g.ecount()
    return g


def create_full_directed(n):
    g = ig.Graph(directed =True)
    g.add_vertices(n)
    node_mat = list(it.permutations(range(n),2))
    g.add_edges(sorted(node_mat))
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


def graph_density(V,E,directed = True):
    V = abs(V)
    E = abs(E)
    if directed:
        x = 1
    else:
        x = 2
    return (x * E)/(V * (V-1))


def edges4density(D,V,directed = True):
    V = abs(V)
    if directed:
        x = 1
    else:
        x = 2
    return int((D*(V * (V-1)))/x)

def nodes4density(d,e):
    top = (d + (4*e))**0.5
    bottom = d**0.5
    return int(abs(((top/bottom)+1)*0.5))


def reciprocity_ratio(gd):
    """
    takes a directed igraph graph as input and outputs it's reciprocity ratio
    """
    reciprocal = 0.0
    edge_list  = gd.get_edgelist()
    for i in it.permutations(range(gd.vcount()),2):
        if i in edge_list and i[::-1] in edge_list:
            reciprocal += 1.0
    return reciprocal/gd.ecount()



def change_reciprocity_ratio(Graph,r):
    if r >= 0.0 and r <= 1.0:
        rr = reciprocity_ratio(Graph) 
        ratio_dif = r-rr
        edge_dif = ratio_dif * Graph.ecount()
        edges = set(Graph.get_edgelist())
        edgesReverse = set([i[::-1] for i in Graph.get_edgelist()])
        biEdges = edges & edgesReverse
        uniEdges = set(edges).difference(set(biEdges))
        if edge_dif < 0:
            edges2remove = random.sample(biEdges,abs(int(edge_dif)))
            eids2remove = Graph.get_eids(pairs = edges2remove)
            Graph.delete_edges(eids2remove)
        elif edge_dif > 0:
            edges2remove = random.sample(uniEdges,abs(int(edge_dif)))
            eids2remove = Graph.get_eids(pairs = edges2remove)
            Graph.delete_edges(eids2remove)
    else:
        raise ValueError("value of 'r' must be between 1 and 0")
        
        

#def transitivity(G):
#    #clustering coefficent calc for R
#    # Calculate transitivity from my formula
#    #A <- as_adj(G)
#    A = np.array(G.get_adjacency().data).reshape
#    print(A.shape)
#    #A2 <- crossprod(A) # A^2
#    A2 = np.cross(A,A)
#    #numerator <- sum(diag(crossprod(A2, A))) # trace(A^3)
#    numerator = np.sum(np.diag(np.cross(A2,A)))
#    #denominator <- sum(A2) 
#    denominator = np.sum(A2)
#    #myTrans <- numerator/denominator
#    return np.true_divide(numerator,denominator)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def path2edge(iterable,graph):
    """
        takes an iterable represnting a path and outputs the edge id 
    """
    return (graph.es[graph.get_eid(pair[0],pair[1])] for pair in pairwise(iterable))

def productZero(iterable):
    if not isinstance(iterable,np.ndarray):
        iterable = np.array(iterable)
    if not any(iterable):
        return 0
    else:
        return np.prod(iterable)

def conductance(g,i,j,mode = "OUT",maxlen=None):#, add_attribute = True):
    """ calculates the conductance between two nodes in a graph
        ARGS:
            g a graph
            i index of starting node
            j index of ending node
        returns:
            float: conductance mesasure
    """
    Conduct_paths = []
    paths = find_all_paths(g, i, j, mode=mode, maxlen=maxlen)
    for path in list(paths):
        path_pairs = list(pairwise(path))
        path_eids = g.get_eids(pairs = path_pairs)
        Wkl = np.array(g.es[path_eids]['weight'])
        path_starts = [i[0] for i in path_pairs]
        Dk = np.array(g.vs[path_starts].degree(mode=mode))  
        Conduct_paths.append(np.prod(Wkl/Dk))
    return sum(Conduct_paths)

def conductance_full(g,mode = "OUT",maxlen=None):#, add_attribute = True):
    """ calculates the conductance between two nodes in a graph
        ARGS:
            g a graph
        returns:
            float: conductance mesasure
            object: an iGraph graph object with added conduct attributes
    """
    Conduct_paths = []
    path_dict = {(edge.tuple[0],edge.tuple[1]):
        find_all_paths(g,edge.tuple[0],edge.tuple[1],mode=mode,maxlen=maxlen) 
        for edge in g.es}
    g.es['conduct'] = [edge['weight']/g.vs[edge.tuple[0]].degree(mode=mode) for edge in g.es]
    for nodes, paths in path_dict.items():
        paths_prods = []
        for path in paths:
            conduct_array = np.array( [edge['conduct'] for edge in path2edge(path,g)] )
            paths_prods.append(productZero(conduct_array))
        Conduct_paths.append(sum(paths_prods))       
    return sum(Conduct_paths),g

if __name__ == "__main__":
    time_list = []
    compare_list = []
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    db = client.IIT_in_orgs
    for i in range(2):
        compareCSV = {}
        timeCSV = {}
    #    client = MongoClient()
    #    client = MongoClient('localhost', 27017)
    #    db = client.IIT_in_orgs
        paperN = 100
        paperM = 5000 
        
        n = 10 # number of nodes
        d = graph_density(paperN, paperM)
        print(d)
        d = 0.5
        m = edges4density(d,n)
        p = 0.3 # probability of triad closer
        r = 0.5
        
        G = create_random_weighted(n,m,weighted=False)
        change_reciprocity_ratio(G,r)
        
        CM = nested.list_convert(np.array,G.get_adjacency())
        CMw = nested.list_convert(np.array,G.get_adjacency('weight'))
        
        G.vs["gate"] = [random.choice(g2t.tableMulti('all')) for i in range(G.vcount())]
        Gates = [g2t.tableMulti(gate) for gate in G.vs['gate']]
        States = g2t.noise(G.vcount())
        TPM = g2t.states2tpm(States,Gates,CM)
        
    
        network = pyphi.Network(TPM, connectivity_matrix=CM)#,
        #                        node_labels=labels)
        state = tuple(random.choice(TPM))
        
        ### In this case, we want the ΦΦ of the entire network, 
        ### so we simply include every node in the network in our subsystem
        
        subsystem = pyphi.Subsystem(network, state, range(network.size) )
#        subsystem = pyphi.Subsystem(network, state, node_indices)
            
        ### Now we use big_phi() function to compute the ΦΦ of our subsystem:
#        big_phi_time1 = time.time()
#        
#        big_phi = pyphi.compute.big_phi(subsystem)
#        big_phi_time2 = time.time()
#
#        big_phi_time = big_phi_time2 - big_phi_time1
#        print("BIG PHI = ",big_phi)
##       print("Calculating Big Phi took: ", big_phi_time)
#            
#        compareCSV["Phi"] = big_phi
#        timeCSV["Phi"] = big_phi_time
           

        time1 = time.time()
        Conductance =sum(conductance(G,edge.tuple[0],edge.tuple[1]) for edge in G.es) 
        time2 = time.time()
        timeF1 = time.time()
        ConductanceF, G = conductance_full(G)
        timeF2 = time.time()
        timeF = timeF2-timeF1
        timeC = time2-time1
        print(Conductance,"Conductance")
        print(ConductanceF,"Conductance Full")
        print(timeC,"Time Conductance")
        print(timeF,"Time Conductance Full")
        print((timeC/timeF)*100,"Percentage faster")
        
#        print("Full Conductance time",conductEnd - conductStart)
#        print("\n")
        
        ### append dict to CSV row list

        