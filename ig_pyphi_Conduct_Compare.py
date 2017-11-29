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

def create_random_weight_range(n,m,Range=(0,10,1)):
    R = Range
    g = ig.Graph(directed = True)
    print(g.is_weighted())
    g.add_vertices(n)
    print(g.is_weighted())
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


def create_random_weighted(n,m,percision = 3):
    g = ig.Graph(directed = True)
    g.is_weighted()
    g.add_vertices(n)
    g.add_edges(random.sample(list(it.permutations(range(n),2)),m))
    g = random_weights(g,percision=percision)
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

def conductance(g,i,j):
    """ calculates the conductance between two nodes in a graph
        ARGS:
            g a graph
            i index of starting node
            j index of ending node
        returns:
            float: conductance mesasure
    """
    Conduct = 0.0
    paths = find_all_paths(g, i, j)
    for path in list(paths):
        path_pairs = list(pairwise(path))
        path_eids = g.get_eids(pairs = path_pairs)
        Wkl = np.array(g.es[path_eids]['weight'])
        path_starts = [i[0] for i in path_pairs]
        Dk = np.array(g.vs[path_starts].degree(mode='OUT'))    
        Conduct = sum(Wkl/Dk)
#        print("Conducatnace between {} & {} = ".format(i,j), Conductance)
    return Conduct



if __name__ == "__main__":
    time_list = []
    compare_list = []
    for i in range(10):
        compareCSV = {}
        timeCSV = {}
    #    client = MongoClient()
    #    client = MongoClient('localhost', 27017)
    #    db = client.IIT_in_orgs
        
        n = 10 # number of nodes
        m = 50 # number of edges
        p = 0.3 # probability of triad closer
        r = 0.5
        start_node = 0
        end_node = 1
        mode = "OUT"
    #    print("Nodes",n,"Edges",m,"Prob Triads",p,"Reciprocity",r)
        density = graph_density(n,m)
    #    print("Density",density)
        edges = edges4density(density,n)
    #    print(edges)
        density = graph_density(n,edges)
    #    print("Density",density)
        nodes = nodes4density(density,m)
    #    print(nodes)
        density = graph_density(n,edges)
    #    print("Density",density)
    #    G = create_random(20,200)
    #    G = create_random_weight_range(n,m)
        G = create_random_weighted(n,m)
        rr = reciprocity_ratio(G)
    #    print("reciprocity reatio",rr)
    #    G = ig.Graph.Full(10,directed =True)
    #    print(G.ecount())
    #    print(G.density())
    #    print(graph_density(G.vcount(),G.ecount()))
        rr = reciprocity_ratio(G)
        RR = G.reciprocity()
    #    print("reciprocity ratio",rr,RR)  
    #    print("Reciprocity",G.reciprocity())
        change_reciprocity_ratio(G,r)
    #    print("Reciprocity",G.reciprocity())
    #    conducTot = sum(i for i in )
        nodes = G.vs
        nodes[8]
        conductStart = time.time()
        
        Conductance =sum(conductance(G,i,j) for i,j in 
                  it.product(range(G.vcount())[:-1],range(G.vcount())[1:]) )
        print("Conductance = ",Conductance)
        compareCSV["Conductance"] = Conductance
        timeCSV["Conductance"] = Conductance
        
        conductEnd = time.time()
        print("Full Conductance time",conductEnd - conductStart)
        CM = G.get_adjacency()
        CM_w = G.get_adjacency(attribute='weight')
    
        CM = g2t.nested.list_convert(np.array,CM)
    
        G.vs["gate"] = [random.choice(g2t.tableMulti('all')) for i in range(G.vcount())]
        Gates = [g2t.tableMulti(gate) for gate in G.vs['gate']]
        States = g2t.noise(G.vcount())
        TPM = g2t.state2tp(States,Gates,CM)
        labels = ('A', 'B', 'C')
    
        network = pyphi.Network(TPM, connectivity_matrix=CM)#,
        #                        node_labels=labels)
        state = tuple(random.choice(TPM))
        
        ### In this case, we want the ΦΦ of the entire network, 
        ### so we simply include every node in the network in our subsystem
        
        node_indices = (i for i in range(G.vcount()))
        try:
            subsystem = pyphi.Subsystem(network, state, node_indices)
            
            ### Now we use big_phi() function to compute the ΦΦ of our subsystem:
            big_phi_time1 = time.time()
        
            big_phi = pyphi.compute.big_phi(subsystem)
        except:
            pass
        big_phi_time2 = time.time()
        big_phi_time = big_phi_time2 - big_phi_time1
        print("BIG PHI = ",big_phi)
        print("Calculating Big Phi took: ", big_phi_time)
        
        compareCSV["Phi"] = big_phi
        timeCSV["Phi"] = big_phi_time
        print("\n")
    
    with open('compare.csv', 'w', newline='') as csvfile:
        fieldnames = ['Phi', 'Conductance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
        for dic in compare_list:
            writer.writerow(dic)
        
    with open('time.csv', 'w', newline='') as csvfile:
        fieldnames = ['Phi', 'Conductance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
        for dic in time_list:
            writer.writerow(dic)
        