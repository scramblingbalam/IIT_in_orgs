#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:51:40 2017

@author: cdrayton
"""
import pyphi
import numpy as np
import nested
import time
def tableMulti(string):
    string = string.lower()
    gates ={
            "or" :lambda x: 1 if sum(x)>0.0 else 0,
            "and":lambda x: 0 if np.mean(x)< 1 else 1,
            "xor":lambda x: 1 if sum(x) ==1 else 0
            }
    return gates[string]

def index2state_LOLI(i,node_num, states = 2):
    i=int(i)
    forB = "{0:b}".format(i)
    forB = forB[::-1]
    while len(forB) < node_num:
        forB = forB + "0"
    return forB

def noise(node_num,states = 2):
    state_vec = []
    state_num = states**node_num
    for i in range(state_num):
        forB = index2state_LOLI(i,node_num,states=states)
        state_vec.append(np.array([int(i) for i in forB]))
    return np.array(state_vec)

print(len(noise(6)))
print(dir(pyphi.examples))
networkLib = pyphi.examples.fig1a()
print(dir(networkLib))

tpmLib = networkLib.tpm
cmLib = networkLib.connectivity_matrix

cmWeb = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
 ])
    
tpmWeb = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0]
])
#print(tpmLib)
#tpmLib = nested.list_convert(list,tpmLib)
tpmLib = nested.map_list(int, tpmLib)  
tpmLib = nested.flatXn(tpmLib,5)
#tpmWeb = nested.list_convert(list,tpmWeb)

print(type(tpmLib))
#print(tpmLib)
print(sum(tpmLib == tpmWeb))
print(sum(cmLib == cmWeb))
print(cmLib,"\n")
print(cmLib[:,0])

gatetypes =['OR','AND','XOR','OR','OR','OR']
gates = [tableMulti(i) for i in gatetypes]

test_vec = [0, 0, 0, 0, 0, 0]
print(gates[0](test_vec))
print(gates[1](test_vec))
print(gates[2](test_vec))



states = noise(6)
statesTest = []
for nodes in states:
    vec = []
    for n,j in enumerate(nodes):
        vec_in = nodes[cmLib[:,n]>=1]
        multiplier = cmLib[:,n][cmLib[:,n]>=1]
        vec.append(gates[n](vec_in*multiplier))
    statesTest.append(np.array(vec))
tpmTest = np.array(statesTest)

#@np.vectorize
def state2tp(state,CM=None):
    for n,j in enumerate(state):
        vec_in = states[CM[:,n]>=1]
        multiplier = CM[:,n][CM[:,n]>=1]
        vec.append(gates[n](vec_in*multiplier))
    return statesTest.append(np.array(vec))

state2tpm = np.vectorize(state2tp(states,CM=cmLib))
tpmTestVec = np.apply_along_axis(state2tpm,1, states)
print(sum(tpmTest==tpmTestVec))


#print(tpmTest == tpmWeb)
print(sum(tpmTest == tpmWeb))
#print(sum(statesTest == tpmLib))
tpmTestBool = tpmTest == tpmWeb  
        
#for x,y in zip(tpmWeb[:,1],tpmTest[:,1]):print(x,y)


network = pyphi.Network(tpmTest, connectivity_matrix=cmLib)#,
#                        node_labels=labels)
state = (1, 0, 0, 0, 0, 0)

### In this case, we want the ΦΦ of the entire network, 
### so we simply include every node in the network in our subsystem

node_indices = (0, 1, 2)
subsystem = pyphi.Subsystem(network, state, node_indices)

### Now we use big_phi() function to compute the ΦΦ of our subsystem:
big_phi_time1 = time.time()
big_phi = pyphi.compute.big_phi(subsystem)
big_phi_time2 = time.time()
big_phi_time = big_phi_time2 - big_phi_time1
print("BIG PHI = ",big_phi)
print("Calculating Big Phi took: ", big_phi_time)