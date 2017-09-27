# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:36:05 2017

@author: scram
"""
import numpy as np
import itertools as it
from collections import Counter
import nested
import string

def table(string):
    string = string.lower()
    gates ={
            "or":{
                    (0,0):0,
                    (0,1):1,
                    (1,0):1,
                    (1,1):1
                  },
            "and":{
                    (0,0):0,
                    (0,1):0,
                    (1,0):0,
                    (1,1):1
                  },
            "xor":{
                    (0,0):0,
                    (0,1):1,
                    (1,0):1,
                    (1,1):0
                  }
            }
    return gates[string]

def turn(matrix):
    shape = matrix.shape
    if len(shape) ==1:
#        shape = (shape[0],0)
        matrix = np.array([matrix])
    return np.reshape(matrix.T,matrix.shape[::-1])

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

def states_list(node_num,states = 2):
    state_vec = []
    state_num = states**node_num
    for i in range(state_num):
        forB = index2state_LOLI(i,node_num,states=states)
        state_vec.append([int(i) for i in forB])
    return state_vec


def vec_combine(vec,vecInsert,indexInsert,typeOut=list,axis=0):
    vec_a = np.array(vec)
    vecInsert_a = np.array(vecInsert)
    insert_shape = vecInsert_a.shape
    vec_shape = vec_a.shape
    vec_a = np.array(vec*insert_shape[0])
    vec_a = turn(vec_a)
    vecInsert_a = np.array(vecInsert*vec_shape[0])
    vec_a = np.insert(vec_a,indexInsert,vecInsert_a,axis=axis)
    
    vec_a = turn(vec_a)
    set_len = len(set(map(tuple,vec_a)))
    if len(vec_a)==set_len:
        return nested.map_list(typeOut,vec_a)
    else:
        vec = vec[::-1]
        vec_a = np.array(vec)
        vecInsert_a = np.array(vecInsert)
        insert_shape = vecInsert_a.shape
        vec_shape = vec_a.shape
        vec_a = np.array(vec*insert_shape[0])
        vec_a = turn(vec_a)
        vecInsert_a = np.array(vecInsert*vec_shape[0])
        vec_a = np.insert(vec_a,indexInsert,vecInsert_a,axis=axis)
        vec_a = np.insert(vec_a,indexInsert,vecInsert_a,axis=axis)
        
        vec_a = turn(vec_a)
        
        return nested.map_list(typeOut,vec_a)

def vec_combine(vec,vecInsert,indexInsert,typeOut=list,axis=0):
    vec_a = np.array(vec)
    vecInsert_a = np.array(vecInsert)
    insert_shape = vecInsert_a.shape
    vec_shape = vec_a.shape
    vec_a = np.array(vec*insert_shape[0])
    vec_a = turn(vec_a)
    vecIplus = vecInsert.copy()
    print(vecIplus)
    print(vec_shape,insert_shape)
    for i in range(vec_shape[0]-1):
        if i%2==0:
            print(i,"I1")
            vecIplus+=vecInsert[::-1]
        else:
            print(i,"I2")
            vecIplus+=vecInsert
        print(i,vecIplus)
    vecInsert_a = np.array(vecIplus)
    print(vecInsert_a,"VECAAAA",len(vecInsert_a),vecInsert*vec_shape[0])
#    vecInsert_a = np.array(vecInsert*vec_shape[0])
    vec_a = np.insert(vec_a,indexInsert,vecInsert_a,axis=axis)
    
    vec_a = turn(vec_a)
    
    return nested.map_list(typeOut,vec_a)

        
def past(current_state,table,index,noise=[0,1]):
    past_states =[]
    for inputs,output in table.items():
        print(output,inputs,current_state[index])
        if output == current_state[index]:
            past_states.append(list(inputs))
    print(past_states)
    past_states = vec_combine(past_states,noise,index)
    print(past_states)
    past_prob = 1/len(past_states)
    return np.array([past_prob if i in past_states else 0 for i in states_list(len(current_state))])

def pastUC(current_state,states = 2):
    #check states is right
    if len(set(current_state)) <= states:
        num_possible_states = states**len(current_state)
        return np.array([1/num_possible_states]*num_possible_states)
    else:
        print("States in Current State Greater than State Number")

# EMD Calculations        
def hamming(i_state,j_state):
    d = 0
    for x,y in zip(i_state,j_state):
        d += int(x!=y)
    return d

def distance_matrix(states):
    i2s = np.vectorize(index2state_LOLI)
    hamming_v = np.vectorize(hamming)
    return np.fromfunction(lambda i,j: hamming_v(i2s(i,3),i2s(j,3)),(len(states),len(states)))

def subMat4dif(dif,mat):
    mask = dif>0
    submat=mat[mask]
    submat = submat[:,np.all(submat,axis=0)]
    return submat,dif[dif>0],dif[dif<=0]

def subPhi(dists, holes, piles):
    hole_sort = np.argsort(holes)[::-1]
    dif_vec = []
    costs = []
#    print(dists,"Dist",holes,"holes",piles,"Piles",hole_sort)
#    print(hole_sort)
#    print(dists[hole_sort],"Dist_sort",holes[hole_sort],"Hole_sort")
    for hole,dist in it.zip_longest(holes[hole_sort],dists[hole_sort]):
        print(dist, hole, piles,"START")
        dist_min = np.min(dist)
        min_index = list(dist).index(dist_min)
#        print(dist)
#        print(piles)
#        print(dif_vec)
        min_pile = piles[min_index]
        dif = hole + min_pile
        dif_vec.append(dif)
        cost = (hole-dif)*dist_min
        costs.append(cost)
#        print(cost,"CURRENT COST")
#        print(dif,"CURRENT DIFFERENCE")
#        print(piles)
        print(dist, hole, min_pile, dif, dist_min, min_index)
#        if dif >=0:
        dists = np.delete(dists[hole_sort],min_index,axis=1)
        piles = np.delete(piles,min_index)
#        else:
#            try:
#                piles[min_index] = dif
#            except IndexError:
#                piles = np.array([dif])
                
#        print(dists.shape,"SHAPE")
#        test = np.arange(np.prod(dists.shape)).reshape(dists.shape)
#        print(dists.shape)
#        print(test)
#        print(test[hole_sort])
        
#    print(dists,"DISTS")
    print(costs,"COST")
    print(dif_vec,"DIF")
#    print(piles,"PILES")
    dif_vec = np.array(dif_vec)
    if np.sum(np.absolute(dif_vec)):
        print("|||||||||||||||One more time||||||||||||||||||")
        costs2 =(subPhi(dists, dif_vec, piles))
        print(costs,costs2,"$$$$$$$$$$$$$COST$$$$$$$$$$$$$$$$")
        costs = np.append(costs,costs2)
        print(costs)
        return np.sum(costs)
    else:
        return np.sum(costs)


def subPhi(dists, holes, piles,recursed = False ):
    hole_sort = np.argsort(holes)[::-1]
    pile_sort = np.argsort(piles)
    test = np.outer(piles*-1,holes)*100
    dif_vec = []
    costs = []
#    print(dists,holes,piles)
    
    for hole,dist in it.zip_longest(holes[hole_sort],dists[hole_sort]):
        dist_min = np.min(dist)
        min_pile = piles[list(dist).index(dist_min)]
        dif = hole + min_pile
        dif_vec.append(dif)
        costs.append((hole-dif)*dist_min)
#        print(dist,hole, min_pile,dif,dist_min)
    dif_vec = np.array(dif_vec)
    costs = np.sum(costs)
    costs1 = costs
    if np.sum(np.absolute(dif_vec)):
        recursed =True
        return costs+subPhi(dists, dif_vec, piles,recursed=recursed)
    else:
        if not recursed:
            costs1 = 0
        return costs-costs1

# Future functions
def gate2ucF(gate):
    return {k:v/len(gate) for k,v in Counter(gate.values()).items()} 

#@np.vectorize
def apply_dict(dic):
    def dict_ufunc(i):
        d_return = dic[i]
        return d_return
    return dict_ufunc

def apply_dict_array(dic):
    def dict_ufunc(vec):
        dic2 = dic[:len(vec)]
        return [d[v] for d,v in zip(dic2,vec)]
    return dict_ufunc

def uc_future(possible_states,node_state_prob):
    state_array = apply_dict_array(node_state_prob)
    return np.apply_along_axis(state_array,1,possible_states)

def future(possible_states,node_state_prob,index):
    state_array = apply_dict_array(node_state_prob)
    out = np.apply_along_axis(state_array,1,possible_states)
    arrayCounter = np.vectorize(apply_dict(Counter(possible_states[:,index])))
    divisor = np.apply_along_axis(arrayCounter,0, possible_states[:,index])
    return np.divide(out.prod(1),divisor)

def filterGates(gates,index):
    gate = gates[index]
    x = len(gates)-1
#    gate_null = [{k:1*len(gate)  for k,v in gate.items()}]*4
#    print(gate_null,"gate_null",len(gate_null))
    gates_null = [{k:1 for k,v in gate.items()}]*x
    gates_null.insert(index,gate)
    return gates_null


c_state = [1,0,0]
cm = np.array([])
gatetypes = ["OR","AND","XOR","OR","OR","OR"]
gates = [table(i) for i in gatetypes]
ego_index = 2


print(gates)
states = noise(len(c_state))
print(states)

past_a = past(c_state,gates[ego_index],ego_index)
print(turn(past_a),"PAST?")
pastUC_a = pastUC(c_state)
print(turn(pastUC_a))

dist_mat = distance_matrix(states)
for i in dist_mat:
    print(i)
    
dif_past = pastUC_a - past_a
print(turn(pastUC_a),"\n - \n",turn(past_a),"\n = \n",turn(dif_past))
print('\n')

hole_distP, hole_sizeP, pilesP  = subMat4dif(dif_past,dist_mat)
print(hole_distP)
print(hole_sizeP)
print(pilesP)

print("\nFuture Phi Calculation")
node_ucF_prob = list(map(gate2ucF,gates))
print(node_ucF_prob)
for label,uc in zip(list(string.ascii_lowercase),node_ucF_prob):
    print(label,uc)
print('\n')
ucF_m = uc_future(states,node_ucF_prob)
print(ucF_m*4,"Unconstrained Future Matrix")
futureUC_v = ucF_m.prod(1)
print(turn(futureUC_v),"Unconstrained Future Vector")

noise_a = noise(2)
print(noise_a)


node_F_prob = filterGates(node_ucF_prob,ego_index)
print(node_F_prob,"Future Probs")
future = future(states,node_F_prob,ego_index)
print(turn(future),"Future\n")
dif_future = futureUC_v-future
print(turn(dif_future),"Dif Future\n")
hole_dist, hole_size, piles  = subMat4dif(dif_future,dist_mat)

print("_________________\nEMD PAST\n_________________")
phi_past = subPhi(hole_distP, hole_sizeP, pilesP)

print(phi_past,"PHI PAST")
print("_________________\nEMD FUTURE\n_________________")
phi_future = subPhi(hole_dist, hole_size, piles)
print(phi_future,"FUTURE PHI") 
#for i,j in zip(dif_future,dist_mat):
#    space = 10-len(str(i))
#    spacing = " "*space
#    print(spacing,i,j)
#print(hole_dist)
#print(hole_distP)

   
    

#print("_________________\nEMD Test\n_________________")
#print(futureUC_v,"FUTRUE_UC")
#print(future*-1,"FUTURE")
##dist_mat,"dist_mat"
#phi_future = subPhi(dist_mat, future, futureUC_v*-1)
#print(phi_future)
    