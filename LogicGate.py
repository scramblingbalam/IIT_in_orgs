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
    combined = []
    for i in vec:
        for j in vecInsert:
            i_copy = i.copy()
            i_copy.insert(indexInsert,j)
            combined.append(np.array(i_copy))
    combined = np.vstack(combined)
#    print("\nCOMBINED",combined,"\n")
    return nested.map_list(typeOut,combined)
        
def past(current_state,truth_table,index,noise=[0,1]):
    past_states =[]
    for inputs,output in truth_table.items():
#        print(output,inputs,current_state[index])
        if output == current_state[index]:
            past_states.append(list(inputs))
#    print("VecCombine Inputs\n",past_states)
    past_states = vec_combine(past_states,noise,index)
#    print("VecCombine Outputs\n",past_states)
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


def subPhi(dists, holes, piles,recursed = False ):
    hole_sort = np.argsort(holes)[::-1]
    pile_sort = np.argsort(piles)
    dif_vec = []
    costs = []
#    print("SubPhi Incoming Variables",dists,holes,piles)
    
    for hole,dist in it.zip_longest(holes[hole_sort],dists[hole_sort]):
        dist_min = np.min(dist)
        min_pile = piles[list(dist).index(dist_min)]
        dif = hole + min_pile
        dif_vec.append(dif)
        costs.append((hole-dif)*dist_min)
#        print(dist,hole, min_pile,dif,dist_min)
    dif_vec = np.array(dif_vec)
    costs = np.sum(np.array(costs))
    costs1 = costs
#    print(costs,"COSTS",dif_vec,recursed,"\n")
    if np.sum(np.absolute(dif_vec))>0.000000001:
        recursed =True
        return np.sum(costs+subPhi(dists, dif_vec, piles,recursed=recursed))
    else:
        if not recursed:
            costs1 = 0
#        print("RETURNING RECURSE",costs,costs1)
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
ego_index = 0
#print(gates)
states = noise(len(c_state))
#print(states)

dist_mat = distance_matrix(states)
print(dist_mat,"DISTANCE MATRIX\n")

print("\nPast Phi Calculation")
past_a = past(c_state,gates[ego_index],ego_index)
pastUC_a = pastUC(c_state)
dif_past = pastUC_a - past_a
print(turn(pastUC_a),
      "Unconstrained Past Vector",
      "\n - \n",
      turn(past_a),
      "Past constrained",
      "\n = \n",
      turn(dif_past),
      "Dif Past\n")
print('\n')

hole_distP, hole_sizeP, pilesP  = subMat4dif(dif_past,dist_mat)
print(hole_distP)
print(hole_sizeP)
print(pilesP)

print("\nFuture Phi Calculation\n")
node_ucF_prob = list(map(gate2ucF,gates))
print(node_ucF_prob)
for label,uc in zip(list(string.ascii_lowercase),node_ucF_prob):
    print(label,uc)
print('\n')
ucF_m = uc_future(states,node_ucF_prob)
#print(ucF_m*4,"Unconstrained Future Matrix")
futureUC_v = ucF_m.prod(1)

node_F_prob = filterGates(node_ucF_prob,ego_index)
future_a = future(states,node_F_prob,ego_index)
dif_future = futureUC_v-future_a

print(turn(futureUC_v),
      "Unconstrained Future Vector",
      "\n - \n",
      turn(future_a),
      "Future constrained",
      "\n = \n",
      turn(dif_future),
      "Dif Future\n")

hole_distF, hole_sizeF, pilesF  = subMat4dif(dif_future,dist_mat)

print("_________________\nEMD PAST\n_________________")
phi_past = subPhi(hole_distP, hole_sizeP, pilesP)
#print("Past inputs\n",hole_distP, hole_sizeP, pilesP)
print(phi_past,"PHI PAST")

print("_________________\nEMD FUTURE\n_________________")
phi_future = subPhi(hole_distF, hole_sizeF, pilesF)
#print("Futrue inputs\n",hole_distF, hole_sizeF, pilesF)
print(phi_future,"FUTURE PHI\n") 

def PastPhi(ego_index,gates,all_states,dist_matrix):
    pastC_a = past(c_state,gates[ego_index],ego_index)
    pastUC_a = pastUC(c_state)
#    print("PAST PHI DIF\nP ",pastUC_a,"\nUP",pastC_a)
    dif_past = pastUC_a - pastC_a
#    print("PAST PHI SubMat input\n",dif_past,"\n",dist_matrix)
    out = subMat4dif(dif_past,dist_matrix)
    return subPhi(out[0],out[1],out[2])

def FuturePhi(ego_index,gates,all_states,dist_matrix):
    nodeUC_prob =list(map(gate2ucF,gates))
    futureUC = uc_future(all_states,list(map(gate2ucF,gates))).prod(1)
    node_F_prob = filterGates(nodeUC_prob,ego_index)
    futureConstrained = future(all_states,node_F_prob,ego_index)
    dif_future = futureUC-futureConstrained
    out = subMat4dif(dif_future,dist_matrix)
    return subPhi(out[0],out[1],out[2])

print(PastPhi(0,gates,states,dist_mat),"PAST PHI Ego=0")
print(FuturePhi(0,gates,states,dist_mat),"FUTURE PHI Ego=0\n__________________")
print(PastPhi(1,gates,states,dist_mat),"PAST PHI Ego=1")
print(FuturePhi(1,gates,states,dist_mat),"FUTURE PHI Ego=1\n__________________")
print(PastPhi(2,gates,states,dist_mat),"PAST PHI Ego=2")
print(FuturePhi(2,gates,states,dist_mat),"FUTURE PHI Ego=2\n__________________")