#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:51:40 2017

@author: cdrayton
"""

import numpy as np

def tableMulti(string):
    string = string.lower()
    gates ={
            "or" :lambda x: 0 if sum(x)<1.0 else 1,#or less than 1
            "and":lambda x: 0 if np.mean(x)< 1 else 1,#YES
#            "xor":lambda x: 1 if sum(x) == 1 else 0#
            "nor":lambda x: 1 if sum(x)<1.0 else 0,
            "nand":lambda x: 1 if np.mean(x)< 1 else 0
            }
    if string == 'all':
        return list(gates.keys())
    elif string in gates.keys():
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


def states2tpm(States,Gates,CM):
    tpmTest = []
    for nodes in States:
        vec = []
        for n,j in enumerate(nodes):
            vec_in = nodes[CM[:,n]>=1]
            multiplier = CM[:,n][CM[:,n]>=1]
            inputs = vec_in * multiplier
            if inputs.any():
                vec.append(Gates[n](inputs))
            else:
                vec.append(j)
        tpmTest.append(np.array(vec))
    return np.array(tpmTest)

