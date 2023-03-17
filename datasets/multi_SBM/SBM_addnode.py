import numpy as np
import pylab as plt
import random
import collections
import scipy.stats as stats
import dateutil.parser as dparser
import re
import copy
from scipy.stats import powerlaw
from networkx.algorithms.community.quality import modularity
import networkx as nx
from networkx.utils import *
from networkx import generators
import hdf5storage

def to_edgelist(G_times, outfile):

    outfile = open(outfile,"w")
    tdx = 0
    for G in G_times:
        
        for (u,v) in G.edges:
            outfile.write(str(tdx) + "," + str(u) + "," + str(v) + "\n")
        tdx = tdx + 1
    outfile.close()
    print("write successful")

def construct_SBM_block(blocks, inter, intra):
    probs = []
    for i in range(len(blocks)):
        prob = [inter]*len(blocks)
        prob[i] = intra
        probs.append(prob)
    return probs



def SBM_snapshot(sizes, probs):
    nodelist = list(range(0,sum(sizes)))
    G_new = nx.stochastic_block_model(sizes, probs, nodelist=nodelist)
    return G_new


'''
generate a SBM model with node additions


maxt = # of timesteps
cps = the time steps for the planted anomalies
initN = initial # of nodes
maxN = max # of number of nodes
increN = # of nodes to be added at each change point

#! start from initN nodes and in t_add steps add to maxN
# of nodes to be added per step = (maxN - initN) / t_add

'''
def evolve_SBM(maxt, cps, initN, maxN, increN, inter_prob, intra_prob,increment,seed=1,outname="none"):

    random.seed(seed)       
    np.random.seed(seed)

    if (outname != "none"):
        fname = outname+".txt"
    else:
        fname = "evolveSBM_"+ str(inter_prob)+ "_"+ str(intra_prob) + ".txt"

    #size for all communities at each change point
    sizes = []
    probs = []

    #at step 0, start 400 nodes 0-16 no change, 2 community
    sizes.append([int(initN/2)]*2)
    probs.append(construct_SBM_block([int(initN/2)]*2, inter_prob, intra_prob))
    #at step 16, increase nodes to 600, 3 community  
    sizes.append([int((initN+increN)/3)]*3)
    probs.append(construct_SBM_block([int((initN+increN)/3)]*3, inter_prob, intra_prob))
    #at step 32, increase nodes to 800, 4 community  
    sizes.append([int(maxN/4)]*4)
    probs.append(construct_SBM_block([int(maxN/4)]*4, inter_prob, intra_prob))
    #at step 61, community 3-4 split in half  
    sizes.append([int(maxN/4)]*2 + [int(maxN/8)]*4)
    probs.append(construct_SBM_block([int(maxN/4)]*2 + [int(maxN/8)]*4, inter_prob, intra_prob))
    #at step 76, community 3-4 merge  
    sizes.append([int(maxN/4)]*4)
    probs.append(construct_SBM_block([int(maxN/4)]*4, inter_prob, intra_prob))
    #at step 91, community 1-2 split in half
    sizes.append([int(maxN/8)]*4 + [int(maxN/4)]*2)
    probs.append(construct_SBM_block([int(maxN/8)]*4 + [int(maxN/4)]*2, inter_prob, intra_prob))
    #at step 106, community 1-2 merge 
    sizes.append([int(maxN/4)]*4)
    probs.append(construct_SBM_block([int(maxN/4)]*4, inter_prob, intra_prob))
    #at step 136, event, cross community edges increase
    sizes.append([int(maxN/4)]*4)
    probs.append(construct_SBM_block([int(maxN/4)]*4, inter_prob, intra_prob))

    G_times = []
    G_0=nx.stochastic_block_model(sizes[0], probs[0])
    G_0=nx.Graph(G_0)
    G_times.append(G_0)
    idx = 0
    cur_sizes = sizes[idx]
    cur_probs = probs[idx]

    for t in range(maxt):
        print ("generating " + str(t), end="\r")
        if (t in cps):
            idx += 1
            cur_sizes = sizes[idx]
            cur_probs = probs[idx]
            if (t == 135):
                event_probs = construct_SBM_block(cur_sizes, inter_prob + increment, intra_prob)
                G_times.append(SBM_snapshot(cur_sizes, event_probs))
            else:
                G_times.append(SBM_snapshot(cur_sizes, cur_probs))
        else:
            G_times.append(SBM_snapshot(cur_sizes, cur_probs))
    
    #write the entire history of snapshots
    to_edgelist(G_times, fname)
            







    





'''
generate both change points and events

maxt: the maximum number of time steps
cps: a list of changes points and events, should all be < maxt
N: number of nodes
inter_prob: inter-community connectivity
intra_prob: intra-community connectivity
alpha: what percentage of edges are resampled from the generative model from one step to another
increment: how much increment for inter-community connectivity at each event
NI: increment noise by this amount after each change point
'''

def generate_event_change(maxt, cps, N, inter_prob, intra_prob, alpha, increment, noise_p=0, seed=1, outname="none", NI=False):

    random.seed(seed)       
    np.random.seed(seed)

    if (outname != "none"):
        fname = outname+".txt"
    else:
        fname = "eventChange_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(noise_p) + "_" + str(alpha) + ".txt"

    cps_sizes = []
    cps_probs = []


    sizes_1 = [int(N/2)]*2
    probs_1 = construct_SBM_block(sizes_1, inter_prob, intra_prob)

    sizes_2 = [int(N/4)]*4
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)

    sizes_3 = [int(N/10)]*10
    probs_3 = construct_SBM_block(sizes_3, inter_prob, intra_prob)

    list_sizes = []
    list_sizes.append(sizes_1)
    list_sizes.append(sizes_2)
    list_sizes.append(sizes_3)


    list_probs = []
    list_probs.append(probs_1)
    list_probs.append(probs_2)
    list_probs.append(probs_3)

    list_idx = 1
    isEvent = True 

    sizes = sizes_2
    probs = probs_2
    
    G_0=nx.stochastic_block_model(sizes, probs)
    G_0 = nx.Graph(G_0)
    G_t = G_0
    G_times = []
    G_times.append(G_t)


    '''
    experiment with drastic changes
    '''
    for t in range(maxt):
        if (t in cps):

            if (isEvent):

                copy_probs = copy.deepcopy(probs)
                '''
                change the intercommunity connectivity
                all   0.05 ---> 0.15
                '''
                for i in range(len(copy_probs)):
                    for j in range(len(copy_probs[0])):
                        if (copy_probs[i][j] < intra_prob):
                            copy_probs[i][j] = inter_prob + increment

                G_t = SBM_snapshot(G_t, alpha, sizes, np.asarray(copy_probs))
                if (noise_p > 0):
                    G_t = add_noise(noise_p, G_t)
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
                isEvent = False
            else:
                if (NI):
                    noise_p = noise_p + increment

                if ((list_idx+1) > len(list_sizes)-1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t, alpha, sizes, probs)
                if (noise_p > 0):
                    G_t = add_noise(noise_p, G_t)
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
                isEvent = True

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            if (noise_p > 0):
                G_t = add_noise(noise_p, G_t)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)







def main():
    '''
    generate SBM with evolving # of nodes
    '''
    cps = [15,30,60,75,90,105,135]
    inter_prob = 0.005
    intra_prob = 0.030
    increment = 0.010
    maxt = 150
    initN = 600 #1000 #400
    maxN = 1200 #2000 #800
    increN = 300 #500 #200
    seed = 1

    evolve_SBM(maxt, cps, initN, maxN, increN, inter_prob, intra_prob, increment, seed=seed, outname="none")




if __name__ == "__main__":
    main()
