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

'''
blocks is an array of sizes
inter is the inter community probability
intra is the intra community probability
'''
def construct_SBM_block(blocks, inter, intra):
    probs = []
    for i in range(len(blocks)):
        prob = [inter]*len(blocks)
        prob[i] = intra
        probs.append(prob)
    return probs

def SBM_snapshot(G_prev, alpha, sizes, probs):

    G_t = G_prev.copy()
    nodelist = list(range(0,sum(sizes)))
    G_new = nx.stochastic_block_model(sizes, probs, nodelist=nodelist)
    n = len(G_t)
    if (alpha == 1.0):
        return G_new

    for i in range(0,n):
        for j in range(i+1,n):
            #randomly decide if remain the same or resample
            #remain the same if prob > alpha
            prob = random.uniform(0, 1)
            if (prob <= alpha):
                if (G_new.has_edge(i,j) and not G_t.has_edge(i, j)):
                    G_t.add_edge(i,j)
                if (not G_new.has_edge(i,j) and G_t.has_edge(i, j)):
                    G_t.remove_edge(i,j)
    return G_t


def add_noise(p, G_t):
    n = len(G_t)
    for i in range(0,n):
        for j in range(i+1,n):
            #flip the existence of an edge
            prob = random.uniform(0, 1)
            if (prob <= p):
                if (G_t.has_edge(i,j)):
                    G_t.remove_edge(i, j)
                else:
                    G_t.add_edge(i,j)
    return G_t



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



'''
change points are noise instead, increasing and decreasing noise
'''
def generate_noise_change(maxt, cps, N, inter_prob, intra_prob, alpha, n_increment, noise_p=0, seed=1, outname="none"):

    random.seed(seed)       
    np.random.seed(seed)

    if (outname != "none"):
        fname = outname+".txt"
    else:
        fname = "eventChange_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(noise_p) + "_" + str(alpha) + ".txt"

    sizes_2 = [int(N/4)]*4
    probs_2 = construct_SBM_block(sizes_2, inter_prob, intra_prob)

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
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            noise_p = noise_p + n_increment
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

'''
generate 1,2 attributes, 1 as male, 2 as female
save as an array for each timestep
'''
def gender_attr(sizes, n, half=False):
    attr = np.zeros(n)
    idx = 0
    isMale=True  #only used when half=False
    for size in sizes:
        min_idx = min(idx + size, n)
        if (half):
            for k in range(min_idx - idx):
                p = random.random()
                if (p > 0.5):
                    attr[idx + k] = 1
                else:
                    attr[idx + k] = 0
        else:
            if (isMale):
                attr[idx: min_idx] = 0  #set as male
            else:
                attr[idx: min_idx] = 1  #set as female
            isMale = not isMale
        idx += size

    attr = attr.astype('double')
    return attr


'''
input: a list of attribute arrays for each snapshot
'''
def write_attr(attr_arrs, out_dir=""):

    for t in range(len(attr_arrs)):
        mdic = {}
        mdic[u'labels'] = attr_arrs[t]
        hdf5storage.write(mdic, ".", out_dir + "labels_"+ str(t+1) + ".mat", matlab_compatible=True)





def generate_SBM_attributes(maxt, cps, N, inter_prob, intra_prob, alpha, seed=1, outname="none"):
    random.seed(seed)       
    np.random.seed(seed)

    if (outname != "none"):
        fname = outname+".txt"
    else:
        fname = "SBMattr_"+ str(inter_prob)+ "_"+ str(intra_prob) + "_" + str(noise_p) + "_" + str(alpha) + ".txt"

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
    attr_arrs = []
    attr_arrs.append(gender_attr(sizes, N, half=isEvent))
    attr_change = []
    attr_change.append(isEvent)


    for t in range(maxt):
        if (t in cps):

            if (isEvent):
                isEvent = False
                attr_change.append(isEvent)
                # events change attributes
                G_t = SBM_snapshot(G_t, alpha, sizes, np.asarray(probs))
                attr_arrs.append(gender_attr(sizes, G_t.number_of_nodes(), half=isEvent))
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")
            else:
                isEvent = True
                attr_change.append(isEvent)
                # change points only change the communities, not attributes
                if ((list_idx+1) > len(list_sizes)-1):
                    list_idx = 0
                else:
                    list_idx = list_idx + 1
                sizes = list_sizes[list_idx]
                probs = list_probs[list_idx]
                G_t = SBM_snapshot(G_t, alpha, sizes, probs)
                attr_arrs.append(gender_attr(sizes, G_t.number_of_nodes(), half=isEvent))
                G_times.append(G_t)
                print ("generating " + str(t), end="\r")

        else:
            G_t = SBM_snapshot(G_t, alpha, sizes, probs)
            attr_arrs.append(gender_attr(sizes, G_t.number_of_nodes(), half=isEvent))
            G_times.append(G_t)
            attr_change.append(isEvent)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)
    write_attr(attr_arrs, out_dir="../RAW/SBM_attr/")







def main():
    '''
    run this block of code to generate SBM (no attributes)
    '''
    alpha = 1.0
    inter_prob = 0.005
    intra_prob = 0.030
    increment = 0.010
    seed = 1
    random.seed(seed)        # or any integer
    np.random.seed(seed)
    cps = [15,30,60,75,90,105,135]
    maxt = 150
    N = 1000
    outname="SBM"+str(N)
    generate_event_change(maxt, cps, N, inter_prob, intra_prob, alpha, increment, noise_p=0, seed=seed, outname=outname, NI=False)


    '''
    run this block of code to run SBM with attributes
    '''
    # alpha = 1.0
    # inter_prob = 0.005
    # intra_prob = 0.030
    # increment = 0.010
    # seed = 29
    # random.seed(seed)        # or any integer
    # np.random.seed(seed)
    # cps = [15,30,60,75,90,105,135]
    # maxt = 150
    # N = 1000
    # outname="SBM"+str(N)+"attr"
    # generate_SBM_attributes(maxt, cps, N, inter_prob, intra_prob, alpha, seed=seed, outname=outname)

if __name__ == "__main__":
    main()
