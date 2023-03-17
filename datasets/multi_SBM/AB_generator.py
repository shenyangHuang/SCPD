'''
dataset generator for synthetic SBM dataset
'''
import collections
import numpy as np
import pylab as plt
import random
import scipy.stats as stats
import dateutil.parser as dparser
import re
import copy
import networkx as nx
from networkx.utils import *
from networkx.generators.random_graphs import barabasi_albert_graph

'''
t u v w
'''
def to_edgelist(G_times, outfile):

    outfile = open(outfile,"w")
    tdx = 0
    for G in G_times:
        
        for (u,v) in G.edges:
            outfile.write(str(tdx) + "," + str(u) + "," + str(v) + "\n")
        tdx = tdx + 1
    outfile.close()
    print("write successful")



#m is the Number of edges to attach from a new node to existing node
def AB_snapshot(G_prev, alpha, m):

    G_t = G_prev.copy()
    n = len(G_t)
    G_new = nx.barabasi_albert_graph(n, m)
    if (alpha == 1.0):
        return G_new

    for i in range(0,n):
        for j in range(i+1,n):
            prob = random.uniform(0, 1)
            if (prob <= alpha):
                if (G_new.has_edge(i,j) and not G_t.has_edge(i, j)):
                    G_t.add_edge(i,j)
                if (not G_new.has_edge(i,j) and G_t.has_edge(i, j)):
                    G_t.remove_edge(i,j)
   
    return G_t


'''
generate just change points
'''

def generate_AB_ChangePoint(m_list, alpha, n=500, seed=0, outname="hi", cps=[15,30,60,75,90,105,135]):

    random.seed(seed)       
    np.random.seed(seed)

    if (len(m_list) < 8):
        raise Exception("there should be 8 m values for 7 change points")

    fname = outname + ".txt"

    maxt = 150
    p_idx = 0
    G_0 = nx.barabasi_albert_graph(n, m_list[p_idx])
    G_0 = nx.Graph(G_0)
    #plot_degree_dis(G_0, 0, outname=outname)
    G_t = G_0
    G_times = []
    G_times.append(G_t)

    for t in range(maxt):
        if (t in cps):
            p_idx = p_idx + 1
            p = m_list[p_idx]
            G_t = AB_snapshot(G_t, alpha, p)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

        else:
            p = m_list[p_idx]
            G_t = AB_snapshot(G_t, alpha, p)
            G_times.append(G_t)
            print ("generating " + str(t), end="\r")

    #write the entire history of snapshots
    to_edgelist(G_times, fname)


def main():
    alpha = 1.0
    seed = 1
    m_list = np.arange(1,9,1)
    m_list = list(m_list)
    n = 1000
    outname = "BA"+str(n)
    generate_AB_ChangePoint(m_list, alpha, n=n, seed=seed, outname=outname)

    


if __name__ == "__main__":
    main()
