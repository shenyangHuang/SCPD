import sys
sys.path.append('../')

from datasets import SBM_loader


import numpy as np 
import random
import os
import pylab as plt
from util import normal_util
import timeit
import rrcf
import pandas as pd
from sklearn.preprocessing import normalize
import metrics





'''
NOTE: SPOTLIGHT assumes node ordering persists over time
generate a source or a destination dictionary
G: the graph at snapshot 0
p: probability of sampling a node into the dictionary

return dict: a dictionary of selected sources or destinations for a subgraph
'''
def make_src_dict(G, p):
    out_dict = {}
    for node in G.nodes():
        if (random.random() <= p):
            out_dict[node] = 1
    return out_dict


'''
main algorithm for SPOTLIGHT
G_times: a list of networkx graphs for each snapshot in order
K: the number of subgraphs to track. 
p: probability of sampling a node into the source
q: probability of sampling a node into the destination

return a list of SPOTLIGHT embeddings (np arrays) for each snapshot
'''
def SPOTLIGHT(G_times, K, p, q):

    '''
    initialize K spotlight sketches at step 0
    '''
    src_dicts = []
    dst_dicts = []
    for _ in range(K):
        src_dicts.append(make_src_dict(G_times[0], p))
        dst_dicts.append(make_src_dict(G_times[0], q))

    sl_embs = []
    for G in G_times:
        sl_emb = np.zeros(K, )
        for u,v,w in G.edges.data("weight", default=1):
            for i in range(len(src_dicts)):
                if (u in src_dicts[i] and v in dst_dicts[i]):
                    sl_emb[i] += w 
        sl_embs.append(sl_emb)

    return sl_embs


def rrcf_offline(X, num_trees=50, tree_size=50):
    n = len(X)
    X = np.asarray(X)
    sample_size_range = (n // tree_size, tree_size)

    # Construct forest
    forest = []
    while len(forest) < num_trees:
        # Select random subsets of points uniformly
        ixs = np.random.choice(n, size=sample_size_range,
                               replace=False)
        # Add sampled trees to forest
        trees = [rrcf.RCTree(X[ix], index_labels=ix)
                 for ix in ixs]
        forest.extend(trees)

    # Compute average CoDisp
    avg_codisp = pd.Series(0.0, index=np.arange(n))
    index = np.zeros(n)
    for tree in forest:
        codisp = pd.Series({leaf : tree.codisp(leaf)
                           for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)
    avg_codisp /= index
    avg_codisp = avg_codisp.tolist()
    return avg_codisp


def run_SPOTLIGHT(fname, K=50, window = 5, percent_ranked= 0.05, use_rrcf=True, seed=0):
    random.seed(seed)
    edgefile = "../datasets/multi_SBM/" + fname + ".txt"
    print (edgefile)
    if (not os.path.isfile(edgefile)):
        edgefile = fname + ".txt"

    p = 0.2
    q = 0.2

    G_times = SBM_loader.load_temporarl_edgelist(edgefile, draw=False)
    start = timeit.default_timer()
    sl_embs = SPOTLIGHT(G_times, K, p, q)
    end = timeit.default_timer()
    sl_time = end-start
    print ('SPOTLIGHT time: '+str(sl_time)+'\n')
    normal_util.save_object(sl_embs, "spotlight" + str(K) + fname + ".pkl")


    if (use_rrcf):
        start = timeit.default_timer()
        num_trees = 50
        tree_size = 151
        scores = rrcf_offline(sl_embs, num_trees=num_trees, tree_size=tree_size)
        end = timeit.default_timer()
        a_time = end-start
        print ('rrcf time: '+str(a_time)+'\n')
        scores = np.asarray(scores)
        num_ranked = int(scores.shape[0]*percent_ranked)
        outliers = scores.argsort()[-num_ranked:][::-1]
        outliers.sort()

    else:
        start = timeit.default_timer()
        scores, outliers = simple_detector(sl_embs)
        end = timeit.default_timer()
        a_time = end-start
        print ('sum predictor time: '+str(a_time)+'\n')
    return outliers, sl_time

def find_anomalies(scores, percent_ranked, initial_window):
    scores = np.array(scores)
    for i in range(initial_window+1):
        scores[i] = 0        #up to initial window + 1 are not considered anomalies. +1 is because of difference score
    num_ranked = int(round(len(scores) * percent_ranked))
    outliers = scores.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return outliers



def simple_detector(sl_embs, plot=False):
    sums = [np.sum(sl) for sl in sl_embs]

    diffs = [0]
    for i in range(1, len(sums)):
        diffs.append(sums[i]-sums[i-1])

    events = find_anomalies(diffs, 0.05, 10)
    scores = diffs
    
    plt.savefig('simple_diff.pdf')
    plt.close()
    return scores, events




    

if __name__ == '__main__':
    fname = "SBM1000"
    use_rrcf = False
    K=50

    if (use_rrcf):
        print ("using robust random cut forest")
    else:
        print ("using simple sum predictor")



    real_events=[16,31,61,76,91,106,136] 
    accus = []
    sl_times = []
    runs = 5
    seeds = list(range(runs))
    for i in range(runs):
        anomalies, sl_time = run_SPOTLIGHT(fname, K=K, use_rrcf=use_rrcf, seed=seeds[i])
        accu = metrics.compute_accuracy(anomalies, real_events)
        accus.append(accu)
        sl_times.append(sl_time)

    accus = np.asarray(accus)
    sl_times = np.asarray(sl_times)
    print (" the mean accuracy is : ", np.mean(accus))
    print (" the std is : ",  np.std(accus))

    print (" the mean spotlight time is : ", np.mean(sl_times))
    print (" the std is : ",  np.std(sl_times))





