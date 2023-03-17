'''
dataset loader for synthetic SBM dataset
undirected graphs
'''
import numpy as np
import networkx as nx
from matplotlib import pyplot, patches
import pylab as plt
import dateutil.parser as dparser
import re
import hdf5storage
import os

'''
treat each day as a discrete time stamp
'''
def load_temporarl_edgelist(fname, draw=False):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    cur_t = 0

    '''
    t u v
    '''
    G_times = []
    G = nx.Graph()

    for i in range(0, len(lines)):
        line = lines[i]
        values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
        t = int(values[0])
        u = int(values[1])
        v = int(values[2])
        #start a new graph with a new date
        if (t != cur_t):
            G_times.append(G)   #append old graph
            if (draw):
                if (t in [1,17,32,62,77,92,107,137]):
                    draw_adjacency_matrix(G, t,  node_order=None, partitions=[], colors=[])
            G = nx.Graph()  #create new graph
            cur_t = t 
        G.add_edge(u, v) 
    G_times.append(G)
    print ("maximum time stamp is " + str(len(G_times)))
    return G_times


def draw_adjacency_matrix(G, t,  node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    # degree_list = sorted(G.degree, key=lambda x: x[1], reverse=True)
    # sortedList = []
    # for u,degree in degree_list:
    #     sortedList.append(u)
    nodelist = list(range(0,500))


    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=nodelist)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imsave(str(t)+".pdf", adjacency_matrix,
                  cmap="Greys")
    pyplot.close()
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)



def export_adj(fname, out_dir, max_size=1000):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()


    max_time = 0
    current_date = 0
    G = nx.Graph()
    idx = 1

    for i in range(0, len(lines)):
        line = lines[i]
        values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
        if (len(values) < 3):
            continue
        else:
            date_str = int(values[0])
            #start a new graph with a new date
            if (date_str != current_date):
                '''
                write old graph
                '''
                mdic = {}
                nid = 1000
                while (G.number_of_nodes() < max_size):
                    G.add_node(nid)
                    nid += 1
                adj = nx.adjacency_matrix(G).toarray()
                adj = adj.astype('double')
                mdic[u'adj'] = adj
                hdf5storage.write(mdic, ".", out_dir + str(idx) + ".mat", matlab_compatible=True)
                idx += 1
                G = nx.Graph()  #create new graph
                current_date = date_str     #update the current date

            v = int(values[-1])     
            u = int(values[-2])
            G.add_edge(u, v)
    mdic = {}
    adj = nx.adjacency_matrix(G).toarray()
    adj = adj.astype('double')
    mdic[u'adj'] = adj
    hdf5storage.write(mdic, ".", out_dir + str(idx) + ".mat", matlab_compatible=True)
    print ("maximum time stamp is " + str(idx))



def main():
    fname = "multi_SBM/BA5000.txt"
    out_dir = "RAW/BA5000/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    max_size = 5000
    export_adj(fname,out_dir,max_size=max_size)


if __name__ == "__main__":
    main()