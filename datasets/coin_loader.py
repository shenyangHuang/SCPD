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
import math

DAY_LEN = 86400

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



'''
#! analyze statistics from the dataset
#* 1). # of unique nodes, 2). # of edges. 3). # of unique edges, 4). # of timestamps 5). min & max of edge weights, 6). recurrence of nodes
'''
def analyze_csv(fname):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    node_dict = {}
    edge_dict = {}
    num_edges = 0
    num_time = 0
    prev_t = "none"
    min_w = 100000
    max_w = 0

    for i in range(1, len(lines)):
        line = lines[i]

        #t,u,v,w
        strs = line.split(",")
        t = strs[0]
        u = strs[1]
        v = strs[2]
        w = float(strs[3].strip())

        # min & max edge weights
        if (w > max_w):
            max_w = w

        if (w < min_w):
            min_w = w

        # count unique time
        if (t != prev_t):
            num_time += 1
            prev_t = t

        #unique nodes
        if (u not in node_dict):
            node_dict[u] = 1
        else:
            node_dict[u] += 1
        
        if (v not in node_dict):
            node_dict[v] = 1
        else:
            node_dict[v] += 1

        #unique edges
        num_edges += 1
        if ((u,v) not in edge_dict):
            edge_dict[(u,v)] = 1
        else:
            edge_dict[(u,v)] += 1
        
    print ("----------------------high level statistics-------------------------")
    print ("number of total edges are ", num_edges)
    print ("number of nodes are ", len(node_dict))
    print ("number of unique edges are ", len(edge_dict))
    print ("number of unique timestamps are ", num_time)
    print ("maximum edge weight is ", max_w)
    print ("minimum edge weight is ", min_w)
    print ("----------------------high level statistics-------------------------")




    print ("plotting recurrence patterns")

    node_data = node_dict.values()

    plt.hist(node_data, bins=20)
    plt.xlabel("number of node occurence")
    plt.ylabel("frequency")
    #plt.yscale('log')
    #plt.xscale('log')
    plt.savefig('node_frequency.pdf')
    plt.close()


    edge_data = edge_dict.values()
    plt.hist(edge_data, bins=20)
    plt.xlabel("number of edge occurence")
    plt.ylabel("frequency")
    #plt.yscale('log')
    #plt.xscale('log')
    plt.savefig('edge_frequency.pdf')
    plt.close()


'''
#* plot the edge weight distribution
x-axis: edge weight 
y_axis: frequency
'''
def plot_edge_weight_dist(fname):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    ws = []


    for i in range(1, len(lines)):
        line = lines[i]

        #t,u,v,w
        strs = line.split(",")
        t = strs[0]
        u = strs[1]
        v = strs[2]
        w = float(strs[3].strip())
        ws.append(w)

    plt.hist(ws, bins=20)
    plt.xlabel("edge weight")
    plt.ylabel("frequency")
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('edge_weight.pdf')
    plt.close()

    



        
           







'''
#! process the raw csv file into temporal edgelist, see raw csv file format below
block_number,transaction_index,from_address,to_address,time_stamp,contract_address,value
14669683,7,0xd30b438df65f4f788563b2b3611bd6059bff4ad9,0xda816e2122a8a39b0926bfa84edd3d42477e9efd,1651105815,0xdac17f958d2ee523a2206206994597c13d831ec7,18.67
14669683,45,0x4941834ed1428089ee76252f6f9d767e800499b0,0x28c6c06298d514db089934071355e5743bf21d60,1651105815,0xdac17f958d2ee523a2206206994597c13d831ec7,10000.0
14669683,46,0x2c1f9a20711e14f8484a41123e20d1b06858ebea,0x28c6c06298d514db089934071355e5743bf21d60,1651105815,0xdac17f958d2ee523a2206206994597c13d831ec7,9942.313005


convert into
t,u,v,value
'''
def process_csv(fname, outfile):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()


    outfile = open(outfile,"w")
    for i in range(1, len(lines)):
        line = lines[i]

        #block_number,transaction_index,from_address,to_address,time_stamp,contract_address,value
        strs = line.split(",")
        t = strs[4].strip()
        u = strs[2].strip()
        v = strs[3].strip()
        w = strs[6].strip()

        outfile.write(str(t) + "," + str(u) + "," + str(v) + "," + str(w) + "\n")
    outfile.close()
    print("write successful")

'''
read UTC timestamp edgelist and parse it to daily timestamps
'''
def parse_daily_graphs(fname):
    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    G_times = []
    G = nx.Graph()

    start = 0
    end = 0
    day = 0
    for i in range(1, len(lines)):
        line = lines[i]

        #block_number,transaction_index,from_address,to_address,time_stamp,contract_address,value
        strs = line.split(",")
        t = int(strs[0].strip())
        u = strs[1].strip()
        v = strs[2].strip()
        w = strs[3].strip()

        #! compute the initial timestamp, start of first day
        if (i == 1):
            start = math.floor(int(t) / DAY_LEN) * DAY_LEN #* first day start 
            end = start + DAY_LEN #* first day end

        if (t > end):
            G_times.append(G)   #append old graph
            G = nx.Graph()  #create new graph
            start = end
            end = start + DAY_LEN
            day += 1
        G.add_edge(u, v, weight=float(w))
    G_times.append(G)
    print ("maximum time stamp is " + str(len(G_times)))
    return G_times






def main():

    # #! generate edge file from raw data
    # fname = "coin/token_transfers_V3.0.0.csv"
    # outfile = "coin/coin_edgelistv3.csv"
    # process_csv(fname, outfile)


    #! analyze dataset statistics
    fname = "coin/coin_edgelistv3.csv" #"coin/coin_edgelist.csv"
    plot_edge_weight_dist(fname)
    # analyze_csv(fname)
    # print ("hi")
    



if __name__ == "__main__":
    main()