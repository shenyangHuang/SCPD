import sys
sys.path.append('../')
from subroutines.DOS import moment_comp, moment_filter, rescale_matrix, plot_cheb
from datasets import SBM_loader, geology_loader, coin_loader
import networkx as nx
import numpy as np
import numpy.random as nr
import scipy.sparse as ss
from scipy import sparse
from scipy.sparse.linalg import svds, eigsh
import scipy.sparse.linalg as ssla
from scipy.sparse import identity
from numpy import linalg as LA
import Anomaly_Detection


import random
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import math
import timeit
from util import normal_util


def load_graph(graphname,path='',mname='A'):
    """
    Load a graph matrix from Matlab file

    Args:
        graphname: name of the .mat file
        path: the folder .mat file is stored in
        mname: name of the Matlab matrix

    Output:
        data[mname]: the graph matrix
    """

    data=sio.loadmat(path+graphname)
    return data[mname]

def matrix_normalize(W,mode='s'):
    """
    Normalize a weighted adjacency matrix.

    Args:
        W: weighted adjacency matrix
        mode: string indicating the style of normalization;
            's': Symmetric scaling by the degree (default)
            'r': Normalize to row-stochastic
            'c': Normalize to col-stochastic

    Output:
        N: a normalized adjacency matrix or stochastic matrix (in sparse form)
    """

    dc = np.asarray(W.sum(0)).squeeze()
    dr = np.asarray(W.sum(1)).squeeze()
    [i,j,wij] = ss.find(W)

    # Normalize in desired style
    if mode in 'sl':
        wij = wij/np.sqrt(dr[i]*dc[j])
    elif mode == 'r':
        wij = wij/dr[i]
    elif mode == 'c':
        wij = wij/dc[j]
    else:
        raise ValueError('Unknown mode!')

    N = ss.csr_matrix((wij,(i,j)),shape=W.shape)
    return N


'''
Nmoment = # of Chebyshev moments
Nz = # of probing vectors
'''
def get_dos(L, Nmoment=50, nZ=100, outname="none", npts=20, maxsize=-1, compute_range=False, adj=False):
    if (not adj):    
        if (compute_range):
            L, _ = rescale_matrix.rescale_matrix(L)      
        else:
            L, _ = rescale_matrix.rescale_matrix(L, range=[0,2])      #the range is specified by normalized Laplacian
    
    n = L.shape[0]
    c = moment_comp.moments_cheb_dos(L,n,N=Nmoment, nZ=nZ)[0]
    c = moment_filter.filter_jackson(c)

    if (outname == "none"):
        xm, yy =  plot_cheb.plot_chebhist((c,), npts=(npts+1), pflag=False)
    else:
        if (maxsize != -1):
            xm, yy =  plot_cheb.plot_chebhist((c,),pflag=True, npts=(npts+1), outname=outname, maxsize=maxsize)
        else:
            xm, yy =  plot_cheb.plot_chebhist((c,),pflag=True, npts=(npts+1), outname=outname)

    return yy



def cosine_similarity(u, v):
    cos_sim = abs(np.dot(u, v) / LA.norm(u) / LA.norm(v))
    return cos_sim



def get_ldos(L, Nmoment=50, outname="none", maxsize=-1, pairwise=False):
    #L = matrix_normalize(L)
    L_copy = L.copy()
    L, _ = rescale_matrix.rescale_matrix(L, range=[0,2])  #range is always[0,2] for normalized Laplacian matrix
    n = L.shape[0]
    c = moment_comp.moments_cheb_dos(L,n,N=Nmoment)[0]
    cl, csl = moment_comp.moments_cheb_ldos(L,n,N=Nmoment)
    yl,idx = plot_cheb.plot_cheb_ldos((cl,),outname=outname)
    if (pairwise):
        highs, lows = compute_pairwise(yl)
    else:
        highs = 0
        lows = 0


    return yl, idx, highs, lows



def get_svd_dos(L):
    L, _ = rescale_matrix.rescale_matrix(L)
    n = L.shape[0]
    s = ssla.eigsh(L, k=(n-1), which='LA',
        return_eigenvectors=False)
    return s


def get_svd(L):
    n = L.shape[0]
    s = ssla.eigsh(L, k=(n-1), which='LA',
        return_eigenvectors=False)
    return s

def mag_history_dos(Nmoment=10, nZ=20, npts=50):
    fname = "../datasets/MAG_history/coauth-MAG-History-full-edgelist.txt"
    directed = False
    G_times = geology_loader.load_temporarl_edgelist(fname)
    vecs = []
    density = []
    total_edges = 0
    total_time = 0
    total_nodes = 0
    for i in range(len(G_times)):
        print ("processing time step " + str(i),end="\r")
        total_nodes += G_times[i].number_of_nodes()
        density.append(G_times[i].number_of_nodes() / G_times[i].number_of_edges())
        total_edges = total_edges + G_times[i].number_of_edges()
        L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G_times[i], weight='weight')
        L = L.asfptype()
        start = timeit.default_timer()        
        vecs.append(get_dos(L, Nmoment=Nmoment, nZ=nZ, npts=npts))
        normal_util.save_object(vecs, "historydos" + 'N' + str(Nmoment)+".pkl")
        end = timeit.default_timer()
        total_time = total_time + end - start
    print ('DOS time: '+str(total_time)+'\n')
    print ("total nodes are " + str(total_nodes))
    print ('total edges are ' + str(total_edges))
    print ("there are " + str(int(total_edges/len(G_times))) + " average edges per snapshot")
    den = np.array(density)
    print ("the average density per snapshot is ", np.mean(den))
    normal_util.save_object(vecs, "historydos" + 'N' + str(Nmoment)+".pkl")

'''
1. load matrix
2. normalized matrix by degree, and in numpy format
3. apply cheb_dos
4. apply jackson_filter
'''
def Laplace_dos(fname, Nmoment=50, nZ=50, npts=20, svd=False, compute_range=False, save=True):

    edgefile = "../datasets/multi_SBM/" + fname + ".txt"
    print (edgefile)
    if (not os.path.isfile(edgefile)):
        edgefile = fname + ".txt"

    G_times = SBM_loader.load_temporarl_edgelist(edgefile, draw=False)
    total_edges = 0
    cps = [15,16,30,31,60,61,75,76,90,91,105,106,135,136]
    #cps=[]
    vecs = []
    if (svd):
        svd_vecs = []
        svd_time = 0

    total_time = 0
    for i in range(len(G_times)):
        total_edges = total_edges + G_times[i].number_of_edges()
        L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G_times[i])
        L = L.asfptype()
        print (str(i), end="\r")
        start = timeit.default_timer()
        if (i in cps):
            vecs.append(get_dos(L.copy(), Nmoment=Nmoment, nZ=nZ, npts=npts, outname="none", compute_range=compute_range))
        else:
            vecs.append(get_dos(L.copy(), Nmoment=Nmoment, nZ=nZ, npts=npts, outname="none", compute_range=compute_range))
        end = timeit.default_timer()
        total_time = total_time + end - start
        if (svd):
            start = timeit.default_timer()
            if (i in cps):
                vals = get_svd(L)
                counts, pos, _ = plt.hist(vals, bins=20, range=(0,2), color="#3690c0")
                for k in range(len(counts)):
                    plt.text(pos[k], counts[k] + 1, str(math.ceil(counts[k])), color='black', fontweight='bold')
                plt.savefig(str(i) + "svdNoscale.pdf")
                plt.close()

            vals = get_svd_dos(L.copy())
            end = timeit.default_timer()
            svd_time = svd_time + end - start
            hist, bin_edges = np.histogram(vals, bins=20, range=(-1,1))
            svd_vecs.append(hist)
            if (i in cps):
                counts, pos, _ = plt.hist(vals, bins=20, range=(-1,1), color="#3690c0")
                for k in range(len(counts)):
                    plt.text(pos[k], counts[k] + 1, str(math.ceil(counts[k])), color='black', fontweight='bold')
                plt.savefig(str(i) + "svddos.pdf")
                plt.close()
    
    print ('DOS time: '+ str(total_time) +'\n')
    if (svd):
        print ('SVD time: '+ str(svd_time) +'\n')
    print ('total edges are ' + str(total_edges))
    print ("there are " + str(int(total_edges/len(G_times))) + " average edges per snapshot")

    if (save):
        normal_util.save_object(vecs, "dosNm" + str(Nmoment) + "Nz" + str(nZ) + "Npts" + str(npts) + fname + ".pkl")
        if (svd):
            normal_util.save_object(svd_vecs, "svddos" + ".pkl")

    return vecs, "dosNm" + str(Nmoment) + "Nz" + str(nZ) + "Npts" + str(npts) + fname, total_time


def load_vec(eigen_file):
    vecs = normal_util.load_object(eigen_file)
    print (vecs)



'''
compute density of states for the Stablecoin dataset
'''
def coin_dos(Nmoment=10, nZ=20, npts=50):
    fname = "../datasets/coin/coin_edgelistv3.csv"#"../datasets/coin/coin_edgelist.csv"
    outname = "coindosv3"
    G_times = coin_loader.parse_daily_graphs(fname)
    vecs = []
    density = []
    total_edges = 0
    total_time = 0
    total_nodes = 0
    for i in range(len(G_times)):
        print ("processing time step " + str(i),end="\r")
        total_nodes += G_times[i].number_of_nodes()
        density.append(G_times[i].number_of_nodes() / G_times[i].number_of_edges())
        total_edges = total_edges + G_times[i].number_of_edges()
        L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G_times[i], weight='weight')
        L = L.asfptype()
        start = timeit.default_timer()        
        vecs.append(get_dos(L, Nmoment=Nmoment, nZ=nZ, npts=npts))
        normal_util.save_object(vecs, outname + 'N' + str(Nmoment)+".pkl")
        end = timeit.default_timer()
        total_time = total_time + end - start
    print ('DOS time: '+str(total_time)+'\n')
    print ("total nodes are " + str(total_nodes))
    print ('total edges are ' + str(total_edges))
    print ("there are " + str(int(total_edges/len(G_times))) + " average edges per snapshot")
    den = np.array(density)
    print ("the average density per snapshot is ", np.mean(den))
    normal_util.save_object(vecs, outname + 'N' + str(Nmoment)+".pkl")

if __name__ == '__main__':

    '''
    run this block for MAG_history
    '''
    Nmoment = 20
    nZ = 100
    npts= 50

    mag_history_dos(Nmoment=Nmoment, nZ=nZ, npts=npts)


    '''
    run this block for coin
    '''
    Nmoment = 20
    nZ = 100
    npts= 50

    coin_dos(Nmoment=Nmoment, nZ=nZ, npts=npts)



    '''
    run this block of code for synthetic experiments
    '''
    svd=False
    dname = "SBM1000"

    Nmoment = 20
    nZ = 100
    npts= 50
    compute_range = False
    print ("using ", Nmoment, " number of Chebyshev moments")
    print ("using ", nZ, " number of probing vectors")
    print ("using ", npts, " number of bins / intervals")

    num_trials = 5
    seeds = list(range(num_trials))
    real_events = [16,31,61,76,91,106,136]

    accus = []
    times = []

    for seed in seeds:

        #set random seed
        random.seed(seed)
        np.random.seed(seed)

        #compute dos
        vecs, vname, time = Laplace_dos(dname, Nmoment=Nmoment, nZ=nZ, npts=npts, svd=svd, compute_range=compute_range)

        #find anomalies
        _, accu = Anomaly_Detection.multi_SBM(vname, real_events=real_events, verbose=False)
        accus.append(accu)
        times.append(time)

    accus = np.asarray(accus)
    times = np.asarray(times)
    print (" mean accuracy is ", np.mean(accus))
    print (" std is ", np.std(accus))

    print (" mean time is ", np.mean(times))
    print (" time std is ", np.std(times))

