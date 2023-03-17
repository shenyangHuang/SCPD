import sys
sys.path.append('../')

import numpy as np
import networkx as nx
import sparse
import scipy.stats as stats
from math import sqrt
import math
from sklearn.preprocessing import normalize
from util import normal_util
import pylab as plt
import argparse

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.xmargin'] = 0



deep_saffron = '#FF9933'
Ruby_red = '#e0115f'
air_force_blue = '#5D8AA8'
forest = '#31a354'

def find_anomalies(z_scores, percent_ranked, initial_window):
    z_scores = np.array(z_scores)
    for i in range(initial_window+1):
        z_scores[i] = 0        #up to initial window + 1 are not considered anomalies. +1 is because of difference score
    num_ranked = int(round(len(z_scores) * percent_ranked))
    outliers = z_scores.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return outliers


'''
find the arithmetic average as typical behavior
'''
def average_typical_behavior(context_vecs):
    avg = np.mean(context_vecs, axis=0)
    return avg

'''
find the left singular vector of the activity matrix
'''
def principal_vec_typical_behavior(context_vecs):
    activity_matrix = context_vecs.T
    u, s, vh = np.linalg.svd(activity_matrix, full_matrices=False)
    return u[:,0]



'''
compute the z score as defined by Akoglu and Faloutsos in EVENT DETECTION IN TIME SERIES OF MOBILE COMMUNICATION GRAPHS
Z = 1-u^(T)r
'''
def compute_Z_score(cur_vec, typical_vec):
    cosine_similarity = abs(np.dot(cur_vec, typical_vec) / LA.norm(cur_vec) / LA.norm(typical_vec))
    z = (1 - cosine_similarity)
    return z



def set_non_negative(z_scores):
    for i in range(len(z_scores)):
        if (z_scores[i] < 0):
            z_scores[i] = 0
    return z_scores

'''
plot different anomaly scores and how they correspond with real world
'''
def plot_anomaly_score(fname, scores, score_labels, events, real_events, initial_window=10):


    for k in range(len(scores)):
        scores[k] = set_non_negative(scores[k])

    '''
    scores at initial windows is 0
    '''
    for score in scores:
        for l in range(initial_window+1):
            score[l] = 0

    max_time = len(scores[0])
    t = list(range(0, max_time))
    plt.rcParams.update({'figure.autolayout': True})
    plt.rc('xtick')
    plt.rc('ytick')
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 1, 1)
    colors = ['#fdbb84', '#43a2ca', '#bc5090', '#e5f5e0','#fa9fb5','#c51b8a', '#bf812d', '#35978f','#542788','#b2182b', '#66c2a5', '#fb9a99','#e31a1c','#ff7f00','#8dd3c7']
    for i in range(len(scores)):
        ax.plot(t, scores[i], color=colors[i], ls='solid', lw=0.5, label=score_labels[i])

    
    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]


        plt.annotate(str(event), # this is the text
                 (event, max_score), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(8,0), # distance from text to points (x,y)
                 ha='center',
                 fontsize=4) # horizontal alignment can be left, right or center

    addLegend = True

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        if (addLegend):
            ax.plot( event, max_score, marker="*", mew=0.1, color='#de2d26', ls='solid', lw=0.5, label="anomalies")
            addLegend=False
        else:
            ax.plot( event, max_score, marker="*", mew=0.1, color='#de2d26', ls='solid', lw=0.5)

    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    ax.set_xlabel('time steps', fontsize=8)
    ax.set_ylabel('anomaly score', fontsize=8)
    plt.legend(fontsize=4)
    plt.savefig(fname +'anomalyScores.pdf')

    print ("plotting anomaly scores complete")
    plt.close()



def plot_anomaly_and_spectro(fname, scores, score_labels, events, eigen_file, initial_window=10, l2normed=False):


    labels = list(range(0,len(scores[0]),1))
    for k in range(len(scores)):
        scores[k] = set_non_negative(scores[k])

    for score in scores:
        for l in range(initial_window+1):
            score[l] = 0

    z_scores = []
    for i in range(len(scores[0])):
        z_scores.append(max(scores[0][i], scores[1][i]))


    fig = plt.figure(figsize=(4, 2))
    axs = fig.add_subplot(1, 1, 1)
    diag_vecs = normal_util.load_object(eigen_file)
    if (l2normed):
        diag_vecs = np.asarray(diag_vecs)
        diag_vecs = diag_vecs.real
        diag_vecs= normalize(diag_vecs, norm='l2')


    diag_vecs = np.transpose(np.asarray(diag_vecs))     
    diag_vecs = np.flip(diag_vecs, 0)

    max_time = len(scores[0])
    t = list(range(0, max_time))
    colors = ["#d53e4f", "#3288bd"]

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        axs.annotate(str(labels[event]), # this is the text
                 (event, max_score), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-12), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

    addLegend = True
    axs.plot(t, z_scores, color=colors[1], ls='solid', label="SCPD score")


    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        if (addLegend):
            axs.plot( event, max_score, marker="*", color='#de2d26', ls='solid', label="anomalies")
            addLegend=False
        else:
            axs.plot( event, max_score, marker="*", color='#de2d26', ls='solid')

    axs.tick_params(axis='both', which='major', labelsize=6)
    axs.set_ylabel('anomaly score', fontsize=6)
    plt.tight_layout()
    axs.legend(fontsize=6)
    axs.set_xlabel('Time Point', fontsize=6)
    plt.savefig(fname+'Spectro.pdf')
    plt.close()



def plot_given_anomaly_spectro(fname, scores, score_labels, events, diag_vecs, dates_idx, dates, initial_window=10):

    labels = list(range(0,len(scores[0]),1))
    for k in range(len(scores)):
        scores[k] = set_non_negative(scores[k])
    for score in scores:
        for l in range(initial_window+1):
            score[l] = 0

    z_scores = []
    for i in range(len(scores[0])):
        z_scores.append(max(scores[0][i], scores[1][i]))

    fig, axs = plt.subplots(2)
    plt.rcParams.update({'figure.autolayout': True})
    diag_vecs = np.transpose(np.asarray(diag_vecs))     #let time be x-axis
    diag_vecs = np.flip(diag_vecs, 0)

    max_time = len(scores[0])
    t = list(range(0, max_time))
    colors = ["#d53e4f", "#3288bd"]
    score_labels = ["2 weeks", "4 weeks"]  #just for Skynet
    axs[0].plot(t, z_scores, color=colors[1], ls='solid', label="SCPD score")

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        date_str = str(event)
        if (event in dates_idx):
            eid = dates_idx.index(event)
            date_str = dates[eid]

        date_encode = ["2020-02-04", "2020-02-11","2020-02-18", "2020-03-10", "2020-03-17", "2020-03-31","2020-04-14", "2020-04-21"]
        encode_idx = [5, 6, 7, 10, 11, 13, 15, 16]
        if (event in encode_idx):
            eid = encode_idx.index(event)
            date_str = date_encode[eid]

        axs[0].annotate((date_str), # this is the text
                 (event, max_score), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-6), # distance from text to points (x,y)
                 ha='center')# horizontal alignment can be left, right or center 
    addLegend = True

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        if (addLegend):
            axs[0].plot( event, max_score, marker="*", color='#de2d26', ls='solid', label="anomalies")
            addLegend=False
        else:
            axs[0].plot( event, max_score, marker="*", color='#de2d26', ls='solid')



    axs[0].set_ylabel('anomaly score')
    axs[0].set_xticks(dates_idx)
    axs[0].xaxis.set_ticklabels([])
    axs[0].legend()
    axs[1].set_xlabel('Time Point')
    axs[1].set_ylabel('spectrum index')
    axs[1].set_xticks(dates_idx)
    axs[1].set_xticklabels(dates, rotation=20)
    im = axs[1].imshow(diag_vecs, aspect='auto')
    plt.savefig(fname+'Spectro.pdf')
    plt.close()


def change_detection_many_windows(spectrums, windows,initial_window, principal=True, percent_ranked=0.05, difference=False):
    z_scores = []
    for i in range(len(windows)):
        z_scores.append([])

    counter = 0
    for j in range(0, initial_window):
        for i in range(0, len(z_scores)):
            z_scores[i].append(0)

    for i in range(initial_window, len(spectrums)):

        for j in range(len(windows)):

            if (windows[j] > i):
                z_scores[j].append(0)
            else:
                #1. compute short term window first
                if (principal):
                    typical_vec = principal_vec_typical_behavior(spectrums[i-windows[j]:i])
                else:
                    typical_vec = average_typical_behavior(spectrums[i-windows[j]:i])
                cur_vec = spectrums[i]
                z = compute_Z_score(cur_vec, typical_vec)
                z_scores[j].append(z)

    #check the change in z score instead
    if (difference):
        for i in range(len(windows)):
            z_scores[i] = difference_score(z_scores[i])

    z_overall = [0] * len(z_scores[0])
    for j in range(len(z_scores[0])):
        if (j < initial_window):
            continue
        else:
            z_overall[j] = max([x[j] for x in z_scores])

    z_overall = np.array(z_overall)
    for i in range(initial_window+1):
        z_overall[i] = 0        #up to initial window + 1 are not considered anomalies. +1 is because of difference score

    num_ranked = int(round(len(z_overall) * percent_ranked))
    outliers = z_overall.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return (z_overall,z_scores,outliers)






def difference_score(z_scores):
    z = []
    for i in range(len(z_scores)):
        if (i==0):
            z.append(z_scores[0])
        else:
            z.append(z_scores[i] - z_scores[i-1])
    return z


def detection_many_windows(fname, real_events, windows, eigen_file, timestamps=195, percent_ranked=0.05, difference=False, normalized=True,symmetric=False, verbose=True):
    principal = True
    spectrums = normal_util.load_object(eigen_file)
    if (len(spectrums) == 2):
        spectrums = spectrums[1][1]


    if (type(spectrums) == sparse._coo.core.COO):
        spectrums = spectrums.todense()
    spectrums = np.asarray(spectrums).real
    spectrums = spectrums.reshape((timestamps,-1))
    
    if (normalized):
        spectrums= normalize(spectrums, norm='l2')

    initial_window = max(windows)

    if (verbose):
        print ("window sizes are :")
        print (windows)
        print ("initial window is " + str(initial_window))
        print (spectrums.shape)

    if (not symmetric):
        (z_overall, z_scores, anomalies) = change_detection_many_windows(spectrums, windows,  initial_window, principal=principal, percent_ranked=percent_ranked, difference=difference)
    else:
        (z_overall, z_scores, anomalies) = change_detection_symmetric_windows(spectrums, windows,  principal=principal, percent_ranked=percent_ranked, difference=difference)

    if (verbose):
        print ("found anomalous time stamps are")
        print (anomalies)

    events = anomalies
    scores = z_scores
    score_labels = ["window size "+ str(window) for window in windows]
    plot_anomaly_and_spectro(fname, scores, score_labels, events, eigen_file, initial_window=initial_window)
    return z_overall




    

def multi_SBM(fname, verbose=True, real_events=[16,31,61,76,91,106,136], timestamps=151, n_init=20):
    percent_ranked= len(real_events) / timestamps
    eigen_file = fname + ".pkl"
    difference=True
    symmetric=False
    normalized=True


    windows = [5,10]
    z_scores = detection_many_windows(fname, real_events, windows, eigen_file, timestamps=timestamps, percent_ranked=percent_ranked, difference=difference, normalized=normalized, symmetric=symmetric, verbose=verbose)
    anomalies = find_anomalies(z_scores, percent_ranked, max(windows))

    accu = metrics.compute_accuracy(anomalies, real_events)
    print ("combined accuracy is " + str(accu))

    z_scores = set_non_negative(z_scores)

    return z_scores, accu


def plot_dblp_spectro(fname, scores, score_labels, events, event_labels, diag_vecs, dates_idx, dates, initial_window=10):

    labels = list(range(0,len(scores[0]),1))
    for k in range(len(scores)):
        scores[k] = set_non_negative(scores[k])


    for score in scores:
        for l in range(initial_window+1):
            score[l] = 0

    z_scores = []
    for i in range(len(scores[0])):
        z_scores.append(max(scores[0][i], scores[1][i]))


    fig = plt.figure(figsize=(5, 2))
    axs = fig.add_subplot(1, 1, 1)
    diag_vecs = np.transpose(np.asarray(diag_vecs))     #let time be x-axis
    diag_vecs = np.flip(diag_vecs, 0)

    max_time = len(scores[0])
    t = list(range(0, max_time))
    colors = ["#d53e4f", "#3288bd"]
    bar_c= "#d53e4f"


    '''
    add grey bars for historical events
    '''
    #American Civil War (1861-1865), Second World War (1939-1945),
    #Kashmir War (1947-1948) Korean war (1950-1953)
    wars = [(24,28), (102,108), (110,111), (113,116)]
    for x1,x2 in wars:
        axs.axvspan(x1, x2, alpha=0.5, color=bar_c)
    axs.axvline(x=97,alpha=0.5, color=bar_c)

    axs.plot(t, z_scores, color=colors[1], ls='solid', lw=0.5, label="SCPD score")

    for e in range(len(events)):
        max_score = 0
        for i in range(len(scores)):
            if scores[i][events[e]] > max_score:
                max_score = scores[i][events[e]]

        if (e == len(events)-1):
            axs.annotate(event_labels[e], # this is the text
                     (events[e], max_score), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(6,0), # distance from text to points (x,y)
                     ha='center',
                     fontsize=6)# horizontal alignment can be left, right or center
        else:
            axs.annotate(event_labels[e], # this is the text
                     (events[e], max_score), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,0), # distance from text to points (x,y)
                     ha='center',
                     fontsize=6)# horizontal alignment can be left, right or center
    addLegend = True

    for event in events:
        max_score = 0
        for i in range(len(scores)):
            if scores[i][event] > max_score:
                max_score = scores[i][event]
        if (addLegend):
            axs.plot( event, max_score, marker="*", color='#de2d26', ls='solid', label="anomalies")
            addLegend=False
        else:
            axs.plot( event, max_score, marker="*", color='#de2d26', ls='solid')

    axs.set_ylabel('anomaly score', fontsize=6)
    axs.legend(fontsize=6)
    plt.tight_layout()
    axs.set_xticks(dates_idx)
    axs.set_xticklabels(dates, rotation=45)
    axs.tick_params(axis='both', which='major', labelsize=6)
    plt.savefig(fname+'Spectro.pdf', bbox_inches='tight')
    plt.close()


def MAG_geology(fname, r1, r2, outname="MAG_Geology"):
    windows = [5,10]
    difference=True
    principal = True
    percent_ranked = 0.05
    real_events = []
    ds = list(range(1837+r1,1837+r1+r2,10))
    dates = [str(d) for d in ds]
    dates_idx = list(range(0,r2-r1,10))

    spectrums = normal_util.load_object(fname)
    spectrums = spectrums[0:-1]
    spectrums = spectrums[r1:r2]
    timestamps = len(spectrums)
    spectrums = np.asarray(spectrums).real
    spectrums = spectrums.reshape((timestamps,-1))
    spectrums= normalize(spectrums, norm='l2')

    initial_window = max(windows)
    (z_overall, z_scores, anomalies) = change_detection_many_windows(spectrums, windows,  initial_window, principal=principal, percent_ranked=percent_ranked, difference=difference)

    events = anomalies
    print ( [ (a + r1) for a in anomalies])
    scores = z_scores
    score_labels = ["window size "+ str(window) for window in windows]

    event_labels = [str(e + 1837 + r1) for e in events]
    print (event_labels)

    diag_vecs = normal_util.load_object(fname)
    diag_vecs = diag_vecs[r1:r2]
    plot_dblp_spectro(outname, scores, score_labels, events, event_labels, diag_vecs, dates_idx, dates, initial_window=max(windows))
    return z_overall



def SkyNet(fname, outname="skynet", percent_ranked=0.05):
    windows = [2,4]
    difference=True
    principal = True
    real_events = []

    spectrums = normal_util.load_object(fname)
    spectrums = spectrums[0:-1]


    timestamps = len(spectrums)
    dates = ["2020-01-01","2020-01-28","2020-02-25","2020-03-24","2020-04-28","2020-05-26","2020-06-30","2020-07-21"]
    dates_idx = [0,4,8,12,17,21,26,29]
    spectrums = np.asarray(spectrums).real
    spectrums = spectrums.reshape((timestamps,-1))
    spectrums= normalize(spectrums, norm='l2')


    initial_window = max(windows)
    (z_overall, z_scores, anomalies) = change_detection_many_windows(spectrums, windows,  initial_window, principal=principal, percent_ranked=percent_ranked, difference=difference)

    events = anomalies
    scores = z_scores
    score_labels = ["window size "+ str(window) for window in windows]

    event_labels = [str(e) for e in events]
    print (event_labels)

    diag_vecs = normal_util.load_object(fname)
    plot_given_anomaly_spectro(outname, scores, score_labels, events, diag_vecs, dates_idx, dates, initial_window=max(windows))
    return z_overall









def main():
    '''
    run this block of code for MAG_History
    '''
    # fname = "historydosN20.pkl"
    # outname = "MAG_History"
    # MAG_geology(fname, 0, 180, outname=outname)

    '''
    run this block of code for SkyNet dataset
    '''
    fname = "skynet_gdos.pkl"
    #fname = "china.pkl"
    percent_ranked = 0.07
    SkyNet(fname, outname="covid_flight", percent_ranked=percent_ranked)
    

    '''
    uncomment this block of code for SBM
    '''

    # parser = argparse.ArgumentParser(description='anomaly detection on signature vectors')
    # parser.add_argument('-f','--file', 
    #                 help='file name', required=True)
    # parser.add_argument('-d','--dataset', 
    #                 help='identifying which dataset', required=False)
    # parser.set_defaults(dataset="SBM")
    # args = vars(parser.parse_args())

    # #real_events = [61,76,91,106,136]
    # real_events = [16,31,61,76,91,106,136]

    # if (args["dataset"] == "SBM"):
    #     multi_SBM("../" + args["file"], real_events=real_events, verbose=False)
   

if __name__ == "__main__":
    main()
