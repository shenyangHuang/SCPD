import numpy as np
import networkx as nx
import pylab as plt
import dateutil.parser as dparser
from scipy.io import savemat
import re

'''
treat each day as a discrete time stamp
'''
def load_temporarl_edgelist(fname, max_nodes=-1):
	edgelist = open(fname, "r")
	lines = list(edgelist.readlines())
	edgelist.close()
	max_time = 0
	current_date = 1837		#start with 1837+
	#create one graph for each day
	G_times = []
	G = nx.Graph()
	if(max_nodes > 0):
		G.add_nodes_from(list(range(0, max_nodes)))

	for i in range(0, len(lines)):
		line = lines[i]
		values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",line)
		if (len(values) < 3):
			continue
		else:
			date_str = int(values[0])
			if (date_str < 1837):	#start with 1837+
				continue

			#start a new graph with a new date
			if (date_str != current_date):
				G_times.append(G)	#append old graph
				G = nx.Graph()
				if(max_nodes > 0):
					G.add_nodes_from(list(range(0, max_nodes)))
				current_date = date_str		#update the current date

			w = int(values[-1]) 	#edge weight by number of characters 
			v = int(values[-2])		
			u = int(values[-3])
			G.add_edge(u, v, weight=w) 
	G_times.append(G)
	print ("maximum time stamp is " + str(len(G_times)))
	return G_times