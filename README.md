# SCPD
Official code repository for PAKDD accepted paper "Fast and Attributed Change Detection on Dynamic Graphs with Density of States" (to appear)


## instructions

MAG_history dataset's edgelist is too large to be included, will be available for camera ready. Instead we include "historydosN20.pkl" which is the computed DOS embedding and can be used with Anomaly_Detection.py

similarly, we include "skynet_gdos.pkl" and "china.pkl" to reproduce COVID flight network experiment

run datasets/multi_SBM/SBM_generator.py to generate SBM hybrid experiments

run datasets/multi_SBM/SBM_addnode.py to generate SBM Evolving Size experiment

in subroutines/ADOS/run_ADOS.mat to run attributed DOS with LDOS 

follow main function in dos.py to generate dos embeddings in python 

follow main function in spotlight.py to run SPOTLIGHT experiments

