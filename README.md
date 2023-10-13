# SCPD
Official code repository for PAKDD 2023 paper "Fast and Attributed Change Detection on Dynamic Graphs with Density of States" 

<p>
  <a href="https://link.springer.com/book/10.1007/978-3-031-33374-3">
    <img src="https://img.shields.io/badge/Paper-link-important">
  </a>
  <a href="https://arxiv.org/abs/2305.08750">
    <img src="https://img.shields.io/badge/arXiv-pdf-yellowgreen">
  </a>
  <a href="https://youtu.be/20zusjJZNdo">
    <img src="https://img.shields.io/badge/Youtube-Recording-orange">
  </a>
</p>

![SCPD](figs/crown_pic.png)


## Dataset Links

- MAG History dataset: [link](https://object-arbutus.cloud.computecanada.ca/tgb/history_scpd.zip)

- COVID flight dataset: [link](https://object-arbutus.cloud.computecanada.ca/tgb/flight_scpd.zip)

- stablecoin dataset: [link](https://object-arbutus.cloud.computecanada.ca/tgb/stablecoin_scpd.zip)


## instructions

We include `historydosN20.pkl` which is the computed DOS embedding and can be used with Anomaly_Detection.py

similarly, we include `skynet_gdos.pkl` and `china.pkl` to reproduce COVID flight network experiment

run `datasets/multi_SBM/SBM_generator.py` to generate SBM hybrid experiments

run `datasets/multi_SBM/SBM_addnode.py` to generate SBM Evolving Size experiment

in `subroutines/ADOS/run_ADOS.mat` to run attributed DOS with LDOS 

follow main function in `dos.py` to generate dos embeddings in python, and for real world experiments

follow main function in `spotlight.py` to run SPOTLIGHT experiments

**To use local DOS to approximate eigenvectors**

You can use the MATLAB code under `SCPD/subroutines/ADOS`. 
Because the interaction with eigenvectors are approximated with the GQL approximation method (currently only implemented in MATLAB so far). 


## Contact:

Feel free to reach out to me if you have any questions: `shenyang.huang@mail.mcgill.ca`


## Citation:

If code or data from this repo is useful for your project, please consider citing our paper:
```
@inproceedings{huang2023fast,
  title={Fast and Attributed Change Detection on Dynamic Graphs with Density of States},
  author={Huang, Shenyang and Danovitch, Jacob and Rabusseau, Guillaume and Rabbany, Reihaneh},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={15--26},
  year={2023},
  organization={Springer}
}
```
