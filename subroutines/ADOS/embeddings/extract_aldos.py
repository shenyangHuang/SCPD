import numpy as np
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 2)

def extract_aldos(fname, aidx):
    vecs = []
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            floats = line.split(",")
            vec = np.zeros(20)
            for i in range(20):
                vec[i] = float(floats[20*aidx + i])
            vecs.append(vec)
    save_object(vecs, "attr_",aidx,".pkl")







if __name__ == "__main__":
    fname = "SBM_attr_dos_ldos.csv"
    extract_aldos(fname, 1)
    extract_aldos(fname, 2)