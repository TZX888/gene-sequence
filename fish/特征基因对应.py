import numpy as np
import pandas as pd
feature= np.load(file="/home/u2208283040/tzx/LSTM/feature_Kmer_11.npy",allow_pickle=True)
# print(feature[940].upper())
# X= np.load(file="/home/u2208283040/tzx/cscwd/k_11/data/X_train.npy")
# print(X)
# print(X.shape)
d=[94073943,106577600,64726196,136253068,73956553,141986834,123677281,75468733,140960284]

for i in d:
    print(feature[i].upper())
    print("=========================================")