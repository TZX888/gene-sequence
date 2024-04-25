from Bio.Seq import Seq
from Bio import SeqIO
from tqdm import tqdm
seqh = SeqIO.parse("/home/u2208283040/tzx/LSTM/data/Hfish.fasta", "fasta")
seqg = SeqIO.parse("/home/u2208283040/tzx/LSTM/data/Gfish.fasta", "fasta")
hlength=[]
glength=[]
name=[]
import numpy as np
num=0
for seq in tqdm(seqg):
    num=num+1
    glength.append(len(seq.seq))
    # print(len(seq.seq))
print(num)
X=np.array([i for i in range(num)])
print(len(X))
import statistics
# res=[i for i in range(len(seq.seq))]
# print(len(res))
Y=np.array(glength)
# X=np.array([i for i in range(len(seq.seq))])
res1=[]
from pandas import DataFrame
import requests
import pandas
import openpyxl
data = { 'sno': [i for i in range(num)], 'length': glength}
df = DataFrame(data)
df.to_excel('Gfishs.xlsx')
# for i in Y:
#     if i<400000:
#         res1.append(i)
# # print(res1)
# print(len(res1))
# max_length = max(len(lst) for lst in FileNameList)
# mdeian=statistics.median(len(lst) for lst in FileNameList)
# print("Hfish_clean",max_length,mdeian-16)
import matplotlib.pyplot as plt





plt.plot(X, glength)
plt.savefig('/home/u2208283040/tzx/LSTM/Hfish_length.eps')
plt.savefig('/home/u2208283040/tzx/LSTM/Hfish_length.png')
plt.show()
