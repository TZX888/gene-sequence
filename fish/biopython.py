from Bio.Seq import Seq
from Bio import SeqIO
from tqdm import tqdm
seqh = SeqIO.parse("/home/u2208283040/tzx/LSTM/data/Hfish.fasta", "fasta")
seqg = SeqIO.parse("/home/u2208283040/tzx/LSTM/data/Gfish.fasta", "fasta")
f=open("/home/u2208283040/tzx/LSTM/data_x/Gfish_4W.csv","w")
f.write('data,lable'+'\n')
res=[]
def split_string_by_length(string, length):
    return [string[i:i+length] for i in range(0, len(string), length)]

length = 400000
for seq in tqdm(seqg):
    if len(seq)<length:
        line=seq.seq.upper()
        f.write(str(line)+',1'+'\n')
    else:
        result = split_string_by_length(str(seq.seq.upper()), length)
        for i in result:
            f.write(i+',1'+'\n')
f.close()