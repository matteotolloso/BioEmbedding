from embeddings_reproduction import embedding_tools
from Bio import SeqIO
from numpy import array2string
import pickle


IN_PATH = "../dataset/ls_orchid.gbk"
IN_FORMAT = "genbank"
ANNOTATION_KEY = "embedding1"
OUT_PATH = "../dataset/output_test.pkl" #must be pkl because gbk doesn't work


seqrec_list = list(SeqIO.parse(IN_PATH, IN_FORMAT)) # list of SeqRec objects

seqstring_list = [str(seqrec.seq) for seqrec in seqrec_list]     #list of sequences as string

embeds_list = embedding_tools.get_embeddings_new('.\models\original_5_7.pkl', seqstring_list, k=5, overlap=False)  # list of embeddings

for seqrec, embed in zip(seqrec_list, embeds_list): # add to each seqrecord the embedding of the relative sequence as an annotation
    seqrec.annotations[ANNOTATION_KEY] = array2string(embed)


#SeqIO.write(seqrec_list, OUT_PATH, "genbank") not working 

with open(OUT_PATH, "wb") as file:
    pickle.dump(seqrec_list, file)
