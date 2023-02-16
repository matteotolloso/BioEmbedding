from embeddings_reproduction import embedding_tools
import json
from Bio.Seq import Seq


# ********* SETTINGS **********

FILE_PATH = "../dataset/NEIS2157.json" # file containing the origina dataset. A key will be added on the dict and the file will be overwrited
ANNOTATION_KEY = "embedding1"   # the key to add

# ******************************


seq_dict = {}

with open(FILE_PATH, "r") as file:
    seq_dict = json.load(file)

IDs = seq_dict.keys()

list_seq_string = [] 
for id in IDs:
    seq_string = seq_dict[id]["sequence"]
    
    if set(seq_string).issubset(set(["A", "C", "G", "T"])):
        seq_string = str(Seq(seq_string).translate())
        print("The nucleotides sequence for ", id, " has been translated")
    
    list_seq_string.append(seq_string)

embeds_list = embedding_tools.get_embeddings_new(
    'models/original_5_7.pkl', 
    list_seq_string, 
    k=5, 
    overlap=False
)  # list of embeddings

for id, embed in zip(IDs, embeds_list): 
    seq_dict[id][ANNOTATION_KEY] = embed.tolist()

with open(FILE_PATH, "w") as file:
    json.dump(seq_dict, file, indent=4)
