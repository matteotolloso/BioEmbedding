from embeddings_reproduction import embedding_tools
import json
from Bio.Seq import Seq


# ********* SETTINGS **********

# FILE_PATH = "../dataset/NEIS2157/NEIS2157.json" # file containing the origina dataset. A key will be added on the dict and the file will be overwrited
# FILE_PATH = "../dataset/globins/globins.json"
FILE_PATH = "../dataset/enrichment_test/proteins.json"
ANNOTATION_KEY = "rep"   # the key to add

# ******************************


seq_dict = {}

with open(FILE_PATH, "r") as file:
    seq_dict = json.load(file)

IDs = seq_dict.keys()

list_seq_string = [] 
for id in IDs:
    seq_string = seq_dict[id]["sequence"]

    seq_string = seq_string.replace(" ", "").replace("\n", "")
    
    if set(seq_string).issubset(set(["A", "C", "G", "T"])):
        seq_string = str(Seq(seq_string).translate(stop_symbol=""))
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
