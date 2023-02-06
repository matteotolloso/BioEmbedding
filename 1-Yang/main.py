from embeddings_reproduction import embedding_tools
from numpy import array2string
import json


FILE_PATH = "dataset/test.json" # file containing the origina dataset. A key will be added on the dict and the file will be overwrited
ANNOTATION_KEY = "embedding1"   # the key to add

seq_dict = {}

with open(FILE_PATH, "r") as file:
    seq_dict = json.load(file)

IDs = seq_dict.keys()

embeds_list = embedding_tools.get_embeddings_new(
    '1-Yang/models/original_5_7.pkl', 
    [seq_dict[id]["sequence"] for id in IDs], 
    k=5, 
    overlap=False
)  # list of embeddings

for id, embed in zip(IDs, embeds_list): 
    seq_dict[id][ANNOTATION_KEY] = array2string(embed)

with open(FILE_PATH, "w") as file:
    json.dump(seq_dict, file, indent=4)
