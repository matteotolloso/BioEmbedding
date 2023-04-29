import numpy as np
import requests
import json


# ********* SETTINGS **********

FILE_PATH = "./dataset/NEIS2157/NEIS2157.json" # file containing the origina dataset. A key will be added on the dict and the file will be overwrited
ANNOTATION_KEY = "esm"   # the key to add

base_url = "https://api.esmatlas.com/fetchEmbedding/ESM2/"

# ******************************

def translate(id):
    




file = open(FILE_PATH, "r")

seq_dict = json.load(file)

IDs = seq_dict.keys()


for id in IDs:

    # the api accepts only proteins in the magnify database
    mgnif_id = translate(id)

    url = base_url + mgnif_id + ".bin"
    header = {"Accept": "application/octet-stream"}
    response = requests.get(url, headers=header)
    embed = np.frombuffer(response.content, dtype=np.float16)
     
    seq_dict[id][ANNOTATION_KEY] = embed.tolist()


with open(FILE_PATH, "w") as file:
    json.dump(seq_dict, file, indent=4)

file.close()