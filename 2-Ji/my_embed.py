
import torch
from transformers import BertModel, BertConfig, DNATokenizer
import json 
import numpy as np



# ********* SETTINGS **********

PATH_TO_MODEL = "dna_model_pre_trained/"
FILE_PATH = "../dataset/NEIS2157.json" # file containing the origina dataset. A key will be added on the dict and the file will be overwrited
ANNOTATION_KEY = "embedding2"   # the key to add

# ******************************


config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
tokenizer = DNATokenizer.from_pretrained('dna6')
model = BertModel.from_pretrained(PATH_TO_MODEL, config=config)

seq_dict = {}

with open(FILE_PATH, "r") as file:
    seq_dict = json.load(file)

for k in seq_dict.keys(): # for each sequence

    # split into substrings of 512 nucleotides
    string = seq_dict[k]["sequence"]
    substrings = []
    for i in range(0, len(string), 512):
        substrings.append(string[i:i+512])

    embeddings = []

    for i, substring in enumerate(substrings):

        with torch.no_grad():
            model_input = tokenizer.encode_plus( substring, add_special_tokens=True, max_length=512)["input_ids"]
            model_input = torch.tensor(model_input, dtype=torch.long)
            model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
            
            # someone suggest to average the output for each token insted of use the last one
            # check the issues on github

            output = model(model_input)
            embeddings.append(output[1].tolist()[0])
    
    
    seq_dict[k][ANNOTATION_KEY] = embeddings
            


with open(FILE_PATH, "w") as file:
    json.dump(seq_dict, file, indent=4)

