
import torch
from transformers import BertModel, BertConfig, DNATokenizer
import json 

dir_to_pretrained_model = "2-Ji_(dnabert)/dna_model/"

config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
tokenizer = DNATokenizer.from_pretrained('dna6')
model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)

FILE_PATH = "dataset/test.json" # file containing the origina dataset. A key will be added on the dict and the file will be overwrited
ANNOTATION_KEY = "embedding2"   # the key to add

seq_dict = {}

with open(FILE_PATH, "r") as file:
    seq_dict = json.load(file)

for k in seq_dict.keys():

    with torch.no_grad():
        model_input = tokenizer.encode_plus(seq_dict[k]["sequence"], add_special_tokens=True, max_length=512)["input_ids"]
        model_input = torch.tensor(model_input, dtype=torch.long)
        model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
        
        output = model(model_input)
        seq_dict[k][ANNOTATION_KEY] = output[1].tolist()


with open(FILE_PATH, "w") as file:
    json.dump(seq_dict, file, indent=4)

