
import torch
from transformers import BertModel, BertConfig, DNATokenizer

dir_to_pretrained_model = "./dna_model/"




config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
tokenizer = DNATokenizer.from_pretrained('dna6')
model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)

sequence = "AATCTA ATCTAG TCTAGC CTAGCA"

with torch.no_grad():
    model_input = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=512)["input_ids"]
    model_input = torch.tensor(model_input, dtype=torch.long)
    model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
    
    output = model(model_input)

print(output[1])

