from typing import Tuple
from pathlib import Path
import json
import numpy as np

# ********* SETTINGS **********

DNABERT_PATH = Path("dnabert")
FILE_PATH = "../dataset/globins/globins.json" # file containing the origina dataset. A key will be added on the dict and the file will be overwrited
ANNOTATION_KEY = "dnabert"   # the key to add


import torch
from transformers import BertModel, BertConfig, AutoTokenizer, BertTokenizerFast





def split_sequence(seq: str, k: int=6) -> str:
    """
    Splits a sequence in a set of 6-mers and the joins it together.

    Arguments
    ---------
    seq (str): a sequence of bases.
    k (int): the length of the k-mer (defaults to 6).

    Returns
    -------
    joined_seq (str): the original string split into k-mers (separated by
    spaces)
    """
    kmers = [seq[x:x+k] for x in range(0, len(seq) + 1 - k)]
    joined_seq = " ".join(kmers)
    return joined_seq


def load_dnabert() -> Tuple[BertModel, BertTokenizerFast]:
    """
    Loads DNABert and the related tokenizer.

    Returns
    -------
    model (BertModel): the model
    tokenizer (BertTokenizerFast): the tokenizer
    """
    config = BertConfig.from_pretrained("zhihan1996/DNA_bert_6")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    model = BertModel.from_pretrained(DNABERT_PATH, config=config)
    return model, tokenizer


def compute_embeddings(device: torch.device) -> None:
    

    with open(FILE_PATH, "r") as file:
        seq_dict = json.load(file)
        
    model, tokenizer = load_dnabert()
    model = model.to(device)
    
    for k in seq_dict.keys(): # for each sequence

        

        string = seq_dict[k]["sequence"]
        substrings = []
        for i in range(0, len(string), 510):
            substrings.append(string[i:i+510])

        kmerized_sequences = [split_sequence(seq) for seq in substrings]

        model_inputs = tokenizer(
            kmerized_sequences,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            embeddings = model(
                input_ids=model_inputs["input_ids"].to(device),
                token_type_ids=model_inputs["token_type_ids"].to(device),
                attention_mask=model_inputs["attention_mask"].to(device),
            )

        seq_dict[k][ANNOTATION_KEY] = np.array(embeddings.pooler_output).tolist()
        

    with open(FILE_PATH, "w") as file:
        json.dump(seq_dict, file, indent=4)



    #torch.save(embeddings, "embeddings.pt")




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_embeddings(torch.device(device))


