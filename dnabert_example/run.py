from typing import Tuple
from pathlib import Path


# ensure to have run `pip install -r requirements.txt`


import torch
from transformers import BertModel, BertConfig, AutoTokenizer, BertTokenizerFast


DNABERT_PATH = Path("dnabert")


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
    """
    Computes DNABert embeddings of sample sequences
    and saves the results in a `embeddings.pt` file.

    Arguments
    ---------
    device (torch.device): cuda for GPU usage, or cpu for CPU usage.
    """

    model, tokenizer = load_dnabert()
    model = model.to(device)

    sample_sequences = [
        "AGCTCGCTAGACGCTCGA",
        "ACTCGCTAGAAGAACTCCGAATCCGATACGCAT",
        "GCATACGACTACGCGGCCATACGACATAACTAGCAAG",
        "GCATAGACATCAGCATCAGGGCAT",
    ]

    kmerized_sequences = [split_sequence(seq) for seq in sample_sequences]

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
    print((embeddings.pooler_output).shape)


    #torch.save(embeddings, "embeddings.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_embeddings(torch.device(device))


