from Bio import SeqIO
import json

def genbank_to_json(genbank_file_path: str, json_file_path : str):
    """
        A parser for genbank format files. Given the path of the file, this function
        creates or overwrite a json file where the keys are the IDs and the values are an other dict that has 
        the key "sequence" i.e. the ORIGIN as a string
    
    """
    
    out_dict = {}
    with open(genbank_file_path, "r") as file:
        for seqrecord in (SeqIO.parse(file, "genbank")):
            out_dict[seqrecord.id] = {"sequence" : str(seqrecord.seq)}
    
    with open(json_file_path, "w") as file:
        json.dump(out_dict, file, indent=4)
    
def fasta_to_json(fasta_file_path: str, json_file_path: str):

    out_dict = {}
    with open(fasta_file_path, "r") as file:
        for seqrecord in (SeqIO.parse(file, "fasta")):
            out_dict[seqrecord.id] = {"sequence" : str(seqrecord.seq)}
    
    with open(json_file_path, "w") as file:
        json.dump(out_dict, file, indent=4)

def get_sequence_dict(path : str) -> dict:
    """
    Given the path of the json file created with the "genbak_to_json" or "fasta_to_json" functions, it returns a a dict where the ID of the sequences is the key
    """
    seq_dict = {}
    with open(path) as file:
        seq_dict = json.load(file)
    return seq_dict


def get_ordered_ids(seq_dict: dict) -> list:
    """
    Given a seqence dictionary, where the keys are the IDs of the sequences, it return an ordered list of IDs taking into account the version fo the protein (that is different from the lexicographchi order). Example: NEIS2157_9 < NEIS2157_10.

    """
    from functools import cmp_to_key

    def compare (a:str, b:str)-> int:
        ind_a = a.index("_")
        ind_b = b.index("_")
        
        if ind_a != ind_b: # diffent main name
            return a < b
        
        version_a = int(a[ind_a+1:])
        version_b = int(b[ind_b+1:])

        return version_a - version_b


    IDs = list(seq_dict.keys())

    IDs.sort(key=cmp_to_key(compare))

    return IDs

