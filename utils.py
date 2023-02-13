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
