from Bio import SeqIO
import json
import pandas as pd
from ete3 import ClusterTree
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage


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

def get_embeddings_dict(path : str) -> dict:
    """
    Given the path of the json file created with the "genbak_to_json" or "fasta_to_json" functions, it returns a a dict where the ID of the sequences is the key
    """
    seq_dict = {}
    with open(path) as file:
        seq_dict = json.load(file)
    return seq_dict


def newick_to_linkage(newick: str):
    
    """
    Convert newick tree into scipy linkage matrix

    :param newick: newick file path or string
    :param label_order: list of labels, e.g. ['A', 'B', 'C']
    :returns: linkage matrix
    """

    # newick string -> cophenetic_matrix
    tree = ClusterTree(newick)
    cophenetic_matrix, newick_labels = tree.cophenetic_matrix()
    cophenetic_matrix = pd.DataFrame(cophenetic_matrix, columns=newick_labels, index=newick_labels)

    # reduce square distance matrix to condensed distance matrices
    pairwise_distances = squareform(cophenetic_matrix)

    # return linkage matrix and labels
    return linkage(pairwise_distances), list(cophenetic_matrix.columns)

