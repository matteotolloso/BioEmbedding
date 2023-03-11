from Bio import SeqIO, Phylo
import json
import numpy as np
import io
import numpy as np
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


def get_ordered_ids(seq_dict: dict) -> list:
    """
    Given a seqence dictionary, where the keys are the IDs of the sequences, it return an ordered list of IDs taking into account the version fo the protein (that is different from the lexicographic order). Example: NEIS2157_9 < NEIS2157_10.

    """
    from functools import cmp_to_key

    def compare (a:str, b:str)-> int:
        ind_a = a.index("_")
        ind_b = b.index("_")
        
        if ind_a != ind_b: # diffent main name
            return a < b
        
        try:
            version_a = int(a[ind_a+1:])
            version_b = int(b[ind_b+1:])

            return version_a - version_b
        except Exception:
            print("exception in get ordered ids")
            return a < b


    IDs = list(seq_dict.keys())

    IDs.sort(key=cmp_to_key(compare))

    return IDs


def newick_to_linkage(newick: str, label_order: str = None):
    
    """
    Convert newick tree into scipy linkage matrix

    :param newick: newick string, e.g. '(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);'
    :param label_order: list of labels, e.g. ['A', 'B', 'C']
    :returns: linkage matrix and list of labels
    """

    # newick string -> cophenetic_matrix
    tree = ClusterTree(newick)
    cophenetic_matrix, newick_labels = tree.cophenetic_matrix()
    cophenetic_matrix = pd.DataFrame(cophenetic_matrix, columns=newick_labels, index=newick_labels)

    if label_order is not None:
        # sanity checks
        missing_labels = set(label_order).difference(set(newick_labels))
        superfluous_labels = set(newick_labels).difference(set(label_order))
        assert len(missing_labels) == 0, f'Some labels are not in the newick string: {missing_labels}'
        if len(superfluous_labels) > 0:
            print.warning(f'Newick string contains unused labels: {superfluous_labels}')

        # reorder the cophenetic_matrix
        cophenetic_matrix = cophenetic_matrix.reindex(index=label_order, columns=label_order)

    # reduce square distance matrix to condensed distance matrices
    pairwise_distances = squareform(cophenetic_matrix)

    # return linkage matrix and labels
    return linkage(pairwise_distances), list(cophenetic_matrix.columns)

