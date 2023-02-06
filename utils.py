from Bio import SeqIO

def gb_seq_parser(path: str) -> dict:
    """
        A parser for genbank format files. Given the path of the file, this function
        return a dict where the keys are the IDs and the values is an other dict that has 
        the key "sequence" i.e. the ORIGIN as a string
    
    """
    
    out_dict = {}
    with open(path, "r") as file:
        for seqrecord in (SeqIO.parse(file, "genbank")):
            out_dict[seqrecord.id] = {"sequence" : str(seqrecord.seq)}
    
    return out_dict
