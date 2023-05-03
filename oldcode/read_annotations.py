from Bio import SeqIO

# specify the path to the UniProtKB text file
filename = "./dataset/enrichment_test/annotation.txt"

# open the file and parse the records
records = SeqIO.parse(filename, "swiss")

# loop over the records and extract the sequence and annotations
for record in records:
    # get the protein sequence
    sequence = str(record.seq)

    # get the functional annotations
    annotations = record.annotations

    # print the sequence and annotations
    #print("Protein sequence:", sequence)
    #print("Functional annotations:", annotations)
    print(annotations.keys())
