from Bio import SeqIO

# specify the path to the UniProtKB text file
filename = "dataset/enrichment_test/annotations.xml"

# open the file and parse the records
records = list(SeqIO.parse(filename, "uniprot-xml"))

# loop over the records and extract the sequence and annotations
for record in records:
    # get the protein sequence
    sequence = str(record.seq)

    # get the functional annotations
    annotations = record.annotations

    #print(annotations['recommendedName_fullName'])
    print(record.dbxrefs)

   # print(annotations.keys())

    # print the sequence and annotations
    #print(annotations['comment_subcellularlocation_location'])
    #print(annotations['taxonomy'])
