from Bio import SeqIO

# specify the path to the UniProtKB text file
filename = "dataset/enrichment_test/annotations.xml"

# open the file and parse the records
records = list(SeqIO.parse(filename, "uniprot-xml"))

annotation_dict = {}

# loop over the records and extract the sequence and annotations
for record in records:

    name = f'sp|{record.id}|{record.name}'

    annotation_dict[name] = {}

    # take the dbxrefs that start with GO
    go_annotations = [i for i in record.dbxrefs if i.startswith('GO')]
    

    annotation_dict[name]['go'] = go_annotations
    annotation_dict[name]['keywords'] = record.annotations['keywords']
    annotation_dict[name]['taxonomy'] = record.annotations['taxonomy']

    



    # print the sequence and annotations
    #print(annotations['comment_subcellularlocation_location'])
    #print(annotations['taxonomy'])

# for each pair of proteins, compute the number of common annotations
for i in range(len(records)):
    for j in range(i+1, len(records)):
        # compute the number of common annotations
        common_annotations = len(set(annotation_dict[records[i].name]['go']).intersection(set(annotation_dict[records[j].name]['go'])))
        common_annotations += len(set(annotation_dict[records[i].name]['keywords']).intersection(set(annotation_dict[records[j].name]['keywords'])))

        # print the number of common annotations
        print("Proteins {} and {} have {} common annotations".format(records[i].name, records[j].name, common_annotations))

