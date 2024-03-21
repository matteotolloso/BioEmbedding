import numpy as np
from autoembedding.combiners import combiner
# from combiners import combiner
from Bio import SeqIO
import os
from autoembedding import utils
# import utils



def build_embeddings_matrix(
        case_study : str,
        embedder : str,
        combiner_method : str
    ) -> tuple[list[str] , np.array]:


    print('build_embeddings_matrix_for_covid19_case_study')
    print(f"embedder: {embedder}")
    print(f"combiner_method: {combiner_method}")

    if case_study == 'covid19':
        return build_embeddings_matrix_for_covid19_case_study(
            embedder = embedder,
            combiner_method = combiner_method
        )

    if case_study == 'hemoglobin':
        return build_embeddings_matrix_for_hemoglobin_case_study(
            embedder = embedder,
            combiner_method = combiner_method
        )
    
    if case_study == 'mouse':
        return build_embeddings_matrix_for_mouse_case_study(
            embedder = embedder,
            combiner_method = combiner_method
        )
    
    if case_study == 'satb2':
        return build_embeddings_matrix_for_satb2_case_study(
            embedder = embedder,
            combiner_method = combiner_method
        )

    if case_study == 'bacterium':
        return build_embeddings_matrix_for_bacterium_case_study(
            embedder = embedder,
            combiner_method = combiner_method
        )
    
    raise Exception(f"case study {case_study} not supported")


def build_embeddings_matrix_for_bacterium_case_study(
        embedder : str, 
        combiner_method : str,
    ) -> tuple[list[str] , np.array]:

    records = list(SeqIO.parse("dataset/bacterium/bacterium_tr.fasta", "fasta"))
    
    IDs = []
    embeddings_matrix = []

    for record in records:
        
        if not os.path.exists(f"/storagenfs/m.tolloso/BioEmbedding/dataset/bacterium/embeddings/{embedder}/{record.id}.npy"):
            raise Exception(f"file {record.id}.npy not found") # in this case study all the embeddings should exist

        try:
            raw_embedding = np.load(f"/storagenfs/m.tolloso/BioEmbedding/dataset/bacterium/embeddings/{embedder}/{record.id}.npy", allow_pickle=True)
        except:
            print(f"Error while loading the embedding of sequence {record.id} from embedder {embedder}")
            raise
        assert len(raw_embedding.shape) == 3

        final_embedding = combiner(
            raw_embedding = raw_embedding,
            method = combiner_method
        )
        
        embeddings_matrix.append(final_embedding)
        
        IDs.append(record.id)

    return IDs, np.array(embeddings_matrix)



def build_embeddings_matrix_for_satb2_case_study(
        embedder : str, 
        combiner_method : str,
    ) -> tuple[list[str] , np.array]:
      
    records = list(SeqIO.parse("dataset/satb2/satb2_tr.txt", "fasta"))
    
    IDs = []
    embeddings_matrix = []

    for record in records:
  
        gene_ID = record.description.split(' ')[0]
        protein_ID = record.description.split('protein:')[1].split(' ')[0]

        if embedder == 'dnabert':
            name_to_use = gene_ID
        else:
            name_to_use = protein_ID

        if not os.path.exists(f"dataset/satb2/embeddings/{embedder}/" + name_to_use + ".npy"):
            continue
        
        raw_embedding = None
        try:
            raw_embedding = np.load(f"dataset/satb2/embeddings/{embedder}/" + name_to_use + ".npy", allow_pickle=True)
        except:
            print(f"Error while loading the embedding of sequence {name_to_use} from embedder {embedder}")
            raise
        
        assert len(raw_embedding.shape) == 3
        # the embedding is of the shape (number_of_chunks, chunks_len, embedding_dimension)

        # combine in a single (embedding_dimension) representation
        final_embedding = combiner(
            raw_embedding = raw_embedding,
            method = combiner_method
        )
        
        embeddings_matrix.append(final_embedding)
        
        IDs.append(gene_ID)

    return IDs, np.array(embeddings_matrix)




def build_embeddings_matrix_for_mouse_case_study(  
        embedder : str, 
        combiner_method : str,
    ) -> tuple[list[str] , np.array]:

    up_records = list(SeqIO.parse("dataset/mouse/mouse.xml", "uniprot-xml"))

    IDs = []
    embeddings_matrix = []

    if embedder == 'dnabert':

        for record in up_records:
            gene_ID = utils.get_gene_id(record)
            if gene_ID is None or \
                gene_ID in IDs: # the protein has not a gene id # TODO why?
                continue

            if not os.path.exists("dataset/mouse/embeddings/dnabert/" + gene_ID + ".npy"): # not able to find the transpriptome for the geneid? # TODO why
                continue
            
            raw_embedding = None
            try:
                raw_embedding = np.load("dataset/mouse/embeddings/dnabert/" + gene_ID + ".npy", allow_pickle=True)
            except:
                print(f"Error while loading the embedding of sequence {gene_ID} from embedder {embedder}")
                raise

            assert len(raw_embedding.shape) == 3
            # the embedding is of the shape (number_of_chunks, chunks_len, embedding_dimension)

            # combine in a single (embedding_dimension) representation
            final_embedding = combiner(
                raw_embedding = raw_embedding,
                method = combiner_method
            )
            
            embeddings_matrix.append(final_embedding)

            if gene_ID in IDs:
                print(gene_ID)

            IDs.append(gene_ID)


        return IDs, np.array(embeddings_matrix)
    
    else:
        
        for record in up_records:
            
            gene_ID = utils.get_gene_id(record)

            if gene_ID is None or \
                not os.path.exists("dataset/mouse/embeddings/dnabert/" + gene_ID + ".npy") or \
                gene_ID in IDs:
                # if the protein sequence does not have a gene id or the transriptome does not exist
                continue

            protein_name = f'sp|{record.id}|{record.name}'.replace('|', '_')

            raw_embedding = None
            try:
                raw_embedding = np.load(f"dataset/mouse/embeddings/{embedder}/" + protein_name + ".npy", allow_pickle=True)
            except:
                print(f"Error while loading the embedding of sequence {protein_name} from embedder {embedder}")
                raise

            assert len(raw_embedding.shape) == 3
            # the embedding is of the shape (number_of_chunks, chunks_len, embedding_dimension)

            # combine in a single (embedding_dimension) representation
            final_embedding = combiner(
                raw_embedding = raw_embedding,
                method = combiner_method
            )
            
            embeddings_matrix.append(final_embedding)

            IDs.append(gene_ID)

        return IDs, np.array(embeddings_matrix)



def build_embeddings_matrix_for_hemoglobin_case_study(  
        embedder : str, 
        combiner_method : str,
    ) -> tuple[list[str] , np.array]:

    def get_gene_id(record):
        for i in record.dbxrefs:
            if i.startswith("GeneID:"):
                geneID = i.split(":")[1]
                return geneID
        return None

    up_records = list(SeqIO.parse("dataset/hemoglobin/hemoglobin.xml", "uniprot-xml"))

    IDs = []
    embeddings_matrix = []

    if embedder == 'dnabert':

        for record in up_records:
            gene_ID = get_gene_id(record)
            if gene_ID is None: # the protein has not a gene id # TODO why?
                continue

            if not os.path.exists("dataset/hemoglobin/embeddings/dnabert/" + gene_ID + ".npy"): # not able to find the transpriptome for the geneid? # TODO why
                continue
            
            raw_embedding = None
            try:
                raw_embedding = np.load("dataset/hemoglobin/embeddings/dnabert/" + gene_ID + ".npy", allow_pickle=True)
            except:
                print(f"Error while loading the embedding of sequence {gene_ID} from embedder {embedder}")
                raise

            assert len(raw_embedding.shape) == 3
            # the embedding is of the shape (number_of_chunks, chunks_len, embedding_dimension)

            # combine in a single (embedding_dimension) representation
            final_embedding = combiner(
                raw_embedding = raw_embedding,
                method = combiner_method
            )
            
            embeddings_matrix.append(final_embedding)

            IDs.append(gene_ID)
        
        return IDs, np.array(embeddings_matrix)
    
    else:
        
        for record in up_records:
            
            gene_ID = get_gene_id(record)

            if gene_ID is None or not os.path.exists("dataset/hemoglobin/embeddings/dnabert/" + gene_ID + ".npy"):
                # if the protein sequence does not have a gene id or the transriptome does not exist
                continue

            protein_name = f'sp|{record.id}|{record.name}'.replace('|', '_')
            
            raw_embedding = None
            try:
                raw_embedding = np.load(f"dataset/hemoglobin/embeddings/{embedder}/" + protein_name + ".npy", allow_pickle=True)
            except:
                print(f"Error while loading the embedding of sequence {protein_name} from embedder {embedder}")
                raise

            assert len(raw_embedding.shape) == 3
            # the embedding is of the shape (number_of_chunks, chunks_len, embedding_dimension)

            # combine in a single (embedding_dimension) representation
            final_embedding = combiner(
                raw_embedding = raw_embedding,
                method = combiner_method
            )
            
            embeddings_matrix.append(final_embedding)

            IDs.append(gene_ID)

        return IDs, np.array(embeddings_matrix)



def build_embeddings_matrix_for_covid19_case_study(
        embedder : str,
        combiner_method : str
    ) -> tuple[list[str] , np.array]:
    
    """
    The data for covid19 are in the covid19.gb file, there are 77 transcripts each one coding 11 proteins
    """

    gb_records = list(SeqIO.parse('dataset/covid19/covid19.gb','genbank'))

    if embedder == 'dnabert':
        
        IDs = [record.id for record in gb_records] # transriptome ids, for each one we have a dnabert embedding
        embeddings_matrix = []

        for id in IDs:
            # load the file as a numpy array
            raw_embedding = None
            try:
                raw_embedding = np.load("dataset/covid19/embeddings/dnabert/" + id + ".npy", allow_pickle=True)
            except:
                print(f"Error while loading the embedding of sequence {id} from embedder {embedder}")
                raise

            assert len(raw_embedding.shape) == 3
            # the embedding is of the shape (number_of_chunks, chunks_len, embedding_dimension)

            # combine in a single (embedding_dimension) representation
            final_embedding = combiner(
                raw_embedding = raw_embedding,
                method = combiner_method
            )
            
            embeddings_matrix.append(final_embedding)
        
        return IDs, np.array(embeddings_matrix)
    
    else: 
        # protein embedding: the embeddings of the 11 proteins for each transcript must be aggregated

        embeddings_matrix = []
        IDs = [] # transcriptome ids
          
        for gb_record in gb_records:
            # for each transcript
            IDs.append(gb_record.id)

            proteins_embedding = []

            for feature in gb_record.features:
                if feature.type == 'CDS': # coding sequence
                    protein_name = feature.qualifiers['protein_id'][0]
                    #for each protein

                    raw_embedding = None
                    try:
                        raw_embedding = np.load(f"dataset/covid19/embeddings/{embedder}/" + protein_name + ".npy", allow_pickle=True)
                    
                    # if the file does not exist is not a problem, it means that the protein is a subset of another protein so the embedding was not computed
                    except OSError:
                        continue      
                    except:
                        print(f"Error while loading the embedding of sequence {protein_name} from embedder {embedder}")
                        raise

                    assert len(raw_embedding.shape) == 3
                    # the embedding is of the shape (number_of_chunks, chunks_len, embedding_dimension)

                    # combine in a single (embedding_dimension) representation
                    protein_embedding = combiner(
                        raw_embedding = raw_embedding,
                        method = combiner_method
                    )

                    proteins_embedding.append(protein_embedding)
                
            # now proteins_embedding is a list of (embedding_dimension) vectors, each representing a protein
            # we need to aggregate them in a single representation for the transcript
            # we can use the same combiner, but first we have to transform the matrix in a tensor (1, num_proteins(11), embedding_dimension)
            
            proteins_embedding = np.array([proteins_embedding])


            final_embedding = combiner(
                raw_embedding = proteins_embedding,
                method = combiner_method
            )

            embeddings_matrix.append(final_embedding)
        
        return IDs, np.array(embeddings_matrix)
        


if __name__ == '__main__':

    r = build_embeddings_matrix_for_satb2_case_study(
        embedder = "prose",
        combiner_method = "average"
    )

    print(r[1].shape)


                
            
