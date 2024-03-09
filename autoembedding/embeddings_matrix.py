import numpy as np
from autoembedding.combiners import combiner
from Bio import SeqIO



def build_embeddings_matrix(
        case_study : str,
        embedder : str,
        combiner_method : str
    ) -> tuple[list[str] , np.array]:

    if case_study == 'covid19':
        return build_embeddings_matrix_for_covid19_case_study(
            embedder = embedder,
            combiner_method = combiner_method
        )
    
    raise Exception(f"case study {case_study} not supported")
    


def build_embeddings_matrix_for_covid19_case_study(
        embedder : str,
        combiner_method : str
    ) -> tuple[list[str] , np.array]:
    
    """
    The data for covid19 are in the covid19.gb file, there are 77 transcripts each one coding 11 proteins
    """

    print('build_embeddings_matrix_for_covid19_case_study')
    print(f"embedder: {embedder}")
    print(f"combiner_method: {combiner_method}")

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
    r = build_embeddings_matrix_for_covid19_case_study(
        embedder = "esm",
        combiner_method = "average"
    )

    print(r[1].shape)


                
            
