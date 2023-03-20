import numpy as np
import utils
from autoembedding.combiners import combiner_for_prose, combiner_for_dnabert, combiner_for_rep

def build_embeddings_matrix(
        embeddings_dict : dict, 
        embedder : str,
        combiner_method : str
    ) -> tuple[list[str] , np.array]:

    """
    Creates a numpy 2D matrix where each row is the embedding of a sequence. All the sequences that are keys of the embeddings_dict
    are present in the matrix ordered by their id, taking account of the version (so not a strictly lexicographic order).
    The number of columns depends on the embedder and the combining method.

    Returns:
        A tuple. The first element is a list of strings: the IDs ordered by version.
        The second element is a 2D np.array containig for each row the embedding of a sequence, in the same order of the IDs.

    """

    IDs = utils.get_ordered_ids(embeddings_dict)

    embeddings_matrix = []

    for id in IDs:

        final_embedding = []

        if embedder == "rep":
            final_embedding = combiner_for_rep(
                raw_embedding=np.array(embeddings_dict[id][embedder]),  # is a 64-dim array
                method = combiner_method
            )
        
        elif embedder == "dnabert":
            final_embedding = combiner_for_dnabert(
                raw_embedding = np.array(embeddings_dict[id][embedder]), # is a (seq_len % 512)*(768) matrix
                method = combiner_method
            )

        elif embedder == "prose":
            final_embedding = combiner_for_prose(
                raw_embedding = np.array(embeddings_dict[id][embedder]),  # is a (seq_len)*(100) matrix (each row is the embedding of an amminoacid)
                method = combiner_method
            ) 
        
        embeddings_matrix.append(final_embedding)

    return IDs, np.array(embeddings_matrix)
