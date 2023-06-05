import numpy as np
import autoembedding.utils as utils
from autoembedding.combiners import combiner

# TODO check id is more efficient to have each combiner in a separate file

def build_embeddings_matrix(
        embeddings_dict : dict, 
        embedder : str,
        combiner_method : str
    ) -> tuple[list[str] , np.array]:

    """
    Creates a numpy 2D matrix where each row is the embedding of a sequence. All the sequences that are keys of the embeddings_dict
    are present in the matrix.
    The number of columns depends on the embedder and the combining method.

    Returns:
        A tuple. The first element is a list of strings: the IDs.
        The second element is a 2D np.array containig for each row the embedding of a sequence, in the same order of the IDs.

    """

    IDs = list(embeddings_dict.keys())

    embeddings_matrix = []

    for id in IDs:

        final_embedding = []

        final_embedding = combiner(
            raw_embedding = embeddings_dict[id][embedder],  # is a 64-dim array
            method = combiner_method
        )
        
        embeddings_matrix.append(final_embedding)

    return IDs, np.array(embeddings_matrix)
