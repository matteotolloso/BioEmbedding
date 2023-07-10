import numpy as np
from autoembedding.combiners import combiner
from os import listdir
from os.path import isfile, join
from pathlib import Path


def build_embeddings_matrix(
        embeddings_path : str, 
        embedder : str,
        combiner_method : str
    ) -> tuple[list[str] , np.array]:

    """
    Creates a numpy 2D matrix where each row is the embedding of a sequence.
    The number of columns depends on the embedder and the combining method.

    Returns:
        A tuple. The first element is a list of strings: the IDs.
        The second element is a 2D np.array containig for each row the embedding of a sequence, in the same order of the IDs.

    """

    folder_path = embeddings_path + "/" + embedder

    # list of files in the folder without the extension
    IDs = [Path(f).stem for f in listdir(folder_path) if isfile(join(folder_path, f))]

    embeddings_matrix = []

    for id in IDs:

        # load the file as a numpy array
        raw_embedding = None
        try:
            raw_embedding = np.load(folder_path + "/" + id + ".npy") 
        except:
            print(f"Error while loading the embedding of sequence {id} from embedder {embedder}")
            raise

        final_embedding = combiner(
            raw_embedding = raw_embedding,
            method = combiner_method
        )
        
        embeddings_matrix.append(final_embedding)

    return IDs, np.array(embeddings_matrix)
