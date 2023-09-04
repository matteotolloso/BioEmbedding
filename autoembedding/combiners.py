import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def combiner(raw_embedding : np.array , method: str)-> np.array :
    """
    Given ad embedding generated by "esm" (i.e. a list of lists (chunks)*(chunk len)*(1280) where each 1280-dim vector is the embedding for a single nucleotide)
    it returns an embedding for the entire protein using some combination methods.
    """

    if method == "average":

        final_chunk_embedding = []
        for chunk_embedding in raw_embedding: # chunk embeddind: (chunk len) * (1280)
            final_chunk_embedding.append(np.mean(chunk_embedding, axis=0)) # append a single 1280-dim vector that embeds the chunk
        
        return np.mean(final_chunk_embedding, axis=0)  # mean between all chunks

    elif method == "max":
        
        final_chunk_embedding = []
        for chunk_embedding in raw_embedding:
            final_chunk_embedding.append(np.max(chunk_embedding, axis=0))

        return np.max(final_chunk_embedding, axis=0)
    
    elif method == "sum":
        
        final_chunk_embedding = []
        for chunk_embedding in raw_embedding:
            final_chunk_embedding.append(np.sum(chunk_embedding, axis=0))

        return np.sum(final_chunk_embedding, axis=0)
    
    elif method == "pca":

        final_chunk_embedding = []
        for chunk_embedding in raw_embedding:
            chunk_embedding = np.transpose(chunk_embedding)
            chunk_embedding = StandardScaler().fit_transform(chunk_embedding)
            pca = PCA(n_components=1)
            final_chunk_embedding.append(pca.fit_transform(chunk_embedding).reshape(-1))

        combined_embedding = np.transpose(final_chunk_embedding)
        combined_embedding = StandardScaler().fit_transform(combined_embedding)
        pca = PCA(n_components=1)
        return pca.fit_transform(combined_embedding).reshape(-1)
        
    else:
        raise Exception(f"Unknown combining method: {method} for esm")
