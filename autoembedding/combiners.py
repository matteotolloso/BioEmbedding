import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def combiner_for_alphafold(raw_embedding : np.array , method: str)-> np.array :
    """
    Given ad embedding generated by "alphafold" (i.e. a matrix (sequence len)*(384) where each 384-dim vector is the embedding for a single nucleotide)
    it returns an embedding for the entire protein using some combination methods.
    """

    if method == "average":
        return np.mean(raw_embedding, axis=0)  # mean between all amminoacids
    elif method == "max":
        return np.max(raw_embedding, axis=0) 
    elif method == "sum":
        return np.sum(raw_embedding, axis=0)
    elif method == "pca":
        # transpose the vector to consider the columns as main elements, preform scaling and pca
        raw_embedding = np.transpose(raw_embedding)
        raw_embedding = StandardScaler().fit_transform(raw_embedding)
        pca = PCA(n_components=1)
        return pca.fit_transform(raw_embedding).reshape(-1)
        
    else:
        raise Exception(f"Unknown combining method: {method} for prose")

def combiner_for_prose(raw_embedding : np.array , method: str)-> np.array :
    """
    Given ad embedding generated by "prose" (i.e. a matrix (sequence len)*(100) where each 100-dim vector is the embedding for a single nucleotide)
    it returns an embedding for the entire protein using some combination methods.
    """

    if method == "average":
        return np.mean(raw_embedding, axis=0)  # mean between all amminoacids
    elif method == "max":
        return np.max(raw_embedding, axis=0) 
    elif method == "sum":
        return np.sum(raw_embedding, axis=0)
    elif method == "pca":
        # transpose the vector to consider the columns as main elements, preform scaling and pca
        raw_embedding = np.transpose(raw_embedding)
        raw_embedding = StandardScaler().fit_transform(raw_embedding)
        pca = PCA(n_components=1)
        return pca.fit_transform(raw_embedding).reshape(-1)
        
    else:
        raise Exception(f"Unknown combining method: {method} for prose")
    

def combiner_for_dnabert(raw_embedding: np.array, method: str) -> np.array:
    """
    Given ad embedding generated by "dnabert" (i.e. a matrix (sequence len / 510)*(768) where each 768-dim vector is the embedding of a consecutive subsequence of 510 nucleotides)
    it returns an embedding for the entire protein using some combination methods.
    """
    if raw_embedding.shape[1] == 0:
        raise Exception(f"Nothing to combine")

    if method == "average":
        return np.mean(raw_embedding, axis=0)  # mean between the embeddings of the subsequences
    elif method == "max":
        return np.max(raw_embedding, axis=0) 
    elif method == "sum":
        return np.sum(raw_embedding, axis=0)
    elif method == "pca":
        # transpose the vector to consider the columns as main elements, preform scaling and pca
        raw_embedding = np.transpose(raw_embedding)
        raw_embedding = StandardScaler().fit_transform(raw_embedding)
        pca = PCA(n_components=1)
        return pca.fit_transform(raw_embedding).reshape(-1)
    else:
        raise Exception(f"Unknown combining method: {method} for dnabert")
    

def combiner_for_rep(raw_embedding : np.array, method: str = "none") -> np.array:
    """
    Given ad embedding generated by "embedding_reproduction" (i.e. an 64-dim array)
    it returns an embedding for the entire protein, i.e. a simple copy of the array.
    """

    return raw_embedding