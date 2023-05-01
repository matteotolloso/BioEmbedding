from ExecutionTree import ExecutionTree
import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from autoembedding.embeddings_matrix import build_embeddings_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import adjusted_rand_score
import numpy as np
from Bio import SeqIO
from results_manager import results2file


def main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH):
    # Loading the embeddings dict
    embeddings_dict = utils.get_embeddings_dict(EMBEDDINGS_PATH)
    et = ExecutionTree(input = {"embeddings_dict" : embeddings_dict} )

    # BUILD EMBEDDING MATRIX (WITH COMBINER)

    def pipeline_build_embeddings_matrix(previous_stage_output : dict, embedder, combiner_method) -> dict:
        
        """
        Built the embeddings matrix from the embeddings dict

        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the embeddings dict
            embedder (str): The embedder to use
            combiner_method (str): The combiner method to use, the effect of this parameter depends on the embedder
        Returns:
            dict: A dict containing the IDs and the embeddings matrix where each row is the embedding of the corresponding ID
        """

        embeddings_dict = previous_stage_output["embeddings_dict"]

        IDs, embeddings_matrix = build_embeddings_matrix(
            embeddings_dict=embeddings_dict,
            embedder=embedder,
            combiner_method=combiner_method
        )
        return {"IDs": IDs, "embeddings_matrix": embeddings_matrix}

    et.add_multistage(
        function=pipeline_build_embeddings_matrix,
        list_args=[
        {"embedder" : "rep", "combiner_method" : "pca" },
        {"embedder" : "rep", "combiner_method" : "average" },
        {"embedder" : "rep", "combiner_method" : "sum" },
        {"embedder" : "rep", "combiner_method" : "max" },
        
        {"embedder" : "dnabert", "combiner_method" : "pca" },
        {"embedder" : "dnabert", "combiner_method" : "average" },
        {"embedder" : "dnabert", "combiner_method" : "sum" },
        {"embedder" : "dnabert", "combiner_method" : "max" },
        
        {"embedder" : "prose", "combiner_method" : "pca" },
        {"embedder" : "prose", "combiner_method" : "average" },
        {"embedder" : "prose", "combiner_method" : "sum" },
        {"embedder" : "prose", "combiner_method" : "max" },

        {"embedder" : "alphafold", "combiner_method" : "pca" },
        {"embedder" : "alphafold", "combiner_method" : "average" },
        {"embedder" : "alphafold", "combiner_method" : "sum" },
        {"embedder" : "alphafold", "combiner_method" : "max" },
        
        ]
    )


    # PRINCIPAL COMPONENT ANALYSIS (with scaling)

    def pipeline_pca(previous_stage_output : dict, n_components) -> dict:

        """
        Performs PCA on the embeddings matrix after scaling it

        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the embeddings matrix and the IDs
            n_components (int): The number of components to keep, if "default", the number of components will be (min(embeddings_matrix.shape)).
            if "all", the PCA will not be performed
        Returns:
            dict: A dict containing the IDs and the embeddings matrix
        """

        embeddings_matrix = previous_stage_output["embeddings_matrix"]
        IDs = previous_stage_output["IDs"]

        if n_components == "all":
            return { "embeddings_matrix" : embeddings_matrix, "IDs": IDs}
        
        scaler = StandardScaler()
        embeddings_matrix = scaler.fit_transform(embeddings_matrix)
        if n_components == "default":
            n_components = min(embeddings_matrix.shape)
        pca = PCA(n_components=n_components)
        embeddings_matrix = pca.fit_transform(embeddings_matrix)
        
        return { "embeddings_matrix" : embeddings_matrix, "IDs": IDs}

    et.add_multistage(
        function=pipeline_pca,
        list_args=[
        {"n_components": "default"},
        {"n_components": "all"},
        ]
    )

    # BUILD LINKAGE MATRIX

    def pipeline_enrichment(previous_stage_output : dict, metric, method, ground_true_path, cluster_range)-> dict:
        """
        Performs the enrichment analysis comparing the number of common annotations between sequences and the distance in the ebmeddings space
        
        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the embeddings matrix and the IDs
            metric (str): The metric to use (euclidean, cosine, etc.)
        Returns:
            dict: A dict containing the linkage matrix and the IDs
        """

        embeddings_matrix = previous_stage_output["embeddings_matrix"]
        IDs = previous_stage_output["IDs"]

        embedding_distances = pdist(embeddings_matrix, metric=metric) # distances of the embeddings in the embedding space

        # preparing the matrix distance in the "enrichment space"
        records = list(SeqIO.parse(ground_true_path, "uniprot-xml"))
        annotation_dict = {}
        for record in records:
            name = f'sp|{record.id}|{record.name}'      # must be the same as the one in the embedding matrix parsed from the fasta file
            annotation_dict[name] = {}
            go_annotations = [i for i in record.dbxrefs if i.startswith('GO')]
            annotation_dict[name]['go'] = go_annotations
            annotation_dict[name]['keywords'] = record.annotations['keywords']
            annotation_dict[name]['taxonomy'] = record.annotations['taxonomy']
        gt_distances = np.zeros((len(IDs), len(IDs)))
        for i, name_i in enumerate(IDs):
            for j, name_j in enumerate(IDs):
                # compute the number of common annotations
                gt_distances[i][j] += len(set(annotation_dict[name_i]['go']).intersection(set(annotation_dict[name_j]['go'])))
                gt_distances[i][j] += len(set(annotation_dict[name_i]['keywords']).intersection(set(annotation_dict[name_j]['keywords'])))


        # the ground true distances is not a distance measure but a similarity, we have to make it a distance and also make the diagonal 0 (maybe not necessary)
        gt_distances = gt_distances.max() - gt_distances
        np.fill_diagonal(gt_distances, 0)

        # check is gt_distances is symmetric
        assert np.allclose(gt_distances, gt_distances.T, atol=1e-8)

        start = cluster_range[0]
        end = cluster_range[1]
        if end == -1:
            end = len(IDs)

        # make condensed distance matrices
        gt_distances = squareform(gt_distances)

        # compute the linkage matrices
        predict_linkage_matrix = linkage(embedding_distances, method=method)
        gt_linkage_matrix = linkage(gt_distances, method=method)

        # compute the labels matrix: an array of shape (n_samples, n_clusters) where n_clusters is the number of clusters in the range
        predict_labels_matrix= cut_tree(predict_linkage_matrix, n_clusters=range(start, end))
        gt_labels_matrix = cut_tree(gt_linkage_matrix, n_clusters=range(start, end))

        adjusted_rand_scores = []
        for i in range(predict_labels_matrix.shape[1]):
            adjusted_rand_scores.append(adjusted_rand_score(predict_labels_matrix[:,i], gt_labels_matrix[:,i]))

        return {"mean_adjusted_rand_score" : np.mean(adjusted_rand_scores)}


    et.add_stage(
        function=pipeline_enrichment,
        args={"metric" : "euclidean", "ground_true_path" : GROUND_TRUE_PATH, "method" : "ward", "cluster_range" : (2, -1)}
    )

    # END

    return et


if __name__ == "__main__":
    
    EMBEDDINGS_PATH = "./dataset/enrichment_test/proteins.json"
    GROUND_TRUE_PATH = "./dataset/enrichment_test/annotations.xml"

    #EMBEDDINGS_PATH = "./dataset/NEIS2157/NEIS2157.json"
    #GROUND_TRUE_PATH = "./dataset/NEIS2157/NEIS2157.dnd"
    
    
    et = main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH)
    et.compute()
    
    r = et.get_results()

    # get the name of the current file


    file_name = "./results/"+ "enrichment_"+"results_" + EMBEDDINGS_PATH.split("/")[-1].split(".")[0] 

    et.dump_results(r, file_name)

    results2file(r, file_name)

    
    