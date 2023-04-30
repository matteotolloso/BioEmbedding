from ExecutionTree import ExecutionTree
import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from autoembedding.embeddings_matrix import build_embeddings_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import adjusted_rand_score
import numpy as np
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

    def pipeline_build_linkage_matrix(previous_stage_output : dict, metric, method)-> dict:
        """
        Builds the linkage matrix (in the scipy format) from the embeddings matrix
        
        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the embeddings matrix and the IDs
            metric (str): The metric to use (euclidean, cosine, etc.)
            method (str): The method to use (average, complete, etc.)
        Returns:
            dict: A dict containing the linkage matrix and the IDs
        """


        embeddings_matrix = previous_stage_output["embeddings_matrix"]
        IDs = previous_stage_output["IDs"]
        condensed_distances = pdist(embeddings_matrix, metric=metric)
        linkage_matrix = linkage(condensed_distances, method=method)
        return { "linkage_matrix" : linkage_matrix, "IDs": IDs}

    et.add_multistage(
        function=pipeline_build_linkage_matrix,
        list_args=[
        {"metric" : "euclidean", "method" : "average"},
        #{"metric" : "euclidean", "method" : "complete"},
        #{"metric" : "euclidean", "method" : "ward"},
        #{"metric" : "euclidean", "method" : "centroid"},
        #{"metric" : "euclidean", "method" : "single"},
        #{"metric" : "euclidean", "method" : "median"},
        #
        #{"metric" : "cosine", "method" : "average"},
        #{"metric" : "cosine", "method" : "complete"},
        #{"metric" : "cosine", "method" : "ward"},
        #{"metric" : "cosine", "method" : "centroid"},
        #{"metric" : "cosine", "method" : "single"},
        #{"metric" : "cosine", "method" : "median"},
        ]
    )

    # COMPARE WITH GROUND TRUE

    def pipeline_mean_adjusted_rand_score(previous_stage_output : dict, cluster_range : tuple, ground_true_path ) -> dict:
        """
        Computes the mean adjusted rand score between two hierarchical clustering averaging over all the cut
        in a given range

        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the linkage matrix and the IDs
            cluster_range (tuple): The range of clusters to consider (the cut to the hierarchical clustering)
            ground_true_path (str): The path to the ground true newick file
        Returns:
            dict: A dict containing the mean adjusted rand score
        """
        predict_linkage_matrix = previous_stage_output["linkage_matrix"]
        predict_IDs = previous_stage_output["IDs"]
        gtrue_linkage_matrix, gtrue_IDs = utils.newick_to_linkage(ground_true_path)

        if len(predict_IDs) != len(gtrue_IDs):
            raise Exception("The number of IDs in the ground true and the predicted clustering is different")
        if cluster_range[0] < 2:
            raise Exception("Cluster range must start at least from 2")
        if not set(predict_IDs) == set(gtrue_IDs):
            raise Exception("The IDs in the ground true and the predicted clustering are different")

        start = cluster_range[0]
        end = cluster_range[1]
        
        if end == -1:
            end = min(len(predict_IDs), len(gtrue_IDs))
        
        predict_labels_matrix= cut_tree(predict_linkage_matrix, n_clusters=range(start, end))
        gtrue_labels_matrix = cut_tree(gtrue_linkage_matrix, n_clusters=range(start, end))

        # order the matrix rows based on the IDs
        predict_labels_matrix = predict_labels_matrix[np.argsort(predict_IDs)]
        gtrue_labels_matrix = gtrue_labels_matrix[np.argsort(gtrue_IDs)]

        # for each iteration, extract the relative column and compute the adjusted rand score
        adjusted_rand_scores = []
        for i in range(predict_labels_matrix.shape[1]):
            adjusted_rand_scores.append(adjusted_rand_score(predict_labels_matrix[:,i], gtrue_labels_matrix[:,i]))
        
        return {"mean_adjusted_rand_score" : np.mean(adjusted_rand_scores)}
        


    et.add_stage(
        function=pipeline_mean_adjusted_rand_score,
        args={"cluster_range" : (2, -1), "ground_true_path" : GROUND_TRUE_PATH}
    )

    # END

    return et


if __name__ == "__main__":
    
    EMBEDDINGS_PATH = "./dataset/globins/globins.json"
    GROUND_TRUE_PATH = "./dataset/globins/globins.dnd"

    #EMBEDDINGS_PATH = "./dataset/NEIS2157/NEIS2157.json"
    #GROUND_TRUE_PATH = "./dataset/NEIS2157/NEIS2157.dnd"
    
    
    et = main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH)
    et.compute()
    
    r = et.get_results()

    file_name = "./results/"+ "results_" + EMBEDDINGS_PATH.split("/")[-1].split(".")[0]

    et.dump_results(r, file_name)

    results2file(r, file_name)

    
    