from autoembedding.ExecutionTree import ExecutionTree
import autoembedding.utils as utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from autoembedding.embeddings_matrix import build_embeddings_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import adjusted_rand_score
import numpy as np
from autoembedding.results_manager import results2file


def main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH):    
    
    et = ExecutionTree(input = {"embeddings_path" : EMBEDDINGS_PATH} )

    # BUILD EMBEDDING MATRIX (WITH COMBINER)
    def pipeline_build_embeddings_matrix(previous_stage_output : dict, embedder: str, combiner_method : str) -> dict:
        
        """
        Built the embeddings matrix from the embeddings dict

        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the embeddings dict
            embedder (str): The embedder to use
            combiner_method (str): The combiner method to use, the effect of this parameter depends on the embedder
        Returns:
            dict: A dict containing the IDs and the embeddings matrix where each row is the embedding of the corresponding ID
        """

        embeddings_path = previous_stage_output["embeddings_path"]

        embeddings_IDs, embeddings_matrix = build_embeddings_matrix(
            embeddings_path=embeddings_path,
            embedder=embedder,
            combiner_method=combiner_method
        )
        return {"embeddings_IDs": embeddings_IDs, "embeddings_matrix": embeddings_matrix}

    et.add_multistage(
        function=pipeline_build_embeddings_matrix,
        list_args=[ 
            # {"embedder" : "rep", "combiner_method" : "pca" },
            # {"embedder" : "rep", "combiner_method" : "average" },
            # {"embedder" : "rep", "combiner_method" : "sum" },
            # {"embedder" : "rep", "combiner_method" : "max" },

            {"embedder" : "seqvec", "combiner_method" : "pca" },
            {"embedder" : "seqvec", "combiner_method" : "average" },
            {"embedder" : "seqvec", "combiner_method" : "sum" },
            {"embedder" : "seqvec", "combiner_method" : "max" },
            
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

            {"embedder" : "esm", "combiner_method" : "pca" },
            {"embedder" : "esm", "combiner_method" : "average" },
            {"embedder" : "esm", "combiner_method" : "sum" },
            {"embedder" : "esm", "combiner_method" : "max" },
            
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
        embeddings_IDs = previous_stage_output["embeddings_IDs"]

        if n_components != "all":
            scaler = StandardScaler()
            embeddings_matrix = scaler.fit_transform(embeddings_matrix)
            if n_components == "default":
                n_components = min(embeddings_matrix.shape)
            pca = PCA(n_components=n_components)
            embeddings_matrix = pca.fit_transform(embeddings_matrix)
        
        return { "embeddings_matrix" : embeddings_matrix, "embeddings_IDs": embeddings_IDs}

    et.add_multistage(
        function=pipeline_pca, # TODO t-sne
        list_args=[
            {"n_components": 10},
            {"n_components": 30},
            {"n_components": 50},
            {"n_components": 70},
            {"n_components": 90},
            {"n_components": "all"},       
        ]
    )

    # BUILD LINKAGE MATRIX

    def pipeline_build_embeddings_linkage_matrix(previous_stage_output : dict, metric, method)-> dict:
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
        embeddings_IDs = previous_stage_output["embeddings_IDs"]
        embeddings_linkage_matrix = linkage(embeddings_matrix, method=method, metric=metric)
        return { "embeddings_linkage_matrix" : embeddings_linkage_matrix, "embeddings_IDs": embeddings_IDs}

    et.add_multistage(
        function=pipeline_build_embeddings_linkage_matrix,
        list_args=[
            {"metric" : "euclidean", "method" : "average"},
            {"metric" : "euclidean", "method" : "complete"},
            {"metric" : "euclidean", "method" : "ward"},
            {"metric" : "euclidean", "method" : "centroid"},
            {"metric" : "euclidean", "method" : "single"},
            {"metric" : "euclidean", "method" : "median"},
            
            {"metric" : "cosine", "method" : "average"},
            {"metric" : "cosine", "method" : "complete"},
            {"metric" : "cosine", "method" : "single"},
        ]
    )

    def pipeline_build_gt_linkage_matrix(
        ground_true_path : str, 
        previous_stage_output : dict, 
        metric, 
        method)-> dict:
        
        embeddings_linkage_matrix = previous_stage_output["embeddings_linkage_matrix"]
        embeddings_IDs = previous_stage_output["embeddings_IDs"]
        
        
        gtrue_IDs, gt_distances = utils.read_distance_matrix(ground_true_path)
        gt_distances = squareform(gt_distances)
        gtrue_linkage_matrix = linkage(gt_distances, method=method, metric=metric)

        return { "embeddings_linkage_matrix" : embeddings_linkage_matrix, "embeddings_IDs": embeddings_IDs, 
                "gtrue_linkage_matrix" : gtrue_linkage_matrix, "gtrue_IDs" : gtrue_IDs}


    et.add_multistage(
        function=pipeline_build_gt_linkage_matrix,
        fixed_args={"ground_true_path" : GROUND_TRUE_PATH},
        list_args=[
            {"metric" : "euclidean", "method" : "average"},
            {"metric" : "euclidean", "method" : "complete"},
            {"metric" : "euclidean", "method" : "ward"},
            {"metric" : "euclidean", "method" : "centroid"},
            {"metric" : "euclidean", "method" : "single"},
            {"metric" : "euclidean", "method" : "median"},
            
            {"metric" : "cosine", "method" : "average"},
            {"metric" : "cosine", "method" : "complete"},
            {"metric" : "cosine", "method" : "single"},
        ]
    )

    # COMPARE WITH GROUND TRUE

    def pipeline_mean_adjusted_rand_score(previous_stage_output : dict, cluster_range : tuple ) -> dict:
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
        embeddings_linkage_matrix = previous_stage_output["embeddings_linkage_matrix"]
        embeddings_IDs = previous_stage_output["embeddings_IDs"]
        gtrue_linkage_matrix = previous_stage_output["gtrue_linkage_matrix"]
        gtrue_IDs = previous_stage_output["gtrue_IDs"]

        if len(embeddings_IDs) != len(gtrue_IDs):
            raise Exception("The number of IDs in the ground true and the predicted clustering is different")
        if cluster_range[0] < 2:
            raise Exception("Cluster range must start at least from 2")
        if not set(embeddings_IDs) == set(gtrue_IDs):
            raise Exception("The IDs in the ground true and the predicted clustering are different")

        start = cluster_range[0]
        end = cluster_range[1]
        
        if end == -1:
            end = min(len(embeddings_IDs), len(gtrue_IDs))
        
        predict_labels_matrix= cut_tree(embeddings_linkage_matrix, n_clusters=range(start, end))
        gtrue_labels_matrix = cut_tree(gtrue_linkage_matrix, n_clusters=range(start, end))

        # order the matrix rows based on the IDs
        predict_labels_matrix = predict_labels_matrix[np.argsort(embeddings_IDs)]
        gtrue_labels_matrix = gtrue_labels_matrix[np.argsort(gtrue_IDs)]

        # for each iteration, extract the relative column and compute the adjusted rand score
        adjusted_rand_scores = []
        for i in range(predict_labels_matrix.shape[1]):
            adjusted_rand_scores.append(adjusted_rand_score(predict_labels_matrix[:,i], gtrue_labels_matrix[:,i]))
        
        return {"mean_adjusted_rand_score" : np.mean(adjusted_rand_scores)}
        


    et.add_stage(
        function=pipeline_mean_adjusted_rand_score,
        args={"cluster_range" : (2, -1)}
    )

    # END

    return et


if __name__ == "__main__":
    
    EMBEDDINGS_PATH = "./dataset/emoglobina/embeddings"
    GROUND_TRUE_PATH = "./dataset/emoglobina/emoglobina.dist"

    # EMBEDDINGS_PATH =  "./dataset/batterio/embeddings"
    # GROUND_TRUE_PATH = "./dataset/batterio/batterio.dist"

    # EMBEDDINGS_PATH =  "./dataset/topo/embeddings"
    # GROUND_TRUE_PATH = "./dataset/topo/topo.dist"
    
    
    et = main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH)
    et.compute()
    
    r = et.get_results()

    file_name = "./results/"+ "phylogenetic_" +"results_" + GROUND_TRUE_PATH.split("/")[-1].split(".")[0]

    et.dump_results(r, file_name)

    results2file(r, file_name)

    
    