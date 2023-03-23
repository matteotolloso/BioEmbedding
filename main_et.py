from ExecutionTree import ExecutionTree
import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from autoembedding.embeddings_matrix import build_embeddings_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import adjusted_rand_score


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
        {"embedder" : "rep", "combiner_method" : "none" },
        {"embedder" : "dnabert", "combiner_method" : "average" },
        {"embedder" : "dnabert", "combiner_method" : "cut" },
        {"embedder" : "prose", "combiner_method" : "sum" },
        {"embedder" : "prose", "combiner_method" : "max" },
        {"embedder" : "prose", "combiner_method" : "average" },
        ]
    )


    # PRINCIPAL COMPONENT ANALYSIS (with scaling)

    def pipeline_pca(previous_stage_output : dict, n_components) -> dict:

        """
        Performs PCA on the embeddings matrix after scaling it

        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the embeddings matrix and the IDs
            n_components (int): The number of components to keep, if -1, keep all components possible (min(embeddings_matrix.shape))
        Returns:
            dict: A dict containing the IDs and the embeddings matrix
        """

        embeddings_matrix = previous_stage_output["embeddings_matrix"]
        IDs = previous_stage_output["IDs"]
        scaler = StandardScaler()
        embeddings_matrix = scaler.fit_transform(embeddings_matrix)
        if n_components == -1:
            n_components = min(embeddings_matrix.shape)
        pca = PCA(n_components=n_components)
        embeddings_matrix = pca.fit_transform(embeddings_matrix)
        return { "embeddings_matrix" : embeddings_matrix, "IDs": IDs}

    et.add_multistage(
        function=pipeline_pca,
        list_args=[
        {"n_components": -1},
        {"n_components": 3},
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
        {"metric" : "euclidean", "method" : "complete"},
        {"metric" : "euclidean", "method" : "ward"},
        {"metric" : "euclidean", "method" : "centroid"},
        {"metric" : "euclidean", "method" : "single"},
        {"metric" : "euclidean", "method" : "median"},
        
        {"metric" : "cosine", "method" : "average"},
        {"metric" : "cosine", "method" : "complete"},
        {"metric" : "cosine", "method" : "ward"},
        {"metric" : "cosine", "method" : "centroid"},
        {"metric" : "cosine", "method" : "single"},
        {"metric" : "cosine", "method" : "median"},
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
        gtrue_matrix, gtrue_IDs = utils.newick_to_linkage(ground_true_path)

        if len(predict_IDs) != len(gtrue_IDs):
            raise Exception("Predicted IDs and ground true IDs have different lengths")
        
        def ars_fixed_clusters(curr_clusters, _predict_linkage_matrix, _gtrue_matrix, _predict_IDs, _gtrue_IDs):
            """
            Computes the adjusted rand score for a fixed number of clusters
            """
            predict_labels = cut_tree(_predict_linkage_matrix, n_clusters=curr_clusters)
            gtrue_labels = cut_tree(_gtrue_matrix, n_clusters=curr_clusters)
            predict_labels= [t[0] for t in predict_labels]
            gtrue_labels = [t[0] for t in gtrue_labels]
            predict_zip = [(name, cluster) for name, cluster in zip(_predict_IDs, predict_labels)]
            gtrue_zip = [(name, cluster) for name, cluster in zip(_gtrue_IDs, gtrue_labels)]
            predict_zip.sort(key=lambda x: x[0])
            gtrue_zip.sort(key=lambda x : x[0])
            predict_labels = [t[1] for t in predict_zip]
            gtrue_labels = [t[1] for t in gtrue_zip]
            return adjusted_rand_score(labels_true=gtrue_labels, labels_pred=predict_labels)
        
        start = cluster_range[0]
        end = cluster_range[1]
        
        if end == -1:
            end = min(len(predict_IDs), len(gtrue_IDs))
        
        sum = 0
        for n_clusters in range(start, end):
            sum+= ars_fixed_clusters(n_clusters, predict_linkage_matrix, gtrue_matrix, predict_IDs, gtrue_IDs)
        mean = sum / (end - start)
        
        return {"mean_adjusted_rand_score" : mean}

    et.add_stage(
        function=pipeline_mean_adjusted_rand_score,
        args={"cluster_range" : (2, -1), "ground_true_path" : GROUND_TRUE_PATH}
    )

    # END

    return et


if __name__ == "__main__":
    
    #EMBEDDINGS_PATH = "./dataset/globins/globins.json"
    #GROUND_TRUE_PATH = "./dataset/globins/globins.dnd"

    EMBEDDINGS_PATH = "./dataset/NEIS2157/NEIS2157.json"
    GROUND_TRUE_PATH = "./dataset/NEIS2157/NEIS2157.dnd"
    
    
    et = main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH)
    et.compute()
    
    
    r = et.get_results()
    r.sort(key=lambda x : x[0]["mean_adjusted_rand_score"], reverse=True)
    # save the list to a file
    with open('results.txt', 'w') as f:
        for result, pipeline in r:
            f.write(f"Score: {result['mean_adjusted_rand_score']}\n")
            for stage in pipeline:
                f.write(f"{stage}\n")
            f.write("\n")