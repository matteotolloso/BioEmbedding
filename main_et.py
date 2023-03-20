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
        {"embedder" : "prose", "combiner_method" : "sum" },
        ]
    )


    # PRINCIPAL COMPONENT ANALYSIS (with scaling)

    def pipeline_pca(previous_stage_output : dict, n_components) -> dict:
        embeddings_matrix = previous_stage_output["embeddings_matrix"]
        IDs = previous_stage_output["IDs"]
        scaler = StandardScaler()
        embeddings_matrix = scaler.fit_transform(embeddings_matrix)
        pca = PCA(n_components=n_components)
        embeddings_matrix = pca.fit_transform(embeddings_matrix)
        return { "embeddings_matrix" : embeddings_matrix, "IDs": IDs}

    et.add_stage(
        function=pipeline_pca,
        args={"n_components": 6}
    )

    # BUILD LINKAGE MATRIX

    def pipeline_build_linkage_matrix(previous_stage_output : dict, metric, method)-> dict:
        embeddings_matrix = previous_stage_output["embeddings_matrix"]
        IDs = previous_stage_output["IDs"]
        condensed_distances = pdist(embeddings_matrix, metric=metric)
        linkage_matrix = linkage(condensed_distances, method=method)
        return { "linkage_matrix" : linkage_matrix, "IDs": IDs}

    et.add_stage(
        function=pipeline_build_linkage_matrix,
        args={"metric" : "euclidean", "method" : "average"}
    )

    # COMPARE WITH GROUND TRUE

    def pipeline_mean_adjusted_rand_score(previous_stage_output : dict, cluster_range : tuple, ground_true_path ) -> dict:
        predict_linkage_matrix = previous_stage_output["linkage_matrix"]
        predict_IDs = previous_stage_output["IDs"]
        gtrue_matrix, gtrue_IDs = utils.newick_to_linkage(ground_true_path)
        
        def ars_fixed_clusters(curr_clusters, _predict_linkage_matrix, _gtrue_matrix, _predict_IDs, _gtrue_IDs):
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
        
        sum = 0
        for n_clusters in range(cluster_range[0],cluster_range[1]+1):
            sum+= ars_fixed_clusters(n_clusters, predict_linkage_matrix, gtrue_matrix, predict_IDs, gtrue_IDs)
        mean = sum / (cluster_range[1] - cluster_range[0] +1)
        
        return {"mean_adjusted_rand_score" : mean}

    et.add_stage(
        function=pipeline_mean_adjusted_rand_score,
        args={"cluster_range" : (2, 5), "ground_true_path" : GROUND_TRUE_PATH}
    )

    # END

    return et


if __name__ == "__main__":
    EMBEDDINGS_PATH = "./dataset/globins/globins.json"
    GROUND_TRUE_PATH = "./dataset/globins/globins.dnd"
    et = main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH)
    et.compute()
    print(et.get_results())