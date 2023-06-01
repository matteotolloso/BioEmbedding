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
from Bio import SeqIO
from autoembedding.results_manager import results2file


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

        embeddings_IDs, embeddings_matrix = build_embeddings_matrix(
            embeddings_dict=embeddings_dict,
            embedder=embedder,
            combiner_method=combiner_method
        )
        return {"embeddings_IDs": embeddings_IDs, "embeddings_matrix": embeddings_matrix}

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
        function=pipeline_pca,
        list_args=[
            {"n_components": "default"},
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
        
        return { 
            "embeddings_linkage_matrix" : embeddings_linkage_matrix, 
            "embeddings_IDs": embeddings_IDs
        }

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

    
    def pipeline_build_gt_linkage_matrix(previous_stage_output : dict, metric, method, edge_weight, ground_true_path)-> dict:

        embeddings_linkage_matrix = previous_stage_output["embeddings_linkage_matrix"]
        embeddings_IDs = previous_stage_output["embeddings_IDs"]
        
        # preparing the matrix distance in the "enrichment space"
        records = list(SeqIO.parse(ground_true_path, "uniprot-xml"))

        assert len(records) == len(embeddings_IDs), "The number of records in the ground true file must be the same as the number of embeddings"

        annotation_dict = {}
        
        for record in records:
            name = f'sp|{record.id}|{record.name}'      # must be the same as the one in the embedding matrix parsed from the fasta file
            annotation_dict[name] = {}
            go_annotations = [i for i in record.dbxrefs if i.startswith('GO')]
            annotation_dict[name]['go'] = go_annotations
            annotation_dict[name]['keywords'] = record.annotations['keywords']
            annotation_dict[name]['taxonomy'] = record.annotations['taxonomy']
        gtrue_distance_matrix = np.zeros((len(embeddings_IDs), len(embeddings_IDs)))

        for i, name_i in enumerate(embeddings_IDs):
            for j, name_j in enumerate(embeddings_IDs):
                
                # the amount of common annotations between the two sequences, i.e. A inter B
                capacity =\
                    len(set(annotation_dict[name_i]['go']).intersection(set(annotation_dict[name_j]['go']))) +\
                    len(set(annotation_dict[name_i]['keywords']).intersection(set(annotation_dict[name_j]['keywords'])))+\
                    len(set(annotation_dict[name_i]['taxonomy']).intersection(set(annotation_dict[name_j]['taxonomy'])))
                # the amount of annotations of the first sequence
                A =\
                    len(set(annotation_dict[name_i]['go'])) +\
                    len(set(annotation_dict[name_i]['keywords'])) +\
                    len(set(annotation_dict[name_i]['taxonomy']))
                # the amount of annotations of the second sequence
                B =\
                    len(set(annotation_dict[name_j]['go'])) +\
                    len(set(annotation_dict[name_j]['keywords'])) +\
                    len(set(annotation_dict[name_j]['taxonomy']))
                
                if edge_weight == 'method_1':
                    # compute the number of common annotations: n = (2*|A inter B|) / (|A| + |B|) 
                    gtrue_distance_matrix[i][j] += 2*capacity/(A+B)
                
                elif edge_weight == 'method_2':
                    # compute the weight as max( (A inter B)/A, (A inter B)/B )
                    gtrue_distance_matrix[i][j] += max(capacity/A, capacity/B)

        # the ground true distances is not a distance measure but a similarity, we have to make it a distance and also make the diagonal 0 (maybe not necessary)
        gtrue_distance_matrix = gtrue_distance_matrix.max() - gtrue_distance_matrix
        
        # the max should be 1, the min 0 and the diagonal 0
        assert np.allclose(gtrue_distance_matrix.diagonal(), np.zeros(len(embeddings_IDs)), atol=1e-8)
        assert gtrue_distance_matrix.max() <= 1
        assert gtrue_distance_matrix.min() >= 0

        # check is gtrue_distance_matrix is symmetric
        assert np.allclose(gtrue_distance_matrix, gtrue_distance_matrix.T, atol=1e-8)

        # make condensed distance matrices
        gtrue_distance_matrix = squareform(gtrue_distance_matrix)

        # compute the linkage matrices
        gtrue_linkage_matrix = linkage(gtrue_distance_matrix, method=method, metric=metric)

        gtrue_IDs = embeddings_IDs # TODO hopefully this is correct and nothing strande with the order of the IDs has happened

        return {"gtrue_linkage_matrix" : gtrue_linkage_matrix, 
                "gtrue_IDs": gtrue_IDs, 
                "embeddings_linkage_matrix" : embeddings_linkage_matrix,
                "embeddings_IDs": embeddings_IDs,
                }
    
    et.add_multistage(
        function=pipeline_build_gt_linkage_matrix,
        fixed_args={ "ground_true_path" : GROUND_TRUE_PATH},
        list_args=[
            { "metric" : "euclidean",   "method" : "ward",        "edge_weight" : "method_1" },
            { "metric" : "euclidean",   "method" : "average",     "edge_weight" : "method_1" },
            { "metric" : "euclidean",   "method" : "complete",    "edge_weight" : "method_1" },
            { "metric" : "euclidean",   "method" : "centroid",    "edge_weight" : "method_1" },
            { "metric" : "euclidean",   "method" : "single",      "edge_weight" : "method_1" },
            { "metric" : "euclidean",   "method" : "median",      "edge_weight" : "method_1" },
            
            { "metric" : "cosine",      "method" : "average",        "edge_weight" : "method_1" },
            { "metric" : "cosine",      "method" : "complete",       "edge_weight" : "method_1" },
            { "metric" : "cosine",      "method" : "single",         "edge_weight" : "method_1" },
        ]
    )


    def pipeline_mean_adjusted_rand_score(previous_stage_output : dict, cluster_range)-> dict:
        """
        Performs the enrichment analysis comparing the number of common annotations between sequences and the distance in the ebmeddings space
        
        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the embeddings matrix and the IDs
            metric (str): The metric to use (euclidean, cosine, etc.)
        Returns:
            dict: A dict containing the linkage matrix and the IDs
        """

        gtrue_linkage_matrix = previous_stage_output["gtrue_linkage_matrix"]
        gtrue_IDs = previous_stage_output["gtrue_IDs"]
        embeddings_linkage_matrix = previous_stage_output["embeddings_linkage_matrix"]
        embeddings_IDs = previous_stage_output["embeddings_IDs"]

        assert len(gtrue_IDs) == len(embeddings_IDs), "The number of IDs in the ground true file must be the same as the number of embeddings"

        start = cluster_range[0]
        end = cluster_range[1]
        if end == -1:
            end = len(gtrue_IDs)

        # compute the labels matrix: an array of shape (n_samples, n_clusters) where n_clusters is the number of clusters in the range
        predict_labels_matrix= cut_tree(embeddings_linkage_matrix, n_clusters=range(start, end))
        gtrue_labels_matrix = cut_tree(gtrue_linkage_matrix, n_clusters=range(start, end))

        adjusted_rand_scores = []
        for i in range(predict_labels_matrix.shape[1]):
            adjusted_rand_scores.append(adjusted_rand_score(predict_labels_matrix[:,i], gtrue_labels_matrix[:,i]))

        return {"mean_adjusted_rand_score" : np.mean(adjusted_rand_scores), "max_adjusted_rand_score" : np.max(adjusted_rand_scores)}


    et.add_stage(
        function=pipeline_mean_adjusted_rand_score,
        args={
            "cluster_range" : (2, -1)
        }
    )

    # END

    return et


if __name__ == "__main__":
    
    # EMBEDDINGS_PATH = "./dataset/enrichment_test/proteins.json"
    # GROUND_TRUE_PATH = "./dataset/enrichment_test/annotations.xml"

    EMBEDDINGS_PATH =  "./dataset/emoglobina/emoglobina.json"
    GROUND_TRUE_PATH = "./dataset/emoglobina/emoglobina.xml"
    
    
    et = main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH)
    et.compute()
    
    r = et.get_results()

    # get the name of the current file


    file_name = "./results/"+ "enrichment_"+"results_" + EMBEDDINGS_PATH.split("/")[-1].split(".")[0] 

    et.dump_results(r, file_name)

    results2file(r, file_name)

    
    