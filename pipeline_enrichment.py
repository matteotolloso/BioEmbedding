from autoembedding.ExecutionTree import ExecutionTree
import autoembedding.utils as utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from autoembedding.embeddings_matrix import build_embeddings_matrix
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import adjusted_rand_score
import numpy as np
from Bio import SeqIO
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

            {"embedder" : "seqvec", "combiner_method" : "pca" },
            {"embedder" : "seqvec", "combiner_method" : "average" },
            {"embedder" : "seqvec", "combiner_method" : "sum" },
            {"embedder" : "seqvec", "combiner_method" : "max" },
            
            # {"embedder" : "dnabert", "combiner_method" : "pca" },
            # {"embedder" : "dnabert", "combiner_method" : "average" },
            # {"embedder" : "dnabert", "combiner_method" : "sum" },
            # {"embedder" : "dnabert", "combiner_method" : "max" },
            
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

    def pipeline_scaling_and_pca(previous_stage_output : dict, n_components) -> dict:

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

        scaler = StandardScaler()
        embeddings_matrix = scaler.fit_transform(embeddings_matrix)

        if n_components != "all":
            if n_components == "default":
                n_components = min(embeddings_matrix.shape)
            pca = PCA(n_components=n_components)
            embeddings_matrix = pca.fit_transform(embeddings_matrix)
        
        return { "embeddings_matrix" : embeddings_matrix, "embeddings_IDs": embeddings_IDs}

    et.add_multistage(
        function=pipeline_scaling_and_pca,
        list_args=[
            {"n_components": 10},
            {"n_components": 50},
            {"n_components": 'all'},
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

    
    def pipeline_build_gt_linkage_matrix(previous_stage_output : dict, metric, method, edge_weight, use_go, use_keywords, use_taxonomy, ground_true_path)-> dict:

        embeddings_linkage_matrix = previous_stage_output["embeddings_linkage_matrix"]
        embeddings_IDs = previous_stage_output["embeddings_IDs"]
        
        # preparing the matrix distance in the "enrichment space"
        records = list(SeqIO.parse(ground_true_path, "uniprot-xml"))

        assert len(records) == len(embeddings_IDs), "The number of records in the ground true file must be the same as the number of embeddings"

        annotation_dict = {}
        
        for record in records:

            name = f'{record.id}|{record.name}'      
            
            annotation_dict[name] = {}
            go_annotations = [i for i in record.dbxrefs if i.startswith('GO')]
            annotation_dict[name]['go'] = go_annotations
            annotation_dict[name]['keywords'] = record.annotations['keywords']
            annotation_dict[name]['taxonomy'] = record.annotations['taxonomy']

        gtrue_distance_matrix = np.zeros((len(embeddings_IDs), len(embeddings_IDs)))

        for i, name_i in enumerate(embeddings_IDs):
            
            name_i = name_i[name_i.find("|")+1:]  # remove the first part of the string (until the first '|') this because the annotation dictionary is indexed without them
            
            for j, name_j in enumerate(embeddings_IDs):

                name_j = name_j[name_j.find("|")+1:]

                # annotations of the first sequence
                A = set()
                # annotations of the second sequence
                B = set()

                if use_go:
                    A = A.union(set(annotation_dict[name_i]['go']))
                    B = B.union(set(annotation_dict[name_j]['go']))
                if use_keywords:
                    A = A.union(set(annotation_dict[name_i]['keywords']))
                    B = B.union(set(annotation_dict[name_j]['keywords']))
                if use_taxonomy:
                    A = A.union(set(annotation_dict[name_i]['taxonomy']))
                    B = B.union(set(annotation_dict[name_j]['taxonomy']))

                if len(A) == 0 or len(B) == 0:
                    gtrue_distance_matrix[i][j] = 1/2
                    continue

                if edge_weight == 'jaccard':
                    gtrue_distance_matrix[i][j] += len(A.intersection(B)) / len(A.union(B))
                
                elif edge_weight == 'overlap':
                    gtrue_distance_matrix[i][j] += len(A.intersection(B)) / min(len(A), len(B))

        assert gtrue_distance_matrix.max() == 1

        # the ground true distances is not a distance measure but a similarity, we have to make it a distance 
        gtrue_distance_matrix = 1 - gtrue_distance_matrix
        
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

        gtrue_IDs = embeddings_IDs

        return {"gtrue_linkage_matrix" : gtrue_linkage_matrix, 
                "gtrue_IDs": gtrue_IDs, 
                "embeddings_linkage_matrix" : embeddings_linkage_matrix,
                "embeddings_IDs": embeddings_IDs,
                }
    
    et.add_multistage(
        function=pipeline_build_gt_linkage_matrix,
        fixed_args={ "ground_true_path" : GROUND_TRUE_PATH},
        list_args=[
            # only go
            { "metric" : "euclidean",   "method" : "ward",        "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "average",     "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "complete",    "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "centroid",    "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "single",      "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "median",      "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "cosine",      "method" : "average",     "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "cosine",      "method" : "complete",    "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "cosine",      "method" : "single",      "edge_weight" : "jaccard" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "ward",        "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "average",     "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "complete",    "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "centroid",    "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "single",      "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "euclidean",   "method" : "median",      "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "cosine",      "method" : "average",     "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "cosine",      "method" : "complete",    "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            { "metric" : "cosine",      "method" : "single",      "edge_weight" : "overlap" , "use_go": True, "use_keywords" : False, "use_taxonomy" : False },
            # only keywords
            { "metric" : "euclidean",   "method" : "ward",        "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "average",     "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "complete",    "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "centroid",    "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "single",      "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "median",      "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "cosine",      "method" : "average",     "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "cosine",      "method" : "complete",    "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "cosine",      "method" : "single",      "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "ward",        "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "average",     "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "complete",    "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "centroid",    "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "single",      "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "euclidean",   "method" : "median",      "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "cosine",      "method" : "average",     "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "cosine",      "method" : "complete",    "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            { "metric" : "cosine",      "method" : "single",      "edge_weight" : "overlap" , "use_go": False, "use_keywords" : True, "use_taxonomy" : False},
            # only taxonomy
            { "metric" : "euclidean",   "method" : "ward",        "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "average",     "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "complete",    "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "centroid",    "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "single",      "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "median",      "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "cosine",      "method" : "average",     "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "cosine",      "method" : "complete",    "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "cosine",      "method" : "single",      "edge_weight" : "jaccard" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "ward",        "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "average",     "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "complete",    "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "centroid",    "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "single",      "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "euclidean",   "method" : "median",      "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "cosine",      "method" : "average",     "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "cosine",      "method" : "complete",    "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
            { "metric" : "cosine",      "method" : "single",      "edge_weight" : "overlap" , "use_go": False, "use_keywords" : False, "use_taxonomy" : True},
        ]
    )


    def pipeline_mean_adjusted_rand_score(previous_stage_output : dict, cluster_range ) -> dict:
        """
        Computes the mean adjusted rand score between two hierarchical clustering averaging over all the cut
        in a given range

        Args:
            previous_stage_output (dict): The output of the previous stage, a dict containing the linkage matrix and the IDs
            cluster_range: The range of clusters to consider (the cut to the hierarchical clustering)
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
        if not set(embeddings_IDs) == set(gtrue_IDs):
            raise Exception("The IDs in the ground true and the predicted clustering are different")

        start = 0
        end = 0

        if cluster_range == "auto":
            start = 2
            end = len(embeddings_IDs) - 1

        predict_labels_matrix = cut_tree(embeddings_linkage_matrix, n_clusters=[i for i in range(start, end+1)])
        gtrue_labels_matrix = cut_tree(gtrue_linkage_matrix, n_clusters=[i for i in range(start, end+1)])

        # order the matrix rows based on the IDs
        predict_labels_matrix = predict_labels_matrix[np.argsort(embeddings_IDs)]
        gtrue_labels_matrix = gtrue_labels_matrix[np.argsort(gtrue_IDs)]

        # for each iteration, extract the relative column and compute the adjusted rand score
        adjusted_rand_scores = []
        for i in range(predict_labels_matrix.shape[1]):
            adjusted_rand_scores.append(adjusted_rand_score(predict_labels_matrix[:,i], gtrue_labels_matrix[:,i]))
        
        return {"mean_adjusted_rand_score" : np.mean(adjusted_rand_scores), "adjusted_rand_scores": adjusted_rand_scores}


    et.add_stage(
        function=pipeline_mean_adjusted_rand_score,
        args={
            "cluster_range" : "auto"
        }
    )

    # END

    return et


if __name__ == "__main__":
    

    EMBEDDINGS_PATH =  "./dataset/batterio/embeddings"
    GROUND_TRUE_PATH = "./dataset/batterio/batterio.xml"

    # EMBEDDINGS_PATH =  "./dataset/emoglobina/embeddings"
    # GROUND_TRUE_PATH = "./dataset/emoglobina/emoglobina.xml"

    # EMBEDDINGS_PATH =  "./dataset/topo/embeddings"
    # GROUND_TRUE_PATH = "./dataset/topo/topo.xml"
    
    
    et = main_et(EMBEDDINGS_PATH, GROUND_TRUE_PATH)
    et.compute()
    
    r = et.get_results()

    # get the name of the current file

    file_name = "./results/"+ "enrichment_"+"results_" + GROUND_TRUE_PATH.split("/")[-1].split(".")[0] 

    et.dump_results(r, file_name)

    results2file(r, file_name)

    
    