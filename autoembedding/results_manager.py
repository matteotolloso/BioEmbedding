import pandas as pd
import numpy as np


def calculate_score(result, percentile=90):

    ars = result['adjusted_rand_scores']

    return np.percentile(ars, percentile)





def results2table(
    r, 
    embedders = ["dnabert", "seqvec", "prose", "alphafold", "esm"],
    combiners = ["pca", "average", "sum", "max"],
    pcas = [ '10', '20', '30', '40', '50', 'all'],
    metric="mean_adjusted_rand_score", 
    preferred_metric_embedding="euclidean",
    preferred_method_embedding="ward",
    preferred_metric_gt="euclidean",
    preferred_method_gt="ward", 
    preferred_edge_weight="jaccard",
    preferred_annotation = "go"
    ):
    
    computations_dict_scores = {} # dict[combiner][pca][embedder] = score
    computational_dict_ars = {} # dict[combiner][pca][embedder] = ars
    best_score = -1
    best_score_list = []

    # generate the empty dict with the argument that are relevant for the table
    for combiner in combiners:
        computations_dict_scores[combiner] = {}
        computational_dict_ars[combiner] = {}
        for pca in pcas:
            computations_dict_scores[combiner][pca] = {}
            computational_dict_ars[combiner][pca] = {}
            for embedder in embedders:
                computations_dict_scores[combiner][pca][embedder] = None
                computational_dict_ars[combiner][pca][embedder] = None
    
    # fill the dict
    for result, pipeline in r:  # for each result and the pipeline that generated it
        combiner = None
        pca = None
        embedder = None  
        edge_weight = preferred_edge_weight
        annotation = preferred_annotation
        for stage, args in pipeline: # for each stage and the arguments that were passed to it
            # keep the arguments that are relevant for the table
            if stage == "pipeline_build_embeddings_matrix":
                embedder = args["embedder"]
                combiner = args["combiner_method"]
            if stage == "pipeline_scaling_and_pca":
                pca = str(args["n_components"])
            if stage == "pipeline_build_embeddings_linkage_matrix":
                metric_embedding = args["metric"]
                method_embedding = args["method"]
            if stage == "pipeline_build_gt_linkage_matrix":
                metric_gt = args["metric"]
                method_gt = args["method"]
                # the edge_witght attribute is not present in all the pipelines
                try:
                    edge_weight = args["edge_weight"]
                except:
                    pass

                try:
                    if args['use_go']:
                        annotation = "go"
                    if args['use_keywords']:
                        annotation = "keywords"
                    if args['use_taxonomy']:
                        annotation = "taxonomy"
                except:
                    pass
                    
        
        if  metric_embedding == preferred_metric_embedding and \
            method_embedding == preferred_method_embedding and \
            metric_gt == preferred_metric_gt and \
            method_gt == preferred_method_gt and \
            edge_weight == preferred_edge_weight and \
            annotation == preferred_annotation:
        
            score = calculate_score(result)
            
            computations_dict_scores[combiner][pca][embedder] = score
            computational_dict_ars[combiner][pca][embedder] = result['adjusted_rand_scores']
            
            if score > best_score:
                best_score = score
                best_score_list = result['adjusted_rand_scores']
    
    data_matrix = [] # the matrix that will be converted to a dataframe, the order of the rows and columns must be consistent whith the one created by the MultiIndex and the "columns" parameter of the DataFrame constructor

    iterables = [combiners, pcas]

    for combiner in combiners:
        for pca in pcas:
            row = []
            for embedder in embedders:
                row.append(computations_dict_scores[combiner][pca][embedder])
            data_matrix.append(row)


    index = pd.MultiIndex.from_product(iterables, names=["combiner", "dimensional PCA"])
    df = pd.DataFrame( data_matrix, index=index, columns=embedders)

    return df, best_score_list, computational_dict_ars

def results2file(r, filepath):
    
    r.sort(key = lambda x: x[0]['mean_adjusted_rand_score'], reverse=True)
    
    with open(filepath + ".txt", "w") as f:
        for result, pipeline in r:
            f.write(f"mean_adjusted_rand_score: {result['mean_adjusted_rand_score']} \n")
            f.write(f"adjusted_rand_scores: {result['adjusted_rand_scores']} \n")
            for name, args in pipeline:
                f.write(f"{name}  {args} \n")
            f.write(f"\n")
 

