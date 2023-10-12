import pandas as pd

def results2table(
    r, 
    metric="mean_adjusted_rand_score", 
    preferred_metric_embedding="euclidean",
    preferred_method_embedding="ward",
    preferred_metric_gt="euclidean",
    preferred_method_gt="ward", 
    preferred_edge_weight="method_1",
    ):
    
    computations_dict = {} # dict[combiner][pca][embedder] = score

    combiners = ["pca", "average", "sum", "max"]
    pcas = ["all", "default"]
    embedders = ["dnabert", "seqvec", "prose", "alphafold", "esm"	]

    # generate the empty dict with the argument that are relevant for the table
    for combiner in combiners:
        computations_dict[combiner] = {}
        for pca in pcas:
            computations_dict[combiner][pca] = {}
            for embedder in embedders:
                computations_dict[combiner][pca][embedder] = None
    
    # fill the dict
    for result, pipeline in r:  # for each result and the pipeline that generated it
        combiner = None
        pca = None
        embedder = None  
        for stage, args in pipeline: # for each stage and the arguments that were passed to it
            # keep the arguments that are relevant for the table
            if stage == "pipeline_build_embeddings_matrix":
                embedder = args["embedder"]
                combiner = args["combiner_method"]
            if stage == "pipeline_pca":
                pca = args["n_components"]
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
                    edge_weight = preferred_edge_weight
                    
        
        if  metric_embedding == preferred_metric_embedding and \
            method_embedding == preferred_method_embedding and \
            metric_gt == preferred_metric_gt and \
            method_gt == preferred_method_gt and \
            edge_weight == preferred_edge_weight:
        
            computations_dict[combiner][pca][embedder] = result[metric]

    
    data_matrix = [] # the matrix that will be converted to a dataframe, the order of the rows and columns must be consistent whith the one created by the MultiIndex and the "columns" parameter of the DataFrame constructor

    iterables = [combiners, pcas]

    for combiner in combiners:
        for pca in pcas:
            row = []
            for embedder in embedders:
                row.append(computations_dict[combiner][pca][embedder])
            data_matrix.append(row)


    index = pd.MultiIndex.from_product(iterables, names=["combiner", "dimensional PCA"])
    df = pd.DataFrame( data_matrix, index=index, columns=embedders)

    return df

def results2file(r, filepath):
    
    r.sort(key = lambda x: x[0]['mean_adjusted_rand_score'], reverse=True)
    
    with open(filepath + ".txt", "w") as f:
        for result, pipeline in r:
            f.write(f"mean_adjusted_rand_score: {result['mean_adjusted_rand_score']} \n")
            for name, args in pipeline:
                f.write(f"{name}  {args} \n")
            f.write(f"\n")
 

