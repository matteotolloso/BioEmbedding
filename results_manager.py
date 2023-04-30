import pandas as pd
import numpy as np
import pickle

def results2table(r):
    
    computations_dict = {} # dict[combiner][pca][embedder] = score

    combiners = ["pca", "average", "sum", "max"]
    pcas = ["all", "default"]
    embedders = ["rep", "dnabert", "prose", "alphafold"]

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
        
        computations_dict[combiner][pca][embedder] = result["mean_adjusted_rand_score"]

    
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
            f.write(f"Score: {result['mean_adjusted_rand_score']} \n")
            for name, args in pipeline:
                f.write(f"{name}  {args} \n")
            f.write(f"\n")
 

