from __future__ import print_function,division
import sys
import numpy as np
import torch
from prose.alphabets import Uniprot21
import json
from numpy import array2string
import os


FILE_PATH = "../dataset/test.json"    
ANNOTATION_KEY = "embedding3"
MODEL_PATH = "prose_mt"
DEVICE = -2
POOL = "avg"



def embed_sequence(model, x, pool='none', use_cuda=False):
    if len(x) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.cuda()

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        
        # !!! change this in model(x) in order to get only the last layer
        
        #z = model.transform(x) # all the network stack

        z = model(x) # only the z layer
       
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    return z


def main():
      

    # load the model
    if MODEL_PATH == 'prose_mt':
        from prose.models.multitask import ProSEMT
        print('# loading the pre-trained ProSE MT model', file=sys.stderr)
        model = ProSEMT.load_pretrained()
    elif MODEL_PATH == 'prose_dlm':
        from prose.models.lstm import SkipLSTM
        print('# loading the pre-trained ProSE DLM model', file=sys.stderr)
        model = SkipLSTM.load_pretrained()
    else:
        print('# loading model:', MODEL_PATH, file=sys.stderr)
        model = torch.load(MODEL_PATH)
    model.eval()

    # set the device
    d = DEVICE
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    if use_cuda:
        model = model.cuda()

    # parse the sequences and embed them
    # write them to hdf5 file
    print('# writing:', FILE_PATH, file=sys.stderr)
    
    seq_dict = []
    with open(FILE_PATH, "r") as file:   # load the list of seqrecords alreay annotated with the others embeddings
        seq_dict = json.load(file)

    pool = POOL
    print('# embedding with pool={}'.format(pool), file=sys.stderr)
    
    for id in seq_dict.keys():
        seq_string = bytes(seq_dict[id]["sequence"], "utf-8")
        embed = embed_sequence(model, seq_string , pool=pool, use_cuda=use_cuda)
        
        seq_dict[id][ANNOTATION_KEY] = embed.tolist()
        
        
    with open(FILE_PATH, "w") as file:
        json.dump(seq_dict, file, indent=4)


if __name__ == '__main__':
    main()
