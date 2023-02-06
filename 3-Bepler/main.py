from __future__ import print_function,division
import sys
import numpy as np
import h5py
import torch
from prose.prose.alphabets import Uniprot21
import prose.prose.fasta as fasta
from Bio import SeqRecord
import pickle
from numpy import array2string

IN_PATH = "dataset/output_test.pkl"    
ANNOTATION_KEY = "embedding2"
OUT_PATH = "../dataset/output_test2.pkl"

MODEL_PATH = "prose_mt"
DEVICE = -2
POOL = "none"



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
        z = model.transform(x)
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
        from prose.prose.models.multitask import ProSEMT
        print('# loading the pre-trained ProSE MT model', file=sys.stderr)
        model = ProSEMT.load_pretrained()
    elif MODEL_PATH == 'prose_dlm':
        from prose.prose.models.lstm import SkipLSTM
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
    print('# writing:', OUT_PATH, file=sys.stderr)
    
    seqrec_list = []
    with open(IN_PATH, "rb") as file:   # load the list of seqrecords alreay annotated with the others embeddings
        seqrec_list = pickle.load(seqrec_list, file)

    pool = POOL
    print('# embedding with pool={}'.format(pool), file=sys.stderr)
    count = 0
    
    for seqrec in seqrec_list:
        string_seq = str(seqrec.seq)
        embed = embed_sequence(model, string_seq, pool=pool, use_cuda=use_cuda)
        
        seqrec.annotations[ANNOTATION_KEY] = array2string(embed)
        
        count += 1
        print('# {} sequences processed...'.format(count), file=sys.stderr, end='\r')
    print(' '*80, file=sys.stderr, end='\r')

    with open(OUT_PATH, "wb") as file:
        pickle.dump(seqrec_list, file)


if __name__ == '__main__':
    main()
