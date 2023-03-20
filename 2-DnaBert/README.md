# Minimal DNABert example

## 1. Create virtual environment

Run:

    conda create -name dnabert python=3.9 -y && conda activate dnabert


## 2. Install requirements

To install pytorch on linux, run:
    
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

To install pytorch on MacOS/Windows, run:

    pip install torch torchvision torchaudio

Finally, install the transformers library

    pip install transformers

You should be ready to go.


## 3. Execute the sample script

Run:

    python run.py

If completed successfully, you will see a new `embeddings.pt` file containing the results (from which you can inspect).
