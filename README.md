# paper: learned protein embeddings for machine nearning

venv: bioemb1

python 3.5.6  (deprecated)

* gensim==1.0.1      (pip)
* numpy==1.13.1      (pip)
* pandas==0.20.3     (pip)
* scipy==0.19.1      (conda)
* sklearn==0.19.0     (conda)
* matplotlib==2.0.2   (conda)
* seaborn==0.8.1      (conda)

`conda list --export > requirements.txt`

`conda create --name <env> --file <this file>`

File created with `conda list -e > requirements.txt`
This file may be used to create an environment using:
`$ conda create --name <env> --file <this file>`
platform: win-64
!!! the module embeddings_reprodiction from the paper is not avaiable on pip or conda and must be downloaded 
from github and then installed, follow the instruction here: https://github.com/fhalab/embeddings_reproduction

# paper: biobert