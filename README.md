# kernelNet MovieLens-1M

State of the art model for MovieLens-1M.

This is a minimal implementation of a kernelNet sparsified autoencoder for MovieLens-1M. 
See http://proceedings.mlr.press/v80/muller18a.html

## Setup
Download this repository

### Requirements
* numpy
* scipy
* tensorflow (tested with version 1.13)

### Dataset
Expects MovieLens-1M dataset in a subdirectory named ml-1m.
Get it here https://grouplens.org/datasets/movielens/1m/

or on linux run in the project directory

```wget --output-document=ml-1m.zip http://www.grouplens.org/system/files/ml-1m.zip; unzip ml-1m.zip```

## Run
```python kernelNet_ml1m.py```
optional arguments are the L2 and sparsity regularization strength. Default is 60. and 0.013

### Results
with the default parameters this slightly outperforms the paper model at 0.823 validation RMSE (10-times repeated random sub-sampling validation)
