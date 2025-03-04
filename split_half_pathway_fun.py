import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorly.decomposition import parafac
import tlviz
import tensorly as tl
import scipy.stats as stats
from sklearn.preprocessing import scale


def split_half(X,mode=0,num_omics=2, random_seed=0): 
    """Split the flattened tensor across the pre-defined axis and return a list of the two tensors 
    
    Parameters
    ----------
    X: DataFrame (not ndarray)
        Flattened cp_tensor along mode=sample
    mode: int
        Mode=0: sample;
        Mode=1: pahtway
    num_omics: int
        Number of layers to construct tensor
    random_seed: int
        Random seed for splitting sampling 

    Returns
    ----------   
    list
        List of the two tensors split across the predefined axis
    """

    I,J = X.shape
    rng = np.random.default_rng(random_seed)
    if mode==0:
        permutation = rng.permutation(I)
        split_tensor_1 = X.iloc[permutation[:I//2]]
        split_tensor_2 = X.iloc[permutation[I//2:]]
        split_tensor_1 = tl.fold(split_tensor_1,mode=0,shape=(I//2,J//num_omics,num_omics))
        split_tensor_2 = tl.fold(split_tensor_2,mode=0,shape=(I-I//2,J//num_omics,num_omics))
        splits_sample = [split_tensor_1,split_tensor_2]
    
    elif mode==1:
        permutation = rng.permutation(J)
        split_tensor_1 = X.iloc[:,permutation[:J//2]]
        split_tensor_2 = X.iloc[:,permutation[J//2:]]
        split_tensor_1 = tl.fold(split_tensor_1,mode=1,shape=(I,(J//2)//num_omics,num_omics))
        split_tensor_2 = tl.fold(split_tensor_2,mode=1,shape=(I,(J-J//2)//num_omics,num_omics))
        splits_sample = [split_tensor_1,split_tensor_2]
    
    return splits_sample

def cal_fms(splits_samples,num_components=10):
    """ For rank in range(num_components=10), run parafac on each of the 100 permuted splits,
    then calculate fms scores. Return a dictionary where each key is
    the rank and value is an FMSResult object containing scores and CI.
    """ 
    fms_dist ={}

    for rank in range(1,num_components+1):
        print(f"{rank} components")
        fms_scores = []
        for splits_tensors in splits_samples:
            cp_1 = parafac(splits_tensors[0], rank=rank, init='svd')
            cp_2 = parafac(splits_tensors[1], rank=rank, init='svd')
            fms = tlviz.factor_tools.factor_match_score(cp_1, cp_2, consider_weights=False, skip_mode=0) 
            fms_scores.append(fms)
        fms_dist[rank] = (fms_scores)
    
    return fms_dist

def plt_fms(fms_dist, num_components):
   """Plot a graph of number of components against mean fms scores of the 100 permuted split-half, showing 95% CI
   """
   fig, ax = plt.subplots()
   means = []
   cis = []

   for rank in range(1, num_components+1):
      if rank in fms_dist:
         fms_scores = fms_dist[rank]
         mean_score = np.mean(fms_scores)
         sem = stats.sem(fms_scores)
         ci = stats.t.interval(0.95,len(fms_scores)-1, loc=mean_score, scale=sem)
      means.append(mean_score)
      cis.append((mean_score - ci[0],ci[1]-mean_score))

   means = np.array(means)
   cis = np.array(cis)

   # Plot mean FMS scores
   ax.errorbar(range(1, num_components + 1), means, 
               yerr=cis.T, fmt='-o', capsize=5, label='Mean FMS Score ± 95% CI')
    
   ax.set_xticks(range(1, num_components + 1))
   ax.set_xticklabels(range(1, num_components + 1))
   ax.set_xlabel("Number of Components")
   ax.set_ylabel("Split Half Stability (FMS Score)")
   ax.set_title("FMS Scores with 95% Confidence Intervals")
   ax.legend()
    
   plt.tight_layout()
   plt.show()