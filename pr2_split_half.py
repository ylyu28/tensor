# %% [markdown]
# # Split-half implementation on parafac2, permutation on the pathway level
# 1. common pathways 
# 2. all pathways separately


# %%
### 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorly.decomposition import parafac2
import pickle
import tlviz
import tensorly as tl
import scipy.stats as stats
from sklearn.preprocessing import scale

def pr2_split_half(kpca_slices,num_omics=2, random_seed=0): 
    """Split the CONCATED kpca score matrices across the pre-defined axis and return a list of the two tensors 
    
    Parameters
    ----------
    X: DataFrame (not ndarray)
        concated matrix along mode=sample
    num_omics: int
        Number of layers to construct tensor
    random_seed: int
        Random seed for splitting sampling 

    Returns
    ----------   
    list
        List of the two tensors split across the predefined axis
    """
    X = np.hstack(kpca_slices)
    I,J = X.shape
    rng = np.random.default_rng(random_seed)
    permutation = rng.permutation(I)
    split_halfmat_1 = X[permutation[:I//2]].T
    split_halfmat_2 = X[permutation[I//2:]].T
    split_half_slice_1 = [split_halfmat_1[:kpca_slices[0].shape[1],:],split_halfmat_1[kpca_slices[0].shape[1]:,:]]
    split_half_slice_2 = [split_halfmat_2[:kpca_slices[0].shape[1],:],split_halfmat_2[kpca_slices[0].shape[1]:,:]]
    splits_sample = [split_half_slice_1,split_half_slice_2]

    return splits_sample

# %%
def normalise_pr2_tensor(pr2_tensor):
    weights, factors, path_factors = pr2_tensor

    if weights is None:
        weights = np.ones(factors[0].shape[1])

    weights = weights.copy()


    # sample score matrix = factors[2], metab score matrix = path_factors[0],
    # prot score matrix = path_factors[1], omics score matrix = factors[0]
    fac_scores = [factors[2], path_factors[0], path_factors[1], factors[0]]

    new_fac_scores = []
    for factor in fac_scores:
        norm = np.linalg.norm(factor, axis=0, keepdims=True)
        weights *=norm.ravel()

        norm[norm == 0] = 1
        new_fac_scores.append(factor/norm)
    return weights, tuple(new_fac_scores)
    

# %%
def parafac2_fms(pr2_tensors, skip_mode=0):
    weights, facs1 = normalise_pr2_tensor(pr2_tensors[0])
    weights, facs2 = normalise_pr2_tensor(pr2_tensors[1])

    congruence_product = 1
    for i, (facs1, facs2) in enumerate(zip(facs1, facs2)):
        if hasattr(facs1,"values"):
            facs1 = facs1.values
        if hasattr(facs2,"values"):
            facs2 = facs2.values
        if i == skip_mode:
            continue
        congruence_product *= facs1.T @ facs2

    return congruence_product.mean()


# %%
def pr2_cal_fms(splits_samples,num_components=10):
    """ For rank in range(num_components=10), run parafac on each of the 100 permuted splits,
    then calculate fms scores. Return a dictionary where each key is
    the rank and value is an FMSResult object containing scores and CI.
    """ 
    fms_dist ={}

    for rank in range(1,num_components+1):
        print(f"{rank} components")
        fms_scores = []
        for splits_tensors in splits_samples:
            pr_1 = parafac2(splits_tensors[0], rank=rank, init='svd')
            pr_2 = parafac2(splits_tensors[1], rank=rank, init='svd')
            fms = parafac2_fms([pr_1, pr_2], skip_mode=0) 
            fms_scores.append(fms)
        fms_dist[rank] = (fms_scores)
    
    return fms_dist

# %%
def plt_fms(fms_dist, num_components, plt_name='plt_name'):
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
   plt.savefig(f'./Figures/{plt_name}_fms_plot.png')
   plt.show()




# %%
# Common pathways 

with open('metab_filtered_kpca.pkl', 'rb') as f:
    metab_filtered_kpca = pickle.load(f)

with open('prot_filtered_kpca.pkl', 'rb') as f:
    prot_filtered_kpca = pickle.load(f)


# 0. No preprocessing
slices = [metab_filtered_kpca,prot_filtered_kpca]

nopre_splits_tensors = []

for seed in range(100):
    nopre_splits_tensors.append(pr2_split_half(slices, num_omics=2, random_seed=seed))

nopre_fms = pr2_cal_fms(nopre_splits_tensors,10)

nopre_plt_fms = plt_fms(nopre_fms,10,'common_nopre')



# %%
concated_mat = np.hstack([metab_filtered_kpca,prot_filtered_kpca])
concated_mat_cen = concated_mat - concated_mat.mean()
concated_mat_cen.shape

# %%
# 1. Centered, no scaling
concated_mat = np.hstack([metab_filtered_kpca,prot_filtered_kpca])
concated_mat_cen = concated_mat - concated_mat.mean()
slices_cen = [concated_mat_cen[:,:60],concated_mat_cen[:,60:]]
              
cen_splits_tensors = []

for seed in range(100):
    cen_splits_tensors.append(pr2_split_half(slices_cen, num_omics=2, random_seed=seed))

cen_fms = pr2_cal_fms(cen_splits_tensors,10)

cen_plt_fms = plt_fms(cen_fms,10,'common_cen')

# %%
# 2. Centered, bro scaling

concated_mat_cen_bro = scale(concated_mat_cen,axis=1,with_mean=False)

slices_cen_bro = [concated_mat_cen_bro[:,:60],concated_mat_cen_bro[:,60:]]

cen_bro_splits_tensors = []

for seed in range(100):
    cen_bro_splits_tensors.append(pr2_split_half(slices_cen_bro, num_omics=2, random_seed=seed))

cen_bro_fms = pr2_cal_fms(cen_bro_splits_tensors,10)

cen_bro_plt_fms = plt_fms(cen_bro_fms,10,'common_cen_bro')

# %%
# 3. bro scaling, no centering
concated_mat_bro = scale(concated_mat,axis=1, with_mean=False)

slices_bro = [concated_mat_bro[:,:60],concated_mat_bro[:,60:]]

bro_splits_tensors = []

for seed in range(100):
    bro_splits_tensors.append(pr2_split_half(slices_cen_bro, num_omics=2, random_seed=seed))

bro_fms = pr2_cal_fms(bro_splits_tensors,10)

bro_plt_fms = plt_fms(bro_fms,10,'common_bro')

# %%
# 4. Bro scaling, then centering
concated_mat_bro_cen = concated_mat_bro - concated_mat_bro.mean()

slices_bro_cen = [concated_mat_bro_cen[:,:60],concated_mat_bro_cen[:,60:]]

bro_cen_splits_tensors = []

for seed in range(2):
    bro_cen_splits_tensors.append(pr2_split_half(slices_bro_cen, num_omics=2, random_seed=seed))

bro_cen_fms = pr2_cal_fms(bro_cen_splits_tensors,10)

bro_cen_plt_fms = plt_fms(bro_cen_fms,10,'common_bro_cen')






# %%
# All pathways 

with open('metab_kpca.pkl', 'rb') as f:
    metab_kpca = pickle.load(f)

with open('prot_kpca.pkl', 'rb') as f:
    prot_kpca = pickle.load(f)


# 0. No preprocessing
slices = [metab_kpca,prot_kpca]

nopre_splits_tensors = []

for seed in range(100):
    nopre_splits_tensors.append(pr2_split_half(slices, num_omics=2, random_seed=seed))

nopre_fms = pr2_cal_fms(nopre_splits_tensors,10)

nopre_plt_fms = plt_fms(nopre_fms,10,'all_nopre')



# %%
concated_mat = np.hstack([metab_kpca,prot_kpca])
concated_mat_cen = concated_mat - concated_mat.mean()
concated_mat_cen.shape

# %%
# 1. Centered, no scaling
concated_mat = np.hstack([metab_kpca,prot_kpca])
concated_mat_cen = concated_mat - concated_mat.mean()
slices_cen = [concated_mat_cen[:,:60],concated_mat_cen[:,60:]]
              
cen_splits_tensors = []

for seed in range(100):
    cen_splits_tensors.append(pr2_split_half(slices_cen, num_omics=2, random_seed=seed))

cen_fms = pr2_cal_fms(cen_splits_tensors,10)

cen_plt_fms = plt_fms(cen_fms,10,'all_cen')

# %%
# 2. Centered, bro scaling

concated_mat_cen_bro = scale(concated_mat_cen,axis=1,with_mean=False)

slices_cen_bro = [concated_mat_cen_bro[:,:60],concated_mat_cen_bro[:,60:]]

cen_bro_splits_tensors = []

for seed in range(100):
    cen_bro_splits_tensors.append(pr2_split_half(slices_cen_bro, num_omics=2, random_seed=seed))

cen_bro_fms = pr2_cal_fms(cen_bro_splits_tensors,10)

cen_bro_plt_fms = plt_fms(cen_bro_fms,10,'all_cen_bro')

# %%
# 3. bro scaling, no centering
concated_mat_bro = scale(concated_mat,axis=1, with_mean=False)

slices_bro = [concated_mat_bro[:,:60],concated_mat_bro[:,60:]]

bro_splits_tensors = []

for seed in range(100):
    bro_splits_tensors.append(pr2_split_half(slices_cen_bro, num_omics=2, random_seed=seed))

bro_fms = pr2_cal_fms(bro_splits_tensors,10)

bro_plt_fms = plt_fms(bro_fms,10,'all_bro')

# %%
# 4. Bro scaling, then centering
concated_mat_bro_cen = concated_mat_bro - concated_mat_bro.mean()

slices_bro_cen = [concated_mat_bro_cen[:,:60],concated_mat_bro_cen[:,60:]]

bro_cen_splits_tensors = []

for seed in range(100):
    bro_cen_splits_tensors.append(pr2_split_half(slices_bro_cen, num_omics=2, random_seed=seed))

bro_cen_fms = pr2_cal_fms(bro_cen_splits_tensors,10)

bro_cen_plt_fms = plt_fms(bro_cen_fms,10,'all_bro_cen')