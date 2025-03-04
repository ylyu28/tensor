# Boostrapping for pathway selection

import pandas as pd
import tensorly as tl
import scipy.stats as stats
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac, parafac2

# 0.1 Parafac 1 bootstrapping
def bootstrap_feature_select_pr1(kpca_slices, sampling_time=10000, rank=2):

    # Converting numpy array to pandas DataFrame for bootstrap sampling then back to numpy array
    df_tensor_unfold_1 = pd.concat(kpca_slices,axis=1)

    # Create a dictionary for each path factor based on the rank
    path_facs_dict = {n: {} for n in range(rank)}

    for i in range(sampling_time):
        # Performing bootstrap sampling
        df_tensor_unfold_samp = df_tensor_unfold_1.sample(frac=1, replace=True)

        # Preprocessing sampled matrix
        resam_cen = df_tensor_unfold_samp - df_tensor_unfold_samp.mean()
        resam_cen_bro = scale(resam_cen,axis=1, with_mean=False)       

        # Refolding back to tensor X_samp
        tensor_samp = tl.base.fold(resam_cen_bro, mode=0, shape=(kpca_slices[0].shape[0],kpca_slices[0].shape[1],len(kpca_slices)))

        # Perform parafac decomposition on X_samp
        (_, cp_factors) = parafac(tensor_samp, rank=rank)
        _, path_facs, _ = cp_factors

        # Store the nth row of path_facs in the corresponding dictionary
        for n in range(rank):
            path_facs_dict[n][i] = path_facs[:, n]

    # Convert the dictionaries to DataFrames
    path_facs_dfs = {}
    for n in range(rank):
        path_facs_dfs[n] = pd.DataFrame.from_dict(path_facs_dict[n], orient='index', columns=kpca_slices[0].columns)

    return path_facs_dfs

# 0.2 Parafac 2 bootstrapping

def bootstrap_feature_select_pr2(kpca_slices, sampling_time=10000, rank=2):
    """
    kpca_slices: dataframes
    
    returns: dictionary{[0] # first omics layer : {rank1: df of pathways vs 1000 sampling results; # shape: len(pathway_list) * sampling_time}
                                                  {rank2: df of pathways vs 1000 sampling results}, 
                                                  ...
    }
    
    """
    concated_df_to_resam = pd.concat(kpca_slices,axis= 1)

    # Create a dictionary for each path factor based on the rank
    path_facs_dict = {l: {n: {} for n in range(rank)} for l in range(len(kpca_slices))}

    for i in range(sampling_time):
        # Performing bootstrap sampling
        concated_df_samp = concated_df_to_resam.sample(frac=1, replace=True)

        # Preprocessing resampled matrices
        resam_cen = concated_df_samp - concated_df_samp.mean()
        resam_cen_bro = scale(resam_cen,axis=1, with_mean=False)

        # splitting arrays into slices
        kpca_pr2_slice1 = resam_cen_bro[:,:kpca_slices[0].shape[1]].T
        kpca_pr2_slice2 = resam_cen_bro[:,kpca_slices[0].shape[1]:].T
        pr2_resampled_slices = [kpca_pr2_slice1,kpca_pr2_slice2]

        # Perform parafac2 decomposition on resampled slices
        (_, _, pr2_path_facs) = parafac2(pr2_resampled_slices, rank=rank, init='svd')
        
        # Store the nth row of path_facs in the corresponding dictionary
        for l in range(len(kpca_slices)):
            for n in range(rank):
                path_facs_dict[l][n][i] = pr2_path_facs[l][:, n]

    # Convert the dictionaries to DataFrames
    path_facs_dfs = {}
    for l in range(len(kpca_slices)):
        path_facs_dfs[l] = {}
        for n in range(rank):
            path_facs_dfs[l][n] = pd.DataFrame.from_dict(path_facs_dict[l][n], orient='index', columns=kpca_slices[l].columns)

    return path_facs_dfs

# 0.3 Calculating 95% CI for each pathway
def feature_select_bs_ci(path_facs_df):
   means = []
   cis = []
   for pathway in path_facs_df.columns:
        scores = path_facs_df[pathway].values
        mean_score = np.mean(scores)
        means.append(mean_score)
        ci_lower = np.percentile(scores, 2.5)
        ci_upper = np.percentile(scores,97.5)
        cis.append((ci_lower,ci_upper))

   means = np.array(means)
   cis = np.array(cis)
   return means, cis

# 1. Return the pathways with lower and upper limits not overlapping with zero
def significant_pathways(path_facs_df):
    means, cis = feature_select_bs_ci(path_facs_df)
    pathways,ci_range = zip(*zip(path_facs_df.columns, cis))
    significant_pathways = []
    for pathway,(lower, upper) in zip(pathways, ci_range):
        if (lower > 0) or (upper < 0): 
            significant_pathways.append(pathway)

    return significant_pathways


# 2. Calculating approximate composite of absolute pseudo z-scores for each pathway mean (obtained from bootstrapping)
def combined_pseudoz(path_facs_bs):
    
    rank = len(path_facs_bs)
    pseudo_zs = { n: {} for n in range(rank)}
    for n in range(rank):
       for pathway in path_facs_bs[n].columns:
            scores = path_facs_bs[n][pathway].values
            pseudo_z = scores.mean()/ np.std(scores, ddof=1)
            pseudo_zs[n][pathway] = pseudo_z
        
    
    pseudoz_df = pd.DataFrame.from_dict(pseudo_zs, orient='columns')
    pseudoz_df.columns = [f"Factor {n+1}" for n in range(rank)]

    combined_abs_pseudozs = []
    for index, pathway_pseudoz in pseudoz_df.iterrows():
        abs_pseudozs = np.array(abs(pathway_pseudoz))
        combined_z = sum(abs_pseudozs)
        combined_abs_pseudozs.append(combined_z)
    
    pseudoz_df['Combined'] = combined_abs_pseudozs

    return pseudoz_df


# 2.2 Return a dict of significant pathways {id: name} by cutting combined_abs_pseudozs at a pre-defined percentile
def significant_pseudoz(pseudoz_df, pathway_dict, percentile):
    threshold = np.percentile(pseudoz_df['Combined'], percentile)
    sig_pseudoz = np.where(pseudoz_df['Combined'] > threshold)
    sig_pathids = pseudoz_df.iloc[sig_pseudoz].index.to_list()
    print(f"{len(sig_pathids)} significant pathways",sig_pathids)
    sig_paths =  {pathway: pathway_dict.get(pathway, None) for pathway in sig_pathids}
    return sig_paths

# 2.3 Plotting combined absolute pseudo z-score distribution
def plot_zscore_distribution(pseudoz_df, percentile):
    """Percentile in %"""

    plt.figure(figsize=(10, 6))
    plt.hist(pseudoz_df['Combined'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Combined absolute pseudo z-scores')
    plt.xlabel('Combined absolute pseudo z-score')
    plt.ylabel('Frequency')
    plt.axvline(pseudoz_df['Combined'].mean(), color='k', linestyle='dashed', linewidth=1)  # Line for mean
    plt.axvline(np.percentile(pseudoz_df['Combined'], percentile), color='r', linestyle='dashed', linewidth=1, label=f'{percentile}% Percentile')
    plt.legend()
    plt.grid()
    plt.show()

# 3. For each factor of a decomposed pathway factor matrix, calculate empirical p-values for each pathway and a pathway-specific combined harmonic mean p-value
def harmonic_pvalue(path_facs_bs: dict, sampling_time: int = 10000) -> pd.DataFrame:
    """
    Calculate individual and combined empirical p-values for pathways based on Bootstrap samples.
    
    Parameters:
    - path_facs_bs: Dict of {rank: DataFrames containing bootstrap scores for each pathway}.
    - sampling_time: Number of bootstrap samples used for calculations (default is 1000).
    
    Returns:
    - A DataFrame (index = pathway, column = factor 1, factor 2, hmp) containing individual and harmonic mean p-values for each pathway.
    """

    rank = len(path_facs_bs)
    empirical_ps = {n: {} for n in range(rank)}

    for n in range(rank):
        for pathway in path_facs_bs[n].columns:
            scores = path_facs_bs[n][pathway].values 
            mean_score = scores.mean()
            if mean_score <0:
                counts = sum(1 for score in scores if score > 0)
            else: 
                counts = sum(1 for score in scores if score < 0)
                
            empirical_p = (counts+1) / (sampling_time+1)
            empirical_ps[n][pathway] = empirical_p

    empirical_df = pd.DataFrame.from_dict(empirical_ps, orient='columns')

    empirical_df.columns = [f"Factor {n+1}" for n in range(rank)]

    hmps = []
    for index, pathway_p in empirical_df.iterrows():
        
        p_vals = np.array(pathway_p)
        p_product = 1
        for p_val in p_vals:
            p_product *= p_val
        hmp = rank * p_product / sum(p_vals)
        hmps.append(hmp)

    empirical_df["Harmonic"]=hmps

    return empirical_df


