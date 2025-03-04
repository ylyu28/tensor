import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorly as tl
import tlviz
from tlviz.utils import normalise
from tensorly.decomposition import parafac
from tensorly.decomposition import parafac2

def degeneracy_score_adapted(cp_tensor):
    weights, factors = cp_tensor
    rank = factors[0].shape[1]
    tucker_congruence_scores = np.ones(shape=(rank, rank))
    if rank == 1:
        ds = 1
    else:
        for factor in factors:
            tucker_congruence_scores *= normalise(factor).T @ normalise(factor)

    # update the diagonal to zero to skip it in the max search
        np.fill_diagonal(tucker_congruence_scores,0)

        ds = abs(np.asarray(tucker_congruence_scores)).max()
    return ds 

#def sampleFS_restrictiveLevel(cp_tensor):
#    weights, factors = cp_tensor
#    rank = factors[0].shape[1]
#    tucker_congruence_scores = np.ones(shape=(rank,rank))
#    if rank == 1:
#        ss = 1
#    else:
#        tucker_congruence_scores *= normalise(factors[0]).T @ normalise(factors[0])
#        
#    np.fill_diagonal(tucker_congruence_scores,0)
#
#    ss_df = pd.DataFrame(tucker_congruence_scores,
#                             index = range(tucker_congruence_scores.shape[0]),
#                             columns=range(tucker_congruence_scores.shape[1]))
#
#    ss = np.asarray(tucker_congruence_scores).max()
#    return ss_df, ss



def df_rank_evaluation(original_tensor, max_rank, predictor_y, sklearn_estimator):
    results_dict = {
        'i':[],
        'cc':[],
        'fit':[],
        'sse':[],
        'pp':[],
        'ds':[] 
    }
    for i in range(1, max_rank+1):
        cp_tensor = parafac(original_tensor,i)

        cc = tlviz.model_evaluation.core_consistency(cp_tensor,original_tensor)

        fit = tlviz.model_evaluation.fit(cp_tensor,original_tensor)

        sse = tlviz.model_evaluation.sse(cp_tensor,original_tensor)

        pp = tlviz.model_evaluation.predictive_power(cp_tensor, predictor_y, sklearn_estimator)

        # Computing degeneracy_score across different components 
        ds = degeneracy_score_adapted(cp_tensor)

        results_dict['i'].append(i)
        results_dict['cc'].append(cc)
        results_dict['fit'].append(fit)
        results_dict['sse'].append(sse)
        results_dict['pp'].append(pp)
        results_dict['ds'].append(ds)
    
    results_df = pd.DataFrame(results_dict)
    results_df.columns = ['Number of components','Core consistency','Fit', 'Sum Squared Error','Predictive Power', 'Degeneracy Score']
    results_df.set_index('Number of components', inplace=True)
    
    return results_df



def degeneracy_score_p2(p2_slices, max_rank):

    ds_dict = {}
    for rank in range(1,max_rank+1):
        cp_tensors = parafac2(p2_slices, rank,init='svd')
        weights, cp_factors, projections = cp_tensors
        # note that pathway factor matrices are ignored as constraints are put on them 
        factors = [cp_factors[1],cp_factors[0]]

        rank = factors[0].shape[1]
        tucker_congruence_scores = np.ones(shape=(rank, rank))
        if rank == 1:
            ds = 1
        else:
            for factor in factors:
                tucker_congruence_scores *= normalise(factor).T @ normalise(factor)

        # update the diagonal to zero to skip it in the max search
            np.fill_diagonal(tucker_congruence_scores,0)

            ds = abs(np.asarray(tucker_congruence_scores)).max()

        ds_dict[rank] = ds
    ds_df = pd.DataFrame.from_dict(ds_dict,orient='index')
    return ds_df


