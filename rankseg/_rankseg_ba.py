# Author: Ben Dai <bendai@cuhk.edu.hk>
# License: BSD 3 clause

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.stats import rv_continuous

def rankseg_ba_(probs, 
                app=2, 
                smooth=0., 
                pruning=True, 
                verbose=0):
    """
    Produce the predicted segmentation by `rankdice` based on the estimated output probability.

    Parameters
    ----------
    probs: Tensor, shape (batch_size, num_class, width, height)
        The estimated probability tensor. 
    
    app: int, {0, 1, 2}
        The approximate algo used to implement `rankdice`. 
        `0` indicates exact evaluation, 
        `1` indicates the truncated refined normal approximation (T-RNA), and 
        `2` indicates the blind approximation (BA).
    
    smooth: float, default=0.0
        A smooth parameter in the Dice metric.
    
    verbose: bool, default=0
        Whether print the results for each batch and class.

    Return
    ------
    predict: Tensor, shape (batch_size, num_class, width, height)
        The predicted segmentation based on `rankdice`.

    tau_rd: Tensor, shape (batch_size, num_class)
        The total number of segmentation pixels

    cutpoint_rd: Tensor, shape (batch_size, num_class)
        The cutpoint of probabilties of segmentation pixels and non-segmentation pixels

    Reference
    ---------
    
    """
    batch_size, num_class, width, height = probs.shape
    probs = torch.flatten(probs, start_dim=2, end_dim=-1)
    dim = probs.shape[-1]
    device = probs.device
    ## initialize
    predict = torch.zeros(batch_size, num_class, dim, dtype=torch.bool, device=device)
    tau_rd = torch.zeros(batch_size, num_class, device=device)
    cutpoint_rd = torch.zeros(batch_size, num_class, device=device)
    ## precomputed constants
    discount = torch.arange(2*dim+1, device=device)

    ## ranking
    sorted_prob, top_index = torch.sort(probs, dim=-1, descending=True)
    cumsum_prob = torch.cumsum(sorted_prob, axis=-1)
    ratio_prob = cumsum_prob[:,:,:-1] / (sorted_prob[:,:,1:]+1e-5)

    ## compute statistics of pb distribution
    pb_mean = sorted_prob.sum(axis=-1) # (batch_size, num_class)
    pb_var = torch.sum(sorted_prob*(1-sorted_prob), axis=-1)
    pb_scale = torch.sqrt(pb_var)
    pb_m3 = torch.sum(sorted_prob*(1-sorted_prob)*(1 - 2*sorted_prob), axis=-1)
    pb_skew = pb_m3 / pb_var**(3/2)

    ## compute up_tau (according to Lemma 1)
    ## we only need to search tau over the range [0, up_tau]
    up_tau = torch.argmax(torch.where(ratio_prob - discount[1:dim] - smooth - dim > 0, 1, 0), axis=-1)
    ## if up_tau == 0, it means that the ratio_prob is always less than discount[1:dim] + smooth + dim
    ## in this case, we set up_tau to be dim-1; we cannot prune the search
    up_tau = torch.where(up_tau == 0, dim-1, up_tau)

    ## compute the PMF of the evaluation interval
    RNPB_rv = RefinedNormalPB(dim=dim, loc=pb_mean, scale=pb_scale, skew=pb_skew)
    # Step 1: truncate the evaluation interval [lq, uq] such that P(lq <= X <= uq) = 1 - p
    lq, uq = RNPB_rv.interval(1e-4)
    max_CI = torch.max(uq - lq)
    supp = (torch.arange(max_CI) + lq).view(batch_size, num_class, -1)
    # Step 2: compute the PMF of the evaluation interval
    pmf_supp = RNPB_rv.pdf(supp)
    pmf_supp = pmf_supp / torch.sum(pmf_supp, axis=1, keepdim=True)

    ## Compute ALL pruning masks upfront (fully vectorized)
    mask_prune_prob = (sorted_prob[:,:,0] < 0.5) & pruning  # (batch, num_class)
    mask_prune_tau = up_tau < lq  # Assuming lq is the lower bound check
    mask_small_mean = pb_mean <= 50
    
    mask_skip = mask_prune_prob | mask_prune_tau  # Skip entirely

    mask_use_ba = (pb_mean > 50) & (~mask_skip)  # Use BA
    mask_use_rna = mask_small_mean & (~mask_skip)  # Use RNA

    for k in range(num_class):
        ba_indices = torch.where(mask_use_ba[:, k])[0]
        rna_indices = torch.where(mask_use_rna[:, k])[0]
        skip_indices = torch.where(mask_skip[:, k])[0]

        if len(ba_indices) > 0:
            pmf_tmp = pmf_supp[mask_use_ba[:, k], k]
            ## use convolutional layer to compute (13) in the reference
            low_tmp = lq[mask_use_ba[:, k], k]
            up_tmp = uq[mask_use_ba[:, k], k] + up_tau[mask_use_ba[:, k], k] - 1
            max_range_len = torch.max(up_tmp - low_tmp)
            right_range = torch.arange(max_range_len) + low_tmp.view(-1,1) + 1
            # left, right in (13) of the paper 
            right_denom_tmp = discount[right_range] + smooth + 1
            left_tmp = pmf_tmp.view(-1,1,1)
            # compute (13)                 
            with torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True):
                ma_tmp = F.conv1d(1.0/(right_denom_tmp+1), left_tmp)
                nu_range = F.conv1d(smooth/right_denom_tmp, left_tmp)

            w_range = 2.0*ma_tmp*cumsum_prob[b,k,:up_tau[b,k]]
            
            ## compute score for the range: tilde pi in the paper
            score_range = w_range + nu_range
            score_range = score_range.flatten()
            opt_tau = torch.argmax(score_range)+1
            best_score = score_range[opt_tau-1]
            score_zero = smooth*torch.sum( (1./(discount[low_tmp:up_tmp]+smooth)) * pmf_tmp )
            
            if best_score <= score_zero:
                best_score = score_zero
                opt_tau = 0



    ## searching for optimal vol for each sample and each class
    for k in range(num_class):
        for b in range(batch_size):
            ## prune cases: 
            # (i) the first probability is less than .5 and pruning is enabled
            # (ii) upper bound of tau is less than the lower bound of the truncated interval
            if (sorted_prob[b,k,0] <= .5) and pruning:
                ## pruning for predicted TP = FP = 0
                continue
            ## If the mean is too small, do not use blind approx
            elif pb_mean[b,k] <= 50:
                ## mean is too small; it is too risky to use BA
                up_tau[b,k] = 5*pb_mean[b,k] + 1
                app_tmp = 1
            else:
                app_tmp = app
            
            ## if the upper bounds of the optimal tau is less than the lower bounds of the truncated interval, 
            # we simply take the upper bound as the optimal tau;
            if up_tau[b,k] <= low_class[b,k]:
                ## To be optimized
                ## previous code is: opt_tau = up_tau[b,k]
                ## I think it is better to directly give zero; since all prob are very small
                ## and it should be pruned
                continue
            if app_tmp > 1:
                # pmf_tmp = PB_RNA(pb_mean[b,k],
                #                 pb_var[b,k],
                #                 pb_m3[b,k],
                #                 device=device,
                #                 up=up_class[b,k], low=low_class[b,k])
                # pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)
                pmf_tmp = pmf_supp[b,k]
                
                ## use convolutional layer to compute (13) in the reference
                low_tmp, up_tmp = lq[b,k], uq[b,k] + up_tau[b,k] - 1
                # left, right in (13) of the paper 
                right_denom_tmp = (discount[low_tmp:up_tmp]+smooth+1).view(1,1,-1)
                left_tmp = pmf_tmp.view(1,1,-1)
                
                # compute (13)                 
                with torch.backends.cudnn.flags(enabled=False, deterministic=True, benchmark=True):
                    ma_tmp = F.conv1d(1.0/(right_denom_tmp+1), left_tmp)
                    nu_range = F.conv1d(smooth/right_denom_tmp, left_tmp)

                w_range = 2.0*ma_tmp*cumsum_prob[b,k,:up_tau[b,k]]
                
                ## compute score for the range: tilde pi in the paper
                score_range = w_range + nu_range
                score_range = score_range.flatten()
                opt_tau = torch.argmax(score_range)+1
                best_score = score_range[opt_tau-1]
                score_zero = smooth*torch.sum( (1./(discount[low_tmp:up_tmp]+smooth)) * pmf_tmp )
                
                if best_score <= score_zero:
                    best_score = score_zero
                    opt_tau = 0
                
                if verbose == 1:
                    print('Pred sample-%d; class-%d; mean_pb: %.1f; up_tau:%d; tau_best: %d; score_best: %.4f' %(b, k, pb_mean[b,k], up_tau[b,k], opt_tau, best_score))
            
            if app_tmp <= 1:
                sorted_prob_tmp = sorted_prob[b,k]
                pmf_tmp_zero = PB_RNA(pb_mean[b,k],
                                pb_var[b,k],
                                pb_m3[b,k],
                                device=device,
                                up=up_class[b,k], low=low_class[b,k])
                pmf_tmp_zero = pmf_tmp_zero / torch.sum(pmf_tmp_zero)
                best_score, opt_tau = 0., 0
                if smooth > 0:
                    best_score = smooth*torch.sum((1./(discount[low_class[b,k]:up_class[b,k]]+smooth))*pmf_tmp_zero)
                w_old = torch.zeros(up_class[b,k]-low_class[b,k], dtype=torch.float32, device=device)
                for tau in range(1, up_tau[b,k]+1):
                    pb_mean_tmp = torch.maximum(pb_mean[b,k] - sorted_prob[b,k,tau-1], torch.tensor(0))
                    pb_var_tmp = pb_var[b,k] - sorted_prob[b,k,tau-1]*(1 - sorted_prob[b,k,tau-1])
                    pb_m3_tmp = pb_m3[b,k] - sorted_prob[b,k,tau-1]*(1 - sorted_prob[b,k,tau-1])*(1 - 2*sorted_prob[b,k,tau-1])
                    
                    pmf_tmp = PB_RNA(pb_mean_tmp,
                                    pb_var_tmp,
                                    pb_m3_tmp,
                                    device=device,
                                    up=up_class[b,k], low=low_class[b,k])
                    
                    pmf_tmp = pmf_tmp / torch.sum(pmf_tmp)
                    w_old = w_old + sorted_prob[b,k,tau-1] * pmf_tmp
                    omega_tmp = torch.sum(2./(discount[low_class[b,k]:up_class[b,k]]+tau+smooth+2)*w_old)
                    nu_tmp = smooth*torch.sum((1./(discount[low_class[b,k]:up_class[b,k]]+tau+smooth+1))*pmf_tmp_zero)
                    score_tmp = omega_tmp + nu_tmp
                    if score_tmp > best_score:
                        opt_tau = tau
                        best_score = score_tmp
                    print('(tau, score_tmp): (%d, %.4f)' %(tau, score_tmp))
                if verbose == 1:
                    print('sample-%d; class-%d; mean_tau: %d; up_tau: %d; tau_best: %d; score_best: %.4f' %(b, k, int(pb_mean[b,k]), up_tau[b,k], opt_tau, best_score))
            predict[b, k, top_index[b,k,:opt_tau]] = True
            tau_rd[b,k] = opt_tau
            cutpoint_rd[b,k] = sorted_prob[b,k,opt_tau]
    return predict.reshape(batch_size, num_class, width, height), tau_rd, cutpoint_rd