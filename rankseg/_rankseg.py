# Author: Ben Dai <bendai@cuhk.edu.hk>
# License: BSD 3 clause

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from rankseg import rankdice_batch

class rankseg(object):
    def __init__(self,
                 metric='dice',
                 smooth=0.,
                 solver='BA',
                 eps=1e-4,
                 pruning_prob=0.5):
        self.metric = metric
        self.smooth = smooth
        self.solver = solver
        self.eps = eps
        self.pruning_prob = pruning_prob

    def predict(self, probs):
        if self.metric == 'dice':
            if self.solver in ['BA', 'TRNA', 'auto']:
                preds = rankdice_batch(probs, 
                                    solver=self.solver, 
                                    smooth=self.smooth, 
                                    eps=self.eps,
                                    pruning_prob=self.pruning_prob)
            elif self.solver == 'exact':
                raise NotImplementedError('Exact solver is not implemented yet')
            elif self.solver == 'RMA':
                raise NotImplementedError('RMA solver is not implemented yet')
            else:
                raise ValueError('Unknown solver: %s' % self.solver)
        elif self.metric == 'IoU':
            raise NotImplementedError('IoU metric is not implemented yet')
        else:
            raise ValueError('Unknown metric: %s' % self.metric)
        return preds