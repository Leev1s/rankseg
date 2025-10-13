# Author: Ben Dai <bendai@cuhk.edu.hk>
# License: BSD 3 clause

import numpy as np
import scipy
import torch
import torch.nn.functional as F


class rankseg(object):
    def __init__(self, 
                 smooth=0., 
                 pruning=True, 
                 verbose=0):
        self.smooth = smooth
        self.pruning = pruning
        self.verbose = verbose

    def predict(self, probs):
        ## TBD
        preds = torch.zeros(probs.shape)
        return preds
