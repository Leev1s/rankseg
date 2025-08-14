# Author: Ben Dai <bendai@cuhk.edu.hk>
# License: BSD 3 clause

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.stats import rv_continuous


class rankseg(object):
    def __init__(self, 
                 output, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 app=2,
                 smooth=0., 
                 pruning=True, 
                 verbose=0):
        self.output = output
        self.device = device
        self.app = app
        self.smooth = smooth
        self.pruning = pruning
        self.verbose = verbose

    def fit(self):
        self.predict, self.tau_rd, self.cutpoint_rd = rank_dice(self.output, self.device, self.app, self.smooth, self.allow_overlap, self.truncate_mean, self.pruning, self.verbose)
        return self