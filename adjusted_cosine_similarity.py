# -*- coding: utf-8 -*-
"""
Implementation of Adjusted Cosine Similarity
Created on Fri Jan  4 15:00:24 2019

@author: sujeeth.kumaravel
@reference: http://www10.org/cdrom/papers/519/node14.html 
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def adjusted_cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    
    norm_a = a - mu_a
    norm_b = b - mu_b
    
    return cosine_similarity(norm_a, norm_b)
    

