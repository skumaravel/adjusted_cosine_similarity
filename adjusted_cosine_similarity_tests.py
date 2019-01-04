# -*- coding: utf-8 -*-
"""
Tests on adjusted cosine similarity
Created on Fri Jan  4 17:18:46 2019

@author: sujeeth.kumaravel
"""
import numpy as np
from adjusted_cosine_similarity import adjusted_cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

a = np.array([4, 0]).reshape(1,-1)
b = np.array([0, 4]).reshape(1,-1)

print(adjusted_cosine_similarity(a,b))

a = np.array([4, 0, 0, 0, 0]).reshape(1,-1)
b = np.array([5, 0, 0, 0, 0]).reshape(1,-1)

cosine_similarity(a, b) 

a = np.array([4, 0])
b = np.array([0, 4])
c = np.c_[a,b]

import pandas as pd

d = pd.DataFrame(c)

print(cosine_similarity(c)) 
