# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:53:49 2018

Computer codes related to the publication: 

"Martens et al. 2018. Terrestrial evaporation response to modes of climate 
variability. npj Climate and Atmospheric Science."

Module file with an example on how to fit the LASSO models.

@author: Brecht Martens - Laboratory of Hydrology and Water Management, 
                          Ghent University
"""

""" Set up environment """

# Numpy
import numpy as np
# LASSO
import LASSO as ls

""" Generate some toy data """

# Predictors (n = 100, p = 50)
PRED = np.random.randn(100, 50)
# Response (n = 100)
RESP = np.random.randn(100)

""" Fit LASSO model and use nested k-fold cross validation to estimate the R2. 
    Given that random predictor and response data were generated, we expect R2 
    to be <= 0."""    
    
# Fit LASSO model    
R2, bfit, alpha = ls.FIT_LASSO(PRED, RESP, 5, True, False)





















