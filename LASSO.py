# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:53:49 2018

Computer codes related to the publication: 

"Martens et al. 2018. Terrestrial evaporation response to modes of climate 
variability. npj Climate and Atmospheric Science."

Module file to fit a LASSO regression to a dataset of response variables and 
predictor variables. 

FUNCTIONS (see docstrings below for more information):
    - CV_part_k
    - R2_kCV_LASSO
    - FIT_LASSO
    - sigR2_BenHoch

DEPENDENCIES:
    - numpy
    - sklearn
    - statsmodels

@author: Brecht Martens - Laboratory of Hydrology and Water Management, 
                          Ghent University
"""

""" Set up environment """
# Numpy
import numpy as np
# Linear models from sklearn
from sklearn import linear_model
# ECDF function
from statsmodels.distributions.empirical_distribution import ECDF as ecdf

""" Function sigR2_BenHoch """
def sigR2_BenHoch(R2_H1, R2_H0, alpha):
    
    """ Calculate the significance of R2 values using to the 
        Benjamini-Hochberg procedure.

    INPUT:
        - R2_H1 = array: R2 values in alternative distribution (= to be tested)
        - R2_H0 = array: R2 values in null distribution (= reference)
        - alpha = float: significance level

    OUTPUT:
        - bl_sign = array: boolean indication of significance in R2_H1
        - th_sign = float: threshold to declare significance (R2)
    """
    
    # Calculate empirical CDF of H0
    ecdf_H0 = ecdf(R2_H0.squeeze())
    
    # Calculate p-values for H1
    p_H1 = 1-ecdf_H0(R2_H1.squeeze())
    # Set boundaries
    p_H1[p_H1 < 0] = 0
    p_H1[p_H1 > 1] = 1
        
    # Apply Benjamini-Hochberg prodecure
    # Number of samples
    m = R2_H1.squeeze().size
    # Sort p-values
    idx = np.argsort(p_H1)
    p_H1_sort = p_H1[idx]
    R2_H1_sort = R2_H1[idx]
    # Find threshold to declare significance
    k = 1
    while p_H1_sort[k-1] <= k/m*alpha and k < m:
        k+=1
        
    # Set output
    th_sign = R2_H1_sort[k]
    bl_sign = R2_H1 > th_sign
    
    # Return output
    return bl_sign, th_sign

""" Function CV_part_k """
def CV_part_k(n, k, r):
    
    """ Function calculates indices (0 --> k) to partition a dataset of n 
    samples in k folds. Each of the n samples is attributed an index from 
    0 --> k such that there is a (more or less) equal number of samples in each 
    fold. Partionning can be random or in blocks (see below).

    INPUT:
        - n = integer: number of samples
        - k = integer: number of folds
        - r = boolean: True = random split, False = block split

    OUTPUT:
        - ind = (n,  ) numpy array: indices (0 --> k) to split dataset
    """
    
    # Get number of members in each fold (ceiled)
    nm = np.ceil(float(n)/k)
    
    # Generate array of indices ranging from 0 --> k-1 for nm instances
    ind = np.repeat(np.arange(k), nm, axis=0)
    
    # Randomly shuflle array
    if r:
        np.random.shuffle(ind)
    
    # Remove redundant indices
    ind = ind[0:n]
    
    # Return output
    return ind

""" Function R2_kCV_LASSO """
def R2_kCV_LASSO(x, y, k, r, c):
    
    """ Function calculates the R2 - based on a k-fold cross-validation - of a 
    LASSO model fitted on x and y using k-fold CV to find the optimal 
    regularization parameter. As such, a nested cross validation is used to 
    estimate the R2 value, avoiding overestimation of the true R2 as a cross
    validation is already used to optimize the regularization parameter.

    INPUT:
        - x = (n, p) numpy array of floats: n = number of samples, p = number 
              of features
        - y = (n,  ) numpy array of floats: n = number of samples
        - k = integer: number of folds to use in the cross validation
        - r = boolean: True = random split, False = block split
        - c = boolean: True = force regression coefficients to be positive, False = no additional constraints on coefficients

    OUTPUT:
        - R2 = float: the explained variance of the LASSO model
    """
    
    # Calculate indices to partition dataset
    CV_ind = CV_part_k(y.shape[0],k,r)
        
    # Initialise arrays to store predicted values
    R2 = np.empty(y.shape[0])
    
    # Loop over k folds (= outer CV)
    for i1 in np.arange(k):

        # Get training and test data    
        # Get predictor and response variables for testing
        x_tst, y_tst = x[CV_ind == i1,:], y[CV_ind == i1]
        # Get predictor and response variables for training
        x_fit, y_fit = x[CV_ind != i1,:], y[CV_ind != i1]
    
        # Fit LASSO model on training fold and predict test response
        if c:
            y_tst_pred = linear_model.LassoCV(normalize=True, positive=True, cv=k). \
            fit(x_fit,y_fit).predict(x_tst)
            
        else:
            y_tst_pred = linear_model.LassoCV(normalize=True, cv=k). \
            fit(x_fit,y_fit).predict(x_tst)
            
        # Calculate squared difference with test data and store in R2 array
        R2[CV_ind == i1] = np.subtract(y_tst_pred,y_tst)**2
    
    # Calculate R2 value
    R2 = 1-np.mean(R2)/np.var(y)
    
    # Return R2 value
    return R2

""" Function FIT_LASSO """
def FIT_LASSO(x, y, k, r, c):
    
    """ Function fits a LASSO model using x as predictors and y as response 
    variable and calculates the R2 of the fitted model using a nested cross 
    validation. See also documentation of R2_kCV_LASSO.

    INPUT:
        - x = (n, p) numpy array of floats: n = number of samples, p = number 
              of features
        - y = (n,  ) numpy array of floats: n = number of samples
        - k = integer: number of folds to use in the cross validation
        - r = boolean: True = random split, False = block split
        - c = boolean: True = force regression coefficients to be positive, False = no additional constraints on coefficients

    OUTPUT:
        - R2 = float: the explained variance of the fitted LASSO model
        - b  = (p,  ): fitted LASSO regression coefficients
        - alpha = float: optimized regularization parameter
    """
    
    # Fit model on all data
    if c:
        MOD = linear_model.LassoCV(normalize=True, positive=True, cv=k).fit(x,y)
        
    else:
        MOD = linear_model.LassoCV(normalize=True, cv=k).fit(x,y)

    # Calculate explained variance
    R2 = R2_kCV_LASSO(x, y, k, r, c)
    
    # Return R2 value
    return R2, MOD.coef_, MOD.alpha_
