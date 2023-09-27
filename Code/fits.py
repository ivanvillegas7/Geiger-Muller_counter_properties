# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:57:05 2023

@author: Iván
"""

import numpy as np

import scipy as sc

def best_fit_func_exp(x: np.array(float), N_0: float, des_const: float):
    
    """
    This function calculates the value of N_0*e^(-des_const*x)
    
    Parameters:
    - x (float): The value of the variable x
    - N_0 (float): The value of the parameter N_0
    - des_const (float): The value of the parameter des_const
    
    Returns:
    - The value of N_0*e^(-des_const*x)
    """
    
    return N_0*np.exp(-des_const*x)

def best_fit_exp(x: np.array(float), y: np.array(float)):
    
    """
    A function that fits a set of data (x,y) to a user-defined function using
    the curve_fit function from scipy.optimize.
    
    Parameters:
    - x (1D array): Independent variable values (time).
    - y (1D array): Dependent variable values (flux).
    
    Returns:
    - popt (1D array): The parameters of the best fit.
    """
    
    return sc.optimize.curve_fit(best_fit_func_exp, x, y, p0=[1, 0])[0]

def best_fit_func_lin(x: np.array(float), lN_0: float, des_const: float):
    
    """
    This function calculates the value of -des_const*x+lN_0
    
    Parameters:
    - x (float): The value of the variable x
    - lN_0 (float): The value of the parameter lN_0
    - des_const (float): The value of the parameter des_const
    
    Returns:
    - The value of -des_const*x+lN_0
    """
    
    return -des_const*x + lN_0

def best_fit_lin(x: np.array(float), y: np.array(float)):
    
    """
    A function that fits a set of data (x,y) to a user-defined function using
    the curve_fit function from scipy.optimize.
    
    Parameters:
    - x (1D array): Independent variable values (time).
    - y (1D array): Dependent variable values (flux).
    
    Returns:
    - popt (1D array): The parameters of the best fit.
    - cov (2D-array): The covariance matrix
    """
    
    return sc.optimize.curve_fit(best_fit_func_lin, x, y)