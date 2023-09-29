#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:41:31 2023

@author: Iván
"""

import numpy as np

import matplotlib.pyplot as plt

import scipy as sc

from sklearn.linear_model import LinearRegression as LR

def how_good(x: np.array(float), y: np.array(float), file: str):

    # Fit a linear regression model
    model = LR()
    model.fit(x.reshape(-1, 1), y)
    
    # Calculate residuals
    residuals = y - model.predict(x.reshape(-1, 1))
    
    # Calculate chi-squared statistic
    chi_squared = np.sum(residuals**2 / np.var(residuals))
    
    # Calculate degrees of freedom (number of data points - number of model parameters)
    degrees_of_freedom = len(x) - 2  # Subtract 2 for the intercept and slope in linear regression
    
    # Calculate p-value
    p_value = 1 - sc.stats.chi2.cdf(chi_squared, degrees_of_freedom)
    
    # Plot histogram of chi-squared values
    plt.figure()
    plt.hist(chi_squared, bins=10, density=True, alpha=0.6)
    plt.xlabel('Chi-squared')
    plt.ylabel('Frequency')
    plt.title('Chi-squared Goodness of Fit')
    plt.axvline(chi_squared, color='red', linestyle='dashed', linewidth=2, label=f'Chi-squared = {chi_squared:.2f}')
    plt.legend()
    plt.savefig(f'../Plots/Histogram chi2 - {file}.pdf')
    
    print(f"Chi-squared: {chi_squared:.2f}")
    print(f"Degrees of Freedom: {degrees_of_freedom}")
    print(f"P-value: {p_value:.4f}")
