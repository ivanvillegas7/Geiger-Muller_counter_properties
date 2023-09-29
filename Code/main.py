# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:04:05 2021

@author: Iván
"""

import numpy as np

import matplotlib.pyplot as plt

import fits

import goodness

def experiment1(file: str):
    
    data = np.loadtxt(f'../Data/{file}.txt', skiprows=1)
    
    voltage: np.array(float) = data[:, 0]
    
    counts1: np.array(float) = data[:, 1]
    
    counts2: np.array(float) = data[:, 2]
    
    counts3: np.array(float) = data[:, 3]
    
    counts: np.array(float) = (counts1 + counts2 + counts3)/3
    
    plt.figure()
    plt.errorbar(voltage, counts, np.sqrt(counts/3), 1)
    plt.yscale("log")
    plt.ylabel(r'$N(V)$')
    plt.xlabel(r'$V$ [V]')
    plt.grid(True)
    plt.savefig('../Plots/N(V) vs V.pdf')
    
def experiments(file: str):
    
    data = np.loadtxt(f'../Data/{file}.txt', skiprows=1)
    
    BG: float
    
    err_BG: float
    
    if file=='exp2-1' or file=='exp2-2':
        
        BG = 6
        
        err_BG = np.abs(BG/10)
        
    elif file=='exp3-1' or file=='exp3-2':
        
        BG = 10
        
        err_BG = np.abs((1/4*1/10)**2*(62+59+54+66))
    
    time: np.array(float) = data[:, 0]
    
    C_BG: np.array(float) = BG*np.ones(len(time))
    
    counts: np.array(float) = data[:, 1]-C_BG
    
    err_counts: np.array(float) = np.abs(data[:, 1])+err_BG*np.ones(len(time))
    
    for i in range(len(counts)):
        
        if counts[i]<0:
            
            counts[i]=1
    
    time_: np.array(float) = np.linspace(0, 600, 1000)
    
    popt_exp: np.array(float) = fits.best_fit_exp(time, counts)
    
    counts_: np.array(float) = popt_exp[0]*np.exp(-1 * popt_exp[1] * time_)
    
    plt.figure()
    plt.errorbar(time, counts, np.sqrt(err_counts), 0.5, ecolor='blue',\
                 linestyle='none', marker='.', label='Experimental data')
    plt.plot(time_, counts_,\
             label=f'N(t)={popt_exp[0]: .3f} exp(-{popt_exp[1]: .3f}t)',\
             color='green')
    plt.ylabel(r'$N(t)$')
    plt.xlabel(r'$t$ [s]')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'../Plots/N(t) vs t exponential - {file}.pdf')
    
    popt_lin: np.array(float)
    
    popt_lin, cov = fits.best_fit_lin(time, np.log(counts))
    
    l_counts_: np.array(float) = -popt_lin[1]*time_+popt_lin[0]
    
    plt.figure()
    plt.errorbar(time, np.log(counts), np.sqrt(err_counts/counts**2), 0.5,\
                 ecolor='blue', linestyle='none', marker='.',\
                 label='Experimental data')
    plt.plot(time_, l_counts_,\
             label=f'N(t)=-{popt_lin[1]: .3f}t+{popt_lin[0]: .3f}',\
             color='green')
    plt.ylabel(r'ln($N(t)$)')
    plt.xlabel(r'$t$ [s]')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'../Plots/N(t) vs t linear - {file}.pdf')
    
    HL: float = np.log(2)/popt_lin[1]
    
    element: str
    
    if file=='exp2-1' or file=='exp2-2':
        
        element='Ba'
        
    elif file=='exp3-1' or file=='exp3-2':
        
        element='Pa'
        
    err: float = (np.log(2)/popt_lin[1]**2)*cov[1][1]
    
    print(f'\nHalf-life for {element}: ({HL}±{err}) s')
    
    goodness.how_good(time, counts, file)
    
def main():
    
    print("")
    
    experiment1(input('Name of the file with the data for determining the plateu and operating voltage for GM-counters: '))
    
    print("")
    
    experiments(input('Name of the file with the data for determining the half-life of metastable barium: '))
    
    print("")
    
    experiments(input('Name of the file with the data for determining the half-life of metastable protactinium: '))
    
main()