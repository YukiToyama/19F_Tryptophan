# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:56:29 2024

@author: toyama
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  

import importlib
import AMX_relax_T1 as AMX_relax

importlib.reload(AMX_relax)

#############################
# Parameters
#############################

tauClist = np.logspace(-11,-7)

rAext = 1000 #2.3e-10#
rMext = 2.5E-10 #2.0e-10#
rXext = 2.5E-10 #2.0e-10#
outname = "CCR_tauC_low_proton_plot"

vd = np.linspace(0,30,200)

#############################
# Function to calculate recovery
#############################
# Single exponential curve
def func(t, A, R):
    return  A*(1-np.exp(-t*R))

###################################
# Calculate R1 with varying tauC values
###################################

Raa_list = np.zeros(len(tauClist))
Rab_ba_list = np.zeros(len(tauClist))
Rbb_list = np.zeros(len(tauClist))

# Define the initial state
initial = np.zeros(8)
initial[0] = 0.5
initial[1] = 0 # A spin starts from 0
initial[2] = AMX_relax.Meq
initial[3] = AMX_relax.Xeq

for j in range(len(tauClist)):
    
    # Calculation with CCR
    # Create relaxation matrix
    Gamma = AMX_relax.Gamma(tauClist[j],rAext,rMext,rXext)

    # Calclate intensities of 4 lines
    AzMaXa = np.zeros(len(vd))
    AzMaXb = np.zeros(len(vd))
    AzMbXa = np.zeros(len(vd))
    AzMbXb = np.zeros(len(vd))
    
    for i in range(len(vd)):
        
        rhot = sp.linalg.expm(-Gamma*vd[i]) @ initial
        
        AzMaXa[i] = 0.25*(rhot[1]+rhot[4]+rhot[5]+rhot[7])
        AzMaXb[i] = 0.25*(rhot[1]+rhot[4]-rhot[5]-rhot[7])
        AzMbXa[i] = 0.25*(rhot[1]-rhot[4]+rhot[5]-rhot[7])
        AzMbXb[i] = 0.25*(rhot[1]-rhot[4]-rhot[5]+rhot[7])
        
    # Fit to a single exponential curve
    R_aa, pcov_aa = curve_fit(func, vd, AzMaXa, p0=[0.25, 0.3]) 
    R_ab, pcov_ab = curve_fit(func, vd, AzMaXb, p0=[0.25, 0.3]) 
    R_ba, pcov_ba = curve_fit(func, vd, AzMbXa, p0=[0.25, 0.3]) 
    R_bb, pcov_bb = curve_fit(func, vd, AzMbXb, p0=[0.25, 0.3]) 
    
    Raa_list[j] = R_aa[1]
    Rab_ba_list[j] = 0.5*(R_ab[1]+R_ba[1])
    Rbb_list[j] = R_bb[1]
    
    

#######
# Plot
#######

fig1 = plt.figure(figsize=(4.0,1.6),dpi=300)
ax = fig1.add_subplot(121)

ax.plot(tauClist,Raa_list, color="dodgerblue",ls="dashed",linewidth=0.5,label="$R_{\\alpha\\alpha}$")
ax.plot(tauClist,Rab_ba_list, color="magenta",ls="dotted",linewidth=0.5,label="$R_{\\alpha\\beta & \\beta\\alpha}$")
ax.plot(tauClist,Rbb_list, color="orangered",ls="solid",linewidth=0.5,label="$R_{\\beta\\beta}$")

ax.set_title("Relaxation rate",fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=8)
ax.locator_params(axis='y',nbins=4)
ax.set_xlabel('Rotational correlation time [ns]',fontsize=6)
ax.set_ylabel('Relaxation rate [s$^{-1}$]',fontsize=6)
ax.set_xscale("log")
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0.5 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )
ax.legend(fontsize=3.5)

ax = fig1.add_subplot(122)

delta_Raa = abs(Raa_list-Rab_ba_list)/Rab_ba_list
delta_Rbb = abs(Rbb_list-Rab_ba_list)/Rab_ba_list

ax.plot(tauClist,delta_Raa, color="dodgerblue",ls="dashed",linewidth=0.5,label="$(R_{\\alpha\\alpha}-R_{\\alpha\\beta&\\beta\\alpha})/R_{\\alpha\\beta&\\beta\\alpha}$")
ax.plot(tauClist,delta_Rbb, color="orangered",ls="solid",linewidth=0.5,label="$(R_{\\beta\\beta}-R_{\\alpha\\beta&\\beta\\alpha})/R_{\\alpha\\beta&\\beta\\alpha}$")

ax.set_title("Relative contribution",fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=8)
ax.locator_params(axis='y',nbins=4)
ax.set_xlabel('Rotational correlation time [ns]',fontsize=6)
ax.set_ylabel('Relative contribution',fontsize=6)
ax.set_xscale("log")
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0.5 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )
ax.legend(fontsize=4)
plt.tight_layout()

plt.savefig(outname+".pdf")