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

tauC = 30e-12
rAext = 1000E-10  # not considered
rMext = 2.5E-10
rXext = 2.5E-10

vd = np.loadtxt("vdlist")

#############################
# Function to calculate recovery
#############################

def calc_recovery(t):
	
    # Define the initial state
    initial = np.zeros(8)
    initial[0] = 0.5
    initial[1] = 0 # A spin starts from 0
    initial[2] = AMX_relax.Meq
    initial[3] = AMX_relax.Xeq

    # Create Liouvillian and relaxation matrices
    Gamma = AMX_relax.Gamma(tauC,rAext,rMext,rXext)
    
    rhot = sp.linalg.expm(-Gamma*t) @ initial
    
    AzMaXa = 0.25*(rhot[1]+rhot[4]+rhot[5]+rhot[7])
    AzMaXb = 0.25*(rhot[1]+rhot[4]-rhot[5]-rhot[7])
    AzMbXa = 0.25*(rhot[1]-rhot[4]+rhot[5]-rhot[7])
    AzMbXb = 0.25*(rhot[1]-rhot[4]-rhot[5]+rhot[7])
    
    return AzMaXa, AzMaXb, AzMbXa, AzMbXb 

# Single exponential curve
def func(t, A, R):
    return  A*(1-np.exp(-t*R))

###################################
# Calclate intensities of 4 lines
###################################

AzMaXa = np.zeros(len(vd))
AzMaXb = np.zeros(len(vd))
AzMbXa = np.zeros(len(vd))
AzMbXb = np.zeros(len(vd))

for i in range(len(vd)):
    AzMaXa[i], AzMaXb[i], AzMbXa[i], AzMbXb[i] = calc_recovery(vd[i])

###################################
# Fit to a single exponential curve
###################################

R_aa, pcov_aa = curve_fit(func, vd, AzMaXa, p0=[0.25, 0.3]) 
R_ab, pcov_ab = curve_fit(func, vd, AzMaXb, p0=[0.25, 0.3]) 
R_ba, pcov_ba = curve_fit(func, vd, AzMbXa, p0=[0.25, 0.3]) 
R_bb, pcov_bb = curve_fit(func, vd, AzMbXb, p0=[0.25, 0.3]) 

#######
# Plot
#######

fig1 = plt.figure(figsize=(5.0,1.5),dpi=250)
ax = fig1.add_subplot(131)

simt = np.linspace(0,np.max(vd)*1.05,100)
ax.plot(simt,func(simt,R_aa[0],R_aa[1])/R_aa[0], color="black",linewidth=0.4)

ax.plot(vd, AzMaXa/R_aa[0],  markeredgewidth=0.5, color='white',linewidth=0.,
         markeredgecolor="black", marker='o', markersize=1.5)

ax.set_title("$R_{\\alpha\\alpha}$ = "+str(round(R_aa[1],2))+" (s$^{-1}$)",fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=6)
ax.locator_params(axis='y',nbins=6)
ax.set_xlabel('Relaxation time (sec)',fontsize=6)
ax.set_ylabel('Intensity',fontsize=6)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0.5 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )

ax = fig1.add_subplot(132)

simt = np.linspace(0,np.max(vd)*1.05,100)
ax.plot(simt,func(simt,R_ab[0],R_ab[1])/R_ab[0], color="black",linewidth=0.4)

ax.plot(vd, AzMaXb/R_ab[0],  markeredgewidth=0.5, color='white',linewidth=0.,
         markeredgecolor="black", marker='o', markersize=1.5)

ax.set_title("$R_{\\alpha\\beta}$ = $R_{\\beta\\alpha}$ = "+str(round(R_ab[1],2))+" (s$^{-1}$)",fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=6)
ax.locator_params(axis='y',nbins=6)
ax.set_xlabel('Relaxation time (sec)',fontsize=6)
ax.set_ylabel('Intensity',fontsize=6)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0.5 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )

ax = fig1.add_subplot(133)

simt = np.linspace(0,np.max(vd)*1.05,100)
ax.plot(simt,func(simt,R_bb[0],R_bb[1])/R_bb[0], color="black",linewidth=0.4)

ax.plot(vd, AzMbXb/R_bb[0],  markeredgewidth=0.5, color='white',linewidth=0.,
         markeredgecolor="black", marker='o', markersize=1.5)

ax.set_title("$R_{\\beta\\beta}$ = "+str(round(R_bb[1],2))+" (s$^{-1}$)",fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=6)
ax.locator_params(axis='y',nbins=6)
ax.set_xlabel('Relaxation time (sec)',fontsize=6)
ax.set_ylabel('Intensity',fontsize=6)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0.5 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )

plt.tight_layout()
plt.savefig("recovery_simulation.pdf")