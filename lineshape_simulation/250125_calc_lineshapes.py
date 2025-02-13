# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:56:29 2024

@author: toyama
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import importlib
import AMXK_relax_T2 as AMXK_relax
import FT as FT

importlib.reload(AMXK_relax)
importlib.reload(FT)

#############################
# Function to calculate FIDs
#############################

def calcFID(tauC,rAext,rMext,rXext,Rinhom,omegaA,TD,SW):
	
    # Define parameters for FT
    step = 1/(2*SW)
    tmax = step*TD 
    t = np.arange(0,tmax,step)
    
    # Define the initial state
    initial = AMXK_relax.initial
    
    # Create Liouvillian and relaxation matrices
    Gamma, Gamma_dec = AMXK_relax.Gamma(tauC,rAext,rMext,rXext,Rinhom)
    L1, L2, L_dec = AMXK_relax.create_L(omegaA)
    
    # Calculate FID for 1H-coupled lineshapes    
    signal = np.zeros(len(t),dtype=complex)
        
    for i in range(len(t)):
      rhot = sp.linalg.expm((L1-Gamma)*t[i]) @ initial + sp.linalg.expm((L2-Gamma)*t[i]) @ initial
      signal[i] = np.sum(rhot)
    
    # Calculate FID for 1H-decoupled lineshapes    
    signal_dec = np.zeros(len(t),dtype=complex)

    for i in range(len(t)):
      rhot = sp.linalg.expm((L_dec-Gamma_dec)*t[i]) @ initial
      signal_dec[i] = np.sum(rhot) * 2 # Factor of 2 for upfield and downfield components with repsect to spin K

    return signal, signal_dec

def norm(S):
    return (S-np.min(S))/(np.max(S)-np.min(S))

#######
# Plot
#######

# FT parameters (globally set)
TD = 1024
SW = 500

####################################
# 30 ps (indole)
####################################

fig1 = plt.figure(figsize=(2,2),dpi=250)
ax = fig1.add_subplot(111)

tauC = 30e-12
rAext = 1000
rMext = rXext = 2.5E-10
Rinhom = 3 # s-1
omegaA = 0 # Hz
hw = 30 # Hz
signal, signal_dec = calcFID(tauC,rAext,rMext,rXext,Rinhom,omegaA,TD,SW)
freq, signal = FT.FT(signal,TD,SW)
freq, signal_dec = FT.FT(signal_dec,TD,SW)

ax.plot(freq,norm(signal),color="black",linewidth=0.5)
ax.plot(freq,norm(signal_dec),color="tomato",linewidth=0.5)

ax.set_title("$\\tau_c$ = "+str(tauC*1E9)+" (ns), rM = "+str(rMext)+", rX = "+str(rXext),fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=False,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=4)
ax.set_yticklabels([])   
ax.set_xlabel('Frequency (Hz)',fontsize=6)
ax.set_xlim(hw,-hw)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )
plt.tight_layout()
plt.savefig("30ps.pdf")

####################################
# 200 ps (indole + glycerol, 25 deg)
####################################

fig1 = plt.figure(figsize=(2,2),dpi=250)
ax = fig1.add_subplot(111)

tauC = 2e-10
rAext = 1000
rMext = rXext = 2.5E-10
Rinhom = 3 # s-1
omegaA = 0 # Hz
hw = 30 # Hz
signal, signal_dec = calcFID(tauC,rAext,rMext,rXext,Rinhom,omegaA,TD,SW)
freq, signal = FT.FT(signal,TD,SW)
freq, signal_dec = FT.FT(signal_dec,TD,SW)

ax.plot(freq,norm(signal),color="black",linewidth=0.5)
ax.plot(freq,norm(signal_dec),color="tomato",linewidth=0.5)

ax.set_title("$\\tau_c$ = "+str(tauC*1E9)+" (ns), rM = "+str(rMext)+", rX = "+str(rXext),fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=False,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=4)
ax.set_yticklabels([])   
ax.set_xlabel('Frequency (Hz)',fontsize=6)
ax.set_xlim(hw,-hw)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )
plt.tight_layout()
plt.savefig("200ps.pdf")


####################################
# 700 ps (indole + glycerol, 4 deg)
####################################

fig1 = plt.figure(figsize=(2,2),dpi=250)
ax = fig1.add_subplot(111)

tauC = 7e-10
rAext = 1000
rMext = rXext = 2.5E-10
Rinhom = 3 # s-1
omegaA = 0 # Hz
hw = 30 # Hz
signal, signal_dec = calcFID(tauC,rAext,rMext,rXext,Rinhom,omegaA,TD,SW)
freq, signal = FT.FT(signal,TD,SW)
freq, signal_dec = FT.FT(signal_dec,TD,SW)

ax.plot(freq,norm(signal),color="black",linewidth=0.5)
ax.plot(freq,norm(signal_dec),color="tomato",linewidth=0.5)

ax.set_title("$\\tau_c$ = "+str(tauC*1E9)+" (ns), rM = "+str(rMext)+", rX = "+str(rXext),fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=False,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=4)
ax.set_yticklabels([])   
ax.set_xlabel('Frequency (Hz)',fontsize=6)
ax.set_xlim(hw,-hw)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )
plt.tight_layout()
plt.savefig("700ps.pdf")


####################################
# 1.6 ns (IDR 25 deg)
####################################

fig1 = plt.figure(figsize=(2,2),dpi=250)
ax = fig1.add_subplot(111)

tauC = 1.6e-9
rAext = 1000
rMext = rXext = 2.5E-10
Rinhom = 3 # s-1
omegaA = 0 # Hz
hw = 30 # Hz
signal, signal_dec = calcFID(tauC,rAext,rMext,rXext,Rinhom,omegaA,TD,SW)
freq, signal = FT.FT(signal,TD,SW)
freq, signal_dec = FT.FT(signal_dec,TD,SW)

ax.plot(freq,norm(signal),color="black",linewidth=0.5)
ax.plot(freq,norm(signal_dec),color="tomato",linewidth=0.5)

ax.set_title("$\\tau_c$ = "+str(tauC*1E9)+" (ns), rM = "+str(rMext)+", rX = "+str(rXext),fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=False,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=4)
ax.set_yticklabels([])   
ax.set_xlabel('Frequency (Hz)',fontsize=6)
ax.set_xlim(hw,-hw)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )
plt.tight_layout()
plt.savefig("1p6ns.pdf")

####################################
# 5.3 ns (FF 25 deg)
####################################

fig1 = plt.figure(figsize=(2,2),dpi=250)
ax = fig1.add_subplot(111)

tauC = 5.3e-9
rAext = 2.3E-10
rMext = rXext = 2.0E-10
Rinhom = 3 # s-1
omegaA = 0 # Hz
hw = 80 # Hz
signal, signal_dec = calcFID(tauC,rAext,rMext,rXext,Rinhom,omegaA,TD,SW)
freq, signal = FT.FT(signal,TD,SW)
freq, signal_dec = FT.FT(signal_dec,TD,SW)

ax.plot(freq,norm(signal),color="black",linewidth=0.5)
ax.plot(freq,norm(signal_dec),color="tomato",linewidth=0.5)

ax.set_title("$\\tau_c$ = "+str(round(tauC*1E9,2))+" (ns), rA = "+str(rAext)+", rM = "+str(rMext)+", rX = "+str(rXext),fontsize=5)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=False,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=4)
ax.set_yticklabels([])   
ax.set_xlabel('Frequency (Hz)',fontsize=6)
ax.set_xlim(hw,-hw)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )
plt.tight_layout()
plt.savefig("5p3ns.pdf")

####################################
# 14 ns (D2 25 deg)
####################################

fig1 = plt.figure(figsize=(2,2),dpi=250)
ax = fig1.add_subplot(111)

tauC = 14e-9
rAext = 2.3E-10
rMext = rXext = 2.0E-10
Rinhom = 3 # s-1
omegaA = 0 # Hz
hw = 100 # Hz
signal, signal_dec = calcFID(tauC,rAext,rMext,rXext,Rinhom,omegaA,TD,SW)
freq, signal = FT.FT(signal,TD,SW)
freq, signal_dec = FT.FT(signal_dec,TD,SW)

ax.plot(freq,norm(signal),color="black",linewidth=0.5)
ax.plot(freq,norm(signal_dec),color="tomato",linewidth=0.5)

ax.set_title("$\\tau_c$ = "+str(round(tauC*1E9,2))+" (ns), rA = "+str(rAext)+", rM = "+str(rMext)+", rX = "+str(rXext),fontsize=5)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=False,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=4)
ax.set_yticklabels([])   
ax.set_xlabel('Frequency (Hz)',fontsize=6)
ax.set_xlim(hw,-hw)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )
plt.tight_layout()
plt.savefig("14ns.pdf")

