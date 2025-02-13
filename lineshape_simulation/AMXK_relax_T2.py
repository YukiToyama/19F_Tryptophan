# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:25:59 2024

@author: toyama
"""

import numpy as np

###########
# Constants
###########

# Spin name is defined as follows.
# A: 19F at 5 position
# M: 1H at 6 position
# X: 1H at 4 position
# K: 1H at 7 position (not dipolar coupled to 19F) 

B0 = 14.1
gF = 2.51815E8 # rad s-1 T-1 # rad s-1 T-1
gH = 2.67522E8 # rad s-1 T-1

wF = B0*gF
wH = B0*gH

hbar = 6.626E-34/2/np.pi

# J coupling constants
JAK = 5
JAX = 10
JAM = 10

# CSA values for 5-F Trp
# Lu et al., Journal of Biomolecular NMR (2019) 73:401–409 
TFA = -76.5E-6
xx =  TFA + 4.8E-6   
yy = TFA - 60.5E-6 # Linear to CF bond
zz = TFA - 86.1E-6 # Prependicular to the indole ring

deltasigma = (xx**2 + yy**2 + zz**2 - xx*yy - yy*zz - zz*xx)

# Angles
theta_AMA_x = 33.5
theta_AMA_y = 56.5
theta_AMA_z = 90
theta_AXA_x = 33.5
theta_AXA_y = 56.5
theta_AXA_z = 90
theta_XAM = 112.3

# Distances
rAM = 2.6E-10
rAX = 2.6E-10

# Amplitudes of the interactions
# Ad: dipolar-dipolar, Ac: CSA
Ad_AM = -np.sqrt(6)*1E-7*(hbar*gH*gF/rAM**3)
Ad_AX = -np.sqrt(6)*1E-7*(hbar*gH*gF/rAX**3)
Ac_A = np.sqrt(2/3)*gF*B0

# Initial state
initial = np.ones(4,dtype=complex)

##############################################
# Function to calculate the Liouvillian matrix
# Basis set: {A+MaXa, A+MaXb, A+MbXa, A+MbXb}  
# L1: -JAK/2 Hz component, L2: +JAK/2 Hz component, L3: MXK decoupled
##############################################

def create_L(omegaA):
    
    L1 = np.zeros([4,4],dtype=complex)
    L1[0,0] = 1j*(np.pi*JAM + np.pi*JAX + omegaA - np.pi*JAK)
    L1[1,1] = 1j*(np.pi*JAM - np.pi*JAX + omegaA - np.pi*JAK)
    L1[2,2] = 1j*(-np.pi*JAM + np.pi*JAX + omegaA - np.pi*JAK)
    L1[3,3] =  1j*(-np.pi*JAM - np.pi*JAX + omegaA - np.pi*JAK)
    
    L2 = np.zeros([4,4],dtype=complex)
    L2[0,0] = 1j*(np.pi*JAM + np.pi*JAX + omegaA + np.pi*JAK)
    L2[1,1] = 1j*(np.pi*JAM - np.pi*JAX + omegaA + np.pi*JAK)
    L2[2,2] = 1j*(-np.pi*JAM + np.pi*JAX + omegaA + np.pi*JAK)
    L2[3,3] =  1j*(-np.pi*JAM - np.pi*JAX + omegaA + np.pi*JAK)
    
    L_dec = np.zeros([4,4],dtype=complex)
    L_dec[0,0] = 1j*omegaA
    L_dec[1,1] = 1j*omegaA
    L_dec[2,2] = 1j*omegaA
    L_dec[3,3] = 1j*omegaA
    
    return L1, L2, L_dec

##############################################
# Function to calculate the relaxation matrix
# Basis set: {A+MaXa, A+MaXb, A+MbXa, A+MbXb}  
# Gamma: 1H coupled, Gamma_dec: 1H decoupled
# Containing external 1H contributions for Gamma
# rMext = rXext = ~2.5e-10 for unstructured Trp
# rMext = rXext = 1.5e-10 to 2e-10 for structured Trp
##############################################

def J(w,tauc):
    return 2/5*tauc/(1+w**2*tauc**2)

def Y0(theta): # theta in degree
    return (3*np.cos(theta*np.pi/180)**2-1)/2

def Gamma(tauC,rAext,rMext,rXext,Rinhom):
        
    # Spectral density terms for auto relaxation
    JA_0 = deltasigma*J(0,tauC)
    JA_wA = deltasigma*J(wF,tauC)
    JAM_0 = J(0,tauC)
    JAM_wA = J(wF,tauC)
    JAM_wM = J(wH,tauC)
    JAM_wApwM = J(wF+wH,tauC)
    JAM_wAmwM = J(wF-wH,tauC)
    JAX_0 = J(0,tauC)
    JAX_wA = J(wF,tauC)
    JAX_wX = J(wH,tauC)
    JAX_wApwX = J(wF+wH,tauC)
    JAX_wAmwX = J(wF-wH,tauC)

    # Spectral density terms for cross-correlated relaxation      
    KAMA_0 = (xx*Y0(theta_AMA_x)+yy*Y0(theta_AMA_y)+zz*Y0(theta_AMA_z))*J(0,tauC)
    KAMA_wA = (xx*Y0(theta_AMA_x)+yy*Y0(theta_AMA_y)+zz*Y0(theta_AMA_z))*J(wF,tauC)
    
    KAXA_0 = (xx*Y0(theta_AXA_x)+yy*Y0(theta_AXA_y)+zz*Y0(theta_AXA_z))*J(0,tauC)
    KAXA_wA = (xx*Y0(theta_AXA_x)+yy*Y0(theta_AXA_y)+zz*Y0(theta_AXA_z))*J(wF,tauC)
    
    KXAM_0 =  Y0(theta_XAM)*J(0,tauC)
    KXAM_wA =  Y0(theta_XAM)*J(wF,tauC)
      
    # A spin CSA
    rhoA_CSA = Ac_A**2*JA_0/3 + Ac_A**2*JA_wA/4
    # A CSA – AM dipolar CCR
    delta_A_AM = Ac_A*Ad_AM*KAMA_0/3 + Ac_A*Ad_AM*KAMA_wA/4
    # A CSA – AX dipolar CCR
    delta_A_AX = Ac_A*Ad_AX*KAXA_0/3 + Ac_A*Ad_AX*KAXA_wA/4
    # A-M dipolar
    rhoA_AM = Ad_AM**2*JAM_0/12 + Ad_AM**2*JAM_wA/16 + Ad_AM**2*JAM_wAmwM/48 + Ad_AM**2*JAM_wApwM/8 + Ad_AM**2*JAM_wM/16
    # A-X dipolar
    rhoA_AX = Ad_AX**2*JAX_0/12 + Ad_AX**2*JAX_wA/16 + Ad_AX**2*JAX_wAmwX/48 + Ad_AX**2*JAX_wApwX/8 + Ad_AX**2*JAX_wX/16
    # AM-AX dipolar CCR
    delta_AX_AM = Ad_AM*Ad_AX*KXAM_0/6 + Ad_AM*Ad_AX*KXAM_wA/8 
    
    # 0-1/2-3 AX-dipolar cross relaxation
    sigma_AX = Ad_AX**2*JAX_wX/16
    # 0-2/1-3 AM-dipolar cross relaxation
    sigma_AM = Ad_AM**2*JAM_wM/16
    
    
    Gamma = np.zeros([4,4],dtype=complex)
    Gamma[0,0] = rhoA_CSA + rhoA_AX + rhoA_AM + delta_A_AM + delta_A_AX + delta_AX_AM + Rinhom
    Gamma[1,1] = rhoA_CSA + rhoA_AX + rhoA_AM + delta_A_AM - delta_A_AX - delta_AX_AM + Rinhom
    Gamma[2,2] = rhoA_CSA + rhoA_AX + rhoA_AM - delta_A_AM + delta_A_AX - delta_AX_AM + Rinhom
    Gamma[3,3] = rhoA_CSA + rhoA_AX + rhoA_AM - delta_A_AM - delta_A_AX + delta_AX_AM + Rinhom
    
    Gamma[0,1] = Gamma[1,0] = Gamma[2,3] = Gamma[3,2] = sigma_AX
    Gamma[0,2] = Gamma[2,0] = Gamma[1,3] = Gamma[3,1] = sigma_AM
    
    # A External spin
    RA_ext = (hbar*1E-7*gH*gF/rAext**3)**2*(4*J(0,tauC) + J(wF-wH,tauC) + 3*J(wF,tauC) + 6*J(wH,tauC) + 6*J(wF+wH,tauC))/8
    
    Gamma_ext1 = np.zeros([4,4],dtype=complex)
    Gamma_ext1[0,0] = Gamma_ext1[1,1] = Gamma_ext1[2,2] = Gamma_ext1[3,3] = RA_ext
    
    # MX External spin
    RM_ext = (hbar*1E-7*gH*gH/rMext**3)**2*(J(0,tauC) + 3*J(wH,tauC) + 6*J(2*wH,tauC))/4
    RX_ext = (hbar*1E-7*gH*gH/rXext**3)**2*(J(0,tauC) + 3*J(wH,tauC) + 6*J(2*wH,tauC))/4
    
    Gamma_ext2 = np.zeros([4,4],dtype=complex)
    Gamma_ext2[0,0] = Gamma_ext2[1,1] = Gamma_ext2[2,2] = Gamma_ext2[3,3] = RM_ext/2 + RX_ext/2
    Gamma_ext2[0,1] = Gamma_ext2[1,0] = Gamma_ext2[2,3] = Gamma_ext2[3,2] = -RX_ext/2
    Gamma_ext2[0,2] = Gamma_ext2[2,0] = Gamma_ext2[1,3] = Gamma_ext2[3,1] = -RM_ext/2
    
    Gamma = Gamma + Gamma_ext1 + Gamma_ext2
    
    # 1H decoupled
    Gamma_dec = np.zeros([4,4],dtype=complex)
    Gamma_dec[0,0] = Gamma_dec[1,1] = Gamma_dec[2,2] = Gamma_dec[3,3] =  rhoA_CSA + rhoA_AX + rhoA_AM + sigma_AX + sigma_AM + RA_ext + Rinhom
    
    return Gamma, Gamma_dec