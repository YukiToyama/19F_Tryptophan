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

B0 = 14.1
gF = 2.51815E8 # rad s-1 T-1 # rad s-1 T-1
gH = 2.67522E8 # rad s-1 T-1

wF = B0*gF
wH = B0*gH

hbar = 6.626E-34/2/np.pi

# Thermal equilibrium magnetization
Aeq = 1
Meq = 1*gH/gF
Xeq = 1*gH/gF


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


##############################################
# Function to calculate the relaxation matrix
# Basis set: {E/2, Az, Mz, Xz, 2AzMz, 2AzXz, 2MzXz, 4AzMzXz}  
# Containing external 1H contributions for Gamma
# rMext = rXext = 2.6e-10 for unstructured Trp
# rMext = rXext = 1.5e-10 to 2e-10 for structured Trp
##############################################

def J(w,tauc):
    return 2/5*tauc/(1+w**2*tauc**2)

def Y0(theta): # theta in degree
    return (3*np.cos(theta*np.pi/180)**2-1)/2

def Gamma(tauC,rAext,rMext,rXext):
        
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
    rhoA_CSA = Ac_A**2*JA_wA/2
    #A-M dipolar
    rhoA_AM = Ad_AM**2*JAM_wA/8 + Ad_AM**2*JAM_wAmwM/24 + Ad_AM**2*JAM_wApwM/4 
    #A-X dipolar
    rhoA_AX = Ad_AX**2*JAX_wA/8 + Ad_AX**2*JAX_wAmwX/24 + Ad_AX**2*JAX_wApwX/4
    
    #A-M 2spin order
    rhoAM_AM = Ad_AM**2*JAM_wA/8 + Ad_AM**2*JAM_wM/8
    #A-X 2spin order
    rhoAX_AX = Ad_AX**2*JAX_wA/8 + Ad_AX**2*JAX_wX/8
    
    #M-A dipolar
    rhoM_AM = Ad_AM**2*JAM_wAmwM/24 + Ad_AM**2*JAM_wApwM/4 + Ad_AM**2*JAM_wM/8
    #X-A dipolar
    rhoX_AX = Ad_AX**2*JAX_wAmwX/24 + Ad_AX**2*JAX_wApwX/4 + Ad_AX**2*JAX_wX/8
    
    #Cross relaxation
    # AM dipolar
    sigma_AM = -Ad_AM**2*JAM_wAmwM/24 + Ad_AM**2*JAM_wApwM/4
    # AX dipolar
    sigma_AX = -Ad_AX**2*JAX_wAmwX/24 + Ad_AX**2*JAX_wApwX/4
    
    # Cross correlated relaxation
    # ACSA – AM dipolar
    delta_A_AM = Ac_A*Ad_AM*KAMA_wA/2
    # ACSA – AX dipolar
    delta_A_AX = Ac_A*Ad_AX*KAXA_wA/2
    # AM-AX dipolar-dipolar
    delta_AX_AM = Ad_AM*Ad_AX*KXAM_wA/4

    
    Gamma = np.zeros([8,8])
    
    Gamma[1,1] = rhoA_CSA + rhoA_AM + rhoA_AX
    Gamma[2,2] = rhoM_AM
    Gamma[3,3] = rhoX_AX
    Gamma[4,4] = rhoA_CSA + rhoA_AX + rhoAM_AM
    Gamma[5,5] = rhoA_CSA + rhoA_AM + rhoAX_AX
    Gamma[6,6] = rhoM_AM + rhoX_AX
    Gamma[7,7] = rhoA_CSA +  rhoAM_AM + rhoAX_AX
    
    Gamma[1,2] = Gamma[2,1] = sigma_AM
    Gamma[5,6] = Gamma[6,5] = sigma_AM
    
    Gamma[1,3] = Gamma[3,1] = sigma_AX
    Gamma[4,6] = Gamma[6,4] = sigma_AX
    
    Gamma[1,4] = Gamma[4,1] = Gamma[5,7] = Gamma[7,5] = delta_A_AM 
    
    Gamma[1,5] = Gamma[5,1] = Gamma[4,7] = Gamma[7,4] = delta_A_AX
    
    Gamma[1,7] = Gamma[7,1] = Gamma[4,5] = Gamma[5,4] = delta_AX_AM
    
    # External spin
    RA_ext = (hbar*1E-7*gH*gF/rAext**3)**2*(J(wF-wH,tauC) + 3*J(wF,tauC) + 6*J(wF+wH,tauC))/4
    RM_ext = (hbar*1E-7*gH*gH/rMext**3)**2*(J(0,tauC) + 3*J(wH,tauC) + 6*J(2*wH,tauC))/4
    RX_ext = (hbar*1E-7*gH*gH/rXext**3)**2*(J(0,tauC) + 3*J(wH,tauC) + 6*J(2*wH,tauC))/4
    
    Gamma_ext = np.zeros([8,8])
    Gamma_ext[1,1] = RA_ext
    Gamma_ext[2,2] = RM_ext
    Gamma_ext[3,3] = RX_ext
    Gamma_ext[4,4] = RA_ext + RM_ext
    Gamma_ext[5,5] = RA_ext + RX_ext
    Gamma_ext[6,6] = RM_ext + RX_ext
    Gamma_ext[7,7] = RA_ext + RM_ext + RX_ext

    Gamma = Gamma + Gamma_ext
    
    # Thermal corrections (return to equilibrium) 
    for i in range(1,8):
        Gamma[i,0] = -2*(Gamma[i,1]*Aeq + Gamma[i,2]*Meq + Gamma[i,3]*Xeq)
    
    return Gamma