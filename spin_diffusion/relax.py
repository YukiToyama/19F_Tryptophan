# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:16:53 2023

@author: 外山侑樹
"""

import numpy as np

## Constants
hbar= 6.626E-34/2/np.pi

## Magnetic field
B0 = 14.1 # T

## Dipolar-dipolar interaction following Palmer's notation
def Ad(r,g1,g2):
    return -1*np.sqrt(6)*1E-7*(hbar*g1*g2/r**3) 

## Spectral density function
def J(tauc,w):
    return 2/5*tauc/(1+w**2*tauc**2)

## Auto-relaxation rate
def rho(r,gA,gB,tauc):
    wA = B0*gA
    wB = B0*gB
    J_wAmwB =  J(tauc,wA-wB)
    J_wApwB =  J(tauc,wA+wB)
    J_wA =  J(tauc,wA)
    return Ad(r,gA,gB)**2*(J_wAmwB+3*J_wA+6*J_wApwB)/24 

## Cross-relaxation rate
def sigma(r,gA,gB,tauc):
    wA = B0*gA
    wB = B0*gB
    J_wAmwB =  J(tauc,wA-wB)
    J_wApwB =  J(tauc,wA+wB)
    return Ad(r,gA,gB)**2*(-J_wAmwB+6*J_wApwB)/24 



