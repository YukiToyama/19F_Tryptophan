# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:25:59 2024

@author: toyama
"""

import numpy as np
import scipy as sp
import scipy.fftpack as fftpack

# Cosine apodization
def SP(FID, off, end, power):
    APOD = np.empty_like(FID)
    tSize = len(FID)
    for i in range(len(FID)):
        APOD[i] = FID[i]*np.sin(np.pi*off + np.pi*(end-off)*i/(tSize-1) )**power
    return APOD
    
# Discrete FT
def FT(signal,TD,SW):
    sampling=np.power(SW*2,-1.)
    S=fftpack.fftshift(fftpack.fft(signal,n=8*TD))
    F=fftpack.fftshift(fftpack.fftfreq(8*TD,sampling))
    return F, np.real(S)


