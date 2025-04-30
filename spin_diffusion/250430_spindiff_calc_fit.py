# -*- coding: utf-8 -*-


#####################
# This script allows one to calulcate the saturation recovery profile of 19F
# using a pdb coordinate
#####################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from Bio.PDB.PDBParser import PDBParser
import relax as relax
import importlib
importlib.reload(relax)

from scipy.optimize import curve_fit  

#####################
# Parameters
#####################

## Constants
gH = 2.67522E8 # rad s-1 T-1
gF = 2.51815E8 # rad s-1 T-1

## Correlation time
tauC = 5.3E-9

## Perform 1H saturation at the begining
## (1H magnetization starts from 0)
dec = True

# Read pdb
pdbfile  = "3spin.pdb"

# Probe atom
# This probe atom is treated as fluorine.
state = 0
chainID = "A"
resnum = 1
atomname = "F"

# Threshold distance in A 
threshold = 1000

# Maximum delay to be calculated
maxdelay = 60
points = 100

# Output filename
outname = "3spin_with_decouple"

#####################
# Create Relaxation matrix
#####################

# Read pdb
parser = PDBParser()
struc = parser.get_structure('X',pdbfile)

probe = struc[state][chainID][resnum][atomname]
probe_coord = probe.get_coord()

# Make a list of distances between the probe and each hydrogen atom
proton_list = []

for model in struc.get_list():
    for chain in model.get_list():
        for residue in chain.get_list():
            for atom in residue.get_list():
                if atom.element == "H":
                    # Get coodrdinate
                    H_coordinate = atom.get_coord()
                    distance = atom-probe
                    if not distance==0:
                        proton_list.append([atom, distance])

                    
# Sort and trim the list
sorted_list = sorted(proton_list, key=lambda x: x[1], reverse=False)
trimmed_list = [sublist for sublist in sorted_list if sublist[1] < threshold]

# Add the probe at the top of the list
trimmed_list.insert(0,[probe])

# Construct a distance matrix
distance_matrix = np.zeros((len(trimmed_list),len(trimmed_list)))

for i in range(len(trimmed_list)):
    for j in range(len(trimmed_list)):
        if not i==j:
            distance_matrix[i,j] = trimmed_list[i][0] - trimmed_list[j][0]

Dist = distance_matrix*1E-10

# Construct a relaxation matrix
Relax = np.zeros([len(Dist),len(Dist)])
gamma = np.ones(len(Dist))*gH
gamma[0] = gF
thermal = gamma/gamma[0]

# Calculate auto relaxation rates
for i in range(len(Relax)):
    gamma_obs = gamma[i]
    nonzero_Dist = Dist[i][Dist[i].nonzero()]
    nonzero_gamma = gamma[Dist[i].nonzero()]  
    Relax[i,i] = -1*np.sum(relax.rho(nonzero_Dist,gamma_obs,nonzero_gamma,tauC))

# Calculate cross relaxation rates
for i in range(len(Relax)):
    for j in range(len(Relax)):
        if not Dist[i,j]== 0 :
            Relax[i,j] = -1*relax.sigma(Dist[i,j],gamma[i],gamma[j],tauC)

# Thermal correction
thermal = np.append(0.5,thermal)
Relax = np.vstack((np.zeros(len(Relax)),Relax))
Relax = np.hstack((np.zeros([len(Relax),1]),Relax))

for i in range(len(Relax)):
    Relax[i,0] = -2*np.sum(Relax[i]*thermal)

##################
# Simulation
##################

time = np.linspace(0,maxdelay,points)

Imag = np.zeros(len(time))
Smag = np.zeros(len(time))

if len(Dist)>2:
    Kmag = np.zeros(len(time))

initial = thermal.copy()

# 1H saturation
if dec == True:
    initial[2:] = 0.0

for i in range(len(time)):
    
    rho = initial

    # 19F sat
    rho[1] = 0 
    
    # Trelax 
    rho = sp.linalg.expm(Relax*time[i]) @ rho
    
    Imag[i] += rho[1]
    Smag[i] += rho[2]
    
    if len(Dist)>2:
        Kmag[i] += rho[3]
    
############################################
# Fitting to the single exponential function
############################################

def recovery(time, R1, M0):
    return M0*(1-np.exp(-R1*time))

popt, pcov = curve_fit(recovery, time, Imag, p0=[1, 0.05]) 
R1I, I0  = popt

if dec == True:
    popt, pcov = curve_fit(recovery, time, Smag, p0=[1, 0.05]) 
    R1S, S0  = popt
    
    popt, pcov = curve_fit(recovery, time, Kmag, p0=[1, 0.05]) 
    R1K, K0  = popt

##################
# Plot
##################

fig1 = plt.figure(figsize=(5.0,1.5),dpi=250)
ax = fig1.add_subplot(131)

ax.plot(time,Imag/thermal[1], color="black",linewidth=0.5,label="Simulation")
ax.plot(time, recovery(time, R1I, I0), color="magenta",ls="--",linewidth=0.4,
        label = "$R_{1,fit}$ ="+str(round(R1I,2))+" [s$^{-1}$]")

ax.set_title("$^{19}$F recovery",fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=6)
ax.locator_params(axis='y',nbins=6)
ax.set_xlabel('Relaxation time (sec)',fontsize=6)
ax.set_ylabel('Intensity',fontsize=6)
ax.set_xlim(0,np.max(time)*1.1)
ax.set_ylim(0,1.1)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0.5 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )

ax.legend(fontsize=4,loc='lower right')
   
ax = fig1.add_subplot(132)

ax.plot(time,Smag/thermal[2], color="black",linewidth=0.4,label="Simulation")

if dec == True:
    ax.plot(time, recovery(time, R1S, S0)/thermal[2], color="magenta",ls="--",linewidth=0.4,
            label = "$R_{1,fit}$ ="+str(round(R1S,2))+" [s$^{-1}$]")
    ax.legend(fontsize=4,loc='lower right')

ax.set_title("Closest $^{1}$H profile",fontsize=6)
ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax.locator_params(axis='x',nbins=6)
ax.locator_params(axis='y',nbins=6)
ax.set_xlabel('Relaxation time (sec)',fontsize=6)
ax.set_ylabel('Intensity',fontsize=6)
ax.set_xlim(0,np.max(time)*1.1)
ax.set_ylim(0,1.1)
ax.spines[ 'top' ].set_linewidth( 0 )
ax.spines[ 'left' ].set_linewidth( 0.5 )
ax.spines[ 'right' ].set_linewidth( 0 )
ax.spines[ 'bottom' ].set_linewidth( 0.5 )

if len(Dist)>2:
    ax = fig1.add_subplot(133)
    
    ax.plot(time,Kmag/thermal[3], color="black",linewidth=0.4,label="Simulation")
    
    if dec == True:
        ax.plot(time, recovery(time, R1K, K0)/thermal[3], color="magenta",ls="--",linewidth=0.4,
                label = "$R_{1,fit}$ ="+str(round(R1K,2))+" [s$^{-1}$]")
        ax.legend(fontsize=4,loc='lower right')

    ax.set_title("2nd closest $^{1}$H profile",fontsize=6)
    ax.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
    ax.locator_params(axis='x',nbins=6)
    ax.locator_params(axis='y',nbins=6)
    ax.set_xlabel('Relaxation time (sec)',fontsize=6)
    ax.set_ylabel('Intensity',fontsize=6)
    ax.set_xlim(0,np.max(time)*1.1)
    ax.set_ylim(0,1.1)
    ax.spines[ 'top' ].set_linewidth( 0 )
    ax.spines[ 'left' ].set_linewidth( 0.5 )
    ax.spines[ 'right' ].set_linewidth( 0 )
    ax.spines[ 'bottom' ].set_linewidth( 0.5 )

plt.tight_layout()
plt.savefig(outname+".pdf")

