#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
    Stefano Maranò

License:
    COPYRIGHT NOTICE See at the end of this file

Description:
    This script implements a method for fitting a given spectral model (eg, the
    Brune spectral model) to given data. It accounts for the non-flat power spectral
    density of the noise.
    
    The code was developed for our publication:
    
    S. Maranò, B. Edwards, G. Ferrari, and D. Fäh, Fitting earthquake spectra:
    Colored noise and incomplete data, Bull. Seism. Soc. Am., 2017.
    
    https://dx.doi.org/10.1785/0120160030

"""


import time, os, errno
import numpy as np
from numpy import var, linspace, argmax, shape, arange, logspace, log10
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import matplotlib.gridspec as gridspec

from EarthquakeFitting_Core import *
from ToeplitzOperations import *



###############
### Options ###
###############

SAVE_PLOT = False            # save plots and CSV to file
USETEX = True              # plots use Latex fonts
OUTPUT = './CompleteDataOutput'   # output folder
MINIMIZE_POLISH = True      # after grid search, refine estimates numerically


#######################
### Input selection ###
#######################

### SBAT HG
INPUT = './data/CH.SBAT..HGN.sac'
Sensitivity = 407880.00
rec_type = 'acceleration'
SignalStart= 123.00; SignalEnd= 136.00
NoiseStart= 20.00; NoiseEnd= 90.00
  
### HASLI HH
#INPUT = './data/CH.HASLI..HHN.sac'
#Sensitivity = 627615000.00
#rec_type = 'velocity'
#SignalStart= 105.00; SignalEnd= 135.00
#NoiseStart= 0.00; NoiseEnd= 99.00

### FUSIO HG
#INPUT = './data/CH.FUSIO..HGN.sac'
#Sensitivity = 407879.00 
#rec_type = 'acceleration'
#SignalStart= 116.00; SignalEnd= 165.00
#NoiseStart= 0.00; NoiseEnd= 105.00

### FUSIO HH
#INPUT = './data/CH.FUSIO..HHN.sac'
#Sensitivity = 600000000.0
#rec_type = 'velocity'
#NoiseStart = 0; NoiseEnd = 105.0
#SignalStart = 116; SignalEnd = 165.0

### WIMIS HH
#INPUT = './data/CH.WIMIS..HHN.sac'
#Sensitivity = 627615000.0
#rec_type = 'velocity'
#SignalStart = 95.3; SignalEnd = 112.0
#NoiseStart = 0; NoiseEnd = 92.8

#### SINS HG
#INPUT = './data/CH.SINS..HGN.sac'
#Sensitivity = 407880.0
#rec_type = 'acceleration'
#SignalStart  = 100.4; SignalEnd = 112.0
#NoiseStart = 0; NoiseEnd = 96.0

### SBUH HG
#INPUT = './data/CH.SBUH..HGN.sac'
#Sensitivity = 407880.0
#rec_type = 'acceleration'
#NoiseStart = 90; NoiseEnd = 120.0
#SignalStart = 138; SignalEnd = 150.0

### SBAT HG
#INPUT = './data/CH.SBAT..HGN.sac'
#Sensitivity = 407880.0
#rec_type = 'acceleration'
#NoiseStart = 20; NoiseEnd = 90.0
#SignalStart = 123; SignalEnd = 136.0

### SCUC HG
#INPUT = './data/CH.SCUC..HGN.sac'
#rec_type = 'acceleration'
#Sensitivity = 407880.0
#SignalStart = 150; SignalEnd = 176.0
#NoiseStart = 0; NoiseEnd = 120.0

### STSP HG
#INPUT = './data/CH.STSP..HGN.sac'
#Sensitivity = 407880.0
#rec_type = 'velocity'
#SignalStart = 150; SignalEnd = 180.0
#NoiseStart = 0; NoiseEnd = 120.0


##################
### Start code ###
##################

plt.close('all')

try:
    os.makedirs(OUTPUT)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

print("Loading {0}".format(INPUT))
(data, t_axis, sachead)= readSac(INPUT)
Fs = round(1/sachead[0]); Ts = 1/Fs

y_all = np.array(data[:]) /Sensitivity
y_all -= np.mean(y_all) # remove DC

z = y_all[int(round(NoiseStart/Ts)):int(round(NoiseEnd/Ts))]    # noise
y = y_all[int(round(SignalStart/Ts)):int(round(SignalEnd/Ts))]  # signal
K = len(y)
t_axis = t_axis[0:K]
    
F_vec = rfftfreq(K,Ts)                          # frequency axis
M_r = int(np.floor(K/2) + 1)                    # number of real coefficients in DFT
M_i = int(np.floor(K/2) - 1 + np.mod(K, 2))     # number of imaginary coefficients in DFT


    
### Noise estimation

sigma2_z = var(z)
CovarianceFunction = np.correlate(z,z,mode='same')
tau_0 = int( np.floor(len(z) /2) )
CovarianceFunction = sigma2_z*CovarianceFunction/CovarianceFunction[tau_0]  # normalization (tau=0 should equal to variance)
T_column = CovarianceFunction[tau_0:tau_0+K]                                # This is the first column of the covariance matrix
Z_dft = Ts*np.fft.rfft(z, K)            # Noise DFT
   
   
# some quantities needed later for likelihood computation
Tinv_prep = toeplitz_inverse_multiplication_prep(T_column) # preprocessing for fast matrix multiplication
(s, logdetT) = toeplitz_slogdet(T_column)                  # covariance matrix determinant


### ML nonparametric spectrum estimation
Y_ML = F(y, K, I=True, Ts=Ts)    # this is equivalent to an inverse DFT

### ML parametric spectrum estimation

t0 = time.time()

# Grid search

omega_vec = linspace(-7, -5, 10)
fc_vec = linspace(0.001, 10, 10)
tstar_vec = linspace(0.001, 0.1, 10)
Omega_vec = 10**omega_vec

tmp_LL = zeros((len(Omega_vec),len(fc_vec),len(tstar_vec)))
for ndx1 in range(0,len(Omega_vec)):
    for ndx2 in range(0,len(fc_vec)):
        for ndx3 in range(0,len(tstar_vec)):
            U = BruneModel(F_vec, Omega_vec[ndx1], fc_vec[ndx2], tstar_vec[ndx3], rec_type)
            (LL, MLE_U) = fitSpectrum(Y_ML, y, Ts, Tinv_prep, logdetT, U, False)
            tmp_LL[ndx1,ndx2,ndx3] = LL

(ndx1, ndx2, ndx3) = np.unravel_index(argmax(tmp_LL), shape(tmp_LL))
X_grid = np.array([omega_vec[ndx1], fc_vec[ndx2], tstar_vec[ndx3]])
LL_grid = tmp_LL[ndx1, ndx2, ndx3]

print("Elapsed time, after grid search {:.1f}s".format(time.time()-t0))


# Finer optimization
if MINIMIZE_POLISH:

    res = minimize(negLL_spectrum, X_grid, args=(F_vec, Y_ML, y, Ts,  Tinv_prep, logdetT, False, rec_type),  method='Nelder-Mead')

    if res.success:
        X_MLE = res.x
        LL_MLE = -res.fun
    else:
        print(res.message)
        X_MLE  = X_grid
        LL_MLE = LL_grid
        
    if np.allclose(X_grid, X_MLE):
        print("*** minimize() did not move")
    
    print("Elapsed time, after numerical minimization {:.1f}s".format(time.time()-t0))
else:
    X_MLE  = X_grid
    LL_MLE = LL_grid


# ML estimates:
MLE_omega = X_MLE[0]
MLE_Omega = 10**MLE_omega
MLE_fc = X_MLE[1]
MLE_tstar = X_MLE[2]

print("Maximum likelihood estimate of Brune parameters:")
print("\tOmega  = {:.2e}".format(MLE_Omega))
print("\tf_c    = {:.2e}".format(MLE_fc))
print("\tt^star = {:.2e}".format(MLE_tstar))

U = BruneModel(F_vec, MLE_Omega, MLE_fc, MLE_tstar, rec_type)
(LL_MLE, MLE_U) = fitSpectrum(Y_ML, y, Ts, Tinv_prep, logdetT, U, OptimizePhases=True)     # Optimize phases
#(LL_MLE, MLE_U) = fitSpectrum(Y_ML, y, Ts, Tinv_prep, logdetT, U, OptimizePhases=False)    # Do not ptimize phases
MLE_u = F(MLE_U, K, Ts=Ts) # ML estimate of time-domain signal





################
### Plotting ###
################

plt.rc('text', usetex=USETEX)
plt.rc('font', family='serif', size=14)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 
plt.rc('legend', fontsize=14) 

plt.ion()

# convert to millimeters
y *= 1000  
z *= 1000  
MLE_u *= 1000 
MLE_U *= 1000 
Y_ML *= 1000 
Z_dft *= 1000 
    

# Figure 1
# noise, noise autocovariance, spectrum
plt.figure()
plt.subplot(3,1,1)
plt.plot(Ts*arange(0,len(z)), z)
plt.xlim(0, Ts*len(z))
if rec_type == 'acceleration':
    if USETEX: plt.ylabel('[$\\textrm{mm}/\\textrm{s}^2$]')
    else: plt.ylabel('[mm/s2]')
elif rec_type == 'velocity':
    if USETEX: plt.ylabel('[$\\textrm{mm}/\\textrm{s}$]')
    else: plt.ylabel('[mm/s]')
else:
    plt.ylabel('[?]')
plt.xlabel('Time [s]')
plt.title("Noise")
plt.subplot(3,1,2)
plt.plot(Ts*arange(0,len(T_column)), T_column)
plt.xlim(0, Ts*len(T_column))
if rec_type == 'acceleration':
    if USETEX: plt.ylabel('[$\\textrm{mm}^2/\\textrm{s}^4$]')
    else: plt.ylabel('[mm/s4]')
elif rec_type == 'velocity':
    if USETEX: plt.ylabel('[$\\textrm{mm}^2/\\textrm{s}^2$]')
    else: plt.ylabel('[mm/s2]')
else:
    plt.ylabel('[?]')
plt.xlabel('Lag [s]')
plt.title("Autocovariance")
plt.subplot(3,1,3)
plt.plot(rfftfreq(K, Ts), Ts*np.abs(rfft(T_column)))
plt.xlim(0,rfftfreq(K, Ts)[-1])
plt.yscale('log')
if rec_type == 'acceleration':
    if USETEX: plt.ylabel('[$\\textrm{mm}/\\textrm{s}$]')
    else: plt.ylabel('[mm/s]')
elif rec_type == 'velocity':
    if USETEX: plt.ylabel('[$\\textrm{mm}$]')
    else: plt.ylabel('[mm]')
else:
    plt.ylabel('[?]')
plt.xlabel('Frequency [Hz]')
plt.title("Power spectral density")
plt.tight_layout()
plt.show()


# Figure 2
# time-domain: measured and fit
plt.figure()
plt.plot(t_axis, y, 'k-', label='Observed', linewidth=2)
plt.plot(t_axis, MLE_u,'rx-', linewidth=1, markersize=10, markeredgewidth=2, markevery=20, label='ML fit')
plt.xlim(t_axis[0],t_axis[-1])
plt.xlabel('Time [s]')
plt.title(INPUT)
if rec_type == 'acceleration':
    if USETEX: plt.ylabel('Acceleration [$\\textrm{mm}/\\textrm{s}^2$]')
    else: plt.ylabel('Acceleration [mm/s2]')
elif rec_type == 'velocity':
    if USETEX: plt.ylabel('Velocity [$\\textrm{mm}/\\textrm{s}$]')
    else: plt.ylabel('Velocity [mm/s2]')
else:
    plt.ylabel('Amplitude [?]')
plt.legend(fancybox=True, shadow=True)
plt.tight_layout()
plt.show()
if SAVE_PLOT: plt.savefig('{0}/Fit_TimeDomain.pdf'.format(OUTPUT), dpi=300, format='pdf')


# Figure 3
# frequency-domain: measured and fit
marker_num = 10
marker_spacing = list(logspace(log10(3*F_vec[1]), log10(F_vec[M_i-1]), marker_num))
marker_positions = []
for m in marker_spacing:
    marker_positions.append(np.argmin( abs(F_vec - m)))
plt.figure()
plt.plot(F_vec, abs(Z_dft),'-', color = '0.75', label='Noise')
plt.plot(F_vec[1:M_i+1], abs(Y_ML[1:M_i+1] + 1j * Y_ML[M_r:]), 'k-', label='Observed (DFT)', linewidth=2)
plt.plot(F_vec[1:M_i+1], abs(MLE_U[1:M_i+1] + 1j * MLE_U[M_r:]),'rx-', linewidth=2,
         markersize=10, markeredgewidth=2, markevery=marker_positions, label='ML fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.title(INPUT)
if rec_type == 'acceleration':
    if USETEX: plt.ylabel('Spectrum [$\\textrm{mm}/\\textrm{s}$]')
    else: plt.ylabel('Spectrum [mm/s]')
elif rec_type == 'velocity':
    if USETEX: plt.ylabel('Spectrum [$\\textrm{mm}$]')
    else: plt.ylabel('Spectrum [mm/s]')
else:
    plt.ylabel('Amplitude [?]')
plt.xlim(F_vec[0], F_vec[-1])
plt.legend(loc='lower left',# bbox_to_anchor=(0.5, 1.0),
          ncol=1, fancybox=True, shadow=True)
plt.tight_layout()
plt.show()
if SAVE_PLOT: plt.savefig('{0}/Fit_FreqDomain.pdf'.format(OUTPUT), dpi=300, format='pdf')


# Save fit to CSV file
if SAVE_PLOT:
    X = np.transpose(np.vstack( (F_vec[1:M_i+1], abs(MLE_U[1:M_i+1] + 1j * MLE_U[M_r:]))) )
    np.savetxt('{0}/ML_fit.csv'.format(OUTPUT), X, fmt='%.5e', delimiter=', ', newline='\n', header='Frequency, Fit', footer='', comments='# ')
    



# Figure 4
# Likelihood slices

# Finer likelihood calculation for plotting
omega_vec = linspace(omega_vec[0], omega_vec[-1], 40)
Omega_vec = 10**omega_vec
DeltaMw_vec = 2/3*np.log10(Omega_vec/MLE_Omega)
fc_vec = linspace(fc_vec[0], fc_vec[-1], 40)
tstar_vec  = linspace(tstar_vec[0], tstar_vec[-1], 40)


# convert back to meters
y /= 1000  
z /= 1000  
MLE_u /= 1000 
MLE_U /= 1000 
Y_ML /= 1000 

Slice_fc_omega = zeros((len(omega_vec), len(fc_vec)))
for ndx1 in range(0,len(omega_vec)):
    for ndx2 in range(0,len(fc_vec)):
        U = BruneModel(F_vec, Omega_vec[ndx1], fc_vec[ndx2], MLE_tstar, rec_type)
        LL = fitSpectrum(Y_ML, y, Ts, Tinv_prep, logdetT, U, False)[0]
        Slice_fc_omega[ndx1,ndx2] = LL
            
Slice_tstar_omega = zeros((len(omega_vec), len(tstar_vec)))
for ndx1 in range(0,len(omega_vec)):
    for ndx3 in range(0,len(tstar_vec)):
        U = BruneModel(F_vec, Omega_vec[ndx1], MLE_fc, tstar_vec[ndx3], rec_type)
        LL = fitSpectrum(Y_ML, y, Ts, Tinv_prep, logdetT, U, False)[0]
        Slice_tstar_omega[ndx1,ndx3] = LL

Slice_tstar_fc = zeros((len(fc_vec), len(tstar_vec)))
for ndx2 in range(0,len(fc_vec)):
    for ndx3 in range(0,len(tstar_vec)):
        U = BruneModel(F_vec, MLE_Omega, fc_vec[ndx2], tstar_vec[ndx3], rec_type)
        LL = fitSpectrum(Y_ML, y, Ts, Tinv_prep, logdetT, U, False)[0]
        Slice_tstar_fc[ndx2,ndx3] = LL
        



Slice_tstar_fc /=1e3
Slice_tstar_omega /=1e3
Slice_fc_omega /=1e3

Vmax = np.max([np.max(Slice_tstar_fc), np.max(Slice_tstar_omega), np.max(Slice_fc_omega)])
Vmin = np.max([np.min(Slice_tstar_fc), np.min(Slice_tstar_omega), np.min(Slice_fc_omega)])

Vmin = -60
myCmap = 'hot_r' # 'hot_r'

fig = plt.figure()
gs = gridspec.GridSpec(2, 2)    
gs.update(wspace=0.1, hspace=0.1)
ax0 = plt.subplot(gs[0])
im0 = plt.imshow(Slice_fc_omega, extent=[fc_vec[0],fc_vec[-1], DeltaMw_vec[0], DeltaMw_vec[-1]], aspect='auto', origin='lower',
                 interpolation='bicubic', cmap=plt.get_cmap(myCmap), vmin=Vmin, vmax=Vmax)
#im0.set_clim(np.min(p), np.max(p))
plt.plot([MLE_fc], [ 0], 'wx-', linewidth=2, markersize=10, markeredgewidth=2)
plt.ylim(DeltaMw_vec[0], DeltaMw_vec[-1])
plt.xlim(fc_vec[0],fc_vec[-1])
if USETEX:
    plt.ylabel('$\Delta M_w$')
    plt.xlabel('$f_c$ [Hz]')
else:
    plt.ylabel('Delta Mw')
    plt.xlabel('fc [Hz]')

ax1 = plt.subplot(gs[1])
im1 = plt.imshow(Slice_tstar_omega, extent=[tstar_vec[0], tstar_vec[-1], DeltaMw_vec[0], DeltaMw_vec[-1]], aspect='auto', origin='lower',
                 interpolation='bicubic', cmap=plt.get_cmap(myCmap), vmin=Vmin, vmax=Vmax)
plt.plot([ MLE_tstar], [ 0], 'wx-', linewidth=2, markersize=10, markeredgewidth=2)
plt.ylim(DeltaMw_vec[0],DeltaMw_vec[-1])
plt.xlim(tstar_vec[0], tstar_vec[-1])
plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off',        # ticks along the bottom edge are off
    right='off',       # ticks along the bottom edge are off
    labelbottom='off',
    labelleft='off'
    )

cbaxes = fig.add_axes([0.13, 0.13, 0.05, 0.3]) 

if USETEX:
    cb = plt.colorbar(cax = cbaxes, label='Loglikelihood [$\cdot 10^3$]') 
else:
    cb = plt.colorbar(cax = cbaxes, label='Loglikelihood [*1000]') 

ax3 = plt.subplot(gs[3])
xx, yy = np.meshgrid(tstar_vec, fc_vec)
plt.plot([ MLE_tstar], [ MLE_fc], 'wx-', linewidth=2, markersize=10, markeredgewidth=2)
im3 = plt.imshow(Slice_tstar_fc, extent=[tstar_vec[0], tstar_vec[-1],fc_vec[0],fc_vec[-1]], aspect='auto', origin='lower',
                 interpolation='bicubic', cmap=plt.get_cmap(myCmap), vmin=Vmin, vmax=Vmax)
plt.ylim(fc_vec[0],fc_vec[-1])
plt.xlim(tstar_vec[0], tstar_vec[-1])
if USETEX:
    plt.ylabel('$f_c$ [Hz]')
    plt.xlabel('$t^*$ [s]')
else:
    plt.ylabel('fc [Hz]')
    plt.xlabel('t* [s]')
plt.show()

if SAVE_PLOT: plt.savefig('{0}/Loglikelihood.pdf'.format(OUTPUT), dpi=300, format='pdf')

print("Close all figures to return to shell.")
plt.show(block=True)