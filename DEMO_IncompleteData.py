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
from numpy import linspace, logspace, argmax, arange, log10, meshgrid
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
OUTPUT = './IncompleteDataOutput'   # output folder
MINIMIZE_POLISH = True      # after grid search, refine estimates numerically


#######################
### Input selection ###
#######################

# Note: gaps are defined below with variables t_gap_start and t_gap_end



sensitivity = 407880.00 # SBAT HG
rec_type = 'acceleration'
SignalStart= 123.00
SignalEnd= 136.00
NoiseStart= 20.00
NoiseEnd= 90.00
INPUT = './data/CH.SBAT..HGN.sac'


#sensitivity = 407879.00 # FUSIO HG
#rec_type = 'acceleration'
#SignalStart= 116.00
#SignalEnd= 165.00
#NoiseStart= 0.00
#NoiseEnd= 105.00
#INPUT = './data/CH.FUSIO..HGN.sac'


##################
### Start code ###
##################


plt.close('all')

try:
    os.makedirs(OUTPUT)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise



(data, t_axis, sachead)= readSac(INPUT)
Fs = round(1/sachead[0]); Ts = 1/Fs

y_all = np.array(data[:]) /sensitivity

y_all -= np.mean(y_all) # we do not model DC component

z = y_all[int(round(NoiseStart/Ts)):int(round(NoiseEnd/Ts))]     # noise from trace
y = y_all[int(round(SignalStart/Ts)):int(round(SignalEnd/Ts))]
K = len(y)
t_axis = t_axis[0:K]
    
F_vec = rfftfreq(K,Ts) 
M_r = int(np.floor(K/2) + 1)                    # number of real coefficients in DFT
M_i = int(np.floor(K/2) - 1 + np.mod(K, 2))     # number of imaginary coefficients in DFT


## define gaps in seismogram
p1 = np.array([True] *K)            # true where signal is valid
k_gap_start = []
k_gap_end = []



t_gap_start = 0.2*np.max(t_axis)
t_gap_end = 0.27*np.max(t_axis)
k_gap_start.append(int(round(t_gap_start/Ts)))
k_gap_end.append(int(round(t_gap_end/Ts)))
p1[k_gap_start[-1]:k_gap_end[-1]] = False   # false in correspondence of gaps

t_gap_start = 0.5*np.max(t_axis)
t_gap_end = 0.6*np.max(t_axis)
k_gap_start.append(int(round(t_gap_start/Ts)))
k_gap_end.append(int(round(t_gap_end/Ts)))
p1[k_gap_start[-1]:k_gap_end[-1]] = False   # false in correspondence of gaps

y_obs = P1(y, p1)
k=len(y_obs)

assert p1[0] == True  # gaps should not be at the start
assert p1[-1] == True # or at the end
assert len(k_gap_start) == len(k_gap_end)


### Noise estimation

sigma2_z = np.var(z)
CovarianceFunction = np.correlate(z,z,mode='same')
tau_0 = int( np.floor(len(z) /2) )
CovarianceFunction = sigma2_z*CovarianceFunction/CovarianceFunction[tau_0] # normalization (tau=0 should equal to variance)
T_column = CovarianceFunction[tau_0:tau_0+K]    # This is the first column of the covariance matrix
   
   
# understand the length of each block of signal and get covariance matrix column
T_columns = []
if len(k_gap_start) > 0:
    for nn in range(0, len(k_gap_start)+1):   
        if nn == 0:                      # first gap
            l = k_gap_start[0]
        elif nn == len(k_gap_start):     # last gap
            l = K - k_gap_end[nn-1]
        else:
            l = k_gap_start[nn] - k_gap_end[nn-1]
        T_columns.append(T_column[0:l])
else:
    T_columns.append(T_column)
#T_columns = tuple(T_columns)


### MAP nonparametric spectrum estimation

alphaInv_vec = linspace(0.0, sigma2_z ,15)*(K*Ts**2/2)
alphaInv_vec = alphaInv_vec[alphaInv_vec>0]
alpha_vec = 1/alphaInv_vec

LL_alpha = zeros(shape(alphaInv_vec))
for aa in range(len(alphaInv_vec)):
    alphaInv = alphaInv_vec[aa]
    LL_alpha[aa] = -negLL_nonparametric_MAP_alpha(alphaInv_vec[aa], y_obs, T_columns, K, Ts)
    
    
MLE_alpha = alpha_vec[np.nanargmax(LL_alpha)]
LL_alpha_max = np.nanmax(LL_alpha)

#res = minimize(negLL_nonparametric_MAP_alpha, alphaInv_vec[np.nanargmax(LL_alpha)], args=(y_obs, T_columns, K, Ts), method='Nelder-Mead')

res = minimize(negLL_nonparametric_MAP_alpha, alphaInv_vec[np.nanargmax(LL_alpha)], args=(y_obs, T_columns, K, Ts), bounds=[(0, None)], method='L-BFGS-B')

if res.success:
    MLE_alpha = 1/res.x
    LL_alpha_max = -res.fun
else:
    print(res.message)
                
                
plt.ion()
plt.figure()
plt.semilogx(alpha_vec, LL_alpha,'b.')
plt.plot(MLE_alpha, LL_alpha_max,'ro')
plt.xlabel('alpha')
plt.show()



Y_MAP_tmp = nonparametric_MAP(y_obs, T_columns, MLE_alpha, Ts, p1)

Y_MAP = zeros((K,))
Y_MAP[0:M_r] = real(Y_MAP_tmp)
Y_MAP[M_r:] = imag(Y_MAP_tmp[1:M_i+1])

Z_dft = Ts*np.fft.rfft(z, K)
Y_dft = Ts*np.fft.rfft(y, K)





### ML parametric spectrum estimation
t0 = time.time()


Tinv_prep = bd_toeplitz_inverse_multiplication_prep(*T_columns)
(s, logdetT) = bd_toeplitz_slogdet(*T_columns)

# Grid search


omega_vec = linspace(-7, -5, 10)
fc_vec = linspace(0.001, 10, 10)
tstar_vec = linspace(0.001, 0.1, 10)
Omega_vec = 10**omega_vec


tmp_LL = zeros((len(omega_vec),len(fc_vec),len(tstar_vec)))
MLE_y = zeros(shape(y))
for ndx1 in range(0,len(omega_vec)):
    for ndx2 in range(0,len(fc_vec)):
        for ndx3 in range(0,len(tstar_vec)):
            U = BruneModel(F_vec, Omega_vec[ndx1], fc_vec[ndx2], tstar_vec[ndx3], rec_type)

            (LL, MLE_U) = fitSpectrum_incomplete(y_obs, Y_MAP, Ts, Tinv_prep, logdetT, p1, U)
            tmp_LL[ndx1,ndx2,ndx3] = LL


(ndx1, ndx2, ndx3) = np.unravel_index(argmax(tmp_LL), shape(tmp_LL))
X_grid = np.array([omega_vec[ndx1], fc_vec[ndx2], tstar_vec[ndx3]])

print("Elapsed time, after grid search {:.1f}s".format(time.time()-t0))


# Finer optimization
if MINIMIZE_POLISH:
    
    res = minimize(negLL_spectrum_incomplete, X_grid, args=(F_vec, y_obs, Y_MAP, Ts, Tinv_prep, logdetT, p1, False, rec_type),  method='Nelder-Mead')

    if res.success:
        X_MLE = res.x
    else:
        print(res.message)
        X_MLE  = X_grid
        
    if np.allclose(X_grid, X_MLE):
        print("*** minimize() did not move")
    
    print("Elapsed time, after numerical minimization {:.1f}s".format(time.time()-t0))
else:
    X_MLE = X_grid


# ML estimates:
MLE_omega = X_MLE[0]
MLE_Omega = 10**MLE_omega
MLE_fc = X_MLE[1]
MLE_tstar = X_MLE[2]

print("Maximum likelihood estimate of Brune parameters:")
print("\tOmega  = {:.2e}".format(MLE_Omega))
print("\tf_c    = {:.2e}".format(MLE_fc))
print("\tt^star = {:.2e}".format(MLE_tstar))

(LL_MLE, MLE_U) = fitSpectrum_incomplete(y_obs, Y_MAP, Ts, Tinv_prep, logdetT, p1, BruneModel(F_vec, MLE_Omega, MLE_fc, MLE_tstar, rec_type), OptimizePhases=True)
MLE_u = F(MLE_U, K, Ts=Ts) # ML estimate of time-domain signal
    
    



################
### Plotting ###
################

plt.rc('text', usetex=USETEX)
plt.rc('font', family='serif', size=14)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 
plt.rc('legend', fontsize=14) 



# convert to mm
y *= 1000  
z *= 1000  
MLE_u *= 1000 
MLE_U *= 1000 
Y_MAP *= 1000 
Z_dft *= 1000 
Y_dft *= 1000 


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
ndx = arange(0,k_gap_start[0])  # plot the first signal portion
plt.plot(t_axis[ndx], y[ndx], 'k-', label='Observed', linewidth=2)
plt.plot(t_axis[ndx], MLE_u[ndx],'rx-', linewidth=2, markersize=10, markeredgewidth=2, markevery=20, label='ML fit')

ndx = arange(k_gap_start[0],k_gap_end[0])
plt.plot(t_axis[ndx], y[ndx], '-', color = '0.60', label='Missing', linewidth=1)
plt.plot(t_axis[ndx], MLE_u[ndx],'g.-', linewidth=1, markersize=8, markeredgewidth=0, markevery=20, label='Reconstructed')
for gg in range(1,len(k_gap_start)): # plot the gaps
    ndx = arange(k_gap_start[gg],k_gap_end[gg])
    plt.plot(t_axis[ndx], y[ndx], '-', color = '0.60', linewidth=1)
    plt.plot(t_axis[ndx], MLE_u[ndx],'g.-', linewidth=1, markersize=8, markeredgewidth=0, markevery=20)

for gg in range(0,len(k_gap_start)-1): # plot the other signal portion
    ndx = arange(k_gap_end[gg],k_gap_start[gg+1])    
    plt.plot(t_axis[ndx], y[ndx], 'k-', linewidth=2)
    plt.plot(t_axis[ndx], MLE_u[ndx],'rx-', linewidth=2, markersize=10, markeredgewidth=2, markevery=20)

ndx = arange(k_gap_end[-1],K)
plt.plot(t_axis[ndx], y[ndx], 'k-', linewidth=2)
plt.plot(t_axis[ndx], MLE_u[ndx],'rx-', linewidth=2, markersize=10, markeredgewidth=2, markevery=20)

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
plt.legend(fancybox=True)
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
plt.plot(F_vec, abs(Y_dft), 'g-', label='Complete (DFT)', linewidth=2)
plt.plot(F_vec[1:M_i+1], abs(Y_MAP[1:M_i+1] + 1j * Y_MAP[M_r:]), 'k-', label='Observed (MAP)', linewidth=2)
marker_num = 10; marker_spacing = list(logspace(log10(6*F_vec[1]), log10(F_vec[M_i-1]), marker_num)); marker_positions = []
for m in marker_spacing:
    marker_positions.append(np.argmin( abs(F_vec - m)))
plt.plot(F_vec[1:M_i+1], abs(MLE_U[1:M_i+1] + 1j * MLE_U[M_r:]),'c.-', linewidth=2,
         markersize=10, markeredgewidth=2, markevery=marker_positions, label='ML fit (incomplete)')
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
          ncol=2, fancybox=True, shadow=True)
plt.show()
if SAVE_PLOT: plt.savefig('{0}/Fit_FreqDomain.pdf'.format(OUTPUT), dpi=300, format='pdf')



# Save fit to CSV file
if SAVE_PLOT:
    X = np.transpose(np.vstack( (F_vec[1:M_i+1], abs(Y_MAP[1:M_i+1] + 1j * Y_MAP[M_r:]))) )
    np.savetxt('{0}/MAP_fit.csv'.format(OUTPUT), X, fmt='%.5e', delimiter=', ', newline='\n', header='Frequency, Fit', footer='', comments='# ')
    





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
Y_MAP /= 1000 


Slice_fc_omega = zeros((len(omega_vec), len(fc_vec)))
for ndx1 in range(0,len(omega_vec)):
    for ndx2 in range(0,len(fc_vec)):
        U = BruneModel(F_vec, Omega_vec[ndx1], fc_vec[ndx2], MLE_tstar, rec_type)
        (LL, U) = fitSpectrum_incomplete(y_obs, Y_MAP, Ts, Tinv_prep, logdetT, p1,  U, False)
        Slice_fc_omega[ndx1,ndx2] = LL
            
Slice_tstar_omega = zeros((len(omega_vec), len(tstar_vec)))
for ndx1 in range(0,len(omega_vec)):
    for ndx3 in range(0,len(tstar_vec)):
        U = BruneModel(F_vec, Omega_vec[ndx1], MLE_fc, tstar_vec[ndx3], rec_type)
        (LL, U) = fitSpectrum_incomplete(y_obs, Y_MAP, Ts, Tinv_prep, logdetT, p1,  U, False)
        Slice_tstar_omega[ndx1,ndx3] = LL

Slice_tstar_fc = zeros((len(fc_vec), len(tstar_vec)))
for ndx2 in range(0,len(fc_vec)):
    for ndx3 in range(0,len(tstar_vec)):
        U = BruneModel(F_vec, MLE_Omega, fc_vec[ndx2], tstar_vec[ndx3], rec_type)
        (LL, U) = fitSpectrum_incomplete(y_obs, Y_MAP, Ts, Tinv_prep, logdetT, p1,  U, False)
        Slice_tstar_fc[ndx2,ndx3] = LL


Slice_tstar_fc /=1e3
Slice_tstar_omega /=1e3
Slice_fc_omega /=1e3


myCmap = 'hot_r' # 'hot_r'
Vmax = np.max([np.max(Slice_tstar_fc), np.max(Slice_tstar_omega), np.max(Slice_fc_omega)])
Vmin = np.max([np.min(Slice_tstar_fc), np.min(Slice_tstar_omega), np.min(Slice_fc_omega)])
Vmin = -40 # manually change this for better visualization




fig = plt.figure()
gs = gridspec.GridSpec(2, 2)    
gs.update(wspace=0.1, hspace=0.1)
ax0 = plt.subplot(gs[0])
im0 = plt.imshow(Slice_fc_omega, extent=[fc_vec[0],fc_vec[-1],DeltaMw_vec[0], DeltaMw_vec[-1]], aspect='auto', origin='lower',
                  interpolation='bicubic', cmap=plt.get_cmap(myCmap), vmin=Vmin, vmax=Vmax)
#im0.set_clim(np.min(p), np.max(p))
plt.plot([MLE_fc], [0], 'w.-', linewidth=2, markersize=10, markeredgewidth=2)
plt.ylim(DeltaMw_vec[0], DeltaMw_vec[-1])
plt.xlim(fc_vec[0],fc_vec[-1])
if USETEX:
    plt.ylabel('$\Delta M_w$')
    plt.xlabel('$f_c$ [Hz]')
else:
    plt.ylabel('Delta Mw')
    plt.xlabel('fc [Hz]')

ax1 = plt.subplot(gs[1])
im1 = plt.imshow(Slice_tstar_omega, extent=[tstar_vec[0], tstar_vec[-1],DeltaMw_vec[0], DeltaMw_vec[-1]], aspect='auto', origin='lower',
                  interpolation='bicubic', cmap=plt.get_cmap(myCmap), vmin=Vmin, vmax=Vmax)
plt.plot([ MLE_tstar], [0], 'w.-', linewidth=2, markersize=10, markeredgewidth=2)
plt.ylim(DeltaMw_vec[0],DeltaMw_vec[-1])
plt.xlim(tstar_vec[0], tstar_vec[-1])
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off',      # ticks along the bottom edge are off
    right='off',      # ticks along the bottom edge are off
    labelbottom='off',
    labelleft='off'
    )

cbaxes = fig.add_axes([0.13, 0.13, 0.05, 0.3]) 

if USETEX:
    cb = plt.colorbar(cax = cbaxes, label='Loglikelihood [$\cdot 10^3$]') 
else:
    cb = plt.colorbar(cax = cbaxes, label='Loglikelihood [*1000]') 
    
ax3 = plt.subplot(gs[3])
xx, yy = meshgrid(tstar_vec, fc_vec)
plt.plot([ MLE_tstar], [ MLE_fc], 'w.-', linewidth=2, markersize=10, markeredgewidth=2)
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