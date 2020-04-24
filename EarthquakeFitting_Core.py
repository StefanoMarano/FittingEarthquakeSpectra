# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:31:18 2014

@author: marra
"""


import numpy as np
import struct
from math import atan2
from scipy.optimize import minimize
from numpy import asarray, zeros, pi, power, exp, cos, sin, mod, real, imag, dot, log
from numpy.fft import rfft, irfft

from ToeplitzOperations import *
    

def BruneModel(F_vec, Omega, f_c , tstar, t='velocity'):
    """
    This function evaluates the Brune spectrum at specified frequency values

    Parameters
    ----------
        F_vec : a vector of frequencies where the spectrum will be evaluated
        Omega : first parameter of the Brune spectral model
        f_c :   second parameter of the Brune spectral model
        tstar : third parameter of the Brune spectral model
        t : type of spectrum. Default is 'velocity'. Alternatives are 'displacement' and 'acceleration'
    Returns
    -------
        U :    containing the amplitude of the spectrum at the specified frequencies
    """
    
    U = zeros(shape(F_vec))
    if t == 'displacement':         # displacement spectrum
        U = Omega/(1 + power(F_vec/f_c,2))*exp(-np.pi*F_vec*tstar)
    elif t == 'velocity':         # velocity spectrum
        U = (2*pi*F_vec) * Omega/(1 + power(F_vec/f_c,2))*exp(-np.pi*F_vec*tstar)
    elif t == 'acceleration':   # accelaration spectrum
        U = (2*pi*F_vec)**2 * Omega/(1 + power(F_vec/f_c,2))*exp(-np.pi*F_vec*tstar)
    else:
        error('unrecognized spectrum option')

    return U
    

def F(x, N, T=False, I=False, Ts=1):
    """
    The operator F takes real and imaginary part of discrete Fourier spectrum
    and outputs IDFT.
    
    y = F x computes the IDFT.
    
    Parameters
    ----------
        x : a real vector of size N, containing M_r real coefficients and M_i imaginary coefficients
        N : the length of the input vector N
        T : if true, computes transpose(F) x
        I : if true, computes inverse(F) x
        Ts : sampling interval
    Returns
    -------
        y : the output vector
    """

    M_r = int(np.floor(N/2) + 1)                    # number of real coefficients in FFT
    M_i = int(np.floor(N/2) - 1 + np.mod(N, 2))     # number of imaginary coefficients in FFT
    assert N == M_r + M_i
    assert Ts > 0 
    
    x = asarray(x)
    
    if I:
        T = not T
        c = Ts
    else:
        c = 2/(N*Ts)
        
    if T == False:
        t1 = zeros((M_r,)) + 1j*zeros((M_r,))
        t1[0:M_r] +=x[0:M_r]
        t1[1:M_i+1] += 1j*x[M_r:]        
        t1[0] *= np.sqrt(2)
        if mod(N,2) == 0: t1[M_r-1] *= np.sqrt(2)
        y = irfft(t1, N)*(N/2)
    else:
        y = zeros((N,))    
        t1 = rfft(x)
        y[0:M_r] = real(t1)
        y[0] /= np.sqrt(2)
        if mod(N,2) == 0: y[M_r-1] /= np.sqrt(2)
        y[M_r:] = imag(t1[1:M_i+1])
        
    y *= c
        
    return y    
    
def P1(x, p1, T=False):
    """
    reduces a vector of length N to length n
    
    x the input vector
    p1 a vector specifying which elements of x to retain
    T for T=True the transpose is computed
    
    Parameters
    ----------
        x : an input vector of length N
        p1 : a vector specifying which elements of x to retain
        T : if true, computes transpose(P) x
        
    Returns
    -------
        y : the output vector of length n
    """
    
    N = len(p1)
    n = np.sum(p1)
    x = asarray(x)
    if T == False:
        assert len(x) == N
        y = x[p1]
    else:
        assert len(x) == n
        y = zeros((N,))
        y[p1] = x
    return y

### Routines for complete data

def nLL_phi(phi, _U, _Y, _M_r, _M_i, Tinv_prep, K, Ts=1):
    """
    Computes the negative loglikelihood of the spectrum as a function of the phases
     
    Parameters
    ----------
        phi : is a vector of M_i phases
        _U :
        _Y :
        _M_r :
        _M_i :
        Tinv_prep :
        K :
        Ts : sampling interval
    
    Returns
    -------
        The negative log likelihood. A constant value is missing in this computation, a constant non depending on phi.
    """
    _m_U = zeros((K,))
    _m_U[1:_M_i+1] = _U[1:_M_i+1]*cos(phi)
    _m_U[_M_r:_M_i+_M_r] = _U[1:_M_i+1]*sin(phi)

    return dot(_Y-_m_U, F(toeplitz_inverse_multiplication(F(_Y-_m_U,K, Ts=Ts), *Tinv_prep), K, Ts=Ts, T=True))
    
def g_nLL_phi(phi, _U, _Y, _M_r, _M_i, Tinv_prep, K, Ts=1):
    """
    The gradient of nLL_phi()
      
    Parameters
    ----------
        same input as nLL_phi()
    Returns
    -------
        g : the gradient  
    """

    _m_U = zeros((K,))
    _m_U[1:_M_i+1] = _U[1:_M_i+1]*cos(phi)
    _m_U[_M_r:_M_i+_M_r] = _U[1:_M_i+1]*sin(phi)
        
    tmp = 2*F(toeplitz_inverse_multiplication(F(_m_U-_Y,K, Ts=Ts), *Tinv_prep), K, Ts=Ts, T=True)

    g = zeros(shape(phi)) # gradient vector  
    g = _U[1:_M_i+1]*-sin(phi)*tmp[1:_M_i+1] + _U[1:_M_i+1]*cos(phi)*tmp[_M_r:_M_i+_M_r]
    return g

def fitSpectrum(Y, y, Ts, Tinv_prep, logdetT, U, OptimizePhases=False):
    """
    Fit the spectrum U to the time-domain data y
    
    Parameters
    ----------
        m_y :  frequency domain representation of the signal
        y :    the time-domain signal
        Ts :   sampling interval
        Tinv_prep : some quantities needed for fast matrix-vector multiplication with the inverse convariance matrix
        logdetT : log-determinant of the covariance matrix
        U :    specifies the spectrum to fit (amplitudes only)
        OptimizePhases : if true, optimize phases. if false, use a guess from Y
    
    Returns
    -------
        LL :     the loglikelihood of the spectrum
        MLE_m_U : ML estimate of the spectrum (amplitudes and phases)
    """

    K = np.int(len(Y))                              # number of frequencies
    M_r = int(np.floor(K/2) + 1)                    # number of real coefficients in FFT
    M_i = int(np.floor(K/2) - 1 + np.mod(K, 2))     # number of imaginary coefficients in FFT
    
    assert M_r == len(U)
    
    MLE_m_U = np.zeros((K,))    # array for ML estimate of spectrum (amplitudes and phases, actually real and imaginary parts
    MLE_phi = np.zeros((M_i,))  # only M_i phases to be estimated, since DC and Nyquist are real
    
    # initial guess for the phase
    for mm in range(0,M_i):
        MLE_phi[mm] = mod(atan2(Y[mm+M_r], Y[mm+1]), 2*pi)

    if OptimizePhases:
        # t0 = time.time()
        # jac is used by: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg
        # bounds supported by : L-BFGS-B, TNC and SLSQP # 
        res = minimize(nLL_phi, MLE_phi, jac=g_nLL_phi, method='L-BFGS-B',  args=(U, Y, M_r, M_i, Tinv_prep, K, Ts))
        # t1 = time.time()
        # print("elapsed: {}".format(t1-t0))
        
        if res.success:
            MLE_phi = mod(res.x, 2*pi)
        else:
            print(res.message)

    # ML estimate of spectrum
    # the amplitude is given by the input U, in this function we only estimated the phases
    MLE_m_U[1:M_i+1] =     U[1:M_i+1]*cos(MLE_phi) # real part
    MLE_m_U[M_r:M_i+M_r] = U[1:M_i+1]*sin(MLE_phi) # imaginary part
    
    # compute the log-likelihood
    # first, the part inside exp()
    LL = dot(Y-0.5*MLE_m_U, F(toeplitz_inverse_multiplication(F(MLE_m_U,K, Ts=Ts), *Tinv_prep), K, Ts=Ts, T=True))
    # second, the part in front of exp()
    logGamma = -0.5*dot(y,toeplitz_inverse_multiplication(y, *Tinv_prep))
    logGamma += -K/2*log(2*pi) -0.5*logdetT
    # and we sum all together (log domain)
    LL += logGamma
    return (LL, MLE_m_U)

def negLL_spectrum(X, F_vec, Y, y, Ts, Tinv_prep, logdetT, OptimizePhases=False, t='velocity'):
    """
    This function is simply a wrapper used for numerical optimization. It returns
    the negative log-likelihood computed with fitSpectrum()

    Parameters
    ----------
        X : a vector containing the spectrum parameters
            for the Brune model: omega = X[0], f_c = X[1], tstar = X[2]
            
        Other input is directly passed to fitSpectrum()
        
    Returns
    -------
        negLL : The negative log-likelihood
    """
    U = BruneModel(F_vec, 10**X[0], X[1], X[2], t)
    (LL, MLE_U) = fitSpectrum(Y, y, Ts, Tinv_prep, logdetT, U, OptimizePhases)
    negLL = -LL
    
    return negLL
    
    

    
def readSac(sacFile):
    """Load a single SAC file.
 
    Parameters
    ----------
    sacFile : string
        Path to the SAC file.
 
    Returns
    -------
    data : float array
        Data, amplitudes
    t : float array
        Time axis
    sachead : float array
        SAC header
    
    """

    for ESTR in ["<", ">"]:
        #ESTR=">" # big endian # eg, SESAME format
        #ESTR="<" # little endian # eg, SED format
        #ESTR="@" # same as machine
        SHEAD="%c70f35l5L8s16s8s8s8s8s8s8s8s8s8s8s8s8s8s8s8s8s8s8s8s8s8s" % ESTR
        f=open(sacFile, mode="rb")
        sachead_size=struct.calcsize(SHEAD)
        tstr=f.read(sachead_size)
        sachead=struct.unpack(SHEAD, tstr)
        
        nvhdr = sachead[76]
        if nvhdr == 4 | nvhdr == 5:
            # Old sac format. Never tested.
            print("NVHDR = {0}, file {1} may be from old SAC version.".format(nvhdr, sacFile))
        elif nvhdr != 6:
            # We are reading in the wrong byte order.
            f.close()
        elif nvhdr == 6:
            # Good, we are reading in the propoer byte order.
            break
        else:
            print("NVHDR = {0}, file {1} may be corrupted.".format(nvhdr, sacFile))
            
    dt=sachead[0]
    npts=sachead[79]

    t=np.arange(0, npts*dt, dt)
    dsize=struct.calcsize("%c%df" % (ESTR,npts))
    dat_str=f.read(dsize)
    data=np.array(struct.unpack("%c%df" % (ESTR, npts), dat_str))

    f.close()
    return(data, t, sachead)
    
    
    
    
    
    
    
    
    
### Routines for incomplete data

  
def negLL_nonparametric_MAP_alpha(X, y_obs, T_columns, K, Ts):
    """
    Negative loglikelihood as a function of the regularization parameter inv(alpha)
    """
    k = len(y_obs) # number of observed measurements < K
    alphaInv = X
    T_columns_r = np.copy(T_columns)
    for tt in range(0,len(T_columns)): # TODO a manual way to do a deepcopy
        T_columns_r[tt] = np.copy(T_columns[tt])
    
    
     
    for tt in range(0,len(T_columns)):
        #T_columns_r[tt][0] += 2/K/Ts**2/alpha
        T_columns_r[tt][0] += 2/K/Ts**2*alphaInv
    
    (s, logdet) = bd_toeplitz_slogdet(*T_columns_r)
    Tinv_prep = bd_toeplitz_inverse_multiplication_prep(*T_columns_r)
    
    
    return  0.5*(k*np.log(2*pi) + logdet + dot(y_obs, bd_toeplitz_inverse_multiplication(y_obs, *Tinv_prep)))


def nonparametric_MAP(y_obs, T_columns, alpha, Ts, p1):
    """
    Get MAP estimate of the spectrum with regularization parameter alpha
    """
    K = len(p1)
    M_r = int(np.floor(K/2) + 1)                    # number of real coefficients in DFT
    M_i = int(np.floor(K/2) - 1 + np.mod(K, 2))     # number of imaginary coefficients in DFT

    T_columns_r = np.copy(T_columns)
    for tt in range(0, len(T_columns)):
        T_columns_r[tt] = np.copy(T_columns[tt])
    for tt in range(0, len(T_columns)):     # regularized covariance matrix
        T_columns_r[tt][0] += 2/K/Ts**2/alpha # Ts**2*K/alpha/2
    

    Tinv_prep = bd_toeplitz_inverse_multiplication_prep(*T_columns)
    
    t1 = bd_toeplitz_inverse_multiplication(y_obs, *Tinv_prep)
    t2 = F(P1(t1, p1, T=True), K, T=True, Ts=Ts)
    
    ## woodbury identity:
    t3 = P1(F(t2, K, Ts=Ts), p1) /alpha
    Tinv_prep = bd_toeplitz_inverse_multiplication_prep(*T_columns_r)
    t4 = bd_toeplitz_inverse_multiplication(t3, *Tinv_prep)
    t5 = F(P1(t4, p1, T=True), K, T=True, Ts=Ts) /alpha
    t6 = t2/alpha - t5
    
    Y_MAP_gap = zeros((M_r)) +1j*zeros((M_r))
    Y_MAP_gap += t6[0:M_r]
    Y_MAP_gap[1:M_i+1] += 1j*t6[M_r:]
    return Y_MAP_gap
    
def nLL_phi_incomplete(phi, _U, _y_obs, _M_r, _M_i, Tinv_prep, p1, K, Ts=1):
    """
     neg-loglikelihood of the spectrum as a function of the phases
     phi is a vector of M_i phases
    """
    _m_U = zeros((K,))
    _m_U[1:_M_i+1] = _U[1:_M_i+1]*cos(phi)
    _m_U[_M_r:_M_i+_M_r] = _U[1:_M_i+1]*sin(phi)

    mu = _y_obs - P1(F(_m_U, K, Ts=Ts), p1)
    return dot(mu, bd_toeplitz_inverse_multiplication(mu, *Tinv_prep))
     
def g_nLL_phi_incomplete(phi, _U, _y_obs, _M_r, _M_i, Tinv_prep, p1, K, Ts=1):
    # gradient of neg-loglikelihood of the spectrum as a function of the phases
    # phi is a vector of M_i phases

    _m_U = zeros((K,))
    _m_U[1:_M_i+1] = _U[1:_M_i+1]*cos(phi)
    _m_U[_M_r:_M_i+_M_r] = _U[1:_M_i+1]*sin(phi)
        
   
    tmp = 2*F(P1(bd_toeplitz_inverse_multiplication( P1(F(_m_U,K, Ts=Ts),p1) -_y_obs, *Tinv_prep),p1,T=True),K, Ts=Ts, T=True)

    g = zeros(shape(phi)) # gradient vector  
    g = _U[1:_M_i+1]*-sin(phi)*tmp[1:_M_i+1] + _U[1:_M_i+1]*cos(phi)*tmp[_M_r:_M_i+_M_r]
    
    
    return g

def fitSpectrum_incomplete(y_obs, guess_U, Ts, Tinv_prep, logdetT, p1, U, OptimizePhases=False):
    # fit a given spectrum U to the data y at freq/samples specified by F
    # this requires MLE of sigma2, and fitting the phase of each sinusoidal component
    #
    # guess_U is a nonparametric guess of the spectrum, used for the phases
    # W_Y the covariance matrix in freq domain
    # U is the maginitude of the Fourier spectrum to fit. Phases are not specified but estimated


    
    K = np.int(len(guess_U)) # total number of samples/frequencies
    k = len(y_obs) # total number of observed samples/frequencies
    
    M_r = int(np.floor(K/2) + 1)                    # number of real coefficients in FFT
    M_i = int(np.floor(K/2) - 1 + np.mod(K, 2))     # number of imaginary coefficients in FFT
    assert M_r == len(U)
    
    
    MLE_m_U = np.zeros((K,))
    MLE_phi = np.zeros((M_i,))  # only M_i phases to be estimated, since DC and Nyquist are real
    


    # initial guess for the phase
    for mm in range(0,M_i):
        MLE_phi[mm] = mod(atan2(guess_U[mm+M_r], guess_U[mm+1]), 2*pi)
   

    if OptimizePhases:

        # jac is used by: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg
        # bounds supported by : L-BFGS-B, TNC and SLSQP # 
        res = minimize(nLL_phi_incomplete, MLE_phi, jac=g_nLL_phi_incomplete, method='L-BFGS-B',  args=(U, y_obs, M_r, M_i, Tinv_prep, p1, K, Ts))
        
        if res.success:
            MLE_phi = mod(res.x, 2*pi)
        else:
            print(res.message)

    # likelihood value
    MLE_m_U[1:M_i+1] =     U[1:M_i+1]*cos(MLE_phi)
    MLE_m_U[M_r:M_i+M_r] = U[1:M_i+1]*sin(MLE_phi)
       
    
       
    mu = y_obs - P1(F(MLE_m_U, K, Ts=Ts), p1)
    logGamma = -k/2*log(2*pi) -0.5*logdetT
    LL = -0.5*dot(mu, bd_toeplitz_inverse_multiplication(mu, *Tinv_prep))
    LL += logGamma
    
    return (LL, MLE_m_U)
    
    

    
def negLL_spectrum_incomplete(X, F_vec, y_obs, guess_U, Ts, Tinv_prep, logdetT, p1, OptimizePhases=False, t='velocity'):
    # wrapper function to be used with optimization library
    #M0 = X[0]
    #f_c = X[1]
    #tstar = X[2]
    U = BruneModel(F_vec, 10**X[0], X[1], X[2], t)
    (LL, MLE_m_U) = fitSpectrum_incomplete(y_obs, guess_U, Ts, Tinv_prep, logdetT, p1, U, OptimizePhases)
    negLL = -LL
    
    return negLL