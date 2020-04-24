# -*- coding: utf-8 -*-
"""

Author:
    Stefano Maranò

License:
    COPYRIGHT NOTICE See at the end of this file

Description:
    This script implements various routines for matrix operations. Operations 
    are fast, mostly relying on the FFT algorithm and do not require the storage 
    of the entire matrix, by exploting properties of the matrices.
    
    - matrix-vector product with inverse of a Toeplitz matrix
    - matrix-vector product with circulant matrix
    - matrix-vector product with Toeplitz matrix
    - recursive computation of Toeplitz matrix determinant
    - Levinson algorithm
    
    The code was developed for our publication:
    
    S. Maranò, B. Edwards, G. Ferrari, and D. Fäh, Fitting earthquake spectra:
    Colored noise and incomplete data, Bull. Seism. Soc. Am., 2017.
    
    https://dx.doi.org/10.1785/0120160030

References:

    Gohberg, I. and V. Olshevsky (1994). Fast algorithms with preprocessing for 
    matrix-vector multiplication problems, Journal of Complexity 10(4), 411–427.
    
    Monahan, J. F. (2011). Numerical Methods of Statistics. Cambridge University 
    Press.
    
    Golub, G.H. and Van Loan, C.F. (2012). Matrix Computations. Johns Hopkins 
    University Press.

"""

import time
import numpy as np
from numpy.linalg import solve, inv, slogdet
from numpy import triu, tril, tri, dot, zeros, real, shape, allclose
from numpy.random import randn
from scipy.linalg import toeplitz, circulant, block_diag
from scipy.fftpack import fft, ifft



def factor_circulant_matrix(x, k):
    """
    Builds a k-factor circulant matrix (A matrix with the structure of circulant 
    matrices, but with the entries above the diagonal multiplied by the same factor.)
    
    The matrix is store in memory.
    
    parameters:
        x the first column of the matrix
        k the multiplying factor
    returns:
        the k-factor circulant matrix
    """
    n=len(x)
    return circulant(x) * (tri(n,n, 0) + k*np.transpose(tri(n,n, -1)))
    
def factor_circulant_multiplication(u, x, k=1):
    """
    Compute the matrix-vector product
    y = Cu
    where C is a k-factor circulant matrix
    
    All matrices are real
    
    parameters:
        u the input vector
        x, k define the k-factor circulant matrix
    returns:
        y the output vector
    """
    n = len(u)    
    D_k = (k**(1/n))**np.arange(0,n)
    Lambda = fft(D_k*x)
    return (1/D_k)*real(ifft(Lambda*fft(D_k*u))) # y
    
def circulant_multiplication(u, a):
    """
    Compute the matrix-vector product
    y = Cu
    where C is a circulant matrix
    
    All matrices are real
    
    parameters:
        u the input vector
        a the first column of the circulant matrix
    returns:
        y the output vector

    """
        
    return real(ifft(fft(a)*fft(u)))
    
def toeplitz_multiplication(u, c, r=None):
    """
    Compute the matrix-vector product
    y = Tu
    where T is a Toeplitz matrix
    
    All matrices are real
    
    parameters:
        u the input vector
        c the first column of the Toeplitz matrix
        r optionally, the first row of the Toeplitz matrix
    returns:
        y the output vector

    """
    n = len(u)
    if r is None:
        r = c
    u1 = zeros((2*n))
    u1[0:n] = u
    
    c = np.concatenate((c, [0], r[-1:0:-1]))    
        
    y1 = circulant_multiplication(u1, c)
    
    return y1[0:n]

def levinson(r, b):
    """
    Solves Tx=b using the Levinson algorithm
    
    where
    T is apositive-definite symmetric Toeplitz matrix
    b is a real vector
    
    parameters:

        r the first row of the Toeplitz matrix
        b the right hand side vector
    returns:
        x the solution
    """

    n = len(b)
    y = zeros((n,))
    x = zeros((n,))
    
    # normalize the system so that the T matrix has diagonal of ones
    r_0 = r/r[0]
    b_0 = b/r[0]
    
    if n == 1:
        return b_0
    
    y[0] = -r_0[1]
    x[0] = b_0[0]
    beta = 1
    alpha = -r_0[1]
    
    for k in range(0,n-1):
        beta = (1 - alpha*alpha)*beta
        mu = (b_0[k+1] - dot(r_0[1:k+2], x[k::-1])) /beta
        x[0:k+1] = x[0:k+1] + mu*y[k::-1]
        x[k+1] = mu
        if k < n-2:
            alpha = -(r_0[k+2] + dot(r_0[1:k+2], y[k::-1]))/beta
            y[0:k+1] = y[0:k+1] + alpha * y[k::-1]
            y[k+1] = alpha

    return x


def toeplitz_slogdet(r):
    """
    Compute the log determinant of a positive-definite symmetric toeplitz matrix.
    The determinant is computed recursively. The intermediate solutions of the 
    Levinson recursion are expolited.
    
    parameters:
        r first row of the Toeplitz matrix
    
    returns:
        sign sign of the determinant
        logdet natural log of the determinant   
    """
    n = len(r)
    r_0 = r[0]
    
    r = np.concatenate((r, np.array([r_0])))
    r /= r_0 # normalize the system so that the T matrix has diagonal of ones
    
    logdet = n*np.log(np.abs(r_0))
    sign = np.sign(r_0)**n
    
    if n == 1:
        return (sign, logdet)
    
    # now on is a modification of Levinson algorithm
    y = zeros((n,))
    x = zeros((n,))

    b = -r[1:n+1]    
    r = r[:n]
    
    y[0] = -r[1]
    x[0] = b[0]
    beta = 1
    alpha = -r[1]
    
    d = 1 + dot(-b[0], x[0])
    sign *= np.sign(d)
    logdet += np.log(np.abs(d))
    
    for k in range(0,n-2):
        
        beta = (1 - alpha*alpha)*beta
        mu = (b[k+1] - dot(r[1:k+2], x[k::-1])) /beta
        x[0:k+1] = x[0:k+1] + mu*y[k::-1]
        x[k+1] = mu
        
        d = 1 + dot(-b[0:k+2], x[0:k+2])
        sign *= np.sign(d)
        logdet += np.log(np.abs(d))
        
        if k < n-2:
            alpha = -(r[k+2] + dot(r[1:k+2], y[k::-1]))/beta
            y[0:k+1] = y[0:k+1] + alpha * y[k::-1]
            y[k+1] = alpha        

    return(sign, logdet)
    
def toeplitz_inverse_multiplication_prep(T_column):
    """
    Preprocessing needed for toeplitz_inverse_multiplication()
    
    parameters:
        T_column first column of the symmetric Toeplitz matrix
    
    returns:
        many things as in Gohberg, I. and V. Olshevsky (1994)
    """
    
    phi=1
    psi=2
    assert phi != 0
    assert psi != 0
    assert phi != psi
    
    n = len(T_column)
    
    x = levinson(T_column, np.concatenate( (np.array([1]), np.zeros((n-1,))) ) )
    y = levinson(T_column, np.concatenate( (np.zeros((n-1,)), np.array([1])) ) )

    
    
    x_0 = x[0]
    
    D_phi = (phi**(1/n))**np.arange(0,n)
    D_psi = (psi**(1/n))**np.arange(0,n)

    Lambda_1 = fft(D_psi*x)
    Lambda_2 = fft(D_phi*np.concatenate(([phi*y[-1]], y[0:-1])))
    Lambda_3 = fft(D_psi*np.concatenate(([psi*y[-1]], y[0:-1])))
    Lambda_4 = fft(D_phi*x)
    
    return (x_0, phi, psi, D_phi, D_psi, Lambda_1, Lambda_2, Lambda_3, Lambda_4)
    
    
def toeplitz_inverse_multiplication(u, x_0, phi, psi, D_phi, D_psi, Lambda_1, Lambda_2, Lambda_3, Lambda_4):
    """
    Compute
    y = inv(T) u
    
    Where T is a symmetrix Toeplitz matrix. Requires preprocessing with toeplitz_inverse_multiplication_prep()
    
    See Gohberg, I. and V. Olshevsky (1994)
    
    parameters:
        u the input vector
        many things as the output of toeplitz_inverse_multiplication_prep()
    
    returns:
        y the output vector
    """

    y = fft(D_phi*u)
    a = Lambda_1*fft(D_psi*(1/D_phi)*ifft(Lambda_2*y))
    b = Lambda_3*fft(D_psi*(1/D_phi)*ifft(Lambda_4*y))
    y = (1/D_psi)*real(ifft(a-b))/(x_0*(phi-psi))
    
    return y



def bd_toeplitz_slogdet(*arrs):
    """
    Determinant of a block-diagonal matrix having Toeplitz blocks
    
    parameters:
        *arrs a tuple of input parameters of toeplitz_slogdet()
    
    returns:
        sign sign of the determinant
        logdet natural log of the determinant 
    """
    
    sign = 1 
    logdet = 0
    for c in arrs: # loop over each block
        (t_sign, t_logdet) = toeplitz_slogdet(c)
        sign *= t_sign
        logdet += t_logdet
    
    return (sign, logdet)
    
    

def bd_toeplitz_inverse_multiplication_prep(*arrs):
    """
    Preprocessing for block diagonal matrices analogous to toeplitz_inverse_multiplication_prep()
    parameters:
        *arrs a tuple of input parameters of toeplitz_inverse_multiplication_prep()
        
    returns:
        a tuple of output parameters of toeplitz_inverse_multiplication_prep()     
    """
    
    t = []
    for c in arrs: # loop over each block
        t.append(toeplitz_inverse_multiplication_prep(c))
    return tuple(t)
    
def bd_toeplitz_inverse_multiplication(u, *arrs):
    """
    matrix multiplication with the inverse of a block-diagonal matrix having 
    Toeplitz blocks.
    y = T u    
    Analogous to toeplitz_inverse_multiplication()
    
    parameters:
        u the input vector
        *arrs a tuple of input parameters of toeplitz_inverse_multiplication()
        
    returns:
        y the output vector
    """
    
    y = zeros(shape(u))
    n_start = 0
    n_end = 0
    for t in arrs:
        n_start = n_end
        n_end += len(t[3]) # len(t[3]) is the length of the block
        y[n_start:n_end] = toeplitz_inverse_multiplication(u[n_start:n_end], *t)
    assert len(y) == n_end
    return y


if __name__ == "__main__":
    """
    This function tests all the routines of this file.
    
    All is fine when the largest differences (max diff) are small. See console output.
    """

    
    n = 500            # size of the matrices used in code testing
    N_LIMIT = 2000     # do not instantiate in memory matrices larger than this size
    c = randn(n)       # random first column of Toeplitz matrix
    r = randn(n)       # random first row of Toeplitz matrix
    u = randn(n)       # random input test vector
    
    
    # solve a Toeplitz linear system
    if n < N_LIMIT:
    
        T = toeplitz(r)    # a symmeric Toeplitz matrix
        b = randn(n)
        
        x1 = solve(T, b)
        x2 = levinson(r, b)
    
        assert allclose(x1, x2)
        print("Levinson algorithm, max diff {:.2e}".format(np.max(np.abs(x1-x2))))
        print("")
    
    
    # compute determinant of Toeplitz matrix
    if n < N_LIMIT:
        
        r = randn(n)
                
        T = toeplitz(r)    # size n
        (s1, logdet1) = slogdet(T)
        
        (s2, logdet2) = toeplitz_slogdet(r)
        
        print("Toeplitz matrix determinant, max diff {:.2e}".format(np.max(np.abs(logdet1-logdet2))))
        print("")
        assert allclose(logdet1, logdet2)
        assert allclose(s1, s2)
      
    
    # Multiplication with a circulant matrix
    if n < N_LIMIT:
        C = np.matrix( circulant(r) )
        
        y1 = dot(C,u)
        y2 = circulant_multiplication(u, r)
        
        assert allclose(y1, y2)
        print("Circulant matrix multiplication, max diff {:.2e}".format(np.max(np.abs(y1-y2))))
        print("")
    
    # Multiplication with a Toeplitz matrix
    if n < N_LIMIT:
        
        C = np.matrix( toeplitz(c,r) )
        
        y1 = dot(C, u)
        y2 = toeplitz_multiplication(u, c, r)
        
        assert allclose(y1, y2)
        print("Toeplitz matrix multiplication, max diff {:.2e}".format(np.max(np.abs(y1-y2))))
        print("")
    
    
    # Multiplication with a factor-circulant matrix
    if n < N_LIMIT:
        phi=3
        C = np.matrix( factor_circulant_matrix(r,phi) )
        
        y1 = dot(C,u)
        y2 = factor_circulant_multiplication(u, r,phi)
        
        assert allclose(y1, y2)
        print("Factor-circulant matrix multiplication, max diff {:.2e}".format(np.max(np.abs(y1-y2))))
        print("")
    
    
    ### Matrix-vector multiplication for inverse of Toeplitz matrix with arbitrary vector
    ### Using matrix factorizatoin
    
    
    
    
    # naive inverse computation
    if n < N_LIMIT:
        start_time = time.time()
        
        T = toeplitz(r)
        Ti_0 = inv(T) 
        
        elapsed_time = time.time() - start_time
        print("Naive inversion in {:.3} s".format(elapsed_time))
    
    # first approach to get inverse
    if n < N_LIMIT:
        start_time = time.time()
        
        # preprocessing
        x = levinson(r, np.concatenate( (np.array([1]), np.zeros((n-1,))) ) )
        y = levinson(r, np.concatenate( (np.zeros((n-1,)), np.array([1])) ) )
    
        X = toeplitz(x, np.concatenate( ([x[0]], x[-1:0:-1]) ))
        Xl = tril(X,0)
        Xu = triu(X,1)
        
        Y = toeplitz(np.concatenate( ([y[-1]], y[0:-1]) ), y[::-1] )
        Yl = tril(Y,-1)
        Yu = triu(Y,0)
        
        Ti_1 = (1/x[0])*(np.dot(Xl, Yu) - np.dot(Yl, Xu))
        
        elapsed_time = time.time() - start_time
        print("First approach in {:.3} s".format(elapsed_time))
    
    # second approach to get inverse
    if n < N_LIMIT:
        start_time = time.time()
        
        # preprocessing
        x = levinson(r, np.concatenate( (np.array([1]), np.zeros((n-1,))) ) )
        y = levinson(r, np.concatenate( (np.zeros((n-1,)), np.array([1])) ) )
        
        
        phi=2
        psi=3
        
        Zphi = np.eye(n,n, -1)
        Zphi[0,-1] = phi
        Zpsi = np.eye(n,n, -1)
        Zpsi[0,-1] = psi
        
        Ti_2 = 1/(x[0]*(phi-psi))*(dot(factor_circulant_matrix(x,psi), factor_circulant_matrix(dot(Zphi,y),phi)) \
                                - dot(factor_circulant_matrix(dot(Zpsi,y),psi), factor_circulant_matrix(x,phi)))
                                
        elapsed_time = time.time() - start_time
        print("Second approach in {:.3} s".format(elapsed_time))
                                
                                
                                
    # direct matrix-vector multiplication
    start_time = time.time()
    
    # preprocessing
    
    (x_0, phi, psi, D_phi, D_psi, Lambda_1, Lambda_2, Lambda_3, Lambda_4) = toeplitz_inverse_multiplication_prep(r)
    
    elapsed_time = time.time() - start_time
    print("Direct matrix-vector multiplication (preprocessing) in {:.3} s".format(elapsed_time))
    
    # multiplication
    y3 = toeplitz_inverse_multiplication(u, x_0, phi, psi, D_phi, D_psi, Lambda_1, Lambda_2, Lambda_3, Lambda_4)
    
    elapsed_time = time.time() - start_time
    print("Direct matrix-vector multiplication (total) in {:.3} s".format(elapsed_time))
    print("")
    
    # Test Toeplitz multiplication and inverse multiplication
    u1 = toeplitz_inverse_multiplication(toeplitz_multiplication(u, r), x_0, phi, psi, D_phi, D_psi, Lambda_1, Lambda_2, Lambda_3, Lambda_4)
    assert allclose(u, u1)
    
    
    if n < N_LIMIT:
        
        y0 = dot(Ti_0, u)
        y1 = dot(Ti_1, u)
        y2 = dot(Ti_2, u)
        
    
        print("\tToeplitz inverse multiplication, max diff {:.2e}".format(np.max(np.abs(y0-y3))))
        print("\tToeplitz inverse multiplication, max diff {:.2e}".format(np.max(np.abs(y2-y3))))
        print("\tToeplitz inverse multiplication, max diff {:.2e}".format(np.max(np.abs(y2-y3))))
        print("")
     
    n_vec = [16,22,50]
    n = sum(n_vec)
    c = (randn(n_vec[0]), randn(n_vec[1]), randn(n_vec[2]))  # first column of Toeplitz matrix
    u = randn(n)                                 # random input test vector
    
    A = []
    for nn in range(0, len(n_vec)):
        A.append(toeplitz(c[nn]))
    
    T = block_diag(*tuple(A))
    
    
    
        
    
    (s1, logdet1) = slogdet(T)
    (s2, logdet2) = bd_toeplitz_slogdet(*c)
    
    print("Block-diagonal Toeplitz matrix determinant, max diff {:.2e}".format(np.max(np.abs(logdet1-logdet2))))
    print("")
    assert allclose(logdet1, logdet2)
    assert allclose(s1, s2)
    
    
    
    
    t = bd_toeplitz_inverse_multiplication_prep(*c)
    y0 = bd_toeplitz_inverse_multiplication(u, *t)
    y1 = dot(inv(T) , u)
    print("Block-diagonal Toeplitz inverse multiplication, max diff {:.2e}".format(np.max(np.abs(y0-y1))))
    print("")
    
    assert allclose(y0, y1)

    

