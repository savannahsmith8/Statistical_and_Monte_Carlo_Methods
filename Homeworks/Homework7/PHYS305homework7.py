# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:21:56 2024

@author: savan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# PROBLEM 1

# calculates the electric potential inside a given domain
def Laplace(H, W, Nx, Ny, VL, VR, VT, VB):
    
    # make grid
    x, dx = np.linspace(0, W, Nx, retstep = 'True')
    y, dy = np.linspace(0, H, Ny, retstep = 'True')
    
    X,Y = np.meshgrid(x,y)
    
    # total number of gridpoints where the solution will be computed
    unknowns = Nx * Ny
    
    # create the link matrix to number the nodes
    link = np.array([i for i in range(unknowns)])
    link = np.reshape(link,[Ny,Nx]) # turns list into matrix same size as grid
    
    # define the laplacian matrix
    lap = np.zeros((unknowns,unknowns))
    
    lap[link,link] += - 2 / dx ** 2 - 2 / dy ** 2
    
    np.roll(link, 1, axis = 1)
    
    # left neighbor
    lap[link, np.roll(link, 1, axis = 1)] += 1 / dx ** 2
    # right neighbor
    lap[link, np.roll(link, -1, axis = 1)] += 1 / dx ** 2
    
    # upper neighbor
    lap[link, np.roll(link, 1, axis = 0)] += 1 / dy ** 2
    # lower neighbor
    lap[link, np.roll(link, -1, axis = 0)] += 1 / dy ** 2
    
    # set a dirichlet boundary condition on the top rows
    lap[link[0,:],:] = 0
    
    # put a one on the diagonal of these rows
    lap[link[0,:],link[0,:]] = 1 
    
    # set a dirichlet BC on bottom rows
    lap[link[Ny-1,:],:] = 0
    lap[link[Ny-1,:],link[Ny-1,:]] = 1
    
    # set a dirichlet BC on the left rows
    lap[link[:,0],:] = 0
    lap[link[:,0],link[:,0]] = 1
    
    # set a dirichlet BC on the right rows
    lap[link[:,Nx-1],:] = 0
    lap[link[:,Nx-1],link[:,Nx-1]] = 1
    
    # initialize source vector
    b = np.zeros(unknowns)
    for i in range(Nx):
        for j in range(Ny):
            b[link[j,i]] = np.sin(2 * np.pi * x[i]) * ((np.sin(np.pi * y[j])) ** 2)
              
    # set BC on source vector using given charge density
    b[link[0,:]] = VT
    b[link[Ny-1,:]] = VB
    b[link[:,0]] = VL
    b[link[:,Nx-1]] = VR
    
    # solve the equation
    phi = np.linalg.solve(lap,b)
    
    # reshape so that phi has the same shape as grid
    phi = np.reshape(phi,[Nx,Ny])
    
    # plot the solution
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(X, Y, phi, cmap = cm.jet)
    
    plt.show()


# PROBLEM 2

# calculates the best fit cubic line by analyzing the least squares using linear algebra
def FitDataToCubic(datafile):
    
    # opening the datafile in read mode and intializing data list
    file = open(datafile,'r')
    data = []
    
    # appending every line in the datafile to the data list
    for line in file:
        data.append(line.split())
    
    # convert data list into a numpy array
    data = np.array(data,dtype='float')
    # determining the number of variables and number of data points 
    N,d = data.shape
    
    # letting x be the first column of numbers in data
    x = data[:,0]
    
    # initializing a 4x4 matrix for chi squared calculation
    M = np.zeros((4,4))
    # using the cubic equation for least sqaures to form the 4x4 matrix
    M[:,0] = [ N, np.sum(x), np.sum(x ** 2), np.sum(x ** 3) ]
    M[:,1] = [ np.sum(x), np.sum(x ** 2), np.sum(x ** 3), np.sum(x ** 4) ]
    M[:,2] = [ np.sum(x ** 2), np.sum(x ** 3), np.sum(x ** 4), np.sum(x ** 5) ]
    M[:,3] = [ np.sum(x ** 3), np.sum(x ** 4), np.sum(x ** 5), np.sum(x ** 6) ]
    
    # initializing the solution vector
    b = np.zeros((4,1))
    # determining the solution vector using the x and y data values
    b[:,0] = [ np.sum(data[:,1]), np.sum(x * data[:,1]), np.sum(x ** 2 * \
            data[:,1]), np.sum(x ** 3 * data[:,1]) ]

    # using np.linalg to solve for the a vector (vector multiplied by the matrix)
    a = np.linalg.solve(M,b)

    # defining the equation of the cubic line using the a vector
    y = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3

    plt.title("Problem2_data with best fit cubic line")
    plt.xlabel("x")
    plt.ylabel("y")
    # plotting the data
    plt.plot(data[:,0],data[:,1],'c', label = 'data')
    # plotting the best fit cubic line
    plt.plot(x,y,'b', label = 'best fit line')
    plt.legend()
    plt.show()
    
    return a


# PROBLEM 3

# WRITE OUT LAST SLIDES FROM PART 19
# CONVERT MATRIX TO CODE THEN USE LINALG
# HOW TO GET BEST FIT????
# GRAPH OF LAMBDA VALUES????

def BestFitWithLambda(datafile, lam):
    
    # opening file to read data and initializing data list
    file = open(datafile,'r')
    data = []
    
    # looping through the lines in the file and splitting them
    for i in file:
        data.append(i.split())

    # creating a numpy array from the data list
    data = np.array(data, dtype='float')
    N,d = data.shape
    # define grid
    x, dx = np.linspace(data[0,0], data[-1,0], N, retstep = 'True')
    
    # create the link matrix to number the nodes
    link = np.array([i for i in range(N)])
    
    # initialize matrix
    M = np.zeros((N,N))
    
    # determine diagonals of matrix M using link
    M[link,link] += - 1 - ((2 * lam) / (dx ** 2))
    # left neighbor
    M[link, np.roll(link, 1)] += lam / dx ** 2
    # right neighbor
    M[link, np.roll(link, -1)] += lam / dx ** 2
    
    # set natural boundary conditions because the solutions are unknown
    M[0,:] = 0
    M[-1,:] = 0
    
    # using forward euler for the first two nodes and the last two nodes
    M[0,0] = - 1 / dx
    M[0,1] = 1 / dx
    M[N-1,N-2] = - 1 / dx
    M[N-1,N-1] = 1 / dx
    
    # intialize solution vector
    b = np.zeros(N)
    
    # assign values to solution vector
    b[:] = - data[:,1]
    b[0] = 0
    b[-1] = 0
    
    a = np.linalg.solve(M,b)
    
    return x, data, a


# PROBLEM 4

# calculates the power spectrum of a given datafile
def PowerSpectrum(datafile):
    
    # opening file to read data and initializing data list
    file = open(datafile,'r')
    data = []

    # looping through the lines in the file and splitting them
    for i in file:
        data.append(i.split())
        
    # creating a numpy array from the data list
    data = np.array(data, dtype='float')
    
    # GRAPH 1: plt.plot(data[:,0], data[:,1])

    # performing the fast fourier transform algorithm
    fFFT = np.fft.fft(data[:,1])
    
    # GRAPH 2: plt.plot(fFFT)

    # to find frequency:
    
    # determining the number of data points from the shape of the data array
    N = data.shape[0]
    
    # using fourier transform
    freq = [ 2 * np.pi * i / (data[-1,0] - data[0,0]) for i in range(N) ]

    # GRAPH 3: plt.plot(freq[0:round(N/2)],fFFT[0:round(N/2)])
    
    # determining the power of each mode by calculating the power spectrum
    pow = np.conj(fFFT) * fFFT
    
    # GRAPH 4: plt.plot(freq[0:round(N/2)],pow[0:round(N/2)])

    # making the frequency a numpy array
    freq = np.array(freq)

    # masking the frequency so that the frequencies are less than 5
    mask = (freq < 5)
    
    # GRAPH 5: the masked frequency versus the masked power spectrum with mask 10000
    
    # plotting the masked frequency versus the masked power spectrum with mask 5 (graph 6)
    plt.figure()
    plt.title("power spectrum of Problem4_data (mask = 5)")
    plt.xlabel("masked frequency")
    plt.ylabel("masked power spectrum")
    plt.plot(freq[mask],pow[mask],'m')
    plt.show()


def main():
    
    # PROBLEM 1
    
    #Laplace(2, 4, 100, 100, 0, 0, 0, 0)
    
    
    # PROBLEM 2
    
    #FitDataToCubic('Problem2_data.txt')
    
    
    # PROBLEM 3
    
    lam_values = [0.01, 0.1, 1, 10]
    
    for lam in lam_values:
        
        x, data, a = BestFitWithLambda('Problem2_data.txt', lam)
        
        plt.plot(x, a, label = "λ = lam")
        
    plt.title("Problem2_data with different lambda values")
    plt.xlabel("x data")
    plt.ylabel("function")
    plt.plot(x, data, 'k--', label = "data")
    plt.legend()
    plt.show()
    
    
    # PROBLEM 4
    
    #PowerSpectrum('Problem4_data.txt')
    
    
    
if __name__ == "__main__":
    main()