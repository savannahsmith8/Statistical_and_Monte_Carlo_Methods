# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:30:14 2024

@author: savan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


# PROBLEM 1

# steady state solution means that it does not change over time
# derivative of the function WRT time will be zero
# dC/dt = d2C/dx2 - d/dx(sin(x) * C) = 0
# first partial derivative represents advection and second partial derivative represents diffusion

# solves the second order PDE involving both diffusion and advection
def DiffusionAndAdvection(TotalTime, dt, skip):
    
    steps = round(TotalTime/dt/skip)
    
    # define grid
    L = 4 * np.pi
    N = 100
    x,dx = np.linspace(0,L,N,retstep='True',dtype='float')
    time = [ skip * dt for i in range(N) ]

    # initialize concentration matrix
    C = np.zeros((N, steps))
    C[:,0] = 1
      
    # initialize derivatives
    dC = np.zeros(N)
    dCd2 = np.zeros(N)
    
    # defining intervals where the velocity is either positive or negative
    # because v = sin(x):
        # velocity is positive from 0 to pi and 2pi to 3pi so we will use backward
        # velocity is negative from pi to 2pi and 3pi to 4pi so we will use forward
    # these values are integers because N is an integer
    I1 = int(np.round(np.pi * N / L))
    I2 = int(np.round(2 * np.pi * N / L))
    I3 = int(np.round(3 * np.pi * N / L))
    I4 = int(np.round(4 * np.pi * N / L) - 1)
    
    # defining the velocity 
    # term that is multiplied by the first partial spatial derivative
    v = np.sin(x)
    
    #set up movie
    metadata = dict(title = "DiffusionAndAdvection", artist = "matplotlib")
    writer = FFMpegWriter(fps = 25, codec = 'h264', bitrate = 3000, metadata = metadata)
    
    fig1 = plt.figure()
    data, = plt.plot([],[], 'm')
    plt.ylim(0, 2)
    plt.xlim(0, 4 * np.pi)
    plt.xlabel('x')
    plt.ylabel('concentration')
    
    with writer.saving(fig1, 'DiffusionAndAdvection.mp4',100):

        # begin time stepping
        for i in range(1,steps):
            
            C[:,i] = C[:,i-1]
            
            for j in range(skip):
                
                # define second derivative (diffusion)
                dCd2[0] = 2 * (C[1,i] - C[0,i]) / (dx ** 2)
                dCd2[1:N-1] = (C[2:N,i] - 2 * C[1:N-1,i] + C[:N-2,i]) / (dx ** 2)
                dCd2[N-1] = 2 * (C[N-2,i] - C[N-1,i]) / (dx ** 2)
            
                # evaulating the first derivative (advection)
            
                # evaluating from 0 to I1 (positive v)
                dC[0] = (C[0, i-1] - C[N-1, i-1])/dx
                dC[1:I1] = (C[1:I1, i-1] - C[:I1-1, i-1]) / dx
            
                # evaluating from I1 to I2 (negative v)
                dC[I1:I2] = (C[I1+1:I2+1, i] - C[I1:I2, i]) / dx
            
                # evaluating from I2 to I3 (positive v)
                dC[I2:I3] = (C[I2:I3, i] - C[I2-1:I3-1, i]) / dx
                
                # evaluating from I3 to I4 (negative v)
                dC[I3:I4-1] = (C[I3+1:I4, i] - C[I3:I4-1, i]) / dx
                dC[I4] = (C[0, i-1] - C[N-1, i-1]) / dx
            
            # time stepping concentration
            C[:,i] = C[:,i] + dt * ( dCd2 - v * dC - C[:,i] * np.cos(x))
        
            #plot movie
            data.set_data(x, C[:,i])
            writer.grab_frame()
            plt.pause(0.02)
            
        return time, C


# PROBLEM 2

# (see write-up for written work)

# perform Gauss-Jordan elimination to solve Ax = b
def GJ2(A,b):
    
    # determine the shape of A
    W,L = A.shape
    
    # define matrix B so that A does not get overwritten
    B = np.array(A, dtype='float')
    
    # intitialize array for solution x
    x = np.array(b, dtype='float')
    
    vals = np.array([k for k in range(W)])
    
    # begin iteration
    for j in range(W):
        
        # check if diagonal element is zero, pivot if it is
        if B[j,j] == 0:
            
            # in the jth column, find and number points that are nonzero
            nonzero = [k for k, e in enumerate(B[j:,j]) if e != 0]
            
            # grab the number of the first nonzero element
            val = nonzero[0]
            
            # extract the current row and dirst row with nonzero diagonal
            b1 = np.copy(B[j + val, :])
            b2 = np.copy(B[j,:])
            
            # swap these rows
            B[j,:] = b1
            B[j + val, :] = b2
            
            # get current row and first row with nonzero diagonal
            c1 = x[j + val]
            c2 = x[j]
            
            # swap these rows
            x[j] = c1
            x[j + val] = c2
            
        # divide the jth row of B by the diagonal element, likewise for q
        norm = B[j,j]
        B[j,:] = B[j,:] / norm
        x[j] = x[j] / norm
        
        # subtract correct multiple of jth row to get zero of the jth
        # column of the ith row
        # use mask to select only entries where i != j and Bij != 0
        for i in vals[(vals != j) & (B[:,j] != 0)]:
            
            norm = B[i,j]
            B[i,:] = B[i,:] - norm * B[j,:]
            x[i] = x[i] - norm * x[j]
            
    return x


def main():
    
    
    # PROBLEM 1
    
    time, C = DiffusionAndAdvection(1, 0.001, 1)
    
    plt.title("steady-state solution profile")
    plt.xlabel("time")
    plt.ylabel("concentration")
    plt.plot(time, C, 'm')
    plt.show()
    
    
    # PROBLEM 2
    
    # defining matrix A
    A = np.zeros((5,5))
    A[0,:] = [-1, 1, 0, 0, 0]
    A[1,:] = [1, -3, 1, 0, 0]
    A[2,:] = [0, 1, -3, 1, 0]
    A[3,:] = [0, 0, 1, -3, 1]
    A[4,:] = [0, 0, 0, -1, 1]
    
    # defining solution matrix
    b = np.array([-1, 0, 0, 0, 1])
    
    print("the solution array for the system of equations is", GJ2(A, b))
    
    x1 = GJ2(A, b)[0]
    x2 = GJ2(A, b)[1]
    x3 = GJ2(A, b)[2]
    
    print("x1 = x5 =", x1)
    print("x2 = x4 =", x2)
    print("x3 =", x3)
    
    
if __name__ == "__main__":
    main()
    
    