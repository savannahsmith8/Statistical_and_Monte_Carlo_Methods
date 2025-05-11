# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:11:00 2024

@author: Savan
"""

import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt


# PROBLEM 1

# alpha, beta, and eta are constants and r = |r1 - r2|

# defining all equations of motion for the self-propelled mass m
# v0 = constant "swimming" speed
# d is a unit vector that points along the object's direction of motion
# v = v0 * d
# problem must be in 3D because we have a cross product of two vectors

# hard-coding constants
alpha = 1 * 10 ** (-3)
beta = 10
eta = 1 * 10 ** (-4)
m = 0.001
v0 = 5 * 10 ** (-4)

# defining the ODEs for the motion of the mass

def dv1dt(d1, v1, v2, r):
    return (1/m) * eta * (v0 * d1 - v1) + (alpha/r) * (v2 - v1)

def dd1dt(d1, v1, d2, v2, r):
    return np.cross((beta/(r ** 2)) * np.cross(d1, d2), d1)
    
def dv2dt(d2, v1, v2, r):
    return (1/m) * eta * (v0 * d2 - v2) + (alpha/r) * (v1 - v2)
    
def dd2dt(d1, v1, d2, v2, r):
    return np.cross((beta/(r ** 2)) * np.cross(d2, d1), d2)

def entrained_motion(num, T):
    
    # initializing all arrays
    
    d1 = np.zeros((num, 3), 'float')
    r1 = np.zeros((num, 3), 'float')
    v1 = np.zeros((num, 3), 'float')
    d2 = np.zeros((num, 3), 'float')
    r2 = np.zeros((num, 3), 'float')
    v2 = np.zeros((num, 3), 'float')
    
    time, dt = np.linspace(0, T, num, 'retstep', 'true', 'float')
    
    # we know v will be dependent on r and d
    
    # changed numbers to investigate graph behavior
    d1[0] = np.array([2, 1, 5]) 
    d2[0] = np.array([1, 6, 1])
    r1[0] = np.array([-0.0001, 0.0001, 0.0001])
    r2[0] = np.array([0.0001, -0.0001, 0.0001])
    v1[0] = v0 * d1[0] 
    v2[0] = -v0 * d2[0]
    
    for i in range(num - 1):
        
        # distance = velocity * time
        r1[i + 1] = r1[i] + v1[i] * dt
        r2[i + 1] = r2[i] + v2[i] * dt
        
        # want the magnitude of the vectors so we need r
        r = np.linalg.norm(r1[i] - r2[i])
        
        # using forward euler to solve d ODEs
        d1[i + 1] = d1[i] + dt * dd1dt(d1[i], v1[i], d2[i], v2[i], r)
        d2[i + 1] = d2[i] + dt * dd2dt(d1[i], v1[i], d2[i], v2[i], r)
        
        # need the magnitude of the d vectors
        d1[i + 1] = d1[i + 1] / np.linalg.norm(d1[i + 1])
        d2[i + 1] = d2[i + 1] / np.linalg.norm(d2[i + 1])
        
        # using forward euler to solve for v ODEs
        v1[i + 1] = v1[i] + dt * dv1dt(d1[i], v1[i], v2[i], r)
        v2[i + 1] = v2[i] + dt * dv2dt(d2[i], v1[i], v2[i], r)
   
    return d1, d2, r1, r2, v1, v2
   


# PROBLEM 2

# computes a two-dimensional random walk of step size b with N steps
def random_walk_2D(b, N):
    
    # initializing x, y, and r arrays
    x = [ 0 for i in range(N) ]
    y = [ 0 for i in range(N) ]
    r = [ 0 for i in range(N) ]
    
    # looping through the steps for randoms walks
    for i in range(1, N):
        
        xflip = rand(1)
        
        # moving the x position of the particle depending on the random number
        if xflip < 0.5:
            x[i] = x[i - 1] - b
        else:
            x[i] = x[i - 1] + b
            
        yflip = rand(1)
        
        # moving the y position of the particle depending on the random number
        if yflip < 0.5:
            y[i] = y[i - 1] - b
        else:
            y[i] = y[i - 1] + b
            
        # calculating the distance from the point using pythagorean theorem
        r[i] = np.sqrt( (x[i] ** 2) + (y[i] ** 2) )
            
    # intializing arrays for avergae displacement and mean squared displacement        
    
    average_disp = [ 0 for k in range(N) ]
    MSD = [ 0 for k in range(N) ]
    
    for k in range(N):
        
        for l in range(N - k):
            
            # calculating avg displacement and MSD
            average_disp[k] += ( r[k + l] - r[k] / (N - k) )
            MSD[k] += ( r[k + l] - r[k] ) ** 2 / (N - k)
                
    return x, y, average_disp, MSD



# PROBLEM 3

# simulates the motion of a brownian particle with radius R through water
def langevin_2D(R, T, dt):
    
    m = 4 * np.pi * (R ** 3) / 3    # particle mass
    kBT = 4 * 10 ** (-14)           # thermal energy
    eta = 0.01                      # viscosity of water
    zeta = 6 * np.pi * eta * R      # drag coefficient
    c = zeta * np.sqrt(kBT/m)       # magnitude of stochastic force
    
    steps = round(T/dt)
    
    # initializing the arrays for position and velocity
    
    x = np.zeros((2, steps), 'float')
    v = np.zeros((2, steps), 'float')
    
    time = [ dt * i for i in range(steps) ]
    
    MSD = np.zeros((steps, 1), 'float')
    
    # looping through the particle's movement in time
    for i in range(1, steps):
        
        # adjusting x from the stochastic force using random number
        xi = c * randn(2)
        
        # calculating the velocities from the x position and the drag coefficient
        v[:,i] = ( (1 - zeta * dt / 2 / m) * v[:, i - 1] + dt * xi / m ) / ( 1 + zeta * dt / 2 / m )
        # finding the average of the velocities at i and i-1
        vmid = 0.5 * ( v[:,i] + v[:, i - 1] )
        
        # adjusting the next posiiton using the mid velocity
        x[:,i] = x[:, i - 1] + dt * vmid
    
    # plotting the position of the particle
    plt.title("position of Brownian particle")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(x[0,:], x[1,:], 'b')
        
    # looping through the steps to calculate MSD
    for j in range(steps):
            
        for k in range(steps-j):
                
            MSD[j] += ( ( x[0, j + k] - x[0, k] ) ** 2 + ( x[1, j + k] - x[1, k] ) ** 2 ) / (steps - j)
    
    return x, time, MSD



def main():
    
    
    # PROBLEM 1
    
    print(entrained_motion(10000, 100))
    
    # plotting
    r1 = entrained_motion(10000, 100)[2]
    r2 = entrained_motion(10000, 100)[3]
    v1 = entrained_motion(10000, 100)[4]
    v2 = entrained_motion(10000, 100)[5]
    
    # plotting the r arrays
    plt.title("self-propelled motion for an object of mass m")
    plt.plot(r1[:,0], r1[:,1], 'm', label = "r1")
    plt.plot(r2[:,0], r2[:,1], 'y', label = "r2")
    plt.legend()
    
    # plotting the velocity arrays
    fig2,ax2 = plt.subplots()
    ax2.set_title("self-propelled velocity for an object of mass m")
    ax2.plot(v1[:,0], v1[:,1], 'g', label = "v1")
    ax2.plot(v2[:,0], v2[:,1], 'b', label = "v2")
    ax2.legend()
    
    
    # PROBLEM 2
    
    print(random_walk_2D(10, 1000))
    
    # assigning variables to the returned tuple
    x = random_walk_2D(10, 1000)[0]
    y = random_walk_2D(10, 1000)[1]
    average_disp = random_walk_2D(10, 1000)[2]
    MSD = random_walk_2D(10, 1000)[3]
    
    plt.title("2D Random Walk")
    plt.xlabel("x position")
    plt.ylabel("y position")
    # plotting start and end point to better visualize walk
    plt.plot(x[0], y[0], 'mo', label = "start")
    plt.plot(x[-1], y[-1], 'yo', label = "end")
    plt.plot(x, y)
    plt.legend()
    
    # plotting the subplots
    
    fig2,ax2 = plt.subplots()
    ax2.plot(MSD)
    ax2.set_title("MSD as a function of the number of steps")
    ax2.set_xlabel("number of steps")
    ax2.set_ylabel("MSD")
    
    fig3,ax3 = plt.subplots()
    ax3.plot(average_disp)
    ax3.set_title("average displacement as a function of the number of steps")
    ax3.set_xlabel("number of steps")
    ax3.set_ylabel("avg displacement")
    
    
    # PROBLEM 3
    
    print(langevin_2D(0.000002, 0.001, (10 ** (-9))))
    
    time = langevin_2D(0.000002, 0.000001, (10 ** (-9)))[1]
    MSD = langevin_2D(0.000002, 0.000001, (10 ** (-9)))[2]
    
    fig2,ax2 = plt.subplots()
    ax2.plot(time, MSD)
    ax2.set_title("MSD as a function of time")
    ax2.set_xlabel("time")
    ax2.set_ylabel("MSD")
    

    
if __name__ == "__main__":
    main()