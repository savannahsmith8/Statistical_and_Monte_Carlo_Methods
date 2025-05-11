# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:24:03 2024

@author: savan
"""

# PROBLEM 1 (see write-up)

# epsilon = 10 * kBT ~ 4e-20
# sigma = R = 10 nm = 1e-8


# PROBLEM 2 (see write-up)

# dt = 2.36e-8 ~ 2e-8


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


# PROBLEM 3

# simulates the motion of N particles due to the Lennard-Jones potential
def brownian_particles(N, R, L, TotalTime, dt, skip, e, sigma):
    
    kBT = 0.004                             # thermal energy (pN / um)
    eta = 0.001                             # viscosity of water (kg / m * s)
    zeta = 6 * np.pi * eta * R              # drag coefficient
    c = np.sqrt((2 * kBT * zeta) / dt)      # magnitude of stochastic force
    
    steps = round(TotalTime/dt/skip)
    
    # initializing arrays to store position and time
    x = np.zeros((2, N, steps))
    time = [ skip * dt * i for i in range(steps) ]
    
    # randomly scattering the particles throughout the box
    x[:,:,0] = L * np.random.rand(2,N)

    # initialing array to store particle interactions
    Fi = np.zeros((2,N))
    
    for i in range(1,steps):
        
        x[:,:,i] = x[:,:,i-1]
        
        for j in range(skip):
            
            # calculating the random force
            xi = c * np.random.rand(2,N)
            
            # setting matrices for the difference between x and y positions
            
            Dx = np.meshgrid(x[0,:,i])
            Dx = Dx - np.transpose(Dx)
            
            Dy = np.meshgrid(x[1,:,i])
            Dy = Dy - np.transpose(Dy)
            
            # setting periodic conditions on particle distances
            Dx[Dx < -L/2] = Dx[Dx < -L/2] + L
            Dx[Dx > L/2] = Dx[Dx > L/2] - L
            
            Dy[Dy < -L/2] = Dy[Dy < -L/2] + L
            Dy[Dy > L/2] = Dy[Dy > L/2] - L
            
            r = np.sqrt((Dx ** 2) + (Dy ** 2))
            
            for k in range(N):
                r[k,k] = 10
                
            # magnitude of the force for the Lennard-Jones potential
            FiMag = ((12 * e) / (sigma ** 2)) * (2 * ((sigma/r) ** 14) - ((sigma/r) ** 8))

            # setting a limit on the force moving the particle
            FiMag[ FiMag > 0.01 * zeta ] = 0.01 * zeta

            # x component of the force
            Fi[0,:] = - np.sum(FiMag * Dx, axis = 1)
            # y component of the force
            Fi[1,:] = - np.sum(FiMag * Dy, axis = 1)
            
            # timestepping the particles position
            x[:,:,i] = x[:,:,i] + ((dt / zeta) * (xi + Fi))

            # setting periodic boundary conditions so the number of 
            # particles inside the box remains constant
            Mask = x[0,:,i] > L
            x[0,Mask,i] = x[0,Mask,i] - L
            
            Mask = x[0,:,i] < 0
            x[0,Mask,i] = x[0,Mask,i] + L
            
            Mask = x[1,:,i] > L
            x[1,Mask,i] = x[1,Mask,i] - L
    
            Mask = x[1,:,i] < 0
            x[1,Mask,i] = x[1,Mask,i] + L 
    
    return x, time
         

# PROBLEM 4

# calculates the speed of a wave as a function of D
def fishers_equation(D, TotalTime, dt, skip):
    
    # number of time steps
    steps = round(TotalTime/dt/skip)
    
    # defining the grid
    N = 100
    x,dx = np.linspace(0, 10, N, retstep='True', dtype='float')
    
    time = [skip * dt * i for i in range(steps) ]
    
    # initializng u matrix
    u = np.zeros((N,steps))     # N for temporal position, steps for spacial position
    u[0,:] = 1 
    u[N-1,:] = 0
    # setting initial condition for u
    u[:,0] = 0.5 * ( 1 - np.tanh(x/0.1) )
    
    # initializing second derivative array
    dud2 = np.zeros(N)
    
    metadata = dict(title = "Fisher's Equation", artist = 'Matplotlib')
    writer = FFMpegWriter(fps = 15, codec = 'h264', metadata = metadata)
    
    fig1 = plt.figure()
    data, = plt.plot([],[],'m')
    
    plt.title(f"particle position (D={D})")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.xlim(x[0],x[-1])
    plt.ylim(0,1)
    
    with writer.saving(fig1,'FishersEquation.mp4',100):
    
        # begin time stepping
        for i in range(1,steps):
            
            u[:,i] = u[:,i-1]
            
            for j in range(skip):
                
                # boundary condition at x = 0  
                dud2[0] = 2 * (u[1,i] - u[0,i]) / (dx ** 2)
                
                # using the central difference derivative
                dud2[1:N-1] = (u[2:N,i] - 2 * u[1:N-1,i] + u[:N-2,i]) / (dx ** 2)
                
                # boundary condition at x = N - 1
                dud2[N-1] = 2 * (u[N-2,i] - u[N-1,i]) / (dx ** 2)
                
                # time stepping u based on Fisher's equation
                u[:,i] = u[:,i] + dt * D * dud2 + u[:,i] * dt * (1 - u[:,i])
                u[0,i] = 1 
                u[N-1,i] = 0
                
            data.set_data(x,u[:,i])
            writer.grab_frame()
            plt.pause(0.02)
        
        cmap = plt.get_cmap('PiYG')
        
        fig2,ax2 = plt.subplots()
        ax2.pcolormesh(u[:N-1,:N-1], shading='flat', cmap=cmap)
    
    # setting a threshold value for wave propogation:
    w = 0.01
    
    v = []
    
    for k in range(steps):
        
        # appending values to the velocity list where the velocity is greater than the threshold value
        # np.where gets the indices of when the velcoity u is greater than w
        # np.max finds the greatest index
        # threshold index holds the index just before w is crossed
        threshold_index = np.max(np.where(u[:,k] > w))
        v.append(u[threshold_index, k])
    
    # calculating the instantaneous velocity for each k
    inst_v = ((v[steps-1] - v[0]) / ((len(v) - 1) * dt * skip)) * dx
    
    return u, time, inst_v



def main():
    
    # PROBLEM 3
    
    # using epsilon and sigma values found from problem 1 for aggregation
    #print(brownian_particles(5, 1e-8, 1e-6, 1e-5, 2e-8, 1, 4e-20, 1e-8))
    
    steps = 50
    agg_x, agg_time = brownian_particles(500, 0.01, 1, 0.1, 2e-4, 10, 0.04, 0.01)
    
    plt.figure(1)
    plt.title("initial position of 500 brownian particles (aggregation)")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(agg_x[0,:,0], agg_x[1,:,0], 'bo', markersize = 3, label = "initial")
    
    plt.figure(2)
    plt.title("final position of 500 brownian particles (aggregation)")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(agg_x[0,:,steps-1], agg_x[1,:,steps-1], 'bo', markersize = 3, label = "final")
    # plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    plt.show()    
    
    # for repulsion:
    #print(brownian_particles(500, 1e-8, 1e-6, 1e-5, 2e-8, 1, 1e-21, 1e-7))
    
    steps = 50
    rep_x, rep_time = brownian_particles(500, 0.01, 1, 0.1, 2e-4, 10, 0.004, 0.1)
    
    plt.figure(3)
    plt.title("intial position of 500 brownian particles (repulsion)")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(rep_x[0,:,0], rep_x[1,:,0], 'co', markersize = 3, label = "initial")
    
    plt.figure(4)
    plt.title("final position of 500 brownian particles (repulsion)")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(rep_x[0,:,steps-1], rep_x[1,:,steps-1], 'co', markersize = 3, label = "final")
    #plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    plt.show()    
    
    
    # PROBLEM 4
    
    # calling the function to produce the movie
    fishers_equation(0.01, 50, 0.001, 100)
    
    # analyzing the relationship between diffusion coefficient and velocity
    velocities = []
    Ds = [0.05 * i for i in range(50)]
    
    for coefficients in Ds:
        u, time, v = fishers_equation(coefficients, 5, 0.001, 100)
        velocities.append(np.abs(v))

    plt.figure()
    plt.title("velocity's dependence on diffusion coefficient")
    plt.xlabel("diffusion coefficient")
    plt.ylabel("velocity")
    plt.plot(Ds, velocities, 'c')
    plt.show()
    

    
if __name__ == "__main__":
    main()
    