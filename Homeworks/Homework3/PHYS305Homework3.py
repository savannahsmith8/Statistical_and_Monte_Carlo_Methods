# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:45:44 2024

@author: Savan
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


# PROBLEM 1

# solves for the two dimensional orbit of a mass at the end of a spring
# can easily change qom to anaylze how the function changes when q/m changes
def spring_orbit(qom, T, dt, skip):
    
    # sets the steps
    steps = round(T/dt/skip)
    
    # initializing the arrays
    r = np.zeros((steps,2), 'float')
    v = np.zeros((steps,2), 'float')
    time = np.zeros((steps,1), 'float')
    
    # defining the initial conditions
    
    r[0,0] = 0
    r[0,1] = 1
    
    v[0,0] = 1
    v[0,1] = 0
    
    # initializing the arrays for mid values
    rmid = np.zeros((1,2), 'float')
    vmid = np.zeros((1,2), 'float')
    
    # looping through the time steps
    for i in range(1, steps):
        
        # inputs current value for later use
        r[i,:] = r[i - 1,:]
        v[i,:] = v[i - 1,:]
        time[i] = time[i - 1]
        
        # looping through the skips
        for j in range(skip):
            
            # using the forward euler method
            rmid[0,:] = r[i,:] + dt * v[i,:] / 2
            
            # defining the magnitude of the distance and of the force
            rmag = np.sqrt((rmid[0,0] ** 2) + (rmid[0,1] ** 2))
            Fmag = -1 * rmag - (qom / (rmag ** 2))
            
            # moving a half step
            vmid[0,:] = 0.5 * v[i,:]
            # assigning velocity based on the force
            v[i,:] = v[i,:] + dt * Fmag * r[i,:]
            # moving another half step
            vmid[0,:] += 0.5 * v[i,:]
            # reassigning the distance based on the velocity
            r[i,:] = r[i,:] + dt * vmid[0,:]
            
    return r


# PROBLEM 2

# calculates the location and velocities of four bodies while producing a movie
# showing their movements and a plot tracing their orbits after time T
# let m1 = sun, m2 = earth, m3 = jupiter, m4 = moon
def four_body_problem(m1, m2, m3, m4, T, dt, skip):
   
    G = 6.67 * 10 ** (-11)
    # defining step size
    steps = round(T/dt/skip)
    
    # intializing the array for position and velocities for the bodies
    
    r1 = np.zeros((steps,2), 'float')
    r2 = np.zeros((steps,2), 'float')
    r3 = np.zeros((steps,2), 'float')
    r4 = np.zeros((steps,2), 'float')
    
    v1 = np.zeros((steps,2), 'float')
    v2 = np.zeros((steps,2), 'float')
    v3 = np.zeros((steps,2), 'float')
    v4 = np.zeros((steps,2), 'float')
    
    # initializing the array for time
    time = np.zeros((steps,1), 'float')
    
    # intial conditions:
        
    # assume the sun is at the center of the system and it's wobble is insignificant
    
    # hard coding distance variables
    
    rSE = 1.496 * 10 ** 11      # from the sun to earth
    rSJ = 7.553 * 10 ** 11      # from the sun to jupiter
    rEM = 3.844 * 10 ** 8       # from the earth to the moon
    
    # let all objects be alligned so we only have to consider one dimension for 
    # initial conditions
    
    r1[0,0] = 0                
    r2[0,0] = 0
    r3[0,0] = 0               
    r4[0,0] = 0
    
    r1[0,1] = 0                 # from the sun to the sun
    r2[0,1] = rSE               # from the sun to earth
    r3[0,1] = rSJ               # from the sun to jupiter
    r4[0,1] = rSE + rEM         # adding distances to get total distance
        
    # using the same initial assumption used for distances
    # assuming circular orbits and constant velocities
    # hard coding the initial velocities for the four bodies
    
    v1[0,0] = 0
    v2[0,0] = - 2.978 * 10 ** 4
    v3[0,0] = - 1.307 * 10 ** 4
    v4[0,0] = v2[0,0] - 1 * 10 ** 3
    
    v1[0,1] = 0
    v2[0,1] = 0
    v3[0,1] = 0
    v4[0,1] = 0
    
    metadata = dict(title = "four body problem", artist = "Matplotlib") #needs to be same as def?
    writer = FFMpegWriter(fps = 15, metadata = metadata)
    
    fig1 = plt.figure()
    l1, = plt.plot([], [], 'yo', markersize = 10)
    l2, = plt.plot([], [], 'bo', markersize = 3)
    l3, = plt.plot([], [], 'ro', markersize = 5)
    l4, = plt.plot([], [], 'ko', markersize = 1)
    
    plt.xlim(-1e12, 1e12)
    plt.ylim(-1e12, 1e12)
    
    with writer.saving(fig1, "four_body_problem.mp4", 100):
    
        for i in range(1, steps):
            
            # Set current rs to previous value
            r1[i,:] = r1[i - 1,:]
            v1[i,:] = v1[i - 1,:]
            
            r2[i,:] = r2[i - 1,:]
            v2[i,:] = v2[i - 1,:]
            
            r3[i,:] = r3[i - 1,:]
            v3[i,:] = v3[i - 1,:]
            
            r4[i,:] = r4[i - 1,:]
            v4[i,:] = v4[i - 1,:]
            
            time[i] = time[i - 1]
            
            for j in range(skip):
                
                # calculating the radii at half time step using the forward euler method
                r1mid = r1[i,:] + dt * v1[i,:] / 2
                r2mid = r2[i,:] + dt * v2[i,:] / 2
                r3mid = r3[i,:] + dt * v3[i,:] / 2
                r4mid = r4[i,:] + dt * v4[i,:] / 2
        
                
                # distance between all the bodies is now different so we need 
                # to know the distance between each pair based on the forward 
                # euler method above
                
                r12 = r1mid - r2mid
                r13 = r1mid - r3mid
                r14 = r1mid - r4mid
                r23 = r2mid - r3mid
                r24 = r2mid - r4mid
                r34 = r3mid - r4mid
                
                # for the equation for gravitational force, we want the magnitude
                
                mag_r12 = np.linalg.norm(r12)
                mag_r13 = np.linalg.norm(r13)
                mag_r14 = np.linalg.norm(r14)
                mag_r23 = np.linalg.norm(r23)
                mag_r24 = np.linalg.norm(r24)
                mag_r34 = np.linalg.norm(r34)
                
                # in general we know that: F = G * (m1 * m2) / (r ** 2)
                # we need the force on every object in the correct direction
                
                F_on_m1 = G * m1 * ( (m2 / (mag_r12 ** 3) * -r12) + (m3 / (mag_r13 ** 3) * -r13) + (m4 / (mag_r14 ** 3) * -r14))
                F_on_m2 = G * m2 * ( (m1 / (mag_r12 ** 3) * r12) + (m3 / (mag_r23 ** 3) * -r23) + (m4 / (mag_r24 ** 3) * -r24))
                F_on_m3 = G * m3 * ( (m2 / (mag_r23 ** 3) * r23) + (m1 / (mag_r13 ** 3) * r13) + (m4 / (mag_r34 ** 3) * -r34))
                F_on_m4 = G * m4 * ( (m1 / (mag_r14 ** 3) * r14) + (m2 / (mag_r24 ** 3) * r24) + (m3 / (mag_r34 ** 3) * r34))
                
                # moving a half step
                v1mid = 0.5 * v1[i,:]
                v2mid = 0.5 * v2[i,:]
                v3mid = 0.5 * v3[i,:]
                v4mid = 0.5 * v4[i,:]
             
                # reassigning values for the velocity based on the defined forces
                v1[i,:] = v1[i,:] + dt * F_on_m1 / m1
                v2[i,:] = v2[i,:] + dt * F_on_m2 / m2
                v3[i,:] = v3[i,:] + dt * F_on_m3 / m3
                v4[i,:] = v4[i,:] + dt * F_on_m4 / m4
                
                # moving another half step
                v1mid += 0.5 * v1[i,:]
                v2mid += 0.5 * v2[i,:]
                v3mid += 0.5 * v3[i,:]
                v4mid += 0.5 * v4[i,:]
                
                # update distance with the mid value
                r1[i,:] = r1[i,:] + dt * v1mid
                r2[i,:] = r2[i,:] + dt * v2mid
                r3[i,:] = r3[i,:] + dt * v3mid
                r4[i,:] = r4[i,:] + dt * v4mid
            
            # setting variables for video
            l1.set_data(r1[i, 0], r1[i, 1])
            l2.set_data(r2[i, 0], r2[i, 1])
            l3.set_data(r3[i, 0], r3[i, 1])
            l4.set_data(r4[i, 0], r4[i, 1])
            writer.grab_frame()
            plt.pause(0.02)    
            
    # plotting
    plt.figure() 
    plt.plot(r1[:,0], r1[:,1], 'y', markersize = 20, label = "Sun")
    plt.plot(r2[:,0], r2[:,1], 'b', markersize = 6, label = "Earth")
    plt.plot(r3[:,0], r3[:,1], 'r', markersize = 10, label = "Jupiter")
    plt.plot(r4[:,0], r4[:,1], 'k', markersize = 2, label = "Moon")
    plt.legend()
    plt.show()
            
    return r1, r2, r3, r4, v1, v2, v3, v4


def main():
    
    # PROBLEM 1
    
    r = spring_orbit(2, 50, 0.01, 10)
    
    plt.figure()    
    plt.plot(r[:,0], r[:,1], 'c')
    
    print(spring_orbit(2, 50, 0.01, 10))


    # PROBLEM 2
    
    # hard-coding mass values
    m1 = 1.989 * 10 ** 30   # mass of the sun in kg
    m2 = 5.972 * 10 ** 24   # mass of the earth in kg
    m3 = 1.898 * 10 ** 27   # mass of jupiter in kg
    m4 = 7.348 * 10 ** 22   # mass of the moon in kg
    
    print(four_body_problem(m1, m2, m3, m4, 1*10**6, 100, 1))
    

if __name__ == "__main__":
    main()