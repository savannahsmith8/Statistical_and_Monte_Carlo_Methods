# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:21:20 2024

@author: savan
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def YukawaOrbit(mBH, mObj, T, dt, skip):
    '''
        This function computes the Yukawa Orbit of two objects
        
            Parameters
            ----------
            mBH : 'float'
                mass of the black hole
            mObj : 'float'
                mass of the orbiting object
            T : 'float'
                total time of the orbit for the simulation to run
            dt : 'float'
                time interval
            skip : 'int'
                number of data point to skip to make code more efficient
        
            Returns
            -------
            rBH : 'float'
                radius of the black hole WRT center of mass
            rObj : 'float'
                radius of the orbiting object WRT center of mass
            vBH : 'float'
                velocity of the black hole
            vObj : 'float'
                velocity of the orbiting object
    '''
    
    # define constants
    g = 1       # normalized gravitational constant
    lam = 0.5   # range parameter
    # calculate the reduced mass
    mu = mBH * mObj / (mBH + mObj)
    steps = round(T/dt/skip)
    
    # initialize the np.arrays for the variables
    rBH = np.zeros((steps,2), 'float')
    rObj = np.zeros((steps,2), 'float')
    vBH = np.zeros((steps,2), 'float')
    vObj = np.zeros((steps,2), 'float')
    time = np.zeros((steps,1), 'float')
    
    # calculatethe position of both objects WRT COM
    rBH[0,0] = -mObj / (mBH + mObj)
    rObj[0,0] = mBH / (mBH + mObj)
    
    # calculate the velocity of both objects using reduced mass
    vBH[0,1] = -np.sqrt(3*mu*np.exp(-0.5)/2)/mBH
    vObj[0,1] = np.sqrt(3*mu*np.exp(-0.5)/2)/mObj
    
    # initialize a midpoint to be used for iteration
    rBHmid = np.zeros((1,2), 'float')
    rObjmid = np.zeros((1,2), 'float')
    vBHmid = np.zeros((1,2), 'float')
    vObjmid = np.zeros((1,2), 'float')
    
    # prep for movie plot
    metadata = dict(title = "YukawaOrbit", \
                    artist = "Matplotlib")
    writer = FFMpegWriter(fps = 15, metadata = metadata)
    fig1 = plt.figure()
    lBH, = plt.plot([], [], 'ko', ms=25)
    lObj, = plt.plot([], [], 'yo', ms=5)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    with writer.saving(fig1, "YukawaOrbit.mp4", 100):
        
        # loop through as many steps as defined above
        for i in range(1, steps):
            
            # define the current values as the previous one
            rBH[i,:] = rBH[i - 1,:]
            vBH[i,:] = vBH[i - 1,:]
            rObj[i,:] = rObj[i - 1,:]
            vObj[i,:] = vObj[i - 1,:]
            time[i] = time[i - 1]
            
            for j in range(skip):
                
                # store midpoint radii
                rBHmid[0,:] = rBH[i,:] + dt * vBH[i,:] / 2
                rObjmid[0,:] = rObj[i,:] + dt * vObj[i,:] / 2
                
                # calculate the COM position
                r = np.sqrt((rBHmid[0,0] - rObjmid[0,0]) \
                    ** 2 + (rBHmid[0,1] - rObjmid[0,1]) ** 2)
                # define the Yukawa force from Yukawa potential
                Fmag = g ** 2 * np.exp(-lam*r) * \
                    (lam*r + 1) / r ** 3
                    
                # store midpoint velocities
                vBHmid[0,:] = 0.5 * vBH[i,:]
                vObjmid[0,:] = 0.5 * vObj[i,:]
                
                # update velocities using Yukawa force
                vBH[i,:] = vBH[i,:] + dt * Fmag * \
                    (rObj[i,:] - rBH[i,:]) / mBH
                vObj[i,:] = vObj[i,:] + dt * Fmag * \
                    (rBH[i,:] - rObj[i,:]) / mObj
                vBHmid[0,:] += 0.5 * vBH[i,:]
                vObjmid[0,:] += 0.5 * vObj[i,:]
                
                # update the positions with new midpoint velocity
                rBH[i,:] = rBH[i,:] + dt * vBHmid[0,:]
                rObj[i,:] = rObj[i,:] + dt * vObjmid[0,:]
            
            lBH.set_data(rBH[i, 0], rBH[i, 1])
            lObj.set_data(rObj[i, 0], rObj[i, 1])
            writer.grab_frame()
            plt.pause(0.02) 
            
    # plot the orbit of the two objects
    plt.figure()
    plt.title("Yukawa orbit of massive object around BH")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(rBH[0,0], rBH[0,1], 'go', label = "start")
    plt.plot(rBH[-1,0], rBH[-1,1], 'co', label = "end")
    plt.plot(rObj[0,0], rObj[0,1], 'go')
    plt.plot(rObj[-1,0], rObj[-1,1], 'co')
    plt.plot(rBH[:,0], rBH[:,1], 'k', ms=50) 
    plt.plot(rObj[:,0], rObj[:,1], 'y', ms=5)
    plt.legend()
    
    return rBH, rObj, vBH, vObj


def GROrbit(E, u, l, dt):
    '''
        This function calculates the general relativity orbit of a massive 
        object orbiting a black hole

            Parameters
            ----------
            E : 'float'
                energy of the system
            u : 'float'
                initial value of the potential
            l : 'float'
                angular momentum
            dt : 'float'
                time interval for iterations
        
            Returns
            -------
            None
                GR orbit is plotted using matplotlib

    '''
    
    # scaling constants for easier math
    G = 1   # gravitational constant
    M = 1   # mass of BH
    c = 1   # speed of light
    
    # derivative of u WRT phi which is the angle of precession
    du = np.sqrt(2*E/(l**2) + 2*G*u/(l**2) - G*u**2 + 2*G*(u**3))
    
    def dudphi(du):
        return du
    def du2dphi2(u):
        V = -u + G*M/(l**2) + 3*G*M*u**2/(c**2)
        return V
    
    # create an array or phi values
    phi = np.arange(0, 12*np.pi, dt)
    
    # initialize position vectors to be the shape of phi
    r_vals = np.zeros(phi.shape)
    x = np.zeros(phi.shape)
    y = np.zeros(phi.shape)
    
    # integrating using fourth order Runge-Kutta
    for i, ph in enumerate(phi):
        k1 = dt*dudphi(du)
        k2 = dt*dudphi(du + k1/2)
        k3 = dt*dudphi(du + k2/2)
        k4 = dt*dudphi(du + k3)
        l1 = dt*du2dphi2(u)
        l2 = dt*du2dphi2(u + l1/2)
        l3 = dt*du2dphi2(u + l2/2)
        l4 = dt*du2dphi2(u + l3)
        u += (k1 + 2*k2 + 2*k3 + k4)/6
        du += (l1 + 2*l2 + 2*l3 + l4)/6
        r = 1 / u
        r_vals[i] = r
        x[i] = r * np.cos(ph)
        y[i] = r * np.sin(ph)
    
    # plot the GR orbit
    plt.subplots(figsize=(10,10))
    plt.title("Massive Object Orbiting a BH")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.plot(x, y, 'm', label = "GR orbit")
    plt.plot(x[0], y[0], 'co', label = "start", ms = 5)
    plt.plot(x[-1], y[-1], 'ro', label = "end", ms = 5)
    plt.plot(0, 0, 'ko', label = "BH", ms = 15)
    plt.legend()
    plt.show()
