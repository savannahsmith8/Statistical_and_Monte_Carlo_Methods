# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:46:53 2024

@author: Savan
"""

import numpy as np
import matplotlib.pyplot as plt

# PROBLEM 1

# definding the first order ordinary differential equation
def ODE(x):
    dx = x - x ** 2
    return dx

# calculating the differential using forward euler method
def forward_euler_calc(ode, t0, tf, x0, num):
    # initializing the xs array as floats
    xs = np.array([ 0 for i in range(num) ], 'float')
    # setting initial and final times, setting dt as dependent on num
    time, dt = np.linspace(t0, tf, num, retstep=True)
    # initial condition
    xs[0] = x0
    # loops through how ever many steps were input, starting as the second array item
    for i in range(1, num):
        # adds the new xs value to the list at index i based on the previous item
        xs[i] = xs[i - 1] + dt * ode(xs[i - 1])
    return xs, time


# PROBLEM 2

# define dx/dt as a function
def ODE_X(x, y, a):
    # dx / dt
    dx = a - x + (x ** 2) * y
    return dx
    
# define dy/dt as a function
def ODE_Y(x, y, b):
    # dy / dt
    dy = b - (x ** 2) * y
    return dy

# calculates the semi-implicit calc
def semi_implicit_euler_calc(a, b, t0, tf, num):
    # initializes the arrays
    x = np.array([ 0 for i in range(num) ], 'float' )
    y = np.array([ 0 for i in range(num) ], 'float' )
    time, dt = np.linspace(t0, tf, num, 'retstep','true','float')
    # initial values from setting the ODEs equal to zero
    # add and subtract 0.5 from each initial value in order to make it slightly 
    # different from equilibrium values
    x[0] = a + b + 0.5
    y[0] = b / ((a + b) ** 2) - 0.5
    # loops through the arrays starting at the second entry
    for i in range(1, num - 1):
        x[i] = (x[i - 1] + dt * (a + (x[i - 1] ** 2) * y[i - 1])) / ( 1 + dt )
        y[i] = y[i - 1] + dt * (b - (x[i - 1] ** 2) * y[i - 1])
    return x, y, time


# PROBLEM 3

# defining the ODEs as their own functions

def m_ODE(m, c, w):
    # dm / dt
    dm = c * (1 - m) - 10 * w * m 
    return dm

def c_ODE(m, c, w, S, t):
    # dc / dt
    dc = 5 * m * (1 - c) - 1.25 * c + S(t)
    return dc

def w_ODE(m, c, w):
    # dw / dt
    dw = 0.1 * (1 - w) - 4 * m * w 
    return dw

# calculating the shift for functions m, c, and w
def fourth_order_RK(f, m, c, w, t, dt, alpha):
    # defining the function S(t) with the important time values
    def S(t):
        if 3 <= t <= 3.2:
            return alpha
        else:
            return 0
        
    # using the iteration of Runge-Kutta until the fourth for each function
    
    first_rk_m = f[0](m, c, w)
    first_rk_c = f[1](m, c, w, S, t)
    first_rk_w = f[2](m, c, w)
    
    second_rk_m = f[0](m + 0.5 * dt * first_rk_m, c + 0.5 * dt * first_rk_c, \
                       w + 0.5 * dt * first_rk_w)
    second_rk_c = f[1](m + 0.5 * dt * first_rk_m, c + 0.5 * dt * first_rk_c, \
                       w + 0.5 * dt * first_rk_w, S, t + 0.5 * dt)
    second_rk_w = f[2](m + 0.5 * dt * first_rk_m, c + 0.5 * dt * first_rk_c, \
                       w + 0.5 * dt * first_rk_w)
    
    third_rk_m = f[0](m + 0.5 * dt * second_rk_m, c + 0.5 * dt * second_rk_c, \
                      w + 0.5 * dt * second_rk_w)
    third_rk_c = f[1](m + 0.5 * dt * second_rk_m, c + 0.5 * dt * second_rk_c, \
                      w + 0.5 * dt * second_rk_w, S, t + 0.5 * dt)
    third_rk_w = f[2](m + 0.5 * dt * second_rk_m, c + 0.5 * dt * second_rk_c, \
                      w + 0.5 * dt * second_rk_w)

    fourth_rk_m = f[0](m + dt * third_rk_m, c + dt * third_rk_c, w + dt * third_rk_w)
    fourth_rk_c = f[1](m + dt * third_rk_m, c + dt * third_rk_c, w + dt * third_rk_w, S, t + dt)
    fourth_rk_w = f[2](m + dt * third_rk_m, c + dt * third_rk_c, w + dt * third_rk_w)

    # using simpsons rule to find the new function value
    new_m = m + (dt / 6) * (first_rk_m + 2 * second_rk_m + 2 * third_rk_m + fourth_rk_m)
    new_c = c + (dt / 6) * (first_rk_c + 2 * second_rk_c + 2 * third_rk_c + fourth_rk_c)
    new_w = w + (dt / 6) * (first_rk_w + 2 * second_rk_w + 2 * third_rk_w + fourth_rk_w)
    
    return new_m, new_c, new_w

# computes the list of values for the different ODEs
def iteration(t0, tf, num, alpha):
    
    # initializes the lists to zeros
    times, dt = np.linspace(t0, tf, num, retstep=True)
    ms = [ 0 for i in range(num) ]
    cs = [ 0 for i in range(num) ]
    ws = [ 0 for i in range(num) ]
    
    # sets the given initial values at the first position of the array 
    ms[0] = 0.0114
    cs[0] = 0.0090
    ws[0] = 0.9374
    
    # loops through the items in each array
    for i in range(1, num):
        # adds the fourth order approximation to the list at index i
        ms[i], cs[i], ws[i] = fourth_order_RK([m_ODE, c_ODE, w_ODE], ms[i - 1], \
                            cs[i - 1], ws[i - 1], times[i - 1], dt, alpha)
    return ms, cs, ws, times


def main():
    
    # PROBLEM 1

    # hard-coding the variables used for problem 1
    t0 = 0 
    tf = 30
    x0 = 0.1
    
    # looping through different num values to see the different graphs
    for num in [3000, 1000, 100, 30, 15, 12, 10]:
        xs, time = forward_euler_calc(ODE, t0, tf, x0, num)
        print("xs:", xs)
        print("times:", time)
        # plotting the results from the function
        plt.plot(time, xs, 'c')
        # titling each plot with the dt value
        plt.title(f"x-value versus time (dt = {30 / num})")
        plt.xlabel("time")
        plt.ylabel("x value")
        plt.show()
      
 
    # PROBLEM 2
    
    # hard-coding the variables used for problem 2
    t0 = 0 
    tf = 30
    num = 1000
    
    # oscillations occur when a and b are not equal
    b = 0.5
    a = (b ** (1/3)) - b    # comes from the equlibrium x and y values
    
    print(semi_implicit_euler_calc(a, b, t0, tf, num))
    
    # assigning values to variables being used to graph
    x_values, y_values, times = semi_implicit_euler_calc(a, b, t0, tf, num)
   
    # titling the plot and axes
    plt.title(f"function versus time (dt = {30 / num})")
    plt.xlabel("time")
    plt.ylabel("function value")
    # plotting the results from the function with a legend
    plt.plot(times, x_values, 'm', label = "x function")
    plt.plot(times, y_values, 'y', label = "y function")
    plt.legend()
    plt.show()
    
    # equilibrium exists when a and b are equal
    a = 1
    b = 1
    
    print(semi_implicit_euler_calc(a, b, t0, tf, num))
    
    # assigning values to variables being used to graph
    x_values, y_values, times = semi_implicit_euler_calc(a, b, t0, tf, num)
   
    # titling the plot and axes
    plt.title(f"equilibrium state: function versus time (dt = {30 / num})")
    plt.xlabel("time")
    plt.ylabel("function value")
    # plotting the results from the function with a legend
    plt.plot(times, x_values, 'm', label = "x function")
    plt.plot(times, y_values, 'y', label = "y function")
    plt.legend()
    plt.show()


    # PROBLEM 3
    
    # hard coding given values
    t0 = 0.0 
    tf = 30.0
    # ensures that dt = 0.002
    num = 15000
    
    # looping through alpha to find where the m and w function switches
    for alpha in [5, 10, 13, 13.5, 13.71, 13.715, 13.72, 13.75, 14, 20, 50]:
        print(iteration(t0, tf, num, alpha))
        # assigning values to variables being used to graph
        m, c, w, time = iteration(t0, tf, num, alpha)
        # titling the plot with the alpha value and axes 
        plt.title(f"functions versus time (alpha = {alpha})")
        plt.xlabel("time")
        plt.ylabel("function value")
        # plotting the results from the function with a legend
        plt.plot(time, m, 'r', label = "m function")
        plt.plot(time, c, 'g', label = "c function")
        plt.plot(time, w, 'b', label = "w function")
        plt.legend()
        plt.show() 


if __name__ == "__main__" :
    main()