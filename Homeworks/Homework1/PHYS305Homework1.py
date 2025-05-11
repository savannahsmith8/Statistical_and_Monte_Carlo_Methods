# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 18:35:55 2024

@author: Savan
"""

# importing math for all code that requires it
import math

# PROBLEM 1

# calculates the temperature in celsius from the temperature in fahrenheit
def fahrenheit_to_celsius(temp):
    # calculating the temperature in celsius
    celsius = (temp - 32) / (9/5)
    return celsius


# PROBLEM 2

# calculated number of molecules assuming CO2
N = 2.2 * 10**23 
# Stephen-Boltzmann constant multiplied by assumed room temperature 298 K
kBT = 4 * 10**(-14)

# calculates the work done using the lefthand reimann sum
def work_done_LH(Vi, Vf, num):
    # defining the size of the step based on the change in volume and the 
    # number of steps
    dV = (Vf - Vi) / (num - 1)
    # setting the first volume in the for loop to be the initial volume
    V = Vi
    # initializing the work variable
    work = 0
    # looping through all lefthand areas
    # will contain the data point with the initial volume but not the final
    for i in range(num - 1):
        # updating the work variable by adding the small area under the curve
        # where (N * kBT / V) is the y-variable and dV is the x-variable
        work = work + (N * kBT / V) * dV
        # moving onto the next data point
        V = V + dV
    return work

# calculates the work done using the righthand reimann sum
def work_done_RH(Vi, Vf, num):
    # defining the size of the step based on the change in vol and step number
    dV = (Vf - Vi) / (num - 1)
    # setting the first volume to be the righthand point at the second x value
    V = Vi + dV
    # initializing the work variable
    work = 0
    # looping through all righthand areas
    # will contain the data point for final volume but not the initial volume
    for i in range(num):
        # updating the work variable by adding the small area under the curve
        # where (N * kBT / V) is the y-variable and dV is the x-variable
        work = work + (N * kBT / V) * dV
        # moving onto the next data point
        V = V + dV
    return work

# calculates the work done using the trapedoizal reimann sum
def work_done_TR(Vi, Vf, num):
    # defining the size of the step based on the change in vol and step number
    dV = (Vf - Vi) / (num - 1)
    # setting the first volume to be the midpoint between the first and second 
    # data points
    V = (1/2) * (Vi + (Vi + dV))
    # initializing the work variable
    work = 0
    # looping through all righthand areas
    # will not contain the data point for final volume nor the initial volume
    for i in range(num - 1):
        # updating the work variable by adding the small area under the curve
        # where (N * kBT / V) is the y-variable and dV is the x-variable
        work = work + (N * kBT / V) * dV
        # moving onto the next data point
        V = V + dV
    return work


# PROBLEM 3

# calculating the analytical solution from the Ideal Gas Law
def analytical(Vi, Vf):
    # equation comes from the integral from initial to final volume with 
    # integrand P * dV
    analytical_sol = (N * kBT) * math.log(Vf/Vi)
    return analytical_sol

# calcualtes the error of the integral estimations from problem 2
# creates lists of errors based on a list of step size (varibale called nums)
def calc_errors(Vi, Vf, nums):
    # assigns the results of analytical to a usable variable
    sol = analytical(Vi, Vf)
    # initializes error lists
    error_LH = [ 0 for i in range(10) ]
    error_RH = [ 0 for i in range(10) ]
    error_TR = [ 0 for i in range(10) ]
    # loops through all ten step sizes and inserts the error values into the 
    # initialized lists
    for i in range(10):
        # calculates the absolute value of the error using sol and inserts into 
        # list depending on i value
        error_LH[i] = abs(work_done_LH(Vi, Vf, nums[i]) - sol)
        error_RH[i] = abs(work_done_RH(Vi, Vf, nums[i]) - sol)
        error_TR[i] = abs(work_done_TR(Vi, Vf, nums[i]) - sol)
    return error_LH, error_RH, error_TR


# PROBLEM 4

# calculates an estimation of the sine integral using the lefthand reimann sum
def sine_integral_LH(lower_limit, upper_limit, num):
    # defining the step size
    dx = (upper_limit - lower_limit) / (num - 1)
    # starting x value will be the lower limit for lefthand
    x = lower_limit
    # initializing the integral estimation variable
    integral_est = 0
    # looping through all lefthand areas
    for i in range(num - 1):
        # updates the integral with new area under the curve
        integral_est = integral_est + math.sin(x) * dx
        # moves onto next x value
        x = x + dx
    # calculates the error based on the known value of the the sine integral
    error_LH = abs(integral_est - ( -math.cos(upper_limit) + math.cos(lower_limit)))
    return integral_est, error_LH

# calculates an estimation of the sine integral using the righthand reimann sum
def sine_integral_RH(lower_limit, upper_limit, num):
    # defining the step size
    dx = (upper_limit - lower_limit) / (num - 1)
    # starting x value will be the second x value for righthand
    x = lower_limit + dx
    # initializing the integral estimation variable
    integral_est = 0
    # looping through all righthand areas
    for i in range(num):
        # updates the integral with new area under the curve
        integral_est = integral_est + math.sin(x) * dx
        # moves onto next x value
        x = x + dx
    # calculates the error based on the known value of the the sine integral
    error_RH = abs(integral_est - ( -math.cos(upper_limit) + math.cos(lower_limit)))
    return integral_est, error_RH

# calculates an estimation of the sine integral using the trapezoidal reimann sum
def sine_integral_TR(lower_limit, upper_limit, num):
    # defining the step size
    dx = (upper_limit - lower_limit) / (num - 1)
    # starting x value will be the midpoint between first and second x values
    x = (1/2) * (lower_limit + (lower_limit + dx))
    # initializing the integral estimation variable
    integral_est = 0
    # looping through all trapezoidal areas
    for i in range(num - 1):
        # updates the integral with new area under the curve
        integral_est = integral_est + math.sin(x) * dx
        # moves onto next x value
        x = x + dx
    # calculates the error based on the known value of the the sine integral
    error_TR = abs(integral_est - ( -math.cos(upper_limit) + math.cos(lower_limit)))
    return integral_est, error_TR

# puts the error calculates from the above functions into list notation for 
# each rule in order to prepare for plot
def sine_errors(lower_limit, upper_limit, nums):
    # initializes error lists
    error_LH = [ 0 for i in range(10) ]
    error_RH = [ 0 for i in range(10) ]
    error_TR = [ 0 for i in range(10) ]
    # loops through all ten step sizes and inserts the error values into the 
    # initialized list
    for i in range(10):
        # allows the function to use the list we have defined as nums which 
        # includes 10 different step sizes
        num = nums[i]
        # assigning the second value in the tuple result from sine_integral to 
        # the error lists
        error_LH[i] = sine_integral_LH(lower_limit, upper_limit, num)[1]
        error_RH[i] = sine_integral_RH(lower_limit, upper_limit, num)[1]
        error_TR[i] = sine_integral_TR(lower_limit, upper_limit, num)[1]
    return error_LH, error_RH, error_TR
    

# PROBLEM 5, PART E

# defines the function that is used as the integrand
def function(x):
    integrand = (1 - ((math.e) ** (-x))) / (x * (1 + (x **2)))
    return integrand

# approximates the integral using the integrand defined above
def approx_fnct(lam, dx):
    # initializes the x-value based on the trapezoidal rule
    x = (1/2) * (0 + (0 + dx))
    # initializes the approximation
    approx = 0
    # looping through all steps based on trapezoidal rule
    for i in range(lam - 1):
        # allows the function to evaluate the "y-value" of the function
        y = function(x)
        # adds new area under the curve to the previous value
        approx = approx + y * dx
        # moves to the next step
        x = x + dx
    # calculating the error using an approximated analytical solution of 0.921
    error = abs(approx - 0.921)
    return approx, error

    

def main():
    
    
    # PROBLEM 1
    
    # asking the user for the variable for the function
    temp = float(input("What is the temperature in fahrenheit?"))
    print(fahrenheit_to_celsius(temp))
    
    
    # PROBLEM 2
    
    # asking the user to provide the values for the variables used by function
    Vi = float(input("What was the initial volume?"))
    Vf = float(input("What was the final volume?"))
    # num must be an integer in order for the range code to work
    num = int(input("How many steps?"))

    print(work_done_LH(Vi, Vf, num))
    print(work_done_RH(Vi, Vf, num))
    print(work_done_TR(Vi, Vf, num))
    
    
    # PROBLEM 3
    
    import matplotlib.pyplot as plt
    
    # defining a list of numbers as sample step sizes
    nums = [ 10 + 10 * i for i in range(10) ]
    # assigns error values from the calculates errors
    err_LH, err_RH, err_TR = calc_errors(Vi, Vf, nums)
    
    # defining the values based on natural log functions
    lnN = [ math.log(i) for i in nums ]
    lnE_LH = [ math.log(i) for i in err_LH ]
    lnE_RH = [ math.log(i) for i in err_RH ]
    lnE_TR = [ math.log(i) for i in err_TR ]
    # addings labels and titles to the plot
    plt.title("Work Error as a Function of Step Size")
    plt.xlabel("ln(num)")
    plt.ylabel("ln(error)")
    # plotting the variables
    plt.plot(lnN, lnE_LH, 'r', label = "Left Hand")
    plt.plot(lnN, lnE_RH, 'g', label = "Right Hand")
    plt.plot(lnN, lnE_TR, 'b', label = "Trapezoidal")
    # adding a legend to the plot using the above labels
    plt.legend()
    plt.show()
    # found that error ~ 1/num
    
    
    # PROBLEM 4
    
    # asking the user for limit inputs
    lower_limit = float(input("What is the lower limit?"))
    upper_limit = float(input("What is the upper limit?"))

    print(sine_integral_LH(lower_limit, upper_limit, num))
    print(sine_integral_RH(lower_limit, upper_limit, num))
    print(sine_integral_TR(lower_limit, upper_limit, num))
    
    # assigns error values from the calculates errors
    sin_err_LH, sin_err_RH, sin_err_TR = sine_errors(lower_limit, upper_limit, nums)
    
    # defining the values based on natural log functions
    lnN = [ math.log(i) for i in nums ]
    lnE_LH = [ math.log(i) for i in sin_err_LH ]
    lnE_RH = [ math.log(i) for i in sin_err_RH ]
    lnE_TR = [ math.log(i) for i in sin_err_TR ]
    # addings labels and titles to the plot
    plt.title("Sine Error as a Function of Step Size")
    plt.xlabel("ln(num)")
    plt.ylabel("ln(error)")
    # plotting the variables
    plt.plot(lnN, lnE_LH, 'm', label = "Left Hand")
    plt.plot(lnN, lnE_RH, 'y', label = "Right Hand")
    plt.plot(lnN, lnE_TR, 'c', label = "Trapezoidal")
    # adding a legend to the plot using the above labels
    plt.legend()
    plt.show()
    # found that error ~ 1/num
    
    
    # PROBLEM 5
    
    # printing the result from the approximating function with chosen lambda 
    # and dx values
    print(approx_fnct(1000000, 0.15))
    
    
if __name__ == "__main__":
    main()
