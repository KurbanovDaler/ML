import math
import matplotlib.pyplot
import scipy
import numpy

#Function F(x) given in the task
def f(x):
    return math.sin(x / 5) * math.exp(x / 10) + 5 * math.exp(-x / 2)

#Calculating gradient

ans1 = scipy.optimize.minimize(f, 2, method = 'BFGS')
ans2 = scipy.optimize.minimize(f, 30, method = 'BFGS')

#Printing results
print("Answer number 1 is %.2f" %ans1.fun)
print("Answer number 2 is %.2f" %ans2.fun)

#Creating an array of points that belong to X axis
x = numpy.arange(1, 30, .1)
y = list(map(f, x))
y2 = list(map(f, x))

#Plotting the Graph
matplotlib.pyplot.plot(x, y)
matplotlib.pyplot.plot(x, y2)
