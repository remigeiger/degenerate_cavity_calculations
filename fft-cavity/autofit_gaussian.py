import numpy as np
import scipy.optimize as so

def gaussian2(xdata, A, w, x0):
    return A*np.exp(-2*(xdata-x0)**2/w**2)

def autofit_gaussian(xdata, ydata):
    guess = [np.max(ydata), (xdata[-1]-xdata[0])/2, xdata[np.argmax(ydata)]]

    popt, _ = so.curve_fit(gaussian2, xdata, ydata, guess,
                           bounds=([0, 0, -np.inf], [+np.inf, +np.inf, +np.inf]))
    return popt, {'amplitude': popt[0],
                  'waist': popt[1],
                  'position': popt[2]}
