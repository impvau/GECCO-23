
import numpy as np
import numexpr as ne
import random

def get_random_vals(expr, valMin, valMax, samples):

    XVals = []
    yVals = []

    print(expr)

    for i in range(0,samples):

        Xs = []
        x = random.uniform(valMin, valMax)
        Xs.append(x)

        z = ''
        if 'z' in expr:
            z = random.uniform(valMin, valMax)
            Xs.append(z)

        XVals.append(Xs)
        y = ne.evaluate(expr)
        yVals.append(y)

        #if z == '':
        #    print( f"{y},{x}")
        #else:
        #    print( f"{y},{x},{z}")

    return XVals, yVals

def get_uniform_vals(expr, valMin, valMax, samples):

    XVals = []
    yVals = []

    print(expr)
    x = np.arange(valMin, valMax, (valMax-valMin)/(samples-1))

    # Sometimes the rounding means that sample number sample-1 + (valMax-valMin)/(samples-1) is > valMax so there is one less item than sample in the array
    # In this case we append valMax to ensure that there are the correct number of samples we are expecting
    # Logic repeats for z if provided
    if( len(x) == samples-1 ):
        x = np.append(x, valMax)

    XVals = np.c_[np.array(x)]
    if 'z' in expr:
        z = np.arange(valMin, valMax, (valMax-valMin)/(samples-1))
        if( len(z) == samples-1 ):
            z = np.append(z, valMax)
        XVals = np.c_[XVals, z]

    yVals = ne.evaluate(expr)

    return XVals, yVals

def get_series_vals(expr, valMin, valMax, samples):

    XVals = []
    yVals = []

    print(expr)

    XArr = np.arange(valMin, valMax, (valMax-valMin)/(samples-1))
    if( len(XArr) == samples-1 ):
        XArr = np.append(XArr, valMax)

    XVals = np.c_[np.array(XArr)]

    for i in range(0,samples):

        x = XArr[0:i+1]
        y = ne.evaluate(expr)

        yVals.append(y)
        #print( f"{y},{x}")

    return XVals, yVals
