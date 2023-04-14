
from data.generate import get_random_vals

def Jin(dataset, no, seed = None, isTrain = True):

    minVal = -3
    maxVal = 3
    samples = 100
    if not isTrain:
        samples = 30

    if no == 1:     return get_random_vals("2.5*x**4-1.3*x**3+0.5*z**2-1.7*z", minVal, maxVal, samples)
    if no == 2:     return get_random_vals("8.0*x**2+8.0*z**3-15.0", minVal, maxVal, samples)
    if no == 3:     return get_random_vals("0.2*x**3+0.5*z**3-1.2*z-0.5*x", minVal, maxVal, samples)
    if no == 4:     return get_random_vals("1.5*exp(x)+5.0*cos(z)", minVal, maxVal, samples)
    if no == 5:     return get_random_vals("6.0*sin(x)*cos(z)", minVal, maxVal, samples)
    if no == 6:     return get_random_vals("1.35*x*z+5.5*sin((x-1.0)*(z-1.0))", minVal, maxVal, samples)

    raise Exception(f"No Jin dataset: '{no}'. Use an integer in set 1-6 ")

def IsJin(dataset):

    if 'jin' in dataset.lower():
        return True
    
    return False