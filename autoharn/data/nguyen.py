
import numpy as np
import numexpr as ne
import random
from data.generate import get_random_vals

def NguyenStd(no, seed = None):

    # Set seed
    if seed == None:    seed = np.random.randint(0,2**32-1)
    random.seed(seed)

    
    if no == 1:     return get_random_vals("x**3+x**2+x", -1, 1, 20)
    if no == 2:     return get_random_vals("x**4+x**3+x**2+x", -1, 1, 20)
    if no == 3:     return get_random_vals("x**5+x**4+x**3+x**2+x", -1, 1, 20)
    if no == 4:     return get_random_vals("x**6+x**5+x**4+x**3+x**2+x", -1, 1, 20)
    if no == 5:     return get_random_vals("sin(x**2)*cos(x)-1", -1, 1, 20)
    if no == 6:     return get_random_vals("sin(x)+sin(x+x**2)", -1, 1, 20)
    if no == 7:     return get_random_vals("log10(x+1)+log10(x**2+1)", 0, 2, 20)
    if no == 8:     return get_random_vals("sqrt(x)", 0, 4, 20)
    if no == 9:     return get_random_vals("sin(x)+sin(z**2)", 0, 1, 20)
    if no == 10:    return get_random_vals("2*sin(x)*cos(z)", 0, 1, 20)
    if no == 11:    return get_random_vals("x**z", 0, 1, 20)
    if no == 12:    return get_random_vals("x**4-x**3+0.5*z**2-z", 0, 1, 20)

    raise Exception(f"No Nguyen dataset: '{no}'. Use an integer number 1 through 12")

def NguyenDash(no, seed = None):

    # Set seed
    if seed == None:    seed = np.random.randint(0,2**32-1)
    random.seed(seed)

    if no == 2:     return get_random_vals("4*x**4+3*x**3+2*x**2+x", -1, 1, 20)
    if no == 5:     return get_random_vals("sin(x**2)*cos(x)-2", -1, 1, 20)
    if no == 8:     return get_random_vals("x**(1/3)", 0, 4, 20)

    raise Exception(f"No Nguyen Dash: '{no}'. Use an integer in set [2, 5, 8]")

def NguyenDashDash(no, seed = None):

    if no == 8:     return get_random_vals("(x**2)**(1/3)", 0, 4, 20)

    raise Exception(f"No Nguyen Dash Dash dataset: '{no}'. Use an integer in set [8] ")

def NguyenC(no, seed = None):

    if no == 1:     return get_random_vals("3.39*x**3+2.12*x**2+1.78*x", -1, 1, 20)
    if no == 5:     return get_random_vals("sin(x**2)*cos(x)-0.75", -1, 1, 20)
    if no == 7:     return get_random_vals("log10(x+1.4)+log10(x**2+1.3)", 0, 2, 20)
    if no == 8:     return get_random_vals("sqrt(1.23*x)", 0, 4, 20)
    if no == 10:    return get_random_vals("sin(1.5*x)*cos(0.5*z)", 0, 1, 20)

    raise Exception(f"No Nguyen C dataset: '{no}'. Use an integer in set [8] ")

def Nguyen(dataset, no, seed = None, isTrain = True):

    if "nguyen''" in dataset.lower():   return NguyenDashDash(no, seed)
    if "nguyen'" in dataset.lower():    return NguyenDash(no, seed)
    if "nguyenc" in dataset.lower():    return NguyenC(no, seed)
    if "nguyen" in dataset.lower():     return NguyenStd(no, seed)

def IsNguyen(dataset):

    if "nguyen" in dataset.lower():
        return True

    return False