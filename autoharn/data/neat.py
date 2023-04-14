
from data.nguyen import Nguyen 
from data.generate import get_uniform_vals, get_random_vals, get_series_vals

def Neat(dataset, no, seed = None, isTrain = True):

    if no == 1:     return Nguyen('Nguyen1', 2, seed)       # Neat1 === Nguyen1
    if no == 2:     return Nguyen('Nguyen2', 3, seed)       # Neat2 === Nguyen2
    if no == 3:     return Nguyen('Nguyen5', 5, seed)       # Neat3 === Nguyen5
    if no == 4:     return Nguyen('Nguyen7', 7, seed)       # Neat4 === Nguyen7
    if no == 5:     return Nguyen('Nguyen10', 10, seed)     # Neat5 === Nguyen10
    if no == 6 and isTrain:        return get_series_vals("sum(1/x)", 1, 50, 50)
    if no == 6 and not isTrain:    return get_series_vals("sum(1/x)", 1, 120, 120)
    if no == 7:     return get_uniform_vals("2-2.1*cos(9.8*x)*sin(1.3*z)", -50, 50, 10**5)
    if no == 8:     return get_random_vals("exp(-(x-1)**2)/(1.2+(z-2.5)**2)", 0.3, 4, 100)
    if no == 9:     return get_uniform_vals("1/(1+x**(-4))+1/(1+z**(-4))", -5, 5, 21) 

    raise Exception(f"No Nguyen Dash Dash dataset: '{no}'. Use an integer in set 1-9 ")

def IsNeat(dataset):

    if 'neat' in dataset.lower():
        return True
    
    return False