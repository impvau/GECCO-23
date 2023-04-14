
from data.generate import get_uniform_vals

def GrammarVAE(dataset, no, seed = None, isTrain = True):

    if no == 1:     return get_uniform_vals("1.0/3.0+x+sin(x**2)", -10, 10, 10**3)

    raise Exception(f"No GrammarVAE dataset: '{no}'. Use an integer in set [1] ")

def IsGrammarVAE(dataset):

    if 'grammarvae' in dataset.lower():
        return True
    
    return False