
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

font = {'family' : 'normal',
        'size'   : 22}

plt.rc('font', **font)

SUMDIR = 'out/summary'
DIR = 'out/results'
DATASET = 'sinx'
SEED = '3581673728'

def getMethods():

    methods = []
    for file in os.listdir(DIR):

        filename = os.fsdecode(file)    
        if os.path.isdir(f"{DIR}/{file}") and filename != 'data':

            methods.append(filename)

    return methods

def custom_plot():
    
    methods = getMethods()

    # Plot original function
    plt.figure(figsize=(15,15))

    # Get the actual results
    filename = f"{DIR}/data/{DATASET}/{SEED}.Train.csv"
    df = pd.read_csv( f"{filename}", index_col=False )
    actual = df['y'].to_numpy()
    xaxis = df['x'].to_numpy()    
    plt.scatter(xaxis, actual, label='ground', c='black', alpha=0.5)

    # Add the results of each method
    for method in methods:

        filename = f"{DIR}/{method}/{DATASET}/{SEED}.Train.Predict.csv"
        df = pd.read_csv( f"{filename}", index_col=False )
        df = df.to_numpy()

        plt.plot(xaxis, df, label=method, alpha=0.5)

    plt.legend()

    plt.ylim(3.5, 5.25)
    plt.xlim(-15, 15)

    plt.savefig("Compare.png")
    plt.close()


def xToExcel(equ):


    splits = re.split("\*|\+|\-",equ)

    lenFromB = ord('Z')-ord('B')        # First excel character is B, so we have the first 25 varaibles with "B" as the variable name [B, C, D, ..., Z]
    lenAlphabet = ord('Z')-ord('A')      # For all on subsequent iterations its from A, e.g. for varaibles 26 to 52 we have "AA" to "AZ" [ AA, AB, AC, ..., AZ ]

    # Loop through in reverse so we encounter "x11" bfore "x1" that will match both "x11" and "x1"
    for split in reversed(splits):

        res = re.search('^x[0-9]*$', split)

        if res:

            # Minus one because we are not 0 based but x1 based
            num = int(split[1:])-1

            lastChar = ''
            
            # Determine number of A's to place in front of variable
            numAs = 0
            if num > lenFromB:
                numAs = (num-lenFromB) % lenAlphabet
                numAs += 1      # For the > lenFromB

                remain = num-lenFromB-lenAlphabet*numAs

                lastChar = chr(ord('A')+remain)

            else:

                lastChar = chr(ord('B')+num)

            excelVar = f"{'A'*numAs}{lastChar}2"
            equ = equ.replace(split, excelVar)

    return equ
        

    '''
    # Generate a continued log term given coeffs and intercept
    ret = ''
    char = ''
    integ = 1

    # set first coeff
    if ty == "excel":
        char = sChr
        ret = str(round(coeffs[0],rnd))+"*"+char+"2"    
    if ty == "^":
        ret = str(round(coeffs[0],rnd))+"*x^"+str(integ)
    if ty == "numpy":
        ret = str(round(coeffs[0],rnd))+"*x**"+str(integ)

    isPastZ = False
    newChar = ''
    indChar = ord(sChr)+1

    # loop through 2nd to n coeffs
    for i in range(1,len(coeffs)):

        if indChar == ord('Z')+1: 
            indChar = ord('A')
            if newChar == '': newChar = 'A'
            else: newChar = chr(ord(newChar)+1)
        
        c = newChar+chr(indChar)

        if coeffs[i] != 0 :
            # raise to the power
            if ty == "excel": ret += " + "+str(round(coeffs[i],rnd))+"*"+c+"2"
            if ty == "^"    : ret += " + "+str(round(coeffs[i],rnd))+"*x^"+str(i+1)
            if ty == "numpy": ret += " + "+str(round(coeffs[i],rnd))+"*x**"+str(i+1)

        indChar += 1

    return ret + " + " +str(round(intercept,rnd))
    '''

def custom_equation_translate():

    filename = f"{SUMDIR}/results.csv"
    df = pd.read_csv( f"{filename}", index_col=False )

    print(df)

    for index, row in df.iterrows():

        df.at[index,'Model'] = xToExcel(row['Model'])

    df.to_csv(filename.replace("results", "results.excel"), index=False)


custom_equation_translate()