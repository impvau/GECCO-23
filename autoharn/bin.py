
import math
import numpy as np
import pandas as pd
import astropy.stats as ast         # Knuths method
import config.settings as cfg
if not cfg.isParallel:
    import matplotlib.pyplot as plt     # Saving plots

def binMedianOld(data, bins, key, width):

    # Create DataFrame object in same form as the data
    dataOut = pd.DataFrame(data=None, columns=data.columns)

    # For each of the bins
    for b in bins:

        # Extract all samples within the bin ranges
        extract = data[ (data[key] >= b) & (data[key] < b+width) ].sort_values('y')

        # Determine the median value for samples within the bin - for all varaibles
        medians = extract.mean()
        # Median can lead to high 
        #medians = extract.median()
        
        # Assign the median values 
        nextIdx = len(dataOut.index)
        for col in dataOut.columns:
            dataOut.at[nextIdx,col] = medians[col]
        
    print(dataOut)

    dataOut = dataOut.astype(float)
    dataOut = dataOut.fillna(method='ffill')

    # Interpolate Nan bins - poor performance where many bins
    #dataOut = dataOut.interpolate()

    print(dataOut)

    return dataOut 

def binMedian(data, bins):

    # Get the range of data to determine a uniform bin width across that data
    minY = data['y'].min()
    maxY = data['y'].max()
    dataRange = maxY-minY

    # Create DataFrame object in same form as the data
    dataOut = pd.DataFrame(data=None, columns=data.columns)

    # Loop over data
    for i in range(0, len(bins)-1):

        extract = data[ (data['y'] >= bins[i] ) & (data['y'] < bins[i+1]+0.0001) ]
        medians = extract.mean()
        
        # Ensure the same number of samples, just apply mean values
        for _ in range(0, len(extract)):

            # Assign the median values 
            nextIdx = len(dataOut.index)
            for col in dataOut.columns:
                dataOut.at[nextIdx,col] = medians[col]
 
        # Only add the mean value for each bin once. If using this, fill NaN bins with last valus or interpolation later
        #nextIdx = len(dataOut.index)
        #for col in dataOut.columns:
        #    dataOut.at[nextIdx,col] = medians[col]
        
    dataOut = dataOut.astype(float)

    # Fill NaN bins with value of the last row 
    dataOut = dataOut.fillna(method='ffill')

    # Interpolate NaN bins
    #dataOut = dataOut.interpolate()

    return dataOut 

def binSqrt(data):

    minY = data.min()
    maxY = data.max()
    dataRange = maxY-minY

    samples = len(data.index)
    binCnt = math.floor(math.sqrt(samples))

    binLen = dataRange/binCnt

    bins = []
    for i in range(0,binCnt):
        bins.append(minY+binLen*i)

    bins.append(minY+binLen*binCnt)

    return bins

def knuthBinMedian(file, iters = 5):

    # Read in file with cols y,<iv1>,<iv2>,<ivN>
    print(f"Bin - file:{file}")
    data = pd.read_csv(file, index_col=False)
    data = data.sort_values('y')

    # Get bin widths from Knuth
    #_, bins = ast.knuth_bin_width(data['y'], return_bins=True)
    bins = binSqrt(data['y'])

    # Get the median value for each bin
    dataOut = binMedian(data, bins)

    # Output
    newFile = file.replace(".csv",".knuth.0.csv")
    dataOut.to_csv(newFile,index=False)
    
    '''
    for column in dataOut:
        plt.bar(dataOut.index, dataOut[column].to_numpy())
        plt.savefig(file.replace(".csv",f".knuth.0.{column}.png"))
        plt.close()
    '''

    # Determine the number of bins to add in each iteration
    # If we consider a dataset that has as many bins as samples we effective have the origianl dataset
    # We move from the Knuth approximation to the original dataset
    # If we have say 10 bins from Knuth, and 100 training samples, we move in equal steps toward 100
    # If we are wanting to bin five times we have (100-10)/(5-1) = 22.5, then we take the floor at 22
    #   knuth.0 = 10 bins
    #   knuth.1 = 10+22 = 32 bins
    #   knuth.2 = 10+22+22 = 52 bins
    #   knuth.3 = 10+22+22+22 = 72 bins
    # The 4th iteration that would be = 10+22+22+22+22 = 94 bins, however at this point we can just
    # use the training set, thus we iterate for iters-1 times
    
    minY = data['y'].min()
    maxY = data['y'].max()
    dataRange = maxY-minY

    # Get number of bins
    binCount = len(bins)

    # Get number of samples
    samples = len(data.index)
    
    # Determine the number of bins each iteration
    binsEachIter = math.floor((samples-binCount)/(iters-1))
        
    print(f"Bin - knuthCnt:{binCount} Samples:{samples} EachIter:{binsEachIter}")

    # For each binning resolution
    for i in range(1,(iters-1)):

        # Determine bins in this iteration
        binsThisIter = binCount+i*binsEachIter

        # Determine uniform bin width across the data
        binWidth = dataRange/binsThisIter
        
        # Add the bins
        bins = []
        startVal = minY
        for _ in range(0, binsThisIter):
            bins.append(startVal)
            startVal += binWidth

        dataOut = binMedian(data, bins)
        dataOut.to_csv(file.replace(".csv",f".knuth.{i}.csv"),index=False)
        
        '''
        for column in dataOut:

            plt.bar(dataOut.index, dataOut[column].to_numpy())
            plt.savefig(file.replace(".csv",f".knuth.{i}.{column}.png"))
            plt.close()
        '''

    return newFile
