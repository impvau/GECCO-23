#
# execute.py
#   Run a series of models against a series of datasets and articular results
#

#from memory_profiler import profile

import config.settings as cfg
import os                                   # Run on command line
from pathlib import Path                    # Add dirs
import socket                               # Hostname
import time                                 # Execution time
import shutil 
import re
import math
import pandas as pd
import numpy as np

from preproc_data import SplitAndMoveData, CopyOriginalData, AddVarsToFile, IsGenerativeDataset, GenerateData
from bin import knuthBinMedian

if not cfg.isParallel:

    import mysql.connector                      # Db Connection
    import bin  
    from pytools import Preprocess              # Split data
    from pytools import Reader

#@profile
def runCmd(cmd, dataset, seed):

    # Print and execute script
    print(f"Executing {cmd} on {dataset} seed {seed}")    
    print(cmd)
    os.system(cmd)

#@profile
def execute():
    ''' 
    Execute algorithms against datasets in the CODES & DATA variables
    '''

    # Get the seeds we need to execute on
    seeds = cfg.GetSeeds()

    # Loop through all the data entries
    i = 0
    for dataset, origTrainFile, origTestFile in cfg.GetDatas():

        seed = seeds[i]
        i += 1
        #if IsGenerativeDataset(dataset):
        #    srcTrainFile, srcTestFile = GenerateData(seed, dataset)                
        #else:

        # Copy the original data to the results data directory which are the 'src' files
        srcTrainFile, srcTestFile = CopyOriginalData(dataset, origTrainFile, origTestFile, seed)            
        
        # Loop through all the algorithm entries
        for algorithm, algStr in cfg.getCodes():
            
            # If there is a test file specified, no split should occur
            if srcTestFile != "" and not IsGenerativeDataset(dataset):

                # Prepare output directory
                outDir = f"{cfg.outResDir}/{algorithm}/{dataset}/{seed}/manual"
                Path(outDir).mkdir(parents=True, exist_ok=True)      

                # Print and execute command
                command = ( 
                            f"{algStr} -s {seed} -t {srcTrainFile} -T {srcTestFile} -lt {outDir}"
                            #f" -mout" if cfg.ppAppendModel else ""
                            f" 1>{outDir}/{seed}.std.log 2>{outDir}/{seed}.err.log"
                        )

                runCmd(command, dataset, seed)

            # If there is no test file, we must split
            else: 

                # For each split configuration we have
                for splitNo, split in enumerate(cfg.splits):

                    outDir = f"{cfg.outResDir}/{algorithm}/{dataset}/{seed}/{split}"
                    Path(outDir).mkdir(parents=True, exist_ok=True)      

                    print(f"Reading {srcTrainFile}")
                    trainFile, testFile = SplitAndMoveData(srcTrainFile, srcTestFile, cfg.dstDataDir, seed, algorithm, dataset, split, splitNo)                    
        
                    #Run command
                    runCmd(f"{algStr} -s {seed} -t {trainFile} -T {testFile} -lt {outDir} 1>{outDir}/{seed}.std.log 2>{outDir}/{seed}.err.log", dataset, seed)

                    if cfg.ppNormalisation[0]:

                        dfTrainResult = pd.read_csv(f"{outDir}/{seed}.TrainResults.csv")
                        dfTestResult = pd.read_csv(f"{outDir}/{seed}.TestResults.csv")
                    
                        print(dfTrainResult.head())
                        dummy = pd.DataFrame(np.zeros((len(dfTrainResult), cfg.ppScaler.n_features_in_)))
                        dummy[0] = dfTrainResult['y']
                        dummy = pd.DataFrame(cfg.ppScaler.inverse_transform(dummy), columns=dummy.columns)
                        dfTrainResult["denorm(y)"] = dummy[0]
                        print(dfTrainResult.head())

                        dummy = pd.DataFrame(np.zeros((len(dfTrainResult), cfg.ppScaler.n_features_in_)))
                        dummy[0] = dfTestResult['y']
                        dummy = pd.DataFrame(cfg.ppScaler.inverse_transform(dummy), columns=dummy.columns)
                        dfTestResult["denorm(y)"] = dummy[0]

                        dfTrainResult.to_csv(f"{outDir}/{seed}.TrainResults.csv", index = None)
                        dfTestResult.to_csv(f"{outDir}/{seed}.TestResults.csv", index = None)
                    

                    print("")
                    
if __name__ == '__main__':

    startTime = time.time()

    print(f"Executing on {socket.gethostname()}")

    Path(cfg.outResDir).mkdir(parents=True, exist_ok=True)
    Path(cfg.outSumDir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(f"{cfg.outDir}/outconfig"):
        shutil.copytree(cfg.configDir,f"{cfg.outDir}/outconfig",)

    execute()
    
    endTime = time.time()
    duration = endTime-startTime
    print("Time Elapsed :"+str(round(duration,2))+"s ("+str(round((duration)/60))+"mins)")

