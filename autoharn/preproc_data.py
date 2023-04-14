              
from pathlib import Path
import shutil   
import pandas as pd
from os.path import exists
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import settings as cfg
import json
import re
import numpy as np

from data.nguyen import IsNguyen, Nguyen
from data.jin import IsJin, Jin
from data.neat import IsNeat, Neat
from data.GrammarVAE import IsGrammarVAE, GrammarVAE

def TrainTest(dataFile, splitType, seed = None):
    '''
    Often we need to split a single dataFile into training and testin samples. 
    We may want to split uniform at random or order and extrapolate with training 
    at the extremities of the data (larger or lower values of the target variable y)
    This function reads the 'split type' that is used to interpret what the user is 
    trying to achieve e.g.
        
        # shuffleMM     Select top MM% as testing, reminaing as training
        # topMM         Select top MM% as testing, remaining as training
        # bottomMM      Select bottom MM% as testing, reamining as training
        # NNextremeMM   Select bottom NN% and top MM% for testing, mid region as training

    returns 
        DataFrame for training samples
        DataFrame for testing samples
    '''

    print(f"Reading {dataFile}")
    # Read in the datafile
    dfData = pd.read_csv(dataFile)

    # Perform a uniform at random split
    if splitType.find('shuffle') > -1:     
        splitPct = float(splitType[-2:])/100
        return train_test_split(dfData, test_size = splitPct, random_state=seed, shuffle=True)

    # Perform an extrapolation split where testing samples have high values of y
    if splitType.find('top') > -1:

        dfData.sort_values(by=['y'], inplace=True)
        splitPct = float( splitType[-2:] )/100
        return train_test_split(dfData, test_size = splitPct, random_state=seed, shuffle=False)
        
    # Perform an extrapolation split where testing samples have low values of y
    if splitType.find('bottom') > -1:

        # Sort based on Y and flip to take bottom
        dfData.sort_values(['y'], ascending=[False], inplace=True)        
        return train_test_split(dfData, test_size = splitPct, random_state=seed, shuffle=False)
    
    # Perform an extrapolation split where testing samples have low or high values of y
    # Currently not tested/implemented
    '''
    if splitType.find('extreme') > -1:

        nums = re.findall(r'\d+', ty)
        splitBottom = float(nums[0])/100
        splitTop = float(nums[1])/100

        # Get the top testing data
        inds = np.argsort(Y)
        Y = Y[inds]
        Xs = Xs[inds]
        dfTrain, dfTest = train_test_split(dfData, test_size = splitTop, random_state=seed, shuffle=False)

        # Get the bottom testing data 
        Y = np.flipud(Y)
        Xs = np.flipud(Xs)
        dfTrain, dfTest = train_test_split(dfData, test_size = splitBottom, random_state=seed, shuffle=False)

        # Remove the samples off the begining of the data to get the training samples
        #!!!
        #XsTrain = XsTrain1[len(YTest2):]
        #YTrain = YTrain1[len(YTest2):]
        
        # Join the testing samples from the top and bottom
        #!!!
        #XsTest = np.concatenate((XsTest1,XsTest2))
        #YTest = np.concatenate((YTest1,YTest2))
    '''   

def IsGenerativeDataset(dataset):

    if  IsNguyen(dataset) or \
        IsJin(dataset) or \
        IsNeat(dataset) or \
        IsGrammarVAE(dataset):
        return True

    return False

def GenerateDataset(dataset, seed):

    XValsTrain = []
    yValsTrain = []
    XValsTest = []
    yValsTest = []

    data_func = None
    if IsNguyen(dataset): data_func = Nguyen      
    if IsJin(dataset): data_func = Jin   
    if IsNeat(dataset): data_func = Neat   
    if IsGrammarVAE(dataset): data_func = GrammarVAE   
        
    reg = re.search("\d+$", dataset.strip())
    if reg == None:
        raise Exception(f"No dataset number in {dataset}")
    
    no = reg.group(0)

    XValsTrain, yValsTrain = data_func(dataset, int(no), seed)
    XValsTest, yValsTest = data_func(dataset, int(no), seed+1, False)
    
    return XValsTrain, yValsTrain, XValsTest, yValsTest

def NoiseData(XValsTrain, yValsTrain):

    for i in range(0,len(yValsTrain)):
        print(f"{yValsTrain[i]},{XValsTrain[i][0]}")

    if cfg.ppYNoise[0] > 0:
        print('adding',cfg.ppYNoise[0],'noise to target')
        yValsTrain += np.random.normal(0, 
                    cfg.ppYNoise[0]*np.sqrt(np.mean(np.square(yValsTrain))),
                    size=len(yValsTrain))
    
    if cfg.ppXNoise[0] > 0:

        print('adding',cfg.ppXNoise[0],'noise to features')

        XValsTrain = np.array(XValsTrain)
        i = 0
        for column in XValsTrain.T:
            column = column+np.random.normal(0, cfg.ppXNoise[0]*np.sqrt(np.mean(np.square(np.array(column)))), size=len(np.array(column)))
            XValsTrain[:,i] = column
            i += 1

    print("")
    for i in range(0,len(yValsTrain)):
        print(f"{yValsTrain[i]},{XValsTrain[i][0]}")

    return XValsTrain, yValsTrain

def GenerateData(seed, dataset):

    '''
    We need to duplicate the origianl data to the 'source data' in the results directory
    The names of the new source data files are returned
    '''

    XValsTrain, yValsTrain, XValsTest, yValsTest = GenerateDataset(dataset, seed)
        
    # Ensure destination exists
    Path(f"{cfg.dstDataDir}/{dataset}/{seed}/").mkdir(parents=True, exist_ok=True)

    # Perform the copy
    srcTrainFile = f"{cfg.dstDataDir}/{dataset}/{seed}/{dataset}.Train.csv"
    srcTestFile = f"{cfg.dstDataDir}/{dataset}/{seed}/{dataset}.Test.csv"

    cols = []
    cols.append('y')
    index = 0
    for X in XValsTrain[0]:
        cols.append(f"x{index}")
        index += 1

    # Build Train Data
    df = pd.DataFrame(XValsTrain)
    df.insert(0, 'y', yValsTrain)
    df.columns = cols
    df.to_csv(srcTrainFile, index = False)

    if cfg.ppYNoise[0] > 0 or cfg.ppXNoise[0] > 0:

        srcTrainFile = srcTrainFile.replace("Train", "TrainNoise")
        XValsTrain, yValsTrain = NoiseData(XValsTrain,yValsTrain)

        # Build Train Data
        df = pd.DataFrame(XValsTrain)
        df.insert(0, 'y', yValsTrain)
        df.columns = cols
        df.to_csv(srcTrainFile, index = False)

    # Build Test Data
    df = pd.DataFrame(XValsTest)
    df.insert(0, 'y', yValsTest)
    df.columns = cols
    df.to_csv(srcTestFile, index = False)

    # Return the new files in the results directory we will be processing
    return srcTrainFile, srcTestFile

def SplitAndMoveData(srcTrainFile, srcTestFile, dstDataDir, seed, algorithm, dataset, split, splitNo):
    '''
    In the data directory we divide our data into splits like shuffle75, top20 etc.
    These are preserved in the data directory (e.g. out/results/data). A copy is provided 
    to the specific results directory such that all of the executions information is 
    closely located. This function performs the copy between these two directories
    '''
    
    # Ensure the source file exists
    if not exists(srcTrainFile):        raise Exception(f"Source file {srcTrainFile} does not exist")

    # Under the results data directory we may have many splits and many iterations 
    # of the program where the data is split by different seeds. We may also have
    # multiple datasets. Thus we need a folder structure to manage files. This is
    # common between the data directory and the execution directory
    resultStructure = f"{dataset}/{seed}/{split}"

    # Set train and test files
    dstTrain = f"{dstDataDir}/{resultStructure}/{seed}.Train.csv"
    dstTest = f"{dstDataDir}/{resultStructure}/{seed}.Test.csv"
    
    # Ensure destination exists
    Path(f"{dstDataDir}/{resultStructure}").mkdir(parents=True, exist_ok=True)

    # Split the data
    dfTrain = None
    dfTest = None

    if split == 'none':

        dfTrain = pd.read_csv(srcTrainFile)
        dfTest = dfTrain
    
    elif split == 'manual':

        dfTrain = pd.read_csv(srcTrainFile)
        dfTest = pd.read_csv(srcTestFile)

    else:

        dfTrain, dfTest = TrainTest(srcTrainFile, split, seed)
    
    if splitNo < len(cfg.ppNormalisation) and cfg.ppNormalisation[splitNo]:

        cols = dfTrain.columns
        cfg.ppScaler = MinMaxScaler()
        dfTrain = pd.DataFrame( cfg.ppScaler.fit_transform(dfTrain), columns = cols)
        dfTest = pd.DataFrame( cfg.ppScaler.transform(dfTest), columns = cols)

    # Create output files
    dfTrain.to_csv(dstTrain, index=False)
    dfTest.to_csv(dstTest, index=False)

    # Return filenames
    return dstTrain, dstTest

def CopyOriginalData(dataset, origTrainFile, origTestFile, seed):
    '''
    We need to duplicate the origianl data to the 'source data' in the results directory
    The names of the new source data files are returned
    '''
    srcTrainFile = ''
    srcTestFile = ''

    # Ensure destination exists
    Path(f"{cfg.dstDataDir}/{dataset}/{seed}/").mkdir(parents=True, exist_ok=True)

    # Perform the copy
    srcTrainFile = f"{cfg.dstDataDir}/{dataset}/{seed}/{dataset}.csv"
    shutil.copyfile(f"{origTrainFile}", srcTrainFile)
    
    # If the test file exists, process it
    if origTestFile != "":
        srcTestFile = f"{cfg.dstDataDir}/{dataset}/{seed}/{dataset}-Test.csv"
        shutil.copyfile(f"{origTestFile}", srcTestFile)

    # Return the new files in the results directory we will be processing
    return srcTrainFile, srcTestFile

def AddVarsToFile(srcFile, testFile, solns):

    dfTrain = pd.read_csv(srcFile)
    dfTest = pd.read_csv(testFile)

    for i, soln in enumerate(solns):

        with open(soln) as file:

            solJson = json.load(file)

            expr = solJson["equations"]["Numpy"]

            # We need to reverse because sometimes later variables such as "Var10"
            # are matched by the earlier varaibles "Var1"
            for column in reversed(dfTrain.columns):

                expr = expr.replace(f"*{column}", f"*dfTrain.{column}")
            
            dfTrain = pd.eval( f"m{i} = {expr}", target=dfTrain)

            # Add model to the testing file if it has not been added already
            if f"m{i}" not in dfTest.columns:
                expr = expr.replace("dfTrain", "dfTest")
                dfTest = pd.eval( f"m{i} = {expr}", target=dfTest)

    dfTrain.to_csv(srcFile, index=False)
    dfTest.to_csv(testFile, index=False)