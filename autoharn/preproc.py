
from pathlib import Path

from sklearn import preprocessing
import shutil                               # Copy files

import config.settings as cfg
import numpy as np
import pandas as pd

import bin

from sklearn.preprocessing import MinMaxScaler

if not cfg.isParallel:
    import seaborn as sb

from preproc_data import TrainTest

def CheckMissingData(self, dataDir, dataTrain, dfTrain, dataTest, dfTest):

    if dfTrain.isnull().any().sum() != 0:
        print( f"!!! Preprocess issue, null values found in train file {dataTrain}")

    if dataTrain != dataTest:

        if dfTest.isnull().any().sum() != 0:
            print( f"!!! Preprocess issue, null values found in test file {dataTest}")

    return

def CheckDistribution(self, dataDir, dataTrain, dfTrain, dataTest, dfTest):

    for column in dfTrain:

        ax = sb.violinplot(data=dfTrain[column])
        ax.set_xticklabels([column])
        fig = ax.get_figure()
        #sfig.set_size_inches(25, 25)
        #fig.autofmt_xdate(rotation=45)
        fig.savefig(f"{dataDir}/_Train.Violin.{column}.png")
        plt.close()

        if dataTrain != dataTest:
            ax = sb.violinplot(data=dfTest[column])
            ax.set_xticklabels([column])
            fig = ax.get_figure()
            fig.savefig(f"{dataDir}/_Test.Violin..{column}.png")
            plt.close()
            
        sb.displot(dfTrain[column])
        plt.savefig(f"{dataDir}/_Train.Dist.{column}.png")
        plt.close()

        if dataTrain != dataTest:
            sb.displot(dfTest[column])
            plt.savefig(f"{dataDir}/_Test.Dist.{column}.png")
            plt.close()

    return

def CheckCorrelation(self, dataDir, dataTrain, dfTrain, dataTest, dfTest):

    correlation_matrix = dfTrain.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    fig = plt.figure()
    sb.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, mask=mask, square = True)
    fig.set_size_inches(15, 15)
    plt.savefig(f"{dataDir}/Train.Correlation.png")
    plt.close()

    if dataTrain != dataTest:

        correlation_matrix = dfTest.corr()
        sb.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, mask=mask, square = True)
        plt.savefig(f"{dataDir}/Test.Correlation.png")
        plt.close()

def CheckCorrelationList(self, dataDir, dataTrain, dfTrain, dataTest, dfTest):

    for column in dfTrain:

        ax = sb.violinplot(data=dfTrain[column])
        ax.set_xticklabels([column])
        fig = ax.get_figure()
        #sfig.set_size_inches(25, 25)
        #fig.autofmt_xdate(rotation=45)
        fig.savefig(f"{dataDir}/_Train.Violin.{column}.png")
        plt.close()

        if dataTrain != dataTest:
            ax = sb.violinplot(data=dfTest[column])
            ax.set_xticklabels([column])
            fig = ax.get_figure()
            fig.savefig(f"{dataDir}/_Test.Violin..{column}.png")
            plt.close()
            
        sb.displot(dfTrain[column])
        plt.savefig(f"{dataDir}/_Train.Dist.{column}.png")
        plt.close()

        if dataTrain != dataTest:
            sb.displot(dfTest[column])
            plt.savefig(f"{dataDir}/_Test.Dist.{column}.png")
            plt.close()

    return

def CheckNormalisation(self, dataDir, dataTrain, dfTrain, dataTest, dfTest):

    arr = dfTrain.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(arr)
    dfTrainNew = pd.DataFrame(x_scaled)
    dfTrainNew.columns = dfTrain.columns
    dfTrainNew.to_csv(f'{dataDir}/_Train.Normalised.csv', header=True, index=False)

    if dataTrain != dataTest:
        arr = dfTest.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(arr)
        dfTestNew = pd.DataFrame(x_scaled)
        dfTestNew.columns = dfTest.columns
        dfTestNew.to_csv(f'{dataDir}/_Test.Normalised.csv', header=True, index=False)

# !! Function moved from postproc.py and not tested in this context
def SummaryCorrelation(self):
    ''' Generate correlation plots for each dataset '''

    for dataLine in self.getDatas():

        dataName, dataIters, dataTest, dataTrain, isDataComparative, _, minY, maxY, minX, maxX= self.stripData(dataLine)
        XsTrain, _, N, dYTrain, WTrain, _ = Reader.CsvDatasetFile(dataTrain)

        file = f"out/summary/{dataName}/correlation.{dataName}"
        Plot.CorrelationMatrix(file, XsTrain, N)

# !! Function moved from postproc.py and not tested in this context
def SummaryCovariance(self):
    ''' Generate coveriance plots for each dataset '''

    for dataLine in self.getDatas():
        
        dataName, dataIters, dataTest, dataTrain, isDataComparative, _, minY, maxY, minX, maxX = self.stripData(dataLine)    
        XsTrain, _, N, dYTrain, WTrain, _ = Reader.CsvDatasetFile(dataTrain)

        file = f"out/summary/{dataName}/covariance.{dataName}"
        Plot.CovarianceMatrix(file, XsTrain, N)

def PreprocessDataset(seed, dataset, srcTrain, srcTest, doBinning = False):

    # Copy original data to results
    
    shutil.copyfile(srcTrain, f"{cfg.dstDataDir}/{dataset}/{seed}/original/{seed}.Train.csv")
    shutil.copyfile(srcTest, f"{cfg.dstDataDir}/{dataset}/{seed}/original/{seed}.Test.csv",)

    # If splitting data
    for idx, split in enumerate(cfg.splits):

        Path(f"{cfg.dstDataDir}/{dataset}/{seed}/{split}").mkdir(parents=True, exist_ok=True)      

        # If we are using training for both train and test datasets            
        if split.lower() == 'train':
            
            dfTrain = pd.read_csv(srcTrain)

            if idx < len(cfg.ppNormalisation) and cfg.ppNormalisation[idx]:
                cols = dfTrain.columns
                scaler = MinMaxScaler()
                dfTrain = pd.DataFrame( scaler.fit_transform(dfTrain), columns = cols)
            
            dfTrain.to_csv(f"{cfg.dstDataDir}/{dataset}/{seed}/{split}/{seed}.Train.csv", index=False)
            dfTrain.to_csv(f"{cfg.dstDataDir}/{dataset}/{seed}/{split}/{seed}.Test.csv", index=False)

        elif split.lower() == 'manual':
            
            dfTrain = pd.read_csv(srcTrain)
            dfTest = pd.read_csv(srcTest)

            if idx < len(cfg.ppNormalisation) and cfg.ppNormalisation[idx]:
                cols = dfTrain.columns
                scaler = MinMaxScaler()
                dfTrain = pd.DataFrame( scaler.fit_transform(dfTrain), columns = cols)
            
            dfTrain.to_csv(f"{cfg.dstDataDir}/{dataset}/{seed}/{split}/{seed}.Train.csv", index=False)
            dfTest.to_csv(f"{cfg.dstDataDir}/{dataset}/{seed}/{split}/{seed}.Test.csv", index=False)

        # If we are using a single source file and we need to split into training and testing
        else:

            dfTrain, dfTest = TrainTest(srcTrain, split, seed)

            if idx < len(cfg.ppNormalisation) and cfg.ppNormalisation[idx]:
                cols = dfTrain.columns  
                scaler = MinMaxScaler()
                dfTrain = pd.DataFrame( scaler.fit_transform(dfTrain), columns = cols)
                dfTest = pd.DataFrame( scaler.transform(dfTest), columns = cols)

            dfTrain.to_csv(f"{cfg.dstDataDir}/{dataset}/{seed}/{split}/{seed}.Train.csv", index=False)
            dfTest.to_csv(f"{cfg.dstDataDir}/{dataset}/{seed}/{split}/{seed}.Test.csv", index=False)

        bin.knuthBinMedian(
            f"{cfg.dstDataDir}/{dataset}/{seed}/{split}/{seed}.Train.csv", 
            cfg.ppBinIters
        )



if __name__ == '__main__':


    '''
    if __name__ == '__main__':

        startTime = time.time()

        cfg.SeedSetup()

        # Ensure seed counts
        seeds = cfg.getDatas(cfg.seedsFile)
        if len(seeds) < cfg.totalIters:
            print('Not enough seeds for the number of iterations')
            exit()

        for i in range(0, cfg.totalIters):
                
            # Get common seed for the iteration across all datasets and methods
            seed = int(seeds[i])

            for dataLine in cfg.getDatas(cfg.datasFile):

                # Extract data config and prepare output directory
                dataName, _, dataTest, dataTrain, isDataComparative, splitMethod, minY, maxY, minX, maxX = cfg.stripData(dataLine)

                #dfTrain = pd.read_csv(dataTrain)
                #dfTest = pd.read_csv(dataTest)

                PreprocessDataset(seed, dataName, dataTrain, dataTest)

                # Perform datachecks if configued
                #if cfg.ppMissingData:
                #    h.CheckMissingData(cfg.srcDataDir, dataTrain, dfTrain, dataTest, dfTest)
                
                # Output histograms if configured
                #if cfg.ppHistogram in cfg.ppOutput or cfg.ppViolin in cfg.ppOutput:
                #    h.CheckDistribution(cfg.srcDataDir, dataTrain, dfTrain, dataTest, dfTest)

                # Output correlation matrix if configured
                #if cfg.ppCorrelation in cfg.ppOutput:
                #    h.CheckCorrelation(cfg.srcDataDir, dataTrain, dfTrain, dataTest, dfTest)
                
                # Output covariance matrix if configured
                # if cfg.ppCovariance:
                #

                # Normalise based on config value
                #if cfg.ppNormalisation:
                #    h.CheckNormalisation(cfg.srcDataDir, dataTrain, dfTrain, dataTest, dfTest)
    '''