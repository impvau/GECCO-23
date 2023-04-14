#
# launch.py
#   Run a series of models against a series of datasets and articular results
#

import  os               # Line processor
import  subprocess       # Command execution
import  socket           # Hostname
import  time             # Execution time
import  outconfig.settings as cfg
import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
import  scipy.stats as ss
import  scikit_posthocs as sp
import  json
from    pathlib import Path                    # Add dirs
from    sklearn.preprocessing import MinMaxScaler

from postproc_memetic import plot_agent_improve

import seaborn as sb
import Orange

''' 
Post-processing functionality for the autonmation harness, run after execute.py generates results
This is segregated as it may run directly after compuation on a single node, or it may first require
the rsyc of multiples nodes before processing
'''

def ResultsToCsv(dir = 'summary', resFile = 'results.csv'):

    """ Create a result CSV with MSE scores and configurations for all methods, datasets and iterations """

    # Get the header from the first <seed>.Run.csv output file and write this to the summary.csv
    header = subprocess.check_output("find out/ -name '*Run.log' -type f -exec sh -c 'cat $0' {} \; -quit | head -1 | sed 's/^/Method,Dataset,Seed,Split,/'", shell=True)
    header = header.decode('utf-8').strip()
    os.system( f"echo '{header}' > out/{dir}/{resFile};" )
    
    # Push results into resFile
    # - results/{method}/{dataset}/<seed>.Run.csv hold the indivdidual tests output
    # - we filter the header by removing where we see 'seed'
    # - we format the output that comes from using the -n tag that gives us the filepath, and thus {method}/{dataset} that form new columns
    # - we push the resultant into resFile in CSV format
    os.system( "find out/ -name *.Run.log -type f -exec sh -c 'grep -Hniv seed $0' {} \; |  sed 's/out\/results\///' | sed 's/\//,/;s/\//,/;s/\//,/;s/\//,/' | sed 's/:[[:digit:]]:/,/' | sed 's/,[[:digit:]]\+.Run.log//' | sed 's/*Run.//' >> "+ f"out/{dir}/{resFile}" )

    # We remove from the results file the 
    results = pd.read_csv(f"out/{dir}/{resFile}")
    results.drop(results.columns[4], axis=1, inplace=True)

    for approach in cfg.GetCodesExclude():
        results.drop( results[results['Method'] == approach].index, inplace=True)

    for dataset in cfg.GetDatasExclude():
        results.drop( results[results['Dataset'] == dataset].index, inplace=True)

    # To be careful with this - only when a known issue causes Inf or NaN
    #results = results[~results['Train MSE'].isin([np.nan, np.inf, -np.inf])]
    #results = results[~results['Test MSE'].isin([np.nan, np.inf, -np.inf])]

    results.to_csv( f"out/{dir}/{resFile}", index=False)

def SummaryToCsv(dir = 'summary', resFile = 'results.csv', sumFile = 'summary.csv'):
    
    """ Create both a total summary and a per-dataset CSV containing the results CSV with average MSE scores & algorithm ranks for all methods and datasets """

    # Read In Summary Data
    df = pd.read_csv( f"out/{dir}/{resFile}", index_col=False )
    data = df.to_numpy()

    # Get list of datasets and methods
    methods = np.unique(data[:,0])          # unique datasets
    datasets = np.unique(data[:,1])         # unique datasets

    # Construct summary header
    header = ["Method", "Dataset", "Split", "Runs"]
    
    # Go through all datasets and methods
    summary = []
    do_header = True
    for dataset in datasets:

        for method in methods:

            for split in cfg.splits:

                # Filter the results for the current dataset and method
                subset = data[(data[:,0] == method) & (data[:,1] == dataset) & (data[:,3] == split)]

                runs = len(subset[:,0])

                # Calculate average duration
                #avgDur = np.average(subset[:,3].astype(float))

                # Calculate the N averages after the 'Objective' column
                averages = []
                for i in range(4, len(df.columns)):

                    if df.columns[i] != "Model":

                        #averages.append(np.average(subset[:,i].astype(float)))
                        averages.append(np.median(subset[:,i].astype(float)))
                        #if(len(subset) > 0):   averages.append(np.amin(subset[:,i].astype(float)))
                        #else:                  averages.append(0)
                            
                        if do_header:
                            header.append(df.columns[i])

                do_header = False

                # Construct the result object with the averages
                result = [
                    method,
                    dataset,
                    split,
                    runs
                ] + averages
                summary.append(result)

    # Output the data to file
    #summaryOut = np.insert(allSummary, 0, , axis=0)
    df = pd.DataFrame(data=summary)
    df.to_csv(f'out/{dir}/{sumFile}', index=False, header=header) 

def SummaryRankCsv(dir = 'summary', resFile = 'results.csv', sumFile = 'summary.csv'):

    results = pd.read_csv(f"out/{dir}/{sumFile}")
    new_results = pd.DataFrame()

    # Get list of datasets and methods
    datasets = np.unique(results['Dataset'].to_numpy())         # unique datasets

    for dataset in datasets:

        for split in cfg.splits:
        
            ds_results = results[ (results['Dataset'] == dataset) & (results['Split'] == split) ]
            ds_results.insert( loc = len(ds_results.columns), column='Train Rank', value = ss.rankdata(ds_results['Train MSE']) )
            ds_results.insert( loc = len(ds_results.columns), column='Test Rank', value = ss.rankdata(ds_results['Test MSE']) )
            new_results = pd.concat([new_results,ds_results])
        
        new_results.to_csv(f"out/{dir}/{sumFile}", index=False) 

def ViolinPlot(dir = 'summary', resFile = 'results.csv', sumFile = 'summary.csv'):

    df = pd.read_csv( f"out/{dir}/{sumFile}", index_col=False )

    approaches, datasets, trainAvg, testAvg = AverageRank(df)
        
    for split in cfg.splits:

        data = df[ df['Split'] == split ]
        
        plt.clf()

        idx = np.argsort(trainAvg)
        orderedApproaches = approaches[idx]
        top = [approaches[i] for i in idx]
        data = data[ data["Method"].isin(top[:7]) ]
        

        font = {'family' : 'normal',
                'size'   : 30}
        plt.rc('font', **font)
        plt.xticks(rotation=75)

        ax = sb.violinplot(x="Method", y="Train Rank", data=data, order=orderedApproaches, scale='width', cut=1)
        fig = ax.get_figure()
        fig.set_size_inches(30, 15)
        plt.tight_layout()
        fig.savefig(f"out/{dir}/VP.Train.{split}.png")

        plt.clf()

        data = df[ df['Split'] == split ]

        idx = np.argsort(testAvg)
        orderedApproaches = approaches[idx]
        top = [approaches[i] for i in idx]
        data = data[ data["Method"].isin(top[:7]) ]

        font = {'family' : 'normal',
                'size'   : 30}
        plt.rc('font', **font)
        plt.xticks(rotation=75)

        ax = sb.violinplot(x="Method", y="Test Rank", data=data, order=orderedApproaches, scale='width', cut=1)
        fig = ax.get_figure()
        fig.set_size_inches(30, 15)
        plt.tight_layout()
        fig.savefig(f"out/{dir}/VP.Test.{split}.png")

    return

def SummaryImages(firstOnly = False):
    ''' Generates PDFs for all iterations of the of a particular datasets. The result of each code used will be
    appended and outputed to the summary/{dataName}/ folder
    '''

    '''
    for dataLine in self.getDatas():

        dataName, dataIters, dataTest, dataTrain, isDataComparative, _, minY, maxY, minX, maxX = self.stripData(dataLine)

        for i in range(0,dataIters):

            if i >= 1 and firstOnly: return

            Path(f"{self.PWD}/summary").mkdir(parents=True, exist_ok=True)

            os.system(  f"convert results/*/{dataName}/Run.{i:05}.Evole.png summary/{dataName}/Evole.{i:05}.pdf;"+
                        f"convert results/*/{dataName}/Run.{i:05}.ModelsRelative.png summary/{dataName}/Models.{i:05}.Relative.pdf;"+
                        f"convert results/*/{dataName}/Run.{i:05}.ModelsXbased.png summary/{dataName}/Models.{i:05}.Xbased.pdf;")
    '''

def SummaryModelComparison():
    ''' Generate a comparison of the different methods in a single comparative graph grid.
    Output will be generated for comparative analysis by default as this compares YPred against YAct
    and is relevant for every dataset. For those datasets marked as x-based, a 
    '''

    # Read In Summary Data
    df = pd.read_csv( f"{cfg.outSumDir}/results.csv", index_col=False )
    
    seeds = df['Seed'].unique()
    datas = df['Dataset'].unique()
    algs = df['Method'].unique()
    splits = df['Split'].unique()

    for seed in seeds:

        for dataset in datas:

            scaler = None
            cols = None
            mse = None
            mseNorm = None

            for split in splits:

                # Train Data
                dfTrain = pd.read_csv( f"{cfg.outResDir}/data/{dataset}/{seed}/{split}/{seed}.Train.csv", index_col=False )
                dfTrain.sort_values(by=['y'], inplace=True, ignore_index=True)

                if "norm" not in split:

                    plt.figure(figsize=(15,15))
                    plt.scatter(dfTrain.index, dfTrain['y'], label='True', alpha=0.5)
                    cols = dfTrain.columns
                    scaler = MinMaxScaler()
                    dfTrain = pd.DataFrame( scaler.fit_transform(dfTrain), columns = cols)
                    

                for alg in algs:

                    with open(f"{cfg.outResDir}/{alg}/{dataset}/{seed}/{split}/{seed}.Sol.json") as file:

                        solJson = json.load(file)
                        expr = solJson["equations"]["Numpy"]
                    
                        for column in reversed(dfTrain.columns):
                            expr = expr.replace(f"*{column}", f"*dfTrain.{column}")

                        origY = dfTrain['y']
                        dfTrain = pd.eval( f"y = {expr}", target=dfTrain, engine='python')

                        if "norm" in split:
                            dfTrain = pd.DataFrame( scaler.inverse_transform(dfTrain), columns = dfTrain.columns)                            
                            diff = origY-dfTrain['y']
                            diff2 = diff*diff
                            mseNorm = diff2.mean()
                            plt.scatter(dfTrain.index, dfTrain['y'], label=f"Model Norm {round(mseNorm,3)}", alpha=0.5)
                            
                        else:
                            diff = origY-dfTrain['y']
                            diff2 = diff*diff
                            mse = diff2.mean()
                            plt.scatter(dfTrain.index, dfTrain['y'], label=f"Model {round(mse,3)}", alpha=0.5)

                        plt.legend(fontsize=18)
        
            dir = f"{cfg.outResDir}/{dataset}"
            Path(dir).mkdir(parents=True, exist_ok=True)      
            plt.savefig( f"{dir}/{seed}.Train.png")
            plt.close()
            print(f"{dataset},{mse},{mseNorm}")

            # Test Data
            #dfTest = pd.read_csv( f"{cfg.outResDir}/data/{dataset}/{seed}/{split}/{seed}.Test.csv", index_col=False )
            #dfTest = pd.eval( f"m = {expr}", target=dfTest)
            '''
            for alg in algs:

                with open(f"{cfg.outResDir}/{alg}/{dataset}/{seed}/{split}/{seed}.Sol.json") as file:
                    
                    solJson = json.load(file)
                    expr = solJson["equations"]["Numpy"]
                
                    for column in reversed(dfTest.columns):
                        expr = expr.replace(f"*{column}", f"*dfTrain.{column}")
                    
                    dfTest = pd.eval( f"m = {expr}", target=dfTest)
                    plt.plot(dfTest.index, dfTest['m'])

            plt.savefig( f"{dir}/{seed}.Test.png")
            plt.close()        
            '''
            
    '''
    dSet = 10000
    i = 0
    iSet = 0
    n = 20       # max number of models on each plot

    for dataLine in cfg.GetDatas():

        dataName, dataIters, dataTest, dataTrain, isDataComparative, _, minY, maxY, minX, maxX = cfg.stripData(dataLine)    
        dataDir = f"out/results/data/{dataName}"

        for i in range(0,dataIters):

            ind = str(i).zfill(5)
            outDataTest = f"{dataDir}/Test.{ind}.csv"
            outDataTrain = f"{dataDir}/Train.{ind}.csv"

            XTrain, YTrain, N, dYTrain, WTrain, _ = Reader.CsvDatasetFile(outDataTrain)
            XTest, YTest, _, dYTest, WTest, _ = Reader.CsvDatasetFile(outDataTest)

            gridType = 'line'

            if isDataComparative:
                XTrain = YTrain
                XTest = YTest
                gridType = 'scatter'

            if XTrain.ndim > 1:     XTrain = XTrain[:,0]
            if XTest.ndim > 1:      XTest = XTest[:,0]

            outFolder = f"out/summary/{dataName}/"

            YTrains = []
            YTests = []
            Headings = []

            for codeLine in cfg.getCodes():
            
                cmdName = codeLine[0]               
                readFolder = f"out/results/{cmdName}/{dataName}/"
                YTrainPred, _ = Reader.CsvFile(readFolder+f"Run.{ind}.Train.Predict.csv")
                YTestPred, _ = Reader.CsvFile(readFolder+f"Run.{ind}.Test.Predict.csv")

                if YTrainPred is None or YTestPred is None: continue

                YTrainPred = YTrainPred.flatten()
                YTestPred = YTestPred.flatten()
                YTrains.append(YTrainPred)
                YTests.append(YTestPred)
                Headings.append(cmdName)

                # break
                i += 1
                if i == n:
                    
                    Plot.TestAndTrainGrid(outFolder+f"Run.{ind}.ModelsRelative", XTrain, YTrain, YTrains, XTest, YTest, YTests, dataName, Headings, isRelative=True, minY= minY, maxY=maxY, minX=minX, maxX=maxX)
                    if not isDataComparative:
                        Plot.TestAndTrainGrid(outFolder+f"Run.{ind}.ModelsXbased", XTrain, YTrain, YTrains, XTest, YTest, YTests, dataName, Headings, isRelative=False, minY= minY, maxY=maxY, minX=minX, maxX=maxX)

                    i = 0
                    iSet += 1
                    YTrains = []
                    YTests = []
                    Headings = []

            # last plot set
            Plot.TestAndTrainGrid(outFolder+f"Run.{ind}.ModelsRelative", XTrain, YTrain, YTrains, XTest, YTest, YTests, dataName, Headings, isRelative=True, minY= minY, maxY=maxY, minX=minX, maxX=maxX)
            if not isDataComparative:
                Plot.TestAndTrainGrid(outFolder+f"Run.{ind}.ModelsXbased", XTrain, YTrain, YTrains, XTest, YTest, YTests, dataName, Headings, isRelative=False, minY= minY, maxY=maxY, minX=minX, maxX=maxX)

            if dYTrain is not None:
                Plot.TestAndTrainErrors(outFolder+f"Run.{ind}.Errors", XTrain, YTrain, YTrains, dYTrain, XTest, YTest, YTests, dYTest, "Testing & Training Model Comparison", methods=Headings, isRelative=False, minY= minY, maxY=maxY, minX=minX, maxX=maxX)

            # Clear for next dataset
            i = 0
            iSet = 0
            dSet += 10000
    '''

def PvalHeatmap(dir = 'summary', sumFile = 'summary.csv'):

    """ Create heatmap for method p values """

    summary = pd.read_csv( f"out/{dir}/{sumFile}", index_col=False )
    approaches = summary['Method'].unique()
    datasets = summary['Dataset'].unique()

    if len(approaches) < 3: 
        print("Skipping Pval Heatmap due to <3 measuremetns")
        return

    trainRanks = np.zeros( (len(datasets),len(approaches)) )
    testRanks = np.zeros( (len(datasets),len(approaches)) )

    i = 0
    for approach in approaches:

        ds_temp = summary[ summary['Method'] == approach ]
        ds_train = pd.DataFrame(ds_temp['Train Rank'])
        ds_test = pd.DataFrame(ds_temp['Test Rank'])

        trainRanks[:,i] = ds_train.transpose().to_numpy()
        testRanks[:,i] = ds_test.transpose().to_numpy()
        i += 1

    # Training Rankings
    stat, p = ss.friedmanchisquare(*trainRanks.T)
    if p <= 0.05:

        plt.figure(np.random.randint(0,99999999), figsize=(20,20))
        pval_data = sp.posthoc_nemenyi_friedman(trainRanks.astype(int))
        pval_data.columns = approaches
        pval_data.index  = approaches

        cmap = ['1', '#FF0000',  '#90EE90',  '#3CB371', '#008000']
        heatmap_args = {'cmap': cmap,
                        'linewidths': 0.25, 
                        'linecolor': '0.5', 
                        'clip_on':False, 
                        'square':True, 
                        'cbar_ax_bbox': [0.93, 0.35, 0.04, 0.3]                            
                        }
        sp.sign_plot(pval_data, **heatmap_args)
        plt.savefig( f'out/{dir}/PV.Train.png', bbox_inches='tight')
    else:   print('!!!!! No significant differences, skipping Training PvalHeatmap')
    
    # Testing Rankings
    stat, p = ss.friedmanchisquare(*testRanks.T)
    if p <= 0.05:
    
        plt.figure(np.random.randint(0,99999999), figsize=(20,20))
        pval_data = sp.posthoc_nemenyi_friedman(testRanks.astype(int))
        pval_data.columns = approaches
        pval_data.index  = approaches

        cmap = ['1', '#FF0000',  '#90EE90',  '#3CB371', '#008000']
        heatmap_args = {'cmap': cmap,
                        'linewidths': 0.25, 
                        'linecolor': '0.5', 
                        'clip_on':False, 
                        'square':True, 
                        'cbar_ax_bbox': [0.93, 0.35, 0.04, 0.3]
                    }
        sp.sign_plot(pval_data, **heatmap_args)
        plt.savefig( f'out/{dir}/PV.Test.png', bbox_inches='tight')
    else:   print('!!!!! No significant differences, skipping Testing PvalHeatmap')

def CritcalDifference(dir = 'summary', sumFile = 'summary.csv'):
    """ Create a critical difference plot for all methods based on ranking. This requires a minimum number of methods to be used (~4) """

    summary = pd.read_csv( f"out/{dir}/{sumFile}", index_col=False )
    
    approaches, datasets, trainAvg, testAvg = AverageRank(summary)

    if len(approaches) > 21:
        print("!!!!! Attempting CD Plot on > 21 methods, current package cannot handle this")
        return

    if len(approaches) < 3:
        print("!!!!! Attempting CD Plot on < 3 methods, current package cannot handle this")
        return

    cd_val = Orange.evaluation.compute_CD(trainAvg, len(datasets), alpha='0.1')
    Orange.evaluation.graph_ranks(trainAvg, approaches, cd=cd_val, width=8, textspace=1.5)
    plt.savefig( f'out/{dir}/CD.Train.png', bbox_inches='tight')

    cd_val = Orange.evaluation.compute_CD(testAvg, len(datasets), alpha='0.1')
    Orange.evaluation.graph_ranks(testAvg, approaches, cd=cd_val, width=8, textspace=1.5)
    plt.savefig( f'out/{dir}/CD.Test.png', bbox_inches='tight')

def AverageRank( summary):

    approaches = summary['Method'].unique()
    datasets = summary['Dataset'].unique()

    trainAvg = []
    testAvg = []

    for approach in approaches:

        ds_temp = summary[ summary['Method'] == approach ]
        trainAvg.append( ds_temp['Train Rank'].mean() )
        testAvg.append( ds_temp['Test Rank'].mean() )
    
    return approaches, datasets, trainAvg, testAvg

if __name__ == '__main__':

    startTime = time.time()

    ResultsToCsv()              # Build summary/results.csv
    SummaryToCsv()              # Build summary/summary.csv
    SummaryRankCsv()            # Add train and test rankings for all
    #SummaryModelComparison()    # Plot the different methods on the the same problem
    #CritcalDifference()   
    #PvalHeatmap()
    ViolinPlot()

    # memetic specific
    #plot_agent_improve()

    endTime = time.time()
    duration = endTime-startTime
    print("Time Elapsed :"+str(round(duration,2))+"s ("+str(round((duration)/60))+"mins)")
