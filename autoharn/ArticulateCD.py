
import pandas as pd                         # data/csv arrays and funcs
import numpy as np                          # list manipulation
import Orange                               # critical diff plots
import matplotlib.pyplot as plt             # critical diff plots

resultsHeader = ['method', 'dataset', 'run', 'seed', 'train MSE', 'test MSE', 'dur']        
summaryHeader = ['method', 'dataset', 'train MSE', 'test MSE', 'dur', 'runs', 'trainRank', 'testRank']

def CriticalDifference( dir = 'summary', sumFile = 'summary.csv'):

    """ Create critical difference plots """

    df = pd.read_csv( f"{dir}/{sumFile}", index_col=False )
    data = df.to_numpy()

    methods = np.unique(data[:,0])          # unique methods
    datasets = np.unique(data[:,1])         # unique datasets

    avgTrainRanks = []
    avgTestRanks = []
    for method in methods:

        subset = data[ data[:,0] == method]
        avgTrainRank = np.average(subset[:,6].astype(int))
        avgTestRank = np.average(subset[:,7].astype(int))

        avgTrainRanks.append(avgTrainRank)
        avgTestRanks.append(avgTestRank)

    if len(methods) > 21:
        print("!!!!! Attempting CD Plot on > 21 methods, current package cannot handle this")
        return

    avgTrainRanks = np.array(avgTrainRanks)
    cd_val = Orange.evaluation.compute_CD(avgTrainRanks, len(datasets), alpha='0.1')
    Orange.evaluation.graph_ranks(avgTrainRanks, methods, cd=cd_val, width=8, textspace=1.5)
    plt.savefig( f'{dir}/Train-CD.png', bbox_inches='tight')

    avgTestRanks = np.array(avgTestRanks)
    cd_val = Orange.evaluation.compute_CD(avgTestRanks, len(datasets), alpha='0.1')
    Orange.evaluation.graph_ranks(avgTrainRanks, methods, cd=cd_val, width=8, textspace=1.5)
    plt.savefig( f'{dir}/Test-CD.png', bbox_inches='tight')
    