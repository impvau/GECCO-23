
import numpy as np
import socket                               # Hostname
from sklearn.preprocessing import MinMaxScaler

### Key Config
totalIters      = 1         # Number of times to execute each dataset and approach pair
isParallel      = "impv-pc-001" not in socket.gethostname()

print(f"IsParallel?: {isParallel}")

# Split Types, for topMM, bottomMM, NNextraMM the samples are ordered by ascending y
# Empty         If train and test files are provided manually
# shuffleNN     Select top NN% as training, (100-NN) as testing
# topMM         Select top MM% as testing, remaining as training
# bottomMM      Select bottom MM% as testing, reamining as training
# NNextrMM      Select bottom NN% and top MM% for testing, mid region as training
splits          = ['manual'  ]#,     'top20',    'bottom20'  ]        #, 'top20',    'bottom20', 'train',    'manual'    ]
ppNormalisation = [False        ]#,            True,       True        ]      #,       False,      False,      False,      False      ]
ppYNoise        = [ 0.1 ]
ppXNoise        = [ 0.1 ]
ppScaler        = MinMaxScaler() 
ppBinIters      = -1
ppAppendModel   = False

# General Config
projDir         = 'autoharn'
configDir       = f"{projDir}/config"
seedsFile       = f"{configDir}/seed.txt"           # List of seeds to split data and initials approaches with
codesFile       = f"{configDir}/code.txt"           # List of approaches
codesExclFile   = f"{configDir}/code_excl.txt"      # List of approaches
datasFile       = f"{configDir}/data_Nguyen1.txt"   # List of datasets
datasExclFile   = f"{configDir}/data_excl.txt"      # List of datasets
existingData    = False                             # Don't process data again, use data that has been dropped in dataDir
onlyExecute     = False
doSplit         = True                              # If the data requires splitting, rather than train and test being defined manually
outDir          = 'out'                             # Output files dir
outResDir       = f"{outDir}/results"               # Raw output files dir
outSumDir       = f"{outDir}/summary"               # Aggregated output dir
srcDataDir      = 'datas'                           # Source data dir
dstDataDir      = f"{outResDir}/data"               # Preprocessed data dir



# Preprocessing Config
ppHistogram     = 'histogram'
ppViolin        = 'varViolins'
ppCovariance    = 'covarMatrix'
ppCorrelation   = 'covarMatrix'
ppOutput = [ppHistogram, ppViolin, ppCovariance, ppCorrelation]

ppMissingData = False

# Data Configs
#minParams = 0
#maxParams = 50
#minSamples = 0
#maxSamples = 2000000

# Approach Configs
#category = ['Symbolic Regression'] 
#types = ['Genetic Approach'] 

def validConfigLine(line):
    ''' Determine if codeline is active. Inactive lines are empty or begin with a hash '''
    return line[0] != "#" and line[0] != '\n' and line[0].strip() != ''

def getCodes():
    ''' 
    We have a series of algorithms we wish to execute against in the form of

        algName, cmdAndArgs

    There will always be both entries for a valid line    

    Returns an array in the form [algName, cmdAndArgs]
    '''
    
    ret = []

    # Open the file and read
    with open(codesFile,"r") as file:

        # Loop through each line
        for codeLine in file:

            # Ensure the line is valid
            if not validConfigLine(codeLine): 
                continue

            # Remove any line ending
            codeLine = codeLine.rstrip("\n")

            # Split like CSV
            args = codeLine.split(",")
        
            if len(args) == 2:      ret.append(args)
            else:                   raise Exception(f"Data configuration in {codesFile} invalid.\nOn line containing {codeLine}. Expecting 'algName, cmdAndArgs'")
            
    return ret

def GetDatas():
    ''' 
    We have a series of datafiles we wish to execute against in the form of

        srcTrainFile, srcTestFile

    Normally the dataname is enough i.e. {cfg.srcDataDir}/{srcTrainFile}.csv and
    we perform splitting on the dataset, however often a train and test is supplied
    as a specific split has been applied by the dataset creator

    Returns an array in the form [srcTrainFile, srcTestFile] where srcTestFile may be ""
    '''
    
    ret = []

    # Open the file and read
    with open(datasFile,"r") as file:

        # Loop through each line
        for dataLine in file:

            # Ensure the line is valid
            if not validConfigLine(dataLine): 
                continue

            # Remove any line ending
            dataLine = dataLine.rstrip("\n")

            # Split like CSV
            args = dataLine.split(",")
        
            # If CSV is placed manually, allow it
            if len(args) >= 1: args[0] = args[0].replace(".csv", "")
            if len(args) == 2: args[1] = args[1].replace(".csv", "")

            # Note we use args[0].split('/')[-1] to handle the scenario where subfolders are used
            # The result is the "dataset" is the last part of the path

            # When only a single source data file
            if len(args) == 1:      
                ret.append([
                    args[0].split('/')[-1],
                    f"{srcDataDir}/{args[0]}.csv", 
                    ""
                ])

            # When train and test file provided
            elif len(args) == 2:    
                ret.append([
                    args[0].split('/')[-1],
                    f"{srcDataDir}/{args[0]}.csv", 
                    f"{srcDataDir}/{args[1]}.csv"
                ])

            # In all other cases
            else:                   
                raise Exception(f"Data configuration in {datasFile} invalid.\n On line containing {dataLine}. Expecting 'dataName, srcTrainFile, srcTestFile' or 'dataName'")
            
    return ret

def GetSeeds():
    '''
    We require unique seeds for every iteration of the application.
    There is a seed file, however if it is empty we must generate the
    seeds
    '''

    fileSeeds = []

    try:

        # Open the file and read
        with open(seedsFile,"r") as file:

            # Loop through each line
            for seedLine in file:

                # Ensure the line is valid
                if not validConfigLine(seedLine): 
                    continue

                # Remove any line ending
                seedLine.rstrip("\n")

                fileSeeds.append(int(seedLine.strip()))

        return fileSeeds

    except Exception as e:
        nothing = 1

    seeds = []

    # When parallel execution, we have no satisfactory way to track
    # which seeds have been used by other nodes. We always (for the moment)
    # want to have random seeds in this case
    if isParallel or len(fileSeeds) == 0:

        for _ in range(0, totalIters):
            seeds.append(np.random.randint(0,2**31-1))

        return seeds        


    # Otherwise, we make sure there is enough seeds in the seed file
    if len(fileSeeds) < totalIters:
        raise Exception(f"Seed file only has {len(seeds)} and we are executing for {totalIters} times")

    # Then return the fileSeeds!
    return fileSeeds


def GetDatasExclude():

    datasExcl = []

    # Open the file and read
    with open(datasExclFile,"r") as file:

        # Loop through each line
        for fileLine in file:

            # Ensure the line is valid
            if not validConfigLine(fileLine): 
                continue

            # Remove any line ending
            fileLine.rstrip("\n")

            datasExcl.append(fileLine.strip())

    return datasExcl

def GetCodesExclude():

    codesExcl = []

    # Open the file and read
    with open(codesExclFile,"r") as file:

        # Loop through each line
        for fileLine in file:

            # Ensure the line is valid
            if not validConfigLine(fileLine): 
                continue

            # Remove any line ending
            fileLine.rstrip("\n")

            codesExcl.append(fileLine.strip())

    return codesExcl
