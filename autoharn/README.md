
## GitHub

Configure GitHub

* Install [Git](https://git-scm.com/download/win) with default options
* Select to Launch Git Bash at the end of the install
* Run `ssh-keygen -o`
* Run `cat ~/.ssh/id_rsa.pub`
* Copy the SSH key to GitHub authorised SSH keys
* Run `ssh -T git@github.com` and observe an authentication response
* Setup the global account with your name and email
```
git config --global user.name "John Smith"
git config --global user.email "john.smith@common.com"
```

Configure Download Repository

* Open VSC 
* Select 'Git: Clone' from the Command Pallegit t (F1)
* Select the 'MTBTM' repository and clone to a local folder

Start the Development Container Environment

* Update the `mounts` section in [devcontainer.json](.devcontainer/devcontainer.json) with your windows user name
* Update the container environment for your GitHub account in the [Dockerfile](.devcontainer/Dockerfile) e.g.
```
RUN git config --global user.name "John Smith"
RUN git config --global user.email "john.smith@common.com"
```
* Add any packages you want in the container for the [Dockerfile](.devcontainer/Dockerfile)
* Select 'Remote-Container: Rebuild and Reopen in Containter' from the Command Pallet
* Rebuild the container with 'Remote-Container: Rebuild Container' from the Command Pallet

After this point
* Define the datasets in `config/data.txt`
* Define the algorithms in `config/code.txt`
* Download the algorithms into `codes` with the setup script `codes/getcodes.sh`
* Run the 'Python: Launch Suite' Task with F5
* Profit


# General Overview

The harness has the following main sections

Item      | Details
----------|--------
launch.py | Main lunch file for execution
launch.sh | HPC bootstrap script
codes/    | Codebases for execution against the datasets
datas/    | Datasets for computation against the codebases
config/   | Configurations specifying parameters of the execution, such as which datasets and codes to execute
Articulate.py| Primary summary codefile
ArticulateCD.py| A secondary file for generating Critical Difference plots which cannot execute on the HPC and must be segregated into a separate file

# Requirements

## Configuration
* `python` command links to the required python version `ln -s /bin/usr/python3 /bin/usr/python`
* Helpful bash aliases (append to ~/.bashrc)
    ```
        alias res='grep -n '\'''\'' results/*/*/Run*log | grep -v seed'
        alias rex="echo 'test,dataset,run,seed,train mse,test mse,dur(s)' > rex.csv; res | sed 's/results\///' | sed 's/\//,/g' | sed 's/:[[:digit:]]:/,/' | sed 's/.log//' | sed 's/Run.//' >> rex.csv; cat rex.csv"
        alias std='grep -n '\'''\'' results/*/*/Out*std*log'
        alias err='grep -n '\'''\'' results/*/*/Out*err*log | grep -v ConvergenceWarning | grep -v coordinate_descent '
        alias hpc-check='. ~/code/mtbtm/hpc-check.sh'
        alias hpc-run='. ~/code/mtbtm/hpc-run.sh'
        alias hpc-sync='. ~/code/mtbtm/hpc-sync.sh'
    ```
* Image processing packages

    ```
        sudo apt install imagemagick-6.q16
        sudo vi /etc/ImageMagick-6/policy.xml

            Comment out
                <policy domain="coder" rights="none" pattern="PDF" />
            As
                <!-- <policy domain="coder" rights="none" pattern="PDF" /> -->

    ```
* ~/.ssh/config defining hpc host and user account
    ```
    Host hpc
        HostName rcglogin.newcastle.edu.au
        User c3128697
    ```
* ssh keys configured for password-less access to the hpc (check that ``ssh hpc`` connects without prompts)

## Launch specifications

The two primary files that regulate the execution are ``config/data.txt`` and ``config/code.txt``

### config/code.txt
```
    #
    # Run config specifying the codes executables which must implement the arguments
    #
    #  -log     - Used for outputting the standard execution output
    #  -t       - Used for specifying the training data (as per the run config below)
    #  -T       - Used for specifying the testing data (as per the run config below)
    #
    # Configs take the form 'name, type, trigger' where
    #
    #  name indicates a uniuqe identifier that is used as the output folder name in results/
    #  type indicates the software type (unused at the moment)
    #  trigger indicates the full execution command line instruction, with arguments but excluding the required above (-log, -t, -T)
    #
    # Example of the CFR codes to explore a variety of parameters
    #
    #cfr2.5-std,c++,./codes/cfr2.5-develop/bin/main -g 20 -d 0.05 -m 0.1 -f 6 -o mse -w 20 -r 5 -nm 4:250:10 -v 1
    #cfr2.5-agressive,c++,./codes/cfr2.5-develop/bin/main -g 20 -d 0.05 -m 0.1 -f 6 -o mse -w 80 -r 5 -nm 4:250:10 -v 1
    #cfr2.5-extensive,c++,./codes/cfr2.5-develop/bin/main -g 200 -d 0.05 -m 0.1 -f 6 -o mse -w 20 -r 5 -nm 4:250:10 -v 1
    #
    #
    # The standing ML methods, encpasulated in the 'pymodels' module. As the methods have a fit and predict function,
    # the same codes can be used to execute all the different models. Similarly, should we implement a python-based regressor# we are able to implement the same fit and predict interface and use the same harness (see later)
    #
    #ada,python,python ./codes/pymodels/Main.py -m "ada-b"
    #
    #
```

### config/data.txt
```
    #
    # Data config rows in form 'Name,Iters,Training,Testing,SetType,testsplit' where
    #
    #  Name is the identifier and used in the title of results/*/<Name>/ directory
    #  Iters is the number of iterations to run the dataset for. Typically it is the same across all datasets but we may wish to modify
    #  Training and testing is are the filepaths for the datasets
    #  SetType is either comparative or xbased. For xbased we display X against Y. For comparative we plot yAct against yPred
    #
    Ackley2,2,./datas/jamil-yang/Ackley2.csv,./datas/jamil-yang/Ackley2.csv, xbased, shuffle10
    Ackley3,2,./datas/jamil-yang/Ackley3.csv,./datas/jamil-yang/Ackley3.csv, xbased, shuffle10
    Adjiman,2,./datas/jamil-yang/Adjiman.csv,./datas/jamil-yang/Adjiman.csv, xbased, shuffle10
    Argon,2,./datas/element/argon-trn.csv,./datas/element/argon-Tst.csv, xbased, shuffle10
    SC-10-below,2,./datas/element/SuperC-10pct-below89-trn.csv,./datas/element/SuperC-10pct-above89-Tst.csv, comparative, shuffle10
    SC-10-below-rand,2,./datas/element/SuperC-10pct-rand-below89-trn.csv,./datas/element/SuperC-10pct-above89-Tst.csv, comparitive, shuffle10
    Penn1,2,./datas/penn/titanic.csv,./datas/penn/titanic.csv, comparitive, shuffle10
    Penn2,2,./datas/penn/banana.csv,./datas/penn/banana.csv, comparitive, shuffle10
```

# Execution

For HPC execution
```
    hpc-run         - copy latest codes to hpc and trigger execution
    hpc-check       - check the status for your account 
    hpc-sync        - sync the results to the local machine
```

For local execution or post-processing summary generation
```
    sudo python launch.py           // if using Visual Studio Code, simply press f5
```

Note package limitations cause post-processing to be completed on the users machine.
After running ``hpc-sync``, running the launch.py again on your local machine will generate summary results

Errors and standard output can be seen with ``std`` and ``err`` respectively. These aliases can be added to your HPC ~/.bashrc and then viewed their

## result/ Output
The standard result output is
```
    results/<Codebase>/                                                 - A folder for each codebase used
    results/<Codebase>/<Dataset>/                                       - A folder for each dataset run by the codebase
    results/<Codebase>/<Dataset>/Out.0000N.err.log                      - The error log file for the Nth iteration of the code
    results/<Codebase>/<Dataset>/Out.0000N.std.log                      - The standard out log file for the Nth iteration of the code
    results/<Codebase>/<Dataset>/Run.0000N.log                          - The standardised model output, see 'Adding Models' for specifics
    results/<Codebase>/<Dataset>/Run.0000N.<Train|Test>.Feature.csv     - Output of the features used (for verification)
    results/<Codebase>/<Dataset>/Run.0000N.<Train|Test>.Predict.csv     - Output of the predicted target values
    results/<Codebase>/<Dataset>/Run.0000N.<Train|Test>.png             - Performance graph of the traing or test model (X vs Y or YAct vs YPred)
```

Each model can implement custom output using as neccessary. For instance, the ranking 

## summary/ Output
The standard summary output is
```
    summary/<Train|Test>PvalHeatmap.png                                 - Heatmap of significance for all codebases
    summary/<Train|Test>CD.png                                          - Critical difference plot for all codebases
    summary/results.csv                                                 - Results for all codebases, datasets and iterations
    summary/summary.csv                                                 - Average results for all codebases and datasets (i.e. average over iterations)
    summary/<DataSet>/correlation.png                                   - Correlation matrix for the dataset features
    summary/<DataSet>/covariance.png                                    - Covariance matrix for the dataset features
    summary/<DataSet>/results.csv                                       - results filtered by the current dataset
    summary/<DataSet>/summary.csv                                       - summary filtered by the current dataset
    summary/<DataSet>/Run.00000.<Train|Test>.AllModelsN.png             - Model comparison for the first execution of the program. This shows 20 models per page, so multiple outputs when codebases > 20
    
    // To review these - currently not exporting them
    summary/<DataSet>Test.pdf                                           - All final test models for <DataSet>
    summary/<DataSet>Train.pdf                                          - All final train models for <DataSet>
    summary/<DataSet>TestEvolve.pdf                                     - Evolution of the test model for iterative methods (run 0)
    summary/<DataSet>Trainvolve.pdf                                     - Evolution of the test model for iterative methods (run 0)
```

# Adding Models

The codes directory contains a series of model codebases that are referenced in execution of the harness. The ``get_codes.sh`` exists for drawing down the latest codebases we have for execution. The codes must implement specific requirements, explicity

* -log     Used for outputting the standard execution output
* -t       Used for specifying the training data (as per the run config below)
* -T       Used for specifying the testing data (as per the run config below)

A library for python-based regressors has been implmented (pymodels) and should be for any python regressors

An example of the standardised model ouput is
```
    seed, trainErr, testErr, dur(s),model
    1703304326,66.29405,66.29405,0.151,
```

### Config

The config directory contains a series of settings for common execution. When not implementing a template, use ``config/datas.txt`` and ``contfig/codes.txt`` for custom configurations. For examples of either, see ``config/code_eg.txt`` and ``config/data_eg.txt``

### Datas

The datas directory contains a series of datasets utilised by the software. This directory can be linked to existing data elsewhere when using linux / WSL, and it is suggested a separate repo explicity for non-sensitive data is established

### Results

After execution of the launch script a results/ directory is created. This should be manually removed between executions to ensure a "clean" directory
The folder structure will be
```
    results/<Method>/<Dataset>/
```

Each iterations execution will exist under this folder and contain
```
    Out.<RunNo>.err.log             Error output from the application
    Out.<RunNo>.std.log             Standard output from the application
    Out.<RunNo>.log                 Output for post processing, correlating to the -log parameter
    Out.<RunNo>.*.png               Image output of models
```
