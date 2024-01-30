
# GECCO-23 Adaptive Depth for Analyic Continued Fractions

This repository defines the code and reproduction steps for manuscript [Adaptive Depth for Analytic Continued Fraction Regression]() accpeted for presentation at GECCO 23'.

The software used in this paper is archived here, while the active code base is found at [memetico](https://github.com/impvau/memetico) which includes improvements, GPU capabilities (CUDA) and a testing suite. For the proposes of repoduction, refer to the old code in `codes/memetico`.

Please cite as
```
@inproceedings{Moscato:2023,
    author      = {Moscato, Pablo and Ciezak, Andrew and Noman, Nasimul},
    title       = {Dynamic Depth for Analytic Continued Fraction Regression},
    booktitle   = {Proceedings of the 2023 Annual Conference on Genetic and Evolutionary Computation},
    series      = {GECCO '23},
    year        = {2023},
    location    = {Lisbon, Portugal},
    pages       = {520--528},
    numpages    = {10},
    url         = {https://doi.org/10.1145/3583131.3590461},
    doi         = {10.1145/3583131.3590461},
    acmid       = {13217.117},
    isbn        = {979-8-4007-0119-1/23/07},
    publisher   = {ACM},
    address     = {New York, NY, USA},
}
```

The reference to example model output is provided here:

| Method | Output |
|--------|--------|
| adm | $(0.9206*x-0.08206)+\dfrac{0.05078*x+0.1398}{-0.8215*x+0.77339}$ |
| dsr | $x*(x + sin(x*(x**5 + x))) + x$ |
| ite | $0.004,-1.3304 + 1.5079*np.exp(x[:,0])$ |
| psr | $Add(Mul(Float('1.3293'), Symbol('x')), Float('0.4804'))$ |
| pst | $x^2*(x+0.8064-\dfrac{0.01074}{(x-0.9330)})+x+0.1148$ |
| ffx | $ -0.144+26.0*max(0,x-0.708)-8.51*abs(x)* max(0,x-0.708)-8.41*max(0,x-0.449)* max(0,x-0.708)-8.29*max(0,x-0.449)* abs(x)-8.20*x**2+6.93*abs(x)+6.72* $  ... (continues for a total of 448 characters)
| sbp | $-703098406036.035400+0.066215*(((((((((x+1.359000)-cos(x)) aqcos((3.427000aqx))) *1.245786)+((-0.038240*(plog(cos(((x*3.754000)-cos(4.183000)))) $ ... (continues for a total of 3165 characters) |

# Pre-requisites

Software
- Install [Docker Desktop](https://www.docker.com/)
- Install [VSCode](https://code.visualstudio.com/)
- Install VSCode Extension: Dev Containers (id: ms-vscode-remote.remote-containers)
- Install VSCode Extension: Docker (id: ms-azuretools.vscode-docker)

## Download Containers

To download containers, after install docker, use `docker pull` on the command line (e.g. Powershell, WSL, Ubuntu terminal). The following containers relate to this publication:

<!---
# Experiements for comparison of six SR Bench methods against adm with exact seeds in publication
docker pull impvsol.azurecr.io/gecco23-e3-srbench-exact
-->

```
# Experiements for comparison of dynamic depth and static depths of 0, 2 and 4
docker pull impvsol.azurecr.io/gecco23-e1-depth             

# Experiements for comparison of adp, adm & adr approaches
docker pull impvsol.azurecr.io/gecco23-e2-adp-types

# Experiements for comparison of six SR Bench methods against adm with new random seeds
docker pull impvsol.azurecr.io/gecco23-e3-srbench

# A containerised version of memetico that facilitates step-by-step debugging for a specific program run
docker pull impvsol.azurecr.io/gecco-23-memeitco
```

## Run Container and Access Terminal

To run a container, after pulling the container, use `docker run` in the following manner to start and connect to a terminal in the container
```
docker run -it --entrypoint /bin/bash -v /path/on/host:/workspaces/out impvsol.azurecr.io/gecco23-e1-depth
```
Where you
- Substitute the last argumnet with the desired container name after ensuring that the image has been pulled
- Substitute `/path/on/host` to the location on the host that you wish to store the results
- Note 
    - We override the default entrypoint that is designed for connecting to our network storage from within a Kubernetes cluster for HPC compute, to a simple bash shell with the `--entrypoint` command
    - You should be able to perform the same steps within Docker Desktop IDE after you have run the `docker pull` command

## Connect to Running Container (IDE/Step-By-Step Debugging)

After running a container with terminal access, we can attach to this container in VS Code

Once the container is running you are able to access in VS Code via the command pallet:
```
F1
Select 'Dev Containers: Attach to a Running Container'
Select the randomly-generated container name 
# This can be seen in the Docker Desktop IDE -> Containers tab
# Also viewable by running `docker ps`
Select to open the folder /workspaces
```

For containers that are configured for debugging and development rather than HPC execution, there will be a .vscode file that allows you to specify the arguments to the function and press f5 to debug the code. e.g. `impvsol.azurecr.io/gecco23-e3-srbench-exact`


For containers that are configured for HPC execution, you can can re-run the experiements and explore some code, but the software packages may be compiled and there is no real benefit simply triggering the code from a terminal.

# Execution

After pulling and running the container, trigger the execution of the software on the containers commandline
```
python autoharn/execute.py
```

After the code has run, run the post-processing command (output details are discussed later)
```
# HPC containers not configured for post-processing, we need to install a few libraries first
pip install matplotlib scikit_posthocs Orange3

python autoharn/postproc.py
```

## Running Exp 1: Depth comparison
Experiment one is used to generate data for the comparison of different static depths and and adaptive depth approach. The container used for this experiement is `impvsol.azurecr.io/gecco23-e1-depth` however the code will select random seeds and results may vary.

```
docker pull impvsol.azurecr.io/gecco23-e1-depth             
docker run -it --entrypoint /bin/bash -v /home/impv/gecco-23-out-e1:/workspaces/out impvsol.azurecr.io/gecco23-e1-depth

# Run the following in the terminal that opens
python autoharn/execute.py

# After execution, perform post-processing
pip install matplotlib scikit_posthocs Orange3
python autoharn/postproc.py
```

## Running Exp 2: 
Experiement two is used to generate the data that compares the different adp, adm and adr approaches. The container used for this experiement is `impvsol.azurecr.io/gecco23-e2-adp-types` however the code will select random seeds and results may vary.

```
docker pull impvsol.azurecr.io/gecco23-e2-adp-types   
docker run -it --entrypoint /bin/bash -v /home/impv/gecco-23-out-e2:/workspaces/out impvsol.azurecr.io/gecco23-e2-adp-types

# Run the following in the terminal that opens
python autoharn/execute.py

# After execution, perform post-processing
pip install matplotlib scikit_posthocs Orange3
python autoharn/postproc.py
```

## Running Exp 3: 
Experiment three is used to generate the adm approach against all comparitors. The container used for this experiement is `impvsol.azurecr.io/gecco23-e3-adp-srbench` however the code will select random seeds and results may vary. For exact replication of see the next section

```
docker pull impvsol.azurecr.io/gecco23-e3-srbench 
docker run -it --entrypoint /bin/bash -v /home/impv/gecco-23-out-e3:/workspaces/out impvsol.azurecr.io/gecco23-e3-srbench

# Run the following in the terminal that opens
python autoharn/execute.py

# After execution, perform post-processing
pip install matplotlib scikit_posthocs Orange3
python autoharn/postproc.py
```

<!-- 
## Running Exact Replication of Exp 3:
The exact replication of experiement three uses the files in this directory. These have been built into the `impvsol.azurecr.io/gecco23-e3-adp-srbench-exact` container, which can be downloaded to save time. This is in contrast to rebuilding from the repository root directory with `docker build -t impvsol.azurecr.io/gecc23-e3-srbench-exact .` which will use the latest versions of the software packages and may become incompatible overtime.

Note that the `autoharn` wrapper-code and `/workspaces/GECCO-23/autoharn/config/seed.txt` is slightly modified to faciltate replication, and changes to these files can lead to results misaligning.

```
docker pull impvsol.azurecr.io/gecco23-e3-srbench
docker run -it --entrypoint /bin/bash -v /home/impv/gecco-23-out-e3-exact:/workspaces/out impvsol.azurecr.io/gecco23-e3-srbench-exact

# Run the following in the terminal that opens
python autoharn/execute.py

# After execution, perform post-processing
pip install matplotlib scikit_posthocs Orange3
python autoharn/postproc.py
```
-->

The experiment will run against `Nyugen1` dataset for 100 iterations. You will need to re-execute after changing `/workspaces/autoharn/config/settings.py` such that `datasFile` is the name of the data file in `/workspaces/autoharn/config` that should be executed against. A file exists for each of the 21 Nguyen datasets. e.g.

```
...
codesExclFile   = f"{configDir}/code_excl.txt"      # List of approaches
datasFile       = f"{configDir}/data_Nguyen2.txt"   # List of datasets
datasExclFile   = f"{configDir}/data_excl.txt"      # List of datasets
...
```

## Inspecting Results

Results from the execution will be placed in the `/workspaces/GECCO-23/out/` directory.
- `/workspaces/GECCO-23/out/results/data` contains a copy of the data used by the algorithm
- `/workspaces/GECCO-23/out/results` will contain a folder for each algorithm
    - Each algorithm folder will contain a folder for each dataset
    - Each dataset folder will contain a folder for each seed
    - Each seed folder will contain folders for each type of split (e.g. 75/25, 80/20, manual)
    - Each split folder will contain
        - Error output of the algorithm in the file `<seed>.Err.log`
        - Standard output of the algorithm `<seed>.Std.log`
        - Result summmary in `<seed>.Run.log` (which is used to concatenated with its analogs for different seeds/splits/algorithms to aggregate results)
        - Copy of the train and test results in `<seed>.Train.csv` and `<seed>.Test.csv`
        - Optionally the predictions of the final solution on the train and test results in `<seed>.TrainResults.csv` and `<seed>.TestResults.csv`

To trigger the post-processing after execution run the following command
```
python /workspaces/GECCO-23/autoharn/postproc.py
```

This process aggregates the data from the `<seed>.Run.log` outputs, performs statistical analysis and generates results in `/workspaces/GECCO-23/out/summary`
- `/workspaces/GECCO-23/out/summary/results.csv` is the concatenation of all results
- `/workspaces/GECCO-23/out/summary/summary.csv` is the statistical averages
- Alternate outputs such as the CD plots may be generated

Note additional software has been used in addition to `postproc.py` to general all plots.

## Step-by-step Debug

Step-by-step debug occurs for a specific algorithm without the `autoharn` software which calls multiple different codebases. A container for the `memetico` software has been made available by pulling `impvsol.azurecr.io/gecco-23-memeitco`.

After running, you can attach to the container in VS Code and you will need to install the `ms-vscode.cpptools` extension in the container (C/C++ by Microsoft)

We will need to specify the `args` within the `.vscode/launch.json` to the specific arguments we want to pass into the software, e.g. 

```
"args": [
    "-t", "data/2774136.Train.csv",
    "-T", "data/2774136.Test.csv",
    "-mr", "0.2",
    "-d", "0",
    "-g", "200",
    "-sd", "",
    "-ld", "0.2",
    "-st", "",
    "-mt", "600",
    "-s", "2774136",
    "-lt", "",
    "-ls", "cnm",
    "-o", "",
    "-f", "5",
    "-dc", "5",
    "-dd", "adp-mu",
    "-dm", "stale-ext",
],
```

To debug an execution that is within the presented results, view the `<seed>.Std.log` file that is generated after running `pyhton autoharn/execute.py` output as you can copy and paste the args output at the top of the file e.g.
```
==================
Arguments
==================
Command: codes/memetico/bin/main
"args": [
    "-t", "out/results/data/Nguyen1.TrainNoise/2774136/manual/2774136.Train.csv",
    "-T", "out/results/data/Nguyen1.TrainNoise/2774136/manual/2774136.Test.csv",
    "-mr", "0.2",
    "-d", "0",
    "-g", "200",
    "-sd", "",
    "-ld", "0.2",
    "-st", "",
    "-mt", "600",
    "-s", "2774136",
    "-lt", "out/results/adm/Nguyen1.TrainNoise/2774136/manual",
    "-ls", "cnm",
    "-o", "",
    "-f", "5",
    "-dc", "5",
    "-dd", "adp-mu",
    "-dm", "stale-ext",
],
```

You should copy the files mentioned in `-t` and `-T` to a folder within the `memetico` folder, such as `data` folder and updated the args as follows. Note we remove the `-lt` argumnet so that the output is not redirected to a file.
```
"args": [
    ...
    "-t", "${workspaceFolder}/data/2774136.Train.csv",
    "-T", "${workspaceFolder}/data/2774136.Test.csv",
    ...
    "-lt", "",
    ...
],
```

In the 'Run and Debug' tab in VS Code, ensure that the 'CFR Launch' task is selected, and then run the code with F5. You can place debug points within the code as per normal with VS Code.

Note that the code is currently configured to replicate the 2774136 seed on the Nguyen1 datasets. To copy specific datasets to the container, cut and paste into a file in the data directory (and update the reference in the `launch.json`), else you can transfer files through the directory you mount between the host and the container with `-v` in the `docker run ` command

### Making Code Changes

The container has the binaries already compiled, but if changes are made, you will need to run the following command from the root project directory.
```
make clean; make
```

Note that the F5 command to run already has the above as task that is run before the code is executed