
FROM ubuntu:latest

# Setup General
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y 'ppa:deadsnakes/ppa'
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y libgomp1        
RUN apt-get install -y nfs-common      
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN apt-get install -y pkg-config
RUN apt-get install -y wget

# Setup miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN ln -s ~/miniconda3/condabin/conda /usr/local/bin/conda

# ITEA
RUN wget -qO- https://get.haskellstack.org/ | sh
RUN apt-get install -y libgsl-dev
# MRGP
RUN apt-get install -y default-jre
# PySR
RUN cd ~; 
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.0-linux-x86_64.tar.gz
RUN tar xf julia-1.3.0-linux-x86_64.tar.gz
RUN ln -s ~/julia-1.3.0/bin/julia /usr/local/bin/julia
#RUN apt-get install -y julia
# Possible to remove
RUN apt-get install -y vim

# Python packages
RUN conda install -y numpy
RUN conda install -y pandas 
RUN conda install -y scikit-learn
RUN conda install -y astropy
RUN conda install -y sympy
RUN conda install -y PyYAML

RUN conda install -y matplotlib
RUN conda install -y -c conda-forge scikit-posthocs
RUN conda install -y -c conda-forge Orange3

# Require for GPGOMEA
RUN apt-get install -y liblapack-dev
RUN apt-get install -y libblas-dev
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libarmadillo-dev

# SRBench - comment out for dev
ADD codes/srbench-min/ /workspaces/codes/srbench-min/
WORKDIR /workspaces/codes/srbench-min/experiment/methods/src/
RUN ./dsr_install.sh
RUN ./psr_install.sh
RUN ./ellyn_install.sh
RUN ./ffx_install.sh
RUN ./pst_install.sh
RUN ./gpl_install.sh
RUN ./ite_install.sh
RUN ./sbp_install.sh

RUN conda init bash

# Memetico - comment out for dev
COPY codes/memetico/bin/ /workspaces/codes/memetico/bin

# Copy harness code - comment out for dev
COPY autoharn/ /workspaces/autoharn
# Copy data to container - comment out for dev
COPY datas/ /workspaces/datas

# Defailt directory
WORKDIR /workspaces

# Copy ENTRYPOINT from mlab-min to setup NFS share
ENTRYPOINT  [ "/bin/bash", "-c", \
              "/etc/init.d/rpcbind start && /etc/init.d/nfs-common start && mkdir /workspaces/remote && mount -t nfs 10.0.0.98:/volume1/mlab /workspaces/remote && mkdir -p /workspaces/remote/charlie/outfolder && ln -s /workspaces/remote/charlie/outfolder /workspaces/out && python autoharn/execute.py" ]

