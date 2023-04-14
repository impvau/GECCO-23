#!/bin/bash

# Config
output=charlie/220103-test
memoryProfile=false

# Mount network share
mkdir /workspace/remote
mount -t cifs -o username=ubuntu,password=ubuntu //10.0.0.98/mlab /workspace/remote
#mount -t cifs //10.0.0.98/mlab/ /workspace/remote -o user=mlabber,pass="Yi6M0y9TIG657xJe1@H&",noauto,user,file_mode=0777,dir_mode=0777

# Create output directory
mkdir -p /workspace/remote/$output

# Direct program output (/workspace/out) to network share
ln -s /workspace/remote/$output /workspace/out

# If we need to profile memory usage
if $memoryProfile
then

    # Ensure that the Docker file building this container has
    #COPY ./autoharn/memory.sh /workspace/run.sh
    #RUN pip install filprofiler

    # Run program with memeory profiler
    fil-profile run autoharn/execute.py

    # Move memory results to output mount
    mv fil-result out/fil-result

else

    # One plain execution coming up
    python autoharn/execute.py

fi
