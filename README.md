# Fast approximations of 1-Wasserstein distance for persistence diagrams

This repository contains two algorithms - modified L<sub>1</sub> embedding and modified flowtree - that approximate the 1-Wasserstein distance for persistence diagrams. The associated paper can be found [here](https://arxiv.org/abs/2104.07710).
They are based on the Quadtree (Indyk, Thaper 2003) and Flowtree (Backurs, Dong, Indyk, Razenshteyn, Wagner, 2019) algorithms developed for standard optimal transport. 
As such, much of the code in this repository is adapted from code provided alongside the Flowtree paper [here](https://github.com/ilyaraz/ot_estimators). 

## Installation instructions
The installation instructions are the same as provided in the [original repository](https://github.com/ilyaraz/ot_estimators) but modified here for this specific package.
Start by cloning the repo with all submodules.

    git clone --recurse-submodules https://github.com/chens5/w1estimators
    
### Ubuntu
Install all necessary dependencies:

    sudo apt-get install -y g++ make cmake python3-dev python3-pip python3-numpy
    sudo pip3 install cython pot

Then to build the Python wrapper:

    $ mkdir build
    $ cd build
    $ cmake ../native
    $ make
Copy the resulting `.so` file to the `"python"` directory in this repository.

## Running the example code
The `"python"` directory contains a file `example.py` contains an example of how to use the implementations of both the modified L<sub>1</sub> embedding algorithm and the modified flowtree algorithm and evaluates each algorithms relative error on sample synthetic data. Additionally, the directory contains a set of sample synthetic persistence diagrams in `sample_data`. Each point, p, in a persistence diagram is generated by sampling p.x from a uniform distribution from 0 to 200 and sampling p.y from a uniform distribution from x to 300.  Note that to run `example.py`, we require the [GUDHI library](https://gudhi.inria.fr/) since calculating the exact 1-Wasserstein uses the implementation of Hera provided in GUDHI. Run `example.py` as follows.

    python3 example.py sample_data 
    
