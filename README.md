# Fast approximations of 1-Wasserstein distance for persistence diagrams

This repository contains two algorithms - modified L<sub>1</sub> embedding and modified flowtree - that approximate the 1-Wasserstein distance for persistence diagrams. 
They are based on the Quadtree (Indyk, Thaper 2003) and Flowtree (Backurs, Dong, Indyk, Razenshteyn, Wagner, 2019) algorithms developed for standard optimal transport. 
As such, much of the code in this repository is adapted from code provided alongside the Flowtree paper [here](https://github.com/ilyaraz/ot_estimators). 
The installation instructions are the same as provided in the [original repository](https://github.com/ilyaraz/ot_estimators). 

## Running the code using the Python wrapper
The `"python"` directory contains a file `example.py` contains an example of how to use the implementations of each algorithm and evaluate each algorithms relative error on sample synthetic data. Note that to run `example.py`, we require the [GUDHI library](https://gudhi.inria.fr/) since calculating the exact 1-Wasserstein uses the implementation of Hera provided in GUDHI. Additionally, when you use the wrapper, be sure to include the `.so` file in the search path for modules. Run `example.py` as follows.

    python3 example.py data_folder 
    
