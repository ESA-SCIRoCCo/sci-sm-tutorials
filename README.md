# sci-sm-tutorials

This repository contains different exercises for a better understanding of the TU Wien soil moisture retrieval algorithm.

The exercises can be viewed online without further requirements, but in order to use them interactively, the following installation process has to be carried out:

## Installation - Linux

* Clone Git repository: 
    - open your computer's terminal
    - change to folder where you want to clone the repository to
    - type `git clone https://github.com/ESA-SCIRoCCo/sci-sm-tutorials.git` and hit Enter

* Install miniconda and create conda environment (= install all the packages needed for the tutorials):
    - if not open anymore, open your computer's terminal
    - change to the previously cloned Git repository
    - type `source setup_env` and hit Enter
    - activate environment: `source activate sci-sm-tutorials`

## Installation - Windows

* Clone Git repository: see Linux

* Install miniconda from http://conda.pydata.org/miniconda.html

* Create conda environment (= install all the packages needed for the tutorials):
    - if not open anymore, open your computer's terminal
    - change to the previously cloned Git repository
    - type `source setup_env` and hit Enter (if Error: install packages listed in conda_environment.yml manually)
    - activate environment: `activate sci-sm-tutorials`

## Installation - Mac OS X

* Clone Git repository: see Linux

* Install miniconda:
    - download miniconda from http://conda.pydata.org/miniconda.html
    - open your computer's terminal; type
        * `cd Downloads/`
        * `bash Miniconda2-latest-MacOSX-x86_64.sh`
        * export the path to miniconda, e.g. `export PATH="/Users/YourName/miniconda2/bin:$PATH"`
        * try out if miniconda installation worked: `conda list`should list all installed packages
        * More information: http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install

* Create conda environment (= install all the packages needed for the tutorials):
    - if not open anymore, open your computer's terminal
    - change to the previously cloned Git repository
    - type `source setup_env` and hit Enter
    - activate environment: `source activate sci-sm-tutorials`


More information on conda environments: http://conda.pydata.org/docs/using/envs.html

## Start jupyter notebook

* Open terminal, type 'jupyter notebook'
* A new tab is opened in your browser where you can select any exercise you wish from the previously cloned Git repository.



Please don't hesitate to raise an issue on any bugs you might find in the tutorials.
