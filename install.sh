#!/usr/bin/sh

# Change these for your system!
os=MacOSX
#os=Linux
CODEDIR=$HOME/projects
FILES_DIR=$HOME/Downloads/GCPro

# Add this to your bashrc or bash_profile or whatever
# And modify SPS_HOME to be where you want the fsps source
export PATH=$HOME/miniconda/bin:$PATH
export SPS_HOME=$HOME/projects/fsps

# -- Install python/Anaconda ---
wget https://repo.continuum.io/miniconda/Miniconda3-latest-${os}-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
conda config --set always_yes yes
conda update -q conda

# -- Create environment --
# This creates the environment `pro` with numpy, scipy, astropy...
cd $FILES_DIR
conda env create -f pro.yml

# --- Install FSPS ---
mkdir -p $SPS_HOME
cd $SPS_HOME/..
git clone https://github.com/cconroy20/fsps
cd fsps/src
cp ${FILES_DIR}/fsps.Makefile Makefile
make clean
make all

# --- Install python packages to the `pro` environment ---
conda activate pro

cd $CODEDIR
git clone https://github.com/dfm/python-fsps
cd python-fsps
python setup.py install

cd $CODEDIR
git clone https://github.com/bd-j/sedpy
cd sedpy
python setup.py install

cd $CODEDIR
git clone https://github.com/bd-j/prospector
cd prospector
python setup.py install

# --- Fire up the notebook ---
cd $FILES_DIR
jupyter notebook GCSpecPhotDemo.ipynb