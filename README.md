# FreezeAnalysis
This repository contains iPython files that can be used to score an animal's motion and freezing while in a conditioning chamber.  It was designed with side-view recording in mind, and with the intention of being able to crop the top of a video frame to remove the influence of fiberoptic/miniscope cables.  In the case where no cables are to be used, recording should be cabable from above the animal.

## Basic Workflow
1. Run **FreezeAnalysis_Calibration.ipynb** on a short video of a chamber with no animal in it (~10 sec).  This allows detection of basal fluctuation in pixel grayscale values.  A suggested cutoff for use with subsequent steps is provided.
2. Process several individual behavior videos with **FreezeAnalysis_Individual.ipynb**.  This will allow extensive visualization of results in order to ensure confidence in selected parameters. 
3. Once you are comfortable with paramaters, use **FreezeAnalysis_BatchProcess.ipynb** on a whole folder of videos!

## Included Files
* **FreezeAnalysis_Calibration.ipynb** is used to find baseline fluctuation of pixel grayscale across time with no animal present.
* **FreezeAnalysis_Individual.ipynb** is used to analyze a single video. Provides extensive visualization abilities.
* **FreezeAnalysis_BatchProcess.ipynb** is used to batch process a set of videos and create output file with summary statistics for each video using user-defined bins.
* **FreezeAnalysis_Functions.py** contains functions used by ipynb files.  This file is required to be in the same folder as ipynb files but does not need to be edited by the user.

## Installation and package requirements
The iPython scripts included in this repository require the following packages to be installed in your Conda environment:
* python (3.6.5)
* jupyter
* imread
* mahotas(1.4.4)
* numpy(1.14.3)
* pandas(0.23.0)
* matplotlib(2.2.2) 
* opencv(3.4.3)

Provided you have installed miniconda (see **Getting Started repository** for more details), the following commands can be executed in your terminal to create the environment: 
1. ```conda config --add channels conda-forge```
2. ```conda create -n EnvironmentName python=3.6.5 mahotas=1.4.4 pandas=0.23.0 matplotlib=2.2.2 opencv=3.4.3 jupyter imread```

## Video requirements
As of yet, mpg1, wmv, and avi (mp4 codec) all seem to work.  

## Running Code
After downloading the files onto your local computer in a single folder, from the terminal activate the necessary Conda environment and open Jupyter Notebook, then navigate to the files on your computer. The individual scripts contain more detailed instructions.
