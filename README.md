# FreezeAnalysis
This repository contains iPython files that can be used to score an animal's motion and freezing while in a conditioning chamber.  It was designed with side-view recording in mind, and with the intention of being able to crop the top of a video frame to remove the influence of fiberoptic/miniscope cables.

## Included Files
* **FreezeAnalysis_Calibration.ipynb** is used to find baseline fluctuation of pixel grayscale across time with no animal present.
* **FreezeAnalysis_Individual.ipynb** is used to analyze a single video. Provides extensive visualization abilities.
* **FreezeAnalysis_BatchProcess.ipynb** is used to batch process a set of videos and create output file with summary statistics for each video using user-defined bins.
* **FreezeAnalysis_Functions.py** contains functions used by ipynb files.  This file is required to be in the same folder as ipynb files but does not need to be edited by the user.

## Package requirements
The iPython scripts included in this repository require the following packages to be installed in your Conda environment:
* python (3.6.5)
* jupyter
* imread
* mahotas(1.4.4)
* numpy(1.14.3)
* pandas(0.23.0)
* matplotlib(2.2.2) 
* opencv(3.4.3)

The following commands can be executed in your terminal to create the environment: 
* ```conda config --add channels conda-forge```
* ```conda create -n EnvironmentName python=3.6.5 mahotas=1.4.4 pandas=0.23.0 matplotlib=2.2.2 opencv=3.4.3 jupyter imread```

## Video requirements
As of yet, only mpg1 videos have been extensively tested, though wmv also seems to work.

## Running Code
After downloading the files onto your local computer in a single folder, from the terminal activate the necessary Conda environment and open Jupyter Notebook, then navigate to the files on your computer. The individual scripts contain more detailed instructions.
