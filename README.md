# Behavior Tracking
This repository contains iPython files that can be used to track the location, motion, and freezing of an animal. For the sake of clarity, these processes are described as two modules: one for the analysis of freezing and motion (with motion being movement independent of location); the other for tracking an animal's location.  .  **If you are unfamiliar with how to use iPython/Jupyter Notebook, please see the [Getting Started repository](https://github.com/ZachPenn/GettingStarted)**

## Location Tracking Module
The location tracking module allows for the analysis of a single animal's location on a frame by frame basis.  In addition to providing the user the with the ability to crop the portion of the video frame in which the animal will be, it also allows the user to specify regions of interest (e.g. left and right sides) and provides tools to quantify the time spent in each region, as well as distance travelled.  Run **LocationTracking.ipynb** to implement.

## Freeze Analysis Module
The freezing module allows the user to automatically score an animal's motion and freezing while in a conditioning chamber.  It was designed with side-view recording in mind, and with the intention of being able to crop the top of a video frame to remove the influence of fiberoptic/miniscope cables.  In the case where no cables are to be used, recording should be capable from above the animal.  

### Basic Workflow for Freeze Analysis
1. Run **FreezeAnalysis_Calibration.ipynb** on a short video of a chamber with no animal in it (~10 sec).  This allows detection of basal fluctuation in pixel grayscale values.  A suggested cutoff for use with subsequent steps is provided.
2. Process several individual behavior videos with **FreezeAnalysis_Individual.ipynb**.  This will allow extensive visualization of results in order to ensure confidence in selected parameters. 
3. Once you are comfortable with parameters, use **FreezeAnalysis_BatchProcess.ipynb** on a whole folder of videos!

## Included Files
* **LocationTracking.ipynb** is used to find the frame-by-frame location and distance travelled of an animal.  ROIs can be specified and analyzed as well.
* **FreezeAnalysis_Calibration.ipynb** is used to find baseline fluctuation of pixel grayscale across time with no animal present.
* **FreezeAnalysis_Individual.ipynb** is used to analyze a single video. Provides extensive visualization abilities.
* **FreezeAnalysis_BatchProcess.ipynb** is used to batch process a set of videos and create output file with summary statistics for each video using user-defined bins.
* **FreezeAnalysis_Functions.py** and **LocationTracking_Functions.py** contains functions used by ipynb files.  These files are required to be in the same folder as ipynb files but do not need to be edited by the user.

## Installation and Package Requirements
The iPython scripts included in this repository require the following packages to be installed in your Conda environment.  Although the package versions used are listed it is likely that latest releases of all will be fine to use:
* python (3.6.5)
* jupyter
* imread
* mahotas(1.4.4)
* numpy(1.14.3)
* pandas(0.23.0)
* matplotlib(2.2.2) 
* opencv(3.4.3)
* holoviews
* scipy

Provided you have installed miniconda (see **[Getting Started repository](https://github.com/ZachPenn/GettingStarted)** for more details), the following commands can be executed in your terminal to create the environment: 
1. ```conda config --add channels conda-forge```
2. ```conda create -n EnvironmentName python=3.6.5 mahotas=1.4.4 pandas=0.23.0 matplotlib=2.2.2 opencv=3.4.3 jupyter imread holoviews scipy```

## Video requirements
As of yet, mpg1, wmv, and avi (mp4 codec) all seem to work.  

## Running Code
After downloading the files onto your local computer in a single folder, from the terminal activate the necessary Conda environment (```source activate EnvironmentName```) and open Jupyter Notebook (```jupyter notebook```), then navigate to the files on your computer. The individual scripts contain more detailed instructions.
