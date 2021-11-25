# Doppler_Processing_using_machine_learning
<img src="https://media.giphy.com/media/VuehuL4fMHLgs/giphy.gif"/>

+ Hello, this document has the objective to guide you through the different python scripts to reproduce the paper results. 
+ In order to use the GPU, you need to install the GPU drivers. If you do not have a GPU, no problem, the code run on CPU too.
+ I strongly recommend the use of an environment to install the different python libraries 
+ In the requirement.txt file you will find the needed libraries to run all scripts. You can install it using pip, e.g. pip install requirement.txt

## Scripts
In the script folder, you will find different python scripts. In order to train the Neural Network, first, you have to generate the training data, then train and validate the NN. You must to execute the scripts following the next order:
+ data_gen_ML_training.py -> This script will generate the training data, you can modify any parameter depending on your interest. The default parameters are the same as in the paper.
+ trainig_script.py -> This script will train and validate the NN. If you want you can modify de default NN model. This script will generate a folder with all the training and validation results, also with the trained NN model.

## Testing the trained NN with diferent experiments
In the directory scripts/experiments you will find three different experiments. The script names are auto descriptives. 

## NN architecture study
In the directory scripts/architectures_study you will find two python scripts, one for training different NN and the other for prediction time study


