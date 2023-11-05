# MetLife ML Engineering Challenge

### Lucas Argento

## About

This repo has the code for a ML challenge that aims to replicate a real life scenario where a company would like to predict insurance charges on clientes based on some user data.

The task is modeled as a Regression Problem, and several models where trained with this objective. 

The development process of the ML Solution can be found in the modelling.ipynb notebook, while the Production Solution (that includes all cleaning, training and predicting steps) can be found in the /Pipelines folder, in training.py and scoring.py

During the process, data is stored in a MySQL instance. That step is managed by the db_builder.py file, under the /Utils folder. 

## To replicate the solution, you can:

> Create a conda virtual environment and install required packages using the environment.yml file
> Create a pip virtual environment and install required packages using the requirements.txt file
> Build a docker image with the provided docker file. 

The last one is the recommended approach since the docker container already has MySQL installed on it. If you want to run this code in your local machine, you will need to install MySQL as well and enable the service.

> An output of the models performance (of each training run) is stored in csv format with the title "training_output - {date}.csv""

> The best model of each run is stored as best_model.pkl


