# Pyr Classifier
Classifier algorithm for pyrheliometer data. 

# Documentation

**Documentation Site:**  https://tamarervin.github.io/pyr_classifier/

# Build conda environment

* update dependencies in conda_env.yml [file](conda_env.yml)   
* run the following from the folder containing the .yml file
    * ``conda env create -f conda_env.yml``  
* to add new dependencies, update conda_env.yml [file](conda_env.yml)  
* run the following from the folder containing the .yml file  
    * ``conda env update -f conda_env.yml``

# Feed in array
inputs:  
1. datetime array  
2. flux array  
3. path to model  
4. path to csv if needed  

``date_labels = classify_array(dates, flux, model_path, csv_name=None)``