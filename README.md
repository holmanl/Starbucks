# Starbucks Marketing Analysis

## Purpose
To create a model that can predict whether a marketing initiative would work on a customer based on the characteristics of the offer and the customer. This would allow tailoring of marketing offers to customers in order to maximize the likelihood of influencing spending. 

## Results
Successfully created a binary categorization model using XGBoost with 72.3% accuracy. Full findings of the code can be found here: https://medium.com/@lance.holman/starbucks-marketing-analysis-ca911debaba

## Files
* Starbucks_Capstone_notebook - exploratory.ipynb - initial data exploration
* data_cleaning.py - data cleaning and feature engineering functions 
* Model Training.ipynb - training models to classify data
* final_model.sav - pickled classification model
* OLD Starbucks_Capstone_notebook.ipynb - old exploratory file, replaced 
* portfolio.json – contains metadata about each marketing offer
*	profile.json – contains demographic data of the customers
*	transcript.zip – contains records of transactions, offers viewed, offers received, and offers completed (zipped to meet GitHub upload requirements) 

## Libraries
matplotlib, seaborn, sklearn.preprocessing, datetime, pandas, numpy, math, json, sklearn.model_selection, sklearn.linear_model, sklearn.metrics, xgboost, pickle

## Links
Links used to help create this project:
https://machinelearningmastery.com/xgboost-python-mini-course/
https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning#A-Guide-on-XGBoost-hyperparameters-tuning
