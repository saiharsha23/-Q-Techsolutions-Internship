# House price prediction 

## Description 
This project is used to predict the prices of the houses based on the housing features like area, no of bedrooms etc.This model uses the house price prediction dataset which consists of house pricing features and predicts the prices of the houses using Ridge and Lasso regression models and save the results.Visualizations help in comparing the accuracy and effectiveness of the models.

## File structure 
**data/**:It has the dataset named Housing.csv
**models/**:It contains the Ridge and Lasso regression models saved as lasso_model.pkl and ridge_model.pkl
**outputs/**:It has the outputs that are predicted("predicted_house_prices.csv) by training the ridge and lasso models
**visualizations/**:consists of the scatter plots of the ridge and lasso models as lasso_plot.png and ridge_plot.png 
**price_prediction.py/**:It has the main source code for training the models and predicting the output

## How to run 

clone the repository 
    https://github.com/saiharsha23/-Q-Techsolutions-Internship
    cd Q-Techsolutions-Internship 

setting the virtual environment 
       python -m venv venv
       venv\scripts\activate
Install the requirements 
        pip install -r requirements.txt

run the source code file 
       price_prediction.py 

       
