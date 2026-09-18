# Flight-price-regression-model
End-to-end machine learning project for predicting flight prices using EDA, feature engineering, feature selection, regression models, hyperparameter tuning, evaluation, and Flask deployment.
Flight Price Regression Model

A machine learning project for predicting flight prices from travel-related features. The project covers the complete workflow from data ingestion and exploratory data analysis to feature engineering, feature selection, model training, hyperparameter tuning, evaluation, model serialization, and Flask-based prediction.

Project objective

Predict the flight price using features such as:

Boarding city

Destination city

Flight type

Booking agency

Day of travel

Week number

Weekday

Machine learning workflow

Load flight, hotel and user datasets.

Perform basic data checks for shape, data types, missing values and duplicates.

Convert the date field and extract day, month, year, week number and weekday.

Perform exploratory data analysis using distributions, box plots, count plots and relationship plots.

Create an additional flight_speed feature from distance and travel time.

Convert categorical variables using one-hot encoding.

Select the final modeling features.

Split the data into training and testing sets and scale the features.

Train and compare multiple regression models:

Linear Regression

Lasso Regression

Ridge Regression

ElasticNet

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

Tune model hyperparameters with GridSearchCV and 3-fold cross-validation.

Compare models using MSE, RMSE, MAE, R² and adjusted R².

Save the trained model and scaler with Pickle.

Expose the prediction workflow through a Flask web application.

Files

project_of_flight_price_regression_model_clean.py — cleaned version of the Colab-exported project code with the exposed ngrok credential removed.

requirements.txt — Python dependencies used by the project.

.gitignore — files and folders that should not be committed.

Running the project

The original project was developed in Google Colab and uses Google Drive paths for the datasets and serialized models. Update those paths for your own environment before running it.

For the Flask portion, make sure the saved model and scaler are available at the paths expected by the application, or update the loading code to match your environment.

Security note

Do not commit API keys, authentication tokens, passwords, .env files, dataset credentials or private model files to a public repository. The original project contained an ngrok authentication command; that credential has been removed from the cleaned copy in this repository.

Reference

The original project file references this GitHub project as a source/reference:
https://github.com/Shuvadip007/Voyage-Analytics-Integrating-MLOps-in-Travel

Review the original repository's license and attribution requirements before publishing or distributing a copied/modified version.
