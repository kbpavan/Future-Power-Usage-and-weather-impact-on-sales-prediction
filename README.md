# Future Power Usage and weather impact on sales prediction

This repo contains py files for a data science regression project aimed at building a machine learning model that can predict future 6-Day power usage for different industrial sectors/ rate segments of a utility company serving 10M customers. This project also involves building a ML model that can predict monthly weather impact on sales.

The code in the notebooks was written in python; the following python libraries were used throughout the project:

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels
- Teradata
- sklearn
- Xgboost
- keras

# Project Context

- The company is a large utilities company serving 10M customers with power production capabilities.
- They want to find a better way to predict future usage and see weather impact usage on their sales for their different industrial sectors/rate segment customers.
- They wanted to use these results to better prepare for the demand and know how much weather is impacting different classes of their customers usage.

# My Role

I&#39;ve to find a data-driven approach to predict usage and weather impact

- They currently have minute level usage data and hour level weather data with 20 features.
- My task is to build a usage prediction model using that dataset.
- If I can build a model to predict usage with an average error less than the process they have in place, then company can replace existing process with my model.

# Current Solution

Company currently uses previous years usage data for any given day and number of new customers to canculate an estimate.

Unfortunately, this method is not very efficient. So, they wanted to revamp this process and make use of ML to make better predictions and know weather impact on sales.

# Problem Specifics

It&#39;s always helpful to scope the problem before starting.

**Deliverable** : Trained models, Deploy final models to automate predictions daily, Writing predictions to Database, Dynamic Tableau Dashboards for visualization, Feature Importance

**Machine learning task** : Regression

**Target variable** : Power Usage (KWH)

**Win condition** : Avg Prediction better than process in place
