## Project 7 OPENCLASSROOMS - Implement a scoring model

The purpose of this project is to assess the risk of a client's default before granting credit.  
Source of data : https://www.kaggle.com/c/home-credit-default-risk/data

Based on the given dataset, we will attribute for each new client a probability of default from 0 to 1 (1 corresponds to the maximum risk).   
The final result should be presented on an interactive online-dashboard, with the possibility to :
- visualize descriptive information related to a client (via a filter system).
- visualize the score and its interpretation in an intelligible way for a person unfamiliar with data science.
- compare descriptive information about a client to the whole customers base or to a group of similar clients

### Imbalanced Classification

It's an imbalanced classification problem, as only 8% of the customers from the provided dataset went bankrupt.  
For each model, we will try and evaluate different strategies to rebalance classes (weighting, SMOTE, random undersampling)

### Custom Metric

The models will be evaluated using a metric adapted to the business problem.  
In this case, granting a loan to an insolvent customer (False negative, type II Error) will lead to a much greater loss than refusing a loan to a solvent customer (False positive, type I Error).  
We then determine the threshold that maximizes the metric, based on the probabilities a posteriori.

### Models

Three classification algorithms are here implemented : Logistic Regression, Random Forest and Gradient Boosting.

### Interpretation

I used the library SHAP (https://shap.readthedocs.io/) to interpret the predictions, either globally (the SHAP Values explain the impact of each feature on the model output) or individually (for a given client, we can see which features are responsible for the high/low score)

### Dashboard

I built an interactive Dashboard on Streamlit.  
The data (customer information and result of the prediction) is collected via an API developed on Flask.  
The Dashboard is currently running on heroku at this URL: https://dashboard-marie.herokuapp.com/

