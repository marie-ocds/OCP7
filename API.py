# Run with python API.py

import flask
from flask import request, jsonify
import pandas as pd
import numpy as np
import json
import pickle
#import lightgbm


app = flask.Flask(__name__)
app.config["DEBUG"] = True

# LOAD AND PREPARE DATA
#___________________________________________________________________________

# Reduce size of data
n_rows=1000
#info_clients = pd.read_csv('info_clients.csv', nrows=n_rows, index_col=0)
raw_data = pd.read_csv('./data_model/application_train1.csv', nrows=n_rows, index_col='SK_ID_CURR')

raw_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
raw_data = raw_data[raw_data['CODE_GENDER'] != 'XNA']
raw_data = raw_data[raw_data['AMT_INCOME_TOTAL'] < 100000000]

good_cols = ['CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
             'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE']
infos = raw_data.loc[:,good_cols]

infos['AGE'] = (infos['DAYS_BIRTH']/-365).astype(int)
infos['YEARS EMPLOYED'] = round((infos['DAYS_EMPLOYED']/-365), 2)
infos.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

infos = infos[[ 'AGE', 'CODE_GENDER','NAME_FAMILY_STATUS', 
               'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE','YEARS EMPLOYED',
               'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
             ]]

infos.columns = [ 'AGE', 'GENDER','FAMILY STATUS', 
               'EDUCATION TYPE', 'OCCUPATION TYPE','YEARS EMPLOYED',
               'YEARLY INCOME', 'AMOUNT CREDIT', 'AMOUNT ANNUITY', 'GOODS PRICE', 
             ]

data_processed = pd.read_csv('./data_model/app_train_clean1.csv', nrows=n_rows, index_col=0)
data_processed = data_processed.drop('TARGET', axis=1)


# DEF KDTREE
from sklearn.neighbors import KDTree
df_vois = pd.get_dummies(infos.iloc[:,:6])
tree = KDTree(df_vois)

# LOAD PRE TRAINED MODEL (Light GBM)
with open('./data_model/light_gbm.pickle', 'rb') as file : 
	LGB = pickle.load(file)

#________________________________________________________

# HOME PAGE
@app.route("/")
def hello():
    return "API pour le Dashboard \'Prêt à dépenser\' "


# GENERAL INFORMATIONS
@app.route('/info', methods=['GET'])
def get_infos():
		
	# Converting the pd.DataFrame to JSON
	info_json = json.loads(infos.to_json())

	return jsonify({ 'data' : info_json})	


# GENERAL INFORMATIONS ON SELECTED CLIENT
@app.route('/info/<int:id_client>', methods=['GET'])
def get_info_id(id_client):
	
	info_client_select = infos.loc[id_client:id_client]
	
	# Converting the pd.Series to JSON
	info_client_json = json.loads(info_client_select.to_json())

	return jsonify({ 'data' : data_client_json})


# DATA PROCESSED AND PREDICTIONS ON SELECTED CLIENT
@app.route('/processed/<int:id_client>', methods=['GET'])
def get_data_pred(id_client):
	
	data_client_select = data_processed.loc[id_client:id_client]
	
	# Converting the pd.Series to JSON
	data_client_json = json.loads(data_client_select.to_json())

	# Make predictions
	client_pred = 100*LGB.predict_proba(data_client_select)[0][1]

	return jsonify({ 'data' : data_client_json,
					'prediction' : client_pred})


# DATA PROCESSED AND PREDICTION ON CLIENT'S NEIGHBORS
@app.route('/voisins/<int:id_client>', methods=['GET'])
def voisins(id_client):
	# get indexes of 10 nearest neighbors
	idx_vois = tree.query(df_vois.loc[id_client:id_client], k=10)[1][0]
	# select processed data of neighbors
	data_vois = data_processed.iloc[idx_vois]
	#make predictions
	predict_vois = 100*LGB.predict_proba(data_vois).mean(axis=0)[1]
	# get mean of features for neighbors
	mean_vois = pd.DataFrame(data_vois.mean(), columns=['voisins']).T
	# Converting the pd.Series to JSON
	mean_vois_json = json.loads(mean_vois.to_json())

	return jsonify({ 'mean' : mean_vois_json,
					'prediction' : predict_vois})    

app.run()
