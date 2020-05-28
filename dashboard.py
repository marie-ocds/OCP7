# streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import requests


def main():

    API_URL = 'http://127.0.0.1:5000/'

    # LOAD DATA
    @st.cache(allow_output_mutation=True)
    def load_data(url):
    	req = requests.get(url)
    	content = json.loads(req.content.decode('utf-8'))
    	return pd.DataFrame(content['data'])

    data_load_state = st.text('Loading data...')
    infos = load_data(API_URL + 'info')
    infos = infos[['AGE', 'GENDER','FAMILY STATUS', 'EDUCATION TYPE', 
    'OCCUPATION TYPE','YEARS EMPLOYED', 'YEARLY INCOME', 'GOODS PRICE',
    'AMOUNT CREDIT', 'AMOUNT ANNUITY']]
    moyennes = pd.read_csv('./data_model/moyennes.csv', index_col=0)
    data_load_state.text('')

    # _____________________________________________________
    # GENERAL INFORMATION
    st.title('Dashboard PRET A DEPENSER')

    # Select client
    client_id = st.sidebar.selectbox('Select ID Client :', infos.index)

    # Display general informations in sidebar
    st.sidebar.table(infos.loc[client_id][:6])

    # Plot data relative to income and credit amounts
    bar_cols = infos.columns[6:10]
    infos.at['Moyenne clients', bar_cols] = infos.loc[:,bar_cols].mean()

    fig = go.Figure(data=[
        go.Bar(name='Client sélectionné', x=bar_cols, y=infos.loc[client_id, bar_cols].values),
        go.Bar(name='Moyenne des clients', x=bar_cols, y=infos.loc['Moyenne clients', bar_cols].values)
    ])
    fig.update_layout(title_text=f'Montants des revenus et du crédit demandé pour le client {client_id}')

    st.plotly_chart(fig, use_container_width=True)

    # ________________________________________________________
    # PREDICTIONS

    st.header('Risque de défaut')
    # Load data client :
    url_data_client = API_URL + 'processed/' + str(client_id)
    req = requests.get(url_data_client)
    content = json.loads(req.content.decode('utf-8'))
    # Get predictions :
    prediction_client = content['prediction']


    # Get predictions for similar clients :
    url_voisins_client = API_URL + 'voisins/' + str(client_id)
    req = requests.get(url_voisins_client)
    content = json.loads(req.content.decode('utf-8'))
    prediction_voisins = content['prediction']

    # Plot gauge
    gauge = go.Figure(go.Indicator(
        mode = "gauge+delta+number",
        value = prediction_client,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 25], 'color': "lightgreen"},
                     {'range': [25, 50], 'color': "lightyellow"},
                     {'range': [50, 75], 'color': "orange"},
                     {'range': [75, 100], 'color': "red"},
                     ],
                 'threshold': {
                'line': {'color': "black", 'width': 10},
                'thickness': 0.8,
                'value': prediction_client},

                 'bar': {'color': "black", 'thickness' : 0.2},
                },
        delta = {'reference': prediction_voisins,
        'increasing': {'color': 'red'},
        'decreasing' : {'color' : 'green'}}
        ))

    st.plotly_chart(gauge)

    st.markdown('Pour le client sélectionné : **{0:.1f}%**'.format(prediction_client))
    st.markdown('Pour les clients similaires : **{0:.1f}%** (critères de similarité : âge, genre,\
         statut familial, éducation, profession, années d\'ancienneté)'.format(prediction_voisins))


    # ________________________________________________________
    # INTERPRETATION

    feature_desc = { 'EXT_SOURCE_2' : 'Score normalisé attribué par un organisme indépendant',
                    'EXT_SOURCE_3' :  'Score normalisé attribué par un organisme indépendant', 
                    'AMT_ANNUITY' : 'Montant des annuités', 
                    'AMT_GOODS_PRICE' : 'Montant du bien immobilier',
                    'CREDIT_INCOME_PERCENT' : 'Crédit demandé par rapport aux revenus', 
                    'DAYS_EMPLOYED_PERCENT' : 'Années travaillées en pourcentage' }

    st.header('Interprétation du résultat')
    feature = st.selectbox('Selectionnez la variable à comparer', moyennes.columns)

    # Load data client :
    req = requests.get(url_data_client)
    content = json.loads(req.content.decode('utf-8'))
    # Conversion in pandas object :
    data_client = pd.DataFrame(content['data']).copy()

    # Load mean of neighbors :
    req = requests.get(url_voisins_client)
    content = json.loads(req.content.decode('utf-8'))
    mean_vois = pd.DataFrame(content['mean']).copy()

    # Compare features
    dfcomp = pd.concat([moyennes, mean_vois, data_client], join = 'inner').round(2)

    fig2 = go.Figure(data=[go.Bar(
        x=dfcomp[feature],
        y=['Moyenne des clients en règle ',
    	  'Moyenne des clients en défaut ', 
    	  'Moyenne des clients similaires ', 
    	  'Client Sélectionné '],
        marker_color=['green','red', 'orange', 'blue'],
        orientation ='h'
    )])
    fig2.update_layout(title_text=feature_desc[feature])

    st.plotly_chart(fig2)

if __name__== '__main__':
    main()




