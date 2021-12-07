# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

import pickle

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

import os



# Antes das APIs
colunas = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents', 'IncomePerPerson', 'NumOfPastDue', 'MonthlyDebt',
       'NumOfOpenCreditLines', 'MonthlyBalance', 'age_sqr']

app = Flask(__name__)

app.config["BASIC_AUTH_USERNAME"] = os.environ.get('BASIC_AUTH_USERNAME')
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app) 

@app.route("/score/<cpf>")
@basic_auth.required
def show_cpf(cpf):
    print("Recebendo: CPF = {}".format(cpf))
    return f"CPF = {cpf}"


def load_model(file_name = 'xgboost_undersampling.pkl'):
    return pickle.load(open(file_name, "rb"))


modelo = load_model(file_name='../../models/xgboost_undersampling.pkl')

# Nova rota - receber um CPF como parâmetro
@app.route("/score/", methods=["POST"])
def score_cpf():
    # Pegar o JSON da requisição
    dados = request.get_json()
    payload = np.array([dados[col] for col in colunas])
    payload = xgb.DMatrix([payload], feature_names=colunas)
    score = np.float64(modelo.predict(payload)[0])
    # return jsonify(score=_score)
    status = 'APROVADO'
    if score <= 0.3:
        status = 'REPROVADO'
    elif score <= 0.6: 
        status = 'MESA DE AVALIACAO'
    resultado = jsonify(cpf=dados["cpf"], score=score, status=status)
    print(resultado)
    return resultado

# Rota padrão
@app.route('/')
def home():
    print('API de predicao de score para crédito')
    return 'API de Predição de Crédito'

# Subir a API

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') #host will select what better address to run server
