# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from util import get_resampling
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

app.config["BASIC_AUTH_USERNAME"] = os.environ.get("BASIC_AUTH_USERNAME")
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get("BASIC_AUTH_PASSWORD")

basic_auth = BasicAuth(app) 

@app.route("/score/<cpf>")
@basic_auth.required
def show_cpf(cpf):
    print("Recebendo: CPF = {}".format(cpf))
    return f"CPF = {cpf}"


def create_model():
    print('Iniciando carga dos dados')
    # df_treino = pd.read_csv('../dataset/cs-training.csv')
    df_treino = pd.read_csv('cs-training.csv')
    df_treino.drop(columns='Unnamed: 0', inplace=True)
    df_treino.rename(columns={'SeriousDlqin2yrs':'target'}, inplace=True)
    df_treino.target = df_treino.target.apply(lambda target: 1 if target==0 else 0)
    print('df_treino.shape:', df_treino.shape)

    print('Preparação dos dados')
    df_treino['MonthlyIncome'] = np.log(1+df_treino['MonthlyIncome'].values)
    df_treino['NumberOfDependents'] = np.log(1+df_treino['NumberOfDependents'].values)
    df_treino['MonthlyIncome_Null'] = pd.isnull(df_treino['MonthlyIncome'])
    df_treino['NoD_Null'] = pd.isnull(df_treino['NumberOfDependents'])
    df_treino.dropna(axis=0,how='any',subset=['NumberOfDependents'],inplace=True)
    df_treino.reset_index()
    df_treino['IncomePerPerson'] = df_treino['MonthlyIncome']/\
        (df_treino['NumberOfDependents']+1)
    df_treino['NumOfPastDue'] = \
        df_treino['NumberOfTimes90DaysLate']+\
        df_treino['NumberOfTime60-89DaysPastDueNotWorse']+\
        df_treino['NumberOfTime30-59DaysPastDueNotWorse']
    df_treino['MonthlyDebt'] = df_treino['DebtRatio']*df_treino['MonthlyIncome']
    df_treino['NumOfOpenCreditLines'] = df_treino['NumberOfOpenCreditLinesAndLoans']-\
        df_treino['NumberRealEstateLoansOrLines']
    df_treino['MonthlyBalance'] = df_treino['MonthlyIncome']-df_treino['MonthlyDebt']
    ## remove outlier
    df_treino = df_treino[df_treino['age'] != 0]
    df_treino = df_treino[df_treino['age'] !=99]
    df_treino = df_treino[df_treino['age'] !=101]
    ## create new features
    df_treino['age_sqr'] = df_treino['age'].values^2 
    ## apply the same operation on testing set
    df_treino['age_sqr'] = df_treino['age'].values^2
    df_treino.drop(['MonthlyIncome_Null','NoD_Null'],axis=1,inplace=True)   

    print('Modelagem...') 
    X_train, X_test, y_train, y_test = \
    train_test_split(df_treino.drop(columns='target'), 
                     df_treino['target'], 
                     test_size=0.33, random_state=42)

    X_train_under, y_train_under = get_resampling(X_train, y_train, 
                                                verbose=False, 
                                                random_state=42, 
                                                by='undersampling', 
                                                good_mult=1)
    
    train = xgb.DMatrix(X_train_under,y_train_under,
                    feature_names=X_train_under.columns)
    test = xgb.DMatrix(X_test,feature_names=X_train_under.columns)
    xgb_params = {
                        'eta':0.03,
                        'max_depth':4,
                        'sub_sample':0.9,
                        'colsample_bytree':0.5,
                        'objective':'binary:logistic',
                        'eval_metric':'auc',
                        'silent':0
                        }
    print('Treinamento do modelo')
    return xgb.train(xgb_params,train,num_boost_round=500)

def load_model(file_name = 'xgboost_undersampling.pkl'):
    return pickle.load(open(file_name, "rb"))

# modelo = create_model()
modelo = load_model()

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
    return 'API de Predição de Crédito'

# Subir a API

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') #host will select what better address to run server
