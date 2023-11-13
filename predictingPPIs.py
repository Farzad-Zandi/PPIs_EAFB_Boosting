## Farzad Zandi, 2023.
# Predicting Protein Protein Interactions.

# Import requaried libraries.
from multiprocessing import pool
import os
import csv
from pickle import NONE
from random import random
from re import sub
from tabnanny import verbose
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from logitboost import LogitBoost
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, matthews_corrcoef
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as LightGBM
from snapml import BoostingMachineClassifier as SnapBoostClassifier
from sklearn.ensemble import VotingClassifier

# Ingnoring warnings.
warnings.filterwarnings('ignore')

# Load data.
print("Farzad Zandi, 2023.")
print("Predicting Protein Protein Iteractions.")
print("Loading Data...")
# dataset = pd.read_csv('D:\\Thesis\\myCodes\\Selected Features\\AFB\\H.pylori\\XGBoost\\AD.csv')
# dataset = pd.read_csv('D:\\Thesis\\myCodes\\Selected Features\\AFB_no_FS\\S.cerevisiae\\XGBoost\\Fusion.csv')
dataset = pd.read_csv('D:\\Thesis\\myCodes\\Extracted Features\\S.cerevisiae\\Fusion\\Fusion.csv')
# dataset = pd.read_csv('D:\\AD.csv')
N = dataset.shape[1]-1
target = dataset.iloc[:,N]
dataset = dataset.drop(dataset.columns[N], axis=1)
dataset = dataset.drop(dataset.columns[0], axis=1)

print("Data Dimension: ", dataset.shape)
# Tarin and Test Split.
predictors = dataset
target = pd.DataFrame(target)
print("Normalizing Data...")
# predictors = preprocessing.normalize(predictors)
predictors = preprocessing.minmax_scale(predictors, feature_range=(0,1))
predictors = pd.DataFrame(predictors)

# K Fold Cross Validation.
k = 10
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
i = 1
# Initializing Metric variables.
accScore_xgb = []; accScore_lgb = []; accScore_cb = []; accScore_ab = []; accScore_lgbm = []; accScore_snap = []; accScore_voting = []
preScore_xgb = []; preScore_lgb = []; preScore_cb = []; preScore_ab = []; preScore_lgbm = []; preScore_snap = []; preScore_voting = []
recScore_xgb = []; recScore_lgb = []; recScore_cb = []; recScore_ab = []; recScore_lgbm = []; recScore_snap = []; recScore_voting = []
f1Score_xgb = []; f1Score_lgb = []; f1Score_cb = []; f1Score_ab = []; f1Score_lgbm = []; f1Score_snap = []; f1Score_voting = []
senScore_xgb = []; senScore_lgb = []; senScore_cb = []; senScore_ab = []; senScore_lgbm = []; senScore_snap = []; senScore_voting = []
speScore_xgb = []; speScore_lgb = []; speScore_cb = []; speScore_ab = []; speScore_lgbm = []; speScore_snap = []; speScore_voting = []
mccScore_xgb = []; mccScore_lgb = []; mccScore_cb = []; mccScore_ab = []; mccScore_lgbm = []; mccScore_snap = []; mccScore_voting = []
print("Performing", k, "Fold Cross Validation...")
# Fitting & Predicting Model by K Fold.
for trainIDX, testIDX in kf.split(predictors):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = predictors.iloc[trainIDX,:], predictors.iloc[testIDX,:]
    yTrain, yTest = target.iloc[trainIDX], target.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # # SnapBoost.
    # print("Classifying by SnapBoost...")
    # model = SnapBoostClassifier()
    # model.fit(np.array(xTrain), yTrain)    
    # yPred = model.predict(np.array(xTest))
    # accScore_snap.append(accuracy_score(yPred, yTest)*100)
    # preScore_snap.append(precision_score(yPred, yTest)*100)
    # recScore_snap.append(recall_score(yPred, yTest)*100)
    # f1Score_snap.append(f1_score(yPred, yTest)*100)
    # cm = confusion_matrix(yPred, yTest)
    # senScore_snap.append((cm[0,0]/(cm[0,0]+cm[0,1]))*100)
    # speScore_snap.append((cm[1,1]/(cm[1,0]+cm[1,1]))*100)
    # mccScore_snap.append(matthews_corrcoef(yPred, yTest)*100)    
    # # XGBoost.
    # print("Classifying by eXtreme Gradient Boosting...")
    # xgb_model = xgb.XGBClassifier()
    # xgb_model.fit(xTrain, yTrain)
    # yPred = xgb_model.predict(xTest)
    # accScore_xgb.append(accuracy_score(yPred, yTest)*100) 
    # preScore_xgb.append(precision_score(yPred, yTest)*100)
    # recScore_xgb.append(recall_score(yPred, yTest)*100)
    # f1Score_xgb.append(f1_score(yPred, yTest)*100)
    # cm = confusion_matrix(yPred, yTest)
    # senScore_xgb.append((cm[0,0]/(cm[0,0]+cm[0,1]))*100)
    # speScore_xgb.append((cm[1,1]/(cm[1,0]+cm[1,1]))*100) 
    # mccScore_xgb.append(matthews_corrcoef(yPred, yTest)*100)
    # # LogitBoost
    # print("Classifying by LogitBoost...")
    # lgb = LogitBoost(n_estimators = 100)
    # lgb.fit(xTrain, yTrain)
    # yPred = lgb.predict(xTest)
    # accScore_lgb.append(accuracy_score(yPred, yTest)*100) 
    # preScore_lgb.append(precision_score(yPred, yTest)*100)
    # recScore_lgb.append(recall_score(yPred, yTest)*100)
    # f1Score_lgb.append(f1_score(yPred, yTest)*100)
    # cm = confusion_matrix(yPred, yTest)
    # senScore_lgb.append((cm[0,0]/(cm[0,0]+cm[0,1]))*100)
    # speScore_lgb.append((cm[1,1]/(cm[1,0]+cm[1,1]))*100)
    # mccScore_lgb.append(matthews_corrcoef(yPred, yTest)*100)
    # # CatBoost.
    # print("Classifying by CatBoost...")
    # cb = CatBoostClassifier(
    #     iterations=40, depth=10, verbose=False, loss_function='Logloss',
    #     learning_rate=1, od_type='Iter', early_stopping_rounds=5)
    # cb = CatBoostClassifier(iterations=40, verbose=False)
    # cb.fit(Pool(xTrain, yTrain))
    # yPred = cb.predict(Pool(xTest))
    # accScore_cb.append(accuracy_score(yPred, yTest)*100)
    # preScore_cb.append(precision_score(yPred, yTest)*100)
    # recScore_cb.append(recall_score(yPred, yTest)*100)
    # f1Score_cb.append(f1_score(yPred, yTest)*100)
    # cm = confusion_matrix(yPred, yTest)
    # senScore_cb.append((cm[0,0]/(cm[0,0]+cm[0,1]))*100)
    # speScore_cb.append((cm[1,1]/(cm[1,0]+cm[1,1]))*100)
    # mccScore_cb.append(matthews_corrcoef(yPred, yTest)*100)
    # # AdaBoost.
    # print("Classifying by AdaBoost...")
    # ab = AdaBoostClassifier(n_estimators=100)
    # ab.fit(xTrain, yTrain)
    # yPred = ab.predict(xTest)
    # accScore_ab.append(accuracy_score(yPred, yTest)*100)
    # preScore_ab.append(precision_score(yPred, yTest)*100)
    # recScore_ab.append(recall_score(yPred, yTest)*100)
    # f1Score_ab.append(f1_score(yPred, yTest)*100)
    # cm = confusion_matrix(yPred, yTest)
    # senScore_ab.append((cm[0,0]/(cm[0,0]+cm[0,1]))*100)
    # speScore_ab.append((cm[1,1]/(cm[1,0]+cm[1,1]))*100)
    # mccScore_ab.append(matthews_corrcoef(yPred, yTest)*100)
    # LightGBM.
    # print("Classifying by LightGBM...")
    # lgbm = LightGBM.LGBMClassifier(n_estimators=100)
    # lgbm.fit(xTrain, yTrain)
    # yPred = lgbm.predict(xTest)
    # accScore_lgbm.append(accuracy_score(yPred, yTest)*100)
    # preScore_lgbm.append(precision_score(yPred, yTest)*100)
    # recScore_lgbm.append(recall_score(yPred, yTest)*100)
    # f1Score_lgbm.append(f1_score(yPred, yTest)*100)
    # cm = confusion_matrix(yPred, yTest)
    # senScore_lgbm.append((cm[0,0]/(cm[0,0]+cm[0,1]))*100)
    # speScore_lgbm.append((cm[1,1]/(cm[1,0]+cm[1,1]))*100)
    # mccScore_lgbm.append(matthews_corrcoef(yPred, yTest)*100)
    # # Voting.
    print("Classifying by Voting...")
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    accScore_voting.append(accuracy_score(yPred, yTest)*100)
    preScore_voting.append(precision_score(yPred, yTest)*100)
    recScore_voting.append(recall_score(yPred, yTest)*100)
    f1Score_voting.append(f1_score(yPred, yTest)*100)
    cm = confusion_matrix(yPred, yTest)
    senScore_voting.append((cm[0,0]/(cm[0,0]+cm[0,1]))*100)
    speScore_voting.append((cm[1,1]/(cm[1,0]+cm[1,1]))*100)
    mccScore_voting.append(matthews_corrcoef(yPred, yTest)*100)
    i = i + 1
# Calculating Mean Metrics.

methods = ['XGBoost', 'LogitBoost', 'CatBoost', 'AdaBoost', 'LightGBM', 'SnapBoost', 'Voting']
accuracy = [sum(accScore_xgb)/k, sum(accScore_lgb)/k, sum(accScore_cb)/k, sum(accScore_ab)/k, sum(accScore_lgbm)/k, sum(accScore_snap)/k, sum(accScore_voting)/k]
precision = [sum(preScore_xgb)/k, sum(preScore_lgb)/k, sum(preScore_cb)/k, sum(preScore_ab)/k, sum(preScore_lgbm)/k, sum(preScore_snap)/k, sum(preScore_voting)/k]
recall = [sum(recScore_xgb)/k, sum(recScore_lgb)/k, sum(recScore_cb)/k, sum(recScore_ab)/k, sum(recScore_lgbm)/k, sum(recScore_snap)/k, sum(recScore_voting)/k]
f1score = [sum(f1Score_xgb)/k, sum(f1Score_lgb)/k, sum(f1Score_cb)/k, sum(f1Score_ab)/k, sum(f1Score_lgbm)/k, sum(f1Score_snap)/k, sum(f1Score_voting)/k]
sensivity = [sum(senScore_xgb)/k, sum(senScore_lgb)/k, sum(senScore_cb)/k, sum(senScore_ab)/k, sum(senScore_lgbm)/k, sum(senScore_snap)/k, sum(senScore_voting)/k]
specifity = [sum(speScore_xgb)/k, sum(speScore_lgb)/k, sum(speScore_cb)/k, sum(speScore_ab)/k, sum(speScore_lgbm)/k, sum(speScore_snap)/k, sum(speScore_voting)/k]
mathCorr = [sum(mccScore_xgb)/k, sum(mccScore_lgb)/k, sum(mccScore_cb)/k, sum(mccScore_ab)/k, sum(mccScore_lgbm)/k, sum(mccScore_snap)/k, sum(mccScore_voting)/k]

XGB = [round(accuracy[0],2), round(precision[0],2), round(recall[0],2), round(f1score[0],2), round(sensivity[0],2), round(specifity[0],2), round(mathCorr[0],2)]
LGB = [round(accuracy[1],2), round(precision[1],2), round(recall[1],2), round(f1score[1],2), round(sensivity[1],2), round(specifity[1],2), round(mathCorr[1],2)]
Cat = [round(accuracy[2],2), round(precision[2],2), round(recall[2],2), round(f1score[2],2), round(sensivity[2],2), round(specifity[2],2), round(mathCorr[2],2)]
Ada = [round(accuracy[3],2), round(precision[3],2), round(recall[3],2), round(f1score[3],2), round(sensivity[3],2), round(specifity[3],2), round(mathCorr[3],2)]
LGBM = [round(accuracy[4],2), round(precision[4],2), round(recall[4],2), round(f1score[4],2), round(sensivity[4],2), round(specifity[4],2), round(mathCorr[4],2)]
Snap = [round(accuracy[5],2), round(precision[5],2), round(recall[5],2), round(f1score[5],2), round(sensivity[5],2), round(specifity[5],2), round(mathCorr[5],2)]
Voting = [round(accuracy[6],2), round(precision[6],2), round(recall[6],2), round(f1score[6],2), round(sensivity[6],2), round(specifity[6],2), round(mathCorr[6],2)]
results = pd.DataFrame(XGB, columns=['XGB'])
results['LGB'] = LGB
results['Cat'] = Cat
results['Ada'] = Ada
results['LightGBM'] = LGBM
results['SnapBoost'] = Snap
results['Voting'] = Voting

# Save Results to CSV file.
results.to_csv("D:\\LightGBMresultsAD.csv")

# Visualizing.
#sns.set(rc = {'figure.figsize':(15,8)})
#plt.xlabel('Methods')
#plt.ylabel('Accuracy Score')
#ax = sns.barplot(methods, Accuracy)
#ax.bar_label(ax.containers[0])
#plt.show()
