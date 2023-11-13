## Farzad Zandi, 2023.
# Ploting ROC Curves, Predicting Protein-Protein Interactions.

# Importing requaired libraries.
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from logitboost import LogitBoost
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
from catboost import CatBoostClassifier, Pool
import lightgbm as LightGBM
from sklearn.ensemble import AdaBoostClassifier
from logitboost import LogitBoost
from snapml import BoostingMachineClassifier as SnapBoostClassifier
from sklearn.ensemble import VotingClassifier

# Ignoring warnings.
warnings.filterwarnings('ignore')

print("Farzad Zandi, 2023.")
print("Predicting Protein Protein Iteractions.")
print("Generating ROC Curve by XGBBoost...")
# print("Generating ROC Curve by AdaBoost...")
# print("Generating ROC Curve by LogitBoost...")
# print("Generating ROC Curve by CatBoost...")
# print("Generating ROC Curve by LightGBM...")

# Loading AD.
print("Loading AD...")
data = pd.read_csv('/Extracted Features/AD.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)

print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)

# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("Loading AD...")
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = CatBoostClassifier()
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0); meanTPR[0] = 0
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='AD (AUC= %0.2f)' % (meanAUC*100)) 
# Loading BLOSUM.
print("=====================================")
print("Loading BLOSUM...")
data = pd.read_csv('/Extracted Features/BLOSUM.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(
    #     iterations=40, random_seed=0, depth=10, 
    #     loss_function='Logloss', learning_rate=1, 
    #     task_type='CPU', od_type='Iter', early_stopping_rounds=5, rsm=0.001)
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0)
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='BLOSUM (AUC= %0.2f)' % (meanAUC*100)) 
# Loading CT.
print("=====================================")
print("Loading CT...")
data = pd.read_csv('/Extracted Features/CT.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(
    #     iterations=40, random_seed=0, depth=10, 
    #     loss_function='Logloss', learning_rate=1, 
    #     task_type='CPU', od_type='Iter', early_stopping_rounds=5, rsm=0.001)
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0)
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='CT (AUC= %0.2f)' % (meanAUC*100)) 
# Loading C-T-D.
print("=====================================")
print("Loading C-T-D...")
data = pd.read_csv('/Extracted Features/CTD.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(
    #     iterations=40, random_seed=0, depth=10, 
    #     loss_function='Logloss', learning_rate=1, 
    #     task_type='CPU', od_type='Iter', early_stopping_rounds=5, rsm=0.001)
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0)
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='C-T-D (AUC= %0.2f)' % (meanAUC*100)) 
# Loading DC.
print("=====================================")
print("Loading DC...")
data = pd.read_csv('/Extracted Features/DC.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(
    #     iterations=40, random_seed=0, depth=10, 
    #     loss_function='Logloss', learning_rate=1, 
    #     task_type='CPU', od_type='Iter', early_stopping_rounds=5, rsm=0.001)
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0)
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='DC (AUC= %0.2f)' % (meanAUC*100)) 
# Loading DDE.
print("=====================================")
print("Loading DDE...")
data = pd.read_csv('/Extracted Features/DDE.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(
    #     iterations=40, random_seed=0, depth=10, 
    #     loss_function='Logloss', learning_rate=1, 
    #     task_type='CPU', od_type='Iter', early_stopping_rounds=5, rsm=0.001)
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0)
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='DDE (AUC= %0.2f)' % (meanAUC*100)) 
# Loading PseAAC.
print("=====================================")
print("Loading PseAAC...")
data = pd.read_csv('/Extracted Features/PseAAC.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(
    #     iterations=40, random_seed=0, depth=10, 
    #     loss_function='Logloss', learning_rate=1, 
    #     task_type='CPU', od_type='Iter', early_stopping_rounds=5, rsm=0.001)
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0)
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='PseAAC (AUC= %0.2f)' % (meanAUC*100)) 
# Loading QSO.
print("=====================================")
print("Loading QSO...")
data = pd.read_csv('/Extracted Features/QSO.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(
    #     iterations=40, random_seed=0, depth=10, 
    #     loss_function='Logloss', learning_rate=1, 
    #     task_type='CPU', od_type='Iter', early_stopping_rounds=5, rsm=0.001)
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier(n_estimators=100)
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0)
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='QSO (AUC= %0.2f)' % (meanAUC*100)) 
# Loading Fusion.
print("=====================================")
print("Loading Fusion...")
data = pd.read_csv('/Extracted Features/Fusion.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
# Normalizing data.
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5
i = 1
tprs= []
aucs = []
meanFPR = np.linspace(0, 1, 100)
kf = KFold(n_splits = k, shuffle=True, random_state = 100)
for trainIDX, testIDX in kf.split(data):
    print("=====================================")
    print("Fold ", i)
    xTrain, xTest = data.iloc[trainIDX,:], data.iloc[testIDX,:]
    yTrain, yTest = label.iloc[trainIDX], label.iloc[testIDX]
    print("Train Set Dimension: ", xTrain.shape)
    print("Test Set Dimension: ", xTest.shape)  
    # model = xgb.XGBClassifier()
    # model = AdaBoostClassifier(n_estimators=100)
    # model = CatBoostClassifier(iterations=100, verbose=False)
    # model = LogitBoost(n_estimators = 100)
    # model = CatBoostClassifier(
    #     iterations=40, random_seed=0, depth=10, 
    #     loss_function='Logloss', learning_rate=1, 
    #     task_type='CPU', od_type='Iter', early_stopping_rounds=5, rsm=0.001)
    # model = LightGBM.LGBMClassifier(n_estimators=100)
    # model.fit(xTrain, yTrain)
    # probs = model.predict_proba(xTest)
    # model = SnapBoostClassifier()
    xgbModel = xgb.XGBClassifier()
    lgbm = LightGBM.LGBMClassifier()
    model = VotingClassifier([('XGBoost', xgbModel), ('LightGBM', lgbm)], voting='soft')
    model.fit(np.array(xTrain), yTrain)    
    probs = model.predict_proba(np.array(xTest))
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(yTest, probs, pos_label = 1)
    aucRoc = auc(fpr, tpr)
    tprI = np.interp(meanFPR, fpr, tpr)
    tprs.append(tprI)
    aucs.append(aucRoc)    
    i = i + 1    
meanTPR = np.mean(tprs, axis=0); meanTPR[0] = 0
meanAUC = auc(meanFPR, meanTPR)
plt.plot(meanFPR, meanTPR, label='Fusion (AUC= %0.2f)' % (meanAUC*100)) 

plt.ylabel("True positive rate")
plt.xlabel("False positive rate")
plt.legend()
plt.show()
print()
