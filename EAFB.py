## Farzad Zandi, 2023.
# Embedded Artificial Feeding Birds feature selection.
# Feature selection by Artificial Feeding Birds and Embedded Artificial Feeding Birds using Boosting algorithms.
# FS() function is Embedded step for Artificial Feeding Birds feature selection.
# To run without Embedded step, disable FS() function.

from array import array
from ctypes import sizeof
from operator import truediv
from pickle import FALSE
import random
import time
import math
from re import L
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from catboost import CatBoostClassifier,Pool, EFeaturesSelectionAlgorithm, EShapCalcType
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, RepeatedStratifiedKFold, KFold
from sklearn.feature_selection import chi2, SelectKBest, f_classif, f_regression, RFE, RFECV, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from logitboost import LogitBoost
import xgboost as xgb
import lightgbm as LightGBM
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline
from snapml import BoostingMachineClassifier as SnapBoostClassifier

print('Farzad Zandi, 2023.')
print('Feature selection with Embedded Artificial Feeding Birds.')
# Loading Data.
print('Loading data...')
data = pd.read_csv('/Extracted Features//AD.csv')
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
label = pd.DataFrame(label)
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))
data = pd.DataFrame(data)
k = 5 # 5 Flod cross validation.
teta = 0.8 # Feature importance.
[M, N] = data.shape # Data Dimension.

# Fly function.
def fly():
    return np.random.randint(2, size=(N))

# Walk function.
def walk(X, n, i):
    j = round(random.uniform(1, n-1))
    while j==i:
        j = round(random.uniform(1, n-1))
    x1 = np.copy(X[i,:])
    x2 = np.copy(X[j,:])
    idx = np.where(np.array(x1)!=np.array(x2))[0]
    delta = len(idx)/N
    if delta==0:
        delta = 0.001
    for j in range(N):
        x1[j] = x1[j] + random.uniform(-1,1)*delta
        if random.random() < 1/(1+math.exp(-x1[j])):
            X[i,j] = 1
        else:
            X[i,j] = 0
    return X

# Embedded step for Artificial Feeding Birds feature selection.
def fs(x):
    idx = np.where(np.array(x)==1)[0]
    xTrain, xTest, yTrain, yTest = train_test_split(data, label, test_size=0.2, random_state=0, shuffle=True)
    xTrain = xTrain.iloc[:, idx]
    xTest = xTest.iloc[:, idx]
    
    ## LogitBoost.    
    # lgb = LogitBoost(n_estimators = 100)
    # lgb.fit(xTrain, np.array(yTrain).ravel())
    # idxNew = lgb.feature_importances_>np.mean(lgb.feature_importances_)
    # idxNew = idx[idxNew]
    
    ## XGBoost.
    # xgb_model = xgb.XGBClassifier(objective='binary:logistic')
    # xgb_model.fit(xTrain, yTrain)
    # idxNew = xgb_model.feature_importances_>np.mean(xgb_model.feature_importances_)
    # idxNew = idx[idxNew]

    ## AdaBoost.
    ada = AdaBoostClassifier(n_estimators=100)
    ada.fit(xTrain, np.array(yTrain).ravel())
    idxNew = ada.feature_importances_>np.mean(ada.feature_importances_)
    idxNew = idx[idxNew]

    ## SnapBoost.
    snap = SnapBoostClassifier(n_estimators=100)
    snap.fit(xTrain, np.array(yTrain).ravel())
    idxNew = snap.feature_importances_>np.mean(snap.feature_importances_)
    idxNew = idx[idxNew]
    
    ## CatBoost.
    # cb = CatBoostClassifier(
    #     iterations=40, depth=10, verbose=False, loss_function='Logloss',
    #     learning_rate=1, od_type='Iter', early_stopping_rounds=5)
    # cb.fit(xTrain, yTrain)
    # idxNew = cb.feature_importances_>np.mean(cb.feature_importances_)
    # idxNew = idx[idxNew]
    
    ## LightGBM
    # lgbm = LightGBM.LGBMClassifier(n_estimators=100)
    # lgbm.fit(xTrain, np.array(yTrain).ravel())
    # idxNew = lgbm.feature_importances_>np.mean(lgbm.feature_importances_)
    # idxNew = idx[idxNew]

    idxNew = np.int64(idxNew)
    x[idx] = 0
    x[idxNew] = 1
    return x

# Cost function.
def cost(x):
    idx = np.where(np.array(x)==1)[0]
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    
    ## LogitBoost.    
    # lgb = LogitBoost(n_estimators = 100)
    # acc = cross_val_score(lgb, data.iloc[:,idx], np.array(label).ravel(), scoring='accuracy', cv=cv)
    
    ## XGBoost.
    # xgb_model = xgb.XGBClassifier(n_estimators=100)
    # acc = cross_val_score(xgb_model, data.iloc[:,idx], np.array(label).ravel(), scoring='accuracy', cv=cv)
    
    ## AdaBoost.
    ada = AdaBoostClassifier(n_estimators=100)
    acc = cross_val_score(ada, data.iloc[:,idx], np.array(label).ravel(), scoring='accuracy', cv=cv)
    
    ## CatBoost.
    # cb = CatBoostClassifier()
    # acc = cross_val_score(cb, data.iloc[:,idx], np.array(label).ravel(), scoring='accuracy', cv=cv)
    
    ## LightGBM
    # lgbm = LightGBM.LGBMClassifier(n_estimators=100)
    # acc = cross_val_score(lgbm, data.iloc[:,idx], np.array(label).ravel(), scoring='accuracy', cv=cv)

    ## SnapBoost
    # snap = SnapBoostClassifier(n_estimators=100)
    # acc = cross_val_score(snap, data.iloc[:,idx], np.array(label).ravel(), scoring='accuracy', cv=cv)
    
    acc = np.mean(acc)
    fit = (teta*acc + (1-teta)*(1-(sum(np.array(x)==1))/N))    
    
    return fit

## Main code.
# Initializing.
pigs = 10 # Number of pigs
r = 0.75 # Rate of small & big birds.
p2 = p3 = p4 = 0.3
maxIter = 50 # Maximum iteration.
fitness = array('f', [])
fitnessNew = array('f', [])
X = np.empty((0,N), float) # pigs positions.
m = array('i',[])
s = array('i',[])

for i in range(pigs):
    sT = time.time()
    x = np.array(fly()) # Initializing pig position.
    x = fs(x)
    fitness.insert(i, cost(x)) # calculating pig fitness.
    eT = time.time()
    print(eT-sT)
    X = np.append(X, [x], axis= 0)    
    m.insert(i, 2)
    if i <= r*pigs:
        s.insert(i, 0)
    else:
        s.insert(i, 1)
fBest = np.copy(fitness)
xBest = np.copy(X)

for t in range(maxIter):
    for i in range(pigs):
        if (m[i]==2) | (m[i]==3) | (m[i]==4) | (fitness[i] == fBest[i]):
            p = 1
        else:
            if s[i]==0:
                p = random.uniform(p4, 1)
            else:
                p = random.random()
        if p >= p2+p3+p4:
            m[i] = 1
            X = walk(X, pigs, i)
            X[i,:] = fs(X[i,:])
            fitness[i] = cost(X[i,:])
        else:
            if p >= p3+p4:
                m[i] = 2
                X[i,:] = np.array(fly())
                X[i,:] = fs(X[i,:])
                fitness[i] = cost(X[i,:])
            else:
                if p >= p4:
                    m[i] = 3
                    X[i,:] = xBest[i,:]
                    fitness[i] = fBest[i]
        if p < p4:
            m[i] = 4
            j = round(random.uniform(1, pigs-1))
            while j==i:
                j = round(random.uniform(1, pigs-1))
            X[i,:] = X[j,:]
            fitness[i] = fitness[j]
        if fitness[i] >= fBest[i]:
            xBest[i,:] = X[i,:]
            fBest[i] = fitness[i]    
    print('Maximum fitness in iteration ', t, 'is: ', max(fBest))
idx = np.array(fBest).argmax()
idx = X[idx,:]
idx = np.where(np.array(idx)==1)[0]
dataReduct = data.iloc[:,idx]
idx = pd.DataFrame(idx, columns=['idx'])
idx.to_csv("D:\\idxAD.csv")
dataReduct['label'] = label
dataReduct.to_csv("/Results/reductAD.csv")




