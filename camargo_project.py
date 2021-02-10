from __future__ import division
import numpy as np
from sklearn.metrics import r2_score

from scipy.stats import spearmanr

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
import multiprocessing as mp
import os
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def score(a,b,dimension):
# a is predict, b is actual. dimension is len(train[0]).
    aa=a.copy()
    bb=b.copy()
    if len(aa)!=len(bb):
        print('not same length')
        return np.nan

    cc=aa-bb
    # wcpfh=sum(cc**2)

    # RR means R_Square
    RR=1-sum((bb-aa)**2)/sum((bb-np.mean(bb))**2)

    n=len(aa)
    p=dimension
    Adjust_RR=1-((1-RR)*(n-1)/(n-p-1))
    # Adjust_RR means Adjust_R_Square

    return RR,Adjust_RR


def bal_score(y_test,y_pred):
    n0 = 0
    n1 = 0

    rr=recall_score(y_test,y_pred)
    # y_pred = y_pred.reshape(1, -1)
    # print(y_test.shape, y_pred.shape)
    n1=np.sum(y_test == 0)


    for i in range(y_pred.shape[0]):
        if y_test[i] == 0 and y_pred[i] == 1:
            n0 = n0 + 1


    pf = n0 / n1

    bal = 1 - np.sqrt(pf * pf + (1 - rr) * (1 - rr)) / np.sqrt(2)
    return bal

def CE_score(SL,defnum, prob):
    df=pd.DataFrame(np.column_stack((SL,defnum,prob)))
    df.columns = ['CountLineCode', 'label', 'prob']
    df["den"] = df["label"] / df["CountLineCode"]

    b = shuffle(df)
    b1 = df.sort_values(by=["prob","CountLineCode"], ascending=[False,True])
    op = df.sort_values(by=["den","CountLineCode"], ascending=[False,True])

    x = []
    y = []
    line = 0
    defec = 0
    for i in range(df.shape[0]):
        line = line + b.iloc[i, 0] * 100
        defec = defec + b.iloc[i, 1] * 100
        x.append(line)
        y.append(defec)

    x1 = []
    y1 = []
    line = 0
    defec = 0
    for i in range(df.shape[0]):
        line = line + b1.iloc[i, 0] * 100
        defec = defec + b1.iloc[i, 1] * 100
        x1.append(line)
        y1.append(defec)

    x_op = []
    y_op = []
    line = 0
    defec = 0
    for i in range(df.shape[0]):
        line = line + op.iloc[i, 0] * 100
        defec = defec + op.iloc[i, 1] * 100
        x_op.append(line)
        y_op.append(defec)

    are_m=np.trapz(y1,x1)
    are_op=np.trapz(y_op,x_op)
    are_rom=np.trapz(y,x)
    ce=(are_m-are_rom)/(are_op-are_rom)

    return ce

def ER_score(SL,defnum, y_pred):
    ss=0
    st=0
    for i in range(y_pred.shape[0]):
        ss=ss+SL[i]*y_pred[i]
        st=st+SL[i]

    eff_m=ss/st

    sf=0
    ff=0

    for i in range(y_pred.shape[0]):
        sf=sf+defnum[i]*y_pred[i]
        ff=ff+defnum[i]

    eff_rand=sf/ff

    if (eff_rand == 0):
        eff = 0
    else:

        # print(eff_rand,eff_m)

        eff = (eff_rand - eff_m) / eff_rand

    return eff

def trans(x_train,x_test):
    for i in range(0,x_train.shape[1] - 2):
        x_train.iloc[:,i]=np.log1p(x_train.iloc[:,i])
    for i in range(0,x_test.shape[1] - 2):
        x_test.iloc[:,i]=np.log1p(x_test.iloc[:,i])
    b=[]
    # print(X_train.shape)
    for i in range(0,x_train.shape[1]-2):
        b.append(x_train.iloc[:,i].median() - x_test.iloc[:,i].median())

    for j in range(0,x_train.shape[1]-2):
        x_train.iloc[:,j] = x_train.iloc[:,j] + b[j]

    for j in range(0,x_test.shape[1]-2):
        x_test.iloc[:,j] = x_test.iloc[:,j] + b[j]

    return x_train,x_test


def fun(file):
    df1 = pd.read_csv('/home/local/SAIL/linagong/mygrap/data2/activemq_class_SM.csv')
    df2 = pd.read_csv('/home/local/SAIL/linagong/mygrap/data2/ActiveMQ5.1.0_SM.csv')
    df3 = pd.read_csv('/home/local/SAIL/linagong/mygrap/data2/ActiveMQ5.2.0_SM.csv')
    df4= pd.read_csv('/home/local/SAIL/linagong/mygrap/data2/ActiveMQ5.3.0_SM.csv')
    df5= pd.read_csv('E:/mygrap/new/data2/aa/Wicket1.5.3_SM.csv')

    df = pd.concat([df1, df2,df3,df4,df5], axis=0)
    # print(df.shape)
    df.drop_duplicates()
    df.dropna(axis=0, how='any')


    s = 'E:/mygrap/new/cross-project/Wicket/'+file

    df_test1 = pd.read_csv(s)

    p_COM = ['Densit(in)','pWeakC(in)',	'nBroke(in)',	'nEgoBe(in)','pWeakC(un)',	'nEgoBe(un)',	'CountDeclMethodDefault','AvgLineComment',
       'CountDeclMethodProtected',	'RatioCommentToCode','AvgLineBlank','MaxInheritanceTree',	'CountClassDerived',	'CountInput_Min',
       'CountOutput_Max','CountOutput_Min',	'MaxNesting_Min']


    # pca_MET = PCA(n_components=0.95)
    # pca_SM = PCA(n_components=0.95)
    # pca_SNA = PCA(n_components=0.95)
    # pca_COM = PCA(n_components=0.95)
    # pca_ENs = PCA(n_components=0.95)
    # pca_GNs = PCA(n_components=0.95)



    n_repeats = 100

    # 预测缺陷数指标
    mae_SM = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    adj_SM = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    co_SM = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    rmse_SM = np.zeros(shape=[n_repeats, 1], dtype=np.float64)

    mae_MET = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    adj_MET = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    co_MET = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    rmse_MET = np.zeros(shape=[n_repeats, 1], dtype=np.float64)

    mae_SNA = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    adj_SNA = np.zeros(shape=[n_repeats, 1])
    co_SNA = np.zeros(shape=[n_repeats, 1])
    rmse_SNA = np.zeros(shape=[n_repeats, 1], dtype=np.float64)

    mae_ENs = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    adj_ENs = np.zeros(shape=[n_repeats, 1])
    co_ENs= np.zeros(shape=[n_repeats, 1])
    rmse_ENs = np.zeros(shape=[n_repeats, 1], dtype=np.float64)

    mae_GNs= np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    adj_GNs= np.zeros(shape=[n_repeats, 1])
    co_GNs = np.zeros(shape=[n_repeats, 1])
    rmse_GNs = np.zeros(shape=[n_repeats, 1], dtype=np.float64)

    mae_COM = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    adj_COM = np.zeros(shape=[n_repeats, 1])
    co_COM = np.zeros(shape=[n_repeats, 1])
    rmse_COM = np.zeros(shape=[n_repeats, 1], dtype=np.float64)

    # 预测错误倾向性指标
    rr_SM = np.zeros(shape=[n_repeats, 1])
    auc_SM = np.zeros(shape=[n_repeats, 1])
    mcc_SM = np.zeros(shape=[n_repeats, 1])
    pre_SM = np.zeros(shape=[n_repeats, 1])
    er_SM = np.zeros(shape=[n_repeats, 1])
    ce_SM = np.zeros(shape=[n_repeats, 1])
    bri_SM = np.zeros(shape=[n_repeats, 1])

    rr_MET = np.zeros(shape=[n_repeats, 1])
    pre_MET=np.zeros(shape=[n_repeats,1])
    auc_MET = np.zeros(shape=[n_repeats, 1])
    mcc_MET = np.zeros(shape=[n_repeats, 1])
    er_MET = np.zeros(shape=[n_repeats, 1])
    ce_MET = np.zeros(shape=[n_repeats, 1])
    bri_MET=np.zeros(shape=[n_repeats,1])

    rr_SNA = np.zeros(shape=[n_repeats, 1])
    pre_SNA=np.zeros(shape=[n_repeats,1])
    auc_SNA = np.zeros(shape=[n_repeats, 1])
    mcc_SNA = np.zeros(shape=[n_repeats, 1])
    # bal_SNA = np.zeros(shape=[n_repeats, 1])
    er_SNA = np.zeros(shape=[n_repeats, 1])
    ce_SNA = np.zeros(shape=[n_repeats, 1])
    bri_SNA=np.zeros(shape=[n_repeats,1])

    rr_ENs = np.zeros(shape=[n_repeats, 1])
    pre_ENs = np.zeros(shape=[n_repeats, 1])
    auc_ENs = np.zeros(shape=[n_repeats, 1])
    mcc_ENs = np.zeros(shape=[n_repeats, 1])
    # bal_SNA = np.zeros(shape=[n_repeats, 1])
    er_ENs = np.zeros(shape=[n_repeats, 1])
    ce_ENs = np.zeros(shape=[n_repeats, 1])
    bri_ENs = np.zeros(shape=[n_repeats, 1])

    rr_GNs = np.zeros(shape=[n_repeats, 1])
    pre_GNs = np.zeros(shape=[n_repeats, 1])
    auc_GNs = np.zeros(shape=[n_repeats, 1])
    mcc_GNs = np.zeros(shape=[n_repeats, 1])
    # bal_SNA = np.zeros(shape=[n_repeats, 1])
    er_GNs = np.zeros(shape=[n_repeats, 1])
    ce_GNs = np.zeros(shape=[n_repeats, 1])
    bri_GNs = np.zeros(shape=[n_repeats, 1])

    rr_COM = np.zeros(shape=[n_repeats, 1])
    pre_COM = np.zeros(shape=[n_repeats, 1])
    auc_COM = np.zeros(shape=[n_repeats, 1])
    mcc_COM = np.zeros(shape=[n_repeats, 1])
    # bal_COM = np.zeros(shape=[n_repeats, 1])
    er_COM = np.zeros(shape=[n_repeats, 1])
    ce_COM = np.zeros(shape=[n_repeats, 1])
    bri_COM = np.zeros(shape=[n_repeats, 1])


    metr_MET = np.empty(shape=[n_repeats, 1], dtype=np.object)
    metr_SNA = np.empty(shape=[n_repeats, 1], dtype=np.object)
    metr_SM = np.empty(shape=[n_repeats, 1], dtype=np.object)
    metr_ENs = np.empty(shape=[n_repeats, 1], dtype=np.object)
    metr_GNs= np.empty(shape=[n_repeats, 1], dtype=np.object)
    metr_COM = np.empty(shape=[n_repeats, 1], dtype=np.object)

    metr_MET[0:n_repeats, :1] = "MET"
    metr_SNA[0:n_repeats, :1] = "SNA"
    metr_SM[0:n_repeats, :1] = "SM"
    metr_ENs[0:1000, :1] = "ENs"
    metr_GNs[0:1000, :1] = "GNs"
    metr_COM[0:1000, :1] = "COM"




    Adj = np.empty(shape=[n_repeats, 1], dtype=np.object)
    Adj[0:n_repeats, :1] = file
    Spearman = np.empty(shape=[n_repeats, 1], dtype=np.object)
    Spearman[0:n_repeats, :1] = file
    MAE=np.empty(shape=[n_repeats, 1], dtype=np.object)
    MAE[0:n_repeats, :1] = file
    RMSE = np.empty(shape=[n_repeats, 1], dtype=np.object)
    RMSE[0:n_repeats, :1] = file

    Recall = np.empty(shape=[n_repeats, 1], dtype=np.object)
    Recall[0:n_repeats, :1] = file
    AUC = np.empty(shape=[n_repeats, 1], dtype=np.object)
    AUC[0:n_repeats, :1] = file
    MCC = np.empty(shape=[n_repeats, 1], dtype=np.object)
    MCC[0:n_repeats, :1] = file
    Bri=np.empty(shape=[n_repeats, 1],dtype=np.object)
    Bri[0:n_repeats, :1] = file
    Pre=np.empty(shape=[n_repeats, 1],dtype=np.object)
    Pre[0:n_repeats, :1] = file

    CE = np.empty(shape=[n_repeats, 1], dtype=np.object)
    CE[0:n_repeats, :1] = file
    ER = np.empty(shape=[n_repeats, 1], dtype=np.object)
    ER[0:n_repeats, :1] = file

    lrclass_MET = GaussianNB()
    lrclass_SM = GaussianNB()
    lrclass_SNA = GaussianNB()
    lrclass_COM = GaussianNB()
    lrclass_ENs = GaussianNB()
    lrclass_GNs = GaussianNB()
    # y= df['defect'].values
    pipe_rf = Pipeline([('clf',BayesianRidge())])
    param_range_alpha_1 = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
    param_range_lambda_1 = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
    param_range_alpha_2 = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
    param_range_lambda_2 = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
    param_grid = [{'clf__alpha_1': param_range_alpha_1,'clf__lambda_1': param_range_lambda_1,'clf__alpha_2': param_range_alpha_2,'clf__lambda_2': param_range_lambda_2}]
    # # Perform grid search cross validation on the parameters listed in param_grid, using accuracy as the measure of fit and number of folds (CV) = 5
    lrgre_MET = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=5)
    lrgre_SM =GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=5)
    lrgre_SNA = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=5)
    lrgre_COM =GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=5)
    lrgre_ENs =GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=5)
    lrgre_GNs =GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=5)

    for i in range(0, n_repeats):
        boot = np.random.choice(df.shape[0], df.shape[0], replace=True)
        x_train = df.iloc[boot]
        x_train = x_train.drop(["ID",], axis=1)

        x_test = df_test1.iloc[:, :]
        x_test = x_test.drop(["ID"], axis=1)
        # x_COM_test = X_test[p_COM]



        # print(x_train.shape,x_test.shape)

        print(file)
        df_train,df_test=trans(x_train,x_test)
        df_train=df_train.dropna(axis=0, how='any')
        df_test=df_test.dropna(axis=0, how='any')

        # print(df_train)

        x_train=df_train.iloc[:,:118]
        y_train=df_train["defect"].values
        y_num_train=df_train["label"].values
        # print(y_train)

        x_test = df_test.iloc[:, :118]
        y_test = df_test["defect"].values
        y_num_test = df_test["label"].values

        x_COM_train = df_train[p_COM]
        x_COM_test = df_test[p_COM]

        sumcode = df_test["CountLineCode"].sum()
        sumdef = df_test["label"].sum()


        x_SM_train = x_train.iloc[:, :118]
        x_ENs_train = x_train.iloc[:, :40]
        x_GNs_train = x_train.iloc[:, 40:64]
        x_MET_train = x_train.iloc[:, 64:118]
        x_SNA_train = x_train.iloc[:, :64]


        # print(x_SNA_train.shape, x_MET_train.shape, x_SM_train.shape)


        y_num_train =np.log1p(y_num_train)
        y_num_test = np.log1p(y_num_test)

        x_SM_test = x_test.iloc[:, :118]
        x_ENs_test = x_test.iloc[:, :40]
        x_GNs_test= x_test.iloc[:, 40:64]
        x_MET_test = x_test.iloc[:, 64:118]
        x_SNA_test = x_test.iloc[:, :64]

        x_SM_train,x_SM_test=trans(x_SM_train,x_SM_test)
        x_MET_train, x_MET_test = trans(x_MET_train, x_MET_test)
        x_SNA_train, x_SNA_test = trans(x_SNA_train, x_SNA_test)
        x_COM_train, x_COM_test = trans(x_COM_train, x_COM_test)
        x_ENs_train, x_ENs_test = trans(x_ENs_train, x_ENs_test)
        x_GNs_train, x_GNs_test = trans(x_GNs_train, x_GNs_test)

        # x_SM_train2 =pca_SM.fit_transform(x_SM_train)
        # x_MET_train2 =pca_MET.fit_transform(x_MET_train)
        # x_SNA_train2 = pca_SNA.fit_transform(x_SNA_train)
        # x_COM_train2 =pca_COM.fit_transform(x_COM_train)
        # # x_ISM_train2=pca_ISM.fit_transform(x_ISM_train)
        # x_ENs_train2 =pca_ENs.fit_transform(x_ENs_train)
        # x_GNs_train2 = pca_GNs.fit_transform(x_GNs_train)
        #
        # x_SM_test2 =pca_SM.transform(x_SM_test)
        # x_MET_test2 =pca_MET.transform(x_MET_test)
        # x_SNA_test2 =pca_SNA.transform(x_SNA_test)
        # x_COM_test2 =pca_COM.transform(x_COM_test)
        # x_ENs_test2 =pca_ENs.transform(x_ENs_test)
        # x_GNs_test2 = pca_GNs.transform(x_GNs_test)


        n=x_test.shape[0]

        # 使用SM预测错误数
        lrgre_SM.fit(x_SM_train, y_num_train)
        y_pred_SM = lrgre_SM.predict(x_SM_test)

        RR=r2_score(y_num_test, y_pred_SM)
        adj_SM[i]=1-((1-RR)*(n-1)/(n-x_SM_test.shape[1]-1))
        co_SM[i], _ = spearmanr(y_num_test, y_pred_SM)
        rmse_SM[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_SM))
        mae_SM[i]=mean_absolute_error(y_num_test, y_pred_SM)

        # 使用SM预测错误倾向性

        lrclass_SM.fit(x_SM_train, y_train)
        y_pred_SM1 = lrclass_SM.predict(x_SM_test)
        y_prob_SM = lrclass_SM.predict_proba(x_SM_test)



        pre_SM[i] = precision_score(y_test, y_pred_SM1)
        rr_SM[i] = recall_score(y_test, y_pred_SM1)
        auc_SM[i] = roc_auc_score(y_test, y_prob_SM[:, 1])
        mcc_SM[i] = matthews_corrcoef(y_test, y_pred_SM1)
        # bal_SM[i] = bal_score(y_test, y_pred_SM1)
        ce_SM[i] = CE_score(df_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_SM[:, 1])
        er_SM[i] = ER_score(df_test["CountLineCode"].values, y_num_test, y_pred_SM1)
        bri_SM[i] = brier_score_loss(y_test, y_prob_SM[:, 1])

        # 使用MET度量元预测缺陷数
        lrgre_MET.fit(x_MET_train, y_num_train)
        y_pred_MET = lrgre_MET.predict(x_MET_test)

        RR = r2_score(y_num_test, y_pred_MET)
        adj_MET[i] = 1 - ((1 - RR) * (n - 1) / (n - x_MET_test.shape[1] - 1))
        co_MET[i], _ = spearmanr(y_num_test, y_pred_MET)
        rmse_MET[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_MET))
        mae_MET[i] = mean_absolute_error(y_num_test, y_pred_MET)

        # 使用MET预测倾向性
        lrclass_MET.fit(x_MET_train, y_train)
        y_pred_MET1 =lrclass_MET.predict(x_MET_test)
        y_prob_MET =lrclass_MET.predict_proba(x_MET_test)

        pre_MET[i] = precision_score(y_test, y_pred_MET1)
        rr_MET[i] = recall_score(y_test, y_pred_MET1)
        auc_MET[i] = roc_auc_score(y_test, y_prob_MET[:, 1])
        bri_MET[i] = brier_score_loss(y_test, y_prob_MET[:, 1])
        mcc_MET[i] = matthews_corrcoef(y_test, y_pred_MET1)
        # bal_MET[i] = bal_score(y_test, y_pred_MET1)
        ce_MET[i] = CE_score(df_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_MET[:, 1])
        er_MET[i] = ER_score(df_test["CountLineCode"].values, y_num_test, y_pred_MET1)

        # 使用SNA预测缺陷数
        lrgre_SNA.fit(x_SNA_train, y_num_train)
        y_pred_SNA = lrgre_SNA.predict(x_SNA_test)

        RR = r2_score(y_num_test, y_pred_SNA)
        adj_SNA[i] = 1 - ((1 - RR) * (n - 1) / (n - x_SNA_test.shape[1] - 1))
        co_SNA[i], _ = spearmanr(y_num_test, y_pred_SNA)
        rmse_SNA[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_SNA))
        mae_SNA[i] = mean_absolute_error(y_num_test, y_pred_SNA)

        # 使用SNA预测倾向性
        lrclass_SNA.fit(x_SNA_train, y_train)
        y_pred_SNA1 = lrclass_SNA.predict(x_SNA_test)
        y_prob_SNA = lrclass_SNA.predict_proba(x_SNA_test)

        pre_SNA[i] = precision_score(y_test, y_pred_SNA1)
        rr_SNA[i] = recall_score(y_test, y_pred_SNA1)
        auc_SNA[i] = roc_auc_score(y_test, y_prob_SNA[:, 1])
        bri_SNA[i] = brier_score_loss(y_test, y_prob_SNA[:, 1])
        mcc_SNA[i] = matthews_corrcoef(y_test, y_pred_SNA1)
        # bal_SNA[i] = bal_score(y_test, y_pred_SNA1)
        ce_SNA[i] = CE_score(df_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_SNA[:, 1])
        er_SNA[i] = ER_score(df_test["CountLineCode"].values, y_num_test, y_pred_SNA1)

        # 使用ENs度量元预测缺陷数
        lrgre_ENs.fit(x_ENs_train, y_num_train)
        y_pred_ENs = lrgre_ENs.predict(x_ENs_test)

        RR = r2_score(y_num_test, y_pred_ENs)
        adj_ENs[i] = 1 - ((1 - RR) * (n - 1) / (n - x_ENs_test.shape[1] - 1))
        co_ENs[i], _ = spearmanr(y_num_test, y_pred_ENs)
        rmse_ENs[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_ENs))
        mae_ENs[i] = mean_absolute_error(y_num_test, y_pred_ENs)

        # 使用ENs预测倾向性
        lrclass_ENs.fit(x_ENs_train, y_train)
        y_pred_ENs1 =lrclass_ENs.predict(x_ENs_test)
        y_prob_ENs =lrclass_ENs.predict_proba(x_ENs_test)

        pre_ENs[i] = precision_score(y_test, y_pred_ENs1)
        rr_ENs[i] = recall_score(y_test, y_pred_ENs1)
        auc_ENs[i] = roc_auc_score(y_test, y_prob_ENs[:, 1])
        bri_ENs[i] = brier_score_loss(y_test, y_prob_ENs[:, 1])
        mcc_ENs[i] = matthews_corrcoef(y_test, y_pred_ENs1)
        # bal_SNA[i] = bal_score(y_test, y_pred_SNA1)
        ce_ENs[i] = CE_score(df_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_ENs[:, 1])
        er_ENs[i] = ER_score(df_test["CountLineCode"].values, y_num_test, y_pred_ENs1)
        #
        # # 使用GNs度量元预测缺陷数
        lrgre_GNs.fit(x_GNs_train, y_num_train)
        y_pred_GNs = lrgre_GNs.predict(x_GNs_test)

        RR = r2_score(y_num_test, y_pred_GNs)
        adj_GNs[i] = 1 - ((1 - RR) * (n - 1) / (n - x_GNs_test.shape[1] - 1))
        co_GNs[i], _ = spearmanr(y_num_test, y_pred_GNs)
        rmse_GNs[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_GNs))
        mae_GNs[i] = mean_absolute_error(y_num_test, y_pred_GNs)

        # 使用GNs预测倾向性
        lrclass_GNs.fit(x_GNs_train, y_train)
        y_pred_GNs1 = lrclass_GNs.predict(x_GNs_test)
        y_prob_GNs = lrclass_GNs.predict_proba(x_GNs_test)

        pre_GNs[i] = precision_score(y_test, y_pred_GNs1)
        rr_GNs[i] = recall_score(y_test, y_pred_GNs1)
        auc_GNs[i] = roc_auc_score(y_test, y_prob_GNs[:, 1])
        bri_GNs[i] = brier_score_loss(y_test, y_prob_GNs[:, 1])
        mcc_GNs[i] = matthews_corrcoef(y_test, y_pred_GNs1)
        # bal_SNA[i] = bal_score(y_test, y_pred_SNA1)
        ce_GNs[i] = CE_score(df_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_GNs[:, 1])
        er_GNs[i] = ER_score(df_test["CountLineCode"].values, y_num_test, y_pred_GNs1)


        # # 使用COM预测倾向性
        lrgre_COM.fit(x_COM_train, y_num_train)
        y_pred_COM = lrgre_COM.predict(x_COM_test)

        RR = r2_score(y_num_test, y_pred_COM)
        adj_COM[i] = 1 - ((1 - RR) * (n - 1) / (n - x_COM_test.shape[1] - 1))
        co_COM[i], _ = spearmanr(y_num_test, y_pred_COM)
        rmse_COM[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_COM))
        mae_COM[i] = mean_absolute_error(y_num_test, y_pred_COM)

        # 使用COM预测倾向性
        lrclass_COM.fit(x_COM_train, y_train)
        y_pred_COM1 = lrclass_COM.predict(x_COM_test)
        y_prob_COM = lrclass_COM.predict_proba(x_COM_test)

        pre_COM[i] = precision_score(y_test, y_pred_COM1)
        rr_COM[i] = recall_score(y_test, y_pred_COM1)
        auc_COM[i] = roc_auc_score(y_test, y_prob_COM[:, 1])
        bri_COM[i] = brier_score_loss(y_test, y_prob_COM[:, 1])
        mcc_COM[i] = matthews_corrcoef(y_test, y_pred_COM1)
        # bal_SNA[i] = bal_score(y_test, y_pred_SNA1)
        ce_COM[i] = CE_score(df_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_COM[:, 1])
        er_COM[i] = ER_score(df_test["CountLineCode"].values, y_num_test, y_pred_COM1)
        print(file)
        print(auc_MET[i],auc_SNA[i],auc_ENs[i],auc_GNs[i],auc_SM[i],auc_COM[i])

        # prob = pd.DataFrame(np.column_stack((y_prob_MET[:, 1], y_prob_SNA[:, 1], y_prob_ENs[:, 1], y_prob_GNs[:, 1],
        #                                      y_prob_SM[:, 1], y_prob_COM[:, 1], y_test)))
        # prob.columns = ['MET', 'SNA', 'ENs', 'GNs', 'SM', 'COM', 'True']
        # prob.to_csv('/home/local/SAIL/linagong/mygrap/crossproject/number/ActiveMQ/' + file + '_' + str(i) + '.csv', index=False)
        #
        # number = pd.DataFrame(np.column_stack((y_pred_MET, y_pred_SNA, y_pred_ENs, y_pred_GNs, y_pred_SM, y_pred_COM, y_num_test)))
        # number.columns = ['MET', 'SNA', 'ENs', 'GNs', 'SM', 'COM', 'True']
        # number.to_csv('/home/local/SAIL/linagong/mygrap/crossproject/proness/ActiveMQ/' + file + '_' + str(i) + '.csv', index=False)



    data_Adj=pd.DataFrame(np.column_stack((adj_MET,adj_SNA,adj_SM,adj_ENs,adj_GNs,adj_COM)))
    data_Adj.columns = ['MET', 'SNA', 'SM',"ENs","GNs","COM"]


    data_MAE = pd.DataFrame(np.column_stack((mae_MET, mae_SNA, mae_SM, mae_ENs, mae_GNs, mae_COM)))
    data_MAE.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_Spearman = pd.DataFrame(np.column_stack((co_MET, co_SNA, co_SM, co_ENs, co_GNs, co_COM)))
    data_Spearman.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_RMSE = pd.DataFrame(np.column_stack((rmse_MET, rmse_SNA, rmse_SM, rmse_ENs, rmse_GNs, rmse_COM)))
    data_RMSE.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_Precision = pd.DataFrame(np.column_stack((pre_MET,pre_SNA, pre_SM, pre_ENs, pre_GNs, pre_COM)))
    data_Precision.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_Recall = pd.DataFrame(np.column_stack((rr_MET, rr_SNA, rr_SM, rr_ENs, rr_GNs, rr_COM)))
    data_Recall.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_AUC = pd.DataFrame(np.column_stack((auc_MET,auc_SNA, auc_SM, auc_ENs, auc_GNs, auc_COM)))
    data_AUC.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_Bri = pd.DataFrame(np.column_stack((bri_MET, bri_SNA, bri_SM, bri_ENs, bri_GNs, bri_COM)))
    data_Bri.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_MCC = pd.DataFrame(np.column_stack((mcc_MET, mcc_SNA, mcc_SM, mcc_ENs, mcc_GNs, mcc_COM)))
    data_MCC.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_CE = pd.DataFrame(np.column_stack((ce_MET, ce_SNA, ce_SM, ce_ENs, ce_GNs, ce_COM)))
    data_CE.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]

    data_ER = pd.DataFrame(np.column_stack((er_MET, er_SNA, er_SM, er_ENs, er_GNs, er_COM)))
    data_ER.columns = ['MET', 'SNA', 'SM', "ENs", "GNs", "COM"]




    data_Adj.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'Adj_' + file, index=False)

    data_MAE.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'MAE_' + file, index=False)

    data_Spearman.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'Spearman_' + file, index=False)
    #

    data_RMSE.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'RMSE_' + file, index=False)

    data_Recall.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'Recall_' + file, index=False)

    data_AUC.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'AUC_' + file, index=False)

    data_MCC.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'MCC_' + file, index=False)

    data_Bri.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'Bri_' + file, index=False)

    data_CE.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'CE_' + file, index=False)

    data_ER.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'ER_' + file, index=False)

    data_Precision.to_csv('E:/mygrap/new/cross-project/new/Wicket/' + 'Precision_' + file, index=False)
    return 1


if __name__ == '__main__':
    # N = mp.cpu_count()
    g1 = os.scandir(r"E:/mygrap/new/cross-project/Wicket/")
    N=10

    with mp.Pool(processes=N) as p:
        results = p.map(fun, [file.name for file in g1])








