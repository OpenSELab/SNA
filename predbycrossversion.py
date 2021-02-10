from __future__ import division
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

def score(a,b,dimension):
# a is predict, b is actual. dimension is len(train[0]).
    aa=a.copy(); bb=b.copy()
    if len(aa)!=len(bb):
        print('not same length')
        return np.nan

    cc=aa-bb
    wcpfh=sum(cc**2)

    # RR means R_Square
    RR=1-sum((bb-aa)**2)/sum((bb-np.mean(bb))**2)

    n=len(aa); p=dimension
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
    b1 = df.sort_values(by="prob", ascending=False)
    op = df.sort_values(by="den", ascending=False)

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



if __name__ == '__main__':
    df_COM = pd.read_csv('C:/gln/mygrap/new/Cross-version/common1/Lucene3.1.0_CCOM.csv')

    df1 = pd.read_csv('C:/gln/mygrap/new/aa/lucene_class_SM.csv')
    df2 = pd.read_csv('C:/gln/mygrap/new/aa/Lucene2.9.0_SM.csv')
    df3 = pd.read_csv('C:/gln/mygrap/new/aa/Lucene3.0.0_SM.csv')
    # df4= pd.read_csv('C:/gln/mygrap/new/aa/ActiveMQ5.3.0_SM.csv')
    # df5= pd.read_csv('E:\\mygrap\\new\\data\\ActiveMQ5.8.0_SM.csv')

    df = pd.concat([df1], axis=0)
    # print(df.shape)
    # df.drop_duplicates()

    p_COM = []
    s_SM = df_COM.columns.astype(str)

    for k in range(0, df_COM.shape[1] - 3):
        p_COM.append(s_SM[k])

    file='Lucene2.9.0_SM.csv'

    s = 'C:/gln/mygrap/new/aa/' + file

    df_test = pd.read_csv(s)

    # p_COM = ['Densit(in)', 'pWeakC(in)', 'nBroke(in)', 'nEgoBe(in)', 'pWeakC(un)', 'nEgoBe(un)',
    #          'CountDeclMethodDefault', 'AvgLineComment',
    #          'CountDeclMethodProtected', 'RatioCommentToCode', 'AvgLineBlank', 'MaxInheritanceTree',
    #          'CountClassDerived', 'CountInput_Min',
    #          'CountOutput_Max', 'CountOutput_Min', 'MaxNesting_Min']
    Sta_SM = StandardScaler()
    Sta_SNA = StandardScaler()
    Sta_MET = StandardScaler()
    Sta_GN = StandardScaler()
    Sta_EN = StandardScaler()
    Sta_COM = StandardScaler()

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
    co_ENs = np.zeros(shape=[n_repeats, 1])
    rmse_ENs = np.zeros(shape=[n_repeats, 1], dtype=np.float64)

    mae_GNs = np.zeros(shape=[n_repeats, 1], dtype=np.float64)
    adj_GNs = np.zeros(shape=[n_repeats, 1])
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
    pre_MET = np.zeros(shape=[n_repeats, 1])
    auc_MET = np.zeros(shape=[n_repeats, 1])
    mcc_MET = np.zeros(shape=[n_repeats, 1])
    er_MET = np.zeros(shape=[n_repeats, 1])
    ce_MET = np.zeros(shape=[n_repeats, 1])
    bri_MET = np.zeros(shape=[n_repeats, 1])

    rr_SNA = np.zeros(shape=[n_repeats, 1])
    pre_SNA = np.zeros(shape=[n_repeats, 1])
    auc_SNA = np.zeros(shape=[n_repeats, 1])
    mcc_SNA = np.zeros(shape=[n_repeats, 1])
    # bal_SNA = np.zeros(shape=[n_repeats, 1])
    er_SNA = np.zeros(shape=[n_repeats, 1])
    ce_SNA = np.zeros(shape=[n_repeats, 1])
    bri_SNA = np.zeros(shape=[n_repeats, 1])

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
    metr_GNs = np.empty(shape=[n_repeats, 1], dtype=np.object)
    metr_COM = np.empty(shape=[n_repeats, 1], dtype=np.object)

    metr_MET[0:n_repeats, :1] = "MET"
    metr_SNA[0:n_repeats, :1] = "SNA"
    metr_SM[0:n_repeats, :1] = "SM"
    metr_ENs[0:n_repeats, :1] = "ENs"
    metr_GNs[0:n_repeats, :1] = "GNs"
    metr_COM[0:n_repeats, :1] = "COM"

    Adj = np.empty(shape=[n_repeats, 1], dtype=np.object)
    Adj[0:n_repeats, :1] = file
    Spearman = np.empty(shape=[n_repeats, 1], dtype=np.object)
    Spearman[0:n_repeats, :1] = file
    MAE = np.empty(shape=[n_repeats, 1], dtype=np.object)
    MAE[0:n_repeats, :1] = file
    RMSE = np.empty(shape=[n_repeats, 1], dtype=np.object)
    RMSE[0:n_repeats, :1] = file

    Recall = np.empty(shape=[n_repeats, 1], dtype=np.object)
    Recall[0:n_repeats, :1] = file
    AUC = np.empty(shape=[n_repeats, 1], dtype=np.object)
    AUC[0:n_repeats, :1] = file
    MCC = np.empty(shape=[n_repeats, 1], dtype=np.object)
    MCC[0:n_repeats, :1] = file
    Bri = np.empty(shape=[n_repeats, 1], dtype=np.object)
    Bri[0:n_repeats, :1] = file
    Pre = np.empty(shape=[n_repeats, 1], dtype=np.object)
    Pre[0:n_repeats, :1] = file

    CE = np.empty(shape=[n_repeats, 1], dtype=np.object)
    CE[0:n_repeats, :1] = file
    ER = np.empty(shape=[n_repeats, 1], dtype=np.object)
    ER[0:n_repeats, :1] = file

    param_range_trees = [50,100,200,500,600]
    # param_range_criterion=['entropy']
    param_range_features = [3, 5, 10, 15]
    param_grid = [{'clf__n_estimators': param_range_trees, 'clf__max_depth': param_range_features}]
    pipe_rf = Pipeline([('clf', RandomForestClassifier(criterion='entropy'))])
    pipe_rf1 = Pipeline([('clf', RandomForestRegressor(criterion='mse'))])
    lrModel1 = GridSearchCV(estimator=pipe_rf1, param_grid=param_grid,cv=5, n_jobs=25)
    lrModel2 = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, cv=5, n_jobs=25)

    for i in range(0, n_repeats):
        # boot = resample([i for i in range(0, df.shape[0])], replace=True, stratify=y)
        X_train = df.iloc[:,:]
        print(X_train.shape)
        y_train = X_train['defect'].values
        y_num_train = X_train["label"].values
        x_COM_train = X_train[p_COM].values

        x_train = X_train.drop(["ID","defect", "label"], axis=1).values

        print(x_train.shape)

        # oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]  # testing data
        X_test = df_test.iloc[:, :]
        y_test = X_test['defect'].values
        y_num_test = X_test["label"].values
        x_test = X_test.drop(["ID","defect", "label"], axis=1).values
        x_COM_test = X_test[p_COM].values

        sumcode = X_test["CountLineCode"].sum()

        sumdef = X_test["label"].sum()

        x_SM_train = x_train[:, :118]
        x_ENs_train = x_train[:, :40]
        x_GNs_train = x_train[:, 40:64]
        x_MET_train = x_train[:, 64:118]
        x_SNA_train = x_train[:, :64]

        print(x_SNA_train.shape, x_MET_train.shape, x_SM_train.shape)

        y_num_train = np.log1p(y_num_train)
        y_num_test = np.log1p(y_num_test)

        x_SM_test = x_test[:, :118]
        x_ENs_test = x_test[:, :40]
        x_GNs_test = x_test[:, 40:64]
        x_MET_test = x_test[:, 64:118]
        x_SNA_test = x_test[:, :64]

        x_SM_train2 = Sta_SM.fit_transform(x_SM_train)
        x_ENs_train2 = Sta_EN.fit_transform(x_ENs_train)
        x_GNs_train2 = Sta_GN.fit_transform(x_GNs_train)
        x_MET_train2 = Sta_MET.fit_transform(x_MET_train)
        x_SNA_train2 = Sta_SNA.fit_transform(x_SNA_train)
        x_COM_train2 = Sta_COM.fit_transform(x_COM_train)

        x_SM_test2 = Sta_SM.transform(x_SM_test)
        x_ENs_test2 = Sta_EN.transform(x_ENs_test)
        x_GNs_test2 = Sta_GN.transform(x_GNs_test)
        x_MET_test2 = Sta_MET.transform(x_MET_test)
        x_SNA_test2 = Sta_SNA.transform(x_SNA_test)
        x_COM_test2 = Sta_COM.transform(x_COM_test)

        # x_SM_train2 = pca_SM.fit_transform(x_SM_train)
        # x_MET_train2 = pca_MET.fit_transform(x_MET_train)
        # x_SNA_train2 = pca_SNA.fit_transform(x_SNA_train)
        # x_COM_train2 = pca_COM.fit_transform(x_COM_train)
        #         # x_ISM_train2=pca_ISM.fit_transform(x_ISM_train)
        # x_ENs_train2 = pca_ENs.fit_transform(x_ENs_train)
        # x_GNs_train2 = pca_GNs.fit_transform(x_GNs_train)
                #
        # x_SM_test2 = pca_SM.transform(x_SM_test)
        # x_MET_test2 = pca_MET.transform(x_MET_test)
        # x_SNA_test2 = pca_SNA.transform(x_SNA_test)
        # x_COM_test2 = pca_COM.transform(x_COM_test)
        # x_ENs_test2 = pca_ENs.transform(x_ENs_test)
        # x_GNs_test2 = pca_GNs.transform(x_GNs_test)
               

        n = x_test.shape[0]

        # 使用SM预测错误数
        lrModel1.fit(x_SM_train2, y_num_train)
        y_pred_SM = lrModel1.predict(x_SM_test2)

        RR = r2_score(y_num_test, y_pred_SM)
        adj_SM[i] = 1 - ((1 - RR) * (n - 1) / (n - x_SM_test2.shape[1] - 1))
        co_SM[i], _ = spearmanr(y_num_test, y_pred_SM)
        rmse_SM[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_SM))
        mae_SM[i] = mean_absolute_error(y_num_test, y_pred_SM)

        # 使用SM预测错误倾向性

        lrModel2.fit(x_SM_train2, y_train)
        y_pred_SM1 = lrModel2.predict(x_SM_test2)
        y_prob_SM = lrModel2.predict_proba(x_SM_test2)

        pre_SM[i] = precision_score(y_test, y_pred_SM1)
        rr_SM[i] = recall_score(y_test, y_pred_SM1)
        auc_SM[i] = roc_auc_score(y_test, y_prob_SM[:, 1])
        mcc_SM[i] = matthews_corrcoef(y_test, y_pred_SM1)
        # bal_SM[i] = bal_score(y_test, y_pred_SM1)
        ce_SM[i] = CE_score(X_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_SM[:, 1])
        er_SM[i] = ER_score(X_test["CountLineCode"].values, y_num_test, y_pred_SM1)
        bri_SM[i] = brier_score_loss(y_test, y_prob_SM[:, 1])

        # 使用MET度量元预测缺陷数
        lrModel1.fit(x_MET_train2, y_num_train)
        y_pred_MET = lrModel1.predict(x_MET_test2)

        RR = r2_score(y_num_test, y_pred_MET)
        adj_MET[i] = 1 - ((1 - RR) * (n - 1) / (n - x_MET_test2.shape[1] - 1))
        co_MET[i], _ = spearmanr(y_num_test, y_pred_MET)
        rmse_MET[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_MET))
        mae_MET[i] = mean_absolute_error(y_num_test, y_pred_MET)

        # 使用MET预测倾向性
        lrModel2.fit(x_MET_train2, y_train)
        y_pred_MET1 = lrModel2.predict(x_MET_test2)
        y_prob_MET = lrModel2.predict_proba(x_MET_test2)

        pre_MET[i] = precision_score(y_test, y_pred_MET1)
        rr_MET[i] = recall_score(y_test, y_pred_MET1)
        auc_MET[i] = roc_auc_score(y_test, y_prob_MET[:, 1])
        bri_MET[i] = brier_score_loss(y_test, y_prob_MET[:, 1])
        mcc_MET[i] = matthews_corrcoef(y_test, y_pred_MET1)
        # bal_MET[i] = bal_score(y_test, y_pred_MET1)
        ce_MET[i] = CE_score(X_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_MET[:, 1])
        er_MET[i] = ER_score(X_test["CountLineCode"].values, y_num_test, y_pred_MET1)

        # 使用SNA预测缺陷数
        lrModel1.fit(x_SNA_train2, y_num_train)
        y_pred_SNA = lrModel1.predict(x_SNA_test2)

        RR = r2_score(y_num_test, y_pred_SNA)
        adj_SNA[i] = 1 - ((1 - RR) * (n - 1) / (n - x_SNA_test2.shape[1] - 1))
        co_SNA[i], _ = spearmanr(y_num_test, y_pred_SNA)
        rmse_SNA[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_SNA))
        mae_SNA[i] = mean_absolute_error(y_num_test, y_pred_SNA)

        # 使用SNA预测倾向性
        lrModel2.fit(x_SNA_train2, y_train)
        y_pred_SNA1 = lrModel2.predict(x_SNA_test2)
        y_prob_SNA = lrModel2.predict_proba(x_SNA_test2)

        pre_SNA[i] = precision_score(y_test, y_pred_SNA1)
        rr_SNA[i] = recall_score(y_test, y_pred_SNA1)
        auc_SNA[i] = roc_auc_score(y_test, y_prob_SNA[:, 1])
        bri_SNA[i] = brier_score_loss(y_test, y_prob_SNA[:, 1])
        mcc_SNA[i] = matthews_corrcoef(y_test, y_pred_SNA1)
        # bal_SNA[i] = bal_score(y_test, y_pred_SNA1)
        ce_SNA[i] = CE_score(X_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_SNA[:, 1])
        er_SNA[i] = ER_score(X_test["CountLineCode"].values, y_num_test, y_pred_SNA1)

        # 使用ENs度量元预测缺陷数
        lrModel1.fit(x_ENs_train2, y_num_train)
        y_pred_ENs = lrModel1.predict(x_ENs_test2)

        RR = r2_score(y_num_test, y_pred_ENs)
        adj_ENs[i] = 1 - ((1 - RR) * (n - 1) / (n - x_ENs_test2.shape[1] - 1))
        co_ENs[i], _ = spearmanr(y_num_test, y_pred_ENs)
        rmse_ENs[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_ENs))
        mae_ENs[i] = mean_absolute_error(y_num_test, y_pred_ENs)

        # 使用ENs预测倾向性
        lrModel2.fit(x_ENs_train2, y_train)
        y_pred_ENs1 = lrModel2.predict(x_ENs_test2)
        y_prob_ENs = lrModel2.predict_proba(x_ENs_test2)

        pre_ENs[i] = precision_score(y_test, y_pred_ENs1)
        rr_ENs[i] = recall_score(y_test, y_pred_ENs1)
        auc_ENs[i] = roc_auc_score(y_test, y_prob_ENs[:, 1])
        bri_ENs[i] = brier_score_loss(y_test, y_prob_ENs[:, 1])
        mcc_ENs[i] = matthews_corrcoef(y_test, y_pred_ENs1)
        # bal_SNA[i] = bal_score(y_test, y_pred_SNA1)
        ce_ENs[i] = CE_score(X_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_ENs[:, 1])
        er_ENs[i] = ER_score(X_test["CountLineCode"].values, y_num_test, y_pred_ENs1)
        #
        # # 使用GNs度量元预测缺陷数
        lrModel1.fit(x_GNs_train2, y_num_train)
        y_pred_GNs = lrModel1.predict(x_GNs_test2)

        RR = r2_score(y_num_test, y_pred_GNs)
        adj_GNs[i] = 1 - ((1 - RR) * (n - 1) / (n - x_GNs_test2.shape[1] - 1))
        co_GNs[i], _ = spearmanr(y_num_test, y_pred_GNs)
        rmse_GNs[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_GNs))
        mae_GNs[i] = mean_absolute_error(y_num_test, y_pred_GNs)

        # 使用GNs预测倾向性
        lrModel2.fit(x_GNs_train2, y_train)
        y_pred_GNs1 = lrModel2.predict(x_GNs_test2)
        y_prob_GNs = lrModel2.predict_proba(x_GNs_test2)

        pre_GNs[i] = precision_score(y_test, y_pred_GNs1)
        rr_GNs[i] = recall_score(y_test, y_pred_GNs1)
        auc_GNs[i] = roc_auc_score(y_test, y_prob_GNs[:, 1])
        bri_GNs[i] = brier_score_loss(y_test, y_prob_GNs[:, 1])
        mcc_GNs[i] = matthews_corrcoef(y_test, y_pred_GNs1)
        # bal_SNA[i] = bal_score(y_test, y_pred_SNA1)
        ce_GNs[i] = CE_score(X_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_GNs[:, 1])
        er_GNs[i] = ER_score(X_test["CountLineCode"].values, y_num_test, y_pred_GNs1)

        # # 使用COM预测倾向性
        lrModel1.fit(x_COM_train2, y_num_train)
        y_pred_COM = lrModel1.predict(x_COM_test2)

        RR = r2_score(y_num_test, y_pred_COM)
        adj_COM[i] = 1 - ((1 - RR) * (n - 1) / (n - x_COM_test2.shape[1] - 1))
        co_COM[i], _ = spearmanr(y_num_test, y_pred_COM)
        rmse_COM[i] = np.sqrt(mean_squared_error(y_num_test, y_pred_COM))
        mae_COM[i] = mean_absolute_error(y_num_test, y_pred_COM)

        # 使用GNs预测倾向性
        lrModel2.fit(x_COM_train2, y_train)
        y_pred_COM1 = lrModel2.predict(x_COM_test2)
        y_prob_COM = lrModel2.predict_proba(x_COM_test2)

        pre_COM[i] = precision_score(y_test, y_pred_COM1)
        rr_COM[i] = recall_score(y_test, y_pred_COM1)
        auc_COM[i] = roc_auc_score(y_test, y_prob_COM[:, 1])
        bri_COM[i] = brier_score_loss(y_test, y_prob_COM[:, 1])
        mcc_COM[i] = matthews_corrcoef(y_test, y_pred_COM1)
        # bal_SNA[i] = bal_score(y_test, y_pred_SNA1)
        ce_COM[i] = CE_score(X_test["CountLineCode"].values / sumcode, y_num_test / sumdef, y_prob_COM[:, 1])
        er_COM[i] = ER_score(X_test["CountLineCode"].values, y_num_test, y_pred_COM1)

        # prob = pd.DataFrame(np.column_stack((y_prob_MET[:, 1], y_prob_SNA[:, 1], y_prob_ENs[:, 1], y_prob_GNs[:, 1],
        #                                      y_prob_SM[:, 1], y_prob_COM[:, 1], y_test)))
        # prob.columns = ['MET', 'SNA', 'ENs', 'GNs', 'SM', 'COM', 'True']
        # prob.to_csv('E:\\mygrap\\new\\Cross-version\\number\\Camel\\' + file + '_' + str(i) + '.csv',
        #             index=False)
        #
        # number = pd.DataFrame(
        #     np.column_stack((y_pred_MET, y_pred_SNA, y_pred_ENs, y_pred_GNs, y_pred_SM, y_pred_COM, y_num_test)))
        # number.columns = ['MET', 'SNA', 'ENs', 'GNs', 'SM', 'COM', 'True']
        # number.to_csv('E:\\mygrap\\new\\Cross-version\\proness\\Camel\\' + file + '_' + str(i) + '.csv',
        #               index=False)

    data_Adj_MET = pd.DataFrame(adj_MET)
    data_Adj_MET['metrics'] = metr_MET
    data_Adj_MET['measure'] = Adj
    data_Adj_MET.columns = ['values', 'metrics', 'measure']

    data_MAE_MET = pd.DataFrame(mae_MET)
    data_MAE_MET['metrics'] = metr_MET
    data_MAE_MET['measure'] = MAE
    data_MAE_MET.columns = ['values', 'metrics', 'measure']

    data_Spearman_MET = pd.DataFrame(co_MET)
    data_Spearman_MET['metrics'] = metr_MET
    data_Spearman_MET['measure'] = Spearman
    data_Spearman_MET.columns = ['values', 'metrics', 'measure']

    data_RMSE_MET = pd.DataFrame(rmse_MET)
    data_RMSE_MET['metrics'] = metr_MET
    data_RMSE_MET['measure'] = RMSE
    data_RMSE_MET.columns = ['values', 'metrics', 'measure']

    data_Precision_MET = pd.DataFrame(pre_MET)
    data_Precision_MET['metrics'] = metr_MET
    data_Precision_MET['measure'] = Pre
    data_Precision_MET.columns = ['values', 'metrics', 'measure']

    data_Recall_MET = pd.DataFrame(rr_MET)
    data_Recall_MET['metrics'] = metr_MET
    data_Recall_MET['measure'] = Recall
    data_Recall_MET.columns = ['values', 'metrics', 'measure']

    data_AUC_MET = pd.DataFrame(auc_MET)
    data_AUC_MET['metrics'] = metr_MET
    data_AUC_MET['measure'] = AUC
    data_AUC_MET.columns = ['values', 'metrics', 'measure']

    data_Bri_MET = pd.DataFrame(bri_MET)
    data_Bri_MET['metrics'] = metr_MET
    data_Bri_MET['measure'] = Bri
    data_Bri_MET.columns = ['values', 'metrics', 'measure']

    data_MCC_MET = pd.DataFrame(mcc_MET)
    data_MCC_MET['metrics'] = metr_MET
    data_MCC_MET['measure'] = MCC
    data_MCC_MET.columns = ['values', 'metrics', 'measure']

    # data_Bal_MET = pd.DataFrame(bal_MET)
    # data_Bal_MET['metrics'] = metr_MET
    # data_Bal_MET['measure'] = Bal
    # data_Bal_MET.columns = ['values', 'metrics', 'measure']

    data_CE_MET = pd.DataFrame(ce_MET)
    data_CE_MET['metrics'] = metr_MET
    data_CE_MET['measure'] = CE
    data_CE_MET.columns = ['values', 'metrics', 'measure']

    data_ER_MET = pd.DataFrame(er_MET)
    data_ER_MET['metrics'] = metr_MET
    data_ER_MET['measure'] = ER
    data_ER_MET.columns = ['values', 'metrics', 'measure']

    data_Adj_SNA = pd.DataFrame(adj_SNA)
    data_Adj_SNA['metrics'] = metr_SNA
    data_Adj_SNA['measure'] = Adj
    data_Adj_SNA.columns = ['values', 'metrics', 'measure']

    data_MAE_SNA = pd.DataFrame(mae_SNA)
    data_MAE_SNA['metrics'] = metr_SNA
    data_MAE_SNA['measure'] = MAE
    data_MAE_SNA.columns = ['values', 'metrics', 'measure']

    data_Spearman_SNA = pd.DataFrame(co_SNA)
    data_Spearman_SNA['metrics'] = metr_SNA
    data_Spearman_SNA['measure'] = Spearman
    data_Spearman_SNA.columns = ['values', 'metrics', 'measure']

    data_RMSE_SNA = pd.DataFrame(rmse_SNA)
    data_RMSE_SNA['metrics'] = metr_SNA
    data_RMSE_SNA['measure'] = RMSE
    data_RMSE_SNA.columns = ['values', 'metrics', 'measure']

    data_Precision_SNA = pd.DataFrame(pre_SNA)
    data_Precision_SNA['metrics'] = metr_SNA
    data_Precision_SNA['measure'] = Pre
    data_Precision_SNA.columns = ['values', 'metrics', 'measure']

    data_Recall_SNA = pd.DataFrame(rr_SNA)
    data_Recall_SNA['metrics'] = metr_SNA
    data_Recall_SNA['measure'] = Recall
    data_Recall_SNA.columns = ['values', 'metrics', 'measure']

    data_AUC_SNA = pd.DataFrame(auc_SNA)
    data_AUC_SNA['metrics'] = metr_SNA
    data_AUC_SNA['measure'] = AUC
    data_AUC_SNA.columns = ['values', 'metrics', 'measure']

    data_MCC_SNA = pd.DataFrame(mcc_SNA)
    data_MCC_SNA['metrics'] = metr_SNA
    data_MCC_SNA['measure'] = MCC
    data_MCC_SNA.columns = ['values', 'metrics', 'measure']

    # data_Bal_SNA = pd.DataFrame(bal_SNA)
    # data_Bal_SNA['metrics'] = metr_SNA
    # data_Bal_SNA['measure'] = Bal
    # data_Bal_SNA.columns = ['values', 'metrics', 'measure']

    data_CE_SNA = pd.DataFrame(ce_SNA)
    data_CE_SNA['metrics'] = metr_SNA
    data_CE_SNA['measure'] = CE
    data_CE_SNA.columns = ['values', 'metrics', 'measure']

    data_ER_SNA = pd.DataFrame(er_SNA)
    data_ER_SNA['metrics'] = metr_SNA
    data_ER_SNA['measure'] = ER
    data_ER_SNA.columns = ['values', 'metrics', 'measure']

    data_Bri_SNA = pd.DataFrame(bri_SNA)
    data_Bri_SNA['metrics'] = metr_SNA
    data_Bri_SNA['measure'] = Bri
    data_Bri_SNA.columns = ['values', 'metrics', 'measure']
    ##################SM

    data_Adj_SM = pd.DataFrame(adj_SM)
    data_Adj_SM['metrics'] = metr_SM
    data_Adj_SM['measure'] = Adj
    data_Adj_SM.columns = ['values', 'metrics', 'measure']

    data_RMSE_SM = pd.DataFrame(rmse_SM)
    data_RMSE_SM['metrics'] = metr_SM
    data_RMSE_SM['measure'] = RMSE
    data_RMSE_SM.columns = ['values', 'metrics', 'measure']

    data_MAE_SM = pd.DataFrame(mae_SM)
    data_MAE_SM['metrics'] = metr_SM
    data_MAE_SM['measure'] = MAE
    data_MAE_SM.columns = ['values', 'metrics', 'measure']

    data_Spearman_SM = pd.DataFrame(co_SM)
    data_Spearman_SM['metrics'] = metr_SM
    data_Spearman_SM['measure'] = Spearman
    data_Spearman_SM.columns = ['values', 'metrics', 'measure']

    data_Precision_SM = pd.DataFrame(pre_SM)
    data_Precision_SM['metrics'] = metr_SM
    data_Precision_SM['measure'] = Pre
    data_Precision_SM.columns = ['values', 'metrics', 'measure']

    data_Recall_SM = pd.DataFrame(rr_SM)
    data_Recall_SM['metrics'] = metr_SM
    data_Recall_SM['measure'] = Recall
    data_Recall_SM.columns = ['values', 'metrics', 'measure']

    data_AUC_SM = pd.DataFrame(auc_SM)
    data_AUC_SM['metrics'] = metr_SM
    data_AUC_SM['measure'] = AUC
    data_AUC_SM.columns = ['values', 'metrics', 'measure']

    data_Bri_SM = pd.DataFrame(bri_SM)
    data_Bri_SM['metrics'] = metr_SM
    data_Bri_SM['measure'] = Bri
    data_Bri_SM.columns = ['values', 'metrics', 'measure']

    data_MCC_SM = pd.DataFrame(mcc_SM)
    data_MCC_SM['metrics'] = metr_SM
    data_MCC_SM['measure'] = MCC
    data_MCC_SM.columns = ['values', 'metrics', 'measure']

    # data_Bal_SM = pd.DataFrame(bal_SM)
    # data_Bal_SM['metrics'] = metr_SM
    # data_Bal_SM['measure'] = Bal
    # data_Bal_SM.columns = ['values', 'metrics', 'measure']

    data_CE_SM = pd.DataFrame(ce_SM)
    data_CE_SM['metrics'] = metr_SM
    data_CE_SM['measure'] = CE
    data_CE_SM.columns = ['values', 'metrics', 'measure']

    data_ER_SM = pd.DataFrame(er_SM)
    data_ER_SM['metrics'] = metr_SM
    data_ER_SM['measure'] = ER
    data_ER_SM.columns = ['values', 'metrics', 'measure']

    ##################################################ENs

    data_Adj_ENs = pd.DataFrame(adj_ENs)
    data_Adj_ENs['metrics'] = metr_ENs
    data_Adj_ENs['measure'] = Adj
    data_Adj_ENs.columns = ['values', 'metrics', 'measure']

    data_MAE_ENs = pd.DataFrame(mae_ENs)
    data_MAE_ENs['metrics'] = metr_ENs
    data_MAE_ENs['measure'] = MAE
    data_MAE_ENs.columns = ['values', 'metrics', 'measure']

    data_Spearman_ENs = pd.DataFrame(co_ENs)
    data_Spearman_ENs['metrics'] = metr_ENs
    data_Spearman_ENs['measure'] = Spearman
    data_Spearman_ENs.columns = ['values', 'metrics', 'measure']

    data_RMSE_ENs = pd.DataFrame(rmse_ENs)
    data_RMSE_ENs['metrics'] = metr_ENs
    data_RMSE_ENs['measure'] = RMSE
    data_RMSE_ENs.columns = ['values', 'metrics', 'measure']

    data_Precision_ENs = pd.DataFrame(pre_ENs)
    data_Precision_ENs['metrics'] = metr_ENs
    data_Precision_ENs['measure'] = Pre
    data_Precision_ENs.columns = ['values', 'metrics', 'measure']

    data_Recall_ENs = pd.DataFrame(rr_ENs)
    data_Recall_ENs['metrics'] = metr_ENs
    data_Recall_ENs['measure'] = Recall
    data_Recall_ENs.columns = ['values', 'metrics', 'measure']

    data_AUC_ENs = pd.DataFrame(auc_ENs)
    data_AUC_ENs['metrics'] = metr_ENs
    data_AUC_ENs['measure'] = AUC
    data_AUC_ENs.columns = ['values', 'metrics', 'measure']

    data_MCC_ENs = pd.DataFrame(mcc_ENs)
    data_MCC_ENs['metrics'] = metr_ENs
    data_MCC_ENs['measure'] = MCC
    data_MCC_ENs.columns = ['values', 'metrics', 'measure']

    data_Bri_ENs = pd.DataFrame(bri_ENs)
    data_Bri_ENs['metrics'] = metr_ENs
    data_Bri_ENs['measure'] = Bri
    data_Bri_ENs.columns = ['values', 'metrics', 'measure']

    data_CE_ENs = pd.DataFrame(ce_ENs)
    data_CE_ENs['metrics'] = metr_ENs
    data_CE_ENs['measure'] = CE
    data_CE_ENs.columns = ['values', 'metrics', 'measure']

    data_ER_ENs = pd.DataFrame(er_ENs)
    data_ER_ENs['metrics'] = metr_ENs
    data_ER_ENs['measure'] = ER
    data_ER_ENs.columns = ['values', 'metrics', 'measure']

    ###########################################GNs
    data_Adj_GNs = pd.DataFrame(adj_GNs)
    data_Adj_GNs['metrics'] = metr_GNs
    data_Adj_GNs['measure'] = Adj
    data_Adj_GNs.columns = ['values', 'metrics', 'measure']

    data_MAE_GNs = pd.DataFrame(mae_GNs)
    data_MAE_GNs['metrics'] = metr_GNs
    data_MAE_GNs['measure'] = MAE
    data_MAE_GNs.columns = ['values', 'metrics', 'measure']

    data_Spearman_GNs = pd.DataFrame(co_GNs)
    data_Spearman_GNs['metrics'] = metr_GNs
    data_Spearman_GNs['measure'] = Spearman
    data_Spearman_GNs.columns = ['values', 'metrics', 'measure']

    data_RMSE_GNs = pd.DataFrame(rmse_GNs)
    data_RMSE_GNs['metrics'] = metr_GNs
    data_RMSE_GNs['measure'] = RMSE
    data_RMSE_GNs.columns = ['values', 'metrics', 'measure']

    data_Precision_GNs = pd.DataFrame(pre_GNs)
    data_Precision_GNs['metrics'] = metr_GNs
    data_Precision_GNs['measure'] = Pre
    data_Precision_GNs.columns = ['values', 'metrics', 'measure']

    data_Recall_GNs = pd.DataFrame(rr_GNs)
    data_Recall_GNs['metrics'] = metr_GNs
    data_Recall_GNs['measure'] = Recall
    data_Recall_GNs.columns = ['values', 'metrics', 'measure']

    data_AUC_GNs = pd.DataFrame(auc_GNs)
    data_AUC_GNs['metrics'] = metr_GNs
    data_AUC_GNs['measure'] = AUC
    data_AUC_GNs.columns = ['values', 'metrics', 'measure']

    data_MCC_GNs = pd.DataFrame(mcc_GNs)
    data_MCC_GNs['metrics'] = metr_GNs
    data_MCC_GNs['measure'] = MCC
    data_MCC_GNs.columns = ['values', 'metrics', 'measure']

    data_Bri_GNs = pd.DataFrame(bri_GNs)
    data_Bri_GNs['metrics'] = metr_GNs
    data_Bri_GNs['measure'] = Bri
    data_Bri_GNs.columns = ['values', 'metrics', 'measure']

    data_CE_GNs = pd.DataFrame(ce_GNs)
    data_CE_GNs['metrics'] = metr_GNs
    data_CE_GNs['measure'] = CE
    data_CE_GNs.columns = ['values', 'metrics', 'measure']

    data_ER_GNs = pd.DataFrame(er_GNs)
    data_ER_GNs['metrics'] = metr_GNs
    data_ER_GNs['measure'] = ER
    data_ER_GNs.columns = ['values', 'metrics', 'measure']

    ###########################################COM
    data_Adj_COM = pd.DataFrame(adj_COM)
    data_Adj_COM['metrics'] = metr_COM
    data_Adj_COM['measure'] = Adj
    data_Adj_COM.columns = ['values', 'metrics', 'measure']

    data_MAE_COM = pd.DataFrame(mae_COM)
    data_MAE_COM['metrics'] = metr_COM
    data_MAE_COM['measure'] = MAE
    data_MAE_COM.columns = ['values', 'metrics', 'measure']

    data_Spearman_COM = pd.DataFrame(co_COM)
    data_Spearman_COM['metrics'] = metr_COM
    data_Spearman_COM['measure'] = Spearman
    data_Spearman_COM.columns = ['values', 'metrics', 'measure']

    data_RMSE_COM = pd.DataFrame(rmse_COM)
    data_RMSE_COM['metrics'] = metr_COM
    data_RMSE_COM['measure'] = RMSE
    data_RMSE_COM.columns = ['values', 'metrics', 'measure']

    data_Precision_COM = pd.DataFrame(pre_COM)
    data_Precision_COM['metrics'] = metr_COM
    data_Precision_COM['measure'] = Pre
    data_Precision_COM.columns = ['values', 'metrics', 'measure']

    data_Recall_COM = pd.DataFrame(rr_COM)
    data_Recall_COM['metrics'] = metr_COM
    data_Recall_COM['measure'] = Recall
    data_Recall_COM.columns = ['values', 'metrics', 'measure']

    data_AUC_COM = pd.DataFrame(auc_COM)
    data_AUC_COM['metrics'] = metr_COM
    data_AUC_COM['measure'] = AUC
    data_AUC_COM.columns = ['values', 'metrics', 'measure']

    data_Bri_COM = pd.DataFrame(bri_COM)
    data_Bri_COM['metrics'] = metr_COM
    data_Bri_COM['measure'] = Bri
    data_Bri_COM.columns = ['values', 'metrics', 'measure']

    data_MCC_COM = pd.DataFrame(mcc_COM)
    data_MCC_COM['metrics'] = metr_COM
    data_MCC_COM['measure'] = MCC
    data_MCC_COM.columns = ['values', 'metrics', 'measure']

    data_CE_COM = pd.DataFrame(ce_COM)
    data_CE_COM['metrics'] = metr_COM
    data_CE_COM['measure'] = CE
    data_CE_COM.columns = ['values', 'metrics', 'measure']

    data_ER_COM = pd.DataFrame(er_COM)
    data_ER_COM['metrics'] = metr_COM
    data_ER_COM['measure'] = ER
    data_ER_COM.columns = ['values', 'metrics', 'measure']

    frames2 = [data_Adj_MET, data_Adj_SNA, data_Adj_SM, data_Adj_ENs, data_Adj_GNs, data_Adj_COM]
    result2 = pd.concat(frames2)
    result2.columns = ['values', 'metrics', 'measure']
    result2.to_csv('C:/gln/mygrap/new/crossversion/new/Adj/' + 'Adj_' + file, index=False)

    frames3 = [data_MAE_MET, data_MAE_SNA, data_MAE_SM, data_MAE_ENs, data_MAE_GNs, data_MAE_COM]
    result3 = pd.concat(frames3)
    result3.columns = ['values', 'metrics', 'measure']
    result3.to_csv('C:/gln/mygrap/new/crossversion/new/MAE/' + 'MAE_' + file, index=False)

    frames4 = [data_Spearman_MET, data_Spearman_SNA, data_Spearman_SM, data_Spearman_ENs, data_Spearman_GNs,
               data_Spearman_COM]
    result4 = pd.concat(frames4)
    result4.columns = ['values', 'metrics', 'measure']
    result4.to_csv('C:/gln/mygrap/new/crossversion/new/Spearman/' + 'Spearman_' + file, index=False)
    #
    frames5 = [data_RMSE_MET, data_RMSE_SNA, data_RMSE_SM, data_RMSE_ENs, data_RMSE_GNs, data_RMSE_COM]
    result5 = pd.concat(frames5)
    result5.columns = ['values', 'metrics', 'measure']
    result5.to_csv('C:/gln/mygrap/new/crossversion/new/RMSE/' + 'RMSE_' + file, index=False)

    frames6 = [data_Recall_MET, data_Recall_SNA, data_Recall_SM, data_Recall_ENs, data_Recall_GNs, data_Recall_COM]
    result6 = pd.concat(frames6)
    result6.columns = ['values', 'metrics', 'measure']
    result6.to_csv('C:/gln/mygrap/new/crossversion/new/Recall/' + 'Recall_' + file, index=False)

    frames7 = [data_AUC_MET, data_AUC_SNA, data_AUC_SM, data_AUC_ENs, data_AUC_GNs, data_AUC_COM]
    result7 = pd.concat(frames7)
    result7.columns = ['values', 'metrics', 'measure']
    result7.to_csv('C:/gln/mygrap/new/crossversion/new/AUC/' + 'AUC_' + file, index=False)

    frames8 = [data_MCC_MET, data_MCC_SNA, data_MCC_SM, data_MCC_ENs, data_MCC_GNs, data_MCC_COM]
    result8 = pd.concat(frames8)
    result8.columns = ['values', 'metrics', 'measure']
    result8.to_csv('C:/gln/mygrap/new/crossversion/new/MCC/' + 'MCC_' + file, index=False)

    frames9 = [data_Bri_MET, data_Bri_SNA, data_Bri_SM, data_Bri_ENs, data_Bri_GNs, data_Bri_COM]
    result9 = pd.concat(frames9)
    result9.columns = ['values', 'metrics', 'measure']
    result9.to_csv('C:/gln/mygrap/new/crossversion/new/Bri/' + 'Bri_' + file, index=False)

    frames10 = [data_CE_MET, data_CE_SNA, data_CE_SM, data_CE_ENs, data_CE_GNs, data_CE_COM]
    result10 = pd.concat(frames10)
    result10.columns = ['values', 'metrics', 'measure']
    result10.to_csv('C:/gln/mygrap/new/crossversion/new/CE/' + 'CE_' + file, index=False)

    frames11 = [data_ER_MET, data_ER_SNA, data_ER_SM, data_ER_ENs, data_ER_GNs, data_ER_COM]
    result11 = pd.concat(frames11)
    result11.columns = ['values', 'metrics', 'measure']
    result11.to_csv('C:/gln/mygrap/new/crossversion/new/ER/' + 'ER_' + file, index=False)

    frames12 = [data_Precision_MET, data_Precision_SNA, data_Precision_SM, data_Precision_ENs, data_Precision_GNs,
                data_Precision_COM]
    result12 = pd.concat(frames12)
    result12.columns = ['values', 'metrics', 'measure']
    result12.to_csv('C:/gln/mygrap/new/crossversion/new/Precision/' + 'Precision_' + file,
                    index=False)







