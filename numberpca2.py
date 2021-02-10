from __future__ import division
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    g = os.walk(r"E:\\mygrap\\new\\data2\\aa\\")
    Sta_MET = StandardScaler()
    Sta_SM = StandardScaler()
    Sta_SNA = StandardScaler()
    Sta_COM = StandardScaler()
    Sta_ENs = StandardScaler()
    Sta_GNs = StandardScaler()
    n_repeats=100

    for path, dir_list, file_list in g:
        for file_name in file_list:
            s = os.path.join(path, file_name)
            s1=os.path.join("E:\\mygrap\\new\\VIF\\COM\\", file_name)
            df = pd.read_csv(s)
            df_COM = pd.read_csv(s1)

            num_MET=[]
            num_SM=[]
            num_EN=[]
            num_GN=[]
            num_SNA=[]
            num_COM=[]

            p_COM = []
            s_SM = df_COM.columns.astype(str)
            for k in range(0, df_COM.shape[1] - 3):
                p_COM.append(s_SM[k])

            for i in range(0, n_repeats):
                boot = np.random.choice(df.shape[0], df.shape[0], replace=True)
                X_train = df.iloc[boot]
                # print(X_train.shape)
                # y_train = X_train['defect'].values
                # y_num_train = X_train["label"].values
                x_COM_train = X_train[p_COM].values

                x_train = X_train.drop(["defect", "label"], axis=1).values

                # print(x_train.shape)

                oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]  # testing data
                X_test = df.iloc[oob]
                # y_test = X_test['defect'].values
                # y_num_test = X_test["label"].values
                x_test = X_test.drop(["defect", "label"], axis=1).values
                x_COM_test = X_test[p_COM].values

                # sumcode = X_test["CountLineCode"].sum()
                #
                # sumdef = X_test["label"].sum()

                x_SM_train = x_train[:, :118]
                x_ENs_train = x_train[:, :40]
                x_GNs_train = x_train[:, 40:64]
                x_MET_train = x_train[:, 64:118]
                x_SNA_train = x_train[:, :64]

                x_SM_train2 = Sta_SM.fit_transform(x_SM_train)
                x_MET_train2 = Sta_MET.fit_transform(x_MET_train)
                x_SNA_train2 = Sta_SNA.fit_transform(x_SNA_train)
                x_COM_train2 = Sta_COM.fit_transform(x_COM_train)
                x_ENs_train2 = Sta_ENs.fit_transform(x_ENs_train)
                x_GNs_train2 = Sta_GNs.fit_transform(x_GNs_train)
                print(x_COM_train2.shape[1],x_ENs_train2.shape[1],x_GNs_train2.shape[1],x_SNA_train2.shape[1],x_MET_train2.shape[1],x_SM_train2.shape[1])
                num_COM.append(x_COM_train2.shape[1])
                num_EN.append(x_ENs_train2.shape[1])
                num_GN.append(x_GNs_train2.shape[1])
                num_SNA.append(x_SNA_train2.shape[1])
                num_MET.append(x_MET_train2.shape[1])
                num_SM.append(x_SM_train2.shape[1])

            df_num=pd.DataFrame(np.column_stack((num_MET,num_GN,num_EN,num_SNA,num_SM,num_COM)))
            df_num.columns=["Code","GN","EN","SNA","SM","COM"]
            ss=os.path.join("E:\\mygrap\\new\\data2\\number\\", file_name)
            df_num.to_csv(ss,index=False)
