import numpy as np
import pandas as pd
from sklearn import ensemble,svm,linear_model,metrics
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import pickle

def trainClassifier_rollingoutsample_modelinput(allowmissing, qstr, modelstr):
    if allowmissing:
        firmspecific = sio.loadmat('../data/fl_fs' + qstr + '_final.mat').get('firmspecific')
    else:
        firmspecific = sio.loadmat('../data/fl_fs' + qstr + '_final_nomissing.mat').get('firmspecific')

    T = np.size(firmspecific, 0)-1  # total periods -1
    outsampleresponses = pd.DataFrame()  # placeholder for test sample
    outsampleprediction = pd.DataFrame()  # placeholder for test sample
    outsamplescores = pd.DataFrame()  # placeholder for test sample

    for t in range(round(3*T/5), T+1): #  round(3*T/5)
        print(t)
        tabledata_training, tabledata_testing = construct_data_CN_rollingoutsample(allowmissing, qstr, t)

        predictorNames = ['StockRet', 'IR', 'NITA', 'RS', 'SIGMA', 'M2B', 'CashTA', 'FinDD', 'nonFinDD', 'FinTLTA','nonFinTLTA']
        trainingPredictors = tabledata_training[predictorNames].replace(np.nan,0)
        trainingResponse = tabledata_training['Def']
        validationPredictors = tabledata_testing[predictorNames].replace(np.nan,0)
        validationResponse = tabledata_testing['Def']

        # print('trainingPredictors',trainingPredictors.shape)
        # print('trainingResponse',trainingResponse.shape)
        # print('validationPredictors',validationPredictors.shape)
        # print('validationResponse',validationResponse.shape)

        if modelstr == 'boostedtree':
            pred, validationScores, impt = boostedtree(trainingPredictors,validationPredictors,trainingResponse,validationResponse)
        elif modelstr == 'svc':
            pred, validationScores = svc(trainingPredictors,validationPredictors,trainingResponse,validationResponse)
        elif modelstr == 'logistics':
            pred, validationScores = logistics(trainingPredictors, validationPredictors, trainingResponse,validationResponse)
        else:
            pred = pd.Series()
            validationScores = pd.Series()
            print('wrong modelstr!')

        pred = pd.DataFrame(pred,index=validationResponse.index)
        validationScores = pd.DataFrame(validationScores,index=validationResponse.index)
        validationResponse = pd.DataFrame(validationResponse)

        scoring(validationResponse,pred,validationScores,t,modelstr)
        outsampleresponses = outsampleresponses.append(validationResponse)
        outsampleprediction = outsampleprediction.append(pred)
        outsamplescores = outsamplescores.append(validationScores)
        # print(outsamplescores.shape,outsampleprediction.shape,outsampleresponses.shape)


    print('overall')
    scoring(outsampleresponses,outsampleprediction,outsamplescores,'overall',modelstr)
    if 'impt' in locals().keys(): #impt for t = T
        print('impt',impt)
    print('outsampleprediction',outsampleprediction[0].unstack())
    print('outsamplescores',outsamplescores[0].unstack())
    outsampleprediction[0].unstack().to_csv('../data/outsampleprediction.csv')
    outsamplescores[0].unstack().to_csv('../data/outsamplescores.csv')

def rolling_single_model(allowmissing, event_date):
    if allowmissing:
        firmspecific = sio.loadmat('../data/fl_fs_final.mat').get('firmspecific')
    else:
        firmspecific = sio.loadmat('../data/fl_fs_final_nomissing.mat').get('firmspecific')

    T = np.size(firmspecific, 0)-1  # total periods -1
    for i in range(100):
        for t in range(T, T+1):  # T
            print(t)
            tabledata_training, tabledata_testing = construct_data_CN_rollingoutsample(allowmissing, '', t)

            predictorNames = ['StockRet', 'IR', 'NITA', 'RS', 'SIGMA', 'M2B', 'CashTA', 'FinDD', 'nonFinDD', 'FinTLTA','nonFinTLTA']
            trainingPredictors = tabledata_training[predictorNames].replace(np.nan,0)
            trainingResponse = tabledata_training['Def']
            validationPredictors = tabledata_testing[predictorNames].replace(np.nan,0)
            validationResponse = tabledata_testing['Def']

            print('trainingPredictors',trainingPredictors.shape)
            print('trainingResponse',trainingResponse.shape)
            print('validationPredictors',validationPredictors.shape)
            print('validationResponse',validationResponse.shape)

            boostedtree = ensemble.AdaBoostClassifier()
            boostedtree.fit(trainingPredictors, trainingResponse)
            with open ('../model/model_boost_%s_%s.pickle' % (str(event_date),str(i)),'wb') as f:
                pickle.dump(boostedtree, f)
            impt = pd.Series(boostedtree.feature_importances_, index=trainingPredictors.columns)
            print(impt)



def boostedtree(X_train, X_test, y_train, y_test):
    boostedtree = ensemble.AdaBoostClassifier()
    boostedtree.fit(X_train, y_train)
    # with open('../model/model_boost_%s.pickle' % str(event_date), 'wb') as f:
    #     pickle.dump(boostedtree, f)
    boostedtree_pred = boostedtree.predict(X_test)
    y_score = boostedtree.predict_proba(X_test)[:, 1]
    impt = pd.Series(boostedtree.feature_importances_, index=X_train.columns)  # feature importance
    return boostedtree_pred, y_score,impt

def svc(X_train,X_test,y_train,y_test):
    svc = svm.SVC(probability=True,kernel='rbf',C=5) # based on cross selection, rbf and 5 should be best parameters for kernel and C
    svc.fit(X_train, y_train)

    svc_pred = svc.predict(X_test)
    y_score = svc.predict_proba(X_test)[:, 1]
    return svc_pred,y_score

def logistics(X_train,X_test,y_train,y_test):
    logistics = linear_model.LogisticRegression(C=0.01)
    logistics.fit(X_train,y_train)

    logistics_pred = logistics.predict(X_test)
    y_score = logistics.predict_proba(X_test)[:,1]
    return logistics_pred,y_score

def scoring(y_test,pred,y_score,t,modelstr):
    print ('accuracy_score',metrics.accuracy_score(y_test,pred))
    print ('classification_report',metrics.classification_report(y_test,pred))
    print ('confusion_matrix',metrics.confusion_matrix(y_test,pred))
    visual(y_test,pred,t,modelstr)
    if any(y_score):
        fpr,tpr,threshold = metrics.roc_curve(y_test,y_score)
        roc_auc = metrics.auc(fpr,tpr)
        print ('roc_auc',roc_auc)

def visual(y_test,pred,t,modelstr):
    plt.clf()
    sns.heatmap(metrics.confusion_matrix(y_test,pred),annot=True,cmap='GnBu')
    plt.title(t)
    plt.savefig('../data/t=%s_modelstr=%s.png'%(str(t),modelstr))


def construct_data_CN_rollingoutsample(allowmissing, qstr, T):
        #逐月将实验数据提取，并标记违约公司
    if allowmissing:
        data = sio.loadmat('../data/fl_fs' + qstr + '_final.mat')
    else:
        data = sio.loadmat('../data/fl_fs' + qstr + '_final_nomissing.mat')
    firmspecific = data.get('firmspecific')
    firmlist = data.get('firmlist')
    allfirmcodes = pd.Series(data.get('allfirmcodes').squeeze()).tolist()
    allfirmcodes = pd.Series([allfirmcodes[i][0] for i in range(len(allfirmcodes))])  # drop [] [i][0]每个数组由两个元素组成，第一个位置的值为公司代码
    mthdates = pd.Series(data.get('mthdates').squeeze()).astype('datetime64[ns]')
    # allfirmcodes = pd.read_csv('../data/allfirmcodes_model.csv')['allfirmcodes']
    # mthdates = pd.read_csv('../data/mthdates.csv',index_col=[0]).squeeze()
    # print(allfirmcodes)
    # print(mthdates) 也可以直接从csv文件中读取，

    period = 1 # 用截止到上一个月的数据训练兵预测下一个月，时间间隔为一个月

    X = pd.DataFrame()
    y = pd.DataFrame()

    firmlist = np.hstack((firmlist, np.array(range(1,np.size(firmlist, 0)+1)).reshape(np.size(firmlist, 0),1)))
    #在水平方向上平铺
    # print('firmspecific',firmspecific.shape,firmspecific[:,:,0])
    # print('firmlist',firmlist)

    for t in range(T-1): #一直训练到投进来的月份的上一个月
        # print(t)
        firmselector = (firmlist[:, 0] <= t) & (firmlist[:, 1] >= t) # 确保所选时间点前后研究对象都有交易数据，排除掉还未上市等没有数据的公司
        # print('firmselector',' 1:',np.sum(firmselector),' 0:',len(firmselector)-np.sum(firmselector))
        # "0:"代表研究时间点没有交易数据的公司数量
        endm_status = firmlist[firmselector, 1:3]
        # print('endm_status',np.size(endm_status, 0)) # 研究对象公司的总数
        y_tmp = pd.DataFrame([0] * np.size(endm_status, 0),index=[allfirmcodes[firmselector],[mthdates.iloc[t]]*np.size(endm_status, 0)])
        # 设一个表格,变成截面数据集，时间相同、所有公司
        defselector = (endm_status[:, 0] <= t + (period - 1)) & (endm_status[:, 1] == 1)
        # print('defselector',' 1:',np.sum(defselector),' 0:',len(defselector)-np.sum(defselector))
        y_tmp[defselector] = 1
        # 转化为截面数据，将t期所有公司的11个元素赋值给X_tmp变量
        X_tmp = pd.DataFrame(firmspecific[t, :, firmselector],index=[allfirmcodes[firmselector],[mthdates.iloc[t]]*np.size(endm_status, 0)])
        # print('y_tmp',' 1:',np.sum(y_tmp),' 0:',len(y_tmp)-np.sum(y_tmp))
        # print('X_tmp',X_tmp)
        # print('X_tmp',X_tmp.shape)

        X = X.append(X_tmp)
        y = y.append(y_tmp)

    tabledata_training = pd.concat([y, X],axis=1) # 将被解释变量与解释变量进行合并
    tabledata_training.columns = ['Def', 'StockRet', 'IR', 'FinDD', 'nonFinDD', 'NITA', 'FinTLTA',
                                                   'nonFinTLTA', 'RS', 'SIGMA', 'M2B', 'CashTA']
    tabledata_training['DD'] = tabledata_training['FinDD'] + tabledata_training['nonFinDD']
    tabledata_training['TLTA'] = tabledata_training['FinTLTA'] + tabledata_training['nonFinTLTA']
    # print(tabledata_training)
    tabledata_training_copy = tabledata_training.copy()
    tabledata_training = undersampling(tabledata_training)


    X = pd.DataFrame()
    y = pd.DataFrame()
    for t in range(T-1,T): # 测试样本直接选取样本末期的（T-1,T)数据进行测试
        print(t)
        firmselector = (firmlist[:,0] <= t) & (firmlist[:, 1] >= t)
        # print('firmselector',' 1:',np.sum(firmselector),' 0:',len(firmselector)-np.sum(firmselector))
        endm_status = firmlist[firmselector, 1:3]
        # print('endm_status',np.size(endm_status, 0))
        # 转化成截面数据
        y_tmp = pd.DataFrame([0] * np.size(endm_status, 0),index=[allfirmcodes[firmselector],[mthdates.iloc[t]]*np.size(endm_status, 0)])
        defselector = (endm_status[:, 0] <= t + (period - 1)) & (endm_status[:, 1] == 1)
        # print('defselector',' 1:',np.sum(defselector),' 0:',len(defselector)-np.sum(defselector))
        y_tmp[defselector] = 1
        X_tmp = pd.DataFrame(firmspecific[t, :, firmselector], index=[allfirmcodes[firmselector],[mthdates.iloc[t]] * np.size(endm_status,0)])
        # print('y_tmp',' 1:',np.sum(y_tmp),' 0:',len(y_tmp)-np.sum(y_tmp))
        # print('X_tmp',X_tmp)
        X = X.append(X_tmp)
        y = y.append(y_tmp)


    tabledata_testing = pd.concat([y, X], axis=1)
    tabledata_testing.columns = ['Def', 'StockRet', 'IR', 'FinDD', 'nonFinDD', 'NITA', 'FinTLTA',
                                  'nonFinTLTA', 'RS', 'SIGMA', 'M2B', 'CashTA']
    tabledata_testing['DD'] = tabledata_testing['FinDD'] + tabledata_testing['nonFinDD']
    tabledata_testing['TLTA'] = tabledata_testing['FinTLTA'] + tabledata_testing['nonFinTLTA']
    # print(tabledata_testing)

    # if T == np.size(firmspecific, 0) - 1:
    #     data = pd.concat([tabledata_training_copy, tabledata_testing], axis=0)
    #     data.index.names = ['allfirmcodes', 'mthdates']
    #     data.to_csv('../data/data.csv')
    #     data.reset_index().groupby(['mthdates'])['Def'].value_counts().to_csv('../data/time_summary_statistics.csv')
    #     print('summary_statistics:\n', data['Def'].value_counts(), '\ntime_summary_statistics:\n',
    #           data.reset_index().groupby(['mthdates'])['Def'].value_counts())
    return tabledata_training, tabledata_testing

def undersampling(tabledata_training,K=1): # 查看样本的描述性统计结果
    print(tabledata_training.shape)
    training_positive = tabledata_training.loc[tabledata_training['Def'] == 1]
    print(training_positive.shape)
    training_negative = tabledata_training.loc[tabledata_training['Def'] == 0].sample(n=int(K * len(training_positive)), axis=0)
    tabledata_training = pd.concat([training_positive,training_negative])
    print(tabledata_training.shape)
    return tabledata_training


def mergesubcomp():
    pred = pd.read_csv('../data/outsampleprediction_modified.csv',index=[0])
    prob = pd.read_csv('../data/outsamplescores_modified.csv',index=[0])
    pred.index = pd.Series(pred.index).str.replace('_.*', '')
    prob.index = pd.Series(prob.index).str.replace('_.*', '')
    pred_new = pred.groupby(level=[0]).sum()
    prob_new = prob.groupby(level=[0]).sum()
    print(np.where(pred_new>1))
    print(np.where(prob_new>1))
    return pred_new, prob_new





if __name__ == '__main__':
    # trainClassifier_rollingoutsample_modelinput(False,'','boostedtree')
    # event_date = 20190228
    event_date=20191231
    rolling_single_model(False, event_date)
