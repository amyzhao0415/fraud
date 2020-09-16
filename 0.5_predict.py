import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
from sqlalchemy import create_engine
import pymysql
import random

engine = create_engine('mysql+pymysql://root:alphai_mysql_passwd@192.168.1.110:33330/sh-backend')
# engine = create_engine('dialect+driver://username:password@host:port/database')
# （数据库类型+数据库驱动选择://数据库用户名:用户密码@服务器地址:端口/数据库）
def data_slice_intime(allowmissing,n):
    if allowmissing:
        data = sio.loadmat('../data/fl_fs_final.mat')
    else:
        data = sio.loadmat('../data/fl_fs_final_nomissing.mat')
    firmspecific = data.get('firmspecific')
    allfirmcodes = pd.Series(data.get('allfirmcodes').squeeze()).tolist()
    allfirmcodes = pd.Series([allfirmcodes[i][0] for i in range(len(allfirmcodes))])  # drop []
    mthdates = pd.Series(data.get('mthdates').squeeze()).astype('datetime64[ns]')

    sliced_data = firmspecific[-n, :, :].T # 假设最后一项为需要更新的数据
    predictorNames = ['StockRet', 'IR', 'NITA', 'RS', 'SIGMA', 'M2B', 'CashTA', 'FinDD', 'nonFinDD', 'FinTLTA',
                      'nonFinTLTA']
    sliced_data = pd.DataFrame(sliced_data, columns=predictorNames, index=[allfirmcodes, [mthdates.iloc[-n]]*np.size(sliced_data, 0)])
    return sliced_data

def run_model_intime(sliced_data, last_event_date,N):
    sliced_data = sliced_data.dropna(how='all').replace(np.nan, 0)
    Pred = pd.DataFrame(index =sliced_data.index)
    for i in range(N):
        with open('../model/model_boost_%s_%s.pickle' % (str(last_event_date),str(i)), 'rb') as f:
            boostedtree = pickle.load(f)
        pred = boostedtree.predict(sliced_data)
        prob = boostedtree.predict_proba(sliced_data)[:, 1]
        pred = pd.DataFrame(pred, index=sliced_data.index)
        prob = pd.DataFrame(prob, index=sliced_data.index)
        Pred = pd.concat([Pred,pred],axis=1)
    Pred = np.sum(Pred,axis=1)/N
    return Pred, prob

def mergesubcomp(pred, prob):
    pred = pred.reset_index().set_index(['level_0'])
    prob = prob.reset_index().set_index(['level_0'])
    pred.index = pd.Series(pred.index).str.replace('_.*', '') # 原来有在代码后边加标记表示违约事件的，现在取消掉
    prob.index = pd.Series(prob.index).str.replace('_.*', '')
    pred_new = pred.groupby(level=[0]).sum() #将同一公司的值相加合并
    prob_new = prob.groupby(level=[0]).sum()
    if np.sum(np.sum(pred_new>1)) > 0 or np.sum(np.sum(prob_new>1)) > 0:
        print('Please check the position where probability or prediction is larger than 1!')
        print('position of prediction:',np.where(pred_new>1))
        print('position of probability:',np.where(prob_new>1))
        return None
    return pred_new, prob_new


# def prob_to_sql(prob, event_date, cur): #demo
#     prob = prob.reset_index()
#     prob['EVENT_DATE'] = str(event_date)[:4]+'-'+str(event_date)[4:6]+'-'+str(event_date)[6:]
#     lastest_update_date = pd.read_sql('''SELECT EVENT_DATE FROM `bond_default_probability` ORDER BY EVENT_DATE DESC limit 1''',
#         engine).squeeze()
#     if lastest_update_date < event_date:
#         cur.executemany('''insert into bond_default_probability(S_INFO_WINDCODE, EVENT_DATE, probability)
#                values(%s,%s,%s)''', list(zip(prob[:, 0], prob[:, 2], prob[:, 1])))
#         conn.commit()
#     # daily updates
#     elif lastest_update_date == event_date:
#         cur.executemany('''insert into bond_default_probability(S_INFO_WINDCODE, EVENT_DATE, probability)
#         values(%s,%s,%s) on duplicate key update probability=values(probability)''', list(zip(prob[:, 0], prob[:, 2], prob[:, 1])))
#         conn.commit()
#     else:
#         print('Wrong event_date!')


if __name__ == '__main__':
    # last_model_date = 20190228
    last_model_date = 20191231
    N=100
    update_new = pd.DataFrame()
    for i in range(1,6): # 里面的数字根据缺失的更新月份数决定
        sliced_data = data_slice_intime(False,i)
        pred, prob = run_model_intime(sliced_data, last_model_date,N)
        pred_new, prob_new = mergesubcomp(pred, prob)
        pred_new['date'] = sliced_data.index.get_level_values(1)[0]
        pred_new = pred_new.rename(columns={0:'value'})
        pred_new.loc[pred_new['value']==0,'value'] = round(random.uniform(0.01,0.1),3)
        update_new = update_new.append(pred_new)
    update_new = update_new.reset_index()
    update_new = update_new.rename(columns={'level_0':'code'})
    update_new.to_csv('D:/SH/bond_default/bond_default_probability_531_0831.csv')

