import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
from sqlalchemy import create_engine
import pymysql

engine4 = create_engine(
    'mysql+pymysql://yyzhao:er2a23mkjwrMq1Z6@192.168.1.140:3306/filesync')

engine = create_engine('mysql+pymysql://shuhao:shuhao123@cdb-n3nb31fy.bj.tencentcdb.com:10232/active_db?charset=utf8')

def balancesheet(sector_list,event_date):
    sql = '''
                SELECT S_INFO_WINDCODE,REPORT_PERIOD,TOT_CUR_ASSETS,TOT_CUR_LIAB,
                TOT_ASSETS,TOT_LIAB,TOT_SHRHLDR_EQY_INCL_MIN_INT
                FROM asharebalancesheet
                WHERE REPORT_PERIOD =%s AND S_INFO_WINDCODE IN (%s) AND STATEMENT_TYPE = 408001000 
                GROUP BY S_INFO_WINDCODE,REPORT_PERIOD,TOT_CUR_ASSETS,TOT_CUR_LIAB,
                TOT_ASSETS,TOT_LIAB,TOT_SHRHLDR_EQY_INCL_MIN_INT  
                '''% (event_date,sector_list)
    df = pd.read_sql_query(sql, engine4, index_col=['S_INFO_WINDCODE', 'REPORT_PERIOD']).sort_index()
    return df

def incomestatement(sector_list,event_date):
    sql = '''
            SELECT S_INFO_WINDCODE,REPORT_PERIOD,S_FA_EBIT,S_FA_RETAINEDEARNINGS
            FROM asharefinancialindicator
            WHERE REPORT_PERIOD = %s AND S_INFO_WINDCODE IN (%s)
            GROUP BY S_INFO_WINDCODE,REPORT_PERIOD,S_FA_EBIT,S_FA_RETAINEDEARNINGS
            '''%(event_date,sector_list)
    df = pd.read_sql_query(sql, engine4, index_col=['S_INFO_WINDCODE', 'REPORT_PERIOD']).sort_index()
    return df

def zscore_order(data):
    columns_positive = ['TOT_CUR_ASSETS', 'TOT_ASSETS', 'S_FA_RETAINEDEARNINGS', 'S_FA_EBIT',
                        'TOT_SHRHLDR_EQY_INCL_MIN_INT']
    columns_reverse = ['TOT_CUR_LIAB', 'TOT_LIAB']
    rank = pd.DataFrame(index=data.index)

    # 最大值最高分
    for column in columns_positive:
        q = pd.DataFrame()
        q['rank'] = data[column].rank(method='min', pct=True)
        q = q.dropna(axis=0)
        q['score'] = 5
        if q['rank'].quantile(0.8) < q['rank'].quantile(1.0):  # 尾部重则拉开差距
            q.loc[q['rank'] <= q['rank'].quantile(0.8), 'score'] = 4
        if q['rank'].quantile(0.6) < q['rank'].quantile(0.8):
            q.loc[q['rank'] <= q['rank'].quantile(0.6), 'score'] = 3
        if q['rank'].quantile(0.4) < q['rank'].quantile(0.6):
            q.loc[q['rank'] <= q['rank'].quantile(0.4), 'score'] = 2
        if q['rank'].quantile(0.2) < q['rank'].quantile(0.4):
            q.loc[q['rank'] <= q['rank'].quantile(0.2), 'score'] = 1
        if all(q['score'] == 5):
            q.loc[q['rank'] < max(q['rank']), 'score'] = 1
        rank = pd.concat([rank, q['score']], axis=1, join_axes=[rank.index])
        rank = rank.rename(columns={'score': column})

    # 最大值最低分
    for column in columns_reverse:
        q = pd.DataFrame()
        q['rank'] = data[column].rank(method='min', ascending=False, pct=True)
        q = q.dropna(axis=0)
        q['score'] = 5
        if q['rank'].quantile(0.8) < q['rank'].quantile(1.0):  # 尾部重则拉开差距
            q.loc[q['rank'] <= q['rank'].quantile(0.8), 'score'] = 4
        if q['rank'].quantile(0.6) < q['rank'].quantile(0.8):
            q.loc[q['rank'] <= q['rank'].quantile(0.6), 'score'] = 3
        if q['rank'].quantile(0.4) < q['rank'].quantile(0.6):
            q.loc[q['rank'] <= q['rank'].quantile(0.4), 'score'] = 2
        if q['rank'].quantile(0.2) < q['rank'].quantile(0.4):
            q.loc[q['rank'] <= q['rank'].quantile(0.2), 'score'] = 1
        if all(q['score'] == 5):
            q.loc[q['rank'] < max(q['rank']), 'score'] = 1
        rank = pd.concat([rank, q['score']], axis=1, join_axes=[rank.index])
        rank = rank.rename(columns={'score': column})

    return rank
def update_to_cloud(rank):
    sql = '''select * from bond_break_score'''
    test = pd.read_sql_query(sql,engine)
    rank = rank.reset_index()
    rank = rank.rename(columns={'S_INFO_WINDCODE':'code'})
    rank = rank.set_index(['code','date'])
    rank = rank.astype(int)

    rank.to_sql('bond_break_score',con=engine,if_exists='append')





if __name__=='__main__':
    path ='D:/SH/auto_report/com/shuhao/latex/stock/financial_shenanigans_query/sector_list/industry_name_20200630.xlsx'
    event_date = 20200630
    stkcd_list = pd.read_excel(path)
    sector_list = stkcd_list['股票代码']
    sector_list = ','.join(["'%s'" % item for item in sector_list]).strip('"')
    # 数据库数据不全
    # income_statement = incomestatement(sector_list,event_date)
    # balance_sheet = balancesheet(sector_list,event_date)
    # data = pd.concat([income_statement,balance_sheet],axis=1)
    data = pd.read_excel('D:/SH/bond_default/altman/raw_bond_score_20200630.xlsx').iloc[:-2,:]
    data['date'] = 20200630
    data = data.set_index(['S_INFO_WINDCODE','date'])
    # 排序
    rank = zscore_order(data)
    rank = rank.fillna(0)
    rank = rank.astype(int)
    rank = rank.rename(columns={'TOT_CUR_LIAB':'factor_1','TOT_CUR_ASSETS':'factor_2','TOT_ASSETS':'factor_3','TOT_LIAB':'factor_4',
                                'TOT_SHRHLDR_EQY_INCL_MIN_INT':'factor_5','S_FA_EBIT':'factor_6','S_FA_RETAINEDEARNINGS':'factor_7'})
    update_to_cloud(rank)




