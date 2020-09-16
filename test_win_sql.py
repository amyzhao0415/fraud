# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import requests
import re
import os
import calendar
from sqlalchemy import create_engine

# engine3 = create_engine(
#     'mssql+pymssql://fli:Saif2016@172.16.210.14:1433/windfilesync')
engine4 = create_engine(
    'mysql+pymysql://yyzhao:er2a23mkjwrMq1Z6@192.168.1.140:3306/filesync')
engine = create_engine(
        'mysql+pymysql://yyzhao:er2a23mkjwrMq1Z6@192.168.1.140:3306/gogoal_v2')


###################################################update new data##################################
def market_value(sector_list,event_date,last_event_date): # 需要交易日的市值
    sql = '''SELECT S_INFO_WINDCODE as ts_code, TRADE_DT as trade_date, S_VAL_MV AS total_mv
            FROM ASHAREEODDERIVATIVEINDICATOR
            WHERE S_INFO_WINDCODE IN (%s) AND TRADE_DT>(%s) AND TRADE_DT<=(%s)'''%(sector_list,last_event_date,event_date)
    df = pd.read_sql_query(sql,engine4)
    df['trade_date'] = df['trade_date'].astype(int)
    df['total_mv'] = df['total_mv'].astype(float)

    return df
def market_value_2(sector_list,event_date,last_event_date):
    code_list = sector_list.reset_index()
    code_list['S_INFO_WINDCODE'] = code_list['S_INFO_WINDCODE'].str.slice(start=0,stop=6)
    code_list = ','.join(["'%s'" % item for item in code_list['S_INFO_WINDCODE']]).strip('"')
    sql = '''select stock_code as ts_code, trade_date, tcap as total_mv
     from qt_stk_daily
     where stock_code in (%s) and trade_date>(%s) and trade_date<=(%s)'''%(code_list,last_event_date,event_date)
    df = pd.read_sql_query(sql,engine)
    df['trade_date'] = df['trade_date'].dt.year*10000 + df['trade_date'].dt.month*100 + df['trade_date'].dt.day
    df['trade_date'] = df['trade_date'].astype('int')
    df['total_mv'] = df['total_mv'].astype(float)
    return df

def balance_sheet(sector_list,report_period,last_report_period):
    sql = '''select S_INFO_WINDCODE as ts_code, ANN_DT as f_ann_date, REPORT_PERIOD as end_date, MONETARY_CAP as money_cap, TRADABLE_FIN_ASSETS as trad_asset, 
     TOT_ASSETS as total_assets, TOT_CUR_LIAB as total_cur_liab, TOT_LIAB as total_liab 
        from AShareBalanceSheet
        where STATEMENT_TYPE in ('408001000') AND REPORT_PERIOD>(%s) AND REPORT_PERIOD<=(%s) AND S_INFO_WINDCODE in (%s)'''%(last_report_period,report_period,sector_list)
    df = pd.read_sql_query(sql,engine4,index_col=['ts_code','end_date']).sort_index()
    df = df.astype(float)
    df['f_ann_date'] = df['f_ann_date'].astype(int)
    return df

def income_statement(sector_list,report_period,last_report_period):
    sql = '''select S_INFO_WINDCODE as ts_code, ANN_DT as f_ann_date, REPORT_PERIOD as end_date, TOT_OPER_REV as total_revenue, TOT_OPER_COST as total_cogs,
    LESS_INT_EXP as int_exp, OPER_EXP as oper_exp,INC_TAX as income_tax 
    from AShareIncome
    where STATEMENT_TYPE in ('408001000') and REPORT_PERIOD > (%s) and REPORT_PERIOD<=(%s) and S_INFO_WINDCODE in (%s)'''%(last_report_period,report_period,sector_list)
    df = pd.read_sql_query(sql,engine4,index_col=['ts_code','end_date']).sort_index()
    df = df.astype(float)
    df['f_ann_date'] = df['f_ann_date'].astype('str')
    return df

def sh_index(event_date,last_event_date):
    sql = '''select TRADE_DT as trade_date, S_DQ_CLOSE as AdjClose
    from AIndexEODPrices
    where S_INFO_WINDCODE = '000001.SH' and TRADE_DT>(%s) and TRADE_DT<=(%s)'''%(last_event_date,event_date)
    df = pd.read_sql_query(sql,engine4)
    return df

def sh_index_2(event_date,last_event_date):
    sql = '''select trade_date,tclose as AdjClose
    from qt_idx_daily
    where index_code = '000001' and trade_date>(%s) and trade_date<=(%s)'''%(last_event_date,event_date)
    df = pd.read_sql_query(sql,engine)
    return df

######################################################与原始数据进行合并
def update_mk(sector_list, event_date):
    mv_original = pd.read_csv('D:/SH/bond_default/data/allAshare/yedatamkvalueall.csv')
    last_event_date = max(mv_original['trade_date'].unique().tolist())
    # update_mv = market_value(sector_list, event_date,last_event_date)
    update_mv = market_value_2(sector_list,event_date,last_event_date)
    sector_list = sector_list.reset_index()
    sector_list['ts_code'] = sector_list['S_INFO_WINDCODE'].str.slice(start=0,stop=6)
    sector_list = sector_list.set_index('ts_code')
    update_mv = update_mv.set_index('ts_code')
    update_mv = pd.concat([sector_list['S_INFO_WINDCODE'],update_mv],axis=1,join_axes=[update_mv.index])
    update_mv =update_mv.reset_index(drop=True).rename(columns={'S_INFO_WINDCODE':'ts_code'})
    mv_original = mv_original.set_index(['ts_code','trade_date'])
    update_mv = update_mv.set_index(['ts_code','trade_date'])
    new_mv = mv_original.append(update_mv.loc[update_mv.index.difference(set(mv_original.index))]).sort_index()

    return new_mv

def update_bs(sector_list,report_period):
    original_bs = pd.read_csv('D:/SH/bond_default/data/allAshare/BS_all.csv')
    last_report_period = max(original_bs['end_date'].tolist()) #锁定原始文件中最新一个报告期
    update_bs = balance_sheet(sector_list,report_period,last_report_period)
    update_bs = update_bs.reset_index()
    update_bs['end_date'] = update_bs['end_date'].astype(int)
    original_bs = original_bs.set_index(['ts_code','end_date'])
    update_bs = update_bs.set_index(['ts_code','end_date'])
    new_bs = original_bs.append(update_bs.loc[update_bs.index.difference(set(original_bs.index))]).sort_index()
    return new_bs

def update_is(sector_list,report_period):
    original_is = pd.read_csv('D:/SH/bond_default/data/allAshare/IS_all.csv')
    last_report_period = max(original_is['end_date'].tolist())
    update_is = income_statement(sector_list, report_period, last_report_period)
    update_is = update_is.reset_index()
    update_is['end_date'] = update_is['end_date'].astype('str') #将实际报告披露日和财务报告日期都设定为字符串
    original_is = original_is.set_index(['ts_code', 'end_date'])
    update_is = update_is.set_index(['ts_code', 'end_date'])
    new_is = original_is.append(update_is.loc[update_is.index.difference(set(original_is.index))]).sort_index()
    return new_is

def update_index(event_date):
    original_index = pd.read_csv('D:/SH/bond_default/data/SHindex.csv')
    original_index['trade_date'] = original_index['Date'].str.slice(start=0,stop=4).astype('int')*10000+original_index['Date'].str.slice(start=5,stop=7).astype('int')*100+\
                             original_index['Date'].str.slice(start=8,stop=10).astype('int')
    last_event_date=max(original_index['trade_date'].unique().tolist())
    # update_index= sh_index(event_date,last_event_date)
    update_index = sh_index_2(event_date,last_event_date)
    update_index['Date'] = update_index['trade_date'].astype('datetime64[ns]').astype(str)
    original_index = original_index[['Date','AdjClose']].set_index('Date')
    update_index = update_index[['Date','AdjClose']].set_index('Date')
    new_index = original_index.append(update_index.loc[update_index.index.difference(set(original_index.index))]).sort_index()
    return new_index

def update_default():
    original_default = pd.read_csv('D:/SH/bond_default/data/Chinadefaultevents.csv')
    original_default['date'] = original_default['DefaultDate'].str.slice(start=0,stop=4).astype('int')*10000+original_default['DefaultDate'].str.slice(start=5,stop=7).astype('int')*100+\
                             original_default['DefaultDate'].str.slice(start=8,stop=10).astype('int')
    new_default = pd.read_excel('D:/SH/bond_default/data/Default events of Chinese firms 20190101-20200106.xlsx')
    new_default['firmcode'] = new_default['Ticker'].str.slice(start=0,stop=6)
    new_default = new_default.loc[(new_default['firmcode'].str.slice(start=0,stop=1)=='0') |(new_default['firmcode'].str.slice(start=0,stop=1)=='3')|(new_default['firmcode'].str.slice(start=0,stop=1)=='6')]
    for i in new_default.index.tolist():
        if new_default['firmcode'][i].startswith('6'):
            new_default['firmcode'][i] =  new_default['firmcode'][i]+'.SH'
        else:
            new_default['firmcode'][i] = new_default['firmcode'][i] + '.SZ'
    new_default = new_default.rename(columns={'Effective_date':'DefaultDate'})
    new_default['date'] = new_default['DefaultDate'].dt.year*10000+new_default['DefaultDate'].dt.month*100+\
                             new_default['DefaultDate'].dt.day
    original_default = original_default.drop_duplicates(['firmcode','date'])
    original_default = original_default.set_index(['firmcode','date'])
    new_default = new_default.drop_duplicates(['firmcode','date'])
    new_default = new_default.set_index(['firmcode','date'])
    latest = original_default.append(new_default.loc[new_default.index.difference(set(original_default.index)),['DefaultDate','Ticker']]).sort_index()
    return latest



if __name__ =='__main__':
    # path = 'D:/SH/bond_default/family_data/private_com/industry_name.xlsx' #20191219
    # path = 'industry_name_20200330.xlsx'
    path = 'D:/SH/auto_report/com/shuhao/latex/stock/financial_shenanigans_query/sector_list/industry_name_20200630.xlsx'
    # co_list,dates = stkcd_ASHARE(path)
    co_list = pd.read_excel(path)
    co_list=co_list.rename(columns={'股票代码':'S_INFO_WINDCODE'})
    sector_list = co_list['S_INFO_WINDCODE'].drop_duplicates().tolist()  # 292
    co_list = co_list.set_index('S_INFO_WINDCODE')
    print(co_list)
    sector_list = ','.join(["'%s'" % item for item in sector_list]).strip('"')
    # event_date = 20191231
    # report_period = 20191231
    event_date = 20200831
    report_period = 20200331

    # new_mk = update_mk(sector_list, event_date) #20200720 需要卡到20200831 [用wind数据库更新市场数据，但是时间上有滞后性]
    # new_mk = update_mk(co_list, event_date) #【朝阳有续数据库可以实时更新】
    # new_bs = update_bs(sector_list,report_period)
    # new_is = update_is(sector_list,report_period)
    new_index = update_index(event_date) #for 0.2 use 20200720 需要更新到20200831
    # new_default = update_default()
'''
    #20200630直接从数据库上读不下来所以手动下载
    data= pd.read_excel('D:/SH/bond_default/data/allAshare/20200630中报数据.xlsx').iloc[:-2,:]
    data['f_ann_date'] =data['f_ann_date'].astype('str').str.slice(start= 0,stop = 4) + data['f_ann_date'].astype('str').str.slice(start= 5,stop = 7) +\
        data['f_ann_date'].astype('str').str.slice(start=8, stop=10)
    data['end_date'] = data['end_date'].astype('str').str.slice(start=0, stop=4) + data['end_date'].astype('str').str.slice(start=5, stop=7) + data['end_date'].astype('str').str.slice(start=8, stop=10)
    new_bs = new_bs.reset_index()
    new_bs = new_bs.append(data[new_bs.columns])
    new_is = new_is.reset_index()
    new_is = new_is.append(data[new_is.columns])
    new_bs =new_bs.set_index('ts_code')
    new_is = new_is.set_index('ts_code')

    # new_bs.to_csv('D:/SH/bond_default/data/allAshare/BS_all.csv')
    # new_is.to_csv('D:/SH/bond_default/data/allAshare/IS_all.csv')
    # new_mk.to_csv('D:/SH/bond_default/data/allAshare/yedatamkvalueall.csv')
    # new_index.to_csv('D:/SH/bond_default/data/SHindex.csv')
    # new_default.to_csv('D:/SH/bond_default/data/Chinadefaultevents.csv')
'''




