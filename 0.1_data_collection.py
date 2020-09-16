import numpy as np
import pandas as pd
import scipy.io as sio
from test_win_sql import *

def bs_data_collection():
    '''
    .MAT:
    allfirmcodes: all codes in BS_all, 3591*1 ndarray
    alldates:all f_ann_date dates in BS_all, 5159*1 ndarray
    ta_num: unstacked total_assets,5159*3591 array
    tl_num: unstacked total_liab,5159*3591 array
    tcl_num: unstacked total_cur_liab,5159*3591 array
    csi_num: unstacked money_cap+trad_asset,5159*3591 array
    :return:ta_num, tl_num, tcl_num, csi_num: 5159*3591 df
    '''

    ############   NOT READABLE  ################
    # bs = sio.loadmat('../data/allAshare/BS_data.mat')
    # BS_all = bs.get('BS_all')
    #############################################

    BS_all = pd.read_csv('../data/allAshare/BS_all.csv')
    BS_all.loc[pd.isnull(BS_all['f_ann_date']),'f_ann_date'] = BS_all.loc[pd.isnull(BS_all['f_ann_date']),'end_date']
    BS_all['f_ann_date'] = BS_all['f_ann_date'].astype('str').str.slice(start=0,stop=8).astype('datetime64[ns]')

    allfirmcodes = pd.unique(BS_all['ts_code'])
    alldates = pd.unique(BS_all['f_ann_date'])
    alldates = np.sort(alldates[pd.isnull(alldates)==False],axis=0)

    BS_all['csi'] = np.nansum([BS_all['money_cap'],BS_all['trad_asset']],axis=0)
    BS_all = BS_all.drop_duplicates(subset=['f_ann_date','ts_code'],keep='first').set_index(['f_ann_date', 'ts_code'])

    ta_num = BS_all['total_assets'].unstack().sort_index()
    tl_num = BS_all['total_liab'].unstack().sort_index()
    tcl_num = BS_all['total_cur_liab'].unstack().sort_index()
    csi_num = BS_all['csi'].unstack().replace(0.,np.nan).sort_index()
    # print(ta_num)
    # print(tl_num)
    # print(tcl_num)
    # print(csi_num)

    sio.savemat('../data/bs_data_all.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes),1),'alldates': alldates.reshape(len(alldates),1),
                                            'ta_num': np.array(ta_num),'tl_num': np.array(tl_num),'tcl_num': np.array(tcl_num),'csi_num': np.array(csi_num)})

    ta_num.to_csv('../data/ta_num.csv')
    tl_num.to_csv('../data/tl_num.csv')
    tcl_num.to_csv('../data/tcl_num.csv')
    csi_num.to_csv('../data/csi_num.csv')

    return ta_num, tl_num, tcl_num, csi_num



def is_data_collection():
    '''
    allfirmcodes: all codes in IS_all, 3594*1 ndarray
    alldates: all daily dates in IS_all, 5158*1 ndarray
    ni_num: unstacked nifromtable,5158*3594 array
    :return: ni_num: unstacked nifromtable,5158*3594 df
    '''

    ############   NOT READABLE  ################
    # bs = sio.loadmat('../data/allAshare/IS_data.mat')
    # IS_all = bs.get('IS_all')
    #############################################

    IS_all = pd.read_csv('../data/allAshare/IS_all.csv') #str converts to datetime
    IS_all.loc[pd.isnull(IS_all['f_ann_date']), 'f_ann_date'] = IS_all.loc[pd.isnull(IS_all['f_ann_date']), 'end_date']
    IS_all['f_ann_date'] = IS_all['f_ann_date'].astype(str).str.slice(start=0,stop=8).astype('datetime64[ns]')


    allfirmcodes = pd.unique(IS_all['ts_code'])
    alldates = pd.unique(IS_all['f_ann_date'])
    alldates = np.sort(alldates[pd.isnull(alldates)==False],axis=0)
    # print(allfirmcodes.shape)
    # print(alldates.shape)

    IS_all['nifromtable'] = np.nansum([IS_all['total_revenue'], -IS_all['total_cogs'], -IS_all['int_exp'], -IS_all['oper_exp'], -IS_all['income_tax']],axis=0)
    IS_all = IS_all.drop_duplicates(subset=['f_ann_date','ts_code'],keep='first').set_index(['f_ann_date', 'ts_code'])

    ni_num = IS_all['nifromtable'].unstack().replace(0.,np.nan).sort_index()
    # print(ni_num)

    sio.savemat('../data/is_data_all.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes),1),'alldates': alldates.reshape(len(alldates),1),'ni_num': np.array(ni_num)})
    ni_num.to_csv('../data/ni_num.csv')

    return ni_num



def mc_data_collection():
    '''
    .MAT:
    allfirmcodes: all codes in yedatamkvalueall, 3593*1 ndarray
    alldates: all daily dates in yedatamkvalueall, 5909*1 ndarray
    mc_num: unstacked total_mv,5909*3593 array
    nomclist: codes for no data, empty list
    :return: mc_num: unstacked total_mv,5909*3593 df
    '''

    ####################      NOT READABLE    #######################
    # mc = sio.loadmat('../data/allAshare/mc_data.mat')
    # yedatamkvalueall = mc.get('yedatamkvalueall')
    # mc_table = yedatamkvalueall
    #################################################################

    mc_table = pd.read_csv('../data/allAshare/yedatamkvalueall.csv',parse_dates=['trade_date'])

    allfirmcodes = pd.unique(mc_table['ts_code'])
    alldates = np.sort(pd.unique(mc_table['trade_date']),axis=0)

    mc_num = mc_table.set_index(['trade_date','ts_code'])['total_mv'].unstack().sort_index()
    nomclist = mc_num.columns[(np.sum(pd.isnull(mc_num),axis=0)==len(mc_num)).tolist()].tolist()

    sio.savemat('../data/mc_data_all.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes),1),'alldates': alldates.reshape(len(alldates),1),'mc_num': np.array(mc_num),'nomclist': nomclist})
    mc_num.to_csv('../data/mc_num.csv')

    return mc_num

if __name__ == '__main__':

    # mc_data_collection()
    # is_data_collection()
    bs_data_collection()

