import pandas as pd
import numpy as np
import scipy.io as sio
from datetime import datetime
import math
import statsmodels.api as sm


def sigma_equity():
    '''
    .MAT:
    allfirmcodes: all codes in mc_data_all, 3593*1 ndarray
    mthdates:all month-end dates(2019-02-28) deriving from alldates, 273*1 ndarray
    sigmae_array: monthly rolling std of last 1 year's daily return, 273*3593 array
    sigmad_array: from sigmae_array, 273*3593 array
    mc_array: month-end mc_num, 273*3593 array
    mu_array:monthly rolling last 1 year's annual return, 273*3593 array
    uniym: all month dates(201902), 273*1 ndarray
    :return:sigmae_array,sigmad_array,mc_array,mu_array,mthdates: df
    '''

    mc = sio.loadmat('../data/mc_data_all.mat')
    # mc_num = mc.get('mc_num')
    alldates = pd.Series(mc.get('alldates').squeeze()).astype('datetime64[ns]')
    # 带标签的数组Series;squeeze()把（[],[],[])【数组形式(n,1)】形式转化为([,,,])【元祖(n,)】并规定数据格式
    allfirmcodes = mc.get('allfirmcodes')
    mc_num = pd.read_csv('../data/mc_num.csv',index_col=[0])
    # 用第一列作为列索引，则索引变为年份
    alldates_num = alldates.dt.year * 1e+4 + alldates.dt.month * 1e+2 + alldates.dt.day
    # Series下的函数，为了方便计算，先将时间的年份\月份提取出来，将时间组成一个整数
    drn_num = pd.DataFrame(data=(np.diff(mc_num,axis=0)/mc_num.iloc[:-1, :]).values,index=mc_num.index[1:],columns=mc_num.columns)
    # 对mc_num按照行逐项做差，(后一个-前一个)/前一个值（仅针对行号来索引，以后一行为标准，对应的所有列的值减去前一行对应的所有列的值）
    #[行操作(:-1表示从序号为0一直到倒数第二个，不包括最后一个【前闭后开】)，列操作]求得“增长率”，重新计算的变量的起点为t+1年，强制规定列索引，不然只为自然序列的Index【0.1.2.3...】
    # print(drn_num)
    drndates = alldates[1:] # 取第t+1天
    rndates_num = alldates_num[1:]

    yyyymm = pd.DataFrame(round(rndates_num/100)) #数值精度设定round
    yyyymm['idx'] = yyyymm.index #建立一个新变量存在index
    uniym = pd.DataFrame(yyyymm).groupby([0])['idx'].nth(-1).index.tolist() #取每一个group的最后一个，在yyyymm变量中提取出现的非重复值,后期做循环使用
    uniymindex = pd.DataFrame(yyyymm).groupby([0])['idx'].nth(-1).values.tolist() # 提取上一步年份对应的月末idx值，或者是计数器,后期做循环使用

    mthdates = drndates[uniymindex] # 从日交易数据中提取每个月的最后一天
    nfirms = np.size(drn_num, 1) # 参数默认值为整个矩阵，为1矩阵列数，0矩阵行数，返回元素个数，一共有N家上市公司
    nmths = len(uniymindex) #交易月份合计总数，一共交易了多少个月

    sigmae_array = np.full([nmths, nfirms],np.nan) # 先给这个变量设定一个形状，并且设定初始元素值为nan
    mc_array = mc_num.iloc[uniymindex,:] # 选各月末的元素值，[选择所在的这一行，所有列的内容]
    mu_array = np.full([nmths, nfirms],np.nan) # 设置一个数组
    # print(mc_array)

    for i in range(nmths):
        mthnum = uniym[i] % 100 # 取余数
        ynum = round(uniym[i] / 100)
        lastymth = (ynum - 1) * 100 + mthnum
        print(i,lastymth)
        tmpindex = np.where(yyyymm == lastymth)[0] # [0]行索引，把一年前的行索引提取出来，np.where新生成了一个数组，第一列为保存的内容为一年以前的行号
        print(tmpindex) # 有数据说明有前一年的数据

        if len(tmpindex) > 0:
            # 非空年份的市值增长率
            startindex = tmpindex[-1] # 取最后一个行索引19960731
            endindex = uniymindex[i] # 取当年的最后一行19970731
            prior1yselector = np.transpose(range(startindex,endindex)) # last 1 year, eg. 19960731-19970731 前闭后开

            drn_i_j = drn_num.iloc[prior1yselector] # 将前一年的交易日市值增长率赋值,表示的是这一年里所有的日交易数据
            nvalid = np.sum(np.isfinite(drn_i_j)) # 里面的是判断是否为空值，返回布尔值True/False, 整句话的意思有效数据的个数
            nvalid[nvalid == len(drn_i_j)] = 252

            sigmae_array[i,nvalid>126] = np.nanstd(drn_i_j.loc[:,nvalid>126], ddof=1,axis=0) * math.sqrt(252) #std of last 1 year's daily return
            # 选取有效数据,忽略空值，自由度为1，按列计算axis=0，
            mu_array[i,nvalid>126] = np.prod((1 + drn_i_j.loc[:,nvalid>126]),axis=0) ** (nvalid[nvalid>126] / 252) - 1 #prod of last 1 year's (daily return + 1)
            #连乘函数，（有效交易天数占比）作为幂
        if np.in1d(i, range(20,6001,20)):
            print(str(i)+'mths out of '+str(nmths)+' completed.')


    sigmad_array = 0.05 + 0.25 * sigmae_array #根据公式算的sigma值

    sio.savemat('../data/sigmaE_data.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes), 1),
                                            'mthdates': mthdates.values.reshape(len(mthdates), 1),
                                            'sigmae_array': sigmae_array, 'sigmad_array': sigmad_array,
                                            'mc_array': np.array(mc_array), 'mu_array': mu_array,'uniym':np.array(uniym)})
    pd.DataFrame(data=sigmae_array,index=mc_array.index,columns=mc_array.columns).to_csv('../data/sigmae_array.csv')
    pd.DataFrame(data=sigmad_array,index=mc_array.index,columns=mc_array.columns).to_csv('../data/sigmad_array.csv')
    mc_array.to_csv('../data/mc_array.csv')
    pd.DataFrame(data=mu_array,index=mc_array.index,columns=mc_array.columns).to_csv('../data/mu_array.csv')
    pd.DataFrame(mthdates).to_csv('../data/mthdates.csv')
    return sigmae_array, sigmad_array, mc_array, mu_array,mthdates


def SIGMA_fun():
    '''
    .MAT:
    allfirmcodes: all codes in mc_data_all, 3593*1 ndarray
    mthdates:all month-end dates(2019-02-28) deriving from alldates, 273*1 ndarray
    tmr_array: monthly rolling last 1 year's annual SHIndex return, 273*3593 array
    SIGMA_array: monthly rolling std of last 1 year's daily residual by regressing daily stock return on daily SHindex return, 273*3593 array
    :return:tmr_array,SIGMA_array: 273*3593 df
    '''
    mc = sio.loadmat('../data/mc_data_all.mat')
    # mc_num = mc.get('mc_num') 也可以从.mat文件中提取数据
    alldates = pd.Series(mc.get('alldates').squeeze()).astype('datetime64[ns]').astype('str').str.replace('-','').astype('int') # error when compare datetime directly
    # 便于日期计算
    allfirmcodes = mc.get('allfirmcodes')
    mc_num = pd.read_csv('../data/mc_num.csv',index_col=[0])
    # print(alldates)

    SHindex = pd.read_csv('../data/SHindex.csv',parse_dates=[0]) # 第一列作为独立的日期列，区别在于时间值不见了
    SHindex['Date'] = SHindex['Date'].astype('str').str.replace('-','').astype('int') # 将时间类型的日期转变为可以直接计算的整型数值
    SHindex = SHindex.loc[(SHindex['Date'] >= alldates.iloc[0]) & (SHindex['Date'] <= alldates.iloc[-1]),:] #drop two sides
    #直接进行比较、计算，将指数区间与实验数据的时间对齐，定位时间及其所有列的数据
    # print(SHindex)

    indexlevel = np.full(np.size(alldates),np.nan) # 如果只给一个整数N，默认设定变量为1*N
    indexlevel[np.in1d(alldates, SHindex['Date'])] = SHindex['AdjClose'] # 赋值条件：alldates里面的日期也出现在指数变量的日期中时，匹配成功，进行赋值
    indexlevel = np.array(pd.Series(indexlevel).fillna(method='ffill')) # 处理缺失值，用前面（forward）的非空值填补空缺，查看用indexlevel[整数]
    # print(indexlevel)

    alldates_num = alldates #already transfered to int

    drn_num = pd.DataFrame(data=(np.diff(mc_num,axis=0)/mc_num.iloc[:-1, :]).values,index=mc_num.index[1:],columns=mc_num.columns)
    # print(drn_num)
    drn_index_num = pd.DataFrame(data=(np.diff(indexlevel,axis=0)/indexlevel[:-1]),index=mc_num.index[1:],columns=['AdjReturn'])
    # indexlevel 交易日的行业指数，个数等于交易日的数量
    # print(drn_index_num)
    drndates = alldates[1:]
    rndates_num = alldates_num[1:] # 交易日期：年月日

    yyyymm = pd.DataFrame(round(rndates_num/100)) # 年月
    yyyymm['idx'] = yyyymm.index # 把index行标作为一个变量存储在表格中
    uniym = pd.DataFrame(yyyymm).groupby([0])['idx'].nth(-1).index.tolist() # groupby()[].nth(-1)生成了一个新的df，其index为年份
    uniymindex = pd.DataFrame(yyyymm).groupby([0])['idx'].nth(-1).values.tolist() # 取值

    mthdates = drndates[uniymindex] # 取每个月的最后一天的日期
    nfirms = np.size(drn_num, 1) # 参数默认值为整个矩阵，为1矩阵列数，0矩阵行数，返回元素个数,经济含义为公司的数量
    nmths = len(uniymindex) # 数据对应的月份个数

    SIGMA_array = np.full([nmths, nfirms], np.nan)
    tmr_array = np.full([nmths, nfirms], np.nan)

    for i in range(nmths):
        mthnum = uniym[i] % 100 #提取月份
        ynum = round(uniym[i] / 100) # 提取年份
        lastymth = (ynum - 1) * 100 + mthnum
        print(i,lastymth)
        tmpindex = np.where(yyyymm == lastymth)[0] # np.where新生成了一个index为行列号的数组
        print(tmpindex)

        if len(tmpindex) > 0: # 有前一年数据的就符合删选条件
            startindex = tmpindex[-1]
            endindex = uniymindex[i] # 月末的行列号
            prior1yselector = np.transpose(range(startindex,endindex))  # last 1 year, eg. 19960731-19970731

            drn_index_i = drn_index_num.iloc[prior1yselector]
            # print(drn_index_i)
            tmr_array[i] = np.prod((1 + drn_index_i),axis=0) - 1
            # print(tmr_array[i,:])
            drn_i_j = drn_num.iloc[prior1yselector] #截面提取，把所有公司的同一时间的数据赋值
            # print(drn_i_j)

            nvalid = np.sum(np.isfinite(drn_i_j))
            nvalid[nvalid == len(drn_i_j)] = 252

            X = sm.add_constant(drn_index_i,prepend=False) # 设定截距，后边的参数有无产生的改变的位置是cons在列表中的第一列还是第二列
            Y = drn_i_j
            for j in range(drn_i_j.shape[1]): #shape(0=行数，1=列数) 计算每家公司的日收益率对指数收益率的回归的残差的标准差
                # print(j)
                if nvalid[j] > 126:
                    y = Y.iloc[:,j].dropna() #把缺失值踢掉
                    residuals = sm.OLS(y,X.loc[y.index]).fit().resid # nan is not acceptable in models,so drop them
                    SIGMA_array[i, j] = np.nanstd(residuals, ddof=1) # 求得是每家公司在一年中每个交易日的sigma值

        if np.in1d(i, range(20,6001,20)):
            print(str(i)+'mths out of '+str(nmths)+' completed.')
    print(SIGMA_array)



    pd.DataFrame(data=SIGMA_array,index=mc_num.iloc[uniymindex,:].index,columns=mc_num.iloc[uniymindex,:].columns).to_csv('../data/SIGMA_array.csv')
    pd.DataFrame(data=tmr_array,index=mc_num.iloc[uniymindex,:].index,columns=mc_num.iloc[uniymindex,:].columns).to_csv('../data/tmr_array.csv')
    sio.savemat('../data/SIGMA_data.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes), 1),
                                            'mthdates': mthdates.values.reshape(len(mthdates), 1),
                                            'SIGMA_array': np.array(SIGMA_array), 'tmr_array': np.array(tmr_array)
                                            })

    return SIGMA_array, tmr_array



def debtface_fun():
    '''
    .MAT:
    allfirmcodes:all codes in bs_data_all, 3593*1 ndarray
    mthdates:all month-end dates(2019-02-28) in sigmae_array, 273*1 ndarray
    tcl_array: last available total-current-liability during around past 1-2 years at each month end, 273*3593 array
    tl_array: last available total-liability during around past 1-2 years at each month end, 273*3593 array
    csi_array: last available money_cap+trade_assets during around past 1-2 years at each month end, 273*3593 array
    fd_array: last available (total-current-liability+total-liability)/2 during around past 1-2 years at each month end, 273*3593 array
    :return: tcl_array,tl_array,csi_array,fd_array: 273*3593 df
    '''

    bs = sio.loadmat('../data/bs_data_all.mat')
    # tcl_num = bs.get('tcl_num') 在.mat文档中，提取变量用.get
    # tl_num = bs.get('tl_num')
    # csi_num = bs.get('csi_num')
    alldates = pd.Series(bs.get('alldates').squeeze()).astype('datetime64[ns]')
    allfirmcodes = bs.get('allfirmcodes')

    tcl_num = pd.read_csv('../data/tcl_num.csv',index_col=[0])
    tl_num = pd.read_csv('../data/tl_num.csv',index_col=[0])
    csi_num = pd.read_csv('../data/csi_num.csv',index_col=[0])
    tcl_num = tcl_num.loc[tcl_num.index.isnull() == False]
    tl_num = tl_num.loc[tl_num.index.isnull() == False]
    csi_num = csi_num.loc[csi_num.index.isnull() == False]

    print(tcl_num.shape)

    sigmae_array = pd.read_csv('../data/sigmae_array.csv',index_col=[0])
    mthdates = pd.read_csv('../data/mthdates.csv',index_col=[0]).squeeze().apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
    # print(mthdates)

    tcl_num_add = pd.DataFrame(columns=list(set(sigmae_array.columns)-set(tcl_num.columns)))  #adjust columns to match with 273*3593 集合作差
    #把没有的列索引找到
    tl_num_add = pd.DataFrame(columns=list(set(sigmae_array.columns)-set(tl_num.columns)))
    csi_num_add = pd.DataFrame(columns=list(set(sigmae_array.columns)-set(csi_num.columns)))
    tcl_num = pd.concat([tcl_num,tcl_num_add],axis=1)[sigmae_array.columns] #axis = 1 columns
    tl_num = pd.concat([tl_num, tl_num_add], axis=1)[sigmae_array.columns]
    csi_num = pd.concat([csi_num, csi_num_add], axis=1)[sigmae_array.columns]
    print(tcl_num.shape)
    # print(all(tcl_num.columns==sigmae_array.columns))


    alldates_num = alldates.dt.year * 1e+2 + alldates.dt.month
    alldates_sigma = mthdates
    y_all = alldates_sigma.dt.year
    m_all = alldates_sigma.dt.month

    tcl_array = np.full((sigmae_array.shape[0],sigmae_array.shape[1]),np.nan)
    tl_array = np.full((sigmae_array.shape[0],sigmae_array.shape[1]),np.nan)
    csi_array = np.full((sigmae_array.shape[0],sigmae_array.shape[1]),np.nan)
    fd_array = np.full((sigmae_array.shape[0],sigmae_array.shape[1]),np.nan)
    # print(tcl_array.shape)
    nmths = len(alldates_sigma)

    for i in range(nmths):
        y_i = y_all.iloc[i]
        m_i = m_all.iloc[i]
        selector_i = ((np.in1d(alldates.dt.year, [y_i - 1,y_i])) & (alldates_num <= (y_i * 100 + m_i))).tolist() #select period from last year to this current month
        print(i,selector_i)
        tcl_j_i = tcl_num.loc[selector_i,:].fillna(method='ffill') #对于缺失值的处理方式，找前一个数据进行填补
        tl_j_i = tl_num.loc[selector_i,:].fillna(method='ffill')
        csi_j_i = csi_num.loc[selector_i,:].fillna(method='ffill')
        # print(tcl_j_i)
        # print(tl_j_i)
        # print(csi_j_i)
        csi_array[i] = np.array(csi_j_i.iloc[-1])
        tcl_array[i] = np.array(tcl_j_i.iloc[-1])
        tl_array[i] = np.array(tl_j_i.iloc[-1])
        fd_array[i] = 0.5 * np.nansum([tcl_array[i], tl_array[i]],axis=0)
    fd_array[fd_array==0.] = np.nan
    print(csi_array)
    print(tcl_array)
    print(tl_array)
    print(fd_array)

    pd.DataFrame(data=tcl_array,index=sigmae_array.index,columns=sigmae_array.columns).to_csv('../data/tcl_array.csv')
    pd.DataFrame(data=tl_array,index=sigmae_array.index,columns=sigmae_array.columns).to_csv('../data/tl_array.csv')
    pd.DataFrame(data=csi_array,index=sigmae_array.index,columns=sigmae_array.columns).to_csv('../data/csi_array.csv')
    pd.DataFrame(data=fd_array,index=sigmae_array.index,columns=sigmae_array.columns).to_csv('../data/fd_array.csv')
    sio.savemat('../data/fd_data_all.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes), 1),
                                            'mthdates': mthdates.values.reshape(len(mthdates), 1),
                                            'tcl_array': tcl_array, 'tl_array': tl_array,
                                            'csi_array': csi_array, 'fd_array': fd_array,
                                            })

    return tcl_array, tl_array,csi_array,fd_array


def ni_fun():
    '''
    .MAT:
    allfirmcodes:all codes in bs_data_all, 3593*1 ndarray
    mthdates:all month-end dates(2019-02-28) in sigmae_array, 273*1 ndarray
    ni_array: last available net-income during around past 1-2 years at each month end, 273*3593 array
    :return: ni_array: last available net-income during around past 1-2 years at each month end, 273*3593 df
    '''

    is_data = sio.loadmat('../data/is_data_all.mat')
    # ni_num = is_data.get('ni_num')
    alldates = pd.Series(is_data.get('alldates').squeeze()).astype('datetime64[ns]')
    allfirmcodes = is_data.get('allfirmcodes')

    ni_num = pd.read_csv('../data/ni_num.csv',index_col=[0])
    print(ni_num.shape)
    ni_num = ni_num.loc[ni_num.index.isnull()==False]

    sigmae_array = pd.read_csv('../data/sigmae_array.csv',index_col=[0])
    mthdates = pd.read_csv('../data/mthdates.csv',index_col=[0]).squeeze().apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
    # print(mthdates)
    sigmae_array = sigmae_array.loc[sigmae_array.index.isnull()==False]
    ni_num = ni_num[sigmae_array.columns] #adjust columns to match with 273*3593
    print(ni_num.shape)

    alldates_num = alldates.dt.year * 1e+2 + alldates.dt.month
    alldates_sigma = mthdates
    y_all = alldates_sigma.dt.year
    m_all = alldates_sigma.dt.month

    ni_array = np.full((sigmae_array.shape[0],sigmae_array.shape[1]),np.nan)

    nmths = len(alldates_sigma)

    for i in range(nmths):
        y_i = y_all.iloc[i]
        m_i = m_all.iloc[i]
        selector_i = ((np.in1d(alldates.dt.year, [y_i - 1,y_i])) & (alldates_num <= (y_i * 100 + m_i))).tolist() #select period from last year to this current month
        print(i,selector_i)
        ni_j_i = ni_num.loc[selector_i,:].fillna(method='ffill')
        # print(ni_j_i)
        ni_array[i] = np.array(ni_j_i.iloc[-1])
    print(ni_array)

    pd.DataFrame(data=ni_array,index=sigmae_array.index,columns=sigmae_array.columns).to_csv('../data/ni_array.csv')
    sio.savemat('../data/ni_data_all.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes), 1),
                                            'mthdates': mthdates.values.reshape(len(mthdates), 1),
                                            'ni_array': ni_array
                                            })

    return ni_array

def ta_fun():
    '''
    .MAT:
    allfirmcodes:all codes in bs_data_all, 3593*1 ndarray
    mthdates:all month-end dates(2019-02-28) in sigmae_array, 273*1 ndarray
    ta_array: last available total-assets during around past 1-2 years at each month end, 273*3593 array
    :return: ta_array: last available total-assets during around past 1-2 years at each month end, 273*3593 df
    '''

    bs = sio.loadmat('../data/bs_data_all.mat')
    # ta_num = bs.get('ta_num')
    alldates = pd.Series(bs.get('alldates').squeeze()).astype('datetime64[ns]')
    allfirmcodes = bs.get('allfirmcodes')

    ta_num = pd.read_csv('../data/ta_num.csv',index_col=[0])
    print(ta_num.shape)
    ta_num = ta_num.loc[ta_num.index.isnull()==False]

    sigmae_array = pd.read_csv('../data/sigmae_array.csv',index_col=[0])
    mthdates = pd.read_csv('../data/mthdates.csv',index_col=[0]).squeeze().apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
    # print(mthdates)
    sigmae_array=sigmae_array.loc[sigmae_array.index.isnull()==False]
    ta_num_add = pd.DataFrame(columns=list(set(sigmae_array.columns) - set(ta_num.columns))) #adjust columns to match with 273*3593
    ta_num = pd.concat([ta_num, ta_num_add], axis=1)[sigmae_array.columns]
    print(ta_num.shape)

    alldates_num = alldates.dt.year * 1e+2 + alldates.dt.month
    alldates_sigma = mthdates
    y_all = alldates_sigma.dt.year
    m_all = alldates_sigma.dt.month

    ta_array = np.full((sigmae_array.shape[0],sigmae_array.shape[1]),np.nan)

    nmths = len(alldates_sigma)

    for i in range(nmths):
        y_i = y_all.iloc[i]
        m_i = m_all.iloc[i]
        selector_i = ((np.in1d(alldates.dt.year, [y_i - 1,y_i])) & (alldates_num <= (y_i * 100 + m_i))).tolist() #select period from last year to this current month
        print(i,selector_i)
        ta_j_i = ta_num.loc[selector_i,:].fillna(method='ffill')
        # print(ta_j_i)
        ta_array[i] = np.array(ta_j_i.iloc[-1])
    print(ta_array)

    pd.DataFrame(data=ta_array,index=sigmae_array.index,columns=sigmae_array.columns).to_csv('../data/ta_array.csv')
    sio.savemat('../data/ta_data_all.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes), 1),
                                            'mthdates': mthdates.values.reshape(len(mthdates), 1),
                                            'ta_array': ta_array
                                            })

    return ta_array



def relativesize_fun():
    '''
    .MAT:
    allfirmcodes: all codes in sigmaE_data, 3593*1 ndarray
    mthdates: all month-end dates in sigmaE_data, 273*1 ndarray
    rs_array: relative size, 273*3593 array
    :return: rs_array: relative size, 273*3593 df
    '''

    sigmaE_data = sio.loadmat('../data/sigmaE_data.mat')
    mthdates = sigmaE_data.get('mthdates')
    allfirmcodes = sigmaE_data.get('allfirmcodes')
    # mc_array = sigmaE_data.get('mc_array')
    mc_array = pd.read_csv('../data/mc_array.csv',index_col=[0])
    mc_array =mc_array.loc[mc_array.index.isnull()==False]

    mdmc = np.nanmedian(mc_array,axis=1) #忽略缺失值求变量的中位数

    rs_array = np.log(np.apply_along_axis(lambda x: x/mdmc,0,mc_array)) # axis= 0 中间参数指的是按列计算（第一个参数：运算函数，第二个参数：计算方式按行还是按列；
    # 第三个参数：应用哪个变量的值进行计算
    print(pd.DataFrame(data=rs_array,index=mc_array.index,columns=mc_array.columns))

    sio.savemat('../data/relativesize_data.mat', {'allfirmcodes': allfirmcodes.reshape(len(allfirmcodes), 1),
                                            'mthdates': mthdates.reshape(len(mthdates), 1),
                                            'rs_array': rs_array})

    pd.DataFrame(data=rs_array,index=mc_array.index,columns=mc_array.columns).to_csv('../data/rs_array.csv')
    return rs_array


def interestrate_fun():
    '''
    .MAT:
    mthdates: all month-end dates in sigmaE_data, 273*1 ndarray
    interestrates: deposit interest rates, 273*1 ndarray
    :return: deposit interest rates, 273*1 Series
    '''

    Chinainterestrates = pd.read_csv('../data/Chinainterestrates.csv',index_col=[0],parse_dates=[0])
    print(Chinainterestrates)

    mthdates = pd.read_csv('../data/mthdates.csv',index_col=[0]).squeeze().apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
    print(mthdates)

    interestrates_raw = Chinainterestrates.groupby(pd.Grouper(freq='m'))['CHINATIMEDEPOSITRATE1YMIDDLERATE'].nth(-1) # according to normal month end
    # 按月freq= 'm' [month end frequency]
    interestrates = Chinainterestrates.loc[mthdates.tolist(),'CHINATIMEDEPOSITRATE1YMIDDLERATE']  #according to mthdates
    interestrates.loc[pd.isnull(interestrates)] = interestrates_raw.loc[interestrates.loc[pd.isnull(interestrates)].index] #fill nan with interestrates_raw
    # 将interestrates中的空值定位出来并赋值，赋予interestrates_raw的同时间（index为时间）的值
    print(interestrates.value_counts())

    sio.savemat('../data/interestrates_data.mat', {'mthdates': mthdates.values.reshape(len(mthdates), 1),'interestrates': np.array(interestrates)})
    interestrates.to_csv('../data/interestrates.csv')


    return interestrates


def sigma_V_fun():
    '''
   .MAT:
   allfirmcodes: all codes in sigmaV_data, 3593*1 ndarray
   mthdates:all month-end dates(2019-02-28) in sigmaV_data, 273*1 ndarray
   sigmav_array: calculated from mc_array,etc., 273*3593 array
   :return: sigmav_array, calculated from mc_array,etc., 273*3593 df
    '''

    E = sio.loadmat('../data/sigmaE_data.mat')
    # sigmae_array = E.get('sigmae_array')
    # sigmad_array = E.get('sigmad_array')
    # mc_array = E.get('mc_array')
    # fd = sio.loadmat('../data/fd_data_all.mat')
    # fd_array = fd.get('fd_array')

    mc_array = pd.read_csv('../data/mc_array.csv', index_col=[0])
    fd_array = pd.read_csv('../data/fd_array.csv', index_col=[0])
    sigmae_array = pd.read_csv('../data/sigmae_array.csv', index_col=[0])
    sigmad_array = pd.read_csv('../data/sigmad_array.csv', index_col=[0])

    mc_array = mc_array.loc[mc_array.index.isnull()==False]
    fd_array = fd_array.loc[fd_array.index.isnull() == False]
    sigmae_array = sigmae_array.loc[sigmae_array.index.isnull() == False]
    sigmad_array = sigmad_array.loc[sigmad_array.index.isnull() == False]

    sigmav_array = mc_array/(mc_array + fd_array)*sigmae_array + fd_array /(mc_array + fd_array)*sigmad_array
    print(sigmav_array)

    sio.savemat('../data/naiveDD_data.mat', {'allfirmcodes': E.get('allfirmcodes'),
                                             'mthdates': E.get('mthdates'),
                                             'sigmav_array': np.array(sigmav_array)})
    sigmav_array.to_csv('../data/sigmav_array.csv')
    return sigmav_array


def naiveDD():
   '''
   .MAT:
   allfirmcodes: all codes in sigmaV_data, 3593*1 ndarray
   mthdates:all month-end dates(2019-02-28) in sigmaV_data, 273*1 ndarray
   naive dd, calculated from mc_array,etc., 273*3593 array
   :return: naive dd, calculated from mc_array,etc., 273*3593 df
   '''

   # V = sio.loadmat('../data/sigmaV_data.mat')
   # print(V)
   E = sio.loadmat('../data/sigmaE_data.mat')
   # # print(E)
   # fd = sio.loadmat('../data/fd_data_all.mat')
   # # print(fd)
   #
   # mc_array = E.get('mc_array')
   # fd_array = fd.get('fd_array')
   # mu_array = E.get('mu_array')
   # sigmav_array = V.get('sigmav_array')

   mc_array = pd.read_csv('../data/mc_array.csv',index_col=[0])
   fd_array = pd.read_csv('../data/fd_array.csv', index_col=[0])
   mu_array = pd.read_csv('../data/mu_array.csv', index_col=[0])
   sigmav_array = pd.read_csv('../data/sigmav_array.csv', index_col=[0])

   mc_array = mc_array.loc[mc_array.index.isnull()==False]
   fd_array = fd_array.loc[fd_array.index.isnull() == False]
   mu_array = mu_array.loc[mu_array.index.isnull() == False]
   sigmav_array = sigmav_array.loc[sigmav_array.index.isnull() == False]


   T = 1

   dd_array = (np.log((mc_array + fd_array)/fd_array) + (mu_array - 0.5*sigmav_array**2)*T ) / (sigmav_array*math.sqrt(T))
   # 计算公式
   print(dd_array)

   dd_array.to_csv('../data/dd_array.csv')
   sio.savemat('../data/naiveDD_data.mat', {'allfirmcodes': E.get('allfirmcodes'),
                                           'mthdates': E.get('mthdates'),
                                           'dd_array': np.array(dd_array)})
   return dd_array


if __name__ == '__main__':
    # sigma_equity()
    # SIGMA_fun()

    # debtface_fun()
    # ta_fun()
    # ni_fun()
    #
    # relativesize_fun()
    # interestrate_fun()
    #
    # sigma_V_fun()
    naiveDD()
