import pandas as pd
import numpy as np
import scipy.io as sio
from sqlalchemy import create_engine
def firmspecifc_separateDD_fun():
    '''
    .MAT:
    firmspecific: concat all variables, 273*11*3593 array
    allfirmcodes:all codes in naiveDD_data, 3593*1 ndarray
    mthdates: all month-end dates(2019-02-28) in naiveDD_data, 273*1 ndarray
    :return: firmspecific: concat all variables, 273*11*3593 array
    '''

    DD = sio.loadmat('../data/naiveDD_data.mat')
    mthdates = DD.get('mthdates')
    allfirmcodes = DD.get('allfirmcodes')
    dd_array = DD.get('dd_array')
    ta = sio.loadmat('../data/ta_data_all.mat')
    ta_array = ta.get('ta_array')
    # print(allfirmcodes)
    # print(pd.Series(mthdates.squeeze()).astype('datetime64[ns]'))

    T = len(mthdates)
    n = len(allfirmcodes)
    firmspecific = np.full([T, 11, n],np.nan) # 构建一个273*3593的矩阵，每个单元格里是由11个元素组成
    print(firmspecific.shape)


    SIG = sio.loadmat('../data/SIGMA_data.mat')
    tmr_array = SIG.get('tmr_array')
    firmspecific[:, 0, :] = tmr_array

    r = sio.loadmat('../data/interestrates_data.mat')
    interestrates = r.get('interestrates')
    firmspecific[:, 1, :] = np.tile(interestrates, (n, 1)).T # .tile(A,n)重复A值n次，当n为（n,m）时，意思是以A为数组重复n次，数组内部数值重复m次
    #对11个要素中的第二个进行赋值，
    # load('../data/naiveDD_data.mat', 'dd_array')
    financials = pd.read_csv('../data/financials.csv',encoding='gbk') # 导入公司简称
    fin_selector = np.in1d(allfirmcodes, financials['wind_code']) #标记出金融行业公司
    # print(fin_selector)
    dd_array_nonfin = dd_array.copy()
    dd_arrary_fin = dd_array.copy()
    dd_array_nonfin[(np.isfinite(dd_array)) & (np.tile(fin_selector, (np.size(dd_array,0),1)))] = 0
    # 是否为空值返回布尔值，以整体重复dd_array的行数的个数np.size(变量，根据shape的形式确定位置以及参数取值（0,1,2）0【行数】/1【中间形式的数量】/2【列数】/默认是整个矩阵的元素个数)，赋值含义是把金融行业的数据刷为0，剩下的有效数据为非金融行业
    # print(dd_array_nonfin)
    dd_arrary_fin[(np.isfinite(dd_array)) & (np.tile(~fin_selector, (np.size(dd_array,0),1)))] = 0 # 赋值含义是把非金融行业的数据刷为0，留下的有效数据为金融行业
    #~取相反值，
    firmspecific[:, 2 , :] = dd_arrary_fin
    firmspecific[:, 3 , :] = dd_array_nonfin

    ni = sio.loadmat('../data/ni_data_all.mat')
    ni_array = ni.get('ni_array')
    firmspecific[:, 4, :] = ni_array/ta_array

    fd = sio.loadmat('../data/fd_data_all.mat')
    tl_array = fd.get('tl_array')
    tlta_array = tl_array/ta_array
    tlta_array_nonfin = tlta_array.copy()
    tlta_arrary_fin = tlta_array.copy()
    tlta_array_nonfin[(np.isfinite(tlta_array)) & (np.tile(fin_selector, (np.size(tlta_array,0),1)))] = 0
    # 非金融行业的指标
    tlta_arrary_fin[(np.isfinite(tlta_array)) & (np.tile(~fin_selector, (np.size(tlta_array,0),1)))] = 0
    # 金融行业的
    firmspecific[:, 5 , :] = tlta_arrary_fin
    firmspecific[:, 6 , :] = tlta_array_nonfin

    rela = sio.loadmat('../data/relativesize_data.mat')
    rs_array = rela.get('rs_array')
    firmspecific[:, 7, :] = rs_array

    SIGMA = sio.loadmat('../data/SIGMA_data.mat',)
    SIGMA_array = SIGMA.get('SIGMA_array')
    firmspecific[:, 8, :] = SIGMA_array

    sigmaE = sio.loadmat('../data/sigmaE_data.mat')
    mc_array = sigmaE.get('mc_array')
    firmspecific[:, 9, :] = mc_array/(ta_array - tl_array)

    fd = sio.loadmat('../data/fd_data_all.mat', )
    csi_array = fd.get('csi_array')
    firmspecific[:, 10, :] = csi_array/ta_array

    pd.DataFrame(allfirmcodes).to_csv('../data/allfirmcodes.csv')

    sio.savemat('../data/firmspecific_separateDD.mat', {'firmspecific':firmspecific, 'allfirmcodes':allfirmcodes, 'mthdates':mthdates})

    for i in range(np.size(firmspecific, 2)):
        for j in range(np.size(firmspecific, 1)):
            firmspecific[:, j, i] = np.array(pd.Series(firmspecific[:, j, i]).fillna(method='ffill'))
            # 以公司为单位，对于每个元素的空值取前一年的相同指标下的值


    sio.savemat('../data/firmspecific_separateDD_nomissing.mat', {'firmspecific':firmspecific, 'allfirmcodes':allfirmcodes, 'mthdates':mthdates})
    return firmspecific


def firmspecific_expanding(allowmissing):
    '''
    :param allowmissing: bool, false for ffilling in firmspecific_seperateDD_fun.py
    .MAT:
    firmspecific:treat any 1 default firm as several firms according to its default dates, remove default firm's original data and add back data of "several firms", 273*11*4166 array
    mthdates: all month-end dates(2019-02-28) in firmspecific_separateDD, 273*1 ndarray
    allfirmcodes: treat any 1 default firm as several firms, remove default firm's code and add back codes of "several_firms", 4166*1 ndarray
    :return: firmspecific,273*11*4166 array
    '''

    if allowmissing:
        data = sio.loadmat('../data/firmspecific_separateDD.mat')
    else:
        data = sio.loadmat('../data/firmspecific_separateDD_nomissing.mat')
    firmspecific = data.get('firmspecific')
    allfirmcodes = pd.Series(data.get('allfirmcodes').squeeze()).tolist()
    allfirmcodes = pd.Series([allfirmcodes[i][0] for i in range(len(allfirmcodes))]) #drop []
    # print(allfirmcodes)
    # print(len(allfirmcodes))
    mthdates = pd.Series(data.get('mthdates').squeeze()).astype('datetime64[ns]')
    # print(mthdates)
    Chinadefaultevents = pd.read_csv('../data/Chinadefaultevents.csv',parse_dates=['DefaultDate']) # 数据为违约事件
    unidefcodes = sorted(pd.unique(Chinadefaultevents['firmcode']).tolist()) #sort return None 提取违规公司的股票代码
    # print(unidefcodes)

    dates_yyyymm = mthdates.dt.year * 100 + mthdates.dt.month # 取交易年月
    firmspecific0 = firmspecific.copy()
    removelist = []
    fs2add = np.full((firmspecific.shape[0],firmspecific.shape[1],1),np.nan) # 现铺一个（时间，指标，公司）三维的平面
    codes2add = []
    residual = fs2add.copy()
    repair_list = []


    for i in range(len(unidefcodes)): # 针对357家违规公司进行操作
    # for i in range(1): # for test
        # print(i,np.in1d(unidefcodes[i], allfirmcodes))
        if np.in1d(unidefcodes[i], allfirmcodes):
            selector_fs = np.where(np.in1d(allfirmcodes, unidefcodes[i]))[0][0] #index for default firmcode 标记违规公司在总公司代码文件中的位置
            # print(selector_fs)
            removelist.append(selector_fs)
            fs_i = firmspecific0[:,:, selector_fs] # 取违规公司的所有数据
            # print(fs_i)

            selector_def = np.in1d(Chinadefaultevents['firmcode'], unidefcodes[i]) #logical position for default firmcode, to obatin respective default date
            # 返回布尔值，违约事件文件中涉及的违规公司的代码是否出现在专属违规公司代码文件中，T/F
            def_yyyymm_i = pd.unique(Chinadefaultevents.loc[selector_def,'DefaultDate'].dt.year * 100 + Chinadefaultevents.loc[selector_def,'DefaultDate'].dt.month)
            # 将违约发生时间年月提取出来
            def_yyyymm_i = def_yyyymm_i[def_yyyymm_i <= dates_yyyymm.iloc[-1]] # 确保违约时间发生在研究对象的时间内
            # print(def_yyyymm_i)

            for j in range(len(def_yyyymm_i)): # 看这家公司有几次违规
                endindex = np.where(dates_yyyymm == def_yyyymm_i[j])[0][0] - 1 #why -1? #定位到发生违约的那一天
                # 把结束索引定位在了违约发生时间的前一个月
                # print(endindex)
                if j == 0:
                    startindex = 0
                else:
                    if np.where(dates_yyyymm == def_yyyymm_i[j - 1])[0][0] + 1 < len(dates_yyyymm) - 1: # 第一次出现违约事件对应时间年月的下一个月不超过实验年月时间个数减1
                        startindex = np.where(dates_yyyymm == def_yyyymm_i[j - 1])[0][0] + 1
                    else:
                        break
                # print(startindex)

                fs_tmp = fs_i.copy() # 违约公司在实验时间内的每个时期的11个指标值
                fs_tmp[list(range(0,startindex+1))+list(range(endindex,len(fs_i))), :] = np.nan # 保留两次违规之间的数据
                # fs_tmp[list(range(0,startindex)) + list(range(endindex,len(fs_i))),:] = np.nan
                # print(fs_tmp)
                fs2add = np.concatenate([fs2add,fs_tmp.reshape(fs_tmp.shape[0],fs_tmp.shape[1],1)],axis=2) # concatenate(axis = 0按行插入,axis = 1按列插入，None全部加入到一个数组当中)
                # 两个数组进行合并的时候，shape要一致，将两次违规中间的数据赋值
                print(fs2add.shape) # 在实验时间内单个公司的指标表现
                print(fs2add[:, :, i+j+1]) # 作为第i+1个插入的违规公司
                codes2add.append(unidefcodes[i]+'_'+str(j)) # 将违规公司的违约前期数据单独设定给_1/2等子标题变量中
                print(codes2add)

            no_def = fs_i.copy()
            no_def[list(range(0,endindex)),:] = np.nan
            residual = np.concatenate([residual,no_def.reshape(no_def.shape[0],no_def.shape[1],1)],axis=2) #调取某家公司的全部年份数据fs2add[:,:,n]
            repair_list.append(unidefcodes[i])

    firmspecific = np.delete(firmspecific,removelist,axis=2) # 删掉违约公司的数据,(删谁，删第几个，按行=0，按列=1，对于三维矩阵来讲按列=2)
    fs2add = np.delete(fs2add,0,axis=2) # 删掉了初始化的第一个全nan的平面数据
    firmspecific = np.concatenate([firmspecific,fs2add],axis=2)
    residual = np.delete(residual,0,axis=2)
    firmspecific = np.concatenate([firmspecific,residual],axis=2)
    # print(firmspecific[107,:,3999].tolist())
    # print(firmspecific[:,:,3799].tolist())

    # 更新违约公司的数据
    allfirmcodes = allfirmcodes.drop(removelist,axis=0) # 按行0，按列1
    allfirmcodes = allfirmcodes.append(pd.Series(codes2add)).reset_index(drop=True)
    allfirmcodes = allfirmcodes.append(pd.Series(repair_list)).reset_index(drop = True)
    # print(allfirmcodes)

    allfirmcodes.to_csv('../data/allfirmcodes.csv')
    if allowmissing:
        sio.savemat('../data/firmspecific_separateDD_defadded.mat', {'firmspecific':firmspecific, 'allfirmcodes':np.array(allfirmcodes), 'mthdates':np.array(mthdates)})
    else:
        sio.savemat('../data/firmspecific_separateDD_defadded_nomissing.mat',{'firmspecific': firmspecific, 'allfirmcodes': np.array(allfirmcodes), 'mthdates': np.array(mthdates)})
    return firmspecific,allfirmcodes, mthdates


def firmlist(allowmissing,qstr):
    '''

    :param allowmissing: bool, false for ffilling in firmspecific_seperateDD_fun.py
    :param qstr: str, ''for normal use
    :return:
    '''

    if allowmissing:
        V = sio.loadmat('../data/firmspecific'+ qstr +'_separateDD_defadded.mat')
    else:
        V = sio.loadmat('../data/firmspecific' + qstr + '_separateDD_defadded_nomissing.mat')

    firmspecific = V.get('firmspecific')
    allfirmcodes = pd.Series(V.get('allfirmcodes').squeeze()).tolist()
    allfirmcodes = pd.Series([allfirmcodes[i][0] for i in range(len(allfirmcodes))])  # drop [] len里是没有违约的公司的数量
    # print(allfirmcodes)
    mthdates = pd.Series(V.get('mthdates').squeeze()).astype('datetime64[ns]')

    firmlist = np.zeros((len(allfirmcodes), 3)) # 设一个非违约公司个数*3的矩阵
    Tvector = np.array(range(len(mthdates))) # 设定一个实验时间内的向量
    removelist = []

    for i in range(len(allfirmcodes)):
    # for i in range(3893,3895):
        fs_i = firmspecific[:,:,i] # 锁定第公司的所有时间数据
        validselector = (np.sum(np.isfinite(fs_i),axis=1)>0).tolist() # 一家公司里每个时期的11个要素是否均为空值
        # print(validselector)
        print(i,sum(validselector))

        if sum(validselector) > 0:
            tindex_tmp = Tvector[validselector] # 里面的变量是一个布尔值组成的数组，含义是将布尔值为1的行索引存储在向量中
            try:
                firmlist[i, 0:2] = tindex_tmp[[0,-1]] # 标记有效数据的行索引始末，存储在firmlist的N*3矩阵当中前两列
            except:
                print('hi')

            if '_' in str(allfirmcodes[i]):
                firmlist[i, 2] = 1 # firmlist第三列用来标记是否为违约公司的数据

        else:
            removelist.append(i)

        print(firmlist[i])




    firmspecific = np.delete(firmspecific,removelist,axis=2) # 删掉一列后面依次向前移动
    allfirmcodes = allfirmcodes.drop(removelist,axis=0).reset_index(drop=True) # 删除行数据让后面的数据依次向上移动,保证索引值连续
    firmlist = np.delete(firmlist,removelist,axis=0)
    print(firmspecific.shape)
    print(allfirmcodes)
    print(pd.DataFrame(firmlist).loc[3859:3983])


    if allowmissing:
        sio.savemat('../data/fl_fs'+ qstr +'_final.mat', {'firmspecific':firmspecific,'firmlist':firmlist, 'allfirmcodes':np.array(allfirmcodes), 'mthdates':np.array(mthdates)})
    else:
        sio.savemat('../data/fl_fs' + qstr + '_final_nomissing.mat',{'firmspecific': firmspecific,'firmlist':firmlist, 'allfirmcodes': np.array(allfirmcodes), 'mthdates': np.array(mthdates)})
    return firmspecific,allfirmcodes,firmlist


if __name__ == '__main__':
    # firmspecifc_separateDD_fun()
    # firmspecific_expanding(False)
    firmlist(False,'')

    # engine = create_engine(
    #     'mysql+pymysql://shuhao:shuhao123@cdb-n3nb31fy.bj.tencentcdb.com:10232/active_db?charset=utf8')
    # sql = '''select * from bond_break_probability'''
    # test = pd.read_sql_query(sql,engine)
