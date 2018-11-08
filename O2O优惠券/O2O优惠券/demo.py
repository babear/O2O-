# 导入包
import pandas as pd # 数据处理包
import numpy as np # 矩阵处理包 
from datetime import datetime,timedelta # 时间处理包

# 读取数据
cookly_offline_train = pd.read_csv('./Data/ccf_offline_stage1_train.csv')
cookly_online_train = pd.read_csv('./Data/ccf_online_stage1_train.csv')
cookly_offline_test = pd.read_csv('./Data/ccf_offline_stage1_test_revised.csv')

# 可以看到读入的数据存在不一致的情况，需要做一些简单的处理
def covToNaN(x, type = None):
    if x == 'null':
        return np.nan
    elif type != None:
        return type(x)
    else:
        return x

cookly_offline_train['User_id'] = cookly_offline_train['User_id'].apply(covToNaN, args=(str,))
# 处理数据
print('data deal begin')
deal_offline_columns = ['User_id','Merchant_id','Coupon_id','Discount_rate', 'Distance', 'Date_received', 'Date']
for columns in deal_offline_columns[2:]:
    cookly_offline_train[columns] = cookly_offline_train[columns].map(covToNaN,str)

deal_online_columns = ['User_id','Merchant_id','Action','Coupon_id','Discount_rate', 'Date_received', 'Date']
for columns in deal_online_columns[3:]:
    cookly_online_train[columns] = cookly_online_train[columns].map(covToNaN)

deal_offline_test_columns = ['User_id','Merchant_id','Coupon_id','Discount_rate', 'Distance', 'Date_received']
cookly_offline_test['Coupon_id'] = cookly_offline_test[columns].map(covToNaN,str())
cookly_offline_test['Discount_rate'] = cookly_offline_test[columns].map(covToNaN)
cookly_offline_test['Distance'] = cookly_offline_test[columns].map(covToNaN)
cookly_offline_test['Date_received'] = cookly_offline_test[columns].map(covToNaN,str())
print('data deal end')

# 数据分表处理
# 分表里有：offline数据：
# 优惠券领取结构相似，方便统一提取特征
# 业务上可以去掉无效字段，使用时再合并        
print('分表处理开始')
# offline 数据分成两张表
cookly_offline_train_notnull_coupon = cookly_offline_train[cookly_offline_train['Coupon_id'].notnull()]
cookly_offline_train_null_coupon = cookly_offline_train[cookly_offline_train['Coupon_id'].isnull()]

cookly_online_train_notnull_coupon = cookly_online_train[cookly_online_train['Coupon_id'].notnull()] # Action '1','2'
cookly_online_train_null_coupon = cookly_online_train[cookly_online_train['Coupon_id'].isnull()] # Action '0','1'

cookly_offline_test_coupon = cookly_offline_test.copy()
print('分表处理结束')

# 建立候选集：训练集和测试集
# \：为代码换行使用
# 数据中存在一个用户同一天领取同一张优惠券的情况
# 也可以预测后去重，预测后去重的好处是，可以提取当天领取的第几张优惠券这类特征
# 本次为baseline，简单处理，在初始就直接去重
# 用户去重：drop_duplicates(['User_id','Coupon_id','Date_received'])
# 两份训练集：线下验证
# train1 时间：20160516--20160615
# train2 时间：20160501--20160531
# test 时间：20160701--20160731
print('候选集建立开始')
cookly_train_user_label1 = cookly_offline_train_notnull_coupon[(cookly_offline_train_notnull_coupon['Date_received']>='20160516')\
                 &(cookly_offline_train_notnull_coupon['Date_received']<='20160615')][['User_id','Merchant_id','Coupon_id','Date_received','Date']].\
                 drop_duplicates(['User_id','Coupon_id','Date_received'])
cookly_train_user_label2 = cookly_offline_train_notnull_coupon[(cookly_offline_train_notnull_coupon['Date_received']>='20160501')\
                 &(cookly_offline_train_notnull_coupon['Date_received']<='20160531')][['User_id','Merchant_id','Coupon_id','Date_received','Date']].\
                 drop_duplicates(['User_id','Coupon_id','Date_received'])
cookly_test_user_label = cookly_offline_test_coupon[(cookly_offline_test_coupon['Date_received']>='20160701')\
                &(cookly_offline_test_coupon['Date_received']<='20160731')][['User_id','Merchant_id','Coupon_id','Date_received']].\
                drop_duplicates(['User_id','Coupon_id','Date_received'])
print('候选集建立结束')

#cookly_offline_train_notnull_coupon.sort_values(by = 'Date_received',axis = 0,ascending = True)


# 打标签逻辑：用户15天内使用则为1，否则为0
def get_label(data):
    data['Date_15'] = (pd.to_datetime(data['Date_received'])+timedelta(15)).astype(str)
    data['Date_15'] = data['Date_15'].map(lambda x: x.replace('-',''))
    data['label'] = (data['Date']<data['Date_15']).astype(int)
    data = data[['User_id','Merchant_id','Coupon_id','Date_received','label']]
    return data

# 数据打标签 训练集处理，预测集无需处理
print('训练集打标签开始')
cookly_train_user_label1 = get_label(cookly_train_user_label1)
cookly_train_user_label2 = get_label(cookly_train_user_label2)
print('训练集打标签结束')

# 简单查看数据：分析训练集和验证集
# 训练集合用户样本量是测试集合的一倍多
# 如果为保持样本数据提取某些特征一致，可以考虑训练集抽样一致
print('训练集 用户数1：',len(cookly_train_user_label1.User_id.unique()))
print('训练集 用户数2：',len(cookly_train_user_label2.User_id.unique()))
print('测试集 用户数：',len(cookly_test_user_label.User_id.unique()))
print('训练集 领取优惠券数1：',len(cookly_train_user_label1))
print('训练集 领取优惠券数2：',len(cookly_train_user_label2))
print('测试集 领取优惠券数：',len(cookly_test_user_label))

# 提取特征代码中使用的函数
# 特征处理函数
def get_max(x):
    try:
        return int(x.split(':')[0])
    except:
        return -1
def get_min(x):
    try:
        return int(x.split(':')[1])
    except:
        return -1
def get_rate(x):
    try:
        return float(x.split(':')[0]) * 1.0 / float(x.split(':')[1])
    except:
        try:
            return float(x)
        except:
            return -1
# 处理时间函数
def get_date_delta(year,month,day,delta_day):
    return str(datetime(year,month,day)-timedelta(delta_day)).split(' ')[0].replace('-','')
def get_date(year,month,day):
    return str(datetime(year,month,day)).split(' ')[0].replace('-','')
# 特征处理函数
def feat_count(df, df_feature, fe,value,name):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
def feat_mean(df, df_feature, fe, value, name):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
def feat_std(df, df_feature, fe, value, name):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
def feat_sum(df, df_feature, fe, value, name):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
def feat_max(df, df_feature, fe, value, name):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
def feat_min(df, df_feature, fe, value, name):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
def feat_nunique(df, df_feature, fe, value, name):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df	

# offline_coupon 数据：特征处理
def offline_coupon_features(data_user, data, end_day, windows_list, FLAG):
    print('offline_coupon 特征处理开始')
    data = data.copy()
    data_user = data_user.copy()
    data['Discount_type_1'] = data['Discount_rate'].map(lambda x: 1 if ':' in x else 0)
    data['Discount_type_0'] = data['Discount_rate'].map(lambda x: 0 if ':' in x else 1)    
    data['Discount_max'] = data['Discount_rate'].map(get_max)
    data['Discount_min'] = data['Discount_rate'].map(get_min)
    data['Discount_rate'] = data['Discount_rate'].map(get_rate)
    data['flag'] = (data['Date'].isnull()).astype(int)
    data['Distance'] = data['Distance'].fillna(0).astype(int)
    end_ = get_date(*end_day)
    print(end_)
    for day in windows_list: 
        begin_ = get_date_delta(*end_day,day)
        data_temp = data[(data['Date_received']>=begin_)&(data['Date_received']<=end_)]
        
        for col in ['User_id','Merchant_id','Coupon_id']:
#             print('领券统计')
            data_user = feat_count(data_user,data_temp,[col],'flag',name=FLAG+col+'_q1_cnt_'+str(day))
#             print('用券统计')
            data_user = feat_count(data_user,data_temp[data_temp['Date'].isnull()],[col],'flag',name=FLAG+col+'_q2_cnt_'+str(day))
#             print('未用券统计')
            data_user = feat_count(data_user,data_temp[~data_temp['Date'].isnull()],[col],'flag',name=FLAG+col+'_q3_cnt_'+str(day))       
#             print('用券率')
            data_user = feat_mean(data_user,data_temp,[col],'flag',name=FLAG+col+'_flag_mean_'+str(day))
#             print('用券方差')
            data_user = feat_std(data_user,data_temp,[col],'flag',name=FLAG+col+'_flag_std_'+str(day))         
#             print('券类型1')
            data_user = feat_sum(data_user,data_temp,[col],'Discount_type_1',name=FLAG+col+'_dt1_sum_'+str(day)) 
#             print('券类型0')
            data_user = feat_sum(data_user,data_temp,[col],'Discount_type_1',name=FLAG+col+'_dt0_sum_'+str(day)) 
            for distance in range(11):
                data_user = feat_count(data_user,data_temp[data_temp['Distance']==distance],[col],'Date',name=FLAG+col+'_'+str(distance)+'_Pd_cnt_'+str(day))
            for col2 in ['Discount_max','Discount_min','Discount_rate','Distance']:
#                 print('最大')
                data_user = feat_max(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_max_'+str(day))
#                 print('最小')
                data_user = feat_min(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_min_'+str(day))               
#                 print('平均')
                data_user = feat_mean(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_mean_'+str(day))                
#                 print('方差')
                data_user = feat_std(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_std_'+str(day)) 
        for col in ['Merchant_id','Coupon_id','Discount_rate','Distance','Date_received','Date']:
            data_user = feat_nunique(data_user, data_temp, ['User_id'], col, name=FLAG+col+"_U_uq_"+str(day))
        for col in ['User_id','Coupon_id','Discount_rate','Distance','Date_received','Date']:
            data_user = feat_nunique(data_user, data_temp, ['Merchant_id'], col, name=FLAG+col+"_M_uq_"+str(day))
        for col in ['User_id','Merchant_id','Discount_rate','Distance','Date_received','Date']:
            data_user = feat_nunique(data_user, data_temp, ['Coupon_id'], col, name=FLAG+col+"_C_uq_"+str(day))   
    print('offline_coupon 特征处理结束')
    return data_user

# offline_consumption 数据：特征处理
def offline_consumption_features(data_user, data, end_day, windows_list, FLAG):
    print('offline_consumption 特征处理开始')
    data = data.copy()
    data_user = data_user.copy()
    data['Distance'] = data['Distance'].astype(int)
    end_ = get_date(*end_day)
    print(end_)
    for day in windows_list: 
        begin_ = get_date_delta(*end_day,day)
        data_temp = data[(data['Date_received']>=begin_)&(data['Date_received']<=end_)]
        
        for col in ['User_id','Merchant_id','Coupon_id']:
#             print('购买统计')
            data_user = feat_count(data_user,data_temp,[col],'Date',name=FLAG+'_q1_cnt_'+str(day))
#             print('具体统计')
            for distance in range(11):
                data_user = feat_count(data_user,data_temp[data_temp['Distance']==distance],[col],'Date',name=FLAG+col+'_'+str(distance)+'_Pd_cnt_'+str(day))
            for col2 in ['Distance']:
#                 print('最大')
                data_user = feat_max(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_max_'+str(day))
#                 print('最小')
                data_user = feat_min(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_min_'+str(day))               
#                 print('平均')
                data_user = feat_mean(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_mean_'+str(day))                
#                 print('方差')
                data_user = feat_std(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_std_'+str(day)) 
        for col in ['Merchant_id','Distance','Date']:
            data_user = feat_nunique(data_user, data_temp, ['User_id'], col, name=FLAG+col+"_U_uq_"+str(day))
        for col in ['User_id','Distance','Date']:
            data_user = feat_nunique(data_user, data_temp, ['Merchant_id'], col, name=FLAG+col+"_M_uq_"+str(day))
    print('offline_consumption 特征处理结束')
    return data_user

# online_coupon 数据：特征处理
def online_coupon_features(data_user, data, end_day, windows_list, FLAG):
    print('online_coupon 特征处理开始')
    data = data.copy()
    data_user = data_user.copy()
    data['Discount_type_1'] = data['Discount_rate'].map(lambda x: 1 if ':' in x else 0)
    data['Discount_type_0'] = data['Discount_rate'].map(lambda x: 0 if ':' in x else 1)    
    data['Discount_max'] = data['Discount_rate'].map(get_max)
    data['Discount_min'] = data['Discount_rate'].map(get_min)
    data['Discount_rate'] = data['Discount_rate'].map(get_rate)
    data['flag'] = (data['Date'].isnull()).astype(int)
    end_ = get_date(*end_day)
    print(end_)
    for Action in ['ALL','1','2']:
        for day in windows_list: 
            begin_ = get_date_delta(*end_day,day)
            if Action == 'ALL':
                data_temp = data[(data['Date_received']>=begin_)&(data['Date_received']<=end_)]
            else:
                data_temp = data[(data['Date_received']>=begin_)&(data['Date_received']<=end_)&(data['Action']==Action)]
            for col in ['User_id','Merchant_id','Coupon_id']:
#                 print('领券统计')
                data_user = feat_count(data_user,data_temp,[col],'flag',name=FLAG+Action+col+'_q1_cnt_'+str(day))
#                 print('用券统计')
                data_user = feat_count(data_user,data_temp[data_temp['Date'].isnull()],[col],'flag',name=FLAG+Action+col+'_q2_cnt_'+str(day))
#                 print('未用券统计')
                data_user = feat_count(data_user,data_temp[~data_temp['Date'].isnull()],[col],'flag',name=FLAG+Action+col+'_q3_cnt_'+str(day))       
#                 print('用券率')
                data_user = feat_mean(data_user,data_temp,[col],'flag',name=FLAG+Action+col+'_flag_mean_'+str(day))
#                 print('用券方差')
                data_user = feat_std(data_user,data_temp,[col],'flag',name=FLAG+Action+col+'_flag_std_'+str(day))         
#                 print('券类型1')
                data_user = feat_sum(data_user,data_temp,[col],'Discount_type_1',name=FLAG+Action+col+'_dt1_sum_'+str(day)) 
#                 print('券类型0')
                data_user = feat_sum(data_user,data_temp,[col],'Discount_type_1',name=FLAG+Action+col+'_dt0_sum_'+str(day)) 
                for col2 in ['Discount_max','Discount_min','Discount_rate']:
#                     print('最大')
                    data_user = feat_max(data_user,data_temp,[col],col2,name=FLAG+Action+col+'_'+col2+'_max_'+str(day))
#                     print('最小')
                    data_user = feat_min(data_user,data_temp,[col],col2,name=FLAG+Action+col+'_'+col2+'_min_'+str(day))               
#                     print('平均')
                    data_user = feat_mean(data_user,data_temp,[col],col2,name=FLAG+Action+col+'_'+col2+'_mean_'+str(day))                
#                     print('方差')
                    data_user = feat_std(data_user,data_temp,[col],col2,name=FLAG+Action+col+'_'+col2+'_std_'+str(day)) 
            for col in ['Merchant_id','Coupon_id','Discount_rate','Date_received','Date']:
                data_user = feat_nunique(data_user, data_temp, ['User_id'], col, name=FLAG+Action+col+"_CU_uq_"+str(day))
            for col in ['User_id','Coupon_id','Discount_rate','Date_received','Date']:
                data_user = feat_nunique(data_user, data_temp, ['Merchant_id'], col, name=FLAG+Action+col+"_CM_uq_"+str(day))
            for col in ['User_id','Merchant_id','Discount_rate','Date_received','Date']:
                data_user = feat_nunique(data_user, data_temp, ['Coupon_id'], col, name=FLAG+Action+col+"_CC_uq_"+str(day))  
    print('online_coupon 特征处理开始')
    return data_user

# online_no_coupon 数据：特征处理
def online_no_coupon_features(data_user, data, end_day, windows_list, FLAG):
    print('online_no_coupon 特征处理开始')
    data = data.copy()
    data_user = data_user.copy()
    end_ = get_date(*end_day)
    print(end_)
    for Action in ['ALL','0','1']:
        for day in windows_list: 
            begin_ = get_date_delta(*end_day,day)
            if Action == 'ALL':
                data_temp = data[(data['Date_received']>=begin_)&(data['Date_received']<=end_)]
            else:
                data_temp = data[(data['Date_received']>=begin_)&(data['Date_received']<=end_)&(data['Action']==Action)]
            
            for col in ['User_id','Merchant_id','Coupon_id']:
#                 print('购买统计')
                data_user = feat_count(data_user,data_temp,[col],'Date',name=FLAG+Action+'_q1_cnt_'+str(day))
#                 print('具体统计')
            for col in ['Merchant_id','Action','Date']:
                data_user = feat_nunique(data_user, data_temp, ['User_id'], col, name=FLAG+Action+col+"_U_uq_"+str(day))
            for col in ['User_id','Action','Date']:
                data_user = feat_nunique(data_user, data_temp, ['Merchant_id'], col, name=FLAG+Action+col+"_M_uq_"+str(day))
    print('online_no_coupon 特征处理结束')
    return data_user

# 候选集用户 特征提取：穿越特征
def data_features(data_user, data, end_day, windows_list, FLAG):
    print('穿越 特征处理开始')  
    data = data.copy()
    data_user = data_user.copy()
    data['Discount_type_1'] = data['Discount_rate'].map(lambda x: 1 if ':' in x else 0)
    data['Discount_type_0'] = data['Discount_rate'].map(lambda x: 0 if ':' in x else 1)    
    data['Discount_max'] = data['Discount_rate'].map(get_max)
    data['Discount_min'] = data['Discount_rate'].map(get_min)
    data['Discount_rate'] = data['Discount_rate'].map(get_rate)
    data['Distance'] = data['Distance'].fillna(0).astype(int)
    data['End_day'] = (datetime(*end_day)-pd.to_datetime(data['Date_received'])).dt.days
    data['End_day_rank1'] = data.groupby(['User_id'])['Date_received'].rank(ascending=0)
    data['End_day_rank2'] = data.groupby(['User_id'])['Date_received'].rank(ascending=1)
    data['week'] = pd.to_datetime(data['Date_received']).dt.weekday
    data_f = data[['User_id','Merchant_id','Coupon_id','Date_received','Discount_type_1','Discount_type_0','Discount_max',\
                  'Discount_min','Discount_rate','Distance','End_day','End_day_rank1','End_day_rank2','week']].\
                    drop_duplicates(['User_id','Coupon_id','Date_received'])
    end_ = get_date(*end_day)
    print(end_)
    for day in windows_list: 
        begin_ = get_date_delta(*end_day,day)
        data_temp = data[(data['Date_received']>=begin_)&(data['Date_received']<=end_)]
        
        for col in ['User_id']:
#             print('领券统计')
            data_user = feat_count(data_user,data_temp,[col],'Date_received',name=FLAG+col+'_q1_cnt_'+str(day))
#             print('券类型1')
            data_user = feat_sum(data_user,data_temp,[col],'Discount_type_1',name=FLAG+col+'_dt1_sum_'+str(day)) 
#             print('券类型0')
            data_user = feat_sum(data_user,data_temp,[col],'Discount_type_1',name=FLAG+col+'_dt0_sum_'+str(day)) 
            for col2 in ['Discount_max','Discount_min','Discount_rate','Distance']:
#                 print('最大')
                data_user = feat_max(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_max_'+str(day))
#                 print('最小')
                data_user = feat_min(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_min_'+str(day))               
#                 print('平均')
                data_user = feat_mean(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_mean_'+str(day))                
#                 print('方差')
                data_user = feat_std(data_user,data_temp,[col],col2,name=FLAG+col+'_'+col2+'_std_'+str(day)) 
        for col in ['Merchant_id','Coupon_id','Discount_rate','Distance','Date_received']:
            data_user = feat_nunique(data_user, data_temp, ['User_id'], col, name=FLAG+col+"_U_uq_"+str(day))
    data_user = data_user.merge(data_f,on=['User_id','Merchant_id','Coupon_id','Date_received'],how='left')
    print('穿越 特征处理结束') 
    return data_user

# 提取线下offline_coupon表一组特征
# train1 end_day = '2016,04,30'
# train2 end_day = '2016,04,15'
# test end_day = '2016,06,15'
cookly_train_f1_1=offline_coupon_features(cookly_train_user_label1, cookly_offline_train_notnull_coupon, end_day=(2016,4,30), windows_list=[66], FLAG='offline_coupon_')
cookly_train_f1_2=offline_coupon_features(cookly_train_user_label2, cookly_offline_train_notnull_coupon, end_day=(2016,4,15), windows_list=[66], FLAG='offline_coupon_')
cookly_test_f1=offline_coupon_features(cookly_test_user_label, cookly_offline_train_notnull_coupon, end_day=(2016,6,15), windows_list=[66], FLAG='offline_coupon_')

# 提取线下consumption一组特征
# train1 end_day = '2016,05,15'
# train2 end_day = '2016,04,30'
# test end_day = '2016,06,30'
cookly_train_f2_1=offline_consumption_features(cookly_train_user_label1, cookly_offline_train_null_coupon, end_day=(2016,5,15), windows_list=[66], FLAG='offline_consumption_')
cookly_train_f2_2=offline_consumption_features(cookly_train_user_label2, cookly_offline_train_null_coupon, end_day=(2016,4,30), windows_list=[66], FLAG='offline_consumption_')
cookly_test_f2=offline_consumption_features(cookly_test_user_label, cookly_offline_train_null_coupon, end_day=(2016,6,30), windows_list=[66], FLAG='offline_consumption_')

# 提取线上online_coupon一组特征
# train1 end_day = '2016,04,30'
# train2 end_day = '2016,04,15'
# test end_day = '2016,06,15'
cookly_train_f3_1=online_coupon_features(cookly_train_user_label1, cookly_online_train_notnull_coupon, end_day=(2016,4,30), windows_list=[66], FLAG='online_coupon_')
cookly_train_f3_2=online_coupon_features(cookly_train_user_label2, cookly_online_train_notnull_coupon, end_day=(2016,4,15), windows_list=[66], FLAG='online_coupon_')
cookly_test_f3=online_coupon_features(cookly_test_user_label, cookly_online_train_notnull_coupon, end_day=(2016,6,15), windows_list=[66], FLAG='online_coupon_')

# 提取线上online_no_coupon一组特征
# train1 end_day = '2016,05,15'
# train2 end_day = '2016,04,30'
# test end_day = '2016,06,30'
cookly_train_f4_1=online_no_coupon_features(cookly_train_user_label1, cookly_online_train_null_coupon, end_day=(2016,5,15), windows_list=[66], FLAG='online_no_coupon_')
cookly_train_f4_2=online_no_coupon_features(cookly_train_user_label2, cookly_online_train_null_coupon, end_day=(2016,4,30), windows_list=[66], FLAG='online_no_coupon_')
cookly_test_f4=online_no_coupon_features(cookly_test_user_label, cookly_online_train_null_coupon, end_day=(2016,6,30), windows_list=[66], FLAG='online_no_coupon_')

# 穿越特征提取表
cookly_data_train_label1 = cookly_offline_train_notnull_coupon[(cookly_offline_train_notnull_coupon['Date_received']>='20160516')\
                                         &(cookly_offline_train_notnull_coupon['Date_received']<='20160615')]
cookly_data_train_label2 = cookly_offline_train_notnull_coupon[(cookly_offline_train_notnull_coupon['Date_received']>='20160501')\
                                         &(cookly_offline_train_notnull_coupon['Date_received']<='20160531')]
cookly_data_test_label = cookly_offline_test_coupon

# 穿越特征提取
# 穿越原因：无法使用用户当天之后的数据
# train1 end_day = '2016,06,15'
# train2 end_day = '2016,05,31'
# test end_day = '2016,07,31'
cookly_train_f5_1=data_features(cookly_train_user_label1, cookly_data_train_label1, end_day=(2016,6,15), windows_list=[66], FLAG='now_')
cookly_train_f5_2=data_features(cookly_train_user_label2, cookly_data_train_label2, end_day=(2016,5,31), windows_list=[66], FLAG='now_')
cookly_test_f5=data_features(cookly_test_user_label, cookly_data_test_label, end_day=(2016,7,31), windows_list=[66], FLAG='now_')

# 封装的Stacking和Bagging
# 本模型在银杏大数据竞赛中拿了第一名
# 在jdata比赛baseline中开源
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
 
class SBBTree():
	"""Stacking,Bootstap,Bagging----SBBTree"""
	""" author：Cookly 洪鹏飞 """
	def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round, early_stopping_rounds):
		"""
		Initializes the SBBTree.
        Args:
          params : lgb params.
          stacking_num : k_flod stacking.
          bagging_num : bootstrap num.
          bagging_test_size : bootstrap sample rate.
          num_boost_round : boost num.
		  early_stopping_rounds : early_stopping_rounds.
        """
		self.params = params
		self.stacking_num = stacking_num
		self.bagging_num = bagging_num
		self.bagging_test_size = bagging_test_size
		self.num_boost_round = num_boost_round
		self.early_stopping_rounds = early_stopping_rounds
 
		self.model = lgb
		self.stacking_model = []
		self.bagging_model = []
 
	def fit(self, X, y):
		""" fit model. """
		if self.stacking_num > 1:
			layer_train = np.zeros((X.shape[0], 2))
			self.SK = StratifiedKFold(n_splits=self.stacking_num, shuffle=True, random_state=1)
			for k,(train_index, test_index) in enumerate(self.SK.split(X, y)):
				X_train = X[train_index]
				y_train = y[train_index]
				X_test = X[test_index]
				y_test = y[test_index]
 
				lgb_train = lgb.Dataset(X_train, y_train)
				lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
 
				gbm = lgb.train(self.params,
							lgb_train,
							num_boost_round=self.num_boost_round,
							valid_sets=lgb_eval,
							early_stopping_rounds=self.early_stopping_rounds)
 
				self.stacking_model.append(gbm)
 
				pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
				layer_train[test_index, 1] = pred_y
 
			X = np.hstack((X, layer_train[:,1].reshape((-1,1)))) 
		else:
			pass
		for bn in range(self.bagging_num):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)
	
			lgb_train = lgb.Dataset(X_train, y_train)
			lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
 
			gbm = lgb.train(self.params,
						lgb_train,
						num_boost_round=10000,
						valid_sets=lgb_eval,
						early_stopping_rounds=200)
 
			self.bagging_model.append(gbm)
		
	def predict(self, X_pred):
		""" predict test data. """
		if self.stacking_num > 1:
			test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
			for sn,gbm in enumerate(self.stacking_model):
				pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
				test_pred[:, sn] = pred
			X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1,1))))  
		else:
			pass 
		for bn,gbm in enumerate(self.bagging_model):
			pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
			if bn == 0:
				pred_out=pred
			else:
				pred_out+=pred
		return pred_out/self.bagging_num

# 导入包
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 2 ** 5 - 1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': .7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'scale_pos_weight': 25,
    'seed': 2018,
    'nthread': 4,
    'verbose': 0,
}

# 所有特征 带穿越 线下测试
train_join_columns = ['User_id','Merchant_id','Coupon_id','Date_received','label']
test_join_columns = ['User_id','Merchant_id','Coupon_id','Date_received']
# 合并特征表
train_data_1 = cookly_train_f1_1.merge(cookly_train_f2_1,on=train_join_columns).\
                merge(cookly_train_f3_1,on=train_join_columns).\
                merge(cookly_train_f4_1,on=train_join_columns).\
                merge(cookly_train_f5_1,on=train_join_columns)
train_data_2 = cookly_train_f1_2.merge(cookly_train_f2_2,on=train_join_columns).\
                merge(cookly_train_f3_2,on=train_join_columns).\
                merge(cookly_train_f4_2,on=train_join_columns).\
                merge(cookly_train_f5_2,on=train_join_columns)
test_data = cookly_test_f1.merge(cookly_test_f2,on=test_join_columns).\
                merge(cookly_test_f3,on=test_join_columns).\
                merge(cookly_test_f4,on=test_join_columns).\
                merge(cookly_test_f5,on=test_join_columns)

# 获取特征表
train_columns = [col for col in train_data_1.columns if col not in ['User_id','Date_received','label']]
X1_error = train_data_1[train_columns].values
y1_error = train_data_1['label'].values
X2_error = train_data_2[train_columns].values
y2_error = train_data_2['label'].values
X_pred_error = test_data[train_columns].values

# 使用模型
model_error = SBBTree(params=params, stacking_num=1, bagging_num=1,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model_error.fit(X1_error,y1_error)
y2_out_error = model_error.predict(X2_error)

# coupon平均auc计算
def get_coupon_auc(coupon_auc):
    coupon_auc = feat_nunique(coupon_auc, coupon_auc, ['Coupon_id'], 'y_true', name='flag')
    coupon_auc = coupon_auc[coupon_auc['flag']>1]
 
    coupon_list = set(coupon_auc.Coupon_id)
    coupon_auc_list = []
    for coupon in coupon_list:
        coupon_df = coupon_auc[coupon_auc['Coupon_id']==coupon]
        y_pred = coupon_df['y_pred']
        y_true = coupon_df['y_true']
        auc_ = roc_auc_score(y_true,y_pred)
        coupon_auc_list.append(auc_)
    return np.mean(coupon_auc_list)

test = train_data_2[['Coupon_id']]
test['y_true'] = y2_error
test['y_pred'] = y2_out_error

# 计算模型预测：总体AUC，coupon平均AUC
print('带穿越特征：模型线下评测')
print('总体AUC：',roc_auc_score(y2_error,y2_out_error))
print('coupon平均AUC：',get_coupon_auc(test))

# 预测提交
# 加穿越特征 预测
sample2_y = model_error.predict(X_pred_error)
submit_sample2 = test_data[['User_id','Coupon_id','Date_received']].copy()
submit_sample2['Probability'] = sample2_y
submit_sample2.drop_duplicates(['User_id','Coupon_id','Date_received']).to_csv('sample_test2_error.csv',header=False,index=False)

# 一组特征 不穿越 线下测试
train_columns = [col for col in cookly_train_f1_1.columns if col not in ['User_id','Date_received','label']]
X = cookly_train_f1_1[train_columns].values
y = cookly_train_f1_1['label'].values
X_pred = cookly_train_f1_2[train_columns].values
y_pred = cookly_train_f1_2['label'].values

# 模型训练和预测
model2 = SBBTree(params=params, stacking_num=1, bagging_num=1,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model2.fit(X,y)
y_pred_out = model2.predict(X_pred)

test = cookly_train_f1_2[['Coupon_id']]
test['y_true'] = y_pred
test['y_pred'] = y_pred_out

# 计算模型预测：总体AUC，coupon平均AUC
print('一组特征 不穿越：模型线下评测')
print('总体AUC：',roc_auc_score(y2_error,y2_out_error))
print('coupon平均AUC：',get_coupon_auc(test))

# 不带穿越的 所有特征
train_join_columns = ['User_id','Merchant_id','Coupon_id','Date_received','label']
test_join_columns = ['User_id','Merchant_id','Coupon_id','Date_received']
# 合并特征
train_data_1_no = cookly_train_f1_1.merge(cookly_train_f2_1,on=train_join_columns).\
                merge(cookly_train_f3_1,on=train_join_columns).\
                merge(cookly_train_f4_1,on=train_join_columns)
train_data_2_no = cookly_train_f1_2.merge(cookly_train_f2_2,on=train_join_columns).\
                merge(cookly_train_f3_2,on=train_join_columns).\
                merge(cookly_train_f4_2,on=train_join_columns)
test_data_no = cookly_test_f1.merge(cookly_test_f2,on=test_join_columns).\
                merge(cookly_test_f3,on=test_join_columns).\
                merge(cookly_test_f4,on=test_join_columns)

train_columns = [col for col in train_data_1_no.columns if col not in ['User_id','Date_received','label']]
X1 = train_data_1_no[train_columns].values
y1 = train_data_1_no['label'].values
X2 = train_data_2_no[train_columns].values
y2 = train_data_2_no['label'].values
X_pred = test_data_no[train_columns].values

model = SBBTree(params=params, stacking_num=1, bagging_num=1,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model.fit(X1,y1)
y2_out = model.predict(X2)

test = train_data_2[['Coupon_id']]
test['y_true'] = y2
test['y_pred'] = y2_out

# 计算模型预测：总体AUC，coupon平均AUC
print('素有特征 不穿越：模型线下评测')
print('总体AUC：',roc_auc_score(y2_error,y2_out_error))
print('coupon平均AUC：',get_coupon_auc(test))