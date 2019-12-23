
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import logging
from sklearn.utils import shuffle

log_fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
logging.basicConfig(format=log_fmt, level=logging.INFO)

import warnings
warnings.filterwarnings('ignore')


def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))


base_path = '../../data'

# 加载邀请回答数据

train = pd.read_csv(f'{base_path}/invite_info_0926.txt', sep='\t', header=None)
train.columns = ['qid', 'uid', 'dt', 'label']
logging.info("invite %s", train.shape)

test = pd.read_csv(f'{base_path}/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
test.columns = ['qid', 'uid', 'dt']
logging.info("test %s", test.shape)

sub = test.copy()

sub_size = len(sub)

train['day'] = extract_day(train['dt'])
train['hour'] = extract_hour(train['dt'])

test['day'] = extract_day(test['dt'])
test['hour'] = extract_hour(test['dt'])

del train['dt'], test['dt']

# 加载问题
ques = pd.read_csv(f'{base_path}/question_info_0926.txt', header=None, sep='\t')
ques.columns = ['qid', 'q_dt', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']
del ques['title_t1'], ques['title_t2'], ques['desc_t1'], ques['desc_t2']
logging.info("ques %s", ques.shape)

ques['q_day'] = extract_day(ques['q_dt'])
ques['q_hour'] = extract_hour(ques['q_dt'])
del ques['q_dt']

# 加载回答
ans = pd.read_csv(f'{base_path}/answer_info_0926.txt', header=None, sep='\t')
ans.columns = ['aid', 'qid', 'uid', 'ans_dt', 'ans_t1', 'ans_t2', 'is_good', 'is_rec', 'is_dest', 'has_img',
               'has_video', 'word_count', 'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
               'reci_xxx', 'reci_no_help', 'reci_dis']
del ans['ans_t1'], ans['ans_t2']
logging.info("ans %s", ans.shape)

ans['a_day'] = extract_day(ans['ans_dt'])
ans['a_hour'] = extract_hour(ans['ans_dt'])
del ans['ans_dt']

ans = pd.merge(ans, ques, on='qid')
#del ques

# 回答距提问的天数
ans['diff_qa_days'] = ans['a_day'] - ans['q_day']

# 时间窗口划分
# train
# val
train_start = 3838
train_end = 3867

val_start = 3868
val_end = 3874

label_end = 3867
label_start = label_end - 6

train_label_feature_end = label_end - 7
train_label_feature_start = train_label_feature_end - 22

train_ans_feature_end = label_end - 7
train_ans_feature_start = train_ans_feature_end - 50

val_label_feature_end = val_start - 1
val_label_feature_start = val_label_feature_end - 22

val_ans_feature_end = val_start - 1
val_ans_feature_start = val_ans_feature_end - 50

train_label_feature = train[(train['day'] >= train_label_feature_start) & (train['day'] <= train_label_feature_end)]
logging.info("train_label_feature %s", train_label_feature.shape)

val_label_feature = train[(train['day'] >= val_label_feature_start) & (train['day'] <= val_label_feature_end)]
logging.info("val_label_feature %s", val_label_feature.shape)

'''mmm'''

train_label = train[(train['day'] > train_label_feature_end)]
train_label['dropoutrate'] = train_label['label'].apply(lambda x: 1.0 if x==1 else np.random.rand())
train_label = train_label[train_label['dropoutrate']>0.5]
train_label = train_label.drop(['dropoutrate'],axis=1)

#train_label = shuffle(pd.concat([train_all_pos, train_all_neg], axis=0, sort=True)) #shuffle
#print("shuffled")
#del train_all_pos, train_all_neg
'''mmm end'''

print("train_label_len: ",len(train_label))

logging.info("train feature start %s end %s, label start %s end %s", train_label_feature['day'].min(),
             train_label_feature['day'].max(), train_label['day'].min(), train_label['day'].max())

logging.info("test feature start %s end %s, label start %s end %s", val_label_feature['day'].min(),
             val_label_feature['day'].max(), test['day'].min(), test['day'].max())

# 确定ans的时间范围
# 3807~3874
train_ans_feature = ans[(ans['a_day'] >= train_ans_feature_start) & (ans['a_day'] <= train_ans_feature_end)]

val_ans_feature = ans[(ans['a_day'] >= val_ans_feature_start) & (ans['a_day'] <= val_ans_feature_end)]

logging.info("train ans feature %s, start %s end %s", train_ans_feature.shape, train_ans_feature['a_day'].min(),
             train_ans_feature['a_day'].max())

logging.info("val ans feature %s, start %s end %s", val_ans_feature.shape, val_ans_feature['a_day'].min(),
             val_ans_feature['a_day'].max())

fea_cols = ['is_good', 'is_rec', 'is_dest', 'has_img', 'has_video', 'word_count',
            'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
            'reci_xxx', 'reci_no_help', 'reci_dis', 'diff_qa_days']

'''
mmm
'''
def extract_feature1(target, label_feature, ans_feature, end_day):
    # 问题特征
    t1 = label_feature.groupby('qid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['qid', 'q_inv_mean', 'q_inv_sum', 'q_inv_std', 'q_inv_count']
    target = pd.merge(target, t1, on='qid', how='left')

    # 用户特征
    t1 = label_feature.groupby('uid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['uid', 'u_inv_mean', 'u_inv_sum', 'u_inv_std', 'u_inv_count']
    target = pd.merge(target, t1, on='uid', how='left')
    #
    # train_size = len(train)
    # data = pd.concat((train, test), sort=True)

    # 回答部分特征

    t1 = ans_feature.groupby('qid')['aid'].count().reset_index()
    t1.columns = ['qid', 'q_ans_count']
    target = pd.merge(target, t1, on='qid', how='left')

    t1 = ans_feature.groupby('uid')['aid'].count().reset_index()
    t1.columns = ['uid', 'u_ans_count']
    target = pd.merge(target, t1, on='uid', how='left')

    for col in fea_cols:
        t1 = ans_feature.groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['uid', f'u_{col}_sum', f'u_{col}_max', f'u_{col}_mean']
        target = pd.merge(target, t1, on='uid', how='left')
        del t1

        ''' mmm '''
        # t1 = ans_feature[ans_feature['a_day']>end_day-3].groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        # t1.columns = ['uid', f'u_{col}_sum_last3day', f'u_{col}_max_last3day', f'u_{col}_mean_last3day']
        # target = pd.merge(target, t1, on='uid', how='left')
        # del t1
        #
        # t1 = ans_feature[ans_feature['a_day']>end_day-7].groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        # t1.columns = ['uid', f'u_{col}_sum_last7day', f'u_{col}_max_last7day', f'u_{col}_mean_last7day']
        # target = pd.merge(target, t1, on='uid', how='left')
        # del t1
        #
        # t1 = ans_feature[ans_feature['a_day']>end_day-14].groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        # t1.columns = ['uid', f'u_{col}_sum_last14day', f'u_{col}_max_last14day', f'u_{col}_mean_last14day']
        # target = pd.merge(target, t1, on='uid', how='left')
        # del t1
        ''' mmm end'''

        t1 = ans_feature.groupby('qid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['qid', f'q_{col}_sum', f'q_{col}_max', f'q_{col}_mean']
        target = pd.merge(target, t1, on='qid', how='left')
        logging.info("extract %s", col)

    target['wk'] = target['day'] % 7

    return target

train_label_feature['wk'] = train_label_feature['day'] %7
train_label = extract_feature1(train_label, train_label_feature, train_ans_feature, train_ans_feature_end)
val_label_feature['wk'] = val_label_feature['day'] %7
test = extract_feature1(test, val_label_feature, val_ans_feature, val_ans_feature_end)

# 特征提取结束
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)
assert len(test) == sub_size

# 加载用户
user = pd.read_csv(f'{base_path}/member_info_0926.txt', header=None, sep='\t')
user.columns = ['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq', 'uf_b1', 'uf_b2',
                'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score', 'follow_topic',
                'inter_topic']
del user['follow_topic'], user['inter_topic']
logging.info("user %s", user.shape)

unq = user.nunique()
logging.info("user unq %s", unq)

for x in unq[unq == 1].index:
    del user[x]
    logging.info('del unq==1 %s', x)

t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in ['follow_topic', 'inter_topic', 'uid']]
logging.info("user cat %s", cats)

for d in cats:
    lb = LabelEncoder()
    user[d] = lb.fit_transform(user[d])
    logging.info('encode %s', d)

q_lb = LabelEncoder()
q_lb.fit(list(ques['qid'].astype(str).values))
train_label_feature['qid_enc'] = q_lb.transform(train_label_feature['qid'])
train_label['qid_enc'] = q_lb.transform(train_label['qid'])

test['qid_enc'] = q_lb.transform(test['qid'])
val_label_feature['qid_enc'] = q_lb.transform(val_label_feature['qid'])

u_lb = LabelEncoder()
u_lb.fit(user['uid'])
train_label_feature['uid_enc'] = u_lb.transform(train_label_feature['uid'])
train_label['uid_enc'] = u_lb.transform(train_label['uid'])
val_label_feature['uid_enc'] = u_lb.transform(val_label_feature['uid'])
test['uid_enc'] = u_lb.transform(test['uid'])

# merge user
train_label_feature = pd.merge(train_label_feature, user, on='uid', how='left')
train_label = pd.merge(train_label, user, on='uid', how='left')
val_label_feature = pd.merge(val_label_feature, user, on='uid', how='left')
test = pd.merge(test, user, on='uid', how='left')
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)


data = pd.concat((train_label, test), axis=0, sort=True)
del train_label, test

'''mmm'''
# count编码
# def creat_count_features(data):
#     count_cols = []
#     count_fea = ['uid_enc', 'qid_enc', 'freq', 'hour', 'uf_c1', 'uf_c3', 'uf_c4','wk']
#     for feat in count_fea:
#         col_name = '{}_count'.format(feat)
#         count_cols.append(col_name)
#         data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
#         data.loc[data[col_name] < 2, feat] = -1
#         data[feat] += 1
#         data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
#         data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())
#     return data[count_fea+count_cols], count_fea
#
# count_train, count_fea = creat_count_features(train_label_feature)
# train_label = pd.merge(train_label, count_train, on=count_fea, how='left')
# count_val, count_fea = creat_count_features(val_label_feature)
# val_label = pd.merge(test, count_val, on=count_fea, how='left')

# count编码evalua
def basic_count1(data, count_fea):
    for feat in count_fea:
        col_name = '{}_count'.format(feat)
        data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
        data.loc[data[col_name] < 2, feat] = -1
        data[feat] += 1
        data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
        data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())
    return data

data = basic_count1(data, ['gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5'])
data = basic_count1(data, ['uid_enc', 'qid_enc'])

'''mmm end'''
# 问题被回答的次数

# 压缩数据
t = data.dtypes
for x in t[t == 'int64'].index:
    data[x] = data[x].astype('int32')

for x in t[t == 'float64'].index:
    data[x] = data[x].astype('float32')

#data['wk'] = data['day'] % 7

feature_cols = [x for x in data.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]
# target编码
logging.info("feature size %s", len(feature_cols))

print(data.head(10))
print(data.info())
print(data.columns.values)

data.to_hdf('data_dropout.h5',key='data')

import random
random.random()