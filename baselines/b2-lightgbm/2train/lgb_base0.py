
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import logging

data_path = '../../data'
#train1 = pd.read_csv(os.path.join(data_path, 'invite_info_0926.txt'), header=None, sep='\t')
test1 = pd.read_csv(os.path.join(data_path, 'invite_info_evaluate_1_0926.txt'), header=None, sep='\t')
data = pd.read_hdf('data.h5',key='data')

'''修改'''
user_answer_feats_val = pd.read_hdf('../user_answer_val.h5',key='data')
user_answer_feats_train = pd.read_hdf('../user_answer_train.h5',key='data')
user_answer_feats_train = user_answer_feats_train[['用户id','u_a_i_diffhour_mean','u_a_i_diffhour_sum',
                                                   'u_a_i_diffhour_max','u_a_i_diffhour_min','u_a_i_diffday_mean',
                                                   'u_a_i_diffday_sum','u_a_i_diffday_max','u_a_i_diffday_min',
                                                   '用户习惯回答时间-hour','用户最近回答数-3天','用户最近回答数-7天','用户最近回答数-14天']]
user_answer_feats_val = user_answer_feats_val[['用户id','u_a_i_diffhour_mean','u_a_i_diffhour_sum',
                                                   'u_a_i_diffhour_max','u_a_i_diffhour_min','u_a_i_diffday_mean',
                                                   'u_a_i_diffday_sum','u_a_i_diffday_max','u_a_i_diffday_min',
                                                   '用户习惯回答时间-hour','用户最近回答数-3天','用户最近回答数-7天','用户最近回答数-14天']]



user_topic_feats = pd.read_hdf('../user_topic_feat.h5',key='data')
print("loaded topic feats")
user_question_topic = pd.read_hdf('../member_question_feat.h5', key='data')

#user_qa_days_feats_train = pd.read_hdf('../u_qa_day_feats_train.h5',key='data')
#user_qa_days_feats_val = pd.read_hdf('../u_qa_day_feats_val.h5',key='data')
#print("loaded u_qa_days")
'''修改end'''

print(data.head(3))

'''mmm'''
feature_cols = [x for x in data.columns if x not in ('label', 'dt', 'day','q_inv_mean', 'q_inv_sum', 'q_inv_std', 'q_inv_count')]
'''end'''

print(feature_cols)

X_train_all = data.iloc[:2593669][feature_cols]
y_train_all = data.iloc[:2593669]['label']
test = data.iloc[2593669:]

'''修改'''
X_train_all = pd.merge(X_train_all, user_answer_feats_train, left_on='uid', right_on='用户id',how='left').drop('用户id',axis=1)
X_train_all = pd.merge(X_train_all, user_topic_feats, left_on='uid', right_on='author_id',how='left').drop('author_id',axis=1)
#X_train_all = pd.merge(X_train_all, user_qa_days_feats_train, left_on='uid', right_on='用户id',how='left')
X_train_all = pd.merge(X_train_all, user_question_topic, left_on=['uid','qid'], right_on=['author_id','question_id'], how='left').drop(['author_id','question_id'],axis=1)
drop_feat = ['uid', 'qid']
X_train_all = X_train_all.drop(drop_feat, axis=1)


test = pd.merge(test, user_answer_feats_val, left_on='uid', right_on='用户id',how='left').drop('用户id',axis=1)
test = pd.merge(test, user_topic_feats, left_on='uid', right_on='author_id',how='left').drop('author_id',axis=1)
#test = pd.merge(test, user_qa_days_feats_val, left_on='uid', right_on='用户id',how='left')
test = pd.merge(test, user_question_topic, left_on=['uid','qid'], right_on=['author_id','question_id'], how='left').drop(['author_id','question_id'],axis=1)

drop_feat = ['uid', 'qid']
test = test.drop(drop_feat, axis=1)

feature_cols = [x for x in feature_cols if x not in ['uid', 'qid']]
'''修改end'''

del data
assert len(test) == test1.shape[0]

logging.info("train shape %d, test shape %s", 2593669, test.shape)

fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    break

X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols], \
                                 y_train_all.iloc[train_idx], \
                                 y_train_all.iloc[val_idx]
del X_train_all

model_lgb = LGBMClassifier(n_estimators=2000, n_jobs=-1, objective='binary', seed=1000, silent=True)
model_lgb.fit(X_train, y_train,
              eval_metric=['logloss', 'auc'],
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50)


sub = test1.copy()
sub_size = len(sub)
sub['label'] = model_lgb.predict_proba(test[feature_cols])[:, 1]


sub.to_csv('result.txt', index=None, header=None, sep='\t')

pd.set_option('display.max_rows', None)
print(pd.DataFrame({
    'column': feature_cols,
    'importance': model_lgb.feature_importances_
}).sort_values(by='importance', ascending=False))

model_lgb.booster_.save_model('model.txt')