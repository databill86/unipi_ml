"""
var0 - binary, on/off
var1 - real, changes very frequently
var2 - real, constant
var3 - real, constant
var4 - real, constant
var5 - real, constant
var6 - real, changes
var7 - real, changes
var8 - real, changes
var9 - real, changes
var10 - binary, partly correlated to var6
var11 - binary, partly correlated to var7
var12 - binary, partly correlated to var8
var13 - binary, partly correlated to var9
var14 - integer, days working
var15 - real, constant
var16 - binary, first day after maintainance works
var17 - binary, some hardware working (usualy turned off during all maintainance works)
var18 - binary, some hardware working (usualy turned off during maintainance works #1,2,5)
var19 - integer, number of days since last maintainance works
var20 - integer, number of days since last maintainance works #1
var21 - binary, some hardware working (usualy turned off during maintainance works #6)
var22 - real, constant
var23 - real, changes very rarely
Date 
var25 - categorical, [0-2]
ID
var27 - categorical, [0-13], type of maintainance work
target - real, debit
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import seaborn as sns
import datetime
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV


# correcting errors in train data (where var16==1 and var27==6, but target==0)
def correct_var16_var27(df, date_from, date_to, ID):
    df.loc[(df.Date==date_from) & (df.ID==ID), 'var16'] = 0
    df.loc[(df.Date==date_to) & (df.ID==ID), 'var16'] = 1
    df.loc[(df.Date==date_to) & (df.ID==ID), 'var27'] = 6

# calculate Mean Average Precision Error
def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# do validation on 5-fold
def mape_cv(model, df, features):
    kf = KFold(5, shuffle=True, random_state=82).get_n_splits(df[features].values)
    pred = np.expm1(cross_val_predict(model, df[features].values, np.log1p(df['target'].values), cv=kf))
    return(mape(df['target'].values, pred))

# averaging helper
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

# stacking helper
class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


###############################################################################
################################## MAIN #######################################
###############################################################################
print('loading data...')
df = pd.read_csv("./input/dataset_train.csv", parse_dates=['Date'])

# correcting errors in train data (where var16==1 and var27==6, but target==0)
correct_var16_var27(df, datetime.date(2014,11,20), datetime.date(2014,11,22), 116)
correct_var16_var27(df, datetime.date(2015,1,8), datetime.date(2015,1,10), 158)
correct_var16_var27(df, datetime.date(2014,10,22), datetime.date(2014,10,24), 380)
correct_var16_var27(df, datetime.date(2016,1,17), datetime.date(2016,1,18), 689)
correct_var16_var27(df, datetime.date(2016,1,21), datetime.date(2016,1,25), 766)
correct_var16_var27(df, datetime.date(2014,12,19), datetime.date(2014,12,26), 819)
correct_var16_var27(df, datetime.date(2014,12,6), datetime.date(2014,12,8), 968)
correct_var16_var27(df, datetime.date(2016,5,7), datetime.date(2016,5,9), 1057)

# join train and test datasets to build features on them
test_df = pd.read_csv("./input/dataset_test.csv", parse_dates=['Date'])
df = df.append(test_df)

# delete maintenance work periods to have easy access to pre-maintenance data
df = df[(df.target>0) | (df.var16==1)]    

# create features
df['log_var1'] = np.log1p(df['var1'])
df['log_var1_1'] = df.groupby(['ID']).log_var1.shift(1)
df['log_var1_2'] = df.groupby(['ID']).log_var1.shift(2).fillna(df['log_var1_1'])
df['log_var1_3'] = df.groupby(['ID']).log_var1.shift(3).fillna(df['log_var1_2'])
df['log_var1_4'] = df.groupby(['ID']).log_var1.shift(4).fillna(df['log_var1_3'])
df['log_var1_5'] = df.groupby(['ID']).log_var1.shift(5).fillna(df['log_var1_4'])
df['log_var1_6'] = df.groupby(['ID']).log_var1.shift(6).fillna(df['log_var1_5'])
df['log_var1_7'] = df.groupby(['ID']).log_var1.shift(7).fillna(df['log_var1_6'])
df['log_var1_8'] = df.groupby(['ID']).log_var1.shift(8).fillna(df['log_var1_7'])
df['log_var1_9'] = df.groupby(['ID']).log_var1.shift(9).fillna(df['log_var1_8'])
df['log_var1_10'] = df.groupby(['ID']).log_var1.shift(10).fillna(df['log_var1_9'])

df['var1_1'] = df.groupby(['ID']).var1.shift(1)
df['var1_2'] = df.groupby(['ID']).var1.shift(2).fillna(df['var1_1'])
df['var1_3'] = df.groupby(['ID']).var1.shift(3).fillna(df['var1_2'])
df['var1_4'] = df.groupby(['ID']).var1.shift(4).fillna(df['var1_3'])
df['var1_5'] = df.groupby(['ID']).var1.shift(5).fillna(df['var1_4'])
df['var1_6'] = df.groupby(['ID']).var1.shift(6).fillna(df['var1_5'])
df['var1_7'] = df.groupby(['ID']).var1.shift(7).fillna(df['var1_6'])
df['var1_8'] = df.groupby(['ID']).var1.shift(8).fillna(df['var1_7'])
df['var1_9'] = df.groupby(['ID']).var1.shift(9).fillna(df['var1_8'])
df['var1_10'] = df.groupby(['ID']).var1.shift(10).fillna(df['var1_9'])

df['var1_max'] = df[['var1_1', 'var1_2', 'var1_3', 'var1_4', 'var1_5', 
                     'var1_6', 'var1_7', 'var1_8', 'var1_9', 'var1_10']].max(axis=1)
df['var1_min'] = df[['var1_1', 'var1_2', 'var1_3', 'var1_4', 'var1_5', 
                     'var1_6', 'var1_7', 'var1_8', 'var1_9', 'var1_10']].min(axis=1)
df['var1_median'] = df[['var1_1', 'var1_2', 'var1_3', 'var1_4', 'var1_5', 
                        'var1_6', 'var1_7', 'var1_8', 'var1_9', 'var1_10']].median(axis=1)
df['var1_std'] = df[['var1_1', 'var1_2', 'var1_3', 'var1_4', 'var1_5', 
                     'var1_6', 'var1_7', 'var1_8', 'var1_9', 'var1_10']].std(axis=1)
df['var1_mean'] = df[['var1_1', 'var1_2', 'var1_3', 'var1_4', 'var1_5', 
                      'var1_6', 'var1_7', 'var1_8', 'var1_9', 'var1_10']].mean(axis=1)
df['var1_var'] = df[['var1_1', 'var1_2', 'var1_3', 'var1_4', 'var1_5', 
                     'var1_6', 'var1_7', 'var1_8', 'var1_9', 'var1_10']].var(axis=1)

df['log_var1_max'] = df[['log_var1_1', 'log_var1_2', 'log_var1_3', 'log_var1_4', 'log_var1_5', 
                       'log_var1_6', 'log_var1_7', 'log_var1_8', 'log_var1_9', 'log_var1_10']].max(axis=1)
df['log_var1_min'] = df[['log_var1_1', 'log_var1_2', 'log_var1_3', 'log_var1_4', 'log_var1_5', 
                       'log_var1_6', 'log_var1_7', 'log_var1_8', 'log_var1_9', 'log_var1_10']].min(axis=1)
df['log_var1_median'] = df[['log_var1_1', 'log_var1_2', 'log_var1_3', 'log_var1_4', 'log_var1_5', 
                          'log_var1_6', 'log_var1_7', 'log_var1_8', 'log_var1_9', 'log_var1_10']].median(axis=1)
df['log_var1_std'] = df[['log_var1_1', 'log_var1_2', 'log_var1_3', 'log_var1_4', 'log_var1_5', 
                       'log_var1_6', 'log_var1_7', 'log_var1_8', 'log_var1_9', 'log_var1_10']].std(axis=1)
df['log_var1_mean'] = df[['log_var1_1', 'log_var1_2', 'log_var1_3', 'log_var1_4', 'log_var1_5', 
                        'log_var1_6', 'log_var1_7', 'log_var1_8', 'log_var1_9', 'log_var1_10']].mean(axis=1)
df['log_var1_var'] = df[['log_var1_1', 'log_var1_2', 'log_var1_3', 'log_var1_4', 'log_var1_5', 
                       'log_var1_6', 'log_var1_7', 'log_var1_8', 'log_var1_9', 'log_var1_10']].var(axis=1)

df['var1/var1_1'] = df['var1'] / df['var1_1']
df['var1/var1_min'] = df['var1'] / df['var1_min']
df['var1-var1_min'] = df['var1'] - df['var1_min']
df['var1-var1_1'] = df['var1'] - df['var1_1']

df['log_var1/log_var1_1'] = df['log_var1'] / df['log_var1_1']
df['log_var1/log_var1_min'] = df['log_var1'] / df['log_var1_min']
df['log_var1-log_var1_min'] = df['log_var1'] - df['log_var1_min']

df['target_1'] = df.groupby(['ID']).target.shift(1)
df['target_2'] = df.groupby(['ID']).target.shift(2).fillna(df['target_1'])
df['target_3'] = df.groupby(['ID']).target.shift(3).fillna(df['target_2'])
df['target_4'] = df.groupby(['ID']).target.shift(4).fillna(df['target_3'])
df['target_5'] = df.groupby(['ID']).target.shift(5).fillna(df['target_4'])

df['log_target'] = np.log1p(df['target'])
df['log_target_1'] = df.groupby(['ID']).log_target.shift(1)
df['log_target_2'] = df.groupby(['ID']).log_target.shift(2).fillna(df['log_target_1'])
df['log_target_3'] = df.groupby(['ID']).log_target.shift(3).fillna(df['log_target_2'])
df['log_target_4'] = df.groupby(['ID']).log_target.shift(4).fillna(df['log_target_3'])
df['log_target_5'] = df.groupby(['ID']).log_target.shift(5).fillna(df['log_target_4'])
df['log_target_6'] = df.groupby(['ID']).log_target.shift(6).fillna(df['log_target_5'])
df['log_target_7'] = df.groupby(['ID']).log_target.shift(7).fillna(df['log_target_6'])
df['log_target_8'] = df.groupby(['ID']).log_target.shift(8).fillna(df['log_target_7'])
df['log_target_9'] = df.groupby(['ID']).log_target.shift(9).fillna(df['log_target_8'])
df['log_target_10'] = df.groupby(['ID']).log_target.shift(10).fillna(df['log_target_9'])
df['log_target_11'] = df.groupby(['ID']).log_target.shift(11).fillna(df['log_target_10'])
df['log_target_12'] = df.groupby(['ID']).log_target.shift(12).fillna(df['log_target_11'])
df['log_target_13'] = df.groupby(['ID']).log_target.shift(13).fillna(df['log_target_12'])
df['log_target_14'] = df.groupby(['ID']).log_target.shift(14).fillna(df['log_target_13'])
df['log_target_15'] = df.groupby(['ID']).log_target.shift(15).fillna(df['log_target_14'])

df['log_target_max'] = df[['log_target_1', 'log_target_2', 'log_target_3', 'log_target_4', 'log_target_5', 
                           'log_target_6', 'log_target_7', 'log_target_8', 'log_target_9', 'log_target_10']].max(axis=1)
df['log_target_min'] = df[['log_target_1', 'log_target_2', 'log_target_3', 'log_target_4', 'log_target_5', 
                           'log_target_6', 'log_target_7', 'log_target_8', 'log_target_9', 'log_target_10']].min(axis=1)
df['log_target_median'] = df[['log_target_1', 'log_target_2', 'log_target_3', 'log_target_4', 'log_target_5', 
                              'log_target_6', 'log_target_7', 'log_target_8', 'log_target_9', 'log_target_10']].median(axis=1)
df['log_target_std'] = df[['log_target_1', 'log_target_2', 'log_target_3', 'log_target_4', 'log_target_5', 
                           'log_target_6', 'log_target_7', 'log_target_8', 'log_target_9', 'log_target_10']].std(axis=1)
df['log_target_mean'] = df[['log_target_1', 'log_target_2', 'log_target_3', 'log_target_4', 'log_target_5', 
                            'log_target_6', 'log_target_7', 'log_target_8', 'log_target_9', 'log_target_10']].mean(axis=1)
df['log_target_var'] = df[['log_target_1', 'log_target_2', 'log_target_3', 'log_target_4', 'log_target_5', 
                           'log_target_6', 'log_target_7', 'log_target_8', 'log_target_9', 'log_target_10']].var(axis=1)

df['log_var2'] = np.log1p(df['var2'])
df['log_var3'] = np.log1p(df['var3'])
df['log_var4'] = np.log1p(df['var4'])
df['log_var5'] = np.log1p(df['var5'])

df['log_var6'] = np.log1p(df['var6'])
df['log_var7'] = np.log1p(df['var7'])
df['log_var8'] = np.log1p(df['var8'])
df['log_var9'] = np.log1p(df['var9'])

df['log_var15'] = np.log1p(df['var15'])
df['log_var20'] = np.log1p(df['var20'])
df['log_var22'] = np.log1p(df['var22'])
df['log_var23'] = np.log1p(df['var23'])

df['month'] = df['Date'].dt.month
df = pd.get_dummies(data = df, columns=['month'], prefix='month' )

features = ['var1', 'var1_1', 'var1_2', 'var1_3', 'var1_4', 'var1_5',
            #'var1_6', 'var1_7', 'var1_8', 'var1_9', 'var1_10',
            'var1_min', #'var1_median', #'var1_var',
            #'var1_std', 
            #'var1_mean',
            #'log_var1', 'log_var1_1', 'log_var1_2', 'log_var1_3', 'log_var1_4', 'log_var1_5',
            'log_target_1', 'log_target_2', 'log_target_3', 'log_target_4', 'log_target_5', 
            'log_target_6', 'log_target_7', 'log_target_8', 'log_target_9', 'log_target_10', 
            'log_target_11', 'log_target_12', 'log_target_13', 'log_target_14', 'log_target_15',
            'log_target_min', 'log_target_median', #'log_target_var',
            #'log_target_std', 'log_target_mean',
            #'log_var2', 'log_var3', 'log_var4', 'log_var5',
            'var2', 'var3', #'var4', 'var5'
            #'var6', 'var7', 'var8', 'var9',
            #'log_var6', 'log_var7', 'log_var8', 'log_var9',
            #'var10', 'var11', 'var12', 'var13'
            #'log_var15'
            'log_var20',
            #'log_var22'
            #'log_var23'
            #'var25'
            'month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12',
            #'target_1','target_2','target_3','target_4','target_5'
            'var1/var1_1', 'var1/var1_min', 'log_var1/log_var1_1',
            #'log_var1/log_var1_min', 'log_var1-log_var1_min', 'var1-var1_1'
            
            ]

# delete all except points of interest, take only points where we have at lest some history (target_1 is notnull)
events = df[(df.var16==1) & (df.var27==6) & (df.target_1.notnull())]

df_train = events[events.Date < datetime.date(2017,1,1)]
df_test = events[events.Date >= datetime.date(2017,1,1)]

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1, tol=0.01))
print("Lasso score: {:.4f}\n".format(mape_cv(lasso, df_train, features)))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.99, random_state=3, tol=0.01, max_iter=2000))
print("ElasticNet score: {:.4f}\n".format(mape_cv(ENet, df_train, features)))

KRR = KernelRidge(alpha=60, kernel='polynomial', degree=2, coef0=2.5)
print("Kernel Ridge score: {:.4f}\n".format(mape_cv(KRR, df_train, features)))

sgdr = make_pipeline(RobustScaler(), SGDRegressor(penalty = 'l2', random_state = 82, tol=0.00001, max_iter=10000, alpha=0.05, loss='huber'))
print("sgdr score: {:.4f}\n".format(mape_cv(sgdr, df_train, features)))

model_lgb = lgb.LGBMRegressor(objective='regression', 
                              num_leaves=3,
                              min_data_in_leaf = 6, 
                              max_bin = 255, 
                              learning_rate=0.01, 
                              n_estimators=1000,
                              bagging_fraction = 0.8,
                              bagging_freq = 5, 
                              bagging_seed=0,
                              feature_fraction = 0.2319,
                              feature_fraction_seed=0, 
                              )
print("LGBM score: {:.4f}\n".format(mape_cv(model_lgb, df_train, features)))

averaged_model = AveragingModels(models = (lasso, ENet, KRR, model_lgb, sgdr))
print("Averaged score: {:.4f} \n".format(mape_cv(averaged_model, df_train, features)))

metaKRR = KernelRidge(alpha=0.1, kernel='polynomial', degree=2, coef0=2.5)
stacked_model = StackingModels(base_models = (lasso, ENet, KRR, model_lgb, sgdr),
                               meta_model = metaKRR
                               )
print("Stacked score: {:.4f} \n".format(mape_cv(stacked_model, df_train, features)))

# fit on hole train data
sub_model = stacked_model
sub_model.fit(df_train[features].values, np.log1p(df_train['target'].values))

# predict on train data
stacked_train_pred = np.expm1(sub_model.predict(df_train[features].values))
#df_train.loc[:,'predicted'] = stacked_train_pred;
print(mape(df_train['target'].values, stacked_train_pred))

# predict on test data
stacked_pred = np.expm1(sub_model.predict(df_test[features].values))

# save submission
sub = pd.DataFrame()
sub['Id'] = df_test['ID'].values
sub['target'] = stacked_pred
sub.to_csv("./output/"+os.path.basename(__file__).replace('.py','.csv'),index=False)


























if 0:
    kfold = KFold(n_splits=5, shuffle=True, random_state=156)
    for train_index, holdout_index in kfold.split(df_train):
        xgtrain = lgb.Dataset(df_train.iloc[train_index][features].values, 
                              label=df_train.iloc[train_index]['log_target'].values,
                              feature_name=features,
                              #categorical_feature=categorical
                              )
        xgvalid = lgb.Dataset(df_train.iloc[holdout_index][features].values, 
                              label=df_train.iloc[holdout_index]['log_target'].values,
                              feature_name=features,
                              #categorical_feature=categorical
                              )
        lgb_params = {
            'boosting': 'gbdt',
            'objective': 'regression_l2',
            'metric':'l2',
            'num_leaves':3,
            'min_data_in_leaf' : 6, 
            'max_bin' : 255, 
            'learning_rate':0.01, 
            'n_estimators':2000,
            'bagging_fraction' : 0.8,
            'bagging_freq' : 5, 
            'bagging_seed':0,
            'feature_fraction' : 0.2319,
            'feature_fraction_seed':0, 
            }
        
        evals_results = {}
        bst = lgb.train(lgb_params, 
                         xgtrain, 
                         valid_sets=[xgtrain, xgvalid], 
                         evals_result=evals_results, 
                         early_stopping_rounds=100,
                         verbose_eval=0, 
                         feval=None)
       
        print('Plot metrics during training...')
        ax = lgb.plot_metric(evals_results, metric='l2')
        plt.show()
        
        ax = lgb.plot_importance(bst, max_num_features=100)
        plt.show()
        
        gain = bst.feature_importance('gain')
        ft = pd.DataFrame({'feature':bst.feature_name(), 'split':bst.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('split', ascending=False)
        print(ft.head(100))
    
        stacked_train_pred = np.expm1(bst.predict(df_train[features].values))
        print(mape(df_train['target'].values, stacked_train_pred))



if 0:

    sgdr = make_pipeline(RobustScaler(), SGDRegressor(penalty = 'l2', random_state = 82, tol=0.00001, max_iter=10000, alpha=0.05, loss='huber'))
    print("sgdr score: {:.4f}\n".format(mape_cv(sgdr, df_train, features)))

    coef0_arr = np.array([10,100,150,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,5000,10000,20000,30000])    
    alpha_arr = np.array([10,100,200,300,1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000,25000,30000,40000,50000])    
    model = make_pipeline(RobustScaler(), KernelRidge(kernel='polynomial', degree=2))
    grid = GridSearchCV(estimator=model, param_grid=dict(coef0=coef0_arr, alpha=alpha_arr))
    grid.fit(df_train[features].values, np.log1p(df_train['target'].values))
    # summarize the results of the grid search
    print(grid.best_score_)
    print(grid.best_estimator_.coef0)
    print(grid.best_estimator_.alpha)

    KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=50000, kernel='polynomial', degree=2, coef0=30000))
    print("Kernel Ridge score: {:.4f}\n".format(mape_cv(KRR, df_train, features)))



if 0:
    events_test = test_df[(test_df.var16==1) & (test_df.var27==6)]
    
    events_test = events[events.target == 0]
    
    sns.distplot(df_train['var22']);
    sns.distplot(np.log1p(df_train['var22']));
   
    var = 'var7'
    data = pd.concat([df['target'], df[var]], axis=1)
    data.plot.scatter(x=var, y='target');
    
    
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);
    
    k = 20 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'target')['target'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    
    sns.set()
    cols = ['target', 'target_1']
    sns.pairplot(df_train[cols], size = 2.5)
    plt.show();