#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scanpy as sc

from copy import deepcopy
import warnings
import sys

import os
import os.path
import shutil
import sklearn
from optparse import OptionParser
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression

from sklearn.inspection import permutation_importance

from scipy import stats
from scipy.stats import multivariate_normal, skew, pearsonr, spearmanr
import scipy.sparse
import statsmodels.api as sm
import shap

from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

import utility_1
import h5py
import pickle

class _Base2_train1(BaseEstimator):
	"""
	Parameters
	----------

	"""

	def __init__(self,peak_read=[],
					rna_exprs=[],
					rna_exprs_unscaled=[],
					df_gene_peak_query=[],
					df_gene_annot_expr=[],
					motif_data = [],
					data_dir = '',
					normalize=0,
					copy_X=True,
					n_jobs=None,
					positive=False,
					fit_intercept=True,
					select_config={}):

		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.copy_X = copy_X
		self.n_jobs = n_jobs
		self.positive = positive

		if (len(rna_exprs)>0):
			sample_id = rna_exprs.index
			if (len(peak_read)>0):
				peak_read = peak_read.loc[sample_id,:]
			if len(rna_exprs_unscaled)>0:
				rna_exprs_unscaled = rna_exprs_unscaled.loc[sample_id,:]

		self.peak_read = peak_read
		self.rna_exprs = rna_exprs
		self.rna_exprs_unscaled= rna_exprs_unscaled
		self.df_gene_peak_query = df_gene_peak_query
		self.df_gene_annot_expr = df_gene_annot_expr
		self.motif_data = motif_data
		self.data_dir = data_dir
		self.save_file_path = data_dir
		self.select_config = select_config

		self.train_mode_cv = 0
		self.gene_motif_prior_1 = []
		self.gene_motif_prior_2 = []
		self.dict_feature_query_ = dict()
		self.pre_model_dict = dict()

	## ====================================================
	# prediction performance evaluation for predicting continous values
	def score_function(self, y, y_predicted):

		"""
		prediction performance evaluation for predicting continous values
		:param y: (array) true values of the varible
		:param y_predicted (array) predicted values of the variable
		:return: (pandas.Series) prediction performance evaluated using multiple metrics, including:
								 mean squared error, Pearson correlation and p-value, explained variance, mean absolute error, median absoluate error, R2 score, Spearman's rank correlation and p-vlaue, mutual information;
		"""

		score1 = mean_squared_error(y, y_predicted)
		score2 = pearsonr(y, y_predicted)
		score3 = explained_variance_score(y, y_predicted)
		score4 = mean_absolute_error(y, y_predicted)
		score5 = median_absolute_error(y, y_predicted)
		score6 = r2_score(y, y_predicted)
		score7, pvalue = spearmanr(y,y_predicted)
		t_mutual_info = mutual_info_regression(y[:,np.newaxis], y_predicted, discrete_features=False, n_neighbors=5, copy=True, random_state=0)
		t_mutual_info = t_mutual_info[0]

		vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6, score7, pvalue, t_mutual_info]

		field_query_1 = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2','mutual_info']
		df_score_pred = pd.Series(index=field_query_1,data=vec1,dtype=np.float32)
			
		return df_score_pred

	## ====================================================
	# prediction performance evaluation for binary classification model
	def score_function_multiclass1(self,y_test, y_pred, y_proba):

		"""
		prediction performance evaluation for binary classification model
		:param y_test: (array) true labels of the variable
		:param y_pred: (array) predicted labels of the variable
		:param y_proba: (array) predicted positive label probabilities of the variable
		:return: (pandas.Series) prediction performance evaluated using multiple metrics, including:
		 						 accuracy, AUROC, AUPR, precision, recall and F1 score;
		"""

		auc = roc_auc_score(y_test,y_proba)
		aupr = average_precision_score(y_test,y_proba)
		precision = precision_score(y_test,y_pred)
		recall = recall_score(y_test,y_pred)
		accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))
		F1 = 2*precision*recall/(precision+recall)

		return accuracy, auc, aupr, precision, recall, F1

	## ====================================================
	# prediction performance evaluation for classification model
	def score_function_multiclass2(self,y_test, y_pred, y_proba, average='macro'):

		"""
		prediction performance evaluation for classification model
		:param y_test: (array) true labels of the variable
		:param y_pred: (array) predicted labels of the variable
		:param y_proba: (array) predicted probabilities of assignment to each class of the variable
		:param average: (str) the type of averaging performed on the data to calcualte the evaluation metric scores
		:return: (pandas.Series) prediction performance evaluated using multiple metrics, including:
		 						 accuracy, precision, recall and F1 score;
		"""

		# auc = roc_auc_score(y_test,y_proba,average=average)
		# aupr = average_precision_score(y_test,y_proba,average=average)
		precision = precision_score(y_test,y_pred,average=average)
		recall = recall_score(y_test,y_pred,average=average)
		accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))
		eps=1E-12
		F1 = 2*precision*recall/(precision+recall+eps)

		vec1 = [accuracy, precision, recall, F1]
		field_query_1 = ['accuracy','precision','recall','F1']
		df_score_pred = pd.Series(index=field_query_1,data=vec1,dtype=np.float32)
		
		return df_score_pred

	## ====================================================
	# initiation of the prediction model
	def test_model_basic_pre1(self,model_type_id=0,select_config={}):

		"""
		initiation of the prediction model
		:param model_type_id: (str) the prediction model name (e.g., LogisticRegression)
		:param select_config: dictionary containing parameters
		:return: the initiated model
		"""

		flag_positive_coef=False
		if 'flag_positive_coef' in select_config:
			flag_positive_coef = select_config['flag_positive_coef']

		fit_intercept = True
		if 'fit_intercept' in select_config:
			fit_intercept = select_config['fit_intercept']

		if model_type_id in ['LR']:
			if 'normalize_type_LR' in select_config:
				normalize_type = select_config['normalize_type_LR']
			else:
				normalize_type = False

			# model_1 = LinearRegression(fit_intercept=True, normalize=normalize_type, copy_X=True, n_jobs=None, max_iter=5000)
			model_1 = LinearRegression(fit_intercept=fit_intercept, copy_X=True, n_jobs=None, positive=flag_positive_coef)

		elif model_type_id in ['Lasso','Lasso_ori']:
			# Lasso
			alpha = select_config['Lasso_alpha']
			# warm_start_type = False
			warm_start_type = select_config['warm_start_type_Lasso']
			normalize_type=False

			selection_type = 'cyclic'
			if 'Lasso_selection_type1' in select_config:
				selection_type = select_config['Lasso_selection_type1']
			
			model_1 = Lasso(alpha=alpha,fit_intercept=fit_intercept, precompute=False, copy_X=True, max_iter=5000, 
							tol=0.0001, warm_start=warm_start_type, positive=flag_positive_coef, random_state=None, selection=selection_type)
		
		elif model_type_id in ['ElasticNet']:
			# ElasticNet
			alpha = select_config['ElasticNet_alpha']
			warm_start_type = False
			normalize_type=False

			if 'warm_start_type_ElasticNet' in select_config:
				warm_start_type = select_config['warm_start_type_ElasticNet']
			
			selection_type = 'cyclic'
			if 'ElasticNet_selection_type1' in select_config:
				selection_type = select_config['ElasticNet_selection_type1']

			l1_ratio = 0.05
			if 'l1_ratio' in select_config:
				l1_ratio = select_config['l1_ratio_ElasticNet']

			model_1 = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,fit_intercept=fit_intercept,normalize='deprecated',precompute=False,max_iter=5000,
									copy_X=True,tol=0.0001,warm_start=warm_start_type,positive=flag_positive_coef,random_state=None,selection=selection_type)

		elif model_type_id in ['Ridge']:
			alpha = 1.0
			# flag_positive_coef = False
			# fit_intercept = True
			if 'Ridge_alpha' in select_config:
				alpha = select_config['Ridge_alpha']

			model_1 = Ridge(alpha=alpha, fit_intercept=fit_intercept, copy_X=True, max_iter=5000, tol=0.0001, solver='auto', positive=flag_positive_coef, random_state=None)

		elif model_type_id in ['LogisticRegression']:
			# GTB
			multi_class='auto'
			if ('num_class' in select_config):
				num_class = select_config['num_class']
				if num_class>1:
					column_1 = 'multi_class_logisticregression'
					if column_1 in select_config:
						multi_class = select_config[column_1]
			print('multi_class_logisticregression: ',multi_class)

			model_1 = LogisticRegression(penalty='l2',
											dual=False, 
											tol=0.0001, 
											C=1.0, 
											fit_intercept=True, 
											intercept_scaling=1, 
											class_weight=None, 
											random_state=None, 
											solver='lbfgs', 
											max_iter=1000, 
											multi_class=multi_class, 
											verbose=0, 
											warm_start=False, 
											n_jobs=None, 
											l1_ratio=None)
			
		# tree-based model
		elif model_type_id in ['XGBR','XGBClassifier','RF','RandomForestClassifier']:

			if 'select_config_comp' in select_config:
				select_config_comp = select_config['select_config_comp']
				max_depth, n_estimators = select_config_comp['max_depth'], select_config_comp['n_estimators']
			else:
				# max_depth, n_estimators = 10, 200
				max_depth, n_estimators = 7, 100
			# print('max_depth, n_estimators ',max_depth,n_estimators)

			if model_type_id in ['XGBClassifier']:
				import xgboost
				# GTB
				type_id1 = 1
				if 'type_classifer_xbgboost' in select_config:
					type_id1 = select_config['type_classifer_xbgboost']
				objective_function_vec = ['binary:logistic','multi:softprob','multi:softmax']
				objective_function_1 = objective_function_vec[type_id1]
				model_1 = xgboost.XGBClassifier(colsample_bytree=1,
												 use_label_encoder=False,
												 gamma=0,
												 n_jobs=10,
												 learning_rate=0.1,
												 max_depth=max_depth,
												 min_child_weight=1,
												 n_estimators=n_estimators,                                                                    
												 reg_alpha=0,
												 reg_lambda=0.1,
												 objective=objective_function_1,
												 subsample=1,
												 random_state=0)

			elif model_type_id in ['XGBR']:
				import xgboost
				# GTB
				model_1 = xgboost.XGBRegressor(colsample_bytree=1,
												 gamma=0,    
												 n_jobs=10,             
												 learning_rate=0.1,
												 max_depth=max_depth,
												 min_child_weight=1,
												 n_estimators=n_estimators,                                                                    
												 reg_alpha=0,
												 reg_lambda=1,
												 objective='reg:squarederror',
												 subsample=1,
												 random_state=0)
			elif model_type_id in ['RF']:
				# random forest
				model_1 = RandomForestRegressor(n_jobs=10,
												n_estimators=n_estimators,
												max_depth=max_depth,
												random_state=0)

			else:
				n_estimators = 500
				model_1 = RandomForestClassifier(n_estimators=n_estimators,
													max_depth=max_depth,
													n_jobs=10,
													random_state=0)
		else:
			print('please specify the model type')
			return

		return model_1

	## ====================================================
	# perform model training
	def test_model_train_basic_pre1(self,model_train,model_type_id,x_train,y_train,sample_weight=[]):

		"""
		perform model training
		:param model_train: the model to train
		:param model_type_id: (str) name of the model 
		:param x_train: (dataframe or numpy.array) feature matrix of the observations (row:observation, column:predictor variable)
		:param y_train: (pandas.Series or numpy.array) response variable values of the observations
		:param sample_weight: (array) sample weight of the training and test samples
		return: 1. the model trained using the training data; the model trained using the training data;
				2. list including the array of estimated regression coeffiecients of the predictor variables 
				   and the learned intercept if regression model is used;
		"""

		if model_type_id in ['XGBR','XGBClassifier','RF']:
			if len(sample_weight)==0:
				# print('model training ')
				model_train.fit(x_train, y_train)
			else:
				print('sample weight maximal and minimal values ',np.max(sample_weight),np.min(sample_weight))
				model_train.fit(x_train,y_train,sample_weight=sample_weight)

			t_coefficient = []
			t_intercept = []

		elif model_type_id in ['LR_2']:
			model_1 = sm.OLS(y_train,x_train)
			model_train = model_1.fit()

			df_2 = model_train.summary2().tables[1]
			column_1, column_2 = 'Coef.','P>|t|'

			df_query1 = df_2.loc[:,[column_1,column_2]]
			df_query1 = df_query1.rename(columns={column_1:'coef',column_2:'pval'})
			query_idvec = df_query1.index

			if ('const' in query_idvec):
				feature_name_vec = query_idvec.difference(['const'],sort=False)
				t_intercept = df_query1.loc['const','coef']
			else:
				feature_name_vec = query_idvec
				t_intercept = 0

			t_coefficient = df_query1.loc[feature_name_vec,'coef']
		else:
			# print('model training ')
			model_train.fit(x_train, y_train)
			t_coefficient = model_train.coef_
			t_intercept = model_train.intercept_

		return model_train, [t_coefficient,t_intercept]

	## ====================================================
	# query if paramters are within specific range
	def _check_params(self,params,upper_bound,lower_bound):

		"""
		:param params: (array) the parameters
		:param upper_bound: (float) upper bound on the parameters
		:param lower_bound: (float) lower bound on the parameters
		:return: indicator of whether the paramters are within the range specified by the lower and upper bounds 
				 (True: within the range; False:there are parameters out of the range)
		"""

		small_eps = 1e-3
		min1, max1 = lower_bound, upper_bound
		flag_1 = (params>=min1-small_eps)&(params<=max1+small_eps)
		flag1 = (np.sum(flag_1)==len(params))
		if flag1==False:
			print(params)
			flag_1 = np.asarray(flag_1)
			id1 = np.where(flag_1==0)[0]
			print(params[id1], len(id1), len(params))

		return flag1

	## ====================================================
	# query estimated regression coefficients of the predictor variables
	def test_query_ceof_1(self,param,feature_name,num_class,response_variable_name='1',query_idvec=[],df_coef_query=[],select_config={}):

		"""
		query estimated coefficient values
		:param param: (array) the parameters learned by the model
		:param feature_name: (array) names of the predictor variables
		:param num_class: (int) the number of classes
		:param response_variable_name: (str) name of the response variable
		:param query_idvec: (list) name of the predictor variables and the intercept
		:param df_coef_query: dataframe to save the estimated regression coefficients of the predictor variables and the intercept of the trained model for the corresponding repsonse variable
		:param select_config: dictionary containing configuration parameters
		:return: dataframe containing the estimated regression coefficients of the predictor variables and the intercept of the trained model for the corresponding repsonse variable
		"""

		alpha_query1 = param[0]
		intercept_ = param[1]
		if len(alpha_query1)>0:
			if len(query_idvec)==0:
				feature_name = pd.Index(feature_name).difference(['const'],sort=False)
				feature_query_vec_coef = feature_name
				query_idvec = list(feature_query_vec_coef)+['alpha0']

			if num_class<=2:	
				alpha_query = np.hstack((alpha_query1,intercept_[:,np.newaxis]))
				alpha_query = alpha_query.T
				if len(df_coef_query)==0:
					df_coef_query = pd.DataFrame(index=query_idvec,columns=[response_variable_name],data=np.asarray(alpha_query),dtype=np.float32)
				else:
					df_coef_query.loc[query_idvec,response_variable_name] = alpha_query
			else:
				alpha_query = np.hstack((alpha_query1,intercept_[:,np.newaxis]))
				alpha_query = alpha_query.T
				df_coef_query = pd.DataFrame(index=query_idvec,columns=np.arange(num_class),data=np.asarray(alpha_query),dtype=np.float32)

			return df_coef_query

	## ====================================================
	# query estimated regression coefficients and p-values of the predictor variables
	def test_query_coef_pval_1(self,model_train,model_type_id=0,select_config={}):

		"""
		model training for peak-TF association prediction
		query estimated regression coefficients and the p-values of the predictor variables
		:param model_train: the trained model
		:param model_type_id: if model_type_id=1, the model estimates p-values for the coefficients of the predictor variables
		:param select_config: dictionary containing parameters
		:return: dataframe with two columns representing the estimated regression coefficients and p-values of the predictor variables; 
				 the intercept of the trained model is also included along with the regression coefficients;
		"""

		if model_type_id==0:
			df1 = model_train.summary2().tables[1]
			column_1, column_2 = 'Coef.','P>|t|'
			df_query1 = df1.loc[:,[column_1,column_2]]
			df_query1 = df_query1.rename(columns={column_1:'coef',column_2:'pval'})
			column_vec_1 = ['coef','pval']

			query_idvec_1 = df_query1.index
			feature_name_vec = query_idvec_1.difference(['const'],sort=False)
			query_idvec = list(feature_name_vec)+['alpha0']
			if 'const' in query_idvec_1:
				query_idvec_2 = list(feature_name_vec)+['const']
				intercept_query = df_query1.loc['const','coef']
				query_id_1 = query_idvec
			else:
				query_idvec_2 = feature_name_vec
				query_id_1 = feature_name_vec
			
			df_coef_pval_ = pd.DataFrame(index=query_idvec,columns=column_vec_1,data=0)
			df_coef_pval_.loc[query_id_1,column_vec_1] = np.asarray(df_query1.loc[query_idvec_2,column_vec_1])
			
			return df_coef_pval_

	## ====================================================
	# use trained model for prediction and estimate feature importance scores of the predictor variables
	def test_model_pred_explain_1(self,model_train,x_test,y_test,sample_id_test_query=[],y_pred=[],y_pred_proba=[],
										x_train=[],y_train=[],response_variable_name='',df_coef_query=[],df_coef_pval_=[],
										fold_id=-1,type_id_model=0,model_explain=1,model_save_filename='',
										output_mode=1,save_mode=0,verbose=0,select_config={}):

		"""
		using trained model for prediction and estimate feature importance scores of the predictor variables
		:param model_train: the trained model
		:param x_test: (dataframe or numpy.array) feature matrix of the test samples (row:observation, column:predictor variable)
		:param y_test: (pandas.Series or numpy.array) response variable values of test samples
		:param sample_id_test_query: (array) the test sample names or identifiers
		:param y_pred: (dataframe) predicted class labels or response variable values of the samples
		:param y_pred_proba: (dataframe) predicted positive label probabilities of the samples
		:param x_train: (dataframe or numpy.array) feature matrix of the training samples (row:observation, column:predictor variable)
		:param y_train: (pandas.Series or numpy.array) response variable values of the training samples
		:param response_variable_name: (str) name of the response variable
		:param df_coef_query: (dataframe) the estimated regression coefficients of the predictor variables and the intercept learned by the model
		:param df_coef_pval_: (dataframe) the estimated regression coefficients and p-values of the predictor variables and the intercept learned by the model
		:param fold_id: (int) index of the fold with which the model is trained if cross-validation is used 
		:param type_id_model: (int) the prediction model type: 0: regression model; 1: classification model
		:param model_explain: indicator of whether to perform model interpretation to estimate feature importance scores of the predictor variables
		:param model_save_filename: (str) path of the file which saves the trained model
		:param output_mode: indicator of whether to save predicted class labels and positive class prababilites or response variable values of the test samples 
							in the prediction dataframes of the full samples
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (pandas.Series) prediction performance scores using different evaluation metrics;
				 2. (array) predicted class labels or response variable values of the test samples;
				 3. (array) predicted positive label probabilities of the test samples;
				 4. dictionary containing estimated original and scaled feature importance scores and 
				 	regression coefficients of the predictor variables and the intercept for the corresponding model type;
		"""

		y_test_pred = model_train.predict(x_test)
		y_test_proba = []
		if len(y_pred)==0:
			output_mode = 0
		if type_id_model==0:
			# regression model
			y_test_pred = np.ravel(y_test_pred)
			score_1 = self.score_function(y_test,y_test_pred)
		else:
			# classification model
			y_test_proba = model_train.predict_proba(x_test)
			select_config1 = select_config['select_config1']
			average_type = select_config1['average_type']
			if verbose>0:
				print('average_type: %s'%(average_type))
			score_1 = self.score_function_multiclass2(y_test,y_test_pred,y_test_proba,average=average_type)
			if output_mode>0:
				y_pred_proba.loc[sample_id_test_query,:] = y_test_proba

		if output_mode>0:
			y_pred.loc[sample_id_test_query] = y_test_pred
			# y_pred_proba.loc[sample_id_test_query,:] = y_test_proba

		flag_model_explain = model_explain
		dict_query1 = dict()
		df_imp_1, df_imp_scaled_1 = [], []
		feature_name=x_train.columns
		if flag_model_explain>0:
			feature_type_id = select_config['feature_type_id']
			model_type_id1 = select_config['model_type_id1']
			if verbose>0:
				print('model explain using feature %d for fold %d'%(feature_type_id,fold_id))
			model_type_id2 = '%s.feature%d'%(model_type_id1,feature_type_id)
			model_train_dict_1 = {model_type_id2:model_train}
			model_save_dict_1 = {model_type_id2:model_save_filename}
			model_path_1 = select_config['data_path_save']
			t_vec_1 = self.test_model_explain_basic_pre1(x_train,y_train,
															feature_name=feature_name,
															x_test=[],y_test=[],
															model_train_dict=model_train_dict_1,
															model_save_dict=model_save_dict_1,
															model_path_1=model_path_1,
															save_mode=save_mode,
															model_save_file=model_save_filename,
															select_config=select_config)

			dict_feature_imp_ = t_vec_1[0]
			dict_query1 = dict_feature_imp_[model_type_id2]
		
		model_type_name = select_config['model_type_id_train']
		print('train, model_type_name: ',model_type_name)
		
		if model_type_name in ['LR','Lasso','ElasticNet','LogisticRegression','LR_2']:
			if model_type_name in ['LR_2']:
				df_coef_pval_1 = self.test_query_coef_pval_1(model_train,model_type_id=0,select_config=select_config)
				query_idvec = df_coef_pval_1.index
				feature_name = feature_name.difference(['alpha0','const'],sort=False)
				feature_name_vec = query_idvec.difference(['alpha0','const'],sort=False)
				coef_query = df_coef_pval_1.loc[feature_name_vec,'coef']
				pval_query = df_coef_pval_1.loc[query_idvec,'pval']
				intercept_query = df_coef_pval_1.loc['alpha0','coef']
				if len(df_coef_pval_)==0:
					df_coef_pval_ = pd.DataFrame(index=query_idvec,columns=[response_variable_name],data=np.asarray(pval_query),dtype=np.float32)
				else:
					df_coef_pval_.loc[query_idvec,response_variable_name] = pval_query

				dict_query1.update({'pval':df_coef_pval_})
			else:
				coef_query, intercept_query = model_train.coef_, model_train.intercept_

			param_vec = [coef_query, intercept_query]
			if type_id_model==0:
				num_class = 1
			else:
				num_class = y_test_proba.shape[1]

			df_coef_query = self.test_query_ceof_1(param=param_vec,feature_name=feature_name,num_class=num_class,response_variable_name=response_variable_name,query_idvec=[],df_coef_query=df_coef_query,select_config=select_config)
			dict_query1.update({'coef':df_coef_query})
		
		return score_1, y_test_pred, y_test_proba, dict_query1

	## ====================================================
	# feature importance estimation from the trained model
	def test_model_explain_basic_pre1(self,x_train,y_train,feature_name,x_test=[],y_test=[],
										model_train_dict=[],model_save_dict=[],
										model_path_1='',save_mode=0,model_save_file='',select_config={}):

		"""
		feature importance estimation from the trained model
		:param x_train: (dataframe or numpy.array) feature matrix of the training samples (row:observation, column:predictor variable)
		:param y_train: (pandas.Series or numpy.array) response variable values of the training samples
		:param feature_name: (array) names of the predictor variables
		:param x_test: (dataframe or numpy.array) feature matrix of the test samples (row:observation, column:predictor variable)
		:param y_test: (pandas.Series or numpy.array) response variable values of test samples
		:param model_train_dict: dictionary containing the trained prediction models for the corresponding model types
		:param model_save_dict: dictionary containing paths of the files which saved the trained models for the corresponding model types
		:param model_path_1: (str) the directory to save the feature importance estimation file
		:param save_mode: indicator of whether to save data
		:param model_save_file: (str) path of the file which saved the trained model
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing the estimated original and scaled feature importance scores and 
					regression coefficients of predictor variables for the corresponding model type;
				 2. dictionary containing file paths of the original and scaled estimated feature importance scores of 
				 	predictor variables for the corresponding model type;
		"""

		model_type_idvec = list(model_save_dict.keys())
		model_type_num = len(model_type_idvec)

		list1 = [str1.split('.') for str1 in model_type_idvec]
		model_type_name_vec = np.asarray([t_vec1[0] for t_vec1 in list1])
		feature_type_vec = np.asarray([t_vec1[1] for t_vec1 in list1])
		annot_vec = np.asarray([t_vec1[-1] for t_vec1 in list1])

		if len(model_train_dict)>0:
			pre_load = 0
		else:
			pre_load = 1

		model_list2 = []
		dict_feature_imp_, filename_dict1 = dict(), dict() # feature importance estimate
		
		file_save_path = model_path_1
		for i1 in range(model_type_num):
			model_type_id, model_type_name, feature_type_id = model_type_idvec[i1], model_type_name_vec[i1], feature_type_vec[i1]
			model_save_file = ''
			if pre_load==0:
				model_train = model_train_dict[model_type_id]
				if model_type_id in model_save_dict:
					model_save_file = model_save_dict[model_type_id]
			else:
				model_save_file = model_save_dict[model_type_id]
				model_train = pickle.load(open(model_save_file, "rb"))

			df_imp_, df_imp_scaled, coef_query_1 = self.test_model_explain_pre2(model_train=model_train,
																				x_train=x_train,
																				y_train=y_train,
																				feature_name=feature_name,
																				model_type_name=model_type_name,
																				linear_type_id=0,
																				save_mode=save_mode,
																				select_config=select_config)

			dict_feature_imp_[model_type_id] = {'imp':df_imp_,'imp_scaled':df_imp_scaled,'coef_query':coef_query_1}
		
			if save_mode==1:
				if model_save_file=='':
					filename_save_annot = select_config['filename_save_annot']
					output_filename1 = '%s/test_query.model_%s.%s.imp.1.txt'%(file_save_path,model_type_id,filename_save_annot)
					output_filename2 = '%s/test_query.model_%s.%s.imp_scaled.1.txt'%(file_save_path,model_type_id,filename_save_annot)		
				else:
					b = model_save_file.find('.h5')
					output_filename1 = model_save_file[0:b]+'.imp.1.txt'
					output_filename2 = model_save_file[0:b]+'.imp_scaled.1.txt'

				df_imp_.to_csv(output_filename1,sep='\t',float_format='%.6E')
				# df_imp_scaled.to_csv(output_filename2,sep='\t',float_format='%.6E')
				intercept_ = 0
				if len(coef_query_1)>0:
					coef_, intercept_ = coef_query_1
				filename_dict1[model_type_id] = {'imp':output_filename1,'imp_scaled':output_filename2,
													'coef_':coef_,'intercept_':intercept_}
			
		return dict_feature_imp_, filename_dict1

	## ====================================================
	# feature importance estimation from the trained model
	def test_model_explain_pre1(self,model,x,y,feature_name,model_type_name,x_test=[],y_test=[],linear_type_id=0,select_config={}):

		"""
		feature importance estimation from the trained model
		:param model: the trained model
		:param x: (dataframe or numpy.array) feature matrix of the observations (row:observation, column:predictor variable)
		:param y: (pandas.Series or numpy.array) response variable values of the observations
		:param feature_name: (array) names of the predictor variables
		:param model_type_name: (str) name of the model
		:param x_test: (dataframe or numpy.array) feature matrix of the test samples (row:observation, column:predictor variable)
		:param y_test: (pandas.Series or numpy.array) response variable values of test samples
		:param linear_type_id: indicator of the type of explainer to use; 
							   if lineary_type_id>=1, use the explainer in SHAP developed for linear models to estimate feature importance scores
		:param select_config: dictionary containing configuration parameters
		:return: 1. the Shapley value matrix of the predictor variables (row:observation, column:predictor variable);
				 2,3. the base Shapley values and the expected Shapley values of the predictor variables;
		"""

		if model_type_name in ['XGBR','XGBClassifier','RF','RandomForestClassifier']:
			explainer = shap.TreeExplainer(model)
			# explainer = shap.Explainer(model,x)
			shap_value_pre1 = explainer(x)
			shap_values = shap_value_pre1.values
			base_values = shap_value_pre1.base_values
			expected_value = base_values[0]
		elif linear_type_id>=1:
			feature_perturbation = ['interventional','correlation_dependent']
			feature_perturbation_id1 = linear_type_id-1
			feature_perturbation_id2 = feature_perturbation[feature_perturbation_id1]
			explainer = shap.explainers.Linear(model,x,nsamples=x.shape[0],feature_perturbation=feature_perturbation_id2)
			shap_value_pre1 = explainer(x)
			shap_values = shap_value_pre1.values
			base_values = shap_value_pre1.base_values
			expected_value = explainer.expected_value
		else:
			explainer = shap.Explainer(model, x, feature_names=feature_name)
			shap_value_pre1 = explainer(x)
			shap_values = shap_value_pre1.values
			base_values = shap_value_pre1.base_values
			expected_value = explainer.expected_value

		return shap_values, base_values, expected_value

	## ====================================================
	# feature importance estimation from the trained model
	def test_model_explain_pre2(self,model_train,x_train,y_train,feature_name,model_type_name,x_test=[],y_test=[],linear_type_id=0,save_mode=1,select_config={}):

		"""
		feature importance estimation from the trained model
		:param model_train: the trained model
		:param x_train: (dataframe or numpy.array) feature matrix of the training samples (row:observation, column:predictor variable)
		:param y_train: (pandas.Series or numpy.array) response variable values of the training samples
		:param feature_name: (array) names of the predictor variables
		:param model_type_name: (str) name of the model
		:param x_test: (dataframe or numpy.array) feature matrix of the test samples (row:observation, column:predictor variable)
		:param y_test: (pandas.Series or numpy.array) response variable values of test samples
		:param linear_type_id: indicator of the type of explainer to use; 
							   if lineary_type_id>=1, use the explainer in SHAP developed for linear models to estimate feature importance scores
		:param save_mode: indicator of whether to save data
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) feature importance scores (e.g., mean absolute Shapley value across the samples) of the predictor variables 
								(row:predictor variable, column:the type of feature importance score (e.g.,'shap_value'));
				 2. (dataframe) scaled feature importance scores of the predictor variables;
				 3. tuplet including estimated regression coefficients of the predictor variables and the intercept if regression model is used;
		"""

		if model_type_name in ['LR','Lasso','LassoCV','ElasticNet']:
			linear_type_id = 1

		shap_value_1, base_value_1, expected_value_1 = self.test_model_explain_pre1(model=model_train,x=x_train,y=y_train,
																					feature_name=feature_name,
																					model_type_name=model_type_name,
																					x_test=x_test,y_test=y_test,
																					linear_type_id=linear_type_id)

		feature_importances_1 = np.mean(np.abs(shap_value_1),axis=0)	# mean absolute Shapley values

		coef_query, intercept_query = [], []
		if feature_importances_1.ndim==1:
			df_imp_ = pd.DataFrame(index=feature_name,columns=['shap_value'])
			df_imp_scaled = pd.DataFrame(index=feature_name,columns=['shap_value'])
			df_imp_['shap_value'] = np.asarray(feature_importances_1)

			# feature importance estimates normalized to [0,1]
			feature_imp1_scale = pd.Series(index=feature_name,data=minmax_scale(feature_importances_1,[1e-07,1]))
			df_imp_scaled['shap_value'] = np.asarray(feature_imp1_scale)

			if model_type_name in ['XGBR','RandomForestRegressor']:
				# feature importance estimated by the model
				feature_importances_2 = model_train.feature_importances_
				df_imp_['imp2'] = np.asarray(feature_importances_2)

				# feature importance estimates normalized to [0,1]
				feature_imp2_scale = pd.Series(index=feature_name,data=minmax_scale(feature_importances_2,[1e-07,1]))
				df_imp_scaled['imp2'] = np.asarray(feature_imp2_scale)

		else:
			num_dim = feature_importances_1.shape[1]
			df_imp_ = pd.DataFrame(index=feature_name,columns=np.arange(num_dim),data=np.asarray(feature_importances_1))
			df_imp_scaled = pd.DataFrame(index=feature_name,columns=np.arange(num_dim),data=minmax_scale(feature_importances_1,[1e-07,1]))

			if model_type_name in ['LR','Lasso','ElasticNet','LogisticRegression']:
				coef_query, intercept_query = model_train.coef_, model_train.intercept_

		return df_imp_, df_imp_scaled, (coef_query, intercept_query)

	## ====================================================
	# query feature importance scores and compute average scores across multiple trained models
	def test_query_feature_mean_1(self,data=[],response_variable_name='',column_id_query='feature_name',column_vec_query=['fold_id'],type_id_1=0,verbose=0,select_config={}):

		"""
		query feature importance scores and compute average scores across multiple trained models
		:param data: (list) dataframes of estimated feature importance scores of the predictor variables by each trained model 
							(e.g., models trained in different folds in cross-validation)
		:param response_variable_name: (str) name of the response variable
		:param column_id_query: (str) name of the column representing predictor variable names;
		:param column_vec_query: (array or list) the columns to be included or exclued to select the columns used to compute the average feature importance scores
		:param type_id_1: indicator of which columns to use to compute the average feature importance scores:
		                  0: with the feature importance score dataframe, excluding the columns in column_vec_query to compute the average scores;
		                  1: using the columns in column_vec_query to compute the average scores;
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1. concatenated dataframe of feature importance scores of the predictor variables estimated from multiple trained models;
			     2. the average feature importance scores of each predictor variable across multiple trained models;
		"""

		list1_imp = data
		df_imp = pd.concat(list1_imp,axis=0,join='outer',ignore_index=False)
		df_imp.loc[:,column_id_query] = np.asarray(df_imp.index)
		if verbose>0:
			print('df_imp, response_variable %s'%(response_variable_name))
			print(df_imp)

		if type_id_1==0:
			column_vec_1 = df_imp.columns.difference(column_vec_query,sort=False)
		else:
			column_vec_1 = df_imp.columns.intersection(column_vec_query,sort=False)

		df_1 = df_imp.loc[:,column_vec_1].groupby(by=column_id_query).mean()
		df_imp1_mean = pd.DataFrame(index=df_1.index,columns=df_1.columns,data=np.asarray(df_1))

		return df_imp, df_imp1_mean

	## ====================================================
	# model training and interpretation
	def test_optimize_pre1_basic_unit1(self,x_train,y_train,x_test=[],y_test=[],sample_weight=[],model_type_id='',fold_id=-1,type_id_model=0,
											flag_model_load=0,flag_model_explain=0,output_mode=0,select_config1={},
											save_mode=0,save_mode_2=0,save_model_train=1,model_path='',output_file_path='',output_filename='',
											filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		model training and interpretation
		:param x_train: (dataframe or numpy.array) feature matrix of the training samples (row:observation, column:predictor variable)
		:param y_train: (pandas.Series or numpy.array) response variable values of the training samples
		:param x_test: (dataframe or numpy.array) feature matrix of the test samples (row:observation, column:predictor variable)
		:param y_test: (pandas.Series or numpy.array) response variable values of test samples
		:param sample_weight: (array) sample weight of the training and test samples
		:param model_type_id: (str) name of the model
		:param fold_id: (int) index of the fold with which the model is trained if cross-validation is used 
		:param type_id_model: (int) the prediction model type: 0: regression model; 1: classification model
		:param flag_model_load: indicator of whether to load trained model
		:param flag_model_explain: indicator of whether to perform model interpretation to estimate feature importance scores of the predictor variables
		:param output_mode: indicator of whether to save predicted class labels and positive class prababilites or response variable values of the test samples 
							in the prediction dataframes of the full samples
		:param select_config1: dictionary containing hyperparameters of the model and parameters for evaluation
		:param save_mode: indicator of whether to save data
		:param save_mode_2: data saving mode in the model interpretation part
		:param save_model_train: indicator of whether to save the trained models
		:param model_path: (str) the directory to save the trained models
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. the trained model;
				 2. list including the array of estimated regression coeffiecients of the predictor variables and the learned intercept if regression model is used;
				 3. tuplet including the following elements:
				 	3-1. (pandas.Series) prediction performance scores using different evaluation metrics;
				 	3-2. (array) predicted class labels or response variable values of the test samples;
				 	3-3. (array) predicted positive label probabilities of the test samples;
				 	3-4. dictionary containing estimated original and scaled feature importance scores and 
				 		 regression coefficients of the predictor variables and the intercept for the corresponding model type; 
		"""

		model_path_1 = model_path
		model_save_filename = ''
		if flag_model_load==0:
			model_1 = self.test_model_basic_pre1(model_type_id=model_type_id,select_config=select_config1)

			model_1, param1 = self.test_model_train_basic_pre1(model_train=model_1,
																model_type_id=model_type_id,
																x_train=x_train,
																y_train=y_train,
																sample_weight=sample_weight)

			if save_model_train>0:
				# save models from the cross validation
				save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
				with open(save_filename,'wb') as output_file:
					pickle.dump(model_1,output_file)
				select_config.update({'model_save_filename':save_filename})
		else:
			model_save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
			with open(model_save_filename, 'rb') as fid:
				model_1 = pickle.load(fid)
				if model_type_id in ['LR','Lasso','ElasticNet','LogisticRegression']:
					try:
						param1 = [model_1.coef_, model_1.intercept_]
					except Exception as error:
						print('error! ',error)

			if verbose>0:
				print('model weights loaded ',model_save_filename)

		# save_mode_2 = (save_model_train==2)
		# model_save_filename = select_config['model_save_filename']
		list_query1 = self.test_model_pred_explain_1(model_train=model_1,
														x_test=x_test,
														y_test=y_test,
														sample_id_test_query=[],
														y_pred=[],
														y_pred_proba=[],
														x_train=x_train,
														y_train=y_train,
														fold_id=fold_id,
														type_id_model=type_id_model,
														model_explain=flag_model_explain,
														model_save_filename=model_save_filename,
														output_mode=output_mode,
														save_mode=save_mode_2,
														verbose=0,
														select_config=select_config)

		return model_1, param1, list_query1

	## ====================================================
	# model training and interpretation
	def test_optimize_pre1(self,model_pre,x_train,y_train,response_variable_name,feature_name=[],
								sample_weight=[],dict_query={},df_coef_query=[],df_pred_query=[],
								model_type_vec=[],model_type_idvec=[],
								type_id_model=0,num_class=1,
								save_mode=0,save_model_train=1,model_path_1='',output_file_path='',output_filename='',
								filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
		
		"""
		model training and interpretation
		:param model_pre: the prediction model to train
		:param x_train: (dataframe or numpy.array) feature matrix of the training samples (row:observation, column:predictor variable)
		:param y_train: (pandas.Series or numpy.array) response variable values of the training samples
		:param response_variable_name: (str) name of the response variable
		:param feature_name: (array) names of the predictor variables
		:param sample_weight: (array) sample weight of the training and test samples
		:param dict_query: the dictionary to save the specific output and attributes of the model, including: 
						   (1) predicted labels and class assignment probabilities of the samples;
						   (2) estimated regression coefficients and importance scores of the predictor variables;
		:param df_coef_query: (dataframe) the regression coefficients of the predictor variables and the intercept learned by the model
		:param df_pred_query: (dataframe) the predicted values of the response variables
		:param model_type_vec: (array or list) names of different types of models
		:param model_type_idvec: (array of list) name of the model to train
		:param type_id_model: (int) the prediction model type: 0: regression model; 1: classification model
		:param num_class: the number of classes
		:param save_mode: indicator of whether to save data
		:param save_model_train: indicator of whether to save the trained models
		:param model_path_1: (str) the directory to save the trained models
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing the specific output and attributes of the trained model, including: 
					predicted labels and class assignment probabilities of the samples;
					estimated regression coefficients and importance scores of the predictor variables;
				 2. prediction performance scores using different evaluation metrics;
		"""

		x_train1, y_train1 = x_train, y_train
		sample_idvec_pre1 = x_train1.index
		feature_query_vec_coef = x_train1.columns.copy()

		num_fold = select_config['num_fold']
		sample_idvec_query = select_config['sample_idvec_train']
		select_config1 = select_config['select_config1']
		train_valid_mode = select_config['train_valid_mode_1']
		# pre_data_dict_1 = pre_data_dict
		
		if type_id_model==1:
			if 'num_class' in select_config:
				num_class = select_config['num_class']
			else:
				num_class = len(np.unique(y_train1))
				select_config.update({'num_class':num_class})
			print('num_class ',num_class)
			if num_class==2:
				# binary classification model
				average_type = 'binary'
				type_id1 = 0
				select_config1.update({'type_classifer_xbgboost':type_id1})
				num_pos = np.sum(y_train1>0)
				num_neg = np.sum(y_train1==0)
				print('num_pos, num_neg ',num_pos,num_neg)
			else:
				# multi-class classification model
				# average_type = 'macro'
				average_type = 'micro'
				type_id1 = 1
				select_config1.update({'type_classifer_xbgboost':type_id1})
				multi_class_query = 'auto'
				column_1 = 'multi_class_logisticregression'
				if column_1 in select_config:
					multi_class_query = select_config[column_1]
				select_config1.update({column_1:multi_class_query})
				print('multi_class_logisticregression: ',multi_class_query)

			if 'average_type' in select_config1:
				average_type_1 = select_config1['average_type']
				if num_class!=2:
					if average_type_1=='binary':
						select_config1.update({'average_type':average_type})
			else:
				select_config1.update({'average_type':average_type})

			print('num_class, average_type ',num_class,average_type)
			
		list1 = []
		list2 = []
		model_type_num = len(model_type_idvec)
		dict_query_1 = dict_query
		filename_save_annot = select_config['filename_save_annot_local']
		run_id = select_config['run_id']
		if model_path_1=='':
			model_path_1 = output_file_path

		np.random.seed(0)
		for i1 in range(model_type_num):
			model_type_id1 = model_type_idvec[i1]
			# print('model_type_id1 ',model_type_id1)
			df_coef_query, df_pred_query, df_score_1 = [], [], []
			df_coef_pval_ = []
			dict_query_1[model_type_id1] = dict()
			
			y_pred1 = pd.Series(index=sample_idvec_pre1,data=0,dtype=np.float32)
			# output_mode = 1
			output_mode = 0
			if type_id_model==1:
				# num_class = select_config['num_class']
				y_pred1_proba = pd.DataFrame(index=sample_idvec_pre1,columns=range(num_class),data=0,dtype=np.float32)
			else:
				y_pred1_proba = []

			flag_model_load = 0
			if 'model_load_filename_annot' in select_config:
				model_load_filename_annot = select_config['model_load_filename_annot']
				flag_model_load = 1

			if 'flag_model_load' in select_config:
				flag_model_load = select_config['flag_model_load']

			feature_type_id = select_config['feature_type_id']
			flag_model_explain = select_config['flag_model_explain']
			linear_type_id = 0
			if 'linear_type_id' in select_config:
				linear_type_id = select_config['linear_type_id']
			if verbose>0:
				print('flag_model_explain, linear_type_id ',flag_model_explain,linear_type_id)

			if model_type_id1 in ['LR_2']:
				intercept_flag = True
				if 'intercept_flag' in select_config:
					intercept_flag = select_config['intercept_flag']
				if intercept_flag==True:
					x_train1 = sm.add_constant(x_train1)

			# train with cross validation for performance evaluation
			list1_imp, list1_imp_scale = [], []

			df_pred_query = []
			df_pred_proba = []
			list_feature_imp = []
			dict_feature_imp = dict()

			if num_fold>0:
				for fold_id in range(num_fold):
					sample_idvec_1 = sample_idvec_query[fold_id]
					sample_id_train_query, sample_id_valid_query, sample_id_test_query = sample_idvec_1

					x_train2, y_train2 = x_train1.loc[sample_id_train_query,:], y_train1.loc[sample_id_train_query]
					x_test2, y_test2 = x_train1.loc[sample_id_test_query,:], y_train1.loc[sample_id_test_query]

					save_mode_2 = (save_model_train==2)
					save_model_train_1 = (save_model_train==2)
					model_1, param1, list_query1 = self.test_optimize_pre1_basic_unit1(x_train=x_train2,y_train=y_train2,
																							x_test=x_test2,
																							y_test=y_test2,
																							sample_weight=sample_weight,
																							model_type_id=model_type_id1,
																							fold_id=fold_id,
																							type_id_model=type_id_model,
																							flag_model_load=flag_model_load,
																							flag_model_explain=flag_model_explain,
																							output_mode=output_mode,
																							select_config1=select_config1,
																							save_mode=save_mode,
																							save_mode_2=save_mode_2,
																							save_model_train=save_model_train_1,
																							model_path=model_path_1,
																							output_file_path=output_file_path,
																							filename_prefix_save=filename_prefix_save,
																							filename_save_annot=filename_save_annot,
																							output_filename='',
																							verbose=0,select_config=select_config)

					score_1, y_test2_pred, y_test2_proba, dict_query1 = list_query1

					y_pred1.loc[sample_id_test_query] = y_test2_pred
					if type_id_model==1:
						y_pred1_proba.loc[sample_id_test_query,:] = y_test2_proba
					list1.append(score_1)

					if flag_model_explain>0:
						if 'imp' in dict_query1:
							df_imp_1 = dict_query1['imp']
							df_imp_1['fold_id'] = fold_id
							list1_imp.append(df_imp_1)

						if 'imp_scaled' in dict_query1:
							df_imp_scaled_1 = dict_query1['imp_scaled']
							df_imp_scaled_1['fold_id'] = fold_id
							list1_imp_scale.append(df_imp_scaled_1)

					if verbose>0:
						print('fold id, x_train2, y_train2, x_test2, y_test2 ',fold_id,x_train2.shape,y_train2.shape,x_test2.shape,y_test2.shape)
						print(score_1,fold_id,num_fold)

				if type_id_model==0:
					# regression model
					score_2 = self.score_function(y_train1,y_pred1)
				else:
					# classification model
					score_2 = self.score_function_multiclass2(y_train1,y_pred1,y_pred1_proba,average=average_type)

				list1.append(score_2)
				if verbose>0:
					print(score_2,sample_idvec_pre1[0:2],x_train1.shape,y_train1.shape)

				df_score_1 = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				df_score_1 = df_score_1.T
				# df_score_1['sample_id'] = [sample_id1]*df_score_1.shape[0]
				df_score_1['fold_id'] = np.asarray(df_score_1.index)

				column_id_query = 'feature_name'
				column_vec_2 = ['fold_id']
				column_id_query = ''
				if (flag_model_explain>0) and (len(list1_imp)>0):
					df_imp, df_imp1_mean = self.test_query_feature_mean_1(data=list1_imp,
																			response_variable_name=response_variable_name,
																			column_id_query=column_id_query,
																			column_vec_query=column_vec_2,
																			type_id_1=0,
																			verbose=verbose,select_config=select_config)
					dict_feature_imp.update({'imp1':df_imp,'imp1_mean':df_imp1_mean})

				if (flag_model_explain>0) and (len(list1_imp_scale)>0):
					df_imp_scaled, df_imp1_scaled_mean = self.test_query_feature_mean_1(data=list1_imp_scale,
																						response_variable_name=response_variable_name,
																						column_id_query=column_id_query,
																						column_vec_query=column_vec_2,
																						type_id_1=0,
																						verbose=verbose,select_config=select_config)
					dict_feature_imp.update({'imp1_scale':df_imp_scaled,'imp1_scale_mean':df_imp1_scaled_mean})

				df_pred_query = y_pred1
				df_pred_proba = y_pred1_proba

				df_pred_query = y_pred1
				df_pred_proba = 1

			# train on the combined data for coefficient estimation
			# param2 = []
			if train_valid_mode>0:
				save_mode_2 = 0
				model_2, param2, list_query2 = self.test_optimize_pre1_basic_unit1(x_train=x_train1,y_train=y_train1,
																						x_test=x_train1,y_test=y_train1,
																						sample_weight=sample_weight,
																						model_type_id=model_type_id1,
																						fold_id=-1,
																						type_id_model=type_id_model,
																						flag_model_load=flag_model_load,
																						flag_model_explain=flag_model_explain,
																						output_mode=output_mode,
																						select_config1=select_config1,
																						save_mode=save_mode,
																						save_mode_2=save_mode_2,
																						save_model_train=save_model_train,
																						model_path=model_path_1,
																						output_file_path=output_file_path,
																						output_filename='',
																						filename_prefix_save=filename_prefix_save,
																						filename_save_annot=filename_save_annot,
																						verbose=0,select_config=select_config)

				score_2, y_pred, y_proba, dict_query2 = list_query2
				list1 = [y_pred,y_proba]
				query_num = len(list1)
				for l1 in range(query_num):
					y_query = list1[l1]
					if len(y_query)>0:
						if y_query.ndim==1:
							y_query = pd.Series(index=sample_idvec_pre1,data=np.asarray(y_query),dtype=np.float32)
							y_query.name = response_variable_name
						else:
							n_dim = y_query.shape[1]
							y_query = pd.DataFrame(index=sample_idvec_pre1,columns=np.arange(n_dim),data=np.asarray(y_query),dtype=np.float32)
					list1[l1] = y_query

				y_pred, y_proba = list1
				if type_id_model==0:
					# regression model
					score_2 = self.score_function(y_train1,y_pred)
				else:
					# classification model
					score_2 = self.score_function_multiclass2(y_train1,y_pred,y_proba,average=average_type)

				field_query = score_2.index
				df_score_2 = pd.DataFrame(index=[response_variable_name],columns=field_query,data=np.asarray(score_2)[np.newaxis,:])
				dict_query_1[model_type_id1].update({'model_combine':model_2,'df_score_2':df_score_2})	# prediction performance on the combined data

				if flag_model_explain>0:
					df_imp_2, df_imp_scaled_2 = dict_query2['imp'], dict_query2['imp_scaled']
					# list_feature_imp.append(df_imp_2, df_imp_scaled_2)
					dict_feature_imp.update({'imp2':df_imp_2,'imp2_scale':df_imp_scaled_2})

				feature_query_vec_coef = x_train1.columns
				if len(param2)>0:
					# alpha_query = param1[0].copy()
					alpha_query1 = param2[0]
					intercept_ = param2[1]
					print('alpha_query1, intercept_: ',len(alpha_query1), intercept_)
					if len(alpha_query1)>0:
						query_idvec = list(feature_query_vec_coef)+['alpha0']
						if num_class<2:
							alpha_query = list(alpha_query1)+[intercept_]
							if len(df_coef_query)==0:
								df_coef_query = pd.Series(index=query_idvec,data=alpha_query,dtype=np.float32)
								df_coef_query.name = response_variable_name
							else:
								df_coef_query.loc[query_idvec,response_variable_name] = alpha_query
						else:
							alpha_query = np.hstack((alpha_query1,intercept_[:,np.newaxis]))
							alpha_query = alpha_query.T
							if num_class==2:
								df_coef_query = pd.DataFrame(index=query_idvec,columns=[response_variable_name],data=np.asarray(alpha_query),dtype=np.float32)
							else:
								df_coef_query = pd.DataFrame(index=query_idvec,columns=np.arange(num_class),data=np.asarray(alpha_query),dtype=np.float32)

					if model_type_id1 in ['LR_2']:
						df_coef_pval_1 = self.test_query_coef_pval_1(model_train=model_2,model_type_id=0,select_config=select_config)
						pval_query = np.asarray(df_coef_pval_1.loc[query_idvec,'pval'])
						if len(df_coef_pval_)==0:
							df_coef_pval_ = pd.DataFrame(index=query_idvec,columns=[response_variable_name],data=pval_query,dtype=np.float32)
						else:
							df_coef_pval_.loc[query_idvec,response_variable_name] = pval_query
						
						dict_query_1[model_type_id1].update({'pval':df_coef_pval_})

			dict_query_1[model_type_id1].update({'coef':df_coef_query, 
													'pred_cv':df_pred_query, 
													'pred_proba_cv':df_pred_proba, 
													'pred':y_pred, 
													'pred_proba':y_proba, 
													'feature_imp':dict_feature_imp})
								
		return dict_query_1, df_score_1

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)

	
	



