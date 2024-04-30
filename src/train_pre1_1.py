#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
import scanpy.external as sce

from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.switch_backend('Agg')
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import xgboost
# import xgbfir

from pandas import read_excel

import pyranges as pr
import warnings

# import palantir 
import phenograph
# import harmony

import sys
# import build_graph
# import metacells_ad
from tqdm.notebook import tqdm

import csv
import os
import os.path
import shutil
import sklearn

from optparse import OptionParser
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoCV
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.svm import SVR
# from processSeq import load_seq_1, kmer_dict, load_signal_1, load_seq_2, load_seq_2_kmer, load_seq_altfeature_1
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import make_regression

# from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA,SparsePCA,TruncatedSVD
from sklearn.decomposition import FastICA, NMF, MiniBatchDictionaryLearning
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.fixes import loguniform
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.feature_selection import chi2, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel,RFE,RFECV,VarianceThreshold
# from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.inspection import permutation_importance
from sklearn.cluster import AffinityPropagation,SpectralClustering,AgglomerativeClustering,DBSCAN,OPTICS,cluster_optics_dbscan
from sklearn.cluster import MiniBatchKMeans,KMeans,MeanShift,estimate_bandwidth,Birch
# import groupyr as gpr

# from group_lasso import GroupLasso
# from group_lasso.utils import extract_ohe_groups
# from group_lasso import LogisticGroupLasso
# import asgl

from scipy import stats
from scipy.stats import multivariate_normal, skew, pearsonr, spearmanr
from scipy.stats import wilcoxon, mannwhitneyu, kstest, ks_2samp, chisquare, fisher_exact, barnard_exact, boschloo_exact
from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq
from scipy.stats import gaussian_kde, zscore
from scipy.stats import poisson, multinomial
from scipy.stats import norm
import scipy.sparse
from scipy.sparse import spmatrix
from scipy.sparse import hstack, csr_matrix, csc_matrix, issparse, vstack
from scipy import signal
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
# import lowess
import shap
import networkx as nx

from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

from scipy.cluster.hierarchy import dendrogram, linkage

# import tensorflow as tf
# import keras
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.models import Model, clone_model, Sequential

import gc
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

import utility_1
# from utility_1 import log_transform, pyranges_from_strings, plot_cell_types, plot_gene_expression
# from utility_1 import _dot_func, impute_data, density_2d, plot_gene_expression
import h5py
import json
import pickle

# pairwise distance metric: spearmanr
# from the notebook
def sp(x, y):
	return spearmanr(x, y)[0]

def _spearmanr(model,x,y):
		
	y_pred = model.predict(x)
	t_score = spearmanr(y,y_pred)
	spearman_corr = t_score[0]

	return spearman_corr

# pairwise distance metric: pearsonr
# from the notebook
def pearson_corr(x, y):
	return pearsonr(x, y)[0]

class _Base2_train1(BaseEstimator):
	"""
	Parameters
	----------

	"""

	def __init__(
		self,
		peak_read=[],
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
		select_config={}
	):
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

	## test estimate peak coefficients
	# LR_compare: using LR model for comparison
	# test estimate peak coefficients
	# def test_peak_coef_est(self,tf_expr_local,peak_read_local,gene_expr,alpha_motif,peak_mtx,sample_id_vec,iter_id,model_type_id,model_train,pre_data_dict={},sample_weight=[],select_config={},LR_compare=0,save_mode=0,feature_imp_est=1,regularize_est=1,save_mode1=1,save_mode2=0,output_file_path=''):
	def test_peak_coef_est(self,peak_loc_query=[],feature_query_vec=[],gene_query_id='',alpha_motif=[],peak_read=[],rna_exprs=[],feature_query_expr=[],gene_query_expr=[],peak_motif_mtx=[],sample_id_vec=[],iter_id=0,model_type_id='',model_train=[],pre_data_dict={},sample_weight=[],LR_compare=1,save_mode=1,feature_imp_est=1,regularize_est=1,save_mode1=1,save_mode2=0,output_file_path='',select_config={}):

		# tf_value1 = np.asarray(tf_expr_local)*np.asarray(alpha_motif) # shape: (sample_num,motif_num)
		tf_value1 = np.asarray(feature_query_expr)*np.asarray(alpha_motif) # shape: (sample_num,motif_num)
		peak_tf_expr_vec_2 = tf_value1.dot(peak_motif_mtx.T)      # shape: (sample_num,peak_num)
		# peak_tf_expr_vec2 = peak_read_local*peak_tf_expr_vec_2  # shape: (sample_num,peak_num)
		peak_tf_expr_vec2 = peak_read*peak_tf_expr_vec_2  # shape: (sample_num,peak_num)
		x = peak_tf_expr_vec2
		y = gene_query_expr

		sample_id_train, sample_id_valid, sample_id_test = sample_id_vec
		# print('feature_imp_est ',feature_imp_est)
		beta_vec, model_train_pre, x, y, feature_imp_est_vec, score_pred_vec = self.test_feature_coef_est_pre1(x,y,model_type_id=model_type_id,sample_id_vec=sample_id_vec,
																								iter_id=iter_id,sample_weight=sample_weight,
																								feature_imp_est=feature_imp_est,
																								regularize_est=regularize_est,
																								LR_compare=LR_compare,
																								save_mode=save_mode,
																								output_file_path=output_file_path,
																								select_config=select_config)

		beta_peak,beta0 = beta_vec
		shap_values, base_values, expected_value = feature_imp_est_vec
		# print('shap_values ', len(shap_values))

		return beta_vec, model_train_pre, x, y, feature_imp_est_vec

	## tf motif coefficients estimate
	# def test_motif_coef_est(self,tf_expr_local,peak_read_local,gene_expr,beta_peak,peak_mtx,sample_id_vec,iter_id,model_type_id1,model_1,pre_data_dict={},sample_weight=[],select_config={},LR_compare=1,save_mode=1,feature_imp_est=1,regularize_est=1,save_mode1=1,save_mode2=0,output_file_path=''):
	def test_motif_coef_est(self,peak_loc_query=[],feature_query_vec=[],gene_query_id='',beta_peak=[],peak_read=[],rna_exprs=[],feature_query_expr=[],gene_query_expr=[],peak_motif_mtx=[],sample_id_vec=[],iter_id=0,model_type_id='',model_train=[],pre_data_dict={},sample_weight=[],LR_compare=1,save_mode=1,feature_imp_est=1,regularize_est=1,save_mode1=1,save_mode2=0,output_file_path='',select_config={}):

		peak_value1 = np.asarray(peak_read)*np.asarray(beta_peak)   # shape: (sample_num,peak_num), peak coefficients for the peak loci
		peak_tf_expr_vec_1 = peak_value1.dot(peak_motif_mtx)  # shape: (sample_num,motif_num)
		# tf_expr_local = rna_exprs.loc[:,feature_query_vec]
		# gene_expr = rna_exprs.loc[:,gene_query_id]
		peak_tf_expr_vec1 = feature_query_expr*peak_tf_expr_vec_1   # shape: (sample_num,motif_num)
		x = peak_tf_expr_vec1
		y = gene_query_expr
		
		sample_id_train, sample_id_valid, sample_id_test = sample_id_vec
		alpha_vec, model_train_pre, x, y, feature_imp_est_vec, score_pred_vec = self.test_feature_coef_est_pre1(x,y,model_type_id=model_type_id,sample_id_vec=sample_id_vec,
																								iter_id=iter_id,sample_weight=sample_weight,
																								feature_imp_est=feature_imp_est,
																								regularize_est=regularize_est,
																								LR_compare=LR_compare,
																								save_mode=save_mode,
																								output_file_path=output_file_path,
																								select_config=select_config)
		alpha_motif,alpha0 = alpha_vec
		shap_values, base_values, expected_value = feature_imp_est_vec

		# return [alpha_motif,alpha0], model_train_pre, x, y, [shap_values, base_values, expected_value]
		return alpha_vec, model_train_pre, x, y, feature_imp_est_vec

	# tf motif coefficients estimate
	# def test_motif_coef_est(self,tf_expr_local,peak_read_local,gene_expr,beta_peak,peak_mtx,sample_id_vec,iter_id,model_type_id1,model_1,pre_data_dict={},sample_weight=[],select_config={},LR_compare=1,save_mode=1,feature_imp_est=1,regularize_est=1,save_mode1=1,save_mode2=0,output_file_path=''):
	def test_feature_coef_est_pre1(self,x,y,model_type_id,peak_read=[],rna_exprs=[],feature_query_expr=[],gene_query_expr=[],peak_motif_mtx=[],sample_id_vec=[],iter_id=0,model_train=[],pre_data_dict={},sample_weight=[],select_config={},LR_compare=1,save_mode=1,feature_imp_est=1,regularize_est=1,type_id_score_vec=[0,0,1],save_mode1=1,save_mode2=0,output_file_path=''):
		
		model_type_id1 = model_type_id
		sample_id_train, sample_id_valid, sample_id_test = sample_id_vec

		# model_1 = pre_model_dict1[t_model_type_id1]
		# model_1 = pre_model_list1[0]
		x_train, y_train = x.loc[sample_id_train,:], y.loc[sample_id_train]
		x_test, y_test = x.loc[sample_id_test,:], y.loc[sample_id_test]
		# print('x_train, y_train ',x_train.shape,y_train.shape)
		# print('x_test, y_test ',x_test.shape,y_test.shape)
		x_valid, y_valid = [], []
		if len(sample_id_valid)>0:
			x_vaild, y_valid = x.loc[sample_id_valid,:], y.loc[sample_id_valid]

		# save_mode2=0
		# if save_mode2>0:
		#   x_1 = np.hstack((y_train[:,np.newaxis],x_train))
		#   run_id = select_config['run_id_pre']
		#   output_filename = 'vbak2_1/test_motif_peak_estimate_iter%d.1.%d.txt'%(iter_id,run_id)
		#   print('x_train write 1 ',iter_id,x_train.shape,y_train.shape)
		#   feature_name = x_train.columns
		#   print(len(feature_name),feature_name)
		#   t_data1 = pd.DataFrame(index=x_train.index,columns=['y']+list(x_train.columns),data=x_1)
		#   t_data1.T.to_csv(output_filename,sep='\t',float_format='%.6E')

		flag_iter_regularize_est = 0
		if 'iter_num_regularize_est' in select_config:
			iter_num_regularize_est = select_config['iter_num_regularize_est']
			iter_interval_regularize_est = select_config['iter_interval_regularize_est']
		else:
			iter_num_regularize_est = select_config['n_iter']
			iter_interval_regularize_est = 1

		## signal threshold
		flag_clip_1 = 0
		if flag_clip_1>0:
			y_train = self.test_signal_clip_1(y_train,type_id=0,thresh_1=0.01,thresh_2=0.99,select_config=select_config)

		flag_clip_2 = 0
		if flag_clip_2>0:
			for t_field_query in x.columns:
				x[t_field_query] = self.test_signal_clip_1(x[t_field_query],type_id=0,thresh_1=0.01,thresh_2=0.99,select_config=select_config)

		## regularization coefficient estimation
		select_config1 = select_config.copy()
		# print('regularize_est ',regularize_est)
		if (regularize_est==1) and (model_type_id in ['Lasso']):
			# estimate regularization coefficients at the begnning steps or at intervals
			if (iter_id < iter_num_regularize_est) or (iter_id%iter_interval_regularize_est==0):
				flag_iter_regularize_est = 1

			Lasso_alpha1, Lasso_alpha2 = self._regularize_coef_est_2(x_train,y_train,type_id_1=select_config['regularize_est2_type'],select_config=select_config)

			# # Lasso_alpha = np.max([Lasso_alpha1,1e-05])
			# a1 = 1e-02
			# a1 = 1.0
			if select_config['regularize_est1_select']==0:
				Lasso_alpha = Lasso_alpha1
			else:
				Lasso_alpha = Lasso_alpha2

			# Lasso_alpha = np.max([Lasso_alpha1*a1,1e-05])
			# run_id: 8: 5e-07
			# run_id: 7: 1e-07
			# select_config1 = select_config.copy()
			select_config1.update({'Lasso_alpha':Lasso_alpha})
			# print('Lasso alpha ', model_type_id1, Lasso_alpha, Lasso_alpha1, Lasso_alpha2)

		motif_group= []
		model_1 = self.test_model_basic_pre1(model_type_id=model_type_id,motif_group=motif_group,pre_data_dict=pre_data_dict,select_config=select_config1)

		flag_pre = 0
		cnt1 = 0
		alpha0, alpha_query = 0, []
		while flag_pre==0:
			# flag1 = 1
			# if flag1>0:
			try:
				# model_1, param1 = self.test_motif_peak_estimate_local_sub1(model_1,t_model_type_id_1,x_train,y_train,sample_weight=sample_weight)
				model_1, param1 = self.test_model_train_basic_pre1(model_1,model_type_id,x_train,y_train,sample_weight=sample_weight)
				alpha_query = param1[0].copy()
				alpha0 = param1[1]
				flag_pre = 1

			except Exception as error:
				print('test_motif_coef_est training, error! ', error)
				if cnt1>1:
					break
				if cnt1==0:
					# reset regularization coefficient
					Lasso_alpha = select_config['Lasso_alpha_vec1'][0]
					print('reuse the original Lasso_alpha ',Lasso_alpha)
				else:
					Lasso_alpha = 1E-07
					# print('Lasso_alpha ',Lasso_alpha,cnt1)
				
				select_config1 = select_config.copy()
				select_config1.update({'Lasso_alpha':Lasso_alpha})
				print('Lasso alpha ', model_type_id1, Lasso_alpha)
				model_1 = self.test_model_basic_pre1(model_type_id=model_type_id1,motif_group=motif_group,pre_data_dict=pre_data_dict,select_config=select_config1)
				cnt1 += 1

		alpha_query = self.test_check_param_1(alpha_query,bound1=1e05,bound2=-1e05)

		# model explainer
		feature_name = x_train.columns
		alpha_query = pd.Series(index=feature_name,data=np.asarray(alpha_query))
		shap_values, base_values, expected_value = [], [], []
		# print('test_feature_coef_est_pre1, feature_imp_est ',feature_imp_est)
		if feature_imp_est==1:
			linear_type_id = 0
			shap_values, base_values, expected_value = self.test_model_explain_pre1(model_1,x_train,y_train,feature_name=feature_name,
																				model_type_id=model_type_id1,
																				x_test=x_test,y_test=y_test,
																				linear_type_id=linear_type_id)
			# print('shap_values ',len(shap_values))
			shap_values = pd.DataFrame(index=x_train.index,columns=feature_name,data=np.asarray(shap_values),dtype=np.float32)

		type_id_score_train, type_id_score_valid, type_id_score_test = type_id_score_vec
		flag_test = type_id_score_test
		flag_valid = type_id_score_train
		df_pred = []
		score_pred = []
		list1 = [x_train,x_valid,x_test]
		list2 = [y_train,y_valid,y_test]
		query_num2 = len(list1)
		score_pred_dict = dict()
		dict_pred = dict()

		field_query = ['train','valid','test']
		for i1 in range(query_num2):
			flag_query = type_id_score_vec[i1]
			x_pre = list1[i1]
			y_ori = list2[i1]
			if (flag_query==1) and (len(x_pre)>0):
				y_pred = model_1.predict(x_pre)
				score_pred = self.score_2a(np.ravel(y_ori), np.ravel(y_pred))
				sample_id_pre = x_pre.index
				df_pred = pd.DataFrame(index=sample_id_pre,columns=['signal','pred'])
				df_pred['signal'], df_pred['pred'] = np.asarray(y_ori), np.asarray(y_pred)

				score_pred_dict.update({field_query[i1]:score_pred})
				dict_pred.update({field_query[i1]:df_pred})

		# if flag_test==1:
		# 	y_test_pred = model_1.predict(x_test)
		# 	score_pred = self.score_2a(np.ravel(y_test), np.ravel(y_test_pred))
		# 	df_pred = pd.DataFrame(index=sample_id_test,columns=['signal','pred'])
		# 	df_pred['signal'], df_pred['pred'] = np.asarray(y_test), np.asarray(y_test_pred)

		# return ((alpha_motif,alpha0), score_1, model_1, shap_values, feature_name, score_compare_1, score_train_valid, y_pred_test1)
		return [alpha_query,alpha0], model_1, x, y, [shap_values, base_values, expected_value], [score_pred_dict,dict_pred]

	## gene-peak association query: index reset
	def test_query_id_reset(self,df_query,column_id=['peak_id','gene_id'],select_config={}):

		df_query.index = ['%s.%s'%(query_id1,query_id2) for (query_id1,query_id2) in zip(df_query[column_id[0]],df_query[column_id[1]])]

		return df_query

	def score_2a(self, y, y_predicted):

		score1 = mean_squared_error(y, y_predicted)
		score2 = pearsonr(y, y_predicted)
		score3 = explained_variance_score(y, y_predicted)
		score4 = mean_absolute_error(y, y_predicted)
		score5 = median_absolute_error(y, y_predicted)
		score6 = r2_score(y, y_predicted)
		score7, pvalue = spearmanr(y,y_predicted)
		t_mutual_info = mutual_info_regression(y[:,np.newaxis], y_predicted, discrete_features=False, n_neighbors=5, copy=True, random_state=0)
		t_mutual_info = t_mutual_info[0]

		# vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6]
		vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6, score7, pvalue, t_mutual_info]

		field_query_1 = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2','mutual_info']
		df_score_pred = pd.Series(index=field_query_1,data=vec1,dtype=np.float32)
		
		# return vec1
		return df_score_pred

	## peak-motif link estimation
	def test_peak_motif_est(self,gene_query_vec=[],peak_read=[],rna_exprs=[],feature_query_expr=[],gene_query_expr=[],motif_data=[],pre_data_dict={},save_mode=1,output_file_path='',select_config={}):
		
		file_path1 = self.data_dir

	## peak-motif link query
	def test_peak_motif_query(self,gene_query_vec=[],df_gene_peak_query=[],peak_read=[],rna_exprs=[],feature_query_expr=[],gene_query_expr=[],motif_data=[],pre_data_dict={},save_mode=1,output_file_path='',select_config={}):
		
		file_path1 = self.data_dir
		peak_loc_query = df_gene_peak_query.loc[gene_query_vec,'peak_id'].unique()
		print('peak_loc_query ', len(peak_loc_query))
		peak_motif_mtx = self.motif_data.loc[peak_loc_query,:]

		return peak_motif_mtx

	# regularization coefficient estimate
	def _regularize_coef_est_1(self, params, x, y, motif_group_list, motif_query_local,select_config={}):

		beta0, beta1 = params[0], params[1:]
		# beta0 = 0
		# beta1 = params
		# motif_group = pre_data_dict['motif_group']
		# motif_group_dict = pre_data_dict['motif_group_dict']
		# motif_group_list = pre_data_dict['motif_group_list']
		# group_num = len(motif_group)
		# x = np.asarray(x)
		# y = np.asarray(y)
		group_num = len(motif_group_list)
		print('motif_group_list ',motif_group_list)
		print('beta0, beta1 ',beta0,beta1.shape,beta1)
		x = np.asarray(x)
		y = np.asarray(y)

		y1 = np.zeros_like(y,dtype=np.float32)+beta0
		group_regularize = 0
		t_motif_query_local = np.asarray(motif_query_local)
		for i1 in range(group_num):
			# t_group_id = motif_group[i]
			# idvec = motif_group_dict[t_group_id]
			# idvec = motif_group_dict[i]
			idvec = motif_group_list[i1]
			print(idvec,i1,t_motif_query_local[idvec])
			y1 = y1 + x[:,idvec].dot(beta1[idvec])
			group_regularize += np.sqrt(len(idvec))*np.linalg.norm(beta1[idvec])

		L1 = 0.5*np.mean(np.square(y-y1))
		# lambda1 = lambda_regularize
		# regularizer = (1-alpha)*lambda_regularize*group_regularize+alpha*lambda_regularize*np.sum(np.abs(beta1))
		# regularizer = lambda1*group_regularize + lambda2*np.sum(np.abs(beta1))
		sample_num = x.shape[0]
		# regularizer = lambda1*group_regularize + lambda2*np.sum(np.abs(beta1))
		# lik = L1+regularizer*1.0/np.sqrt(sample_num)
		regularize_1 = np.sum(np.abs(beta1))

		weight1 = 1.0/np.sqrt(sample_num)
		group_regularize = group_regularize*weight1
		regularize_1 = regularize_1*weight1

		ratio1 = select_config['regularize_lambda_ratio']
		eps = 1e-12
		thresh1 = 0.05
		if 'regularize_lambda_thresh' in select_config:
			thresh1 = select_config['regularize_lambda_thresh']
		thresh2 = 1e-04
		lambda1 = np.min([L1*ratio1/(group_regularize+eps),thresh1])
		lambda2 = np.min([L1*ratio1/(regularize_1+eps),thresh1])
		print('lambda estimate ',L1,group_regularize,regularize_1,weight1,ratio1,lambda1,lambda2)

		return lambda1, lambda2

	# regularization coefficient estimate
	def _regularize_coef_est_2(self, x, y, num_fold=5, motif_group_list=[], motif_query_local=[], initial_guess=[], pre_data_dict={}, type_id_1=1, select_config={}):

		sel_num1 = 5
		if type_id_1==0:
			reg_model = LassoCV(cv=num_fold,random_state=0,selection='random')
			reg_model.fit(x,y)
			alphas_ = reg_model.alphas_
			mse_ = reg_model.mse_path_
			mean_mse = np.mean(mse_,axis=1)
			id1 = np.argsort(mean_mse)
			alphas_1 = alphas_[id1]
			mean_mse_sort = mean_mse[id1]
			# print('mse ',len(mean_mse_sort),mean_mse_sort)
			# print(alphas_1)
			print('mse ',len(mean_mse_sort),mean_mse_sort[0:5])
			print(alphas_1[0:5])
			lambda_2 = np.mean(alphas_1[0:sel_num1])
			lambda_1 = reg_model.alpha_

			# thresh1 = 0.01
			thresh1 = 0.05
			lambda1 = np.min([lambda_1,thresh1])
			lambda2 = np.min([np.min([lambda_2,lambda1]),thresh1])

		elif type_id_1==1:
			model_type_id = 'Lasso'
			lambda1, Lasso_alpha_sort, score_vec_1 = self._LassoCV(x,y,model_type_id=model_type_id,pre_data_dict=pre_data_dict,select_config=select_config,cv=num_fold,random_state=0)
			lambda2 = np.mean(Lasso_alpha_sort[0:sel_num1])

		elif type_id_1==2:
			# GridSearch
			model_1 = Lasso(fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
			t_vec1 = np.power(10,np.arange(-7,-2,dtype=np.float32))
			t_vec2 = 5*np.power(10,np.arange(-7,-3,dtype=np.float32))
			t_vec3 = np.sort(list(t_vec1)+list(t_vec2))

			param_grid = {'alpha':t_vec3}
			grid_search = GridSearchCV(model_1,scoring=_spearmanr,param_grid=param_grid,cv=num_fold,error_score=0)
			start = time.time()
			grid_search.fit(x,y)
			stop = time.time()
			print('GridSearchCV ',stop-start)
			utility_1.report(grid_search.cv_results_)

			alpha_est = grid_search.best_params_['alpha']
			lambda_1 = alpha_est
			param_search = grid_search
		else:
			# Randomized search
			# n_iter_search = 30
			# n_iter_search = 50
			n_iter_search = select_config['n_iter_search_randomized']
			# n_iter_search = 100
			# selection_type = 'random'
			selection_type = 'cyclic'
			if 'Lasso_selection_type1' in select_config:
				selection_type = select_config['Lasso_selection_type1']
			
			model_1 = Lasso(fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection=selection_type)
			param_dist = {'alpha':loguniform(1e-07,1e-02)}
			random_search = RandomizedSearchCV(model_1,scoring=_spearmanr,param_distributions=param_dist,n_iter=n_iter_search,cv=num_fold,error_score=9)
			start = time.time()
			random_search.fit(x,y)
			stop = time.time()
			print('RandomizedSearchCV ',stop-start,n_iter_search,x.shape,y.shape)
			n_top = sel_num1
			utility_1.report(random_search.cv_results_,n_top=n_top)

			alpha_est = random_search.best_params_['alpha']
			lambda_1 = alpha_est

			param_search = random_search

		if type_id_1 in [2,3]:
			t_results = param_search.cv_results_
			t_list1 = []
			n_top = sel_num1
			for i in range(1, n_top + 1):
				candidates = np.flatnonzero(t_results['rank_test_score'] == i)
				for candidate in candidates:
					# print("Model with rank: {0}".format(i))
					# print("Mean validation score: {0:.3f} (std: {1:.3f})"
					#     .format(results['mean_test_score'][candidate],
					#             results['std_test_score'][candidate]))
					# print("Parameters: {0}".format(results['params'][candidate]))
					# print("")
					t_param_ = t_results['params'][candidate]['alpha']
					t_list1.append(t_param_)
			
			t_param_vec = np.asarray(t_list1)
			lambda_2_pre = np.mean(t_param_vec)
			lambda_2 = np.min([lambda_2_pre,lambda_1])
			lambda1, lambda2 = lambda_1, lambda_2

		print('regularization coefficient estimated 2 ',type_id_1,lambda1,lambda2)

		return lambda1, lambda2

	## model preparation
	def test_model_basic_pre1(self,model_type_id=0,motif_group=[],pre_data_dict={},select_config={}):

		pre_model_dict1 = dict()
		# print('model_type_id ', model_type_id)
		# print('test_model_basic_pre1 configuration: ')
		# print(select_config)

		flag_positive_coef=False
		if 'flag_positive_coef' in select_config:
			flag_positive_coef = select_config['flag_positive_coef']

		fit_intercept = True
		if 'fit_intercept' in select_config:
			fit_intercept = select_config['fit_intercept']

		# print('flag_positive_coef, fit_intercept: ',flag_positive_coef,fit_intercept)

		if model_type_id in ['LR']:
			# LR
			# model_1 = LinearRegression()
			if 'normalize_type_LR' in select_config:
				normalize_type = select_config['normalize_type_LR']
			else:
				normalize_type = False

			# print('normalize type LR ',normalize_type)
			# class sklearn.linear_model.Lasso(alpha=1.0, *, fit_intercept=True, normalize='deprecated', precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
			# model_1 = LinearRegression(fit_intercept=True, normalize=normalize_type, copy_X=True, n_jobs=None, max_iter=5000)
			model_1 = LinearRegression(fit_intercept=fit_intercept, copy_X=True, n_jobs=None, positive=flag_positive_coef)

		elif model_type_id in ['Lasso','Lasso_ori']:
			# Lasso
			alpha = select_config['Lasso_alpha']
			# model_2 = Lasso(alpha=alpha)
			# warm_start_type = False
			warm_start_type = select_config['warm_start_type_Lasso']
			# fit_intercept = True
			# if 'fit_intercept' in select_config:
			# 	fit_intercept = select_config['fit_intercept']
			# if 'normalize_type_Lasso' in select_config:
			# 	normalize_type = select_config['normalize_type_Lasso']
			# else:
			# 	normalize_type = False
			normalize_type=False

			# print('normalize type Lasso ',normalize_type)
			selection_type = 'cyclic'
			if 'Lasso_selection_type1' in select_config:
				selection_type = select_config['Lasso_selection_type1']
			
			# model_1 = Lasso(alpha=alpha,fit_intercept=True, normalize=normalize_type, precompute=False, copy_X=True, max_iter=5000, 
			# 				tol=0.0001, warm_start=warm_start_type, positive=False, random_state=None, selection=selection_type)
			model_1 = Lasso(alpha=alpha,fit_intercept=fit_intercept, precompute=False, copy_X=True, max_iter=5000, 
							tol=0.0001, warm_start=warm_start_type, positive=flag_positive_coef, random_state=None, selection=selection_type)
		
		elif model_type_id in ['ElasticNet']:
			# # Lasso
			# alpha = select_config['Lasso_alpha']
			# ElasticNet
			alpha = select_config['ElasticNet_alpha']
			# model_2 = Lasso(alpha=alpha)
			# warm_start_type = False
			warm_start_type = select_config['warm_start_type_Lasso']
			# if 'normalize_type_Lasso' in select_config:
			# 	normalize_type = select_config['normalize_type_Lasso']
			# else:
			# 	normalize_type = False
			normalize_type=False

			# print('normalize type Lasso ',normalize_type)
			selection_type = 'cyclic'
			if 'Lasso_selection_type1' in select_config:
				selection_type = select_config['Lasso_selection_type1']

			# model_1 = Lasso(alpha=alpha,fit_intercept=True, normalize=normalize_type, precompute=False, copy_X=True, max_iter=5000, 
			# 				tol=0.0001, warm_start=warm_start_type, positive=False, random_state=None, selection=selection_type)
			# model_1 = Lasso(alpha=alpha,fit_intercept=True, precompute=False, copy_X=True, max_iter=5000, 
			# 				tol=0.0001, warm_start=warm_start_type, positive=flag_positive_coef, random_state=None, selection=selection_type)

			l1_ratio = 0.05
			if 'l1_ratio' in select_config:
				l1_ratio = select_config['l1_ratio_ElasticNet']
			# fit_intercept = True
			# if 'fit_intercept' in select_config:
			# 	fit_intercept = select_config['fit_intercept']
			model_1 = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,fit_intercept=fit_intercept,normalize='deprecated',precompute=False,max_iter=5000,
									copy_X=True,tol=0.0001,warm_start=warm_start_type,positive=flag_positive_coef,random_state=None,selection=selection_type)

		elif model_type_id in ['Ridge']:
			# class sklearn.linear_model.Ridge(alpha=1.0, *, fit_intercept=True, normalize='deprecated', copy_X=True, max_iter=None, tol=0.001, solver='auto', positive=False, random_state=None)[source]
			alpha = 1.0
			# flag_positive_coef = False
			# fit_intercept = True
			if 'Ridge_alpha' in select_config:
				alpha = select_config['Ridge_alpha']
			# if 'fit_intercept' in select_config:
			# 	fit_intercept = select_config['fit_intercept']
			# if 'flag_positive_coef' in select_config:
			# 	flag_positive_coef = select_config['flag_positive_coef']
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
			# model_1 = LogisticRegression(penalty='elasticnet',
			# 								dual=False, 
			# 								tol=0.0001, 
			# 								C=1.0, 
			# 								fit_intercept=True, 
			# 								intercept_scaling=1, 
			# 								class_weight=None, 
			# 								random_state=None, 
			# 								solver='saga', 
			# 								max_iter=1000, 
			# 								multi_class='auto', 
			# 								verbose=0, 
			# 								warm_start=False, 
			# 								n_jobs=None, 
			# 								l1_ratio=0.01)

		## tree-based model
		elif model_type_id in ['XGBR','XGBClassifier','RF','RandomForestClassifier']:

			if 'select_config_comp' in select_config:
				select_config_comp = select_config['select_config_comp']
				max_depth, n_estimators = select_config_comp['max_depth'], select_config_comp['n_estimators']
				# if (type_id in [1,2]):
				# 	# max_depth = 20
				# 	max_depth = 10
			else:
				# max_depth, n_estimators = 10, 500
				# max_depth, n_estimators = 10, 200
				max_depth, n_estimators = 7, 100
			# print('max_depth, n_estimators ',max_depth,n_estimators)

			if model_type_id in ['XGBClassifier']:
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
				model_1 = RandomForestRegressor(
								n_jobs=10,
								n_estimators=n_estimators,
								max_depth=max_depth,
								random_state=0)

			else:
				# class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
				n_estimators = 500
				model_1 = RandomForestClassifier(n_estimators=n_estimators,
												max_depth=max_depth,
												n_jobs=10,
												random_state=0)

		elif model_type_id in ['GroupLasso1','group_lasso']:
			## sparse group lasso model
			# group lasso 1
			# groups = pre_data_dict['motif_group']
			groups = motif_group
			select_config_1 = select_config['group_lasso']
			group_reg = select_config_1['group_reg']
			l1_reg = select_config_1['l1_reg']
			model_1 = GroupLasso(
				groups=groups,
				group_reg=group_reg,
				l1_reg=l1_reg,
				frobenius_lipschitz=True,
				scale_reg="group_size",
				fit_intercept=True,
				subsampling_scheme=1,
				supress_warning=True,
				n_iter=select_config_1['max_iter'],
				tol=select_config_1['tol']
				)

			# print('model training')
			# gl_1.fix(x_train,y_train)
			# y_pred_test = gl_1.predict(x_test)

		elif model_type_id in ['sgl']:

			# group lasso 2; sgl
			# groups = pre_data_dict['motif_group']
			groups = motif_group
			select_config_2 = select_config['sgl']
			l1_ratio = select_config_2['l1_ratio']
			alpha = select_config_2['reg_lambda']
			model_1 = gpr.SGL(
					groups=groups, 
					l1_ratio=l1_ratio,
					alpha=alpha,
					fit_intercept=True,
					max_iter=select_config_2['max_iter'], 
					tol=select_config_2['tol']
					)
		else:
			print('model need to be specified ')
			return

		# pre_model_dict1 = {'lr':model_1,'lasso':model_2
		#                   'xgbr':model_xgbr,'random_forest':model_random_forest,
		#                   'sparse_group_lasso_1':gl_1,'sparse_group_lasso_2':sgl}

		return model_1

	# training model
	# def test_motif_peak_estimate_local_sub1(self,model_train,model_type_id,x_train,y_train,sample_weight=[]):
	def test_model_train_basic_pre1(self,model_train,model_type_id,x_train,y_train,sample_weight=[]):

		# print(x_train.shape,y_train.shape)
		# if model_type_id in ['XGBR','XGBClassifier','RF','LogisticRegression']:
		if model_type_id in ['XGBR','XGBClassifier','RF']:
			if len(sample_weight)==0:
				# print('model training ')
				model_train.fit(x_train, y_train)
			else:
				print('sample weight',np.max(sample_weight),np.min(sample_weight))
				model_train.fit(x_train,y_train,sample_weight=sample_weight)

			t_coefficient = []
			t_intercept = []

		elif model_type_id in ['LR_2','LR_3']:
			if model_type_id in ['LR_2']:
				# intercept_flag = True
				# if 'intercept_flag' in select_config:
				# 	intercept_flag = select_config['intercept_flag']

				# if intercept_flag==True:
				# 	x_train1 = sm.add_constant(x_train)
				# else:
				# 	x_train1 = x_train
				model_1 = sm.OLS(y_train,x_train)
				model_train = model_1.fit()

				# df_1 = model_train.summary2().tables[0]
				# df1_1 = df_1.loc[:,[0,1]]
				# df1_1.index = np.asarray(df1_1[0])
				# r2_score = float(df1_1.loc['R-squared:',1])
				# df1_2 = df_1.loc[:,[2,3]]
				# df1_2.index = np.asarray(df1_2[2])
				# r2_adj = float(df1_2.loc['Adj. R-squared:',3])

				df_2 = model_train.summary2().tables[1]
				column_1, column_2 = 'Coef.','P>|t|'
				# coef_ = df_2[column_1]
				# pvalue = df_2[column_2]
				df_query1 = df_2.loc[:,[column_1,column_2]]
				df_query1 = df_query1.rename(columns={column_1:'coef',column_2:'pval'})
				# df_query1['response'] = response_query1
				# df_query1['r2'] = r2_score
				# df_query1['adj_r2'] = r2_adj
				query_idvec = df_query1.index
				# if intercept_flag==True:
				if ('const' in query_idvec):
					# feature_name_vec = query_idvec[1:]
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

	def test_check_param_1(self,params,bound1,bound2):

		# bound1 = 1e05
		# bound2 = -1e05
		b1 = np.where(params>bound1)[0]
		if len(b1)>0:
			print('params out of bound ',params[b1][0:2])
			# params[b1] = bound1
			params[b1] = 0
		b2 = np.where(params<bound2)[0]
		if len(b2)>0:
			print('params out of bound ',params[b2][0:2])
			# params[b2] = bound2
			params[b2] = 0

		return params

	def _check_params(self,params,upper_bound,lower_bound):

		small_eps = 1e-3
		min1, max1 = lower_bound, upper_bound
		# param1 = params[1:]
		flag_1 = (params>=min1-small_eps)&(params<=max1+small_eps)
		# print(flag_1)
		# print(param1)
		flag1 = (np.sum(flag_1)==len(params))
		if flag1==False:
			print(params)
			flag_1 = np.asarray(flag_1)
			id1 = np.where(flag_1==0)[0]
			print(params[id1], len(id1), len(params))

		return flag1

	## tf motif coefficients estimate
	def train_basic(self,gene_query_id,feature_type_id,model_type_id,sample_id_vec,score_query_type,score_list=[],score_list_query_id=-1,peak_read=[],rna_exprs=[],scale_type_id=0,scale_type_vec=[],feature_imp_est=1,regularize_est=1,iter_id=0,sample_weight=[],LR_compare=0,save_mode=1,output_file_path='',select_config={}):
		# gene_query_id = self.gene_query_id
		if len(peak_read)==0:
			peak_read = self.peak_read
		
		if len(rna_exprs)==0:
			rna_exprs = self.rna_exprs

		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]
		self.sample_id = sample_id

		x, y, feature_query_vec = self.test_feature_query_pre1(gene_query_id=gene_query_id,feature_type_id=feature_type_id,
												score_query_type=score_query_type,
												peak_read=peak_read,rna_exprs=rna_exprs,
												score_list=score_list,score_list_query_id=score_list_query_id,
												scale_type_id=scale_type_id,scale_type_vec=scale_type_vec,
												select_config=select_config)

		sample_id_train, sample_id_valid, sample_id_test = sample_id_vec
		alpha_vec, model_train_pre, x, y, feature_imp_est_vec, score_pred_vec = self.test_feature_coef_est_pre1(x,y,model_type_id=model_type_id,
																								sample_id_vec=sample_id_vec,
																								iter_id=iter_id,sample_weight=sample_weight,
																								feature_imp_est=feature_imp_est,
																								regularize_est=regularize_est,
																								LR_compare=LR_compare,
																								save_mode=save_mode,
																								output_file_path=output_file_path,
																								select_config=select_config)

		self.alpha_ = alpha_vec
		self.model_train_ = model_train_pre
		self.data_vec_ = [x,y]
		self.feature_imp_est_ = feature_imp_est_vec
		self.feature_query_id = feature_query_vec
		self.dict_feature_query_[gene_query_id] = feature_query_vec
		self.score_pred_vec = score_pred_vec
		score_pred = score_pred_vec[0]
		self.score_pred_ = score_pred

		return True

	## prediction performance evaluation for regression model
	def score_2a_1(self, y, y_predicted):

		score1 = mean_squared_error(y, y_predicted)
		score2 = pearsonr(y, y_predicted)
		score3 = explained_variance_score(y, y_predicted)
		score4 = mean_absolute_error(y, y_predicted)
		score5 = median_absolute_error(y, y_predicted)
		score6 = r2_score(y, y_predicted)
		score7, pvalue = spearmanr(y,y_predicted)
		t_mutual_info = mutual_info_regression(y[:,np.newaxis], y_predicted, discrete_features=False, n_neighbors=5, copy=True, random_state=0)
		t_mutual_info = t_mutual_info[0]

		# vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6]
		vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6, score7, pvalue, t_mutual_info]

		field_query_1 = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2','mutual_info']
		df_score_pred = pd.Series(index=field_query_1,data=vec1,dtype=np.float32)
			
		# return vec1
		return df_score_pred

	## prediction performance evaluation for binary classification model
	def score_function_multiclass1(self,y_test, y_pred, y_proba):

		auc = roc_auc_score(y_test,y_proba)
		aupr = average_precision_score(y_test,y_proba)
		precision = precision_score(y_test,y_pred)
		recall = recall_score(y_test,y_pred)
		accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))
		F1 = 2*precision*recall/(precision+recall)

		# print(auc,aupr,precision,recall)
		
		return accuracy, auc, aupr, precision, recall, F1

	## prediction performance evaluation for multi-class classification model
	def score_function_multiclass2(self,y_test, y_pred, y_proba, average='macro'):

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

		# print(auc,aupr,precision,recall)
		
		# return accuracy, auc, aupr, precision, recall, F1
		# return accuracy, precision, recall, F1
		return df_score_pred

	## query coef value
	def test_query_ceof_1(self,param,feature_name,num_class,response_variable_name='1',query_idvec=[],df_coef_query=[],select_config={}):

		alpha_query1 = param[0]
		intercept_ = param[1]
		if len(alpha_query1)>0:
			if len(query_idvec)==0:
				feature_name = pd.Index(feature_name).difference(['const'],sort=False)
				feature_query_vec_coef = feature_name
				query_idvec = list(feature_query_vec_coef)+['alpha0']

			if num_class<=2:
				# print('alpha_query1: ',alpha_query1)
				# print('intercept_: ',intercept_)
				# if num_class==2:
				# 	alpha_query = list(alpha_query1[0])+intercept_

				# else:
				# 	alpha_query = list(alpha_query1)+[intercept_]
				
				alpha_query = np.hstack((alpha_query1,intercept_[:,np.newaxis]))
				alpha_query = alpha_query.T
				# print('alpha_query: ',alpha_query.shape)
				# print(alpha_query)
				# print(num_class)
				if len(df_coef_query)==0:
					# df_coef_query = pd.DataFrame(index=[response_variable_name],columns=feature_query_vec_coef,dtype=np.float32)
					# df_coef_query = pd.Series(index=query_idvec,data=alpha_query,dtype=np.float32)
					# df_coef_query.name = response_variable_name
					df_coef_query = pd.DataFrame(index=query_idvec,columns=[response_variable_name],data=np.asarray(alpha_query),dtype=np.float32)
				else:
					# df_coef_query.loc[feature_query_vec_coef,response_variable_name] = alpha_query
					df_coef_query.loc[query_idvec,response_variable_name] = alpha_query
			else:
				alpha_query = np.hstack((alpha_query1,intercept_[:,np.newaxis]))
				alpha_query = alpha_query.T
				# df_coef_query = pd.DataFrame(index=np.arange(num_class),columns=feature_query_vec_coef,data=np.asarray(alpha_query),dtype=np.float32)
				df_coef_query = pd.DataFrame(index=query_idvec,columns=np.arange(num_class),data=np.asarray(alpha_query),dtype=np.float32)

			return df_coef_query

	## query pvalue
	def test_query_coef_pval_1(self,model_train,model_type_id=0,select_config={}):

		if model_type_id==0:
			df1 = model_train.summary2().tables[1]
			column_1, column_2 = 'Coef.','P>|t|'
			# coef_ = df_2[column_1]
			# pvalue = df_2[column_2]
			df_query1 = df1.loc[:,[column_1,column_2]]
			df_query1 = df_query1.rename(columns={column_1:'coef',column_2:'pval'})
			column_vec_1 = ['coef','pval']
			# y_pred = model_1.predict(X)
			# list1.append([r2_score,r2_adj])
			# df_query1['response'] = response_query1
			# df_query1['r2'] = r2_score
			# df_query1['adj_r2'] = r2_adj

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
				
			# coef_query = np.asarray(df_query1.loc[query_idvec_2,'coef'])
			# pval_query = np.asarray(df_query1.loc[query_idvec_2,'pval'])
			df_coef_pval_ = pd.DataFrame(index=query_idvec,columns=column_vec_1,data=0)
			df_coef_pval_.loc[query_id_1,column_vec_1] = np.asarray(df_query1.loc[query_idvec_2,column_vec_1])
			
			return df_coef_pval_

	## prediction and feature importance estimation
	def test_model_pred_explain_1(self,model_train,x_test,y_test,sample_id_test_query=[],y_pred=[],y_pred_proba=[],
										x_train=[],y_train=[],response_variable_name='',df_coef_query=[],df_coef_pval_=[],
										fold_id=-1,type_id_model=0,model_explain=1,model_save_filename='',
										output_mode=1,save_mode=0,verbose=0,select_config={}):

		# if 'model_type_id1' in select_config:
		# 	model_type_id1 = select_config['model_type_id1']
		# 	if model_type_id1 in ['LR_2']:
		# 		intercept_flag = True
		# 		if 'intercept_flag' in select_config:
		# 			intercept_flag = select_config['intercept_flag']
		# 		if intercept_flag==True:
		# 			x_test = sm.add_constant(x_test)
		y_test_pred = model_train.predict(x_test)
		# print('x_test, y_test, y_test_pred ',x_test.shape,y_test.shape,y_test_pred.shape)
		# print('y_test_pred ',y_test_pred.shape)
		# print(y_test[0:5],y_test_pred[0:5])
		y_test_proba = []
		if type_id_model==0:
			## regression model
			y_test_pred = np.ravel(y_test_pred)
			score_1 = self.score_2a_1(y_test,y_test_pred)
		else:
			## classification model
			y_test_proba = model_train.predict_proba(x_test)
			select_config1 = select_config['select_config1']
			average_type = select_config1['average_type']
			if verbose>0:
				print('average_type: %s'%(average_type))
			score_1 = self.score_function_multiclass2(y_test,y_test_pred,y_test_proba,average=average_type)
			if output_mode>0:
				y_pred_proba.loc[sample_id_test_query,:] = y_test_proba
				# print('y_test_proba ',y_test_proba.shape)
				# print(y_test_proba[0:1])

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

			dict_feature_imp_, dict_interaction_, filename_dict1, filename_dict2, save_filename_dict2 = t_vec_1
			dict_query1 = dict_feature_imp_[model_type_id2]
			# df_imp_1 = dict_query1['imp'].loc[:,['shap_value']]
			# df_imp_scaled_1 = dict_query1['imp_scaled'].loc[:,['shap_value']]

		# model_type_name = select_config['model_type_id1']
		model_type_name = select_config['model_type_id_train']
		print('train, model_type_name: ',model_type_name)
		if model_type_name in ['LR','Lasso','LassoCV','ElasticNet','LogisticRegression','LR_2']:
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

			# if type_id_model==0:
			# 	# if flag_model_explain>0:
			# 	# 	df_imp_1 = dict_query1['imp']
			# 	# 	df_imp_scaled_1 = dict_query1['imp_scaled']
			# 		# df_imp_1['coef'] = np.asarray(coef_query)
			# 		# df_imp_1['intercept_'] = np.asarray(intercept_query)
			# 		# df2 = pd.DataFrame(index=['alpha0'],columns=df_imp_1.index,data=0)
			# 		# df_imp_1 = pd.concat([df_imp_1,df2],axis=0,join='outer',ignore_index=False)
			# 		# df_imp_1['coef'] = list(coef_query)+[intercept_query]
			# 	# else:
			# 	# 	df_imp_1 = pd.DataFrame(index=feature_name,columns=['coef'],data=np.asarray(coef_query))
			# 	# 	df_imp_1['intercept_'] = np.asarray(intercept_query)
			# 	coef_value = list(coef_query)+[intercept_query]
			# 	df_coef_query = pd.DataFrame(index=list(feature_name)+['alpha0'],columns=['coef'],data=np.asarray(coef_value))
			# else:
			# 	num_class = y_test_pred.shape[1]
			# 	alpha_query = np.hstack((coef_query,intercept_[:,np.newaxis]))
			# 	alpha_query = alpha_query.T
			# 	df_coef_query = pd.DataFrame(index=list(feature_name)+['alpha0'],columns=np.arange(num_class),data=np.asarray(alpha_query))
			# 	# dict_query1.update({'imp':df_imp_1,'imp_scaled':df_imp_scaled_1,'coef_':df_coef_query})
			
			param_vec = [coef_query, intercept_query]
			if type_id_model==0:
				num_class = 1
			else:
				num_class = y_test_proba.shape[1]

			df_coef_query = self.test_query_ceof_1(param=param_vec,feature_name=feature_name,num_class=num_class,response_variable_name=response_variable_name,query_idvec=[],df_coef_query=df_coef_query,select_config=select_config)
			dict_query1.update({'coef':df_coef_query})
		
		# return score_1, df_imp_1, df_imp_scaled_1
		# return score_1, y_test_pred, y_test_proba, df_imp_1, df_imp_scaled_1
		return score_1, y_test_pred, y_test_proba, dict_query1

	## query feature importance mean value
	def test_query_feature_mean_1(self,data=[],response_variable_name='',column_id_query='feature_name',column_vec_query=['fold_id'],type_id_1=0,verbose=0,select_config={}):

		list1_imp = data
		df_imp = pd.concat(list1_imp,axis=0,join='outer',ignore_index=False)
		# df_imp_scaled = pd.concat(list1_imp_scale,axis=0,join='outer',ignore_index=False)
		# df_imp.loc[:,'feature_name'] = np.asarray(df_imp.index)
		df_imp.loc[:,column_id_query] = np.asarray(df_imp.index)
		# df_imp_scaled.loc[:,'feature_name'] = np.asarray(df_imp_scaled.index)
		if verbose>0:
			# print('df_imp, gene %s'%(response_variable_name))
			print('df_imp, response_variable %s'%(response_variable_name))
			print(df_imp)
			# print('df_imp_scaled, gene %s'%(response_variable_name))
			# print('df_imp_scaled, response_variable %s'%(response_variable_name))
			# print(df_imp_scaled)

		# df1 = list1_imp[0]
		# column_vec_1 = df_imp.columns.difference(['fold_id'],sort=False)
		# df_1 = df_imp.loc[:,column_vec_1].groupby(by='feature_name').mean()
		if type_id_1==0:
			column_vec_1 = df_imp.columns.difference(column_vec_query,sort=False)
		else:
			column_vec_1 = df_imp.columns.intersection(column_vec_query,sort=False)

		df_1 = df_imp.loc[:,column_vec_1].groupby(by=column_id_query).mean()
		df_imp1_mean = pd.DataFrame(index=df_1.index,columns=df_1.columns,data=np.asarray(df_1))

		return df_imp, df_imp1_mean

	## query tf feature importance estimate for gene query; control strength initialization
	# input: feature importance estimate filename
	# output: initial control strength estimate
	def test_optimize_pre1_basic2(self,model_pre,
										x_train,
										y_train,
										response_variable_name,
										x_train_feature2=[],
										sample_weight=[],
										dict_query={},
										df_coef_query=[],
										df_pred_query=[],
										model_type_vec=[],
										model_type_idvec=[],
										filename_annot_vec=[],
										dict_score_query={},
										score_type_idvec=[],
										pre_data_dict={},
										type_id_train=0,
										type_id_model=0,
										save_model_train=1,
										model_path_1='',
										save_mode=0,
										output_file_path='',
										output_filename='',
										filename_prefix_save='',
										filename_save_annot='',
										verbose=0,
										select_config={}):
		
		# y_train1 = y_mtx.loc[sample_id1,:]
		# the tfs with expression in the cell
		# tf_expr = meta_exprs_2.loc[sample_id1,motif_query_vec_pre1]
		# motif_query_vec_expr = motif_query_vec_pre1[tf_expr>0]
		# motif_query_num1 = len(motif_query_vec_pre1)
		# motif_query_num2 = len(motif_query_vec_expr)
		# print('motif_query_vec_pre1, motif_query_vec_expr ',i1,sample_id1,motif_query_num1,motif_query_num2)

		x_train1, y_train1 = x_train, y_train
		x_train1_feature2 = x_train_feature2
		sample_idvec_pre1 = x_train1.index
		feature_query_vec_coef = x_train1.columns.copy()

		# sample_id_vec=[]
		num_fold = select_config['num_fold']
		sample_idvec_query = select_config['sample_idvec_train']
		select_config1 = select_config['select_config1']
		# train_valid_mode = select_config['train_valid_mode_2']
		train_valid_mode = select_config['train_valid_mode_1']
		pre_data_dict_1 = pre_data_dict
		
		# list1 = []
		# list2 = []
		# model_type_num = len(model_type_idvec)
		# dict_query_1 = dict_query
		# filename_save_annot = select_config['filename_save_annot']
		# run_id = select_config['run_id']
		# select_config1 = select_config['select_config1'] # the configuration of models
		if type_id_model==1:
			num_class = len(np.unique(y_train1))
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
				# average_type = 'weighted'
				# average_type = 'samples'
				type_id1 = 1
				select_config1.update({'type_classifer_xbgboost':type_id1})

			if 'average_type' in select_config1:
				average_type = select_config1['average_type']
			else:
				select_config1.update({'average_type':average_type})

		list1 = []
		list2 = []
		model_type_num = len(model_type_idvec)
		dict_query_1 = dict_query
		filename_save_annot = select_config['filename_save_annot_local']
		# filename_save_annot1 = select_config['filename_save_annot1']
		run_id = select_config['run_id']
		model_path_1 = output_file_path
		# flag_peak_1=0
		flag_feature2=0
		if len(x_train_feature2)>0:
			flag_feature2=1

		np.random.seed(0)
		for i1 in range(model_type_num):
			model_type_id1 = model_type_idvec[i1]
			print('model_type_id1 ',model_type_id1)
			df_coef_query, df_pred_query, df_score_1 = [], [], []
			df_coef_pval_ = []
			dict_query_1[model_type_id1] = dict()
			# if len(dict_query_1)>0:
			# 	dict_query_pre1 = dict_query_1[model_type_id1]
			# 	key_vec = list(dict_query_pre1.keys())
			# 	print(key_vec)
			# 	df_coef_query, df_pred_query = dict_query_pre1['coef'], dict_query_pre1['pred_cv']
			# 	# df_coef_query, df_pred_query = dict_query_pre1['coef'], dict_query_pre1['pred']

			# df_pred_query_feature2,df_pred_proba_feature2,df_score_2 = [], [], []
			y_pred1 = pd.Series(index=sample_idvec_pre1,data=0,dtype=np.float32)
			# output_mode = 1
			output_mode = 0
			if type_id_model==1:
				# num_class = select_config['num_class']
				y_pred1_proba = pd.DataFrame(index=sample_idvec_pre1,columns=range(num_class),data=0,dtype=np.float32)
			else:
				num_class = 1
				y_pred1_proba = []

			# y_pred1_feature2 = pd.Series(index=sample_idvec_pre1,data=0,dtype=np.float32)
			# y_pred1_proba_feature2 = pd.DataFrame(index=sample_idvec_pre1,columns=range(num_class),data=0,dtype=np.float32)
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

			## train with cross validation for performance evaluation
			list1_imp, list1_imp_scale = [], []
			# list2_imp, list2_imp_scale = [], []
			df_pred_query = []
			df_pred_proba = []
			list_feature_imp = []

			if num_fold>0:
				for fold_id in range(num_fold):
					sample_idvec_1 = sample_idvec_query[fold_id]
					sample_id_train_query, sample_id_valid_query, sample_id_test_query = sample_idvec_1

					x_train2, y_train2 = x_train1.loc[sample_id_train_query,:], y_train1.loc[sample_id_train_query]
					x_test2, y_test2 = x_train1.loc[sample_id_test_query,:], y_train1.loc[sample_id_test_query]
					# feature_name = x_train2.columns
					if flag_model_load==0:
						model_1 = self.test_model_basic_pre1(model_type_id=model_type_id1,
																pre_data_dict=pre_data_dict_1,
																select_config=select_config1)

						model_1, param1 = self.test_model_train_basic_pre1(model_train=model_1,model_type_id=model_type_id1,
																			x_train=x_train2,y_train=y_train2,
																			sample_weight=sample_weight)
						if save_model_train==2:
							## save models from the cross validation
							save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
							# pickle.dump(model_1, open(save_filename, 'wb'))
							with open(save_filename,'wb') as output_file:
								pickle.dump(model_1,output_file)
							select_config.update({'model_save_filename':save_filename})
					else:
						model_save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
						with open(model_save_filename, 'rb') as fid:
							model_1 = pickle.load(fid)
						print('model weights loaded ',model_save_filename)

					# save_mode_2 = (save_model_train==2)
					save_mode_2 = 0
					model_save_filename = ''
					if 'model_save_filename' in select_config:
						model_save_filename = select_config['model_save_filename']
					list_query1 = self.test_model_pred_explain_1(model_train=model_1,
																	x_test=x_test2,
																	y_test=y_test2,
																	sample_id_test_query=[],
																	y_pred=[],
																	y_pred_proba=[],
																	x_train=x_train2,y_train=y_train2,
																	fold_id=fold_id,
																	type_id_model=type_id_model,
																	model_explain=flag_model_explain,
																	model_save_filename=model_save_filename,
																	output_mode=output_mode,
																	save_mode=save_mode_2,
																	verbose=0,
																	select_config=select_config)

					# dict_feature_imp_[model_type_id] = {'imp':df_imp_,'imp_scaled':df_imp_scaled}
					# score_1, y_test2_pred, y_test2_proba, df_imp_1, df_imp_scaled_1 = list_query1
					score_1, y_test2_pred, y_test2_proba, dict_query1 = list_query1

					y_pred1.loc[sample_id_test_query] = y_test2_pred
					if type_id_model==1:
						# the regression model
						y_pred1_proba.loc[sample_id_test_query,:] = y_test2_proba
					list1.append(score_1)

					if verbose>0:
						print('fold id, x_train2, y_train2, x_test2, y_test2 ',fold_id,x_train2.shape,y_train2.shape,x_test2.shape,y_test2.shape)
						print(score_1,fold_id,num_fold)

					if len(dict_query1)>0:
						if 'imp' in dict_query1:
							df_imp_1 = dict_query1['imp']
							df_imp_1['fold_id'] = fold_id
							list1_imp.append(df_imp_1)

						if 'imp_scaled' in dict_query1:
							df_imp_scaled_1 = dict_query1['imp_scaled']
							df_imp_scaled_1['fold_id'] = fold_id
							list1_imp_scale.append(df_imp_scaled_1)

						# df_imp_1, df_imp_scaled_1 = dict_query1['imp'], dict_query1['imp_scaled']
						# df_imp_1['fold_id'] = fold_id
						# df_imp_scaled_1['fold_id'] = fold_id
						# list1_imp.append(df_imp_1) # df_imp_1: shape: feature_num by column_num (columns: shap_value, imp2, coef, intercept_)
						# list1_imp_scale.append(df_imp_scaled_1)				

				if type_id_model==0:
					## regression model
					score_2 = self.score_2a_1(y_train1,y_pred1)
				else:
					## classification model
					# score_2 = utility_1.score_function_multiclass2(y_train1,y_pred1,y_pred1_proba,average=average_type)
					score_2 = self.score_function_multiclass2(y_train1,y_pred1,y_pred1_proba,average=average_type)

				list1.append(score_2)
				if verbose>0:
					# print(score_2,sample_id1[0:2],x_train1.shape,y_train1.shape)
					print(score_2,sample_idvec_pre1[0:2],x_train1.shape,y_train1.shape)

				df_score_1 = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				df_score_1 = df_score_1.T
				# df_score_1['sample_id'] = [sample_id1]*df_score_1.shape[0]
				df_score_1['fold_id'] = np.asarray(df_score_1.index)

				list_feature_imp = []
				df_imp, df_imp1_mean = [],[]
				df_imp_scaled, df_imp1_scaled_mean = [],[]
				column_id_query = 'feature_name'
				column_vec_2 = ['fold_id']
				if (flag_model_explain>0) and (len(list1_imp)>0):
					# df_imp = pd.concat(list1_imp,axis=0,join='outer',ignore_index=False)
					# # df_imp_scaled = pd.concat(list1_imp_scale,axis=0,join='outer',ignore_index=False)
					# df_imp.loc[:,'feature_name'] = np.asarray(df_imp.index)
					# # df_imp_scaled.loc[:,'feature_name'] = np.asarray(df_imp_scaled.index)
					# if verbose>0:
					# 	# print('df_imp, gene %s'%(response_variable_name))
					# 	print('df_imp, response_variable %s'%(response_variable_name))
					# 	print(df_imp)
					# 	# print('df_imp_scaled, gene %s'%(response_variable_name))
					# 	# print('df_imp_scaled, response_variable %s'%(response_variable_name))
					# 	# print(df_imp_scaled)

					# # df1 = list1_imp[0]
					# column_vec_1 = df_imp.columns.difference(['fold_id'],sort=False)
					# df_1 = df_imp.loc[:,column_vec_1].groupby(by='feature_name').mean()
					# df_imp1_mean = pd.DataFrame(index=df_1.index,columns=df_1.columns,data=np.asarray(df_1))xw
					df_imp, df_imp1_mean = self.test_query_feature_mean_1(data=list1_imp,response_variable_name=response_variable_name,column_id_query=column_id_query,column_vec_query=column_vec_2,type_id_1=0,verbose=verbose,select_config=select_config)

				if (flag_model_explain>0) and (len(list1_imp_scale)>0):
					# df_imp = pd.concat(list1_imp,axis=0,join='outer',ignore_index=False)
					# df_imp_scaled = pd.concat(list1_imp_scale,axis=0,join='outer',ignore_index=False)
					# # df_imp.loc[:,'feature_name'] = np.asarray(df_imp.index)
					# df_imp_scaled.loc[:,'feature_name'] = np.asarray(df_imp_scaled.index)
					# if verbose>0:
					# 	print('df_imp_scaled, response_variable %s'%(response_variable_name))
					# 	print(df_imp_scaled)

					# column_vec_2 = df_imp_scaled.columns.difference(['fold_id'],sort=False)
					# df_2 = df_imp_scaled.loc[:,column_vec_2].groupby(by='feature_name').mean()
					# df_imp1_scaled_mean = pd.DataFrame(index=df_2.index,columns=df_2.columns,data=np.asarray(df_2))
					df_imp_scaled, df_imp1_scaled_mean = self.test_query_feature_mean_1(data=list1_imp_scale,response_variable_name=response_variable_name,column_id_query=column_id_query,column_vec_query=column_vec_2,type_id_1=0,verbose=verbose,select_config=select_config)

				list_feature_imp.append([df_imp,df_imp_scaled])
				list_feature_imp.append([df_imp1_mean,df_imp1_scaled_mean])
				
				# if len(df_pred_query)>0:
				# 	print('y_pred1 ',y_pred1.shape,response_variable_name)
				# 	print(y_pred1)
				# 	print('df_pred_query ',df_pred_query.shape)
				# 	print(df_pred_query)
				# 	df_pred_query.loc[:,response_variable_name] = y_pred1
				# else:
				# 	print('y_pred1 ',y_pred1.shape,response_variable_name)
				# 	df_pred_query = y_pred1
				
				df_pred_query = y_pred1
				df_pred_proba = y_pred1_proba

			# train on the combined data for coefficient estimation
			if train_valid_mode>0:
				if flag_model_load==0:
					model_2 = self.test_model_basic_pre1(model_type_id=model_type_id1,
																pre_data_dict=pre_data_dict_1,
																select_config=select_config1)

					model_2, param2 = self.test_model_train_basic_pre1(model_2,
																		model_type_id1,
																		x_train1,
																		y_train1,
																		sample_weight=sample_weight)

					if save_model_train>=1:
						save_filename = '%s/test_model_%s.h5'%(model_path_1,filename_save_annot)
						# pickle.dump(model_1, open(save_filename, 'wb'))
						with open(save_filename,'wb') as output_file:
							pickle.dump(model_2,output_file)
						model_save_filename = save_filename
				else:
					model_save_filename = '%s/test_model_%s.h5'%(model_path_1,filename_save_annot)
					with open(model_save_filename, 'rb') as fid:
						model_2 = pickle.load(fid)
						if model_type_id1 in ['LR','Lasso','ElasticNet','LogisticRegression']:
							try:
								param2 = [model_2.coef_, model_2.intercept_]
							except Exception as error:
								print('error! ',error)
					print('model weights loaded ',model_save_filename)

				# if model_type_id1 in ['LR_2']:
				# 	intercept_flag = True
				# 	if 'intercept_flag' in select_config:
				# 		intercept_flag = select_config['intercept_flag']
				# 	if intercept_flag==True:
				# 		x_train1 = sm.add_constant(x_train1)
				# y_pred = model_2.predict(x_train1)
				# y_proba = []
				# if type_id_model==1:
				# 	y_proba = model_2.predict_proba(x_train1)
				save_mode_2 = 0
				list_query2 = self.test_model_pred_explain_1(model_train=model_2,
																x_test=x_train1,
																y_test=y_train1,
																sample_id_test_query=[],
																y_pred=[],
																y_pred_proba=[],
																x_train=x_train1,y_train=y_train1,
																fold_id=-1,
																type_id_model=type_id_model,
																model_explain=flag_model_explain,
																model_save_filename=model_save_filename,
																output_mode=output_mode,
																save_mode=save_mode_2,
																verbose=0,
																select_config=select_config)

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

				# alpha_query = param1[0].copy()
				if len(param2)>0:
					alpha_query1 = param2[0]
					intercept_ = param2[1]
					# feature_name = pd.Index(feature_query_vec_coef).difference(['alpha0','const'],sort=False)
					# query_idvec = list(feature_query_vec_coef)+['alpha0']
					# alpha_query = list(alpha_query1)+[intercept_]
					# if len(alpha_query)>0:
					# 	if len(df_coef_query)==0:
					# 		# df_coef_query = pd.DataFrame(index=[response_variable_name],columns=feature_query_vec_coef,dtype=np.float32)
					# 		df_coef_query = pd.Series(index=query_idvec,data=alpha_query,dtype=np.float32)
					# 		df_coef_query.name = response_variable_name
					# 	else:
					# 		# df_coef_query.loc[feature_query_vec_coef,response_variable_name] = alpha_query
					# 		df_coef_query.loc[query_idvec,response_variable_name] = alpha_query
					
					if len(alpha_query1)>0:
						query_idvec = list(feature_query_vec_coef)+['alpha0']
						if num_class<2:
							alpha_query = list(alpha_query1)+[intercept_]
							if len(df_coef_query)==0:
								# df_coef_query = pd.DataFrame(index=[response_variable_name],columns=feature_query_vec_coef,dtype=np.float32)
								df_coef_query = pd.Series(index=query_idvec,data=alpha_query,dtype=np.float32)
								df_coef_query.name = response_variable_name
							else:
								# df_coef_query.loc[feature_query_vec_coef,response_variable_name] = alpha_query
								df_coef_query.loc[query_idvec,response_variable_name] = alpha_query
						else:
							alpha_query = np.hstack((alpha_query1,intercept_[:,np.newaxis]))
							alpha_query = alpha_query.T
							# df_coef_query = pd.DataFrame(index=np.arange(num_class),columns=feature_query_vec_coef,data=np.asarray(alpha_query),dtype=np.float32)
							df_coef_query = pd.DataFrame(index=query_idvec,columns=np.arange(num_class),data=np.asarray(alpha_query),dtype=np.float32)

					if model_type_id1 in ['LR_2']:
						df_coef_pval_1 = self.test_query_coef_pval_1(model_train=model_2,model_type_id=0,select_config=select_config)
						pval_query = np.asarray(df_coef_pval_1.loc[query_idvec,'pval'])
						if len(df_coef_pval_)==0:
							# df_coef_query = pd.DataFrame(index=[response_variable_name],columns=feature_query_vec_coef,dtype=np.float32)
							# df_coef_pval_ = pd.Series(index=query_idvec,data=pval_query,dtype=np.float32)
							# df_coef_pval_.name = response_variable_name
							df_coef_pval_ = pd.DataFrame(index=query_idvec,columns=[response_variable_name],data=pval_query,dtype=np.float32)
						else:
							df_coef_pval_.loc[query_idvec,response_variable_name] = pval_query
						
						dict_query_1[model_type_id1].update({'pval':df_coef_pval_})

			# if (train_valid_mode>0) and (save_mode>0) and (output_filename!='') and (len(df_coef_query)>0):
			# if (train_valid_mode>0) and (save_mode==2) and (output_filename!='') and (len(df_coef_query)>0):
			# 	# output_file_path = input_file_path
			# 	# # output_filename = '%s/%s.tf_query.pre1.%d.txt'%(output_file_path,filename_prefix_1,run_id)
			# 	# output_filename = '%s/%s.tf_query.pre1.%s.txt'%(output_file_path,filename_prefix_1,filename_annot1)
			# 	df_coef_query.to_csv(output_filename,sep='\t',float_format='%.6E')
			# 	print('df_coef_query ',df_coef_query.shape)

			# df_feature2_query_list = [df_pred_query_feature2,df_pred_proba_feature2,df_score_2]
			# dict_query_1[model_type_id1] = [df_coef_query, df_pred_query, df_pred_proba, y_pred, y_proba, df_feature2_query_list, list_feature_imp]
			# dict_query_1[model_type_id1] = [df_coef_query, df_pred_query, df_pred_proba, y_pred, y_proba, list_feature_imp]
			dict_query_1[model_type_id1].update({'coef':df_coef_query, 
													'pred_cv':df_pred_query, 
													'pred_proba_cv':df_pred_proba, 
													'pred':y_pred, 
													'pred_proba':y_proba, 
													'feature_imp':list_feature_imp})

		# return df_coef_query, df_pred_query, df_score_1
		return dict_query_1, df_score_1

	# regression coefficient estimation
	def test_optimize_pre1_basic2_unit1(self,x_train,y_train,x_test=[],y_test=[],sample_weight=[],model_type_id='',fold_id=-1,type_id_model=0,pre_data_dict={},
										flag_model_load=0,flag_model_explain=0,output_mode=0,select_config1={},
										save_mode=0,save_mode_2=0,save_model_train=1,model_path='',output_file_path='',filename_prefix_save='',filename_save_annot='',output_filename='',verbose=0,select_config={}):

		model_path_1 = model_path
		model_save_filename = ''
		# print('test_optimize_pre1_basic2_unit1')
		# print('save_model_train: ',save_model_train)
		if flag_model_load==0:
			model_1 = self.test_model_basic_pre1(model_type_id=model_type_id,
													pre_data_dict=pre_data_dict,
													select_config=select_config1)

			model_1, param1 = self.test_model_train_basic_pre1(model_1,
																model_type_id,
																x_train,
																y_train,
																sample_weight=sample_weight)

			if save_model_train>0:
				## save models from the cross validation
				save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
				# pickle.dump(model_1, open(save_filename, 'wb'))
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
														x_train=x_train,y_train=y_train,
														fold_id=fold_id,
														type_id_model=type_id_model,
														model_explain=flag_model_explain,
														model_save_filename=model_save_filename,
														output_mode=output_mode,
														save_mode=save_mode_2,
														verbose=0,
														select_config=select_config)

		# dict_feature_imp_[model_type_id] = {'imp':df_imp_,'imp_scaled':df_imp_scaled}
		# score_1, y_test2_pred, y_test2_proba, df_imp_1, df_imp_scaled_1 = list_query1
		# score_1, y_test2_pred, y_test2_proba, dict_query1 = list_query1
		# df_imp_1, df_imp_scaled_1 = dict_query1['imp'], dict_query1['imp_scaled']

		return model_1, param1, list_query1

	## query tf feature importance estimate for gene query; control strength initialization
	# input: feature importance estimate filename
	# output: initial control strength estimate
	def test_optimize_pre1_basic2_1(self,model_pre,x_train,y_train,response_variable_name,feature_name=[],
										x_train_feature2=[],sample_weight=[],dict_query={},df_coef_query=[],df_pred_query=[],
										model_type_vec=[],model_type_idvec=[],filename_annot_vec=[],
										dict_score_query={},score_type_idvec=[],pre_data_dict={},
										type_id_train=0,type_id_model=0,num_class=1,
										save_mode=0,save_model_train=1,model_path_1='',output_file_path='',output_filename='',
										filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
		
		# y_train1 = y_mtx.loc[sample_id1,:]
		# the tfs with expression in the cell
		# tf_expr = meta_exprs_2.loc[sample_id1,motif_query_vec_pre1]
		# motif_query_vec_expr = motif_query_vec_pre1[tf_expr>0]
		# motif_query_num1 = len(motif_query_vec_pre1)
		# motif_query_num2 = len(motif_query_vec_expr)
		# print('motif_query_vec_pre1, motif_query_vec_expr ',i1,sample_id1,motif_query_num1,motif_query_num2)
		x_train1, y_train1 = x_train, y_train
		# x_train1_feature2 = x_train_feature2
		sample_idvec_pre1 = x_train1.index
		feature_query_vec_coef = x_train1.columns.copy()

		# sample_id_vec=[]
		num_fold = select_config['num_fold']
		sample_idvec_query = select_config['sample_idvec_train']
		select_config1 = select_config['select_config1']
		# train_valid_mode = select_config['train_valid_mode_2']
		train_valid_mode = select_config['train_valid_mode_1']
		pre_data_dict_1 = pre_data_dict
		
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
				# average_type = 'weighted'
				# average_type = 'samples'
				type_id1 = 1
				select_config1.update({'type_classifer_xbgboost':type_id1})
				multi_class_query = 'auto'
				column_1 = 'multi_class_logisticregression'
				if column_1 in select_config:
					multi_class_query = select_config[column_1]
				select_config1.update({'multi_class_logisticregression':multi_class_query})
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
		# filename_save_annot1 = select_config['filename_save_annot1']
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
			
			# df_pred_query_feature2,df_pred_proba_feature2,df_score_2 = [], [], []
			y_pred1 = pd.Series(index=sample_idvec_pre1,data=0,dtype=np.float32)
			# output_mode = 1
			output_mode = 0
			if type_id_model==1:
				# num_class = select_config['num_class']
				y_pred1_proba = pd.DataFrame(index=sample_idvec_pre1,columns=range(num_class),data=0,dtype=np.float32)
			else:
				y_pred1_proba = []

			# y_pred1_feature2 = pd.Series(index=sample_idvec_pre1,data=0,dtype=np.float32)
			# y_pred1_proba_feature2 = pd.DataFrame(index=sample_idvec_pre1,columns=range(num_class),data=0,dtype=np.float32)
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

			## train with cross validation for performance evaluation
			list1_imp, list1_imp_scale = [], []

			# list2_imp, list2_imp_scale = [], []
			df_pred_query = []
			df_pred_proba = []
			list_feature_imp = []
			dict_feature_imp = dict()

			if num_fold>0:
				# list2_imp, list2_imp_scale = [], []
				for fold_id in range(num_fold):
					sample_idvec_1 = sample_idvec_query[fold_id]
					sample_id_train_query, sample_id_valid_query, sample_id_test_query = sample_idvec_1

					x_train2, y_train2 = x_train1.loc[sample_id_train_query,:], y_train1.loc[sample_id_train_query]
					x_test2, y_test2 = x_train1.loc[sample_id_test_query,:], y_train1.loc[sample_id_test_query]
					# feature_name = x_train2.columns
					# if flag_model_load==0:
					# 	model_1 = self.test_model_basic_pre1(model_type_id=model_type_id1,
					# 											pre_data_dict=pre_data_dict_1,
					# 											select_config=select_config1)

					# 	model_1, param1 = self.test_model_train_basic_pre1(model_1,
					# 														model_type_id1,
					# 														x_train2,
					# 														y_train2,
					# 														sample_weight=sample_weight)

					# 	if save_model_train==2:
					# 		## save models from the cross validation
					# 		save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
					# 		# pickle.dump(model_1, open(save_filename, 'wb'))
					# 		with open(save_filename,'wb') as output_file:
					# 			pickle.dump(model_1,output_file)
					# 		select_config.update({'model_save_filename':save_filename})
					# else:
					# 	model_save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
					# 	with open(model_save_filename, 'rb') as fid:
					# 		model_1 = pickle.load(fid)
					# 	if verbose>0:
					# 		print('model weights loaded ',model_save_filename)

					# save_mode_2 = (save_model_train==2)
					# model_save_filename = select_config['model_save_filename']
					# list_query1 = self.test_model_pred_explain_1(model_train=model_1,
					# 												x_test=x_test2,
					# 												y_test=y_test2,
					# 												sample_id_test_query=[],
					# 												y_pred=[],
					# 												y_pred_proba=[],
					# 												x_train=x_train2,y_train=y_train2,
					# 												fold_id=fold_id,
					# 												type_id_model=type_id_model,
					# 												model_explain=flag_model_explain,
					# 												model_save_filename=model_save_filename,
					# 												output_mode=output_mode,
					# 												save_mode=save_mode_2,
					# 												verbose=0,
					# 												select_config=select_config)

					# dict_feature_imp_[model_type_id] = {'imp':df_imp_,'imp_scaled':df_imp_scaled}
					# score_1, y_test2_pred, y_test2_proba, df_imp_1, df_imp_scaled_1 = list_query1
					# score_1, y_test2_pred, y_test2_proba, dict_query1 = list_query1
					# df_imp_1, df_imp_scaled_1 = dict_query1['imp'], dict_query1['imp_scaled']

					save_mode_2 = (save_model_train==2)
					save_model_train_1 = (save_model_train==2)
					model_1, param1, list_query1 = self.test_optimize_pre1_basic2_unit1(x_train=x_train2,y_train=y_train2,x_test=x_test2,y_test=y_test2,sample_weight=sample_weight,
																							model_type_id=model_type_id1,fold_id=fold_id,type_id_model=type_id_model,pre_data_dict=pre_data_dict_1,
																							flag_model_load=flag_model_load,flag_model_explain=flag_model_explain,output_mode=output_mode,
																							select_config1=select_config1,
																							save_mode=save_mode,save_mode_2=save_mode_2,save_model_train=save_model_train_1,
																							model_path=model_path_1,output_file_path=output_file_path,
																							filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,output_filename='',
																							verbose=0,select_config=select_config)

					score_1, y_test2_pred, y_test2_proba, dict_query1 = list_query1

					y_pred1.loc[sample_id_test_query] = y_test2_pred
					if type_id_model==1:
						y_pred1_proba.loc[sample_id_test_query,:] = y_test2_proba
					list1.append(score_1)

					# if len(dict_query1)>0:
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
					## regression model
					score_2 = self.score_2a_1(y_train1,y_pred1)
				else:
					## classification model
					# score_2 = utility_1.score_function_multiclass2(y_train1,y_pred1,y_pred1_proba,average=average_type)
					# select_config1 = select_config['select_config1']
					# average_type = select_config1['average_type']
					score_2 = self.score_function_multiclass2(y_train1,y_pred1,y_pred1_proba,average=average_type)

				list1.append(score_2)
				if verbose>0:
					# print(score_2,sample_id1[0:2],x_train1.shape,y_train1.shape)
					print(score_2,sample_idvec_pre1[0:2],x_train1.shape,y_train1.shape)

				df_score_1 = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				df_score_1 = df_score_1.T
				# df_score_1['sample_id'] = [sample_id1]*df_score_1.shape[0]
				df_score_1['fold_id'] = np.asarray(df_score_1.index)

				# list_feature_imp = []
				# dict_feature_imp = dict()
				column_id_query = 'feature_name'
				column_vec_2 = ['fold_id']
				column_id_query = ''
				if (flag_model_explain>0) and (len(list1_imp)>0):
					df_imp, df_imp1_mean = self.test_query_feature_mean_1(data=list1_imp,response_variable_name=response_variable_name,column_id_query=column_id_query,column_vec_query=column_vec_2,type_id_1=0,verbose=verbose,select_config=select_config)
					dict_feature_imp.update({'imp1':df_imp,'imp1_mean':df_imp1_mean})

				if (flag_model_explain>0) and (len(list1_imp_scale)>0):
					df_imp_scaled, df_imp1_scaled_mean = self.test_query_feature_mean_1(data=list1_imp_scale,response_variable_name=response_variable_name,column_id_query=column_id_query,column_vec_query=column_vec_2,type_id_1=0,verbose=verbose,select_config=select_config)
					dict_feature_imp.update({'imp1_scale':df_imp_scaled,'imp1_scale_mean':df_imp1_scaled_mean})

				df_pred_query = y_pred1
				df_pred_proba = y_pred1_proba

				df_pred_query = y_pred1
				df_pred_proba = 1

			# train on the combined data for coefficient estimation
			# param2 = []
			if train_valid_mode>0:
				# if flag_model_load==0:
				# 	model_2 = self.test_model_basic_pre1(model_type_id=model_type_id1,
				# 												pre_data_dict=pre_data_dict_1,
				# 												select_config=select_config1)

				# 	model_2, param2 = self.test_model_train_basic_pre1(model_2,
				# 															model_type_id1,
				# 															x_train1,
				# 															y_train1,
				# 															sample_weight=sample_weight)

				# 	if save_model_train>=1:
				# 		save_filename = '%s/test_model_%s.h5'%(model_path_1,filename_save_annot)
				# 		# pickle.dump(model_1, open(save_filename, 'wb'))
				# 		with open(save_filename,'wb') as output_file:
				# 			pickle.dump(model_2,output_file)
				# 		select_config.update({'model_save_filename':save_filename})
				# else:
				# 	model_save_filename = '%s/test_model_%s.h5'%(model_path_1,filename_save_annot)
				# 	with open(model_save_filename, 'rb') as fid:
				# 		model_2 = pickle.load(fid)
				# 		if model_type_id1 in ['LR','Lasso','ElasticNet','LogisticRegression']:
				# 			try:
				# 				param2 = [model_2.coef_, model_2.intercept_]
				# 			except Exception as error:
				# 				print('error! ',error)
				# 	print('model weights loaded ',model_save_filename)

				# # y_pred = model_2.predict(x_train1)
				# # y_proba = []
				# # if type_id_model==1:
				# # 	y_proba = model_2.predict_proba(x_train1)
				# list_query2 = self.test_model_pred_explain_1(model_train=model_2,
				# 												x_test=x_train1,
				# 												y_test=y_train1,
				# 												sample_id_test_query=[],
				# 												y_pred=[],
				# 												y_pred_proba=[],
				# 												x_train=x_train1,y_train=y_train1,
				# 												fold_id=-1,
				# 												type_id_model=type_id_model,
				# 												model_explain=flag_model_explain,
				# 												model_save_filename=model_save_filename,
				# 												output_mode=output_mode,
				# 												save_mode=save_mode_2,
				# 												verbose=0,
				# 												select_config=select_config)
				save_mode_2 = 0
				model_2, param2, list_query2 = self.test_optimize_pre1_basic2_unit1(x_train=x_train1,y_train=y_train1,x_test=x_train1,y_test=y_train1,sample_weight=sample_weight,
																						model_type_id=model_type_id1,fold_id=-1,type_id_model=type_id_model,pre_data_dict=pre_data_dict_1,
																						flag_model_load=flag_model_load,flag_model_explain=flag_model_explain,output_mode=output_mode,
																						select_config1=select_config1,
																						save_mode=save_mode,save_mode_2=save_mode_2,save_model_train=save_model_train,
																						model_path=model_path_1,output_file_path=output_file_path,
																						filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,output_filename='',
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
					## regression model
					score_2 = self.score_2a_1(y_train1,y_pred)
				else:
					## classification model
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
								# df_coef_query = pd.DataFrame(index=[response_variable_name],columns=feature_query_vec_coef,dtype=np.float32)
								df_coef_query = pd.Series(index=query_idvec,data=alpha_query,dtype=np.float32)
								df_coef_query.name = response_variable_name
							else:
								# df_coef_query.loc[feature_query_vec_coef,response_variable_name] = alpha_query
								df_coef_query.loc[query_idvec,response_variable_name] = alpha_query
						else:
							alpha_query = np.hstack((alpha_query1,intercept_[:,np.newaxis]))
							alpha_query = alpha_query.T
							if num_class==2:
								df_coef_query = pd.DataFrame(index=query_idvec,columns=[response_variable_name],data=np.asarray(alpha_query),dtype=np.float32)
							else:
								# df_coef_query = pd.DataFrame(index=np.arange(num_class),columns=feature_query_vec_coef,data=np.asarray(alpha_query),dtype=np.float32)
								df_coef_query = pd.DataFrame(index=query_idvec,columns=np.arange(num_class),data=np.asarray(alpha_query),dtype=np.float32)

					if model_type_id1 in ['LR_2']:
						df_coef_pval_1 = self.test_query_coef_pval_1(model_train=model_2,model_type_id=0,select_config=select_config)
						pval_query = np.asarray(df_coef_pval_1.loc[query_idvec,'pval'])
						if len(df_coef_pval_)==0:
							# df_coef_query = pd.DataFrame(index=[response_variable_name],columns=feature_query_vec_coef,dtype=np.float32)
							# df_coef_pval_ = pd.Series(index=query_idvec,data=pval_query,dtype=np.float32)
							# df_coef_pval_.name = response_variable_name
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
								
		# return df_coef_query, df_pred_query, df_score_1
		return dict_query_1, df_score_1

	## query tf feature importance estimate for gene query; control strength initialization
	# input: feature importance estimate filename
	# output: initial control strength estimate
	def test_optimize_pre1_basic2_2(self,model_pre,x_train,y_train,
										response_variable_name,feature_name=[],
										x_train_feature2=[],sample_weight=[],dict_query={},
										df_coef_query=[],df_pred_query=[],
										model_type_vec=[],model_type_idvec=[],
										filename_annot_vec=[],
										dict_score_query={},score_type_idvec=[],
										pre_data_dict={},
										type_id_train=0,type_id_model=0,num_class=1,
										save_mode=0,save_model_train=1,
										output_file_path='',output_filename='',
										verbose=1,
										select_config={}):
		
		# y_train1 = y_mtx.loc[sample_id1,:]
		# the tfs with expression in the cell
		# tf_expr = meta_exprs_2.loc[sample_id1,motif_query_vec_pre1]
		# motif_query_vec_expr = motif_query_vec_pre1[tf_expr>0]
		# motif_query_num1 = len(motif_query_vec_pre1)
		# motif_query_num2 = len(motif_query_vec_expr)
		# print('motif_query_vec_pre1, motif_query_vec_expr ',i1,sample_id1,motif_query_num1,motif_query_num2)

		x_train1, y_train1 = x_train, y_train
		# x_train1_feature2 = x_train_feature2
		# sample_idvec_pre1 = x_train1.index
		sample_id = x_train1.index
		feature_query_vec_coef = x_train1.columns

		# sample_id_vec=[]
		num_fold = select_config['num_fold']
		sample_idvec_query = select_config['sample_idvec_train']
		select_config1 = select_config['select_config1']
		# train_valid_mode = select_config['train_valid_mode_2']
		train_valid_mode = select_config['train_valid_mode_1']
		pre_data_dict_1 = pre_data_dict
		
		# list1 = []
		# list2 = []
		# model_type_num = len(model_type_idvec)
		# dict_query_1 = dict_query
		# filename_save_annot = select_config['filename_save_annot']
		# run_id = select_config['run_id']
		# select_config1 = select_config['select_config1'] # the configuration of models
		# if type_id_model==1:
		# 	num_class = len(np.unique(y_train1))
		# 	print('num_class ',num_class)
		# 	if num_class==2:
		# 		# binary classification model
		# 		average_type = 'binary'
		# 		type_id1 = 0
		# 		select_config1.update({'type_classifer_xbgboost':type_id1})
		# 		num_pos = np.sum(y_train1>0)
		# 		num_neg = np.sum(y_train1==0)
		# 		print('num_pos, num_neg ',num_pos,num_neg)
		# 	else:
		# 		# multi-class classification model
		# 		# average_type = 'macro'
		# 		average_type = 'micro'
		# 		# average_type = 'weighted'
		# 		# average_type = 'samples'
		# 		type_id1 = 1
		# 		select_config1.update({'type_classifer_xbgboost':type_id1})

		list1 = []
		list2 = []
		model_type_num = len(model_type_idvec)
		dict_query_1 = dict_query
		filename_save_annot = select_config['filename_save_annot_local']
		# filename_save_annot1 = select_config['filename_save_annot1']
		run_id = select_config['run_id']
		model_path_1 = output_file_path

		np.random.seed(0)
		for i1 in range(model_type_num):
			model_type_id1 = model_type_idvec[i1]
			# print('model_type_id1 ',model_type_id1)
			df_coef_query, df_pred_query, df_score_1 = [], [], []
			
			# df_pred_query_feature2,df_pred_proba_feature2,df_score_2 = [], [], []
			y_pred1 = pd.Series(index=sample_id,data=0,dtype=np.float32)
			# output_mode = 1
			output_mode = 0
			if type_id_model==1:
				num_class = select_config['num_class']
				y_pred1_proba = pd.DataFrame(index=sample_id,columns=range(num_class),data=0,dtype=np.float32)
			else:
				y_pred1_proba = []

			# y_pred1_feature2 = pd.Series(index=sample_id,data=0,dtype=np.float32)
			# y_pred1_proba_feature2 = pd.DataFrame(index=sample_id,columns=range(num_class),data=0,dtype=np.float32)
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
				print('model_type_id1: %s'%(model_type_id1))
				print('flag_model_explain: %d, linear_type_id: %d'%(flag_model_explain,linear_type_id))

			## train with cross validation for performance evaluation
			# list1_score = []
			list1_imp, list1_imp_scale = [], []
			list1_coef_query = []
			# list2_imp, list2_imp_scale = [], []
			for fold_id in range(num_fold):
				sample_idvec_1 = sample_idvec_query[fold_id]
				sample_id_train_query, sample_id_valid_query, sample_id_test_query = sample_idvec_1

				x_train2, y_train2 = x_train1.loc[sample_id_train_query,:], y_train1.loc[sample_id_train_query]
				x_test2, y_test2 = x_train1.loc[sample_id_test_query,:], y_train1.loc[sample_id_test_query]
				# feature_name = x_train2.columns
				if flag_model_load==0:
					model_1 = self.test_model_basic_pre1(model_type_id=model_type_id1,
															pre_data_dict=pre_data_dict_1,
															select_config=select_config1)

					model_1, param1 = self.test_model_train_basic_pre1(model_1,
																		model_type_id1,
																		x_train2,
																		y_train2,
																		sample_weight=sample_weight)

					if save_model_train==2:
						## save models from the cross validation
						save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
						# pickle.dump(model_1, open(save_filename, 'wb'))
						with open(save_filename,'wb') as output_file:
							pickle.dump(model_1,output_file)
				else:
					model_save_filename = '%s/test_model_%d_%s.h5'%(model_path_1,fold_id,filename_save_annot)
					with open(model_save_filename, 'rb') as fid:
						model_1 = pickle.load(fid)
					print('model weights loaded ',model_save_filename)

				save_mode_2 = (save_model_train==2)
				model_save_filename = select_config['model_save_filename']
				list_query1 = self.test_model_pred_explain_1(model_train=model_1,
																x_test=x_test2,
																y_test=y_test2,
																sample_id_test_query=[],
																y_pred=[],
																y_pred_proba=[],
																x_train=x_train2,y_train=y_train2,
																response_variable_name=response_variable_name,
																df_coef_query=[],
																fold_id=fold_id,
																type_id_model=type_id_model,
																model_explain=flag_model_explain,
																model_save_filename=model_save_filename,
																output_mode=output_mode,
																save_mode=save_mode_2,
																verbose=0,
																select_config=select_config)

				# dict_feature_imp_[model_type_id] = {'imp':df_imp_,'imp_scaled':df_imp_scaled}
				# score_1, y_test2_pred, y_test2_proba, df_imp_1, df_imp_scaled_1 = list_query1
				# dict_query1: 'imp','imp_scaled','coef_'
				score_1, y_test2_pred, y_test2_proba, dict_query1 = list_query1
				# df_imp_1, df_imp_scaled_1 = dict_query1['imp'], dict_query1['imp_scaled']
				y_pred1.loc[sample_id_test_query] = y_test2_pred
				if type_id_model==1:
					y_pred1_proba.loc[sample_id_test_query,:] = y_test2_proba
				list1.append(score_1)

				if 'coef_' in dict_query1:
					df_coef_query1 = dict_query1['coef_']
					df_coef_query1 = df_coef_query1.T # shape: (1,feature_dim+1) or (num_class, feature_dim+1)
					df_coef_query1['fold_id'] = fold_id
					df_coef_query1['response_variable'] = response_variable_name
					list1_coef_query.append(df_coef_query1)

				# if len(dict_query1)>0:
				if flag_model_explain>0:
					df_imp_1, df_imp_scaled_1 = dict_query1['imp'], dict_query1['imp_scaled']
					df_imp_1['fold_id'] = fold_id
					df_imp_scaled_1['fold_id'] = fold_id
					list1_imp.append(df_imp_1) # df_imp_1: shape: feature_num by column_num (columns: shap_value, imp2)
					list1_imp_scale.append(df_imp_scaled_1)

				if verbose>0:
					print('fold id, x_train2, y_train2, x_test2, y_test2 ',fold_id,x_train2.shape,y_train2.shape,x_test2.shape,y_test2.shape)
					print(score_1,fold_id,num_fold)

			if type_id_model==0:
				## regression model
				# score_2 = self.score_2a_1(y_train1,y_pred1)
				score_pre1 = self.score_2a_1(y_train1,y_pred1) # the prediction performance on the full data from different folds
			else:
				## classification model
				# score_2 = utility_1.score_function_multiclass2(y_train1,y_pred1,y_pred1_proba,average=average_type)
				select_config1 = select_config['select_config1']
				average_type = select_config1['average_type']
				# score_2 = self.score_function_multiclass2(y_train1,y_pred1,y_pred1_proba,average=average_type)
				score_pre1 = self.score_function_multiclass2(y_train1,y_pred1,y_pred1_proba,average=average_type)

			# list1.append(score_2)
			list1.append(score_pre1)
			if verbose>0:
				# print(score_2,sample_id1[0:2],x_train1.shape,y_train1.shape)
				print(score_pre1,sample_id[0:2],x_train1.shape,y_train1.shape)

			df_score_1 = pd.concat(list1,axis=1,join='outer',ignore_index=False)
			df_score_1 = df_score_1.T
			# df_score_1['sample_id'] = [sample_id1]*df_score_1.shape[0]
			df_score_1['fold_id'] = np.asarray(df_score_1.index)

			# list_feature_imp = []
			dict_feature_imp = dict()
			if (flag_model_explain>0) and (len(list1_imp)>0):
				df_imp = pd.concat(list1_imp,axis=0,join='outer',ignore_index=False)
				df_imp_scaled = pd.concat(list1_imp_scale,axis=0,join='outer',ignore_index=False)
				df_imp.loc[:,'feature_name'] = np.asarray(df_imp.index)
				df_imp_scaled.loc[:,'feature_name'] = np.asarray(df_imp_scaled.index)
				# print('df_imp, gene %s'%(response_variable_name))
				print('df_imp: ',df_imp)
				# print('df_imp_scaled, gene %s'%(response_variable_name))
				print('df_imp_scaled: ',df_imp_scaled)
				# list_feature_imp.append([df_imp,df_imp_scaled])
				# df1 = list1_imp[0]
				column_vec_1 = df_imp.columns.difference(['fold_id'],sort=False)
				column_vec_2 = df_imp_scaled.columns.difference(['fold_id'],sort=False)
				df_1 = df_imp.loc[:,column_vec_1].groupby(by='feature_name').mean()
				df_2 = df_imp_scaled.loc[:,column_vec_2].groupby(by='feature_name').mean()
				df_imp1_mean = pd.DataFrame(index=df_1.index,columns=df_1.columns,data=np.asarray(df_1))
				df_imp1_scale_mean = pd.DataFrame(index=df_2.index,columns=df_2.columns,data=np.asarray(df_2))
				# list_feature_imp.append([df_imp1_mean,df_imp1_scale_mean])
				dict_feature_imp.update({'imp1':df_imp,'imp1_scale':df_imp_scaled,'imp1_mean':df_imp1_mean,'imp1_scale_mean':df_imp1_scale_mean})
			
			if len(list1_coef_)>0:
				# the coefficients from cv, shape: (num_fold,feature_num+3), or (num_fold*num_class,feature_num+3), (feature_query,1,fold_id,gene_id)
				df_coef_query_2 = pd.concat(list1_coef_query,axis=0,join='outer',ignore_index=False)
				dict_feature_imp.update({'coef':df_coef_query_2})

			df_pred_query = y_pred1
			df_pred_proba = y_pred1_proba
			df_pred_query.name = response_variable_name
			# train on the combined data for coefficient estimation
			param2 = []
			if train_valid_mode>0:
				if flag_model_load==0:
					model_2 = model_pre.test_model_basic_pre1(model_type_id=model_type_id1,
																pre_data_dict=pre_data_dict_1,
																select_config=select_config1)

					model_2, param2 = model_pre.test_model_train_basic_pre1(model_1,
																			model_type_id1,
																			x_train1,
																			y_train1,
																			sample_weight=sample_weight)

					if save_model_train>=1:
						save_filename = '%s/test_model_%s.h5'%(model_path_1,filename_save_annot)
						# pickle.dump(model_1, open(save_filename, 'wb'))
						with open(save_filename,'wb') as output_file:
							pickle.dump(model_2,output_file)

				else:
					model_save_filename = '%s/test_model_%s.h5'%(model_path_1,filename_save_annot)
					with open(model_save_filename, 'rb') as fid:
						model_2 = pickle.load(fid)
						if model_type_id1 in ['LR','Lasso','ElasticNet','LogisticRegression']:
							try:
								param2 = [model_2.coef_, model_2.intercept_]
							except Exception as error:
								print('error! ',error)
					print('model weights loaded ',model_save_filename)

				# y_pred = model_2.predict(x_train1)
				# y_proba = []
				# if type_id_model==1:
				# 	y_proba = model_2.predict_proba(x_train1)
				list_query2 = self.test_model_pred_explain_1(model_train=model_2,
																x_test=x_train1,
																y_test=y_train1,
																sample_id_test_query=[],
																y_pred=[],
																y_pred_proba=[],
																x_train=x_train1,y_train=y_train1,
																response_variable_name=response_variable_name,
																df_coef_query=df_coef_query,
																fold_id=-1,
																type_id_model=type_id_model,
																model_explain=flag_model_explain,
																model_save_filename=model_save_filename,
																output_mode=output_mode,
																save_mode=save_mode_2,
																verbose=0,
																select_config=select_config)

				score_2, y_pred, y_proba, dict_query2 = list_query2
				list1 = [y_pred,y_proba]
				query_num = len(list1)
				for l1 in range(query_num):
					y_query = list1[l1]
					if len(y_query)>0:
						if y_query.ndim==1:
							y_query = pd.Series(index=sample_id,data=np.asarray(y_query),dtype=np.float32)
							y_query.name = response_variable_name
						else:
							n_dim = y_query.shape[1]
							y_query = pd.DataFrame(index=sample_id,columns=np.arange(n_dim),data=np.asarray(y_query),dtype=np.float32)
					list1[l1] = y_query
				y_pred, y_proba = list1

				field_query1 = score_2.index
				df_score_2 = pd.DataFrame(index=[num_fold+1],columns=df_score_1.columns)
				df_score_2.loc[num_fold+1,field_query1] = np.asarray(score_2)
				df_score_1 = df_score_1.append(df_score_2)
				df_score_1['fold_id'] = np.asarray(df_score_1.index)

				if flag_model_explain>0:
					df_imp_2, df_imp_scaled_2 = dict_query2['imp'], dict_query2['imp_scaled']
					# list_feature_imp.append(df_imp_2, df_imp_scaled_2)
					dict_feature_imp.update({'imp2':df_imp_2,'imp2_scale':df_imp_scaled_2})

				feature_query_vec_coef = x_train1.columns
				if len(param2)>0:
					if 'coef' in dict_query2:
						df_coef_query = dict_query2['coef']
					else:
						# alpha_query = param1[0].copy()
						alpha_query1 = param2[0]
						intercept_ = param2[1]
						if len(alpha_query1)>0:
							query_idvec = list(feature_query_vec_coef)+['alpha0']
							# if num_class<2:
							# 	alpha_query = list(alpha_query1)+[intercept_]
							# 	if len(df_coef_query)==0:
							# 		# df_coef_query = pd.DataFrame(index=[response_variable_name],columns=feature_query_vec_coef,dtype=np.float32)
							# 		df_coef_query = pd.Series(index=query_idvec,data=alpha_query,dtype=np.float32)
							# 		df_coef_query.name = response_variable_name
							# 	else:
							# 		# df_coef_query.loc[feature_query_vec_coef,response_variable_name] = alpha_query
							# 		df_coef_query.loc[query_idvec,response_variable_name] = alpha_query
							# else:
							# 	alpha_query = np.hstack((alpha_query1,intercept_[:,np.newaxis]))
							# 	alpha_query = alpha_query.T
							# 	# df_coef_query = pd.DataFrame(index=np.arange(num_class),columns=feature_query_vec_coef,data=np.asarray(alpha_query),dtype=np.float32)
							# 	df_coef_query = pd.DataFrame(index=query_idvec,columns=np.arange(num_class),data=np.asarray(alpha_query),dtype=np.float32)
							df_coef_query = self.test_query_ceof_1(param=param2,
																	feature_name=feature_query_vec_coef,
																	num_class=num_class,
																	response_variable_name=response_variable_name,
																	query_idvec=query_idvec,
																	df_coef_query=[],
																	select_config=select_config)

			# if (train_valid_mode>0) and (save_mode>0) and (output_filename!='') and (len(df_coef_query)>0):
			# 	# output_file_path = input_file_path
			# 	# # output_filename = '%s/%s.tf_query.pre1.%d.txt'%(output_file_path,filename_prefix_1,run_id)
			# 	# output_filename = '%s/%s.tf_query.pre1.%s.txt'%(output_file_path,filename_prefix_1,filename_annot1)
			# 	df_coef_query.to_csv(output_filename,sep='\t',float_format='%.6E')
			# 	print('df_coef_query ',df_coef_query.shape)

			# df_feature2_query_list = [df_pred_query_feature2,df_pred_proba_feature2,df_score_2]
			# dict_query_1[model_type_id1] = [df_coef_query, df_pred_query, df_pred_proba, y_pred, y_proba, df_feature2_query_list, list_feature_imp]
			# dict_query_1[model_type_id1] = [df_coef_query, df_pred_query, df_pred_proba, y_pred, y_proba, list_feature_imp]
			# dict_query_1[model_type_id1] = {'coef':df_coef_query, 
			# 								'pred_cv':df_pred_query, 
			# 								'pred_proba_cv':df_pred_proba, 
			# 								'pred':y_pred, 
			# 								'pred_proba':y_proba, 
			# 								'feature_imp':list_feature_imp}
			dict_query_1[model_type_id1] = {'coef':df_coef_query, 
											'pred_cv':df_pred_query, 
											'pred_proba_cv':df_pred_proba, 
											'pred':y_pred, 
											'pred_proba':y_proba, 
											'feature_imp':dict_feature_imp}
								
		# return df_coef_query, df_pred_query, df_score_1
		return dict_query_1, df_score_1

	## model explanation
	# feature interaction importance estimate from the learned model
	def test_model_explain_basic_pre1(self,x_train,y_train,feature_name,x_test=[],y_test=[],
											model_train_dict=[],model_save_dict=[],
											model_path_1='',save_mode=0,model_save_file='',
											select_config={}):

		## model_type_id: model_type_name.feature_type_id
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
		filename_dict2, dict_interaction_, save_filename_dict2 = dict(), dict(), dict() # feature interaction importance estimate
		# file_save_path = self.save_path_1
		file_save_path = model_path_1
		# save_mode_1 = 1
		for i1 in range(model_type_num):
			model_type_id, model_type_name, feature_type_id = model_type_idvec[i1], model_type_name_vec[i1], feature_type_vec[i1]
			# print('model_type_id ', model_type_id, i1, model_type_name, feature_type_id)

			model_save_file = ''
			if pre_load==0:
				model_train = model_train_dict[model_type_id]
				if model_type_id in model_save_dict:
					model_save_file = model_save_dict[model_type_id]
			else:
				model_save_file = model_save_dict[model_type_id]
				model_train = pickle.load(open(model_save_file, "rb"))

			df_imp_, df_imp_scaled, coef_query_1 = self.test_model_explain_pre2(model_train,
																				x_train,y_train,
																				feature_name,
																				model_type_name,
																				linear_type_id=0,
																				model_save_file=model_save_file,
																				select_config=select_config,
																				save_mode=save_mode)

			dict_feature_imp_[model_type_id] = {'imp':df_imp_,'imp_scaled':df_imp_scaled,'coef_query':coef_query_1}
			# dict_feature_imp_[model_type_id] = {'imp':df_imp_,'imp_scaled':df_imp_scaled}

			if save_mode==1:
				if model_save_file=='':
					# output_filename1 = '%s/test_peak_motif_estimate.model_%s.imp.1.txt'%(file_save_path,model_type_id)
					# output_filename2 = '%s/test_peak_motif_estimate.model_%s.imp_scaled.1.txt'%(file_save_path,model_type_id)
					 #filename_annot2 = select_config['filename_annot2']
					filename_save_annot = select_config['filename_save_annot']
					output_filename1 = '%s/test_query.model_%s.%s.imp.1.txt'%(file_save_path,model_type_id,filename_save_annot)
					output_filename2 = '%s/test_query.model_%s.%s.imp_scaled.1.txt'%(file_save_path,model_type_id,filename_save_annot)		
				else:
					b = model_save_file.find('.h5')
					output_filename1 = model_save_file[0:b]+'.imp.1.txt'
					output_filename2 = model_save_file[0:b]+'.imp_scaled.1.txt'

				df_imp_.to_csv(output_filename1,sep='\t',float_format='%.5E')
				# df_imp_scaled.to_csv(output_filename2,sep='\t',float_format='%.6E')
				intercept_ = 0
				# if 'intercept_' in df_imp_:
				# 	intercept_ = df_imp_['intercept_']
				# if 'intercept_' in df_imp_:
				# 	intercept_ = df_imp_['intercept_']
				if len(coef_query_1)>0:
					coef_, intercept_ = coef_query_1
				filename_dict1[model_type_id] = {'imp':output_filename1,'imp_scaled':output_filename2,'coef_':coef_,'intercept_':intercept_}
			
			# if model_type_name in ['XGBR']:
			# 	model_list2.append(model_train)

			# 	filename_save_interaction = self.test_model_explain_interaction_pre1(model_train,feature_name,model_type_id,
			# 																			model_save_file=model_save_file,
			# 																			select_config=select_config)
			# 	filename_dict2[model_type_id] = filename_save_interaction

		# sel_num1, sel_num2 = select_config['sel_num1_interaction'], select_config['sel_num2_interaction']
		# save_mode_2 = 1
		# if len(filename_dict2)>0:
		# 	dict_interaction_, save_filename_dict2 = self.test_model_explain_interaction_pre2(filename_dict2,sel_num1=sel_num1,sel_num2=sel_num2,save_mode=save_mode_2)

		return dict_feature_imp_, dict_interaction_, filename_dict1, filename_dict2, save_filename_dict2

	# # estimate feature importance from the learned model
	# def test_model_explain_1(self,model,x,y,feature_name,model_type_id,x_test=[],y_test=[],linear_type_id=0):

	# 	if model_type_id in ['XGBR','XGBClassifier','RF','RandomForestClassifier']:
	# 		# pred = model.predict(x, output_margin=True)
	# 		# pred = model.predict(x)
	# 		explainer = shap.TreeExplainer(model)
	# 		# explainer = shap.Explainer(model,x)
	# 		shap_value_pre1 = explainer(x)
	# 		shap_values = shap_value_pre1.values
	# 		base_values = shap_value_pre1.base_values

	# 		# expected_value = []
	# 		expected_value = base_values[0]
	# 		# expected_value = explainer.expected_value
	# 		# t1 = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
	# 		# print(t1)
	# 	elif linear_type_id>=1:
	# 		# shap.explainers.Linear(model, masker, link=CPUDispatcher(<function identity>), nsamples=1000, feature_perturbation=None, **kwargs)
	# 		feature_perturbation = ['interventional','correlation_dependent']
	# 		# explainer = shap.explainers.Linear(model,x,nsamples=x.shape[0],feature_perturbation=None)
	# 		feature_perturbation_id1 = linear_type_id-1
	# 		feature_perturbation_id2 = feature_perturbation[feature_perturbation_id1]
	# 		print(feature_perturbation_id2)
	# 		explainer = shap.explainers.Linear(model,x,nsamples=x.shape[0],feature_perturbation=feature_perturbation_id2)
	# 		shap_value_pre1 = explainer(x)
	# 		shap_values = shap_value_pre1.values
	# 		base_values = shap_value_pre1.base_values
	# 		expected_value = explainer.expected_value
	# 	else:
	# 		explainer = shap.Explainer(model, x, feature_names=feature_name)
	# 		shap_value_pre1 = explainer(x)
	# 		shap_values = shap_value_pre1.values
	# 		base_values = shap_value_pre1.base_values
	# 		# data_vec1 = shap_value_pre1.data
	# 		# shap_values = explainer.shap_values(x)
	# 		# base_values = []
	# 		expected_value = explainer.expected_value
	# 		# shap_values_test = explainer(x_test)
	# 		pred = model.predict(x)
	# 		t1 = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
	# 		# print(t1)
		
	# 	return shap_values, base_values, expected_value

	# feature importance estimate from the learned model
	def test_model_explain_pre1(self,model,x,y,feature_name,model_type_name,x_test=[],y_test=[],linear_type_id=0,select_config={}):

		if model_type_name in ['XGBR','XGBClassifier','RF','RandomForestClassifier']:
			# pred = model.predict(x, output_margin=True)
			# pred = model.predict(x)
			explainer = shap.TreeExplainer(model)
			# explainer = shap.Explainer(model,x)
			shap_value_pre1 = explainer(x)
			shap_values = shap_value_pre1.values
			base_values = shap_value_pre1.base_values

			# expected_value = []
			expected_value = base_values[0]
			# expected_value = explainer.expected_value
			# t1 = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
			# print(t1)
		elif linear_type_id>=1:
			# shap.explainers.Linear(model, masker, link=CPUDispatcher(<function identity>), nsamples=1000, feature_perturbation=None, **kwargs)
			feature_perturbation = ['interventional','correlation_dependent']
			# explainer = shap.explainers.Linear(model,x,nsamples=x.shape[0],feature_perturbation=None)
			feature_perturbation_id1 = linear_type_id-1
			feature_perturbation_id2 = feature_perturbation[feature_perturbation_id1]
			# print(feature_perturbation_id2)
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
			# data_vec1 = shap_value_pre1.data
			# shap_values = explainer.shap_values(x)
			# base_values = []
			expected_value = explainer.expected_value
			# shap_values_test = explainer(x_test)
			# pred = model.predict(x)
			# t1 = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
			# print(t1)
		
		return shap_values, base_values, expected_value

	# feature importance estimate from the learned model
	def test_model_explain_pre2_ori(self,model_train,
										x_train,y_train,
										feature_name,
										model_type_name,
										x_test=[],y_test=[],
										linear_type_id=0,
										model_save_file='',
										select_config={},
										save_mode=1):

		if model_type_name in ['LR','Lasso','LassoCV','ElasticNet']:
			linear_type_id = 1

		# threshold_select = 'median'
		# # sfm_selector = SelectFromModel(estimator=model_train, threshold=threshold_select, prefit=False, norm_order=1, max_features=None, importance_getter='auto')
		# sfm_selector = SelectFromModel(estimator=model_train, threshold=threshold_select, prefit=False, norm_order=1, max_features=None)
		# sfm_selector.fit(x,y)
		# estimator_, feature_sel_thresh_ = sfm_selector.estimator_, sfm_selector.threshold_

		# if flag1==1:
		# 	feature_importances_1 = estimator_.feature_importances_
		# 	feature_importances_2 = model_train.feature_importances_
		# else:
		# 	shap_value_2, base_value_2, expected_value_2 = self.test_model_explain_pre1(estimator_,x,y,feature_name,model_type_id=model_type_name,x_test=[],y_test=[],linear_type_id=0)
		# 	shap_value_1, base_value_1, expected_value_1 = self.test_model_explain_pre1(model_train,x,y,feature_name,model_type_id=model_type_name,x_test=[],y_test=[],linear_type_id=0)
		# 	feature_importances_1 = np.mean(np.abs(shap_value_1),axis=0)	# mean_abs_value
		# 	feature_importances_2 = np.mean(np.abs(shap_value_2),axis=0)	# mean_abs_value

		shap_value_1, base_value_1, expected_value_1 = self.test_model_explain_pre1(model_train,
																					x=x_train,y=y_train,
																					feature_name=feature_name,
																					model_type_name=model_type_name,
																					x_test=x_test,y_test=y_test,
																					linear_type_id=0)

		feature_importances_1 = np.mean(np.abs(shap_value_1),axis=0)	# mean_abs_value
		df_imp_['shap_value'] = np.asarray(feature_importances_1)
		print('shap_value_1, feature_importances_1: ',shap_value_1.shape,feature_importances_1.shape)
		
		type_id_model = (feature_importances_1.ndim==1)
		df_imp_ = pd.DataFrame(index=feature_name,columns=['shap_value','imp2'])
		df_imp_scaled = pd.DataFrame(index=feature_name,columns=['shap_value','imp2'])

		# feature importance estimate normalized to [0,1]
		feature_imp1_scale = pd.Series(index=feature_name,data=minmax_scale(feature_importances_1,[1e-07,1]))
		df_imp_scaled['shap_value'] = np.asarray(feature_imp1_scale)

		if model_type_name in ['XGBR','RandomForestRegressor']:
			# feature importance estimated by the model
			feature_importances_2 = model_train.feature_importances_
			df_imp_['imp2'] = np.asarray(feature_importances_2)

			# feature importance estimate normalized to [0,1]
			feature_imp2_scale = pd.Series(index=feature_name,data=minmax_scale(feature_importances_2,[1e-07,1]))
			df_imp_scaled['imp2'] = np.asarray(feature_imp2_scale)

		if model_type_name in ['LR','Lasso','ElasticNet','LogisticRegression']:
			coef_query, intercept_query = model_train.coef_, model_train.intercept_
			df_imp_['coef'] = np.asarray(coef_query)
			df_imp_['intercept_'] = np.asarray(intercept_query)

		# feature_sel_id = sfm_selector.get_support()
		# df2 = pd.DataFrame(index=feature_name,columns=[celltype,'%s.1'%(celltype)],data=np.column_stack((feature_importances_1,feature_importances_2)))
		# df_list2.append(df2)

		return df_imp_, df_imp_scaled

	# feature importance estimate from the learned model
	def test_model_explain_pre2(self,model_train,x_train,y_train,feature_name,model_type_name,x_test=[],y_test=[],linear_type_id=0,model_save_file='',select_config={},save_mode=1):

		if model_type_name in ['LR','Lasso','LassoCV','ElasticNet']:
			linear_type_id = 1

		shap_value_1, base_value_1, expected_value_1 = self.test_model_explain_pre1(model_train,x=x_train,y=y_train,
																					feature_name=feature_name,
																					model_type_name=model_type_name,
																					x_test=x_test,y_test=y_test,
																					linear_type_id=linear_type_id)

		# print('shap_value_1 ',shap_value_1.shape)
		feature_importances_1 = np.mean(np.abs(shap_value_1),axis=0)	# mean_abs_value
		# print('feature_importances_1 ',feature_importances_1.shape)

		coef_query, intercept_query = [], []
		if feature_importances_1.ndim==1:
			# df_imp_ = pd.DataFrame(index=feature_name,columns=['shap_value','imp2'])
			# df_imp_scaled = pd.DataFrame(index=feature_name,columns=['shap_value','imp2'])
			df_imp_ = pd.DataFrame(index=feature_name,columns=['shap_value'])
			df_imp_scaled = pd.DataFrame(index=feature_name,columns=['shap_value'])
			df_imp_['shap_value'] = np.asarray(feature_importances_1)

			# feature importance estimate normalized to [0,1]
			feature_imp1_scale = pd.Series(index=feature_name,data=minmax_scale(feature_importances_1,[1e-07,1]))
			df_imp_scaled['shap_value'] = np.asarray(feature_imp1_scale)

			if model_type_name in ['XGBR','RandomForestRegressor']:
				# feature importance estimated by the model
				feature_importances_2 = model_train.feature_importances_
				df_imp_['imp2'] = np.asarray(feature_importances_2)

				# feature importance estimate normalized to [0,1]
				feature_imp2_scale = pd.Series(index=feature_name,data=minmax_scale(feature_importances_2,[1e-07,1]))
				df_imp_scaled['imp2'] = np.asarray(feature_imp2_scale)

			# if model_type_name in ['LR','Lasso','ElasticNet','LogisticRegression']:
			# 	coef_query, intercept_query = model_train.coef_, model_train.intercept_
			# 	df_imp_['coef'] = np.asarray(coef_query)
			# 	df_imp_['intercept_'] = np.asarray(intercept_query)

		else:
			num_dim = feature_importances_1.shape[1]
			df_imp_ = pd.DataFrame(index=feature_name,columns=np.arange(num_dim),data=np.asarray(feature_importances_1))
			df_imp_scaled = pd.DataFrame(index=feature_name,columns=np.arange(num_dim),data=minmax_scale(feature_importances_1,[1e-07,1]))

			if model_type_name in ['LR','Lasso','LassoCV','ElasticNet','LogisticRegression']:
				coef_query, intercept_query = model_train.coef_, model_train.intercept_

		# feature_sel_id = sfm_selector.get_support()
		# df2 = pd.DataFrame(index=feature_name,columns=[celltype,'%s.1'%(celltype)],data=np.column_stack((feature_importances_1,feature_importances_2)))
		# df_list2.append(df2)

		return df_imp_, df_imp_scaled, (coef_query, intercept_query)

	# feature interaction importance estimate from the learned model
	def test_model_explain_interaction_pre1(self,model,feature_name,model_type_id,model_save_file='',select_config={}):

		flag1 = 1
		run_id = select_config['run_id']
		if flag1==1:
			# model_path2 = '%s/model_%s.merge.%d.%d.h5'%(model_path_1,t_label,type_id_2,self.run_id)
			# model_path2 = '%s/model_%s.merge.%d.%d.%s.pre1.h5'%(model_path_1,t_label,type_id_2,self.run_id,annot1)
			# model_2.save_model(model_path2)
			# model_2.dump_model(model_path2)
			# pickle.dump(model_2, open(model_path2, "wb"))
			if model_save_file!='':
				loaded_model = pickle.load(open(model_save_file, "rb"))
			else:
				loaded_model = model

			if 'max_interaction_depth' in select_config:
				max_interaction_depth = select_config['max_interaction_depth']
			else:
				max_interaction_depth = 10

			annot_pre1 = '%s_%d_%d'%(model_type_id,run_id,max_interaction_depth)
			output_file_path = select_config['output_file_path_explain_1']
			# output_filename = '%s/model_%s.feature_interaction.1.xlsx'%(self.save_path_1,annot_pre1)
			output_filename = '%s/model_%s.feature_interaction.1.xlsx'%(output_file_path,annot_pre1)
			xgbfir.saveXgbFI(loaded_model,feature_names=feature_name,MaxInteractionDepth=max_interaction_depth,OutputXlsxFile=output_filename)

			filename_save_interaction = output_filename

		return filename_save_interaction

	# feature interaction importance estimate from the learned model
	# feature interaction importance estimate at specified interaction depth levels
	# sel_num1: interaction depth
	# sel_num2: the number of interactions to select
	def test_model_explain_interaction_pre2(self,filename_dict,sel_num1=2,sel_num2=-1,save_mode=1):
		
		# model_type_idvec = list(filename_dict.keys())
		# model_type_num = len(model_type_idvec)

		query_idvec = list(filename_dict.keys())
		query_num = len(query_idvec)

		print(filename_dict)
		flag = 1
		interaction_depth = sel_num1-1
		dict1, save_filename_dict1 = dict(), dict()

		for i1 in range(query_num):
			# input_filename1 = '%s/train2/%s'%(self.path_1,t_filename1)
			# input_filename1 = '%s/%s'%(self.path_1,t_filename1)
			# model_type_id = model_type_idvec[i1]
			query_id = query_idvec[i1]
			# input_filename1 = filename_dict[model_type_id]
			model_type_id = query_id
			input_filename1 = filename_dict[model_type_id]
			if os.path.exists(input_filename1)==False:
				print('file does not exist ',input_filename1)
				continue

			list1, list_1, list_2 = [], [], []
			if flag==1:
			# try:
				sheet_name = 'Interaction Depth %d'%(interaction_depth)
				# data1 = pd.read_excel(input_filename1,sheet_name=sheet_name)
				# print(data1.head())

				data1 = pd.ExcelFile(input_filename1)
				df = {sheet_name: data1.parse(sheet_name) for sheet_name in data1.sheet_names}
				# print(df.keys())

				df1 = df[sheet_name]
				# print(input_filename1,df1.head())
				print(input_filename1,df1[0:2])
				feature_name_interaction = df1['Interaction']
				sel_column_vec = ['Gain','FScore','wFScore','Average wFScore','Average Gain','Expected Gain',
									'Gain Rank','Fscore Rank','wFScore Rank','Avg wFScore Rank','Avg Gain Rank','Expected Gain Rank',
									'Average Rank','Average Tree Index','Average Tree Depth']

				# sel_column_id1 = sel_column_vec[0]
				sel_column_id1 = 'Gain'
				feature_imp = df1[sel_column_id1]

				if sel_num2>0:
					t_sel_num2 = np.min([len(feature_name_interaction),sel_num2])
				else:
					t_sel_num2 = len(feature_name_interaction)

				for t_feature_name in feature_name_interaction[0:t_sel_num2]:
					str_vec1 = t_feature_name.split('|')
					str_vec1 = np.sort(str_vec1)
					# flag = 1
					# if type_id_1==2:
					#   for tf_id in str1:
					#       if not (tf_id in motif_name_exprs):
					#           flag = 0
					#           print(str1,tf_id)
					# if flag==1:
					#   list1.extend(str1)
					# list1.extend(str1)
					list_1.append(str_vec1)

			# except:
			#   continue

			if len(list_1)>0:
				# list1 = list(np.unique(list1))
				print(len(list_1),list_1[0:5])

				idvec = np.asarray(['.'.join(t_vec1) for t_vec1 in list_1])
				mtx1 = np.asarray(list_1)
				interaction_sel_num, feature_query_num = mtx1.shape[0], mtx1.shape[1] # the number of selected feature interactions and the number of features in the interaction
				t_columns = ['feature%d'%(query_id1+1) for query_id1 in range(feature_query_num)]
				df1 = pd.DataFrame(index=idvec,columns=t_columns,data=mtx1)
				df1[sel_column_id1] = np.asarray(feature_imp)
				dict1[query_id] = df1

				if save_mode==1:
					b = input_filename1.find('.xlsx')
					output_filename1 = input_filename1[0:b]+'.interaction%d.txt'%(query_num)
					df1.to_csv(output_filename1,sep='\t')
					# save_filename_list1.append(output_filename1)
					save_filename_dict1[query_id] = output_filename1

		return dict1, save_filename_dict1

	def model_query_1(self,gene_query_id='',motif_query_id='',feature_query_vec=[],feature_imp_est=1,select_config={}):

		alpha_vec = self.alpha_
		model_train_pre = self.model_train_
		x, y = self.data_vec_
		score_pred = self.score_pred_
		score_pred_vec = self.score_pred_vec
		# df_score_pred,df_pred,mean_score = score_pred_vec

		# feature_query_id = self.feature_query_id
		if len(feature_query_vec)==0:
			feature_query_vec = self.feature_query_id
		alpha_motif, alpha0 = alpha_vec
		df_pre1 = pd.DataFrame(index=feature_query_vec,columns=['coef'],dtype=np.float32)
		df_pre1['coef'] = alpha_motif

		feature_imp_est_vec = []
		train_mode_cv = self.train_mode_cv
		if feature_imp_est==1:
			feature_imp_est_vec = self.feature_imp_est_
			shap_values, base_values, expected_value = feature_imp_est_vec
			print('shap_values, base_values, expected_value ',shap_values.shape,len(base_values),expected_value)
			print('shap_values ', np.max(np.abs(shap_values)),np.max(np.abs(base_values)),np.max(np.abs(expected_value)),gene_query_id)

			shap_values_mean = np.mean(np.abs(shap_values),axis=0)
			# df1 = pd.DataFrame(index=feature_query_id,columns=['shap_value_mean','coef'],dtype=np.float32)
			df_pre1['shap_value_mean'] = shap_values_mean
			# df1_sort = df1.sort_values(by=['shap_value_mean'],ascending=False)

			if train_mode_cv==1:
				# if train_mode_cv=1: shap_value_mean: from one run selected; shap_value_mean_2: the average of estimates from n folds
				df_pre1['shap_value_mean_2'] = self.shap_values_mean
				feature_imp_est_vec_list2 = self.feature_imp_list2 # feature_imp_est_vec, coefficient estimate, model_train of each fold
			else:
				feature_imp_est_vec_list2 = [alpha_vec,feature_imp_est_vec,model_train_pre]

		df_feature_query = df_pre1
		print('score_pred ', np.asarray(score_pred))

		return (model_train_pre, df_feature_query, alpha0, feature_imp_est_vec, feature_imp_est_vec_list2, x, y, score_pred_vec)

	def model_query_2(self,gene_query_id='',motif_query_id='',feature_imp_est=1,select_config={}):

		# alpha_vec = self.alpha_
		# model_train_pre = self.model_train_
		# x, y = self.data_vec_

		model_1,alpha_motif_vec,motif_feature_imp_est_vec,motif_query_vec = self.pre_model_dict[gene_query_id]['motif']
		model_2,beta_peak_vec,peak_feature_imp_est_vec,peak_loc_query = self.pre_model_dict[gene_query_id]['peak']

		pre_model_dict = self.pre_model_dict
		field_query = ['motif','peak']

		feature_imp_est_vec = []
		# list1 = [motif_feature_imp_est_vec,peak_feature_imp_est_vec]
		query_num1 = len(field_query)
		
		# feature_imp_est_vec = self.feature_imp_est_
		for i1 in range(query_num1):
			t_field_query = field_query[i1]
			model_train,beta_vec,feature_imp_est_vec,feature_query_id = pre_model_dict[gene_query_id][t_field_query]
			
			beta1, beta0 = beta_vec
			df_pre1 = pd.DataFrame(index=feature_query_id,columns=['coef'],dtype=np.float32)
			df_pre1['coef'] = beta1

			if feature_imp_est==1:
				shap_values, base_values, expected_value = feature_imp_est_vec
				print('shap_values ', np.max(np.abs(shap_values)),np.max(np.abs(base_values)),np.max(np.abs(expected_value)),gene_query_id)

				shap_values_mean = np.mean(np.abs(shap_values),axis=0)
				# df1 = pd.DataFrame(index=feature_query_id,columns=['shap_value_mean','coef'],dtype=np.float32)
				df_pre1['shap_value_mean'] = shap_values_mean
				# df1_sort = df1.sort_values(by=['shap_value_mean'],ascending=False)

			df_feature_query = df_pre1
			query_list1 = [model_train,df_feature_query,beta0,feature_imp_est_vec,feature_query_id]
			pre_model_dict[gene_query_id].update({t_field_query:query_list1})

		# return (model_train_pre, df_feature_query, alpha0, feature_imp_est_vec, x, y)

		return pre_model_dict

	## tf motif coefficients estimate
	def test_signal_clip_1(self,x,type_id=0,thresh_1=0.01,thresh_2=0.99,select_config={}):

		# thresh_1, thresh_2 = 0.01, 0.99
		thresh1, thresh2 = np.quantile(x,thresh_1), np.quantile(x,thresh_2)
		thresh_pre1, thresh_pre2 = np.quantile(x,0.25), np.quantile(x,0.75)
		IQR = thresh_pre2 - thresh_pre1
		lower_bound, upper_bound = thresh_pre1-1.5*IQR, thresh_pre2+1.5*IQR
		thresh1_1, thresh2_1 = np.min(x[x>lower_bound]), np.max(x[x<upper_bound])
		
		x_ori = x.copy()
		if type_id==0:
			if np.max(x)>upper_bound:
				x[x>thresh2] = thresh2
			if np.min(x)<lower_bound:
				x[x<thresh1] = thresh1
		else:
			if type_id==1:
				x[x>upper_bound] = thresh1_2
				x[x<lower_bound] = thresh1_1

		quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.995]
		t_value1 = utility_1.test_stat_1(x_ori,quantile_vec=quantile_vec_1)
		t_value2 = utility_1.test_stat_1(x,quantile_vec=quantile_vec_1)
		print('x_ori, x ', t_value1, t_value2)

		return x

	## tf motif coefficients estimate
	def test_feature_query_pre1(self,gene_query_id='',feature_type_id='',peak_read=[],rna_exprs=[],score_list=[],score_list_query_id=0,score_query_type=[],scale_type_id=0,scale_type_vec=[],select_config={}):

		## minmax_scale: regular; minmax_scale_1: upper and lower bound; minmax_scale_2: lower bound;
		# scale: regular; scale_2: minmax_scale between upper and lower bound and scale; pre: without normalization
		score_scale_type_vec = ['minmax_scale','minmax_scale_2','scale','scale_2',-1,'quantile_transform','minmax_scale_1','binarize','pre']
		score_scale_type_vec_1 = ['minmax.1','minmax.2','scale.1','scale',-1,'quantile_transform','minmax','binarize','pre']

		tf_score_scale_type = 6
		tf_score_product_scale_type = 3
		minmax_scale_type, standard_scale_type, binarize_type = 6, 3, 7
		# scale_type_vec = [0,1,2,3,5,6]
		if len(scale_type_vec)==0:
			scale_type_vec = [0,1,2,3,5,6]
		
		sample_id = rna_exprs.index
		# x = score_mtx.loc[sample_id,:]
		gene_query_expr = rna_exprs.loc[:,gene_query_id]
		y = gene_query_expr
		# flag_clip_1 = 0
		# if flag_clip_1>0:
		# 	y = self.test_signal_clip_1(gene_query_expr,type_id=0,thresh_1=0.01,thresh_2=0.99,select_config=select_config)

		df_gene_annot_expr = self.df_gene_annot_expr
		# thresh_dispersions_norm_1 = 0
		thresh_dispersions_norm_query = 0.5
		if 'thresh_dispersions_norm_query' in select_config:
			thresh_dispersions_norm_query = select_config['thresh_dispersions_norm_query']

		gene_query_id_pre_1 = df_gene_annot_expr.index[df_gene_annot_expr['dispersions_norm']>thresh_dispersions_norm_query]
		print('gene_query_id_pre_1 ', len(gene_query_id_pre_1))

		data1 = score_list[score_list_query_id]
		list_pre1 = data1[gene_query_id]
		score_mtx, tf_id = list_pre1
		tf_query_id = pd.Index(tf_id).intersection(gene_query_id_pre_1,sort=False)
		print('gene_query_id, score_mtx, tf_id, tf_query_id ', gene_query_id, score_mtx.shape, len(tf_id), len(tf_query_id))

		if feature_type_id==0:
			# x = score_mtx.loc[sample_id,tf_query_id]
			# print('score_mtx, gene_query_expr ', x.shape, y.shape)
			score_query = score_mtx.loc[sample_id,tf_query_id]	# tf score
		else:
			score_query = rna_exprs.loc[sample_id,tf_query_id]	# tf exprs

		if scale_type_id>0:
			tf_score_type, log_type_id, scale_type_pre = score_query_type
			score_normalized_pre, flag_pre = self.test_motif_peak_estimate_tf_score_scale_1(score_query=score_query,
																							score_type=tf_score_type,
																							log_type_id=log_type_id,scale_type=scale_type_pre,
																							scale_type_vec=scale_type_vec,
																							minmax_scale_type=minmax_scale_type,
																							binarize_type=binarize_type,
																							select_config=select_config)
			x = score_normalized_pre
		else:
			x = score_query

		# flag_clip_2 = 0
		# if (scale_type_id>0) and (scale_type_pre in [0,1,6,7]):
		# 	flag_clip_2=0
		# if flag_clip_2>0:
		# 	for t_field_query in x.columns:
		# 		x[t_field_query] = self.test_signal_clip_1(x[t_field_query],type_id=0,thresh_1=0.01,thresh_2=0.99,select_config=select_config)

		print('score_mtx, gene_query_expr ', x.shape, y.shape)

		return (x,y,tf_query_id)

	## motif-peak estimate: tf accessibility score
	# tf accessibility score: cell_num by feature_query_num
	# def test_motif_peak_estimate_tf_score_1(self,peak_dict_sel={},dict_gene_motif_query={},peak_read=[],meta_exprs=[],gene_query_vec=[],feature_query_vec=[],peak_motif_mtx=[],type_score_pair=1,pre_load=1,save_mode=1,filename_annot1='1',select_config={}):
	def test_motif_peak_estimate_tf_score_scale_1(self,score_query,score_type,log_type_id,scale_type,scale_type_vec=[],minmax_scale_type=-1,binarize_type=-1,select_config={}):

		thresh1 = 0
		flag1 = 1
		flag_pre = 0
		if flag1==1:
			if log_type_id>0:
				## tf accessibility score log-transformation and scaling
				const1 = 1.0
				print('log transformation ', score_type, log_type_id)
				score_query_ori = score_query.copy()
				score_query = np.log(score_query+const1)
				flag_pre = 2

			if scale_type in scale_type_vec:
				# print('minmax_scale ', score_type, scale_type)
				print('scale ', score_type, scale_type)
				score_normalized_pre = utility_1.test_motif_peak_estimate_score_scale_1(score=score_query,feature_query_vec=[],
																						select_config=select_config,
																						scale_type_id=scale_type)
				flag_pre = 2

			elif scale_type in [binarize_type]:
				print('score binarize ', score_type, scale_type)
				score_normalized_pre = (score_query>thresh1).astype(np.float32)
				flag_pre = 2
			else:
				score_normalized_pre = score_query
		else:
			score_normalized_pre = score_query

		return score_normalized_pre, flag_pre

	## tf motif coefficients estimate
	def train_basic_2(self,gene_query_id,feature_type_id,model_type_id1,model_type_id2,sample_id_vec,peak_read=[],rna_exprs=[],pre_data_dict={},score_query_type=[],scale_type_id=0,scale_type_vec=[],thresh_dispersions_norm_query=-10,thresh_expr_mean_query=0,feature_imp_est=1,regularize_est=1,iter_num=1,sample_weight=[],LR_compare=0,save_mode=1,output_file_path='',select_config={}):
		
		# gene_query_id = self.gene_query_id
		if len(peak_read)==0:
			peak_read = self.peak_read
		if len(rna_exprs)==0:
			rna_exprs = self.rna_exprs

		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]

		# self.test_motif_peak_estimate_est_unit1(gene_query_id,pre_model_list1,model_type_id1,model_type_id2,sample_id_vec=[],data_vec=[],pre_data_dict={},select_config={},regularize_est=0,save_mode1=0,output_file_path='',iter_num=1):
		thresh_feature_query = select_config['thresh_feature_query']
		pre_model_dict = self.test_motif_peak_estimate_est_unit1(gene_query_id=gene_query_id,model_type_id1=model_type_id1,model_type_id2=model_type_id2,
															peak_read=peak_read,rna_exprs=rna_exprs,
															pre_model_list1=[],sample_id_vec=sample_id_vec,data_vec=[],
															pre_data_dict=pre_data_dict,
															thresh_feature_query=thresh_feature_query,
															feature_imp_est=feature_imp_est,
															regularize_est=regularize_est,
															save_mode1=0,
															output_file_path=output_file_path,
															iter_num=iter_num,
															LR_compare=0,
															select_config=select_config)

		self.pre_model_dict[gene_query_id] = pre_model_dict[gene_query_id]

		return pre_model_dict

	## gene-motif coefficients estimate
	def train_basic_cv_2(self,gene_query_id,score_query_dict,model_type_id,sample_id_vec,score_query_type,feature_type_id=1,peak_read=[],rna_exprs=[],scale_type_id=0,scale_type_vec=[],feature_imp_est=1,regularize_est=1,iter_id=0,num_fold=10,sample_weight=[],LR_compare=0,save_mode=1,output_file_path='',select_config={}):
	
		# gene_query_id = self.gene_query_id
		if len(peak_read)==0:
			peak_read = self.peak_read
		if len(rna_exprs)==0:
			rna_exprs = self.rna_exprs

		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]

		# self.test_motif_peak_estimate_est_unit1(gene_query_id,pre_model_list1,model_type_id1,model_type_id2,sample_id_vec=[],data_vec=[],pre_data_dict={},select_config={},regularize_est=0,save_mode1=0,output_file_path='',iter_num=1):
		thresh_feature_query = select_config['thresh_feature_query']

		score_query = score_query_dict[gene_query_id]
		feature_query_vec = score_query.columns
		# print('motif_query_id, score_query ', motif_query_id, score_query.shape)
		print('gene_query_id, score_query ', gene_query_id, score_query.shape)

		x = score_query
		# y = rna_exprs.loc[:,motif_query_id]
		y = rna_exprs.loc[:,gene_query_id]
		# print('x, y, motif_query_id ',x.shape,y.shape,motif_query_id)
		print('x, y, gene_query_id ',x.shape,y.shape,gene_query_id)

		df_pred = pd.DataFrame(index=sample_id,columns=['signal','pred'],dtype=np.float32)
		field_query_1 = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2','mutual_info']

		score_list1, t_coef_list1 = [], []
		feature_imp_list1, feature_imp_list2 = [], []
		self.train_mode_cv = 1
		for i1 in range(num_fold):
			sample_id_vec_1 = sample_id_vec[i1]
			# sample_vec = [sample_id_train, sample_id_valid, sample_id_test]
			sample_id_train, sample_id_valid, sample_id_test = sample_id_vec_1
			t_vec1 = self.test_feature_coef_est_pre1(x,y,model_type_id=model_type_id,
												sample_id_vec=sample_id_vec_1,
												sample_weight=sample_weight,
												feature_imp_est=feature_imp_est,
												regularize_est=regularize_est,
												LR_compare=LR_compare,
												save_mode=save_mode,
												output_file_path=output_file_path,
												select_config=select_config)

			alpha_vec, model_train_pre, x_train, y_train, feature_imp_est_vec, score_pred_vec = t_vec1
			score_pred1, df_pred1 = score_pred_vec[0], score_pred_vec[1]
			df_pred.loc[sample_id_test,'signal'] = df_pred1['signal']
			df_pred.loc[sample_id_test,'pred'] = df_pred1['pred']
			# print('score_pred ', np.asarray(score_pred1), i1, motif_query_id)
			print('score_pred ', np.asarray(score_pred1), i1, gene_query_id)
			score_list1.append(np.asarray(score_pred1))

			shap_values, base_values, expected_value = feature_imp_est_vec
			shap_values_mean_1 = np.mean(np.abs(shap_values),axis=0) 
			
			feature_imp_list1.append(shap_values_mean_1)
			feature_imp_list2.append(feature_imp_est_vec)
			t_coef_list1.append([alpha_vec,model_train_pre])

			pre_model_dict = self.test_motif_peak_estimate_est_unit1(gene_query_id=gene_query_id,
															model_type_id1=model_type_id1,model_type_id2=model_type_id2,
															peak_read=peak_read,rna_exprs=rna_exprs,
															pre_model_list1=[],
															sample_id_vec=sample_id_vec_1,
															data_vec=[],
															pre_data_dict=pre_data_dict,
															thresh_feature_query=thresh_feature_query,
															feature_imp_est=feature_imp_est,
															regularize_est=regularize_est,
															save_mode1=0,
															output_file_path=output_file_path,
															iter_num=iter_num,
															LR_compare=0,
															select_config=select_config)

			alpha_vec, model_train_pre, x_train, y_train, feature_imp_est_vec, score_pred_vec = t_vec1
			score_pred1, df_pred1 = score_pred_vec[0], score_pred_vec[1]
			df_pred.loc[sample_id_test,'signal'] = df_pred1['signal']
			df_pred.loc[sample_id_test,'pred'] = df_pred1['pred']
			# print('score_pred ', np.asarray(score_pred1), i1, motif_query_id)
			print('score_pred ', np.asarray(score_pred1), i1, gene_query_id)
			score_list1.append(np.asarray(score_pred1))

			shap_values, base_values, expected_value = feature_imp_est_vec
			shap_values_mean_1 = np.mean(np.abs(shap_values),axis=0)

			feature_imp_list1.append(shap_values_mean_1)
			feature_imp_list2.append(feature_imp_est_vec)
			t_coef_list1.append([alpha_vec,model_train_pre])

		df_score_pred1 = pd.DataFrame(index=range(num_fold),columns=field_query_1,data=np.asarray(score_list1))
		mean_score = df_score_pred1.mean(axis=0)
		df_score_pred = self.score_2a(df_pred['signal'],df_pred['pred'])

		# shap_values = [feature_imp_est_vec[0] for feature_imp_est_vec in feature_imp_list2]
		shap_values_mean = np.mean(np.asarray(feature_imp_list1),axis=0)
		
		model_id_pre = df_score_pred1['spearmanr'].idxmax()
		# print('model_id_pre ',model_id_pre,motif_query_id)
		print('model_id_pre ',model_id_pre,gene_query_id)

		t_coef_vec = t_coef_list1[model_id_pre]
		alpha_vec, model_train_pre = t_coef_vec
		feature_imp_est_pre = feature_imp_list2[model_id_pre]

		self.alpha_ = alpha_vec
		self.model_train_ = model_train_pre
		self.data_vec_ = [x,y]
		self.feature_imp_est_ = feature_imp_est_pre
		self.feature_query_id = feature_query_vec
		self.dict_feature_query_[gene_query_id] = feature_query_vec

		# self.score_pred_ = score_pred
		self.score_pred_ = df_score_pred
		# self.df_pred = df_pred
		score_pred_vec = [df_score_pred,df_pred,mean_score]
		self.score_pred_vec = score_pred_vec
		self.shap_values_mean = shap_values_mean
		self.feature_imp_list1 = feature_imp_list1
		self.feature_imp_list2 = feature_imp_list2

		return True

	## gene-motif coefficients estimate
	def train_basic_3(self,motif_query_id,score_query_dict,model_type_id,sample_id_vec,score_query_type,feature_type_id=1,peak_read=[],rna_exprs=[],scale_type_id=0,scale_type_vec=[],feature_imp_est=1,regularize_est=1,iter_id=0,sample_weight=[],LR_compare=0,save_mode=1,output_file_path='',select_config={}):
	
		# gene_query_id = self.gene_query_id
		if len(peak_read)==0:
			peak_read = self.peak_read
		if len(rna_exprs)==0:
			rna_exprs = self.rna_exprs

		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]

		score_query = score_query_dict[motif_query_id]
		feature_query_vec = score_query.columns
		print('motif_query_id, score_query ', motif_query_id, score_query.shape)

		x = score_query
		y = rna_exprs.loc[:,motif_query_id]
		print('x, y, motif_query_id ',x.shape,y.shape,motif_query_id)

		t_vec1 = self.test_feature_coef_est_pre1(x,y,model_type_id=model_type_id,
												sample_id_vec=sample_id_vec,
												sample_weight=sample_weight,
												feature_imp_est=feature_imp_est,
												regularize_est=regularize_est,
												LR_compare=LR_compare,
												save_mode=save_mode,
												output_file_path=output_file_path,
												select_config=select_config)

		alpha_vec, model_train_pre, x, y, feature_imp_est_vec, score_pred_vec = t_vec1

		self.alpha_ = alpha_vec
		self.model_train_ = model_train_pre
		self.data_vec_ = [x,y]
		self.feature_imp_est_ = feature_imp_est_vec
		self.feature_query_id = feature_query_vec
		self.dict_feature_query_[motif_query_id] = feature_query_vec
		score_pred = score_pred_vec[0]
		self.score_pred_ = score_pred
		self.score_pred_vec = score_pred_vec

		return True

	## gene-motif coefficients estimate
	def train_basic_cv_3(self,gene_query_id,score_query_dict,model_type_id,sample_id_vec,score_query_type,feature_type_id=1,peak_read=[],rna_exprs=[],scale_type_id=0,scale_type_vec=[],feature_imp_est=1,regularize_est=1,iter_id=0,num_fold=10,sample_weight=[],LR_compare=0,save_mode=1,output_file_path='',select_config={}):
	
		# gene_query_id = self.gene_query_id
		if len(peak_read)==0:
			peak_read = self.peak_read
		if len(rna_exprs)==0:
			rna_exprs = self.rna_exprs

		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]

		score_query = score_query_dict[gene_query_id]
		feature_query_vec = score_query.columns
		# print('motif_query_id, score_query ', motif_query_id, score_query.shape)
		print('gene_query_id, score_query ', gene_query_id, score_query.shape)

		x = score_query
		# y = rna_exprs.loc[:,motif_query_id]
		y = rna_exprs.loc[:,gene_query_id]
		# print('x, y, motif_query_id ',x.shape,y.shape,motif_query_id)
		print('x, y, gene_query_id ',x.shape,y.shape,gene_query_id)

		df_pred = pd.DataFrame(index=sample_id,columns=['signal','pred'],dtype=np.float32)
		field_query_1 = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2','mutual_info']

		score_list1, t_coef_list1 = [], []
		feature_imp_list1, feature_imp_list2 = [], []
		self.train_mode_cv = 1
		for i1 in range(num_fold):
			sample_id_vec_1 = sample_id_vec[i1]
			# sample_vec = [sample_id_train, sample_id_valid, sample_id_test]
			sample_id_train, sample_id_valid, sample_id_test = sample_id_vec_1
			t_vec1 = self.test_feature_coef_est_pre1(x,y,model_type_id=model_type_id,
												sample_id_vec=sample_id_vec_1,
												sample_weight=sample_weight,
												feature_imp_est=feature_imp_est,
												regularize_est=regularize_est,
												LR_compare=LR_compare,
												save_mode=save_mode,
												output_file_path=output_file_path,
												select_config=select_config)

			alpha_vec, model_train_pre, x_train, y_train, feature_imp_est_vec, score_pred_vec = t_vec1
			score_pred1, df_pred1 = score_pred_vec[0], score_pred_vec[1]
			df_pred.loc[sample_id_test,'signal'] = df_pred1['signal']
			df_pred.loc[sample_id_test,'pred'] = df_pred1['pred']
			# print('score_pred ', np.asarray(score_pred1), i1, motif_query_id)
			print('score_pred ', np.asarray(score_pred1), i1, gene_query_id)
			score_list1.append(np.asarray(score_pred1))

			shap_values, base_values, expected_value = feature_imp_est_vec
			shap_values_mean_1 = np.mean(np.abs(shap_values),axis=0) 
			
			feature_imp_list1.append(shap_values_mean_1)
			feature_imp_list2.append(feature_imp_est_vec)
			t_coef_list1.append([alpha_vec,model_train_pre])

		df_score_pred1 = pd.DataFrame(index=range(num_fold),columns=field_query_1,data=np.asarray(score_list1))
		mean_score = df_score_pred1.mean(axis=0)
		df_score_pred = self.score_2a(df_pred['signal'],df_pred['pred'])

		# shap_values = [feature_imp_est_vec[0] for feature_imp_est_vec in feature_imp_list2]
		shap_values_mean = np.mean(np.asarray(feature_imp_list1),axis=0)
		
		model_id_pre = df_score_pred1['spearmanr'].idxmax()
		# print('model_id_pre ',model_id_pre,motif_query_id)
		print('model_id_pre ',model_id_pre,gene_query_id)

		t_coef_vec = t_coef_list1[model_id_pre]
		alpha_vec, model_train_pre = t_coef_vec
		feature_imp_est_pre = feature_imp_list2[model_id_pre]

		self.alpha_ = alpha_vec
		self.model_train_ = model_train_pre
		self.data_vec_ = [x,y]
		self.feature_imp_est_ = feature_imp_est_pre
		self.feature_query_id = feature_query_vec
		self.dict_feature_query_[gene_query_id] = feature_query_vec

		# self.score_pred_ = score_pred
		self.score_pred_ = df_score_pred
		# self.df_pred = df_pred
		score_pred_vec = [df_score_pred,df_pred,mean_score]
		self.score_pred_vec = score_pred_vec
		self.shap_values_mean = shap_values_mean
		self.feature_imp_list1 = feature_imp_list1
		self.feature_imp_list2 = feature_imp_list2

		return True

	## gene-motif coefficients estimate
	def train_basic_pre1(self,x,y,model_type_id,sample_id_vec,motif_query_id='motif',score_query_dict={},score_query_type='score_ori',feature_type_id=1,peak_read=[],rna_exprs=[],scale_type_id=0,scale_type_vec=[],feature_imp_est=1,regularize_est=1,iter_id=0,sample_weight=[],LR_compare=0,type_id_score_vec=[0,0,1],save_mode=1,output_file_path='',select_config={}):
	
		# gene_query_id = self.gene_query_id
		if len(peak_read)==0:
			peak_read = self.peak_read
		if len(rna_exprs)==0:
			rna_exprs = self.rna_exprs

		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]

		# score_query = score_query_dict[motif_query_id]
		score_query = x
		feature_query_vec = score_query.columns
		print('motif_query_id, score_query ', motif_query_id, score_query.shape)

		# x = score_query
		# y = rna_exprs.loc[:,motif_query_id]
		print('x, y, motif_query_id ',x.shape,y.shape,motif_query_id)
		t_vec1 = self.test_feature_coef_est_pre1(x,y,model_type_id=model_type_id,
												sample_id_vec=sample_id_vec,
												sample_weight=sample_weight,
												feature_imp_est=feature_imp_est,
												regularize_est=regularize_est,
												LR_compare=LR_compare,
												type_id_score_vec=type_id_score_vec,
												save_mode=save_mode,
												output_file_path=output_file_path,
												select_config=select_config)

		alpha_vec, model_train_pre, x, y, feature_imp_est_vec, score_pred_vec = t_vec1

		self.alpha_ = alpha_vec
		self.model_train_ = model_train_pre
		self.data_vec_ = [x,y]
		self.feature_imp_est_ = feature_imp_est_vec
		self.feature_query_id = feature_query_vec
		self.dict_feature_query_[motif_query_id] = feature_query_vec
		score_pred = score_pred_vec[0]
		self.score_pred_ = score_pred
		self.score_pred_vec = score_pred_vec

		return t_vec1

	## gene-motif coefficients estimate
	def train_basic_cv_pre1(self,x,y,gene_query_id='gene_query',score_query_dict={},model_type_id='LR',sample_id_vec=[],score_query_type='score_ori',feature_type_id=1,peak_read=[],rna_exprs=[],scale_type_id=0,scale_type_vec=[],feature_imp_est=1,regularize_est=1,iter_id=0,num_fold=10,sample_weight=[],LR_compare=0,type_id_score_vec=[0,0,1],save_mode=1,output_file_path='',select_config={}):
	
		# gene_query_id = self.gene_query_id
		if len(peak_read)==0:
			peak_read = self.peak_read
		if len(rna_exprs)==0:
			rna_exprs = self.rna_exprs

		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]

		# score_query = score_query_dict[gene_query_id]
		score_query = x
		feature_query_vec = score_query.columns
		# print('motif_query_id, score_query ', motif_query_id, score_query.shape)
		print('gene_query_id, score_query ', gene_query_id, score_query.shape)

		# x = score_query
		# y = rna_exprs.loc[:,motif_query_id]
		# y = rna_exprs.loc[:,gene_query_id]
		# print('x, y, motif_query_id ',x.shape,y.shape,motif_query_id)
		print('x, y, gene_query_id ',x.shape,y.shape,gene_query_id)

		df_pred = pd.DataFrame(index=sample_id,columns=['signal','pred'],dtype=np.float32)
		field_query_1 = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2','mutual_info']

		score_query_list1, t_coef_list1 = [], []
		feature_imp_list1, feature_imp_list2 = [], []
		self.train_mode_cv = 1
		for i1 in range(num_fold):
			sample_id_vec_1 = sample_id_vec[i1]
			# sample_vec = [sample_id_train, sample_id_valid, sample_id_test]
			# print('sample_id_vec_1 ', sample_id_vec_1, i1)
			sample_id_train, sample_id_valid, sample_id_test = sample_id_vec_1
			t_vec1 = self.test_feature_coef_est_pre1(x,y,model_type_id=model_type_id,
												sample_id_vec=sample_id_vec_1,
												sample_weight=sample_weight,
												feature_imp_est=feature_imp_est,
												regularize_est=regularize_est,
												LR_compare=LR_compare,
												type_id_score_vec=type_id_score_vec,
												save_mode=save_mode,
												output_file_path=output_file_path,
												select_config=select_config)

			alpha_vec, model_train_pre, x_train, y_train, feature_imp_est_vec, score_pred_vec = t_vec1
			# score_pred1, df_pred1 = score_pred_vec[0], score_pred_vec[1]
			score_pred1, dict_pred1 = score_pred_vec[0], score_pred_vec[1] # score_pred1, dict_pred1: train, valid, test
			df_pred1 = dict_pred1['test']
			df_pred.loc[sample_id_test,'signal'] = df_pred1['signal']
			df_pred.loc[sample_id_test,'pred'] = df_pred1['pred']
			score_pred1_pre = score_pred1['test']
			# print('score_pred ', np.asarray(score_pred1_pre), i1, motif_query_id)	
			print('score_pred ', np.asarray(score_pred1_pre), i1, gene_query_id)
			score_query_list1.append(np.asarray(score_pred1_pre))

			shap_values, base_values, expected_value = feature_imp_est_vec
			shap_values_mean_1 = np.mean(np.abs(shap_values),axis=0) 
			
			feature_imp_list1.append(shap_values_mean_1)
			# feature_imp_list2.append(feature_imp_est_vec)
			# t_coef_list1.append([alpha_vec,model_train_pre])
			feature_imp_list2.append([alpha_vec,feature_imp_est_vec,model_train_pre])
			
		df_score_pred1 = pd.DataFrame(index=range(num_fold),columns=field_query_1,data=np.asarray(score_query_list1))
		mean_score = df_score_pred1.mean(axis=0)
		df_score_pred = self.score_2a(df_pred['signal'],df_pred['pred'])

		# shap_values = [feature_imp_est_vec[0] for feature_imp_est_vec in feature_imp_list2]
		shap_values_mean = np.mean(np.asarray(feature_imp_list1),axis=0)
		
		model_id_pre = df_score_pred1['spearmanr'].idxmax()
		# print('model_id_pre ',model_id_pre,motif_query_id)
		print('model_id_pre ',model_id_pre,gene_query_id)

		# t_coef_vec = t_coef_list1[model_id_pre]
		feature_imp_list2_query = feature_imp_list2[model_id_pre]
		# alpha_vec, model_train_pre = t_coef_vec
		alpha_vec, model_train_pre = feature_imp_list2_query[0], feature_imp_list2_query[2]
		# feature_imp_est_pre = feature_imp_list2[model_id_pre]
		feature_imp_est_pre = feature_imp_list2_query[1]
		print('feature_imp_est_pre ',len(feature_imp_est_pre))

		self.alpha_ = alpha_vec
		self.model_train_ = model_train_pre
		self.data_vec_ = [x,y]
		self.feature_imp_est_ = feature_imp_est_pre
		self.feature_query_id = feature_query_vec
		self.dict_feature_query_[gene_query_id] = feature_query_vec

		# self.score_pred_ = score_pred
		self.score_pred_ = df_score_pred
		# self.df_pred = df_pred
		dict_pred = {'test':df_pred}
		dict_score_pred = {'test':df_score_pred}
		# score_pred_vec = [df_score_pred,dict_pred,mean_score]
		score_pred_vec = [dict_score_pred,dict_pred,mean_score]

		self.score_pred_vec = score_pred_vec
		self.shap_values_mean = shap_values_mean
		self.feature_imp_list1 = feature_imp_list1
		self.feature_imp_list2 = feature_imp_list2

		return True

	## model parameter estimation
	# motif and peak coefficient estimate
	# def test_motif_peak_estimate_local_sub1_unit1(self,pre_model_list1,model_type_id1,model_type_id2,sample_id_vec,data_vec=[],pre_data_dict={},select_config={},regularize_est=0,save_mode1=0,output_file_path='',iter_num=1):
	def test_motif_peak_estimate_est_unit1(self,gene_query_id,model_type_id1,model_type_id2,peak_read=[],rna_exprs=[],pre_model_list1=[],sample_id_vec=[],data_vec=[],pre_data_dict={},thresh_feature_query=[],feature_imp_est=1,regularize_est=0,save_mode1=0,output_file_path='',iter_num=1,LR_compare=0,select_config={}):

		# data_vec = self.test_motif_peak_estimate_load_pre1(data_vec=data_vec,pre_data_dict=pre_data_dict)
		# tf_expr_local,peak_read_local,gene_expr,alpha_motif,beta_peak,peak_motif_prob,motif_binding_score = data_vec

		if len(sample_id_vec)==0:
			if ('train_valid_mode' in select_config):
				train_valid_mode = select_config['train_valid_mode']
			else:
				train_valid_mode = 0

			sample_id_vec = self.test_motif_peak_estimate_sample_idvec(fold_id=fold_id,train_valid_mode=train_valid_mode)
			self.sample_id_vec = sample_id_vec

		compare_mode = 0
		save_mode = 0
		save_mode1, save_mode2 = 1, 0
		t_list1 = []
		t_list2 = []
		# peak_mtx = peak_motif_prob

		if len(pre_model_list1)>0:
			model_1 = pre_model_list1[0]
			model_2 = pre_model_list1[1]

		# feature_imp_est=0
		# regularize_est=1
		output_file_path = self.save_file_path

		column_id = 'label_corr'
		peak_distance_thresh = 2000
		tol = 0.5
		df_gene_peak_query = self.df_gene_peak_query
		peak_label = df_gene_peak_query[column_id]
		df_gene_peak_query = df_gene_peak_query.loc[(peak_label.abs()>0)&(df_gene_peak_query['distance'].abs()<(peak_distance_thresh+tol)),:]
		query_id_ori = df_gene_peak_query.index.copy()
		df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])

		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]
		motif_data = self.motif_data
		peak_motif_mtx = self.test_peak_motif_query(gene_query_vec=[gene_query_id],df_gene_peak_query=df_gene_peak_query,motif_data=motif_data,select_config=select_config)
		motif_query_name = peak_motif_mtx.columns
		peak_loc_query = peak_motif_mtx.index
		peak_read_local = peak_read.loc[:,peak_loc_query]
		print('peak_read, peak_read_local ',peak_read.shape,peak_read_local.shape)

		t_value1 = np.sum(peak_motif_mtx,axis=0)
		motif_query_vec = motif_query_name[t_value1>0]

		## motif query with variable expr
		list_query_1 = []
		df_gene_annot_expr = self.df_gene_annot_expr
		df_gene_annot_expr_1 = df_gene_annot_expr.loc[motif_query_vec,:]
		if len(thresh_feature_query)>0:
			print('thresh_feature_query ', thresh_feature_query)
			motif_query_vec_ori = motif_query_vec.copy()
			print('motif_query_vec_ori ', len(motif_query_vec_ori))
			gene_name_query_ori = df_gene_annot_expr_1.index
			for thresh_vec_1 in thresh_feature_query:
				thresh_dispersions_norm_query, thresh_expr_mean_query = thresh_vec_1
				motif_query_vec = motif_query_vec_ori.copy()

				if thresh_dispersions_norm_query>-10:
					gene_highly_variable = gene_name_query_ori[df_gene_annot_expr_1['dispersions_norm']>thresh_dispersions_norm_query]
					motif_query_vec = pd.Index(motif_query_vec).intersection(gene_highly_variable,sort=False)

				if thresh_expr_mean_query>0:
					gene_query_1 = gene_name_query_ori[df_gene_annot_expr_1['means']>thresh_expr_mean_query]
					motif_query_vec = pd.Index(motif_query_vec).intersection(gene_query_1,sort=False)

				print('motif_query_vec ', len(motif_query_vec))
				data_query = df_gene_annot_expr.loc[motif_query_vec,:]
				output_file_path = self.save_file_path
				output_filename = '%s/test_motif_query.thresh_dispersions_norm_%.2f.thresh_expr_mean_%.2f.1.txt'%(output_file_path,thresh_dispersions_norm_query,thresh_expr_mean_query)
				data_query.to_csv(output_filename,sep='\t')
				list_query_1.extend(motif_query_vec)

			motif_query_vec = pd.Index(list_query_1).unique()

		motif_query_vec = pd.Index(motif_query_vec).difference([gene_query_id],sort=False)
		output_file_path = self.save_file_path
		# data_query = df_gene_annot_expr.loc[motif_query_vec,:]
		# # output_filename = '%s/test_motif_query.thresh2.1.txt'%(output_file_path)
		# output_filename = '%s/test_motif_query.thresh2.1_2.txt'%(output_file_path)
		# data_query = data_query.sort_values(by=['dispersions_norm','means'],ascending=False)
		# data_query.to_csv(output_filename,sep='\t')

		# data_query = df_gene_annot_expr.loc[motif_query_name,:]
		# output_filename = '%s/test_motif_query.ori.1.txt'%(output_file_path)
		# data_query = data_query.sort_values(by=['dispersions_norm','means'],ascending=False)
		# data_query.to_csv(output_filename,sep='\t')

		feature_query_vec = motif_query_vec
		peak_motif_mtx = peak_motif_mtx.loc[:,motif_query_vec]
		self.dict_feature_query_[gene_query_id] = {'motif':motif_query_vec,'peak':peak_loc_query}

		rna_exprs_unscaled = self.rna_exprs_unscaled
		motif_query_expr = rna_exprs_unscaled.loc[:,motif_query_vec]
		gene_query_expr = rna_exprs.loc[:,gene_query_id]
		print('motif_query_expr, peak_motif_mtx, gene_query_expr ',motif_query_expr.shape,peak_motif_mtx.shape,gene_query_expr.shape)

		data_vec_pre = [peak_motif_mtx,motif_query_expr,peak_read]
		self.data_vec = data_vec_pre

		motif_query_num, peak_num_query = len(motif_query_vec), len(peak_loc_query)
		alpha_motif = np.ones(motif_query_num,dtype=np.float32)
		column_id1, column_id2 = 'spearmanr','pval1'
		df_peak_corr = df_gene_peak_query.loc[gene_query_id,['peak_id',column_id1]]
		df_peak_corr.index = np.asarray(df_peak_corr['peak_id'])
		beta_peak = df_peak_corr.loc[peak_loc_query,column_id1].abs()
		print('beta_peak ',np.max(beta_peak),np.min(beta_peak),np.mean(beta_peak),np.median(beta_peak))
		peak_motif_prob = []
		motif_binding_score = []
		print('peak_read_local, motif_query_expr ',peak_read_local.shape,motif_query_expr.shape)

		# return

		model_type_vec = ['LR','XGBR','Lasso','ElasticNet']
		# model_type_id_1, model_type_id_2 = 2, 2
		model_type_id_1, model_type_id_2 = 3, 3
		# model_type_id1, model_type_id2 = model_type_vec[model_type_id_1], model_type_vec[model_type_id_2]
		print('model_type_id1, model_type_id2 ', model_type_id1, model_type_id2)
		model_1, model_2 = [], []
		#compare_mode, save_mode = 0, 1
		save_mode = 1
		save_mode1, save_mode2 = 0, 0
		pre_model_dict_1 = {'motif':[],'peak':[]}
		pre_model_dict1 = {gene_query_id:pre_model_dict_1}
		iter_num_search_pre = 10
		if 'iter_num_search_pre' in select_config:
			iter_num_search_pre = select_config['iter_num_search_pre']

		n_iter_search_randomized = select_config['n_iter_search_randomized']
		n_iter_search_randomized_2 = 30
		if 'n_iter_search_randomized_2' in select_config:
			n_iter_search_randomized_2 = select_config['n_iter_search_randomized_2']

		for iter_id in range(iter_num):
			if iter_id>0:
				model_2 = pre_model_dict1[gene_query_id]['peak'][0]

			if iter_id>iter_num_search_pre:
				n_iter_search_randomized_pre2 = np.min([n_iter_search_randomized_2,n_iter_search_randomized])
				select_config.update({'n_iter_search_randomized':n_iter_search_randomized_pre2})

			# model_2 = pre_model_list1[1]
			regularize_est = select_config['regularize_est2']
			iter_id1 = iter_id-iter_num
			# print('test_motif_peak_estimate_local_sub1_unit1, peak coefficient estimate ',iter_id)
			# print('peak coefficient estimate ',iter_id,gene_query_id)
			save_mode2 = 1
			# print('feature_imp_est ', feature_imp_est)
			peak_est_vec2 = self.test_peak_coef_est(peak_loc_query=peak_loc_query,
													feature_query_vec=feature_query_vec,
													gene_query_id=gene_query_id,
													alpha_motif=alpha_motif,
													peak_read=peak_read_local,rna_exprs=rna_exprs,
													feature_query_expr=motif_query_expr,
													gene_query_expr=gene_query_expr,
													peak_motif_mtx=peak_motif_mtx,
													sample_id_vec=sample_id_vec,iter_id=iter_id1,
													model_type_id=model_type_id2,model_train=model_2,
													LR_compare=LR_compare,
													save_mode=save_mode,
													feature_imp_est=feature_imp_est,
													regularize_est=regularize_est,
													save_mode1=save_mode1,save_mode2=save_mode2,
													output_file_path=output_file_path,
													select_config=select_config)

			# beta_peak_vec, score_2, model_2, shap_value_peak, feature_name2, score_compare_2, score_valid_2, y_pred_test2 = peak_est_vec2
			# beta_peak_vec, score_2, model_2, shap_value_peak, feature_name2, score_compare_2, score_train_valid_2, y_pred_test2 = peak_est_vec2
			beta_peak_vec, model_2, x2, y2, peak_feature_imp_est_vec = peak_est_vec2
			# print('peak_feature_imp_est_vec ', len(peak_feature_imp_est_vec[0]))
			# pre_model_list1[1] = model_2
			pre_model_dict1[gene_query_id].update({'peak':[model_2,beta_peak_vec,peak_feature_imp_est_vec]})
			beta_peak, beta0 = beta_peak_vec
			# t_list2.append(score_2)

			if iter_id>0:
				model_1 = pre_model_dict1[gene_query_id]['motif'][0]
			# model_1 = pre_model_list1[0]
			regularize_est = select_config['regularize_est1']
			# print('test_motif_peak_estimate_local_sub1_unit2, motif coefficient estimate ',iter_id)
			# print('motif coefficient estimate ',iter_id,gene_query_id)
			save_mode2 = 1
			motif_est_vec1 = self.test_motif_coef_est(peak_loc_query=peak_loc_query,
													feature_query_vec=feature_query_vec,
													gene_query_id=gene_query_id,
													beta_peak=beta_peak,
													peak_read=peak_read_local,rna_exprs=rna_exprs,
													feature_query_expr=motif_query_expr,
													gene_query_expr=gene_query_expr,
													peak_motif_mtx=peak_motif_mtx,
													sample_id_vec=sample_id_vec,iter_id=iter_id1,
													model_type_id=model_type_id1,model_train=model_1,
													LR_compare=LR_compare,
													save_mode=save_mode,
													feature_imp_est=feature_imp_est,
													regularize_est=regularize_est,
													save_mode1=save_mode1,save_mode2=save_mode2,
													output_file_path=output_file_path,
													select_config=select_config)

			# alpha_motif_vec, score_1, model_1, shap_value_motif, feature_name1, score_compare_1, score_train_valid_1, y_pred_test1 = motif_est_vec1
			alpha_motif_vec, model_1, x1, y1, motif_feature_imp_est_vec = motif_est_vec1
			# print('motif_feature_imp_est_vec ', len(motif_feature_imp_est_vec[0]))
			# pre_model_list1[0] = model_1
			pre_model_dict1[gene_query_id].update({'motif':[model_1,alpha_motif_vec,motif_feature_imp_est_vec]})
			alpha_motif, alpha0 = alpha_motif_vec
			# t_list2.append(score_1)
			t_list1.append([beta_peak_vec,alpha_motif_vec])

		pre_model_dict1[gene_query_id]['motif'].append(motif_query_vec)
		pre_model_dict1[gene_query_id]['peak'].append(peak_loc_query)

		# self.pre_model_dict[gene_query_id] = pre_model_dict1[gene_query_id]

		# return t_list1, t_list2
		return pre_model_dict1

	## model parameter estimation
	# motif and peak coefficient estimate
	# def test_motif_peak_estimate_local_sub1_unit1(self,pre_model_list1,model_type_id1,model_type_id2,sample_id_vec,data_vec=[],pre_data_dict={},select_config={},regularize_est=0,save_mode1=0,output_file_path='',iter_num=1):
	def test_motif_peak_estimate_est_unit1_pred(self,gene_query_id,model_type_id1,model_type_id2,peak_read=[],rna_exprs=[],pre_model_dict={},sample_id_vec=[],data_vec=[],pre_data_dict={},thresh_feature_query=[],feature_imp_est=1,regularize_est=0,save_mode1=0,output_file_path='',iter_num=1,LR_compare=0,select_config={}):

		# data_vec = self.test_motif_peak_estimate_load_pre1(data_vec=data_vec,pre_data_dict=pre_data_dict)
		# tf_expr_local,peak_read_local,gene_expr,alpha_motif,beta_peak,peak_motif_prob,motif_binding_score = data_vec

		# if len(sample_id_vec)==0:
		# 	if ('train_valid_mode' in select_config):
		# 		train_valid_mode = select_config['train_valid_mode']
		# 	else:
		# 		train_valid_mode = 0

		# 	sample_id_vec = self.test_motif_peak_estimate_sample_idvec(fold_id=fold_id,train_valid_mode=train_valid_mode)
		# 	self.sample_id_vec = sample_id_vec

		field_query = ['motif','peak']
		list_query_1 = []
		field_num2 = len(field_query)
		for i1 in range(field_num2):
			t_field_query = field_query[i1]
			# model_train,df_feature_query,beta0,feature_imp_est_vec,feature_query_id = pre_model_dict[gene_query_id][t_field_query]
			t_vec1 = pre_model_dict[gene_query_id][t_field_query]
			list_query_1.append(t_vec1)

		data_vec_pre = self.data_vec
		peak_motif_mtx,motif_query_expr,peak_read = data_vec_pre

		# tf_value1 = np.asarray(tf_expr_local)*np.asarray(alpha_motif) # shape: (sample_num,motif_num)
		feature_query_expr = motif_query_expr

		t_vec1 = list_query_1[0]
		t_vec2 = list_query_1[1]
		if len(t_vec1)>=5:
			model_train_motif,df_feature_query_motif,alpha0,motif_feature_imp_est_vec,motif_query_id = list_query_1[0]
			model_train_peak,df_feature_query_peak,beta0,peak_feature_imp_est_vec,peak_loc_query = list_query_1[1]
			alpha_motif = df_feature_query_motif['coef']
			beta_peak = df_feature_query_motif['coef']
		else:
			model_train_motif,alpha_motif_vec,motif_feature_imp_est_vec,motif_query_id = list_query_1[0]
			model_train_peak,beta_peak_vec,peak_feature_imp_est_vec,peak_loc_query = list_query_1[1]
			alpha_motif, alpha0 = alpha_motif_vec
			beta_peak, beta0 = beta_peak_vec

		tf_value1 = np.asarray(feature_query_expr)*np.asarray(alpha_motif) # shape: (sample_num,motif_num)
		peak_tf_expr_vec_2 = tf_value1.dot(peak_motif_mtx.T)      # shape: (sample_num,peak_num)
		peak_tf_expr_vec2 = peak_read*peak_tf_expr_vec_2  # shape: (sample_num,peak_num)
		x_peak = peak_tf_expr_vec2
		y = gene_query_expr

		peak_value1 = np.asarray(peak_read)*np.asarray(beta_peak)   # shape: (sample_num,peak_num), peak coefficients for the peak loci
		peak_tf_expr_vec_1 = peak_value1.dot(peak_motif_mtx)  # shape: (sample_num,motif_num)
		# tf_expr_local = rna_exprs.loc[:,feature_query_vec]
		# gene_expr = rna_exprs.loc[:,gene_query_id]
		peak_tf_expr_vec1 = feature_query_expr*peak_tf_expr_vec_1   # shape: (sample_num,motif_num)
		x_motif = peak_tf_expr_vec1
		y = gene_query_expr

		list_query_2 = [[model_train_motif,x_motif],[model_train_peak,x_peak]]
		list_query_3 = []
		sample_id_train, sample_id_valid, sample_id_test = sample_id_vec

		dict1 = dict()
		for i1 in range(query_num2):
			t_field_query = field_query[i1]
			t_vec_1 = list_query_2[i1]
			model_train, x = t_vec_1
			# x_train, y_train = x.loc[sample_id_train,:], y.loc[sample_id_train]
			x_valid, y_valid = x.loc[sample_id_valid,:], y.loc[sample_id_valid]
			x_test, y_test = x.loc[sample_id_test,:], y.loc[sample_id_test]

			y_valid_pred = model_train.predict(x_test)
			score_pred_valid = self.score_2a(np.ravel(y_valid), np.ravel(y_valid_pred))
			# df_pred = pd.DataFrame(index=sample_id_valid,columns=['signal','pred'])
			# df_pred['signal'], df_pred['pred'] = np.asarray(y_test), np.asarray(y_valid_pred)

			y_test_pred = model_train.predict(x_test)
			score_pred_test = self.score_2a(np.ravel(y_test), np.ravel(y_test_pred))
			df_pred = pd.DataFrame(index=sample_id_test,columns=['signal','pred'])
			df_pred['signal'], df_pred['pred'] = np.asarray(y_test), np.asarray(y_test_pred)

			list_query_3.append([score_pred_valid,score_pred_test,df_pred])

		dict1 = dict(zip(field_query,list_query_3))
		score_pred_valid_1 = [dict1[t_field_query][0] for t_field_query in field_query]
		score_pred_test_1 = [dict1[t_field_query][1] for t_field_query in field_query]
		score_pred_valid_peak = dict1['peak'][0]

		score_pred_valid_1 = np.asarray(score_pred_valid_1)
		score_pred_test_1 = np.asarray(score_pred_test_1)
		df_score_query_valid = pd.DataFrame(index=field_query,columns=score_pred_valid_peak.index,data=score_pred_valid_1,dtype=np.float32)
		df_score_query_test = pd.DataFrame(index=field_query,columns=score_pred_valid_peak.index,data=score_pred_test_1,dtype=np.float32)
		
		df_pred_1, df_pred_2 = dict1['motif'][2], dict1['peak'][2]
		df_pred_1.columns = ['signal','motif']
		df_pred_2.columns = ['signal','peak']
		df_pred2 = df_pred_2[['peak']]
		df_pred_test = pd.concat([df_pred_1,df_pred2],axis=1,join='outer',ignore_index=False)

		column_id1, column_id2 = 'spearmanr','pval1'
		query_id1 = df_score_query_valid[column_id1].idxmax()
		print('query_id1 ', query_id1)
		df_score_1 = df_score_query_test.loc[query_id1,:]
		df_pred_1 = df_pred_test.loc[:,['signal',query_id1]]

		return (df_score_query_valid,df_score_query_test,df_score_1,df_pred_1)

	# load data preparation
	def test_motif_peak_estimate_load_pre1(self,data_vec=[],pre_data_dict={}):

		if len(data_vec)==0:
			tf_expr_local, peak_read_local = pre_data_dict['tf_expr'], pre_data_dict['peak_read']
			gene_expr = pre_data_dict['gene_expr']
			motif_peak_local = pre_data_dict['motif_peak_local']
			# peak_loc = pre_data_dict['peak_loc']
			peak_loc = motif_peak_local.index
			motif_query_local = motif_peak_local.columns

			alpha_motif = pre_data_dict['alpha_motif']	# alpha_motif initial estimate
			beta_peak = pre_data_dict['beta_peak']		# beta_peak initial estimate

			peak_motif_prob = pre_data_dict['z_prob']    # peak_motif_prob initial estimate
			peak_motif_prob = peak_motif_prob.loc[peak_loc,motif_query_local]

			motif_binding_score = pre_data_dict['motif_binding_score']

			data_vec = (tf_expr_local,peak_read_local,gene_expr,alpha_motif,beta_peak,peak_motif_prob,motif_binding_score)

		return data_vec

	# load sample_id_vec
	def test_motif_peak_estimate_sample_idvec(self,fold_id,train_valid_mode=0):

		if fold_id<0:
			id_train, id_test = self.id_train, self.id_test
			sample_id_train, sample_id_test = self.sample_id_train, self.sample_id_test
		else:
			sample_id_train, sample_id_test, id_train, id_test = self.sample_idvec[fold_id]

		sample_id_train_ori = sample_id_train.copy()
		if train_valid_mode==1:
			sample_id_train, sample_id_valid, sample_id_train_, sample_id_valid_ = train_test_split(sample_id_train_ori,sample_id_train_ori,test_size=0.1,random_state=0)
		else:
			sample_id_valid = []

		sample_id_vec = [sample_id_train,sample_id_valid,sample_id_test]
		
		return sample_id_vec

	## motif-peak estimate: optimize preparation
	def test_motif_peak_estimate_optimize_pre2(self,gene_query_vec,feature_query_vec,gene_tf_mtx_pre1=[],gene_tf_prior_1=[],gene_tf_prior_2=[],select_config={}):

		## commented
		# gene_tf_mtx_pre1, gene_tf_mtx_pre2 = self.test_motif_peak_estimate_motif_prior_2(gene_query_vec=gene_query_vec,motif_query_vec=[],type_id=0,select_config=select_config,pre_load_1=pre_load_1,pre_load_2=pre_load_2)

		# self.gene_motif_prior_1 = gene_tf_mtx_pre1    # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
		# self.gene_motif_prior_2 = gene_tf_mtx_pre2    # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

		# self.gene_motif_prior_1 = gene_tf_prior_1   # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
		# self.gene_motif_prior_2 = gene_tf_prior_2   # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

		print('gene_motif_prior_1, gene_motif_prior_2 ', self.gene_motif_prior_1.shape, self.gene_motif_prior_2.shape)
		
		# return


	## motif-peak estimate: optimize preparation
	def test_motif_peak_estimate_optimize_pre1(self,gene_query_vec,feature_query_vec,gene_tf_mtx_pre1=[],gene_tf_prior_1=[],gene_tf_prior_2=[],select_config={}):

		## commented
		# gene_tf_mtx_pre1, gene_tf_mtx_pre2 = self.test_motif_peak_estimate_motif_prior_2(gene_query_vec=gene_query_vec,motif_query_vec=[],type_id=0,select_config=select_config,pre_load_1=pre_load_1,pre_load_2=pre_load_2)

		# self.gene_motif_prior_1 = gene_tf_mtx_pre1    # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
		# self.gene_motif_prior_2 = gene_tf_mtx_pre2    # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

		self.gene_motif_prior_1 = gene_tf_prior_1   # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
		self.gene_motif_prior_2 = gene_tf_prior_2   # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

		print('gene_motif_prior_1, gene_motif_prior_2 ', self.gene_motif_prior_1.shape, self.gene_motif_prior_2.shape)
		# return

		## initialize the output variable graph based on expression correlation
		# the edge set for the vertex-edge incidence matrix (VE) of output variables based on expression correlation
		thresh_corr_repsonse, thresh_pval_response = 0.2, 0.05
		if 'thresh_corr_repsonse' in select_config:
			thresh_corr_repsonse = select_config['thresh_corr_repsonse']
		if 'thresh_pval_response' in select_config:
			thresh_pval_repsonse = select_config['thresh_pval_repsonse']
		
		query_type_id = 0
		response_edge_set, response_graph_connect, response_graph_list = self.test_motif_peak_estimate_graph_pre1(feature_query_vec=gene_query_vec,
																													query_type_id=query_type_id,
																													select_config=select_config,
																													thresh_corr=thresh_corr_repsonse,
																													thresh_pval=thresh_pval_response,
																													load_mode=0)

		gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1 = response_graph_list
		self.gene_expr_corr_, self.gene_expr_pval_, self.gene_expr_pval_corrected = gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1
		print('response edge set ', len(gene_query_vec), len(response_edge_set))

		# return

		## initialize the input variable graph based on expression correlation
		# the edge set for the VE matrix of input variables based on expression correlation
		thresh_corr_predictor, thresh_pval_predictor = 0.3, 0.01
		if 'thresh_corr_predictor' in select_config:
			thresh_corr_predictor = select_config['thresh_corr_predictor']
		if 'thresh_pval_predictor' in select_config:
			thresh_pval_predictor = select_config['thresh_pval_predictor']

		query_type_id = 1
		# motif_query_name = self.motif_query_name_expr
		# print('motif_query_name ', len(motif_query_name))
		predictor_edge_set, predictor_graph_connect, predictor_graph_list = self.test_motif_peak_estimate_graph_pre1(feature_query_vec=feature_query_vec,
																														query_type_id=query_type_id,
																														select_config=select_config,
																														thresh_corr=thresh_corr_predictor,
																														thresh_pval=thresh_pval_predictor,
																														load_mode=0)

		gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2 = predictor_graph_list
		self.tf_expr_corr_, self.tf_expr_pval_, self.tf_expr_pval_corrected = gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2
		motif_query_vec = feature_query_vec
		print('predictor edge set ', len(motif_query_vec), len(predictor_edge_set))

		# return

		## initialize the VE matrix of output variables
		H = self.test_motif_peak_estimate_VE_init_(query_id=gene_query_vec,
													edge_set=response_edge_set,
													df_graph=gene_expr_corr_1)
		self.H_mtx = H
		print('VE matrix of response variable graph ', H.shape)

		## initialize the vertex-edge incidence matrix of input variables
		H_p = self.test_motif_peak_estimate_VE_init_(query_id=motif_query_vec,
													edge_set=predictor_edge_set,
													df_graph=gene_expr_corr_2)
		self.H_p = H_p
		print('VE matrix of predictor variable graph ', H_p.shape)

		return (response_edge_set, response_graph_connect, predictor_edge_set, predictor_graph_connect, H, H_p)

	# ## motif-peak estimate: motif-gene prior
	def test_motif_peak_estimate_graph_pre1(self,feature_query_vec,query_type_id,select_config={},thresh_corr=-2,thresh_pval=-1,load_mode=0):

		dict_gene_motif_prior_ = dict()
		motif_query_name = self.motif_query_name_expr

		input_file_path2 = '%s/data1_annotation_repeat1/test_basic_est_imp1'%(self.path_1)
		filename2 = '%s/test_motif_peak_estimate_df_motif_pre.Lpar3.repeat1.correction.copy.2.log0_scale0.txt'%(input_file_path2)
		df_motif_pre = pd.read_csv(filename2,index_col=0,sep='\t')
		tf_query_ens, motif_label = df_motif_pre.index, df_motif_pre['label']
		print('motif_query_ens with expr ', len(tf_query_ens))
		self.tf_query_ens = tf_query_ens

		# motif_query_name_ens = self.tf_query_ens
		edge_set_query, graph_connect_query = [], []
		# self.test_motif_peak_estimate_motif_prior_1_pre(select_config=select_config)

		# return

		## select tf by gene-tf expression correlation
		if (len(self.gene_expr_corr_)==0) or (load_mode==0):
			## initialize the output or input variable graph based on expression correlation
			gene_expr_corr_, gene_expr_pval_, gene_expr_pval_corrected = self.test_motif_peak_estimate_graph_1(query_type_id=query_type_id,select_config=select_config)
			# self.gene_expr_corr_, self.gene_expr_pval_, self.gene_expr_pval_corrected = gene_expr_corr_, gene_expr_pval_, gene_expr_pval_corrected

		if thresh_corr>-2:
			# thresh_corr_repsonse, thresh_pval_response = 0.2, 0.05
			edge_set_query, graph_connect_query = self.test_motif_peak_estimate_graph_edgeset_1(query_idvec_1=feature_query_vec,
																								query_idvec_2 = [],
																								graph_similarity=gene_expr_corr_,
																								graph_pval=gene_expr_pval_corrected,
																								query_type_id=query_type_id,
																								thresh_similarity=thresh_corr,
																								thresh_pval=thresh_pval)
		
		# print('response edge set ', len(gene_query_vec), len(response_edge_set))
		print('edge set ', len(feature_query_vec), len(edge_set_query))
		graph_list1 = [gene_expr_corr_, gene_expr_pval_, gene_expr_pval_corrected]

		return edge_set_query, graph_connect_query, graph_list1

	## motif-peak estimate: edge set for the VE matrix
	# edge set from output variable graph and input variable graph based on expression correlation or DORC score correlation
	def test_motif_peak_estimate_graph_edgeset_1(self,query_idvec_1,query_idvec_2=[],graph_similarity=[],graph_pval=[],query_type_id=0,thresh_similarity=0.2,thresh_pval=0.05):

		# expr_corr = self.gene_expr_corr
		# expr_corr_pval_= self.gene_expr_corr_pval

		# t_corr_mtx = expr_corr.loc[gene_query_vec,gene_query_vec]
		# t_pval_mtx = expr_corr_pval_.loc[gene_query_vec,gene_query_vec]

		symmetry_type = 0
		if len(query_idvec_2)==0:
			query_idvec_2 = query_idvec_1
			symmetry_type = 1

		print('thresh_similarity, thresh_pval ', thresh_similarity, thresh_pval)
		graph_simi_local = graph_similarity.loc[query_idvec_1,query_idvec_2]

		## graph_simi_local is symmetric matrix
		if symmetry_type==1:
			# graph_simi_local_upper = pd.DataFrame(index=graph_simi_local.index,columns=graph_simi_local.columns,data=np.triu(graph_simi_local))
			keep1 = np.triu(np.ones(graph_simi_local.shape)).astype(bool)
			graph_simi_local = graph_simi_local.where(keep1)

		flag_simi = (graph_simi_local.abs()>thresh_similarity)
		flag_1 = flag_simi

		graph_simi_local_pre1 = graph_simi_local[flag_simi]
		edge_set_simi_1 = graph_simi_local_pre1.stack().reset_index()
		edge_set_simi_1.columns = ['node1','node2','corr_value']

		# df_graph_simi_local = graph_simi_local.stack().reset_index()
		# df_graph_simi_local.columns = ['node1','node2','corr_value']

		if len(graph_pval)>0:
			graph_pval_local = graph_pval.loc[query_idvec_1,query_idvec_2]

			if symmetry_type==1:
				# keep1 = np.triu(np.ones(graph_pval_local.shape)).astype(bool)
				graph_pval_local = graph_pval_local.where(keep1)

			flag_pval = (graph_pval_local<thresh_pval)
			flag_1 = (flag_simi&flag_pval)

			graph_pval_local_pre2 = graph_pval_local[flag_simi]
			edge_set_pval_2 = graph_pval_local_pre2.stack().reset_index()
			edge_set_simi_1['pval'] = edge_set_pval_2[edge_set_pval_2.columns[-1]]

			graph_pval_local_pre1 = graph_pval_local[flag_pval]
			edge_set_pval_1 = graph_pval_local_pre1.stack().reset_index()

			graph_simi_local_pre2 = graph_simi_local[flag_pval]
			edge_set_simi_2 = graph_simi_local_pre2.stack().reset_index()

			edge_set_pval_1.columns = ['node1','node2','pval']
			edge_set_pval_1['corr_value'] = edge_set_simi_2[edge_set_simi_2.columns[-1]]

			id1 = (edge_set_pval_1['node1']!=edge_set_pval_1['node2'])
			edge_set_pval_1 = edge_set_pval_1[id1]
			edge_set_pval_1.reset_index(drop=True,inplace=True)
			
			output_filename = '%s/test_motif_peak_estimate_edge_set_pval_%d.txt'%(self.save_path_1,query_type_id)
			edge_set_pval_1.to_csv(output_filename,sep='\t',float_format='%.6E')
			print('edge_set_pval_1 ', edge_set_simi_1.shape)

		id1 = (edge_set_simi_1['node1']!=edge_set_simi_1['node2'])
		edge_set_simi_1 = edge_set_simi_1[id1]
		edge_set_simi_1.reset_index(drop=True,inplace=True)
			
		output_filename = '%s/test_motif_peak_estimate_edge_set_corr_%d.txt'%(self.save_path_1,query_type_id)
		edge_set_simi_1.to_csv(output_filename,sep='\t',float_format='%.6E')
		print('edge_set_simi_1 ', edge_set_simi_1.shape)

		num1 = flag_simi.sum().sum()
		num2 = 0
		if len(graph_pval)>0:
			num2 = flag_pval.sum().sum()

		num_1 = flag_1.sum().sum()
		if symmetry_type==0:
			num_2 = flag_simi.shape[0]*flag_simi.shape[1]
		else:
			n1 = flag_simi.shape[0]
			num_2 = n1*(n1-1)/2

		ratio1, ratio2, ratio3 = num1/(num_2*1.0), num2/(num_2*1.0), num_1/(num_2*1.0)
		print('graph_simi_local, graph_pval_local ',graph_simi_local.shape,num1,num2,num_1,ratio1,ratio2,ratio3)

		graph_simi_local_pre = graph_simi_local[flag_1]
		graph_pval_local_pre = graph_pval_local[flag_1]

		## commented
		# output_filename1 = 'test_motif_peak_estimate_graph_connect_%d.1.txt'%(query_type_id)
		# output_filename2 = 'test_motif_peak_estimate_graph_connect_%d.2.txt'%(query_type_id)
		# flag_simi.to_csv(output_filename1,sep='\t',float_format='%.6E')
		# flag_pval.to_csv(output_filename2,sep='\t',float_format='%.6E')

		## commented
		# b1 = np.where(flag_1>0)
		# id1, id2 = b1[0], b1[1]
		# query_id1 = query_idvec_1[id1]
		# query_id2 = query_idvec_2[id2]
		# edge_set = np.column_stack((query_id1,query_id2))
		# graph_connect = flag_1

		## id1 = graph_simi_local.index
		# t_data1 = pd.DataFrame(index=query_id1,columns=['node1','node2'])
		# t_data1['node1'] = query_id1
		# t_data1['node2'] = query_id2

		# if query_type_id==0:
		#   t_query1, t_query2 = 'Lpar3', 'Pdx1'
		#   print(graph_similarity.loc[t_query1,t_query2])
		#   print(graph_pval.loc[t_query1,t_query2])

		edge_set = graph_simi_local_pre.stack().reset_index()
		edge_set_pval_ = graph_pval_local_pre.stack().reset_index()

		edge_set.columns = ['node1','node2','corr_value']
		edge_set['pval'] = edge_set_pval_[edge_set_pval_.columns[-1]]
		id1 = (edge_set['node1']!=edge_set['node2'])
		edge_set = edge_set[id1]
		edge_set.reset_index(drop=True,inplace=True)

		graph_connect = flag_1
		output_filename = '%s/test_motif_peak_estimate_edge_set_%d.txt'%(self.save_path_1,query_type_id)
		edge_set.to_csv(output_filename,sep='\t',float_format='%.6E')
		print('edge_set ', edge_set.shape)

		return edge_set, graph_connect

	## motif-peak estimate: objective function
	# initialize variables that are not changing with parameters
	def test_motif_peak_estimate_optimize_init_pre1(self,gene_query_vec=[],sample_id=[],feature_query_vec=[],select_config={},pre_load_1=0,pre_load_2=0):
		
		## load data
		# self.gene_name_query_expr_
		# self.gene_highly_variable
		# self.motif_data; self.motif_data_cluster; self.motif_group_dict
		# self.pre_data_dict_1
		# self.motif_query_name_expr, self.motif_data_local_
		# self.test_motif_peak_estimate_pre_load(select_config=select_config)

		if len(gene_query_vec)==0:
			# gene_idvec = self.gene_idvec   # the set of genes
			# gene_query_vec = gene_idvec
			gene_query_vec = self.gene_highly_variable # the set of genes

		gene_query_num = len(gene_query_vec)
		self.gene_query_vec = gene_query_vec

		if len(motif_query_vec)==0:
			motif_query_vec= self.motif_query_name_expr

		# sample_id = self.meta_scaled_exprs_2.index   # sample id
		sample_num = len(sample_id)
		self.sample_id = sample_id

		## commented
		## prepare the correlated peaks of each gene
		# dict_peak_local_ =  self.test_motif_peak_estimate_peak_gene_association1(gene_query_vec=gene_query_vec,select_config=select_config)
		# key_vec = list(dict_peak_local_.keys())
		# print('dict_peak_local ', len(key_vec), key_vec[0:5])

		# self.dict_peak_local_ = dict_peak_local_

		## prepare the motif prior of each gene
		# pre_load_1 = 1
		# pre_load_2 = 0

		## commented
		gene_tf_mtx_pre1, gene_tf_mtx_pre2 = self.test_motif_peak_estimate_motif_prior_2(gene_query_vec=gene_query_vec,motif_query_vec=[],type_id=0,select_config=select_config,pre_load_1=pre_load_1,pre_load_2=pre_load_2)

		self.gene_motif_prior_1 = gene_tf_mtx_pre1  # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
		self.gene_motif_prior_2 = gene_tf_mtx_pre2  # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

		print('gene_motif_prior_1, gene_motif_prior_2 ', self.gene_motif_prior_1.shape, self.gene_motif_prior_2.shape)
		# return

		## initialize the output variable graph based on expression correlation
		# the edge set for the vertex-edge incidence matrix (VE) of output variables based on expression correlation
		thresh_corr_repsonse, thresh_pval_response = 0.2, 0.05
		if 'thresh_corr_repsonse' in select_config:
			thresh_corr_repsonse = select_config['thresh_corr_repsonse']
		if 'thresh_pval_response' in select_config:
			thresh_pval_repsonse = select_config['thresh_pval_repsonse']
		
		query_type_id = 0
		response_edge_set, response_graph_connect, response_graph_list = self.test_motif_peak_estimate_motif_prior_1(feature_query_vec=gene_query_vec,
																								query_type_id=query_type_id,
																								select_config=select_config,
																								thresh_corr=thresh_corr_repsonse,
																								thresh_pval=thresh_pval_response,
																								load_mode=0)

		gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1 = response_graph_list
		self.gene_expr_corr_, self.gene_expr_pval_, self.gene_expr_pval_corrected = gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1
		print('response edge set ', len(gene_query_vec), len(response_edge_set))

		# return

		## initialize the input variable graph based on expression correlation
		# the edge set for the VE matrix of input variables based on expression correlation
		thresh_corr_predictor, thresh_pval_predictor = 0.3, 0.05
		query_type_id = 1
		motif_query_name = self.motif_query_name_expr
		print('motif_query_name ', len(motif_query_name))
		predictor_edge_set, predictor_graph_connect, predictor_graph_list = self.test_motif_peak_estimate_motif_prior_1(feature_query_vec=motif_query_name,
																								query_type_id=query_type_id,
																								select_config=select_config,
																								thresh_corr=thresh_corr_predictor,
																								thresh_pval=thresh_pval_predictor,
																								load_mode=0)

		gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2 = predictor_graph_list
		print('predictor edge set ', len(motif_query_name), len(predictor_edge_set))

		# return

		## initialize the VE matrix of output variables
		H = self.test_motif_peak_estimate_VE_init_(query_id=gene_query_vec,
													edge_set=response_edge_set,
													df_graph=gene_expr_corr_1)
		self.H_mtx = H
		print('VE matrix of response variable graph ', H.shape)

		## initialize the vertex-edge incidence matrix of input variables
		H_p = self.test_motif_peak_estimate_VE_init_(query_id=motif_query_name,
													edge_set=predictor_edge_set,
													df_graph=gene_expr_corr_2)
		self.H_p = H_p
		print('VE matrix of predictor variable graph ', H_p.shape)

		return True

	## motif-peak estimate: parameter id preparation, beta matrix initialization
	def test_motif_peak_estimate_param_id(self,gene_query_vec,feature_query_vec,df_gene_motif_prior,intercept=True,save_mode=1):
		
		list1, list2 = [], []
		start_id1, start_id2 = 0, 0
		gene_query_num = len(gene_query_vec)

		df_gene_motif_prior = df_gene_motif_prior.loc[:,feature_query_vec]
		# feature_query_name = df_gene_motif_prior.columns
		dict_gene_param_id = dict()

		beta_mtx = pd.DataFrame(index=gene_query_vec,columns=feature_query_vec,data=0.0,dtype=np.float32)   # shape: gene_num by feature_num

		if intercept==True:
			beta_mtx.insert(0,'1',1)    # the first dimension corresponds to the intercept

		for i1 in range(gene_query_num):
			gene_query_id = gene_query_vec[i1]
			t_gene_query = gene_query_id
			t_motif_prior = df_gene_motif_prior.loc[t_gene_query,:]
			# b1 = np.where(t_motif_prior>0)[0]
			t_feature_query = feature_query_vec[t_motif_prior>0]
			str1 = ','.join(list(t_feature_query))

			t_feature_query_num = len(t_feature_query)
			list1.append(t_feature_query_num)
			list2.append(str1)
			
			start_id2 = start_id1+t_feature_query_num+int(intercept)
			dict_gene_param_id[t_gene_query] = np.arange(start_id1,start_id2)
			beta_mtx.loc[t_gene_query,t_feature_query] = 1

			print(t_gene_query,i1,start_id1,start_id2,t_feature_query_num)
			start_id1 = start_id2

		param_num = start_id2

		if save_mode==1:
			df_gene_motif_prior_str = pd.DataFrame(index=gene_query_vec,columns=['feature_num','feature_query'])
			df_gene_motif_prior_str['feature_num'] = list1
			df_gene_motif_prior_str['feature_query'] = list2

			annot_pre1 = str(len(gene_query_vec))
			# output_filename = '%s/df_gene_motif_prior_str_%s.txt'%(self.save_path_1,annot_pre1)
			output_filename = 'df_gene_motif_prior_str_%s.txt'%(annot_pre1)
			if os.path.exists(output_filename)==False:
				df_gene_motif_prior_str.to_csv(output_filename,sep='\t')
			print('df_gene_motif_prior_str, param num ', df_gene_motif_prior_str.shape, param_num)

		return dict_gene_param_id, param_num, beta_mtx

	## motif-peak estimate: objective function, estimate tf score beta
	# tf_score: gene_num by tf_num by cell_num
	def test_motif_peak_estimate_param_score_1(self,gene_query_vec,sample_id,feature_query_vec,y,beta,beta_mtx,beta_mtx_id,score_mtx=[],peak_motif_mtx=[],motif_score_dict={},motif_prior_type=0):
		
		beta_mtx_id = np.asarray(beta_mtx_init>0)
		beta_mtx = np.asarray(beta_mtx)
		beta_mtx[beta_mtx_id] = beta
		beta_mtx[~beta_mtx_id] = 0.0
		score_beta = score_mtx*beta_mtx # shape: cell_num by gene_num by motif_num

		# tf_score_beta = mtx1*np.asarray(beta_mtx) # shape: cell_num by gene_num by motif_num

		# y_pred = pd.DataFrame(index=sample_id,columns=gene_query_vec,data=0.0)
		# for i2 in range(sample_num):
		#   sample_id1 = sample_id[i2]
		#   y_pred.loc[sample_id1] = tf_score_beta[i2].dot(x.loc[sample_id1,motif_query_name])

		# squared_error = ((y_pred-y)**2).sum().sum()

		# return tf_score_beta, squared_error
		
		return score_beta, beta_mtx

	## motif group estimate: objective function, regularization 1
	def test_motif_peak_estimate_obj_regularize_1(self,beta_mtx,H_mtx):
		
		# query_num_edge = len(edge_set)
		# motif_query_num = beta_mtx.shape[1]
		# H = pd.DataFrame(index=gene_id,columns=range(edge_num),data=0.0)      
		# for i1 in range(edge_num):
		#   gene_query_id1, gene_query_id2 = edge_set[i1]
		#   t_gene_expr_corr = gene_expr_corr.loc[gene_query_id1,gene_query_id2]
		#   f_value = np.absolute(t_gene_expr_corr)
		#   H.loc[gene_query_id1,i1] = f_value
		#   H.loc[gene_query_id2,i1] = -(f_value>0)*f_value

		regularize_1 = beta_mtx.T.dot(H_mtx)

		return regularize_1

	## motif group estimate: objective function, regularization 2
	def test_motif_peak_estimate_obj_regularize_2(self,beta_param,motif_group_list,motif_group_vec):
		
		# group_num = len(motif_group_list)
		# vec1 = [np.linalg.norm(beta1[motif_group_list[i]]) for i in range(group_num)]
		# query_num1, query_num2 = beta_param.shape[0], beta_param.shape[1]
		# query_vec_1, query_vec_2 = beta_param.index, beta_param.columns
		# beta_param_square = np.square(beta_param)

		# vec1 = [np.linalg.norm(beta_param[idvec]) for idvec in motif_group_list]
		vec1 = [np.linalg.norm(beta_param.loc[:,feature_query_id].values,axis=1) for feature_query_id in motif_group_list]
		group_regularize_vec = np.asarray(vec1).T.dot(motif_group_vec)
		group_regularize = np.sum(group_regularize_vec)

		return group_regularize

	# likelihood function
	# def _motif_peak_prob_lik_constraint(self, params, x, y, alpha, lambda_regularize, pre_data_dict={}):
	def _motif_peak_prob_lik_constraint_pre1(self, params, x, y, feature_query_name, beta_mtx, regularize_lambda_query, regularize_lambda_type, motif_group_list, tf_group_list):
	
		# n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes		
		# c = state_id
		# flag = self._check_params(params)
		# if flag <= -2:
		# 	params = self.init_ou_params[state_id].copy()
		# 	lik = self._ou_lik_varied_constraint(params, state_id)
		# 	return lik

		beta0, beta1 = params[0], params[1:]

		param_vec = pd.Series(index=feature_query_name,data=beta1,dtype=np.float32)
		flag1 = 1
		if flag1>0:
			sample_num = x.shape[0]
			# y1 = np.zeros_like(sample_num,dtype=np.float32)+params[0]
			y_pred_vec = np.asarray([x.loc[:,feature_query_id].dot(param_vec[feature_query_id]) for feature_query_id in motif_group_list])
			y_pred = np.sum(y_pred_vec,axis=0)+params[0]
			vec1 = [np.linalg.norm(param_vec[feature_query_id]) for feature_query_id in motif_group_list]
			# vec1 = [[x.loc[:,feature_query_id].dot(param_vec[feature_query_id]),]]
			# y1 = 0
			# for feature_query_id in motif_group_list:
			# 	y1 += x.loc[:,feature_query_id].dot(param_vec[feature_query_id])

			group_regularize_vec_1 = np.asarray(vec1).dot(np.sqrt(motif_group_len_vec))
			group_regularize_term1 = np.sum(group_regularize_vec_1)
			
			tf_group_list = dict_tf_group_pre1.loc[gene_query_id1]
			# vec1 = [np.linalg.norm(beta_param[idvec]) for idvec in motif_group_list]
			# vec2 = [np.linalg.norm(beta_param.loc[:,feature_query_id].values,axis=1) for feature_query_id in motif_group_list]
			# group_regularize_vec_2 = np.asarray(vec2).T.dot(tf_group_vec)

			# vec2 = [np.linalg.norm(beta_param.loc[gene_query_id1,feature_query_id].values,axis=1)**2 for feature_query_id in tf_group_list]
			# vec2 = [np.linalg.norm(beta_param.loc[gene_query_id1,feature_query_id].values)**2 for feature_query_id in tf_group_list]
			vec2 = [np.linalg.norm(param_vec[feature_query_id])**2 for feature_query_id in tf_group_list]
			# group_regularize_vec_2 = vec2
			group_regularize_term2 = np.sum(group_regularize_vec_2)
			
			query_id_enrich_pos = dict_motif_enrich_1.loc[gene_query_id1,'pos']
			query_id_enrich_neg = dict_motif_enrich_1.loc[gene_query_id1,'neg']
			# query_id_enrich = pd.Index(query_id_enrich_pos).union(query_id_enrich_neg,sort=False)
			# print('query_id_enrich_pos, query_id_enrich_neg ',len(query_id_enrich_pos),len(query_id_enrich_neg))
			# print(query_id_enrich_pos)
			# print(query_id_enrich_neg)
			lambda_regularize_qeury = pd.Series(index=feature_query_name,data=1.0,dtype=np.float32) # Lasso regularization weight
			regularize_value_1 = 0.001
			regularize_lambda_query.loc[query_id_enrich_pos] = regularize_value_1
			regularize_lambda_query.loc[query_id_enrich_neg] = regularize_value_1

			L1 = 0.5*np.mean(np.square(y-y1))

			# Lasso_regularize_1 = np.linalg.norm(param_vec)
			Lasso_regularize_1 = np.sum(regularize_lambda_query*np.abs(param_vec))

			beta_mtx.loc[gene_query_id,feature_query_name] = param_vec
			# scale_factor_mtx = self.scale_factor_mtx
			# beta_mtx_scaled = beta_mtx*scale_factor_mtx
			beta_mtx_scaled = beta_mtx
			# regularize_1 = np.sum(np.sum(np.absolute(beta_mtx_scaled.T.dot(H_mtx))))
			# print('regularize_1',regularize_1, H_mtx.shape)

			regularize_graph_1 = np.sum(np.sum(np.absolute(beta_mtx_scaled.T.dot(H_mtx))))
			print('regularize_graph_1',regularize_1,H_mtx.shape,regularize_graph_1)

			t_value1 = beta_mtx_scaled.loc[[gene_query_id],:]
			regularize_graph_2 = np.sum(np.sum(np.absolute(t_value1.dot(H_p))))
			print('regularize_grahp_2',regularize_1,H_p.shape,regularize_graph_2)

			regularize_Lasso = np.sum(np.absolute(param_vec))
			regularize_1 = [regularize_graph_1,regularize_graph_2,Lasso_regularize_1]
			regularize_term = regularize_1.dot(regularize_lambda_type)

			ratio_1 = 1.0/np.sqrt(sample_num)
			lik = L1+regularize_term*ratio_1

		return lik

	# motif-peak estimate: optimize unit
	def test_motif_peak_estimate_optimize1_unit_pre1(self,initial_guess,x_train,y_train,feature_query_name,beta_mtx=[],pre_data_dict={},type_id_regularize=0,select_config={}):

		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG','Powell','CG','TNC','COBYLA','trust-constr',
						'dogleg','trust-ncg','trust-exact','trust-krylov']
		method_id = 2
		method_type_id = method_vec[method_id]
		method_type_id = 'SLSQP'
		# method_type_id = 'L-BFGS-B'

		id1, cnt = 0, 0
		flag1 = False
		small_eps = 1e-12
		# con1 = ({'type':'ineq','fun':lambda x:x-small_eps},
		#       {'type':'ineq','fun':lambda x:-x+1})

		# lower_bound, upper_bound = -100, 100
		lower_bound, upper_bound = pre_data_dict['param_lower_bound'], pre_data_dict['param_upper_bound']
		# regularize_lambda_vec = pre_data_dict['regularize_lambda_vec']
		# ratio1 = pre_data_dict['ratio']
		regularize_lambda_query = pre_data_dict['regularize_lambda_query']
		regularize_lambda_type = pre_data_dict['regularize_lambda_type']
		ratio_sample = pre_data_dict['ratio_sample']
		# motif_group_list, motif_group_vec = pre_data_dict['motif_group_list'], pre_data_dict['motif_group_vec']
		
		motif_group_list, tf_group_list = pre_data_dict['motif_group_list'], pre_data_dict['tf_group_list']
		# beta_mtx = self.beta_mtx
		print('beta_mtx ', beta_mtx.shape)

		con1 = ({'type':'ineq','fun':lambda x:x-lower_bound},
				{'type':'ineq','fun':lambda x:-x+upper_bound})
		bounds = scipy.optimize.Bounds(lower_bound,upper_bound)

		x_train = np.asarray(x_train)
		y_train = np.asarray(y_train)

		# flag1 = True
		iter_cnt1 = 0
		tol = 0.0001
		if 'tol_pre1' in pre_data_dict:
			tol = pre_data_dict['tol_pre1']

		while (flag1==False):
			try:
				start=time.time()
				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
				# 				method=method_type_id,constraints=con1,tol=tol,options={'disp':False})

				# _motif_peak_prob_lik_constraint_pre1(params, x, y, feature_query_name, lambda_regularize_query, regularize_lambda_type, motif_group_list, tf_group_list)
				res = minimize(self._motif_peak_prob_lik_constraint_pre1,initial_guess,args=(x_train,y_train,feature_query_name,beta_mtx,regularize_lambda_query,regularize_lambda_type,ratio_sample,motif_group_list,tf_group_list),
								method=method_type_id,constraints=con1,tol=tol,options={'disp':False})

				flag1 = True
				iter_cnt1 += 1
				stop=time.time()
				print('iter_cnt1 ',iter_cnt1,stop-start)
			except Exception as err:
				flag1 = False
				print("OS error: {0} motif_peak_estimate_optimize1_unit1 {1}".format(err,flag1))
				cnt = cnt + 1
				if cnt > 10:
					print('cannot find the solution! %d'%(cnt))
					break

		if flag1==True:
			param1 = res.x
			flag = self._check_params(param1,upper_bound=upper_bound,lower_bound=lower_bound)
			# t_value1 = param1[1:]
			small_eps = 1e-12
			# t_value1[t_value1<=0]=small_eps
			# t_value1[t_value1>1.0]=1.0
			quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
			t_vec1 = utility_1.test_stat_1(param1,quantile_vec=quantile_vec_1)
			print('parameter est ',flag, param1.shape, param1[0:10], t_vec1)
			return flag, param1

		else:
			print('did not find the solution!')
			return -2, initial_guess

	## motif-peak estimate: objective function
	# tf_score: gene_num by tf_num
	def test_motif_peak_estimate_obj_pre1(self,gene_query_vec,sample_id,feature_query_vec,beta=[],score_mtx=[],beta_mtx=[],peak_read=[],meta_exprs=[],motif_group_list=[],motif_group_vec=[],type_id_regularize=0):
		
		# gene_idvec = self.gene_idvec   # the set of genes
		# gene_query_vec = gene_idvec

		## load data
		gene_query_num = len(gene_query_vec)

		# sample_id = self.meta_scaled_exprs_2.index   # sample id
		# sample_id = self.sample_id
		sample_num = len(sample_id)

		# if len(feature_query_vec)==0:
		#   feature_query_vec = self.motif_query_name_expr
		feature_query_num = len(feature_query_vec)

		# self.meta_scaled_exprs = self.meta_scaled_exprs_2
		y = meta_exprs.loc[sample_id,gene_query_vec]
		x = meta_exprs.loc[sample_id,feature_query_vec]

		squared_error = 0
		# ratio1 = 1.0/sample_num
		ratio1 = 1.0/(sample_num*gene_query_num)

		motif_prior_type = 0
		# tf_score_beta = self.test_motif_peak_estimate_peak_motif_score_1(y,beta,tf_score_mtx=tf_score_mtx,motif_score_dict = {})
		# tf_score_beta, beta_mtx = self.test_motif_peak_estimate_param_score_1(gene_query_vec,sample_id,feature_query_vec,y,beta,tf_score_mtx=tf_score_mtx,motif_score_dict={},motif_prior_type=motif_prior_type)
		
		# beta_mtx_id = np.asarray(beta_mtx_init>0)
		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
		beta_mtx[beta_mtx_id1] = beta
		beta_mtx[beta_mtx_id2] = 0.0
		tf_score_beta = score_mtx*beta_mtx  # shape: cell_num by gene_num by motif_num

		print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
		y_pred = pd.DataFrame(index=sample_id,columns=gene_query_vec,data=0.0)
		for i2 in range(sample_num):
			sample_id1 = sample_id[i2]
			# t_value1 = x.loc[sample_id,motif_query_vec]
			# b1 = (t_value1>=0)
			# b2 = (t_value1<0)
			# y_pred.loc[sample_id1] = tf_score_beta[i2].dot(x.loc[sample_id1,motif_query_vec].abs())
			y_pred.loc[sample_id1] = np.sum(tf_score_beta[i2],axis=1)

		gene_query_idvec = gene_query_vec.copy()
		gene_query_num1 = len(gene_query_idvec)

		# df1 = pd.DataFrame(index=gene_query_idvec,columns=['motif_enrich1','motif_enrich2'])
		df1 = pd.DataFrame(index=feature_query_vec,columns=gene_query_idvec)
		df_motif_enrich_ori = []
		df_motif_enrich = df_motif_enrich_ori.loc[gene_query_vec,feature_query_vec]

		df_motif_group_pre1 = []

		df_motif_group = []
		motif_group_query = []

		df_motif_group_pre1 = pd.DataFrame(index=gene_query_idvec,columns=motif_group_query)
		df_motif_group_pre2 = pd.DataFrame(index=motif_group_query,columns=motif_query_name_expr)

		df_tf_group_pre1 = pd.DataFrame(index=gene_query_idvec,columns=tf_group_query)
		df_tf_group_pre2 = pd.DataFrame(index=tf_group_query,columns=feature_query_name)

		# motif_group_id_ori = df_motif_group_pre1.columns
		# motif_query_id_ori = df_motif_group_pre2.columns

		# tf_group_id_ori = df_tf_group_pre1.columns
		# tf_query_id_ori = df_tf_group_pre2.columns

		motif_group_id_ori = df_motif_group['group'].unique()
		motif_query_id_ori = df_motif_group.index

		tf_group_id_ori = df_tf_group['group'].unique()
		tf_query_id_ori = df_tf_group.index

		motif_group_len_vec = pd.DataFrame(index=motif_group_id_ori)
		motif_group_num = len(motif_group_vec)
		
		tf_group_len_vec = pd.DataFrame(index=tf_group_id_ori)
		tf_group_num = len(tf_group_vec)

		dict_motif_group_query = dict()
		dict_tf_group_query = dict()

		for i1 in range(motif_group_num):
			t_motif_group_id = motif_group_id_ori[i1]
			# motif_query_name = motif_query_id_ori[df_motif_group_pre2.loc[t_motif_group_id]>0]
			motif_query_name = motif_query_id_ori[df_motif_group['group']==t_motif_group_id]
			motif_group_len_vec[t_motif_group_id] = len(motif_query_name)
			dict_motif_group_query[t_motif_group_id] = motif_query_name

		for i1 in range(tf_group_num):
			tf_group_id = tf_group_id_ori[i1]
			# feature_query_name = tf_query_id_ori[df_tf_group_pre2.loc[tf_group_id]>0]
			feature_query_name = tf_query_id_ori[df_tf_group['group']==t_group_id]
			tf_group_len_vec[tf_group_id] = len(feature_query_name)
			dict_tf_group_query[tf_group_id] = feature_query_name

		for i1 in range(gene_query_num1):
			gene_query_id1 = gene_query_idvec[i1]
			# motif_group_id1 = motif_group_id_ori[df_motif_group_pre1.loc[gene_query_id1,:]>0]	# motif group presented in the locus
			motif_group_dict = dict()
			# feature_query_id = df_gene_motif_prior[gene_query_id1]
			motif_query_pos, motif_query_neg = df_gene_motif_prior[gene_query_id1]
			feature_query_id = pd.Index(motif_query_pos).union(motif_query_neg,sort=False)

			for t_motif_group_id in motif_group_id1:
				# motif_query_name = motif_query_id_ori[df_motif_group_pre2.loc[t_motif_group_id]>0]
				motif_query_name_pre1 = dict_motif_group_query.loc[t_motif_group_id]
				motif_query_name = pd.Index(motif_query_name_pre1).intersection(feature_query_id,sort=False)
				# motif_group_list.append(motif_query_name)
				if len(motif_query_name)>0:
					motif_group_dict.update({t_motif_group_id:motif_query_name})

			dict_motif_group_pre1[gene_query_id1] = motif_group_dict

			tf_group_dict = dict()
			for tf_group_id in tf_group_id1:
				# feature_query_name = tf_query_id_ori[df_tf_group_pre2.loc[tf_group_id]>0]
				feature_query_name_pre1 = dict_tf_group_query.loc[tf_group_id]
				feature_query_name = pd.Index(feature_query_name_pre1).intersection(feature_query_id,sort=False)
				# tf_group_list.append(feature_query_name)
				if len(feature_query_name)>0:
					tf_group_dict.update({tf_group_id:feature_query_name})

			dict_tf_group_pre1[gene_query_id1] = tf_group_dict

		save_file_path = '../data2'
		file_path1 = save_file_path
		thresh_dispersions_norm_peak, thresh_dispersions_norm = select_config['thresh_dispersions_norm_peak'], select_config['thresh_dispersions_norm']
		# input_file_path1 = '%s/thresh%s/peak_corr'%(self.save_path_1,thresh_dispersions_norm)
		# input_file_path2 = '%s/thresh%s/gene_score_pre1'%(self.save_path_1,thresh_dispersions_norm)
		
		filename_prefix_1 = 'thresh_peak%s.thresh%s'%(thresh_dispersions_norm_peak,thresh_dispersions_norm)
		# filename_prefix_1 = 'thresh_peak-10.thresh1.0'%(thresh_peak_dispersion_norm,)
		peak_thresh1, peak_thresh2 = 250, 500
		peak_distance_thresh_pre = 2000
		peak_thresh3 = peak_distance_thresh_pre
		peak_thresh_3 = 2500
		filename_annot_pre1 = '%d.%d.%d'%(peak_thresh1, peak_thresh2, peak_thresh_3)
		filename_annot_1 = '%d.%d.%d'%(peak_thresh1, peak_thresh2, peak_thresh3)

		tol = 0.5
		df_gene_peak_query_pre = df_gene_peak_query.loc[df_gene_peak_query['peak_distance'].abs()<(peak_distance_thresh_pre+tol),:]
		
		flag_gene_tf_sel=0
		motif_data_expr = self.motif_data_expr
		if flag_gene_tf_sel>0:
			filename_annot1 = '1'

		flag_motif_query=0
		if flag_motif_query>0:
			filename_prefix = '%s.%s'%(filename_prefix_1,filename_annot_1)
			input_filename = '%s/test_motif_enrichment.neighbor_query.%s.genome-wide.weighted.log0_scale3.pre1.2.annot1.txt'%(save_file_path,filename_prefix)
			df_motif_enrich = pd.read_csv(input_filename,index_col=0,sep='\t')

			field_query1, field_query2 = 'motif_query_pos_pre_1', 'motif_query_neg_pre_1'
			for i1 in range(gene_query_num1):
				gene_query_id = gene_query_idvec[i1]
				motif_enrich_pos_str = df_motif_enrich.loc[gene_query_id,field_query1]
				motif_enrich_neg_str = df_motif_enrich.loc[gene_query_id,field_query2]

				# motif_enrich_pos_num = df_motif_enrich.loc[gene_query_id,'%s.num'%(field_query1)]
				# motif_enrich_neg_num = df_motif_enrich.loc[gene_query_id,'%s.num'%(field_query2)]

				if (pd.isna(motif_query_pos_str)==False) and (motif_query_pos_str!=''):
				# if motif_enrich_pos_num>0:
					motif_enrich_pos = motif_enrich_pos_str.split(',')

				if (pd.isna(motif_query_neg_str)==False) and (motif_query_neg_str!=''):
				# if motif_enrich_neg_num>0:
					motif_enrich_neg = motif_enrich_neg_str.split(',')
				df_motif_enrich[gene_query_id] = [motif_enrich_pos,motif_enrich_neg]

		flag_basic_train=0
		if flag_basic_train>0:
			dict_basic_train = dict()
			for i1 in range(gene_query_num1):
				gene_query_id = gene_query_idvec[i1]
				## tf motif coefficients estimate
				print('gene_query_id ',gene_query_id,i1)
				meta_exprs_2 = self.meta_exprs_2
				quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
				t_value1 = utility_1.test_stat_1(meta_exprs_2,quantile_vec=quantile_vec_1)
				print('meta_exprs_2 ', t_value1)
				model_pre = _Base2_train1(peak_read=peak_read,rna_exprs=rna_exprs,
											rna_exprs_unscaled=meta_exprs_2,
											df_gene_peak_query=df_gene_peak_query,
											df_gene_annot_expr=df_gene_annot_expr,
											motif_data=motif_data_expr,
											data_dir=save_file_path,
											select_config=select_config)

				pre_model_dict = dict()
				if train_type_id==1:
					# model_type_id1, model_type_id2 = 'Lasso', 'Lasso'
					model_type_id1, motif_type_id2 = 'Lasso',                                                                                                                                    
					model_type_id1, model_type_id2 = model_type_id, model_type_id
					sample_weight = []
					iter_num = select_config['iter_num_est']
					try:
						model_pre.train_basic_2(gene_query_id=gene_query_id,feature_type_id=feature_type_id,
										model_type_id1=model_type_id1,model_type_id2=model_type_id2,
										sample_id_vec=sample_id_vec,
										score_query_type=score_query_type,
										scale_type_id=scale_type_id,scale_type_vec=scale_type_vec_pre,
										feature_imp_est=feature_imp_est_mode,
										regularize_est=regularize_est,
										iter_num=iter_num,
										sample_weight=sample_weight,
										LR_compare=0,
										save_mode=save_mode,
										output_file_path=output_file_path,
										select_config=select_config)

					except Exception as error:
						print('error! ',error,gene_query_id1,i1)
						continue

					pre_model_dict_1 = model_pre.model_query_2(gene_query_id=gene_query_id,feature_imp_est=1,select_config=select_config)
					pre_model_dict = pre_model_dict_1[gene_query_id]

					dict_basic_train[gene_query_id] = pre_model_dict

		for i1 in range(gene_query_num1):
			gene_query_id1 = gene_query_idvec[i1]
			y_pred1 = y_pred.loc[:,gene_query_id1]
			y_ori = y.loc[:,gene_query_id1]
			squared_error = np.sum((y_pre1-y_ori)**2)

			motif_group_list = dict_motif_group_pre1.loc[gene_query_id1]
			# vec1 = [np.linalg.norm(beta_param[idvec]) for idvec in motif_group_list]
			# vec1 = [np.linalg.norm(beta_param.loc[gene_query_id1,feature_query_id].values,axis=1) for feature_query_id in motif_group_list]
			# group_regularize_vec_1 = np.asarray(vec1).T.dot(np.sqrt(motif_group_len_vec))

			param_vec = pd.Series(index=feature_query_name,data=param_vec_pre[1:],dtype=np.float32)
			# param_vec = beta_param.loc[gene_query_id1]
			# vec1 = [np.linalg.norm(beta_param.loc[gene_query_id1,feature_query_id].values) for feature_query_id in motif_group_list]
			vec1 = [np.linalg.norm(param_vec[feature_query_id].values) for feature_query_id in motif_group_list]
			# group_regularize_vec_1 = np.asarray(vec1).dot(np.sqrt(motif_group_len_vec))
			# group_regularize_term1 = np.sum(group_regularize_vec_1)
			
			tf_group_list = dict_tf_group_pre1.loc[gene_query_id1]
			# vec1 = [np.linalg.norm(beta_param[idvec]) for idvec in motif_group_list]
			# vec2 = [np.linalg.norm(beta_param.loc[:,feature_query_id].values,axis=1) for feature_query_id in motif_group_list]
			# group_regularize_vec_2 = np.asarray(vec2).T.dot(tf_group_vec)

			# vec2 = [np.linalg.norm(beta_param.loc[gene_query_id1,feature_query_id].values,axis=1)**2 for feature_query_id in tf_group_list]
			# vec2 = [np.linalg.norm(beta_param.loc[gene_query_id1,feature_query_id].values)**2 for feature_query_id in tf_group_list]
			vec2 = [np.linalg.norm(param_vec[feature_query_id])**2 for feature_query_id in tf_group_list]
			# group_regularize_vec_2 = vec2
			group_regularize_term2 = np.sum(group_regularize_vec_2)
			
			query_id_enrich_pos = dict_motif_enrich_1.loc[gene_query_id1,'pos']
			query_id_enrich_neg = dict_motif_enrich_1.loc[gene_query_id1,'neg']
			query_id_enrich = pd.Index(query_id_enrich_pos).union(query_id_enrich_neg,sort=False)
			print('query_id_enrich_pos, query_id_enrich_neg ',len(query_id_enrich_pos),len(query_id_enrich_neg))
			print(query_id_enrich_pos)
			print(query_id_enrich_neg)

			Lasso_regularize_vec_pre1 = pd.Series(index=feature_query_name,data=1.0,dtype=np.float32) # Lasso regularization weight
			regularize_value_1 = 0.001
			Lasso_regularize_vec_pre1.loc[query_id_enrich_pos] = regularize_value_1
			Lasso_regularize_vec_pre1.loc[query_id_enrich_neg] = regularize_value_1

			# # Lasso_regularize_1 = np.linalg.norm(param_vec)
			# Lasso_regularize_1 = np.sum(Lasso_regularize_vec_pre1*np.abs(param_vec))

			# beta_mtx.loc[gene_query_id,feature_query_name] = param_vec
			# # scale_factor_mtx = self.scale_factor_mtx
			# # beta_mtx_scaled = beta_mtx*scale_factor_mtx
			# beta_mtx_scaled = beta_mtx
			# # regularize_1 = np.sum(np.sum(np.absolute(beta_mtx_scaled.T.dot(H_mtx))))
			# # print('regularize_1',regularize_1, H_mtx.shape)

			# regularize_graph_1 = np.sum(np.sum(np.absolute(beta_mtx_scaled.T.dot(H_mtx))))
			# print('regularize_graph_1',regularize_1,H_mtx.shape,regularize_graph_1)

			# regularize_graph_2 = np.sum(np.sum(np.absolute(beta_mtx_scaled.dot(H_p))))
			# print('regularize_grahp_2',regularize_1,H_p.shape,regularize_graph_2)

			# # for i2 in range(gene_neighbor_query_num):
			# # 	x = 1

			# scale_factor_mtx = self.scale_factor_mtx
			# beta_mtx.loc[gene_query_id1] = param_vec
			# # beta_mtx_scaled = beta_mtx*scale_factor_mtx

			# beta_mtx_scaled = beta_mtx
			# regularize_1 = np.sum(np.sum(np.absolute(beta_mtx_scaled.T.dot(H_mtx))))
			# print('regularize_1',regularize_1, H_mtx.shape)

			## regularization based on tf groups
			# regularize_2 = self.test_motif_peak_estimate_obj_regularize_2(beta_mtx,motif_group_list,motif_group_vec)
			# print('regularize_2',regularize_2)

			## regularization based on gene expression correlation of tfs, graph-guided fused Lasso
			# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
			# if type_id_regularize==1:
			# 	regularize_3 = np.sum(np.sum(np.absolute(beta_mtx_scaled.dot(H_p))))
			# 	print('regularize_3',regularize_3, H_p.shape)
			# else:
			# 	regularize_3 = 0.0

			type_id_regularize = 0
			regularize_lambda_query = pre_data_dict['regularize_lambda_query']
			regularize_lambda_type = pre_data_dict['regularize_lambda_type']
			ratio_sample = pre_data_dict['ratio_sample']
			# motif_group_list, motif_group_vec = pre_data_dict['motif_group_list'], pre_data_dict['motif_group_vec']
			
			motif_group_list, tf_group_list = pre_data_dict['motif_group_list'], pre_data_dict['tf_group_list']
			# beta_mtx = self.beta_mtx
			print('beta_mtx ', beta_mtx.shape)

			flag, param_est = self.test_motif_peak_estimate_optimize1_unit_pre1(initial_guess,x_train,y_train,
																				feature_query_name=feature_query_name,
																				beta_mtx=beta_mtx,
																				pre_data_dict=pre_data_dict,
																				type_id_regularize=type_id_regularize,
																				select_config=select_config)

		squared_error = ((y_pred-y)**2).sum().sum()*ratio1
		# y_pred.columns = ['pred.%s'%(t_gene_query) for t_gene_query in gene_query_vec]
		# df1 = pd.concat([y,y_pred],axis=1,join='outer',ignore_index=False,keys=None,levels=None,names=None,
		#                   verify_integrity=False,copy=True)

		# output_filename = '%s/test_peak_motif_estimate_pred.1.txt'%(self.save_path_1)
		# df1.to_csv(output_filename,sep='\t',float_format='%.6E')
		# print('objective function value ', squared_error)

		# return squared_error

		res = minimize(self._motif_peak_prob_lik_constraint_pre1,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
						constraints=con1,tol=1e-5,options={'disp':False})

		Lasso_1 = np.sum(np.absolute(beta))
		# motif_group = self.motif_group
		# motif_group = self.df_motif_group_query.loc[feature_query_vec,'group']
		gene_expr_corr = self.gene_expr_corr_
		print('H_mtx, H_p ',self.H_mtx.shape, self.H_p.shape)
		H_mtx = self.H_mtx.loc[gene_query_vec,:]
		H_p = self.H_p.loc[feature_query_vec,:]
		print('H_mtx, H_p ',H_mtx.shape, H_p.shape)

		## regularization based on gene expression correlation, graph-guided fused Lasso
		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
		# scale_factor_mtx = np.ones(beta_mtx.shape,dtype=np.float32)
		scale_factor_mtx = self.scale_factor_mtx
		beta_mtx_scaled = beta_mtx*scale_factor_mtx
		regularize_1 = np.sum(np.sum(np.absolute(beta_mtx_scaled.T.dot(H_mtx))))
		print('regularize_1',regularize_1, H_mtx.shape)

		## regularization based on tf groups
		regularize_2 = self.test_motif_peak_estimate_obj_regularize_2(beta_mtx,motif_group_list,motif_group_vec)
		print('regularize_2',regularize_2)

		## regularization based on gene expression correlation of tfs, graph-guided fused Lasso
		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
		if type_id_regularize==1:
			regularize_3 = np.sum(np.sum(np.absolute(beta_mtx_scaled.dot(H_p))))
			print('regularize_3',regularize_3, H_p.shape)
		else:
			regularize_3 = 0.0

		self.lambda_vec = [1E-03,[1E-03,1E-03,1E-03]]
		Lasso_eta_1, lambda_vec_2 = self.lambda_vec
		regularizer_vec_2 = np.asarray([regularize_1,regularize_2,regularize_3])

		# type_id_regularize = 1
		# if type_id_regularize==1:
		#   regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2.dot(lambda_vec_2)
		# else:
		#   regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2[0:2].dot(lambda_vec_2[0:2])
		regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2.dot(lambda_vec_2)
		
		squared_error_ = squared_error + regularize_pre
		print('squared_error_, squared_error, regularize_pre ', squared_error_, squared_error, regularize_pre)

		return squared_error_

	## motif-peak estimate: objective function
	# tf_score: gene_num by tf_num
	def test_motif_peak_estimate_obj_1_ori(self,gene_query_vec,sample_id,feature_query_vec,beta=[],score_mtx=[],beta_mtx=[],peak_read=[],meta_exprs=[],motif_group_list=[],motif_group_vec=[],type_id_regularize=0):
		
		# gene_idvec = self.gene_idvec   # the set of genes
		# gene_query_vec = gene_idvec

		## load data
		gene_query_num = len(gene_query_vec)

		# sample_id = self.meta_scaled_exprs_2.index   # sample id
		# sample_id = self.sample_id
		sample_num = len(sample_id)

		# if len(feature_query_vec)==0:
		#   feature_query_vec = self.motif_query_name_expr
		feature_query_num = len(feature_query_vec)

		# self.meta_scaled_exprs = self.meta_scaled_exprs_2
		y = meta_exprs.loc[sample_id,gene_query_vec]
		x = meta_exprs.loc[sample_id,feature_query_vec]

		squared_error = 0
		# ratio1 = 1.0/sample_num
		ratio1 = 1.0/(sample_num*gene_query_num)

		motif_prior_type = 0
		# tf_score_beta = self.test_motif_peak_estimate_peak_motif_score_1(y,beta,tf_score_mtx=tf_score_mtx,motif_score_dict = {})
		# tf_score_beta, beta_mtx = self.test_motif_peak_estimate_param_score_1(gene_query_vec,sample_id,feature_query_vec,y,beta,tf_score_mtx=tf_score_mtx,motif_score_dict={},motif_prior_type=motif_prior_type)
		
		# beta_mtx_id = np.asarray(beta_mtx_init>0)
		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
		beta_mtx[beta_mtx_id1] = beta
		beta_mtx[beta_mtx_id2] = 0.0
		tf_score_beta = score_mtx*beta_mtx  # shape: cell_num by gene_num by motif_num

		print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
		y_pred = pd.DataFrame(index=sample_id,columns=gene_query_vec,data=0.0)
		for i2 in range(sample_num):
			sample_id1 = sample_id[i2]
			# t_value1 = x.loc[sample_id,motif_query_vec]
			# b1 = (t_value1>=0)
			# b2 = (t_value1<0)
			# y_pred.loc[sample_id1] = tf_score_beta[i2].dot(x.loc[sample_id1,motif_query_vec].abs())
			y_pred.loc[sample_id1] = np.sum(tf_score_beta[i2],axis=1)
			
		squared_error = ((y_pred-y)**2).sum().sum()*ratio1
		# y_pred.columns = ['pred.%s'%(t_gene_query) for t_gene_query in gene_query_vec]
		# df1 = pd.concat([y,y_pred],axis=1,join='outer',ignore_index=False,keys=None,levels=None,names=None,
		#                   verify_integrity=False,copy=True)

		# output_filename = '%s/test_peak_motif_estimate_pred.1.txt'%(self.save_path_1)
		# df1.to_csv(output_filename,sep='\t',float_format='%.6E')
		# print('objective function value ', squared_error)

		# return squared_error

		Lasso_1 = np.sum(np.absolute(beta))
		# motif_group = self.motif_group
		# motif_group = self.df_motif_group_query.loc[feature_query_vec,'group']
		gene_expr_corr = self.gene_expr_corr_
		print('H_mtx, H_p ',self.H_mtx.shape, self.H_p.shape)
		H_mtx = self.H_mtx.loc[gene_query_vec,:]
		H_p = self.H_p.loc[feature_query_vec,:]
		print('H_mtx, H_p ',H_mtx.shape, H_p.shape)

		## regularization based on gene expression correlation, graph-guided fused Lasso
		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
		# scale_factor_mtx = np.ones(beta_mtx.shape,dtype=np.float32)
		scale_factor_mtx = self.scale_factor_mtx
		beta_mtx_scaled = beta_mtx*scale_factor_mtx
		regularize_1 = np.sum(np.sum(np.absolute(beta_mtx_scaled.T.dot(H_mtx))))
		print('regularize_1',regularize_1, H_mtx.shape)

		## regularization based on tf groups
		regularize_2 = self.test_motif_peak_estimate_obj_regularize_2(beta_mtx,motif_group_list,motif_group_vec)
		print('regularize_2',regularize_2)

		## regularization based on gene expression correlation of tfs, graph-guided fused Lasso
		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
		if type_id_regularize==1:
			regularize_3 = np.sum(np.sum(np.absolute(beta_mtx_scaled.dot(H_p))))
			print('regularize_3',regularize_3, H_p.shape)
		else:
			regularize_3 = 0.0

		self.lambda_vec = [1E-03,[1E-03,1E-03,1E-03]]
		Lasso_eta_1, lambda_vec_2 = self.lambda_vec
		regularizer_vec_2 = np.asarray([regularize_1,regularize_2,regularize_3])

		# type_id_regularize = 1
		# if type_id_regularize==1:
		#   regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2.dot(lambda_vec_2)
		# else:
		#   regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2[0:2].dot(lambda_vec_2[0:2])
		regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2.dot(lambda_vec_2)
		
		squared_error_ = squared_error + regularize_pre
		print('squared_error_, squared_error, regularize_pre ', squared_error_, squared_error, regularize_pre)

		return squared_error_

	## motif-peak estimate: objective function
	# tf_score: gene_num by tf_num
	def test_motif_peak_estimate_obj_constraint1(self,beta,score_mtx,y,beta_mtx=[],ratio=1,motif_group_list=[],motif_group_vec=[],lambda_vec=[],iter_cnt1=0,type_id_regularize=0):
		
		# beta_mtx_id = np.asarray(beta_mtx_init>0)
		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
		beta_mtx[beta_mtx_id1] = np.ravel(beta)
		beta_mtx[beta_mtx_id2] = 0.0
		tf_score_beta = score_mtx*beta_mtx  # shape: cell_num by gene_num by motif_num

		# print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
		y_pred = np.sum(tf_score_beta,axis=2)   # shape: cell_num by gene_num
		squared_error = ((y_pred-y)**2).sum().sum()*ratio

		## regularization using Lasso
		Lasso_1 = np.sum(np.absolute(beta))
		
		# H_mtx = self.H_mtx.loc[gene_query_vec,:]
		# H_p = self.H_p.loc[feature_query_vec,:]
		H_mtx = self.H_mtx_1
		H_p = self.H_p_1
		# print('H_mtx, H_p ',H_mtx.shape, H_p.shape)

		## regularization based on gene expression correlation, graph-guided fused Lasso
		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
		# scale_factor_mtx = np.ones(beta_mtx.shape,dtype=np.float32)
		# scale_factor_mtx = self.scale_factor_mtx
		# beta_mtx_scaled = beta_mtx*scale_factor_mtx
		# the first dimension corresponds to the intercept
		regularize_1 = np.sum(np.sum(np.absolute(beta_mtx[:,1:].T.dot(H_mtx))))
		# print('regularize_1',regularize_1, H_mtx.shape)

		## regularization based on tf groups
		# regularize_2 = self.test_motif_peak_estimate_obj_regularize_2(beta_mtx,motif_group_list,motif_group_vec)
		vec1 = [np.linalg.norm(beta_mtx[:,feature_query_id],axis=1) for feature_query_id in motif_group_list]
		group_regularize_vec = np.asarray(vec1).T.dot(motif_group_vec)
		regularize_2 = np.sum(group_regularize_vec)
		# print('regularize_2',regularize_2)

		## regularization based on gene expression correlation of tfs, graph-guided fused Lasso
		if type_id_regularize==1:
			regularize_3 = np.sum(np.sum(np.absolute(beta_mtx[:,1:].dot(H_p))))
			# print('regularize_3',regularize_3, H_p.shape)
		else:
			regularize_3 = 0.0

		# self.lambda_vec = [1E-03,[1E-03,1E-03,1E-03]]
		# Lasso_eta_1, lambda_vec_2 = self.lambda_vec
		Lasso_eta_1, lambda_vec_2 = lambda_vec
		regularize_vec_2 = np.asarray([regularize_1,regularize_2,regularize_3])
		regularize_vec_pre1 = [Lasso_1]+list(regularize_vec_2)

		regularize_pre = Lasso_1*Lasso_eta_1 + regularize_vec_2.dot(lambda_vec_2)
		
		if self.iter_cnt1==0:
			scale_factor = 1.1*regularize_pre/(squared_error+1E-12)
			print('scale_factor est ', scale_factor)
			self.scale_factor = scale_factor

		scale_factor = self.scale_factor
		squared_error_ = scale_factor*squared_error + regularize_pre

		if squared_error < self.config['obj_value_pre1']:
			print('obj_value_pre1 ', self.config['obj_value_pre1'], squared_error, self.iter_cnt1)
			self.config.update({'obj_value_pre1':squared_error})
			self.config.update({'regularize_vec_pre1':regularize_vec_pre1,'regularize_pre1':regularize_pre})

		if squared_error_ < self.config['obj_value_pre2']:
			print('obj_value_pre2 ', self.config['obj_value_pre2'], squared_error_, self.iter_cnt1)
			self.config.update({'obj_value_pre2':squared_error_})
			self.config.update({'regularize_vec_pre2':regularize_vec_pre1,'regularize_pre2':regularize_pre})

		self.iter_cnt1 += 1

		# if iter_cnt1%100==0:
		#   print('squared_error_, squared_error, regularize_pre ', squared_error_, squared_error, regularize_pre)
		#   print('regularize_1, regularize_2, regularize_3 ', regularize_1, regularize_2, regularize_3)

		return squared_error_

	## motif-peak estimate: objective function
	# tf_score: gene_num by tf_num
	def test_motif_peak_estimate_obj_constraint2(self,beta,score_mtx,y,regularize_vec=[], beta_mtx=[],ratio=1,motif_group_list=[],motif_group_vec=[],lambda_vec=[],iter_cnt1=0,type_id_regularize=0):
		
		# beta_mtx_id = np.asarray(beta_mtx_init>0)
		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
		beta_mtx[beta_mtx_id1] = np.ravel(beta)
		beta_mtx[beta_mtx_id2] = 0.0
		tf_score_beta = score_mtx*beta_mtx  # shape: cell_num by gene_num by motif_num

		# print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
		y_pred = np.sum(tf_score_beta,axis=2)   # shape: cell_num by gene_num
		squared_error = ((y_pred-y)**2).sum().sum()*ratio

		## regularization using Lasso
		Lasso_1 = np.sum(np.absolute(beta))
		x = score_mtx
		x_ori = x.copy()
		eps = 1e-12
		regularize_vec_1 = 1.0/(regluarize_vec+eps)
		x = x*np.asarray(regularize_vec)

		# H_mtx = self.H_mtx.loc[gene_query_vec,:]
		# H_p = self.H_p.loc[feature_query_vec,:]
		H_mtx = self.H_mtx_1
		H_p = self.H_p_1
		# print('H_mtx, H_p ',H_mtx.shape, H_p.shape)

		## regularization based on gene expression correlation, graph-guided fused Lasso
		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
		# scale_factor_mtx = np.ones(beta_mtx.shape,dtype=np.float32)
		# scale_factor_mtx = self.scale_factor_mtx
		# beta_mtx_scaled = beta_mtx*scale_factor_mtx
		# the first dimension corresponds to the intercept
		regularize_1 = np.sum(np.sum(np.absolute(beta_mtx[:,1:].T.dot(H_mtx))))
		# print('regularize_1',regularize_1, H_mtx.shape)

		## regularization based on tf groups
		# regularize_2 = self.test_motif_peak_estimate_obj_regularize_2(beta_mtx,motif_group_list,motif_group_vec)
		vec1 = [np.linalg.norm(beta_mtx[:,feature_query_id],axis=1) for feature_query_id in motif_group_list]
		group_regularize_vec = np.asarray(vec1).T.dot(motif_group_vec)
		regularize_2 = np.sum(group_regularize_vec)
		# print('regularize_2',regularize_2)

		## regularization based on gene expression correlation of tfs, graph-guided fused Lasso
		if type_id_regularize==1:
			regularize_3 = np.sum(np.sum(np.absolute(beta_mtx[:,1:].dot(H_p))))
			# print('regularize_3',regularize_3, H_p.shape)
		else:
			regularize_3 = 0.0

		# self.lambda_vec = [1E-03,[1E-03,1E-03,1E-03]]
		# Lasso_eta_1, lambda_vec_2 = self.lambda_vec
		Lasso_eta_1, lambda_vec_2 = lambda_vec
		regularize_vec_2 = np.asarray([regularize_1,regularize_2,regularize_3])
		regularize_vec_pre1 = [Lasso_1]+list(regularize_vec_2)

		regularize_pre = Lasso_1*Lasso_eta_1 + regularize_vec_2.dot(lambda_vec_2)
		
		if self.iter_cnt1==0:
			scale_factor = 1.1*regularize_pre/(squared_error+1E-12)
			print('scale_factor est ', scale_factor)
			self.scale_factor = scale_factor

		scale_factor = self.scale_factor
		squared_error_ = scale_factor*squared_error + regularize_pre

		if squared_error < self.config['obj_value_pre1']:
			print('obj_value_pre1 ', self.config['obj_value_pre1'], squared_error, self.iter_cnt1)
			self.config.update({'obj_value_pre1':squared_error})
			self.config.update({'regularize_vec_pre1':regularize_vec_pre1,'regularize_pre1':regularize_pre})

		if squared_error_ < self.config['obj_value_pre2']:
			print('obj_value_pre2 ', self.config['obj_value_pre2'], squared_error_, self.iter_cnt1)
			self.config.update({'obj_value_pre2':squared_error_})
			self.config.update({'regularize_vec_pre2':regularize_vec_pre1,'regularize_pre2':regularize_pre})

		self.iter_cnt1 += 1

		# if iter_cnt1%100==0:
		#   print('squared_error_, squared_error, regularize_pre ', squared_error_, squared_error, regularize_pre)
		#   print('regularize_1, regularize_2, regularize_3 ', regularize_1, regularize_2, regularize_3)

		return squared_error_

	## motif-peak estimate: objective function
	# tf_score: gene_num by tf_num
	def test_motif_peak_estimate_rescale_1(self,beta,score_mtx,y,beta_mtx=[],regularize_vec=[],ratio=1,motif_group_list=[],motif_group_vec=[],lambda_vec=[],iter_cnt1=0,type_id_regularize=0):
		
		# beta_mtx_id = np.asarray(beta_mtx_init>0)
		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
		beta_mtx[beta_mtx_id1] = np.ravel(beta)
		beta_mtx[beta_mtx_id2] = 0.0
		tf_score_beta = score_mtx*beta_mtx  # shape: cell_num by gene_num by motif_num

		# print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
		y_pred = np.sum(tf_score_beta,axis=2)   # shape: cell_num by gene_num
		squared_error = ((y_pred-y)**2).sum().sum()*ratio

	# motif-peak estimate: optimize unit
	def test_motif_peak_estimate_optimize1_unit_pre(self,initial_guess,x_train,y_train,beta_mtx=[],pre_data_dict={},type_id_regularize=0):

		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG','Powell','CG','TNC','COBYLA','trust-constr',
						'dogleg','trust-ncg','trust-exact','trust-krylov']
		method_id = 2
		method_type_id = method_vec[method_id]
		method_type_id = 'SLSQP'
		# method_type_id = 'L-BFGS-B'

		id1, cnt = 0, 0
		flag1 = False
		small_eps = 1e-12
		# con1 = ({'type':'ineq','fun':lambda x:x-small_eps},
		#       {'type':'ineq','fun':lambda x:-x+1})

		# lower_bound, upper_bound = -100, 100
		lower_bound, upper_bound = pre_data_dict['param_lower_bound'], pre_data_dict['param_upper_bound']
		regularize_lambda_vec = pre_data_dict['regularize_lambda_vec']
		ratio1 = pre_data_dict['ratio']
		motif_group_list, motif_group_vec = pre_data_dict['motif_group_list'], pre_data_dict['motif_group_vec']
		beta_mtx = self.beta_mtx
		print('beta_mtx ', beta_mtx.shape)

		con1 = ({'type':'ineq','fun':lambda x:x-lower_bound},
				{'type':'ineq','fun':lambda x:-x+upper_bound})
		bounds = scipy.optimize.Bounds(lower_bound,upper_bound)

		x_train = np.asarray(x_train)
		y_train = np.asarray(y_train)

		# flag1 = True
		iter_cnt1 = 0
		tol = 0.0001
		if 'tol_pre1' in pre_data_dict:
			tol = pre_data_dict['tol_pre1']

		while (flag1==False):
			try:
				# res = minimize(self._motif_peak_prob_lik_constraint,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
				#       method=method_vec[method_id],constraints=con1,tol=1e-6,options={'disp':False})
				start=time.time()
				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
				#               constraints=con1,tol=tol,options={'disp':False})
				res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
								method=method_type_id,constraints=con1,tol=tol,options={'disp':False})

				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
				#               method=method_type_id,bounds=bounds,tol=tol,options={'disp':False})
				
				flag1 = True
				iter_cnt1 += 1
				stop=time.time()
				print('iter_cnt1 ',iter_cnt1,stop-start)
			except Exception as err:
				flag1 = False
				print("OS error: {0} motif_peak_estimate_optimize1_unit1 {1}".format(err,flag1))
				cnt = cnt + 1
				if cnt > 10:
					print('cannot find the solution! %d'%(cnt))
					break

		if flag1==True:
			param1 = res.x
			flag = self._check_params(param1,upper_bound=upper_bound,lower_bound=lower_bound)
			# t_value1 = param1[1:]
			small_eps = 1e-12
			# t_value1[t_value1<=0]=small_eps
			# t_value1[t_value1>1.0]=1.0
			quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
			t_vec1 = utility_1.test_stat_1(param1,quantile_vec=quantile_vec_1)
			print('parameter est ',flag, param1.shape, param1[0:10], t_vec1)
			return flag, param1

		else:
			print('did not find the solution!')
			return -2, initial_guess

	# motif-peak estimate: optimize unit
	def test_motif_peak_estimate_optimize1_unit2(self,initial_guess,x_train,y_train,beta_mtx=[],pre_data_dict={},type_id_regularize=0):

		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG','Powell','CG','TNC','COBYLA','trust-constr',
						'dogleg','trust-ncg','trust-exact','trust-krylov']
		method_id = 2
		method_type_id = method_vec[method_id]
		method_type_id = 'SLSQP'
		# method_type_id = 'L-BFGS-B'

		id1, cnt = 0, 0
		flag1 = False
		small_eps = 1e-12
		# con1 = ({'type':'ineq','fun':lambda x:x-small_eps},
		#       {'type':'ineq','fun':lambda x:-x+1})

		# lower_bound, upper_bound = -100, 100
		lower_bound, upper_bound = pre_data_dict['param_lower_bound'], pre_data_dict['param_upper_bound']
		regularize_lambda_vec = pre_data_dict['regularize_lambda_vec']
		ratio1 = pre_data_dict['ratio']
		motif_group_list, motif_group_vec = pre_data_dict['motif_group_list'], pre_data_dict['motif_group_vec']
		beta_mtx = self.beta_mtx
		print('beta_mtx ', beta_mtx.shape)

		con1 = ({'type':'ineq','fun':lambda x:x-lower_bound},
				{'type':'ineq','fun':lambda x:-x+upper_bound})
		bounds = scipy.optimize.Bounds(lower_bound,upper_bound)

		x_train = np.asarray(x_train)
		y_train = np.asarray(y_train)

		# flag1 = True
		iter_cnt1 = 0
		tol = 0.0001
		if 'tol_pre1' in pre_data_dict:
			tol = pre_data_dict['tol_pre1']

		while (flag1==False):
			try:
				# res = minimize(self._motif_peak_prob_lik_constraint,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
				#       method=method_vec[method_id],constraints=con1,tol=1e-6,options={'disp':False})
				start=time.time()
				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
				#               constraints=con1,tol=tol,options={'disp':False})
				res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
								method=method_type_id,constraints=con1,tol=tol,options={'disp':False})

				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
				#               method=method_type_id,bounds=bounds,tol=tol,options={'disp':False})
				
				flag1 = True
				iter_cnt1 += 1
				stop=time.time()
				print('iter_cnt1 ',iter_cnt1,stop-start)
			except Exception as err:
				flag1 = False
				print("OS error: {0} motif_peak_estimate_optimize1_unit1 {1}".format(err,flag1))
				cnt = cnt + 1
				if cnt > 10:
					print('cannot find the solution! %d'%(cnt))
					break

		if flag1==True:
			param1 = res.x
			flag = self._check_params(param1,upper_bound=upper_bound,lower_bound=lower_bound)
			# t_value1 = param1[1:]
			small_eps = 1e-12
			# t_value1[t_value1<=0]=small_eps
			# t_value1[t_value1>1.0]=1.0
			quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
			t_vec1 = utility_1.test_stat_1(param1,quantile_vec=quantile_vec_1)
			print('parameter est ',flag, param1.shape, param1[0:10], t_vec1)
			return flag, param1

		else:
			print('did not find the solution!')
			return -2, initial_guess

	# motif-peak estimate: optimize unit
	def test_motif_peak_estimate_optimize1_unit1(self,initial_guess,x_train,y_train,beta_mtx=[],pre_data_dict={},type_id_regularize=0):

		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG','Powell','CG','TNC','COBYLA','trust-constr',
						'dogleg','trust-ncg','trust-exact','trust-krylov']
		method_id = 2
		method_type_id = method_vec[method_id]
		method_type_id = 'SLSQP'
		# method_type_id = 'L-BFGS-B'

		id1, cnt = 0, 0
		flag1 = False
		small_eps = 1e-12
		# con1 = ({'type':'ineq','fun':lambda x:x-small_eps},
		#       {'type':'ineq','fun':lambda x:-x+1})

		# lower_bound, upper_bound = -100, 100
		lower_bound, upper_bound = pre_data_dict['param_lower_bound'], pre_data_dict['param_upper_bound']
		regularize_lambda_vec = pre_data_dict['regularize_lambda_vec']
		ratio1 = pre_data_dict['ratio']
		motif_group_list, motif_group_vec = pre_data_dict['motif_group_list'], pre_data_dict['motif_group_vec']
		beta_mtx = self.beta_mtx
		print('beta_mtx ', beta_mtx.shape)

		con1 = ({'type':'ineq','fun':lambda x:x-lower_bound},
				{'type':'ineq','fun':lambda x:-x+upper_bound})
		bounds = scipy.optimize.Bounds(lower_bound,upper_bound)

		x_train = np.asarray(x_train)
		y_train = np.asarray(y_train)

		# lambda1 for group Lasso regularization, lambda2 for Lasso regularization
		# lambda1, lambda2 = pre_data_dict['lambda1'], pre_data_dict['lambda2']
		# lambda1 = (1-alpha)*lambda_regularize
		# lambda2 = alpha*lambda_regularize

		# res = minimize(self._motif_peak_prob_lik_constraint,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
		#               method=method_vec[method_id],constraints=con1,tol=1e-6,options={'disp':True})

		# res = minimize(self._motif_peak_prob_lik_constraint_copy,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
		#               constraints=con1,tol=1e-5,options={'disp':False})

		# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,motif_group_list,motif_group_vec,lambda_vec,type_id_regularize),
		#               constraints=con1,tol=1e-5,options={'disp':False})


		# flag1 = True
		iter_cnt1 = 0
		tol = 0.0001
		if 'tol_pre1' in pre_data_dict:
			tol = pre_data_dict['tol_pre1']

		while (flag1==False):
			try:
				# res = minimize(self._motif_peak_prob_lik_constraint,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
				#       method=method_vec[method_id],constraints=con1,tol=1e-6,options={'disp':False})
				start=time.time()
				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
				#               constraints=con1,tol=tol,options={'disp':False})
				res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
								method=method_type_id,constraints=con1,tol=tol,options={'disp':False})

				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
				#               method=method_type_id,bounds=bounds,tol=tol,options={'disp':False})
				
				flag1 = True
				iter_cnt1 += 1
				stop=time.time()
				print('iter_cnt1 ',iter_cnt1,stop-start)
			except Exception as err:
				flag1 = False
				print("OS error: {0} motif_peak_estimate_optimize1_unit1 {1}".format(err,flag1))
				cnt = cnt + 1
				if cnt > 10:
					print('cannot find the solution! %d'%(cnt))
					break

		if flag1==True:
			param1 = res.x
			flag = self._check_params(param1,upper_bound=upper_bound,lower_bound=lower_bound)
			# t_value1 = param1[1:]
			small_eps = 1e-12
			# t_value1[t_value1<=0]=small_eps
			# t_value1[t_value1>1.0]=1.0
			quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
			t_vec1 = utility_1.test_stat_1(param1,quantile_vec=quantile_vec_1)
			print('parameter est ',flag, param1.shape, param1[0:10], t_vec1)
			return flag, param1

		else:
			print('did not find the solution!')
			return -2, initial_guess

	# motif-peak estimate: beta parameter initialize
	def test_motif_peak_estimate_param_init_1(self,x_train,y_train,x_valid=[],y_valid=[],sample_weight=[],gene_query_vec=[],sample_id=[],feature_query_vec=[],peak_read=[],meta_exprs=[],motif_prior_type=0,output_filename='',save_mode=1,score_type=1,select_config={}):

		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG']
		method_id = 2
		id1, cnt = 0, 0
		flag1 = False
		small_eps = 1e-12
		sample_num, gene_num, feature_query_num = x_train.shape

		dict_gene_param_id = self.dict_gene_param_id_
		param_num = self.param_num
		beta_param_init = np.zeros(param_num,dtype=np.float32)

		if len(x_valid)==0:
			validation_size = 0.1
			id1 = np.arange(sample_num)
			id_train, id_valid, id_train1, id_valid1 = train_test_split(id1,id1,test_size=validation_size,random_state=1)
			x_train_pre, x_valid, y_train_pre, y_valid = x_train[id_train], x_train[id_valid], y_train[id_train], y_train[id_valid]
			if len(sample_weight)>0:
				sample_weight_train = sample_weight[id_train]
			else:
				sample_weight_train = []
			print('train, test ', x_train.shape, y_train.shape, x_train_pre.shape, y_train_pre.shape, x_valid.shape, y_valid.shape)

		# model_type_vec = ['LR','Lasso','XGBClassifier','XGBR','RF']
		# model_type_id = 'Lasso'
		# model_type_id1 = 1
		# intercept_flag = select_config['intercept_flag']
		# print('intercept_flag ', intercept_flag)
		# if model_type_id in [3,'Lasso']:
		#   Lasso_alpha = 1E-03
		#   Lasso_max_iteration = 5000
		#   tol = 0.0005
		#   select_config.update({'Lasso_alpha':Lasso_alpha,'Lasso_max_iteration':Lasso_max_iteration,'Lasso_tol':tol})
		
		df_gene_motif_prior = self.df_gene_motif_prior_
		df_gene_motif_prior = df_gene_motif_prior.loc[:,feature_query_vec]

		score_list_1 = []
		# if intercept_flag==True, the first dimension in the feature matrix corresponds to the intercept term
		intercept_flag = select_config['intercept_flag']
		model_type_id, model_type_id1 = select_config['model_type_id_init'],select_config['model_type_id1_init']
		id1 = int(intercept_flag)
		for i1 in range(gene_num):
			x_train1, x_valid1 = x_train_pre[:,i1,id1:], x_valid[:,i1,id1:]
			y_train1, y_valid1 = y_train_pre[:,i1], y_valid[:,i1]
			
			model_train = self.training_1(x_train1,y_train1,x_valid,y_valid,sample_weight=sample_weight_train,type_id=model_type_id,type_id1=model_type_id1,model_path1="",select_config=select_config)
			coef_, intercept_ = model_train.coef_, model_train.intercept_

			y_pred_valid1 = model_train.predict(x_valid1)
			score_list = self.test_score_pred_1(y_valid1, y_pred_valid1)
			score_vec = score_list[0]
			score_list_1.append(score_vec)

			t_gene_query = gene_query_vec[i1]
			gene_query_id = t_gene_query
			param_id = dict_gene_param_id[gene_query_id]
			t_vec1 = utility_1.test_stat_1(coef_)
			print(gene_query_id,i1,intercept_,t_vec1)
			print(score_vec)

			t_motif_prior = df_gene_motif_prior.loc[gene_query_id,:]
			# b1 = np.where(t_motif_prior>0)[0]
			motif_query_vec1 = feature_query_vec[t_motif_prior>0]

			beta_param_init_est = pd.Series(index=feature_query_vec,data=np.asarray(coef_))
			t_param_init_est = beta_param_init_est.loc[motif_query_vec1]
			if intercept_flag==True:
				t_param_init_est_1 = np.asarray([intercept_]+list(t_param_init_est))
			else:
				t_param_init_est_1 = np.asarray(t_param_init_est)

			beta_param_init[param_id] = t_param_init_est_1

		field_1 = ['mse','pearsonr','pval1','explained_variance','mean_absolute_error','median_absolute_error','r2_score','spearmanr','pval2']
		score_pred = np.asarray(score_list_1)
		df1 = pd.DataFrame(index=gene_query_vec,columns=field_1,data=score_pred,dtype=np.float32)

		if save_mode==1:
			if output_filename=='':
				output_filename = 'test_motif_peak_estimate_param_init_score_1.txt'
			df1.to_csv(output_filename,sep='\t',float_format='%.6E')

		return beta_param_init, df1

	# motif-peak estimate: model train evaluation
	def test_score_pred_1(self,y,y_pred,x=[],sample_weight=[],model_train=[],gene_query_vec=[],sample_id=[],feature_query_vec=[],save_mode=1,select_config={}):

		y, y_pred = np.asarray(y), np.asarray(y_pred)
		# print('y, y_pred ',y.shape, y_pred.shape)
		score_list = []
		if y.ndim==1:
			y, y_pred = y[:,np.newaxis], y_pred[:,np.newaxis]
		
		query_num = y.shape[1]
		# print('y, y_pred ',y.shape, y_pred.shape)
		for i1 in range(query_num):
			# mse, pearsonr, pval1, explained_variance, median_abs_error, mean_abs_error, r2_score, spearmanr, pval2
			vec1 = self.score_2a(y[:,i1],y_pred[:,i1])
			print('score pred ',vec1,i1)
			score_list.append(list(vec1))
		
		return score_list

	## motif-peak estimate: model train prediction
	def test_motif_peak_estimate_pred_1(self,x,select_config={}):

		beta = self.param_est
		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
		beta_mtx[beta_mtx_id1] = np.ravel(beta)
		beta_mtx[beta_mtx_id2] = 0.0
		tf_score_beta = x*beta_mtx  # shape: cell_num by gene_num by motif_num

		# print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
		y_pred = np.sum(tf_score_beta,axis=2)   # shape: cell_num by gene_num
		
		return y_pred

	## motif-peak estimate: model train explain
	def test_motif_peak_estimate_explain_1(self,gene_query_vec=[],sample_id=[],feature_query_vec=[],peak_read=[],meta_exprs=[],input_filename1='',input_filename2='',select_config={}):
		# pre_data_dict_1 = {'tf_score_mtx':tf_score_mtx,'gene_expr':gene_expr,
		#                   'sample_id_train':sample_id_train,'sample_id_test':sample_id_test,
		#                   'sample_id':sample_id,'gene_query_vec':gene_query_vec,'feature_query_vec':feature_query_vec,
		#                   'dict_gene_param_id':dict_gene_param_id,
		#                   'df_gene_motif_prior':df_gene_motif_prior_}

		score_type_vec1 = ['unnormalized','normalized']
		score_type_vec2 = ['tf_score','tf_score_product','tf_score_product_2']
		# score_type_id = select_config['tf_score_type']
		# score_type_id2 = select_config['tf_score_type_id2']
		score_type_id = 1
		score_type_id2 = 0
		score_type_1, score_type_2 = score_type_vec1[score_type_id], score_type_vec2[score_type_id2]
		print('tf score type 1, tf score type 2 ', score_type_1, score_type_2)
		
		file_path2 = 'vbak1'
		# score_type_1, score_type_2 = 1, 2
		filename_annot1_pre1 = '%s_%s'%(score_type_2,score_type_1)
		input_filename1 = '%s/test_pre_data_dict_1_%s.1.npy'%(file_path2,filename_annot1_pre1)
		data1 = np.load(input_filename1,allow_pickle=True)
		data1 = data1[()]
		print(list(data1.keys()))
		
		pre_data_dict_1 = data1
		gene_query_vec, sample_id, feature_query_vec = pre_data_dict_1['gene_query_vec'], pre_data_dict_1['sample_id'], pre_data_dict_1['feature_query_vec']
		dict_gene_param_id = pre_data_dict_1['dict_gene_param_id']
		df_gene_motif_prior_ = pre_data_dict_1['df_gene_motif_prior']

		# field_query = ['tf_score_mtx','gene_expr','sample_id_train','sample_id_test',
		#               'gene_query_vec','sample_id','feature_query_vec',
		#               'dict_gene_param_id','df_gene_motif_prior']

		# list1 = [pre_data_dict_1[t_field_query] for t_field_query in field_query]

		tf_score_mtx, gene_expr = pre_data_dict_1['tf_score_mtx'], pre_data_dict_1['gene_expr']
		sample_id_train, sample_id_test = pre_data_dict_1['sample_id_train'], pre_data_dict_1['sample_id_test']
		
		print('tf_score_mtx, gene_expr ', tf_score_mtx.shape, gene_expr.shape)
		print('sample_id_train, sample_id_test ', sample_id_train.shape, sample_id_test.shape)

		# pre_data_dict_2.update({'beta_param_est':beta_param_est,'scale_factor':self.scale_factor,'iter_cnt1':self.iter_cnt1,
		#                               'config_pre':config_pre,'y_test':y_test,'y_pred_test':y_pred_test})

		ratio_1, ratio_2 = 0.9, 0.1
		Lasso_eta_1 = 0.001
		lambda_1 = 0.01

		if 'regularize_lambda_vec' in select_config:
			regularize_lambda_vec = select_config['regularize_lambda_vec']
			Lasso_eta_1, lambda_vec_2 = regularize_lambda_vec
			lambda_1,lambda_2,lambda_3 = lambda_vec_2

		filename_annot1_1 = '%s_%s_%s_%s'%(str(ratio_1),str(ratio_2),str(Lasso_eta_1),str(lambda_1))
		filename_annot1 = '%s_%s'%(filename_annot1_1,filename_annot1_pre1)

		input_filename2 = '%s/test_pre_data_dict_2_%s.1.npy'%(file_path2,filename_annot1)
		
		data2 = np.load(input_filename2,allow_pickle=True)
		data2 = data2[()]
		print(list(data2.keys()))
		pre_data_dict_2 = data2
		beta_param_est = pre_data_dict_2['beta_param_est']
		param_num = beta_param_est.shape[0]

		print('gene_query_vec, sample_id, feature_query_vec', len(gene_query_vec), len(sample_id), len(feature_query_vec))
		print('beta_param_est ', beta_param_est.shape)

		save_mode = 0
		intercept_flag = True
		dict_gene_param_id_1, param_num_1, beta_mtx_init = self.test_motif_peak_estimate_param_id(gene_query_vec=gene_query_vec,
																								feature_query_vec=feature_query_vec,
																								df_gene_motif_prior=df_gene_motif_prior_,
																								intercept=intercept_flag,
																								save_mode=save_mode)

		assert param_num==param_num_1

		beta_mtx_id1 = np.asarray(beta_mtx_init>0)
		beta_mtx = np.zeros(beta_mtx_init.shape,dtype=np.float32)
		beta_mtx[beta_mtx_id1] = np.ravel(beta_param_est)

		output_filename = 'test_beta_mtx_est.1.txt'
		if intercept_flag==True:
			t_columns = ['1']+list(feature_query_vec)
		else:
			t_columns = list(feature_query_vec)

		df_beta_1 = pd.DataFrame(index=gene_query_vec,columns=t_columns,data=beta_mtx,dtype=np.float32)
		df_beta_1.to_csv(output_filename,sep='\t',float_format='%.6E')
		print('df_beta_1, param_num_1 ', df_beta_1.shape, param_num_1)

		sample_id_train_pre = [pd.Index(sample_id).get_loc(sample_id1) for sample_id1 in sample_id_train]
		sample_id_test_pre = [pd.Index(sample_id).get_loc(sample_id1) for sample_id1 in sample_id_test]

		print('sample_id_train, sample_id_test ', len(sample_id_train), len(sample_id_test), sample_id_train[0:5], sample_id_test[0:5])
		print('sample_id_train_pre, sample_id_test_pre ', len(sample_id_train_pre), len(sample_id_test_pre), sample_id_train_pre[0:5], sample_id_test_pre[0:5])

		x_train = tf_score_mtx[sample_id_train_pre]
		y_train = gene_expr.loc[sample_id_train]

		x_test = tf_score_mtx[sample_id_test_pre]
		y_test = gene_expr.loc[sample_id_test]

		print('x_train, y_train, x_test, y_test ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
		x_train, y_train = np.asarray(x_train), np.asarray(y_train)
		x_test, y_test = np.asarray(x_test), np.asarray(y_test)

		model_type_name = 'LR'
		dict1 = dict()
		gene_query_num = len(gene_query_vec)
		param_est_1 = []
		param_est_2 = feature_query_vec

		param_est_1, param_est_2 = pd.Index(param_est_1), pd.Index(param_est_2)

		print('load TF motif group')
		method_type_id1 = 2
		df_motif_group, df_motif_group_query = self.test_motif_peak_estimate_motif_group_pre1(motif_query_vec=feature_query_vec,method_type_id1=method_type_id1)
		self.df_motif_group = df_motif_group
		self.df_motif_group_query = df_motif_group_query
		print('df_motif_group, df_motif_group_query ', df_motif_group.shape, df_motif_group_query.shape)

		df_motif_group = self.df_motif_group
		motif_cluster_dict = self.motif_cluster_dict

		for i1 in range(gene_query_num):
			gene_query_id = gene_query_vec[i1]
			t_gene_query = gene_query_id
			model_1 = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
			# model.fit(x_train, y_train)

			model_1.intercept_ = df_beta_1.loc[gene_query_id,'1']
			t_param_est = df_beta_1.loc[gene_query_id,feature_query_vec]
			model_1.coef_ = t_param_est
			print('model_1 ', gene_query_id, i1)
			print(model_1.intercept_, model_1.coef_)

			# shap_value_1, base_value_1, expected_value_1 = self.test_model_explain_pre1(model_1,x=x_train,y=y_train,
			#                                                                               feature_name=feature_query_vec,
			#                                                                               model_type_name=model_type_name,
			#                                                                               x_test=x_test,y_test=y_test,
			#                                                                               linear_type_id=0,select_config=select_config)

			# feature_importances_1 = np.mean(np.abs(shap_value_1),axis=0)  # mean_abs_value
			# feature_importances_pre1 = pd.DataFrame(index=feature_query_vec,columns=['mean_abs_shap_value'],data=feature_importances_1,dtype=np.float32)
			# feature_importances_pre1_sort = feature_importances_pre1.sort_values(by=['mean_abs_shap_value'],ascending=False)

			feature_importances_pre1_sort = []
 
			thresh1 = 0
			b1 = np.where(np.abs(t_param_est)>thresh1)[0]
			feature_query_1 = feature_query_vec[b1]
			t_param_est_1 = t_param_est[feature_query_1]
			sort_id_1 = t_param_est_1.abs().sort_values(ascending=False).index
			t_param_est_sort1 = t_param_est_1[sort_id_1]
			df1 = pd.DataFrame(index=sort_id_1,columns=['coef_'],data=np.asarray(t_param_est_sort1))
			df1['motif_group'] = df_motif_group.loc[sort_id_1,'group']
			motif_group_query = df_motif_group.loc[sort_id_1,'group']
			motif_group_query_vec = np.unique(motif_group_query)
			
			list2 = []
			list3 = []
			t_motif_prior = df_gene_motif_prior.loc[gene_query_id,:]
			# b1 = np.where(t_motif_prior>0)[0]
			t_feature_query = feature_query_vec[t_motif_prior>0]
			motif_query_ori = df_motif_group.index

			for t_group_id in motif_group_query_vec:
				motif_group_query_1 = motif_query_ori[df_motif_group['group']==t_group_id]
				motif_query_num1 = len(motif_group_query_1)
				list2.append(motif_query_num1)

				motif_group_query_2 = motif_group_query_1.intersection(t_feature_query)
				motif_query_num2 = len(motif_group_query_2)
				list3.append(motif_query_num2)

			df2 = pd.DataFrame(index=motif_group_query_vec,columns=['motif_num1'],data=np.asarray(list2))
			print('motif_group_query_vec ', len(motif_group_query_vec))
			df2['motif_num2'] = np.asarray(list3)

			motif_query_num1 = df2.loc[np.asarray(motif_group_query),'motif_num1']
			motif_query_num2 = df2.loc[np.asarray(motif_group_query),'motif_num2']
			df1['motif_neighbor_ori'] = np.asarray(motif_query_num1)
			df1['motif_neighbor'] = np.asarray(motif_query_num2)

			dict1[gene_query_id] = [feature_importances_pre1_sort, df1]
			print('feature importance est ', feature_importances_pre1_sort[0:5], t_param_est_sort1[0:5], len(feature_query_1), gene_query_id, i1)

			param_est_1 = param_est_1.union(feature_query_1,sort=False)
			param_est_2 = param_est_2.intersection(feature_query_1,sort=False)

		output_filename = 'test_beta_param_est_pre1.%s.npy'%(filename_annot1_1)
		np.save(output_filename,dict1,allow_pickle=True)

		df1 = df_beta_1.loc[:,param_est_1]
		df2 = df_beta_1.loc[:,param_est_2]
		output_filename1 = 'test_beta_param_est_sort_1.%s.txt'%(filename_annot1_1)
		mean_abs_value1 = df1.abs().mean(axis=0)
		sort_id1 = mean_abs_value1.sort_values(ascending=False).index
		df1_sort = df1.loc[:,sort_id1]
		df1_sort.to_csv(output_filename1,sep='\t',float_format='%.6E')

		output_filename2 = 'test_beta_param_est_sort_2.%s.txt'%(filename_annot1_1)
		mean_abs_value2 = df2.abs().mean(axis=0)
		sort_id2 = mean_abs_value2.sort_values(ascending=False).index
		df2_sort = df2.loc[:,sort_id2]
		df2_sort.to_csv(output_filename2,sep='\t',float_format='%.6E')

		feature_query_vec_1 = ['Pdx1','Foxa1','Foxa3']
		for t_feature_query in feature_query_vec_1:
			print(t_feature_query)
			print(df1_sort[t_feature_query])

		gene_query_vec_1 = gene_query_vec
		gene_query_num1 = len(gene_query_vec_1)
		for i1 in range(gene_query_num1):
			gene_query_id = gene_query_vec_1[i1]
			feature_importances_pre1_sort, df1 = dict1[gene_query_id]
			print(gene_query_id,i1)
			print(df1[0:50])

		print('df1_sort, df2_sort ', df1_sort.shape, df2_sort.shape)

		return

	## motif-peak estimate: load gene query vector
	## motif-peak estimate: optimization
	def test_motif_peak_estimate_optimize_1(self,gene_query_vec=[],sample_id=[],feature_query_vec=[],dict_score_query={},gene_tf_prior_1=[],gene_tf_prior_2=[],peak_read=[],meta_exprs=[],motif_prior_type=0,save_mode=1,load_score_mode=0,score_type=1,type_score_pair=0,select_config={}):
		
		## load data
		# self.gene_name_query_expr_
		# self.gene_highly_variable
		# self.motif_data; self.motif_data_cluster; self.motif_group_dict
		# self.pre_data_dict_1
		# self.motif_query_name_expr, self.motif_data_local_
		# self.test_motif_peak_estimate_pre_load(select_config=select_config)

		# ## load the gene query vector
		# # celltype_query_vec = ['Trachea.Lung']
		# # gene_query_vec = self.test_motif_peak_gene_query_load(celltype_query_vec=celltype_query_vec)

		# if len(gene_query_vec)==0:
		#   # gene_idvec = self.gene_idvec   # the set of genes
		#   # gene_query_vec = gene_idvec
		#   gene_query_vec = self.gene_highly_variable # the set of genes

		# gene_query_num = len(gene_query_vec)
		# print('celltype query, gene_query_vec ', celltype_query_vec, gene_query_num)
		# self.gene_query_vec = gene_query_vec

		# motif_query_name_expr = self.motif_query_name_expr
		# self.motif_data_expr = self.motif_data.loc[:,motif_query_name_expr]
		# print('motif_query_name_expr ', len(motif_query_name_expr), self.motif_data_expr.shape)
		# if len(feature_query_vec)==0:
		#   feature_query_vec = motif_query_name_expr

		# # motif_query_vec = motif_query_name_expr
		# motif_query_vec = feature_query_vec

		# # sample_id = self.meta_scaled_exprs_2.index   # sample id
		# sample_num = len(sample_id)
		# self.sample_id = sample_id

		# ## motif-peak estimate: objective function
		# # initialize variables that are not changing with parameters
		# pre_load_1, pre_load_2 = 1, 1
		# # self.test_motif_peak_estimate_optimize_init_pre1(gene_query_vec=gene_query_vec,
		# #                                                 sample_id=sample_id,
		# #                                                 feature_query_vec=motif_query_vec,
		# #                                                 select_config=select_config,
		# #                                                 pre_load_1=pre_load_1,
		# #                                                 pre_load_2=pre_load_2)

		# ## commented
		# ## prepare the correlated peaks of each gene
		# # dict_peak_local_ =  self.test_motif_peak_estimate_peak_gene_association1(gene_query_vec=gene_query_vec,select_config=select_config)
		# # key_vec = list(dict_peak_local_.keys())
		# # print('dict_peak_local ', len(key_vec), key_vec[0:5])
		# # self.dict_peak_local_ = dict_peak_local_

		# ## prepare the motif prior of each gene
		self.gene_motif_prior_1 = gene_tf_prior_1   # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
		self.gene_motif_prior_2 = gene_tf_prior_2   # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

		print('gene_motif_prior_1, gene_motif_prior_2 ', self.gene_motif_prior_1.shape, self.gene_motif_prior_2.shape)
		# return

		# ## initialize the output variable graph based on expression correlation
		# # the edge set for the vertex-edge incidence matrix (VE) of output variables based on expression correlation
		# thresh_corr_repsonse, thresh_pval_response = 0.2, 0.05
		# if 'thresh_corr_repsonse' in select_config:
		#   thresh_corr_repsonse = select_config['thresh_corr_repsonse']
		# if 'thresh_pval_response' in select_config:
		#   thresh_pval_repsonse = select_config['thresh_pval_repsonse']
		
		# query_type_id = 0
		# response_edge_set, response_graph_connect, response_graph_list = self.test_motif_peak_estimate_graph_pre1(feature_query_vec=gene_query_vec,
		#                                                                                       query_type_id=query_type_id,
		#                                                                                       select_config=select_config,
		#                                                                                       thresh_corr=thresh_corr_repsonse,
		#                                                                                       thresh_pval=thresh_pval_response,
		#                                                                                       load_mode=0)

		# gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1 = response_graph_list
		# self.gene_expr_corr_, self.gene_expr_pval_, self.gene_expr_pval_corrected = gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1
		# print('response edge set ', len(gene_query_vec), len(response_edge_set))

		# # return

		# ## initialize the input variable graph based on expression correlation
		# # the edge set for the VE matrix of input variables based on expression correlation
		# thresh_corr_predictor, thresh_pval_predictor = 0.3, 0.05
		# if 'thresh_corr_predictor' in select_config:
		#   thresh_corr_predictor = select_config['thresh_corr_predictor']
		# if 'thresh_pval_predictor' in select_config:
		#   thresh_pval_predictor = select_config['thresh_pval_predictor']

		# query_type_id = 1
		# # motif_query_name = self.motif_query_name_expr
		# # print('motif_query_name ', len(motif_query_name))
		# predictor_edge_set, predictor_graph_connect, predictor_graph_list = self.test_motif_peak_estimate_graph_pre1(feature_query_vec=motif_query_vec,
		#                                                                                                               query_type_id=query_type_id,
		#                                                                                                               select_config=select_config,
		#                                                                                                               thresh_corr=thresh_corr_predictor,
		#                                                                                                               thresh_pval=thresh_pval_predictor,
		#                                                                                                               load_mode=0)

		# gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2 = predictor_graph_list
		# self.tf_expr_corr_, self.tf_expr_pval_, self.tf_expr_pval_corrected = gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2
		# print('predictor edge set ', len(motif_query_name), len(predictor_edge_set))

		# # return

		# ## initialize the VE matrix of output variables
		# H = self.test_motif_peak_estimate_VE_init_(query_id=gene_query_vec,
		#                                           edge_set=response_edge_set,
		#                                           df_graph=gene_expr_corr_1)
		# self.H_mtx = H
		# print('VE matrix of response variable graph ', H.shape)

		# ## initialize the vertex-edge incidence matrix of input variables
		# H_p = self.test_motif_peak_estimate_VE_init_(query_id=motif_query_name,
		#                                           edge_set=predictor_edge_set,
		#                                           df_graph=gene_expr_corr_2)
		# self.H_p = H_p
		# print('VE matrix of predictor variable graph ', H_p.shape)

		flag_pre_load=load_score_mode
		field_1 = ['unweighted','weighted','product.unweighted','product.weighted','gene_query_id','sample_id','feature_query']
		
		# score_type, type_score_pair = 1, 0
		# if 'tf_score_type' in select_config:
		#   score_type = select_config['tf_score_type']
		# if 'type_score_pair' in select_config:
		#   type_score_pair = select_config['type_score_pair']

		# motif_query_vec = feature_query_vec
		if flag_pre_load==0:
			# dict1, dict2 = self.test_motif_peak_estimate_feature_mtx_scale(gene_query_vec=gene_query_vec,
			#                                                               motif_query_vec=motif_query_vec,
			#                                                               select_config=select_config)

			# score_mtx = numpy.swapaxes(score_mtx, axis1=0, axis2=1) # size: sample_num by gene num by feature query num
			# score_dict = {'gene_query':gene_query_vec,'sample_id':sample_id,'feature_query':feature_query_vec,'score_mtx':score_mtx}

			score_dict = self.test_motif_peak_estimate_gene_tf_estimate_pre1(gene_query_vec=gene_query_vec,
																				feature_query_vec=feature_query_vec,
																				dict_score_query=dict_score_query,
																				peak_read=peak_read,meta_exprs=meta_exprs,
																				score_type=score_type,
																				type_score_pair=type_score_pair,
																				save_mode=save_mode,
																				select_config=select_config)

			# return

		else:
			log_type_id_atac, scale_type_id = 1, 2
			# output_filename1 = '%s/test_peak_motif_estimate_tf_score.log_%d.scale_%d.pre1.npy'%(self.save_path_1,log_type_id_atac,scale_type_id)
			# output_filename2 = '%s/test_peak_motif_estimate_tf_score.log_%d.scale_%d.npy'%(self.save_path_1,log_type_id_atac,scale_type_id)
			# # np.save(output_filename1,dict1,allow_pickle=True)
			# # np.save(output_filename2,dict2,allow_pickle=True)
			# dict1 = np.load(output_filename1,allow_pickle=True)
			# dict1 = dict1[()]
			# dict2 = np.load(output_filename2,allow_pickle=True)
			# dict2 = dict2[()]
			input_filename = ''
			dict1 = np.load(input_filename,allow_pickle=True)
			score_dict = dict1[()]
		
		# scale_type_id = 0
		# # scale_type_id = 1
		# t_field1 ='product.weighted' 
		# mtx2 = dict2[t_field1]
		# mtx2 = mtx2[:,:,0:-1]
		# mtx1 = dict1[t_field1]

		# print('mtx2',t_field1,mtx2.shape)
		# print('mtx1',t_field1,mtx1.shape)

		# list_1 = [mtx1,mtx2]

		# gene_query_vec_1 = dict2['gene_query_id']
		# gene_query_vec = gene_query_vec_1
		# sample_id = dict2['sample_id']
		# motif_query_vec = dict2['feature_query']
		gene_query_vec, sample_id, feature_query_vec_pre1 = score_dict['gene_query'], score_dict['sample_id'], score_dict['feature_query']
		# motif_query_vec = feature_query_vec_pre1
		tf_score_mtx = score_dict['score_mtx']
		motif_query_vec = feature_query_vec_pre1
		gene_query_num, sample_num, feature_query_num_pre1 = len(gene_query_vec), len(sample_id), len(motif_query_vec)
		print('gene_query_vec_1, sample_id, feature_query_vec_pre1 ', gene_query_num, sample_num, feature_query_num_pre1)

		# self.meta_scaled_exprs = self.meta_scaled_exprs_2
		# x_expr = self.meta_scaled_exprs.loc[sample_id,motif_query_vec]
		# x_expr = meta_exprs.loc[sample_id,feature_query_vec]
		load_mode_idvec = 0
		train_idvec_1 = []

		# scale_type_id2 = 1    # log transformation and scaling of peak access-tf expr product
		# scale_type_id2 = 0    # log transformation without scaling of peak access-tf expr product
		feature_type_id = 1
		if feature_type_id<=1:
			# tf_score_mtx = list_1[0]
			x = 0
		else:
			# tf_score_mtx = list_1[1]
			x = 1

		fields = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2']
		# field_1 = ['mse','pearsonr','pval1','explained_variance','mean_absolute_error','median_absolute_error','r2_score','spearmanr','pval2']
		# score_list1 = []

		## commented
		# model_type_vec = ['LR','XGBR','Lasso']
		# # model_type_idvec = [1,0]
		# # model_type_idvec = [0]
		# model_type_idvec = [1]
		# # model_path_1 = self.save_path_1
		# model_path_1 = '%s/model_train'%(self.save_path_1)
		# run_id = 1
		# run_num, num_fold = 1, 10
		# max_interaction_depth, sel_num1, sel_num2 = 10, 2, -1
		# select_config.update({'run_id':run_id,'run_num':run_num,'max_interaction_depth':max_interaction_depth,
		#                       'sel_num1_interaction':sel_num1,'sel_num2_interaction':sel_num2})
		
		# df_score1 = self.training_regression_3(gene_query_vec=gene_query_vec,
		#                                       feature_query_vec=motif_query_vec,
		#                                       sample_id=sample_id,
		#                                       tf_score_mtx=tf_score_mtx,
		#                                       model_type_idvec=model_type_idvec,
		#                                       model_type_vec=model_type_vec,
		#                                       model_path_1=model_path_1,
		#                                       feature_type_id=feature_type_id,
		#                                       num_fold=num_fold,
		#                                       train_idvec=train_idvec_1,
		#                                       load_mode_idvec=load_mode_idvec,
		#                                       select_config=select_config)

		# # output_filename1 = '%s/test_motif_peak_estimate.train_%s.%d.%d.%d.1.txt'%(self.save_path_1,model_type_id1,num_fold,gene_query_num,feature_type_id)
		# # df_score1.to_csv(output_filename1,sep='\t',float_format='%.6E')

		# return

		dict_gene_motif_local_ = dict()
		self.dict_gene_param_id_ = dict()

		# param_id = self.dict_gene_param_id_[gene_query_id]
		# beta_local_ = beta_param_[param_id]
		# beta_param_ = beta
		print('peak_read ', self.peak_read.shape)

		# dict_peak_local_ = self.dict_peak_local_
		dict_peak_local_ = self.peak_dict
		print('dict_peak_local_ ', len(dict_peak_local_.keys()))

		## load gene-tf link prior (tf with motif)
		if motif_prior_type==0:
			# df_gene_motif_prior_ = self.gene_motif_prior_1.T.loc[:,motif_query_name_expr]
			# df_gene_motif_prior_ = self.gene_motif_prior_1.T.loc[:,motif_query_vec]
			gene_motif_prior_pre = self.gene_motif_prior_1
		else:
			# df_gene_motif_prior_ = self.gene_motif_prior_2.T.loc[:,motif_query_name_expr]
			gene_motif_prior_pre = self.gene_motif_prior_2
		
		df_gene_motif_prior_ = gene_motif_prior_pre.loc[:,motif_query_vec]
		self.df_gene_motif_prior_ = df_gene_motif_prior_
		print('df_gene_motif_prior_ ', df_gene_motif_prior_.shape, df_gene_motif_prior_.columns)

		# return

		# t_value1 = np.sum(df_gene_motif_prior_,axis=0)
		# b1 = np.where(t_value1>0)[0]
		# feature_query_name_pre2 = df_gene_motif_prior_.columns
		# motif_query_name = motif_query_name_expr[b1]
		# feature_query_vec_pre2 = feature_query_name_pre2[b1]
		feature_query_vec_pre2 = utility_1.test_columns_nonzero_1(df_gene_motif_prior_,type_id=1)
		print('feature_query_vec_pre2 ', len(feature_query_vec_pre2))

		feature_query_vec_ori = feature_query_vec.copy()
		feature_query_vec_pre1 = pd.Index(feature_query_vec_pre1)
		feature_query_vec = feature_query_vec_pre1.intersection(feature_query_vec_pre2,sort=False)
		feature_query_idvec = np.asarray([feature_query_vec_pre1.get_loc(query_id) for query_id in feature_query_vec])
		print(feature_query_idvec[0:10])
		tf_score_mtx = tf_score_mtx[:,:,feature_query_idvec]
		print('tf_score_mtx ', tf_score_mtx.shape)

		motif_query_name = feature_query_vec
		motif_query_num = len(motif_query_name)
		print('motif_query_name ', len(motif_query_name))

		motif_query_vec = motif_query_name
		self.motif_query_vec = motif_query_vec

		# list1, list2 = [], []
		# start_id1, start_id2 = 0, 0
		# for i1 in range(gene_query_num):
		#   t_gene_query = gene_query_vec[i1]
		#   t_motif_prior = df_gene_motif_prior_.loc[t_gene_query,:]
		#   b1 = np.where(t_motif_prior>0)[0]
		#   t_feature_query = motif_query_name[b1]
		#   str1 = ','.join(list(t_feature_query))

		#   t_feature_query_num = len(t_feature_query)
		#   list1.append(t_feature_query_num)
		#   list2.append(str1)

		#   start_id2 = start_id1+t_feature_query_num
		#   self.dict_gene_param_id_[t_gene_query] = np.arange(start_id1,start_id2)
		#   print(t_gene_query,i1,start_id1,start_id2,t_feature_query_num)
		#   start_id1 = start_id2

		# # self.param_num = start_id2
		# # print('param_num ', self.param_num)

		# df_gene_motif_prior_str = pd.DataFrame(index=gene_query_vec,columns=['feature_num','feature_query'])
		# df_gene_motif_prior_str['feature_num'] = list1
		# df_gene_motif_prior_str['feature_query'] = list2

		# annot_pre1 = str(len(gene_query_vec))
		# output_filename = '%s/df_gene_motif_prior_str_%s.txt'%(self.save_path_1,annot_pre1)
		# df_gene_motif_prior_str.to_csv(output_filename,sep='\t')

		# print('df_gene_motif_prior_str ', df_gene_motif_prior_str.shape)

		## the parameter id of beta for each gene
		intercept_flag = True
		select_config.update({'intercept_flag':intercept_flag})
		dict_gene_param_id, param_num, beta_mtx_init = self.test_motif_peak_estimate_param_id(gene_query_vec=gene_query_vec,
																								feature_query_vec=feature_query_vec,
																								df_gene_motif_prior=df_gene_motif_prior_,
																								intercept=intercept_flag)

		self.dict_gene_param_id_ = dict_gene_param_id
		self.param_num = param_num
		self.beta_mtx_init_ = beta_mtx_init
		beta_mtx_id1 = np.asarray(beta_mtx_init>0)
		self.beta_mtx_id1, self.beta_mtx_id2 = beta_mtx_id1, (~beta_mtx_id1)
		self.beta_mtx = np.zeros(beta_mtx_init.shape,dtype=np.float32)
		# return

		print('param_num ', self.param_num)

		motif_group_query = self.df_motif_group_query.loc[feature_query_vec,'group']
		motif_group_idvec = np.unique(motif_group_query)
		motif_group_list = [feature_query_vec[motif_group_query==group_id] for group_id in motif_group_idvec]
		motif_group_vec_pre = pd.Series(index=motif_group_idvec,data=[len(query_id) for query_id in motif_group_list],dtype=np.float32)
		motif_group_vec = np.asarray(motif_group_vec_pre)

		feature_query_vec_1 = self.beta_mtx_init_.columns
		df1 = pd.Series(index=feature_query_vec_1,data=np.arange(len(feature_query_vec_1)))
		motif_group_list_pre = [np.asarray(df1.loc[query_id]) for query_id in motif_group_list] # the index of the motif query id in the beta matrix

		self.motif_group_list = motif_group_list
		self.motif_group_vec = motif_group_vec_pre
		
		print('motif_group_list ', len(motif_group_idvec), motif_group_list[0:2])
		motif_group_vec_sort = motif_group_vec_pre.sort_values(ascending=False)
		print(motif_group_vec_sort)
		print(np.sum(motif_group_vec_sort>10),np.sum(motif_group_vec_sort>2),np.sum(motif_group_vec_sort==2),np.sum(motif_group_vec_sort==1))

		# peak_read = self.peak_read
		sample_id = meta_exprs.index
		peak_read = peak_read.loc[sample_id,:]
		# type_id_regularize = 0
		type_id_regularize = 1

		gene_query_num = len(gene_query_vec)
		sample_num = len(sample_id)
		feature_query_num = len(feature_query_vec)

		score_type_vec1 = ['unnormalized','normalized']
		score_type_vec2 = ['tf_score','tf_score_product','tf_score_product_2']
		score_type_id = select_config['tf_score_type']
		score_type_id2 = select_config['tf_score_type_id2']
		score_type_1, score_type_2 = score_type_vec1[score_type_id], score_type_vec2[score_type_id2]
		print('tf score type 1, tf score type 2 ', score_type_1, score_type_2)

		# self.meta_scaled_exprs = self.meta_scaled_exprs_2
		gene_expr = meta_exprs.loc[sample_id,gene_query_vec]
		# y = meta_exprs.loc[sample_id,gene_query_vec]
		# x = meta_exprs.loc[sample_id,feature_query_vec]
		y1 = np.asarray(gene_expr)
		x1_pre = np.asarray(tf_score_mtx)
		x0_pre = np.ones((sample_num,gene_query_num,1),dtype=np.float32)    # the first dimension
		x1 = np.concatenate((x0_pre,x1_pre),axis=2)

		# ratio1 = 1.0/sample_num
		ratio1 = 1.0/(sample_num*gene_query_num)
		# motif_prior_type = 0

		# x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.33, random_state=1)
		test_size = 0.125
		sample_id1 = np.arange(sample_num)
		sample_id_train, sample_id_test, sample_id1_train, sample_id1_test = train_test_split(sample_id,sample_id1,test_size=test_size,random_state=1)
		x_train, x_test, y_train, y_test = x1[sample_id1_train,:], x1[sample_id1_test,:], y1[sample_id1_train,:], y1[sample_id1_test,:]
		print('train, test ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

		pre_data_dict_1 = {'tf_score_mtx':tf_score_mtx,'gene_expr':gene_expr,
							'sample_id_train':sample_id_train,'sample_id_test':sample_id_test,
							'sample_id':sample_id,'gene_query_vec':gene_query_vec,'feature_query_vec':feature_query_vec,
							'dict_gene_param_id':dict_gene_param_id,
							'df_gene_motif_prior':df_gene_motif_prior_,
							'beta_mtx_id1':beta_mtx_id1}
		
		filename_annot1_pre1 = '%s_%s'%(score_type_2,score_type_1)

		output_filename1 = 'test_pre_data_dict_1_%s.1.npy'%(filename_annot1_pre1)
		if os.path.exists(output_filename1)==False:
			np.save(output_filename1,pre_data_dict_1,allow_pickle=True)

		field_1 = ['param_lower_bound','param_upper_bound','regularize_lambda_vec','tol_pre1']
		field_2 = ['motif_group_list','motif_group_vec','ratio']
		field_pre1 = field_1+field_2

		param_lower_bound, param_upper_bound = -100, 100
		Lasso_eta_1, lambda_vec_2 = 1E-03, [1E-03,1E-03,1E-03]
		regularize_lambda_vec = [Lasso_eta_1, lambda_vec_2]
		tol_pre1 = 0.0005
		list1_pre = [param_lower_bound,param_upper_bound,regularize_lambda_vec,tol_pre1]
		default_value_ = dict(zip(field_1,list1_pre))
		list1, list_1 = [], []
		for t_field_query in field_1:
			if t_field_query in select_config:
				t_value1 = select_config[t_field_query]
				print(t_field_query,t_value1)
				list1.append(t_value1)
			else:
				print('the query %s not included in config'%(t_field_query))
				print('the default value ',default_value_[t_field_query])
				list1.append(default_value_[t_field_query])

		# list1 = [param_lower_bound,param_upper_bound,regularize_lambda_vec,motif_group_list_pre,motif_group_vec]
		ratio1 = 1.0/(sample_num*gene_query_num)
		# ratio1 = 1.0/sample_num

		list_1 = list1 + [motif_group_list_pre,motif_group_vec,ratio1]
		pre_data_dict = dict(zip(field_pre1,list_1))
		# print('pre_data_dict ',pre_data_dict)

		param_init_est_mode = 1
		if param_init_est_mode==0:
			beta_param_init = np.random.rand(self.param_num)
		else:
			model_type_vec = ['LR','Lasso','XGBClassifier','XGBR','RF']
			model_type_id = 'Lasso'
			model_type_id1 = 1
			intercept_flag = select_config['intercept_flag']
			print('intercept_flag ', intercept_flag)
			select_config.update({'model_type_id_init':model_type_id,'model_type_id1_init':model_type_id1})
			if model_type_id in [3,'Lasso']:
				Lasso_alpha = 1E-03
				Lasso_max_iteration = 5000
				tol = 0.0005
				select_config.update({'Lasso_alpha':Lasso_alpha,'Lasso_max_iteration':Lasso_max_iteration,'Lasso_tol':tol})

			# output_filename = 'test_motif_peak_estimate_param_init_score_1.txt'%(filename_annot1_pre1)
			output_filename = 'test_motif_peak_estimate_param_init_score_%s.1.txt'%(filename_annot1_pre1)
			beta_param_init_pre, df_param_init_score = self.test_motif_peak_estimate_param_init_1(x_train,y_train,x_valid=[],y_valid=[],sample_weight=[],
																								gene_query_vec=gene_query_vec,sample_id=[],feature_query_vec=feature_query_vec,
																								peak_read=[],meta_exprs=[],motif_prior_type=0,
																								output_filename=output_filename,save_mode=1,select_config=select_config)

			np.random.seed(0)
			beta_param_init_1 = np.random.rand(self.param_num)-0.5
			# ratio_1, ratio_2 = 0.85, 0.15
			# ratio_1, ratio_2 = 1.0, 0
			ratio_1, ratio_2 = 0.9, 0.1
			# ratio_1, ratio_2 = 0.75, 0.25
			if 'beta_param_init_ratio1' in select_config:
				ratio_1 = select_config['beta_param_init_ratio1']
			if 'beta_param_init_ratio2' in select_config:
				ratio_2 = select_config['beta_param_init_ratio2']
			print('beta param init ratio_1, ratio_2 ', ratio_1, ratio_2)

			beta_param_init = ratio_1*beta_param_init_pre + ratio_2*beta_param_init_1

		beta_mtx = self.beta_mtx
		quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
		t_vec1 = utility_1.test_stat_1(beta_param_init,quantile_vec=quantile_vec_1)
		print('beta_param_init ', beta_param_init.shape,t_vec1)

		H_mtx = self.H_mtx.loc[gene_query_vec,:]
		H_p = self.H_p.loc[feature_query_vec,:]
		self.H_mtx_1 = H_mtx
		self.H_p_1 = H_p
		print('H_mtx, H_p ',H_mtx.shape, H_p.shape)

		obj_value_pre1, obj_value_pre2 = 1000, 1000
		self.iter_cnt1 = 0
		self.scale_factor = 1.0*gene_query_num
		self.config.update({'obj_value_pre1':obj_value_pre1,'obj_value_pre2':obj_value_pre2})

		regularize_lambda_vec = pre_data_dict['regularize_lambda_vec']
		Lasso_eta_1, lambda_vec_2 = regularize_lambda_vec
		lambda_1 = lambda_vec_2[0]

		filename_annot1_1 = '%s_%s_%s_%s'%(str(ratio_1),str(ratio_2),str(Lasso_eta_1),str(lambda_1))
		# filename_annot1_1 = '%.2f_%.2f_%s_%s'%(ratio_1,ratio_2,str(Lasso_eta_1),str(lambda_1))
		filename_annot1 = '%s_%s'%(filename_annot1_1,filename_annot1_pre1)

		beta_param_est = []
		pre_data_dict_2 = {'beta_param_init_pre':beta_param_init_pre,'beta_param_init':beta_param_init,'beta_param_est':beta_param_est,
							'pre_data_dict':pre_data_dict,'select_config':select_config}
		output_filename2 = 'test_pre_data_dict_2_%s.1.npy'%(filename_annot1)
		np.save(output_filename2,pre_data_dict_2,allow_pickle=True)

		flag_est, param_est = self.test_motif_peak_estimate_optimize1_unit1(initial_guess=beta_param_init,
														x_train=x_train,y_train=y_train,
														beta_mtx=beta_mtx,
														pre_data_dict=pre_data_dict,
														type_id_regularize=type_id_regularize)

		if flag_est==True:
			self.param_est = param_est
			y_pred_test = self.test_motif_peak_estimate_pred_1(x_test,select_config=select_config)
			print('y_test, y_pred_test ', y_test.shape, y_pred_test.shape)
			score_list = self.test_score_pred_1(y_test, y_pred_test)

			field_1 = ['mse','pearsonr','pval1','explained_variance','mean_absolute_error','median_absolute_error','r2_score','spearmanr','pval2']
			score_pred = np.asarray(score_list)
			df1 = pd.DataFrame(index=gene_query_vec,columns=field_1,data=score_pred,dtype=np.float32)
			print('df1 ',df1)

			save_mode = 1
			if save_mode==1:
				# if output_filename=='':
				#   output_filename = 'test_motif_peak_estimate_param_est_pred_score_1.txt'
				# df1.to_csv(output_filename,sep='\t',float_format='%.6E')
				# print('df1 ',df1)

				output_filename = 'test_motif_peak_estimate_param_est_pred_score_%s_1.txt'%(filename_annot1)
				df1.to_csv(output_filename,sep='\t',float_format='%.6E')

				beta_param_est = param_est
				# obj_value_pre1, obj_value_pre2 = self.config['obj_value_pre1'], self.config['obj_value_pre2']
				# regularize_vec_pre1, regularize_vec_pre2 = self.config['regularize_vec_pre1'], self.config['regularize_vec_pre2']
				# regularize_pre1, regularize_pre2 = self.config['regularize_pre1'], self.config['regularize_pre2']
				config_pre = self.config
				# pre_data_dict_2.update({'beta_param_est':beta_param_est,'scale_factor':self.scale_factor,'iter_cnt1':self.iter_cnt1,
				#                       'obj_value_pre1':obj_value_pre1,'obj_value_pre2':obj_value_pre2,
				#                       'regularize_vec_pre1':regularize_vec_pre1,'regularize_vec_pre2':regularize_vec_pre2,
				#                       'regularize_pre1':regularize_pre1,'regularize_pre2':regularize_pre2})
				pre_data_dict_2.update({'beta_param_est':beta_param_est,'scale_factor':self.scale_factor,'iter_cnt1':self.iter_cnt1,
										'config_pre':config_pre,'y_test':y_test,'y_pred_test':y_pred_test})

				output_filename2 = 'test_pre_data_dict_2_%s.1.npy'%(filename_annot1)
				np.save(output_filename2,pre_data_dict_2,allow_pickle=True)

		# for i2 in range(5):
		#   beta_param = np.random.rand(self.param_num)
		#   squared_error_ = self.test_motif_peak_estimate_obj_constraint1(gene_query_vec=gene_query_vec,
		#                                                                   sample_id=sample_id,
		#                                                                   feature_query_vec=motif_query_vec,
		#                                                                   beta=beta_param,
		#                                                                   tf_score_mtx=tf_score_mtx,
		#                                                                   peak_read=peak_read,
		#                                                                   meta_exprs=meta_exprs,
		#                                                                   motif_group_list=motif_group_list_pre,
		#                                                                   motif_group_vec=motif_group_vec,
		#                                                                   type_id_regularize=type_id_regularize,
		#                                                                   select_config=select_config)

		#   print(squared_error_, i2)

		return True

	## motif-peak estimate: load gene query vector
	## motif-peak estimate: optimization
	def test_motif_peak_estimate_optimize_2(self,gene_query_vec=[],sample_id=[],feature_query_vec=[],dict_score_query={},gene_tf_prior_1=[],gene_tf_prior_2=[],peak_read=[],meta_exprs=[],motif_prior_type=0,save_mode=1,load_score_mode=0,score_type=1,type_score_pair=0,select_config={}):
		
		## load data
		# self.gene_name_query_expr_
		# self.gene_highly_variable
		# self.motif_data; self.motif_data_cluster; self.motif_group_dict
		# self.pre_data_dict_1
		# self.motif_query_name_expr, self.motif_data_local_
		# self.test_motif_peak_estimate_pre_load(select_config=select_config)

		# ## prepare the motif prior of each gene
		self.gene_motif_prior_1 = gene_tf_prior_1   # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
		self.gene_motif_prior_2 = gene_tf_prior_2   # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

		print('gene_motif_prior_1, gene_motif_prior_2 ', self.gene_motif_prior_1.shape, self.gene_motif_prior_2.shape)
		# return


	def test_motif_peak_gene_query_load(self,celltype_query_vec=[],thresh_fc=1,thresh_fc_celltype_num=8,thresh_fc_celltype_num_2=-1):

		celltype_vec = ['Pharynx','Thymus','Thyroid','Thyroid/Trachea','Trachea/Lung','Esophagus',
						'Stomach','Liver','Pancreas 1','Pancreas 2','Small int','Colon']
		
		celltype_vec_str = ['Pharynx','Thymus','Thyroid','Thyroid.Trachea','Trachea.Lung','Esophagus',
						'Stomach','Liver','Pancreas.1','Pancreas.2','Small.int','Colon']

		# celltype_query = 'Trachea.Lung'
		celltype_num = len(celltype_vec_str)

		self.celltype_vec = celltype_vec
		self.celltype_vec_str = celltype_vec_str

		if len(celltype_query_vec)==0:
			gene_query_vec = self.gene_highly_variable # the set of genes

		else:

			list1 = []
			compare_type = 'pos'
			tol = 1
			filename_annot1 = '%s.tol%d'%(compare_type,tol)
			# thresh_fc = 2.0
			# thresh_fc_celltype_num = 8
			
			thresh_fc_celltype_num_1 = 8
			thresh_fc_celltype_num_pre1 = thresh_fc_celltype_num_2

			input_file_path = '%s/meta_exprs_compare_1'%(self.save_path_1)
			tol = 0

			for celltype_query in celltype_query_vec:

				# input_filename = '%s/DESeq2_data_1/test_motif_estimate.DESeq2.%s.merge.1A.subset2.fc%.1f.thresh%d.txt'%(self.path_1,celltype_query,thresh_fc,thresh_fc_celltype_num)  
				# input_filename = '%s/DESeq2_data_1/test_motif_estimate.DESeq2.%s.merge.1A.subset2.fc%.1f.thresh%d.%s.txt'%(self.path_1,celltype_query,thresh_fc,thresh_fc_celltype_num,filename_annot1)
				# input_filename = '%s/DESeq2_data_1/test_motif_estimate.DESeq2.%s.merge.1A.subset2.fc%.1f.thresh%d.%s.txt'%(self.path_1,celltype_query,thresh_fc,thresh_fc_celltype_num_1,filename_annot1)
				# input_filename = '%s/test_meta_exprs.MAST.%s.merge.1A.thresh%d.txt'%(input_file_path,celltype_query,thresh_fc_celltype_num_1)
				# input_filename = '%s/test_meta_exprs.MAST.%s.merge.1A.subset2.fc1.fdr0.05.thresh%d.txt'%(input_file_path,celltype_query,thresh_fc_celltype_num_1)
				input_filename = '%s/compare1/test_meta_exprs.MAST.%s.merge.1A.subset2.fc1.0.fdr0.05.thresh%d.pos.tol%d.txt'%(input_file_path,celltype_query,thresh_fc_celltype_num_1,tol)

				# compare_type = 'neg'
				# input_filename = '%s/DESeq2_data_1/test_motif_estimate.DESeq2.%s.merge.1A.subset2.fc%.1f.thresh%d.%s.tol%d.txt'%(self.path_1,celltype_query,thresh_fc,thresh_fc_celltype_num,compare_type,tol)

				data_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				gene_query_idvec_ori = data_query_1.index

				# label_1 = data_query_1['label_fc_1.0']
				# label_2 = data_query_1['label_fc_2.0']

				# if thresh_fc_celltype_num_pre1>-1:
				#   id1 = (label_1>=thresh_fc_celltype_num_pre1)|(label_2>=thresh_fc_celltype_num)
				# else:
				#   id1 = (label_2>=thresh_fc_celltype_num)

				thresh_fdr = 0.05
				# label_query = 'label_fc%s_fdr%s'%(str(thresh_fc),str(thresh_fdr))
				label_query = 'label_fc1.0_fdr0.05'
				label_1 = data_query_1[label_query]

				id1 = (label_1>=thresh_fc_celltype_num)

				gene_query_idvec = data_query_1.index[id1]
				# gene_query_num = len(gene_query_idvec)

				list1.extend(gene_query_idvec)
				print('cell type query ', celltype_query, len(gene_query_idvec))

			gene_query_vec = np.asarray(list1)

		return gene_query_vec

# class _Base2_train(BaseEstimator):
# 	"""
# 	Ordinary least squares Linear Regression.
# 	LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
# 	to minimize the residual sum of squares between the observed targets in
# 	the dataset, and the targets predicted by the linear approximation.
# 	Parameters
# 	----------
# 	fit_intercept : bool, default=True
# 	"""

# 	def __init__(
# 		self,
# 		*,
# 		fit_intercept=True,
# 		normalize="deprecated",
# 		copy_X=True,
# 		n_jobs=None,
# 		positive=False,
# 	):
# 		self.fit_intercept = fit_intercept
# 		self.normalize = normalize
# 		self.copy_X = copy_X
# 		self.n_jobs = n_jobs
# 		self.positive = positive

# 		self.gene_motif_prior_1 = []
# 		self.gene_motif_prior_2 = []


# 	## motif-peak estimate: optimize preparation
# 	def test_motif_peak_estimate_optimize_pre1(self,gene_query_vec,feature_query_vec,gene_tf_mtx_pre1=[],gene_tf_prior_1=[],gene_tf_prior_2=[],select_config={}):

# 		## commented
# 		# gene_tf_mtx_pre1, gene_tf_mtx_pre2 = self.test_motif_peak_estimate_motif_prior_2(gene_query_vec=gene_query_vec,motif_query_vec=[],type_id=0,select_config=select_config,pre_load_1=pre_load_1,pre_load_2=pre_load_2)

# 		# self.gene_motif_prior_1 = gene_tf_mtx_pre1    # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
# 		# self.gene_motif_prior_2 = gene_tf_mtx_pre2    # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

# 		self.gene_motif_prior_1 = gene_tf_prior_1   # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
# 		self.gene_motif_prior_2 = gene_tf_prior_2   # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

# 		print('gene_motif_prior_1, gene_motif_prior_2 ', self.gene_motif_prior_1.shape, self.gene_motif_prior_2.shape)
# 		# return

# 		## initialize the output variable graph based on expression correlation
# 		# the edge set for the vertex-edge incidence matrix (VE) of output variables based on expression correlation
# 		thresh_corr_repsonse, thresh_pval_response = 0.2, 0.05
# 		if 'thresh_corr_repsonse' in select_config:
# 			thresh_corr_repsonse = select_config['thresh_corr_repsonse']
# 		if 'thresh_pval_response' in select_config:
# 			thresh_pval_repsonse = select_config['thresh_pval_repsonse']
		
# 		query_type_id = 0
# 		response_edge_set, response_graph_connect, response_graph_list = self.test_motif_peak_estimate_graph_pre1(feature_query_vec=gene_query_vec,
# 																													query_type_id=query_type_id,
# 																													select_config=select_config,
# 																													thresh_corr=thresh_corr_repsonse,
# 																													thresh_pval=thresh_pval_response,
# 																													load_mode=0)

# 		gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1 = response_graph_list
# 		self.gene_expr_corr_, self.gene_expr_pval_, self.gene_expr_pval_corrected = gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1
# 		print('response edge set ', len(gene_query_vec), len(response_edge_set))

# 		# return

# 		## initialize the input variable graph based on expression correlation
# 		# the edge set for the VE matrix of input variables based on expression correlation
# 		thresh_corr_predictor, thresh_pval_predictor = 0.3, 0.01
# 		if 'thresh_corr_predictor' in select_config:
# 			thresh_corr_predictor = select_config['thresh_corr_predictor']
# 		if 'thresh_pval_predictor' in select_config:
# 			thresh_pval_predictor = select_config['thresh_pval_predictor']

# 		query_type_id = 1
# 		# motif_query_name = self.motif_query_name_expr
# 		# print('motif_query_name ', len(motif_query_name))
# 		predictor_edge_set, predictor_graph_connect, predictor_graph_list = self.test_motif_peak_estimate_graph_pre1(feature_query_vec=feature_query_vec,
# 																														query_type_id=query_type_id,
# 																														select_config=select_config,
# 																														thresh_corr=thresh_corr_predictor,
# 																														thresh_pval=thresh_pval_predictor,
# 																														load_mode=0)

# 		gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2 = predictor_graph_list
# 		self.tf_expr_corr_, self.tf_expr_pval_, self.tf_expr_pval_corrected = gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2
# 		motif_query_vec = feature_query_vec
# 		print('predictor edge set ', len(motif_query_vec), len(predictor_edge_set))

# 		# return

# 		## initialize the VE matrix of output variables
# 		H = self.test_motif_peak_estimate_VE_init_(query_id=gene_query_vec,
# 													edge_set=response_edge_set,
# 													df_graph=gene_expr_corr_1)
# 		self.H_mtx = H
# 		print('VE matrix of response variable graph ', H.shape)

# 		## initialize the vertex-edge incidence matrix of input variables
# 		H_p = self.test_motif_peak_estimate_VE_init_(query_id=motif_query_vec,
# 													edge_set=predictor_edge_set,
# 													df_graph=gene_expr_corr_2)
# 		self.H_p = H_p
# 		print('VE matrix of predictor variable graph ', H_p.shape)

# 		return (response_edge_set, response_graph_connect, predictor_edge_set, predictor_graph_connect, H, H_p)

# 	# ## motif-peak estimate: motif-gene prior
# 	def test_motif_peak_estimate_graph_pre1(self,feature_query_vec,query_type_id,select_config={},thresh_corr=-2,thresh_pval=-1,load_mode=0):

# 		dict_gene_motif_prior_ = dict()
# 		motif_query_name = self.motif_query_name_expr

# 		input_file_path2 = '%s/data1_annotation_repeat1/test_basic_est_imp1'%(self.path_1)
# 		filename2 = '%s/test_motif_peak_estimate_df_motif_pre.Lpar3.repeat1.correction.copy.2.log0_scale0.txt'%(input_file_path2)
# 		df_motif_pre = pd.read_csv(filename2,index_col=0,sep='\t')
# 		tf_query_ens, motif_label = df_motif_pre.index, df_motif_pre['label']
# 		print('motif_query_ens with expr ', len(tf_query_ens))
# 		self.tf_query_ens = tf_query_ens

# 		# motif_query_name_ens = self.tf_query_ens
# 		edge_set_query, graph_connect_query = [], []
# 		# self.test_motif_peak_estimate_motif_prior_1_pre(select_config=select_config)

# 		# return

# 		## select tf by gene-tf expression correlation
# 		if (len(self.gene_expr_corr_)==0) or (load_mode==0):
# 			## initialize the output or input variable graph based on expression correlation
# 			gene_expr_corr_, gene_expr_pval_, gene_expr_pval_corrected = self.test_motif_peak_estimate_graph_1(query_type_id=query_type_id,select_config=select_config)
# 			# self.gene_expr_corr_, self.gene_expr_pval_, self.gene_expr_pval_corrected = gene_expr_corr_, gene_expr_pval_, gene_expr_pval_corrected

# 		if thresh_corr>-2:
# 			# thresh_corr_repsonse, thresh_pval_response = 0.2, 0.05
# 			edge_set_query, graph_connect_query = self.test_motif_peak_estimate_graph_edgeset_1(query_idvec_1=feature_query_vec,
# 																								query_idvec_2 = [],
# 																								graph_similarity=gene_expr_corr_,
# 																								graph_pval=gene_expr_pval_corrected,
# 																								query_type_id=query_type_id,
# 																								thresh_similarity=thresh_corr,
# 																								thresh_pval=thresh_pval)
		
# 		# print('response edge set ', len(gene_query_vec), len(response_edge_set))
# 		print('edge set ', len(feature_query_vec), len(edge_set_query))
# 		graph_list1 = [gene_expr_corr_, gene_expr_pval_, gene_expr_pval_corrected]

# 		return edge_set_query, graph_connect_query, graph_list1

# 	## motif-peak estimate: edge set for the VE matrix
# 	# edge set from output variable graph and input variable graph based on expression correlation or DORC score correlation
# 	def test_motif_peak_estimate_graph_edgeset_1(self,query_idvec_1,query_idvec_2=[],graph_similarity=[],graph_pval=[],query_type_id=0,thresh_similarity=0.2,thresh_pval=0.05):

# 		# expr_corr = self.gene_expr_corr
# 		# expr_corr_pval_= self.gene_expr_corr_pval

# 		# t_corr_mtx = expr_corr.loc[gene_query_vec,gene_query_vec]
# 		# t_pval_mtx = expr_corr_pval_.loc[gene_query_vec,gene_query_vec]

# 		symmetry_type = 0
# 		if len(query_idvec_2)==0:
# 			query_idvec_2 = query_idvec_1
# 			symmetry_type = 1

# 		print('thresh_similarity, thresh_pval ', thresh_similarity, thresh_pval)
# 		graph_simi_local = graph_similarity.loc[query_idvec_1,query_idvec_2]

# 		## graph_simi_local is symmetric matrix
# 		if symmetry_type==1:
# 			# graph_simi_local_upper = pd.DataFrame(index=graph_simi_local.index,columns=graph_simi_local.columns,data=np.triu(graph_simi_local))
# 			keep1 = np.triu(np.ones(graph_simi_local.shape)).astype(bool)
# 			graph_simi_local = graph_simi_local.where(keep1)

# 		flag_simi = (graph_simi_local.abs()>thresh_similarity)
# 		flag_1 = flag_simi

# 		graph_simi_local_pre1 = graph_simi_local[flag_simi]
# 		edge_set_simi_1 = graph_simi_local_pre1.stack().reset_index()
# 		edge_set_simi_1.columns = ['node1','node2','corr_value']

# 		# df_graph_simi_local = graph_simi_local.stack().reset_index()
# 		# df_graph_simi_local.columns = ['node1','node2','corr_value']

# 		if len(graph_pval)>0:
# 			graph_pval_local = graph_pval.loc[query_idvec_1,query_idvec_2]

# 			if symmetry_type==1:
# 				# keep1 = np.triu(np.ones(graph_pval_local.shape)).astype(bool)
# 				graph_pval_local = graph_pval_local.where(keep1)

# 			flag_pval = (graph_pval_local<thresh_pval)
# 			flag_1 = (flag_simi&flag_pval)

# 			graph_pval_local_pre2 = graph_pval_local[flag_simi]
# 			edge_set_pval_2 = graph_pval_local_pre2.stack().reset_index()
# 			edge_set_simi_1['pval'] = edge_set_pval_2[edge_set_pval_2.columns[-1]]

# 			graph_pval_local_pre1 = graph_pval_local[flag_pval]
# 			edge_set_pval_1 = graph_pval_local_pre1.stack().reset_index()

# 			graph_simi_local_pre2 = graph_simi_local[flag_pval]
# 			edge_set_simi_2 = graph_simi_local_pre2.stack().reset_index()

# 			edge_set_pval_1.columns = ['node1','node2','pval']
# 			edge_set_pval_1['corr_value'] = edge_set_simi_2[edge_set_simi_2.columns[-1]]

# 			id1 = (edge_set_pval_1['node1']!=edge_set_pval_1['node2'])
# 			edge_set_pval_1 = edge_set_pval_1[id1]
# 			edge_set_pval_1.reset_index(drop=True,inplace=True)
			
# 			output_filename = '%s/test_motif_peak_estimate_edge_set_pval_%d.txt'%(self.save_path_1,query_type_id)
# 			edge_set_pval_1.to_csv(output_filename,sep='\t',float_format='%.6E')
# 			print('edge_set_pval_1 ', edge_set_simi_1.shape)

# 		id1 = (edge_set_simi_1['node1']!=edge_set_simi_1['node2'])
# 		edge_set_simi_1 = edge_set_simi_1[id1]
# 		edge_set_simi_1.reset_index(drop=True,inplace=True)
			
# 		output_filename = '%s/test_motif_peak_estimate_edge_set_corr_%d.txt'%(self.save_path_1,query_type_id)
# 		edge_set_simi_1.to_csv(output_filename,sep='\t',float_format='%.6E')
# 		print('edge_set_simi_1 ', edge_set_simi_1.shape)

# 		num1 = flag_simi.sum().sum()
# 		num2 = 0
# 		if len(graph_pval)>0:
# 			num2 = flag_pval.sum().sum()

# 		num_1 = flag_1.sum().sum()
# 		if symmetry_type==0:
# 			num_2 = flag_simi.shape[0]*flag_simi.shape[1]
# 		else:
# 			n1 = flag_simi.shape[0]
# 			num_2 = n1*(n1-1)/2

# 		ratio1, ratio2, ratio3 = num1/(num_2*1.0), num2/(num_2*1.0), num_1/(num_2*1.0)
# 		print('graph_simi_local, graph_pval_local ',graph_simi_local.shape,num1,num2,num_1,ratio1,ratio2,ratio3)

# 		graph_simi_local_pre = graph_simi_local[flag_1]
# 		graph_pval_local_pre = graph_pval_local[flag_1]

# 		## commented
# 		# output_filename1 = 'test_motif_peak_estimate_graph_connect_%d.1.txt'%(query_type_id)
# 		# output_filename2 = 'test_motif_peak_estimate_graph_connect_%d.2.txt'%(query_type_id)
# 		# flag_simi.to_csv(output_filename1,sep='\t',float_format='%.6E')
# 		# flag_pval.to_csv(output_filename2,sep='\t',float_format='%.6E')

# 		## commented
# 		# b1 = np.where(flag_1>0)
# 		# id1, id2 = b1[0], b1[1]
# 		# query_id1 = query_idvec_1[id1]
# 		# query_id2 = query_idvec_2[id2]
# 		# edge_set = np.column_stack((query_id1,query_id2))
# 		# graph_connect = flag_1

# 		## id1 = graph_simi_local.index
# 		# t_data1 = pd.DataFrame(index=query_id1,columns=['node1','node2'])
# 		# t_data1['node1'] = query_id1
# 		# t_data1['node2'] = query_id2

# 		# if query_type_id==0:
# 		#   t_query1, t_query2 = 'Lpar3', 'Pdx1'
# 		#   print(graph_similarity.loc[t_query1,t_query2])
# 		#   print(graph_pval.loc[t_query1,t_query2])

# 		edge_set = graph_simi_local_pre.stack().reset_index()
# 		edge_set_pval_ = graph_pval_local_pre.stack().reset_index()

# 		edge_set.columns = ['node1','node2','corr_value']
# 		edge_set['pval'] = edge_set_pval_[edge_set_pval_.columns[-1]]
# 		id1 = (edge_set['node1']!=edge_set['node2'])
# 		edge_set = edge_set[id1]
# 		edge_set.reset_index(drop=True,inplace=True)

# 		graph_connect = flag_1
# 		output_filename = '%s/test_motif_peak_estimate_edge_set_%d.txt'%(self.save_path_1,query_type_id)
# 		edge_set.to_csv(output_filename,sep='\t',float_format='%.6E')
# 		print('edge_set ', edge_set.shape)

# 		return edge_set, graph_connect

# 	## motif-peak estimate: objective function
# 	# initialize variables that are not changing with parameters
# 	def test_motif_peak_estimate_optimize_init_pre1(self,gene_query_vec=[],sample_id=[],feature_query_vec=[],select_config={},pre_load_1=0,pre_load_2=0):
		
# 		## load data
# 		# self.gene_name_query_expr_
# 		# self.gene_highly_variable
# 		# self.motif_data; self.motif_data_cluster; self.motif_group_dict
# 		# self.pre_data_dict_1
# 		# self.motif_query_name_expr, self.motif_data_local_
# 		# self.test_motif_peak_estimate_pre_load(select_config=select_config)

# 		if len(gene_query_vec)==0:
# 			# gene_idvec = self.gene_idvec   # the set of genes
# 			# gene_query_vec = gene_idvec
# 			gene_query_vec = self.gene_highly_variable # the set of genes

# 		gene_query_num = len(gene_query_vec)
# 		self.gene_query_vec = gene_query_vec

# 		if len(motif_query_vec)==0:
# 			motif_query_vec= self.motif_query_name_expr

# 		# sample_id = self.meta_scaled_exprs_2.index   # sample id
# 		sample_num = len(sample_id)
# 		self.sample_id = sample_id

# 		## commented
# 		## prepare the correlated peaks of each gene
# 		# dict_peak_local_ =  self.test_motif_peak_estimate_peak_gene_association1(gene_query_vec=gene_query_vec,select_config=select_config)
# 		# key_vec = list(dict_peak_local_.keys())
# 		# print('dict_peak_local ', len(key_vec), key_vec[0:5])

# 		# self.dict_peak_local_ = dict_peak_local_

# 		## prepare the motif prior of each gene
# 		# pre_load_1 = 1
# 		# pre_load_2 = 0

# 		## commented
# 		gene_tf_mtx_pre1, gene_tf_mtx_pre2 = self.test_motif_peak_estimate_motif_prior_2(gene_query_vec=gene_query_vec,motif_query_vec=[],type_id=0,select_config=select_config,pre_load_1=pre_load_1,pre_load_2=pre_load_2)

# 		self.gene_motif_prior_1 = gene_tf_mtx_pre1  # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
# 		self.gene_motif_prior_2 = gene_tf_mtx_pre2  # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

# 		print('gene_motif_prior_1, gene_motif_prior_2 ', self.gene_motif_prior_1.shape, self.gene_motif_prior_2.shape)
# 		# return

# 		## initialize the output variable graph based on expression correlation
# 		# the edge set for the vertex-edge incidence matrix (VE) of output variables based on expression correlation
# 		thresh_corr_repsonse, thresh_pval_response = 0.2, 0.05
# 		if 'thresh_corr_repsonse' in select_config:
# 			thresh_corr_repsonse = select_config['thresh_corr_repsonse']
# 		if 'thresh_pval_response' in select_config:
# 			thresh_pval_repsonse = select_config['thresh_pval_repsonse']
		
# 		query_type_id = 0
# 		response_edge_set, response_graph_connect, response_graph_list = self.test_motif_peak_estimate_motif_prior_1(feature_query_vec=gene_query_vec,
# 																								query_type_id=query_type_id,
# 																								select_config=select_config,
# 																								thresh_corr=thresh_corr_repsonse,
# 																								thresh_pval=thresh_pval_response,
# 																								load_mode=0)

# 		gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1 = response_graph_list
# 		self.gene_expr_corr_, self.gene_expr_pval_, self.gene_expr_pval_corrected = gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1
# 		print('response edge set ', len(gene_query_vec), len(response_edge_set))

# 		# return

# 		## initialize the input variable graph based on expression correlation
# 		# the edge set for the VE matrix of input variables based on expression correlation
# 		thresh_corr_predictor, thresh_pval_predictor = 0.3, 0.05
# 		query_type_id = 1
# 		motif_query_name = self.motif_query_name_expr
# 		print('motif_query_name ', len(motif_query_name))
# 		predictor_edge_set, predictor_graph_connect, predictor_graph_list = self.test_motif_peak_estimate_motif_prior_1(feature_query_vec=motif_query_name,
# 																								query_type_id=query_type_id,
# 																								select_config=select_config,
# 																								thresh_corr=thresh_corr_predictor,
# 																								thresh_pval=thresh_pval_predictor,
# 																								load_mode=0)

# 		gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2 = predictor_graph_list
# 		print('predictor edge set ', len(motif_query_name), len(predictor_edge_set))

# 		# return

# 		## initialize the VE matrix of output variables
# 		H = self.test_motif_peak_estimate_VE_init_(query_id=gene_query_vec,
# 													edge_set=response_edge_set,
# 													df_graph=gene_expr_corr_1)
# 		self.H_mtx = H
# 		print('VE matrix of response variable graph ', H.shape)

# 		## initialize the vertex-edge incidence matrix of input variables
# 		H_p = self.test_motif_peak_estimate_VE_init_(query_id=motif_query_name,
# 													edge_set=predictor_edge_set,
# 													df_graph=gene_expr_corr_2)
# 		self.H_p = H_p
# 		print('VE matrix of predictor variable graph ', H_p.shape)

# 		return True

# 	## motif-peak estimate: parameter id preparation, beta matrix initialization
# 	def test_motif_peak_estimate_param_id(self,gene_query_vec,feature_query_vec,df_gene_motif_prior,intercept=True,save_mode=1):
		
# 		list1, list2 = [], []
# 		start_id1, start_id2 = 0, 0
# 		gene_query_num = len(gene_query_vec)

# 		df_gene_motif_prior = df_gene_motif_prior.loc[:,feature_query_vec]
# 		# feature_query_name = df_gene_motif_prior.columns
# 		dict_gene_param_id = dict()

# 		beta_mtx = pd.DataFrame(index=gene_query_vec,columns=feature_query_vec,data=0.0,dtype=np.float32)   # shape: gene_num by feature_num

# 		if intercept==True:
# 			beta_mtx.insert(0,'1',1)    # the first dimension corresponds to the intercept

# 		for i1 in range(gene_query_num):
# 			gene_query_id = gene_query_vec[i1]
# 			t_gene_query = gene_query_id
# 			t_motif_prior = df_gene_motif_prior.loc[t_gene_query,:]
# 			# b1 = np.where(t_motif_prior>0)[0]
# 			t_feature_query = feature_query_vec[t_motif_prior>0]
# 			str1 = ','.join(list(t_feature_query))

# 			t_feature_query_num = len(t_feature_query)
# 			list1.append(t_feature_query_num)
# 			list2.append(str1)
			
# 			start_id2 = start_id1+t_feature_query_num+int(intercept)
# 			dict_gene_param_id[t_gene_query] = np.arange(start_id1,start_id2)
# 			beta_mtx.loc[t_gene_query,t_feature_query] = 1

# 			print(t_gene_query,i1,start_id1,start_id2,t_feature_query_num)
# 			start_id1 = start_id2

# 		param_num = start_id2

# 		if save_mode==1:
# 			df_gene_motif_prior_str = pd.DataFrame(index=gene_query_vec,columns=['feature_num','feature_query'])
# 			df_gene_motif_prior_str['feature_num'] = list1
# 			df_gene_motif_prior_str['feature_query'] = list2

# 			annot_pre1 = str(len(gene_query_vec))
# 			# output_filename = '%s/df_gene_motif_prior_str_%s.txt'%(self.save_path_1,annot_pre1)
# 			output_filename = 'df_gene_motif_prior_str_%s.txt'%(annot_pre1)
# 			if os.path.exists(output_filename)==False:
# 				df_gene_motif_prior_str.to_csv(output_filename,sep='\t')
# 			print('df_gene_motif_prior_str, param num ', df_gene_motif_prior_str.shape, param_num)

# 		return dict_gene_param_id, param_num, beta_mtx

# 	## motif-peak estimate: objective function, estimate tf score beta
# 	# tf_score: gene_num by tf_num by cell_num
# 	def test_motif_peak_estimate_param_score_1(self,gene_query_vec,sample_id,feature_query_vec,y,beta,beta_mtx,beta_mtx_id,score_mtx=[],peak_motif_mtx=[],motif_score_dict={},motif_prior_type=0):
		
# 		beta_mtx_id = np.asarray(beta_mtx_init>0)
# 		beta_mtx = np.asarray(beta_mtx)
# 		beta_mtx[beta_mtx_id] = beta
# 		beta_mtx[~beta_mtx_id] = 0.0
# 		score_beta = score_mtx*beta_mtx # shape: cell_num by gene_num by motif_num

# 		# tf_score_beta = mtx1*np.asarray(beta_mtx) # shape: cell_num by gene_num by motif_num

# 		# y_pred = pd.DataFrame(index=sample_id,columns=gene_query_vec,data=0.0)
# 		# for i2 in range(sample_num):
# 		#   sample_id1 = sample_id[i2]
# 		#   y_pred.loc[sample_id1] = tf_score_beta[i2].dot(x.loc[sample_id1,motif_query_name])

# 		# squared_error = ((y_pred-y)**2).sum().sum()

# 		# return tf_score_beta, squared_error
		
# 		return score_beta, beta_mtx

# 	## motif group estimate: objective function, regularization 1
# 	def test_motif_peak_estimate_obj_regularize_1(self,beta_mtx,H_mtx):
		
# 		# query_num_edge = len(edge_set)
# 		# motif_query_num = beta_mtx.shape[1]
# 		# H = pd.DataFrame(index=gene_id,columns=range(edge_num),data=0.0)      
# 		# for i1 in range(edge_num):
# 		#   gene_query_id1, gene_query_id2 = edge_set[i1]
# 		#   t_gene_expr_corr = gene_expr_corr.loc[gene_query_id1,gene_query_id2]
# 		#   f_value = np.absolute(t_gene_expr_corr)
# 		#   H.loc[gene_query_id1,i1] = f_value
# 		#   H.loc[gene_query_id2,i1] = -(f_value>0)*f_value

# 		regularize_1 = beta_mtx.T.dot(H_mtx)

# 		return regularize_1

# 	## motif group estimate: objective function, regularization 2
# 	def test_motif_peak_estimate_obj_regularize_2(self,beta_param,motif_group_list,motif_group_vec):
		
# 		# group_num = len(motif_group_list)
# 		# vec1 = [np.linalg.norm(beta1[motif_group_list[i]]) for i in range(group_num)]
# 		# query_num1, query_num2 = beta_param.shape[0], beta_param.shape[1]
# 		# query_vec_1, query_vec_2 = beta_param.index, beta_param.columns
# 		# beta_param_square = np.square(beta_param)

# 		# vec1 = [np.linalg.norm(beta_param[idvec]) for idvec in motif_group_list]
# 		vec1 = [np.linalg.norm(beta_param.loc[:,feature_query_id].values,axis=1) for feature_query_id in motif_group_list]
# 		group_regularize_vec = np.asarray(vec1).T.dot(motif_group_vec)
# 		group_regularize = np.sum(group_regularize_vec)

# 		return group_regularize

# 	def _check_params(self,params,upper_bound,lower_bound):

# 		small_eps = 1e-3
# 		min1, max1 = lower_bound, upper_bound
# 		# param1 = params[1:]
# 		flag_1 = (params>=min1-small_eps)&(params<=max1+small_eps)
# 		# print(flag_1)
# 		# print(param1)
# 		flag1 = (np.sum(flag_1)==len(params))
# 		if flag1==False:
# 			print(params)
# 			flag_1 = np.asarray(flag_1)
# 			id1 = np.where(flag_1==0)[0]
# 			print(params[id1], len(id1), len(params))

# 		return flag1

# 	## motif-peak estimate: objective function
# 	# tf_score: gene_num by tf_num
# 	def test_motif_peak_estimate_obj_1_ori(self,gene_query_vec,sample_id,feature_query_vec,beta=[],score_mtx=[],beta_mtx=[],peak_read=[],meta_exprs=[],motif_group_list=[],motif_group_vec=[],type_id_regularize=0):
		
# 		# gene_idvec = self.gene_idvec   # the set of genes
# 		# gene_query_vec = gene_idvec

# 		## load data
# 		gene_query_num = len(gene_query_vec)

# 		# sample_id = self.meta_scaled_exprs_2.index   # sample id
# 		# sample_id = self.sample_id
# 		sample_num = len(sample_id)

# 		# if len(feature_query_vec)==0:
# 		#   feature_query_vec = self.motif_query_name_expr
# 		feature_query_num = len(feature_query_vec)

# 		# self.meta_scaled_exprs = self.meta_scaled_exprs_2
# 		y = meta_exprs.loc[sample_id,gene_query_vec]
# 		x = meta_exprs.loc[sample_id,feature_query_vec]

# 		squared_error = 0
# 		# ratio1 = 1.0/sample_num
# 		ratio1 = 1.0/(sample_num*gene_query_num)

# 		motif_prior_type = 0
# 		# tf_score_beta = self.test_motif_peak_estimate_peak_motif_score_1(y,beta,tf_score_mtx=tf_score_mtx,motif_score_dict = {})
# 		# tf_score_beta, beta_mtx = self.test_motif_peak_estimate_param_score_1(gene_query_vec,sample_id,feature_query_vec,y,beta,tf_score_mtx=tf_score_mtx,motif_score_dict={},motif_prior_type=motif_prior_type)
		
# 		# beta_mtx_id = np.asarray(beta_mtx_init>0)
# 		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
# 		beta_mtx[beta_mtx_id1] = beta
# 		beta_mtx[beta_mtx_id2] = 0.0
# 		tf_score_beta = score_mtx*beta_mtx  # shape: cell_num by gene_num by motif_num

# 		print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
# 		y_pred = pd.DataFrame(index=sample_id,columns=gene_query_vec,data=0.0)
# 		for i2 in range(sample_num):
# 			sample_id1 = sample_id[i2]
# 			# t_value1 = x.loc[sample_id,motif_query_vec]
# 			# b1 = (t_value1>=0)
# 			# b2 = (t_value1<0)
# 			# y_pred.loc[sample_id1] = tf_score_beta[i2].dot(x.loc[sample_id1,motif_query_vec].abs())
# 			y_pred.loc[sample_id1] = np.sum(tf_score_beta[i2],axis=1)
			
# 		squared_error = ((y_pred-y)**2).sum().sum()*ratio1
# 		# y_pred.columns = ['pred.%s'%(t_gene_query) for t_gene_query in gene_query_vec]
# 		# df1 = pd.concat([y,y_pred],axis=1,join='outer',ignore_index=False,keys=None,levels=None,names=None,
# 		#                   verify_integrity=False,copy=True)

# 		# output_filename = '%s/test_peak_motif_estimate_pred.1.txt'%(self.save_path_1)
# 		# df1.to_csv(output_filename,sep='\t',float_format='%.6E')
# 		# print('objective function value ', squared_error)

# 		# return squared_error

# 		Lasso_1 = np.sum(np.absolute(beta))
# 		# motif_group = self.motif_group
# 		# motif_group = self.df_motif_group_query.loc[feature_query_vec,'group']
# 		gene_expr_corr = self.gene_expr_corr_
# 		print('H_mtx, H_p ',self.H_mtx.shape, self.H_p.shape)
# 		H_mtx = self.H_mtx.loc[gene_query_vec,:]
# 		H_p = self.H_p.loc[feature_query_vec,:]
# 		print('H_mtx, H_p ',H_mtx.shape, H_p.shape)

# 		## regularization based on gene expression correlation, graph-guided fused Lasso
# 		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
# 		# scale_factor_mtx = np.ones(beta_mtx.shape,dtype=np.float32)
# 		scale_factor_mtx = self.scale_factor_mtx
# 		beta_mtx_scaled = beta_mtx*scale_factor_mtx
# 		regularize_1 = np.sum(np.sum(np.absolute(beta_mtx_scaled.T.dot(H_mtx))))
# 		print('regularize_1',regularize_1, H_mtx.shape)

# 		## regularization based on tf groups
# 		regularize_2 = self.test_motif_peak_estimate_obj_regularize_2(beta_mtx,motif_group_list,motif_group_vec)
# 		print('regularize_2',regularize_2)

# 		## regularization based on gene expression correlation of tfs, graph-guided fused Lasso
# 		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
# 		if type_id_regularize==1:
# 			regularize_3 = np.sum(np.sum(np.absolute(beta_mtx_scaled.dot(H_p))))
# 			print('regularize_3',regularize_3, H_p.shape)
# 		else:
# 			regularize_3 = 0.0

# 		self.lambda_vec = [1E-03,[1E-03,1E-03,1E-03]]
# 		Lasso_eta_1, lambda_vec_2 = self.lambda_vec
# 		regularizer_vec_2 = np.asarray([regularize_1,regularize_2,regularize_3])

# 		# type_id_regularize = 1
# 		# if type_id_regularize==1:
# 		#   regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2.dot(lambda_vec_2)
# 		# else:
# 		#   regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2[0:2].dot(lambda_vec_2[0:2])
# 		regularize_pre = Lasso_1*Lasso_eta_1 + regularizer_vec_2.dot(lambda_vec_2)
		
# 		squared_error_ = squared_error + regularize_pre
# 		print('squared_error_, squared_error, regularize_pre ', squared_error_, squared_error, regularize_pre)

# 		return squared_error_

# 	## motif-peak estimate: objective function
# 	# tf_score: gene_num by tf_num
# 	def test_motif_peak_estimate_obj_constraint1(self,beta,score_mtx,y,beta_mtx=[],ratio=1,motif_group_list=[],motif_group_vec=[],lambda_vec=[],iter_cnt1=0,type_id_regularize=0):
		
# 		# beta_mtx_id = np.asarray(beta_mtx_init>0)
# 		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
# 		beta_mtx[beta_mtx_id1] = np.ravel(beta)
# 		beta_mtx[beta_mtx_id2] = 0.0
# 		tf_score_beta = score_mtx*beta_mtx  # shape: cell_num by gene_num by motif_num

# 		# print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
# 		y_pred = np.sum(tf_score_beta,axis=2)   # shape: cell_num by gene_num
# 		squared_error = ((y_pred-y)**2).sum().sum()*ratio

# 		## regularization using Lasso
# 		Lasso_1 = np.sum(np.absolute(beta))
		
# 		# H_mtx = self.H_mtx.loc[gene_query_vec,:]
# 		# H_p = self.H_p.loc[feature_query_vec,:]
# 		H_mtx = self.H_mtx_1
# 		H_p = self.H_p_1
# 		# print('H_mtx, H_p ',H_mtx.shape, H_p.shape)

# 		## regularization based on gene expression correlation, graph-guided fused Lasso
# 		# regularize_1 = self.test_motif_peak_estimate_obj_regularize_1(gene_expr_corr,beta_mtx)
# 		# scale_factor_mtx = np.ones(beta_mtx.shape,dtype=np.float32)
# 		# scale_factor_mtx = self.scale_factor_mtx
# 		# beta_mtx_scaled = beta_mtx*scale_factor_mtx
# 		# the first dimension corresponds to the intercept
# 		regularize_1 = np.sum(np.sum(np.absolute(beta_mtx[:,1:].T.dot(H_mtx))))
# 		# print('regularize_1',regularize_1, H_mtx.shape)

# 		## regularization based on tf groups
# 		# regularize_2 = self.test_motif_peak_estimate_obj_regularize_2(beta_mtx,motif_group_list,motif_group_vec)
# 		vec1 = [np.linalg.norm(beta_mtx[:,feature_query_id],axis=1) for feature_query_id in motif_group_list]
# 		group_regularize_vec = np.asarray(vec1).T.dot(motif_group_vec)
# 		regularize_2 = np.sum(group_regularize_vec)
# 		# print('regularize_2',regularize_2)

# 		## regularization based on gene expression correlation of tfs, graph-guided fused Lasso
# 		if type_id_regularize==1:
# 			regularize_3 = np.sum(np.sum(np.absolute(beta_mtx[:,1:].dot(H_p))))
# 			# print('regularize_3',regularize_3, H_p.shape)
# 		else:
# 			regularize_3 = 0.0

# 		# self.lambda_vec = [1E-03,[1E-03,1E-03,1E-03]]
# 		# Lasso_eta_1, lambda_vec_2 = self.lambda_vec
# 		Lasso_eta_1, lambda_vec_2 = lambda_vec
# 		regularize_vec_2 = np.asarray([regularize_1,regularize_2,regularize_3])
# 		regularize_vec_pre1 = [Lasso_1]+list(regularize_vec_2)

# 		regularize_pre = Lasso_1*Lasso_eta_1 + regularize_vec_2.dot(lambda_vec_2)
		
# 		if self.iter_cnt1==0:
# 			scale_factor = 1.1*regularize_pre/(squared_error+1E-12)
# 			print('scale_factor est ', scale_factor)
# 			self.scale_factor = scale_factor

# 		scale_factor = self.scale_factor
# 		squared_error_ = scale_factor*squared_error + regularize_pre

# 		if squared_error < self.config['obj_value_pre1']:
# 			print('obj_value_pre1 ', self.config['obj_value_pre1'], squared_error, self.iter_cnt1)
# 			self.config.update({'obj_value_pre1':squared_error})
# 			self.config.update({'regularize_vec_pre1':regularize_vec_pre1,'regularize_pre1':regularize_pre})

# 		if squared_error_ < self.config['obj_value_pre2']:
# 			print('obj_value_pre2 ', self.config['obj_value_pre2'], squared_error_, self.iter_cnt1)
# 			self.config.update({'obj_value_pre2':squared_error_})
# 			self.config.update({'regularize_vec_pre2':regularize_vec_pre1,'regularize_pre2':regularize_pre})

# 		self.iter_cnt1 += 1

# 		# if iter_cnt1%100==0:
# 		#   print('squared_error_, squared_error, regularize_pre ', squared_error_, squared_error, regularize_pre)
# 		#   print('regularize_1, regularize_2, regularize_3 ', regularize_1, regularize_2, regularize_3)

# 		return squared_error_

# 	# motif-peak estimate: optimize unit
# 	def test_motif_peak_estimate_optimize1_unit2(self,initial_guess,x_train,y_train,beta_mtx=[],pre_data_dict={},type_id_regularize=0):

# 		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG','Powell','CG','TNC','COBYLA','trust-constr',
# 						'dogleg','trust-ncg','trust-exact','trust-krylov']
# 		method_id = 2
# 		method_type_id = method_vec[method_id]
# 		method_type_id = 'SLSQP'
# 		# method_type_id = 'L-BFGS-B'

# 		id1, cnt = 0, 0
# 		flag1 = False
# 		small_eps = 1e-12
# 		# con1 = ({'type':'ineq','fun':lambda x:x-small_eps},
# 		#       {'type':'ineq','fun':lambda x:-x+1})

# 		# lower_bound, upper_bound = -100, 100
# 		lower_bound, upper_bound = pre_data_dict['param_lower_bound'], pre_data_dict['param_upper_bound']
# 		regularize_lambda_vec = pre_data_dict['regularize_lambda_vec']
# 		ratio1 = pre_data_dict['ratio']
# 		motif_group_list, motif_group_vec = pre_data_dict['motif_group_list'], pre_data_dict['motif_group_vec']
# 		beta_mtx = self.beta_mtx
# 		print('beta_mtx ', beta_mtx.shape)

# 		con1 = ({'type':'ineq','fun':lambda x:x-lower_bound},
# 				{'type':'ineq','fun':lambda x:-x+upper_bound})
# 		bounds = scipy.optimize.Bounds(lower_bound,upper_bound)

# 		x_train = np.asarray(x_train)
# 		y_train = np.asarray(y_train)

# 		# flag1 = True
# 		iter_cnt1 = 0
# 		tol = 0.0001
# 		if 'tol_pre1' in pre_data_dict:
# 			tol = pre_data_dict['tol_pre1']

# 		while (flag1==False):
# 			try:
# 				# res = minimize(self._motif_peak_prob_lik_constraint,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
# 				#       method=method_vec[method_id],constraints=con1,tol=1e-6,options={'disp':False})
# 				start=time.time()
# 				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
# 				#               constraints=con1,tol=tol,options={'disp':False})
# 				res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
# 								method=method_type_id,constraints=con1,tol=tol,options={'disp':False})

# 				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
# 				#               method=method_type_id,bounds=bounds,tol=tol,options={'disp':False})
				
# 				flag1 = True
# 				iter_cnt1 += 1
# 				stop=time.time()
# 				print('iter_cnt1 ',iter_cnt1,stop-start)
# 			except Exception as err:
# 				flag1 = False
# 				print("OS error: {0} motif_peak_estimate_optimize1_unit1 {1}".format(err,flag1))
# 				cnt = cnt + 1
# 				if cnt > 10:
# 					print('cannot find the solution! %d'%(cnt))
# 					break

# 		if flag1==True:
# 			param1 = res.x
# 			flag = self._check_params(param1,upper_bound=upper_bound,lower_bound=lower_bound)
# 			# t_value1 = param1[1:]
# 			small_eps = 1e-12
# 			# t_value1[t_value1<=0]=small_eps
# 			# t_value1[t_value1>1.0]=1.0
# 			quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
# 			t_vec1 = utility_1.test_stat_1(param1,quantile_vec=quantile_vec_1)
# 			print('parameter est ',flag, param1.shape, param1[0:10], t_vec1)
# 			return flag, param1

# 		else:
# 			print('did not find the solution!')
# 			return -2, initial_guess


# 	# motif-peak estimate: optimize unit
# 	def test_motif_peak_estimate_optimize1_unit1(self,initial_guess,x_train,y_train,beta_mtx=[],pre_data_dict={},type_id_regularize=0):

# 		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG','Powell','CG','TNC','COBYLA','trust-constr',
# 						'dogleg','trust-ncg','trust-exact','trust-krylov']
# 		method_id = 2
# 		method_type_id = method_vec[method_id]
# 		method_type_id = 'SLSQP'
# 		# method_type_id = 'L-BFGS-B'

# 		id1, cnt = 0, 0
# 		flag1 = False
# 		small_eps = 1e-12
# 		# con1 = ({'type':'ineq','fun':lambda x:x-small_eps},
# 		#       {'type':'ineq','fun':lambda x:-x+1})

# 		# lower_bound, upper_bound = -100, 100
# 		lower_bound, upper_bound = pre_data_dict['param_lower_bound'], pre_data_dict['param_upper_bound']
# 		regularize_lambda_vec = pre_data_dict['regularize_lambda_vec']
# 		ratio1 = pre_data_dict['ratio']
# 		motif_group_list, motif_group_vec = pre_data_dict['motif_group_list'], pre_data_dict['motif_group_vec']
# 		beta_mtx = self.beta_mtx
# 		print('beta_mtx ', beta_mtx.shape)

# 		con1 = ({'type':'ineq','fun':lambda x:x-lower_bound},
# 				{'type':'ineq','fun':lambda x:-x+upper_bound})
# 		bounds = scipy.optimize.Bounds(lower_bound,upper_bound)

# 		x_train = np.asarray(x_train)
# 		y_train = np.asarray(y_train)

# 		# lambda1 for group Lasso regularization, lambda2 for Lasso regularization
# 		# lambda1, lambda2 = pre_data_dict['lambda1'], pre_data_dict['lambda2']
# 		# lambda1 = (1-alpha)*lambda_regularize
# 		# lambda2 = alpha*lambda_regularize

# 		# res = minimize(self._motif_peak_prob_lik_constraint,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
# 		#               method=method_vec[method_id],constraints=con1,tol=1e-6,options={'disp':True})

# 		# res = minimize(self._motif_peak_prob_lik_constraint_copy,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
# 		#               constraints=con1,tol=1e-5,options={'disp':False})

# 		# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,motif_group_list,motif_group_vec,lambda_vec,type_id_regularize),
# 		#               constraints=con1,tol=1e-5,options={'disp':False})


# 		# flag1 = True
# 		iter_cnt1 = 0
# 		tol = 0.0001
# 		if 'tol_pre1' in pre_data_dict:
# 			tol = pre_data_dict['tol_pre1']

# 		while (flag1==False):
# 			try:
# 				# res = minimize(self._motif_peak_prob_lik_constraint,initial_guess,args=(x_train,y_train,lambda1,lambda2,motif_group_list),
# 				#       method=method_vec[method_id],constraints=con1,tol=1e-6,options={'disp':False})
# 				start=time.time()
# 				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
# 				#               constraints=con1,tol=tol,options={'disp':False})
# 				res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
# 								method=method_type_id,constraints=con1,tol=tol,options={'disp':False})

# 				# res = minimize(self.test_motif_peak_estimate_obj_constraint1,initial_guess,args=(x_train,y_train,beta_mtx,ratio1,motif_group_list,motif_group_vec,regularize_lambda_vec,iter_cnt1,type_id_regularize),
# 				#               method=method_type_id,bounds=bounds,tol=tol,options={'disp':False})
				
# 				flag1 = True
# 				iter_cnt1 += 1
# 				stop=time.time()
# 				print('iter_cnt1 ',iter_cnt1,stop-start)
# 			except Exception as err:
# 				flag1 = False
# 				print("OS error: {0} motif_peak_estimate_optimize1_unit1 {1}".format(err,flag1))
# 				cnt = cnt + 1
# 				if cnt > 10:
# 					print('cannot find the solution! %d'%(cnt))
# 					break

# 		if flag1==True:
# 			param1 = res.x
# 			flag = self._check_params(param1,upper_bound=upper_bound,lower_bound=lower_bound)
# 			# t_value1 = param1[1:]
# 			small_eps = 1e-12
# 			# t_value1[t_value1<=0]=small_eps
# 			# t_value1[t_value1>1.0]=1.0
# 			quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
# 			t_vec1 = utility_1.test_stat_1(param1,quantile_vec=quantile_vec_1)
# 			print('parameter est ',flag, param1.shape, param1[0:10], t_vec1)
# 			return flag, param1

# 		else:
# 			print('did not find the solution!')
# 			return -2, initial_guess

# 	# training for classification or regression
# 	## baseline method
# 	## type_id: 0: LR, 1: XGBClassifier or XGBR, 2: RF
# 	## type_id1: type_id=1, 0: XGBClassifier, 1: multi-class XGBClassifier, 5: XGBR
# 	# return: trained model
# 	def training_1(self,x_train,y_train,x_valid,y_valid,sample_weight=[],type_id=1,type_id1=0,model_path1="",select_config={}):

# 		# x_train, x_valid, y_train, y_valid = train_test_split(x_train1, y_train1, test_size=0.2, random_state=42)
# 		if 'select_config_comp' in select_config:
# 			select_config_comp = select_config['select_config_comp']
# 			max_depth, n_estimators = select_config_comp['max_depth'], select_config_comp['n_estimators']
# 			if (type_id in [1,2]):
# 				max_depth = 20
# 		else:
# 			max_depth, n_estimators = 10, 500

# 		if type_id in [0,'LR']:
# 			# print("linear regression")
# 			# model = LinearRegression().fit(x_train, y_train)
# 			model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
# 			model.fit(x_train, y_train)

# 		elif type_id in [1,'XGBClassifier']:
# 			objective_function_vec = ['binary:logistic','multi:softprob','multi:softmax']
# 			# type_id1=0
# 			objective_function_1 = objective_function_vec[type_id1]
# 			# print("xgboost classification")
# 			print(objective_function_1)
# 			model = xgboost.XGBClassifier(colsample_bytree=1,use_label_encoder=False,
# 											gamma=0, n_jobs=10, learning_rate=0.1,
# 											max_depth=max_depth, min_child_weight=1, 
# 											n_estimators=n_estimators,                                                                    
# 											reg_alpha=0, reg_lambda=0.1,
# 											objective=objective_function_1,
# 											subsample=1,random_state=0)
# 			# print("fitting model...")
# 			# if len(sample_weight)==0:
# 			#   model.fit(x_train, y_train)
# 			# else:
# 			#   print('sample weight',np.max(sample_weight),np.min(sample_weight))
# 			#   model.fit(x_train, y_train, sample_weight=sample_weight)

# 		elif type_id in [2,'XGBR']:
# 			# print("xgboost regression")
# 			model = xgboost.XGBRegressor(colsample_bytree=1,gamma=0,n_jobs=10,learning_rate=0.1,
# 											 max_depth=max_depth,min_child_weight=1,
# 											 n_estimators=n_estimators,                                                                    
# 											 reg_alpha=0,reg_lambda=1,
# 											 objective='reg:squarederror',
# 											 subsample=1,random_state=0)

# 		elif type_id in [3,'Lasso']:
# 			# print("xgboost regression")
# 			alpha = select_config['Lasso_alpha']
# 			max_iteration, tol = select_config['Lasso_max_iteration'], select_config['Lasso_tol']
# 			intercept_flag = select_config['intercept_flag']
# 			print('Lasso_alpha, max_interation, tol, intercept_flag ', alpha, max_iteration, tol, intercept_flag)
# 			# model = Lasso(alpha=alpha)
# 			selection_type_vec = ['random','cyclic']
# 			selection_type = selection_type_vec[1]
# 			model = Lasso(alpha=alpha,fit_intercept=intercept_flag, precompute=False, copy_X=True, max_iter=max_iteration, tol=tol, warm_start=False, positive=False, random_state=None, selection=selection_type)

# 			# print("fitting model...")
# 			# model.fit(x_train, y_train)

# 		elif type_id in [5,'RF']:
# 			# print("random forest regression")
# 			model = RandomForestRegressor(n_jobs=10,n_estimators=n_estimators,max_depth=max_depth,random_state=0)

# 		else:
# 			pass

# 		# print("fitting model...")
# 		if type_id in [0,3,'LR','Lasso']:
# 			model.fit(x_train, y_train)
# 		elif type_id in [1,2,5,'XGBClassifier','XGBR','RF']:
# 			if len(sample_weight)==0:
# 				model.fit(x_train, y_train)
# 			else:
# 				print('sample weight',np.max(sample_weight),np.min(sample_weight))
# 				model.fit(x_train, y_train,sample_weight=sample_weight)
# 		else:
# 			pass

# 		return model

# 	# feature importance estimate from the learned model
# 	def test_model_explain_pre1(self,model,x,y,feature_name,model_type_name,x_test=[],y_test=[],linear_type_id=0,select_config={}):

# 		if model_type_name in ['XGBR','XGBClassifier','RF','RandomForestClassifier']:
# 			# pred = model.predict(x, output_margin=True)
# 			# pred = model.predict(x)
# 			explainer = shap.TreeExplainer(model)
# 			# explainer = shap.Explainer(model,x)
# 			shap_value_pre1 = explainer(x)
# 			shap_values = shap_value_pre1.values
# 			base_values = shap_value_pre1.base_values

# 			# expected_value = []
# 			expected_value = base_values[0]
# 			# expected_value = explainer.expected_value
# 			# t1 = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
# 			# print(t1)
# 		elif linear_type_id>=1:
# 			# shap.explainers.Linear(model, masker, link=CPUDispatcher(<function identity>), nsamples=1000, feature_perturbation=None, **kwargs)
# 			feature_perturbation = ['interventional','correlation_dependent']
# 			# explainer = shap.explainers.Linear(model,x,nsamples=x.shape[0],feature_perturbation=None)
# 			feature_perturbation_id1 = linear_type_id-1
# 			feature_perturbation_id2 = feature_perturbation[feature_perturbation_id1]
# 			print(feature_perturbation_id2)
# 			explainer = shap.explainers.Linear(model,x,nsamples=x.shape[0],feature_perturbation=feature_perturbation_id2)
# 			shap_value_pre1 = explainer(x)
# 			shap_values = shap_value_pre1.values
# 			base_values = shap_value_pre1.base_values
# 			expected_value = explainer.expected_value
# 		else:
# 			explainer = shap.Explainer(model, x, feature_names=feature_name)
# 			shap_value_pre1 = explainer(x)
# 			shap_values = shap_value_pre1.values
# 			base_values = shap_value_pre1.base_values
# 			# data_vec1 = shap_value_pre1.data
# 			# shap_values = explainer.shap_values(x)
# 			# base_values = []
# 			expected_value = explainer.expected_value
# 			# shap_values_test = explainer(x_test)
# 			pred = model.predict(x)
# 			t1 = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
# 			# print(t1)
		
# 		return shap_values, base_values, expected_value

# 	# estimate feature importance from the learned model
# 	def test_model_explain_1(self,model,x,y,feature_name,model_type_id,x_test=[],y_test=[],linear_type_id=0):

# 		if model_type_id in ['XGBR','XGBClassifier','RF','RandomForestClassifier']:
# 			# pred = model.predict(x, output_margin=True)
# 			# pred = model.predict(x)
# 			explainer = shap.TreeExplainer(model)
# 			# explainer = shap.Explainer(model,x)
# 			shap_value_pre1 = explainer(x)
# 			shap_values = shap_value_pre1.values
# 			base_values = shap_value_pre1.base_values

# 			# expected_value = []
# 			expected_value = base_values[0]
# 			# expected_value = explainer.expected_value
# 			# t1 = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
# 			# print(t1)
# 		elif linear_type_id>=1:
# 			# shap.explainers.Linear(model, masker, link=CPUDispatcher(<function identity>), nsamples=1000, feature_perturbation=None, **kwargs)
# 			feature_perturbation = ['interventional','correlation_dependent']
# 			# explainer = shap.explainers.Linear(model,x,nsamples=x.shape[0],feature_perturbation=None)
# 			feature_perturbation_id1 = linear_type_id-1
# 			feature_perturbation_id2 = feature_perturbation[feature_perturbation_id1]
# 			print(feature_perturbation_id2)
# 			explainer = shap.explainers.Linear(model,x,nsamples=x.shape[0],feature_perturbation=feature_perturbation_id2)
# 			shap_value_pre1 = explainer(x)
# 			shap_values = shap_value_pre1.values
# 			base_values = shap_value_pre1.base_values
# 			expected_value = explainer.expected_value
# 		else:
# 			explainer = shap.Explainer(model, x, feature_names=feature_name)
# 			shap_value_pre1 = explainer(x)
# 			shap_values = shap_value_pre1.values
# 			base_values = shap_value_pre1.base_values
# 			# data_vec1 = shap_value_pre1.data
# 			# shap_values = explainer.shap_values(x)
# 			# base_values = []
# 			expected_value = explainer.expected_value
# 			# shap_values_test = explainer(x_test)
# 			pred = model.predict(x)
# 			t1 = np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
# 			# print(t1)
		
# 		return shap_values, base_values, expected_value

# 	# motif-peak estimate: beta parameter initialize
# 	def test_motif_peak_estimate_param_init_1(self,x_train,y_train,x_valid=[],y_valid=[],sample_weight=[],gene_query_vec=[],sample_id=[],feature_query_vec=[],peak_read=[],meta_exprs=[],motif_prior_type=0,output_filename='',save_mode=1,score_type=1,select_config={}):

# 		method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG']
# 		method_id = 2
# 		id1, cnt = 0, 0
# 		flag1 = False
# 		small_eps = 1e-12
# 		sample_num, gene_num, feature_query_num = x_train.shape

# 		dict_gene_param_id = self.dict_gene_param_id_
# 		param_num = self.param_num
# 		beta_param_init = np.zeros(param_num,dtype=np.float32)

# 		if len(x_valid)==0:
# 			validation_size = 0.1
# 			id1 = np.arange(sample_num)
# 			id_train, id_valid, id_train1, id_valid1 = train_test_split(id1,id1,test_size=validation_size,random_state=1)
# 			x_train_pre, x_valid, y_train_pre, y_valid = x_train[id_train], x_train[id_valid], y_train[id_train], y_train[id_valid]
# 			if len(sample_weight)>0:
# 				sample_weight_train = sample_weight[id_train]
# 			else:
# 				sample_weight_train = []
# 			print('train, test ', x_train.shape, y_train.shape, x_train_pre.shape, y_train_pre.shape, x_valid.shape, y_valid.shape)

# 		# model_type_vec = ['LR','Lasso','XGBClassifier','XGBR','RF']
# 		# model_type_id = 'Lasso'
# 		# model_type_id1 = 1
# 		# intercept_flag = select_config['intercept_flag']
# 		# print('intercept_flag ', intercept_flag)
# 		# if model_type_id in [3,'Lasso']:
# 		#   Lasso_alpha = 1E-03
# 		#   Lasso_max_iteration = 5000
# 		#   tol = 0.0005
# 		#   select_config.update({'Lasso_alpha':Lasso_alpha,'Lasso_max_iteration':Lasso_max_iteration,'Lasso_tol':tol})
		
# 		df_gene_motif_prior = self.df_gene_motif_prior_
# 		df_gene_motif_prior = df_gene_motif_prior.loc[:,feature_query_vec]

# 		score_list_1 = []
# 		# if intercept_flag==True, the first dimension in the feature matrix corresponds to the intercept term
# 		intercept_flag = select_config['intercept_flag']
# 		model_type_id, model_type_id1 = select_config['model_type_id_init'],select_config['model_type_id1_init']
# 		id1 = int(intercept_flag)
# 		for i1 in range(gene_num):
# 			x_train1, x_valid1 = x_train_pre[:,i1,id1:], x_valid[:,i1,id1:]
# 			y_train1, y_valid1 = y_train_pre[:,i1], y_valid[:,i1]
			
# 			model_train = self.training_1(x_train1,y_train1,x_valid,y_valid,sample_weight=sample_weight_train,type_id=model_type_id,type_id1=model_type_id1,model_path1="",select_config=select_config)
# 			coef_, intercept_ = model_train.coef_, model_train.intercept_

# 			y_pred_valid1 = model_train.predict(x_valid1)
# 			score_list = self.test_score_pred_1(y_valid1, y_pred_valid1)
# 			score_vec = score_list[0]
# 			score_list_1.append(score_vec)

# 			t_gene_query = gene_query_vec[i1]
# 			gene_query_id = t_gene_query
# 			param_id = dict_gene_param_id[gene_query_id]
# 			t_vec1 = utility_1.test_stat_1(coef_)
# 			print(gene_query_id,i1,intercept_,t_vec1)
# 			print(score_vec)

# 			t_motif_prior = df_gene_motif_prior.loc[gene_query_id,:]
# 			# b1 = np.where(t_motif_prior>0)[0]
# 			motif_query_vec1 = feature_query_vec[t_motif_prior>0]

# 			beta_param_init_est = pd.Series(index=feature_query_vec,data=np.asarray(coef_))
# 			t_param_init_est = beta_param_init_est.loc[motif_query_vec1]
# 			if intercept_flag==True:
# 				t_param_init_est_1 = np.asarray([intercept_]+list(t_param_init_est))
# 			else:
# 				t_param_init_est_1 = np.asarray(t_param_init_est)

# 			beta_param_init[param_id] = t_param_init_est_1

# 		field_1 = ['mse','pearsonr','pval1','explained_variance','mean_absolute_error','median_absolute_error','r2_score','spearmanr','pval2']
# 		score_pred = np.asarray(score_list_1)
# 		df1 = pd.DataFrame(index=gene_query_vec,columns=field_1,data=score_pred,dtype=np.float32)

# 		if save_mode==1:
# 			if output_filename=='':
# 				output_filename = 'test_motif_peak_estimate_param_init_score_1.txt'
# 			df1.to_csv(output_filename,sep='\t',float_format='%.6E')

# 		return beta_param_init, df1

# 	# motif-peak estimate: model train evaluation
# 	def test_score_pred_1(self,y,y_pred,x=[],sample_weight=[],model_train=[],gene_query_vec=[],sample_id=[],feature_query_vec=[],save_mode=1,select_config={}):

# 		y, y_pred = np.asarray(y), np.asarray(y_pred)
# 		# print('y, y_pred ',y.shape, y_pred.shape)
# 		score_list = []
# 		if y.ndim==1:
# 			y, y_pred = y[:,np.newaxis], y_pred[:,np.newaxis]
		
# 		query_num = y.shape[1]
# 		# print('y, y_pred ',y.shape, y_pred.shape)
# 		for i1 in range(query_num):
# 			# mse, pearsonr, pval1, explained_variance, median_abs_error, mean_abs_error, r2_score, spearmanr, pval2
# 			vec1 = self.score_2a(y[:,i1],y_pred[:,i1])
# 			print('score pred ',vec1,i1)
# 			score_list.append(list(vec1))
		
# 		return score_list

# 	## motif-peak estimate: model train prediction
# 	def test_motif_peak_estimate_pred_1(self,x,select_config={}):

# 		beta = self.param_est
# 		beta_mtx, beta_mtx_id1, beta_mtx_id2 = self.beta_mtx, self.beta_mtx_id1, self.beta_mtx_id2
# 		beta_mtx[beta_mtx_id1] = np.ravel(beta)
# 		beta_mtx[beta_mtx_id2] = 0.0
# 		tf_score_beta = x*beta_mtx  # shape: cell_num by gene_num by motif_num

# 		# print('tf_score_beta, beta_mtx ', tf_score_beta.shape, beta_mtx.shape)
# 		y_pred = np.sum(tf_score_beta,axis=2)   # shape: cell_num by gene_num
		
# 		return y_pred

# 	## motif-peak estimate: model train explain
# 	def test_motif_peak_estimate_explain_1(self,gene_query_vec=[],sample_id=[],feature_query_vec=[],peak_read=[],meta_exprs=[],input_filename1='',input_filename2='',select_config={}):
# 		# pre_data_dict_1 = {'tf_score_mtx':tf_score_mtx,'gene_expr':gene_expr,
# 		#                   'sample_id_train':sample_id_train,'sample_id_test':sample_id_test,
# 		#                   'sample_id':sample_id,'gene_query_vec':gene_query_vec,'feature_query_vec':feature_query_vec,
# 		#                   'dict_gene_param_id':dict_gene_param_id,
# 		#                   'df_gene_motif_prior':df_gene_motif_prior_}

# 		score_type_vec1 = ['unnormalized','normalized']
# 		score_type_vec2 = ['tf_score','tf_score_product','tf_score_product_2']
# 		# score_type_id = select_config['tf_score_type']
# 		# score_type_id2 = select_config['tf_score_type_id2']
# 		score_type_id = 1
# 		score_type_id2 = 0
# 		score_type_1, score_type_2 = score_type_vec1[score_type_id], score_type_vec2[score_type_id2]
# 		print('tf score type 1, tf score type 2 ', score_type_1, score_type_2)
		
# 		file_path2 = 'vbak1'
# 		# score_type_1, score_type_2 = 1, 2
# 		filename_annot1_pre1 = '%s_%s'%(score_type_2,score_type_1)
# 		input_filename1 = '%s/test_pre_data_dict_1_%s.1.npy'%(file_path2,filename_annot1_pre1)
# 		data1 = np.load(input_filename1,allow_pickle=True)
# 		data1 = data1[()]
# 		print(list(data1.keys()))
		
# 		pre_data_dict_1 = data1
# 		gene_query_vec, sample_id, feature_query_vec = pre_data_dict_1['gene_query_vec'], pre_data_dict_1['sample_id'], pre_data_dict_1['feature_query_vec']
# 		dict_gene_param_id = pre_data_dict_1['dict_gene_param_id']
# 		df_gene_motif_prior_ = pre_data_dict_1['df_gene_motif_prior']

# 		# field_query = ['tf_score_mtx','gene_expr','sample_id_train','sample_id_test',
# 		#               'gene_query_vec','sample_id','feature_query_vec',
# 		#               'dict_gene_param_id','df_gene_motif_prior']

# 		# list1 = [pre_data_dict_1[t_field_query] for t_field_query in field_query]

# 		tf_score_mtx, gene_expr = pre_data_dict_1['tf_score_mtx'], pre_data_dict_1['gene_expr']
# 		sample_id_train, sample_id_test = pre_data_dict_1['sample_id_train'], pre_data_dict_1['sample_id_test']
		
# 		print('tf_score_mtx, gene_expr ', tf_score_mtx.shape, gene_expr.shape)
# 		print('sample_id_train, sample_id_test ', sample_id_train.shape, sample_id_test.shape)

# 		# pre_data_dict_2.update({'beta_param_est':beta_param_est,'scale_factor':self.scale_factor,'iter_cnt1':self.iter_cnt1,
# 		#                               'config_pre':config_pre,'y_test':y_test,'y_pred_test':y_pred_test})

# 		ratio_1, ratio_2 = 0.9, 0.1
# 		Lasso_eta_1 = 0.001
# 		lambda_1 = 0.01

# 		if 'regularize_lambda_vec' in select_config:
# 			regularize_lambda_vec = select_config['regularize_lambda_vec']
# 			Lasso_eta_1, lambda_vec_2 = regularize_lambda_vec
# 			lambda_1,lambda_2,lambda_3 = lambda_vec_2

# 		filename_annot1_1 = '%s_%s_%s_%s'%(str(ratio_1),str(ratio_2),str(Lasso_eta_1),str(lambda_1))
# 		filename_annot1 = '%s_%s'%(filename_annot1_1,filename_annot1_pre1)

# 		input_filename2 = '%s/test_pre_data_dict_2_%s.1.npy'%(file_path2,filename_annot1)
		
# 		data2 = np.load(input_filename2,allow_pickle=True)
# 		data2 = data2[()]
# 		print(list(data2.keys()))
# 		pre_data_dict_2 = data2
# 		beta_param_est = pre_data_dict_2['beta_param_est']
# 		param_num = beta_param_est.shape[0]

# 		print('gene_query_vec, sample_id, feature_query_vec', len(gene_query_vec), len(sample_id), len(feature_query_vec))
# 		print('beta_param_est ', beta_param_est.shape)

# 		save_mode = 0
# 		intercept_flag = True
# 		dict_gene_param_id_1, param_num_1, beta_mtx_init = self.test_motif_peak_estimate_param_id(gene_query_vec=gene_query_vec,
# 																								feature_query_vec=feature_query_vec,
# 																								df_gene_motif_prior=df_gene_motif_prior_,
# 																								intercept=intercept_flag,
# 																								save_mode=save_mode)

# 		assert param_num==param_num_1

# 		beta_mtx_id1 = np.asarray(beta_mtx_init>0)
# 		beta_mtx = np.zeros(beta_mtx_init.shape,dtype=np.float32)
# 		beta_mtx[beta_mtx_id1] = np.ravel(beta_param_est)

# 		output_filename = 'test_beta_mtx_est.1.txt'
# 		if intercept_flag==True:
# 			t_columns = ['1']+list(feature_query_vec)
# 		else:
# 			t_columns = list(feature_query_vec)

# 		df_beta_1 = pd.DataFrame(index=gene_query_vec,columns=t_columns,data=beta_mtx,dtype=np.float32)
# 		df_beta_1.to_csv(output_filename,sep='\t',float_format='%.6E')
# 		print('df_beta_1, param_num_1 ', df_beta_1.shape, param_num_1)

# 		sample_id_train_pre = [pd.Index(sample_id).get_loc(sample_id1) for sample_id1 in sample_id_train]
# 		sample_id_test_pre = [pd.Index(sample_id).get_loc(sample_id1) for sample_id1 in sample_id_test]

# 		print('sample_id_train, sample_id_test ', len(sample_id_train), len(sample_id_test), sample_id_train[0:5], sample_id_test[0:5])
# 		print('sample_id_train_pre, sample_id_test_pre ', len(sample_id_train_pre), len(sample_id_test_pre), sample_id_train_pre[0:5], sample_id_test_pre[0:5])

# 		x_train = tf_score_mtx[sample_id_train_pre]
# 		y_train = gene_expr.loc[sample_id_train]

# 		x_test = tf_score_mtx[sample_id_test_pre]
# 		y_test = gene_expr.loc[sample_id_test]

# 		print('x_train, y_train, x_test, y_test ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 		x_train, y_train = np.asarray(x_train), np.asarray(y_train)
# 		x_test, y_test = np.asarray(x_test), np.asarray(y_test)

# 		model_type_name = 'LR'
# 		dict1 = dict()
# 		gene_query_num = len(gene_query_vec)
# 		param_est_1 = []
# 		param_est_2 = feature_query_vec

# 		param_est_1, param_est_2 = pd.Index(param_est_1), pd.Index(param_est_2)

# 		print('load TF motif group')
# 		method_type_id1 = 2
# 		df_motif_group, df_motif_group_query = self.test_motif_peak_estimate_motif_group_pre1(motif_query_vec=feature_query_vec,method_type_id1=method_type_id1)
# 		self.df_motif_group = df_motif_group
# 		self.df_motif_group_query = df_motif_group_query
# 		print('df_motif_group, df_motif_group_query ', df_motif_group.shape, df_motif_group_query.shape)

# 		df_motif_group = self.df_motif_group
# 		motif_cluster_dict = self.motif_cluster_dict

# 		for i1 in range(gene_query_num):
# 			gene_query_id = gene_query_vec[i1]
# 			t_gene_query = gene_query_id
# 			model_1 = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
# 			# model.fit(x_train, y_train)

# 			model_1.intercept_ = df_beta_1.loc[gene_query_id,'1']
# 			t_param_est = df_beta_1.loc[gene_query_id,feature_query_vec]
# 			model_1.coef_ = t_param_est
# 			print('model_1 ', gene_query_id, i1)
# 			print(model_1.intercept_, model_1.coef_)

# 			# shap_value_1, base_value_1, expected_value_1 = self.test_model_explain_pre1(model_1,x=x_train,y=y_train,
# 			#                                                                               feature_name=feature_query_vec,
# 			#                                                                               model_type_name=model_type_name,
# 			#                                                                               x_test=x_test,y_test=y_test,
# 			#                                                                               linear_type_id=0,select_config=select_config)

# 			# feature_importances_1 = np.mean(np.abs(shap_value_1),axis=0)  # mean_abs_value
# 			# feature_importances_pre1 = pd.DataFrame(index=feature_query_vec,columns=['mean_abs_shap_value'],data=feature_importances_1,dtype=np.float32)
# 			# feature_importances_pre1_sort = feature_importances_pre1.sort_values(by=['mean_abs_shap_value'],ascending=False)

# 			feature_importances_pre1_sort = []
 
# 			thresh1 = 0
# 			b1 = np.where(np.abs(t_param_est)>thresh1)[0]
# 			feature_query_1 = feature_query_vec[b1]
# 			t_param_est_1 = t_param_est[feature_query_1]
# 			sort_id_1 = t_param_est_1.abs().sort_values(ascending=False).index
# 			t_param_est_sort1 = t_param_est_1[sort_id_1]
# 			df1 = pd.DataFrame(index=sort_id_1,columns=['coef_'],data=np.asarray(t_param_est_sort1))
# 			df1['motif_group'] = df_motif_group.loc[sort_id_1,'group']
# 			motif_group_query = df_motif_group.loc[sort_id_1,'group']
# 			motif_group_query_vec = np.unique(motif_group_query)
			
# 			list2 = []
# 			list3 = []
# 			t_motif_prior = df_gene_motif_prior.loc[gene_query_id,:]
# 			# b1 = np.where(t_motif_prior>0)[0]
# 			t_feature_query = feature_query_vec[t_motif_prior>0]
# 			motif_query_ori = df_motif_group.index

# 			for t_group_id in motif_group_query_vec:
# 				motif_group_query_1 = motif_query_ori[df_motif_group['group']==t_group_id]
# 				motif_query_num1 = len(motif_group_query_1)
# 				list2.append(motif_query_num1)

# 				motif_group_query_2 = motif_group_query_1.intersection(t_feature_query)
# 				motif_query_num2 = len(motif_group_query_2)
# 				list3.append(motif_query_num2)

# 			df2 = pd.DataFrame(index=motif_group_query_vec,columns=['motif_num1'],data=np.asarray(list2))
# 			print('motif_group_query_vec ', len(motif_group_query_vec))
# 			df2['motif_num2'] = np.asarray(list3)

# 			motif_query_num1 = df2.loc[np.asarray(motif_group_query),'motif_num1']
# 			motif_query_num2 = df2.loc[np.asarray(motif_group_query),'motif_num2']
# 			df1['motif_neighbor_ori'] = np.asarray(motif_query_num1)
# 			df1['motif_neighbor'] = np.asarray(motif_query_num2)

# 			dict1[gene_query_id] = [feature_importances_pre1_sort, df1]
# 			print('feature importance est ', feature_importances_pre1_sort[0:5], t_param_est_sort1[0:5], len(feature_query_1), gene_query_id, i1)

# 			param_est_1 = param_est_1.union(feature_query_1,sort=False)
# 			param_est_2 = param_est_2.intersection(feature_query_1,sort=False)

# 		output_filename = 'test_beta_param_est_pre1.%s.npy'%(filename_annot1_1)
# 		np.save(output_filename,dict1,allow_pickle=True)

# 		df1 = df_beta_1.loc[:,param_est_1]
# 		df2 = df_beta_1.loc[:,param_est_2]
# 		output_filename1 = 'test_beta_param_est_sort_1.%s.txt'%(filename_annot1_1)
# 		mean_abs_value1 = df1.abs().mean(axis=0)
# 		sort_id1 = mean_abs_value1.sort_values(ascending=False).index
# 		df1_sort = df1.loc[:,sort_id1]
# 		df1_sort.to_csv(output_filename1,sep='\t',float_format='%.6E')

# 		output_filename2 = 'test_beta_param_est_sort_2.%s.txt'%(filename_annot1_1)
# 		mean_abs_value2 = df2.abs().mean(axis=0)
# 		sort_id2 = mean_abs_value2.sort_values(ascending=False).index
# 		df2_sort = df2.loc[:,sort_id2]
# 		df2_sort.to_csv(output_filename2,sep='\t',float_format='%.6E')

# 		feature_query_vec_1 = ['Pdx1','Foxa1','Foxa3']
# 		for t_feature_query in feature_query_vec_1:
# 			print(t_feature_query)
# 			print(df1_sort[t_feature_query])

# 		gene_query_vec_1 = gene_query_vec
# 		gene_query_num1 = len(gene_query_vec_1)
# 		for i1 in range(gene_query_num1):
# 			gene_query_id = gene_query_vec_1[i1]
# 			feature_importances_pre1_sort, df1 = dict1[gene_query_id]
# 			print(gene_query_id,i1)
# 			print(df1[0:50])

# 		print('df1_sort, df2_sort ', df1_sort.shape, df2_sort.shape)

# 		return

# 	## motif-peak estimate: load gene query vector
# 	## motif-peak estimate: optimization
# 	def test_motif_peak_estimate_optimize_1(self,gene_query_vec=[],sample_id=[],feature_query_vec=[],dict_score_query={},gene_tf_prior_1=[],gene_tf_prior_2=[],peak_read=[],meta_exprs=[],motif_prior_type=0,save_mode=1,load_score_mode=0,score_type=1,type_score_pair=0,select_config={}):
		
# 		## load data
# 		# self.gene_name_query_expr_
# 		# self.gene_highly_variable
# 		# self.motif_data; self.motif_data_cluster; self.motif_group_dict
# 		# self.pre_data_dict_1
# 		# self.motif_query_name_expr, self.motif_data_local_
# 		# self.test_motif_peak_estimate_pre_load(select_config=select_config)

# 		# ## load the gene query vector
# 		# # celltype_query_vec = ['Trachea.Lung']
# 		# # gene_query_vec = self.test_motif_peak_gene_query_load(celltype_query_vec=celltype_query_vec)

# 		# if len(gene_query_vec)==0:
# 		#   # gene_idvec = self.gene_idvec   # the set of genes
# 		#   # gene_query_vec = gene_idvec
# 		#   gene_query_vec = self.gene_highly_variable # the set of genes

# 		# gene_query_num = len(gene_query_vec)
# 		# print('celltype query, gene_query_vec ', celltype_query_vec, gene_query_num)
# 		# self.gene_query_vec = gene_query_vec

# 		# motif_query_name_expr = self.motif_query_name_expr
# 		# self.motif_data_expr = self.motif_data.loc[:,motif_query_name_expr]
# 		# print('motif_query_name_expr ', len(motif_query_name_expr), self.motif_data_expr.shape)
# 		# if len(feature_query_vec)==0:
# 		#   feature_query_vec = motif_query_name_expr

# 		# # motif_query_vec = motif_query_name_expr
# 		# motif_query_vec = feature_query_vec

# 		# # sample_id = self.meta_scaled_exprs_2.index   # sample id
# 		# sample_num = len(sample_id)
# 		# self.sample_id = sample_id

# 		# ## motif-peak estimate: objective function
# 		# # initialize variables that are not changing with parameters
# 		# pre_load_1, pre_load_2 = 1, 1
# 		# # self.test_motif_peak_estimate_optimize_init_pre1(gene_query_vec=gene_query_vec,
# 		# #                                                 sample_id=sample_id,
# 		# #                                                 feature_query_vec=motif_query_vec,
# 		# #                                                 select_config=select_config,
# 		# #                                                 pre_load_1=pre_load_1,
# 		# #                                                 pre_load_2=pre_load_2)

# 		# ## commented
# 		# ## prepare the correlated peaks of each gene
# 		# # dict_peak_local_ =  self.test_motif_peak_estimate_peak_gene_association1(gene_query_vec=gene_query_vec,select_config=select_config)
# 		# # key_vec = list(dict_peak_local_.keys())
# 		# # print('dict_peak_local ', len(key_vec), key_vec[0:5])
# 		# # self.dict_peak_local_ = dict_peak_local_

# 		# ## prepare the motif prior of each gene
# 		self.gene_motif_prior_1 = gene_tf_prior_1   # motif_prior based on union of prior from motif-gene association and prior from gene-tf association
# 		self.gene_motif_prior_2 = gene_tf_prior_2   # motif_prior based on intersection of prior from motif-gene association and prior from gene-tf association

# 		print('gene_motif_prior_1, gene_motif_prior_2 ', self.gene_motif_prior_1.shape, self.gene_motif_prior_2.shape)
# 		# return

# 		# ## initialize the output variable graph based on expression correlation
# 		# # the edge set for the vertex-edge incidence matrix (VE) of output variables based on expression correlation
# 		# thresh_corr_repsonse, thresh_pval_response = 0.2, 0.05
# 		# if 'thresh_corr_repsonse' in select_config:
# 		#   thresh_corr_repsonse = select_config['thresh_corr_repsonse']
# 		# if 'thresh_pval_response' in select_config:
# 		#   thresh_pval_repsonse = select_config['thresh_pval_repsonse']
		
# 		# query_type_id = 0
# 		# response_edge_set, response_graph_connect, response_graph_list = self.test_motif_peak_estimate_graph_pre1(feature_query_vec=gene_query_vec,
# 		#                                                                                       query_type_id=query_type_id,
# 		#                                                                                       select_config=select_config,
# 		#                                                                                       thresh_corr=thresh_corr_repsonse,
# 		#                                                                                       thresh_pval=thresh_pval_response,
# 		#                                                                                       load_mode=0)

# 		# gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1 = response_graph_list
# 		# self.gene_expr_corr_, self.gene_expr_pval_, self.gene_expr_pval_corrected = gene_expr_corr_1, gene_expr_pval_1, gene_expr_pval_corrected1
# 		# print('response edge set ', len(gene_query_vec), len(response_edge_set))

# 		# # return

# 		# ## initialize the input variable graph based on expression correlation
# 		# # the edge set for the VE matrix of input variables based on expression correlation
# 		# thresh_corr_predictor, thresh_pval_predictor = 0.3, 0.05
# 		# if 'thresh_corr_predictor' in select_config:
# 		#   thresh_corr_predictor = select_config['thresh_corr_predictor']
# 		# if 'thresh_pval_predictor' in select_config:
# 		#   thresh_pval_predictor = select_config['thresh_pval_predictor']

# 		# query_type_id = 1
# 		# # motif_query_name = self.motif_query_name_expr
# 		# # print('motif_query_name ', len(motif_query_name))
# 		# predictor_edge_set, predictor_graph_connect, predictor_graph_list = self.test_motif_peak_estimate_graph_pre1(feature_query_vec=motif_query_vec,
# 		#                                                                                                               query_type_id=query_type_id,
# 		#                                                                                                               select_config=select_config,
# 		#                                                                                                               thresh_corr=thresh_corr_predictor,
# 		#                                                                                                               thresh_pval=thresh_pval_predictor,
# 		#                                                                                                               load_mode=0)

# 		# gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2 = predictor_graph_list
# 		# self.tf_expr_corr_, self.tf_expr_pval_, self.tf_expr_pval_corrected = gene_expr_corr_2, gene_expr_pval_2, gene_expr_pval_corrected2
# 		# print('predictor edge set ', len(motif_query_name), len(predictor_edge_set))

# 		# # return

# 		# ## initialize the VE matrix of output variables
# 		# H = self.test_motif_peak_estimate_VE_init_(query_id=gene_query_vec,
# 		#                                           edge_set=response_edge_set,
# 		#                                           df_graph=gene_expr_corr_1)
# 		# self.H_mtx = H
# 		# print('VE matrix of response variable graph ', H.shape)

# 		# ## initialize the vertex-edge incidence matrix of input variables
# 		# H_p = self.test_motif_peak_estimate_VE_init_(query_id=motif_query_name,
# 		#                                           edge_set=predictor_edge_set,
# 		#                                           df_graph=gene_expr_corr_2)
# 		# self.H_p = H_p
# 		# print('VE matrix of predictor variable graph ', H_p.shape)

# 		flag_pre_load=load_score_mode
# 		field_1 = ['unweighted','weighted','product.unweighted','product.weighted','gene_query_id','sample_id','feature_query']
		
# 		# score_type, type_score_pair = 1, 0
# 		# if 'tf_score_type' in select_config:
# 		#   score_type = select_config['tf_score_type']
# 		# if 'type_score_pair' in select_config:
# 		#   type_score_pair = select_config['type_score_pair']

# 		# motif_query_vec = feature_query_vec
# 		if flag_pre_load==0:
# 			# dict1, dict2 = self.test_motif_peak_estimate_feature_mtx_scale(gene_query_vec=gene_query_vec,
# 			#                                                               motif_query_vec=motif_query_vec,
# 			#                                                               select_config=select_config)

# 			# score_mtx = numpy.swapaxes(score_mtx, axis1=0, axis2=1) # size: sample_num by gene num by feature query num
# 			# score_dict = {'gene_query':gene_query_vec,'sample_id':sample_id,'feature_query':feature_query_vec,'score_mtx':score_mtx}

# 			score_dict = self.test_motif_peak_estimate_gene_tf_estimate_pre1(gene_query_vec=gene_query_vec,
# 																				feature_query_vec=feature_query_vec,
# 																				dict_score_query=dict_score_query,
# 																				peak_read=peak_read,meta_exprs=meta_exprs,
# 																				score_type=score_type,
# 																				type_score_pair=type_score_pair,
# 																				save_mode=save_mode,
# 																				select_config=select_config)

# 			# return

# 		else:
# 			log_type_id_atac, scale_type_id = 1, 2
# 			# output_filename1 = '%s/test_peak_motif_estimate_tf_score.log_%d.scale_%d.pre1.npy'%(self.save_path_1,log_type_id_atac,scale_type_id)
# 			# output_filename2 = '%s/test_peak_motif_estimate_tf_score.log_%d.scale_%d.npy'%(self.save_path_1,log_type_id_atac,scale_type_id)
# 			# # np.save(output_filename1,dict1,allow_pickle=True)
# 			# # np.save(output_filename2,dict2,allow_pickle=True)
# 			# dict1 = np.load(output_filename1,allow_pickle=True)
# 			# dict1 = dict1[()]
# 			# dict2 = np.load(output_filename2,allow_pickle=True)
# 			# dict2 = dict2[()]
# 			input_filename = ''
# 			dict1 = np.load(input_filename,allow_pickle=True)
# 			score_dict = dict1[()]
		
# 		# scale_type_id = 0
# 		# # scale_type_id = 1
# 		# t_field1 ='product.weighted' 
# 		# mtx2 = dict2[t_field1]
# 		# mtx2 = mtx2[:,:,0:-1]
# 		# mtx1 = dict1[t_field1]

# 		# print('mtx2',t_field1,mtx2.shape)
# 		# print('mtx1',t_field1,mtx1.shape)

# 		# list_1 = [mtx1,mtx2]

# 		# gene_query_vec_1 = dict2['gene_query_id']
# 		# gene_query_vec = gene_query_vec_1
# 		# sample_id = dict2['sample_id']
# 		# motif_query_vec = dict2['feature_query']
# 		gene_query_vec, sample_id, feature_query_vec_pre1 = score_dict['gene_query'], score_dict['sample_id'], score_dict['feature_query']
# 		# motif_query_vec = feature_query_vec_pre1
# 		tf_score_mtx = score_dict['score_mtx']
# 		motif_query_vec = feature_query_vec_pre1
# 		gene_query_num, sample_num, feature_query_num_pre1 = len(gene_query_vec), len(sample_id), len(motif_query_vec)
# 		print('gene_query_vec_1, sample_id, feature_query_vec_pre1 ', gene_query_num, sample_num, feature_query_num_pre1)

# 		# self.meta_scaled_exprs = self.meta_scaled_exprs_2
# 		# x_expr = self.meta_scaled_exprs.loc[sample_id,motif_query_vec]
# 		# x_expr = meta_exprs.loc[sample_id,feature_query_vec]
# 		load_mode_idvec = 0
# 		train_idvec_1 = []

# 		# scale_type_id2 = 1    # log transformation and scaling of peak access-tf expr product
# 		# scale_type_id2 = 0    # log transformation without scaling of peak access-tf expr product
# 		feature_type_id = 1
# 		if feature_type_id<=1:
# 			# tf_score_mtx = list_1[0]
# 			x = 0
# 		else:
# 			# tf_score_mtx = list_1[1]
# 			x = 1

# 		fields = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2']
# 		# field_1 = ['mse','pearsonr','pval1','explained_variance','mean_absolute_error','median_absolute_error','r2_score','spearmanr','pval2']
# 		# score_list1 = []

# 		## commented
# 		# model_type_vec = ['LR','XGBR','Lasso']
# 		# # model_type_idvec = [1,0]
# 		# # model_type_idvec = [0]
# 		# model_type_idvec = [1]
# 		# # model_path_1 = self.save_path_1
# 		# model_path_1 = '%s/model_train'%(self.save_path_1)
# 		# run_id = 1
# 		# run_num, num_fold = 1, 10
# 		# max_interaction_depth, sel_num1, sel_num2 = 10, 2, -1
# 		# select_config.update({'run_id':run_id,'run_num':run_num,'max_interaction_depth':max_interaction_depth,
# 		#                       'sel_num1_interaction':sel_num1,'sel_num2_interaction':sel_num2})
		
# 		# df_score1 = self.training_regression_3(gene_query_vec=gene_query_vec,
# 		#                                       feature_query_vec=motif_query_vec,
# 		#                                       sample_id=sample_id,
# 		#                                       tf_score_mtx=tf_score_mtx,
# 		#                                       model_type_idvec=model_type_idvec,
# 		#                                       model_type_vec=model_type_vec,
# 		#                                       model_path_1=model_path_1,
# 		#                                       feature_type_id=feature_type_id,
# 		#                                       num_fold=num_fold,
# 		#                                       train_idvec=train_idvec_1,
# 		#                                       load_mode_idvec=load_mode_idvec,
# 		#                                       select_config=select_config)

# 		# # output_filename1 = '%s/test_motif_peak_estimate.train_%s.%d.%d.%d.1.txt'%(self.save_path_1,model_type_id1,num_fold,gene_query_num,feature_type_id)
# 		# # df_score1.to_csv(output_filename1,sep='\t',float_format='%.6E')

# 		# return

# 		dict_gene_motif_local_ = dict()
# 		self.dict_gene_param_id_ = dict()

# 		# param_id = self.dict_gene_param_id_[gene_query_id]
# 		# beta_local_ = beta_param_[param_id]
# 		# beta_param_ = beta
# 		print('peak_read ', self.peak_read.shape)

# 		# dict_peak_local_ = self.dict_peak_local_
# 		dict_peak_local_ = self.peak_dict
# 		print('dict_peak_local_ ', len(dict_peak_local_.keys()))

# 		## load gene-tf link prior (tf with motif)
# 		if motif_prior_type==0:
# 			# df_gene_motif_prior_ = self.gene_motif_prior_1.T.loc[:,motif_query_name_expr]
# 			# df_gene_motif_prior_ = self.gene_motif_prior_1.T.loc[:,motif_query_vec]
# 			gene_motif_prior_pre = self.gene_motif_prior_1
# 		else:
# 			# df_gene_motif_prior_ = self.gene_motif_prior_2.T.loc[:,motif_query_name_expr]
# 			gene_motif_prior_pre = self.gene_motif_prior_2
		
# 		df_gene_motif_prior_ = gene_motif_prior_pre.loc[:,motif_query_vec]
# 		self.df_gene_motif_prior_ = df_gene_motif_prior_
# 		print('df_gene_motif_prior_ ', df_gene_motif_prior_.shape, df_gene_motif_prior_.columns)

# 		# return

# 		# t_value1 = np.sum(df_gene_motif_prior_,axis=0)
# 		# b1 = np.where(t_value1>0)[0]
# 		# feature_query_name_pre2 = df_gene_motif_prior_.columns
# 		# motif_query_name = motif_query_name_expr[b1]
# 		# feature_query_vec_pre2 = feature_query_name_pre2[b1]
# 		feature_query_vec_pre2 = utility_1.test_columns_nonzero_1(df_gene_motif_prior_,type_id=1)
# 		print('feature_query_vec_pre2 ', len(feature_query_vec_pre2))

# 		feature_query_vec_ori = feature_query_vec.copy()
# 		feature_query_vec_pre1 = pd.Index(feature_query_vec_pre1)
# 		feature_query_vec = feature_query_vec_pre1.intersection(feature_query_vec_pre2,sort=False)
# 		feature_query_idvec = np.asarray([feature_query_vec_pre1.get_loc(query_id) for query_id in feature_query_vec])
# 		print(feature_query_idvec[0:10])
# 		tf_score_mtx = tf_score_mtx[:,:,feature_query_idvec]
# 		print('tf_score_mtx ', tf_score_mtx.shape)

# 		motif_query_name = feature_query_vec
# 		motif_query_num = len(motif_query_name)
# 		print('motif_query_name ', len(motif_query_name))

# 		motif_query_vec = motif_query_name
# 		self.motif_query_vec = motif_query_vec

# 		# list1, list2 = [], []
# 		# start_id1, start_id2 = 0, 0
# 		# for i1 in range(gene_query_num):
# 		#   t_gene_query = gene_query_vec[i1]
# 		#   t_motif_prior = df_gene_motif_prior_.loc[t_gene_query,:]
# 		#   b1 = np.where(t_motif_prior>0)[0]
# 		#   t_feature_query = motif_query_name[b1]
# 		#   str1 = ','.join(list(t_feature_query))

# 		#   t_feature_query_num = len(t_feature_query)
# 		#   list1.append(t_feature_query_num)
# 		#   list2.append(str1)

# 		#   start_id2 = start_id1+t_feature_query_num
# 		#   self.dict_gene_param_id_[t_gene_query] = np.arange(start_id1,start_id2)
# 		#   print(t_gene_query,i1,start_id1,start_id2,t_feature_query_num)
# 		#   start_id1 = start_id2

# 		# # self.param_num = start_id2
# 		# # print('param_num ', self.param_num)

# 		# df_gene_motif_prior_str = pd.DataFrame(index=gene_query_vec,columns=['feature_num','feature_query'])
# 		# df_gene_motif_prior_str['feature_num'] = list1
# 		# df_gene_motif_prior_str['feature_query'] = list2

# 		# annot_pre1 = str(len(gene_query_vec))
# 		# output_filename = '%s/df_gene_motif_prior_str_%s.txt'%(self.save_path_1,annot_pre1)
# 		# df_gene_motif_prior_str.to_csv(output_filename,sep='\t')

# 		# print('df_gene_motif_prior_str ', df_gene_motif_prior_str.shape)

# 		## the parameter id of beta for each gene
# 		intercept_flag = True
# 		select_config.update({'intercept_flag':intercept_flag})
# 		dict_gene_param_id, param_num, beta_mtx_init = self.test_motif_peak_estimate_param_id(gene_query_vec=gene_query_vec,
# 																								feature_query_vec=feature_query_vec,
# 																								df_gene_motif_prior=df_gene_motif_prior_,
# 																								intercept=intercept_flag)

# 		self.dict_gene_param_id_ = dict_gene_param_id
# 		self.param_num = param_num
# 		self.beta_mtx_init_ = beta_mtx_init
# 		beta_mtx_id1 = np.asarray(beta_mtx_init>0)
# 		self.beta_mtx_id1, self.beta_mtx_id2 = beta_mtx_id1, (~beta_mtx_id1)
# 		self.beta_mtx = np.zeros(beta_mtx_init.shape,dtype=np.float32)
# 		# return

# 		print('param_num ', self.param_num)

# 		motif_group_query = self.df_motif_group_query.loc[feature_query_vec,'group']
# 		motif_group_idvec = np.unique(motif_group_query)
# 		motif_group_list = [feature_query_vec[motif_group_query==group_id] for group_id in motif_group_idvec]
# 		motif_group_vec_pre = pd.Series(index=motif_group_idvec,data=[len(query_id) for query_id in motif_group_list],dtype=np.float32)
# 		motif_group_vec = np.asarray(motif_group_vec_pre)

# 		feature_query_vec_1 = self.beta_mtx_init_.columns
# 		df1 = pd.Series(index=feature_query_vec_1,data=np.arange(len(feature_query_vec_1)))
# 		motif_group_list_pre = [np.asarray(df1.loc[query_id]) for query_id in motif_group_list] # the index of the motif query id in the beta matrix

# 		self.motif_group_list = motif_group_list
# 		self.motif_group_vec = motif_group_vec_pre
		
# 		print('motif_group_list ', len(motif_group_idvec), motif_group_list[0:2])
# 		motif_group_vec_sort = motif_group_vec_pre.sort_values(ascending=False)
# 		print(motif_group_vec_sort)
# 		print(np.sum(motif_group_vec_sort>10),np.sum(motif_group_vec_sort>2),np.sum(motif_group_vec_sort==2),np.sum(motif_group_vec_sort==1))

# 		# peak_read = self.peak_read
# 		sample_id = meta_exprs.index
# 		peak_read = peak_read.loc[sample_id,:]
# 		# type_id_regularize = 0
# 		type_id_regularize = 1

# 		gene_query_num = len(gene_query_vec)
# 		sample_num = len(sample_id)
# 		feature_query_num = len(feature_query_vec)

# 		score_type_vec1 = ['unnormalized','normalized']
# 		score_type_vec2 = ['tf_score','tf_score_product','tf_score_product_2']
# 		score_type_id = select_config['tf_score_type']
# 		score_type_id2 = select_config['tf_score_type_id2']
# 		score_type_1, score_type_2 = score_type_vec1[score_type_id], score_type_vec2[score_type_id2]
# 		print('tf score type 1, tf score type 2 ', score_type_1, score_type_2)

# 		# self.meta_scaled_exprs = self.meta_scaled_exprs_2
# 		gene_expr = meta_exprs.loc[sample_id,gene_query_vec]
# 		# y = meta_exprs.loc[sample_id,gene_query_vec]
# 		# x = meta_exprs.loc[sample_id,feature_query_vec]
# 		y1 = np.asarray(gene_expr)
# 		x1_pre = np.asarray(tf_score_mtx)
# 		x0_pre = np.ones((sample_num,gene_query_num,1),dtype=np.float32)    # the first dimension
# 		x1 = np.concatenate((x0_pre,x1_pre),axis=2)

# 		# ratio1 = 1.0/sample_num
# 		ratio1 = 1.0/(sample_num*gene_query_num)
# 		# motif_prior_type = 0

# 		# x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.33, random_state=1)
# 		test_size = 0.125
# 		sample_id1 = np.arange(sample_num)
# 		sample_id_train, sample_id_test, sample_id1_train, sample_id1_test = train_test_split(sample_id,sample_id1,test_size=test_size,random_state=1)
# 		x_train, x_test, y_train, y_test = x1[sample_id1_train,:], x1[sample_id1_test,:], y1[sample_id1_train,:], y1[sample_id1_test,:]
# 		print('train, test ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 		pre_data_dict_1 = {'tf_score_mtx':tf_score_mtx,'gene_expr':gene_expr,
# 							'sample_id_train':sample_id_train,'sample_id_test':sample_id_test,
# 							'sample_id':sample_id,'gene_query_vec':gene_query_vec,'feature_query_vec':feature_query_vec,
# 							'dict_gene_param_id':dict_gene_param_id,
# 							'df_gene_motif_prior':df_gene_motif_prior_,
# 							'beta_mtx_id1':beta_mtx_id1}
		
# 		filename_annot1_pre1 = '%s_%s'%(score_type_2,score_type_1)

# 		output_filename1 = 'test_pre_data_dict_1_%s.1.npy'%(filename_annot1_pre1)
# 		if os.path.exists(output_filename1)==False:
# 			np.save(output_filename1,pre_data_dict_1,allow_pickle=True)

# 		field_1 = ['param_lower_bound','param_upper_bound','regularize_lambda_vec','tol_pre1']
# 		field_2 = ['motif_group_list','motif_group_vec','ratio']
# 		field_pre1 = field_1+field_2

# 		param_lower_bound, param_upper_bound = -100, 100
# 		Lasso_eta_1, lambda_vec_2 = 1E-03, [1E-03,1E-03,1E-03]
# 		regularize_lambda_vec = [Lasso_eta_1, lambda_vec_2]
# 		tol_pre1 = 0.0005
# 		list1_pre = [param_lower_bound,param_upper_bound,regularize_lambda_vec,tol_pre1]
# 		default_value_ = dict(zip(field_1,list1_pre))
# 		list1, list_1 = [], []
# 		for t_field_query in field_1:
# 			if t_field_query in select_config:
# 				t_value1 = select_config[t_field_query]
# 				print(t_field_query,t_value1)
# 				list1.append(t_value1)
# 			else:
# 				print('the query %s not included in config'%(t_field_query))
# 				print('the default value ',default_value_[t_field_query])
# 				list1.append(default_value_[t_field_query])

# 		# list1 = [param_lower_bound,param_upper_bound,regularize_lambda_vec,motif_group_list_pre,motif_group_vec]
# 		ratio1 = 1.0/(sample_num*gene_query_num)
# 		# ratio1 = 1.0/sample_num

# 		list_1 = list1 + [motif_group_list_pre,motif_group_vec,ratio1]
# 		pre_data_dict = dict(zip(field_pre1,list_1))
# 		# print('pre_data_dict ',pre_data_dict)

# 		param_init_est_mode = 1
# 		if param_init_est_mode==0:
# 			beta_param_init = np.random.rand(self.param_num)
# 		else:
# 			model_type_vec = ['LR','Lasso','XGBClassifier','XGBR','RF']
# 			model_type_id = 'Lasso'
# 			model_type_id1 = 1
# 			intercept_flag = select_config['intercept_flag']
# 			print('intercept_flag ', intercept_flag)
# 			select_config.update({'model_type_id_init':model_type_id,'model_type_id1_init':model_type_id1})
# 			if model_type_id in [3,'Lasso']:
# 				Lasso_alpha = 1E-03
# 				Lasso_max_iteration = 5000
# 				tol = 0.0005
# 				select_config.update({'Lasso_alpha':Lasso_alpha,'Lasso_max_iteration':Lasso_max_iteration,'Lasso_tol':tol})

# 			# output_filename = 'test_motif_peak_estimate_param_init_score_1.txt'%(filename_annot1_pre1)
# 			output_filename = 'test_motif_peak_estimate_param_init_score_%s.1.txt'%(filename_annot1_pre1)
# 			beta_param_init_pre, df_param_init_score = self.test_motif_peak_estimate_param_init_1(x_train,y_train,x_valid=[],y_valid=[],sample_weight=[],
# 																								gene_query_vec=gene_query_vec,sample_id=[],feature_query_vec=feature_query_vec,
# 																								peak_read=[],meta_exprs=[],motif_prior_type=0,
# 																								output_filename=output_filename,save_mode=1,select_config=select_config)

# 			np.random.seed(0)
# 			beta_param_init_1 = np.random.rand(self.param_num)-0.5
# 			# ratio_1, ratio_2 = 0.85, 0.15
# 			# ratio_1, ratio_2 = 1.0, 0
# 			ratio_1, ratio_2 = 0.9, 0.1
# 			# ratio_1, ratio_2 = 0.75, 0.25
# 			if 'beta_param_init_ratio1' in select_config:
# 				ratio_1 = select_config['beta_param_init_ratio1']
# 			if 'beta_param_init_ratio2' in select_config:
# 				ratio_2 = select_config['beta_param_init_ratio2']
# 			print('beta param init ratio_1, ratio_2 ', ratio_1, ratio_2)

# 			beta_param_init = ratio_1*beta_param_init_pre + ratio_2*beta_param_init_1

# 		beta_mtx = self.beta_mtx
# 		quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9,0.95]
# 		t_vec1 = utility_1.test_stat_1(beta_param_init,quantile_vec=quantile_vec_1)
# 		print('beta_param_init ', beta_param_init.shape,t_vec1)

# 		H_mtx = self.H_mtx.loc[gene_query_vec,:]
# 		H_p = self.H_p.loc[feature_query_vec,:]
# 		self.H_mtx_1 = H_mtx
# 		self.H_p_1 = H_p
# 		print('H_mtx, H_p ',H_mtx.shape, H_p.shape)

# 		obj_value_pre1, obj_value_pre2 = 1000, 1000
# 		self.iter_cnt1 = 0
# 		self.scale_factor = 1.0*gene_query_num
# 		self.config.update({'obj_value_pre1':obj_value_pre1,'obj_value_pre2':obj_value_pre2})

# 		regularize_lambda_vec = pre_data_dict['regularize_lambda_vec']
# 		Lasso_eta_1, lambda_vec_2 = regularize_lambda_vec
# 		lambda_1 = lambda_vec_2[0]

# 		filename_annot1_1 = '%s_%s_%s_%s'%(str(ratio_1),str(ratio_2),str(Lasso_eta_1),str(lambda_1))
# 		# filename_annot1_1 = '%.2f_%.2f_%s_%s'%(ratio_1,ratio_2,str(Lasso_eta_1),str(lambda_1))
# 		filename_annot1 = '%s_%s'%(filename_annot1_1,filename_annot1_pre1)

# 		beta_param_est = []
# 		pre_data_dict_2 = {'beta_param_init_pre':beta_param_init_pre,'beta_param_init':beta_param_init,'beta_param_est':beta_param_est,
# 							'pre_data_dict':pre_data_dict,'select_config':select_config}
# 		output_filename2 = 'test_pre_data_dict_2_%s.1.npy'%(filename_annot1)
# 		np.save(output_filename2,pre_data_dict_2,allow_pickle=True)

# 		flag_est, param_est = self.test_motif_peak_estimate_optimize1_unit1(initial_guess=beta_param_init,
# 														x_train=x_train,y_train=y_train,
# 														beta_mtx=beta_mtx,
# 														pre_data_dict=pre_data_dict,
# 														type_id_regularize=type_id_regularize)

# 		if flag_est==True:
# 			self.param_est = param_est
# 			y_pred_test = self.test_motif_peak_estimate_pred_1(x_test,select_config=select_config)
# 			print('y_test, y_pred_test ', y_test.shape, y_pred_test.shape)
# 			score_list = self.test_score_pred_1(y_test, y_pred_test)

# 			field_1 = ['mse','pearsonr','pval1','explained_variance','mean_absolute_error','median_absolute_error','r2_score','spearmanr','pval2']
# 			score_pred = np.asarray(score_list)
# 			df1 = pd.DataFrame(index=gene_query_vec,columns=field_1,data=score_pred,dtype=np.float32)
# 			print('df1 ',df1)

# 			save_mode = 1
# 			if save_mode==1:
# 				# if output_filename=='':
# 				#   output_filename = 'test_motif_peak_estimate_param_est_pred_score_1.txt'
# 				# df1.to_csv(output_filename,sep='\t',float_format='%.6E')
# 				# print('df1 ',df1)

# 				output_filename = 'test_motif_peak_estimate_param_est_pred_score_%s_1.txt'%(filename_annot1)
# 				df1.to_csv(output_filename,sep='\t',float_format='%.6E')

# 				beta_param_est = param_est
# 				# obj_value_pre1, obj_value_pre2 = self.config['obj_value_pre1'], self.config['obj_value_pre2']
# 				# regularize_vec_pre1, regularize_vec_pre2 = self.config['regularize_vec_pre1'], self.config['regularize_vec_pre2']
# 				# regularize_pre1, regularize_pre2 = self.config['regularize_pre1'], self.config['regularize_pre2']
# 				config_pre = self.config
# 				# pre_data_dict_2.update({'beta_param_est':beta_param_est,'scale_factor':self.scale_factor,'iter_cnt1':self.iter_cnt1,
# 				#                       'obj_value_pre1':obj_value_pre1,'obj_value_pre2':obj_value_pre2,
# 				#                       'regularize_vec_pre1':regularize_vec_pre1,'regularize_vec_pre2':regularize_vec_pre2,
# 				#                       'regularize_pre1':regularize_pre1,'regularize_pre2':regularize_pre2})
# 				pre_data_dict_2.update({'beta_param_est':beta_param_est,'scale_factor':self.scale_factor,'iter_cnt1':self.iter_cnt1,
# 										'config_pre':config_pre,'y_test':y_test,'y_pred_test':y_pred_test})

# 				output_filename2 = 'test_pre_data_dict_2_%s.1.npy'%(filename_annot1)
# 				np.save(output_filename2,pre_data_dict_2,allow_pickle=True)

# 		# for i2 in range(5):
# 		#   beta_param = np.random.rand(self.param_num)
# 		#   squared_error_ = self.test_motif_peak_estimate_obj_constraint1(gene_query_vec=gene_query_vec,
# 		#                                                                   sample_id=sample_id,
# 		#                                                                   feature_query_vec=motif_query_vec,
# 		#                                                                   beta=beta_param,
# 		#                                                                   tf_score_mtx=tf_score_mtx,
# 		#                                                                   peak_read=peak_read,
# 		#                                                                   meta_exprs=meta_exprs,
# 		#                                                                   motif_group_list=motif_group_list_pre,
# 		#                                                                   motif_group_vec=motif_group_vec,
# 		#                                                                   type_id_regularize=type_id_regularize,
# 		#                                                                   select_config=select_config)

# 		#   print(squared_error_, i2)

# 		return True
	
# 	def test_motif_peak_gene_query_load(self,celltype_query_vec=[],thresh_fc=1,thresh_fc_celltype_num=8,thresh_fc_celltype_num_2=-1):

# 		celltype_vec = ['Pharynx','Thymus','Thyroid','Thyroid/Trachea','Trachea/Lung','Esophagus',
# 						'Stomach','Liver','Pancreas 1','Pancreas 2','Small int','Colon']
		
# 		celltype_vec_str = ['Pharynx','Thymus','Thyroid','Thyroid.Trachea','Trachea.Lung','Esophagus',
# 						'Stomach','Liver','Pancreas.1','Pancreas.2','Small.int','Colon']

# 		# celltype_query = 'Trachea.Lung'
# 		celltype_num = len(celltype_vec_str)

# 		self.celltype_vec = celltype_vec
# 		self.celltype_vec_str = celltype_vec_str

# 		if len(celltype_query_vec)==0:
# 			gene_query_vec = self.gene_highly_variable # the set of genes

# 		else:

# 			list1 = []
# 			compare_type = 'pos'
# 			tol = 1
# 			filename_annot1 = '%s.tol%d'%(compare_type,tol)
# 			# thresh_fc = 2.0
# 			# thresh_fc_celltype_num = 8
			
# 			thresh_fc_celltype_num_1 = 8
# 			thresh_fc_celltype_num_pre1 = thresh_fc_celltype_num_2

# 			input_file_path = '%s/meta_exprs_compare_1'%(self.save_path_1)
# 			tol = 0

# 			for celltype_query in celltype_query_vec:

# 				# input_filename = '%s/DESeq2_data_1/test_motif_estimate.DESeq2.%s.merge.1A.subset2.fc%.1f.thresh%d.txt'%(self.path_1,celltype_query,thresh_fc,thresh_fc_celltype_num)  
# 				# input_filename = '%s/DESeq2_data_1/test_motif_estimate.DESeq2.%s.merge.1A.subset2.fc%.1f.thresh%d.%s.txt'%(self.path_1,celltype_query,thresh_fc,thresh_fc_celltype_num,filename_annot1)
# 				# input_filename = '%s/DESeq2_data_1/test_motif_estimate.DESeq2.%s.merge.1A.subset2.fc%.1f.thresh%d.%s.txt'%(self.path_1,celltype_query,thresh_fc,thresh_fc_celltype_num_1,filename_annot1)
# 				# input_filename = '%s/test_meta_exprs.MAST.%s.merge.1A.thresh%d.txt'%(input_file_path,celltype_query,thresh_fc_celltype_num_1)
# 				# input_filename = '%s/test_meta_exprs.MAST.%s.merge.1A.subset2.fc1.fdr0.05.thresh%d.txt'%(input_file_path,celltype_query,thresh_fc_celltype_num_1)
# 				input_filename = '%s/compare1/test_meta_exprs.MAST.%s.merge.1A.subset2.fc1.0.fdr0.05.thresh%d.pos.tol%d.txt'%(input_file_path,celltype_query,thresh_fc_celltype_num_1,tol)

# 				# compare_type = 'neg'
# 				# input_filename = '%s/DESeq2_data_1/test_motif_estimate.DESeq2.%s.merge.1A.subset2.fc%.1f.thresh%d.%s.tol%d.txt'%(self.path_1,celltype_query,thresh_fc,thresh_fc_celltype_num,compare_type,tol)

# 				data_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
# 				gene_query_idvec_ori = data_query_1.index

# 				# label_1 = data_query_1['label_fc_1.0']
# 				# label_2 = data_query_1['label_fc_2.0']

# 				# if thresh_fc_celltype_num_pre1>-1:
# 				#   id1 = (label_1>=thresh_fc_celltype_num_pre1)|(label_2>=thresh_fc_celltype_num)
# 				# else:
# 				#   id1 = (label_2>=thresh_fc_celltype_num)

# 				thresh_fdr = 0.05
# 				# label_query = 'label_fc%s_fdr%s'%(str(thresh_fc),str(thresh_fdr))
# 				label_query = 'label_fc1.0_fdr0.05'
# 				label_1 = data_query_1[label_query]

# 				id1 = (label_1>=thresh_fc_celltype_num)

# 				gene_query_idvec = data_query_1.index[id1]
# 				# gene_query_num = len(gene_query_idvec)

# 				list1.extend(gene_query_idvec)
# 				print('cell type query ', celltype_query, len(gene_query_idvec))

# 			gene_query_vec = np.asarray(list1)

# 		return gene_query_vec


