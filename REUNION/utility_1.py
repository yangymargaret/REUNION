#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

from copy import deepcopy

import pyranges as pr
import warnings

import sys

import os
import os.path
from optparse import OptionParser

import sklearn
from sklearn.utils import check_array, check_random_state

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, LabelEncoder, KBinsDiscretizer
from sklearn.pipeline import make_pipeline

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA,SparsePCA,TruncatedSVD
from sklearn.decomposition import FastICA, NMF, MiniBatchDictionaryLearning
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,accuracy_score,matthews_corrcoef
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics import pairwise_distances

from scipy import stats
from scipy.stats import multivariate_normal, skew, pearsonr, spearmanr
from scipy.stats import wilcoxon, mannwhitneyu, kstest, ks_2samp, chisquare, fisher_exact, chi2_contingency
from scipy.stats import gaussian_kde, zscore, poisson, multinomial, norm, rankdata
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, issparse
from scipy import signal
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

import h5py
import json
import pickle

import itertools
from itertools import combinations

## ====================================================
# pairwise distance metric: spearmanr
def spearman_corr(x, y):
	return spearmanr(x, y)[0]

## ====================================================
# pairwise distance metric: pearsonr
def pearson_corr(x, y):
	return pearsonr(x, y)[0]

## ====================================================
# save data as anndata
def test_save_anndata(data,sparse_format='csr',obs_names=None,var_names=None,dtype=np.float32,select_config={}):
	
	import scanpy as sc
	import anndata as ad
	from anndata import AnnData
	adata = sc.AnnData(data,dtype=dtype)
	if sparse_format!=None:
		adata.X = csr_matrix(adata.X)

	if obs_names!=None:
		adata.obs_names = obs_names
	if var_names!=None:
		adata.var_names = var_names
		
	return adata

## ====================================================
# from SEACells: genescores.py
def pyranges_from_strings(pos_list):
	# Chromosome and positions
	chr = pos_list.str.split(':').str.get(0)
	start = pd.Series(pos_list.str.split(':').str.get(1)).str.split('-').str.get(0)
	end = pd.Series(pos_list.str.split(':').str.get(1)).str.split('-').str.get(1)
	
	# Create ranges
	gr = pr.PyRanges(chromosomes=chr, starts=start, ends=end)
	
	return gr

## ====================================================
def pyranges_from_strings_1(pos_list,type_id=1):
	# Chromosome and positions
	chrom = pos_list.str.split(':').str.get(0)
	start = pd.Series(pos_list.str.split(':').str.get(1)).str.split('-').str.get(0)
	end = pd.Series(pos_list.str.split(':').str.get(1)).str.split('-').str.get(1)

	start = [int(i) for i in start]
	end = [int(i) for i in end]
	chrom, start, end = np.asarray(chrom), np.asarray(start), np.asarray(end)
	
	if type_id==1:
		# Create ranges
		gr = pr.PyRanges(chromosomes=chrom, starts=start, ends=end)
		return gr, chrom, start, end

	else:
		return chrom, start, end

## ====================================================
# reset dataframe index
def test_query_index(df_query,column_vec,symbol_vec=['.','.']):
	if len(symbol_vec)==0:
		symbol_vec = ['.','.']
	
	if len(column_vec)==3:
		column_id1,column_id2,column_id3 = column_vec[0:3]
		symbol1, symbol2 = symbol_vec
		query_id_1 = ['%s%s%s%s%s'%(query_id1,symbol1,query_id2,symbol2,query_id3) for (query_id1,query_id2,query_id3) in zip(df_query[column_id1],df_query[column_id2],df_query[column_id3])]
	elif len(column_vec)==2:
		column_id1,column_id2 = column_vec[0:2]
		symbol1 = symbol_vec[0]
		query_id_1 = ['%s%s%s'%(query_id1,symbol1,query_id2) for (query_id1,query_id2) in zip(df_query[column_id1],df_query[column_id2])]
	elif len(column_vec)>2:
		list1 = []
		query_idvec = df_query.index
		sample_num = df_query.shape[0]
		symbol_1 = symbol_vec[0]
		for i1 in range(sample_num):
			query_id1 = query_idvec[i1]
			query_vec = df_query.loc[query_id1,column_vec]
			query_vec = [str(query1) for query1 in query_vec]
			str1 = symbol_1.join(query_vec)
			list1.append(str1)
		query_id_1 = list1
	else:
		query_id_1 = column_vec[0]*df_query.shape[0]

	return query_id_1

## ====================================================
# copy specified columns from the other dataframes to the first dataframe
def test_column_query_1(input_filename_list,id_column,column_vec,df_list=[],type_id_1=0,type_id_2=0,type_include=0,index_col_1=0,index_col_2=0,reset_index=True,select_config={}):

	"""
	copy specified columns from the other dataframes to the first dataframe	
	:param input_filename_list: (list) paths of saved files; 
								the first file path: the file of dataframe that query columns from the other files;
					   			the other file paths: the other files of dataframes that provide the data of the corresponding columns;
	:param id_column: (array or list) the columns that define the unique rows in the dataframe;
	:param column_vec: (array or list) the columns to query from each of the other files
	:param df_list: list of dataframes; the first dataframe query columns from the other dataframes;
	:param select_config: dictionary containing parameters
	:return: (dataframe) the first dataframe with added columns of annotations copied from the other dataframes
	"""

	if (type_id_2==1) or (len(df_list)==0):
		# load dataframe from input_filename_list
		file_num = len(input_filename_list)
		# assert len(column_vec)==(file_num-1)
		input_filename_1 = input_filename_list[0]
		df_query1 = pd.read_csv(input_filename_1,index_col=index_col_1,sep='\t')
		type_id_query = 0
	else:
		# load dataframe from df_list
		file_num = len(df_list)
		df_query1 = df_list[0]
		type_id_query = 1

	assert len(column_vec)==(file_num-1)
	if type_id_1 in [0,3]:
		# reset the index of the first dataframe
		query_id_ori = df_query1.index.copy()
		df_query1.index = test_query_index(df_query1,column_vec=id_column)
	else:
		reset_index = False
	query_id1 = df_query1.index
	for i1 in range(1,file_num):
		if type_id_query==0:
			df_query2 = pd.read_csv(input_filename_list[i1],index_col=index_col_2,sep='\t')
		else:
			df_query2 = df_list[i1]

		if type_id_1 in [0,1]:
			df_query2.index = test_query_index(df_query2,column_vec=id_column)
		query_id2 = df_query2.index

		if type_include==0:
			query_id = query_id1.intersection(query_id2,sort=False)
		else:
			query_id = query_id1

		field_id = column_vec[i1-1]
		df_query1.loc[query_id,field_id] = df_query2.loc[query_id,field_id]

	if (reset_index==True):
		df_query1.index = query_id_ori

	return df_query1

## ====================================================
# copy column from the second dataframe to the first dataframe
# the two dataframes have a shared column
# the second dataframe has unique rownames
def test_column_query_2(df_list=[],id_column=[],query_idvec=[],column_vec_1=[],column_vec_2=[],type_id_1=0,reset_index=True,flag_unduplicate=0,verbose=0,select_config={}):

	df1, df2 = df_list[0:2]  # copy columns from df2 to df1
	column_id1 = id_column[0]
	if reset_index==True:
		# query_id1_ori, query_id2_ori = df1.index.copy(), df2.index.copy()
		query_id1_ori = df1.index.copy()

	if flag_unduplicate>0:
		df2 = df2.drop_duplicates(subset=[column_id1])

	df1.index = np.asarray(df1[column_id1])
	df2.index = np.asarray(df2[column_id1])
	if len(query_idvec)==0:
		query_idvec = df1.index
		df1.loc[:,column_vec_2] = np.asarray(df2.loc[query_idvec,column_vec_1])
	else:
		df_query1 = df1.loc[query_idvec,:]
		query_id1 = np.asarray(df_query1[column_id1])
		# print('df1: ',df1.shape)
		# print('df_query1: ',df_query1.shape)
		df1.loc[query_idvec,column_vec_2] = np.asarray(df2.loc[query_id1,column_vec_1])
		
	if reset_index==True:
		df1.index = query_id1_ori

	return df1

## ====================================================
# query default parameter
def test_query_default_parameter_1(field_query=[],default_parameter=[],overwrite=False,select_config={}):

	field_num = len(field_query)
	param_vec = default_parameter

	for i1 in range(field_num):
		field_id = field_query[i1]
		if (not (field_id in select_config)) or (overwrite==True):
			select_config.update({field_id:default_parameter[i1]})
		else:
			param_vec[i1] = select_config[field_id]

	return select_config, param_vec

## ====================================================
def score_function(y_test, y_pred, y_proba=[]):

	if len(y_proba)>0:
		auc = roc_auc_score(y_test,y_proba)
		aupr = average_precision_score(y_test,y_proba)
	else:
		auc, aupr = -1, -1
	precision = precision_score(y_test,y_pred)
	recall = recall_score(y_test,y_pred)
	accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))
	F1 = 2*precision*recall/(precision+recall)

	field_query_1 = ['accuracy','precision','recall','F1','auc','aupr']
	vec1 = [accuracy,precision,recall,F1,auc,aupr]
	df_score_pred = pd.Series(index=field_query_1,data=vec1,dtype=np.float32)
	return df_score_pred

## ====================================================
def score_function_multiclass1(y_test,y_pred,y_proba=[],average='binary',average_2='macro'):

	auc, aupr = 0, 0
	type_id_1 = 0
	if len(y_proba)>0:
		type_id_1 = 1
		try:
			auc = roc_auc_score(y_test,y_proba,average=average_2)
		except Exception as error:
			print('error! ',error)
			auc = 0
		try:
			aupr = average_precision_score(y_test,y_proba,average=average_2)
		except Exception as error:
			print('error!',error)
			aupr = 0
	
	precision = precision_score(y_test,y_pred,average=average)
	recall = recall_score(y_test,y_pred,average=average)
	accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))
	eps=1E-12
	F1 = 2*precision*recall/(precision+recall+eps)

	if type_id_1==0:
		vec1 = [accuracy, precision, recall, F1]
		field_query_1 = ['accuracy','precision','recall','F1']
	else:
		vec1 = [accuracy, precision, recall, F1, auc, aupr]
		field_query_1 = ['accuracy','precision','recall','F1','auc','aupr']

	df_score_pred = pd.Series(index=field_query_1,data=vec1,dtype=np.float32)

	return df_score_pred

## ====================================================
def score_function_multiclass2(y_test,y_pred,y_proba=[],average='macro',average_2='macro'):

	auc, aupr = 0, 0
	type_id_1 = 0
	if len(y_proba)>0:
		type_id_1 = 1
		try:
			auc = roc_auc_score(y_test,y_proba,average=average_2)
		except Exception as error:
			print('error! ',error)
			auc = 0
		try:
			aupr = average_precision_score(y_test,y_proba,average=average_2)
		except Exception as error:
			print('error!',error)
			aupr = 0
	
	precision = precision_score(y_test,y_pred,average=average)
	recall = recall_score(y_test,y_pred,average=average)
	accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))
	eps=1E-12
	F1 = 2*precision*recall/(precision+recall+eps)

	if type_id_1==0:
		vec1 = [accuracy, precision, recall, F1]
		field_query_1 = ['accuracy','precision','recall','F1']
	else:
		vec1 = [accuracy, precision, recall, F1, auc, aupr]
		field_query_1 = ['accuracy','precision','recall','F1','auc','aupr']

	df_score_pred = pd.Series(index=field_query_1,data=vec1,dtype=np.float32)

	return df_score_pred

## ====================================================
# feature scaling
def test_motif_peak_estimate_score_scale_1(score=[],feature_query_vec=[],with_mean=True,with_std=True,scale_type_id=1,verbose=0,select_config={}):

	thresh_upper_1, thresh_lower_1 = 0.99, 0.01
	thresh_upper_2, thresh_lower_2 = 0.995, 0.005
	thresh_2, thresh2 = 1E-05, 1E-05
	# print('scale_type_id ', scale_type_id)
	t_value1 = score.sum(axis=1)
	# print(score.shape,score[0:5],np.max(t_value1),np.min(t_value1),np.mean(t_value1),np.median(t_value1))
	quantile_vec_1 = [thresh_lower_2,thresh_lower_1,0.1,0.25,0.5,0.75,0.9,0.95,thresh_upper_1,thresh_upper_2]
		
	warnings.filterwarnings('ignore')
	t_columns = test_columns_nonzero_1(score,type_id=1)
	# print('columns with non-zero values: ', len(t_columns))
	if scale_type_id in [0,'minmax_scale']:
		# minmax normalization
		score_mtx = minmax_scale(score,[0,1])

	if scale_type_id in [6,'minmax_scale_1']:
		# minmax normalization
		score_1 = pd.DataFrame(index=score.index,columns=score.columns,data=0.0)

		for (i1,t_feature_query) in enumerate(t_columns):
			t_value1 = score[t_feature_query]
			# t_vec1 = test_stat_1(t_value1,quantile_vec=quantile_vec_1)
			# print(t_vec1, t_feature_query, i1)
			thresh1 = np.quantile(t_value1, thresh_upper_2)
			b1 = (t_value1>thresh1)
			t_value1[b1] = thresh1
			min_value = 0
			if np.min(t_value1)>thresh_2:
				min_value = thresh2
			t_score = minmax_scale(t_value1,[min_value, 1.0])
			score_1[t_feature_query] = t_score

	elif scale_type_id in [1,'minmax_scale_2']:
		score_1 = pd.DataFrame(index=score.index,columns=score.columns,data=0.0)
		for t_feature_query in t_columns:
			t_value1 = score[t_feature_query]
			min_value = 0
			if np.min(t_value1)>thresh_2:
				min_value = thresh2
			score_1[t_feature_query] = minmax_scale(t_value1, [min_value, np.quantile(score[t_feature_query], thresh_upper_2)])

	elif scale_type_id in [2,'scale']:
		# score_mtx = scale(score)
		score_mtx = scale(score,with_mean=with_mean,with_std=with_std,copy=True)

	elif scale_type_id in [3,'scale_2']:
		score_1 = pd.DataFrame(index=score.index,columns=score.columns,data=0.0)
		i1 = 0
		cnt = 0
		for (i1,t_feature_query) in enumerate(t_columns):
			# t_score = minmax_scale(score[t_feature_query],[0, np.percentile(score[t_feature_query], 99)])
			try:
				# t_score = minmax_scale(score[t_feature_query],[np.quantile(score[t_feature_query],thresh_lower_2), np.quantile(score[t_feature_query], thresh_upper_2)])
				t_value1 = score[t_feature_query]
				min_value_1 = 0
				if np.min(t_value1)>thresh_2:
					min_value_1 = np.quantile(t_value1,thresh_lower_1)
				max_value_1 = np.quantile(t_value1, thresh_upper_1)
				t_score = minmax_scale(t_value1,[min_value_1, max_value_1])
			except Exception as error:
				cnt = cnt+1
				print('error! ', error, t_feature_query, i1, cnt)
				t_vec1 = test_stat_1(score[t_feature_query],quantile_vec=quantile_vec_1)
				print(t_vec1, t_feature_query)
				t_score = score[t_feature_query]
			score_1[t_feature_query] = scale(t_score,with_mean=with_mean,with_std=with_std)
			if (verbose>0) and (i1%1000==0):
				print('feature_query: ',t_feature_query,i1)

	elif scale_type_id in [7,'scale']:
		score_mtx = scale(score,with_mean=False,with_std=True,copy=True)

	elif scale_type_id in [5,'quantile_transform']:
		# quantile normalization
		normalize_type = 'uniform'   # normalize_type: 'uniform', 'normal'
		if 'score_quantile_normalize_type' in select_config:
			normalize_type = select_config['score_quantile_normalize_type']
			
		# print('score normalize type', normalize_type)
		score_mtx = quantile_transform(score,n_quantiles=1000,output_distribution=normalize_type)
	
	else:
		score_1 = score

	if scale_type_id in [0,2,5,7,'minmax_scale','scale','quantile_transform']:
		score_1 = pd.DataFrame(index=score.index,columns=score.columns,data=np.asarray(score_mtx))

	warnings.filterwarnings('default')
		
	return score_1

## ====================================================
# combine different files
def test_file_merge_1(input_filename_list,input_file_path='',column_vec_query=[],index_col=0,header=0,axis_join=0,float_format=-1,flag_unduplicate=0,save_mode=0,output_filename='',verbose=0,select_config={}):

	iter_num = len(input_filename_list)
	df_query = []
	if iter_num==0:
		print('please provide input filename list')
		return df_query

	type_query=0
	if len(column_vec_query)>0:
		type_query=1

	list1=[]
	for iter_id in range(iter_num):
		input_filename = input_filename_list[iter_id]
		if input_file_path!='':
			input_filename = '%s/%s'%(input_file_path,input_filename)
		if os.path.exists(input_filename)==False:
			print('the file does not exist ',input_filename,iter_id)
			continue
		df_query1 = pd.read_csv(input_filename,index_col=index_col,header=header,sep='\t')
		if type_query>0:
			if iter_id==0:
				column_vec_1 = df_query1.columns
				column_vec_query1 = pd.Index(column_vec_query).intersection(column_vec_1,sort=False)
			df_query1 = df_query1.loc[:,column_vec_query1]
			
		if verbose>0:
			print('df_query1 ',input_filename,df_query1.shape,iter_id)
			print(df_query1[0:2])
		list1.append(df_query1)
			
	df_query = pd.concat(list1,axis=axis_join,join='outer',ignore_index=False)
	if flag_unduplicate>0:
		df_query = df_query.loc[(~df_query.index.duplicated(keep='first')),:]

	if (save_mode>0) and (output_filename!=''):
		if float_format!=-1:
			df_query.to_csv(output_filename,sep='\t',float_format=float_format)
		else:
			df_query.to_csv(output_filename,sep='\t')

	if verbose>0:
		print('df_query ',df_query.shape)
	return df_query

## ====================================================
# combine different files
def test_file_merge_column(input_filename_list,input_file_path='',column_idvec=[],index_col=0,header=0,float_format=-1,flag_unduplicate=0,save_mode=0,output_filename='',verbose=0,select_config={}):

	iter_num = len(input_filename_list)
	df_query = []
	if iter_num==0:
		print('please provide input filename list')
		return df_query

	list1 = []
	for i1 in range(iter_num):
		input_filename = input_filename_list[i1]
		df_query = pd.read_csv(input_filename,index_col=index_col,sep='\t')
		if index_col==False:
			df_query.index = test_query_index(df_query,column_vec=column_idvec)
		column_vec_1 = df_query.columns
		if i1==0:
			column_vec_pre1 = column_vec_1
		else:
			column_vec_2 = pd.Index(column_vec_1).difference(column_vec_pre1,sort=False) # the added columns
			# column_vec_1 = pd.Index(column_vec_pre1).union(column_vec_1,sort=False)
			df_query = df_query.loc[:,column_vec_2]
			list1.append(df_query)
	
	# df_link_query_1 = pd.concat(df_list1,axis=1,join='outer',ignore_index=False)
	df_query = pd.concat(list1,axis=1,join='outer',ignore_index=False)
	if (save_mode>0) and (output_filename!=''):
		if float_format!=-1:
			df_query.to_csv(output_filename,sep='\t',float_format=float_format)
		else:
			df_query.to_csv(output_filename,sep='\t')

	print('df_query ',df_query.shape)
	return df_query

## ====================================================
# correlation and pvalue calculation 
# from the website: https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/data/stats.html
def test_correlation_pvalues_pair(df1,df2,correlation_type='spearmanr',float_precision=7):
	
	# df1 = df1.select_dtypes(include=['number'])
	# df2 = df2.select_dtypes(include=['number'])
	pairs = pd.MultiIndex.from_product([df1.columns, df2.columns])
	if correlation_type=='pearsonr':
		t_list1 = [pearsonr(df1[a], df2[b]) for a, b in pairs]
	else:
		t_list1= [spearmanr(df1[a], df2[b]) for a, b in pairs]

	corr_values = [t_vec1[0] for t_vec1 in t_list1]
	pvalues = [t_vec1[1] for t_vec1 in t_list1]

	corr_values = pd.Series(corr_values, index=pairs).unstack()
	pvalues = pd.Series(pvalues, index=pairs).unstack()

	if float_precision>0:
		corr_values = corr_values.round(float_precision)
		pvalues = pvalues.round(float_precision)
	
	return corr_values, pvalues

## ====================================================
# peak-motif probability estimate, peak and motif estimate
# p-value correction
def test_pvalue_correction(pvals,alpha=0.05,method_type_id='fdr_bh'):

	pvals = np.asarray(pvals)
	t_correction_vec = multipletests(pvals,alpha=alpha,method=method_type_id,is_sorted=False,returnsorted=False)
	id1, pvals_corrected, alpha_Sidak, alpha_Bonferroni = t_correction_vec

	b1 = np.where(pvals_corrected<alpha)[0]
	if len(b1)>0:
		pval_1 = pvals[b1]
		pval_thresh1 = np.max(pval_1)
	else:
		pval_thresh1 = -1

	return (id1, pvals_corrected, alpha_Sidak, alpha_Bonferroni), pval_thresh1

## ====================================================
# query peak-gene link attributes
def test_gene_peak_query_attribute_1(df_gene_peak_query=[],df_gene_peak_query_ref=[],column_idvec=[],field_query=[],column_name=[],reset_index=True,select_config={}):

	query_id1_ori = df_gene_peak_query.index.copy()
	if len(column_idvec)==0:
		column_idvec = ['peak_id','gene_id']
	df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_idvec)
	df_gene_peak_query_ref.index = test_query_index(df_gene_peak_query_ref,column_vec=column_idvec)
	query_id1 = df_gene_peak_query.index
	df_gene_peak_query.loc[:,field_query] = df_gene_peak_query_ref.loc[query_id1,field_query]
	if len(column_name)>0:
		df_gene_peak_query = df_gene_peak_query.rename(columns=dict(zip(field_query,column_name)))
	if reset_index==True:
		df_gene_peak_query.index = query_id1_ori # reset the index

	return df_gene_peak_query

## ====================================================
# find the columns with non-zero values
def test_columns_nonzero_1(df,type_id=0):

	t_value1 = np.sum(df.abs(),axis=0)
	b1 = np.where(t_value1>0)[0]
	if type_id==0:
		vec1 = b1
	else:
		vec1 = df.columns[b1]

	return vec1

## ====================================================
# the diagonal values of matrix
def test_diagonal_value(df,value):

	num1 = df.shape[0]
	rows, cols = df.index, df.columns
	for i1 in range(num1):
		df.loc[rows[i1],cols[i1]] = value[i1]

	return df

## ====================================================
# query feature frequency
def test_query_frequency_1(query_vec,select_config={}):
		
	query_vec = np.asarray(query_vec)
	t_query_vec = np.unique(query_vec)
	df_query = pd.DataFrame(index=t_query_vec,columns=['count','ratio'])
	list1 = [np.sum(query_vec==query_value) for query_value in t_query_vec]
	df_query['count'] = np.asarray(list1)
	df_query['ratio'] = df_query['count']/np.sum(df_query['count'])

	return df_query

## ====================================================
# query the group label frequence in the group estimation
def test_group_frequency_query(feature_id,group_id,count_query=1):

	"""
	group frequency query
	:param feature_id: observations
	:param group_id: group assignment of observations
	:param count_query: indicator of whether to save the group size information
	:return: dataframe containing the group label frequence and group size information of group assignment
	"""
	
	group_vec = np.unique(group_id)
	group_num = len(group_vec)
	cnt_vec = np.zeros(group_num)
	for i1 in range(group_num):
		num1 = np.sum(group_id==group_vec[i1])
		cnt_vec[i1] = num1
	frequency_vec = cnt_vec/np.sum(cnt_vec)
	df_query_frequency = pd.DataFrame(index=group_vec,columns=frequency_vec)

	if count_query>0:
		df_query_frequency['count'] = cnt_vec

	return df_query_frequency

## ====================================================
# query feature links
# use the matrix format to represent feature links
def test_query_feature_link_format_1(data,column_idvec=[],flag_link_score=0,column_link_score='',flag_duplicate=0,sort_ascending=False,verbose=0,select_config={}):

	df_link_query = data
	column_id1_feature, column_id2_feature = column_idvec
	feature1_ori = df_link_query[column_id1_feature].unique()
	feature1_num_ori = len(feature1_ori)

	feature2_ori = df_link_query[column_id2_feature].unique()
	feature2_num_ori = len(feature2_ori)
	if verbose>0:
		print('feature1_ori: %d'%(feature1_num_ori))
		print('feature2_ori: %d'%(feature2_num_ori))

	if flag_duplicate>0:
		if (column_link_score!='') and (sort_ascending!=None):
			df_link_query = df_link_query.sort_values(by=[column_link_score],ascending=sort_ascending)
		df_link_query = df_link_query.drop_duplicates(subset=column_idvec,keep='first')
		if verbose>0:
			print('df_link_query unduplicated: ',df_link_query.shape)

	df_link = pd.DataFrame(index=feature2_ori,columns=feature1_ori,data=0,dtype=np.float32)
	if flag_link_score>0:
		df_link_score = df_link.copy()
	else:
		df_link_score = []
	
	for i1 in range(feature1_num_ori):
		feature_id1 = feature1_ori[i1]
		id1 = (df_link_query[column_id1_feature]==feature_id1)
		feature2_query = df_link_query.loc[id1,column_id2_feature].unique()
		feature2_num = len(feature2_query)
		if (verbose>0) and ((i1%100==0) or (feature2_num>100)):
			print('feature1_query: %s, feature2_query number: %d'%(feature_id1,feature2_num))
			
		df_link.loc[feature2_query,feature_id1] = 1
		if flag_link_score>0:
			df_link1 = df_link_query.loc[id1,:]
			df_link1.index = np.asarray(df_link1[column_id2_feature])
			df_link_score.loc[feature2_query,feature_id1] = np.asarray(df_link1.loc[feature2_query,column_link_score]) # for example: field_query='peak_tf_corr'; use peak_tf_corr as temporary feature link score

	return df_link, df_link_score

## ====================================================
# convert the long format dataframe to wide format
def test_query_feature_format_1(df_feature_link=[],feature_query_vec=[],feature_type_vec=[],column_vec=[],column_value='',flag_unduplicate=1,format_type=0,
								save_mode=0,filename_prefix_save='',output_file_path='',output_filename='',verbose=0,select_config={}):

	df_link_query = df_feature_link
	if len(column_vec)==0:
		column_idvec = ['%s_id'%(feature_query) for feature_query in feature_type_vec]
	else:
		column_idvec = column_vec
	column_id1, column_id2 = column_idvec[0:2]

	if column_value!='':
		column_query = column_value
		if not (column_query in df_link_query.columns):
			print('the column not included: %s'%(column_query))
			return 
	else:
		column_query = 'count'
		df_link_query[column_query] = 1

	df_link_query1 = df_link_query.loc[:,[column_id1,column_id2,column_query]]

	if flag_unduplicate>0:
		df_link_query1 = df_link_query1.drop_duplicates(subset=column_idvec)

	feature_mtx_1 = df_link_query1.pivot(index=column_id2,columns=column_id1,values=column_query)
	if verbose>0:
		print('convert the long format dataframe to wide format dataframe')
		print('feature link annotations, dataframe of size ',df_feature_link.shape)
		print('data preview:\n',df_feature_link[0:2])
		print('feature link matrix, dataframe of size ',feature_mtx_1.shape)
		print('datat preview:\n',feature_mtx_1[0:2])

	feature_vec_1 = feature_mtx_1.index
	feature_vec_2 = feature_mtx_1.columns
	if format_type>0:
		feature_mtx_1 = feature_mtx_1.fillna(0)
		feature_mtx_1 = csr_matrix(feature_mtx_1)

	return df_link_query1, feature_mtx_1, feature_vec_1, feature_vec_2

## ====================================================
# shuffle motif site
def test_query_feature_permuation_1(feature_mtx,feature_query_vec=[],type_id_1=0,verbose=0,select_config={}):

	feature_vec_1, feature_vec_2 = feature_mtx.index, feature_mtx.columns
	if len(feature_query_vec)==0:
		feature_query_vec = feature_vec_2

	feature_query_num = len(feature_query_vec)
	feature_num1 = len(feature_vec_1)
	feature_mtx_2 = pd.DataFrame(index=feature_vec_1,columns=feature_query_vec,data=0)
	np.random.seed(0)
	for i1 in range(feature_query_num):
		feature_query1 = feature_query_vec[i1]
		query_value = np.asarray(feature_mtx[feature_query1])
		t_vec_1 = np.random.permutation(feature_num1)
		feature_mtx_2[feature_query1] = query_value[t_vec_1].copy()

	return feature_mtx_2

## ====================================================
# save data to file
def test_save_file_1(output_filename,file,file_type,float_format=''):

	if os.path.exists(output_filename)==True:
		print('the file exist ', output_filename)
		filename1 = output_filename

		if file_type=='DataFrame':
			file_type1 = 'txt'
		else:
			file_type1 = file_type

		b = filename1.find('.%s'%(file_type1))
		if b>-1:
			output_filename_1 = filename1[0:b]+'.copy.%s'%(file_type1)
		else:
			output_filename_1 = filename1 + '.copy'

		import shutil
		shutil.copyfile(filename1,output_filename_1)

	if file_type=='txt':
		np.savetxt(output_filename,file,delimiter='\t')
	elif file_type=='DataFrame':
		if float_format!='':
			file.to_csv(output_filename,sep='\t',float_format=float_format)
		else:
			file.to_csv(output_filename,sep='\t')
	else:
		np.save(output_filename,file,allow_pickle=True)

	return True

## ====================================================
# query quantile values of feature vector
def test_stat_1(vec_query,quantile_vec=[]):

	# vec_query = np.ravel(np.asarray(vec_query))
	vec_query_1 = np.ravel(np.asarray(vec_query))
	vec_query = pd.Series(index=range(len(vec_query_1)),data=vec_query_1)
	vec1 = [np.max(vec_query),np.min(vec_query),np.mean(vec_query),np.median(vec_query)]

	if len(quantile_vec)>0:
		vec1 = vec1 + list(np.quantile(vec_query,quantile_vec))

	return vec1

## ====================================================
# dimension reduction methods
def dimension_reduction(x_ori,feature_dim,type_id,shuffle=False,sub_sample=-1,filename_prefix='',filename_load='test1',save_mode=1,verbose=0,select_config={}):

	# if shuffle==1 and sub_sample>0:
	# 	idx = np.random.permutation(x_ori.shape[0])
	# else:
	# 	idx = np.asarray(range(0,x_ori.shape[0]))
	idx = np.asarray(range(0,x_ori.shape[0]))
	if (sub_sample>0) and (type_id!=7) and (type_id!=11):
		x_ori = np.asarray(x_ori)
		id1 = idx[0:sub_sample]
	else:
		id1 = idx

	if type_id==0:
		# PCA
		pca = PCA(n_components=feature_dim, whiten = False, random_state = 0)
		if sub_sample>0:
			pca.fit(x_ori[id1,:])
			x = pca.transform(x_ori)
		else:
			x = pca.fit_transform(x_ori)
		dimension_model = pca
		# X_pca_reconst = pca.inverse_transform(x)
	elif type_id==1:
		# Incremental PCA
		n_batches = 10
		inc_pca = IncrementalPCA(n_components=feature_dim)
		for X_batch in np.array_split(x_ori, n_batches):
			inc_pca.partial_fit(X_batch)
		x = inc_pca.transform(x_ori)
		dimension_model = inc_pca
		# X_ipca_reconst = inc_pca.inverse_transform(x)
	elif type_id==2:
		# Kernel PCA
		kpca = KernelPCA(kernel="rbf",n_components=feature_dim, gamma=None, fit_inverse_transform=True, random_state = 0, n_jobs=50)
		kpca.fit(x_ori[id1,:])
		x = kpca.transform(x_ori)
		dimension_model = kpca
		# X_kpca_reconst = kpca.inverse_transform(x)
	elif type_id==3:
		# Sparse PCA
		sparsepca = SparsePCA(n_components=feature_dim, alpha=0.0001, random_state=0, n_jobs=50)
		sparsepca.fit(x_ori[id1,:])
		x = sparsepca.transform(x_ori)
		dimension_model = sparsepca
	elif type_id==4:
		# SVD
		SVD_ = TruncatedSVD(n_components=feature_dim,algorithm='randomized', random_state=0, n_iter=10)
		if sub_sample>0:
			SVD_.fit(x_ori[id1,:])
			x = SVD_.transform(x_ori)
		else:
			x = SVD_.fit_transform(x_ori)
		dimension_model = SVD_
		# X_svd_reconst = SVD_.inverse_transform(x)
	elif type_id==5:
		# Gaussian Random Projection
		GRP = GaussianRandomProjection(n_components=feature_dim,eps = 0.5, random_state=2019)
		GRP.fit(x_ori[id1,:])
		x = GRP.transform(x_ori)
		dimension_model = GRP
	elif type_id==6:
		# Sparse random projection
		SRP = SparseRandomProjection(n_components=feature_dim,density = 'auto', eps = 0.5, random_state=2019, dense_output = False)
		SRP.fit(x_ori[id1,:])
		x = SRP.transform(x_ori)
		dimension_model = SRP
	elif type_id==7:
		# MDS
		mds = MDS(n_components=feature_dim, n_init=12, max_iter=1200, metric=True, n_jobs=4, random_state=2019)
		x = mds.fit_transform(x_ori[id1])
		dimension_model = mds
	elif type_id==8:
		# ISOMAP
		isomap = Isomap(n_components=feature_dim, n_jobs = 4, n_neighbors = 5)
		isomap.fit(x_ori[id1,:])
		x = isomap.transform(x_ori)
		dimension_model = isomap
	elif type_id==9:
		# MiniBatch dictionary learning
		miniBatchDictLearning = MiniBatchDictionaryLearning(n_components=feature_dim,batch_size = 1000,alpha = 1,n_iter = 25,  random_state=2019)
		if sub_sample>0:
			miniBatchDictLearning.fit(x_ori[id1,:])
			x = miniBatchDictLearning.transform(x_ori)
		else:
			x = miniBatchDictLearning.fit_transform(x_ori)
		dimension_model = miniBatchDictLearning
	elif type_id==10:
		# ICA
		fast_ICA = FastICA(n_components=feature_dim, algorithm = 'parallel',whiten = True,max_iter = 100,  random_state=2019)
		if sub_sample>0:
			fast_ICA.fit(x_ori[id1])
			x = fast_ICA.transform(x_ori)
		else:
			x = fast_ICA.fit_transform(x_ori)
		dimension_model = fast_ICA
		# X_fica_reconst = FastICA.inverse_transform(x)
	elif type_id==12:
		# Locally linear embedding
		lle = LocallyLinearEmbedding(n_components=feature_dim, n_neighbors = np.max((int(feature_dim*1.5),500)),method = 'modified', n_jobs = 20,  random_state=2019)
		lle.fit(x_ori[id1,:])
		x = lle.transform(x_ori)
		dimension_model = lle
	elif type_id==13:
		# Autoencoder
		feature_dim_ori = x_ori.shape[1]
		m = Sequential()
		m.add(Dense(512,  activation='elu', input_shape=(feature_dim_ori,)))
		# m.add(Dense(256,  activation='elu'))
		m.add(Dense(feature_dim,   activation='linear', name="bottleneck"))
		# m.add(Dense(256,  activation='elu'))
		m.add(Dense(512,  activation='elu'))
		m.add(Dense(feature_dim_ori,  activation='sigmoid'))
		m.compile(loss='mean_squared_error', optimizer = Adam())
		history = m.fit(x_ori[id1], x_ori[id1], batch_size=256, epochs=20, verbose=1)

		encoder = Model(m.input, m.get_layer('bottleneck').output)
		x = encoder.predict(x_ori)
		Renc = m.predict(x_ori)
		dimension_model = encoder
	elif type_id==15:
		max_iter = 500
		from sklearn.decomposition import NMF
		# init{‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’}, default=None
		dimension_model = NMF(n_components=feature_dim, init=None, solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=max_iter, random_state=0, 
								alpha_W=0.0, alpha_H='same', l1_ratio=0.0, verbose=verbose, shuffle=False)

		if sub_sample>0:
			dimension_model.fit(x_ori[id1,:])
			x = dimension_model.transform(x_ori)
		else:
			x = dimension_model.fit_transform(x_ori)

	if save_mode>0:
		# save transfrom model
		method_type_query = type_id
		if filename_prefix=='':
			filename1 = '%s_%d_dimensions.%d.h5'%(filename_load,feature_dim,method_type_query)
		else:
			filename1 = '%s_%d_%d_dimensions.%d.h5'%(filename_prefix,type_id,feature_dim,method_type_query)
			
		# np.save(filename1, self.dimension_model)
		pickle.dump(dimension_model, open(filename1, 'wb'))

	return x, dimension_model

## ====================================================
# figure parameter configuration
def test_figure_configuration(fontsize=10,fontsize1=11,fontsize2=11):

	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.pylab as pylab

	params = {'legend.fontsize': fontsize1,
			'font.size':fontsize,
			'figure.figsize':(6,5),
			'axes.labelsize': fontsize1,
			'axes.titlesize':fontsize2,
			'xtick.labelsize':fontsize,
			'ytick.labelsize':fontsize}
	
	pylab.rcParams.update(params)

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="1", help="experiment id")
	parser.add_option("-b","--cell",default="1",help="cell type")

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()





