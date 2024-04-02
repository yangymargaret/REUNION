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

import pyranges as pr
import warnings

# import palantir 
import phenograph

import sys
from tqdm.notebook import tqdm

import csv
import os
import os.path
import shutil
from optparse import OptionParser

import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.pipeline import make_pipeline

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA,SparsePCA,TruncatedSVD
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.decomposition import FastICA, NMF, MiniBatchDictionaryLearning
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler,MultiLabelBinarizer,LabelEncoder,OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import AffinityPropagation,SpectralClustering,AgglomerativeClustering,DBSCAN,OPTICS,cluster_optics_dbscan
from sklearn.cluster import MiniBatchKMeans,KMeans,MeanShift,estimate_bandwidth,Birch

from scipy import stats
from scipy.stats import multivariate_normal, skew, pearsonr, spearmanr
from scipy.stats import wilcoxon, mannwhitneyu, kstest, ks_2samp, chisquare, fisher_exact, chi2_contingency
from scipy.stats.contingency import expected_freq
from scipy.stats import gaussian_kde, zscore, poisson, multinomial, norm, rankdata
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, issparse, hstack, vstack
from scipy import signal
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import gc
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

import h5py
import json
import pickle

import itertools
from itertools import combinations


# pairwise distance metric: spearmanr
# from the notebook
def spearman_corr(x, y):
	return spearmanr(x, y)[0]

# pairwise distance metric: pearsonr
# from the notebook
def pearson_corr(x, y):
	return pearsonr(x, y)[0]

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
	# print(auc,aupr,precision,recall)
	# return accuracy, auc, aupr, precision, recall, F1
	return df_score_pred

## save as anndata
def test_save_anndata(data,sparse_format='csr',obs_names=None,var_names=None,dtype=np.float32,select_config={}):
	
	adata = sc.AnnData(data,dtype=dtype)
	if sparse_format!=None:
		adata.X = csr_matrix(adata.X)

	if obs_names!=None:
		adata.obs_names = obs_names
	if var_names!=None:
		adata.var_names = var_names
		
	return adata

# from SEACells: genescores.py
def pyranges_from_strings(pos_list):
	# Chromosome and positions
	chr = pos_list.str.split(':').str.get(0)
	start = pd.Series(pos_list.str.split(':').str.get(1)).str.split('-').str.get(0)
	end = pd.Series(pos_list.str.split(':').str.get(1)).str.split('-').str.get(1)
	
	# Create ranges
	gr = pr.PyRanges(chromosomes=chr, starts=start, ends=end)
	
	return gr

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

# gene-peak link query
# from SEACells: query peaks within specific distance of gene query
def test_gene_peak_query_pre1(gene_query, transcripts, span, peaks_pr):

	gene_transcripts = transcripts[transcripts.gene_name == gene_query]
	if len(gene_transcripts) == 0:
		return 0
	longest_transcript= gene_transcripts[np.arange(len(gene_transcripts)) == np.argmax(gene_transcripts.End - gene_transcripts.Start)]
	start = longest_transcript.Start.values[0] - span
	end = longest_transcript.End.values[0] + span
	
	# Gene span
	gene_pr = pr.from_dict({'Chromosome': [longest_transcript.Chromosome.values[0]],
				  'Start': [start],
				  'End': [end]})
	gene_peaks = peaks_pr.overlap(gene_pr)
	if len(gene_peaks) == 0:
		return 0

	gene_peaks_str = pyranges_to_strings(gene_peaks)

	return gene_peaks_str

def density_2d(x, y):
	"""return x and y and their density z, sorted by their density (smallest to largest)
	:param x:
	:param y:
	:return:
	"""
	xy = np.vstack([np.ravel(x), np.ravel(y)])
	z = gaussian_kde(xy)(xy)
	i = np.argsort(z)
	return np.ravel(x)[i], np.ravel(y)[i], z[i]

def score_2a(y, y_predicted):

	score1 = mean_squared_error(y, y_predicted)
	score2 = pearsonr(y, y_predicted)
	score3 = explained_variance_score(y, y_predicted)
	score4 = mean_absolute_error(y, y_predicted)
	score5 = median_absolute_error(y, y_predicted)
	score6 = r2_score(y, y_predicted)
	score7, pvalue = spearmanr(y,y_predicted)
	# vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6]
	vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6, score7, pvalue]

	return vec1

def score_2a_1(y,y_predicted,feature_name=''):

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
	if feature_name!='':
		df_score_pred.name = feature_name
		
	# return vec1
	return df_score_pred
	
## reset dataframe index
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

## copy columns of one dataframe to another dataframe
# input_filename_list: first filename: the file that query columns from the other files
#					   the other filenames: the other files that provide the column information
# df_list: the list of dataframe
# id_column: the columns that define the unique rows
# column_vec: the columns for query from each of the other files
def test_column_query_1(input_filename_list,id_column,column_vec,df_list=[],type_id_1=0,type_id_2=0,type_include=0,index_col_1=0,index_col_2=0,reset_index=True,select_config={}):

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

## copy columns of one dataframe to another
# the two dataframes have a shared column
# df2 has unique rownames
def test_column_query_2(df_list=[],id_column=[],query_idvec=[],column_vec_1=[],column_vec_2=[],type_id_1=0,reset_index=True,flag_unduplicate=0,verbose=0,select_config={}):

	df1, df2 = df_list[0:2]
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
		print('df1: ',df1.shape)
		print('df_query1: ',df_query1.shape)
		df1.loc[query_idvec,column_vec_2] = np.asarray(df2.loc[query_id1,column_vec_1])
		
	if reset_index==True:
		df1.index = query_id1_ori

	return df1

# query default parameter
def test_query_default_parameter_1(field_query=[],default_parameter=[],overwrite=False,select_config={}):

	# field_query1 = ['root_path_1','root_path_2','run_id','type_id_feature']
	# default_parameter = [file_path1,file_path1,run_id,type_id_feature]
	field_num = len(field_query)
	param_vec = default_parameter

	for i1 in range(field_num):
		field_id = field_query[i1]
		if (not (field_id in select_config)) or (overwrite==True):
			select_config.update({field_id:default_parameter[i1]})
		else:
			param_vec[i1] = select_config[field_id]

	return select_config, param_vec

## motif-peak estimate: tf accessibility score scaling
# tf accessibility score: cell_num by feature_query_num
def test_motif_peak_estimate_score_scale_1(score=[],feature_query_vec=[],with_mean=True,with_std=True,verbose=0,select_config={},scale_type_id=1):

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

			# for t_feature_query in tqdm(score.columns):
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
			# for t_feature_query in tqdm(score.columns):
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
			# for t_feature_query in tqdm(score.columns):
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

## combine different files
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

## combine different files
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

## load peak-gene link query
# peak-gene link for feature query vec and peak-gene link by distance
def test_query_feature_link_1(feature_query_vec=[],df_feature_link=[],input_filename='',column_id_query1='gene_id',column_id_query2='distance',thresh_distance=-1,type_id_query=0,verbose=0,select_config={}):
		
		if len(df_feature_link)==0:
			if os.path.exists(input_filename)==False:
				print('the file does not exist: %s'%(input_filename))
				return
			else:
				df_feature_link = pd.read_csv(input_filename,index_col=0,sep='\t')
				print('feature_link: ',df_feature_link.shape,input_filename)

		feature_link = df_feature_link
		column_id1 = column_id_query1
		if verbose>0:
			print('feature_link: ',df_feature_link.shape)
		if len(feature_query_vec)==0:
			feature_query_vec = pd.Index(feature_link[column_id1].unique())
		else:
			feature_query_vec_ori = feature_query_vec
			feature_vec_1 = feature_link[column_id1].unique()
			if type_id_query==2:
				feature_query_vec_2 = pd.Index(feature_query_vec).difference(feature_vec_1,sort=False)
			feature_query_vec = pd.Index(feature_query_vec_ori).intersection(feature_vec_1,sort=False)
			feature_num = len(feature_query_vec)
			feature_link['id_ori'] = feature_link.index.copy()
			feature_link.index = np.asarray(feature_link[column_id1])
			feature_link = feature_link.loc[feature_query_vec,:]
			feature_link.index = np.asarray(feature_link['id_ori'])
			t_columns = feature_link.columns.difference(['id_ori'],sort=False)
			feature_link = feature_link.loc[:,t_columns]

		column_id2 = column_id_query2
		if thresh_distance>0:
			feature_link = feature_link.loc[feature_link[column_id2].abs()<thresh_distance]

		if type_id_query==1:
			return feature_link, feature_query_vec
		elif type_id_query==2:
			return feature_link, feature_query_vec, feature_query_vec_2
		else:
			return feature_link

## retain feature link by peak group and gene group query
def test_query_feature_group_2(self,df_link_query,df_group,column_id_query1='',column_id_query2='',group_label=[],verbose=0,select_config={}):
		
	flag_query1=1
	if flag_query1>0:
		if column_id_query2!='':
			df_group = df_group[column_id_query2]

		feature_query_1 = df_link_query[column_id_query1].unique()
		feature_query_1 = pd.Index(feature_query_1)
		feature_idvec = df_group.index
		list1 = []
		for group_query in group_label:
			list1.extend(feature_idvec[df_group==group_query])

		feature_query_pre1 = pd.Index(list1)
		feature_query1 = feature_query_1.intersection(feature_query_pre1,sort=False)
		feature_query2 = feature_query_1.difference(feature_query_pre1,sort=False)

		df_link_query.index = np.asarray(df_link_query[column_id_query1])
		# df_link_query.loc[feature_query1,'label_query'] = 1
			
		df_link_query1 = df_link_query.loc[feature_query2,:]
		df_link_query2 = df_link_query.loc[feature_query1,:]

		if verbose>0:
			feature_num_1 = len(feature_query_pre1)
			feature_num1, feature_num2 = len(feature_query1), len(feature_query2)
			print('feature_query_pre1, feature_query1, feature_query2: ',feature_num_1,feature_num1,feature_num2)
			print('feature_link 1, feature_link 2: ',df_link_query1.shape,df_link_query2.shape)

		return df_link_query1, df_link_query2

## correlation and pvalue calculation 
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

	# corr_values = pd.Series(corr_values, index=pairs).unstack().round(float_precision)
	# pvalues = pd.Series(pvalues, index=pairs).unstack().round(float_precision)
	
	corr_values = pd.Series(corr_values, index=pairs).unstack()
	pvalues = pd.Series(pvalues, index=pairs).unstack()

	if float_precision>0:
		corr_values = corr_values.round(float_precision)
		pvalues = pvalues.round(float_precision)
	
	return corr_values, pvalues

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
	# print('pvals_corrected thresh ',pval_thresh1)

	return (id1, pvals_corrected, alpha_Sidak, alpha_Bonferroni), pval_thresh1

## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
# input: the gene query, the peak distance threshold
# output: peak accessibility-gene expr correlation (dataframe)
def test_peak_tf_correlation_query_1(motif_data=[],peak_query_vec=[],motif_query_vec=[],peak_read=[],rna_exprs=[],correlation_type='spearmanr',
										pval_correction=1,alpha=0.05,method_type_id_correction='fdr_bh',flag_load=0,field_load=[],parallel_mode=0,
										save_mode=1,input_file_path='',input_filename_list=[],output_file_path='',
										filename_prefix='',verbose=0,select_config={}):

		if filename_prefix=='':
			filename_prefix = 'test_peak_tf_correlation'
		if flag_load>0:
			if len(field_load)==0:
				field_load = [correlation_type,'pval','pval_corrected']
			field_num = len(field_load)

			file_num = len(input_filename_list)
			list_query = []
			if file_num==0:
				input_filename_list = ['%s/%s.%s.1.txt'%(input_file_path,filename_prefix,filename_annot) for filename_annot in field_load]

			# list_query = [pd.read_csv(input_filename,sep='\t') for input_filename in input_filename_list]
			dict_query = dict()
			for i1 in range(field_num):
				filename_annot1 = field_load[i1]
				input_filename = input_filename_list[i1]
				# if file_num>0:
				# 	input_filename = input_filename_list[i1]
				# else:
				# 	input_filename = '%s/%s.%s.1.txt'%(input_file_path,filename_prefix,filename_annot1)
				if os.path.exists(input_filename)==True:
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
					field_query1 = filename_annot1
					dict_query.update({field_query1:df_query})
					print('df_query ',df_query.shape,filename_annot1)
				else:
					print('the file does not exist: %s'%(input_filename))
					flag_load = 0
				
				# list_query.append(df_query)
			# dict_query = dict(zip(field_load,list_query))
			if len(dict_query)==field_num:
				return dict_query
			# else:
			# 	flag_load = 0

		# else:
		if flag_load==0:
			print('peak accessibility-TF expr correlation estimation ')
			start = time.time()
			df_peak_tf_corr_, df_peak_tf_pval_, df_peak_tf_pval_corrected, df_motif_basic = test_peak_tf_correlation_1(motif_data=motif_data,
																														peak_query_vec=peak_query_vec,
																														motif_query_vec=motif_query_vec,
																														peak_read=peak_read,
																														rna_exprs=rna_exprs,
																														correlation_type=correlation_type,
																														pval_correction=pval_correction,
																														alpha=alpha,
																														method_type_id_correction=method_type_id_correction,
																														parallel_mode=parallel_mode,
																														select_config=select_config)

			field_query = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected','motif_basic']
			filename_annot_vec = [correlation_type,'pval','pval_corrected','motif_basic']
			list_query1 = [df_peak_tf_corr_, df_peak_tf_pval_, df_peak_tf_pval_corrected, df_motif_basic]
			# dict_query = dict(zip(field_query,list_query1))
			dict_query = dict(zip(filename_annot_vec,list_query1))
			query_num1 = len(list_query1)
			stop = time.time()
			print('peak accessibility-TF expr correlation estimation used: %.5fs'%(stop-start))
			# if filename_prefix=='':
			# 	# filename_prefix_1 = 'test_peak_tf_correlation'
			# 	filename_prefix = 'test_peak_tf_correlation'
			# filename_annot_vec = [correlation_type,'pval','pval_corrected','motif_basic']
			
			flag_save_text = 1
			if 'flag_save_text_peak_tf' in select_config:
				flag_save_text = select_config['flag_save_text_peak_tf']
			
			if save_mode>0:
				# input_file_path2 = '%s/peak_local'%(input_file_path)
				# output_file_path = input_file_path2
				if output_file_path=='':
					output_file_path = select_config['data_path']
				if flag_save_text>0:
					for i1 in range(query_num1):
						df_query = list_query1[i1]
						if len(df_query)>0:
							filename_annot1 = filename_annot_vec[i1]
							output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix,filename_annot1)
							# output_filename = '%s/%s.%s.1.copy1.txt'%(output_file_path,filename_prefix,filename_annot1)
							if i1 in [3]:
								df_query.to_csv(output_filename,sep='\t',float_format='%.6f')
							else:
								df_query.to_csv(output_filename,sep='\t',float_format='%.5E')
							print('df_query ',df_query.shape,filename_annot1)

		# return list_query1
		return dict_query

## peak accessibility-TF expression correlation
def test_peak_tf_correlation_1(motif_data,peak_query_vec=[],motif_query_vec=[],
								peak_read=[],rna_exprs=[],correlation_type='spearmanr',pval_correction=1,
								alpha=0.05,method_type_id_correction = 'fdr_bh',parallel_mode=0,verbose=1,select_config={}):

		if len(motif_query_vec)==0:
			motif_query_name_ori = motif_data.columns
			motif_query_name_expr = motif_query_name_ori.intersection(rna_exprs.columns,sort=False)
			print('motif_query_name_ori, motif_query_name_expr ',len(motif_query_name_ori),len(motif_query_name_expr))
			motif_query_vec = motif_query_name_expr
		else:
			motif_query_vec_1 = motif_query_vec
			motif_query_vec = pd.Index(motif_query_vec).intersection(rna_exprs.columns,sort=False)
		
		motif_query_num = len(motif_query_vec)
		# print('motif_query_vec ',motif_query_num)
		print('TF number: %d'%(motif_query_num))
		peak_loc_ori_1 = motif_data.index
		if len(peak_query_vec)>0:
			peak_query_1 = pd.Index(peak_query_vec).intersection(peak_loc_ori_1,sort=False)
			motif_data_query = motif_data.loc[peak_query_1,:]
		else:
			motif_data_query = motif_data

		peak_loc_ori = motif_data_query.index
		feature_query_vec_1, feature_query_vec_2 = peak_loc_ori, motif_query_vec
		# df_corr_ = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
		# df_pval_ = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
		flag_pval_correction = pval_correction
		# if flag_pval_correction>0:
		# 	df_pval_corrected = df_pval_.copy()
		# else:
		# 	df_pval_corrected = []
		
		# df_motif_basic = pd.DataFrame(index=feature_query_vec_2,columns=['peak_num','corr_max','corr_min'])

		# correlation_type = 'spearmanr'
		# flag_pval_correction = 1
		# flag_pval_correction = pval_correction
		# alpha = 0.05
		# method_type_id_correction = 'fdr_bh'
		# parallel_mode=0
		if parallel_mode==0:
			t_vec_1 = test_peak_tf_correlation_unit1(motif_data=motif_data_query,peak_query_vec=[],motif_query_vec=motif_query_vec,
															peak_read=peak_read,rna_exprs=rna_exprs,correlation_type=correlation_type,pval_correction=pval_correction,
															alpha=alpha,method_type_id_correction=method_type_id_correction,parallel_mode=0,verbose=1,select_config=select_config)
			df_corr_, df_pval_, df_pval_corrected, df_motif_basic = t_vec_1		
			
			# for i1 in range(motif_query_num):
			# 	motif_id = motif_query_vec[i1]
			# 	peak_loc_query = peak_loc_ori[motif_data_query.loc[:,motif_id]>0]

			# 	df_feature_query1 = peak_read.loc[:,peak_loc_query]
			# 	df_feature_query2 = rna_exprs.loc[:,[motif_id]]
			# 	df_corr_1, df_pval_1 = test_correlation_pvalues_pair(df_feature_query1,df_feature_query2,correlation_type=correlation_type,float_precision=6)
			# 	# df_corr_.loc[peak_loc_query,[motif_id]] = np.asarray(df_corr_1)
			# 	# df_pval_.loc[peak_loc_query,[motif_id]] = np.asarray(df_pval_1)

			# 	df_corr_.loc[peak_loc_query,motif_id] = df_corr_1.loc[peak_loc_query,motif_id]
			# 	df_pval_.loc[peak_loc_query,motif_id] = df_pval_1.loc[peak_loc_query,motif_id]

			# 	corr_max, corr_min = df_corr_1.max().max(), df_corr_1.min().min()
			# 	peak_num = len(peak_loc_query)
			# 	df_motif_basic.loc[motif_id] = [peak_num,corr_max,corr_min]
			# 	if verbose>0:
			# 		if i1%10==0:
			# 			print('motif_id: %s, id_query: %d, peak_num: %s, maximum peak accessibility-TF expr. correlation: %s, minimum correlation: %s'%(motif_id,i1,peak_num,corr_max,corr_min))
			# 	if flag_pval_correction>0:
			# 		pvals = np.asarray(df_pval_1.loc[peak_loc_query,motif_id])
			# 		pvals_correction_vec1, pval_thresh1 = test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_id_correction)
			# 		id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
			# 		df_pval_corrected.loc[peak_loc_query,motif_id] = pvals_corrected1
			# 		if (verbose>0) and (i1%100==0):
			# 			print('pvalue correction: alpha: %s, method_type: %s, minimum pval_corrected: %s, maximum pval_corrected: %s '%(alpha,method_type_id_correction,np.min(pvals_corrected1),np.max(pvals_corrected1)))
		else:
			dict_query_1 = dict()
			field_query = ['correlation','pval','pval_corrected','motif_basic']
			for field_id in field_query:
				dict_query_1[field_id] = []

			query_res_local = Parallel(n_jobs=-1)(delayed(test_peak_tf_correlation_unit1)(motif_data=motif_data_query,motif_query_vec=[motif_id_query],peak_read=peak_read,rna_exprs=rna_exprs,
																							correlation_type=correlation_type,pval_correction=pval_correction,alpha=alpha,method_type_id_correction=method_type_id_correction,verbose=verbose,select_config=select_config) for motif_id_query in motif_query_vec)
			
			for t_query_res in query_res_local:
				# dict_query = t_query_res
				if len(t_query_res)>0:
					# query_res.append(t_query_res)
					for (field_id, df_query) in zip(field_query,t_query_res):
						print(field_id,df_query.shape)
						print(df_query[0:2])
						dict_query_1[field_id].append(df_query)

			field_num1 = len(field_query)
			list_1 = []
			for i1 in range(field_num1):
				field_id = field_query[i1]
				list_query = dict_query_1[field_id]
				if i1<3:
					df_query = pd.concat(list_query,axis=1,join='outer',ignore_index=False)
				else:
					df_query = pd.concat(list_query,axis=0,join='outer',ignore_index=False)
				list_1.append(df_query)
			df_corr_, df_pval_, df_pval_corrected, df_motif_basic = list_1

		return df_corr_, df_pval_, df_pval_corrected, df_motif_basic

## peak accessibility-TF expression correlation
def test_peak_tf_correlation_unit1(motif_data,peak_query_vec=[],motif_query_vec=[],
									peak_read=[],rna_exprs=[],correlation_type='spearmanr',pval_correction=1,
									alpha=0.05,method_type_id_correction='fdr_bh',parallel_mode=0,verbose=1,select_config={}):
		
	motif_query_num = len(motif_query_vec)
	motif_data_query = motif_data
	peak_loc_ori = motif_data_query.index
	df_corr_ = pd.DataFrame(index=peak_loc_ori,columns=motif_query_vec)
	df_pval_ = pd.DataFrame(index=peak_loc_ori,columns=motif_query_vec)

	df_motif_basic = pd.DataFrame(index=motif_query_vec,columns=['peak_num','corr_max','corr_min'])
	flag_pval_correction = pval_correction

	if flag_pval_correction>0:
		df_pval_corrected = df_pval_.copy()
	else:
		df_pval_corrected = []

	for i1 in range(motif_query_num):
		motif_id = motif_query_vec[i1]
		peak_loc_query = peak_loc_ori[motif_data_query.loc[:,motif_id]>0]
		print('peak_loc_query: ',len(peak_loc_query))
		print(peak_loc_query[0:10])
		
		df_feature_query1 = peak_read.loc[:,peak_loc_query]
		df_feature_query2 = rna_exprs.loc[:,[motif_id]]
		df_corr_1, df_pval_1 = test_correlation_pvalues_pair(df_feature_query1,df_feature_query2,correlation_type=correlation_type,float_precision=6)
		# df_corr_.loc[peak_loc_query,[motif_id]] = np.asarray(df_corr_1)
		# df_pval_.loc[peak_loc_query,[motif_id]] = np.asarray(df_pval_1)

		df_corr_.loc[peak_loc_query,motif_id] = df_corr_1.loc[peak_loc_query,motif_id]
		df_pval_.loc[peak_loc_query,motif_id] = df_pval_1.loc[peak_loc_query,motif_id]

		corr_max, corr_min = df_corr_1.max().max(), df_corr_1.min().min()
		peak_num = len(peak_loc_query)
		df_motif_basic.loc[motif_id] = [peak_num,corr_max,corr_min]
		
		if verbose>0:
			if i1%10==0:
				print('motif_id: %s, id_query: %d, peak_num: %s, maximum peak accessibility-TF expr. correlation: %s, minimum correlation: %s'%(motif_id,i1,peak_num,corr_max,corr_min))
		
		if flag_pval_correction>0:
			pvals = np.asarray(df_pval_1.loc[peak_loc_query,motif_id])
			pvals_correction_vec1, pval_thresh1 = test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_id_correction)
			id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
			df_pval_corrected.loc[peak_loc_query,motif_id] = pvals_corrected1
			if (verbose>0) and (i1%100==0):
				print('pvalue correction: alpha: %s, method_type: %s, minimum pval_corrected: %s, maximum pval_corrected: %s '%(alpha,method_type_id_correction,np.min(pvals_corrected1),np.max(pvals_corrected1)))

	return (df_corr_, df_pval_, df_pval_corrected, df_motif_basic)

## query peak-gene link attributes
def test_gene_peak_query_attribute_1(df_gene_peak_query=[],df_gene_peak_query_ref=[],field_query=[],column_name=[],reset_index=True,select_config={}):

		print('df_gene_peak_query, df_gene_peak_query_ref ',df_gene_peak_query.shape,df_gene_peak_query_ref.shape)
		query_id1_ori = df_gene_peak_query.index.copy()
		df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=['peak_id','gene_id'])
		df_gene_peak_query_ref.index = test_query_index(df_gene_peak_query_ref,column_vec=['peak_id','gene_id'])
		query_id1 = df_gene_peak_query.index
		df_gene_peak_query.loc[:,field_query] = df_gene_peak_query_ref.loc[query_id1,field_query]
		if len(column_name)>0:
			df_gene_peak_query = df_gene_peak_query.rename(columns=dict(zip(field_query,column_name)))
		if reset_index==True:
			df_gene_peak_query.index = query_id1_ori # reset the index

		return df_gene_peak_query

## query the feature group
def test_feature_group_query_basic_1(df_query=[],field_query=[],query_vec=[],column_vec=[],type_id_1=0,select_config={}):

		if type_id_1==0:
			column_id1='count'
			# if 'count' in df_query:
			# 	column_id1='count1'
			df_query[column_id1] = 1
			df_query_group = df_query.loc[:,[column_id1,field_query]].groupby(by=field_query).sum()
		elif type_id_1==1:
			df_query_group = df_query.loc[:,query_vec+[field_query]].groupby(by=field_query).max()
		elif type_id_1==2:
			df_query_group = df_query.loc[:,query_vec+[field_query]].groupby(by=field_query).min()
		else:
			df_query_group = df_query.loc[:,query_vec+[field_query]].groupby(by=field_query).mean()

		query_id = df_query.index
		if type_id_1==0:
			if len(column_vec)==0:
				column_vec = ['group_count']
				column_id = column_vec[0]
			df_query[column_id] = df_query_group.loc[query_id,column_id1]
		else:
			if len(column_vec)==0:
				column_vec = query_vec
			df_query.loc[:,column_vec] = df_query_group.loc[query_id,query_vec]

		return df_query

## find the columns with non-zero values
def test_columns_nonzero_1(df,type_id=0):

	# t_value1 = np.sum(df,axis=0).abs()
	t_value1 = np.sum(df.abs(),axis=0)
	b1 = np.where(t_value1>0)[0]
	if type_id==0:
		vec1 = b1
	else:
		vec1 = df.columns[b1]

	return vec1

## the diagonal values of matrix
def test_diagonal_value(df,value):

	num1 = df.shape[0]
	rows, cols = df.index, df.columns
	for i1 in range(num1):
		df.loc[rows[i1],cols[i1]] = value[i1]

	return df

## query feature frequency
def test_query_frequency_1(query_vec,select_config={}):
		
	query_vec = np.asarray(query_vec)
	t_query_vec = np.unique(query_vec)
	df_query = pd.DataFrame(index=t_query_vec,columns=['count','ratio'])
	list1 = [np.sum(query_vec==query_value) for query_value in t_query_vec]
	df_query['count'] = np.asarray(list1)
	df_query['ratio'] = df_query['count']/np.sum(df_query['count'])

	return df_query

## query celltype frequencey for the metacells
def test_query_celltype_frequency_1(sample_id_query,df_annot=[],input_filename='',column_idvec_1=[],column_idvec_2=[],select_config={}):

	flag_query1=1
	if flag_query1>0:
		df_annot1 = df_annot
		if len(df_annot)==0:
			filename_1 = input_filename
			df_annot1 = pd.read_csv(filename_1,index_col=0,sep='\t')
		
		celltype_vec = select_config['celltype_vec']
		celltype_num = len(celltype_vec)
		celltype_idvec_1 = np.arange(celltype_num)
		dict_celltype_1 = dict(zip(celltype_vec,celltype_idvec_1))
		dict_celltype_2 = dict(zip(celltype_idvec_1,celltype_vec))
	
		sample_id = sample_id_query # metacell sample id query
		sample_num = len(sample_id)

		if len(column_idvec_1)==0:
			column_idvec_1 = ['celltype','celltype_frequency']
			column_idvec_pre1 = ['celltype_id','celltype_id_freq']
		else:
			column_idvec_pre1 = ['%s_id'%(column_query) for column_query in column_idvec_1]

		column_id_1, column_id1 = column_idvec_1
		column_id_2, column_id2 = column_idvec_pre1
		field_query = [column_id1,column_id2]
		
		column_vec = list(celltype_vec)+field_query
		df_ratio_query = pd.DataFrame(index=sample_id,columns=column_vec,data=0,dtype=np.float32)
		# df_label_query = pd.DataFrame(index=sample_id,columns=[column_id1,column_id2])
		if len(column_idvec_2)==0:
			column_idvec_2  =['Metacell','CellType']
		
		column_id_query1, column_id_query2 = column_idvec_2
		for i1 in range(sample_num):
			sample_id1 = sample_id[i1]
			df_query = df_annot1.loc[df_annot1[column_id_query1]==sample_id1]
			df_ratio = test_query_frequency_1(df_query[column_id_query2],select_config=select_config)
			query_name_vec = df_ratio.index
			df_ratio_query.loc[sample_id1,query_name_vec] = df_ratio['ratio']
			df_ratio_query.loc[sample_id1,column_id1] = df_ratio['ratio'].idxmax()
			df_ratio_query.loc[sample_id1,'count'] = df_ratio['count'].sum()

		df_ratio_query[column_id_1] = df_annot1.loc[sample_id,column_id_query2]
		query_num1 = len(column_idvec_1)
		for i1 in range(query_num1):
			column_query1, column_query2 = column_idvec_1[i1], column_idvec_pre1[i1]
			df_ratio_query[column_query2] = df_ratio_query[column_query1].map(dict_celltype_1)
		# df_ratio_query['label_comp'] = (df_ratio_query['celltype']!=df_ratio_query['celltype_frequency'])
		df_ratio_query['label_comp'] = (df_ratio_query[column_id_1]!=df_ratio_query[column_id1]).astype(int)

		return df_ratio_query

## group frequency query
def test_group_frequency_query(feature_id,group_id):

	group_vec = np.unique(group_id)
	group_num = len(group_vec)
	cnt_vec = np.zeros(group_num)
	for i1 in range(group_num):
		num1 = np.sum(group_id==group_vec[i1])
		cnt_vec[i1] = num1
	frequency_vec = cnt_vec/np.sum(cnt_vec)
	df_query_frequency = pd.DataFrame(index=group_vec,columns=frequency_vec)

	return df_query_frequency

## motif enrichment
# statistical tests for motif enrichment analysis
# def test_feature_enrichment_pre1(peak_loc_query,motif_query=[],peak_sel_bg=[],motif_data=[],motif_data_score=[],motif_data_score_quantile=[],select_config={}):
# feature1: group of feature query of feature type 1; feature2: group of feature query of feature ytpe 2
def test_feature_enrichment_pre1(feature1=[],feature2=[],feature2_bg=[],group_id='',df_link=[],df_link_score=[],df_link_score_2=[],flag_link_score=0,type_id_1=0,type_id_pval_correction=1,verbose=0,select_config={}):

		feature1_ori = df_link.columns
		# t_value1 = np.sum(motif_data.loc[peak_loc_query,:],axis=0)
		t_value1 = df_link.loc[feature2,:].sum(axis=0)
		feature1_pre1 = feature1_ori[t_value1>0]

		if len(feature1)==0:
			feature1_query = feature1_pre1
		else:
			feature1_query = pd.Index(feature1).intersection(feature1_pre1,sort=False)

		if verbose>0:
			print('feature1_query:%d'%(len(feature1_query)))

		# flag_link_score=1
		if len(df_link_score)==0:
			df_link_score = df_link
			flag_link_score = 0

		df_link_subset = df_link.loc[feature2,feature1_query] # links between feature2 and feature query of feature1
		df_link_bg =  df_link.loc[feature2_bg,feature1_query] # links between feature2_bg and feature query of feature1

		list1 = []
		thresh1 = 0
		field_query_1 = ['fc_1','fc_2','fc_3','stat_ks','pval_ks','stat_fisher_exact','pval_fisher_exact',
							'stat_chi2','pval_chi2','stat_barnard','pval_barnard','score1','score2']
		if flag_link_score>0:
			field_query_1 = field_query_1 + ['feature_score_max','feature_score_2_max']
		field_query_pre2 = ['pval_ks','pval_fisher_exact','pval_chi2']
		
		# field_query_pre_2 = ['pval_ks','pval_fisher_exact','pval_chi2']
		field_query_num2 = len(field_query_pre2)
		field_query_2 = ['%s.corrected'%(t_field_query) for t_field_query in field_query_pre2]

		field_query_pre2 = np.asarray(field_query_pre2)
		field_query_2= np.asarray(field_query_2)

		# df1 = pd.DataFrame(index=motif_query,columns=field_query_1,dtype=np.float32)
		df1 = pd.DataFrame(index=feature1_query,columns=field_query_1,dtype=np.float32)

		field_query_3_1 = ['num1','num2','num1_bg','num2_bg']
		field_query_3_2 = ['ratio1','ratio2','ratio1_1','ratio2_1','ratio1_2','ratio2_2']
		field_query_3 = field_query_3_1+field_query_3_2
		# df2 = pd.DataFrame(index=motif_query,columns=field_query_3,dtype=np.int32)
		df2 = pd.DataFrame(index=feature1_query,columns=field_query_3,dtype=np.int32)

		thresh1 = 0
		type_id_2 = 0
		if len(df_link_score_2)>0:
			type_id_2 = 1

		feature2_num = len(feature2)
		feature2_num_bg = len(feature2_bg)

		feature1_compare = df_link.columns.difference(feature1_query,sort=False)
		feature1_compare_num = len(feature1_compare)
		if verbose>0:
			print('feature1_compare: %d'%(feature1_compare_num))

		link_num_feature2 = (df_link.loc[feature2,:]).sum().sum() # the number of links between feature1 and feature2 foreground
		link_num_feature2_bg = (df_link.loc[feature2_bg,:]>thresh1).sum().sum() # the number of links between feature1 and feature2 background
		link_num_1 = link_num_feature2 + link_num_feature2_bg # the number of links between feature1 and feature2 

		feature_num1 = len(feature1_query)
		flag_barnard_ = 0
		if 'flag_barnard_' in select_config:
			flag_barnard_ = select_config['flag_barnard_']
		# for (i1, t_feature_query) in enumerate(feature1_query):
		for i1 in range(feature_num1):
			t_feature_query = feature1_query[i1]
			id_1, id_2 = (df_link.loc[feature2,t_feature_query]>thresh1), (df_link.loc[feature2_bg,t_feature_query]>thresh1)

			link_num_sel = np.sum(id_1)	# the number of links between feature1 foreground and feature2 foreground
			link_num_bg = np.sum(id_2)	# the number of links between feature1 foreground and feature2 background

			link_num_sel2 = link_num_feature2-link_num_sel	# links between feature1 background and feature2 foreground
			link_num_bg2 = link_num_feature2_bg-link_num_bg	# links between feature1 background and feature2 background

			link_num_feature1 = (link_num_sel+link_num_bg) # links between feature1 foreground and feature2 (foreground and background)
			link_num_feature1_bg = (link_num_1-link_num_feature1) # the number of links between feature1 background and feature2

			ratio1 = link_num_sel/link_num_feature2 # frequency in links between feature1 and feature2 foreground
			ratio2 = link_num_bg/link_num_feature2_bg  # frequency in links between feature1 and feature2 background

			ratio1_1 = link_num_sel/feature2_num # frequency in feature2 foreground
			ratio2_1 = link_num_bg/feature2_num_bg  # frequency in feature2 background

			ratio1_2 = link_num_sel/link_num_feature1 # frequency in links between feature1 foreground and feature2
			ratio2_2 = link_num_sel2/link_num_feature1_bg	# frequency in links between feature1 background and feature2

			eps = 1e-12
			fold_change1 = np.log2(ratio1/(ratio2+eps))
			fold_change2 = np.log2(ratio1_1/(ratio2_1+eps))
			fold_change3 = np.log2(ratio1_2/(ratio2_2+eps))
			# contingency_table = [[peak_num_motif_sel,peak_num_motif_bg],[peak_num_sel-peak_num_motif_sel,peak_num_bg1-peak_num_motif_bg]]
			# row1: the links between feature1 and feature2 (e.g., feature1: motif, feature2: peak), the links between feature 1 and background feature 2
			# row2: the links with feature2 but not with feature1, the links with feature2_bg but not with feature1
			# contingency_table = [[link_num_sel,link_num_bg],[feature2_num-link_num_sel,feature2_num_bg-link_num_bg]]
			# contingency_table = [[link_num_sel,link_num_bg],[link_num_feature2-link_num_sel,link_num_feature2_bg-link_num_bg]]
			contingency_table = [[link_num_sel,link_num_bg],[link_num_sel2,link_num_bg2]]
			contingency_table = np.asarray(contingency_table)
			contingency_table_ori = contingency_table.copy()
			if type_id_1==1:
				contingency_table = contingency_table.T
			
			t_score1 = -1
			t_score2 = -1
			if flag_link_score>0:
				# t_score1 = np.max(vec1)	# maximal feature score
				link_score_sel = df_link_score.loc[feature2,t_feature_query]
				link_score_bg = df_link_score.loc[feature2_bg,t_feature_query]
				t_score1 = np.max(link_score_sel) # maximal feature score
				if type_id_2>0:
					t_score2 = np.max(df_link_score_2.loc[feature2,t_feature_query]) # maximal feature score 2

			flag1 = 1
			if flag1>0:
				stat_ks_, pval_ks_ = -1, -1
				# stat_barnard_, pval_barnard_ = -1, -1
				stat_chi2_, pval_chi2_, dof_chi2_, ex_chi2_ = -1,-1,-1,-1
				stat_fisher_exact_, pval_fisher_exact_ = -1,-1
				stat_barnard_, pval_barnard_ = -1, -1
				stat_boschloo_, pval_boschloo_ = -1, -1
				try:
					if flag_link_score>0:
						# scipy.stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='auto')
						stat_ks_, pval_ks_ =  scipy.stats.ks_2samp(link_score_sel, link_score_bg, alternative='less', mode='auto')
						# print('KS test ',stat_ks_, pval_ks_)

						# stat_mannwhitenyu_, pval_mannwhitenyu_ = mannwhitneyu(vec1, vec2, alternative='greater', method="exact")
						# print('Mann Whiteny U test ',stat_mannwhitenyu_,pval_mannwhitenyu_)

					## commented
					# res = barnard_exact(contingency_table,alternative='greater')
					# stat_barnard_, pval_barnard_ = res.statistic, res.pvalue
					# print('Barnard exact test ',stat_barnard_, pval_barnard_)

					# res_2 = boschloo_exact(contingency_table,alternative='greater')
					# stat_boschloo_, pval_boschloo_ = res_2.statistic, res_2.pvalue
					# print('Boschloo exact test ',stat_boschloo_,pval_boschloo_)

					stat_chi2_, pval_chi2_, dof_chi2_, ex_chi2_ = chi2_contingency(contingency_table,correction=True)
					# print('chi2 ',stat_chi2_, pval_chi2_)

					stat_fisher_exact_, pval_fisher_exact_ = fisher_exact(contingency_table,alternative='greater')
					# print('Fisher exact test ',stat_fisher_exact_, pval_fisher_exact_)

					if flag_barnard_>0:
						thresh_pval_1 = 1E-05
						if (pval_fisher_exact_<thresh_pval_1) and (pval_chi2_<thresh_pval_1):
							res = barnard_exact(contingency_table,alternative='greater')
							stat_barnard_, pval_barnard_ = res.statistic, res.pvalue
							print('Barnard exact test ',stat_barnard_,pval_barnard_,t_feature_query,i1,group_id)

				except Exception as error:
					print('error! ', error)
					print(t_feature_query,i1)
					print(contingency_table)
					return

				t_vec1 = [fold_change1,fold_change2,fold_change3,stat_ks_,pval_ks_,stat_fisher_exact_,pval_fisher_exact_,stat_chi2_,pval_chi2_,
							stat_barnard_,pval_barnard_,t_score1,t_score2]
				# list1.append(t_vec1)
				df1.loc[t_feature_query,field_query_1] = t_vec1

				# if (i1%1000==0) or (pval_ks_<0.1):
				if (verbose>0)&((i1%1000==0) or (pval_fisher_exact_<0.01)):
					# print(t_motif_query,i1,fold_change1,pval_ks_,pval_barnard_,pval_fisher_exact_,pval_chi2_,t_score1,t_score2)
					# print(t_motif_query,i1,t_vec1)
					print(t_feature_query,i1,group_id,t_vec1)
					print(contingency_table)

				t_vec2 = np.ravel(np.asarray(contingency_table_ori).T)
				t_vec3 = [ratio1,ratio2,ratio1_1,ratio2_1,ratio1_2,ratio2_2]
				df2.loc[t_feature_query,field_query_3_1] = t_vec2
				df2.loc[t_feature_query,field_query_3_2] = t_vec3

		if flag_link_score==0:
			df1 = df1.loc[:,field_query_1[:-2]]
			sel_idvec = [1,2]
		else:
			sel_idvec = [0,1,2]

		# type_id_pval_correction = 0
		flag_pval_correction = type_id_pval_correction
		# sel_idvec = [0,2,3]
		if flag_pval_correction==1:
			## pvalue correction
			alpha = 0.05
			method_type_id_correction = 'fdr_bh'
			df_pval_corrected = pd.DataFrame(index=df1.index,columns=field_query_2[sel_idvec],dtype=np.float32)
			for i2 in sel_idvec:
				t_field_query_1 = field_query_pre2[i2]
				t_field_query_2 = field_query_2[i2]
				print('p-value correction ', t_field_query_1, t_field_query_2)
				pvals = df1.loc[:,t_field_query_1]
				id_pre1 = (pd.isna(pvals)|(pvals==-1)|(pvals==1))
				pvals = np.asarray(pvals)
				sel_id1 = np.where(id_pre1==True)[0]
				sel_id2 = np.where(id_pre1==False)[0]
				pvals[sel_id1] = 1.0

				pvals_correction_vec1, pval_thresh1 = test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_id_correction)
				id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
					
				# df_pval_corrected_list[i2].loc[gene_query_id,query_vec_1] = pvals_corrected1
				df_pval_corrected.loc[(id_pre1==False),t_field_query_2] = pvals_corrected1[sel_id2]

			df1 = pd.concat([df1,df_pval_corrected],axis=1,join='outer',ignore_index=False,keys=None,levels=None,names=None,verify_integrity=False,copy=True)
		
		df1['group_id'] = group_id
		df2['group_id'] = group_id

		return df1, df2

## query feature link
# use matrix format to represent feature link
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
		# field_id1 = column_link_score
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

## update field query
def test_field_query_pre1(field_query=[],query_value=[],overwrite=False,select_config={}):
	# query_num1 = len(query_value)
	for (field_id,query_value) in zip(field_query,query_value):
		if (not (field_id in select_config)) or (overwrite==True):
			select_config.update({field_id:query_value})

	return select_config

## convert the long format dataframe to wide format
def test_query_feature_format_1(df_feature_link=[],feature_query_vec=[],feature_type_vec=[],column_vec=[],column_value='',flag_unduplicate=1,format_type=0,save_mode=0,filename_prefix_save='',output_file_path='',output_filename='',verbose=0,select_config={}):

		df_link_query = df_feature_link
		# if len(feature_type_vec)==0:
		# 	feature_type_vec = ['gene','peak']

		if len(column_vec)==0:
			column_idvec = ['%s_id'%(feature_query) for feature_query in feature_type_vec]
		else:
			column_idvec = column_vec
		column_id1, column_id2 = column_idvec[0:2]

		if column_value!='':
			column_query = column_value
			# assert (column_value in df_link_query1.columns)
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
			print('df_feature_link: ',df_feature_link.shape)
			print('feature_mtx_1: ',feature_mtx_1.shape)
			print(df_feature_link[0:2])
			print(feature_mtx_1[0:2])

		feature_vec_1 = feature_mtx_1.index
		feature_vec_2 = feature_mtx_1.columns
		if format_type>0:
			feature_mtx_1 = feature_mtx_1.fillna(0)
			feature_mtx_1 = csr_matrix(feature_mtx_1)

		return df_link_query1, feature_mtx_1, feature_vec_1, feature_vec_2

## motif binding site shuffle
def test_query_feature_permuation_1(feature_mtx,feature_query_vec=[],type_id_1=0,verbose=0,select_config={}):

		# column_idvec = ['gene_id','peak_id','motif_id']
		# column_id1, column_id2, column_id3 = column_idvec[0:3]
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
			# print('feature_query: ',feature_query1,i1)
			# print('query_value: ',len(query_value),query_value[0:2])
			# print(len(t_vec_1),t_vec_1[0:2])
			feature_mtx_2[feature_query1] = query_value[t_vec_1].copy()

		return feature_mtx_2

## motif-peak estimate: select config field query
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

## feature interaction importance estimate from the learned model
# feature interaction importance estimate at specified interaction depth levels
# sel_num1: interaction depth
# sel_num2: the number of interactions to select
def test_model_explain_interaction_pre2(filename_dict,sel_num1=2,sel_num2=-1,save_mode=1):
		
	# model_type_idvec = list(filename_dict.keys())
	# model_type_num = len(model_type_idvec)
	query_idvec = list(filename_dict.keys())
	query_num = len(query_idvec)

	# print(filename_dict)
	flag = 1
	interaction_depth = sel_num1-1
	dict1, save_filename_dict1 = dict(), dict()

	for i1 in range(query_num):
		query_id = query_idvec[i1]
		input_filename1 = filename_dict[query_id]
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

			# print('sel_num2 ', t_sel_num2, input_filename1)
			for t_feature_name in feature_name_interaction[0:t_sel_num2]:
				str_vec1 = t_feature_name.split('|')
				str_vec1 = np.sort(str_vec1)
				list_1.append(str_vec1)

		# except:
		#   continue

		if len(list_1)>0:
			# list1 = list(np.unique(list1))
			print(len(list_1),list_1[0:5])

			idvec = np.asarray(['.'.join(t_vec1) for t_vec1 in list_1])
			mtx1 = np.asarray(list_1)
			interaction_sel_num, query_num = mtx1.shape[0], mtx1.shape[1] # the number of selected feature interactions and the number of features in the interaction
			t_columns = ['feature%d'%(query_id1+1) for query_id1 in range(query_num)]
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

# query quantile values of feature vector
def test_stat_1(vec_query,quantile_vec=[]):

	# vec_query = np.ravel(np.asarray(vec_query))
	vec_query_1 = np.ravel(np.asarray(vec_query))
	vec_query = pd.Series(index=range(len(vec_query_1)),data=vec_query_1)
	vec1 = [np.max(vec_query),np.min(vec_query),np.mean(vec_query),np.median(vec_query)]

	if len(quantile_vec)>0:
		vec1 = vec1 + list(np.quantile(vec_query,quantile_vec))

	return vec1

def plot_scores_pre3(df, meta_fdl, genes, n_cols=5, 
						vmin=-2, vmax=2, 
						plot_subset=None, s=30,
						output_filename = '',
						scale_type_id=1,
						scale_type_id_2=0,
						query_name_vec=[]):

	if plot_subset is None:
		plot_subset = df.index
	else:
		df_ori = df.copy()
		df = df.loc[plot_subset,:]
	
	# fig = palantir.plot.FigureGrid(len(genes), n_cols)
	# plt.rcParams.update({'axes.titlesize': 'large'})
	n_rows = int(np.ceil(len(genes)/n_cols))
	print(n_rows,n_cols)
	fig = plt.figure(figsize = (6*n_cols, 6*n_rows))
	i1 = 1
	query_num1 = len(genes)
	# for tf, ax in zip(genes, fig):
	# for (id1,tf) in enumerate(genes):
	for id1 in range(query_num1):
		tf = genes[id1]
		ax = fig.add_subplot(n_rows, n_cols, i1)
		i1 += 1
		# vmax = np.percentile(df[tf][plot_subset], 99)
		vmax = np.percentile(df[tf], 99)
		# vmax = np.max([vmax, -np.percentile(df[tf][plot_subset], 1)])
		vmax = np.max([vmax, -np.percentile(df[tf], 1)])

		min_value = np.min(df[tf])
		if min_value<0:
			vmin = -vmax
			cmap1 = matplotlib.cm.RdBu_r
		else:
			vmin = 0
			# cmap1 = matplotlib.cm.RdBu_r
			cmap1 = matplotlib.cm.Reds

		ax.scatter(meta_fdl['x'], 
				   meta_fdl['y'], 
				   s=s,
				   c=df[tf], vmin=vmin, vmax=vmax, 
				   cmap=cmap1, 
				   edgecolor='black', linewidths=0.25)
		ax.set_axis_off()
		feature_query1 = tf
		if len(query_name_vec)>0:
			feature_query = query_name_vec[id1]
		else:
			feature_query = tf
		# ax.set_title(tf)
		ax.set_title(feature_query)
		
		normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
		cax, _ = matplotlib.colorbar.make_axes(ax)
		# matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=matplotlib.cm.RdBu_r)
		matplotlib.colorbar.ColorbarBase(cax, norm=normalize, cmap=cmap1)

	if output_filename!='':
		# output_filename = 'test_1.png'
		plt.savefig(output_filename,format='png')

def impute_data(dm_res, ad, n_steps=3, n_jobs=-1):
	T_steps = dm_res['T'] ** n_steps
	T_steps = T_steps.astype(np.float32)

	# RUn in parallel
	seq = np.append(np.arange(0, ad.X.shape[1], 100), [ad.X.shape[1]])
	res = Parallel(n_jobs=n_jobs)(delayed(_dot_func)(T_steps, ad.X[:, seq[i - 1]:seq[i]]) for i in range(1, len(seq)))
	imputed_data = hstack(res)
	imputed_data = imputed_data.todense()
	imputed_data[imputed_data < 1e-2] = 0
	gc.collect()

	return imputed_data

# dimension reduction methods
def dimension_reduction(x_ori,feature_dim,type_id,shuffle=False,sub_sample=-1,filename_prefix='',filename_load='test1',save_mode=1,select_config={}):

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
		# elif type_id==11:
		# 	# t-SNE
		# 	tsne = TSNE(n_components=feature_dim,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
		# 	x = tsne.fit_transform(x_ori)
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
		# self.dimension_model = pickle.load(open(filename1, 'rb'))

	return x, dimension_model

## feature query dimension reduction
def test_feature_query_dimension_reduction(input_filename='',data_pre=[],transpose=False,save_mode=1,output_file_path='',filename_prefix='',type_id_1=1,type_id_2=0,select_config={}):

	file_path1 = '../data2'
	field_query = ['n_components','n_neighbors','zero_center','use_highly_variable']
	list1 = []
	for t_field_query in field_query:
		list1.append(select_config[t_field_query])

	n_components, n_neighbors_1, zero_center, use_highly_variable = list1

	if len(data_pre)==0:
		data_pre = pd.read_csv(input_filename,index_col=0,sep='\t')
	if transpose==True:
		data_pre = data_pre.T
	
	feature_query_id = data_pre.index
	feature_query_id_2 = data_pre.columns
	print('data_pre ', data_pre.shape, feature_query_id[0:2],feature_query_id_2[0:2])
	score_query_ad = sc.AnnData(data_pre)
	score_query_ad.X = csr_matrix(score_query_ad.X)
	pre_ad = score_query_ad
			
	sc.pp.pca(pre_ad,zero_center=zero_center,n_comps=n_components,use_highly_variable=use_highly_variable)
	
	df_pca = pre_ad.obsm['X_pca']
	print('X_pca ',df_pca.shape)
	filename_prefix_1 = '%s.pca%d'%(filename_prefix,n_components)
	output_filename_1 = '%s/%s.feature.1.txt'%(output_file_path,filename_prefix_1)
	feature_dim = n_components
	feature_mtx = pd.DataFrame(index=feature_query_id,columns=range(feature_dim),data=np.asarray(df_pca),dtype=np.float32)
	feature_mtx.to_csv(output_filename_1,sep='\t')

	flag_neighbor = type_id_1
	if flag_neighbor>0:
		sc.pp.neighbors(pre_ad,use_rep='X_pca',n_neighbors=n_neighbors_1)

	flag_connectivity = type_id_2
	list_query_1 = []
	if flag_connectivity>0:
		## scanpy.pp.neighbors(adata, n_neighbors=15, n_pcs=None, use_rep=None, knn=True, random_state=0, method='umap', metric='euclidean', metric_kwds=mappingproxy({}), key_added=None, copy=False)
		# neighbors
		# sc.pp.neighbors(pre_ad,use_rep='X_pca',n_neighbors=n_neighbors_1)
		method_type_id = 'umap'
		# method_type_id = 'gauss'
		method_type_vec = ['umap','gauss']
		n_neighbor_vec = [n_neighbors_1]

		for method_type_id in method_type_vec[0:1]:
			# for n_neighbors in [15,20,50]:
			for n_neighbors in n_neighbor_vec:
				sc.pp.neighbors(pre_ad,method=method_type_id,use_rep='X_pca',n_neighbors=n_neighbors)
				filename_prefix_2 = '%s.pca%d.%d.%s'%(filename_prefix,n_components,n_neighbors,method_type_id)
				
				if (save_mode==1) and (output_file_path!=''):
					output_filename = '%s/%s.1.ad'%(output_file_path,filename_prefix_2)
					pre_ad.write(output_filename)

					connectivity_mtx = pre_ad.obsp['connectivities']
					distance_mtx = pre_ad.obsp['distances']
					print('connectivities, distances ', connectivity_mtx.shape, distance_mtx.shape)
					df1 = pd.DataFrame(index=feature_query_id,columns=feature_query_id,data=distance_mtx.toarray(),dtype=np.float32)
					df2 = pd.DataFrame(index=feature_query_id,columns=feature_query_id,data=connectivity_mtx.toarray(),dtype=np.float32)
					b = output_filename.find('.txt')
					output_filename1 = '%s/%s.distance.txt'%(output_file_path,filename_prefix_2)
					output_filename2 = '%s/%s.connectivity.txt'%(output_file_path,filename_prefix_2)
					df1.to_csv(output_filename1,sep='\t')
					df2.to_csv(output_filename2,sep='\t')

				# ## scanpy.tl.umap(adata, min_dist=0.5, spread=1.0, n_components=2, maxiter=None, alpha=1.0, gamma=1.0, negative_sample_rate=5, init_pos='spectral', random_state=0, a=None, b=None, copy=False, method='umap', neighbors_key=None)
				# # UMAP
				# # obsm['X_umap']
				# sc.tl.umap(pre_ad)

				print(pre_ad)
				list_query_1.append(feature_mtx)
	else:
		if (save_mode==1) and (output_file_path!=''):
			filename_prefix_2 = '%s.pca%d.%d'%(filename_prefix,n_components,n_neighbors_1)
			output_filename = '%s/%s.1.ad'%(output_file_path,filename_prefix_2)
			pre_ad.write(output_filename)
		list_query_1.append(feature_mtx)

	return list_query_1

# motif similarity 
def test_motif_similarity(thresh_type_1='E-value',thresh_type_2='q-value',thresh1=0.001,thresh2=0.05,type_id_1=1):

	file_path_1 = '../example_datasets/data1'
	filename1 = '%s/tomtom_motif/motif_output_1/tomtom.tsv'%(file_path_1)
	filename2 = '%s/tomtom_motif/cisbp_motif_name.txt'%(file_path_1)
	filename3 = '%s/tomtom_motif/test_motif_name_cisbp.2.repeat1.txt'%(file_path_1)

	# data1 = pd.read_csv(filename1,sep='\t')
	data1 = pd.read_csv(filename1,sep='\t',skipfooter=4, engine='python')
	motif_query = np.asarray(data1['Query_ID'])
	data1.index = motif_query
	motif_compare = np.asarray(data1['Target_ID'])
	print(filename1,data1.shape)

	data2 = pd.read_csv(filename3,sep='\t')
	motif_id_query_ori_1 = np.asarray(data2['motif_name_ori'])
	motif_id_query = np.asarray(data2['motif_name'])
	motif_id_1 = np.asarray(data2['motif_id'])
	data2.index = list(motif_id_query)
	data2_1 = data2.copy()
	data2_1.index = motif_id_1

	motif_name_query = np.unique(motif_id_query)
	motif_num = len(motif_name_query)
	motif_num_ori = len(np.unique(motif_id_query_ori_1))
	print('motif num', motif_num, motif_num_ori)

	assert motif_num==motif_num_ori

	list1 = []
	list2 = []
	for i in range(motif_num):
		t_motif_id_query = motif_name_query[i]
		t_motif_id_1 = data2.loc[t_motif_id_query,['motif_id']]
		# print(t_motif_id_1)
		assert len(t_motif_id_1)==1
		t_motif_id_1 = list(t_motif_id_1)[0]
		t_motif_id_query_ori = list(data2.loc[t_motif_id_query,['motif_name_ori']])[0]

		# print(t_motif_id_1)
		t_motif_compare = data1.loc[t_motif_id_1,'Target_ID']
		t_motif_compare = np.asarray(t_motif_compare)
		t_motif_compare_evalue = data1.loc[t_motif_id_1,'E-value']
		t_motif_compare_qvalue = data1.loc[t_motif_id_1,'q-value']

		id1 = (t_motif_compare_evalue<thresh1)&(t_motif_compare_qvalue<thresh2)
		id_1 = np.where((id1>0)&(t_motif_compare!=t_motif_id_1))[0]

		num1 = len(id_1)
		if num1>0:
			t_motif_compare_1 = t_motif_compare[id_1]
			if i%100==0:
				print('motif query',t_motif_id_query,len(t_motif_compare),len(t_motif_compare_1))
			
			t_motif_compare_2 = pd.Series(index=t_motif_compare_1).index.intersection(data2_1.index)
			t_motif_name_compare_2 = data2_1.loc[t_motif_compare_2,'motif_name']
			num1 = len(t_motif_name_compare_2)
			assert num1==len(np.unique(t_motif_name_compare_2))
			str1 = ','.join(list(t_motif_name_compare_2))
			str2 = ','.join(t_motif_compare_2)

			if type_id_1==1:
				vec1 = [str(t1) for t1 in t_motif_compare_evalue[id_1]]
				vec2 = [str(t1) for t1 in t_motif_compare_qvalue[id_1]]
				str3_1 = ','.join(vec1)
				str3_2 = ','.join(vec2)
		else:
			str1, str2 = '-1', '-1',
			str3_1, str3_2 = '-1','-1'

		if type_id_1==1:
			list1.append([t_motif_id_query,t_motif_id_query_ori,t_motif_id_1,str1,str2,str3_1,str3_2])
		else:
			list1.append([t_motif_id_query,t_motif_id_query_ori,t_motif_id_1,str1,str2])
		list2.append(num1)

	t_fields = ['motif_name','motif_name_ori','motif_id','motif_num_compare','motif_name_compare',
					'motif_id_compare','motif_evalue_compare','motif_qvalue_compare']
	if type_id_1==1:
		data1 = pd.DataFrame(index=motif_name_query,columns=t_fields)
		data1.loc[:,t_fields[0:3]+t_fields[4:]] = np.asarray(list1)
		data1['motif_num_compare'] = list2
		output_filename = '%s/tomtom_motif/test_motif_compare_cisbp.%s.%s.2.txt'%(file_path_1,str(thresh1),str(thresh2))

	else:
		data1 = pd.DataFrame(index=motif_name_query,columns=['motif_name','motif_name_ori','motif_id','motif_num_compare','motif_name_compare','motif_id_compare'])
		data1.loc[:,t_fields[0:3]+t_fields[4:-2]] = np.asarray(list1)
		data1['motif_num_compare'] = list2
		output_filename = '%s/tomtom_motif/test_motif_compare_cisbp.%s.%s.1.txt'%(file_path_1,str(thresh1),str(thresh2))
	
	data1.to_csv(output_filename,sep='\t')

	return True

## minibatch K-means clustering
def test_feature_query_clustering_pre2(input_filename='',output_filename='',feature_mtx=[],similarity_mtx=[],feature_name=[],
						sel_id=[],symmetry_type_id=0,n_clusters=-1,connectivity_type_id=0,connectivity_thresh=0.5,distance_thresh=-1,
						linkage_type_id=1,alpha_connectivity=1.0,max_iter=500,transpose=False,dimension_reduction=False,save_mode=1,output_file_path='',select_config={}):
		
	if len(feature_mtx)==0:
		feature_mtx = pd.read_csv(input_filename,index_col=0,sep='\t')
		if transpose==True:
			feature_mtx = feature_mtx.T

	flag_dimension_reduction = dimension_reduction
	feature_mtx = feature_mtx.fillna(0)
	if flag_dimension_reduction>0:
		dimension_reduction_config = select_config['dimension_reduction_config']
		filename_prefix_1 = dimension_reduction_config['filename_prefix_1']
		data_list1 = test_feature_query_dimension_reduction(data_pre=feature_mtx,
															transpose=False,
															save_mode=save_mode,
															output_file_path=output_file_path,
															filename_prefix=filename_prefix_1,
															type_id_1=0,
															type_id_2=0,
															select_config=dimension_reduction_config)
		feature_mtx = data_list1[0]

	sample_id = feature_mtx.index
	# model_type_vec = ['MiniBatchKMeans']
	# sklearn.cluster.MiniBatchKMeans(n_clusters=8, *, init='k-means++', max_iter=100, batch_size=1024, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)[source]

	# model_type_vec = ['DBSCAN','OPTICS','AgglomerativeClustering','AffinityPropagation','SpectralClustering','cluster_optics_dbscan','MiniBatchKMeans']
	model_type_vec = ['DBSCAN','OPTICS','AgglomerativeClustering','AffinityPropagation','SpectralClustering','MiniBatchKMeans']
	model_type_vec = np.asarray(model_type_vec)

	# cluster_model1 = DBSCAN(eps=1,min_samples=1,n_jobs=-1,metric='precomputed')
	# cluster_model2 = OPTICS(min_samples=2,n_jobs=-1,metric='precomputed')
	cluster_model1 = DBSCAN(eps=18,min_samples=1,n_jobs=-1)
	cluster_model2 = OPTICS(min_samples=2,n_jobs=-1)

	if distance_thresh>0:
		distance_threshold = distance_thresh
	else:
		distance_threshold = None
	linkage_type_vec = ['ward','average','complete','single']
	linkage_type = linkage_type_vec[linkage_type_id]

	sample_num = len(sample_id)
	# n_clusters = np.int(np.min([sample_num*0.3,150]))
	# n_clusters = np.int(np.min([sample_num*0.1,50]))
	if n_clusters<0:
		n_clusters = np.int(np.min([sample_num*0.1,100]))
	print('n_clusters ', n_clusters)
	cluster_model3 = AgglomerativeClustering(n_clusters=n_clusters,distance_threshold=distance_threshold,linkage=linkage_type,
												compute_full_tree=True) # distance

	## affinity = [“euclidean”,“precomputed”]
	# max_iter = 200
	# max_iter = 500	# after using distance_thresh=10
	# cluster_model3_1 = AffinityPropagation(affinity='precomputed',max_iter=max_iter) # affinity
	cluster_model3_1 = AffinityPropagation(max_iter=max_iter) # affinity

	# cluster_model5 = SpectralClustering(n_clusters=n_clusters,affinity='precomputed')	# affinity
	cluster_model5 = SpectralClustering(n_clusters=n_clusters)	# affinity
	# cluster_model3_1 = SpectralClustering(n_clusters=100,affinity='precomputed_nearest_neighbors')	# distance
	# cluster_model6 = cluster_optics_dbscan()
	# list1 = [cluster_model1,cluster_model2,cluster_model3,cluster_model3_1,cluster_model5,cluster_model6]

	batch_size = 1280
	init_size = 5000
	n_init = 10
	cluster_model_1 = MiniBatchKMeans(n_clusters=n_clusters,max_iter=max_iter,batch_size=batch_size,init_size=init_size,n_init=n_init)
	# cluster_model_1.fix(feature_mtx)

	list1 = [cluster_model1,cluster_model2,cluster_model3,cluster_model3_1,cluster_model5,cluster_model_1]

	if len(sel_id)==0:
		# sel_id = [0,1,2,3]
		sel_id = [0]

	sel_id = np.asarray(sel_id)
	num1 = len(sel_id)
	data1 = pd.DataFrame(index=sample_id,columns=['feature_name']+list(model_type_vec[sel_id]))
	data1['feature_name'] = sample_id
	b = output_filename.find('.txt')
	filename_prefix_1 = output_filename[0:b]

	dict_query = dict()
	flag_1 = 0
	for model_id in sel_id:
		model_type_id = model_type_vec[model_id]
		cluster_model = list1[model_id]
		print(model_id, model_type_id)

		start = time.time()
		cluster_model.fit(feature_mtx)
		dict_query.update({model_type_id:cluster_model})
		t_labels = cluster_model.labels_
		data1.loc[:,model_type_id] = t_labels
		label_vec = np.unique(t_labels)
		label_num = len(label_vec)
		stop = time.time()
		print(model_type_id,label_num,stop-start)

		df1 = pd.DataFrame(index=label_vec,columns=['cluster','feature_num','feature'])
		if (save_mode==1) and (flag_1==1):
			for t_label_1 in label_vec:
				feature_query_id_1 = np.asarray(sample_id[t_labels==t_label_1])
				str1 = ','.join(feature_query_id_1)
				df1.loc[t_label_1,'cluster'] = t_label_1
				df1.loc[t_label_1,'feature_num'] = len(feature_query_id_1)
				df1.loc[t_label_1,'feature'] = str1
				output_filename_1 = '%s.%s.txt'%(filename_prefix_1,model_type_id)
				df1.to_csv(output_filename_1,index=False,sep='\t')

	annot_pre1 = '%s.%s.%s'%(str(distance_thresh),str(connectivity_thresh),str(linkage_type_id))
	if save_mode==1:
		data1.to_csv(output_filename,sep='\t')

	return data1, dict_query

def test_figure_configuration(fontsize=10,fontsize1=11,fontsize2=11):

	params = {'legend.fontsize': fontsize1,
			'font.size':fontsize,
			'figure.figsize':(6,5),
			'axes.labelsize': fontsize1,
			'axes.titlesize':fontsize2,
			'xtick.labelsize':fontsize,
			'ytick.labelsize':fontsize}
	
	pylab.rcParams.update(params)
	# plt.rcParams.update({'font.size': fontsize})
	# plt.rc('xtick', labelsize=fontsize)
	# plt.rc('ytick', labelsize=fontsize)

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="1", help="experiment id")
	parser.add_option("-b","--cell",default="1",help="cell type")

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()





