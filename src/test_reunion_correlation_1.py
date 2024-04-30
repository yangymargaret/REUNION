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

import utility_1
from utility_1 import pyranges_from_strings, test_file_merge_1
from utility_1 import spearman_corr, pearson_corr
import h5py
import json
import pickle

import itertools
from itertools import combinations

# pairwise distance metric: spearmanr
# from the notebook
def spearman_corr(self, x, y):
	return spearmanr(x, y)[0]

# pairwise distance metric: pearsonr
# from the notebook
def pearson_corr(x, y):
	return pearsonr(x, y)[0]

class _Base2_correlation(BaseEstimator):
	"""Base class for compute correlation
	"""
	def __init__(self,file_path='',run_id=1,species_id=1,cell='ES', 
					generate=1,
					featureid=1,typeid=1,method=1,
					config={},
					select_config={}):

		# Initializes RepliSeq
		self.run_id = run_id
		self.cell = cell
		self.generate = generate

		# path_1 = '../example_datasets/data1'
		self.path_1 = file_path
		self.save_path_1 = file_path
		self.config = config

	## p-value correction
	def test_correlation_pvalue_correction_pre1(self,pvalues,alpha=0.05,method_type_correction='fdr_bh',filename_prefix='',correlation_type_vec=['spearmanr'],
													type_id_pval_correction=1,save_mode=1,output_file_path='',type_id_1=0,select_config={}):
		# flag1 = 0
		method_type_id_correction = method_type_correction	
		df_pval_ = pvalues
		feature_query_vec_1 = df_pval_.index
		feature_query_vec_2 = df_pval_.columns

		feature_query_num = len(feature_query_vec_1)
		# df_pval_corrected_1 = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,data=1.0,dtype=np.float32)
		df_pval_corrected_1 = pd.DataFrame(index=feature_query_vec_1,
											columns=feature_query_vec_2,
											dtype=np.float32)
		float_precision = '%.5E'
		if 'float_precision' in select_config:
			float_precision = select_config['float_precision']
		for i1 in range(feature_query_num):
		# for i1 in range(1000):
			feature_query_id = feature_query_vec_1[i1]
			if type_id_1==0:
				query_vec_1 = pd.Index(feature_query_vec_2).difference([feature_query_id],sort=False)
			else:
				query_vec_1 = pd.Index(feature_query_vec_2)
				
			query_vec_1 = query_vec_1[pd.isna(pvalues.loc[feature_query_id,query_vec_1])==False]
			pvals = pvalues.loc[feature_query_id,query_vec_1]
			pvals_correction_vec1, pval_thresh1 = self.test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_id_correction)
			id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
			df_pval_corrected_1.loc[feature_query_id,query_vec_1] = pvals_corrected1
			if i1%1000==0:
				print('pvals_corrected1 ',np.max(pvals_corrected1),np.min(pvals_corrected1),np.mean(pvals_corrected1),np.median(pvals_corrected1),i1,feature_query_id)
			
			if i1==1000:
				if save_mode>0:
					filename_prefix_1 = filename_prefix
					output_filename = '%s/%s.pval_corrected.1.txt'%(output_file_path,filename_prefix_1)
					# df_pval_corrected_1.to_csv(output_filename,sep='\t',float_format='%.6E')
					df_pval_corrected_1.to_csv(output_filename,sep='\t',float_format=float_precision)

		if save_mode>0:
			filename_prefix_1 = filename_prefix
			output_filename = '%s/%s.pval_corrected.1.txt'%(output_file_path,filename_prefix_1)
			# df_pval_corrected_1.to_csv(output_filename,sep='\t',float_format='%.6E')
			df_pval_corrected_1.to_csv(output_filename,sep='\t',float_format=float_precision)

		return df_pval_corrected_1

	# peak-motif probability estimate, peak and motif estimate
	# p-value correction
	def test_pvalue_correction(self,pvals,alpha=0.05,method_type_id='fdr_bh'):

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

	## query correlation between gene expr
	# def test_gene_peak_query_basic_gene_tf_corr_pre1(self,df_gene_peak_query=[],df_gene_annot_expr=[],motif_data=[],peak_read=[],rna_exprs=[],type_id_pval_correction=1,thresh_corr_vec=[],filename_prefix='',save_mode=1,output_file_path='',select_config={}):
	def test_feature_correlation_1(self,df_feature_query_1=[],df_feature_query_2=[],feature_vec_1=[],feature_vec_2=[],correlation_type_vec=['spearmanr'],
											peak_read=[],rna_exprs=[],df_gene_annot_expr=[],symmetry_mode=0,type_id_1=0,type_id_pval_correction=1,
											thresh_corr_vec=[],filename_prefix='',save_mode=1,save_symmetry=0,output_file_path='',select_config={}):	
		save_mode_1 = save_mode
		if len(feature_vec_1)>0:
			df_feature_query1 = df_feature_query_1.loc[:,feature_vec_1]
		else:
			feature_vec_1 = df_feature_query1.columns

		if len(df_feature_query_2)==0:
			symmetry_mode=1
			df_feature_query2=[]
			feature_vec_2 = feature_vec_1
		else:
			if len(feature_vec_2)>0:
				df_feature_query2 = df_feature_query_2.loc[:,feature_vec_2]
			else:
				feature_vec_2=df_feature_query_2.columns
		
		print('df_feature_query1, df_feature_query2 ', df_feature_query1.shape, len(df_feature_query2))
		dict_query_1 = self.test_correlation_pvalues_pre1(df_feature_query1=df_feature_query1,
															df_feature_query2=df_feature_query2,
															filename_prefix=filename_prefix,
															correlation_type_vec=correlation_type_vec,
															type_id_pval_correction=type_id_pval_correction,
															type_id_1=type_id_1,
															save_symmetry=save_symmetry,
															save_mode=save_mode,
															output_file_path=output_file_path,
															select_config=select_config)

		return dict_query_1

	## feature correlation estimation
	# feature correlation estimation
	def test_correlation_pvalues_pre1(self,df_feature_query1,df_feature_query2=[],filename_prefix='',correlation_type_vec=['spearmanr'],
										type_id_pval_correction=1,type_id_1=0,save_symmetry=0,save_mode=1,output_file_path='',verbose=1,select_config={}):

		flag1 = 0
		feature_query_vec_1 = df_feature_query1.columns
		if len(df_feature_query2)==0:
			flag1 = 1
			df_feature_query2 = df_feature_query1
			feature_query_vec_2 = feature_query_vec_1
			symmetry_mode=1
		else:
			feature_query_vec_2 = df_feature_query2.columns
			symmetry_mode=0

		feature_query_num1, feature_query_num2 = len(feature_query_vec_1), len(feature_query_vec_2)
		print('feature_query_vec_1, feature_query_vec_2 ',feature_query_num1,feature_query_num2)

		filename_prefix_1 = filename_prefix
		if filename_prefix=='':
			filename_prefix_1 = 'test_query1'
			
		dict_query_1 = dict()
		flag_pval_correction = type_id_pval_correction
		for correlation_type in correlation_type_vec:
			if symmetry_mode==0:
				df_corr_1, df_pval_1 = utility_1.test_correlation_pvalues_pair(df_feature_query1,
																				df_feature_query2,
																				correlation_type=correlation_type,
																				float_precision=7)
				# list_query_1.append([df_corr_1,df_pval_1])
				dict_query_1[correlation_type] = [df_corr_1,df_pval_1]
				df_corr_1 = df_corr_1.loc[feature_query_vec_1,feature_query_vec_2]
				df_pval_1 = df_pval_1.loc[feature_query_vec_1,feature_query_vec_2]
				df_corr_1 = df_corr_1.fillna(0)
				df_pval_1 = df_pval_1.fillna(1)
				df_pval_1_ori = df_pval_1
			else:
				# x = 1
				feature_query_vec_1_sort = np.sort(feature_query_vec_1)
				feature_query_num1 = len(feature_query_vec_1_sort)
				df_corr_1 = pd.DataFrame(index=feature_query_vec_1_sort,columns=feature_query_vec_1_sort,dtype=np.float32)
				df_pval_1 = pd.DataFrame(index=feature_query_vec_1_sort,columns=feature_query_vec_1_sort,dtype=np.float32)
				# df_corr_1 = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
				# df_pval_1 = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
				for i1 in range(feature_query_num1-1):
					# feature_idvec_1 = feature_query_vec_1_sort[i1:(i1+1)]
					feature_id1 = feature_query_vec_1_sort[i1]
					feature_idvec_1 = [feature_id1]
					feature_idvec_2 = feature_query_vec_1_sort[(i1+1):]
					df_corr_pre1, df_pval_pre1 = utility_1.test_correlation_pvalues_pair(df_feature_query1.loc[:,feature_idvec_1],
																							df_feature_query1.loc[:,feature_idvec_2],
																							correlation_type=correlation_type,
																							float_precision=7)
					# if flag_pval_correction>0:
					# 	df_corr_pre1 = df_corr_pre1.fillna(0)
					# 	df_pval_pre1 = df_pval_pre1.fillna(1)
					df_corr_pre1 = df_corr_pre1.fillna(0)
					df_pval_pre1 = df_pval_pre1.fillna(1)
					df_corr_1.loc[feature_idvec_1,feature_idvec_2] = df_corr_pre1.loc[feature_idvec_1,feature_idvec_2]
					df_pval_1.loc[feature_idvec_1,feature_idvec_2] = df_pval_pre1.loc[feature_idvec_1,feature_idvec_2]
					if (verbose>0) and (i1%100==0):
						print('feature_idvec_1, feature_idvec_2 ',feature_idvec_1,len(feature_idvec_2),i1)
						t_corr_value_1 = df_corr_pre1.loc[feature_id1,:]
						t_pval_1 = df_pval_pre1.loc[feature_id1,:]
						print(t_corr_value_1.max(),t_corr_value_1.min(),t_corr_value_1.mean(),t_corr_value_1.median())
						print(t_pval_1.max(),t_pval_1.min(),t_pval_1.mean(),t_pval_1.median())

				df_corr_1 = df_corr_1.loc[feature_query_vec_1,feature_query_vec_2]
				df_pval_1 = df_pval_1.loc[feature_query_vec_1,feature_query_vec_2]
				df_pval_1_ori = df_pval_1
				df_corr_1_ori = df_corr_1
				# print(df_corr_1,df_pval_1)
				# print(df_corr_1.T,df_pval_1.T)
				df_corr_1 = df_corr_1.fillna(0)
				df_pval_1 = df_pval_1.fillna(0)
				df_corr_1 = df_corr_1+df_corr_1.T
				df_pval_1 = df_pval_1+df_pval_1.T
				# np.fill_diagonal(df_corr_1,1)
				# np.fill_diagonal(df_pval_1,0)
				id1, id2 = feature_query_vec_1[0:5], feature_query_vec_2[0:5]
				if verbose>0:
					print('df_corr_1, df_pval_1 ',df_corr_1.loc[id1,id2],df_pval_1.loc[id1,id2])
				# print('df_corr_1_ori ',df_corr_1_ori.loc[id1,id2])
				# print('df_pval_1_ori ',df_pval_1_ori.loc[id1,id2])
			print('df_corr_1, df_pval_1 ',df_corr_1.shape, df_pval_1.shape)

			## pvalue correction
			df_pval_corrected_1 = []
			if flag_pval_correction==1:
				print('pvalue correction')
				alpha = 0.05
				method_type_id_correction = 'fdr_bh'
				df_pval_corrected_1 = self.test_correlation_pvalue_correction_pre1(pvalues=df_pval_1,
																					alpha=alpha,
																					method_type_correction=method_type_id_correction,
																					filename_prefix='',
																					correlation_type_vec=['spearmanr'],
																					save_mode=0,
																					output_file_path='',
																					type_id_1=type_id_1,
																					select_config=select_config)
				print('df_pval_1 ',df_pval_1.shape)

			dict_query_1[correlation_type] = [df_corr_1,df_pval_1,df_pval_corrected_1]
			if save_mode>0:
				df_corr_1_save, df_pval_1_save = df_corr_1, df_pval_1
				if symmetry_mode>0:
					if save_symmetry==0:
						df_corr_1_save = df_corr_1_ori
						df_pval_1_save = df_pval_1_ori
					# else:
					# 	df_corr_1 =  df_corr_1+df_corr_1.T
				output_filename_1 = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_1,correlation_type)
				float_precision = '%.6E'
				if 'float_precision' in select_config:
					float_precision = select_config['float_precision']
				# df_corr_1_save.to_csv(output_filename_1,sep='\t',float_format='%.6E')
				df_corr_1_save.to_csv(output_filename_1,sep='\t',float_format=float_precision)
				output_filename_2 = '%s/%s.%s.pval.1.txt'%(output_file_path,filename_prefix_1,correlation_type)
				# df_pval_1_save.to_csv(output_filename_2,sep='\t',float_format='%.6E')
				df_pval_1_save.to_csv(output_filename_2,sep='\t',float_format=float_precision)
				if flag_pval_correction>0:
					output_filename_3 = '%s/%s.%s.pval_corrected.1.txt'%(output_file_path,filename_prefix_1,correlation_type)
					# df_pval_corrected_1.to_csv(output_filename_3,sep='\t',float_format='%.6E')
					df_pval_corrected_1.to_csv(output_filename_3,sep='\t',float_format=float_precision)
					
		return dict_query_1

	## query correlations above threshold
	def test_query_correlation_threshold(self,corr_value,pval_value,thresh_corr_vec,thresh_pval_vec,type_id_1=0,select_config={}):

		# id1 = (corr_value.abs()>thresh_corr_1.abs())
		# if type_id_1==0:
		# 	id1 = (corr_value>thresh_corr_)&(pval_vlaue<thresh_pval_)
		# else:
		# 	id1 = (corr_value<thresh_corr_)&(pval_vlaue<thresh_pval_)
		# if type_id_1>0:
		# 	corr_value = corr_value.abs()
		# 	thresh_corr_vec = np.abs(thresh_corr_vec)
		thresh_num1 = len(thresh_corr_vec)
		# corr_value_abs = corr_value.abs()
		list1 = []
		for i1 in range(thresh_num1):
			thresh_corr_, thresh_pval_ = thresh_corr_vec[i1], thresh_pval_vec[i1]
			type_id1 = type_id_1[i1]
			if type_id1==0:
				id1 = (corr_value_>thresh_corr_)&(pval_value<thresh_pval_)
			else:
				id1 = (corr_value_<thresh_corr_)&(pval_value<thresh_pval_)
			list1.append(id1)

		return list1

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)


