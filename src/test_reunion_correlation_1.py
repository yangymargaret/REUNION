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
	def __init__(self,file_path='',run_id=1,,method=1,verbose=1,select_config={}):

		self.save_path_1 = file_path
		self.run_id = run_id
		self.method = method
		self.select_config = select_config
		self.verbose_internal = verbose

	## ====================================================
	# query correlation between feature query
	# query correlation between gene expression
	def test_feature_correlation_1(self,df_feature_query_1=[],df_feature_query_2=[],feature_vec_1=[],feature_vec_2=[],correlation_type_vec=['spearmanr'],
											peak_read=[],rna_exprs=[],df_gene_annot_expr=[],symmetry_mode=0,type_id_1=0,type_id_pval_correction=1,
											thresh_corr_vec=[],filename_prefix='',save_mode=1,save_symmetry=0,output_file_path='',select_config={}):	
		
		if len(feature_vec_1)>0:
			df_feature_query1 = df_feature_query_1.loc[:,feature_vec_1]
		else:
			feature_vec_1 = df_feature_query1.columns

		if len(df_feature_query_2)==0:
			symmetry_mode = 1
			df_feature_query2 = []
			feature_vec_2 = feature_vec_1
		else:
			if len(feature_vec_2)>0:
				df_feature_query2 = df_feature_query_2.loc[:,feature_vec_2]
			else:
				feature_vec_2 = df_feature_query_2.columns
		
		verbose_internal = self.verbose_internal
		if verbose_internal>0:
			print('dataframe of feature 1, size of ',df_feature_query1.shape)
			if (symmetry_mode==0) or (len(df_feature_query2)>0):
				print('dataframe of feature 2, size of ',df_feature_query2.shape)

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

	## ====================================================
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
		print('feature_vec_1: %d, feature_vec_2: %d'%(feature_query_num1,feature_query_num2))

		filename_prefix_1 = filename_prefix
		if filename_prefix=='':
			filename_prefix_1 = 'test_query1'
			
		dict_query_1 = dict()
		flag_pval_correction = type_id_pval_correction
		for correlation_type in correlation_type_vec:
			if symmetry_mode==0:
				# compute feature correlation
				df_corr_1, df_pval_1 = self.test_correlation_pvalues_pair(df1=df_feature_query1,
																			df2=df_feature_query2,
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
				feature_query_vec_1_sort = np.sort(feature_query_vec_1)
				feature_query_num1 = len(feature_query_vec_1_sort)
				df_corr_1 = pd.DataFrame(index=feature_query_vec_1_sort,columns=feature_query_vec_1_sort,dtype=np.float32)
				df_pval_1 = pd.DataFrame(index=feature_query_vec_1_sort,columns=feature_query_vec_1_sort,dtype=np.float32)
				# df_corr_1 = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
				# df_pval_1 = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
				for i1 in range(feature_query_num1-1):
					feature_id1 = feature_query_vec_1_sort[i1]
					feature_idvec_1 = [feature_id1]
					feature_idvec_2 = feature_query_vec_1_sort[(i1+1):]
					
					# compute feature correlation
					df_corr_pre1, df_pval_pre1 = self.test_correlation_pvalues_pair(df1=df_feature_query1.loc[:,feature_idvec_1],
																					df2=df_feature_query1.loc[:,feature_idvec_2],
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

			print('df_corr_1, df_pval_1 ',df_corr_1.shape, df_pval_1.shape)

			# pvalue correction
			df_pval_corrected_1 = []
			if flag_pval_correction==1:
				print('pvalue correction')
				alpha = 0.05
				method_type_correction = 'fdr_bh'
				df_pval_corrected_1 = self.test_correlation_pvalue_correction_pre1(pvalues=df_pval_1,
																					alpha=alpha,
																					method_type_correction=method_type_correction,
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

	## ====================================================
	# p-value correction
	def test_correlation_pvalue_correction_pre1(self,pvalues,alpha=0.05,method_type_correction='fdr_bh',filename_prefix='',correlation_type_vec=['spearmanr'],
													type_id_pval_correction=1,save_mode=1,output_file_path='',type_id_1=0,select_config={}):

		df_pval_ = pvalues
		feature_query_vec_1 = df_pval_.index
		feature_query_vec_2 = df_pval_.columns

		feature_query_num = len(feature_query_vec_1)
		df_pval_corrected_1 = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
		
		float_precision = '%.5E'
		if 'float_precision' in select_config:
			float_precision = select_config['float_precision']
		
		for i1 in range(feature_query_num):
			feature_query_id = feature_query_vec_1[i1]
			if type_id_1==0:
				query_vec_1 = pd.Index(feature_query_vec_2).difference([feature_query_id],sort=False)
			else:
				query_vec_1 = pd.Index(feature_query_vec_2)
				
			query_vec_1 = query_vec_1[pd.isna(pvalues.loc[feature_query_id,query_vec_1])==False]
			pvals = pvalues.loc[feature_query_id,query_vec_1]
			pvals_correction_vec1, pval_thresh1 = self.test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_correction)
			id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
			df_pval_corrected_1.loc[feature_query_id,query_vec_1] = pvals_corrected1
			if i1%1000==0:
				print('pvals_corrected1 ',np.max(pvals_corrected1),np.min(pvals_corrected1),np.mean(pvals_corrected1),np.median(pvals_corrected1),i1,feature_query_id)
			
			if i1==1000:
				if save_mode>0:
					# save the current adjusted p-values
					filename_prefix_1 = filename_prefix
					output_filename = '%s/%s.pval_corrected.1.txt'%(output_file_path,filename_prefix_1)
					df_pval_corrected_1.to_csv(output_filename,sep='\t',float_format=float_precision)

		if save_mode>0:
			filename_prefix_1 = filename_prefix
			output_filename = '%s/%s.pval_corrected.1.txt'%(output_file_path,filename_prefix_1)
			df_pval_corrected_1.to_csv(output_filename,sep='\t',float_format=float_precision)

		return df_pval_corrected_1

	## ====================================================
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

	## ====================================================
	# correlation and pvalue calculation 
	# from the website: https://enterprise-docs.anaconda.com/en/latest/data-science-workflows/data/stats.html
	def test_correlation_pvalues_pair(self,df1,df2,correlation_type='spearmanr',float_precision=7):
		
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

	## ======================================================
	# compute peak accessibility-TF expression correlation
	def test_peak_tf_correlation_query_1(self,motif_data=[],peak_query_vec=[],motif_query_vec=[],peak_read=[],rna_exprs=[],correlation_type='spearmanr',
											pval_correction=1,alpha=0.05,method_type_correction='fdr_bh',flag_load=0,field_load=[],parallel_mode=0,
											save_mode=1,input_file_path='',input_filename_list=[],output_file_path='',
											filename_prefix='',verbose=0,select_config={}):

		if filename_prefix=='':
			filename_prefix = 'test_peak_tf_correlation'
		if flag_load>0:
			if len(field_load)==0:
				field_load = [correlation_type,'pval','pval_corrected']
				field_annot = ['correlation','p-value','corrected p-value']
			field_num = len(field_load)

			file_num = len(input_filename_list)
			list_query = []
			if file_num==0:
				input_filename_list = ['%s/%s.%s.1.txt'%(input_file_path,filename_prefix,filename_annot) for filename_annot in field_load]

			dict_query = dict()
			print('load estimated peak accessibility-TF expression correlation and p-value')
			for i1 in range(field_num):
				filename_annot1 = field_load[i1]
				input_filename = input_filename_list[i1]
				if os.path.exists(input_filename)==True:
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
					field_query1 = filename_annot1
					dict_query.update({field_query1:df_query})

					field_id1 = field_annot[i1]
					print('%s, dataframe of size '%(field_id1),df_query.shape)
					print('input_filename: %s'%(input_filename))
				else:
					print('the file does not exist: %s'%(input_filename))
					flag_load = 0
				
			if len(dict_query)==field_num:
				return dict_query
			
		if flag_load==0:
			print('peak accessibility-TF expression correlation estimation ')
			start = time.time()
			df_peak_tf_corr_, df_peak_tf_pval_, df_peak_tf_pval_corrected, df_motif_basic = self.test_peak_tf_correlation_1(motif_data=motif_data,
																																peak_query_vec=peak_query_vec,
																																motif_query_vec=motif_query_vec,
																																peak_read=peak_read,
																																rna_exprs=rna_exprs,
																																correlation_type=correlation_type,
																																pval_correction=pval_correction,
																																alpha=alpha,
																																method_type_correction=method_type_correction,
																																parallel_mode=parallel_mode,
																																select_config=select_config)

			field_query = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected','motif_basic']
			filename_annot_vec = [correlation_type,'pval','pval_corrected','motif_basic']
			list_query1 = [df_peak_tf_corr_, df_peak_tf_pval_, df_peak_tf_pval_corrected, df_motif_basic]
			# dict_query = dict(zip(field_query,list_query1))
			dict_query = dict(zip(filename_annot_vec,list_query1))
			query_num1 = len(list_query1)
			stop = time.time()
			print('peak accessibility-TF expression correlation estimation used: %.5fs'%(stop-start))
			
			flag_save_text = 1
			if 'flag_save_text_peak_tf' in select_config:
				flag_save_text = select_config['flag_save_text_peak_tf']
			
			if save_mode>0:
				if output_file_path=='':
					output_file_path = select_config['data_path']
				if flag_save_text>0:
					for i1 in range(query_num1):
						df_query = list_query1[i1]
						if len(df_query)>0:
							filename_annot1 = filename_annot_vec[i1]
							output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix,filename_annot1)
							if i1 in [3]:
								df_query.to_csv(output_filename,sep='\t',float_format='%.6f')
							else:
								df_query.to_csv(output_filename,sep='\t',float_format='%.5E')
							print('df_query ',df_query.shape,filename_annot1)

		return dict_query

	## ====================================================
	# peak accessibility-TF expression correlation
	def test_peak_tf_correlation_1(self,motif_data,peak_query_vec=[],motif_query_vec=[],
									peak_read=[],rna_exprs=[],correlation_type='spearmanr',pval_correction=1,
									alpha=0.05,method_type_correction = 'fdr_bh',parallel_mode=0,verbose=1,select_config={}):

		if len(motif_query_vec)==0:
			motif_query_name_ori = motif_data.columns
			motif_query_name_expr = motif_query_name_ori.intersection(rna_exprs.columns,sort=False)
			# print('motif_query_name_ori, motif_query_name_expr ',len(motif_query_name_ori),len(motif_query_name_expr))
			motif_query_vec = motif_query_name_expr
		else:
			motif_query_vec_1 = motif_query_vec
			motif_query_vec = pd.Index(motif_query_vec).intersection(rna_exprs.columns,sort=False)
		
		motif_query_num = len(motif_query_vec)
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

		if parallel_mode==0:
			t_vec_1 = self.test_peak_tf_correlation_unit1(motif_data=motif_data_query,peak_query_vec=[],motif_query_vec=motif_query_vec,
															peak_read=peak_read,rna_exprs=rna_exprs,correlation_type=correlation_type,pval_correction=pval_correction,
															alpha=alpha,method_type_correction=method_type_correction,parallel_mode=0,verbose=1,select_config=select_config)
			df_corr_, df_pval_, df_pval_corrected, df_motif_basic = t_vec_1
		else:
			dict_query_1 = dict()
			field_query = ['correlation','pval','pval_corrected','motif_basic']
			for field_id in field_query:
				dict_query_1[field_id] = []

			query_res_local = Parallel(n_jobs=-1)(delayed(self.test_peak_tf_correlation_unit1)(motif_data=motif_data_query,motif_query_vec=[motif_id_query],peak_read=peak_read,rna_exprs=rna_exprs,
																								correlation_type=correlation_type,pval_correction=pval_correction,alpha=alpha,method_type_correction=method_type_correction,verbose=verbose,select_config=select_config) for motif_id_query in motif_query_vec)
			
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

	## ======================================================
	# peak accessibility-TF expression correlation
	def test_peak_tf_correlation_unit1(self,motif_data,peak_query_vec=[],motif_query_vec=[],
										peak_read=[],rna_exprs=[],correlation_type='spearmanr',pval_correction=1,
										alpha=0.05,method_type_correction='fdr_bh',parallel_mode=0,verbose=1,select_config={}):
			
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
			
			df_feature_query1 = peak_read.loc[:,peak_loc_query]
			df_feature_query2 = rna_exprs.loc[:,[motif_id]]
			df_corr_1, df_pval_1 = self.test_correlation_pvalues_pair(df1=df_feature_query1,
																		df2=df_feature_query2,
																		correlation_type=correlation_type,
																		float_precision=6)
			
			df_corr_.loc[peak_loc_query,motif_id] = df_corr_1.loc[peak_loc_query,motif_id]
			df_pval_.loc[peak_loc_query,motif_id] = df_pval_1.loc[peak_loc_query,motif_id]

			corr_max, corr_min = df_corr_1.max().max(), df_corr_1.min().min()
			peak_num = len(peak_loc_query)
			df_motif_basic.loc[motif_id] = [peak_num,corr_max,corr_min]
			
			interval_1 = 100
			if verbose>0:
				if i1%interval_1==0:
					print('TF: %s, %d, number of peaks with motif: %d, maximum peak accessibility-TF expression correlation: %s, minimum correlation: %s'%(motif_id,i1,peak_num,corr_max,corr_min))
			
			if flag_pval_correction>0:
				pvals = np.asarray(df_pval_1.loc[peak_loc_query,motif_id])
				pvals_correction_vec1, pval_thresh1 = self.test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_correction)
				id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
				df_pval_corrected.loc[peak_loc_query,motif_id] = pvals_corrected1
				if (verbose>0) and (i1%100==0):
					print('p-value correction: alpha: %s, method type: %s, minimum p-value corrected: %s, maximum p-value corrected: %s '%(alpha,method_type_correction,np.min(pvals_corrected1),np.max(pvals_corrected1)))

		return (df_corr_, df_pval_, df_pval_corrected, df_motif_basic)

	## ======================================================
	# query correlations above threshold
	def test_query_correlation_threshold(self,corr_value,pval_value,thresh_corr_vec,thresh_pval_vec,type_id_1=0,select_config={}):

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


