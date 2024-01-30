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
# import xgboost
# import xgbfir

from pandas import read_excel

import pyranges as pr
import warnings

import palantir 
import phenograph

import sys
from tqdm.notebook import tqdm

import csv
import os
import os.path
import shutil
import sklearn

from optparse import OptionParser
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from scipy import stats
from scipy.stats import multivariate_normal, skew, pearsonr, spearmanr
from scipy.stats import wilcoxon, mannwhitneyu, kstest, ks_2samp, chisquare, fisher_exact
from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq
from scipy.stats import gaussian_kde, zscore
from scipy.stats import poisson, multinomial
from scipy.stats import norm
from scipy.stats import rankdata
import scipy.sparse
from scipy.sparse import spmatrix
from scipy.sparse import hstack, csr_matrix, csc_matrix, issparse, vstack
from scipy import signal
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from sklearn.preprocessing import KBinsDiscretizer

from scipy.cluster.hierarchy import dendrogram, linkage

import gc
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

import utility_1
from utility_1 import log_transform, pyranges_from_strings, plot_cell_types, plot_gene_expression
from utility_1 import _dot_func, impute_data, density_2d, test_query_index
from utility_1 import score_function, test_file_merge_1, _pyranges_to_strings, test_save_anndata
from utility_1 import spearman_corr, pearson_corr
import h5py
import json
import pickle

import itertools
from itertools import combinations

from scipy import signal

# import test_reunion_1
# from test_reunion_1 import _Base2_correlation
import test_reunion_2
from test_reunion_2 import _Base2_correlation3

# get_ipython().run_line_magic('matplotlib', 'inline')
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

# %matplotlib inline
sns.set_style('ticks')
# matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['image.cmap'] = 'Spectral_r'
warnings.filterwarnings(action="ignore", module="matplotlib", message="findfont")

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
# matplotlib.rc('xlabel', labelsize=12)
# matplotlib.rc('ylabel', labelsize=12)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 15
# plt.rcParams["figure.autolayout"] = True
# warnings.filterwarnings(action="ignore", module="matplotlib", message="findfont")

class _Base2_correlation5(_Base2_correlation3):
	"""Base class for peak-TF-gene link estimation
	"""
	def __init__(self,file_path,run_id=1,species_id=1,cell='ES', 
					generate=1,
					chromvec=[1],
					test_chromvec=[2],
					featureid=1,
					typeid=1,
					df_gene_annot_expr=[],
					method=1,
					flanking=50,
					normalize=1,
					type_id_feature=0,
					config={},
					select_config={}):

		_Base2_correlation3.__init__(self,file_path=file_path,
										run_id=run_id,
										species_id=species_id,
										cell=cell,
										generate=generate,
										chromvec=chromvec,
										test_chromvec=test_chromvec,
										featureid=featureid,
										df_gene_annot_expr=df_gene_annot_expr,
										typeid=typeid,
										method=method,
										flanking=flanking,
										normalize=normalize,
										type_id_feature=type_id_feature,
										config=config,
										select_config=select_config)

	## file_path query
	def test_config_query_pre2(self,beta_mode=0,save_mode=1,overwrite=False,select_config={}):

		# print('test_config_query')
		correlation_type = 'spearmanr'

		column_idvec = ['motif_id','peak_id','gene_id']
		column_gene_tf_corr_peak =  ['gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1']
		thresh_insilco_ChIP_seq = 0.1
		flag_save_text = 1

		field_query = ['column_idvec','correlation_type','column_gene_tf_corr_peak','thresh_insilco_ChIP-seq','flag_save_text']
		list1 = [column_idvec,correlation_type,column_gene_tf_corr_peak,thresh_insilco_ChIP_seq,flag_save_text]
		field_num1 = len(field_query)
		for i1 in range(field_num1):
			field1 = field_query[i1]
			select_config.update({field1:list1[i1]})

		self.select_config = select_config

		return select_config

	## peak accessibility-TF expression correlation
	def test_peak_tf_correlation_1(self,motif_data,peak_query_vec=[],motif_query_vec=[],
									peak_read=[],rna_exprs=[],correlation_type='spearmanr',pval_correction=1,
									alpha=0.05,method_type_id_correction = 'fdr_bh',verbose=1,select_config={}):

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
		df_corr_ = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
		df_pval_ = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
		flag_pval_correction = pval_correction
		if flag_pval_correction>0:
			df_pval_corrected = df_pval_.copy()
		else:
			df_pval_corrected = []
		df_motif_basic = pd.DataFrame(index=feature_query_vec_2,columns=['peak_num','corr_max','corr_min'])

		for i1 in range(motif_query_num):
			motif_id = motif_query_vec[i1]
			peak_loc_query = peak_loc_ori[motif_data_query.loc[:,motif_id]>0]

			df_feature_query1 = peak_read.loc[:,peak_loc_query]
			df_feature_query2 = rna_exprs.loc[:,[motif_id]]
			df_corr_1, df_pval_1 = utility_1.test_correlation_pvalues_pair(df_feature_query1,df_feature_query2,correlation_type=correlation_type,float_precision=6)
			
			df_corr_.loc[peak_loc_query,motif_id] = df_corr_1.loc[peak_loc_query,motif_id]
			df_pval_.loc[peak_loc_query,motif_id] = df_pval_1.loc[peak_loc_query,motif_id]

			corr_max, corr_min = df_corr_1.max().max(), df_corr_1.min().min()
			peak_num = len(peak_loc_query)
			df_motif_basic.loc[motif_id] = [peak_num,corr_max,corr_min]
			if verbose>0:
				if i1%10==0:
					print('motif_id: %s, id_query: %d, peak_num: %s, maximum peak accessibility-TF expr. correlation: %s, minimum correlation: %s'%(motif_id,i1,peak_num,corr_max,corr_min))
			if flag_pval_correction>0:
				pvals = df_pval_1.loc[peak_loc_query,motif_id]
				pvals_correction_vec1, pval_thresh1 = utility_1.test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_id_correction)
				id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
				df_pval_corrected.loc[peak_loc_query,motif_id] = pvals_corrected1
				if (verbose>0) and (i1%100==0):
					print('pvalue correction: alpha: %s, method_type: %s, minimum pval_corrected: %s, maximum pval_corrected: %s '%(alpha,method_type_id_correction,np.min(pvals_corrected1),np.max(pvals_corrected1)))

		return df_corr_, df_pval_, df_pval_corrected, df_motif_basic

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_peak_tf_correlation_query_1(self,motif_data=[],peak_query_vec=[],motif_query_vec=[],peak_read=[],rna_exprs=[],correlation_type='spearmanr',flag_load=0,field_load=[],
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
			# if len(input_filename_list)==0:
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
					df_query = pd.read_csv(input_filename,sep='\t')
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
			df_peak_tf_corr_, df_peak_tf_pval_, df_peak_tf_pval_corrected, df_motif_basic = self.test_peak_tf_correlation_1(motif_data=motif_data,
																															peak_query_vec=peak_query_vec,
																															motif_query_vec=motif_query_vec,
																															peak_read=peak_read,
																															rna_exprs=rna_exprs,
																															correlation_type=correlation_type,
																															select_config=select_config)

			field_query = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected','motif_basic']
			list_query1 = [df_peak_tf_corr_, df_peak_tf_pval_, df_peak_tf_pval_corrected, df_motif_basic]
			dict_query = dict(zip(field_query,list_query1))
			query_num1 = len(list_query1)
			stop = time.time()
			print('peak accessibility-TF expr correlation estimation used: %.5fs'%(stop-start))
			# if filename_prefix=='':
			# 	# filename_prefix_1 = 'test_peak_tf_correlation'
			# 	filename_prefix = 'test_peak_tf_correlation'
			filename_annot_vec = [correlation_type,'pval','pval_corrected','motif_basic']
			
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
							# output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix,filename_annot1)
							output_filename = '%s/%s.%s.1.copy1.txt'%(output_file_path,filename_prefix,filename_annot1)
							if i1 in [3]:
								df_query.to_csv(output_filename,sep='\t',float_format='%.6f')
							else:
								df_query.to_csv(output_filename,sep='\t',float_format='%.5E')
							print('df_query ',df_query.shape,filename_annot1)
				
		return dict_query

	## ====================================================
	# the in silico ChIP-seq library method
	# TF binding score with normalization
	# motif_data: the motif scanning results (pandas DataFrame, peak num by TF num, index:peak loci, columns: TF names)
	# motif_data_score: the motif scores from motif scanning results (pandas DataFrame, peak num by TF num, index:peak loci, columns: TF name)
	# peak_read: the peak accessibility matrix (cell num by peak num, index:sample id of metacells, columns: peak loci)
	# peak_read_celltype: the peak accessibility matrix for cell types (if we want to use maximal chromatin accessiblity across cell types, by default we use the maximal chromatin accessiblity of peak loci across the metacells)
	# if peak_read_celltype is provided as input, both maximal chromatin accessibility across metacells and across the cell types will be computed
	# input_filename: the filename of the peak accessibility-TF expression correlation matrix (peak num by TF num, pandas DataFrame, index: peak loci, columns: TF names)
	# peak_query_vec: the vector of peak loci, default: the index of motif_data
	# motif_query_vec: the vector of TF names, default: the columns of the peak accessibility-TF expression correlation matrix
	# output_filename: the output filename of the estimated in silico ChIP-seq TF binding score
	def test_peak_tf_score_normalization_1(self,peak_query_vec=[],motif_query_vec=[],motif_data=[],motif_data_score=[],
												df_peak_tf_expr_corr_=[],input_filename='',
												peak_read=[],rna_exprs=[],peak_read_celltype=[],df_peak_annot=[],correlation_type='spearmanr',
												filename_annot='',overwrite=False,save_mode=1,output_file_path='',output_filename='',beta_mode=0,verbose=0,select_config={}):

		input_filename_pre1 = output_filename
		if (os.path.exists(input_filename_pre1)==True) and (overwrite==False):
			print('the file exists: %s'%(input_filename_pre1))
			# print('overwrite: ',overwrite)
			df_pre1 = pd.read_csv(input_filename_pre1,index_col=0,sep='\t') # retrieve the TF binding score estimated and saved
			return df_pre1

		sample_id = peak_read.index
		if len(motif_data)==0:
			motif_data = (motif_data_score.abs()>0)
		print('motif_data, motif_data_score ', motif_data.shape, motif_data_score.shape, motif_data_score[0:5])

		df_pre1 = []
		# compute peak accessibility-TF expression correlation
		if len(df_peak_tf_expr_corr_)==0:
			input_filename_expr_corr = input_filename
			if input_filename_expr_corr!='':
				if os.path.exists(input_filename_expr_corr)==True:
					b = input_filename_expr_corr.find('.txt')
					if b>0:
						df_peak_tf_expr_corr_ = pd.read_csv(input_filename_expr_corr,index_col=0,sep='\t')
					else:
						adata = sc.read(input_filename_expr_corr)
						df_peak_tf_expr_corr_ = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=adata.X.toarray(),dtype=np.float32)
					print('df_peak_tf_expr_corr_ ',df_peak_tf_expr_corr_.shape,input_filename_expr_corr)
				else:
					print('the file does not exist:%s'%(input_filename_expr_corr))
					return
			else:
				print('peak accessibility-TF expr correlation not provided\n perform peak accessibility-TF expr correlation estimation')
				filename_prefix = 'test_peak_tf_correlation'
				column_id1 = 'peak_tf_corr'
				dict_peak_tf_corr_ = self.test_peak_tf_correlation_query_1(motif_data=motif_data,
																			peak_query_vec=[],
																			motif_query_vec=[],
																			peak_read=peak_read,
																			rna_exprs=rna_exprs,
																			correlation_type=correlation_type,
																			save_mode=save_mode,
																			output_file_path=output_file_path,
																			filename_prefix=filename_prefix,
																			select_config=select_config)
				df_peak_tf_expr_corr_ = dict_peak_tf_corr_[column_id1]

		# peak_id_1, motif_id_1 = df_peak_tf_expr_corr_.index, df_peak_tf_expr_corr_.columns
		peak_loc_ori = motif_data.index
		peak_loc_num = len(peak_loc_ori)
		if len(motif_query_vec)==0:
			motif_query_vec = df_peak_tf_expr_corr_.columns

		if len(peak_query_vec)==0:
			peak_query_vec = motif_data.index

		motif_query_num = len(motif_query_vec)
		peak_query_num = len(peak_query_vec)
		print('peak number:%d, TF number:%d'%(peak_query_num,motif_query_num))
		
		motif_data = motif_data.loc[peak_query_vec,motif_query_vec]
		motif_data_score = motif_data_score.loc[peak_query_vec,motif_query_vec]
		df_peak_tf_expr_corr_1 = df_peak_tf_expr_corr_.loc[peak_query_vec,motif_query_vec]
		df_peak_tf_expr_corr_1 = df_peak_tf_expr_corr_1.fillna(0)
		peak_id, motif_id = peak_query_vec, motif_query_vec
		
		mask = (motif_data>0)	# shape: (peak_num,motif_num)
		mask_1 = mask
		df_peak_tf_expr_corr_1[~mask_1] = 0

		## query peak accessibility by cell type
		flag_query_by_celltype = 0
		if len(peak_read_celltype)>0:
			flag_query_by_celltype = 1

		min_peak_number = 1
		field_query_1 = ['correlation_score','max_accessibility_score','motif_score','motif_score_normalize',
							'score_1','score_pred1']
		field_query = field_query_1
		list_query1 = []
		column_id_query = 'max_accessibility_score'
		load_mode = 0
		if (len(df_peak_annot)>0) and (column_id_query in df_peak_annot.columns):
			load_mode = 1  # query maximum peak accessibility from peak annotation

		print('load_mode: ',load_mode)
		motif_query_num_ori = motif_query_num

		for i1 in range(motif_query_num):
			motif_id1 = motif_query_vec[i1]
			id1 = motif_data.loc[:,motif_id1]>0
			peak_id1 = peak_query_vec[id1]
			motif_score = motif_data_score.loc[peak_id1,motif_id1]
			motif_score_1 = motif_score/np.max(motif_score) # normalize motif_score per motif
			# motif_score_minmax = motif_score/np.max(motif_score)	# normalize motif score per motif
			correlation_score = df_peak_tf_expr_corr_1.loc[peak_id1,motif_id1]
			if load_mode==0:
				max_accessibility_score = peak_read.loc[:,peak_id1].max(axis=0)
			else:
				max_accessibility_score = peak_annot.loc[peak_id,column_id_query]
			
			score_1 = minmax_scale(max_accessibility_score*motif_score,[0,1])
			# score_1 = minmax_scale(max_accessibility_score*motif_score_1,[0,1])
			score_pred1 = correlation_score*score_1
			list1 = [correlation_score,max_accessibility_score,motif_score,motif_score_1,score_1,score_pred1]

			score_2, score_pred2 = [], []
			if flag_query_by_celltype>0:
				max_accessibility_score_celltype = peak_read_celltype.loc[:,peak_id1].max(axis=0)
				score_2 = minmax_scale(max_accessibility_score_celltype*motif_score_1,[0,1])
				score_pred2 = correlation_score*score_2
				field_query = field_query_1 + ['max_accessibility_score_celltype','score_celltype','score_pred_celltype']
				list1 = list1+[max_accessibility_score_celltype,score_2,score_pred2]
			
			dict1 = dict(zip(field_query,list1))
			df1 = pd.DataFrame.from_dict(data=dict1,orient='columns',dtype=np.float32)
			df1['peak_id'] = peak_id1
			df1 = df1.loc[:,['peak_id']+field_query]
			df1.index = [motif_id1]*df1.shape[0]
			df1 = df1.sort_values(by=['score_pred1'],ascending=False)
			list_query1.append(df1)
			if (verbose>0) and (i1%100==0):
				# print('motif_id1, peak_id1, score_pred1 ',motif_id1,i1,len(peak_id1))
				print('TF:%s, peak number:%d, %d'%(motif_id1,len(peak_id1),i1))
				print(np.max(score_pred1),np.min(score_pred1),np.mean(score_pred1),np.median(score_pred1))

		df_pre1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
		if (save_mode>0) and (output_filename!=''):
			df_pre1.to_csv(output_filename,sep='\t',float_format='%.6f')

		return df_pre1

	## ======================================================
	# peak-TF link query by threshold from the TF motif score estimated using the in silico ChIP-seq library method
	def test_peak_tf_score_normalization_query_1(self,gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],
													df_gene_peak_query=[],motif_data=[],input_filename='',
													thresh_score=0.1,peak_read=[],rna_exprs=[],peak_read_celltype=[],
													filename_annot='',save_mode=1,output_filename='',select_config={}):

		sample_id = peak_read.index
		filename_annot1 = filename_annot
		if input_filename=='':
			# input_file_path1 = self.save_path_1
			input_file_path1 = select_config['data_path']
			input_filename_1 = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path1,filename_annot1)
		else:
			input_filename_1 = input_filename

		if (os.path.exists(input_filename_1)==False):
			print('the file does not exist: %s'%(input_filename_1))
			return
		df_pre1 = pd.read_csv(input_filename_1,index_col=0,sep='\t')

		thresh_1 = thresh_score
		# thresh_1 = 0.1
		# df_pre2 = df_pre1.loc[df_pre1['score_pred1'].abs()>thresh_1]
		df_pre2 = df_pre1.loc[df_pre1['score_pred1']>thresh_1]
		print('df_pre1, df_pre2 ', df_pre1.shape, df_pre2.shape)

		if save_mode>0:
			if output_filename=='':
				# output_file_path = self.save_path_1
				output_file_path = select_config['data_path']
				output_filename = '%s/test_motif_score_normalize_insilico.%s.thresh%s.txt'%(output_file_path,filename_annot1,thresh_1)
			df_pre2.to_csv(output_filename,sep='\t')
		print('df_pre2 ',df_pre2.shape)

		return df_pre2

	## ====================================================
	# TF motif score normalization
	# dataframe: motif_score_minmax, motif_score_log_normalize_bound, score_accessibility_minmax, score_accessibility, score_1
	def test_peak_tf_score_normalization_pre_compute(self,gene_query_vec=[],
														peak_query_vec=[],
														motif_query_vec=[],
														df_gene_peak_query=[],
														motif_data=[],
														motif_data_score=[],
														peak_read=[],
														rna_exprs=[],
														peak_read_celltype=[],
														df_peak_annot=[],
														flag_motif_score_quantile=0,
														flag_motif_score_basic=1,
														overwrite=False,
														save_mode=1,
														filename_annot='',
														output_file_path='',
														beta_mode=0,
														verbose=0,
														select_config={}):

		sample_id = peak_read.index
		if len(motif_data_score)==0:
			motif_data_score = self.motif_data_score
		if len(motif_data)==0:
			motif_data = (motif_data_score.abs()>0)
		print('motif_data, motif_data_score ', motif_data.shape, motif_data_score.shape, motif_data_score[0:5])

		motif_query_vec_ori = np.unique(motif_data.columns)
		peak_loc_ori = motif_data.index
		peak_loc_num = len(peak_loc_ori)
		if len(motif_query_vec)==0:
			motif_query_vec = motif_query_vec_ori
		motif_query_num = len(motif_query_vec)

		if len(peak_query_vec)==0:
			peak_query_vec = peak_loc_ori
		peak_query_num = len(peak_query_vec)
		print('motif_query_vec_ori:%d, peak_loc_ori:%d, motif_query_vec:%d, peak_query_vec:%d'%(len(motif_query_vec_ori),len(peak_loc_ori),motif_query_num,peak_query_num))

		motif_data = motif_data.loc[peak_query_vec,motif_query_vec]
		motif_query_vec_ori = np.unique(motif_data.columns)
		motif_query_num_ori = len(motif_query_vec_ori)
		peak_loc_ori = motif_data.index
		peak_loc_num = len(peak_loc_ori)
		print('motif_data ',motif_data.shape)

		min_peak_number = 1
		field_query1 = ['coef_1','coef_std','coef_quantile','coef_mean_deviation']
		quantile_vec_1 = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
		field_query2 = ['max','min','mean','median']+quantile_vec_1
		df_motif_score_basic1 = pd.DataFrame(index=motif_query_vec,columns=field_query1,dtype=np.float32)
		df_motif_access_basic1 = pd.DataFrame(index=motif_query_vec,columns=field_query2,dtype=np.float32)
		# df_motif_access_basic2 = pd.DataFrame(index=motif_query_vec,columns=field_query2,dtype=np.float32)
		b = 0.75
		thresh_pre1 = 1
		coef_motif_score_combination = -np.log(b)
		print('coefficient for motif score combination ',coef_motif_score_combination)
		dict_motif_query = dict()
		flag_query1=1
		save_mode=1

		column_id_query = 'max_accessibility_score'
		load_mode = 0
		if (len(df_peak_annot)>0) and (column_id_query in df_peak_annot.columns):
			load_mode = 1  # query maximum peak accessibility from peak annotation

		list_motif_score = []
		field_query_1 = ['motif_score_minmax','motif_score_log_normalize_bound']
		if flag_query1>0:
			motif_query_num_ori = len(motif_query_vec_ori)
			motif_query_num = motif_query_num_ori

			for i1 in range(motif_query_num):
				motif_id1 = motif_query_vec_ori[i1]
				id1 = motif_data.loc[:,motif_id1]>0
				peak_id1 = peak_loc_ori[id1]
				motif_score = motif_data_score.loc[peak_id1,motif_id1]

				## normalize motif score
				motif_score_minmax = motif_score/np.max(motif_score) # normalize motif_score per motif
				motif_score_log = np.log(1+motif_score) # log transformation of motif score
				score_compare = np.quantile(motif_score_log,0.95)
				motif_score_log_normalize = motif_score_log/np.max(motif_score_log) # normalize motif_score per motif
				motif_score_log_normalize_bound = motif_score_log/score_compare
				motif_score_log_normalize_bound[motif_score_log_normalize_bound>1] = 1.0

				field_query = field_query_1
				list1 = [motif_score_minmax,motif_score_log_normalize_bound]
				if flag_motif_score_quantile>0:
					t_vec_1 = np.asarray(motif_score)[:,np.newaxis]
					normalize_type = 'uniform'
					score_mtx = quantile_transform(t_vec_1,n_quantiles=1000,output_distribution=normalize_type)
					motif_score_quantile = score_mtx[:,0]

					field_query = field_query_1+['motif_score_quantile']
					list1 = list1 + [motif_score_quantile]
				
				dict1 = dict(zip(field_query,list1))
				df_motif_score_2 = pd.DataFrame.from_dict(data=dict1,orient='columns',dtype=np.float32)
				df_motif_score_2['peak_id'] = peak_id1
				df_motif_score_2.index = np.asarray(peak_id1)
			
				## normalize peak accessibility for peak loci with TF motif
				if load_mode==0:
					max_accessibility_score = peak_read.loc[:,peak_id1].max(axis=0)
				else:
					max_accessibility_score = df_peak_annot.loc[peak_id1,column_id_query]

				# max_accessibility_score_celltype = peak_read_celltype.loc[:,peak_id1].max(axis=0)
				t_value_1 = utility_1.test_stat_1(max_accessibility_score,quantile_vec=quantile_vec_1)
				df_motif_access_basic1.loc[motif_id1,field_query2] = np.asarray(t_value_1)

				median_access_value = df_motif_access_basic1.loc[motif_id1,'median']
				max_access_value = df_motif_access_basic1.loc[motif_id1,'max']
				# thresh_score_accessibility = median_value
				thresh_score_accessibility = median_access_value
				# if median_value<0.1:
				if median_access_value<0.01:
					thresh_score_accessibility = df_motif_accessibility_1.loc[motif_id1,0.75]
				# b2_score = 0.95
				b2_score = 0.90
				# a1 = 1.0/(thresh_pre1)*np.log((1+b)/(1-b))
				# score_accessility = 1-2/(1+np.exp(a1*max_accessibility_score))
				# a2 = -np.log(1-b2_score)/(thresh_pre2)
				a2 = -np.log(1-b2_score)/(thresh_score_accessibility)
				score_accessibility = 1-np.exp(-a2*max_accessibility_score) # y=1-exp(-ax)
				score_accessibility_minmax = max_accessibility_score/max_access_value
				if verbose>0:
					if i1%100==0:
						# print('df_motif_score_2, mean_value ',df_motif_score_2.shape,np.asarray(np.mean(df_motif_score_2.loc[:,field_query],axis=0)),i1,motif_id1)
						print('df_motif_score_2, mean_value ',df_motif_score_2.shape,np.asarray(df_motif_score_2.mean(axis=0,numeric_only=True)),i1,motif_id1)
						# print('median_value, thresh_score_accessibility, b2_score, a2 ',median_value,thresh_pre2,b2_score,a2,i1,motif_id1)
						print('median_value, thresh_score_accessibility, b2_score, a2 ',median_access_value,thresh_score_accessibility,b2_score,a2,i1,motif_id1)
						print('score_accessibility_minmax ',motif_id1,i1,score_accessibility_minmax.max(),score_accessibility_minmax.min(),score_accessibility_minmax.idxmax(),score_accessibility_minmax.idxmin(),score_accessibility_minmax.mean(),score_accessibility_minmax.median())
				
				lower_bound = 0.5
				score_1 = minmax_scale(score_accessibility*motif_score_log_normalize_bound,[lower_bound,1])
				df_motif_score_2['max_accessibility_score'] = max_accessibility_score
				df_motif_score_2['score_accessibility'] = score_accessibility
				df_motif_score_2['score_accessibility_minmax'] = score_accessibility_minmax
				df_motif_score_2['score_1'] = score_1
				df_motif_score_2.index = [motif_id1]*df_motif_score_2.shape[0]
				list_motif_score.append(df_motif_score_2)

				## query basic statistics of motif score on motif score variation
				if flag_motif_score_basic>0:
					# max_value, min_value, mean_value, median_value = np.max(motif_score), np.min(motif_score), np.mean(motif_score), np.median(motif_score)
					t_value_2 = utility_1.test_stat_1(motif_score,quantile_vec=quantile_vec_1)
					max_value, min_value, mean_value, median_value = t_value_2[0:4]
					coef_1 = (max_value-min_value)/(max_value+min_value)
					coef_std = np.std(motif_score)/mean_value
					Q1, Q3 = np.quantile(motif_score,0.25), np.quantile(motif_score,0.75)
					coef_quantile = (Q3-Q1)/(Q1+Q3)
					coef_mean_deviation = np.mean(np.abs(motif_score-mean_value))/mean_value
					# coef_query = coef_quantile
					if verbose>0:
						print('motif_id1, peak_id1 ',i1,motif_id1,len(peak_id1))
						print('coef_1, coef_std, coef_quantile, coef_mean_deviation ',coef_1,coef_std,coef_quantile,coef_mean_deviation,i1,motif_id1)
					df_motif_score_basic1.loc[motif_id1,field_query1] = [coef_1,coef_std,coef_quantile,coef_mean_deviation]
					df_motif_score_basic1.loc[motif_id1,field_query2] = np.asarray(t_value_2)

			df_motif_score_query = pd.concat(list_motif_score,axis=0,join='outer',ignore_index=False)		
			if save_mode==1:
				output_filename1 = '%s/test_motif_score_basic1.%s.1.txt'%(output_file_path,filename_annot)
				df_motif_score_basic1.to_csv(output_filename1,sep='\t',float_format='%.6f')
				# output_filename2 = '%s/test_motif_score_normalize.pre_compute.%s.npy'%(output_file_path,filename_annot)
				# np.save(output_filename2,dict_motif_query,allow_pickle=True)
				output_filename2 = '%s/test_motif_score_normalize.pre_compute.%s.txt'%(output_file_path,filename_annot)
				df_motif_score_query.to_csv(output_filename2,sep='\t',float_format='%.5f')
				output_filename3 = '%s/test_motif_access_basic1.%s.1.txt'%(output_file_path,filename_annot)
				df_motif_access_basic1.to_csv(output_filename3,sep='\t',float_format='%.6f')

			return df_motif_score_query, df_motif_score_basic1, df_motif_access_basic1

	## load dataframe
	def test_load_peak_gene_query(self,input_filename='',column_vec=[],select_config={}):

		df_gene_peak_query = pd.read_csv(input_filename,index_col=0,sep='\t')
		if len(column_vec)>0:
			df_gene_peak_query.index = self.test_query_index(df_gene_peak_query,column_vec=column_vec)

		return df_gene_peak_query

	## ====================================================
	# load the ATAC-seq and RNA-seq datad
	def test_query_load_data_1(self,atac_ad=[],rna_ad=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],motif_data=[],motif_data_score=[],flag_format=1,save_mode=1,verbose=0,select_config={}):

		self.atac_meta_ad = atac_ad
		self.rna_meta_ad = rna_ad
		self.peak_read = peak_read
		self.rna_exprs = rna_exprs
		self.rna_exprs_unscaled = rna_exprs_unscaled

		if len(motif_data)>0:
			self.motif_data = motif_data
			data_file_type_query = select_config['data_file_type_query']
			# if data_file_type_query in ['CD34_bonemarrow','pbmc']:
			if flag_format>0:
				motif_query_vec = motif_data.columns.str.upper()
				gene_query_name_ori = rna_ad.var_names.str.upper()
			else:
				motif_query_vec = motif_data.columns
				gene_query_name_ori = rna_ad.var_names
			
			motif_query_name_expr = pd.Index(motif_query_vec).intersection(gene_query_name_ori,sort=False)
			motif_query_num_1, gene_query_num_1, motif_query_num1 = len(motif_query_vec),len(gene_query_name_ori),len(motif_query_name_expr)
			print('motif_query_vec, gene_query_name_ori, motif_query_name_expr: ',motif_query_num_1, gene_query_num_1, motif_query_num1)
			self.motif_query_name_expr = motif_query_name_expr
		
		if len(motif_data_score)>0:
			self.motif_data_score = motif_data_score

	## ====================================================
	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_gene_pre2(self,gene_query_vec=[],motif_query_vec=[],df_gene_peak_query=[],peak_distance_thresh=500,
														df_peak_query=[],peak_loc_query=[],df_gene_tf_corr_peak=[],
														atac_ad=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],
														motif_data=[],motif_data_score=[],dict_motif_data=[],
														interval_peak_corr=50,interval_local_peak_corr=10,annot_mode=1,flag_load_pre1=0,flag_load_1=0,overwrite_1=False,overwrite_2=False,parallel_mode=1,
														input_file_path='',save_mode=1,filename_prefix_save='',output_filename='',output_file_path='',verbose=0,select_config={}):

		# file_path1 = self.save_path_1
		## provide file paths
		input_file_path1 = self.save_path_1
		data_file_type_ori = select_config['data_file_type']
		data_file_type_query = select_config['data_file_type_query']
		data_file_type = data_file_type_query
		data_file_type_annot = data_file_type_query.lower()
		
		run_id = select_config['run_id']
		type_id_feature = select_config['type_id_feature']
		# input_file_path = '%s/metacell_%d/run%d'%(data_dir,metacell_num,run_id)
		file_save_path = select_config['data_path_save']
		input_file_path = file_save_path
		print('input_file_path: %s'%(input_file_path))
		if output_file_path=='':
			output_file_path_ori = file_save_path
		output_file_path_ori = output_file_path
		
		# flag_load_pre1=0
		if flag_load_pre1>0:
			## query ATAC-seq and RNA-seq normalized read counts
			# select_config.update({'data_path_2':input_file_path_2})
			# peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1(peak_read=[],meta_exprs=[],select_config=select_config)
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori(peak_read=[],meta_exprs=[],select_config=select_config)
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.peak_read = peak_read

			## load gene annotations
			flag_gene_annot_query=0
			if flag_gene_annot_query>0:
				print('load gene annotations')
				start = time.time()
				self.test_gene_annotation_query1(select_config=select_config)
				stop = time.time()
				print('used: %.5fs'%(stop-start))
			
		# flag_load_1=0
		if len(motif_data)==0:
			flag_load_1 = 1
		if flag_load_1>0:
			self.test_motif_peak_estimate_load_1(peak_read=peak_read,rna_exprs=rna_exprs,rna_exprs_unscaled=rna_exprs_unscaled,
													dict_motif_data=dict_motif_data,select_config=select_config)

		if (len(motif_data)==0) or (flag_load_1==1):
			motif_data = self.motif_data
			motif_data_score = self.motif_data_score

		print('motif_data, motif_data_score: ',motif_data.shape,motif_data_score.shape)
		print(motif_data[0:2],motif_data_score[0:2])
		
		peak_loc_ori = motif_data.index
		motif_query_name_expr = motif_data.columns
		if len(motif_query_vec)==0:
			motif_query_vec = motif_query_name_expr
		
		motif_query_num = len(motif_query_vec)
		print('motif_query_vec: %d'%(motif_query_num))
		# motif_query_vec = motif_query_vec[0:5]

		if len(peak_read)==0:
			peak_read = self.peak_read

		if len(rna_exprs)==0:
			meta_scaled_exprs = self.meta_scaled_exprs
			meta_exprs_2 = self.meta_exprs_2
			rna_exprs = meta_scaled_exprs
			rna_exprs_unscaled = meta_exprs_2

		sample_id = rna_exprs.index
		rna_exprs_unscaled = rna_exprs_unscaled.loc[sample_id,:]
		peak_read = peak_read.loc[sample_id,:]
	
		## peak accessibility-TF expr correlation
		filename_prefix_1 = filename_prefix_save
		# correlation_type = 'spearmanr'
		correlation_type = select_config['correlation_type']
		# flag_query1=0
		flag_query_peak_basic=0

		# flag_motif_score_normalize_2=1
		flag_query_peak_ratio=1
		flag_query_distance=0
		
		flag_peak_tf_corr = 1
		flag_gene_tf_corr = 1
		flag_motif_score_normalize = 1
		flag_gene_tf_corr_peak_compute = 1
		flag_gene_tf_corr_peak_combine = 0
		field_query = ['flag_query_peak_basic','flag_query_peak_ratio',
						'flag_gene_tf_corr','flag_peak_tf_corr','flag_motif_score_normalize',
						'flag_gene_tf_corr_peak_compute','flag_gene_tf_corr_peak_combine']

		list1 = [flag_query_peak_basic,flag_query_peak_ratio,
				flag_gene_tf_corr,flag_peak_tf_corr,flag_motif_score_normalize,
				flag_gene_tf_corr_peak_compute,flag_gene_tf_corr_peak_combine]

		select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=list1,overwrite=False,select_config=select_config)
			
		flag_query_peak_basic,flag_query_peak_ratio,flag_gene_tf_corr,flag_peak_tf_corr,flag_motif_score_normalize,flag_gene_tf_corr_peak_compute,flag_gene_tf_corr_peak_combine = param_vec
		for (field_id,query_value) in zip(field_query,param_vec):
			print('field_id, query_value: ',field_id,query_value)

		if flag_motif_score_normalize>0:
			flag_motif_score_normalize_1=1
			# flag_motif_score_normalize_1=0
			flag_motif_score_normalize_1_query=1
			# flag_motif_score_normalize_1_query=0
			flag_motif_score_normalize_2=1
			# flag_motif_score_normalize_2=0
		else:
			flag_motif_score_normalize_1=0
			flag_motif_score_normalize_1_query=0
			flag_motif_score_normalize_2=0

		if 'flag_motif_score_normalize_2' in select_config:
			flag_motif_score_normalize_2 = select_config['flag_motif_score_normalize_2']

		## query tf binding activity score
		# flag_query6_5=0
		## re-compute gene_tf_corr_peak for peak-gene link query added
		flag_query_recompute=0
	
		flag_motif_score_normalize_thresh1 = flag_motif_score_normalize_1_query
		# overwrite_1 = False
		# overwrite_2 = False
		df_gene_peak_query1 = []
		df_link_query_1 = self.test_gene_peak_query_correlation_gene_pre2_compute_1(gene_query_vec=gene_query_vec,
																						motif_query_vec=motif_query_vec,
																						df_gene_peak_query=df_gene_peak_query,
																						peak_distance_thresh=peak_distance_thresh,
																						df_peak_query=[],
																						peak_loc_query=[],
																						atac_ad=atac_ad,
																						peak_read=peak_read,
																						rna_exprs=rna_exprs,
																						rna_exprs_unscaled=rna_exprs_unscaled,
																						motif_data=motif_data,
																						motif_data_score=motif_data_score,
																						flag_query_peak_basic=flag_query_peak_basic,
																						flag_peak_tf_corr=flag_peak_tf_corr,
																						flag_gene_tf_corr=flag_gene_tf_corr,
																						flag_motif_score_normalize_1=flag_motif_score_normalize_1,
																						flag_motif_score_normalize_thresh1=flag_motif_score_normalize_thresh1,
																						flag_motif_score_normalize_2=flag_motif_score_normalize_2,
																						flag_gene_tf_corr_peak_compute=flag_gene_tf_corr_peak_compute,
																						interval_peak_corr=50,
																						interval_local_peak_corr=10,
																						annot_mode=annot_mode,
																						overwrite_1=overwrite_1,
																						overwrite_2=overwrite_2,
																						save_mode=save_mode,
																						filename_prefix_save=filename_prefix_save,
																						output_filename='',
																						output_file_path=output_file_path,
																						verbose=verbose,
																						select_config=select_config)

		df_gene_tf_corr_peak_1 = df_link_query_1
		df_gene_peak_query1 = df_gene_tf_corr_peak_1
		
		type_id_query = 2
		type_id_compute = 1
		if flag_gene_tf_corr_peak_combine>0:
			self.test_gene_peak_query_correlation_gene_pre2_combine_1(gene_query_vec=[],
																		peak_distance_thresh=peak_distance_thresh,
																		df_peak_query=[],
																		type_id_query=type_id_query,type_id_compute=type_id_compute,
																		save_mode=save_mode,
																		filename_prefix_save=filename_prefix_save,
																		input_file_path='',
																		output_file_path=output_file_path,
																		output_filename='',
																		verbose=verbose,
																		select_config=select_config)

		flag_save_interval = 0
		if len(df_gene_tf_corr_peak)>0:
			df_gene_tf_corr_peak_pre1 = df_gene_tf_corr_peak
		else:
			df_gene_tf_corr_peak_pre1 = df_gene_tf_corr_peak_1
		
		flag_init_score=1
		if 'flag_score_pre1' in select_config:
			flag_score_pre1 = select_config['flag_score_pre1']
			flag_init_score = (flag_score_pre1 in [2,3])

		if flag_init_score>0:
			df_gene_peak_query1 = self.test_gene_peak_query_correlation_gene_pre2_compute_3(gene_query_vec=[],df_gene_peak_query=df_gene_peak_query,peak_distance_thresh=peak_distance_thresh,df_peak_query=[],peak_loc_query=[],df_gene_tf_corr_peak=df_gene_tf_corr_peak_pre1,
																							flag_save_ori=0,flag_save_interval=flag_save_interval,parallel_mode=parallel_mode,save_mode=1,filename_prefix_save=filename_prefix_save,output_filename='',output_file_path=output_file_path,
																							verbose=verbose,select_config=select_config)

		return df_gene_peak_query1

	## ====================================================
	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_gene_pre2_compute_1(self,gene_query_vec=[],
																motif_query_vec=[],
																df_gene_peak_query=[],
																peak_distance_thresh=500,
																df_peak_query=[],
																filename_prefix_save='',
																output_filename='',
																peak_loc_query=[],
																atac_ad=[],
																peak_read=[],
																rna_exprs=[],
																rna_exprs_unscaled=[],
																motif_data=[],
																motif_data_score=[],
																flag_query_peak_basic=0,
																flag_peak_tf_corr=0,
																flag_gene_tf_corr=0,
																flag_motif_score_normalize_1=0,
																flag_motif_score_normalize_thresh1=0,
																flag_motif_score_normalize_2=0,
																flag_gene_tf_corr_peak_compute=0,
																interval_peak_corr=50,
																interval_local_peak_corr=10,
																annot_mode=1,
																overwrite_1=False,
																overwrite_2=False,
																save_mode=1,
																output_file_path='',
																verbose=0,
																select_config={}):

		file_save_path_1 = select_config['data_path_save']
		if output_file_path=='':
			output_file_path = file_save_path_1
		file_save_path = output_file_path
		
		## peak accessibility-TF expr correlation estimation
		data_file_type = select_config['data_file_type']
		data_file_type_query = select_config['data_file_type_query']
		output_file_path_default = file_save_path

		filename_prefix_save_default = select_config['filename_prefix_save_default']
		filename_prefix_save_pre1 = filename_prefix_save_default
		# filename_annot_1 = data_file_type
		if 'filename_annot_save_default' in select_config:
			filename_annot_default = select_config['filename_annot_save_default']
		else:
			filename_annot_default = data_file_type_query
			select_config.update({'filename_annot_save_default':filename_annot_default})
		# filename_annot_1 = filename_annot_default
		filename_annot = filename_annot_default

		if 'file_path_motif_score' in select_config:
			file_path_motif_score = select_config['file_path_motif_score']
		else:
			file_path_motif_score = file_save_path

		file_save_path2 = file_path_motif_score
		input_file_path = file_save_path2
		input_file_path2 = file_save_path2
		output_file_path = file_save_path2
		## query the ratio of open peak loci
		# query the maximum accessibility of peak loci in the metacells		
		# flag_query_peak_basic=0
		df_peak_annot = self.atac_meta_ad.var.copy()
		if flag_query_peak_basic>0:
			# type_id_peak_ratio=2
			type_id_peak_ratio=0
			if type_id_peak_ratio in [0,2]:
				print('query the ratio of open peak loci')
				df_peak_access_basic_1, quantile_value_1 = self.test_peak_access_query_basic_1(peak_read=peak_read,
																								rna_exprs=rna_exprs,
																								df_annot=df_peak_annot,
																								thresh_value=0.1,
																								flag_ratio=1,
																								flag_access=1,
																								save_mode=1,
																								filename_annot=filename_annot,
																								output_file_path=output_file_path,
																								output_filename='',
																								select_config=select_config)
				self.peak_annot = df_peak_access_basic_1
				
			if type_id_peak_ratio in [1,2]:
				low_dim_embedding = 'X_svd'
				# pval_cutoff = 1e-2
				n_neighbors = 3
				# bin_size = 500
				# bin_size = 5000
				atac_meta_ad = self.atac_meta_ad
				# print('atac_meta_ad \n',atac_meta_ad)
				atac_meta_ad = self.test_peak_access_query_basic_pre1(adata=atac_meta_ad,low_dim_embedding=low_dim_embedding,n_neighbors=n_neighbors,select_config=select_config)
				self.atac_meta_ad = atac_meta_ad

		self.dict_peak_tf_corr_ = dict()
		# filename_annot = filename_annot_default
		if 'filename_annot_motif_score' in select_config:
			filename_annot = select_config['filename_annot_motif_score']
		else:
			filename_annot = filename_annot_default
		
		filename_motif_score_1 = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path,filename_annot)
		filename_motif_score_2 = '%s/test_motif_score_normalize.pre_compute.%s.txt'%(input_file_path,filename_annot)
		select_config.update({'filename_motif_score_normalize_1':filename_motif_score_1,
								'filename_motif_score_normalize_2':filename_motif_score_2})

		# flag_peak_tf_corr=0
		if 'flag_peak_tf_corr' in select_config:
			flag_peak_tf_corr = select_config['flag_peak_tf_corr']
		
		correlation_type=select_config['correlation_type']
		filename_prefix = 'test_peak_tf_correlation.%s'%(data_file_type_query)
		if 'filename_prefix_peak_tf' in select_config:
			filename_prefix_peak_tf = select_config['filename_prefix_peak_tf']
			filename_prefix = filename_prefix_peak_tf

		field_load = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected']
		filename_annot_vec = [correlation_type,'pval','pval_corrected','motif_basic']
		if flag_peak_tf_corr>0:
			save_mode = 1
			flag_load_1 = 0
			if 'flag_load_peak_tf' in select_config:
				flag_load_1 = select_config['flag_load_peak_tf']
			
			# field_load = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected']
			# filename_annot_vec = [correlation_type,'pval','pval_corrected','motif_basic']
			input_filename_list1 = ['%s/%s.%s.1.copy1.txt'%(file_save_path,filename_prefix,filename_annot1) for filename_annot1 in filename_annot_vec[0:3]]
			dict_peak_tf_corr_ = self.test_peak_tf_correlation_query_1(motif_data=motif_data,
																		peak_query_vec=[],
																		motif_query_vec=motif_query_vec,
																		peak_read=peak_read,
																		rna_exprs=rna_exprs,
																		correlation_type=correlation_type,
																		flag_load=flag_load_1,field_load=field_load,
																		save_mode=save_mode,
																		input_filename_list=input_filename_list1,
																		output_file_path=output_file_path_default,
																		filename_prefix=filename_prefix,
																		select_config=select_config)
			self.dict_peak_tf_corr_ = dict_peak_tf_corr_
			# select_config = self.select_config

		## motif score normalize 1 estimation: in silico ChIP-seq
		# flag_motif_score_normalize_1=0
		# select_config_1 = self.select_config
		if not ('filename_peak_tf_corr' in select_config):
			print('please provide peak accessibility-TF expression correlation file')
		else:
			# input_filename_1 = self.select_config['filename_peak_tf_corr']
			input_filename_1 = select_config['filename_peak_tf_corr']
		
		df_peak_annot = self.df_peak_annot
		beta_mode = 0
		if 'beta_mode' in select_config:
			beta_mode = select_config['beta_mode']

		if flag_motif_score_normalize_1>0:
			print('normalization of TF motif score of in silico ChIP-seq')
			start = time.time()
			output_filename = filename_motif_score_1
			dict_peak_tf_corr_ = self.dict_peak_tf_corr_
			df_peak_tf_expr_corr_ = []
			if len(dict_peak_tf_corr_)>0:
				df_peak_tf_expr_corr_ = dict_peak_tf_corr_['peak_tf_corr']

			# the in silico ChIP-seq library method
			# TF binding score with normalization
			df_pre1 = self.test_peak_tf_score_normalization_1(peak_query_vec=[],motif_query_vec=motif_query_vec,
																motif_data=motif_data,
																motif_data_score=motif_data_score,
																df_peak_tf_expr_corr_=df_peak_tf_expr_corr_,
																input_filename=input_filename_1,
																peak_read=peak_read,
																rna_exprs=rna_exprs,
																peak_read_celltype=[],
																df_peak_annot=df_peak_annot,
																filename_annot=filename_annot_default,
																overwrite=overwrite_1,
																output_file_path=output_file_path,
																output_filename=output_filename,
																beta_mode=beta_mode,
																verbose=verbose,
																select_config=select_config)

			stop = time.time()
			if len(df_pre1)>0:
				print('normalization of TF motif score used: %.5fs'%(stop-start))
			else:
				print('please provide the information for normalization of TF motif score')

		## TF binding prediction by in silico ChIP-seq
		# flag_query5=0
		# flag_motif_score_normalize_1_query=0
		flag_motif_score_normalize_1_query = flag_motif_score_normalize_thresh1
		quantile_vec_1 = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
		field_query_1 = ['max','min','mean','median']+quantile_vec_1
		output_file_path = file_save_path2
		# if flag_query5>0:
		if flag_motif_score_normalize_1_query>0:
			# peak-TF link query by threshold from the TF motif score estimated using the in silico ChIP-seq library method
			thresh_score_1 = 0.10
			if 'thresh_insilco_ChIP-seq' in select_config:
				thresh_score_1 = select_config['thresh_insilco_ChIP-seq']

			# input_filename = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path,filename_annot_1)
			input_filename = select_config['filename_motif_score_normalize_1']
			b = input_filename.find('.txt')
			# output_filename = '%s/test_motif_score_normalize_insilico.%s.thresh%s.txt'%(output_file_path,filename_annot_1,thresh_score_1)
			output_filename = '%s.thresh%s.txt'%(input_filename[0:b],thresh_score_1)
			select_config.update({'filename_motif_score_normalize_1_thresh1':output_filename})
			df_query_1 = self.test_peak_tf_score_normalization_query_1(gene_query_vec=[],peak_query_vec=[],motif_query_vec=motif_query_vec,
																		df_gene_peak_query=[],motif_data=[],
																		input_filename=input_filename,
																		thresh_score=thresh_score_1,
																		peak_read=peak_read,
																		rna_exprs=rna_exprs,
																		filename_annot=filename_annot_default,
																		output_filename=output_filename,
																		select_config=select_config)
			
		## motif score normalize 2 estimation
		# flag_query2=0
		# flag_motif_score_normalize_2=0
		if flag_motif_score_normalize_2>0:
			# output_file_path = input_file_path2
			filename_annot = filename_annot_default
			print('normalization of TF motif score')
			start = time.time()
			# input_file_path = file_save_path2
			input_filename = '%s/test_motif_score_normalize.pre_compute.%s.txt'%(output_file_path,filename_annot)
			if (os.path.exists(input_filename)==True) and (overwrite_2==False):
				print('the file exists: %s'%(input_filename))
				# print('overwrite_2: ',overwrite_2)
				df_motif_score_query = pd.read_csv(input_filename,index_col=0,sep='\t')
			else:
				# TF motif score normalization
				df_motif_score_query = self.test_peak_tf_score_normalization_pre_compute(gene_query_vec=[],
																							peak_query_vec=[],
																							motif_query_vec=motif_query_vec,
																							df_gene_peak_query=[],
																							motif_data=motif_data,
																							motif_data_score=motif_data_score,
																							peak_read=peak_read,
																							rna_exprs=rna_exprs,
																							peak_read_celltype=[],
																							df_peak_annot=df_peak_annot,
																							filename_annot=filename_annot,
																							output_file_path=output_file_path,
																							select_config=select_config)
			self.df_motif_score_query = df_motif_score_query
			stop = time.time()
			print('normalization of TF motif score used: %.5fs'%(stop-start))
			
		df_gene_peak = df_gene_peak_query
		column_idvec = ['gene_id','peak_id']
		column_id1, column_id2 = column_idvec[0:2]

		df_gene_tf_corr_peak = []
		if flag_gene_tf_corr_peak_compute>0:
			if len(gene_query_vec)==0:
				# gene_query_vec = df_gene_peak['gene_id'].unique()
				gene_query_vec = df_gene_peak[column_id1].unique()

			query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']

			gene_query_num_1 = len(gene_query_vec)
			start_id1, start_id2 = -1, -1
			if (query_id1>=0) and (query_id2>query_id1):
				start_id1, start_id2 = query_id1, np.min([query_id2,gene_query_num_1])
				if start_id1>start_id2:
					print('start_id1, start_id2: ',start_id1,start_id2)
					return
			if start_id1>=0:
				filename_annot_save = '%d_%d'%(start_id1,start_id2)
			else:
				filename_annot_save = '1'
			
			filename_prefix_save_1 = select_config['filename_prefix_default_1']
			if 'filename_prefix_save_2' in select_config:
				filename_prefix_save_2 = select_config['filename_prefix_save_2']
				filename_prefix_save_pre1 = filename_prefix_save_2
			else:
				filename_prefix_save_pre1 = filename_prefix_save_1

			# output_filename = '%s/%s.pcorr_query1.%s.txt'%(output_file_path,filename_prefix_save_1,filename_annot_save)
			output_filename = '%s/%s.pcorr_query1.%s.txt'%(output_file_path,filename_prefix_save_pre1,filename_annot_save)
			self.select_config = select_config
			filename_query = output_filename
			self.select_config.update({'filename_gene_tf_cond':filename_query})

			filename_list_pcorr_1 = []
			if 'filename_list_pcorr_1' in select_config:
				filename_list_pcorr_1 = select_config['filename_pcorr_1']
			filename_list_pcorr_1.append(output_filename)
			select_config.update({'filename_list_pcorr_1':filename_list_pcorr_1})

			gene_query_vec_pre1 = gene_query_vec
			gene_query_vec = gene_query_vec_pre1[start_id1:start_id2]
			gene_query_num = len(gene_query_vec)
			print('gene_query_vec_pre1, gene_query_vec ',gene_query_num_1,gene_query_num,start_id1,start_id2)
			
			# flag_query_2=0
			flag_gene_tf_corr_peak_1=1
			flag_gene_tf_corr_peak_pval=1

			# df_gene_tf_corr_peak.index = np.asarray(df_gene_tf_corr_peak['gene_id'])
			# self.df_gene_tf_corr_peak = df_gene_tf_corr_peak
			if flag_gene_tf_corr_peak_1>0:
				## estimate gene-TF expresssion partial correlation given peak accessibility
				print('estimate gene-TF expression partial correlation given peak accessibility')
				start = time.time()
				peak_query_vec, motif_query_vec = [], []
				type_id_query_2 = 2
				type_id_compute = 1
				motif_query_name_expr = self.motif_query_name_expr
				motif_data = self.motif_data
				motif_data_expr = motif_data.loc[:,motif_query_name_expr]
				df_gene_tf_corr_peak = self.test_query_score_function1(df_gene_peak_query=df_gene_peak,motif_data=motif_data_expr,gene_query_vec=gene_query_vec,peak_query_vec=peak_query_vec,
																		motif_query_vec=motif_query_vec,
																		peak_read=peak_read,rna_exprs=rna_exprs,rna_exprs_unscaled=rna_exprs_unscaled,
																		type_id_query=type_id_query_2,type_id_compute=type_id_compute,
																		flag_peak_tf_corr=0,flag_gene_tf_corr_peak=flag_gene_tf_corr_peak_1,flag_pval_1=0,flag_pval_2=flag_gene_tf_corr_peak_pval,
																		save_mode=1,output_file_path=output_file_path,output_filename=output_filename,verbose=verbose,select_config=select_config)
				self.df_gene_tf_corr_peak = df_gene_tf_corr_peak
				print('df_gene_tf_corr_peak ',df_gene_tf_corr_peak.shape)
				print(df_gene_tf_corr_peak[0:5])

		flag_gene_tf_corr_1 = flag_gene_tf_corr
		if 'flag_gene_tf_corr' in select_config:
			flag_gene_tf_corr_1 = select_config['flag_gene_tf_corr']

		if flag_gene_tf_corr_1>0:
			gene_query_vec_pre1 = rna_exprs.columns
			feature_vec_1 = gene_query_vec_pre1
			motif_query_name_expr = self.motif_query_name_expr
			# motif_data = self.motif_data
			motif_query_vec_pre1 = motif_query_name_expr
			feature_vec_2 = motif_query_vec_pre1

			input_file_path = select_config['file_path_motif_score']
			data_file_type_query = select_config['data_file_type_query']
			filename_prefix_pre1 = data_file_type_query
			filename_prefix_1 = select_config['filename_prefix_default']
			filename_prefix_default_1 = select_config['filename_prefix_default_1']

			correlation_type = 'spearmanr'
			correlation_type_vec = [correlation_type]
			# filename_prefix_save_1 = '%s.gene_tf_corr'%(filename_prefix_1)

			query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
			print('query_id1:%d, query_id2:%d'%(query_id1,query_id2))
			
			if (query_id1>=0) and (query_id2>query_id1):
				if 'feature_query_num_1' in select_config:
					feature_query_num_1 = select_config['feature_query_num_1']
					if query_id1>feature_query_num_1:
						print('query_id1, feature_query_num_1: ',query_id1,feature_query_num_1)
						return
					else:
						query_id2 = np.min([query_id2,feature_query_num_1])

				input_filename = '%s/%s.pcorr_query1.%d_%d.txt'%(input_file_path,filename_prefix_default_1,query_id1,query_id2)
				df_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				feature_vec_1 = df_query_1[column_id1].unique()
				filename_prefix_save_1 = 'test_gene_tf_correlation.%s.%d_%d'%(filename_prefix_pre1,query_id1,query_id2)
			else:
				filename_prefix_save_1 = 'test_gene_tf_correlation.%s'%(filename_prefix_pre1)

			print('compute gene-TF expression correlation')
			print('feature_vec_1, feature_vec_2: ',len(feature_vec_1),len(feature_vec_2))
			start = time.time()
			
			filename_save_1 = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_save_1,correlation_type)
			overwrite_2 = False
			if (os.path.exists(filename_save_1)==True) and (overwrite_2==False):
				print('the file exists: %s'%(filename_save_1))
			else:
				self.test_gene_expr_correlation_1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,peak_read=peak_read,rna_exprs=rna_exprs,correlation_type_vec=correlation_type_vec,
													symmetry_mode=0,type_id_1=0,type_id_pval_correction=1,thresh_corr_vec=[],
													filename_prefix=filename_prefix_save_1,save_mode=1,save_symmetry=0,output_file_path=output_file_path,select_config=select_config)

			stop = time.time()
			print('compute gene-TF expression correlation used: %.2fs'%(stop-start))

		filename_annot2 = select_config['filename_save_annot_pre1']
		filename_prefix_query1 = 'test_gene_tf_expr_correlation.%s'%(filename_annot2)
		filename_prefix_query2 = 'test_gene_expr_correlation.%s'%(filename_annot2)
		input_filename_pre1 = '%s/%s.%s.1.txt'%(input_file_path2,filename_prefix_query1,correlation_type)
		input_filename_pre2 = '%s/%s.%s.pval_corrected.1.txt'%(input_file_path2,filename_prefix_query1,correlation_type)
		input_filename_1 = '%s/%s.%s.1.txt'%(input_file_path2,filename_prefix_query2,correlation_type)
		input_filename_2 = '%s/%s.%s.pval_corrected.1.txt'%(input_file_path2,filename_prefix_query2,correlation_type)
		select_config.update({'filename_gene_tf_expr_correlation':input_filename_pre1,
								'filename_gene_tf_expr_pval_corrected':input_filename_pre2,
								'filename_gene_expr_correlation':input_filename_1,
								'filename_gene_expr_pval_corrected':input_filename_2})

		return df_gene_tf_corr_peak

	## ====================================================
	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_gene_pre2_combine_1(self,gene_query_vec=[],
																peak_distance_thresh=500,
																df_peak_query=[],
																type_id_query=2,type_id_compute=1,
																save_mode=1,
																filename_prefix_save='',
																input_file_path='',
																output_file_path='',
																output_filename='',
																verbose=0,
																select_config={}):

		flag_query_3=0
		if flag_query_3>0:
			# flag_gene_tf_corr_peak_combine = 1
			flag_gene_tf_corr_peak_combine = 0
			
			## combine gene-TF expresssion partial correlation estimation from different runs
			if flag_gene_tf_corr_peak_combine>0:
				print('combine gene-TF expresssion partial correlation estimation from different runs')
				start = time.time()
				df1, df1_ratio = self.test_partial_correlation_gene_tf_cond_peak_combine_pre1(input_file_path=input_file_path,
																								filename_prefix_vec=[],
																								filename_prefix_save=filename_prefix_save,
																								type_id_query=type_id_query,
																								type_id_compute=type_id_compute,
																								save_mode=save_mode,output_file_path='',
																								output_filename='',output_filename_list=[],
																								overwrite=0,
																								select_config=select_config)
				print('df1, df1_ratio ',df1.shape,df1_ratio.shape)
				stop = time.time()
				print('combining gene-TF expresssion partial correlation estimation from different runs used %.5fs'%(stop-start))
				
				flag_save_subset_1=1
				if flag_save_subset_1>0:
					df1_1 = df1[0:10000]
					output_filename = '%s/%s.subset1.txt'%(output_file_path,filename_prefix_save_pre1)
					df1_1.to_csv(output_filename,sep='\t',float_format='%.5f')
																
	## ====================================================
	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_gene_pre2_compute_3(self,gene_query_vec=[],df_gene_peak_query=[],peak_distance_thresh=500,df_peak_query=[],peak_loc_query=[],df_gene_tf_corr_peak=[],
																flag_save_ori=0,flag_save_interval=0,parallel_mode=1,
																save_mode=1,filename_prefix_save='',output_file_path='',output_filename='',verbose=0,select_config={}):

		## initial calculation of peak-tf-gene association score
		print('initial calculation of peak-TF-gene association score')
		# flag_init_score=0
		flag_init_score=1
		if 'flag_score_pre1' in select_config:
			flag_score_pre1 = select_config['flag_score_pre1']
			flag_init_score = flag_score_pre1

		lambda1=0.5
		lambda2=1-lambda1
		# df_gene_peak_query_pre1 = df_gene_tf_corr_peak_pre1
		df_gene_peak_query1_1 = df_gene_peak_query
		if flag_init_score>0:
			# flag_load_1=0
			dict_peak_tf_query = self.dict_peak_tf_query
			dict_gene_tf_query = self.dict_gene_tf_query
			# flag_load_1 = (len(dict_peak_tf_query)==0)
			load_mode_2 = -1
			if len(dict_peak_tf_query)==0:
				load_mode_2 = 0

			if len(dict_gene_tf_query)==0:
				load_mode_2 = load_mode_2 + 2

			print('load_mode_2: ',load_mode_2)
			if load_mode_2>=0:
				# select_config.update({'filename_peak_tf_corr':filename_peak_tf_corr,'filename_gene_expr_corr':filename_gene_expr_corr})
				dict_peak_tf_query, dict_gene_tf_query = self.test_gene_peak_tf_query_score_init_pre1_1(gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],
																										load_mode=load_mode_2,save_mode=0,input_file_path='',output_file_path=output_file_path,output_filename='',verbose=verbose,select_config=select_config)

				if load_mode_2 in [0,2]:
					self.dict_peak_tf_query = dict_peak_tf_query

				if load_mode_2 in [1,2]:
					self.dict_gene_tf_query = dict_gene_tf_query

			flag_load_2=0
			if (self.df_peak_tf_1 is None):
				flag_load_2 = 1

			if flag_load_2>0:
				# input_file_path = select_config['file_path_motif_score']
				filename_annot = select_config['filename_annot_motif_score']
				input_file_path2 = select_config['file_path_motif_score']
				filename_motif_score_1 = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path2,filename_annot)
				filename_motif_score_2 = '%s/test_motif_score_normalize.pre_compute.%s.txt'%(input_file_path2,data_file_type_query)
				df_peak_tf_1 = pd.read_csv(filename_motif_score_1,index_col=0,sep='\t')
				df_peak_tf_2 = pd.read_csv(filename_motif_score_2,index_col=0,sep='\t')
				print('df_peak_tf_1, df_peak_tf_2: ',df_peak_tf_1.shape,df_peak_tf_2.shape)
				print(filename_motif_score_1)
				print(df_peak_tf_1[0:2])
				print(filename_motif_score_2)
				print(df_peak_tf_2[0:2])
				self.df_peak_tf_1 = df_peak_tf_1
				self.df_peak_tf_2 = df_peak_tf_2

			## gene_tf_corr_peak p-value query
			column_pval_cond = select_config['column_pval_cond']
			flag_gene_tf_corr_peak_pval = 0
			if not (column_pval_cond in df_gene_tf_corr_peak.columns):
				flag_gene_tf_corr_peak_pval = 1

			if flag_gene_tf_corr_peak_pval>0:
				print('df_gene_tf_corr_peak ',df_gene_tf_corr_peak.shape)
				print('estimate p-value corrected for gene-TF expression partial correlation given peak accessibility')
				start = time.time()
				df_gene_tf_corr_peak = self.test_gene_tf_corr_peak_pval_corrected_query_1(df_gene_peak_query=df_gene_tf_corr_peak,
																							gene_query_vec=[],
																							motif_query_vec=[],
																							parallel_mode=parallel_mode,
																							verbose=verbose,
																							select_config=select_config)
				stop = time.time()
				print('estimating p-value corrected for gene-TF expression partial correlation used %.5fs'%(stop-start))
				# print(df_gene_tf_corr_peak[0:5])
				if (save_mode>0) and (output_filename!=''):
					df_gene_tf_corr_peak.index = np.asarray(df_gene_tf_corr_peak['gene_id'])
					compression = None
					b = output_filename.find('.gz')
					if b>-1:
						compression = 'gzip'
					df_gene_tf_corr_peak.to_csv(output_filename,sep='\t',float_format='%.5f',compression=compression)

			df_gene_peak_query1_1 = self.test_gene_peak_tf_query_score_init_pre1(df_gene_peak_query=df_gene_peak_query,
																					df_gene_tf_corr_peak=df_gene_tf_corr_peak,
																					lambda1=lambda1,lambda2=lambda2,
																					type_id_1=0,column_id1=-1,
																					select_config=select_config)

			field_query_3 = ['motif_id','peak_id','gene_id','score_1','score_combine_1','score_pred1_correlation','score_pred1','score_pred1_1','score_pred2','score_pred_combine']
			if 'column_score_query_pre1' in select_config:
				field_query_3 = select_config['column_score_query_pre1']
			df_gene_peak_query1_1 = df_gene_peak_query1_1.loc[:,field_query_3]

		# flag_save_interval=0
		# flag_save_ori=0
		if flag_save_interval>0:
			if 'filename_prefix_save'=='':
				filename_prefix_save = select_config['filename_prefix_save_pre1']
			save_mode = flag_save_ori
			self.test_gene_peak_tf_query_score_init_save(df_gene_peak_query=df_gene_peak_query1_1,lambda1=lambda1,lambda2=lambda2,
															flag_init_score=0,flag_save_interval=flag_save_interval,
															feature_type='gene_id',query_mode=0,save_mode=save_mode,
															output_file_path='',output_filename='',filename_prefix_save=filename_prefix_save,float_format='%.5f',
															select_config=select_config)
		return df_gene_peak_query1_1

	## ====================================================
	# gene expression correlation query
	def test_gene_expr_correlation_1(self,feature_vec_1=[],feature_vec_2=[],peak_read=[],rna_exprs=[],correlation_type_vec=['spearmanr'],
											symmetry_mode=0,type_id_1=0,type_id_pval_correction=1,thresh_corr_vec=[],
											filename_prefix='',save_mode=1,save_symmetry=0,output_file_path='',select_config={}):

		# correlation_type = 'spearmanr'
		# correlation_type_vec = ['spearmanr']
		if len(rna_exprs)==0:
			rna_exprs = self.meta_scaled_exprs
		df_feature_query_1 = rna_exprs
		print('df_feature_query_1 ',df_feature_query_1)
		df_feature_query2 = []
		if len(feature_vec_1)>0:
			df_feature_query1 = df_feature_query_1.loc[:,feature_vec_1]
		else:
			df_feature_query1 = df_feature_query_1
			feature_vec_1 = df_feature_query_1.columns

		if len(feature_vec_2)>0:
			if list(np.unique(feature_vec_2))!=list(np.unique(feature_vec_1)):
				df_feature_query2 = df_feature_query_1.loc[:,feature_vec_2]
				symmetry_mode=0
			else:
				df_feature_query2 = []
				symmetry_mode=1
		print('df_feature_query1, df_feature_query2, symmetry_mode ',df_feature_query1.shape,len(df_feature_query2),symmetry_mode)
		start = time.time()
		feature_num1 = len(feature_vec_1)
		gene_num = feature_num1
		print('estimate gene expression correlation for %d genes'%(gene_num))

		file_path1 = self.save_path_1
		test_estimator1 = _Base2_correlation(file_path=file_path1)
		self.test_estimator1 = test_estimator1
		dict_query_1 = test_estimator1.test_feature_correlation_1(df_feature_query_1=df_feature_query1,
																	df_feature_query_2=df_feature_query2,
																	feature_vec_1=feature_vec_1,
																	feature_vec_2=feature_vec_2,
																	correlation_type_vec=correlation_type_vec,
																	symmetry_mode=symmetry_mode,
																	type_id_pval_correction=type_id_pval_correction,
																	type_id_1=type_id_1,
																	thresh_corr_vec=thresh_corr_vec,
																	filename_prefix=filename_prefix,
																	save_mode=save_mode,
																	save_symmetry=save_symmetry,
																	output_file_path=output_file_path,
																	select_config=select_config)

		stop = time.time()
		print('estimating gene expression correlation for %d genes used %.5fs'%(gene_num,stop-start))
			
		return dict_query_1

	## ====================================================
	# TF expression-gene expression partial correlation conditioned on peak accessibility
	# type_id_query: 1: the entire sample;
	#				 2: peak_accessibility>0 or (peak accessibility=0,gene_expr=0);
	#				 3: tf_expr>0;
	def test_partial_correlation_gene_tf_cond_peak_1(self,motif_data=[],gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],
															df_gene_peak_query=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],
															type_id_query=2,type_id_compute=1,parallel_mode=0,save_mode=1,output_filename='',verbose=0,select_config={}):

		input_file_path1 = self.save_path_1
		list_pre1 = []
		# type_id_query = 2  # type_id_query:0,1,2,3
		# type_id_compute = 0
		# motif_query_name_ori = motif_data.columns
		motif_query_name = motif_data.columns
		if len(motif_query_vec)>0:
			# motif_data_ori = motif_data.copy()
			motif_data = motif_data.loc[:,motif_query_vec]
		else:
			motif_query_vec = motif_data.columns

		# column_idvec = ['motif_id','peak_id','gene_id']
		column_idvec = select_config['column_idvec']
		column_id3, column_id2, column_id1 = column_idvec
		
		if len(peak_query_vec)==0:
			peak_query_vec = df_gene_peak_query[column_id2].unique()

		if len(gene_query_vec)==0:
			gene_query_vec = df_gene_peak_query[column_id1].unique()
		
		gene_query_num = len(gene_query_vec)
		peak_query_num = len(peak_query_vec)
		list_query1 = []
		list_query2 = []
		df_query1 = []
		thresh1, thresh2 = 0, 1E-05
		# df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
		df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])
		sample_id = rna_exprs.index
		sample_num = len(sample_id)
		peak_read = peak_read.loc[sample_id,:]
		rna_exprs_unscaled = rna_exprs_unscaled.loc[sample_id,:]
		df_peak_annot1 = pd.DataFrame(index=peak_query_vec,columns=['motif_num'])
		dict_motif = dict()
		motif_query_num = len(motif_query_vec)
		if type_id_query in [3]:
			for i1 in range(motif_query_num):
				motif_query1 = motif_query_vec[i1]
				tf_expr = rna_exprs_unscaled[motif_query1]
				id_tf = (tf_expr>thresh1)
				sample_id2 = sample_id[id_tf]
				dict_motif.update({motif_query1:id_tf})
				if (verbose>0) and (i1%100==0):
					print('motif_query1 ',motif_query1,i1,len(id_tf))

		# field_query = ['gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1']
		field_query = select_config['column_gene_tf_corr_peak']
		field_id1, field_id2, field_id3 = field_query
		
		print('field_id1:%s,field_id2:%s,field_id3:%s'%(field_id1,field_id2,field_id3))
		flag_1 = 0
		# field_query = ['motif_id','peak_id','gene_id']
		field_query = column_idvec
		for i1 in range(gene_query_num):
		# for i1 in range(20):
			gene_query_id = gene_query_vec[i1]
			# peak_loc_query = df_gene_peak_query.loc[[gene_query_id],'peak_id'].unique() # candidate peaks linked with gene;
			peak_loc_query = df_gene_peak_query.loc[[gene_query_id],column_id2].unique() # candidate peaks linked with gene;
			peak_num2 = len(peak_loc_query)
			Y_expr = rna_exprs[gene_query_id] # gene expr scaled
			Y_expr_unscaled = rna_exprs_unscaled[gene_query_id] # gene expr unscaled
			id_gene_expr = (Y_expr_unscaled>thresh1) # gene expr above zero
			if verbose>0:
				print('gene_id:%s, peak_loc:%d, %d'%(gene_query_id,peak_num2,i1))
				# print(peak_loc_query[0:2])
			for i2 in range(peak_num2):
				peak_id = peak_loc_query[i2]
				motif_idvec = motif_query_name[motif_data.loc[peak_id,:]>0] # TFs with motifs in the peak
				motif_num = len(motif_idvec)
				df_peak_annot1.loc[peak_id,'motif_num'] = motif_num
				# thresh1, thresh2 = 0, 1E-05
				Y_peak = peak_read[peak_id]
				id_peak_1 = (Y_peak>thresh1) # the samples with peak accessibility above the threshold
				id_peak = sample_id[id_peak_1]	# the samples with peak accessibility above the threshold
				sample_query_peak = id_peak
				if motif_num>0:
					df_pre1 = pd.DataFrame(index=motif_idvec,columns=field_query,dtype=np.float32)
					df_pre1[column_id3] = motif_idvec
					df_pre1[column_id2] = peak_id
					df_pre1[column_id1] = gene_query_id

					X = rna_exprs.loc[:,motif_idvec]
					gene_tf_corr_peak_pval_2 = []
					# print_mode=0
					# if i2%100==0:
					# 	print_mode=1
					## the gene query may be the same as the motif query
					if gene_query_id in motif_idvec:
						gene_query_id_2 = '%s.1'%(gene_query_id)
					else:
						gene_query_id_2 = gene_query_id

					df_pre2 = []
					try:
						if type_id_query in [1,2]:
							if type_id_query==1:
								sample_id_query = sample_id
							else:
								id_peak_2 = (id_peak_1)|(~id_gene_expr) # peak with accessibility above threshold or gene with expression below threshold
								sample_id_query = sample_id[id_peak_2]
							
							ratio_cond = len(sample_id_query)/sample_num
							ratio_1 = len(id_peak)/sample_num
							df_pre1['ratio_1'] = [ratio_1]*motif_num
							df_pre1['ratio_cond1'] = [ratio_cond]*motif_num
							if len(sample_id_query)<=2:
								sample_id_query = list(sample_id_query)*3

							mtx_1 = np.hstack((np.asarray(X.loc[sample_id_query,:]),np.asarray(Y_peak[sample_id_query])[:,np.newaxis],np.asarray(Y_expr[sample_id_query])[:,np.newaxis]))
							df_query1 = pd.DataFrame(index=sample_id_query,columns=list(motif_idvec)+[peak_id]+[gene_query_id_2],data=mtx_1,dtype=np.float32)
							if i2%100==0:
								print('df_query1: ',df_query1.shape,i1,i2,len(sample_id_query),len(sample_query_peak),len(motif_idvec),peak_id,gene_query_id_2)
							
							# flag_query_2 = 1
							flag_query_2 = 0
							if flag_query_2>0:
								t_value_1 = df_query1.max(axis=0)-df_query1.min(axis=0)
								# feature_id_1 = df_query1.columns
								# feature_id2 = feature_id_1[t_value_1<1E-07]
								column_vec_1 = np.asarray(df_query1.columns)
								t_value_1 = np.asarray(t_value_1)
								# feature_id2 = df_query1.columns[t_value_1<1E-07]
								feature_id2 = column_vec_1[t_value_1<1E-07]

								if len(feature_id2)>0:
									# print('constant value in the vector ',gene_query_id_2,i1,peak_id,i2,feature_id2)
									if gene_query_id_2 in feature_id2:
										# print('gene expression is constant in the subsample')
										print('gene expression is constant in the subsample: %s, %s, %d, %s, %d'%(feature_id2,gene_query_id_2,i1,peak_id,i2))
										# gene_expr_1 = Y_expr[sample_id_query]
										# print(gene_expr_1)
										# continue
									if peak_id in feature_id2:
										# print('peak read value is constant in the subsample')
										print('peak read value is constant in the subsample: %s, %s, %d, %s, %d'%(feature_id2,gene_query_id_2,i1,peak_id,i2))
									motif_idvec_2 = pd.Index(motif_idvec).intersection(feature_id2,sort=False)
									if len(motif_idvec_2)>0:
										print('TF expression is constant in the subsample ',len(motif_idvec_2),motif_idvec_2)
										motif_idvec_ori = motif_idvec.copy()
										motif_idvec = pd.Index(motif_idvec).difference(feature_id2,sort=False)
										column_vec = list(motif_idvec)+[peak_id]+[gene_query_id_2]
										df_query1 = df_query1.loc[:,column_vec]
										motif_num = len(motif_idvec)
										# print('gene_query_id_2, peak_id, motif_idvec ',gene_query_id_2,i1,peak_id,i2,motif_num)
										print('gene_id: %s, %d, peak_id: %s, %d, motif_idvec: %d'%(gene_query_id_2,i1,peak_id,i2,motif_num))
							
							# if print_mode>0:
							if (verbose>0) and (i1%100==0) and (i2%100==0):
								# print('gene_query_id_2, peak_id, motif_idvec ',gene_query_id_2,i1,peak_id,i2,motif_num)
								print('gene_id: %s, %d, peak_id: %s, %d, motif_idvec: %d'%(gene_query_id_2,i1,peak_id,i2,motif_num))
							
							if type_id_compute==0:
								# only estimate raw p-value
								t_vec_1 = [pg.partial_corr(data=df_query1,x=motif_query1,y=gene_query_id_2,covar=peak_id,alternative='two-sided',method='spearman') for motif_query1 in motif_idvec]
								gene_tf_corr_peak_1 = [t_value_1['r'] for t_value_1 in t_vec_1]
								gene_tf_corr_peak_pval_1 = [t_value_1['p-val'] for t_value_1 in t_vec_1]
								# df_pre2 = pd.DataFrame.from_dict(data={field_id1:np.asarray(gene_tf_corr_peak_1),field_id2:np.asarray(gene_tf_corr_peak_pval_1)})
								# df_pre2.index = np.asarray(motif_idvec)
								df_pre2 = pd.DataFrame(index=motif_idvec)
								df_pre2[field_id1] = np.asarray(gene_tf_corr_peak_1)
								df_pre2[field_id2] = np.asarray(gene_tf_corr_peak_pval_1)
							else:
								# p-value correction for TF motifs in the same peak
								df1 = pg.pairwise_corr(data=df_query1,columns=[[gene_query_id_2],list(motif_idvec)],covar=peak_id,alternative='two-sided',method='spearman',padjust='fdr_bh')
								df1.index = np.asarray(df1['Y'])
								# if print_mode>0:
								if verbose>0:
									print('df1, gene_query_id_2, peak_id, motif_idvec ',df1.shape,gene_query_id_2,i1,peak_id,i2,motif_num)
									print(df1)
								# df1 = df1.loc[motif_idvec,:]
								# gene_tf_corr_peak_1 = df1['r']
								# gene_tf_corr_peak_pval_1 = df1['p-unc']
								if 'p-corr' in df1:
									# gene_tf_corr_peak_pval_2 = df1['p-corr']
									df_pre2 = df1.loc[:,['r','p-unc','p-corr']]
								else:
									df_pre2 = df1.loc[:,['r','p-unc']]
								# df_pre2 = df_pre2.rename(columns={'r':'gene_tf_corr_peak','p-unc':'gene_tf_corr_peak_pval','p-corr':'gene_tf_corr_peak_pval_corrected1'})
								df_pre2 = df_pre2.rename(columns={'r':field_id1,'p-unc':field_id2,'p-corr':field_id3})
						else:
							gene_tf_corr_peak_1, gene_tf_corr_peak_pval_1 = [], []
							for l2 in range(motif_num):
								motif_query1 = motif_idvec[l2]
								id_tf = dict_motif[motif_query1] # TF with expression
								# sample_id_2 = sample_id[~((id_tf)&(~id_peak_2))]
								# sample_id_query = sample_id[(~id_tf)|id_peak_2]
								sample_id_query = sample_id[(~id_tf)|id_peak_1]
								ratio_cond  = len(sample_id_query)/sample_num
								df_pre1.loc[motif_query1,'ratio_cond2'] = ratio_cond
								if len(sample_id_query)<=2:
									sample_id_query = list(sample_id_query)*3

								mtx_2 = np.hstack((np.asarray(X.loc[sample_id_query,[motif_query1]]),np.asarray(Y_peak[sample_id_query])[:,np.newaxis],np.asarray(Y_expr[sample_id_query])[:,np.newaxis]))
								df_query2 = pd.DataFrame(index=sample_id_query,columns=[motif_query1]+[peak_id1]+[gene_query_id_2],data=mtx_2,dtype=np.float32)
								if i2%100==0:
									print('df_query2: ',df_query2.shape,i1,i2,len(sample_id_query),len(sample_query_peak),motif_query1,peak_id,gene_query_id_2)
								t_value_1 = pg.partial_corr(data=df_query2,x=motif_query1,y=gene_query_id_2,covar=peak_id,alternative='two-sided',method='spearman')
								gene_tf_corr_peak_1.append(t_value_1['r'])
								gene_tf_corr_peak_pval_1.append(t_value_1['p-val'])
							
							df_pre2 = pd.DataFrame(index=motif_idvec)
							df_pre2[field_id1] = np.asarray(gene_tf_corr_peak_1)
							df_pre2[field_id2] = np.asarray(gene_tf_corr_peak_pval_1)
					except Exception as error:
						print('error! ',error)
						print('gene_query_id_2, peak_id, motif_idvec, df_query1',gene_query_id_2,i1,peak_id,i2,motif_num,motif_idvec,df_query1.shape)
						# return
						continue

					motif_idvec_1 = df_pre2.index
					df_pre1.loc[motif_idvec_1,[field_id1,field_id2]] = df_pre2.loc[motif_idvec_1,[field_id1,field_id2]]
					# if len(gene_tf_corr_peak_pval_2)>0:
					# 	df_pre1[field_id3] = np.asarray(gene_tf_corr_peak_pval_2)
					if field_id3 in df_pre2.columns:
						df_pre1.loc[motif_idvec_1,field_id3] = df_pre2.loc[motif_idvec_1,field_id3]
					if save_mode>0:
						list_query1.append(df_pre1)
					list_query2.append(df_pre1)

			# interval = 20
			interval = 100
			if (save_mode>0) and (i1%interval==0) and (len(list_query1)>0):
				df_query1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
				# df1.index = np.asarray(df1['peak_id'])
				# output_file_path = self.save_path_1
				# output_filename = '%s/test_query_correlation.%s.1.txt'%(output_file_path,filename_annot1)
				if (i1==0) or (flag_1==0):
					df_query1.to_csv(output_filename,sep='\t',float_format='%.5f')
					flag_1=1
				else:
					df_query1.to_csv(output_filename,header=False,mode='a',sep='\t',float_format='%.5f')
				list_query1 = []

		load_mode_2 = 1
		
		if load_mode_2>0:
			df_query1 = pd.concat(list_query2,axis=0,join='outer',ignore_index=False)
			df_query1.to_csv(output_filename,sep='\t',float_format='%.5f')
				
		return df_query1

	## ====================================================
	# query gene_tf_corr_peak p-value corrected
	def test_gene_tf_corr_peak_pval_corrected_query_1(self,df_gene_peak_query=[],gene_query_vec=[],motif_query_vec=[],alpha=0.05,
														method_type_id_correction='fdr_bh',type_id_1=0,parallel_mode=0,save_mode=1,verbose=0,select_config={}):

		flag_query1=1
		if len(df_gene_peak_query)>0:
			df_gene_peak_query_1 = df_gene_peak_query
			# df_gene_peak_query_1 = df_gene_tf_corr_peak
			column_idvec = ['motif_id','peak_id','gene_id']
			# column_idvec = select_config['column_idvec']
			column_id3, column_id2, column_id1 = column_idvec
			# field_query = ['gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1']
			field_query = select_config['column_gene_tf_corr_peak']
			field_id1, field_id2, field_id3 = field_query
			field_id_query = '%s_corrected2'%(field_id2)
			print('gene_tf_corr_peak p-value correction')
			print(column_idvec,field_query)

			if len(gene_query_vec)==0:
				gene_query_vec = df_gene_peak_query_1['gene_id'].unique()
			gene_query_num = len(gene_query_vec)

			if len(motif_query_vec)==0:
				motif_query_vec = df_gene_peak_query_1['motif_id'].unique()
			motif_query_num = len(motif_query_vec)
			
			df_gene_peak_query_1.index = test_query_index(df_gene_peak_query_1,column_vec=['motif_id','peak_id','gene_id'])
			query_id_1 = df_gene_peak_query_1.index
			
			## p-value correction for gene_tf_corr_peak_pval
			column_id_pre1 = 'gene_tf_corr_peak_pval_corrected1'
			if flag_query1>0:
				# df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1['motif_id'])
				if verbose>0:
					print('pvalue correction')
				
				start = time.time()
				column_id_1 = field_id2
				if type_id_1==0:
					column_id_query = column_id1 # p-value query per gene
					# df_pval_query1 = df_gene_peak_query_1.loc[:,[column_id3,column_id_1]]
					feature_query_vec = gene_query_vec
				else:
					column_id_query = column_id3 # p-value query per TF
					feature_query_vec = motif_query_vec

				# df_pval_query1 = df_gene_peak_query_1.loc[:,[column_id_query,column_id_pre1,column_id_1]]
				df_pval_query1 = df_gene_peak_query_1.loc[:,[column_id_query,column_id_1]]
				df_pval_query1 = df_pval_query1.fillna(1) # p-value
				feature_query_num = len(feature_query_vec)

				alpha=0.05
				method_type_id_correction='fdr_bh'
				field_query_2 = [field_id2,field_id_query]
				interval_1 = 100
				if parallel_mode==0:
					df_gene_peak_query_1 = self.test_gene_tf_corr_peak_pval_corrected_unit1(data=df_gene_peak_query_1,feature_query_vec=feature_query_vec,feature_id='',column_id_query=column_id_query,field_query=field_query_2,alpha=alpha,method_type_id_correction=method_type_id_correction,type_id_1=1,interval=interval_1,save_mode=1,verbose=verbose,select_config=select_config)
				else:
					# feature_query_num1 = feature_query_num
					# feature_query_num1 = 500
					query_res_local = Parallel(n_jobs=-1)(delayed(self.test_gene_tf_corr_peak_pval_corrected_unit1)(data=df_pval_query1,feature_query_vec=feature_query_vec[i2:(i2+1)],column_id_query=column_id_query,field_query=field_query_2,
																														alpha=alpha,method_type_id_correction=method_type_id_correction,interval=interval_1,save_mode=1,verbose=(i2%interval_1),select_config=select_config) for i2 in tqdm(np.arange(feature_query_num)))
					
					for t_query_res in query_res_local:
						# dict_query = t_query_res
						if len(t_query_res)>0:
							df_query = t_query_res
							query_id1 = df_query.index
							df_gene_peak_query_1.loc[query_id1,field_id_query] = df_query.loc[query_id1,field_id_query]

				stop = time.time()
				print('pvalue correction ',stop-start)

		print(df_gene_peak_query_1[0:5])
		return df_gene_peak_query_1

	# query gene_tf_corr_peak p-value corrected
	def test_gene_tf_corr_peak_pval_corrected_unit1(self,data=[],feature_query_vec=[],feature_id='',column_id_query='',field_query=[],alpha=0.05,method_type_id_correction='fdr_bh',type_id_1=0,interval=1000,save_mode=1,verbose=0,select_config={}):

		df_query_1 = data
		feature_query_num = len(feature_query_vec)
		list1 = []
		field_id_query1 = 'gene_tf_corr_peak_pval_corrected1'
		for i2 in range(feature_query_num):
			feature_id1 = feature_query_vec[i2]
			query_id1 = (df_query_1[column_id_query]==feature_id1)
			field_id, field_id_query = field_query[0:2]
			# pvals = np.asarray(df_query_1.loc[query_id1,field_id])
			pvals = df_query_1.loc[query_id1,field_id]
			pvals_correction_vec1, pval_thresh1 = utility_1.test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_id_correction)
			id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
							
			if type_id_1==1:
				df_query_1.loc[query_id1,field_id_query] = pvals_corrected1
			else:
				df_query1 = df_query_1.loc[query_id1,[field_id]]
				query_vec_1 = df_query1.index

				if field_id_query1 in df_query_1.columns:
					type_query = 1
					column_vec = [field_id,field_id_query1,field_id_query]
				else:
					type_query = 0
					column_vec = [field_id,field_id_query]

				df_query2 = pd.DataFrame(index=query_vec_1,columns=column_vec,dtype=np.float32)
				df_query2[field_id] = pvals
				if type_query>0:
					df_query2[field_id_query1] =  df_query_1.loc[query_id1,field_id_query1]
				df_query2[field_id_query] =  pvals_corrected1
				list1.append(df_query2)
			
			if (verbose>0) and (i2%interval==0):
				query_num1 = len(pvals_corrected1)
				# print('motif_id1, pvals_corrected1 ',motif_id1,i2,query_num1,np.max(pvals_corrected1),np.min(pvals_corrected1),np.mean(pvals_corrected1),np.median(pvals_corrected1))
				print('feature_id1, pvals_corrected1 ',feature_id1,i2,query_num1,np.max(pvals_corrected1),np.min(pvals_corrected1),np.mean(pvals_corrected1),np.median(pvals_corrected1))
				# print(df_gene_peak_query_1[0:5])
				# print(df_gene_peak_query_1.loc[query_id1,:])
				if type_id_1==1:
					print(df_query_1.loc[query_id1,:])
				else:
					print(df_query2)

		if (type_id_1==0):
			if (feature_query_num>0):
				df_query_2 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
			else:
				df_query_2 = list1[0]
		else:
			df_query_2 = df_query_1

		return df_query_2

	## ====================================================
	# query TF expression-gene expression partial correlation conditioned on peak accessibility
	# combine estimation from different runs
	def test_partial_correlation_gene_tf_cond_peak_combine_1(self,input_file_path='',
																filename_prefix_vec=[],
																save_mode=0,
																output_filename_list=[],
																select_config={}):
		if input_file_path!='':
			if len(filename_prefix_vec)>0:
				# filename_prefix_1,filename_prefix_2,filename_annot2 = filename_prefix_vec
				filename_prefix_save_1 = filename_prefix_vec[0]
				
				input_filename_list1 = []
				gene_query_num_1 = 21528
				start_id_1, start_id_2, interval = 0, 22020, 1000
				# start_id_1, start_id_2, interval = 7000, 12500, 500
				for start_id1 in range(start_id_1,start_id_2,interval):
					start_id2 = start_id1+interval
					# if start_id1==12000:
					# 	start_id2 = gene_query_num_1
					if start_id1==21000:
						start_id2 = gene_query_num_1
					input_filename = '%s/%s.%d_%d.txt'%(input_file_path,filename_prefix_save_1,start_id1,start_id2)
					input_filename_list1.append(input_filename)

				list_1, list_ratio_1 = [], []
				file_num1 = len(input_filename_list1)
				## query gene_tf_corr_peak and peak ratio
				for i1 in range(file_num1):
				# for input_filename in input_filename_list1:
					input_filename = input_filename_list1[i1]
					if os.path.exists(input_filename)==False:
						print('the file does not exist ',input_filename,i1)
						continue
						# return
					df_query1 = pd.read_csv(input_filename,index_col=0,sep='\t')
					df_query1.index = self.test_query_index(df_query1,column_vec=['motif_id','peak_id','gene_id'])
					t_columns = df_query1.columns.difference(['ratio_1','ratio_cond1'],sort=False)
					df_query_1 = df_query1.loc[:,t_columns]
					df_query2 = df_query1.loc[:,['peak_id','gene_id','ratio_1','ratio_cond1']]
					df_query2.index = self.test_query_index(df_query2,column_vec=['peak_id','gene_id'])
					# df_query2= df_query2.drop_duplicates(subset=['ratio_1','ratio_cond1'])
					df_query_2 = df_query2.loc[~df_query2.index.duplicated(keep='first')]
					print('df_query1 ',df_query1.shape,input_filename,i1)
					print(df_query1[0:5])
					print(df_query2[0:5])
					list_1.append(df_query_1)
					list_ratio_1.append(df_query_2)

				# df1 = pd.concat(list_1,axis=0,join='outer',ignore_index=False)
				# df1_ratio = pd.concat(list_ratio_1,axis=0,join='outer',ignore_index=False)
				df1_pre1 = pd.concat(list_1,axis=0,join='outer',ignore_index=False)
				df1_ratio_pre1 = pd.concat(list_ratio_1,axis=0,join='outer',ignore_index=False)
				df1 = df1_pre1.loc[~df1_pre1.index.duplicated(keep='first')]
				df1_ratio = df1_ratio_pre1.loc[~df1_ratio_pre1.index.duplicated(keep='first')]
				df1 = df1.sort_values(by=['gene_id','peak_id','gene_tf_corr_peak'],ascending=[True,True,False])
				df1.index = np.asarray(df1['motif_id'])
				df1_ratio = df1_ratio.sort_values(by=['gene_id','peak_id'],ascending=True)
				# df1_ratio.index = np.asarray(df1_ratio['peak_id'])
				df1_ratio.index = np.asarray(df1_ratio['gene_id'])
				if (save_mode>0) and (len(output_filename_list)>0):
					output_filename_1, output_filename_2 = output_filename_list
					if os.path.exists(output_filename_1)==True:
						print('the file exists ',output_filename_1)
					else:
						df1.to_csv(output_filename_1,sep='\t',float_format='%.5f')
					# output_filename_2 = '%s/%s.%s.%s.peak_ratio.%d_%d.txt'%(output_file_path,filename_prefix_1,filename_prefix_2,filename_annot2,start_id1,start_id2)
					if os.path.exists(output_filename_2)==True:
						print('the file exists ',output_filename_2)
					else:
						df1_ratio.to_csv(output_filename_2,sep='\t',float_format='%.5f')
					print('df1, df1_ratio ',df1.shape,df1_ratio.shape)

				return df1, df1_ratio

	## ====================================================
	# query TF expression-gene expression partial correlation conditioned on peak accessibility
	# combine estimation from different runs
	def test_partial_correlation_gene_tf_cond_peak_combine_pre1(self,input_file_path='',
																	filename_prefix_vec=[],
																	filename_prefix_save='',
																	type_id_query=0,
																	type_id_compute=0,
																	save_mode=1,
																	output_file_path='',
																	output_filename='',
																	output_filename_list=[],
																	overwrite=0,
																	select_config={}):

		if len(filename_prefix_vec)==0:
			filename_prefix_save_pre1 = '%s.pcorr_query1.%d.%d'%(filename_prefix_save,type_id_query,type_id_compute)
			# filename_prefix_vec = [filename_prefix_1,filename_prefix_2,filename_annot2]
			filename_prefix_vec = [filename_prefix_save_pre1_2]

		df1, df1_ratio = self.test_partial_correlation_gene_tf_cond_peak_combine_1(input_file_path=input_file_path,
																					filename_prefix_vec=filename_prefix_vec,
																					save_mode=save_mode,
																					output_filename_list=[],
																					select_config=select_config)

		gene_query_vec = df1['gene_id'].unique()
		peak_query_vec = df1['peak_id'].unique()
		gene_query_num, peak_query_num = len(gene_query_vec), len(peak_query_vec)
		print('gene number:%d, peak number:%d '%(gene_query_num,peak_query_num))

		if save_mode>0:
			if output_file_path=='':
				output_file_path = input_file_path
			# start_id1, start_id2 = 0, gene_query_num1
			# output_filename = '%s/%s.%s.%s.%d_%d.txt'%(output_file_path,filename_prefix_1,filename_prefix_2,filename_annot2,start_id1,start_id2)
			if output_filename=="":
				output_filename = '%s/%s.ori.txt'%(output_file_path,filename_prefix_save_pre1)
			# self.filename_save_1=output_filename
			if os.path.exists(output_filename)==True:
				print('the file exists ',output_filename)
			else:
				df1.to_csv(output_filename,sep='\t',float_format='%.5f')
			# output_filename = '%s/%s.peak_ratio.%d_%d.txt'%(output_file_path,filename_prefix_save_pre1,start_id1,start_id2)
			output_filename = '%s/%s.peak_ratio.1.txt'%(output_file_path,filename_prefix_save_pre1)
			flag_write=1
			if os.path.exists(output_filename)==True:
				print('the file exists ',output_filename)
				if overwrite==0:
					flag_write=0
			# else:
			# 	df1_ratio.to_csv(output_filename,sep='\t',float_format='%.5f')
			if flag_write>0:
				df1_ratio.to_csv(output_filename,sep='\t',float_format='%.5f')
			print('df1, df1_ratio ',df1.shape,df1_ratio.shape)
			
		return df1, df1_ratio

	## ==================================================================
	# score function 1: query peak-tf-gene score 1
	def test_query_score_function1(self,df_peak_tf_corr=[],df_gene_tf_corr_peak=[],df_gene_peak_query=[],
										motif_data=[],motif_data_score=[],gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],
										peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],type_id_query=2,type_id_compute=1,
										flag_peak_tf_corr=0,flag_gene_tf_corr_peak=1,flag_pval_1=1,flag_pval_2=1,flag_load=1,field_load=[],parallel_mode=1,
										save_mode=1,input_file_path='',filename_prefix='',filename_annot='',output_file_path='',output_filename='',verbose=0,select_config={}):

		if flag_peak_tf_corr in [1,2]:
			## estimate peak-TF expresssion correlation
			print('estimate peak-TF expression correlation')
			start = time.time()
			correlation_type = select_config['correlation_type']
			if len(field_load)==0:
				field_load = [correlation_type,'pval','pval_corrected']
			dict_query = self.test_peak_tf_correlation_query_1(motif_data=motif_data,peak_query_vec=[],motif_query_vec=[],
																peak_read=peak_read,rna_exprs=rna_exprs,
																correlation_type=correlation_type,
																flag_load=flag_load,field_load=field_load,
																save_mode=save_mode,
																input_file_path=input_file_path,
																input_filename_list=[],
																output_file_path=output_file_path,
																filename_prefix=filename_prefix,select_config=select_config)
			stop = time.time()
			print('estimate peak-TF expression correlation for %d genes used %.5fs'%(gene_query_num,stop-start))

		if flag_peak_tf_corr in [2]:
			df_peak_annot = self.peak_annot
			# TF motif score normalization
			df_motif_score_query = self.test_peak_tf_score_normalization_pre_compute(gene_query_vec=[],
																						peak_query_vec=[],
																						motif_query_vec=[],
																						df_gene_peak_query=[],
																						motif_data=motif_data,
																						motif_data_score=motif_data_score,
																						peak_read=peak_read,
																						rna_exprs=rna_exprs,
																						peak_read_celltype=[],
																						df_peak_annot=df_peak_annot,
																						filename_annot=filename_annot,
																						output_file_path=output_file_path,
																						select_config=select_config)

		flag_gene_tf_corr_peak_1 = flag_gene_tf_corr_peak
		# flag_gene_tf_corr_peak_pval=0
		flag_gene_tf_corr_peak_pval=flag_pval_2
		if 'flag_gene_tf_corr_peak_pval' in select_config:
			flag_gene_tf_corr_peak_pval = select_config['flag_gene_tf_corr_peak_pval']
		df_gene_peak = df_gene_peak_query
		if len(motif_query_vec)==0:
			# motif_query_vec = df_gene_peak_query['motif_id'].unique()
			motif_query_name_expr = self.motif_query_name_expr
			motif_query_vec = motif_query_name_expr
		
		if flag_gene_tf_corr_peak_1>0:
			## estimate gene-TF expresssion partial correlation given peak accessibility
			print('estimate gene-TF expression partial correlation given peak accessibility')
			start = time.time()
			peak_query_vec, motif_query_vec = [], []
			type_id_query_2 = 2
			type_id_compute = 1
			gene_query_num = len(gene_query_vec)
			df_gene_tf_corr_peak = self.test_partial_correlation_gene_tf_cond_peak_1(motif_data=motif_data,
																						gene_query_vec=gene_query_vec,
																						peak_query_vec=peak_query_vec,
																						motif_query_vec=motif_query_vec,
																						df_gene_peak_query=df_gene_peak,
																						peak_read=peak_read,
																						rna_exprs=rna_exprs,
																						rna_exprs_unscaled=rna_exprs_unscaled,
																						type_id_query=type_id_query,
																						type_id_compute=type_id_compute,
																						parallel_mode=parallel_mode,
																						save_mode=save_mode,
																						output_filename=output_filename,
																						select_config=select_config)
			stop = time.time()
			print('estimating gene-TF expression partial correlation given peak accessibility for %d genes used %.5fs'%(gene_query_num,stop-start))

		## gene_tf_corr_peak p-value query
		if flag_gene_tf_corr_peak_pval>0:
			print('df_gene_tf_corr_peak ',df_gene_tf_corr_peak.shape)
			print('estimate p-value corrected for gene-TF expression partial correlation given peak accessibility')
			start = time.time()
			# parallel_mode = 1
			if 'parallel_mode_pval_correction' in select_config:
				parallel_mode = select_config['parallel_mode_pval_correction']
			df_gene_tf_corr_peak = self.test_gene_tf_corr_peak_pval_corrected_query_1(df_gene_peak_query=df_gene_tf_corr_peak,
																						gene_query_vec=[],
																						motif_query_vec=motif_query_vec,
																						parallel_mode=parallel_mode,
																						verbose=verbose,
																						select_config=select_config)
			stop = time.time()
			print('estimating p-value corrected for gene-TF expression partial correlation used %.5fs'%(stop-start))
			# print(df_gene_tf_corr_peak[0:5])
			if (save_mode>0) and (output_filename!=''):
				df_gene_tf_corr_peak.to_csv(output_filename,sep='\t',float_format='%.5f')

		df_gene_tf_corr_peak.index = np.asarray(df_gene_tf_corr_peak['gene_id'])

		return df_gene_tf_corr_peak

	## ==================================================================
	# score function 2: query peak-tf-gene score 2
	def test_query_score_function2(self,df_peak_tf_corr=[],
										df_gene_tf_corr_peak=[],
										df_gene_peak_query=[],
										motif_data=[],
										gene_query_vec=[],
										peak_query_vec=[],
										motif_query_vec=[],
										flag_peak_gene_corr=1,
										flag_gene_tf_corr_peak=1,
										flag_pval_1=1,
										flag_pval_2=1,
										parallel_mode=1,
										save_mode=1,
										output_filename='',
										select_config={}):
		
		# x = 1
		# flag_gene_tf_corr_peak_1=1
		flag_gene_tf_corr_peak_1 = flag_gene_tf_corr_peak
		# flag_gene_tf_corr_peak_pval=0
		flag_gene_tf_corr_peak_pval=flag_pval_2
		df_gene_peak = df_gene_peak_query
		if len(motif_query_vec)==0:
			motif_query_vec = df_gene_peak_query['motif_id'].unique()
		if len(gene_query_vec)==0:
			gene_query_vec = df_gene_peak_query['gene_id'].unique()

		if flag_gene_tf_corr_peak_1>0:
			## estimate gene-TF expresssion partial correlation given peak accessibility
			print('estimate gene-TF expression partial correlation given peak accessibility')
			start = time.time()
			peak_query_vec, motif_query_vec = [], []
			type_id_query_2 = 2
			type_id_compute = 1
			df_gene_tf_corr_peak = self.test_partial_correlation_gene_tf_cond_peak_1(motif_data=motif_data,
																						gene_query_vec=gene_query_vec,
																						peak_query_vec=peak_query_vec,
																						motif_query_vec=motif_query_vec,
																						df_gene_peak_query=df_gene_peak,
																						peak_read=peak_read,
																						rna_exprs=rna_exprs,
																						rna_exprs_unscaled=rna_exprs_unscaled,
																						type_id_query=type_id_query_2,
																						type_id_compute=type_id_compute,
																						save_mode=save_mode,
																						output_filename=output_filename,
																						select_config=select_config)
			stop = time.time()
			print('estimating gene-TF expression partial correlation given peak accessibility for %d genes used %.5fs'%(gene_query_num,stop-start))

		## gene_tf_corr_peak p-value query
		if flag_gene_tf_corr_peak_pval>0:
			print('df_gene_tf_corr_peak ',df_gene_tf_corr_peak.shape)
			print('estimate p-value corrected for gene-TF expression partial correlation given peak accessibility')
			start = time.time()
			df_gene_tf_corr_peak = self.test_gene_tf_corr_peak_pval_corrected_query_1(df_gene_peak_query=df_gene_tf_corr_peak,
																						gene_query_vec=gene_query_vec,
																						motif_query_vec=motif_query_vec,
																						parallel_mode=parallel_mode,
																						select_config=select_config)
			stop = time.time()
			print('estimating p-value corrected for gene-TF expression partial correlation used %.5fs'%(stop-start))

		df_gene_tf_corr_peak.index = np.asarray(df_gene_tf_corr_peak['gene_id'])

		return df_gene_peak_query

	## convert wide format dataframe to long format and combine dataframe
	def test_query_feature_combine_format_1(self,df_list,column_idvec=[],field_query=[],dropna=False,select_config={}):

		query_num1 = len(field_query)
		column_id1, column_id2 = column_idvec[0:2]
		for i1 in range(query_num1):
			df_query = df_list[i1]
			field1 = field_query[i1]
			df_query[column_id1] = np.asarray(df_query.index)
			df_query = df_query.melt(id_vars=[column_id1],var_name=column_id2,value_name=field1)
			if dropna==True:
				df_query = df_query.dropna(axis=0,subset=[field1])
			df_query.index = test_query_index(df_query,column_vec=column_idvec)
			df_list[i1] = df_query

		df_query_1 = pd.concat(df_list,axis=1,ignore_index=False)

		return df_query_1

	## ====================================================
	# initial estimation of peak-tf-gene link query scores
	# retrieve estimated peak-TF correlation and gene-TF partial correlation conditioned on peak accessibility
	def test_gene_peak_tf_query_score_init_pre1_1(self,gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],
														load_mode=0,save_mode=0,input_file_path='',output_file_path='',output_filename='',verbose=0,select_config={}):

		field_query1 = ['peak_tf_corr','peak_tf_pval_corrected']
		field_query2 = ['gene_tf_corr','gene_tf_pval_corrected']
		field_list_ori = [field_query1,field_query2]
		query_num1 = len(field_list_ori)

		flag_load_1 = (load_mode in [1,2])
		input_filename_list = []
		field_list = []
		correlation_type = 'spearmanr'
		if 'correlation_type' in select_config:
			correlation_type = select_config['correlation_type']
			
		filename_annot_vec = [correlation_type,'pval_corrected']
		input_file_path2 = select_config['file_path_motif_score']
		file_path_gene_tf = select_config['file_path_gene_tf']
		input_file_path_query2 = file_path_gene_tf

		# data_file_type_query = select_config['data_file_type_query']
		if flag_load_1>0:
			data_file_type_query = select_config['data_file_type_query']
			# filename_prefix_1 = 'test_gene_tf_correlation.%s'%(data_file_type_query)
			filename_prefix_gene_tf = select_config['filename_prefix_gene_tf']
			filename_prefix_1 = filename_prefix_gene_tf

			input_filename_1 = '%s/%s.%s.1.txt'%(input_file_path_query2,filename_prefix_1,correlation_type)
			input_filename_2 = '%s/%s.%s.pval_corrected.1.txt'%(input_file_path_query2,filename_prefix_1,correlation_type)

			if os.path.exists(input_filename_1)==False:
				print('the file does not exist: %s'%(input_filename_1))
				query_id1 = select_config['query_id1']
				query_id2 = select_config['query_id2']
				iter_mode = 0
				if (query_id1>=0) and (query_id2>query_id1):
					iter_mode = 1

				if iter_mode>0:
					feature_query_num_1 = select_config['feature_query_num_1']
					query_id2_pre = np.min([query_id2,feature_query_num_1])
					# if query_id2>feature_query_num_1:
					# 	query_id2 = feature_query_num_1
					# filename_prefix_1 = 'test_gene_tf_correlation.%s.%d_%d'%(data_file_type_query,query_id1,query_id2_pre)
					filename_prefix_1 = '%s.%d_%d'%(filename_prefix_gene_tf,query_id1,query_id2)

				input_filename_1 = '%s/%s.%s.1.txt'%(input_file_path_query2,filename_prefix_1,correlation_type)
				input_filename_2 = '%s/%s.%s.pval_corrected.1.txt'%(input_file_path_query2,filename_prefix_1,correlation_type)
			
			# input_filename_list1 = ['%s/%s.%s.1.txt'%(input_file_path_2,filename_prefix_1,filename_annot1) for filename_annot1 in filename_annot_vec[0:2]]
			# filename_gene_expr_corr = input_filename_list1
			filename_gene_expr_corr = [input_filename_1,input_filename_2]
			select_config.update({'filename_gene_expr_corr':filename_gene_expr_corr})

			input_filename_list.append(filename_gene_expr_corr)
			feature_type_annot1 = 'gene_tf_corr'
			field_list.append([field_query2,feature_type_annot1])
		
		flag_load_2 = (load_mode in [0,2])
		if flag_load_2>0:
			if 'file_path_peak_tf' in select_config:
				file_save_path_2 = select_config['file_path_peak_tf']
				print('file_path_peak_tf: ',file_save_path_2)
			else:
				file_save_path = select_config['data_path_save_local']
				file_save_path_2 = file_save_path

			if 'filename_prefix_peak_tf' in select_config:
				filename_prefix_peak_tf = select_config['filename_prefix_peak_tf']
			else:
				data_file_type_query = select_config['data_file_type_query']
				filename_prefix_peak_tf = 'test_peak_tf_correlation.%s'%(data_file_type_query)
				
			filename_save_annot_peak_tf = '1'
			if 'filename_save_annot_peak_tf' in select_config:
				filename_save_annot_peak_tf = select_config['filename_save_annot_peak_tf']

			filename_prefix = filename_prefix_peak_tf
			# input_filename_list2 = ['%s/%s.%s.1.copy1.txt'%(file_save_path_2,filename_prefix,filename_annot1) for filename_annot1 in filename_annot_vec[0:2]]
			input_filename_list2 = ['%s/%s.%s.%s.txt'%(file_save_path_2,filename_prefix,filename_annot1,filename_save_annot_peak_tf) for filename_annot1 in filename_annot_vec[0:2]]
			filename_peak_tf_corr = input_filename_list2
				
			# select_config.update({'filename_peak_tf_corr':filename_peak_tf_corr,'filename_gene_expr_corr':filename_gene_expr_corr})
			select_config.update({'filename_peak_tf_corr':filename_peak_tf_corr})

			input_filename_list.append(filename_peak_tf_corr)
			feature_type_annot2 = 'peak_tf_corr'
			field_list.append([field_query1,feature_type_annot2])

		list_query_1 = []
		dict_query_pre1 = dict()
		feature_type_vec = ['gene_tf_corr','peak_tf_corr']
		for feature_type_annot in feature_type_vec:
			dict_query_pre1[feature_type_annot] = dict()

		query_num1 = len(input_filename_list)
		for i1 in range(query_num1):
			filename_query_1 = input_filename_list[i1]
			# input_filename = input_filename_list[i1]
			input_filename = filename_query_1[0]
			t_vec_1 = field_list[i1]
			field_query, feature_type_annot = t_vec_1[0:2]
			print('input_filename: ',input_filename,field_query,feature_type_annot)

			b1 = input_filename.find('.txt')
			if b1<0:
				adata = sc.read(input_filename)
				if (i1==1) and (len(motif_query_vec)>0):
					adata = adata[:,motif_query_vec]
				feature_query1, feature_query2 = adata.obs_names, adata.var_names
				try:
					df_query_1 = pd.DataFrame(index=feature_query1,columns=feature_query2,data=adata.X.toarray(),dtype=np.float32)								
				except Exception as error:
					print('error! ',error)
					df_query_1 = pd.DataFrame(index=feature_query1,columns=feature_query2,data=np.asarray(adata.X),dtype=np.float32)								

				field_id1 = field_query[1]
				df_query = adata.obsm[field_id1]
				df_query_2 = pd.DataFrame(index=feature_query1,columns=feature_query2,data=df_query.toarray(),dtype=np.float32)
				 #list1 = [df_query_1,df_query_2]
			else:
				df_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t') # the correlation value
				input_filename_2 = filename_query_1[1]
				df_query_2 = pd.read_csv(input_filename_2,index_col=0,sep='\t') # the p-value corrected

				print('correlation value: ',df_query_1.shape)
				print('p-value corrected: ',df_query_2.shape)

				if len(motif_query_vec)>0:
					df_query_1 = df_query_1.loc[:,motif_query_vec]
					df_query_2 = df_query_2.loc[:,motif_query_vec]

				print('correlation value: ',df_query_1.shape)
				print('p-value corrected: ',df_query_2.shape)

			list1 = [df_query_1,df_query_2]
			dict_query_1 = dict(zip(field_query,list1))
			# list_query_1.append(dict_query_1)
			dict_query_pre1.update({feature_type_annot:dict_query_1})

		list1 = [dict_query_pre1[feature_type_annot] for feature_type_annot in feature_type_vec]
		dict_gene_tf_query, dict_peak_tf_query = list1[0:2]

		return dict_peak_tf_query, dict_gene_tf_query

	## ====================================================
	# initial estimation of peak-tf-gene link query scores
	# add the annotations of correlation and partial correlation and pvalues, and the regularization scores
	def test_gene_peak_tf_query_score_init_pre1_2(self,df_gene_peak_query=[],df_peak_tf_1=[],df_peak_tf_2=[],dict_peak_tf_query={},dict_gene_tf_query={},df_gene_tf_corr_peak=[],
														load_mode=0,save_mode=0,input_file_path='',output_file_path='',output_filename='',verbose=0,select_config={}):

		## peak-tf link, gene-tf link and peak-gene link query
		flag_link_type_query=0
		field_query_pre1 = ['peak_tf_corr','peak_tf_pval_corrected',
								'gene_tf_corr_peak','gene_tf_corr_peak_pval_corrected1','gene_tf_corr_peak_pval_corrected2',
								'gene_tf_corr','gene_tf_pval_corrected',
								'peak_gene_corr_','peak_gene_corr_pval']

		# score from the in silico ChIP-seq method
		field_query_pre2_1 = ['correlation_score','max_accessibility_score',
								'motif_score','motif_score_normalize',
								'score_1','score_pred1']

		# score from the motif score normalization
		field_query_pre2_2 = ['motif_score','motif_score_minmax','motif_score_log_normalize_bound',
								'max_accessibility_score','score_accessibility','score_accessibility_minmax','score_1']

		# df_gene_tf_corr_peak = self.df_gene_tf_corr_peak
		df_query1 = df_gene_tf_corr_peak

		## copy columns of one dataframe to another dataframe
		feature_type_vec = ['motif','peak','gene']
		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec

		# gene_query_1 = df_gene_tf_corr_peak[column_id1].unique()
		# peak_query_1 = df_gene_tf_corr_peak[column_id2].unique()
		# motif_query_1 = df_gene_tf_corr_peak[column_id3].unique()
		feature_query_list = []
		query_num1 = len(column_idvec)
		for i1 in range(query_num1):
			column_id_query = column_idvec[i1]
			feature_type_query = feature_type_vec[i1]
			feature_query_1 = df_gene_tf_corr_peak[column_id_query].unique()
			feature_query_list.append(feature_query_1)
			feature_query_num1 = len(feature_query_1)
			print('feature_query: ',feature_type_query,feature_query_num1)
		motif_query_1, peak_query_1, gene_query_1 = feature_query_list

		## query peak-gene correlation
		column_idvec_1 = [column_id2,column_id1]
		column_vec_1 = [['spearmanr','pval1']]
		print('df_gene_tf_corr_peak ',df_gene_tf_corr_peak.shape)
		print(df_gene_tf_corr_peak[0:2])
		print('df_gene_peak_query ',df_gene_peak_query.shape)
		print(df_gene_peak_query[0:2])

		df_gene_peak_query_ori = df_gene_peak_query
		df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])
		df_gene_peak_query = df_gene_peak_query.loc[gene_query_1,:]
		print('df_gene_peak_query_ori, df_gene_peak_query ',df_gene_peak_query_ori.shape,df_gene_peak_query.shape)
		df_list1 = [df_gene_tf_corr_peak,df_gene_peak_query]

		print('query peak accessibility-gene expression correlation')
		from utility_1 import test_column_query_1
		df_query1 = test_column_query_1(input_filename_list=[],id_column=column_idvec_1,column_vec=column_vec_1,
											df_list=df_list1,type_id_1=0,type_id_2=0,reset_index=True,select_config=select_config)

		column_1 = column_vec_1[0]
		column_2 = ['peak_gene_corr_','peak_gene_corr_pval']
		# df_query1 = df_query1.rename(columns={column_1:column_2})
		dict1 = dict(zip(column_1,column_2))
		df_query1 = df_query1.rename(columns=dict1)

		## query peak-TF correlation
		print('query peak accessibility-TF expression correlation')
		start = time.time()
		field_query = ['peak_tf_corr','peak_tf_pval_corrected']
		list2_ori = [dict_peak_tf_query[field1] for field1 in field_query]
		# list2 = [df_query.loc[peak_query_1,motif_query_1] for df_query in list2_ori]
		field_query_num1 = len(field_query)
		list2 = []
		for i1 in range(field_query_num1):
			df_query_ori = list2_ori[i1]
			df_query = df_query_ori.loc[peak_query_1,motif_query_1]
			print('df_query_ori, df_query: ',df_query_ori.shape,df_query.shape)
			list2.append(df_query)

		column_idvec_2 = [column_id2,column_id3] # column_idvec_2 = ['peak_id','motif_id']
		df_peak_tf_corr = self.test_query_feature_combine_format_1(df_list=list2,column_idvec=column_idvec_2,field_query=field_query,dropna=False,select_config=select_config)

		print('df_peak_tf_corr: ',df_peak_tf_corr.shape)
		print(df_peak_tf_corr[0:2])

		df_list2 = [df_query1,df_peak_tf_corr]
		column_vec_2 = [field_query]
		type_id_1 = 3
		df_query1 = test_column_query_1(input_filename_list=[],id_column=column_idvec_2,column_vec=column_vec_2,
										df_list=df_list2,type_id_1=type_id_1,type_id_2=0,reset_index=True,select_config=select_config)

		stop = time.time()
		print('query peak accessibility-TF expression correlation used %.2fs'%(stop-start))

		## query motif score normalized
		# field_query = ['motif_score','motif_score_minmax','max_accessibility_score','score_accessibility','score_1']
		field_query1 = ['motif_score','score_normalize_1','score_normalize_pred']
		field_query2 = ['motif_score_minmax','motif_score_log_normalize_bound','score_accessibility','score_1']
		print('query normalized motif score')
		start = time.time()
		print('df_peak_tf_1, df_peak_tf_2: ',df_peak_tf_1.shape,df_peak_tf_2.shape)
		df_peak_tf_1[column_id3] = np.asarray(df_peak_tf_1.index)
		df_peak_tf_2[column_id3] = np.asarray(df_peak_tf_2.index)
		if len(df_peak_tf_1)>0:
			column_vec_2 = [field_query2,field_query1]
			df_peak_tf_1 = df_peak_tf_1.rename(columns={'score_1':'score_normalize_1','score_pred1':'score_normalize_pred'})
			df_list2_1 = [df_query1,df_peak_tf_2,df_peak_tf_1]
		else:
			column_vec_2 = [field_query2]
			df_list2_1 = [df_query1,df_peak_tf_2]
		
		df_query1 = test_column_query_1(input_filename_list=[],id_column=column_idvec_2,column_vec=column_vec_2,
										df_list=df_list2_1,type_id_1=0,type_id_2=0,reset_index=True,select_config=select_config)
		stop = time.time()
		print('query normalized motif score used %.2fs'%(stop-start))

		## query gene-TF correlation
		if len(dict_gene_tf_query)>0:
			print('query gene-TF expression correlation')
			start = time.time()
			field_query = ['gene_tf_corr','gene_tf_pval_corrected']
			list3 = [dict_gene_tf_query[field1] for field1 in field_query]
			column_idvec_3 = [column_id1,column_id3] # column_idvec_2=['gene_id','motif_id']
			
			df_gene_tf_corr = self.test_query_feature_combine_format_1(df_list=list3,column_idvec=column_idvec_3,field_query=field_query,dropna=False,select_config=select_config)

			print('df_gene_tf_corr: ',df_gene_tf_corr.shape)
			print(df_gene_tf_corr[0:2])

			df_list3 = [df_query1,df_gene_tf_corr]
			column_vec_3 = [field_query]
			type_id_1 = 3
			df_query1 = test_column_query_1(input_filename_list=[],id_column=column_idvec_3,column_vec=column_vec_3,
											df_list=df_list3,type_id_1=type_id_1,type_id_2=0,reset_index=True,select_config=select_config)
			stop = time.time()
			print('query gene-TF expression correlation used %.2fs'%(stop-start))

		return df_query1


	## ====================================================
	# initial estimation of peak-tf-gene link query scores
	def test_gene_peak_tf_query_score_init_pre1(self,df_gene_peak_query=[],df_gene_tf_corr_peak=[],lambda1=0.5,lambda2=0.5,type_id_1=0,column_id1=-1,flag_init_score_pre1=0,flag_link_type_query=0,flag_init_score_1=0,flag_save_1=1,flag_save_2=1,flag_save_3=1,save_mode=1,input_file_path='',output_file_path='',output_filename='',verbose=0,select_config={}):

		## peak-tf link, gene-tf link and peak-gene link query
		# flag_init_score_pre1=0
		flag_init_score_pre1=1
		if 'column_idvec' in select_config:
			column_idvec = select_config['column_idvec']
		else:
			column_idvec = ['motif_id','peak_id','gene_id']
			select_config.update({'column_idvec':column_idvec})

		df_link_query_1 = []
		# file_save_path = select_config['data_path_save']
		file_save_path = select_config['data_path_save_local']
		file_save_path2 = select_config['file_path_motif_score']
		input_file_path2 = file_save_path2

		data_file_type_query = select_config['data_file_type_query']
		filename_prefix_pre1 = data_file_type_query
		filename_prefix_default = select_config['filename_prefix_default']
		filename_prefix_default_1 = select_config['filename_prefix_default_1']

		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		iter_mode = 0
		if (query_id1>=0) and (query_id2>query_id1):
			iter_mode = 1
			select_config.update({'iter_mode':iter_mode})

		output_file_path = file_save_path2
		if iter_mode==0:
			filename_prefix_save_pre2 = '%s.pcorr_query1'%(filename_prefix_default_1)
		else:
			filename_prefix_save_pre2 = '%s.pcorr_query1.%d_%d'%(filename_prefix_default_1,query_id1,query_id2)

		if flag_init_score_pre1>0:
			# combine the different scores
			filename_peak_tf_corr = []
			filename_gene_expr_corr = []
			# file_save_path = select_config['data_path_save']
			input_file_path = file_save_path
			print('input_file_path: ',input_file_path)

			field_query_1 = ['gene_tf_corr','peak_tf_corr']
			field_query1, field_query2 = field_query_1[0:2]
			correlation_type = 'spearmanr'
			
			dict_peak_tf_query = self.dict_peak_tf_query
			dict_gene_tf_query = self.dict_gene_tf_query

			# input_file_path = select_config['file_path_motif_score']
			df_peak_tf_1 = self.df_peak_tf_1
			df_peak_tf_2 = self.df_peak_tf_2

			df_link_query_1 = self.test_gene_peak_tf_query_score_init_pre1_2(df_gene_peak_query=df_gene_peak_query,df_peak_tf_1=df_peak_tf_1,df_peak_tf_2=df_peak_tf_2,dict_peak_tf_query=dict_peak_tf_query,dict_gene_tf_query=dict_gene_tf_query,df_gene_tf_corr_peak=df_gene_tf_corr_peak,
																				load_mode=0,save_mode=1,input_file_path='',output_file_path=output_file_path,output_filename='',verbose=verbose,select_config=select_config)

			# flag_save_1=1
			# column_idvec = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_idvec
			if flag_save_1>0:
				output_file_path = file_save_path2
				
				# peak and peak-gene attributes
				field_query1 = ['ratio_1','ratio_cond1'] 
				# peak-tf attributes
				field_query2 = ['motif_score_minmax','motif_score_log_normalize_bound','score_accessibility','score_1','motif_score','score_normalize_1','score_normalize_pred']

				field_query3 = ['peak_tf_corr','peak_tf_pval_corrected',
								'gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1',
								'gene_tf_corr','gene_tf_pval_corrected',
								'peak_gene_corr_','peak_gene_corr_pval']

				field_query_1 = field_query3 + ['motif_score','score_normalize_pred']
				# df_link_query_1.index = np.asarray(df_link_query_1[column_id1])
				# list_1 = [field_query1,field_query2,field_query_1]
				list_1 = [field_query_1,field_query1,field_query2]
				query_num1 = len(list_1)
				column_vec_1 = df_link_query_1.columns
				compression = None
				for i2 in range(query_num1):
					# output_filename = '%s/%s.annot1.1.txt'%(output_file_path,filename_prefix_save_2)
					# output_filename = '%s/%s.annot1_%d.1.txt'%(output_file_path,filename_prefix_save_pre2,i2+1)
					field_query_pre1 = list_1[i2]
					field_query = pd.Index(field_query_pre1).intersection(column_vec_1,sort=False)
					field_query_2 = list(column_idvec)+list(field_query)
					df_link_query_2 = df_link_query_1.loc[:,field_query_2]
					if i2==1:
						df_link_query_2 = df_link_query_2.drop_duplicates(subset=[column_id1,column_id2])	# peak-gene associations
					elif i2==2:
						df_link_query_2 = df_link_query_2.drop_duplicates(subset=[column_id2,column_id3])	# peak-TF associations
					
					if i2==0:
						format_str1 = 'txt.gz'
						compression = 'gzip'
					else:
						format_str1 = 'txt'
						compression = None
					output_filename = '%s/%s.annot1_%d.1.%s'%(output_file_path,filename_prefix_save_pre2,i2+1,format_str1)
					df_link_query_2.to_csv(output_filename,index=False,sep='\t',float_format='%.5f',compression=compression)

		# flag_link_type_query=0
		flag_link_type_query=1
		df_gene_peak_query_pre2=[]
		filename_save_annot_2 = 'annot2'
		column_idvec = select_config['column_idvec']
		# column_idvec = ['motif_id','peak_id','gene_id']
		field_query_1 = ['peak_gene_link','gene_tf_link','peak_tf_link']
		field_query_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']
		if flag_link_type_query>0:
			print('peak-tf-gene link query ')
			start = time.time()
			
			df_gene_peak_query_1_ori = df_gene_peak_query
			
			if len(df_link_query_1)==0:
				if 'filename_gene_tf_peak_query_1' in select_config:
					filename_query_1 = select_config['filename_gene_tf_peak_query_1']
					from utility_1 import test_file_merge_column
					df_link_query_1 = test_file_merge_column(filename_query_1,column_idvec=column_idvec,index_col=False,select_config=select_config)
				else:
					print('please provide df_gene_tf_peak_query file')
			else:
				df_gene_peak_query_pre1_1 = df_link_query_1

			df_gene_peak_query_pre2 = self.test_query_tf_peak_gene_pair_link_type(gene_query_vec=[],peak_query_vec=[],
																					motif_query_vec=[],df_gene_peak_query=df_gene_peak_query_1_ori,
																					df_gene_peak_tf_query=df_gene_peak_query_pre1_1,
																					filename_annot='',select_config=select_config)
			output_file_path = input_file_path2
			# output_filename_1 = '%s/%s.link_thresh_%s.copy1.1.txt'%(output_file_path,filename_prefix_save_pre2,filename_annot_thresh)
			# filename_save_annot_2 = 'annot1'
			# output_filename_1 = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2)
			df_gene_peak_query_pre2.index = np.asarray(df_gene_peak_query_pre2['gene_id'])

			# flag_annot_2=0
			field_query_pre1 = list(column_idvec) + field_query_1 + field_query_2 # link type annotation
			
			field_query_3 = ['ratio_1','ratio_cond1','motif_score_log_normalize_bound','score_accessibility']
			field_query_pre2 = list(column_idvec) + field_query_3
			field_query_pre3 = df_gene_peak_query_pre2.columns.intersection(field_query_pre1,sort=False)
			
			flag_annot_2=0
			if flag_annot_2>0:
				df_gene_peak_query_annot2 = df_gene_peak_query_pre2.loc[:,field_query_pre2]
				
				# output_filename_3 = '%s/%s.%s.3.txt'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2)
				output_filename_3 = '%s/%s.%s_2.1.txt'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2)
				df_gene_peak_query_annot2.index = np.asarray(df_gene_peak_query_annot2['gene_id'])
				df_gene_peak_query_annot2.to_csv(output_filename_3,sep='\t',float_format='%.5f')
				print('df_gene_peak_query_annot2 ',df_gene_peak_query_annot2.shape,df_gene_peak_query_annot2[0:2])
			
			flag_annot_3=1
			if flag_annot_3>0:
				column_query1 = 'group'
				if column_query1 in df_gene_peak_query_pre2:
					field_query_pre1 = field_query_pre1+['group']
				df_gene_peak_query_2 = df_gene_peak_query_pre2.loc[:,field_query_pre1]
				# output_filename_1 = '%s/%s.%s_1.1.txt'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2)
				format_str1 = 'txt.gz'
				compression = 'infer'
				if format_str1 in ['txt.gz']:
					compression = 'gzip'
				output_filename_1 = '%s/%s.%s_1.1.%s'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2,format_str1)
				# df_gene_peak_query_pre2.to_csv(output_filename_1,sep='\t',float_format='%.5f')
				for field_id1 in field_query_1:
					df_gene_peak_query_2[field_id1] = np.int8(df_gene_peak_query_2[field_id1])
				df_gene_peak_query_2.to_csv(output_filename_1,sep='\t',float_format='%.5f',compression=compression)
				print('df_gene_peak_query_2: ',df_gene_peak_query_2.shape)
				print(df_gene_peak_query_2[0:2])
			
			# flag_save_3=1
			# flag_save_3=0
			# df_gene_peak_query_pre2 = df_gene_peak_query_pre2.loc[:,t_columns]
			if flag_save_2>0:
				# df_gene_peak_query_2 = df_gene_peak_query_pre2.loc[:,field_query_pre1]
				# output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2)
				format_str1 = 'txt.gz'
				compression = 'infer'
				if format_str1 in ['txt.gz']:
					compression = 'gzip'
				output_filename = '%s/%s.%s.1.%s'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2,format_str1)
				# df_gene_peak_query_pre2.to_csv(output_filename_1,sep='\t',float_format='%.5f')
				# for field_id1 in field_query_1:
				# 	df_gene_peak_query_pre2[field_id1] = np.int8(df_gene_peak_query_pre2[field_id1])
				df_gene_peak_query_pre2.to_csv(output_filename,sep='\t',float_format='%.5f',compression=compression)
				print('df_gene_peak_query_pre2 ',df_gene_peak_query_pre2.shape,df_gene_peak_query_pre2[0:2])

			stop = time.time()
			print('peak-tf-gene link query: ',stop-start)

		## initial calculation of peak-tf-gene association score
		# add the columns: 'score_pred1_1'
		flag_init_score_1=1
		# flag_init_score_1=0
		if flag_init_score_1>0:
			# filename_save_annot_2 = 'annot1'
			if len(df_gene_peak_query_pre2)==0:
				filename_query_1 = select_config['filename_gene_tf_peak_query_2']
				from utility_1 import test_file_merge_column
				df_gene_peak_query_pre2 = test_file_merge_column(filename_query_1,column_idvec=column_idvec,index_col=False,select_config=select_config)

			print('df_gene_peak_query_pre2 ',df_gene_peak_query_pre2.shape,df_gene_peak_query_pre2[0:2])
			# field_query_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']

			lambda1 = 0.50
			lambda2 = 1-lambda1
			df_gene_peak_query_pre2[field_query_2] = df_gene_peak_query_pre2[field_query_2].fillna(lambda1)

			column_id1 = 'score_pred1_1'
			print('initial calculation of peak-tf-gene association score')
			df_gene_peak_query_1 = self.test_gene_peak_tf_query_score_init_1(df_gene_peak_query=df_gene_peak_query_pre2,
																				lambda1=lambda1,
																				lambda2=lambda2,
																				type_id_1=0,
																				column_id1=column_id1,
																				select_config=select_config)
				
			print('df_gene_peak_query_1 ',df_gene_peak_query_1.shape,df_gene_peak_query_1[0:2])

			# retrieve the columns of score and subset of the annotations
			column_idvec = select_config['column_idvec']
			# field_query_3 = ['motif_id','peak_id','gene_id','score_1','score_combine_1','score_pred1_correlation','score_pred1','score_pred1_1','score_pred2','score_pred_combine']
			if 'column_score_query2' in select_config:
				column_score_query2 = select_config['column_score_query2']
			else:
				column_score_query2 = ['peak_tf_corr','gene_tf_corr_peak','peak_gene_corr_','gene_tf_corr','score_1','score_combine_1','score_normalize_pred','score_pred1_correlation','score_pred1','score_pred1_1','score_combine_2','score_pred2','score_pred_combine']

			field_query_3 = column_idvec + column_score_query2
			df_gene_peak_query1_1 = df_gene_peak_query_1.loc[:,field_query_3]
			# flag_save_3=1
			if flag_save_3>0:
				format_str1 = 'txt.gz'
				compression = 'gzip'
				output_filename = '%s/%s.%s.init.1.%s'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2,format_str1)
				df_gene_peak_query1_1.index = np.asarray(df_gene_peak_query1_1['gene_id'])
				df_gene_peak_query1_1.to_csv(output_filename,index=False,sep='\t',float_format='%.5f',compression=compression)

			return df_gene_peak_query1_1

	## ====================================================
	# estimation of peak-tf-gene link query scores
	def test_gene_peak_tf_query_score_compute_unit_1(self,df_feature_link=[],flag_link_type=0,flag_compute=0,flag_annot_1=0,retrieve_mode=0,save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_query1 = flag_link_type
		df_gene_peak_query_ori = df_feature_link
		lambda1 = 0.50
		lambda2 = 1-lambda1
		if flag_query1>0:
			df_gene_peak_query_pre1 = self.test_query_tf_peak_gene_pair_link_type(gene_query_vec=[],peak_query_vec=[],
																					motif_query_vec=[],df_gene_peak_query=df_gene_peak_query_ori,
																					df_gene_peak_tf_query=df_gene_peak_query_ori,
																					filename_annot='',flag_annot_1=flag_annot_1,verbose=verbose,select_config=select_config)

			# field_query_1 = ['lambda_gene_peak','lambda_peak_tf','lambda_gene_tf_cond','lambda_gene_tf_cond2']
			# df_gene_peak_query_pre1.loc[:,field_query_1] = df_gene_peak_query_pre1.loc[:,field_query_1].fillna(lambda1)

		flag_query2 = flag_compute
		if flag_query2>0:
			column_id1 = 'score_pred1_1'
			print('initial calculation of peak-tf-gene association score')
			df_gene_peak_query_1 = self.test_gene_peak_tf_query_score_init_1(df_gene_peak_query=df_gene_peak_query_pre1,
																				lambda1=lambda1,
																				lambda2=lambda2,
																				type_id_1=0,
																				column_id1=column_id1,
																				select_config=select_config)
				
			print('df_gene_peak_query_1 ',df_gene_peak_query_1.shape,df_gene_peak_query_1[0:2])

			# retrieve_mode: 0, query and save the original link annotations; 1, query the original annotations and save subset of the annotations; 2, query and save subest of annotations; 
			if (retrieve_mode==2) or ((retrieve_mode==1) and (save_mode>0)):
				column_idvec = select_config['column_idvec']
				# field_query_3 = ['motif_id','peak_id','gene_id','score_1','score_combine_1','score_pred1_correlation','score_pred1','score_pred1_1','score_pred2','score_pred_combine']
				field_query = column_idvec+['peak_tf_corr','gene_tf_corr_peak','peak_gene_corr_','gene_tf_corr','score_1','score_combine_1','score_normalize_pred','score_pred1_correlation','score_pred1','score_pred1_1','score_combine_2','score_pred2','score_pred_combine']
				if 'column_score_query2' in select_config:
					field_query = select_config['column_score_query2']

				# field_query_3 = field_query_3+field_query_1+field_query_2
				# field_query_3 = column_idvec+['score_1','score_pred1_correlation','score_pred1','score_pred2','score_pred_combine']
				df_gene_peak_query1_1 = df_gene_peak_query_1.loc[:,field_query]
			
			flag_save_1=1
			# if flag_save_1>0:
			if (save_mode>0) and (output_filename!=''):
				if retrieve_mode in [1,2]:
					df_gene_peak_query_2 = df_gene_peak_query1_1
				else:
					df_gene_peak_query_2 = df_gene_peak_query_1
				df_gene_peak_query_2.to_csv(output_filename,index=False,sep='\t',float_format='%.5f')

			if retrieve_mode==2:
				df_gene_peak_query1 = df_gene_peak_query1_1
			else:
				df_gene_peak_query1 = df_gene_peak_query_1
			
			return df_gene_peak_query1

	## ====================================================
	# initial estimation of peak-tf-gene link query scores
	def test_gene_peak_tf_query_score_init_1(self,df_gene_peak_query=[],lambda1=0.5,lambda2=0.5,type_id_1=0,column_id1=-1,verbose=0,select_config={}):

		flag_query1=1
		if flag_query1>0:
			df_query_1 = df_gene_peak_query
			# lambda_1, lambda_2 = df_query_1['lambda1'], df_query_1['lambda2']

			# lambda1 = 0.50
			# lambda2 = 1-lambda1
			field_query_1 = ['lambda_gene_peak','lambda_peak_tf','lambda_gene_tf_cond','lambda_gene_tf_cond2']
			df_query_1.loc[:,field_query_1] = df_query_1.loc[:,field_query_1].fillna(lambda1)

			lambda_gene_peak, lambda_gene_tf_cond2 = df_query_1['lambda_gene_peak'], df_query_1['lambda_gene_tf_cond2']
			lambda_peak_tf, lambda_gene_tf_cond = df_query_1['lambda_peak_tf'], df_query_1['lambda_gene_tf_cond']
				
			field_query = ['column_peak_tf_corr','column_peak_gene_corr','column_query_cond']
			list1 = ['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak']
			# column_peak_tf_corr = 'peak_tf_corr'
			# column_peak_gene_corr = 'peak_gene_corr_'
			# column_query_cond = 'gene_tf_corr_peak'
			query_num1 = len(field_query)
			for i1 in range(query_num1):
				field_id = field_query[i1]
				if field_id in select_config:
					list1[i1] = select_config[field_id]

			column_1, column_2, column_3 = list1[0:3]
			# peak_tf_corr = df_query_1['peak_tf_corr']
			# peak_gene_corr_, gene_tf_corr_peak = df_query_1['peak_gene_corr_'], df_query_1['gene_tf_corr_peak']
			peak_tf_corr = df_query_1[column_1]
			peak_gene_corr_, gene_tf_corr_peak = df_query_1[column_2], df_query_1[column_3]
			
			## recompute score_pred2
			score_combine_1 = lambda_peak_tf*peak_tf_corr+lambda_gene_tf_cond*gene_tf_corr_peak
			# score_pred2_recompute= score_combine_1*df_query_1['score_1']
			score_1 = df_query_1['score_1']
			score_pred1_correlation = peak_tf_corr*score_1
			score_pred1 = score_combine_1*score_1
			df_query_1['score_combine_1'] = score_combine_1 # the score 1 before the normalization

			column_score_1 = 'score_pred1'
			column_score_2 = 'score_pred2'
			if 'column_score_1' in select_config:
				column_score_1 = select_config['column_score_1']
			if 'column_score_2' in select_config:
				column_score_2 = select_config['column_score_2']

			# df_query_1['score_pred2_recompute'] = score_pred2_recompute
			df_query_1['score_pred1_correlation'] = score_pred1_correlation
			# df_query_1['score_pred1'] = score_pred1
			df_query_1[column_score_1] = score_pred1

			score_combine_2 = lambda_gene_peak*peak_gene_corr_ +lambda_gene_tf_cond2*gene_tf_corr_peak # the score 2 before the normalization
			df_query_1['score_combine_2'] = score_combine_2
			# df_query_1['score_pred2'] = score_combine_2*score_1
			df_query_1[column_score_2] = score_combine_2*score_1

			if column_id1==-1:
				column_id1='score_pred1_1'
			
			df_gene_peak_query[column_id1] = ((lambda_peak_tf*peak_tf_corr).abs()+(lambda_gene_tf_cond*gene_tf_corr_peak).abs())*score_1
			
			a1 = 2/3.0
			score_pred_combine = (lambda_peak_tf*peak_tf_corr+0.5*(lambda_gene_tf_cond+lambda_gene_tf_cond2)*gene_tf_corr_peak+lambda_gene_peak*peak_gene_corr_)*a1
			df_query_1['score_pred_combine'] = score_pred_combine*score_1

			field_query_1 = ['peak_gene_link','gene_tf_link','peak_tf_link']
			field_query_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']
			field_query = field_query_1+field_query_2
			column_idvec = ['motif_id','peak_id','gene_id']

			df_gene_peak_query = df_query_1

		return df_gene_peak_query

	## ====================================================
	# save initial peak-tf-gene link score
	def test_gene_peak_tf_query_score_init_save(self,df_gene_peak_query=[],lambda1=0.5,lambda2=0.5,
													flag_init_score=1,flag_save_interval=1,
													feature_type='gene_id',
													save_mode=1,query_mode=0,output_file_path='',output_filename='',
													filename_prefix_save='',float_format='%.5f',select_config={}):

		## initial calculation of peak-tf-gene association score
		if flag_init_score>0:
			df_gene_peak_query_1 = self.test_gene_peak_tf_query_score_init_1(df_gene_peak_query=df_gene_peak_query,
																				lambda1=lambda1,
																				lambda2=lambda2,
																				select_config=select_config)
		else:
			df_gene_peak_query_1 = df_gene_peak_query

		# flag_save_interval, flag_save_ori = 0, 0
		if flag_save_interval>0:
			# interval = 5000
			interval = 2500
			# feature_type = 'gene_id'
			list_query_interval = self.test_gene_peak_tf_query_save_interval_1(df_gene_peak_query=df_gene_peak_query_1,
																				interval=interval,
																				feature_type=feature_type,
																				save_mode=save_mode,
																				query_mode=query_mode,
																				output_file_path=output_file_path,
																				filename_prefix_save=filename_prefix_save,
																				select_config=select_config)
		
		# if flag_save_ori>0:
		if save_mode>0:
			# output_filename = '%s/%s.init.1.txt'%(output_file_path,filename_prefix_save)
			df_gene_peak_query_1.to_csv(output_filename,sep='\t',float_format='%.5f')

		return df_gene_peak_query_1

	## ====================================================
	# peak-tf link, gene-tf link and peak-gene link query
	# adjust the lambda based on the link type
	def test_query_tf_peak_gene_pair_link_type(self,gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],df_gene_peak_query=[],df_gene_peak_tf_query=[],column_idvec=[],
												motif_data=[],peak_read=[],rna_exprs=[],filename_annot='',reset_index=True,flag_annot_1=1,type_query=0,save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		df_query_1 = df_gene_peak_tf_query
		from utility_1 import test_query_index
		if len(column_idvec)==0:
			column_idvec = ['motif_id','peak_id','gene_id']
		if reset_index==True:
			df_query_1.index = test_query_index(df_query_1,column_vec=column_idvec)
		
		field_link_query1, field_link_query2 = [], []
		if 'field_link_query1' in select_config:
			field_link_query1 = select_config['field_link_query1']
		else:
			field_link_query1 = ['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak','gene_tf_corr']

		if 'field_link_query2' in select_config:
			field_link_query2 = select_config['field_link_query2']
		else:
			column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
			# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
			if 'column_pval_cond' in select_config:
				column_pval_cond = select_config['column_pval_cond']
			field_link_query2 = ['peak_tf_pval_corrected','peak_gene_corr_pval',column_pval_cond,'gene_tf_pval_corrected']

		field_link_query_1 = field_link_query1 + field_link_query2
		column_motif_1 = 'motif_score_log_normalize_bound'
		column_motif_2 = 'score_accessibility'
		field_motif_score = [column_motif_1,column_motif_2]
		# select_config.update({'field_link_query1':field_link_query1,'field_link_query2':field_link_query2})
		df_link_query_1 = df_query_1
		if 'flag_annot_link_type' in select_config:
			flag_annot_1 = select_config['flag_annot_link_type']

		if flag_annot_1>0:
			flag_annot1=0
			df_1 = df_query_1
			column_vec = df_1.columns
			# there are columns not included in the current dataframe
			t_columns_1 = pd.Index(field_link_query_1).difference(column_vec,sort=False)
			t_columns_2 = pd.Index(field_motif_score).difference(column_vec,sort=False)
			df_list1 = [df_1]
			print('annotation query: ',t_columns_1,t_columns_2)
			
			column_vec_1 = []
			if len(t_columns_1)>0:
				flag_annot1 = 1
				# query correlation and p-value
				if 'filename_annot1' in select_config:
					filename_annot1 = select_config['filename_annot1']
					df_2 = pd.read_csv(filename_annot1,index_col=False,sep='\t')
					print('df_2: ',df_2.shape)
					print(filename_annot1)
					print(df_2.columns)
					print(df_2[0:2])
					# df_list1 = [df_1,df_2]
					df_list1 = df_list1+[df_2]
					column_vec_1.append(t_columns_1)
				else:
					print('please provide annotation file')
					return
				
			if len(t_columns_2)>0:
				flag_annot1 = 1
				# query motif score annotation
				if 'filename_motif_score' in select_config:
					filename_annot2 = select_config['filename_motif_score']
					df_3 = pd.read_csv(filename_annot2,index_col=False,sep='\t')
					print('df_3: ',df_3.shape)
					print(filename_annot2)
					print(df_3.columns)
					print(df_3[0:2])
					df_list1 = df_list1+[df_3]
					column_vec_1.append(t_columns_2)
				else:
					print('please provide annotation file')
					return

			if flag_annot1>0:
				column_idvec_1 = column_idvec
				# column_vec_1 = [[column_pval_cond]]
				df_1.index = utility_1.test_query_index(df_1,column_vec=column_idvec)
				df_link_query_1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec_1,column_vec=column_vec_1,
																df_list=df_list1,type_id_1=0,type_id_2=0,reset_index=False,select_config=select_config)

		column_peak_tf_corr, column_peak_gene_corr, column_query_cond, column_gene_tf_corr = field_link_query1
		column_peak_tf_pval, column_peak_gene_pval, column_pval_cond, column_gene_tf_pval = field_link_query2

		df_query_1 = df_link_query_1
		peak_tf_corr, gene_tf_corr_peak, peak_gene_corr_ = df_query_1[column_peak_tf_corr], df_query_1[column_query_cond], df_query_1[column_peak_gene_corr]
		peak_tf_pval_corrected, gene_tf_corr_peak_pval_corrected, peak_gene_corr_pval = df_query_1[column_peak_tf_pval], df_query_1[column_pval_cond], df_query_1[column_peak_gene_pval]
		gene_tf_corr_, gene_tf_corr_pval_corrected = df_query_1[column_gene_tf_corr], df_query_1[column_gene_tf_pval]

		flag_query1=1
		if flag_query1>0:
			if not ('config_link_type' in select_config):
				# thresh_corr_1, thresh_pval_1 = 0.05, 0.1 # the previous parameter
				thresh_corr_1, thresh_pval_1 = 0.1, 0.1  # the alternative parameter to use; use relatively high threshold for negative peaks
				thresh_corr_2, thresh_pval_2 = 0.1, 0.05 # stricter p-value threshold for negative-correlated peaks
				thresh_corr_3, thresh_pval_3 = 0.15, 1
				thresh_motif_score_neg_1, thresh_motif_score_neg_2 = 0.80, 0.90 # higher threshold for regression mechanism
				# thresh_score_accessibility = 0.1
				thresh_score_accessibility = 0.25
				thresh_list_query = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2],[thresh_corr_3, thresh_pval_3]]

				config_link_type = {'thresh_list_query':thresh_list_query,
										'thresh_motif_score_neg_1':thresh_motif_score_neg_1,
										'thresh_motif_score_neg_2':thresh_motif_score_neg_2,
										'thresh_score_accessibility':thresh_score_accessibility}
				select_config.update({'config_link_type':config_link_type})
			else:
				config_link_type = select_config['config_link_type']
				thresh_list_query = config_link_type['thresh_list_query']
				thresh_corr_1, thresh_pval_1 = thresh_list_query[0]
				thresh_corr_2, thresh_pval_2 = thresh_list_query[1]
				thresh_corr_3, thresh_pval_3 = thresh_list_query[2]
				thresh_motif_score_neg_1 ,thresh_motif_score_neg_2 = config_link_type['thresh_motif_score_neg_1'], config_link_type['thresh_motif_score_neg_2']
				thresh_score_accessibility = config_link_type['thresh_score_accessibility']

			print('config_link_type: ',config_link_type)

			# column_1 = 'motif_score_normalize_bound'
			# column_2 = 'score_accessibility'
			motif_score_query = df_query_1['motif_score_log_normalize_bound']
			id_motif_score_1 = (motif_score_query>thresh_motif_score_neg_1)
			id_motif_score_2 = (motif_score_query>thresh_motif_score_neg_2) # we use id_motif_score_2
			id_score_accessibility = (df_query_1['score_accessibility']>thresh_score_accessibility)
			
			if ('field_link_1' in select_config):
				field_link_1 = select_config['field_link_1']
				t_columns = field_link_1
			else:
				t_columns = ['peak_gene_link','gene_tf_link','peak_tf_link']
			
			query_num1 = len(t_columns)
			list1 = [[peak_gene_corr_,peak_gene_corr_pval],[gene_tf_corr_peak,gene_tf_corr_peak_pval_corrected],[peak_tf_corr,peak_tf_pval_corrected]]
			dict1 = dict(zip(t_columns,list1))

			df_query_1.loc[:,t_columns] = 0,0,0
			for i1 in range(query_num1):
				column_1 = t_columns[i1]
				corr_query, pval_query = dict1[column_1]
				id1_query_pval = (pval_query<thresh_pval_1)
				id1_query_1 = id1_query_pval&(corr_query>thresh_corr_1)
				id1_query_2 = id1_query_pval&(corr_query<-thresh_corr_1)

				id1_query_1 = id1_query_1|(corr_query>thresh_corr_3) # only query correlation, without threshold on p-value
				id1_query_2 = id1_query_2|(corr_query<-thresh_corr_3) # only query correlation, without threshold on p-value
				df_query_1.loc[id1_query_1,column_1] = 1
				df_query_1.loc[id1_query_2,column_1] = -1
				df_query_1[column_1] = np.int32(df_query_1[column_1])
				print('column_1, id1_query1, id1_query2 ',column_1,i1,np.sum(id1_query_1),np.sum(id1_query_2))

			# self.test_query_tf_peak_gene_pair_pre1(select_config=select_config)
			# lambda1 = 0.5
			# lambda2 = 1-lambda1
			lambda_gene_peak = 0.5 # lambda1: peak-gene link query
			lambda_gene_tf_cond2 = 1-lambda_gene_peak # lambda2: gene-tf link query
			lambda_peak_tf = 0.5
			lambda_gene_tf_cond = 1-lambda_peak_tf
			peak_tf_link, gene_tf_link, peak_gene_link = df_query_1['peak_tf_link'], df_query_1['gene_tf_link'], df_query_1['peak_gene_link']
			
			t_columns = ['peak_gene_link','gene_tf_link','peak_tf_link','gene_tf_link_1']
			query_num1 = len(t_columns)
			# list1 = [[peak_gene_corr_,peak_gene_corr_pval],[gene_tf_corr_peak,gene_tf_corr_peak_pval_corrected],[peak_tf_corr,peak_tf_pval_corrected]]
			list1 = list1+[[gene_tf_corr_, gene_tf_corr_pval_corrected]]
			dict1 = dict(zip(t_columns,list1))

			list2 = []
			for i1 in range(query_num1):
				column_1 = t_columns[i1]
				corr_query, pval_query = dict1[column_1]
				id1_query_pval = (pval_query<thresh_pval_2)
				id1_query_1 = id1_query_pval&(corr_query>thresh_corr_2)
				id1_query_2 = id1_query_pval&(corr_query<(-thresh_corr_2))

				id1_query_1 = id1_query_1|(corr_query>thresh_corr_3) # only query correlation, without threshold on p-value
				id1_query_2 = id1_query_2|(corr_query<-thresh_corr_3) # only query correlation, without threshold on p-value
				list2.append([id1_query_1,id1_query_2])

			## gene-tf expression correlation not conditioned on peak accessibility
			# group: thresh 1: p-value threshold 1
			id_gene_tf_corr_neg_thresh1 = ((gene_tf_corr_<(-thresh_corr_2))&(gene_tf_corr_pval_corrected<thresh_pval_1))	# use higher threshold
			id_gene_tf_corr_pos_thresh1 = (gene_tf_corr_>thresh_corr_2)&(gene_tf_corr_pval_corrected<thresh_pval_1)

			## gene-tf epxression correlation conditioned on peak accessibility
			id_gene_tf_corr_peak_neg_thresh1 = (gene_tf_corr_peak<(-thresh_corr_2))&(gene_tf_corr_peak_pval_corrected<thresh_pval_1)
			id_gene_tf_corr_peak_pos_thresh1 = (gene_tf_corr_peak>thresh_corr_2)&(gene_tf_corr_peak_pval_corrected<thresh_pval_1)

			# group: thresh 2: p-value threshold 2 (stricter threshold)
			id_gene_tf_corr_pos_thresh2, id_gene_tf_corr_neg_thresh2 = list2[3]

			id_gene_tf_corr_peak_pos_thresh2, id_gene_tf_corr_peak_neg_thresh2 = list2[1]

			## peak-tf correlation 
			id_peak_tf_corr_pos_thresh2, id_peak_tf_corr_neg_thresh2 = list2[2]

			## peak-gene correlation
			id_peak_gene_pos_thresh2, id_peak_gene_neg_thresh2 = list2[0]

			list_pre1 = [id_gene_tf_corr_pos_thresh2, id_gene_tf_corr_pos_thresh1]
			list_pre2 = [id_gene_tf_corr_neg_thresh2, id_gene_tf_corr_neg_thresh1]
			id_gene_tf_corr_pos_thresh_query = list_pre1[type_query]
			id_gene_tf_corr_neg_thresh_query = list_pre2[type_query]
			print('id_gene_tf_corr_pos_thresh, id_gene_tf_corr_neg_thresh: ',id_gene_tf_corr_pos_thresh_query,id_gene_tf_corr_neg_thresh_query)

			## repression with peak-tf correlation above zero
			id1 = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link<0)	# neg-pos: repression (negative peak, positive peak-tf correlation); the previous threshold
			id1 = (id1&id_gene_tf_corr_neg_thresh_query)	# change the threshold to be stricter
			
		
			## repression with peak-tf correlation above zero but not significant partial gene-tf correlation conditioned on peak accessibility
			# id1_1 = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link==0)&(gene_tf_corr_<-thresh_corr_2)&(gene_tf_corr_pval<thresh_pval_1)	# neg-pos: repression (negative peak, positive peak-tf correlation)
			# id1_1 = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link==0)&(id_gene_tf_corr_neg_thresh1)	# neg-pos: repression (negative peak, positive peak-tf correlation)
			id1_1 = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link==0)&(id_gene_tf_corr_neg_thresh_query)	# neg-pos: repression (negative peak, positive peak-tf correlation)

			id1_2 = (id1|id1_1)
			df_query_1.loc[id1_2,'lambda_gene_peak'] = -lambda_gene_peak
			df_query_1.loc[id1_2,'lambda_gene_tf_cond2'] = -lambda_gene_tf_cond2
			df_query_1.loc[id1_2,'lambda_gene_tf_cond'] = -lambda_gene_tf_cond

			## repression with peak-tf correlation under zero
			id2 = (peak_tf_link<0)&(peak_gene_link>0)&(gene_tf_link>0)	# contradiction
			# df_query_1.loc[id2,'lambda_gene_tf_cond2'] = -lambda_gene_tf_cond2
			# df_query_1.loc[id2,'lambda_gene_tf_cond'] = -lambda_gene_tf_cond
			df_query_1.loc[id2,'lambda_gene_tf_cond2'] = 0
			df_query_1.loc[id2,'lambda_gene_tf_cond'] = 0

			## repression with peak-tf correlation under zero;
			id_2_ori = (peak_tf_link<0)&(peak_gene_link>=0)&(gene_tf_link<0)	# pos-neg: repression (positive peak accessibility-gene expr. correlation, negative peak accessibility-tf expr. correlation)
			id_link_2 = (id_2_ori&id_motif_score_2&id_score_accessibility)	# use higher threshold
			
			# id_2 = (id_2&id_peak_tf_corr_2&id_gene_tf_corr_2) # the previous threshold
			# id_2 = (id_link_2&id_peak_tf_corr_neg_thresh2&id_gene_tf_corr_peak_neg_thresh2&id_gene_tf_corr_neg_thresh1) # change the threshold to be stricter
			id_2 = (id_link_2&id_peak_tf_corr_neg_thresh2&id_gene_tf_corr_peak_neg_thresh2&id_gene_tf_corr_neg_thresh_query) # change the threshold to be stricter
			df_query_1.loc[id_2,'lambda_gene_tf_cond2'] = -lambda_gene_tf_cond2
			df_query_1.loc[id_2,'lambda_peak_tf'] = -lambda_peak_tf
			df_query_1.loc[id_2,'lambda_gene_tf_cond'] = -lambda_gene_tf_cond

			## up-regulation with peak-tf correlation under zero
			# the group may be of lower probability
			id3_ori = (peak_tf_link<0)&(peak_gene_link<0)&(gene_tf_link>0)	# neg-neg: activation (negative peak, negative peak-tf correlation)
			
			id_link_3 = id3_ori&(id_motif_score_2)&(id_score_accessibility)	# use higher threshold; the previous threshold
			id3 = (id_link_3&id_peak_tf_corr_neg_thresh2&id_peak_gene_neg_thresh2&id_gene_tf_corr_peak_pos_thresh2&id_gene_tf_corr_pos_thresh_query)
			df_query_1.loc[id3,'lambda_gene_peak'] = -lambda_gene_peak
			df_query_1.loc[id3,'lambda_peak_tf'] = -lambda_peak_tf

			## up-regulation with peak-tf correlation above zero but peak-gene correlation under zero
			# the peak may be linked with other gene query
			id5_ori = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link>0)	# pos-neg: contraction (negative peak, positive peak-tf correlation, positive tf-gene correlation)
			# id5 = (id5_ori&id_peak_tf_corr_pos_thresh2&id_peak_gene_neg_thresh2&id_gene_tf_corr_peak_pos_thresh2&id_gene_tf_corr_pos_thresh1)
			id5 = (id5_ori&id_peak_tf_corr_pos_thresh2&id_peak_gene_neg_thresh2&id_gene_tf_corr_peak_pos_thresh2&id_gene_tf_corr_pos_thresh_query)

			df_query_1.loc[id5,'lambda_gene_tf_cond2'] = 0
			df_query_1.loc[id5,'lambda_gene_tf_cond'] = 0

			# the groups of peak-gene links that not use the default lambda
			list_query1 = [id1_2,id2,id_2,id3,id5]
			query_num1 = len(list_query1)
			# query_id1 = df_query_1.index
			for i2 in range(query_num1):
				id_query = list_query1[i2]
				t_value_1 = np.sum(id_query)
				df_query_1.loc[id_query,'group'] = (i2+1)
				print('group %d: %d'%(i2+1,t_value_1))

			# return df_query_1, df_query_2
			return df_query_1

	## score query by specific column of dataframe
	def test_score_query_1(self,df_query,column_id_query,thresh_vec,quantile_query=1,quantile_vec=[],scale_type_id=0,select_config={}):

		query_id1 = df_query.index
		if len(quantile_vec)==0:
			quantile_vec_1 = [0.05,0.1,0.25,0.5,0.75,0.90,0.95]
		else:
			quantile_vec_1 = quantile_vec

		score_query = df_query[column_id_query]
		if quantile_query>0:
			t_value1 = utility_1.test_stat_1(score_query,quantile_vec=quantile_vec_1)
			list1_quantile = list(t_value1)
			list1_quantile = list1_quantile+[len(query_id1)]

		## thresholding based on specific score query
		thresh_lower_bound, thresh_upper_bound, thresh_quantile = thresh_vec
		if thresh_quantile>0:
			thresh_1_ori = np.quantile(score_query,thresh_quantile)
			thresh_1 = thresh_1_ori
			if thresh_lower_bound>-1:
				thresh_1 = np.min([np.max([thresh_lower_bound,thresh_1_ori]),thresh_upper_bound])
		else:
			thresh_1_ori = thresh_lower_bound
			thresh_1 = thresh_1_ori

		if np.isnan(thresh_1)==True:
			# thresh_1 = 0.15
			thresh_1 = thresh_lower_bound
		id_query = (score_query>thresh_1)
		query_id2 = query_id1[id_query]
		z_score_1=[]
		if scale_type_id>0:
			z_score_1 = scale(score_query)
			# df_query.loc[query_id1,'%s_scale'%(column_id_query)] = z_score_1

		thresh_query_vec = [thresh_1,thresh_1_ori]
		return (query_id2,id_query,list1_quantile,thresh_query_vec,z_score_1)

	## score query by quantile
	def test_score_query_pre2(self,data=[],feature_query='',feature_query_id=0,column_id_query='',column_query_vec=[],column_label_vec=[],verbose=0,select_config={}):

		df_feature = data
		id1 = (df_feature[column_id_query]==feature_query)
		# df_query1 = df_feature.loc[feature_query1,:]
		df_query1 = df_feature.loc[id1,:]

		# query_id1 = df_query1['query_id']
		query_id_1 = df_feature.index
		query_id1 = query_id_1[id1]
		query_num1 = len(query_id1)
		if (verbose>0) and (feature_query_id%100==0):
			print('feature_query: ',feature_query,feature_query_id,query_num1)

		normalize_type = 'uniform'
		score_query = df_query1.loc[:,column_query_vec]
		num_quantiles = np.min([query_num1,1000])
		# score_mtx = quantile_transform(score_query,n_quantiles=1000,output_distribution=normalize_type)
		score_mtx = quantile_transform(score_query,n_quantiles=num_quantiles,output_distribution=normalize_type)
		score_mtx = pd.DataFrame(index=query_id1,columns=column_label_vec,data=np.asarray(score_mtx),dtype=np.float32)

		return score_mtx

	## score query by quantile
	def test_score_query_2(self,data=[],feature_query_vec=[],column_id_query='',column_idvec=[],column_query_vec=[],column_label_vec=[],flag_annot=1,reset_index=1,parallel_mode=0,interval=100,verbose=0,select_config={}):

		df_feature = data
		if len(feature_query_vec)==0:
			feature_query_vec = df_feature[column_id_query].unique()
		feature_query_num = len(feature_query_vec)

		# column_query_vec_1 = [column_query1,column_query2]
		column_num1 = len(column_query_vec)
		print('feature_query_vec: ',feature_query_num)
		print('column_id_query: ',column_id_query)

		if reset_index>0:
			query_id_ori = df_feature.index.copy()
			df_feature.index = utility_1.test_query_index(df_feature,column_vec=column_idvec)

		# query_id_ori = pd.Index(query_id_ori)
		# df_feature.index = np.asarray(df_feature[column_id_query])
		query_id_1 = df_feature.index

		if parallel_mode>0:
			if interval<0:
				score_query_1 = Parallel(n_jobs=-1)(delayed(self.test_score_query_pre2)(data=df_feature,feature_query=feature_query_vec[id1],feature_query_id=id1,
																							column_id_query=column_id_query,column_query_vec=column_query_vec,column_label_vec=column_label_vec,
																							verbose=verbose,select_config=select_config) for id1 in range(feature_query_num))
				query_num1 = len(score_query_1)
				for i1 in range(query_num1):
					score_mtx = score_query_1[i1]
					query_id1 = score_mtx.index
					df_feature.loc[query_id1,column_label_vec] = score_mtx

			else:
				iter_num = int(np.ceil(feature_query_num/interval))
				print('iter_num, interval: ',iter_num,interval)
				for iter_id in range(iter_num):
					start_id1, start_id2 = interval*iter_id, np.min([interval*(iter_id+1),feature_query_num])
					print('start_id1, start_id2: ',start_id1,start_id2,iter_id)
					query_vec_1 = np.arange(start_id1,start_id2)
					# estimate feature score quantile
					score_query_1 = Parallel(n_jobs=-1)(delayed(self.test_score_query_pre2)(data=df_feature,feature_query=feature_query_vec[id1],feature_query_id=id1,
																							column_id_query=column_id_query,column_query_vec=column_query_vec,column_label_vec=column_label_vec,
																							verbose=verbose,select_config=select_config) for id1 in query_vec_1)
					query_num1 = len(score_query_1)
					for i1 in range(query_num1):
						score_mtx = score_query_1[i1]
						query_id1 = score_mtx.index
						df_feature.loc[query_id1,column_label_vec] = score_mtx
		else:
			for i1 in range(feature_query_num):
			# for i1 in range(10):
				feature_query1 = feature_query_vec[i1]
				feature_query_id1 = i1
				# estimate feature score quantile
				score_mtx = self.test_score_query_pre2(data=df_feature,feature_query=feature_query1,feature_query_id=feature_query_id1,
														column_id_query=column_id_query,column_query_vec=column_query_vec,column_label_vec=column_label_vec,verbose=verbose,select_config=select_config)
				
				if (i1%500==0):
					print('df_feature: ',df_feature.shape,feature_query1,i1)
					print(df_feature[0:2])
					print('score_mtx: ',score_mtx.shape)
					print(score_mtx[0:2])
				query_id1 = score_mtx.index
				try:
					df_feature.loc[query_id1,column_label_vec] = score_mtx
				except Exception as error:
					print('error! ',error,feature_query1,i1,len(query_id1))
					# query_id_1 = df1.index
					# id1 = df1.index.duplicated(keep='first')
					query_id_2 = query_id1.unique()
					df1 = df_feature.loc[query_id_2,:]
					print('df1 ',df1.shape)
					# query_id2 = query_id_1[df1.index.duplicated(keep='first')]
					df2 = df1.loc[df1.index.duplicated(keep=False),:]
					query_id2 = df2.index.unique()
					print('query_id_2: ',len(query_id_2))
					print('duplicated idvec, query_id2: ',len(query_id2))
					file_save_path2 = select_config['file_path_motif_score']
					output_filename = '%s/test_query_score_quantile.duplicated.query1.%s.1.txt'%(file_save_path2,feature_query1)
					if os.path.exists(output_filename)==True:
						print('the file exists: %s'%(output_filename))
						filename1 = output_filename
						b = filename1.find('.txt')
						output_filename = filename1[0:b]+'.copy1.txt'
					df2.to_csv(output_filename,sep='\t')
					df_feature = df_feature.loc[(~df_feature.index.duplicated(keep='first')),:]
					# return

		if reset_index==1:
			df_feature.index = query_id_ori

		return df_feature

	## gene-peak-tf association: the difference between gene_tf_corr_peak and gene-tf expression correlation
	def test_gene_peak_tf_query_compare_1(self,input_filename='',df_gene_peak_query=[],df_annot=[],thresh_corr_1=0.30,thresh_corr_2=0.05,save_mode=0,output_file_path='',filename_prefix_save='',select_config={}):

		if len(df_gene_peak_query)==0:
			# df_gene_peak_query_pre1 = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_gene_peak_query_pre1 = pd.read_csv(input_filename,index_col=False,sep='\t')
		else:
			df_gene_peak_query_pre1 = df_gene_peak_query

		from utility_1 import test_query_index
		df_gene_peak_query_pre1.index = test_query_index(df_gene_peak_query_pre1,column_vec=['motif_id','peak_id','gene_id'])
		print('df_gene_peak_query_pre1 ',df_gene_peak_query_pre1.shape)

		thresh_corr_query_1, thresh_corr_compare_1 = thresh_corr_1, thresh_corr_2

		column_query_cond = 'gene_tf_corr_peak'
		column_gene_tf_corr = 'gene_tf_corr'
		if 'column_query_cond' in select_config:
			column_query_cond = select_config['column_query_cond']

		if 'column_gene_tf_corr' in select_config:
			column_gene_tf_corr = select_config['column_gene_tf_corr']

		id1 = (df_gene_peak_query_pre1[column_query_cond].abs()>thresh_corr_query_1)
		id2 = (df_gene_peak_query_pre1[column_gene_tf_corr].abs()<thresh_corr_compare_1)
		id3 = (id1&id2)
		id_pre1 = (~id3)
		df_query_1 = df_gene_peak_query_pre1.loc[id_pre1,:]
		df_query_2 = df_gene_peak_query_pre1.loc[id3,:]
		query_id_ori = df_gene_peak_query_pre1.index
		query_id_1 = query_id_ori[id_pre1]
		query_id_2 = query_id_ori[id3]

		if (save_mode>0) and (output_file_path!=''):
			filename_annot_save = '%s_%s'%(thresh_corr_1,thresh_corr_2)
			output_filename_1 = '%s/%s.%s.subset1.txt'%(output_file_path,filename_prefix_save,filename_annot_save)
			df_query_1.index = np.asarray(df_query_1['gene_id'])		
			output_filename_2 = '%s/%s.%s.subset2.txt'%(output_file_path,filename_prefix_save,filename_annot_save)
			df_query_2.index = np.asarray(df_query_2['gene_id'])
			df_query_2.to_csv(output_filename_2,sep='\t',float_format='%.5f')
			print('df_query_1', df_query_1.shape)
			print('df_query_2', df_query_2.shape)

		return df_gene_peak_query_pre1, query_id_1, query_id_2

	## parameter configuration for feature score computation
	def test_query_score_config_1(self,column_pval_cond='',thresh_corr_1=0.1,thresh_pval_1=0.1,overwrite=False,flag_config_1=1,flag_config_2=1,save_mode=1,verbose=0,select_config={}):

		if flag_config_1>0:
			filename_prefix_default_1 = select_config['filename_prefix_default_1']
			filename_prefix_score = '%s.pcorr_query1'%(filename_prefix_default_1)
			filename_annot_score_1 = 'annot2.init.1'
			filename_annot_score_2 = 'annot2.init.query1'
			select_config.update({'filename_prefix_score':filename_prefix_score,'filename_annot_score_1':filename_annot_score_1,'filename_annot_score_2':filename_annot_score_2})

			correlation_type = 'spearmanr'
			column_idvec = ['motif_id','peak_id','gene_id']
			column_gene_tf_corr_peak =  ['gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1']
			# thresh_insilco_ChIP_seq = 0.1
			flag_save_text = 1

			field_query = ['column_idvec','correlation_type','column_gene_tf_corr_peak','flag_save_text']
			list_1 = [column_idvec,correlation_type,column_gene_tf_corr_peak,flag_save_text]
			field_num1 = len(field_query)
			for i1 in range(field_num1):
				field1 = field_query[i1]
				if (not (field1 in select_config)) or (overwrite==True):
					select_config.update({field1:list_1[i1]})

			field_query = ['column_peak_tf_corr','column_peak_gene_corr','column_query_cond','column_gene_tf_corr','column_score_1','column_score_2']
			column_score_1, column_score_2 = 'score_pred1', 'score_pred2'
			column_vec_1 = ['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak','gene_tf_corr']
			# list1 = ['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak','gene_tf_corr',column_score_1,column_score_2]
			list1 = column_vec_1+[column_score_1,column_score_2]

			if column_pval_cond=='':
				# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
				column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
			
			field_query_2 = ['column_peak_tf_pval','column_peak_gene_pval','column_pval_cond','column_gene_tf_pval']
			list2 = ['peak_tf_pval_corrected','peak_gene_corr_pval',column_pval_cond,'gene_tf_pval_corrected']

			field_query = field_query + field_query_2
			list1 = list1 + list2
			query_num1 = len(field_query)
			for (field_id,query_value) in zip(field_query,list1):
				select_config.update({field_id:query_value})

			column_score_query = [column_score_1,column_score_2,'score_pred_combine']
			select_config.update({'column_score_query':column_score_query})

			field_link_1 = ['peak_gene_link','gene_tf_link','peak_tf_link']
			field_link_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']
			select_config.update({'field_link_1':field_link_1,'field_link_2':field_link_2})

			column_idvec = ['motif_id','peak_id','gene_id']
			# column_score_query_1 = column_idvec + ['score_1','score_combine_1','score_pred1_correlation','score_pred1','score_pred1_1','score_pred2','score_pred_combine']
			
			column_score_query_pre1 = column_idvec + ['score_1','score_combine_1','score_pred1_correlation','score_pred1_1']+column_score_query
			column_annot_1 = ['feature1_score1_quantile', 'feature1_score2_quantile','feature2_score1_quantile','peak_tf_corr_thresh1','peak_gene_corr_thresh1','gene_tf_corr_peak_thresh1']
			
			column_score_query_1 = column_idvec + ['score_1','score_pred1_correlation','score_pred1_1'] + column_score_query + column_annot_1
			select_config.update({'column_idvec':column_idvec,
									'column_score_query_pre1':column_score_query_pre1,
									'column_score_query_1':column_score_query_1})

			if (not ('config_link_type' in select_config)) or (overwrite==True):
				# thresh_corr_1, thresh_pval_1 = 0.05, 0.1 # the previous parameter
				thresh_corr_1, thresh_pval_1 = 0.1, 0.1  # the alternative parameter to use; use relatively high threshold for negative peaks
				thresh_corr_2, thresh_pval_2 = 0.1, 0.05
				thresh_corr_3, thresh_pval_3 = 0.15, 1
				thresh_motif_score_neg_1, thresh_motif_score_neg_2 = 0.80, 0.90 # higher threshold for regression mechanism
				# thresh_score_accessibility = 0.1
				thresh_score_accessibility = 0.25
				thresh_list_query = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2],[thresh_corr_3, thresh_pval_3]]

				config_link_type = {'thresh_list_query':thresh_list_query,
									'thresh_motif_score_neg_1':thresh_motif_score_neg_1,
									'thresh_motif_score_neg_2':thresh_motif_score_neg_2,
									'thresh_score_accessibility':thresh_score_accessibility}

				select_config.update({'config_link_type':config_link_type})

		if flag_config_2>0:
			if not ('thresh_score_query_1' in select_config):
				thresh_vec_1 = [[0.10,0.05],[0.05,0.10]]
				select_config.update({'thresh_score_query_1':thresh_vec_1})
			
			thresh_gene_tf_corr_peak = 0.30
			thresh_gene_tf_corr_ = 0.05
			if not ('thresh_gene_tf_corr_compare' in select_config):
				thresh_gene_tf_corr_compare = [thresh_gene_tf_corr_peak,thresh_gene_tf_corr_]
				select_config.update({'thresh_gene_tf_corr_compare':thresh_gene_tf_corr_compare})

			print(select_config['thresh_score_query_1'])
			print(select_config['thresh_gene_tf_corr_compare'])
			
			column_label_1 = 'feature1_score1_quantile'
			column_label_2 = 'feature1_score2_quantile'
			column_label_3 = 'feature2_score1_quantile'
			select_config.update({'column_quantile_1':column_label_1,'column_quantile_2':column_label_2,
									'column_quantile_feature2':column_label_3})

			if not ('thresh_vec_score_2' in select_config):
				thresh_corr_1 = 0.30
				thresh_corr_2 = 0.50
				# thresh_pval_1, thresh_pval_2 = 0.25, 1
				thresh_pval_1, thresh_pval_2, thresh_pval_3 = 0.05, 0.10, 1
				# thresh_vec_1 = [[thresh_corr_1,thresh_pval_2],[thresh_corr_2,thresh_pval_2]]	# original
				thresh_vec_1 = [[thresh_corr_1,thresh_pval_2],[thresh_corr_2,thresh_pval_3]]	# updated
				# thresh_vec_2 = thresh_vec_1.copy()
				# thresh_vec_2 = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2]]	# original
				thresh_vec_2 = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_3]]	# updated
				# thresh_vec_3 = [[thresh_corr_1,thresh_pval_2],[thresh_corr_2,thresh_pval_2]]	# original
				thresh_vec_3 = [[thresh_corr_1,thresh_pval_2],[thresh_corr_2,thresh_pval_3]]	# updated
				thresh_vec = [thresh_vec_1,thresh_vec_2,thresh_vec_3]
				select_config.update({'thresh_vec_score_2':thresh_vec})

			thresh_score_quantile = 0.95
			select_config.update({'thresh_score_quantile':thresh_score_quantile})

		self.select_config = select_config

		return select_config

	## compute feature score
	def test_query_feature_score_compute_1(self,df_feature_link=[],input_filename='',overwrite=False,iter_mode=1,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		field_query = ['thresh_list_query','thresh_motif_score_neg_1','thresh_motif_score_neg_2','thresh_score_accessibility']

		if ('config_link_type' in select_config) and (overwrite==False):
			config_link_type = select_config['config_link_type']
			list1 = [config_link_type[field_id] for field_id in field_query]
			thresh_list_query,thresh_motif_score_neg_1,thresh_motif_score_neg_2,thresh_score_accessibility = list1
		else:
			# thresh_corr_1, thresh_pval_1 = 0.05, 0.1 # the previous parameter
			thresh_corr_1, thresh_pval_1 = 0.1, 0.1  # the alternative parameter to use; use relatively high threshold for negative peaks
			thresh_corr_2, thresh_pval_2 = 0.1, 0.05
			thresh_corr_3, thresh_pval_3 = 0.15, 1
			thresh_motif_score_neg_1, thresh_motif_score_neg_2 = 0.80, 0.90 # higher threshold for regression mechanism
			# thresh_score_accessibility = 0.1
			thresh_score_accessibility = 0.25
			thresh_list_query = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2],[thresh_corr_3, thresh_pval_3]]

			list1 = [thresh_list_query,thresh_motif_score_neg_1,thresh_motif_score_neg_2,thresh_score_accessibility]
			
			config_link_type = dict(zip(field_query,list1))
			select_config.update({'config_link_type':config_link_type})

		column_idvec = ['motif_id','peak_id','gene_id']
		if 'column_idvec' in select_config:
			column_idvec = select_config['column_idvec']

		column_pval_cond = select_config['column_pval_cond']
		flag_query_1=1
		if flag_query_1>0:
			list_query1 = []
			flag_link_type = 1
			flag_compute = 1
			from utility_1 import test_query_index, test_column_query_1
			if len(df_feature_link)==0:
				if iter_mode>0:
					# input_filename_1 = 'test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.4000_4500.annot1_1.1.txt'
					# input_filename_2 = 'test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.4000_4500.txt'
					input_filename_list1 = select_config['filename_list_score']
					input_filename_list2 = select_config['filename_annot_list']
					input_filename_list3 = select_config['filename_link_list']
					input_filename_list_motif = select_config['filename_motif_score_list']

					query_num1 = len(input_filename_list1)
					for i1 in range(query_num1):
						input_filename_1 = input_filename_list1[i1]
						df_1 = pd.read_csv(input_filename_1,index_col=False,sep='\t')

						filename_annot1, filename_annot2 = input_filename_list2[i1], input_filename_list_motif[i1]
						filename_link = input_filename_list3[i1]

						df_link_query_1 = df_1
						df_feature_link = df_link_query_1
						print('df_link_query_1: ',df_link_query_1.shape)
						print(input_filename_1)
						print(df_link_query_1[0:2])

						select_config.update({'filename_annot1':filename_annot1,'filename_motif_score':filename_annot2,
												'filename_link':filename_link})				
						retrieve_mode = 0
						flag_annot_1=1
						df_link_query_pre1 = self.test_gene_peak_tf_query_score_compute_unit_1(df_feature_link=df_link_query_1,flag_link_type=flag_link_type,flag_compute=flag_compute,flag_annot_1=flag_annot_1,
																								retrieve_mode=retrieve_mode,save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=verbose,select_config=select_config)

						list_query1.append(df_link_query_pre1)

						if save_mode>0:
							b = input_filename_1.find('.txt')
							output_filename = input_filename_1[0:b]+'.recompute.txt'
							column_score_query_pre1 = select_config['column_score_query_pre1']

							# retrieve the columns of score estimation and subset of annotations
							column_idvec = select_config['column_idvec']
							# column_vec_1 = list(column_idvec)+list(column_score_query2)
							column_vec_1 = pd.Index(column_score_query_pre1).union(column_idvec,sort=False)

							df_link_query_pre2 = df_link_query_pre1.loc[:,column_vec_1]
							float_format = '%.5f'
							df_link_query_pre2.to_csv(output_filename,index=False,sep='\t',float_format=float_format)

							if ('field_link_1' in select_config):
								field_link_1 = select_config['field_link_1']
							else:
								field_link_1 = ['peak_gene_link','gene_tf_link','peak_tf_link']

							if ('field_link_2' in select_config):
								field_link_2 = select_config['field_link_2']
							else:
								field_link_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']
							
							# retrieve the columns of link type annotation
							column_vec_2 = list(column_idvec) + field_link_1 + field_link_2 + ['group']
							df_link_query_pre3 = df_link_query_pre1.loc[:,column_vec_2]

							b = filename_link.find('.txt')
							output_filename_2 = filename_link[0:b]+'.recompute.txt'
							df_link_query_pre3.to_csv(output_filename_2,index=False,sep='\t')

					df_feature_link = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
				
				else:
					df_feature_link = pd.read_csv(input_filename,index_col=0,sep='\t')
			
			if (iter_mode==0) and (len(df_feature_link)>0):
				df_feature_link = self.test_gene_peak_tf_query_score_compute_unit_1(df_feature_link=df_feature_link,flag_link_type=flag_link_type,flag_compute=flag_compute,select_config=select_config)

			return df_feature_link

	## feature score quantile estimation
	def test_query_feature_score_quantile_1(self,df_feature_link=[],input_filename_list=[],index_col=0,column_idvec=['peak_id','gene_id','motif_id'],column_vec_query=[],column_score_vec=[],column_label_vec=[],column_id_query='motif_id',iter_mode=0,
												save_mode=0,filename_prefix_save='',output_file_path='',output_filename_1='',output_filename_2='',compression='gzip',float_format='%.5E',flag_unduplicate=0,verbose=0,select_config={}):

		flag_query_1 = 1
		if flag_query_1>0:
			if len(df_feature_link)==0:
				if len(input_filename_list)>0:
					# column_vec_query1 = ['score_pred1','score_pred2','score_pred_combine']
					df_link_query = utility_1.test_file_merge_1(input_filename_list,column_vec_query=column_vec_query,index_col=index_col,header=0,float_format=-1,flag_unduplicate=flag_unduplicate,
																save_mode=0,verbose=verbose,output_filename=output_filename_1)

					
					if (save_mode>0) and (output_filename_1!=''):
						df_link_query.to_csv(output_filename_1,sep='\t')
				else:
					print('please provide feature association query')
			else:
				df_link_query = df_feature_link

			if not (column_id_query in df_link_query.columns):
				df_link_query[column_id_query] = np.asarray(df_link_query.index)
			
			print('df_link_query: ',df_link_query.shape)
			print(df_link_query.columns)
			print(df_link_query[0:2])

			column_idvec = ['motif_id','peak_id','gene_id']
			df_link_query.index = utility_1.test_query_index(df_link_query,column_vec=column_idvec)
			df_link_query = df_link_query.loc[(~df_link_query.index.duplicated(keep='first')),:]
			print('df_link_query: ',df_link_query.shape)
			print(df_link_query.columns)
			print(df_link_query[0:2])

			df_link_query_pre1 = self.test_score_query_2(data=df_link_query,feature_query_vec=[],column_id_query=column_id_query,column_idvec=column_idvec,
															column_query_vec=column_score_vec,column_label_vec=column_label_vec,
															flag_annot=1,reset_index=0,parallel_mode=0,interval=100,verbose=verbose,select_config=select_config)

			if (save_mode>0) and (output_filename_2!=''):
				# if (compression!=-1):
				if (not (compression is None)):
					df_link_query_pre1.to_csv(output_filename_2,index=False,sep='\t',float_format=float_format,compression=compression)
				else:
					df_link_query_pre1.to_csv(output_filename_2,index=False,sep='\t',float_format=float_format)
				
			return df_link_query_pre1

	## ====================================================
	# peak-tf-gene link query selection
	def test_gene_peak_tf_query_select_1(self,df_gene_peak_query=[],df_annot_query=[],df_score_annot=[],lambda1=0.5,lambda2=0.5,type_id_1=0,column_id1=-1,
											flag_compare_thresh1=1,flag_select_pair_1=1,flag_select_feature_1=1,flag_select_feature_2=1,
											flag_select_local=1,flag_select_link_type=1,iter_mode=0,
											input_file_path='',save_mode=1,filename_prefix_save='',output_file_path='',verbose=1,select_config={}):

			## pre-selection of peak-tf-gene link query by not strict thresholds to filter link query with relatively low estimated scores
			flag_select_thresh1_1=0
			# flag_select_thresh1=0
			thresh_score_1 = 0.10
			thresh_corr_1, thresh_corr_2 = 0.10, 0.30
			# thresh_corr_2 = 0.15
			pval_thresh_vec = [0.05,0.10,0.15,0.25,0.50]
			pval_thresh_1 = pval_thresh_vec[3]
			thresh_score_1, thresh_corr_1, thresh_corr_2, thresh_pval_1, thresh_pval_2 = 0.10, 0.10, 0.30, 0.25, 0.5
			filename_annot_thresh = '%s_%s_%s_%s'%(thresh_score_1,thresh_corr_1,pval_thresh_1,thresh_corr_2)
			
			# file_save_path = select_config['data_path_save']
			# input_file_path2 = file_save_path
			# output_file_path = input_file_path2
			input_file_path2 = input_file_path
			output_file_path = input_file_path2
			df_gene_peak_query_1 = df_gene_peak_query
			if 'column_idvec' in select_config:
				column_idvec = select_config['column_idvec']
			else:
				column_idvec = ['motif_id','peak_id','gene_id']
				select_config.update({'column_idvec':column_idvec})
			
			print('column_idvec ',column_idvec)
			from utility_1 import test_query_index
			df_gene_peak_query_1.index = test_query_index(df_gene_peak_query_1,column_vec=column_idvec)
			df_gene_peak_query_1 = df_gene_peak_query_1.fillna(0)
			query_id_ori = df_gene_peak_query_1.index.copy()
			
			## gene-peak-tf association: the difference between gene_tf_corr_peak and gene-tf expression correlation
			# flag_compare_thresh1=1
			flag_annot_1 = 1
			df_gene_peak_query_pre1 = []

			# query estimated correlation and partial correlation annotations
			column_query_1 = [['peak_tf_corr','peak_tf_pval_corrected'],['peak_gene_corr_','peak_gene_corr_pval'],
								['gene_tf_corr_peak','gene_tf_corr_peak_pval_corrected1']]			

			field_query1 = ['peak_tf','peak_gene']
			field_query2 = ['corr','pval']
			list1 = []
			for field_id1 in field_query1:
				list1.append(['column_%s_%s'%(field_id1,field_id2) for field_id2 in field_query2])
			list1.append(['column_query_cond','column_pval_cond'])
			field_query_1 = list1

			field_num1 = len(field_query_1)
			for i1 in range(field_num1):
				# field1, field2 = field_query_1[i1], field_query_2[i1]
				for i2 in range(2):
					field1 = field_query_1[i1][i2]
					if (field1 in select_config):
						column_1 = select_config[field1]
						print(field1,column_1,i1,i2)
						column_query_1[i1][i2] = column_1

			column_query_pre1 = column_query_1.copy()

			column_num1 = len(column_query_1)
			columns_1 = [query_vec[0] for query_vec in column_query_1] # the columns for correlation
			# column_gene_tf_corr, column_gene_tf_pval = 'gene_tf_corr','gene_tf_pval_corrected'
			field_query = ['column_gene_tf_corr','column_gene_tf_pval']
			columns_1 = columns_1 + [select_config[field_id] for field_id in field_query]

			flag_select_thresh1_feature_local=flag_select_local
			if flag_select_thresh1_feature_local>0:
				columns_pval = [query_vec[1] for query_vec in column_query_1]
				columns_1 = columns_1 + columns_pval

			column_annot_query = pd.Index(columns_1).difference(df_gene_peak_query_1.columns,sort=False)
			if len(column_annot_query)>0:
				print('load annotaions from file')
				print(column_annot_query)
						
				if len(df_annot_query)==0:
					if 'filename_annot_1' in select_config:
						filename_annot_1 = select_config['filename_annot_1']
						df_annot_query = pd.read_csv(filename_annot_1,index_col=False,sep='\t')
					else:
						print('please provide the estimated correlation and p-value')

				df_list1 = [df_gene_peak_query_1,df_annot_query]
				column_idvec_1 = column_idvec
				column_vec_1 = [column_annot_query]
				df_gene_peak_query_1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec_1,column_vec=column_vec_1,
																		df_list=df_list1,type_id_1=0,type_id_2=0,reset_index=False,select_config=select_config)

			if flag_compare_thresh1>0:
				type_id_1=2
				thresh_gene_tf_corr_peak = 0.30
				thresh_gene_tf_corr_ = 0.05
				if 'thresh_gene_tf_corr_compare' in select_config:
					thresh_gene_tf_corr_compare = select_config['thresh_gene_tf_corr_compare']
					thresh_gene_tf_corr_peak, thresh_gene_tf_corr_ = thresh_gene_tf_corr_compare[0:2]

				output_file_path = input_file_path2
				input_filename_1 = ''
				# query_id_1: the link to keep; query_id_2: the link with difference between gene_tf_corr_peak and gene_tf_corr_ above threshold
				df_gene_peak_query_pre1, query_id_1, query_id_2 = self.test_gene_peak_tf_query_compare_1(input_filename=input_filename_1,
																											df_gene_peak_query=df_gene_peak_query_1,
																											thresh_corr_1=thresh_gene_tf_corr_peak,
																											thresh_corr_2=thresh_gene_tf_corr_,
																											save_mode=1,output_file_path=output_file_path,
																											filename_prefix_save=filename_prefix_save,
																											select_config=select_config)

				df_query_1 = df_gene_peak_query_pre1.loc[query_id_1]
				df_query_2 = df_gene_peak_query_pre1.loc[query_id_2]

				print('df_query_1, df_query_2 ',df_query_1.shape,df_query_2.shape)
				query_compare_group1, query_compare_group2 = query_id_1, query_id_2

				if flag_annot_1>0:
					column_label_pre1 = 'label_gene_tf_corr_peak_compare'
					df_gene_peak_query_pre1.loc[query_id_2,column_label_pre1] = 1

			# select by pre-defined threshold
			flag_select_thresh1_pair_1=flag_select_pair_1
			df_link_query_1 = df_gene_peak_query_pre1

			column_query1, column_query2 = select_config['column_score_1'], select_config['column_score_2']
			column_query_vec = [column_query1,column_query2]

			# column_idvec = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_idvec[0:3]
			if flag_select_thresh1_pair_1>0:
				thresh_vec_1 = [[0.10,0.05],[0.05,0.10]]
				if 'thresh_score_query_1' in select_config:
					thresh_vec_1 = select_config['thresh_score_query_1']
				
				thresh_num1 = len(thresh_vec_1)
				score_query1, score_query2 = df_link_query_1[column_query1], df_link_query_1[column_query2]
				list1 = []
				for i1 in range(thresh_num1):
					thresh_vec_query = thresh_vec_1[i1]
					thresh_score_1, thresh_score_2 = thresh_vec_query[0:2]
					id1 = (score_query1>thresh_score_1)
					id2 = (score_query2>thresh_score_2)
					id_1 = (id1&id2)
					query_id1 = query_id_ori[id_1]
					query_num1 = len(query_id1)
					list1.append(query_id1)
					if verbose>0:
						print('thresh_1, thresh_2: ',thresh_score_1,thresh_score_2,query_num1)

				query_id_1, query_id_2 = list1[0:2]
				column_label_1, column_label_2 = 'label_score_1', 'label_score_2'
				df_link_query_1.loc[query_id_1,column_label_1] = 1
				df_link_query_1.loc[query_id_1,column_label_2] = 1

			return df_link_query_1

	## ====================================================
	# query tf-(peak,gene) link type: 1: positive regulation; -1: negative regulation
	# add the columns: 'peak_gene_link','gene_tf_link','peak_tf_link'
	# use the link type for selection
	def test_link_type_query_1(self,df_gene_peak_query=[],df_link_type=[],input_filename='',field_query=[],link_query=0,type_id_1=0,type_id_2=0,reset_index=True,verbose=0,select_config={}):

		if len(df_link_type)==0:
			if (input_filename!=''):
				if os.path.exists(input_filename)==True:
					# df_link_type = pd.read_csv(input_filename,index_col=0,sep='\t')
					df_link_type = pd.read_csv(input_filename,index_col=False,sep='\t')
				else:
					print('the file does not exist: %s'%(input_filename))
					return
			else:
				print('please provide link type annotation')
				return
		print('df_link_type ',df_link_type.shape)

		column_vec_1 = ['motif_id','peak_id','gene_id']
		if reset_index==True:
			query_id_ori = df_gene_peak_query.index.copy()
			df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_vec_1)
		df_link_type.index = test_query_index(df_link_type,column_vec=column_vec_1)
		query_id_1 = df_gene_peak_query.index
		query_id_2 = df_link_type.index
		query_id_pre1 = query_id_1.intersection(query_id_2,sort=False)
		if len(field_query)==0:
			field_query = ['peak_gene_link','gene_tf_link','peak_tf_link']
		df_query = df_gene_peak_query
		df_query.loc[query_id_pre1,field_query] = df_link_type.loc[query_id_pre1,field_query]
		if link_query>0:
			# filter peak-tf-gene link query based on link type
			# filter link query uncertain to explain
			peak_gene_link, gene_tf_link, peak_tf_link = df_query['peak_gene_link'],df_query['gene_tf_link'],df_query['peak_tf_link']
			id1 = ((peak_gene_link*gene_tf_link*peak_tf_link)>=0)	# lower threshold
			id2 = ((peak_gene_link*gene_tf_link*peak_tf_link)>0)	# higher threshold
			query_id1 = query_id_1[id1]
			query_id2 = query_id_1[id2]
			df_query.loc[id1,'link_query'] = 1
			df_query.loc[id2,'link_query'] = 2
			id_2 = (~id1)
			df_query.loc[id_2,'link_query'] = -1
			df_query1 = df_query.loc[id1,:]
			df_query2 = df_query.loc[id2,:]
			print('df_gene_peak_query, df_query1, df_query2 ',df_gene_peak_query.shape,df_query1.shape,df_query2.shape)

		if reset_index==True:
			df_query.index = query_id_ori

		return df_query, df_query1

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)


		
