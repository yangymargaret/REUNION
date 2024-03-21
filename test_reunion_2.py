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
# from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors

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
import pingouin as pg
import shap
import networkx as nx

from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
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

import test_reunion_1
from test_reunion_1 import _Base2_correlation2
import itertools

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

class _Base2_correlation3(_Base2_correlation2):
	"""Base class for peak-TF-gene link estimation
	"""
	def __init__(self,file_path,run_id,species_id=1,cell='ES', 
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

		_Base2_correlation2.__init__(self,file_path=file_path,
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
	def test_config_query_pre1(self,beta_mode=0,save_mode=1,overwrite=False,select_config={}):

		print('test_config_query')
		flag_query1=1
		if flag_query1>0:
			thresh1_ratio, thresh1, distance_tol_1 = 1.0, 0, -250 # smaller distance and similar or higher correlation
			thresh2_ratio, thresh2, distance_tol_2 = 1.5, 0.15, 50 # similar distance and higher correlation
			thresh3_ratio, thresh3, distance_tol_3 = 0.9, -0.02, -250
			# thresh5_ratio, thresh5, distance_tol_5 = 0.75, -0.1, -500
			thresh5_ratio, thresh5, distance_tol_5 = 0.8, -0.1, -500
			thresh6_ratio, thresh6, distance_tol_6 = 0.9, -0.02, -100
			# thresh6_ratio, thresh6, distance_tol_6 = 0.9, -0.02, -50
			thresh7_ratio, thresh7, distance_tol_7 = 1.5, 0.1, 25
			thresh_vec_1 = [thresh1_ratio, thresh1, distance_tol_1]
			thresh_vec_2 = [thresh2_ratio, thresh2, distance_tol_2]
			thresh_vec_3 = [thresh3_ratio, thresh3, distance_tol_3]
			thresh_vec_5 = [thresh5_ratio, thresh5, distance_tol_5]
			thresh_vec_6 = [thresh6_ratio, thresh6, distance_tol_6]
			# thresh_vec_7 = [thresh6_ratio, thresh7, distance_tol_7]
			thresh_vec_7 = [thresh7_ratio, thresh7, distance_tol_7]
			
			thresh_vec_compare = [thresh_vec_6,thresh_vec_5,thresh_vec_7]
		
			peak_distance_thresh_compare = 50
			# select_config.update({'peak_distance_thresh_compare':peak_distance_thresh_compare})

			parallel_mode = 0
			interval_peak_query = 100
			# select_config.update({'interval_peak_query':interval_peak_query})

			# used in function: test_peak_score_distance_1()
			decay_value_vec = [1,0.9,0.75,0.6]
			distance_thresh_vec = [50,500,1000,2000]

			# used in function: test_peak_score_distance_1()
			decay_value_vec = [1,0.9,0.75,0.6]
			distance_thresh_vec = [50,500,1000,2000]
			list1 = [thresh_vec_compare,peak_distance_thresh_compare,parallel_mode,interval_peak_query,decay_value_vec,distance_thresh_vec]
			# select_config.update({'decay_value_vec':decay_value_vec,'distance_thresh_vec':distance_thresh_vec})
			field_query = ['thresh_vec_compare','peak_distance_thresh_compare','parallel_mode','interval_peak_query',
							'decay_value_vec','distance_thresh_vec']
			field_num1 = len(field_query)
			for i1 in range(field_num1):
				field_id = field_query[i1]
				if not (field_id in select_config):
					select_config.update({field_id:list1[i1]})

			save_file_path = select_config['data_path_save_local']
			input_file_path = save_file_path
			filename_prefix_1 = select_config['filename_prefix_default']
			filename_prefix_2 = select_config['filename_prefix_save_default']
			input_filename_peak_query = '%s/%s.%s.peak_basic.txt'%(input_file_path,filename_prefix_1,filename_prefix_2)
			
			select_config.update({'input_filename_peak_query':input_filename_peak_query})
			file_path_basic_filter = '%s/temp2'%(save_file_path)
			if os.path.exists(file_path_basic_filter)==False:
				print('the directory does not exist:%s'%(file_path_basic_filter))
				os.mkdir(file_path_basic_filter)
			select_config.update({'file_path_basic_filter':file_path_basic_filter})
			
			self.select_config = select_config

			return select_config

	## peak-gene correlation in different distance range
	# query GC_bin and distance_bin
	def test_query_link_correlation_distance_1(self,df_link_query=[],df_feature_annot=[],column_id_query='',n_bins_vec=[50,100],distance_bin=25,flag_unduplicate=1,save_mode=0,filename_prefix_save='',output_file_path='',select_config={}):
		
		if column_id_query=='':
			column_id_query = 'peak_gene_corr_'
		
		column_idvec = ['peak_id','gene_id']
		column_id2, column_id1 = column_idvec

		# the peak-gene links
		if flag_unduplicate>0:
			df_link_query = df_link_query.drop_duplicates(subset=column_idvec,keep='first')
		
		column_corr_1 = column_id_query
		peak_gene_corr_ = df_link_query[column_corr_1]
		df_link_query.index = test_query_index(df_link_query,column_vec=column_idvec)
		df_link_query_ori = df_link_query
		# df_link_query = df_link_query.loc[pd.isna(df_link_query[column_id_query])==False,:]
		df_link_query = df_link_query.dropna(subset=[column_id_query])
		print('df_link_query_ori, df_link_query: ',df_link_query_ori.shape,df_link_query.shape)

		
		df_feature_annot_1 = df_feature_annot
		column_1, column_2 = 'GC_bin', 'distance_bin'
		n_bins_GC, n_bins_distance = n_bins_vec[0:2]
		column_1_ori = 'GC'
		interval_1 = 1.0/n_bins_GC
		column_1_query = '%s_%d'%(column_1,n_bins_GC)
		type_id2 = 0
		if type_id2==0:
			query_value = df_feature_annot_1[column_1_ori]
			df_feature_annot_1[column_1_query] = np.digitize(query_value, np.linspace(0, 1, n_bins_GC+1))
		else:
			query_value = df_feature_annot_1[column_1_ori]
			query_value_2 = np.int32(np.ceil(query_value/interval_1))
			id1 = (query_value_2==0)
			query_value_2[id1] = 1
			df_feature_annot_1[column_1_query] = query_value_2

		output_filename_1 = '%s/%s.GC.annot1.txt'%(output_file_path,filename_prefix_save)
		df_feature_annot_1.to_csv(output_filename_1,sep='\t')

		distance = df_link_query['distance']
		distance_abs = distance.abs()

		# max_distance, min_distance = distance_abs.max(), distance_abs.min()
		max_distance, min_distance = distance_abs.max(), 0
		if distance_bin>0:
			n_bins_distance = int(np.ceil((max_distance-min_distance)/distance_bin))
		else:
			distance_bin = (max_distance-min_distance)/n_bins_distance

		print('n_bins_distance, distance_bin: ',n_bins_distance,distance_bin)
		# df_link_query[column_2] = np.digitize(distance_abs, np.linspace(min_distance, max_distance, n_bins_distance))
		distance_bin_vec = np.unique(np.asarray([distance_bin]+[50]))
		peak_distance_thresh_1 = 2000
		for distance_bin_value in distance_bin_vec:
			column_2_query = '%s_%d'%(column_2,distance_bin_value)
			if type_id2==0:
				t_vec_1 = np.arange(0,peak_distance_thresh_1+distance_bin_value,distance_bin_value)
				df_link_query[column_2_query] = np.digitize(distance_abs, t_vec_1)
			else:
				df_link_query[column_2_query] = np.int32(np.ceil(distance_abs/distance_bin_value))
				id1 = (df_link_query[column_2_query]==0)
				df_link_query.loc[id1,column_2_query] = 1
		
		df_link_query.index = np.asarray(df_link_query[column_id2])
		query_id1 = df_link_query.index
		df_link_query.loc[:,column_1_query] = df_feature_annot_1.loc[query_id1,column_1_query] # the GC group
		df_link_query.loc[:,column_1_ori] = df_feature_annot_1.loc[query_id1,column_1_ori]

		normalize_type = 'uniform'
		column_vec_query = ['distance_bin','spearmanr','GC_bin_%d'%(n_bins_GC)]
		column_annot = ['distance_pval1','distance_pval2']
		query_num1 = len(distance_bin_vec)
		df_link_query.index = test_query_index(df_link_query,column_vec=column_idvec)
		for i1 in range(query_num1): 
			distance_bin_value = distance_bin_vec[i1]
			column_1, column_2, column_3 = column_vec_query[0:3]
			column_vec_query1 = ['%s_%d'%(column_1,distance_bin_value),column_2,column_3]
			column_annot_query1 = ['%s_%d'%(column_query,distance_bin_value) for column_query in column_annot]
			df_link_query1, df_annot1, dict_annot1 = self.test_attribute_query_distance_1(df_link_query,column_vec_query=column_vec_query1,column_annot=column_annot_query1,normalize_type=normalize_type,
																							verbose=1,select_config=select_config)

			filename_annot1 = distance_bin_value
			output_filename = '%s/%s.distance.%s.annot1.txt'%(output_file_path,filename_prefix_save,filename_annot1)
			# distance_bin = 25
			id1 = np.asarray(df_annot1.index)
			df_annot1['distance_1'], df_annot1['distance_2'] = distance_bin*(id1-1), distance_bin*id1
			df_annot1.to_csv(output_filename,sep='\t')

			output_filename = '%s/%s.distance.%s.annot1.npy'%(output_file_path,filename_prefix_save,filename_annot1)
			np.save(output_filename,dict_annot1,allow_pickle=True)

			if i1==0:
				df_link_query = df_link_query1
				query_id1 = df_link_query.index
			else:
				df_link_query.loc[query_id1,column_annot_query1] = df_link_query1.loc[query_id1,column_annot_query1]

		filename_annot1 = distance_bin
		output_filename = '%s/%s.distance.%s.annot2.txt'%(output_file_path,filename_prefix_save,filename_annot1)
		df_link_query.index = np.asarray(df_link_query[column_id1])
		df_link_query.to_csv(output_filename,sep='\t',float_format='%.5E')
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
		output_filename = '%s/%s.distance.%s.annot2.1.txt'%(output_file_path,filename_prefix_save,filename_annot1)
		df_link_query_sort = df_link_query.sort_values(by=[column_id2,'distance'],ascending=[True,True])
		df_link_query_sort.to_csv(output_filename,sep='\t',float_format='%.5E')
		print('df_feature_annot_1, df_annot1, df_link_query: ',df_feature_annot_1.shape,df_annot1.shape,df_link_query.shape)
		
		return df_link_query, df_annot1, dict_annot1


	## estimate the empirical distribution of the difference of peak-gene correlations in different paired bins
	def test_attribute_query_distance_2(self,df_link_query,df_annot=[],column_vec_query=[],column_annot=[],n_sample=500,distance_bin=25,distance_tol=2,normalize_type='uniform',type_id_1=0,save_mode=1,filename_prefix_save='',output_file_path='',select_config={}):

		# column_1, column_2 = 'GC_bin', 'distance_bin'
		column_1, column_2 = 'GC_bin_20', 'distance_bin_50'
		column_corr_1 = 'peak_gene_corr_'
		if column_vec_query==[]:
			# column_vec_query = [column_2,column_corr_1]
			column_vec_query = [column_2,column_corr_1,column_1]

		if len(column_annot)==0:
			# column_annot = ['distance_pval1','distance_pval2']
			column_annot = ['distance_pval1_50','distance_pval2_50']

		column_id1, column_id2 = column_vec_query[0:2]
		column_query1, column_query2 = column_annot[0:2]

		group_query = df_link_query[column_id1]	# distance bin group
		query_vec = np.sort(np.unique(group_query))
		group_id_min, group_id_max = query_vec[0], query_vec[-1]

		query_num = len(query_vec)
		# n_sample = 1000
		distance_tol = query_num
		# import itertools
		column_vec_1 = ['max_corr','min_corr','median_corr','mean_corr','std_corr']
		column_vec_2 = ['mean_corr_difference','std_corr_difference']
		column_vec = column_vec_1+column_vec_2
		if len(df_annot)==0:
			# df1 = pd.DataFrame(index=query_vec,columns=column_vec_1,dtype=np.float32)
			df1 = pd.DataFrame(index=query_vec, dtype=np.float32)
		else:
			df1 = df_annot

		dict_query1 = dict()
		query_value_ori = df_link_query[column_id2]
		# prepare sample for each group
		flag_query1=0
		if not ('mean_corr' in df1.columns):
			flag_query1=1
		for i1 in range(query_num):
			group_id1 = query_vec[i1]
			query_value1 = df_link_query.loc[(group_query==group_id1),column_id2]
			query_id1 = query_value1.index
			query_num1 = len(query_id1)
			if query_num1==0:
				print('group_id1:%d,query_num1:%d'%(group_id1,query_num1))
				continue

			if flag_query1>0:
				t_vec_1 = [np.max(query_value1),np.min(query_value1),np.median(query_value1)]
				m1_ori,v1_ori = np.mean(query_value1), np.std(query_value1)
				t_vec_2 = t_vec_1+[m1_ori,v1_ori]
				df1.loc[group_id1,column_vec_1] = t_vec_2

			sample_id1 = query_id1
			dict_query1.update({group_id1:sample_id1})

		flag_sort = 1
		verbose = 1
		# estimate the empirical distribution of the difference between values from different groups
		df_annot1, df_annot2, dict_query1, dict_query2 = self.test_query_group_feature_compare_1(df_query=df_link_query,df_annot=df1,dict_query=dict_query1,
																								column_vec=column_vec_query,distance_tol=distance_tol,flag_sort=flag_sort,type_id_1=type_id_1,
																								verbose=verbose,
																								select_config=select_config)

		# output_file_path = select_config['data_path_save_local']
		filename_annot = str(distance_bin)
		list1 = [df_annot1, df_annot2]
		list2 = [dict_query1, dict_query2]
		query_num1 = len(list1)
		for i1 in range(query_num1):
			# output_filename_1 = '%s/test_group_feature_compare.%s.annot%d.txt'%(output_file_path,filename_annot,i1+1)
			output_filename_1 = '%s/%s.group_feature_compare.%s.annot%d.%d.txt'%(output_file_path,filename_prefix_save,filename_annot,i1+1,type_id_1)
			df_query = list1[i1]
			df_query.to_csv(output_filename_1,sep='\t',float_format='%.5f')

			# output_filename_2 = '%s/test_group_feature_compare.%s.annot%d.npy'%(output_file_path,filename_annot,i1+1)
			output_filename_2 = '%s/%s.group_feature_compare.%s.annot%d.%d.npy'%(output_file_path,filename_prefix_save,filename_annot,i1+1,type_id_1)
			dict_query = list2[i1]
			np.save(output_filename_2,dict_query,allow_pickle=True)
		
		return df_annot, df_annot2, dict_query1, dict_query2

	## pre-selection of gene-peak query
	def test_gene_peak_query_basic_filter_1_pre2_basic_1(self,peak_id=[],df_peak_query=[],df_gene_peak_query=[],df_gene_peak_distance_annot=[],field_query=['distance','correlation'],column_vec_query=['distance','spearmanr'],column_score='spearmanr',
															distance_bin_value=50,score_bin_value=0.05,peak_distance_thresh=2000,thresh_vec_compare=[],column_label='label_thresh2',thresh_type=3,flag_basic_query=3,flag_unduplicate=1,
															type_query_compare=2,type_score=0,type_id_1=0,print_mode=0,save_mode=0,filename_prefix_save='',output_file_path='',verbose=0,select_config={}):

		column_idvec = ['peak_id','gene_id']
		column_id2, column_id1 = column_idvec[0:2]
		filename_prefix_save_pre1 = select_config['filename_prefix_default_1']
		filename_prefix_save_1 = filename_prefix_save
		column_distance, column_score_1 = column_vec_query[0:2]
		field1, field2 = field_query[0:2]
		column_vec_2 = ['%s_abs'%(column_distance),'%s_abs'%(column_score_1)]
		column_value_1_ori = column_distance
		column_value_2_ori = column_score_1

		column_value_1 = '%s_abs'%(column_distance)
		if type_score==0:
			column_value_2 = '%s_abs'%(column_score_1)
		else:
			column_value_2 = column_score_1
		
		df_link_query_1 = df_gene_peak_query 
		if flag_basic_query in [1,3]:
			# data_file_type_query = select_config['data_file_type_query']
			if type_score==0:
				input_filename_1 = select_config['input_filename_pre2']
				df_gene_peak_query_thresh1 = pd.read_csv(input_filename_1,index_col=False,sep='\t')
				# df_gene_peak_query_thresh1.index = np.asarray(df_gene_peak_query_thresh1[column_id1])

				input_filename = select_config['filename_save_thresh2']
				df_gene_peak_query_thresh2 = pd.read_csv(input_filename,index_col=False,sep='\t')
				print('df_gene_peak_query_thresh1, df_gene_peak_query_thresh2: ',df_gene_peak_query_thresh1.shape,df_gene_peak_query_thresh2.shape)
		
				df_gene_peak_distance_1 = df_gene_peak_distance_annot
				if len(df_gene_peak_distance_annot)>0:
					df_gene_peak_distance_1.index = np.asarray(df_gene_peak_distance_1[column_id2])
					
				list_1 = [df_gene_peak_distance_1,df_gene_peak_query_thresh1,df_gene_peak_query_thresh2]
				filename_annot_vec = ['df_gene_peak_distance','df_gene_peak_thresh1','df_gene_peak_thresh2']
				
				query_num1 = len(list_1)
				for i2 in range(query_num1):
					df_query = list_1[i2]
					if len(df_query)>0:
						query_id_1 = test_query_index(df_query,column_vec=column_idvec)
						id1 = pd.Index(query_id_1).duplicated(keep='first')
						t_value_1 = np.sum(id1)
						filename_annot_str1 = filename_annot_vec[i2]
						print('filename_annot, df_query, duplicated: ',filename_annot_str1,df_query.shape,t_value_1)
				
			# peak_query = df_gene_peak_query_thresh2[column_id2].unique()
			peak_query = df_gene_peak_query[column_id2].unique()
			peak_num1 = len(peak_query)
			print('peak_query: %d'%(peak_num1))
			print('df_gene_peak_query: ',df_gene_peak_query.shape)
			column_query = column_label

			type_query_1 = type_query_compare
			if type_query_1==0:
				if len(df_gene_peak_distance_annot)>0:
					df_query1 = df_gene_peak_distance_1.loc[peak_query,:]

			elif type_query_1==1:
				input_filename_1 = select_config['input_filename_pre2']
				df_gene_peak_query_thresh1 = pd.read_csv(input_filename_1,index_col=False,sep='\t')
				df_query1 = df_gene_peak_query_thresh1

			elif type_query_1==2:
				df_query1= df_gene_peak_query
				if not (column_label in df_query1.columns):
					df_query1[column_label] = 1

			if flag_unduplicate>0:
				df_query1 = df_query1.drop_duplicates(subset=column_idvec)
			df_query1.index = utility_1.test_query_index(df_query1,column_vec=column_idvec)

			df_link_query = df_query1
			print('df_link_query: ',df_link_query.shape)
			print(df_link_query[0:5])

			field1, field2 = field_query[0:2]
			# column_1, column_2 = 'distance_bin', 'correlation_bin'
			column_1, column_2 = '%s_bin'%(field1), '%s_bin'%(field2)
			column_1_query, column_2_query = column_1, column_2
			
			column_distance, column_score_1 = column_vec_query[0:2]
			column_value_1 = '%s_abs'%(column_distance)
			df_link_query[column_value_1] = df_link_query[column_distance].abs()

			peak_distance_thresh_1 = peak_distance_thresh
			distance_abs = df_link_query[column_distance].abs()
			df_link_query[column_value_1] = distance_abs
			t_vec_1 = np.arange(0,peak_distance_thresh_1+distance_bin_value,distance_bin_value)
			df_link_query[column_1_query] = np.digitize(distance_abs,t_vec_1)

			score_query_abs = df_link_query[column_score_1].abs()
			n_bins_score = int(np.ceil(1.0/score_bin_value))
			t_vec_2 = np.linspace(0,1,n_bins_score+1)

			type_id_1 = 1
			if type_score==0:
				column_value_2 = '%s_abs'%(column_score_1)
				df_link_query[column_value_2] = score_query_abs
			else:
				# score_query = df_link_query[column_score_1]
				column_value_2 = column_score_1
			
			df_link_query[column_2_query] = np.digitize(score_query_abs,t_vec_2)
			# if (type_id_1>0) or (type_score==1):
			if (type_score==1):
				id1 = (df_link_query[column_score_1]<0)
				df_link_query.loc[id1,column_2_query] = -df_link_query.loc[id1,column_2_query]

			print('query distance and score annotation')
			start = time.time()
			df_link_query2, df_link_query2_2 = self.test_gene_peak_query_link_basic_filter_1_pre2(df_gene_peak_query=df_link_query,field_query=field_query,thresh_vec_compare=[],column_vec_query=column_vec_query,column_label='',type_id_1=type_id_1,print_mode=0,select_config=select_config)

			stop = time.time()
			print('distance and score annotation query used: %.2fs'%(stop-start))

			df_link_query2.index = np.asarray(df_link_query2['peak_id'])
			df_link_query2 = df_link_query2.sort_values(by=['peak_id',column_score_1],ascending=[True,False])
		
			filename_annot1 = '%d.%d.%d'%(type_id_1,type_query_1,thresh_type)
			filename_annot_1 = filename_annot1
			
			output_filename = '%s/%s.df_link_query2.%s.txt'%(output_file_path,filename_prefix_save_1,filename_annot_1)
			t_columns = df_link_query2.columns.difference(column_vec_2,sort=False)
			df_link_query2 = df_link_query2.loc[:,t_columns]
			df_link_query2.to_csv(output_filename,index=False,sep='\t')

			output_filename = '%s/%s.df_link_query2.1.%s.txt'%(output_file_path,filename_prefix_save_1,filename_annot_1)
			sel_num1 = 1000
			peak_query_vec = peak_query
			peak_query_1 = peak_query_vec[0:sel_num1]
			df_link_query2_1 = df_link_query2.loc[peak_query_1,:]
			df_link_query2_1.to_csv(output_filename,index=False,sep='\t')

			output_filename = '%s/%s.df_link_query2.2.%s.txt'%(output_file_path,filename_prefix_save_1,filename_annot_1)
			t_columns = df_link_query2_2.columns.difference(column_vec_2,sort=False)
			df_link_query2_2 = df_link_query2_2.loc[:,t_columns]
			df_link_query2_2.to_csv(output_filename,index=False,sep='\t')

		if flag_basic_query in [2,3]:
			type_id_1 = 1
			save_file_path2 = output_file_path
			input_file_path = save_file_path2
			# output_file_path = save_file_path2

			if flag_basic_query==2:
				filename_annot1 = '%d.%d'%(type_id_1,type_query_1)
				input_filename = '%s/%s.df_link_query2.%s.txt'%(input_file_path,filename_prefix_save_1,filename_annot1)
			else:
				df_link_query2_ori = df_link_query2

			df_link_query2_ori.index = test_query_index(df_link_query2_ori,column_vec=column_idvec)
			print('df_link_query2_ori: ',df_link_query2_ori.shape)

			gene_query_1 = df_link_query2_ori[column_id1].unique()
			print('gene_query_1: %d'%(len(gene_query_1)))

			df_link_query2 = df_link_query2_ori

			# distance_abs = df_link_query2[column_distance].abs()
			if not (column_value_1 in df_link_query2.columns):
				df_link_query2[column_value_1] = df_link_query2[column_value_1_ori].abs()

			# correlation_abs = df_link_query2[column_corr_1].abs()
			if not (column_value_2 in df_link_query2.columns):
				if type_score==0:
					df_link_query2[column_value_2] = df_link_query2[column_value_2_ori].abs()
			
			df_link_query_2 = df_link_query2.copy()

			type_query = 0
			thresh_type = 3
			if 'thresh_vec_group1' in select_config:
				thresh_vec_query = select_config['thresh_vec_group1']
				thresh_vec_query1, thresh_vec_query2 = thresh_vec_query[0:2]
				thresh_value_1, thresh_value_2 = thresh_vec_query1[0:2]
				thresh_value_1_2,thresh_value_2_2 = thresh_vec_query2[0:2]
			else:
				thresh_value_1 = 100 # distance threshold
				# thresh_value_2 = 0.1 # correlation threshold
				thresh_value_2 = 0.15 # correlation threshold
				thresh_value_1_2 = 500 # distance threshold
				# thresh_value_2_2 = 0 # correlation threshold
				thresh_value_2_2 = -0.05 # correlation threshold

				thresh_vec_query = [[thresh_value_1,thresh_value_2],[thresh_value_1_2,thresh_value_2_2]]
				# thresh_type = len(thresh_vec_query)
				select_config.update({'thresh_vec_group1':thresh_vec_query})

			if 'thresh_vec_group2' in select_config:
				thresh_vec_group2 = select_config['thresh_vec_group2']
				thresh_vec_query_1 = thresh_vec_group2[0]
				thresh_vec_query_2 = thresh_vec_group2[1]

				thresh_query3 = thresh_vec_query_2[0]
				thresh_value_1_3,thresh_value_2_3 = thresh_query3[0:2]
				# if len(thresh_vec_group2)>0:
				# 	thresh_type = thresh_type + len(thresh_vec_query_2)
			else:
				thresh_value_1_3 = -50 # distance threshold
				# thresh_value_2_2 = 0 # correlation threshold
				thresh_value_2_3 = 0.20 # correlation threshold
				thresh_vec_query_1 = [150,[0.3,0.1]]
				thresh_vec_query_2 = [[thresh_value_1_3,thresh_value_2_3]]
				thresh_vec_group2 = [thresh_vec_query_1,thresh_vec_query_2]
				select_config.update({'thresh_vec_group2':thresh_vec_group2})
			
			# verbose = 1
			# type_combine = 0
			type_combine = 1
			save_mode_2 = 1
			# filename_save_annot = '%s_%s.%s_%s.%d.2'%(thresh_value_1,thresh_value_2,thresh_value_1_2,thresh_value_2_2,type_id_1)
			filename_save_annot_1 = '%s_%s.%s_%s.%d.%d'%(thresh_value_1,thresh_value_2,thresh_value_1_2,thresh_value_2_2,type_id_1,thresh_type)
			filename_save_annot = '%s.%d'%(filename_save_annot_1,type_query_1)
			# column_query1 = 'label_thresh2'
			column_query1 = column_label

			# field_query = ['distance','correlatin']
			# column_vec_query = ['distance','spearmanr']
			# for type_combine in [0,1]:
			for type_combine in [1]:
				df_link_query2 = df_link_query_2.copy()
				print('df_link_query2: ',df_link_query2.shape)
				print(df_link_query2.columns)
				for type_query in [0,1]:
					type_query_2 = (1-type_query)
					print('peak-gene link query comparison for smaller distance and similar or higher similarity score')
					start = time.time()
					df_link_pre1, df_link_pre2 = self.test_gene_peak_query_link_basic_filter_1_pre2_1(df_feature_link=df_link_query2,field_query=field_query,column_vec_query=column_vec_query,
																										thresh_vec=thresh_vec_query,type_score=type_score,type_id_1=type_query,
																										verbose=verbose,select_config=select_config)
					print('df_link_pre1, df_link_pre2: ',df_link_pre1.shape,df_link_pre2.shape)
					stop = time.time()
					print('peak-gene link query comparison 1 used: %.2fs'%(stop-start))

					if len(thresh_vec_query_2)>0:
						print('peak-gene link query comparison for similar or smaller distance and higher similarity score')
						start = time.time()
						df_link_1, df_link_2 = self.test_gene_peak_query_link_basic_filter_1_pre2_2(df_feature_link=df_link_pre1,field_query=field_query,column_vec_query=column_vec_query,
																										thresh_vec_1=thresh_vec_query_1,thresh_vec_2=thresh_vec_query_2,type_score=type_score,type_id_1=type_query,
																										verbose=verbose,select_config=select_config)
						stop = time.time()
						print('peak-gene link query comparison 1 used: %.2fs'%(stop-start))
						df_link_2 = pd.concat([df_link_pre2,df_link_2],axis=0,join='outer',ignore_index=False)
					else:
						df_link_1, df_link_2 = df_link_pre1, df_link_pre2
					print('df_link_1, df_link_2: ',df_link_1.shape,df_link_2.shape)

					if type_combine>0:
						df_link_query2 = df_link_1
					
					if (save_mode_2>0):
						if type_combine==0:
							list1 = [df_link_1,df_link_2]
							query_num2 = len(list1)
							for i2 in range(query_num2):
								df_query = list1[i2]
								t_columns = df_query.columns.difference(column_vec_2,sort=False)
								# output_filename = '%s/%s.df_link_query2.2_%d.%d.%d.txt'%(output_file_path,filename_prefix_save_1,(i2+1),type_query,type_id_1)
								output_filename = '%s/%s.df_link_query2.2_%d.%d.%s.txt'%(output_file_path,filename_prefix_save_1,(i2+1),type_query,filename_annot1)
								df1 = df_query.loc[:,t_columns]
								df1.to_csv(output_filename,index=False,sep='\t')
								
								if type_query_1!=2:
									df2 = df1.loc[df1[column_query1]>0,:]
									output_filename = '%s/%s.df_link_query2.2_%d.%d.%s.2.txt'%(output_file_path,filename_prefix_save_1,(i2+1),type_query,filename_annot1)
									df2.to_csv(output_filename,index=False,sep='\t')

						else:
							df_query = df_link_2
							t_columns = df_query.columns.difference(column_vec_2,sort=False)
							output_filename = '%s/%s.df_link_query2.2_1.combine.%d.%s.txt'%(output_file_path,filename_prefix_save_1,type_query,filename_save_annot)
							df1 = df_query.loc[:,t_columns]
							df1.to_csv(output_filename,index=False,sep='\t')
							
							df2 = df1.loc[df1[column_query1]>0,:]
							output_filename = '%s/%s.df_link_query2.2_1.combine.%d.%s.2.txt'%(output_file_path,filename_prefix_save_1,type_query,filename_save_annot)
							df2.to_csv(output_filename,index=False,sep='\t')
				
				df_link_query_1 = df_link_1
				if (type_combine>0) and (save_mode_2)>0:
					# list1 = [df_link_1,df_link_2]
					list1 = [df_link_1]
					query_num2 = len(list1)
					for i2 in range(query_num2):
						df_query = list1[i2]
						# t_columns = df_query.columns.difference(['distance_abs','spearmanr_abs'],sort=False)
						t_columns = df_query.columns.difference(column_vec_2,sort=False)
						output_filename = '%s/%s.df_link_query2.2_%d.combine.%s.txt'%(output_file_path,filename_prefix_save_1,(i2+1),filename_save_annot)
						df1 = df_query.loc[:,t_columns]
						df1.to_csv(output_filename,index=False,sep='\t')
						
						df2 = df1.loc[df1[column_query1]>0,:]
						output_filename = '%s/%s.df_link_query2.2_%d.combine.%s.2.txt'%(output_file_path,filename_prefix_save_1,(i2+1),filename_save_annot)
						df2.to_csv(output_filename,index=False,sep='\t')

						gene_query_vec_1 = df1[column_id1].unique()
						gene_query_vec_2 = df2[column_id1].unique()
						print('df1, df2: ',df1.shape,df2.shape)
						print('gene_query_vec_1, gene_query_vec_2: ',len(gene_query_vec_1),len(gene_query_vec_2))

		return df_link_query_1

	## pre-selection of gene-peak query
	def test_gene_peak_query_link_basic_filter_1_pre2(self,peak_id=[],df_peak_query=[],df_gene_peak_query=[],field_query=[],thresh_vec_compare=[],column_idvec=['peak_id','gene_id'],column_vec_query=[],column_label='',type_score=0,type_id_1=0,print_mode=0,save_mode=0,verbose=0,select_config={}):

		column_id2, column_id1 = column_idvec[0:2]
		if len(field_query)==0:
			if type_score==0:
				field_query = ['distance','correlation'] # peak-gene correlation
			else:
				field_query = ['distance','score'] # peak-TF-gene score

		field1, field2 = field_query[0:2]
		annot_str_vec = field_query

		# column_1, column_2 = 'distance_bin', 'score_bin'
		column_1, column_2 = '%s_bin'%(field1), '%s_bin'%(field2)
		column_vec_1 = [column_1,column_2]
		column_pre1, column_pre2 = '%s_min'%(column_1), '%s_max'%(column_2)
		column_vec_2 = [column_pre1,column_pre2]

		# column_score_1: similarity score
		column_distance, column_score_1 = column_vec_query[0:2]
		# column_corr_1 = column_score_1
		
		column_value_1 = '%s_abs'%(column_distance)
		if not (column_value_1 in df_gene_peak_query.columns):
			df_gene_peak_query[column_value_1] = df_gene_peak_query[column_distance].abs()

		column_value_2 = column_score_1
		if type_score==0:
			column_value_query2 = '%s_abs'%(column_score_1)
			column_value_2 = column_value_query2
			if not (column_value_2 in df_gene_peak_query.columns):
				df_gene_peak_query[column_value_2] = df_gene_peak_query[column_score_1].abs()

			column_group_annot = 'group2'
			df_gene_peak_query[column_group_annot] = 1
			id_pre1 = (df_gene_peak_query[column_score_1]<0)
			df_gene_peak_query.loc[id_pre1,column_group_annot] = -1

		column_vec_pre1 = [column_distance,column_score_1] # the original value
		column_vec_pre2 = [column_value_1,column_value_2] # the original or absolute value
		column_annot_1, column_annot_2 = '%s_min_%s'%(field1,field2), '%s_max_%s'%(field2,field1)
		
		from utility_1 import test_query_index
		df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_idvec)
		
		query_num1 = len(column_vec_1)
		df1_query = df_gene_peak_query
		feature_type_query = column_id2
		type_query_vec = ['min','max']

		# id_1 = 1
		list_1 = []
		list_2 = []
		for id_1 in [1,0]:
			id_2 = (1-id_1)
			column_value_group1 = column_vec_pre2[id_1]	# numeric value: correlation or distance_abs
			column_value_group2 = column_vec_pre2[id_2]	# numeric value: distance_abs or correlation
			
			column_value_group1_ori = column_vec_pre1[id_1] # original score or distance value
			column_value_group2_ori = column_vec_pre1[id_2]	# original distance or score value
			
			column_group1 = column_vec_1[id_1]	# id_1:1,correlation_bin;0,distance_bin
			column_group2 = column_vec_1[id_2]	# id_2:0,distance_bin;1,correlation_bin
			
			ascending_vec = [True,False]
			if (type_score==1) or (id_1==0):
				df2_query = df1_query.sort_values(by=[column_value_group1],ascending=ascending_vec[id_1]) # sort the dataframe by correlation
				# df_gene_peak_2 = df_gene_peak_query.sort_values(by=[column_value_2],ascending=False) # sort the dataframe by correlation
			else:
				# column_value_group1_1 = '%s_abs'%(column_value_group1)
				# sort the positive and negative peak-TF correlations
				df2_query = df1_query.sort_values(by=[column_group_annot,column_value_group1],ascending=[False,ascending_vec[id_1]]) # sort the dataframe by correlation
			
			df_group_2 = df2_query.groupby(by=[feature_type_query])	# group peak-gene link query by peaks
			df_2 = df_group_2[[column_group2,column_value_group2]]

			column_vec_query1 = [column_value_group2_ori,column_value_group1_ori]
			column_annot_1 = '%s_%s'%(annot_str_vec[id_2],type_query_vec[id_2])
			column_annot_2 = '%s_%s'%(column_annot_1,annot_str_vec[id_1])
			column_vec_query2 = [['%s_2'%(column_annot_1),'%s2'%(column_annot_2)],[column_annot_1,column_annot_2]]

			print(column_value_group1,column_value_group2)
			print(column_value_group1_ori,column_value_group2_ori)
			print(column_group1,column_group2)
			print(column_vec_query1)
			print(column_vec_query2)

			list_1.append(column_vec_query2)
			if id_1==1:
				df2 = df_2.idxmin() # the link query with smallest peak-gene distance bin for each peak; the link query is sorted by peak-gene correlation
			else:
				df2 = df_2.idxmax() # the link query with highest correlation bin for each peak; the link query is sorted by peak-gene distance
			
			idvec_1 = np.asarray(df2[column_group2])	# the link query with smallest peak-gene distance bin for each peak and correlation rank 1
			idvec_2 = np.asarray(df2[column_value_group2]) # the link query with smallest peak-gene distance for each peak, which may not be the highest correlation
			
			# list_query1 = [idvec_1,idvec_1,idvec_2]
			id_2 = (idvec_1!=idvec_2)
			query_id1 = idvec_1[id_2]
			query_id2 = idvec_2[id_2]
			query_num1, query_num2 = len(query_id1),len(query_id2)
			print('query_id1:%d, query_id2:%d'%(query_num1,query_num2))
			print(query_id1)
			print(query_id2)
			list_2.append([query_id1,query_id2])
			list_query1 = [idvec_1,query_id2]

			query_num = len(list_query1)
			from utility_1 import test_column_query_2
			for i2 in range(query_num):
				query_id = list_query1[i2]
				df_query = df2_query.loc[np.asarray(query_id),:]
				print('df_query: ',df_query.shape)
				print(df_query[0:5])
				# df2.index = np.asarray(df2[column_id2])
				query_idvec = []
				if i2==1:
					query_idvec = df_query[column_id2].unique()
				df1 = test_column_query_2(df_list=[df2_query,df_query],id_column=[column_id2],query_idvec=query_idvec,column_vec_1=column_vec_query1,column_vec_2=column_vec_query2[i2],
											type_id_1=0,reset_index=True,flag_unduplicate=0,verbose=0,select_config=select_config)

			df1_query = df1

		df_gene_peak_query = df1
		query_num2 = len(list_2)
		list_query2 = []
		for i2 in range(query_num2):
			query_id1, query_id2 = list_2[i2]
			df2_1 = df_gene_peak_query.loc[np.asarray(query_id1),:]
			df2_2 = df_gene_peak_query.loc[np.asarray(query_id2),:]

			df_2 = pd.concat([df2_1,df2_2],axis=0,join='outer',ignore_index=False)
			# df_2 = df_2.sort_values(by=['peak_id','distance'],ascending=True)
			# type_query: 0, the link query with highest correlation in the smallest peak-gene distance bin is not with the smallest distance (the difference is bounded by 50Kb)
			# type_query: 1, the link query with smallest distance in the rank 1 correlation bin is not with the highest correlation (the difference is bounded by 0.05)
			df_2['type_query'] = i2
			list_query2.append(df_2)

		df_gene_peak_query2 = pd.concat(list_query2,axis=0,join='outer',ignore_index=False)
		df_gene_peak_query2 = df_gene_peak_query2.sort_values(by=['peak_id','distance'],ascending=True)

		return df_gene_peak_query, df_gene_peak_query2


