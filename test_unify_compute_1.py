# #!/usr/bin/env python
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

from test_unify_compute_pre2 import _Base2_correlation2_1

# get_ipython().run_line_magic('matplotlib', 'inline')
sc.settings.verbosity = 3    # verbosity: errors (0), warnings (1), info (2), hints (3)
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

class _Base2_correlation2(_Base2_correlation2_1):
	"""Feature association estimation
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

		_Base2_correlation2_1.__init__(self,file_path=file_path,
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

		# Initializes RepliSeq
		self.run_id = run_id
		self.cell = cell
		self.generate = generate
		self.train_chromvec = chromvec
		self.chromosome = chromvec[0]

		# path_1 = '../example_datasets/data1'
		self.path_1 = file_path
		self.config = config
		self.run_id = run_id
		# self.select_config = select_config

		# self.save_path_1 = '../data2'
		self.save_path_1 = file_path
		self.pre_rna_ad = []
		self.pre_atac_ad = []
		self.fdl = []
		self.motif_data = []
		self.motif_data_score = []
		self.motif_query_name_expr = []

		data_type_id = 1
		if 'data_type_id' in self.config:
			data_type_id = self.config['data_type_id']

		if data_type_id==0:
			# load scRNA-seq data
			# self.test_init_1()
			# self.adata = self.rna_ad_1
			cluster_name1 = 'UpdatedCellType'
		else:
			# load multiome scRNA-seq data
			load_mode = 1
			# load_mode = self.config['load_mode_metacell']
			if 'load_mode' in config:
				load_mode = config['load_mode_metacell']
			# self.test_init_2(load_mode=load_mode)
			# self.adata = self.pre_rna_ad
			cluster_name1 = 'CellType'

			# load_mode_rna, load_mode_atac = 1, 1
			load_mode_rna, load_mode_atac = 1, 1
			if 'load_mode_rna' in config:
				load_mode_rna = config['load_mode_rna']
			if 'load_mode_atac' in config:
				load_mode_atac = config['load_mode_atac']

			# use_raw_count, pre_normalize = 1, 0
			# self.test_motif_peak_estimate_rna_pre1(select_config=config,use_raw_count=use_raw_count,pre_normalize=pre_normalize,load_mode=load_mode_rna)

			# use_raw_count, pre_normalize = 1, 0
			# self.test_motif_peak_estimate_atac_pre1(select_config=config,use_raw_count=use_raw_count,pre_normalize=pre_normalize,load_mode=load_mode_atac)

		data_file_type = select_config['data_file_type']
		# input_file_path1 = self.path_1
		input_file_path1 = self.save_path_1
		
		overwrite = False
		if 'overwrite' in select_config:
			overwrite = select_config['overwrite']
		# select_config = self.test_config_query_2(overwrite=overwrite,select_config=select_config)
		self.select_config = select_config
		self.gene_name_query_expr_ = []
		self.gene_highly_variable = []
		self.peak_dict_ = []
		self.df_gene_peak_ = []
		self.df_gene_peak_list_ = []
		# self.df_gene_peak_distance = []
		self.motif_data = []
		self.gene_expr_corr_ = []
		self.df_tf_expr_corr_list_pre1 = []	# tf-tf expr correlation
		self.df_expr_corr_list_pre1 = []	# gene-tf expr correlation
		self.df_gene_query = []
		self.df_gene_peak_query = []
		self.df_gene_annot_1 = []
		self.df_gene_annot_2 = []
		# self.df_gene_annot_ori, self.df_gene_annot_expr = [], []
		self.df_gene_annot_ori = []
		self.df_gene_annot_expr = df_gene_annot_expr
		self.df_peak_annot = []
		self.pre_data_dict_1 = dict()
		self.df_rna_obs = []
		self.df_atac_obs = []
		self.df_rna_var = []
		self.df_atac_var = []
		self.df_peak_annot = []
		self.df_gene_peak_distance = []
		self.df_gene_tf_expr_corr_ = []
		self.df_gene_tf_expr_pval_ = []
		self.df_gene_expr_corr_ = []
		self.df_gene_expr_pval_ = []

	## file_path query
	def test_config_query_1(self,select_config={}):

		print('test_config_query')
		if 'data_file_type' in select_config:
			data_file_type = select_config['data_file_type']
			input_file_path1 = self.path_1
			if 'root_path_1' in select_config:
				root_path_1 = select_config['root_path_1']
			else:
				root_path_1 = input_file_path1
				select_config.update({'root_path_1':root_path_1})
			if 'root_path_2' in select_config:
				root_path_2 = select_config['root_path_2']
			else:
				root_path_2 = root_path_1
				select_config.update({'root_path_2':root_path_2})
			print('data_file_type:%s'%(data_file_type))
			if data_file_type in ['mouse_endoderm','E8.75','E9.25']:
				# input_file_path = '%s/data_pre2/mouse_endoderm'%(input_file_path1)
				input_file_path_1 = root_path_2
				input_file_path = '%s/mouse_endoderm'%(input_file_path_1)
				# root_path_2 = input_file_path
				type_id_feature = select_config['type_id_feature']
				# run_id = select_config['run_id']
				# filename_save_annot_1 = 'E8.75#Multiome'
				input_filename_1 = '%s/E8.75_E9.25_rna_endoderm_with_scrna.2.h5ad'%(input_file_path1)
				input_filename_2 = '%s/E8.75_E9.25_atac_endoderm_with_scatac.2.h5ad'%(input_file_path1)
				data_timepoint = select_config['data_timepoint']
				# run_id_load = -1	# the previous annotation
				# if 'run_id_load' in select_config:
				# 	run_id_load = select_config['run_id_load']
				# if run_id_load>0:
				# 	filename_save_annot_1 = '%s.%d'%(data_timepoint,run_id_load)
				# else:
				# 	filename_save_annot_1 = 'E8.75#Multiome'
				select_config.update({'input_filename_rna':input_filename_1,
									'input_filename_atac':input_filename_2})
				print('input_filename_rna:%s, input_filename_atac:%s'%(input_filename_1,input_filename_2))
			else:
				print('use input file path ')

		self.save_path_1 = self.path_1
		input_file_path1 = self.save_path_1
		root_path_1 = select_config['root_path_1']
		root_path_2 = select_config['root_path_2']
		if 'data_file_type' in select_config:
			data_file_type = select_config['data_file_type']
			data_file_type_annot = data_file_type.lower()
			data_file_type_vec = ['CD34_bonemarrow','pbmc','bonemarrow_Tcell','E8.75','E9.25']
			file_path_annot = ['cd34_bonemarrow','10x_pbmc/data_1','bonemarrow_Tcell','mouse_endoderm/data_1/E8.75','mouse_endoderm/data_1/E9.25']
			file_path_dict = dict(zip(data_file_type_vec,file_path_annot))
			human_cell_type = ['CD34_bonemarrow','pbmc','bonemarrow_Tcell']
			mouse_cell_type = ['E8.75','E9.25']
			if data_file_type in human_cell_type:
				self.species_id = 'hg38'
			else:
				self.species_id = 'mm10'
			select_config.update({'file_path_dict':file_path_dict})

		return select_config

	## file_path query
	def test_config_query_2(self,beta_mode=0,save_mode=1,overwrite=False,select_config={}):

		print('test_config_query')
		# if 'data_file_type' in select_config:
		# 	data_file_type = select_config['data_file_type']
		# 	input_file_path1 = self.path_1
		# 	root_path_1 = select_config['root_path_1']
		# 	print('data_file_type:%s'%(data_file_type))
		# file_path_dict = self.file_path_dict
		file_path_dict = select_config['file_path_dict']
		# filename_prefix_1 = select_config['filename_prefix_default']
		# filename_prefix_save_1 = select_config['filename_prefix_save_local']
		# filename_annot_vec = select_config['filename_annot_local']
		data_file_type = select_config['data_file_type']
		root_path_2 = select_config['root_path_2']
		file_path_annot = file_path_dict[data_file_type]
		# data_dir = '%s/%s'%(input_file_path1,file_path_annot)
		data_dir = '%s/%s'%(root_path_2,file_path_annot)
		run_id = select_config['run_id']
		type_id_feature = select_config['type_id_feature']
		metacell_num = select_config['metacell_num']
		input_file_path = '%s/metacell_%d/run%d'%(data_dir,metacell_num,run_id)
		print('root_path_2: %s \n file_path_annot: %s \n data_dir: %s \n'%(root_path_2,file_path_annot,data_dir))
		print('input_file_path: %s'%(input_file_path))

		filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
		data_path_1 = select_config['data_path']
		select_config.update({'data_path':input_file_path,
								'data_path_1':data_path_1,
								'filename_save_annot_1':filename_save_annot_1,
								'filename_save_annot_pre1':filename_save_annot_1})

		if (data_file_type in ['mouse_endoderm','E8.75','E9.25']) and ('run_id_load' in select_config):
			data_timepoint = select_config['data_timepoint']
			input_file_path_2 = select_config['data_path_2']
			# print('input_file_path_2:%s'%(input_file_path_2))
			filename_save_annot_2 = '%s#Multiome'%(data_timepoint)
			input_filename_1 = '%s/test_%s_meta_rna.normalize.log1.1.1.0.h5ad'%(input_file_path_2,filename_save_annot_2)
			input_filename_2 = '%s/test_%s_meta_atac.normalize.log1.1.1.0.h5ad'%(input_file_path_2,filename_save_annot_2)
			# input_filename_3 = '%s/test_rna_meta_ad.%s.0.1.meta_scaled_exprs.2.txt'%(input_file_path_2,data_timepoint)
			input_filename_3 = '%s/test_%s_meta_exprs.normalize.log1_scale2.2.1.0.txt'%(input_file_path_2,filename_save_annot_2)
			run_id_load = select_config['run_id_load']
			if run_id_load<0:
				# input_filename_1 = '%s/test_E8.75#Multiome_meta_rna.normalize.log1.1.1.0.h5ad'%(input_file_path)
				# input_filename_2 = '%s/test_E8.75#Multiome_meta_atac.normalize.log1.1.1.0.h5ad'%(input_file_path)
				# input_filename_3 = '%s/test_E8.75#Multiome_meta_exprs.normalize.log1_scale2.2.1.0.txt'%(input_file_path)
				input_filename_3 = '%s/test_%s_meta_exprs.normalize.log1_scale2.2.1.0.txt'%(input_file_path_2,filename_save_annot_2)
			# else:
			# 	input_filename_3 = '%s/test_rna_meta_ad.%s.0.1.meta_scaled_exprs.2.txt'%(input_file_path_2,data_timepoint)
		else:
			input_filename_1 = '%s/test_rna_meta_ad.%s.h5ad'%(input_file_path,filename_save_annot_1)
			input_filename_2 = '%s/test_atac_meta_ad.%s.h5ad'%(input_file_path,filename_save_annot_1)
			input_filename_3 = '%s/test_rna_meta_ad.%s.meta_scaled_exprs.2.txt'%(input_file_path,filename_save_annot_1)

		select_config.update({'filename_rna':input_filename_1,'filename_atac':input_filename_2,
								'filename_rna_exprs_1':input_filename_3})

		# filename_prefix_default = 'test_gene_peak_local_1'
		filename_prefix_default = 'test_query_gene_peak_local_1'
		filename_annot_default = '1'
		filename_prefix_save_1 = 'pre1'
		filename_prefix_save_2 = 'pre2'
		filename_prefix_default_1 = '%s.%s'%(filename_prefix_default,filename_prefix_save_1)
		select_config.update({'filename_prefix_default':filename_prefix_default,
								'filename_prefix_default_1':filename_prefix_default_1,
								'filename_annot_default':filename_annot_default,
								'filename_prefix_save_default':filename_prefix_save_1,
								'filename_prefix_save_2':filename_prefix_save_2})
		peak_bg_num_ori = 100
		peak_bg_num = 100
		# interval_peak_corr = 500
		# interval_local_peak_corr = -1
		interval_peak_corr = 10
		interval_local_peak_corr = -1
		input_filename_peak = '%s/test_peak_GC.1.1.bed'%(input_file_path)
		input_filename_bg = '%s/test_peak_read.%s.normalize.bg.%d.1.csv'%(input_file_path,data_file_type,peak_bg_num_ori)

		list1 = [peak_bg_num,interval_peak_corr,interval_local_peak_corr]
		field_query = ['peak_bg_num','interval_peak_corr','interval_local_peak_corr']
		# print('select_config ',select_config)
		query_num1 = len(list1)
		for i1 in range(query_num1):
			field_id = field_query[i1]
			query_value = list1[i1]
			if (not (field_id in select_config)) or (overwrite==True):
				select_config.update({field_id:query_value})

		select_config.update({'input_filename_peak':input_filename_peak,
								'input_filename_bg':input_filename_bg})

		# filename_prefix_save_1 = select_config['filename_prefix_save_local']
		# filename_annot_vec = select_config['filename_annot_local']
		# column_highly_variable = 'highly_variable_thresh%s'%(highly_variable_thresh)
		# correlation_type = 'spearmanr'
		# save_file_path_local = select_config['save_file_path_local']
		# type_id_correlation = select_config['type_id_correlation']
		# field_query = ['interval_peak_corr_bg','interval_local_peark_corr']

		save_file_path = '%s/peak_local'%(data_path_1)
		if os.path.exists(save_file_path)==False:
			print('the directory does not exist ',save_file_path)
			os.mkdir(save_file_path)
		select_config.update({'data_path_save':save_file_path})

		filename_prefix_1 = select_config['filename_prefix_default']
		filename_prefix_save_1 = select_config['filename_prefix_save_default']
		filename_prefix = '%s.%s'%(filename_prefix_1,filename_prefix_save_1)
		filename_annot1 = filename_annot_default
		select_config.update({'filename_prefix_peak_gene':filename_prefix})

		input_filename_pre1 = '%s/%s.combine.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
		input_filename_pre2 = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix,filename_annot1)

		# output_filename_1 = input_filename_pre2
		# output_filename_2 = '%s/%s.combine.thresh2.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
		filename_peak_gene_thresh2 = '%s/%s.combine.thresh2.%s.txt'%(save_file_path,filename_prefix,filename_annot1)

		filename_save_annot_pre1 = select_config['filename_save_annot_pre1']
		filename_distance_annot = '%s/df_gene_peak_distance_annot.%s.txt'%(save_file_path,filename_save_annot_pre1)

		# column_highly_variable = 'highly_variable_thresh0.5'
		# correlation_type = 'spearmanr'
		highly_variable = True
		highly_variable_thresh = 0.5
		column_correlation = ['spearmanr','pval1','pval1_ori']
		correlation_type_1 = column_correlation[0]
		column_distance = 'distance'
		column_thresh2 = 'label_thresh2'
		column_highly_variable = 'highly_variable_thresh%s'%(highly_variable_thresh)
		# correlation_type = 'spearmanr'

		## threshold for pre-selection of peak-gene links to estimate empirical p-values
		thresh_distance_1 = 50 # to update
		thresh_corr_distance_1 = [[0,thresh_distance_1,0],
									[thresh_distance_1,500,0.01],
									[500,1000,0.1],
									[1000,2050,0.15]]

		## threshold for pre-selection of peak-gene links as candidate peaks
		thresh_distance_1_2 = 50
		thresh_corr_distance_2 = [[0,thresh_distance_1_2,[[0,1,0,1]]],
									[thresh_distance_1_2,500,[[0.01,0.1,-0.01,0.1],[0.15,0.15,-0.15,0.15]]],
									[500,1000,[[0.1,0.1,-0.1,0.1]]],
									[1000,2050,[[0.15,0.1,-0.15,0.1]]]]

		thresh_corr_retain = [0.3,0.35]

		list2 = [input_filename_pre1,input_filename_pre2,filename_peak_gene_thresh2,filename_distance_annot,
					column_correlation,correlation_type_1,column_distance,column_highly_variable,column_thresh2,
					thresh_distance_1,thresh_corr_distance_1,
					thresh_distance_1_2,thresh_corr_distance_2,thresh_corr_retain,beta_mode]

		field_query2 = ['input_filename_pre1','input_filename_pre2','filename_save_thresh2','filename_distance_annot',
						'column_correlation','correlation_type_1','column_distance','column_highly_variable','column_thresh2',
						'thresh_distance_default_1','thresh_corr_distance_1','thresh_distance_default_2','thresh_corr_distance_2',
						'thresh_corr_retain','beta_mode']

		query_num2 = len(field_query2)
		for i1 in range(query_num2):
			field_id = field_query2[i1]
			query_value = list2[i1]
			select_config.update({field_id:query_value})

		return select_config

	## update field query
	def test_field_query_pre1(self,field_query=[],query_value=[],select_config={}):
		# query_num1 = len(query_value)
		for (field_id,query_value) in zip(field_query,query_value):
			select_config.update({field_id:query_value})

		return select_config

	## query peak-gene link attributes
	def test_gene_peak_query_attribute_1(self,df_gene_peak_query=[],df_gene_peak_query_ref=[],column_idvec=[],field_query=[],column_name=[],reset_index=True,verbose=0,select_config={}):

		if verbose>0:
			print('df_gene_peak_query, df_gene_peak_query_ref ',df_gene_peak_query.shape,df_gene_peak_query_ref.shape)
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

	## peak num and gene num query for gene and peak
	def test_peak_gene_query_basic_1(self,data=[],input_filename='',save_mode=1,filename_prefix_save='',output_filename='',output_file_path='',verbose=0,select_config={}):

		if output_file_path=='':
			output_file_path = self.save_path_1
		if filename_prefix_save=='':
			filename_prefix_save = 'df'

		df_gene_peak_query_group_1, df_gene_peak_query_group_2 = [], []
		df_query = data

		feature_type_vec = ['gene','peak']
		feature_type_query1, feature_type_query2 = feature_type_vec
		# column_id_1 = '%s_id'%(feature_type_query1)
		column_idvec= ['%s_id'%(feature_type_query) for feature_type_query in feature_type_vec]
		column_id1, column_id2 = column_idvec
		
		flag_query=1
		if len(df_query)>0:
			df_gene_peak_query_1 = df_query
		else:
			if os.path.exists(input_filename)==True:
				# df_gene_peak_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				df_gene_peak_query_1 = pd.read_csv(input_filename,index_col=False,sep='\t')
				df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1[column_id1])
				print('df_gene_peak_query_1 ',df_gene_peak_query_1.shape)
			else:
				print('the file does not exist ',input_filename)
				flag_query=0
				return

		# feature_type_vec = ['gene','peak']
		# feature_type_query1, feature_type_query2 = feature_type_vec
		# # column_id_1 = '%s_id'%(feature_type_query1)
		# column_idvec= ['%s_id'%(feature_type_query) for feature_type_query in feature_type_vec]
		# column_id1, column_id2 = column_idvec
		# column_distance = select_config['column_distance']
		# column_correlation1 = select_config['column_correlation'][0]
		# gene_query_vec_ori = df_gene_peak_query_1['gene_id'].unique()
		gene_query_vec_ori = df_gene_peak_query_1[column_id1].unique()
		gene_query_num_ori = len(gene_query_vec_ori)
		if verbose>0:
			print('gene_query_vec_ori ',gene_query_num_ori)
		if flag_query>0:
			# thresh_highly_variable = 0.5
			column_id_query1 = select_config['column_highly_variable']
			if column_id_query1 in df_gene_peak_query_1.columns:
				id1=(df_gene_peak_query_1[column_id_query1]>0)
				id2=(~id1)
				# gene_idvec_1 = df_gene_peak_query_1.loc[id1,'gene_id'].unique()
				# gene_idvec_2 = df_gene_peak_query_1.loc[id2,'gene_id'].unique()
				# df_gene_peak_query_1.loc[pd.isna(df_gene_peak_query_1[column_id_query])==True,column_id_query]=0
				gene_idvec_1 = df_gene_peak_query_1.loc[id1,column_id1].unique()
				gene_idvec_2 = df_gene_peak_query_1.loc[id2,column_id2].unique()
				# df_gene_peak_query_1.loc[id1,column_id_query1]=1
				df_gene_peak_query_1.loc[id2,column_id_query1]=0
			else:
				# rna_meta_ad = self.rna_meta_ad
				# df_var_1 = rna_meta_ad.var
				# id1 = (df_var_1['highly_variable']>0)
				# gene_name_query_expr = df_var_1.index
				df_annot = self.df_gene_annot_expr
				gene_name_query_expr = df_annot.index
				id1 = (df_annot['highly_variable']>0)
				gene_highly_variable = gene_name_query_expr[id1]
				gene_group2 = gene_name_query_expr[(~id1)]
				gene_query_num1, gene_query_num2 = len(gene_highly_variable), len(gene_group2)
				if verbose>0:
					print('gene_highly_variable, gene_group2 ',gene_query_num1,gene_query_num2)
				gene_idvec_1 = pd.Index(gene_highly_variable).intersection(gene_query_vec_ori,sort=False)
				gene_idvec_2 = pd.Index(gene_group2).intersection(gene_query_vec_ori,sort=False)
				df_gene_peak_query_1[column_id_query1] = 0
				df_gene_peak_query_1.loc[gene_idvec_1,column_id_query1] = 1

			gene_num1, gene_num2 = len(gene_idvec_1), len(gene_idvec_2)
			# column_id_query2 = '%s_count'%(column_id_query)
			# df_gene_peak_query_1[column_id_query2] = 0
			# df_gene_peak_query_1.loc[gene_idvec_1,column_id_query2] = 1
			if verbose>0:
				print('gene_idvec_1, gene_idvec_2 ',gene_num1,gene_num2)

			df_gene_peak_query_1['count'] = 1
			# df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1['gene_id'])
			df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1[column_id1])
			column_distance = select_config['column_distance']
			column_distance_1 = '%s_abs'%(column_distance)
			# df_gene_peak_query_1['distance_abs'] = df_gene_peak_query_1['distance'].abs()
			df_gene_peak_query_1[column_distance_1] = df_gene_peak_query_1[column_distance].abs()
			column_correlation_1 = select_config['column_correlation'][0]
			# column_id_1 = 'gene_id'
			# column_vec = ['gene_id','peak_id']
			# query_vec_1 = ['peak_num','gene_num']
			column_vec = column_idvec
			query_num1 = len(column_idvec)
			column_1, column_2 = '%s_num'%(feature_type_query1), '%s_num'%(feature_type_query2)
			query_vec_1 = [column_2,column_1]
			# query_vec_2 = [['peak_num',column_id_query1],['gene_num']]
			query_vec_2 = [[column_2,column_id_query1],[column_1]]
			# annot_vec = ['gene_query1','peak_query1']
			# annot_vec = ['%s_query1'%(feature_type_query) for feature_type_query in feature_type_vec]
			annot_vec = ['%s_basic'%(feature_type_query) for feature_type_query in feature_type_vec]
			column_vec_query = [column_distance_1,column_correlation_1]
			column_vec_annot = ['distance','corr']
			list_query1 = []
			for i1 in range(query_num1):
				column_id_1 = column_vec[i1]
				column_name, column_sort = query_vec_1[i1], query_vec_2[i1]
				filename_annot1 = annot_vec[i1]
				df_gene_peak_query_group = df_gene_peak_query_1.loc[:,[column_id_1,'count']].groupby(by=[column_id_1]).sum()
				df_gene_peak_query_group = df_gene_peak_query_group.rename(columns={'count':column_name})
				# if column_id_1=='gene_id':
				if column_id_1==column_id1:
					df_gene_peak_query_group.loc[gene_idvec_1,column_id_query1] = 1
				df_gene_peak_query_group = df_gene_peak_query_group.sort_values(by=column_sort,ascending=False)
				# filename_annot1 = 'gene_query1'
				# query_vec_1 = ['max','min']
				# column_vec_query = [column_distance_1,column_correlation]
				# column_vec_annot = ['distance','corr']
				# df_gene_peak_query_group_1_distance_1 = df_gene_peak_query_1.loc[:,[column_id_1,'distance_abs']].groupby(by=[column_id_1]).max().rename(columns={'distance_abs':'distance_max'})
				# df_gene_peak_query_group_1_distance_2 = df_gene_peak_query_1.loc[:,[column_id_1,'distance_abs']].groupby(by=[column_id_1]).min().rename(columns={'distance_abs':'distance_min'})
				# df_gene_peak_query_group_1_corr_1 = df_gene_peak_query_1.loc[:,[column_id_1,'spearmanr']].groupby(by=[column_id_1]).max().rename(columns={'spearmanr':'corr_max'})
				# df_gene_peak_query_group_1_corr_2 = df_gene_peak_query_1.loc[:,[column_id_1,'spearmanr']].groupby(by=[column_id_1]).min().rename(columns={'spearmanr':'corr_min'})
				query_num2 = len(column_vec_query)
				list1 = [df_gene_peak_query_group]
				for i2 in range(query_num2):
					column_query = column_vec_query[i2]
					column_annot = column_vec_annot[i2]
					df_query_1 = df_gene_peak_query_1.loc[:,[column_id_1,column_query]]
					df_query1 = df_query_1.groupby(by=[column_id_1]).max().rename(columns={column_query:'%s_max'%(column_annot)})
					df_query2 = df_query_1.groupby(by=[column_id_1]).min().rename(columns={column_query:'%s_min'%(column_annot)})
					list1.extend([df_query1,df_query2])

				# list1 = [df_gene_peak_query_group_1,df_gene_peak_query_group_1_distance_1,df_gene_peak_query_group_1_distance_2,df_gene_peak_query_group_1_corr_1,df_gene_peak_query_group_1_corr_2]
				df_gene_peak_query_group_combine = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				if verbose>0:
					print('median_value')
					print(df_gene_peak_query_group_combine.median(axis=0))
					print('mean_value')
					print(df_gene_peak_query_group_combine.mean(axis=0))

				if save_mode>0:
					# output_filename_1 = '%s/%s.gene_query1.txt'%(output_file_path,filename_prefix_save)
					output_filename_1 = '%s/%s.%s.txt'%(output_file_path,filename_prefix_save,filename_annot1)
					# df_gene_peak_query_group_1_combine.to_csv(output_filename_1,sep='\t',float_format='%d')
					df_gene_peak_query_group_combine.to_csv(output_filename_1,sep='\t')
					print('save file: ',output_filename_1)
					list_query1.append(df_gene_peak_query_group_combine)

			df_gene_peak_query_group_1, df_gene_peak_query_group_2 = list_query1

		return df_gene_peak_query_group_1, df_gene_peak_query_group_2

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_gene_pre1_1(self,gene_query_vec=[],peak_distance_thresh=500,df_peak_query=[],
														filename_prefix_save='',input_filename='',peak_loc_query=[],
														atac_ad=[],rna_exprs=[],highly_variable=False,parallel=0,
														save_mode=1,output_filename='',save_file_path='',interval_peak_corr=50,interval_local_peak_corr=10,
														annot_mode=1,verbose=0,select_config={}):

		# atac_ad = self.atac_meta_ad
		# rna_exprs = self.meta_scaled_exprs
		
		if len(peak_loc_query)==0:
			atac_ad = self.atac_meta_ad
			peak_loc_query = atac_ad.var_names
		
		df_gene_annot_expr = self.df_gene_annot_expr
		df_gene_query_1 = df_gene_annot_expr
		print('df_gene_annot_expr: ',df_gene_annot_expr.shape)
		
		## search for peaks within the distance threshold of gene query
		# flag_distance_query=0
		flag_peak_query=1
		if flag_peak_query>0:
			# input_filename = '%s/test_gene_annot_expr.%s.peak_query.%d.txt'%(output_file_path,filename_prefix_1,peak_distance_thresh)
			if os.path.exists(input_filename)==False:
				print('the file does not exist: %s'%(input_filename))
				print('search for peak loci within distance %d Kb of the gene TSS '%(peak_distance_thresh))
				start = time.time()
				# df_gene_query_1 = df_gene_query_1[0:5000]
				type_id2 = 0
				save_mode = 1
				# output_filename = '%s/test_gene_annot_expr.%s.peak_query.%d.txt'%(output_file_path,filename_prefix_1,peak_distance_thresh)
				df_gene_peak_query = self.test_gene_peak_query_distance(gene_query_vec=gene_query_vec,
																		df_gene_query=df_gene_query_1,
																		peak_loc_query=peak_loc_query,
																		peak_distance_thresh=peak_distance_thresh,
																		type_id_1=type_id2,parallel=parallel,
																		save_mode=save_mode,
																		output_filename=output_filename,
																		verbose=verbose,
																		select_config=select_config)
				stop = time.time()
				print('search for peak loci within distance %d Kb of the gene TSS used %.5fs'%(peak_distance_thresh, stop-start))
			else:
				print('load gene-peak link query: %s'%(input_filename))
				df_gene_peak_query = pd.read_csv(input_filename,index_col=False,sep='\t')

			df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
			self.df_gene_peak_distance = df_gene_peak_query
			print('df_gene_peak_query: ',df_gene_peak_query.shape)

			# return True
			return df_gene_peak_query

	## gene-peak association query: seach for peaks within specific distance of the gene TSS
	# for each gene query, search for peaks within the distance threshold
	# input: the gene query, the gene position and TSS annotation, the peak loci query, the peak distance threshold
	# output: peak loci within specific distance of the gene TSS (dataframe)
	def test_gene_peak_query_distance(self,gene_query_vec=[],df_gene_query=[],peak_loc_query=[],peak_distance_thresh=500,type_id_1=0,parallel=0,save_mode=1,output_filename='',verbose=0,select_config={}):

		file_path1 = self.save_path_1
		if type_id_1==0:
			df_gene_query.index = np.asarray(df_gene_query['gene_name'])
			if len(gene_query_vec)>0:
				gene_query_vec_ori = gene_query_vec.copy()
				gene_query_num_ori = len(gene_query_vec_ori)
				gene_query_vec = pd.Index(gene_query_vec).intersection(df_gene_query.index,sort=False)
			else:
				gene_query_vec = df_gene_query.index
				gene_query_num_ori = len(gene_query_vec)

			gene_query_num = len(gene_query_vec)
			if not ('tss1' in df_gene_query.columns):
				df_gene_query['tss1'] = df_gene_query['start_site']
			df_tss_query = df_gene_query['tss1']
			if verbose>0:
				print('gene_query_vec_ori: %d, gene_query_vec: %d'%(gene_query_num_ori, gene_query_num))

			if len(peak_loc_query)==0:
				atac_ad = self.atac_meta_ad
				peak_loc_query = atac_ad.var_names

			peaks_pr = utility_1.pyranges_from_strings(peak_loc_query)
			peak_loc_num = len(peak_loc_query)

			bin_size = 1000
			span = peak_distance_thresh*bin_size
			if verbose>0:
				print('peak_loc_query: %d'%(peak_loc_num))
				print('peak_distance_thresh: %d bp'%(span))

			start = time.time()
			list1 = []
			interval = 5000
			if parallel==0:
				for i1 in range(gene_query_num):
					gene_query = gene_query_vec[i1]
					start = df_tss_query[gene_query]-span
					stop = df_tss_query[gene_query]+span
					chrom = df_gene_query.loc[gene_query,'chrom']
					gene_pr = pr.from_dict({'Chromosome':[chrom],'Start':[start],'End':[stop]})
					gene_peaks = peaks_pr.overlap(gene_pr)  # search for peak loci within specific distance of the gene
					# if i1%1000==0:
					if i1%interval==0:
						print('gene_peaks ', len(gene_peaks), gene_query, chrom, start, stop, i1)

					if len(gene_peaks)>0:
						df1 = pd.DataFrame.from_dict({'chrom':gene_peaks.Chromosome.values,
											'start':gene_peaks.Start.values,'stop':gene_peaks.End.values})

						df1.index = [gene_query]*df1.shape[0]
						list1.append(df1)
					else:
						print('gene query without peaks in the region query: %s %d'%(gene_query,i1))

				df_gene_peak_query = pd.concat(list1,axis=0,join='outer',ignore_index=False,keys=None,levels=None,names=None,verify_integrity=False,copy=True)

			else:
				interval_2 = 500
				iter_num = int(np.ceil(gene_query_num/interval_2))
				for iter_id in range(iter_num):
					start_id1 = int(iter_id*interval_2)
					start_id2 = np.min([(iter_id+1)*interval_2,gene_query_num])
					iter_vec = np.arange(start_id1,start_id2)
					res_local = Parallel(n_jobs=-1)(delayed(self.test_gene_peak_query_distance_unit1)(gene_query=gene_query_vec[i1],peaks_pr=peaks_pr,df_gene_annot=df_gene_query,df_annot_2=df_tss_query,
																	span=span,query_id=i1,interval=interval,save_mode=1,verbose=verbose,select_config=select_config) for i1 in tqdm(iter_vec))

					for df_query in res_local:
						if len(df_query)>0:
							list1.append(df_query)
				
				df_gene_peak_query = pd.concat(list1,axis=0,join='outer',ignore_index=False,keys=None,levels=None,names=None,verify_integrity=False,copy=True)

			df_gene_peak_query['gene_id'] = np.asarray(df_gene_peak_query.index)
			df_gene_peak_query.loc[df_gene_peak_query['start']<0,'start']=0
			query_num1 = df_gene_peak_query.shape[0]
			peak_id = utility_1.test_query_index(df_gene_peak_query,column_vec=['chrom','start','stop'],symbol_vec=[':','-'])
			df_gene_peak_query['peak_id'] = np.asarray(peak_id)
			if (save_mode==1) and (output_filename!=''):
				df_gene_peak_query = df_gene_peak_query.loc[:,['gene_id','peak_id']]
				df_gene_peak_query.to_csv(output_filename,index=False,sep='\t')

			stop = time.time()
			# print('search for peaks within distance threshold of gene query used %.2fs'%(stop-start))
		else:
			print('load existing peak-gene link query')
			input_filename = output_filename
			df_gene_peak_query = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_gene_peak_query['gene_id'] = np.asarray(df_gene_peak_query.index)
			gene_query_vec = df_gene_peak_query['gene_id'].unique()

		print('df_gene_peak_query ', df_gene_peak_query.shape)
		print('query peak distance to gene TSS ')
		df_gene_peak_query = self.test_gene_peak_query_distance_pre1(gene_query_vec=gene_query_vec,
																	df_gene_peak_query=df_gene_peak_query,
																	df_gene_query=df_gene_query,
																	select_config=select_config)

		if (save_mode==1) and (output_filename!=''):
			df_gene_peak_query = df_gene_peak_query.loc[:,['gene_id','peak_id','distance']]
			df_gene_peak_query.to_csv(output_filename,index=False,sep='\t')

		return df_gene_peak_query

	## gene-peak association query: 
	def test_gene_peak_query_distance_unit1(self,gene_query,peaks_pr,df_gene_annot=[],df_annot_2=[],span=2000,query_id=-1,interval=5000,save_mode=1,verbose=0,select_config={}):
		
		df_gene_query = df_gene_annot
		df_tss_query = df_annot_2
		
		start = df_tss_query[gene_query]-span
		stop = df_tss_query[gene_query]+span
		chrom = df_gene_query.loc[gene_query,'chrom']
		
		gene_pr = pr.from_dict({'Chromosome':[chrom],'Start':[start],'End':[stop]})
		gene_peaks = peaks_pr.overlap(gene_pr)  # search for peak loci within specific distance of the gene
		
		if (interval>0) and (query_id%interval==0):
			print('gene_peaks ', len(gene_peaks), gene_query, chrom, start, stop, query_id)

		if len(gene_peaks)>0:
			df1 = pd.DataFrame.from_dict({'chrom':gene_peaks.Chromosome.values,
										'start':gene_peaks.Start.values,'stop':gene_peaks.End.values})

			df1.index = [gene_query]*df1.shape[0]
		else:
			print('gene query without peaks in the region query: %s %d'%(gene_query,query_id))
			df1 = []

		return df1

	## gene-peak association query: peak distance to the gene TSS query
	# input: the gene query, the gene-peak pair query, the gene position and TSS annotation
	# output: update the peak-gene distance in the gene-peak pair annotation (dataframe)
	def test_gene_peak_query_distance_pre1(self,gene_query_vec=[],df_gene_peak_query=[],df_gene_query=[],select_config={}):

		file_path1 = self.save_path_1
		gene_query_id = np.asarray(df_gene_peak_query['gene_id'])
		field_query_1 = ['chrom','start','stop','strand']

		flag_query1=1
		if flag_query1>0:
			list1 = [np.asarray(df_gene_query.loc[gene_query_id,field_query1]) for field_query1 in field_query_1]
			chrom1, start1, stop1, strand1 = list1
			start_site = start1
			id1 = (strand1=='-')
			start_site[id1] = stop1[id1]
			start_site = np.asarray(start_site)

		field_query_2 = ['chrom','start','stop']
		if not ('start' in df_gene_peak_query.columns):
			peak_id = pd.Index(df_gene_peak_query['peak_id'])
			chrom2, start2, stop2 = utility_1.pyranges_from_strings_1(peak_id,type_id=0)
			df_gene_peak_query['chrom'] = chrom2
			df_gene_peak_query['start'], df_gene_peak_query['stop'] = start2, stop2
		else:
			list1 = [np.asarray(df_gene_peak_query[field_query1]) for field_query1 in field_query_2]
			chrom2, start2, stop2 = list1

		peak_distance = start2-start_site
		peak_distance_2 = stop2-start_site

		id1 = (peak_distance<0)
		id2 = (peak_distance_2<0)
		peak_distance[id1&(~id2)]=0
		peak_distance[id2] = peak_distance_2[id2]

		print('peak-gene association: ', df_gene_peak_query.shape, df_gene_peak_query.columns, df_gene_peak_query[0:5])
		print('peak_distance: ', peak_distance.shape)
		bin_size = 1000.0
		df_gene_peak_query['distance'] = np.asarray(peak_distance/bin_size)

		return df_gene_peak_query

	## load peak-gene distance;
	def test_gene_peak_query_distance_load(self,input_filename_distance='',peak_distance_thresh=2000,select_config={}):

		df_gene_peak_query_distance = pd.read_csv(input_filename_distance,index_col=0,sep='\t')
		print('df_gene_peak_query_distance ',df_gene_peak_query_distance.shape)

		return df_gene_peak_query_distance

	## load background peak loci
	def test_gene_peak_query_bg_load(self,input_filename_peak='',input_filename_bg='',peak_bg_num=100,verbose=0,select_config={}):

		# peak_counts = pd.read_csv(input_filename_peak,index_col=0)
		if input_filename_peak=='':
			input_filename_peak = select_config['input_filename_peak']
		if input_filename_bg=='':
			input_filename_bg = select_config['input_filename_bg']

		peak_query = pd.read_csv(input_filename_peak,header=None,index_col=False,sep='\t')
		# self.atac_meta_peak_loc = np.asarray(peak_counts.index)
		peak_query.columns = ['chrom','start','stop','name','GC','score']
		peak_query.index = ['%s:%d-%d'%(chrom_id,start1,stop1) for (chrom_id,start1,stop1) in zip(peak_query['chrom'],peak_query['start'],peak_query['stop'])]
		atac_ad = self.atac_meta_ad
		peak_loc_1 = atac_ad.var_names
		assert list(peak_query.index) == list(peak_loc_1)
		self.atac_meta_peak_loc = np.asarray(peak_query.index)
		print('atac matecell peaks', len(self.atac_meta_peak_loc), self.atac_meta_peak_loc[0:5])

		# input_filename_bg = '%s/chromvar/test_e875_endoderm_rna_chromvar_bg.%s.2.csv'%(self.path_1,annot1)
		# print(input_filename_bg)
		peak_bg = pd.read_csv(input_filename_bg,index_col=0)
		peak_bg_num_ori = peak_bg.shape[1]
		peak_id = np.int64(peak_bg.index)
		# print('background peaks', peak_bg.shape, len(peak_id), peak_bg_num)
		# peak_bg_num = 10
		peak_bg.index = self.atac_meta_peak_loc[peak_id-1]
		peak_bg = peak_bg.loc[:,peak_bg.columns[0:peak_bg_num]]
		peak_bg.columns = np.arange(peak_bg_num)
		peak_bg = peak_bg.astype(np.int64)
		self.peak_bg = peak_bg
		# print('background peaks', peak_bg.shape, len(peak_id), peak_bg_num, peak_bg.index[0:10])
		if verbose>0:
			print(input_filename_peak)
			print(input_filename_bg)
			print('atac matecell peaks', len(self.atac_meta_peak_loc), self.atac_meta_peak_loc[0:5])
			print('background peaks', peak_bg.shape, len(peak_id), peak_bg_num)

		return peak_bg

	## peak attribute query: peak open ratio query
	def test_peak_access_query_basic_1(self,peak_read=[],rna_exprs=[],df_annot=[],thresh_value=0.1,flag_ratio=1,flag_access=1,save_mode=1,filename_annot='',
											output_file_path='',output_filename='',verbose=0,select_config={}):

		# peak_loc_ori = motif_data.index
		peak_loc_ori = peak_read.columns
		sample_num = peak_read.shape[0]
		if len(df_annot)>0:
			df1 = df_annot
		else:
			df1 = pd.DataFrame(index=peak_loc_ori,columns=['ratio'],dtype=np.float32)

		field_query = []
		if flag_ratio>0:
			peak_read_num1 = (peak_read.loc[:,peak_loc_ori]>0).sum(axis=0)
			ratio_1 = peak_read_num1/sample_num
			thresh_1 = thresh_value
			peak_read_num2 = (peak_read.loc[:,peak_loc_ori]>thresh_1).sum(axis=0)
			ratio_2 = peak_read_num2/sample_num
			# df1 = pd.DataFrame(index=peak_loc_ori,columns=['ratio'],data=np.asarray(ratio_1))
			column_1 = 'ratio'
			column_2 = 'ratio_%s'%(thresh_1)
			df1[column_1] = np.asarray(ratio_1)
			# df1['ratio_0.1'] = np.asarray(ratio_2)
			df1[column_2] = np.asarray(ratio_2)
			field_query = field_query + [column_1,column_2]

		if flag_access>0:
			column_3 = 'max_accessibility_score'
			df1[column_3] = peak_read.max(axis=0)
			field_query += [column_3]

		if save_mode>0:
			if output_filename=='':
				if filename_annot=='':
					# filename_annot = select_config['filename_save_annot_pre1']
					filename_annot = select_config['filename_annot_save_default']
				output_filename = '%s/test_peak_query_basic_1.%s.1.txt'%(output_file_path,filename_annot)
			print('field_query: ',field_query)
			print('df1 ',df1.shape)
			df_query1 = df1.loc[:,field_query]
			df_query1.to_csv(output_filename,sep='\t',float_format='%.6f')

		# t_value_1 = utility_1.test_stat_1(df1,quantile_vec=quantile_vec_1)
		column_id_query = ['ratio']
		quantile_vec_1 = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
		t_value_1 = utility_1.test_stat_1(df1[column_id_query],quantile_vec=quantile_vec_1)
		ratio1 = t_value_1
		print('df1, ratio1 ',df1.shape,t_value_1)

		return df1, ratio1

	## peak attribute query: peak open ratio query
	def test_peak_access_query_basic_2(self,atac_meta_ad=[],peak_set=None, low_dim_embedding='X_svd', pval_cutoff=1e-2,read_len=147,n_neighbors=3,bin_size=5000,n_jobs=1,
											save_mode=1,output_file_path='',output_filename='',filename_save_annot='',select_config={}):

		if len(atac_meta_ad)==0:
			atac_meta_ad = self.atac_meta_ad

		open_peaks = self._determine_metacell_open_peaks(atac_meta_ad,peak_set=None, low_dim_embedding=low_dim_embedding,
															pval_cutoff=pval_cutoff,read_len=147,n_neighbors=n_neighbors,
															bin_size=bin_size,n_jobs=1)
		peak_loc_ori = atac_meta_ad.var_names
		sample_num = atac_meta_ad.shape[0]
		df_open_peaks = open_peaks
		peak_num1 = df_open_peaks.sum(axis=0)
		ratio_1 = peak_num1/sample_num
		df1 = pd.DataFrame(index=peak_loc_ori,columns=['ratio'],data=np.asarray(ratio_1))

		# column_id1 = ['ratio']
		# quantile_vec_1 = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
		# t_value_1 = utility_1.test_stat_1(df1[column_id1],quantile_vec=quantile_vec_1)
		# print('df1, ratio: ',df1.shape,t_value_1)

		if save_mode>0:
			filename_annot = filename_save_annot
			if output_filename=='':
				if filename_annot=='':
					filename_annot = select_config['filename_annot_save_default']
				output_filename = '%s/test_peak_query_basic.%s.txt'%(output_file_path,filename_annot)
			
			df1.to_csv(output_filename,sep='\t',float_format='%.6f')

			if 'output_filename_open_peaks' in select_config:
				output_filename_1 = select_config['output_filename_open_peaks']
				open_peaks.to_csv(output_filename_1,sep='\t',float_format='%d')

			if 'output_filename_nbrs_atac' in select_config:
				output_filename_2 = select_config['output_filename_nbrs_atac']
				meta_nbrs = self.select_config['meta_nbrs_atac']
				meta_nbrs.to_csv(output_filename_2,sep='\t')

		return atac_meta_ad, open_peaks

	## query the set of peaks that are open in each metacell
	# from SEACells
	def _determine_metacell_open_peaks(self,atac_meta_ad,peak_set=None,low_dim_embedding='X_svd',
											pval_cutoff=1e-2,read_len=147,n_neighbors=3,bin_size=5000,n_jobs=1):
		"""
		Determine the set of peaks that are open in each metacell
		:param atac_meta_ad: (Anndata) ATAC metacell Anndata created using `prepare_multiome_anndata`
		:param peak_set: (pd.Series) Subset of peaks to test. All peaks are tested by default
		:param low_dim_embedding: (str) `atac_meta_ad.obsm` field for nearest neighbor computation
		:param p_val_cutoff: (float) Nominal p-value cutoff for open peaks
		:param read_len: (int) Fragment length
		:param n_jobs: (int) number of jobs for parallel processing

		:atac_meta_ad is modified inplace with `.obsm['OpenPeaks']` indicating the set of open peaks in each metacell
		"""
		from sklearn.neighbors import NearestNeighbors
		from scipy.stats import poisson, multinomial

		# Effective genome length for background computaiton
		# eff_genome_length = atac_meta_ad.shape[1] * 5000
		# bin_size = 500
		eff_genome_length = atac_meta_ad.shape[1] * bin_size

		# Set up container
		if peak_set is None:
			peak_set = atac_meta_ad.var_names
		open_peaks = pd.DataFrame(0, index=atac_meta_ad.obs_names, columns=peak_set)

		# Metacell neighbors
		nbrs = NearestNeighbors(n_neighbors=n_neighbors)
		nbrs.fit(atac_meta_ad.obsm[low_dim_embedding])
		meta_nbrs = pd.DataFrame(atac_meta_ad.obs_names.values[nbrs.kneighbors(atac_meta_ad.obsm[low_dim_embedding])[1]],
								 index=atac_meta_ad.obs_names)

		self.select_config.update({'meta_nbrs_atac':meta_nbrs})

		for m in tqdm(open_peaks.index):
			# Boost using local neighbors
			frag_counts = np.ravel(
				atac_meta_ad[meta_nbrs.loc[m, :].values, :][:, peak_set].X.sum(axis=0))
			frag_distr = frag_counts / np.sum(frag_counts).astype(np.float64)

			# Multinomial distribution
			while not 0 < np.sum(frag_distr) < 1 - 1e-5:
				frag_distr = np.absolute(frag_distr - np.finfo(np.float32).epsneg)
			# Sample from multinomial distribution
			frag_counts = multinomial.rvs(np.percentile(
				atac_meta_ad.obs['n_counts'], 100), frag_distr)

			# Compute background poisson distribution
			total_frags = frag_counts.sum()
			glambda = (read_len * total_frags) / eff_genome_length

			# Significant peaks
			cutoff = pval_cutoff / np.sum(frag_counts > 0)
			open_peaks.loc[m, frag_counts >= poisson.ppf(1 - cutoff, glambda)] = 1

		# Update ATAC Metadata object
		atac_meta_ad.layers['OpenPeaks'] = open_peaks.values

		return open_peaks

	## query open peaks
	# query open peaks in the metacells
	def test_peak_access_query_basic_pre1(self,adata=[],n_neighbors=3,low_dim_embedding='X_svd',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		# query the ratio of open peak loci
		flag_query_peak_ratio=1
		if flag_query_peak_ratio>0:
			# atac_meta_ad = self.atac_meta_ad
			atac_meta_ad = adata
			print('atac_meta_ad \n',atac_meta_ad)

			# input_file_path1 = self.save_path_1
			if output_file_path=='':
				file_save_path2 = select_config['data_path_save']
				output_file_path = file_save_path2
			
			type_id_query=1
			filename_prefix = filename_prefix_save
			filename_annot = filename_save_annot

			filename_annot2 = '%s.%d'%(filename_annot,(type_id_query+1))
			output_filename = '%s/test_peak_query_basic.%s.txt'%(output_file_path,filename_annot2)
			output_filename_peak_query = '%s/test_query_peak_access.%s.txt'%(output_file_path,filename_annot2)
			output_filename_nbrs_atac = '%s/test_query_meta_nbrs.%s.txt'%(output_file_path,filename_annot2)
			select_config.update({'output_filename_open_peaks':output_filename_peak_query,
									'output_filename_nbrs_atac':output_filename_nbrs_atac})

			if not(low_dim_embedding in atac_meta_ad.obsm):
				if type_id_query==0:
					print('the embedding not estimated: %s'%(low_dim_embedding))
					input_filename_atac = self.select_config['input_filename_atac']
					atac_ad_ori = sc.read_h5ad(input_filename_atac)
					print('atac_ad_ori: ',atac_ad_ori.shape)
					print(atac_ad_ori)
					
					input_filename = select_config['filename_rna_obs']
					df_obs_rna = pd.read_csv(input_filename,index_col=0,sep='\t')
					print('df_obs_rna: ',df_obs_rna.shape)
					
					sample_id_rna_ori = df_obs_rna.index
					sample_id_atac_ori = atac_ad_ori.obs_names
					# common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names,sort=False)
					common_cells = sample_id_atac_ori.intersection(sample_id_rna_ori,sort=False)
					atac_mod_ad = atac_ad_ori[common_cells,:]
					print('common_cells: %d'%(len(common_cells)))
					print('atac_mod_ad: ',atac_mod_ad.shape)
					atac_svd = atac_mod_ad.obsm['X_svd']
					print('atac_svd: ',atac_svd.shape)

					svd = pd.DataFrame(data=atac_mod_ad.obsm['X_svd'], index=atac_mod_ad.obs_names)
					
					SEACells_label = 'SEACell'
					df_obs_atac_ori = atac_ad_ori.obs
					df_var_atac_ori = atac_ad_ori.var
					output_file_path = input_file_path1

					df_obs_rna_1 = df_obs_rna.loc[common_cells,:]
					# atac_mod_ad.obs[SEACells_label] = np.asarray(df_obs_rna_1['Metacell'])
					# summ_svd = svd.groupby(atac_mod_ad.obs[SEACells_label]).mean()
					
					df_obs_atac_ori.loc[common_cells,SEACells_label] = np.asarray(df_obs_rna.loc[common_cells,'Metacell'])
					df_obs_atac_1 = df_obs_atac_ori.loc[common_cells,:]
					summ_svd = svd.groupby(df_obs_atac_1[SEACells_label]).mean()
					atac_meta_ad.obsm['X_svd'] = summ_svd.loc[atac_meta_ad.obs_names, :].values

					if save_mode>0:
						output_filename_1 = '%s/test_%s_atac.df_obs.txt'%(output_file_path,filename_prefix)
						output_filename_2 = '%s/test_%s_atac.df_var.txt'%(output_file_path,filename_prefix)
						
						df_obs_atac_ori.to_csv(output_filename_1,sep='\t')
						df_var_atac_ori.to_csv(output_filename_2,sep='\t')

						output_filename_1 = '%s/test_%s_atac.common.df_obs.txt'%(output_file_path,filename_prefix)
						output_filename_2 = '%s/test_%s_rna.common.df_obs.txt'%(output_file_path,filename_prefix)
						
						df_obs_atac_1.to_csv(output_filename_1,sep='\t')
						df_obs_rna_1.to_csv(output_filename_2,sep='\t')

				else:
					n_components = 100
					sc.tl.pca(atac_meta_ad,n_comps=n_components,zero_center=False,use_highly_variable=False)
					atac_meta_ad.obsm[low_dim_embedding] = atac_meta_ad.obsm['X_pca'].copy()
					atac_feature = atac_meta_ad.obsm[low_dim_embedding]
					print('atac_meta_ad\n',atac_meta_ad)
					print('atac_feature: ',atac_feature.shape)
					if save_mode>0:
						output_filename_1 = '%s/test_%s_meta_atac.normalize.h5ad'%(output_file_path,filename_prefix)
						atac_meta_ad.write(output_filename_1)

			pval_cutoff = 1e-2
			# n_neighbors = 3
			# bin_size = 500
			bin_size = 5000
			atac_meta_ad, open_peaks = self.test_peak_access_query_basic_2(atac_meta_ad=atac_meta_ad,
																			peak_set=None,
																			low_dim_embedding=low_dim_embedding,
																			pval_cutoff=pval_cutoff,
																			read_len=147,
																			n_neighbors=n_neighbors,
																			bin_size=bin_size,
																			n_jobs=1,
																			save_mode=1,
																			filename_annot='',
																			output_file_path=output_file_path,
																			output_filename=output_filename,
																			select_config=select_config)

			self.atac_meta_ad = atac_meta_ad

			return atac_meta_ad

	## load peak read and rna exprs data
	def test_motif_peak_estimate_load_1(self,peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],dict_motif_data=[],select_config={}):

		sample_id = rna_exprs.index
		rna_exprs_unscaled = rna_exprs_unscaled.loc[sample_id,:]
		peak_read = peak_read.loc[sample_id,:]
		meta_scaled_exprs = rna_exprs  # scaled metacell gene expression data
		meta_exprs_2 = rna_exprs_unscaled	# normalized and log-transformed gene expression data

		self.meta_scaled_exprs = meta_scaled_exprs
		self.meta_exprs_2 = meta_exprs_2
		self.peak_read = peak_read

		motif_data, motif_data_score = [],[]
		if len(dict_motif_data)>0:
			field_query = ['motif_data','motif_data_score','motif_query_name_expr']
			list1 = [dict_motif_data[field1] for field1 in field_query]

			motif_data, motif_data_score, motif_query_name_expr = list1
			motif_data = motif_data.loc[:,motif_query_name_expr]
			motif_data_score = motif_data_score.loc[:,motif_query_name_expr]
			self.motif_data = motif_data
			self.motif_data_score = motif_data_score
			self.motif_query_name_expr = motif_query_name_expr

		return motif_data, motif_data_score

	## query gene set
	def test_query_gene_1(self,df_annot=[],highly_variable=True,celltype_vec_query=[],input_filename='',type_query=0,verbose=0,select_config={}):

		if len(df_annot)==0:
			df_var_rna = self.rna_meta_ad.var
			df_annot = df_var_rna

		df_annot = df_annot.sort_values(by=['dispersions_norm'],ascending=False)
		gene_name_query_expr = df_annot.index
		if type_query==0:
			# load from gene annotation
			if highly_variable==True:
				## query highly variable genes
				if 'highly_variable' in df_annot:
					gene_highly_variable = gene_name_query_expr[df_annot['highly_variable']==True]
					gene_group2 = gene_name_query_expr[df_annot['highly_variable']==False]
					gene_num1, gene_num2 = len(gene_highly_variable), len(gene_group2)
					if verbose>0:
						print('gene_highly_variable: %d, gene_group2: %d'%(gene_num1,gene_num2))
						print(gene_highly_variable[0:10])
				else:
					## TODO: perform estimation for highly variables genes
					pass

				gene_query_vec_pre1 = gene_highly_variable
			else:
				## query genome-wide genes
				gene_query_vec_pre1 = gene_name_query_expr

			gene_query_vec = gene_query_vec_pre1
		else:
			# load estimated genes with cell type-specific expressions
			df1 = pd.read_csv(input_filename,index_col=0,sep='\t')
			list1 = []
			for celltype_vec_query1 in celltype_vec_query:
				celltype1, celltype2 = celltype_vec_query1[0:2]
				id1 = (df1['group1']==celltype1)&(df1['group2']==celltype2)
				id2 = (df1['group1']==celltype2)&(df1['group2']==celltype1)
				id_1 = (id1|id2)

				df2 = df1.loc[id_1,:]
				gene_query1 = df2.index.unique()
				list1.extend(gene_query1)
			gene_query_vec = np.asarray(list1)
			gene_query_vec = pd.Index(gene_query_vec).unique()

		return gene_query_vec

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_pre1(self,gene_query_vec=[],peak_loc_query=[],df_gene_peak_query=[],df_gene_peak_compute_1=[],df_gene_peak_compute_2=[],atac_ad=[],rna_exprs=[],flag_computation_vec=[1,3],highly_variable=False,recompute=0,annot_mode=1,save_mode=1,filename_prefix_save='',output_filename='',save_file_path='',verbose=0,select_config={}):

		data_path = select_config['data_path_save']
		
		peak_bg_num_ori = 100
		# input_filename_peak = '%s/test_peak_GC.1.bed'%(input_file_path)
		# input_filename_bg = '%s/test_peak_read.%s.normalize.bg.%d.1.csv'%(input_file_path,data_file_type,peak_bg_num_ori)
		
		input_filename_peak, input_filename_bg = select_config['input_filename_peak'], select_config['input_filename_bg']
		if os.path.exists(input_filename_bg)==False:
			print('the file does not exist: %s'%(input_filename_bg))
			# return

		peak_bg_num = 100
		# interval_peak_corr = 500
		# interval_local_peak_corr = -1
		interval_peak_corr = 10
		interval_local_peak_corr = -1

		list1 = [peak_bg_num,interval_peak_corr,interval_local_peak_corr]
		field_query = ['peak_bg_num','interval_peak_corr','interval_local_peak_corr']

		for i1 in range(3):
			field_id = field_query[i1]
			if (field_id in select_config):
				list1[i1] = select_config[field_id]
		peak_bg_num,interval_peak_corr,interval_local_peak_corr = list1

		flag_correlation_1 = select_config['flag_correlation_1']

		save_file_path = select_config['data_path_save_local']
		input_file_path = save_file_path
		output_file_path = save_file_path

		peak_distance_thresh = select_config['peak_distance_thresh']

		df_gene_peak_query_thresh2 = []
		if flag_correlation_1>0:
			interval_peak_corr = select_config['interval_peak_corr']
			interval_local_peak_corr = select_config['interval_local_peak_corr']
			
			input_filename_pre1, input_filename_pre2 = select_config['input_filename_pre1'], select_config['input_filename_pre2']
			flag_compute = 0
			if (os.path.exists(input_filename_pre1)==False) or (recompute>0):
				print('the file to be prepared: %s'%(input_filename_pre1))
				flag_compute = 1
			else:
				if (os.path.exists(input_filename_pre1)==True):
					print('the file exists: %s'%(input_filename_pre1))

			print('flag_computation_vec: ',flag_computation_vec)
			compute_mode_1 = 1
			compute_mode_2 = 3
			# compute_mode_2 = 1
			flag_compute_fg = (compute_mode_1 in flag_computation_vec)
			flag_compute_bg = (compute_mode_2 in flag_computation_vec)
			flag_compute = (flag_compute|flag_compute_bg)

			flag_thresh1 = 1
			if 'flag_correlation_thresh1' in select_config:
				flag_thresh1 = select_config['flag_correlation_thresh1']

			column_idvec = ['peak_id','gene_id']
			dict_query1 = dict()
			if flag_compute>0:
				df_gene_peak_query_thresh1 = df_gene_peak_compute_2 # pre-selected peak-gene links with peak accessibility-gene expression correlation above threshold
				print('flag_computation_vec: ',flag_computation_vec)

				for flag_computation_1 in flag_computation_vec:
					select_config.update({'flag_computation_1':flag_computation_1})

					if flag_computation_1==1:
						if len(df_gene_peak_query)==0:
							df_gene_peak_query = self.df_gene_peak_distance
					
					elif flag_computation_1==2:
						if len(df_gene_peak_query)==0:
							df_gene_peak_query = df_gene_peak_compute_1
					
					elif flag_computation_1==3:
						if len(df_gene_peak_query_thresh1)==0:
							df_gene_peak_query = pd.read_csv(input_filename_pre2,index_col=0,sep='\t')
						else:
							df_gene_peak_query = df_gene_peak_query_thresh1

					compute_mode = flag_computation_1
					if flag_computation_1 in [1,3]:
						df_gene_peak_1 = self.test_gene_peak_query_correlation_pre1_compute(gene_query_vec=gene_query_vec,
																							gene_query_vec_2=[],
																							peak_distance_thresh=peak_distance_thresh,
																							df_gene_peak_query=df_gene_peak_query,
																							df_gene_peak_compute=df_gene_peak_compute_1,
																							peak_loc_query=[],
																							atac_ad=atac_ad,
																							rna_exprs=rna_exprs,
																							flag_thresh1=0,
																							highly_variable=highly_variable,
																							save_mode=1,
																							save_file_path=save_file_path,
																							filename_prefix_save='',
																							output_filename='',
																							annot_mode=1,
																							verbose=verbose,
																							select_config=select_config)
						dict_query1.update({compute_mode:df_gene_peak_1})

					# load estimated peak-gene correlations and perform pre-selection
					# compute_mode: 1, calculate peak-gene correlation; 2, perform peak-gene selection by threshold; 3, calculate peak-gene correlation for background peaks
					# compute_mode = flag_computation_1
					if flag_thresh1>0:
						if compute_mode in [1,2]:
							start = time.time()

							df_peak_query = self.df_gene_peak_distance
							# correlation_type = 'spearmanr'
							correlation_type = select_config['correlation_type_1']
							input_filename_pre2 = select_config['input_filename_pre2']
							output_filename = input_filename_pre2
							df_gene_peak_query = self.test_gene_peak_query_correlation_thresh1(gene_query_vec=gene_query_vec,
																								df_gene_peak_query=df_gene_peak_1,
																								df_peak_annot=df_peak_query,
																								correlation_type=correlation_type,
																								save_mode=1,
																								output_filename=output_filename,
																								select_config=select_config)
							stop = time.time()
							print('pre-selection of peak-gene link query used %.5fs'%(stop-start))

				# combine the foreground and background peak-gene correlation estimation
				if (flag_compute_fg>0) and (flag_compute_bg>0):
					coherent_mode = 1
					column_vec_query = ['pval1']
					df_gene_peak_pre1 = dict_query1[compute_mode_1]
					df_gene_peak_bg = dict_query1[compute_mode_2]
					save_mode_2 = 1
					input_filename_pre2 = select_config['input_filename_pre2']
					output_filename = input_filename_pre2
					
					print('peak-gene link, peak-gene link background: ',df_gene_peak_pre1.shape,df_gene_peak_bg.shape)
					print(df_gene_peak_pre1[0:2])
					print(df_gene_peak_bg[0:2])
					
					df_gene_peak_query_1 = self.test_query_feature_correlation_merge_2(df_gene_peak_query=df_gene_peak_pre1,
																						df_gene_peak_bg=df_gene_peak_bg,
																						filename_list=[],column_idvec=column_idvec,
																						column_vec_query=column_vec_query,
																						flag_combine=1,compute_mode=-1,
																						coherent_mode=coherent_mode,index_col=0,
																						save_mode=1,output_path='',output_filename='',
																						verbose=verbose,select_config=select_config)

			data_file_type_query = select_config['data_file_type_query']
			input_filename_pre2 = select_config['input_filename_pre2']
			file_path_save_local = select_config['data_path_save_local']
			filename_prefix_default = select_config['filename_prefix_default']

			flag_iteration_bg = 1
			if flag_iteration_bg>0:
				input_file_path_2 = '%s/data2'%(file_path_save_local)

				gene_query_num1 = len(gene_query_vec)
				interval = 500
				iter_num = int(np.ceil(gene_query_num1/interval))
				filename_list_bg = []
				for i1 in range(iter_num):
					start_id1 = interval*i1
					start_id2 = np.min([interval*(i1+1),gene_query_num1])
					
					input_filename = '%s/%s.pre1_bg_%d_%d.combine.thresh1.1.txt'%(input_file_path_2,filename_prefix_default,start_id1,start_id2)
					filename_list_bg.append(input_filename)

					if os.path.exists(input_filename)==False:
						print('the file does not exist: %s'%(input_filename))
						return

				select_config.update({'filename_list_bg':filename_list_bg})

			flag_combine_empirical_1 = 0
			if 'flag_combine_empirical_1' in select_config:
				flag_combine_empirical_1 = select_config['flag_combine_empirical_1']

			if flag_combine_empirical_1>0:
				# combine the foreground and background peak-gene correlation estimation
				column_vec_query = ['pval1']
				coherent_mode = 0
				filename_list = [input_filename_pre2] + filename_list_bg
				output_filename = input_filename_pre2
				
				#  combine the foreground and background peak-gene correlation estimation
				print('query emprical p-value estimation')
				print('filename_list_bg: ',len(filename_list_bg))
				if verbose>0:
					print(filename_list_bg[0:2])
				
				self.test_query_feature_correlation_merge_2(df_gene_peak_query=[],df_gene_peak_bg=[],
															filename_list=filename_list,column_idvec=column_idvec,
															column_vec_query=column_vec_query,
															flag_combine=1,compute_mode=-1,
															coherent_mode=coherent_mode,index_col=0,
															save_mode=1,output_path='',output_filename=output_filename,
															verbose=verbose,select_config=select_config)

			flag_merge_1=0
			if 'flag_merge_1' in select_config:
				flag_merge_1 = select_config['flag_merge_1']

			if flag_merge_1>0:
				# combine the peak-gene correlation estimation from different runs
				filename_list = select_config['filename_list_bg']
				output_file_path = input_file_path_2
				output_filename = '%s/%s.pre1_bg.combine.thresh1.1.txt'%(output_file_path,filename_prefix_default)
				compute_mode_query = 3
				self.test_query_feature_correlation_merge_1(df_gene_peak_query=[],filename_list=[],flag_combine=1,compute_mode=compute_mode_query,index_col=0,
															save_mode=1,output_path=output_file_path,output_filename=output_filename,verbose=verbose,select_config=select_config)

		
		
		## add columns to the original peak-link query dataframe: empirical p-values for subset of peak-gene link query
		# save_file_path = select_config['data_path_save']
		input_file_path = save_file_path
		output_file_path = save_file_path
		filename_save_annot_pre1 = select_config['filename_save_annot_pre1']
		
		flag_combine_empirical_2 = select_config['flag_combine_empirical']
		if flag_combine_empirical_2>0:

			output_filename = '%s/df_gene_peak_distance_annot.%s.txt'%(output_file_path,filename_save_annot_pre1)
			highly_variable_thresh = select_config['highly_variable_thresh']
			# overwrite_1 = False
			query_mode = 2
			# query_mode = 0
			
			flag_query = 1
			if (os.path.exists(output_filename)==True):
				print('the file exists: %s'%(output_filename))
				if query_mode in [2]:
					input_filename = output_filename
					df_gene_peak_distance = pd.read_csv(input_filename,index_col=0,sep='\t') # add to the existing file
				elif query_mode==0:
					flag_query = 0
			else:
				query_mode = 1

			if flag_query>0:
				load_mode = 0
				if query_mode in [1]:
					df_gene_peak_distance = self.df_gene_peak_distance
				
				df_gene_peak_query_compute1, df_gene_peak_query_thresh1 = self.test_gene_peak_query_correlation_pre1_combine(gene_query_vec=[],peak_distance_thresh=peak_distance_thresh,
																																df_gene_peak_distance=df_gene_peak_distance,
																																highly_variable=highly_variable,highly_variable_thresh=highly_variable_thresh,
																																load_mode=load_mode,
																																save_mode=1,filename_prefix_save='',input_file_path=input_file_path,output_filename=output_filename,
																																save_file_path=output_file_path,
																																select_config=select_config)

		flag_query_thresh2 = select_config['flag_query_thresh2']
		## pre-select peak-gene link query by empirical p-values estimated from background peaks matching GC content and average chromatin accessibility
		if flag_query_thresh2>0:
			
			input_filename_pre2 = select_config['input_filename_pre2']
			input_filename = input_filename_pre2
			
			output_filename_1 = input_filename
			# output_filename_2 = '%s/%s.combine.thresh2.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
			output_filename_2 = select_config['filename_save_thresh2']
			
			overwrite_1 = False
			if 'overwrite_thresh2' in select_config:
				overwrite_1 = select_config['overwrite_thresh2']
			
			flag_query = 1
			if os.path.exists(output_filename_2)==True:
				print('the file exists: %s'%(output_filename_2))
				if overwrite_1==False:
					flag_query = 0

			if flag_query>0:
				df_gene_peak_query_thresh2, df_gene_peak_query = self.test_gene_peak_query_correlation_pre1_select_1(gene_query_vec=[],
																														df_gene_peak_query=[],
																														peak_loc_query=[],
																														input_filename=input_filename,
																														highly_variable=highly_variable,
																														peak_distance_thresh=peak_distance_thresh,
																														save_mode=1,
																														filename_prefix_save='',
																														output_filename_1=output_filename_1,
																														output_filename_2=output_filename_2,
																														save_file_path=save_file_path,
																														verbose=verbose,
																														select_config=select_config)
			else:
				input_filename_1 = output_filename_1
				input_filename_2 = output_filename_2
				input_filename_list1 = [input_filename_1,input_filename_2]
				list1 = [pd.read_csv(input_filename,sep='\t') for input_filename in input_filename_list1]
				df_gene_peak_query, df_gene_peak_query_thresh2 = list1

			self.df_gene_peak_query_thresh2 = df_gene_peak_query_thresh2

		return df_gene_peak_query_thresh2, df_gene_peak_query

	## peak-gene_link
	def test_query_feature_correlation_thresh1(self,save_mode=1,verbose=0,select_config={}):

		input_filename_pre2 = select_config['input_filename_pre2']
		output_filename = input_filename_pre2
		flag_query1=1
		if os.path.exists(input_filename_pre2)==True:
			print('the file exists: %s'%(input_filename_pre2))
			if recompute==0:
				df_gene_peak_query = pd.read_csv(input_filename_pre2,index_col=0,sep='\t')
				flag_query2=0

		# df_gene_peak_query_1 = df_gene_peak_query
		df_gene_peak_query_1 = df_gene_peak_compute_1
		if flag_query1>0:
			# print('the file to be prepared: %s'%(input_filename_pre2))
			print('load estimated peak-gene correlations and perform pre-selection ')
			
			column_idvec = ['gene_id','peak_id']
			column_id1, column_id2 = column_idvec[0:2]
			
			if (len(df_gene_peak_query_1)==0):
				input_filename_pre1 = select_config['input_filename_pre1']
				df_gene_peak_query_1 = pd.read_csv(input_filename_pre1,index_col=0,sep='\t')
				
				# column_id1 = 'gene_id'
				if not (column_id1 in df_gene_peak_query_1.columns):
					df_gene_peak_query_1[column_id1] = np.asarray(df_gene_peak_query_1.index)
					
			print('peak-gene link: ',df_gene_peak_query_1.shape)
			print(df_gene_peak_query_1[0:2])

			# df_gene_peak_query_compute1 = df_gene_peak_query
			if len(gene_query_vec_2)==0:
				gene_query_vec_2 = gene_query_vec
			
			start = time.time()
			# df_peak_query = df_gene_peak_query_ori
			df_peak_query = self.df_gene_peak_distance
			# correlation_type = 'spearmanr'
			correlation_type = select_config['correlation_type_1']
			df_gene_peak_query = self.test_gene_peak_query_correlation_thresh1(gene_query_vec=gene_query_vec_2,
																				df_gene_peak_query=df_gene_peak_query_1,
																				df_peak_annot=df_peak_query,
																				correlation_type=correlation_type,
																				save_mode=1,
																				output_filename=output_filename,
																				select_config=select_config)
			stop = time.time()

			print('pre-selection of peak-gene link query used %.5fs'%(stop-start))

			return df_gene_peak_query
		
	## combine the peak-gene correlation estimation
	def test_query_feature_correlation_merge_1(self,df_gene_peak_query=[],filename_list=[],flag_combine=1,compute_mode=-1,index_col=0,save_mode=0,output_path='',output_filename='',verbose=0,select_config={}):

		flag_query_1 = flag_combine
		df_gene_peak_query_1 = []
		if flag_query_1>0:
			input_filename_list1 = filename_list
			if len(filename_list)==0:
				field_query_vec = ['filename_list_pre1','filename_list_bg']
				id1 = int((compute_mode-1)/2)
				field_query = field_query_vec[id1]

				if field_query in select_config:
					input_filename_list1 = select_config[field_query]
					print('load estimations from the file ',len(input_filename_list1))
					if verbose>0:
						print(input_filename_list1[0:2])

			if len(input_filename_list1)>0:
				df_gene_peak_query_1 = utility_1.test_file_merge_1(input_filename_list1,index_col=index_col,header=0,float_format=-1,
																	save_mode=1,verbose=verbose,output_filename=output_filename)

				if (save_mode>0) and (output_filename!=''):
					df_gene_peak_query_1.to_csv(output_filename,sep='\t')
			else:
				print('please perform peak accessibility-gene expression correlation estimation or load estimated correlations')

		return df_gene_peak_query_1

	## combine the foreground and background peak-gene correlation estimation
	def test_query_feature_correlation_merge_2(self,df_gene_peak_query=[],df_gene_peak_bg=[],filename_list=[],column_idvec=['peak_id','gene_id'],column_vec_query=[],flag_combine=1,compute_mode=-1,coherent_mode=1,index_col=0,save_mode=0,output_path='',output_filename='',verbose=0,select_config={}):

		flag_query_1 = flag_combine
		if flag_query_1>0:
			load_mode = (len(df_gene_peak_query)>0)&(len(df_gene_peak_bg)>0)

			if len(column_vec_query)==0:
				column_vec = ['pval1']
			else:
				column_vec = column_vec_query
			
			if load_mode>0:
				df_list = [df_gene_peak_query,df_gene_peak_bg]
				column_vec_1 = [column_vec]
				df_gene_peak_query_1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec,df_list=df_list,
																		column_vec=column_vec_1,reset_index=False)
			else:
				if coherent_mode==1:
					if len(filename_list)>0:
						input_filename_pre2, input_filename_pre2_bg = filename_list[0:2]
					else:
						input_filename_pre2 = select_config['input_filename_pre2']
						input_filename_pre2_bg = select_config['input_filename_pre2_bg']

					if os.path.exists(input_filename_pre2)==False:
						print('the file does not exist: %s'%(input_filename_pre2))
						return

					if os.path.exists(input_filename_pre2_bg)==False:
						print('the file does not exist: %s'%(input_filename_pre2_bg))
						return

					input_filename_list = [input_filename_pre2,filename_bg]
					# copy columns of one dataframe to another dataframe
					column_vec_1 = [column_vec]
					# df_gene_peak_query_1 = utility_1.test_column_query_1(input_filename_list,id_column=column_idvec,df_list=[],
					# 														column_vec=column_vec_1,reset_index=False)

				elif coherent_mode==0:
					if len(filename_list)==0:
						input_filename_pre2 = select_config['input_filename_pre2']
						filename_list_bg = select_config['filename_list_bg']
						input_filename_list = [input_filename_pre2]+filename_list_bg
					else:
						input_filename_pre2 = filename_list[0]
						filename_list_bg = filename_list[1:]
						input_filename_list = filename_list
					
					file_num1 = len(filename_list_bg)
					column_vec_1 = [column_vec]*file_num1

				elif coherent_mode==2:
					filename_list_pre2 = select_config['filename_list_pre2']
					filename_list_bg = select_config['filename_list_bg']
					
					file_num1 = len(filename_list_pre2)
					column_vec_1 = [column_vec]
					list_query1 = []
					for i1 in range(file_num1):
						filename_1 = filename_list_pre2[i1]
						filename_2 = filename_list_bg[i1]
						input_filename_list1 = [filename_1,filename_2]
						df_gene_peak_query = utility_1.test_column_query_1(input_filename_list1,id_column=column_idvec,df_list=[],
																			column_vec=column_vec_1,reset_index=False)
						list_query1.append(df_gene_peak_query)

					df_gene_peak_query_1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)

				if coherent_mode in [0,1]:
					df_gene_peak_query_1 = utility_1.test_column_query_1(input_filename_list,id_column=column_idvec,df_list=[],
																			column_vec=column_vec_1,reset_index=False)

			if save_mode>0:
				# output_filename_1 = input_filename_pre2
				df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1['gene_id'])
				df_gene_peak_query_1.to_csv(output_filename,sep='\t')

			return df_gene_peak_query_1

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_pre1_compute(self,gene_query_vec=[],gene_query_vec_2=[],peak_distance_thresh=500,
														df_gene_peak_query=[],df_gene_peak_compute=[],peak_loc_query=[],atac_ad=[],rna_exprs=[],
														flag_thresh1=1,highly_variable=False,interval_peak_corr=50,interval_local_peak_corr=10,
														save_mode=1,save_file_path='',filename_prefix_save='',
														output_filename='',annot_mode=1,verbose=0,select_config={}):

		gene_query_vec_pre1 = gene_query_vec
		gene_query_num_1 = len(gene_query_vec_pre1)
		print('target gene set: %d '%(gene_query_num_1))
		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		recompute=0
		if 'recompute' in select_config:
			recompute = select_config['recompute']

		if len(atac_ad)==0:
			atac_ad = self.atac_meta_ad
			print('load atac_ad ',atac_ad.shape)
		if len(rna_exprs)==0:
			rna_exprs = self.meta_scaled_exprs
			print('load rna_exprs ',rna_exprs.shape)

		filename_prefix_1 = select_config['filename_prefix_default']
		filename_prefix_save_1 = select_config['filename_prefix_save_default']

		iter_mode = 0
		if (query_id1>=0) and (query_id2>query_id1):
			iter_mode = 1  # query gene subset
			start_id1 = query_id1
			start_id2 = np.min([query_id2,gene_query_num_1])
			gene_query_vec = gene_query_vec_pre1[start_id1:start_id2]
			filename_prefix_save = '%s_%d_%d'%(filename_prefix_save_1,start_id1,start_id2)
			filename_prefix_save_bg = '%s_bg_%d_%d'%(filename_prefix_save_1,start_id1,start_id2)
		elif query_id1<-1:
			iter_mode = 2  # combine peak-gene estimation from different runs
		else:
			gene_query_vec = gene_query_vec_pre1
			filename_prefix_save = filename_prefix_save_1
			filename_prefix_save_bg = '%s_bg'%(filename_prefix_save_1)
			start_id1 = 0
			start_id2 = gene_query_num_1
		select_config.update({'iter_mode':iter_mode})

		filename_prefix = '%s.%s'%(filename_prefix_1,filename_prefix_save_1)
		filename_prefix_local = '%s.%s'%(filename_prefix_1,filename_prefix_save)
		filename_prefix_bg = '%s.%s_bg'%(filename_prefix_1,filename_prefix_save_1)
		filename_prefix_bg_local = '%s.%s'%(filename_prefix_1,filename_prefix_save_bg)
		filename_annot1 = select_config['filename_annot_default']
		# input_filename_pre2_bg = '%s/data2/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix_bg,filename_annot1)
		input_filename_pre2_bg = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix_bg,filename_annot1)
		filename_pre2_bg_local = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix_bg_local,filename_annot1)

		select_config.update({'filename_prefix_peak_gene':filename_prefix,
								'filename_prefix_bg_peak_gene':filename_prefix_bg,
								'filename_prefix_local':filename_prefix_local,
								'filename_prefix_bg_local':filename_prefix_bg_local,
								'input_filename_pre2_bg':input_filename_pre2_bg})

		df_gene_peak_query_ori = df_gene_peak_query.copy()
		df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
		gene_name_query_1 = df_gene_peak_query['gene_id'].unique()
		gene_query_vec_ori = gene_query_vec
		gene_query_vec = pd.Index(gene_query_vec_ori).intersection(gene_name_query_1,sort=False)
		
		gene_query_vec_2 = gene_query_vec
		gene_query_num_1 = len(gene_name_query_1)
		gene_query_num_ori = len(gene_query_vec_ori)
		gene_query_num = len(gene_query_vec)

		df_gene_peak_query = df_gene_peak_query.loc[gene_query_vec,:]
		# print('peak accessibility-gene expr correlation estimation ')
		
		gene_query_vec_bg = gene_query_vec
		# gene_query_num = len(gene_query_vec)
		gene_query_num_bg = len(gene_query_vec_bg)
		# if verbose>0:
		# 	print('gene_query_vec:%d, gene_query_vec_bg:%d, query_id1:%d, query_id2:%d, start_id1:%d, start_id2:%d'%(len(gene_query_vec),len(gene_query_vec_bg),query_id1,query_id2,start_id1,start_id2))
		if verbose>0:
			print('gene_query_vec_ori ',gene_query_num_ori,gene_query_vec_ori[0:2])
			print('gene_name_query_1 ',gene_query_num_1,gene_name_query_1[0:2])
			print('gene_query_vec ',gene_query_num,gene_query_vec[0:2])
			print('query_id1:%d, query_id2:%d, start_id1:%d, start_id2:%d'%(query_id1,query_id2,start_id1,start_id2))
			# print('gene_query_vec:%d, gene_query_vec_bg:%d, query_id1:%d, query_id2:%d, start_id1:%d, start_id2:%d'%(len(gene_query_vec),len(gene_query_vec_bg),query_id1,query_id2,start_id1,start_id2))

		interval_peak_corr, interval_local_peak_corr = select_config['interval_peak_corr'], select_config['interval_local_peak_corr']
		interval_peak_corr_bg, interval_local_peak_corr_bg = interval_peak_corr, interval_local_peak_corr

		flag_combine_bg=1
		# flag_combine_bg=0
		# recompute_1=0
		flag_corr_, method_type, type_id_1 = 1, 1, 1 # correlation without estimating emprical p-value; correlation and p-value; spearmanr
		select_config.update({'flag_corr_':flag_corr_,
								'method_type_correlation':method_type,
								'type_id_correlation':type_id_1})

		select_config.update({'gene_query_vec_bg':gene_query_vec_bg,
								'flag_combine_bg':flag_combine_bg,
								'interval_peak_corr_bg':interval_peak_corr_bg,
								'interval_local_peak_corr_bg':interval_peak_corr_bg})

		flag_computation_1 = select_config['flag_computation_1']
		compute_mode = flag_computation_1
		# peak-gene correlation estimation
		df_gene_peak_compute_1 = df_gene_peak_compute
		# if flag_computation_1>0:
		if flag_computation_1 in [1,3]:
			df_gene_peak_query = self.test_gene_peak_query_correlation_unit1(gene_query_vec=gene_query_vec,gene_query_vec_2=[],peak_distance_thresh=500,
																				df_gene_peak_query=df_gene_peak_query,peak_loc_query=[],atac_ad=atac_ad,rna_exprs=rna_exprs,flag_computation_1=1,
																				highly_variable=False,interval_peak_corr=interval_peak_corr,interval_local_peak_corr=interval_local_peak_corr,
																				save_mode=1,save_file_path=save_file_path,filename_prefix_save='',
																				output_filename='',annot_mode=1,verbose=0,select_config=select_config)
			df_gene_peak_compute_1 = df_gene_peak_query

		coherent_mode = 0

		return df_gene_peak_query

	## gene-peak association query pre-selection by thresholding
	def test_gene_peak_query_correlation_thresh1(self,gene_query_vec=[],df_gene_peak_query=[],df_peak_annot=[],correlation_type='spearmanr',
													save_mode=1,output_filename='',float_format='%.5E',select_config={}):

		flag_query1=1
		if flag_query1>0:
			df_gene_peak_compute1 = df_gene_peak_query
			print('peak-gene link: ',df_gene_peak_compute1.shape)

			if len(gene_query_vec)>0:
				print('perform pre-selection for the gene subset: %d genes'%(len(gene_query_vec)))
				query_id_ori = df_gene_peak_query.index.copy()
				df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
				# df_gene_peak_query_compute1 = df_gene_peak_query.copy()
				
				gene_query_idvec = pd.Index(gene_query_vec).intersection(df_gene_peak_query.index,sort=False)
				print('gene_query_vec_2: %d, gene_query_idvec: %d'%(len(gene_query_vec),len(gene_query_idvec)))
				df_gene_peak_query = df_gene_peak_query.loc[gene_query_idvec,:]
			else:
				print('perform pre-selection for genes with estimated peak-gene correlations')
				# df_gene_peak_query_compute1 = df_gene_peak_query

			print('peak-gene link ',df_gene_peak_query.shape)

			## select gene-peak query above correlation thresholds for each distance range for empricial p-value calculation
			# tol_1 = 0.5
			# thresh_corr_distance = [[0,2,0],[2,50,0.001],[50,500,0.01],[500,1000,0.1],[1000,2050,0.15]]
			# thresh_corr_distance = [[0,50,0],[50,500,0.01],[500,1000,0.1],[1000,2050,0.15]]
			
			if 'thresh_corr_distance_1' in select_config:
				thresh_corr_distance = select_config['thresh_corr_distance_1']
			else:
				thresh_distance_1 = 50
				if 'thresh_distance_default_1' in select_config:
					thresh_distance_1 = select_config['thresh_distance_default_1'] # the distance threshold with which we retain the peaks without thresholds of correlation and p-value
				thresh_corr_distance = [[0,thresh_distance_1,0],
										[thresh_distance_1,500,0.01],
										[500,1000,0.1],
										[1000,2050,0.15]]

			if not ('distance' in df_gene_peak_query):
				field_query = ['distance']
				df_peak_annot = self.df_gene_peak_distance
				df_gene_peak_query = utility_1.test_gene_peak_query_attribute_1(df_gene_peak_query=df_gene_peak_query,
																					df_gene_peak_query_ref=df_peak_annot,
																					field_query=field_query,
																					column_name=[],
																					reset_index=False,
																					select_config=select_config)


			column_idvec = ['gene_id','peak_id']
			column_id1, column_id2 = column_idvec[0:2]
			df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_idvec)
			distance_abs = df_gene_peak_query['distance'].abs()
			list1 = []
			
			# correlation_type = 'spearmanr'
			column_id = correlation_type
			query_idvec = df_gene_peak_query.index
			for thresh_vec in thresh_corr_distance:
				constrain_1, constrain_2, thresh_corr_ = thresh_vec
				print(constrain_1, constrain_2, thresh_corr_)
				id1 = (distance_abs<constrain_2)&(distance_abs>=constrain_1)
				id2 = (df_gene_peak_query[column_id].abs()>thresh_corr_)
				query_id1 = query_idvec[id1&id2]
				list1.extend(query_id1)

			query_id_sub1 = pd.Index(list1).unique()
			df_gene_peak_query_ori = df_gene_peak_query.copy()
			df_gene_peak_query = df_gene_peak_query_ori.loc[query_id_sub1]
			df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
			df_gene_peak_query = df_gene_peak_query.sort_values(by=['gene_id','distance'],ascending=True)

			print('df_gene_peak_query_ori, df_gene_peak_query after pre-selection by correlation thresholds: ',df_gene_peak_query_ori.shape,df_gene_peak_query.shape)
			if (save_mode>0) and (output_filename!=''):
				# df_gene_peak_query.to_csv(output_filename,sep='\t',float_format='%.6E')
				df_gene_peak_query.to_csv(output_filename,sep='\t',float_format=float_format)

			return df_gene_peak_query

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_unit1(self,gene_query_vec=[],gene_query_vec_2=[],peak_distance_thresh=500,
													df_gene_peak_query=[],peak_loc_query=[],atac_ad=[],rna_exprs=[],flag_computation_1=1,
													highly_variable=False,interval_peak_corr=50,interval_local_peak_corr=10,
													save_mode=1,save_file_path='',filename_prefix_save='',
													output_filename='',annot_mode=1,verbose=0,select_config={}):

		# flag_correlation_1=1
		# flag_computation_1=0
		# flag_computation_1=1
		if ('flag_computation_1' in select_config):
			flag_computation_1 = select_config['flag_computation_1']

		recompute=0
		if 'recompute' in select_config:
			recompute = select_config['recompute']
		flag_query_1 = 1

		df_gene_peak_query_1 = []
		computation_mode_vec = [[0,0,0],[1,1,0],[0,1,0],[0,0,1]]
		# computation_mode = flag_computation_1
		compute_mode = flag_computation_1
		# flag_query1, flag_query2, background_query = computation_mode_vec[computation_mode]
		flag_query1, flag_query2, background_query = computation_mode_vec[compute_mode]

		iter_mode = select_config['iter_mode']
		field_query = ['input_filename_pre1','input_filename_pre1','input_filename_pre2_bg']

		if (flag_computation_1>0) or (flag_query_1>0):
			if compute_mode==3:
				# estimate empirical p-value using background peaks
				save_file_path2 = '%s/data2'%(save_file_path) # the directory to save the .npy file and estimation file for subsets
				filename_prefix_bg_local = select_config['filename_prefix_bg_local']
				filename_prefix_query = filename_prefix_bg_local
				if iter_mode==0:
					# filename_prefix_bg = select_config['filename_prefix_bg']
					# filename_prefix_query = filename_prefix_bg
					save_file_path1 = save_file_path
					# output_filename = field_query[compute_mode-1]
				else:
					# the parallel mode
					save_file_path1 = save_file_path2
					# output_filename = '%s/%s.combine.thresh1.1.txt'%(save_file_path1,filename_prefix_bg_local)

				if len(df_gene_peak_query)==0:
					input_filename_pre2 = select_config['input_filename_pre2']
					df_gene_peak_query = pd.read_csv(input_filename_pre2,index_col=0,sep='\t')
					print('load pre-selected peak-gene associations: %s'%(input_filename_pre2))
			else:
				# filename_prefix_local = select_config['filename_prefix_local']
				filename_prefix_local = select_config['filename_prefix_local']
				filename_prefix_query = filename_prefix_local
				save_file_path2 = '%s/data1'%(save_file_path)
				if iter_mode==0:
					# filename_prefix_query = filename_prefix_save
					save_file_path1 = save_file_path
					field_id1 = field_query[compute_mode-1]
					output_filename = select_config[field_id1]
				else:
					# filename_prefix_query = filename_prefix_local
					save_file_path1 = save_file_path2
					# output_filename = '%s/%s.combine.1.txt'%(save_file_path1,filename_prefix_local)

			if (compute_mode in [1,3]):
				if output_filename=='':
					if iter_mode==0:
						field_id1 = field_query[compute_mode-1]
						output_filename = select_config[field_id1]
					else:
						filename_annot_vec = ['combine',-1,'combine.thresh1']
						filename_annot_1 = filename_annot_vec[compute_mode-1]
						output_filename = '%s/%s.%s.1.txt'%(save_file_path1,filename_prefix_query,filename_annot_1)

		if flag_computation_1>0:
			## query the compuation mode
			# flag_query1: peak-gene correlation for foreground peaks
			# flag_query2: thresholding for foreground peaks
			# background_query: peak-gene correlation for background peaks
			# flag_query1, flag_query2, background_query = 1,1,0 # 0,0,1;
			# flag_query1, flag_query2, background_query = 0,0,1 # 1,1,0;

			select_config.update({'flag_query1':flag_query1,
									'flag_query2':flag_query2,
									'background_query':background_query})

			if os.path.exists(save_file_path1)==False:
				print('the directory does not exist: %s'%(save_file_path1))
				os.mkdir(save_file_path1)
			select_config.update({'save_file_path_local':save_file_path1})

			if compute_mode in [1,3]:
				print('peak accessibility-gene expr correlation estimation for peaks ')
				start = time.time()
				df_gene_peak_query_compute1, df_gene_peak_query = self.test_gene_peak_query_correlation_pre2(gene_query_vec=gene_query_vec,
																												gene_query_vec_2=gene_query_vec_2,
																												df_peak_query=df_gene_peak_query,
																												peak_dict=[],
																												atac_ad=atac_ad,
																												rna_exprs=rna_exprs,
																												compute_mode=compute_mode,
																												flag_query1=flag_query1,
																												flag_query2=flag_query2,
																												save_file_path=save_file_path,
																												save_file_path_local=save_file_path1,
																												interval_peak_corr=interval_peak_corr,
																												interval_local_peak_corr=interval_local_peak_corr,
																												recompute=recompute,
																												recompute_bg=0,
																												filename_prefix_save=filename_prefix_query,
																												output_filename=output_filename,
																												select_config=select_config)
				stop = time.time()
				print('peak accessibility-gene expr correlation estimation for peaks used %.5fs'%(stop-start))
				df_gene_peak_query_1 = df_gene_peak_query

		df_gene_peak_query_2 = df_gene_peak_query_1
		
		flag_query_1 = 0
		if flag_query_1>0:

			flag_combine_1=0
			if iter_mode==0:
				if 'gene_pre1_flag_combine_1' in select_config:
					flag_combine_1 = select_config['gene_pre1_flag_combine_1']

			df_gene_peak_query_2 = df_gene_peak_query_1
			
			# if (flag_combine_1>0) and (iter_mode!=0):
			if (flag_combine_1>0):
				if (iter_mode==0):
					import glob
					filename_annot1 = select_config['filename_save_annot_1']
					if compute_mode==1:
						filename_prefix = select_config['filename_prefix_peak_gene']
						if 'filename_list_pre1' in select_config:
							input_filename_list1 = select_config['filename_list_pre1']
						else:
							input_filename_list1 = glob.glob('%s/data1/%s_*.txt'%(save_file_path,filename_prefix))
						
					elif compute_mode==3:
						if 'filename_list_bg' in select_config:
							input_filename_list1 = select_config['filename_list_bg']
						else:
							filename_prefix_bg = select_config['filename_prefix_bg_peak_gene']
							input_filename_list1 = glob.glob('%s/data2/%s_*.txt'%(save_file_path,filename_prefix_bg))	
					else:
						pass

					if len(input_filename_list1)>0:
						df_gene_peak_query_1 = utility_1.test_file_merge_1(input_filename_list1,index_col=0,header=0,float_format=-1,
																			save_mode=1,verbose=verbose,output_filename=output_filename)
					else:
						print('please perform peak accessibility-gene expression correlation estimation or load estimated correlations')

			flag_combine_2=0
			if 'gene_pre1_flag_combine_2' in select_config:
				flag_combine_2 = select_config['gene_pre1_flag_combine_2']

			df_gene_peak_query_2 = df_gene_peak_query_1
			if flag_combine_2>0:
				# combine estimated empirical p-values with orignal p-values
				if compute_mode==3:
					input_filename_pre2 = select_config['input_filename_pre2']
					input_filename_pre2_bg = select_config['input_filename_pre2_bg']

					if os.path.exists(input_filename_pre2_bg)==False:
						print('the file does not exist: %s'%(input_filename_pre2_bg))
						filename_bg = output_filename
					else:
						filename_bg = input_filename_pre2_bg
					print('combine peak-gene correlation estimation: ',filename_bg)

					# input_filename_list = [input_filename_pre2,input_filename_pre2_bg]
					input_filename_list = [input_filename_pre2,filename_bg]
					
					# copy columns of one dataframe to another dataframe
					df_gene_peak_query_2 = utility_1.test_column_query_1(input_filename_list,id_column=['peak_id','gene_id'],df_list=[],
																			column_vec=['pval1'],reset_index=False)
					output_filename_1 = input_filename_pre2
					df_gene_peak_query_2.index = np.asarray(df_gene_peak_query_2['gene_id'])
					df_gene_peak_query_2.to_csv(output_filename_1,sep='\t')
				# else:
				# 	df_gene_peak_query_2 = df_gene_peak_query_1

		return df_gene_peak_query_2

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each peak-gene query, estimate peak accessibility-gene expr correlation
	# input: the peak-gene annotation prepared (by distance threshold or by specific criteria)
	# prepare peak_dict: the peak loci associated with gene query
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_correlation_pre2(self,gene_query_vec=[],gene_query_vec_2=[],df_peak_query=[],peak_dict=[],atac_ad=[],rna_exprs=[],
													interval_peak_corr=50,interval_local_peak_corr=10,
													compute_mode=1,flag_query1=0,flag_query2=0,background_query=0,
													correlation_type='spearmanr',save_file_path='',save_file_path_local='',
													filename_prefix_save='',output_filename='',recompute=0, recompute_bg=0,
													save_mode=1,verbose=0,select_config={}):

		file_path1 = self.save_path_1 # the default file path
		if len(atac_ad)==0:
			atac_ad = self.atac_meta_ad
		if len(rna_exprs)==0:
			rna_exprs = self.meta_scaled_exprs

		## parameter configuration
		flag_corr_, method_type, type_id_1 = 1, 1, 1 # correlation without estimating emprical p-value; correlation and p-value; spearmanr
		# flag_corr_, method_type, type_id_1 = 0, 0, 1 # correlation for background peaks without estimating emprical p-value; correlation; spearmanr
		recompute_1 = 1
		config_default = {'flag_corr_':flag_corr_,
							'method_type_correlation':method_type,
							'type_id_correlation':type_id_1,
							'recompute':recompute_1}

		field_query = list(config_default.keys())
		for field_id1 in field_query:
			if not (field_id1 in select_config):
				select_config.update({field_id1:config_default[field_id1]})

		## compute correlation for foreground peaks: mode 1;
		# compute correlation for background peaks: mode 3;
		# thresholding for foreground peaks: mode 2
		if compute_mode in [1,3]:
			## peak accessibilty-gene expression correlation calculation without estimating emprical p-value
			flag_compute=1
			field_query = ['input_filename_pre1','input_filename_pre1','input_filename_pre2_bg']
			
			if (os.path.exists(output_filename)==True) and (recompute==0):
				print('the file exists: %s'%(output_filename))
				df_gene_peak_query = pd.read_csv(output_filename,index_col=0,sep='\t')
				df_gene_peak_query_compute1 = df_gene_peak_query
			
			else:
				if compute_mode==1:
					interval_peak_corr_1, interval_local_peak_corr_1 = interval_peak_corr, interval_local_peak_corr
					df_gene_peak_query_1 = df_peak_query
					# filename_prefix_save = select_config['filename_prefix_save']
					if filename_prefix_save=='':
						# filename_prefix_save = select_config['filename_prefix_peak_gene']
						filename_prefix_local = select_config['filename_prefix_local']
						filename_prefix_save = select_config['filename_prefix_local']
						print('filename_prefix_local:%s'%(filename_prefix_local))

					type_id_1 = select_config['type_id_correlation']
					rename_column=1
					select_config.update({'rename_column':rename_column})
					peak_bg_num = -1
				
				else:
					print('load background peak loci ')
					input_filename_peak, input_filename_bg, peak_bg_num = select_config['input_filename_peak'],select_config['input_filename_bg'],select_config['peak_bg_num']
					print('peak_bg_num: %d'%(peak_bg_num))
					peak_bg = self.test_gene_peak_query_bg_load(input_filename_peak=input_filename_peak,
																input_filename_bg=input_filename_bg,
																peak_bg_num=peak_bg_num)

					self.peak_bg = peak_bg
					# print('peak_bg ',peak_bg.shape,peak_bg[0:5])
					if verbose>0:
						print('peak_bg ',peak_bg.shape)
						print('peak_bg_num ',peak_bg_num)

					list_interval = [interval_peak_corr, interval_local_peak_corr]
					field_query = ['interval_peak_corr_bg','interval_local_peark_corr']
					for i1 in range(2):
						if field_query[i1] in select_config:
							list_interval[i1] = select_config[field_query[i1]]
					interval_peak_corr_1, interval_local_peak_corr_1 = list_interval

					flag_corr_, method_type, type_id_1 = 0, 0, 1
					# select_config.update({'flag_corr_':flag_corr_,'method_type_correlation':method_type})
					select_config.update({'flag_corr_':flag_corr_})
					if not ('type_id_correlation') in select_config:
						select_config.update({'type_id_correlation':type_id_1})
					rename_column=0
					select_config.update({'rename_column':rename_column})
					if 'gene_query_vec_bg' in select_config:
						gene_query_vec_bg = select_config['gene_query_vec_bg']
					else:
						gene_query_vec_bg = gene_query_vec

					if filename_prefix_save=='':
						# filename_prefix_save_bg = select_config['filename_prefix_bg_peak_gene']
						filename_prefix_save_bg = select_config['filename_prefix_bg_local']
						filename_prefix_save = filename_prefix_save_bg
						print('filename_prefix_save_bg:%s'%(filename_prefix_save_bg))

					if len(df_peak_query)==0:
						input_filename_pre2 = select_config['input_filename_pre2']
						df_gene_peak_query = pd.read_csv(input_filename_pre2,index_col=0,sep='\t')
						print('load pre-selected peak-gene associations: %s'%(input_filename_pre2))
					else:
						df_gene_peak_query = df_peak_query

					print('background_query, df_gene_peak_query ',df_gene_peak_query.shape)

					gene_query_idvec_1 = df_gene_peak_query['gene_id'].unique()
					gene_query_num1 = len(gene_query_idvec_1)
					gene_query_vec_bg1 = pd.Index(gene_query_vec_bg).intersection(gene_query_idvec_1,sort=False)
					gene_query_num_bg, gene_query_num_bg1 = len(gene_query_vec_bg), len(gene_query_vec_bg1)
					print('gene_query_idvec_1:%d, gene_query_vec_bg:%d, gene_query_vec_bg1:%d '%(gene_query_num1,gene_query_num_bg,gene_query_num_bg1))

					gene_query_vec = gene_query_vec_bg1
					df_gene_peak_query_1 = df_gene_peak_query

				warnings.filterwarnings('ignore')
				print('peak accessibility-gene expression correlation estimation')
				# print('gene_query_vec: ',gene_query_vec)
				print('atac_ad: ',atac_ad.shape)
				print('rna_exprs: ',rna_exprs.shape)
				print('filename_prefix_save: %s'%(filename_prefix_save))
				print('save_file_path: %s, save_file_path_local: %s'%(save_file_path,save_file_path_local))
				
				df_gene_peak_query = self.test_motif_peak_estimate_gene_peak_query_correlation_1(gene_query_vec=gene_query_vec,peak_dict=peak_dict,
																								df_gene_peak_query=df_gene_peak_query_1,
																								atac_ad=atac_ad,rna_exprs=rna_exprs,
																								flag_compute=flag_compute,
																								save_file_path=save_file_path,
																								save_file_path_local=save_file_path_local,
																								filename_prefix_save=filename_prefix_save,
																								interval_peak_corr=interval_peak_corr_1,
																								interval_local_peak_corr=interval_local_peak_corr_1,
																								peak_bg_num=peak_bg_num,
																								verbose=verbose,
																								select_config=select_config)

				if (save_mode>0) and (output_filename!=''):
					df_gene_peak_query.to_csv(output_filename,sep='\t',float_format='%.6E')
					print('save file: %s'%(output_filename))

				df_gene_peak_query_compute1 = df_gene_peak_query
				print('df_gene_peak_query, compute_mode: ', df_gene_peak_query.shape,compute_mode)
				
				warnings.filterwarnings('default')

		return df_gene_peak_query_compute1, df_gene_peak_query
		# return df_gene_peak_query

	## gene-peak association query: peak accessibility correlation with gene expr, retrieve peak accessibility-gene expr correlation estimated
	def test_motif_peak_estimate_gene_peak_query_correlation_1(self,gene_query_vec=[],peak_dict=[],df_gene_peak_query=[],
																atac_ad=[],rna_exprs=[],flag_compute=1,save_file_path='',save_file_path_local='',
																filename_prefix_save='',interval_peak_corr=50,interval_local_peak_corr=10,peak_bg_num=-1,verbose=0,select_config={}):

		# peak accessibility-gene expr correlation query
		print('peak accessibility-gene expr correlation query')
		# start = time.time()
		if len(atac_ad)==0:
			atac_ad = self.atac_meta_ad
		if len(rna_exprs)==0:
			rna_exprs = self.meta_scaled_exprs
		select_config.update({'save_file_path':save_file_path,
								'save_file_path_local':save_file_path_local,
								'filename_prefix_save_peak_corr':filename_prefix_save,
								'interval_peak_corr_1':interval_peak_corr,
								'interval_local_peak_corr_1':interval_local_peak_corr})

		# flag_query1=flag_compute
		if flag_compute>0:
			## peak selection; peak accessibility-gene expr correlation estimation
			start = time.time()
			
			field_query = ['flag_corr_','method_type_correlation','type_id_correlation']
			flag_corr_,method_type,type_id_1 = 1,1,1
			list1 = [flag_corr_,method_type,type_id_1]
			for i1 in range(3):
				if field_query[i1] in select_config:
					list1[i1] = select_config[field_query[i1]]
			flag_corr_,method_type,type_id_1 = list1
			print('flag_corr_, method_type, type_id_1 ',flag_corr_,method_type,type_id_1)
			recompute=0
			if 'recompute' in select_config:
				recompute = select_config['recompute']
			save_filename_list = self.test_search_peak_dorc_pre1(atac_ad=atac_ad,rna_exprs=rna_exprs,gene_query_vec=gene_query_vec,
																	df_gene_peak_query=df_gene_peak_query,
																	peak_dict=peak_dict,
																	flag_corr_=flag_corr_,
																	method_type=method_type,
																	type_id_1=type_id_1,	
																	recompute=recompute,
																	peak_bg_num=peak_bg_num,
																	save_mode=1,filename_prefix_save=filename_prefix_save,
																	save_file_path=save_file_path_local,
																	select_config=select_config)

			stop = time.time()
			print('peak accessibility-gene expr correlation query used %.5fs'%(stop-start))
		else:
			if 'save_filename_list' in select_config:
				save_filename_list = select_config['save_filename_list']
			else:
				# save_filename_list = ['%s/test_gene_peak_local_1.%s.%d.1.npy'%(save_file_path_local,filename_prefix_save,iter_id) for iter_id in range(iter_num)]
				interval_save = -1
				if 'interval_save' in select_config:
					interval_save = select_config['interval_save']
				if interval_save<0:
					gene_query_num = len(gene_query_vec)
					iter_num = np.int(np.ceil(gene_query_num/interval_peak_corr))
					save_filename_list = ['%s/%s.%d.1.npy'%(save_file_path_local,filename_prefix_save,iter_id) for iter_id in range(iter_num)]
				else:
					save_filename_list = ['%s/%s.1.npy'%(save_file_path_local,filename_prefix_save)]

		file_num1 = len(save_filename_list)
		list_1 = []
		if file_num1>0:

			df_gene_peak_query = self.test_gene_peak_query_combine_1(save_filename_list=save_filename_list,verbose=verbose,select_config=select_config)

		return df_gene_peak_query

	## gene-peak association query: load and combine previously estimated peak accessibility-gene expression correlations
	def test_gene_peak_query_combine_1(self,save_filename_list=[],verbose=0,select_config={}):

		file_num1 = len(save_filename_list)
		list_1 = []
		if file_num1>0:
			for i1 in range(file_num1):
				input_filename = save_filename_list[i1]
				if os.path.exists(input_filename)==False:
					df_gene_peak_query = []
					print('the file does not exist: %s'%(input_filename))
					return df_gene_peak_query
				t_data1 = np.load(input_filename,allow_pickle=True)
				gene_peak_local = t_data1[()]
				gene_query_vec_1 = list(gene_peak_local.keys())
				
				df_gene_peak_query_1 = self.test_motif_peak_estimate_gene_peak_query_load_unit(gene_query_vec=gene_query_vec_1,
																								gene_peak_annot=gene_peak_local,
																								df_gene_peak_query=[],
																								field_query=[],
																								verbose=verbose,
																								select_config=select_config)
				list_1.append(df_gene_peak_query_1)		
				print('peak-gene link, gene_query_vec ', df_gene_peak_query_1.shape, len(gene_query_vec_1), i1)

			if file_num1>1:
				df_gene_peak_query = pd.concat(list_1,axis=0,join='outer',ignore_index=False)
			else:
				df_gene_peak_query = list_1[0]

			return df_gene_peak_query

	## gene-peak association query: peak accessibility correlated with gene expr
	# peak accessibility-gene expr query with peak set and background peaks
	# type_id_1: 1: spearmanr; 2: pearsonr; 3: spearmanr and pearsonr
	def test_search_peak_dorc_pre1(self,atac_ad,rna_exprs,gene_query_vec=[],df_gene_peak_query=[],peak_dict=[],flag_corr_=1,method_type=1,type_id_1=1,
										recompute=1,peak_bg_num=100,save_mode=1,filename_prefix_save='',save_file_path='',verbose=0,select_config={}):

		# peak_loc = atac_data.var # DataFrame: peak position and annotation in ATAC-seq data
		atac_meta_ad = atac_ad
		peak_loc = atac_meta_ad.var # DataFrame: peak position and annotation in ATAC-seq data
		sample_id = rna_exprs.index
		sample_id_atac = atac_meta_ad.obs_names
		atac_meta_ad = atac_meta_ad[sample_id,:]

		np.random.seed(0)
		gene_query_vec_ori = gene_query_vec
		gene_query_num = len(gene_query_vec)
		if verbose>0:
			print('df_gene_peak_query ')
			print(df_gene_peak_query[0:5])
			print('gene_query_vec: ',gene_query_num,gene_query_vec[0:10])

		interval, pre_id1, start_id = 50, -1, 0
		if 'interval_peak_corr_1' in select_config:
			interval = select_config['interval_peak_corr_1']
		if interval>0:
			query_num = (pre_id1+1)+int(np.ceil((gene_query_num-start_id)/interval))
		else:
			query_num = 1
			interval = gene_query_num

		# print('gene_query_num:%d,start_id:%d,query_num:%d'%(gene_query_num,start_id,query_num))
		interval_local = -1
		interval_save = -1
		if 'interval_save' in select_config:
			interval_save = select_config['interval_save']
		else:
			select_config.update({'interval_save':interval_save})
		# interval_save = -1
		if 'interval_local_peak_corr_1' in select_config:
			interval_local = select_config['interval_local_peak_corr_1']

		warnings.filterwarnings('ignore')

		save_filename_list = []
		corr_thresh = 0.3
		column_corr_1 = select_config['column_correlation'][0]
		
		if interval_save>0:
			# only save one file for the combined correlation estimation
			output_filename = '%s/%s.1.npy'%(save_file_path,filename_prefix_save)
			gene_peak_local = dict()

		for i1_ori in tqdm(range(query_num)):
			if interval_save<0:
				# save file for the estimation at each inteval
				output_filename = '%s/%s.%d.1.npy'%(save_file_path,filename_prefix_save,i1_ori)
				gene_peak_local = dict()
			if os.path.exists(output_filename)==True:
				print('the file exists', output_filename)
				if recompute==0:
					save_filename_list.append(output_filename)
					continue
				# return

			i1 = i1_ori
			num_2 = np.min([start_id+(i1+1)*interval,gene_query_num])
			gene_num2 = num_2
			gene_idvec1 = gene_query_vec[(start_id+i1*interval):num_2]
			# print(len(gene_query_vec),len(gene_name_query_1),len(gene_idvec1),gene_idvec1[0:10])
			print(len(gene_query_vec),len(gene_idvec1),gene_idvec1[0:10])
			df_query = []
			if flag_corr_==0:
				field_query_vec = [['spearmanr'],['pearsonr'],['spearmanr','pearsonr']]
				df_query = df_gene_peak_query.loc[gene_idvec1,['peak_id']+field_query_vec[type_id_1-1]].fillna(-1)
				print('df_query ',df_query.shape,df_query)

			## dorc_func_pre1: (gene_query, df)
			if interval_local<=0:
				gene_res = Parallel(n_jobs=-1)(delayed(self.dorc_func_pre1)(np.asarray(df_gene_peak_query.loc[[t_gene_query],'peak_id']),
																			t_gene_query,
																			atac_meta_ad,
																			rna_exprs,
																			flag_corr_=flag_corr_,
																			df_query=df_query,
																			corr_thresh=-2,
																			type_id_1=type_id_1,
																			method_type=method_type)
																			for t_gene_query in tqdm(gene_idvec1))
			else:
				## running in parallel for a subset of the genes
				query_num_local = int(np.ceil(interval/interval_local))
				gene_res = []
				gene_query_num_local = len(gene_idvec1)
				for i2 in range(query_num_local):
					t_id1 = interval_local*i2
					t_id2 = np.min([interval_local*(i2+1),gene_query_num_local])
					if i2%500==0:
						print('gene query', i1, i2, t_id1, t_id2)
						print(gene_idvec1[t_id1:t_id2])
					gene_res_local_query = Parallel(n_jobs=-1)(delayed(self.dorc_func_pre1)(np.asarray(df_gene_peak_query.loc[[t_gene_query],'peak_id']),
																							t_gene_query,
																							atac_meta_ad,
																							rna_exprs,
																							flag_corr_=flag_corr_,
																							df_query=df_query,
																							corr_thresh=-2,
																							type_id_1=type_id_1,
																							method_type=method_type)
																							for t_gene_query in tqdm(gene_idvec1[t_id1:t_id2]))
					for t_gene_res in gene_res_local_query:
						gene_res.append(t_gene_res)

			gene_query_num_1 = len(gene_res)
			for i2 in tqdm(range(gene_query_num_1)):
				vec1 = gene_res[i2]
				if type(vec1) is int:
					continue
				t_gene_query, df = vec1[0], vec1[1]
				try:
				# if len(df)>0:
					gene_peaks = df.index[df[column_corr_1].abs()>corr_thresh]
					gene_peak_local[t_gene_query] = df
					query_num = len(gene_peaks)
					try:
						if query_num>0:
							print(t_gene_query,query_num,gene_peaks,df.loc[gene_peaks,:])
					except Exception as error:
						print('error! ', error)
						print(t_gene_query,gene_peaks,query_num)

				except Exception as error:
					print('error! ', error,t_gene_query)

			print(len(gene_peak_local.keys()))
			if interval_save>0:
				if (gene_num2%interval_save==0):
					np.save(output_filename,gene_peak_local,allow_pickle=True)
			else:
				np.save(output_filename,gene_peak_local,allow_pickle=True)
				save_filename_list.append(output_filename)

		if interval_save>0:
			try:
				if (gene_num2%interval_save!=0):
					np.save(output_filename,gene_peak_local,allow_pickle=True)
				save_filename_list.append(output_filename)
			except Exception as error:
				print('error! ',error)
		warnings.filterwarnings('default')

		return save_filename_list

	## find the chromatin accessibility peaks correlated with a gene
	# from the notebook
	# flag_corr_: 0: background peaks only; 1: foreground peaks only; 2: foreground and background peaks
	# TODO: need to rewrite
	def dorc_func_pre1(self,peak_loc,gene_query,atac_read,rna_exprs,flag_corr_=1,df_query=[],spearman_cors=[],pearson_cors=[],gene_id_query='',
							corr_thresh=0.01,type_id_1=0,method_type=0,background_query=0,verbose=0):

		# peak_loc = peak_dict[gene_query]
		# print('gene_query ',gene_query)
		try:
			if verbose>0:
				print(len(peak_loc),peak_loc[0:5])
		except Exception as error:
			print('error! ',error)
			flag = -1
			return flag
			# return (gene_query,[],gene_id_query)
		warnings.filterwarnings('ignore')

		## correlations
		flag = 0
		if flag_corr_>0:
			try:
				if method_type==0:
					if type(atac_read) is sc.AnnData:
						X = atac_read[:, peak_loc].X.toarray().T
					else:
						X = atac_read.loc[:, peak_loc].T
				else:
					if type(atac_read) is sc.AnnData:
						sample_id = atac_read.obs_names
						X = pd.DataFrame(index=sample_id,columns=peak_loc,data=atac_read[:, peak_loc].X.toarray())
					else:
						X = atac_read.loc[:, peak_loc]

				if type_id_1 in [1,3]:
					df = pd.DataFrame(index=peak_loc, columns=['spearmanr'])
					if method_type==0:
						spearman_cors = pd.Series(np.ravel(pairwise_distances(X,
										rna_exprs[gene_query].T.values.reshape(1, -1),
										metric=spearman_corr, n_jobs=-1)),
										index=peak_loc)
						df['spearmanr'] = spearman_cors
					else:
						spearman_cors, spearman_pvals = utility_1.test_correlation_pvalues_pair(X,rna_exprs.loc[:,[gene_query]],correlation_type='spearmanr',float_precision=-1)
						spearman_cors, spearman_pvals = spearman_cors[gene_query], spearman_pvals[gene_query]
						df['spearmanr'] = spearman_cors
						df['pval1_ori'] = spearman_pvals

				if type_id_1 in [2,3]:
					if type_id_1==2:
						df = pd.DataFrame(index=peak_loc, columns=['pearsonr'])
					if method_type==0:
						pearson_cors = pd.Series(np.ravel(pairwise_distances(X,
										rna_exprs[gene_query].T.values.reshape(1, -1),
										metric=pearson_corr, n_jobs=-1)),
										index=peak_loc)
						df['pearsonr'] = pearson_cors
					else:
						pearson_cors, pearson_pvals = utility_1.test_correlation_pvalues_pair(X,rna_exprs.loc[:,[gene_query]],correlation_type='spearmanr',float_precision=-1)
						pearson_cors, pearson_pvals = pearson_cors[gene_query], pearson_pvals[gene_query]
						df['pearsonr'] = pearson_cors
						df['pval2_ori'] = pearson_pvals

			except Exception as error:
				print('error!')
				print(error)
				flag = 1
				return

			if len(spearman_cors)==0:
				print('spearman_cors length zero ')
				flag = 1

		## Random background
		# if background_query>0:
		if flag_corr_ in [0,2]:
			flag_query1 = 1
			# if flag_query1>0:
			try:
				colnames = self.peak_bg.columns
				if flag_corr_==0:
					df = df_query.loc[gene_query,:]
					df.index = np.asarray(df['peak_id'])
					print('background peak query ',df.shape)
				
				for p in df.index:
					id1 = np.int64(self.peak_bg.loc[p,:]-1)
					rand_peaks = self.atac_meta_peak_loc[id1]
					# try:
					#   rand_peaks = np.random.choice(atac_ad.var_names[(atac_ad.var['GC_bin'] == atac_ad.var['GC_bin'][p])  &\
					#                                               (atac_ad.var['counts_bin'] == atac_ad.var['counts_bin'][p])], 100, False)
					# except:
					#   rand_peaks = np.random.choice(atac_ad.var_names[(atac_ad.var['GC_bin'] == atac_ad.var['GC_bin'][p])  &\
					#                                               (atac_ad.var['counts_bin'] == atac_ad.var['counts_bin'][p])], 100, True)
					
					if type(atac_read) is sc.AnnData:
						X = atac_read[:, rand_peaks].X.toarray().T
					else:
						X = atac_read.loc[:, rand_peaks].T

					# type_id_1: 1: estimate spearmanr; 3: estimate spearmanr and pearsonr
					if type_id_1 in [1,3]:
						column_id1, column_id2 = 'spearmanr','pval1'
						rand_cors_spearman = pd.Series(np.ravel(pairwise_distances(X,
														rna_exprs[gene_query].T.values.reshape(1, -1),
														metric=spearman_corr, n_jobs=-1)),
														index=rand_peaks)

						m1, v1 = np.mean(rand_cors_spearman), np.std(rand_cors_spearman)
						spearmanr_1 = df.loc[p,column_id1]
						pvalue1 = 1 - norm.cdf(spearmanr_1, m1, v1)

						if (spearmanr_1<0) and (pvalue1>0.5):
							pvalue1 = 1-pvalue1
							# print(spearmanr_1,pvalue1,p)
						df.loc[p,column_id2]= pvalue1

					# type_id_1: 2: estimate pearsonr; 3: estimate spearmanr and pearsonr
					if type_id_1 in [2,3]:
						column_id1, column_id2 = 'pearsonr','pval2'
						rand_cors_pearson = pd.Series(np.ravel(pairwise_distances(X,
														rna_exprs[gene_query].T.values.reshape(1, -1),
														metric=pearson_corr, n_jobs=-1)),
														index=rand_peaks)

						m2, v2 = np.mean(rand_cors_pearson), np.std(rand_cors_pearson)
						pearsonr_1 = df.loc[p,column_id1]
						pvalue2 = 1 - norm.cdf(pearsonr_1, m2, v2)

						if (pearsonr_1<0) and (pvalue2>0.5):
							pvalue2 = 1-pvalue2
							# print(pearsonr_1,pvalue2,p)
						df.loc[p,column_id2]= pvalue2

			except Exception as error:
				print('error!')
				print(error)
				# return (gene_query, [], gene_id_query)
				flag = 1

		# warnings.filterwarnings('default')
		if flag==1:
			# return (gene_query, [], gene_id_query)
			return flag

		return (gene_query, df, gene_id_query)

	## gene-peak association query: load previously estimated peak-gene correlations
	def test_motif_peak_estimate_gene_peak_query_load_unit(self,gene_query_vec,gene_peak_annot,df_gene_peak_query=[],
																field_query=[],verbose=1,select_config={}):
		gene_query_num = len(gene_query_vec)
		flag1 = (len(df_gene_peak_query)>0)
		list1 = []
		gene_query_idvec = []
		if flag1>0:
			gene_query_idvec = df_gene_peak_query.index

		for i1 in range(gene_query_num):
			gene_query_id = gene_query_vec[i1]
			df = gene_peak_annot[gene_query_id]	# retrieve the correlation estimation
			if (verbose>0) and (i1%100==0):
				print('gene_query_id: ',gene_query_id,df.shape)

			if len(field_query)==0:
				field_query = df.columns
			if (gene_query_id in gene_query_idvec):
				peak_local = df_gene_peak_query.loc[gene_query_id]
				id1_pre = peak_local.index.copy()
				peak_local.index = np.asarray(peak_local['peak_id'])
				peak_loc_1 = df.index
				peak_loc_pre = peak_local.index.intersection(peak_loc_1,sort=False)
				peak_local.loc[peak_loc_pre,field_query] = df.loc[peak_loc_pre,field_query]
				peak_local.index = id1_pre  # reset the index
				df_gene_peak_query.loc[gene_query_id,field_query] = peak_local.loc[:,field_query]
			else:
				df = df.loc[:,field_query]
				df['peak_id'] = np.array(df.index)
				df.index = [gene_query_id]*df.shape[0]
				df['gene_id'] = [gene_query_id]*df.shape[0]
				if flag1>0:
					df_gene_peak_query = pd.concat([df_gene_peak_query,df],axis=0,join='outer',ignore_index=False)
				else:
					list1.append(df)

		if len(list1)>0:
			df_gene_peak_query = pd.concat(list1,axis=0,join='outer',ignore_index=False)

		# print('peak-gene link ', df_gene_peak_query)
		return df_gene_peak_query

	## query filename for peak-gene correlation estimated
	# query filename for files saved for peak-gene correlation estimated
	def test_feature_link_query_correlation_file_pre1(self,type_id_query=0,input_file_path='',select_config={}):

		# filename_prefix_1 = select_config['filename_prefix_default']
		# filename_prefix_save_1 = select_config['filename_prefix_save_local']
		filename_annot_vec = select_config['filename_annot_local']
		# filename_prefix = '%s.%s'%(filename_prefix_1,filename_prefix_save_1)
		filename_prefix = select_config['filename_prefix_save_local']
		query_num1 = len(filename_annot_vec)
		input_filename_list = []
		for i1 in range(query_num1):
			filename_annot = filename_annot_vec[i1]
			input_filename_1 = '%s/%s.combine.%s.txt'%(input_file_path,filename_prefix,filename_annot) # highly variable genes; original p-value
			input_filename_2 = '%s/%s.combine.thresh1.%s.txt'%(input_file_path,filename_prefix,filename_annot)	# empirical p-value for the subset of gene-peak link query pre-selected with thresholds
			input_filename_list.extend([input_filename_1,input_filename_2])

		select_config.update({'filename_list_combine_1':input_filename_list})

		return input_filename_list, select_config

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# combine the empirical p-values estimated for the pre-selected peak-gene links with the raw p-values estimated for genome-wide genes
	# add columns to the original peak-link query dataframe: empirical p-values for subset of peak-gene link query
	def test_gene_peak_query_correlation_pre1_combine(self,gene_query_vec=[],peak_distance_thresh=500,df_gene_peak_distance=[],df_peak_query=[],
														peak_loc_query=[],atac_ad=[],rna_exprs=[],highly_variable=False,highly_variable_thresh=0.5,
														load_mode=1,save_mode=1,filename_prefix_save='',input_file_path='',output_filename='',
														save_file_path='',verbose=0,select_config={}):

		## combine original p-value and estimated empirical p-value for different gene groups: highly-variable and not highly-variable
		flag_combine_1=1
		if flag_combine_1>0:
			if len(df_gene_peak_distance)==0:
				df_gene_peak_distance = self.df_gene_peak_distance
			df_pre1 = df_gene_peak_distance
			column_idvec = ['peak_id','gene_id']
			column_id1, column_id2 = column_idvec
			# df_pre1.index = ['%s.%s'%(peak_id,gene_id) for (peak_id,gene_id) in zip(df_pre1['peak_id'],df_pre1['gene_id'])]
			df_pre1.index = test_query_index(df_pre1,column_vec=column_idvec)
			input_filename = output_filename
			input_filename_pre1 = input_filename
			if (load_mode!=1):
				if not ('filename_list_combine_1' in select_config):
					input_filename_list, select_config = self.test_feature_link_query_correlation_file_pre1(input_file_path=input_file_path,select_config=select_config)
				else:
					input_filename_list = select_config['filename_list_combine_1']
			else:
				# the dataframe with peak-gene correlation for genome-wide peak-gene link query
				input_filename_1 = select_config['input_filename_pre1']
				input_filename_2 = select_config['input_filename_pre2']

				if (os.path.exists(input_filename_1)==True) and (os.path.exists(input_filename_2)==True):
					input_filename_list = [input_filename_1,input_filename_2]
			
			flag_query1 = 1
			if flag_query1>0:
				list1,list2 = [],[]
				query_num1 = len(input_filename_list)
				column_correlation = select_config['column_correlation']
				column_1, column_2, column_2_ori = column_correlation[0:3]
				
				field_query_1 = [column_1,column_2_ori]
				field_query_2 = [column_2]
				column_query_1 = [field_query_1,field_query_2]
				
				type_id_1 = 2 # type_id_1:0, use new index; (1,2) use the present index
				type_id_2 = 0 # type_id_2:0, load dataframe from df_list; 1, load dataframe from input_filename_list
				
				reset_index = False
				interval = 2
				group_num = int(query_num1/interval)
				for i1 in range(group_num):
					id1 = (interval*i1)
					# id2 = (id1+interval)
					# df_list2 = [pd.read_csv(input_filename) for input_filename in input_filename_list[id1:id2]]
					df_list2 = []
					for i2 in range(interval):
						df1 = pd.read_csv(input_filename_list[id1+i2],sep='\t')
						df1.index = test_query_index(df1,column_vec=column_idvec)
						df_list2.append(df1)

					df_list = [df_pre1]+df_list2
					df_pre1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec,column_vec=column_query_1,df_list=df_list,
															type_id_1=type_id_1,type_id_2=type_id_2,reset_index=reset_index,select_config=select_config)

				df_gene_peak_query_compute1 = df_pre1
				field_id1 = field_query_2[0]
				
				id1 = (~df_pre1[field_id1].isna())
				df_gene_peak_query = df_pre1.loc[id1,:]

				df_gene_peak_query_compute1.index = np.asarray(df_gene_peak_query_compute1[column_id1])
				df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])
				if (save_mode>0) and (output_filename!=''):
					df_gene_peak_query_compute1.to_csv(output_filename,sep='\t')

			self.df_gene_peak_distance = df_gene_peak_query_compute1  # update columns of df_gene_peak_distance: add column empirical p-value for subset of peak-gene link query
			select_config.update({'filename_gene_peak_annot':output_filename})
			
			self.select_config = select_config
			if verbose>0:
				print('peak-gene link ',df_gene_peak_query_compute1.shape,df_gene_peak_query.shape)

			return df_gene_peak_query_compute1, df_gene_peak_query

	## gene-peak association query: peak accessibility correlation with gene expr, retrieve peak accessibility-gene expr correlation estimated
	# gene-peak association selection by thresholds
	def test_gene_peak_query_correlation_thresh_pre1(self,df_gene_peak_query=[],column_idvec=['peak_id','gene_id'],column_vec_query=[],thresh_corr_distance=[],verbose=0,select_config={}):

		# peak accessibility-gene expr correlation query
		# print('peak accessibility-gene expr correlation query')
		# start = time.time()

		df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_idvec)
		query_idvec = df_gene_peak_query.index
		column_id1, column_id2 = column_idvec
		if len(column_vec_query)==0:
			column_vec_query = ['label_thresh2']
		column_vec_1 = select_config['column_correlation'] # column_vec_1:['spearmanr','pval1']
		column_1, column_2 = column_vec_1[0:2]
		column_2_ori = column_vec_1[2]
		column_distance = select_config['column_distance'] # column_distance: 'distance'
		column_label_1 = column_vec_query[0]
		distance_abs = df_gene_peak_query[column_distance].abs()

		peak_corr, peak_pval = df_gene_peak_query[column_1], df_gene_peak_query[column_2]
		id1 = (pd.isna(peak_pval)==True)
		print('peak_pval: ',np.sum(id1))
		
		df1 = df_gene_peak_query.loc[id1]	
		print('peak-gene link: ',df1.shape)
		print(df1)

		peak_pval[id1] = df_gene_peak_query.loc[id1,column_2_ori]
		list1 = []
		for thresh_vec in thresh_corr_distance:
			constrain_1, constrain_2, thresh_peak_corr_vec = thresh_vec
			id1 = (distance_abs<constrain_2)&(distance_abs>=constrain_1) # constraint by distance
			# id2 = (df_gene_peak_query['spearmanr'].abs()>thresh_corr_)
			
			df_gene_peak_query_sub1 = df_gene_peak_query.loc[id1]
			peak_corr, peak_pval = df_gene_peak_query_sub1[column_1], df_gene_peak_query_sub1[column_2]
			# peak_local_1 = peak_local.copy()
			peak_sel_corr_pos = (peak_corr>=1)
			peak_sel_corr_neg = (peak_corr>=1)

			## constraint by correlation value
			for thresh_peak_corr_vec_1 in thresh_peak_corr_vec:
				thresh_peak_corr_pos, thresh_peak_corr_pval_pos, thresh_peak_corr_neg, thresh_peak_corr_pval_neg = thresh_peak_corr_vec_1
				# peak_sel_pval_ = (peak_pval<thresh_peak_corr_pval)
				peak_sel_corr_pos = ((peak_corr>thresh_peak_corr_pos)&(peak_pval<thresh_peak_corr_pval_pos))|peak_sel_corr_pos
				peak_sel_corr_neg = ((peak_corr<thresh_peak_corr_neg)&(peak_pval<thresh_peak_corr_pval_neg))|peak_sel_corr_neg
				if verbose>0:
					print('distance_thresh1:%d, distance_thresh2:%d'%(constrain_1,constrain_2))
					print('thresh_sel_corr_pos, thresh_peak_corr_pval_pos, thresh_sel_corr_neg, thresh_peak_corr_pval_neg ', thresh_peak_corr_pos, thresh_peak_corr_pval_pos, thresh_peak_corr_neg, thresh_peak_corr_pval_neg)
					print('peak_sel_corr_pos, peak_sel_corr_neg ', np.sum(peak_sel_corr_pos), np.sum(peak_sel_corr_neg))

			peak_sel_corr_ = (peak_sel_corr_pos|peak_sel_corr_neg)
			peak_sel_corr_num1 = np.sum(peak_sel_corr_)
			if verbose>0:
				print('peak_sel_corr_num ', peak_sel_corr_num1)

			# df_gene_peak_query_sub2 = df_gene_peak_query_sub1.loc[peak_sel_corr_]
			query_id1 = df_gene_peak_query_sub1.index
			query_id2 = query_id1[peak_sel_corr_]
			list1.extend(query_id2)

		
		peak_corr, peak_pval = df_gene_peak_query[column_1], df_gene_peak_query[column_2]
		# df_gene_peak_query['distance_abs'] = df_gene_peak_query['distance'].abs()
		
		distance_abs = df_gene_peak_query['distance'].abs()
		df_gene_peak_query['distance_abs'] = distance_abs
		peak_distance_thresh_1 = 500
		
		if 'thresh_corr_retain' in select_config:
			thresh_corr_retain = np.asarray(select_config['thresh_corr_retain'])
			if thresh_corr_retain.ndim==2:
				for (thresh_corr_1, thresh_pval_1) in thresh_corr_retain:
					# query_id_1 = df_gene_peak_query.index
					id1 = (peak_corr.abs()>thresh_corr_1)
					print('thresh correlation: ',np.sum(id1),thresh_corr_1)
					if (thresh_pval_1<1):
						id2 = (peak_pval<thresh_pval_1) # pvalue threshold
						id3 = (distance_abs<peak_distance_thresh_1) # distance threshold; only use correlation threshold within specific distance
						id1_1 = id1&(id2|id3)
						id1_2 = id1&(~id2)
						id1 = id1_1
						print('thresh correlation and pval: ',np.sum(id1),thresh_corr_1,thresh_pval_1)
						df1 = df_gene_peak_query.loc[id1_2,:]
						print('df1 ',df1.shape)
						print(df1)
					query_id3 = query_idvec[id1] # retain gene-peak query with high peak accessibility-gene expression correlation
					df2 = df_gene_peak_query.loc[query_id3,:]
					
					filename_save_thresh2 = select_config['filename_save_thresh2']
					b = filename_save_thresh2.find('.txt')
					output_filename = filename_save_thresh2[0:b]+'.%s.2.txt'%(thresh_corr_1)
					df2.index = np.asarray(df2['gene_id'])
					df2 = df2.sort_values(by=['gene_id','distance'],ascending=[True,True])
					df2.to_csv(output_filename,sep='\t')

					output_filename = filename_save_thresh2[0:b]+'.%s.2.sort1.txt'%(thresh_corr_1)
					df2 = df2.sort_values(by=[column_1,'distance_abs'],ascending=[False,True])
					df2.to_csv(output_filename,sep='\t')

					output_filename = filename_save_thresh2[0:b]+'.%s.2.sort2.txt'%(thresh_corr_1)
					df2 = df2.sort_values(by=['peak_id',column_1,'distance_abs'],ascending=[True,False,True])
					df2.to_csv(output_filename,sep='\t')
					list1.extend(query_id3)
			else:
				thresh_corr_1, thresh_pval_1 = thresh_corr_retain
				id1 = (peak_corr.abs()>thresh_corr_1)
				print('thresh correlation: ',np.sum(id1),thresh_corr_1)
				if (thresh_pval_1<1):
					id2 = (peak_pval<thresh_pval_1)
					id3 = (distance_abs<peak_distance_thresh_1)
					id1 = id1&(id2|id3)
					print('thresh correlation and pval: ',np.sum(id1),thresh_corr_1,thresh_pval_1)
				
				query_id3 = query_idvec[id1] # retain gene-peak query with high peak accessibility-gene expression correlation
				df2 = df_gene_peak_query.loc[query_id3,:]
				
				filename_save_thresh2 = select_config['filename_save_thresh2']
				b = filename_save_thresh2.find('.txt')
				output_filename = filename_save_thresh2[0:b]+'.%s.2.txt'%(thresh_corr_1)
				df2.index = np.asarray(df2['gene_id'])
				df2 = df2.sort_values(by=['gene_id','distance'],ascending=[True,True])
				df2.to_csv(output_filename,sep='\t')

				output_filename = filename_save_thresh2[0:b]+'.%s.2.sort1.txt'%(thresh_corr_1)
				df2 = df2.sort_values(by=[column_1,'distance_abs'],ascending=[False,True])
				df2.to_csv(output_filename,sep='\t')

				output_filename = filename_save_thresh2[0:b]+'.%s.2.sort2.txt'%(thresh_corr_1)
				df2 = df2.sort_values(by=['peak_id',column_1,'distance_abs'],ascending=[True,False,True])
				df2.to_csv(output_filename,sep='\t')
				list1.extend(query_id3)

		query_id_sub1 = pd.Index(list1).unique()
		t_columns = df_gene_peak_query.columns.difference(['distance_abs'],sort=False)
		df_gene_peak_query = df_gene_peak_query.loc[:,t_columns]
		
		df_gene_peak_query_ori = df_gene_peak_query.copy()
		df_gene_peak_query = df_gene_peak_query_ori.loc[query_id_sub1]
		
		# df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
		# df_gene_peak_query_ori.loc[query_id_sub1,'label_thresh2'] = 1
		df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])
		df_gene_peak_query_ori.loc[query_id_sub1,column_label_1] = 1
		df_gene_peak_query_ori.index = np.asarray(df_gene_peak_query_ori[column_id1])
		
		# column_id_1 = 'highly_variable_thresh0.5'
		column_id_1 = select_config['column_highly_variable']
		
		if column_id_1 in df_gene_peak_query.columns:
			column_vec_1 = [column_id_1,column_id1,column_distance]
			df_gene_peak_query = df_gene_peak_query.sort_values(by=column_vec_1,ascending=[False,True,True])
		else:
			column_vec_1 = [column_id1,column_distance]
			df_gene_peak_query = df_gene_peak_query.sort_values(by=column_vec_1,ascending=[True,True])

		return df_gene_peak_query, df_gene_peak_query_ori

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# pre-selection of peak-gene link query using thresholds of the peak-gene correlations and empirical p-values
	def test_gene_peak_query_correlation_pre1_select_1(self,gene_query_vec=[],df_gene_peak_query=[],peak_loc_query=[],atac_ad=[],rna_exprs=[],input_filename='',index_col=0,highly_variable=False,
														peak_distance_thresh=500,save_mode=1,filename_prefix_save='',output_filename_1='',output_filename_2='',save_file_path='',verbose=0,select_config={}):

		## peak-gene query pre-selection
		# pre-selection 2: select based on emprical p-value
		flag_select_thresh2=1
		if flag_select_thresh2>0:
			thresh_corr_1, thresh_pval_1 = 0.01,0.05
			thresh_corr_2, thresh_pval_2 = 0.1,0.1
			
			## thresh 2
			if 'thresh_corr_distance_2' in select_config:
				thresh_corr_distance = select_config['thresh_corr_distance_2']
			else:
				# thresh_distance_1 = 50
				thresh_distance_1 = 100
				if 'thresh_distance_default_2' in select_config:
					thresh_distance_1 = select_config['thresh_distance_default_2'] # the distance threshold with which we retain the peaks without thresholds of correlation and p-value

				thresh_corr_distance = [[0,thresh_distance_1,[[0,1,0,1]]],
										[thresh_distance_1,500,[[0.01,0.1,-0.01,0.1],[0.15,0.15,-0.15,0.15]]],
										[500,1000,[[0.1,0.1,-0.1,0.1]]],
										[1000,2050,[[0.15,0.1,-0.15,0.1]]]]

			start = time.time()
			if len(df_gene_peak_query)==0:
				# df_gene_peak_query = pd.read_csv(input_filename,index_col=0,sep='\t')
				# df_gene_peak_query = pd.read_csv(input_filename,index_col=False,sep='\t')
				df_gene_peak_query = pd.read_csv(input_filename,index_col=index_col,sep='\t')
				column_id1 = 'gene_id'
				if not (column_id1 in df_gene_peak_query.columns):
					df_gene_peak_query[column_id1] = np.asarray(df_gene_peak_query.index)
				else:
					# df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
					df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])

			df_gene_peak_query_pre1, df_gene_peak_query = self.test_gene_peak_query_correlation_thresh_pre1(df_gene_peak_query=df_gene_peak_query,
																											thresh_corr_distance=thresh_corr_distance,
																											verbose=verbose,
																											select_config=select_config)
			
			print('original peak-gene link, peak-gene link after pre-selection by correlation and p-value thresholds ',df_gene_peak_query.shape,df_gene_peak_query_pre1.shape)
			
			stop = time.time()
			print('the pre-selection used %.5fs'%(stop-start))

			# if (save_mode>0) and (output_filename!=''):
			if (save_mode>0):
				if (output_filename_2!=''):
					df_gene_peak_query_pre1.index = np.asarray(df_gene_peak_query_pre1['gene_id'])
					# df_gene_peak_query_1.to_csv(output_filename,sep='\t',float_format='%.5E')
					df_gene_peak_query_pre1.to_csv(output_filename_2,sep='\t')
					# df_gene_peak_query_pre1.to_csv(output_filename_2,index=False,sep='\t')

				if (output_filename_1!=''):
					df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
					# df_gene_peak_query_1.to_csv(output_filename,sep='\t',float_format='%.5E')
					df_gene_peak_query.to_csv(output_filename_1,sep='\t')
					# df_gene_peak_query.to_csv(output_filename_1,index=False,sep='\t')

			# return
			return df_gene_peak_query_pre1, df_gene_peak_query

	## gene-peak association query: search peak-gene association, peak accessibility-gene expr correlation estimation
	# for each gene query, search for peaks within the distance threshold, estimate peak accessibility-gene expr correlation
	# input: the gene query, the peak distance threshold
	# output: peak accessibility-gene expr correlation (dataframe)
	def test_gene_peak_query_pre1_basic_1(self,data=[],input_file_path='',input_filename='',gene_query_vec=[],df_gene_peak_query=[],peak_distance_thresh=500,
															save_mode=1,filename_prefix_save='',output_filename='',save_file_path='',annot_mode=1,verbose=0,select_config={}):

		## query gene-peak association basics for each gene and each peak
		flag_count_query1=1
		if flag_count_query1>0:
			if len(data)==0:
				if input_filename=='':
					input_filename = select_config['filename_save_thresh2']

			# output_file_path = input_file_path2
			output_file_path = save_file_path
			# df_gene_peak_query_group_1: gene-peak association statistics for each gene
			# df_gene_peak_query_group_2: gene-peak association statistics for each peak
			df_gene_peak_query_group_1, df_gene_peak_query_group_2 = self.test_peak_gene_query_basic_1(data=data,input_filename=input_filename,
																										save_mode=1,filename_prefix_save=filename_prefix_save,
																										output_filename='',
																										output_file_path=output_file_path,
																										verbose=verbose,
																										select_config=select_config)

			self.df_gene_peak_query_group_1 = df_gene_peak_query_group_1
			self.df_gene_peak_query_group_2 = df_gene_peak_query_group_2

			# return
			return df_gene_peak_query_group_1, df_gene_peak_query_group_2

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)




