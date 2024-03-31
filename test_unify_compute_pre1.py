#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
import SEACells
from SEACells import core, genescores
# import SEACells_1 as SEACells
# from SEACells_1 import core, genescores
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
# from utility_1 import pyranges_from_strings, test_file_merge_1
from utility_1 import spearman_corr, pearson_corr
import test_rediscover_compute_1
from test_rediscover_compute_1 import _Base2_2
import h5py
import json
import pickle

import itertools
from itertools import combinations

class _Base2_pre1(BaseEstimator):
	"""Feature association estimation.
	"""
	def __init__(self,file_path,run_id=1,species_id=1,cell='ES', 
					generate=1,
					chromvec=[1],
					test_chromvec=[2],
					featureid=1,
					df_gene_annot_expr=[],
					typeid=1,
					method=1,
					flanking=50,
					normalize=1,
					type_id_feature=0,
					config={},
					select_config={}):

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
		# input_file_path1 = self.save_path_1
		# if data_file_type in ['mouse_endoderm','E8.75','E9.25']:
		# 	input_file_path = '%s/data_pre2/mouse_endoderm'%(input_file_path1)
		# 	# root_path_2 = input_file_path
		# 	type_id_feature = select_config['type_id_feature']
		# 	# run_id = select_config['run_id']
		# 	# filename_save_annot_1 = 'E8.75#Multiome'
		# 	input_filename_1 = '%s/E8.75_E9.25_rna_endoderm_with_scrna.2.h5ad'%(input_file_path1)
		# 	input_filename_2 = '%s/E8.75_E9.25_atac_endoderm_with_scatac.2.h5ad'%(input_file_path1)
		# 	data_timepoint = select_config['data_timepoint']
		# 	run_id_load = -1 # the previous annotation
		# select_config = self.test_config_query_1(select_config=select_config)
		# select_config.update({'input_filename_rna':input_filename_1,
		# 						'input_filename_atac':input_filename_2})
		# print('input_filename_rna:%s, input_filename_atac:%s'%(input_filename_1,input_filename_2))
		if not ('type_id_feature' in select_config):
			select_config.update({'type_id_feature':type_id_feature})
		
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
			if data_file_type=='CD34_bonemarrow':
				input_file_path = '%s/data_pre2/cd34_bonemarrow'%(input_file_path1)
				# root_path_2 = '%s/data_pre2/cd34_bonemarrow'%(root_path_1)
				filename_save_annot_1 = data_file_type
				input_filename_1 = '%s/cd34_multiome_rna.copy1.h5ad'%(input_file_path)
				input_filename_2 = '%s/cd34_multiome_atac.copy1.h5ad'%(input_file_path)
			elif data_file_type=='pbmc':
				input_file_path = '%s/data_pre2/10x_pbmc/data_1'%(input_file_path1)
				# root_path_2 = '%s/data_pre2/10x_pbmc/data_1'%(root_path_1)
				type_id_feature = select_config['type_id_feature']
				run_id = select_config['run_id']
				run_id_1 = 1
				input_filename_1 = '%s/pbmc_multiome_rna.%d.copy1.h5ad'%(input_file_path,run_id_1)
				input_filename_2 = '%s/pbmc_multiome_atac.%d.copy1.h5ad'%(input_file_path,run_id_1)
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
			elif data_file_type=='bonemarrow_Tcell':
				input_file_path = '%s/data_pre2/bonemarrow_Tcell/data_1'%(input_file_path1)
				# root_path_2 = '%s/data_pre2/bonemarrow_Tcell/data_1'%(root_path_1)
				type_id_feature = select_config['type_id_feature']
				run_id = select_config['run_id']
				run_id_1 = 1
				input_filename_1 = '%s/vbak1/bm_multiome_rna.h5ad'%(input_file_path)
				input_filename_2 = '%s/vbak1/bm_multiome_atac.h5ad'%(input_file_path)
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
			elif data_file_type in ['mouse_endoderm','E8.75','E9.25']:
				# input_file_path = '%s/data_pre2/mouse_endoderm'%(input_file_path1)
				# input_file_path = '%s/data_pre2/mouse_endoderm/data_1'%(input_file_path1)
				input_file_path = '%s/data_pre2/mouse_endoderm/data_1/%s'%(input_file_path1,data_file_type)
				# root_path_2 = input_file_path
				type_id_feature = select_config['type_id_feature']
				# run_id = select_config['run_id']
				# filename_save_annot_1 = 'E8.75#Multiome'
				input_filename_1 = '%s/E8.75_E9.25_rna_endoderm_with_scrna.2.h5ad'%(input_file_path1)
				input_filename_2 = '%s/E8.75_E9.25_atac_endoderm_with_scatac.2.h5ad'%(input_file_path1)
				data_timepoint = select_config['data_timepoint']
				run_id_load = -1	# the previous annotation
				if 'run_id_load' in select_config:
					run_id_load = select_config['run_id_load']
				if run_id_load>0:
					filename_save_annot_1 = '%s.%d'%(data_timepoint,run_id_load)
				else:
					filename_save_annot_1 = 'E8.75#Multiome'
			else:
				print('use input file path ')

			# select_config.update({'root_path_2':root_path_2})
			select_config.update({'input_filename_rna':input_filename_1,
									'input_filename_atac':input_filename_2})
			print('input_filename_rna:%s, input_filename_atac:%s'%(input_filename_1,input_filename_2))
			if not('data_path' in select_config):
				select_config.update({'data_path':input_file_path})
			if not('filename_save_annot_1' in select_config):
				select_config.update({'filename_save_annot_1':filename_save_annot_1})
				select_config.update({'filename_save_annot_pre1':filename_save_annot_1})

		## provide file paths
		# file_path1 = self.save_path_1
		input_file_path1 = self.save_path_1
		root_path_1 = select_config['root_path_1']
		root_path_2 = select_config['root_path_2']
		data_file_type = select_config['data_file_type']
		data_file_type_annot = data_file_type.lower()
		data_file_type_vec = ['CD34_bonemarrow','pbmc','bonemarrow_Tcell','-1','E8.75','E9.25']
		# file_path_annot = ['cd34_bonemarrow','10x_pbmc/data_1','bonemarrow_Tcell','mouse_endoderm/E8.75','mouse_endoderm/E9.25']
		file_path_annot = ['cd34_bonemarrow','10x_pbmc/data_1','bonemarrow_Tcell','-1','mouse_endoderm/data_1/E8.75','mouse_endoderm/data_1/E9.25']
		file_path_dict = dict(zip(data_file_type_vec,file_path_annot))
		file_path_annot_1 = ['cd34_bonemarrow','10x_pbmc/data_1','bonemarrow_Tcell','-1','mouse_endoderm','mouse_endoderm']
		file_path_dict_1 = dict(zip(data_file_type_vec,file_path_annot_1))
		human_cell_type = ['CD34_bonemarrow','pbmc','bonemarrow_Tcell']
		mouse_cell_type = ['mouse_endoderm','E8.75','E9.25']
		if data_file_type in human_cell_type:
			self.species_id = 'hg38'
		else:
			self.species_id = 'mm10'

		file_path_annot = file_path_dict[data_file_type]
		# data_dir = '%s/%s'%(input_file_path1,file_path_annot)
		data_dir = '%s/%s'%(root_path_2,file_path_annot)

		file_path_annot_1 = file_path_dict_1[data_file_type]
		data_dir_1 = '%s/%s'%(root_path_2,file_path_annot_1)
		run_id = select_config['run_id']
		type_id_feature = select_config['type_id_feature']
		metacell_num = select_config['metacell_num']
		input_file_path = '%s/metacell_%d/run%d'%(data_dir,metacell_num,run_id)
		file_save_path = '%s/metacell_%d/run%d'%(data_dir,metacell_num,run_id)
		print('root_path_2: %s \n file_path_annot: %s \n data_dir: %s \n'%(root_path_2,file_path_annot,data_dir))
		print('input_file_path: %s'%(input_file_path))
		filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
		if ('run_id_load' in select_config):
			run_id_load = select_config['run_id_load']
			if run_id_load<0:
				data_timepoint = select_config['data_timepoint']
				filename_save_annot_1 = '%s#Multiome'%(data_timepoint)
				input_file_path = select_config['data_path']

		data_path_1 = select_config['data_path']
		data_path_2 = self.save_path_1
		print('data_path_1:%s \n data_path_2:%s'%(data_path_1,data_path_2))
		select_config.update({'data_path':input_file_path,
							'data_path_1':data_path_1,
							'data_path_2':data_path_2,
							'data_path_save':file_save_path,
							'data_dir':data_dir,
							'data_dir_1':data_dir_1,
							'filename_save_annot_1':filename_save_annot_1,
							'filename_save_annot_pre1':filename_save_annot_1})

		if self.species_id=='mm10':
			# filename_annot_prefix = 'Mus_musculus.GRCm38.102.annotations_mm38'
			filename_prefix_1 = 'Mus_musculus.GRCm38.102'
			input_file_path = data_path_2
			input_filename_gene_annot_ori = '%s/gene_annotation/test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.txt'%(input_file_path)
			input_filename_gene_annot = input_filename_gene_annot_ori
			# input_filename = 'test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.1.txt'
			# data_path = select_config['data_path']
			# input_file_path = data_path
			# input_filename_gene_annot = '%s/gene_annotation/test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.1.txt'%(input_file_path)
			# input_filename_gene_annot = '%s/test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.1.txt'%(input_file_path)
			# select_config.update({'input_filename_gene_annot':input_filename_gene_annot})
		elif self.species_id=='hg38':
			# filename_prefix_1 = 'hg38'
			filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
			# gene_annotation_filename = '%s/hg38.1.txt'%(input_file_path1)
			# gene_annotation_filename = '%s/Homo_sapiens.GRCh38.108.1.txt'%(input_file_path1)
			# gene_annotation_filename = '%s/%s.1.txt'%(input_file_path1,filename_prefix_1)
			input_file_path1 = data_path_2
			input_file_path2 = data_dir
			input_filename_gene_annot_ori = '%s/test_gene_annot_ori.Homo_sapiens.GRCh38.108.1.txt'%(input_file_path1)
			input_filename_gene_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path2)

		select_config.update({'filename_gene_annot':input_filename_gene_annot,
								'filename_gene_annot_ori':input_filename_gene_annot_ori})

		self.select_config = select_config

		return select_config

	## file_path query
	def test_file_path_query_2(self,select_config={}):

		input_file_path1 = self.save_path_1
		# rna_ad = sc.read_h5ad('/Users/sharmar1/Dropbox/multiome_jamboree/10x_pbmc/adata_rna.h5ad')
		# atac_ad = sc.read_h5ad('/Users/sharmar1/Dropbox/multiome_jamboree/10x_pbmc/adata_atac.h5ad')
		data_file_type = select_config['data_file_type']
		# data_file_type = 'pbmc'
		# data_file_type = 'CD34_bonemarrow'
		# data_file_type = 'CD34_bonemarrow_no_bcells'
		flag_SEACell_estimate=0
		if 'flag_SEACell_estimate' in select_config:
			flag_SEACell_estimate = select_config['flag_SEACell_estimate']
		flag_summarize=0
		if ('flag_summarize' in select_config):
			flag_summarize = select_config['flag_summarize']
		# print_mode=1
		run_id = select_config['run_id']
		type_id_feature = select_config['type_id_feature']
		feature_type = select_config['feature_type_1']
		if data_file_type == 'pbmc':
			# input_file_path = '%s/data_pre2/10x_pbmc/10x_pbmc'%(input_file_path1)
			# output_file_path = '%s/data_pre2/10x_pbmc/10x_pbmc'%(input_file_path1)
			# input_filename_1 = '%s/adata_rna.h5ad'%(input_file_path)
			# # input_filename_1 = '%s/adata_rna.copy1.h5ad'%(input_file_path)
			# input_filename_2 = '%s/adata_atac.h5ad'%(input_file_path)
			input_file_path_ori = '%s/data_pre2/10x_pbmc/data_1/vbak1'%(input_file_path1)	# the previous file folder
			input_file_path = '%s/data_pre2/10x_pbmc/data_1'%(input_file_path1)
			output_file_path = '%s/data_pre2/10x_pbmc/data_1'%(input_file_path1)
			input_filename_1 = '%s/pbmc_multiome_rna.h5ad'%(input_file_path_ori)
			# input_filename_2 = '%s/pbmc_multiome_atac.h5ad'%(input_file_path_ori)
			# input_filename_1 = '%s/pbmc_multiome_rna.copy1.h5ad'%(input_file_path)
			input_filename_2 = '%s/pbmc_multiome_atac.copy1.h5ad'%(input_file_path_ori)
			# input_filename_1 = '%s/pbmc_multiome_rna.%d.copy1.h5ad'%(input_file_path,run_id)
			# input_filename_2 = '%s/pbmc_multiome_atac.%d.copy1.h5ad'%(input_file_path,run_id)
			if (flag_SEACell_estimate==0) and (flag_summarize>0):
				data_path_save = '%s/run%d'%(input_file_path,run_id)
				# feature_type = select_config['feature_type_1']
				input_filename_load1 = '%s/pbmc_multiome_%s.%d.copy1.h5ad'%(data_path_save,feature_type,run_id)
				# print_mode=0
				if feature_type=='rna':
					input_filename_1 = input_filename_load1
				else:
					input_filename_2 = input_filename_load1
				print('input_filename_1, input_filename_2 ',input_filename_1,input_filename_2)
		elif data_file_type == 'CD34_bonemarrow':
			# input_file_path = '%s/data_pre2/cd34_bonemarrow'%(input_file_path1)
			# output_file_path = '%s/data_pre2/cd34_bonemarrow'%(input_file_path1)
			# input_file_path = '%s/data_pre2/cd34_bonemarrow_2'%(input_file_path1)
			# input_filename_1 = '%s/adata_rna.h5ad'%(input_file_path)
			# input_file_path_ori = '%s/data_pre2/cd34_bonemarrow/data_1/vbak1'%(input_file_path1)	# the previous file folder
			input_file_path = '%s/data_pre2/cd34_bonemarrow/data_1'%(input_file_path1)
			output_file_path = '%s/data_pre2/cd34_bonemarrow/data_1'%(input_file_path1)
			input_file_path_ori = '%s/vbak1'%(input_file_path)	# the previous file folder
			input_filename_1 = '%s/cd34_multiome_rna.h5ad'%(input_file_path_ori)
			input_filename_2 = '%s/cd34_multiome_atac.h5ad'%(input_file_path_ori)
			# input_filename_1 = '%s/cd34_multiome_rna.copy1.h5ad'%(input_file_path)
			# input_filename_2 = '%s/cd34_multiome_atac.copy1.h5ad'%(input_file_path)
			if (flag_SEACell_estimate==0) and (flag_summarize>0):
				data_path_save = '%s/run%d'%(input_file_path,run_id)
				# feature_type = select_config['feature_type_1']
				input_filename_load1 = '%s/cd34_multiome_%s.%d.copy1.h5ad'%(data_path_save,feature_type,run_id)
				if feature_type=='rna':
					input_filename_1 = input_filename_load1
				else:
					input_filename_2 = input_filename_load1

		elif data_file_type == 'CD34_bonemarrow_no_bcells':
			input_file_path = '%s/data_pre2/cd34_bonemarrow_2'%(input_file_path1)
			output_file_path = '%s/data_pre2/cd34_bonemarrow_2'%(input_file_path1)
			input_filename_1 = '%s/cd34_multiome_rna_no_bcells.h5ad'%(input_file_path)
			input_filename_2 = '%s/cd34_multiome_atac_no_bcells.h5ad'%(input_file_path)

		elif data_file_type == 'bonemarrow_Tcell':
			# input_file_path_ori = '%s/data_pre2/cd34_bonemarrow/data_1/vbak1'%(input_file_path1)	# the previous file folder
			input_file_path = '%s/data_pre2/bonemarrow_Tcell/data_1'%(input_file_path1)
			output_file_path = '%s/data_pre2/bonemarrow_Tcell/data_1'%(input_file_path1)
			input_file_path_ori = '%s/vbak1'%(input_file_path)	# the previous file folder
			input_filename_1 = '%s/bm_multiome_rna.h5ad'%(input_file_path_ori)
			# input_filename_2 = '%s/bm_multiome_atac.h5ad'%(input_file_path_ori)
			input_filename_2 = '%s/bm_multiome_atac.copy1.h5ad'%(input_file_path_ori)
			# input_filename_1 = '%s/cd34_multiome_rna.copy1.h5ad'%(input_file_path)
			# input_filename_2 = '%s/cd34_multiome_atac.copy1.h5ad'%(input_file_path)
			if (flag_SEACell_estimate==0) and (flag_summarize>0):
				data_path_save = '%s/run%d'%(input_file_path,run_id)
				# feature_type = select_config['feature_type_1']
				input_filename_load1 = '%s/bm_multiome_%s.%d.copy1.h5ad'%(data_path_save,feature_type,run_id)
				if feature_type=='rna':
					input_filename_1 = input_filename_load1
				else:
					input_filename_2 = input_filename_load1

		elif data_file_type in ['mouse_endoderm','E8.75','E9.25']:
			input_file_path = '%s/data_pre2/mouse_endoderm'%(input_file_path1)
			output_file_path = '%s/data_pre2/mouse_endoderm'%(input_file_path1)
			input_filename_1 = '%s/E8.75_E9.25_rna_endoderm_with_scrna.2.h5ad'%(input_file_path1)
			input_filename_2 = '%s/E8.75_E9.25_atac_endoderm_with_scatac.2.h5ad'%(input_file_path1)
		else:
			print('please input data file type ')
			return

		filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
		root_path_2 = input_file_path
		select_config.update({'data_path':input_file_path,
								'filename_save_annot_1':filename_save_annot_1,
								'filename_save_annot_pre1':filename_save_annot_1})
		select_config.update({'root_path_2':root_path_2})

		select_config.update({'input_filename_rna':input_filename_1,
								'input_filename_atac':input_filename_2})

		return select_config

	## data embedding compuation
	def test_query_dimension_reduction_1(self,adata=[],feature_type_id=0,normalize=1,pca_compute=1,num_components=50,zero_center=True,highly_variable_query=True,use_highly_variable=True,flag_umap=1,verbose=0,select_config={}):

		ad = adata
		if normalize>0:
			print('normalization ',ad.shape)
			raw_ad = sc.AnnData(ad.X)
			raw_ad.obs_names, raw_ad.var_names = ad.obs_names, ad.var_names
			ad.raw = raw_ad

			sc.pp.normalize_per_cell(ad)
			sc.pp.log1p(ad)
			# sc.pp.highly_variable_genes(ad, n_top_genes=1500)
			# num_top_genes = 2000
			# if 'num_top_genes' in select_config:
			# 	num_top_genes = select_config['num_top_genes']
			# sc.pp.highly_variable_genes(ad, n_top_genes=num_top_genes)

		# if (feature_type_id==1) and (not ('num_top_genes' in select_config)):
		# 	highly_variable_query = False
		if highly_variable_query>0:
			# sc.pp.highly_variable_genes(ad, n_top_genes=1500)
			num_top_genes = 3000
			if 'num_top_genes' in select_config:
				num_top_genes = select_config['num_top_genes']
			
			if ('log1p' in ad.uns_keys()):
				if not ('base' in ad.uns['log1p']):
					base_value = np.e   # use natural logarithm
					print('adata log1p: ',ad.uns['log1p'])
					print('adata log1p base: ',base_value)
					ad.uns['log1p']['base'] = base_value
			sc.pp.highly_variable_genes(ad, n_top_genes=num_top_genes)

		if pca_compute>0:
			print('PCA computation ',ad.shape)
			if feature_type_id==1:
				zero_center = False
			else:
				zero_center = True
			# sc.tl.pca(ad, zero_center=zero_center, n_comps=num_components, use_highly_variable=True)
			sc.tl.pca(ad, zero_center=zero_center, n_comps=num_components, use_highly_variable=use_highly_variable)
			if feature_type_id==1:
				ad.obsm['X_svd'] = ad.obsm['X_pca'].copy()
			print(ad)

		# flag_umap = 1
		n_neighbors = 15
		n_pcs = 50
		field_query = ['neighbors','n_pcs']
		list1 = [n_neighbors,n_pcs]
		field_num = len(field_query)
		for i1 in range(field_num):
			field_id = field_query[i1]
			if field_id in select_config:
				list1[i1] = select_config[field_id]
		
		n_neighbors, n_pcs = list1
		# if 'neighbors' in select_config:
		# 	n_neighbors = select_config['neighbors']
		# if 'n_pcs' in select_config:
		# 	n_pcs = select_config['n_pcs']
		if (flag_umap>0):
			print('umap estimation')
			layer_name_vec = ['X_pca','X_svd']
			layer_name_query = layer_name_vec[feature_type_id]
			# print('search for neighbors: ',n_neighbors,n_pcs,layer_name_query)
			sc.pp.neighbors(ad, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=layer_name_query)
			sc.tl.umap(ad)
			# sc.pl.umap(adata,color=[column_id],palette='tab20', # 'palette' specifies the colormap to use)
			# 									title=["Clusters"])
			# sc.pl.umap(adata,color=[column_id],palette=palatte_name, # 'palette' specifies the colormap to use)
			# 									title=[title])

		return ad

	## metacell estimation
	def test_metacell_compute_unit_1(self,adata,feature_type_id=0,normalize=True,zero_center=True,highly_variable_query=True,use_highly_variable=True,n_SEACells=500,obsm_build_kernel='X_pca',pca_compute=1,num_components=50,n_waypoint_eigs=10,waypoint_proportion=1.0,plot_convergence=1,save_mode=1,select_config={}):

		ad = adata
		if normalize>0:
			print('normalization ',ad.shape)
			raw_ad = sc.AnnData(ad.X)
			raw_ad.obs_names, raw_ad.var_names = ad.obs_names, ad.var_names
			ad.raw = raw_ad

			sc.pp.normalize_per_cell(ad)
			sc.pp.log1p(ad)
			# sc.pp.highly_variable_genes(ad, n_top_genes=1500)
			# num_top_genes = 2000
			# if 'num_top_genes' in select_config:
			# 	num_top_genes = select_config['num_top_genes']
			# sc.pp.highly_variable_genes(ad, n_top_genes=num_top_genes)

		if highly_variable_query>0:
			# sc.pp.highly_variable_genes(ad, n_top_genes=1500)
			num_top_genes = 3000
			if 'num_top_genes' in select_config:
				num_top_genes = select_config['num_top_genes']
			
			if ('log1p' in ad.uns_keys()):
				if not ('base' in ad.uns['log1p']):
					base_value = np.e   # use natural logarithm
					print('adata log1p: ',ad.uns['log1p'])
					print('adata log1p base: ',base_value)
					ad.uns['log1p']['base'] = base_value
			sc.pp.highly_variable_genes(ad, n_top_genes=num_top_genes)

		if pca_compute>0:
			print('PCA computation ',ad.shape)
			if feature_type_id==1:
				zero_center = False
			# sc.tl.pca(ad, zero_center=zero_center, n_comps=num_components, use_highly_variable=True)
			sc.tl.pca(ad, zero_center=zero_center, n_comps=num_components, use_highly_variable=use_highly_variable)
			if feature_type_id==1:
				ad.obsm['X_svd'] = ad.obsm['X_pca'].copy()
			print(ad)

		flag_umap = 1
		n_neighbors = 15
		n_pcs = 50
		field_query = ['neighbors','n_pcs']
		list1 = [n_neighbors,n_pcs]
		field_num = len(field_query)
		for i1 in range(field_num):
			field_id = field_query[i1]
			if field_id in select_config:
				list1[i1] = select_config[field_id]
		n_neighbors, n_pcs = list1
		# if 'neighbors' in select_config:
		# 	n_neighbors = select_config['neighbors']
		# if 'n_pcs' in select_config:
		# 	n_pcs = select_config['n_pcs']
		if (flag_umap>0):
			print('umap estimation')
			layer_name_vec = ['X_pca','X_svd']
			layer_name_query = layer_name_vec[feature_type_id]
			# print('search for neighbors: ',n_neighbors,n_pcs,layer_name_query)
			sc.pp.neighbors(ad, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=layer_name_query)
			sc.tl.umap(ad)
			# sc.pl.umap(adata,color=[column_id],palette='tab20', # 'palette' specifies the colormap to use)
			# 									title=["Clusters"])
			# sc.pl.umap(adata,color=[column_id],palette=palatte_name, # 'palette' specifies the colormap to use)
			# 									title=[title])

		# n_SEACells = 90
		# build_kernel_on = 'X_pca' # key in ad.obsm to use for computing metacells
		# 						  # This would be replaced by 'X_svd' for ATAC data

		build_kernel_on = obsm_build_kernel # key in ad.obsm to use for computing metacells
											# This would be replaced by 'X_svd' for ATAC data
		print('build_kernel_on, n_waypoint_eigs ',build_kernel_on,n_waypoint_eigs)

		## Additional parameters
		# n_waypoint_eigs = 10 # Number of eigenvalues to consider when initializing metacells

		model = SEACells.core.SEACells(ad, 
				  build_kernel_on=build_kernel_on, 
				  n_SEACells=n_SEACells, 
				  n_waypoint_eigs=n_waypoint_eigs,
				  convergence_epsilon = 1e-5)

		model.construct_kernel_matrix()
		M = model.kernel_matrix
		# sns.clustermap(M.toarray()[:500,:500])
		sns.clustermap(M.toarray()[:100,:100])
		# input_file_path = select_config['data_path']
		# output_file_path = self.save_path_1
		# output_file_path = input_file_path
		output_file_path = select_config['data_path_save']
		filename_prefix = select_config['filename_prefix_save_local']
		run_id = select_config['run_id']
		if save_mode>0:
			output_filename = '%s/test_%s_clustermap_1.%d.png'%(output_file_path,filename_prefix,run_id)
			plt.savefig(output_filename,format='png')

		np.random.seed(0)
		model.initialize_archetypes()
		output_filename = '%s/test_%s_initialize_1.%d.png'%(output_file_path,filename_prefix,run_id)
		
		layer_name_query1 = 'X_umap'
		if not (layer_name_query1 in ad.obsm):
			print('umap estimation')
			# n_neighbors = 15
			# n_pcs = 50
			layer_name_vec = ['X_pca','X_svd']
			layer_name_query = layer_name_vec[feature_type_id]
			# print('search for neighbors: ',n_neighbors,n_pcs,layer_name_query)
			# sc.pp.neighbors(ad, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=layer_name_query)
			sc.tl.umap(ad)
		SEACells.plot.plot_initialization(ad, model, save_as=output_filename)

		min_iter=10
		max_iter=200
		if 'min_iter' in select_config:
			min_iter=select_config['min_iter']
		if 'max_iter' in select_config:
			max_iter=select_config['max_iter']
		print('min_iter:%d, max_iter:%d'%(min_iter,max_iter))
		# model.fit(min_iter=10, max_iter=200)
		model.fit(min_iter=min_iter, max_iter=max_iter)
		flag_addition_iteration=1
		# flag_addition_iteration=0
		if flag_addition_iteration>0:
			## run additional iterations step-wise using the .step() function
			print(f'Ran for {len(model.RSS_iters)} iterations')
			for _ in range(5):
				model.step()
			print(f'Ran for {len(model.RSS_iters)} iterations')

		output_filename = '%s/test_%s_converge_1.%d.png'%(output_file_path,filename_prefix,run_id)
		model.plot_convergence(save_as=output_filename)

		return ad, model

	## from SEACells: genescores
	def _add_atac_meta_data(self,atac_meta_ad, atac_ad, n_bins_for_gc):
		atac_ad.var['log_n_counts'] = np.ravel(np.log10(atac_ad.X.sum(axis=0)))
		
		if 'GC' in atac_ad.var:
			atac_meta_ad.var['GC_bin'] = np.digitize(atac_ad.var['GC'], np.linspace(0, 1, n_bins_for_gc))
		atac_meta_ad.var['counts_bin'] = np.digitize(atac_ad.var['log_n_counts'],
													 np.linspace(atac_ad.var['log_n_counts'].min(),
																 atac_ad.var['log_n_counts'].max(), 
																 n_bins_for_gc))
	## from SEACells: genescores
	def _normalize_ad(self,meta_ad,target_sum=None,save_raw=True,save_normalize=False):
		if save_raw:
			# Save in raw
			meta_ad.raw = meta_ad.copy()

		# Normalize 
		# sc.pp.normalize_total(meta_ad, key_added='n_counts')
		sc.pp.normalize_total(meta_ad, target_sum=target_sum, key_added='n_counts')
		if save_normalize==True:
			meta_ad.layers['normalize'] = meta_ad.X.copy() # save the normalized read count without log-transformation
		sc.pp.log1p(meta_ad)

	## from SEACells.genescores
	def _prepare_multiome_anndata(self,atac_ad, rna_ad, SEACells_label='SEACell', summarize_layer_atac='X',summarize_layer_rna='X',flag_normalize=1,save_raw_ad_atac=1,save_raw_ad_rna=1,n_bins_for_gc=50,type_id_1=0):
		"""
		Function to create metacell Anndata objects from single-cell Anndata objects for multiome data
		:param atac_ad: (Anndata) ATAC Anndata object with raw peak counts in `X`. These anndata objects should be constructed 
		 using the example notebook available in 
		:param rna_ad: (Anndata) RNA Anndata object with raw gene expression counts in `X`. Note: RNA and ATAC anndata objects 
		 should contain the same set of cells
		:param SEACells_label: (str) `atac_ad.obs` field for constructing metacell matrices. Same field will be used for 
		  summarizing RNA and ATAC metacells. 
		:param n_bins_gc: (int) Number of bins for creating GC bins of ATAC peaks.
		:return: ATAC metacell Anndata object and RNA metacell Anndata object
		"""

		# Subset of cells common to ATAC and RNA
		common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names)
		if len(common_cells) != atac_ad.shape[0]:
			print('Warning: The number of cells in RNA and ATAC objects are different. Only the common cells will be used.')
		atac_mod_ad = atac_ad[common_cells, :]
		rna_mod_ad = rna_ad[common_cells, :]

		# #################################################################################
		# Generate metacell matrices

		# Set of metacells
		# if type_id_1==0
		# 	metacells = rna_mod_ad.obs[SEACells_label].astype(str).unique()
		# 	metacells = metacells[rna_mod_ad.obs[SEACells_label].value_counts()[metacells] > 1]
		# else:
		# 	metacells = atac_mod_ad.obs[SEACells_label].astype(str).unique()
		# 	metacells = metacells[atac_mod_ad.obs[SEACells_label].value_counts()[metacells] > 1]

		if type_id_1==0:
			metacells = rna_mod_ad.obs[SEACells_label].astype(str).unique()
			metacells = metacells[rna_mod_ad.obs[SEACells_label].value_counts()[metacells] > 1]
			print('meta_cells rna ',len(metacells))
		else:
			metacells = atac_mod_ad.obs[SEACells_label].astype(str).unique()
			metacells = metacells[atac_mod_ad.obs[SEACells_label].value_counts()[metacells] > 1]
			print('meta_cells atac ',len(metacells))
		
		if 'SEACell_ori' in atac_mod_ad.obs:
			metacells_atac = atac_mod_ad.obs['SEACell_ori'].astype(str).unique()
			metacells_atac = metacells_atac[atac_mod_ad.obs['SEACell_ori'].value_counts()[metacells_atac] > 1]
			print('meta_cells atac ',len(metacells_atac))
			# if type_id_1==1:
			# 	metacells = metacells_atac

		SEACells_label_atac, SEACells_label_rna = SEACells_label, SEACells_label
		SEACells_label_1 = 'SEACell_temp'
		if type_id_1==0:
			# ATAC and RNA summaries using RNA SEACells
			atac_mod_ad.obs[SEACells_label_1] = rna_mod_ad.obs[SEACells_label]
			SEACells_label_atac = SEACells_label_1
		else:
			# ATAC and RNA summaries using ATAC SEACells
			rna_mod_ad.obs[SEACells_label_1] = atac_mod_ad.obs[SEACells_label]
			SEACells_label_rna = SEACells_label_1
		print('SEACells_label_atac, SEACells_label_rna ',SEACells_label_atac,SEACells_label_rna)

		print('Generating Metacell matrices...')
		print(' ATAC')
		atac_meta_ad = core.summarize_by_SEACell(atac_mod_ad, SEACells_label=SEACells_label_atac, summarize_layer=summarize_layer_atac)
		atac_meta_ad = atac_meta_ad[metacells, :]
		## the raw ad was saved in the core.summarize_by_SEACell() function
		# if save_raw_ad>0:
		# 	raw_atac_meta_ad = sc.AnnData(atac_meta_ad.X)
		# 	raw_atac_meta_ad.obs_names, raw_atac_meta_ad.var_names = atac_meta_ad.obs_names, atac_meta_ad.var_names
		# 	atac_meta_ad.raw = raw_atac_meta_ad
		# ATAC - Summarize SVD representation
		if 'X_svd' in atac_mod_ad.obsm:
			svd = pd.DataFrame(atac_mod_ad.obsm['X_svd'], index=atac_mod_ad.obs_names)
			summ_svd = svd.groupby(atac_mod_ad.obs[SEACells_label_atac]).mean()
			atac_meta_ad.obsm['X_svd'] = summ_svd.loc[atac_meta_ad.obs_names, :].values

		# ATAC - Normalize
		# genescores._add_atac_meta_data(atac_meta_ad, atac_mod_ad, n_bins_for_gc)
		self._add_atac_meta_data(atac_meta_ad, atac_mod_ad, n_bins_for_gc)
		sc.pp.filter_genes(atac_meta_ad, min_cells=1)
		if flag_normalize>0:
			# genescores._normalize_ad(atac_meta_ad)
			target_sum=None
			if 'target_sum_atac' in select_config:
				target_sum = select_config['target_sum_atac']
			self._normalize_ad(atac_meta_ad,target_sum=target_sum)

		# RNA summaries using ATAC SEACells
		print(' RNA')
		# rna_mod_ad.obs['temp'] = atac_mod_ad.obs[SEACells_label]
		rna_meta_ad = core.summarize_by_SEACell(rna_mod_ad, SEACells_label=SEACells_label_rna, summarize_layer=summarize_layer_rna)
		rna_meta_ad = rna_meta_ad[metacells, :]
		if flag_normalize>0:
			# genescores._normalize_ad(rna_meta_ad)
			target_sum=None
			if 'target_sum_rna' in select_config:
				target_sum = select_config['target_sum_rna']
			self._normalize_ad(rna_meta_ad,target_sum=target_sum)

		return atac_meta_ad, rna_meta_ad

	## highly-variable gene query
	def test_query_metacell_high_variable_feature(self,pre_meta_ad,highly_variable_feature_query_type=True,select_config={}):
		
		## select highly variable peaks or highly variable genes
		if highly_variable_feature_query_type>0:
			mean_value = np.mean(pre_meta_ad.X.toarray(),axis=0)
			min_mean1, max_mean1 = np.min(mean_value), np.max(mean_value)
			print('mean_value ',min_mean1,max_mean1)
			# min_mean, max_mean = 0.001, 5
			min_mean, max_mean = min_mean1-1E-07, max_mean1+1E-07
			min_disp = 0.5
			if 'highly_variable_min_mean' in select_config:
				min_mean_1 = select_config['highly_variable_min_mean']
				if min_mean_1>0:
					min_mean = min_mean_1

			if 'highly_variable_max_mean' in select_config:
				max_mean_1 = select_config['highly_variable_max_mean']
				if max_mean_1>0:
					max_mean = max_mean_1

			if 'highly_variable_min_disp' in select_config:
				min_disp = select_config['highly_variable_min_disp']

			# print('highly variable peak ', min_mean, max_mean)
			print('highly variable feature query ', min_mean, max_mean)
			sc.pp.highly_variable_genes(pre_meta_ad, layer=None, n_top_genes=None, 
										min_disp=min_disp, max_disp=np.inf, 
											min_mean=min_mean, max_mean=max_mean, 
											span=0.3, n_bins=50, 
											flavor='seurat', subset=False, 
											inplace=True, batch_key=None, check_values=True)

			return pre_meta_ad

	## metacell data scaing
	def test_metacell_compute_unit_2(self,pre_meta_ad,save_mode=1,output_file_path='',filename_prefix='test_meta_ad',output_filename='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			scale_type_id = select_config['scale_type_id']
			# peak read before scaling
			# atac_read_pre_ = atac_read_1.copy()
			# if scale_type_id in [2]:
			# 	pre_read = pd.DataFrame(index=pre_meta_ad.obs_names,
			# 								columns=pre_meta_ad.var_names,
			# 								data=np.asarray(pre_meta_ad.X.todense()))
			pre_meta_ad_scaled = []
			if scale_type_id>=1:
				warnings.filterwarnings('ignore')
				# print(type_id1_atac_scale)
				print(scale_type_id)
				# meta_scaled_read = pd.DataFrame(0.0, index=atac_read_1.obs_names,
				# 					 columns=atac_read_1.var_names)

				# ATAC-seq peak read after normalization, log transformation, and before scaling
				if scale_type_id==1:
					print('read scaling')
					# mtx1 = scale(np.asarray(pre_meta_ad.X.todense()))
					mtx1 = scale(pre_meta_ad.X.toarray())
					pre_meta_scaled_read = pd.DataFrame(data=mtx1,index=pre_meta_ad.obs_names,
															columns=pre_meta_ad.var_names,
															dtype=np.float32)

				elif scale_type_id==2:
					# gene expression after normalization, log transformation, and scaling
					pre_read = pd.DataFrame(index=pre_meta_ad.obs_names,
											columns=pre_meta_ad.var_names,
											data=np.asarray(pre_meta_ad.X.todense()))
					pre_meta_scaled_read = pd.DataFrame(0.0, index=pre_meta_ad.obs_names,
																columns=pre_meta_ad.var_names,
																dtype=np.float32)
					warnings.filterwarnings('ignore')
					thresh_upper_1, thresh_lower_1 = 99, 1
					feature_query_vec = pre_read.columns
					feature_num1 = len(feature_query_vec)
					# for feature_query in tqdm(pre_read.columns):
					for i1 in range(feature_num1):
						feature_query = feature_query_vec[i1]
						t_value1 = pre_read[feature_query]
						thresh_2 = np.percentile(t_value1, thresh_upper_1)
						thresh_1_ori = np.percentile(t_value1, thresh_lower_1)
						# thresh_1 = 0
						if (thresh_2<1E-03):
							if (np.max(t_value1)>2):
								print('feature_query ',feature_query,i1,thresh_2,thresh_1_ori,np.max(t_value1),np.min(t_value1),np.mean(t_value1),np.median(t_value1))
							thresh_2=np.max(t_value1)
							if thresh_2==0:
								continue
						# if np.min(t_value1)>1E-03:
						# 	# thresh_1=np.percentile(t_value1, thresh_lower_1)
						# 	thresh_1=thresh_1_ori
						thresh_1=thresh_1_ori
						# exprs = minmax_scale(pre_read[feature_query], 
						# 					[0, np.percentile(pre_read[feature_query], 99)])
						exprs = minmax_scale(t_value1,[thresh_1,thresh_2])
						pre_meta_scaled_read[feature_query] = scale(exprs)
					warnings.filterwarnings('default')

				else:
					print('read minmax scaling')
					# mtx1 = minmax_scale(np.asarray(pre_meta_ad.todense()))
					mtx1 = minmax_scale(pre_meta_ad.X.toarray())
					pre_meta_scaled_read = pd.DataFrame(data=mtx1,index=pre_meta_ad.obs_names,
															columns=pre_meta_ad.var_names,
															dtype=np.float32)

				pre_meta_scaled_read_1 = csr_matrix(pre_meta_scaled_read)
				# pre_meta_ad_scaled = sc.AnnData(pre_meta_scaled_read)
				# pre_meta_ad_scaled.X = csr_matrix(pre_meta_ad_scaled.X)
				pre_meta_ad_scaled = sc.AnnData(pre_meta_scaled_read_1)
				pre_meta_ad_scaled.obs_names, pre_meta_ad_scaled.var_names = pre_meta_ad.obs_names, pre_meta_ad.var_names
				warnings.filterwarnings('default')

				if save_mode>0:
					# if output_file_path=='':
					# 	output_file_path = self.save_path_1
					# if (output_filename==''):
					# 	if (output_file_path!=''):
					# 		output_filename = '%s/%s.scale%d.1.h5ad'%(output_file_path,filename_prefix,scale_type_id)
					# 		# pre_meta_atac_ad.write(output_filename)
					# 		pre_meta_ad_scaled.write(output_filename)
					# 		# pre_meta_read = pre_meta_scaled_read
					if (output_filename!=''):
						b = output_filename.find('txt')
						if b>=0:
							float_format = '%.6E'
							pre_meta_scaled_read.to_csv(output_filename,sep='\t',float_format=float_format)	# save txt file
						else:
							pre_meta_ad_scaled.write(output_filename)	# save anndata

				elif save_mode==2:
					pre_meta_ad.layers['scale_%d'%(scale_type_id)] = pre_meta_scaled_read
			# else:
			# 	pre_meta_ad_scaled = []
				# pre_meta_read = pd.DataFrame(index=pre_meta_ad.obs_names,
				# 							columns=pre_meta_ad.var_names,
				# 							data=np.asarray(pre_meta_ad.X.todense()),
				# 							dtype=np.float32)

		# return pre_ad, pre_meta_read, pre_meta_ad, pre_meta_ad_scaled
		return pre_meta_ad, pre_meta_ad_scaled

	## the plot function from plot.py of SEACells
	def plot_assignment_entropy(ad,
								title='Entropy of Metacell Assignment',
								save_as = None,
								show = True,
								bins=None,
								figsize=(5, 5)):
		"""
		Plot the distribution of assignment entropy over all cells. Each cell is assigned with a partial weight
		to a Metacell, and these weights can be used to compute the entropy of assignment as a proxy for confidence
		of each Metacell assignment - lower entropy assignments are more confidence than high entropy assignments.

		:param ad: annData containing 'Metacells_Entropy' column in .obs
		:param title: (str) title for figure
		:param save_as: (str or None) file name to which figure is saved
		:param bins: (int) number of bins for histogram
		:param figsize: (int,int) tuple of integers representing figure size
		:return:
		"""

		plt.figure(figsize=figsize)
		sns.distplot(ad.obs['Metacell_Entropy'], bins=bins)
		plt.title(title)
		sns.despine()
		
		if save_as is not None:
			plt.savefig(save_as, dpi=150, transparent=True)
		if show:
			plt.show()
		plt.close()

	## the plot function from plot.py of SEACells
	def plot_2D(ad, key='X_umap',
				colour_metacells=True,
				title='Metacell Assignments',
				save_as = None,
				show = True,
				cmap='Set2',
				figsize=(5,5),
				SEACell_size = 20,
				cell_size=10
				):
		"""
		Plot 2D visualization of metacells using the embedding provided in 'key'.

		:param ad: annData containing 'Metacells' label in .obs
		:param key: (str) 2D embedding of data. Default: 'X_umap'
		:param colour_metacells: (bool) whether to colour cells by metacell assignment. Default: True
		:param title: (str) title for figure
		:param save_as: (str or None) file name to which figure is saved
		:param cmap: (str) matplotlib colormap for metacells. Default: 'Set2'
		:param figsize: (int,int) tuple of integers representing figure size
		"""
		umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs['SEACell'])
		mcs = umap.loc[ad.obs['SEACell'].unique()]

		plt.figure(figsize=figsize)
		if colour_metacells:
			sns.scatterplot(x=0, y=1,
							hue='SEACell',
							data=umap,
							s=cell_size,
							cmap=cmap,
							legend=None)
			sns.scatterplot(x=0, y=1, s=SEACell_size,
							hue='SEACell',
							data=mcs,
							cmap=cmap,
							edgecolor='black', linewidth=1.25,
							legend=None)
		else:
			sns.scatterplot(x=0, y=1,
							color='grey',
							data=umap,
							s=cell_size,
							cmap=cmap,
							legend=None)
			sns.scatterplot(x=0, y=1, s=SEACell_size,
							color='red',
							data=mcs,
							cmap=cmap,
							edgecolor='black', linewidth=1.25,
							legend=None)

		plt.xlabel(f'{key}-0')
		plt.ylabel(f'{key}-1')
		plt.title(title)
		ax = plt.gca()
		ax.set_axis_off()
		
		if save_as is not None:
			plt.savefig(save_as, dpi=150, transparent=True)
		if show:
			plt.show()
		plt.close()

	## the plot function from plot.py of SEACells
	def plot_SEACell_sizes(ad,
							save_as=None,
							show = True,
							title='Distribution of Metacell Sizes',
							bins = None,
							figsize=(5,5)):

		"""
		Plot distribution of number of cells contained per metacell.
		:param ad: annData containing 'Metacells' label in .obs
		:param save_as: (str) path to which figure is saved. If None, figure is not saved.
		:param title: (str) title of figure.
		:param bins: (int) number of bins for histogram
		:param figsize: (int,int) tuple of integers representing figure size
		:return: None
		"""

		assert 'SEACell' in ad.obs, 'AnnData must contain "SEACell" in obs DataFrame.'
		label_df = ad.obs[['SEACell']].reset_index()
		plt.figure(figsize=figsize)
		sns.distplot(label_df.groupby('SEACell').count().iloc[:, 0], bins=bins)
		sns.despine()
		plt.xlabel('Number of Cells per SEACell')
		plt.title(title)
		
		if save_as is not None:
			plt.savefig(save_as)
		if show:
			plt.show()
		plt.close()
		return pd.DataFrame(label_df.groupby('SEACell').count().iloc[:, 0]).rename(columns={'index':'size'})

	## the plot function from plot.py of SEACells
	def plot_initialization(ad,
							model,
							plot_basis='X_umap',
							save_as=None,
							show = True,):

		"""
		Plot archetype initizlation
		:param ad: annData containing 'Metacells' label in .obs
		:param model: Initilized SEACells model
		:return: None
		"""

		plt.figure()

		plt.scatter(ad.obsm[plot_basis][:, 0],
			ad.obsm[plot_basis][:, 1],
			s=1, color='lightgrey')
		init_points = ad.obs_names[model.archetypes]
		plt.scatter(ad[init_points].obsm[plot_basis][:, 0],
			ad[init_points].obsm[plot_basis][:, 1],
			s=20)
		ax = plt.gca()
		ax.set_axis_off()

		if save_as is not None:
			plt.savefig(save_as)
		if show:
			plt.show()
		plt.close()

	# the plot function
	def plot_1(self,ad,plot_basis='X_umap',sample_query_id=[],figsize=(5,5),title='',save_as=None,show = True):

		"""
		Plot archetype initizlation
		:param ad: annData containing 'Metacells' label in .obs
		:param model: Initilized SEACells model
		:return: None
		"""

		plt.figure(figsize=figsize)
		plt.scatter(ad.obsm[plot_basis][:, 0],
			ad.obsm[plot_basis][:, 1],
			s=1, color='lightgrey')
		# init_points = ad.obs_names[model.archetypes]
		points_1 = sample_query_id
		plt.scatter(ad[points_1].obsm[plot_basis][:, 0],
			ad[points_1].obsm[plot_basis][:, 1],
			s=20)

		if 'title'!='':
			plt.title(title)
		ax = plt.gca()
		ax.set_axis_off()

		if save_as is not None:
			plt.savefig(save_as)
		if show:
			plt.show()
		plt.close()

	# the plot function
	def plot_2(self,ad,plot_basis='X_umap',sample_id_query=[],sample_id_query2=[],column_query=[],figsize=(5,5),title='',cmap='Set2',legend_query=1,size_1=10,size_2=20,save_as=None,show = True):

		"""
		Plot archetype initizlation
		:param ad: annData containing 'Metacells' label in .obs
		:param model: Initilized SEACells model
		:return: None
		"""
		# plt.figure()
		# plt.scatter(ad.obsm[plot_basis][:, 0],
		# 	ad.obsm[plot_basis][:, 1],
		# 	s=1, color='lightgrey')
		# # init_points = ad.obs_names[model.archetypes]
		# points_1 = sample_query_id
		# plt.scatter(ad[points_1].obsm[plot_basis][:, 0],
		# 	ad[points_1].obsm[plot_basis][:, 1],
		# 	s=20)
		# ax = plt.gca()
		# ax.set_axis_off()

		# cmap_pre1 = {0:'gray',1:'red'}
		# sns.scatterplot(ax=ax_1,data=df1,x=column_id1,y=column_id2,hue=feature_query,palette=cmap_pre1)

		# column_id_query = 'label'
		# cmap = {0:'gray',1:'red'}

		if len(figsize)>0:
			plt.figure(figsize=figsize)
		else:
			plt.figure()
		feature_id1 = str(plot_basis[2:]).upper()
		# column_id1 = '%s1'%(feature_id1)
		# column_id2 = '%s2'%(feature_id1)
		# n_dim1 = 2
		sample_id1 = ad.obs_names
		feature_mtx = ad.obsm[plot_basis]
		# df1 = pd.DataFrame(index=sample_id1,columns=column_vec,data=np.asaray(feature_mtx[:,0:n_dim1]))
		# umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs['SEACell'])
		# mcs = umap.loc[ad.obs['SEACell'].unique()]

		n_dim1 = feature_mtx.shape[1]
		column_vec = ['%s%d'%(feature_id1,query_id1) for query_id1 in range(1,n_dim1+1)]
		column_id1, column_id2 = column_vec[0:2]

		# df1 = pd.DataFrame(ad.obsm[plot_basis]).set_index(ad.obs_names).join(ad.obs.loc[:,[column_id_query]])
		df1 = pd.DataFrame(index=sample_id1,columns=column_vec,data=np.asarray(feature_mtx))
		df1 = df1.join(ad.obs.loc[:,column_query])
		column_id_query = column_query[0]
		df1[column_id_query] = df1[column_id_query].astype('category')
			
		if len(sample_id_query)>0:
			df_query1 = df1.loc[sample_id_query,:]
		else:
			df_query1 = df1
		
		if legend_query>0:
			ax = sns.scatterplot(data=df_query1,x=column_id1,y=column_id2,hue=column_id_query,s=size_1,palette=cmap)
			# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
			# ax.legend(fontsize=5)
			plt.setp(ax.get_legend().get_texts(), fontsize='5')
			# sns.move_legend(ax, "upper left", bbox_to_anchor=(1.0, 1.0))
		else:
			ax = sns.scatterplot(data=df_query1,x=column_id1,y=column_id2,hue=column_id_query,s=size_1,palette=cmap,legend=False)
		
		if len(sample_id_query2)>0:
			df_query2 = df1.loc[sample_id_query2,:]
			sns.scatterplot(data=df_query2,x=column_id1,y=column_id2,hue=column_id_query,s=size_2,palette=cmap,
								edgecolor='black',linewidth=1.25,legend=False)

		# plt.xlabel(f'{key}-0')
		# plt.ylabel(f'{key}-1')
		plt.xlabel(column_id1)
		plt.ylabel(column_id2)
		if 'title'!='':
			plt.title(title)
		ax = plt.gca()
		ax.set_axis_off()

		if save_as is not None:
			plt.savefig(save_as)
		if show:
			plt.show()
		plt.close()

	# query adata subset
	def test_query_data_load_1(self,feature_type_vec=[],data_list=[],query_idvec=[],query_id_1='',column_id_query='',print_mode=1,save_mode=1,output_file_path='',verbose=0,select_config={}):

			dict_query1 = dict()
			if query_id_1=='':
				query_id_1 = query_idvec[0]
			data_list1 = data_list
			data_list1_ori = data_list1.copy()
			feature_num = len(feature_type_vec)

			for query_id in query_idvec:
				dict_query1[query_id] = dict()
				for i1 in range(feature_num):
					feature_type = feature_type_vec[i1]

					pre_ad = data_list1_ori[i1]

					id1 = (pre_ad.obs[column_id_query]==query_id)
					
					sample_id2 = pre_ad.obs_names[id1]
					pre_ad2 = pre_ad[sample_id2,:]
					if verbose>0:
						print('pre_ad2 ',pre_ad2.shape)
						
					if pre_ad.raw!=None:
						raw_ad = sc.AnnData(pre_ad2.raw.X)
						raw_ad.X = csr_matrix(raw_ad.X)
						raw_ad.obs_names = pre_ad2.obs_names
						raw_ad.var_names = pre_ad2.var_names
						pre_ad2.raw = raw_ad.copy()	# prepare for the field raw data
						if verbose>0:
							print('pre_ad_raw ',pre_ad.raw,pre_ad.raw.shape,pre_ad.raw.X.shape)
							print(pre_ad.raw)
							# print(pre_ad.raw.X)
							print(pre_ad.raw.X[0:5,0:5])
							print('pre_ad2_raw ',pre_ad2.raw,pre_ad2.raw.shape,pre_ad2.raw.X.shape)
					
					if (i1==0) and (print_mode>0):
						output_filename = '%s/test_%s_df_obs.%s.1.ori.txt'%(output_file_path,feature_type,query_id)
						pre_ad.obs.to_csv(output_filename,sep='\t')
						output_filename = '%s/test_%s_df_var.%s.1.ori.txt'%(output_file_path,feature_type,query_id)
						pre_ad.var.to_csv(output_filename,sep='\t')

					if verbose>0:
						print('pre_ad2\n',pre_ad2)
						print('pre_ad2 raw ',pre_ad2.raw)
						if (pre_ad2!=None):
							print(pre_ad2.raw.X[0:5,0:5])
						# try:
						# 	print(pre_ad2.raw.X)
						# except Exception as error:
						# 	print('error! ',error)
					# dict_query1[data_timepoint].update({feature_type:pre_ad2})
					dict_query1[query_id].update({feature_type:pre_ad2})
				
			data_list2 = [dict_query1[query_id_1][feature_type_query] for feature_type_query in feature_type_vec]
			# if verbose>0:
			# 	print('rna_ad, atac_ad ',rna_ad.shape,atac_ad.shape)

			return data_list2

	# query data attributes and save plots
	def test_query_attribute_1(self,data_list=[],input_filename_list=[],feature_type_vec=[],feature_id='',column_id_query='',flag_plot=1,flag_query_1=1,flag_query_2=1,save_mode=1,filename_save_annot='',output_file_path='',print_mode=1,verbose=0,select_config={}):

		feature_num = len(feature_type_vec)
		data_list1 = data_list
		for i1 in range(feature_num):
			# celltype_query_1 = rna_ad.obs['celltype']
			pre_ad = data_list1[i1]
			feature_type_query = feature_type_vec[i1]
			column_id1 = column_id_query
			if print_mode>0:
				# write the obs and var annotations to file
				if column_id1 in pre_ad.obs:
					celltype_query_1 = pre_ad.obs[column_id1]
					celltype_vec_1= celltype_query_1.unique()
					celltype_num1 = len(celltype_vec_1)
					print('celltype_vec_1 ',celltype_num1,celltype_vec_1)
					# df_1 = rna_ad.obs.loc[:,['celltype']]
					df_1 = pre_ad.obs.loc[:,[column_id1]]
					df_1['count'] = 1
					# df1 = df_1.groupby(by=['celltype']).sum()
					df1 = df_1.groupby(by=[column_id1]).sum()
					# output_filename = '%s/test_rna_celltype_query.%s.1.txt'%(output_file_path,data_file_type)
					# output_filename = '%s/test_%s_celltype_query.%s.1.txt'%(output_file_path,feature_type_query,filename_save_annot)
					output_filename = '%s/test_%s_ad.%s.celltype_query.1.txt'%(output_file_path,feature_type_query,filename_save_annot)
					df1.to_csv(output_filename,sep='\t')
				else:
					print('the column %s not include'%(column_id1))
				
				# output_filename = '%s/test_rna_df_obs.%s.1.txt'%(output_file_path,data_file_type)
				# output_filename = '%s/test_%s_df_obs.%s.1.txt'%(output_file_path,feature_type_query,filename_save_annot)
				output_filename = '%s/test_%s_ad.%s.df_obs.txt'%(output_file_path,feature_type_query,filename_save_annot)
				pre_ad.obs.to_csv(output_filename,sep='\t')
				# output_filename = '%s/test_rna_df_var.%s.1.txt'%(output_file_path,data_file_type)
				# output_filename = '%s/test_%s_df_var.%s.1.txt'%(output_file_path,feature_type_query,filename_save_annot)
				output_filename = '%s/test_%s_ad.%s.df_var.txt'%(output_file_path,feature_type_query,filename_save_annot)
				pre_ad.var.to_csv(output_filename,sep='\t')
				# print(rna_ad.X)
				read_count = pre_ad.X
				read_count_1 = read_count.sum(axis=1)
				print(np.max(read_count_1),np.min(read_count_1),np.median(read_count_1),np.mean(read_count_1))
				
				# raw_counts = rna_ad.layers['raw_counts']
				# print('raw_counts ',raw_counts.shape)
				# print('raw_counts ',raw_counts)
				# print(raw_counts[0:10])
				# read_count1_1 = raw_counts.sum(axis=1)
				# print(np.max(read_count1_1),np.min(read_count1_1),np.median(read_count1_1),np.mean(read_count1_1))

				# log_counts = rna_ad.layers['log']
				# print('log_counts ',log_counts.shape)
				# print('log_counts ',log_counts)
				# read_count_log_1 = log_counts.sum(axis=1)
				# print(np.max(read_count_log_1),np.min(read_count_log_1),np.median(read_count_log_1),np.mean(read_count_log_1))

				# median_counts = rna_ad.layers['median']
				# print('median_counts ',median_counts[0:1])
				# print(median_counts.shape)

			if flag_plot>0:
				if 'celltype_colors' in pre_ad.uns:
					celltype_color_vec = pre_ad.uns['celltype_colors']
					# print('atac_ad ',celltype_color_vec,len(celltype_color_vec))
					print('celltype_color: ',celltype_color_vec,len(celltype_color_vec))
					# print(len(celltype_color_vec))
				# output_filename = '%s/test_umap_atac_%s.png'%(output_file_path,data_file_type)
				try:
					# sc.pl.scatter(atac_ad, basis='umap', color='celltype', legend_fontsize='xx-small', frameon=False)
					sc.pl.scatter(pre_ad, basis=feature_id, color=column_id1, legend_fontsize=10, frameon=False)
					# output_filename = '%s/test_umap_atac_%s.adata.1.png'%(output_file_path,data_file_type)
					# output_filename = '%s/test_umap_atac_%s.adata.1.1.png'%(output_file_path,data_file_type)
					# output_filename = '%s/test_%s_atac_%s.adata.1.1.png'%(output_file_path,feature_id,data_file_type)
					output_filename = '%s/test_%s_%s_%s.adata.1.1.png'%(output_file_path,feature_id,feature_type_query,filename_save_annot)
					plt.savefig(output_filename,format='png')
				except Exception as error:
					print('error! ',error)

			type_id_query = flag_query_1
			if flag_query_1>0:
				# prepare for the field raw_data
				if pre_ad.raw!=None:
					pre_ad_raw = pre_ad.raw.X
					read_count1 = pre_ad_raw.sum(axis=1)
					print(np.max(read_count1),np.min(read_count1),np.median(read_count1),np.mean(read_count1))
				else:
					if type_id_query==2:
						raw_ad = sc.AnnData(pre_ad.X)
						raw_ad.obs_names, raw_ad.var_names = pre_ad.obs_names, pre_ad.var_names
						pre_ad.raw = pre_ad
						# sample_id1 = atac_ad.obs_names
						# sample_id2 = sample_id1.str.split('#').str.get(1)
						# atac_ad.obs_names = sample_id2
						x0 = pre_ad.X
						x1 = pre_ad.raw.X
						difference_1=x0-x1
						t1 = np.max(np.max(np.abs(difference_1)))
						data_list1[i1] = pre_ad
						if verbose>0:
							print('pre_ad.raw\n')
							print(pre_ad.raw)
							print(pre_ad.raw.X[0:5,0:5])
							print('difference (1)',t1)
						if save_mode>0:
							# output_file_path_1 = input_file_path
							input_filename = input_filename_list[i1]
							b = input_filename.find('.h5ad')
							# output_filename = '%s/pbmc_multiome_atac.copy1.h5ad'%(output_file_path_1) # save the atac-seq data with raw counts
							output_filename = input_filename[0:b]+'.copy1.h5ad' # save the atac-seq data with raw counts
							if os.path.exists(output_filename)==False:
								pre_ad.write(output_filename)
								print('pre_ad\n',pre_ad)
							else:
								print('the file exists ',output_filename)
								input_filename=output_filename
								pre_ad_2 = sc.read_h5ad(input_filename)
								# x0 = atac_ad.X
								# x1 = atac_ad.raw.X
								x2 = pre_ad_2.raw.X
								# difference_1=x0-x2
								difference_2=x1-x2
								# t1 = np.max(np.max(np.abs(difference_1)))
								t2 = np.max(np.max(np.abs(difference_2)))
								if verbose>0:
									print('pre_ad_2.raw\n')
									print(pre_ad_2.raw)
									print(pre_ad_2.raw.X[0:5,0:5])
									print('difference (2)',t2)

			if flag_query_2>0:
				column_id_1 = 'SEACell'
				if not (column_id_1 in pre_ad.obs):
					column_id_1 = 'Metacell'
				if column_id_1 in pre_ad.obs:	
					SEACell_vec_1 = pre_ad.obs[column_id_1].unique() # the previously estimated SEACells
					SEACell_num1 = len(SEACell_vec_1)
					print('SEACell_num1 ',SEACell_num1)

		return data_list1
	
	# query data attributes
	def test_query_attribute_2(self,adata=[],query_type_vec=['obs','var'],save_mode=1,overwrite=0,output_file_path='',filename_prefix_save='',annot_vec=['df_obs','df_var'],verbose=0,select_config={}):

		adata1 = adata
		df_obs_1 = adata1.obs
		df_var_1 = adata1.var
		query_type_vec_1 = ['obs','var']
		list_query1 = [df_obs_1,df_var_1]
		dict_query1 = dict(zip(query_type_vec_1,list_query1))
		# list_query2 = []

		# annot_vec = ['df_obs','df_var']	
		# query_num2 = len(annot_vec)
		query_num2 = len(query_type_vec)
		for i2 in range(query_num2):
			query_type_1 = query_type_vec[i2]
			# df_annot_query = list_query1[i2]
			df_annot_query = dict_query1[query_type_1]

			annot_str = annot_vec[i2]
			# output_filename = '%s/test_query_%s_meta_ad.%s.log_normalize.%s.1.txt'%(output_file_path,feature_type_query,data_file_type_query,annot_str)
			output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_save,annot_str)
			if (os.path.exists(output_filename)==True) and (overwrite==0):
				print('the file exists: %s'%(output_filename))
				output_filename = '%s/%s.%s.copy1.1.txt'%(output_file_path,filename_prefix_save,annot_str)
			df_annot_query.to_csv(output_filename,sep='\t')

		data_list1 = list_query1
		return data_list1

	# metacell estimation
	def test_metacell_compute_pre1(self,flag_path_query=0,flag_SEACell_estimate=0,flag_summarize=0,overwrite=0,save_mode=1,output_file_path='',print_mode=1,verbose=0,select_config={}):

		input_file_path1 = self.save_path_1
		# rna_ad = sc.read_h5ad('/Users/sharmar1/Dropbox/multiome_jamboree/10x_pbmc/adata_rna.h5ad')
		# atac_ad = sc.read_h5ad('/Users/sharmar1/Dropbox/multiome_jamboree/10x_pbmc/adata_atac.h5ad')
		data_file_type = select_config['data_file_type']

		# if flag_path_query>0:
		# 	# select_config.update({'data_file_type':data_file_type})
		# 	select_config.update({'data_path':input_file_path})
		if flag_path_query>0:
			select_config = self.test_file_path_query_2(select_config=select_config)

		input_file_path = select_config['data_path']
		print('input_file_path ',input_file_path)
		# filename_prefix_1 = data_file_type
		# select_config.update({'filename_prefix_1':filename_prefix_1})
		# overwrite=0
		if 'overwrite' in select_config:
			overwrite = select_config['overwrite']
		else:
			select_config.update({'overwrite':overwrite})
		
		data_path = select_config['data_path']
		run_id = select_config['run_id']
		output_file_path_1 = data_path # save data that are not dependent on metacell estimations
		output_file_path = '%s/run%d'%(data_path,run_id) # save data that are dependent on metacell estimations
		if (flag_SEACell_estimate==0) and (flag_summarize>0):
			output_file_path_1 = output_file_path # load the previous estimation
		if os.path.exists(output_file_path)==False:
			print('the directory does not exist ',output_file_path)
			os.mkdir(output_file_path)
		else:
			print('the directory exists ',output_file_path)
			# overwrite = select_config['overwrite']
			if (overwrite==0) and (flag_SEACell_estimate>0):
				return

		input_filename_1 = select_config['input_filename_rna']  # the file_path of RNA-seq data
		input_filename_2 = select_config['input_filename_atac']	# the file_path of ATAC-seq data
		input_filename_list = [input_filename_1,input_filename_2]
		select_config.update({'data_path_save':output_file_path})
		
		rna_ad = sc.read_h5ad(input_filename_1)
		atac_ad = sc.read_h5ad(input_filename_2)
		
		data_list1 = [rna_ad,atac_ad]
		feature_type_vec = ['rna','atac']
		feature_num = len(feature_type_vec)
		data_list_1 = []

		feature_id = 'umap'
		column_id1 = 'celltype'
		color_id = column_id1
		filename_save_annot2 = data_file_type

		# print_mode=1
		data_list1 = self.test_query_attribute_1(data_list=data_list1,input_filename_list=input_filename_list,feature_type_vec=feature_type_vec,
													feature_id=feature_id,column_id_query=column_id1,flag_plot=1,flag_query_1=1,flag_query_2=2,
													save_mode=1,filename_save_annot=filename_save_annot2,output_file_path=output_file_path_1,
													print_mode=1,verbose=verbose,select_config=select_config)

		## retain the cells in both modalities
		rna_ad, atac_ad = data_list1[0:2]
		common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names)
		if len(common_cells) != atac_ad.shape[0]:
			print('Warning: The number of cells in RNA and ATAC objects are different. Only the common cells will be used.')
		print('common_cells ',len(common_cells))

		atac_ad_ori = atac_ad
		rna_ad_ori = rna_ad
		atac_ad = atac_ad[common_cells, :]
		rna_ad = rna_ad[common_cells, :]
		# print('rna_ad_1, atac_ad_1 ',rna_ad_1.shape,atac_ad_1.shape)
		print('rna_ad_ori, atac_ad_ori ',rna_ad_ori.shape,atac_ad_ori.shape)
		print('rna_ad, atac_ad ',rna_ad.shape,atac_ad.shape)

		flag_query_pre=0
		if ('flag_query_pre' in select_config):
			flag_query_pre=select_config['flag_query_pre']
		if flag_query_pre>0:
			return

		field_query = ['n_SEACells', 'obsm_build_kernel_rna', 'obsm_build_kernel_atac','num_components','n_waypoint_eigs','waypoint_proportion']
		# n_SEACells = 150
		# n_SEACells = 500
		n_SEACells = select_config['metacell_num']
		# default_param = [500,'X_pca','X_svd',50,10]
		default_param = [n_SEACells,'X_pca','X_svd',50,10,1.0]
		dict_param = dict(zip(field_query,default_param))
		list1 = default_param.copy()
		field_num = len(field_query)
		for i1 in range(field_num):
			field_id = field_query[i1]
			if field_id in select_config:
				list1[i1] = select_config[field_id]

		n_SEACells,obsm_build_kernel_rna,obsm_build_kernel_atac,num_components,n_waypoint_eigs,waypoint_proportion = list1
		# print(field_query)
		print(n_SEACells,obsm_build_kernel_rna,obsm_build_kernel_atac,num_components,n_waypoint_eigs,waypoint_proportion)

		# flag_SEACell_estimate=1
		# if 'flag_SEACell_estimate' in select_config:
		# 	flag_SEACell_estimate = select_config['flag_SEACell_estimate']
		# normalize, pca_compute = True, 1
		normalize, pca_compute = False, 0
		output_file_path = input_file_path
		feature_type = select_config['feature_type_1']
		run_id = select_config['run_id']
		type_id_feature = select_config['type_id_feature']
		# type_id_1=0
		type_id_1=type_id_feature

		filename_prefix_1 = '%s.%s.%d'%(data_file_type,feature_type,run_id)
		# filename_prefix_save = '%s_multiome_%s.%d'%(data_file_type,feature_type,run_id)
		filename_prefix_save = '%s_multiome.%s.%d'%(data_file_type,feature_type,run_id)
		filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_1,run_id)

		select_config.update({'filename_prefix_1':filename_prefix_1,
								'filename_prefix_save':filename_prefix_save,
								'filename_save_annot_1':filename_save_annot_1})

		# data_path = select_config['data_path']
		# run_id = select_config['run_id']
		# output_file_path = '%s/run%d'%(data_path,run_id)
		# if os.path.exists(output_file_path)==False:
		# 	print('the directory does not exist ',output_file_path)
		# 	os.mkdir(output_file_path)
		# else:
		# 	print('the directory exists ',output_file_path)
		# 	overwrite = select_config['overwrite']
		# 	if overwrite==0:
		# 		return
		# select_config.update({'data_path_save':output_file_path})
		
		output_file_path = select_config['data_path_save']
		print('output_file_path ',output_file_path)
		if flag_SEACell_estimate>0:
			min_iter=10
			max_iter=200
			select_config.update({'min_iter':min_iter,'max_iter':max_iter})
			
			# feature_type = 'rna'
			# feature_type = 'atac'
			# feature_type = select_config['feature_type_1']
			# output_file_path = input_file_path
			# filename_prefix_1 = '%s.%s'%(data_file_type,feature_type)
			# select_config.update({'filename_prefix_1':filename_prefix_1})
			print('data_file_type, feature_type ',data_file_type,feature_type)
			
			np.random.seed(0)
			if feature_type=='rna':
				if 'SEACell' in rna_ad.obs:
					rna_ad.obs['SEACell_1'] = rna_ad.obs['SEACell'].copy()
				num_components_pca = num_components
				rna_ad_1, model_1 = self.test_metacell_compute_unit_1(adata=rna_ad,normalize=normalize,n_SEACells=n_SEACells,obsm_build_kernel=obsm_build_kernel_rna,
																		pca_compute=pca_compute,num_components=num_components_pca,n_waypoint_eigs=n_waypoint_eigs,
																		waypoint_proportion=waypoint_proportion,
																		plot_convergence=1,select_config=select_config)
				# output_filename = '%s/adata_rna.%s.1.h5ad'%(output_file_path,data_file_type)
				output_filename = '%s/%s.copy1.h5ad'%(output_file_path,filename_prefix_save)
				rna_ad_1.write(output_filename)
				rna_ad = rna_ad_1
				print('rna_ad_1\n',rna_ad_1)
				save_filename = '%s/model_rna_%s.h5'%(output_file_path,filename_prefix_save)
				with open(save_filename,'wb') as output_file:
					pickle.dump(model_1,output_file)
			else:
				if 'SEACell' in atac_ad.obs:
					atac_ad.obs['SEACell_1'] = atac_ad.obs['SEACell'].copy()
				num_components_atac = num_components
				atac_ad_1, model_1 = self.test_metacell_compute_unit_1(adata=atac_ad,normalize=normalize,n_SEACells=n_SEACells,obsm_build_kernel=obsm_build_kernel_atac,
																		pca_compute=0,num_components=num_components_atac,n_waypoint_eigs=n_waypoint_eigs,
																		waypoint_proportion=waypoint_proportion,
																		plot_convergence=1,select_config=select_config)

				# output_filename = '%s/adata_atac.%s.1.h5ad'%(output_file_path,data_file_type)
				output_filename = '%s/%s.copy1.h5ad'%(output_file_path,filename_prefix_save)
				atac_ad_1.write(output_filename)
				atac_ad = atac_ad_1
				print('atac_ad_1\n',atac_ad_1)
				save_filename = '%s/model_atac_%s.h5'%(output_file_path,filename_prefix_save)
				with open(save_filename,'wb') as output_file:
					pickle.dump(model_1,output_file)
			# return

		# flag_summarize=1
		# flag_summarize=0
		# if ('flag_summarize' in select_config):
		# 	flag_summarize = select_config['flag_summarize']
		# data_timepoint = select_config['data_timepoint']
		if flag_summarize>0:
			# atac_meta_ad, rna_meta_ad = SEACells.genescores.prepare_multiome_anndata(atac_ad, rna_ad, SEACells_label='SEACell')
			summarize_layer_type = 'raw'
			# summarize_layer_type = 'X'
			# if summarize_layer_type=='raw':
			# 	try:
			# 		rna_ad.X = rna.raw.X  # raw count
			# 		print('rna_ad raw ', rna.raw.X.toarray()[0:10,0:10])
			# 	except Exception as error:
			# 		print('error!', error)
			# 		print('rna_ad raw ', rna_ad.raw)

			# 	try:
			# 		atac_ad.X = atac.raw.X  # raw count
			# 		print('atac_ad raw ', atac.raw.X.toarray()[0:10,0:10])
			# 	except Exception as error:
			# 		print('error!', error)
			# 		print('atac_ad raw ', rna_ad.raw)

			common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names)
			if len(common_cells) != atac_ad.shape[0]:
				print('Warning: The number of cells in RNA and ATAC objects are different. Only the common cells will be used.')
			print('common_cells ',len(common_cells))

			## use SEACells estimated on RNA-seq data for SEACell assignment on ATAC-seq data
			if 'SEACell' in atac_ad.obs:
				atac_ad.obs['SEACell_ori'] = atac_ad.obs['SEACell'].copy()
			if 'SEACell' in rna_ad.obs:
				rna_ad.obs['SEACell_ori'] = rna_ad.obs['SEACell'].copy()
			atac_ad_1 = atac_ad[common_cells, :]
			rna_ad_1 = rna_ad[common_cells, :]
			print('rna_ad_1, atac_ad_1 ',rna_ad_1.shape,atac_ad_1.shape)

			type_id_feature = select_config['type_id_feature']
			# type_id_1=0
			type_id_1=type_id_feature
			if type_id_1==0:
				atac_ad_1.obs['SEACell'] = rna_ad_1.obs['SEACell'].copy()
			else:
				rna_ad_1.obs['SEACell'] = atac_ad_1.obs['SEACell'].copy()
			
			# rna_ad_1.obs['celltype_1'] = atac_ad_1.obs['celltype'].copy()
			rna_ad_1.obs['%s_1'%(column_id1)] = atac_ad_1.obs[column_id1].copy()
			
			# summarize_layer_type_1 = 'raw_counts'	# rna data
			summarize_layer_type_1 = 'raw'	# rna data
			if atac_ad.raw==None:
				summarize_layer_type_2 = 'X'	# atac data
			else:
				summarize_layer_type_2 = 'raw'	# atac data
				print(atac_ad.raw)
				print(atac_ad.raw.X)
			print('summarize_layer_type_1, summarize_layer_type_2',summarize_layer_type_1,summarize_layer_type_2)
			
			atac_meta_ad, rna_meta_ad = self._prepare_multiome_anndata(atac_ad=atac_ad_1, rna_ad=rna_ad_1, SEACells_label='SEACell', summarize_layer_atac=summarize_layer_type_2,summarize_layer_rna=summarize_layer_type_1,type_id_1=type_id_1)
			# print('atac_meta_ad\n', atac_meta_ad)
			# print('rna_meta_ad\n', rna_meta_ad)

			## select highly variable peaks or highly variable genes
			highly_variable_feature_query_type = True
			rna_meta_ad = self.test_query_metacell_high_variable_feature(pre_meta_ad=rna_meta_ad,highly_variable_feature_query_type=highly_variable_feature_query_type,select_config=select_config)
			print('atac_meta_ad\n', atac_meta_ad)
			print('rna_meta_ad\n', rna_meta_ad)

			save_mode = 1
			# output_file_path = input_file_path
			output_file_path = select_config['data_path_save']
			run_id = select_config['run_id']
			filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_1,run_id)
			
			if save_mode>0:
				# if output_file_path=='':
				# 	output_file_path = self.save_path_1
				output_filename_1 = '%s/test_atac_meta_ad.%s.h5ad'%(output_file_path,filename_save_annot_1)
				atac_meta_ad.write(output_filename_1)
				output_filename_2 = '%s/test_rna_meta_ad.%s.h5ad'%(output_file_path,filename_save_annot_1)
				rna_meta_ad.write(output_filename_2)
				select_config.update({'filename_rna':output_filename_1,'filename_atac':output_filename_2})

			# output_file_path = self.save_path_1
			# filename_prefix = 'test_rna_meta_ad.%s.%d'%(data_file_type,type_id_1)
			filename_prefix = 'test_rna_meta_ad.%s'%(filename_save_annot_1)
			scale_type_id = 2
			select_config.update({'scale_type_id':scale_type_id})
			pre_meta_ad_rna, pre_meta_ad_scaled_rna = self.test_metacell_compute_unit_2(pre_meta_ad=rna_meta_ad,
																						save_mode=1,output_file_path=output_file_path,
																						filename_prefix=filename_prefix,
																						select_config=select_config)

			self.atac_meta_ad = atac_meta_ad
			self.rna_meta_ad = rna_meta_ad
			self.select_config = select_config

			# output_filename = input_filename
			save_mode1 = 1
			rna_meta_ad_scaled = pre_meta_ad_scaled_rna
			meta_scaled_exprs = pd.DataFrame(index=rna_meta_ad_scaled.obs_names,columns=rna_meta_ad_scaled.var_names,
												data=rna_meta_ad_scaled.X.toarray(),dtype=np.float32)
			self.meta_scaled_exprs = meta_scaled_exprs
			if save_mode1>0:
				output_filename = '%s/%s.meta_scaled_exprs.%d.txt'%(output_file_path,filename_prefix,scale_type_id)
				meta_scaled_exprs.to_csv(output_filename,sep='\t',float_format='%.6E')
				# print('meta_scaled_exprs ',meta_scaled_exprs.shape)

			meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
			# sample_id_1 = meta_exprs_2.index
			# assert list(sample_id_1)==list(peak_read.index)
			# self.meta_exprs_2_ori = meta_exprs_2
			# meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			self.meta_exprs_2 = meta_exprs_2
			print('meta_exprs_2 ', meta_exprs_2.shape)

			return atac_meta_ad, rna_meta_ad

	# metacell estimation
	def test_metacell_compute_pre2(self,data_timepoint='',feature_id='',flag_path_query=0,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
		
		if flag_path_query>0:
			select_config = self.test_file_path_query_2(select_config=select_config)

		feature_type_vec = ['rna','atac']
		input_filename_list1 = [select_config['input_filename_%s'%(feature_type_query)] for feature_type_query in feature_type_vec] # filename for origiinal single cell data
		input_filename_list2 = [select_config['filename_%s'%(feature_type_query)] for feature_type_query in feature_type_vec] # filename for metacell data
		
		input_filename_2 = input_filename_list2[0]
		print('input_filename_2 ',input_filename_2)
		pre_meta_ad = sc.read_h5ad(input_filename_2)
		metacell_id = pre_meta_ad.obs_names
		metacell_num = len(metacell_id)
		print('pre_meta_ad ',pre_meta_ad.shape)
		print(pre_meta_ad)
		print('metacell_num ',metacell_num)
		
		# data_timepoint = select_config['data_timepoint']
		filename_save_annot2 = filename_save_annot
		for i1 in range(1):
			input_filename_1 = input_filename_list1[i1]
			feature_type = feature_type_vec[i1]
			print('input_filename_1 ',input_filename_1)
			pre_ad_ori = sc.read_h5ad(input_filename_1)
			print(pre_ad_ori)
			# print(rna_ad.X)
			print(pre_ad_ori.raw)
			# try:
			# 	print(rna_ad.raw.X)
			# except Exception as error:
			# 	print('error! ',error)

			# pre_ad = pre_ad_ori[pre_ad_ori.obs['DataTimepoint']=='%s#Multiome'%(data_timepoint),:]
			if data_timepoint!='':
				pre_ad = pre_ad_ori[pre_ad_ori.obs['DataTimepoint']=='%s#Multiome'%(data_timepoint),:]
			else:
				pre_ad = pre_ad_ori
			print(pre_ad)
			# feature_id = 'X_FDL_Harmony'
			save_filename = output_filename
			if output_filename=='':
				save_filename = '%s/test_%s_%s_metacell_plot.%s.1.png'%(output_file_path,feature_type,filename_prefix_save,filename_save_annot2)
			sample_query_id = metacell_id
			self.plot_1(pre_ad,plot_basis=feature_id,sample_query_id=sample_query_id,save_as=save_filename,show=True)

		return True

	# metacell estimation
	def test_metacell_compute_pre1_query1(self,save_mode=1,output_file_path='',select_config={}):

		input_file_path1 = self.save_path_1
		# rna_ad = sc.read_h5ad('/Users/sharmar1/Dropbox/multiome_jamboree/10x_pbmc/adata_rna.h5ad')
		# atac_ad = sc.read_h5ad('/Users/sharmar1/Dropbox/multiome_jamboree/10x_pbmc/adata_atac.h5ad')
		data_file_type = select_config['data_file_type']

		select_config = self.test_file_path_query_2(select_config=select_config)
		flag_SEACell_estimate=0
		if 'flag_SEACell_estimate' in select_config:
			flag_SEACell_estimate = select_config['flag_SEACell_estimate']
		flag_summarize=0
		if ('flag_summarize' in select_config):
			flag_summarize = select_config['flag_summarize']
		
		# print_mode=1
		data_path = select_config['data_path']
		run_id = select_config['run_id']
		type_id_feature = select_config['type_id_feature']
		feature_type_1 = select_config['feature_type_1']
		input_file_path = select_config['data_path']
		data_path_save = '%s/run%d'%(input_file_path,run_id)

		feature_type_vec = ['rna','atac']
		filename_save_annot = data_file_type
		if 'run_id_load' in select_config:
			run_id_load = select_config['run_id_load']
		else:
			run_id_load = run_id

		dict_query1 = dict()
		dict1 = dict()
		dict_query1.update({run_id_load:dict1})
		filename_save_annot2 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id_load)
		for i1 in range(2):
			# type_id_feature = i1
			feature_type_query = feature_type_vec[i1]
			## combine the annotation from single cell data and the annotation from the metacell data
			input_filename_1 = '%s/test_%s_ad.%s.df_obs.txt'%(input_file_path,feature_type_query,data_file_type)
			# the annotation from metacell data
			input_filename = '%s/test_%s_meta_ad.%s.h5ad'%(data_path_save,feature_type_query,filename_save_annot2)
			adata1 = sc.read_h5ad(input_filename)
			df_obs = adata1.obs
			sample_id1 = df_obs.index
			print('feature_type_query: %s, sample_id: %d'%(feature_type_query,len(sample_id1)))
			print(adata1)
			print('df_obs ',df_obs.shape)

			df_annot1 = pd.read_csv(input_filename_1,index_col=0,sep='\t')
			column_query = df_annot1.columns.difference(df_obs.columns,sort=False)
			df_annot2 = df_annot1.loc[sample_id1,column_query]
			df_obs_combine = pd.concat([df_obs,df_annot2],axis=1,join='outer',ignore_index=False)
			output_file_path = data_path_save
			# output_filename = '%s/test_%s_meta_ad_df_obs.%s.1.txt'%(output_file_path,feature_type_query,filename_save_annot2)
			output_filename = '%s/test_%s_meta_ad.%s.df_obs.1.txt'%(output_file_path,feature_type_query,filename_save_annot2)
			df_obs_combine.to_csv(output_filename,sep='\t') # the annotations of each metacell including the orignal annotations from the single cell data
			print('df_obs_combine ',df_obs_combine.shape)

			# output_filename = '%s/test_%s_meta_ad_df_var.%s.1.txt'%(output_file_path,feature_type_query,filename_save_annot2)
			output_filename = '%s/test_%s_meta_ad.%s.df_var.1.txt'%(output_file_path,feature_type_query,filename_save_annot2)
			df_var = adata1.var
			df_var.to_csv(output_filename,sep='\t')
			print('df_var ',df_var.shape)

			# input_filename_pre1 = '%s/CD34_bonemarrow_multiome_rna.6.copy1.h5ad'%(data_path_save,data_file_type,feature_type_query)
			if feature_type_query==feature_type_1:
				input_filename_pre1 = '%s/%s_multiome_%s.%d.copy1.h5ad'%(data_path_save,data_file_type,feature_type_query,run_id_load)
				pre_ad = sc.read(input_filename_pre1)
				df_obs_1 = pre_ad.obs
				column_id1 = 'celltype'
				column_id2 = 'SEACell'
				metacell_id_ori = df_obs_1[column_id2].unique()
				metacell_id1 = pd.Index(metacell_id_ori).intersection(sample_id1,sort=False)
				print(len(metacell_id_ori),metacell_id_ori[0:2])
				print(len(sample_id1),sample_id1[0:2])
				print(len(metacell_id1),metacell_id1[0:2])
				# assert list(np.unique(metacell_id_oir))==list(np.unique(sample_id1))
				celltype_query = df_obs_1[column_id1]
				metacell_id = sample_id1

				feature_type_query_2 = pd.Index(feature_type_vec).difference([feature_type_query])[0]
				print('feature_type_query_2 ',feature_type_query_2)
				# feature_type_query_2 = np.setdiff1d(feature_type_vec,feature_type_query)[0]
				# input_filename_pre2 = '%s/test_%s_df_obs.%s.1.txt'%(input_file_path,feature_type_query_2,data_file_type)
				input_filename_pre2 = '%s/test_%s_ad.%s.df_obs.txt'%(input_file_path,feature_type_query_2,data_file_type)
				df_obs_2 = pd.read_csv(input_filename_pre2,index_col=0,sep='\t')
				sample_id_2 = df_obs_2.index
				print('df_obs_2, feature_type_query_2 ',df_obs_2.shape,feature_type_query_2)
				sample_id2 = pd.Index(metacell_id).intersection(sample_id_2,sort=False)
				sample_id3 = pd.Index(metacell_id).difference(sample_id_2,sort=False)
				assert len(sample_id3)==0

				df_1 = df_obs_1.loc[metacell_id,[column_id1]]
				df_1['count'] = 1
				# df1 = df_1.groupby(by=['celltype']).sum()
				df1 = df_1.groupby(by=[column_id1]).sum()
				
				 #input_filename = '%s/test_%s_celltype_query.%s.1.txt'%(input_file_path,feature_type_1,data_file_type)
				input_filename = '%s/test_%s_ad.%s.celltype_query.1.txt'%(input_file_path,feature_type_1,data_file_type)
				df1_pre1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				df1_pre1 = df1_pre1.rename(columns={'count':'count_ori'})
				df1 = pd.concat([df1,df1_pre1],axis=1,join='outer',ignore_index=False)
				df1.loc[:,'ratio'] = df1['count']/df1['count_ori']

				output_file_path_1 = output_file_path
				output_filename = '%s/test_%s_meta_ad.%s.celltype_query.1.txt'%(output_file_path_1,feature_type_1,filename_save_annot2)
				# df1.to_csv(output_filename,sep='\t')

				metacell_num = len(metacell_id)
				sample_id_1 = pre_ad.obs_names
				# field_query = ['num1','num2','ratio_association','celltype']
				field_query_1 = ['num1','num2','ratio_association']
				df_query1 = pd.DataFrame(index=metacell_id,columns=field_query_1)
				for i2 in range(metacell_num):
					metacell_id_query = metacell_id[i2]
					celltype_query = pre_ad.obs.loc[metacell_id_query,column_id1]
					id1 = (pre_ad.obs[column_id2]==metacell_id_query)
					id2 = (pre_ad.obs.loc[id1,column_id1]==celltype_query)
					sample_id_query1 = sample_id_1[id1]
					sample_id_query2 = sample_id_query1[id2]
					num1, num2 = len(sample_id_query1), len(sample_id_query2)
					ratio = num2/num1
					df_query1.loc[metacell_id_query,field_query_1] = [num1,num2,ratio]
					# df_query1.loc[metacell_id_query,'celltype'] = celltype_query
				
				df_obs_combine_2 = pd.concat([df_obs_combine,df_query1],axis=1,join='outer',ignore_index=False)

				df_2 = df_obs_combine_2.loc[:,[column_id1,'num1','num2']]
				df2 = df_2.groupby(by=[column_id1]).sum()
				celltype_idvec_ori = df1.index
				celltype_idvec = celltype_idvec_ori.intersection(df2.index,sort=False)
				df1.loc[:,'count_recall'] = df2.loc[celltype_idvec,'num2']
				df1.loc[:,'count_association'] = df2.loc[celltype_idvec,'num1']

				df1.loc[:,'recall'] = df1['count_recall']/df1['count_ori']
				df1.loc[:,'precision'] = df1['count_recall']/df1['count_association']
				eps = 1E-12
				df1.loc[:,'F1'] = 2*df1['precision']*df1['recall']/(df1['precision']+df1['recall']+eps)
				df1.to_csv(output_filename,sep='\t')

				# output_filename = '%s/test_%s_meta_ad_df_obs.%s.1.copy1.txt'%(output_file_path,feature_type_query,filename_save_annot2)
				output_filename = '%s/test_%s_meta_ad.%s.df_obs.1.copy1.txt'%(output_file_path,feature_type_query,filename_save_annot2)
				df_obs_combine_2 = df_obs_combine_2.sort_values(by=['celltype','ratio_association'],ascending=[True,False])
				df_obs_combine_2.to_csv(output_filename,sep='\t')
				print('df_obs_combine_2 ',df_obs_combine_2.shape)

				quantile_vec_1 = [0.1,0.25,0.5,0.75,0.9]
				t_value_1 = utility_1.test_stat_1(df_obs_combine_2['ratio_association'],quantile_vec=quantile_vec_1)
				dict1 = dict_query1[run_id_load]
				field_query_1 = ['max','min','mean','median']+quantile_vec_1
				# df_value = pd.Series(index=field_query_1,data=t_value_1)
				df_value = pd.DataFrame(index=[run_id_load],columns=field_query_1,data=np.asarray(t_value_1)[np.newaxis,:])
				
				field_id1 = 'cellnum'
				field_id2 = 'ratio'
				field_id3 = ['recall','precision','F1']
				field_query_pre2 = ['max','min','mean','median']
				field_query_2 = ['%s_%s'%(field_id1,field_id_query) for field_id_query in field_query_pre2]
				field_query_3 = ['%s_%s'%(field_id2,field_id_query) for field_id_query in field_query_pre2]
				cell_num2 = df1['count']
				celltype_ratio = df1['ratio']
				# celltype_recall = df1['recall']
				df_value.loc[:,field_query_2] = utility_1.test_stat_1(np.asarray(cell_num2))
				df_value.loc[:,field_query_3] = utility_1.test_stat_1(np.asarray(celltype_ratio))
				for field_id in field_id3:
					# field_id = field_id3[i2]
					field_query_5 = ['%s_%s'%(field_id,field_id_query) for field_id_query in field_query_pre2]
					query_value = df1[field_id]
					df_value.loc[:,field_query_5] = utility_1.test_stat_1(np.asarray(query_value))
				
				df_value.loc[:,'recall_merge'] = df1['count_recall'].sum()/df1['count_ori'].sum()
				df_value.loc[:,'precision_merge'] = df1['count_recall'].sum()/df1['count_association'].sum()
				df_value.loc[:,'F1_merge'] = 2*df_value['precision_merge']*df_value['recall_merge']/(df_value['precision_merge']+df_value['recall_merge']+eps)
				dict1.update({feature_type_query:df_value})
				dict_query1.update({run_id_load:dict1})

		return dict_query1

	# metacell estimation
	def test_metacell_compute_pre1_query2(self,run_idvec=[1],save_mode=1,output_file_path='',flag_path_query=0,select_config={}):

		# input_file_path1 = self.save_path_1
		# rna_ad = sc.read_h5ad('adata_rna.h5ad')
		# atac_ad = sc.read_h5ad('adata_atac.h5ad')
		
		data_file_type = select_config['data_file_type']
		if flag_path_query>0:
			select_config = self.test_file_path_query_2(select_config=select_config)
		
		run_id_ori = select_config['run_id']
		feature_type_1 = select_config['feature_type_1']
		
		query_num1 = len(run_idvec)
		flag_id_query1 = 1
		select_config.update({'flag_id_query1':flag_id_query1})
		
		list1 = []
		type_id_1=1
		# type_id_1=0
		df_query1=[]
		if type_id_1==0:
			for i1 in range(query_num1):
			# for i1 in range(1):
				run_id1 = run_idvec[i1]
				select_config.update({'run_id':run_id1,'run_id_load':run_id1})
				print('run_id1 ',run_id1)
				dict_query1 = self.test_metacell_compute_pre1_query1(select_config=select_config)

				dict1 = dict_query1[run_id1]
				df_value = dict1[feature_type_1]
				list1.append(df_value)

			df_query1 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
			data_path = select_config['data_path']
			output_file_path = data_path
			# output_filename = '%s/test_%s_meta_ad_query_basic.%s.1.txt'%(output_file_path,feature_type_1,data_file_type)
			output_filename = '%s/test_%s_meta_ad_query_basic.%s.2.txt'%(output_file_path,feature_type_1,data_file_type)
			df_query1.to_csv(output_filename,sep='\t',float_format='%.6f')
			print('df_query1 ',df_query1.shape)

		else:
			query_num2 = len(run_idvec_pre1)
			for i1 in range(query_num2):
			# for i1 in range(1):
				run_id1 = run_idvec_pre1[i1]
				select_config.update({'run_id':run_id1})
				select_config.update({'run_id_load':run_id1})
				flag_count=1
				flag_annot=1
				flag_id_query1 = 0
				select_config.update({'flag_id_query1':flag_id_query1})
				file_type_vec=['rna','atac']
				# file_type_vec=['rna']
				self.test_gene_peak_query_correlation_gene_pre1_2_ori(file_type_vec=file_type_vec,
																		flag_count=flag_count,
																		flag_annot=flag_annot,
																		select_config=select_config)

				flag_count=1
				flag_annot=0
				flag_id_query1 = 1
				select_config.update({'flag_id_query1':flag_id_query1})
				# file_type_vec=['rna','atac']
				file_type_vec=['rna']
				self.test_gene_peak_query_correlation_gene_pre1_2_ori(file_type_vec=file_type_vec,
																		flag_count=flag_count,
																		flag_annot=flag_annot,
																		select_config=select_config)

		return df_query1

	## prepare peak accessibility and gene expression matrix
	def test_gene_peak_query_correlation_gene_pre1_2_ori(self,gene_query_vec=[],flag_count=0,flag_annot=0,file_type_vec=['rna','atac'],normalize_type_vec=[0,1,2],filename_prefix_save='',output_filename='',save_file_path='',annot_mode=1,save_mode=1,select_config={}):

		# file_path1 = self.save_path_1
		# input_file_path1 = self.save_path_1
		
		data_file_type = select_config['data_file_type']
		data_path = select_config['data_path']
		run_id = select_config['run_id']
		input_file_path = data_path
		
		data_path_save = '%s/run%d'%(input_file_path,run_id)
		# input_file_path2 = '%s/peak_local'%(input_file_path)
		input_file_path2 = '%s/peak_local'%(data_path_save)
		
		filename_annot_vec = ['rna','atac']
		type_id_feature = select_config['type_id_feature']

		filename_save_annot1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
		input_filename_1 = '%s/test_rna_meta_ad.%s.h5ad'%(data_path_save,filename_save_annot1)
		input_filename_2 = '%s/test_atac_meta_ad.%s.h5ad'%(data_path_save,filename_save_annot1)
		
		output_file_path = data_path_save
		flag_query_count=flag_count
		# normalize_type=1
		normalize_type=0
		if flag_query_count>0:
			input_filename_list = [input_filename_1,input_filename_2]
			# query_num1 = len(input_filename_list)
			# filename_annot_vec = ['rna','atac']
			filename_annot_vec = file_type_vec
			query_num1 = len(filename_annot_vec)

			# index_name_vec = ['geneID','peakID']
			index_name_vec = ['ENSEMBL','peakID']
			normalize_type_annot = ['raw_count','log_normalize','normalize']

			# for normalize_type in [0,1,2]:
			# for normalize_type in [2]:
			for normalize_type in normalize_type_vec:
				for i1 in range(query_num1):
					input_filename = input_filename_list[i1]
					feature_type_query = filename_annot_vec[i1]
					adata = sc.read_h5ad(input_filename)
					print(input_filename)
					print(adata)
					print(adata.X[0:10])
					print(adata.raw)
					print(adata.raw.X[0:10])
					if normalize_type==0:
						df_count = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=adata.raw.X.toarray())
					elif normalize_type==1:
						df_count = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=adata.X.toarray(),dtype=np.float32)
					else:
						# pre_meta_ad = adata.raw
						# pre_meta_ad.X = csr_matrix(pre_meta_ad.X)
						# raw_count = pre_meta_ad.X
						# raw_count = adata.layers['raw']
						# raw_count = adata.raw.X.toarray()
						# pre_meta_ad = sc.AnnData(raw_count)
						# pre_meta_ad.X = csr_matrix(pre_meta_ad.X)
						pre_meta_ad = sc.AnnData(adata.raw.X)
						raw_count = adata.raw.X
						pre_meta_ad.obs_names, pre_meta_ad.var_names = adata.raw.obs_names, adata.raw.var_names
						# pre_meta_ad.obs['n_counts'] = pre_meta_ad.X.sum(axis=1)
						print('pre_meta_ad, normalize_type ',pre_meta_ad,normalize_type)
						print('raw_count ',raw_count[0:5,0:10])
						print(pre_meta_ad.obs_names,pre_meta_ad.var_names)
						# pre_meta_ad = sc.pp.normalize_total(pre_meta_ad,inplace=False,key_added='n_counts')
						sc.pp.normalize_total(pre_meta_ad,inplace=True,key_added='n_counts')
						df_count = pd.DataFrame(index=pre_meta_ad.obs_names,columns=pre_meta_ad.var_names,data=pre_meta_ad.X.toarray(),dtype=np.float32)

					flag_id_query1=0
					if 'flag_id_query1' in select_config:
						flag_id_query1 = select_config['flag_id_query1']
					if flag_id_query1>0:
						# if i1==0:
						if feature_type_query=='rna':
							# input_filename_annot = '%s/Homo_sapiens.GRCh38.108.1.txt'%(self.save_path_1)
							# input_filename_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path)
							input_filename_annot = select_config['filename_gene_annot']

							df_gene_annot_ori = pd.read_csv(input_filename_annot,index_col=False,sep='\t')
							df_gene_annot_ori = df_gene_annot_ori.sort_values(by=['length'],ascending=False)
							df_gene_annot_ori = df_gene_annot_ori.drop_duplicates(subset=['gene_name'])
							df_gene_annot_ori.index = np.asarray(df_gene_annot_ori['gene_name'])
							# gene_query_name_1 = df_count.columns
							gene_query_name_ori = df_gene_annot_ori.index.unique()
							gene_query_name = df_count.columns.intersection(gene_query_name_ori,sort=False)
							gene_id_query = df_gene_annot_ori.loc[gene_query_name,'gene_id']
							# df_count.columns = gene_id_query
							if data_file_type in ['CD34_bonemarrow']:
								df1 = df_gene_annot_ori.loc[gene_query_name,['gene_id','gene_name','gene_name_2','means','dispersions','dispersions_norm','length']]
							else:
								df1 = df_gene_annot_ori.loc[gene_query_name,['gene_id','gene_name','gene_name_ori','means','dispersions','dispersions_norm','length']]
							# df1 = df1.sort_values(by=['dispersions_norm'],ascending=False)
							
							df1 = df1.sort_values(by=['length'],ascending=False)
							df1['duplicated'] = df1.duplicated(subset=['gene_id'])
							# output_file_path = input_file_path
							output_filename = '%s/test_gene_name_query.1.txt'%(output_file_path)
							df1.to_csv(output_filename,sep='\t')
							gene_query_vec_1 = df1.index[df1['duplicated']==False]
							gene_query_vec_2 = df1.index[df1['duplicated']==True]
							print('gene_query_vec_1, gene_query_vec_2 ',len(gene_query_vec_1),len(gene_query_vec_2),gene_query_vec_2)
							df_count = df_count.loc[:,gene_query_vec_1]
							df_count_ori = df_count
							df_count.columns = df_gene_annot_ori.loc[gene_query_vec_1,'gene_id']
							print('df_count_ori, df_count ',df_count_ori.shape,df_count.shape)

					df_count = df_count.T
					df_count.index.name = index_name_vec[i1]
					print(df_count.min(axis=1))
					print(df_count.max(axis=1))
					print('df_count ',df_count.shape)
					filename_annot1 = filename_annot_vec[i1]
					filename_annot2 = normalize_type_annot[normalize_type]
					output_filename = '%s/test_%s_meta_ad.%s.%s.tsv.gz'%(output_file_path,filename_annot1,filename_save_annot1,filename_annot2)
					if (flag_id_query1==0) and (feature_type_query=='rna'):
						output_filename = '%s/test_%s_meta_ad.%s.%s.ori.tsv.gz'%(output_file_path,filename_annot1,filename_save_annot1,filename_annot2)
					if normalize_type==0:
						df_count.to_csv(output_filename,sep='\t',float_format='%d',compression='gzip')
					else:
						df_count.to_csv(output_filename,sep='\t',compression='gzip')

		flag_annot_1=flag_annot
		if flag_annot_1>0:
			# feature_type_1, feature_type_2 = 'rna','atac'
			feature_type_vec = ['rna','atac']
			feature_type_1 = select_config['feature_type_1']
			feature_type_2 = pd.Index(feature_type_vec).difference([feature_type_1],sort=False)[0]
			input_filename_query1 = '%s/test_%s_df_obs.%s.1.txt'%(input_file_path,feature_type_1,data_file_type)
			input_filename_query2 = '%s/test_%s_df_obs.%s.1.txt'%(input_file_path,feature_type_2,data_file_type)
			# input_filename_2 = '%s/test_%s_meta_ad.%s.df_obs.1.txt'%(data_path_save,feature_type_1,data_file_type)
			input_filename_2 = '%s/test_%s_meta_ad_df_obs.%s.1.txt'%(data_path_save,feature_type_1,filename_save_annot1)

			# df1_rna = pd.read_csv(input_filename_rna,index_col=0,sep='\t')
			# df1_atac = pd.read_csv(input_filename_atac,index_col=0,sep='\t')
			df_feature1 = pd.read_csv(input_filename_query1,index_col=0,sep='\t')
			df_feature2 = pd.read_csv(input_filename_query2,index_col=0,sep='\t')
			df2 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			sample_id = df2.index

			# sample_id1 = df1_rna.index
			# sample_id2 = df1_atac.index
			sample_id1 = df_feature1.index
			sample_id2 = df_feature2.index
			sample_id_query = sample_id1.intersection(sample_id2,sort=False)
			print('sample_id_query ',len(sample_id_query))

			sample_id1_query1 = sample_id.intersection(sample_id1,sort=False)
			sample_id1_query2 = sample_id.difference(sample_id1,sort=False)
			print('sample_id1_query1, sample_id1_query2 ',len(sample_id1_query1),len(sample_id1_query2))
			assert len(sample_id1_query2)==0

			# field_query1 = ['phenograph','celltype']
			field_query1 = df_feature1.columns.difference(df2.columns,sort=False)
			df2.loc[sample_id1_query1,field_query1] = df_feature1.loc[sample_id1_query1,field_query1]
			df2 = df2.rename(columns={'celltype':'celltype_%s'%(feature_type_1)})

			sample_id2_query1 = sample_id.intersection(sample_id2,sort=False)
			sample_id2_query2 = sample_id.difference(sample_id2,sort=False)
			print('sample_id2_query1, sample_id2_query2 ',len(sample_id2_query1),len(sample_id2_query2))
			assert len(sample_id2_query2)==0

			# df1_atac = df1_atac.rename(columns={'SEACell':'SEACell_atac','SEACell_1':'SEACell_1_atac'})
			# field_query2 = df1_atac.columns.difference(df2.columns,sort=False)
			field_query2 = df_feature2.columns.difference(df2.columns,sort=False)
			df2.loc[sample_id2_query1,field_query2] = df_feature2.loc[sample_id2_query1,field_query2]

			# output_file_path=input_file_path
			output_filename = '%s/sampleMetadata.%s.tsv.gz'%(output_file_path,filename_save_annot1)
			df2.index.name='sample_id'
			df2.to_csv(output_filename,sep='\t',compression='gzip')
			# print('df1_rna, df1_atac, df2 ',df1_rna.shape,df1_atac.shape,df2.shape)
			print('df_feature1, df_feature2, df2 ',df_feature1.shape,df_feature2.shape,df2.shape)

		return True

	## query gene annotations
	def test_gene_annotation_query_pre1(self,flag_query1=0,flag_query2=0,flag_query3=0,select_config={}):

		## gene name query and matching between the gene annotation file and gene name in the gene expression file
		flag_gene_annot_query_1=flag_query1
		df_gene_annot1, df_gene_annot2, df_gene_annot3 = [], [], []
		if flag_gene_annot_query_1>0:	
			filename_prefix_1 = 'hg38'
			filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
			# gene_annotation_filename = '%s/hg38.1.txt'%(input_file_path1)
			# gene_annotation_filename = '%s/Homo_sapiens.GRCh38.108.1.txt'%(input_file_path1)
			gene_annotation_filename = '%s/%s.1.txt'%(input_file_path1,filename_prefix_1)
			if len(self.df_gene_annot_expr)==0:
				df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2 = self.test_motif_peak_estimate_gene_annot_load_1(input_filename=gene_annotation_filename,select_config=select_config)
				self.df_gene_annot_ori = df_gene_annot_ori
				self.df_gene_annot_expr = df_gene_annot_expr

				output_file_path1 = self.save_path_1
				output_file_path2 = input_file_path
				output_file_path_list = [output_file_path1,output_file_path2,output_file_path2]
				filename_prefix_list = ['ori','expr','expr_2']
				df_annot_list = [df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2]
				for i1 in range(3):
					output_filename = '%s/test_gene_annot_%s.%s.1.txt'%(output_file_path_list[i1],filename_prefix_list[i1],filename_prefix_1)
					df_annot_query = df_annot_list[i1]
					df_annot_query.to_csv(output_filename,index=False,sep='\t')

			df_gene_annot = self.df_gene_annot_ori
			df_gene_annot.index = np.asarray(df_gene_annot['gene_name'])

			print('df_gene_annot ',df_gene_annot.shape)
			print(df_gene_annot)
			df_gene_annot1 = df_gene_annot

		flag_gene_annot_query_2=flag_query2
		if flag_gene_annot_query_2>0:
			## gene_name annotation
			filename_prefix_1 = 'hg38'
			filename_prefix_2 = 'Homo_sapiens.GRCh38.108'
			input_filename_1 = '%s/test_gene_annot_expr.%s.1.txt'%(input_file_path,filename_prefix_1)
			input_filename_1_2 = '%s/test_gene_annot_expr_2.%s.1.txt'%(input_file_path,filename_prefix_1)
			input_filename_2 = '%s/test_gene_annot_expr.%s.1.txt'%(input_file_path,filename_prefix_2)
			input_filename_2_2 = '%s/test_gene_annot_expr_2.%s.1.txt'%(input_file_path,filename_prefix_2)
			input_filename_list = [input_filename_1,input_filename_1_2,input_filename_2,input_filename_2_2]
			list_query1 = []
			query_num1 = len(input_filename_list)
			for i1 in range(query_num1):
				input_filename = input_filename_list[i1]
				df_query_1 = pd.read_csv(input_filename,index_col=False,sep='\t')
				df_query_1.index = np.asarray(df_query_1['gene_name'])
				list_query1.append(df_query_1)

			df_gene_annot_expr_pre1, df_gene_annot_expr_2_pre1, df_gene_annot_expr_pre2, df_gene_annot_expr_2_pre2 = list_query1
			gene_name_query1 = df_gene_annot_expr_2_pre2.index.intersection(df_gene_annot_expr_pre1.index,sort=False)
			gene_name_query1_2 = df_gene_annot_expr_2_pre2.index.difference(df_gene_annot_expr_pre1.index,sort=False)
			df_gene_annot_expr_2_pre2_1 = df_gene_annot_expr_pre1.loc[gene_name_query1]
			df_gene_annot_expr_2_pre2_1['gene_version'] = 'hg38'
			df_gene_annot_expr_2_pre2_2 = df_gene_annot_expr_2_pre2.loc[gene_name_query1_2]
			df_annot1 = pd.concat([df_gene_annot_expr_pre2,df_gene_annot_expr_2_pre2_1],axis=0,join='outer',ignore_index=False)
			print('gene_name_query1, df_gene_annot_expr_2_pre2, df_annot1 ',len(gene_name_query1),df_gene_annot_expr_2_pre2.shape,df_annot1.shape)
			output_file_path = input_file_path
			output_filename = '%s/test_gene_annot_expr.%s.combine.1.txt'%(output_file_path,filename_prefix_2)
			df_annot1 = df_annot1.sort_values(by=['chrom','start','stop','gene_name'],ascending=True)
			df_annot1.to_csv(output_filename,index=False,sep='\t')
			output_filename = '%s/test_gene_annot_expr_2.%s.2.txt'%(output_file_path,filename_prefix_2)
			df_gene_annot_expr_2_pre2_2.to_csv(output_filename,index=False,sep='\t')
			print('df_gene_annot_expr_2_pre2_2 highly_variable ',np.sum(df_gene_annot_expr_2_pre2_2['highly_variable']))

			gene_name_query2 = df_gene_annot_expr_2_pre1.index.intersection(df_gene_annot_expr_pre2.index,sort=False)
			gene_name_query2_2 = df_gene_annot_expr_2_pre1.index.difference(df_gene_annot_expr_pre2.index,sort=False)
			df_gene_annot_expr_2_pre1_1 = df_gene_annot_expr_pre2.loc[gene_name_query2]
			df_gene_annot_expr_2_pre1_1['gene_version'] = 'GRCh38.108'
			df_gene_annot_expr_2_pre1_2 = df_gene_annot_expr_2_pre1.loc[gene_name_query2_2]
			df_annot2 = pd.concat([df_gene_annot_expr_pre1,df_gene_annot_expr_2_pre1_1],axis=0,join='outer',ignore_index=False)
			print('gene_name_query2, df_gene_annot_expr_2_pre1, df_annot2 ',len(gene_name_query2),df_gene_annot_expr_2_pre1.shape,df_annot2.shape)
			output_filename = '%s/test_gene_annot_expr.%s.combine.1.txt'%(output_file_path,filename_prefix_1)
			df_annot2 = df_annot2.sort_values(by=['chrom','start','stop','gene_name'],ascending=True)
			df_annot2.to_csv(output_filename,index=False,sep='\t')
			output_filename = '%s/test_gene_annot_expr_2.%s.2.txt'%(output_file_path,filename_prefix_1)
			df_gene_annot_expr_2_pre1_2.to_csv(output_filename,index=False,sep='\t')
			print('df_gene_annot_expr_2_pre1_2 highly_variable ',np.sum(df_gene_annot_expr_2_pre1_2['highly_variable']))

			df_gene_annot2 = df_annot2
			# return

		flag_gene_annot_query_3=flag_query3
		if flag_gene_annot_query_3>0:
			## gene_name annotation
			# filename_prefix_1 = 'hg38'
			# filename_prefix_2 = 'Homo_sapiens.GRCh38.108'
			filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
			input_filename_1 = '%s/test_gene_annot_expr.%s.combine.1.txt'%(input_file_path,filename_prefix_1)
			input_filename_1_2 = '%s/test_gene_annot_expr_2.%s.2.copy1.txt'%(input_file_path,filename_prefix_1)

			df1 = pd.read_csv(input_filename_1,index_col=False,sep='\t')
			df2 = pd.read_csv(input_filename_1_2,index_col=0,sep='\t')
			field_query_1 = df1.columns
			field_query_2 = df2.columns
			field_query = field_query_1.intersection(field_query_2,sort=False)
			df_query1 = df2.loc[:,field_query_2]
			df_query1['gene_id_1'] = df_query1.index.copy()
			df_query1['start'] = np.int32(df_query1['start'])
			df_query1['stop'] = np.int32(df_query1['stop'])
			id_1 = (pd.isna(df_query1['gene_id'])==True)
			id_2 = (~id_1)
			df_query2 = df_query1.loc[id_2,:]
			id1 = (df_query2['strand']=='-')
			id2 = (~id1)
			df_query2.loc[id1,'start_site'] = df_query2.loc[id1,'stop']
			df_query2.loc[id2,'start_site'] = df_query2.loc[id2,'start']
			df_query2['length'] = df_query2['stop']-df_query2['start']
			query_num_1 = df_query2.shape[0]
			query_id1 = df_query2.index
			# for i1 in range(query_num_1):
			# 	id_pre1 = query_id1[id1]
			# 	try:
			# 		length_1 = np.int32(df_query2.loc[id_pre1,'stop'])-np.int32(df_query2.loc[id_pre1,'start'])
			# 		df_query2.loc[id_pre1,'length'] = length_1
			# 		print('gene_id_1 ',id_pre1,i1,length_1)
			# 	except Exception as error:
			# 		print('error! ',error,id_pre1,i1)

			df_query_1 = pd.concat([df1,df_query2],axis=0,join='outer',ignore_index=False)
			print('df1, df2, df_query2 ',df1.shape,df2.shape,df_query2.shape)
			print('df_query_1 ',df_query_1.shape)
			id_query1 = (df_query_1.duplicated(subset=['gene_name']))
			query_num1 = np.sum(id_query1)
			print('df_query_1 gene_name duplicated ',query_num1)
			df_query_1_duplicated = df_query_1.loc[id_query1,:]
			print(df_query_1_duplicated)

			output_file_path = input_file_path
			output_filename = '%s/test_gene_annot_expr.%s.combine.2.txt'%(output_file_path,filename_prefix_1)
			df_query_1 = df_query_1.sort_values(by=['chrom','start','stop'],ascending=True)
			df_query_1.to_csv(output_filename,index=False,sep='\t')
			df_gene_annot3 = df_query_1

		return df_gene_annot1, df_gene_annot2, df_gene_annot3

	## query gene annotations
	def test_gene_annotation_query1(self,select_config={}):

		flag_gene_annot_query=1
		df_gene_annot_expr = []
		input_filename_gene_annot = ''
		data_path_1 = select_config['data_path_1']
		data_path_2 = select_config['data_path_2']
		if 'filename_gene_annot' in select_config:
			input_filename_gene_annot = select_config['filename_gene_annot']
		if flag_gene_annot_query>0:
			data_file_type = select_config['data_file_type']
			# input_file_path = data_path_1
			flag_query=1
			# if (data_file_type in ['mouse_endoderm','E8.75','E9.25']) or (self.species_id=='mm10'):
			if (self.species_id=='mm10'):
				if input_filename_gene_annot=='':
					filename_prefix_1 = 'Mus_musculus.GRCm38.102'
					# input_filename = 'test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.1.txt'
					# data_path = select_config['data_path']
					# input_file_path = data_path
					# input_filename_gene_annot = '%s/gene_annotation/test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.1.txt'%(input_file_path)
					input_file_path = data_path_2
					input_filename_gene_annot = '%s/gene_annotation/test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.txt'%(input_file_path)
					# input_filename_gene_annot = '%s/test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.1.txt'%(input_file_path)
				print('input_filename_gene_annot: ',input_filename_gene_annot)
				df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2 = self.test_motif_peak_estimate_gene_annot_load_1(input_filename=input_filename_gene_annot,
																																type_id_1=0,
																																select_config=select_config)
				# print('input_filename_gene_annot: %s'%(input_filename_gene_annot))
				# print('df_gene_annot_expr ',df_gene_annot_expr.shape)
				# self.df_gene_annot_ori = df_gene_annot_ori	# gene annotation
				# self.df_gene_annot_expr = df_gene_annot_expr   # gene query with expression
			
			elif (self.species_id=='hg38'):
				input_file_path = data_path_1
				if input_filename_gene_annot=='':
					# filename_prefix_1 = 'hg38'
					filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
					# gene_annotation_filename = '%s/hg38.1.txt'%(input_file_path1)
					# gene_annotation_filename = '%s/Homo_sapiens.GRCh38.108.1.txt'%(input_file_path1)
					# gene_annotation_filename = '%s/%s.1.txt'%(input_file_path1,filename_prefix_1)
					# input_file_path = data_path_1
					input_filename_gene_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path)
				print('input_filename_gene_annot: ',input_filename_gene_annot)
				df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2 = self.test_motif_peak_estimate_gene_annot_load_2(input_filename=input_filename_gene_annot,
																																input_file_path=input_file_path,
																																select_config=select_config)
				# print('input_filename_gene_annot: %s'%(input_filename_gene_annot))
				# print('df_gene_annot_expr ',df_gene_annot_expr.shape)
				# self.df_gene_annot_ori = df_gene_annot_ori	# gene annotation
				# self.df_gene_annot_expr = df_gene_annot_expr
			else:
				print('please provide gene annotations')
				flag_query=0
				pass

			if (flag_query>0):
				print('input_filename_gene_annot: %s'%(input_filename_gene_annot))
				print('df_gene_annot_expr ',df_gene_annot_expr.shape)
				self.df_gene_annot_ori = df_gene_annot_ori	# gene annotation
				self.df_gene_annot_expr = df_gene_annot_expr   # gene query with expression
				
			return df_gene_annot_expr

	## gene-peak association query: genome annotation load
	# input: the genome annotation filename; if not given, use default genome annotation filename
	# output: genome annotation (dataframe); gene position, strand, tss, transcript num
	# dataframe1: genes original; dataframe 2: genes with expr in the dataset
	def test_motif_peak_estimate_gene_annot_load_1(self,input_filename='',input_filename_2='',type_id_1=0,flag_query_2=0,select_config={}):

		input_file_path = self.save_path_1
		print('load gene annotations')
		# input_filename_annot_1 = '%s/gene_annotation/test_Mus_musculus.GRCm38.102.txt'%(file_path1)
		# if input_filename=='':
		# 	# input_filename = '%s/gene_annotation/test_Mus_musculus.GRCm38.102.annotations_mm38.pre1.txt'%(input_file_path)
		# 	# input_filename = '%s/gene_annotation/test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.1.txt'%(input_file_path)
		# 	input_filename = '%s/hg38.1.txt'%(input_file_path)
		df_gene_annot_ori = pd.read_csv(input_filename,index_col=False,sep='\t')
		df_gene_annot_ori = df_gene_annot_ori.sort_values(by=['length'],ascending=False)
		rna_meta_ad = self.rna_meta_ad
		meta_scaled_exprs = self.meta_scaled_exprs
		# gene_name_expr = self.meta_scaled_exprs.columns
		# gene_name_expr = meta_scaled_exprs.columns
		if type_id_1==0:
			## gene query by gene name; keep gene name unduplicated
			df_gene_annot = df_gene_annot_ori.drop_duplicates(subset=['gene_name'])
			df_gene_annot.index = np.asarray(df_gene_annot['gene_name'])
			df_gene_annot = df_gene_annot.sort_values(by=['chrom','start','stop','gene_name'],ascending=True)
			print('gene annotation load 1 ',df_gene_annot_ori.shape, df_gene_annot.shape)

			# gene_name_expr = meta_scaled_exprs.columns
			gene_name_expr = rna_meta_ad.var_names
			gene_query = gene_name_expr
			gene_query_1 = pd.Index(gene_query).intersection(df_gene_annot.index,sort=False) # genes with expr in the dataset
			gene_query_2 = pd.Index(gene_query).difference(df_gene_annot.index,sort=False)
			df_gene_annot_expr = df_gene_annot.loc[gene_query_1,:]
			field_query = ['highly_variable', 'means', 'dispersions', 'dispersions_norm']

			df_var = rna_meta_ad.var
			print('df_var: ',df_var.shape)
			print(df_var.columns)

			column_vec_1 = df_var.columns
			column_vec_2 = pd.Index(field_query).intersection(column_vec_1,sort=False)

			load_mode = 0
			if list(column_vec_2)==list(field_query):
				load_mode = 1
			
			if load_mode>0:
				df_gene_annot_expr.loc[:,field_query] = df_var.loc[gene_query_1,field_query]
				gene_num1 = np.sum(df_gene_annot_expr['highly_variable']==True)
			
			meta_exprs_2 = self.meta_exprs_2
			df_gene_annot_expr_2 = pd.DataFrame(index=gene_query_2,columns=['gene_name','meta_scaled_exprs_max','meta_exprs_max'],dtype=np.float32)
			df_gene_annot_expr_2['gene_name'] = np.asarray(df_gene_annot_expr_2.index)
			if len(meta_scaled_exprs)>0:
				df_gene_annot_expr_2['meta_scaled_exprs_max'] = meta_scaled_exprs.loc[:,gene_query_2].max(axis=0)
			if len(meta_exprs_2)>0:
				df_gene_annot_expr_2['meta_exprs_max'] = meta_exprs_2.loc[:,gene_query_2].max(axis=0)
			
			if load_mode>0:
				df_gene_annot_expr_2.loc[gene_query_2,field_query] = df_var.loc[gene_query_2,field_query]
				gene_num2 = np.sum(df_gene_annot_expr_2['highly_variable']==True)
				gene_name_expr_1, gene_name_expr_2 = gene_query_1, gene_query_2

		else:
			## gene query by gene id; gene id are supposed to be unduplicated
			df_gene_annot = df_gene_annot_ori
			df_gene_annot.index = np.asarray(df_gene_annot['gene_id'])
			df_annot1 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			# df_annot1['gene_name'] = df_annot1.index.copy()
			# df_annot1.index = np.asarray(df_annot1['gene_id'])
			df_var = rna_meta_ad.var
			gene_name_expr = np.asarray(df_var.index)
			df_var['gene_name'] = df_var.index.copy()
			df_var['gene_ids'] = df_annot1.loc[gene_name_expr,'gene_ids']
			df_var.index = np.asarray(df_var['gene_ids'])
			query_num2 = np.sum(df_var.duplicated(subset=['gene_name']))
			print('gene query name 2 ',query_num2)

			gene_query = np.asarray(df_var['gene_ids'])
			gene_query_1 = pd.Index(gene_query).intersection(df_gene_annot.index,sort=False) # genes with expr in the dataset
			gene_query_2 = pd.Index(gene_query).difference(df_gene_annot.index,sort=False)
			df_gene_annot_expr = df_gene_annot.loc[gene_query_1,:]
			df_gene_annot_expr['gene_name_ori'] = df_gene_annot_expr['gene_name'].copy()
			df_gene_annot_expr.loc[gene_query_1,'gene_name'] = np.asarray(df_var.loc[gene_query_1,'gene_name'])
			
			field_query = ['highly_variable', 'means', 'dispersions', 'dispersions_norm']

			column_vec_1 = df_var.columns
			column_vec_2 = pd.Index(field_query).intersection(column_vec_1,sort=False)
			
			load_mode = 0
			if list(column_vec_2)==list(field_query):
				load_mode = 1
			
			if load_mode>0:			
				df_gene_annot_expr.loc[gene_query_1,field_query] = df_var.loc[gene_query_1,field_query]
				gene_num1 = np.sum(df_gene_annot_expr['highly_variable']==True)

			meta_exprs_2 = self.meta_exprs_2
			df_gene_annot_expr_2 = pd.DataFrame(index=gene_query_2,columns=['gene_name','meta_scaled_exprs_max','meta_exprs_max'],dtype=np.float32)
			
			# df_gene_annot_expr_2['gene_name'] = np.asarray(df_gene_annot_expr_2.index)
			df_gene_annot_expr_2['gene_name'] = np.asarray(df_var.loc[gene_query_2,'gene_name'])

			if load_mode>0:
				df_gene_annot_expr_2.loc[gene_query_2,field_query] = df_var.loc[gene_query_2,field_query]
				gene_num2 = np.sum(df_gene_annot_expr_2['highly_variable']==True)

			if flag_query_2>0:
				# query_id_ori_1 = meta_scaled_exprs.columns.copy()
				# query_id_ori_2 = meta_exprs_2.columns.copy()
				if len(meta_scaled_exprs)>0:
					meta_scaled_exprs.columns = df_var.index.copy()
					df_gene_annot_expr_2['meta_scaled_exprs_max'] = meta_scaled_exprs.loc[:,gene_query_2].max(axis=0)
				if len(meta_exprs_2)>0:
					meta_exprs_2.columns = df_var.index.copy()
					df_gene_annot_expr_2['meta_exprs_max'] = meta_exprs_2.loc[:,gene_query_2].max(axis=0)
				# df_gene_annot_expr_2.loc[gene_query_2,field_query] = df_var.loc[gene_query_2,field_query]
				# gene_num2 = np.sum(df_gene_annot_expr_2['highly_variable']==True)
				# meta_exprs_2.columns = np.asarray(df_var['gene_name'])

			# meta_scaled_exprs.columns = np.asarray(df_var['gene_name'])
			gene_name_expr_1 = df_var.loc[gene_query_1,'gene_name']
			gene_name_expr_2 = df_var.loc[gene_query_2,'gene_name']

		## query how many genes are highly variable but not match the gene name in the used gene annotation version
		gene_query_2 = df_gene_annot_expr_2.index
		
		if load_mode>0:
			gene_query_2_highly_variable = gene_query_2[df_gene_annot_expr_2['highly_variable']==True]
		else:
			print('gene_highly_variable not included ')
			gene_query_2_highly_variable = []
		print('gene_name_expr, gene_name_expr_1, gene_name_expr_2 ', len(gene_name_expr), len(gene_name_expr_1), len(gene_name_expr_2))
		print('gene_name_expr_1, highly_variable ',len(gene_name_expr_1),gene_name_expr_1[0:5],gene_num1)
		print('gene_name_expr_2, highly_variable ',len(gene_name_expr_2),gene_name_expr_2[0:5],gene_num2,gene_query_2_highly_variable)
		print('gene annotation load 1, df_gene_annot_expr ', df_gene_annot_expr.shape)

		return df_gene_annot, df_gene_annot_expr, df_gene_annot_expr_2

	## gene-peak association query: gene annotation load
	# input: the gene annotation filenames; if not given, use default gene annotation filenames
	# output: gene annotation (dataframe); gene features in the dataset
	def test_motif_peak_estimate_gene_annot_load_2(self,input_filename='',input_file_path='',select_config={}):

		input_file_path1 = self.save_path_1
		## load gene query annotations
		print('load gene query annotations')
		try:
			# self.df_gene_annot = self.pre_rna_ad_var
			# filename_1 = '%s/test_E8.75#Multiome_meta_rna.df_var.pre1.2.txt'%(input_file_path)
			# filename_2 = '%s/test_E8.75#Multiome_meta_rna.normalize.log1.df_var.1.0.txt'%(input_file_path)
			# df_1 = pd.read_csv(filename_1,index_col=0,sep='\t')
			# df_2 = pd.read_csv(filename_2,index_col=0,sep='\t')
			# df_2.columns=['n_cells_2','highly_variable1','means','dispersions','dispersions_norm']
			# query_id1 = df_1.index.intersection(df_2.index,sort=False)
			# df_1 = df_1.loc[query_id1,:]
			# data_2 = pd.concat([df_1,df_2],axis=1,join='outer',ignore_index=False,
			# 					keys=None, levels=None, names=None, verify_integrity=True, copy=True)

			# df_gene_annot_expr = self.df_gene_annot_expr
			# if len(df_gene_annot_expr)==0:
			# 	df_gene_annot, df_gene_annot_expr = self.test_motif_peak_estimate_gene_annot_load_1(input_filename='',select_config=select_config)
			# # data_2['tf_id'] = 0
			# # gene_query_vec = query_id1.intersection(df_gene_query_expr.index,sort=False)
			# # data_2.loc[gene_query_vec,'tf_id'] = df_gene_query_expr.loc[gene_query_vec,'tf_id']
			# data_2 = pd.concat([df_gene_annot_expr,df_1,df_2],axis=1,join='outer',ignore_index=False,
			# 					keys=None, levels=None, names=None, verify_integrity=True, copy=True)
			# # self.df_gene_annot_2 = data_2
			# self.df_gene_annot_expr = data_2
			# input_filename_1 = '%s/Homo_sapiens.GRCh38.108.1.txt'%(input_file_path1)
			if 'filename_gene_annot_ori' in select_config:
				input_filename_1 = select_config['filename_gene_annot_ori']
			else:
				input_filename_1 = '%s/Homo_sapiens.GRCh38.108.1.txt'%(input_file_path1)
			df_gene_annot = pd.read_csv(input_filename_1,index_col=False,sep='\t')
			df_gene_annot.index = np.asarray(df_gene_annot['gene_id'])
			self.df_gene_annot_ori = df_gene_annot

			# input_filename = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path)
			df_gene_annot_expr_1 = pd.read_csv(input_filename,index_col=False,sep='\t')
			# df_gene_annot_expr = df_gene_annot_expr_1.loc[pd.isna(df_gene_annot_expr_1['chrom'])==False,:]
			df_gene_annot_expr = df_gene_annot_expr_1.dropna(subset=['chrom'])
			df_gene_annot_expr.index = np.asarray(df_gene_annot_expr['gene_name'])
			print('df_gene_annot_expr_1, df_gene_annot_expr ',df_gene_annot_expr_1.shape,df_gene_annot_expr.shape)
			self.df_gene_annot_expr = df_gene_annot_expr

		except Exception as error:
			print('error! ', error)
			return

		# print('df_gene_annot ', self.df_gene_annot_2.shape)
		print('gene annotation load 2', self.df_gene_annot_expr.shape)
		# df_gene_annot_expr = self.df_gene_annot_expr
		# output_filename = '%s/test_E8.75#Multiome_meta_rna.gene_annot_expr.1.txt'%(self.save_path_1)
		# filename_prefix_ori = self.filename_prefix_ori
		# output_filename = '%s/test_%s_meta_rna.gene_annot_expr.1.txt'%(self.save_path_1,filename_prefix_ori)
		# if os.path.exists(output_filename)==True:
		# 	filename1 = output_filename
		# 	print('the file exists ', filename1)
		# else:
		# 	df_gene_annot_expr.to_csv(output_filename,sep='\t')
		# return df_gene_annot_expr
		df_gene_annot_expr_2 = []
		return df_gene_annot, df_gene_annot_expr, df_gene_annot_expr_2

	## motif-peak estimate: load meta_exprs and peak_read
	def test_motif_peak_estimate_control_load_pre1_ori_1(self,dict_feature=[],meta_exprs=[],peak_read=[],flag_scale=0,flag_format=False,save_mode=0,filename_prefix_save='',output_file_path='',verbose=0,select_config={}):

		input_file_path1 = self.save_path_1
		flag_load = 0
		if len(dict_feature)==0:
			flag_load = 1

		if flag_load>0:
			data_file_type = select_config['data_file_type']
			
			# input_filename_1, input_filename_2 = select_config['filename_rna'],select_config['filename_atac']
			input_filename_1, input_filename_2 = select_config['filename_rna_meta'],select_config['filename_atac_meta']
			rna_meta_ad = sc.read(input_filename_1)
			atac_meta_ad = sc.read(input_filename_2)
			# flag_scale: -1, not query scaled exprs; {0,1}, query scaled exprs
			# input_filename_3 = ''
			meta_scaled_exprs = []
			if (flag_scale!=-1):
				flag_scale = 1
				if ('filename_rna_exprs_1' in select_config):
					input_filename_3 = select_config['filename_rna_exprs_1']
					print('filename_rna_exprs_1: %s'%(input_filename_3))
					if os.path.exists(input_filename_3)==False:
						print('the file does not exist: %s'%(input_filename_3))
					else:
						meta_scaled_exprs = pd.read_csv(input_filename_3,index_col=0,sep='\t')
						flag_scale = 0

			if verbose>0:
				print(input_filename_1,input_filename_2)
				print('rna_meta_ad\n', rna_meta_ad)
				print('atac_meta_ad\n', atac_meta_ad)
		else:
			field_query = ['rna','atac','rna_exprs']
			list1 = [dict_feature[field_id] for field_id in field_query[0:2]]
			rna_meta_ad, atac_meta_ad = list1[0:2]
			if (flag_scale!=-1):
				if ('rna_exprs' in dict_feature):
					meta_scaled_exprs = dict_feature['rna_exprs']
				else:
					flag_scale = 1

		# flag_scale = 0
		if filename_prefix_save=='':
			filename_prefix_save = select_config['filename_prefix_default']
		else:
			filename_prefix = filename_prefix_save

		print('flag_scale: ',flag_scale)
		if flag_scale>0:
			scale_type_id = 2
			if 'scale_type_id' in select_config:
				scale_type_id = select_config['scale_type_id']
			else:
				select_config.update({'scale_type_id':scale_type_id})

			save_mode_1 = 2
			pre_meta_ad_rna, pre_meta_ad_scaled_rna = self.test_metacell_compute_unit_2(pre_meta_ad=rna_meta_ad,
																						save_mode=save_mode_1,output_file_path=output_file_path,
																						filename_prefix=filename_prefix,
																						select_config=select_config)

			self.atac_meta_ad = atac_meta_ad
			self.rna_meta_ad = rna_meta_ad
			# self.select_config = select_config

			# output_filename = input_filename
			# save_mode1 = 1
			save_mode1 = save_mode
			rna_meta_ad_scaled = pre_meta_ad_scaled_rna
			meta_scaled_exprs = pd.DataFrame(index=rna_meta_ad_scaled.obs_names,columns=rna_meta_ad_scaled.var_names,
												data=rna_meta_ad_scaled.X.toarray(),dtype=np.float32)
			self.meta_scaled_exprs = meta_scaled_exprs
			self.rna_meta_ad = pre_meta_ad_rna  # add the layer ['scale_%d'%(scale_type_id)]
				
			if save_mode1>0:
				if save_mode1 in [1,3]:
					output_filename = '%s/%s.meta_scaled_exprs.%d.txt'%(output_file_path,filename_prefix,scale_type_id)
					meta_scaled_exprs.to_csv(output_filename,sep='\t',float_format='%.6E')
					# print('meta_scaled_exprs ',meta_scaled_exprs.shape)

				if save_mode1 in [2,3]:
					output_filename = '%s/%s.scale%d.1.h5ad'%(output_file_path,filename_prefix,scale_type_id)
					# pre_meta_atac_ad.write(output_filename)
					pre_meta_ad_scaled.write(output_filename)

				input_filename_3 = output_filename
				select_config.update({'filename_rna_exprs_1':input_filename_3})
			
			self.select_config = select_config
			meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
			# sample_id_1 = meta_exprs_2.index
			# assert list(sample_id_1)==list(peak_read.index)
			# self.meta_exprs_2_ori = meta_exprs_2
			# meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			self.meta_exprs_2 = meta_exprs_2
			if verbose>0:
				print('meta_exprs_2 ', meta_exprs_2.shape)

		# atac_meta_ad = self.atac_meta_ad
		# meta_scaled_exprs = self.meta_scaled_exprs
		if flag_format==True:
			# meta_scaled_exprs.columns = meta_scaled_exprs.columns.str.upper()
			rna_meta_ad.var_names = rna_meta_ad.var_names.str.upper()
			rna_meta_ad.var.index = rna_meta_ad.var.index.str.upper()

		self.rna_meta_ad = rna_meta_ad
		sample_id = rna_meta_ad.obs_names
		assert list(sample_id)==list(atac_meta_ad.obs_names)
		
		atac_meta_ad = atac_meta_ad[sample_id,:]
		self.atac_meta_ad = atac_meta_ad

		peak_read = pd.DataFrame(index=atac_meta_ad.obs_names,columns=atac_meta_ad.var_names,data=atac_meta_ad.X.toarray(),dtype=np.float32)
		meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
		self.meta_exprs_2 = meta_exprs_2

		vec1 = utility_1.test_stat_1(np.mean(atac_meta_ad.X.toarray(),axis=0))
		# vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
		vec3 = utility_1.test_stat_1(np.mean(meta_exprs_2,axis=0))

		if verbose>0:
			print('atac_meta_ad mean values ',atac_meta_ad.shape,vec1)
			# print('meta_scaled_exprs mean values ',meta_scaled_exprs.shape,vec2)
			print('meta_exprs_2 mean values ',meta_exprs_2.shape,vec3)

		if len(meta_scaled_exprs)>0:
			if flag_format==True:
				meta_scaled_exprs.columns = meta_scaled_exprs.columns.str.upper()
			sample_id1 = meta_scaled_exprs.index
			assert list(sample_id)==list(sample_id1)
			meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
			self.meta_scaled_exprs = meta_scaled_exprs
			# print('atac_meta_ad, meta_scaled_exprs ',atac_meta_ad.shape,meta_scaled_exprs.shape,input_filename_3)
			vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
			if verbose>0:
				# print('atac_meta_ad, meta_scaled_exprs ',atac_meta_ad.shape,meta_scaled_exprs.shape)
				print('meta_scaled_exprs mean values ',meta_scaled_exprs.shape,vec2)
			# if (verbose>0) and (flag_load>0):
			# 	# print('input_filename: ',input_filename_1,input_filename_2,input_filename_3)
			# 	print('input_filename: ',input_filename_1,input_filename_2)
			# 	if input_filename_3!='':
			# 		print(input_filename_3)		

		return peak_read, meta_scaled_exprs, meta_exprs_2

	## query obs and var attribute
	def test_attribute_query(self,data_vec,feature_type_vec=[],save_mode=1,output_file_path='',filename_save_annot='',select_config={}):
		
		if len(feature_type_vec)==0:
			feature_type_vec = list(data_vec.keys())  # feature_type_vec=['atac','rna']
			
		dict_query = dict()
		if filename_save_annot=='':
			data_file_type = select_config['data_file_type']
			filename_save_annot = data_file_type
		for feature_type in feature_type_vec:
			adata = data_vec[feature_type]
			data_list = [adata.obs,adata.var]
			annot_list = ['obs','var']
			# dict_query[feature_type] = []
			dict_query[feature_type] = dict()
			for (df_query,annot) in zip(data_list,annot_list):
				if save_mode>0:
					output_filename = '%s/test_%s_meta_ad.%s.df_%s.txt'%(output_file_path,feature_type,filename_save_annot,annot)
					df_query.to_csv(output_filename,sep='\t')
				# dict_query[feature_type].append(df_query)
				field_id1 = annot
				dict_query[feature_type].update({field_id1:df_query})

		return dict_query

	## query normalized peak read
	def test_read_count_query_normalize(self,adata=[],feature_type_query='atac',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',float_format='%.5f',verbose=0,select_config={}):
		
		flag_read_normalize_1=1
		if flag_read_normalize_1>0:
			if len(adata)==0:
				adata = self.atac_meta_ad

			# atac_ad = self.atac_meta_ad
			# rna_ad = self.rna_meta_ad
			if ('normalize' in adata.layers):
				read_count_1 = adata.layers['normalize']
				print('read_count_1: ',read_count_1)
				type_id_query = 0
			
			else:
				type_id_query = 1 # the read count data need to be normalized
				try:
					# peak_read_1 = adata.layers['raw']
					read_count_1 = adata.layers['raw']
					# print('peak_read_1 ',peak_read_1.shape)
					# print(peak_read_1)
					# atac_ad_1 = sc.AnnData(peak_read_1)
					# atac_ad_1.obs_names, atac_ad_1.var_names = atac_ad.obs_names, atac_ad.var_names
					# atac_ad_1.X = csr_matrix(atac_ad_1.X)
					# adata1 = sc.AnnData(read_count_1)
					# adata1.obs_names, adata1.var_names = adata.obs_names, adata.var_names
					# adata1.X = csr_matrix(adata1.X)
				
				except Exception as error:
					print('error! ',error)
					if adata.raw!=None:
						# read_count_1 = adata.raw
						adata1 = adata.raw
						adata1.obs_names, adata1.var_names = adata.obs_names, adata.var_names
						type_id_query = 3
						# peak_read_1 = atac_ad.raw
						# print('peak_read_1 ',peak_read_1.shape)
						# print(peak_read_1)
						# if verbose>0:
						# 	print('read_count_1 ',read_count_1.shape)
						# 	print(read_count_1)
						# adata1 = sc.AnnData(read_count_1)
						# adata1.obs_names, adata1.var_names = adata.obs_names, adata.var_names
						# adata1.X = csr_matrix(adata1.X)
					else:
						# data_file_type = select_config['data_file_type']
						# if (data_file_type in ['mouse_endoderm','E8.75']):
						# 	# input_filename = '%s/test_E8.75#Multiome_meta_atac.normalize.1.0.h5ad'%(self.save_path_1)
						# 	input_filename = '%s/test_E8.75#Multiome_meta_%s.normalize.1.0.h5ad'%(self.save_path_1,feature_type_query)
						# 	# atac_ad_1 = sc.read(input_filename)
						# 	# print('atac_ad_1 ',atac_ad_1.shape)
						# 	# print(atac_ad_1)
						# 	adata1 = sc.read(input_filename)
						# 	type_id_query = 2
						# 	if verbose>0:
						# 		print('adata1 ',adata1.shape)
						# 		print(adata1)
						# else:
						# 	print('please provide peak read file ')
						# 	return

						field_id = 'filename_%d_meta_ad'%(feature_type_query)
						input_filename = select_config[field_id]
						if os.path.exists(input_filename)==False:
							print('please provide peak read file ')
							return

			if (type_id_query in [1,3]) or ((save_mode==2) and (type_id_query!=2)):
				if type_id_query in [1]:
					adata1 = sc.AnnData(read_count_1)
					adata1.obs_names, adata1.var_names = adata.obs_names, adata.var_names
					adata1.X = csr_matrix(adata1.X)
					
					# adata1.obs = adata.obs
					if verbose>0:
						print('read_count_1 ',read_count_1.shape)
						print(read_count_1)
				
				if type_id_query in [1,3]:
					# atac_ad_1 = sc.AnnData(peak_read_1)
					# Normalize
					# peak_read_mtx = peak_read_1.toarray()
					# print(np.min(np.min(peak_read_mtx)),np.max(np.max(peak_read_mtx)))
					# sc.pp.normalize_total(atac_ad_1, key_added='n_counts')
					sc.pp.normalize_total(adata1, key_added='n_counts')

			if save_mode>0:
				filename_prefix_1 = '%s_meta_ad'%(feature_type_query)
				filename_save_annot_1 = filename_save_annot
				if filename_save_annot=='':
					filename_save_annot_1 = select_config['filename_save_annot_pre1']

				if save_mode==2:
					# save anndata
					# output_filename = '%s/test_atac_meta_ad.%s.normalize.h5ad'%(output_file_path,filename_save_annot_1)
					output_filename = '%s/test_%s.%s.normalize.h5ad'%(output_file_path,filename_prefix_1,filename_save_annot_1)
					# atac_ad_1.write(output_filename)
					adata1.write(output_filename)

				if type_id_query==0:
					# read_count_normalize_1 = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=np.asarray(atac_ad_1.X.toarray()),dtype=np.float32)
					read_count_normalize_1 = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=np.asarray(read_count_1),dtype=np.float32)
				else:
					read_count_normalize_1 = pd.DataFrame(index=adata1.obs_names,columns=adata1.var_names,data=np.asarray(adata1.X.toarray()),dtype=np.float32)

				# print('peak_read_1 ',peak_read_1.shape, peak_read_1)
				# print('peak_read_normalize_1 ',peak_read_normalize_1.shape, peak_read_normalize_1)
				# print(np.min(np.min(peak_read_normalize_1)),np.max(np.max(peak_read_normalize_1)))
				if verbose>0:
					print('read_count_normalize_1 ',read_count_normalize_1.shape, read_count_normalize_1)

				# prepare the input to chromVAR
				read_count_normalize_1 = read_count_normalize_1.T # shape: (peak_num,sample_num)
		
				output_filename_1 = '%s/test_%s.%s.normalize.1.csv'%(output_file_path,filename_prefix_1,filename_save_annot_1)
				read_count_normalize_1.to_csv(output_filename_1,sep=',',float_format=float_format)
				t_value_1 = read_count_normalize_1.sum(axis=0)
				
				if verbose>0:
					print('read_count_normalize_1 ',read_count_normalize_1.shape, read_count_normalize_1)
					print('t_value_1 ',np.max(t_value_1),np.min(t_value_1),np.mean(t_value_1),np.median(t_value_1))

			# return peak_read_normalize_1
			return read_count_normalize_1

	## query normalized and log-transformed RNA-seq data
	def test_read_count_query_log_normalize(self,feature_type_vec=[],peak_read=[],rna_exprs_unscaled=[],save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',save_format='tsv.gz',float_format='%.5f',verbose=0,select_config={}):
		
		flag_read_normalize=1
		if flag_read_normalize>0:
			if filename_save_annot=='':
				filename_save_annot = select_config['filename_save_annot_pre1']
			
			feature_type_vec_1 = ['atac','rna']
			if len(peak_read)==0:
				peak_read = self.peak_read
			if len(rna_exprs_unscaled)==0:
				rna_exprs_unscaled = self.meta_exprs_2

			list1 = [peak_read,rna_exprs_unscaled]
			dict_query = dict(zip(feature_type_vec_1,list1))
			for feature_type_query in feature_type_vec:
				df_query_1 = dict_query[feature_type_query]
				# output_filename = '%s/test_%s_meta_ad.%s.%s.tsv.gz'%(output_file_path,filename_annot1,filename_save_annot1,filename_annot2)
				filename_prefix_1 = '%s_meta_ad' % (feature_type_query)
				output_filename = '%s/test_%s.%s.log_normalize.%s'%(output_file_path,filename_prefix_1,filename_save_annot,save_format)
				
				df_query_1.to_csv(output_filename,sep='\t',float_format=float_format)
				if verbose>0:
					print('data: ',df_query_1.shape,feature_type_query)

			return dict_query

	## motif-peak estimate: load meta_exprs and peak_read
	def test_motif_peak_estimate_control_load_pre1_ori_2(self,meta_exprs=[],peak_read=[],flag_format=False,flag_scale=0,save_mode=1,output_file_path='',select_config={}):

		input_file_path1 = self.save_path_1
		data_file_type = select_config['data_file_type']
		# input_file_path = select_config['data_path']
		# filename_save_annot_1 = select_config['filename_save_annot_1']
		
		input_filename_1, input_filename_2 = select_config['filename_rna_meta'],select_config['filename_atac_meta']
		print('input_filename_1 ',input_filename_1)
		print('input_filename_2 ',input_filename_2)
		rna_meta_ad = sc.read_h5ad(input_filename_1)
		atac_meta_ad = sc.read_h5ad(input_filename_2)

		print(input_filename_1,input_filename_2)
		print('rna_meta_ad\n', rna_meta_ad)
		print('atac_meta_ad\n', atac_meta_ad)

		if flag_format==True:
			rna_meta_ad.var_names = rna_meta_ad.var_names.str.upper()
			rna_meta_ad.var.index = rna_meta_ad.var.index.str.upper()

		self.rna_meta_ad = rna_meta_ad
		sample_id = rna_meta_ad.obs_names
		assert list(sample_id)==list(atac_meta_ad.obs_names)

		atac_meta_ad = atac_meta_ad[sample_id,:]
		self.atac_meta_ad = atac_meta_ad
		
		column_1 = 'filename_rna_exprs'
		meta_scaled_exprs = []
		if column_1 in select_config:
			# input_filename_3 = select_config['filename_rna_exprs']
			input_filename_3 = select_config[column_1]
			meta_scaled_exprs = pd.read_csv(input_filename_3,index_col=0,sep='\t')

			if flag_format==True:
				meta_scaled_exprs.columns = meta_scaled_exprs.columns.str.upper()
		
			meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
			# self.meta_scaled_exprs = meta_scaled_exprs
			# print('atac_meta_ad, meta_scaled_exprs ',atac_meta_ad.shape,meta_scaled_exprs.shape,input_filename_3)

			# vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
			# print('meta_scaled_exprs mean values ',meta_scaled_exprs.shape,vec2)
		else:
			if flag_scale>0:
				scale_type_id = 2
				if 'scale_type_id' in select_config:
					scale_type_id = select_config['scale_type_id']
				else:
					select_config.update({'scale_type_id':scale_type_id})

				# save_mode_1 = 2
				save_mode_1 = 1
				filename_prefix = data_file_type
				# output_filename = select_config['filename_rna_exprs']
				output_filename = select_config[column_1]
				pre_meta_ad_rna, pre_meta_ad_scaled_rna = self.test_metacell_compute_unit_2(pre_meta_ad=rna_meta_ad,
																							save_mode=save_mode_1,output_file_path=output_file_path,
																							output_filename=output_filename,
																							filename_prefix=filename_prefix,
																							select_config=select_config)
				# self.atac_meta_ad = atac_meta_ad
				# self.rna_meta_ad = rna_meta_ad
				# self.select_config = select_config

				# output_filename = input_filename
				# save_mode1 = 1
				save_mode1 = save_mode
				rna_meta_ad_scaled = pre_meta_ad_scaled_rna
				meta_scaled_exprs = pd.DataFrame(index=rna_meta_ad_scaled.obs_names,columns=rna_meta_ad_scaled.var_names,
													data=rna_meta_ad_scaled.X.toarray(),dtype=np.float32)
				self.meta_scaled_exprs = meta_scaled_exprs
				print('meta_scaled_exprs ')
				print(meta_scaled_exprs[0:2])
				# self.rna_meta_ad = pre_meta_ad_rna  # add the layer ['scale_%d'%(scale_type_id)]

		# if len(meta_scaled_exprs)>0:
		# 	sample_id1 = meta_scaled_exprs.index
		# 	assert list(sample_id)==list(sample_id1)

		peak_read = pd.DataFrame(index=atac_meta_ad.obs_names,columns=atac_meta_ad.var_names,data=atac_meta_ad.X.toarray(),dtype=np.float32)
		meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
		self.peak_read = peak_read
		self.meta_exprs_2 = meta_exprs_2
		self.meta_scaled_exprs = meta_scaled_exprs

		vec1 = utility_1.test_stat_1(np.mean(atac_meta_ad.X.toarray(),axis=0))
		vec3 = utility_1.test_stat_1(np.mean(meta_exprs_2,axis=0))

		print('atac_meta_ad mean values ',atac_meta_ad.shape,vec1)
		# print('meta_exprs_2 mean values ',meta_exprs_2.shape,vec3)
		print('rna_meta_ad mean values ',meta_exprs_2.shape,vec3)

		if len(meta_scaled_exprs)>0:
			vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
			print('meta_scaled_exprs mean values ',meta_scaled_exprs.shape,vec2)

		return peak_read, meta_scaled_exprs, meta_exprs_2

	## load data and query configuration parameters
	# load motif data; load ATAC-seq and RNA-seq data of the metacells
	def test_query_load_pre1(self,data=[],method_type_vec_query=[],flag_config_1=0,flag_motif_data_load_1=1,flag_load_1=1,flag_format=False,flag_scale=1,input_file_path='',save_mode=1,verbose=0,select_config={}):

		# flag_config_1=1
		# if flag_config_1>0:
		# 	root_path_1 = select_config['root_path_1']
		# 	root_path_2 = select_config['root_path_2']
		# 	data_file_type_query = select_config['data_file_type']
		
		method_type_feature_link = select_config['method_type_feature_link']
		# flag_motif_data_load_1 = 1
		# load motif data
		if flag_motif_data_load_1>0:
			print('load motif data')
			if len(method_type_vec_query)==0:
				method_type_vec_query = [method_type_feature_link]

			input_dir = select_config['input_dir']
			file_path_1 = input_dir
			test_estimator1 = _Base2_2(file_path=file_path_1,select_config=select_config)
			
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_vec_query = [method_type_feature_link]
			data_path_save_local = select_config['data_path_save_local']
			# data_path_save_motif = select_config['data_path_save_motif']
			file_path_motif = data_path_save_local
			select_config.update({'file_path_motif':file_path_motif})
			save_file_path = data_path_save_local
			dict_motif_data, select_config = test_estimator1.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																						save_mode=1,save_file_path=save_file_path,
																						select_config=select_config)

			self.dict_motif_data = dict_motif_data

		# flag_load_1 = 1
		# load the ATAC-seq data and RNA-seq data of the metacells
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			# print('load ATAC-seq and RNA-seq count matrices of the metacells')
			start = time.time()
			data_path_save_local = select_config['data_path_save_local']
			output_file_path = data_path_save_local
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori_2(meta_exprs=[],peak_read=[],flag_format=flag_format,flag_scale=flag_scale,save_mode=1,output_file_path=output_file_path,select_config=select_config)

			# sample_id = meta_scaled_exprs.index
			# peak_read = peak_read.loc[sample_id,:]
			sample_id = peak_read.index
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			if len(meta_scaled_exprs)>0:
				meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
				rna_exprs = meta_scaled_exprs	# scaled RNA-seq data
			else:
				rna_exprs = meta_exprs_2	# unscaled RNA-seq data
			# print('peak_read, rna_exprs: ',peak_read.shape,rna_exprs.shape)

			print('ATAC-seq count matrx: ',peak_read.shape)
			print(peak_read[0:2])

			print('RNA-seq count matrx: ',rna_exprs.shape)
			print(rna_exprs[0:2])

			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs
			# peak_loc_ori = peak_read.columns

			stop = time.time()
			print('load peak accessiblity and gene expression data used %.2fs'%(stop-start))
			
		return select_config

	## bedGraph file for each metacell or cell type
	def test_query_bedGraph_1(self,input_filename_annot='',peak_read=[],column_id='',output_file_path='',type_id_1=1,select_config={}):

		flag_query=1
		if flag_query>0:
			# x = 1
			# file_save_path = '%s/peak_read_data_1'%(self.save_path_1)
			file_save_path = '%s/peak_read_data_1'%(output_file_path)
			if os.path.exists(file_save_path)==False:
				print('the directory does not exist ',file_save_path)
				os.mkdir(file_save_path)

			sample_id = peak_read.index
			sample_num = len(sample_id)

			input_filename_1 = input_filename_annot
			df_annot_1 = pd.read_csv(input_filename_1,index_col=0,sep='\t')
			sample_id_1 = df_annot_1.index
			if column_id=='':
				column_id='Metacell'
			# sample_id_2 = df_annot_1['Metacell'].unique()
			sample_id_2 = df_annot_1[column_id].unique()

			celltype_vec = select_config['celltype_vec']
			celltype_vec_str = select_config['celltype_vec_str']

			df_query_1 = df_annot_1.loc[sample_id_2,:]
			celltype_query_vec = df_query_1['CellType']
			print('df_annot_1, df_query_1 ', df_annot_1.shape, df_query_1.shape)

			celltype_num = len(celltype_vec)
			peak_loc_query = peak_read.columns
			sample_id_2 = pd.Index(sample_id_2)
			chrom_id, start, stop = utility_1.pyranges_from_strings_1(peak_loc_query,type_id=0)
			print('peak_read ', peak_read.shape)
			for i1 in range(celltype_num):
				# x = 1
				celltype_query_1 = celltype_vec[i1]
				id1 = (df_query_1['CellType']==celltype_query_1)
				sample_id_query = sample_id_2[id1]
				sample_id_num1 = len(sample_id_query)
				print('sample_id_query, celltype_query ',sample_id_num1,celltype_query_1)
				if type_id_1>0:
					peak_read_1 = peak_read.loc[sample_id_query,:].mean(axis=0)
					df_1 = pd.DataFrame(index=peak_loc_query,columns=['chrom','start','stop','mean_value'],dtype=np.float32)

					df_1['chrom'] = np.asarray(chrom_id)
					df_1['start'] = np.asarray(start)
					df_1['stop'] = np.asarray(stop)
					df_1['mean_value'] = np.asarray(peak_read_1)

					# output_file_path= file_save_path
					celltype_query_str_1 = celltype_vec_str[i1]
					output_filename = '%s/df_peak_read_mean_%s.1.bedGraph'%(output_file_path,celltype_query_str_1)
					df_1.to_csv(output_filename,index=False,header=False,sep='\t',float_format='%.6f')

				if type_id_1 in [0,1]:
					# peak_read_1 = peak_read.loc[sample_id_query,:].mean(axis=0)
					for i2 in range(sample_id_num1):
						df_1 = pd.DataFrame(index=peak_loc_query,columns=['chrom','start','stop','mean_value'],dtype=np.float32)
						sample_id1 = sample_id_query[i2]
						peak_read_1 = peak_read.loc[sample_id1,:]
						df_1['chrom'] = np.asarray(chrom_id)
						df_1['start'] = np.asarray(start)
						df_1['stop'] = np.asarray(stop)
						df_1['mean_value'] = np.asarray(peak_read_1)

						# output_file_path= file_save_path
						celltype_query_str_1 = celltype_vec_str[i1]
						output_filename = '%s/df_peak_read_mean_%s.%d.1.bedGraph'%(output_file_path,celltype_query_str_1,i2+1)
						df_1.to_csv(output_filename,index=False,header=False,sep='\t',float_format='%.6f')
						if i2%10==0:
							print('sample_id1, celltype_query ',sample_id1,i2,celltype_query_1)

			return True


def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)


