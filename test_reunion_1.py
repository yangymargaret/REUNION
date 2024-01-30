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

import test_annotation_3_1
from test_annotation_3_1 import _Base2_correlation2_1


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

class _Base2_correlation2(_Base2_correlation2_1):
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

		self.path_1 = file_path
		self.config = config
		self.run_id = run_id
		# self.select_config = select_config

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
			cluster_name1 = 'UpdatedCellType'
		else:
			# load multiome scRNA-seq data
			load_mode = 1
			if 'load_mode' in config:
				load_mode = config['load_mode_metacell']
			cluster_name1 = 'CellType'

			# load_mode_rna, load_mode_atac = 1, 1
			load_mode_rna, load_mode_atac = 1, 1
			if 'load_mode_rna' in config:
				load_mode_rna = config['load_mode_rna']
			if 'load_mode_atac' in config:
				load_mode_atac = config['load_mode_atac']

		data_file_type = select_config['data_file_type']
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

		self.save_path_1 = self.path_1

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
														atac_ad=[],rna_exprs=[],highly_variable=False,
														save_mode=1,output_filename='',save_file_path='',interval_peak_corr=50,interval_local_peak_corr=10,
														annot_mode=1,verbose=0,select_config={}):

		if len(peak_loc_query)==0:
			atac_ad = self.atac_meta_ad
			peak_loc_query = atac_ad.var_names
		# save_mode = 1
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
																		type_id_1=type_id2,
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
	def test_gene_peak_query_distance(self,gene_query_vec,df_gene_query=[],peak_loc_query=[],peak_distance_thresh=500,type_id_1=0,save_mode=1,output_filename='',verbose=0,select_config={}):

		# file_path1 = 'scATAC_aug16'
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

			list1 = []
			for i1 in range(gene_query_num):
				gene_query = gene_query_vec[i1]
				start = df_tss_query[gene_query]-span
				stop = df_tss_query[gene_query]+span
				chrom = df_gene_query.loc[gene_query,'chrom']
				gene_pr = pr.from_dict({'Chromosome':[chrom],'Start':[start],'End':[stop]})
				gene_peaks = peaks_pr.overlap(gene_pr)  # search for peak loci within specific distance of the gene
				if i1%1000==0:
					print('gene_peaks ', len(gene_peaks), gene_query, chrom, start, stop, i1)

				if len(gene_peaks)>0:
					df1 = pd.DataFrame.from_dict({'chrom':gene_peaks.Chromosome.values,
										'start':gene_peaks.Start.values,'stop':gene_peaks.End.values})

					df1.index = [gene_query]*df1.shape[0]
					list1.append(df1)
				else:
					print('gene query without peaks in the region query: %s %d'%(gene_query,i1))

			df_gene_peak_query = pd.concat(list1,axis=0,join='outer',ignore_index=False,keys=None,levels=None,names=None,verify_integrity=False,copy=True)

			df_gene_peak_query['gene_id'] = np.asarray(df_gene_peak_query.index)
			df_gene_peak_query.loc[df_gene_peak_query['start']<0,'start']=0
			query_num1 = df_gene_peak_query.shape[0]
			peak_id = test_query_index(df_gene_peak_query,column_vec=['chrom','start','stop'],symbol_vec=[':','-'])
			df_gene_peak_query['peak_id'] = np.asarray(peak_id)
			if (save_mode==1) and (output_filename!=''):
				df_gene_peak_query = df_gene_peak_query.loc[:,['gene_id','peak_id']]
				df_gene_peak_query.to_csv(output_filename,index=False,sep='\t')
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

	## gene-peak association query: peak distance to the gene TSS query
	# input: the gene query, the gene-peak pair query, the gene position and TSS annotation
	# output: update the peak-gene distance in the gene-peak pair annotation (dataframe)
	def test_gene_peak_query_distance_pre1(self,gene_query_vec=[],df_gene_peak_query=[],df_gene_query=[],select_config={}):

		# file_path1 = 'scATAC_aug16'
		file_path1 = self.save_path_1

		# print('peak-gene distance query')
		gene_query_id = np.asarray(df_gene_peak_query['gene_id'])
		field_query_1 = ['chrom','start','stop','strand']
		flag_query1=1
		if flag_query1>0:
			# field_query = ['chrom','start','stop','strand']
			list1 = [np.asarray(df_gene_query.loc[gene_query_id,field_query1]) for field_query1 in field_query_1]
			chrom1, start1, stop1, strand1 = list1
			start_site = start1
			id1 = (strand1=='-')
			start_site[id1] = stop1[id1]
			# print('chrom1, start1, stop1, strand1 ', chrom1.shape, start1.shape, stop1.shape, strand1.shape)
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

		print('df_gene_peak_query: ', df_gene_peak_query.shape, df_gene_peak_query.columns, df_gene_peak_query.index[0:5])
		print('peak_distance: ', peak_distance.shape)
		bin_size = 1000.0
		df_gene_peak_query['distance'] = np.asarray(peak_distance/bin_size)

		return df_gene_peak_query

	## load background peak loci
	def test_gene_peak_query_bg_load(self,input_filename_peak='',input_filename_bg='',peak_bg_num=100,verbose=0,select_config={}):

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
				# print('gene_query_vec_1 ',i1,len(gene_query_vec_1),gene_query_vec_1)
				df_gene_peak_query_1 = self.test_motif_peak_estimate_gene_peak_query_load_unit(gene_query_vec=gene_query_vec_1,
																								gene_peak_annot=gene_peak_local,
																								df_gene_peak_query=[],
																								field_query=[],
																								verbose=verbose,
																								select_config=select_config)
				list_1.append(df_gene_peak_query_1)
				# print('df_gene_peak_query_1, gene_query_vec_1 ', df_gene_peak_query_1.shape, len(gene_query_vec_1), i1)
				# print(input_filename)
				print('df_gene_peak_query, gene_query_vec ', df_gene_peak_query_1.shape, len(gene_query_vec_1), i1)

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
		# sample_num = len(sample_id)
		sample_id_atac = atac_meta_ad.obs_names
		atac_meta_ad = atac_meta_ad[sample_id,:]

		np.random.seed(0)
		gene_query_vec_ori = gene_query_vec
		# gene_query_vec = pd.Index(gene_query_vec).intersection(df_gene_peak_query['gene_id'],sort=False)
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
		# interval_local = 10
		interval_local = -1
		# interval_save = 100
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
		# corr_thresh = 0.1
		corr_thresh = 0.3
		column_corr_1 = select_config['column_correlation'][0]
		if interval_save>0:
			# only save one file for the combined correlation estimation
			output_filename = '%s/%s.1.npy'%(save_file_path,filename_prefix_save)
			gene_peak_local = dict()

		for i1_ori in tqdm(range(query_num)):
			# output_filename = '%s/test_gene_peak_local_1.%s.%d.1.npy'%(save_file_path1,filename_prefix_save,i1_ori)
			# output_filename = '%s/%s.%d.1.npy'%(save_file_path1,filename_prefix_save,i1_ori)
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
			# gene_peak_local = dict()
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
	# def dorc_func_pre1(self,peak_dict,gene_query,atac_read,rna_exprs,gene_id_query='',corr_thresh=0.01,type_id_1=0):
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
				# return

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

		# print('df_gene_peak_query ', df_gene_peak_query)
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


