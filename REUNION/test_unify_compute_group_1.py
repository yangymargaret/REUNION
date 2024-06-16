#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData

from copy import deepcopy
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

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.stats import wilcoxon, mannwhitneyu, kstest, ks_2samp, chisquare, fisher_exact, chi2_contingency
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, issparse
from scipy import signal
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

# import utility_1
from . import utility_1
from .utility_1 import test_query_index
import h5py
import pickle

from .test_unify_compute_3 import _Base2_correlation5

sc.settings.verbosity = 3 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

class _Base2_pre_2(_Base2_correlation5):
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

		_Base2_correlation5.__init__(self,file_path=file_path,
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

		self.test_config_query_1(run_id=run_id,type_id_feature=type_id_feature,select_config=select_config)

	## ====================================================
	# file path query and paramter configuration
	def test_config_query_1(self,run_id=1,type_id_feature=0,select_config={}):

		"""
		file path query and paramter configuration
		:param run_id: index of the current computation
		:param type_id_feature: the feature type based on which to estimate metecells (0:RNA-seq data; 1:ATAC-seq data)
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""

		print('test_config_query')
		input_file_path1 = self.path_1
		if 'data_file_type' in select_config:
			data_file_type = select_config['data_file_type']
			data_file_type_query = data_file_type

			print('data_file_type:%s'%(data_file_type))
		
			data_file_type_query = data_file_type
			select_config.update({'data_file_type_query':data_file_type_query})

		input_dir = select_config['input_dir']
		output_dir = select_config['output_dir']
		input_file_path= input_dir
		output_file_path = output_dir

		column_1 = 'data_path_metacell'
		column_2 = 'data_path_save_motif' # the file path to save the motif data
		column_pre2 = 'data_path_save_local' # the file path to save the estimated peak-gene links
		column_3 = 'file_path_motif_score'   # the file path to save the TF binding activity score estimation
		# column_5 = 'file_path_basic_filter'  # the file path to save the partial correlation estimation
		# column_6 = 'file_path_basic_filter2'	# the file path to save the filtered links

		data_path_metacell = input_file_path
		data_path_save_motif = input_file_path
		data_path_save_local = '%s/folder_link_1'%(output_file_path)
		file_path_motif_score = '%s/folder_link_2'%(output_file_path)
		# file_path_basic_filter = '%s/temp1'%(file_path_motif_score)
		# file_path_basic_filter_2 = '%s/temp2'%(file_path_motif_score)

		# list_1 = [data_path_save_local,file_path_motif_score,file_path_basic_filter,file_path_basic_filter_2]
		# column_vec = [column_pre2,column_3,column_5,column_6]

		list_1 = [data_path_save_local,file_path_motif_score]
		column_vec = [column_pre2,column_3]

		select_config.update({column_1:data_path_metacell,column_2:data_path_save_motif})

		query_num1 = len(list_1)
		for i1 in range(query_num1):
			column_query = column_vec[i1]
			file_path_query = list_1[i1]
			if not (column_query in select_config):
				if os.path.exists(file_path_query)==False:
					print('the directory does not exist: ',file_path_query)
					os.makedirs(file_path_query,exist_ok=True)
				select_config.update({column_query:file_path_query})

		data_dir = input_dir
		data_path_save_1 = data_dir # the folder of the metacell data
		data_file_type_annot = data_file_type_query
		select_config.update({'data_dir':data_dir,
								'file_path_motif_score_2':file_path_motif_score, # directory to save peak-TF-gene links
								'data_path_save':data_path_save_local, # directory to save peak-gene links
								'data_path_save_1':data_path_save_1})

		# query the directories to save annotations of gene-TF links and peak-TF links
		file_path_vec = [data_dir,file_path_motif_score]
		field_vec = ['gene_tf','peak_tf']
		column_vec_query = ['file_path_%s'%(field_id) for field_id in field_vec]
		query_vec = [1,0]
		query_num1 = len(field_vec)

		for i1 in [1]:
			field_id, column_query = field_vec[i1], column_vec_query[i1]
			# query_mode=0: estimate genome-wide regulatory associations
			# query_mode=1: estimate a subset of regulatory associations
			query_mode = query_vec[i1]
			if not (column_query in select_config):
				file_save_path2 = file_path_vec[query_mode]
				file_path_query = '%s/data_%s'%(file_save_path2,field_id)
				if os.path.exists(file_path_query)==False:
					print('the directory does not exist: %s'%(file_path_query))
					os.makedirs(file_path_query,exist_ok=True)
				select_config.update({column_query:file_path_query})

			filename_prefix = 'test_%s_correlation.%s'%(field_id,data_file_type_query)
			field_id_config = 'filename_prefix_%s'%(field_id)
			select_config.update({field_id_config:filename_prefix})

		input_filename_1 = select_config['filename_rna']	# single cell RNA-seq data (with raw counts)
		input_filename_2 = select_config['filename_atac']	# single cell ATAC-seq data (with raw counts)
		print('input_filename_rna:%s, input_filename_atac:%s'%(input_filename_1,input_filename_2))

		filename_save_annot = data_file_type_query
		filename_rna_exprs = '%s/test_rna_meta_ad.%s.meta_scaled_exprs.txt'%(input_file_path,filename_save_annot)

		column_query = 'filename_rna_exprs'
		if not (column_query in select_config):
			select_config.update({column_query:filename_rna_exprs})

		human_cell_type = ['pbmc']
		if data_file_type in human_cell_type:
			self.species_id = 'hg38'
			filename_prefix_1 = 'Homo_sapiens.GRCh38.108' # to update
			# input_filename_gene_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path) # gene annotation data
			# select_config.update({'filename_gene_annot':input_filename_gene_annot})
		else:
			self.species_id = 'mm10'
			filename_prefix_1 =  'Mus_musculus.GRCm38.102'

		# prepare filename_prefix and filename_annot_save
		# filename_prefix_save_pre2 = 'test_query.%s'%(data_file_type_query)
		filename_prefix_save_pre2 = 'test_query_gene_peak.%s'%(data_file_type_query)
		filename_prefix_save = 'test_query_gene_peak.%s'%(data_file_type_query)
		filename_annot_save_motif = 'test_query_motif'

		select_config.update({'filename_prefix_default':filename_prefix_save,
								'filename_prefix_default_1':filename_prefix_save,
								'filename_prefix_default_pre2':filename_prefix_save_pre2,
								'filename_annot_save_motif':filename_annot_save_motif})

		filename_prefix_cond = filename_prefix_save_pre2
		select_config.update({'filename_prefix_cond':filename_prefix_cond})

		filename_prefix_score = '%s.pcorr_query1'%(filename_prefix_cond)
		filename_annot_score_1 = 'annot2.init.1'
		# filename_annot_score_2 = 'annot2.init.query1'
		filename_annot_score_2 = 'annot2.init.1.query1'
		select_config.update({'filename_prefix_score':filename_prefix_score,
								'filename_annot_score_1':filename_annot_score_1,
								'filename_annot_score_2':filename_annot_score_2})
		
		filename_save_annot_1 = data_file_type_query
		print('filename_save_annot_1: ',filename_save_annot_1)
		
		if not('filename_save_annot_1' in select_config):
			select_config.update({'filename_save_annot_1':filename_save_annot_1})
			select_config.update({'filename_save_annot_pre1':filename_save_annot_1})
		else:
			print('filename_save_annot_1 original: ',select_config['filename_save_annot_1'])
		
		key_vec = list(select_config.keys())
		for field_id in key_vec:
			query_value = select_config[field_id]
			# print('field, value: ',field_id, query_value)
			print('%s: '%(field_id), query_value)

		select_config = self.test_config_query_pre1_1(select_config=select_config)

		select_config = self.test_config_query_pre1_2(select_config=select_config)

		self.dict_peak_tf_corr_ = dict()
		self.dict_gene_tf_corr_ = dict()
		self.df_peak_tf_1 = None
		self.df_peak_tf_2 = None
		self.dict_peak_tf_query = dict()
		self.dict_gene_tf_query = dict()
		self.select_config = select_config

		return select_config

	## ====================================================
	# paramater configuration for motif scanning data query and peak-gene link estimation 
	def test_config_query_pre1_1(self,overwrite=False,select_config={}):
		
		"""
		paramater configuration for motif scanning data query and peak-gene link estimation 
		:param overwrite: indicator of whether to overwrite the current parameter
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""
		
		peak_bg_num_ori = 100
		peak_bg_num = 100
		# interval_peak_corr = 10
		interval_peak_corr = 100
		interval_local_peak_corr = -1

		data_file_type = select_config['data_file_type']
		data_file_type_query = data_file_type
		input_file_path = select_config['data_path_save_motif']
		
		column_1 = 'input_filename_peak'
		column_2 = 'input_filename_bg'
		input_filename_peak = select_config[column_1]
		input_filename_bg = select_config[column_2]

		list1 = [peak_bg_num,interval_peak_corr,interval_local_peak_corr]
		field_query = ['peak_bg_num','interval_peak_corr','interval_local_peak_corr']
		query_num1 = len(list1)
		for i1 in range(query_num1):
			field_id = field_query[i1]
			query_value = list1[i1]
			if (not (field_id in select_config)) or (overwrite==True):
				select_config.update({field_id:query_value})

		motif_data_thresh = 5e-05
		column_query = 'motif_data_thresh'
		if column_query in select_config:
			motif_data_thresh = select_config[column_query]
		else:
			select_config.update({column_query:motif_data_thresh})

		filename_translation = select_config['filename_translation']

		data_file_query_motif = data_file_type_query
		data_path_save_motif = select_config['data_path_save_motif']
		motif_filename1 = '%s/test_motif_data.%s.h5ad'%(data_path_save_motif,data_file_query_motif) # motif data saved in anndata
		motif_filename2 = '%s/test_motif_data_score.%s.h5ad'%(data_path_save_motif,data_file_query_motif) # motif score data save in anndata
		select_config.update({'motif_filename1':motif_filename1,'motif_filename2':motif_filename2})

		field_query = ['motif_filename1','motif_filename2','filename_translation',column_1,column_2]
		for field_id in field_query:
			print('%s: %s'%(field_id,select_config[field_id]))

		column_motif = 'motif_id'
		select_config.update({'column_motif':column_motif})

		filename_annot_default = '1'
		filename_prefix_save_1 = 'pre1'
		filename_prefix_save_2 = 'pre2'
		select_config.update({'filename_annot_default':filename_annot_default,
								'filename_prefix_save_default':filename_prefix_save_1,
								'filename_prefix_save_2':filename_prefix_save_2})

		filename_prefix_1 = select_config['filename_prefix_default']
		filename_prefix_save_1 = select_config['filename_prefix_save_default']
		
		filename_prefix = '%s.%s'%(filename_prefix_1,filename_prefix_save_1)
		filename_annot1 = filename_annot_default
		select_config.update({'filename_prefix_peak_gene':filename_prefix})

		data_path_save_local = select_config['data_path_save_local']
		save_file_path = data_path_save_local
		# the original peak-gene correlations
		input_filename_pre1 = '%s/%s.combine.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
		# the peak-gene links selected using threshold 1 on the peak-gene correlation
		input_filename_pre2 = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix,filename_annot1)

		# the peak-gene links selected using threshold 2 on the peak-gene correlation
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
		
		## threshold for pre-selection of peak-gene links to estimate empirical p-values
		# thresh_distance_1 = 50 # to update
		thresh_distance_1 = 100 # to update
		thresh_corr_distance_1 = [[0,thresh_distance_1,0],
									[thresh_distance_1,500,0.01],
									[500,1000,0.1],
									[1000,2050,0.15]]

		## threshold for pre-selection of peak-gene links as candidate peaks
		# thresh_distance_1_2 = 50
		thresh_distance_1_2 = 100
		thresh_corr_distance_2 = [[0,thresh_distance_1_2,[[0,1,0,1]]],
									[thresh_distance_1_2,500,[[0.01,0.1,-0.01,0.1],[0.15,0.15,-0.15,0.15]]],
									[500,1000,[[0.1,0.1,-0.1,0.1]]],
									[1000,2050,[[0.15,0.1,-0.15,0.1]]]]

		thresh_corr_retain = [0.3,0.35]

		beta_mode = 0
		if 'beta_mode' in select_config:
			beta_mode = select_config['beta_mode']
		
		list2 = [input_filename_pre1,input_filename_pre2,filename_peak_gene_thresh2,
					filename_distance_annot,
					column_correlation,correlation_type_1,
					column_distance,column_highly_variable,column_thresh2,
					thresh_distance_1,thresh_corr_distance_1,
					thresh_distance_1_2,thresh_corr_distance_2,thresh_corr_retain,beta_mode]

		field_query2 = ['input_filename_pre1','input_filename_pre2','filename_save_thresh2',
						'filename_distance_annot',
						'column_correlation','correlation_type_1',
						'column_distance','column_highly_variable','column_thresh2',
						'thresh_distance_default_1','thresh_corr_distance_1',
						'thresh_distance_default_2','thresh_corr_distance_2','thresh_corr_retain','beta_mode']

		query_num2 = len(field_query2)
		for i1 in range(query_num2):
			field_id = field_query2[i1]
			query_value = list2[i1]
			select_config.update({field_id:query_value})

		self.select_config = select_config
		# for field in select_config:
		# 	print('field: ',field,select_config[field])

		return select_config

	## ====================================================
	# paramater configuation for feature link comparison
	def test_config_query_pre1_2(self,beta_mode=0,overwrite=False,save_mode=1,select_config={}):

		"""
		paramater configuation for feature link comparison
		:param beta_mode: indicator of whether to perform regulatory association inference for a subset of the target genes
		:param overwrite: indicator of whether to overwrite the current file
		:param save_mode: indicator of whether to save data
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""

		# print('configuration for feature association comparison')
		flag_query1=1
		if flag_query1>0:
			thresh_ratio_query1, thresh_query1, distance_tol_query1 = 0.9, -0.02, -100
			thresh_ratio_query2, thresh_query2, distance_tol_query2 = 0.8, -0.1, -500
			thresh_ratio_query3, thresh_query3, distance_tol_query3 = 1.5, 0.1, 25

			thresh_vec_query1 = [thresh_ratio_query1, thresh_query1, distance_tol_query1]
			thresh_vec_query2 = [thresh_ratio_query2, thresh_query2, distance_tol_query2]
			thresh_vec_query3 = [thresh_ratio_query3, thresh_query3, distance_tol_query3]

			thresh_vec_compare = [thresh_vec_query1,thresh_vec_query2,thresh_vec_query3]
		
			peak_distance_thresh_compare = 50

			parallel_mode = 0
			interval_peak_query = 100
			
			# used in function: test_peak_score_distance_1()
			decay_value_vec = [1,0.9,0.75,0.6]
			distance_thresh_vec = [50,500,1000,2000]
			
			list1 = [thresh_vec_compare,peak_distance_thresh_compare,parallel_mode,interval_peak_query,decay_value_vec,distance_thresh_vec]
			
			field_query = ['thresh_vec_compare','peak_distance_thresh_compare','parallel_mode','interval_peak_query',
							'decay_value_vec','distance_thresh_vec']
			
			field_num1 = len(field_query)
			for i1 in range(field_num1):
				field_id = field_query[i1]
				if not (field_id in select_config):
					select_config.update({field_id:list1[i1]})

			# save_file_path = select_config['data_path_save']
			save_file_path = select_config['data_path_save_local']

			input_file_path = save_file_path
			filename_prefix_1 = select_config['filename_prefix_default']
			filename_prefix_2 = select_config['filename_prefix_save_default']
			
			column_query = 'input_filename_peak_query'
			if column_query in select_config:
				input_filename_peak_query = select_config[column_query]
			else:
				input_filename_peak_query = '%s/%s.%s.peak_basic.txt'%(input_file_path,filename_prefix_1,filename_prefix_2)
				select_config.update({column_query:input_filename_peak_query})

			self.select_config = select_config

			return select_config

	## ====================================================
	# query metacell data annotations and count matrices of ATAC-seq and RNA-seq data 
	def test_metacell_compute_basic_1(self,flag_attribute_query=1,flag_read_normalize_1=1,flag_read_normalize_2=1,save_mode=1,save_file_path='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query metacell data annotations and count matrices of ATAC-seq and RNA-seq data 
		:param flag_attribute_query: indicator of whether to query observation and variable attributes of the ATAC-seq and RNA-seq data of the metacells
		:param flag_read_normalize_1: indicator of query normalized ATAC-seq data without log transformation
		:param flag_read_normalize_2: indicator of query normalized and log-transformed ATAC-seq data and RNA-seq data
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing observation and variable attributes of the ATAC-seq and RNA-seq data of the metacells;
				 2. (dataframe) normalized count matrix of ATAC-seq data (row:cell or metacell, column:ATAC-seq peak locus) with or without log-transformation;
				 3. (dataframe) normalized count matrix of RNA-seq data (row:cell or metacell, column:gene) with log-transformation;
		"""

		# query observation and variable attributes of the ATAC-seq and RNA-seq AnnData
		output_file_path = save_file_path
		dict_attribute_query = dict()
		peak_read = []
		rna_exprs_unscaled = []
		if flag_attribute_query>0:
			feature_type_vec = ['atac','rna']
			atac_ad = self.atac_meta_ad
			rna_ad = self.rna_meta_ad
			data_list = [atac_ad,rna_ad]
			data_vec = dict(zip(feature_type_vec,data_list))
			filename_save_annot_1 = filename_save_annot
			print('query attributes of the ATAC-seq and RNA-seq data')
			dict_attribute_query = self.test_attribute_query(data_vec,save_mode=1,output_file_path=output_file_path,
																filename_save_annot=filename_save_annot_1,
																select_config=select_config)

		# query normalized ATAC-seq data without log transformation
		if flag_read_normalize_1>0:
			print('normalized chromatin accessibility data without log transformation')
			start = time.time()
			feature_type_query = 'atac'
			peak_read_normalize = self.test_read_count_query_normalize(output_file_path=output_file_path,
																		feature_type_query=feature_type_query,
																		select_config=select_config)
			peak_read = peak_read_normalize
			stop = time.time()
			print('used: %.5fs'%(stop-start))

		# query normalized and log-transformed ATAC-seq and RNA-seq data
		if flag_read_normalize_2>0:
			print('normalized chromatin accessibility and gene expression data with log transformation')
			start = time.time()
			feature_type_vec_query = ['atac','rna']
			feature_type_query1, feature_type_query2 = feature_type_vec_query[0:2]
			dict_query_1 = self.test_read_count_query_log_normalize(output_file_path=output_file_path,select_config=select_config)
			peak_read = dict_query_1[feature_type_query1]
			rna_exprs_unscaled = dict_query_1[feature_type_query2]
			stop = time.time()
			print('used: %.5fs'%(stop-start))
		
		return dict_attribute_query, peak_read, rna_exprs_unscaled

	## ====================================================
	# query and prepare gene annotations
	def test_query_gene_annot_basic_1(self,flag_gene_annot_1=0,flag_gene_annot_2=0,flag_annot1=0,flag_annot2=0,flag_annot3=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		query and prepare gene annotations
		:param flag_gene_annot_1: indicator of whether to query gene annotations and match gene names between gene annotations and gene expression data;
		:param flag_gene_annot_2: indicator of whether to query gene annotations of genes with expressions in the RNA-seq data
		:param flag_annot1: indicator of whether to query gene annotations for genes with expressions in the RNA-seq data
		:param flag_annot2: indicator of whether to merge annotations of the given genes from different versions of gene annotations
		:param flag_annot3: indicator of whether to query transcription start sites of genes
		:param save_mode: indicator of whether to save data
		:param input_file_path: the directory to retrieve data from
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) gene annotations of genes with expressions in the RNA-seq data
		"""

		df_gene_annot_query = []
		if flag_gene_annot_1>0:
			print('prepare gene annotations')
			start = time.time()
			# flag_gene_annot_query_1, flag_gene_annot_query_2, flag_gene_annot_query_3 = 0, 0, 0
			# 1. df_gene_annot1: gene annotations of the genome-wide genes;
			# 2. df_gene_annot2: gene annotations of genes with expressions merged from different verisons of gene annotations
			# 3. df_gene_annot3: gene annotations of genes with expressions, including the transcription start site information;
			df_gene_annot1, df_gene_annot2, df_gene_annot3 = self.test_gene_annotation_query_pre1(flag_query1=flag_annot1,
																									flag_query2=flag_annot2,
																									flag_query3=flag_annot3,
																									input_file_path=input_file_path,
																									select_config=select_config)
			stop = time.time()
			print('used: %.5fs'%(stop-start))
			df_gene_annot_query = df_gene_annot3

		# load gene annotations
		if flag_gene_annot_2>0:
			print('load gene annotations')
			start = time.time()
			# query gene annotations of genes with expressions in the RNA-seq data
			self.test_gene_annotation_query1(select_config=select_config)
			stop = time.time()
			print('load gene annotations used: %.5fs'%(stop-start))
			df_gene_annot_expr = self.df_gene_annot_expr
			print('columns: ',np.asarray(df_gene_annot_expr.columns))

			if save_mode>0:
				data_path_save = output_file_path
				if output_file_path=='':
					data_path_save = select_config['data_path_save']

				if output_filename=='':
					data_file_type_query = select_config['data_file_type']
					output_filename = '%s/test_query_gene_annot.%s.1.txt'%(data_path_save,data_file_type_query)
				df_gene_annot_expr.to_csv(output_filename,sep='\t')
			df_gene_annot_query = df_gene_annot_expr

		return df_gene_annot_query

	## ====================================================
	# update filename for computation in batch mode
	def test_query_filename_batch_pre1(self,select_config={}):

		"""
		update filename for computation in batch mode
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""

		column_1 = 'filename_prefix_default'
		column_query1 = '%s_ori'%(column_1)
		filename_prefix_default = select_config[column_1]

		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		annot_str1 = '%d_%d'%(query_id1,query_id2)

		b = filename_prefix_default.find(annot_str1)
		if b<0:
			filename_prefix_default_ori = filename_prefix_default
			select_config.update({column_query1:filename_prefix_default_ori})

			filename_prefix_save = '%s.%s'%(filename_prefix_default,annot_str1)
			select_config.update({column_1:filename_prefix_save})

		data_path_save_local = select_config['data_path_save_local']
		save_file_path = data_path_save_local
		
		filename_prefix_1 = select_config[column_query1]
		filename_prefix_save_default = select_config['filename_prefix_save_default']
		filename_prefix = '%s.%s_%s'%(filename_prefix_1,filename_prefix_save_default,annot_str1)
		filename_annot1 = select_config['filename_annot_default']
		
		# the original peak-gene correlations
		input_filename_pre1 = '%s/%s.combine.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
		
		# the peak-gene links selected using threshold 1 on the peak-gene correlation
		input_filename_pre2 = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix,filename_annot1)

		# the peak-gene links selected using threshold 2 on the peak-gene correlation
		filename_peak_gene_thresh2 = '%s/%s.combine.thresh2.%s.txt'%(save_file_path,filename_prefix,filename_annot1)

		field_query = ['input_filename_pre1','input_filename_pre2','filename_save_thresh2']
		list1 = [input_filename_pre1,input_filename_pre2,filename_peak_gene_thresh2]
		for (field_id,query_value) in zip(field_query,list1):
			select_config.update({field_id:query_value})

		return select_config

	## ====================================================
	# peak-TF-gene associaton estimation
	def test_query_feature_link_pre1(self,gene_query_vec=[],peak_distance_thresh=2000,df_peak_query=[],peak_loc_query=[],
										atac_ad=[],rna_exprs=[],highly_variable=False,interval_peak_corr=50,interval_local_peak_corr=10,beta_mode=0,
										save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		peak-TF-gene associaton estimation;
		peak-gene association query: for each gene, search for peaks within the distance threshold, and compute peak accessibility-gene expr correlation and p-value;
									 pre-select peak-gene links and sample background peaks to comptue empirical p-value of peak-gene correlation;
									 pre-select peak-gene links;
									 compare alternative peak-gene links and perform selection based on peak accessibility-gene expr correlation and distance;
		query TF motif scanning data;
		compute peak accessibility-TF expression correlation and regularization scores of peak-TF links;
		compute gene-TF expr partial correlation given peak accessibility and gene-TF expr correlation;
		estimate the types of peak-gene, peak-TF, and gene-TF links (activation or repression);
		compute peak-TF-gene association scores;
		:param flag_compute: indicator of whether to compute the peak-gene correlations or load the computed correlations from saved files 
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param interval_local_peak_corr: the number of genes in the sub-batch for which to compute peak-gene correlations in the batch mode
		:param peak_bg_num: the number of background peaks to sample for each candidate peak, which match the candidate peak by GC content and average accessibility
		:param save_file_path: the directory to save data
		:param save_file_path_local: the second directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: anntations of the selected peak-TF-gene links and the pre-selected candidate peak-TF-gene links and peak-gene links are saved to files
		"""

		data_file_type = select_config['data_file_type']
		data_file_type_query = data_file_type
		select_config.update({'data_file_type_query':data_file_type_query})
		verbose_internal = self.verbose_internal
		print('dataset annotation: ',data_file_type_query)
		
		column_query = 'data_file_type_annot'
		if column_query in select_config:
			data_file_type_annot = select_config[column_query]
		else:
			data_file_type_annot = data_file_type_query
			select_config.update({column_query:data_file_type_annot})
		
		input_dir = select_config['input_dir']
		output_dir = select_config['output_dir']
		input_file_path_pre1 = input_dir
		output_file_path_pre1 = output_dir
		print('input file path: ',input_file_path_pre1)
		print('output file path: ',output_file_path_pre1)

		# select_config['flag_distance'] = 1
		field_query = ['flag_attribute_query',
						'flag_read_normalize_1','flag_read_normalize_2',
						'flag_gene_annot_query_pre1','flag_gene_annot_query',
						'flag_motif_data_load',
						'flag_distance',
						'flag_correlation_query','flag_correlation_1','flag_combine_empirical','flag_query_thresh2',
						'flag_basic_query',
						'flag_basic_filter_1','flag_basic_filter_combine_1',
						'flag_combine_1','flag_combine_2']

		config_flag = pd.Series(index=field_query,data=0,dtype=np.int32)
		config_flag.loc['flag_gene_annot_query'] = 1 # load gene annotations
		for field_id1 in field_query:
			if field_id1 in select_config:
				config_flag.loc[field_id1] = select_config[field_id1]
			else:
				select_config.update({field_id1:config_flag[field_id1]})

		# flag_attribute_query=0
		# flag_query_list1 = [0,0,0]
		field_query1 = ['flag_attribute_query','flag_read_normalize_1','flag_read_normalize_2']
		flag_attribute_query, flag_read_normalize_1, flag_read_normalize_2 = [select_config[field_id1] for field1 in field_query1]

		# load gene annotation data
		flag_gene_annot_query=1
		if flag_gene_annot_query>0:
			print('load gene annotations')
			# filename_gene_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path_pre1)
			# select_config.update({'filename_gene_annot':filename_gene_annot})
			filename_gene_annot = select_config['filename_gene_annot']
			
			df_gene_annot_ori = self.test_query_gene_annot_1(filename_gene_annot,verbose=verbose,select_config=select_config)
			self.df_gene_annot_ori = df_gene_annot_ori
			self.df_gene_annot_expr = df_gene_annot_ori
			print('gene annotations loaded from: %s'%(filename_gene_annot))

		# load motif data
		# load ATAC-seq and RNA-seq data of the metacells
		flag_load_1 = 1
		# flag_motif_data_load = select_config['flag_motif_data_load']
		flag_motif_data_load = 1
		flag_load_pre1 = (flag_load_1>0)|(flag_motif_data_load>0)
		select_config.update({'flag_motif_data_load':flag_motif_data_load})

		method_type_feature_link = select_config['method_type_feature_link']
		method_type_query = method_type_feature_link

		if verbose_internal>0:
			print('the method to estimate the initial peak-TF-gene associations: %s'%(method_type_query))
			
		if flag_load_pre1>0:
			flag_scale = 1
			flag_format = False
			
			method_type_vec_query1 = [method_type_feature_link]
			# flag_config_1 = 0
			# load motif data; load ATAC-seq and RNA-seq data of the metacells
			motif_data_rewrite = False
			select_config.update({'motif_data_rewrite':motif_data_rewrite})
			select_config = self.test_query_load_pre1(method_type_vec=method_type_vec_query1,
														flag_motif_data_load_1=flag_motif_data_load,
														flag_load_1=flag_load_1,
														save_mode=1,
														verbose=verbose,select_config=select_config)

			rna_exprs = self.rna_exprs
			meta_scaled_exprs = self.meta_scaled_exprs
			meta_exprs_2 = self.meta_exprs_2  # normalized and log-transformed data
			peak_read = self.peak_read  # normalized and log-transformed data
			peak_loc_ori = peak_read.columns

			dict_motif_data = self.dict_motif_data
			motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']

			motif_data = motif_data_query1.loc[peak_loc_ori,:]
			motif_data_score = motif_data_score_query1.loc[peak_loc_ori,:]
			self.motif_data  = motif_data
			self.motif_data_score = motif_data_score

			motif_name_ori = motif_data.columns
			gene_name_expr_ori = rna_exprs.columns
			motif_query_name_expr = pd.Index(motif_name_ori).intersection(gene_name_expr_ori,sort=False)
			self.motif_query_name_expr = motif_query_name_expr
			print('TFs with expression: ',len(motif_query_name_expr))
			print('preview: ',np.asarray(motif_query_name_expr)[0:5])

			# return

		data_path_save_1 = select_config['data_path_save_1'] # the directory of the metacell data
		input_file_path = data_path_save_1
		data_path_1 = data_path_save_1

		save_file_path_1 = data_path_save_1
		save_file_path = select_config['data_path_save_local']	# the folder to save feature link estimation
		output_file_path = save_file_path

		# query obs and var attributes of the ATAC-seq and RNA-seq anndata
		# query normalized peak read data without log transformation
		# query normalized and log-transformed ATAC-seq data and RNA-seq data
		flag_attribute_query = 1
		flag_read_normalize_1 = 0
		flag_read_normalize_2 = 0

		# query metacell data attributes and prepreare feature matrix of metacells
		t_vec_1 = self.test_metacell_compute_basic_1(flag_attribute_query=flag_attribute_query,
														flag_read_normalize_1=flag_read_normalize_1,
														flag_read_normalize_2=flag_read_normalize_2,
														save_mode=1,save_file_path=save_file_path,
														verbose=verbose,select_config=select_config)
		dict_attribute_query, peak_read_normalize, rna_exprs_normalize = t_vec_1[0:3]

		feature_type_query = 'rna'
		dict_attribute_query1 = dict_attribute_query[feature_type_query]
		df_rna_obs_metacell = dict_attribute_query1['obs']
		df_rna_var = dict_attribute_query1['var']
		# print('df_rna_obs_meta ',df_rna_obs_metacell.shape)
		print('RNA-seq metacell annotation, dataframe of size ',df_rna_obs_metacell.shape)
		print('data preview:')
		print(df_rna_obs_metacell[0:2])
		# print('df_rna_var ',df_rna_var.shape)
		print('gene attributes in the RNA-seq metacell data, dataframe of size ',df_rna_var.shape)
		print('data preview: ')
		print(df_rna_var[0:2])

		flag_gene_annot_query_pre1 = 0
		column_1 = 'flag_gene_annot_query_pre1'
		if column_1 in select_config:
			flag_gene_annot_query_pre1 = select_config[column_1]
		# flag_gene_annot_query = select_config['flag_gene_annot_query']
		df_gene_annot_1 = self.test_query_gene_annot_basic_1(flag_gene_annot_1=flag_gene_annot_query_pre1,
																flag_gene_annot_2=0,
																input_file_path='',
																save_mode=1,
																verbose=verbose,select_config=select_config)

		# search for peak loci within distance (2Mb) of the gene TSS
		if peak_distance_thresh<0:
			peak_distance_thresh = 2000
		bin_size = 1000

		rna_exprs = self.meta_scaled_exprs
		atac_ad = self.atac_meta_ad
		sample_id = rna_exprs.index
		atac_ad = atac_ad[sample_id,:]

		highly_variable = False
		filename_prefix_default_pre1 = 'test_query_gene_peak_1' # filename prefix not dependent on sample
		column_1 = 'filename_prefix_default_pre1'
		if column_1 in select_config:
			filename_prefix_default_pre1 = select_config[column_1]
		else:
			select_config.update({column_1:filename_prefix_default_pre1})

		data_file_type_query = data_file_type

		filename_prefix_default = 'test_query_gene_peak.%s'%(data_file_type_query) # filename prefix dependent on sample
		filename_annot_default = '1'
		filename_prefix_save_1 = 'pre1'
		filename_prefix_save_2 = 'pre2'
		select_config.update({'filename_prefix_default':filename_prefix_default,
								'filename_annot_default':filename_annot_default,
								'filename_prefix_save_default':filename_prefix_save_1,
								'filename_prefix_save_2':filename_prefix_save_2})

		select_config.update({'peak_distance_thresh':peak_distance_thresh})

		flag_distance = config_flag['flag_distance']
		beta_mode = select_config['beta_mode']
		df_gene_annot = self.df_gene_annot_ori
		
		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		if (query_id1>=0) and (query_id2>query_id1):
			iter_mode = 1
		elif query_id1<-1:
			iter_mode = 2  # combine peak-gene estimation from different runs
		else:
			iter_mode = 0
		select_config.update({'iter_mode':iter_mode})

		if flag_distance>0:
			filename_prefix_1 = select_config['filename_prefix_default_pre1']
			parallel_mode_peak = select_config['parallel_mode_peak']
			parallel_mode_query = parallel_mode_peak
			# input_filename = '%s/%s.peak_query.%d.txt'%(input_file_path,filename_prefix_1,peak_distance_thresh)
			input_filename = '%s/%s.peak_query.%d.%d.txt'%(input_file_path,filename_prefix_1,parallel_mode_query,peak_distance_thresh)
			output_filename = input_filename
			gene_name_query_ori = df_gene_annot['gene_name']
			gene_query_vec_pre1 = []

			# search for peaks within the distance threshold of the targat gene
			df_gene_peak_distance = self.test_gene_peak_query_link_pre1(gene_query_vec=gene_query_vec_pre1,
																		peak_distance_thresh=peak_distance_thresh,
																		df_peak_query=[],
																		input_filename=input_filename,
																		peak_loc_query=[],
																		atac_ad=[],
																		rna_exprs=[],
																		highly_variable=False,
																		parallel=parallel_mode_query,
																		save_mode=1,save_file_path='',
																		output_filename=output_filename,
																		filename_prefix_save='',	
																		select_config=select_config)

			self.df_gene_peak_distance = df_gene_peak_distance
			print('peak-gene links by distance threshold, dataframe of size ',df_gene_peak_distance.shape)
			print('preview: ')
			print(df_gene_peak_distance[0:5])
			# print(input_filename)

		flag_correlation_query = select_config['flag_correlation_query']
		save_file_path = select_config['data_path_save_local']

		# peak-gene correlation estimation
		if flag_correlation_query>0:
			# flag_computation_vec = [1]
			# flag_computation_vec = [3]
			flag_computation_vec = [1,3]
			
			# gene_pre1_flag_computation, gene_pre1_flag_combine_1 = 1, 1
			gene_pre1_flag_computation = 1
			gene_pre1_flag_combine_1 = 0
			gene_pre1_flag_combine_2 = 0
			recompute = 0

			# flag_correlation_1 = 1
			flag_combine_empirical = 1
			flag_query_thresh2 = 1
			
			list1 = [flag_computation_vec,gene_pre1_flag_computation,gene_pre1_flag_combine_1,
						peak_distance_thresh,recompute,flag_combine_empirical]
			field_query1 = ['flag_computation_vec','gene_pre1_flag_computation','gene_pre1_flag_combine_1',
							'peak_distance_thresh','recompute','flag_combine_empirical']
			
			overwrite_2 = False
			# select_config = utility_1.test_field_query_pre1(field_query=field_query1,query_value=list1,overwrite=overwrite_2,select_config=select_config)
			select_config, list_query1 = utility_1.test_query_default_parameter_1(field_query=field_query1,default_parameter=list1,overwrite=overwrite_2,select_config=select_config)
			flag_computation_vec,gene_pre1_flag_computation,gene_pre1_flag_combine_1,peak_distance_thresh,recompute,flag_combine_empirical = list_query1

			df_gene_annot_expr = self.df_gene_annot_expr
			highly_variable = True

			# select highly variable genes
			rna_meta_ad = self.rna_meta_ad
			gene_idvec = rna_meta_ad.var_names
			df_gene_annot2 = rna_meta_ad.var
			column_query1 = 'dispersions_norm'
			df_gene_annot2 = df_gene_annot2.sort_values(by=['dispersions_norm','dispersions'],ascending=False)
			gene_vec_1 = df_gene_annot2.index
			
			# thresh_dispersions_norm = 0.5
			thresh_dispersions_norm = 1.0
			# num_top_genes = 3000
			gene_num_pre1 = 3000
			if 'gene_num_query' in select_config:
				gene_num_pre1 = select_config['gene_num_query']

			id_query1 = (df_gene_annot2[column_query1]>thresh_dispersions_norm)
			gene_highly_variable = gene_vec_1[id_query1]
			gene_highly_variable_num = len(gene_highly_variable)
			print('highly variable gene (normalized dispersion above %s): '%(thresh_dispersions_norm),gene_highly_variable_num)
			
			# type_id2 = 0
			type_id2 = 1
			if gene_num_pre1<0:
				type_id2 = 3
			group_type_id_1 = type_id2
			select_config.update({'group_type_id_1':group_type_id_1})

			field_query_1 = ['filename_prefix_default','filename_annot_default',
							'filename_prefix_save_default','filename_prefix_save_2']

			df_gene_peak_1 = [] # peak-gene link with distance annotation
			# peak-gene link with peak-gene correlation estimation; 
			# pre-selected peak-gene link with peak-gene correlation estimation;
			df_gene_peak_compute_1, df_gene_peak_compute_2 = [], []
			
			if type_id2==0:
				filename_prefix_default = 'test_query_gene_peak.%s.highly_variable'%(data_file_type_query)

			filename_annot_default = '1'
			filename_prefix_save_1 = 'pre1'
			filename_prefix_save_2 = 'pre2'
			filename_prefix_list = [filename_prefix_default,filename_annot_default,filename_prefix_save_1,filename_prefix_save_2]
			
			filename_prefix = select_config['filename_prefix_default']
			filename_prefix_1 = filename_prefix
			filename_annot1 = select_config['filename_annot_default']

			if type_id2==0:
				# gene_query_vec_1 = gene_vec_1[df_gene_annot2[column_query1]>thresh_dispersions_norm]
				gene_query_vec_1 = gene_highly_variable
			
			elif type_id2==1:
				gene_query_vec_1 = gene_vec_1[0:gene_num_pre1]
			
			elif type_id2==2:
				gene_query_vec_1 = gene_vec_1[~id_query1]
				# input_filename_pre1 = '%s/%s.combine.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
				# filename_prefix_save_1 = select_config['filename_prefix_default_1']
				filename_prefix_1 = 'test_query_gene_peak.%s'%(data_file_type_query)
				filename_prefix_save_1 = '%s.%s'%(filename_prefix_1,filename_prefix_save_1)
				filename_annot1 = select_config['filename_annot_default']
				
				filename_distance_annot = '%s/%s.%d.distance_annot.1.txt'%(save_file_path,filename_prefix_save_1,peak_distance_thresh)
				input_filename_pre1 = filename_distance_annot
				input_filename_pre2 = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
				filename_peak_gene_thresh2 = '%s/%s.combine.thresh2.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
				select_config.update({'input_filename_pre1':input_filename_pre1,
										'input_filename_pre2':input_filename_pre2,
										'filename_save_thresh2':filename_peak_gene_thresh2})

				df_gene_peak_compute_1 = pd.read_csv(input_filename_pre1,index_col=False,sep='\t')
				df_gene_peak_compute_1.index = np.asarray(df_gene_peak_compute_1['gene_id'])
				print('df_gene_peak_compute_1: ',df_gene_peak_compute_1.shape)
				print(df_gene_peak_compute_1[0:2])

				# flag_computation_vec = [2]
				flag_computation_vec = [3]
				select_config.update({'flag_computation_vec':flag_computation_vec})
			else:
				gene_query_vec_1 = gene_vec_1

			if beta_mode>0:
				# gene_query_vec = gene_query_vec_1[0:100]
				gene_query_vec = gene_query_vec_1[0:20]
			else:
				gene_query_vec = gene_query_vec_1

			self.gene_query_vec = gene_query_vec
			gene_query_num = len(gene_query_vec)
			select_config.update({'feature_query_num_1':gene_query_num})
			if (verbose>0):
				print('target gene set: %d'%(gene_query_num))
				print('preview: ',np.asarray(gene_query_vec)[0:10])

			data_dir = select_config['input_dir']
			file_path_motif_score = select_config['file_path_motif_score']
			file_path_vec = [data_dir,file_path_motif_score]
			# query_mode=0:query genome-wide regulatory associations
			# query_mode=1:query a subset of regulatory associations
			query_mode = int((type_id2<=2)|(beta_mode>0))

			file_save_path2 = file_path_vec[query_mode]
			field_id = 'gene_tf'
			file_path_query = '%s/data_%s'%(file_save_path2,field_id)
			column_query = 'file_path_%s'%(field_id)
			if os.path.exists(file_path_query)==False:
				print('the directory does not exist: %s'%(file_path_query))
				os.makedirs(file_path_query,exist_ok=True)
			select_config.update({column_query:file_path_query})

			filename_prefix = 'test_%s_correlation.%s'%(field_id,data_file_type_query)
			field_id_config = 'filename_prefix_%s'%(field_id)
			select_config.update({field_id_config:filename_prefix})
			
			# estimate peak accessibility-gene expr correlation for given gene query and peak-gene links
			t_vec_1 = self.test_gene_peak_query_correlation_pre1(gene_query_vec=gene_query_vec,peak_loc_query=[],																			
																	df_gene_peak_query=df_gene_peak_1,
																	df_gene_peak_compute_1=df_gene_peak_compute_1,
																	df_gene_peak_compute_2=df_gene_peak_compute_2,
																	atac_ad=atac_ad,
																	rna_exprs=rna_exprs,
																	flag_computation_vec=flag_computation_vec,
																	highly_variable=highly_variable,
																	recompute=recompute,
																	save_mode=1,
																	save_file_path='',
																	output_filename='',
																	filename_prefix_save='',
																	verbose=verbose,select_config=select_config)

			df_gene_peak_query_thresh2, df_gene_peak_query_1 = t_vec_1 # pre-selected peak-gene link query by empirical p-value thresholds and peak-gene link query by correlation thresholds

		iter_mode = select_config['iter_mode']
		print('iter_mode: ',iter_mode)

		if iter_mode==1:
			print('computation was in batch mode')
			# print('further computation will be performed after computations from batches are merged')
			# return
			select_config = self.test_query_filename_batch_pre1(select_config=select_config)

		flag_basic_query = select_config['flag_basic_query']
		# query gene and peak basic statistics
		if flag_basic_query>0:
			filename_prefix_1 = select_config['filename_prefix_default']
			filename_prefix_2 = select_config['filename_prefix_save_default']
			filename_prefix_save = '%s.%s'%(filename_prefix_1,filename_prefix_2)
			
			output_file_path = save_file_path
			filename_save_thresh2 = select_config['filename_save_thresh2']
			input_filename = filename_save_thresh2
			print('filename_save_thresh2: %s'%(filename_save_thresh2))
			self.test_peak_gene_query_basic_1(data=[],input_filename=input_filename,
												save_mode=1,
												output_file_path=output_file_path,
												output_filename='',
												filename_prefix_save=filename_prefix_save,
												verbose=verbose,select_config=select_config)

		flag_basic_query_2 =0
		if 'flag_basic_query_2' in select_config:
			flag_basic_query_2 = select_config['flag_basic_query_2']

		# peak-gene link comparison and pre-selection 1 using distance bin and correlation bin
		peak_distance_thresh_query = 2000
		save_file_path = select_config['data_path_save_local']
		input_file_path_1 = '%s/temp1'%(save_file_path)
		file_path_query1 = input_file_path_1
		if os.path.exists(file_path_query1)==False:
			print('the directory does not exist: %s'%(file_path_query1))
			os.makedirs(file_path_query1,exist_ok=True)

		if flag_basic_query_2>0:
			column_1 = 'list_distance_annot'
			# input_filename = 'test_query_gene_peak.pbmc.df_gene_peak_annot2.1.ori.txt'
			filename_prefix_1 = select_config['filename_prefix_default']
			
			# input_filename = '%s.df_gene_peak_annot2.1.ori.txt'%(input_file_path_1,filename_prefix_1)
			input_filename = '%s/%s.feature_correlation.2.txt'%(input_file_path_1,filename_prefix_1)
			list_distance_annot = [input_filename]
			select_config.update({column_1:list_distance_annot})

			# perform peak-gene link comparison and selection
			# peak_distance_thresh_query = 2000
			distance_bin = 50
			n_bins_vec = [20,100]
			df_link_query_1, select_config = self.test_feature_link_query_compare_pre1(df_gene_peak=[],
																						df_gene_peak_combine=[],
																						input_filename='',
																						distance_bin=distance_bin,
																						n_bins_vec=n_bins_vec,
																						flag_discrete_1=1,
																						flag_discrete_2=0,
																						atac_ad=[],rna_exprs=[],
																						type_query_compare=2,
																						peak_distance_thresh=peak_distance_thresh_query,
																						save_mode=1,
																						save_file_path=file_path_query1,
																						filename_prefix_save=filename_prefix_1,
																						filename_save_annot='',
																						verbose=0,select_config=select_config)

		flag_basic_filter_1 = select_config['flag_basic_filter_1']

		# peak-gene link comparison and selection 2 for each peak linked to multiple genes
		if flag_basic_filter_1>0:
			# peak_distance_thresh_query = 2000
			type_query_compare = 2
			self.test_feature_link_query_compare_pre3(input_filename='',
														atac_ad=[],
														rna_exprs=[],
														type_query_compare=type_query_compare,
														peak_distance_thresh=peak_distance_thresh_query,
														save_mode=1,
														save_file_path='',
														filename_prefix_save='',
														filename_save_annot='',
														verbose=0,select_config=select_config)
		
		flag_cond_query_1 = 0
		if 'flag_cond_query_1' in select_config:
			flag_cond_query_1 = select_config['flag_cond_query_1']

		# compute gene-TF expression partial correlation given peak accessibility
		if flag_cond_query_1>0:
			# flag_score_pre1 : 1. compute gene-TF partial correlation given peak accessibility; 2. compute the peak-TF-gene score; 3. compute gene-TF paritial correation and the peak-TF-gene score
			# flag_compute : 1. compute the peak-TF-gene score; 2. perform peak-TF-gene link selection; 3. compute the peak-TF-gene score and perform peak-TF-gene link selection
			flag_compute = flag_cond_query_1
			column_1 = 'filename_feature_link_pre1' # file path of pre-selected peak-gene links
			if column_1 in select_config:
				filename_feature_link_pre1 = select_config[column_1]
			else:
				data_path_save_local = select_config['data_path_save_local']
				input_file_path_query= '%s/temp1'%(data_path_save_local)
				filename_prefix_1 = select_config['filename_prefix_default']
				thresh_annot = '100_0.15.500_-0.05'
				type_query_compare = select_config['type_query_compare']
				type_score_vec_query1 = [0,2]
				type_score_query1,type_score_query2 = type_score_vec_query1[0:2]
				
				filename_feature_link_pre1 = '%s/%s.df_link_query2.1.combine.%s.%d.%d_%d.2.txt'%(input_file_path_query,filename_prefix_1,thresh_annot,type_query_compare,type_score_query1,type_score_query2)
				select_config.update({column_1:filename_feature_link_pre1})

			atac_ad = self.atac_meta_ad
			rna_exprs = self.rna_exprs

			data_dir = select_config['data_path_save_1']
			file_path_motif_score = select_config['file_path_motif_score']

			# query the directories to save annotations of gene-TF links and peak-TF links
			file_path_vec = [data_dir,file_path_motif_score]
			field_vec = ['gene_tf','peak_tf']
			column_vec_query = ['file_path_%s'%(field_id) for field_id in field_vec]
			query_vec = [1,0]
			query_num1 = len(field_vec)

			for i1 in range(query_num1):
				field_id, column_query = field_vec[i1], column_vec_query[i1]
				# query_mode=0: estimate genome-wide regulatory associations
				# query_mode=1: estimate a subset of regulatory associations
				query_mode = query_vec[i1]
				if not (column_query in select_config):
					file_save_path2 = file_path_vec[query_mode]
					file_path_query = '%s/data_%s'%(file_save_path2,field_id)
					if os.path.exists(file_path_query)==False:
						print('the directory does not exist: %s'%(file_path_query))
						os.makedirs(file_path_query,exist_ok=True)
					select_config.update({column_query:file_path_query})

			# compuate peak-TF-gene score
			# compute score 1 and score 2
			self.test_feature_link_query_cond_pre1(atac_ad=atac_ad,rna_exprs=rna_exprs,flag_compute=flag_compute,
													save_mode=1,save_file_path='',verbose=0,select_config=select_config)

			type_query = 0
			flag_reduce = select_config['flag_reduce']
			if iter_mode>0:
				feature_query_num1 = query_id2-query_id1
				# interval = query_id2-query_id1
				interval = -1
			else:
				feature_query_num1 = gene_query_num
				interval = -1
			print('feature_query_num: %d, interval: %d'%(feature_query_num1,interval))
			print(type_query,flag_reduce)
			self.test_file_combine_1(feature_query_num=feature_query_num1,interval=interval,
												type_query=type_query,
												flag_reduce=flag_reduce,
												save_mode=1,
												verbose=0,select_config=select_config)

		flag_combine = 0
		# combine peak-TF-gene association scores of different subsets of peak-TF-gene links
		if flag_combine>0:
			interval = 3000
			df_link_query_pre2, df_link_query_pre2_1 = self.test_feature_link_query_combine_pre1(feature_query_num=gene_query_num,feature_query_vec=[],
																									column_vec_score=[],
																									column_vec_query=[],
																									interval=interval,
																									flag_quantile=0,
																									save_mode=1,save_mode_2=1,
																									save_file_path='',output_filename='',
																									verbose=0,select_config=select_config)

		flag_select_pre2 = 0
		# perform feature link selection
		if flag_select_pre2>0:
			dict_query_1 = self.test_feature_link_query_select_pre1(thresh_vec_query=[],
																	save_mode=1,
																	save_mode_2=1,
																	save_file_path='',
																	verbose=0,select_config=select_config)

		# flag_basic_filter_2 = select_config['flag_basic_filter_2']
		# # perform feature link comparison and selection after computing the peak-TF-gene score
		# if flag_basic_filter_2==2:
		# 	t_vec_1 = self.test_feature_link_query_compare_1(data=[],input_filename='',atac_ad=[],rna_exprs=[],type_query_compare=2,peak_distance_thresh=2000,sub_sample_num=-1,save_mode=1,save_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)
		# 	df_feature_link_query, df_feature_link, dict_query_1 = t_vec_1[0:3]

	## ====================================================
	# query peak-gene TSS distance for the peak-gene links
	def test_query_distance_annot_1(self,data=[],df_annot=[],column_distance='distance',column_idvec=['peak_id','gene_id'],
										column_vec_sort=[],ascending_vec=[],flag_sort=1,save_mode=1,verbose=0,select_config={}):

		"""
		query peak-gene TSS distance for the peak-gene links
		:param data: (dataframe) annotations of peak-gene links
		:param df_annot: (dataframe) annotations of genome-wide peak-gene links including peak-gene TSS distances
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param column_vec_sort: (array or list) columns by which to sort the dataframe
		:param ascending_vec: (array or list) boolen values indicating sorting the datafram by ascending or descending order using the corresponding column in column_vec_sort
		:param flag_sort: indicator of whether to sort the dataframe
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) updated annotations of peak-gene links including peak-gene TSS distances
		"""

		column_id2, column_id1 = column_idvec
		flag_distance = 1
		df_query1 = data
		verbose_internal = self.verbose_internal
		if (column_distance in df_query1.columns):
			id1 = (pd.isna(df_query1[column_distance])==True)
			df_query2 = df_query1.loc[id1,:]
			# print('df_query2 ',df_query2.shape)
			query_num_distance = np.sum(id1)
			if query_num_distance==0:
				flag_distance = 0

		if flag_distance>0:
			if len(df_annot)==0:
				df_gene_peak_distance = self.df_gene_peak_distance
			else:
				df_gene_peak_distance = df_annot

			if verbose_internal==2:
				print('peak-gene links by distance threshold, dataframe of size ',df_gene_peak_distance.shape)
			df_list = [df_query1,df_gene_peak_distance]
			column_query_1 = [column_distance]
			reset_index = True
			# copy specified columns from the second dataframe to the first dataframe
			df_query1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec,
														column_vec=column_query_1,
														df_list=df_list,
														type_id_1=0,type_id_2=0,
														reset_index=reset_index,select_config=select_config)

		if flag_sort>0:
			column_distance_abs = '%s_abs'%(column_distance)
			if len(column_vec_sort)==0:
				column_vec_sort = [column_id2,column_distance_abs]

			if (column_distance_abs in column_vec_sort):
				df_query1[column_distance_abs] = df_query1[column_distance].abs()

			if len(ascending_vec)==0:
				query_num1 = len(column_vec_sort)
				ascending_vec = [True]*query_num1

			df_query1 = df_query1.sort_values(by=column_vec_sort,ascending=ascending_vec)
			t_columns = df_query1.columns.difference([column_distance_abs],sort=False)
			df_query1 = df_query1.loc[:,t_columns]

		return df_query1

	## ====================================================
	# query alternative potential peak-gene links for comparison with the pre-selected peak-gene links
	def test_query_peak_gene_link_pre2(self,df_gene_peak=[],df_gene_peak_compare=[],column_distance='distance',thresh_distance=100,type_compare=0,save_mode=1,verbose=0,select_config={}):

		"""
		query alternative potential peak-gene links for comparison with the pre-selected peak-gene links
		:param df_gene_peak: (dataframe) annotations of pre-selected peak-gene links (candidate links)
		:param df_gene_peak_compare: (dataframe) annotations of possible peak-gene links from which to identify alternative links to compare with the candidate links
		:param column_distance: (str) column representing peak-gene distance in the peak annotations
		:param thresh_distance: (int) threshold on the difference between peak-gene distance in the alternative peak-gene links and the maximal peak-gene distanc in the candidate links
		:param type_compare: (int) the type of alternative peak-gene links:0: peak-gene links not sharing target genes with the candidate links;
																		   1: peak-gene links which are different from candidate peaks but may contain target genes included in candidate links;
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) possible peak-gene links with added annotations
				 2. (dataframe) selected alternative peak-gene links to compare with the candidate peak-gene links
		"""

		column_idvec = ['gene_id','peak_id']
		column_id1, column_id2 = column_idvec[0:2]
		
		df_gene_peak_query = df_gene_peak
		feature_query_vec_2 = df_gene_peak_query[column_id2].unique()	# the current candidate peak query
		feature_query_num2 = len(feature_query_vec_2)
		print('candidate peak number: %d'%(feature_query_num2))
		# print(feature_query_vec_2[0:2])

		feature_query_vec_1 = df_gene_peak_query[column_id1].unique()	# the current target gene query
		feature_query_num1 = len(feature_query_vec_1)
		print('gene nubmer: %d'%(feature_query_num1))
		# print(feature_query_vec_1[0:2])

		df_gene_peak_compare_pre1_1 = df_gene_peak_compare
		id1 = df_gene_peak_compare_pre1_1[column_id2].isin(feature_query_vec_2)
		df_gene_peak_compare_pre1 = df_gene_peak_compare_pre1_1.loc[id1,:]  # peak-gene links associated with peaks in the candidate links

		if type_compare==0:
			id2 = df_gene_peak_compare_pre1[column_id1].isin(feature_query_vec_1) # peak-gene links associated with the target genes in the candidate links
			id_pre1 = (~id2)
			df_gene_peak_query_compare = df_gene_peak_compare_pre1.loc[id_pre1,:] # peak-gene links not associated with the target genes in the candidate links
			df_gene_peak_query_compare.index = test_query_index(df_gene_peak_query_compare,column_vec=column_idvec) # reindex the peak-gene links by including the peak and gene index information
		else:
			df_gene_peak_compare_pre1.index = test_query_index(df_gene_peak_compare_pre1,column_vec=column_idvec)
			df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_idvec)

			query_id_ori = df_gene_peak_compare_pre1.index
			query_id2 = df_gene_peak_query.index
				
			query_id2_2 = query_id_ori.difference(query_id2,sort=False)
			df_gene_peak_query_compare = df_gene_peak_compare_pre1.loc[query_id2_2,:] # peak-gene links different from the candidate links, which may contain target genes included in the candiate links

		print('initial alternative potential peak-gene links, dataframe of size ',df_gene_peak_query_compare.shape)

		df_query_compare_pre1 = []
		if thresh_distance>0:
			column_distance_abs = '%s_abs'%(column_distance)
			df_gene_peak_query[column_distance_abs] = df_gene_peak_query[column_distance].abs()
			
			# group the peak-gene links by peak
			df_group_query = df_gene_peak_query.groupby(by=[column_id2])

			column_value_1 = column_distance_abs  # column representing absolute peak-gene distance
			df_query_1 = df_group_query[[column_value_1]].max()
			column_query1 = 'distance_max_1'
			
			print('maximal peak-gene distance for peak loci, dataframe of size ',df_query_1.shape)
			print('data preview:')
			print(df_query_1[0:2])

			df_gene_peak_query_compare.index = np.asarray(df_gene_peak_query_compare[column_id2])
			query_id1 = df_gene_peak_query_compare.index

			df_gene_peak_query_compare[column_query1] = df_query_1.loc[query_id1,column_value_1] # the maximal peak-gene distance for the peak loci in the candidate links;
			df_gene_peak_query_compare[column_distance_abs] = df_gene_peak_query_compare[column_distance].abs()

			thresh_value_1 = thresh_distance
			# retain alternative peak-gene links with peak-gene distance below the threshold which depends on the maximal peak-gene distance for each peak in the candidate links
			# if the peak-gene link has distance above the threshold, it will not filter out candidate links by the current comparing approach
			id_1 = (df_gene_peak_query_compare[column_value_1]<(df_gene_peak_query_compare[column_query1]+thresh_value_1))
			id_2 = (~id_1)

			df_query_compare_pre1 = df_gene_peak_query_compare.loc[id_1,:]
			df_query_compare_pre2 = df_gene_peak_query_compare.loc[id_2,:]

			print('alternative potential peak-gene links within distance threshold, dataframe of size ',df_query_compare_pre1.shape)
			print('data preview: ')
			print(df_query_compare_pre1[0:2])
			print('alternative potential peak-gene links above distance threshold, dataframe of size ',df_query_compare_pre2.shape)
			print('data preview: ')
			print(df_query_compare_pre2[0:2])

		return df_gene_peak_query_compare, df_query_compare_pre1

	## ====================================================
	# parameter configuration for peak-gene link comparison
	def test_query_config_link_compare_1(self,verbose=0,select_config={}):

		"""
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing updated parameters, including the thresholds to use for peak-gene link comparison
				 2. the number of threshold types to use for peak-gene link comparison
		"""

		thresh_value_1 = 100 # distance threshold
		thresh_value_2 = 0.15 # correlation threshold
		thresh_value_1_2 = 500 # distance threshold
		thresh_value_2_2 = -0.05 # correlation threshold

		thresh_vec_query = [[thresh_value_1,thresh_value_2],[thresh_value_1_2,thresh_value_2_2]]
		thresh_type = len(thresh_vec_query)
		select_config.update({'thresh_vec_group1':thresh_vec_query})
			
		thresh_value_1_3 = -50 # distance threshold
		thresh_value_2_3 = 0.20 # correlation threshold
		thresh_vec_query_1 = [150,[0.3,0.1]]
		thresh_vec_query_2 = [[thresh_value_1_3,thresh_value_2_3]]
		thresh_vec_group2 = [thresh_vec_query_1, thresh_vec_query_2]
		select_config.update({'thresh_vec_group2':thresh_vec_group2})
		thresh_type = 3

		return select_config, thresh_type

	## ====================================================
	# perform peak-gene link comparison and selection
	def test_feature_link_query_compare_pre1(self,df_gene_peak=[],df_gene_peak_combine=[],input_filename='',
												distance_bin=50,n_bins_vec=[20,100],flag_discrete_1=1,flag_discrete_2=0,
												atac_ad=[],rna_exprs=[],type_query_compare=2,peak_distance_thresh=2000,
												save_mode=1,save_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		perform peak-gene link comparison and selection
		:param df_gene_peak: (dataframe) annotations of the pre-selected candidate peak-TF-gene links
		:param df_gene_peak_combine: (dataframe) the peak-TF-gene links from which to select the alternative potential links 
												 to compare with the given candidate peak-TF-gene links
		:param input_filename: (str) path of the file which saved annotations of the candidate peak-TF-gene links
		:param distance_bin: the distance bin size (unit:Kb)
		:param n_bins_vec: (array or list) the number of GC content bins for peak loci and the number of peak-gene distance bins
		:param flag_discrete_1: indicator of whether to query GC content bins for peaks and distance bins for peak-gene links
		:param flag_discrete_2: indicator of whether to estimate the empirical distribution of the difference of peak-gene correlations in different paired bins
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param type_query_compare: the type of comparison to perform for the peak-gene links
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) annotations of peak-gene links selected after peak-gene link comparsion;
				 2. dictionary containing updated parameters;
		"""
		
		# combine peak accessibility-gene expression correlations for the two groups of peak-gene associations 
		# (the pre-selected and the added peak-gene associations)
		flag_correlation_2=0
		if 'flag_correlation_2' in select_config:
			flag_correlation_2 = select_config['flag_correlation_2']
		
		df_gene_peak_compute_pre1 = df_gene_peak_combine
		recompute_distance = 1
		verbose_internal = self.verbose_internal
		input_filename_1 = input_filename

		if save_file_path=='':
			save_file_path_1 = select_config['data_path_save_local']
			save_file_path = '%s/temp1'%(save_file_path_1)
			if os.path.exists(save_file_path)==False:
				print('the directory does not exist: %s'%(save_file_path))
				os.makedirs(save_file_path,exist_ok=True)
		output_file_path = save_file_path

		if flag_correlation_2>0:
			df_gene_peak_compare_pre1_1 = self.df_gene_peak_distance
			if len(df_gene_peak)==0:
				if input_filename=='':
					input_filename = select_config['filename_save_thresh2']
				df_gene_peak_query_thresh2 = pd.read_csv(input_filename,index_col=0,sep='\t')
			else:
				df_gene_peak_query_thresh2 = df_gene_peak

			# interval_peak_corr = 500
			interval_peak_corr = -1
			query_id1, query_id2 = -1, -1
			flag_distance_annot = 0
			# flag_distance_annot = 1
			filename_distance_annot = '%s/%s.distance_annot.1.txt'%(save_file_path,filename_prefix_save)
			select_config.update({'filename_distance_annot':filename_distance_annot})
			
			if filename_prefix_save=='':
				# filename_prefix_save = select_config['filename_prefix_default_1']
				filename_prefix_save = select_config['filename_prefix_default']

			if filename_save_annot=='':
				filename_save_annot = '1'

			column_correlation=['spearmanr','pval1','pval1_ori']
			column_idvec = ['gene_id','peak_id']
			column_label = 'label_corr'

			column_id1, column_id2 = column_idvec[0:2]
			column_distance = 'distance'
			thresh_distance = 100
			type_compare = 0
			# query alternative potential peak-gene links for comparison
			df_gene_peak_compare_1, df_gene_peak_query_compare = self.test_query_peak_gene_link_pre2(df_gene_peak=df_gene_peak_query_thresh2,
																										df_gene_peak_compare=df_gene_peak_compare_pre1_1,
																										column_distance=column_distance,
																										thresh_distance=thresh_distance,
																										type_compare=type_compare,
																										save_mode=1,verbose=verbose,select_config=select_config)

			query_id_compare = df_gene_peak_query_compare.index.copy()

			# compute peak accessibility-gene expression correlation
			if flag_correlation_2 in [1]:
				iter_mode_1 = 0
				# iter_idvec = []
				iter_idvec = [-1,-1]
				if ('query_id1' in select_config) and ('query_id2' in select_config):
					start_id1 = select_config['query_id1']
					start_id2 = select_config['query_id2']
					if (start_id1>=0) and (start_id2>start_id1):
						# iter_mode_1 = 1
						# iter_idvec = [start_id1,start_id2]

						filename_prefix_save_1 = filename_prefix_save
						annot_str1 = '%d_%d'%(start_id1,start_id2)
						b = filename_prefix_save.find(annot_str1)
						if b<-0:
							filename_prefix_save = '%s.%d_%d'%(filename_prefix_save_1,start_id1,start_id2)
				
				filename_save_annot_2 = 'feature_correlation'
				# search peak-gene association; 
				# compute peak accessibility-gene expr correlation
				df_gene_peak_query_compare2 = self.test_gene_peak_query_basic_filter_1_pre1(df_gene_peak_query=df_gene_peak_query_thresh2,
																								df_gene_peak_compare=df_gene_peak_query_compare,
																								atac_ad=atac_ad,
																								rna_exprs=rna_exprs,
																								column_correlation=column_correlation,
																								column_idvec=column_idvec,
																								column_label=column_label,
																								interval_peak_corr=interval_peak_corr,
																								iter_idvec=iter_idvec,
																								flag_distance_annot=flag_distance_annot,
																								save_mode=1,
																								output_file_path=output_file_path,
																								output_filename='',
																								filename_prefix_save=filename_prefix_save,
																								filename_save_annot=filename_save_annot_2,
																								verbose=verbose,select_config=select_config)
				# if iter_mode_1==0:
				if iter_mode_1 in [0,1]:
					# combine the added peak-gene correlations and the current peak-gene correlations
					if len(df_gene_peak_query_compare2)>0:
						list1 = [df_gene_peak_query_thresh2,df_gene_peak_query_compare2]
						df_gene_peak_compute_pre1 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
						column_idvec_query = column_idvec
						df_gene_peak_compute_pre1 = df_gene_peak_compute_pre1.drop_duplicates(subset=column_idvec_query)
						print('pre-selected peak-gene links, dataframe of size ',df_gene_peak_query_thresh2.shape)
						print('alternative potential peak-gene links, dataframe of size ',df_gene_peak_query_compare2.shape)
						print('combined peak-gene links for comparison, dataframe of size ',df_gene_peak_compute_pre1.shape)

			elif flag_correlation_2 in [2]:
				# load peak-gene correlations from saved files
				# combine the added peak-gene correlations with the peak-gene correlations of the pre-selected links
				column_1 = 'list_distance_annot'
				list_distance_annot = select_config[column_1]
				query_num1 = len(list_distance_annot)
				list1 = []
				for i1 in range(query_num1):
					input_filename_query = list_distance_annot[i1]
					df_query = pd.read_csv(input_filename_query,index_col=0,sep='\t')
					print('peak-gene links, dataframe of size ',df_query.shape,i1)
					print('data loaded from %s'%(input_filename_query))
					list1.append(df_query)
				
				if query_num1>1:
					df_gene_peak_compare_1 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
				else:
					df_gene_peak_compare_1 = list1[0]

				df_gene_peak_compare_1.index = test_query_index(df_gene_peak_compare_1,column_vec=column_idvec)
				query_idvec = df_gene_peak_compare_1.index.unique()
				query_id_compare_2 = query_idvec.intersection(query_id_compare,sort=False)
				df_gene_peak_compare_2 = df_gene_peak_compare_1.loc[query_id_compare_2,:]

				print('alternative potential peak-gene links, dataframe of size ',df_gene_peak_compare_2.shape)

				list2 = [df_gene_peak_query_thresh2,df_gene_peak_compare_2]
				query_id_2 = df_gene_peak_query_thresh2.index
				df_gene_peak_compute_pre1 = pd.concat(list2,axis=0,join='outer',ignore_index=False)
				df_gene_peak_compute_pre1 = df_gene_peak_compute_pre1.drop_duplicates(subset=column_idvec)

				column_query = 'label_thresh2'
				df_gene_peak_compute_pre1.loc[query_id_2,column_query] = 1
				print('pre-selected peak-gene links, dataframe of size ',df_gene_peak_query_thresh2.shape)
				print('combined peak-gene links for comparison, dataframe of size ',df_gene_peak_compute_pre1.shape)

				# query peak-gene distance
				df_gene_peak_distance = self.df_gene_peak_distance
				column_distance = 'distance'
				column_distance_abs = '%s_abs'%(column_distance)
				column_vec_sort = [column_id2,column_distance_abs]
				ascending_vec = [True,True]
				# query peak-gene distance
				df_gene_peak_compute_pre1 = self.test_query_distance_annot_1(data=df_gene_peak_compute_pre1,
																				df_annot=df_gene_peak_distance,
																				column_distance=column_distance,
																				column_idvec=column_idvec,
																				column_vec_sort=column_vec_sort,
																				ascending_vec=ascending_vec,
																				flag_sort=1,
																				save_mode=1,verbose=verbose,select_config=select_config)

				column_1 = 'filename_save_thresh2_combine'
				if column_1 in select_config:
					output_filename = select_config[column_1]
				else:
					filename_query1 = select_config['filename_save_thresh2']
					b = filename_query1.find('.txt')
					output_filename = '%s.1_2.txt'%(filename_query1[0:b])
					select_config.update({column_1:output_filename})

				df_gene_peak_compute_pre1.index = np.asarray(df_gene_peak_compute_pre1[column_id1])
				df_gene_peak_compute_pre1.to_csv(output_filename,sep='\t')

		# flag_correlation_query1 = 1
		flag_discrete_query1 = 0
		# save_file_path = select_config['data_path_save_local']
		
		if 'flag_discrete_query1' in select_config:
			flag_discrete_query1 = select_config['flag_discrete_query1']

		# query distance bin for peak-gene links and GC bin for peaks
		df_gene_peak_compute_1 = []
		if flag_discrete_query1>0:
			input_filename = select_config['input_filename_peak']
			df_peak_annot = pd.read_csv(input_filename,index_col=False,header=None,sep='\t')
			df_peak_annot.columns = ['chrom','start','stop','strand','GC','name']
			df_peak_annot.index = test_query_index(df_peak_annot,column_vec=['chrom','start','stop'],symbol_vec=[':','-'])
			column_id_query = 'spearmanr'
			
			if len(df_gene_peak_compute_pre1)==0:
				input_filename_2 = select_config['filename_save_thresh2']
				df_gene_peak_compute_1 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			else:
				df_gene_peak_compute_1 = df_gene_peak_compute_pre1
				# recompute_distance = 0
			
			print('query distance bin and correlation bin of peak-gene link')
			print('peak-gene links, dataframe of size ',df_gene_peak_compute_1.shape)

			column_idvec = ['peak_id','gene_id']
			column_id2, column_id1 = column_idvec
			column_distance = 'distance'
			
			# query peak-gene distance
			df_gene_peak_distance = self.df_gene_peak_distance
			column_distance = 'distance'
			column_distance_abs = '%s_abs'%(column_distance)
			column_vec_sort = [column_id2,column_distance_abs]
			ascending_vec = [True,True]
			if recompute_distance>0:
				# query peak-gene distance
				df_gene_peak_compute_1 = self.test_query_distance_annot_1(data=df_gene_peak_compute_1,
																			df_annot=df_gene_peak_distance,
																			column_distance=column_distance,
																			column_idvec=column_idvec,
																			column_vec_sort=column_vec_sort,
																			ascending_vec=ascending_vec,
																			flag_sort=1,
																			save_mode=1,verbose=verbose,select_config=select_config)

			filename_prefix_save_1 = select_config['filename_prefix_default']
			output_file_path1 = save_file_path
			input_file_path = save_file_path

			flag_1=flag_discrete_1
			if flag_1>0:
				# flag_load1 = 0
				flag_load1 = 1
				# input_filename = '%s/%s.distance.%d.annot2.txt'%(input_file_path,filename_prefix_save_1,distance_bin)
				input_filename = '%s/%s.distance.%d.annot2.sort.txt'%(input_file_path,filename_prefix_save_1,distance_bin)
				overwrite_2 = False
				flag_query = 1
				if (os.path.exists(input_filename)==False):
					print('the file does not exist: %s'%(input_filename))
					flag_load1=0
				else:
					print('the file exists: %s'%(input_filename))
					if (overwrite_2==False):
						df_link_query = pd.read_csv(input_filename,index_col=0,sep='\t')
						flag_query = 0

				if flag_query>0:
					df_link_query, df_annot1, dict_annot1 = self.test_query_link_correlation_distance_1(df_link_query=df_gene_peak_compute_1,
																										df_feature_annot=df_peak_annot,
																										column_id_query=column_id_query,
																										n_bins_vec=n_bins_vec,
																										distance_bin=distance_bin,
																										flag_unduplicate=1,
																										save_mode=1,
																										output_file_path=output_file_path1,
																										filename_prefix_save=filename_prefix_save_1,
																										select_config=select_config)

			flag_2=flag_discrete_2
			if flag_2>0:
				distance_bin_query = 50
				filename_annot = str(distance_bin_query)
				type_id_1 = 0
				filename_query = '%s/%s.group_feature_compare.%s.annot1.%d.txt'%(save_file_path,filename_prefix_save_1,filename_annot,type_id_1)
				if os.path.exists(filename_query)==True:
					print('the file exists: %s'%(filename_query))
				
				overwrite_2 = False
				# overwrite_2 = True
				if (os.path.exists(filename_query)==False) or (overwrite_2==True):
					distance_bin_query = 50
					column_vec_query = ['distance_bin_%d'%(distance_bin_query),'spearmanr','GC_bin_20']
					column_annot = ['distance_pval1','distance_pval2']
					column_annot_query1 = ['%s_%d'%(column_query,distance_bin_query) for column_query in column_vec_query]
					normalize_type = 'uniform'
					n_sample = 500
					df_link_query.index = test_query_index(df_link_query,column_vec=column_idvec)
					# df_link_query = df_link_query.drop_duplicates(subset=column_idvec)
					# estimate the empirical distribution of the difference of peak-gene correlations in different paired bins
					df_annot, df_annot2, dict_query1, dict_query2 = self.test_attribute_query_distance_2(df_link_query,df_annot=[],
																											column_vec_query=column_vec_query,
																											column_annot=column_annot_query1,
																											n_sample=n_sample,
																											distance_bin=distance_bin_query,
																											distance_tol=2,
																											type_id_1=type_id_1,
																											save_mode=1,output_file_path=output_file_path1,
																											filename_prefix_save=filename_prefix_save_1,
																											select_config=select_config)

		flag_basic_query_2 =0
		if 'flag_basic_query_2' in select_config:
			flag_basic_query_2 = select_config['flag_basic_query_2']

		df_link_query_1 = []
		if flag_basic_query_2>0:
			data_file_type_query = select_config['data_file_type']
			column_idvec = ['peak_id','gene_id']
			feature_type_vec = ['peak','gene']
			column_id2, column_id1 = column_idvec[0:2]
			column_vec_1 = column_idvec
			dict_feature_type = dict(zip(column_idvec,feature_type_vec))
			
			type_query_1 = 0 # compare between genome-wide links with the peak
			# type_query_1 = 1 # compare between initially selected links with the peak
			# type_query_1 = 2   # compare between pre-selected links with the peak
			if 'type_query_compare' in select_config:
				type_query_1 = select_config['type_query_compare']
			else:
				select_config.update({'type_query_compare':type_query_1})
			
			filename_prefix_save_1 = select_config['filename_prefix_default']

			input_filename = select_config['filename_save_thresh2']
			df_gene_peak_query_thresh2 = pd.read_csv(input_filename,index_col=0,sep='\t')
			if verbose_internal==2:
				print('pre-selected peak-gene links, dataframe of size ',df_gene_peak_query_thresh2.shape)
			
			if len(df_gene_peak_compute_1)==0:
				type_query_1 = 2
				df_gene_peak_query = df_gene_peak_query_thresh2
			else:
				df_gene_peak_query = df_gene_peak_compute_1

			for column_query in column_vec_1:
				feature_query = df_gene_peak_query[column_query].unique()
				feature_query_num = len(feature_query)
				feature_type = dict_feature_type[column_query]
				print('feature type: %s, number: %d'%(feature_type,feature_query_num))

			column_correlation=['spearmanr','pval1','pval1_ori']
			peak_distance_thresh_1 = peak_distance_thresh
			column_distance, column_corr_1 = 'distance', column_correlation[0]
			# column_label = 'label_corr'
			column_label = 'label_thresh2'

			if not (column_label in df_gene_peak_query):
				df_gene_peak_query_thresh2.index = test_query_index(df_gene_peak_query_thresh2,column_vec=column_idvec)
				df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_idvec)
				query_id_2 = df_gene_peak_query_thresh2.index
				query_id_1 = df_gene_peak_query.index
				query_id_pre2 = query_id_1.intersection(query_id_2,sort=False)
				df_gene_peak_query.loc[query_id_pre2,column_label] = 1

			# save_file_path2 = '%s/temp1'%(save_file_path)
			# if os.path.exists(save_file_path2)==False:
			# 	print('the directory does not exist: %s'%(save_file_path2))
			# 	os.makedirs(save_file_path2,exist_ok=True)
			# output_file_path = save_file_path2

			field_query = ['distance','correlation']
			column_vec_query = [column_distance, column_corr_1]
			column_score = column_corr_1
			flag_basic_query = 3
			type_score = 0 # type_score = 0, compare the absolute score; type_score = 1, compare the original score; type_score = 2, compare the positive score

			type_query = 0
			thresh_type = 3
			flag_config=1
			if flag_config>0:
				# thresh_value_1 = 100 # distance threshold
				# thresh_value_2 = 0.15 # correlation threshold
				# thresh_value_1_2 = 500 # distance threshold
				# thresh_value_2_2 = -0.05 # correlation threshold

				# thresh_vec_query = [[thresh_value_1,thresh_value_2],[thresh_value_1_2,thresh_value_2_2]]
				# thresh_type = len(thresh_vec_query)
				# select_config.update({'thresh_vec_group1':thresh_vec_query})
			
				# thresh_value_1_3 = -50 # distance threshold
				# thresh_value_2_3 = 0.20 # correlation threshold
				# thresh_vec_query_1 = [150,[0.3,0.1]]
				# thresh_vec_query_2 = [[thresh_value_1_3,thresh_value_2_3]]
				# thresh_vec_group2 = [thresh_vec_query_1, thresh_vec_query_2]
				# select_config.update({'thresh_vec_group2':thresh_vec_group2})
				# thresh_type = 3
				# parameter configuration for peak-gene link comparison
				select_config, thresh_type = self.test_query_config_link_compare_1(select_config=select_config)

			filename_prefix_save = filename_prefix_save_1
			df_gene_peak_distance = self.df_gene_peak_distance
			df_gene_peak_query_1 = df_gene_peak_query.copy()
			column_score_query = column_corr_1

			type_score_vec_query1 = [0,2]
			for type_score in type_score_vec_query1:
				print('perform peak-gene link comparison')
				start = time.time()
				if type_score in [2]:
					id1 = (df_gene_peak_query_1[column_score_query]>0) # keep the positive peak-gene correlation
					df_gene_peak_query = df_gene_peak_query_1.loc[id1,:]

				# print('type_score ',type_score)
				if type_score in [0]:
					print('compare using absolute peak accessibility-gene expression correlation')
				elif type_score in [2]:
					print('compare peak-gene links with positive correlation')
				else:
					print('compare using original peak accessibility-gene expression correlation')

				if verbose_internal==2:
					print('peak-gene links, dataframe of size ',df_gene_peak_query.shape)
					print('data preview: ')
					print(df_gene_peak_query[0:2])
				df_link_query_1 = self.test_gene_peak_query_basic_filter_1(peak_id=[],df_peak_query=[],
																				df_gene_peak_query=df_gene_peak_query,
																				df_gene_peak_distance_annot=df_gene_peak_distance,
																				field_query=field_query,
																				column_vec_query=column_vec_query,
																				column_score=column_score,
																				peak_distance_thresh=peak_distance_thresh_1,
																				thresh_vec_compare=[],
																				column_label=column_label,
																				thresh_type=thresh_type,
																				flag_basic_query=flag_basic_query,
																				flag_unduplicate=1,
																				type_query_compare=type_query_1,
																				type_score=type_score,
																				type_id_1=0,
																				save_mode=1,
																				output_file_path=output_file_path,
																				filename_prefix_save=filename_prefix_save,
																				verbose=verbose,select_config=select_config)
				stop = time.time()
				print('perform peak-gene link query comparison used: %.2fs'%(stop-start))
				
			# flag_combine_query = 0
			flag_combine_query = 1
			if flag_combine_query>0:
				# combine peak-gene links selected using both positive and negative correlations for link comparison with peak-gene links selected using positive correlations only for comparison
				df_link_query_2, select_config = self.test_feature_link_score_combine_pre1(data=[],save_mode=1,verbose=0,select_config=select_config)

		return df_link_query_2, select_config
		
	## ====================================================
	# perform peak-gene link comparison and selection for each peak linked to multiple genes
	def test_feature_link_query_compare_pre3(self,input_filename='',atac_ad=[],rna_exprs=[],type_query_compare=2,peak_distance_thresh=2000,
												save_mode=1,save_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		perform peak-gene link comparison and selection for each peak linked to multiple genes
		:param input_filename: (str) path of the file which saved annotations of the pre-selected candidate peak-TF-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param type_query_compare: the type of comparison to perform for the peak-gene links
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-gene links selected after peak-gene link comparsion
		"""

		flag_basic_filter_1=1
		if flag_basic_filter_1==1:
			# peak-gene association comparison and selection
			column_correlation = select_config['column_correlation'] # ['spearmanr','pval1_ori','pval1']
			column_corr_1, column_pval_ori = column_correlation[0], column_correlation[2]
			column_pval_1 = column_correlation[1]

			field_query_1 = [column_corr_1,column_pval_ori]
			field_query_2 = [column_corr_1,column_pval_ori,column_pval_1]
			
			column_distance = 'distance'
			column_vec_2 = field_query_2+[column_distance]

			column_idvec = ['gene_id','peak_id']
			column_id1, column_id2 = column_idvec[0:2]
			field_query_annot = [column_id1,column_id2]
			
			save_file_path = select_config['data_path_save_local']
			save_file_path2 = '%s/temp1'%(save_file_path)
			input_file_path = save_file_path2

			# filename_prefix_save_1 = select_config['filename_prefix_default_1']
			filename_prefix_save_1 = select_config['filename_prefix_default']
			print('filename_prefix_save_1: %s'%(filename_prefix_save_1))
			
			type_id_1 = 1 # keep positive or negative correlation information
			thresh_type = 3 # three thresholds
			# type_query_2 = 2 # compare within pre-selected peak-gene link query
			type_query_2 = select_config['type_query_compare']

			filename_save_annot_1 = select_config['filename_annot_basic_filter']
			filename_save_annot = '%s.%d.%d.%d'%(filename_save_annot_1,type_id_1,thresh_type,type_query_2)
			input_filename_2 = select_config['filename_save_thresh2_2']
			
			index_col = False
			df_gene_peak_query_1 = pd.read_csv(input_filename_2,index_col=index_col,sep='\t')	# the peak-gene link with the specific peak query
			df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1[column_id1])
			print(input_filename_2)
			print('peak-gene link, dataframe of size',df_gene_peak_query_1.shape)
			print(df_gene_peak_query_1[0:2])
			
			gene_query_vec = df_gene_peak_query_1[column_id1].unique()
			gene_query_num = len(gene_query_vec)
			print('gene_query_vec: %d'%(gene_query_num))

			flag_append = 0
			type_query = 1 # filter for peak linked with one gene and peak linked with multiple genes in the given group of gene query
			beta_mode = select_config['beta_mode']
			if beta_mode>0:
				sel_num1 = 100
				gene_query_1 = gene_query_vec[0:sel_num1]
				# df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1[column_id1])
				
				df_gene_peak_query1_ori = df_gene_peak_query_1.copy()
				df_gene_peak_query_1 = df_gene_peak_query_1.loc[gene_query_1,:]
				print('peak-gene link: ',df_gene_peak_query_1.shape)
				
			flag_1=1
			if flag_1>0:
				df_gene_peak_query_compute_1 = df_gene_peak_query_1.copy()
				# peak_distance_thresh_compare = 50
				peak_distance_thresh_compare = 100
				type_query = 0
				correlation_type = [0]
				flag_computation_query = 1
				flag_combine_query = 1

				# filename_prefix_default_1 = select_config['filename_prefix_default_1']
				filename_prefix_default_1 = select_config['filename_prefix_default']
				filename_prefix_save_1 = '%s.link_query1'%(filename_prefix_default_1)
				file_path_basic_filter = save_file_path2
				select_config.update({'file_path_basic_filter':file_path_basic_filter})

				thresh1_ratio, thresh1, distance_tol_1 = 1.0, 0.15, 100
				thresh2_ratio, thresh2, distance_tol_2 = 1.0, -0.05, 500
				thresh_vec_1 = [distance_tol_1, thresh1]
				thresh_vec_2 = [distance_tol_2, thresh2]
				thresh_vec_compare = [thresh_vec_1,thresh_vec_2]

				select_config.update({'thresh_vec_compare':thresh_vec_compare})
				# compare alternative peak-gene associations
				df_gene_peak_query_2 = self.test_gene_peak_query_correlation_basic_filter_pre1(gene_query_vec=[],df_gene_peak_query_compute=df_gene_peak_query_compute_1,
																								df_gene_peak_query=df_gene_peak_query_1,
																								peak_distance_thresh=peak_distance_thresh,
																								peak_distance_thresh_compare=peak_distance_thresh_compare,
																								df_peak_query=[],
																								peak_loc_query=[],
																								atac_ad=atac_ad,
																								rna_exprs=rna_exprs,
																								highly_variable=False,
																								interval_peak_corr=50,
																								interval_local_peak_corr=10,
																								type_id_1=type_query,
																								correlation_type=[],
																								flag_computation_1=flag_computation_query,
																								flag_combine_1=flag_combine_query,
																								input_file_path='',
																								input_filename='',
																								input_filename_2='',					
																								save_mode=1,
																								save_file_path='',
																								output_filename='',
																								filename_prefix_save=filename_prefix_save,																		
																								verbose=verbose,select_config=select_config)

	## ====================================================
	# compute peak-TF-gene association scores
	# perform peak-TF-gene link selection
	def test_feature_link_query_cond_pre1(self,df_feature_link=[],atac_ad=[],rna_exprs=[],flag_compute=1,save_mode=1,save_file_path='',verbose=0,select_config={}):

		"""
		compute peak-TF-gene association scores;
		perform peak-TF-gene link selection;
		:param df_feature_link: (dataframe) annotations of peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param flag_compute: indicator of the type of computation to perform:
							 1. calculate peak-TF-gene association scores;
							 2. select peak-TF-gene links;
							 3. perform both 1 and 2;
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) annotations of peak-TF-gene links including association scores;
				 2. (dataframe) updated annotations of initial peak-TF-gene links including associations scores and labels of which links are selected by specific thresholds;
				 2. (dataframe) annotations of selected peak-TF-gene links;
		"""
		
		# file_path1 = self.save_path_1
		# query file paths
		flag_cond_query_1 = 0
		if 'flag_cond_query_1' in select_config:
			flag_cond_query_1 = select_config['flag_cond_query_1']

		beta_mode = select_config['beta_mode']
		data_file_type = select_config['data_file_type']
		data_file_type_query = data_file_type
		save_file_path = select_config['data_path_save_local']
		# thresh_motif_1 = select_config['thresh_motif_1']
		# file_save_path2 = '%s/motif_score_%s'%(save_file_path,thresh_motif_1)
		column_1 = 'thresh_motif_1'
		if column_1 in select_config:
			thresh_motif_1 = select_config[column_1] # threshold on motif scores from motif scanning to identify motif presence in peak loci
		else:
			# thresh_motif_1 = '5e-05'
			thresh_motif_1 = '1'
			select_config.update({column_1:thresh_motif_1})
		
		filename_annot_default = data_file_type_query
		filename_annot_motif_score = '%s.%s'%(filename_annot_default,thresh_motif_1)
		select_config.update({'filename_annot_motif_score':filename_annot_motif_score})

		file_path_motif_score = select_config['file_path_motif_score']
		file_save_path2 = file_path_motif_score
		
		if flag_cond_query_1>0:
			flag_motif_data_load = select_config['flag_motif_data_load']
			dict_motif_data = dict()
			if flag_motif_data_load>0:
				motif_data = self.motif_data
				motif_data_score = self.motif_data_score
				motif_query_name_expr = self.motif_query_name_expr
				
				dict_motif_data = {'motif_data':motif_data,'motif_data_score':motif_data_score,'motif_query_name_expr':motif_query_name_expr}
				motif_query_num1 = len(motif_query_name_expr)
				print('motif_data, motif_data_score, motif_query_name_expr: ',motif_data.shape,motif_data_score.shape,motif_query_num1)
				motif_query_vec = motif_query_name_expr
				
				if beta_mode>0:
					sel_num1 = 10
					motif_query_vec_1 = motif_query_vec
					motif_query_vec = motif_query_vec_1[0:sel_num1]
			else:
				motif_data, motif_data_score = [], []
				motif_query_vec = []

			rna_exprs = self.meta_scaled_exprs
			rna_exprs_unscaled = self.meta_exprs_2

			correlation_type = 'spearmanr'
			# flag_peak_tf_corr = 0
			flag_peak_tf_corr = 1
			flag_load_peak_tf = 1
			thresh_insilco_ChIP_seq = 0.1
			flag_save_text_peak_tf = 1
			select_config.update({'correlation_type':correlation_type,
									'flag_peak_tf_corr':flag_peak_tf_corr,
									'flag_load_peak_tf':flag_load_peak_tf,
									'flag_save_text_peak_tf':flag_save_text_peak_tf,
									'thresh_insilco_ChIP-seq':thresh_insilco_ChIP_seq})

			filename_annot_default = data_file_type_query
			select_config.update({'filename_annot_save_default':filename_annot_default})

			file_save_path_1 = select_config['data_path_save']
			file_save_path_local = select_config['data_path_save_local']
			
			# input_filename_1 = '%s/test_peak_tf_correlation.%s.spearmanr.1.txt'%(file_save_path_local,data_file_type_query)
			# select_config.update({'filename_peak_tf_corr':input_filename_1})
			# print('filename of peak accessibility-TF expression correlation: %s'%(input_filename_1))

			file_save_path_2 = select_config['file_path_motif_score']
			filename_annot = select_config['filename_annot_save_default']
			filename_1 = '%s/test_query_meta_nbrs.%s.1.txt'%(file_save_path_2,filename_annot)
			filename_2 = '%s/test_query_peak_access.%s.1.txt'%(file_save_path_2,filename_annot)
			select_config.update({'output_filename_open_peaks':filename_1,'output_filename_nbrs_atac':filename_2})

			print('output_filename_open_peaks ',filename_1)
			print('output_filename_nbrs_atac ',filename_2)

			if len(df_feature_link)==0:
				# input_filename_query = select_config['filename_feature_link_pre1'] # the file of the pre-selected peak-gene link query
				filename_feature_link_pre1 = select_config['filename_feature_link_pre1'] # path of the file of pre-selected peak-gene links
				input_filename_query = filename_feature_link_pre1

				if os.path.exists(input_filename_query)==True:
					df_link_query1 = pd.read_csv(input_filename_query,index_col=False,sep='\t')
					print('load peak-gene link query from %s'%(input_filename_query))
			else:
				df_link_query1 = df_feature_link
				
			print('pre-selected peak-gene links, dataframe of size ',df_link_query1.shape)
			print('columns: ',np.asarray(df_link_query1.columns))
			print('data preview:')
			print(df_link_query1[0:2])

			column_vec_1 = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_vec_1[0:3]

			flag_sort_1 = 1
			column_distance = 'distance'
			if (flag_sort_1>0) and (column_distance in df_link_query1.columns):
				df_link_query1 = df_link_query1.sort_values(by=[column_id1,column_distance],ascending=[True,True])

			gene_query_vec_1 = df_link_query1[column_id1].unique()
			gene_query_num1 = len(gene_query_vec_1)
			feature_query_num_1 = gene_query_num1
			print('gene number: %d'%(gene_query_num1))
			select_config.update({'feature_query_num_1':feature_query_num_1})

			column_gene_tf_corr_peak = ['gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1']
			select_config.update({'column_gene_tf_corr_peak':column_gene_tf_corr_peak,'column_idvec':column_vec_1})

			input_file_path_2 = select_config['file_path_motif_score']
			filename_prefix_cond = select_config['filename_prefix_cond']
			filename_prefix_default_1 = filename_prefix_cond
			print('filename_prefix_cond ',filename_prefix_cond)

			query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
			iter_mode=0
			if (query_id1>=0) and (query_id2>query_id1):
				iter_mode = 1
				# if query_id1>feature_query_num_1:
				# 	print('query_id1, query_id2, feature_query_num_1: ',query_id1,query_id2,feature_query_num_1)
				# 	return
				# else:
				# 	query_id2_ori = query_id2
				# 	query_id2 = np.min([query_id2,feature_query_num_1])

				query_id2_ori = query_id2
				# filename_prefix_save_pre2 = '%s.pcorr_query1.%d_%d'%(filename_prefix_default_1,query_id1,query_id2)
				filename_prefix_save_pre2 = '%s.pcorr_query1.%d_%d'%(filename_prefix_default_1,query_id1,query_id2_ori)
			else:
				# filename_prefix_save_pre2 = filename_prefix_default_1
				filename_prefix_save_pre2 = '%s.pcorr_query1'%(filename_prefix_default_1)

			# input_filename = '%s/%s.pcorr_query1.%d_%d.txt'%(input_file_path_2,filename_prefix_default_1,query_id1,query_id2)
			select_config.update({'filename_prefix_save_pre2':filename_prefix_save_pre2,
									'feature_query_num_1':feature_query_num_1,
									'iter_mode':iter_mode})

			input_filename = '%s/%s.txt'%(input_file_path_2,filename_prefix_save_pre2)
			df_gene_tf_corr_peak_1 = []
			
			if os.path.exists(input_filename)==True:
				df_gene_tf_corr_peak_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				# print('df_gene_tf_corr_peak_1: ',df_gene_tf_corr_peak_1.shape)
				print('initial peak-TF-gene links, dataframe of size ',df_gene_tf_corr_peak_1.shape)
				print('data loaded from %s'%(input_filename))
			else:
				print('the file does not exist:%s'%(input_filename))

			file_save_path2 = select_config['file_path_motif_score']
			input_file_path = file_save_path2

			filename_list1 = []
			query_vec = [1,3]
			extension_vec = ['txt.gz','txt']
			# for i2 in [1,3]:
			for (i2, extension) in zip(query_vec,extension_vec):
				input_filename_query = '%s/%s.annot1_%d.1.%s'%(input_file_path,filename_prefix_save_pre2,i2,extension)
				filename_list1.append(input_filename_query)
			select_config.update({'filename_gene_tf_peak_query_1':filename_list1})

			# input_filename_query_1 = '%s/%s.annot1_1.1.txt'%(input_file_path,filename_prefix_save_pre2)
			input_filename_query_1 = '%s/%s.annot1_1.1.txt.gz'%(input_file_path,filename_prefix_save_pre2)
			input_filename_query_2 = '%s/%s.annot2_1.1.txt'%(input_file_path,filename_prefix_save_pre2)
			filename_list2 = [input_filename_query_1,input_filename_query_2]
			select_config.update({'filename_gene_tf_peak_query_2':filename_list2})

			flag_query_1 = 1
			df_link_query2 = []
			df_feature_link_pre1, df_feature_link_pre2 = [], []
			if flag_query_1>0:
				# if iter_mode>0:
				# 	# query_id1, query_id2_ori = select_config['query_id1'], select_config['query_id2']
				# 	# filename_prefix_save_pre_2 = '%s.pcorr_query1.%d_%d'%(filename_prefix_default_1,query_id1,query_id2_ori)
				# 	filename_prefix_save_pre_2 = '%s.pcorr_query1.%d_%d'%(filename_prefix_default_1,query_id1,query_id2)
				# else:
				# 	# filename_prefix_save_pre_2 = filename_prefix_default_1
				# 	filename_prefix_save_pre_2 = '%s.pcorr_query1'%(filename_prefix_default_1)

				# input_filename = '%s/%s.annot2.init.1.txt'%(file_save_path2,filename_prefix_save_pre_2)
				input_filename = '%s/%s.annot2.init.1.1.txt'%(file_save_path2,filename_prefix_save_pre2)
				load_mode = 0
				if os.path.exists(input_filename)==True:
					print('the file exists: %s'%(input_filename))
					load_mode = 1

				# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
				column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
				# thresh_vec_1 = [[0.10,0.05],[0.05,0.10]]	# the threshold for score 1 and score 2
				thresh_vec_1 = [[0.10,0],[0,0.10]]	# the threshold for score 1 and score 2
				select_config.update({'thresh_score_query_1':thresh_vec_1})
			
				# parameter configuration for peak-TF-gene association score computation
				select_config = self.test_query_score_config_1(column_pval_cond=column_pval_cond,
																thresh_corr_1=0.1,
																thresh_pval_1=0.1,
																overwrite=True,
																flag_config_1=1,
																flag_config_2=1,
																save_mode=1,
																verbose=verbose,select_config=select_config)

				# flag_compute = 1
				df_feature_link = []
				if flag_compute in [1,3]:
					# estimate the peak-TF-gene link score
					if load_mode==0:
						flag_peak_tf_corr = 1
						flag_gene_tf_corr = 1
						flag_motif_score_normalize = 1
						flag_gene_tf_corr_peak_compute = 1
						field_query = ['flag_gene_tf_corr','flag_peak_tf_corr','flag_motif_score_normalize','flag_gene_tf_corr_peak_compute']
						list1 = [flag_gene_tf_corr,flag_peak_tf_corr,flag_motif_score_normalize,flag_gene_tf_corr_peak_compute]
						for (field_id,query_value) in zip(field_query,list1):
							select_config.update({field_id:query_value})

						# compute peak-TF-gene link scores
						filename_prefix_save = select_config['filename_prefix_default']
						data_path_save_local = select_config['data_path_save_local']
						file_path_motif_score = select_config['file_path_motif_score']
						input_file_path = data_path_save_local
						output_file_path = file_path_motif_score
						peak_distance_thresh = select_config['peak_distance_thresh']

						if len(atac_ad)==0:
							atac_ad = self.atac_meta_ad

						peak_read = self.peak_read
						interval_peak_corr = -1
						interval_local_peak_corr = 10
						df_link_query2 = self.test_query_feature_link_pre2(gene_query_vec=[],motif_query_vec=motif_query_vec,
																				df_gene_peak_query=df_link_query1,
																				peak_distance_thresh=peak_distance_thresh,
																				df_peak_query=[],
																				peak_loc_query=[],
																				df_gene_tf_corr_peak=df_gene_tf_corr_peak_1,
																				atac_ad=atac_ad,
																				peak_read=peak_read,
																				rna_exprs=rna_exprs,
																				rna_exprs_unscaled=rna_exprs_unscaled,
																				motif_data=motif_data,
																				motif_data_score=motif_data_score,
																				dict_motif_data=dict_motif_data,
																				interval_peak_corr=interval_peak_corr,
																				interval_local_peak_corr=interval_local_peak_corr,
																				flag_load_pre1=0,
																				flag_load_1=0,
																				input_file_path=input_file_path,
																				save_mode=1,
																				output_file_path=output_file_path,
																				output_filename='',
																				filename_prefix_save=filename_prefix_save,
																				verbose=verbose,select_config=select_config)
					else:
						df_link_query2 = pd.read_csv(input_filename,index_col=False,sep='\t')
						
					print('initial peak-TF-gene links with scores, dataframe of size ',df_link_query2.shape)
					print('columns ',np.asarray(df_link_query2.columns))
					print('data preview: ')
					print(df_link_query2[0:2])

					df_feature_link = df_link_query2

					column_score_query = select_config['column_score_query']
					column_score_1, column_score_2 = column_score_query[0:2]
					id1 = (column_score_1 in df_link_query2.columns)&(column_score_2 in df_link_query2.columns)

					if (load_mode==0) and (id1>0):
						output_filename_query = input_filename
						float_format = '%.6f'
						df_link_query2.to_csv(output_filename_query,sep='\t',float_format=float_format)
						print(output_filename_query)

					filename_feature_link_query1 = input_filename
					column_1 = 'filename_feature_link_query1'
					select_config.update({column_1:filename_feature_link_query1})

				if flag_compute in [2,3]:
					# select the peak-TF links
					file_save_path2 = select_config['file_path_motif_score']
					# filename_prefix_default_1 = select_config['filename_prefix_default_1']
					filename_prefix_default_1 = select_config['filename_prefix_default']
					input_file_path = file_save_path2
					output_file_path = file_save_path2

					flag_compare_thresh1 = 1
					flag_select_pair_1 = 1
					flag_select_feature_1 = 0
					flag_select_feature_2 = 0
					flag_select_local = 1
					flag_select_link_type = 0
					feature_query_num = feature_query_num_1
					
					column_1 = 'feature_score_interval'
					iter_mode = 0
					feature_score_interval = -1
					
					# query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
					if (query_id1>=0) and (query_id2>query_id1):
						feature_score_interval = query_id2 - query_id1
						iter_mode = 1
						select_config.update({column_1:feature_score_interval})
					else:
						if not (column_1 in select_config):
							select_config.update({column_1:feature_score_interval})
						else:
							feature_score_interval = select_config[column_1]
							if feature_score_interval>0:
								iter_mode = 1

					select_config.update({'feature_query_num':feature_query_num,'iter_mode':iter_mode})

					index_col = False
					# recompute = 0
					recompute = 1
					flag_score_quantile_1 = 0
					flag_score_query_1 = 1

					# thresh_vec_1 = [[0.10,0.05],[0.05,0.10]]
					thresh_vec_1 = [[0.10,0],[0,0.10]]
					select_config.update({'thresh_score_query_1':thresh_vec_1})

					column_query = 'filename_feature_link_query1'
					input_filename = ''
					if column_query in select_config:
						filename_feature_link_query1 = select_config[column_query]
						input_filename = filename_feature_link_query1

					# perform selection of peak-TF-gene links
					df_feature_link_pre1, df_feature_link_pre2 = self.test_query_feature_score_init_pre1(df_feature_link=df_feature_link,
																											input_filename_list=[],
																											input_filename=input_filename,
																											index_col=index_col,
																											iter_mode=iter_mode,
																											recompute=recompute,
																											flag_score_quantile_1=flag_score_quantile_1,
																											flag_score_query_1=flag_score_query_1,
																											flag_compare_thresh1=flag_compare_thresh1,
																											flag_select_pair_1=flag_select_pair_1,
																											flag_select_feature_1=flag_select_feature_1,
																											flag_select_feature_2=flag_select_feature_2,
																											flag_select_local=flag_select_local,
																											flag_select_link_type=flag_select_link_type,
																											input_file_path=input_file_path,
																											save_mode=1,output_file_path=output_file_path,
																											filename_prefix_save='',filename_save_annot='',
																											verbose=verbose,select_config=select_config)

				return df_link_query2, df_feature_link_pre1, df_feature_link_pre2

	## ====================================================
	# perform peak-TF-gene link selection
	def test_feature_link_query_select_pre1(self,thresh_vec_query=[],save_mode=1,save_mode_2=1,save_file_path='',verbose=0,select_config={}):

		"""
		# perform peak-TF-gene link selection
		:param thresh_vec_query: list containg two pairs of thresholds to select the peak-TF-gene links (each pair with threshods on score 1 and score 2 respectively)
		:param save_mode: indicator of whether to save data
		:param save_mode_2: indicator of whether to save the annotations of different types of selected or unselected peak-TF-gene links
		:param save_file_path: the directory to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing the annotations of different types of selected or unselected peak-TF-gene links
		"""

		flag_query1=1
		if flag_query1>0:
			iter_mode = select_config['iter_mode']
			file_save_path2 = select_config['file_path_motif_score']
			if iter_mode>0:
				filename_prefix_default_1 = select_config['filename_prefix_cond']
				query_id1, query_id2_ori = select_config['query_id1'], select_config['query_id2']
				filename_prefix_save_pre_2 = '%s.pcorr_query1.%d_%d'%(filename_prefix_default_1,query_id1,query_id2_ori)

				input_file_path = file_save_path2
				if not ('filename_combine' in select_config):
					# filename_prefix_save_2 = '%s.pcorr_query1.combine'%(filename_prefix_default_1)
					input_filename_query = '%s/%s.annot2.init.1.txt.gz'%(input_file_path,filename_prefix_save_2)
					# input_filename_query = '%s/%s.annot2.init.1.copy1.txt.gz'%(input_file_path,filename_prefix_save_2)

					if os.path.exists(input_filename_query)==True:
						select_config.update({'filename_combine':input_filename_query})
					else:
						print('please provide association estimation file')
						return
			else:
				filename_prefix_save_pre_2 = filename_prefix_default_1
			
			extension = 'txt.gz'
			# input_filename_2 = '%s/%s.annot2_1.1.txt'%(file_save_path2,filename_prefix_save_pre_2)
			input_filename_2 = '%s/%s.annot2_1.1.%s'%(file_save_path2,filename_prefix_save_pre_2,extension)
			select_config.update({'filename_link_type':input_filename_2})

			filename_annot_1 = '%s/%s.annot1_1.1.%s'%(file_save_path2,filename_prefix_save_pre_2,extension)
			select_config.update({'filename_annot_1':filename_annot_1})

			thresh_vec_1 = thresh_vec_query
			if len(thresh_vec_query)==0:
				column_1 = 'thresh_score_query_1'
				if column_1 in select_config:
					thresh_vec_1 = select_config[column_1]
				else:
					thresh_vec_1 = [[0.10,0.05],[0.05,0.10]]
					select_config.update({'thresh_score_query_1':thresh_vec_1})
			
			lambda1 = 0.5
			lambda2 = 0.5
			flag_score_quantile_1=1
			flag_score_query_1=1
			flag_compare_thresh1=1
			flag_select_pair_1=1
			flag_select_feature_1=0
			flag_select_feature_2=0
			flag_select_local=1
			flag_select_link_type=0

			# perform peak-TF-gene link selection
			# 1. df_link_query3: the updated annotations of peak-TF-gene links, including which links are selected by specific thresholds;
			# 2. df_link_pre1: selected peak-TF-gene links without using selection criteria on link types;
			# 3. df_link_pre2: selected peak-TF-gene links using selection criteria on link types and the other thresholds which select the links in (2);
			df_link_query3, df_link_pre1, df_link_pre2 = self.test_gene_peak_tf_query_select_1(df_gene_peak_query=df_link_query2,
																								df_annot_query=[],
																								df_score_annot=[],
																								lambda1=lambda1,
																								lambda2=lambda2,
																								type_id_1=0,
																								column_id1=-1,
																								flag_compare_thresh1=flag_compare_thresh1,
																								flag_select_pair_1=flag_select_pair_1,
																								flag_select_feature_1=flag_select_feature_1,
																								flag_select_feature_2=flag_select_feature_2,
																								flag_select_local=flag_select_local,
																								flag_select_link_type=flag_select_link_type,
																								iter_mode=iter_mode,
																								input_file_path='',
																								save_mode=1,
																								output_file_path='',
																								filename_prefix_save='',
																								verbose=verbose,select_config=select_config)

			print('the initial peak-TF-gene links with scores, dataframe of size ',df_link_query3.shape)
			print('columns: ',np.asarray(df_link_query3.columns))
			print('data preview:\n',df_link_query3[0:2])

			# file_save_path2 = select_config['file_path_motif_score']
			output_file_path = file_save_path2
			dict_query_1 = dict()
			if save_mode_2>0:
				field_query1 = ['peak_gene_link','gene_tf_link','peak_tf_link']
				t_columns = df_link_query3.columns.difference(field_query1,sort=False)

				field_query2 = ['label_gene_tf_corr_peak_compare']
				field_id1 = field_query2[0]

				df_link_query = df_link_query3.loc[:,t_columns]
				id1 = (df_link_query[field_id1]==1) # the link with difference between gene_tf_corr_peak and gene_tf_corr_ above threshold
					
				field_query3 = ['link_query']
				field_id2 = field_query3[0]

				if (field_id2 in df_link_query.columns):
					id2 = (df_link_query[field_id2]<0)
					# id_1 = (~id1)&(~id2)
					id_1 = (~id1)
					field_query_2 = ['group1','group2_1','group2_2']
					list_1 = [id_1,id1,id2]
				else:
					id_1 = (~id1)
					field_query_2 = ['group1','group2_1']
					list_1 = [id_1,id1]

				# dict_query_1 = dict()
				query_num1 = len(list_1)
				for i1 in range(query_num1):
					field_id = field_query_2[i1]
					id_query1 = list_1[i1]
					df_query1 = df_link_query.loc[id_query1,:]
					
					t_columns = df_query1.columns.difference(['query_id'],sort=False)
					df_query1 = df_query1.loc[:,t_columns]
					dict_query_1.update({field_id:df_query1})
					print('feature_link: ',df_query1.shape)

					output_filename_1 = '%s/%s.annot2.init.2_%d.txt'%(output_file_path,filename_prefix_save_pre_2,(i1+1))
					if ((save_mode_2==1) and (i1==0)) or (save_mode_2>1):
						df_query1.to_csv(output_filename_1,index=False,sep='\t')
					
			return dict_query_1

def run(run_id,chromsome,generate,chromvec,test_chromvec,species_id,featureid,celltype,file_path,path_id,
		flag_distance,data_file_type,data_file_type_id,input_dir,filename_gene_annot,filename_atac,filename_rna,filename_atac_meta,filename_rna_meta,
		filename_motif_data,filename_motif_data_score,file_mapping,file_peak,file_bg,metacell_num,peak_distance_thresh,highly_variable,gene_num_query,
		method_type_feature_link,output_dir,output_filename,
		beta_mode,recompute,interval_save,query_id1,query_id2,fold_id,n_iter_init,n_iter,
		flag_motif_ori,iter_mode_1,restart,config_id,feature_num_query,parallel,parallel_1,
		flag_motif_data_load,motif_data_thresh,motif_data_type,
		flag_correlation_query_1,flag_correlation_query,flag_correlation_1,flag_computation,flag_combine_empirical_1,flag_combine_empirical,
		flag_query_thresh2,overwrite_thresh2,flag_merge_1,flag_correlation_2,flag_correlation_query1,flag_discrete_query1,
		flag_peak_tf_corr,flag_gene_tf_corr,flag_gene_expr_corr,flag_compute_1,flag_score_pre1,feature_score_interval,flag_group_query,
		flag_feature_query1,flag_feature_query2,flag_feature_query3,
		flag_basic_query,flag_basic_query_2,type_query_compare,flag_basic_filter_1,flag_basic_filter_combine_1,flag_basic_filter_2,Lasso_alpha,
		peak_distance_thresh1,peak_distance_thresh2,flag_pred_1,flag_pred_2,flag_group_1,flag_pcorr_interval,flag_reduce,flag_combine_1,flag_combine_2,flag_cond_query_1):

	data_file_type = str(data_file_type)
	data_timepoint = data_file_type
	# run_id = 1
	run_id_load = -1
	run_id = int(run_id)
	type_id_feature = 0
	path_id = int(path_id)

	input_dir = str(input_dir)
	root_path_1 = input_dir
	root_path_2 = input_dir

	data_file_type_query = data_file_type
	data_file_type_id= int(data_file_type_id)

	metacell_num = int(metacell_num)
	peak_distance_thresh = int(peak_distance_thresh)
	highly_variable = int(highly_variable)

	filename_gene_annot = str(filename_gene_annot)
	filename_atac = str(filename_atac)
	filename_rna = str(filename_rna)
	filename_atac_meta = str(filename_atac_meta)
	filename_rna_meta = str(filename_rna_meta)
	filename_motif_data = str(filename_motif_data)
	filename_motif_data_score = str(filename_motif_data_score)
	file_mapping = str(file_mapping)
	file_peak = str(file_peak)
	file_bg = str(file_bg)
	method_type_feature_link = str(method_type_feature_link)
	output_dir = str(output_dir)
	output_filename = str(output_filename)

	gene_num_query = int(gene_num_query)
	query_id1, query_id2 = int(query_id1), int(query_id2)
	fold_id = int(fold_id)
	n_iter_init = int(n_iter_init)
	n_iter = int(n_iter)
	flag_motif_ori = int(flag_motif_ori)
	iter_mode_1 = int(iter_mode_1)
	restart = int(restart)
	config_id = int(config_id)
	feature_num_query = int(feature_num_query)
	flag_distance = int(flag_distance)
	# beta_mode = 0
	# beta_mode = 1
	beta_mode = int(beta_mode)
	recompute = int(recompute)
	parallel_mode = int(parallel)
	parallel_mode_peak = int(parallel_1)
	thresh_dispersions_norm = 0.5
	type_id_feature = 0
	metacell_num = 500
	select_config = dict()
	flag_motif_data_load = int(flag_motif_data_load)
	if flag_motif_ori>0:
		flag_motif_data_load = 1

	motif_data_thresh = int(motif_data_thresh)
	motif_data_type = int(motif_data_type)
	flag_correlation_query_1 = int(flag_correlation_query_1)
	flag_correlation_query = int(flag_correlation_query)
	flag_correlation_1 = int(flag_correlation_1)
	flag_computation_1 = str(flag_computation).split(',')
	flag_computation_vec = [int(flag_computation_query) for flag_computation_query in flag_computation_1]
	flag_combine_empirical_1 = int(flag_combine_empirical_1)
	flag_combine_empirical = int(flag_combine_empirical)
	flag_query_thresh2 = int(flag_query_thresh2)
	overwrite_thresh2 = (int(overwrite_thresh2)>0)
	flag_merge_1 = int(flag_merge_1)
	flag_correlation_2 = int(flag_correlation_2)
	flag_correlation_query1 = int(flag_correlation_query1)
	flag_discrete_query1 = int(flag_discrete_query1)
	flag_basic_query = int(flag_basic_query)
	flag_basic_query_2 = int(flag_basic_query_2)
	type_query_compare = int(type_query_compare)
	flag_basic_filter_1 = int(flag_basic_filter_1)
	flag_basic_filter_combine_1 = int(flag_basic_filter_combine_1)
	flag_basic_filter_2 = int(flag_basic_filter_2)
	Lasso_alpha = float(Lasso_alpha)
	peak_distance_thresh1 = str(peak_distance_thresh1)
	peak_distance_thresh_query1 = peak_distance_thresh1.split(',')
	peak_distance_thresh_query1 = [int(thresh_value) for thresh_value in peak_distance_thresh_query1]

	peak_distance_thresh2 = str(peak_distance_thresh2)
	peak_distance_thresh_query2 = peak_distance_thresh2.split(',')
	peak_distance_thresh_query2 = [int(thresh_value) for thresh_value in peak_distance_thresh_query2]

	flag_pred_1 = int(flag_pred_1)
	flag_pred_2 = int(flag_pred_2)
	flag_group_1 = int(flag_group_1)
	flag_combine_1 = int(flag_combine_1)
	flag_combine_2 = int(flag_combine_2)
	flag_cond_query_1 = int(flag_cond_query_1)
	flag_peak_tf_corr = int(flag_peak_tf_corr)
	flag_gene_tf_corr = int(flag_gene_tf_corr)
	flag_compute_1 = int(flag_compute_1)
	flag_score_pre1 = int(flag_score_pre1)
	feature_score_interval = int(feature_score_interval)
	flag_group_query = int(flag_group_query)
	flag_gene_expr_corr = int(flag_gene_expr_corr)
	flag_feature_query1 = int(flag_feature_query1)
	flag_feature_query2 = int(flag_feature_query2)
	flag_feature_query3 = int(flag_feature_query3)
	flag_pcorr_interval = int(flag_pcorr_interval)
	flag_reduce = int(flag_reduce)
	# flag_normalize_2 = 1
	flag_normalize_2 = 0

	select_config = {'data_file_type':data_file_type,
						'root_path_1':root_path_1,'root_path_2':root_path_2,'path_id':path_id,
						'input_dir':input_dir,
						'run_id':run_id,'run_id_load':run_id_load,
						'data_file_type_query':data_file_type_query,
						'data_file_type_id':data_file_type_id,
						'filename_gene_annot':filename_gene_annot,
						'filename_rna':filename_rna,
						'filename_atac':filename_atac,
						'filename_atac_meta':filename_atac_meta,
						'filename_rna_meta':filename_rna_meta,
						'filename_motif_data':filename_motif_data,
						'filename_motif_data_score':filename_motif_data_score,
						'filename_translation':file_mapping,
						'input_filename_peak':file_peak,
						'input_filename_bg':file_bg,
						'method_type_feature_link':method_type_feature_link,
						'output_dir':output_dir,
						'output_filename_link':output_filename,
						'metacell_num':metacell_num,
						'beta_mode':beta_mode,
						'gene_num_query':gene_num_query,
						'recompute':recompute,
						'type_id_feature':type_id_feature,
						'flag_motif_query_ori':flag_motif_ori,'iter_mode_1':iter_mode_1,'restart':restart,
						'query_id1':query_id1,'query_id2':query_id2,'fold_id':fold_id,'n_iter_init':n_iter_init,'n_iter':n_iter,'config_id':config_id,
						'feature_num_query':feature_num_query,
						'parallel_mode':parallel_mode,
						'parallel_mode_peak':parallel_mode_peak,
						'thresh_dispersions_norm':thresh_dispersions_norm}

	select_config.update({'flag_correlation_query_1':flag_correlation_query_1,
							'flag_correlation_query':flag_correlation_query,
							'flag_feature_query1':flag_feature_query1,
							'flag_feature_query2':flag_feature_query2,
							'flag_feature_query3':flag_feature_query3})

	select_config.update({'flag_motif_data_load':flag_motif_data_load,
							'motif_data_thresh':motif_data_thresh,
							'motif_data_type':motif_data_type,
							'flag_distance':flag_distance,
							'flag_correlation_1':flag_correlation_1,
							'flag_computation_vec':flag_computation_vec,
							'flag_combine_empirical_1':flag_combine_empirical_1,
							'flag_combine_empirical':flag_combine_empirical,
							'flag_query_thresh2':flag_query_thresh2,
							'overwrite_thresh2':overwrite_thresh2,
							'flag_merge_1':flag_merge_1,
							'flag_correlation_query1':flag_correlation_query1,
							'flag_correlation_2':flag_correlation_2,
							'flag_basic_query':flag_basic_query,
							'flag_basic_query_2':flag_basic_query_2,
							'type_query_compare':type_query_compare,
							'flag_discrete_query1':flag_discrete_query1,
							'flag_basic_filter_combine_1':flag_basic_filter_combine_1,
							'flag_basic_filter_1':flag_basic_filter_1,
							'flag_basic_filter_2':flag_basic_filter_2,
							'Lasso_alpha':Lasso_alpha,
							'peak_distance_thresh_query1':peak_distance_thresh_query1,
							'peak_distance_thresh_query2':peak_distance_thresh_query2,
							'flag_pred_1':flag_pred_1,
							'flag_pred_2':flag_pred_2,
							'flag_group_1':flag_group_1,
							'flag_combine_1':flag_combine_1,
							'flag_combine_2':flag_combine_2,
							'flag_peak_tf_corr':flag_peak_tf_corr,
							'flag_gene_tf_corr':flag_gene_tf_corr,
							'flag_gene_expr_corr':flag_gene_expr_corr,
							'flag_compute_1':flag_compute_1,
							'flag_score_pre1':flag_score_pre1,
							'flag_pcorr_interval':flag_pcorr_interval,
							'feature_score_interval':feature_score_interval,
							'flag_normalize_2':flag_normalize_2,
							'flag_reduce':flag_reduce,
							'flag_group_query':flag_group_query,
							'flag_cond_query_1':flag_cond_query_1})

	test_estimator = _Base2_pre_2(file_path=file_path,run_id=run_id,type_id_feature=type_id_feature,select_config=select_config)

	flag_correlation_query_1=0
	if 'flag_correlation_query_1' in select_config:
		flag_correlation_query_1 = select_config['flag_correlation_query_1']
	
	if flag_correlation_query_1>0:
		flag_query = 1
		if flag_query >0:
			peak_distance_thresh = 2000
			interval_peak_corr = 100
			interval_local_peak_corr = -1
			verbose = 1
			test_estimator.test_query_feature_link_pre1(gene_query_vec=[],peak_distance_thresh=peak_distance_thresh,
														df_peak_query=[],peak_loc_query=[],atac_ad=[],rna_exprs=[],highly_variable=False,
														interval_peak_corr=interval_peak_corr,interval_local_peak_corr=interval_local_peak_corr,beta_mode=beta_mode,
														save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',verbose=verbose,select_config=select_config)

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="1", help="experiment id")
	parser.add_option("-f","--chromosome", default="1", help="Chromosome name")
	parser.add_option("-g","--generate", default="0", help="whether to generate feature vector: 1: generate; 0: not generate")
	parser.add_option("-c","--chromvec",default="1",help="chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-t","--testchromvec",default="10",help="test chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-i","--species",default="0",help="species id")
	parser.add_option("-j","--featureid",default="0",help="feature idx")
	parser.add_option("-b","--cell",default="1",help="cell type")
	parser.add_option("--file_path",default="1",help="file_path")
	parser.add_option("--path1",default="1",help="file_path_id")
	parser.add_option("--flag_distance",default="1",help="flag_distance")
	parser.add_option("--data_file_type",default="pbmc",help="the cell type or dataset annotation")
	parser.add_option("--data_file_query",default="0",help="data_file_type_id")
	parser.add_option("--input_dir",default=".",help="the directory where the ATAC-seq and RNA-seq data of the metacells are saved")
	parser.add_option("--gene_annot",default="-1",help="file path of gene position annotation file")
	parser.add_option("--atac_data",default="-1",help="file path of ATAC-seq data of the single cells")
	parser.add_option("--rna_data",default="-1",help="file path of RNA-seq data of the single cells")
	parser.add_option("--atac_meta",default="-1",help="file path of ATAC-seq data of the single cells")
	parser.add_option("--rna_meta",default="-1",help="file path of RNA-seq data of the metacells")
	parser.add_option("--gene_annot",default="-1",help="file path of gene position annotation file")
	parser.add_option("--motif_data",default="-1",help="file path of binary motif scannning results")
	parser.add_option("--motif_data_score",default="-1",help="file path of the motif scores by motif scanning")
	parser.add_option("--file_mapping",default="-1",help="file path of the mapping between TF motif identifier and the TF name")
	parser.add_option("--file_peak",default="-1",help="file containing the ATAC-seq peak loci annotations")
	parser.add_option("--file_bg",default="-1",help="file containing the estimated background peak loci")
	parser.add_option("--metacell",default="500",help="metacell number")
	parser.add_option("--peak_distance",default="500",help="peak distance threshold")
	parser.add_option("--highly_variable",default="1",help="highly variable gene")
	parser.add_option("--gene_num_query",default="3000",help="selected highly variable gene number")
	parser.add_option("--method_type_feature_link",default="Unify",help='method_type_feature_link')
	parser.add_option("--output_dir",default='output_file',help='the directory to save the output')
	parser.add_option("--output_filename",default='-1',help='filename of the predicted regulatory assocations')
	parser.add_option("--beta_mode",default="0",help="beta_mode")
	parser.add_option("--recompute",default="0",help="recompute")
	parser.add_option("--interval_save",default="-1",help="interval_save")
	parser.add_option("--q_id1",default="-1",help="query_id1")
	parser.add_option("--q_id2",default="-1",help="query_id2")
	parser.add_option("--fold_id",default="0",help="fold_id")
	parser.add_option("--n_iter_init",default="-1",help="initial estimation iteration number")
	parser.add_option("--n_iter",default="15",help="iteration number")
	parser.add_option("--flag_motif_ori",default="0",help="original motif number")
	parser.add_option("--iter_mode_1",default="0",help="iteration mode")
	parser.add_option("--restart",default="1",help="restart iteration or continue with iteration")
	parser.add_option("--config_id",default="-1",help="config_id")
	parser.add_option("--feature_num",default="200",help="feature query number")
	parser.add_option("--parallel",default="0",help="parallel_mode")
	parser.add_option("--parallel_1",default="0",help="parallel_mode_peak")
	parser.add_option("--flag_motif_data_load",default="0",help="flag_motif_data_load")
	parser.add_option("--motif_data_thresh",default="0",help="threshold for motif scanning")
	parser.add_option("--motif_data_type",default="0",help="motif data type")
	parser.add_option("--flag_correlation_query_1",default="1",help="flag_correlation_query_1")
	parser.add_option("--flag_correlation_query",default="1",help="flag_correlation_query")
	parser.add_option("--flag_correlation_1",default="1",help="flag_correlation_1")
	parser.add_option("--flag_computation",default="1",help="flag_computation_vec")
	parser.add_option("--flag_combine_empirical_1",default="0",help="flag_combine_empirical_1")
	parser.add_option("--flag_combine_empirical",default="0",help="flag_combine_empirical")
	parser.add_option("--flag_query_thresh2",default="0",help="flag_query_thresh2")
	parser.add_option("--flag_merge_1",default="0",help="flag_merge_1")
	parser.add_option("--overwrite_thresh2",default="0",help="overwrite thresh2 file")
	parser.add_option("--flag_correlation_2",default="0",help="flag_correlation_2")
	parser.add_option("--flag_correlation_query1",default="1",help="flag_correlation_query1")
	parser.add_option("--flag_discrete_query1",default='1',help='flag_discrete_query1')
	parser.add_option("--flag_peak_tf_corr",default="0",help="flag_peak_tf_corr")
	parser.add_option("--flag_gene_tf_corr",default="0",help="flag_gene_tf_corr")
	parser.add_option("--flag_gene_expr_corr",default="0",help="flag_gene_expr_corr")
	parser.add_option("--flag_compute_1",default="1",help="initial score computation and selection")
	parser.add_option("--flag_score_pre1",default="2",help="initial score computation")
	parser.add_option("--feature_score_interval",default="-1",help="feature_score_interval")
	parser.add_option("--flag_group_query",default="0",help="flag_group_query")
	parser.add_option("--flag_feature_query1",default="0",help="differential feature query")
	parser.add_option("--flag_feature_query2",default="0",help="differential feature query 2")
	parser.add_option("--flag_feature_query3",default="0",help="differential feature query 3")
	parser.add_option("--flag_basic_query",default="0",help="flag_basic_query")
	parser.add_option("--flag_basic_query_2",default="0",help="flag_basic_query_2")
	parser.add_option("--type_query_compare",default="2",help="type_query_compare")
	parser.add_option("--flag_basic_filter_1",default="0",help="flag_basic_filter_1")
	parser.add_option("--flag_basic_filter_combine_1",default="0",help="flag_basic_filter_combine_1")
	parser.add_option("--flag_basic_filter_2",default="0",help="flag_basic_filter_2")
	parser.add_option("--Lasso_alpha",default="0.001",help="Lasso_alpha")
	parser.add_option("--peak_distance_thresh1",default="500",help="peak distance threshold 1")
	parser.add_option("--peak_distance_thresh2",default="500",help="peak distance threshold 2")
	parser.add_option("--flag_pred_1",default="0",help="flag_pred_1")
	parser.add_option("--flag_pred_2",default="0",help="flag_pred_2")
	parser.add_option("--flag_group_1",default="0",help="flag_group_1")
	parser.add_option("--flag_pcorr_interval",default="0",help="flag_pcorr_interval")
	parser.add_option("--flag_reduce",default="1",help="reduce intermediate files")
	parser.add_option("--flag_combine_1",default="0",help="basic_filter_combine_1")
	parser.add_option("--flag_combine_2",default="0",help="basic_filter_combine_2")
	parser.add_option("--flag_cond_query_1",default="0",help="flag_cond_query_1")
	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	run(opts.run_id,
		opts.chromosome,
		opts.generate,
		opts.chromvec,
		opts.testchromvec,
		opts.species,
		opts.featureid,
		opts.cell,
		opts.file_path,
		opts.path1,
		opts.flag_distance,
		opts.data_file_type,
		opts.data_file_query,
		opts.input_dir,
		opts.gene_annot,
		opts.atac_data,
		opts.rna_data,
		opts.atac_meta,
		opts.rna_meta,
		opts.motif_data,
		opts.motif_data_score,
		opts.file_mapping,
		opts.file_peak,
		opts.file_bg,
		opts.metacell,
		opts.peak_distance,
		opts.highly_variable,
		opts.gene_num_query,
		opts.method_type_feature_link,
		opts.output_dir,
		opts.output_filename,
		opts.beta_mode,
		opts.recompute,
		opts.interval_save,
		opts.q_id1,
		opts.q_id2,
		opts.fold_id,
		opts.n_iter_init,
		opts.n_iter,
		opts.flag_motif_ori,
		opts.iter_mode_1,
		opts.restart,
		opts.config_id,
		opts.feature_num,
		opts.parallel,
		opts.parallel_1,
		opts.flag_motif_data_load,
		opts.motif_data_thresh,
		opts.motif_data_type,
		opts.flag_correlation_query_1,
		opts.flag_correlation_query,
		opts.flag_correlation_1,
		opts.flag_computation,
		opts.flag_combine_empirical_1,
		opts.flag_combine_empirical,
		opts.flag_query_thresh2,
		opts.overwrite_thresh2,
		opts.flag_merge_1,
		opts.flag_correlation_2,
		opts.flag_correlation_query1,
		opts.flag_discrete_query1,
		opts.flag_peak_tf_corr,
		opts.flag_gene_tf_corr,
		opts.flag_gene_expr_corr,
		opts.flag_compute_1,
		opts.flag_score_pre1,
		opts.feature_score_interval,
		opts.flag_group_query,
		opts.flag_feature_query1,
		opts.flag_feature_query2,
		opts.flag_feature_query3,
		opts.flag_basic_query,
		opts.flag_basic_query_2,
		opts.type_query_compare,
		opts.flag_basic_filter_1,
		opts.flag_basic_filter_combine_1,
		opts.flag_basic_filter_2,
		opts.Lasso_alpha,
		opts.peak_distance_thresh1,
		opts.peak_distance_thresh2,
		opts.flag_pred_1,
		opts.flag_pred_2,
		opts.flag_group_1,
		opts.flag_pcorr_interval,
		opts.flag_reduce,
		opts.flag_combine_1,
		opts.flag_combine_2,
		opts.flag_cond_query_1)



