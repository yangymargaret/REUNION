import pandas as pd
import numpy as np
import scipy
import scipy.io
import sklearn
import math
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

import os
import os.path
from optparse import OptionParser
# import test_pre2_3 as test_pre1
# import test_annotation_8 as test_pre1
# from test_annotation_8 import _Base2_1
# from test_annotation_8_copy1 import _Base2_1
# from test_annotation_9_2_copy1_1 import _Base2_2
from test_annotation_9_2_copy1_2 import _Base2_2_1
# from test_annotation_6 import _Base2_correlation5
from test_annotation_2 import _Base2_correlation
from test_group_1 import _Base2_group1
# from test_annotation_11_3_2 import _Base2_train5_2_pre1
import test_annotation_11_3_2
import test_annotation_3_1
import test_annotation_6
# import test_group_7_2
# import test_group_7_2_pre2
import train_pre1_1

from scipy import stats
from scipy.stats import multivariate_normal, skew, pearsonr, spearmanr
from scipy.stats import wilcoxon, mannwhitneyu, kstest, ks_2samp, chisquare, fisher_exact
from scipy.stats import barnard_exact, boschloo_exact
from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq
from scipy.stats import gaussian_kde, zscore
from scipy.stats import poisson, multinomial
from scipy.stats import norm
from scipy.stats import hypergeom
# from scipy.stats import fisher_exact
from scipy.cluster.hierarchy import dendrogram, linkage

import scipy.sparse
from scipy.sparse import spmatrix
from scipy.sparse import hstack, csr_matrix, issparse, vstack
from scipy import signal
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences
import networkx as nx

from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,accuracy_score,matthews_corrcoef
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics import accuracy_score, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
# from processSeq import load_seq_1, kmer_dict, load_signal_1, load_seq_2, load_seq_2_kmer, load_seq_altfeature_1
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import time
from timeit import default_timer as timer

import gc
from joblib import Parallel, delayed
import multiprocessing as mp
import threading
from tqdm.notebook import tqdm
import utility_1
from utility_1 import test_query_index
import h5py
import json
import pickle

# def test_query_index(df_query,column_vec,symbol_vec=['.','.']):

# 	if len(symbol_vec)==0:
# 		symbol_vec = ['.','.']
# 	if len(column_vec)>2:
# 		column_id1,column_id2,column_id3 = column_vec[0:3]
# 		symbol1, symbol2 = symbol_vec
# 		query_id_1 = ['%s%s%s%s%s'%(query_id1,symbol1,query_id2,symbol2,query_id3) for (query_id1,query_id2,query_id3) in zip(df_query[column_id1],df_query[column_id2],df_query[column_id3])]
# 	else:
# 		column_id1,column_id2 = column_vec[0:2]
# 		symbol1 = symbol_vec[0]
# 		query_id_1 = ['%s%s%s'%(query_id1,symbol1,query_id2) for (query_id1,query_id2) in zip(df_query[column_id1],df_query[column_id2])]

# 	return query_id_1

class _Base2_2_pre1(_Base2_2_1):
	"""Base class for Hidden Markov Models.
	"""
	# def __init__(self, n_components=1, run_id=0,
	#            startprob_prior=1.0, transmat_prior=1.0,
	#            algorithm="viterbi", random_state=None,
	#            n_iter=10, tol=1e-2, verbose=False,
	#            params=string.ascii_letters,
	#            init_params=string.ascii_letters):
	#   self.n_components = n_components
	#   self.params = params
	#   self.init_params = init_params
	#   self.startprob_prior = startprob_prior
	#   self.transmat_prior = transmat_prior
	#   self.algorithm = algorithm
	#   self.random_state = random_state
	#   self.n_iter = n_iter
	#   self.tol = tol
	#   self.verbose = verbose
	#   self.run_id = run_id

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

		_Base2_2_1.__init__(self,file_path=file_path,
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
	# query the basic file path
	def test_file_path_query_1_ori(self,run_id=1,select_config={}):

			# input_file_path1 = self.save_path_1
			data_file_type = select_config['data_file_type']
			# root_path_1: '../data2'
			root_path_1 = select_config['root_path_1']
			input_file_path1 = root_path_1
			data_file_type_id1 = 0
			# run_id = select_config['run_id']
			type_id_feature = select_config['type_id_feature']

			if data_file_type=='CD34_bonemarrow':
				# input_file_path = '%s/data_pre2/cd34_bonemarrow'%(input_file_path1)
				input_file_path = '%s/data_pre2/cd34_bonemarrow/data_1'%(input_file_path1)
				# filename_save_annot_1 = data_file_type
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)

			elif data_file_type=='pbmc':
				# input_file_path = '%s/data_pre2/10x_pbmc/data_1'%(input_file_path1)
				path_id = select_config['path_id']
				if path_id==1:
					input_file_path = '%s/data_pre2/10x_pbmc/data_1_vbak1'%(input_file_path1)
				else:
					input_file_path = '%s/data_pre2/10x_pbmc/data_1/data1_vbak1'%(input_file_path1)
				type_id_feature = select_config['type_id_feature']
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)

			elif data_file_type in ['mouse_endoderm']:
				input_file_path = '%s/data_pre2/mouse_endoderm'%(input_file_path1)
				# input_file_path_2 = input_file_path1
				# root_path_2 = input_file_path_2
				filename_save_annot_1 = 'E8.75#Multiome'
				data_file_type_id1 = 1
				# select_config.update({'data_path_2':input_file_path_2})

			# input_file_path = '%s/data_pre2/cd34_bonemarrow_2'%(input_file_path1)
			# select_config.update({'data_path':input_file_path,
			# 						'filename_save_annot_1':filename_save_annot_1,
			# 						'filename_save_annot_pre1':filename_save_annot_1})

			# select_config.update({'data_path_2':input_file_path_2})
			# select_config.update({'root_path_2':root_path_2})
			# select_config.update({'filename_save_annot_pre1':filename_save_annot_1})
			select_config_query = {'data_path':input_file_path,
									'filename_save_annot_1':filename_save_annot_1,
									'filename_save_annot_pre1':filename_save_annot_1}

			return select_config_query

	## file_path query
	# query the peak-TF motif scanning matrix of the methods
	def test_file_path_query_2(self,method_type_vec=[],select_config={}):

			# input_file_path1 = self.save_path_1
			data_file_type = select_config['data_file_type']
			# root_path_1: '../data2'
			# root_path_2: '../data1'
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
			input_file_path1 = root_path_1
			# data_file_type_id1 = 0
			# run_id = select_config['run_id']
			# type_id_feature = select_config['type_id_feature']
			# data_path = select_config['data_path']
			# input_file_path = select_config['data_path']
			# method_type_vec = select_config['method_type_vec_query']
			method_type_num = len(method_type_vec)
			data_file_type = select_config['data_file_type']
			data_file_type_annot = data_file_type.lower()
			metacell_num = select_config['metacell_num']

			dict_query1 = dict()	# the motif data file path of the method query
			dict_query2 = dict()	# the file path of the method query
			
			# method_type_vec = ['Pando','GRaNIE']
			for method_type_id in range(method_type_num):
				method_type = method_type_vec[method_type_id]
				# if method_type in ['Pando','GRaNIE']:
				# 	file_path1 = '%s/%s/%s'%(root_path_1,method_type,data_file_type_annot)
				# file_path1 = '%s/%s/%s'%(root_path_2,method_type,data_file_type_annot)
				# file_path1 = '%s/%s'%(root_path_2,method_type)
				filename_motif = ''
				filename_motif_score = ''
				pre_config = select_config['config_query'][method_type]
				metacell_num_query,run_id_query = pre_config['metacell_num'], pre_config['run_id']
				file_path1 = '%s/%s'%(root_path_2,method_type)
				file_path2 = '%s/%s/metacell_%d/run%d'%(file_path1,data_file_type_annot,metacell_num_query,run_id_query)
				
				if method_type in ['Pando']:
				# ~/Downloads/doc5/data1/GRaNIE/cd34_bonemarrow/metacell_500/run1
					# peak_distance_thresh = [100,0]
					# file_path1 = '%s/%s/%s'%(root_path_1,method_type,data_file_type_annot)
					# file_path = '%s/%s/%s/metacell_%d/run%d'%(root_path_1,method_type,data_file_type_annot,metacell_num,run_id)
					# file_path1 = '%s/%s/%s/metacell_%d/run%d'%(root_path_2,method_type,data_file_type_annot,metacell_num_query,run_id_query)
					filename_motif = '%s/test_peak_tf_overlap.tsv.gz'%(file_path2)
					# filename_motif = '%s/test_peak_tf_overlap.matrix.txt'%(file_path2)
				
				elif method_type in ['TRIPOD']:
					pre_config = select_config['config_query'][method_type]
					type_id_query = pre_config['type_id_query']
					filename_motif = '%s/data1_1_%d/test_peak_tf_overlap.tsv.gz'%(file_path2,type_id_query)
					if (os.path.exists(filename_motif)==False):
						print('the file does not exist ',filename_motif)
						metacell_num_load1=100
						run_id_load1=111
						file_path_motif_save = '%s/%s/metacell_%d/run%d'%(file_path1,data_file_type_annot,metacell_num_load1,run_id_load1)
						filename_motif = '%s/data1_1_%d/test_peak_tf_overlap.tsv.gz'%(file_path_motif_save,type_id_query)
				
				elif method_type in ['GRaNIE']:
					# file_path1 = '%s/%s'%(root_path_2,method_type)
					# file_path1 = '%s/%s/%s/metacell_%d/run%d'%(root_path_2,method_type,data_file_type_annot,metacell_num_query,run_id_query)
					filename_motif='%s/df_peak_tf_overlap.pearsonr.%s.normalize0.tsv.gz'%(file_path2,data_file_type)
				
				else:
					run_id1 = 1
					metacell_num1 = 500
					select_config_query = self.test_file_path_query_1_ori(run_id=run_id1,select_config=select_config)
					input_file_path = select_config_query['data_path']
					# file_path1 = '%s/peak_local'%(input_file_path)
					if data_file_type in ['CD34_bonemarrow']:
						run_id_1 = 0
						file_path1 = '%s/run%d'%(input_file_path,run_id_1)
					elif data_file_type in ['pbmc']:
						file_path1 = input_file_path

				dict1 = {'motif_data':filename_motif,'motif_data_score':filename_motif_score}
				dict_query1.update({method_type:dict1})
				dict_query2.update({method_type:file_path1})

			select_config['filename_motif_data'] = dict_query1
			select_config['input_file_path_query'] = dict_query2

			return select_config

	## the configuration of the different methods
	def test_config_query_1(self,method_type_vec=[],select_config={}):

		# # input_file_path1 = self.save_path_1
		# data_file_type = select_config['data_file_type']
		# # root_path_1: '../data2'
		# # root_path_2: '../data1'
		# root_path_1 = select_config['root_path_1']
		# root_path_2 = select_config['root_path_2']

		method_type_num = len(method_type_vec)
		dict_feature = dict()
		data_file_type = select_config['data_file_type']
		# dict_method_type = select_config['dict_method_type']
		# dict_feature_pre1 = dict()
		# dict_feature = dict()
		type_id_feature = select_config['type_id_feature']
		metacell_num = select_config['metacell_num']
		run_id = select_config['run_id']
		config_query = dict()
		for i1 in range(method_type_num):
			# pre_config = dict()
			pre_config = {'data_file_type':data_file_type,
							'type_id_feature':type_id_feature,
							'metacell_num':metacell_num,
							'run_id':run_id}
			# method_type_id = method_type_vec[i1]
			# method_type = dict_method_type[method_type_id]
			method_type = method_type_vec[i1]
			if method_type in ['Pando']:
				run_id_query = 1
				metacell_num_query = 500
				upstream, downstream = 100, 0
				exclude_exons = 1
				type_id_region = 1
				method = 'glm'
				padj_thresh = 0.05
				# padj_thresh = 0.1
				# padj_thresh = 0.2
				# padj_thresh = 0.25
				pre_config.update({'type_id_region':type_id_region,
									'exclude_exons':exclude_exons,
									'upstream':upstream,'downstream':downstream,'method':method,
									'padj_thresh':padj_thresh,
									'run_id':run_id_query,
									'metacell_num':metacell_num_query})

			elif method_type in ['TRIPOD']:
				if data_file_type in ['pbmc']:
					run_id_save = select_config['run_id_save']
					if run_id_save==20:
						run_id_query = 111
						metacell_num_query = 100
					else:
						run_id_query = 1
						metacell_num_query = 500
				else:
					run_id_query = 0
					metacell_num_query = 500

				# upstream, downstream = 100, 100
				upstream, downstream = 200, 200
				thresh_fdr = 0.01
				normalize_type = 1
				# type_id_query = 0
				# type_id_query = 1
				upstream, downstream, type_id_query = select_config['upstream_tripod'], select_config['downstream_tripod'], select_config['type_id_tripod']
				print('TRIPOD upstream:%d, downstream:%d, type_id_query:%d'%(upstream,downstream,type_id_query))
				pre_config.update({'normalize_type':normalize_type,
									'upstream':upstream,'downstream':downstream,
									'thresh_fdr':thresh_fdr,
									'run_id':run_id_query,
									'metacell_num':metacell_num_query,
									'type_id_query':type_id_query})

			elif method_type in ['GRaNIE']:
				# run_id = 111
				# metacell_num = select_config['metacell_num']
				if data_file_type in ['pbmc']:
					run_id_query = 111
					metacell_num_query = 100
				else:
					run_id_query = 1
					metacell_num_query = 500

				peak_distance_thresh = 250
				correlation_type_vec = ['pearsonr','spearmanr']
				correlation_typeid = 0
				correlation_type = correlation_type_vec[correlation_typeid]
				thresh_fdr_save = 0.3
				# thresh_fdr_peak_tf = 0.3
				thresh_fdr_peak_tf = 0.2
				thresh_fdr_peak_gene = 0.2
				if 'thresh_fdr_peak_tf_GRaNIE' in select_config:
					thresh_fdr_peak_tf = select_config['thresh_fdr_peak_tf_GRaNIE']
				# thresh_fdr_peak_tf = 0.25
				print('thresh_fdr_peak_tf: %.2f'%(thresh_fdr_peak_tf))
				normalize_type = 0
				pre_config.update({'peak_distance_thresh':peak_distance_thresh,
									'correlation_type':correlation_type,
									'thresh_fdr_save':thresh_fdr_save,
									'thresh_fdr_peak_tf':thresh_fdr_peak_tf,
									'thresh_fdr_peak_gene':thresh_fdr_peak_gene,
									'normalize_type':normalize_type,
									'run_id':run_id_query,
									'metacell_num':metacell_num_query
									})
			else:
				run_id_query = 1
				metacell_num_query = 500
				pre_config.update({'run_id':run_id_query,'metacell_num':metacell_num_query})

			config_query.update({method_type:pre_config})
		select_config['config_query'] = config_query

		return select_config

	## the configuration of the methods
	# query the file path of the metacell data
	def test_config_query_2(self,method_type_vec=[],data_file_type='',select_config={}):

		if data_file_type=='':
			data_file_type = select_config['data_file_type']

		data_file_type_query = data_file_type
		root_path_1 = select_config['root_path_1']
		# run_id_1 = 0

		path_id = select_config['path_id']
		if data_file_type_query in ['CD34_bonemarrow']:
			data_file_type_annot = data_file_type_query.lower()
			run_id_1 = 0
			# input_file_path2 = '%s/data_pre2/%s/peak_local'%(root_path_1,data_file_type_annot)
			input_file_path_query1 = '%s/data_pre2/%s/data_1/run%d/peak_local'%(root_path_1,data_file_type_annot,run_id_1)
			input_file_path_query = '%s/seacell_1'%(input_file_path_query1)
			file_path_motif_score = '%s/motif_score_thresh1'%(input_file_path_query)
			data_path_save_1 = input_file_path_query1 
			select_config.update({'data_path_save_local':input_file_path_query,
									'file_path_motif_score_2':file_path_motif_score,
									'file_path_motif_score':file_path_motif_score,
									'data_file_type_annot':data_file_type_annot,
									'data_path_save_1':data_path_save_1})

			input_file_path2 = '%s/data_pre2/%s/peak_local'%(root_path_1,data_file_type_annot)
			filename_1 = '%s/test_rna_meta_ad.%s.log_normalize.2.h5ad'%(input_file_path_query,data_file_type_query)
			filename_2 = '%s/test_atac_meta_ad.%s.log_normalize.2.h5ad'%(input_file_path_query,data_file_type_query)
			filename_3_ori = '%s/test_query.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)

		elif data_file_type_query in ['pbmc']:
			data_file_type_annot = '10x_pbmc'
			if path_id==1:
				input_file_path_query1 = '%s/data_pre2/%s/data_1_vbak1/peak_local'%(root_path_1,data_file_type_annot)
			else:
				input_file_path_query1 = '%s/data_pre2/%s/data_1/data1_vbak1/peak_local'%(root_path_1,data_file_type_annot)

			input_file_path_query  = '%s/seacell_1'%(input_file_path_query1)
			file_path_motif_score = '%s/motif_score_thresh1'%(input_file_path_query)
			data_path_save_1 = input_file_path_query1
			select_config.update({'data_path_save_local':input_file_path_query,
										'file_path_motif_score_2':file_path_motif_score,
										'file_path_motif_score':file_path_motif_score,
										'data_file_type_annot':data_file_type_annot,
										'data_path_save_1':data_path_save_1})

			# input_file_path2 = '%s/data_pre2/%s/peak_local'%(root_path_1,data_file_type_annot)

			type_id_feature = 0
			run_id1 = 1
			filename_save_annot = '%s.%d.%d'%(data_file_type_query,type_id_feature,run_id1)
			filename_1 = '%s/test_rna_meta_ad.%s.h5ad'%(input_file_path_query,filename_save_annot)
			filename_2 = '%s/test_atac_meta_ad.%s.h5ad'%(input_file_path_query,filename_save_annot)
			# filename_3_ori = '%s/test_query.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
			# filename_3_ori = '%s/test_rna_meta_ad.pbmc.0.1.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
			filename_3_ori = '%s/test_rna_meta_ad.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,filename_save_annot)

		select_config.update({'filename_rna':filename_1,'filename_atac':filename_2,
								'filename_rna_exprs_1':filename_3_ori})
		select_config.update({'data_file_type_annot':data_file_type_annot})

		return select_config

	## file_path and configuration query
	def test_query_config_pre1_1(self,data_file_type_query='',method_type_vec=[],flag_config_1=1,select_config={}):

		if flag_config_1>0:
			if data_file_type_query=='':
				data_file_type_query = select_config['data_file_type']

			if len(method_type_vec)==0:
				method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre2.thresh3','joint_score_pre1.thresh22','joint_score_pre2.thresh6']

			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
			path_id = select_config['path_id']
			if path_id==1:
				input_file_path_query = '%s/data_pre2/data1_2'%(root_path_1)
			else:
				input_file_path_query = root_path_2

			if data_file_type_query in ['CD34_bonemarrow']:
				input_file_path = '%s/peak1'%(input_file_path_query)
			elif data_file_type_query in ['pbmc']:
				input_file_path = '%s/peak2'%(input_file_path_query)

			file_save_path_1 = input_file_path
			select_config.update({'file_path_peak_tf':file_save_path_1})
			# peak_distance_thresh = 100
			peak_distance_thresh = 500
			filename_prefix_1 = 'test_motif_query_binding_compare'
			method_type_vec_query = method_type_vec

			# input_file_path_query = '/data/peer/yangy4/data1/data_pre2/cd34_bonemarrow/data_1/run0/'
			# root_path_1 = select_config['root_path_1']
			# data_file_type_annot = data_file_type_query.lower()
			# run_id_1 = 0
			# input_file_path2 = '%s/data_pre2/%s/peak_local'%(root_path_1,data_file_type_annot)
			# input_file_path_query1 = '%s/data_pre2/%s/data_1/run%d/peak_local'%(root_path_1,data_file_type_annot,run_id_1)
			# input_file_path_query = '%s/seacell_1'%(input_file_path_query1)

			select_config = self.test_config_query_2(method_type_vec=method_type_vec,data_file_type=data_file_type_query,select_config=select_config)
			data_file_type_annot = select_config['data_file_type_annot']
			data_path_save_local = select_config['data_path_save_local']
			input_file_path_query = data_path_save_local

			## query the configurations of the methods
			# thresh_fdr_peak_tf_GRaNIE = 0.2
			# select_config.update({'thresh_fdr_peak_tf_GRaNIE':thresh_fdr_peak_tf_GRaNIE})
			select_config = self.test_config_query_1(method_type_vec=method_type_vec,select_config=select_config)

			## query the file path of the peak-TF motif scanning matrix of the different methods
			select_config = self.test_file_path_query_2(method_type_vec=method_type_vec,select_config=select_config)

			# query the file path for TF binding prediction
			file_save_path_1 = select_config['file_path_peak_tf']
			# folder_id = select_config['folder_id']
			folder_idvec_1 = [0,1,2] # folder_id
			config_idvec_2 = [20,10,12] # config_id_2
			dict_query1 = dict(zip(folder_idvec_1,config_idvec_2))

			dict_file_annot1 = dict()
			dict_file_annot2 = dict()
			if data_file_type_query in ['pbmc']:
				for folder_id_query in [0,1,2]:
					# folder_id_query = 2
					group_id_1 = folder_id_query+1
					if folder_id_query in [1,2]:
						# file_path_query1 = '%s/vbak2_6_7_0.1_0_0.1_0.1_0.25_0.1_0.01_3_2'%(file_save_path_1)
						file_path_query1 = '%s/vbak2_6_7_0.1_0_0.1_0.1_0.25_0.1_0.01_3_%d'%(file_save_path_1,group_id_1)
					elif folder_id_query in [0]:
						file_path_query1 = '%s/vbak2_6_7_0.1_0_0.1_0.1_0.25_0.1_0.01_2'%(file_save_path_1)
					dict_file_annot1.update({folder_id_query:file_path_query1})

					config_group_annot = select_config['config_group_annot']
					type_query_scale = 0
					config_id_2 = dict_query1[folder_id_query]
					file_path_query2 = '%s/train%d_%d_%d_pre2'%(file_path_query1,config_id_2,config_group_annot,type_query_scale)
					dict_file_annot2.update({folder_id_query:file_path_query2})

				select_config.update({'dict_file_annot1':dict_file_annot1,'dict_file_annot2':dict_file_annot2,'dict_config_annot1':dict_query1})

		return select_config

	## motif-peak estimate: load meta_exprs and peak_read
	def test_motif_peak_estimate_control_load_pre1_ori(self,meta_exprs=[],peak_read=[],flag_format=False,select_config={}):

		input_file_path1 = self.save_path_1
		# data_file_type = 'CD34_bonemarrow'
		# input_file_path = '%s/data_pre2/cd34_bonemarrow'%(input_file_path1)
		data_file_type = select_config['data_file_type']
		# input_file_path = select_config['data_path']
		# filename_save_annot_1 = select_config['filename_save_annot_1']
		
		input_filename_1, input_filename_2 = select_config['filename_rna'],select_config['filename_atac']
		input_filename_3 = select_config['filename_rna_exprs_1']
		# rna_meta_ad = sc.read_h5ad(input_filename_1)
		# atac_meta_ad = sc.read_h5ad(input_filename_2)
		rna_meta_ad = sc.read(input_filename_1)
		atac_meta_ad = sc.read(input_filename_2)
		meta_scaled_exprs = pd.read_csv(input_filename_3,index_col=0,sep='\t')
		print(input_filename_1,input_filename_2)
		print('rna_meta_ad\n', rna_meta_ad)
		print('atac_meta_ad\n', atac_meta_ad)

		# atac_meta_ad = self.atac_meta_ad
		# meta_scaled_exprs = self.meta_scaled_exprs
		if flag_format==True:
			meta_scaled_exprs.columns = meta_scaled_exprs.columns.str.upper()
			rna_meta_ad.var_names = rna_meta_ad.var_names.str.upper()
			rna_meta_ad.var.index = rna_meta_ad.var.index.str.upper()

		self.rna_meta_ad = rna_meta_ad
		sample_id = rna_meta_ad.obs_names
		sample_id1 = meta_scaled_exprs.index
		assert list(sample_id)==list(atac_meta_ad.obs_names)
		assert list(sample_id)==list(sample_id1)
		atac_meta_ad = atac_meta_ad[sample_id,:]
		self.atac_meta_ad = atac_meta_ad

		meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
		self.meta_scaled_exprs = meta_scaled_exprs
		print('atac_meta_ad, meta_scaled_exprs ',atac_meta_ad.shape,meta_scaled_exprs.shape,input_filename_3)

		peak_read = pd.DataFrame(index=atac_meta_ad.obs_names,columns=atac_meta_ad.var_names,data=atac_meta_ad.X.toarray(),dtype=np.float32)
		meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
		self.meta_exprs_2 = meta_exprs_2

		vec1 = utility_1.test_stat_1(np.mean(atac_meta_ad.X.toarray(),axis=0))
		vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
		vec3 = utility_1.test_stat_1(np.mean(meta_exprs_2,axis=0))

		print('atac_meta_ad mean values ',atac_meta_ad.shape,vec1)
		print('meta_scaled_exprs mean values ',meta_scaled_exprs.shape,vec2)
		print('meta_exprs_2 mean values ',meta_exprs_2.shape,vec3)

		return peak_read, meta_scaled_exprs, meta_exprs_2

	## load motif data
	def test_load_motif_data_1(self,method_type_vec=[],select_config={}):
		
		# x = 1
		flag_query1=1
		method_type_num = len(method_type_vec)
		dict_motif_data = dict()
		flag_query1=1
		motif_data_pre1, motif_data_score_pre1 = [], []
		data_file_type = select_config['data_file_type']
		for i1 in range(method_type_num):
			# method_type = method_type_vec[method_type_id]
			method_type = method_type_vec[i1]
			data_path = select_config['input_file_path_query'][method_type]
			input_file_path = data_path
			print('data_path_save: ',data_path)

			 #if (method_type in ['insilico','insilico_1']) or (method_type.find('joint_score')>-1):
			if (method_type.find('insilico')>-1) or (method_type.find('joint_score')>-1):
			# if method_type_id in [0,1]:
				# data_path = select_config['data_path']
				# data_path = select_config['input_file_path_query'][method_type_id]
				# input_file_path = data_path
				if (len(motif_data_pre1)==0) and (len(motif_data_score_pre1)==0):
					# input_filename1 = '%s/test_motif_data.1.h5ad'%(input_file_path)
					# input_filename2 = '%s/test_motif_data_score.1.h5ad'%(input_file_path)
					input_file_path_2 = '%s/peak_local/run1_1'%(input_file_path)
					data_file_type_query = select_config['data_file_type']

					if 'motif_filename_list1' in select_config:
						input_filename_list1 = select_config['motif_filename_list1']
						input_filename_list2 = []
					else:
						input_filename1 = '%s/test_motif_data.%s.1.thresh1.h5ad'%(input_file_path_2,data_file_type_query)
						input_filename2 = '%s/test_motif_data_score.%s.1.thresh1.h5ad'%(input_file_path_2,data_file_type_query)

						if os.path.exists(input_filename1)==False:
							print('the file does not exist: %s'%(input_filename1))
							input_filename1 = '%s/test_motif_data.%s.1.h5ad'%(input_file_path_2,data_file_type_query)
							input_filename2 = '%s/test_motif_data_score.%s.1.h5ad'%(input_file_path_2,data_file_type_query)

						input_filename_list1 = [input_filename1,input_filename2]
						input_filename_list2 = []

					print('motif_filename_list1: ',input_filename_list1)
					# input_file_path2 = '%s/peak_local'%(data_path)
					# output_file_path = input_file_path2
					# save_file_path = output_file_path
					save_file_path = ''
					flag_query2 = 1
					motif_data, motif_data_score = self.test_load_motif_data_pre1(input_filename_list1=input_filename_list1,
																						input_filename_list2=input_filename_list2,
																						flag_query1=1,flag_query2=flag_query2,
																						input_file_path=input_file_path,
																						save_file_path=save_file_path,type_id=1,
																						select_config=select_config)
					dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
					flag_query1=0
					motif_data_pre1 = motif_data
					motif_data_score_pre1 = motif_data_score
				else:
					motif_data, motif_data_score = motif_data_pre1, motif_data_score_pre1
					print('motif_data loaded ',motif_data.shape,motif_data_score.shape,method_type,i1)
					dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
			else:
				# motif_data = pd.read_csv('google-us-data.csv.gz', compression='gzip', header=0,    sep=' ', quotechar='"', error_bad_lines=False)
				# scipy.io.mmread("sparse_from_file")
				# input_filename1 = select_config['filename_motif_data'][method_type]['motif_data']
				# input_filename2 = select_config['filename_motif_data'][method_type]['motif_data_score']
				# input_filename = 'df_peak_tf_overlap.pearsonr.CD34_bonemarrow.normalize0.tsv.gz'
				dict_file_query = select_config['filename_motif_data'][method_type]
				key_vec = list(dict_file_query.keys()) # key_vec: ['motif_data','motif_data_score']
				dict_query = dict()
				for key_query in key_vec:
					input_filename = dict_file_query[key_query]
					motif_data = []
					flag_matrix=0
					if input_filename!='':
						if (os.path.exists(input_filename)==True):
							print('load motif data: %s'%(input_filename))
							b = input_filename.find('.gz')
							if b>-1:
								motif_data = pd.read_csv(input_filename,compression='gzip',index_col=0,sep='\t')
							else:
								b = input_filename.find('.matrix')
								if b>-1:
									print('load matrix market format data ',method_type)
									motif_data = scipy.io.mmread(input_filename)
									motif_data = motif_data.toarray()
									flag_matrix=1	
								else:
									motif_data = pd.read_csv(input_filename,index_col=0,sep='\t')
							print('motif_data ',motif_data.shape)
							print(motif_data[0:5])	
						else:
							print('the file does not exist ',input_filename)
							continue
					else:
						print('please provide motif data file name ')

					if (method_type in ['GRaNIE','Pando','TRIPOD']) and (len(motif_data)>0):
						# x = 1
						# input_file_path = select_config[]
						# input_file_path = select_config['input_file_path_query'][method_type]
						input_filename_annot = '%s/TFBS/translationTable.csv'%(input_file_path)
						if method_type in ['TRIPOD']:
							pre_config = select_config['config_query'][method_type]
							type_id_query = pre_config['type_id_query']
							input_filename_annot = '%s/TFBS/translationTable%d.csv'%(input_file_path,type_id_query)
							if type_id_query==0:
								input_filename_annot = '%s/TFBS/translationTable%d_pre.csv'%(input_file_path,type_id_query) # temporary: MA0091.1	TAL1	TAL1; MA0091.1	TAL1::TCF3	TAL1::TCF3
						if method_type in ['GRaNIE']:
							df_annot1 = pd.read_csv(input_filename_annot,index_col=0,sep=' ')
							df_annot1.loc[:,'tf'] = np.asarray(df_annot1.index)
							df_annot1.index = np.asarray(df_annot1['HOCOID'])
							print('df_annot1 ',df_annot1.shape,method_type)
							print(df_annot1[0:2])
						else:
							df_annot1 = pd.read_csv(input_filename_annot,index_col=0,header=None,sep='\t')
							if len(df_annot1.columns)==1:
								df_annot1.columns = ['tf_ori']
								tf_id_ori = df_annot1['tf_ori']
								tf_id = pd.Index(tf_id_ori).str.split('(').str.get(0)
								df_annot1.loc[:,'tf'] = tf_id
							else:
								df_annot1.columns = ['tf_ori','tf']
							print('df_annot1 ',df_annot1.shape,method_type)
							print(df_annot1[0:2])
							if method_type in ['Pando']:
								pre_config = select_config['config_query'][method_type]
								run_id = pre_config['run_id']
								metacell_num = pre_config['metacell_num']
								exclude_exons = pre_config['exclude_exons']
								type_id_region = pre_config['type_id_region']
								data_file_type_annot = data_file_type.lower()
								input_file_path2 = '%s/%s/metacell_%d/run%d'%(input_file_path,data_file_type_annot,metacell_num,run_id)
								input_filename = '%s/test_region.%d.%d.bed'%(input_file_path2,exclude_exons,type_id_region)
								flag_region_query=((exclude_exons==True)|(type_id_region>0))
								if os.path.exists(input_filename)==True:
									df_region = pd.read_csv(input_filename,index_col=False,sep='\t')
									df_region.index = np.asarray(df_region['id'])
									# pre_config.update({'df_region':df_region})
									df_region_ori = df_region.copy()
									df_region = df_region.sort_values(by=['overlap'],ascending=False)
									df_region = df_region.loc[~df_region.index.duplicated(keep='first'),:]
									df_region = df_region.sort_values(by=['region_id'],ascending=True)
									output_file_path = input_file_path2
									output_filename = '%s/test_region.%d.%d.2.bed'%(output_file_path,exclude_exons,type_id_region)
									df_region.to_csv(output_filename,sep='\t')
									select_config['config_query'][method_type].update({'df_region':df_region})
								else:
									print('the file does not exist ',input_filename)

								if flag_matrix==1:
									## the motif data is loaded from MM format file and the rownames and colnames to be added
									motif_idvec_ori = df_annot1.index
									# motif_data.columns = motif_idvec_ori
									# motif_data.index = df_region.index
									motif_data = pd.DataFrame(index=df_region.index,columns=motif_idvec_ori,data=np.asarray(motif_data))
									# print('motif_data ',motif_data.shape,method_type)
									# print(motif_data[0:5])
							
								if flag_region_query>0:
									region_id = motif_data.index
									motif_data_ori = motif_data.copy()
									motif_data.loc[:,'peak_id'] = df_region.loc[region_id,'peak_loc']
									motif_data = motif_data.groupby('peak_id').max()
									print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape,method_type)
									dict_query.update({'%s.ori'%(key_query):motif_data_ori})

						# motif_idvec_1= df_annot1.index
						motif_idvec = motif_data.columns.intersection(df_annot1.index,sort=False)
						motif_data = motif_data.loc[:,motif_idvec]
						motif_data_ori = motif_data.copy()
						motif_data1 = motif_data.T
						motif_idvec = motif_data1.index
						motif_data1.loc[:,'tf'] = df_annot1.loc[motif_idvec,'tf']
						motif_data1 = motif_data1.groupby('tf').max()
						motif_data = motif_data1.T
						print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape,method_type)
						print(motif_data[0:5])
						field_id = '%s.ori'%(key_query)
						if not (field_id in dict_query):
							dict_query.update({'%s.ori'%(key_query):motif_data_ori})

					dict_query.update({key_query:motif_data})
			dict_motif_data[method_type] = dict_query

		return dict_motif_data, select_config

	## query file save path
	# query the filename of the estimated peak-TF-gene link query
	def test_query_file_path_1(self,data_file_type='',save_mode=1,verbose=0,select_config={}):

		if data_file_type=='':
			data_file_type_query = select_config['data_file_type']
		else:
			data_file_type_query = data_file_type

		dict_file_query = dict()
		# if len(dict_file_query)==0:
		# 	file_path_motif_score = select_config['file_path_motif_score_2']
		# 	input_file_path_query = file_path_motif_score
			
		# 	# input_filename_1 = '%s/test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_2)
		# 	# input_filename_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.txt.gz'%(input_file_path_query,data_file_type_query)
		# 	input_filename_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
		# 	# input_filename_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_2,data_file_type_query)
		# 	# input_filename_3 = '%s/test_query_gene_peak.%s.2.pre1.link_query.2_1.combine.100_0.01.500_-0.05.1.3.2.2.txt.gz'%(input_file_path_2,data_file_type_query)
		# 	# input_filename_2 = 'test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'
		# 	input_filename_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
		# 	input_filename_3 = '%s/test_query_gene_peak.%s.2.pre1.link_query.2_1.combine.100_0.01.500_-0.05.1.3.2.2.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
					
		# 	# input_filename_pre1_1 = '%s/test_motif_score_normalize_insilico.CD34_bonemarrow.thresh1.1.txt'%(input_file_path_query)
		# 	# input_filename_pre1_2 = '%s/test_motif_score_normalize_insilico.CD34_bonemarrow.thresh1.1.thresh0.1.txt'%(input_file_path_query)
		# 	input_filename_pre1_1 = '%s/test_motif_score_normalize_insilico.%s.txt'%(input_file_path_query,data_file_type_query)
		# 	input_filename_pre1_2 = '%s/test_motif_score_normalize_insilico.%s.thresh0.1.txt'%(input_file_path_query,data_file_type_query)
		# 	input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.copy2_1.txt.gz'%(input_file_path_query,data_file_type_query)
		# 	input_filename_pre2_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.copy2_1.txt.gz'%(input_file_path_query,data_file_type_query)

		# 	filename_list2 = [input_filename_1,input_filename_2,input_filename_3]+[input_filename_pre1_1,input_filename_pre1_2,input_filename_pre2_1,input_filename_pre2_2]
		# 	method_type_annot = ['joint_score_1','joint_score_2','joint_score_3']+['insilico','insilico_0.1','joint_score_pre1','joint_score_pre2']

		# 	# filename_list2 = [input_filename_1,input_filename_2,input_filename_3]+[input_filename_pre1_2,input_filename_pre1_2,input_filename_pre2_2]
		# 	# method_type_annot = ['joint_score_1','joint_score_2','joint_score_3']+['insilico','joint_score_pre1','joint_score_pre2']

		# 	dict_file_query = dict(zip(method_type_annot,filename_list2))
		# 	# query_num2 = len(filename_list2)

		if len(dict_file_query)==0:
			if data_file_type_query in ['CD34_bonemarrow']:
				file_path_motif_score = select_config['file_path_motif_score_2']
				input_file_path_query = file_path_motif_score

				# input_filename_1 = '%s/test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_2)
				# input_filename_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.txt.gz'%(input_file_path_query,data_file_type_query)
				input_filename_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
				# input_filename_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_2,data_file_type_query)
				# input_filename_3 = '%s/test_query_gene_peak.%s.2.pre1.link_query.2_1.combine.100_0.01.500_-0.05.1.3.2.2.txt.gz'%(input_file_path_2,data_file_type_query)
				# input_filename_2 = 'test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'
				input_filename_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
				input_filename_3 = '%s/test_query_gene_peak.%s.2.pre1.link_query.2_1.combine.100_0.01.500_-0.05.1.3.2.2.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
						
				# input_filename_pre1_1 = '%s/test_motif_score_normalize_insilico.CD34_bonemarrow.thresh1.1.txt'%(input_file_path_query)
				# input_filename_pre1_2 = '%s/test_motif_score_normalize_insilico.CD34_bonemarrow.thresh1.1.thresh0.1.txt'%(input_file_path_query)
				input_filename_pre1_1 = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path_query,data_file_type_query)
				input_filename_pre1_2 = '%s/test_motif_score_normalize_insilico.%s.thresh0.1.txt'%(input_file_path_query,data_file_type_query)
				input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.txt.gz'%(input_file_path_query,data_file_type_query)
				input_filename_pre2_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_query,data_file_type_query)

				filename_list2 = [input_filename_1,input_filename_2,input_filename_3]+[input_filename_pre1_1,input_filename_pre1_2,input_filename_pre2_1,input_filename_pre2_2]
				method_type_annot = ['joint_score_1','joint_score_2','joint_score_3']+['insilico','insilico_1','joint_score_pre1','joint_score_pre2']
				dict_file_query = dict(zip(method_type_annot,filename_list2))
				# query_num2 = len(filename_list2)

			elif data_file_type_query in ['pbmc']:
				file_path_motif_score = select_config['file_path_motif_score_2']
				input_file_path_query = file_path_motif_score

				input_filename_pre1_1 = '%s/test_motif_score_normalize_insilico.%s.thresh1.1.txt'%(input_file_path_query,data_file_type_query)
				input_filename_pre1_2 = '%s/test_motif_score_normalize_insilico.%s.thresh1.1.thresh0.1.txt'%(input_file_path_query,data_file_type_query)
				input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.txt.gz'%(input_file_path_query,data_file_type_query)
				input_filename_pre2_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_query,data_file_type_query)

				filename_list2 = [input_filename_pre1_1,input_filename_pre1_2,input_filename_pre2_1,input_filename_pre2_2]
				method_type_annot = ['insilico','insilico_1','joint_score_pre1','joint_score_pre2']
				dict_file_query = dict(zip(method_type_annot,filename_list2))

		return dict_file_query

	## feature query for TF and peak loci
	# perform feature dimension reduction
	def test_query_feature_pre1_1(self,feature_mtx=[],method_type='SVD',n_components=50,sub_sample=-1,verbose=0,select_config={}):

		vec1 = ['PCA','Incremental PCA','Kernel PCA','Sparse PCA','SVD',
					'GRP','SRP','MDS','ISOMAP','Minibatch','ICA','tSNE','LLE','Encoder',-1,'NMF']
		query_num1 = len(vec1)
		idvec_1 = np.arange(query_num1)
		dict_1 = dict(zip(vec1,idvec_1))

		start = time.time()
		# method_type_query = vec1[type_id_reduction]
		method_type_query = method_type
		type_id_reduction = dict_1[method_type_query]
		feature_mtx_1 = feature_mtx
		if verbose>0:
			print('feature_mtx, method_type_query: ',feature_mtx_1.shape,method_type_query)
			print(feature_mtx_1[0:2])

		# sub_sample = -1
		from utility_1 import dimension_reduction
		feature_mtx_pre, dimension_model = dimension_reduction(x_ori=feature_mtx_1,feature_dim=n_components,type_id=type_id_reduction,shuffle=False,sub_sample=sub_sample)
		df_latent = feature_mtx_pre
		df_component = dimension_model.components_  # shape: (n_components,n_features)

		return dimension_model, df_latent, df_component

	## feature query for TF and peak loci
	def test_query_feature_pre1_2(self,peak_query_vec=[],gene_query_vec=[],motif_data=[],motif_data_score=[],motif_group=[],method_type_vec=['SVD','SVD','SVD'],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],n_components=50,sub_sample=-1,flag_shuffle=False,float_format='%.6f',input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',output_filename='',verbose=0,select_config={}):

		motif_query_vec = motif_data.columns.intersection(rna_exprs.columns,sort=False)
		if len(gene_query_vec)==0:
			# gene_query_vec = motif_query_vec
			annot_str_vec = ['peak_tf','peak_motif','peak_motif_ori']
		else:
			# gene_query_vec = pd.Index(gene_query_vec).union(motif_query_vec,sort=False)
			annot_str_vec = ['peak_gene','peak_motif','peak_motif_ori']

		sample_id = rna_exprs_unscaled.index
		peak_read = peak_read.loc[sample_id,:]
		
		feature_expr_query1 = rna_exprs_unscaled.loc[:,gene_query_vec].T # tf expression, shape: (tf_num,cell_num)
		feature_mtx_query1 = peak_read.loc[:,peak_query_vec].T  # peak matrix, shape: (peak_num,cell_num)
		
		feature_motif_query1 = motif_data.loc[peak_query_vec,motif_query_vec] # motif matrix of peak, shape: (peak_num,motif_num)
		feature_motif_query2 = motif_data.loc[peak_query_vec,:] # motif matrix of peak, shape: (peak_num,motif_num)
					
		# feature_query_2 = df_1.loc[peak_query_1,['signal',column_motif]+field_query_2]
		# print('feature_query_2: ',feature_query_2.shape,group_id)
		# print(feature_query_2[0:2])
					
		# feature_motif_query1 = motif_data.loc[peak_query_1,[motif_id_query]] # (peak_num,motif_num)
		# feature_motif_query2 = motif_data_score_query1.loc[peak_query_1,[motif_id_query]] # (peak_num,motif_num)
		# feature_motif_query2 = feature_motif_query2.rename(columns={motif_id_query:column_1})
		# list2 = [feature_mtx_query1,feature_motif_query1,feature_motif_query2,feature_query_2]
		# list1 = [feature_mtx_query1,feature_motif_query1]

		flag_group = 0
		if len(motif_group)>0:
			flag_group = 1

		feature_motif_query_2 = []
		list1 = []
		if flag_group>0:
			feature_motif_query_2 = motif_group.loc[peak_query_vec,:] # (peak_num,group_num)
			list1 = list1 + [feature_motif_query_2]

		feature_mtx_1 = pd.concat([feature_expr_query1,feature_mtx_query1],axis=0,join='outer',ignore_index=False)
		feature_mtx_2 = feature_motif_query1
		feature_mtx_2_ori = feature_motif_query2

		list_pre1 = [feature_mtx_1,feature_mtx_2,feature_mtx_2_ori]
		# method_type_vec = ['SVD','SVD']

		query_num1 = len(list_pre1)
		dict_query1 = dict()
		dict_query1.update({'df_exprs_1':feature_expr_query1,'df_peak':feature_mtx_query1,
							'df_peak_motif':feature_motif_query1,'df_peak_motif_ori':feature_motif_query2})

		# flag_shuffle = False
		annot_str_vec_2 = annot_str_vec[0:1]+['motif','motif_ori']
		for i1 in range(query_num1):
			feature_mtx_query = list_pre1[i1]
			annot_str1 = annot_str_vec[i1]

			query_id_1 = feature_mtx_query.index.copy()
			print('feature_mtx_query: ',feature_mtx_query.shape,annot_str1,i1)

			if (flag_shuffle>0):
				query_num = len(query_id_1)
				id1 = np.random.permutation(query_num)
				query_id_1 = query_id_1[id1]
				feature_mtx_query = feature_mtx_query.loc[query_id_1,:]

			# sub_sample = -1
			method_type = method_type_vec[i1]

			# n_components_query = 50
			n_components_query = n_components

			# perform feature dimension reduction
			dimension_model, df_latent, df_component = self.test_query_feature_pre1_1(feature_mtx=feature_mtx_query,method_type=method_type,n_components=n_components_query,sub_sample=sub_sample,verbose=verbose,select_config=select_config)

			feature_dim_vec = ['feature%d'%(id1+1) for id1 in range(n_components_query)]
			feature_vec_1 = query_id_1
			df_latent = pd.DataFrame(index=feature_vec_1,columns=feature_dim_vec,data=df_latent)

			feature_vec_2 = feature_mtx_query.columns
			df_component = df_component.T
			df_component = pd.DataFrame(index=feature_vec_2,columns=feature_dim_vec,data=df_component)
			
			if i1==0:
				df_latent_peak = df_latent.loc[peak_query_vec,:]
				df_latent_tf = df_latent.loc[motif_query_vec,:]
				df_latent_gene = df_latent.loc[gene_query_vec,:]
				print('df_latent_peak: ',df_latent_peak.shape,annot_str1)
				print(df_latent_peak[0:2])
				print('df_latent_tf: ',df_latent_tf.shape)
				print(df_latent_tf[0:2])
				print('df_latent_gene: ',df_latent_gene.shape)
				print(df_latent_gene[0:2])
				dict_query1.update({'dimension_model_1':dimension_model}) # dimension reduction model for peak accessibility and TF expression
				# dict_query1.update({'latent_peak':df_latent_peak,'latent_tf':df_latent_tf,'latent_gene':df_latent_gene})
				dict_query1.update({'latent_peak':df_latent_peak,'latent_gene':df_latent_gene,
									'component_mtx':df_component})
			else:
				print('df_latent: ',df_latent.shape,annot_str1)
				print(df_latent[0:2])
				df_latent_peak_motif = df_latent.loc[peak_query_vec,:]
				print('df_latent_peak_motif: ',df_latent_peak_motif.shape)
				print('df_component: ',df_component.shape)
				# print(df_latent_peak_motif[0:2])
				annot_str2 = annot_str_vec_2[i1]
				# dict_query1.update({'dimension_model_motif':dimension_model}) # dimension reduction model for motif feature of peak query
				# dict_query1.update({'latent_peak_motif':df_latent_peak_motif,'component_peak_motif':df_component})

				dict_query1.update({'dimension_model_%s'%(annot_str2):dimension_model}) # dimension reduction model for motif feature of peak query
				dict_query1.update({'latent_%s'%(annot_str1):df_latent_peak_motif,'component_%s'%(annot_str1):df_component})

			if save_mode>0:
				filename_save_annot_2 = '%s_%s'%(method_type,n_components_query)
				output_filename_1 = '%s/%s.dimension_model.%s.%s.1.h5'%(output_file_path,filename_prefix_save,annot_str1,filename_save_annot_2)
				pickle.dump(dimension_model, open(output_filename_1, 'wb'))

				list_query2 = [df_latent,df_component]
				field_query_2 = ['df_latent','df_component']
				for (field_id,df_query) in zip(field_query_2,list_query2):
					filename_prefix_save_2 = '%s.%s'%(filename_prefix_save,field_id)
					output_filename = '%s/%s.%s.%s.1.txt'%(output_file_path,filename_prefix_save_2,annot_str1,filename_save_annot_2)
					df_query.to_csv(output_filename,sep='\t',float_format=float_format)

		return dict_query1

	## feature query for TF and peak loci
	def test_query_feature_pre1_3(self,df_feature_link=[],df_annot=[],feature_query_vec=[],column_id_query='',column_vec=[],column_value='',feature_type_vec=[],peak_query_vec=[],gene_query_vec=[],motif_data=[],motif_data_score=[],motif_group=[],method_type_vec=['SVD'],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],n_components=50,sub_sample=-1,flag_shuffle=False,flag_binary=1,thresh_value=-0.1,float_format='%.6f',flag_unduplicate=1,input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',output_filename='',verbose=0,select_config={}):

		motif_query_vec = motif_data.columns.intersection(rna_exprs.columns,sort=False)
		# if len(gene_query_vec)==0:
		# 	# gene_query_vec = motif_query_vec
		# 	annot_str_vec = ['peak_tf','peak_motif','peak_motif_ori']
		# else:
		# 	# gene_query_vec = pd.Index(gene_query_vec).union(motif_query_vec,sort=False)
		# 	annot_str_vec = ['peak_gene','peak_motif','peak_motif_ori']

		flag_query1 = 1
		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]
		
		if flag_query1>0:
			# flag_unduplicate = 1
			if (column_value!=''):
				if (not (column_value in df_feature_link.columns)):
					if len(df_annot)==0:
						print('please provide anontation file for %s'%(column_value))
						return

					from utility_1 import test_query_index, test_column_query_1
					if column_value in ['peak_tf_corr']:
						if flag_unduplicate>0:
							df_feature_link.drop_duplicates(subset=[column_id2,column_id3])

						flag_unduplicate = 0
						df_feature_link.index = utility_1.test_query_index(df_feature_link,column_vec=[column_id2,column_id3])

						df_list1 = [df_feature_link,df_annot]				
						# column_idvec_1 = ['motif_id','peak_id','gene_id']
						column_vec_1 = [column_id2,column_id3]
						column_value_query = 'correlation_score'
						column_vec_annot = [[column_value_query]]
						df_feature_link = utility_1.test_column_query_1(input_filename_list=[],id_column=column_vec_1,column_vec=column_vec_annot,
																			df_list=df_list1,type_id_1=0,type_id_2=0,reset_index=False,select_config=select_config)

						df_feature_link = df_feature_link.rename(columns={column_value_query:column_value})
					else:
						column_vec_annot = [[column_value]]
						column_vec_1 = column_idvec
						df_feature_link = utility_1.test_column_query_1(input_filename_list=[],id_column=column_vec_1,column_vec=column_vec_annot,
																			df_list=df_list1,type_id_1=0,type_id_2=0,reset_index=False,select_config=select_config)

					column_vec_query = column_idvec+[column_value]
			else:
				column_vec_query = column_idvec

			if len(feature_query_vec)>0:
				df_feature_link.index = np.asarray(df_feature_link[column_id_query])
				df_link_query = df_feature_link.loc[feature_query_vec,column_vec_query]
			else:
				df_link_query = df_feature_link.loc[:,column_vec_query]
			
			# feature_type_vec_1 = [feature_type_query2,feature_type_query1]
			
			# column_vec = [column_id3,column_id2]
			# feature_type_vec_1 = ['motif','peak']
			from utility_1 import test_query_feature_format_1
			# convert the long format dataframe to wide format
			t_vec_1 = test_query_feature_format_1(df_feature_link=df_link_query,feature_query_vec=feature_query_vec,feature_type_vec=feature_type_vec,column_vec=column_vec,column_value=column_value,flag_unduplicate=flag_unduplicate,
													format_type=0,save_mode=0,filename_prefix_save='',output_file_path='',output_filename='',verbose=verbose,select_config=select_config)
				
			df_link_query1, feature_mtx_1, feature_vec_1, feature_vec_2 = t_vec_1

			df_feature_link_1 = feature_mtx_1
			df_mask = df_feature_link_1  # binary feature association matrix
			
			t_value_1 = df_feature_link_1.sum(axis=1)
			print('df_feature_link_1: ',df_feature_link_1.shape)
			print(df_feature_link_1[0:2])
			print(t_value_1[0:2])
			print(np.max(t_value_1),np.min(t_value_1),np.mean(t_value_1),np.median(t_value_1))

			print('feature_vec_1: ',len(feature_vec_1))
			print(feature_vec_1[0:2])

			print('feature_vec_2: ',len(feature_vec_2))
			print(feature_vec_2[0:2])

			# flag_binary = 1
			feature_mtx_1 = feature_mtx_1.fillna(0)
			if flag_binary>0:
				# feature_mtx_query = 2.0*(feature_mtx_1>thresh_value)-1
				feature_mtx_query = feature_mtx_1
				feature_mtx_query[feature_mtx_query>=thresh_value] = 1.0
				feature_mtx_query[feature_mtx_query<thresh_value] = -1.0
			else:
				feature_mtx_query = feature_mtx_1

			print('feature_mtx_query: ',feature_mtx_query.shape)
			print(np.max(feature_mtx_query.max(axis=0)),np.min(feature_mtx_query.min(axis=0)))

			peak_loc_ori = motif_data.index
			# motif_query_vec_pre1 = motif_data.columns
			feature_query_1 = feature_mtx_query.index
			feature_query_2 = feature_mtx_query.columns

			motif_query_vec = feature_query_2
			feature_mtx_query_1 = pd.DataFrame(index=peak_loc_ori,columns=motif_query_vec)
			feature_mtx_query_1.loc[feature_query_1,feature_query_2] = feature_mtx_query.loc[feature_query_1,feature_query_2]

			feature_vec_2 = pd.Index(peak_loc_ori).difference(feature_query_1,sort=False) # the peak loci not included
			feature_mtx_query_1.loc[feature_vec_2,motif_query_vec] = motif_data.loc[feature_vec_2,motif_query_vec].copy() # use the motif scanning value for the peak loci not included
			feature_mtx_query_1 = feature_mtx_query_1.fillna(0)
			print('feature_vec_2: ',len(feature_vec_2))
			print('feature_mtx_query_1: ',feature_mtx_query_1.shape)
			# print(feature_mtx_query_1.columns)
			print(feature_mtx_query_1[0:2])

			method_type = method_type_vec[0]
			# n_components_query = 50
			n_components_query = n_components
			# perform feature dimension reduction
			dimension_model, df_latent, df_component = self.test_query_feature_pre1_1(feature_mtx=feature_mtx_query_1,method_type=method_type,n_components=n_components_query,sub_sample=sub_sample,verbose=verbose,select_config=select_config)

			feature_dim_vec = ['feature%d'%(id1+1) for id1 in range(n_components_query)]
			# feature_vec_1 = query_id_1
			feature_vec_query_1 = feature_mtx_query_1.index
			df_latent = pd.DataFrame(index=feature_vec_query_1,columns=feature_dim_vec,data=df_latent)

			feature_vec_query_2 = feature_mtx_query_1.columns
			df_component = df_component.T
			df_component = pd.DataFrame(index=feature_vec_query_2,columns=feature_dim_vec,data=df_component)

			annot_str1 = 'peak_tf_link'
			print('df_latent: ',df_latent.shape,annot_str1)
			print(df_latent[0:2])
			print('df_component: ',df_component.shape)
			print(df_component[0:2])
			
			# print(df_latent_peak_motif[0:2])
			# annot_str2 = annot_str_vec_2[i1]
			# dict_query1.update({'dimension_model_motif':dimension_model}) # dimension reduction model for motif feature of peak query
			# dict_query1.update({'latent_peak_motif':df_latent_peak_motif,'component_peak_motif':df_component})

			dict_query1 = dict()
			dict_query1.update({'dimension_model_%s'%(annot_str1):dimension_model}) # dimension reduction model for motif feature of peak query
			dict_query1.update({'latent_%s'%(annot_str1):df_latent,'component_%s'%(annot_str1):df_component})

			if save_mode>0:
				filename_save_annot_2 = '%s_%s'%(method_type,n_components_query)
				output_filename_1 = '%s/%s.dimension_model.%s.%s.1.h5'%(output_file_path,filename_prefix_save,annot_str1,filename_save_annot_2)
				pickle.dump(dimension_model, open(output_filename_1, 'wb'))

				list_query2 = [df_latent,df_component]
				field_query_2 = ['df_latent','df_component']
				for (field_id,df_query) in zip(field_query_2,list_query2):
					filename_prefix_save_2 = '%s.%s'%(filename_prefix_save,field_id)
					output_filename = '%s/%s.%s.%s.1.txt'%(output_file_path,filename_prefix_save_2,annot_str1,filename_save_annot_2)
					df_query.to_csv(output_filename,sep='\t',float_format=float_format)

			# feature_query_1 = feature_query_vec
			# feature_query_2 = np.unique(df_feature_link.loc[feature_query_vec,column_id2])
			# feature_vec_2 = df_mask.columns
			# assert list(np.unique(feature_vec_2))==list(np.unique(feature_query_2))
			# assert list(feature_vec_2)==list(feature_query_2)
			# print('feature_vec_2: ',len(feature_vec_2))
			# print(feature_vec_2[0:2])
			
			# print('feature_query_2: ',len(feature_query_2))
			# print(feature_query_2[0:2])
			
			# # assert list(feature_vec_2)==list(feature_query_2)
			# feature_vec = pd.Index(feature_vec_2).difference(feature_query_2,sort=False)
			# print('feature_vec: ',len(feature_vec))
			# print(feature_vec)
			# df_mask = df_mask.loc[:,feature_query_2]
			# feature_query_2 = np.unique(df_feature_link.loc[feature_query_vec,column_id2])

			return dict_query1

	## load metacell data and motif data
	# def test_query_compare_binding_pre1_3(self,data=[],motif_id_query='',motif_id='',method_type_vec=[],method_type_vec_query=[],peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
	def test_query_feature_load_basic_1(self,data=[],method_type_vec=[],peak_read=[],rna_exprs=[],peak_distance_thresh=100,flag_config_1=1,flag_load_1=1,flag_load_2=1,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		run_id1 = select_config['run_id']
		# thresh_num1 = 5
		# method_type_vec = ['insilico_1','TRIPOD','GRaNIE','Pando']+['joint_score.thresh%d'%(i1+1) for i1 in range(thresh_num1)]+['joint_score_2.thresh3']
		# method_type_vec = ['insilico_1','GRaNIE','joint_score.thresh1']
		# method_type_vec = ['GRaNIE']
		# method_type_vec = ['insilico_1','joint_score.thresh1','joint_score.thresh2','joint_score.thresh3']
		if len(method_type_vec)==0:
			# method_type_vec = ['insilico_1','TRIPOD','GRaNIE','Pando']+['joint_score_2.thresh3']
			method_type_vec = ['insilico_0.1','TRIPOD','GRaNIE','Pando']+['joint_score_pre1.thresh3','joint_score_pre1.thresh22']

		# root_path_1 = select_config['root_path_1']
		# root_path_2 = select_config['root_path_2']
		# method_type_vec_query = method_type_vec
		# if data_file_type_query in ['CD34_bonemarrow']:
		# 	input_file_path = '%s/peak1'%(root_path_2)
		# elif data_file_type_query in ['pbmc']:
		# 	input_file_path = '%s/peak2'%(root_path_2)

		# peak_distance_thresh = 100
		# filename_prefix_1 = 'test_motif_query_binding_compare'
		# file_save_path_1 = input_file_path

		method_query_num1 = len(method_type_vec)
		method_type_idvec = np.arange(method_query_num1)
		dict_method_type = dict(zip(method_type_vec,method_type_idvec))
		select_config.update({'dict_method_type':dict_method_type})

		# file_path_query1 = '%s/vbak2_6'%(input_file_path)
		# file_path_query1 = '%s/vbak2_6_5_0.1_0_0.1_0.1_0.25_0.1'%(input_file_path)
		# file_path_query1 = '%s/vbak2_6_7_0.1_0_0.1_0.1_0.25_0.1_0.01'%(input_file_path)
		# input_file_path = file_path_query1
		# output_file_path = file_path_query1
		# method_type_vec_query = method_type_vec
		# input_file_path_query = '/data/peer/yangy4/data1/data_pre2/cd34_bonemarrow/data_1/run0/'
		# root_path_1 = select_config['root_path_1']

		# if data_file_type_query in ['CD34_bonemarrow']:
		# 	data_file_type_annot = data_file_type_query.lower()
		# 	run_id_1 = 0
		# 	input_file_path2 = '%s/data_pre2/%s/peak_local'%(root_path_1,data_file_type_annot)
		# 	input_file_path_query1 = '%s/data_pre2/%s/data_1/run%d/peak_local'%(root_path_1,data_file_type_annot,run_id_1)
		# 	input_file_path_query = '%s/seacell_1'%(input_file_path_query1)

		# 	filename_1 = '%s/test_rna_meta_ad.%s.log_normalize.2.h5ad'%(input_file_path_query,data_file_type_query)
		# 	filename_2 = '%s/test_atac_meta_ad.%s.log_normalize.2.h5ad'%(input_file_path_query,data_file_type_query)
		# 	filename_3_ori = '%s/test_query.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
		# 	# filename_3 = '%s/test_rna_meta_ad.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
		
		# elif data_file_type_query in ['pbmc']:
		# 	data_file_type_annot = '10x_pbmc'
		# 	# run_id_1 = 0
		# 	input_file_path2 = '%s/data_pre2/%s/peak_local'%(root_path_1,data_file_type_annot)
		# 	# input_file_path_query1 = '%s/data_pre2/%s/data_1/run%d/peak_local'%(root_path_1,data_file_type_annot,run_id_1)
		# 	input_file_path_query1 = '%s/data_pre2/%s/data_1/data1_vbak1/peak_local'%(root_path_1,data_file_type_annot)
		# 	input_file_path_query = '%s/seacell_1'%(input_file_path_query1)

		# 	type_id_feature = 0
		# 	run_id1 = 1
		# 	filename_save_annot = '%s.%d.%d'%(data_file_type_query,type_id_feature,run_id1)
		# 	filename_1 = '%s/test_rna_meta_ad.%s.h5ad'%(input_file_path_query,filename_save_annot)
		# 	filename_2 = '%s/test_atac_meta_ad.%s.h5ad'%(input_file_path_query,filename_save_annot)
		# 	# filename_3_ori = '%s/test_query.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
		# 	# filename_3_ori = '%s/test_rna_meta_ad.pbmc.0.1.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
		# 	filename_3_ori = '%s/test_rna_meta_ad.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,filename_save_annot)
			
		# select_config.update({'filename_rna':filename_1,'filename_atac':filename_2,
		# 						'filename_rna_exprs_1':filename_3_ori})

		# flag_config_1=0
		if flag_config_1>0:
			# root_path_1 = select_config['root_path_1']
			# root_path_2 = select_config['root_path_2']
			# data_file_type_query = select_config['data_file_type']
			# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre2.thresh3','joint_score_pre1.thresh22','joint_score_pre2.thresh6']
			# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre1.thresh22']
			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)

		if flag_load_1>0:
			print('load metacell peak accessibility and gene expression data')
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori(meta_exprs=[],peak_read=[],flag_format=False,select_config=select_config)
		
			# rna_exprs = meta_scaled_exprs
			# rna_exprs = meta_exprs_2
			# sample_id = rna_exprs.index
			sample_id = meta_scaled_exprs.index
			peak_read = peak_read.loc[sample_id,:]
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			rna_exprs = meta_scaled_exprs
			print('peak_read, rna_exprs: ',peak_read.shape,rna_exprs.shape)

			print(peak_read[0:2])
			print(rna_exprs[0:2])

		## load motif data
		dict_motif_data = dict()
		if flag_load_2>0:
			print('load motif data')
			print('method type: ',method_type_vec)
			start = time.time()
			# data_path = select_config['input_file_path_query'][method_type]
			# dict_file_query = select_config['filename_motif_data'][method_type]

			# dict_query: {'motid_data','motif_data_score'}
			# dict_motif_data[method_type] = dict_query
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec,
																			select_config=select_config)

			stop = time.time()
			print('load motif data used %.2fs'%(stop-start))

		return (peak_read, meta_scaled_exprs, meta_exprs_2), dict_motif_data, select_config

	# compute feature embedding
	def test_query_feature_mtx_1(self,feature_query_vec=[],feature_type_vec=[],method_type_vec_dimension=[],n_components=50,type_id_group=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=1,select_config={}):

		# dict_query1: {'latent_peak','latent_tf','latent_peak_motif'}
		if len(method_type_vec_dimension)==0:
			method_type_vec_dimension = ['SVD','SVD','SVD']
		
		# n_components = 50
		float_format='%.6f'
		# print('peak_query_vec_pre1: ',len(peak_query_vec_pre1))
		# perform feature dimension reduction
		filename_prefix_save_2 = '%s.%d'%(filename_prefix_save,type_id_group)

		# type_id_group_2 = select_config['type_id_group_2']
		# load_mode_2 = type_id_group_2

		# load_mode_2 = select_config['type_group_load_mode']
		# # field_query = ['latent_peak', 'latent_gene', 'latent_peak_motif']
		# field_query = ['latent_peak', 'latent_gene', 'latent_peak_motif','latent_peak_motif_ori']
		# field_query_pre2 = ['latent_peak_tf_link']

		latent_peak = []
		latent_peak_motif,latent_peak_motif_ori = [], []
		latent_peak_tf_link = []
		load_mode_2 = load_mode

		if load_mode_2==0:
			dict_query1 = self.test_query_feature_pre1_2(peak_query_vec=peak_query_vec_pre1,gene_query_vec=motif_query_vec,
															motif_data=motif_data_query1,motif_data_score=motif_data_score_query1,motif_group=motif_group,
															method_type_vec=method_type_vec_dimension,
															peak_read=peak_read,rna_exprs=rna_exprs,rna_exprs_unscaled=rna_exprs_unscaled,
															n_components=n_components,sub_sample=-1,flag_shuffle=False,float_format=float_format,
															input_file_path=input_file_path,save_mode=1,output_file_path=output_file_path,filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot,output_filename='',verbose=verbose,select_config=select_config)

		elif load_mode_2==1:
			input_file_path_query = output_file_path
			annot_str_vec = ['peak_gene','peak_motif','peak_motif_ori']
			field_query_2 = ['df_latent','df_component']
			dict_query1 = dict()

			# field_num = len(field_query)
			query_num = len(annot_str_vec)
			for i2 in range(query_num):
				method_type_dimension = method_type_vec_dimension[i2]
				filename_save_annot_2 = '%s_%s'%(method_type_dimension,n_components)

				annot_str1 = annot_str_vec[i2]
				field_id1 = 'df_latent'
				filename_prefix_save_query = '%s.%s'%(filename_prefix_save_2,field_id1)
				input_filename = '%s/%s.%s.%s.1.txt'%(input_file_path_query,filename_prefix_save_query,annot_str1,filename_save_annot_2)
				df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
				print('df_query: ',df_query.shape,annot_str1)
				print(df_query[0:2])

				if i2==0:
					feature_query_pre1 = df_query.index
					feature_query_1 = feature_query_pre1.difference(motif_query_vec,sort=False)
					feature_query_2 = pd.Index(peak_query_vec_pre1).intersection(feature_query_1,sort=False)
					feature_query_3 = pd.Index(peak_query_vec_pre1).difference(feature_query_1,sort=False)
					print('feature_query_2: ',len(feature_query_2))
					print('feature_query_3: ',len(feature_query_3))

					latent_peak = df_query.loc[peak_query_vec_pre1,:]
					latent_gene = df_query.loc[motif_query_vec,:]
					print('latent_peak: ',latent_peak.shape)
					print(latent_peak[0:2])
					print('latent_gene: ',latent_gene.shape)
					print(latent_gene[0:2])
					dict_query1.update({'latent_peak':latent_peak,'latent_gene':latent_gene})
				else:
					field_id2 = field_query[i2+1]
					dict_query1.update({field_id2:df_query})

		elif load_mode_2==2:
			feature_type_vec_query = ['motif','peak']
			column_id_query = 'peak_id'
			column_value = 'peak_tf_corr'
			column_vec = [column_id3,column_id2]
			flag_binary = 1
			thresh_value = -0.1
			# n_components = 50
			flag_unduplicate = 0
			method_type_vec_dimension = ['SVD']
			df_feature_link_query = df_peak_tf_query1
			input_file_path_query = select_config['file_path_motif_score_2']
			input_filename = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path_query,data_file_type_query)
			
			if os.path.exists(input_filename)==False:
				print('the file does not exist: %s'%(input_filename))
				input_filename = '%s/test_motif_score_normalize_insilico.%s.thresh1.1.txt'%(input_file_path_query,data_file_type_query)

			df_annot = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_annot['motif_id'] = np.asarray(df_annot.index)
			
			print('df_feature_link_query: ',df_feature_link_query.shape)
			print(df_feature_link_query.columns)
			print(df_feature_link_query[0:2])
			
			print('df_annot: ',df_annot.shape)
			print(df_annot.columns)
			print(df_annot[0:2])
			df_annot = df_annot.drop_duplicates(subset=[column_id3,column_id2])
			peak_query_1 = df_feature_link_query[column_id2].unique()
			feature_query_vec = pd.Index(peak_query_vec_pre1).intersection(peak_query_1,sort=False)
			print('peak_query_1, feature_query_vec: ',len(peak_query_1),len(feature_query_vec))
			# df_annot.index = utility_1.test_query_index(df_annot,column_vec=column_vec)
			dict_query1 = self.test_query_feature_pre1_3(df_feature_link=df_feature_link_query,df_annot=df_annot,feature_query_vec=feature_query_vec,column_id_query=column_id_query,
															column_vec=column_vec,column_value=column_value,feature_type_vec=feature_type_vec_query,
															peak_query_vec=feature_query_vec,gene_query_vec=[],motif_data=motif_data,motif_data_score=motif_data_score,motif_group=[],
															method_type_vec=method_type_vec_dimension,peak_read=peak_read,rna_exprs=rna_exprs,rna_exprs_unscaled=rna_exprs_unscaled,
															n_components=n_components,sub_sample=-1,flag_shuffle=False,flag_binary=flag_binary,thresh_value=thresh_value,
															float_format=float_format,flag_unduplicate=flag_unduplicate,input_file_path=input_file_path,
															save_mode=1,output_file_path=output_file_path,filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot,output_filename='',verbose=verbose,select_config=select_config)

		elif load_mode_2==3:
			input_file_path_query = output_file_path
			annot_str_vec = ['peak_tf_link']
			field_query_2 = ['df_latent','df_component']
			dict_query1 = dict()

			method_type_vec_dimension = ['SVD']
			# field_num = len(field_query)
			query_num = len(annot_str_vec)
			for i2 in range(query_num):
				method_type_dimension = method_type_vec_dimension[i2]
				filename_save_annot_2 = '%s_%s'%(method_type_dimension,n_components)

				annot_str1 = annot_str_vec[i2]
				field_id1 = 'df_latent'
				filename_prefix_save_query = '%s.%s'%(filename_prefix_save_2,field_id1)
				input_filename = '%s/%s.%s.%s.1.txt'%(input_file_path_query,filename_prefix_save_query,annot_str1,filename_save_annot_2)
				df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
				print('df_query: ',df_query.shape,annot_str1)
				print(df_query[0:2])

				field_id2 = field_query_pre2[i2]
				dict_query1.update({field_id2:df_query})

		if load_mode_2 in [0,1]:
			list1 = [dict_query1[field_id] for field_id in field_query]
			latent_peak, latent_gene, latent_peak_motif, latent_peak_motif_ori = list1
			latent_tf = latent_gene.loc[motif_query_vec,:]
			print('latent_peak, latent_tf, latent_peak_motif, latent_peak_motif_ori: ',latent_peak.shape,latent_tf.shape,latent_peak_motif.shape,latent_peak_motif_ori.shape)
			print(latent_peak[0:2])
			print(latent_tf[0:2])
			print(latent_peak_motif[0:2])
			print(latent_peak_motif_ori[0:2])

		elif load_mode_2 in [2,3]:
			list1 = [dict_query1[field_id] for field_id in field_query_pre2]
			latent_peak_tf_link = list1[0]
			print('latent_peak_tf_link: ',latent_peak_tf_link.shape)
			print(latent_peak_tf_link[0:2])

	# perform clustering of peak loci or peak loci and TFs based on the low-dimensional embeddings
	def test_query_feature_clustering_1(self,select_config={}):

		flag_query_1 = 1
		if flag_query_1>0:
			file_path1 = self.save_path_1
			run_id = select_config['run_id']
			test_estimator_1 = test_annotation_11_3_2._Base2_train5_2_pre1(file_path=file_path1,run_id=run_id,select_config=select_config)

			flag_cluster_query_1 = 1
			flag_cluster_query_2 = 0
			n_components = 50

			feature_type_query = 'peak'
			dict_feature = []
			feature_type_vec = ['gene','peak']
			method_type_query = method_type_vec_dimension[0]
			field_query_ori = ['latent_mtx','component_mtx','reconstruct_mtx']
			field_query = ['latent_mtx']

			# annot_str_vec = ['latent_peak_tf','latent_peak_motif']
			annot_str_vec = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']

			iter_id = -1
			config_id_load = -1
			overwrite_2 = False

			file_save_path_2 = output_file_path
			# type_id_group_2 = 0
			type_id_group_2 = select_config['type_id_group_2']
			
			output_file_path_2 = '%s/group1_%d'%(file_save_path_2,type_id_group_2+1)
			if os.path.exists(output_file_path_2)==False:
				print('the directory does not exist: %s'%(output_file_path_2))
				os.makedirs(output_file_path_2,exist_ok=True)

			annot_str1 = annot_str_vec[type_id_group_2]
			filename_prefix_save_2 = '%s.feature_group.%s.%d'%(filename_prefix_save,annot_str1,type_id_group)
			filename_save_annot_2 = filename_save_annot
			dict_query_1 = dict()
			field_id1 = 'latent_mtx'
			
			# if type_id_group_2==0:
			# 	# subsample_ratio = 0.75
			# 	latent_peak_query = latent_peak
			# 	print('latent_peak: ',latent_peak.shape,type_id_group_2)
			# 	print(latent_peak[0:2])
			# elif type_id_group_2==1:
			# 	latent_peak_query = latent_peak_motif
			# 	print('latent_peak_motif: ',latent_peak_motif.shape,type_id_group_2)
			# 	print(latent_peak_motif[0:2])
			# else:
			# 	latent_peak_query = latent_peak_motif_ori
			# 	print('latent_peak_motif_ori: ',latent_peak_motif_ori.shape,type_id_group_2)
			# 	print(latent_peak_motif_ori[0:2])

			field_query_2 = ['latent_peak','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
			list_query1 = [latent_peak,latent_peak_motif,latent_peak_motif_ori,latent_peak_tf_link]
			field_id2 = field_query_2[type_id_group_2]
			latent_peak_query = list_query1[type_id_group_2]

			print(field_id2,latent_peak_query.shape,type_id_group_2)
			print(latent_peak_query[0:2])

			subsample_ratio = -1
			# subsample_ratio = 0.1
			if subsample_ratio>0:
				latent_peak_ori = latent_peak_query.copy()
				peak_query_vec = np.asarray(latent_peak_query.index)
				np.random.shuffle(peak_query_vec)
				peak_num_ori = len(peak_query_vec)
				peak_num1 = int(peak_num_ori*subsample_ratio)
				latent_peak_query = latent_peak_ori.loc[peak_query_vec[0:peak_num1],:]
				print('latent_peak_query: ',latent_peak_query.shape)

			if type_id_group_2==0:
				df_latent_1 = pd.concat([latent_peak_query,latent_tf],axis=0,join='outer',ignore_index=False)
			else:
				df_latent_1 = latent_peak_query

			print('df_latent_1: ',df_latent_1.shape)
			dict_query_1.update({field_id1:df_latent_1})

			# method_type_vec_1 = ['MiniBatchKMeans','phenograph']
			method_type_vec_1 = ['phenograph']
			select_config.update({'method_type_vec_group':method_type_vec_1})

			neighbors_vec = [20, 30] # the neighbors in phenograph clustering
			n_clusters_vec = [30, 50, 100] # the number of clusters
			distance_threshold_vec = [20, 50, -1] # the distance in agglomerative clustering
			metric = 'euclidean'
			linkage_type_idvec = [0]
			select_config.update({'neighbors_vec':neighbors_vec,'n_clusters_vec':n_clusters_vec,'distance_threshold_vec':distance_threshold_vec,
									'linkage_type_idvec':linkage_type_idvec})

			self.test_query_association_pre1_group1_1(data=dict_query_1,feature_type_query=feature_type_query,dict_feature=[],feature_type_vec=feature_type_vec,method_type=method_type_query,field_query=field_query,
														peak_read=peak_read,rna_exprs=rna_exprs,
														flag_cluster_query_1=flag_cluster_query_1,flag_cluster_query_2=flag_cluster_query_2,n_components=n_components,iter_id=iter_id,config_id_load=config_id_load,input_file_path=input_file_path,overwrite=overwrite_2,
														save_mode=save_mode,output_file_path=output_file_path_2,output_filename='',filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot_2,verbose=verbose,select_config=select_config)

	## query neighbors of feature
	# query neighbors of peak loci
	def test_query_feature_neighbor_pre1_1(self,data=[],n_neighbors=20,return_distance=True,save_mode=1,verbose=0,select_config={}):

		from sklearn.neighbors import NearestNeighbors
		from scipy.stats import poisson, multinomial

		# Effective genome length for background computaiton
		# eff_genome_length = atac_meta_ad.shape[1] * 5000
		# bin_size = 500
		# eff_genome_length = atac_meta_ad.shape[1] * bin_size

		# Metacell neighbors
		# peak feature neighbors
		# nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)
		nbrs = NearestNeighbors(n_neighbors=n_neighbors,radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2)
		# nbrs.fit(atac_meta_ad.obsm[low_dim_embedding])
		# meta_nbrs = pd.DataFrame(atac_meta_ad.obs_names.values[nbrs.kneighbors(atac_meta_ad.obsm[low_dim_embedding])[1]],
		# 						 index=atac_meta_ad.obs_names)
		# select_config.update({'meta_nbrs_atac':meta_nbrs})
		feature_mtx = data
		nbrs.fit(feature_mtx)
		# sample_id = feature_mtx.index
		query_id_1 = feature_mtx.index
		neighbor_dist, neighbor_id = nbrs.kneighbors(feature_mtx)
		column_vec = ['neighbor%d'%(id1) for id1 in np.arange(n_neighbors)]
		feature_nbrs = pd.DataFrame(index=query_id_1,columns=column_vec,data=query_id_1.values[neighbor_id])
		dist_nbrs = []
		if return_distance>0:
			dist_nbrs = pd.DataFrame(index=query_id_1,columns=column_vec,data=neighbor_dist)

		return feature_nbrs, dist_nbrs

	# load neighbors of feature query
	def test_query_feature_neighbor_load_1(self,dict_feature=[],feature_type_vec=[],n_neighbors=30,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

			data_file_type_query = select_config['data_file_type']
			# n_neighbors = 30
			# n_neighbors = 50
			if 'neighbor_num' in select_config:
				n_neighbors = select_config['neighbor_num']
			
			n_neighbors_query = n_neighbors+1
			flag_neighbor_query = 1
			feature_type_vec_query = feature_type_vec
			
			if flag_neighbor_query>0:
				query_num = 2
				# load_mode_pre2 = 0
				load_mode_pre2 = 1
				filename_annot1 = '%d.%s'%(n_neighbors,data_file_type_query)
				if load_mode_pre2>0:
					list1 = []
					list2 = []
					for i2 in range(query_num):
						# list_query2 = list_query1[i2]
						feature_type_query = feature_type_vec_query[i2]
						# feature_nbrs_query, dist_nbrs_query = list_query2[0:2]
						# input_filename_1 = '%s/test_feature_nbrs_%d.%s.txt'%(input_file_path,i2+1,data_file_type_query)
						input_filename_1 = '%s/test_feature_nbrs_%d.%s.txt'%(input_file_path,i2+1,filename_annot1)
						if os.path.exists(input_filename_1)==True:
							feature_nbrs_query = pd.read_csv(input_filename_1,index_col=0,sep='\t')
							print('feature_nbrs_query: ',feature_nbrs_query.shape,feature_type_query)
							# print(output_filename_1)
							print(input_filename_1)
							list1.append(feature_nbrs_query)
						else:
							load_mode_pre2 = 0

						# input_filename_2 = '%s/test_dist_nbrs_%d.%s.txt'%(input_file_path,i2+1,data_file_type_query)
						input_filename_2 = '%s/test_dist_nbrs_%d.%s.txt'%(input_file_path,i2+1,filename_annot1)
						if os.path.exists(input_filename_2)==True:
							dist_nbrs_query = pd.read_csv(input_filename_2,index_col=0,sep='\t')
							print('dist_nbrs_query: ',dist_nbrs_query.shape,feature_type_query)
							# print(output_filename_1)
							print(input_filename_2)
							list2.append(dist_nbrs_query)
						else:
							load_mode_pre2 = 0

				if load_mode_pre2>0:
					feature_nbrs_1, feature_nbrs_2 = list1[0:2]
					dist_nbrs_1, dist_nbrs_2 = list2[0:2]
					list_query1 = [[feature_nbrs_1,dist_nbrs_1],[feature_nbrs_2,dist_nbrs_2]]
				else:
					feature_type_num = len(feature_type_vec_query)
					# feature_type_query_1, feature_type_query_2 = feature_type_vec_query[0:2]
					list_query1 = []
					for i1 in range(feature_type_num):
						feature_type_query = feature_type_vec_query[i1]
						if feature_type_query in ['latent_peak_tf']:
							feature_type_query = 'latent_peak_gene'

						print('find nearest neighbors of peak loci')
						start = time.time()
						# df_feature_1 = dict_feature[feature_type_query_1]
						df_feature_1 = dict_feature[feature_type_query]
						print('df_feature_1: ',df_feature_1.shape,feature_type_query)
						print(df_feature_1.shape)
						# subsample_num = 1000
						subsample_num = -1
						if subsample_num>0:
							df_feature_1 = df_feature_1[0:subsample_num]
							print('df_feature_1 sub_sample: ',df_feature_1.shape)
						feature_nbrs_1, dist_nbrs_1 = self.test_query_feature_neighbor_pre1_1(data=df_feature_1,n_neighbors=n_neighbors_query,return_distance=True,save_mode=1,verbose=0,select_config=select_config)

						stop = time.time()
						print('find nearest neighbors of peak loci using feature %s used %.2fs'%(feature_type_query,stop-start))

						list_query1.append([feature_nbrs_1, dist_nbrs_1])

					# start = time.time()
					# df_feature_2 = dict_feature[feature_type_query_2]
					# print('df_feature_2: ',df_feature_2.shape)
					# print(df_feature_2.shape)
					# if subsample_num>0:
					# 	df_feature_2 = df_feature_2[0:subsample_num]
					# 	print('df_feature_2 sub_sample: ',df_feature_2.shape)
					# feature_nbrs_2, dist_nbrs_2 = self.test_query_feature_neighbor_pre1_1(data=df_feature_2,n_neighbors=n_neighbors_query,return_distance=True,save_mode=1,verbose=0,select_config=select_config)

					# stop = time.time()
					# print('find nearest neighbors of peak loci using feature %s used %.2fs'%(feature_type_query_2,stop-start))

					# list_query1 = [[feature_nbrs_1,dist_nbrs_1],[feature_nbrs_2,dist_nbrs_2]]
					
					query_num = len(list_query1)
					for i2 in range(query_num):
						list_query2 = list_query1[i2]
						feature_type_query = feature_type_vec_query[i2]
						feature_nbrs_query, dist_nbrs_query = list_query2[0:2]
						# output_filename_1 = '%s/test_feature_nbrs_%d.%s.txt'%(output_file_path,i2+1,data_file_type_query)
						output_filename_1 = '%s/test_feature_nbrs_%d.%s.txt'%(output_file_path,i2+1,filename_annot1)
						feature_nbrs_query = feature_nbrs_query.round(7)
						feature_nbrs_query.to_csv(output_filename_1,sep='\t')
						print('feature_nbrs_query: ',feature_nbrs_query.shape,feature_type_query)
						print(output_filename_1)

						# output_filename_2 = '%s/test_dist_nbrs_%d.%s.txt'%(output_file_path,i2+1,data_file_type_query)
						output_filename_2 = '%s/test_dist_nbrs_%d.%s.txt'%(output_file_path,i2+1,filename_annot1)
						dist_nbrs_query = dist_nbrs_query.round(7)
						dist_nbrs_query.to_csv(output_filename_2,sep='\t')
						print('feature_nbrs_query: ',dist_nbrs_query.shape,feature_type_query)
						print(output_filename_2)

				return list_query1

	# load feature group estimation for peak or peak and TF
	# input: the method_type_group annotation
	# df_group_1, df_group_2: the group assignment of peak_loc_ori
	# df_overlap_compare: the overlap between the pairs of groups of peak_loc_ori
	# dict_group_basic_1: the group size of the group of each feature type
	def test_query_feature_group_load_1(self,data=[],feature_type_vec=[],feature_query_vec=[],method_type_group='',input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',output_filename='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			method_type_vec_group = [method_type_group]
			type_id_group = 0
			# thresh_size_1 = 20
			# thresh_size_1 = 100
			thresh_size_1 = 0
			# load the feature group query
			# dict_query1_1: (feature_type,df_group), (feature_type_gruop,df_group_statistics); dict_query1_2: (feature_type,peak_loci)
			dict_query1_1, dict_query1_2 = self.test_query_compare_binding_pre1_5_2(data=[],peak_query_vec=[],motif_id_query='',motif_id='',feature_type_vec=feature_type_vec,method_type_vec=method_type_vec_group,peak_read=[],rna_exprs=[],
																						thresh_size=thresh_size_1,type_id_group=type_id_group,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=verbose,select_config=select_config)

			# feature_type_vec_query = ['latent_peak_motif','latent_peak_tf']
			# feature_type_vec_query = [feature_type_query_1,feature_type_query_2]
			feature_type_vec_query = feature_type_vec
			feature_type_query_1,feature_type_query_2 = feature_type_vec_query[0:2]
			filename_save_annot2_ori = filename_save_annot
			filename_save_annot2 = '%s.%s_%s'%(filename_save_annot2_ori,feature_type_query_1,feature_type_query_2)
			# filename_save_annot2_2 = '%s.%s_%s'%(method_type_group,feature_type_query_1,feature_type_query_2)

			df_group_1_ori = dict_query1_1[feature_type_query_1]
			df_group_2 = dict_query1_1[feature_type_query_2]

			if len(feature_query_vec)==0:
				peak_loc_ori = df_group_1.index
			else:
				peak_loc_ori = feature_query_vec

			# peak_loc_ori = peak_read.columns
			df_group_1 = df_group_1_ori.loc[peak_loc_ori,:]
			df_group_2 = df_group_2.loc[peak_loc_ori,:]

			# df_group_query1_ori = dict_query1_2[feature_type_query_1]
			# df_group_query2_ori = dict_query1_2[feature_type_query_2]

			print('df_group_1, df_group_2: ',df_group_1.shape,df_group_2.shape)
			print(df_group_1[0:2])
			print(df_group_2[0:2])

			data_file_type_query = select_config['data_file_type']
			filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
			input_filename = '%s/test_query_df_overlap.%s.1.txt'%(input_file_path,filename_save_annot2_2_pre1)
			# input_filename = '%s/test_query_df_overlap.%s.1.txt'%(input_file_path,data_file_type_query)
			if os.path.exists(input_filename)==True:
				df_overlap_compare = pd.read_csv(input_filename,index_col=0,sep='\t')
				print('df_overlap_compare ',df_overlap_compare.shape)
				print(df_overlap_compare[0:5])
				print(input_filename)			
			else:
				# print('df_group_query1_ori, df_group_query2_ori: ',df_group_query1_ori.shape,df_group_query2_ori.shape)
				# print(df_group_query1_ori[0:2])
				# print(df_group_query2_ori[0:2])
				print('the file does not exist: %s'%(input_filename))
				# query the overlap between the pairs of groups
				df_overlap_pre1 = self.test_query_group_overlap_1(df1=df_group_1,df2=df_group_2,feature_query_vec=[],save_mode=1,select_config=select_config)
				column_1, column_2 = 'group1','group2'

				idvec = [column_1]
				df_overlap_pre1[column_1] = np.asarray(df_overlap_pre1.index)
				df_overlap_compare = df_overlap_pre1.melt(id_vars=idvec,var_name=column_2,value_name='overlap')
				df_overlap_compare.index = utility_1.test_query_index(df_overlap_compare,column_vec=[column_1,column_2],symbol_vec=['_'])
				df_overlap_compare['freq_obs'] = df_overlap_compare['overlap']/np.sum(df_overlap_compare['overlap'])
				# group_1 = np.asarray(df_overlap_compare['group1'])
				# df_overlap_compare['group1_count'] = df_group_query1_ori.loc[group_1,'count']
				# group_2 = np.asarray(df_overlap_compare['group2'])
				# df_overlap_compare['group2_count'] = df_group_query2_ori.loc[group_2,'count']

				# output_filename = '%s/test_query_df_overlap.%s.1.txt'%(output_file_path,data_file_type_query)
				output_filename = '%s/test_query_df_overlap.%s.1.txt'%(output_file_path,filename_save_annot2_2_pre1)
				df_overlap_compare.to_csv(output_filename,sep='\t')

			# column_vec_2 = ['overlap','freq_obs','freq_expect']
			group_vec_query = ['group1','group2']
			# list_group_1 = []
			dict_group_basic_1 = dict()
			column_vec = ['overlap','freq_obs']
			for group_type in group_vec_query:
				df_group_basic_pre1 = df_overlap_compare.groupby(by=[group_type])
				df_group_freq_pre1 = df_group_basic_pre1[column_vec].sum()
				df_group_freq_pre1 = df_group_freq_pre1.rename(columns={'overlap':'count'})

				# list_group_1.append([df_group_basic_pre1,df_group_freq_pre1])
				df_group_freq_pre1['group_type'] = group_type
				dict_group_basic_1.update({group_type:df_group_freq_pre1})

			return df_overlap_compare, dict_group_basic_1, df_group_1, df_group_2

	## group frequency query
	def test_group_frequency_query(self,feature_id,group_id,count_query=0,verbose=0,select_config={}):

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

	## query the overlap between groups
	def test_query_group_overlap_1(self,df1=[],df2=[],feature_query_vec=[],column_query='',query_mode=0,parallel=0,save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			if column_query=='':
				column_query = 'group'
			
			group_vec_ori_1 = np.unique(df1[column_query])
			group_vec_ori_2 = np.unique(df2[column_query])

			if len(feature_query_vec)>0:
				# use selected peak loci
				feature_query_1 = pd.Index(feature_query_vec).intersection(df1.index,sort=False)
				feature_query_2 = pd.Index(feature_query_vec).intersection(df2.index,sort=False)
				df_1_ori = df1
				df_2_ori = df2

				df1 = df1.loc[feature_query_1,:]
				df2 = df2.loc[feature_query_2,:]

				group_vec_1 = np.unique(df1[column_query])
				group_vec_2 = np.unique(df2[column_query])
			else:
				group_vec_1 = group_vec_ori_1
				group_vec_2 = group_vec_ori_2

			feature_query_1 = df1.index
			feature_query_2 = df2.index
			df_overlap = pd.DataFrame(index=group_vec_ori_1,columns=group_vec_ori_2,data=0)

			dict_group_annot_1 = dict()
			dict_group_annot_2 = dict()

			for group_id1 in group_vec_1:
				# peak_vec_query1 = dict_1[feature_type_1][group_id1]
				id_query1 = (df1[column_query]==group_id1)
				# feature_vec_query1 = feature_query_1[df1[column_query]==group_id1]
				feature_vec_query1 = feature_query_1[id_query1]
				# dict_group_annot_1.update({group_id1:id_query1})
				dict_group_annot_1.update({group_id1:feature_vec_query1})

			for group_id2 in group_vec_2:
				id_query2 = (df2[column_query]==group_id2)
				feature_vec_query2 = feature_query_2[id_query2]
				dict_group_annot_2.update({group_id2:feature_vec_query2})

			self.dict_group_annot_1 = dict_group_annot_1
			self.dict_group_annot_2 = dict_group_annot_2

			if 'parallel_overlap' in select_config:
				parallel = select_config['parallel_overlap']

			if parallel==0:
				for group_id1 in group_vec_1:
					# peak_vec_query1 = dict_1[feature_type_1][group_id1]
					# id_query1 = (df1[column_query]==group_id1)
					# feature_vec_query1 = feature_query_1[df1[column_query]==group_id1]
					# feature_vec_query1 = feature_query_1[id_query1]
					# dict_group_annot_1.update({group_id1:id_query1})
					# dict_group_annot_1.update({group_id1:feature_vec_query1})
					feature_vec_query1 = dict_group_annot_1[group_id1]

					for group_id2 in group_vec_2:
						# peak_vec_query2 = dict_1[feature_type_2][group_id2]
						# peak_vec_overlap = pd.Index(peak_vec_query1).intersection(peak_vec_query2,sort=False)
						# feature_vec_query2 = feature_query_2[df2[column_query]==group_id2]
						# id_query2 = dict_group_annot_2[group_id2]
						# feature_vec_query2 = feature_query_2[id_query2]
						feature_vec_query2 = dict_group_annot_2[group_id2]

						feature_vec_overlap = pd.Index(feature_vec_query1).intersection(feature_vec_query2,sort=False)
						# df_overlap.loc[group_id1,group_id2] = len(peak_vec_overlap)
						# feature_num_overlap = np.sum(id_query1&id_query2)
						df_overlap.loc[group_id1,group_id2] = len(feature_vec_overlap)
			else:
				# x = 1
				import itertools
				from itertools import permutations
				list1 = []
				for group_id1 in list1:
					list1.append([group_id1, group_id2] for group_id2 in group_vec_2)
	

			self.df_overlap = df_overlap
			self.dict_group_annot_1 = dict_group_annot_1
			self.dict_group_annot_2 = dict_group_annot_2

			if query_mode>0:
				return df_overlap, dict_group_annot_1, dict_group_annot_2
			else:
				return df_overlap

	# query the overlap between groups
	def test_query_group_overlap_unit1(self,dict_group_annot_1=[],dict_group_annot_2=[],group_id1=-1,group_id2=-1,save_mode=0,verbose=0,select_config={}):

		feature_vec_query1 = dict_group_annot_1[group_id1]
		feature_vec_query2 = dict_group_annot_2[group_id2]

		feature_vec_overlap = pd.Index(feature_vec_query1).intersection(feature_vec_query2,sort=False)
		# df_overlap.loc[group_id1,group_id2] = len(peak_vec_overlap)
		# feature_num_overlap = np.sum(id_query1&id_query2)
		feature_num_overlap = len(feature_vec_overlap)
		# self.df_overlap.loc[group_id1,group_id2] = len(feature_vec_overlap)
		self.df_overlap.loc[group_id1,group_id2] = feature_num_overlap

		return [group_id1,group_id2,feature_num_overlap]

	## query feature enrichment
	# df1: the foreground dataframe
	# df2: the background dataframe (expected dataframe)
	# column_query: the column of value
	def test_query_enrichment_pre1(self,df1,df2,column_query='',stat_chi2_correction=True,stat_fisher_alternative='greater',save_mode=1,verbose=0,select_config={}):
		
		query_idvec = df1.index
		query_num1 = len(query_idvec)
		df_query1 = df1
		df_query_compare = df2

		column_vec_1 = ['stat_chi2_','pval_chi2_']
		column_vec_2 = ['stat_fisher_exact_','pval_fisher_exact_']

		count1 = np.sum(df1[column_query])
		count2 = np.sum(df2[column_query])
		for i1 in range(query_num1):
			query_id1 = query_idvec[i1]
			num1 = df_query1.loc[query_id1,column_query]
			num2 = df_query_compare.loc[query_id1,column_query]

			if num2>0:
				contingency_table = [[num1,count1-num1],[num2,count2-num2]]
				if verbose>0:
					print('contingency table: \n')
					print(contingency_table)

				if not (stat_chi2_correction is None):
					correction = stat_chi2_correction
					stat_chi2_, pval_chi2_, dof_chi2_, ex_chi2_ = chi2_contingency(contingency_table,correction=correction)
					df_query1.loc[query_id1,column_vec_1] = [stat_chi2_, pval_chi2_]

				if not (stat_fisher_alternative is None):
					alternative = stat_fisher_alternative
					stat_fisher_exact_, pval_fisher_exact_ = fisher_exact(contingency_table,alternative=alternative)
					df_query1.loc[query_id1,column_vec_2] = [stat_fisher_exact_, pval_fisher_exact_]

		return df_query1, contingency_table

	# ## query feature enrichment
	# # query the enrichment of predicted peak loci in one type of group
	# def test_query_enrichment_group_1_unit1(self,data=[],dict_group=[],dict_thresh=[],group_type_vec=['group1','group2'],column_vec_query=[],flag_enrichment=1,flag_size=0,type_id_1=1,type_id_2=0,save_mode=0,verbose=0,select_config={}):

	# 	flag_query1 = 1
	# 	if flag_query1>0:
	# 		df_overlap_query = data
	# 		df_query_1 = data  # the number and percentage of feature query in each group 
	# 		dict_group_basic = dict_group # the group annotation of feature query

	# 		thresh_overlap_default_1 = 0
	# 		thresh_overlap_default_2 = 0
	# 		thresh_overlap = 0
	# 		# thresh_pval_1 = 0.20
	# 		thresh_pval_1 = 0.25

	# 		column_1 = 'thresh_overlap_default_1'
	# 		column_2 = 'thresh_overlap_default_2'
	# 		column_3 = 'thresh_overlap'
	# 		column_pval = 'thresh_pval_1'

	# 		flag_1 = 1
	# 		if flag_1>0:
	# 			## feature type 1: motif feature
	# 			# group_type_1, group_type_2 = group_type_vec[0:2]
	# 			group_type_1 = group_type_vec[0]
	# 			id1 = (df_query_1['group_type']==group_type_1)
	# 			df_query1_1 = df_query_1.loc[id1,:]
	# 			# query the enrichment of predicted peak loci in paired groups
	# 			df_query_group1_1, dict_query_group1_1 = self.test_query_enrichment_group_2_unit1(data=df_query1_1,dict_group=dict_group,dict_thresh=dict_thresh,column_vec_query=column_vec_query,flag_enrichment=flag_enrichment,flag_size=flag_size,type_id_1=type_id_1,type_id_2=type_id_2,save_mode=save_mode,verbose=verbose,select_config=select_config)
				
	# 			list1 = [df_query_group1_1, dict_query_group1_1]
	# 			dict_query_1 = {group_type_1:list1}

	# 			# df_query_group2_1 = []
	# 			# dict_query_group2_1 = dict()
	# 			if len(group_type_vec)>1:
	# 				group_type_2 = group_type_vec[1]
	# 				id2 = (df_query_1['group_type']==group_type_2)
	# 				df_query1_2 = df_query_1.loc[id2,:]
	# 				# query the enrichment of predicted peak loci in paired groups
	# 				df_query_group2_1, dict_query_group2_1 = self.test_query_enrichment_group_2_unit1(data=df_query1_2,dict_group=dict_group,dict_thresh=dict_thresh,column_vec_query=column_vec_query,flag_enrichment=flag_enrichment,flag_size=flag_size,type_id_1=type_id_1,type_id_2=type_id_2,save_mode=save_mode,verbose=verbose,select_config=select_config)

	# 				list2 = [df_query_group2_1,dict_query_group2_1]
	# 				dict_query_1.update({group_type_2:list2})

	# 		return dict_query_1

	# ## query feature enrichment
	# # query the enrichment of predicted peak loci in paired groups
	# def test_query_enrichment_group_2_unit1(self,data=[],dict_group=[],dict_thresh=[],group_type_vec=['group1','group2'],column_vec_query=['overlap','pval_fisher_exact_'],flag_enrichment=1,flag_size=0,type_id_1=1,type_id_2=0,save_mode=0,verbose=0,select_config={}):

	# 	flag_query1 = 1
	# 	if flag_query1>0:
	# 		df_overlap_query = data
	# 		df_query_1 = df_overlap_query # the overlaping between groups
	# 		# dict_group_basic = dict_group # the group annotation of feature query

	# 		thresh_overlap_default_1 = 0
	# 		thresh_overlap_default_2 = 0
	# 		thresh_overlap = 0
	# 		# thresh_pval_1 = 0.20
	# 		thresh_pval_1 = 0.25

	# 		column_1 = 'thresh_overlap_default_1'
	# 		column_2 = 'thresh_overlap_default_2'
	# 		column_3 = 'thresh_overlap'
	# 		column_pval = 'thresh_pval_1'
	# 		column_query1, column_query2 = column_vec_query[0:2]
	# 		# id1 = (df_overlap_query['overlap']>thresh_value_overlap)
	# 		# id2 = (df_overlap_query['pval_chi2_']<thresh_pval_1)

	# 		enrichment_query = flag_enrichment
	# 		df_overlap_query_pre1 = df_overlap_query
	# 		dict_query = dict()
	# 		if flag_enrichment>0:
	# 			print('select group based on enrichment')
	# 			if column_1 in dict_thresh:
	# 				thresh_overlap_default_1 = dict_thresh[column_1]

	# 			if column_pval in dict_thresh:
	# 				thresh_pval_1 = dict_thresh[column_pval]

	# 			flag1=0
	# 			try:
	# 				id1 = (df_query_1[column_query1]>thresh_overlap_default_1)
	# 			except Exception as error:
	# 				print('error! ',error)
	# 				flag1=1

	# 			flag2=0
	# 			try:
	# 				id2 = (df_query_1[column_query2]<thresh_pval_1)
	# 			except Exception as error:
	# 				print('error! ',error)
	# 				try: 
	# 					column_query2_1 = 'pval_chi2_'
	# 					id2 = (df_query_1[column_query2_1]<thresh_pval_1)
	# 				except Exception as error:
	# 					print('error! ',error)
	# 					flag2=1

	# 			id_1 = []
	# 			if (flag2==0):
	# 				if (flag1==0):
	# 					id_1 = (id1&id2)
	# 				else:
	# 					id_1 = id2
	# 			else:
	# 				if (flag1==0):
	# 					id_1 = id1

	# 			if (flag1+flag2<2):
	# 				df_overlap_query1 = df_query_1.loc[id_1,:]
	# 				print('the original overlap, the overlap with enrichment above threshold')
	# 				print('df_overlap_query, df_overlap_query1: ',df_query_1.shape,df_overlap_query1.shape)
	# 			else:
	# 				df_overlap_query1 = []
	# 				print('df_overlap_query, df_overlap_query1: ',df_query_1.shape,len(df_overlap_query1))

	# 		df_query_2 = df_query_1.loc[df_query_1[column_query1]>0]
	# 		print('the original overlap, the overlap with number above zero')
	# 		print('df_query_1, df_query_2: ',df_query_1.shape,df_query_2.shape)

	# 		query_value_1 = df_query_1[column_query1]
	# 		query_value_2 = df_query_2[column_query1]
	# 		quantile_vec_1 = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
	# 		query_vec_1 = ['max','min','mean','median']+['percentile_%.2f'%(percentile) for percentile in quantile_vec_1]
	# 		t_value_1 = utility_1.test_stat_1(query_value_1,quantile_vec=quantile_vec_1)
	# 		t_value_2 = utility_1.test_stat_1(query_value_2,quantile_vec=quantile_vec_1)
	# 		query_value = np.asarray([t_value_1,t_value_2]).T
	# 		df_quantile_1 = pd.DataFrame(index=query_vec_1,columns=['value_1','value_2'],data=query_value)
	# 		dict_query.update({'group_size_query':df_quantile_1})

	# 		if flag_size>0:
	# 			print('select group based on number of members')
	# 			if column_2 in dict_thresh:
	# 				thresh_overlap_default_2 = dict_thresh[column_2]

	# 			if column_3 in dict_thresh:
	# 				thresh_overlap = dict_thresh[column_3]

	# 			# thresh_quantile_1 = 0.25
	# 			thresh_quantile_1 = -1
	# 			column_pre2 = 'thresh_quantile_overlap'
	# 			if column_pre2 in dict_thresh:
	# 				thresh_quantile_1 = dict_thresh[column_pre2]
	# 				print('thresh_quantile_1: ',thresh_quantile_1)
				
	# 			# df_query_2 = df_query_1.loc[df_query_1[column_query1]>0]
	# 			# print('the original overlap, the overlap with number above zero')
	# 			# print('df_query_1, df_query_2: ',df_query_1.shape,df_query_2.shape)
	# 			if thresh_quantile_1>0:
	# 				query_value = df_query_2[column_query1]
	# 				thresh_size_1 = np.quantile(query_value,thresh_quantile_1)

	# 				if type_id_1>0:
	# 					thresh_size_ori = thresh_size_1
	# 					thresh_size_1 = np.max([thresh_overlap_default_2,thresh_size_1])
	# 			else:
	# 				thresh_size_1 = thresh_overlap

	# 			id_2 = (df_query_1[column_query1]>=thresh_size_1)
	# 			df_overlap_query2 = df_query_1.loc[id_2,:]
	# 			print('the original overlap, the overlap with number above the threshold')
	# 			print('df_overlap_query, df_overlap_query2: ',df_query_1.shape,df_overlap_query1.shape)
	# 			print('thresh_size_1: ',thresh_size_1)

	# 			if enrichment_query>0:
	# 				if type_id_2==0:
	# 					id_pre1 = (id_1&id_2)
	# 				else:
	# 					id_pre1 = (id_1|id_2)
	# 				df_overlap_query_pre1 = df_query_1.loc[id_pre1,:]

	# 				df_overlap_query_pre1.loc[id_1,'enrichment'] = 1
	# 				df_overlap_query_pre1.loc[id_2,'group_size'] = 1
	# 				print('df_overlap_query, df_overlap_query_pre1: ',df_query_1.shape,df_overlap_query_pre1.shape)
	# 			else:
	# 				df_overlap_query_pre1 = df_overlap_query2
	# 		else:
	# 			df_overlap_query_pre1 = df_overlap_query1

	# 		dict_query.update({'enrichment':df_overlap_query1,'group_size':df_overlap_query2})
	# 		return df_overlap_query_pre1, dict_query

	## query the overlap between groups
	# query the overlap between groups
	def test_query_group_overlap_pre1_1(self,df_group_1=[],df_group_2=[],df_overlap_1=[],df_query_compare=[],feature_query_vec=[],column_query='',flag_shuffle=0,stat_chi2_correction=True,stat_fisher_alternative='greater',save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			if column_query=='':
				column_query = 'group'
			
			# group_vec_ori_1 = np.unique(df_1['%s_group'%(feature_type_1)])
			# group_vec_ori_2 = np.unique(df_1['%s_group'%(feature_type_2)])
			group_vec_ori_1 = np.unique(df_group_1[column_query])
			group_vec_ori_2 = np.unique(df_group_2[column_query])
			# query the overlap between groups
			df1 = df_group_1
			df2 = df_group_2
			# query the overlap
			df_overlap = self.test_query_group_overlap_1(df1=df1,df2=df2,feature_query_vec=feature_query_vec,save_mode=1,select_config=select_config)

			column_1 = 'group1'
			column_2 = 'group2'
			df_overlap[column_1] = np.asarray(df_overlap.index)
			idvec = [column_1]
			df_query1 = df_overlap.melt(id_vars=idvec,var_name=column_2,value_name='overlap')
			df_query1.index = utility_1.test_query_index(df_query1,column_vec=[column_1,column_2],symbol_vec=['_'])
			print('df_query1: ',df_query1.shape)
			print(df_query1)
			output_filename = 'test_query.df_query1.txt'
			df_query1.to_csv(output_filename,sep='\t')

			if flag_shuffle>0:
				np.random.seed(0)
				feature_num = len(feature_query_vec)
				feature_query_1 = df1.index
				id1 = np.random.permutation(np.arange(feature_num))
				feature_query_vec_1 = feature_query_1[id1]	# randomly select the same number of peak loci
				df_overlap_2 = self.test_query_group_overlap_1(df1=df1,df2=df2,feature_query_vec=feature_query_vec_1,save_mode=1,select_config=select_config)
				df_query2 = df_overlap_2.melt(id_vars=idvec,var_name=column_2,value_name='overlap')

			# query the overlap for all the peak loci
			if len(df_query_compare)==0:
				if len(df_overlap_1)==0:
					df_overlap_1 = self.test_query_group_overlap_1(df1=df1,df2=df2,feature_query_vec=[],save_mode=1,select_config=select_config)
			
				df_overlap_1[column_1] = np.asarray(df_overlap_1.index)

				# convert wide format to long format
				idvec = [column_1]
				df_query_compare = df_overlap_1.melt(id_vars=idvec,var_name=column_2,value_name='overlap')
			
			df_query_compare.index = utility_1.test_query_index(df_query_compare,column_vec=[column_1,column_2],symbol_vec=['_'])
			query_id_1 = df_query1.index

			print('df_query_compare: ',df_query_compare.shape)
			print(df_query_compare)

			# output_filename = 'test_query.df_query_compare.txt'
			# df_query_compare.to_csv(output_filename,sep='\t')
			df_query1['overlap_ori'] = df_query_compare.loc[query_id_1,'overlap']

			count1 = np.sum(df_query1['overlap'])
			count2 = np.sum(df_query_compare['overlap'])

			# t_vec_1 = scipy.stats.chisquare(f_obs, f_exp=None, ddof=0, axis=0)
			column_query = 'overlap'
			# df_query_compare_1 = df_query_compare # the original dataframe
			# df_query1_ori = df_query1  # the original dataframe

			eps = 1E-12
			t_value_1 = np.sum(df_query1[column_query])
			t_value_1 = np.max([t_value_1,eps])
			df_freq_query = df_query1[column_query]/t_value_1

			t_value_2 = np.sum(df_query_compare[column_query])
			t_value_2 = np.max([t_value_2,eps])
			df_freq_1 = df_query_compare[column_query]/t_value_2

			# column_pre1, column_pre2 = 'stat_chi2_', 'pval_chi2_'
			# df_query1[column_pre1] = stat_chi2_
			# df_query1[column_pre2] = pval_chi2_
			# column_vec_query = ['freq_obs','freq_expect','stat_chi2_','pval_chi2_']
			# list1 = [df_freq_query,df_freq_1,stat_chi2_,pval_chi2_]
			column_vec_query = ['freq_obs','freq_expect']
			list1 = [df_freq_query,df_freq_1]
			for (column_query,query_value) in zip(column_vec_query,list1):
				df_query1[column_query] = query_value

			# t_vec_1 = scipy.stats.chisquare(f_obs=df_freq_query,f_exp=df_freq_1,ddof=0,axis=0)
			# stat_chi2_, pval_chi2_ = t_vec_1[0:2]
			# print('stat_chi2_, pval_chi2_: ',stat_chi2_,pval_chi2_)
			# query_id_1 = df_query_compare.index
			# column_vec_1 = ['stat_chi2_','pval_chi2_']
			# column_vec_2 = ['stat_fisher_exact_','pval_fisher_exact_']
			# column_vec_query = column_vec_1 + column_vec_2

			# # df1 = pd.DataFrame(index=query_id_1,columns=column_vec_query)
			# query_num1 = len(query_id_1)
			# column_1 = 'overlap'
			# for i1 in range(query_num1):
			# 	query_id1 = query_id_1[i1]
			# 	num1 = df_query1.loc[query_id1,column_1]
			# 	num2 = df_query_compare.loc[query_id1,column_1]

			# 	if num2>0:
			# 		contingency_table = [[num1,count1-num1],[num2,count2-num2]]
			# 		stat_chi2_, pval_chi2_, dof_chi2_, ex_chi2_ = chi2_contingency(contingency_table,correction=True)
			# 		stat_fisher_exact_, pval_fisher_exact_ = fisher_exact(contingency_table,alternative='greater')

			# 		df_query1.loc[query_id1,column_vec_1] = [stat_chi2_, pval_chi2_]
			# 		df_query1.loc[query_id1,column_vec_2] = [stat_fisher_exact_, pval_fisher_exact_]

			# test_query_enrichment_pre1(self,df1,df2,column_query='',stat_chi2_correction=True,stat_fisher_alternative='greater',save_mode=1,verbose=0,select_config={}):
			# stat_chi2_correction = True
			# stat_fisher_alternative = 'greater'
			column_query = 'overlap'
			df_query1, contingency_table = self.test_query_enrichment_pre1(df1=df_query1,df2=df_query_compare,column_query=column_query,stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,save_mode=1,verbose=verbose,select_config=select_config)

			return df_query1, contingency_table, df_overlap

	## query the overlap between groups
	def test_query_group_overlap_pre1_2(self,data=[],dict_group_compare=[],df_group_1=[],df_group_2=[],df_overlap_1=[],df_query_compare=[],column_sort=[],flag_sort=1,flag_sort_2=1,flag_group=1,flag_annot=1,stat_chi2_correction=True,stat_fisher_alternative='greater',save_mode=1,output_file_path='',output_filename='',output_filename_2='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_signal_query=1
		flag_query1=1
		if flag_query1>0:
			df_query1 = data
			# peak_loc_query = df_query1.index
			# peak_num= len(peak_loc_query)
			# print('peak_loc_query ',peak_num)

			feature_vec = df_query1.index
			feature_num = len(feature_vec)
			print('feature_vec: ',feature_num)
			column_query1 = 'group'
			# query the overlap between groups for the given peak loci
			feature_query = df_query1.index
			print('df_group_1, df_group_2: ',df_group_1.shape,df_group_2.shape)
			print(df_group_1[0:2])
			print(df_group_2[0:2])
			feature_query_1 = df_group_1.index
			feature_query_2 = df_group_2.index

			feature_vec_1 = pd.Index(feature_vec).intersection(feature_query_1,sort=False)
			feature_vec_2 = pd.Index(feature_vec).intersection(feature_query_2,sort=False)
			print('feature_vec_1, feature_vec_2: ',len(feature_vec_1),len(feature_vec_2))

			df_group_query1 = df_group_1.loc[feature_vec,:]
			df_group_query2 = df_group_2.loc[feature_vec,:]
			print('df_group_query1, df_group_query2: ',df_group_query1.shape,df_group_query2.shape)
			# use the function test_query_enrichment_pre1();
			# compute the enrichment of the paired groups
			df_overlap_query, contingency_table_1, df_overlap_mtx = self.test_query_group_overlap_pre1_1(df_group_1=df_group_query1,df_group_2=df_group_query2,df_overlap_1=df_overlap_1,df_query_compare=df_query_compare,
																											feature_query_vec=feature_vec,column_query=column_query1,flag_shuffle=0,
																											save_mode=1,output_file_path='',output_filename='',verbose=0,select_config=select_config)

			if flag_sort>0:
				if len(column_sort)==0:
					# column_sort = ['freq_obs','pval_chi2_']
					column_sort = ['freq_obs','pval_fisher_exact_']
				# df_overlap_query = df_overlap_query.sort_values(by=['freq_obs','pval_chi2_'],ascending=[False,True])
				df_overlap_query = df_overlap_query.sort_values(by=column_sort,ascending=[False,True])
				print('test_query_group_overlap_pre1_2')
				print('df_overlap_query: ',df_overlap_query.shape)
				print(df_overlap_query.columns)
				print(df_overlap_query[0:5])

				# print('df_overlap_mtx: ',df_overlap_mtx.shape)
				# print(df_overlap_mtx[0:5])

				if (save_mode>0) and (output_filename!=''):
					# output_filename = '%s/test_query_df_overlap.%s.%s.signal.1.txt'%(output_file_path,motif_id1,data_file_type_query)
					df_overlap_query.to_csv(output_filename,sep='\t')
					print('save df_overlap_query: ',output_filename)

			flag_group=1
			dict_group_basic_1 = dict()
			if flag_group>0:
				column_vec_2 = ['overlap','freq_obs','freq_expect']
				group_vec_query = ['group1','group2']
				# the enrichment in group for each feature type
				for group_type in group_vec_query:
					df_group_basic_pre1 = df_overlap_query.groupby(by=[group_type])
					df_group_freq_pre1 = df_group_basic_pre1[column_vec_2].sum()
					df_group_freq_pre1 = df_group_freq_pre1.rename(columns={'overlap':'count'})

					df_group_freq_compare = dict_group_compare[group_type]
					# stat_chi2_correction = True
					# stat_fisher_alternative = 'greater'
					# column_query = 'overlap'
					column_query = 'count'
					df_group_freq_pre1, contingency_table_2 = self.test_query_enrichment_pre1(df1=df_group_freq_pre1,df2=df_group_freq_compare,column_query=column_query,
																								stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,save_mode=1,verbose=verbose,select_config=select_config)

					df_group_freq_pre1['group_type'] = group_type
					dict_group_basic_1.update({group_type:df_group_freq_pre1})

					# add the columns group1_count, group2_count to the overlap dataframe
					if flag_annot>0:
						group_query = df_overlap_query[group_type]
						df_overlap_query['%s_count'%(group_type)] = np.asarray(df_group_freq_pre1.loc[group_query,'count']) # the number of members in each group

				list1 = [dict_group_basic_1[group_type] for group_type in group_vec_query]
				df_query_pre2 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
				if flag_sort_2>0:
					df_query_pre2 = df_query_pre2.sort_values(by=['group_type','stat_fisher_exact_','pval_fisher_exact_'],ascending=[True,False,True])

				if (save_mode>0) and (output_filename_2!=''):
					# output_filename = '%s/test_query_df_overlap.%s.%s.pre1.2.txt' % (output_file_path, motif_id1, data_file_type_query)
					df_query_pre2 = df_query_pre2.round(7)
					df_query_pre2.to_csv(output_filename_2,sep='\t')
					print('save data: ',output_filename_2)

				dict_group_basic_1.update({'combine':df_query_pre2})

			return df_overlap_query, df_overlap_mtx, dict_group_basic_1
	
	## load the feature group query and the peak query
	# load the estimated group label
	def test_query_compare_binding_pre1_5_2(self,data=[],peak_query_vec=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],peak_read=[],rna_exprs=[],thresh_size=20,type_id_group=0,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		# run_id1 = select_config['run_id']
		# thresh_num1 = 5

		root_path_1 = select_config['root_path_1']
		if data_file_type_query in ['CD34_bonemarrow']:
			input_file_path = '%s/data_pre2/data1_2/peak1'%(root_path_1)
		elif data_file_type_query in ['pbmc']:
			input_file_path = '%s/data_pre2/data1_2/peak2'%(root_path_1)
		
		# input_filename = 'CD34_bonemarrow.pre1.feature_group.latent_peak_motif.0.SVD.feature_dimension.1.50_0.0.df_obs.1.txt'
		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]

		if len(method_type_vec)>0:
			method_type_group = method_type_vec[0]
		else:
			# method_type_group = 'MiniBatchKMeans.50'
			method_type_group = 'phenograph.30'

		method_type_dimension = 'SVD'
		# feature_type_query = 'latent_peak_motif'
		# dict_query1 = dict()
		dict_query1 = data
		dict_query2 = dict()

		load_mode = 0
		if len(dict_query1)==0:
			dict_query1 = dict()
			load_mode = 1
		
		for feature_type_query in feature_type_vec:		
			if load_mode>0:
				if feature_type_query in ['latent_peak_motif']:
					input_file_path_query = '%s/group1/group1_2'%(input_file_path)

				elif feature_type_query in ['latent_peak_tf']:
					input_file_path_query = '%s/group1/group1_1'%(input_file_path)

				elif feature_type_query in ['latent_peak_motif_ori']:
					input_file_path_query = '%s/group1/group1_3'%(input_file_path)

				elif feature_type_query in ['latent_peak_tf_link']:
					input_file_path_query = '%s/group1/group1_4'%(input_file_path)

				filename_prefix_1 = '%s.pre1.feature_group.%s.%d.%s'%(data_file_type_query,feature_type_query,type_id_group,method_type_dimension)
				input_filename = '%s/%s.feature_dimension.1.50_0.0.df_obs.1.txt'%(input_file_path_query,filename_prefix_1)

				df_group = pd.read_csv(input_filename,index_col=0,sep='\t')
				df_group[column_id2] = df_group.index.copy()
				df_group['group'] = np.asarray(df_group[method_type_group])
				dict_query1.update({feature_type_query:df_group})
			else:
				df_group = dict_query1[feature_type_query]
				if not (column_id2 in df_group.columns):
					df_group[column_id2] = df_group.index.copy()
				# else:
				# 	df_group.index = np.asarray(df_group[column_id2])

			print('df_group: ',df_group.shape,feature_type_query)
			print(df_group.columns)
			print(df_group[0:2])
				
			if len(peak_query_vec)>0:
				# df_group['group'] = df_group[method_type_group].copy()
				df_group_query = df_group.loc[peak_query_vec,[method_type_group]] # the group of the peak loci with signal and with motif; the peak loci with score query above threshold and with motif
				# df_group_query = df_group.loc[peak_query_vec,'group'] # the group of the peak loci with signal and with motif; the peak loci with score query above threshold and with motif
				print('df_group_query: ',df_group_query.shape)
				print(df_group_query.columns)
				print(df_group_query[0:2])
				
				df_group_query['count'] = 1
				df_group_query1_ori = df_group_query.groupby([method_type_group]).sum()
				# thresh_1 = 10
				# thresh_1 = 20
				# thresh_group_size = 20
				thresh_1 = thresh_size
				df_group_query1 = df_group_query1_ori.loc[df_group_query1_ori['count']>thresh_1] # the groups with number of members above threshold;
				# group_vec = df_group_query[method_type_group].unique()
				group_vec = df_group_query1.index.unique()
				group_num1 = len(group_vec)
				print('group_vec: ',group_num1)
				print(group_vec)
				
				# df_group.index = np.asarray(df_group['group'])
				df_group.index = np.asarray(df_group[method_type_group])
				peak_query_2 = df_group.loc[group_vec,column_id2].unique() # the peaks in the same group
				peak_vec_2 = pd.Index(peak_query_2).difference(peak_query_vec,sort=False)	# the peaks in the same group but not estimated as peaks with binding sites
				df_group.index = np.asarray(df_group[column_id2]) # reset the index

				column_1 = 'label_1'
				# df_query = pd.DataFrame(index=peak_query_2,columns=[column_1],data=0)
				# df_query.loc[peak_query_vec,column_1] = 1
				df_group['group'] = np.asarray(df_group[method_type_group])
				df_group['label_1'] = 0
				df_group.loc[peak_query_vec,'label_1'] = 1
				
				dict_query2.update({feature_type_query:peak_vec_2})
				field_id1 = '%s_group'%(feature_type_query)
				dict_query1.update({feature_type_query:df_group})
				dict_query1.update({field_id1:df_group_query1_ori}) # the number of members in each group;

		return dict_query1, dict_query2

	# load df_latent and df_component; compute reconstructed matrix
	def test_query_compare_binding_pre1_5_3(self,data=[],motif_id_query='',motif_id='',method_type_vec=[],feature_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],reconstruct=1,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		run_id1 = select_config['run_id']
		# thresh_num1 = 5

		root_path_1 = select_config['root_path_1']
		if data_file_type_query in ['CD34_bonemarrow']:
			input_file_path = '%s/data_pre2/data1_2/peak1'%(root_path_1)
		elif data_file_type_query in ['pbmc']:
			input_file_path = '%s/data_pre2/data1_2/peak2'%(root_path_1)

		# input_filename = 'CD34_bonemarrow.pre1.feature_group.latent_peak_motif.0.SVD.feature_dimension.1.50_0.0.df_obs.1.txt'
			
		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]

		# input_filename_1 = 'CD34_bonemarrow.pre1.df_latent.peak_motif.SVD_50.1.txt'
		# input_filename_2 = 'CD34_bonemarrow.pre1.df_component.peak_motif.SVD_50.1.txt'

		input_file_path_query = '%s/group1'%(input_file_path)
		# feature_type_vec = ['peak_motif','peak_motif_ori']
		# filename_prefix_1 = '%s'%(data_file_type_query)
		filename_prefix_1 = filename_prefix_save
		# method_type_dimension = 'SVD'
		n_components_query = n_components
		filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_components_query)
		
		dict_query1 = dict()
		for feature_type_query in feature_type_vec:
			# input_filename_1 = '%s/%s.pre1.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)
			# input_filename_2 = '%s/%s.pre1.df_component.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)
			input_filename_1 = '%s/%s.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)
			input_filename_2 = '%s/%s.df_component.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)

			dict_query1[feature_type_query] = dict()
			df_latent_query = pd.read_csv(input_filename_1,index_col=0,sep='\t')
			df_component_query = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			dict_query1[feature_type_query].update({'df_latent':df_latent_query,'df_component':df_component_query})

			if reconstruct>0:
				reconstruct_mtx = df_latent_query.dot(df_component_query.T)
				# dict_query1.update({feature_type_query:reconstruct_mtx})
				dict_query1[feature_type_query].update({'reconstruct_mtx':reconstruct_mtx})

		return dict_query1

	## compare TF binding prediction
	# perform clustering of peak and TF
	# load low_dimension_embedding
	def test_query_feature_embedding_load_1(self,data=[],feature_query_vec=[],motif_id_query='',motif_id='',feature_type_vec=[],feature_type_vec_group=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,reconstruct=1,peak_read=[],rna_exprs=[],flag_combine=1,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		dict_query_1 = self.test_query_compare_binding_pre1_5_3(data=data,motif_id_query=motif_id_query,motif_id=motif_id,
																	method_type_vec=method_type_vec,feature_type_vec=feature_type_vec,
																	method_type_dimension=method_type_dimension,n_components=n_components,
																	peak_read=peak_read,rna_exprs=rna_exprs,reconstruct=reconstruct,
																	load_mode=load_mode,input_file_path=input_file_path,
																	save_mode=save_mode,output_file_path=output_file_path,output_filename=output_filename,filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,verbose=verbose,select_config=select_config)

		if save_mode>0:
			self.dict_latent_query_1 = dict_query_1
		
		# if len(feature_query_vec_group)==0:
		# 	feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
	
		# type_id_group_2 = select_config['type_id_group_2']
		# feature_type_query_1 = feature_type_vec_group[type_id_group_2]
		# feature_type_query_2 = 'latent_peak_tf'

		# feature_type_vec_query = ['latent_peak_motif','latent_peak_tf']
		# feature_type_vec_query = [feature_type_query_1,feature_type_query_2]
		# feature_type_query_1,feature_type_query_2 = feature_type_vec_query[0:2]
		# filename_save_annot2_2 = 'copy2_2_1.thresh1'
		# filename_save_annot2_2 = 'copy2_2_1'
		# filename_save_annot2_2 = 'copy2_2_1.thresh1.%s'%(model_type_id1)

		flag_query1 = 1
		# type_id_query_2 = select_config['typeid2']
		# flag_query1 = (type_id_query_2 in [0,2])
		# flag_query1 = 1
		if flag_query1>0:
			# file_path_query1 = '%s/vbak2_6_5_0.1_0_0.1_0.1_0.25_0.1'%(file_save_path_1)
			# file_path_query1 = '%s/vbak2_6_7_0.1_0_0.1_0.1_0.25_0.1_0.01'%(file_save_path_1)
			# input_file_path = file_path_query1
			# output_file_path = file_path_query1
			# list_1 = [dict_query_1[feature_type_query]['df_latent'] for feature_type_query in feature_type_vec]
			feature_type_vec_query = ['latent_%s'%(feature_type_query) for feature_type_query in feature_type_vec]
			feature_type_num = len(feature_type_vec)

			query_mode = 0
			if len(feature_query_vec)>0:
				query_mode = 1
			
			list_1 = []
			for i1 in range(feature_type_num):
				feature_type_query = feature_type_vec[i1]
				df_query = dict_query_1[feature_type_query]['df_latent']
				if query_mode>0:
					df_query = df_query.loc[feature_query_vec,:]
				else:
					if i1==0:
						feature_query_1 = df_query.index
					else:
						df_query = df_query.loc[feature_query_1,:]

				column_vec = df_query.columns
				df_query.columns = ['%s.%s'%(column_1,feature_type_query) for column_1 in column_vec]
				print('df_query: ',df_query.shape,feature_type_query,i1)
				list_1.append(df_query)

			dict_query1 = dict(zip(feature_type_vec_query,list_1))

			if (feature_type_num>0) and (flag_combine>0):
				list1 = [dict_query1[feature_type_query] for feature_type_query in feature_type_vec_query[0:2]]
				latent_mtx_combine = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				print('latent_mtx_combine: ',latent_mtx_combine.shape)
				print(latent_mtx_combine[0:2])

				feature_type_query1,feature_type_query2 = feature_type_vec[0:2]
				feature_type_combine = 'latent_%s_%s_combine'%(feature_type_query1,feature_type_query2)
				dict_query1.update({feature_type_combine:latent_mtx_combine})

			return dict_query1

	# query peak loci predicted with binding sites using clustering
	# dict_group: the original group assignment query
	def test_query_binding_clustering_1(self,data1=[],data2=[],dict_group=[],dict_neighbor=[],dict_group_basic_1=[],df_overlap_1=[],df_overlap_compare=[],group_type_vec=['group1','group2'],feature_type_vec=[],group_vec_query=[],input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		# feature group query and feature neighbor query
		df_pre1 = data1
		df_query1 = data2
		# df_overlap_query, df_overlap_query2 = self.test_query_feature_group_neighbor_pre1_1(data=df_query1,dict_group=dict_group,dict_neighbor=dict_neighbor,dict_group_basic_1=dict_group_basic_1,
		# 																						dict_thresh=[],df_overlap_1=df_overlap_1,df_overlap_compare=df_overlap_compare,
		# 																						group_type_vec=group_type_vec,feature_type_vec=[],
		# 																						group_vec_query=[],column_vec_query=[],input_file_path='',
		# 																						save_mode=0,output_file_path=output_file_path,output_filename='',filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,verbose=0,select_config=select_config)

		# feature group query and feature neighbor query
		df_pre1 = self.test_query_feature_group_neighbor_pre1_2(data=df_pre1,dict_group=dict_group,dict_neighbor=dict_neighbor,
																	group_type_vec=group_type_vec,feature_type_vec=feature_type_vec,
																	group_vec_query=group_vec_query,column_vec_query=[],n_neighbors=30,input_file_path='',
																	save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)
			
		# return df_overlap_query, df_overlap_query2, df_pre1
		return df_pre1

	# feature group query and feature neighbor query
	def test_query_feature_group_neighbor_pre1_1(self,data=[],dict_group=[],dict_neighbor=[],dict_group_basic_1=[],dict_thresh=[],df_overlap_1=[],df_overlap_compare=[],group_type_vec=['group1','group2'],feature_type_vec=[],group_vec_query=[],column_vec_query=[],n_neighbors=30,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			filename_save_annot_1 = filename_save_annot
			filename_query = '%s/test_query_df_overlap.%s.pre1.1.txt' % (input_file_path, filename_save_annot_1)
			filename_query_2 = '%s/test_query_df_overlap.%s.pre1.2.txt' % (input_file_path, filename_save_annot_1)
			
			input_filename = filename_query
			input_filename_2 = filename_query_2
			load_mode_2 = 0

			if os.path.exists(input_filename)==True:
				# overlap between the paired groups and the enrichment statistical significance value
				df_overlap_query = pd.read_csv(input_filename,index_col=0,sep='\t')
				load_mode_2 = load_mode_2+1
				print('df_overlap_query: ',df_overlap_query.shape)
				print(input_filename)

			if os.path.exists(input_filename_2)==True:
				# group size for each feature type
				df_group_basic_query = pd.read_csv(input_filename_2,index_col=0,sep='\t')
				load_mode_2 = load_mode_2+1
				print('df_group_basic_query: ',df_group_basic_query.shape)
				print(input_filename_2)

			df_query1 = data
			# dict_group_basic_1 = dict_group
			# df_group_1 = dict_group['group1']
			# df_group_2 = dict_group['group2']
			if len(group_type_vec)==0:
				group_type_vec = ['group1','group2']

			list_query1 = [dict_group[group_type_query] for group_type_query in group_type_vec]
			df_group_1, df_group_2 = list_query1[0:2] # group annation of feature query in sequence feature space and peak accessibility feature space

			if load_mode_2<2:
				stat_chi2_correction = True
				stat_fisher_alternative = 'greater'
				# dict_group_basic_2: the enrichment of group assignment for each feature type
				df_overlap_query, df_overlap_mtx, dict_group_basic_2 = self.test_query_group_overlap_pre1_2(data=df_query1,dict_group_compare=dict_group_basic_1,df_group_1=df_group_1,df_group_2=df_group_2,
																												df_overlap_1=df_overlap_1,df_query_compare=df_overlap_compare,flag_sort=1,flag_group=1,
																												stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,
																												save_mode=0,output_filename='',verbose=verbose,select_config=select_config)

				# list_query1 = [dict_group_basic_2[group_type] for group_type in group_vec_query]
				list_query1 = []
				key_vec = list(dict_group_basic_2.keys())
				print('dict_group_basic_2: ',key_vec)

				# if len(group_type_vec)==0:
				# 	# group_type_vec = key_vec
				# 	group_type_vec = ['group1','group2']

				for group_type in group_type_vec:
					df_query = dict_group_basic_2[group_type]
					print('df_query: ',len(df_query),group_type)
					print(df_query[0:2])
					df_query['group_type'] = group_type
					list_query1.append(df_query)

				df_query = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
				flag_sort_2=1
				if flag_sort_2>0:
					df_query = df_query.sort_values(by=['group_type','stat_fisher_exact_','pval_fisher_exact_'],ascending=[True,False,True])
				
				# output_filename = '%s/test_query_df_overlap.%s.%s.pre1.2.txt' % (output_file_path, motif_id1, data_file_type_query)
				# output_filename = '%s/test_query_df_overlap.%s.pre1.2.txt' % (output_file_path,filename_save_annot2_2)
				output_filename = '%s/test_query_df_overlap.%s.pre1.2.txt' % (output_file_path,filename_save_annot_1)
				df_query = df_query.round(7)
				df_query.to_csv(output_filename,sep='\t')
				print(output_filename)

			# TODO: automatically adjust the group size threshold
			# df_overlap_query = df_overlap_query.sort_values(by=['freq_obs','pval_chi2_'],ascending=[False,True])
			if len(dict_thresh)==0:
				thresh_value_overlap = 0
				thresh_pval_1 = 0.20
				field_id1 = 'overlap'
				field_id2 = 'pval_fisher_exact_'
				# field_id2 = 'pval_chi2_'
			else:
				column_1 = 'thresh_overlap'
				column_2 = 'thresh_pval_overlap'
				column_3 = 'field_value'
				column_5 = 'field_pval'
				column_vec_query1 = [column_1,column_2,column_3,column_5]
				list_query1 = [dict_thresh[column_query] for column_query in column_vec_query1]
				thresh_value_overlap, thresh_pval_1, field_id1, field_id2 = list_query1
			
			# id1 = (df_overlap_query['overlap']>thresh_value_overlap)
			# id2 = (df_overlap_query['pval_chi2_']<thresh_pval_1)
			id1 = (df_overlap_query[field_id1]>thresh_value_overlap)
			id2 = (df_overlap_query[field_id2]<thresh_pval_1)

			df_overlap_query2 = df_overlap_query.loc[id1,:]
			print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape)

			return df_overlap_query, df_overlap_query2

	# feature group query and feature neighbor query
	# TF binding prediction by feature group query and feature neighbor query
	def test_query_feature_group_neighbor_pre1_2(self,data=[],dict_group=[],dict_neighbor=[],group_type_vec=['group1','group2'],feature_type_vec=[],group_vec_query=[],column_vec_query=[],n_neighbors=30,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		df_pre1 = data
		# df_group_1 = dict_group[group_type_1]
		# df_group_2 = dict_group[group_type_2]
		list_query1 = [dict_group[group_type_query] for group_type_query in group_type_vec]
		df_group_1, df_group_2 = list_query1[0:2] # group annation of feature query in sequence feature space and peak accessibility feature space

		flag_neighbor = 1
		flag_neighbor_2 = 1 	# query neighbor of selected peak in the paired groups
		# flag_neighbor_2 = 0 	# query neighbor of selected peak in the paired groups
		column_neighbor = ['neighbor%d'%(id1) for id1 in np.arange(1,n_neighbors+1)]

		if len(feature_type_vec)==0:
			feature_type_vec = ['latent_peak_motif','latent_peak_tf']
		feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]

		method_type_feature_link = select_config['method_type_feature_link']
		column_1 = '%s_group_neighbor'%(feature_type_query_1)
		column_2 = '%s_group_neighbor'%(feature_type_query_2)

		column_query1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peak
		column_query2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peak

		if len(column_vec_query)==0:	
			# column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
			column_pred2 = '%s.pred'%(method_type_feature_link) # selected peak loci with predicted binding sites
			column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)

			column_pred_3 = '%s.pred_group_neighbor'%(method_type_feature_link)
			column_pred_5 = '%s.pred_group_neighbor_1'%(method_type_feature_link)
			column_pred_6 = '%s.pred_neighbor_2_group'%(method_type_feature_link)
			column_pred_7 = '%s.pred_neighbor_2'%(method_type_feature_link)
			column_pred_8 = '%s.pred_neighbor_1'%(method_type_feature_link)
		else:
			column_pred2, column_pred_2, column_pred_3, column_pred_5, column_pred_6, column_pred_7, column_pred_8 = column_vec_query[0:7]

		id_query1 = (df_pre1[column_pred2]>0)
		df_pre1.loc[id_query1,column_vec_query] = 1
		peak_loc_ori = df_pre1.index

		if flag_neighbor>0:
			# column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
			# column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)
			field_id1 = 'feature_nbrs'
			list_query2 = [dict_neighbor[feature_type_query][field_id1] for feature_type_query in feature_type_vec]
			feature_nbrs_1, feature_nbrs_2 = list_query2[0:2]

			id2 = (df_pre1[column_pred2]>0)
			column_annot1 = 'group_id2'
			if not (column_annot1) in df_pre1:
				# column_vec_2 = ['latent_peak_motif_group','latent_peak_tf_group']
				column_vec_2 = ['%s_group'%(feature_type_query) for feature_type_query in feature_type_vec]
				df_pre1[column_annot1] = utility_1.test_query_index(df_pre1,column_vec=column_vec_2,symbol_vec='_')

			column_id2 = 'peak_id'
			
			for (group_id_1,group_id_2) in group_vec_query:
				start = time.time()
				group_id2 = '%s_%s'%(group_id_1,group_id_2)
				# id1 = (df_group_1['group']==group_id_1)&(df_group_2['group']==group_id_2)
				id1 = (df_pre1[column_annot1]==group_id2)
				
				peak_query_pre1 = peak_loc_ori[id1]
				peak_query_pre1_1 = peak_loc_ori[(id1&id2)]
				peak_query_pre1_2 = peak_loc_ori[id1&(~id2)]
				# df_pre1.loc[peak_query_pre1_2,column_pred_2] = 1
				df_pre1.loc[peak_query_pre1,column_pred_2] = 1

				# stop = time.time()
				# print('group 1, group 2: ',group_id_1,group_id_2,stop-start)

				# df_pre1.index = np.asarray(df_pre1[column_annot1])
				# df_query1 = df_pre1.loc[group_id2,:]
				# peak_query_pre1 = np.asarray(df_query1[column_id2])
				# peak_query_pre1 = pd.Index(peak_query_pre1)
				
				# df_query1.index = peak_query_pre1
				# id_pred = (df_query1[column_pred2]>0)
				# peak_query_pre1_1 = peak_query_pre1[id_pred]
				# peak_query_pre1_2 = peak_query_pre1[~id_pred]

				# df_pre1.index = np.asarray(df_pre1[column_id2])
				# df_pre1.loc[peak_query_pre1_2,column_pred_2] = 1
				# df_pre1.loc[peak_query_pre1,column_pred_2] = 1

				flag_neighbor_pre2 = 1
				if flag_neighbor_pre2>0:
					# start = time.time()
					# start_1 = time.time()
					peak_neighbor_1 = np.ravel(feature_nbrs_1.loc[peak_query_pre1_1,column_neighbor]) # 0.25s
					peak_neighbor_1 = pd.Index(peak_neighbor_1).unique()
					peak_query_1 = pd.Index(peak_neighbor_1).intersection(peak_query_pre1_2,sort=False)

					peak_neighbor_2 = np.ravel(feature_nbrs_2.loc[peak_query_pre1_1,column_neighbor])
					peak_neighbor_2 = pd.Index(peak_neighbor_2).unique()
					peak_query_2 = pd.Index(peak_neighbor_2).intersection(peak_query_pre1_2,sort=False)
						
					# stop_1 = time.time()
					# print('query neighbor of peak loci ',stop_1-start_1)

					# column_query1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peak
					# column_query2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peak
					df_pre1.loc[peak_neighbor_1,column_query1] = 1
					df_pre1.loc[peak_neighbor_2,column_query2] = 1

					# column_1 = '%s_group_neighbor'%(feature_type_query_1)
					# column_2 = '%s_group_neighbor'%(feature_type_query_2)
					df_pre1.loc[peak_query_1,column_1] = 1
					df_pre1.loc[peak_query_2,column_2] = 1

					# stop = time.time()
					# print('query neighbors of peak loci within paired group',group_id_1,group_id_2,stop-start)

					# start = time.time()
					# column_pred_3 = '%s.pred_group_neighbor'%(method_type_feature_link)
					peak_query_vec_2 = pd.Index(peak_query_1).intersection(peak_query_2,sort=False)
					# peak_query_vec_2 = peak_query_pre1_2[query_value==2]
					# peak_query_vec_3 = peak_query_pre1_2[query_value>0]
					df_pre1.loc[peak_query_vec_2,column_pred_3] = 1

					# column_pred_5 = '%s.pred_group_neighbor_1'%(method_type_feature_link)
					peak_query_vec_3 = pd.Index(peak_query_1).union(peak_query_2,sort=False)
					df_pre1.loc[peak_query_vec_3,column_pred_5] = 1

					# print(df_pre1.loc[:,[column_pred_2,column_1,column_2]])
					# peaks within the same groups with the selected peak in the two feature space and peaks are neighbors of the selected peak
					# df_pre1[column_pred_5] = ((df_pre1[column_pred_2]>0)&((df_pre1[column_1]>0)|(df_pre1[column_2]>0))).astype(int)

					# stop = time.time()
					# print('query neighbors of peak loci 1',group_id_1,group_id_2,stop-start)

					if flag_neighbor_2>0:
						peak_neighbor_pre2 = pd.Index(peak_neighbor_1).intersection(peak_neighbor_2,sort=False)
						df_pre1.loc[peak_neighbor_pre2,column_pred_6] = 1

				stop = time.time()
				if (group_id_1%10==0) and (group_id_2%10==0):
					print('query neighbors of peak loci',group_id_1,group_id_2,stop-start)

			# df_pre1[column_pred_5] = ((df_pre1[column_pred_2]>0)&((df_pre1[column_1]>0)|(df_pre1[column_2]>0))).astype(int)
				
		flag_neighbor_3 = 1  # query neighbor of selected peak
		if (flag_neighbor_3>0):
			# column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
			# column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)
			# column_query1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peak
			# column_query2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peak
			id2 = (df_pre1[column_pred2]>0)
			peak_query_pre1_1 = peak_loc_ori[id2] # selected peak loci
					
			peak_neighbor_1 = np.ravel(feature_nbrs_1.loc[peak_query_pre1_1,column_neighbor])
			peak_neighbor_1 = pd.Index(peak_neighbor_1).unique()
			peak_neighbor_num1 = len(peak_neighbor_1)

			peak_neighbor_2 = np.ravel(feature_nbrs_2.loc[peak_query_pre1_1,column_neighbor])
			peak_neighbor_2 = pd.Index(peak_neighbor_2).unique()
			peak_neighbor_num2 = len(peak_neighbor_2)

			df_pre1.loc[peak_neighbor_1,column_query1] = 1
			df_pre1.loc[peak_neighbor_2,column_query2] = 1
			# df_pre1['neighbor_num'] = 0

			df_pre1.loc[:,[column_query1,column_query2,column_pred_7,column_pred_8]] = 0

			peak_num1 = len(peak_query_pre1_1)
			print('peak_query_pre1_1, peak_neighbor_1, peak_neighbor_2: ',peak_num1,peak_neighbor_num1,peak_neighbor_num2)
			for i2 in range(peak_num1):
				peak_query = peak_query_pre1_1[i2]
				peak_neighbor_query1 = np.ravel(feature_nbrs_1.loc[peak_query,column_neighbor])
				peak_neighbor_query2 = np.ravel(feature_nbrs_2.loc[peak_query,column_neighbor])
						
				peak_neighbor_pre2_1 = pd.Index(peak_neighbor_query1).intersection(peak_neighbor_query2,sort=False)
				# peak_neighbor_pre2_2 = pd.Index(peak_neighbor_query1).union(peak_neighbor_query2,sort=False)
				if i2%1000==0:
					peak_neighbor_num_1 = len(peak_neighbor_pre2_1)
					# peak_neighbor_num_2 = len(peak_neighbor_pre2_2)
					print('peak_neighbor_pre2_1: ',peak_neighbor_num_1,i2,peak_query)
					# print('peak_neighbor_pre2_2: ',peak_neighbor_num,i2,peak_query)
					# print('peak_neighbor_pre2_1, peak_neighbor_pre2_2: ',peak_neighbor_num_1,peak_neighbor_num_2,i2,peak_query)
						
				df_pre1.loc[peak_neighbor_query1,column_query1] += 1
				df_pre1.loc[peak_neighbor_query2,column_query2] += 1

				df_pre1.loc[peak_neighbor_pre2_1,column_pred_7] += 1
				# df_pre1.loc[peak_neighbor_pre2_2,column_pred_8] += 1

			df_pre1[column_pred_8] = df_pre1[column_query1]+df_pre1[column_query2]-df_pre1[column_pred_7]

		return df_pre1

	## query estimated feature link
	def test_peak_tf_gene_query_1(self,method_type_vec=[],select_config={}):
		# x = 1
		flag_query1=1
		method_type_num = len(method_type_vec)
		dict_feature = dict()
		data_file_type = select_config['data_file_type']
		dict_method_type = select_config['dict_method_type']
		dict_feature_pre1 = dict()
		dict_feature = dict()

		field_id = 'filename_feature_link'
		dict_annot_1 = []
		if field_id in select_config:
			dict_annot_1= select_config['filename_feature_link']

		for i1 in range(method_type_num):
			# method_type = method_type_vec[method_type_id]
			# method_type_id = method_type_vec[i1]
			# method_type = dict_method_type[method_type_id]
			# data_path = select_config['input_file_path_query'][method_type]
			# input_file_path = data_path
			method_type = method_type_vec[i1]
			input_file_path = select_config['input_file_path_query'][method_type]
			df_gene_peak_query = []
			# if method_type_id in [0,1]:
				# data_path = select_config['data_path']
				# data_path = select_config['input_file_path_query'][method_type_id]
				# input_file_path = data_path
				# input_filename1 = '%s/test_motif_data.1.h5ad'%(input_file_path)
				# input_filename2 = '%s/test_motif_data_score.1.h5ad'%(input_file_path)
				# input_filename_list1 = [input_filename1,input_filename2]
				# input_filename_list2 = []
				# input_file_path2 = '%s/peak_local'%(data_path)

			file_path_motif_score = '%s/peak_local/seacell_1/motif_score_thresh1'%(input_file_path)
			# if method_type in ['insilico','insilico_1']:
			if (method_type.find('insilico')>-1):
				# # input_file_path = '%s/peak_local'%(data_path)
				# # input_file_path2 = '%s/peak_local'%(input_file_path)
				# input_file_path2 = file_path_motif_score
				# input_filename = '%s/test_motif_score_normalize_insilico.%s.thresh0.1.txt'%(input_file_path2,data_file_type)
				
				# if method_type in dict_annot_1:
				# 	input_filename = dict_annot_1[method_type]

				# df_peak_tf_query = pd.read_csv(input_filename,index_col=0,sep='\t')
				# df_peak_tf_query['motif_id'] = np.asarray(df_peak_tf_query.index)
				# # dict_feature[method_type_id] = df_gene_peak_query
				# # print('df_gene_peak_query ',df_gene_peak_query.shape)
				# # df_query = df_gene_peak_query
				# print('df_peak_tf_query ',df_peak_tf_query.shape,method_type)
				print('please provide anontation file ',method_type)

			elif (method_type.find('joint_score')>-1):
				# input_file_path2 = '%s/peak_local'%(input_file_path)
				# input_file_path2 = file_path_motif_score
				# method_type_idvec = [1]
				# # method_type_vec = ['insilico_1','joint_score.thresh1']
				# method_type_vec_query = ['joint_score']
				# if len(dict_feature_pre1)==0:
				# 	thresh_num_query = 5
				# 	thresh_query_vec = np.arange(thresh_num_query)
				# 	dict_feature_pre1, dict_query1_pre1, id_thresh_vec = self.test_peak_tf_gene_query_load_2(method_type_vec=method_type_vec_query,
				# 																								thresh_query_vec=thresh_query_vec,
				# 																								input_file_path=input_file_path2,
				# 																								select_config=select_config)
				# thresh_query = method_type.split('.')[1]
				# thresh_id = int(thresh_query[6:])
				# df_gene_peak_query = dict_query1_pre1[thresh_id-1]
				# # dict_feature.update({method_type_id:df_gene_peak_query})
				# print('df_gene_peak_query ',df_gene_peak_query.shape,method_type)
				print('please provide anontation file ',method_type)

			elif method_type in ['Pando']:
				if data_file_type in ['pbmc']:
					start_id_list = [[0,2500,1],[2500,5000,1],[0,18000,0]]
				else:
					start_id_list = []
				select_config_query = select_config['config_query'][method_type]
				# gene-peak-tf association query for Pando
				df_gene_peak_query = self.test_peak_tf_gene_query_pre1(gene_query_vec=[],
																		method_type=method_type,
																		input_file_path=input_file_path,
																		start_id_list=start_id_list,
																		select_config=select_config_query)
				print('df_gene_peak_query ',df_gene_peak_query.shape,method_type)

			elif method_type in ['TRIPOD']:
				start_id_list = [[0,15500,0]]
				select_config_query = select_config['config_query'][method_type]
				thresh_fdr_ori = 0.01
				# type_id_query = 0
				# type_id_query = 1
				# gene-peak-tf association query for TRIPOD
				if 'thresh_fdr_1' in select_config:
					thresh_fdr_ori = select_config['thresh_fdr_ori']
				df_gene_peak_query = self.test_peak_tf_gene_query_pre2(gene_query_vec=[],
																		method_type=method_type,
																		input_file_path=input_file_path,
																		start_id_list=start_id_list,
																		thresh_fdr=thresh_fdr_ori,
																		select_config=select_config_query)
				print('df_gene_peak_query ',df_gene_peak_query.shape,method_type)

			elif method_type in ['GRaNIE']:
				# run_id = 111
				# metacell_num = 100
				# peak_distance_thresh = 250
				# # input_file_path_pre1 = data_path
				# # select_config.update({'input_file_path_pre1':input_file_path_pre1})
				# input_file_path2 = '%s/%s/metacell_%d/run%d/group%d_%d'%(input_file_path,data_file_type,metacell_num,run_id,(2-normalize_type),peak_distance_thresh)
				# correlation_type_vec = ['pearsonr','spearmanr']
				# correlation_typeid = 0
				# correlation_type = correlation_type_vec[correlation_typeid]
				# thresh_fdr = 0.3
				# normalize_type = 0
				# filename_prefix = '%s.%s'%(correlation_type,data_file_type)
				# # input_filename2 = '%s/test_peak_tf_query_1.pearsonr.%s.thresh0.3.annot1.normalize0.1.txt'%(input_file_path2)
				# input_filename = '%s/test_peak_tf_query_1.%s.thresh%s.annot1.normalize%d.1.txt'%(input_file_path2,filename_prefix,thresh_fdr,normalize_type)
				# input_filename_annot = '%s/TFBS/translationTable.csv'%(input_file_path)
				select_config_query = select_config['config_query'][method_type]
				# df_peak_tf_query = self.test_peak_tf_query_pre2(gene_query_vec=[],
				# 											method_type=method_type,
				# 											input_file_path=input_file_path,
				# 											select_config=select_config_query)

				# gene-peak-tf association query for GRaNIE
				df_gene_peak_query = self.test_peak_tf_gene_query_pre3(gene_query_vec=[],
																		method_type=method_type,
																		input_file_path=input_file_path,
																		start_id_list=[],
																		thresh_fdr=0.01,
																		select_config=select_config_query)

				# rint('df_peak_tf_query ',df_peak_tf_query.shape,method_type)
				print('df_gene_peak_query ',df_gene_peak_query.shape,method_type)
			else:
				print('please provide method_type_id ')

			# if ~(method_type in ['insilico','insilico_1','GRaNIE']):
			if ~(method_type in ['insilico','insilico_1']):
				if ('gene_id' in df_gene_peak_query):
					df_query = df_gene_peak_query.copy()
					df_query.index = test_query_index(df_query,column_vec=['peak_id','motif_id'])
					df_peak_tf_query = df_query.loc[~df_query.index.duplicated(keep='first'),:]
					print('df_peak_tf_query ',df_peak_tf_query.shape,method_type)
					print(df_peak_tf_query[0:5])
				# else:
				# 	df_peak_tf_query = df_gene_peak_query
			else:
				df_peak_tf_query.index = test_query_index(df_peak_ptf_query,column_vec=['peak_id','motif_id'])
				df_peak_tf_query = df_peak_tf_query.loc[~df_peak_tf_query.index.duplicated(keep='first'),:]
				print('df_peak_tf_query ',df_peak_tf_query.shape,method_type)
				# print(df_peak_tf_query[0:5])

			dict1 = {'peak_tf_gene':df_gene_peak_query,'peak_tf':df_peak_tf_query}
			dict_feature.update({method_type:dict1})

		return dict_feature

	## gene-peak-tf association query
	# gene-peak-tf association query for Pando
	def test_peak_tf_gene_query_pre1(self,gene_query_vec=[],method_type='',input_file_path='',start_id_list=[],padj_thresh=0.05,flag_sort=1,select_config={}):

			# columns = ['gene_id','peak_id','distance','spearmanr','pval1','pearsonr','pval2','label','label_corr','label_2']
			columns = ['tf','target','region','term','estimate','std_err','statistic','pval','padj','corr']
			# input_file_path = '../Pando'
			# data_file_type = select_config['data_file_type']
			# type_id_feature = 0
			# exclude_exons = 1
			# type_id_region = 1
			padj_thresh_ori = padj_thresh
			field_query = ['data_file_type','type_id_feature','run_id','metacell_num','exclude_exons','type_id_region',
							'upstream','downstream','padj_thresh']
			list1 = [select_config[field] for field in field_query]
			data_file_type, type_id_feature, run_id, metacell_num, exclude_exons, type_id_region, upstream, downstream, padj_thresh = list1
			data_file_type_annot = data_file_type.lower()

			# run_id = select_config['run_id']
			# metacell_num = select_config['metacell_num']
			input_file_path2 = '%s/%s/metacell_%d/run%d/peak_distance_%d_%d/group1'%(input_file_path,data_file_type_annot,metacell_num,run_id,upstream,downstream)
			filename_prefix_1 = '%s.%d.%d.%d.%d'%(data_file_type,type_id_feature,run_id,exclude_exons,type_id_region)
			# start_id1 = 0
			# start_id2 = 2500
			# start_id_list = [[0,2500],[2500,5000],[0,18000]]
			# start_id_list = [[0,5000],[0,18000]]
			query_num1 = len(start_id_list)
			list_query = []
			column_1 = 'padj_thresh_1'
			if column_1 in select_config:
				padj_thresh = select_config[column_1]
			if query_num1>0:
				for i1 in range(query_num1):
					start_id1, start_id2, type_id2 = start_id_list[i1]
					print(start_id1,start_id2)
					# input_filename = '%s/grn_pando.motif_group1.pbmc.0.1.1.1.0_2500_1.df_coef.txt'%(input_file_path)
					# input_filename = '%s/grn_pando.motif_group1.%s.%d_%d_1.df_coef.txt'%(input_file_path,filename_prefix_1,start_id1,start_id2)
					if padj_thresh==0.05:
						input_filename = '%s/grn_pando.motif_group1.%s.%d_%d_%d.df_coef.2.txt'%(input_file_path2,filename_prefix_1,start_id1,start_id2,type_id2)
						print('input_filename:%s, method:%s'%(input_filename,method_type))
						df_query1 = pd.read_csv(input_filename,index_col=False,sep='\t')
					else:
						input_filename = '%s/grn_pando.motif_group1.%s.%d_%d_%d.df_coef.txt'%(input_file_path2,filename_prefix_1,start_id1,start_id2,type_id2)
						print('input_filename:%s, method:%s'%(input_filename,method_type))
						df_query1_ori = pd.read_csv(input_filename,index_col=False,sep='\t')
						df_query1 = df_query1_ori.loc[df_query1_ori['padj']<padj_thresh,:]
					
					# df_query1 = df_query1.rename(columns={'tf':'motif_id','target':'gene_id','region':'peak_id'})
					df_query1 = df_query1.rename(columns={'tf':'motif_id','target':'gene_id','region':'region_id'})
					df_query1.index = np.asarray(df_query1['motif_id'])
					list_query.append(df_query1)
				df_query = pd.concat(list_query,axis=0,join='outer',ignore_index=False)
			else:
				if padj_thresh==0.05:
					input_filename = '%s/grn_pando.motif_group1.%s.df_coef.2.txt'%(input_file_path2,filename_prefix_1)
					print('input_filename:%s, method:%s'%(input_filename,method_type))
					df_query = pd.read_csv(input_filename,index_col=False,sep='\t')
				else:
					input_filename = '%s/grn_pando.motif_group1.%s.df_coef.txt'%(input_file_path2,filename_prefix_1)
					print('input_filename:%s, method:%s'%(input_filename,method_type))
					df_query1_ori = pd.read_csv(input_filename,index_col=False,sep='\t')
					df_query = df_query1_ori.loc[df_query1_ori['padj']<padj_thresh,:]

				df_query = df_query.rename(columns={'tf':'motif_id','target':'gene_id','region':'region_id'})
				df_query.index = np.asarray(df_query['motif_id'])

			if (exclude_exons==True) or (type_id_region>0):
				df_region = select_config['df_region']
				print('df_region ',df_region.shape,method_type)

			region_id_ori = np.asarray(df_query['region_id'])
			t_vec_str = pd.Index(region_id_ori).str.split('-').str
			chrom, start, stop = t_vec_str.get(0), t_vec_str.get(1), t_vec_str.get(2)
			query_num = len(chrom)
			region_id = ['%s:%s-%s'%(chrom[i1],start[i1],stop[i1]) for i1 in range(query_num)]
			df_query['region_id'] = np.asarray(region_id)
			# tf_query = np.asarray(df_query['motif_id'])
			# df_query['term'] = ['%s:%s'%(tf_query[i2],region_id[i2] for i2 in range(query_num))]
			df_query.loc[:,'peak_id'] = np.asarray(df_region.loc[region_id,'peak_loc'])
			print(df_query[0:5])

			df_gene_peak_query = df_query
			output_file_path = input_file_path2
			output_filename = '%s/grn_pando.motif_group1.%s.df_coef.padj%s.txt'%(output_file_path,filename_prefix_1,padj_thresh)
			print('save file: %s, method_type: %s'%(output_filename,method_type))
			df_gene_peak_query.to_csv(output_filename,sep='\t',float_format='%.6E')
			print('df_gene_peak_query ',df_gene_peak_query.shape)
			# df_gene_peak_query['score_pred1'] = -df_gene_peak_query['padj']
			df_gene_peak_query['score_pred1'] = df_gene_peak_query['padj']
			if flag_sort>0:
				# df_gene_peak_query = df_gene_peak_query.sort_values(by=['peak_id','score_pred1'],ascending=[True,False])
				df_gene_peak_query = df_gene_peak_query.sort_values(by=['peak_id','score_pred1'],ascending=[True,True])

			return df_gene_peak_query

	## gene-peak-tf association query
	# gene-peak-tf association query for TRIPOD
	def test_peak_tf_gene_query_pre2(self,gene_query_vec=[],method_type='',input_file_path='',start_id_list=[],thresh_fdr=0.01,flag_sort=1,select_config={}):

			# columns = ['gene_id','peak_id','distance','spearmanr','pval1','pearsonr','pval2','label','label_corr','label_2']
			# columns = ['tf','target','region','term','estimate','std_err','statistic','pval','padj','corr']
			columns = ['gene','peak_num','TF_num','peak','TF','coef','pval','adj','label1','level1.peak','level2.peak','level1.tf','level2.tf','sign','peak_width']
			thresh_fdr_ori = thresh_fdr
			field_query = ['data_file_type','type_id_feature','run_id','metacell_num','upstream','downstream','thresh_fdr']
			list1 = [select_config[field] for field in field_query]

			data_file_type, type_id_feature, run_id, metacell_num, upstream, downstream, thresh_fdr = list1
			data_file_type_annot = data_file_type.lower()

			input_file_path2 = '%s/%s/metacell_%d/run%d/peak_distance_%d_%d'%(input_file_path,data_file_type_annot,metacell_num,run_id,upstream,downstream)
			filename_prefix_1 = 'test_peak_tf_gene_query_1'
			filename_prefix_save = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
			# input_filename = '%s/%s.%s.1.df_coef.txt'%(input_file_path2,filename_prefix_1,filename_prefix_save)
			type_id_query = select_config['type_id_query']
			input_filename = '%s/%s.%s.%d.1.df_coef.txt'%(input_file_path2,filename_prefix_1,filename_prefix_save,type_id_query)
			print('input_filename:%s, method:%s, type_id_query:%d'%(input_filename,method_type,type_id_query))
			df_query = pd.read_csv(input_filename,index_col=False,sep='\t')
			df_query = df_query.rename(columns={'peak':'peak_id','TF':'motif_id','gene':'gene_id'})

			output_file_path = input_file_path2
			if thresh_fdr!=0.01:
				df_query_ori = df_query
				df_query = df_query_ori.loc[df_query_ori['adj']<thresh_fdr,:]
				output_filename = '%s/%s.%s.%d.df_coef.thresh_fdr%s.txt'%(output_file_path,filename_prefix_1,filename_prefix_save,type_id_query,thresh_fdr)
				print('save file: %s, method_type: %s'%(output_filename,method_type))
				df_query.to_csv(output_filename,sep='\t')

			df_gene_peak_query = df_query
			print('df_gene_peak_query ',df_gene_peak_query.shape)
			# df_gene_peak_query['score_pred1'] = -df_gene_peak_query['adj']
			df_gene_peak_query['score_pred1'] = df_gene_peak_query['adj']
			if flag_sort>0:
				# df_gene_peak_query = df_gene_peak_query.sort_values(by=['peak_id','score_pred1'],ascending=[True,False])
				df_gene_peak_query = df_gene_peak_query.sort_values(by=['peak_id','score_pred1'],ascending=[True,True])

			return df_gene_peak_query

	## gene-peak-tf association query for GRaNIE
	def test_peak_tf_gene_query_pre3(self,gene_query_vec=[],method_type='',input_file_path='',start_id_list=[],thresh_fdr=0.01,flag_sort=1,select_config={}):

			# columns = ['gene_id','peak_id','distance','spearmanr','pval1','pearsonr','pval2','label','label_corr','label_2']
			# columns = ['tf','target','region','term','estimate','std_err','statistic','pval','padj','corr']
			# columns = ['gene','peak_num','TF_num','peak','TF','coef','pval','adj','label1','level1.peak','level2.peak','level1.tf','level2.tf','sign','peak_width']
			columns = ['TF.name','TF.ENSEMBL','peak.ID','TF_peak.r_bin','TF_peak.r','TF_peak.fdr','TF_peak.fdr_direction','TF_peak.connectionType',
						'gene.ENSEMBL','gene.name.x','gene.type.x','gene.mean','gene.median','gene.CV','gene.chr','gene.start','gene.end','gene.strand','gene.type.y','gene.name.y',
						'peak_gene.distance','peak_gene.r','peak_gene.p_raw','peak_gene.p_adj','TF_gene.r','TF_gene.p_raw']

			# thresh_fdr_ori = thresh_fdr
			# field_query = ['data_file_type','type_id_feature','run_id','metacell_num','upstream','downstream','thresh_fdr']
			field_query = ['data_file_type','type_id_feature','run_id','metacell_num','peak_distance_thresh','correlation_type','normalize_type',
							'thresh_fdr_peak_tf','thresh_fdr_peak_gene']
			list1 = [select_config[field] for field in field_query]
			
			data_file_type, type_id_feature, run_id, metacell_num, peak_distance_thresh, correlation_type, normalize_type, thresh_fdr_peak_tf, thresh_fdr_peak_gene = list1
			data_file_type_annot = data_file_type.lower()

			input_file_path2 = '%s/%s/metacell_%d/run%d/group%d_%d'%(input_file_path,data_file_type_annot,metacell_num,run_id,(2-normalize_type),peak_distance_thresh)
			# correlation_type = 'pearsonr'
			# input_filename = 'grn1_connection_filtered.pearsonr.CD34_bonemarrow.normalize0'
			filename_prefix_1 = 'grn1_connection_filtered'
			input_filename = '%s/%s.%s.%s.normalize%d.txt'%(input_file_path2,filename_prefix_1,correlation_type,data_file_type,normalize_type)
			
			input_filename_annot = '%s/TFBS/translationTable.csv'%(input_file_path)
			df_annot = pd.read_csv(input_filename_annot,index_col=False,sep=' ')
			# df_annot.index = np.asarray(df_annot['ENSEMBL'])
			df_annot.index = np.asarray(df_annot['HOCOID'])

			# print('input_filename:%s, method:%s, type_id_query:%d'%(input_filename,method_type,type_id_query))
			print('input_filename:%s, method:%s'%(input_filename,method_type))
			df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_query.index = np.asarray(df_query['TF.name'])
			motif_name = np.asarray(df_query['TF.name'])
			df_query.loc[:,'motif_id'] = np.asarray(df_annot.loc[motif_name,'SYMBOL'])
			df_query.index = np.asarray(df_query['motif_id'])

			# df_query = df_query.rename(columns={'peak':'peak_id','TF':'motif_id','gene':'gene_id'})
			df_query = df_query.rename(columns={'peak.ID':'peak_id','gene.name.x':'gene_id'})

			output_file_path = input_file_path2
			column_id1 = 'TF_peak.fdr'
			column_id2 = 'peak_gene.p_adj'
			# if thresh_fdr!=0.01:
			# 	df_query_ori = df_query
			# 	df_query = df_query_ori.loc[df_query_ori[column_id1]<thresh_fdr,:]
			# 	# output_filename = '%s/%s.%s.%d.df_coef.thresh_fdr%s.txt'%(output_file_path,filename_prefix_1,filename_prefix_save,type_id_query,thresh_fdr)
			# 	b = input_filename.find('.txt')
			# 	output_filename = '%s.thresh_fdr_peak_tf%s.txt'%(input_filename[0:b],thresh_fdr)
			# 	print('save file: %s, method_type: %s'%(output_filename,method_type))
			# 	df_query.to_csv(output_filename,sep='\t')

			df_gene_peak_query = df_query
			print('df_gene_peak_query ',df_gene_peak_query.shape)
			# df_gene_peak_query['score_pred1'] = -df_gene_peak_query[column_id1]
			# df_gene_peak_query['score_pred2'] = -df_gene_peak_query[column_id2]
			df_gene_peak_query['score_pred1'] = df_gene_peak_query[column_id1]
			df_gene_peak_query['score_pred2'] = df_gene_peak_query[column_id2]
			if flag_sort>0:
				df_gene_peak_query = df_gene_peak_query.sort_values(by=['peak_id',column_id1,column_id2],ascending=[True,True,True])

			return df_gene_peak_query

	## peak-tf association query
	# peak-tf association query for GRaNIE
	def test_peak_tf_query_pre2(self,gene_query_vec=[],method_type='',df_gene_peak_query=[],input_filename='',input_filename_annot='',input_file_path='',input_file_path_annot='',select_config={}):

			# columns = ['gene_id','peak_id','distance','spearmanr','pval1','pearsonr','pval2','label','label_corr','label_2']
			columns = ['TF.name','TF_peak.r_bin','TF_peak.r','TF_peak.fdr','TF_peak.fdr_orig','peak.ID','TF_peak.fdr_direction','TF_peak.connectionType']
			# input_file_path = '../GRaNIE'
			data_file_type = select_config['data_file_type']
			data_file_type_annot = data_file_type.lower()
			# input_file_path_1 = '/data/peer/yangy4/data1/data_pre2/data1_1/data1_pre2'
			# input_file_path_pre1 = select_config['input_file_path_pre1']

			# run_id = 111
			# metacell_num = 100
			# peak_distance_thresh = 250
			# correlation_type_vec = ['pearsonr','spearmanr']
			# correlation_typeid = 0
			# correlation_type = correlation_type_vec[correlation_typeid]
			# thresh_fdr = 0.3
			# normalize_type = 0

			run_id = select_config['run_id']
			metacell_num = select_config['metacell_num']
			peak_distance_thresh = select_config['peak_distance_thresh']
			# correlation_type_vec = ['pearsonr','spearmanr']
			correlation_type = select_config['correlation_type']
			thresh_fdr_save = select_config['thresh_fdr_save']
			thresh_fdr_peak_tf = select_config['thresh_fdr_peak_tf']
			normalize_type = select_config['normalize_type']
			filename_prefix = '%s.%s'%(correlation_type,data_file_type)

			input_file_path2 = '%s/%s/metacell_%d/run%d/group%d_%d'%(input_file_path,data_file_type_annot,metacell_num,run_id,(2-normalize_type),peak_distance_thresh)
			# input_filename2 = '%s/test_peak_tf_query_1.pearsonr.%s.thresh0.3.annot1.normalize0.1.txt'%(input_file_path2)
			input_filename = '%s/test_peak_tf_query_1.%s.thresh%s.annot1.normalize%d.1.txt'%(input_file_path2,filename_prefix,thresh_fdr_save,normalize_type)
			input_filename_annot = '%s/TFBS/translationTable.csv'%(input_file_path)

			df_annot1 = pd.read_csv(input_filename_annot,index_col=False,sep=' ')
			# df_annot1.loc[:,'motif_id'] = np.asarray(df_annot1.index)
			df_annot1.index = np.asarray(df_annot1['HOCOID'])

			# # input_filename2 = '%s/test_peak_tf_query_1.pearsonr.%s.thresh0.3.annot1.normalize0.1.txt'%(input_file_path2)
			# input_filename2 = '%s/test_peak_tf_query_1.%s.thresh%s.annot1.normalize%d.1.txt'%(input_file_path2,filename_prefix,thresh_fdr,normalize_type)
			if os.path.exists(input_filename)==True:
				print('input_filename:%s, method:%s'%(input_filename,method_type))
				df_query1_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
				df_query1 = df_query1_ori.loc[df_query1_ori['TF_peak.fdr']<thresh_fdr_peak_tf,:]
				print('df_query1_ori, df_query1: ',df_query1_ori.shape,df_query1.shape)
				print(df_query1_ori[0:5])
				print(df_query1[0:5])
				df_query1['motif_name'] = np.asarray(df_query1.index).copy()
				query_id = df_query1.index
				df_query1['motif_id'] = np.asarray(df_annot1.loc[query_id,'SYMBOL'])
				df_query1.index = np.asarray(df_query1['motif_id'])
				df_query1 = df_query1.rename(columns={'peak.ID':'peak_id'})
				# field_query = ['motif_name','motif_id','peak_id']
				# df_peak_tf_query = df_query1.loc[:,field_query]
				df_peak_tf_query = df_query1
				# df_peak_tf_query['score_pred1'] = -df_peak_tf_query['TF_peak.fdr']
				df_peak_tf_query['score_pred1'] = df_peak_tf_query['TF_peak.fdr']
				print('df_peak_tf_query: ',df_peak_tf_query.shape)
				print(df_peak_tf_query[0:5])
			else:
				print('the file does not exist ',input_filename)
				return

			return df_peak_tf_query

	## add score annotation to peak loci
	def test_query_peak_annot_score_1(self,data=[],df_annot=[],method_type='',motif_id='',column_score='',column_name_vec=[],ascending=False,format_type=0,flag_binary=1,flag_sort=0,flag_unduplicate=1,thresh_vec=[],save_mode=1,verbose=0,select_config={}):

		df_pre1 = data
		column_idvec = ['gene_id','peak_id','motif_id']
		column_id1, column_id2, column_id3 = column_idvec[0:3]

		flag1 = df_annot.duplicated(subset=[column_id2,column_id3])
		if np.sum(flag1)>0:
			flag_unduplicate = 1

		if flag_sort>0:
			df_annot = df_annot.sort_values(by=column_score,ascending=ascending)

		if flag_unduplicate>0:
			df_annot = df_annot.drop_duplicates(subset=[column_id2,column_id3])

		df_annot1 = df_annot.loc[df_annot[column_id3]==motif_id,]
		df_annot1.index = np.asarray(df_annot1[column_id2])
		
		peak_loc_1 = df_pre1.index
		query_id_1 = df_annot1[column_id2]
		query_id_2 = pd.Index(query_id_1).intersection(peak_loc_1,sort=False)  # find the intersection of the peak loci
		if verbose>0:
			print('peak_loc_1, query_id_1, query_id_2: ',len(peak_loc_1),len(query_id_1),len(query_id_2),motif_id)

		column_name_1, column_name_2 = column_name_vec[0:2]
		df_pre1.loc[query_id_2,column_name_2] = df_annot1.loc[query_id_2,column_score]

		# add the binary motif scanning results
		# add the binary prediction results
		if flag_binary>0:
			thresh_score_1, thresh_type = thresh_vec[0:2]
			df_query = df_pre1.loc[query_id_2,:]

			column_query = column_name_2
			if thresh_type in [0,1]:
				if thresh_type==0:
					id1 = (df_query[column_query]>thresh_score_1)
				else:
					id1 = (df_query[column_query]<thresh_score_1)
				query_idvec = query_id_2[id1]
			else:
				query_idvec = query_id_2

			df_pre1.loc[query_idvec,column_name_1] = 1
			query_num1 = len(query_idvec)
			print('query_idvec, motif_id: ',query_num1,motif_id)

		return df_pre1

	## add motif annotation to peak loci
	# input: the dataframe, the motif data 
	def test_query_peak_annot_motif_1(self,data=[],df_annot=[],method_type='',motif_id='',column_score='',column_name='',format_type=0,flag_sort=0,save_mode=1,verbose=0,select_config={}):

		df_pre1 = data
		column_idvec = ['gene_id','peak_id','motif_id']
		column_id1, column_id2, column_id3 = column_idvec[0:3]

		peak_loc_1 = df_pre1.index
		query_id_1 = df_annot.index
		query_id_2 = query_id_1.intersection(peak_loc_1,sort=False) # find the intersection of the peak loci
		if column_score=='':
			column_score = motif_id

		if column_score in df_annot.columns:
			df_pre1.loc[query_id_2,column_name] = np.asarray(df_annot.loc[query_id_2,column_score]).astype(int)
		else:
			print('the column not included: %s'%(column_score))
		
		return df_pre1

	## compare TF binding prediction
	# def test_query_compare_binding_pre1_5_1_basic_2_6_pre2_1_annot1()
	def test_query_binding_pred_load_1(self,data=[],method_type_vec=[],save_mode=1,verbose=0,select_config={}):

		# add the scores for the methods
		# x = 1
		# input_file_path_1 = '%s/%s/metacell_%d/run%d'%(input_file_path,data_file_type_annot,metacell_num,run_id)
		
		method_type_vec = ['insilico_1','joint_score.thresh1','joint_score.thresh2','joint_score.thresh3','joint_score.thresh4','joint_score.thresh5','Pando','GRaNIE','TRIPOD']
		method_query_num1 = len(method_type_vec)
		method_type_idvec = np.arange(method_query_num1)
		dict_method_type = dict(zip(method_type_vec,method_type_idvec))
		select_config.update({'dict_method_type':dict_method_type})

		## query the configurations of the methods
		# thresh_fdr_peak_tf_GRaNIE = 0.2
		# select_config.update({'thresh_fdr_peak_tf_GRaNIE':thresh_fdr_peak_tf_GRaNIE})
		select_config = self.test_config_query_1(method_type_vec=method_type_vec,select_config=select_config)

		## query the file path
		select_config = self.test_file_path_query_2(method_type_vec=method_type_vec,select_config=select_config)

		## load motif data
		# data_path = select_config['input_file_path_query'][method_type]
		# dict_file_query = select_config['filename_motif_data'][method_type]
		# dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec,
		# 																select_config=select_config)
		
		# method_type_vec_1 = method_type_vec
		# if len(method_type_vec)==0:	
		# 	method_type_vec_1 = ['TRIPOD','GRaNIE','Pando']
		
		method_type_vec_1 = ['TRIPOD','GRaNIE','Pando']

		## load the region annotation for Pando
		root_path_1 = select_config['root_path_1']
		# input_file_path = '/data/peer/yangy4/data1/data_pre2/data1_2'
		input_file_path = '%s/data_pre2/data1_2'%(root_path_1)
		data_file_type_annot = 'pbmc'
		run_id = 1
		metacell_num = 500
		exclude_exons = True
		type_id_region = 1
		method_type_query = 'Pando'
		input_file_path2 = '%s/%s/%s/metacell_%d/run%d'%(input_file_path,method_type_query,data_file_type_annot,metacell_num,run_id)
		
		input_filename = '%s/test_region.%d.%d.bed'%(input_file_path2,exclude_exons,type_id_region)
		flag_region_query=((exclude_exons==True)|(type_id_region>0))
		if os.path.exists(input_filename)==True:
			df_region = pd.read_csv(input_filename,index_col=False,sep='\t')
			df_region.index = np.asarray(df_region['id'])
			# pre_config.update({'df_region':df_region})
			df_region_ori = df_region.copy()

			df_region = df_region.sort_values(by=['overlap'],ascending=False)
			df_region = df_region.loc[~df_region.index.duplicated(keep='first'),:]
			df_region = df_region.sort_values(by=['region_id'],ascending=True)
			output_file_path = input_file_path2
			output_filename = '%s/test_region.%d.%d.2.bed'%(output_file_path,exclude_exons,type_id_region)
			df_region.to_csv(output_filename,sep='\t')
			method_type = method_type_query
			select_config['config_query'][method_type].update({'df_region':df_region})
		else:
			print('the file does not exist ',input_filename)

		if 'padj_thresh_save' in select_config:
			padj_thresh_save = select_config['padj_thresh_save']
		else:
			# padj_thresh_save = 0.5
			padj_thresh_save = 1.0
		select_config['config_query'][method_type_query].update({'padj_thresh_1':padj_thresh_save})
		dict_feature_query = self.test_peak_tf_gene_query_1(method_type_vec=method_type_vec_1,
																select_config=select_config)
		key_vec_1 = list(dict_feature_query.keys())
		key_vec_1 = np.asarray(key_vec_1)
		print('dict_feature_query: ',key_vec_1)

		## load the peak-TF links of GRaNIE
		# run_id = select_config['run_id']
		method_type_query2 = 'GRaNIE'
		run_id = 111
		metacell_num = 100
		peak_distance_thresh_query = 250
		# correlation_type = 'spearmanr'
		correlation_type = 'pearsonr'
		normalize_type = 0
		thresh_fdr_save = 0.3
		# thresh_fdr_peak_tf = 0.2
		thresh_fdr_peak_tf = 0.3
		field_query_1 = ['run_id','metacell_num','peak_distance_thresh','correlation_type','thresh_fdr_save','thresh_fdr_peak_tf','normalize_type']
		list_query_1 = [run_id,metacell_num,peak_distance_thresh_query,correlation_type,thresh_fdr_save,thresh_fdr_peak_tf,normalize_type]
		query_num1 = len(field_query_1)
		for i1 in range(query_num1):
			field_id = field_query_1[i1]
			query_value = list_query_1[i1]
			print('field_id, query_value: ',field_id,query_value)
			select_config.update({field_id:query_value})

		input_file_path = select_config['input_file_path_query'][method_type_query2]
		df_peak_tf_query1 = self.test_peak_tf_query_pre2(gene_query_vec=[],method_type=method_type_query2,df_gene_peak_query=[],input_filename='',input_filename_annot='',input_file_path=input_file_path,input_file_path_annot='',select_config=select_config)
		dict_feature_query[method_type_query2].update({'peak_tf_2':df_peak_tf_query1})

		return dict_feature_query

	## recompute based on clustering of peak and TF
	# recompute based on training
	def test_query_compare_binding_pre1_5_1_recompute_5_ori(self,data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		run_id1 = select_config['run_id']
		thresh_num1 = 5
		# method_type_vec = ['insilico_1','TRIPOD','GRaNIE','Pando']+['joint_score.thresh%d'%(i1+1) for i1 in range(thresh_num1)]+['joint_score_2.thresh3']
		# method_type_vec = ['insilico_1','GRaNIE','joint_score.thresh1']
		# method_type_vec = ['GRaNIE']
		# method_type_vec = ['insilico_1','joint_score.thresh1','joint_score.thresh2','joint_score.thresh3']
		# method_type_vec = ['insilico_1','GRaNIE']+['joint_score_2.thresh3']
		# method_type_vec = ['insilico_1','GRaNIE','Pando','TRIPOD']+['joint_score_2.thresh3']
		# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre2.thresh3']
		# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre2.thresh3','joint_score_pre1.thresh22','joint_score_pre2.thresh6']
		# method_type_feature_link_1 = 'joint_score_pre2'
		# method_type_feature_link = 'joint_score_pre2.thresh3'
		# method_type_feature_link_1 = 'joint_score_pre1'
		# method_type_feature_link = 'joint_score_pre1.thresh22'
		# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre1.thresh22']
		method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh22']

		if 'method_type_feature_link' in select_config:
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_feature_link_1 = method_type_feature_link.split('.')[0]
		
		filename_save_annot = '1'
		method_type_vec = pd.Index(method_type_vec).union([method_type_feature_link],sort=False)

		flag_config_1=1
		if flag_config_1>0:
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
			# if data_file_type_query in ['CD34_bonemarrow']:
			# 	input_file_path = '%s/peak1'%(root_path_2)
			# elif data_file_type_query in ['pbmc']:
			# 	input_file_path = '%s/peak2'%(root_path_2)

			# file_save_path_1 = input_file_path
			# select_config.update({'file_path_peak_tf':file_save_path_1})
			# # peak_distance_thresh = 100
			# peak_distance_thresh = 500
			# filename_prefix_1 = 'test_motif_query_binding_compare'
			# # file_path_query1 = '%s/vbak2_6'%(input_file_path)
			# # file_path_query1 = '%s/vbak2_6_5_0.1_0_0.1_0.1_0.25_0.1'%(input_file_path)
			# # input_file_path = file_path_query1
			# # output_file_path = file_path_query1
			# method_type_vec_query = method_type_vec

			# # input_file_path_query = '/data/peer/yangy4/data1/data_pre2/cd34_bonemarrow/data_1/run0/'
			# # root_path_1 = select_config['root_path_1']
			# # data_file_type_annot = data_file_type_query.lower()
			# # run_id_1 = 0
			# # input_file_path2 = '%s/data_pre2/%s/peak_local'%(root_path_1,data_file_type_annot)
			# # input_file_path_query1 = '%s/data_pre2/%s/data_1/run%d/peak_local'%(root_path_1,data_file_type_annot,run_id_1)
			# # input_file_path_query = '%s/seacell_1'%(input_file_path_query1)

			# select_config = self.test_config_query_2(method_type_vec=method_type_vec,data_file_type=data_file_type_query,select_config=select_config)
			# data_file_type_annot = select_config['data_file_type_annot']
			# data_path_save_local = select_config['data_path_save_local']
			# input_file_path_query = data_path_save_local

			# ## query the configurations of the methods
			# # thresh_fdr_peak_tf_GRaNIE = 0.2
			# # select_config.update({'thresh_fdr_peak_tf_GRaNIE':thresh_fdr_peak_tf_GRaNIE})
			# select_config = self.test_config_query_1(method_type_vec=method_type_vec,select_config=select_config)

			# ## query the file path
			# select_config = self.test_file_path_query_2(method_type_vec=method_type_vec,select_config=select_config)

			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)
			# file_path_peak_tf = select_config['file_path_peak_tf']
			# file_save_path_1 = select_config['file_path_peak_tf']

		flag_motif_data_load_1 = 1
		if flag_motif_data_load_1>0:
			print('load motif data')
			# method_type_vec_query = ['insilico_1','joint_score_2.thresh3']
			# method_type_vec_query = ['insilico_0.1','joint_score_pre2.thresh3']
			method_type_vec_query = ['insilico_0.1']+[method_type_feature_link]
			method_type_1, method_type_2 = method_type_vec_query[0:2]
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																			select_config=select_config)

			motif_data_query1 = dict_motif_data[method_type_2]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_2]['motif_data_score']
			peak_loc_ori = motif_data_query1.index
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			self.dict_motif_data = dict_motif_data

		flag_load_1 = 1
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			# filename_1 = '%s/test_rna_meta_ad.%s.log_normalize.2.h5ad'%(input_file_path_query,data_file_type_query)
			# filename_2 = '%s/test_atac_meta_ad.%s.log_normalize.2.h5ad'%(input_file_path_query,data_file_type_query)
			# filename_3_ori = '%s/test_query.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
			# # filename_3 = '%s/test_rna_meta_ad.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)

			# select_config.update({'filename_rna':filename_1,'filename_atac':filename_2,
			# 						'filename_rna_exprs_1':filename_3_ori})
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori(meta_exprs=[],peak_read=[],flag_format=False,select_config=select_config)
			# rna_exprs = meta_scaled_exprs
			# rna_exprs = meta_exprs_2
			# sample_id = rna_exprs.index
			sample_id = meta_scaled_exprs.index
			peak_read = peak_read.loc[sample_id,:]
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			rna_exprs = meta_scaled_exprs
			print('peak_read, rna_exprs: ',peak_read.shape,rna_exprs.shape)

			print(peak_read[0:2])
			print(rna_exprs[0:2])
			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs
			peak_loc_ori = peak_read.columns

		# file_save_path_1 = select_config['file_path_peak_tf']
		# if data_file_type_query in ['pbmc']:
		# 	# file_path_query1 = '%s/vbak2_6_7_0.1_0_0.1_0.1_0.25_0.1_0.01_3_2'%(file_save_path_1)
		# 	if folder_id in [1,2]:
		# 		file_path_query1 = '%s/vbak2_6_7_0.1_0_0.1_0.1_0.25_0.1_0.01_3_%d'%(file_save_path_1,group_id_1)
		# 	else:
		# 		file_path_query1 = '%s/vbak2_6_7_0.1_0_0.1_0.1_0.25_0.1_0.01_2'%(file_save_path_1)
		# select_config.update({'file_path_query_1':file_path_query1})

		file_save_path_1 = select_config['file_path_peak_tf']
		if data_file_type_query in ['pbmc']:
			dict_file_annot1 = select_config['dict_file_annot1']
			dict_file_annot2 = select_config['dict_file_annot2']
			dict_config_annot1 = select_config['dict_config_annot1']

		folder_id = select_config['folder_id']
		# group_id_1 = folder_id+1
		file_path_query_1 = dict_file_annot1[folder_id] # the first level directory
		file_path_query_2 = dict_file_annot2[folder_id] # the second level directory including the configurations

		input_file_path = file_path_query_2
		output_file_path = file_path_query_2

		folder_id_query = 2 # the folder to save annotation files
		file_path_query1 = dict_file_annot1[folder_id_query]
		file_path_query2 = dict_file_annot2[folder_id_query]
		input_file_path_query = file_path_query1
		input_file_path_query_2 = '%s/vbak1'%(file_path_query1)

		dict_query_1 = dict()
		feature_type_vec = ['latent_peak_motif','latent_peak_motif_ori','latent_peak_tf','latent_peak_tf_link']
		feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
		type_id_group_2 = select_config['type_id_group_2']
		feature_type_query_1 = feature_type_vec_group[type_id_group_2]
		feature_type_query_2 = 'latent_peak_tf'

		feature_type_vec_query = [feature_type_query_1,feature_type_query_2]
		feature_type_vec_2_ori = []
		prefix_str1 = 'latent_'
		prefix_str1_len = len(prefix_str1)
		for feature_type_query in feature_type_vec_query:
			b = feature_type_query.find(prefix_str1)
			feature_type = feature_type_query[(b+prefix_str1_len):]
			feature_type_vec_2_ori.append(feature_type)

		flag_annot_1 = 1
		# method_type_group = 'MiniBatchKMeans.50'
		method_type_vec_group_ori = ['MiniBatchKMeans.%d'%(n_clusters_query) for n_clusters_query in [30,50,100]]+['phenograph.%d'%(n_neighbors_query) for n_neighbors_query in [10,15,20,30]]
		# method_type_group = 'MiniBatchKMeans.%d'%(n_clusters)
		# method_type_group_id = 1
		# n_neighbors_query = 30
		n_neighbors_query = 20
		method_type_group = 'phenograph.%d'%(n_neighbors_query)
		# method_type_group_id = 6
		if 'method_type_group' in select_config:
			method_type_group = select_config['method_type_group']
		print('method_type_group: ',method_type_group)

		if flag_annot_1>0:
			thresh_size_1 = 100
			if 'thresh_size_group' in select_config:
				thresh_size_group = select_config['thresh_size_group']
				thresh_size_1 = thresh_size_group

			# for selecting the peak loci predicted with TF binding
			# thresh_score_query_1 = 0.125
			# thresh_size_1 = 20
			thresh_score_query_1 = 0.15
			if 'thresh_score_group_1' in select_config:
				thresh_score_group_1 = select_config['thresh_score_group_1']
				thresh_score_query_1 = thresh_score_group_1
			
			thresh_score_default_1 = thresh_score_query_1
			thresh_score_default_2 = 0.10

			peak_distance_thresh_1 = 500
			thresh_fdr_peak_tf_GRaNIE = 0.2
			upstream_tripod = peak_distance_thresh_1
			# type_id_tripod = 0
			# type_id_tripod = 1
			type_id_tripod = select_config['type_id_tripod']
			filename_save_annot_2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod,type_id_tripod)
			# thresh_size_1 = 20

			# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
			# filename_save_annot_2 = filename_save_annot
			filename_save_annot2_ori = '%s.%s.%d.%s'%(filename_save_annot_2,thresh_score_query_1,thresh_size_1,method_type_group)

		flag_group_load_1 = 1
		if flag_group_load_1>0:
			# load feature group estimation for peak or peak and TF
			n_clusters = 50
			peak_loc_ori = peak_read.columns
			feature_query_vec = peak_loc_ori
			# the overlap matrix; the group member number and frequency of each group; the group assignment for the two feature types
			df_overlap_compare, dict_group_basic_1, df_group_1, df_group_2 = self.test_query_feature_group_load_1(data=[],feature_type_vec=feature_type_vec_query,feature_query_vec=feature_query_vec,method_type_group=method_type_group,input_file_path=input_file_path,
																													save_mode=1,output_file_path=output_file_path,filename_prefix_save='',filename_save_annot=filename_save_annot2_ori,output_filename='',verbose=0,select_config=select_config)

			self.df_group_pre1 = df_group_1
			self.df_group_pre2 = df_group_2
			self.dict_group_basic_1 = dict_group_basic_1

		flag_query2 = 1
		if flag_query2>0:
			# select the feature type for group query
			# feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']

			flag_load_2 = 1
			if flag_load_2>0:
				feature_type_1, feature_type_2 = feature_type_vec_2_ori[0:2]
				if feature_type_2 in ['peak_tf']:
					feature_type_vec_2 = [feature_type_1] + ['peak_gene']
				else:
					feature_type_vec_2 = [feature_type_1,feature_type_2]

				method_type_dimension = 'SVD'
				n_components = 50
				type_id_group = select_config['type_id_group']
				filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
				reconstruct = 0
				# load latent matrix;
				# recontruct: 1, load reconstructed matrix;
				flag_combine = 1
				dict_latent_query1 = self.test_query_feature_embedding_load_1(data=[],feature_query_vec=[],motif_id_query='',motif_id='',feature_type_vec=feature_type_vec_2,method_type_vec=[],method_type_dimension=method_type_dimension,
																				n_components=n_components,reconstruct=reconstruct,peak_read=[],rna_exprs=[],flag_combine=flag_combine,
																				load_mode=0,input_file_path='',
																				save_mode=0,output_file_path='',output_filename='',filename_prefix_save=filename_prefix_save_2,filename_save_annot='',
																				verbose=0,select_config=select_config)

				dict_feature = dict_latent_query1

			# n_neighbors = 30
			# n_neighbors = 50
			n_neighbors = 100
			if 'neighbor_num' in select_config:
				n_neighbors = select_config['neighbor_num']
			n_neighbors_query = n_neighbors+1

			# query the neighbors of feature query
			flag_neighbor_query=1
			if flag_neighbor_query>0:
				# list_query1 = [[feature_nbrs_1,dist_nbrs_1],[feature_nbrs_2,dist_nbrs_2]]
				list_query1 = self.test_query_feature_neighbor_load_1(dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,n_neighbors=n_neighbors,input_file_path=input_file_path,
																		save_mode=save_mode,output_file_path=output_file_path,verbose=0,select_config=select_config)

				feature_nbrs_1,dist_nbrs_1 = list_query1[0]
				feature_nbrs_2,dist_nbrs_2 = list_query1[1]

				field_id1, field_id2 = 'feature_nbrs', 'dist_nbrs'
				field_query_1 = [field_id1,field_id2]
				# query_num = len(field_query_1)
				feature_type_num = len(feature_type_vec_query)
				dict_neighbor = dict()
				for i2 in range(feature_type_num):
					feature_type_query = feature_type_vec_query[i2]
					dict_neighbor[feature_type_query] = dict(zip(field_query_1,list_query1[i2]))
				self.dict_neighbor = dict_neighbor
					
			# flag_motif_query_1=1
			flag_motif_query_pre1=0
			if flag_motif_query_pre1>0:
				folder_id = 1
				if 'folder_id' in select_config:
					folder_id = select_config['folder_id']
				df_peak_file, motif_idvec_query = self.test_query_file_annotation_load_1(data_file_type_query=data_file_type_query,folder_id=folder_id,save_mode=1,verbose=verbose,select_config=select_config)

				motif_query_num = len(motif_idvec_query)
				motif_idvec = ['%s.%d'%(motif_id_query,i1) for (motif_id_query,i1) in zip(motif_idvec_query,np.arange(motif_query_num))]
				filename_list1 = np.asarray(df_peak_file['filename'])
				file_num1 = len(filename_list1)
				motif_idvec_2 = []
				for i1 in range(file_num1):
					filename_query = filename_list1[i1]
					b = filename_query.find('.bed')
					motif_id2 = filename_query[0:b]
					motif_idvec_2.append(motif_id2)

				print('motif_idvec_query: ',len(motif_idvec_query),motif_idvec_query[0:5])
			
				# motif_idvec_query = ['ATF2','ATF3','ATF7','BACH1','BACH2','BATF','BATF3']
				# motif_idvec = ['ATF2.0','ATF3.1','ATF7.2','BACH1.3','BACH2.4','BATF.6','BATF3.7']
				# sel_num1 = 12
				sel_num1 = -1
				if sel_num1>0:
					motif_idvec_query = motif_idvec_query[0:sel_num1]
					motif_idvec = motif_idvec[0:sel_num1]
				select_config.update({'motif_idvec_query':motif_idvec_query,'motif_idvec':motif_idvec})

				motif_idvec_query = select_config['motif_idvec_query']
				motif_idvec = select_config['motif_idvec']
				query_num_ori = len(motif_idvec_query)

			input_filename = '%s/test_peak_file.ChIP-seq.pbmc.1_2.2.copy2.txt'%(input_file_path_query_2)
			df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_annot1_1 = df_annot_ori.drop_duplicates(subset=['motif_id2'])
			# folder_id_query1 = select_config['folder_id']
			# id1 = (df_annot1_1['folder_id']==folder_id_query1)
			# df_annot_1 = df_annot1_1.loc[id1,:]
			df_annot_1 = df_annot1_1
			print('df_annot_ori, df_annot_1: ',df_annot_ori.shape,df_annot_1.shape)

			motif_idvec_query = df_annot_1.index.unique()
			motif_idvec_1 = df_annot_1['motif_id1']
			motif_idvec_2 = df_annot_1['motif_id2']
			df_annot_1.index = np.asarray(df_annot_1['motif_id2'])
			dict_motif_query_1 = dict(zip(motif_idvec_2,list(motif_idvec_query)))

			motif_query_num = len(motif_idvec_query)
			motif_num2 = len(motif_idvec_2)
			query_num_ori = len(motif_idvec_2)
			print('motif_idvec_query, motif_idvec_2: ',motif_query_num,motif_num2)

			column_signal = 'signal'
			column_motif = '%s.motif'%(method_type_feature_link)
			column_pred1 = '%s.pred'%(method_type_feature_link)
			column_score_1 = 'score_pred1'
			df_score_annot = []

			method_type_vec_2 = ['TRIPOD','GRaNIE','Pando']
			# column_motif_vec_2 = ['%s.motif'%(method_type_query) for method_type_query in method_type_vec_2]
			# column_motif_vec_3 = ['%s.pred'%(method_type_query) for method_type_query in method_type_vec_2]
			column_motif_vec_2 = []
			for method_type_query in method_type_vec_2:
				column_motif_vec_2.extend(['%s.motif'%(method_type_query),'%s.pred'%(method_type_query)])

			column_idvec = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_idvec[0:3]

			# load_mode = 1
			load_mode = 0
			flag_unduplicate_query = 1
			if load_mode>0:
				# copy the score estimation
				if len(df_score_annot)==0:
					# load_mode_query_2 = 1  # 1, load the peak-TF correlation score; 2, load the estimated score 1; 3, load both the peak-TF correlation score and the estimated score 1;
					load_mode_query_2 = 2  # load the estimated score 1
					df_score_annot_query1, df_score_annot_query2 = self.test_query_score_annot_1(data=[],df_score_annot=[],input_file_path='',load_mode=load_mode_query_2,save_mode=0,verbose=0,select_config=select_config)
					df_score_annot = df_score_annot_query2

				if flag_unduplicate_query>0:
					df_score_annot = df_score_annot.drop_duplicates(subset=[column_id2,column_id3])
				df_score_annot.index = np.asarray(df_score_annot[column_id2])

				column_score_query1 = '%s.%s'%(method_type_feature_link,column_score_1)
				# id1 = (df_score_annot[column_id3]==motif_id_query)
				# df_score_annot_query = df_score_annot.loc[id1,:]
				# df_score_annot_query = df_score_annot_query.drop_duplicates(subset=[column_id2,column_id3])
			else:
				column_score_query1 = '%s.score'%(method_type_feature_link)

			query_num_1 = query_num_ori
			stat_chi2_correction = True
			stat_fisher_alternative = 'greater'
			list_score_query_1 = []
			interval_save = True
			config_id_load = select_config['config_id_load']
			config_id_2 = select_config['config_id_2']
			config_group_annot = select_config['config_group_annot']
			flag_scale_1 = select_config['flag_scale_1']
			type_query_scale = flag_scale_1

			model_type_id1 = 'LogisticRegression'
			# select_config.update({'model_type_id1':model_type_id1})
			if 'model_type_id1' in select_config:
				model_type_id1 = select_config['model_type_id1']

			output_file_path_1 = file_path_query_1
			if model_type_id1 in ['XGBClassifier']:
				output_file_path_query = '%s/train%d_%d_%d'%(output_file_path_1,config_id_2,config_group_annot,type_query_scale)
			else:
				output_file_path_query = '%s/train%d_%d_%d_pre2'%(output_file_path_1,config_id_2,config_group_annot,type_query_scale)

			# output_file_path_query2 = '%s/model_train_1'%(output_file_path_query)
			if os.path.exists(output_file_path_query)==False:
				print('the directory does not exist: %s'%(output_file_path_query))
				os.makedirs(output_file_path_query,exist_ok=True)

			motif_vec_group2_query2 = ['NR2F1','RCOR1','STAG1','TAF1','ZNF24','ZNF597']
			# motif_vec_group2_query_2 = ['VDR']
			motif_vec_group2_query_2 = ['STAT1','IKZF1','RUNX3','MYB','YY1']
			# motif_vec_group2_query_2 = ['MYB','MAX','YY1']
			motif_vec_group2_query_2 = ['EBF1','PAX5','POU2F2','USF2','MAX','ETS1','IRF1','IRF8','GATA3','RUNX1','CEBPA','CEBPB']

			beta_mode = select_config['beta_mode']
			# motif_id_1 = select_config['motif_id_1']
			if beta_mode>0:
				# if motif_id_1!='':
				# 	str_vec_1 = motif_id_1.split(',')
				# 	motif_id1 = str_vec_1[0]
				# 	motif_id2 = str_vec_1[1]
				# 	motif_id_query = motif_id1.split('.')[0]
				# 	motif_idvec_query = [motif_id_query]
				# 	motif_idvec = [motif_id1]
				# 	motif_idvec_2 = [motif_id2]	
				iter_vec_1 = [0]
			else:
				iter_vec_1 = np.arange(query_num_1)

			file_path_query_pre1 =  output_file_path_query

			method_type_feature_link = select_config['method_type_feature_link']
			n_neighbors = select_config['neighbor_num']
			peak_loc_ori = peak_read.columns
			# df_pre1_ori = pd.DataFrame(index=peak_loc_ori)
			method_type_group = select_config['method_type_group']

			dict_motif_data = self.dict_motif_data
			method_type_query = method_type_feature_link
			motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']
			# peak_loc_ori = motif_data_query1.index
			
			motif_data_query1 = motif_data_query1.loc[peak_loc_ori,:]
			motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_ori,:]
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			motif_data = motif_data_query1
			motif_data_score = motif_data_score_query1
			# self.dict_motif_data = dict_motif_data

			list_annot_peak_tf = []
			# iter_vec_1 = [0]
			for i1 in iter_vec_1:
			# for i1 in range(query_num_1):
			# for i1 in [0,1,2]:
			# for i1 in [0]:
				# motif_id_query = motif_idvec_query[i1]
				# motif_id1 = motif_idvec[i1]
				# motif_id2 = motif_idvec_2[i1]
				motif_id2_query = motif_idvec_2[i1]
				motif_id_query = df_annot_1.loc[motif_id2_query,'motif_id']
				motif_id1_query = df_annot_1.loc[motif_id2_query,'motif_id1']
				folder_id_query = df_annot_1.loc[motif_id2_query,'folder_id']
				motif_id1, motif_id2 = motif_id1_query, motif_id2_query
				folder_id = folder_id_query

				input_file_path_query_1 = dict_file_annot1[folder_id_query] # the first level directory
				input_file_path_query_2 = dict_file_annot2[folder_id_query] # the second level directory including the configurations

				# motif_id1 = motif_id
				print('motif_id_query, motif_id: ',motif_id_query,motif_id1,i1)

				if motif_id_query in motif_vec_group2_query2:
					print('the estimation not included: ',motif_id_query,motif_id1,i1)
					continue

				if not (motif_id_query in motif_vec_group2_query_2):
					continue

				overwrite_2 = False
				filename_prefix_save = 'test_query.%s'%(method_type_group)
				# filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
				iter_id1 = 0
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)

				ratio_1,ratio_2 = select_config['ratio_1'],select_config['ratio_2']
				filename_annot2 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
				filename_annot_train_pre1 = filename_annot2
				filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
				filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
				# output_filename = '%s/test_query_train.%s.%s.%s.%s.1.txt'%(output_file_path,method_type_group,filename_annot_train_pre1,motif_id1,filename_save_annot_1)
				# output_filename = '%s/test_query_train.%s.%s.%s.1.txt'%(output_file_path_query,filename_save_annot_query,motif_id1,filename_save_annot_1)
				# filename_query_pre1 = '%s/test_query_train.%s.%s.%s.1.txt'%(file_path_query_pre1,filename_save_annot_query,motif_id1,filename_save_annot_1)
				file_path_query_pre2 = '%s/train1'%(file_path_query_pre1)
				filename_query_pre1 = '%s/test_query_train.%s.%s.%s.1.txt'%(file_path_query_pre2,filename_save_annot_query,motif_id1,filename_save_annot_1)
				
				if (os.path.exists(filename_query_pre1)==True) and (overwrite_2==False):
					print('the file exists: %s'%(filename_query_pre1))
					continue

				flag1=1
				if flag1>0:
					start_1 = time.time()
				# try:
					# load the TF binding prediction file
					# the possible columns: (signal,motif,predicted binding,motif group)
					# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
					# input_filename = '%s/%s.peak_loc.flanking_0.%s.1.0.2.%d.1.copy2.txt'%(input_file_path,filename_prefix_1,motif_id1,peak_distance_thresh_1)
					# input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.copy2.txt'%(input_file_path,filename_prefix_1,motif_id1,filename_save_annot_2)
					# input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.group2.txt'%(input_file_path,filename_prefix_1,motif_id1,filename_save_annot_2)
					# df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
					# peak_loc_1 = df_1.index
					# peak_num_1 = len(peak_loc_1)
					# print('peak_loc_1: ',peak_num_1)
					# df_query1 = df_1
					# input_filename_2 = '%s/%s.peak_loc.flanking_0.%s.1.%s.group2.txt'%(input_file_path,filename_prefix_1,motif_id1,filename_save_annot_2)
					# input_filename_2 = '%s/test_motif_query_binding_compare.CTCF.15.MiniBatchKMeans.50.latent_peak_motif_latent_peak_tf.neighbor50.2.copy2_2.txt'%(input_file_path)
					filename_prefix_1 = 'test_motif_query_binding_compare'
					# input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					
					# flag_group_query_1 = 0
					flag_group_query_1 = 1
					# if (os.path.exists(input_filename_query1)==True):
					# 	df_pre1 = pd.read_csv(input_filename_query1,index_col=0,sep='\t')
					# 	print('df_pre1: ',df_pre1.shape)
					# 	# df_pre1 = df_pre1.drop_duplicates(subset=['peak_id'])
					# 	# df_pre1_1 = df_pre1_1.drop_duplicates(subset=['peak_id'])
					# 	df_pre1 = df_pre1.loc[~(df_pre1.index.duplicated(keep='first')),:]
					# 	print('df_pre1: ',df_pre1.shape)
					# 	print(df_pre1[0:5])
					# 	print(input_filename_query1)
					# else:
					# 	print('the file does not exist: %s'%(input_filename_query1))
					# 	print('please provide feature group estimation')
					# 	flag_group_query_1 = 1
					# 	# continue
					# 	# return

					if flag_group_query_1==0:
						# peak_loc_1 = df_pre1.index
						# df_pre1 = df_pre1.loc[peak_loc_ori,:]
						df_query1 = df_pre1
					else:
						load_mode_pre1_1 = 1
						if load_mode_pre1_1>0:
							# load the TF binding prediction file
							# the possible columns: (signal,motif,predicted binding,motif group)
							# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
							# input_filename = '%s/%s.peak_loc.flanking_0.%s.1.0.2.%d.1.copy2.txt'%(input_file_path,filename_prefix_1,motif_id1,peak_distance_thresh_1)
							# input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.copy2.txt'%(input_file_path,filename_prefix_1,motif_id1,filename_save_annot_2)
							# folder_id = select_config['folder_id']
							folder_id = folder_id_query
							if folder_id in [1,2]:
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.group2.txt'%(input_file_path_query_1,filename_prefix_1,motif_id1,filename_save_annot_2)
							elif folder_id in [0]:
								upstream_tripod_2 = 100
								filename_save_annot_2_pre2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod_2,type_id_tripod)
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.txt'%(input_file_path_query_1,filename_prefix_1,motif_id1,filename_save_annot_2_pre2)

							if os.path.exists(input_filename==True):
								df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
								peak_loc_1 = df_1.index
								peak_num_1 = len(peak_loc_1)
								print('peak_loc_1: ',peak_num_1)

								method_type_vec_pre2 = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
								# column_motif_vec_2 = ['%s.motif'%(method_type_query) for method_type_query in method_type_vec_2]
								# column_motif_vec_3 = ['%s.pred'%(method_type_query) for method_type_query in method_type_vec_2]
								# column_motif_vec_2 = []
								annot_str_vec = ['motif','pred','score']
								column_vec_query = ['signal']
								for method_type_query in method_type_vec_pre2:
									column_vec_query.extend(['%s.%s'%(method_type_query,annot_str1) for annot_str1 in annot_str_vec])

								# df_query1 = df_1
								df_query1 = df_1.loc[:,column_vec_query]
								print('df_query1: ',df_query1.shape)
								print(df_query1.columns)
								print(df_query1[0:2])
								print(input_filename)
							else:
								print('the file does not exist: %s'%(input_filename))
								df_query1 = pd.DataFrame(index=peak_loc_ori)
								load_mode_pre1_1 = 0

					if (flag_group_query_1==0) or (load_mode_pre1_1>0):
						df_query1_ori = df_query1.copy()
						peak_loc_1 = df_query1.index
						column_vec = df_query1.columns
						df_query1 = pd.DataFrame(index=peak_loc_ori)
						df_query1.loc[peak_loc_1,column_vec] = df_query1_ori
						print('df_query1: ',df_query1.shape)

					column_signal = 'signal'
					if column_signal in df_query1.columns:
						# peak_signal = df_query1['signal']
						peak_signal = df_query1[column_signal]
						id_signal = (peak_signal>0)
						# peak_signal_1_ori = peak_signal[id_signal]
						df_query1_signal = df_query1.loc[id_signal,:]	# the peak loci with peak_signal>0
						peak_loc_signal = df_query1_signal.index
						peak_num_signal = len(peak_loc_signal)
						print('signal_num: ',peak_num_signal)

					if not (column_motif in df_query1.columns):
						peak_loc_motif = peak_loc_ori[motif_data[motif_id_query]>0]
						df_query1.loc[peak_loc_motif,column_motif] = motif_data_score.loc[peak_loc_motif,motif_id_query]

					motif_score = df_query1[column_motif]
					id_motif = (df_query1[column_motif].abs()>0)
					df_query1_motif = df_query1.loc[id_motif,:]	# the peak loci with binding motif identified
					peak_loc_motif = df_query1_motif.index
					peak_num_motif = len(peak_loc_motif)
					# print('motif_num: ',peak_num_motif)
					print('peak_loc_motif ',peak_num_motif)
						
					if peak_num_motif==0:
						continue

					flag_motif_query=1
					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					if flag_motif_query>0:
						df_query1_motif = df_query1.loc[id_motif,:] # peak loci with motif

						# the peak loci with signal
						peak_loc_motif = df_query1_motif.index
						peak_num_motif = len(peak_loc_motif)
						print('peak_loc_motif ',peak_num_motif)
						# filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
						filename_query = '%s/test_query_df_overlap.%s.motif.1.txt' % (input_file_path, filename_save_annot2_2)
						filename_query_2 = '%s/test_query_df_overlap.%s.motif.2.txt' % (input_file_path, filename_save_annot2_2)
						input_filename = filename_query
						input_filename_2 = filename_query_2
						load_mode_2 = 0
						# overwrite_2 = False
						overwrite_2 = True
						if os.path.exists(input_filename)==True:
							if (overwrite_2==False):
								df_overlap_motif = pd.read_csv(input_filename,index_col=0,sep='\t')
								load_mode_2 = load_mode_2+1
						else:
							print('the file does not exist: %s'%(input_filename))

						if os.path.exists(input_filename_2)==True:
							if (overwrite_2==False):
								df_group_basic_motif = pd.read_csv(input_filename_2,index_col=0,sep='\t')
								load_mode_2 = load_mode_2+1
						else:
							print('the file does not exist: %s'%(input_filename))

						if load_mode_2<2:
							# output_filename = '%s/test_query_df_overlap.%s.motif.1.txt' % (output_file_path, filename_save_annot2_2)
							# output_filename_2 = '%s/test_query_df_overlap.%s.motif.2.txt' % (output_file_path, filename_save_annot2_2)
							output_filename = filename_query
							output_filename_2 = filename_query_2
							df_overlap_motif, df_overlap_mtx_motif, dict_group_basic_motif = self.test_query_group_overlap_pre1_2(data=df_query1_motif,dict_group_compare=dict_group_basic_1,
																																  	df_group_1=df_group_1,df_group_2=df_group_2,df_overlap_1=[],df_query_compare=df_overlap_compare,flag_sort=1,flag_group=1,
																																  	stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,
																																	save_mode=1,output_filename=output_filename,output_filename_2=output_filename_2,verbose=verbose,select_config=select_config)
							self.dict_group_basic_motif = dict_group_basic_motif

						# list2 = [dict_group_basic_motif[group_type] for group_type in group_vec_2]
						# df_query_2 = pd.concat(list2,axis=0,join='outer',ignore_index=False)
						# output_filename = '%s/test_query_df_overlap.%s.%s.motif.2.txt' % (output_file_path, motif_id1, data_file_type_query)
						# df_query_2.to_csv(output_filename,sep='\t')
					
					flag_select_query=1
					column_pred1 = '%s.pred'%(method_type_feature_link)
					id_pred1 = (df_query1[column_pred1]>0)
					df_pre1 = df_query1
					# df_query1_2 = df_query1.loc[id_pred1,:]
					
					if flag_select_query>0:
						# select the peak loci predicted with TF binding
						# the selected peak loci
						# thresh_score_query_1 = 0.15
						# thresh_score_query_1 = 0.20

						# df_query1_2 = df_query1.loc[(id_pred1&id_score_query1),:]
						# df_query1_2 = df_query1.loc[(id_pred1&id_score_query2),:]
						# thresh_score_default_3 = 0.10
						# id_score_query_2 = (df_query1[column_score_query1]>thresh_score_default_3)
						# id_1 = id_pred1
						# id_1 = (id_pred1&id_score_query_2)
						# df_query1_2 = df_query1.loc[id_pred1,:]
						# df_query1_2 = df_query1.loc[(id_pred1&id_score_query_2),:]
						id_1 = id_pred1
						df_query1_2 = df_query1.loc[id_1,:] # the selected peak loci
						df_pred1 = df_query1_2
						
						peak_loc_query_group2 = df_query1_2.index
						peak_num_group2 = len(peak_loc_query_group2)
						print('peak_loc_query_group2: ',peak_num_group2)

						# feature_query_vec_2 = peak_loc_query_group2
						peak_query_vec = peak_loc_query_group2

						column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
						column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)
						
						filename_query = '%s/test_query_df_overlap.%s.pre1.1.txt' % (input_file_path, filename_save_annot2_2)
						filename_query_2 = '%s/test_query_df_overlap.%s.pre1.2.txt' % (input_file_path, filename_save_annot2_2)
						input_filename = filename_query
						input_filename_2 = filename_query_2
						load_mode_2 = 0
						# overwrite_2 = False
						overwrite_2 = True
						dict_group_basic_2 = dict()
						if (os.path.exists(input_filename)==True):
							if (overwrite_2==False):
								df_overlap_query = pd.read_csv(input_filename,index_col=0,sep='\t')
								load_mode_2 = load_mode_2+1
						else:
							print('the file does not exist: %s'%(input_filename))

						if (os.path.exists(input_filename_2)==True):
							if (overwrite_2==False):
								df_group_basic_query_2 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
								load_mode_2 = load_mode_2+1
						else:
							print('the file does not exist: %s'%(input_filename_2))

						if load_mode_2<2:
							df_overlap_query, df_overlap_mtx, dict_group_basic_2 = self.test_query_group_overlap_pre1_2(data=df_pred1,dict_group_compare=dict_group_basic_1,df_group_1=df_group_1,df_group_2=df_group_2,df_overlap_1=[],df_query_compare=df_overlap_compare,flag_sort=1,flag_group=1,
																															stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,
																															save_mode=0,output_filename='',verbose=verbose,select_config=select_config)

							# list_query1 = [dict_group_basic_2[group_type] for group_type in group_vec_query]
							list_query1 = []
							group_vec_query = ['group1','group2']
							for group_type in group_vec_query:
								df_query = dict_group_basic_2[group_type]
								df_query['group_type'] = group_type
								list_query1.append(df_query)

							df_query = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
							flag_sort_2=1
							if flag_sort_2>0:
								df_query = df_query.sort_values(by=['group_type','stat_fisher_exact_','pval_fisher_exact_'],ascending=[True,False,True])
							# output_filename = '%s/test_query_df_overlap.%s.%s.pre1.2.txt' % (output_file_path, motif_id1, data_file_type_query)
							output_filename = '%s/test_query_df_overlap.%s.pre1.2.txt' % (output_file_path,filename_save_annot2_2)
							df_query = df_query.round(7)
							df_query.to_csv(output_filename,sep='\t')
							df_group_basic_query_2 = df_query
							self.df_group_basic_query_2 = df_group_basic_query_2
							self.dict_group_basic_2 = dict_group_basic_2

						# TODO: automatically adjust the group size threshold
						# df_overlap_query = df_overlap_query.sort_values(by=['freq_obs','pval_chi2_'],ascending=[False,True])
						dict_thresh = dict()
						if len(dict_thresh)==0:
							thresh_value_overlap = 0
							thresh_pval_1 = 0.20
							field_id1 = 'overlap'
							field_id2 = 'pval_fisher_exact_'
							# field_id2 = 'pval_chi2_'
						else:
							column_1 = 'thresh_overlap'
							column_2 = 'thresh_pval_overlap'
							column_3 = 'field_value'
							column_5 = 'field_pval'
							column_vec_query1 = [column_1,column_2,column_3,column_5]
							list_query1 = [dict_thresh[column_query] for column_query in column_vec_query1]
							thresh_value_overlap, thresh_pval_1, field_id1, field_id2 = list_query1
						
						# id1 = (df_overlap_query['overlap']>thresh_value_overlap)
						# id2 = (df_overlap_query['pval_chi2_']<thresh_pval_1)
						id1 = (df_overlap_query[field_id1]>thresh_value_overlap)
						id2 = (df_overlap_query[field_id2]<thresh_pval_1)

						df_overlap_query2 = df_overlap_query.loc[id1,:]
						print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape)

						df_overlap_query.loc[id1,'label_1'] = 1
						group_vec_query_1 = np.asarray(df_overlap_query2.loc[:,['group1','group2']])
						print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape,motif_id1)

						self.df_overlap_query = df_overlap_query
						self.df_overlap_query2 = df_overlap_query2
						# TODO: automatically adjust the group size threshold
						# df_overlap_query = df_overlap_query.sort_values(by=['freq_obs','pval_chi2_'],ascending=[False,True])
						# thresh_value_overlap = 10
						# thresh_value_overlap = 0
						# thresh_pval_1 = 0.20
						# id1 = (df_overlap_query['overlap']>thresh_value_overlap)
						# # id2 = (df_overlap_query['pval_chi2_']<thresh_pval_1)

						# # id_1 = (id1&id2)
						# id_1 = id1
						# df_overlap_query2 = df_overlap_query.loc[id_1,:]
						# df_overlap_query.loc[id_1,'label_1'] = 1
						# group_vec_query_1 = np.asarray(df_overlap_query2.loc[:,['group1','group2']])
						# print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape,motif_id1)

						if load_mode_2<2:
							# output_filename = '%s/test_query_df_overlap.%s.%s.pre1.1.txt'%(output_file_path,motif_id1,data_file_type_query)
							output_filename = '%s/test_query_df_overlap.%s.pre1.1.txt'%(output_file_path,filename_save_annot2_2)
							df_overlap_query.to_csv(output_filename,sep='\t')

					flag_neighbor_query_1 = 0
					if flag_group_query_1>0:
						flag_neighbor_query_1 = 1
					flag_neighbor = 1
					flag_neighbor_2 = 1  # query neighbor of selected peak in the paired groups

					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					# feature_type_vec_query = [feature_type_query_1,feature_type_query_2]
					# feature_type_vec = ['latent_peak_motif','latent_peak_tf']
					feature_type_vec = feature_type_vec_query
					print('feature_type_vec: ',feature_type_vec)
					select_config.update({'feature_type_vec':feature_type_vec})
					self.feature_type_vec = feature_type_vec
					df_group_1 = self.df_group_pre1
					df_group_2 = self.df_group_pre2

					feature_type_vec = select_config['feature_type_vec']
					feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]
					group_type_vec_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]

					group_type_query1, group_type_query2 = group_type_vec_2[0:2]
					peak_loc_pre1 = df_pre1.index
					df_pre1[group_type_query1] = df_group_1.loc[peak_loc_pre1,method_type_group]
					df_pre1[group_type_query2] = df_group_2.loc[peak_loc_pre1,method_type_group]

					if flag_neighbor_query_1>0:
						# query peak loci predicted with binding sites using clustering
						start = time.time()
						df_overlap_1 = []
						group_type_vec = ['group1','group2']
						# group_vec_query = ['group1','group2']
						list_group_query = [df_group_1,df_group_2]
						dict_group = dict(zip(group_type_vec,list_group_query))
						
						dict_neighbor = self.dict_neighbor
						dict_group_basic_1 = self.dict_group_basic_1
						# the overlap and the selected overlap above count and p-value thresholds
						# group_vec_query_1: the group enriched with selected peak loci
						column_id2 = 'peak_id'
						df_pre1[column_id2] = np.asarray(df_pre1.index)
						df_pre1 = self.test_query_binding_clustering_1(data1=df_pre1,data2=df_pred1,dict_group=dict_group,dict_neighbor=dict_neighbor,dict_group_basic_1=dict_group_basic_1,
																		df_overlap_1=df_overlap_1,df_overlap_compare=df_overlap_compare,
																		group_type_vec=group_type_vec,feature_type_vec=feature_type_vec,group_vec_query=group_vec_query_1,input_file_path='',																												
																		save_mode=1,output_file_path=output_file_path,output_filename='',
																		filename_prefix_save='',filename_save_annot=filename_save_annot2_2,verbose=verbose,select_config=select_config)
						
						stop = time.time()
						print('query feature group and neighbor annotation for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop-start))

					# column_vec_query1 = [column_signal,column_motif,column_pred1,column_score_query1]
					# column_vec_query1 = [column_signal,column_motif,column_pred1]+column_motif_vec_2
					column_score_query_1 = '%s.score'%(method_type_feature_link)
					column_vec_query1 = [column_signal]+column_motif_vec_2+[column_motif,column_pred1]
					column_vec_query1 = column_vec_query1 + [column_score_query_1]
					
					column_vec_query1_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]
					# column_vec_query2_2 = [feature_type_query_1,feature_type_query_2]
					# column_vec_query2 = column_vec_query1 + column_vec_query1_2 + column_vec_query2_2
					column_vec_query2 = column_vec_query1 + column_vec_query1_2

					# df_pre1 = pd.DataFrame(index=peak_loc_ori,columns=column_vec_query2)
					# # df_pre1[feature_type_query_1]=0
					# # df_pre1[feature_type_query_2]=0
					# df_pre1.loc[peak_loc_1,column_vec_query1] = df_query1.loc[peak_loc_1,column_vec_query1]
					# df_pre1[column_vec_query1_2[0]] = df_group_1.loc[peak_loc_ori,'group']
					# df_pre1[column_vec_query1_2[1]] = df_group_2.loc[peak_loc_ori,'group']

					field_id1 = 'peak_tf_corr'
					field_id2 = 'peak_tf_pval_corrected'
					# query peak accessibility-TF expression correlation
					# flag_peak_tf_corr = 0
					flag_peak_tf_corr = 1
					column_peak_tf_corr = 'peak_tf_corr'
					column_query = column_peak_tf_corr
					if column_query in df_pre1.columns:
						flag_peak_tf_corr = 0

					if flag_peak_tf_corr>0:
						# input_filename_1, input_filename_2 = select_config['filename_rna'],select_config['filename_atac']
						# input_filename_3 = select_config['filename_rna_exprs_1']

						# motif_data = self.motif_data
						# correlation_type='spearmanr'
						# correlation_type=select_config['correlation_type']
						save_mode = 1
						# filename_prefix = 'test_peak_tf_correlation'
						# filename_prefix = 'test_peak_tf_correlation.%s'%(data_file_type_query)
						# field_query = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected','motif_basic']
						# flag_load_1 = 0
						flag_load_1 = 1
						# field_load = []
						
						# data_file_type_query = select_config['data_file_type']
						# input_filename_list1 = ['%s/%s.%s.1.copy1.txt'%(file_save_path,filename_prefix,filename_annot1) for filename_annot1 in filename_annot_vec[0:3]]
						input_filename_list1 = []
						motif_query_vec = [motif_id_query]
						# motif_data_query = pd.DataFrame(index=peak_query_group2,columns=motif_query_vec,data=1)
						peak_loc_2 = peak_loc_ori
						motif_data_query = pd.DataFrame(index=peak_loc_2,columns=motif_query_vec,data=1)
						
						correlation_type = 'spearmanr'
						alpha = 0.05
						method_type_id_correction = 'fdr_bh'
						# filename_prefix = 'test_peak_tf_correlation.%s.%s.2'%(motif_id1,data_file_type_query)
						filename_prefix = 'test_peak_tf_correlation.%s.%s.2'%(motif_id_query,data_file_type_query)
						input_file_path_query1 = input_file_path_query_1
						output_file_path_query1 = input_file_path_query_1
						save_mode_2 = 1
						# field_query = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected','motif_basic']
						field_load = [correlation_type,'pval','pval_corrected']
						dict_peak_tf_corr_ = utility_1.test_peak_tf_correlation_query_1(motif_data=motif_data_query,
																							peak_query_vec=[],
																							motif_query_vec=motif_query_vec,
																							peak_read=peak_read,
																							rna_exprs=rna_exprs,
																							correlation_type=correlation_type,
																							pval_correction=1,
																							alpha=alpha,method_type_id_correction=method_type_id_correction,
																							flag_load=flag_load_1,field_load=field_load,
																							save_mode=save_mode_2,
																							input_file_path=input_file_path_query1,
																							input_filename_list=input_filename_list1,
																							output_file_path=output_file_path_query1,
																							filename_prefix=filename_prefix,
																							select_config=select_config)

						# self.dict_peak_tf_corr_ = dict_peak_tf_corr_
						# select_config = self.select_config

						# field_id1 = 'peak_tf_corr'
						# field_id2 = 'peak_tf_pval_corrected'
						field_id1_query = field_load[0]
						field_id2_query = field_load[2]
						peak_tf_corr_1 = dict_peak_tf_corr_[field_id1_query]
						peak_tf_pval_corrected1 = dict_peak_tf_corr_[field_id2_query]
						print('peak_tf_corr_1: ',peak_tf_corr_1.shape)
						print(peak_tf_corr_1[0:2])

						print('peak_tf_pval_corrected: ',peak_tf_pval_corrected1.shape)
						print(peak_tf_pval_corrected1[0:2])

						peak_vec_2 = peak_loc_2
						column_corr_1 = field_id1
						column_pval = field_id2
						df_pre1.loc[peak_vec_2,column_corr_1] = peak_tf_corr_1.loc[peak_vec_2,motif_id_query]
						df_pre1.loc[peak_vec_2,column_pval] = peak_tf_pval_corrected1.loc[peak_vec_2,motif_id_query]

						# input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.copy2_2.txt'%(input_file_path,filename_prefix_1,motif_id1,filename_save_annot_pre2)
						# output_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.copy2_2_1.txt'%(output_file_path,filename_prefix_1,motif_id1,filename_save_annot2)
						# df_pre2 = df_pre1.loc[peak_loc_1,:]
						# df_pre2.to_csv(output_filename,sep='\t')
						# df_pre2 = df_pre1.sort_values(by=[column_signal],ascending=False)
						# df_pre2.to_csv(output_filename,sep='\t')

						output_file_path = file_path_query1
						# filename_save_annot2_2 = '%s.%s_%s.neighbor%d.%d'%(method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
						# filename_save_annot2 = '%s.%s_%s'%(filename_save_annot2,feature_type_query_1,feature_type_query_2)
						# output_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.copy2_2.txt'%(output_file_path,filename_prefix_1,motif_id1,filename_save_annot2)
						output_filename = '%s/%s.%s.%s.copy2_2.txt'%(output_file_path,filename_prefix_1,motif_id1,filename_save_annot2_2)
						# df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
						# peak_loc_1 = df_1.index
						# df_pre1 = df_pre1.sort_values(by=['signal',column_motif],ascending=False)
						if os.path.exists(output_filename)==True:
							print('the file exists: %s'%(output_filename))
							df_pre2.to_csv(output_filename,sep='\t')

					# flag_peak_tf_corr_2 = 1
					flag_peak_tf_corr_2 = 0
					if (flag_peak_tf_corr_2>0):
						if column_signal in df_pre1.columns:
							id_signal = (df_pre1[column_signal]>0)
							peak_signal = peak_loc_pre1[id_signal]
							# meta_exprs_2 = self.meta_exprs_2
							# sample_id = peak_read.index
							# meta_exprs_2 = meta_exprs_2.loc[sample_id,:]

							# peak_read_1 = peak_read.loc[:,peak_signal]
							# tf_expr_1 = meta_exprs_2.loc[:,motif_id_query]
							column_query = column_peak_tf_corr
							# thresh_1 = -0.01
							thresh_1 = -0.05
							id_1 = (df_pre1[column_peak_tf_corr]>thresh_1)&(id_signal)
							id_2 = (~id_1)&(id_signal)
							peak_query_1 = peak_loc_pre1[id_1]
							peak_query_2 = peak_loc_pre1[id_2]

							peak_tf_corr_1 = df_pre1.loc[peak_signal,column_query]
							peak_tf_corr_2 = df_pre1.loc[peak_query_1,column_query]
							peak_tf_corr_3 = df_pre1.loc[peak_query_2,column_query]
							list1 = [peak_tf_corr_1,peak_tf_corr_2,peak_tf_corr_3]

							# peak_num_1, peak_num_2, peak_num_3 = len(peak_query_1),len(peak_query_2),len(peak_query_3)
							peak_num_1, peak_num_2, peak_num_3 = len(peak_signal),len(peak_query_1),len(peak_query_2)
							list2 = [peak_num_1,peak_num_2,peak_num_3]
							list_query1 = []
							query_num = len(list1)
							for i2 in range(query_num):
								peak_tf_corr_query = list1[i2]
								# peak_query = list2[i2]
								# peak_num_query = len(peak_query)
								peak_num_query = list2[i2]
								t_vec_1 = [peak_num_query,np.max(peak_tf_corr_query),np.min(peak_tf_corr_query),np.mean(peak_tf_corr_query),np.median(peak_tf_corr_query)]
								list_query1.append(t_vec_1)

							mtx_1 = np.asarray(list_query1)
							group_vec_peak_tf = ['peak_signal','peak_signal_1','peak_signal_2']
							column_vec = ['peak_num','corr_max','corr_min','corr_mean','corr_median']
							df_annot_peak_tf = pd.DataFrame(index=group_vec_peak_tf,columns=column_vec,data=mtx_1)
							field_query = ['motif_id','motif_id1','motif_id2']
							list_pre1 = [motif_id_query,motif_id1,motif_id2]
							field_num1 = len(field_query)
							for i2 in range(file_num1):
								field_id, query_value = field_query[i2], list_pre1[i2]
								df_annot_peak_tf[field_id] = query_value
							list_annot_peak_tf.append(df_annot_peak_tf)

					# column_score_1 = 'score_pred1'
					# column_score_query1 = column_score_1
					# column_score_query1 = '%s.%s'%(method_type_feature_link,column_score_1)
					# df_score_annot = []
					if load_mode>0:
						if not (column_score_query1 in df_query1.columns):
							id1 = (df_score_annot[column_id3]==motif_id_query)
							df_score_annot_query = df_score_annot.loc[id1,:]
							peak_loc_2 = df_score_annot_query[column_id2].unique()
							df_query1.loc[peak_loc_2,column_score_query1] = df_score_annot_query.loc[peak_loc_2,column_score_1]

					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					# flag_signal_query=1
					# if flag_signal_query>0:
					# 	df_query1_signal = df_query1.loc[id_signal,:]
					# 	peak_loc_signal = df_query1_signal.index  # the peak loci with signal
					# 	peak_num_signal = len(peak_loc_signal)
					# 	print('peak_loc_signal ',peak_num_signal)

					# 	# df_overlap_signal: column: group1, group2, overlap, enrichment statistics and p-value;
					# 	# df_overlap_mtx_signal: wide format matrix
					# 	# dict_group_basic_signal: group size, enrichment statistics and p-value for each group
					# 	output_filename = '%s/test_query_df_overlap.%s.signal.1.txt'%(output_file_path,filename_save_annot2_2)
					# 	output_filename_2 = '%s/test_query_df_overlap.%s.signal.2.txt'%(output_file_path,filename_save_annot2_2)
					# 	df_overlap_signal, df_overlap_mtx_signal, dict_group_basic_signal = self.test_query_group_overlap_pre1_2(data=df_query1_signal,dict_group_compare=dict_group_basic_1,
					# 																											 	df1=df_group_1,df2=df_group_2,df_overlap_1=[],df_query_compare=df_overlap_compare,flag_sort=1,flag_group=1,
					# 																												 stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,
					# 																												save_mode=1,output_filename=output_filename,output_filename_2=output_filename_2,verbose=verbose,select_config=select_config)

					# 	# list1 = [dict_group_basic_signal[group_type] for group_type in group_vec_2]
					# 	# df_query_1 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
					# 	# output_filename_2 = '%s/test_query_df_overlap.%s.%s.signal.2.txt'%(output_file_path, motif_id1, data_file_type_query)
					# 	# df_query_1.to_csv(output_filename_2,sep='\t')
					column_motif_group = 'motif_group_1'
					column_peak_tf_corr_1 = 'group_correlation'
					column_motif_group_corr_1 = 'motif_group_correlation'

					# method_type_feature_link = select_config['method_type_feature_link']
					# n_neighbors = select_config['neighbor_num']
					column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
					column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)
					column_neighbor = ['neighbor%d'%(id1) for id1 in np.arange(1,n_neighbors+1)]
					column_1 = '%s_group_neighbor'%(feature_type_query_1)
					column_2 = '%s_group_neighbor'%(feature_type_query_2)
					column_query1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peak
					column_query2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peak

					column_pred_3 = '%s.pred_group_neighbor'%(method_type_feature_link)
					column_pred_5 = '%s.pred_group_neighbor_1'%(method_type_feature_link)
					column_pred_6 = '%s.pred_neighbor_2_group'%(method_type_feature_link)
					column_pred_7 = '%s.pred_neighbor_2'%(method_type_feature_link)
					column_pred_8 = '%s.pred_neighbor_1'%(method_type_feature_link)

					column_vec_query_pre1 = [column_motif_group,column_peak_tf_corr_1,column_motif_group_corr_1]
					column_vec_query_pre2 = [column_pred2,column_pred_2,column_pred_3,column_pred_5,column_1,column_2,column_pred_6,column_pred_7,column_pred_8,column_query1,column_query2]
					column_vec_query_pre2_2 = column_vec_query_pre1 + column_vec_query_pre2

					flag_query2_1=0
					if flag_query2_1>0:
						# flag_1=0
						flag_motif_ori = 0
						# TF binding prediction based on motif scanning with sequence feature
						if flag_motif_ori>0:
							# peak_signal_1 = df_query1_1['signal'] # the peak loci with peak_signal>0 and motif identified by motif scanning
							# peak_loc_query1 = df_query1.index # the peak loci with peak_signal>0 or motif identified by motif scanning
							# peak_loc_query2 = df_query1_1.index # the peak loci with peak_signal>0 and motif identified by motif scanning
							# print('df_query1, df_query1_1, df_query1_signal, df_query1_motif ',df_query1.shape,df_query1_1.shape,df_query1_signal.shape,df_query1_motif.shape)
								
							# query_num1 = df_query1_1.shape[0]
							# precision_motif = query_num1/peak_num_motif
							# recall_motif = query_num1/peak_num_signal
							# print('precision_motif, recall_motif ',query_num1,peak_num_motif,peak_num_signal,precision_motif,recall_motif)

							column_vec = ['signal',column_motif]
							df_query1_1 = df_query1.loc[(id_motif|id_signal), :]
							t_vec_1 = self.test_query_pred_score_1(data=df_query1_1,column_vec=column_vec,iter_mode=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config=select_config)
							df_score_1, df_score_2, contingency_table, dict_query1 = t_vec_1
							print('df_score_1: \n',df_score_1)
							print('contingency_table: \n',contingency_table)

						# column_vec_query = [column_pred2,column_pred_2,column_pred_3,column_pred_5,column_pred_6,column_pred_7,column_pred_8]
						column_vec_query = [column_pred2,column_pred_2,column_pred_3,column_pred_5,column_1,column_2,column_pred_6,column_pred_7,column_pred_8,column_query1,column_query2]
						# column_vec = ['signal',column_motif]
						# t_vec_1 = self.test_query_pred_score_1(data=df_query1_1,column_vec=column_vec,iter_mode=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config=select_config)
						# df_score_1, df_score_2, contingency_table, dict_query1 = t_vec_1
						# print('df_score_1: \n',df_score_1)
						# print('contingency_table: \n',contingency_table)

						id_pred2 = (df_pre1[column_pred2]>0)
						df_pre1.loc[id_pred2,column_vec_query] = 1

						id1 = (df_pre1[column_motif].abs()>0)
						id2 = (~id1)
						df_query_group1 = df_pre1.loc[id1,:]
						df_query_group2 = df_pre1.loc[id2,:]
						print('df_pre1, df_query_group1, df_query_group2: ',df_pre1.shape,df_query_group1.shape,df_query_group2.shape)

						list_group = [df_query_group1,df_query_group2]
						query_num_1 = len(list_group)
						query_num_2 = len(column_vec_query)

						list_query1 = []
						for i2 in range(query_num_1):
							df_query_group = list_group[i2]
							list_query2 = []
							list_query3 = []
							for t_id1 in range(query_num_2):
								# if (i2==1) and (t_id1==0):
								# 	continue
								try:
									column_query = column_vec_query[t_id1]
									column_vec = [column_signal,column_query]
									# TF binding prediction performance using the feature group method
									t_vec_1 = self.test_query_pred_score_1(data=df_query_group,column_vec=column_vec,iter_mode=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config=select_config)
									
									df_score_1, df_score_2, contingency_table, dict_query1 = t_vec_1
									print('field_query: ',column_query)
									print('df_score_1: \n',df_score_1)
									print('contingency_table: \n',contingency_table)
									list_query2.append(df_score_1)
									list_query3.append(column_query)
								except Exception as error:
									print('error! ',error)

							if len(list_query2)>0:
								df_score_query = pd.concat(list_query2,axis=1,join='outer',ignore_index=False)
								df_score_query = df_score_query.T

								# if i2==0:
								# 	column_vec_query_1 = column_vec_query
								# else:
								# 	column_vec_query_1 = column_vec_query[1:]

								column_vec_query_1 = np.asarray(list_query3)
								df_score_query['motif_id'] = motif_id_query
								df_score_query['motif_id1'] = motif_id1
								df_score_query['motif_id2'] = motif_id2
								df_score_query['group_motif'] = int(1-i2)
								df_score_query['neighbor_num'] = n_neighbors
								df_score_query['method_type'] = column_vec_query_1
								df_score_query['method_type_group'] = method_type_group

								list_query1.append(df_score_query)

						if len(list_query1)>0:
							df_score_query_1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)					
							# output_filename = '%s/test_query_df_score.%s.%s.1.txt'%(output_file_path,motif_id1,data_file_type_query)
							# output_filename = '%s/test_query_df_score.%s.1.txt'%(output_file_path,filename_save_annot2_2)
							# df_score_query_1.to_csv(output_filename,sep='\t')
							list_score_query_1.append(df_score_query_1)

						if (interval_save>0):
							if len(list_score_query_1)>0:
								df_score_query_2 = pd.concat(list_score_query_1,axis=0,join='outer',ignore_index=False)
								# filename_save_annot2_1 = '%s.%s'%(method_type_group,data_file_type_query)
								filename_save_annot2_1 = '%s.neighbor%d.%s.%d'%(method_type_group,n_neighbors,data_file_type_query,config_id_load)
								output_filename = '%s/test_query_df_score.%s.1.txt'%(output_file_path,filename_save_annot2_1)
								df_score_query_2.to_csv(output_filename,sep='\t')

					if save_mode>0:
						output_file_path = file_path_query1
						# output_file_path = file_path_query2
						# filename_save_annot2_2 = '%s.%s_%s'%(method_type_group,feature_type_query_1,feature_type_query_2)
						filename_save_annot2_2 = '%s.%s_%s.neighbor%d.%d'%(method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
						# filename_save_annot2 = '%s.%s_%s'%(filename_save_annot2,feature_type_query_1,feature_type_query_2)
						
						# output_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.copy2_2.txt'%(output_file_path,filename_prefix_1,motif_id1,filename_save_annot2)
						output_filename = '%s/%s.%s.%s.copy2_2.txt'%(output_file_path,filename_prefix_1,motif_id1,filename_save_annot2_2)
						# df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
						# peak_loc_1 = df_1.index
						df_pre1 = df_pre1.sort_values(by=['signal',column_motif],ascending=False)
						df_pre1.to_csv(output_filename,sep='\t')

					flag_query_2=1
					method_type_feature_link = select_config['method_type_feature_link']
					if flag_query_2>0:
						flag_select_1=1
						# column_pred1 = '%s.pred'%(method_type_feature_link)
						# id_pred1 = (df_query1[column_pred1]>0)
						# df_query2 = df_query1.loc[id_pred1,:]
						# peak_loc_pre1 = df_query1.index
						# peak_loc_pred1 = peak_loc_pre1[id_pred1]

						# column_corr_1 = field_id1
						# column_pval = field_id2
						column_corr_1 = 'peak_tf_corr'
						column_pval = 'peak_tf_pval_corrected'
						thresh_corr_1, thresh_pval_1 = 0.30, 0.05
						thresh_corr_2, thresh_pval_2 = 0.1, 0.1
						thresh_corr_3, thresh_pval_2 = 0.05, 0.1

						if flag_select_1>0:
							# find the paired groups with enrichment
							# thresh_overlap_default_1 = 0
							# thresh_overlap_default_2 = 0
							# thresh_overlap = 0
							# # thresh_pval_1 = 0.10
							# # thresh_pval_1 = 0.20
							# thresh_pval_group = 0.25
							# # thresh_quantile_overlap = 0.50
							# thresh_quantile_overlap = 0.75

							# column_1 = 'thresh_overlap_default_1'
							# column_2 = 'thresh_overlap_default_2'
							# column_3 = 'thresh_overlap'
							# column_pval_group = 'thresh_pval_1'
							# column_quantile = 'thresh_quantile_overlap'

							# column_thresh_query = [column_1,column_2,column_3,column_pval_group,column_quantile]
							# thresh_vec = [thresh_overlap_default_1,thresh_overlap_default_2,thresh_overlap,thresh_pval_group,thresh_quantile_overlap]
							# dict_thresh = dict(zip(column_thresh_query,thresh_vec))

							# # query the enrichment of predicted peak loci in one type of group
							# group_type_vec = ['group1','group2']
							# print('df_group_basic_query_2: ',df_group_basic_query_2.shape)
							# # print(df_group_basic_query_2[0:2])
							# column_vec_query = ['count','pval_fisher_exact_']
							# flag_enrichment = 1
							# flag_size = 1
							# type_id_1, type_id_2 = 1, 1
							# # dict_query = {'enrichment':df_overlap_query1,'group_size':df_overlap_query2}
							# dict_query_pre1 = self.test_query_enrichment_group_1_unit1(data=df_group_basic_query_2,dict_group=dict_group_basic_2,dict_thresh=dict_thresh,group_type_vec=group_type_vec,
							# 															column_vec_query=column_vec_query,flag_enrichment=flag_enrichment,flag_size=flag_size,type_id_1=type_id_1,type_id_2=type_id_2,save_mode=1,verbose=verbose,select_config=select_config)
							
							# group_type_1, group_type_2 = group_type_vec[0:2]
							# df_query_group1_1,dict_query_group1_1 = dict_query_pre1[group_type_1]
							# df_query_group2_1,dict_query_group2_1 = dict_query_pre1[group_type_2]

							# # group_vec_query1_1 = df_query_group1_1.index
							# # group_vec_query2_1 = df_query_group2_1.index
							# # group_num1, group_num2 = len(group_vec_query1_1), len(group_vec_query2_1)
							# # print('group_vec_query1_1, group_vec_query2_1: ',group_num1,group_num2)

							# field_query_2 = ['enrichment','group_size','group_size_query']
							# field_id1, field_id2 = field_query_2[0:2]
							# field_id3 = field_query_2[2]

							# list_query1 = []
							# dict_query1 = dict()
							# for group_type in group_type_vec:
							# 	print('group_type: ',group_type)
							# 	dict_query_group = dict_query_pre1[group_type][1]
							# 	group_vec_query1_1 = dict_query_group[field_id1].index.unique()
							# 	group_vec_query2_1 = dict_query_group[field_id2].index.unique()
							# 	group_num1_1, group_num2_1 = len(group_vec_query1_1), len(group_vec_query2_1)
							# 	# list_query1.append([group_vec_query1_1,group_vec_query2_1])
							# 	dict_query1.update({group_type:[group_vec_query1_1,group_vec_query2_1]})
								
							# 	print('group_vec_query1_1, group_vec_query2_1: ',group_num1_1,group_num2_1)
							# 	print(group_vec_query1_1)
							# 	print(group_vec_query2_1)

							# 	df_quantile_1 = dict_query_group[field_id3]
							# 	print('df_quantile_1: ',df_quantile_1.shape)
							# 	print(df_quantile_1[0:5])

							# 	# iter_id1 = 0
							# 	filename_save_annot_query_1 = '%s.%d'%(data_file_type_query,config_id_load)
							# 	filename_save_annot_query1 = '%s.neighbor%d'%(method_type_group,n_neighbors)
							# 	# filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
							# 	# output_filename = '%s/test_query_quantile.%s.2.txt'%(output_file_path,group_type)
							# 	output_filename = '%s/test_query_quantile.%s.%s.%s.%s.2.txt'%(output_file_path,filename_save_annot_query1,group_type,motif_id1,filename_save_annot_query_1)
							# 	df_quantile_1.to_csv(output_filename,sep='\t')

							# # print('group_type: ',group_type_2)
							# # group_vec_query1_2 = dict_query_group2_1[field_id1].index.unique()
							# # group_vec_query2_2 = dict_query_group2_1[field_id2].index.unique()
							# # group_num1_2, group_num2_2 = len(group_vec_query1_2), len(group_vec_query2_2)
							# # print('group_vec_query1_2, group_vec_query2_2: ',group_num1_2,group_num2_2)

							# # group_vec_query1_1,group_vec_query2_1 = list_query1[0]
							# # group_vec_query1_2,group_vec_query2_2 = list_query1[1]

							# group_vec_query1_1, group_vec_query2_1 = dict_query1[group_type_1]
							# group_vec_query1_2, group_vec_query2_2 = dict_query1[group_type_2]

							# # query the enrichment of predicted peak loci in paired groups
							# column_vec_query_2 = ['overlap','pval_fisher_exact_']
							# flag_enrichment = 1
							# flag_size = 1
							# type_id_1, type_id_2 = 1, 1
							# df_overlap_query_pre2, dict_query_pre2 = self.test_query_enrichment_group_2_unit1(data=df_overlap_query,dict_group=[],dict_thresh=dict_thresh,group_type_vec=group_type_vec,
							# 																					column_vec_query=column_vec_query_2,flag_enrichment=flag_enrichment,flag_size=flag_size,type_id_1=type_id_1,type_id_2=type_id_2,save_mode=1,verbose=verbose,select_config=select_config)
							# group_vec_query2 = np.asarray(df_overlap_query_pre2.loc[:,group_type_vec].astype(int))
							# group_num_2 = len(group_vec_query2)
							# print('group_vec_query2: ',group_num_2)
							# print(group_vec_query2[0:5])

							# df_1 = dict_query_pre2[field_id1] # group with enrichment above threshold
							# df_2 = dict_query_pre2[field_id2] # group with group_size above threshold
							# group_vec_query1_pre2 = df_1.loc[:,group_type_vec].astype(int)
							# group_vec_query2_pre2 = df_2.loc[:,group_type_vec].astype(int)
							# group_num1_pre2 = len(group_vec_query1_pre2)
							# group_num2_pre2 = len(group_vec_query2_pre2)

							# print('field_id, group_vec_query1_pre2: ',field_id1,group_num1_pre2)
							# # print(group_vec_query1_pre2)
							# print(df_1)

							# print('field_id, group_vec_query2_pre2: ',field_id2,group_num2_pre2)
							# # print(group_vec_query2_pre2)
							# print(df_2)

							# group_vec_1 = group_vec_query1_1
							# group_vec_1_overlap = group_vec_query1_pre2[group_type_1].unique()

							# group_vec_2 = group_vec_query1_2
							# group_vec_2_overlap = group_vec_query1_pre2[group_type_2].unique()

							# group_vec_pre1 = pd.Index(group_vec_1).difference(group_vec_1_overlap,sort=False)
							# print('group with enrichment in feature type 1 but not enriched in joint groups')
							# print('group_vec_pre1: ',len(group_vec_pre1))
							# # df1_query2 = df_overlap_query.loc[df_overlap_query[group_type_1].isin(group_vec_pre1),:]
							# list1 = []
							# column_query1, column_query2 = column_vec_query_2[0:2]
							# df_overlap_query = df_overlap_query.sort_values(by=['pval_fisher_exact_'],ascending=True)
							# thresh_size_query1 = 1

							# flag_group_pre2 = 0
							# if flag_group_pre2>0:
							# 	for group_type_query in group_vec_pre1:
							# 		df1 = df_overlap_query.loc[df_overlap_query[group_type_1]==group_type_query,:]
							# 		df1_1 = df1.loc[df1[column_query1]>thresh_size_query1,:]
							# 		group_query_1 = np.asarray(df1.loc[:,group_type_vec])[0]
							# 		group_query_2 = np.asarray(df1_1.loc[:,group_type_vec])
							# 		list1.append(group_query_1)
							# 		list1.extend(group_query_2)

							# list2 = []
							# group_vec_pre2 = pd.Index(group_vec_2).difference(group_vec_2_overlap,sort=False)
							# print('group with enrichment in feature type 2 but not enriched in joint groups')
							# print('group_vec_pre2: ',len(group_vec_pre2))
							# if flag_group_pre2>0:
							# 	for group_type_query in group_vec_pre2:
							# 		df2 = df_overlap_query.loc[df_overlap_query[group_type_2]==group_type_query,:]
							# 		df2_1 = df1.loc[df1[column_query1]>thresh_size_query1,:]
							# 		group_query_1 = np.asarray(df2.loc[:,group_type_vec])[0]
							# 		group_query_2 = np.asarray(df2_1.loc[:,group_type_vec])
							# 		list2.append(group_query_1)
							# 		list2.extend(group_query_2)

							# group_vec_query1_pre2 = np.asarray(group_vec_query1_pre2)
							# list_pre1 = list(group_vec_query1_pre2)+list1+list2

							# query_vec = np.asarray(list_pre1)
							# df_1 = pd.DataFrame(data=query_vec,columns=group_type_vec).astype(int)
							# df_1.index = utility_1.test_query_index(df_1,column_vec=[group_type_1,group_type_2],symbol_vec=['_'])
							# df_1 = df_1.drop_duplicates(subset=group_type_vec)
							# group_id_1 = df_1.index
							# group_id_pre1 = ['%s_%s'%(group_1,group_2) for (group_1,group_2) in group_vec_query1_pre2]
							# group_id_2 = group_id_1.difference(group_id_pre1,sort=False)

							# column_query_1 = df_overlap_query.columns.difference(group_type_vec,sort=False)
							# df_1.loc[:,column_query_1] = df_overlap_query.loc[group_id_1,column_query_1]
							# group_vec_query1 = df_1.loc[:,group_type_vec]
							# print('df_1: ',df_1.shape)
							# print(df_1)

							# group_type_vec_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]
							# df_query1['group_id2'] = utility_1.test_query_index(df_query1,column_vec=group_type_vec_2,symbol_vec=['_'])
							# id1 = (df_query1['group_id2'].isin(group_id_1))
							# id2 = (df_query1['group_id2'].isin(group_id_2))
							# df_query1.loc[id1,'group_overlap'] = 1 # the group with enrichment
							# df_query1.loc[id2,'group_overlap'] = 2 # the paired group without enrichment but the single group with enrichment
							# print('group_id_1, group_id_2: ',len(group_id_1),len(group_id_2))

							df_annot_vec = [df_group_basic_query_2,df_overlap_query]
							dict_group_annot_1 = {'df_group_basic_query_2':df_group_basic_query_2,'df_overlap_query':df_overlap_query,
													'dict_group_basic_2':dict_group_basic_2}

							key_vec_query = list(dict_group_annot_1.keys())
							for field_id in key_vec_query:
								print(field_id)
								print(dict_group_annot_1[field_id])

							output_file_path_query = file_path_query2
							df_query1 = self.test_query_training_group_pre1(data=df_query1,motif_id1=motif_id1,dict_annot=dict_group_annot_1,method_type_feature_link=method_type_feature_link,dict_thresh=[],thresh_vec=[],input_file_path='',
																				save_mode=1,output_file_path=output_file_path_query,verbose=verbose,select_config=select_config)

							# column_corr_1 = 'peak_tf_corr'
							# column_pval = 'peak_tf_pval_corrected'
							# thresh_corr_1, thresh_pval_1 = 0.30, 0.05
							# thresh_corr_2, thresh_pval_2 = 0.1, 0.1
							# thresh_corr_3, thresh_pval_2 = 0.05, 0.1

							# flag_corr_1=1
							# peak_loc_query_group2_1 = []
							# df_query1['peak_id'] = df_query1.index.copy()

							# query_value_1 = df_query1[column_corr_1]
							# query_value_1 = query_value_1.fillna(0)

							# column_quantile_pre1 = '%s_quantile'%(column_corr_1)
							# normalize_type = 'uniform'	# normalize_type: 'uniform', 'normal'
							# score_mtx = quantile_transform(query_value_1[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
							# df_query1[column_quantile_pre1] = score_mtx[:,0]

							# df_pre2 = df_query1.loc[id_pred1,:]
							# # df_pre2 = df_query1.loc[id_pred2,:]
							# # peak_query_pred2 = df_pre2.index
							# column_score_query1 = '%s.score'%(method_type_feature_link)
							# df_pre2 = df_pre2.sort_values(by=[column_score_query1],ascending=False)

							# query_value = df_pre2[column_corr_1]
							# query_value = query_value.fillna(0)

							# column_quantile_1 = '%s_quantile_2'%(column_corr_1)
							# normalize_type = 'uniform'   # normalize_type: 'uniform', 'normal'
							# score_mtx = quantile_transform(query_value[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
							# df_pre2[column_quantile_1] = score_mtx[:,0]
							
							# query_value_2 = df_pre2[column_score_query1]
							# query_value_2 = query_value_2.fillna(0)

							# column_score_query1 = '%s.score'%(method_type_feature_link)
							# column_quantile_2 = '%s_quantile'%(column_score_query1)
							# normalize_type = 'uniform'   # normalize_type: 'uniform', 'normal'
							# score_mtx_2 = quantile_transform(query_value_2[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
							# df_pre2[column_quantile_2] = score_mtx_2[:,0]

							column_corr_1 = 'peak_tf_corr'
							column_pval = 'peak_tf_pval_corrected'
							method_type_feature_link = select_config['method_type_feature_link']
							column_score_query1 = '%s.score'%(method_type_feature_link)
							column_vec_query = [column_corr_1,column_pval,column_score_query1]

							column_pred1 = '%s.pred'%(method_type_feature_link)
							id_pred1 = (df_query1[column_pred1]>0)
							df_pre2 = df_query1.loc[id_pred1,:]
							df_pre2, select_config = self.test_query_feature_quantile_1(data=df_pre2,query_idvec=[],column_vec_query=column_vec_query,save_mode=1,verbose=verbose,select_config=select_config)

							# df_query1 = df_query1.sort_values(by=['group_overlap',column_quantile_pre1],ascending=False)
							# df_query1 = df_query1.round(7)
							# output_filename = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.2.txt'%(output_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)				
							# df_query1.to_csv(output_filename,sep='\t')

							peak_loc_query_1 = []
							peak_loc_query_2 = []
							flag_corr_1 = 1
							flag_score_1 = 0
							flag_enrichment_sel = 1
							peak_loc_query_group2_1 = self.test_query_training_select_pre1(data=df_pre2,column_vec_query=[],flag_corr_1=flag_corr_1,flag_score_1=flag_score_1,flag_enrichment_sel=flag_enrichment_sel,input_file_path='',
																									save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=verbose,select_config=select_config)
							
							peak_num_group2_1 = len(peak_loc_query_group2_1)
							# if flag_corr_1>0:
							# 	# select peak with peak accessibility-TF expression correlation below threshold
							# 	df_query = df_pre2
							# 	peak_loc_query = df_query.index
							# 	id_score_query2_1 = (df_query[column_corr_1]>thresh_corr_1)&(df_query[column_pval]<thresh_pval_1)

							# 	# query_value_1 = df_query1.loc[id_pred2,column_corr_1]
							# 	query_value = df_query[column_corr_1]
							# 	query_value = query_value_1.fillna(0)
								
							# 	thresh_corr_quantile = 0.90
							# 	thresh_corr_query1 = np.quantile(query_value,thresh_corr_quantile)
							# 	thresh_corr_query2 = np.min([thresh_corr_1,thresh_corr_query1])
							# 	print('thresh_corr_query1, thresh_corr_query2: ',thresh_corr_query1, thresh_corr_query2)
							# 	# id_score_query_2_1 = (df_query[column_corr_1]>thresh_corr_query2)&(df_query[column_pval]<thresh_pval_1)
							# 	id_score_query_2_1 = (df_query[column_corr_1]>thresh_corr_query2)
								
							# 	# thresh_corr_quantile_1 = 0.90
							# 	# id_score_query_2_1 = (df_pre2[column_1]>thresh_corr_quantile_1)

							# 	# id_score_query2 = (id_pred2)&(id_score_query2_1)
							# 	# id_score_query_2 = (id_pred2)&(id_score_query_2_1)

							# 	# peak_loc_query_2_1 = peak_loc_ori[id_score_query2]
							# 	# peak_loc_query_2 = peak_loc_ori[id_score_query_2]

							# 	peak_loc_query_2_1 = peak_loc_query[id_score_query2_1]
							# 	peak_loc_query_2 = peak_loc_query[id_score_query_2_1]

							# 	print('peak_loc_query_2_1, peak_loc_query_2: ',len(peak_loc_query_2_1),len(peak_loc_query_2))
							# 	# peak_loc_query_group2_1 = peak_loc_query_2

							# flag_score_1=0
							# if flag_score_1>0:
							# 	# select peak query based on score threshold
							# 	thresh_score_query_pre1 = 0.15
							# 	thresh_score_query_1 = 0.15
							# 	# thresh_score_query_1 = 0.20
							# 	column_1 = 'thresh_score_group_1'
							# 	if column_1 in select_config:
							# 		thresh_score_group_1 = select_config[column_1]
							# 		thresh_score_query_1 = thresh_score_group_1

							# 	# column_score_query1 = '%s.score'%(method_type_feature_link)
							# 	id_score_query1 = (df_pre2[column_score_query1]>thresh_score_query_1)
							# 	id_score_query1 = (id_score_query1)&(df_pre2[column_corr_1]>thresh_corr_2)
							# 	df_query1_2 = df_pre2.loc[id_score_query1,:]

							# 	peak_loc_query_1 = df_query1_2.index 	# the peak loci with prediction and with score above threshold
							# 	peak_num_1 = len(peak_loc_query_1)
							# 	print('peak_loc_query_1: ',peak_num_1)
							# 	# peak_loc_query_group2_1 = pd.Index(peak_loc_query_1).union(peak_loc_query_2,sort=False)

							# flag_enrichment_sel=1
							# df_query = df_pre2
							# id_group_overlap = (df_query['group_overlap']>0)
							# print('df_query1: ',df_query1.shape)
							# # print(df_query1.columns)
							# print('df_query: ',df_query.shape)
							# # print(df_query.columns)
							# if flag_enrichment_sel>0:
							# 	id1 = (id_group_overlap)
							# 	id2 = (~id_group_overlap)
							# 	# df_query = df_query.sort_values(by=column_score_query1,ascending=False)
							# 	# df_group_pre2 = df_pre2.groupby(by=['group_id2'])
							# 	group_id_query = df_query.loc[id1,'group_id2'].unique()

							# 	thresh_quantile_query1_1, thresh_quantile_query2_1 = 0.25, 0.25
							# 	thresh_quantile_query1_2, thresh_quantile_query2_2 = 0.75, 0.75

							# 	id_score_1 = (df_query[column_quantile_1]>thresh_quantile_query1_1) # lower threshold for group with enrichment
							# 	id_score_2 = (df_query[column_quantile_2]>thresh_quantile_query2_1)

							# 	id1_1 = id1&(id_score_1|id_score_2)
							# 	# id2_1 = 1d2&(id_score_1|id_score_2)

							# 	id_score_1_2 = (df_query[column_quantile_1]>thresh_quantile_query1_2) # higher threshold for group without enrichment
							# 	id_score_2_2 = (df_query[column_quantile_2]>thresh_quantile_query2_2)

							# 	# id1_2 = id1&(id_score_1_2|id_score_2_2)
							# 	id2_2 = id2&(id_score_1_2|id_score_2_2)

							# 	id_query_2 = (id1_1|id2_2)

							# 	thresh_corr_uppper_bound, thresh_corr_lower_bound = 0.95, 0.001
							# 	if thresh_corr_lower_bound>0:
							# 		id_corr_1 = (df_query[column_corr_1].abs()>thresh_corr_lower_bound)
							# 		id_query_2 = id_query_2&(id_corr_1)

							# 	df_query_pre2 = df_query.loc[id_query_2,:]

							# 	peak_loc_query_3 = df_query_pre2.index 	# the peak loci with prediction and with score above threshold
							# 	peak_num_3 = len(peak_loc_query_3)
							# 	print('peak_loc_query_3: ',peak_num_3)

							# peak_loc_query_pre2 = pd.Index(peak_loc_query_2).union(peak_loc_query_1,sort=False)
							# peak_loc_query_group2_1 = pd.Index(peak_loc_query_pre2).union(peak_loc_query_3,sort=False)
							
							# peak_num_group2_1 = len(peak_loc_query_group2_1)
							# print('peak_loc_query_group2_1: ',peak_num_group2_1)

							# df2_query2 = df_overlap_query.loc[df_overlap_query[group_type_2].isin(group_vec_pre2),:]
							# print('field_id, group_vec_query1_pre2: ',field_id1,group_num1_pre2)
							# print(group_vec_query1_pre2)

							# print('field_id, group_vec_query2_pre2: ',field_id1,group_num2_pre2)
							# print(group_vec_query2_pre2)
							
						flag_select_2=1
						# print('df_query1: ',df_query1.shape)
						# print(df_query1.columns)
						# print('df_query: ',df_query.shape)
						# print(df_query.columns)
						if flag_select_2>0:
							# column_vec_query_pre1, column_vec_query_pre2, column_vec_query_pre2_2 = self.test_query_column_method_1(feature_type_vec=feature_type_vec_query,select_config=select_config)
							column_motif_group = 'motif_group_1'
							column_peak_tf_corr_1 = 'group_correlation'
							column_motif_group_corr_1 = 'motif_group_correlation'

							# method_type_feature_link = select_config['method_type_feature_link']
							column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
							column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)

							column_neighbor = ['neighbor%d'%(id1) for id1 in np.arange(1,n_neighbors+1)]
							column_1 = '%s_group_neighbor'%(feature_type_query_1)
							column_2 = '%s_group_neighbor'%(feature_type_query_2)

							column_query1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peak
							column_query2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peak

							column_pred_3 = '%s.pred_group_neighbor'%(method_type_feature_link)
							column_pred_5 = '%s.pred_group_neighbor_1'%(method_type_feature_link)
							column_pred_6 = '%s.pred_neighbor_2_group'%(method_type_feature_link)
							column_pred_7 = '%s.pred_neighbor_2'%(method_type_feature_link)
							column_pred_8 = '%s.pred_neighbor_1'%(method_type_feature_link)

							column_vec_query_pre2 = [column_pred_2,column_query1,column_query2,column_pred_6]
							# column_vec_query_pre2 = [column_pred_2,column_query1,column_query2]
							column_vec_pre2_1 = [column_pred_3,column_pred_6,column_pred_7]
							column_vec_pre2_2 = [column_pred_5,column_pred_6,column_pred_7] # the default column query
							column_vec_pre2_3 = [column_motif_group,column_pred_3,column_pred_6,column_pred_7]

							peak_loc_ori = peak_read.columns
							# df_query1 = df_pre1.loc[peak_loc_ori,:]
							df_pre1 = df_pre1.loc[peak_loc_ori,:]
							df_query1 = df_query1.loc[peak_loc_ori,:]
							column_motif = '%s.motif'%(method_type_feature_link)

							motif_score = df_query1[column_motif]
							id_motif = (df_query1[column_motif].abs()>0)
							df_query1_motif = df_query1.loc[id_motif,:]	# the peak loci with binding motif identified
							peak_loc_motif = df_query1_motif.index
							peak_num_motif = len(peak_loc_motif)
							print('motif_num: ',peak_num_motif)

							id_pred1 = (df_query1[column_pred1]>0)
							# id_pred2 = (df_query1[column_pred2]>0)
							peak_loc_pred1 = df_query1.loc[id_pred1,:]
							# peak_loc_pred2 = df_query1.loc[id_pred2,:

							df_query2_2 = df_query1.loc[(~id_pred1)&id_motif,:]
							peak_loc_query_group2_2_ori = df_query2_2.index  # the peak loci without prediction and with motif
							peak_num_group2_2_ori = len(peak_loc_query_group2_2_ori)
							print('peak_loc_query_group2_2_ori: ',peak_num_group2_2_ori)

							# config_id_2 = 0
							# config_id_2 = select_config['config_id_2']
							config_id_2 = dict_config_annot1[folder_id_query]
							column_query_pre1 = column_vec_query_pre2
							print('config_id_2, motif_id_query: ',config_id_2,motif_id_query,i1)
							if config_id_2%2==0:
								column_query_pre2 = column_vec_query_pre2
								print('use threshold 1 for pre-selection')
							else:
								column_query_pre2 = column_vec_pre2_2
								print('use threshold 2 for pre-selection')
							
							query_num1 = len(column_query_pre1)
							mask_1 = (df_pre1.loc[:,column_query_pre1]>0)
							id_pred1_group = (mask_1.sum(axis=1)>0)	# the peak loci predicted with TF binding by the clustering method
							id1_2 = (~id_pred1_group)

							query_num2 = len(column_query_pre2)
							mask_2 = (df_pre1.loc[:,column_query_pre2]>0)
							id_pred2_group = (mask_2.sum(axis=1)>0)	# the peak loci predicted with TF binding by the clustering method
							id2_2 = (~id_pred2_group)

							thresh_1, thresh_2 = 5, 5
							id_neighbor_1 = (df_pre1[column_query1]>=thresh_1)
							id_neighbor_2 = (df_pre1[column_query2]>=thresh_2)

							flag_neighbor_2_2 = 1
							if flag_neighbor_2_2>0:
								id_neighbor_query1 = (id_neighbor_1&id_neighbor_2)
								id1 = (id_neighbor_1)&(df_pre1[column_query2]>1)
								id2 = (id_neighbor_2)&(df_pre1[column_query1]>1)
								id_neighbor_query2 = (id1|id2)
								id2_2 = (id2_2)&(~id_neighbor_query2)

							# id_pre2 = (id2_2&(~id_pred2))
							id_pre2 = (id2_2&(~id_pred1))
							# peak_group2_ori = peak_loc_ori[id_pre2]
							# peak_group2_1 = peak_loc_ori[id_pre2&id_motif] # not predicted with TF binding but with TF motif scanned
							# peak_group2_2 = peak_loc_ori[id_pre2&(~id_motif)] # not predicted with TF binding and without TF motif scanned

							id_1 = (id_pre2&id_motif)	# not predicted with TF binding but with TF motif scanned
							id_2 = (id_pre2&(~id_motif))	# not predicted with TF binding and without TF motif scanned

							# id_score_query2 = (df_pre1[column_corr_1]>thresh_corr_1)&(df_pre1[column_pval]<thresh_pval_1)

							# select peak with peak accessibility-TF expression correlation below threshold
							id_corr_ = (df_pre1[column_corr_1].abs()<thresh_corr_2)
							id_pval = (df_pre1[column_pval]>thresh_pval_2)
							
							# id_score_query3_1 = (id_score_query3_1&id_pval)
							# id_score_query3_1 = (id_corr_&id_pval)
							id_score_query3_1 = id_corr_
							# id_score_query3_2 = (~id_group)&(df_pre1[column_corr_1].abs()<thresh_corr_3)&(id_pval)
								
							config_group_annot = select_config['config_group_annot']
							id_group = 0
							if 'motif_group_1' in df_pre1.columns:
								id_group = (df_pre1['motif_group_1']>0)
							else:
								config_group_annot = 0

							if config_group_annot>0:
								print('use motif group annotation for peak selection')
								id_score_query3_2 = (~id_group)&(id_corr_)
							else:
								print('without using motif group annotation for peak selection')
								id_score_query3_2 = (id_corr_)

							if (config_id_2>=10):
								print('use group_overlap for pre-selection')
								id_group_overlap_1 = (df_query1['group_overlap']>0)
								id_score_query3_1 = (id_score_query3_1&(~id_group_overlap_1))
								id_score_query3_2 = (id_score_query3_2&(~id_group_overlap_1))

							# df_pre1.loc[id_score_query2,'group_correlation'] = 1
							# id_pre2_1 = (id_score_query3_1)&(~id_pred1)&(id_motif) # the peak loci without prediction and with motif and with peak-TF correlation below threshold
							# id_pre2_2 = (id_score_query3_2)&(~id_motif) # the peak loci without motif and with peak-TF correlation below threshold

							id_pre2_1 = (id_score_query3_1)&(id_pre2)&(id_motif) # the peak loci without prediction and with motif and with peak-TF correlation below threshold
							id_pre2_2 = (id_score_query3_2)&(id_pre2)&(~id_motif) # the peak loci without motif and with peak-TF correlation below threshold
							
							list_query2 = [id_pre2_1,id_pre2_2]
							list_query2_2 = []
							column_corr_abs_1 = '%s_abs'%(column_corr_1)
							column_corr_abs_1 = '%s_abs'%(column_corr_1)
							for i2 in range(2):
								id_query = list_query2[i2]
								df_pre2 = df_pre1.loc[id_query,[column_corr_1,column_pval]].copy()
								df_pre2[column_corr_abs_1] = df_pre2[column_corr_1].abs()
								df_pre2 = df_pre2.sort_values(by=[column_corr_abs_1,column_pval],ascending=[True,False])
								peak_query_pre2 = df_pre2.index
								list_query2_2.append(peak_query_pre2)

							peak_vec_2_1_ori, peak_vec_2_2_ori = list_query2_2[0:2]
							peak_num_2_1_ori = len(peak_vec_2_1_ori) # peak loci in class 2 with motif 
							peak_num_2_2_ori = len(peak_vec_2_2_ori) # peak loci in class 2 without motif
							print('peak_vec_2_1_ori, peak_vec_2_2_ori: ',peak_num_2_1_ori,peak_num_2_2_ori)

							peak_query_vec = peak_loc_query_group2_1  # the peak loci in class 1
							peak_query_num_1 = len(peak_query_vec)
							# ratio_1, ratio_2 = 1.5, 1.0
							# ratio_1, ratio_2 = 1.0, 1.0
							# ratio_1, ratio_2 = 0.5, 1.0
							# if peak_query_num_1>500:
							# 	# ratio_1, ratio_2 = 0.5, 1.5
							# 	ratio_1, ratio_2 = 0.25, 1.75
							# else:
							# 	# ratio_1, ratio_2 = 0.5, 2
							# 	ratio_1, ratio_2 = 0.25, 2

							# ratio_1, ratio_2 = 0.25, 2
							ratio_1, ratio_2 = 0.25, 1.5
							column_1, column_2 = 'ratio_1', 'ratio_2'
							if column_1 in select_config:
								ratio_1 = select_config[column_1]

							if column_2 in select_config:
								ratio_2 = select_config[column_2]

							# ratio_1, ratio_2 = 0.25, 1.75
							# peak_num_2_1 = np.min([int(peak_num_group2_1*ratio_1),peak_num_2_1_ori])
							peak_num_2_1 = np.min([int(peak_query_num_1*ratio_1),peak_num_2_1_ori])
							peak_vec_2_1 = peak_vec_2_1_ori[0:peak_num_2_1]

							# peak_num_2_2 = np.min([int(peak_num_group2_1*ratio_2),peak_num_2_2_ori])
							peak_num_2_2 = np.min([int(peak_query_num_1*ratio_2),peak_num_2_2_ori])
							peak_vec_2_2 = peak_vec_2_2_ori[0:peak_num_2_2]

						if flag_select_1>0:
							df_pre1.loc[peak_query_vec,'class'] = 1
						if flag_select_2>0:
							df_pre1.loc[peak_vec_2_1,'class'] = -1
							df_pre1.loc[peak_vec_2_2,'class'] = -2

						# peak_query_num_1 = len(peak_query_vec)
						print('peak_query_vec: ',peak_query_num_1)
						print('peak_vec_2_1, peak_vec_2_2: ',peak_num_2_1,peak_num_2_2)

						flag_thresh1 = 0
						# select positive and negative group
						if flag_thresh1>0:
							# select peak query based on score threshold
							# the previous selection threshold
							flag_thresh1_1=0
							if flag_thresh1_1>0:
								thresh_score_query_pre1 = 0.15
								thresh_score_query_1 = 0.15
								# thresh_score_query_1 = 0.20
								column_1 = 'thresh_score_group_1'
								if column_1 in select_config:
									thresh_score_group_1 = select_config[column_1]
									thresh_score_query_1 = thresh_score_group_1

								column_score_query1 = '%s.score'%(method_type_feature_link)
								id_score_query1 = (df_pre1[column_score_query1]>thresh_score_query_1)
								id_score_query1 = (id_score_query1)&(df_pre1[column_corr_1]>thresh_corr_2)
								df_query1_2 = df_pre1.loc[(id_pred1&id_score_query1),:]
								
								peak_loc_query_group2_1 = df_query1_2.index 	# the peak loci with prediction and with score above threshold
								peak_num_group2_1 = len(peak_loc_query_group2_1)
								print('peak_loc_query_group2_1: ',peak_num_group2_1)

								df_query2_2 = df_pre1.loc[(~id_pred1)&id_motif,:]
								peak_loc_query_group2_2_ori = df_query2_2.index  # the peak loci without prediction and with motif
								peak_num_group2_2_ori = len(peak_loc_query_group2_2_ori)
								print('peak_loc_query_group2_2_ori: ',peak_num_group2_2_ori)

								id_score_query2 = (df_pre1[column_corr_1]>thresh_corr_1)&(df_pre1[column_pval]<thresh_pval_1)
								id_group = (df_pre1['motif_group_1']>0)
								id_score_query3_1 = (df_pre1[column_corr_1].abs()<thresh_corr_2)
								id_score_query3_2 = (~id_group)&(df_pre1[column_corr_1].abs()<thresh_corr_3)

								df_pre1.loc[id_score_query2,'group_correlation'] = 1
								id_pre2_1 = (id_score_query3_1)&(~id_pred1)&(id_motif) # the peak loci without prediction and with motif and with peak-TF correlation below threshold
								id_pre2_2 = (id_score_query3_2)&(~id_motif) # the peak loci without motif and with peak-TF correlation below threshold
								
								list_query2 = [id_pre2_1,id_pre2_2]
								list_query2_2 = []
								column_corr_abs_1 = '%s_abs'%(column_corr_1)
								column_corr_abs_1 = '%s_abs'%(column_corr_1)
								for i2 in range(2):
									id_query = list_query2[i2]
									df_pre2 = df_pre1.loc[id_query,[column_corr_1,column_pval]].copy()
									df_pre2[column_corr_abs_1] = df_pre2[column_corr_1].abs()
									df_pre2 = df_pre2.sort_values(by=[column_corr_abs_1,column_pval],ascending=[True,False])
									peak_query_pre2 = df_pre2.index
									list_query2_2.append(peak_query_pre2)

								peak_vec_2_1_ori, peak_vec_2_2_ori = list_query2_2[0:2]
								peak_num_2_1_ori = len(peak_vec_2_1_ori)
								peak_num_2_2_ori = len(peak_vec_2_2_ori)
								print('peak_vec_2_1_ori, peak_vec_2_2_ori: ',peak_num_2_1_ori,peak_num_2_2_ori)

								# peak_num_group2_1 = len(peak_loc_query_group2_1)
								# print('peak_loc_query_group2_1: ',peak_num_group2_1)

								# ratio_1, ratio_2 = 1.5, 1.0
								# ratio_1, ratio_2 = 1.0, 1.0
								# ratio_1, ratio_2 = 0.5, 1.0
								ratio_1, ratio_2 = 0.5, 1.5
								# ratio_1, ratio_2 = 0.25, 1.75
								peak_num_2_1 = np.min([int(peak_num_group2_1*ratio_1),peak_num_2_1_ori])
								peak_vec_2_1 = peak_vec_2_1_ori[0:peak_num_2_1]

								peak_num_2_2 = np.min([int(peak_num_group2_1*ratio_2),peak_num_2_2_ori])
								peak_vec_2_2 = peak_vec_2_2_ori[0:peak_num_2_2]

							# peak_num_group2_1 = len(peak_loc_query_group2_1)
							# print('peak_loc_query_group2_1: ',peak_num_group2_1)

							# peak_query_vec = peak_loc_query2  # the peak loci with peak_signal>0 and motif identified by motif scanning
							peak_query_vec = peak_loc_query_group2_1
							df_pre1.loc[peak_query_vec,'group1'] = 1
							df_pre1.loc[peak_vec_2_1,'group1'] = -1
							df_pre1.loc[peak_vec_2_2,'group1'] = -2

						peak_vec_1 = peak_query_vec
						peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)
						sample_id_train = pd.Index(peak_vec_1).union(peak_vec_2,sort=False)

						df_query_pre1 = df_pre1.loc[sample_id_train,:]
						filename_annot2 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
						filename_annot_train_pre1 = filename_annot2

						flag_scale_1 = select_config['flag_scale_1']
						type_query_scale = flag_scale_1

						iter_id1 = 0
						filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)
						filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
						# filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
						# output_filename = '%s/test_query_train.%s.%s.2.txt'%(output_file_path,motif_id1,filename_annot2)
						output_filename = '%s/test_query_train.%s.%s.%s.2.txt'%(output_file_path_query,filename_save_annot_query,motif_id1,filename_save_annot_1)
						df_query_pre1.to_csv(output_filename,sep='\t')

						# flag_shuffle=False
						flag_shuffle=True
						if flag_shuffle>0:
							sample_num_train = len(sample_id_train)
							id_query1 = np.random.permutation(sample_num_train)
							sample_id_train = sample_id_train[id_query1]

						train_valid_mode_2 = 0
						if 'train_valid_mode_2' in select_config:
							train_valid_mode_2 = select_config['train_valid_mode_2']
						if train_valid_mode_2>0:
							sample_id_train_ori = sample_id_train.copy()
							sample_id_train, sample_id_valid, sample_id_train_, sample_id_valid_ = train_test_split(sample_id_train_ori,sample_id_train_ori,test_size=0.1,random_state=0)
						else:
							sample_id_valid = []
						
						sample_id_test = peak_loc_ori
						sample_idvec_query = [sample_id_train,sample_id_valid,sample_id_test]
						# df_query_1 = df_pre1.loc[sample_id_train,:]

						df_pre1[motif_id_query] = 0
						df_pre1.loc[peak_vec_1,motif_id_query] = 1
						# df_pre1.loc[peak_vec_2,motif_id_query] = 0
						peak_num1 = len(peak_vec_1)
						print('peak_vec_1: ',peak_num1)
						print(df_pre1.loc[peak_vec_1,['signal',column_motif,motif_id_query]])

						# feature_type_vec_query = ['latent_peak_motif','latent_peak_tf']
						feature_type_query_1, feature_type_query_2 = feature_type_vec_query[0:2]

						print('motif_id_query, motif_id: ',motif_id_query,motif_id1,i1)
						iter_num = 5
						flag_train1 = 1
						if flag_train1>0:
							# model_type_id1 = 'XGBClassifier'
							# model_type_id1 = 'LogisticRegression'
							# # select_config.update({'model_type_id1':model_type_id1})
							# if 'model_type_id1' in select_config:
							# 	model_type_id1 = select_config['model_type_id1']
							print('feature_type_vec_query: ',feature_type_vec_query)
							key_vec = np.asarray(list(dict_feature.keys()))
							print('dict_feature: ',key_vec)
							peak_loc_pre1 = df_pre1.index
							id1 = (df_pre1['class']==1)
							peak_vec_1 = peak_loc_pre1[id1]
							peak_query_num1 = len(peak_vec_1)

							# train_id1 = 1
							train_id1 = select_config['train_id1']
							flag_scale_1 = select_config['flag_scale_1']
							type_query_scale = flag_scale_1

							file_path_query_pre2 = dict_file_annot2[folder_id_query]
							# output_file_path_query = file_path_query_pre2
							output_file_path_query = '%s/train1'%(file_path_query_pre2)
							output_file_path_query2 = '%s/model_train_1'%(output_file_path_query)
							if os.path.exists(output_file_path_query2)==False:
								print('the directory does not exist: %s'%(output_file_path_query2))
								os.makedirs(output_file_path_query2,exist_ok=True)

							model_path_1 = output_file_path_query2
							select_config.update({'model_path_1':model_path_1})

							select_config.update({'file_path_query_1':file_path_query_pre2})

							filename_prefix_save = 'test_query.%s'%(method_type_group)
							# filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
							iter_id1 = 0
							filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)
							filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
							filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
							# output_filename = '%s/test_query_train.%s.%s.%s.%s.1.txt'%(output_file_path,method_type_group,filename_annot_train_pre1,motif_id1,filename_save_annot_1)
							output_filename = '%s/test_query_train.%s.%s.%s.1.txt'%(output_file_path_query,filename_save_annot_query,motif_id1,filename_save_annot_1)
									
							df_pre2 = self.test_query_compare_binding_train_unit1(data=df_pre1,peak_query_vec=peak_loc_pre1,peak_vec_1=peak_vec_1,motif_id_query=motif_id_query,dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,sample_idvec_query=sample_idvec_query,motif_data=motif_data_query1,flag_scale=flag_scale_1,input_file_path=input_file_path,
																					save_mode=1,output_file_path=output_file_path_query,output_filename=output_filename,filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,verbose=verbose,select_config=select_config)
					stop_1 = time.time()
					print('TF binding prediction for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop_1-start_1))
				
				# except Exception as error:
				# 	print('error! ',error, motif_id_query,motif_id1,motif_id2,i1)
				# 	# return

			if len(list_annot_peak_tf)>0:
				df_annot_peak_tf_1 = pd.concat(list_annot_peak_tf,axis=0,join='outer',ignore_index=False)
				output_filename = '%s/test_query_df_annot.peak_tf.%s.1.txt'%(output_file_path,filename_save_annot2_1)
				df_annot_peak_tf_1.to_csv(output_filename,sep='\t')

			if len(list_score_query_1)>0:
				df_score_query_2 = pd.concat(list_score_query_1,axis=0,join='outer',ignore_index=False)
				filename_save_annot2_1 = '%s.%s'%(method_type_group,data_file_type_query)
				# run_id2 = 1
				# run_id2 = 2
				run_id2 = '2_2'
				output_filename = '%s/test_query_df_score.%s.%s.txt'%(output_file_path,filename_save_annot2_1,run_id2)
				df_score_query_2.to_csv(output_filename,sep='\t')

	## file annotation query
	def test_query_file_annotation_1(self,data=[],method_type_feature_link='',load_mode=0,save_mode=0,verbose=0,select_config={}):

		flag_query1=1
		if flag_query1>0:
			# method_type_feature_link = 'joint_score_pre1'
			# method_type_feature_link = 'joint_score_pre2'
			if load_mode>0:
				dict_file_query = data
				input_filename_query_1 = dict_file_query[method_type_feature_link]
				df_gene_peak_query_1_ori = pd.read_csv(input_filename_query_1,index_col=False,sep='\t')
				print('df_gene_peak_query_1_ori: ',df_gene_peak_query_1_ori.shape)
				print(df_gene_peak_query_1_ori.columns)
				print(input_filename_query_1)

			column_peak_tf_pval = 'peak_tf_pval_corrected'
			column_peak_gene_pval = 'peak_gene_corr_pval'
			# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
			column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
			column_gene_tf_pval = 'gene_tf_pval_corrected'
			list1 = [column_peak_tf_pval,column_peak_gene_pval,column_pval_cond,column_gene_tf_pval]
			
			field_query = ['column_peak_tf_pval','column_peak_gene_pval','column_pval_cond','column_gene_tf_pval']
			select_config, list1 = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=list1,overwrite=False,select_config=select_config)
			column_vec_annot_query1 = list1
			column_vec_annot = ['score_pred1_correlation']+column_vec_annot_query1

			if load_mode>0:
				column_vec = df_gene_peak_query_1_ori.columns
				column_vec_annot = pd.Index(column_vec_annot).difference(column_vec,sort=False)

			data_file_type_query = select_config['data_file_type']
			input_file_path_query = select_config['file_path_motif_score_2']
			filename_prefix_1 = 'test_query_gene_peak.%s.2.pre1.pcorr_query1'%(data_file_type_query)
			
			df_score_annot = []
			if len(column_vec_annot)>0:
				print('column_vec_annot: ',column_vec_annot)
				input_filename_1 = '%s/%s.annot1_1.1.txt.gz'%(input_file_path_query,filename_prefix_1)
				if os.path.exists(input_filename_1)==True:
					df_score_annot = pd.read_csv(input_filename_1,index_col=False,sep='\t')

				if len(df_score_annot)==0:
					print('please provide score annotation file')
					# return
					# feature_query_num = 12500
					# interval = 500
					if data_file_type_query in ['CD34_bonemarrow']:
						feature_query_num = 12500
						interval = 500
					elif data_file_type_query in ['pbmc']:
						feature_query_num = 21528
						interval = 1000
					
					iter_num = int(feature_query_num/interval)
					list1 = []
					for iter_id in range(iter_num):
						# input_filename = '%s/test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.9000_9500.annot1_1.1.txt'%(input_file_path_query)
						start_id1 = iter_id*interval
						start_id2 = (iter_id+1)*interval
						input_filename = '%s/%s.%d_%d.annot1_1.1.txt'%(input_file_path_query,filename_prefix_1,start_id1,start_id2)
						df_1 = pd.read_csv(input_filename,index_col=False,sep='\t')
						list1.append(df_1)
						if (iter_id%10==0):
							print('df_1: ',df_1.shape)
							print(df_1.columns)
							print(df_1[0:2])
							print(input_filename)

					df_score_annot = pd.concat(list1,axis=0,join='outer',ignore_index=False)
					output_file_path = input_file_path_query
					output_filename = '%s/%s.annot1_1.1.txt.gz'%(output_file_path,filename_prefix_1)
					compression = 'gzip'
					# df_score_annot.to_csv(output_filename,index=False,sep='\t',compression=compression)
				
				print('df_score_annot: ',df_score_annot.shape)
				print(df_score_annot.columns)
				print(df_score_annot[0:2])

				column_idvec_1 = ['motif_id','peak_id','gene_id']
				df_score_annot.index = utility_1.test_query_index(df_score_annot,column_vec=column_idvec_1)
				
				if load_mode>0:
					df_gene_peak_query_1_ori.index = utility_1.test_query_index(df_gene_peak_query_1_ori,column_vec=column_idvec_1)

					df_list1 = [df_gene_peak_query_1_ori,df_score_annot]
					column_vec_query_1 = [column_vec_annot]
					df_gene_peak_query_1_ori = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec_1,column_vec=column_vec_query_1,
																				df_list=df_list1,type_id_1=0,type_id_2=0,reset_index=False,select_config=select_config)

					print('df_gene_peak_query_1_ori: ',df_gene_peak_query_1_ori.shape)
					print(df_gene_peak_query_1_ori.columns)

				if save_mode>0:
					b = input_filename_query_1.find('.txt.gz')
					compression = 'gzip'
					output_filename = input_filename_query_1[0:b]+'.copy2_1.txt.gz'
					df_gene_peak_query_1_ori.to_csv(output_filename,index=False,sep='\t',compression=compression)

			if load_mode>0:
				return df_score_annot, df_gene_peak_query_1_ori
			else:
				return df_score_annot

	# feature_link selection
	def test_query_feature_link_select_pre2(self,df_feature_link=[],df_score_annot=[],column_score_vec=['score_pred1','score_pred2'],thresh_query_vec=[],thresh_score_vec=[[0.1,0.05],[0.1,0.05]],thresh_score_vec_2=[0,0.1,0.15],thresh_pval_vec=[0.1,0.1,0.25,0.1,0.01],overwrite=False,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_query1=1
		if flag_query1>0:
			# thresh_score_1,thresh_score_2 = thresh_score_vec[0:2]
			# thresh_query_1 = 0.05
			# thresh_query_2 = 0.05
			thresh_score_1, thresh_query_1 = thresh_score_vec[0]
			thresh_score_2, thresh_query_2 = thresh_score_vec[1]

			column_1, column_2 = 'label_score_1', 'label_score_2'
			column_score_1, column_score_2 = column_score_vec[0:2]
			score_query1 = df_feature_link[column_score_1]
			score_query2 = df_feature_link[column_score_2]

			if (not (column_1 in df_feature_link.columns)) or (overwrite==True):
				# thresh_query_1 = 0.05
				# thresh_query_1 = 0.1
				id1 = (score_query1>thresh_score_1)&(score_query2>thresh_query_1)
				# df_feature_link[column_1] = (df_feature_link[column_score_1]>thresh_score_1).astype(int)
				df_feature_link[column_1] = (id1).astype(int)

			if (not (column_2 in df_feature_link.columns)) or (overwrite==True):
				# thresh_query_2 = 0.05
				# thresh_query_2 = 0.1
				id2 = (score_query2>thresh_score_2)&(score_query1>thresh_query_2)
				# df_feature_link[column_2] = (df_feature_link[column_score_2]>thresh_score_2).astype(int)
				df_feature_link[column_2] = (id2).astype(int)

			df_gene_peak_query_1_ori = df_feature_link
			
			id1 = (df_gene_peak_query_1_ori['label_score_1']>0)
			id2 = (df_gene_peak_query_1_ori['label_score_2']>0)
			
			if len(thresh_score_vec_2)>0:
				thresh_1, thresh_2, thresh_3 = thresh_score_vec_2[0:3]
			else:
				thresh_1, thresh_2 = 0, 0.15
				thresh_3 = 0.10

			column_peak_tf_corr = 'peak_tf_corr'
			column_peak_gene_corr = 'peak_gene_corr_'
			# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
			column_cond = 'gene_tf_corr_peak'
			column_gene_tf_corr = 'gene_tf_corr'

			column_peak_tf_pval = 'peak_tf_pval_corrected'
			column_peak_gene_pval = 'peak_gene_corr_pval'
			# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
			column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
			column_gene_tf_pval = 'gene_tf_pval_corrected'
			
			list1 = [column_peak_tf_pval,column_peak_gene_pval,column_pval_cond,column_gene_tf_pval]
			list1 += [column_peak_tf_corr,column_peak_gene_corr,column_cond,column_gene_tf_corr]
			
			field_query1 = ['column_peak_tf_pval','column_peak_gene_pval','column_pval_cond','column_gene_tf_pval']
			field_query2 = ['column_peak_tf_corr','column_peak_gene_corr','column_cond','column_gene_tf_corr']

			field_query = field_query1+field_query2
			select_config, list1 = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=list1,overwrite=False,select_config=select_config)
			column_vec_annot_query1 = list1
			column_vec_annot = ['score_pred1_correlation']+column_vec_annot_query1

			column_vec = df_gene_peak_query_1_ori.columns
			column_vec_annot = pd.Index(column_vec_annot).difference(column_vec,sort=False)

			if len(column_vec_annot)>0:
				print('column_vec_annot: ',column_vec_annot)
				column_idvec_1 = ['motif_id','peak_id','gene_id']
				if len(df_score_annot)==0:
					print('please provide score annotation file')
					return
				# else:
				# 	df_gene_peak_query_1_ori.index = utility_1.test_query_index(df_gene_peak_query_1_ori,column_vec=column_idvec_1)

				df_list1 = [df_gene_peak_query_1_ori,df_score_annot]				
				df_gene_peak_query_1_ori = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec_1,column_vec=[column_vec_annot],
																			df_list=df_list1,type_id_1=0,type_id_2=0,reset_index=False,select_config=select_config)


				print('df_gene_peak_query_1_ori: ',df_gene_peak_query_1_ori.shape)
				print(df_gene_peak_query_1_ori.columns)

				file_path_motif_score = select_config['file_path_motif_score']

			# id_thresh1 = (df_gene_peak_query_1_ori['score_pred1']>thresh_1)
			id_thresh1 = (df_gene_peak_query_1_ori['score_pred2']>thresh_1)
			id_thresh2 = (df_gene_peak_query_1_ori['score_pred1_correlation']>thresh_2)
			id_thresh_2 = (df_gene_peak_query_1_ori['score_pred1_correlation']>thresh_3)
			id_pre2 = (id1|(id2&id_thresh2))	# combination of selection by label_score_1 and label_score_2
			# id_thresh_vec = [id1,id_pre2,(id1|id2)&id_thresh1,id2&id_thresh_2,id2&id_thresh2]
			id_pre1 = (id1|id2)
			id_thresh_vec = [id1,id_pre2,(id1|id2),id2&id_thresh_2,id2&id_thresh2]

			thresh_num1 = 5
			thresh_query_vec_pre1 = list(np.arange(thresh_num1))+[5,21]
			thresh_query_vec_pre2 = thresh_query_vec_pre1+[-1]
			type_query = 0
			thresh_query_vec_2 = pd.Index(thresh_query_vec).difference(thresh_query_vec_pre2,sort=False)
			if len(thresh_query_vec_2)>0:
				type_query = 1

			if len(thresh_pval_vec)>0:
				thresh_pval_peak_tf = thresh_pval_vec[0]
				thresh_pval_peak_gene = thresh_pval_vec[1]
				thresh_pval_cond = thresh_pval_vec[2]
				thresh_pval_gene_tf = thresh_pval_vec[3]
				thresh_pval_peak_tf_2 = thresh_pval_vec[-1]

				# column_peak_tf_pval = 'peak_tf_pval_corrected'
				# column_peak_gene_pval = 'peak_gene_corr_pval'
				# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
				# column_gene_tf_pval = 'gene_tf_pval_corrected'
				# list1 = [column_peak_tf_pval,column_peak_gene_pval,column_pval_cond,column_gene_tf_pval]
			
				query_num1 = len(field_query1)
				list2 = []
				for i1 in range(query_num1):
					field_id1 = field_query1[i1]
					thresh_value = thresh_pval_vec[i1]
					column_query = select_config[field_id1]
					print('thresh_value: ',thresh_value,column_query)
					id_pval = (df_gene_peak_query_1_ori[column_query]<thresh_value)
					list2.append(id_pval)

				id_pval_1, id_pval_2, id_pval_cond, id_pval_gene_tf = list2

				id_thresh_pval_1 = (id_pval_1|id_pval_cond)
				# id_thresh_pval_query1 = (id1|id2)&(id_thresh_pval_1)

				# id_thresh_pval_1_pre2 = (id_pre1)&(id_pval_1|id_pval_cond)
				id_thresh_pval_1_pre2 = (id_pre1)&(id_thresh_pval_1)	# combination of thresholds on joint score and threshold on the peak-TF correlation or the gene-TF partial correlation given the peak accessibility

				id_thresh_vec = id_thresh_vec + [id_thresh_pval_1,id_thresh_pval_1_pre2]

				if type_query>0:
					# id_thresh_pval_2 = (id_pval_1|(id_pval_2&id_pval_cond)|(id_pval_cond&id_pval_gene_tf))
					id_thresh_pval_2 = (id_pval_1|(id_pval_cond&id_pval_gene_tf))
					# id_thresh_pval_query2 = (id1|id2)&(id_thresh_pval_2)

					id_thresh_pval_3_1 = (id_pval_1&id_pval_2)|(id_pval_cond)
					id_thresh_pval_3_2 = (id_pval_1&id_pval_2)|(id_pval_cond&id_pval_gene_tf)
					id_thresh_pval_3_3 = (id_thresh_pval_1&id_pval_2)
					id_thresh_pval_3_5 = (id_thresh_pval_2&id_pval_2)

					id_thresh_pval_3_6 = (id_pval_1)|(id_pval_cond&id_pval_2)
					id_thresh_pval_3_7 = (id_pval_1)|(id_pval_cond&id_pval_gene_tf&id_pval_2)

					# thresh_value_1 = 0.01
					thresh_value_1 = thresh_pval_peak_tf_2 # stricter p-value threshold for peak-TF correlation
					column_query1 = select_config['column_peak_tf_pval']
					id_pval_pre1 = df_gene_peak_query_1_ori[column_query1]<thresh_value_1

					id_thresh_pval_3_8 = (id_pval_pre1)|(id_thresh_pval_1&id_pval_2)
					id_thresh_pval_3_9 = (id_pval_pre1)|(id_thresh_pval_2&id_pval_2)
					id_thresh_pval_3_10 = (id_pval_1|id_pval_2)
					id_thresh_pval_3_11 = (id_pval_1&id_pval_2)
					id_thresh_pval_3_12 = (id_pval_1) # peak-TF correlation threshold only
					id_thresh_pval_3_13 = (id_pval_2) # peak-gene correlation threshold only
					id_thresh_pval_3_14 = (id_pval_cond)
					id_thresh_pval_3_15 = (id_pval_cond&id_pval_gene_tf)

					id_thresh_pval_query1 = [id_thresh_pval_1,id_thresh_pval_2,id_thresh_pval_3_1,id_thresh_pval_3_2,id_thresh_pval_3_3,id_thresh_pval_3_5,
												id_thresh_pval_3_6,id_thresh_pval_3_7,id_thresh_pval_3_8,id_thresh_pval_3_9,
												id_thresh_pval_3_10,id_thresh_pval_3_11,id_thresh_pval_3_12,id_thresh_pval_3_13,id_thresh_pval_3_14,id_thresh_pval_3_15]

					id_thresh_pval_query2 = [(id_pre1&id_query) for id_query in id_thresh_pval_query1]

					# id_thresh_vec.extend(id_thresh_pval_query1)
					id_thresh_vec.extend(id_thresh_pval_query2)

					thresh_num2 = len(id_thresh_pval_query2)
					thresh_query_vec_pre1 = list(np.arange(thresh_num1))+[5,21]+list(np.arange(22,22+thresh_num2))

			dict_query1 = dict()
			dict_annot1 = dict()
			# if len(thresh_query_vec)==0:
			# 	thresh_num1 = len(id_thresh_query)
			# 	thresh_query_vec = np.arange(thresh_num1)

			# thresh_num1 = 5
			# thresh_query_vec_pre1 = list(np.arange(thresh_num1))+[5,21]
			list_pre1 = thresh_query_vec_pre1
			list1 = thresh_query_vec
			list2 = id_thresh_vec
			query_num_1 = len(list_pre1)
			query_num1 = len(list1)
			query_num2 = len(list2)
			print('thresh_query_vec: ',query_num1,thresh_query_vec)
			print('list2: ',query_num2)
			# print(query_num2,list2)
			dict_annot1 = dict(zip(list_pre1,list2))
			
			for i1 in range(query_num1):
			# for thresh_query in thresh_query_vec:
				# id_thresh_query = id_thresh_vec[0]
				thresh_query = list1[i1]
				if thresh_query>=0:
					# id_thresh_query = list2[i1]
					id_thresh_query = dict_annot1[thresh_query]
					print('thresh_query ',thresh_query)
					# id_thresh_query = id_thresh_vec[thresh_query]
					df_gene_peak_query = df_gene_peak_query_1_ori.loc[id_thresh_query,:]
					# df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=['motif_id','peak_id'])
					df_gene_peak_query.index = np.asarray(df_gene_peak_query['motif_id'])
					dict_query1[thresh_query] = df_gene_peak_query

			thresh_query_1 = -1
			dict_query1[thresh_query_1] = df_gene_peak_query_1_ori

			return dict_query1

	## query feature link
	def test_query_feature_link_pre1_1(self,method_type='',method_type_vec=[],dict_method_type=[],dict_file_query=[],dict_feature_query=[],df_score_annot=[],thresh_query_vec=[],input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			# estimation 2
			data_file_type_query = select_config['data_file_type']
			if len(dict_file_query)==0:
				if data_file_type_query in ['CD34_bonemarrow']:
					file_path_motif_score = select_config['file_path_motif_score_2']
					input_file_path_query = file_path_motif_score

					# input_filename_1 = '%s/test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_2)
					# input_filename_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.txt.gz'%(input_file_path_query,data_file_type_query)
					input_filename_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
					# input_filename_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_2,data_file_type_query)
					# input_filename_3 = '%s/test_query_gene_peak.%s.2.pre1.link_query.2_1.combine.100_0.01.500_-0.05.1.3.2.2.txt.gz'%(input_file_path_2,data_file_type_query)
					# input_filename_2 = 'test_query_gene_peak.CD34_bonemarrow.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'
					input_filename_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
					input_filename_3 = '%s/test_query_gene_peak.%s.2.pre1.link_query.2_1.combine.100_0.01.500_-0.05.1.3.2.2.copy1.txt.gz'%(input_file_path_query,data_file_type_query)
						
					# input_filename_pre1_1 = '%s/test_motif_score_normalize_insilico.CD34_bonemarrow.thresh1.1.txt'%(input_file_path_query)
					# input_filename_pre1_2 = '%s/test_motif_score_normalize_insilico.CD34_bonemarrow.thresh1.1.thresh0.1.txt'%(input_file_path_query)
					input_filename_pre1_1 = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path_query,data_file_type_query)
					input_filename_pre1_2 = '%s/test_motif_score_normalize_insilico.%s.thresh0.1.txt'%(input_file_path_query,data_file_type_query)
					input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.txt.gz'%(input_file_path_query,data_file_type_query)
					# input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.copy2_1.txt.gz'%(input_file_path_query,data_file_type_query)
					input_filename_pre2_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_query,data_file_type_query)

					filename_list2 = [input_filename_1,input_filename_2,input_filename_3]+[input_filename_pre1_1,input_filename_pre1_2,input_filename_pre2_1,input_filename_pre2_2]
					method_type_annot = ['joint_score_1','joint_score_2','joint_score_3']+['insilico','insilico_1','joint_score_pre1','joint_score_pre2']
					dict_file_query = dict(zip(method_type_annot,filename_list2))
					# query_num2 = len(filename_list2)

				elif data_file_type_query in ['pbmc']:
					file_path_motif_score = select_config['file_path_motif_score_2']
					input_file_path_query = file_path_motif_score

					input_filename_pre1_1 = '%s/test_motif_score_normalize_insilico.%s.thresh1.1.txt'%(input_file_path_query,data_file_type_query)
					input_filename_pre1_2 = '%s/test_motif_score_normalize_insilico.%s.thresh1.1.thresh0.1.txt'%(input_file_path_query,data_file_type_query)
					# input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.txt.gz'%(input_file_path_query,data_file_type_query)
					input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.copy2_1.txt.gz'%(input_file_path_query,data_file_type_query)
					input_filename_pre2_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_query,data_file_type_query)

					filename_list2 = [input_filename_pre1_1,input_filename_pre1_2,input_filename_pre2_1,input_filename_pre2_2]
					method_type_annot = ['insilico','insilico_1','joint_score_pre1','joint_score_pre2']
					dict_file_query = dict(zip(method_type_annot,filename_list2))

			compression = 'gzip'
			list_pre2 = []
			if len(method_type_vec)==0:
				key_vec = list(dict_method_type.keys())
				method_type_vec = np.asarray(key_vec)

			# for i2 in range(1,query_num2):
			# for i2 in [1]:
			for method_type_query1 in method_type_vec:
				# input_filename = filename_list2[i2]
				# method_type_query1 = 'joint_score_2'
				# method_type_query1 = method_type_query
				input_filename = dict_file_query[method_type_query1]
				method_type = method_type_query1
				try:
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t',compression=compression) # load feature link
				except Exception as error:
					print('error! ',error)
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t')

				print('df_query: ',df_query.shape)
				print(input_filename)
				print(df_query.columns)
				print(df_query[0:2])
				# list_pre2.append(df_query)
						
				# df_query2, df_query3 = list_pre2[0:2]
				# df_query2 = list_pre2[0]
				# df_link_query_1 = list_pre2[0]
				df_link_query_1 = df_query
				thresh_query_vec = []
				dict_thresh_score = []
				print('method_type: ',method_type)

				if method_type_query1 in dict_method_type:
					dict1 = dict_method_type[method_type_query1]
					thresh_query_vec = dict1['thresh_query_vec']
					if 'thresh_score' in dict1:
						dict_thresh_score = dict1['dict_thresh_score']
				else:
					thresh_query_vec = []
					
				b1 = method_type.find('joint_score')
				b2 = method_type.find('joint_score_2')
				if (b1>-1) or (b2>-1):
					column_score_vec = ['score_pred1','score_pred2']
					if len(dict_thresh_score)>0:
						field_query = ['thresh_score_vec','thresh_score_vec_2','thresh_pval_vec']
						list1 = [dict_thresh_score[field_id] for field_id in field_query]
						thresh_score_vec, thresh_score_vec_2, thresh_pval_vec = list1[0:3]
					else:
						# use the default parameter
						thresh_score_vec_2 = [0.05,0.15,0.10]
						# thresh_score_vec = [[0.10,0.05],[0.10,0.05]]
						thresh_score_vec = [[0.10,0],[0.10,0]]
						# thresh_score_vec_2 = [0.1,0.15,0.10]
						# thresh_score_vec_2 = [0.05,0.15,0.10]
						# thresh_pval_vec = [0.1,0.1,0.25,0.1]
						# the p-value threshold for peak_tf_corr, peak_gene_corr_, gene_tf_corr_peak, gene_tf_corr and stricter threshold for peak_tf_corr
						thresh_pval_vec = [0.1,0.1,0.25,0.1,0.01]
					
					# type_query = 0
					type_query = 1
					overwrite_2 = True

					# perform feature link selection using thresholds
					thresh_query_vec_pre1 = list(np.arange(5))+[5,21]
					dict_feature_link_1 = self.test_query_feature_link_select_pre2(df_feature_link=df_link_query_1,df_score_annot=df_score_annot,column_score_vec=column_score_vec,
																						thresh_query_vec=thresh_query_vec_pre1,thresh_score_vec=thresh_score_vec,
																						thresh_score_vec_2=thresh_score_vec_2,thresh_pval_vec=thresh_pval_vec,overwrite=overwrite_2,
																						save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)
					
					# thresh_query_vec = thresh_query_vec_pre1+[-1]
					thresh_query_vec = [2,5,21]

					# list2 = []
					for thresh_query in thresh_query_vec:
						df_link_query = dict_feature_link_1[thresh_query]
						method_type_query = '%s.thresh%d'%(method_type,thresh_query+1)
						print('df_link_query, method_type_query: ',df_link_query.shape,method_type_query)

						df_gene_peak_query = df_link_query
						if ('gene_id' in df_gene_peak_query):
							df_query = df_gene_peak_query.copy()
							df_query.index = test_query_index(df_query,column_vec=['peak_id','motif_id'])
							df_peak_tf_query = df_query.loc[~df_query.index.duplicated(keep='first'),:]
							print('df_peak_tf_query ',df_peak_tf_query.shape,method_type)
							print(df_peak_tf_query[0:5])

						dict1 = {'peak_tf_gene':df_gene_peak_query,'peak_tf':df_peak_tf_query}
						dict_feature_query.update({method_type_query:dict1})

				# elif (method_type_query in ['insilico','insilico_1']):
				else:
					b_1 = method_type.find('insilico')
					print('b_1: ',b_1)
					if b_1>-1:
						# thresh_insilco_ChIP_seq = 0.1
						column_1 = 'thresh_insilco_ChIP-seq'
						column_score = 'score_pred1'
						for thresh_query in thresh_query_vec:
							method_type_query = 'insilico_%s'%(thresh_query)
							df_link_query = df_link_query_1.loc[df_link_query_1[column_score]>thresh_query,:]
							df_link_query['motif_id'] = np.asarray(df_link_query.index)

							df_gene_peak_query = []
							df_peak_tf_query = df_link_query
							df_peak_tf_query.index = test_query_index(df_peak_tf_query,column_vec=['peak_id','motif_id'])
							df_peak_tf_query = df_peak_tf_query.loc[~df_peak_tf_query.index.duplicated(keep='first'),:]
							print('df_peak_tf_query ',df_peak_tf_query.shape,method_type)
							# print(df_peak_tf_query[0:5])

							dict1 = {'peak_tf_gene':df_gene_peak_query,'peak_tf':df_peak_tf_query}
							dict_feature_query.update({method_type_query:dict1})

			return dict_feature_query

	## load feature link
	def test_query_feature_link_load_pre1(self,data=[],save_mode=1,verbose=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',select_config={}):

		load_mode_query1 = 0
		df_score_annot = self.test_query_file_annotation_1(data=dict_file_query,method_type_feature_link=method_type_feature_link,load_mode=load_mode_query1,save_mode=0,verbose=0,select_config=select_config)
		dict_feature_query = self.test_query_feature_link_pre1_1(method_type=method_type_feature_link_1,method_type_vec=method_type_vec,
																	dict_method_type=dict_method_type,dict_file_query=dict_file_query,
																	dict_feature_query=dict_feature_query,df_score_annot=df_score_annot,
																	thresh_query_vec=thresh_query_vec,input_file_path='',
																	save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',
																	verbose=verbose,select_config=select_config)
			
		# self.dict_file_query = dict_file_query
		self.dict_feature_query = dict_feature_query
		# select_config.update({'dict_file_query':dict_file_query})

		# if load_mode>0:
		# 	return data_vec_1, dict_motif_data, dict_feature_query, select_config
		dict_1 = dict_feature_query[method_type_feature_link]
		df_gene_peak_query1 = dict_1['peak_tf_gene']
		df_peak_tf_query1 = dict_1['peak_tf']

		print('df_gene_peak_query1, df_peak_tf_query1: ',df_gene_peak_query1.shape,df_peak_tf_query1.shape)
		print(df_gene_peak_query1[0:2])
		print(df_peak_tf_query1[0:2])

	## load feature link
	def test_query_feature_link_load_pre2(self,data=[],input_file_path='',save_mode=1,verbose=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',select_config={}):

						if load_mode_pre1_1>0:
							folder_id = folder_id_query
							if folder_id in [1,2]:
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.group2.txt'%(input_file_path_query_1,filename_prefix_1,motif_id1,filename_save_annot_2)
							elif folder_id in [0]:
								upstream_tripod_2 = 100
								filename_save_annot_2_pre2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod_2,type_id_tripod)
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.txt'%(input_file_path_query_1,filename_prefix_1,motif_id1,filename_save_annot_2_pre2)

							if os.path.exists(input_filename==True):
								df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
								peak_loc_1 = df_1.index
								peak_num_1 = len(peak_loc_1)
								print('peak_loc_1: ',peak_num_1)

								method_type_vec_pre2 = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
								# column_motif_vec_2 = ['%s.motif'%(method_type_query) for method_type_query in method_type_vec_2]
								# column_motif_vec_3 = ['%s.pred'%(method_type_query) for method_type_query in method_type_vec_2]
								# column_motif_vec_2 = []
								annot_str_vec = ['motif','pred','score']
								column_vec_query = ['signal']
								for method_type_query in method_type_vec_pre2:
									column_vec_query.extend(['%s.%s'%(method_type_query,annot_str1) for annot_str1 in annot_str_vec])

								# df_query1 = df_1
								df_query1 = df_1.loc[:,column_vec_query]
								print('df_query1: ',df_query1.shape)
								print(df_query1.columns)
								print(df_query1[0:2])
								print(input_filename)
							else:
								print('the file does not exist: %s'%(input_filename))
								df_query1 = pd.DataFrame(index=peak_loc_ori)
								load_mode_pre1_1 = 0

						if load_mode_pre1_1==0:
							# peak_loc_ori = peak_read.columns
							# df_query1 = pd.DataFrame(index=peak_loc_ori)
							column_motif = '%s.motif'%(method_type_feature_link)
							column_motif_2 = '%s_score'%(column_motif)

							motif_data_score_query2 = []
							if len(motif_data_score_query2)>0:
								mask_1 = (motif_data_query1.loc[peak_loc_ori,motif_id_query]>0).astype(int)
								df_query1[column_motif] = mask_1
								df_query1[column_motif_2] = motif_data_score_query2.loc[peak_loc_ori,motif_id_query]
							else:
								df_query1[column_motif] = motif_data_score_query1.loc[peak_loc_ori,motif_id_query]

							df_score_annot = df_feature_link
							method_type_query = method_type_feature_link
							column_name_1 = '%s.pred'%(method_type_query)
							column_name_2 = '%s.score'%(method_type_query)
							column_name_vec = [column_name_1,column_name_2]
							flag_binary = 1
							# column_score_2 = column_score_query
							thresh_type = -1
							thresh_score_1 = -1
							thresh_vec = [thresh_score_1,thresh_type]
							print('thresh_score_1: ',thresh_score_1,method_type_query)
							print('df_score_annot: ',df_score_annot.shape,method_type_query)
							print(df_score_annot.columns)
							print(df_score_annot[0:5])
							column_score_query = 'score_pred1'
							ascending = False
							flag_sort = 0
							flag_unduplicate = 0
							if flag_sort>0:
								df_score_annot = df_score_annot.sort_values(by=column_score_query,ascending=ascending)

							if flag_unduplicate>0:
								df_score_annot = df_score_annot.drop_duplicates(subset=[column_id2,column_id3])

							df_query1 = self.test_query_peak_annot_score_1(data=df_query1,df_annot=df_score_annot,method_type=method_type_query,motif_id=motif_id_query,
																			column_score=column_score_query,column_name_vec=column_name_vec,format_type=0,
																			flag_sort=flag_sort,ascending=ascending,flag_unduplicate=flag_unduplicate,flag_binary=flag_binary,
																			thresh_vec=thresh_vec,
																			save_mode=1,verbose=verbose,select_config=select_config)

	## feature overlap query
	def test_query_feature_overlap_1(self,data=[],motif_id_query='',motif_id1='',column_motif='',df_overlap_compare=[],input_file_path='',save_mode=1,verbose=0,output_file_path='',filename_prefix_save='',filename_save_annot='',select_config={}):

					flag_motif_query=1
					data_file_type_query = select_config['data_file_type']
					method_type_group = select_config['method_type_group']
					df_query1 = data

					if flag_motif_query>0:
						# df_query1_motif = df_query1.loc[id_motif,:] # peak loci with motif
						df_query1_motif = df_query1
						peak_loc_motif = df_query1_motif.index
						# peak_num_motif = len(peak_loc_motif)
						# print('peak_loc_motif ',peak_num_motif)
						# filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
						filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id_query,data_file_type_query)
						
						filename_query = '%s/test_query_df_overlap.%s.motif.1.txt' % (input_file_path, filename_save_annot2_2)
						filename_query_2 = '%s/test_query_df_overlap.%s.motif.2.txt' % (input_file_path, filename_save_annot2_2)
						input_filename = filename_query
						input_filename_2 = filename_query_2
						load_mode_2 = 0
						# overwrite_2 = False
						overwrite_2 = True
						df_group_basic_motif = []
						dict_group_basic_motif = dict()
						if os.path.exists(input_filename)==True:
							if (overwrite_2==False):
								df_overlap_motif = pd.read_csv(input_filename,index_col=0,sep='\t')
								load_mode_2 = load_mode_2+1
						else:
							print('the file does not exist: %s'%(input_filename))

						if os.path.exists(input_filename_2)==True:
							if (overwrite_2==False):
								df_group_basic_motif = pd.read_csv(input_filename_2,index_col=0,sep='\t')
								load_mode_2 = load_mode_2+1
								self.df_group_basic_motif = df_group_basic_motif
						else:
							print('the file does not exist: %s'%(input_filename))

						if load_mode_2<2:
							# output_filename = '%s/test_query_df_overlap.%s.motif.1.txt' % (output_file_path, filename_save_annot2_2)
							# output_filename_2 = '%s/test_query_df_overlap.%s.motif.2.txt' % (output_file_path, filename_save_annot2_2)
							output_filename = filename_query
							output_filename_2 = filename_query_2
							dict_group_basic_1 = self.dict_group_basic_1
							df_group_1 = self.df_group_pre1
							df_group_2 = self.df_group_pre2
							if len(df_overlap_compare)==0:
								df_overlap_compare = self.df_overlap_compare

							stat_chi2_correction = True
							stat_fisher_alternative = 'greater'
							df_overlap_motif, df_overlap_mtx_motif, dict_group_basic_motif = self.test_query_group_overlap_pre1_2(data=df_query1_motif,dict_group_compare=dict_group_basic_1,
																																  	df_group_1=df_group_1,df_group_2=df_group_2,
																																  	df_overlap_1=[],df_query_compare=df_overlap_compare,
																																  	flag_sort=1,flag_group=1,
																																  	stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,
																																	save_mode=1,output_filename=output_filename,output_filename_2=output_filename_2,
																																	verbose=verbose,select_config=select_config)
							self.dict_group_basic_motif = dict_group_basic_motif
						
						self.df_overlap_motif = df_overlap_motif

						load_mode_query = load_mode_2
						return df_overlap_motif, df_group_basic_motif, df_group_basic_motif, load_mode_query

	## feature overlap query
	def test_query_feature_overlap_2(self,data=[],motif_id_query='',motif_id1='',column_motif='',df_overlap_compare=[],input_file_path='',save_mode=1,verbose=0,output_file_path='',filename_prefix_save='',filename_save_annot='',select_config={}):

					flag_select_query=1
					data_file_type_query = select_config['data_file_type']
					method_type_group = select_config['method_type_group']
					method_type_feature_link = select_config['method_type_feature_link']
					# column_pred1 = '%s.pred'%(method_type_feature_link)
					# df_query_1 = data
					# id_pred1 = (df_query_1[column_pred1]>0)
					# df_query1 = df_query_1.loc[id_pred1,:]
					df_query1 = data

					if flag_select_query>0:
						# select the peak loci predicted with TF binding
						df_pred1 = df_query1
						
						peak_loc_query_group2 = df_pred1.index
						peak_num_group2 = len(peak_loc_query_group2)
						print('peak_loc_query_group2: ',peak_num_group2)

						# feature_query_vec_2 = peak_loc_query_group2
						peak_query_vec = peak_loc_query_group2

						# column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
						# column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)
						
						filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id_query,data_file_type_query)
						filename_query = '%s/test_query_df_overlap.%s.pre1.1.txt' % (input_file_path, filename_save_annot2_2)
						filename_query_2 = '%s/test_query_df_overlap.%s.pre1.2.txt' % (input_file_path, filename_save_annot2_2)
						input_filename = filename_query
						input_filename_2 = filename_query_2
						load_mode_2 = 0
						# overwrite_2 = False
						overwrite_2 = True
						df_group_basic_query_2 = []
						dict_group_basic_2 = dict()
						if (os.path.exists(input_filename)==True):
							if (overwrite_2==False):
								df_overlap_query = pd.read_csv(input_filename,index_col=0,sep='\t')
								load_mode_2 = load_mode_2+1
						else:
							print('the file does not exist: %s'%(input_filename))

						if (os.path.exists(input_filename_2)==True):
							if (overwrite_2==False):
								df_group_basic_query_2 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
								load_mode_2 = load_mode_2+1
								self.df_group_basic_query_2 = df_group_basic_query_2
						else:
							print('the file does not exist: %s'%(input_filename_2))

						if load_mode_2<2:
							dict_group_basic_1 = self.dict_group_basic_1
							df_group_1 = self.df_group_pre1
							df_group_2 = self.df_group_pre2
							if len(df_overlap_compare)==0:
								self.df_overlap_compare = df_overlap_compare

							stat_chi2_correction = True
							stat_fisher_alternative = 'greater'
							output_filename = filename_query
							output_filename_2 = filename_query_2
							df_overlap_query, df_overlap_mtx, dict_group_basic_2 = self.test_query_group_overlap_pre1_2(data=df_pred1,dict_group_compare=dict_group_basic_1,
																															df_group_1=df_group_1,df_group_2=df_group_2,
																															df_overlap_1=[],df_query_compare=df_overlap_compare,
																															flag_sort=1,flag_group=1,
																															stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,
																															save_mode=1,output_filename=output_filename,output_filename_2=output_filename_2,
																															verbose=verbose,select_config=select_config)

							# list_query1 = [dict_group_basic_2[group_type] for group_type in group_vec_query]
							list_query1 = []
							group_vec_query = ['group1','group2']
							for group_type in group_vec_query:
								df_query = dict_group_basic_2[group_type]
								df_query['group_type'] = group_type
								list_query1.append(df_query)

							df_query = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
							flag_sort_2=1
							if flag_sort_2>0:
								df_query = df_query.sort_values(by=['group_type','stat_fisher_exact_','pval_fisher_exact_'],ascending=[True,False,True])
							# output_filename = '%s/test_query_df_overlap.%s.%s.pre1.2.txt' % (output_file_path, motif_id1, data_file_type_query)
							output_filename = '%s/test_query_df_overlap.%s.pre1.2.txt' % (output_file_path,filename_save_annot2_2)
							df_query = df_query.round(7)
							df_query.to_csv(output_filename,sep='\t')
							df_group_basic_query_2 = df_query

							self.dict_group_basic_2 = dict_group_basic_2
							
						self.df_overlap_query = df_overlap_query
						
						load_mode_query = load_mode_2
						return df_overlap_query, df_group_basic_query_2, dict_group_basic_2, load_mode_query

	## recompute based on clustering of peak and TF
	# recompute based on training
	def test_query_compare_binding_pre1_5_1_recompute_5(self,data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		run_id1 = select_config['run_id']
		thresh_num1 = 5
		method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh22']

		if 'method_type_feature_link' in select_config:
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_feature_link_1 = method_type_feature_link.split('.')[0]
		
		filename_save_annot = '1'
		method_type_vec = pd.Index(method_type_vec).union([method_type_feature_link],sort=False)

		flag_config_1=1
		if flag_config_1>0:
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
		
			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)
			# file_path_peak_tf = select_config['file_path_peak_tf']
			# file_save_path_1 = select_config['file_path_peak_tf']

		flag_motif_data_load_1 = 1
		if flag_motif_data_load_1>0:
			print('load motif data')
			# method_type_vec_query = ['insilico_1','joint_score_2.thresh3']
			# method_type_vec_query = ['insilico_0.1','joint_score_pre2.thresh3']
			method_type_vec_query = ['insilico_0.1']+[method_type_feature_link]
			method_type_1, method_type_2 = method_type_vec_query[0:2]
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																			select_config=select_config)

			motif_data_query1 = dict_motif_data[method_type_2]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_2]['motif_data_score']
			peak_loc_ori = motif_data_query1.index
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			self.dict_motif_data = dict_motif_data

		flag_load_1 = 1
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')

			# select_config.update({'filename_rna':filename_1,'filename_atac':filename_2,
			# 						'filename_rna_exprs_1':filename_3_ori})
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori(meta_exprs=[],peak_read=[],flag_format=False,select_config=select_config)
			# rna_exprs = meta_scaled_exprs
			# rna_exprs = meta_exprs_2
			# sample_id = rna_exprs.index
			sample_id = meta_scaled_exprs.index
			peak_read = peak_read.loc[sample_id,:]
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			rna_exprs = meta_scaled_exprs
			print('peak_read, rna_exprs: ',peak_read.shape,rna_exprs.shape)

			print(peak_read[0:2])
			print(rna_exprs[0:2])
			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs
			peak_loc_ori = peak_read.columns

		file_save_path_1 = select_config['file_path_peak_tf']
		if data_file_type_query in ['pbmc']:
			dict_file_annot1 = select_config['dict_file_annot1']
			dict_file_annot2 = select_config['dict_file_annot2']
			dict_config_annot1 = select_config['dict_config_annot1']

		folder_id = select_config['folder_id']
		# group_id_1 = folder_id+1
		file_path_query_1 = dict_file_annot1[folder_id] # the first level directory
		file_path_query_2 = dict_file_annot2[folder_id] # the second level directory including the configurations

		input_file_path = file_path_query_2
		output_file_path = file_path_query_2

		folder_id_query = 2 # the folder to save annotation files
		file_path_query1 = dict_file_annot1[folder_id_query]
		file_path_query2 = dict_file_annot2[folder_id_query]
		input_file_path_query = file_path_query1
		input_file_path_query_2 = '%s/vbak1'%(file_path_query1)
		output_file_path_query = file_path_query1

		dict_query_1 = dict()
		feature_type_vec = ['latent_peak_motif','latent_peak_motif_ori','latent_peak_tf','latent_peak_tf_link']
		feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
		type_id_group_2 = select_config['type_id_group_2']
		feature_type_query_1 = feature_type_vec_group[type_id_group_2]
		feature_type_query_2 = 'latent_peak_tf'

		feature_type_vec_query = [feature_type_query_1,feature_type_query_2]
		feature_type_vec_2_ori = []
		prefix_str1 = 'latent_'
		prefix_str1_len = len(prefix_str1)
		for feature_type_query in feature_type_vec_query:
			b = feature_type_query.find(prefix_str1)
			feature_type = feature_type_query[(b+prefix_str1_len):]
			feature_type_vec_2_ori.append(feature_type)

		flag_annot_1 = 1
		# method_type_group = 'MiniBatchKMeans.50'
		method_type_vec_group_ori = ['MiniBatchKMeans.%d'%(n_clusters_query) for n_clusters_query in [30,50,100]]+['phenograph.%d'%(n_neighbors_query) for n_neighbors_query in [10,15,20,30]]
		# method_type_group = 'MiniBatchKMeans.%d'%(n_clusters)
		# method_type_group_id = 1
		# n_neighbors_query = 30
		n_neighbors_query = 20
		method_type_group = 'phenograph.%d'%(n_neighbors_query)
		# method_type_group_id = 6
		if 'method_type_group' in select_config:
			method_type_group = select_config['method_type_group']
		print('method_type_group: ',method_type_group)

		if flag_annot_1>0:
			thresh_size_1 = 100
			if 'thresh_size_group' in select_config:
				thresh_size_group = select_config['thresh_size_group']
				thresh_size_1 = thresh_size_group

			# for selecting the peak loci predicted with TF binding
			# thresh_score_query_1 = 0.125
			# thresh_size_1 = 20
			thresh_score_query_1 = 0.15
			if 'thresh_score_group_1' in select_config:
				thresh_score_group_1 = select_config['thresh_score_group_1']
				thresh_score_query_1 = thresh_score_group_1
			
			thresh_score_default_1 = thresh_score_query_1
			thresh_score_default_2 = 0.10

			peak_distance_thresh_1 = 500
			thresh_fdr_peak_tf_GRaNIE = 0.2
			upstream_tripod = peak_distance_thresh_1
			# type_id_tripod = 0
			# type_id_tripod = 1
			type_id_tripod = select_config['type_id_tripod']
			filename_save_annot_2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod,type_id_tripod)
			# thresh_size_1 = 20

			# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
			# filename_save_annot_2 = filename_save_annot
			filename_save_annot2_ori = '%s.%s.%d.%s'%(filename_save_annot_2,thresh_score_query_1,thresh_size_1,method_type_group)

		flag_group_load_1 = 1
		if flag_group_load_1>0:
			# load feature group estimation for peak or peak and TF
			n_clusters = 50
			peak_loc_ori = peak_read.columns
			feature_query_vec = peak_loc_ori
			# the overlap matrix; the group member number and frequency of each group; the group assignment for the two feature types
			df_overlap_compare, dict_group_basic_1, df_group_1, df_group_2 = self.test_query_feature_group_load_1(data=[],feature_type_vec=feature_type_vec_query,feature_query_vec=feature_query_vec,method_type_group=method_type_group,input_file_path=input_file_path,
																													save_mode=1,output_file_path=output_file_path,filename_prefix_save='',filename_save_annot=filename_save_annot2_ori,output_filename='',verbose=0,select_config=select_config)

			self.df_group_pre1 = df_group_1
			self.df_group_pre2 = df_group_2
			self.dict_group_basic_1 = dict_group_basic_1
			self.df_overlap_compare = df_overlap_compare

		flag_query2 = 1
		if flag_query2>0:
			# select the feature type for group query
			# feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']

			flag_load_2 = 1
			if flag_load_2>0:
				feature_type_1, feature_type_2 = feature_type_vec_2_ori[0:2]
				if feature_type_2 in ['peak_tf']:
					feature_type_vec_2 = [feature_type_1] + ['peak_gene']
				else:
					feature_type_vec_2 = [feature_type_1,feature_type_2]

				method_type_dimension = 'SVD'
				n_components = 50
				type_id_group = select_config['type_id_group']
				filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
				reconstruct = 0
				# load latent matrix;
				# recontruct: 1, load reconstructed matrix;
				flag_combine = 1
				dict_latent_query1 = self.test_query_feature_embedding_load_1(data=[],feature_query_vec=[],motif_id_query='',motif_id='',feature_type_vec=feature_type_vec_2,method_type_vec=[],method_type_dimension=method_type_dimension,
																				n_components=n_components,reconstruct=reconstruct,peak_read=[],rna_exprs=[],flag_combine=flag_combine,
																				load_mode=0,input_file_path='',
																				save_mode=0,output_file_path='',output_filename='',filename_prefix_save=filename_prefix_save_2,filename_save_annot='',
																				verbose=0,select_config=select_config)

				dict_feature = dict_latent_query1

			# n_neighbors = 30
			# n_neighbors = 50
			n_neighbors = 100
			if 'neighbor_num' in select_config:
				n_neighbors = select_config['neighbor_num']
			n_neighbors_query = n_neighbors+1

			# query the neighbors of feature query
			flag_neighbor_query=1
			if flag_neighbor_query>0:
				# list_query1 = [[feature_nbrs_1,dist_nbrs_1],[feature_nbrs_2,dist_nbrs_2]]
				list_query1 = self.test_query_feature_neighbor_load_1(dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,n_neighbors=n_neighbors,input_file_path=input_file_path,
																		save_mode=save_mode,output_file_path=output_file_path,verbose=0,select_config=select_config)

				feature_nbrs_1,dist_nbrs_1 = list_query1[0]
				feature_nbrs_2,dist_nbrs_2 = list_query1[1]

				field_id1, field_id2 = 'feature_nbrs', 'dist_nbrs'
				field_query_1 = [field_id1,field_id2]
				# query_num = len(field_query_1)
				feature_type_num = len(feature_type_vec_query)
				dict_neighbor = dict()
				for i2 in range(feature_type_num):
					feature_type_query = feature_type_vec_query[i2]
					dict_neighbor[feature_type_query] = dict(zip(field_query_1,list_query1[i2]))
				self.dict_neighbor = dict_neighbor
					
			# flag_motif_query_1=1
			flag_motif_query_pre1=0
			if flag_motif_query_pre1>0:
				folder_id = 1
				if 'folder_id' in select_config:
					folder_id = select_config['folder_id']
				df_peak_file, motif_idvec_query = self.test_query_file_annotation_load_1(data_file_type_query=data_file_type_query,folder_id=folder_id,save_mode=1,verbose=verbose,select_config=select_config)

				motif_query_num = len(motif_idvec_query)
				motif_idvec = ['%s.%d'%(motif_id_query,i1) for (motif_id_query,i1) in zip(motif_idvec_query,np.arange(motif_query_num))]
				filename_list1 = np.asarray(df_peak_file['filename'])
				file_num1 = len(filename_list1)
				motif_idvec_2 = []
				for i1 in range(file_num1):
					filename_query = filename_list1[i1]
					b = filename_query.find('.bed')
					motif_id2 = filename_query[0:b]
					motif_idvec_2.append(motif_id2)

				print('motif_idvec_query: ',len(motif_idvec_query),motif_idvec_query[0:5])
			
				# motif_idvec_query = ['ATF2','ATF3','ATF7','BACH1','BACH2','BATF','BATF3']
				# motif_idvec = ['ATF2.0','ATF3.1','ATF7.2','BACH1.3','BACH2.4','BATF.6','BATF3.7']
				# sel_num1 = 12
				sel_num1 = -1
				if sel_num1>0:
					motif_idvec_query = motif_idvec_query[0:sel_num1]
					motif_idvec = motif_idvec[0:sel_num1]
				select_config.update({'motif_idvec_query':motif_idvec_query,'motif_idvec':motif_idvec})

				motif_idvec_query = select_config['motif_idvec_query']
				motif_idvec = select_config['motif_idvec']
				query_num_ori = len(motif_idvec_query)

			input_filename = '%s/test_peak_file.ChIP-seq.pbmc.1_2.2.copy2.txt'%(input_file_path_query_2)
			df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_annot1_1 = df_annot_ori.drop_duplicates(subset=['motif_id2'])
			# folder_id_query1 = select_config['folder_id']
			# id1 = (df_annot1_1['folder_id']==folder_id_query1)
			# df_annot_1 = df_annot1_1.loc[id1,:]
			df_annot_1 = df_annot1_1
			print('df_annot_ori, df_annot_1: ',df_annot_ori.shape,df_annot_1.shape)

			motif_idvec_query = df_annot_1.index.unique()
			motif_idvec_1 = df_annot_1['motif_id1']
			motif_idvec_2 = df_annot_1['motif_id2']
			df_annot_1.index = np.asarray(df_annot_1['motif_id2'])
			dict_motif_query_1 = dict(zip(motif_idvec_2,list(motif_idvec_query)))

			motif_query_num = len(motif_idvec_query)
			motif_num2 = len(motif_idvec_2)
			query_num_ori = len(motif_idvec_2)
			print('motif_idvec_query, motif_idvec_2: ',motif_query_num,motif_num2)

			column_signal = 'signal'
			column_motif = '%s.motif'%(method_type_feature_link)
			column_pred1 = '%s.pred'%(method_type_feature_link)
			column_score_1 = 'score_pred1'
			df_score_annot = []

			column_idvec = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_idvec[0:3]

			# load_mode = 1
			load_mode = 0
			flag_sort = 1
			flag_unduplicate_query = 1
			ascending = False
			column_score_query = 'score_pred1'
			if load_mode>0:
				# copy the score estimation
				if len(df_score_annot)==0:
					# load_mode_query_2 = 1  # 1, load the peak-TF correlation score; 2, load the estimated score 1; 3, load both the peak-TF correlation score and the estimated score 1;
					load_mode_query_2 = 2  # load the estimated score 1
					df_score_annot_query1, df_score_annot_query2 = self.test_query_score_annot_1(data=[],df_score_annot=[],input_file_path='',load_mode=load_mode_query_2,save_mode=0,verbose=0,select_config=select_config)
					df_score_annot = df_score_annot_query2

				if flag_sort>0:
					df_score_annot = df_score_annot.sort_values(by=column_score_query,ascending=ascending)

				if flag_unduplicate_query>0:
					df_score_annot = df_score_annot.drop_duplicates(subset=[column_id2,column_id3])
				df_score_annot.index = np.asarray(df_score_annot[column_id2])

				column_score_query1 = '%s.%s'%(method_type_feature_link,column_score_1)
				# id1 = (df_score_annot[column_id3]==motif_id_query)
				# df_score_annot_query = df_score_annot.loc[id1,:]
				# df_score_annot_query = df_score_annot_query.drop_duplicates(subset=[column_id2,column_id3])
			else:
				column_score_query1 = '%s.score'%(method_type_feature_link)

			motif_query_num = len(motif_idvec_query)
			query_num_1 = motif_query_num
			# query_num_1 = query_num_ori
			# stat_chi2_correction = True
			# stat_fisher_alternative = 'greater'
			list_score_query_1 = []
			interval_save = True
			config_id_load = select_config['config_id_load']
			config_id_2 = select_config['config_id_2']
			config_group_annot = select_config['config_group_annot']
			flag_scale_1 = select_config['flag_scale_1']
			type_query_scale = flag_scale_1

			model_type_id1 = 'LogisticRegression'
			# select_config.update({'model_type_id1':model_type_id1})
			if 'model_type_id1' in select_config:
				model_type_id1 = select_config['model_type_id1']

			beta_mode = select_config['beta_mode']
			# motif_id_1 = select_config['motif_id_1']
			if beta_mode>0:
				# if motif_id_1!='':
				# 	str_vec_1 = motif_id_1.split(',')
				# 	motif_id1 = str_vec_1[0]
				# 	motif_id2 = str_vec_1[1]
				# 	motif_id_query = motif_id1.split('.')[0]
				# 	motif_idvec_query = [motif_id_query]
				# 	motif_idvec = [motif_id1]
				# 	motif_idvec_2 = [motif_id2]	
				iter_vec_1 = [0]
			else:
				iter_vec_1 = np.arange(query_num_1)

			file_path_query_pre1 =  output_file_path_query

			method_type_feature_link = select_config['method_type_feature_link']
			n_neighbors = select_config['neighbor_num']
			peak_loc_ori = peak_read.columns
			# df_pre1_ori = pd.DataFrame(index=peak_loc_ori)
			method_type_group = select_config['method_type_group']

			dict_motif_data = self.dict_motif_data
			method_type_query = method_type_feature_link
			motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']
			# peak_loc_ori = motif_data_query1.index
			
			motif_data_query1 = motif_data_query1.loc[peak_loc_ori,:]
			motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_ori,:]
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			motif_data = motif_data_query1
			motif_data_score = motif_data_score_query1
			# self.dict_motif_data = dict_motif_data

			list_annot_peak_tf = []
			for i1 in iter_vec_1:
				motif_id_query = motif_idvec_query[i1]

				id1 = (df_annot_1['motif_id']==motif_id_query)
				motif_id2_query = df_annot_1.loc[id1,'motif_id2'][0]
				# motif_id_query = df_annot_1.loc[motif_id2_query,'motif_id']
				motif_id1_query = df_annot_1.loc[motif_id2_query,'motif_id1']
				folder_id_query = df_annot_1.loc[motif_id2_query,'folder_id']
				motif_id1, motif_id2 = motif_id1_query, motif_id2_query
				
				folder_id = folder_id_query
				config_id_2 = dict_config_annot1[folder_id]
				select_config.update({'config_id_2_query':config_id_2})

				input_file_path_query_1 = dict_file_annot1[folder_id_query] # the first level directory
				input_file_path_query_2 = dict_file_annot2[folder_id_query] # the second level directory including the configurations

				print('motif_id_query, motif_id1, motif_id2: ',motif_id_query,motif_id1,motif_id2,i1)

				overwrite_2 = False
				filename_prefix_save = 'test_query.%s'%(method_type_group)
				# filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
				iter_id1 = 0
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)

				ratio_1,ratio_2 = select_config['ratio_1'],select_config['ratio_2']
				filename_annot2 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
				filename_annot_train_pre1 = filename_annot2
				filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
				filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
				file_path_query_pre2 = '%s/train1'%(file_path_query_pre1)
				run_id2 = 2
				self.run_id2 = run_id2
				filename_query_pre1 = '%s/test_query_train.%s.%s.%s.%s.txt'%(file_path_query_pre2,filename_save_annot_query,motif_id1,filename_save_annot_1,run_id2)
				
				if (os.path.exists(filename_query_pre1)==True) and (overwrite_2==False):
					print('the file exists: %s'%(filename_query_pre1))
					continue

				flag1=1
				if flag1>0:
					start_1 = time.time()
					filename_prefix_1 = 'test_motif_query_binding_compare'
					# input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					
					# flag_group_query_1 = 0
					flag_group_query_1 = 1
					if flag_group_query_1==0:
						df_query1 = df_pre1
					else:
						load_mode_pre1_1 = 1
						if load_mode_pre1_1>0:
							# load the TF binding prediction file
							# the possible columns: (signal,motif,predicted binding,motif group)
							folder_id = folder_id_query
							if folder_id in [1,2]:
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.group2.txt'%(input_file_path_query_1,filename_prefix_1,motif_id1,filename_save_annot_2)
							elif folder_id in [0]:
								upstream_tripod_2 = 100
								filename_save_annot_2_pre2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod_2,type_id_tripod)
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.txt'%(input_file_path_query_1,filename_prefix_1,motif_id1,filename_save_annot_2_pre2)

							if os.path.exists(input_filename==True):
								df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
								peak_loc_1 = df_1.index
								peak_num_1 = len(peak_loc_1)
								print('peak_loc_1: ',peak_num_1)

								method_type_vec_pre2 = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
								annot_str_vec = ['motif','pred','score']
								column_vec_query = ['signal']
								for method_type_query in method_type_vec_pre2:
									column_vec_query.extend(['%s.%s'%(method_type_query,annot_str1) for annot_str1 in annot_str_vec])

								# df_query1 = df_1
								df_query1 = df_1.loc[:,column_vec_query]
								print('df_query1: ',df_query1.shape)
								print(df_query1.columns)
								print(df_query1[0:2])
								print(input_filename)
							else:
								print('the file does not exist: %s'%(input_filename))
								df_query1 = pd.DataFrame(index=peak_loc_ori)
								load_mode_pre1_1 = 0

					if (flag_group_query_1==0) or (load_mode_pre1_1>0):
						df_query1_ori = df_query1.copy()
						peak_loc_1 = df_query1.index
						column_vec = df_query1.columns
						df_query1 = pd.DataFrame(index=peak_loc_ori)
						df_query1.loc[peak_loc_1,column_vec] = df_query1_ori
						print('df_query1: ',df_query1.shape)

					column_signal = 'signal'
					if column_signal in df_query1.columns:
						# peak_signal = df_query1['signal']
						peak_signal = df_query1[column_signal]
						id_signal = (peak_signal>0)
						# peak_signal_1_ori = peak_signal[id_signal]
						df_query1_signal = df_query1.loc[id_signal,:]	# the peak loci with peak_signal>0
						peak_loc_signal = df_query1_signal.index
						peak_num_signal = len(peak_loc_signal)
						print('signal_num: ',peak_num_signal)

					if not (column_motif in df_query1.columns):
						peak_loc_motif = peak_loc_ori[motif_data[motif_id_query]>0]
						df_query1.loc[peak_loc_motif,column_motif] = motif_data_score.loc[peak_loc_motif,motif_id_query]

					motif_score = df_query1[column_motif]
					id_motif = (df_query1[column_motif].abs()>0)
					df_query1_motif = df_query1.loc[id_motif,:]	# the peak loci with binding motif identified
					peak_loc_motif = df_query1_motif.index
					peak_num_motif = len(peak_loc_motif)
					# print('motif_num: ',peak_num_motif)
					print('peak_loc_motif ',peak_num_motif)
						
					if peak_num_motif==0:
						continue

					flag_motif_query=1
					stat_chi2_correction = True
					stat_fisher_alternative = 'greater'
					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					input_file_path_2 = '%s/folder_group'%(input_file_path_query_1)
					if os.path.exists(input_file_path_2)==False:
						print('the directory does not exist: %s'%(input_file_path_2))
						os.makedirs(input_file_path_2,exist_ok=True)
					output_file_path_2 = input_file_path_2
					if flag_motif_query>0:
						df_query1_motif = df_query1.loc[id_motif,:] # peak loci with motif
						peak_loc_motif = df_query1_motif.index
						peak_num_motif = len(peak_loc_motif)
						print('peak_loc_motif ',peak_num_motif)
						t_vec_1 = self.test_query_feature_overlap_1(data=df_query1_motif,motif_id_query=motif_id_query,motif_id1=motif_id1,
																		column_motif=column_motif,df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,output_file_path=output_file_path_2,filename_prefix_save='',filename_save_annot='',
																		verbose=verbose,select_config=select_config)
						
						df_overlap_motif, df_group_basic_motif, dict_group_basic_motif, load_mode_query1 = t_vec_1
						
					flag_select_query=1
					method_type_query = method_type_feature_link
					column_pred1 = '%s.pred'%(method_type_query)
					id_pred1 = (df_query1[column_pred1]>0)
					df_pre1 = df_query1
					# df_query1_2 = df_query1.loc[id_pred1,:]
					column_pred2 = '%s.pred_sel'%(method_type_query) # selected peak loci with predicted binding sites
					column_pred_2 = '%s.pred_group_2'%(method_type_query)				
					if flag_select_query>0:
						# select the peak loci predicted with TF binding
						id_1 = id_pred1
						df_query1_2 = df_query1.loc[id_1,:] # the selected peak loci
						df_pred1 = df_query1_2
						t_vec_2 = self.test_query_feature_overlap_2(data=df_pred1,motif_id_query=motif_id_query,motif_id1=motif_id1,
																		df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,output_file_path=output_file_path_2,filename_prefix_save='',filename_save_annot='',
																		verbose=verbose,select_config=select_config)
						
						df_overlap_query, df_group_basic_query_2, dict_group_basic_query, load_mode_query2 = t_vec_2
						
						# TODO: automatically adjust the group size threshold
						# df_overlap_query = df_overlap_query.sort_values(by=['freq_obs','pval_chi2_'],ascending=[False,True])
						dict_thresh = dict()
						if len(dict_thresh)==0:
							# thresh_value_overlap = 10
							thresh_value_overlap = 0
							thresh_pval_1 = 0.20
							field_id1 = 'overlap'
							field_id2 = 'pval_fisher_exact_'
							# field_id2 = 'pval_chi2_'
						else:
							column_1 = 'thresh_overlap'
							column_2 = 'thresh_pval_overlap'
							column_3 = 'field_value'
							column_5 = 'field_pval'
							column_vec_query1 = [column_1,column_2,column_3,column_5]
							list_query1 = [dict_thresh[column_query] for column_query in column_vec_query1]
							thresh_value_overlap, thresh_pval_1, field_id1, field_id2 = list_query1

						id1 = (df_overlap_query[field_id1]>thresh_value_overlap)
						id2 = (df_overlap_query[field_id2]<thresh_pval_1)

						df_overlap_query2 = df_overlap_query.loc[id1,:]
						print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape)

						df_overlap_query.loc[id1,'label_1'] = 1
						group_vec_query_1 = np.asarray(df_overlap_query2.loc[:,['group1','group2']])
						print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape,motif_id1)

						self.df_overlap_query = df_overlap_query
						self.df_overlap_query2 = df_overlap_query2
						
						load_mode_2 = load_mode_query2
						if load_mode_2<2:
							# output_filename = '%s/test_query_df_overlap.%s.%s.pre1.1.txt'%(output_file_path,motif_id1,data_file_type_query)
							filename_save_annot2_query = '%s.%s.%s'%(method_type_group,motif_id_query,data_file_type_query)
							output_filename = '%s/test_query_df_overlap.%s.pre1.1.txt'%(output_file_path_2,filename_save_annot2_query)
							df_overlap_query.to_csv(output_filename,sep='\t')

					flag_neighbor_query_1 = 0
					if flag_group_query_1>0:
						flag_neighbor_query_1 = 1
					flag_neighbor = 1
					flag_neighbor_2 = 1  # query neighbor of selected peak in the paired groups

					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					feature_type_vec = feature_type_vec_query
					print('feature_type_vec: ',feature_type_vec)
					select_config.update({'feature_type_vec':feature_type_vec})
					self.feature_type_vec = feature_type_vec
					df_group_1 = self.df_group_pre1
					df_group_2 = self.df_group_pre2

					feature_type_vec = select_config['feature_type_vec']
					feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]
					group_type_vec_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]

					group_type_query1, group_type_query2 = group_type_vec_2[0:2]
					peak_loc_pre1 = df_pre1.index
					df_pre1[group_type_query1] = df_group_1.loc[peak_loc_pre1,method_type_group]
					df_pre1[group_type_query2] = df_group_2.loc[peak_loc_pre1,method_type_group]

					if flag_neighbor_query_1>0:
						# query peak loci predicted with binding sites using clustering
						start = time.time()
						df_overlap_1 = []
						group_type_vec = ['group1','group2']
						# group_vec_query = ['group1','group2']
						list_group_query = [df_group_1,df_group_2]
						dict_group = dict(zip(group_type_vec,list_group_query))
						
						dict_neighbor = self.dict_neighbor
						dict_group_basic_1 = self.dict_group_basic_1
						# the overlap and the selected overlap above count and p-value thresholds
						# group_vec_query_1: the group enriched with selected peak loci
						column_id2 = 'peak_id'
						df_pre1[column_id2] = np.asarray(df_pre1.index)
						df_pre1 = self.test_query_binding_clustering_1(data1=df_pre1,data2=df_pred1,dict_group=dict_group,dict_neighbor=dict_neighbor,dict_group_basic_1=dict_group_basic_1,
																		df_overlap_1=df_overlap_1,df_overlap_compare=df_overlap_compare,
																		group_type_vec=group_type_vec,feature_type_vec=feature_type_vec,group_vec_query=group_vec_query_1,input_file_path='',																												
																		save_mode=1,output_file_path=output_file_path,output_filename='',
																		filename_prefix_save='',filename_save_annot=filename_save_annot2_2,verbose=verbose,select_config=select_config)
						
						stop = time.time()
						print('query feature group and neighbor annotation for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop-start))

					# method_type_vec_pre2 = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
					method_type_vec_2 = ['TRIPOD','GRaNIE','Pando']
					column_motif_vec_2 = ['%s.motif'%(method_type_query) for method_type_query in method_type_vec_2]

					column_score_query_1 = '%s.score'%(method_type_feature_link)
					column_vec_query1 = [column_signal]+column_motif_vec_2+[column_motif,column_pred1,column_score_query_1]
					
					column_vec_query1_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]
					# column_vec_query2_2 = [feature_type_query_1,feature_type_query_2]
					# column_vec_query2 = column_vec_query1 + column_vec_query1_2 + column_vec_query2_2
					column_vec_query2 = column_vec_query1 + column_vec_query1_2

					field_id1 = 'peak_tf_corr'
					field_id2 = 'peak_tf_pval_corrected'
					# query peak accessibility-TF expression correlation
					# flag_peak_tf_corr = 0
					flag_peak_tf_corr = 1
					column_peak_tf_corr = 'peak_tf_corr'
					column_query = column_peak_tf_corr
					if column_query in df_pre1.columns:
						flag_peak_tf_corr = 0

					if flag_peak_tf_corr>0:
						column_value = column_peak_tf_corr
						thresh_value=-0.05
						input_file_path_query1 = '%s/folder_correlation'%(input_file_path_query_1)
						if os.path.exists(input_file_path_query1)==False:
							print('the directory does not exist: %s'%(input_file_path_query1))
							os.makedirs(input_file_path_query1,exist_ok=True)

						output_file_path_query1 = input_file_path_query1
						filename_prefix_save_query = 'test_peak_tf_correlation.%s.%s.2'%(motif_id_query,data_file_type_query)
						df_query1, df_annot_peak_tf = self.test_query_compare_peak_tf_corr_1(data=df_pre1,motif_id_query=motif_id_query,motif_id1=motif_id1,motif_id2=motif_id2,
																								column_signal=column_signal,column_value=column_value,thresh_value=thresh_value,
																								motif_data=motif_data,motif_data_score=motif_data_score,
																								peak_read=peak_read,rna_exprs=rna_exprs,
																								flag_query=0,input_file_path=input_file_path_query1,
																								save_mode=1,output_file_path=output_file_path_query1,
																								filename_prefix_save=filename_prefix_save_query,filename_save_annot='',output_filename='',
																								verbose=verbose,select_config=select_config)

						# list_annot_peak_tf.append(df_annot_peak_tf)

					if load_mode>0:
						if not (column_score_query1 in df_query1.columns):
							id1 = (df_score_annot[column_id3]==motif_id_query)
							df_score_annot_query = df_score_annot.loc[id1,:]
							peak_loc_2 = df_score_annot_query[column_id2].unique()
							df_query1.loc[peak_loc_2,column_score_query1] = df_score_annot_query.loc[peak_loc_2,column_score_1]

					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)

					column_motif_group = 'motif_group_1'
					column_peak_tf_corr_1 = 'group_correlation'
					column_motif_group_corr_1 = 'motif_group_correlation'

					# method_type_feature_link = select_config['method_type_feature_link']
					# n_neighbors = select_config['neighbor_num']
					column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
					column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)
					# column_neighbor = ['neighbor%d'%(id1) for id1 in np.arange(1,n_neighbors+1)]
					column_1 = '%s_group_neighbor'%(feature_type_query_1)
					column_2 = '%s_group_neighbor'%(feature_type_query_2)
					column_query1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peak
					column_query2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peak

					column_pred_3 = '%s.pred_group_neighbor'%(method_type_feature_link)
					column_pred_5 = '%s.pred_group_neighbor_1'%(method_type_feature_link)
					column_pred_6 = '%s.pred_neighbor_2_group'%(method_type_feature_link)
					column_pred_7 = '%s.pred_neighbor_2'%(method_type_feature_link)
					column_pred_8 = '%s.pred_neighbor_1'%(method_type_feature_link)

					column_vec_query_pre1 = [column_motif_group,column_peak_tf_corr_1,column_motif_group_corr_1]
					column_vec_query_pre2 = [column_pred2,column_pred_2,column_pred_3,column_pred_5,column_1,column_2,column_pred_6,column_pred_7,column_pred_8,column_query1,column_query2]
					column_vec_query_pre2_2 = column_vec_query_pre1 + column_vec_query_pre2

					# flag_query2_1=0
					# if flag_query2_1>0:
					# 	flag_motif_ori = 1

					flag_query_2=1
					method_type_feature_link = select_config['method_type_feature_link']
					if flag_query_2>0:
						flag_select_1=1
						# column_pred1 = '%s.pred'%(method_type_feature_link)

						# column_corr_1 = field_id1
						# column_pval = field_id2
						column_corr_1 = 'peak_tf_corr'
						column_pval = 'peak_tf_pval_corrected'
						thresh_corr_1, thresh_pval_1 = 0.30, 0.05
						thresh_corr_2, thresh_pval_2 = 0.1, 0.1
						thresh_corr_3, thresh_pval_2 = 0.05, 0.1

						if flag_select_1>0:
							# select training sample in class 1
							df_annot_vec = [df_group_basic_query_2,df_overlap_query]
							dict_group_basic_2 = self.dict_group_basic_2
							dict_group_annot_1 = {'df_group_basic_query_2':df_group_basic_query_2,'df_overlap_query':df_overlap_query,
													'dict_group_basic_2':dict_group_basic_2}

							key_vec_query = list(dict_group_annot_1.keys())
							for field_id in key_vec_query:
								print(field_id)
								print(dict_group_annot_1[field_id])

							output_file_path_query = file_path_query2
							df_query1 = self.test_query_training_group_pre1(data=df_query1,motif_id1=motif_id1,dict_annot=dict_group_annot_1,
																				method_type_feature_link=method_type_feature_link,
																				dict_thresh=[],thresh_vec=[],input_file_path='',
																				save_mode=1,output_file_path=output_file_path_query,verbose=verbose,select_config=select_config)

							column_corr_1 = 'peak_tf_corr'
							column_pval = 'peak_tf_pval_corrected'
							method_type_feature_link = select_config['method_type_feature_link']
							column_score_query1 = '%s.score'%(method_type_feature_link)
							column_vec_query = [column_corr_1,column_pval,column_score_query1]

							column_pred1 = '%s.pred'%(method_type_feature_link)
							id_pred1 = (df_query1[column_pred1]>0)
							df_pre2 = df_query1.loc[id_pred1,:]
							df_pre2, select_config = self.test_query_feature_quantile_1(data=df_pre2,query_idvec=[],column_vec_query=column_vec_query,save_mode=1,verbose=verbose,select_config=select_config)

							peak_loc_query_1 = []
							peak_loc_query_2 = []
							flag_corr_1 = 1
							flag_score_1 = 0
							flag_enrichment_sel = 1
							peak_loc_query_group2_1 = self.test_query_training_select_pre1(data=df_pre2,column_vec_query=[],
																								flag_corr_1=flag_corr_1,flag_score_1=flag_score_1,
																								flag_enrichment_sel=flag_enrichment_sel,input_file_path='',
																								save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',
																								verbose=verbose,select_config=select_config)
							
							peak_num_group2_1 = len(peak_loc_query_group2_1)
							peak_query_vec = peak_loc_query_group2_1  # the peak loci in class 1
							
						flag_select_2=1
						if flag_select_2>0:
							# select training sample in class 2
							print('feature_type_vec_query: ',feature_type_vec_query)
							peak_vec_2_1, peak_vec_2_2 = self.test_query_training_select_group2(data=df_pre1,motif_id_query=motif_id_query,
																									peak_query_vec_1=peak_query_vec,
																									feature_type_vec=feature_type_vec_query,
																									save_mode=save_mode,verbose=verbose,select_config=select_config)

						if flag_select_1>0:
							df_pre1.loc[peak_query_vec,'class'] = 1
						if flag_select_2>0:
							df_pre1.loc[peak_vec_2_1,'class'] = -1
							df_pre1.loc[peak_vec_2_2,'class'] = -2

						peak_query_num_1 = len(peak_query_vec)
						peak_num_2_1 = len(peak_vec_2_1)
						peak_num_2_2 = len(peak_vec_2_2)
						print('peak_query_vec: ',peak_query_num_1)
						print('peak_vec_2_1, peak_vec_2_2: ',peak_num_2_1,peak_num_2_2)

						peak_vec_1 = peak_query_vec
						peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)
						sample_id_train = pd.Index(peak_vec_1).union(peak_vec_2,sort=False)

						df_query_pre1 = df_pre1.loc[sample_id_train,:]
						filename_annot2 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
						filename_annot_train_pre1 = filename_annot2

						flag_scale_1 = select_config['flag_scale_1']
						type_query_scale = flag_scale_1

						iter_id1 = 0
						filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)
						filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
						# filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
						# output_filename = '%s/test_query_train.%s.%s.2.txt'%(output_file_path,motif_id1,filename_annot2)
						output_filename = '%s/test_query_train.%s.%s.%s.2.txt'%(output_file_path_query,filename_save_annot_query,motif_id1,filename_save_annot_1)
						df_query_pre1.to_csv(output_filename,sep='\t')

						# flag_shuffle=False
						flag_shuffle=True
						if flag_shuffle>0:
							sample_num_train = len(sample_id_train)
							id_query1 = np.random.permutation(sample_num_train)
							sample_id_train = sample_id_train[id_query1]

						train_valid_mode_2 = 0
						if 'train_valid_mode_2' in select_config:
							train_valid_mode_2 = select_config['train_valid_mode_2']
						if train_valid_mode_2>0:
							sample_id_train_ori = sample_id_train.copy()
							sample_id_train, sample_id_valid, sample_id_train_, sample_id_valid_ = train_test_split(sample_id_train_ori,sample_id_train_ori,test_size=0.1,random_state=0)
						else:
							sample_id_valid = []
						
						sample_id_test = peak_loc_ori
						sample_idvec_query = [sample_id_train,sample_id_valid,sample_id_test]
						# df_query_1 = df_pre1.loc[sample_id_train,:]

						df_pre1[motif_id_query] = 0
						df_pre1.loc[peak_vec_1,motif_id_query] = 1
						# df_pre1.loc[peak_vec_2,motif_id_query] = 0
						peak_num1 = len(peak_vec_1)
						print('peak_vec_1: ',peak_num1)
						print(df_pre1.loc[peak_vec_1,['signal',column_motif,motif_id_query]])

						# feature_type_vec_query = ['latent_peak_motif','latent_peak_tf']
						feature_type_query_1, feature_type_query_2 = feature_type_vec_query[0:2]

						print('motif_id_query, motif_id: ',motif_id_query,motif_id1,i1)
						iter_num = 5
						flag_train1 = 1
						if flag_train1>0:
							print('feature_type_vec_query: ',feature_type_vec_query)
							key_vec = np.asarray(list(dict_feature.keys()))
							print('dict_feature: ',key_vec)
							peak_loc_pre1 = df_pre1.index
							id1 = (df_pre1['class']==1)
							peak_vec_1 = peak_loc_pre1[id1]
							peak_query_num1 = len(peak_vec_1)

							# train_id1 = 1
							train_id1 = select_config['train_id1']
							flag_scale_1 = select_config['flag_scale_1']
							type_query_scale = flag_scale_1
							# config_id_2: configuration for selecting class 0 sample
							# flag_scale_1: 0, without feature scaling; 1, with feature scaling
							# output_file_path_query = '%s/train%d'%(output_file_path,train_id1)

							file_path_query_pre2 = dict_file_annot2[folder_id_query]
							# output_file_path_query = file_path_query_pre2
							output_file_path_query = '%s/train1'%(file_path_query_pre2)
							output_file_path_query2 = '%s/model_train_1'%(output_file_path_query)
							if os.path.exists(output_file_path_query2)==False:
								print('the directory does not exist: %s'%(output_file_path_query2))
								os.makedirs(output_file_path_query2,exist_ok=True)

							model_path_1 = output_file_path_query2
							select_config.update({'model_path_1':model_path_1})
							select_config.update({'file_path_query_1':file_path_query_pre2})

							filename_prefix_save = 'test_query.%s'%(method_type_group)
							# filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
							iter_id1 = 0
							filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)
							filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
							filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
							run_id2 = self.run_id2
							output_filename = '%s/test_query_train.%s.%s.%s.%s.txt'%(output_file_path_query,filename_save_annot_query,motif_id1,filename_save_annot_1,run_id2)
									
							df_pre2 = self.test_query_compare_binding_train_unit1(data=df_pre1,peak_query_vec=peak_loc_pre1,peak_vec_1=peak_vec_1,motif_id_query=motif_id_query,dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,sample_idvec_query=sample_idvec_query,motif_data=motif_data_query1,flag_scale=flag_scale_1,input_file_path=input_file_path,
																					save_mode=1,output_file_path=output_file_path_query,output_filename=output_filename,filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,verbose=verbose,select_config=select_config)

					stop_1 = time.time()
					print('TF binding prediction for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop_1-start_1))
				
				# except Exception as error:
				# 	print('error! ',error, motif_id_query,motif_id1,motif_id2,i1)
				# 	# return

			if len(list_annot_peak_tf)>0:
				df_annot_peak_tf_1 = pd.concat(list_annot_peak_tf,axis=0,join='outer',ignore_index=False)
				output_filename = '%s/test_query_df_annot.peak_tf.%s.1.txt'%(output_file_path,filename_save_annot2_1)
				df_annot_peak_tf_1.to_csv(output_filename,sep='\t')

			if len(list_score_query_1)>0:
				df_score_query_2 = pd.concat(list_score_query_1,axis=0,join='outer',ignore_index=False)
				filename_save_annot2_1 = '%s.%s'%(method_type_group,data_file_type_query)
				# run_id2 = 1
				run_id2 = 2
				output_filename = '%s/test_query_df_score.%s.%d.txt'%(output_file_path,filename_save_annot2_1,run_id2)
				df_score_query_2.to_csv(output_filename,sep='\t')

	## TF binding prediction performance comparison
	# load TF binding prediction
	def test_query_binding_pred_compare_pre1(self,data=[],feature_query_vec=[],method_type_vec=[],dict_config_1=[],flag_config_1=1,flag_motif=1,flag_score=1,save_mode=1,verbose=0,select_config={}):

		# flag_config_1=1
		if flag_config_1>0:
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
			data_file_type_query = select_config['data_file_type']
			# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre2.thresh3','joint_score_pre1.thresh22','joint_score_pre2.thresh6']
			# method_type_vec = ['insilico_0.1']+['joint_score_pre1.thresh3','joint_score_pre2.thresh3','joint_score_pre1.thresh22','joint_score_pre2.thresh6']
			# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD'] + ['joint_score_pre2.thresh3','joint_score_pre1.thresh22']
			method_type_vec_pre1 = ['insilico_0.1','GRaNIE','Pando','TRIPOD'] + ['joint_score_pre1.thresh22']
			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec_pre1,flag_config_1=flag_config_1,select_config=select_config)

		thresh_motif = 5e-05
		method_type_feature_link = select_config['method_type_feature_link']
		flag_motif_data_load = 1
		if flag_motif_data_load>0:
			dict_motif_data = self.test_query_motif_data_pre1_1(method_type_vec=method_type_vec,thresh_motif=thresh_motif,save_mode=save_mode,verbose=verbose,select_config=select_config)
			self.dict_motif_data = dict_motif_data

			method_type_query = method_type_feature_link
			motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']

			peak_loc_ori = motif_data_query1.index
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:5])
			print(motif_data_score_query1[0:5])

		# load estimated feature link
		flag_link_query = 1
		if flag_link_query>0:
			dict_feature_query = self.test_query_binding_pred_load_1(data=[],method_type_vec=method_type_vec,save_mode=1,verbose=verbose,select_config=select_config)

		# method_type_vec_query_1 = ['insilico_0.1','GRaNIE','Pando']
		method_type_vec_query_1 = method_type_vec
		if len(method_type_vec)==0:
			method_type_vec_query_1 = ['GRaNIE','Pando']
		method_type_num1 = len(method_type_vec_query_1)

		# motif_query_vec = ['STAT1','IKZF1','RUNX3']
		dict_pre1 = dict()
		load_mode = 0
		if len(feature_query_vec)>0:
			motif_query_vec = feature_query_vec
			motif_query_num = len(motif_query_vec)
			load_mode = 1
			for i1 in range(motif_query_num):
				motif_id_query = motif_query_vec[i1]
				df_pre1 = pd.DataFrame(index=peak_loc_ori)
				dict_pre1.update({motif_id_query:df_pre1})
		
		column_idvec = ['gene_id','peak_id','motif_id']
		column_id1, column_id2, column_id3 = column_idvec[0:3]
		column_idvec_2 = [column_id2,column_id3]
		for i2 in range(method_type_num1):
			method_type_query = method_type_vec_query_1[i2]

			flag_motif_1 = flag_motif
			if flag_motif_1>0:
				dict_motif_data_query = dict_motif_data[method_type_query]
				motif_data_query = dict_motif_data_query['motif_data']
				motif_data_score_query = dict_motif_data[method_type_query]['motif_data_score']
				print('motif_data: ',motif_data_query.shape,method_type_query)
				print(motif_data_query.columns)
				print(motif_data_query[0:5])

				print('motif_data_score: ',len(motif_data_score_query))
				if len(motif_data_score_query)>0:
					print(motif_data_score_query.columns)
					print(motif_data_score_query[0:5])

			flag_score_annot1 = flag_score
			if flag_score_annot1>0:
				column_name_1 = '%s.score'%(method_type_query)
				flag_sort = 0
				flag_unduplicate = 0
				ascending = True
				dict1 = dict_config_1[method_type_query]
				field_query = ['field_id','column_score','thresh_type','thresh_score','column_score_thresh']
				list1 = [dict1[field_id1] for field_id1 in field_query]
				field_id, column_score_query, thresh_type, thresh_score_1, column_score_thresh = list1
				df_feature_link_1 = dict_feature_query[method_type_query][field_id]
	
				df_feature_link = df_feature_link_1.sort_values(by=column_score_query,ascending=True)
				df_feature_link = df_feature_link.drop_duplicates(subset=column_idvec_2)

				if load_mode==0:
					motif_query_vec = df_feature_link[column_id3].unique()
					motif_query_num = len(motif_query_vec)
					print('motif_query_vec, method_type_query: ',motif_query_num,method_type_query)
	
			for i1 in range(motif_query_num):
			# for i1 in range(10):
				motif_id_query = motif_query_vec[i1]
				if motif_id_query in dict_pre1:
					df_pre1 = dict_pre1[motif_id_query]
				else:
					print('motif query not included: ',motif_id_query,i1,method_type_query)
					df_pre1 = pd.DataFrame(index=peak_loc_ori)
					# dict_pre1.update({motif_id_query:df_pre1})
				
				print('motif_id_query: ',motif_id_query,i1,method_type_query)		
				if len(motif_data_score_query)>0:
					df_annot_1 = motif_data_score_query
				else:
					df_annot_1 = motif_data_query

				if flag_motif_1>0:
					column_score_1 = motif_id_query
					column_name_1 = '%s.motif'%(method_type_query) # the binary motif detection
					df_pre1 = self.test_query_peak_annot_motif_1(data=df_pre1,df_annot=df_annot_1,method_type=method_type_query,motif_id=motif_id_query,
																	column_score=column_score_1,column_name=column_name_1,format_type=0,flag_sort=0,
																	save_mode=1,verbose=verbose,select_config=select_config)

				if flag_score_annot1>0:
					df_score_annot = df_feature_link
					column_name_1 = '%s.pred'%(method_type_query)
					column_name_2 = '%s.score'%(method_type_query)
					column_name_vec = [column_name_1,column_name_2]
					flag_binary = 1
					# column_score_2 = column_score_query
					thresh_vec = [thresh_score_1,thresh_type]
					print('thresh_score_1: ',thresh_score_1,method_type_query)
					print('df_score_annot: ',df_score_annot.shape,method_type_query)
					print(df_score_annot.columns)
					print(df_score_annot[0:5])
					df_pre1 = self.test_query_peak_annot_score_1(data=df_pre1,df_annot=df_score_annot,method_type=method_type_query,motif_id=motif_id_query,
																	column_score=column_score_query,column_name_vec=column_name_vec,format_type=0,
																	flag_sort=flag_sort,ascending=ascending,flag_unduplicate=flag_unduplicate,flag_binary=flag_binary,
																	thresh_vec=thresh_vec,
																	save_mode=1,verbose=verbose,select_config=select_config)

					print('df_pre1: ',df_pre1.shape)
					print(df_pre1.columns)
					print(df_pre1[0:2])

				dict_pre1.update({motif_id_query:df_pre1})

		return dict_pre1

	## TF binding prediction performance comparison
	def test_query_binding_pred_compare_1(self,data=[],feature_query_vec=[],input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_config_1=1
		data_file_type_query = select_config['data_file_type']
		if flag_config_1>0:
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_vec_pre1 = ['insilico_0.1','GRaNIE','Pando','TRIPOD'] + [method_type_feature_link]
			# method_type_vec = ['GRaNIE','Pando']
			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec_pre1,flag_config_1=flag_config_1,select_config=select_config)

		file_save_path_1 = select_config['file_path_peak_tf']
		if data_file_type_query in ['pbmc']:
			dict_file_annot1 = select_config['dict_file_annot1']
			dict_file_annot2 = select_config['dict_file_annot2']
			dict_config_annot1 = select_config['dict_config_annot1']

		folder_id_query = 2 # the folder to save annotation files
		file_path_query1 = dict_file_annot1[folder_id_query]
		file_path_query2 = dict_file_annot2[folder_id_query]
		input_file_path_query = file_path_query1
		input_file_path_query_2 = '%s/vbak1'%(file_path_query1)
		output_file_path = '%s/folder_save_1'%(input_file_path_query_2)
		if os.path.exists(output_file_path)==False:
			print('the directory does not exist: %s'%(output_file_path))
			os.makedirs(output_file_path,exist_ok=True)

		input_filename = '%s/test_peak_file.ChIP-seq.pbmc.1_2.2.copy2.txt'%(input_file_path_query_2)
		df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
		df_annot1_1 = df_annot_ori.drop_duplicates(subset=['motif_id2'])
		# folder_id_query1 = select_config['folder_id']
		# id1 = (df_annot1_1['folder_id']==folder_id_query1)
		# df_annot_1 = df_annot1_1.loc[id1,:]
		df_annot_1 = df_annot1_1
		print('df_annot_ori, df_annot_1: ',df_annot_ori.shape,df_annot_1.shape)

		motif_idvec_query = df_annot_1.index.unique()
		motif_idvec_1 = df_annot_1['motif_id1']
		motif_idvec_2 = df_annot_1['motif_id2']
		df_annot_1.index = np.asarray(df_annot_1['motif_id2'])
			
		motif_query_num = len(motif_idvec_query)
		motif_num2 = len(motif_idvec_2)
		query_num_ori = len(motif_idvec_2)
		print('motif_idvec_query, motif_idvec_2: ',motif_query_num,motif_num2)
		# feature_query_vec = motif_idvec_query

		flag_config_1=0
		flag_motif=1
		flag_score=1
		# motif_query_vec = ['STAT1','IKZF1','RUNX3']
		# feature_query_vec = motif_query_vec
		# feature_query_num = len(feature_query_vec)

		# method_type_feature_link = select_config['method_type_feature_link']
		# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD'] + [method_type_feature_link]
		# load TF binding prediction
		column_score_thresh_1 = 'thresh_fdr_peak_tf_GRaNIE'
		column_score_thresh_2 = 'padj_thresh_1'
		thresh_fdr_peak_tf_GRaNIE = 0.2
		padj_thresh_1 = 0.05
		# padj_thresh_1 = 0.1
		padj_thresh_save = 1.0
		column_score_save_2 = 'padj_thresh_save'
		select_config.update({column_score_thresh_1:thresh_fdr_peak_tf_GRaNIE,column_score_thresh_2:padj_thresh_1,
								column_score_save_2:padj_thresh_save})

		field_query = ['field_id','column_score','thresh_type','thresh_score','column_score_thresh']

		method_type_vec_1 = ['GRaNIE','Pando']
		method_type_num1 = len(method_type_vec_1)
		method_type_vec_query = method_type_vec_1
		dict_config_1 = dict()
		for i1 in range(method_type_num1):
			method_type_query = method_type_vec_1[i1]
			if method_type_query in ['GRaNIE']:
				field_id = 'peak_tf_2'
				column_score_query = 'score_pred1'
				thresh_type = 1 # thresh type: 0, above threshold; 1, below threshold
				# thresh_score_1 = 0.2 # the threshold for binary prediction
				thresh_score_1 = 0.3 # the threshold for binary prediction
				column_score_thresh = 'thresh_fdr_peak_tf_GRaNIE'
				thresh_fdr_peak_tf_GRaNIE = thresh_score_1
				select_config.update({column_score_thresh:thresh_score_1})

			elif method_type_query in ['Pando']:
				field_id = 'peak_tf_gene'
				column_score_query = 'score_pred1'
				thresh_type = 1
				# thresh_score_1 = 0.05
				thresh_score_1 = 0.1
				column_score_thresh = 'padj_thresh_1'
				padj_thresh_1 = thresh_score_1
				select_config.update({column_score_thresh:thresh_score_1})

			list1 = [field_id,column_score_query,thresh_type,thresh_score_1,column_score_thresh]
			dict1 = dict(zip(field_query,list1))
			dict_config_1[method_type_query] = dict1

		# flag_query1=1
		flag_query1=0
		feature_query_vec = motif_idvec_query
		feature_query_num = len(feature_query_vec)

		if flag_query1>0:
			# feature_query_vec = []
			dict_pre1 = self.test_query_binding_pred_compare_pre1(data=[],feature_query_vec=feature_query_vec,method_type_vec=method_type_vec_query,dict_config_1=dict_config_1,
																	flag_config_1=flag_config_1,flag_motif=flag_motif,flag_score=flag_score,
																	save_mode=1,verbose=verbose,select_config=select_config)

			if filename_prefix_save=='':
				filename_prefix_save = 'test_query_binding'

			feature_query_vec = list(dict_pre1.keys())
			feature_query_num = len(feature_query_vec)
			list_query1 = []
			for i1 in range(feature_query_num):
				feature_query = feature_query_vec[i1]
				df_pre1 = dict_pre1[feature_query]
				if save_mode>0:
					output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_save,feature_query)
					df_pre1.to_csv(output_filename,sep='\t')
					method_type_vec_2 = ['GRaNIE','Pando']
					column_vec_1 = df_pre1.columns
					
					column_vec_query = ['%s.score'%(method_type_query) for method_type_query in method_type_vec_2]
					column_vec_query1 = pd.Index(column_vec_query).intersection(column_vec_1,sort=False)
					id1 = (df_pre1.loc[:,column_vec_query1].sum(axis=1)>0)
					df_pre2 = df_pre1.loc[id1,:]
					output_filename = '%s/%s.%s.2.txt'%(output_file_path,filename_prefix_save,feature_query)
					df_pre2.to_csv(output_filename,sep='\t')

					column_vec_query_2 = ['%s.pred'%(method_type_query) for method_type_query in method_type_vec_2]
					column_vec_query2 = pd.Index(column_vec_query_2).intersection(column_vec_1,sort=False)
					id2 = (df_pre1.loc[:,column_vec_query2].sum(axis=1)>0)
					df_pre3 = df_pre1.loc[id2,:]
					df_pre3['motif_id'] = feature_query
					list_query1.append(df_pre3)
			
			if save_mode>0:
				df_query = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
				t_vec_1 = df_query.max(axis=0)
				print('t_vec_1: ',t_vec_1)
				# output_filename = '%s/%s.combine.2_2.txt'%(output_file_path,filename_prefix_save)
				output_filename = '%s/%s.combine.2_2_1.txt'%(output_file_path,filename_prefix_save)
				df_query.to_csv(output_filename,sep='\t')

		flag_query2=1
		if flag_query2>0:
			# dict_file_annot1 = select_config['dict_file_annot1']
			# dict_file_annot2 = select_config['dict_file_annot2']

			dict_config_annot1 = select_config['dict_config_annot1']
			folder_id_query = 2 # the folder to save annotation files
			file_path_query1 = dict_file_annot1[folder_id_query]
			file_path_query2 = dict_file_annot2[folder_id_query]
			input_file_path_query = file_path_query1
			input_file_path_query_2 = '%s/vbak1'%(file_path_query1)

			input_filename = '%s/test_peak_file.ChIP-seq.pbmc.1_2.2.copy2.txt'%(input_file_path_query_2)
			df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_annot1_1 = df_annot_ori.drop_duplicates(subset=['motif_id2'])
			# folder_id_query1 = select_config['folder_id']
			# id1 = (df_annot1_1['folder_id']==folder_id_query1)
			# df_annot_1 = df_annot1_1.loc[id1,:]
			df_annot_1 = df_annot1_1
			print('df_annot_ori, df_annot_1: ',df_annot_ori.shape,df_annot_1.shape)

			motif_idvec_query = df_annot_1.index.unique()
			motif_idvec_1 = df_annot_1['motif_id1']
			motif_idvec_2 = df_annot_1['motif_id2']
			df_annot_1.index = np.asarray(df_annot_1['motif_id2'])
			
			motif_query_num = len(motif_idvec_query)
			motif_num2 = len(motif_idvec_2)
			query_num_ori = len(motif_idvec_2)
			print('motif_idvec_query, motif_idvec_2: ',motif_query_num,motif_num2)

			input_file_path_2 = '%s/folder_save_1'%(input_file_path_query_2)
			# feature_query_vec = motif_idvec_query
			feature_query_vec = motif_idvec_2
			feature_query_num = len(feature_query_vec)

			# filename_prefix_save_1 = 'test_query_train.phenograph.20.0.25_1.5.10.neighbor100.ZEB1.82.pbmc.0.1.1.copy1_1.txt'
			
			ratio_1,ratio_2 = select_config['ratio_1'],select_config['ratio_2']
			# filename_annot2 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
			method_type_group = select_config['method_type_group']
			n_neighbors = select_config['neighbor_num']
			# filename_prefix_save_1 = 'test_query_train.phenograph.20.0.25_1.5.10.neighbor100'

			# filename_annot_1 = 'pbmc.0.1.1.copy1_1'
			filename_save_annot_1 = '%s.0.1.1.copy1_1'%(data_file_type_query)
			filename_prefix_save_2 = 'test_query_binding'
			dict_file_query = dict()

			method_type_vec_query = ['GRaNIE','Pando']
			method_type_vec_2 = method_type_vec_query.copy()
			list1 = []
			annot_str_vec = ['pred','score']
			for method_type_query in method_type_vec_query:
				# for annot_str1 in annot_str_vec:
				t_vec_1 = ['%s.%s'%(method_type_query,annot_str1) for annot_str1 in annot_str_vec]
				list1.append(t_vec_1)

			column_vec_query = list1
			column_vec_query1 = np.ravel(column_vec_query)
			column_vec_query_1 = [t_vec_1[0] for t_vec_1 in column_vec_query]
			dict_method_type = dict(zip(column_vec_query_1,method_type_vec_query))
			print('dict_method_type: ',dict_method_type)

			thresh_vec_query1 = [0.15,0.2,0.25,0.5]
			thresh_vec_query2 = [0.25]
			print('dict_method_type: ',dict_method_type)

			column_signal = 'signal'
			column_motif_pre1 = '%s.motif'%(method_type_feature_link)

			list_score_query_1 = []
			for i1 in range(feature_query_num):
				feature_query = feature_query_vec[i1]
				# df_pre1 = dict_pre1[feature_query]
				motif_id2 = feature_query

				motif_id_query = df_annot_1.loc[motif_id2,'motif_id']
				motif_id1 = df_annot_1.loc[motif_id2,'motif_id1']
				folder_id = df_annot_1.loc[motif_id2,'folder_id']

				config_id_2 = dict_config_annot1[folder_id]
				# filename_annot_train_pre1 = filename_annot2
				filename_annot_train_pre1 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
				filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)

				file_path_query_2 = dict_file_annot2[folder_id]
				input_file_path_1 = file_path_query_2
				# input_filename_1 = '%s/%s.%s.%s.txt'%(input_file_path_1,filename_prefix_save_1,motif_id1,filename_annot_1)
				input_filename_1 = '%s/test_query_train.%s.%s.%s.txt'%(input_file_path_1,filename_save_annot_query,motif_id1,filename_save_annot_1)
				input_filename_2 = '%s/%s.%s.1.txt'%(input_file_path_2,filename_prefix_save_2,motif_id_query)
				
				if os.path.exists(input_filename_1)==False:
					print('the file does not exist: ',input_filename_1)
					filename_prefix_1 = 'test_query_train.%s'%(filename_save_annot_query)
					run_id2 = 1
					filename_save_annot1_1 = '%s.0.1.%d'%(data_file_type_query,run_id2)
					input_filename_1 = '%s/%s.%s.%s.txt'%(input_file_path_1,filename_prefix_1,motif_id1,filename_save_annot1_1)
					if os.path.exists(input_filename_1)==False:
						print('the file does not exist: ',input_filename_1)
						continue

				select_config.update({'file_path_query_1':file_path_query_2})

				# input_filename_1 = '%s/%s.%s.%s.txt'%(input_file_path_1,filename_prefix_save_1,motif_id1,filename_annot_1)
				# input_filename_2 = '%s/%s.%s.1.txt'%(input_file_path_2,filename_prefix_save_2,motif_id_query)

				dict_file_query.update({feature_query:[input_filename_1,input_filename_2]})
				df_1_ori = pd.read_csv(input_filename_1,index_col=0,sep='\t')
				df_2 = pd.read_csv(input_filename_2,index_col=0,sep='\t')

				peak_loc_ori = df_2.index
				id1 = (~df_1_ori.index.duplicated(keep='first'))
				df_1 = df_1_ori.loc[id1,:]
				df_1 = df_1.loc[peak_loc_ori,:]
				column_vec_1 = [column_signal,column_motif_pre1]
				column_vec_2 = column_vec_query1
				# column_vec_2 = list(column_vec_2) + ['%s.motif'%(method_type_query) for method_type_query in method_type_vec_2]

				df_query1 = df_1.loc[:,column_vec_1]
				df_query2 = df_2.loc[:,column_vec_2]

				method_type_query1 = 'Pando'
				score_type_vec = [1,1]
				method_type_query2 = 'GRaNIE'
				df_pre1 = pd.concat([df_query1,df_query2],axis=1,join='outer',ignore_index=False)
				print('df_pre1: ',df_pre1.shape)
				print(df_pre1.columns)
				print(df_pre1[0:2])

				# group_query_vec = [2,1,0]
				# # group_query_vec = [2,0]
				# score_type = 1
				group_query_vec = [2]
				# score_type_vec = [1,1]
				log_transform = False
				# log_transform = True
				df_score_query_1, dict_query_1 = self.test_query_compare_binding_pre1_5_1_basic_2_unit1_2(data=df_pre1,dict_feature=[],
																			motif_id_query=motif_id_query,motif_id_1=motif_id1,motif_id_2=motif_id2,
																			group_query_vec=group_query_vec,
																			column_signal=column_signal,column_motif=column_motif_pre1,
																			column_vec_query=column_vec_query,
																			dict_method_type=dict_method_type,
																			score_type_vec=score_type_vec,log_transform=log_transform,
																			feature_type_vec=[],
																			method_type_vec=method_type_vec_query,method_type_group='',flag_compare_1=0,
																			type_id_1=0,load_mode=0,input_file_path='',
																			save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

				print('df_score_query_1: ',df_score_query_1.shape)
				print(df_score_query_1.columns)
				print(df_score_query_1[0:5])
				list_score_query_1.append(df_score_query_1)

			df_score_query_2 = pd.concat(list_score_query_1,axis=0,join='outer',ignore_index=False)
			output_file_path = input_file_path_2
			# output_filename = '%s/%s.score.1.txt'%(output_file_path,filename_prefix_save_2)
			# run_id2 = 1
			# run_id2 = 2
			run_id2 = 3
			output_filename = '%s/%s.score.%d.thresh%s_%s.%d.txt'%(output_file_path,filename_prefix_save_2,int(log_transform),thresh_fdr_peak_tf_GRaNIE,padj_thresh_1,run_id2)
			df_score_query_2 = df_score_query_2.round(7)
			df_score_query_2.to_csv(output_filename,sep='\t')

		return df_score_query_2

	## TF binding prediction performance
	def test_query_compare_binding_pre1_5_1_basic_1(self,data=[],type_query=0,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		run_id1 = select_config['run_id']

		# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh22']
		method_type_feature_link = select_config['method_type_feature_link']
		method_type_group = select_config['method_type_group']

		method_type_vec_1 = ['insilico_0.1','GRaNIE','Pando','TRIPOD']
		method_type_vec_query = [method_type_feature_link]
		method_type_vec = method_type_vec_1 + [method_type_feature_link]

		flag_config_1=1
		if flag_config_1>0:
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)

		file_save_path_1 = select_config['file_path_peak_tf']
		if data_file_type_query in ['pbmc']:
			dict_file_annot1 = select_config['dict_file_annot1']
			dict_file_annot2 = select_config['dict_file_annot2']
			dict_config_annot1 = select_config['dict_config_annot1']

		folder_id_query = 2 # the folder to save annotation files
		file_path_query1 = dict_file_annot1[folder_id_query]
		file_path_query2 = dict_file_annot2[folder_id_query]
		input_file_path_query = file_path_query1
		input_file_path_query_2 = '%s/vbak1'%(file_path_query1)

		dict_query_1 = dict()
		# feature_type_vec = ['latent_peak_motif','latent_peak_motif_ori','latent_peak_tf','latent_peak_tf_link']
		feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
		# type_id_group_2 = select_config['type_id_group_2']
		type_id_group_2 = 1
		feature_type_query_1 = feature_type_vec_group[type_id_group_2]
		feature_type_query_2 = 'latent_peak_tf'
		feature_type_vec_query = [feature_type_query_1,feature_type_query_2]

		column_signal = 'signal'
		column_motif = '%s.motif'%(method_type_feature_link)

		# query the column of the prediction
		model_type_id1 = 'LogisticRegression'
		method_type_query1 = method_type_feature_link
		method_type_query2 = 'latent_peak_motif_peak_gene_combine_%s'%(model_type_id1)
		method_type_annot1 = 'Unify'
		method_type_annot2 = 'REUNION'

		annot_str_vec = ['pred','proba_1']
		t_vec_1 = ['%s.pred'%(method_type_query1),[]]
		t_vec_2 = ['%s_%s'%(method_type_query2,annot_str1) for annot_str1 in annot_str_vec]
		column_vec_query = [t_vec_1,t_vec_2]
		column_vec_query1 = [column_motif] + [t_vec_1[0]] + t_vec_2

		method_type_vec_query1 = [query_vec[0] for query_vec in column_vec_query]
		method_type_vec_annot1 = [method_type_annot1,method_type_annot2]
		dict_method_type = dict(zip(method_type_vec_query1,method_type_vec_annot1))

		input_filename = '%s/test_peak_file.ChIP-seq.pbmc.1_2.2.copy2.txt'%(input_file_path_query_2)
		df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
		df_annot1_1 = df_annot_ori.drop_duplicates(subset=['motif_id2'])
		# folder_id_query1 = select_config['folder_id']
		# id1 = (df_annot1_1['folder_id']==folder_id_query1)
		# df_annot_1 = df_annot1_1.loc[id1,:]
		df_annot_1 = df_annot1_1
		print('df_annot_ori, df_annot_1: ',df_annot_ori.shape,df_annot_1.shape)

		motif_idvec_query = df_annot_1.index.unique()
		motif_idvec_1 = df_annot_1['motif_id1']
		motif_idvec_2 = df_annot_1['motif_id2']
		df_annot_1.index = np.asarray(df_annot_1['motif_id2'])
		dict_motif_query_1 = dict(zip(motif_idvec_2,list(motif_idvec_query)))

		# dict_motif = []
		df_annot_motif = df_annot_1
		print('df_annot_motif: ',df_annot_motif.shape)
		print(df_annot_motif)
		print('dict_method_type: ',dict_method_type)

		motif_query_vec_1 = motif_idvec_query
		motif_query_num1 = len(motif_query_vec_1)
		list1 = []
		dict_signal_1 = dict()
		dict_query_1 = dict()

		ratio_1, ratio_2 = select_config['ratio_1'], select_config['ratio_2']
		n_neighbors = select_config['neighbor_num']
		run_id2 = 2
		folder_id_1 = 2
		# type_query = 0
		# type_query = 1
		# filename_save_annot_1 = '%s.0.1.1'%(data_file_type_query)
		filename_save_annot_1 = '%s.0.1'%(data_file_type_query)
		input_file_path_1 = input_file_path_query_2
		start = time.time()
		for i1 in range(motif_query_num1):
			motif_id = motif_query_vec_1[i1]
			motif_id_query = motif_id
			motif_id2_query = df_annot_1.loc[df_annot_1['motif_id']==motif_id,'motif_id2']
			motif_id2_num = len(motif_id2_query)
			print('motif_id, motif_id2: ',motif_id,motif_id2_query,motif_id2_num,i1)
			list1.extend(motif_id2_query)

			for i2 in range(motif_id2_num):
				motif_id2 = motif_id2_query[i2]
				motif_id1 = df_annot_1.loc[motif_id2,'motif_id1']
				folder_id = df_annot_1.loc[motif_id2,'folder_id']
				config_id_2 = dict_config_annot1[folder_id]

				if type_query==0:
					# file_path_query_2 = dict_file_annot2[folder_id]
					file_path_query_2 = dict_file_annot2[folder_id_1]
					# input_file_path_query = '%s/train1'%(file_path_query_2)
					input_file_path_query = '%s/train1_vbak1'%(file_path_query_2)

					# test_query_train.phenograph.20.0.25_1.5.12.neighbor100.VDR.35.pbmc.0.1.1.txt
					filename_prefix_1 = 'test_query_train.%s.%s_%s.%d.neighbor%d'%(method_type_group,ratio_1,ratio_2,config_id_2,n_neighbors)
					input_filename = '%s/%s.%s.%s.%d.txt'%(input_file_path_query,filename_prefix_1,motif_id1,filename_save_annot_1,run_id2)
					if os.path.exists(input_filename)==False:
						print('the file does not exist: %s'%(input_filename))
						run_id2_1 = 1
						input_filename = '%s/%s.%s.%s.%d.txt'%(input_file_path_query,filename_prefix_1,motif_id1,filename_save_annot_1,run_id2_1)

						if os.path.exists(input_filename)==False:
							print('the file does not exist: %s'%(input_filename))
							file_path_query_pre2 = dict_file_annot2[folder_id]
							input_file_path_2 = file_path_query_pre2
							input_filename = '%s/%s.%s.%s.1.txt'%(input_file_path_2,filename_prefix_1,motif_id1,filename_save_annot_1)

				elif type_query==1:
					file_path_query_2 = dict_file_annot2[folder_id_1]
					# run_id_2_ori = 2
					# run_id_2_ori = 3
					run_id_2_ori = select_config['run_id_2_ori']
					flag_select_1, flag_select_2 = 2, 2
					# flag_sample = 0
					flag_sample = select_config['flag_sample']
					filename_annot_pre1 = '%s.0.1'%(data_file_type_query)
					filename_annot_2 = '%s_%d_%d_%d'%(run_id_2_ori,flag_select_1,flag_select_2,flag_sample)
					input_file_path_query = '%s/train%s'%(file_path_query_2,filename_annot_2)
					# input_filename = 'test_query_train.pred.STAT1.46.pbmc.0.1.2_2_2_0.1.txt'
					filename_prefix_1 = 'test_query_train.pred'
					filename_save_annot_1 = '%s.%s.1'%(filename_annot_pre1,filename_annot_2)
					input_filename = '%s/%s.%s.%s.txt'%(input_file_path_query,filename_prefix_1,motif_id_query,filename_save_annot_1)

				if os.path.exists(input_filename)==False:
					print('the file does not exist: ',input_filename)
					continue

				df_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				df_query_1 = df_query_1.loc[(~df_query_1.index.duplicated(keep='first')),:]
				t_columns_1 = df_query_1.columns.difference([column_signal],sort=False)
				print(input_filename)
				
				df_signal = df_query_1.loc[:,[column_signal]]
				# df_query1 = df_query_1.loc[:,t_columns_1]
				df_query1 = df_query_1.loc[:,column_vec_query1]

				print('df_signal,df_query1,motif_id: ',df_signal.shape,df_query1.shape,motif_id,motif_id1,motif_id2)
				print(df_signal[0:2])
				print(df_query1[0:2])
				
				dict_signal_1.update({motif_id2:df_signal})
				dict_query_1.update({motif_id2:df_query1})

		motif_query_vec = list1
		stop = time.time()
		print('load data used %.2fs'%(stop-start))

		# score query for performance comparison
		start = time.time()
		flag_compare_1=0
		parallel = 0
		# parallel = 1
		df_score_query_1 = self.test_query_compare_binding_basic_1(data1=dict_signal_1,data2=dict_query_1,motif_query_vec=motif_query_vec,motif_query_vec_2=[],
																	df_annot_motif=df_annot_motif,dict_method_type=dict_method_type,
																	feature_type_vec=[],column_vec_query=column_vec_query,
																	column_signal=column_signal,column_motif=column_motif,
																	method_type_vec=[],method_type_group='',
																	flag_score_1=0,flag_score_2=0,flag_compare_1=flag_compare_1,
																	type_id_1=0,parallel=parallel,input_file_path='',
																	save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=verbose,select_config=select_config)

		stop = time.time()
		print('performance comparison used %.2fs'%(stop-start))

		output_file_path = input_file_path_query_2
		# output_filename = '%s/test_query_df_score.beta.1.copy1.txt'%(output_file_path)
		# output_filename = '%s/test_query_df_score.beta.1.copy2.txt'%(output_file_path)
		output_filename = '%s/test_query_df_score.beta.%s.1.txt'%(output_file_path,filename_save_annot_1)
		df_score_query_1.to_csv(output_filename,sep='\t')

		return df_score_query_1

	## load dataframe
	def test_query_feature_link_load_1(self,data=[],motif_id1='',peak_loc_ori=[],folder_id='',method_type_vec=[],input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',filename_save_annot_2='',output_filename='',verbose=0,select_config={}):

				flag_load = 1
				if flag_load>0:
					# filename_prefix_1 = 'test_motif_query_binding_compare'
					filename_prefix_1 = filename_prefix_save
					# input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					# input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					input_filename_query1 = '%s/%s.%s.%s.txt'%(input_file_path,filename_prefix_1,motif_id1,filename_save_annot)

					flag_group_query_1 = 0
					if (os.path.exists(input_filename_query1)==True):
						df_pre1 = pd.read_csv(input_filename_query1,index_col=0,sep='\t')
						print('df_pre1: ',df_pre1.shape)
						# df_pre1 = df_pre1.drop_duplicates(subset=['peak_id'])
						# df_pre1_1 = df_pre1_1.drop_duplicates(subset=['peak_id'])
						df_pre1 = df_pre1.loc[~(df_pre1.index.duplicated(keep='first')),:]
						print('df_pre1: ',df_pre1.shape)
						print(df_pre1[0:5])
						print(input_filename_query1)
					else:
						print('the file does not exist: %s'%(input_filename_query1))
						print('please provide feature group estimation')
						flag_group_query_1 = 1
						# continue
						# return

					if flag_group_query_1==0:
						# peak_loc_1 = df_pre1.index
						# df_pre1 = df_pre1.loc[peak_loc_ori,:]
						df_query1 = df_pre1
					else:
						load_mode_pre1_1 = 1
						if load_mode_pre1_1>0:
							# load the TF binding prediction file
							# the possible columns: (signal,motif,predicted binding,motif group)
							# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
							if folder_id in [1,2]:
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.group2.txt'%(input_file_path,filename_prefix_1,motif_id1,filename_save_annot_2)
							elif folder_id in [0]:
								# thresh_fdr_peak_tf_GRaNIE = select_config['thresh_fdr_peak_tf_GRaNIE']
								thresh_fdr_peak_tf_GRaNIE = 0.2
								upstream_tripod_2 = 100
								type_id_tripod = select_config['type_id_tripod']
								filename_save_annot_2_pre2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod_2,type_id_tripod)
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.txt'%(input_file_path,filename_prefix_1,motif_id1,filename_save_annot_2_pre2)

							if os.path.exists(input_filename==True):
								df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
								peak_loc_1 = df_1.index
								peak_num_1 = len(peak_loc_1)
								print('peak_loc_1: ',peak_num_1)

								method_type_feature_link = select_config['method_type_feature_link']
								if len(method_type_vec)==0:
									method_type_vec_pre2 = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
								else:
									method_type_vec_pre2 = method_type_vec

								annot_str_vec = ['motif','pred','score']
								column_vec_query = ['signal']
								for method_type_query in method_type_vec_pre2:
									column_vec_query.extend(['%s.%s'%(method_type_query,annot_str1) for annot_str1 in annot_str_vec])

								# df_query1 = df_1
								df_query1 = df_1.loc[:,column_vec_query]
								print('df_query1: ',df_query1.shape)
								print(df_query1.columns)
								print(df_query1[0:2])
								print(input_filename)
							else:
								print('the file does not exist: %s'%(input_filename))
								df_query1 = pd.DataFrame(index=peak_loc_ori)
								load_mode_pre1_1 = 0

					if (flag_group_query_1==0) or (load_mode_pre1_1>0):
						df_query1_ori = df_query1.copy()
						peak_loc_1 = df_query1.index
						column_vec = df_query1.columns
						df_query1 = pd.DataFrame(index=peak_loc_ori)
						df_query1.loc[peak_loc_1,column_vec] = df_query1_ori
						print('df_query1: ',df_query1.shape)

					return df_query1

	## compute peak-TF correlation
	# compute peak-TF correlation
	def test_query_compare_peak_tf_corr_1(self,data=[],motif_id_query='',motif_id1='',motif_id2='',column_signal='signal',column_value='',thresh_value=-0.05,motif_data=[],motif_data_score=[],peak_read=[],rna_exprs=[],flag_query=0,input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',output_filename='',verbose=0,select_config={}):

				flag_query1 = 1
				if flag_query1>0:
					df_pre1 = data
					# field_id1 = 'peak_tf_corr'
					# field_id2 = 'peak_tf_pval_corrected'
					# query peak accessibility-TF expression correlation
					# flag_peak_tf_corr = 0
					flag_peak_tf_corr = 1
					column_peak_tf_corr = column_value
					if column_value=='':
						column_peak_tf_corr = 'peak_tf_corr'
					column_query = column_peak_tf_corr
					if column_query in df_pre1.columns:
						flag_peak_tf_corr = 0

					if flag_peak_tf_corr>0:
						save_mode = 1
						flag_load_1 = 1
						field_load = []
						
						# data_file_type_query = select_config['data_file_type']
						# input_filename_list1 = ['%s/%s.%s.1.copy1.txt'%(file_save_path,filename_prefix,filename_annot1) for filename_annot1 in filename_annot_vec[0:3]]
						input_filename_list1 = []
						motif_query_vec = [motif_id_query]
						# motif_data_query = pd.DataFrame(index=peak_query_group2,columns=motif_query_vec,data=1)
						peak_loc_ori = peak_read.columns
						peak_loc_2 = peak_loc_ori
						motif_data_query = pd.DataFrame(index=peak_loc_2,columns=motif_query_vec,data=1)
						
						# correlation_type=select_config['correlation_type']
						correlation_type = 'spearmanr'
						alpha = 0.05
						method_type_id_correction = 'fdr_bh'
						# filename_prefix = 'test_peak_tf_correlation.%s.%s.2'%(motif_id1,data_file_type_query)
						filename_prefix = filename_prefix_save
						save_mode_2 = 1
						# field_query = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected','motif_basic']
						field_load = [correlation_type,'pval','pval_corrected']
						dict_peak_tf_corr_ = utility_1.test_peak_tf_correlation_query_1(motif_data=motif_data_query,
																							peak_query_vec=[],
																							motif_query_vec=motif_query_vec,
																							peak_read=peak_read,
																							rna_exprs=rna_exprs,
																							correlation_type=correlation_type,
																							pval_correction=1,
																							alpha=alpha,method_type_id_correction=method_type_id_correction,
																							flag_load=flag_load_1,field_load=field_load,
																							save_mode=save_mode_2,
																							input_file_path=input_file_path,
																							input_filename_list=input_filename_list1,
																							output_file_path=output_file_path,
																							filename_prefix=filename_prefix,
																							select_config=select_config)

						# self.dict_peak_tf_corr_ = dict_peak_tf_corr_
						# select_config = self.select_config

						field_id1 = 'peak_tf_corr'
						field_id2 = 'peak_tf_pval_corrected'
						field_id1_query = field_load[0]
						field_id2_query = field_load[2]
						peak_tf_corr_1 = dict_peak_tf_corr_[field_id1_query]
						peak_tf_pval_corrected1 = dict_peak_tf_corr_[field_id2_query]
						print('peak_tf_corr_1: ',peak_tf_corr_1.shape)
						print(peak_tf_corr_1[0:2])

						print('peak_tf_pval_corrected: ',peak_tf_pval_corrected1.shape)
						print(peak_tf_pval_corrected1[0:2])

						peak_vec_2 = peak_loc_2
						column_corr_1 = field_id1
						column_pval = field_id2
						df_pre1.loc[peak_vec_2,column_corr_1] = peak_tf_corr_1.loc[peak_vec_2,motif_id_query]
						df_pre1.loc[peak_vec_2,column_pval] = peak_tf_pval_corrected1.loc[peak_vec_2,motif_id_query]

						if (save_mode>0) and (output_filename!=''):
							if os.path.exists(output_filename)==True:
								print('the file exists: %s'%(output_filename))
							df_pre1.to_csv(output_filename,sep='\t')

					# flag_peak_tf_corr_2 = 1
					flag_peak_tf_corr_2 = flag_query
					df_annot_peak_tf = []
					if (flag_peak_tf_corr_2>0):
						if column_signal in df_pre1.columns:
							peak_loc_pre1 = df_pre1.index
							id_signal = (df_pre1[column_signal]>0)
							peak_signal = peak_loc_pre1[id_signal]

							thresh_1 = thresh_value
							id_1 = (df_pre1[column_peak_tf_corr]>thresh_1)&(id_signal)
							id_2 = (~id_1)&(id_signal)
							peak_query_1 = peak_loc_pre1[id_1]
							peak_query_2 = peak_loc_pre1[id_2]

							peak_tf_corr_1 = df_pre1.loc[peak_signal,column_query]
							peak_tf_corr_2 = df_pre1.loc[peak_query_1,column_query]
							peak_tf_corr_3 = df_pre1.loc[peak_query_2,column_query]
							list1 = [peak_tf_corr_1,peak_tf_corr_2,peak_tf_corr_3]
							peak_num_1, peak_num_2, peak_num_3 = len(peak_signal),len(peak_query_1),len(peak_query_2)
							list2 = [peak_num_1,peak_num_2,peak_num_3]
							list_query1 = []
							query_num = len(list1)
							for i2 in range(query_num):
								peak_tf_corr_query = list1[i2]
								# peak_query = list2[i2]
								# peak_num_query = len(peak_query)
								peak_num_query = list2[i2]
								t_vec_1 = [peak_num_query,np.max(peak_tf_corr_query),np.min(peak_tf_corr_query),np.median(peak_tf_corr_query),np.mean(peak_tf_corr_query),np.std(peak_tf_corr_query)]
								list_query1.append(t_vec_1)

							mtx_1 = np.asarray(list_query1)
							group_vec_peak_tf = ['peak_signal','peak_signal_1','peak_signal_2']
							column_vec = ['peak_num','corr_max','corr_min','corr_median','corr_mean','corr_std']
							df_annot_peak_tf = pd.DataFrame(index=group_vec_peak_tf,columns=column_vec,data=mtx_1)
							df_annot_peak_tf['group'] = [0,1,2]
							field_query = ['motif_id','motif_id1','motif_id2']
							list_value = [motif_id_query,motif_id1,motif_id2]
							field_num = len(field_query)
							for i2 in range(field_num):
								field_id, query_value = field_query[i2], list_value[i2]
								df_annot_peak_tf[field_id] = query_value
							# list_annot_peak_tf.append(df_annot_peak_tf)

					return df_pre1, df_annot_peak_tf
	
	## recompute based on clustering of peak and TF
	# recompute based on training
	# compute peak-TF correlation
	def test_query_compare_binding_pre1_5_1_recompute_5_1(self,data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		run_id1 = select_config['run_id']
		thresh_num1 = 5
		# method_type_vec = ['insilico_1','TRIPOD','GRaNIE','Pando']+['joint_score.thresh%d'%(i1+1) for i1 in range(thresh_num1)]+['joint_score_2.thresh3']
		method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre1.thresh22']

		if 'method_type_feature_link' in select_config:
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_feature_link_1 = method_type_feature_link.split('.')[0]
		
		filename_save_annot = '1'
		method_type_vec = pd.Index(method_type_vec).union([method_type_feature_link],sort=False)

		flag_config_1=1
		if flag_config_1>0:
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
			# ## query the file path
			# select_config = self.test_file_path_query_2(method_type_vec=method_type_vec,select_config=select_config)

			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)
			# file_path_peak_tf = select_config['file_path_peak_tf']
			# file_save_path_1 = select_config['file_path_peak_tf']

		flag_motif_data_load_1 = 1
		if flag_motif_data_load_1>0:
			print('load motif data')
			# method_type_vec_query = ['insilico_1','joint_score_2.thresh3']
			# method_type_vec_query = ['insilico_0.1','joint_score_pre2.thresh3']
			method_type_vec_query = ['insilico_0.1']+[method_type_feature_link]
			method_type_1, method_type_2 = method_type_vec_query[0:2]
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																			select_config=select_config)

			motif_data_query1 = dict_motif_data[method_type_2]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_2]['motif_data_score']
			peak_loc_ori = motif_data_query1.index
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			self.dict_motif_data = dict_motif_data

		flag_load_1 = 1
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			# filename_1 = '%s/test_rna_meta_ad.%s.log_normalize.2.h5ad'%(input_file_path_query,data_file_type_query)
			# filename_2 = '%s/test_atac_meta_ad.%s.log_normalize.2.h5ad'%(input_file_path_query,data_file_type_query)
			# filename_3_ori = '%s/test_query.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
			# # filename_3 = '%s/test_rna_meta_ad.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)

			# select_config.update({'filename_rna':filename_1,'filename_atac':filename_2,
			# 						'filename_rna_exprs_1':filename_3_ori})
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori(meta_exprs=[],peak_read=[],flag_format=False,select_config=select_config)
			# rna_exprs = meta_scaled_exprs
			# rna_exprs = meta_exprs_2
			# sample_id = rna_exprs.index
			sample_id = meta_scaled_exprs.index
			peak_read = peak_read.loc[sample_id,:]
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			rna_exprs = meta_scaled_exprs
			print('peak_read, rna_exprs: ',peak_read.shape,rna_exprs.shape)

			print(peak_read[0:2])
			print(rna_exprs[0:2])
			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs
			peak_loc_ori = peak_read.columns

		file_save_path_1 = select_config['file_path_peak_tf']
		if data_file_type_query in ['pbmc']:
			dict_file_annot1 = select_config['dict_file_annot1']
			dict_file_annot2 = select_config['dict_file_annot2']
			dict_config_annot1 = select_config['dict_config_annot1']

		folder_id = select_config['folder_id']
		# group_id_1 = folder_id+1
		file_path_query_1 = dict_file_annot1[folder_id] # the first level directory
		file_path_query_2 = dict_file_annot2[folder_id] # the second level directory including the configurations

		input_file_path = file_path_query_2
		output_file_path = file_path_query_2

		folder_id_query = 2 # the folder to save annotation files
		file_path_query1 = dict_file_annot1[folder_id_query]
		file_path_query2 = dict_file_annot2[folder_id_query]
		input_file_path_query = file_path_query1
		input_file_path_query_pre2 = '%s/vbak1'%(file_path_query1)

		dict_query_1 = dict()
		feature_type_vec = ['latent_peak_motif','latent_peak_motif_ori','latent_peak_tf','latent_peak_tf_link']
		feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
		type_id_group_2 = select_config['type_id_group_2']
		feature_type_query_1 = feature_type_vec_group[type_id_group_2]
		feature_type_query_2 = 'latent_peak_tf'

		feature_type_vec_query = [feature_type_query_1,feature_type_query_2]
		feature_type_vec_2_ori = []
		prefix_str1 = 'latent_'
		prefix_str1_len = len(prefix_str1)
		for feature_type_query in feature_type_vec_query:
			b = feature_type_query.find(prefix_str1)
			feature_type = feature_type_query[(b+prefix_str1_len):]
			feature_type_vec_2_ori.append(feature_type)

		flag_annot_1 = 1
		# method_type_group = 'MiniBatchKMeans.50'
		method_type_vec_group_ori = ['MiniBatchKMeans.%d'%(n_clusters_query) for n_clusters_query in [30,50,100]]+['phenograph.%d'%(n_neighbors_query) for n_neighbors_query in [10,15,20,30]]
		# method_type_group = 'MiniBatchKMeans.%d'%(n_clusters)
		# method_type_group_id = 1
		# n_neighbors_query = 30
		n_neighbors_query = 20
		method_type_group = 'phenograph.%d'%(n_neighbors_query)
		# method_type_group_id = 6
		if 'method_type_group' in select_config:
			method_type_group = select_config['method_type_group']
		print('method_type_group: ',method_type_group)

		if flag_annot_1>0:
			thresh_size_1 = 100
			if 'thresh_size_group' in select_config:
				thresh_size_group = select_config['thresh_size_group']
				thresh_size_1 = thresh_size_group

			# for selecting the peak loci predicted with TF binding
			# thresh_score_query_1 = 0.125
			# thresh_size_1 = 20
			thresh_score_query_1 = 0.15
			if 'thresh_score_group_1' in select_config:
				thresh_score_group_1 = select_config['thresh_score_group_1']
				thresh_score_query_1 = thresh_score_group_1
			
			thresh_score_default_1 = thresh_score_query_1
			thresh_score_default_2 = 0.10

			peak_distance_thresh_1 = 500
			thresh_fdr_peak_tf_GRaNIE = 0.2
			upstream_tripod = peak_distance_thresh_1
			# type_id_tripod = 0
			# type_id_tripod = 1
			type_id_tripod = select_config['type_id_tripod']
			filename_save_annot_2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod,type_id_tripod)
			# thresh_size_1 = 20

			# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
			# filename_save_annot_2 = filename_save_annot
			filename_save_annot2_ori = '%s.%s.%d.%s'%(filename_save_annot_2,thresh_score_query_1,thresh_size_1,method_type_group)

		flag_group_load_1 = 1
		if flag_group_load_1>0:
			# load feature group estimation for peak or peak and TF
			n_clusters = 50
			peak_loc_ori = peak_read.columns
			feature_query_vec = peak_loc_ori
			# the overlap matrix; the group member number and frequency of each group; the group assignment for the two feature types
			df_overlap_compare, dict_group_basic_1, df_group_1, df_group_2 = self.test_query_feature_group_load_1(data=[],feature_type_vec=feature_type_vec_query,feature_query_vec=feature_query_vec,method_type_group=method_type_group,input_file_path=input_file_path,
																													save_mode=1,output_file_path=output_file_path,filename_prefix_save='',filename_save_annot=filename_save_annot2_ori,output_filename='',verbose=0,select_config=select_config)

			self.df_group_pre1 = df_group_1
			self.df_group_pre2 = df_group_2
			self.dict_group_basic_1 = dict_group_basic_1
		
		method_type_feature_link = select_config['method_type_feature_link']
		column_signal = 'signal'
		column_motif = '%s.motif'%(method_type_feature_link)
		column_pred1 = '%s.pred'%(method_type_feature_link)
		column_score_1 = 'score_pred1'
		df_score_annot = []
		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]

		n_neighbors = select_config['neighbor_num']
		peak_loc_ori = peak_read.columns
		# df_pre1_ori = pd.DataFrame(index=peak_loc_ori)
		method_type_group = select_config['method_type_group']

		dict_motif_data = self.dict_motif_data
		method_type_query = method_type_feature_link
		motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
		motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']
		# peak_loc_ori = motif_data_query1.index
		motif_data_query1 = motif_data_query1.loc[peak_loc_ori,:]
		motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_ori,:]
			
		print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
		print(motif_data_query1[0:2])
		print(motif_data_score_query1[0:2])
		motif_data = motif_data_query1
		motif_data_score = motif_data_score_query1
		# self.dict_motif_data = dict_motif_data

		input_filename = '%s/test_peak_file.ChIP-seq.pbmc.1_2.2.copy2.txt'%(input_file_path_query_pre2)
		df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
		df_annot_1 = df_annot_ori.drop_duplicates(subset=['motif_id2'])
		print('df_annot_ori, df_annot_1: ',df_annot_ori.shape,df_annot_1.shape)

		motif_idvec_query = df_annot_1.index.unique()
		motif_idvec_1 = df_annot_1['motif_id1']
		motif_idvec_2 = df_annot_1['motif_id2']
		df_annot_1.index = np.asarray(df_annot_1['motif_id2'])
		dict_motif_query_1 = dict(zip(motif_idvec_2,list(motif_idvec_query)))

		motif_query_num = len(motif_idvec_query)
		motif_num2 = len(motif_idvec_2)
		query_num_ori = len(motif_idvec_2)
		print('motif_idvec_query, motif_idvec_2: ',motif_query_num,motif_num2)

		flag_query2=0
		if flag_query2>0:
			# method_type_vec_2 = ['TRIPOD','GRaNIE','Pando']

			list_annot_peak_tf = []
			iter_vec_1 = np.arange(query_num_ori)
			for i1 in iter_vec_1:
				motif_id2_query = motif_idvec_2[i1]
				motif_id_query = df_annot_1.loc[motif_id2_query,'motif_id']
				motif_id1_query = df_annot_1.loc[motif_id2_query,'motif_id1']
				
				folder_id_query = df_annot_1.loc[motif_id2_query,'folder_id']
				motif_id1, motif_id2 = motif_id1_query, motif_id2_query

				input_file_path_query_1 = dict_file_annot1[folder_id_query] # the first level directory
				input_file_path_query_2 = dict_file_annot2[folder_id_query] # the second level directory including the configurations

				# motif_id1 = motif_id
				print('motif_id_query, motif_id: ',motif_id_query,motif_id1,i1)

				overwrite_2 = False
				filename_prefix_save = 'test_query.%s'%(method_type_group)
				# filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
				iter_id1 = 0
				config_id_load = select_config['config_id_load']
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)

				ratio_1,ratio_2 = select_config['ratio_1'],select_config['ratio_2']
				config_id_2 = select_config['config_id_2']
				filename_annot2 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
				filename_annot_train_pre1 = filename_annot2
				filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
				filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)

				flag1=1
				if flag1>0:
				# try:
					# load the TF binding prediction file
					# the possible columns: (signal,motif,predicted binding,motif group)
					# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
					filename_prefix_1 = 'test_motif_query_binding_compare'
					# input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					# input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					
					method_type_group = select_config['method_type_group']
					filename_save_annot = '%s.%s_%s.neighbor%d.%d.copy2_2.2'%(method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
						
					method_type_feature_link = select_config['method_type_feature_link']
					method_type_vec = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
					df_query1 = self.test_query_feature_link_load_1(data=[],motif_id1=motif_id1,peak_loc_ori=peak_loc_ori,folder_id=folder_id_query,method_type_vec=method_type_vec,input_file_path=input_file_path_query_1,
																		save_mode=1,output_file_path='',filename_prefix_save=filename_prefix_1,
																		filename_save_annot=filename_save_annot,filename_save_annot_2=filename_save_annot_2,output_filename='',verbose=verbose,select_config=select_config)

					column_signal = 'signal'
					column_peak_tf_corr = 'peak_tf_corr'
					flag_peak_tf_corr=0
					if flag_peak_tf_corr>0:
						column_value = column_peak_tf_corr
						thresh_value=-0.05
						output_file_path = input_file_path_query_1
						df_query1, df_annot_peak_tf = self.test_query_compare_peak_tf_corr_1(data=df_query1,motif_id_query=motif_id_query,motif_id1=motif_id1,motif_id2=motif_id2,column_signal=column_signal,column_value=column_value,thresh_value=thresh_value,
																								motif_data=motif_data,motif_data_score=motif_data_score,peak_read=peak_read,rna_exprs=rna_exprs,flag_query=1,input_file_path='',
																								save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',output_filename='',verbose=verbose,select_config=select_config)

						list_annot_peak_tf.append(df_annot_peak_tf)

			if len(list_annot_peak_tf)>0:
				output_file_path = input_file_path_query_pre2
				df_annot_peak_tf_1 = pd.concat(list_annot_peak_tf,axis=0,join='outer',ignore_index=False)
				# output_filename = '%s/test_query_df_annot.peak_tf.%s.1.txt'%(output_file_path,filename_save_annot2_1)
				df_annot_peak_tf_1 = df_annot_peak_tf_1.round(7)
				output_filename = '%s/test_query_df_annot.peak_tf.1.txt'%(output_file_path)
				df_annot_peak_tf_1.to_csv(output_filename,sep='\t')
				print('df_annot_peak_tf_1: ',df_annot_peak_tf_1.shape)
				print(output_filename)

				df_annot_peak_tf_sort1  = df_annot_peak_tf_1.sort_values(by=['group','corr_mean','motif_id'],ascending=[True,False,True])
				output_filename_2 = '%s/test_query_df_annot.peak_tf.sort1.txt'%(output_file_path)
				df_annot_peak_tf_sort1.to_csv(output_filename_2,sep='\t')
				print('df_annot_peak_tf_sort1: ',df_annot_peak_tf_sort1.shape)
				print(output_filename_2)

		flag_query2_1=1
		if flag_query2_1>0:
			input_file_path_1 = input_file_path_query_pre2
			input_filename_1 = '%s/test_query_df_annot.peak_tf.sort1.txt'%(input_file_path_1)
			df_peak_tf_query1 = pd.read_csv(input_filename_1,index_col=0,sep='\t')
			df_peak_tf_query1.index = np.asarray(df_peak_tf_query1['motif_id2'])
			
			input_file_path_2 = '%s/folder1_1'%(input_file_path_query_pre2)
			input_filename_2 = '%s/test_motif_query_binding_compare.1.0.2.500.0.combine.2_1.2.txt'%(input_file_path_2)
			df_score_query = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			peak_distance_thresh_1 = 100
			column_1 = 'peak_distance_thresh'
			df_score_query[column_1] = df_score_query[column_1].fillna(peak_distance_thresh_1)

			field_query = ['config_group_annot','feature_scale','method_type_group']
			method_type_group = 'phenograph.20'
			default_value_vec = [0,0,method_type_group]
			for (field_id, query_value) in zip(field_query,default_value_vec):
				df_score_query[field_id] = df_score_query[field_id].fillna(query_value)

			field_id1, field_id2, field_id3 = field_query[0:3]
			id1 = (df_score_query[column_1]==peak_distance_thresh_1)
			id2 = (df_score_query[field_id1]==0)&(df_score_query[field_id2]==0)&(df_score_query[field_id3]==method_type_group)

			group_motif = 2
			id_group_motif = (df_score_query['group_motif']==2)
			id_1 = (id1&id2&id_group_motif)

			df_score_query_ori = df_score_query.copy()
			df_score_query = df_score_query_ori.loc[id_1,:]
			print('df_score_query_ori, df_score_query: ',df_score_query_ori.shape,df_score_query.shape)

			method_type_feature_link_1 = 'joint_score_pre1.thresh22'
			method_type_feature_link_2 = 'latent_peak_motif_peak_gene_combine_LogisticRegression_pred'

			method_type_vec_pre1 = ['CIS-BP_motif','HOCOMOCO_PWMScan','Pando_motif','JASPAR_motif']
			method_type_vec_pre2 = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+[method_type_feature_link_1,method_type_feature_link_2]
			method_type_vec_query = method_type_vec_pre1 + method_type_vec_pre2

			id_2 = df_score_query['method_type'].isin(method_type_vec_query)
			df_score_query = df_score_query.loc[id_2,:]
			print('df_score_query: ',df_score_query.shape)

			column_vec_1 = ['motif_id','motif_id2','group_motif','method_type','method_type_group',
								'config_group_annot','feature_scale']
			df_score_query = df_score_query.drop_duplicates(subset=column_vec_1)
			print('df_score_query (unduplicated): ',df_score_query.shape)

			output_file_path = input_file_path_query_pre2
			output_filename = '%s/test_query_df_score.1.txt'%(output_file_path)
			df_score_query = df_score_query.sort_values(by=['motif_id','motif_id2','method_type'],ascending=True)
			df_score_query.to_csv(output_filename,sep='\t')

			column_vec_query1 = ['F1','motif_id','motif_id2','group_motif','method_type']
			# df_score_query1 = df_score_query.loc[id_2,column_vec_query1]
			df_score_query1 = df_score_query.loc[:,column_vec_query1]
			print('df_score_query1: ',df_score_query1.shape)

			column_idvec_1 = ['motif_id','motif_id2','group_motif']
			column_idvec_2 = 'method_type'
			column_query = 'F1'
			column_id_1 = 'motif_id2'
			df_score_query2 = df_score_query1.pivot(index=column_id_1,columns=column_idvec_2,values=column_query)
			print('df_score_query2: ',df_score_query2.shape)
			print(df_score_query2.columns)
			print(df_score_query2[0:5])

			df_score_query2['motif_id2'] = np.asarray(df_score_query2.index)
			df_score_query2_ori = df_score_query2.copy()
			id_query = df_score_query2_ori['motif_id2'].isin(motif_idvec_2)
			df_score_query2 = df_score_query2_ori.loc[id_query,:]
			print('df_score_query2: ',df_score_query2.shape)

			query_idvec = df_peak_tf_query1.index
			df_score_query2.index = np.asarray(df_score_query2['motif_id2'])
			df_peak_tf_query1.loc[:,method_type_vec_query] = np.asarray(df_score_query2.loc[query_idvec,method_type_vec_query])

			output_file_path = input_file_path_query_pre2
			output_filename = '%s/test_query_df_annot.peak_tf.sort1.2.txt'%(output_file_path)
			df_peak_tf_query1.to_csv(output_filename,sep='\t')

			return df_peak_tf_query1


	## recompute based on clustering of peak and TF
	# recompute based on training
	def test_query_compare_binding_pre1_5_1_recompute_5_2(self,data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		run_id1 = select_config['run_id']
		thresh_num1 = 5
		# method_type_vec = ['insilico_1','TRIPOD','GRaNIE','Pando']+['joint_score.thresh%d'%(i1+1) for i1 in range(thresh_num1)]+['joint_score_2.thresh3']
		method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh22']

		if 'method_type_feature_link' in select_config:
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_feature_link_1 = method_type_feature_link.split('.')[0]
		
		filename_save_annot = '1'
		method_type_vec = pd.Index(method_type_vec).union([method_type_feature_link],sort=False)

		flag_config_1=1
		if flag_config_1>0:
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']
			# if data_file_type_query in ['CD34_bonemarrow']:
			# 	input_file_path = '%s/peak1'%(root_path_2)
			# elif data_file_type_query in ['pbmc']:
			# 	input_file_path = '%s/peak2'%(root_path_2)
			# file_save_path_1 = input_file_path
		
			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)
			# file_path_peak_tf = select_config['file_path_peak_tf']
			# file_save_path_1 = select_config['file_path_peak_tf']

		flag_motif_data_load_1 = 1
		if flag_motif_data_load_1>0:
			print('load motif data')
			# method_type_vec_query = ['insilico_1','joint_score_2.thresh3']
			# method_type_vec_query = ['insilico_0.1','joint_score_pre2.thresh3']
			method_type_vec_query = ['insilico_0.1']+[method_type_feature_link]
			method_type_1, method_type_2 = method_type_vec_query[0:2]
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																			select_config=select_config)

			motif_data_query1 = dict_motif_data[method_type_2]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_2]['motif_data_score']
			peak_loc_ori = motif_data_query1.index
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			self.dict_motif_data = dict_motif_data

		flag_load_1 = 1
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori(meta_exprs=[],peak_read=[],flag_format=False,select_config=select_config)
			# rna_exprs = meta_scaled_exprs
			# rna_exprs = meta_exprs_2
			# sample_id = rna_exprs.index
			sample_id = meta_scaled_exprs.index
			peak_read = peak_read.loc[sample_id,:]
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			rna_exprs = meta_scaled_exprs
			print('peak_read, rna_exprs: ',peak_read.shape,rna_exprs.shape)

			print(peak_read[0:2])
			print(rna_exprs[0:2])
			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs
			peak_loc_ori = peak_read.columns

		file_save_path_1 = select_config['file_path_peak_tf']
		if data_file_type_query in ['pbmc']:
			dict_file_annot1 = select_config['dict_file_annot1']
			dict_file_annot2 = select_config['dict_file_annot2']
			dict_config_annot1 = select_config['dict_config_annot1']

		folder_id = select_config['folder_id']
		# group_id_1 = folder_id+1
		file_path_query_1 = dict_file_annot1[folder_id] # the first level directory
		file_path_query_2 = dict_file_annot2[folder_id] # the second level directory including the configurations

		input_file_path = file_path_query_2
		output_file_path = file_path_query_2

		folder_id_query = 2 # the folder to save annotation files
		folder_id_query_1 = folder_id_query
		file_path_query1 = dict_file_annot1[folder_id_query]
		file_path_query2 = dict_file_annot2[folder_id_query]
		input_file_path_query = file_path_query1
		input_file_path_query_2 = '%s/vbak1'%(file_path_query1)
		output_file_path_query = file_path_query1

		dict_query_1 = dict()
		feature_type_vec = ['latent_peak_motif','latent_peak_motif_ori','latent_peak_tf','latent_peak_tf_link']
		feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
		type_id_group_2 = select_config['type_id_group_2']
		feature_type_query_1 = feature_type_vec_group[type_id_group_2]
		feature_type_query_2 = 'latent_peak_tf'

		feature_type_vec_query = [feature_type_query_1,feature_type_query_2]
		feature_type_vec_2_ori = []
		prefix_str1 = 'latent_'
		prefix_str1_len = len(prefix_str1)
		for feature_type_query in feature_type_vec_query:
			b = feature_type_query.find(prefix_str1)
			feature_type = feature_type_query[(b+prefix_str1_len):]
			feature_type_vec_2_ori.append(feature_type)

		flag_annot_1 = 1
		# method_type_group = 'MiniBatchKMeans.50'
		method_type_vec_group_ori = ['MiniBatchKMeans.%d'%(n_clusters_query) for n_clusters_query in [30,50,100]]+['phenograph.%d'%(n_neighbors_query) for n_neighbors_query in [10,15,20,30]]
		# method_type_group = 'MiniBatchKMeans.%d'%(n_clusters)
		# method_type_group_id = 1
		# n_neighbors_query = 30
		n_neighbors_query = 20
		method_type_group = 'phenograph.%d'%(n_neighbors_query)
		# method_type_group_id = 6
		if 'method_type_group' in select_config:
			method_type_group = select_config['method_type_group']
		print('method_type_group: ',method_type_group)

		if flag_annot_1>0:
			thresh_size_1 = 100
			if 'thresh_size_group' in select_config:
				thresh_size_group = select_config['thresh_size_group']
				thresh_size_1 = thresh_size_group

			# for selecting the peak loci predicted with TF binding
			# thresh_score_query_1 = 0.125
			# thresh_size_1 = 20
			thresh_score_query_1 = 0.15
			if 'thresh_score_group_1' in select_config:
				thresh_score_group_1 = select_config['thresh_score_group_1']
				thresh_score_query_1 = thresh_score_group_1
			
			thresh_score_default_1 = thresh_score_query_1
			thresh_score_default_2 = 0.10

			peak_distance_thresh_1 = 500
			thresh_fdr_peak_tf_GRaNIE = 0.2
			upstream_tripod = peak_distance_thresh_1
			# type_id_tripod = 0
			# type_id_tripod = 1
			type_id_tripod = select_config['type_id_tripod']
			filename_save_annot_2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod,type_id_tripod)
			# thresh_size_1 = 20

			# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
			# filename_save_annot_2 = filename_save_annot
			filename_save_annot2_ori = '%s.%s.%d.%s'%(filename_save_annot_2,thresh_score_query_1,thresh_size_1,method_type_group)

		flag_group_load_1 = 1
		if flag_group_load_1>0:
			# load feature group estimation for peak or peak and TF
			n_clusters = 50
			peak_loc_ori = peak_read.columns
			feature_query_vec = peak_loc_ori
			# the overlap matrix; the group member number and frequency of each group; the group assignment for the two feature types
			df_overlap_compare, dict_group_basic_1, df_group_1, df_group_2 = self.test_query_feature_group_load_1(data=[],feature_type_vec=feature_type_vec_query,feature_query_vec=feature_query_vec,method_type_group=method_type_group,input_file_path=input_file_path,
																													save_mode=1,output_file_path=output_file_path,filename_prefix_save='',filename_save_annot=filename_save_annot2_ori,output_filename='',verbose=0,select_config=select_config)

			self.df_group_pre1 = df_group_1
			self.df_group_pre2 = df_group_2
			self.dict_group_basic_1 = dict_group_basic_1
			self.df_overlap_compare = df_overlap_compare

		flag_query2 = 1
		if flag_query2>0:
			# select the feature type for group query
			# feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']

			flag_load_2 = 1
			if flag_load_2>0:
				feature_type_1, feature_type_2 = feature_type_vec_2_ori[0:2]
				if feature_type_2 in ['peak_tf']:
					feature_type_vec_2 = [feature_type_1] + ['peak_gene']
				else:
					feature_type_vec_2 = [feature_type_1,feature_type_2]

				method_type_dimension = 'SVD'
				n_components = 50
				type_id_group = select_config['type_id_group']
				filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
				reconstruct = 0
				# load latent matrix;
				# recontruct: 1, load reconstructed matrix;
				flag_combine = 1
				dict_latent_query1 = self.test_query_feature_embedding_load_1(data=[],feature_query_vec=[],motif_id_query='',motif_id='',feature_type_vec=feature_type_vec_2,method_type_vec=[],method_type_dimension=method_type_dimension,
																				n_components=n_components,reconstruct=reconstruct,peak_read=[],rna_exprs=[],flag_combine=flag_combine,
																				load_mode=0,input_file_path='',
																				save_mode=0,output_file_path='',output_filename='',filename_prefix_save=filename_prefix_save_2,filename_save_annot='',
																				verbose=0,select_config=select_config)

				dict_feature = dict_latent_query1

			# n_neighbors = 30
			# n_neighbors = 50
			n_neighbors = 100
			if 'neighbor_num' in select_config:
				n_neighbors = select_config['neighbor_num']
			n_neighbors_query = n_neighbors+1

			# query the neighbors of feature query
			flag_neighbor_query=1
			if flag_neighbor_query>0:
				# list_query1 = [[feature_nbrs_1,dist_nbrs_1],[feature_nbrs_2,dist_nbrs_2]]
				list_query1 = self.test_query_feature_neighbor_load_1(dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,n_neighbors=n_neighbors,input_file_path=input_file_path,
																		save_mode=save_mode,output_file_path=output_file_path,verbose=0,select_config=select_config)

				feature_nbrs_1,dist_nbrs_1 = list_query1[0]
				feature_nbrs_2,dist_nbrs_2 = list_query1[1]

				field_id1, field_id2 = 'feature_nbrs', 'dist_nbrs'
				field_query_1 = [field_id1,field_id2]
				# query_num = len(field_query_1)
				feature_type_num = len(feature_type_vec_query)
				dict_neighbor = dict()
				for i2 in range(feature_type_num):
					feature_type_query = feature_type_vec_query[i2]
					dict_neighbor[feature_type_query] = dict(zip(field_query_1,list_query1[i2]))
				self.dict_neighbor = dict_neighbor
					
			# flag_motif_query_1=1
			flag_motif_query_pre1=0
			if flag_motif_query_pre1>0:
				folder_id = 1
				if 'folder_id' in select_config:
					folder_id = select_config['folder_id']
				df_peak_file, motif_idvec_query = self.test_query_file_annotation_load_1(data_file_type_query=data_file_type_query,folder_id=folder_id,save_mode=1,verbose=verbose,select_config=select_config)

				motif_query_num = len(motif_idvec_query)
				motif_idvec = ['%s.%d'%(motif_id_query,i1) for (motif_id_query,i1) in zip(motif_idvec_query,np.arange(motif_query_num))]
				filename_list1 = np.asarray(df_peak_file['filename'])
				file_num1 = len(filename_list1)
				motif_idvec_2 = []
				for i1 in range(file_num1):
					filename_query = filename_list1[i1]
					b = filename_query.find('.bed')
					motif_id2 = filename_query[0:b]
					motif_idvec_2.append(motif_id2)

				print('motif_idvec_query: ',len(motif_idvec_query),motif_idvec_query[0:5])
			
				# sel_num1 = 12
				sel_num1 = -1
				if sel_num1>0:
					motif_idvec_query = motif_idvec_query[0:sel_num1]
					motif_idvec = motif_idvec[0:sel_num1]
				select_config.update({'motif_idvec_query':motif_idvec_query,'motif_idvec':motif_idvec})

				motif_idvec_query = select_config['motif_idvec_query']
				motif_idvec = select_config['motif_idvec']
				query_num_ori = len(motif_idvec_query)

			input_filename = '%s/test_peak_file.ChIP-seq.pbmc.1_2.2.copy2.txt'%(input_file_path_query_2)
			df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_annot1_1 = df_annot_ori.drop_duplicates(subset=['motif_id2'])
			# folder_id_query1 = select_config['folder_id']
			# id1 = (df_annot1_1['folder_id']==folder_id_query1)
			# df_annot_1 = df_annot1_1.loc[id1,:]
			df_annot_1 = df_annot1_1
			print('df_annot_ori, df_annot_1: ',df_annot_ori.shape,df_annot_1.shape)

			motif_idvec_query = df_annot_1.index.unique()
			motif_idvec_1 = df_annot_1['motif_id1']
			motif_idvec_2 = df_annot_1['motif_id2']
			df_annot_1.index = np.asarray(df_annot_1['motif_id2'])
			dict_motif_query_1 = dict(zip(motif_idvec_2,list(motif_idvec_query)))

			motif_query_num = len(motif_idvec_query)
			motif_num2 = len(motif_idvec_2)
			query_num_ori = len(motif_idvec_2)
			print('motif_idvec_query, motif_idvec_2: ',motif_query_num,motif_num2)

			column_signal = 'signal'
			column_motif = '%s.motif'%(method_type_feature_link)
			column_pred1 = '%s.pred'%(method_type_feature_link)
			column_score_1 = 'score_pred1'
			df_score_annot = []

			column_idvec = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_idvec[0:3]

			# load_mode = 1
			load_mode = 0
			flag_sort = 1
			flag_unduplicate_query = 1
			ascending = False
			column_score_query = 'score_pred1'
			if load_mode>0:
				# copy the score estimation
				if len(df_score_annot)==0:
					# load_mode_query_2 = 1  # 1, load the peak-TF correlation score; 2, load the estimated score 1; 3, load both the peak-TF correlation score and the estimated score 1;
					load_mode_query_2 = 2  # load the estimated score 1
					df_score_annot_query1, df_score_annot_query2 = self.test_query_score_annot_1(data=[],df_score_annot=[],input_file_path='',load_mode=load_mode_query_2,save_mode=0,verbose=0,select_config=select_config)
					df_score_annot = df_score_annot_query2

				if flag_sort>0:
					df_score_annot = df_score_annot.sort_values(by=column_score_query,ascending=ascending)

				if flag_unduplicate_query>0:
					df_score_annot = df_score_annot.drop_duplicates(subset=[column_id2,column_id3])
				df_score_annot.index = np.asarray(df_score_annot[column_id2])

				column_score_query1 = '%s.%s'%(method_type_feature_link,column_score_1)
				# id1 = (df_score_annot[column_id3]==motif_id_query)
				# df_score_annot_query = df_score_annot.loc[id1,:]
				# df_score_annot_query = df_score_annot_query.drop_duplicates(subset=[column_id2,column_id3])
			else:
				column_score_query1 = '%s.score'%(method_type_feature_link)

			motif_query_num = len(motif_idvec_query)
			query_num_1 = motif_query_num
			# query_num_1 = query_num_ori
			# stat_chi2_correction = True
			# stat_fisher_alternative = 'greater'
			list_score_query_1 = []
			interval_save = True
			config_id_load = select_config['config_id_load']
			config_id_2 = select_config['config_id_2']
			config_group_annot = select_config['config_group_annot']
			flag_scale_1 = select_config['flag_scale_1']
			type_query_scale = flag_scale_1

			model_type_id1 = 'LogisticRegression'
			# select_config.update({'model_type_id1':model_type_id1})
			if 'model_type_id1' in select_config:
				model_type_id1 = select_config['model_type_id1']

			motif_vec_group2_query2 = ['NR2F1','RCOR1','STAG1','TAF1','ZNF24','ZNF597']
			# motif_vec_group2_query_2 = ['VDR']
			# motif_vec_group2_query_2 = ['STAT1','IKZF1','RUNX3']
			# motif_vec_group2_query_2 = ['STAT1']
			motif_vec_group2_query_2 = ['STAT1','IKZF1','RUNX3','MYB','YY1']
			# motif_vec_group2_query_2 = ['MYB','MAX','YY1']

			beta_mode = select_config['beta_mode']
			# motif_id_1 = select_config['motif_id_1']
			if beta_mode>0:
				iter_vec_1 = [0]
			else:
				iter_vec_1 = np.arange(query_num_1)

			file_path_query_pre1 =  output_file_path_query

			method_type_feature_link = select_config['method_type_feature_link']
			n_neighbors = select_config['neighbor_num']
			peak_loc_ori = peak_read.columns
			# df_pre1_ori = pd.DataFrame(index=peak_loc_ori)
			method_type_group = select_config['method_type_group']

			dict_motif_data = self.dict_motif_data
			method_type_query = method_type_feature_link
			motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']
			# peak_loc_ori = motif_data_query1.index
			
			motif_data_query1 = motif_data_query1.loc[peak_loc_ori,:]
			motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_ori,:]
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			motif_data = motif_data_query1
			motif_data_score = motif_data_score_query1
			# self.dict_motif_data = dict_motif_data

			list_annot_peak_tf = []
			# iter_vec_1 = [0]
			for i1 in iter_vec_1:
				motif_id_query = motif_idvec_query[i1]

				id1 = (df_annot_1['motif_id']==motif_id_query)
				motif_id2_query = df_annot_1.loc[id1,'motif_id2'][0]
				# motif_id_query = df_annot_1.loc[motif_id2_query,'motif_id']
				motif_id1_query = df_annot_1.loc[motif_id2_query,'motif_id1']
				folder_id_query = df_annot_1.loc[motif_id2_query,'folder_id']
				motif_id1, motif_id2 = motif_id1_query, motif_id2_query
				
				folder_id = folder_id_query
				config_id_2 = dict_config_annot1[folder_id]
				select_config.update({'config_id_2_query':config_id_2})

				input_file_path_query_1 = dict_file_annot1[folder_id_query] # the first level directory
				input_file_path_query_2 = dict_file_annot2[folder_id_query] # the second level directory including the configurations

				# motif_id1 = motif_id
				print('motif_id_query, motif_id1, motif_id2: ',motif_id_query,motif_id1,motif_id2,i1)

				if motif_id_query in motif_vec_group2_query2:
					print('the estimation not included: ',motif_id_query,motif_id1,i1)
					continue

				# if not (motif_id_query in motif_vec_group2_query_2):
				# 	continue

				overwrite_2 = False
				filename_prefix_save = 'test_query.%s'%(method_type_group)
				# filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
				iter_id1 = 0
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)

				ratio_1,ratio_2 = select_config['ratio_1'],select_config['ratio_2']
				filename_annot2 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
				filename_annot_train_pre1 = filename_annot2
				filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
				# filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
				filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id_query,filename_save_annot_1)
				# output_filename = '%s/test_query_train.%s.%s.%s.%s.1.txt'%(output_file_path,method_type_group,filename_annot_train_pre1,motif_id1,filename_save_annot_1)
				# output_filename = '%s/test_query_train.%s.%s.%s.1.txt'%(output_file_path_query,filename_save_annot_query,motif_id1,filename_save_annot_1)
				# filename_query_pre1 = '%s/test_query_train.%s.%s.%s.1.txt'%(file_path_query_pre1,filename_save_annot_query,motif_id1,filename_save_annot_1)
				file_path_query_pre2 = '%s/train1'%(file_path_query_pre1)
				run_id2 = 2
				self.run_id2 = run_id2
				filename_query_pre1 = '%s/test_query_train.%s.%s.%s.%s.txt'%(file_path_query_pre2,filename_save_annot_query,motif_id1,filename_save_annot_1,run_id2)
				
				if (os.path.exists(filename_query_pre1)==True) and (overwrite_2==False):
					print('the file exists: %s'%(filename_query_pre1))
					continue

				flag1=1
				if flag1>0:
					start_1 = time.time()
				# try:
					# load the TF binding prediction file
					# the possible columns: (signal,motif,predicted binding,motif group)
					# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
					filename_prefix_1 = 'test_motif_query_binding_compare'
					# input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.2.txt'%(input_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					
					# flag_group_query_1 = 0
					flag_group_query_1 = 1
					if flag_group_query_1==0:
						# peak_loc_1 = df_pre1.index
						# df_pre1 = df_pre1.loc[peak_loc_ori,:]
						df_query1 = df_pre1
					else:
						load_mode_pre1_1 = 1
						if load_mode_pre1_1>0:
							# load the TF binding prediction file
							# the possible columns: (signal,motif,predicted binding,motif group)
							# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
							folder_id = folder_id_query
							if folder_id in [1,2]:
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.group2.txt'%(input_file_path_query_1,filename_prefix_1,motif_id1,filename_save_annot_2)
							elif folder_id in [0]:
								upstream_tripod_2 = 100
								filename_save_annot_2_pre2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod_2,type_id_tripod)
								input_filename = '%s/%s.peak_loc.flanking_0.%s.1.%s.txt'%(input_file_path_query_1,filename_prefix_1,motif_id1,filename_save_annot_2_pre2)

							if os.path.exists(input_filename==True):
								df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
								peak_loc_1 = df_1.index
								peak_num_1 = len(peak_loc_1)
								print('peak_loc_1: ',peak_num_1)

								method_type_vec_pre2 = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
								# column_motif_vec_2 = ['%s.motif'%(method_type_query) for method_type_query in method_type_vec_2]
								# column_motif_vec_3 = ['%s.pred'%(method_type_query) for method_type_query in method_type_vec_2]
								# column_motif_vec_2 = []
								annot_str_vec = ['motif','pred','score']
								column_vec_query = ['signal']
								for method_type_query in method_type_vec_pre2:
									column_vec_query.extend(['%s.%s'%(method_type_query,annot_str1) for annot_str1 in annot_str_vec])

								# df_query1 = df_1
								df_query1 = df_1.loc[:,column_vec_query]
								print('df_query1: ',df_query1.shape)
								print(df_query1.columns)
								print(df_query1[0:2])
								print(input_filename)
							else:
								print('the file does not exist: %s'%(input_filename))
								df_query1 = pd.DataFrame(index=peak_loc_ori)
								load_mode_pre1_1 = 0

					if (flag_group_query_1==0) or (load_mode_pre1_1>0):
						df_query1_ori = df_query1.copy()
						peak_loc_1 = df_query1.index
						column_vec = df_query1.columns
						df_query1 = pd.DataFrame(index=peak_loc_ori)
						df_query1.loc[peak_loc_1,column_vec] = df_query1_ori
						print('df_query1: ',df_query1.shape)

					column_signal = 'signal'
					if column_signal in df_query1.columns:
						# peak_signal = df_query1['signal']
						peak_signal = df_query1[column_signal]
						id_signal = (peak_signal>0)
						# peak_signal_1_ori = peak_signal[id_signal]
						df_query1_signal = df_query1.loc[id_signal,:]	# the peak loci with peak_signal>0
						peak_loc_signal = df_query1_signal.index
						peak_num_signal = len(peak_loc_signal)
						print('signal_num: ',peak_num_signal)

					if not (column_motif in df_query1.columns):
						peak_loc_motif = peak_loc_ori[motif_data[motif_id_query]>0]
						df_query1.loc[peak_loc_motif,column_motif] = motif_data_score.loc[peak_loc_motif,motif_id_query]

					motif_score = df_query1[column_motif]
					id_motif = (df_query1[column_motif].abs()>0)
					df_query1_motif = df_query1.loc[id_motif,:]	# the peak loci with binding motif identified
					peak_loc_motif = df_query1_motif.index
					peak_num_motif = len(peak_loc_motif)
					# print('motif_num: ',peak_num_motif)
					print('peak_loc_motif ',peak_num_motif)
						
					if peak_num_motif==0:
						continue

					flag_motif_query=0
					stat_chi2_correction = True
					stat_fisher_alternative = 'greater'
					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					input_file_path_2 = '%s/folder_group'%(input_file_path_query_1)
					if os.path.exists(input_file_path_2)==False:
						print('the directory does not exist: %s'%(input_file_path_2))
						os.makedirs(input_file_path_2,exist_ok=True)
					output_file_path_2 = input_file_path_2
					if flag_motif_query>0:
						df_query1_motif = df_query1.loc[id_motif,:] # peak loci with motif
						peak_loc_motif = df_query1_motif.index
						peak_num_motif = len(peak_loc_motif)
						print('peak_loc_motif ',peak_num_motif)
						t_vec_1 = self.test_query_feature_overlap_1(data=df_query1_motif,motif_id_query=motif_id_query,motif_id1=motif_id1,
																		column_motif=column_motif,df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,output_file_path=output_file_path_2,filename_prefix_save='',filename_save_annot='',
																		verbose=verbose,select_config=select_config)
						
						df_overlap_motif, df_group_basic_motif, dict_group_basic_motif, load_mode_query1 = t_vec_1
						
					flag_select_query=0
					method_type_query = method_type_feature_link
					column_pred1 = '%s.pred'%(method_type_query)
					id_pred1 = (df_query1[column_pred1]>0)
					df_pre1 = df_query1
					# df_query1_2 = df_query1.loc[id_pred1,:]
					column_pred2 = '%s.pred_sel'%(method_type_query) # selected peak loci with predicted binding sites
					column_pred_2 = '%s.pred_group_2'%(method_type_query)				
					if flag_select_query>0:
						# select the peak loci predicted with TF binding
						# the selected peak loci
						# thresh_score_query_1 = 0.15
						# thresh_score_query_1 = 0.20
						id_1 = id_pred1
						df_query1_2 = df_query1.loc[id_1,:] # the selected peak loci
						df_pred1 = df_query1_2
						t_vec_2 = self.test_query_feature_overlap_2(data=df_pred1,motif_id_query=motif_id_query,motif_id1=motif_id1,
																		df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,output_file_path=output_file_path_2,filename_prefix_save='',filename_save_annot='',
																		verbose=verbose,select_config=select_config)
						
						df_overlap_query, df_group_basic_query_2, dict_group_basic_query, load_mode_query2 = t_vec_2
						
						# TODO: automatically adjust the group size threshold
						# df_overlap_query = df_overlap_query.sort_values(by=['freq_obs','pval_chi2_'],ascending=[False,True])
						dict_thresh = dict()
						if len(dict_thresh)==0:
							# thresh_value_overlap = 10
							thresh_value_overlap = 0
							thresh_pval_1 = 0.20
							field_id1 = 'overlap'
							field_id2 = 'pval_fisher_exact_'
							# field_id2 = 'pval_chi2_'
						else:
							column_1 = 'thresh_overlap'
							column_2 = 'thresh_pval_overlap'
							column_3 = 'field_value'
							column_5 = 'field_pval'
							column_vec_query1 = [column_1,column_2,column_3,column_5]
							list_query1 = [dict_thresh[column_query] for column_query in column_vec_query1]
							thresh_value_overlap, thresh_pval_1, field_id1, field_id2 = list_query1
						
						# id1 = (df_overlap_query['overlap']>thresh_value_overlap)
						# id2 = (df_overlap_query['pval_chi2_']<thresh_pval_1)
						id1 = (df_overlap_query[field_id1]>thresh_value_overlap)
						id2 = (df_overlap_query[field_id2]<thresh_pval_1)

						df_overlap_query2 = df_overlap_query.loc[id1,:]
						print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape)

						df_overlap_query.loc[id1,'label_1'] = 1
						group_vec_query_1 = np.asarray(df_overlap_query2.loc[:,['group1','group2']])
						print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape,motif_id1)

						self.df_overlap_query = df_overlap_query
						self.df_overlap_query2 = df_overlap_query2
						
						load_mode_2 = load_mode_query2
						if load_mode_2<2:
							# output_filename = '%s/test_query_df_overlap.%s.%s.pre1.1.txt'%(output_file_path,motif_id1,data_file_type_query)
							filename_save_annot2_query = '%s.%s.%s'%(method_type_group,motif_id_query,data_file_type_query)
							output_filename = '%s/test_query_df_overlap.%s.pre1.1.txt'%(output_file_path_2,filename_save_annot2_query)
							df_overlap_query.to_csv(output_filename,sep='\t')

					flag_neighbor_query_1 = 0
					# if flag_group_query_1>0:
					# 	flag_neighbor_query_1 = 1
					flag_neighbor = 1
					flag_neighbor_2 = 1  # query neighbor of selected peak in the paired groups

					feature_type_vec = feature_type_vec_query
					print('feature_type_vec: ',feature_type_vec)
					select_config.update({'feature_type_vec':feature_type_vec})
					self.feature_type_vec = feature_type_vec

					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					if flag_neighbor_query_1>0:
						# feature_type_vec_query = [feature_type_query_1,feature_type_query_2]
						# feature_type_vec = ['latent_peak_motif','latent_peak_tf']
						df_group_1 = self.df_group_pre1
						df_group_2 = self.df_group_pre2

						feature_type_vec = select_config['feature_type_vec']
						feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]
						group_type_vec_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]

						group_type_query1, group_type_query2 = group_type_vec_2[0:2]
						peak_loc_pre1 = df_pre1.index
						df_pre1[group_type_query1] = df_group_1.loc[peak_loc_pre1,method_type_group]
						df_pre1[group_type_query2] = df_group_2.loc[peak_loc_pre1,method_type_group]

						# query peak loci predicted with binding sites using clustering
						start = time.time()
						df_overlap_1 = []
						group_type_vec = ['group1','group2']
						# group_vec_query = ['group1','group2']
						list_group_query = [df_group_1,df_group_2]
						dict_group = dict(zip(group_type_vec,list_group_query))
						
						dict_neighbor = self.dict_neighbor
						dict_group_basic_1 = self.dict_group_basic_1
						# the overlap and the selected overlap above count and p-value thresholds
						# group_vec_query_1: the group enriched with selected peak loci
						column_id2 = 'peak_id'
						df_pre1[column_id2] = np.asarray(df_pre1.index)
						df_pre1 = self.test_query_binding_clustering_1(data1=df_pre1,data2=df_pred1,dict_group=dict_group,dict_neighbor=dict_neighbor,dict_group_basic_1=dict_group_basic_1,
																		df_overlap_1=df_overlap_1,df_overlap_compare=df_overlap_compare,
																		group_type_vec=group_type_vec,feature_type_vec=feature_type_vec,group_vec_query=group_vec_query_1,input_file_path='',																												
																		save_mode=1,output_file_path=output_file_path,output_filename='',
																		filename_prefix_save='',filename_save_annot=filename_save_annot2_2,verbose=verbose,select_config=select_config)
						
						stop = time.time()
						print('query feature group and neighbor annotation for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop-start))

					# method_type_vec_pre2 = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
					method_type_vec_2 = ['TRIPOD','GRaNIE','Pando']
					column_motif_vec_2 = ['%s.motif'%(method_type_query) for method_type_query in method_type_vec_2]
					# column_motif_vec_3 = ['%s.pred'%(method_type_query) for method_type_query in method_type_vec_2]

					# column_vec_query1 = [column_signal,column_motif,column_pred1,column_score_query1]
					# column_vec_query1 = [column_signal,column_motif,column_pred1]+column_motif_vec_2
					column_score_query_1 = '%s.score'%(method_type_feature_link)
					column_vec_query1 = [column_signal]+column_motif_vec_2+[column_motif,column_pred1,column_score_query_1]
					
					if flag_neighbor_query_1>0:
						column_vec_query1_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]
						# column_vec_query2_2 = [feature_type_query_1,feature_type_query_2]
						# column_vec_query2 = column_vec_query1 + column_vec_query1_2 + column_vec_query2_2
						column_vec_query2 = column_vec_query1 + column_vec_query1_2
					else:
						column_vec_query2 = column_vec_query1

					field_id1 = 'peak_tf_corr'
					field_id2 = 'peak_tf_pval_corrected'
					# query peak accessibility-TF expression correlation
					flag_peak_tf_corr = 0
					# flag_peak_tf_corr = 1
					column_peak_tf_corr = 'peak_tf_corr'
					column_query = column_peak_tf_corr
					if column_query in df_pre1.columns:
						flag_peak_tf_corr = 0

					if flag_peak_tf_corr>0:
						column_value = column_peak_tf_corr
						thresh_value=-0.05
						input_file_path_query1 = '%s/folder_correlation'%(input_file_path_query_1)
						if os.path.exists(input_file_path_query1)==False:
							print('the directory does not exist: %s'%(input_file_path_query1))
							os.makedirs(input_file_path_query1,exist_ok=True)

						output_file_path_query1 = input_file_path_query1
						filename_prefix_save_query = 'test_peak_tf_correlation.%s.%s.2'%(motif_id_query,data_file_type_query)
						df_query1, df_annot_peak_tf = self.test_query_compare_peak_tf_corr_1(data=df_pre1,motif_id_query=motif_id_query,motif_id1=motif_id1,motif_id2=motif_id2,
																								column_signal=column_signal,column_value=column_value,thresh_value=thresh_value,
																								motif_data=motif_data,motif_data_score=motif_data_score,
																								peak_read=peak_read,rna_exprs=rna_exprs,
																								flag_query=0,input_file_path=input_file_path_query1,
																								save_mode=1,output_file_path=output_file_path_query1,
																								filename_prefix_save=filename_prefix_save_query,filename_save_annot='',output_filename='',
																								verbose=verbose,select_config=select_config)

						# list_annot_peak_tf.append(df_annot_peak_tf)

					# column_score_1 = 'score_pred1'
					# column_score_query1 = column_score_1
					# column_score_query1 = '%s.%s'%(method_type_feature_link,column_score_1)
					# df_score_annot = []
					load_mode = 0
					if load_mode>0:
						if not (column_score_query1 in df_query1.columns):
							id1 = (df_score_annot[column_id3]==motif_id_query)
							df_score_annot_query = df_score_annot.loc[id1,:]
							peak_loc_2 = df_score_annot_query[column_id2].unique()
							df_query1.loc[peak_loc_2,column_score_query1] = df_score_annot_query.loc[peak_loc_2,column_score_1]

					filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)

					flag_query_2=1
					method_type_feature_link = select_config['method_type_feature_link']
					if flag_query_2>0:
						# flag_select_1=1
						# flag_select_2=1

						flag_select_1=2 # select peak loci with predicted TF binding for class 1
						flag_select_2=2 # select peak loci without predicted TF binding for class 2

						# column_corr_1 = field_id1
						# column_pval = field_id2
						column_corr_1 = 'peak_tf_corr'
						column_pval = 'peak_tf_pval_corrected'
						thresh_corr_1, thresh_pval_1 = 0.30, 0.05
						thresh_corr_2, thresh_pval_2 = 0.1, 0.1
						thresh_corr_3, thresh_pval_2 = 0.05, 0.1
						
						peak_loc_pre1 = df_query1.index
						# run_id_2_ori = 2
						# run_id_2_ori = 3
						run_id_2_ori = select_config['run_id_2_ori']
						flag_sample = select_config['flag_sample']

						if flag_select_1==1:
							# select training sample in class 1
							# find the paired groups with enrichment
							df_annot_vec = [df_group_basic_query_2,df_overlap_query]
							dict_group_basic_2 = self.dict_group_basic_2
							dict_group_annot_1 = {'df_group_basic_query_2':df_group_basic_query_2,'df_overlap_query':df_overlap_query,
													'dict_group_basic_2':dict_group_basic_2}

							key_vec_query = list(dict_group_annot_1.keys())
							for field_id in key_vec_query:
								print(field_id)
								print(dict_group_annot_1[field_id])

							output_file_path_query = file_path_query2
							df_query1 = self.test_query_training_group_pre1(data=df_query1,motif_id1=motif_id1,dict_annot=dict_group_annot_1,
																				method_type_feature_link=method_type_feature_link,
																				dict_thresh=[],thresh_vec=[],input_file_path='',
																				save_mode=1,output_file_path=output_file_path_query,verbose=verbose,select_config=select_config)

							column_corr_1 = 'peak_tf_corr'
							column_pval = 'peak_tf_pval_corrected'
							method_type_feature_link = select_config['method_type_feature_link']
							column_score_query1 = '%s.score'%(method_type_feature_link)
							column_vec_query = [column_corr_1,column_pval,column_score_query1]

							column_pred1 = '%s.pred'%(method_type_feature_link)
							id_pred1 = (df_query1[column_pred1]>0)
							df_pre2 = df_query1.loc[id_pred1,:]
							df_pre2, select_config = self.test_query_feature_quantile_1(data=df_pre2,query_idvec=[],column_vec_query=column_vec_query,save_mode=1,verbose=verbose,select_config=select_config)

							peak_loc_query_1 = []
							peak_loc_query_2 = []
							flag_corr_1 = 1
							flag_score_1 = 0
							flag_enrichment_sel = 1
							peak_loc_query_group2_1 = self.test_query_training_select_pre1(data=df_pre2,column_vec_query=[],
																								flag_corr_1=flag_corr_1,flag_score_1=flag_score_1,
																								flag_enrichment_sel=flag_enrichment_sel,input_file_path='',
																								save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',
																								verbose=verbose,select_config=select_config)
							
							peak_num_group2_1 = len(peak_loc_query_group2_1)
							peak_query_vec = peak_loc_query_group2_1  # the peak loci in class 1
						
						elif flag_select_1==2:
							if run_id_2_ori==2:
								column_pred1 = '%s.pred'%(method_type_feature_link)
								id_pred1 = (df_query1[column_pred1]>0)
							elif run_id_2_ori==3:
								print('with motif scanning')
								column_pred1 = '%s.motif'%(method_type_feature_link)
								id_pred1 = (df_query1[column_pred1].abs()>0)
							
							df_pre2 = df_query1.loc[id_pred1,:]
							peak_query_vec = df_pre2.index
							peak_query_num_1 = len(peak_query_vec)

						# flag_select_2=1
						# print('df_query1: ',df_query1.shape)
						# print(df_query1.columns)
						if flag_select_2==1:
							# select training sample in class 2
							print('feature_type_vec_query: ',feature_type_vec_query)
							peak_vec_2_1, peak_vec_2_2 = self.test_query_training_select_group2(data=df_pre1,motif_id_query=motif_id_query,
																									peak_query_vec_1=peak_query_vec,
																									feature_type_vec=feature_type_vec_query,
																									save_mode=save_mode,verbose=verbose,select_config=select_config)

							peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)

						elif flag_select_2 in [2,3]:
							column_motif = '%s.motif'%(method_type_feature_link)
							column_pred1 = '%s.pred'%(method_type_feature_link)
							id_motif = (df_query1[column_motif].abs()>0)
							# id_pred1 = (df_query1[column_pred1]>0)
							id_pred2 = (~id_pred1)
							id_pred2_1 = (~id_pred1)&(id_motif)
							id_pred2_2 = (~id_pred1)&(~id_motif)

							peak_vec_2_ori = np.asarray(peak_loc_pre1[id_pred2])
							peak_vec_2_1_ori = np.asarray(peak_loc_pre1[id_pred2_1])
							peak_vec_2_2_ori = np.asarray(peak_loc_pre1[id_pred2_2])

							# flag_sample = 0
							if flag_sample>0:
								ratio_1, ratio_2 = select_config['ratio_1'], select_config['ratio_2']
								if flag_select_2 in [2]:
									np.random.shuffle(peak_vec_2_ori)
									peak_query_num_2 = int(peak_query_num_1*ratio_2)
									peak_vec_2 = peak_vec_2_ori[0:peak_query_num_2]

								elif flag_select_2 in [3]:
									np.random.shuffle(peak_vec_2_1_ori)
									np.random.shuffle(peak_vec_2_2_ori)
									peak_query_num2_1 = int(peak_query_num_1*ratio_1)
									peak_query_num2_2 = int(peak_query_num_1*ratio_2)
									peak_vec_2_1 = peak_vec_2_1_ori[0:peak_query_num2_1]
									peak_vec_2_2 = peak_vec_2_2_ori[0:peak_query_num2_2]
							else:
								if flag_select_2 in [2]:
									peak_vec_2 = peak_vec_2_ori
								elif flag_select_2 in [3]:
									peak_vec_2_1 = peak_vec_2_1_ori
									peak_vec_2_2 = peak_vec_2_2_ori

						if flag_select_1>0:
							df_pre1.loc[peak_query_vec,'class'] = 1

						if flag_select_2 in [1,3]:
							df_pre1.loc[peak_vec_2_1,'class'] = -1
							df_pre1.loc[peak_vec_2_2,'class'] = -2

							peak_num_2_1 = len(peak_vec_2_1)
							peak_num_2_2 = len(peak_vec_2_2)
							print('peak_vec_2_1, peak_vec_2_2: ',peak_num_2_1,peak_num_2_2)

						elif flag_select_2 in [2]:
							df_pre1.loc[peak_vec_2,'class'] = -1

							peak_num_2 = len(peak_vec_2)
							print('peak_vec_2: ',peak_num_2)

						peak_query_num_1 = len(peak_query_vec)
						print('peak_query_vec: ',peak_query_num_1)

						peak_vec_1 = peak_query_vec
						sample_id_train = pd.Index(peak_vec_1).union(peak_vec_2,sort=False)

						df_query_pre1 = df_pre1.loc[sample_id_train,:]
						filename_annot2 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
						filename_annot_train_pre1 = filename_annot2

						flag_scale_1 = select_config['flag_scale_1']
						type_query_scale = flag_scale_1

						iter_id1 = 0
						# run_id_2 = 2
						run_id_2 = '%s_%d_%d_%d'%(run_id_2_ori,flag_select_1,flag_select_2,flag_sample)
						filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)
						filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
						# filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
						# output_filename = '%s/test_query_train.%s.%s.2.txt'%(output_file_path,motif_id1,filename_annot2)
						output_filename = '%s/test_query_train.%s.%s.%s.%s.2.txt'%(output_file_path_query,filename_save_annot_query,motif_id1,filename_save_annot_1,run_id_2)
						# df_query_pre1.to_csv(output_filename,sep='\t')

						# flag_shuffle=False
						flag_shuffle=True
						if flag_shuffle>0:
							sample_num_train = len(sample_id_train)
							id_query1 = np.random.permutation(sample_num_train)
							sample_id_train = sample_id_train[id_query1]

						train_valid_mode_2 = 0
						if 'train_valid_mode_2' in select_config:
							train_valid_mode_2 = select_config['train_valid_mode_2']
						if train_valid_mode_2>0:
							sample_id_train_ori = sample_id_train.copy()
							sample_id_train, sample_id_valid, sample_id_train_, sample_id_valid_ = train_test_split(sample_id_train_ori,sample_id_train_ori,test_size=0.1,random_state=0)
						else:
							sample_id_valid = []
						
						sample_id_test = peak_loc_ori
						sample_idvec_query = [sample_id_train,sample_id_valid,sample_id_test]
						# df_query_1 = df_pre1.loc[sample_id_train,:]

						df_pre1[motif_id_query] = 0
						df_pre1.loc[peak_vec_1,motif_id_query] = 1
						# df_pre1.loc[peak_vec_2,motif_id_query] = 0
						peak_num1 = len(peak_vec_1)
						print('peak_vec_1: ',peak_num1)
						print(df_pre1.loc[peak_vec_1,['signal',column_motif,motif_id_query]])

						# feature_type_vec_query = ['latent_peak_motif','latent_peak_tf']
						# feature_type_query_1, feature_type_query_2 = feature_type_vec_query[0:2]

						print('motif_id_query, motif_id: ',motif_id_query,motif_id1,i1)
						iter_num = 5
						flag_train1 = 1
						if flag_train1>0:
							# model_type_id1 = 'XGBClassifier'
							# model_type_id1 = 'LogisticRegression'
							# # select_config.update({'model_type_id1':model_type_id1})
							# if 'model_type_id1' in select_config:
							# 	model_type_id1 = select_config['model_type_id1']
							print('feature_type_vec_query: ',feature_type_vec_query)
							key_vec = np.asarray(list(dict_feature.keys()))
							print('dict_feature: ',key_vec)
							peak_loc_pre1 = df_pre1.index
							id1 = (df_pre1['class']==1)
							peak_vec_1 = peak_loc_pre1[id1]
							peak_query_num1 = len(peak_vec_1)

							train_id1 = select_config['train_id1']
							flag_scale_1 = select_config['flag_scale_1']
							type_query_scale = flag_scale_1

							# file_path_query_pre2 = dict_file_annot2[folder_id_query]
							file_path_query_pre2 = dict_file_annot2[folder_id_query_1]
							# output_file_path_query = file_path_query_pre2
							# output_file_path_query = '%s/train1'%(file_path_query_pre2)
							output_file_path_query = '%s/train%s'%(file_path_query_pre2,run_id_2)
							output_file_path_query2 = '%s/model_train_1'%(output_file_path_query)
							if os.path.exists(output_file_path_query2)==False:
								print('the directory does not exist: %s'%(output_file_path_query2))
								os.makedirs(output_file_path_query2,exist_ok=True)

							model_path_1 = output_file_path_query2
							select_config.update({'model_path_1':model_path_1})
							select_config.update({'file_path_query_1':file_path_query_pre2})

							# filename_prefix_save = 'test_query.%s'%(method_type_group)
							filename_prefix_save = 'test_query.pred'
							# filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
							iter_id1 = 0
							filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)
							# filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)
							filename_save_annot_query = 'pred'
							# filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
							filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id_query,filename_save_annot_1)
							
							# output_filename = '%s/test_query_train.%s.%s.%s.%s.1.txt'%(output_file_path,method_type_group,filename_annot_train_pre1,motif_id1,filename_save_annot_1)
							# run_id2 = 1
							# run_id2 = 2
							# run_id2 = self.run_id2
							output_filename = '%s/test_query_train.%s.%s.%s.%s.1.txt'%(output_file_path_query,filename_save_annot_query,motif_id_query,filename_save_annot_1,run_id_2)
									
							df_pre2 = self.test_query_compare_binding_train_unit1(data=df_pre1,peak_query_vec=peak_loc_pre1,peak_vec_1=peak_vec_1,
																					motif_id_query=motif_id_query,
																					dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,
																					sample_idvec_query=sample_idvec_query,
																					motif_data=motif_data_query1,
																					flag_scale=flag_scale_1,input_file_path=input_file_path,
																					save_mode=1,output_file_path=output_file_path_query,output_filename=output_filename,
																					filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,
																					verbose=verbose,select_config=select_config)

					stop_1 = time.time()
					print('TF binding prediction for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop_1-start_1))
				
				# except Exception as error:
				# 	print('error! ',error, motif_id_query,motif_id1,motif_id2,i1)
				# 	# return

			if len(list_annot_peak_tf)>0:
				df_annot_peak_tf_1 = pd.concat(list_annot_peak_tf,axis=0,join='outer',ignore_index=False)
				output_filename = '%s/test_query_df_annot.peak_tf.%s.1.txt'%(output_file_path,filename_save_annot2_1)
				df_annot_peak_tf_1.to_csv(output_filename,sep='\t')

			if len(list_score_query_1)>0:
				df_score_query_2 = pd.concat(list_score_query_1,axis=0,join='outer',ignore_index=False)
				filename_save_annot2_1 = '%s.%s'%(method_type_group,data_file_type_query)
				# run_id2 = 1
				run_id2 = 2
				output_filename = '%s/test_query_df_score.%s.%d.txt'%(output_file_path,filename_save_annot2_1,run_id2)
				df_score_query_2.to_csv(output_filename,sep='\t')

	

	def run_pre1(self,chromosome='1',run_id=1,species='human',cell=0,generate=1,chromvec=[],testchromvec=[],metacell_num=500,peak_distance_thresh=100,
						highly_variable=1,upstream=100,downstream=100,type_id_query=1,thresh_fdr_peak_tf=0.2,path_id=2,save=1,type_group=0,type_group_2=0,type_group_load_mode=0,
						method_type_group='phenograph.20',thresh_size_group=50,thresh_score_group_1=0.15,method_type_feature_link='joint_score_pre1.thresh3',neighbor_num=30,model_type_id='XGBClassifier',typeid2=0,folder_id=1,
						config_id_2=1,config_group_annot=1,ratio_1=0.25,ratio_2=2,flag_group=-1,train_id1=1,flag_scale_1=1,beta_mode=0,motif_id_1='',query_id1=-1,query_id2=-1,query_id_1=-1,query_id_2=-1,train_mode=0,config_id_load=-1):
		
		chromosome = str(chromosome)
		run_id = int(run_id)
		species_id = str(species)
		# cell = str(cell)
		cell_type_id = int(cell)
		print('cell_type_id: %d'%(cell_type_id))
		metacell_num = int(metacell_num)
		peak_distance_thresh = int(peak_distance_thresh)
		highly_variable = int(highly_variable)
		upstream, downstream = int(upstream), int(downstream)
		if downstream<0:
			downstream = upstream
		type_id_query = int(type_id_query)

		thresh_fdr_peak_tf = float(thresh_fdr_peak_tf)
		type_group = int(type_group)
		type_group_2 = int(type_group_2)
		type_group_load_mode = int(type_group_load_mode)
		method_type_group = str(method_type_group)
		thresh_size_group = int(thresh_size_group)
		thresh_score_group_1 = float(thresh_score_group_1)
		method_type_feature_link = str(method_type_feature_link)
		neighbor_num = int(neighbor_num)
		model_type_id1 = str(model_type_id)
		typeid2 = int(typeid2)
		folder_id = int(folder_id)
		config_id_2 = int(config_id_2)
		config_group_annot = int(config_group_annot)
		ratio_1 = float(ratio_1)
		ratio_2 = float(ratio_2)
		flag_group = int(flag_group)
		train_id1 = int(train_id1)
		flag_scale_1 = int(flag_scale_1)
		beta_mode = int(beta_mode)
		motif_id_1 = str(motif_id_1)

		path_id = int(path_id)
		run_id_save = int(save)
		if run_id_save<0:
			run_id_save = run_id

		config_id_load = int(config_id_load)

		celltype_vec = ['CD34_bonemarrow','pbmc']
		flag_query1=1
		if flag_query1>0:
			query_id1 = int(query_id1)
			query_id2 = int(query_id2)
			query_id_1 = int(query_id_1)
			query_id_2 = int(query_id_2)
			train_mode = int(train_mode)
			# data_file_type = 'pbmc'
			# data_file_type = 'CD34_bonemarrow'
			data_file_type = celltype_vec[cell_type_id]
			print('data_file_type: %s'%(data_file_type))
			run_id = 1
			type_id_feature = 0
			metacell_num = 500
			# print('query_id1, query_id2: ',query_id1,query_id2)

			if path_id==1:
				save_file_path_default = '../data2/data_pre2'
				root_path_2 = '.'
				root_path_1 = '../data2'
			elif path_id==2:
				root_path_1 = '/data/peer/yangy4/data1'
				root_path_2 = '%s/data_pre2/data1_2'%(root_path_1)
				save_file_path_default = root_path_2

			select_config = {'root_path_1':root_path_1,'root_path_2':root_path_2,
								'data_file_type':data_file_type,
								'type_id_feature':type_id_feature,
								'metacell_num':metacell_num,
								'run_id':run_id,
								'upstream_tripod':upstream,
								'downstream_tripod':downstream,
								'type_id_tripod':type_id_query,
								'thresh_fdr_peak_tf_GRaNIE':thresh_fdr_peak_tf,
								'path_id':path_id,
								'run_id_save':run_id_save,
								'type_id_group':type_group,
								'type_id_group_2':type_group_2,
								'type_group_load_mode':type_group_load_mode,
								'method_type_group':method_type_group,
								'thresh_size_group':thresh_size_group,
								'thresh_score_group_1':thresh_score_group_1,
								'method_type_feature_link':method_type_feature_link,
								'neighbor_num':neighbor_num,
								'model_type_id1':model_type_id1,
								'typeid2':typeid2,
								'folder_id':folder_id,
								'config_id_2':config_id_2,
								'config_group_annot':config_group_annot,
								'ratio_1':ratio_1,
								'ratio_2':ratio_2,
								'train_id1':train_id1,
								'flag_scale_1':flag_scale_1,
								'beta_mode':beta_mode,
								'motif_id_1':motif_id_1,
								'query_id1':query_id1,'query_id2':query_id2,
								'query_id_1':query_id_1,'query_id_2':query_id_2,
								'train_mode':train_mode,
								'config_id_load':config_id_load,
								'save_file_path_default':save_file_path_default}
			
			# self.test_peak_motif_query_1(select_config)
			verbose = 1
			flag_score_1 = 0
			flag_score_2 = 0
			flag_compare_1 = 1

			flag_group_1 = flag_group
			flag_15=(flag_group_1==15)
			if flag_15>0:
				# self.test_query_compare_binding_pre1_5_1_recompute_5_ori(data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)
				self.test_query_compare_binding_pre1_5_1_recompute_5(data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)
				
				type_query = 0
				df_score_query = self.test_query_compare_binding_pre1_5_1_basic_1(data=[],type_query=type_query,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

			flag_16=(flag_group_1==16)
			if flag_16>0:
				self.test_query_compare_binding_pre1_5_1_recompute_5_1(data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

			flag_17=(flag_group_1==17)
			if flag_17>0:
				dict_pre1 = self.test_query_binding_pred_compare_1(data=[],feature_query_vec=[],input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

			flag_20=(flag_group_1==20)
			if flag_20>0:
				# type_query = 0
				type_query = 1
				run_id2_vec = [3]
				for run_id_2 in run_id2_vec:
					for flag_sample in [0,1]:
						run_id_2_ori = run_id_2
						select_config.update({'run_id_2_ori':run_id_2_ori,'flag_sample':flag_sample})
						df_score_query = self.test_query_compare_binding_pre1_5_1_basic_1(data=[],type_query=type_query,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

			flag_21=(flag_group_1==21)
			if flag_21>0:
				run_id2_vec = [2,3]
				# run_id2_vec = [3]
				# for run_id_2 in run_id2_vec:
				# 	for flag_sample in [0,1]:
				# 		run_id_2_ori = run_id_2
				# 		select_config.update({'run_id_2_ori':run_id_2_ori,'flag_sample':flag_sample})
				# 		self.test_query_compare_binding_pre1_5_1_recompute_5_2(data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

				for run_id_2 in run_id2_vec:
					for flag_sample in [0,1]:
						run_id_2_ori = run_id_2
						select_config.update({'run_id_2_ori':run_id_2_ori,'flag_sample':flag_sample})
						type_query = 1
						df_score_query = self.test_query_compare_binding_pre1_5_1_basic_1(data=[],type_query=type_query,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)


def run(chromosome,run_id,species,cell,generate,chromvec,testchromvec,metacell_num,peak_distance_thresh,
			highly_variable,upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
			method_type_group,thresh_size_group,thresh_score_group_1,method_type_feature_link,neighbor_num,model_type_id,typeid2,folder_id,
			config_id_2,config_group_annot,ratio_1,ratio_2,flag_group,train_id1,flag_scale_1,beta_mode,motif_id_1,query_id1,query_id2,query_id_1,query_id_2,train_mode,config_id_load):
	
	if path_id==1:
		file_path_1 = '../data2'
	else:
		file_path_1 = '/data/peer/yangy4/data1'
	test_estimator1 = _Base2_2_pre1(file_path=file_path_1)

	test_estimator1.run_pre1(chromosome,run_id,species,cell,generate,chromvec,testchromvec,metacell_num,peak_distance_thresh,
								highly_variable,upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
								method_type_group,thresh_size_group,thresh_score_group_1,method_type_feature_link,neighbor_num,model_type_id,typeid2,folder_id,
								config_id_2,config_group_annot,ratio_1,ratio_2,flag_group,train_id1,flag_scale_1,beta_mode,motif_id_1,query_id1,query_id2,query_id_1,query_id_2,train_mode,config_id_load)
		
def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="1", help="experiment id")
	parser.add_option("-f","--chromosome", default="1", help="Chromosome name")
	parser.add_option("-g","--generate", default="0", help="whether to generate feature vector: 1: generate; 0: not generate")
	parser.add_option("-c","--chromvec",default="1",help="chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-t","--testchromvec",default="10",help="test chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-i","--species",default="0",help="species id")
	parser.add_option("-b","--cell",default="0",help="cell type")
	parser.add_option("--metacell",default="500",help="metacell number")
	parser.add_option("--peak_distance",default="500",help="peak distance")
	parser.add_option("--highly_variable",default="1",help="highly variable")
	parser.add_option("--upstream",default="100",help="TRIPOD upstream")
	parser.add_option("--downstream",default="-1",help="TRIPOD downstream")
	parser.add_option("--typeid1",default="0",help="TRIPOD type_id_query")
	parser.add_option("--thresh_fdr_peak_tf",default="0.2",help="GRaNIE thresh_fdr_peak_tf")
	parser.add_option("--path1",default="2",help="file_path_id")
	parser.add_option("--save",default="-1",help="run_id_save")
	parser.add_option("--type_group",default="0",help="type_id_group")
	parser.add_option("--type_group_2",default="0",help="type_id_group_2")
	parser.add_option("--type_group_load_mode",default="1",help="type_group_load_mode")
	parser.add_option("--method_type_group",default="MiniBatchKMeans.50",help="method_type_group")
	parser.add_option("--thresh_size_group",default="50",help="thresh_size_group")
	parser.add_option("--thresh_score_group_1",default="0.15",help="thresh_score_group_1")
	parser.add_option("--method_type_feature_link",default="joint_score_pre1.thresh3",help='method_type_feature_link')
	parser.add_option("--neighbor",default='30',help='neighbor num')
	parser.add_option("--model_type",default="XGBClassifier",help="model_type")
	parser.add_option("--typeid2",default="0",help="type_id_query_2")
	parser.add_option("--folder_id",default="1",help="folder_id")
	parser.add_option("--config_id_2",default="1",help="config_id_2")
	parser.add_option("--config_group_annot",default="1",help="config_group_annot")
	parser.add_option("--ratio_1",default="0.25",help="ratio_1")
	parser.add_option("--ratio_2",default="2",help="ratio_2")
	parser.add_option("--flag_group",default="-1",help="flag_group")
	parser.add_option("--train_id1",default="1",help="train_id1")
	parser.add_option("--flag_scale_1",default="1",help="flag_scale_1")
	parser.add_option("--beta_mode",default="0",help="beta_mode")
	parser.add_option("--motif_id_1",default="1",help="motif_id_1")
	parser.add_option("--q_id1",default="-1",help="query id1")
	parser.add_option("--q_id2",default="-1",help="query id2")
	parser.add_option("--q_id_1",default="-1",help="query_id_1")
	parser.add_option("--q_id_2",default="-1",help="query_id_2")
	parser.add_option("--train_mode",default="0",help="train_mode")
	parser.add_option("--config_id",default="-1",help="config_id_load")

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':

	opts = parse_args()
	run(opts.chromosome,
		opts.run_id,
		opts.species,
		opts.cell,
		opts.generate,
		opts.chromvec,
		opts.testchromvec,
		opts.metacell,
		opts.peak_distance,
		opts.highly_variable,
		opts.upstream,
		opts.downstream,
		opts.typeid1,
		opts.thresh_fdr_peak_tf,
		opts.path1,
		opts.save,
		opts.type_group,
		opts.type_group_2,
		opts.type_group_load_mode,
		opts.method_type_group,
		opts.thresh_size_group,
		opts.thresh_score_group_1,
		opts.method_type_feature_link,
		opts.neighbor,
		opts.model_type,
		opts.typeid2,
		opts.folder_id,
		opts.config_id_2,
		opts.config_group_annot,
		opts.ratio_1,
		opts.ratio_2,
		opts.flag_group,
		opts.train_id1,
		opts.flag_scale_1,
		opts.beta_mode,
		opts.motif_id_1,
		opts.q_id1,
		opts.q_id2,
		opts.q_id_1,
		opts.q_id_2,
		opts.train_mode,
		opts.config_id)







