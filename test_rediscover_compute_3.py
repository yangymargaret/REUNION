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
# from test_annotation_9_2_copy1_2 import _Base2_2_1
from test_rediscover_compute_2 import _Base2_2_1
from test_group_1 import _Base2_group1

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

class _Base2_2_pre1(_Base2_2_1):
	"""Base class for peak-TF association estimation.
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
	def test_file_path_query_1_ori(self,input_file_path='',run_id=1,select_config={}):

		# input_file_path1 = self.save_path_1
		data_file_type = select_config['data_file_type']
		root_path_1 = select_config['root_path_1']
		input_file_path1 = root_path_1
		data_file_type_id1 = 0
		type_id_feature = select_config['type_id_feature']

		filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
		select_config_query = {'data_path':input_file_path,	
									'filename_save_annot_1':filename_save_annot_1,
									'filename_save_annot_pre1':filename_save_annot_1}

		return select_config_query

	## file_path query
	# query the peak-TF motif scanning matrix of the methods
	def test_file_path_query_2(self,method_type_vec=[],select_config={}):

		# input_file_path1 = self.save_path_1
		data_file_type = select_config['data_file_type']
		root_path_1 = select_config['root_path_1']
		root_path_2 = select_config['root_path_2']
		input_file_path1 = root_path_1

		method_type_num = len(method_type_vec)
		data_file_type = select_config['data_file_type']
		data_file_type_annot = data_file_type.lower()
		metacell_num = select_config['metacell_num']

		dict_query1 = dict()	# the motif data file path of the method query
		dict_query2 = dict()	# the file path of the method query
			
		for method_type_id in range(method_type_num):
			method_type = method_type_vec[method_type_id]
			filename_motif = ''
			filename_motif_score = ''
			pre_config = select_config['config_query'][method_type]
			metacell_num_query,run_id_query = pre_config['metacell_num'], pre_config['run_id']
			file_path1 = '%s/%s'%(root_path_2,method_type)
			file_path2 = '%s/%s/metacell_%d/run%d'%(file_path1,data_file_type_annot,metacell_num_query,run_id_query)
				
			dict1 = {'motif_data':filename_motif,'motif_data_score':filename_motif_score}
			dict_query1.update({method_type:dict1})
			dict_query2.update({method_type:file_path1})

		select_config['filename_motif_data'] = dict_query1
		select_config['input_file_path_query'] = dict_query2

		return select_config

	## the configuration of the different methods
	def test_config_query_1(self,method_type_vec=[],select_config={}):

		method_type_num = len(method_type_vec)
		dict_feature = dict()
		data_file_type = select_config['data_file_type']

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
				pre_config.update({'type_id_region':type_id_region,
									'exclude_exons':exclude_exons,
									'upstream':upstream,'downstream':downstream,'method':method,
									'padj_thresh':padj_thresh,
									'run_id':run_id_query,
									'metacell_num':metacell_num_query})

			elif method_type in ['TRIPOD']:
				run_id_query = 0
				metacell_num_query = 500
				upstream, downstream = 100, 100
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
				run_id_query = 1
				if run_id_query==1:
					metacell_num_query = 500
				else:
					metacell_num_query = 100

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
	def test_config_query_2(self,input_file_path='',file_path_motif_score='',dict_annot=[],method_type_vec=[],data_file_type='',select_config={}):

		if data_file_type=='':
			data_file_type = select_config['data_file_type']

		data_file_type_query = data_file_type
		data_file_type_annot = dict_annot[data_file_type_query]
		root_path_1 = select_config['root_path_1']
		# run_id_1 = 0

		input_file_path_query = input_file_path
		data_path_save_1 = input_file_path
		select_config.update({'data_path_save_local':input_file_path_query,
								'file_path_motif_score_2':file_path_motif_score,
								'file_path_motif_score':file_path_motif_score,
								'data_file_type_annot':data_file_type_annot,
								'data_path_save_1':data_path_save_1})

		type_id_feature = select_config['type_id_feature']
		run_id1 = select_config['run_id1']
		filename_save_annot = '%s.%d.%d'%(data_file_type_query,type_id_feature,run_id1)
		filename_1 = '%s/test_rna_meta_ad.%s.h5ad'%(input_file_path_query,filename_save_annot)
		filename_2 = '%s/test_atac_meta_ad.%s.h5ad'%(input_file_path_query,filename_save_annot)
		# filename_3_ori = '%s/test_query.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
		# filename_3_ori = '%s/test_rna_meta_ad.pbmc.0.1.meta_scaled_exprs.2.txt'%(input_file_path_query,data_file_type_query)
		filename_3_ori = '%s/test_rna_meta_ad.%s.meta_scaled_exprs.2.txt'%(input_file_path_query,filename_save_annot)

		select_config.update({'filename_rna':filename_1,'filename_atac':filename_2,
								'filename_rna_exprs_1':filename_3_ori})
		
		return select_config

	## file_path and configuration query
	def test_query_config_pre1_1(self,data_file_type_query='',method_type_vec=[],file_path_query1='',file_path_query2='',folder_idvec=[0],config_idvec=[0],flag_config_1=1,select_config={}):

		if flag_config_1>0:
			if data_file_type_query=='':
				data_file_type_query = select_config['data_file_type']

			if len(method_type_vec)==0:
				method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre2.thresh3','joint_score_pre1.thresh22','joint_score_pre2.thresh6']

			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']

			input_file_path_query = root_path_2
			input_file_path = '%s/%s_peak'%(data_file_type_query)

			file_save_path_1 = input_file_path
			select_config.update({'file_path_peak_tf':file_save_path_1})
			# peak_distance_thresh = 100
			peak_distance_thresh = 500
			filename_prefix_1 = 'test_motif_query_binding_compare'
			method_type_vec_query = method_type_vec

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

			dict_file_annot1 = dict()
			dict_file_annot2 = dict()
			config_group_annot = select_config['config_group_annot']
			type_query_scale = 0

			dict_file_annot1.update({folder_id_query:file_path_query1})
			dict_file_annot2.update({folder_id_query:file_path_query2})

			dict_query1 = dict(zip(folder_idvec,config_idvec))
			select_config.update({'dict_file_annot1':dict_file_annot1,'dict_file_annot2':dict_file_annot2,'dict_config_annot1':dict_query1})

		return select_config

	## group query
	def test_query_association_pre1_group1_1(self,data=[],feature_type_query='',dict_feature=[],feature_type_vec=[],method_type='',field_query=[],peak_read=[],rna_exprs=[],flag_cluster_query_1=1,flag_cluster_query_2=0,n_components=50,iter_id=-1,config_id_load=-1,input_file_path='',overwrite=True,
												save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		# flag_cluster_query_1 = 0
		file_path1 = self.save_path_1
		test_estimator_1 = _Base2_group1(file_path=file_path1,select_config=select_config)

		# perform feature clustering
		type_id_compute, type_id_feature = 0, 0
		if 'type_id_compute' in select_config:
			type_id_compute = select_config['type_id_compute']

		if 'type_id_feature' in select_config:
			type_id_feature = select_config['type_id_feature']

		filename_prefix_default_1 = filename_prefix_save
		if flag_cluster_query_1>0:
			if feature_type_query=='':
				feature_type_query = 'peak'
			# output_file_path = file_save_path2
			# filename_prefix_save = '%s.group'%(filename_prefix_default_1)
			# filename_prefix_save = '%s.group'%(filename_prefix_default_1)
			if filename_prefix_save=='':
				filename_prefix_save = '%s.group'%(feature_type_query)
			else:
				filename_prefix_save = '%s.%s.group'%(filename_prefix_default_1,feature_type_query)

			# if len(field_query)==0:
			# 	field_query = ['latent_mtx','component_mtx','reconstruct_mtx']

			dict_query_1 = data
			list1 = []
			column_1 = 'alpha0'
			for field_id in field_query:
				df_query = []
				if field_id in dict_query_1:
					df_query_1 = dict_query_1[field_id]
					if len(df_query_1)>0:
						t_columns = df_query_1.columns.difference([column_1],sort=False)
						df_query = df_query_1.loc[:,t_columns]
					else:
						df_query = df_query_1
				list1.append(df_query)
			
			feature_type_2 = field_query[0]
			latent_mtx = list1[0]
			feature_mtx_query = latent_mtx
			if method_type=='':
				method_type_dimension = select_config['method_type_dimension']
				method_type = method_type_dimension

			method_type_query = method_type
			# print('feature_mtx, method_type_query: ',feature_mtx_query.shape,method_type_query)
			
			flag_iter_2 = 1
			flag_clustering_1 = 1

			feature_query1 = latent_mtx.index
			n_components_query = latent_mtx.shape[1]

			filename_prefix_save_2 = '%s.%s.%s'%(filename_prefix_save,feature_type_2,method_type_query)
			# filename_prefix_save_2 = '%s.%s'%(filename_prefix_save,method_type_query,iter_id_query1)
			method_type_group = 'MiniBatchKMeans'
			filename_save_annot_2 = '%s.%d.1'%(method_type_group,n_components_query)

			if output_file_path=='':
				output_file_path = select_config['file_path_group_query']

			output_filename = '%s/%s.sse_query.%s.txt'%(output_file_path,filename_prefix_save_2,filename_save_annot_2)
			filename1 = output_filename
			
			# overwrite_2 = False
			overwrite_2 = True
			if (os.path.exists(filename1) == True) and (overwrite_2==False):
				print('the file exists: %s' % (filename1))
				df_cluster_query1 = pd.read_csv(filename1, index_col=0, sep='\t')
				flag_iter_2 = 0
			else:
				# save_mode_2 = 1
				# test_estimator_1.test_query_cluster_pre1(adata=[],feature_mtx=latent_mtx,method_type=method_type,n_clusters=n_clusters_pre,
				# 											neighbors=20,save_mode=save_mode_2,output_filename=output_filename,verbose=verbose,select_config=select_config)
				# n_clusters_pre = 300
				n_clusters_pre1 = 100
				n_clusters_pre2 = 300
				interval = 10
				interval_2 = 20
				# cluster_num_vec = list(np.arange(2,20))+list(np.arange(20,n_clusters_pre+interval,interval))
				# cluster_num_vec = list(np.arange(2,20))+list(np.arange(20,n_clusters_pre1+interval,interval))+list(np.arange(n_clusters_pre1,n_clusters_pre2+interval_2,interval_2))
				cluster_num_vec = list(np.arange(2,20))+list(np.arange(20,n_clusters_pre1+interval,interval))+list(np.arange(n_clusters_pre1,n_clusters_pre2+interval_2,interval_2))
				# cluster_num_vec = [20,50,100]
				select_config.update({'cluster_num_vec':cluster_num_vec})

			n_components = n_components_query
			# type_id_compute, type_id_feature = 0, 0
			config_vec_1 = [n_components, type_id_compute, type_id_feature]
			select_config.update({'config_vec_1':config_vec_1})

			flag_config1=1
			method_type_vec_1 = ['MiniBatchKMeans', 'phenograph', 'AgglomerativeClustering']
			if 'method_type_vec_group' in select_config:
				method_type_vec_group = select_config['method_type_vec_group']
				method_type_vec_1 = method_type_vec_group
				
			method_type_vec_1 = np.asarray(method_type_vec_1)
			
			if flag_config1>0:
				neighbors_vec = [20, 30] # the neighbors in phenograph clustering
				column_1 = 'method_type_group_neighbor'
				if column_1 in select_config:
					method_type_group_neighbor = select_config[column_1]
					neighbor_num_1 = method_type_group_neighbor
					neighbors_vec = [neighbor_num_1]

				n_clusters_vec = [30, 50, 100] # the number of clusters
				distance_threshold_vec = [20, 50, -1] # the distance in agglomerative clustering
				
				# distance_threshold_vec = [20, -1] # the distance in agglomerative clustering
				# distance_threshold_vec = [-1] # the distance in agglomerative clustering
				# linkage_type_idvec = [0, 1]
				metric = 'euclidean'
				linkage_type_idvec = [0]
				field_query = ['neighbors_vec', 'n_clusters_vec', 'distance_threshold_vec', 'linkage_type_idvec']

				list1 = [neighbors_vec, n_clusters_vec, distance_threshold_vec, linkage_type_idvec]
				dict_config1 = dict(zip(field_query, list1))
				for field1 in field_query:
					if (not (field1 in select_config)) or (overwrite==True):
						select_config.update({field1: dict_config1[field1]})

				distance_threshold_pre, linkage_type_id_pre, neighbors_pre, n_clusters_pre = -1, 0, 20, 100
				list_config = test_estimator_1.test_cluster_query_config_1(method_type_vec=method_type_vec_1,
																			distance_threshold=distance_threshold_pre,
																			linkage_type_id=linkage_type_id_pre,
																			neighbors=neighbors_pre,
																			n_clusters=n_clusters_pre,
																			metric=metric,
																			select_config=select_config)

			# perform feature group estimation
			test_estimator_1.test_query_group_1(data=feature_mtx_query,adata=[],feature_type_query=feature_type_query,list_config=list_config,flag_iter_2=flag_iter_2,flag_clustering_1=flag_clustering_1,
												save_mode=1,output_file_path=output_file_path,output_filename=output_filename,filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot,verbose=verbose,select_config=select_config)

		# flag_cluster_query_2 = 0
		# query signal value of each cluster
		dict_feature_2 = dict()
		if flag_cluster_query_2>0:
			data_file_type_query = select_config['data_file_type_query']
			
			# method_type_query = method_type_vec_dimension[0]
			method_type_query = method_type
			# filename_prefix_save = filename_prefix_save_1
			filename_prefix_save_2 = '%s.%s'%(filename_prefix_save,method_type_query)

			filename_thresh_annot1 = '%d_%d.%d' % (n_components, type_id_compute, type_id_feature)
			filename_annot_1 = '%s.%s'%(filename_save_annot,filename_thresh_annot1)
			
			# input_file_path_query = '%s/group1'%(input_file_path)
			input_file_path_query = select_config['file_path_group_query']
			input_filename = '%s/%s.feature_dimension.%s.1.h5ad'%(input_file_path_query,filename_prefix_save_2,filename_annot_1)
			adata = sc.read_h5ad(input_filename)
			print('adata', adata)

			flag_config2=1
			method_type_vec_1 = ['MiniBatchKMeans', 'phenograph', 'AgglomerativeClustering']
			method_type_vec_1 = np.asarray(method_type_vec_1)
			if flag_config2>0:
				neighbors_vec = [10, 15, 20, 30] # the neighbors in phenograph clustering
				n_clusters_vec = [30, 50, 100] # the number of clusters
				distance_threshold_vec = [20, 50, -1] # the distance in agglomerative clustering
				metric = 'euclidean'
				linkage_type_idvec = [0]
				field_query = ['neighbors_vec', 'n_clusters_vec', 'distance_threshold_vec', 'linkage_type_idvec']
						
				list1 = [neighbors_vec, n_clusters_vec, distance_threshold_vec, linkage_type_idvec]
				dict_config1 = dict(zip(field_query, list1))
				for field1 in field_query:
					if not (field1 in select_config):
						select_config.update({field1: dict_config1[field1]})
			
				distance_threshold_pre, linkage_type_id_pre, neighbors_pre, n_clusters_pre = -1, 0, 20, 100
				list_config2 = test_estimator_1.test_cluster_query_config_1(method_type_vec=method_type_vec_1,
																			distance_threshold=distance_threshold_pre,
																			linkage_type_id=linkage_type_id_pre,
																			neighbors=neighbors_pre,
																			n_clusters=n_clusters_pre,
																			metric=metric,
																			select_config=select_config)

			query_num2 = len(list_config2)
			title = 'Clusters'
			save_mode_2 = 1
			# type_id_compute, type_id_feature = select_config['type_id_compute'], select_config['type_id_feature']
			for i2 in range(query_num2):
				t_vec_1 = list_config2[i2]
				method_type_query, method_type_annot_query, n_clusters, neighbors, distance_threshold, linkage_type_id, metric = t_vec_1
				# print('method_type: ', t_vec_1)
				method_type_id1 = method_type_annot_query

				adata.obs[method_type_id1] = pd.Categorical(adata.obs[method_type_id1])
				group_vec = adata.obs[method_type_id1].unique()
				group_num = len(group_vec)
				print('method_type: ', t_vec_1, group_num)

				flag_plot1=0
				method_type_query1 = method_type
				method_type_vec_dimension = [method_type_query1]
				method_type_dimension = method_type_vec_dimension[0]
				n_components_query = n_components
				iter_id_query1 = iter_id
				# filename_annot1 = '%s.%d_%d.%d'%(method_type_dimension,n_components_query, type_id_compute, type_id_feature)
				filename_annot1 = '%s.%s.%d_%d.%d'%(method_type_dimension,iter_id_query1,n_components_query, type_id_compute, type_id_feature)
				
				flag_signal_1 = 1
				dict_feature_query = dict()
				if flag_signal_1>0:
					df_obs = adata.obs
					cluster_query = df_obs[method_type_annot_query]
					filename_prefix_save_2 = 'test_query.signal.%s' % (filename_annot1)
					save_mode_2 = 1
					# scale_type = 3
					scale_type = 2
					flag_plot = 1

					feature_type_vec_query = feature_type_vec
					dict_feature_query = dict_feature
					feature_type1, feature_type2 = feature_type_vec_query[0:2]
					feature_vec_1 = df_obs.index

					# column_id1, column_id2 = column_idvec[0:2]
					if len(dict_feature_query)==0:
						# feature_vec_2 = df_feature_link[column_id1].unique()
						# motif_query_vec = feature_vec_2
						# feature_vec_1 = df_feature_link[column_id2].unique()
						peak_query_vec = feature_vec_1
						
						peak_read_1 = peak_read.loc[:,peak_query_vec]
						# print('peak_read_1: ',peak_read_1.shape)
						dict_feature_query = {feature_type1:rna_exprs,feature_type2:peak_read_1}
					
					# feature_type_query = feature_type2
					feature_type_query = feature_type1
					df_feature_1 = dict_feature_query[feature_type_query]
					feature_mtx_query = df_feature_1.T
					print('feature_mtx_query, feature_type_query: ',feature_mtx_query.shape,feature_type_query)
					# thresh_group_size = 10
					thresh_group_size = 5
					flag_ratio = 1
					dict_feature_2 = test_estimator_1.test_query_cluster_signal_1(method_type=method_type_annot_query,
																					feature_mtx=feature_mtx_query,
																					cluster_query=cluster_query,
																					df_obs=df_obs,
																					thresh_group_size=thresh_group_size,
																					scale_type=scale_type,
																					type_id_query=1,
																					flag_plot=flag_plot,
																					flag_ratio=flag_ratio,
																					filename_prefix='',
																					filename_prefix_save=filename_prefix_save_2,
																					save_mode=save_mode_2,
																					output_file_path=output_file_path,
																					output_filename='',
																					select_config=select_config)

	# perform clustering of peak loci or peak loci and TFs based on the low-dimensional embeddings
	# rewrite the function
	def test_query_feature_clustering_pre1_1(self,data=[],dict_feature=[],feature_type_vec_1=['gene','peak'],feature_type_vec_2=[],method_type_vec_dimension=['SVD'],method_type_vec_group=['phenograph'],feature_mode=1,type_id_group=0,n_components=50,metric='euclidean',subsample_ratio=-1,subsample_ratio_vec=[],peak_read=[],rna_exprs=[],flag_cluster_1=1,flag_cluster_2=0,flag_combine=0,overwrite=False,overwrite_2=False,input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_query_1 = 1
		if flag_query_1>0:
			file_path1 = self.save_path_1
			run_id = select_config['run_id']
			# test_estimator_1 = test_annotation_11_3_2._Base2_train5_2_pre1(file_path=file_path1,run_id=run_id,select_config=select_config)
			# flag_cluster_query_1 = 1
			# flag_cluster_query_2 = 0
			# n_components = 50
			flag_cluster_query_1 = flag_cluster_1
			flag_cluster_query_2 = flag_cluster_2

			# feature_type_query = 'peak'
			# dict_feature = []
			# feature_type_vec = ['gene','peak']
			# method_type_query = method_type_vec_dimension[0]
			method_type_dimension = method_type_vec_dimension[0]
			field_query_ori = ['df_latent','df_component','reconstruct_mtx']
			# field_id1 = 'df_latent'
			field_query = ['df_latent']

			if len(feature_type_vec_2)==0:
				# if feature_mode in [1,2]:
				# 	# peak read and rna exprs are given
				# 	# annot_str_vec = ['latent_peak_tf','latent_peak_motif']
				# 	# annot_str_vec = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
				# 	feature_type_vec_2 = ['peak_tf','peak_motif','peak_motif_ori','peak_tf_link','peak_mtx']
				# else:
				# 	# annot_str_vec = ['latent_peak_mtx','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
				# 	feature_type_vec_2 = ['peak_mtx','peak_motif_ori','peak_tf_link']

				feature_type_vec_2 = ['peak_tf','peak_motif']
					
			field_query_2 = ['latent_%s'%(feature_type) for feature_type in feature_type_vec_2]
			annot_str_vec = field_query_2

			iter_id = -1
			config_id_load = -1
			overwrite_2 = False

			# file_save_path = output_file_path
			# type_id_group_2 = 0
			# type_id_group_2 = select_config['type_id_group_2'] # the feature type

			# field_query_2 = ['latent_peak','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
			# list_query1 = [latent_peak,latent_peak_motif,latent_peak_motif_ori,latent_peak_tf_link]
			# field_id2 = field_query_2[type_id_group_2]
			# latent_peak_query = list_query1[type_id_group_2]
			# dict_query_1 = data
			# latent_mtx_query = dict_query_1[field_id]

			# print(field_id2,latent_peak_query.shape,type_id_group_2)
			# print(field_id2,latent_mtx_query.shape,type_id_group_2)
			# print(latent_peak_query[0:2])
			
			# dict_query_1 = data
			# subsample_ratio = -1
			# subsample_ratio = 0.1
			# if subsample_ratio>0:
			# 	latent_peak_ori = latent_peak_query.copy()
			# 	peak_query_vec = np.asarray(latent_peak_query.index)
			# 	np.random.shuffle(peak_query_vec)
			# 	peak_num_ori = len(peak_query_vec)
			# 	peak_num1 = int(peak_num_ori*subsample_ratio)
			# 	latent_peak_query = latent_peak_ori.loc[peak_query_vec[0:peak_num1],:]
			# 	print('latent_peak_query: ',latent_peak_query.shape)

			# if (feature_mode in [1,2]) and (type_id_group_2==0):
			# 	df_latent_1 = pd.concat([latent_peak_query,latent_tf],axis=0,join='outer',ignore_index=False)
			# else:
			# 	df_latent_1 = latent_peak_query

			dict_query_1 = data
			if len(dict_query_1)==0:
				data_file_type_query = select_config['data_file_type']
				# method_type_dimension = 'SVD'
				# n_components = 50
				type_id_group = select_config['type_id_group']
				
				# filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
				filename_prefix_save_pre2 = '%s.pre%d.%d'%(data_file_type_query,feature_mode,type_id_group)
				reconstruct = 0
				# load latent matrix;
				# recontruct: 1, load reconstructed matrix;
				# flag_combine = 0
				# dict_file = dict()
				dict_file = self.dict_file_feature
				dict_latent_query1 = self.test_query_feature_embedding_load_1(data=[],dict_file=dict_file,feature_query_vec=[],motif_id_query='',motif_id='',
																				feature_type_vec=feature_type_vec_2,
																				method_type_vec=[],method_type_dimension=method_type_dimension,
																				n_components=n_components,reconstruct=reconstruct,
																				peak_read=[],rna_exprs=[],flag_combine=flag_combine,
																				load_mode=0,input_file_path=input_file_path,
																				save_mode=0,output_file_path='',output_filename='',
																				filename_prefix_save=filename_prefix_save_pre2,filename_save_annot='',
																				verbose=0,select_config=select_config)

				dict_query_1 = dict_latent_query1

			query_num1 = len(field_query_2)
			query_num2 = len(subsample_ratio_vec)
			if (query_num2>0) or (subsample_ratio>0):
				dict_query_1 = self.test_query_subsample_1(data=dict_query_1,field_query=field_query_2,
															subsample_ratio_vec=subsample_ratio_vec,subsample_ratio=subsample_ratio,
															save_mode=save_mode,verbose=verbose,select_config=select_config)

			# df_latent_1 = latent_mtx_query
			# print('df_latent_1: ',df_latent_1.shape)
			# dict_query_1.update({field_id1:df_latent_1})
			# method_type_vec_1 = ['MiniBatchKMeans','phenograph']
			# method_type_vec_1 = ['phenograph']
			# select_config.update({'method_type_vec_group':method_type_vec_1})
			column_1 = 'method_type_vec_group'
			if (column_1 in select_config):
				method_type_vec_group = select_config[column_1]
			else:
				select_config.update({column_1:method_type_vec_group})

			# neighbors_vec = [20, 30] # the neighbors in phenograph clustering
			neighbors_vec = [20] # the neighbors in phenograph clustering
			
			n_clusters_vec = [30, 50, 100] # the number of clusters
			distance_threshold_vec = [20, 50, -1] # the distance in agglomerative clustering
			# metric = 'euclidean'
			linkage_type_idvec = [0]

			field_query_pre2 = ['neighbors_vec','n_clusters_vec','distance_threshold_vec','linkage_type_idvec']
			list1 = [neighbors_vec, n_clusters_vec, distance_threshold_vec, linkage_type_idvec]
			dict_config1 = dict(zip(field_query_pre2,list1))
			overwrite_query = False
			for field_id in field_query_pre2:
				if (not (field_id in select_config)) or (overwrite_query==True):
					select_config.update({field_id: dict_config1[field_id]})
				print('field_id, query_value: ',field_id,select_config[field_id])

			# file_save_path = input_file_path
			file_save_path = output_file_path
			feature_type_query_1 = 'peak'
			method_type_clustering_init = 'MiniBatchKMeans'
			select_config.update({'method_type_clustering_init':method_type_clustering_init})

			field_num = len(field_query_2)
			for i2 in range(field_num):
				type_id_group_2 = i2
				# output_file_path_2 = '%s/group1_%d'%(file_save_path,type_id_group_2+1)
				output_file_path_2 = '%s/group1_%d_2'%(file_save_path,type_id_group_2+1)
				if os.path.exists(output_file_path_2)==False:
					print('the directory does not exist: %s'%(output_file_path_2))
					os.makedirs(output_file_path_2,exist_ok=True)

				annot_str1 = annot_str_vec[type_id_group_2]
				feature_type_query = annot_str1
				# filename_prefix_save_2 = '%s.feature_group.%s.%d'%(filename_prefix_save,annot_str1,type_id_group)
				filename_prefix_save_2 = '%s.feature_group.%s'%(filename_prefix_save,annot_str1)
				# filename_prefix_save_2 = '%s.feature_group.%s.2'%(filename_prefix_save,annot_str1)
				
				filename_save_annot_2 = filename_save_annot
				# dict_query_1 = dict()
				feature_type_ori = feature_type_vec_2[i2]
				field_id = field_query_2[i2]

				if field_id in dict_query_1:
					dict_query_pre2 = dict_query_1
					field_id_query = field_id
				else:
					dict_query_pre2 = dict_query_1[feature_type_ori]
					field_id_query = 'df_latent'

				field_query = [field_id_query]
				feature_mtx = dict_query_pre2[field_id_query]
				print('feature_mtx ',feature_mtx.shape,field_id_query,i2)
				print(feature_mtx[0:2])

				column_1 = 'n_component_sel'
				if column_1 in select_config:
					n_components_query1 = select_config[column_1]
					column_vec = feature_mtx.columns
					column_vec_query = column_vec[0:n_components_query1]
					feature_mtx = feature_mtx.loc[:,column_vec_query]

					print('feature_mtx ',feature_mtx.shape,field_id_query,i2)
					print(feature_mtx[0:2])

				file_path1 = self.save_path_1
				test_estimator_1 = _Base2_group1(file_path=file_path1,select_config=select_config)

				test_estimator_1.test_query_association_pre1_group1_1(data=feature_mtx,feature_type_query=feature_type_query_1,
																		dict_feature=dict_feature,feature_type_vec=feature_type_vec,
																		method_type=method_type_dimension,field_query=field_query,
																		peak_read=peak_read,rna_exprs=rna_exprs,
																		flag_cluster_1=flag_cluster_query_1,flag_cluster_2=flag_cluster_query_2,
																		n_components=n_components,iter_id=iter_id,config_id_load=config_id_load,
																		input_file_path=input_file_path,overwrite=overwrite,overwrite_2=overwrite_2,
																		save_mode=save_mode,output_file_path=output_file_path_2,output_filename='',
																		filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot_2,
																		verbose=verbose,select_config=select_config)

	# perform clustering of peak loci or peak loci and TFs based on the low-dimensional embeddings
	def test_query_feature_clustering_1(self,data=[],dict_feature=[],feature_type_vec=['gene','peak'],feature_type_vec_2=[],method_type_vec_dimension=['SVD'],method_type_vec_group=['phenograph'],feature_mode=3,type_id_group=0,n_components=50,metric='euclidean',subsample_ratio=-1,subsample_ratio_vec=[],peak_read=[],rna_exprs=[],flag_cluster_1=1,flag_cluster_2=0,flag_combine=0,overwrite=False,overwrite_2=False,input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_query_1 = 1
		if flag_query_1>0:
			# file_path1 = self.save_path_1
			# run_id = select_config['run_id']

			flag_cluster_query_1 = 1
			flag_cluster_query_2 = 0
			# n_components = 50

			feature_type_query = 'peak'
			# dict_feature = []
			feature_type_vec = ['gene','peak']
			
			method_type_query = method_type_vec_dimension[0]
			# field_query_ori = ['latent_mtx','component_mtx','reconstruct_mtx']
			# field_query = ['latent_mtx']

			# annot_str_vec = ['latent_peak_tf','latent_peak_motif']
			# annot_str_vec = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']

			if len(feature_type_vec_2)==0:
				feature_type_vec_2 = ['peak_motif','peak_tf']
			field_query_2 = ['latent_%s'%(feature_type_query) for feature_type_query in feature_type_vec_2]
			# annot_str_vec = ['latent_%s'%(feature_type_query) for feature_type_query in feature_type_vec_2]
			annot_str_vec = field_query_2

			iter_id = -1
			config_id_load = -1
			overwrite_2 = False

			# file_save_path_2 = output_file_path
			# type_id_group_2 = 0
			# type_id_group_2 = select_config['type_id_group_2']
			
			# output_file_path_2 = '%s/group1_%d'%(file_save_path_2,type_id_group_2+1)
			# if os.path.exists(output_file_path_2)==False:
			# 	print('the directory does not exist: %s'%(output_file_path_2))
			# 	os.makedirs(output_file_path_2,exist_ok=True)

			# annot_str1 = annot_str_vec[type_id_group_2]
			# filename_prefix_save_2 = '%s.feature_group.%s.%d'%(filename_prefix_save,annot_str1,type_id_group)
			# filename_save_annot_2 = filename_save_annot
			dict_query_1 = dict()
			# field_id1 = 'latent_mtx'
			
			# field_query_2 = ['latent_peak','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
			# list_query1 = [latent_peak,latent_peak_motif,latent_peak_motif_ori,latent_peak_tf_link]
			column_1 = 'type_id_group_2'
			type_id_group_2 = -1
			if column_1 in select_config:
				type_id_group_2 = select_config[column_1]
				type_id_group_2_ori = type_id_group_2

			if type_id_group_2>0:
				type_vec_group_2 = [type_id_group_2]
			else:
				field_num_2 = len(field_query_2)
				type_vec_group_2 = np.arange(field_num_2)
			print('type_vec_group_2: ',type_vec_group_2)

			# method_type_vec_1 = ['MiniBatchKMeans','phenograph']
			method_type_group_name = select_config['method_type_group_name']
			method_type_group_neighbor = select_config['method_type_group_neighbor']
			neighbor_num_1 = method_type_group_neighbor
			
			# method_type_vec_1 = ['phenograph']
			method_type_vec_1 = [method_type_group_name]
			select_config.update({'method_type_vec_group':method_type_vec_1})

			# neighbors_vec = [20, 30] # the neighbors in phenograph clustering
			neighbors_vec = [neighbor_num_1]
			n_clusters_vec = [30, 50, 100] # the number of clusters
			distance_threshold_vec = [20, 50, -1] # the distance in agglomerative clustering
			metric = 'euclidean'
			linkage_type_idvec = [0]
			select_config.update({'neighbors_vec':neighbors_vec,'n_clusters_vec':n_clusters_vec,'distance_threshold_vec':distance_threshold_vec,
									'linkage_type_idvec':linkage_type_idvec})

			file_path_group_query = select_config['file_path_group_query']
			output_file_path_2 = file_path_group_query

			type_id_compute, type_id_feature = 0, 0
			column_2, column_3 = 'type_id_compute', 'type_id_feature'
			if column_2 in select_config:
				type_id_compute = select_config[column_2]

			if column_3 in select_config:
				type_id_feature = select_config[column_3]

			n_component_sel = select_config['n_component_sel']
			filename_save_annot_2 = '%d_%d.%d'%(n_component_sel,type_id_compute,type_id_feature)

			for type_id_query in type_vec_group_2:
				# field_id2 = field_query_2[type_id_group_2]
				# latent_peak_query = list_query1[type_id_group_2]

				field_id2 = field_query_2[type_id_query]
				latent_peak_query = dict_feature[field_id2]

				# print(field_id2,latent_peak_query.shape,type_id_group_2)
				print(field_id2,latent_peak_query.shape,type_id_query)
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

				# if type_id_group_2==0:
				# 	df_latent_1 = pd.concat([latent_peak_query,latent_tf],axis=0,join='outer',ignore_index=False)
				# else:
				# 	df_latent_1 = latent_peak_query

				df_latent_1 = latent_peak_query
				print('df_latent_1: ',df_latent_1.shape)
				# dict_query_1.update({field_id1:df_latent_1})
				dict_query_1.update({field_id2:df_latent_1})
				field_query = [field_id2]

				self.test_query_association_pre1_group1_1(data=dict_query_1,feature_type_query=feature_type_query,dict_feature=[],feature_type_vec=feature_type_vec,method_type=method_type_query,field_query=field_query,
															peak_read=peak_read,rna_exprs=rna_exprs,
															flag_cluster_query_1=flag_cluster_query_1,flag_cluster_query_2=flag_cluster_query_2,n_components=n_components,iter_id=iter_id,config_id_load=config_id_load,input_file_path=input_file_path,overwrite=overwrite_2,
															save_mode=save_mode,output_file_path=output_file_path_2,output_filename='',filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot_2,verbose=verbose,select_config=select_config)

	## query neighbors of feature
	# query neighbors of peak loci
	def test_query_feature_neighbor_pre1_1(self,data=[],n_neighbors=20,return_distance=True,save_mode=1,verbose=0,select_config={}):

		from sklearn.neighbors import NearestNeighbors
		from scipy.stats import poisson, multinomial

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
					# if feature_type_query in ['latent_peak_tf']:
					# 	feature_type_query = 'latent_peak_gene'

					print('find nearest neighbors of peak loci')
					start = time.time()
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

				query_num = len(list_query1)
				for i2 in range(query_num):
					list_query2 = list_query1[i2]
					feature_type_query = feature_type_vec_query[i2]
					feature_nbrs_query, dist_nbrs_query = list_query2[0:2]
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

	## load the feature group query and the peak query
	# load the estimated group label
	def test_query_feature_group_load_pre1(self,data=[],peak_query_vec=[],feature_type_vec=[],method_type_vec=[],peak_read=[],rna_exprs=[],thresh_size=20,type_id_group=0,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		# run_id1 = select_config['run_id']
		# thresh_num1 = 5

		# root_path_1 = select_config['root_path_1']
		# if data_file_type_query in ['CD34_bonemarrow']:
		# 	input_file_path = '%s/data_pre2/data1_2/peak1'%(root_path_1)
		# elif data_file_type_query in ['pbmc']:
		# 	input_file_path = '%s/data_pre2/data1_2/peak2'%(root_path_1)
		
		# input_filename = 'CD34_bonemarrow.pre1.feature_group.latent_peak_motif.0.SVD.feature_dimension.1.50_0.0.df_obs.1.txt'
		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]

		if len(method_type_vec)>0:
			method_type_group = method_type_vec[0]
		else:
			# method_type_group = 'MiniBatchKMeans.50'
			# method_type_group = 'phenograph.30'
			method_type_group =select_config['method_type_group']

		# method_type_dimension = 'SVD'
		# feature_type_query = 'latent_peak_motif'
		method_type_dimension = select_config['method_type_dimension']
		# dict_query1 = dict()
		dict_query1 = data
		dict_query2 = dict()

		if input_file_path=='':
			file_path_group_query = select_config['file_path_group_query']
			input_file_path_query = file_path_group_query
		else:
			input_file_path_query = input_file_path

		load_mode = 0
		if len(dict_query1)==0:
			dict_query1 = dict()
			load_mode = 1
		
		for feature_type_query in feature_type_vec:		
			if load_mode>0:
				# if feature_type_query in ['latent_peak_motif']:
				# 	input_file_path_query = '%s/group1/group1_2'%(input_file_path)

				# elif feature_type_query in ['latent_peak_tf']:
				# 	input_file_path_query = '%s/group1/group1_1'%(input_file_path)

				# elif feature_type_query in ['latent_peak_motif_ori']:
				# 	input_file_path_query = '%s/group1/group1_3'%(input_file_path)

				# elif feature_type_query in ['latent_peak_tf_link']:
				# 	input_file_path_query = '%s/group1/group1_4'%(input_file_path)

				# filename_prefix_1 = '%s.pre1.feature_group.%s.%d.%s'%(data_file_type_query,feature_type_query,type_id_group,method_type_dimension)
				filename_prefix_1 = '%s.%s'%(filename_prefix_save,feature_type_query)
				# input_filename = '%s/%s.feature_dimension.1.50_0.0.df_obs.1.txt'%(input_file_path_query,filename_prefix_1)
				input_filename = '%s/%s.%s.df_obs.1.txt'%(input_file_path_query,filename_prefix_1,filename_save_annot)

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
			# dict_query1_1: (feature_type,df_group), (feature_type_group,df_group_statistics); dict_query1_2: (feature_type,peak_loci)
			dict_query1_1, dict_query1_2 = self.test_query_feature_group_load_pre1(data=[],peak_query_vec=[],feature_type_vec=feature_type_vec,method_type_vec=method_type_vec_group,peak_read=[],rna_exprs=[],
																						thresh_size=thresh_size_1,type_id_group=type_id_group,load_mode=0,input_file_path='',
																						save_mode=1,output_file_path='',output_filename='',filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,verbose=verbose,select_config=select_config)

			feature_type_vec_query = feature_type_vec
			feature_type_query_1,feature_type_query_2 = feature_type_vec_query[0:2]
			# filename_save_annot2_ori = filename_save_annot
			# filename_save_annot2 = '%s.%s_%s'%(filename_save_annot2_ori,feature_type_query_1,feature_type_query_2)

			df_group_1_ori = dict_query1_1[feature_type_query_1]
			df_group_2_ori = dict_query1_1[feature_type_query_2]

			if len(feature_query_vec)==0:
				peak_loc_ori = df_group_1_ori.index
			else:
				peak_loc_ori = feature_query_vec

			# peak_loc_ori = peak_read.columns
			df_group_1 = df_group_1_ori.loc[peak_loc_ori,:]
			df_group_2 = df_group_2_ori.loc[peak_loc_ori,:]

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
				print('the file does not exist: %s'%(input_filename))
				# query the overlap between the pairs of groups
				df_overlap_pre1 = self.test_query_group_overlap_1(df1=df_group_1,df2=df_group_2,feature_query_vec=[],save_mode=1,select_config=select_config)
				column_1, column_2 = 'group1','group2'

				idvec = [column_1]
				df_overlap_pre1[column_1] = np.asarray(df_overlap_pre1.index)
				df_overlap_compare = df_overlap_pre1.melt(id_vars=idvec,var_name=column_2,value_name='overlap')
				df_overlap_compare.index = utility_1.test_query_index(df_overlap_compare,column_vec=[column_1,column_2],symbol_vec=['_'])
				df_overlap_compare['freq_obs'] = df_overlap_compare['overlap']/np.sum(df_overlap_compare['overlap'])
				
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
					feature_vec_query1 = dict_group_annot_1[group_id1]

					for group_id2 in group_vec_2:
						# peak_vec_query2 = dict_1[feature_type_2][group_id2]
						feature_vec_query2 = dict_group_annot_2[group_id2]

						feature_vec_overlap = pd.Index(feature_vec_query1).intersection(feature_vec_query2,sort=False)
						df_overlap.loc[group_id1,group_id2] = len(feature_vec_overlap)
			else:
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
		feature_num_overlap = len(feature_vec_overlap)
		self.df_overlap.loc[group_id1,group_id2] = feature_num_overlap

		return [group_id1,group_id2,feature_num_overlap]

	## query feature enrichment
	# df1: the foreground dataframe; df2: the background dataframe (expected dataframe)
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

	## query the overlap between groups
	# query the overlap between groups
	def test_query_group_overlap_pre1_1(self,df_group_1=[],df_group_2=[],df_overlap_1=[],df_query_compare=[],feature_query_vec=[],column_query='',flag_shuffle=0,stat_chi2_correction=True,stat_fisher_alternative='greater',save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			if column_query=='':
				column_query = 'group'
			
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

			df_query1['overlap_ori'] = df_query_compare.loc[query_id_1,'overlap']

			count1 = np.sum(df_query1['overlap'])
			count2 = np.sum(df_query_compare['overlap'])

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

			# column_vec_query = ['freq_obs','freq_expect','stat_chi2_','pval_chi2_']
			column_vec_query = ['freq_obs','freq_expect']
			list1 = [df_freq_query,df_freq_1]
			for (column_query,query_value) in zip(column_vec_query,list1):
				df_query1[column_query] = query_value

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
	
	## compute feature embeddings
	def test_query_feature_embedding_pre1(self,data=[],dict_feature=[],feature_type_vec=[],method_type='',field_query=[],peak_read=[],rna_exprs=[],
												n_components=50,iter_id=-1,config_id_load=-1,flag_config=1,flag_motif_data_load=1,flag_load_1=1,input_file_path='',overwrite=False,
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		# run_id1 = select_config['run_id']

		# root_path_1 = select_config['root_path_1']
		# if input_file_path=='':
		# 	if data_file_type_query in ['CD34_bonemarrow']:
		# 		input_file_path = '%s/data_pre2/data1_2/peak1'%(root_path_1)
		# 	elif data_file_type_query in ['pbmc']:
		# 		input_file_path = '%s/data_pre2/data1_2/peak2'%(root_path_1)

		flag_config_1 = flag_config
		flag_motif_data_load_1 = flag_motif_data_load
		# flag_load_1 = 1

		method_type_query = method_type
		if method_type=='':
			# method_type_feature_link_query1 = 'joint_score_pre1.thresh22'
			# method_type_feature_link_query1 = 'motif_CIS_BP'
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_query = method_type_feature_link

		# load motif data, RNA-seq and ATAC-seq data
		method_type_vec_query = [method_type_query]
		select_config = self.test_query_load_pre1(data=[],method_type_vec_query=method_type_vec_query,
													flag_config_1=flag_config_1,
													flag_motif_data_load_1=flag_motif_data_load_1,
													flag_load_1=flag_load_1,
													save_mode=save_mode,verbose=verbose,select_config=select_config)

		dict_motif_data = self.dict_motif_data
		key_vec = list(dict_motif_data.keys())
		print('dict_motif_data: ',key_vec)
		print(dict_motif_data)

		feature_mode = select_config['feature_mode']
		peak_read = self.peak_read
		peak_loc_ori = peak_read.columns
		
		if feature_mode in [3]:
			rna_exprs = []
			rna_exprs_unscaled = []
			feature_type_vec = ['peak_mtx','peak_motif_ori']
		else:
			# rna_exprs = self.meta_scaled_exprs
			rna_exprs = self.rna_exprs
			rna_exprs_unscaled = self.meta_exprs_2
			feature_type_vec = ['peak_mtx','peak_motif','peak_motif_ori']
			if feature_mode in [1]:
				feature_type_vec = ['peak_tf','peak_motif','peak_motif_ori']
		
		# query motif data and motif data score of given peak loci
		# query TFs with expressions
		motif_data, motif_data_score, motif_query_vec_1 = self.test_query_motif_data_annotation_1(data=dict_motif_data,data_file_type=data_file_type_query,
																										gene_query_vec=[],feature_query_vec=peak_loc_ori,
																										method_type=method_type_query,
																										peak_read=peak_read,rna_exprs=rna_exprs,
																										save_mode=save_mode,verbose=verbose,select_config=select_config)
		type_id_group = 0
		column_1 = 'type_id_group'
		select_config.update({column_1:type_id_group})
		
		method_type_dimension = select_config['method_type_dimension']
		# method_type_vec_dimension = ['SVD','SVD','SVD']
		num1 = 3
		method_type_vec_dimension = [method_type_dimension]*num1
		
		# n_components = 50
		motif_group = []
		load_mode = 0
		# file_path_query_1 = input_file_path
		output_file_path_default = output_file_path
		
		column_1 = 'file_path_group_query'
		if column_1 in select_config:
			file_path_group_query = select_config[column_1]
		else:
			output_file_path_query = '%s/group%d'%(output_file_path,feature_mode)
			
			if os.path.exists(output_file_path_query)==False:
				print('the directory does not exist:%s'%(output_file_path_query))
				os.makedirs(output_file_path_query,exist_ok=True)

			file_path_group_query = output_file_path_query
			column_1 = 'file_path_group_query'
			select_config.update({column_1:file_path_group_query})

		print('file_path_group_query: ',file_path_group_query)

		filename_prefix_save = '%s.pre%d'%(data_file_type_query,feature_mode)
		filename_save_annot = '1'
		feature_query_vec = peak_loc_ori
		motif_data = motif_data.astype(np.float32)
		output_file_path_2 = file_path_group_query

		# compute feature embedding
		# dict_query1: {'peak_motif':dict1,'peak_tf':dict1}
		# dict1:{'df_latent','df_component'}
		dict_query1 = self.test_query_feature_mtx_1(feature_query_vec=feature_query_vec,
														feature_type_vec=feature_type_vec,
														gene_query_vec=motif_query_vec_1,
														method_type_vec_dimension=method_type_vec_dimension,
														n_components=n_components,
														type_id_group=type_id_group,
														motif_data=motif_data,motif_data_score=motif_data_score,motif_group=motif_group,
														peak_read=peak_read,rna_exprs=rna_exprs,rna_exprs_unscaled=rna_exprs_unscaled,
														load_mode=load_mode,input_file_path=input_file_path,
														save_mode=save_mode,output_file_path=output_file_path_2,filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,
														verbose=verbose,select_config=select_config)
		self.select_config = select_config

		return dict_query1, select_config

	# load df_latent and df_component; compute reconstructed matrix
	def test_query_feature_embedding_load_pre1(self,data=[],dict_file={},method_type_vec=[],feature_type_vec=[],method_type_dimension='SVD',n_components=100,n_component_sel=50,peak_read=[],rna_exprs=[],reconstruct=1,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		# run_id1 = select_config['run_id']
		# thresh_num1 = 5

		load_mode_2 = 1   # use the given filename
		if len(dict_file)==0:
			load_mode_2 = 0

		# root_path_1 = select_config['root_path_1']
		# root_path_2 = select_config['root_path_2']
		# if input_file_path=='':
		# 	input_file_path = '%s/%s_peak'%(root_path_2,data_file_type_query)
	
		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]
		
		if input_file_path=='':
			input_file_path = select_config['file_path_group_query']

		input_file_path_query = input_file_path
		# input_file_path_query = '%s/group1'%(input_file_path)
		# feature_type_vec = ['peak_motif','peak_motif_ori']
		# filename_prefix_1 = '%s'%(data_file_type_query)
		filename_prefix_1 = filename_prefix_save
		# method_type_dimension = 'SVD'
		n_components_query = n_components
		if filename_save_annot=='':
			filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_components_query)
		else:
			filename_save_annot_2 = filename_save_annot

		# the number of components to use is equal to or smaller than the number of components used in dimension reduction
		column_1 = 'n_component_sel'
		if n_component_sel<0:
			if column_1 in select_config:
				n_component_sel = select_config[column_1]
			else:
				n_component_sel = n_components
		print('n_component_sel ',n_component_sel)

		type_query = 0
		if n_component_sel!=n_components:
			type_query = 1
		
		dict_query1 = dict()
		for feature_type_query in feature_type_vec:
			if load_mode_2==0:
				# input_filename_1 = '%s/%s.pre1.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)
				# input_filename_2 = '%s/%s.pre1.df_component.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)
				input_filename_1 = '%s/%s.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)
				input_filename_2 = '%s/%s.df_component.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)
			else:
				input_filename_1 = dict_file[feature_type_query]['df_latent']
				input_filename_2 = dict_file[feature_type_query]['df_component']
				
			dict_query1[feature_type_query] = dict()
			df_latent_query_ori = pd.read_csv(input_filename_1,index_col=0,sep='\t')
			df_component_query_ori = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			
			if type_query>0:
				column_vec_1 = df_latent_query_ori.columns
				column_vec_query1 = column_vec_1[0:n_component_sel]
				df_latent_query = df_latent_query_ori.loc[:,column_vec_query1]

				column_vec_2 = df_component_query_ori.columns
				column_vec_query2 = column_vec_2[0:n_component_sel]
				df_component_query = df_component_query_ori.loc[:,column_vec_query2]
			else:
				df_latent_query = df_latent_query_ori
				df_component_query = df_component_query_ori

			print('df_latent_query_ori, df_latent_query: ',df_latent_query_ori.shape,df_latent_query.shape)
			print(df_latent_query_ori[0:2])
			print(df_latent_query[0:2])

			print('df_component_query_ori, df_component_query: ',df_component_query_ori.shape,df_component_query.shape)
			print(df_component_query_ori[0:2])
			print(df_component_query[0:2])
		
			dict_query1[feature_type_query].update({'df_latent':df_latent_query,'df_component':df_component_query})

			if reconstruct>0:
				reconstruct_mtx = df_latent_query.dot(df_component_query.T)
				# dict_query1.update({feature_type_query:reconstruct_mtx})
				dict_query1[feature_type_query].update({'reconstruct_mtx':reconstruct_mtx})

		return dict_query1

	## compare TF binding prediction
	# perform clustering of peak and TF
	# load low_dimension_embedding
	def test_query_feature_embedding_load_1(self,data=[],dict_file={},feature_query_vec=[],motif_id_query='',motif_id='',feature_type_vec=[],feature_type_vec_group=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,n_component_sel=50,reconstruct=1,peak_read=[],rna_exprs=[],flag_combine=1,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		# load df_latent and df_component; compute reconstructed matrix
		dict_query_1 = self.test_query_feature_embedding_load_pre1(data=data,dict_file=dict_file,
																	method_type_vec=method_type_vec,feature_type_vec=feature_type_vec,
																	method_type_dimension=method_type_dimension,n_components=n_components,n_component_sel=n_component_sel,
																	peak_read=peak_read,rna_exprs=rna_exprs,reconstruct=reconstruct,
																	load_mode=load_mode,input_file_path=input_file_path,
																	save_mode=save_mode,output_file_path=output_file_path,output_filename=output_filename,filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,verbose=verbose,select_config=select_config)


		if save_mode>0:
			self.dict_latent_query_1 = dict_query_1
		
		flag_query1 = 1
		# load the latent components
		# type_id_query_2 = select_config['typeid2']
		# flag_query1 = (type_id_query_2 in [0,2])
		# flag_query1 = 1
		if flag_query1>0:
			feature_type_vec_query = ['latent_%s'%(feature_type_query) for feature_type_query in feature_type_vec]
			feature_type_num = len(feature_type_vec)

			query_mode = 0
			if len(feature_query_vec)>0:
				query_mode = 1
			
			list_1 = []
			column_1 = 'n_component_sel'
			if n_component_sel<0:
				if column_1 in select_config:
					n_component_sel = select_config[column_1]
				else:
					n_component_sel = n_components

			# query the embeddings of the given feature query with the specific number of components
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
				select_config.update({'feature_type_combine':feature_type_combine})
				dict_query1.update({feature_type_combine:latent_mtx_combine})

			self.select_config = select_config

			return dict_query1

	# query peak loci predicted with binding sites using clustering
	# dict_group: the original group assignment query
	def test_query_binding_clustering_1(self,data1=[],data2=[],dict_group=[],dict_neighbor=[],dict_group_basic_1=[],df_overlap_1=[],df_overlap_compare=[],group_type_vec=['group1','group2'],feature_type_vec=[],group_vec_query=[],input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		# feature group query and feature neighbor query
		df_pre1 = data1
		df_query1 = data2

		# feature group query and feature neighbor query
		type_query_group = 0
		if 'type_query_group' in select_config:
			type_query_group = select_config['type_query_group']

		if type_query_group==0:
			df_pre1 = self.test_query_feature_group_neighbor_pre1_2(data=df_pre1,dict_group=dict_group,dict_neighbor=dict_neighbor,
																		group_type_vec=group_type_vec,feature_type_vec=feature_type_vec,
																		group_vec_query=group_vec_query,column_vec_query=[],n_neighbors=30,input_file_path='',
																		save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)
		else:
			parallel=0
			if 'parallel_group' in select_config:
				parallel = select_config['parallel_group']
			df_pre1 = self.test_query_feature_group_neighbor_pre1_2_unit1(data=df_pre1,dict_group=dict_group,dict_neighbor=dict_neighbor,
																			group_type_vec=group_type_vec,feature_type_vec=feature_type_vec,
																			group_vec_query=group_vec_query,column_vec_query=[],n_neighbors=30,parallel=parallel,input_file_path='',
																			save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

		return df_pre1

	# feature group query and feature neighbor query
	# def test_query_feature_group_neighbor_pre1_1(self,data=[],dict_group=[],dict_neighbor=[],dict_group_basic_1=[],dict_thresh=[],df_overlap_1=[],df_overlap_compare=[],group_type_vec=['group1','group2'],feature_type_vec=[],group_vec_query=[],column_vec_query=[],n_neighbors=30,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

	# 	flag_query1 = 1
	# 	if flag_query1>0:
	# 		filename_save_annot_1 = filename_save_annot
	# 		filename_query = '%s/test_query_df_overlap.%s.pre1.1.txt' % (input_file_path, filename_save_annot_1)
	# 		filename_query_2 = '%s/test_query_df_overlap.%s.pre1.2.txt' % (input_file_path, filename_save_annot_1)
			
	# 		input_filename = filename_query
	# 		input_filename_2 = filename_query_2
	# 		load_mode_2 = 0

	# 		if os.path.exists(input_filename)==True:
	# 			# overlap between the paired groups and the enrichment statistical significance value
	# 			df_overlap_query = pd.read_csv(input_filename,index_col=0,sep='\t')
	# 			load_mode_2 = load_mode_2+1
	# 			print('df_overlap_query: ',df_overlap_query.shape)
	# 			print(input_filename)

	# 		if os.path.exists(input_filename_2)==True:
	# 			# group size for each feature type
	# 			df_group_basic_query = pd.read_csv(input_filename_2,index_col=0,sep='\t')
	# 			load_mode_2 = load_mode_2+1
	# 			print('df_group_basic_query: ',df_group_basic_query.shape)
	# 			print(input_filename_2)

	# 		df_query1 = data
	# 		# dict_group_basic_1 = dict_group
	# 		# df_group_1 = dict_group['group1']
	# 		# df_group_2 = dict_group['group2']
	# 		if len(group_type_vec)==0:
	# 			group_type_vec = ['group1','group2']

	# 		list_query1 = [dict_group[group_type_query] for group_type_query in group_type_vec]
	# 		df_group_1, df_group_2 = list_query1[0:2] # group annation of feature query in sequence feature space and peak accessibility feature space

	# 		if load_mode_2<2:
	# 			stat_chi2_correction = True
	# 			stat_fisher_alternative = 'greater'
	# 			# dict_group_basic_2: the enrichment of group assignment for each feature type
	# 			df_overlap_query, df_overlap_mtx, dict_group_basic_2 = self.test_query_group_overlap_pre1_2(data=df_query1,dict_group_compare=dict_group_basic_1,df_group_1=df_group_1,df_group_2=df_group_2,
	# 																											df_overlap_1=df_overlap_1,df_query_compare=df_overlap_compare,flag_sort=1,flag_group=1,
	# 																											stat_chi2_correction=stat_chi2_correction,stat_fisher_alternative=stat_fisher_alternative,
	# 																											save_mode=0,output_filename='',verbose=verbose,select_config=select_config)

	# 			# list_query1 = [dict_group_basic_2[group_type] for group_type in group_vec_query]
	# 			list_query1 = []
	# 			key_vec = list(dict_group_basic_2.keys())
	# 			print('dict_group_basic_2: ',key_vec)

	# 			# if len(group_type_vec)==0:
	# 			# 	# group_type_vec = key_vec
	# 			# 	group_type_vec = ['group1','group2']

	# 			for group_type in group_type_vec:
	# 				df_query = dict_group_basic_2[group_type]
	# 				print('df_query: ',len(df_query),group_type)
	# 				print(df_query[0:2])
	# 				df_query['group_type'] = group_type
	# 				list_query1.append(df_query)

	# 			df_query = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
	# 			flag_sort_2=1
	# 			if flag_sort_2>0:
	# 				df_query = df_query.sort_values(by=['group_type','stat_fisher_exact_','pval_fisher_exact_'],ascending=[True,False,True])
				
	# 			output_filename = '%s/test_query_df_overlap.%s.pre1.2.txt' % (output_file_path,filename_save_annot_1)
	# 			df_query = df_query.round(7)
	# 			df_query.to_csv(output_filename,sep='\t')
	# 			print(output_filename)

	# 		# TODO: automatically adjust the group size threshold
	# 		# df_overlap_query = df_overlap_query.sort_values(by=['freq_obs','pval_chi2_'],ascending=[False,True])
	# 		if len(dict_thresh)==0:
	# 			thresh_value_overlap = 0
	# 			thresh_pval_1 = 0.20
	# 			field_id1 = 'overlap'
	# 			field_id2 = 'pval_fisher_exact_'
	# 			# field_id2 = 'pval_chi2_'
	# 		else:
	# 			column_1 = 'thresh_overlap'
	# 			column_2 = 'thresh_pval_overlap'
	# 			column_3 = 'field_value'
	# 			column_5 = 'field_pval'
	# 			column_vec_query1 = [column_1,column_2,column_3,column_5]
	# 			list_query1 = [dict_thresh[column_query] for column_query in column_vec_query1]
	# 			thresh_value_overlap, thresh_pval_1, field_id1, field_id2 = list_query1
			
	# 		# id1 = (df_overlap_query['overlap']>thresh_value_overlap)
	# 		# id2 = (df_overlap_query['pval_chi2_']<thresh_pval_1)
	# 		id1 = (df_overlap_query[field_id1]>thresh_value_overlap)
	# 		id2 = (df_overlap_query[field_id2]<thresh_pval_1)

	# 		df_overlap_query2 = df_overlap_query.loc[id1,:]
	# 		print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape)

	# 		return df_overlap_query, df_overlap_query2

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
			# column_pred2 = '%s.pred'%(method_type_feature_link) # selected peak loci with predicted binding sites
			column_pred1 = select_config['column_pred1']
			column_pred2 = column_pred1
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
					# print('peak_neighbor_pre2_1, peak_neighbor_pre2_2: ',peak_neighbor_num_1,peak_neighbor_num_2,i2,peak_query)
						
				df_pre1.loc[peak_neighbor_query1,column_query1] += 1
				df_pre1.loc[peak_neighbor_query2,column_query2] += 1

				df_pre1.loc[peak_neighbor_pre2_1,column_pred_7] += 1
				# df_pre1.loc[peak_neighbor_pre2_2,column_pred_8] += 1

			df_pre1[column_pred_8] = df_pre1[column_query1]+df_pre1[column_query2]-df_pre1[column_pred_7]

		return df_pre1

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
				df_pre1 = df_pre1.loc[~(df_pre1.index.duplicated(keep='first')),:]
				print('df_pre1: ',df_pre1.shape)
				print(df_pre1[0:5])
				print(input_filename_query1)
			else:
				print('the file does not exist: %s'%(input_filename_query1))
				print('please provide feature group estimation')
				flag_group_query_1 = 1
				# return

			if flag_group_query_1==0:
				# peak_loc_1 = df_pre1.index
				# df_pre1 = df_pre1.loc[peak_loc_ori,:]
				df_query1 = df_pre1
			else:
				load_mode = 1
				df_query1, load_mode = self.test_query_feature_link_load_pre2(data=[],load_mode=load_mode,input_file_path='',
																				save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',
																				verbose=0,select_config=select_config)

			if (flag_group_query_1==0) or (load_mode>0):
				df_query1_ori = df_query1.copy()
				peak_loc_1 = df_query1.index
				column_vec = df_query1.columns
				df_query1 = pd.DataFrame(index=peak_loc_ori)
				df_query1.loc[peak_loc_1,column_vec] = df_query1_ori
				print('df_query1: ',df_query1.shape)

			return df_query1

	## load feature link
	def test_query_feature_link_load_pre2(self,data=[],load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		if load_mode>0:
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

				# method_type_vec_pre2 = ['TRIPOD','GRaNIE','Pando','insilico_0.1']+[method_type_feature_link]
				method_type_vec_pre2 = [method_type_feature_link]
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
				load_mode = 0

		# if load_mode_pre1_1==0:
		if load_mode==0:
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

		return df_query1, load_mode

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
			filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id_query,data_file_type_query)
			filename_query = '%s/test_query_df_overlap.%s.motif.1.txt' % (input_file_path, filename_save_annot2_2)
			filename_query_2 = '%s/test_query_df_overlap.%s.motif.2.txt' % (input_file_path, filename_save_annot2_2)
			
			column_1 = 'filename_overlap_motif_1'
			column_2 = 'filename_overlap_motif_2'
			if column_1 in select_config:
				filename_query = select_config[column_1]

			if column_2 in select_config:
				filename_query_2 = select_config[column_2]

			input_filename = filename_query
			input_filename_2 = filename_query_2
			load_mode = 0
			# overwrite_2 = False
			overwrite_2 = True
			df_group_basic_motif = []
			dict_group_basic_motif = dict()
			
			if os.path.exists(input_filename)==True:
				if (overwrite_2==False):
					df_overlap_motif = pd.read_csv(input_filename,index_col=0,sep='\t')
					load_mode = load_mode+1
			else:
				print('the file does not exist: %s'%(input_filename))

			if os.path.exists(input_filename_2)==True:
				if (overwrite_2==False):
					df_group_basic_motif = pd.read_csv(input_filename_2,index_col=0,sep='\t')
					load_mode = load_mode+1
					self.df_group_basic_motif = df_group_basic_motif
			else:
				print('the file does not exist: %s'%(input_filename))

			if load_mode<2:
				dict_group_basic_1 = self.dict_group_basic_1
				df_group_1 = self.df_group_pre1
				df_group_2 = self.df_group_pre2
				
				if len(df_overlap_compare)==0:
					df_overlap_compare = self.df_overlap_compare

				stat_chi2_correction = True
				stat_fisher_alternative = 'greater'
				output_filename = filename_query
				output_filename_2 = filename_query_2
				df_overlap_motif, df_overlap_mtx_motif, dict_group_basic_motif = self.test_query_group_overlap_pre1_2(data=df_query1_motif,dict_group_compare=dict_group_basic_1,
																														df_group_1=df_group_1,df_group_2=df_group_2,
																														df_overlap_1=[],df_query_compare=df_overlap_compare,
																														flag_sort=1,flag_group=1,
																														stat_chi2_correction=stat_chi2_correction,
																														stat_fisher_alternative=stat_fisher_alternative,
																														save_mode=1,output_filename=output_filename,output_filename_2=output_filename_2,
																														verbose=verbose,select_config=select_config)
				self.dict_group_basic_motif = dict_group_basic_motif
						
			self.df_overlap_motif = df_overlap_motif

			load_mode_query = load_mode
			return df_overlap_motif, df_group_basic_motif, df_group_basic_motif, load_mode_query

	## feature overlap query
	def test_query_feature_overlap_2(self,data=[],motif_id_query='',motif_id1='',column_motif='',df_overlap_compare=[],input_file_path='',save_mode=1,verbose=0,output_file_path='',filename_prefix_save='',filename_save_annot='',select_config={}):

		flag_select_query=1
		data_file_type_query = select_config['data_file_type']
		method_type_group = select_config['method_type_group']
		# method_type_feature_link = select_config['method_type_feature_link']
		df_query1 = data

		if flag_select_query>0:
			# select the peak loci predicted with TF binding
			# the selected peak loci
			df_pred1 = df_query1
						
			peak_loc_query_group2 = df_pred1.index
			peak_num_group2 = len(peak_loc_query_group2)
			print('peak_loc_query_group2: ',peak_num_group2)

			# feature_query_vec_2 = peak_loc_query_group2
			peak_query_vec = peak_loc_query_group2

			# column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
			# column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)
						
			method_type_query = select_config['method_type_query']
			filename_save_annot2_2 = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)
			filename_query = '%s/test_query_df_overlap.%s.1.txt' % (input_file_path, filename_save_annot2_2)
			filename_query_2 = '%s/test_query_df_overlap.%s.2.txt' % (input_file_path, filename_save_annot2_2)
						
			column_1 = 'filename_overlap_1'
			column_2 = 'filename_overlap_2'
			if column_1 in select_config:
				filename_query = select_config[column_1]

			if column_2 in select_config:
				filename_query_2 = select_config[column_2]

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
																												stat_chi2_correction=stat_chi2_correction,
																												stat_fisher_alternative=stat_fisher_alternative,
																												save_mode=1,output_filename=output_filename,
																												output_filename_2=output_filename_2,
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
				# output_filename = '%s/test_query_df_overlap.%s.pre1.2.txt' % (output_file_path,filename_save_annot2_2)
				output_filename = filename_query_2

				df_query = df_query.round(7)
				df_query.to_csv(output_filename,sep='\t')
				df_group_basic_query_2 = df_query

				self.dict_group_basic_2 = dict_group_basic_2
							
			self.df_overlap_query = df_overlap_query						
			load_mode_query = load_mode_2
			return df_overlap_query, df_group_basic_query_2, dict_group_basic_2, load_mode_query

	## compute peak-TF correlation
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
						
				input_filename_list1 = []
				motif_query_vec = [motif_id_query]
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
			# if (flag_peak_tf_corr_2>0):
			# 	df_annot_peak_tf = self.test_query_compare_peak_tf_corr_2(data=df_pre1,motif_id_query=motif_id_query,
			# 																motif_id1='',motif_id2='',
			# 																column_signal=column_signal,
			# 																column_value=column_value,
			# 																thresh_value=thresh_value,
			# 																motif_data=motif_data,
			# 																motif_data_score=motif_data_score,
			# 																peak_read=peak_read,rna_exprs=rna_exprs,
			# 																flag_query=flag_query,
			# 																input_file_path=input_file_path,
			# 																save_mode=save_mode,output_file_path=output_file_path,
			# 																filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,
			# 																output_filename=output_filename,
			# 																verbose=verbose,select_config=select_config)

			return df_pre1, df_annot_peak_tf

	## compute peak-TF correlation
	# compute peak-TF correlation in peak loci with ChIP-seq signal
	def test_query_compare_peak_tf_corr_2(self,data=[],motif_id_query='',motif_id1='',motif_id2='',column_signal='signal',column_value='',thresh_value=-0.05,motif_data=[],motif_data_score=[],peak_read=[],rna_exprs=[],flag_query=0,input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',output_filename='',verbose=0,select_config={}):

		# flag_query1 = 1
		df_annot_peak_tf = []
		df_pre1 = data
		if column_signal in df_pre1.columns:
			peak_loc_pre1 = df_pre1.index
			id_signal = (df_pre1[column_signal]>0)
			peak_signal = peak_loc_pre1[id_signal]

			# thresh_1 = -0.01
			# thresh_1 = -0.05
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

			return df_annot_peak_tf

	## recompute based on clustering of peak and TF
	# recompute based on training
	def test_query_compare_binding_compute_1(self,data=[],dict_feature=[],motif_id_query='',motif_id='',feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		run_id1 = select_config['run_id']
		print('data_file_type_query: ',data_file_type_query)
		print('run_id: ',run_id1)
		
		if 'method_type_feature_link' in select_config:
			method_type_feature_link = select_config['method_type_feature_link']
			# method_type_feature_link_1 = method_type_feature_link.split('.')[0]
		
		filename_save_annot = '1'
		# the methods used
		method_type_vec = pd.Index(method_type_vec).union([method_type_feature_link],sort=False)

		flag_config_1 = 1
		flag_motif_data_load_1 = 1
		flag_load_1 = 1
		method_type_feature_link_query1 = 'Unify'
		method_type_vec_query1 = [method_type_feature_link_query1]

		select_config = self.test_query_load_pre1(data=[],method_type_vec_query=method_type_vec_query1,flag_config_1=flag_config_1,
													flag_motif_data_load_1=flag_motif_data_load_1,
													flag_load_1=flag_load_1,
													save_mode=1,verbose=verbose,select_config=select_config)

		file_save_path_1 = select_config['file_path_peak_tf']
		if data_file_type_query in ['pbmc']:
			dict_file_annot1 = select_config['dict_file_annot1']
			dict_file_annot2 = select_config['dict_file_annot2']
			dict_config_annot1 = select_config['dict_config_annot1']

			folder_id = select_config['folder_id']
			# group_id_1 = folder_id+1
			file_path_query_1 = dict_file_annot1[folder_id] # the first level directory
			file_path_query_2 = dict_file_annot2[folder_id] # the second level directory including the configurations

			input_file_path = file_path_query_1
			output_file_path = file_path_query_1

			folder_id_query = 2 # the folder to save annotation files
			folder_id_query_1 = folder_id_query
			file_path_query1 = dict_file_annot1[folder_id_query]
			file_path_query2 = dict_file_annot2[folder_id_query]
			input_file_path_query = file_path_query1
			input_file_path_query_2 = '%s/vbak1'%(file_path_query1)
			input_file_path_query_pre2 = input_file_path_query_2
			output_file_path_query = file_path_query1
			output_file_path_query_2 = file_path_query2

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
		# method_type_vec_group_ori = ['MiniBatchKMeans.%d'%(n_clusters_query) for n_clusters_query in [30,50,100]]+['phenograph.%d'%(n_neighbors_query) for n_neighbors_query in [10,15,20,30]]
		
		# n_neighbors_query = 30
		n_neighbors_query = 20
		method_type_group = 'phenograph.%d'%(n_neighbors_query)
		# method_type_group_id = 6
		if 'method_type_group' in select_config:
			method_type_group = select_config['method_type_group']
		print('method_type_group: ',method_type_group)

		method_type_vec_group_ori = [method_type_group]

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
			type_id_tripod = select_config['type_id_tripod']
			filename_save_annot_2 = '%s.%d.%d'%(thresh_fdr_peak_tf_GRaNIE,upstream_tripod,type_id_tripod)

			filename_save_annot2_ori = '%s.%s.%d.%s'%(filename_save_annot_2,thresh_score_query_1,thresh_size_1,method_type_group)

		flag_group_load_1 = 1
		peak_read = self.peak_read
		meta_scaled_exprs = self.meta_scaled_exprs
		rna_exprs = meta_scaled_exprs
		meta_exprs_2 = self.meta_exprs_2
		print('peak_read: ',peak_read.shape)
		print('rna_exprs: ',rna_exprs.shape)
		
		if flag_group_load_1>0:
			# load feature group estimation for peak or peak and TF
			n_clusters = 50
			peak_loc_ori = peak_read.columns
			feature_query_vec = peak_loc_ori
			# the overlap matrix; the group member number and frequency of each group; the group assignment for the two feature types
			df_overlap_compare, dict_group_basic_1, df_group_1, df_group_2 = self.test_query_feature_group_load_1(data=[],feature_type_vec=feature_type_vec_query,
																													feature_query_vec=feature_query_vec,
																													method_type_group=method_type_group,
																													input_file_path=input_file_path,
																													save_mode=1,output_file_path=output_file_path,filename_prefix_save='',filename_save_annot=filename_save_annot2_ori,output_filename='',verbose=0,select_config=select_config)

			self.df_group_pre1 = df_group_1
			self.df_group_pre2 = df_group_2
			self.dict_group_basic_1 = dict_group_basic_1
			self.df_overlap_compare = df_overlap_compare

		flag_query2 = 1
		if flag_query2>0:
			# select the feature type for group query
			# feature_type_vec_group = ['latent_peak_tf','latent_peak_motif']

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
					
			dict_motif_data = self.dict_motif_data

			method_type_query = method_type_feature_link_query1
			print(method_type_query)
			motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']
			
			motif_data_query1 = motif_data_query1.loc[peak_loc_ori,:]
			motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_ori,:]
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			motif_data = motif_data_query1
			motif_data_score = motif_data_score_query1
			# self.dict_motif_data = dict_motif_data

			motif_name_ori = motif_data_query1.columns
			gene_name_expr_ori = rna_exprs.columns
			motif_query_name_expr = pd.Index(motif_name_ori).intersection(gene_name_expr_ori,sort=False)
			print('motif_query_name_expr: ',len(motif_query_name_expr))

			type_motif_query=0
			if 'type_motif_query' in select_config:
				type_motif_query = select_config['type_motif_query']

			if type_motif_query==0:
				input_filename = select_config['file_motif_annot']	# the file to save the TF names for estimation;
				df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
				df_annot_1 = df_annot1_ori.drop_duplicates(subset=['motif_id'])
				motif_idvec_query = df_annot_1['motif_id'].unique()
				
			elif type_motif_query>0:	
				motif_idvec_query = motif_query_name_expr # perform estimation for the TFs with expression
			
			motif_query_num = len(motif_idvec_query)
			print('motif_idvec_query: ',motif_query_num,type_motif_query)
			query_num_ori = motif_query_num

			#column_signal = 'signal'
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
			# method_type_feature_link = select_config['method_type_feature_link']
			n_neighbors = select_config['neighbor_num']
			peak_loc_ori = peak_read.columns
			# df_pre1_ori = pd.DataFrame(index=peak_loc_ori)
			method_type_group = select_config['method_type_group']

			query_id1,query_id2 = select_config['query_id_1'],select_config['query_id_2']
			iter_mode = 0
			query_num_1 = motif_query_num
			iter_vec_1 = np.arange(query_num_1)
			print('query_id1, query_id2: ',query_id1,query_id2)

			if (query_id1>=0) and (query_id1<query_num_1) and (query_id2>query_id1) :
				iter_mode = 1
				start_id1 = query_id1
				start_id2 = np.min([query_id2,query_num_1])
				iter_vec_1 = np.arange(start_id1,start_id2)
				interval_save = False
			else:
				print('query_id1, query_id2: ',query_id1,query_id2)
				return

			run_id_2_ori = 1
			column_1 = 'run_id_2_ori'
			if column_1 in select_config:
				run_id_2_ori = select_config[column_1]
			print('run_id_2_ori: ',run_id_2_ori)
			flag_select_1 = 1
			flag_select_2 = 1
			flag_sample = 1

			column_1 = 'flag_select_1'
			column_2 = 'flag_select_2'
			column_3 = 'flag_sample'

			list1 = [flag_select_1,flag_select_2,flag_sample]
			field_query = [column_1,column_2,column_3]
			select_config, list1 = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=list1,overwrite=False,select_config=select_config)
			flag_select_1, flag_select_2, flag_sample = list1[0:3]
			print('flag_select_1, flag_select_2, flag_sample: ',flag_select_1,flag_select_2,flag_sample)

			run_idvec = [-1,1]
			method_type_vec_query = ['Unify']
			# dict_annot_pre2 = dict(zip(run_idvec,method_type_vec_query))
			dict_annot_pre2 = dict()
			method_type_query_1 = method_type_vec_query[0]
			for run_id_2 in run_idvec:
				dict_annot_pre2.update({run_id_2:method_type_query_1})
			method_type_query = dict_annot_pre2[run_id_2_ori]

			if run_id_2_ori>0:
				column_1 = 'method_type_feature_link_ori'
				if not (column_1 in select_config):
					method_type_feature_link_ori = select_config['method_type_feature_link']
					select_config.update({column_1:method_type_feature_link_ori})
					method_type_feature_link = method_type_query
					select_config.update({'method_type_feature_link':method_type_feature_link})
					# print('method_type_feature_link_ori: ',method_type_feature_link_ori)
					print('method_type_feature_link: ',method_type_feature_link)

			list_annot_peak_tf = []
			iter_vec_query = iter_vec_1

			for i1 in iter_vec_query:
				motif_id_query = motif_idvec_query[i1]
				motif_id1, motif_id2 = motif_id_query, motif_id_query
				folder_id_query = 2

				folder_id = folder_id_query
				config_id_2 = dict_config_annot1[folder_id]
				select_config.update({'config_id_2_query':config_id_2})

				input_file_path_query_1 = dict_file_annot1[folder_id_query] # the first level directory
				input_file_path_query_2 = dict_file_annot2[folder_id_query] # the second level directory including the configurations

				print('motif_id_query, motif_id1, motif_id2: ',motif_id_query,motif_id1,motif_id2,i1)

				if motif_id_query in motif_vec_group2_query2:
					print('the estimation not included: ',motif_id_query,motif_id1,i1)
					continue

				file_path_query_pre1 =  output_file_path_query
				file_path_query_pre2 =  output_file_path_query_2

				overwrite_2 = False
				method_type_query = method_type_feature_link
				filename_prefix_save = 'test_query.%s.%s'%(method_type_query,method_type_group)
				iter_id1 = 0
				filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)

				ratio_1,ratio_2 = select_config['ratio_1'],select_config['ratio_2']
				n_neighbors = select_config['neighbor_num'] # the number of neighbors of a peak with predicted TF binding
				filename_annot_train_pre1 = '%s_%s'%(ratio_1,ratio_2)
				filename_save_annot_query = '%s.%s.%s.neighbor%d'%(method_type_query,method_type_group,filename_annot_train_pre1,n_neighbors)
				filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id_query,filename_save_annot_1)

				run_id2 = run_id_2_ori
				self.run_id2 = run_id2

				run_id_2 = '%s_%d_%d_%d'%(run_id_2_ori,flag_select_1,flag_select_2,flag_sample)
				file_path_query_pre2_2 = '%s/train%s'%(file_path_query_pre2,run_id_2)

				output_file_path_query = file_path_query_pre2_2
				filename_prefix_2 = '%s.%s.%s.%s'%(filename_save_annot_query,motif_id_query,filename_save_annot_1,run_id_2)
				filename_query_pre1 = '%s/test_query_train.%s.1.txt'%(output_file_path_query,filename_prefix_2)

				if (os.path.exists(filename_query_pre1)==False):
					print('the file does not exist: %s'%(filename_query_pre1))
				else:
					print('the file exists: %s'%(filename_query_pre1))
					if (overwrite_2==False):
						continue

				flag1=1
				try:
					# load the TF binding prediction file
					# the possible columns: (signal,motif,predicted binding,motif group)
					# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
					start_1 = time.time()
					filename_prefix_1 = 'test_motif_query_binding_compare'
					input_filename_query1 = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.txt'%(input_file_path_query_2,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)
					
					# flag_group_query_1 = 0
					flag_group_query_1 = 1
					if flag_group_query_1==0:
						# peak_loc_1 = df_pre1.index
						# df_pre1 = df_pre1.loc[peak_loc_ori,:]
						if (os.path.exists(input_filename_query1)==True):
							df_pre1 = pd.read_csv(input_filename_query1,index_col=0,sep='\t')
							df_query_1 = df_pre1
						else:
							print('the file does not exist: %s'%(input_filename_query1))
							flag_group_query_1 = 1
							
					if flag_group_query_1>0:
						load_mode_pre1_1 = 1
						if load_mode_pre1_1>0:
							# load the TF binding prediction file
							# the possible columns: (signal,motif,predicted binding,motif group)
							# folder_id = select_config['folder_id']
							folder_id = folder_id_query
							dict_file_load = select_config['dict_file_load']
							input_filename = dict_file_load[motif_id_query] # the file which saves the previous estimation for each TF;

							if os.path.exists(input_filename)==True:
								df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
								peak_loc_1 = df_1.index
								peak_num_1 = len(peak_loc_1)
								print('peak_loc_1: ',peak_num_1)

								method_type_feature_link_ori = method_type_feature_link
								method_type_vec_pre2 = [method_type_feature_link]
								annot_str_vec = ['motif','pred','score']
								column_vec_query = []
								for method_type_query in method_type_vec_pre2:
									column_vec_query.extend(['%s.%s'%(method_type_query,annot_str1) for annot_str1 in annot_str_vec])

								# df_query1 = df_1
								column_vec = df_1.columns
								column_vec_query_1 = pd.Index(column_vec_query).intersection(column_vec,sort=False)
								
								column_query = '%s.pred'%(method_type_feature_link)
								if not (column_query in column_vec_query_1):
									print('the estimation not included ')
									continue

								df_query_1 = df_1.loc[:,column_vec_query_1]
								print('df_query_1: ',df_query_1.shape,motif_id_query,i1)
								print(df_query_1.columns)
								print(df_query_1[0:2])
								print(input_filename)

					peak_loc_1 = df_query_1.index
					column_vec = df_query_1.columns
					df_query1 = pd.DataFrame(index=peak_loc_ori)
					df_query1.loc[peak_loc_1,column_vec] = df_query_1
					print('df_query1: ',df_query1.shape)

					# column_signal = 'signal'
					# if column_signal in df_query1.columns:
					# 	peak_signal = df_query1[column_signal]
					# 	id_signal = (peak_signal>0)
					# 	# peak_signal_1_ori = peak_signal[id_signal]
					# 	df_query1_signal = df_query1.loc[id_signal,:]	# the peak loci with peak_signal>0
					# 	peak_loc_signal = df_query1_signal.index
					# 	peak_num_signal = len(peak_loc_signal)
					# 	print('signal_num: ',peak_num_signal)

					if not (column_motif in df_query1.columns):
						peak_loc_motif = peak_loc_ori[motif_data[motif_id_query]>0]
						df_query1.loc[peak_loc_motif,column_motif] = motif_data_score.loc[peak_loc_motif,motif_id_query]

					motif_score = df_query1[column_motif]
					print('column_motif: ',column_motif)
					print(df_query1[column_motif])
					query_vec_1 = df_query1[column_motif].unique()
					print('query_vec_1: ',query_vec_1)

					try:
						id_motif = (df_query1[column_motif].abs()>0)
					except Exception as error:
						print('error! ',error)
						id_motif = (df_query1[column_motif].isin(['True',True,1,'1']))

					df_query1_motif = df_query1.loc[id_motif,:]	# the peak loci with binding motif identified
					peak_loc_motif = df_query1_motif.index
					peak_num_motif = len(peak_loc_motif)
					print('peak_loc_motif ',peak_num_motif)
						
					if peak_num_motif==0:
						continue

					flag_motif_query=1
					flag_select_query=1
					if flag_select_1 in [2]:
						flag_motif_query = 0
						flag_select_query = 0
					
					stat_chi2_correction = True
					stat_fisher_alternative = 'greater'
					# filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					method_type_query = method_type_feature_link
					filename_save_annot2_2 = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)

					method_type_query = method_type_feature_link
					column_motif_2 = '%s.motif'%(method_type_query) # motif detection used by the method
					column_pred1 = '%s.pred'%(method_type_query)	# peak-TF association prediction by the method
					column_score_query1 = '%s.score'%(method_type_query)	# estimated peak-TF association score
					select_config.update({'method_type_query':method_type_query})

					if run_id_2_ori in run_idvec:
						# load the file of peak-TF association estimation
						input_file_path_2 = '%s/folder_save_1'%(input_file_path_query_pre2)
						input_filename = '%s/test_query_binding.2.%s.1.txt'%(input_file_path_2,motif_id_query)
						df_query_pre2 = pd.read_csv(input_filename,index_col=0,sep='\t')
						print('df_query_pre2: ',df_query_pre2.shape)
						print(df_query_pre2.columns)
						print(df_query_pre2[0:5])
						print(input_filename)

						peak_loc_pre1 = df_query1.index
						df_query_pre2 = df_query_pre2.loc[peak_loc_pre1,:]

						column_vec_query_1 = [column_motif_2,column_pred1,column_score_query1]
						column_vec = df_query_pre2.columns
						column_vec_query_2 = pd.Index(column_vec_query_1).difference(column_vec,sort=False)
						# the estimation of the motif not included
						if len(column_vec_query_2)>0:
							print('the column not included: ',column_vec_query_2,motif_id_query,i1)
							continue

						df_query1.loc[:,column_vec_query_1] = df_query_pre2.loc[peak_loc_pre1,column_vec_query_1]
						print('df_query1: ',df_query1.shape)

					input_file_path_2 = '%s/folder_group_pre1_%d'%(input_file_path_query_1,run_id_2_ori)
					column_query = 'folder_group_save'
					if column_query in select_config:
						input_file_path_2 = select_config[column_query]

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
						
					id_pred1 = (df_query1[column_pred1]>0)
					peak_loc_pre1 = df_query1.index
					peak_loc_query1 = peak_loc_pre1[id_pred1]
					
					df_pre1 = df_query1
					id_1 = id_pred1
					df_query1_2 = df_query1.loc[id_1,:] # the selected peak loci
					df_pred1 = df_query1_2

					if flag_select_query>0:
						# select the peak loci predicted with TF binding
						# the selected peak loci
						t_vec_2 = self.test_query_feature_overlap_2(data=df_pred1,motif_id_query=motif_id_query,motif_id1=motif_id1,
																		df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,output_file_path=output_file_path_2,filename_prefix_save='',filename_save_annot='',
																		verbose=verbose,select_config=select_config)
						
						df_overlap_query, df_group_basic_query_2, dict_group_basic_query, load_mode_query2 = t_vec_2
						
						# TODO: automatically adjust the group size threshold
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
							filename_save_annot2_query = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)
							output_filename = '%s/test_query_df_overlap.%s.pre1.1.txt'%(output_file_path_2,filename_save_annot2_query)
							df_overlap_query.to_csv(output_filename,sep='\t')

					flag_neighbor_query_1 = 0
					if flag_group_query_1>0:
						flag_neighbor_query_1 = 1

					if flag_select_1 in [2]:
						flag_neighbor_query_1 = 0

					flag_neighbor = 1
					flag_neighbor_2 = 1  # query neighbor of selected peak in the paired groups

					feature_type_vec = feature_type_vec_query
					print('feature_type_vec: ',feature_type_vec)
					select_config.update({'feature_type_vec':feature_type_vec})
					self.feature_type_vec = feature_type_vec

					# filename_save_annot2_2 = '%s.%s.%s'%(method_type_group,motif_id1,data_file_type_query)
					filename_save_annot2_2 = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)
					if flag_neighbor_query_1>0:
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

					load_mode = 0
					if load_mode>0:
						if not (column_score_query1 in df_query1.columns):
							id1 = (df_score_annot[column_id3]==motif_id_query)
							df_score_annot_query = df_score_annot.loc[id1,:]
							peak_loc_2 = df_score_annot_query[column_id2].unique()
							df_query1.loc[peak_loc_2,column_score_query1] = df_score_annot_query.loc[peak_loc_2,column_score_1]

					flag_query_2=1
					method_type_feature_link = select_config['method_type_feature_link']
					if flag_query_2>0:
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
						# flag_sample = select_config['flag_sample']

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
							# select training sample
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
							flag_score_query_1 = 0
							flag_enrichment_sel = 1
							# thresh_vec_sel_1 = [0.5,0.9]
							thresh_vec_sel_1 = [0.25,0.75]
							select_config.update({'thresh_vec_sel_1':thresh_vec_sel_1})
							peak_loc_query_group2_1 = self.test_query_training_select_pre1(data=df_pre2,column_vec_query=[],
																								flag_corr_1=flag_corr_1,flag_score_1=flag_score_query_1,
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

						if flag_select_2==1:
							# select training sample in class 2
							print('feature_type_vec_query: ',feature_type_vec_query)
							peak_vec_2_1, peak_vec_2_2 = self.test_query_training_select_group2(data=df_pre1,motif_id_query=motif_id_query,
																									peak_query_vec_1=peak_query_vec,
																									feature_type_vec=feature_type_vec_query,
																									save_mode=save_mode,verbose=verbose,select_config=select_config)

							peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)

						elif flag_select_2 in [2,3]:
							peak_vec_2, peak_vec_2_1, peak_vec_2_2 = self.test_query_training_select_group2_2(data=df_pre1,id_query=id_pred1,method_type_query=method_type_feature_link,
																												flag_sample=flag_sample,flag_select_2=flag_select_2,
																												save_mode=1,verbose=verbose,select_config=select_config)

						if flag_select_1>0:
							df_pre1.loc[peak_query_vec,'class'] = 1

						if flag_select_2 in [1,3]:
							df_pre1.loc[peak_vec_2_1,'class'] = -1
							df_pre1.loc[peak_vec_2_2,'class'] = -2

							peak_num_2_1 = len(peak_vec_2_1)
							peak_num_2_2 = len(peak_vec_2_2)
							print('peak_vec_2_1, peak_vec_2_2: ',peak_num_2_1,peak_num_2_2)
							peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)

						elif flag_select_2 in [2]:
							df_pre1.loc[peak_vec_2,'class'] = -1

						peak_num_2 = len(peak_vec_2)
						print('peak_vec_2: ',peak_num_2)

						peak_query_num_1 = len(peak_query_vec)
						print('peak_query_vec: ',peak_query_num_1)

						peak_vec_1 = peak_query_vec
						sample_id_train = pd.Index(peak_vec_1).union(peak_vec_2,sort=False)

						df_query_pre1 = df_pre1.loc[sample_id_train,:]
						filename_annot2 = '%s_%s'%(ratio_1,ratio_2)
						filename_annot_train_pre1 = filename_annot2

						flag_scale_1 = select_config['flag_scale_1']
						type_query_scale = flag_scale_1

						iter_id1 = 0
						# run_id_2 = 2
						run_id_2 = '%s_%d_%d_%d'%(run_id_2_ori,flag_select_1,flag_select_2,flag_sample)
						method_type_query = method_type_feature_link

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

						df_pre1[motif_id_query] = 0
						df_pre1.loc[peak_vec_1,motif_id_query] = 1
						# df_pre1.loc[peak_vec_2,motif_id_query] = 0
						peak_num1 = len(peak_vec_1)
						print('peak_vec_1: ',peak_num1)
						print(df_pre1.loc[peak_vec_1,[column_motif,motif_id_query]])

						# print('motif_id_query, motif_id: ',motif_id_query,motif_id1,i1)
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

							# config_id_2: configuration for selecting class 0 sample
							# flag_scale_1: 0, without feature scaling; 1, with feature scaling
							# train_id1 = 1
							train_id1 = select_config['train_id1']
							flag_scale_1 = select_config['flag_scale_1']
							type_query_scale = flag_scale_1

							file_path_query_pre2 = dict_file_annot2[folder_id_query]

							output_file_path_query = '%s/train%s_2'%(file_path_query_pre2,run_id_2)
							output_file_path_query2 = '%s/model_train_1'%(output_file_path_query)
							if os.path.exists(output_file_path_query2)==False:
								print('the directory does not exist: %s'%(output_file_path_query2))
								os.makedirs(output_file_path_query2,exist_ok=True)

							model_path_1 = output_file_path_query2
							select_config.update({'model_path_1':model_path_1})
							select_config.update({'file_path_query_1':file_path_query_pre2})

							filename_prefix_save = 'test_query.%s.%s'%(method_type_query,method_type_group)
							
							iter_id1 = 0
							filename_save_annot_1 = '%s.%d.%d'%(data_file_type_query,iter_id1,config_id_load)
							filename_save_annot_query = '%s.%s.%s.neighbor%d'%(method_type_query,method_type_group,filename_annot_train_pre1,n_neighbors)				
							filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id_query,filename_save_annot_1)
							
							# run_id2 = self.run_id2
							filename_prefix_2 = '%s.%s.%s.%s'%(filename_save_annot_query,motif_id_query,filename_save_annot_1,run_id_2)
							output_filename = '%s/test_query_train.%s.1.txt'%(output_file_path_query,filename_prefix_2)
							
							try:
								df_pre2 = self.test_query_compare_binding_train_unit1(data=df_pre1,peak_query_vec=peak_loc_pre1,peak_vec_1=peak_vec_1,
																						motif_id_query=motif_id_query,
																						dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,
																						sample_idvec_query=sample_idvec_query,
																						motif_data=motif_data_query1,
																						flag_scale=flag_scale_1,input_file_path=input_file_path,
																						save_mode=1,output_file_path=output_file_path_query,output_filename=output_filename,
																						filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,
																						verbose=verbose,select_config=select_config)
							except Exception as error:
								print('error! ',error)
								print('motif query: ',motif_id_query,i1)
								continue

					stop_1 = time.time()
					print('TF binding prediction for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop_1-start_1))
				
				except Exception as error:
					print('error! ',error, motif_id_query,motif_id1,motif_id2,i1)
					# return

			# flag_score_query=flag_score_2
			flag_score_query=0
			if flag_score_query>0:
				type_query = 1
				feature_query_vec = []
				df_score_query = self.test_query_compare_binding_pre1_5_1_basic_1(data=[],feature_query_vec=feature_query_vec,type_query=type_query,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

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

	## query latent feature embeddings and perform peak clustering
	def test_query_feature_clustering_pre1(self,data=[],feature_query_vec=[],feature_type_vec=[],save_mode=1,verbose=0,select_config={}):

		flag_clustering_1=1
		if flag_clustering_1>0:
			data_file_type_query = select_config['data_file_type']
			metric = 'euclidean'
			# n_components = 100
			n_components = select_config['n_components']
			n_component_sel = select_config['n_component_sel']
			feature_mode_query = select_config['feature_mode']
			flag_cluster_1 = 1
			flag_cluster_2 = 0
			flag_combine = 0
			
			column_1 = 'file_path_group_query'
			file_path_group_query = select_config[column_1]
			input_file_path = file_path_group_query
			output_file_path = input_file_path

			# type_id_group = 0
			type_id_group = select_config['type_id_group']
			filename_prefix_save = '%s.pre%d.%d'%(data_file_type_query,feature_mode_query,type_id_group)
			# filename_save_annot = '1'

			feature_type_vec_2 = feature_type_vec
			if len(feature_type_vec)==0:
				feature_type_vec_2 = ['peak_motif','peak_tf']
			
			dict_query_1 = data
			method_type_dimension = select_config['method_type_dimension']
			if len(dict_query_1)==0:
				print('load feature embeddings')
				start = time.time()
				# method_type_dimension = 'SVD'
				# method_type_dimension = select_config['method_type_dimension']
				# n_components = 50
				# type_id_group = select_config['type_id_group']
				# reconstruct = 0
				# load latent matrix;
				# feature_type_vec = ['latent_peak_motif','latent_peak_tf']
				
				feature_type_vec_query2 = feature_type_vec_2
				# recontruct: 1, load reconstructed matrix;
				reconstruct = 0
				# if feature_mode_query==3:
				# 	feature_type_vec_query2 = ['peak_motif','peak_mtx']
				n_components_query = n_components
				# n_component_sel = select_config['n_component_sel']
				filename_save_annot = '%s_%d.1'%(method_type_dimension,n_components_query)
				dict_file = self.dict_file_feature # the filename of the feature embeddings;
				dict_latent_query1 = self.test_query_feature_embedding_load_1(data=[],dict_file=dict_file,feature_query_vec=feature_query_vec,motif_id_query='',motif_id='',
																				feature_type_vec=feature_type_vec_query2,
																				method_type_vec=[],
																				method_type_dimension=method_type_dimension,
																				n_components=n_components,
																				n_component_sel=n_component_sel,
																				reconstruct=reconstruct,
																				peak_read=[],rna_exprs=[],
																				flag_combine=flag_combine,
																				load_mode=0,
																				input_file_path=input_file_path,
																				save_mode=0,output_file_path='',output_filename='',
																				filename_prefix_save=filename_prefix_save,
																				filename_save_annot=filename_save_annot,
																				verbose=verbose,select_config=select_config)

				print('dict_latent_query1: ',dict_latent_query1)

				# if feature_mode_query==3:
				# 	dict_latent_query1['peak_tf'] = dict_latent_query1['peak_mtx']
				stop = time.time()
				print('load feature embeddings used %s.2fs'%(stop-start))

			else:
				dict_latent_query1 = dict_query_1

			print('perform peak clustering')
			start = time.time()
			overwrite = False
			# overwrite_2 = True
			overwrite_2 = False
			# n_component_sel = 10
			# n_component_sel = select_config['n_components_2']
			# print('n_component_sel ',n_component_sel)
			
			flag_iter_2 = 0
			# method_type_vec_group = ['phenograph']
			method_type_group_name = select_config['method_type_group_name']
			method_type_vec_group = [method_type_group_name]
			# select_config.update({'method_type_vec_group':method_type_vec_group,
			# 						'n_component_sel':n_component_sel,
			# 						'flag_iter_2':flag_iter_2})
			
			# perform clustering of peak loci based on the low-dimensional embeddings
			filename_prefix_save_pre2 = filename_prefix_save
			filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_component_sel)
			feature_type_vec_pre1 = ['gene','peak']
			self.test_query_feature_clustering_1(data=dict_latent_query1,dict_feature=dict_latent_query1,feature_type_vec=feature_type_vec_pre1,feature_type_vec_2=feature_type_vec_2,
													method_type_vec_group=method_type_vec_group,feature_mode=feature_mode_query,type_id_group=type_id_group,
													n_components=n_components,metric=metric,
													subsample_ratio=-1,subsample_ratio_vec=[],
													peak_read=[],rna_exprs=[],
													flag_cluster_1=flag_cluster_1,flag_cluster_2=flag_cluster_2,flag_combine=flag_combine,
													overwrite=overwrite,overwrite_2=overwrite_2,
													input_file_path=input_file_path,save_mode=1,output_file_path=output_file_path,
													filename_prefix_save=filename_prefix_save_pre2,filename_save_annot=filename_save_annot_2,verbose=0,select_config=select_config)

			stop = time.time()
			print('performing peak clustering used %.2fs'%(stop-start))

			return dict_latent_query1

	## load gene annotation data
	def test_query_gene_annot_1(self,data=[],input_filename='',save_mode=1,verbose=0,select_config={}):

		if input_filename=='':
			input_filename_annot = select_config['filename_gene_annot']
		else:
			input_filename_annot = input_filename

		df_gene_annot_ori = pd.read_csv(input_filename_annot,index_col=False,sep='\t')
		df_gene_annot_ori = df_gene_annot_ori.sort_values(by=['length'],ascending=False)
		df_gene_annot_ori = df_gene_annot_ori.drop_duplicates(subset=['gene_name'])
		df_gene_annot_ori = df_gene_annot_ori.drop_duplicates(subset=['gene_id'])
		df_gene_annot_ori.index = np.asarray(df_gene_annot_ori['gene_name'])
		print('gene annotation ',df_gene_annot_ori.shape)
		print(df_gene_annot_ori.columns)
		print(df_gene_annot_ori[0:2])

		return df_gene_annot_ori

	## query motif data filename
	def test_query_motif_data_filename_1(self,data=[],input_file_path='',save_mode=1,verbose=0,select_config={}):

		if input_file_path=='':
			input_file_path = select_config['input_dir']

		# filename_motif_data = '%s/test_peak_read.pbmc.0.1.normalize.1_motif.1.2.csv'%(input_file_path_pre1)
		# filename_motif_data_score = '%s/test_peak_read.pbmc.0.1.normalize.1_motif_scores.1.csv'%(input_file_path_pre1)
		filename_motif_data = '%s/test_peak_read.pbmc.normalize.motif.thresh5e-05.csv'%(input_file_path)
		filename_motif_data_score = '%s/test_peak_read.pbmc.normalize.motif_scores.thresh5e-05.csv'%(input_file_path)
		filename_translation = '%s/translationTable.csv'%(input_file_path_pre1)

		field_query_1 = ['filename_motif_data','filename_motif_data_score','filename_translation']
		list1 = [filename_motif_data,filename_motif_data_score,filename_translation]
		for (field_id,query_value) in zip(field_query_1,list1):
			if field_id in select_config:
				query_value_1 = select_config[field_id]
				print('field_id, query_value_1 ',field_id,query_value_1)
			else:
				print('field_id, query_value ',field_id,query_value)
				select_config.update({field_id:query_value})

		return select_config

	## recompute based on clustering of peak and TF
	# recompute based on training
	def test_query_compare_binding_compute_2(self,data=[],dict_feature=[],feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		# run_id1 = select_config['run_id']
		print('data_file_type_query: ',data_file_type_query)
		# print('run_id: ',run_id1)
		
		if 'method_type_feature_link' in select_config:
			method_type_feature_link = select_config['method_type_feature_link']
			# method_type_feature_link_1 = method_type_feature_link.split('.')[0]
		
		filename_save_annot = '1'
		# the methods used
		method_type_vec = pd.Index(method_type_vec).union([method_type_feature_link],sort=False)

		flag_config_1 = 1
		flag_gene_annot_1 = 1
		flag_motif_data_load_1 = 1
		flag_load_1 = 1
		method_type_feature_link_query1 = 'Unify'
		method_type_vec_query1 = [method_type_feature_link_query1]

		root_path_1 = select_config['root_path_1']
		data_path_save_1 = root_path_1

		input_dir = select_config['input_dir']
		output_dir = select_config['output_dir']
		input_file_path_pre1 = input_dir
		output_file_path_pre1 = output_dir
		print('input_file_path_pre1: ',input_file_path_pre1)
		print('output_file_path_pre1: ',output_file_path_pre1)

		file_path_motif = input_file_path_pre1
		select_config.update({'file_path_motif':file_path_motif})

		# load gene annotation data
		if flag_gene_annot_1>0:
			print('load gene annotations')
			filename_gene_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path_pre1)
			select_config.update({'filename_gene_annot':filename_gene_annot})
			df_gene_annot_ori = self.test_query_gene_annot_1(filename_gene_annot,select_config=select_config)
			self.df_gene_annot_ori = df_gene_annot_ori

		# load motif data
		# load ATAC-seq and RNA-seq data of the metacells
		flag_load_pre1 = (flag_load_1>0)|(flag_motif_data_load_1>0)
		if (flag_load_pre1>0):
			# select_config = self.test_query_motif_data_filename_1(input_file_path=input_file_path_pre1,save_mode=1,verbose=verbose,select_config=select_config)

			select_config = self.test_query_load_pre1(data=[],method_type_vec_query=method_type_vec_query1,flag_config_1=flag_config_1,
														flag_motif_data_load_1=flag_motif_data_load_1,
														flag_load_1=flag_load_1,
														save_mode=1,verbose=verbose,select_config=select_config)

			# return

		dict_query_1 = dict()
		# feature_type_vec = ['latent_peak_motif','latent_peak_motif_ori','latent_peak_tf','latent_peak_tf_link']
		feature_type_vec_group = ['latent_peak_tf','latent_peak_motif','latent_peak_motif_ori','latent_peak_tf_link']
		# type_id_group_2 = select_config['type_id_group_2']
		type_id_group_2 = 1
		
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

		print('feature_type_vec_query: ',feature_type_vec_query)
		print('feature_type_vec_2_ori: ',feature_type_vec_2_ori)

		# flag_annot_1 = 1
		method_type_group = select_config['method_type_group']
		t_vec_1 = method_type_group.split('.')
		method_type_group_name = t_vec_1[0]
		n_neighbors_query = int(t_vec_1[1])
		print('method_type_group: ',method_type_group)
		print('peak clustering method: %s'%(method_type_group_name))
		print('using number of neighbors: %d'%(n_neighbors_query))

		method_type_vec_group_ori = [method_type_group_name]
		method_type_group_neighbor = n_neighbors_query
		select_config.update({'method_type_group_name':method_type_group_name,
								'method_type_group_neighbor':method_type_group_neighbor})

		# flag_clustering_1 = 0
		# flag_clustering_1 = 1
		output_dir = select_config['output_dir']
		file_save_path_1 = output_dir

		feature_mode = 1  # with RNA-seq and ATAC-seq data
		feature_mode_query = feature_mode
		select_config.update({'feature_mode':feature_mode})

		file_path_group_query = '%s/group%d'%(file_save_path_1,feature_mode_query)
		if os.path.exists(file_path_group_query)==False:
			print('the directory does not exist: %s'%(file_path_group_query))
			os.makedirs(file_path_group_query,exist_ok=True)

		select_config.update({'file_path_group_query':file_path_group_query})

		# n_component_sel = select_config['n_component_sel']
		# print('n_component_sel ',n_component_sel)

		flag_iter_2 = 0
		# method_type_vec_group = ['phenograph']
		method_type_vec_group = method_type_vec_group_ori
		select_config.update({'method_type_vec_group':method_type_vec_group,
								'flag_iter_2':flag_iter_2})

		# flag_embedding_compute=1
		flag_embedding_compute=0
		flag_clustering_1=0
		flag_group_load_1=1
		column_1 = 'flag_embedding_compute'
		if column_1 in select_config:
			flag_embedding_compute = select_config[column_1]

		field_query = ['flag_embedding_compute','flag_clustering','flag_group_load']
		default_parameter_vec = [flag_embedding_compute,flag_clustering_1,flag_group_load_1]
		list1 = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=default_parameter_vec,overwrite=False,select_config=select_config)
		flag_embedding_compute, flag_clustering_1, flag_group_load_1 = list1[0:3]

		dict_query_1 = dict()
		dict_latent_query1 = dict()
		peak_read = self.peak_read
		peak_loc_ori = peak_read.columns
		rna_exprs = self.rna_exprs
		print('peak_read: ',peak_read.shape)
		print('rna_exprs: ',rna_exprs.shape)

		n_components = 100
		column_1 = 'n_components'
		
		method_type_dimension = 'SVD'
		column_2 = 'method_type_dimension'

		field_query_1 = [column_1,column_2]
		param_vec_1 = [n_components,method_type_dimension]
		select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=field_query_1,default_parameter=param_vec_1,overwrite=False,select_config=select_config)

		n_components = select_config[column_1]
		n_component_sel = select_config['n_component_sel']
		method_type_dimension = select_config[column_2]
		print('method_type_dimension ',method_type_dimension)
		print('n_components ',n_components)
		print('n_component_sel ',n_component_sel)

		# compute feature embeddings
		if flag_embedding_compute>0:
			print('compute feature embeddings')
			start = time.time()
			
			method_type_query = method_type_feature_link
			type_combine = 0
			select_config.update({'type_combine':type_combine})
			
			feature_mode_vec = [1]

			input_file_path = input_file_path_pre1
			output_file_path = file_path_group_query

			# column_3 = 'flag_peak_tf_combine'
			flag_peak_tf_combine = 0
			select_config.update({'flag_peak_tf_combine':flag_peak_tf_combine})

			for feature_mode_query in feature_mode_vec:
				select_config.update({'feature_mode':feature_mode_query})
				dict_query_1, select_config = self.test_query_feature_embedding_pre1(data=[],dict_feature=[],feature_type_vec=[],
																						method_type=method_type_query,
																						field_query=[],peak_read=[],rna_exprs=[],
																						n_components=n_components,
																						iter_id=-1,config_id_load=-1,
																						flag_config=1,flag_motif_data_load=0,
																						flag_load_1=0,input_file_path=input_file_path,overwrite=False,
																						save_mode=1,output_file_path=output_file_path,output_filename='',filename_prefix_save='',filename_save_annot='',
																						verbose=verbose,select_config=select_config)
			stop = time.time()
			print('computing feature embeddings used %.2fs'%(stop-start))

		# flag_clustering_1=1
		# flag_clustering_1=0
		dict_file_feature = dict()
		feature_type_vec_2 = feature_type_vec_2_ori
		feature_type1, feature_type2 = feature_type_vec_2[0:2]
		input_file_path_query = file_path_group_query

		type_id_group = select_config['type_id_group']
		n_components_query = n_components
		filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
		filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_components_query)

		for feature_type_query in feature_type_vec_2:
			input_filename_1 = '%s/%s.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_save_2,feature_type_query,filename_save_annot_2)
			input_filename_2 = '%s/%s.df_component.%s.%s.txt'%(input_file_path_query,filename_prefix_save_2,feature_type_query,filename_save_annot_2)
			dict1 = {'df_latent':input_filename_1,'df_component':input_filename_2}
			dict_file_feature.update({feature_type_query:dict1})

		self.dict_file_feature = dict_file_feature
		print('dict_file_feature ',dict_file_feature)

		if flag_clustering_1>0:
			# feature_type_vec = feature_type_vec_query
			dict_latent_query1 = self.test_query_feature_clustering_pre1(data=dict_query_1,feature_query_vec=peak_loc_ori,feature_type_vec=feature_type_vec_2,save_mode=1,verbose=verbose,select_config=select_config)

			# return

		flag_annot_1 = 1
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

		# flag_group_load_1 = 1
	
		input_file_path_2 = file_path_group_query
		output_file_path_2 = file_path_group_query
		if flag_group_load_1>0:
			# load feature group estimation for peak or peak and TF
			n_clusters = 50
			peak_loc_ori = peak_read.columns
			feature_query_vec = peak_loc_ori
			# the overlap matrix; the group member number and frequency of each group; the group assignment for the two feature types
			filename_prefix_save_query2 = '%s.peak.group'%(filename_prefix_save_2)

			type_id_compute, type_id_feature = 0, 0
			column_2, column_3 = 'type_id_compute', 'type_id_feature'
			if column_2 in select_config:
				type_id_compute = select_config[column_2]

			if column_3 in select_config:
				type_id_feature = select_config[column_3]

			filename_save_annot2_ori = '%s.%d_%d.%d'%(method_type_dimension,n_component_sel,type_id_compute,type_id_feature)
			df_overlap_compare, dict_group_basic_1, df_group_1, df_group_2 = self.test_query_feature_group_load_1(data=[],feature_type_vec=feature_type_vec_query,
																													feature_query_vec=feature_query_vec,
																													method_type_group=method_type_group,
																													input_file_path=input_file_path_2,
																													save_mode=1,output_file_path=output_file_path_2,filename_prefix_save=filename_prefix_save_query2,filename_save_annot=filename_save_annot2_ori,output_filename='',verbose=0,select_config=select_config)

			self.df_group_pre1 = df_group_1
			self.df_group_pre2 = df_group_2
			self.dict_group_basic_1 = dict_group_basic_1
			self.df_overlap_compare = df_overlap_compare

		flag_query2 = 1
		if flag_query2>0:
			# select the feature type for group query
			# feature_type_vec_group = ['latent_peak_tf','latent_peak_motif']

			flag_load_2 = 1
			if flag_load_2>0:
				feature_type_1, feature_type_2 = feature_type_vec_2_ori[0:2]
			
				# method_type_dimension = 'SVD'
				# method_type_dimension = select_config['method_type_dimension']
				# n_components = 50
				n_components = select_config['n_components']
				n_component_sel = select_config['n_component_sel']
				type_id_group = select_config['type_id_group']
				filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
				filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_components_query)
				
				reconstruct = 0
				# load latent matrix;
				# reconstruct: 1, load reconstructed matrix;
				flag_combine = 1
				feature_type_vec_2 = feature_type_vec_2_ori
				select_config.update({'feature_type_vec_2':feature_type_vec_2})
				print('feature_type_1, feature_type_2: ',feature_type_1,feature_type_2)
				dict_latent_query1 = self.test_query_feature_embedding_load_1(data=[],dict_file=dict_file_feature,feature_query_vec=peak_loc_ori,feature_type_vec=feature_type_vec_2,method_type_vec=[],method_type_dimension=method_type_dimension,
																				n_components=n_components,n_component_sel=n_component_sel,reconstruct=reconstruct,peak_read=[],rna_exprs=[],flag_combine=flag_combine,
																				load_mode=0,input_file_path='',
																				save_mode=0,output_file_path='',output_filename='',filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot_2,
																				verbose=0,select_config=select_config)

				dict_feature = dict_latent_query1

			n_neighbors = 100
			if 'neighbor_num' in select_config:
				n_neighbors = select_config['neighbor_num']
			n_neighbors_query = n_neighbors+1

			# query the neighbors of feature query
			flag_neighbor_query=1
			if flag_neighbor_query>0:
				# list_query1 = [[feature_nbrs_1,dist_nbrs_1],[feature_nbrs_2,dist_nbrs_2]]
				list_query1 = self.test_query_feature_neighbor_load_1(dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,n_neighbors=n_neighbors,input_file_path=input_file_path_2,
																		save_mode=save_mode,output_file_path=output_file_path_2,verbose=verbose,select_config=select_config)

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
					
			dict_motif_data = self.dict_motif_data

			method_type_query = method_type_feature_link_query1
			print(method_type_query)
			motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']
			
			motif_data_query1 = motif_data_query1.loc[peak_loc_ori,:]
			motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_ori,:]
			print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			print(motif_data_query1[0:2])
			print(motif_data_score_query1[0:2])
			motif_data = motif_data_query1
			motif_data_score = motif_data_score_query1
			# self.dict_motif_data = dict_motif_data

			motif_name_ori = motif_data_query1.columns
			gene_name_expr_ori = rna_exprs.columns
			motif_query_name_expr = pd.Index(motif_name_ori).intersection(gene_name_expr_ori,sort=False)
			print('motif_query_name_expr: ',len(motif_query_name_expr))

			type_motif_query=0
			if 'type_motif_query' in select_config:
				type_motif_query = select_config['type_motif_query']

			tf_name = select_config['tf_name']
			t_vec_1 = np.asarray(tf_name.split(','))
			motif_idvec_query = t_vec_1

			motif_query_num = len(motif_idvec_query)
			# print('motif_idvec_query: ',motif_query_num,type_motif_query)
			print('motif_idvec_query: ',motif_query_num)
			print(motif_idvec_query)
			query_num_ori = motif_query_num

			columns_1 = select_config['columns_1']
			t_vec_2 = columns_1.split(',')
			column_pred1, column_score_1 = t_vec_2[0:2]
			column_score_query = column_score_1
			column_score_query1 = column_score_query
			column_motif = '-1'
			if len(t_vec_2)>2:
				column_motif = t_vec_2[2]
			column_vec_query = [column_pred1,column_score_1,column_motif]
			print('column_vec_query: ',column_vec_query)
			field_query = ['column_pred1','column_score_query1','column_motif']
			for (field_id,query_value) in zip(field_query,column_vec_query):
				select_config.update({field_id:query_value})

			column_vec_link = column_vec_query
			select_config.update({'column_vec_link':column_vec_link})

			df_score_annot = []
			column_idvec = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_idvec[0:3]

			# load_mode = 1
			load_mode = 0
			flag_sort = 1
			flag_unduplicate_query = 1
			ascending = False
			
			motif_query_num = len(motif_idvec_query)
			query_num_1 = motif_query_num
			# query_num_1 = query_num_ori
			list_score_query_1 = []
			interval_save = True
			config_id_load = select_config['config_id_load']
			# config_id_2 = select_config['config_id_2']
			# config_group_annot = select_config['config_group_annot']
			flag_scale_1 = select_config['flag_scale_1']
			type_query_scale = flag_scale_1

			model_type_id1 = 'LogisticRegression'
			# select_config.update({'model_type_id1':model_type_id1})
			if 'model_type_id1' in select_config:
				model_type_id1 = select_config['model_type_id1']

			beta_mode = select_config['beta_mode']
			
			# method_type_feature_link = select_config['method_type_feature_link']
			method_type_group = select_config['method_type_group']
			n_neighbors = select_config['neighbor_num']

			query_id1,query_id2 = select_config['query_id_1'],select_config['query_id_2']
			iter_mode = 0
			query_num_1 = motif_query_num
			iter_vec_1 = np.arange(query_num_1)
			print('query_id1, query_id2: ',query_id1,query_id2)

			if (query_id1>=0) and (query_id1<query_num_1) and (query_id2>query_id1) :
				iter_mode = 1
				start_id1 = query_id1
				start_id2 = np.min([query_id2,query_num_1])
				iter_vec_1 = np.arange(start_id1,start_id2)
				interval_save = False
			else:
				print('query_id1, query_id2: ',query_id1,query_id2)
				# return

			# run_id_2_ori = 1
			flag_select_1 = 1  # select pseudo positive training sample;
			flag_select_2 = 1  # select pseudo negative training sample;
			flag_sample = 1    # select pseudo training sample

			column_1 = 'flag_select_1'
			column_2 = 'flag_select_2'
			column_3 = 'flag_sample'

			list1 = [flag_select_1,flag_select_2,flag_sample]
			field_query = [column_1,column_2,column_3]
			select_config, list1 = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=list1,overwrite=False,select_config=select_config)
			flag_select_1, flag_select_2, flag_sample = list1[0:3]
			print('flag_select_1, flag_select_2, flag_sample: ',flag_select_1,flag_select_2,flag_sample)

			run_idvec = [-1,1]
			method_type_vec_query = ['Unify']
			# dict_annot_pre2 = dict(zip(run_idvec,method_type_vec_query))
			dict_annot_pre2 = dict()
			method_type_query_1 = method_type_vec_query[0]
			for run_id_2 in run_idvec:
				dict_annot_pre2.update({run_id_2:method_type_query_1})
			run_id_2_ori = 1
			run_id2 = run_id_2_ori
			self.run_id2 = run_id2
			method_type_query = dict_annot_pre2[run_id_2_ori]

			list_annot_peak_tf = []
			iter_vec_query = iter_vec_1
			file_path_query_pre1 = output_file_path_pre1 # the first output directory
			
			output_file_path_pre2 = '%s/file_link'%(output_file_path_pre1) # the second output directory
			if os.path.exists(output_file_path_pre2)==False:
				print('the directory does not exist: %s'%(output_file_path_pre2))
				os.makedirs(output_file_path_pre2,exist_ok=True)

			file_path_query_pre2 = output_file_path_pre2
			input_file_path_query_1 = file_path_query_pre1
			select_config.update({'file_path_save_link':file_path_query_pre2})

			# prepare the folder to save the peak-TF correlation
			column_query = 'folder_correlation'
			if not (column_query in select_config):
				input_file_path_query1 = '%s/folder_correlation'%(input_file_path_query_1)
			else:
				input_file_path_query1 = select_config[column_query]

			if os.path.exists(input_file_path_query1)==False:
				print('the directory does not exist: %s'%(input_file_path_query1))
				os.makedirs(input_file_path_query1,exist_ok=True)
			select_config.update({column_query:input_file_path_query1})

			# prepare the folder to save the peak enrichment in the paired groups
			column_query2 = 'folder_group_save'
			if not (column_query2 in select_config):
				input_file_path_query2 = '%s/folder_group_save'%(input_file_path_query_1)
			else:
				input_file_path_query2 = select_config[column_query2]

			if os.path.exists(input_file_path_query2)==False:
				print('the directory does not exist: %s'%(input_file_path_query2))
				os.makedirs(input_file_path_query2,exist_ok=True)
			select_config.update({column_query2:input_file_path_query2})

			# prepare the folder to save the trained models
			# output_file_path_query = '%s/train%s_2'%(file_path_query_pre2,run_id_2)
			output_file_path_query2 = '%s/model_train_1'%(file_path_query_pre2)
			if os.path.exists(output_file_path_query2)==False:
				print('the directory does not exist: %s'%(output_file_path_query2))
				os.makedirs(output_file_path_query2,exist_ok=True)

			model_path_1 = output_file_path_query2
			select_config.update({'model_path_1':model_path_1,
									'file_path_query_1':file_path_query_pre2})

			# the threshold on peak-TF association score for pseudo training sample selection
			thresh_score = select_config['thresh_score']
			t_vec_1 = thresh_score.split(',')
			thresh_vec_sel_1 = [float(query_value) for query_value in t_vec_1]
			# thresh_vec_sel_1 = [0.25,0.75]
			print('thresh_vec_sel_1: ',thresh_vec_sel_1)
			select_config.update({'thresh_vec_sel_1':thresh_vec_sel_1})

			column_query = 'output_filename_link'
			filename_save_link = select_config[column_query]
			filename_link_prefix = select_config['filename_prefix']
			filename_link_annot = select_config['filename_annot']

			method_type_query = method_type_feature_link
			select_config.update({'method_type_query':method_type_query})

			config_id_2 = 10
			config_group_annot = 0
			select_config.update({'config_id_2':config_id_2,'config_group_annot':config_group_annot})

			file_path_link = select_config['input_link']
			dict_file_load = dict()
			for i1 in iter_vec_query:
				motif_id_query = motif_idvec_query[i1]
				filename_link_query1 = '%s/%s.%s.%s.txt'%(file_path_link,filename_link_prefix,motif_id_query,filename_link_annot)
				dict_file_load.update({motif_id_query:filename_link_query1})
			select_config.update({'dict_file_load':dict_file_load})
			overwrite_2 = False

			for i1 in iter_vec_query:
				motif_id_query = motif_idvec_query[i1]
				motif_id1, motif_id2 = motif_id_query, motif_id_query
				
				filename_query_pre1 = '%s/%s.%s.%s.pred2.txt'%(output_file_path_pre2,filename_link_prefix,motif_id_query,filename_link_annot)
				filename_save_link_pre1 = filename_query_pre1
				
				if (os.path.exists(filename_query_pre1)==False):
					print('the file does not exist: %s'%(filename_query_pre1))
				else:
					print('the file exists: %s'%(filename_query_pre1))
					if (overwrite_2==False):
						continue

				flag1=1
				if flag1>0:
				# try:
					# load the TF binding prediction file
					# the possible columns: (signal,motif,predicted binding,motif group)
					# filename_save_annot2 = '%s.%d'%(filename_save_annot_2,thresh_size_1)
					start_1 = time.time()
					# filename_prefix_1 = 'test_motif_query_binding_compare'
					
					flag_group_query_1 = 1
					load_mode_pre1_1 = 1
					if load_mode_pre1_1>0:
						dict_file_load = select_config['dict_file_load']
						input_filename = dict_file_load[motif_id_query] # the file which saves the previous estimation for each TF;

						if os.path.exists(input_filename)==True:
							df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
							peak_loc_1 = df_1.index
							peak_num_1 = len(peak_loc_1)
							print('peak_loc_1: ',peak_num_1)
							print(input_filename)
						else:
							print('the file does not exist: %s'%(input_filename))
							continue

						column_vec = df_1.columns
						column_vec_query = select_config['column_vec_link']
						column_vec_query_1 = pd.Index(column_vec_query).intersection(column_vec,sort=False)
						if not (column_pred1 in column_vec_query_1):
							print('the estimation not included')
							continue

						column_vec_query_2 = pd.Index(column_vec_query).difference(column_vec,sort=False)
						# the estimation of the motif not included
						if len(column_vec_query_2)>0:
							print('the column not included: ',column_vec_query_2,motif_id_query,i1)
							# continue

						df_query_1 = df_1.loc[:,column_vec_query_1]
						print('df_query_1: ',df_query_1.shape,motif_id_query,i1)
						print(df_query_1.columns)
						print(df_query_1[0:2])
						print(input_filename)

					peak_loc_1 = df_query_1.index
					column_vec = df_query_1.columns
					df_query1 = pd.DataFrame(index=peak_loc_ori)
					df_query1.loc[peak_loc_1,column_vec] = df_query_1
					print('df_query1: ',df_query1.shape)

					if not (column_motif in df_query1.columns):
						print('query the motif score ',column_motif,motif_id_query,i1)
						peak_loc_motif = peak_loc_ori[motif_data[motif_id_query]>0]
						df_query1.loc[peak_loc_motif,column_motif] = motif_data_score.loc[peak_loc_motif,motif_id_query]

					motif_score = df_query1[column_motif]
					print('column_motif: ',column_motif)
					print(df_query1[column_motif])
					query_vec_1 = df_query1[column_motif].unique()
					print('query_vec_1: ',query_vec_1)

					try:
						id_motif = (df_query1[column_motif].abs()>0)
					except Exception as error:
						print('error! ',error)
						id_motif = (df_query1[column_motif].isin(['True',True,1,'1']))

					df_query1_motif = df_query1.loc[id_motif,:]	# the peak loci with binding motif identified
					peak_loc_motif = df_query1_motif.index
					peak_num_motif = len(peak_loc_motif)
					print('peak_loc_motif ',peak_num_motif)
						
					if peak_num_motif==0:
						continue

					flag_motif_query=1
					flag_select_query=1
				
					# input_file_path_2 = '%s/folder_group_save'%(output_file_path_pre1)
					column_query = 'folder_group_save'
					input_file_path_2 = select_config[column_query]
					output_file_path_2 = input_file_path_2

					# compute the enrichment of peak loci with TF motif in paired groups
					filename_prefix_1 = 'test_query_df_overlap'
					method_type_query = method_type_feature_link
					if flag_motif_query>0:
						df_query1_motif = df_query1.loc[id_motif,:] # peak loci with motif
						peak_loc_motif = df_query1_motif.index
						peak_num_motif = len(peak_loc_motif)
						print('peak_loc_motif ',peak_num_motif)

						filename_save_annot2_1 = '%s.%s.%s'%(method_type_group,motif_id_query,data_file_type_query)
						for query_id1 in [1,2]:
							column_1 = 'filename_overlap_motif_%d'%(query_id1)
							filename_query = '%s/%s.%s.motif.%d.txt' % (input_file_path_2,filename_prefix_1,filename_save_annot2_1,query_id1)
							select_config.update({column_1:filename_query})

						# feature overlap query
						t_vec_1 = self.test_query_feature_overlap_1(data=df_query1_motif,motif_id_query=motif_id_query,motif_id1=motif_id1,
																		column_motif=column_motif,df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,output_file_path=output_file_path_2,filename_prefix_save='',filename_save_annot='',
																		verbose=verbose,select_config=select_config)
						
						df_overlap_motif, df_group_basic_motif, dict_group_basic_motif, load_mode_query1 = t_vec_1
						
					id_pred1 = (df_query1[column_pred1]>0)
					peak_loc_pre1 = df_query1.index
					peak_loc_query1 = peak_loc_pre1[id_pred1]
					print('peak_loc_pre1, peak_loc_query1 ',len(peak_loc_pre1),len(peak_loc_query1))
					
					df_pre1 = df_query1
					id_1 = id_pred1
					df_pred1 = df_query1.loc[id_1,:] # the selected peak loci

					if flag_select_query>0:
						# select the peak loci predicted with TF binding
						# query enrichment of peak loci in paired groups
						filename_save_annot2_2 = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)
						for query_id1 in [1,2]:
							column_1 = 'filename_overlap_%d'%(query_id1)
							filename_query = '%s/%s.%s.%d.txt' % (input_file_path_2,filename_prefix_1,filename_save_annot2_2,query_id1)
							select_config.update({column_1:filename_query})

						t_vec_2 = self.test_query_feature_overlap_2(data=df_pred1,motif_id_query=motif_id_query,motif_id1=motif_id1,
																		df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,output_file_path=output_file_path_2,filename_prefix_save='',filename_save_annot='',
																		verbose=verbose,select_config=select_config)
						
						df_overlap_query, df_group_basic_query_2, dict_group_basic_query, load_mode_query2 = t_vec_2
						
						# TODO: automatically adjust the group size threshold
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

						type_group_select = 0
						if type_group_select==0:
							id_pre1 = id1
						else:
							id_pre1 = id2

						df_overlap_query2 = df_overlap_query.loc[id_pre1,:]

						df_overlap_query.loc[id_pre1,'label_1'] = 1
						group_vec_query_1 = np.asarray(df_overlap_query2.loc[:,['group1','group2']])
						print('df_overlap_query, df_overlap_query2: ',df_overlap_query.shape,df_overlap_query2.shape,motif_id_query,i1)

						self.df_overlap_query = df_overlap_query
						self.df_overlap_query2 = df_overlap_query2
						
						load_mode_2 = load_mode_query2
						if load_mode_2<2:
							# filename_save_annot2_query = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)
							# output_filename = '%s/test_query_df_overlap.%s.1.txt'%(output_file_path_2,filename_save_annot2_query)
							output_filename = select_config['filename_overlap_1']
							df_overlap_query.to_csv(output_filename,sep='\t')

					flag_neighbor_query_1 = 0
					if flag_group_query_1>0:
						flag_neighbor_query_1 = 1

					# if flag_select_1 in [2]:
					# 	flag_neighbor_query_1 = 0

					flag_neighbor = 1
					flag_neighbor_2 = 1  # query neighbor of selected peak in the paired groups

					feature_type_vec = feature_type_vec_query
					print('feature_type_vec: ',feature_type_vec)
					select_config.update({'feature_type_vec':feature_type_vec})
					self.feature_type_vec = feature_type_vec

					filename_save_annot2_2 = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)
					if flag_neighbor_query_1>0:
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
						print('query feature group and neighbor annotation for TF %s used %.2fs'%(motif_id_query,motif_id2,stop-start))

					column_vec_query1 = column_vec_link

					if flag_neighbor_query_1>0:
						column_vec_query1_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]
						column_vec_query2 = column_vec_query1 + column_vec_query1_2
					else:
						column_vec_query2 = column_vec_query1

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
						column_signal = 'signal'
						column_query1 = 'folder_correlation'
						input_file_path_query1 = select_config[column_query1]
						output_file_path_query1 = input_file_path_query1
						filename_prefix_save_query = 'test_peak_tf_correlation.%s.%s'%(motif_id_query,data_file_type_query)
						df_query1, df_annot_peak_tf = self.test_query_compare_peak_tf_corr_1(data=df_pre1,motif_id_query=motif_id_query,motif_id1=motif_id1,motif_id2=motif_id2,
																								column_signal=column_signal,column_value=column_value,thresh_value=thresh_value,
																								motif_data=motif_data,motif_data_score=motif_data_score,
																								peak_read=peak_read,rna_exprs=rna_exprs,
																								flag_query=0,input_file_path=input_file_path_query1,
																								save_mode=1,output_file_path=output_file_path_query1,
																								filename_prefix_save=filename_prefix_save_query,filename_save_annot='',output_filename='',
																								verbose=verbose,select_config=select_config)

					flag_query_2=1
					method_type_feature_link = select_config['method_type_feature_link']
					if flag_query_2>0:
						# column_corr_1 = field_id1
						# column_pval = field_id2
						column_corr_1 = 'peak_tf_corr'
						column_pval = 'peak_tf_pval_corrected'
						thresh_corr_1, thresh_pval_1 = 0.30, 0.05
						thresh_corr_2, thresh_pval_2 = 0.1, 0.1
						thresh_corr_3, thresh_pval_2 = 0.05, 0.1
						
						peak_loc_pre1 = df_query1.index

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

							# output_file_path_query = file_path_query2
							file_path_save_group = select_config['folder_group_save']
							output_file_path_query = file_path_save_group
							# select training sample

							column_1 = 'thresh_overlap_default_1'
							column_2 = 'thresh_overlap_default_2'
							column_3 = 'thresh_overlap'
							column_pval_group = 'thresh_pval_1'
							column_quantile = 'thresh_quantile_overlap'
							column_thresh_query = [column_1,column_2,column_3,column_pval_group,column_quantile]

							dict_thresh = dict()
							thresh_vec = []
							if len(dict_thresh)==0:
								if len(thresh_vec)==0:
									thresh_overlap_default_1 = 0
									thresh_overlap_default_2 = 0
									thresh_overlap = 0
													
									# thresh_pval_1 = 0.10
									# thresh_pval_1 = 0.20
									thresh_pval_group = 0.25
									# thresh_quantile_overlap = 0.50
									thresh_quantile_overlap = 0.75
									thresh_vec = [thresh_overlap_default_1,thresh_overlap_default_2,thresh_overlap,thresh_pval_group,thresh_quantile_overlap]
								
								dict_thresh = dict(zip(column_thresh_query,thresh_vec))

							df_query1 = self.test_query_training_group_pre1(data=df_query1,motif_id=motif_id_query,dict_annot=dict_group_annot_1,
																				method_type_feature_link=method_type_feature_link,
																				dict_thresh=dict_thresh,thresh_vec=thresh_vec,input_file_path='',
																				save_mode=1,output_file_path=output_file_path_query,verbose=verbose,select_config=select_config)

							column_corr_1 = 'peak_tf_corr'
							column_pval = 'peak_tf_pval_corrected'
							column_vec_query = [column_corr_1,column_pval,column_score_query1]

							# column_pred1 = '%s.pred'%(method_type_feature_link)
							id_pred1 = (df_query1[column_pred1]>0)
							df_pre2 = df_query1.loc[id_pred1,:]
							df_pre2, select_config = self.test_query_feature_quantile_1(data=df_pre2,query_idvec=[],column_vec_query=column_vec_query,save_mode=1,verbose=verbose,select_config=select_config)

							peak_loc_query_1 = []
							peak_loc_query_2 = []
							flag_corr_1 = 1
							flag_score_query_1 = 0
							flag_enrichment_sel = 1
							peak_loc_query_group2_1 = self.test_query_training_select_pre1(data=df_pre2,column_vec_query=[],
																								flag_corr_1=flag_corr_1,flag_score_1=flag_score_query_1,
																								flag_enrichment_sel=flag_enrichment_sel,input_file_path='',
																								save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',
																								verbose=verbose,select_config=select_config)
							
							peak_num_group2_1 = len(peak_loc_query_group2_1)
							peak_query_vec = peak_loc_query_group2_1  # the peak loci in class 1
						
						elif flag_select_1==2:
							df_pre2 = df_query1.loc[id_pred1,:]
							peak_query_vec = df_pre2.index
							peak_query_num_1 = len(peak_query_vec)

						if flag_select_2==1:
							# select training sample in class 2
							print('feature_type_vec_query: ',feature_type_vec_query)
							peak_vec_2_1, peak_vec_2_2 = self.test_query_training_select_group2(data=df_pre1,motif_id_query=motif_id_query,
																									peak_query_vec_1=peak_query_vec,
																									feature_type_vec=feature_type_vec_query,
																									save_mode=save_mode,verbose=verbose,select_config=select_config)

							peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)

						elif flag_select_2 in [2,3]:
							peak_vec_2, peak_vec_2_1, peak_vec_2_2 = self.test_query_training_select_group2_2(data=df_pre1,id_query=id_pred1,method_type_query=method_type_feature_link,
																												flag_sample=flag_sample,flag_select_2=flag_select_2,
																												save_mode=1,verbose=verbose,select_config=select_config)

						if flag_select_1>0:
							df_pre1.loc[peak_query_vec,'class'] = 1

						if flag_select_2 in [1,3]:
							df_pre1.loc[peak_vec_2_1,'class'] = -1
							df_pre1.loc[peak_vec_2_2,'class'] = -2

							peak_num_2_1 = len(peak_vec_2_1)
							peak_num_2_2 = len(peak_vec_2_2)
							print('peak_vec_2_1, peak_vec_2_2: ',peak_num_2_1,peak_num_2_2)
							peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)

						elif flag_select_2 in [2]:
							df_pre1.loc[peak_vec_2,'class'] = -1

						peak_num_2 = len(peak_vec_2)
						print('peak_vec_2: ',peak_num_2)

						peak_query_num_1 = len(peak_query_vec)
						print('peak_query_vec: ',peak_query_num_1)

						peak_vec_1 = peak_query_vec
						sample_id_train = pd.Index(peak_vec_1).union(peak_vec_2,sort=False)

						df_query_pre1 = df_pre1.loc[sample_id_train,:]
						
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

						df_pre1[motif_id_query] = 0
						df_pre1.loc[peak_vec_1,motif_id_query] = 1
						# df_pre1.loc[peak_vec_2,motif_id_query] = 0
						peak_num1 = len(peak_vec_1)
						print('peak_vec_1: ',peak_num1)
						print(df_pre1.loc[peak_vec_1,[column_motif,motif_id_query]])

						iter_num = 1
						flag_train1 = 1
						if flag_train1>0:
							print('feature_type_vec_query: ',feature_type_vec_query)
							key_vec = np.asarray(list(dict_feature.keys()))
							print('dict_feature: ',key_vec)
							peak_loc_pre1 = df_pre1.index
							id1 = (df_pre1['class']==1)
							peak_vec_1 = peak_loc_pre1[id1]
							peak_query_num1 = len(peak_vec_1)

							# config_id_2: configuration for selecting class 0 sample
							# flag_scale_1: 0, without feature scaling; 1, with feature scaling
							# train_id1 = 1
							# train_id1 = select_config['train_id1']
							flag_scale_1 = select_config['flag_scale_1']
							type_query_scale = flag_scale_1

							iter_id1 = 0
							filename_prefix_save = filename_link_prefix
							filename_save_annot = filename_link_annot
							output_filename = filename_save_link_pre1
							input_file_path = input_file_path_pre1
							output_file_path_query = output_file_path_pre2

							flag2=1
							if flag2>0:
							# try:
								df_pre2 = self.test_query_compare_binding_train_unit1(data=df_pre1,peak_query_vec=peak_loc_pre1,peak_vec_1=peak_vec_1,
																						motif_id_query=motif_id_query,
																						dict_feature=dict_feature,feature_type_vec=feature_type_vec_query,
																						sample_idvec_query=sample_idvec_query,
																						motif_data=motif_data_query1,
																						flag_scale=flag_scale_1,input_file_path=input_file_path,
																						save_mode=1,output_file_path=output_file_path_query,output_filename=output_filename,
																						filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,
																						verbose=verbose,select_config=select_config)
							# except Exception as error:
							# 	print('error! ',error)
							# 	print('motif query: ',motif_id_query,i1)
							# 	continue

					stop_1 = time.time()
					print('TF binding prediction for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop_1-start_1))
				
				# except Exception as error:
				# 	print('error! ',error, motif_id_query,motif_id1,motif_id2,i1)
					# return

	def run_pre1(self,chromosome='1',run_id=1,species='human',cell=0,generate=1,chromvec=[],testchromvec=[],data_file_type='',metacell_num=500,peak_distance_thresh=100,
						highly_variable=1,input_dir='',filename_atac_meta='',filename_rna_meta='',filename_motif_data='',filename_motif_data_score='',file_mapping='',
						method_type_feature_link='',tf_name='',filename_prefix='',filename_annot='',input_link='',columns_1='',
						output_dir='',output_filename='',path_id=2,save=1,type_group=0,type_group_2=0,type_group_load_mode=1,method_type_group='phenograph.20',
						n_components=100,n_components_2=50,neighbor_num=100,neighbor_num_sel=30,model_type_id='LogisticRegression',ratio_1=0.25,ratio_2=1.5,thresh_size_group=50,thresh_score_group_1=0.15,
						thresh_score='0.25,0.75',flag_group=-1,flag_embedding_compute=0,flag_clustering=0,flag_group_load=1,
						flag_scale_1=0,beta_mode=0,query_id1=-1,query_id2=-1,query_id_1=-1,query_id_2=-1,train_mode=0,config_id_load=-1):

		chromosome = str(chromosome)
		run_id = int(run_id)
		species_id = str(species)
		# cell = str(cell)
		cell_type_id = int(cell)
		print('cell_type_id: %d'%(cell_type_id))
		data_file_type = str(data_file_type)
		metacell_num = int(metacell_num)
		peak_distance_thresh = int(peak_distance_thresh)
		highly_variable = int(highly_variable)
		# upstream, downstream = int(upstream), int(downstream)
		# if downstream<0:
		# 	downstream = upstream
		# type_id_query = int(type_id_query)

		# thresh_fdr_peak_tf = float(thresh_fdr_peak_tf)
		type_group = int(type_group)
		type_group_2 = int(type_group_2)
		type_group_load_mode = int(type_group_load_mode)
		method_type_group = str(method_type_group)
		thresh_size_group = int(thresh_size_group)
		thresh_score_group_1 = float(thresh_score_group_1)
		thresh_score = str(thresh_score)
		method_type_feature_link = str(method_type_feature_link)

		n_components = int(n_components)
		n_component_sel = int(n_components_2)
		neighbor_num = int(neighbor_num)
		neighbor_num_sel = int(neighbor_num_sel)
		model_type_id1 = str(model_type_id)

		input_link = str(input_link)
		columns_1 = str(columns_1)
		filename_prefix = str(filename_prefix)
		filename_annot = str(filename_annot)
		tf_name = str(tf_name)

		if filename_prefix=='':
			filename_prefix = data_file_type
		
		# typeid2 = int(typeid2)
		# folder_id = int(folder_id)
		# config_id_2 = int(config_id_2)
		# config_group_annot = int(config_group_annot)
		
		ratio_1 = float(ratio_1)
		ratio_2 = float(ratio_2)
		flag_group = int(flag_group)

		flag_embedding_compute = int(flag_embedding_compute)
		flag_clustering = int(flag_clustering)
		flag_group_load = int(flag_group_load)

		# train_id1 = int(train_id1)
		flag_scale_1 = int(flag_scale_1)
		beta_mode = int(beta_mode)
		# motif_id_1 = str(motif_id_1)

		input_dir = str(input_dir)
		output_dir = str(output_dir)
		filename_atac_meta = str(filename_atac_meta)
		filename_rna_meta = str(filename_rna_meta)
		filename_motif_data = str(filename_motif_data)
		filename_motif_data_score = str(filename_motif_data_score)
		file_mapping = str(file_mapping)
		output_filename = str(output_filename)

		path_id = int(path_id)
		run_id_save = int(save)
		if run_id_save<0:
			run_id_save = run_id

		config_id_load = int(config_id_load)

		# celltype_vec = ['CD34_bonemarrow','pbmc']
		celltype_vec = ['pbmc']
		flag_query1=1
		if flag_query1>0:
			query_id1 = int(query_id1)
			query_id2 = int(query_id2)
			query_id_1 = int(query_id_1)
			query_id_2 = int(query_id_2)
			train_mode = int(train_mode)
			data_file_type = str(data_file_type)

			# run_id = 1
			type_id_feature = 0

			root_path_1 = '.'
			root_path_2 = '.'

			save_file_path_default = output_dir
			file_path_motif_score = input_dir

			select_config = {'root_path_1':root_path_1,'root_path_2':root_path_2,
								'data_file_type':data_file_type,
								'input_dir':input_dir,
								'output_dir':output_dir,
								'type_id_feature':type_id_feature,
								'metacell_num':metacell_num,
								'run_id':run_id,
								'filename_atac_meta':filename_atac_meta,
								'filename_rna_meta':filename_rna_meta,
								'filename_motif_data':filename_motif_data,
								'filename_motif_data_score':filename_motif_data_score,
								'filename_translation':file_mapping,
								'output_filename_link':output_filename,
								'path_id':path_id,
								'run_id_save':run_id_save,
								'input_link':input_link,
								'columns_1':columns_1,
								'filename_prefix':filename_prefix,
								'filename_annot':filename_annot,
								'tf_name':tf_name,
								'n_components':n_components,
								'n_component_sel':n_component_sel,
								'type_id_group':type_group,
								'type_id_group_2':type_group_2,
								'type_group_load_mode':type_group_load_mode,
								'method_type_group':method_type_group,
								'thresh_size_group':thresh_size_group,
								'thresh_score_group_1':thresh_score_group_1,
								'thresh_score':thresh_score,
								'method_type_feature_link':method_type_feature_link,
								'neighbor_num':neighbor_num,
								'neighbor_num_sel':neighbor_num_sel,
								'model_type_id1':model_type_id1,
								'ratio_1':ratio_1,
								'ratio_2':ratio_2,
								'flag_embedding_compute':flag_embedding_compute,
								'flag_clustering':flag_clustering,
								'flag_group_load':flag_group_load,
								'flag_scale_1':flag_scale_1,
								'beta_mode':beta_mode,
								'query_id1':query_id1,'query_id2':query_id2,
								'query_id_1':query_id_1,'query_id_2':query_id_2,
								'train_mode':train_mode,
								'config_id_load':config_id_load,
								'save_file_path_default':save_file_path_default}
			
			verbose = 1
			flag_score_1 = 0
			flag_score_2 = 0
			flag_compare_1 = 1
			if flag_group<0:
				flag_group_1 = 1
			else:
				flag_group_1 = flag_group
			
			flag_1 = flag_group_1
			if flag_1>0:
				flag_select_1 = 1
				flag_select_2 = 1
				select_config.update({'flag_select_1':flag_select_1,'flag_select_2':flag_select_2})
				flag1=1
				if flag1>0:
					# type_query_group = 1
					type_query_group = 0
					# parallel_group = 1
					parallel_group = 0
					# flag_score_query = 1
					flag_score_query = 0
					select_config.update({'type_query_group':type_query_group,'parallel_group':parallel_group})

					flag_select_1 = 1
					flag_select_2 = 1
					select_config.update({'flag_select_1':flag_select_1,'flag_select_2':flag_select_2})

					self.test_query_compare_binding_compute_2(data=[],dict_feature=[],feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)
					

def run(chromosome,run_id,species,cell,generate,chromvec,testchromvec,data_file_type,input_dir,
			filename_atac_meta,filename_rna_meta,filename_motif_data,filename_motif_data_score,file_mapping,metacell_num,peak_distance_thresh,
			highly_variable,method_type_feature_link,tf_name,filename_prefix,filename_annot,input_link,columns_1,
			output_dir,output_filename,method_type_group,thresh_size_group,thresh_score_group_1,
			n_components,n_components_2,neighbor_num,neighbor_num_sel,model_type_id,ratio_1,ratio_2,thresh_score,
			upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
			typeid2,folder_id,config_id_2,config_group_annot,flag_group,flag_embedding_compute,flag_clustering,flag_group_load,train_id1,flag_scale_1,beta_mode,motif_id_1,query_id1,query_id2,query_id_1,query_id_2,train_mode,config_id_load):
	
	file_path_1 = '.'
	test_estimator1 = _Base2_2_pre1(file_path=file_path_1)

	flag_1=1
	if flag_1==1:
		test_estimator1.run_pre1(chromosome,run_id,species,cell,generate,chromvec,testchromvec,data_file_type=data_file_type,metacell_num=metacell_num,peak_distance_thresh=peak_distance_thresh,
									highly_variable=highly_variable,input_dir=input_dir,filename_atac_meta=filename_atac_meta,filename_rna_meta=filename_rna_meta,
									filename_motif_data=filename_motif_data,filename_motif_data_score=filename_motif_data_score,file_mapping=file_mapping,
									method_type_feature_link=method_type_feature_link,tf_name=tf_name,filename_prefix=filename_prefix,filename_annot=filename_annot,input_link=input_link,columns_1=columns_1,
									output_dir=output_dir,output_filename=output_filename,path_id=path_id,save=save,type_group=type_group,type_group_2=type_group_2,type_group_load_mode=type_group_load_mode,method_type_group=method_type_group,
									n_components=n_components,n_components_2=n_components_2,neighbor_num=neighbor_num,neighbor_num_sel=neighbor_num_sel,ratio_1=ratio_1,ratio_2=ratio_2,thresh_size_group=thresh_size_group,thresh_score_group_1=thresh_score_group_1,
									thresh_score=thresh_score,flag_group=flag_group,flag_embedding_compute=flag_embedding_compute,flag_clustering=flag_clustering,flag_group_load=flag_group_load,
									flag_scale_1=flag_scale_1,beta_mode=beta_mode,query_id1=query_id1,query_id2=query_id2,query_id_1=query_id_1,query_id_2=query_id_2,train_mode=train_mode,config_id_load=config_id_load)

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="1", help="experiment id")
	parser.add_option("-f","--chromosome", default="1", help="Chromosome name")
	parser.add_option("-g","--generate", default="0", help="whether to generate feature vector: 1: generate; 0: not generate")
	parser.add_option("-c","--chromvec",default="1",help="chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-t","--testchromvec",default="10",help="test chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-i","--species",default="0",help="species id")
	parser.add_option("-b","--cell",default="0",help="cell type")
	parser.add_option("--data_file_type",default="pbmc",help="the cell type or dataset annotation")
	parser.add_option("--input_dir",default=".",help="the directory where the ATAC-seq and RNA-seq data of the metacells are saved")
	parser.add_option("--atac_meta",default="-1",help="file path of ATAC-seq data of the metacells")
	parser.add_option("--rna_meta",default="-1",help="file path of RNA-seq data of the metacells")
	parser.add_option("--motif_data",default="-1",help="file path of binary motif scannning results")
	parser.add_option("--motif_data_score",default="-1",help="file path of the motif scores by motif scanning")
	parser.add_option("--file_mapping",default="-1",help="file path of the mapping between TF motif identifier and the TF name")
	parser.add_option("--metacell",default="500",help="metacell number")
	parser.add_option("--peak_distance",default="500",help="peak distance")
	parser.add_option("--highly_variable",default="1",help="highly variable gene")
	parser.add_option("--method_type_feature_link",default="Unify",help='method_type_feature_link')
	parser.add_option("--tf",default='-1',help='the TF for which to predict peak-TF associations')
	parser.add_option("--filename_prefix",default='-1',help='prefix as part of the filenname of the initially predicted peak-TF assocations')
	parser.add_option("--filename_annot",default='1',help='annotation as part of the filename of the initially predicted peak-TF assocations')
	parser.add_option("--input_link",default='-1',help=' the directory where initially predicted peak-TF associations are saved')
	parser.add_option("--columns_1",default='pred,score',help='the columns corresponding to binary prediction and peak-TF association score')
	parser.add_option("--output_dir",default='output_file',help='the directory to save the output')
	parser.add_option("--output_filename",default='-1',help='filename of the predicted peak-TF assocations')
	parser.add_option("--method_type_group",default="phenograph.20",help="the method for peak clustering")
	parser.add_option("--thresh_size_group",default="1",help="the threshold on peak cluster size")
	parser.add_option("--thresh_score_group_1",default="0.15",help="the threshold on peak-TF assocation score")
	parser.add_option("--component",default="100",help='the number of components to keep when applying SVD')
	parser.add_option("--component2",default="50",help='feature dimensions to use in each feature space')
	parser.add_option("--neighbor",default='100',help='the number of nearest neighbors estimated for each peak')
	parser.add_option("--neighbor_sel",default='30',help='the number of nearest neighbors to use for each peak when performing pseudo training sample selection')
	parser.add_option("--model_type",default="LogisticRegression",help="the prediction model")
	parser.add_option("--ratio_1",default="0.25",help="the ratio of pseudo negative training samples selected from peaks with motifs but without initially predicted TF binding compared to selected pseudo positive training samples")
	parser.add_option("--ratio_2",default="1.5",help="the ratio of pseudo negative training samples selected from peaks without motifs compared to selected pseudo positive training samples")
	parser.add_option("--thresh_score",default="0.25,0.75",help="thresholds on the normalized peak-TF scores to select pseudo positive training samples from the paired peak groups with or without enrichment of initially predicted TF-binding peaks")
	parser.add_option("--upstream",default="100",help="TRIPOD upstream")
	parser.add_option("--downstream",default="-1",help="TRIPOD downstream")
	parser.add_option("--typeid1",default="0",help="TRIPOD type_id_query")
	parser.add_option("--thresh_fdr_peak_tf",default="0.2",help="GRaNIE thresh_fdr_peak_tf")
	parser.add_option("--path1",default="2",help="file_path_id")
	parser.add_option("--save",default="-1",help="run_id_save")
	parser.add_option("--type_group",default="0",help="type_id_group")
	parser.add_option("--type_group_2",default="0",help="type_id_group_2")
	parser.add_option("--type_group_load_mode",default="1",help="type_group_load_mode")
	parser.add_option("--typeid2",default="0",help="type_id_query_2")
	parser.add_option("--folder_id",default="1",help="folder_id")
	parser.add_option("--config_id_2",default="1",help="config_id_2")
	parser.add_option("--config_group_annot",default="1",help="config_group_annot")
	parser.add_option("--flag_group",default="-1",help="flag_group")
	parser.add_option("--flag_embedding_compute",default="0",help="compute feature embeddings")
	parser.add_option("--flag_clustering",default="-1",help="perform clustering")
	parser.add_option("--flag_group_load",default="-1",help="load group annotation")
	parser.add_option("--train_id1",default="1",help="train_id1")
	parser.add_option("--flag_scale_1",default="0",help="flag_scale_1")
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
		opts.data_file_type,
		opts.input_dir,
		opts.atac_meta,
		opts.rna_meta,
		opts.motif_data,
		opts.motif_data_score,
		opts.file_mapping,
		opts.metacell,
		opts.peak_distance,
		opts.highly_variable,
		opts.method_type_feature_link,
		opts.tf,
		opts.filename_prefix,
		opts.filename_annot,
		opts.input_link,
		opts.columns_1,
		opts.output_dir,
		opts.output_filename,
		opts.method_type_group,
		opts.thresh_size_group,
		opts.thresh_score_group_1,
		opts.component,
		opts.component2,
		opts.neighbor,
		opts.neighbor_sel,
		opts.model_type,
		opts.ratio_1,
		opts.ratio_2,
		opts.thresh_score,
		opts.upstream,
		opts.downstream,
		opts.typeid1,
		opts.thresh_fdr_peak_tf,
		opts.path1,
		opts.save,
		opts.type_group,
		opts.type_group_2,
		opts.type_group_load_mode,
		opts.typeid2,
		opts.folder_id,
		opts.config_id_2,
		opts.config_group_annot,
		opts.flag_group,
		opts.flag_embedding_compute,
		opts.flag_clustering,
		opts.flag_group_load,
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







