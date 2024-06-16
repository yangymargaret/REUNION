#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import scipy
import scipy.io
import sklearn
import math
import scanpy as sc
import scanpy.external as sce
import anndata as ad
from anndata import AnnData

from copy import deepcopy

import os
import os.path
from optparse import OptionParser

from scipy import stats
from scipy.stats import chisquare, fisher_exact, chi2_contingency, zscore, poisson, multinomial, norm, pearsonr, spearmanr
from scipy.stats.contingency import expected_freq

import time
from timeit import default_timer as timer

from joblib import Parallel, delayed
from .test_rediscover_compute_2 import _Base2_2_1
from .test_group_1 import _Base2_group1
from . import utility_1
from .utility_1 import test_query_index
import h5py
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

		self.verbose_internal = 1
		file_path1 = self.save_path_1
		test_estimator_group = _Base2_group1(file_path=file_path1,select_config=select_config)
		self.test_estimator_group = test_estimator_group

	## ====================================================
	# perform clustering of observations
	def test_query_feature_clustering_pre1(self,data=[],feature_query_vec=[],feature_type_vec=[],flag_cluster_1=1,flag_cluster_2=0,save_mode=1,verbose=0,select_config={}):

		"""
		perform group estimation of observations
		:param data: dictionary containing the low-dimensional embeddings of observations
		:param feature_query_vec: (array or list) the observations for which to perform clustering; if not specified, all observations in the latent representation matrix are included
		:param feature_type_vec: (array or list) feature types of feature representations of the observations
		:param flag_cluster_1: indicator of whether to perform clustering
		:param flag_cluster_2: indicator of whether to query average signals in each group
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1: dictionary containing the embeddings of observations for each feature type
				 2: dictionary containing group assignment of observations for each feature type (if flag_cluster_1>0) 
				 3: dictionary containing the average signals of the specific feature type (e.g., accessibility) of members in each group for each feature type (if flag_cluster_2>0)
		"""	

		flag_clustering_1=1
		verbose_internal = self.verbose_internal
		if flag_clustering_1>0:
			data_file_type_query = select_config['data_file_type']
			metric = 'euclidean'
			n_components = select_config['n_components']
			n_component_sel = select_config['n_component_sel']
			feature_mode_query = select_config['feature_mode']
			# flag_cluster_1 = 1
			# flag_cluster_2 = 0
			flag_combine = 0
			
			column_1 = 'file_path_group_query'
			file_path_group_query = select_config[column_1]
			input_file_path = file_path_group_query
			output_file_path = input_file_path

			# type_id_group = 0
			type_id_group = select_config['type_id_group']
			filename_prefix_save = '%s.pre%d.%d'%(data_file_type_query,feature_mode_query,type_id_group)
			# filename_save_annot = '1'
			if len(feature_type_vec)==0:
				feature_type_vec = ['peak_motif','peak_tf']

			dict_query_1 = data
			method_type_dimension = select_config['method_type_dimension']
			if len(dict_query_1)==0:
				print('load feature embeddings')
				start = time.time()

				feature_type_vec_query2 = feature_type_vec
				# recontruct: 1: load reconstructed matrix;
				reconstruct = 0
				n_components_query = n_components
				# n_component_sel = select_config['n_component_sel']
				filename_save_annot = '%s_%d.1'%(method_type_dimension,n_components_query)
				dict_file = self.dict_file_feature  # filename of the feature embeddings;
				
				# query computed low-dimensional embeddings of observations
				dict_latent_query1 = self.test_query_feature_embedding_load_1(dict_file=dict_file,
																				feature_query_vec=feature_query_vec,
																				feature_type_vec=feature_type_vec_query2,
																				method_type_vec=[],
																				method_type_dimension=method_type_dimension,
																				n_components=n_components,
																				n_component_sel=n_component_sel,
																				reconstruct=reconstruct,
																				flag_combine=flag_combine,
																				input_file_path=input_file_path,
																				save_mode=0,output_file_path='',output_filename='',
																				filename_prefix_save=filename_prefix_save,
																				filename_save_annot=filename_save_annot,
																				verbose=verbose,select_config=select_config)

				if verbose_internal==2:
					print('dict_latent_query1: ',dict_latent_query1)

				stop = time.time()
				print('load feature embeddings used %s.2fs'%(stop-start))
			else:
				dict_latent_query1 = dict_query_1

			print('perform peak clustering')
			start = time.time()
			overwrite = False
			# overwrite_2 = True
			overwrite_2 = False
			
			flag_iter_2 = 0
			# method_type_vec_group = ['phenograph']
			method_type_group_name = select_config['method_type_group_name']
			method_type_vec_group = [method_type_group_name]
			
			# perform clustering of peak loci based on the low-dimensional embeddings
			filename_prefix_save_pre2 = filename_prefix_save
			filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_component_sel)
			feature_type_vec_pre1 = ['gene','peak']  #  the first level feature type, which may include peak, gene, and motif (associated with TF)
			dict_group_query1, dict_group_query2 = self.test_query_feature_clustering_1(dict_feature=dict_latent_query1,
																							feature_type_vec=feature_type_vec,
																							method_type_vec_group=method_type_vec_group,
																							type_id_group=type_id_group,
																							n_components=n_components,
																							metric=metric,
																							subsample_ratio=-1,
																							flag_cluster_1=flag_cluster_1,
																							flag_cluster_2=flag_cluster_2,
																							overwrite=overwrite,
																							overwrite_2=overwrite_2,
																							input_file_path=input_file_path,
																							save_mode=1,
																							output_file_path=output_file_path,
																							filename_prefix_save=filename_prefix_save_pre2,
																							filename_save_annot=filename_save_annot_2,
																							verbose=verbose,select_config=select_config)

			stop = time.time()
			print('performing peak clustering used %.2fs'%(stop-start))

			return dict_latent_query1, dict_group_query1, dict_group_query2

	## ====================================================
	# perform clustering of observations
	def test_query_feature_clustering_1(self,dict_feature=[],feature_type_vec=[],feature_type_query='peak',method_type_vec_dimension=['SVD'],method_type_vec_group=['phenograph'],
											metric='euclidean',subsample_ratio=-1,flag_iter_1=1,flag_cluster_1=1,flag_cluster_2=0,flag_combine=0,overwrite=False,overwrite_2=False,input_file_path='',
											save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		perform clustering of observations (peak loci)
		:param data: dictionary containing the low-dimensional embeddings of observations in the specific feature space
		:param feature_type_vec: (array or list) feature types of feature representations of the observations
		:param feature_type_query: (str) the observation type (e.g., peak, gene, and TF)
		:param method_type: (str) the method used for feature dimension reduction
		:param method_type_vec_dimension: (list) method used for dimension reduction for each feature type
		:param method_type_vec_group: (list) method used for clustering for each feature type
		:param metric: distance metric
		:param subsample_ratio: percentage of the observations to sample; if subsample_ratio=-1 , all the observations are included
		:param flag_iter_1: indicator of whether to compute SSE with different numbers of clusters using K-means clustering
		:param flag_cluster_1: indicator of whether to perform clustering
		:param flag_cluster_2: indicator of whether to query average signals in each group
		:param overwrite: indicator of whether to overwrite the parameter configuration in select_config for clustering
		:param overwrite_2: indicator of whether to re-compute SSE with different numbers of clusters and overwrite the current file of the SSE information
		:param input_file_path: the directroy to retrieve the data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1: dictionary containing group assignment of observations (if flag_cluster_1>0) for each feature type
				 2: dictionary containing the average signals of the specific feature type (e.g., accessibility) of members in each group (if flag_cluster_2>0) for each feature type
		"""

		flag_query_1 = 1
		verbose_internal = self.verbose_internal # verbosity level for printing intermediate information
		if flag_query_1>0:
			flag_cluster_query_1 = 1
			flag_cluster_query_2 = 0

			# feature_type_vec_1 = ['gene','peak'] # the first level feature type vector, which may include peak, gene, and motif (associated with TF)
			method_type_query = method_type_vec_dimension[0] # the method used for feature dimension reduction
		
			if len(feature_type_vec)==0:
				# the feature types of peak loci
				# peak-motif: sequence feature of motif occurrences in peak loci; 
				# peak_tf: peak accessiblity feature
				feature_type_vec = ['peak_motif','peak_tf']
			field_query = ['latent_%s'%(feature_type_query) for feature_type_query in feature_type_vec] # represent embeddings based on the corresponding feature type of peaks
			annot_str_vec = field_query

			# query feature type annotation
			dict_feature_type_annot1, dict_feature_type_annot2 = self.test_query_feature_type_annot_1(save_mode=1,verbose=verbose,select_config=select_config)

			column_1 = 'type_id_group_2'
			type_id_group_2 = -1
			if column_1 in select_config:
				type_id_group_2 = select_config[column_1]

			if type_id_group_2>=0:
				type_vec_group_2 = [type_id_group_2]
			else:
				field_num = len(field_query)
				type_vec_group_2 = np.arange(field_num)
			# print('type_vec_group_2: ',type_vec_group_2)

			method_type_group_name = select_config['method_type_group_name']
			method_type_group_neighbor = select_config['method_type_group_neighbor']
			neighbor_num_1 = method_type_group_neighbor

			if len(method_type_vec_group)==0:
				method_type_vec_group = [method_type_group_name]
				select_config.update({'method_type_vec_group':method_type_vec_group})

			neighbors_vec = [neighbor_num_1] # the parameter of the number of neighbors used in PhenoGraph clustering
			n_clusters_vec = [30, 50, 100] # the number of clusters
			distance_threshold_vec = [20, 50, -1] # the parameter of distance threshold used in agglomerative clustering
			# metric = 'euclidean'	# the distance metric used for clustering
			linkage_type_idvec = [0]
			select_config.update({'neighbors_vec':neighbors_vec,
									'n_clusters_vec':n_clusters_vec,
									'distance_threshold_vec':distance_threshold_vec,
									'linkage_type_idvec':linkage_type_idvec})

			file_path_group_query = select_config['file_path_group_query'] # the directory to save the group label estimations
			output_file_path = file_path_group_query

			type_id_compute, type_id_feature = 0, 0
			column_2, column_3 = 'type_id_compute', 'type_id_feature'
			# query the parameters type_id_computed, type_id_feature from select_config
			select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=[column_2,column_3],
																				default_parameter=[type_id_compute,type_id_feature],
																				overwrite=False,select_config=select_config)
			type_id_compute, type_id_feature = param_vec[0:2]

			n_component_sel = select_config['n_component_sel'] # the number of latent components retained in dimension reduction
			if filename_save_annot=='':
				filename_save_annot = '%d_%d.%d'%(n_component_sel,type_id_compute,type_id_feature)

			dict_query_1 = dict()
			dict_query_2 = dict()
			for type_id_query in type_vec_group_2:
				field_id = field_query[type_id_query]
				df_latent_query = dict_feature[field_id] # feature embeddings of peak loci;
				
				dict_query_1.update({field_id:df_latent_query})
				field_query_2 = [field_id]
				if verbose_internal==2:
					annot_str1 = dict_feature_type_annot2[field_id]
					print('feature embeddings of peak loci, dataframe of size ',df_latent_query.shape, ' %s'%(annot_str1))
					print('preview:\n',df_latent_query[0:2])

				# perform group estimation
				t_vec_1 = self.test_query_association_group_1(data=dict_query_1,feature_type_query=feature_type_query,
																method_type_vec_group=method_type_vec_group,
																field_query=field_query_2,
																flag_iter_1=flag_iter_1,
																flag_cluster_1=flag_cluster_query_1,
																flag_cluster_2=flag_cluster_query_2,
																overwrite=overwrite_2,
																input_file_path=input_file_path,
																save_mode=save_mode,
																output_file_path=output_file_path,
																output_filename='',
																filename_prefix_save=filename_prefix_save,
																filename_save_annot=filename_save_annot,
																verbose=verbose,select_config=select_config)

				# group assignment of peak loci
				df_group_query, dict_feature_group, select_config = t_vec_1[0:3]
				dict1 = {'group':df_group_query,'signal':dict_feature_group}
				dict_query_2.update({field_id:dict1})

			return dict_query_1, dict_query_2

	## ====================================================
	# parameter configuration for the methods used for clustering
	def test_cluster_query_config_pre1(self,method_type_vec=[],neighbors_vec=[20,30],column_query='method_type_group_neighbor',n_clusters_vec=[30,50,100],distance_threshold_vec=[20,50,-1],metric='euclidean',linkage_type_idvec=[0],overwrite=False,verbose=0,select_config={}):

		"""
		parameter configuration for the methods used for clustering
		:param data: dictionary containing the low-dimensional embeddings of observations in the specific feature space
		:param feature_type_query: (str) feature type of the observations (e.g., peak, gene, and TF)
		:param method_type: the method used for feature dimension reduction
		:param method_type_vec_group: (list) methods used for clustering
		:param neighbors_vec: (list) the parameter of the number of neighbors used in PhenoGraph clustering
		:param column_query: (str) the field in select_config specifying the parameter of number of the neighbors used in PhenoGraph clustering
		:param n_clusters_vec: (list) the number of clusters
		:param distance_threshold_vec: (list) the parameter of distance threshold used in agglomerative clustering
		:param metric: (str) the distance metric
		:param linkage_type_idvec: (list) identifiers corresponding to the parameter of linkage type used in agglomerative clustering
		:param overwrite: indicator of whether to overwrite the parameter configuration in select_config for clustering
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1: list containing configuration parameters for each method used for clustering
				 2: dictionary containing updated parameters
		"""

		test_estimator_group = self.test_estimator_group
		if len(method_type_vec)==0:
			column_1 = 'method_type_vec_group'
			if column_1 in select_config:
				method_type_vec_group = select_config[column_1]
				method_type_vec = method_type_vec_group
			else:
				# method_type_vec = ['MiniBatchKMeans', 'phenograph', 'AgglomerativeClustering'] # the methods used for peak clustering
				method_type_vec = ['phenograph'] # the methods used for peak clustering 
				select_config.update({column_1:method_type_vec})

		if (column_query!='') and (column_query in select_config):
			method_type_group_neighbor = select_config[column_query]
			neighbor_num_1 = method_type_group_neighbor
			neighbors_vec = [neighbor_num_1]

		metric = 'euclidean'	# the distance metric used for clustering
		linkage_type_idvec = [0] 	# identifier corresponding to the parameter of linkage type used in agglomerative clustering
		field_query = ['neighbors_vec', 'n_clusters_vec', 'distance_threshold_vec', 'linkage_type_idvec']

		list1 = [neighbors_vec, n_clusters_vec, distance_threshold_vec, linkage_type_idvec]
		dict_config1 = dict(zip(field_query, list1))
		select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=list1,overwrite=overwrite,select_config=select_config)

		distance_threshold_pre, linkage_type_id_pre, neighbors_pre, n_clusters_pre = -1, 0, 20, 100 # the default parameter used for clustering
		list_config = test_estimator_group.test_cluster_query_config_1(method_type_vec=method_type_vec,
																			distance_threshold=distance_threshold_pre,
																			linkage_type_id=linkage_type_id_pre,
																			neighbors=neighbors_pre,
																			n_clusters=n_clusters_pre,
																			metric=metric,
																			select_config=select_config)

		return list_config, select_config

	## ====================================================
	# query filename to save SSE with different numbers of clusters
	def test_query_filename_cluster_num_1(self,feature_type,n_components=-1,method_type_dimension='SVD',method_type_group='MiniBatchKMeans',output_file_path='',filename_prefix_save='',select_config={}):

		"""
		query filename to save SSE with different numbers of clusters
		:param feature_type: (str) feature type of feature representations of the observations
		:param n_components: the nubmer of latent components used in the low-dimensional feature embedding of observations
		:param method_type_dimension: method used for feature dimension reduction
		:param method_type_group: method used for clustering
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param select_config: dictionary containing parameters
		:return: 1. filename to save SSE with different numbers of clusters
				 2. dictionary containing updated parameters
		"""

		if method_type_dimension=='':
			# the method used for feature dimension reduction
			method_type_dimension = select_config['method_type_dimension']  

		# filename_prefix_save_2 = '%s.%s.%s'%(filename_prefix_save,feature_type,method_type_query)
		# method_type_group = 'MiniBatchKMeans'	# the clustering method used to estimate the number of clusters
		# filename_save_annot_2 = '%s.%d.1'%(method_type_group,n_components)

		column_query = 'filename_%s_cluster_num'%(feature_type)
		if column_query in select_config:
			filename1 = select_config[column_query]
		else:
			column_1 = 'filename_prefix_cluster_num'
			column_2 = 'filename_annot_cluster_num'
			if column_1 in select_config:
				filename_prefix_save_2 = select_config['filename_prefix_cluster_num']
			else:
				filename_prefix_save_2 = '%s.%s.%s'%(filename_prefix_save,feature_type,method_type_dimension)
				select_config.update({column_1:filename_prefix_save_2})
				
			if column_2 in select_config:
				filename_save_annot_2 = select_config['filename_annot_cluster_num']
			else:
				if n_components<0:
					n_components = select_config['n_component_sel']
				filename_save_annot_2 = '%s.%d.1'%(method_type_group,n_components)
				select_config.update({column_2:filename_save_annot_2})

			if output_file_path=='':
				output_file_path = select_config['file_path_group_query'] # the directory to save the group label estimation

			# the file to save the SSE with different numbers of clusters
			filename1 = '%s/%s.sse_query.%s.txt'%(output_file_path,filename_prefix_save_2,filename_save_annot_2)		
			select_config.update({column_query:filename1})

		return filename1, select_config

	## ====================================================
	# perform clustering of observations
	def test_query_association_group_1(self,data=[],feature_type_query='peak',field_query=[],method_type_vec_group=[],flag_iter_1=1,flag_cluster_1=1,flag_cluster_2=0,overwrite=True,overwrite_2=False,input_file_path='',
											save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		perform clustering of observations
		:param data: dictionary containing the low-dimensional embeddings of observations in the specific feature space
		:param feature_type_query: (str) feature type of the observations (e.g., peak, gene, and TF)
		:param field_query: (array or list) fields used for retrieving feature representations from the argument data; if specified, the first field in the array will be used for retrieving feature representations
		:param method_type_vec_group: (array or list) methods used for clustering
		:param flag_iter_1: indicator of whether to compute SSE with different numbers of clusters using K-means clustering
		:param flag_cluster_1: indicator of whether to perform clustering
		:param flag_cluster_2: indicator of whether to query signal related attributes of the clusters
		:param overwrite: indicator of whether to overwrite the parameter configuration in select_config for clustering
		:param overwrite_2: indicator of whether to re-compute SSE with different numbers of clusters if overwrite the current file containing the SSE information
		:param input_file_path: the directroy to retrieve the data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1: dataframe containing group assignment of observations (if flag_cluster_1>0)
				 2: dictionary containing signal related attributes of the clusters
				 3. dictionary containing parameters
		"""

		test_estimator_group = self.test_estimator_group  # use test_estimator_group to perform clustering
		
		filename_prefix_default_1 = filename_prefix_save
		verbose_internal = self.verbose_internal
		df_group_query = []  # dataframe to save group assignment of observations
		dict_feature_group = dict() # dictionary to save the average signals of the specific feature type of members in each group
		if flag_cluster_1>0:
			# if feature_type_query=='':
			# 	feature_type_query = 'peak'
			if filename_prefix_save=='':
				filename_prefix_save = '%s.group'%(feature_type_query)
			else:
				filename_prefix_save = '%s.%s.group'%(filename_prefix_default_1,feature_type_query)

			dict_query_1 = data
			list1 = []
			column_1 = 'alpha0'
			# query the feature embeddings
			for field_id in field_query:
				df_query = []
				if field_id in dict_query_1:
					df_query_1 = dict_query_1[field_id]
					if len(df_query_1)>0:
						t_columns = df_query_1.columns.difference([column_1],sort=False)
						df_query = df_query_1.loc[:,t_columns]
				list1.append(df_query)
			
			feature_type = field_query[0] # the feature type used for clustering
			latent_mtx = list1[0] 	# the feature embedding matrix, shape: (peak_num,n_components)

			feature_query1 = latent_mtx.index 	# the indices of peak loci
			n_components = latent_mtx.shape[1]	# the number of latent feature dimensions
			feature_mtx_query = latent_mtx

			method_type_dimension = select_config['method_type_dimension']
			method_type_group_query = 'MiniBatchKMeans'	# the clustering method used to estimate the number of clusters
			filename1, select_config = self.test_query_filename_cluster_num_1(feature_type=feature_type,
																				n_components=n_components,
																				method_type_dimension=method_type_dimension,
																				method_type_group=method_type_group_query,
																				output_file_path=output_file_path,
																				filename_prefix_save=filename_prefix_save,
																				select_config=select_config)

			if (os.path.exists(filename1) == True) and (overwrite_2==False):
				print('the file exists: %s' % (filename1))
				df_cluster_query1 = pd.read_csv(filename1, index_col=0, sep='\t') # load previous estimation of SSE with different numbers of clusters
				flag_iter_1 = 0
			else:
				n_clusters_pre1 = 100
				n_clusters_pre2 = 300
				interval = 10
				interval_2 = 20
				# the numbers of clusters to perform peak clustering
				cluster_num_vec = list(np.arange(2,20))+list(np.arange(20,n_clusters_pre1+interval,interval))+list(np.arange(n_clusters_pre1,n_clusters_pre2+interval_2,interval_2))
				select_config.update({'cluster_num_vec':cluster_num_vec})
				output_filename = filename1

			# query the parameters type_id_compute, type_id_feature from select_config
			type_id_compute, type_id_feature = 0, 0
			column_2, column_3 = 'type_id_compute', 'type_id_feature'
			select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=[column_2,column_3],default_parameter=[type_id_compute,type_id_feature],overwrite=False,select_config=select_config)
			type_id_compute, type_id_feature = param_vec[0:2]
			config_vec_1 = [n_components, type_id_compute, type_id_feature]
			select_config.update({'config_vec_1':config_vec_1})

			flag_config1=1
			if len(method_type_vec_group)==0:
				column_1 = 'method_type_vec_group'
				# method_type_vec_group = ['MiniBatchKMeans', 'phenograph', 'AgglomerativeClustering']
				method_type_vec_1 = ['phenograph']
				
				select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=[column_1],default_parameter=[method_type_vec_group],overwrite=False,select_config=select_config)
				method_type_vec_group = param_vec[0]
				print('clustering method: ',method_type_vec_group)

			# parameter configuration for different methods for clustering
			if flag_config1>0:
				neighbors_vec = [20,30] # the parameter of the number of neighbors used in PhenoGraph clustering
				column_query1 = 'method_type_group_neighbor' # the field in select_config specifying the parameter of number of neighbors used in PhenoGraph clustering
				n_clusters_vec = [30,50,100] # the number of clusters
				distance_threshold_vec = [20,50,-1] # the parameter of distance threshold used in agglomerative clustering
				metric = 'euclidean'	# the distance metric used for clustering
				linkage_type_idvec = [0] 	# the parameter of linkage type used in agglomerative clustering
				list_config, select_config = self.test_cluster_query_config_pre1(method_type_vec=method_type_vec_group,
																					neighbors_vec=neighbors_vec,
																					n_clusters_vec=n_clusters_vec,
																					column_query=column_query1,
																					distance_threshold_vec=distance_threshold_vec,
																					metric=metric,
																					linkage_type_idvec=linkage_type_idvec,
																					overwrite=overwrite,
																					select_config=select_config)

			# perform group estimation
			filename_prefix_save_2 = '%s.%s.%s'%(filename_prefix_save,feature_type,method_type_dimension)
			df_group_query = test_estimator_group.test_query_group_1(data=feature_mtx_query,adata=[],
																		feature_type_query=feature_type_query,
																		list_config=list_config,
																		flag_iter=flag_iter_1,
																		flag_cluster=1,
																		save_mode=1,
																		output_file_path=output_file_path,
																		output_filename=output_filename,
																		filename_prefix_save=filename_prefix_save_2,
																		filename_save_annot=filename_save_annot,
																		verbose=verbose,select_config=select_config)

		return df_group_query, dict_feature_group, select_config

	## ====================================================
	# query group assignments of observations (peak loci)
	def test_query_feature_group_load_pre1(self,data=[],feature_type_vec=[],feature_query_vec=[],column_idvec=[],method_type_vec=[],thresh_size=0,load_mode=0,input_file_path='',
												save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query group assignment of observations (peak loci)
		:param data: dictionary containing the group assignment of observations (peaks) in each feature space
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param feature_query_vec: (array or list) observations; if not specified, all the observations in the dataframe of group assignment are included
		:param column_idvec: (array or list) columns representing the names or indices of TF, peak, and gene
		:param method_type_vec: (array or list) the methods used for peak clustering; if specified, group assignment by the first method in the array will be used
		:param thresh_size: threshold on group size to select groups
		:param load_mode: (int) indicator of whether to use group assignment saved in the argument data or load group assignment from files  
		:param input_file_path: the directory where group assignment data are saved
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing the dataframes of group assignment of peaks with added annotations (e.g.,label of initially predicted peak-TF association) for each feature type;
					the dictionary is also updated with the group size annotation for each feature type;
				 2. dictionary containing the indices of peaks without predicted TF binding but in the same group with the initially predicted TF-binding peaks for each feature type;
		"""

		if len(column_idvec)==0:
			column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]

		if len(method_type_vec)>0:
			method_type_group = method_type_vec[0]
		else:
			method_type_group = select_config['method_type_group']	# use the parameter in select_config

		method_type_dimension = select_config['method_type_dimension'] # the method used for feature dimension reduction
		dict_query1 = data
		dict_query2 = dict()

		if len(dict_query1)==0:
			dict_query1 = dict()
			load_mode = 1

		if input_file_path=='':
			file_path_group_query = select_config['file_path_group_query'] # the directory where group estimation is saved
			input_file_path_query = file_path_group_query
		else:
			input_file_path_query = input_file_path
		
		verbose_internal = self.verbose_internal
		for feature_type_query in feature_type_vec:		
			if load_mode>0:
				filename_prefix_1 = '%s.%s'%(filename_prefix_save,feature_type_query)
				input_filename = '%s/%s.%s.df_obs.1.txt'%(input_file_path_query,filename_prefix_1,filename_save_annot)
				if verbose_internal>0:
					print('load group assignment data from %s'%(input_filename))
				df_group = pd.read_csv(input_filename,index_col=0,sep='\t')
				df_group[column_id2] = df_group.index.copy()
				df_group['group'] = np.asarray(df_group[method_type_group])
				dict_query1.update({feature_type_query:df_group})
			else:
				df_group = dict_query1[feature_type_query]
				if not (column_id2 in df_group.columns):
					df_group[column_id2] = df_group.index.copy()

			if len(feature_query_vec)>0:
				df_group_query = df_group.loc[feature_query_vec,[method_type_group]] # the group of the peak loci with signal and with motif; the peak loci with score query above threshold and with motif
				df_group_query['count'] = 1
				df_group_query1_ori = df_group_query.groupby([method_type_group]).sum() # the number of members in each group using the specific clustering method

				df_group_query1 = df_group_query1_ori.loc[df_group_query1_ori['count']>thresh_size] # the groups with number of members above threshold
				group_vec = df_group_query1.index.unique()
				group_num1 = len(group_vec)

				if verbose_internal>0:
					print('group size threshold: ',thresh_size)
					print('the number of groups with group size above threshold: %d'%(group_num1))
					print('the groups: ',np.asarray(group_vec))
				
				df_group.index = np.asarray(df_group[method_type_group])
				feature_query_2 = df_group.loc[group_vec,column_id2].unique() # the peaks in the same group
				feature_vec_2 = pd.Index(feature_query_2).difference(feature_query_vec,sort=False)	# the peaks in the same group but without predicted TF binding
				df_group.index = np.asarray(df_group[column_id2]) # reset the index

				column_1 = 'label_1'
				column_2 = 'group'
				df_group[column_2] = np.asarray(df_group[method_type_group])
				df_group[column_1] = 0
				df_group.loc[feature_query_vec,column_1] = 1  # the peaks in each group and with predicted TF binding
				dict_query2.update({feature_type_query:feature_vec_2})
				
				field_id1 = '%s_group'%(feature_type_query)
				# save the group assigment and the number of members in each group
				dict_query1.update({feature_type_query:df_group,
									field_id1:df_group_query1_ori})
				
		return dict_query1, dict_query2

	## ====================================================
	# query group assignment of observations
	def test_query_feature_group_load_1(self,data=[],feature_type_vec=[],feature_query_vec=[],method_type_group='',thresh_size=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query group assignment of observations (peak loci)
		:param data: dictionary containing the group assignment of observations (peaks) in each feature space
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param feature_query_vec: (array or list) observations; if not specified, all the observations in the dataframe of group assignment are included
		:param method_type_group: (str) method used for clustering
		:param thresh_size: threshold on group size to select groups
		:param input_file_path: the directory where group assignment data are saved
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the number of shared members in paired groups
				 2. dictionary containing group annotations for each feature type, including the group size and the percentage of the population contained in each group
				 3. (dataframe) group assignment of observations in feature space 1
				 4. (dataframe) group assignment of observations in feature space 2
		"""

		flag_query1 = 1
		if flag_query1>0:
			method_type_vec_group = [method_type_group]
			type_id_group = 0
			load_mode = 0
			# load group assignment of observations
			# dict_query1_1: (feature_type,df_group), (feature_type_group,df_group_statistics); dict_query1_2: (feature_type,peak_loci)
			dict_query1_1, dict_query1_2 = self.test_query_feature_group_load_pre1(data=data,feature_type_vec=feature_type_vec,
																					feature_query_vec=feature_query_vec,
																					method_type_vec=method_type_vec_group,
																					thresh_size=thresh_size,
																					load_mode=load_mode,
																					input_file_path='',
																					save_mode=1,output_file_path='',output_filename='',
																					filename_prefix_save=filename_prefix_save,
																					filename_save_annot=filename_save_annot,
																					verbose=verbose,select_config=select_config)

			feature_type_vec_query = feature_type_vec
			feature_type_query_1,feature_type_query_2 = feature_type_vec_query[0:2]

			df_group_1_ori = dict_query1_1[feature_type_query_1]
			df_group_2_ori = dict_query1_1[feature_type_query_2]

			if len(feature_query_vec)==0:
				peak_loc_ori = df_group_1_ori.index
			else:
				peak_loc_ori = feature_query_vec

			df_group_1 = df_group_1_ori.loc[peak_loc_ori,:]
			df_group_2 = df_group_2_ori.loc[peak_loc_ori,:]
			verbose_internal = self.verbose_internal
			dict_annot1 = dict()
			feature_type_vec_annot = ['peak_mtx','peak_tf','peak_motif']
			for feature_type_query in feature_type_vec_query:
				list1 = [feature_type_query.find(feature_type_str) for feature_type_str in feature_type_vec_annot]
				b1, b2, b3 = list1[0:3]
				if (b1>=0) or (b2>=0):
					annot_str = 'peak accessibility feature'
				else:
					annot_str1 = 'peak-motif sequence feature'
				dict_annot1.update({feature_type_query:annot_str1})

			if verbose_internal>0:
				annot_str1 = dict_annot1[feature_type_query_1]
				annot_str2 = dict_annot1[feature_type_query_2]
				print('group annotation 1 (%s), dataframe of size '%(annot_str1),df_group_1.shape)
				print('data preview: ')
				print(df_group_1[0:2])
				print('group annotation 2 (%s), dataframe of size '%(annot_str2),df_group_2.shape)
				print('data preview: ')
				print(df_group_2[0:2])

			data_file_type_query = select_config['data_file_type']
			filename_save_annot2_2_pre1 = '%s.%s'%(method_type_group,data_file_type_query)
			input_filename = '%s/test_query_df_overlap.%s.1.txt'%(input_file_path,filename_save_annot2_2_pre1)
			
			if os.path.exists(input_filename)==True:
				df_overlap_compare = pd.read_csv(input_filename,index_col=0,sep='\t')
				print('df_overlap_compare ',df_overlap_compare.shape)
				print(df_overlap_compare[0:5])
				print(input_filename)			
			else:
				print('the file does not exist: %s'%(input_filename))
				# query the overlap between the pair of groups
				df_overlap_pre1 = self.test_query_group_overlap_1(df_list=[df_group_1,df_group_2],
																	feature_type_vec=feature_type_vec,
																	feature_query_vec=[],
																	select_config=select_config)
				column_1, column_2 = 'group1','group2'

				idvec = [column_1]
				df_overlap_pre1[column_1] = np.asarray(df_overlap_pre1.index)
				df_overlap_compare = df_overlap_pre1.melt(id_vars=idvec,var_name=column_2,value_name='overlap')
				df_overlap_compare.index = utility_1.test_query_index(df_overlap_compare,column_vec=[column_1,column_2],symbol_vec=['_'])
				df_overlap_compare['freq_obs'] = df_overlap_compare['overlap']/np.sum(df_overlap_compare['overlap'])
				
				output_filename = '%s/test_query_df_overlap.%s.1.txt'%(output_file_path,filename_save_annot2_2_pre1)
				df_overlap_compare.to_csv(output_filename,sep='\t')

			# column_vec_2 = ['overlap','freq_obs','freq_expect']
			group_vec_query = ['group1','group2']
			dict_group_basic_1 = dict()
			column_vec = ['overlap','freq_obs']
			for group_type in group_vec_query:
				df_group_basic_pre1 = df_overlap_compare.groupby(by=[group_type])
				df_group_freq_pre1 = df_group_basic_pre1[column_vec].sum()
				df_group_freq_pre1 = df_group_freq_pre1.rename(columns={'overlap':'count'})

				df_group_freq_pre1['group_type'] = group_type
				dict_group_basic_1.update({group_type:df_group_freq_pre1})

			return df_overlap_compare, dict_group_basic_1, df_group_1, df_group_2

	## ====================================================
	# query the overlap between members in two groups in the two feature spaces
	def test_query_group_overlap_1(self,df_list=[],feature_type_vec=[],feature_query_vec=[],column_query='group',query_mode=0,verbose=0,select_config={}):

		"""
		query the overlap between members in two groups in the two feature spaces
		:param df_list: the list of dataframes of group assignment of observations (peaks)
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param feature_query_vec: (array) observations; if not specified, all the observations in the dataframe of group assignment will be included
		:param column_query: the column in the dataframe of group assignment showing the group labels
		:param query_mode: indicator of which information to retrieve: 0: number of members in paired groups; 1: number of members in paired groups and the members in each group in each feature space
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1: dataframe of number of shared members in each pair of groups in the two feature spaces;
				 2 and 3: dictionaries containing the members in each group in feature space 1 and 2 (if query_mode>0)
		"""
		
		flag_query1 = 1
		if flag_query1>0:
			feature_type_num = len(df_list)
			if len(feature_type_vec)==0:
				feature_type_vec = ['feature%d'%(i1) for i1 in range(1,feature_type_num+1)]

			dict_query_1 = dict()
			field_query = ['group_vec_ori','group_vec_query','dict_group_annot']
			for i1 in range(feature_type_num):
				feature_type = feature_type_vec[i1]
				df_query = df_list[i1]

				# the unique group labels in group assignment in the corresponding feature space
				group_vec_ori = np.unique(df_query[column_query])
				dict_group_annot = dict()

				if len(feature_query_vec)>0:
					# query group assignment of the selected peak loci
					feature_query_1 = pd.Index(feature_query_vec).intersection(df_query.index,sort=False)
					df_query = df_query.loc[feature_query_1,:]
					group_vec_query = np.unique(df_query[column_query])
				else:
					group_vec_query = group_vec_ori
				
				feature_query_1 = df_query.index  # the peak loci
				for group_id_query in group_vec_query:
					feature_vec = feature_query_1[df_query[column_query]==group_id_query] # the members in the group
					dict_group_annot.update({group_id_query:feature_vec})

				list_1 = [group_vec_ori,group_vec_query,dict_group_annot]
				dict_1 = dict(zip(field_query,list_1))
				dict_query_1.update({feature_type:dict_1})

			list_query1 = []
			for field_id in field_query:
				list_query2 = [dict_query_1[feature_type][field_id] for feature_type in feature_type_vec]
				list_query1.append(list_query2)

			group_vec_ori_1, group_vec_ori_2 = list_query1[0]
			group_vec_1, group_vec_2 = list_query1[1]
			dict_group_annot_1, dict_group_annot_2 = list_query1[2]

			df_overlap = pd.DataFrame(index=group_vec_ori_1,columns=group_vec_ori_2,data=0)
			for group_id1 in group_vec_1:
				feature_vec_query1 = dict_group_annot_1[group_id1]
				for group_id2 in group_vec_2:
					feature_vec_query2 = dict_group_annot_2[group_id2]
					feature_vec_overlap = pd.Index(feature_vec_query1).intersection(feature_vec_query2,sort=False)
					df_overlap.loc[group_id1,group_id2] = len(feature_vec_overlap)

			self.df_overlap = df_overlap
			self.dict_group_annot_1 = dict_group_annot_1
			self.dict_group_annot_2 = dict_group_annot_2

			if query_mode>0:
				return df_overlap, dict_group_annot_1, dict_group_annot_2
			else:
				return df_overlap

	## ====================================================
	# compute peak enrichment in the groups
	def test_query_enrichment_pre1(self,df1,df2,column_query='count',stat_chi2_correction=True,stat_fisher_alternative='greater',contingency_table_query=0,save_mode=1,verbose=0,select_config={}):
		
		"""
		compute the enrichment of observations in the groups
		:param df1: (dataframe) the number of specified observations (e.g.,peaks with predicted TF binding) in each group
		:param df2: (dataframe) the number of observations (e.g.,genome-wide peaks) in each group
		:param stat_chi2_correction: indicator of whether to perform correction for chi-squared test
		:param stat_fisher_alternative: hypothesis used in Fisher's exact test
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) group annotations including the number of specified observations, 
								Fisher's exact test and chi-squared test statistics and p-values for each group
				 2. dictionary containing contingency tables based on which to compute enrichment of specified observations in each group 
				 	using statistical tests (if contingency_table_query>0)
		"""

		query_idvec = df1.index  # the groups
		query_num1 = len(query_idvec) # the nubmer of groups
		df_query1 = df1
		df_query_compare = df2

		column_vec_1 = ['stat_chi2_','pval_chi2_']
		column_vec_2 = ['stat_fisher_exact_','pval_fisher_exact_']

		count1 = np.sum(df1[column_query])
		count2 = np.sum(df2[column_query])
		dict_contingency_table = dict()
		for i1 in range(query_num1):
			query_id1 = query_idvec[i1]
			num1 = df_query1.loc[query_id1,column_query]
			num2 = df_query_compare.loc[query_id1,column_query]
			if num2>0:
				contingency_table = [[num1,count1-num1],[num2,count2-num2]]
				if verbose==2:
					print('contingency table for group %s: \n'%(query_id1))
					print(contingency_table)

				if not (stat_chi2_correction is None):
					correction = stat_chi2_correction
					stat_chi2_, pval_chi2_, dof_chi2_, ex_chi2_ = chi2_contingency(contingency_table,correction=correction)
					df_query1.loc[query_id1,column_vec_1] = [stat_chi2_, pval_chi2_]

				if not (stat_fisher_alternative is None):
					alternative = stat_fisher_alternative
					stat_fisher_exact_, pval_fisher_exact_ = fisher_exact(contingency_table,alternative=alternative)
					df_query1.loc[query_id1,column_vec_2] = [stat_fisher_exact_, pval_fisher_exact_]

				if contingency_table_query>0:
					dict_contingency_table.update({query_id1:contingency_table})

		return df_query1, dict_contingency_table

	## ====================================================
	# compute peak enrichment in each group for the specific feature type
	def test_query_group_enrichment_pre1(self,df_overlap=[],dict_group_compare={},column_query='count',column_vec_query=[],group_vec=['group1','group2'],
											flag_annot=1,flag_sort=1,stat_chi2_correction=True,stat_fisher_alternative='greater',contingency_table_query=0,
											save_mode=1,output_filename='',verbose=0,select_config={}):
	
		"""
		compute peak enrichment in each group for the specific feature type
		:param df_overlap: dataframe containing paired group assignment of peak loci
		:param dict_group_compare: dictionary containing group annotations based on group assignment of genome-wide peaks for each feature type 
		:param column_query: (str) name of the column representing number of members in each group
		:param column_vec_query: (array or list) columns in the group annotation dataframe representing number of members, percentage of members in the population, and the expeceted percentage
		:param group_vec: (array or list) group type associated with each feature space
		:param flag_annot: indicator of whether to add the columns 'group1_count', 'group2_count' to the group annotation dataframe
		:param flag_sort: indicator of whether to sort the group annotation dataframe
		:param stat_chi2_correction: indicator of whether to perform correction for chi-squared test
		:param stat_fisher_alternative: hypothesis used in Fisher's exact test
		:param contingency_table_query: indicator of whether to retrieve the contingency tables based on which to compute the peak enrichment in the groups
		:param save_mode: indicator of whether to save data
		:param output_filename: output filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing dataframes of group annotations (including the number of specified peaks and peak enrichment statistics and p-values for each group) for each feature type
				 2. the concatenated dataframe of group annotations for each feature type
		"""
		
		group_vec_query = group_vec
		if len(column_vec_query)==0:
			column_vec_query = ['overlap','freq_obs','freq_expect']
		column_query1 = column_vec_query[0]

		df_overlap_query = df_overlap
		dict_group_query = dict()
		verbose_internal = self.verbose_internal
		# compute the enrichment of peaks in each group for each feature type
		print('df_overlap_query ',df_overlap_query.shape)
		print(df_overlap_query[0:2])
		for group_type in group_vec_query:
			df_group_basic_pre1 = df_overlap_query.groupby(by=[group_type]) # group assignment of peaks using the feature type
			df_group_freq_pre1 = df_group_basic_pre1[column_vec_query].sum()  # number of members in each group using the feature type                                                                                                                                          vpeak number in each group in the given feature space
			if column_query1!=column_query:
				df_group_freq_pre1 = df_group_freq_pre1.rename(columns={column_query1:column_query})
			df_group_freq_compare = dict_group_compare[group_type]
			print('df_group_freq_compare ',df_group_freq_compare.shape)
			print(df_group_freq_compare[0:2])

			# compute peak enrichment in each group in the given feature space
			df_group_freq, dict_contingency_table = self.test_query_enrichment_pre1(df1=df_group_freq_pre1,df2=df_group_freq_compare,
																					column_query=column_query,
																					stat_chi2_correction=stat_chi2_correction,
																					stat_fisher_alternative=stat_fisher_alternative,
																					contingency_table_query=contingency_table_query,
																					verbose=verbose_internal,select_config=select_config)

			df_group_freq['group_type'] = group_type
			dict_group_query.update({group_type:df_group_freq})

			# add the columns 'group1_count', 'group2_count' to the paired group assignment dataframe
			if flag_annot>0:
				group_query = df_overlap_query[group_type]
				df_overlap_query['%s_count'%(group_type)] = np.asarray(df_group_freq.loc[group_query,'count']) # the number of members in each group

		list1 = [dict_group_query[group_type] for group_type in group_vec_query]
		df_query_pre1 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
		if flag_sort>0:
			df_query_pre1 = df_query_pre1.sort_values(by=['group_type','stat_fisher_exact_','pval_fisher_exact_'],ascending=[True,False,True])

		if (save_mode>0) and (output_filename!=''):
			df_query_pre1 = df_query_pre1.round(7)
			df_query_pre1.to_csv(output_filename,sep='\t')
			print('save data: ',output_filename)

		return dict_group_query, df_query_pre1

	## ====================================================
	# compute enrichment of peaks in the paired groups
	def test_query_group_overlap_pre1(self,df_group_1=[],df_group_2=[],df_overlap_1=[],df_query_compare=[],
											feature_query_vec=[],column_query='overlap',column_vec_query=['freq_obs','freq_expect'],
											flag_shuffle=0,stat_chi2_correction=True,stat_fisher_alternative='greater',contingency_table_query=0,verbose=0,select_config={}):
	
		"""
		compute enrichment of observations (peaks) in the paired groups
		:param df_group_1: dataframe containing group assignment of peaks in feature space 1
		:param df_group_2: dataframe containing group assignment of peaks in feature space 2
		:param feature_query_vec: the vector of observation names or indices
		:param column_query: the column corresponding to the number of members in paired groups
		:param column_vec_query: columns corresponding to percentage of group members in the population and the expected percentage for each paired group
		:param flag_shuffle: indicator of whether to shuffle the observations (peaks)
		:param stat_chi2_correction: indicator of whether to perform correction for chi-squared test
		:param stat_fisher_alternative: hypothesis used in Fisher's exact test
		:param contingency_table_query: indicator of whether to retrieve the contingency tables based on which to compute the peak enrichment in the paired groups
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1. dataframe of peak-TF associations with added columns including percentages of paired group assignment in genome-wide peaks and in peaks with predicted TF binding, and peak enrichment in the paired groups;
				 2. dictionary containing contingency tables based on which to compute the peak enrichment in paired groups using Fisher's exact test (if contingency_table_query>0)
				 3. dataframe of the number of shared members in paired groups
		"""

		flag_query1 = 1
		if flag_query1>0:
			group_vec_ori_1 = np.unique(df_group_1[column_query])
			group_vec_ori_2 = np.unique(df_group_2[column_query])
			# query the overlap between groups in the two feature spaces
			df1 = df_group_1
			df2 = df_group_2
			df_overlap = self.test_query_group_overlap_1(df_list=[df1,df2],feature_query_vec=feature_query_vec,select_config=select_config)

			column_1 = 'group1'
			column_2 = 'group2'
			df_overlap[column_1] = np.asarray(df_overlap.index)
			idvec = [column_1]
			# convert wide format dataframe to long format dataframe
			df_query1 = df_overlap.melt(id_vars=idvec,var_name=column_2,value_name='overlap')
			df_query1.index = utility_1.test_query_index(df_query1,column_vec=[column_1,column_2],symbol_vec=['_'])
			
			if flag_shuffle>0:
				np.random.seed(0)
				feature_num = len(feature_query_vec)
				feature_query_1 = np.asarary(df1.index)
				feature_num_1 = len(feature_query_1)
				id1 = np.random.permutation(np.arange(feature_num_1))[0:feature_num]
				feature_query_vec_1 = feature_query_1[id1]	# randomly select the same number of peak loci
				df_overlap_2 = self.test_query_group_overlap_1(df_list=[df1,df2],feature_query_vec=feature_query_vec_1,select_config=select_config)
				df_query2 = df_overlap_2.melt(id_vars=idvec,var_name=column_2,value_name='overlap')

			# query the overlap between between groups in the two feature spaces for genome-wide peak loci
			if len(df_query_compare)==0:
				if len(df_overlap_1)==0:
					df_overlap_1 = self.test_query_group_overlap_1(df_list=[df1,df2],feature_query_vec=[],select_config=select_config)
				df_overlap_1[column_1] = np.asarray(df_overlap_1.index)

				# convert wide format dataframe to long format dataframe
				idvec = [column_1]
				df_query_compare = df_overlap_1.melt(id_vars=idvec,var_name=column_2,value_name='overlap')
			
			df_query_compare.index = utility_1.test_query_index(df_query_compare,column_vec=[column_1,column_2],symbol_vec=['_'])
			query_id_1 = df_query1.index
			if verbose>0:
				print('df_query_compare: ',df_query_compare.shape)
				print(df_query_compare)

			column_query_ori = column_query
			column_query = 'overlap'
			df_query1['%s_ori'%(column_query)] = df_query_compare.loc[query_id_1,column_query]
			count1 = np.sum(df_query1[column_query])
			count2 = np.sum(df_query_compare[column_query])

			eps = 1E-12
			t_value_1 = np.sum(df_query1[column_query])
			t_value_1 = np.max([t_value_1,eps])
			df_freq_query = df_query1[column_query]/t_value_1

			t_value_2 = np.sum(df_query_compare[column_query])
			t_value_2 = np.max([t_value_2,eps])
			df_freq_1 = df_query_compare[column_query]/t_value_2

			list1 = [df_freq_query,df_freq_1]
			for (column_query1,query_value) in zip(column_vec_query,list1):
				print('column: %s'%(column_query1))
				df_query1[column_query1] = query_value

			# compute peak enrichment in the paired groups
			df_query1, dict_contingency_table = self.test_query_enrichment_pre1(df1=df_query1,df2=df_query_compare,
																				column_query=column_query,
																				stat_chi2_correction=stat_chi2_correction,
																				stat_fisher_alternative=stat_fisher_alternative,
																				contingency_table_query=contingency_table_query,
																				verbose=verbose,select_config=select_config)

			return df_query1, dict_contingency_table, df_overlap

	## ====================================================
	# compute enrichment of peaks in the paired groups and in the groups in each feature space
	def test_query_group_overlap_pre2(self,data=[],dict_group_compare=[],df_group_1=[],df_group_2=[],df_overlap_1=[],df_query_compare=[],column_sort=[],
											flag_sort=1,flag_sort_2=1,flag_annot=1,stat_chi2_correction=True,stat_fisher_alternative='greater',contingency_table_query=0,
											save_mode=1,output_filename='',output_filename_2='',verbose=0,select_config={}):
		
		"""
		compute enrichment of peaks in the paired groups and in the groups in each feature space
		:param data: dataframe of annotations including group assignment for the specific peaks (e.g., peaks with predicted TF binding)
		:param dict_group_compare: dictionary containing dataframes of group annotations, including the group size of each group for each feature type
		:param df_group_1: dataframe containing group assignment of peaks in feature space 1
		:param df_group_2: dataframe containing group assignment of peaks in feature space 2
		:param column_sort: the columns used to sort the paired group annotation dataframe
		:param flag_sort: indicator of whether to sort the paired group annotation dataframe by specific columns
		:param flag_sort_2: indicator of whether to sort the group annotation dataframe by specific columns for each feature type
		:param flag_annot: indicator of whether to add the columns 'group1_count', 'group2_count' to the paired group assignment dataframe
		:param stat_chi2_correction: indicator of whether to perform correction for chi-squared test
		:param stat_fisher_alternative: hypothesis used in Fisher's exact test
		:param contingency_table_query: indicator of whether to retrieve the contingency tables based on which to compute the peak enrichment in the paired groups
		:param save_mode: indicator of whether to save data
		:param output_filename: output filename to save data
		:param output_filename_2: the second output filename to save data
		:param verobse: verbosity level to print the intermediate information
		:param select_config: dictionary storing configuration parameters
		:return: 1. dataframe of peak-TF associations with added columns including percentages of paired group assignment in genome-wide peaks and in peaks with predicted TF binding, and peak enrichment in the paired groups;
				 2. dataframe of the number of shared members in paired groups
				 3. dictionary containing dataframes of the updated group annotations for each feature type and the concatenated dataframe of group annotations for different feature types
		"""

		flag_signal_query=1
		flag_query1=1
		if flag_query1>0:
			df_query1 = data
			feature_vec = df_query1.index
			feature_num = len(feature_vec)
			verbose_internal = self.verbose_internal
			# print('feature_vec: ',feature_num)
			column_query1 = 'group'
			# query the overlap between groups for the given peak loci
			# feature_query = df_query1.index
			
			feature_query_1 = df_group_1.index
			feature_query_2 = df_group_2.index

			feature_vec_1 = pd.Index(feature_vec).intersection(feature_query_1,sort=False)
			feature_vec_2 = pd.Index(feature_vec).intersection(feature_query_2,sort=False)
			# print('feature_vec_1, feature_vec_2: ',len(feature_vec_1),len(feature_vec_2))

			df_group_query1 = df_group_1.loc[feature_vec,:]
			df_group_query2 = df_group_2.loc[feature_vec,:]
			if verbose_internal==2:
				annot_str1 = 'group assignment of peak loci'
				annot_str2 = 'group assignment of specified peak loci'
				print('%s using feature type 1, dataframe of size '%(annot_str1),df_group_1.shape)
				print('data preview:\n ',df_group_1[0:2])
				print('%s using feature type 2, dataframe of size '%(annot_str1),df_group_2.shape)
				print('data preview:\n ',df_group_2[0:2])

				print('%s using feature type 1, dataframe of size '%(annot_str2),df_group_query1.shape)
				print('%s using feature type 2, dataframe of size '%(annot_str2),df_group_query2.shape)
			
			# compute the enrichment of specified peaks in the paired groups
			df_overlap_query, dict_contingency_table, df_overlap_mtx = self.test_query_group_overlap_pre1(df_group_1=df_group_query1,df_group_2=df_group_query2,
																											df_overlap_1=df_overlap_1,
																											df_query_compare=df_query_compare,
																											feature_query_vec=feature_vec,
																											column_query=column_query1,
																											flag_shuffle=0,
																											stat_chi2_correction=stat_chi2_correction,
																											stat_fisher_alternative=stat_fisher_alternative,
																											contingency_table_query=contingency_table_query,
																											verbose=verbose,select_config=select_config)

			if flag_sort>0:
				if len(column_sort)==0:
					# column_sort = ['freq_obs','pval_chi2_']
					column_sort = ['freq_obs','pval_fisher_exact_']
				df_overlap_query = df_overlap_query.sort_values(by=column_sort,ascending=[False,True])
				if verbose_internal>0:
					print('paired group assignment of candidate peak loci, dataframe of size ',df_overlap_query.shape)
					print('columns: ',np.asarray(df_overlap_query.columns))
					print('data preview: ')
					print(df_overlap_query[0:5])

				if (save_mode>0) and (output_filename!=''):
					# output_filename = '%s/test_query_df_overlap.%s.%s.signal.1.txt'%(output_file_path,motif_id1,data_file_type_query)
					df_overlap_query.to_csv(output_filename,sep='\t')
					print('save paired group assignment and annotation: ',output_filename)

			flag_group_1=1
			# dict_group_basic_1 = dict()
			if flag_group_1>0:
				# compute the enrichment of specific peaks in each group for each feature type
				# column_query = 'overlap'
				column_query = 'count'
				column_vec_query = ['overlap','freq_obs','freq_expect']
				dict_group_basic_1, df_query_pre2 = self.test_query_group_enrichment_pre1(df_overlap=df_overlap_query,
																							dict_group_compare=dict_group_compare,
																							column_query=column_query,
																							column_vec_query=column_vec_query,
																							flag_annot=flag_annot,
																							flag_sort=flag_sort_2,
																							stat_chi2_correction=stat_chi2_correction,
																							stat_fisher_alternative=stat_fisher_alternative,
																							save_mode=save_mode,output_filename=output_filename_2,
																							verbose=verbose,select_config=select_config)

				dict_group_basic_1.update({'combine':df_query_pre2})

			return df_overlap_query, df_overlap_mtx, dict_group_basic_1

	## ====================================================
	# query enrichment of peaks with detected TF motif in the paired groups
	def test_query_feature_overlap_1(self,data=[],filename_list=[],motif_id_query='',df_overlap_compare=[],overwrite=False,input_file_path='',save_mode=1,verbose=0,select_config={}):

		"""
		compute enrichment of peaks with detected TF motif in the paired groups
		:param data: dataframe of annotations including group assignment for the peaks with motif detected for the given TF
		:param filename_list: (list) names of two files: 1. paired group annotations including the number of shared members in paired groups; 2. group annotations including group size for each feature type
		:param motif_id_query: name of the TF for which to predict peak-TF associations
		:param df_overlap_compare: (dataframe) the number of shared members in paired groups for genome-wide peaks
		:param overwrite: indicator of whether to overwrite the current file
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1. dataframe of peak-TF associations with added columns including percentages of paired group assignment in genome-wide peaks and in the peaks with detected TF motif, and peak enrichment in the paired groups;
				 2. group annotations including enrichment of peaks with detected TF motif in each group for each feature type
				 3. dictionary containing dataframes of group annotations for each feature type and the concatenated dataframe of group annotations for different feature types
				 4. indicator of whether the group or paired group annotations were loaded from files or generated by computation
		"""

		flag_motif_query=1
		data_file_type_query = select_config['data_file_type']
		method_type_group = select_config['method_type_group']
		if flag_motif_query>0:
			filename_save_annot2 = '%s.%s.%s'%(method_type_group,motif_id_query,data_file_type_query)
			filename_prefix_1 = '%s/test_query_df_overlap.%s.motif'%(input_file_path, filename_save_annot2)
			filename_prefix_2 = 'filename_overlap_motif'

			# compute enrichment of peaks in paired groups
			query_vec, select_config = self.test_query_feature_overlap_unit1(data=data,df_overlap_compare=df_overlap_compare,
																				filename_list=filename_list,
																				filename_prefix_1=filename_prefix_1,
																				filename_prefix_2=filename_prefix_2,
																				overwrite=overwrite,
																				verbose=verbose,select_config=select_config)

			df_overlap_query, df_group_basic_query, dict_group_basic_query, load_mode = query_vec

			if load_mode<2:
				self.dict_group_basic_motif = dict_group_basic_query
				column_1 = 'combine'
				if column_1 in dict_group_basic_query:
					df_group_basic_query = dict_group_basic_query[column_1]

			if len(df_group_basic_query)>0:
				self.df_group_basic_motif = df_group_basic_query

			self.df_overlap_motif = df_overlap_query

			return df_overlap_query, df_group_basic_query, dict_group_basic_query, load_mode

	## ====================================================
	# compute enrichment of peaks with predicted TF binding in the paired groups
	def test_query_feature_overlap_2(self,data=[],filename_list=[],motif_id_query='',df_overlap_compare=[],input_file_path='',overwrite=False,save_mode=1,verbose=0,select_config={}):

		"""
		compute enrichment of peaks with predicted TF binding in the paired groups
		:param data: dataframe of annotations including group assignment for the peaks with predicted TF binding for the given TF
		:param filename_list: (list) names of two files: 1. paired group annotations including the number of shared members in paired groups; 2. group annotations including group size for each feature type
		:param motif_id_query: name of the TF for which to predict peak-TF associations
		:param df_overlap_compare: (dataframe) the number of shared members in paired groups for genome-wide peaks
		:param overwrite: indicator of whether to overwrite the current file
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param output_filename: filename to save the data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1. dataframe of peak-TF associations with added columns including percentages of paired group assignment in genome-wide peaks and in the peaks with predicted TF binding, and peak enrichment in the paired groups;
				 2. group annotations including enrichment of peaks with predicted TF binding in each group for each feature type
				 3. dictionary containing dataframes of group annotations for each feature type and the concatenated dataframe of group annotations for different feature types
				 4. indicator of whether the group or paired group annotations were loaded from files or generated by computation
		"""

		flag_select_query=1
		data_file_type_query = select_config['data_file_type']
		method_type_group = select_config['method_type_group']
		if flag_select_query>0:
			method_type_query = select_config['method_type_feature_link']
			filename_save_annot2 = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)
			filename_prefix_1 = '%s/test_query_df_overlap.%s'%(input_file_path, filename_save_annot2)
			filename_prefix_2 = 'filename_overlap'

			# compute enrichment of peaks in paired groups
			query_vec, select_config = self.test_query_feature_overlap_unit1(data=data,df_overlap_compare=df_overlap_compare,
																				filename_list=filename_list,
																				filename_prefix_1=filename_prefix_1,
																				filename_prefix_2=filename_prefix_2,
																				overwrite=overwrite,
																				verbose=verbose,select_config=select_config)

			df_overlap_query, df_group_basic_query, dict_group_basic_query, load_mode = query_vec

			if load_mode<2:
				self.dict_group_basic_2 = dict_group_basic_query
				column_1 = 'combine'
				if column_1 in dict_group_basic_query:
					df_group_basic_query = dict_group_basic_query[column_1]

			if len(df_group_basic_query)>0:
				self.df_group_basic_query_2 = df_group_basic_query

			self.df_overlap_query = df_overlap_query

			return df_overlap_query, df_group_basic_query, dict_group_basic_query, load_mode

	## ====================================================
	# compute enrichment of peaks in paired groups
	def test_query_feature_overlap_unit1(self,data=[],df_overlap_compare=[],filename_list=[],field_query=[],filename_prefix_1='',filename_prefix_2='',method_type_id=0,overwrite=False,verbose=0,select_config={}):

		"""
		compute enrichment of peaks in the paired groups
		:param data: dataframe of annotations including group assignment for the specific peaks (e.g., peaks with predicted TF binding)
		:param df_overlap_compare: (dataframe) the number of shared members in paired groups for genome-wide peaks
		:param filename_list: (list) names of two files: 1. paired group annotations including the number of shared members in paired groups; 2. group annotations including group size for each feature type
		:param field_query: (list) fields in select_config that correspond to the filenames in filename_list
		:param filename_prefix_1: (str) prefix used to define filenames in filename_list
		:param filename_prefix_2: (str) prefix used to define field names in field_query
		:param verbose: verbosity level to print the intermediate information
		:param method_type_id: (int) the method used to compute peak enrichment: 0: Fisher's exact test; 1: chi-squared test;
		:param overwrite: (bool) indicator of whether to overwrite the current file
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. tuplet containing the following data:
					1.1 dataframe of peak-TF associations with added columns including percentages of paired group assignment in genome-wide peaks and in the specific type of peaks, and peak enrichment in the paired groups;
				 	1.2 group annotations including enrichment of the specific peaks in each group for each feature type
				 	1.3 dictionary containing dataframes of group annotations for each feature type and the concatenated dataframe of group annotations for different feature types
					1.4 indicator of whether the group or paired group annotations were loaded from files or generated by computation
				 2. dictionary containing parameters
		"""

		feature_type_num = 2
		feature_type_idvec = np.arange(1,feature_type_num+1)
		extension = 'txt'
		if len(field_query)==0:
			field_query = ['%s_%d'%(filename_prefix_2,feature_type_id) for feature_type_id in feature_type_idvec]
		if len(filename_list)==0:
			filename_list = ['%s.%d.%s'%(filename_prefix_1,feature_type_id,extension) for feature_type_id in feature_type_idvec]
			select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=filename_list,overwrite=overwrite,select_config=select_config)
			filename_list = param_vec

		filename_query, filename_query_2 = filename_list[0:2]
		input_filename, input_filename_2 = filename_query, filename_query_2

		stat_chi2_correction = True
		stat_fisher_alternative = 'greater'
		contingency_table_query = 0

		field_query = ['stat_chi2_correction','stat_fisher_alternative','contingency_table_query']
		list1 = [stat_chi2_correction,stat_fisher_alternative,contingency_table_query]
		select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=list1,overwrite=False,select_config=select_config)
		stat_chi2_correction, stat_fisher_alternative, contingency_table_query = param_vec[0:3]
		verbose_internal = self.verbose_internal
		if verbose_internal==2:
			for (field_id,query_value) in zip(field_query,param_vec):
				print('%s: %s'%(field_id,query_value))

		df_group_basic_query = []
		df_overlap_query = []
		dict_group_basic_query = dict()
		load_mode = 0
		if os.path.exists(input_filename)==True:
			if (overwrite==False):
				df_overlap_query = pd.read_csv(input_filename,index_col=0,sep='\t')
				load_mode = load_mode+1
		else:
			print('the file does not exist: %s'%(input_filename))

		if os.path.exists(input_filename_2)==True:
			if (overwrite==False):
				df_group_basic_query = pd.read_csv(input_filename_2,index_col=0,sep='\t')
				load_mode = load_mode+1
		else:
			print('the file does not exist: %s'%(input_filename_2))

		if load_mode<2:
			dict_group_basic_1 = self.dict_group_basic_1
			df_group_1 = self.df_group_pre1
			df_group_2 = self.df_group_pre2

			if len(df_overlap_compare)==0:
				df_overlap_compare = self.df_overlap_compare

			output_filename = filename_query
			output_filename_2 = filename_query_2
			if method_type_id==0:
				column_sort = ['freq_obs','pval_fisher_exact_']
			else:
				column_sort = ['freq_obs','pval_chi2_']

			df_query1 = data
			# compute enrichment of peaks in the paired groups
			df_overlap_query, df_overlap_mtx_query, dict_group_basic_query = self.test_query_group_overlap_pre2(data=df_query1,dict_group_compare=dict_group_basic_1,
																													df_group_1=df_group_1,
																													df_group_2=df_group_2,
																													df_overlap_1=[],
																													df_query_compare=df_overlap_compare,
																													column_sort=column_sort,
																													flag_sort=1,
																													flag_sort_2=1,
																													flag_annot=1,
																													stat_chi2_correction=stat_chi2_correction,
																													stat_fisher_alternative=stat_fisher_alternative,
																													contingency_table_query=contingency_table_query,
																													save_mode=1,output_filename=output_filename,output_filename_2=output_filename_2,
																													verbose=verbose,select_config=select_config)

		return (df_overlap_query, df_group_basic_query, dict_group_basic_query, load_mode), select_config

	## ====================================================
	# compute feature embeddings of observations
	def test_query_feature_embedding_pre1(self,feature_type_vec=[],method_type='',n_components=50,flag_config=0,flag_motif_data_load=1,flag_load_1=1,overwrite=False,input_file_path='',
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		compute feature embeddings of observations (peak loci)
		:param feature_type_vec: (array or list) feature types of feature representations of the observations
		:param method_type: the method used to predict peak-TF associations initially
		:param n_components: the nubmer of latent components used in feature dimension reduction
		:param flag_config: indicator of whether to query configuration parameters
		:param flag_motif_data_load: indicator of whether to query motif scanning data
		:param flag_load_1: indicator of whether to query peak accessibility and gene expression data
		:param overwrite: (bool) indicator of whether to overwrite the current data
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing feature embeddings of observations for each feature type
				 2. dictionary containing updated parameters
		"""

		data_file_type_query = select_config['data_file_type']
		flag_motif_data_load_1 = flag_motif_data_load
		
		method_type_query = method_type
		if method_type=='':
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_query = method_type_feature_link

		# load motif data, RNA-seq and ATAC-seq data
		method_type_vec_query = [method_type_query]
		select_config = self.test_query_load_pre1(method_type_vec=method_type_vec_query,
													flag_motif_data_load_1=flag_motif_data_load_1,
													flag_load_1=flag_load_1,
													save_mode=save_mode,verbose=verbose,select_config=select_config)

		dict_motif_data = self.dict_motif_data
		verbose_internal = self.verbose_internal
		key_vec = list(dict_motif_data.keys())
		if verbose_internal==2:
			print('annotation of motif scanning data ',key_vec)
			print(dict_motif_data)

		peak_read = self.peak_read  # peak accessibility matrix (normalized and log-transformed) 
		peak_loc_ori = peak_read.columns
		
		rna_exprs = self.meta_exps_2  # gene expression matrix (normalized and log-transformed)
		feature_type_vec = ['peak_tf','peak_motif','peak_motif_ori']
		
		# query motif scanning data and motif scores of given peak loci
		# query TFs with motifs and expressions
		motif_data, motif_data_score, motif_query_vec_1 = self.test_query_motif_data_annotation_1(data=dict_motif_data,
																									data_file_type=data_file_type_query,
																									gene_query_vec=[],
																									feature_query_vec=peak_loc_ori,
																									method_type=method_type_query,
																									peak_read=peak_read,
																									rna_exprs=rna_exprs,
																									save_mode=save_mode,
																									verbose=verbose,select_config=select_config)

		method_type_dimension = select_config['method_type_dimension']
		feature_type_num1 = len(feature_type_vec)
		num1 = feature_type_num1
		method_type_vec_dimension = [method_type_dimension]*num1
		
		output_file_path_default = output_file_path
		
		column_1 = 'file_path_group_query'
		feature_mode = 1
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
		print('directory to save feature embedding data: %s'%(file_path_group_query))

		feature_mode = select_config['feature_mode']
		filename_prefix_save = '%s.pre%d'%(data_file_type_query,feature_mode)
		filename_save_annot = '1'
		feature_query_vec = peak_loc_ori
		motif_data = motif_data.astype(np.float32)
		output_file_path_2 = file_path_group_query
		load_mode = 0

		# compute feature embedding
		# dict_query1: {feature_type:latent representation matrix, feature_type:component matrix}
		dict_query1 = self.test_query_feature_mtx_1(feature_query_vec=feature_query_vec,
														feature_type_vec=feature_type_vec,
														gene_query_vec=motif_query_vec_1,
														method_type_vec_dimension=method_type_vec_dimension,
														n_components=n_components,
														motif_data=motif_data,
														motif_data_score=motif_data_score,
														peak_read=peak_read,
														rna_exprs=rna_exprs,
														load_mode=load_mode,
														input_file_path=input_file_path,
														save_mode=save_mode,
														output_file_path=output_file_path_2,
														filename_prefix_save=filename_prefix_save,
														filename_save_annot=filename_save_annot,
														verbose=verbose,select_config=select_config)
		self.select_config = select_config
		return dict_query1, select_config

	## ====================================================
	# query computed embeddings of observations
	def test_query_feature_embedding_load_pre1(self,dict_file={},feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=100,n_component_sel=50,reconstruct=1,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query computed embeddings of observations
		:param dict_file: dictionary containing filenames of the computed feature embeddings
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param method_type_vec: (array or list) methods used for feature dimension reduction for each feature type
		:param method_type_dimension: the method used for feature dimension reduction
		:param n_components: the number of latent components used in dimension reduction
		:param n_component_sel: the number of latent components used in the low-dimensional feature embedding of observations
		:param reconstruct: indicator of whether to compute the reconstructed feature matrix from the latent representation matrix (embedding) and the loading matrix (component matrix)
		:param input_file_path: the directory where the feature embeddings are saved
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: dictionary containing the latent representation matrix (embedding), loading matrix, and reconstructed marix (if reconstruct>0)
		"""

		load_mode = 1  # load data from the paths of files in dict_file
		if len(dict_file)==0:
			load_mode = 0
	
		n_components_query = n_components
		# the number of components used in embedding is equal to or smaller than the number of components used in feature dimension reduction
		column_1 = 'n_component_sel'
		if n_component_sel<0:
			if column_1 in select_config:
				n_component_sel = select_config[column_1]
			else:
				n_component_sel = n_components
		
		type_query = 0
		if (n_component_sel!=n_components):
			type_query = 1
		
		if input_file_path=='':
			input_file_path = select_config['file_path_group_query'] # the directory where the feature embeddings are saved;
		
		input_file_path_query = input_file_path
		filename_prefix_1 = filename_prefix_save
		feature_type_num = len(feature_type_vec)
		if len(method_type_vec)==0:
			method_type_vec = [method_type_dimension]*feature_type_num

		dict_query1 = dict()
		for i1 in range(feature_type_num):
			feature_type_query = feature_type_vec[i1]
			method_type_dimension_query = method_type_vec[i1]
			if filename_save_annot=='':
				filename_save_annot_2 = '%s_%d.1'%(method_type_dimension_query,n_components_query)
			else:
				filename_save_annot_2 = filename_save_annot

			if load_mode==0:
				input_filename_1 = '%s/%s.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2) # use the default input filename
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

			dict_query1[feature_type_query].update({'df_latent':df_latent_query,'df_component':df_component_query})

			if reconstruct>0:
				reconstruct_mtx = df_latent_query.dot(df_component_query.T)
				dict_query1[feature_type_query].update({'reconstruct_mtx':reconstruct_mtx})

		return dict_query1

	## ====================================================
	# query computed low-dimensional embeddings of observations
	def test_query_feature_embedding_load_1(self,dict_file={},feature_query_vec=[],feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,n_component_sel=50,
												reconstruct=1,flag_combine=1,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query computed low-dimensional embeddings of observations
		:param dict_file: dictionary containing paths of files which saved the computed feature embeddings
		:param feature_query_vec: (array or list) the observations for which to compute feature embeddings; if not specified, all observations in the latent representation matrix are included
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param method_type_vec: (array or list) methods used for feature dimension reduction for each feature type
		:param method_type_dimension: the method used for feature dimension reduction
		:param n_components: the number of latent components used in feature dimension reduction
		:param n_component_sel: the number of latent components used in the low-dimensional feature embedding of observations
		:param reconstruct: indicator of whether to compute the reconstructed feature matrix from the latent representation matrix (embedding) and the loading matrix (component matrix)
		:param flag_combine: indicator of whether to concatenate feature embeddings of different feature types
		:param input_file_path: the directory where the feature embeddings are saved
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: dictionary containing the embeddings of observations for each feature type and the concatenated embedding (if flag_combine>0)
		"""

		column_1 = 'n_component_sel'
		if n_component_sel<0:
			if column_1 in select_config:
				n_component_sel = select_config[column_1]
			else:
				n_component_sel = n_components

		# query the embeddings with specific number of dimensions, loading matrix, and reconstructed matrix (optional)
		dict_query_1 = self.test_query_feature_embedding_load_pre1(dict_file=dict_file,
																	feature_type_vec=feature_type_vec,
																	method_type_vec=method_type_vec,
																	method_type_dimension=method_type_dimension,
																	n_components=n_components,
																	n_component_sel=n_component_sel,
																	reconstruct=reconstruct,
																	input_file_path=input_file_path,
																	save_mode=save_mode,output_file_path=output_file_path,output_filename=output_filename,filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,verbose=verbose,select_config=select_config)

		if save_mode>0:
			self.dict_latent_query_1 = dict_query_1
		
		flag_query1 = 1
		if flag_query1>0:
			feature_type_vec_query = ['latent_%s'%(feature_type_query) for feature_type_query in feature_type_vec]
			feature_type_num = len(feature_type_vec)

			query_mode = 0
			if len(feature_query_vec)>0:
				query_mode = 1  # query embeddings of the given observations

			list_1 = []
			annot_str_vec = ['peak accessibility','peak motif matrix']
			verbose_internal = self.verbose_internal
			for i1 in range(feature_type_num):
				feature_type_query = feature_type_vec[i1]
				df_query = dict_query_1[feature_type_query]['df_latent'] # load the latent representation matrix
				if query_mode>0:
					df_query = df_query.loc[feature_query_vec,:]
				else:
					if i1==0:
						feature_query_1 = df_query.index
					else:
						df_query = df_query.loc[feature_query_1,:]

				column_vec = df_query.columns
				df_query.columns = ['%s.%s'%(column_1,feature_type_query) for column_1 in column_vec]
				if verbose_internal>0:
					if feature_type_query in ['peak_tf','peak_mtx']:
						annot_str1 = 'peak accessibility'
					elif feature_type_query in ['peak_motif','peak_motif_ori']:
						annot_str1 = 'peak-motif matrix'
					print('feature embeddings of %s, dataframe of size '%(annot_str1),df_query.shape)
				list_1.append(df_query)

			dict_query1 = dict(zip(feature_type_vec_query,list_1))

			if (feature_type_num>1) and (flag_combine>0):
				list1 = [dict_query1[feature_type_query] for feature_type_query in feature_type_vec_query[0:2]]
				latent_mtx_combine = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				if verbose_internal>0:
					print('combined feature embeddings, dataframe of size ',latent_mtx_combine.shape)
					print('data preview: ')
					print(latent_mtx_combine[0:2])

				feature_type_query1,feature_type_query2 = feature_type_vec[0:2]
				feature_type_combine = 'latent_%s_%s_combine'%(feature_type_query1,feature_type_query2)
				select_config.update({'feature_type_combine':feature_type_combine})
				dict_query1.update({feature_type_combine:latent_mtx_combine})

			self.select_config = select_config

			return dict_query1

	## ====================================================
	# query neighbors of observations in the feature space
	def test_query_feature_neighbor_pre1(self,data=[],n_neighbors=100,return_distance=True,verbose=0,select_config={}):

		"""
		query neighbors of observations in the feature space
		:param data: feature matrix of the observations (row:observation,column:feature)
		:param n_neighbors: the number of nearest neighbors to search for each observation
		:param return_distance: indicator of which to compute distance between an observation and the neighbors
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) estimated K nearest neighbors of each observation (row:observation,column:neighbor);
				 2. (dataframe) distances between each observation and the identified neighbors (row:observation,column:neighbor);
				 columns in dataframe 1 and 2 are in ascending order of the distance between the observation and a neighbor;
		"""

		from sklearn.neighbors import NearestNeighbors
		from scipy.stats import poisson, multinomial

		# search for K nearest neighbors of each observation
		# nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)
		nbrs = NearestNeighbors(n_neighbors=n_neighbors,radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2)
		
		feature_mtx = data  # feature matrix of observations (row:observation,column:feature)
		nbrs.fit(feature_mtx)
		query_id = feature_mtx.index  # observation name
		neighbor_dist, neighbor_id = nbrs.kneighbors(feature_mtx)
		column_vec = ['neighbor%d'%(id1) for id1 in np.arange(n_neighbors)]
		feature_nbrs = pd.DataFrame(index=query_id,columns=column_vec,data=query_id.values[neighbor_id])
		dist_nbrs = []
		if return_distance>0:
			# compute distance between an observation and the neighbors
			dist_nbrs = pd.DataFrame(index=query_id,columns=column_vec,data=neighbor_dist)

		return feature_nbrs, dist_nbrs

	## ====================================================
	# query neighbors of observations in each feature space
	def test_query_feature_neighbor_load_1(self,dict_feature=[],feature_type_vec=[],n_neighbors=100,load_mode=1,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query neighbors of observations in each feature space
		:param dict_feature: dictionary containing feature matrix of observations for each feature type
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param feature_query_vec: (array) observations; if not specified, all the observations in the dataframe of group assignment will be included
		:param load_mode: indicator of whether to load the estimated neighbors and distance matrices from the current files or search for the neighbors by computation
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_filename: filename to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: list containing a list with two dataframes (row:observation,column:neighbor) for each feature type:
				 1. estimated K nearest neighbors of each observation;
				 2. distances between each observation and the identified neighbors;
		"""

		data_file_type_query = select_config['data_file_type']
		if 'neighbor_num' in select_config:
			n_neighbors = select_config['neighbor_num'] # the number of nearest neighbors to search for each observation
			
		n_neighbors_query = n_neighbors+1 # the observation itself is included as the first neighbor by the algorithm used to find neighbors
		flag_neighbor_query = 1
		if flag_neighbor_query>0:
			feature_type_num = len(feature_type_vec)
			filename_annot1 = '%d.%s'%(n_neighbors,data_file_type_query)
			field_query = ['feature','dist']
			list_query1 = []
			if load_mode>0:
				annot_str_vec = ['the neighors of observations','the distance matrix']
				for i1 in range(feature_type_num):
					feature_type_query = feature_type_vec[i1]
					list1 = []
					for (field_id,annot_str1) in zip(field_query,annot_str_vec):
						input_filename = '%s/test_%s_nbrs_%d.%s.txt'%(input_file_path,field_id,i1+1,filename_annot1)
						if os.path.exists(input_filename)==True:
							df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
							print('%s, dataframe of size '%(annot_str1),df_query.shape)
							print('feature type: %s'%(feature_type_query))
							print('data loaded from %s'%(input_filename))
							list1.append(df_query)
						else:
							load_mode = 0
					list_query1.append(list1)
			else:
				for i2 in range(feature_type_num):
					feature_type_query = feature_type_vec[i2]		
					print('find nearest neighbors of peak loci, neighbor number: %d'%(n_neighbors))
					start = time.time()
					df_feature_query = dict_feature[feature_type_query]  # feature matrix of observations (row:observation,column:feature)
					
					print('feature matrix for %s, dataframe of size '%(feature_type_query),df_feature_query.shape)
					feature_nbrs_query, dist_nbrs_query = self.test_query_feature_neighbor_pre1(data=df_feature_query,n_neighbors=n_neighbors_query,return_distance=True,verbose=0,select_config=select_config)

					stop = time.time()
					print('find nearest neighbors of peak loci using feature %s used %.2fs'%(feature_type_query,stop-start))
					list_query1.append([feature_nbrs_query, dist_nbrs_query])

					output_filename_1 = '%s/test_feature_nbrs_%d.%s.txt'%(output_file_path,i2+1,filename_annot1)
					feature_nbrs_query.to_csv(output_filename_1,sep='\t')
					print('estimated nearest neighbors using feature %s, dataframe of size '%(feature_type_query),feature_nbrs_query.shape)
					print('save data: %s'%(output_filename_1))

					output_filename_2 = '%s/test_dist_nbrs_%d.%s.txt'%(output_file_path,i2+1,filename_annot1)
					dist_nbrs_query = dist_nbrs_query.round(7)
					dist_nbrs_query.to_csv(output_filename_2,sep='\t')
					print('distances between nearest neighbors and the given peak loci using feature %s, dataframe of size '%(feature_type_query),dist_nbrs_query.shape)
					print('save data: %s'%(output_filename_2))

			return list_query1

	## ====================================================
	# initial TF binding estimation based on feature group assignment and neighbors in the feature space
	def test_query_feature_group_neighbor_pre1(self,data=[],dict_group=[],dict_neighbor=[],group_type_vec=['group1','group2'],feature_type_vec=[],group_vec_query=[],column_vec_query=[],n_neighbors=30,verbose=0,select_config={}):

		"""
		initial TF binding estimation based on feature group assignment and neighbors in the feature space
		:param data: (dataframe) peak annotations
		:param dict_group: the dictionary containing the group assignments of peaks in the feature spaces
		:param dict_neighbor: the dictonary containing the neighbors of peaks and the distance matrices in the feature spaces
		:param group_type_vec: (list) group name for different feature types
		:param feature_type_vec: vector of feature types
		:param group_vec_query: selected paired groups
		:param column_vec_query: (array or list) the columns to add in the dataframe of peak-TF associations which contain neighbor annotation
		:param n_neighbors: (int) the number of nearest neighbors of a peak locus
		:param verbose: verbosity level to print intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: dataframe of peak annotations with columns representing group and paried group assignment, sharing group labels with peaks with predicted TF binding, and neighbor information
		"""

		df_pre1 = data
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
			column_pred1 = select_config['column_pred1']
			column_pred2 = column_pred1  # representing predicted peak-TF association label (binary)
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
				
				peak_query_pre1 = peak_loc_ori[id1]	# peaks with the paired group assignment	
				peak_query_pre1_1 = peak_loc_ori[(id1&id2)]	# peaks in the paired group and with predicted TF binding
				peak_query_pre1_2 = peak_loc_ori[id1&(~id2)]	# peaks in the paired group and without predicted TF binding
				# df_pre1.loc[peak_query_pre1_2,column_pred_2] = 1
				df_pre1.loc[peak_query_pre1,column_pred_2] = 1

				flag_neighbor_pre2 = 1
				if flag_neighbor_pre2>0:
					# start = time.time()
					peak_neighbor_1 = np.ravel(feature_nbrs_1.loc[peak_query_pre1_1,column_neighbor]) # 0.25s
					peak_neighbor_1 = pd.Index(peak_neighbor_1).unique()
					peak_query_1 = pd.Index(peak_neighbor_1).intersection(peak_query_pre1_2,sort=False)

					peak_neighbor_2 = np.ravel(feature_nbrs_2.loc[peak_query_pre1_1,column_neighbor])
					peak_neighbor_2 = pd.Index(peak_neighbor_2).unique()
					peak_query_2 = pd.Index(peak_neighbor_2).intersection(peak_query_pre1_2,sort=False)
					
					df_pre1.loc[peak_neighbor_1,column_query1] = 1
					df_pre1.loc[peak_neighbor_2,column_query2] = 1

					df_pre1.loc[peak_query_1,column_1] = 1
					df_pre1.loc[peak_query_2,column_2] = 1

					# stop = time.time()
					# print('query neighbors of peak loci within paired groups',group_id_1,group_id_2,stop-start)

					# start = time.time()
					peak_query_vec_2 = pd.Index(peak_query_1).intersection(peak_query_2,sort=False)
					df_pre1.loc[peak_query_vec_2,column_pred_3] = 1

					peak_query_vec_3 = pd.Index(peak_query_1).union(peak_query_2,sort=False)
					df_pre1.loc[peak_query_vec_3,column_pred_5] = 1

					# peaks within the same groups with the selected peak in the two feature space and peaks are neighbors of the selected peak
					# df_pre1[column_pred_5] = ((df_pre1[column_pred_2]>0)&((df_pre1[column_1]>0)|(df_pre1[column_2]>0))).astype(int)

					# stop = time.time()
					# print('query neighbors of peak loci',group_id_1,group_id_2,stop-start)

					if flag_neighbor_2>0:
						peak_neighbor_pre2 = pd.Index(peak_neighbor_1).intersection(peak_neighbor_2,sort=False)
						df_pre1.loc[peak_neighbor_pre2,column_pred_6] = 1

				stop = time.time()
				if (group_id_1%10==0) and (group_id_2%10==0):
					print('query neighbors of peak loci',group_id_1,group_id_2,stop-start)
				
		flag_neighbor_2 = 1  # query neighbor of selected peak
		if (flag_neighbor_2>0):
			list_feature = [feature_nbrs_1,feature_nbrs_2]
			column_vec_query = [column_query1,column_query2,column_pred_7,column_pred_8]
			df_pre1 = self.test_query_feature_neighbor_pre2(data=df_pre1,column_query=column_pred2,
																	list_feature=list_feature,
																	column_neighbor=column_neighbor,
																	column_vec_query=column_vec_query,
																	verbose=verbose,select_config=select_config)

		return df_pre1

	## ====================================================
	# query neighbors of observations in the feature spaces
	def test_query_feature_neighbor_pre2(self,data=[],column_query='',list_feature=[],column_neighbor='',column_vec_query=[],verbose=0,select_config={}):

		"""
		query neighbors of observations in the feature spaces
		:param data: (dataframe) peak annotations
		:param column_query: column corresponding to peak-TF association label predicted by the first method for the specific TF
		:param list_feature: (list) dataframes containing K-nearest neighbors of each peak in feature space 1 and 2
		:param column_neighbor: (list) columns corresponding to the specific number of neighbors to query for the given peak
		:param column_vec_query: (list) columns to add in the dataframe of peak-TF pairs which contain neighbor annotation
		:param verbose: verbosity level to print intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: dataframe of peak-TF pairs with added neighbor information in feature space 1 and 2
		"""

		flag_neighbor = 1  # query neighbor of selected peak
		if (flag_neighbor>0):
			df_pre1 = data
			peak_loc_ori = df_pre1.index

			column_pred2 = column_query
			id_query1 = (df_pre1[column_pred2]>0)
			peak_query_pre1_1 = peak_loc_ori[id_query1] # selected peak loci
			
			if len(list_feature)==0:
				# K-nearest neighbors of each peak in feature space 1 and 2
				feature_nbrs_1 = self.feature_nbrs_1
				feature_nbrs_2 = self.feature_nbrs_2
			else:
				feature_nbrs_1, feature_nbrs_2 = list_feature[0:2]

			peak_neighbor_1 = np.ravel(feature_nbrs_1.loc[peak_query_pre1_1,column_neighbor])
			peak_neighbor_1 = pd.Index(peak_neighbor_1).unique()
			peak_neighbor_num1 = len(peak_neighbor_1)

			peak_neighbor_2 = np.ravel(feature_nbrs_2.loc[peak_query_pre1_1,column_neighbor])
			peak_neighbor_2 = pd.Index(peak_neighbor_2).unique()
			peak_neighbor_num2 = len(peak_neighbor_2)

			column_query1, column_query2, column_pred_neighbor_2, column_pred_neighbor_1 = column_vec_query
			df_pre1.loc[peak_neighbor_1,column_query1] = 1  # neighbors of the selected peaks in feature space 1
			df_pre1.loc[peak_neighbor_2,column_query2] = 1  # neighbors of the selected peaks in feature space 2
			df_pre1.loc[:,[column_query1,column_query2,column_pred_neighbor_2,column_pred_neighbor_1]] = 0

			peak_num1 = len(peak_query_pre1_1)
			print('selected peak loci: %d, neighbors in feature space 1: %d, neighbors in feature space 2: %d'%(peak_num1,peak_neighbor_num1,peak_neighbor_num2))
			interval_1 = 1000 	# the interval of peak number to print the intermediate information
			for i2 in range(peak_num1):
				peak_query = peak_query_pre1_1[i2]
				peak_neighbor_query1 = np.ravel(feature_nbrs_1.loc[peak_query,column_neighbor])
				peak_neighbor_query2 = np.ravel(feature_nbrs_2.loc[peak_query,column_neighbor])
						
				peak_neighbor_pre2_1 = pd.Index(peak_neighbor_query1).intersection(peak_neighbor_query2,sort=False)
				# peak_neighbor_pre2_2 = pd.Index(peak_neighbor_query1).union(peak_neighbor_query2,sort=False)
				if i2%interval_1==0:
					peak_neighbor_num_1 = len(peak_neighbor_pre2_1)
					print('neighbors of the peak in two feature spaces: ',peak_neighbor_num_1,i2,peak_query)
					# peak_neighbor_num_2 = len(peak_neighbor_pre2_2)
					# print('neighbors of the peak in at least one feature space: ',peak_neighbor_num_2,i2,peak_query)
						
				df_pre1.loc[peak_neighbor_query1,column_query1] += 1
				df_pre1.loc[peak_neighbor_query2,column_query2] += 1

				df_pre1.loc[peak_neighbor_pre2_1,column_pred_neighbor_2] += 1
				# df_pre1.loc[peak_neighbor_pre2_2,column_pred_neighbor_1] += 1

			df_pre1[column_pred_neighbor_1] = df_pre1[column_query1]+df_pre1[column_query2]-df_pre1[column_pred_neighbor_2]

			return df_pre1

	## ====================================================
	# add score annotation of feature associations
	def test_query_feature_annot_score_1(self,data=[],df_annot=[],feature_query='',column_idvec=[],column_score='',column_name_vec=[],thresh_vec=[],ascending=False,flag_unduplicate=1,flag_sort=0,flag_binary=1,verbose=0,select_config={}):

		"""
		add score annotation of feature associations
		:param data: (dataframe) peak annotations
		:param df_annot: (dataframe) annotations of peak-TF links including estimated peak-TF association scores
		:param feature_query: feature name, for example, TF name (with binding motif)
		:param column_idvec: columns corresponding to specific feature entities (peak,gene,TF) 
		:param column_score: column of estimated peak-TF association score
		:param column_name_vec: columns to add which represent the binary feature associations and the peak-TF association scores
		:param thresh_vec: thresholds for selecting feature associations
		:param ascending: the order used for sorting the dataframe of feature assocations and the association scores
		:param flag_unduplicate: indicator of whehter to keep unduplicated feature associations
		:param flag_binary: indicator of whether to select binary feature associations using threshold on the association scores
		:param flag_sort: indicator of whether to sort the feature association dataframe by the specific association score
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing the configuration parameters
		:return: dataframe of feature associations with added columns of specific association scores and selected associations
		"""

		df_pre1 = data
		if len(column_idvec)==0:
			column_idvec = ['gene_id','peak_id','motif_id']
		column_id1, column_id2, column_id3 = column_idvec[0:3]

		flag1 = df_annot.duplicated(subset=[column_id2,column_id3])
		if np.sum(flag1)>0:
			flag_unduplicate = 1

		if flag_sort>0:
			df_annot = df_annot.sort_values(by=column_score,ascending=ascending)

		if flag_unduplicate>0:
			df_annot = df_annot.drop_duplicates(subset=[column_id2,column_id3])

		# df_annot1 = df_annot.loc[df_annot[column_id3]==motif_id,:]
		df_annot1 = df_annot.loc[df_annot[column_id3]==feature_query,:]
		df_annot1.index = np.asarray(df_annot1[column_id2])
		
		peak_loc_1 = df_pre1.index
		query_id_1 = df_annot1[column_id2]
		query_id_2 = pd.Index(query_id_1).intersection(peak_loc_1,sort=False)  # find the intersection of the peak loci
		# if verbose>0:
		# 	print('peak_loc_1, query_id_1, query_id_2: ',len(peak_loc_1),len(query_id_1),len(query_id_2),motif_id)

		column_name_1, column_name_2 = column_name_vec[0:2]
		df_pre1.loc[query_id_2,column_name_2] = df_annot1.loc[query_id_2,column_score]

		# add column of binary peak-TF link label by selecting links using threshold on the association score
		if flag_binary>0:
			thresh_score_1, thresh_type = thresh_vec[0:2]
			df_query = df_pre1.loc[query_id_2,:]

			column_query = column_name_2
			if thresh_type in [0,1]:
				if thresh_type==0:
					id1 = (df_query[column_query]>thresh_score_1)	# score above threshold
				else:
					id1 = (df_query[column_query]<thresh_score_1)	# score below threshold
				query_idvec = query_id_2[id1]
			else:
				query_idvec = query_id_2

			df_pre1.loc[query_idvec,column_name_1] = 1
			query_num1 = len(query_idvec)
			if verbose>1:
				print('selected feature assocations: %d, %s'%(query_num1,feature_query))
		
		return df_pre1

	## ====================================================
	# compute peak accessibility-TF expression correlation
	def test_query_compare_peak_tf_corr_1(self,data=[],motif_id_query='',column_value='',motif_data=[],peak_read=[],rna_exprs=[],input_file_path='',
											save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',output_filename='',verbose=0,select_config={}):

		"""
		compute peak accessibility-TF expression correlation
		:param data: (dataframe) peak annotations
		:param motif_id_query: name of the TF for which we perform TF binding prediction
		:param column_value: the column of peak accessibility-TF expression correlation
		:param motif data: motif scanning data matrix (row:peak, column:TF (with binding motif))
		:param peak_read: (dataframe) peak accessibility matrix (row:metacell, column:peak)
		:param rna_exprs: (dataframe) gene expression matrix (row:metacell, column:gene)
		:param input_file_path: the directory to retreive the previously estimated peak accessibility-TF expression correlation data
		:param save_mode: whether to save estimated feature correlation data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param output_filename: filename to save the estimated feature correlation data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing the parameters
		:return: dataframe of peak annotations including peak accessibility-TF expression correlations and corrected p-values for the given TF
		"""

		flag_query1 = 1
		if flag_query1>0:
			df_pre1 = data
			# query peak accessibility-TF expression correlation
			# flag_peak_tf_corr = 0
			flag_peak_tf_corr = 1
			if column_value=='':
				column_value = 'peak_tf_corr'
			column_peak_tf_corr = column_value

			column_query = column_peak_tf_corr
			if column_query in df_pre1.columns:
				flag_peak_tf_corr = 0

			verbose_internal = self.verbose_internal
			if flag_peak_tf_corr>0:
				input_filename_list1 = []
				motif_query_vec = [motif_id_query]
				peak_query_vec = peak_read.columns
				# initialize the peak-TF association matrix for computing correlation between TF expression and accessibility of genome-wide peak loci
				motif_data_query = pd.DataFrame(index=peak_query_vec,columns=motif_query_vec,data=1)
						
				correlation_type = 'spearmanr'  # compute Spearman correlation by default
				column_1 = 'correlation_type'
				if column_1 in select_config:
					correlation_type = select_config[column_1]
				alpha = 0.05  # FDR threshold used in p-value correction
				method_type_correction = 'fdr_bh' # p-value correction method
				filename_prefix = filename_prefix_save
				flag_load_1 = 1
				save_mode_2 = 1
				field_load = [correlation_type,'pval','pval_corrected']

				from .test_reunion_correlation_1 import _Base2_correlation
				file_path1 = self.save_path_1
				test_estimator_correlation = _Base2_correlation(file_path=file_path1)
				# compute peak accessibility-TF expression correlation
				dict_peak_tf_corr_ = test_estimator_correlation.test_peak_tf_correlation_query_1(motif_data=motif_data_query,
																									peak_query_vec=[],
																									motif_query_vec=motif_query_vec,
																									peak_read=peak_read,
																									rna_exprs=rna_exprs,
																									correlation_type=correlation_type,
																									pval_correction=1,
																									alpha=alpha,
																									method_type_correction=method_type_correction,
																									flag_load=flag_load_1,
																									field_load=field_load,
																									parallel_mode=0,
																									save_mode=save_mode_2,
																									input_file_path=input_file_path,
																									input_filename_list=input_filename_list1,
																									output_file_path=output_file_path,
																									filename_prefix=filename_prefix,
																									select_config=select_config)

				field_load = [correlation_type,'pval','pval_corrected']
				column_vec_query = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected']
				annot_str_vec = ['peak accessibility-TF expression correlation','p-value','p-value corrected']
				for i1 in [0,2]:
					column_query, field_id_query = column_vec_query[i1], field_load[i1]
					df_query1 = dict_peak_tf_corr_[field_id_query]
					# add the columns of peak-TF correlation and p-value corrected to the dataframe of peak-TF associations
					df_pre1.loc[peak_query_vec,column_query] = df_query1.loc[peak_query_vec,motif_id_query]
					annot_str_1 = annot_str_vec[i1]
					if verbose_internal==2:
						print('%s, dataframe of size ',annot_str_1,df_query1.shape)
						print('preview:\n ',df_query1[0:2])

				if (save_mode>0) and (output_filename!=''):
					if os.path.exists(output_filename)==True:
						print('the file exists: %s'%(output_filename))
					df_pre1.to_csv(output_filename,sep='\t')

			# return df_pre1, df_annot_peak_tf
			return df_pre1

	## ====================================================
	# model training for peak-TF association prediction
	def test_query_compare_binding_compute_2(self,data=[],dict_feature=[],feature_type_vec=[],method_type_vec=[],
												method_type_dimension='SVD',n_components=50,peak_read=[],rna_exprs=[],load_mode=0,input_file_path='',
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		model training for peak-TF association prediction
		:param data: (dataframe) ATAC-seq peak loci annotations
		:param dict_feature: dictionary containing feature matrix of observations for each feature type
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param method_type_vec: (array or list) methods used for initial TF binding prediction for the given TFs
		:param method_type_dimension: the method used for feature dimension reduction
		:param n_components: the number of latent components used in feature dimension reduction
		:param peak_read: (dataframe) peak accessibility matrix (row:metacell, column:peak)
		:param rna_exprs: (dataframe) gene expression matrix (row:metacell, column:gene)
		:param load_mode: indicator of whether to retrieve data from the files
		:param input_file_path: the directory to retreive the previously estimated peak accessibility-TF expression correlation data
		:param save_mode: whether to save estimated feature correlation data
		:param output_file_path: the directory to save data
		:param outupt_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing the parameters
		:return: dictionary containing the updated peak annotations including predicted TF binding label (binary) and TF binding probability for the given TFs
		"""

		data_file_type = select_config['data_file_type']
		data_file_type_query = data_file_type
		select_config.update({'data_file_type_query':data_file_type_query})
		print('data_file_type: ',data_file_type_query)

		if 'method_type_feature_link' in select_config:
			method_type_feature_link = select_config['method_type_feature_link']

		filename_save_annot = '1'
		# the methods used to predict peak-TF associations initially
		method_type_vec = pd.Index(method_type_vec).union([method_type_feature_link],sort=False)
		method_type_feature_link_query1 = method_type_feature_link
		method_type_vec_query1 = [method_type_feature_link_query1]

		verbose_internal = self.verbose_internal

		flag_config_1 = 1
		flag_gene_annot_1 = 1
		flag_motif_data_load_1 = 1
		flag_load_1 = 1
	
		root_path_1 = select_config['root_path_1']
		data_path_save_1 = root_path_1

		input_dir = select_config['input_dir']
		output_dir = select_config['output_dir']
		input_file_path_pre1 = input_dir
		output_file_path_pre1 = output_dir
		print('input_file_path: ',input_file_path_pre1)
		print('output_file_path: ',output_file_path_pre1)

		file_path_motif = input_file_path_pre1
		select_config.update({'file_path_motif':file_path_motif})

		# load gene annotation data
		if flag_gene_annot_1>0:
			print('load gene annotations')
			# filename_gene_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path_pre1)
			# select_config.update({'filename_gene_annot':filename_gene_annot})
			filename_gene_annot = select_config['filename_gene_annot']
			
			df_gene_annot_ori = self.test_query_gene_annot_1(filename_gene_annot,verbose=verbose,select_config=select_config)
			self.df_gene_annot_ori = df_gene_annot_ori
			print('gene annotations loaded from: %s'%(filename_gene_annot))

		# load motif scanning data
		# load ATAC-seq and RNA-seq data of the metacells
		flag_load_pre1 = (flag_load_1>0)|(flag_motif_data_load_1>0)
		if (flag_load_pre1>0):
			select_config = self.test_query_load_pre1(method_type_vec=method_type_vec_query1,
														flag_motif_data_load_1=flag_motif_data_load_1,
														flag_load_1=flag_load_1,
														save_mode=1,
														verbose=verbose,select_config=select_config)

		dict_query_1 = dict()
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

		print('feature type: %s %s'%(feature_type_vec_query[0],feature_type_vec_query[1]))

		self.feature_type_vec_query = feature_type_vec_query
		self.feature_type_vec_2_ori = feature_type_vec_2_ori

		method_type_group = select_config['method_type_group']
		t_vec_1 = method_type_group.split('.')
		method_type_group_name = t_vec_1[0]
		n_neighbors_query = int(t_vec_1[1])
		print('peak clustering method: %s, using number of neighbors: %d'%(method_type_group_name,n_neighbors_query))

		method_type_vec_group_ori = [method_type_group_name]
		method_type_group_neighbor = n_neighbors_query
		select_config.update({'method_type_group_name':method_type_group_name,
								'method_type_group_neighbor':method_type_group_neighbor})

		feature_mode = 1  # with RNA-seq and ATAC-seq data
		feature_mode_query = feature_mode
		select_config.update({'feature_mode':feature_mode})

		file_save_path_1 = output_dir
		file_path_group_query = '%s/group%d'%(file_save_path_1,feature_mode_query)
		if os.path.exists(file_path_group_query)==False:
			print('the directory does not exist: %s'%(file_path_group_query))
			os.makedirs(file_path_group_query,exist_ok=True)

		select_config.update({'file_path_group_query':file_path_group_query})

		flag_iter_2 = 0
		method_type_vec_group = method_type_vec_group_ori
		select_config.update({'method_type_vec_group':method_type_vec_group,
								'flag_iter_2':flag_iter_2})

		# flag_embedding_compute=1
		flag_embedding_compute=0
		flag_clustering_1=0
		flag_group_load_1=1
		
		field_query = ['flag_embedding_compute','flag_clustering','flag_group_load']
		default_parameter_vec = [flag_embedding_compute,flag_clustering_1,flag_group_load_1]
		select_config, list1_param = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=default_parameter_vec,overwrite=False,select_config=select_config)
		flag_embedding_compute, flag_clustering_1, flag_group_load_1 = list1_param[0:3]

		dict_query_1 = dict()
		dict_latent_query1 = dict()
		peak_read = self.peak_read
		peak_loc_ori = peak_read.columns
		rna_exprs = self.rna_exprs
		print('peak accessibility matrix, dataframe of size ',peak_read.shape)
		print('gene expression matrix, dataframe of size ',rna_exprs.shape)

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
		print('method for dimension reduction: %s'%(method_type_dimension))
		print('the number of components: %d'%(n_component_sel))

		# compute feature embeddings
		if flag_embedding_compute>0:
			print('compute feature embeddings')
			start = time.time()
			
			method_type_query = method_type_feature_link
			# type_combine = 0
			# column_1 = 'type_combine'
			# # select_config.update({'type_combine':type_combine})
			# if (column_1 in select_config):
			# 	type_combine = select_config[column_1]
			# else:
			# 	select_config.update({column_1:type_combine})
			# feature_mode_vec = [1]

			input_file_path = input_file_path_pre1
			output_file_path = file_path_group_query

			# column_query = 'flag_peak_tf_combine'
			flag_peak_tf_combine = 0
			select_config.update({'flag_peak_tf_combine':flag_peak_tf_combine})

			# compute feature embeddings
			dict_query_1, select_config = self.test_query_feature_embedding_pre1(feature_type_vec=[],
																					method_type=method_type_query,
																					n_components=n_components,
																					iter_id=-1,
																					config_id_load=-1,
																					flag_config=1,
																					flag_motif_data_load=0,
																					flag_load_1=0,
																					input_file_path=input_file_path,
																					save_mode=1,output_file_path=output_file_path,output_filename='',
																					filename_prefix_save='',
																					filename_save_annot='',
																					verbose=verbose,select_config=select_config)
			stop = time.time()
			print('computing feature embeddings used %.2fs'%(stop-start))

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
		
		if flag_clustering_1>0:
			# feature_type_vec = feature_type_vec_query
			dict_latent_query1 = self.test_query_feature_clustering_pre1(data=dict_query_1,feature_query_vec=peak_loc_ori,
																			feature_type_vec=feature_type_vec_2,
																			save_mode=1,
																			verbose=verbose,select_config=select_config)

		flag_annot_1 = 1
		thresh_size_group_query = 0
		if flag_annot_1>0:
			# thresh_size_1 = 100
			thresh_size_1 = 50
			if 'thresh_size_group' in select_config:
				thresh_size_group = select_config['thresh_size_group']
				thresh_size_1 = thresh_size_group

			thresh_size_group_query = thresh_size_1

			# for selecting the peak loci predicted with TF binding
			thresh_score_query_1 = 0.15
			if 'thresh_score_group_1' in select_config:
				thresh_score_group_1 = select_config['thresh_score_group_1']
				thresh_score_query_1 = thresh_score_group_1
			
			thresh_score_default_1 = thresh_score_query_1
			thresh_score_default_2 = 0.10

		input_file_path_2 = file_path_group_query
		output_file_path_2 = file_path_group_query
		if flag_group_load_1>0:
			# load feature group estimation for peak loci
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
																													thresh_size=thresh_size_group_query,
																													input_file_path=input_file_path_2,
																													save_mode=1,
																													output_file_path=output_file_path_2,
																													filename_prefix_save=filename_prefix_save_query2,
																													filename_save_annot=filename_save_annot2_ori,
																													output_filename='',
																													verbose=0,select_config=select_config)

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
				# query computed low-dimensional embeddings of observations
				dict_latent_query1 = self.test_query_feature_embedding_load_1(dict_file=dict_file_feature,
																				feature_query_vec=peak_loc_ori,
																				feature_type_vec=feature_type_vec_2,
																				method_type_vec=[],
																				method_type_dimension=method_type_dimension,
																				n_components=n_components,
																				n_component_sel=n_component_sel,
																				reconstruct=reconstruct,
																				flag_combine=flag_combine,
																				input_file_path='',
																				save_mode=0,output_file_path='',output_filename='',
																				filename_prefix_save=filename_prefix_save_2,
																				filename_save_annot=filename_save_annot_2,
																				verbose=0,select_config=select_config)

				dict_feature = dict_latent_query1

			n_neighbors = 100
			if 'neighbor_num' in select_config:
				n_neighbors = select_config['neighbor_num']
			n_neighbors_query = n_neighbors+1

			# query the neighbors of feature query
			# query the neighbors of observations (peaks)
			flag_neighbor_query=1
			if flag_neighbor_query>0:
				# list_query1 = [[feature_nbrs_1,dist_nbrs_1],[feature_nbrs_2,dist_nbrs_2]]
				list_query1 = self.test_query_feature_neighbor_load_1(dict_feature=dict_feature,
																		feature_type_vec=feature_type_vec_query,
																		n_neighbors=n_neighbors,
																		load_mode=1,
																		input_file_path=input_file_path_2,
																		save_mode=1,
																		output_file_path=output_file_path_2,
																		verbose=verbose,select_config=select_config)

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
			# print(method_type_query)
			print('method for initial peak-TF association estimation: %s'%(method_type_query))
			dict_motif_data_query = dict_motif_data[method_type_query]
		
			motif_data = []
			motif_data_score = []
			motif_name_ori = []
			if 'motif_data' in dict_motif_data_query:
				motif_data_query1 = dict_motif_data_query['motif_data']
				motif_data_query1 = motif_data_query1.loc[peak_loc_ori,:]
				motif_data = motif_data_query1
				motif_name_ori = motif_data_query1.columns
			else:
				filename_translation = select_config['filename_translation']
				df_annot_motif_1 = pd.read_csv(filename_translation,index_col=0,sep='\t')
				column_query = 'tf'
				motif_name_ori = df_annot_motif_1[column_query]

			if 'motif_data_score' in dict_motif_data_query:
				motif_data_score_query1 = dict_motif_data_query['motif_data_score']
				motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_ori,:]
				motif_data_score = motif_data_score_query1

			motif_data_query1 = motif_data
			motif_data_score_query1 = motif_data_score

			# motif_name_ori = motif_data_query1.columns
			gene_name_expr_ori = rna_exprs.columns
			motif_query_name_expr = pd.Index(motif_name_ori).intersection(gene_name_expr_ori,sort=False)
			motif_query_num1 = len(motif_query_name_expr)
			# print('the number of TFs with expression: %d'%(motif_query_num1))

			type_motif_query=0
			if 'type_motif_query' in select_config:
				type_motif_query = select_config['type_motif_query']

			tf_name = select_config['tf_name']
			if (len(tf_name)>0) and (tf_name!='-1'):
				t_vec_1 = np.asarray(tf_name.split(','))
				motif_idvec_query = t_vec_1
			else:
				motif_idvec_query = motif_query_name_expr

			motif_query_num = len(motif_idvec_query)
			query_num_ori = motif_query_num
			print('the number of TFs with expression: %d'%(motif_query_num))

			columns_1 = select_config['columns_1']
			t_vec_2 = columns_1.split(',')
			column_pred1, column_score_1 = t_vec_2[0:2]
			# column_score_query = column_score_1
			# column_score_query1 = column_score_query
			column_score_query1 = column_score_1
			column_motif = '-1'
			if len(t_vec_2)>2:
				column_motif = t_vec_2[2]
			
			column_vec_query = [column_pred1,column_score_query1,column_motif]
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
			if 'model_type_id1' in select_config:
				model_type_id1 = select_config['model_type_id1']
			else:
				select_config.update({model_type_id1:model_type_id1})

			beta_mode = select_config['beta_mode']
			method_type_group = select_config['method_type_group']
			n_neighbors = select_config['neighbor_num']

			query_id1,query_id2 = select_config['query_id_1'],select_config['query_id_2']
			iter_mode = 0
			query_num_1 = motif_query_num
			iter_vec_1 = np.arange(query_num_1)

			if (query_id1>=0) and (query_id1<query_num_1) and (query_id2>query_id1) :
				iter_mode = 1
				start_id1 = query_id1
				start_id2 = np.min([query_id2,query_num_1])
				iter_vec_1 = np.arange(start_id1,start_id2)
				interval_save = False
				print('query_id1, query_id2: ',query_id1,query_id2)
				print('start_id1, start_id2: ',start_id1,start_id2)

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
			# print('flag_select_1, flag_select_2, flag_sample: ',flag_select_1,flag_select_2,flag_sample)

			run_idvec = [-1,1]
			method_type_vec_query = [method_type_feature_link]
			
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

			# prepare the folder to save the peak accessibility-TF expression correlations
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
			print('thresholds for selecting positive samples: ',thresh_vec_sel_1)
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
			# overwrite_2 = True
			verbose_internal = 1
			# verbose_internal = 0
			column_1 = 'verbose_mode'
			if column_1 in select_config:
				verbose_internal = select_config[column_1]
			else:
				select_config.update({column_1:verbose_internal})
			self.verbose_internal = verbose_internal

			save_mode_pre2 = 1
			dict_file_query_1 = dict()
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

				select_config.update({'motif_id_query':motif_id_query})

				flag1=1
				if flag1>0:
				# try:
					# load the TF binding prediction file
					# the possible columns: (signal,motif,predicted binding,motif group)
					start_1 = time.time()
					
					flag_group_query_1 = 1
					load_mode_pre1_1 = 1
					if load_mode_pre1_1>0:
						dict_file_load = select_config['dict_file_load']
						input_filename = dict_file_load[motif_id_query] # the file which saves the previous estimation for each TF;

						if os.path.exists(input_filename)==True:
							df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
							peak_loc_1 = df_1.index
							peak_num_1 = len(peak_loc_1)
							print('load the peak-TF association estimated by %s'%(method_type_feature_link))
							print('the number of peak loci: %d'%(peak_num_1))
							print('input filename: %s'%(input_filename))
						else:
							print('the file does not exist: %s'%(input_filename))
							continue

						column_vec = df_1.columns
						column_vec_query = select_config['column_vec_link']
						column_vec_query_1 = pd.Index(column_vec_query).intersection(column_vec,sort=False)
						if not (column_pred1 in column_vec_query_1):
							# print('the estimation not included')
							print('the column %s not included '%(column_pred1))
							continue

						column_vec_query_2 = pd.Index(column_vec_query).difference(column_vec,sort=False)
						# the estimation of the motif not included
						if len(column_vec_query_2)>0:
							print('the column not included: ',column_vec_query_2,motif_id_query,i1)
							# continue

						df_query_1 = df_1.loc[:,column_vec_query_1]

					peak_loc_1 = df_query_1.index
					column_vec = df_query_1.columns
					df_query1 = pd.DataFrame(index=peak_loc_ori)
					df_query1.loc[peak_loc_1,column_vec] = df_query_1
					if verbose_internal==2:
						print('the dataframe, size of ',df_query1.shape)

					flag_motif_query=1
					flag_select_query=1

					if not (column_motif in df_query1.columns):
						print('query the motif score ',column_motif,motif_id_query,i1)
						if len(motif_data)>0:
							peak_loc_motif = peak_loc_ori[motif_data[motif_id_query]>0]
							df_query1.loc[peak_loc_motif,column_motif] = motif_data_score.loc[peak_loc_motif,motif_id_query]
						else:
							print('please provide motif scanning data ')
							return

					motif_score = df_query1[column_motif]
					query_vec_1 = np.unique(df_query1[column_motif])
					if verbose_internal>0:
						t_vec_1 = utility_1.test_stat_1(query_vec_1)
						print('the number of unique motif scores: %d'%(len(query_vec_1)))
						print('the maximum, mininum, mean, and median of motif scores for TF %s: '%(motif_id_query),t_vec_1)

					try:
						id_motif = (df_query1[column_motif].abs()>0)
					except Exception as error:
						print('error! ',error)
						id_motif = (df_query1[column_motif].isin(['True',True,1,'1']))

					df_query1_motif = df_query1.loc[id_motif,:]	# the peak loci with binding motif identified
					peak_loc_motif = df_query1_motif.index
					peak_num_motif = len(peak_loc_motif)
					print('peak loci with the motif of TF %s: %d'%(motif_id_query,peak_num_motif))
						
					if peak_num_motif==0:
						continue
				
					column_query = 'folder_group_save'
					input_file_path_2 = select_config[column_query]
					output_file_path_2 = input_file_path_2

					# compute the enrichment of peak loci with TF motif in paired groups
					filename_prefix_1 = 'test_query_df_overlap'
					method_type_query = method_type_feature_link
					if flag_motif_query>0:
						print('estimate enrichment of peak loci with TF motif in the paired groups')
						filename_save_annot2_1 = '%s.%s.%s'%(method_type_group,motif_id_query,data_file_type_query)
						for query_id1 in [1,2]:
							column_1 = 'filename_overlap_motif_%d'%(query_id1)
							filename_query = '%s/%s.%s.motif.%d.txt' % (input_file_path_2,filename_prefix_1,filename_save_annot2_1,query_id1)
							select_config.update({column_1:filename_query})

						# estimate enrichment of peak loci with detected TF motif in the paired groups
						t_vec_1 = self.test_query_feature_overlap_1(data=df_query1_motif,motif_id_query=motif_id_query,
																		df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,verbose=verbose,select_config=select_config)
						
						df_overlap_motif, df_group_basic_motif, dict_group_basic_motif, load_mode_query1 = t_vec_1
						
					id_pred1 = (df_query1[column_pred1]>0)
					peak_loc_pre1 = df_query1.index
					peak_loc_query1 = peak_loc_pre1[id_pred1]
					print('ATAC-seq peak loci: %d'%(len(peak_loc_pre1)))
					print('peak loci with TF binding predicted by %s: %d'%(method_type_feature_link,len(peak_loc_query1)))
					
					df_pre1 = df_query1
					id_1 = id_pred1
					df_pred1 = df_query1.loc[id_1,:] # the selected peak loci

					if flag_select_query>0:
						print('estimate enrichment of peak loci with predicted TF binding in the paired groups')
						# select the peak loci predicted with TF binding
						# query enrichment of peak loci in paired groups
						filename_save_annot2_2 = '%s.%s.%s.%s'%(method_type_query,method_type_group,motif_id_query,data_file_type_query)
						for query_id1 in [1,2]:
							column_1 = 'filename_overlap_%d'%(query_id1)
							filename_query = '%s/%s.%s.%d.txt' % (input_file_path_2,filename_prefix_1,filename_save_annot2_2,query_id1)
							select_config.update({column_1:filename_query})

						# estimate enrichment of peak loci with predicted TF binding in the paired groups
						t_vec_2 = self.test_query_feature_overlap_2(data=df_pred1,motif_id_query=motif_id_query,
																		df_overlap_compare=df_overlap_compare,
																		input_file_path=input_file_path_2,
																		save_mode=1,verbose=verbose,select_config=select_config)
						
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
						group_vec_query_1 = np.asarray(df_overlap_query2.loc[:,['group1','group2']]) # the paired group with selected peaks (peaks with predicted TF binding)
						print('the number of paired groups for TF %s: %d'%(motif_id_query,df_overlap_query.shape[0]))
						print('the number of selected paired groups for TF %s: %d'%(motif_id_query,df_overlap_query2.shape[0]))

						self.df_overlap_query = df_overlap_query
						self.df_overlap_query2 = df_overlap_query2
						self.df_group_basic_query_2 = df_group_basic_query_2
						
						load_mode_2 = load_mode_query2
						if load_mode_2<2:
							output_filename = select_config['filename_overlap_1']
							df_overlap_query.to_csv(output_filename,sep='\t')

					flag_neighbor_query_1 = 0
					if flag_group_query_1>0:
						flag_neighbor_query_1 = 1

					flag_neighbor = 1
					flag_neighbor_2 = 1  # query neighbor of selected peak in the paired groups

					feature_type_vec = feature_type_vec_query
					# print('feature_type_vec: ',feature_type_vec)
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

						# query peak loci predicted with TF binding sites using clustering
						start = time.time()
						df_overlap_1 = []
						group_type_vec = ['group1','group2']
						# group_vec_query = ['group1','group2']
						list_group_query = [df_group_1,df_group_2]
						dict_group = dict(zip(group_type_vec,list_group_query))
						
						dict_neighbor = self.dict_neighbor
						dict_group_basic_1 = self.dict_group_basic_1
						# the overlap and the selected overlap above count and p-value thresholds
						# group_vec_query_1: the group with selected peak loci or enriched with selected peak loci
						column_id2 = 'peak_id'
						df_pre1[column_id2] = np.asarray(df_pre1.index)
						
						neighbor_num_sel = select_config['neighbor_num_sel']
						df_pre1 = self.test_query_feature_group_neighbor_pre1(data=df_pre1,dict_group=dict_group,dict_neighbor=dict_neighbor,
																				group_type_vec=group_type_vec,
																				feature_type_vec=feature_type_vec,
																				group_vec_query=group_vec_query_1,
																				column_vec_query=[],
																				n_neighbors=neighbor_num_sel,
																				verbose=verbose,select_config=select_config)

						stop = time.time()
						# print('query feature group and neighbor annotation for TF %s (%s) used %.2fs'%(motif_id_query,motif_id2,stop-start))
						print('query feature group and neighbor annotation for TF %s used %.2fs'%(motif_id_query,stop-start))

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
						print('query peak accessibility-TF expression correlation')
						column_value = column_peak_tf_corr
						thresh_value=-0.05
						column_signal = 'signal'
						column_query1 = 'folder_correlation'
						input_file_path_query1 = select_config[column_query1]
						output_file_path_query1 = input_file_path_query1
						filename_prefix_save_query = 'test_peak_tf_correlation.%s.%s'%(motif_id_query,data_file_type_query)
						
						df_query1 = self.test_query_compare_peak_tf_corr_1(data=df_pre1,motif_id_query=motif_id_query,
																			column_value=column_value,
																			motif_data=motif_data,
																			peak_read=peak_read,
																			rna_exprs=rna_exprs,
																			input_file_path=input_file_path_query1,
																			save_mode=1,
																			output_file_path=output_file_path_query1,
																			output_filename='',
																			filename_prefix_save=filename_prefix_save_query,
																			filename_save_annot='',
																			verbose=verbose,select_config=select_config)
						df_pre1 = df_query1

					flag_query_2=1
					method_type_feature_link = select_config['method_type_feature_link']
					if flag_query_2>0:
						# training sample selection
						df_pre1,feature_query_1,dict_peak_query = self.test_query_binding_compute_select_1(data=df_pre1,
																											feature_type_vec=feature_type_vec,
																											flag_select_1=1,
																											flag_select_2=1,
																											input_file_path='',
																											save_mode=1,
																											output_file_path='',
																											verbose=verbose,select_config=select_config)

						sample_id_train = feature_query_1
						flag_shuffle=True
						filename_prefix_save = filename_link_prefix
						filename_save_annot = filename_link_annot
						output_filename = filename_save_link_pre1
						input_file_path = input_file_path_pre1
						output_file_path_query = output_file_path_pre2
						type_combine_query = 0
						column_1 = 'type_combine'
						if (column_1 in select_config):
							type_combine_query = select_config[column_1]
						else:
							select_config.update({column_1:type_combine_query})

						file_path_query1 = select_config['file_path_save_link']
						data_path_save = file_path_query1
						select_config.update({'data_path_save':file_path_query1})

						# type_id_model: 0,regression model; 1,classification model
						type_id_model = 1
						select_config.update({'type_id_model':type_id_model})

					flag_train_1 = 1
					if flag_train_1>0:
						# model training for peak-TF association prediction
						dict_file_query_1 = self.test_query_binding_compute_train_1(data=df_pre1,sample_id_train=sample_id_train,
																					peak_query_vec=peak_loc_pre1,
																					feature_query=motif_id_query,
																					dict_feature=dict_feature,
																					dict_file_query=dict_file_query_1,
																					feature_type_vec=feature_type_vec_query,
																					motif_data=motif_data_query1,
																					flag_shuffle=flag_shuffle,
																					input_file_path=input_file_path,
																					save_mode=1,
																					output_file_path=output_file_path_query,
																					output_filename=output_filename,
																					filename_prefix_save=filename_prefix_save,
																					filename_save_annot=filename_save_annot,
																					verbose=verbose,select_config=select_config)

					stop_1 = time.time()
					print('TF binding prediction for TF %s used %.2fs'%(motif_id_query,stop_1-start_1))
				
				# except Exception as error:
				# 	print('error! ',error, motif_id_query,motif_id2,i1)
				# 	# return

			return dict_file_query_1

	## ====================================================
	# training sample selection
	def test_query_binding_compute_select_1(self,data=[],feature_query='',feature_type_vec=[],method_type='',dict_thresh={},thresh_vec=[],
												flag_select_1=1,flag_select_2=1,input_file_path='',
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		training sample selection
		:param data: (dataframe) ATAC-seq peak loci annotations
		:param feature_query: (str) name of the TF for which we perform TF binding prediction in peak loci
		:param feature_type_vec: (array or list) feature types of feature representations of observations (peak loci)
		:param method_type: (str) the method used to predict peak-TF associations initially
		:param dict_thresh: dictionary containing thresholds for group (or paired groups) selection using peak number and peak enrichment
		:param thresh_vec: (array or list) thresholds for group (or paired groups) selection using peak number and peak enrichment
		:param flag_select_1: the type of approach to select pseudo positive training sample
		:param flag_select_2: the type of approach to select pseudo negative training sample
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) updated ATAC-seq peak loci annotations including which peaks are selected as pseudo positive or pseudo negative training samples;
				 2. (pandas.Index) the ATAC-seq peak loci selected as pseudo positive or pseudo negative training samples;
				 3. dictionary containing four subsets of selected training samples:
				 	(1) pseudo positive training sample;
				 	(2) pseudo negative training sample;
				 	(3) pseudo negative training sample selected from peak loci with the TF motif detected;
				 	(4) pseudo negative training sample selected from peak loci without the TF motif detected;
		"""

		flag_query1=1
		if method_type=='':
			method_type_feature_link = select_config['method_type_feature_link']
		else:
			method_type_feature_link = method_type

		if flag_query1>0:
			df_query1 = data
			column_corr_1 = 'peak_tf_corr'
			column_pval = 'peak_tf_pval_corrected'
			thresh_corr_1, thresh_pval_1 = 0.30, 0.05
			thresh_corr_2, thresh_pval_2 = 0.1, 0.1
			thresh_corr_3, thresh_pval_2 = 0.05, 0.1

			peak_loc_pre1 = df_query1.index
			df_group_basic_query_2 = self.df_group_basic_query_2
			df_overlap_query = self.df_overlap_query
			verbose_internal = self.verbose_internal
			column_pred1 = select_config['column_pred1']
			id_pred1 = (df_query1[column_pred1]>0) # peak loci with predicted TF binding
			if flag_select_1==1:
				print('select pseudo positive training sample ')
				# select pseudo positive training sample
				# find the paired groups with enrichment
				df_annot_vec = [df_group_basic_query_2,df_overlap_query]
				dict_group_basic_2 = self.dict_group_basic_2
				dict_group_annot_1 = {'df_group_basic_query_2':df_group_basic_query_2,
										'df_overlap_query':df_overlap_query,
										'dict_group_basic_2':dict_group_basic_2}

				if verbose_internal==2:
					key_vec_query = list(dict_group_annot_1.keys())
					print('the fields in dict_group_annot_1: ')
					for field_id in key_vec_query:
						print(field_id)
						print(dict_group_annot_1[field_id])

				file_path_save_group = select_config['folder_group_save']
				output_file_path_query = file_path_save_group
							
				# select training sample
				column_1 = 'thresh_overlap_default_1'
				column_2 = 'thresh_overlap_default_2'
				column_3 = 'thresh_overlap'
				column_pval_group = 'thresh_pval_1'
				column_quantile = 'thresh_quantile_overlap'
				column_thresh_query = [column_1,column_2,column_3,column_pval_group,column_quantile]

				if len(dict_thresh)==0:
					if len(thresh_vec)==0:
						thresh_overlap_default_1 = 0
						thresh_overlap_default_2 = 0
						thresh_overlap = 0
													
						thresh_pval_group = 0.25
						# thresh_quantile_overlap = 0.50
						thresh_quantile_overlap = 0.75
						thresh_vec = [thresh_overlap_default_1,thresh_overlap_default_2,thresh_overlap,thresh_pval_group,thresh_quantile_overlap]
								
					dict_thresh = dict(zip(column_thresh_query,thresh_vec))

				# select pseudo positive training sample
				motif_id_query = feature_query
				df_query1 = self.test_query_training_group_pre1(data=df_query1,dict_annot=dict_group_annot_1,
																	motif_id=motif_id_query,
																	dict_thresh=dict_thresh,
																	thresh_vec=thresh_vec,
																	save_mode=1,output_file_path=output_file_path_query,
																	verbose=verbose,select_config=select_config)

				column_corr_1 = 'peak_tf_corr'
				column_pval = 'peak_tf_pval_corrected'
				column_score_query1 = select_config['column_score_query1']
				column_vec_query = [column_corr_1,column_pval,column_score_query1]

				df_pre2 = df_query1.loc[id_pred1,:]  # peak loci with predicted TF binding
				# select training sample based on quantiles of peak-TF link scores
				df_pre2, select_config = self.test_query_feature_quantile_1(data=df_pre2,query_idvec=[],
																			column_vec_query=column_vec_query,
																			verbose=verbose,select_config=select_config)

				peak_loc_query_1 = []
				peak_loc_query_2 = []
				flag_corr_1 = 1
				flag_score_query_1 = 0
				flag_enrichment_sel = 1
				peak_loc_query_group2_1 = self.test_query_training_select_pre1(data=df_pre2,column_vec_query=[],
																				flag_corr_1=flag_corr_1,
																				flag_score_1=flag_score_query_1,
																				flag_enrichment_sel=flag_enrichment_sel,
																				save_mode=1,
																				verbose=verbose,select_config=select_config)
							
				peak_num_group2_1 = len(peak_loc_query_group2_1)
				peak_query_vec = peak_loc_query_group2_1  # the peak loci in class 1
						
			elif flag_select_1==2:
				# include each peak predicted with TF binding by the first method as pseudo positive training sample
				df_pre2 = df_query1.loc[id_pred1,:]
				peak_query_vec = df_pre2.index

			df_pre1 = df_query1
			peak_vec_1 = peak_query_vec # selected pseudo positive training sample
			peak_query_num_1 = len(peak_query_vec)
			if flag_select_2==1:
				# select pseudo negative training sample
				# print('feature_type_vec_query: ',feature_type_vec_query)
				peak_vec_2_1, peak_vec_2_2 = self.test_query_training_select_group2(data=df_pre1,motif_id=motif_id_query,
																					peak_query_vec_1=peak_vec_1,
																					feature_type_vec=feature_type_vec,
																					save_mode=1,
																					verbose=verbose,select_config=select_config)

				peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)

			elif flag_select_2 in [2,3]:
				flag_sample = 1
				peak_vec_2, peak_vec_2_1, peak_vec_2_2 = self.test_query_training_select_group2_2(data=df_pre1,id_query=id_pred1,
																									peak_query_vec_1=peak_vec_1,
																									method_type=method_type_feature_link,
																									flag_sample=flag_sample,
																									flag_select=flag_select_2,
																									save_mode=1,
																									verbose=verbose,select_config=select_config)

			if flag_select_1>0:
				df_pre1.loc[peak_query_vec,'class'] = 1

			if flag_select_2 in [1,3]:
				df_pre1.loc[peak_vec_2_1,'class'] = -1
				df_pre1.loc[peak_vec_2_2,'class'] = -2

				peak_num_2_1 = len(peak_vec_2_1)
				peak_num_2_2 = len(peak_vec_2_2)
				# print('peak_vec_2_1, peak_vec_2_2: ',peak_num_2_1,peak_num_2_2)
				print('selected pseudo negative training sample from peak loci with detected TF motif: %d'%(peak_num_2_1))
				print('selected pseudo negative training sample from peak loci without detected TF motif: %d'%(peak_num_2_2))
							
				peak_vec_2 = pd.Index(peak_vec_2_1).union(peak_vec_2_2,sort=False)

			elif flag_select_2 in [2]:
				df_pre1.loc[peak_vec_2,'class'] = -1

			peak_num_2 = len(peak_vec_2)

			peak_query_num_1 = len(peak_vec_1)
			# peak_query_num_1 = len(peak_query_vec)

			print('selected pseudo positive training sample: %d'%(peak_query_num_1))
			print('selected pseudo negative training sample: %d'%(peak_num_2))

			sample_id_train = pd.Index(peak_vec_1).union(peak_vec_2,sort=False)

			feature_query_1 = sample_id_train
			field_query = ['group1','group2','group2_1','group2_2']
			list1 = [peak_vec_1,peak_vec_2,peak_vec_2_1,peak_vec_2_2]
			dict_feature_query = dict(zip(field_query,list1))

			return df_pre1,feature_query_1,dict_feature_query

	## ====================================================
	# model training for peak-TF association prediction
	def test_query_binding_compute_train_1(self,data=[],sample_id_train=[],peak_query_vec=[],feature_query='',
												dict_feature={},dict_file_query={},feature_type_vec=[],motif_data=[],flag_shuffle=True,input_file_path='',
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		model training for peak-TF association prediction
		:param data: (dataframe) peak annotations including initially predicated peak-TF associations for the given TF
		:param sample_id_train: (array) the identifiers of the observations
		:param peak_query_vec: (array) ATAC-seq peak loci for which we perform TF binding prediction for the given TF
		:param feature_query: (str) name of the TF for which we perform binding prediction in peak loci
		:param dict_feature: dictionary containing feature matrices of predictor variables and response variable values
		:param dict_file_query: the dictionary to save updated peak annotations including predicted TF binding (binary) and TF binding probability for the given TF
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param motif_data: (dataframe) motif presence in peak loci by motif scanning (binary) (row:peak, column:TF (associated with motif))
		:param flag_shuffle: indicator of whether to shuffle the observations
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		return: dictionary containing the updated peak annotations including predicted TF binding label (binary) and TF binding probability for the given TF
		"""

		df_pre1 = data
		if flag_shuffle>0:
			sample_num_train = len(sample_id_train)
			id_query1 = np.random.permutation(sample_num_train)
			sample_id_train = sample_id_train[id_query1]

		train_valid_mode_2 = 0
		if 'train_valid_mode_2' in select_config:
			train_valid_mode_2 = select_config['train_valid_mode_2']
						
		if train_valid_mode_2>0:
			sample_id_train_ori = sample_id_train.copy()
			from sklearn.model_selection import train_test_split
			sample_id_train, sample_id_valid, sample_id_train_, sample_id_valid_ = train_test_split(sample_id_train_ori,sample_id_train_ori,test_size=0.1,random_state=0)
		else:
			sample_id_valid = []
		
		peak_loc_pre1 = df_pre1.index
		if len(peak_query_vec)==0:
			peak_query_vec = peak_loc_pre1
		
		sample_id_test = peak_query_vec
		sample_idvec_query = [sample_id_train,sample_id_valid,sample_id_test]
		motif_id_query = feature_query
		feature_type_vec_query = feature_type_vec

		id1 = (df_pre1['class']==1)
		peak_vec_1 = peak_loc_pre1[id1]
		peak_query_num1 = len(peak_vec_1)

		df_pre1[motif_id_query] = 0
		df_pre1.loc[peak_vec_1,motif_id_query] = 1

		verbose_internal = self.verbose_internal
		if (verbose_internal==2):
			print('selected pseudo positive training sample, data preview: ')
			print(df_pre1.loc[peak_vec_1,:])

		iter_num = 1
		flag_train1 = 1
		if flag_train1>0:
			if verbose_internal==2:
				key_vec = np.asarray(list(dict_feature.keys()))
				print('dict_feature: ',key_vec)

			# flag_scale_1: 0, without feature scaling; 1, with feature scaling
			flag_scale_1 = select_config['flag_scale_1']

			iter_id1 = 0
			output_file_path_query = output_file_path

			# model training for peak-TF link prediction
			df_pre2 = self.test_query_compare_binding_train_unit1(data=df_pre1,peak_query_vec=peak_loc_pre1,
																	peak_vec_1=peak_vec_1,
																	motif_id_query=motif_id_query,
																	dict_feature=dict_feature,
																	feature_type_vec=feature_type_vec_query,
																	sample_idvec_query=sample_idvec_query,
																	motif_data=motif_data,
																	flag_scale=flag_scale_1,
																	input_file_path=input_file_path,
																	save_mode=save_mode,
																	output_file_path=output_file_path_query,
																	output_filename=output_filename,
																	filename_prefix_save=filename_prefix_save,
																	filename_save_annot=filename_save_annot,
																	verbose=verbose,select_config=select_config)

			if save_mode>0:
				dict_file_query.update({motif_id_query:df_pre2})

			return dict_file_query
		
def run_pre1(run_id=1,species='human',cell=0,generate=1,chromvec=[],testchromvec=[],data_file_type='',metacell_num=500,peak_distance_thresh=100,highly_variable=0,
				input_dir='',filename_gene_annot='',filename_atac_meta='',filename_rna_meta='',filename_motif_data='',filename_motif_data_score='',file_mapping='',file_peak='',
				method_type_feature_link='',method_type_dimension='',
				tf_name='',filename_prefix='',filename_annot='',input_link='',columns_1='',
				output_dir='',output_filename='',path_id=2,save=1,type_group=0,type_group_2=0,type_group_load_mode=1,type_combine=0,
				method_type_group='phenograph.20',thresh_size_group=50,thresh_score_group_1=0.15,
				n_components=100,n_components_2=50,neighbor_num=100,neighbor_num_sel=30,
				model_type_id='LogisticRegression',ratio_1=0.25,ratio_2=1.5,thresh_score='0.25,0.75',
				flag_group=-1,flag_embedding_compute=0,flag_clustering=0,flag_group_load=1,flag_scale_1=0,flag_reduce=1,
				beta_mode=0,verbose_mode=1,query_id1=-1,query_id2=-1,query_id_1=-1,query_id_2=-1,train_mode=0,config_id_load=-1):
	
	flag_query_1=1
	if flag_query_1>0:
		run_id = int(run_id)
		species_id = str(species)
		cell_type_id = int(cell)

		# print('cell_type_id: %d'%(cell_type_id))
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
		type_combine = int(type_combine)
		method_type_group = str(method_type_group)
		thresh_size_group = int(thresh_size_group)
		thresh_score_group_1 = float(thresh_score_group_1)
		thresh_score = str(thresh_score)
		method_type_feature_link = str(method_type_feature_link)
		method_type_dimension = str(method_type_dimension)

		n_components = int(n_components)
		n_component_sel = int(n_components_2)
		neighbor_num = int(neighbor_num)
		print('neighbor_num ',neighbor_num)
		print('neighbor_num_sel ',neighbor_num_sel)
		neighbor_num_sel = int(neighbor_num_sel)
		model_type_id1 = str(model_type_id)

		input_link = str(input_link)
		columns_1 = str(columns_1)
		filename_prefix = str(filename_prefix)
		filename_annot = str(filename_annot)
		tf_name = str(tf_name)

		if filename_prefix=='':
			filename_prefix = data_file_type
		
		ratio_1 = float(ratio_1)
		ratio_2 = float(ratio_2)
		flag_group = int(flag_group)

		flag_embedding_compute = int(flag_embedding_compute)
		flag_clustering = int(flag_clustering)
		flag_group_load = int(flag_group_load)

		flag_scale_1 = int(flag_scale_1)
		flag_reduce = int(flag_reduce)
		beta_mode = int(beta_mode)
		verbose_mode = int(verbose_mode)

		input_dir = str(input_dir)
		output_dir = str(output_dir)
		filename_gene_annot = str(filename_gene_annot)
		filename_atac_meta = str(filename_atac_meta)
		filename_rna_meta = str(filename_rna_meta)
		filename_motif_data = str(filename_motif_data)
		filename_motif_data_score = str(filename_motif_data_score)
		file_mapping = str(file_mapping)
		file_peak = str(file_peak)
		output_filename = str(output_filename)
		
		path_id = int(path_id)
		run_id_save = int(save)
		if run_id_save<0:
			run_id_save = run_id

		config_id_load = int(config_id_load)

		celltype_vec = ['pbmc']
		flag_query1=1
		if flag_query1>0:
			query_id1 = int(query_id1)
			query_id2 = int(query_id2)
			query_id_1 = int(query_id_1)
			query_id_2 = int(query_id_2)
			train_mode = int(train_mode)
			data_file_type = str(data_file_type)

			type_id_feature = 0
			root_path_1 = '.'
			root_path_2 = '.'

			save_file_path_default = output_dir
			file_path_motif_score = input_dir
			correlation_type = 'spearmanr'

			select_config = {'root_path_1':root_path_1,'root_path_2':root_path_2,
								'data_file_type':data_file_type,
								'input_dir':input_dir,
								'output_dir':output_dir,
								'type_id_feature':type_id_feature,
								'metacell_num':metacell_num,
								'run_id':run_id,
								'filename_gene_annot':filename_gene_annot,
								'filename_atac_meta':filename_atac_meta,
								'filename_rna_meta':filename_rna_meta,
								'filename_motif_data':filename_motif_data,
								'filename_motif_data_score':filename_motif_data_score,
								'filename_translation':file_mapping,
								'input_filename_peak':file_peak,
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
								'type_combine':type_combine,
								'method_type_group':method_type_group,
								'thresh_size_group':thresh_size_group,
								'thresh_score_group_1':thresh_score_group_1,
								'thresh_score':thresh_score,
								'correlation_type':correlation_type,
								'method_type_feature_link':method_type_feature_link,
								'method_type_dimension':method_type_dimension,
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
								'verbose_mode':verbose_mode,
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
				if flag_1 in [1,2,3]:
					# type_query_group = 1
					type_query_group = 0
					parallel_group = 0
					# flag_score_query = 1
					flag_score_query = 0
					flag_select_1 = 1
					flag_select_2 = 1
					verbose = 1
					
					select_config.update({'type_query_group':type_query_group,
											'parallel_group':parallel_group,
											'flag_select_1':flag_select_1,
											'flag_select_2':flag_select_2})

					file_path_1 = '.'
					test_estimator1 = _Base2_2_pre1(file_path=file_path_1,select_config=select_config)
					if flag_1 in [1]:
						test_estimator1.test_query_compare_binding_compute_2(data=[],dict_feature=[],
																				feature_type_vec=[],
																				method_type_vec=[],
																				method_type_dimension=method_type_dimension,
																				n_components=n_components,
																				peak_read=[],
																				rna_exprs=[],
																				load_mode=0,
																				input_file_path='',
																				save_mode=1,
																				output_file_path='',
																				output_filename='',
																				filename_prefix_save='',
																				filename_save_annot='',
																				verbose=verbose,select_config=select_config)
					else:
						test_estimator1.test_query_compare_binding_pre1(data=[],dict_signal={},
																		df_signal=[],
																		dict_file_pred={},
																		feature_query_vec=[],
																		method_type_vec=[],
																		type_query_format=0,
																		save_mode=1,
																		output_file_path='',
																		output_filename='',
																		filename_prefix_save='',
																		filename_save_annot='',
																		verbose=verbose,select_config=select_config)


def run(chromosome,run_id,species,cell,generate,chromvec,testchromvec,data_file_type,input_dir,filename_gene_annot,
			filename_atac_meta,filename_rna_meta,filename_motif_data,filename_motif_data_score,file_mapping,file_peak,metacell_num,peak_distance_thresh,
			highly_variable,method_type_feature_link,method_type_dimension,tf_name,filename_prefix,filename_annot,input_link,columns_1,
			output_dir,output_filename,method_type_group,thresh_size_group,thresh_score_group_1,
			n_components,n_components_2,neighbor_num,neighbor_num_sel,model_type_id,ratio_1,ratio_2,thresh_score,
			upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
			typeid2,type_combine,folder_id,config_id_2,config_group_annot,flag_group,flag_embedding_compute,flag_clustering,flag_group_load,flag_scale_1,flag_reduce,train_id1,
			beta_mode,verbose_mode,query_id1,query_id2,query_id_1,query_id_2,train_mode,config_id_load):

	flag_1=1
	if flag_1==1:
		run_pre1(run_id,species,cell,generate,chromvec,testchromvec,data_file_type=data_file_type,
					metacell_num=metacell_num,
					peak_distance_thresh=peak_distance_thresh,
					highly_variable=highly_variable,
					input_dir=input_dir,
					filename_gene_annot=filename_gene_annot,
					filename_atac_meta=filename_atac_meta,
					filename_rna_meta=filename_rna_meta,
					filename_motif_data=filename_motif_data,
					filename_motif_data_score=filename_motif_data_score,
					file_mapping=file_mapping,
					file_peak=file_peak,
					method_type_feature_link=method_type_feature_link,
					method_type_dimension=method_type_dimension,
					tf_name=tf_name,
					filename_prefix=filename_prefix,filename_annot=filename_annot,
					input_link=input_link,
					columns_1=columns_1,
					output_dir=output_dir,
					output_filename=output_filename,
					path_id=path_id,save=save,
					type_group=type_group,type_group_2=type_group_2,type_group_load_mode=type_group_load_mode,
					type_combine=type_combine,
					method_type_group=method_type_group,
					thresh_size_group=thresh_size_group,thresh_score_group_1=thresh_score_group_1,
					n_components=n_components,n_components_2=n_components_2,
					neighbor_num=neighbor_num,neighbor_num_sel=neighbor_num_sel,
					model_type_id=model_type_id,
					ratio_1=ratio_1,ratio_2=ratio_2,
					thresh_score=thresh_score,
					flag_group=flag_group,
					flag_embedding_compute=flag_embedding_compute,
					flag_clustering=flag_clustering,
					flag_group_load=flag_group_load,
					flag_scale_1=flag_scale_1,
					flag_reduce=flag_reduce,
					beta_mode=beta_mode,
					verbose_mode=verbose_mode,
					query_id1=query_id1,query_id2=query_id2,query_id_1=query_id_1,query_id_2=query_id_2,
					train_mode=train_mode,config_id_load=config_id_load)

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
	parser.add_option("--gene_annot",default="-1",help="file path of gene position annotation file")
	parser.add_option("--atac_meta",default="-1",help="file path of ATAC-seq data of the metacells")
	parser.add_option("--rna_meta",default="-1",help="file path of RNA-seq data of the metacells")
	parser.add_option("--motif_data",default="-1",help="file path of binary motif scannning results")
	parser.add_option("--motif_data_score",default="-1",help="file path of the motif scores by motif scanning")
	parser.add_option("--file_mapping",default="-1",help="file path of the mapping between TF motif identifier and the TF name")
	parser.add_option("--file_peak",default="-1",help="file containing the ATAC-seq peak loci annotations")
	parser.add_option("--metacell",default="500",help="metacell number")
	parser.add_option("--peak_distance",default="500",help="peak distance")
	parser.add_option("--highly_variable",default="1",help="highly variable gene")
	parser.add_option("--method_type_feature_link",default="Unify",help='method for initial peak-TF association prediction')
	parser.add_option("--method_type_dimension",default="SVD",help='method for dimension reduction')
	parser.add_option("--tf",default='-1',help='the TF for which to predict peak-TF associations')
	parser.add_option("--filename_prefix",default='-1',help='prefix as part of the filenname of the initially predicted peak-TF assocations')
	parser.add_option("--filename_annot",default='1',help='annotation as part of the filename of the initially predicted peak-TF assocations')
	parser.add_option("--input_link",default='-1',help=' the directory where initially predicted peak-TF associations are saved')
	parser.add_option("--columns_1",default='pred,score',help='the columns corresponding to binary prediction and peak-TF association score')
	parser.add_option("--output_dir",default='output_file',help='the directory to save the output')
	parser.add_option("--output_filename",default='-1',help='filename of the predicted peak-TF assocations')
	parser.add_option("--method_type_group",default="phenograph.20",help="the method for peak clustering")
	parser.add_option("--thresh_size_group",default="0",help="the threshold on peak cluster size")
	parser.add_option("--thresh_score_group_1",default="0.15",help="the threshold on peak-TF association score")
	parser.add_option("--component",default="100",help='the number of components to keep when applying SVD')
	parser.add_option("--component2",default="50",help='feature dimensions to use in each feature space')
	parser.add_option("--neighbor",default='100',help='the number of nearest neighbors estimated for each peak')
	parser.add_option("--neighbor_sel",default='30',help='the number of nearest neighbors to use for each peak when performing pseudo training sample selection')
	parser.add_option("--model_type",default="LogisticRegression",help="the prediction model")
	parser.add_option("--ratio_1",default="0.25",help="the ratio of pseudo negative training samples selected from peaks with motifs and without initially predicted TF binding compared to selected pseudo positive training samples")
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
	parser.add_option("--type_combine",default="0",help="type_combine")
	parser.add_option("--folder_id",default="1",help="folder_id")
	parser.add_option("--config_id_2",default="1",help="config_id_2")
	parser.add_option("--config_group_annot",default="1",help="config_group_annot")
	parser.add_option("--flag_group",default="-1",help="flag_group")
	parser.add_option("--flag_embedding_compute",default="0",help="compute feature embeddings")
	parser.add_option("--flag_clustering",default="-1",help="perform clustering")
	parser.add_option("--flag_group_load",default="1",help="load group annotation")
	parser.add_option("--train_id1",default="1",help="train_id1")
	parser.add_option("--flag_scale_1",default="0",help="flag_scale_1")
	parser.add_option("--flag_reduce",default="1",help="reduce intermediate files")
	parser.add_option("--beta_mode",default="0",help="beta_mode")
	parser.add_option("--motif_id_1",default="1",help="motif_id_1")
	parser.add_option("--verbose_mode",default="1",help="verbose mode")
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
		opts.gene_annot,
		opts.atac_meta,
		opts.rna_meta,
		opts.motif_data,
		opts.motif_data_score,
		opts.file_mapping,
		opts.file_peak,
		opts.metacell,
		opts.peak_distance,
		opts.highly_variable,
		opts.method_type_feature_link,
		opts.method_type_dimension,
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
		opts.type_combine,
		opts.folder_id,
		opts.config_id_2,
		opts.config_group_annot,
		opts.flag_group,
		opts.flag_embedding_compute,
		opts.flag_clustering,
		opts.flag_group_load,
		opts.flag_scale_1,
		opts.flag_reduce,
		opts.train_id1,
		opts.beta_mode,
		opts.verbose_mode,
		opts.q_id1,
		opts.q_id2,
		opts.q_id_1,
		opts.q_id_2,
		opts.train_mode,
		opts.config_id)







