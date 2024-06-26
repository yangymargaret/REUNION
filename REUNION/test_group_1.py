#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData

from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import warnings
import phenograph

import sys

import os
import os.path
from optparse import OptionParser

import sklearn
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import AffinityPropagation,SpectralClustering,AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans,KMeans

from scipy import stats
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, issparse

import time
from timeit import default_timer as timer

from . import utility_1
import pickle

sc.settings.verbosity = 3   # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=100, facecolor='white')

class _Base2_group1(BaseEstimator):
	"""Base class for group estimation
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
					config={},
					select_config={}):

		self.run_id = run_id

		# self.path_1 = file_path
		self.save_path_1 = file_path
		self.config = config
		self.run_id = run_id
		self.fdl = []
		
		data_type_id = 1
		if 'data_type_id' in self.config:
			data_type_id = self.config['data_type_id']

		input_file_path1 = self.save_path_1

		self.select_config = select_config
		self.pre_data_dict_1 = dict()
		self.df_rna_obs = []
		self.df_atac_obs = []
		self.df_rna_var = []
		self.df_atac_var = []
		self.dict_feature_query_1 = dict()
		self.dict_feature_scale_1 = dict()

	## ======================================================
	# query the number of dimensions for dimension reduction
	def test_query_component_1(self,feature_mtx,adata=[],type_id_feature=0,type_id_compute=0,normalize_type=0,zero_center=False,thresh_cml_var=0.9,
									save_mode=1,output_file_path='',output_filename='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query the number of components for dimension reduction
		:param feature_mtx: (dataframe) the feature matrix of the observations (row:observation, column:variable)
		:param adata: AnnData object which saves the feature matrix of the observations
		:param type_id_feature: (int) the feature type: 0: ATAC-seq data; 1: RNA-seq data
		:param type_id_compute: indicator of which method to use to perform dimension reduction:
								0: using TruncatedSVD() function in sklearn;
								1: using TruncatedSVD() function in sklearn with optional normalization;
								2: using PCA performed by Scanpy;
								3: usinig PCA() function in sklearn;
		:param normalize_type: indicator of whether to normalize each observation to unit form
		:param zero_center: indicator of whether to zero-center the variables
		:param thresh_cml_var: (float) threshold on the cumulative explained variance to estimate the number of dimensions to keep
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		return: 1. (dataframe) the explained variance of each principal component using dimension reduction;
				2. the number of components with which the cumulative explained variance is above the threshold specified by thresh_cml_var;
				3. AnnData object which saves the transformed feature matrix after dimension reduction;
		"""

		flag_query1=1
		if flag_query1>0:
			adata2 = adata.copy()
			filename_annot1 = filename_save_annot
			if type_id_compute in [0,2]:
				n_components_2 = feature_mtx.shape[1]-1
			else:
				n_components_2 = feature_mtx.shape[1]
			
			# num_vec_1 = np.arange(1,n_components_2+1)
			feature_mtx_2, dimension_model_2, query_vec_2, adata2 = self.dimension_reduction(feature_mtx=feature_mtx,
																								ad=adata2,
																								n_components=n_components_2,
																								type_id_compute=type_id_compute,
																								normalize_type=normalize_type,
																								zero_center=zero_center,
																								save_mode=save_mode,
																								output_file_path=output_file_path,
																								output_filename=output_filename,
																								select_config=select_config)
				
			num_vec = np.arange(1,len(query_vec_2)+1)
			df_component_query = pd.DataFrame(index=num_vec,columns=['explained_variance'],data=np.asarray(query_vec_2)[:,np.newaxis])
			# cml_var_explained = np.cumsum(adata.uns['pca']['variance_ratio'])
			cml_var_explained = np.cumsum(df_component_query['explained_variance'])

			feature_mtx_2 = csr_matrix(feature_mtx_2)
			layer_name_pre1 = 'layer%d'%(type_id_compute)
			adata2.obsm[layer_name_pre1] = feature_mtx_2
			
			df_component_query.loc[:,'cml_var_explained'] = cml_var_explained
			if 'filename_component_query' in select_config:
				output_filename = select_config['filename_component_query']
			else:
				output_filename = '%s/test_cml_var_explained.%s.%d.%d.txt'%(output_file_path,filename_annot1,type_id_feature,type_id_compute)
			df_component_query.to_csv(output_filename,sep='\t')

			# thresh_cml_var = 0.9
			id_num_vec = df_component_query.index
			n_components_query = id_num_vec[df_component_query['cml_var_explained']>thresh_cml_var][0]

		return df_component_query, n_components_query, adata2

	## ======================================================
	# query the number of dimensions for dimension reduction
	def test_query_component_2(self,df=[],input_filename='',thresh_cml_var=0.90,thresh_cml_var_2=0.80,n_components_default=[50,300],flag_default=1,verbose=0,select_config={}):

		"""
		:param df: dataframe containing the explained variance of the each principal component using dimension reduction;
		:param input_filename: (str) file path of the dataframe containing the explained variance of each principal component using dimension reduction
		:param thresh_cml_var: (float) threshold on the cumulative variance to estimate the number of dimensions to keep
		:param thresh_cml_var_2: (float) the second threshold on the cumulative variance to estimate the number of dimensions to keep
		:param n_components_default: (array or list) the default number of principal components to use for dimension reduction;
		:param flag_default: indicator of whether to use the default number of principal components as thresholds on the number of components to use
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. the estimated number of dimensions to keep; 
				 2. threshold on the cumulative explained variance used to estimate the number of dimensions to keep;
		"""

		df_component_query = df
		if len(df)==0:
			if os.path.exists(input_filename)==False:
				print('the file does not exist: %s'%(input_filename))
				return -1
			else:
				df_component_query = pd.read_csv(input_filename,index_col=0,sep='\t')

		id_num_vec = df_component_query.index
		n_components_thresh1 = id_num_vec[df_component_query['cml_var_explained'] > thresh_cml_var][0]
		n_components_thresh2 = id_num_vec[df_component_query['cml_var_explained'] > thresh_cml_var_2][0]

		if verbose>0:
			print('number of components:%d, cumulative variance explained:%.5f'%(n_components_thresh1, thresh_cml_var))
			print('number of components:%d, cumulative variance explained:%.5f'%(n_components_thresh2, thresh_cml_var_2))

		thresh_cml_var_query = thresh_cml_var
		n_components_default1, n_components_default2 = n_components_default
		if n_components_thresh1>n_components_default2:
			thresh_cml_var_query = thresh_cml_var_2
			if flag_default>0:
				n_components_thresh = np.max([n_components_default2,n_components_thresh2])
			else:
				n_components_thresh = n_components_thresh2
		else:
			if flag_default>0:
				n_components_thresh = np.max([n_components_default1,n_components_thresh1])
			else:
				n_components_thresh = n_components_thresh1

		return n_components_thresh, thresh_cml_var_query

	## ======================================================
	# dimension reduction using methods: TruncatedSVD and PCA
	def dimension_reduction(self,feature_mtx,ad=[],n_components=100,type_id_compute=0,normalize_type=0,zero_center=False,copy=False,save_mode=1,output_file_path='',output_filename='',select_config={}):

		"""
		:param feature_mtx: (dataframe) the feature matrix of the observations (row:observation, column:variable)
		:param ad: AnnData object which saves the feature matrix of the observations
		:param n_components: the number of dimensions to select in dimension reduction
		:param type_id_compute: indicator of which method to use to perform dimension reduction:
								0: using TruncatedSVD() function in sklearn;
								1: using TruncatedSVD() function in sklearn with optionally normalization;
								2: using PCA performed by Scanpy;
								3: usinig PCA() function in sklearn;
		:param normalize_type: indicator of whether to normalize each observation to unit form
		:param zero_center: indicator of whether to zero-center the variables
		:param copy: indicator of whether to return a copy of an AnnData object in the pca() function in Scanpy
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save datas
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (array) the transformed feature matrix after dimension reduction;
				 2. the dimension reduction model;
				 3. (dataframe) the explained variance of each principal component using dimension reduction;
				 4. AnnData object which saves the transformed feature matrix after dimension reduction;
		"""

		query_vec = []
		if type_id_compute==0:
			feature_mtx = csr_matrix(feature_mtx)
			SVD_ = TruncatedSVD(n_components=n_components,algorithm='randomized',random_state=0,n_iter=10)
			x1 = feature_mtx
			t0 = time.time()
			SVD_.fit(x1)
			feature_mtx_transform = SVD_.transform(x1)
			dimension_model = SVD_
			explained_variance = SVD_.explained_variance_ratio_.sum()
			print('feature_mtx_transform: ',feature_mtx_transform.shape)
			print(f"SVD performed in {time.time() - t0:.3f} s")
			print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")
		
		elif type_id_compute==1:
			feature_mtx = csr_matrix(feature_mtx)
			# normalize_type=1
			if normalize_type>0:
				lsa = make_pipeline(TruncatedSVD(n_components,algorithm='randomized', random_state=0, n_iter=10), Normalizer(copy=False))
			else:
				lsa = make_pipeline(TruncatedSVD(n_components,algorithm='randomized', random_state=0, n_iter=10))
			t0 = time.time()
			X_lsa = lsa.fit_transform(feature_mtx)
			feature_mtx_transform = X_lsa
			dimension_model = lsa[0]
			explained_variance = lsa[0].explained_variance_ratio_.sum()
			print(f"LSA performed in {time.time() - t0:.3f} s")
			print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")
		
		elif type_id_compute==2:
			dimension_model = None
			if len(ad)==0:
				ad = sc.AnnData(feature_mtx)
				ad.X = csr_matrix(ad.X)
			
			np.random.seed(0)
			t0 = time.time()

			sc.tl.pca(ad,n_comps=n_components,zero_center=zero_center,use_highly_variable=False,copy=copy)
			feature_mtx_transform = ad.obsm['X_pca']
			explained_variance = np.sum(ad.uns['pca']['variance_ratio'])

			print('feature_mtx_transform: ',feature_mtx_transform.shape)
			print('type_id_compute: %d'%(type_id_compute))
			print(f"PCA performed in {time.time() - t0:.3f} s")
			print(f"Explained variance of the PCA step: {explained_variance * 100:.1f}%")

		else:
			pca = PCA(n_components=n_components, whiten = False, random_state = 0)
			t0 = time.time()
			feature_mtx_transform = pca.fit_transform(feature_mtx)
			dimension_model = pca
			explained_variance = dimension_model.explained_variance_ratio_.sum()
			print('feature_mtx_transform: ',feature_mtx_transform.shape)
			print(f"PCA performed in {time.time() - t0:.3f} s")
			print(f"Explained variance of the PCA step: {explained_variance * 100:.1f}%")
		
		if type_id_compute in [2]:
			query_vec = ad.uns['pca']['variance_ratio']
		else:	
			query_vec = dimension_model.explained_variance_ratio_

		if (save_mode>0) and (dimension_model!=None):
			if output_filename=='':
				output_filename = '%s/test_dimension_reduction.%d.%d.%d.h5'%(output_file_path,type_id_compute,normalize_type,n_components)

			with open(output_filename,'wb') as output_file:
				pickle.dump(dimension_model,output_file)

		return feature_mtx_transform, dimension_model, query_vec, ad

	## ======================================================
	# parameter configuration for the clustering methods
	def test_cluster_query_config_1(self,method_type_vec,distance_threshold=-1,linkage_type_id=0,neighbors=20,metric='euclidean',n_clusters=100,select_config={}):

		"""
		:param method_type_vec: (array or list) the methods used for clustering
		:param distance_threshold: (float) the parameter of distance threshold used in agglomerative clustering
		:param linkage_type_id: (int) identifier corresponding to the parameter of linkage type used in agglomerative clustering
		:param neighbors: (int) the parameter of the number of neighbors used in PhenoGraph clustering
		:param metric: (str) the distance metric
		:param n_clusters: (int) the number of clusters
		:param select_config: dictionary containing parameters
		:return: list containing configuration parameters for each method used for clustering
		"""

		list_config = []
		distance_threshold_pre = distance_threshold
		linkage_type_id_pre = linkage_type_id
		neighbors_pre = neighbors
		n_clusters_pre = n_clusters
		metric = metric
		neighbors_vec = [10,20,30]
		n_clusters_vec = [30,50,100]
		distance_threshold_vec = [20,50,-1]
		linkage_type_idvec = [0,1]
		field_query = 'neighbors_vec','n_clusters_vec','distance_threshold_vec','linkage_type_idvec'
		list1 = [neighbors_vec,n_clusters_vec,distance_threshold_vec,linkage_type_idvec]
		dict_query = dict(zip(field_query,list1))
		list2 = []
		for field1 in field_query:
			if field1 in select_config:
				dict_query.update({field1:select_config[field1]})

		list2 = [dict_query[field1] for field1 in dict_query]
		neighbors_vec,n_clusters_vec,distance_threshold_vec,linkage_type_idvec = list2

		for method_type_1 in method_type_vec:
			if method_type_1 in ['phenograph']:
				for neighbors in neighbors_vec:
					method_type_annot1 = '%s.%d'%(method_type_1,neighbors)
					list_config.append([method_type_1,method_type_annot1,n_clusters_pre,neighbors,distance_threshold_pre,linkage_type_id_pre,metric])
			elif method_type_1 in ['MiniBatchKMeans']:
				for n_clusters in n_clusters_vec:
					method_type_annot1 = '%s.%d'%(method_type_1,n_clusters)
					list_config.append([method_type_1,method_type_annot1,n_clusters,neighbors_pre,distance_threshold_pre,linkage_type_id_pre,metric])
			elif method_type_1 in ['AgglomerativeClustering']:
				for distance_threshold in distance_threshold_vec:
					for linkage_type_id in linkage_type_idvec:
						if distance_threshold>=0:
							method_type_annot1 = '%s.%d_%d'%(method_type_1,distance_threshold,linkage_type_id)
							list_config.append([method_type_1,method_type_annot1,n_clusters_pre,neighbors_pre,distance_threshold,linkage_type_id,metric])
						else:
							for n_clusters in n_clusters_vec:
								method_type_annot1 = '%s.%d_%d_%d'%(method_type_1,distance_threshold,linkage_type_id,n_clusters)
								list_config.append([method_type_1,method_type_annot1,n_clusters,neighbors_pre,distance_threshold,linkage_type_id,metric])
			else:
				pass

		return list_config

	## ======================================================
	# estimate the number of clusters
	def test_query_cluster_pre1(self,adata=[],feature_mtx=[],method_type='MiniBatchKMeans',n_clusters=300,cluster_num_vec=[],neighbors=20,save_mode=0,output_filename='',verbose=0,select_config={}):

		"""
		:param adata: AnnData object which saves the feature matrix of the observations
		:param feature_mtx: (dataframe) the feature matrix of the observations (row:observation, column:variable)
		:param method_type: (str) the method used for clustering
		:param n_clusters: (int) the number of clusters
		:param cluster_num_vec: (array or list): the different numbers of clusters 
		:param neighbors: (int) the parameter of the number of neighbors used in PhenoGraph clustering
		:param save_mode: indicator of whether to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) the SSE of clusters for different numbers of clusters by using the specific method for clustering
		"""

		flag_query1=1
		if flag_query1>0:
			method_type_vec = [method_type]
			n_clusters_pre = n_clusters

			if len(cluster_num_vec)==0:
				if 'cluster_num_vec' in select_config:
					cluster_num_vec = select_config['cluster_num_vec']
				else:
					interval = 10
					cluster_num_vec = list(np.arange(2,20))+list(np.arange(20,n_clusters_pre+interval,interval))
			
			t_list1 = []
			max_iter_num = 1000
			n_init, reassignment_ratio = 10, 0.01
			select_config.update({'max_iter_num':max_iter_num,
									'n_init':n_init,'reassignment_ratio':reassignment_ratio})
			
			for n_clusters_query in cluster_num_vec:
				method_type_annot1 = '%s.%d'%(method_type,n_clusters_query)
				select_config.update({'method_type_annot':method_type_annot1})
				adata_pre, query_vec = self.test_cluster_query_1(feature_mtx=feature_mtx,
																	n_clusters=n_clusters_query,
																	ad=adata.copy(),
																	method_type_vec=method_type_vec,
																	neighbors=neighbors,
																	select_config=select_config)

				if len(query_vec)>0:
					inertia_ = query_vec[0][method_type]
					t_list1.append(inertia_)

			df_cluster_query = pd.DataFrame(index=cluster_num_vec,columns=['inertia_'],data=np.asarray(t_list1))

			if save_mode>0:
				df_cluster_query.to_csv(output_filename,sep='\t')
				x = cluster_num_vec
				y = df_cluster_query['inertia_']
				plt.figure()
				plt.scatter(x, y, s=4)
				plt.xlabel('Number of clusters')
				plt.ylabel('SSE')
				plt.title('K-means clustering SSE')
				filename1 = output_filename
				b = filename1.find('.txt')
				output_filename_2 = '%s.png'%(filename1[0:b])
				plt.savefig(output_filename_2,format='png')

			return df_cluster_query

	## ======================================================
	# perform clustering
	def test_cluster_query_1(self,feature_mtx,n_clusters,ad=[],method_type_vec=[],neighbors=30,select_config={}):

		"""
		:param feature_mtx: (dataframe) the feature matrix of the observations (row:observation, column:variable)
		:param n_clusters: (int) the number of clusters
		:param ad: AnnData object which saves the feature matrix of the observations
		:param method_type_vec: (array or list) the methods used for clustering
		:param neighbors: (int) the parameter of the number of neighbors used in PhenoGraph clustering
		:param select_config: dictionary containing parameters
		:return: 1. AnnData object which saves the feature matrix and the group assignment (cluster labels) of the observations;
		         2. list containing the SSE of the clusters by using the specific method for clustering;
		"""

		sample_id = feature_mtx.index
		# print('please provide clustering method type')
		# model_type_vec = ['DBSCAN','OPTICS','AgglomerativeClustering','AffinityPropagation','SpectralClustering','cluster_optics_dbscan']
		model_type_vec = ['AgglomerativeClustering','AffinityPropagation','SpectralClustering']
		model_type_vec = np.asarray(model_type_vec)
		method_type_vec_1 = pd.Index(method_type_vec).intersection(model_type_vec,sort=False)
		if len(method_type_vec_1)>0:
			distance_threshold = 10
			if 'distance_thresh' in select_config:
				distance_threshold = select_config['distance_thresh']

			linkage_type_id = 1
			linkage_type_vec = ['ward','average','complete','single']
			if 'linkage_type_id' in select_config:
				linkage_type_id = select_config['linkage_type_id']
			linkage_type = linkage_type_vec[linkage_type_id]

			metric_vec = ['euclidean','l1','l2','manhattan','cosine','precomputed']
			metric_type_id = 0
			metric = metric_vec[metric_type_id]
			if 'metric_type' in select_config:
				metric = select_config['metric_type']
			metric_pre = metric
			if linkage_type=='ward':
				metric_pre = 'euclidean'
			connectivity_mtx = None
			print('metric: %s'%(metric))
			print('distance_threshold: %d'%(distance_threshold))
			if distance_threshold>=0:
				print('cluster query 1')
				cluster_model3 = AgglomerativeClustering(n_clusters=None,affinity=metric_pre,distance_threshold=distance_threshold,linkage=linkage_type,
															connectivity=connectivity_mtx,compute_full_tree=True)	# distance
			else:
				print('cluster query 2')
				cluster_model3 = AgglomerativeClustering(n_clusters=n_clusters,affinity=metric_pre,distance_threshold=None,linkage=linkage_type,
															connectivity=connectivity_mtx,compute_full_tree=True)	# distance

			# max_iter = 200
			max_iter = 500	# after using distance_thresh=10
			# cluster_model3_1 = AffinityPropagation(affinity='precomputed',max_iter=max_iter) # affinity
			cluster_model3_1 = AffinityPropagation(affinity='euclidean',max_iter=max_iter,random_state=0) # affinity
			sample_num = len(sample_id)
			cluster_model5 = SpectralClustering(n_clusters=n_clusters,random_state=0)
			list1 = [cluster_model3,cluster_model3_1,cluster_model5]
			dict_query = dict(zip(model_type_vec,list1))

		if len(ad)==0:
			ad = sc.AnnData(feature_mtx)
			ad.X = csr_matrix(ad.X)

		query_vec = []
		for method_type in method_type_vec:
			method_type_annot = select_config['method_type_annot']
			if method_type in [0,10,'MiniBatchKMeans','KMeans']:
				max_iter_num = 1000
				if 'max_iter_num_Kmeans' in select_config:
					max_iter_num = select_config['max_iter_num_Kmeans']
				n_init, reassignment_ratio = 10, 0.01
				if 'n_init' in select_config:
					n_init = select_config['n_init']
				if 'reassignment_ratio' in select_config:
					reassignment_ratio = select_config['reassignment_ratio']
				
				start=time.time()
				random_state = 0
				np.random.seed(0)
				# predictor = MiniBatchKMeans(n_clusters=n_clusters,init='k-means++',max_iter=max_iter_num,batch_size=1024,verbose=0,compute_labels=True,random_state=random_state,tol=0.0,max_no_improvement=10,init_size=None,n_init=10,reassignment_ratio=0.01)
				if method_type in [0,'MiniBatchKMeans']:
					predictor = MiniBatchKMeans(n_clusters=n_clusters,init='k-means++',max_iter=max_iter_num,batch_size=1024,verbose=0,compute_labels=True,random_state=random_state,tol=0.0,max_no_improvement=10,init_size=None,n_init=n_init,reassignment_ratio=reassignment_ratio)
				else:
					predictor = KMeans(n_clusters=n_clusters,init='k-means++',max_iter=max_iter_num,tol=0.0001,verbose=0,random_state=random_state,n_init=n_init)
				predictor = predictor.fit(feature_mtx)
				cluster_label =predictor.predict(feature_mtx)
				ad.obs[method_type_annot] = cluster_label

				stop = time.time()
				label_vec = np.unique(cluster_label)
				label_num = len(label_vec)
				# to add: the different numbers of clusters
				inertia_ = predictor.inertia_
				# print('cluster query: %.5fs %s %d %.5f'%(stop-start,method_type,label_num,inertia_))
				print('clustering used %.2fs'%(stop-start))
				print('method: %s, the number of clusters: %d, inertia_: %.5E'%(method_type,label_num,inertia_))
				query_vec.append({method_type:inertia_})

			elif method_type in [1,'phenograph']:
				# k = 30	# choose k, the number of the k nearest neibhbors
				k = neighbors # choose k, the number of the k nearest neibhbors
				sc.settings.verbose = 0
				start=time.time()
				np.random.seed(0)
				communities, graph, Q = phenograph.cluster(pd.DataFrame(feature_mtx),k=k) # run PhenoGraph
				cluster_label = pd.Categorical(communities)
				
				ad.obs[method_type_annot] = cluster_label
				ad.uns['PhenoGraph_Q'] = Q
				ad.uns['PhenoGraph_k'] = k

				stop = time.time()
				label_vec = np.unique(cluster_label)
				label_num = len(label_vec)
				print('clustering used %.2fs'%(stop-start))
				print('method: %s, the number of clusters: %d'%(method_type,label_num))
			else:
				cluster_model = dict_query[method_type]
				# print(method_type)
				start = time.time()
				x = feature_mtx
				np.random.seed(0)
				cluster_model.fit(x)
				t_labels = cluster_model.labels_
				ad.obs[method_type_annot] = t_labels
				label_vec = np.unique(t_labels)
				label_num = len(label_vec)
				stop = time.time()
				print('clustering used %.2fs'%(stop-start))
				print('method: %s, the number of clusters: %d'%(method_type,label_num))
				
		return ad, query_vec

	## ======================================================
	# perform clustering
	def test_query_group_1(self,data=[],adata=[],feature_type_query='peak',list_config=[],flag_iter=1,flag_cluster=1,
								save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		perform clustering
		:param data: (dataframe) the feature matrix of the observations (row:observation, column:variable)
		:param adata: AnnData object which saves the feature matrix of the observations
		:param feature_type_query: the type of observations (e.g., ATAC-seq peak loci)
		:param list_config: list containing configuration parameters for each method used for clustering
		:param flag_iter: indicator of whether to estimate the number of clusters
		:param flag_cluster: indicator of whether to perferm clustering
		:param save_mode: indicator of whether to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dataframe containing group assignment of the observations by using the methods specified in list_config for clustering
		"""

		filename_annot1 = filename_save_annot
		flag_cluster_1 = 1
		if flag_cluster_1 > 0:
			# n_clusters = 100
			n_clusters = 50
			neighbors = 20
			data1 = data
			ad1 = adata

			linkage_type_vec = ['ward', 'average', 'complete', 'single']
			# linkage_type_id = 1
			linkage_type_id = 0

			config_vec_1 = select_config['config_vec_1']
			n_components, type_id_compute, type_id_feature_2 = config_vec_1
			if flag_iter > 0:
				if feature_type_query == 'peak':
					method_type_query = 'MiniBatchKMeans'
				else:
					method_type_query = 'KMeans'
				
				if output_filename=='':
					filename_thresh_annot1 = '%d_%d.%d' % (n_components, type_id_compute, type_id_feature_2)
					# filename_annot_2 = '%s.%s' % (filename_annot1, filename_thresh_annot1)
					filename_annot_2 = '%s.%s' % (filename_save_annot, filename_thresh_annot1)
					output_filename = '%s/test_sse.%s.%s.txt' % (output_file_path, filename_annot_2, method_type_query)
					
				filename1 = output_filename
				if (os.path.exists(filename1) == True):
					print('the file exists: %s' % (filename1))
					df_cluster_query1 = pd.read_csv(filename1, index_col=0, sep='\t')
				else:
					df_cluster_query1 = self.test_query_cluster_pre1(adata=ad1, feature_mtx=data1,
																		method_type=method_type_query,
																		n_clusters=300,
																		neighbors=20,
																		save_mode=1, output_filename=output_filename,
																		select_config=select_config)

			flag_clustering_1 = flag_cluster
			if flag_clustering_1 > 0:
				distance_threshold_pre = -1
				linkage_type_id_pre = 0
				neighbors_pre = 20
				n_clusters_pre = 100

				if len(list_config)==0:
					neighbors_vec = [5, 10, 15, 20, 30] # the neighbors in phenograph clustering
					n_clusters_vec = [30, 50, 100] # the number of clusters
					distance_threshold_vec = [20, 50, -1] # the distance in agglomerative clustering
					linkage_type_idvec = [0]
					field_query = 'neighbors_vec', 'n_clusters_vec', 'distance_threshold_vec', 'linkage_type_idvec'
					
					list1 = [neighbors_vec, n_clusters_vec, distance_threshold_vec, linkage_type_idvec]
					dict_config1 = dict(zip(field_query, list1))
					for field1 in field_query:
						if not (field1 in select_config):
							select_config.update({field1: dict_config1[field1]})
					
					method_type_vec_query1 = ['phenograph']
					list_config = self.test_cluster_query_config_1(method_type_vec=method_type_vec_query1,
																	distance_threshold=distance_threshold_pre,
																	linkage_type_id=linkage_type_id_pre,
																	neighbors=neighbors_pre,
																	n_clusters=n_clusters_pre,
																	metric=metric,
																	select_config=select_config)

				query_num1 = len(list_config)
				for i1 in range(query_num1):
					t_vec_1 = list_config[i1]
					method_type_1, method_type_annot1, n_clusters, neighbors, distance_threshold, linkage_type_id, metric = t_vec_1
					print('method_type: ', t_vec_1)
					select_config.update({'method_type_clustering': method_type_1,
											'method_type_annot': method_type_annot1,
											'n_clusters': n_clusters,
											'neighbors': neighbors,
											'distance_thresh': distance_threshold,
											'linkage_type_id': linkage_type_id,
											'metric_type': metric})

					method_type_vec_query = [method_type_1]
					ad1, query_vec = self.test_cluster_query_1(feature_mtx=data1,
																n_clusters=n_clusters,
																ad=ad1,
																method_type_vec=method_type_vec_query,
																neighbors=neighbors,
																select_config=select_config)

					print('AnnData with group assignment of observations')
					print(ad1)
					filename_annot_2 = filename_annot1
					output_filename_1 = '%s/%s.%s.1.h5ad' % (output_file_path,filename_prefix_save,filename_save_annot)
					ad1.write(output_filename_1)
					print(ad1)
					df_obs = ad1.obs
					output_filename_2 = '%s/%s.%s.df_obs.1.txt' % (output_file_path,filename_prefix_save,filename_save_annot)
					df_obs.to_csv(output_filename_2, sep='\t')
				
				return df_obs

	## ======================================================
	# query mean and standard deviation values of signals in each cluster and cluster percentages
	def test_query_cluster_signal_1(self,feature_mtx,cluster_query=[],df_obs=[],method_type='',thresh_group_size=5,scale_type=3,type_id_compute=0,transpose=False,colname_annot=False,
										flag_ratio=0,save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):
		
		"""
		query mean and std values of signals in each cluster and cluster percentages
		:param feature_mtx: (dataframe) the feature matrix of the observations (row:observation, column:variable)
		:param cluster_query: (pandas.Series) group (cluster) assignment of observations by using the specific method for clustering
		:param df_obs: (dataframe) annotations of observations including the group assigment by using the specific method for clustering
		:param method_type: (str) the method used for clustering
		:param thresh_group_size: threshold on the group size
		:param scale_type: the type of scaling to perform on the feature matrix of the observations
		:param type_id_compute: indicator of which method to use to perform dimension reduction
		:param transpose: indicator of whether to transpose the feature matrix
		:param colname_annot: indicator of whether to rename columns of the dataframes of the mean and standard deviation values of observation features in each group
		:param flag_ratio:indicator of whether to query the number of members in each group and the percentage each group occupies in the population
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing the dataframes:
		         1,2. the mean and standard deviation values of observation features in each group (row:group, column:variable);
		         3. the scaled mean values of observation features in each group (row:group, column:variable) (if scale_type!=-1);
		         4. the number of members in each group and the percentage each group occupies in the population (if flag_ratio>0);
		"""

		flag_query1=1
		if flag_query1>0:
			if len(cluster_query)==0:
				cluster_query = df_obs[method_type]
			# cluster_query = pd.Categorical(cluster_query)

			# feature_vec = cluster_query.index
			sample_id_pre1 = cluster_query.index
			if transpose==True:
				feature_mtx = feature_mtx.T
			
			sample_id_ori = feature_mtx.index
			query_idvec = feature_mtx.columns
			sample_id = sample_id_pre1.intersection(sample_id_ori,sort=False)
			if verbose>0:
				sample_num_pre1, sample_num_ori, sample_num = len(sample_id_pre1), len(sample_id_ori), len(sample_id)
				print('sample_id_ori:%d, sample_id_pre1:%d, sample_id:%d'%(sample_num_pre1,sample_num_ori,sample_num))
			
			feature_mtx = feature_mtx.loc[sample_id,:]
			cluster_query = cluster_query.loc[sample_id]
			if type_id_compute==1:
				cluster_vec = np.unique(cluster_query)
				cluster_num = len(cluster_vec)
				df_mean = pd.DataFrame(index=query_idvec,columns=np.asarray(cluster_vec))
				df_std = pd.DataFrame(index=query_idvec,columns=np.asarray(cluster_vec))
				for i1 in range(cluster_num):
					cluster_id = cluster_vec[i1]
					sample_query = sample_id[cluster_query==cluster_id]
					mean_value = feature_mtx.loc[sample_query,:].mean(axis=0)
					std_value = feature_mtx.loc[sample_query,:].std(axis=0)
					df_mean.loc[:,i1] = mean_value
					df_std.loc[:,i1] = std_value
			else:
				feature_mtx['group'] = cluster_query
				df_mean = feature_mtx.groupby(by=['group'],sort=True).mean().T  # shape: (feature_num,group_num)
				df_std = feature_mtx.groupby(by=['group'],sort=True).std().T    # shape: (feature_num,group_num)

			method_type_annot1 = method_type
			if method_type=='':
				method_type_annot1 = 'group'
			if colname_annot==True:
				df_mean.columns = ['%s.%s'%(method_type_annot1,column_id1) for column_id1 in df_mean.columns]
				df_std.columns = ['%s.%s'%(method_type_annot1,column_id1) for column_id1 in df_std.columns]
			
			dict_feature = dict()
			annot_vec = ['mean','std','mean_scale','df_ratio']
			list1 = [df_mean,df_std]
			query_num1 = len(list1)
			annot_vec_pre1 = annot_vec[0:2]
			dict_feature = dict(zip(annot_vec_pre1,list1))
			# query scaled mean value of features
			if scale_type!=-1:
				annot1 = annot_vec[0]
				df_feature = dict_feature['mean']
				df_feature_scale = utility_1.test_motif_peak_estimate_score_scale_1(score=df_feature,
																					feature_query_vec=[],
																					scale_type_id=scale_type,
																					select_config=select_config)
				dict_feature.update({'%s_scale'%(annot1):df_feature_scale})

			if flag_ratio>0:
				df_ratio = utility_1.test_query_frequency_1(cluster_query,select_config=select_config)
				df_ratio['group_id'] = ['%s.%s'%(method_type_annot1,query_id1) for query_id1 in df_ratio.index]
				dict_feature.update({'df_ratio':df_ratio})

			if save_mode>0:
				for annot1 in annot_vec:
					df_feature = dict_feature[annot1]
					filename_annot = '%s.%s'%(annot1,method_type_annot1)
					output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_save,filename_annot)
					df_feature.to_csv(output_filename,sep='\t',float_format='%.5f')

			return dict_feature

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()


