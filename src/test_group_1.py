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

import warnings

import phenograph

import sys
from tqdm.notebook import tqdm

import os
import os.path
import sklearn

from optparse import OptionParser
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, TSNE
from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA,SparsePCA,TruncatedSVD
from sklearn.decomposition import FastICA, NMF, MiniBatchDictionaryLearning
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score, consensus_score

from sklearn.cluster import AffinityPropagation, SpectralClustering, SpectralCoclustering, AgglomerativeClustering, DBSCAN,OPTICS, cluster_optics_dbscan
from sklearn.cluster import MiniBatchKMeans, KMeans,MeanShift, estimate_bandwidth,Birch

from scipy import stats
from scipy.stats import norm
import scipy.sparse
from scipy.sparse import spmatrix
from scipy.sparse import hstack, csr_matrix, csc_matrix, issparse, vstack
from scipy import signal
from scipy.cluster.hierarchy import dendrogram, linkage

import gc
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

import utility_1
import h5py
import json
import pickle

# get_ipython().run_line_magic('matplotlib', 'inline')
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
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

class _Base2_group1(BaseEstimator):
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
		self.cell = cell
		self.generate = generate
		self.train_chromvec = chromvec
		self.chromosome = chromvec[0]

		self.path_1 = file_path
		self.config = config
		self.run_id = run_id

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

		# data_file_type = select_config['data_file_type']
		# input_file_path1 = self.path_1
		input_file_path1 = self.save_path_1

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
		self.dict_feature_query_1 = dict()
		self.dict_feature_scale_1 = dict()
		self.dict_pre_data = dict()

	## feature matrix query
	def test_query_feature_mtx_1(self,feature_type_vec=[],feature_type='peak',dict_feature_query=[],dict_feature_query2=[],dict_scale_type=[],feature_vec=[],type_id_feature=3,verbose=0,select_config={}):

		dict_feature_query_1 = self.dict_feature_query_1
		type_id_query = 0
		type_id_scale = 0
		if len(dict_feature_query2)>0:
			type_id_query = 1   # query subset of anndata

		if len(dict_scale_type)>0:
			type_id_scale = 1   # feature scale

		dict_feature_scale_1 = dict()
		feature_type_vec_2 = ['feature1','feature2','feature3']
		query_num1 = len(feature_type_vec_2)
		for feature_type_query in feature_type_vec:
			dict_query1 = dict_feature_query[feature_type_query]
			dict_query2 = dict()
			dict_feature_query_1[feature_type_query] = dict()
			dict_feature_scale_1[feature_type_query] = dict()
			# for query_id in feature_type_vec_2[0:2]:
			for i1 in range(2):
				query_id = feature_type_vec_2[i1]
				adata1 = dict_query1[query_id]
				df_feature = []
				if len(adata1)>0:
					if type_id_query>0:
						feature_vec = dict_feature_query2[feature_type_query]
						adata1 = adata1[feature_vec,:]	# query subset of anndata
					
					try:
						df_feature = pd.DataFrame(index=adata1.obs_names,columns=adata1.var_names,data=adata1.X.toarray(),dtype=np.float32)
					except Exception as error:
						print('error! ',error)
						df_feature = pd.DataFrame(index=adata1.obs_names,columns=adata1.var_names,data=np.asarray(adata1.X),dtype=np.float32)

					if (type_id_scale>0) and (feature_type_query in dict_scale_type):
						scale_type_vec = dict_scale_type[feature_type_query]
						scale_type = scale_type_vec[i1]
						if scale_type>0:
							df_feature_pre1 = utility_1.test_motif_peak_estimate_score_scale_1(score=df_feature,feature_query_vec=[],
																								select_config=select_config,
																								scale_type_id=scale_type)
							dict_feature_scale_1[feature_type_query].update({query_id:df_feature_pre1})
						
				dict_feature_query_1[feature_type_query].update({query_id:df_feature})

		self.dict_feature_query_1 = dict_feature_query_1
		self.dict_feature_scale_1 = dict_feature_query_1

		return dict_feature_query_1

	## query the number of components
	def test_query_component_1(self,feature_mtx,adata=[],type_id_feature=0,type_id_compute=0,normalize_type=0,zero_center=False,thresh_cml_var=0.9,flag_plot=1,save_mode=1,output_file_path='',output_filename='',filename_save_annot='',verbose=0,select_config={}):

		flag_query1=1
		if flag_query1>0:
			# ad2 = ad1.copy()
			adata2 = adata.copy()
			filename_annot1 = filename_save_annot
			if type_id_compute in [0,2]:
				n_components_2 = feature_mtx.shape[1]-1
			else:
				n_components_2 = feature_mtx.shape[1]
			# num_vec_1 = np.arange(1,n_components_2+1)
			feature_mtx_2, dimension_model_2, query_vec_2, adata2 = self.dimension_reduction_2(feature_mtx=feature_mtx,
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
			# x = range(len(adata.uns['pca']['variance_ratio']))

			feature_mtx_2 = csr_matrix(feature_mtx_2)
			layer_name_pre1 = 'layer%d'%(type_id_compute)
			adata2.obsm[layer_name_pre1] = feature_mtx_2
			if flag_plot>0:
				n_rows, n_cols = 1,2
				tol_1, tol_2 = 2,0
				size_col, size_row = 5,5
				fig, axes = plt.subplots(n_rows, n_cols, sharex=False, sharey=False,
											figsize=(n_cols * size_col + tol_1, n_rows * size_row + tol_2))
				# plt.figure(0)
				x = num_vec
				y = cml_var_explained
				ax_1 = axes[0]
				ax_1.scatter(x, y, s=4)
				ax_1.set_xlabel('PC')
				ax_1.set_ylabel('Cumulative variance explained')
				ax_1.set_title('Cumulative variance explained by PCs')
				# plt.show()
				# output_filename = '%s/test_cml_var_explained.%s.%d.%d.png'%(output_file_path,filename_annot1,type_id_feature,type_id_compute)
				# plt.savefig(output_filename,format='png')

				x = num_vec
				explained_variance = df_component_query['explained_variance']
				y = explained_variance
				# plt.figure(1)
				ax_1 = axes[1]
				ax_1.scatter(x, y, s=4)
				ax_1.set_xlabel('PC')
				ax_1.set_ylabel('Explained variance')
				ax_1.set_title('Variance explained by PCs')
				# plt.show()
				output_filename_1 = '%s/test_var_explained.%s.%d.%d.png'%(output_file_path,filename_annot1,type_id_feature,type_id_compute)
				plt.savefig(output_filename_1,format='png')

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

	## query explained variance threshold
	def test_query_component_2(self,df=[],input_filename='',thresh_cml_var=0.90,thresh_cml_var_2=0.80,n_components_default=[50,300],flag_default=1,verbose=0,select_config={}):

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
			print('n_components_query1:%d, cml_var_explained:%.5f' % (n_components_thresh1, thresh_cml_var))
			print('n_components_query2:%d, cml_var_explained:%.5f' % (n_components_thresh2, thresh_cml_var_2))

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

	# dimension reduction methods
	def dimension_reduction(self,x_ori,feature_dim,type_id,shuffle=False,sub_sample=-1,filename_prefix='',filename_load='test1'):

		idx = np.asarray(range(0,x_ori.shape[0]))
		if (sub_sample>0) and (type_id!=7) and (type_id!=11):
			id1 = idx[0:sub_sample]
		else:
			id1 = idx

		if type_id==0:
			# PCA
			pca = PCA(n_components=feature_dim, whiten = False, random_state = 0)
			if sub_sample>0:
				pca.fit(x_ori[id1,:])
				x = pca.transform(x_ori)
			else:
				x = pca.fit_transform(x_ori)
			dimension_model = pca
			# X_pca_reconst = pca.inverse_transform(x)
		elif type_id==1:
			# Incremental PCA
			n_batches = 10
			inc_pca = IncrementalPCA(n_components=feature_dim)
			for X_batch in np.array_split(x_ori, n_batches):
				inc_pca.partial_fit(X_batch)
			x = inc_pca.transform(x_ori)
			dimension_model = inc_pca
			# X_ipca_reconst = inc_pca.inverse_transform(x)
		elif type_id==2:
			# Kernel PCA
			kpca = KernelPCA(kernel="rbf",n_components=feature_dim, gamma=None, fit_inverse_transform=True, random_state = 0, n_jobs=50)
			kpca.fit(x_ori[id1,:])
			x = kpca.transform(x_ori)
			dimension_model = kpca
			# X_kpca_reconst = kpca.inverse_transform(x)
		elif type_id==3:
			# Sparse PCA
			sparsepca = SparsePCA(n_components=feature_dim, alpha=0.0001, random_state=0, n_jobs=50)
			sparsepca.fit(x_ori[id1,:])
			x = sparsepca.transform(x_ori)
			dimension_model = sparsepca
		elif type_id==4:
			# SVD
			SVD_ = TruncatedSVD(n_components=feature_dim,algorithm='randomized', random_state=0, n_iter=10)
			SVD_.fit(x_ori[id1,:])
			x = SVD_.transform(x_ori)
			dimension_model = SVD_
			# X_svd_reconst = SVD_.inverse_transform(x)
		elif type_id==5:
			# Gaussian Random Projection
			GRP = GaussianRandomProjection(n_components=feature_dim,eps = 0.5, random_state=2019)
			GRP.fit(x_ori[id1,:])
			x = GRP.transform(x_ori)
			dimension_model = GRP
		elif type_id==6:
			# Sparse random projection
			SRP = SparseRandomProjection(n_components=feature_dim,density = 'auto', eps = 0.5, random_state=2019, dense_output = False)
			SRP.fit(x_ori[id1,:])
			x = SRP.transform(x_ori)
			dimension_model = SRP
		elif type_id==7:
			# MDS
			mds = MDS(n_components=feature_dim, n_init=12, max_iter=1200, metric=True, n_jobs=4, random_state=2019)
			x = mds.fit_transform(x_ori[id1])
			dimension_model = mds
		elif type_id==8:
			# ISOMAP
			isomap = Isomap(n_components=feature_dim, n_jobs = 4, n_neighbors = 5)
			isomap.fit(x_ori[id1,:])
			x = isomap.transform(x_ori)
			dimension_model = isomap
		elif type_id==9:
			# MiniBatch dictionary learning
			miniBatchDictLearning = MiniBatchDictionaryLearning(n_components=feature_dim,batch_size = 1000,alpha = 1,n_iter = 25,  random_state=2019)
			if sub_sample>0:
				miniBatchDictLearning.fit(x_ori[id1,:])
				x = miniBatchDictLearning.transform(x_ori)
			else:
				x = miniBatchDictLearning.fit_transform(x_ori)
			dimension_model = miniBatchDictLearning
		elif type_id==10:
			# ICA
			fast_ICA = FastICA(n_components=feature_dim, algorithm = 'parallel',whiten = True,max_iter = 100,  random_state=2019)
			if sub_sample>0:
				fast_ICA.fit(x_ori[id1])
				x = fast_ICA.transform(x_ori)
			else:
				x = fast_ICA.fit_transform(x_ori)
			dimension_model = fast_ICA
			# X_fica_reconst = FastICA.inverse_transform(x)
			# elif type_id==11:
			# 	# t-SNE
			# 	tsne = TSNE(n_components=feature_dim,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
			# 	x = tsne.fit_transform(x_ori)
		elif type_id==12:
			# Locally linear embedding
			lle = LocallyLinearEmbedding(n_components=feature_dim, n_neighbors = np.max((int(feature_dim*1.5),500)),method = 'modified', n_jobs = 20,  random_state=2019)
			lle.fit(x_ori[id1,:])
			x = lle.transform(x_ori)
			dimension_model = lle
		elif type_id==13:
			# Autoencoder
			feature_dim_ori = x_ori.shape[1]
			m = Sequential()
			m.add(Dense(512,  activation='elu', input_shape=(feature_dim_ori,)))
			# m.add(Dense(256,  activation='elu'))
			m.add(Dense(feature_dim,   activation='linear', name="bottleneck"))
			# m.add(Dense(256,  activation='elu'))
			m.add(Dense(512,  activation='elu'))
			m.add(Dense(feature_dim_ori,  activation='sigmoid'))
			m.compile(loss='mean_squared_error', optimizer = Adam())
			history = m.fit(x_ori[id1], x_ori[id1], batch_size=256, epochs=20, verbose=1)

			encoder = Model(m.input, m.get_layer('bottleneck').output)
			x = encoder.predict(x_ori)
			Renc = m.predict(x_ori)
			dimension_model = encoder

		# save transfrom model
		if filename_prefix=='':
			filename1 = '%s_%d_dimensions.h5'%(filename_load,feature_dim)
		else:
			filename1 = '%s_%d_%d_dimensions.h5'%(filename_prefix,type_id,feature_dim)
			
		# np.save(filename1, self.dimension_model)
		pickle.dump(dimension_model, open(filename1, 'wb'))
		# self.dimension_model = pickle.load(open(filename1, 'rb'))
		return x, dimension_model

	## dimension reduction
	# dimension reduction using methods: TruncatedSVD 1, TruncatedSVD 2, PCA (2: from scanpy, 3: from sklearn)
	def dimension_reduction_2(self,feature_mtx,ad=[],n_components=100,type_id_compute=0,normalize_type=0,zero_center=False,copy=False,save_mode=1,output_file_path='',output_filename='',select_config={}):

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
			# ad.obs_names = peak_read.columns
			# ad.var_names = peak_read.index
			np.random.seed(0)
			t0 = time.time()
			# sc.tl.pca(ad,n_comps=n_components,zero_center=False,use_highly_variable=False)
			sc.tl.pca(ad,n_comps=n_components,zero_center=zero_center,use_highly_variable=False,copy=copy)
			feature_mtx_transform = ad.obsm['X_pca']
			explained_variance = np.sum(ad.uns['pca']['variance_ratio'])
			print('feature_mtx_transform: ',feature_mtx_transform.shape)
			print('type_id_compute: %d'%(type_id_compute))
			print(f"PCA performed in {time.time() - t0:.3f} s")
			print(f"Explained variance of the PCA step: {explained_variance * 100:.1f}%")
			# output_filename_1 = '%s/test_query_feature_dimension.1.h5ad' % (output_file_path)
			# ad.write(output_filename_1)
			# print(ad)
		else:
			pca = PCA(n_components=n_components, whiten = False, random_state = 0)
			t0 = time.time()
			feature_mtx_transform = pca.fit_transform(feature_mtx)
			dimension_model = pca
			explained_variance = dimension_model.explained_variance_ratio_.sum()
			print('feature_mtx_transform: ',feature_mtx_transform.shape)
			print(f"PCA performed in {time.time() - t0:.3f} s")
			print(f"Explained variance of the PCA step: {explained_variance * 100:.1f}%")
			# to add: different numbers of clusters
		
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

	## clustering configuration query
	def test_cluster_query_config_1(self,method_type_vec,distance_threshold=-1,linkage_type_id=0,neighbors=20,metric='euclidean',n_clusters=100,select_config={}):

		list_config = []
		distance_threshold_pre = distance_threshold
		linkage_type_id_pre = linkage_type_id
		neighbors_pre = neighbors
		n_clusters_pre = n_clusters
		metric = metric
		neighbors_vec = [5,10,15,20,30]
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
				# for neighbors in [5,7,10,15,20,30]:
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

	## query the number of clusters
	def test_query_cluster_pre1(self,adata=[],feature_mtx=[],method_type='MiniBatchKMeans',n_clusters=300,cluster_num_vec=[],neighbors=20,save_mode=0,output_filename='',verbose=0,select_config={}):

		flag_query1=1
		if flag_query1>0:
			# method_type = 'MiniBatchKMeans'
			method_type_vec = [method_type]
			# n_clusters_pre = 300
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
																	ad=adata.copy(),
																	n_clusters=n_clusters_query,
																	method_type_vec=method_type_vec,
																	neighbors=neighbors,
																	select_config=select_config)
				# method_type = method_type_vec[0]
				if len(query_vec)>0:
					inertia_ = query_vec[0][method_type]
					t_list1.append(inertia_)

			df_cluster_query = pd.DataFrame(index=cluster_num_vec,columns=['inertia_'],data=np.asarray(t_list1))
			# filename_annot1_1 = '%s.%d_%d'%(filename_annot1,type_id_feature_2,type_id_compute)
			# output_filename = '%s/test_sse.%s.%s.txt'%(output_file_path,filename_annot1_1,method_type_2)
			if save_mode>0:
				df_cluster_query.to_csv(output_filename,sep='\t')
				x = cluster_num_vec
				y = df_cluster_query['inertia_']
				plt.figure()
				plt.scatter(x, y, s=4)
				plt.xlabel('Number of clusters')
				plt.ylabel('SSE')
				plt.title('K-means clustering SSE')
				# plt.show()
				# output_filename = '%s/test_sse.%s.%s.png'%(output_file_path,filename_annot1_1,method_type_2)
				filename1 = output_filename
				b = filename1.find('.txt')
				output_filename_2 = '%s.png'%(filename1[0:b])
				plt.savefig(output_filename_2,format='png')

			return df_cluster_query

	## clustering analysis
	def test_cluster_query_1(self,feature_mtx,n_clusters,ad=[],method_type_vec=[],neighbors=30,select_config={}):

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
				# print('cluster query: %.5fs %s %d'%(stop-start,method_type,label_num))
				## to add: the different numbers of clusters
				inertia_ = predictor.inertia_
				print('cluster query: %.5fs %s %d %.5f'%(stop-start,method_type,label_num,inertia_))
				query_vec.append({method_type:inertia_})

			elif method_type in [1,'phenograph']:
				# k = 30	# choose k, the number of the k nearest neibhbors
				k = neighbors # choose k, the number of the k nearest neibhbors
				sc.settings.verbose = 0
				start=time.time()
				np.random.seed(0)
				communities, graph, Q = phenograph.cluster(pd.DataFrame(feature_mtx),k=k) # run PhenoGraph
				cluster_label = pd.Categorical(communities)
				# ad.obs['PhenoGraph_clusters'] = cluster_label
				ad.obs[method_type_annot] = cluster_label
				ad.uns['PhenoGraph_Q'] = Q
				ad.uns['PhenoGraph_k'] = k
				# df = pd.DataFrame(index=sample_id,columns=[method_type])
				# df[method_type] = cluster_label
				stop = time.time()
				label_vec = np.unique(cluster_label)
				label_num = len(label_vec)
				print('cluster query: %.5fs %s %d'%(stop-start,method_type,label_num))
			else:
				cluster_model = dict_query[method_type]
				print(method_type)
				start = time.time()
				x = feature_mtx
				np.random.seed(0)
				cluster_model.fit(x)
				t_labels = cluster_model.labels_
				ad.obs[method_type_annot] = t_labels
				label_vec = np.unique(t_labels)
				label_num = len(label_vec)
				stop = time.time()
				# print(method_type,label_num,stop-start)
				print('cluster query: %.5fs %s %d'%(stop-start,method_type,label_num))

		return ad, query_vec

	## cluster plot query
	def test_cluster_query_plot_1(self,adata,column_id,n_pcs=100,n_neighbors=20,layer_name_query='X_svd',palatte_name='tab20',title='Group',flag_umap=1,flag_tsne=1,flag_fdl=1,save_mode=1,filename_prefix_save='',select_config={}):

			if (flag_umap>0) or (flag_fdl>0):
				try:
					sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=layer_name_query)
				except Exception as error:
					print('error! ',error)
					return
			
			matplotlib.rcParams['figure.figsize'] = [3.5,3.5]
			if flag_umap>0:
				# sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=layer_name_query)
				sc.tl.umap(adata)
				# sc.pl.umap(adata,color=[column_id],palette='tab20', # 'palette' specifies the colormap to use)
				# 									title=["Clusters"])
				sc.pl.umap(adata,color=[column_id],palette=palatte_name, # 'palette' specifies the colormap to use)
													title=[title])

				if save_mode>0:
					# b = input_filename.find('.h5ad')
					# output_filename = input_filename[0:b]+'.umap.%d.%d.1.png'%(n_pcs,method_type_id1)
					# method_type_id1 = column_id
					# output_filename = filename_prefix_save+'.umap.%d.%s.1.png'%(n_pcs,method_type_id1)
					output_filename = filename_prefix_save+'.umap.%d.%s.1.png'%(n_pcs,column_id)
					plt.savefig(output_filename,format='png')

			if flag_tsne>0:
				# sc.pp.neighbors(adata, n_pcs=n_pcs, use_rep='X_svd')
				# sc.tl.tsne(adata,use_rep='X_svd')
				sc.tl.tsne(adata,use_rep=layer_name_query)
				sc.pl.tsne(adata,use_raw=False,color=[column_id],palette=palatte_name, # 'palette' specifies the colormap to use)
																	title=[title])
				if save_mode>0:
					output_filename = filename_prefix_save+'.tsne.%d.%s.1.png'%(n_pcs,column_id)
					plt.savefig(output_filename,format='png')

			if flag_fdl>0:
				sc.tl.draw_graph(adata)
				sc.pl.draw_graph(adata,color=[column_id],palette=palatte_name, # 'palette' specifies the colormap to use)
															title=[title])

				if save_mode>0:
					output_filename = filename_prefix_save+'.fdl.%d.%s.1.png'%(n_pcs,column_id)
					plt.savefig(output_filename,format='png')

			return True

	# query cluster mean value, std value and cluster frequency for each clustering prediction
	# return: dict_feature: {'mean','std','mean_scale','df_ratio'}
	# to update: combine different fields into anndata
	def test_query_cluster_signal_1(self,feature_mtx,cluster_query=[],df_obs=[],method_type='',thresh_group_size=5,scale_type=3,type_id_query=1,type_id_compute=0,transpose=False,colname_annot=False,
										flag_plot=0,flag_ratio=0,filename_prefix='',filename_prefix_save='',save_mode=0,output_file_path='',output_filename='',verbose=0,select_config={}):
		
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
			# print(df_mean)
			# print(df_std)

			dict_feature = dict()
			annot_vec = ['mean','std','mean_scale','df_ratio']
			list1 = [df_mean,df_std]
			query_num1 = len(list1)
			annot_vec_pre1 = annot_vec[0:2]
			dict_feature = dict(zip(annot_vec_pre1,list1))
			# scaled mean value
			if scale_type!=-1:
				annot1 = annot_vec[0]
				df_feature = dict_feature['mean']
				df_feature_scale = utility_1.test_motif_peak_estimate_score_scale_1(score=df_feature,feature_query_vec=[],select_config=select_config,scale_type_id=scale_type)
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

			# return True
			return dict_feature

	## cluseter matching
	def test_cluster_query_overlap_1(self,method_type_vec,df_cluster,save_mode=1,output_file_path='',filename_prefix_save='',output_filename_1='',output_filename_2='',verbose=0,select_config={}):

		flag_query1=1
		if flag_query1>0:
			method_type_1 = method_type_vec[0]
			# df_group1 = df_obs[method_type_1]
			df_group1 = df_cluster[method_type_1]
			dict_query = dict()
			for method_type_2 in method_type_vec[1:]:
				# df_group2 = df_obs[method_type_2]
				df_group2 = df_cluster[method_type_2]
				group_vec_1, group_vec_2 = np.unique(df_group1), np.unique(df_group2)
				group_num1, group_num2 = len(group_vec_1), len(group_vec_2)
				
				if verbose>0:
					print('group_vec_1: %d, group_vec_2: %d'%(group_num1,group_num2))
				df_count_pre = pd.DataFrame(index=group_vec_2,columns=group_vec_1,data=np.float32)
				for l1 in range(group_num1):
					group_id1 = group_vec_1[l1]
					for l2 in range(group_num2):
						group_id2 = group_vec_2[l2]
						count1 = np.sum((df_group1==group_id1)&(df_group2==group_id2))
						# print('count:%d, %s, %s'%(count1,group_id1,group_id2))
						df_count_pre.loc[group_id2,group_id1] = count1

				if verbose>0:
					print(df_count_pre)
				df_ratio_pre = df_count_pre/np.outer(df_count_pre.sum(axis=1),np.ones(df_count_pre.shape[1]))
				
				if save_mode>0:
					df_count_pre.to_csv(output_filename_1,sep='\t',float_format='%d')
					df_ratio_pre.to_csv(output_filename_2,sep='\t',float_format='%.5f')

				dict_query.update({(method_type_2,method_type_1):[df_count_pre,df_ratio_pre]})

			return dict_query

	## feature group estimation
	def test_query_group_1(self,data=[],adata=[],feature_type_query='peak',list_config=[],flag_iter_2=1,flag_clustering_1=1,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_clustering = 1
		# filename_annot_save = select_config['filename_annot_save']
		filename_annot_save = filename_save_annot
		filename_annot1 = filename_annot_save
		
		if flag_clustering > 0:
			# n_clusters = 100
			n_clusters = 50
			# neighbors = 30
			neighbors = 20
			data1 = data
			ad1 = adata
			linkage_type_vec = ['ward', 'average', 'complete', 'single']
			# distance_threshold = 20
			# linkage_type_id = 1
			linkage_type_id = 0
			# metric = 'euclidean'
			# group_type = feature_type_query
			config_vec_1 = select_config['config_vec_1']
			n_components, type_id_compute, type_id_feature_2 = config_vec_1
			# flag_iter_2 = 1
			if flag_iter_2 > 0:
				if feature_type_query == 'peak':
					method_type_query = 'MiniBatchKMeans'
				else:
					method_type_query = 'KMeans'
				
				if output_filename=='':
					filename_thresh_annot1 = '%d_%d.%d' % (n_components, type_id_compute, type_id_feature_2)
					filename_annot_2 = '%s.%s' % (filename_annot1, filename_thresh_annot1)
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

			# flag_clustering_1 = 1
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
					
					list_config = self.test_cluster_query_config_1(method_type_vec=method_type_vec_1,
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
					ad1, query_vec = self.test_cluster_query_1(feature_mtx=data1,ad=ad1,n_clusters=n_clusters,
																method_type_vec=method_type_vec_query,neighbors=neighbors,
																select_config=select_config)
					print(ad1)

					filename_thresh_annot1 = '%d_%d.%d' % (n_components, type_id_compute, type_id_feature_2)
					filename_annot_2 = '%s.%s' % (filename_annot1, filename_thresh_annot1)
					output_filename_1 = '%s/%s.feature_dimension.%s.1.h5ad' % (output_file_path,filename_prefix_save,filename_annot_2)
					ad1.write(output_filename_1)
					print(ad1)
					df_obs = ad1.obs
					# output_filename_2 = '%s/test_query_feature_dimension.%s.df_obs.1.txt' % (output_file_path, filename_annot_2)
					output_filename_2 = '%s/%s.feature_dimension.%s.df_obs.1.txt' % (output_file_path,filename_prefix_save,filename_annot_2)
					df_obs.to_csv(output_filename_2, sep='\t')
				
				return df_obs

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)


