#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.switch_backend('Agg')
import seaborn as sns

import warnings
import sys

import os
import os.path
from optparse import OptionParser

import sklearn
from sklearn.base import BaseEstimator
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
from sklearn.pipeline import make_pipeline
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, issparse

import time
from timeit import default_timer as timer

# import utility_1
from . import utility_1
import pickle

class _Base_pre1(BaseEstimator):
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

		# initialization
		self.run_id = run_id
		self.cell = cell
		self.generate = generate
		self.chromvec = chromvec

		self.path_1 = file_path
		self.save_path_1 = file_path
		self.config = config
		self.run_id = run_id

		self.pre_rna_ad = []
		self.pre_atac_ad = []
		self.fdl = []
		self.motif_data = []
		self.motif_data_score = []
		self.motif_query_name_expr = []
		
		if not ('type_id_feature' in select_config):
			select_config.update({'type_id_feature':type_id_feature})
		
		self.select_config = select_config
		self.gene_name_query_expr_ = []
		self.gene_highly_variable = []
		self.peak_dict_ = []
		self.df_gene_peak_ = []
		self.df_gene_peak_list_ = []
		self.motif_data = []
		
		self.df_tf_expr_corr_list_pre1 = []	# TF-TF expr correlation
		self.df_expr_corr_list_pre1 = []	# gene-TF expr correlation
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
		self.verbose_internal = 2

	## ====================================================
	# update parameters in the parameter dictionary
	def test_config_query_1(self,input_filename_1='',input_filename_2='',input_file_path='',save_mode=1,filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		update parameters in the parameter dictionary;
		including paths of the single RNA-seq and ATAC-seq data
		:param input_filename_1: (str) path of the single cell RNA-seq data
		:param input_filename_2: (str) path of the single cell ATAC-seq data
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""

		print('test_config_query')
		if 'data_file_type' in select_config:
			data_file_type = select_config['data_file_type']
			print('data_file_type: %s'%(data_file_type))
			
			select_config.update({'input_filename_rna':input_filename_1,
									'input_filename_atac':input_filename_2})
			
			print('input_filename_rna:%s, input_filename_atac:%s'%(input_filename_1,input_filename_2))
			
			if not('data_path' in select_config):
				select_config.update({'data_path':input_file_path})
			
			filename_save_annot_1 = filename_save_annot
			if not('filename_save_annot_1' in select_config):
				select_config.update({'filename_save_annot_1':filename_save_annot_1})
				select_config.update({'filename_save_annot_pre1':filename_save_annot_1})

		self.select_config = select_config

		return select_config

	## ====================================================
	# update parameters in the parameter dictionary
	def test_file_path_query_2(self,input_filename_1='',input_filename_2='',input_file_path='',type_id_feature=0,run_id=1,save_mode=1,filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		update parameters in the parameter dictionary
		:param input_filename_1: (str) path of the single cell RNA-seq data
		:param input_filename_2: (str) path of the single cell ATAC-seq data
		:param input_file_path: the directory to retrieve data from
		:param type_id_feature: the feature type based on which to estimate metecells (0:RNA-seq data; 1:ATAC-seq data)
		:param run_id: index of the current computation
		:param save_mode: indicator of whether to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""

		input_file_path1 = self.save_path_1
		data_file_type = select_config['data_file_type']

		filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)
		select_config.update({'data_path':input_file_path,
								'filename_save_annot_1':filename_save_annot_1,
								'filename_save_annot_pre1':filename_save_annot_1})

		select_config.update({'input_filename_rna':input_filename_1,
								'input_filename_atac':input_filename_2})

		return select_config

	## ====================================================
	# perform metacell estimation using SEACells method
	def test_metacell_compute_unit_1(self,adata,feature_type_id=0,normalize=True,zero_center=True,highly_variable_query=True,use_highly_variable=True,n_SEACells=500,
										obsm_build_kernel='X_pca',pca_compute=1,num_components=50,n_waypoint_eigs=10,waypoint_proportion=1.0,plot_convergence=1,
										save_mode=1,select_config={}):

		"""
		perform metacell estimation using SEACells method
		:param adata: AnnData object of RNA-seq or ATAC-seq data of single cells
		:param feature_type_id: the feature type of data (0:RNA-seq data; 1:ATAC-seq data)
		:param normalize: indicator of whether to normalize the RNA-seq or ATAC-seq count matrix
		:param zero_center: indicator of whether to perform zero-centering of the variables for PCA
		:param highly_variable_query: indicator of whether to query highly variable genes or peaks
		:param use_highly_variable: indicator of whether to highly variable genes or peaks only to perform PCA
		:param n_SEACells: (int) the number or SEACells to estimate
		:param obsm_build_kernel: (str) key in .obsm of the AnnData object to use for computing metacells
		:param pca_compute: (int) indicator of whether to perform PCA
		:param num_components: (int) the number of principal components to compute in PCA
		:param n_waypoint_eigs: (int) number of eigenvalues to consider when initializing metacells
		:param waypoint_proportion: (float) the proportion of SEACells to initialize using waypoint analysis
		:param plot_convergence: indicator of whether to plot the convergence curve
		:param save_mode: indicator of whether to save data
		:param select_config: dictionary containing parameters
		:return: 1. updated RNA-seq or ATAC-seq AnnData object of single cells, with SEACell assignment saved in .obs dataframe;
		         2. the SEACells model learned to estimate SEACells
		"""

		ad = adata
		if normalize>0:
			# print('normalization ',ad.shape)
			print('normalization, anndata of size ',ad.shape)
			raw_ad = sc.AnnData(ad.X)
			raw_ad.obs_names, raw_ad.var_names = ad.obs_names, ad.var_names
			ad.raw = raw_ad
			sc.pp.normalize_per_cell(ad)
			sc.pp.log1p(ad)

		if highly_variable_query>0:
			num_top_genes = 3000
			if 'num_top_genes' in select_config:
				num_top_genes = select_config['num_top_genes']
			
			if ('log1p' in ad.uns_keys()):
				if not ('base' in ad.uns['log1p']):
					base_value = np.e   # use natural logarithm
					print('adata log1p: ',ad.uns['log1p'])
					print('adata log1p base: ',base_value)
					ad.uns['log1p']['base'] = base_value
			sc.pp.highly_variable_genes(ad,n_top_genes=num_top_genes)

		if pca_compute>0:
			print('PCA computation ',ad.shape)
			if feature_type_id==1:
				zero_center = False
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
		
		layer_name_query1 = 'X_umap'
		if (flag_umap>0):
			print('umap estimation')
			layer_name_vec = ['X_pca','X_svd']
			layer_name_query = layer_name_vec[feature_type_id]
			print('search for neighbors; neighbor number: %d, PC number: %d, using layer %s'%(n_neighbors,n_pcs,layer_name_query))
			sc.pp.neighbors(ad, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=layer_name_query)
			sc.tl.umap(ad)

		import SEACells
		build_kernel_on = obsm_build_kernel # key in ad.obsm to use for computing metacells
											# This would be replaced by 'X_svd' for ATAC data
		print('build_kernel_on, n_waypoint_eigs ',build_kernel_on,n_waypoint_eigs)

		# perform SEACell estimation
		model = SEACells.core.SEACells(ad, 
									  build_kernel_on=build_kernel_on, 
									  n_SEACells=n_SEACells, 
									  n_waypoint_eigs=n_waypoint_eigs,
									  convergence_epsilon=1e-5)

		model.construct_kernel_matrix()
		M = model.kernel_matrix
		sel_num1 = 100
		sns.clustermap(M.toarray()[0:sel_num1,0:sel_num1])
		
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
			layer_name_vec = ['X_pca','X_svd']
			layer_name_query = layer_name_vec[feature_type_id]

			n_neighbors = 15
			n_pcs = 50
			print('search for neighbors; neighbor number: %d, PC number: %d, using layer %s'%(n_neighbors,n_pcs,layer_name_query))
			sc.pp.neighbors(ad, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=layer_name_query)
			sc.tl.umap(ad)
		SEACells.plot.plot_initialization(ad, model, save_as=output_filename)

		min_iter=10
		max_iter=200
		if 'min_iter' in select_config:
			min_iter=select_config['min_iter']
		if 'max_iter' in select_config:
			max_iter=select_config['max_iter']
		print('min_iter:%d, max_iter:%d'%(min_iter,max_iter))
		
		model.fit(min_iter=min_iter, max_iter=max_iter)
		flag_addition_iteration=1
		if flag_addition_iteration>0:
			# run additional iterations step-wise using the .step() function
			print(f'Ran for {len(model.RSS_iters)} iterations')
			for _ in range(5):
				model.step()
			print(f'Ran for {len(model.RSS_iters)} iterations')

		output_filename = '%s/test_%s_converge_1.%d.png'%(output_file_path,filename_prefix,run_id)
		model.plot_convergence(save_as=output_filename)

		return ad, model

	## ====================================================
	# from SEACells.genescores
	# add variable annotations to the ATAC-seq AnnData object of the metacells
	def _add_atac_meta_data(self,atac_meta_ad, atac_ad, n_bins_for_gc):

		"""
		add variable annotations to the ATAC-seq AnnData object of the metacells (GC content bin;counts bin)
		:param atac_meta_ad: ATAC-seq seq AnnData object of the metacells
		:param atac_ad: ATAC-seq seq AnnData object of the single cells
		:param n_bins_for_gc: (int) the number of GC content bins and counts bins
		:return: None
		"""

		atac_ad.var['log_n_counts'] = np.ravel(np.log10(atac_ad.X.sum(axis=0)))
		
		if 'GC' in atac_ad.var:
			atac_meta_ad.var['GC_bin'] = np.digitize(atac_ad.var['GC'], np.linspace(0, 1, n_bins_for_gc))
		atac_meta_ad.var['counts_bin'] = np.digitize(atac_ad.var['log_n_counts'],
													 np.linspace(atac_ad.var['log_n_counts'].min(),
																 atac_ad.var['log_n_counts'].max(), 
																 n_bins_for_gc))
	
	## ====================================================
	# from SEACells.genescores
	# normlize counts per metacell or cell and perform log-transformation
	def _normalize_ad(self,meta_ad,target_sum=None,save_raw=True,save_normalize=False):

		"""
		normlize counts per metacell or cell and perform log-transformation
		:param meta_ad: RNA-seq or ATAC-seq AnnData object of the metacells or cells
		:param target_sum: the total count over all genes or peaks in each metacell or cell after normalization;
						   if target_sum=None, target_num will the median of total counts for the metacells or cells before normalization;
		:param save_raw: indicator of whehter to save the raw count matrix in the AnnData object
		:param save_normalize: indicator of whether to save the normalized count matrix before log-transformation
		:return: None
		"""

		if save_raw:
			# Save the raw count matrix in raw
			meta_ad.raw = meta_ad.copy()

		# Normalize 
		# sc.pp.normalize_total(meta_ad, key_added='n_counts')
		sc.pp.normalize_total(meta_ad, target_sum=target_sum, key_added='n_counts')
		if save_normalize==True:
			meta_ad.layers['normalize'] = meta_ad.X.copy() # save the normalized read count without log-transformation
		sc.pp.log1p(meta_ad)  # perform log-transformation of the normalized count matrix

	## ====================================================
	# from SEACells.genescores
	# create metacell AnnData objects from single-cell AnnData objects for multiome data
	def _prepare_multiome_anndata(self,atac_ad,rna_ad,SEACells_label='SEACell',summarize_layer_atac='X',summarize_layer_rna='X',flag_normalize=1,save_raw_ad_atac=1,save_raw_ad_rna=1,n_bins_for_gc=50,type_id_1=0,select_config={}):
		
		"""
		Function to create metacell AnnData objects from single-cell AnnData objects for multiome data
		:param atac_ad: (AnnData) ATAC AnnData object with raw peak counts in `X`. These anndata objects should be constructed 
		 using the example notebook available in 
		:param rna_ad: (AnnData) RNA AnnData object with raw gene expression counts in `X`. Note: RNA and ATAC anndata objects 
		 should contain the same set of cells
		:param SEACells_label: (str) `atac_ad.obs` field for constructing metacell matrices. Same field will be used for 
		  summarizing RNA and ATAC metacells. 
		:param n_bins_gc: (int) Number of bins for creating GC bins of ATAC peaks.
		:param select_config: dictionary containing parameters
		:return: ATAC metacell AnnData object and RNA metacell AnnData object
		"""

		import SEACells
		from SEACells import core, genescores
		
		# Subset of cells common to ATAC and RNA
		common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names)
		if len(common_cells) != atac_ad.shape[0]:
			print('Warning: The number of cells in RNA and ATAC objects are different. Only the common cells will be used.')
		atac_mod_ad = atac_ad[common_cells, :]
		rna_mod_ad = rna_ad[common_cells, :]

		# #################################################################################
		# Generate metacell matrices
		# Set of metacells
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
		# the raw ad was saved using the SEACells.core.summarize_by_SEACell() function
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

	## ====================================================
	# query highly variable genes or peaks
	def test_query_metacell_high_variable_feature(self,pre_meta_ad,highly_variable_feature_query=True,select_config={}):
		
		"""
		query highly variable genes or peaks
		:param pre_meta_ad: (AnnData object) gene expression or peak accessibility data of the metacells
		:param highly_variable_feature_query: indicator of whether to estimate highly variable peaks or genes
		:param select_config: dictionary containing parameters
		:return: (AnnData object) gene expression or peak accessibility data of the metacells, 
								  with the identified highly variable genes or peaks labeled in the var dataframe;
		"""

		# select highly variable peaks or highly variable genes
		if highly_variable_feature_query>0:
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

	## ====================================================
	# perform scaling of the metacell data
	def test_metacell_compute_unit_2(self,pre_meta_ad,save_mode=1,output_file_path='',output_filename='',filename_prefix='',verbose=0,select_config={}):

		"""
		perform scaling of the metacell data
		:param pre_meta_ad: (AnnData object) peak accessibility or gene expression data of the metacells
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1.(AnnData object) peak accessibility or gene expression data of metacells (with potentially added layers)
				 2.(AnnData object) scaled peak accessibility or gene expression data of metacells
		"""

		flag_query1 = 1
		if flag_query1>0:
			scale_type_id = select_config['scale_type_id'] # the scaling method type
			pre_meta_ad_scaled = []
			if scale_type_id>=1:
				warnings.filterwarnings('ignore')
				print('scale_type: ',scale_type_id)

				if scale_type_id==1:
					# ATAC-seq peak read count or gene expression matrix after normalization, log transformation, and before scaling
					print('read scaling')
					# mtx1 = scale(np.asarray(pre_meta_ad.X.todense()))
					mtx1 = scale(pre_meta_ad.X.toarray())
					pre_meta_scaled_read = pd.DataFrame(data=mtx1,index=pre_meta_ad.obs_names,
															columns=pre_meta_ad.var_names,
															dtype=np.float32)

				elif scale_type_id==2:
					# ATAC-seq peak read count or gene expression after normalization, log transformation, and scaling
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
					for i1 in range(feature_num1):
						feature_query = feature_query_vec[i1]
						t_value1 = pre_read[feature_query]
						thresh_2 = np.percentile(t_value1, thresh_upper_1)
						thresh_1_ori = np.percentile(t_value1, thresh_lower_1)
						# thresh_1 = 0
						# if np.min(t_value1)>1E-03:
						# 	thresh_1=thresh_1_ori
						thresh_1=thresh_1_ori
						if (thresh_2<1E-03):
							if (np.max(t_value1)>2):
								print('feature_query ',feature_query,i1,thresh_2,thresh_1_ori,np.max(t_value1),np.min(t_value1),np.mean(t_value1),np.median(t_value1))
							thresh_2=np.max(t_value1)
							if thresh_2==0:
								continue
								
						exprs = minmax_scale(t_value1,[thresh_1,thresh_2])
						pre_meta_scaled_read[feature_query] = scale(exprs)
					warnings.filterwarnings('default')
				else:
					print('min-max scaling of the count matrix')
					mtx1 = minmax_scale(pre_meta_ad.X.toarray())
					pre_meta_scaled_read = pd.DataFrame(data=mtx1,index=pre_meta_ad.obs_names,
															columns=pre_meta_ad.var_names,
															dtype=np.float32)

				pre_meta_scaled_read_1 = csr_matrix(pre_meta_scaled_read)
				pre_meta_ad_scaled = sc.AnnData(pre_meta_scaled_read_1)
				pre_meta_ad_scaled.obs_names, pre_meta_ad_scaled.var_names = pre_meta_ad.obs_names, pre_meta_ad.var_names
				warnings.filterwarnings('default')

				if save_mode==1:
					if output_filename=='':
						if filename_prefix!='':
							output_filename = '%s/%s.h5ad'%(output_file_path,filename_prefix)

					if (output_filename!=''):
						b = output_filename.find('txt')
						if b>=0:
							float_format = '%.6E'
							pre_meta_scaled_read.to_csv(output_filename,sep='\t',float_format=float_format)	# save txt file
						else:
							pre_meta_ad_scaled.write(output_filename)	# save AnnData

				elif save_mode==2:
					pre_meta_ad.layers['scale_%d'%(scale_type_id)] = pre_meta_scaled_read

		return pre_meta_ad, pre_meta_ad_scaled

	## ====================================================
	# metacell estimation
	def test_metacell_compute_pre1(self,flag_path_query=0,flag_SEACell_estimate=0,flag_summarize=0,overwrite=0,
										save_mode=1,output_file_path='',print_mode=1,verbose=0,select_config={}):

		"""
		metacell estimation
		:param flag_path_query: indicator of whether to update the dictionary of parameters
		:param flag_SEACell_estimate: indicator of whether to estimate SEACells
		:param flag_summarize: indicator of whether to summarize the RNA-seq or ATAC-seq read counts across single cells associated with each metacell (SEACell)
		:param overwrite: indicator of whether to overwrite the current file
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: ATAC-seq and RNA-seq AnnData objects of the metacells
		"""

		input_file_path1 = self.save_path_1
		data_file_type = select_config['data_file_type']

		if flag_path_query>0:
			select_config = self.test_file_path_query_2(select_config=select_config)

		input_file_path = select_config['data_path']
		print('input_file_path ',input_file_path)
		
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

		# query RNA-seq and ATAC-seq metacell data observation and variable attributes
		data_list1 = self.test_attribute_query_2(data_list=data_list1,
													input_filename_list=input_filename_list,
													feature_type_vec=feature_type_vec,
													feature_id=feature_id,
													column_id_query=column_id1,
													flag_plot=1,
													flag_query_1=1,
													flag_query_2=2,
													save_mode=1,
													output_file_path=output_file_path_1,
													filename_save_annot=filename_save_annot2,
													print_mode=print_mode,
													verbose=verbose,select_config=select_config)

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
		print('RNA-seq anndata of size ',rna_ad_ori.shape)
		print('ATAC-seq anndata of size ',atac_ad_ori.shape)
		print('RNA-seq anndata of common cells ',rna_ad.shape)
		print('ATAC-seq anndata of common cells ',atac_ad.shape)

		flag_query_pre=0
		if ('flag_query_pre' in select_config):
			flag_query_pre=select_config['flag_query_pre']
		if flag_query_pre>0:
			return

		field_query = ['n_SEACells', 'obsm_build_kernel_rna', 'obsm_build_kernel_atac','num_components','n_waypoint_eigs','waypoint_proportion']
		n_SEACells = select_config['metacell_num']
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

		# normalize, pca_compute = True, 1
		normalize, pca_compute = False, 0
		output_file_path = input_file_path
		feature_type = select_config['feature_type_1']
		run_id = select_config['run_id']
		type_id_feature = select_config['type_id_feature']
		# type_id_1=0
		type_id_1=type_id_feature

		filename_prefix_1 = '%s.%s.%d'%(data_file_type,feature_type,run_id)
		filename_prefix_save = '%s_multiome.%s.%d'%(data_file_type,feature_type,run_id)
		filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_1,run_id)

		select_config.update({'filename_prefix_1':filename_prefix_1,
								'filename_prefix_save':filename_prefix_save,
								'filename_save_annot_1':filename_save_annot_1})

		output_file_path = select_config['data_path_save']
		print('output_file_path ',output_file_path)

		column_query_pre1 = 'SEACell'
		if flag_SEACell_estimate>0:
			min_iter=10
			max_iter=200
			select_config.update({'min_iter':min_iter,'max_iter':max_iter})
			print('data_file_type: %s, feature_type: %s'%(data_file_type,feature_type))

			column_query1 = column_query_pre1
			column_query2 = '%s_1'%(column_query1)
			
			np.random.seed(0)
			if feature_type in ['rna']:
				if column_query1 in rna_ad.obs:
					rna_ad.obs[column_query2] = rna_ad.obs[column_query1].copy()
				
				num_components_pca = num_components
				rna_ad_1, model_1 = self.test_metacell_compute_unit_1(adata=rna_ad,normalize=normalize,
																		n_SEACells=n_SEACells,
																		obsm_build_kernel=obsm_build_kernel_rna,
																		pca_compute=pca_compute,
																		num_components=num_components_pca,
																		n_waypoint_eigs=n_waypoint_eigs,
																		waypoint_proportion=waypoint_proportion,
																		plot_convergence=1,
																		select_config=select_config)
				
				output_filename = '%s/%s.copy1.h5ad'%(output_file_path,filename_prefix_save)
				rna_ad_1.write(output_filename)
				rna_ad = rna_ad_1
				print('RNA-seq anndata\n',rna_ad_1)
				
				save_filename = '%s/model_rna_%s.h5'%(output_file_path,filename_prefix_save)
				with open(save_filename,'wb') as output_file:
					pickle.dump(model_1,output_file)
			else:
				if column_query1 in atac_ad.obs:
					atac_ad.obs[column_query2] = atac_ad.obs[column_query1].copy()
				num_components_atac = num_components
				atac_ad_1, model_1 = self.test_metacell_compute_unit_1(adata=atac_ad,normalize=normalize,
																		n_SEACells=n_SEACells,
																		obsm_build_kernel=obsm_build_kernel_atac,
																		pca_compute=0,
																		num_components=num_components_atac,
																		n_waypoint_eigs=n_waypoint_eigs,
																		waypoint_proportion=waypoint_proportion,
																		plot_convergence=1,
																		select_config=select_config)

				output_filename = '%s/%s.copy1.h5ad'%(output_file_path,filename_prefix_save)
				atac_ad_1.write(output_filename)
				atac_ad = atac_ad_1
				print('ATAC-seq anndata\n',atac_ad_1)
				
				save_filename = '%s/model_atac_%s.h5'%(output_file_path,filename_prefix_save)
				with open(save_filename,'wb') as output_file:
					pickle.dump(model_1,output_file)

		if flag_summarize>0:
			summarize_layer_type = 'raw'

			common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names)
			if len(common_cells) != atac_ad.shape[0]:
				print('Warning: The number of cells in RNA and ATAC objects are different. Only the common cells will be used.')
			print('common_cells ',len(common_cells))

			# use SEACells estimated on RNA-seq data for SEACell assignment on ATAC-seq data
			column_query1 = column_query_pre1
			column_query2 = '%s_ori'%(column_query1)
			if column_query1 in atac_ad.obs:
				atac_ad.obs[column_query2] = atac_ad.obs[column_query1].copy()

			if column_query1 in rna_ad.obs:
				rna_ad.obs[column_query2] = rna_ad.obs[column_query1].copy()

			atac_ad_1 = atac_ad[common_cells, :]
			rna_ad_1 = rna_ad[common_cells, :]

			type_id_feature = select_config['type_id_feature']
			# type_id_1=0
			type_id_1=type_id_feature
			if type_id_1==0:
				atac_ad_1.obs[column_query1] = rna_ad_1.obs[column_query1].copy()
			else:
				rna_ad_1.obs[column_query1] = atac_ad_1.obs[column_query1].copy()
			
			rna_ad_1.obs['%s_1'%(column_id1)] = atac_ad_1.obs[column_id1].copy()
			
			# summarize_layer_type_1 = 'raw_counts'	# rna data
			summarize_layer_type_1 = 'raw'	# rna data
			if atac_ad.raw==None:
				summarize_layer_type_2 = 'X'	# atac data
			else:
				summarize_layer_type_2 = 'raw'	# atac data
				print('raw count data: ')
				print(atac_ad.raw)
				print('data preview: ')
				print(atac_ad.raw.X[0:2,0:10])
			
			print('summarize_layer_type_1, summarize_layer_type_2',summarize_layer_type_1,summarize_layer_type_2)
			
			atac_meta_ad, rna_meta_ad = self._prepare_multiome_anndata(atac_ad=atac_ad_1, 
																		rna_ad=rna_ad_1, 
																		SEACells_label='SEACell', 
																		summarize_layer_atac=summarize_layer_type_2,
																		summarize_layer_rna=summarize_layer_type_1,
																		type_id_1=type_id_1,
																		select_config=select_config)
			
			# select highly variable peaks or highly variable genes
			highly_variable_feature_query = True
			rna_meta_ad = self.test_query_metacell_high_variable_feature(pre_meta_ad=rna_meta_ad,highly_variable_feature_query=highly_variable_feature_query,select_config=select_config)
			print('atac_meta_ad\n', atac_meta_ad)
			print('rna_meta_ad\n', rna_meta_ad)

			save_mode = 1
			output_file_path = select_config['data_path_save']
			run_id = select_config['run_id']
			filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_1,run_id)
			
			if save_mode>0:
				output_filename_1 = '%s/test_atac_meta_ad.%s.h5ad'%(output_file_path,filename_save_annot_1)
				atac_meta_ad.write(output_filename_1)
				output_filename_2 = '%s/test_rna_meta_ad.%s.h5ad'%(output_file_path,filename_save_annot_1)
				rna_meta_ad.write(output_filename_2)
				select_config.update({'filename_rna':output_filename_1,'filename_atac':output_filename_2})

			filename_prefix = 'test_rna_meta_ad.%s'%(filename_save_annot_1)
			scale_type_id = 2
			select_config.update({'scale_type_id':scale_type_id})
			# perform scaling of the metacell data
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
			# print('meta_exprs_2 ', meta_exprs_2.shape)
			self.meta_exprs_2 = meta_exprs_2

			return atac_meta_ad, rna_meta_ad

	## ====================================================
	# plot the metacells using the low-dimensional embedding
	def test_metacell_compute_pre2(self,feature_id='',feature_type_vec=['rna','atac'],column_vec_query=[],flag_path_query=0,
										save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
		
		"""
		plot the metacells using the low-dimensional embedding
		:param feature_id: (str) layer name in the AnnData object representing the low-dimensional embedding used to plot the metacells
		:param feature_type_vec: (array or list) the feature type of data: RNA-seq (rna) or ATAC-seq (atac)
		:param column_vec_query: (array or list) the column name and value in the .obs dataframe of the AnnData object used to select the subset of metacells
		:param file_path_query: indicator of whether to update the dictionary of parameters
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: True
		"""

		if flag_path_query>0:
			select_config = self.test_file_path_query_2(select_config=select_config)

		# feature_type_vec = ['rna','atac']
		feature_type_1, feature_type_2 = feature_type_vec[0:2]
		# single cell data filename
		input_filename_list1 = [select_config['input_filename_%s'%(feature_type_query)] for feature_type_query in feature_type_vec] # filename for origiinal single cell data
		# metacell data filename
		input_filename_list2 = [select_config['filename_%s'%(feature_type_query)] for feature_type_query in feature_type_vec] # filename for metacell data
		
		input_filename_2 = input_filename_list2[0]
		feature_type_annot_1 = feature_type_1.upper()
		print('load metacell data of %s from %s '%(feature_type_annot_1,input_filename_2))
		pre_meta_ad = sc.read_h5ad(input_filename_2)
		metacell_id = pre_meta_ad.obs_names
		metacell_num = len(metacell_id)
		print('metacell data, anndata of size ',pre_meta_ad.shape)
		print(pre_meta_ad)
		print('metacell number: %d'%(metacell_num))
		
		for i1 in range(1):
			input_filename_1 = input_filename_list1[i1]
			feature_type = feature_type_vec[i1]
			print('load single data of %s from %s '%(feature_type_1,input_filename_1))
			pre_ad_ori = sc.read_h5ad(input_filename_1)
			print('single cell data, anndata of size ',pre_ad_ori.shape)
			print(pre_ad_ori)
			# print(pre_ad_ori.raw)

			if len(column_vec_query)>0:
				# select subset of the data
				column_query, query_value = column_vec_query[0:2]
				pre_ad = pre_ad_ori[pre_ad_ori.obs[column_query]==query_value,:]
				# print(pre_ad)
			else:
				pre_ad = pre_ad_ori

			save_filename = output_filename
			if output_filename=='':
				save_filename = '%s/test_%s_%s_metacell_plot.%s.1.png'%(output_file_path,feature_type,filename_prefix_save,filename_save_annot)
			
			sample_query_id = metacell_id
			self.plot_1(pre_ad,plot_basis=feature_id,sample_query_id=sample_query_id,save_as=save_filename,show=True)

		return True

	## ====================================================
	# query metacell annotations and RNA-seq or ATAC-seq count matrix of metacells
	def test_metacell_compute_pre1_query2(self,run_idvec=[1],file_type_vec_1=['rna','atac'],file_type_vec_2=['rna'],flag_path_query=0,save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		query metacell annotations and RNA-seq or ATAC-seq count matrix of metacells
		:param run_idvec: (array or list) indices of the computations; each computation correspond to a set of estimated metacells;
		:param file_type_vec_1: (array or list) the feature types of data (rna or atac) for which to query the metacell annotations and the count matrix
		:param file_type_vec_2: (array or list) the feature types of data for which to query the count matrix
		:param flag_path_query: indicator of whether to update the dictionary of parameters
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: True
		"""

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
		if type_id_1==1:
			query_num2 = len(run_idvec)
			for i1 in range(query_num2):
				run_id1 = run_idvec[i1]
				select_config.update({'run_id':run_id1})
				select_config.update({'run_id_load':run_id1})
				flag_count=1
				flag_annot=1
				flag_id_query1 = 0
				select_config.update({'flag_id_query1':flag_id_query1})
				
				file_type_vec = file_type_vec_1
				self.test_feature_mtx_query_pre1(file_type_vec=file_type_vec,
														flag_count=flag_count,
														flag_annot=flag_annot,
														select_config=select_config)

				flag_count=1
				flag_annot=0
				flag_id_query1 = 1
				select_config.update({'flag_id_query1':flag_id_query1})

				file_type_vec = file_type_vec_2
				self.test_feature_mtx_query_pre1(file_type_vec=file_type_vec,
														flag_count=flag_count,
														flag_annot=flag_annot,
														select_config=select_config)

		# return df_query1
		return True

	## ====================================================
	# query and save specific types of ATAC-seq or RNA-seq count matrix
	def test_feature_mtx_query_pre1(self,gene_query_vec=[],filename_list=[],flag_count=0,flag_annot=0,file_type_vec=['rna','atac'],normalize_type_vec=[0,1,2],data_path_save='',
										save_mode=1,output_filename='',filename_prefix_save='',select_config={}):

		"""
		query and save specific types of ATAC-seq or RNA-seq count matrix
		:param gene_query_vec: (array or list) the target genes
		:param filename_list: (list) paths of the RNA-seq and ATAC-seq data of the metacells
		:param flag_count: indicator of whether to query the raw or transformed count matrix of the RNA-seq or ATAC-seq data of the metacells
		:param flag_annot: indicator of whether to query RNA-seq and ATAC-seq observation attribute annotations of the metacells and save to file
		:param feature_type_vec: (array or list) the feature type of data: RNA-seq (rna) or ATAC-seq (atac) 
		:param normalize_type_vec: (array or list) the types of count matrices to query (0:raw, 1:normalized and log-transformed, 2:normalized without log-transformaion)
		:param data_path_save: the directory to retrieve data from and save data
		:param save_mode: indicator of whether to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param select_config: dictionary containing parameters
		:return: True
		"""

		# input_file_path1 = self.save_path_1
		data_file_type = select_config['data_file_type']
		data_path = select_config['data_path']
		run_id = select_config['run_id']
		input_file_path = data_path
		
		if data_path_save=='':
			data_path_save = input_file_path
		
		filename_annot_vec = file_type_vec
		feature_type_1, feature_type_2 = file_type_vec[0:2]

		type_id_feature = select_config['type_id_feature']
		filename_save_annot1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)

		if len(filename_list)>0:
			input_filename_1, input_filename_2 = filename_list[0:2]
		else:
			input_filename_1 = '%s/test_rna_meta_ad.%s.h5ad'%(data_path_save,filename_save_annot1)
			input_filename_2 = '%s/test_atac_meta_ad.%s.h5ad'%(data_path_save,filename_save_annot1)
		
		output_file_path = data_path_save
		flag_query_count = flag_count
		
		# normalize_type=1
		normalize_type=0
		if flag_query_count>0:
			input_filename_list = [input_filename_1,input_filename_2]
			
			filename_annot_vec = file_type_vec
			query_num1 = len(filename_annot_vec)

			feature_type_annot = ['rna','atac']
			# index_name_vec = ['geneID','peakID']
			index_name_vec = ['ENSEMBL','peakID']
			dict_annot_1 = dict(zip(feature_type_annot,index_name_vec))
			normalize_type_annot = ['raw_count','log_normalize','normalize']

			for normalize_type in normalize_type_vec:
				for i1 in range(query_num1):
					input_filename = input_filename_list[i1]
					feature_type_query = filename_annot_vec[i1]
					adata = sc.read_h5ad(input_filename)
					print('input filename: %s'%(input_filename))
					print(adata)
					print('data preview: ')
					print(adata.X[0:2,0:10])
					print(adata.raw.X[0:2,0:10])
					
					if normalize_type==0:
						# raw count matrix
						df_count = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=adata.raw.X.toarray())
					elif normalize_type==1:
						# log-transformed normalized count matrix
						df_count = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=adata.X.toarray(),dtype=np.float32)
					else:
						# normalize count matrix
						pre_meta_ad = sc.AnnData(adata.raw.X)
						pre_meta_ad.obs_names, pre_meta_ad.var_names = adata.raw.obs_names, adata.raw.var_names
						# pre_meta_ad.obs['n_counts'] = pre_meta_ad.X.sum(axis=1)
						print('AnnData of metacell raw count matrix, normalize_type: %s'%(normalize_type))
						print(pre_meta_ad)
						# raw_count = adata.raw.X
						# print('raw count matrix, preview: ',raw_count[0:2,0:10])
						# print(pre_meta_ad.obs_names,pre_meta_ad.var_names)
						
						sc.pp.normalize_total(pre_meta_ad,inplace=True,key_added='n_counts')
						df_count = pd.DataFrame(index=pre_meta_ad.obs_names,columns=pre_meta_ad.var_names,data=pre_meta_ad.X.toarray(),dtype=np.float32)

					flag_id_query1=0
					if 'flag_id_query1' in select_config:
						flag_id_query1 = select_config['flag_id_query1']
					
					if flag_id_query1>0:
						if feature_type_query=='rna':
							input_filename_annot = select_config['filename_gene_annot']

							df_gene_annot_ori = pd.read_csv(input_filename_annot,index_col=False,sep='\t')
							df_gene_annot_ori = df_gene_annot_ori.sort_values(by=['length'],ascending=False)
							df_gene_annot_ori = df_gene_annot_ori.drop_duplicates(subset=['gene_name'])
							df_gene_annot_ori.index = np.asarray(df_gene_annot_ori['gene_name'])
							
							gene_query_name_ori = df_gene_annot_ori.index.unique()
							gene_query_name = df_count.columns.intersection(gene_query_name_ori,sort=False)
							gene_id_query = df_gene_annot_ori.loc[gene_query_name,'gene_id']
							# df_count.columns = gene_id_query
							
							column_vec_query = ['gene_id','gene_name','gene_name_ori','means','dispersions','dispersions_norm','length']
							df1 = df_gene_annot_ori.loc[gene_query_name,column_vec_query]

							df1 = df1.sort_values(by=['length'],ascending=False)
							df1['duplicated'] = df1.duplicated(subset=['gene_id'])
							
							output_filename = '%s/test_gene_name_query.1.txt'%(output_file_path)
							df1.to_csv(output_filename,sep='\t')
							
							gene_query_vec_1 = df1.index[df1['duplicated']==False]
							gene_query_vec_2 = df1.index[df1['duplicated']==True]
							
							print('the number of genes with unduplicated name: %d'%(len(gene_query_vec_1)))
							print('the number of genes with duplicated name: %d'%(len(gene_query_vec_2)))

							df_count = df_count.loc[:,gene_query_vec_1]
							df_count_ori = df_count
							df_count.columns = df_gene_annot_ori.loc[gene_query_vec_1,'gene_id']
							# print('df_count_ori, df_count ',df_count_ori.shape,df_count.shape)

					df_count = df_count.T  # shape: (variable number, metacell number)
					index_name_query = dict_annot_1[feature_type_query]
					df_count.index.name = index_name_query
					
					print('count matrix, dataframe of size ',df_count.shape)
					print('minimal value of each variable: ')
					print(df_count.min(axis=1))
					print('maximal value of each variable: ')
					print(df_count.max(axis=1))
					
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
			feature_type_vec = ['rna','atac']
			feature_type_1 = select_config['feature_type_1']
			feature_type_2 = pd.Index(feature_type_vec).difference([feature_type_1],sort=False)[0]
			
			input_filename_query1 = '%s/test_%s_df_obs.%s.1.txt'%(input_file_path,feature_type_1,data_file_type)
			input_filename_query2 = '%s/test_%s_df_obs.%s.1.txt'%(input_file_path,feature_type_2,data_file_type)
			input_filename_2 = '%s/test_%s_meta_ad_df_obs.%s.1.txt'%(data_path_save,feature_type_1,filename_save_annot1)

			df_feature1 = pd.read_csv(input_filename_query1,index_col=0,sep='\t')
			df_feature2 = pd.read_csv(input_filename_query2,index_col=0,sep='\t')
			df2 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			sample_id = df2.index

			sample_id1 = df_feature1.index  # feature type 1 observation
			sample_id2 = df_feature2.index  # feature type 2 observation
			sample_id_query = sample_id1.intersection(sample_id2,sort=False)
			print('the number of cells with both RNA-seq and ATAC-seq data: %d'%(len(sample_id_query)))

			sample_id1_query1 = sample_id.intersection(sample_id1,sort=False)
			sample_id1_query2 = sample_id.difference(sample_id1,sort=False)
			assert len(sample_id1_query2)==0

			# copy sample annotation from RNA-seq data annotation
			# field_query1 = ['phenograph','celltype']
			field_query1 = df_feature1.columns.difference(df2.columns,sort=False)
			df2.loc[sample_id1_query1,field_query1] = df_feature1.loc[sample_id1_query1,field_query1]
			df2 = df2.rename(columns={'celltype':'celltype_%s'%(feature_type_1)})

			sample_id2_query1 = sample_id.intersection(sample_id2,sort=False)
			sample_id2_query2 = sample_id.difference(sample_id2,sort=False)
			assert len(sample_id2_query2)==0

			# copy sample annotation from ATAC-seq data annotation
			field_query2 = df_feature2.columns.difference(df2.columns,sort=False)
			df2.loc[sample_id2_query1,field_query2] = df_feature2.loc[sample_id2_query1,field_query2]

			output_filename = '%s/sampleMetadata.%s.tsv.gz'%(output_file_path,filename_save_annot1)
			df2.index.name='sample_id'
			df2.to_csv(output_filename,sep='\t',compression='gzip')
			print('observation annotation of feature type 1, dataframe of size ',df_feature1.shape)
			print('observation annotation of feature type 2, dataframe of size ',df_feature2.shape)
			print('annotation of the metacells, dataframe of size ',df2.shape)

		return True

	## ====================================================
	# query observation and variable attributes of the ATAC-seq and RNA-seq data of the metacells
	def test_attribute_query(self,data_vec,feature_type_vec=[],save_mode=1,output_file_path='',filename_save_annot='',select_config={}):
		
		"""
		query observation and variable attributes of the ATAC-seq and RNA-seq data of the metacells
		:param data_vec: dictionary containing ATAC-seq and RNA-seq AnnData objects of the metacells
		:param feature_type_vec: (array or list) the feature type of data: RNA-seq (rna) or ATAC-seq (atac)
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param select_config: dictionary containing parameters
		:return: dictionary containing observation and variable attributes of the ATAC-seq and RNA-seq data of the metacells
		"""

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

	## ====================================================
	# query observation attributes and the RNA-seq or ATAC-seq raw count matrix
	def test_attribute_query_2(self,data_list=[],input_filename_list=[],feature_type_vec=[],feature_id='',column_id_query='celltype',
									flag_plot=1,flag_query_1=1,flag_query_2=1,
									save_mode=1,output_file_path='',filename_save_annot='',print_mode=1,verbose=0,select_config={}):

		"""
		query observation attributes and the RNA-seq or ATAC-seq raw count matrix
		:param data_list: (list) AnnData objects of the corresponding feature types specified in feature_type_vec
		:param input_filename_list (list) paths of AnnData objects of the corresponding feature types specified in feature_type_vec
		:param feature_type_vec: (array or list) the feature type of data: RNA-seq (rna) or ATAC-seq (atac)
		:param feature_id: (str) the plotting tool that computed the coordinates used in the scatter() function in Scanpy
		:param column_id_query: (str) column in the .obs dataframe of the AnnData object that corresponds to the cell type
		:param flag_plot: indicator of whether to plot the metacells or cells using low-dimensional feature embeddings
		:param flag_query_1: indicator of whether to query the raw count matrix
		:param flag_query_2: indicator of whether to query the number of metacells
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param print_mode: indicator of whether to save the observation and variable annotations to file
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (list) updated AnnData objects of the corresponding feature types specified in feature_type_vec
		"""

		feature_num = len(feature_type_vec)
		data_list1 = data_list
		for i1 in range(feature_num):
			pre_ad = data_list1[i1]
			feature_type_query = feature_type_vec[i1]
			column_id1 = column_id_query
			if print_mode>0:
				# query the cell type annotations of observations and save to file
				if column_id1 in pre_ad.obs:
					celltype_query_1 = pre_ad.obs[column_id1]
					celltype_vec_1= celltype_query_1.unique()
					celltype_num1 = len(celltype_vec_1)
					print('cell type: ',celltype_num1,celltype_vec_1)
					
					df_1 = pre_ad.obs.loc[:,[column_id1]]
					df_1['count'] = 1
					df1 = df_1.groupby(by=[column_id1]).sum()

					output_filename = '%s/test_%s_ad.%s.celltype_query.1.txt'%(output_file_path,feature_type_query,filename_save_annot)
					df1.to_csv(output_filename,sep='\t')
				else:
					print('the column %s not include'%(column_id1))
				
				output_filename = '%s/test_%s_ad.%s.df_obs.txt'%(output_file_path,feature_type_query,filename_save_annot)
				pre_ad.obs.to_csv(output_filename,sep='\t')

				output_filename = '%s/test_%s_ad.%s.df_var.txt'%(output_file_path,feature_type_query,filename_save_annot)
				pre_ad.var.to_csv(output_filename,sep='\t')
				
				read_count = pre_ad.X
				read_count_1 = read_count.sum(axis=1)
				print('maximum, minimum, median and mean value of the count matrix: ')
				print(np.max(read_count_1),np.min(read_count_1),np.median(read_count_1),np.mean(read_count_1))
				
			if flag_plot>0:
				if 'celltype_colors' in pre_ad.uns:
					celltype_color_vec = pre_ad.uns['celltype_colors']
				try:
					sc.pl.scatter(pre_ad, basis=feature_id, color=column_id1, legend_fontsize=10, frameon=False)
					
					output_filename = '%s/test_%s_%s_%s.adata.1.1.png'%(output_file_path,feature_id,feature_type_query,filename_save_annot)
					plt.savefig(output_filename,format='png')
				except Exception as error:
					print('error! ',error)

			type_id_query = flag_query_1
			if flag_query_1>0:
				# query the raw count matrix
				if pre_ad.raw!=None:
					pre_ad_raw = pre_ad.raw.X
					read_count1 = pre_ad_raw.sum(axis=1)
					print(np.max(read_count1),np.min(read_count1),np.median(read_count1),np.mean(read_count1))
				else:
					if type_id_query==2:
						raw_ad = sc.AnnData(pre_ad.X)
						raw_ad.obs_names, raw_ad.var_names = pre_ad.obs_names, pre_ad.var_names
						pre_ad.raw = pre_ad
						data_list1[i1] = pre_ad
						
						if verbose>0:
							print('pre_ad.raw\n')
							print(pre_ad.raw)
							print(pre_ad.raw.X[0:5,0:5])

						if save_mode>0:
							input_filename = input_filename_list[i1]
							b = input_filename.find('.h5ad')
		
							output_filename = input_filename[0:b]+'.copy1.h5ad' # save the atac-seq data with raw counts
							if os.path.exists(output_filename)==False:
								pre_ad.write(output_filename)
								print('pre_ad\n',pre_ad)
							else:
								print('the file exists ',output_filename)
								input_filename=output_filename

			if flag_query_2>0:
				column_id_1 = 'SEACell'
				if not (column_id_1 in pre_ad.obs):
					column_id_1 = 'Metacell'
				if column_id_1 in pre_ad.obs:	
					SEACell_vec_1 = pre_ad.obs[column_id_1].unique() # the previously estimated SEACells
					SEACell_num1 = len(SEACell_vec_1)
					print('SEACell number: %d'%(SEACell_num1))

		return data_list1

	## ====================================================
	# compute normalized count matrix of ATAC-seq or RNA-seq data
	def test_read_count_query_normalize(self,adata=[],feature_type_query='atac',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',float_format='%.5f',verbose=0,select_config={}):
		
		"""
		compute normalized count matrix of ATAC-seq or RNA-seq data
		:param adata: AnnData object of ATAC-seq or RNA-seq data
		:param feature_type_query: (str) the feature type of data: RNA-seq (rna) or ATAC-seq (atac)
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param float_format: format to keep data precision in saving data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: normalized count matrix of ATAC-seq or RNA-seq data (row:cell or metacell, column:ATAC-seq peak locus or gene)
		"""

		flag_read_normalize_1=1
		if flag_read_normalize_1>0:
			if len(adata)==0:
				adata = self.atac_meta_ad

			if ('normalize' in adata.layers):
				read_count_1 = adata.layers['normalize']
				# print('read_count_1: ',read_count_1)
				type_id_query = 0
			else:
				type_id_query = 1 # the read count data need to be normalized
				try:
					read_count_1 = adata.layers['raw']
								
				except Exception as error:
					print('error! ',error)
					if adata.raw!=None:
						adata1 = adata.raw
						adata1.obs_names, adata1.var_names = adata.obs_names, adata.var_names
						type_id_query = 3
			
					else:
						field_id = 'filename_%d_meta_ad'%(feature_type_query)
						input_filename = select_config[field_id]
						if os.path.exists(input_filename)==False:
							print('please provide ATAC-seq data of the metacells')
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
					# Normalize
					sc.pp.normalize_total(adata1, key_added='n_counts')

			if save_mode>0:
				filename_prefix_1 = '%s_meta_ad'%(feature_type_query)
				filename_save_annot_1 = filename_save_annot
				if filename_save_annot=='':
					filename_save_annot_1 = select_config['filename_save_annot_pre1']

				if save_mode==2:
					# save anndata
					output_filename = '%s/test_%s.%s.normalize.h5ad'%(output_file_path,filename_prefix_1,filename_save_annot_1)
					adata1.write(output_filename)

				if type_id_query==0:
					# read_count_normalize_1 = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=np.asarray(atac_ad_1.X.toarray()),dtype=np.float32)
					read_count_normalize_1 = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=np.asarray(read_count_1),dtype=np.float32)
				else:
					read_count_normalize_1 = pd.DataFrame(index=adata1.obs_names,columns=adata1.var_names,data=np.asarray(adata1.X.toarray()),dtype=np.float32)

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

			return read_count_normalize_1

	## ====================================================
	# query normalized and log-transformed ATAC-seq and RNA-seq data
	def test_read_count_query_log_normalize(self,feature_type_vec=[],peak_read=[],rna_exprs_unscaled=[],
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',
												save_format='tsv.gz',float_format='%.5f',verbose=0,select_config={}):
		
		"""
		query normalized and log-transformed ATAC-seq and RNA-seq data
		:param feature_type_vec: (array or list) the feature type of data: RNA-seq (rna) or ATAC-seq (atac)
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_unscaled_exprs: (dataframe) log-transformed normalized gene expression matrix of the metacells (row:metacell, column:gene)
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param save_format: the file format used to save data
		:param float_format: format to keep data precision in saving data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing the normalized and log-transformed ATAC-seq and RNA-seq count matrices of the metacells
		"""

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
				filename_prefix_1 = '%s_meta_ad' % (feature_type_query)
				output_filename = '%s/test_%s.%s.log_normalize.%s'%(output_file_path,filename_prefix_1,filename_save_annot,save_format)
				
				df_query_1.to_csv(output_filename,sep='\t',float_format=float_format)
				if verbose>0:
					print('data: ',df_query_1.shape,feature_type_query)

			return dict_query

	## ====================================================
	# load the ATAC-seq data and RNA-seq data of the metacells
	def test_load_data_pre1(self,flag_format=False,select_config={}):
		
		"""
		load the ATAC-seq data and RNA-seq data of the metacells
		:param select_config: dictionary containing filenames of the ATAC-seq data and RNA-seq data of the metacells
		:param flag_format: indicator of whether to use uppercase variable names in the RNA-seq data of the metacells
		:return: 1.(dataframe) log-transformed normalized peak accessibility matrix
				 2.(dataframe) z-scores of log-transformed normalized gene expression matrix
				 3.(dataframe) log-transformed normalized gene expression matrix
		"""

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

		column_1 = 'filename_rna_exprs_1'
		meta_scaled_exprs = []
		if column_1 in select_config:
			input_filename_3 = select_config['filename_rna_exprs_1']
			meta_scaled_exprs = pd.read_csv(input_filename_3,index_col=0,sep='\t')

			if flag_format==True:
				meta_scaled_exprs.columns = meta_scaled_exprs.columns.str.upper()
		
			meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
			vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
			print('scaled metacell gene expression data, dataframe of size ',meta_scaled_exprs.shape)
			print('mean values: ',vec2)

		if len(meta_scaled_exprs)>0:
			sample_id1 = meta_scaled_exprs.index
			assert list(sample_id)==list(sample_id1)

		peak_read = pd.DataFrame(index=atac_meta_ad.obs_names,columns=atac_meta_ad.var_names,data=atac_meta_ad.X.toarray(),dtype=np.float32)
		meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
		self.peak_read = peak_read
		self.meta_exprs_2 = meta_exprs_2
		self.meta_scaled_exprs = meta_scaled_exprs

		vec1 = utility_1.test_stat_1(np.mean(atac_meta_ad.X.toarray(),axis=0))
		vec3 = utility_1.test_stat_1(np.mean(meta_exprs_2,axis=0))

		print('atac_meta_ad mean values ',atac_meta_ad.shape,vec1)
		print('rna_meta_ad mean values ',meta_exprs_2.shape,vec3)

		return peak_read, meta_scaled_exprs, meta_exprs_2

	## ====================================================
	# load the ATAC-seq data and RNA-seq data of the metacells
	def test_load_data_pre2(self,flag_format=False,flag_scale=0,save_mode=1,output_file_path='',verbose=0,select_config={}):

		"""
		load the ATAC-seq data and RNA-seq data of the metacells
		:param select_config: dictionary containing filenames of the ATAC-seq data and RNA-seq data of the metacells
		:param flag_format: indicator of whether to use uppercase variable names in the RNA-seq data of the metacells
		:param flag_scale: indicator of whether to perform scaling of the data
		:param save_mode: verbosity level to print the intermediate information
		:param output_file_path: the directory to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1.(dataframe) log-transformed normalized peak accessibility matrix
				 2.(dataframe) z-scores of log-transformed normalized gene expression matrix
				 3.(dataframe) log-transformed normalized gene expression matrix
		"""

		input_filename_1, input_filename_2 = select_config['filename_rna_meta'],select_config['filename_atac_meta']
		print('input_filename_1 ',input_filename_1)
		print('input_filename_2 ',input_filename_2)

		extension_vec = ['.h5ad','.tsv','.txt','.csv']
		data_type_vec = [0,1,1,2]
		query_num1 = len(extension_vec)
		data_type = -1
		for i1 in range(query_num1):
			extension_query = extension_vec[i1]
			b1 = input_filename_1.find(extension_query)
			if b1>=0:
				data_type = data_type_vec[i1]
				break

		if data_type==0:
			rna_meta_ad = sc.read_h5ad(input_filename_1)
			atac_meta_ad = sc.read_h5ad(input_filename_2)
		else:
			if data_type==1:
				sep = '\t'
			elif data_type==2:
				sep=','

			df_rna_meta = pd.read_csv(input_filename_1,index_col=0,sep=sep)
			df_atac_meta = pd.read_csv(input_filename_2,index_col=0,sep=sep)

			rna_meta_ad = sc.AnnData(df_rna_meta,dtype=df_rna_meta.values.dtype)
			rna_meta_ad.X = csr_matrix(rna_meta_ad.X)

			atac_meta_ad = sc.AnnData(df_atac_meta,dtype=df_atac_meta.values.dtype)
			atac_meta_ad.X = csr_matrix(atac_meta_ad.X)

		if flag_format==True:
			rna_meta_ad.var_names = rna_meta_ad.var_names.str.upper()
			rna_meta_ad.var.index = rna_meta_ad.var.index.str.upper()

		print('RNA-seq metacell data\n', rna_meta_ad)
		print('ATAC-seq metacell data\n', atac_meta_ad)

		self.rna_meta_ad = rna_meta_ad
		sample_id = rna_meta_ad.obs_names
		assert list(sample_id)==list(atac_meta_ad.obs_names)

		atac_meta_ad = atac_meta_ad[sample_id,:]
		self.atac_meta_ad = atac_meta_ad
		
		column_1 = 'filename_rna_exprs'
		meta_scaled_exprs = []
		load = 1
		if column_1 in select_config:
			input_filename_3 = select_config[column_1]
			
			if os.path.exists(input_filename_3)==False:
				print('the file does not exist: ',input_filename_3)
				load = 0
			else:
				meta_scaled_exprs = pd.read_csv(input_filename_3,index_col=0,sep='\t')
				if flag_format==True:
					meta_scaled_exprs.columns = meta_scaled_exprs.columns.str.upper()
			
				meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
			
		if load==0:
			if flag_scale>0:
				scale_type_id = 2
				# if 'scale_type_id' in select_config:
				# 	scale_type_id = select_config['scale_type_id']
				# else:
				# 	select_config.update({'scale_type_id':scale_type_id})
				select_config.update({'scale_type_id':scale_type_id})

				# save_mode_1 = 2
				save_mode_1 = 1
				data_file_type = select_config['data_file_type']
				filename_prefix = data_file_type
				output_filename = select_config[column_1]
				# perform scaling of the metacell data
				pre_meta_ad_rna, pre_meta_ad_scaled_rna = self.test_metacell_compute_unit_2(pre_meta_ad=rna_meta_ad,
																							save_mode=save_mode_1,output_file_path=output_file_path,
																							output_filename=output_filename,
																							filename_prefix=filename_prefix,
																							select_config=select_config)
				# self.select_config = select_config

				save_mode1 = save_mode
				rna_meta_ad_scaled = pre_meta_ad_scaled_rna
				meta_scaled_exprs = pd.DataFrame(index=rna_meta_ad_scaled.obs_names,columns=rna_meta_ad_scaled.var_names,
													data=rna_meta_ad_scaled.X.toarray(),dtype=np.float32)
				self.meta_scaled_exprs = meta_scaled_exprs
				print('meta_scaled_exprs ')
				print(meta_scaled_exprs[0:2])
				
		peak_read = pd.DataFrame(index=atac_meta_ad.obs_names,columns=atac_meta_ad.var_names,data=atac_meta_ad.X.toarray(),dtype=np.float32)
		meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
		self.peak_read = peak_read
		self.meta_exprs_2 = meta_exprs_2
		self.meta_scaled_exprs = meta_scaled_exprs

		vec1 = utility_1.test_stat_1(np.mean(atac_meta_ad.X.toarray(),axis=0))
		vec3 = utility_1.test_stat_1(np.mean(meta_exprs_2,axis=0))

		print('atac_meta_ad mean values ',atac_meta_ad.shape,vec1)
		print('rna_meta_ad mean values ',meta_exprs_2.shape,vec3)

		if len(meta_scaled_exprs)>0:
			vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
			print('meta_scaled_exprs mean values ',meta_scaled_exprs.shape,vec2)

		return peak_read, meta_scaled_exprs, meta_exprs_2

	## ====================================================
	# the plot function
	# plot for the given cells or metacells
	def plot_1(self,ad,plot_basis='X_umap',sample_query_id=[],figsize=(5,5),title='',save_as=None,show=True):

		"""
		Plot for given cells or metacells
		:param ad: AnnData object containing data of the given cells or metacells
		:param plot_basis: (str) layer name in the AnnData object representing the low-dimensional embedding used to plot the cells
		:param sample_query_id: (pandas.Series) identifiers of the given set of cells or metacells
		:param figsize: (int,int) tuplet of integers representing figure size
		:param title: (str) title of figure.
		:param save_as: (str) path to which figure is saved. If None, figure is not saved.
		:return: None
		"""

		plt.figure(figsize=figsize)
		plt.scatter(ad.obsm[plot_basis][:, 0],
					ad.obsm[plot_basis][:, 1],
					s=1, color='lightgrey')
		
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

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)


