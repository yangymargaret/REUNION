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
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator
from sklearn.preprocessing import minmax_scale, scale, quantile_transform

import time
from timeit import default_timer as timer

from joblib import Parallel, delayed
from .test_reunion_compute_pre2 import _Base_pre2
from . import train_pre1
from . import utility_1
from .utility_1 import test_query_index
import h5py
import pickle

class _Base2_2(_Base_pre2):
	"""Feature association estimation
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

		_Base_pre2.__init__(self,file_path=file_path,
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

		self.test_config_pre1()
		self.data_pre_dict = {}
		self.data_pre_dict['peak_group'] = {}

	## ====================================================
	# parameter configuration
	def test_config_pre1(self,save_mode=1,verbose=0,select_config={}):

		"""
		parameter configuration
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		return: dictionary containing parameters
		"""

		feature_type_vec = ['peak_motif','peak_motif_ori','peak_tf']
		feature_type_annot = feature_type_annot = ['peak-motif sequence feature']*2+['accessibility feature'] # the annotations for different feature types of peak loci

		# feature type annotation
		self.test_query_feature_type_annot_1(feature_type_vec=feature_type_vec,feature_type_annot=feature_type_annot,save_mode=1)
		self.dict_motif_data = {}
		self.dict_group_basic_1 = {}
		self.dict_group_basic_2 = {}
		
		return select_config

	## ====================================================
	# query feature type annotation
	def test_query_feature_type_annot_1(self,feature_type_vec=[],feature_type_annot=[],save_mode=1,verbose=0,select_config={}):

		"""
		query feature type annotation
		:param feature_type_vec: (array) feature types for feature representations
		:param feature_type_annot: (array) annotations of the feature types for feature representations
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1. dictionary with mapping between feature type and feature type annotation
				 2. dictionary with mapping between name for feature embedding and feature type annotation
		"""

		if len(feature_type_vec)==0:
			feature_type_vec = ['peak_motif','peak_motif_ori','peak_tf']  # feature types of peak loci
		feature_type_vec_pre1 = feature_type_vec

		if len(feature_type_annot)==0:
			feature_type_annot = ['peak-motif sequence feature']*2+['accessibility feature'] # the annotations for different feature types of peak loci
		
		feature_type_vec_pre2 = ['latent_%s'%(feature_type_query) for feature_type_query in feature_type_vec_pre1]
		dict_feature_type_annot1 = dict(zip(feature_type_vec_pre1,feature_type_annot))
		dict_feature_type_annot2 = dict(zip(feature_type_vec_pre2,feature_type_annot))
		if save_mode>0:
			self.dict_feature_type_annot1 = dict_feature_type_annot1 	# the dictionary with mapping between feature type and feature type annotation
			self.dict_feature_type_annot2 = dict_feature_type_annot2	# the dictionary with mapping between name for feature embedding and feature type annotation
		
		return dict_feature_type_annot1, dict_feature_type_annot2

	## ====================================================
	# query the filename of motif scanning data
	def test_query_motif_filename_pre1(self,data_file_type='',thresh_motif=5e-05,column_motif='motif_id',format_type=1,retrieve_mode=0,verbose=0,select_config={}):

		"""
		query the filename of motif scanning data
		:param data_file_type: (str) name or identifier of the data
		:param thresh_motif: (float) threshold on motif score used to identify motif presence by motif scanning
		:param column_motif: (str) column corresponding to the names of TFs with binding motifs
		:param retrieve_mode: indicator of whether to update the parameters in select_config or create a dictionary storing parameters
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary storing parameters
		:return: dictionary storing parameters
		"""

		flag_query1 = 1
		if flag_query1>0:
			data_file_query_motif = data_file_type
			thresh_vec_1 = [5e-05,1e-04,0.001,-1,0.01] # the thresholds used for motif scanning
			annot_vec_1 = ['5e-05','1e-04','0.001',-1,'0.01']
			thresh_num1 = len(thresh_vec_1)
			thresh_idvec_1 = np.arange(1,thresh_num1+1)
			dict_1 = dict(zip(thresh_vec_1,thresh_idvec_1))
			dict_2 = dict(zip(thresh_vec_1,annot_vec_1))

			if retrieve_mode==0:
				select_config_1 = select_config # update field in the original
			else:
				select_config_1 = dict()
			
			thresh_motif_id = dict_1[thresh_motif]
			thresh_motif_annot = dict_2[thresh_motif]
			print('thresh_motif, thresh_motif_id: ',thresh_motif,thresh_motif_id)
			select_config_1.update({'dict_motif_thresh_annot':dict_2})

			data_path_save_motif = select_config['data_path_save_motif']
			filename_prefix = select_config['filename_prefix']
			filename_annot_1 = select_config['filename_annot_1']

			motif_filename1 = '%s/test_motif_data.%s.h5ad'%(data_path_save_motif,filename_annot_1)
			motif_filename2 = '%s/test_motif_data_score.%s.h5ad'%(data_path_save_motif,filename_annot_1)

			file_format = 'csv'
			if format_type==0:
				input_filename_1 = '%s/%s_motif.1.%s'%(data_path_save_motif,filename_prefix,file_format)
				input_filename_2 = '%s/%s_motif_scores.1.%s'%(data_path_save_motif,filename_prefix,file_format)
				filename_chromvar_score = '%s/%s_chromvar_scores.1.csv'%(data_path_save_motif,filename_prefix)
			else:
				filename_annot1 = thresh_motif_annot
				input_filename_1 = '%s/%s.motif.%s.%s'%(data_path_save_motif,filename_prefix,filename_annot1,file_format)
				input_filename_2 = '%s/%s.motif_scores.%s.%s'%(data_path_save_motif,filename_prefix,filename_annot1,file_format)
				filename_chromvar_score = '%s/%s.chromvar_scores.%s.csv'%(data_path_save_motif,filename_prefix,filename_annot1)

			file_path_2 = '%s/TFBS'%(data_path_save_motif)
			if (os.path.exists(file_path_2)==False):
				print('the directory does not exist: %s'%(file_path_2))
				os.makedirs(file_path_2,exist_ok=True)

			input_filename_annot = '%s/translationTable.csv'%(file_path_2)

			select_config_1.update({'data_path_save_motif':data_path_save_motif})

			select_config_1.update({'input_filename_motif_annot':input_filename_annot,'filename_translation':input_filename_annot,
									'column_motif':column_motif})
			
			select_config_1.update({'motif_filename_1':input_filename_1,'motif_filename_2':input_filename_2,
									'filename_chromvar_score':filename_chromvar_score,
									'motif_filename1':motif_filename1,'motif_filename2':motif_filename2})

			filename_save_annot_1 = data_file_type
			select_config.update({'filename_save_annot_pre1':filename_save_annot_1})
			field_query_1 = ['input_filename_motif_annot','filename_translation','motif_filename_1','motif_filename_2',
								'filename_chromvar_score','motif_filename1','motif_filename2']
			
			for field_id in field_query_1:
				value = select_config_1[field_id]
				print('field, value: ',field_id,value)

			return select_config_1

	## ====================================================
	# perform feature dimension reduction
	def test_query_feature_pre2(self,feature_mtx=[],method_type='SVD',n_components=50,sub_sample=-1,verbose=0,select_config={}):

		"""
		perform feature dimension reduction
		:param feature_mtx: (dataframe) feature matrix (row:observation, column:feature)
		:param method_type: (str) method to perform feature dimension reduction
		:param n_components: (int) the nubmer of latent components used in feature dimension reduction
		:param sub_sample: (int) the number of observations selected in subsampling; if sub_sample=-1, keep all the observations
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. feature dimension reduction model
				 2. (dataframe) low-dimensional feature embeddings of observations (row:observation,column:latent components)
				 3. (dataframe) loading matrix (associations between latent components and features)
		"""

		vec1 = ['PCA','Incremental PCA','Kernel PCA','Sparse PCA','SVD',
					'GRP','SRP','MDS','ISOMAP','Minibatch','ICA','tSNE','LLE','Encoder',-1,'NMF']
		query_num1 = len(vec1)
		idvec_1 = np.arange(query_num1)
		dict_1 = dict(zip(vec1,idvec_1))

		start = time.time()
		method_type_query = method_type
		type_id_reduction = dict_1[method_type_query]
		feature_mtx_1 = feature_mtx
		if verbose>0:
			print('feature_mtx, method_type_query: ',feature_mtx_1.shape,method_type_query)
			print(feature_mtx_1[0:2])

		from .utility_1 import dimension_reduction
		feature_mtx_pre, dimension_model = dimension_reduction(x_ori=feature_mtx_1,feature_dim=n_components,type_id=type_id_reduction,shuffle=False,sub_sample=sub_sample)
		df_latent = feature_mtx_pre
		df_component = dimension_model.components_  # shape: (n_components,n_features)

		return dimension_model, df_latent, df_component

	## ====================================================
	# compute feature embeddings of observations
	def test_query_feature_pre1(self,peak_query_vec=[],gene_query_vec=[],method_type_vec=[],motif_data=[],motif_data_score=[],
								peak_read=[],rna_exprs=[],n_components=50,sub_sample=-1,flag_shuffle=False,float_format='%.6f',input_file_path='',
								save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		perform feature dimension reduction
		:param peak_query_vec: (array) peak loci; if not specified, genome-wide peaks in the peak accessibility matrix are included
		:param gene_query_vec: (array) genes with expressions or TFs with expressions to include in analysis
		:param method_type_vec: (array or list) methods for feature dimension reduction for the different feature types
		:param motif_data: (dataframe) motif presence in peak loci by motif scanning (binary) (row:peak, column:TF (associated with motif))
		:param motif_data_score: (dataframe) motif scores by motif scanning (row:peak, column:TF (associated with motif))
		:param peak_read: (dataframe) peak accessibility matrix (normalized and log-transformed) (row:metacell, column:peak)
		:param rna_exprs: (dataframe) gene expression matrix (row:metacell, column:gene)
		:param n_components: (int) the nubmer of latent components used in feature dimension reduction 
		:param sub_sample: (int) the number of observations selected in subsampling; if sub_sample=-1, keep all the observations
		:param flag_shuffle: indicator of whether to shuffle the observations
		:param float_format: format to keep data precision
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing feature dimension reduction model, feature embeddings of observations and loading matrix for each feature type
		"""

		motif_query_vec = motif_data.columns.intersection(rna_exprs.columns,sort=False) # TF with motif and expression
		motif_query_num = len(motif_query_vec)
		print('TFs (with motif and expression): ',motif_query_num)
		
		if len(peak_query_vec)>0:
			feature_mtx_query1 = peak_read.loc[:,peak_query_vec].T  # peak accessibility matrix, shape: (peak_num,cell_num)
		else:
			peak_query_vec = peak_read.columns
			feature_mtx_query1 = peak_read.T

		feature_motif_query1 = motif_data.loc[peak_query_vec,motif_query_vec] # motif matrix of peak, shape: (peak_num,motif_num)
		feature_motif_query2 = motif_data.loc[peak_query_vec,:] # motif matrix of peak, shape: (peak_num,motif_num)

		column_1 = 'flag_peak_tf_combine'
		# flag_peak_tf_combine=1: combine peak accessibility and TF expression matrix to perform dimension reduction
		# since peak number >> TF number, using peak accessibility and TF expression for dimension reduction is similar to using peak accessibility for dimension reduction
		flag_peak_tf_combine = 0
		if column_1 in select_config:
			flag_peak_tf_combine = select_config[column_1]

		if flag_peak_tf_combine>0:
			sample_id = peak_read.index
			rna_exprs = rna_exprs.loc[sample_id,:]
			if len(gene_query_vec)==0:
				gene_query_vec = motif_query_vec # the genes are TFs
			
			feature_expr_query1 = rna_exprs.loc[:,gene_query_vec].T # tf expression, shape: (tf_num,cell_num)
			feature_mtx_1 = pd.concat([feature_mtx_query1,feature_expr_query1],axis=0,join='outer',ignore_index=False)
		else:
			feature_mtx_1 = feature_mtx_query1
		
		list_pre1 = [feature_mtx_1,feature_motif_query1,feature_motif_query2]
		query_num1 = len(list_pre1)
		dict_query1 = dict()

		feature_type_vec_pre1 = ['peak_tf','peak_motif','peak_motif_ori']
		feature_type_annot = ['peak accessibility','peak-motif (TF with expr) sequence feature','peak-motif sequence feature']
		
		if len(method_type_vec)==0:
			method_type_dimension = select_config['method_type_dimension']
			method_type_vec = [method_type_dimension]*query_num1

		verbose_internal = self.verbose_internal
		for i1 in range(query_num1):
			feature_mtx_query = list_pre1[i1]
			feature_type_query = feature_type_vec_pre1[i1]
			feature_type_annot_query = feature_type_annot[i1]

			field_id1 = 'df_%s'%(feature_type_query)
			dict_query1.update({field_id1:feature_mtx_query}) # the feature matrix

			query_id_1 = feature_mtx_query.index.copy()
			if verbose_internal>0:
				print('feature matrix (feature type: %s), dataframe of size ',feature_mtx_query.shape,feature_type_annot_query,i1)

			if (flag_shuffle>0):
				query_num = len(query_id_1)
				id1 = np.random.permutation(query_num)
				query_id_1 = query_id_1[id1]
				feature_mtx_query = feature_mtx_query.loc[query_id_1,:]

			method_type = method_type_vec[i1]

			# perform feature dimension reduction
			dimension_model, df_latent, df_component = self.test_query_feature_pre2(feature_mtx=feature_mtx_query,
																					method_type=method_type,
																					n_components=n_components,
																					sub_sample=sub_sample,
																					verbose=verbose,select_config=select_config)

			feature_dim_vec = ['feature%d'%(id1+1) for id1 in range(n_components)]
			feature_vec_1 = query_id_1
			df_latent = pd.DataFrame(index=feature_vec_1,columns=feature_dim_vec,data=df_latent)

			feature_vec_2 = feature_mtx_query.columns
			df_component = df_component.T
			df_component = pd.DataFrame(index=feature_vec_2,columns=feature_dim_vec,data=df_component)

			if feature_type_query in ['peak_tf']:
				if flag_peak_tf_combine>0:
					feature_query_vec = list(peak_query_vec)+list(gene_query_vec)
					df_latent = df_latent.loc[feature_query_vec,:]

					df_latent_gene = df_latent.loc[gene_query_vec,:]
					dict_query1.update({'latent_gene':df_latent_gene})
					df_latent_peak = df_latent.loc[peak_query_vec,:]
				else:
					df_latent = df_latent.loc[peak_query_vec,:]
					df_latent_peak = df_latent
				df_latent_query = df_latent
			else:
				df_latent_query = df_latent.loc[peak_query_vec,:]
				df_latent_peak = df_latent_query
				
			if (verbose_internal>0):
				feature_type_annot_query1 = feature_type_annot_query
				flag_2 = ((feature_type_query in ['peak_tf']) and (flag_peak_tf_combine>0))
				if flag_2>0:
					feature_type_annot_query1 = '%s and TF exprs'%(feature_type_annot_query)
					
				print('feature embeddings using %s, dataframe of size '%(feature_type_annot_query1),df_latent_query.shape)
				print('data preview:\n',df_latent_query[0:2])
				print('component_matrix, dataframe of size ',df_component.shape)

				if flag_2>0:
					print('peak embeddings using %s, dataframe of size '%(feature_type_annot_query1),df_latent_peak.shape)
					print('data preview:\n',df_latent_peak[0:2])

			field_query_pre1 = ['dimension_model','latent','component']
			field_query_1 = ['%s_%s'%(field_id_query,feature_type_query) for field_id_query in field_query_pre1]
			list_query1 = [dimension_model, df_latent_query, df_component]
			for (field_id1,query_value) in zip(field_query_1,list_query1):
				dict_query1.update({field_id1:query_value})

			if save_mode>0:
				filename_save_annot_2 = '%s.%s_%s'%(feature_type_query,method_type,n_components)
				output_filename_1 = '%s/%s.dimension_model.%s.1.h5'%(output_file_path,filename_prefix_save,filename_save_annot_2)
				pickle.dump(dimension_model, open(output_filename_1, 'wb'))

				field_query_2 = ['df_latent','df_component']
				list_query2 = [df_latent_query,df_component]
				for (field_id,df_query) in zip(field_query_2,list_query2):
					filename_prefix_save_2 = '%s.%s'%(filename_prefix_save,field_id)
					output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_save_2,filename_save_annot_2)
					df_query.to_csv(output_filename,sep='\t',float_format=float_format)

		return dict_query1

	## ====================================================
	# query peak-motif matrix and motif scores by motif scanning for peak loci
	# query TFs with motifs and expressions
	def test_query_motif_data_annotation_1(self,data=[],gene_query_vec=[],feature_query_vec=[],method_type='',peak_read=[],rna_exprs=[],verbose=0,select_config={}):

		"""
		query peak-motif matrix and motif scores by motif scanning for given peak loci;
		query TFs with motifs and expressions;
		:param data: dictionary containing motif scanning data used the method for initial prediction of peak-TF associations
		:param gene_query_vec: (array) genes with expressions or TFs with expressions to include in analysis
		:param feature_query_vec: (array) selected peak loci; if not specified, genome-wide peak loci are included
		:param method_type: method used for initially predicting peak-TF associations
		:param peak_read: (dataframe) peak accessibility matrix (row: metacell; column: peak)
		:param rna_exprs: (dataframe) gene expression matrix (row: metacell; column: gene)
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1-2. (dataframe) binary matrix of motif presence in peak loci and motif score matrix by motif scanning (row: peak; column: TF (associated with motif))
				 3. (array) TFs with motifs and expressions
		"""

		flag_query1 = 1
		if flag_query1>0:
			dict_motif_data_ori = data
			method_type_query = method_type
			print('method for predicting peak-TF associations: %s'%(method_type_query))

			if method_type_query in dict_motif_data_ori:
				dict_motif_data = dict_motif_data_ori[method_type_query]
			else:
				dict_motif_data = dict_motif_data_ori

			peak_loc_1 = feature_query_vec
			motif_data_query1 = dict_motif_data['motif_data']
			flag_1 = (len(feature_query_vec)>0) # query motif data of the given peak loci;
			if flag_1==0:
				if len(peak_read)>0:
					flag_1 = 1
					peak_loc_1 = peak_read.columns # use the peaks included in the peak accessibility matrix
					
			if flag_1>0:
				motif_data_query1 = motif_data_query1.loc[peak_loc_1,:]

			verbose_internal = self.verbose_internal
			if verbose_internal>0:
				print('motif scanning data (binary), dataframe of size ',motif_data_query1.shape)
				print('preview:')
				print(motif_data_query1[0:2])
			
			if 'motif_data_score' in dict_motif_data:
				motif_data_score_query1 = dict_motif_data['motif_data_score']
				if flag_1>0:
					motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_1,:]

				print('motif scores, dataframe of size ',motif_data_score_query1.shape)
				print('preview:')
				print(motif_data_score_query1[0:2])

			else:
				motif_data_score_query1 = motif_data_query1

			# query TFs with motifs and expressions
			motif_query_vec = self.test_query_motif_annotation_1(data=motif_data_query1,gene_query_vec=gene_query_vec,rna_exprs=rna_exprs)

			return motif_data_query1, motif_data_score_query1, motif_query_vec

	## ====================================================
	# query TFs with motifs and expressions
	def test_query_motif_annotation_1(self,data=[],gene_query_vec=[],rna_exprs=[]):

		"""
		query TFs with motifs and expressions
		:param data: (dataframe) the motif scanning data matrix (row: peak; column: TF motif)
		:param gene_query_vec: (array) genes with expression
		:param rna_exprs: (dataframe) gene expression matrix (row: metacell; column: gene)
		:return: (array) TFs with motifs and expressions
		"""

		motif_data_query1 = data
		motif_name_ori = motif_data_query1.columns
		if len(gene_query_vec)==0:
			if len(rna_exprs)>0:
				gene_name_expr_ori = rna_exprs.columns
				gene_query_vec = gene_name_expr_ori
				
		if len(gene_query_vec)>0:
			motif_query_vec = pd.Index(motif_name_ori).intersection(gene_query_vec,sort=False)
			print('motif_query_vec (with expression): ',len(motif_query_vec))
		else:
			motif_query_vec = motif_name_ori

		return motif_query_vec

	## ====================================================
	# compute feature embeddings of observations
	def test_query_feature_mtx_1(self,feature_query_vec=[],feature_type_vec=[],gene_query_vec=[],method_type_vec_dimension=[],n_components=50,
										motif_data=[],motif_data_score=[],peak_read=[],rna_exprs=[],load_mode=0,input_file_path='',
										save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=1,select_config={}):

		"""
		compute feature embeddings of observations (peak loci)
		:param feature_query_vec: (array) peak loci; if not specified, peaks in the peak accessibility matrix are included
		:param feature_type_vec: (array or list) feature types of feature representations of the observations
		:param gene_query_vec: (array) genes with expressions or TFs with expressions to include in analysis
		:param method_type_vec_dimension: (array or list) methods for feature dimension reduction for the different feature types
		:param n_components: (int) the nubmer of latent components used in feature dimension reduction 
		:param type_id_group: (int) the type of peak-motif sequence feature to use: 0: use motifs of TFs with expressions; 1: use all TF motifs
		:param motif_data: (dataframe) motif presence in peak loci by motif scanning (binary) (row:peak, column:TF (associated with motif))
		:param motif_data_score: (dataframe) motif scores by motif scanning (row:peak, column:TF (associated with motif))
		:param peak_read: (dataframe) peak accessibility matrix (normalized and log-transformed) (row:metacell, column:peak)
		:param rna_exprs: (dataframe) gene expression matrix (row:metacell, column:gene)
		:param load_mode: indicator of whether to compute feature embedding or load embeddings from saved files
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing feature embeddings of observations for each feature type
		"""

		if len(method_type_vec_dimension)==0:
			feature_type_num = len(feature_type_vec)
			method_type_vec_dimension = ['SVD']*feature_type_num

		column_1 = 'type_id_group'
		if column_1 in select_config:
			type_id_group = select_config[column_1]
		else:
			type_id_group = 0
			select_config.update({column_1:type_id_group})

		filename_prefix_save_2 = '%s.%d'%(filename_prefix_save,type_id_group)

		latent_peak = []
		latent_peak_motif,latent_peak_motif_ori = [], []
		latent_peak_tf_link = []
		if len(feature_query_vec)==0:
			feature_query_vec = peak_read.columns # include peaks in peak accessibility matrix

		flag_shuffle = False
		sub_sample = -1
		float_format='%.6f'
		verbose_internal = self.verbose_internal
		if load_mode==0:
			# perform feature dimension reduction
			# dict_query1: {'latent_peak_tf','latent_peak_motif','latent_peak_motif_ori'}
			dict_query1 = self.test_query_feature_pre1(peak_query_vec=feature_query_vec,
														gene_query_vec=gene_query_vec,
														method_type_vec=method_type_vec_dimension,
														motif_data=motif_data,motif_data_score=motif_data_score,
														peak_read=peak_read,rna_exprs=rna_exprs,
														n_components=n_components,
														sub_sample=sub_sample,
														flag_shuffle=flag_shuffle,float_format=float_format,
														input_file_path=input_file_path,
														save_mode=save_mode,output_file_path=output_file_path,output_filename='',
														filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot,
														verbose=verbose,select_config=select_config)

		elif load_mode==1:
			# load computed feature embeddings
			input_file_path_query = output_file_path
			annot_str_vec = ['peak_motif','peak_tf']
			annot_str_vec_2 = ['peak-motif sequence feature','peak accessibility']
			field_query_2 = ['df_latent','df_component']
			dict_query1 = dict()

			query_num = len(annot_str_vec)
			for i2 in range(query_num):
				method_type_dimension = method_type_vec_dimension[i2]
				filename_save_annot_2 = '%s_%s'%(method_type_dimension,n_components)

				annot_str1 = annot_str_vec[i2]
				field_id1 = 'df_latent'
				
				filename_prefix_save_query = '%s.%s'%(filename_prefix_save_2,field_id1)
				input_filename = '%s/%s.%s.%s.1.txt'%(input_file_path_query,filename_prefix_save_query,annot_str1,filename_save_annot_2)
				df_query = pd.read_csv(input_filename,index_col=0,sep='\t')

				if verbose_internal>0:
					print('feature embedding using %s, dataframe of size '%(annot_str_vec_2[i2]),df_query.shape)
					print('data preview:\n ',df_query[0:2])

				feature_query_pre1 = df_query.index
				feature_query_pre2 = pd.Index(feature_query_vec).intersection(feature_query_pre1,sort=False)
				df_query = df_query.loc[feature_query_pre2,:]
				field_id2 = 'latent_%s'%(annot_str1)
				dict_query1.update({field_id2:df_query})

				if annot_str1 in ['peak_tf']:
					feature_vec_2 = pd.Index(feature_query_pre1).difference(feature_query_vec,sort=False)
					feature_vec_3 = pd.Index(feature_query_vec).difference(feature_query_pre1,sort=False)
					if len(gene_query_vec)==0:
						gene_query_pre2 = feature_vec_2
					else:
						gene_query_pre2 = pd.Index(gene_query_vec).intersection(feature_query_pre1,sort=False)

					latent_gene = df_query.loc[gene_query_pre2,:]
					if verbose_internal==2:
						print('feature_vec_2: %d',len(feature_vec_2))
						print('feature_vec_3: %d',len(feature_vec_3))
						print('latent_gene, dataframe of size ',latent_gene.shape)
						print('data preview:\n',latent_gene[0:2])
					dict_query1.update({'latent_gene':latent_gene})

		return dict_query1

	## ====================================================
	# select groups for each feature type based on enrichment of peak loci with predicted TF binding and peak number
	def test_query_enrichment_group_1(self,data=[],dict_group=[],dict_thresh=[],group_type_vec=['group1','group2'],column_vec_query=[],
										flag_enrichment=1,flag_size=1,type_id_1=1,type_id_2=1,save_mode=0,verbose=0,select_config={}):
		
		"""
		select groups for each feature type based on enrichment of peak loci with predicted TF binding and peak number
		:param data: (dataframe) annotations containing the number and percentage of specific peaks in each group for each feature type
		:param dict_group: dictionary of the group assignment of peaks in each feature space
		:param dict_thresh: dictionary containing thresholds for group (or paired groups) selection using peak number and peak enrichment
		:param group_type_vec: (array or list) group name for different feature types
		:param column_vec_query: (array or list) columns representing peak number in each group and p-value of peak enrichment in each group by statistical test
		:param flag_enrichment: indicator of whether to select groups using peak enrichment 
		:param flag_size: indicator of whether to select groups using peak number
		:param type_id_1: indicator of whether to adjust the threshold on peak number for group selection
		:param type_id_2: indicator of which criteria to use to select groups:
						  0: selecting based on both peak enrichment and peak number; 
						  1: selecting based on peak enrichment or peak number; 
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing a list with two elements for each feature type:
		         1. (dataframe) the annotations of groups selected based on peak enrichment and peak number;
				 2. dictionary containing two dataframes of group annotations:
				 	2.1 groups selected based on peak enrichment and requiring peak number above a threshold
				 	2.2 groups selected based on peak number only 
		"""

		flag_query1 = 1
		if flag_query1>0:
			df_query_1 = data  # the number and percentage of peaks in each group 
			dict_group_basic = dict_group # the group assignment of peaks in each feature space

			thresh_overlap_default_1 = 0
			thresh_overlap_default_2 = 0
			thresh_overlap = 0
			# thresh_pval_1 = 0.20
			thresh_pval_1 = 0.25
			column_1 = 'thresh_overlap_default_1'
			column_2 = 'thresh_overlap_default_2'
			column_3 = 'thresh_overlap'
			column_pval = 'thresh_pval_1'

			flag_1 = 1
			if flag_1>0:
				# feature type 1: peak-motif sequence feature
				group_type_1 = group_type_vec[0]
				id1 = (df_query_1['group_type']==group_type_1)
				df_query1_1 = df_query_1.loc[id1,:]
				# query the enrichment of peak loci with predicted TF binding in each group in feature space 1
				df_query_group1_1, dict_query_group1_1 = self.test_query_enrichment_group_2(data=df_query1_1,
																							dict_thresh=dict_thresh,
																							column_vec_query=column_vec_query,
																							flag_enrichment=flag_enrichment,
																							flag_size=flag_size,
																							type_id_1=type_id_1,
																							type_id_2=type_id_2,
																							type_group=0,
																							save_mode=save_mode,verbose=verbose,select_config=select_config)
				
				list1 = [df_query_group1_1, dict_query_group1_1]
				dict_query_1 = {group_type_1:list1}

				if len(group_type_vec)>1:
					# feature type 2: peak accessibility feature
					group_type_2 = group_type_vec[1]
					id2 = (df_query_1['group_type']==group_type_2)
					df_query1_2 = df_query_1.loc[id2,:]
					# query the enrichment of peak loci with predicted TF binding in each group in feature space 2
					df_query_group2_1, dict_query_group2_1 = self.test_query_enrichment_group_2(data=df_query1_2,
																								dict_thresh=dict_thresh,
																								column_vec_query=column_vec_query,
																								flag_enrichment=flag_enrichment,
																								flag_size=flag_size,
																								type_id_1=type_id_1,
																								type_id_2=type_id_2,
																								type_group=0,
																								save_mode=save_mode,verbose=verbose,select_config=select_config)

					list2 = [df_query_group2_1,dict_query_group2_1]
					dict_query_1.update({group_type_2:list2})

			return dict_query_1

	## ====================================================
	# select groups for each feature type or paired groups based on enrichment of peak loci with predicted TF binding and peak number
	def test_query_enrichment_group_2(self,data=[],dict_thresh=[],thresh_overlap=0,thresh_quantile=-1,thresh_pval=0.25,
										group_type_vec=['group1','group2'],column_vec_query=[],flag_enrichment=1,flag_size=0,
										type_id_1=1,type_id_2=0,type_group=0,save_mode=0,verbose=0,select_config={}):

		"""
		select groups for each feature type or paired groups based on enrichment of peak loci with predicted TF binding and peak number
		:param data: (dataframe) annotations containing the number of specific peaks in each group (or paired groups)
		:param dict_thresh: dictionary containing thresholds for group (or paired groups) selection using peak number and peak enrichment
		:param thresh_overlap: threshold on peak number in group or (paired groups) for group (or paired groups) selection
		:param thresh_quantile: threshold on the quantile of peak number in group (or paired groups) for selection
		:param thresh_pval: threshold on p-value of peak enrichment in group (or paired groups) for selection
		:param column_vec_query: (array or list) columns representing peak number in each group (or paired groups) and p-value of peak enrichment in each group (or paired groups) by statistical test
		:param flag_enrichment: indicator of whether to select groups (or paired groups) using peak enrichment 
		:param flag_size: indicator of whether to select groups (or paired groups) using peak number
		:param type_id_1: indicator of whether to adjust the threshold on peak number for group selection
		:param type_id_2: indicator of which criteria to use to select groups:
						  0: selecting based on both peak enrichment and peak number; 
						  1: selecting based on peak enrichment or peak number; 
		:param type_group: the type of group: 0: groups in one feature space; 
											  1: paired groups representing the intersection of members of each pair of groups in the two feature spaces;
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the annotations of groups (or paired groups) selected based on peak enrichment and peak number
				 2. dictionary containing two dataframes of group (or paired groups) annotations:
				 	2.1 groups selected based on peak enrichment and requiring peak number above a threshold
				 	2.2 groups selected based on peak number only
		"""

		flag_query1 = 1
		if flag_query1>0:
			df_overlap_query = data
			df_query_1 = df_overlap_query # annotations containing the number and percentage of specific peaks in each group (or paired groups)
			
			thresh_overlap_default_1 = 0
			thresh_overlap_default_2 = 0
			thresh_pval_1 = thresh_pval

			column_1 = 'thresh_overlap_default_1'
			column_2 = 'thresh_overlap_default_2'
			column_3 = 'thresh_overlap'
			column_pval = 'thresh_pval_1'

			if len(column_vec_query)==0:
				column_vec_query=['overlap','pval_fisher_exact_']

			column_query1, column_query2 = column_vec_query[0:2] # columns representing peak number in groups and p-value of peak enrichment in groups by statistical test
			verbose_internal = self.verbose_internal

			group_annot_vec = ['groups','paired groups']
			group_annot = group_annot_vec[type_group]

			df_overlap_query_pre1 = df_overlap_query
			dict_query = dict()
			if verbose_internal>0:
				print('the number of %s: %d'%(group_annot,df_query_1.shape[0]))

			if flag_enrichment>0:
				print('select %s based on enrichment of peaks with predicted TF binding'%(group_annot))
				if column_1 in dict_thresh:
					thresh_overlap_default_1 = dict_thresh[column_1] # threshold on candidate peak number (predicted TF-binding peaks) in the group

				if column_pval in dict_thresh:
					thresh_pval_1 = dict_thresh[column_pval]	# threshold on p-value of candidate peak enrichment in the group by statistical test

				flag1=0
				try:
					id1 = (df_query_1[column_query1]>thresh_overlap_default_1) # requiring candidate peak number above the threshold
				except Exception as error:
					print('error! ',error)
					flag1=1

				flag2=0
				try:
					id2 = (df_query_1[column_query2]<thresh_pval_1)
				except Exception as error:
					print('error! ',error)
					try: 
						# if Fisher's exact test was not perfomed, using p-value of candidate peak enrichment by chi-squared test
						column_query2_1 = 'pval_chi2_'
						id2 = (df_query_1[column_query2_1]<thresh_pval_1)
					except Exception as error:
						print('error! ',error)
						flag2=1

				id_1 = []
				if (flag2==0):
					if (flag1==0):
						id_1 = (id1&id2)  # selections based on candidate peak number and enrichment in the group were both performed
					else:
						id_1 = id2  # select groups based on candidate peak enrichment only
				else:
					if (flag1==0):
						id_1 = id1 	# select groups based on candidate peak number only

				if (flag1+flag2<2):
					df_overlap_query1 = df_query_1.loc[id_1,:]
					if verbose_internal>0:
						print('the number of selected %s enriched with predicted TF-binding peaks above threshold: '%(group_annot),df_overlap_query1.shape[0])
				else:
					df_overlap_query1 = []
					if verbose_internal>0:
						print('lacking information to select %s'%(group_annot))

			df_query_2 = df_query_1.loc[df_query_1[column_query1]>0]
			if verbose_internal>0:
				print('the number of %s with predicted TF-binding peak number above zero: '%(group_annot),df_query_2.shape[0])

			query_value_1 = df_query_1[column_query1] 	# candidate peak number in each group
			query_value_2 = df_query_2[column_query1]	# candidate peak number in groups with candidate peak number above zero
			quantile_vec_1 = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
			query_vec_1 = ['max','min','mean','median']+['percentile_%.2f'%(percentile) for percentile in quantile_vec_1]
			t_value_1 = utility_1.test_stat_1(query_value_1,quantile_vec=quantile_vec_1)
			t_value_2 = utility_1.test_stat_1(query_value_2,quantile_vec=quantile_vec_1)
			query_value = np.asarray([t_value_1,t_value_2]).T
			df_quantile_1 = pd.DataFrame(index=query_vec_1,columns=['value_1','value_2'],data=query_value)
			dict_query.update({'group_size_query':df_quantile_1})

			if flag_size>0:
				print('select %s based on the number of peaks with predicted TF binding'%(group_annot))
				if column_2 in dict_thresh:
					thresh_overlap_default_2 = dict_thresh[column_2]

				if column_3 in dict_thresh:
					thresh_overlap = dict_thresh[column_3]

				thresh_quantile_1 = thresh_quantile
				column_pre2 = 'thresh_quantile_overlap'
				if column_pre2 in dict_thresh:
					thresh_quantile_1 = dict_thresh[column_pre2]
					print('threshold on the quantile of the number of predicted TF-binding peaks in the %s: '%(group_annot),thresh_quantile_1)
				
				if thresh_quantile_1>0:
					# use threshold by quantile
					query_value = df_query_2[column_query1]
					thresh_size_1 = np.quantile(query_value,thresh_quantile_1) 
					if type_id_1>0:
						thresh_size_ori = thresh_size_1
						thresh_size_1 = np.max([thresh_overlap_default_2,thresh_size_1])
				else:
					thresh_size_1 = thresh_overlap  # use the specified threshold by candidate peak number

				id_2 = (df_query_1[column_query1]>=thresh_size_1)
				df_overlap_query2 = df_query_1.loc[id_2,:]  # select groups with candidate peak number above threshold
				if verbose_internal>0:
					print('the number of %s: %d'%(group_annot,df_query_1.shape[0]))
					print('the number of %s with predicted TF-binding peak number above threshold: %d'%(group_annot,df_overlap_query2.shape[0]))
					print('threshold on the predicted TF-binding peak number: ',thresh_size_1)

				if flag_enrichment>0:
					if type_id_2==0:
						id_pre1 = (id_1&id_2) # select groups (or paired groups) based on both candiate peak enrichment and number
					else:
						id_pre1 = (id_1|id_2) # select groups (or paired groups) based on candiate peak enrichment or number
					df_overlap_query_pre1 = df_query_1.loc[id_pre1,:]

					df_overlap_query_pre1.loc[id_1,'enrichment'] = 1  # the groups (or paired groups) with candiate peak enrichment above threshold
					df_overlap_query_pre1.loc[id_2,'group_size'] = 1  # the groups (or paired groups) with candiate peak number above threshold
					if verbose_internal>0:
						if type_id_2==0:
							annot_str_1 = 'peak number and peak enrichment'
						else:
							annot_str_1 = 'peak number or peak enrichment'
						print('the number of selected %s using thresholds on %s: %d'%(group_annot,annot_str_1,df_overlap_query_pre1.shape[0]))
				else:
					df_overlap_query_pre1 = df_overlap_query2  # select groups (or paired groups) based on candidate peak number only
			else:
				df_overlap_query_pre1 = df_overlap_query1    # select groups (or paired groups) based on candidate peak enrichment and requiring peak number above a threshold

			dict_query.update({'enrichment':df_overlap_query1,'group_size':df_overlap_query2})
			return df_overlap_query_pre1, dict_query

	## ====================================================
	# select paired groups based on enrichment of peak loci with predicted TF binding and peak number
	def test_query_training_group_pre1(self,data=[],dict_annot=[],motif_id='',dict_thresh=[],thresh_vec=[],flag_select_2=0,input_file_path='',save_mode=1,output_file_path='',verbose=0,select_config={}):

		"""
		select paired groups based on enrichment of peak loci with predicted TF binding and peak number
		:param data: (dataframe) peak annotations including group assignment in the two feature spaces
		:param dict_annot: dictionary containing the following data:
						   1. (dataframe) annotations containing the number and percentage of candiate peaks (peaks with initially predicted TF binding) in each group for each feature type
						   2. dictionary of the group assignment of peaks in each feature space
						   3. (dataframe) annotations containing the number of candidate peaks in paired groups
		:param motif_id: (str) name of the TF for which we perform TF binding prediction in peak loci
		:param dict_thresh: dictionary containing thresholds for group (or paired groups) selection using peak number and peak enrichment
		:param thresh_vec: (array or list) thresholds for group (or paired groups) selection using peak number and peak enrichment
		:param flag_select_2: indicator of whether to include paired groups without candidate peak enrichment but associated with a group with enrichment of candidate peaks in the individual feature space
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) the updated peak annotations with the column representing selected paired groups
		"""

		df_query1 = data   # peak annotations including group assignment in the two feature spaces
		flag_select_1=1
		if flag_select_1>0:
			# search for the paired groups with enrichment of peak loci with initially predicted TF binding
			column_1 = 'thresh_overlap_default_1'
			column_2 = 'thresh_overlap_default_2'
			column_3 = 'thresh_overlap'
			column_pval_group = 'thresh_pval_1'
			column_quantile = 'thresh_quantile_overlap'
			column_thresh_query = [column_1,column_2,column_3,column_pval_group,column_quantile]
			verbose_internal = self.verbose_internal

			if len(dict_thresh)==0:
				if len(thresh_vec)==0:
					thresh_overlap_default_1 = 0
					thresh_overlap_default_2 = 0
					thresh_overlap = 0
									
					# thresh_pval_group = 0.20
					thresh_pval_group = 0.25
					thresh_quantile_overlap = 0.75
					thresh_vec = [thresh_overlap_default_1,thresh_overlap_default_2,thresh_overlap,thresh_pval_group,thresh_quantile_overlap]
				
				# dictionary containing thresholds for group selection using peak number and peak enrichment
				dict_thresh = dict(zip(column_thresh_query,thresh_vec))

			# -----------------------------------------------------
			# select groups for each feature type based on enrichment of peak loci with predicted TF binding and peak number
			group_type_vec = ['group1','group2']
			df_group_basic_query_2 = dict_annot['df_group_basic_query_2']
			dict_group_basic_2 = dict_annot['dict_group_basic_2']

			if verbose_internal>0:
				print('peak group annotation, dataframe of size ',df_group_basic_query_2.shape)
				print('columns: ',np.asarray(df_group_basic_query_2.columns))
				print('data preview: ')
				print(df_group_basic_query_2[0:5])
			
			column_vec_query = ['count','pval_fisher_exact_']
			flag_enrichment = 1
			flag_size = 1
			type_id_1, type_id_2 = 1, 1

			dict_query_pre1 = self.test_query_enrichment_group_1(data=df_group_basic_query_2,
																	dict_group=dict_group_basic_2,
																	dict_thresh=dict_thresh,
																	group_type_vec=group_type_vec,
																	column_vec_query=column_vec_query,
																	flag_enrichment=flag_enrichment,
																	flag_size=flag_size,
																	type_id_1=type_id_1,
																	type_id_2=type_id_2,
																	save_mode=1,verbose=verbose,select_config=select_config)
							
			group_type_1, group_type_2 = group_type_vec[0:2]
			df_query_group1_1,dict_query_group1_1 = dict_query_pre1[group_type_1]
			df_query_group2_1,dict_query_group2_1 = dict_query_pre1[group_type_2]

			field_query_2 = ['enrichment','group_size','group_size_query']
			field_id1, field_id2 = field_query_2[0:2]
			field_id3 = field_query_2[2]

			dict_query1 = dict()
			save_mode_2 = 0

			for group_type in group_type_vec:
				print('group_type: ',group_type)
				dict_query_group = dict_query_pre1[group_type][1]
				group_vec_query1_1 = dict_query_group[field_id1].index.unique()
				group_vec_query2_1 = dict_query_group[field_id2].index.unique()
				group_num1_1, group_num2_1 = len(group_vec_query1_1), len(group_vec_query2_1)
				dict_query1.update({group_type:[group_vec_query1_1,group_vec_query2_1]})
				
				if verbose_internal>0:
					print('the number of groups with enrichment of predicted TF-binding peaks above threshold: %d'%(group_num1_1))
					print('groups: ',np.asarray(group_vec_query1_1))
					print('the number of groups with predicted TF-binding peak number above threshold: %d'%(group_num2_1))
					print('groups: ',np.asarray(group_vec_query2_1))

				if save_mode_2>0:
					df_quantile_1 = dict_query_group[field_id3]
					filename_link_annot = select_config['filename_annot']
					output_filename = '%s/test_query_quantile.%s.%s.txt'%(output_file_path,motif_id,filename_link_annot)
					df_quantile_1.to_csv(output_filename,sep='\t')

			group_vec_query1_1, group_vec_query2_1 = dict_query1[group_type_1]
			group_vec_query1_2, group_vec_query2_2 = dict_query1[group_type_2]

			# -----------------------------------------------------
			# select paired groups based on enrichment of peak loci with predicted TF binding and peak number
			column_vec_query_2 = ['overlap','pval_fisher_exact_']
			flag_enrichment = 1
			flag_size = 1
			type_id_1, type_id_2 = 1, 1
			df_overlap_query = dict_annot['df_overlap_query']

			df_overlap_query_pre2, dict_query_pre2 = self.test_query_enrichment_group_2(data=df_overlap_query,
																						dict_thresh=dict_thresh,
																						group_type_vec=group_type_vec,
																						column_vec_query=column_vec_query_2,
																						flag_enrichment=flag_enrichment,
																						flag_size=flag_size,
																						type_id_1=type_id_1,
																						type_id_2=type_id_2,
																						type_group=1,
																						save_mode=1,verbose=verbose,select_config=select_config)
			
			group_vec_query2 = np.asarray(df_overlap_query_pre2.loc[:,group_type_vec].astype(int))
			group_num_2 = len(group_vec_query2)
			if verbose_internal>0:
				print('the number of selected paired groups: %d'%(group_num_2))
				print('preview: ')
				print(group_vec_query2[0:5])

			df_1 = dict_query_pre2[field_id1] # paired groups with peak enrichment above threshold
			group_vec_query1_pre2 = df_1.loc[:,group_type_vec].astype(int)
			group_num1_pre2 = len(group_vec_query1_pre2)
			
			df_2 = dict_query_pre2[field_id2] # paired groups with group size above threshold
			group_vec_query2_pre2 = df_2.loc[:,group_type_vec].astype(int)
			group_num2_pre2 = len(group_vec_query2_pre2)

			if verbose_internal==2:
				print('the number of selected paired groups with enrichment of predicted TF-binding peaks above threshold: ',group_num1_pre2)
				print(df_1)

				print('the number of selected paired groups with predicted TF-binding peak number above threshold: ',group_num2_pre2)
				print(df_2)

			group_vec_1 = group_vec_query1_1
			group_vec_1_overlap = group_vec_query1_pre2[group_type_1].unique()

			group_vec_2 = group_vec_query1_2
			group_vec_2_overlap = group_vec_query1_pre2[group_type_2].unique()

			group_vec_pre1 = pd.Index(group_vec_1).difference(group_vec_1_overlap,sort=False)
			group_vec_pre2 = pd.Index(group_vec_2).difference(group_vec_2_overlap,sort=False)
			
			list_query1 = [group_vec_pre1,group_vec_pre2]
			feature_type_num = len(list_query1)
			if verbose_internal>0:
				for i1 in range(feature_type_num):
					group_vec_query = list_query1[i1]
					print('the number of groups with enrichment for feature type %d but not enriched in paired groups: %d'%(i1+1,len(group_vec_query)))

			list1 = []
			list2 = []
			column_query1, column_query2 = column_vec_query_2[0:2]
			df_overlap_query = df_overlap_query.sort_values(by=['pval_fisher_exact_'],ascending=True)
			thresh_size_query1 = 1

			if flag_select_2>0:
				# include paired groups without peak enrichment but associated with a group with enrichment in one feature space
				list_query = []
				for (group_type,group_vec_query) in zip(group_type_vec,list_query1):
					list_group = []
					for group_type_query in group_vec_query:
						df1 = df_overlap_query.loc[df_overlap_query[group_type]==group_type_query,:] # paired groups associated with the given group in the feature space
						df1_1 = df1.loc[df1[column_query1]>thresh_size_query1,:]
						group_query_1 = np.asarray(df1.loc[:,group_type_vec])[0] # the paired groups with the relatively smallest peak enrichment p-value
						group_query_2 = np.asarray(df1_1.loc[:,group_type_vec])	 # the paired groups with peak number above threshold
						list_group.append(group_query_1)
						list_group.extend(group_query_2)
					list_query.append(list_group)
				list1, list2 = list_query[0:2]

			group_vec_query1_pre2 = np.asarray(group_vec_query1_pre2)
			list_pre1 = list(group_vec_query1_pre2)+list1+list2

			query_vec = np.asarray(list_pre1)
			df_1 = pd.DataFrame(data=query_vec,columns=group_type_vec).astype(int)
			df_1.index = utility_1.test_query_index(df_1,column_vec=[group_type_1,group_type_2],symbol_vec=['_'])
			df_1 = df_1.drop_duplicates(subset=group_type_vec)
			group_id_1 = df_1.index
			group_id_pre1 = ['%s_%s'%(group_1,group_2) for (group_1,group_2) in group_vec_query1_pre2] # paired groups with peak enrichment above threshold
			group_id_2 = group_id_1.difference(group_id_pre1,sort=False)

			column_query_1 = df_overlap_query.columns.difference(group_type_vec,sort=False)
			df_1.loc[:,column_query_1] = df_overlap_query.loc[group_id_1,column_query_1]
			group_vec_query1 = df_1.loc[:,group_type_vec]

			feature_type_vec = select_config['feature_type_vec']
			feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]
			group_type_vec_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]

			df_query1['group_id2'] = utility_1.test_query_index(df_query1,column_vec=group_type_vec_2,symbol_vec=['_'])
			id1 = (df_query1['group_id2'].isin(group_id_1))
			id2 = (df_query1['group_id2'].isin(group_id_2))
			df_query1.loc[id1,'group_overlap'] = 1 # selected paired groups
			df_query1.loc[id2,'group_overlap'] = 2 # the paired groups without peak enrichment but associated with the single group enriched with peaks of predicted TF binding in the feature space
			# print('group_id_1, group_id_2: ',len(group_id_1),len(group_id_2))

			return df_query1

	## ====================================================
	# compute quantiles of the feature association scores
	def test_query_feature_quantile_1(self,data=[],query_idvec=[],column_vec_query=[],verbose=0,select_config={}):

		"""
		compute quantiles of the feature association scores
		:param data: (dataframe) peak annotations including initially predicated peak-TF associations for the given TF
		:param query_idvec: (array) selected peak loci; if not specified, using all the peaks in the peak annoation data
		:param column_vec_query: (list) columns in peak annotation data to retrieve information from 
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary storing parameters
		:return: dictionary storing updated parameters, including column names corresponding to computed score quantiles
		"""

		if len(column_vec_query)==0:
			column_corr_1 = 'peak_tf_corr'
			column_pval = 'peak_tf_pval_corrected'
			method_type_feature_link = select_config['method_type_feature_link']
			column_score_query1 = '%s.score'%(method_type_feature_link)
			column_vec_query = [column_corr_1,column_pval,column_score_query1]
		else:
			column_corr_1, column_pval = column_vec_query[0:2]
			column_score_query1 = column_vec_query[2]

		thresh_corr_1, thresh_pval_1 = 0.30, 0.05
		thresh_corr_2, thresh_pval_2 = 0.1, 0.1
		thresh_corr_3, thresh_pval_2 = 0.05, 0.1

		df_query1 = data
		column_id2 = 'peak_id'
		if not (column_id2 in df_query1.columns):
			df_query1['peak_id'] = df_query1.index.copy()

		query_value_1 = df_query1[column_corr_1]
		query_value_1 = query_value_1.fillna(0)

		column_quantile_pre1 = '%s_quantile'%(column_corr_1)
		normalize_type = 'uniform'	# normalize_type: 'uniform', 'normal'
		score_mtx = quantile_transform(np.asarray(query_value_1)[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
		df_query1[column_quantile_pre1] = score_mtx[:,0]

		if len(query_idvec)>0:
			df_pre2 = df_query1.loc[query_idvec,:] # peak annotations for selected peak loci
		else:
			df_pre2 = df_query1
			query_idvec = df_query1.index

		df_pre2 = df_pre2.sort_values(by=[column_score_query1],ascending=False)

		query_value = df_pre2[column_corr_1]
		query_value = query_value.fillna(0)

		column_quantile_1 = '%s_quantile_2'%(column_corr_1)
		normalize_type = 'uniform'   # normalize_type: 'uniform', 'normal'
		score_mtx = quantile_transform(np.asarray(query_value)[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
		df_query1.loc[query_idvec,column_quantile_1] = score_mtx[:,0]
							
		query_value_2 = df_pre2[column_score_query1]
		query_value_2 = query_value_2.fillna(0)

		column_quantile_2 = '%s_quantile'%(column_score_query1)
		normalize_type = 'uniform'   # normalize_type: 'uniform', 'normal'
		score_mtx_2 = quantile_transform(np.asarray(query_value_2)[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
		df_query1.loc[query_idvec,column_quantile_2] = score_mtx_2[:,0]

		column_vec_quantile = [column_quantile_pre1,column_quantile_1,column_quantile_2]
		select_config.update({'column_vec_quantile':column_vec_quantile})

		return df_query1, select_config

	## ====================================================
	# select pseuso postive training sample utilizing peak accessibility-TF expression correlation
	def test_query_training_select_correlation_1(self,data=[],thresh_vec=[0.1,0.90],verbose=0,select_config={}):
		
		"""
		select pseuso postive training sample utilizing peak accessibility-TF expression correlation value
		:param data: (dataframe) annotations of peak loci with TF binding predicted by the first method
		:param thrseh_vec: (array or list) the thresholds on peak accessibility-TF expression correlation to select pseuso postive training samples
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (array) selected peak loci using threshold on peak accessibility-TF expression correlation and p-value
				 2. (array) selected peak loci using threshold on quantile of peak accessibility-TF expression correlation
		"""

		flag_corr_1 = 1
		if flag_corr_1>0:
			# select peak with peak accessibility-TF expression correlation above threshold
			df_query = data
			peak_loc_query = df_query.index

			column_corr_1 = 'peak_tf_corr'
			column_pval = 'peak_tf_pval_corrected'
			if len(thresh_vec)>0:
				thresh_corr_1, thresh_corr_quantile = thresh_vec[0:2]
			else:
				thresh_corr_1 = 0.1
				thresh_corr_quantile = 0.90
			thresh_pval_1 = 0.1

			# peak loci with correlation above threshold and p-value below threshold
			# id_score_query2_1 = (df_query[column_corr_1]>thresh_corr_1)&(df_query[column_pval]<thresh_pval_1)
			id_score_query_1 = (df_query[column_corr_1]>thresh_corr_1)&(df_query[column_pval]<thresh_pval_1)

			# query_value_1 = df_query1.loc[id_pred2,column_corr_1]
			query_value = df_query[column_corr_1]
			query_value = query_value.fillna(0)
			
			thresh_corr_query1 = np.quantile(query_value,thresh_corr_quantile)
			thresh_corr_query2 = np.min([thresh_corr_1,thresh_corr_query1])
			
			# print('thresh_corr_query1, thresh_corr_query2: ',thresh_corr_query1, thresh_corr_query2)
			# id_score_query_2_1 = (df_query[column_corr_1]>thresh_corr_query2)&(df_query[column_pval]<thresh_pval_1)
			# peak loci with correlation above threshold
			 #id_score_query_2_1 = (df_query[column_corr_1]>thresh_corr_query2)
			id_score_query_2 = (df_query[column_corr_1]>thresh_corr_query2)
								
			# peak_loc_query_2_1 = peak_loc_query[id_score_query2_1]
			# peak_loc_query_2 = peak_loc_query[id_score_query_2_1]

			# peak_loc_query_group2_1 = peak_loc_query_2
			# feature_query_vec_1 = peak_loc_query_2_1
			# feature_query_vec_2 = peak_loc_query_2

			feature_query_vec_1 = peak_loc_query[id_score_query_1]
			feature_query_vec_2 = peak_loc_query[id_score_query_2]

			verbose_internal = self.verbose_internal
			if verbose_internal>0:
				print('the original and the adjusted thresholds on quantile of peak accessibility-TF expression correlation: ',thresh_corr_query1,thresh_corr_query2)
				print('selected peak loci using threshold on peak accessibility-TF expression correlation and p-value: %d'%(len(feature_query_vec_1)))
				print('selected peak loci using threshold on quantile of peak accessibility-TF expression correlation: %d'%(len(feature_query_vec_2)))

			return feature_query_vec_1, feature_query_vec_2

	## ====================================================
	# select pseudo positive training sample based on peak-TF association score and peak accessibility-TF expression correlation
	def test_query_training_select_feature_link_score_1(self,data=[],column_vec_query=[],thresh_vec=[],save_mode=1,verbose=0,select_config={}):

		"""
		select pseudo positive training sample based on peak-TF association score and peak accessibility-TF expression correlation
		:param data: (dataframe) annotations of peak loci with initially predicted TF binding, including estimated peak-TF association scores
		:param column_vec_query: (array or list) columns representing peak accessibility-TF expression correlation and estimated peak-TF association score in the peak annotation dataframe
		:param thresh_vec: (array or list) thresholds used to select peak loci based on peak-TF association score and peak accessibility-TF expression correlation
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (array) selected peak loci based on peak-TF association score and peak accessibility-TF expression correlation
		"""

		flag_score_1=1
		if flag_score_1>0:
			df_query1 = data
			if len(thresh_vec)>0:
				thresh_score_query_pre1, thresh_score_query_1, thresh_corr_query = thresh_vec[0:3]
			else:
				thresh_corr_query = 0.10
				thresh_score_query_pre1 = 0.15
				thresh_score_query_1 = 0.15
			column_1 = 'thresh_score_group_1'
			if column_1 in select_config:
				thresh_score_group_1 = select_config[column_1]
				thresh_score_query_1 = thresh_score_group_1

			column_corr_1, column_score_query1 = column_vec_query[0:2]
			id_score_query1 = (df_query1[column_score_query1]>thresh_score_query_1)  # select peak loci based on estimated peak-TF association score
			id_score_query1 = (id_score_query1)&(df_query1[column_corr_1]>thresh_corr_query)  # requiring peak accessibility-TF expression correlation above the threshold
			df_query1_2 = df_query1.loc[id_score_query1,:]

			peak_loc_query_1 = df_query1_2.index 	# the peak loci with prediction and with score above threshold
			peak_num_1 = len(peak_loc_query_1)
			print('the number of selected peak loci with predicted TF binding and with score above threshold: %d'%(peak_num_1))

			feature_query_vec_1 = peak_loc_query_1

			return feature_query_vec_1

	## ====================================================
	# select pseudo positive training sample
	def test_query_training_select_pre1(self,data=[],column_vec_query=[],flag_corr_1=1,flag_score_1=0,flag_enrichment_sel=1,input_file_path='',save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		select pseudo positive training sample
		:param data: (dataframe) annotations of peak loci with TF binding predicted by the first method
		:param column_vec_query: (array or list) columns representing peak accessibility-TF expression correlation and estimated peak-TF association score in the peak annotation dataframe
		:param flag_corr_1: indicating whether to select peak loci using peak accessibility-TF expression correlation
		:param flag_score_1: indicating whether to select peak loci using peak-TF association score and peak accessibility-TF expression correlation
		:param flag_enrichment_sel: indicating whether to select peak loci using different thresholds depending on peak enrichment in paired groups
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: fillename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (array) selected pseudo positive training sample
		"""

		df_query1 = data  # annotations of peak loci with TF binding predicted by the first method
		column_vec_quantile = select_config['column_vec_quantile']
		column_quantile_pre1,column_quantile_1,column_quantile_2 = column_vec_quantile[0:3]
		df_query1 = df_query1.sort_values(by=['group_overlap',column_quantile_pre1],ascending=False)
						
		flag_query1=1
		if flag_query1>0:
			peak_loc_query_1 = []
			peak_loc_query_2 = []
			if len(column_vec_query)==0:
				method_type_feature_link = select_config['method_type_feature_link']
				column_corr_1 = 'peak_tf_corr'
				column_score_query1 = '%s.score'%(method_type_feature_link)
				column_vec_query = [column_corr_1,column_score_query1]
			else:
				column_corr_1, column_score_query1 = column_vec_query[0:2]

			peak_loc_query_2_1 = []
			peak_loc_query_2 = []
			if flag_corr_1>0:
				# select peak loci using peak accessibility-TF expression correlation
				thresh_corr_1 = 0.1
				thresh_corr_quantile = 0.90
				thresh_vec_1 = [thresh_corr_1,thresh_corr_quantile]
				peak_loc_query_2_1, peak_loc_query_2 = self.test_query_training_select_correlation_1(data=df_query1,thresh_vec=thresh_vec_1,verbose=verbose,select_config=select_config)

			peak_loc_query_1 = []
			if flag_score_1>0:
				# select peak loci based on peak-TF association score and peak accessibility-TF expression correlation
				thresh_score_query_pre1 = 0.15
				thresh_score_query_1 = 0.15
				thresh_corr_1 = 0.1
				thresh_vec_2 = [thresh_score_query_pre1,thresh_score_query_1,thresh_corr_1]
				peak_loc_query_1 = self.test_query_training_select_feature_link_score_1(data=df_query1,column_vec_query=column_vec_query,
																						thresh_vec=thresh_vec_2,
																						verbose=verbose,select_config=select_config)		

			peak_loc_query_pre2 = pd.Index(peak_loc_query_2).union(peak_loc_query_1,sort=False)
			feature_query_vec_pre2 = peak_loc_query_pre2
			peak_loc_query_group2_1 = feature_query_vec_pre2
			verbose_internal = self.verbose_internal

			if flag_enrichment_sel>0:
				# select peak loci using different thresholds depending on peak enrichment in paired groups
				# thresh_vec_query1=[0.25,0.75]
				thresh_vec_query1=[0.5,0.9]
				column_1 = 'thresh_vec_sel_1'
				if column_1 in select_config:
					thresh_vec_query1 = select_config[column_1]

				if verbose_internal>0:
					print('threshold on the quantile of peak-TF link score for selection: ',thresh_vec_query1)
				thresh_vec_query2=[0.95,0.001]
				
				# column_vec_quantile = select_config['column_vec_quantile']
				# column_quantile_pre1,column_quantile_1,column_quantile_2 = column_vec_quantile[0:3]
				column_vec_query_2 = [column_quantile_1,column_quantile_2]
				feature_query_group1 = self.test_query_training_select_pre2(data=df_query1,feature_query_vec=feature_query_vec_pre2,
																				column_vec_query=column_vec_query_2,
																				thresh_vec_1=thresh_vec_query1,
																				thresh_vec_2=thresh_vec_query2,
																				verbose=verbose,select_config=select_config)

			return feature_query_group1

	## ====================================================
	# select pseudo positive training sample using different thresholds depending on peak enrichment in paired groups
	def test_query_training_select_pre2(self,data=[],feature_query_vec=[],column_vec_query=[],thresh_vec_1=[0.25,0.75],thresh_vec_2=[0.95,0.001],save_mode=1,verbose=0,select_config={}):
		
		"""
		select pseudo positive training sample using different thresholds depending on peak enrichment in paired groups
		:param data: (dataframe) annotations of peak loci with TF binding predicted by the first method
		:param feature_query_vec: (array) the currently selected pseudo positive training samples
		:param column_vec_query: (array or list) columns (in the peak annotation dataframe) representing quantile of scores computed using peaks with initially predicted TF binding and computed using all the peaks
		:param thresh_vec_1: (array or list) thresholds on quantile of score for paired groups with or without enrichment of peaks with initially predicted TF binding
		:param thresh_vec_2: (array or list) upper and lower thresholds on peak accessibility-TF expression correlation
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (array) selected pseudo positive training samples
		"""

		df_query = data  # annotations of peak loci with TF binding predicted by the first method (candidate peaks)
		id_group_overlap = (df_query['group_overlap']>0) # find paired groups with candidate peak enrichment above threshold
		flag_enrichment_sel = 1
		if flag_enrichment_sel>0:
			id1 = (id_group_overlap)
			id2 = (~id_group_overlap)  # find paired groups without enrichment of candidate peaks
			group_id_query = df_query.loc[id1,'group_id2'].unique()  # name of paired groups with candidate peak enrichment above threshold

			if len(thresh_vec_1)>0:
				thresh_1, thresh_2 = thresh_vec_1[0:2]
			else:
				thresh_1, thresh_2 = 0.25, 0.75

			thresh_quantile_query1_1, thresh_quantile_query2_1 = thresh_1, thresh_1  # threshold on quantile of score for groups enriched with candidata peaks
			thresh_quantile_query1_2, thresh_quantile_query2_2 = thresh_2, thresh_2  # threshold on quantile of score for groups without enrichment of candidata peaks

			# column_quantile_1: quantile of scores computed using the peaks with predicted TF binding
			# column_quantile_2: quantile of scores computed using all the peaks
			column_quantile_1, column_quantile_2 = column_vec_query[0:2]
			id_score_1 = (df_query[column_quantile_1]>thresh_quantile_query1_1) # lower threshold for groups enriched with candidata peaks
			id_score_2 = (df_query[column_quantile_2]>thresh_quantile_query2_1)

			id1_1 = id1&(id_score_1|id_score_2)
			# id2_1 = id2&(id_score_1|id_score_2)

			id_score_1_2 = (df_query[column_quantile_1]>thresh_quantile_query1_2) # higher threshold for groups without enrichment of candidate peaks
			id_score_2_2 = (df_query[column_quantile_2]>thresh_quantile_query2_2)

			# id1_2 = id1&(id_score_1_2|id_score_2_2)
			id2_2 = id2&(id_score_1_2|id_score_2_2)
			id_query_2 = (id1_1|id2_2)

			if len(thresh_vec_2)>0:
				thresh_corr_uppper_bound, thresh_corr_lower_bound = thresh_vec_2[0:2]
			else:
				thresh_corr_uppper_bound, thresh_corr_lower_bound = 0.95, 0.001

			column_corr_1 = 'peak_tf_corr'
			if thresh_corr_lower_bound>0:
				print('use threshold (lower bound) on peak accessibility-TF expression correlation for selection')
				print('threshold (absolute correlation): ',thresh_corr_lower_bound)
				id_corr_1 = (df_query[column_corr_1].abs()>thresh_corr_lower_bound)
				id_query_2 = id_query_2&(id_corr_1)

			df_query_pre2 = df_query.loc[id_query_2,:]

			peak_loc_query_2 = df_query_pre2.index 	# the peak loci with initially predicted TF binding and with association score above threshold
			peak_num_2 = len(peak_loc_query_2)
			print('selected peak loci with predicted TF binding and with score above threshold: %d'%(peak_num_2))

		# peak_loc_query_pre2 = pd.Index(peak_loc_query_2).union(peak_loc_query_1,sort=False)
		# peak_loc_query_pre2 = feature_query_vec
		# peak_loc_query_group2_1 = pd.Index(peak_loc_query_pre2).union(peak_loc_query_3,sort=False)
		# peak_num_group2_1 = len(peak_loc_query_group2_1)
		# print('peak_loc_query_group2_1: ',peak_num_group2_1)
		# feature_query_vec_2 = peak_loc_query_group2_1

		feature_query_vec_2 = pd.Index(feature_query_vec).union(peak_loc_query_2,sort=False)
		feature_query_num_2 = len(feature_query_vec_2)
		# print('selected pseudo positive training samples: %d'%(feature_query_num_2))

		return feature_query_vec_2

	## ====================================================
	# query column names representing the shared paired group assignment and neighbor information of peak loci
	def test_query_column_method_1(self,feature_type_vec=[],method_type_feature_link='',select_config={}):

		"""
		query column names representing the shared paired group assignment and neighbor information of peak loci
		:param feature_type_vec: (array or list) feature types of feature representations of observations (peak loci)
		:param method_type_feature_link: the method used to predict peak-TF associations initially
		:param select_config: dictionary containing the configuration parameters
		:return: (array) names of columns representing the shareg paired group assignment and neighbor information of peak loci related to peak loci with initially predicted TF binding
		"""

		flag_query1 = 1
		if flag_query1>0:
			if method_type_feature_link=='':
				method_type_feature_link = select_config['method_type_feature_link']

			column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
			column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)

			# feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]
			# column_group_neighbor_feature1 = '%s_group_neighbor'%(feature_type_query_1)
			# column_group_neighbor_feature2 = '%s_group_neighbor'%(feature_type_query_2)
			column_vec_query1 = ['%s_group_neighbor'%(feature_type_query) for feature_type_query in feature_type_vec[0:2]]

			# column_neighbor_feature1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peak query
			# column_neighbor_feature2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peak query

			# query neighbor in feature space 1 and 2, without restriction that peaks are in the same group with the selected peaks
			column_vec_query2 = ['%s_neighbor'%(feature_type_query) for feature_type_query in feature_type_vec[0:2]]

			# column_group_neighbor = '%s.pred_group_neighbor'%(method_type_feature_link)
			# column_group_neighbor_1 = '%s.pred_group_neighbor_1'%(method_type_feature_link)
			# column_neighbor_2_group = '%s.pred_neighbor_2_group'%(method_type_feature_link)
			# column_neighbor_2 = '%s.pred_neighbor_2'%(method_type_feature_link)
			# column_neighbor_1 = '%s.pred_neighbor_1'%(method_type_feature_link)

			field_query = ['pred_group_neighbor','pred_group_neighbor_1','pred_neighbor_2_group',
							'pred_neighbor_2','pred_neighbor_1']
			column_vec_query3 = ['%s.%s'%(method_type_feature_link,field_id) for field_id in field_query]

			column_vec_query_1 = [column_pred2,column_pred_2]+column_vec_query1+column_vec_query2+column_vec_query3
			
			return column_vec_query_1

	## ====================================================
	# select pseudo negative training sample
	def test_query_training_select_group2(self,data=[],motif_id='',peak_query_vec_1=[],feature_type_vec=[],save_mode=0,verbose=0,select_config={}):

		"""
		select pseudo negative training sample
		:param data: dataframe of peak annotations including initially predicted peak-TF associations
		:param feature_type_vec: (array or list) feature types of feature representations of observations (peak loci)
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1: (array) pseudo negative training samples selected from peaks with TF motif detected and without predicted TF binding
				 2: (array) pseudo negative training samples selected from peaks without TF motif detected
		"""

		flag_select=1
		verbose_internal = self.verbose_internal
		if flag_select>0:
			method_type_feature_link = select_config['method_type_feature_link']
			column_vec_query_pre1 = self.test_query_column_method_1(feature_type_vec=feature_type_vec,method_type_feature_link=method_type_feature_link,select_config=select_config)
			
			column_pred2, column_pred_2 = column_vec_query_pre1[0:2]
			column_group_neighbor_feature1, column_group_neighbor_feature2, column_neighbor_feature1, column_neighbor_feature2 =  column_vec_query_pre1[2:6]
			column_group_neighbor, column_group_neighbor_1, column_neighbor_2_group, column_neighbor_2, column_neighbor_1 = column_vec_query_pre1[6:11]

			column_vec_query_pre2 = [column_pred_2,column_neighbor_feature1,column_neighbor_feature2,column_neighbor_2_group]
			column_vec_pre2_1 = [column_group_neighbor,column_neighbor_2_group,column_neighbor_2]
			column_vec_pre2_2 = [column_group_neighbor_1,column_neighbor_2_group,column_neighbor_2]

			if verbose_internal==2:
				field_query = ['column_pred2','column_pred_2','column_group_neighbor_feature1','column_group_neighbor_feature2',
								'column_neighbor_feature1, column_neighbor_feature2',
								'column_group_neighbor, column_group_neighbor_1, column_neighbor_2_group',
								'column_neighbor_2, column_neighbor_1']
				for (field_id,query_value) in zip(field_query,column_vec_query_pre1):
					print('%s: %s\n'%(field_id,query_value))
				
			df_pre1 = data
			df_query1 = data
			peak_loc_ori = df_query1.index # the ATAC-seq peak loci

			column_motif = select_config['column_motif']
			column_pred1 = select_config['column_pred1']

			if (column_motif!='-1'):
				motif_score = df_query1[column_motif]
				id_motif = (df_query1[column_motif].abs()>0)
				df_query1_motif = df_query1.loc[id_motif,:]	# peak loci with the TF binding motif identified
				peak_loc_motif = df_query1_motif.index
				peak_num_motif = len(peak_loc_motif)
				print('peak loci with detected TF motif: ',peak_num_motif)

			id_pred1 = (df_query1[column_pred1]>0)
			peak_loc_pred1 = df_query1.loc[id_pred1,:]	# peak loci with predicted TF binding

			df_query2_2 = df_query1.loc[(~id_pred1)&id_motif,:]
			peak_loc_query_group2_2_ori = df_query2_2.index  # peak loci without predicted TF binding and with motif
			peak_num_group2_2_ori = len(peak_loc_query_group2_2_ori)
			print('peak loci without predicted TF binding and with motif: %d'%(peak_num_group2_2_ori))

			config_id_2 = select_config['config_id_2']
			column_query_pre1 = column_vec_query_pre2
			
			if verbose_internal>0:
				motif_id_query = motif_id
				print('config_id_2:%d, motif_id_query:%s'%(config_id_2,motif_id_query))
			if config_id_2%2==0:
				column_query_pre2 = column_vec_query_pre2
				print('use threshold 1 for pre-selection of pseudo negative peak loci')
			else:
				column_query_pre2 = column_vec_pre2_2
				print('use threshold 2 for pre-selection of pseudo negative peak loci') # stricter threshold
							
				query_num1 = len(column_query_pre1)
				mask_1 = (df_pre1.loc[:,column_query_pre1]>0)
				id_pred1_group = (mask_1.sum(axis=1)>0)	# the peak loci predicted with TF binding by the clustering method
				id1_2 = (~id_pred1_group)

			query_num2 = len(column_query_pre2)
			mask_2 = (df_pre1.loc[:,column_query_pre2]>0)
			id_pred2_group = (mask_2.sum(axis=1)>0)	# the peak loci predicted with TF binding by the clustering method
			id2_2 = (~id_pred2_group)

			thresh_1, thresh_2 = 5, 5
			id_neighbor_1 = (df_pre1[column_neighbor_feature1]>=thresh_1)
			id_neighbor_2 = (df_pre1[column_neighbor_feature2]>=thresh_2)

			flag_neighbor_2_2 = 1
			if flag_neighbor_2_2>0:
				id_neighbor_query1 = (id_neighbor_1&id_neighbor_2)

				id1 = (id_neighbor_1)&(df_pre1[column_neighbor_feature2]>1)
				id2 = (id_neighbor_2)&(df_pre1[column_neighbor_feature1]>1)

				id_neighbor_query2 = (id1|id2)
				id2_2 = (id2_2)&(~id_neighbor_query2)

			# id_pre2 = (id2_2&(~id_pred2))
			id_pre2 = (id2_2&(~id_pred1))
			
			id_1 = (id_pre2&id_motif)	# not predicted with TF binding but with TF motif scanned
			id_2 = (id_pre2&(~id_motif))	# not predicted with TF binding and without TF motif scanned

			# select peak with peak accessibility-TF expression correlation below threshold
			column_corr_1 = 'peak_tf_corr'
			column_pval = 'peak_tf_pval_corrected'
			thresh_corr_1, thresh_pval_1 = 0.30, 0.05
			thresh_corr_2, thresh_pval_2 = 0.1, 0.1
			thresh_corr_3, thresh_pval_2 = 0.05, 0.1

			column_corr_abs_1 = '%s_abs'%(column_corr_1)
			if not (column_corr_abs_1 in df_pre1.columns):
				df_pre1[column_corr_abs_1] = df_pre1[column_corr_1].abs()
			# df_pre1 = df_pre1.sort_values(by=[column_corr_abs_1,column_pval],ascending=[True,False]) # sort the peak-TF links by peak accessibility-TF expr correlation and p-value
				
			# id_corr_ = (df_pre1[column_corr_1].abs()<thresh_corr_2)
			id_corr_ = (df_pre1[column_corr_abs_1]<thresh_corr_2)
			id_pval = (df_pre1[column_pval]>thresh_pval_2)
							
			# id_score_query3_1 = (id_corr_&id_pval)
			id_score_query3_1 = id_corr_
								
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

			id_pre2_1 = (id_score_query3_1)&(id_pre2)&(id_motif) # the peak loci without prediction and with motif and with peak-TF correlation below threshold
			id_pre2_2 = (id_score_query3_2)&(id_pre2)&(~id_motif) # the peak loci without motif and with peak-TF correlation below threshold
							
			list_query2 = [id_pre2_1,id_pre2_2]
			list_query2_2 = []
			
			query_num = len(list_query2)
			for i2 in range(query_num):
				id_query = list_query2[i2]
				# df_pre2 = df_pre1.loc[id_query,[column_corr_1,column_pval]].copy()
				df_pre2 = df_pre1.loc[id_query,[column_corr_1,column_corr_abs_1,column_pval]].copy()

				# df_pre2[column_corr_abs_1] = df_pre2[column_corr_1].abs()
				df_pre2 = df_pre2.sort_values(by=[column_corr_abs_1,column_pval],ascending=[True,False]) # sort the peak-TF links by peak accessibility-TF expr correlation and p-value
				peak_query_pre2 = df_pre2.index
				list_query2_2.append(peak_query_pre2)

			peak_vec_2_1_ori, peak_vec_2_2_ori = list_query2_2[0:2]
			peak_num_2_1_ori = len(peak_vec_2_1_ori) # peak loci in class 2 with motif 
			peak_num_2_2_ori = len(peak_vec_2_2_ori) # peak loci in class 2 without motif
			print('candidate pseudo negative peak loci with the TF motif: ',peak_num_2_1_ori)
			print('candidate pseudo negative peak loci without the TF motif: ',peak_num_2_2_ori)

			method_type_feature_link = select_config['method_type_feature_link']
			method_type_query = method_type_feature_link
			dict1 = {'peak_group2_1':peak_vec_2_1_ori,
						'peak_group2_2':peak_vec_2_2_ori}

			column_query = 'peak_group'
			if not (column_query in self.data_pre_dict):
				self.data_pre_dict[column_query] = dict()
			self.data_pre_dict[column_query].update({method_type_query:dict1})

			peak_query_vec = peak_query_vec_1
			peak_query_num_1 = len(peak_query_vec)

			ratio_1, ratio_2 = 0.25, 1.5
			column_1, column_2 = 'ratio_1', 'ratio_2'
			if column_1 in select_config:
				ratio_1 = select_config[column_1]

			if column_2 in select_config:
				ratio_2 = select_config[column_2]

			# ratio_1, ratio_2 = 0.25, 1.75
			peak_num_2_1 = np.min([int(peak_query_num_1*ratio_1),peak_num_2_1_ori])
			peak_vec_2_1 = peak_vec_2_1_ori[0:peak_num_2_1]

			peak_num_2_2 = np.min([int(peak_query_num_1*ratio_2),peak_num_2_2_ori])
			peak_vec_2_2 = peak_vec_2_2_ori[0:peak_num_2_2]

			return peak_vec_2_1, peak_vec_2_2

	## ====================================================
	# select pseudo negative training sample with base model
	def test_query_training_select_group2_2(self,data=[],id_query=[],peak_query_vec_1=[],method_type='',flag_sample=1,flag_select=2,save_mode=0,verbose=0,select_config={}):

		"""
		select pseudo negative training sample with base model
		:param feature_mode: feature_mode = 1, use peak accessibility feature and peak-motif sequence feature
		:param method_type: the method used to predict peak-TF associations initially
		:param n_components: the number of latent components used for feature representation
		:param flag_sample: indicator of whether to random samplely the pseudo negative training samples
		:param flag_select: indicator of whether to use the motif presence in peak loci information for pseudo negative training sample selection
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1: (array) pseudo negative training samples selected
				 2: (array) pseudo negative training samples selected from peaks with TF motif detected and without predicted TF binding
				 3: (array) pseudo negative training samples selected from peaks without TF motif detected
		"""

		method_type_feature_link = method_type
		if method_type=='':
			method_type_feature_link = select_config['method_type_feature_link']

		column_motif = '%s.motif'%(method_type_feature_link)
		column_pred1 = '%s.pred'%(method_type_feature_link)

		df_query1 = data
		peak_loc_pre1 = df_query1.index

		if len(id_query)>0:
			id_pred1 = id_query
		else:
			id_pred1 = (df_query1[column_pred1]>0)

		id_pred2 = (~id_pred1)
		peak_vec_2_ori = np.asarray(peak_loc_pre1[id_pred2])

		peak_query_vec = peak_query_vec_1
		peak_query_num_1 = len(peak_query_vec)

		if column_motif in df_query1.columns:
			try:
				id_motif = (df_query1[column_motif].abs()>0)
			except Exception as error:
				print('error! ',error)
				id_motif = (df_query1[column_motif].isin(['True',True,1,'1']))

			id_pred2_1 = (~id_pred1)&(id_motif)
			id_pred2_2 = (~id_pred1)&(~id_motif)

			peak_vec_2_1_ori = np.asarray(peak_loc_pre1[id_pred2_1])
			peak_vec_2_2_ori = np.asarray(peak_loc_pre1[id_pred2_2])

		peak_vec_2 = []
		if flag_sample>0:
			ratio_1, ratio_2 = select_config['ratio_1'], select_config['ratio_2']
			if flag_select in [2]:
				np.random.shuffle(peak_vec_2_ori)
				peak_query_num_2 = int(peak_query_num_1*ratio_2)
				peak_vec_2 = peak_vec_2_ori[0:peak_query_num_2]

			elif flag_select in [3]:
				np.random.shuffle(peak_vec_2_1_ori)
				np.random.shuffle(peak_vec_2_2_ori)
				peak_query_num2_1 = int(peak_query_num_1*ratio_1)
				peak_query_num2_2 = int(peak_query_num_1*ratio_2)
				peak_vec_2_1 = peak_vec_2_1_ori[0:peak_query_num2_1]
				peak_vec_2_2 = peak_vec_2_2_ori[0:peak_query_num2_2]
		else:
			if flag_select in [2]:
				peak_vec_2 = peak_vec_2_ori
			elif flag_select in [3]:
				peak_vec_2_1 = peak_vec_2_1_ori
				peak_vec_2_2 = peak_vec_2_2_ori

		return peak_vec_2, peak_vec_2_1, peak_vec_2_2

	## ====================================================
	# parameter configuration for prediction model training
	def test_optimize_configure_1(self,model_type_id,Lasso_alpha=0.01,Ridge_alpha=1.0,l1_ratio=0.01,ElasticNet_alpha=1.0,select_config={}):

		"""
		parameter configuration for prediction model training
		:param model_type_id: the prediction model type
		:param Lasso_alpha: coefficient of the L1 norm term in the Lasso model (using sklearn)
		:param Ridge_alpha: coefficient of the L2 norm term in the Ridge regression model (using sklearn)
		:param l1_ratio: parameter l1_ratio (related to coefficient of the L1 norm term) in the ElasticNet model in sklearn
		:param ElasticNet_alpha: parameter alpha the ElasticNet model in skearn
		:param select_config: dictionary storing configuration parameters
		:return: dictionary storing configuration parameters for prediction model training
		"""

		flag_select_config_1 = 1
		model_type_id1 = model_type_id
		if flag_select_config_1>0:
			flag_positive_coef = False
			warm_start_type = False
			# fit_intercept = False
			fit_intercept = True
			if 'fit_intercept' in select_config:
				fit_intercept = select_config['fit_intercept']
			if 'warm_start_type' in select_config:
				warm_start_type = select_config['warm_start_type']
			if 'flag_positive_coef' in select_config:
				flag_positive_coef = select_config['flag_positive_coef']
			
			select_config1 = select_config
			select_config1.update({'flag_positive_coef':flag_positive_coef,
									'warm_start_type_Lasso':warm_start_type,
									'fit_intercept':fit_intercept})

			if model_type_id1 in ['Lasso']:
				# Lasso_alpha = 0.001
				if 'Lasso_alpha' in select_config:
					Lasso_alpha = select_config['Lasso_alpha']
				select_config1.update({'Lasso_alpha':Lasso_alpha})
				filename_annot2 = '%s'%(Lasso_alpha)
			elif model_type_id1 in ['ElasticNet']:
				# l1_ratio = 0.01
				if 'l1_ratio_ElasticNet' in select_config:
					l1_ratio = select_config['l1_ratio_ElasticNet']
				if 'ElasticNet_alpha' in select_config:
					ElasticNet_alpha = select_config['ElasticNet_alpha']
				select_config1.update({'ElasticNet_alpha':ElasticNet_alpha,
										'l1_ratio_ElasticNet':l1_ratio})
				filename_annot2 = '%s.%s'%(ElasticNet_alpha,l1_ratio)
			elif model_type_id1 in ['Ridge']:
				# Ridge_alpha = 0.01
				if 'Ridge_alpha' in select_config:
					Ridge_alpha = select_config['Ridge_alpha']
				select_config1.update({'Ridge_alpha':Ridge_alpha})
				filename_annot2 = '%s'%(Ridge_alpha)
			else:
				filename_annot2 = '1'

			run_id = select_config['run_id']
			filename_annot1 = '%s.%d.%s.%d'%(model_type_id1,int(fit_intercept),filename_annot2,run_id)
			select_config1.update({'filename_annot1':filename_annot1})

		return select_config1

	## ====================================================
	# prepare training, validation and test sample indices
	def test_train_idvec_pre1(self,sample_id,num_fold=10,train_valid_mode=1,load=0,input_filename='',save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):
		
		"""
		prepare training and test sample indices
		:param sample_id: (array) sample indices
		:param num_fold: the number of folds for cross-validation
		:param train_valid_mode: indicator of whether to use validation data
		:param load: indicator of whether to load the training, validation, and test sample indices from the current file
		:param input_filename: path of the file containing the prepared training, validation, and test sample indices
		:param save_mode: indicator of whether to save data
		:param output_file_path: the director to save data
		:param output_filename: the filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dataframe containing the test sample indices prepared for each fold
				 2. dictionary containing the training, validation, and test sample indices prepared for each fold
		"""

		input_file_path = output_file_path
		flag_query=1
		if load>0:
			input_filename = '%s/train_id_cv.pre1.txt'%(input_file_path)
			if os.path.exists(input_filename)==True:
				data1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				flag_query = 0
				sample_idvec = dict()

		if flag_query>0:
			from sklearn.model_selection import KFold, train_test_split
			sample_num = len(sample_id)
			sample_idvec = dict()
			# num_fold = 10
			np.random.seed(0)
			split1 = KFold(n_splits=num_fold,shuffle=True,random_state=0)
			id_query_pre1 = np.arange(sample_num)
			id_vec1 = split1.split(id_query_pre1)

			# label_vec = np.zeros(sampel_num,dtype=np.int8)
			data1 = pd.DataFrame(index=sample_id,columns=['fold','type_id','id'])
			# train_valid_mode_1, train_valid_mode_2 = 1, 1
			cnt1 = 0
			for train_id1, test_id1 in id_vec1:
				train_num, test_num = len(train_id1), len(test_id1)
				id_train, id_test = train_id1, test_id1
				sample_id_train = sample_id[id_train]
				sample_id_test = sample_id[id_test]

				sample_id_train_ori = sample_id_train
				if train_valid_mode>0:
					sample_id_train1, sample_id_valid = train_test_split(sample_id_train, test_size=0.1, random_state=0)
					sample_id_train = sample_id_train1
				else:
					sample_id_valid = []

				sample_idvec[cnt1] = [sample_id_train, sample_id_valid, sample_id_test]

				# t_vec_1 = [sample_id_test,sample_id_train,sample_id_valid]
				print('sample_id_train, sample_id_valid, sample_id_test ',len(sample_id_train),len(sample_id_valid),len(sample_id_test))
				data1.loc[sample_id_test,['fold','type_id']] = [cnt1,0]
				cnt1 += 1

			if save_mode>0:
				if output_filename=='':
					output_filename = '%s/train_id_cv.pre1.txt'%(output_file_path)
				data1.to_csv(output_filename,sep='\t',float_format='%d')

		return data1, sample_idvec

	## ====================================================
	# model training for peak-TF association prediction
	def test_query_association_unit_pre1(self,feature_query_vec=[],dict_feature=[],feature_type_vec=['motif','peak'],model_type_vec=[],model_type_id='LogisticRegression',sample_idvec_train=[],
											type_id_model=1,num_class=2,num_fold=-1,parallel_mode=0,
											save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		model training for peak-TF association prediction
		:param feature_query_vec: (array) TFs for which to predict binding in the genome-wide peak loci
		:param dict_feature: dictionary containing the feature matrix of predictor variables and the response variable values
		:param feature_type_vec: (array or list) the two types of features for which to estimate the associations
		:param model_type_vec: (list) different types of prediction models
		:param model_type_id: (str) method used by the prediction model
		:param sample_idvec_train: (list) selected training, validation, and test samples
		:param type_id_model: prediction model type: 0,regression model; 1,classification model
		:param num_class: the number of classes
		:param num_fold: the number of folds used in cross validation
		:param parallel_mode: indicator of whether to perform model training in parallel
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1. (dataframe) predicted peak-TF association label for each peak locus and for the TF
				 2. (dataframe) predicted TF binding probability in each peak locus for the TF
		"""

		feature_type_query1, feature_type_query2 = feature_type_vec[0:2]
		df_feature_1 = dict_feature[feature_type_query1]
		df_feature_2 = dict_feature[feature_type_query2]
		verbose_internal = self.verbose_internal
		if verbose_internal==2:
			# print('df_feature_1, df_feature_2: ',df_feature_1.shape,df_feature_2.shape)
			print('feature matrix of predictor variables, dataframe of size ',df_feature_1.shape)
			print('response variable values, dataframe of size ',df_feature_2.shape)

		sample_idvec_query = sample_idvec_train
		if len(sample_idvec_train)==0:
			if num_fold>0:
				load_1 = 0
				input_filename_query1 = ''
				field_id1, field_id2 = 'sample_idvec_load','filename_sample_idvec'
				if field_id1 in select_config:
					load_1 = select_config[field_id1]
					if (load_1>0) and (field_id2 in select_config):
						input_filename_query1 = select_config[field_id2]
				sample_id = df_feature_1.index
				train_valid_mode_2 = 0
				column_1 = 'train_valid_mode_2'
				if column_1 in select_config:
					train_valid_mode_2 = select_config[column_1]
				data_vec_query, sample_idvec_query = self.test_train_idvec_pre1(sample_id,num_fold=num_fold,
																				train_valid_mode=train_valid_mode_2,
																				load=load_1,
																				input_filename=input_filename_query1,
																				save_mode=1,output_file_path=output_file_path,
																				select_config=select_config)
				sample_idvec_train = sample_idvec_query

		sample_id_train, sample_id_valid, sample_id_test = sample_idvec_query[0:3]
		x_train1 = df_feature_2.loc[sample_id_train,:]
		x_train1_ori = x_train1.copy()

		# sample_id = x_train1.index
		sample_id = df_feature_2.index
		feature_query_1 = feature_query_vec
		feature_query_2 = df_feature_2.columns

		y_mtx = df_feature_1.loc[sample_id_train,feature_query_1]
		if verbose_internal>0:
			print('input feature, dataframe of size ',x_train1.shape)
			print('class label, dataframe of size ',y_mtx.shape)

		motif_data = []
		if len(model_type_vec)==0:
			model_type_vec = ['LR','XGBClassifier','XGBR','Lasso',-1,'RF','ElasticNet','LogisticRegression']

		model_type_id1 = model_type_id  # model_type_name
		# model_type = model_type_id1
		# print('model_type: ',model_type_id1,model_type)
		print('model_type: ',model_type_id1)

		file_path1 = self.save_path_1
		run_id = select_config['run_id']
		select_config1 = dict()
		# parameter configuration for model training
		select_config1 = self.test_optimize_configure_1(model_type_id=model_type_id1,select_config=select_config)
		# print('parameter configuration: ',select_config1)

		max_depth, n_estimators = 7, 100
		select_config_comp = {'max_depth':max_depth,'n_estimators':n_estimators} # the parameters of the XGBClassifier
		select_config1.update({'select_config_comp':select_config_comp})

		column_1 = 'multi_class_logisticregression'
		multi_class_query = 'auto'
		if column_1 in select_config:
			multi_class_query = select_config[column_1]
			select_config1.update({column_1:multi_class_query}) # copy the field
		print('multi_class_logisticregression: ',multi_class_query)

		train_valid_mode_1, train_valid_mode_2 = 1, 0 # train_valid_mode_1:1, train on the combined data; train_valid_mode_2:0,only use train and test data; 1, use train,valid,and test data
		list_param = [train_valid_mode_1,train_valid_mode_2]
		
		field_query = ['train_valid_mode_1','train_valid_mode_2']
		query_num1 = len(list_param)
		from .utility_1 import test_query_default_parameter_1
		select_config, list_param = test_query_default_parameter_1(field_query=field_query,default_parameter=list_param,overwrite=False,select_config=select_config)
		train_valid_mode_1,train_valid_mode_2 = list_param[0:2]
		
		print('train_valid_mode_1: %d, train_valid_mode_2: %d'%(train_valid_mode_1,train_valid_mode_2))
		if train_valid_mode_1>0:
			print('train on the combined data')

		select_config.update({'num_fold':num_fold,'select_config1':select_config1,
								'sample_idvec_train':sample_idvec_query})

		if train_valid_mode_1>0:
			response_variable_name = feature_query_1[0]
			response_query1 = response_variable_name
			y_train1 = y_mtx.loc[:,response_query1]
			x_train1_ori = x_train1
			if verbose_internal>0:
				print('class labels: ',np.unique(y_train1))
			
			# pre_data_dict = dict()
			model_pre = train_pre1._Base2_train1(select_config=select_config)

			sample_weight = []
			dict_query_1 = dict()
			save_mode_1 = 1
			save_model_train = 1
			model_path_1 = select_config['model_path_1']
			dict_query_1, df_score_1 = model_pre.test_optimize_pre1(model_pre=[],x_train=x_train1,y_train=y_train1,
																		response_variable_name=response_query1,
																		sample_weight=sample_weight,
																		dict_query=dict_query_1,
																		df_coef_query=[],
																		df_pred_query=[],
																		model_type_vec=model_type_vec,
																		model_type_idvec=[model_type_id1],
																		type_id_model=type_id_model,
																		num_class=num_class,
																		save_mode=save_mode_1,
																		save_model_train=save_model_train,
																		model_path_1=model_path_1,
																		output_file_path=output_file_path,
																		filename_prefix_save=filename_prefix_save,
																		filename_save_annot=filename_save_annot,
																		output_filename=output_filename,
																		verbose=0,
																		select_config=select_config)
			
			dict1 = dict_query_1[model_type_id1]
			y_pred = dict1['pred']

			df_pred_query = y_pred
			if verbose_internal>0:
				# print('y_pred: ',y_pred.shape)
				print('prediction on training data, dataframe of size ',y_pred.shape)
				print('data preview: ')
				print(y_pred[0:2])

			# dict_query_1[model_type_id1].update({'model_combine':model_2,'df_score_2':df_score_2})	# prediction performance on the combined data
			model_train = dict1['model_combine']
			# df_score_2 = dict1['df_score_2']
			x_test = df_feature_2.loc[sample_id_test,feature_query_2]
			y_test = df_feature_1.loc[sample_id_test,feature_query_1]

			if verbose_internal>0:
				print('predict on test data')
				print('x_test, dataframe of size ',x_test.shape)
				print('y_test, dataframe of size ',y_test.shape)

			y_test_pred = model_train.predict(x_test)
			y_test_proba = model_train.predict_proba(x_test)
			
			df_pred_2 = pd.DataFrame(index=sample_id_test,columns=[feature_query_1],data=np.asarray(y_test_pred)[:,np.newaxis])
			df_proba_2 = pd.DataFrame(index=sample_id_test,columns=np.arange(num_class),data=np.asarray(y_test_proba))

			return df_pred_2, df_proba_2

	## ====================================================
	# model training for peak-TF association prediction
	def test_query_compare_binding_train_unit1(self,data=[],peak_query_vec=[],peak_vec_1=[],motif_id_query='',dict_feature=[],feature_type_vec=[],
													sample_idvec_query=[],motif_data=[],flag_scale=1,input_file_path='',
													save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
		
		"""
		model training for peak-TF association prediction
		:param data: (dataframe) peak annotations including initially predicated peak-TF associations for the given TF
		:param peak_query_vec: (array) ATAC-seq peak loci for which we perform TF binding prediction for the given TF
		:param peak_vec_1: (array) selected pseudo positive training samples (peak loci)
		:param motif_id_query: (str) name of the TF for which we perform binding prediction in peak loci
		:param dict_feature: dictionary containing feature matrices of predictor variables and response variable values
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param sample_idvec_query: (list) selected training, validation, and test samples
		:param motif_data: (dataframe) motif presence in peak loci by motif scanning (binary) (row:peak, column:TF (associated with motif))
		:param flag_scale: indicator of whether to scale the feature matrix
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		return: (dataframe) peak annotations including predicted TF binding label (binary) and TF binding probability for the given TF
		"""

		flag_query1 = 1
		if flag_query1>0:
			sample_id_valid = []
			peak_loc_ori = peak_query_vec
			sample_id_test = peak_loc_ori

			df_pre1 = data
			df_pre1[motif_id_query] = 0
			peak_query_vec_pre1 = peak_vec_1.copy()
			df_pre1.loc[peak_vec_1,motif_id_query] = 1 # the selected peak loci with predicted TF binding

			peak_num1 = len(peak_vec_1)
			print('selected pseudo positive peak loci: %d'%(peak_num1))

			# column_signal = 'signal'
			method_type_feature_link = select_config['method_type_feature_link']
			column_motif = select_config['column_motif']
			verbose_internal = self.verbose_internal
			if verbose_internal==2:
				field_query_1 = [motif_id_query]
				if column_motif in df_pre1.columns:
					field_query_1 = [column_motif,motif_id_query]

				print('data preview: ')
				print(df_pre1.loc[peak_vec_1,field_query_1])

			# feature_type_vec_query = ['latent_peak_motif','latent_peak_tf']
			feature_type_vec_query = feature_type_vec
			feature_type_query_1, feature_type_query_2 = feature_type_vec_query[0:2]

			model_type_id1 = select_config['model_type_id1']
			num_class = 2
			select_config.update({'num_class':num_class})
			num_fold = -1
			feature_query_vec_1 = [motif_id_query]
			feature_type_vec_pre1 = ['motif','peak']
			column_vec_pre1 =['motif_id','peak_id']

			type_query = 0
			# type_query = 1
			column_1 = 'type_combine'
			if column_1 in select_config:
				type_query = select_config[column_1]
			else:
				select_config.update({column_1:type_query})

			if type_query==0:
				feature_type_vec_2 = select_config['feature_type_vec_2']
				feature_type_1, feature_type_2 = feature_type_vec_2[0:2]
				feature_type_combine = 'latent_%s_%s_combine'%(feature_type_1,feature_type_2)
				feature_type_query_vec_2 = [feature_type_combine]
			else:
				feature_type_query_vec_2 = np.asarray(list(dict_feature.keys()))
			feature_type_num1 = len(feature_type_query_vec_2)
				
			file_path_query1 = select_config['file_path_save_link']
			dict_feature_query = dict_feature
			for i2 in range(feature_type_num1):
				feature_type_query = feature_type_query_vec_2[i2]
				latent_mtx_query_ori = dict_feature[feature_type_query]
				if flag_scale>0:
					print('perform feature scaling')
					scale_type = 2 # compute z-score
					with_mean = True
					with_std = True
					# perform feature scaling
					latent_mtx_query = utility_1.test_motif_peak_estimate_score_scale_1(score=latent_mtx_query_ori,
																							feature_query_vec=[],
																							with_mean=with_mean,
																							with_std=with_std,
																							scale_type_id=scale_type,
																							select_config=select_config)
				else:
					latent_mtx_query = latent_mtx_query_ori

				df_feature_2 = latent_mtx_query.loc[peak_loc_ori,:]
				df_feature_1 = df_pre1.loc[peak_loc_ori,feature_query_vec_1]

				feature_type_pre1, feature_type_pre2 = feature_type_vec_pre1[0:2]
				dict_feature_query = {feature_type_pre1:df_feature_1,feature_type_pre2:df_feature_2}
				if verbose_internal==2:
					print('response variable: ',feature_query_vec_1,feature_type_pre1)
					print('data preview: ')
					print(df_feature_1[0:2])
					
					print('input feature, dataframe of size ',df_feature_2.shape)
					print('feature type: %s'%(feature_type_query))
					print('data preview: ')
					print(df_feature_2[0:2])

				sample_idvec_train = sample_idvec_query
				run_id1 = 1
				select_config.update({'run_id':run_id1})
				feature_type_id1 = 0
				flag_model_explain = 1
				select_config.update({'feature_type_id':feature_type_id1,'flag_model_explain':flag_model_explain})

				column_1 = 'filename_save_annot_local'
				column_2 = 'data_path_save'

				filename_save_annot_2 = filename_save_annot
				data_path_save = file_path_query1
				filename_save_annot_local = '%s.%s_%s.%s'%(filename_save_annot_2,feature_type_query,model_type_id1, motif_id_query)
				select_config.update({column_1:filename_save_annot_local})

				if not (column_2 in select_config):
					select_config.update({column_2:data_path_save})

				# data_path_save = file_path_query1
				# select_config.update({'filename_save_annot_local':filename_save_annot_local,
				# 						'data_path_save':file_path_query1})

				start = time.time()
				# classification model training
				model_type_id_train = model_type_id1
				select_config.update({'model_type_id_train':model_type_id_train})
				type_id_model = 1
				if 'type_id_model' in select_config:
					type_id_model = select_config['type_id_model']
				else:
					select_config.update({'type_id_model':type_id_model})

				# model training for peak-TF association prediction
				df_pred_2, df_proba_2 = self.test_query_association_unit_pre1(feature_query_vec=feature_query_vec_1,
																				dict_feature=dict_feature_query,
																				feature_type_vec=feature_type_vec_pre1,
																				model_type_vec=[],
																				model_type_id=model_type_id1,
																				sample_idvec_train=sample_idvec_train,
																				type_id_model=type_id_model,
																				num_class=num_class,
																				num_fold=num_fold,
																				parallel_mode=0,
																				save_mode=1,
																				output_file_path=output_file_path,
																				output_filename='',
																				filename_prefix_save=filename_prefix_save,
																				filename_save_annot=filename_save_annot_2,
																				verbose=verbose,select_config=select_config)
					
				stop = time.time()
				print('model training used %.2fs'%(stop-start),feature_type_query)

				if verbose_internal>0:
					print('predicted class label, dataframe of size ',df_pred_2.shape,feature_type_query)
					print('data preview: ')
					print(df_pred_2[0:2])

					print('predicted probability, dataframe of size ',df_proba_2.shape,feature_type_query)
					print('data preview: ')
					print(df_proba_2[0:2])

				peak_loc_query1 = df_pred_2.index
				peak_loc_query2 = df_proba_2.index
				assert list(peak_loc_query1)==list(peak_loc_ori)
				assert list(peak_loc_query2)==list(peak_loc_ori)
					
				column_1 = '%s_%s_pred'%(feature_type_query,model_type_id1)
				column_vec_2 = ['%s_%s_proba_%d'%(feature_type_query,model_type_id1,i2) for i2 in range(1,num_class)]
				df_pre1.loc[peak_loc_query1,column_1] = np.asarray(df_pred_2[motif_id_query])
				column_2 = column_vec_2[0]
				df_proba_2 = df_proba_2.round(6)
				df_pre1.loc[peak_loc_query2,column_2] = np.asarray(df_proba_2)[:,1]
				select_config.update({'column_pred_%s'%(feature_type_query):column_1,'colum_proba_%s'%(feature_type_query):column_vec_2})

			column_signal = 'signal'
			if column_signal in df_pre1.columns:
				column_vec_sort = [column_signal,column_motif]
			else:
				column_vec_sort = [column_motif]
					
			df_pre1 = df_pre1.sort_values(by=column_vec_sort,ascending=False)
			if (save_mode>0) and (output_filename!=''):
				column_vec = df_pre1.columns
				column_vec_query2 = ['peak_id',motif_id_query]
				t_columns = pd.Index(column_vec).difference(column_vec_query2,sort=False)
				df_pre1 = df_pre1.loc[:,t_columns]
				df_pre1.to_csv(output_filename,sep='\t')

				# save the selected pseudo-labeled training sample
				# column_1 = 'class'
				# id1 = (df_pre1[column_1].abs()>0)
				# df_query = df_pre1.loc[id1,:]
				# df_query = df_query.sort_values(by=[column_1],ascending=False)
				# b = output_filename.find('.txt')
				# filename_prefix = output_filename[0:b]
				# output_filename_2 = '%s.2.txt'%(filename_prefix)
				# df_query.to_csv(output_filename_2,sep='\t')

			return df_pre1
	
	## ====================================================
	# load motif scanning data; load ATAC-seq and RNA-seq data of the metacells
	def test_query_load_pre1(self,method_type_vec=[],flag_motif_data_load_1=1,flag_load_1=1,flag_format=False,flag_scale=0,input_file_path='',save_mode=1,verbose=0,select_config={}):

		"""
		load motif scanning data; load ATAC-seq and RNA-seq data of the metacells
		:param method_type_vec: the methods used to predict peak-TF associations initially
		:param flag_motif_data_load: indicator of whether to query motif scanning data
		:param flag_load_1: indicator of whether to query peak accessibility and gene expression data
		:param flag_format: indicator of whether to use uppercase variable names in the RNA-seq data of the metacells
		:param flag_scale: indicator of whether to scale the feature matrix
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated configuration parameters
		"""

		# flag_motif_data_load_1 = 1
		# load motif data
		method_type_feature_link = select_config['method_type_feature_link']
		if flag_motif_data_load_1>0:
			print('load motif data')
			method_type_vec_query = method_type_vec
			if len(method_type_vec_query)==0:
				method_type_vec_query = [method_type_feature_link]

			# load motif scanning data
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																			select_config=select_config)

			self.dict_motif_data = dict_motif_data

		# flag_load_1 = 1
		# load the ATAC-seq data and RNA-seq data of the metacells
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			# print('load ATAC-seq and RNA-seq count matrices of the metacells')
			start = time.time()
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_load_data_pre1(flag_format=flag_format,
																					select_config=select_config)

			sample_id = peak_read.index
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			if len(meta_scaled_exprs)>0:
				meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
				rna_exprs = meta_scaled_exprs	# scaled RNA-seq data
			else:
				rna_exprs = meta_exprs_2	# unscaled RNA-seq data

			print('ATAC-seq count matrix: ',peak_read.shape)
			print('data preview:\n',peak_read[0:2])
			print('RNA-seq count matrix: ',rna_exprs.shape)
			print('data preview:\n',rna_exprs[0:2])

			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs

			stop = time.time()
			print('load peak accessiblity and gene expression data used %.2fs'%(stop-start))
			
		return select_config

	def run_pre1(self,chromosome='1',run_id=1,species='human',cell=0,generate=1,chromvec=[],testchromvec=[],metacell_num=500,peak_distance_thresh=100,
						highly_variable=1,upstream=100,downstream=100,type_id_query=1,thresh_fdr_peak_tf=0.2,path_id=2,save=1,type_group=0,type_group_2=0,type_group_load_mode=0,
						method_type_group='phenograph.20',thresh_size_group=50,thresh_score_group_1=0.15,method_type_feature_link='joint_score_pre1.thresh3',neighbor_num=30,model_type_id='XGBClassifier',typeid2=0,type_combine=0,folder_id=1,
						config_id_2=1,config_group_annot=1,ratio_1=0.25,ratio_2=2,flag_group=-1,train_id1=1,flag_scale_1=1,beta_mode=0,motif_id_1='',query_id1=-1,query_id2=-1,query_id_1=-1,query_id_2=-1,train_mode=0,config_id_load=-1):
		
		chromosome = str(chromosome)
		run_id = int(run_id)
		species_id = str(species)
		cell_type_id = int(cell)
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
		type_combine = int(type_combine)
		beta_mode = int(beta_mode)
		motif_id_1 = str(motif_id_1)

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
			
			data_file_type = celltype_vec[cell_type_id]
			print('data_file_type: %s'%(data_file_type))
			run_id = 1
			type_id_feature = 0
			metacell_num = 500

			root_path_1 = '.'
			root_path_2 = '.'
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
			
			verbose = 1
			flag_score_1 = 0
			flag_score_2 = 0
			flag_compare_1 = 1

def run(chromosome,run_id,species,cell,generate,chromvec,testchromvec,metacell_num,peak_distance_thresh,
			highly_variable,upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
			method_type_group,thresh_size_group,thresh_score_group_1,method_type_feature_link,neighbor_num,model_type_id,typeid2,folder_id,
			config_id_2,config_group_annot,ratio_1,ratio_2,flag_group,train_id1,flag_scale_1,beta_mode,motif_id_1,query_id1,query_id2,query_id_1,query_id_2,train_mode,config_id_load):
	
	file_path_1 = '.'
	test_estimator1 = _Base2_2(file_path=file_path_1)

	test_estimator1.run_pre1(chromosome,run_id,species,cell,generate,chromvec,testchromvec,metacell_num,peak_distance_thresh,
								highly_variable,upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
								method_type_group,thresh_size_group,thresh_score_group_1,method_type_feature_link,neighbor_num,model_type_id,typeid2,type_combine,folder_id,
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
	parser.add_option("--type_combine",default="0",help="feature type used for model training")
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
		opts.type_combine,
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







