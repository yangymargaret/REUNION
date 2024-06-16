#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import scanpy.external as sce

from copy import deepcopy
import warnings
import sys
from tqdm.notebook import tqdm

import os
import os.path
from optparse import OptionParser

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, LabelEncoder, KBinsDiscretizer
from sklearn.pipeline import make_pipeline

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.stats import gaussian_kde, zscore, poisson, multinomial, norm, rankdata
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, issparse
import pingouin as pg
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

# import utility_1
from . import utility_1
from .utility_1 import test_file_merge_1, test_query_index
import h5py
import pickle

# import test_unify_compute_2
from . import test_unify_compute_2
from .test_unify_compute_2 import _Base2_correlation3
# import test_reunion_correlation_1
from . import test_reunion_correlation_1
from .test_reunion_correlation_1 import _Base2_correlation

sc.settings.verbosity = 3 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

class _Base2_correlation5(_Base2_correlation3):
	"""Base class for peak-TF-gene link estimation
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

		_Base2_correlation3.__init__(self,file_path=file_path,
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

	## ====================================================
	# the in silico ChIP-seq library method
	# compute TF binding score using the in silico ChIP-seq library method
	def test_peak_tf_score_normalization_1(self,peak_query_vec=[],motif_query_vec=[],motif_data=[],motif_data_score=[],
												df_peak_tf_expr_corr_=[],input_filename='',peak_read=[],rna_exprs=[],peak_read_celltype=[],
												df_peak_annot=[],correlation_type='spearmanr',overwrite=False,beta_mode=0,
												save_mode=1,output_file_path='',output_filename='',filename_annot='',verbose=0,select_config={}):

		"""
		compute TF binding score using the in silico ChIP-seq library method
		:param peak_query_vec: (array) the ATAC-seq peak loci for which to estimate peak-TF links, default: the indices of motif_data
		:param motif_query_vec: (array) TF names, default: columns of the peak accessibility-TF expression correlation matrix
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param motif_data_score: (dataframe) motif scores from motif scanning results (row:ATAC-seq peak locus, column:TF)
		:param df_peak_tf_expr_corr_: (dataframe) the peak accessibility-TF expression correlation matrix (row: ATAC-seq peak locus, column:TF)
		:param input_filename: path of the file which saved the pre-computed peak accessibility-TF expression correlation matrix
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param peak_read_celltype: peak accessibility matrix for cell types (if we want to use maximal peak accessibility across cell types, by default we use the maximal accessiblity of a peak locus across the metacells)
								   if peak_read_celltype is provided as input, both maximal chromatin accessibility across metacells and across the cell types will be computed
		:param df_peak_annot: (dataframe) peak loci attributes including maximal peak accessibility across metacells
		:param correlation_type: the type of peak accessibility-TF expression correlation
		:param overwrite: indicator of whether to overwrite the current file of computed in silico ChIP-seq TF binding scores
		:param beta_mode: indicator of whether to perform estimation for all the TFs with expressions or for a subset of TFs
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: the filename to save the computed in silico ChIP-seq TF binding score
		:param filename_annot: annotation used in potential filename to save data
		return: (dataframe) the TF binding scores estimated by the in silico ChIP-seq library method and other associated scores of the peak-TF links
		"""

		input_filename_pre1 = output_filename
		if (os.path.exists(input_filename_pre1)==True) and (overwrite==False):
			print('the file exists: %s'%(input_filename_pre1))
			# print('overwrite: ',overwrite)
			df_pre1 = pd.read_csv(input_filename_pre1,index_col=0,sep='\t') # retrieve the TF binding score estimated and saved
			return df_pre1

		sample_id = peak_read.index
		if len(motif_data)==0:
			motif_data = (motif_data_score.abs()>0)

		verbose_internal = self.verbose_internal
		if verbose_internal>0:
			print('motif scanning data (binary), dataframe of size ',motif_data.shape)
			print('motif scores, dataframe of size ',motif_data_score.shape)

		df_pre1 = []
		# compute peak accessibility-TF expression correlation
		if len(df_peak_tf_expr_corr_)==0:
			input_filename_expr_corr = input_filename
			if input_filename_expr_corr!='':
				if os.path.exists(input_filename_expr_corr)==True:
					b = input_filename_expr_corr.find('.txt')
					if b>0:
						df_peak_tf_expr_corr_ = pd.read_csv(input_filename_expr_corr,index_col=0,sep='\t')
					else:
						adata = sc.read(input_filename_expr_corr)
						df_peak_tf_expr_corr_ = pd.DataFrame(index=adata.obs_names,columns=adata.var_names,data=adata.X.toarray(),dtype=np.float32)
					print('load peak accessibility-TF expr correlation from %s'%(input_filename_expr_corr))
				else:
					print('the file does not exist:%s'%(input_filename_expr_corr))
					return
			else:
				print('peak accessibility-TF expr correlation not provided\n perform peak accessibility-TF expr correlation estimation')
				filename_prefix = 'test_peak_tf_correlation'
				column_id1 = 'peak_tf_corr'
				dict_peak_tf_corr_ = self.test_peak_tf_correlation_query_1(motif_data=motif_data,
																			peak_query_vec=[],
																			motif_query_vec=[],
																			peak_read=peak_read,
																			rna_exprs=rna_exprs,
																			correlation_type=correlation_type,
																			save_mode=save_mode,
																			output_file_path=output_file_path,
																			filename_prefix=filename_prefix,
																			select_config=select_config)
				df_peak_tf_expr_corr_ = dict_peak_tf_corr_[column_id1]
		# print('df_peak_tf_expr_corr_ ',df_peak_tf_expr_corr_.shape)
		# print(df_peak_tf_expr_corr_[0:2])
		print('peak accessibility-TF expr correlation, dataframe of size ',df_peak_tf_expr_corr_.shape)
		print('data preview:\n', df_peak_tf_expr_corr_[0:2])

		peak_loc_ori = motif_data.index
		peak_loc_num = len(peak_loc_ori)
		if len(motif_query_vec)==0:
			motif_query_vec = df_peak_tf_expr_corr_.columns

		if len(peak_query_vec)==0:
			peak_query_vec = motif_data.index

		motif_query_num = len(motif_query_vec)
		peak_query_num = len(peak_query_vec)
		print('peak number:%d, TF number:%d'%(peak_query_num,motif_query_num))
		
		motif_data = motif_data.loc[peak_query_vec,motif_query_vec]
		motif_data_score = motif_data_score.loc[peak_query_vec,motif_query_vec]
		df_peak_tf_expr_corr_1 = df_peak_tf_expr_corr_.loc[peak_query_vec,motif_query_vec]
		df_peak_tf_expr_corr_1 = df_peak_tf_expr_corr_1.fillna(0)
		peak_id, motif_id = peak_query_vec, motif_query_vec

		mask = (motif_data>0)	# shape: (peak_num,motif_num)
		mask_1 = mask
		df_peak_tf_expr_corr_1[~mask_1] = 0 # only include peak accessibility-TF expression correlations for peak loci with TF binding motifs detected

		# query peak accessibility by cell type
		flag_query_by_celltype = 0
		if len(peak_read_celltype)>0:
			flag_query_by_celltype = 1

		min_peak_number = 1
		field_query_1 = ['correlation_score','max_accessibility_score','motif_score','motif_score_normalize',
							'score_1','score_pred1']
		field_query = field_query_1
		list_query1 = []
		column_id_query = 'max_accessibility_score'
		load_mode = 0
		if (len(df_peak_annot)>0) and (column_id_query in df_peak_annot.columns):
			load_mode = 1  # query maximum peak accessibility from peak annotation

		motif_query_num_ori = motif_query_num
		for i1 in range(motif_query_num):
			motif_id1 = motif_query_vec[i1]
			id1 = motif_data.loc[:,motif_id1]>0
			peak_id1 = peak_query_vec[id1]
			motif_score = motif_data_score.loc[peak_id1,motif_id1]
			motif_score_1 = motif_score/np.max(motif_score) # normalize motif_score per motif;
			correlation_score = df_peak_tf_expr_corr_1.loc[peak_id1,motif_id1]
			if load_mode==0:
				max_accessibility_score = peak_read.loc[:,peak_id1].max(axis=0)
			else:
				max_accessibility_score = df_peak_annot.loc[peak_id,column_id_query]
			
			score_1 = minmax_scale(max_accessibility_score*motif_score,[0,1])
			# score_1 = minmax_scale(max_accessibility_score*motif_score_1,[0,1])
			score_pred1 = correlation_score*score_1
			list1 = [correlation_score,max_accessibility_score,motif_score,motif_score_1,score_1,score_pred1]

			score_2, score_pred2 = [], []
			if flag_query_by_celltype>0:
				max_accessibility_score_celltype = peak_read_celltype.loc[:,peak_id1].max(axis=0)
				score_2 = minmax_scale(max_accessibility_score_celltype*motif_score_1,[0,1])
				score_pred2 = correlation_score*score_2
				field_query = field_query_1 + ['max_accessibility_score_celltype','score_celltype','score_pred_celltype']
				list1 = list1+[max_accessibility_score_celltype,score_2,score_pred2]
			
			dict1 = dict(zip(field_query,list1))
			df1 = pd.DataFrame.from_dict(data=dict1,orient='columns',dtype=np.float32)
			df1['peak_id'] = peak_id1
			df1 = df1.loc[:,['peak_id']+field_query]
			df1.index = [motif_id1]*df1.shape[0]
			df1 = df1.sort_values(by=['score_pred1'],ascending=False)
			list_query1.append(df1)
			if (verbose>0) and (i1%100==0):
				print('TF:%s, peak number:%d, %d'%(motif_id1,len(peak_id1),i1))
				print(np.max(score_pred1),np.min(score_pred1),np.mean(score_pred1),np.median(score_pred1))

		df_pre1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
		if (save_mode>0) and (output_filename!=''):
			df_pre1.to_csv(output_filename,sep='\t',float_format='%.6f')

		return df_pre1

	## ======================================================
	# select peak-TF links using thresholds on the TF binding scores estimated by the in silico ChIP-seq library method
	def test_peak_tf_score_normalization_query_1(self,data=[],peak_query_vec=[],motif_query_vec=[],input_filename='',thresh_score=0.1,
													save_mode=1,output_filename='',filename_annot='',select_config={}):

		"""
		select peak-TF links using thresholds on the TF binding score estimated by the in silico ChIP-seq library method
		:param data: (dataframe) the TF binding scores computed by the in silico ChIP-seq method (each row corresponds to a peak-TF link established by motif presence in the peak locus)
		:param peak_query_vec: (array) the ATAC-seq peak loci for which to estimate peak-TF links
		:param motif_query_vec: (array) TF names
		:param input_filename: path of the file which saved the computed in silico ChIP-seq TF binding score dataframe
		:param thresh_score: threshold on the in silico ChIP-seq TF binding score to select the peak-tf links
		:param save_mode: indicator of whether to save data
		:param output_filename: the filename to save the computed in silico ChIP-seq TF binding scores
		:param filename_annot: annotation used in potential filename to save data
		:param select_config: dictionary containing parameters
		:return: (dataframe) the in silico ChIP-seq TF binding scores and other associated scores of the selected peak-TF links
		"""

		filename_annot1 = filename_annot
		df_pre1 = data
		if len(df_pre1)==0:
			if input_filename=='':
				# input_file_path1 = self.save_path_1
				input_file_path1 = select_config['data_path']
				input_filename_1 = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path1,filename_annot1)
			else:
				input_filename_1 = input_filename

			if (os.path.exists(input_filename_1)==False):
				print('the file does not exist: %s'%(input_filename_1))
				return
			df_pre1 = pd.read_csv(input_filename_1,index_col=0,sep='\t') # load TF binding scores computed by the in silico ChIP-seq library method

		df_pre2 = df_pre1.loc[df_pre1['score_pred1']>thresh_score]
		print('peak-TF link annotations including TF binding scores estimated by in silico ChIP-seq, dataframe of size ',df_pre1.shape)
		print('peak-TF links with TF binding score above %s, dataframe of size '%(thresh_score),df_pre2.shape)

		if save_mode>0:
			if output_filename=='':
				output_file_path = select_config['data_path']
				output_filename = '%s/test_motif_score_normalize_insilico.%s.thresh%s.txt'%(output_file_path,filename_annot1,thresh_score)
			df_pre2.to_csv(output_filename,sep='\t')

		return df_pre2

	## ====================================================
	# compute regularization scores of peak-TF links
	def test_peak_tf_score_normalization_pre_compute(self,gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],df_gene_peak_query=[],
														motif_data=[],motif_data_score=[],peak_read=[],rna_exprs=[],peak_read_celltype=[],df_peak_annot=[],
														flag_motif_score_quantile=0,flag_motif_score_basic=1,overwrite=False,beta_mode=0,
														save_mode=1,output_file_path='',filename_annot='',verbose=0,select_config={}):

		"""
		compute regularization scores of peak-TF links
		:param peak_query_vec: (array) the ATAC-seq peak loci for which to estimate peak-TF links, default: the indices of motif_data
		:param motif_query_vec: (array) TF names, default: columns of the peak accessibility-TF expression correlation matrix
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param motif_data_score: (dataframe) motif scores from motif scanning results (row:ATAC-seq peak locus, column:TF)
		:param df_peak_tf_expr_corr_: (dataframe) the peak accessibility-TF expression correlation matrix (row: ATAC-seq peak locus, column:TF)
		:param input_filename: path of the file which saved the pre-computed peak accessibility-TF expression correlation matrix
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param peak_read_celltype: peak accessibility matrix for cell types (if we want to use maximal peak accessibility across cell types, by default we use the maximal accessiblity of a peak locus across the metacells)
		:param df_peak_annot: (dataframe) peak loci attributes including maximal peak accessibility across metacells
		:param flag_motif_score_quantile: indicator of whether to estimate quantiles of motif scores in the peak loci with motifs of the given TF
		:param flag_motif_score_basic: indicator of whether to query motif score variation across peak loci with motifs of the given TF
		:param overwrite: indicator of whether to overwrite the current file of computed peak-TF link regularization scores
		:param beta_mode: indicator of whether to perform computation for all the TFs with expressions or for a subset of TFs
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) annotations of the estimated regularization scores of the peak-TF links and other related scores;
		             dataframe includes the columns: motif_score_minmax, motif_score_log_normalize_bound, score_accessibility_minmax, score_accessibility, score_1 (the regularization score);
		         2. (dataframe) motif score variation across peak loci with the motifs of each TF
		         3. (dataframe) peak accessibility-related peak-TF link annotations including the quantiles of the maximal peak accessibilities across the metacells of the peak loci with the motifs of each TF
		"""

		sample_id = peak_read.index
		if len(motif_data_score)==0:
			motif_data_score = self.motif_data_score
		if len(motif_data)==0:
			motif_data = (motif_data_score.abs()>0)

		verbose_internal = self.verbose_internal
		if verbose_internal>0:
			print('motif scanning data, dataframe of size ',motif_data.shape)
			print('motif scores, dataframe of size ',motif_data_score.shape)

		motif_query_vec_ori = np.unique(motif_data.columns)
		motif_query_num_ori = len(motif_query_vec_ori)
		peak_loc_ori = motif_data.index
		peak_loc_num = len(peak_loc_ori)
		if len(motif_query_vec)==0:
			motif_query_vec = motif_query_vec_ori
		else:
			motif_query_vec = pd.Index(motif_query_vec).intersection(motif_query_vec_ori,sort=False)
		motif_query_num = len(motif_query_vec)

		if len(peak_query_vec)==0:
			peak_query_vec = peak_loc_ori
		else:
			peak_query_vec = pd.Index(peak_query_vec).intersection(peak_loc_ori,sort=False)
		peak_query_num = len(peak_query_vec)
		print('TF number and peak loci number in the motif scanning data: %d, %d'%(motif_query_num_ori,peak_loc_num))
		print('TFs and peak loci to estimate peak-TF associations: %d, %d'%(motif_query_num,peak_query_num))

		motif_data = motif_data.loc[peak_query_vec,motif_query_vec]
		motif_query_vec_ori = np.unique(motif_data.columns)
		motif_query_num_ori = len(motif_query_vec_ori)
		peak_loc_ori = motif_data.index
		peak_loc_num = len(peak_loc_ori)

		min_peak_number = 1
		field_query1 = ['coef_1','coef_std','coef_quantile','coef_mean_deviation']
		quantile_vec_1 = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
		field_query2 = ['max','min','mean','median']+quantile_vec_1
		df_motif_score_basic1 = pd.DataFrame(index=motif_query_vec,columns=field_query1,dtype=np.float32)
		df_motif_access_basic1 = pd.DataFrame(index=motif_query_vec,columns=field_query2,dtype=np.float32)
		
		b = 0.75
		thresh_pre1 = 1
		coef_motif_score_combination = -np.log(b)
		print('coefficient for motif score combination ',coef_motif_score_combination)
		dict_motif_query = dict()
		flag_query1=1
		save_mode=1

		column_id_query = 'max_accessibility_score'
		load_mode = 0
		if (len(df_peak_annot)>0) and (column_id_query in df_peak_annot.columns):
			load_mode = 1  # query maximal peak accessibility across metacells from peak annotations

		list_motif_score = []
		field_query_1 = ['motif_score_minmax','motif_score_log_normalize_bound']
		if flag_query1>0:
			motif_query_num_ori = len(motif_query_vec_ori)
			motif_query_num = motif_query_num_ori

			for i1 in range(motif_query_num):
				motif_id1 = motif_query_vec_ori[i1]
				id1 = motif_data.loc[:,motif_id1]>0
				peak_id1 = peak_loc_ori[id1]
				motif_score = motif_data_score.loc[peak_id1,motif_id1]

				# normalize motif score
				motif_score_minmax = motif_score/np.max(motif_score) # normalize motif_score per motif
				motif_score_log = np.log(1+motif_score) # log transformation of motif score
				score_compare = np.quantile(motif_score_log,0.95)
				motif_score_log_normalize = motif_score_log/np.max(motif_score_log) # normalize motif_score per motif
				motif_score_log_normalize_bound = motif_score_log/score_compare
				motif_score_log_normalize_bound[motif_score_log_normalize_bound>1] = 1.0

				field_query = field_query_1
				list1 = [motif_score_minmax,motif_score_log_normalize_bound]
				if flag_motif_score_quantile>0:
					t_vec_1 = np.asarray(motif_score)[:,np.newaxis]
					normalize_type = 'uniform'
					query_num1 = t_vec_1.shape[0]
					num_quantiles = np.min([query_num1,1000])
					score_mtx = quantile_transform(t_vec_1,n_quantiles=num_quantiles,output_distribution=normalize_type)
					motif_score_quantile = score_mtx[:,0]

					field_query = field_query_1+['motif_score_quantile']
					list1 = list1 + [motif_score_quantile]
				
				dict1 = dict(zip(field_query,list1))
				df_motif_score_2 = pd.DataFrame.from_dict(data=dict1,orient='columns',dtype=np.float32)
				df_motif_score_2['peak_id'] = peak_id1
				df_motif_score_2.index = np.asarray(peak_id1)
			
				# normalize peak accessibility for peak loci with TF motif
				if load_mode==0:
					max_accessibility_score = peak_read.loc[:,peak_id1].max(axis=0)
				else:
					max_accessibility_score = df_peak_annot.loc[peak_id1,column_id_query]

				t_value_1 = utility_1.test_stat_1(max_accessibility_score,quantile_vec=quantile_vec_1)
				df_motif_access_basic1.loc[motif_id1,field_query2] = np.asarray(t_value_1)

				median_access_value = df_motif_access_basic1.loc[motif_id1,'median']
				max_access_value = df_motif_access_basic1.loc[motif_id1,'max']
				thresh_score_accessibility = median_access_value

				if median_access_value<0.01:
					thresh_query_2= 0.75
					thresh_score_accessibility = df_motif_access_basic1.loc[motif_id1,thresh_query_2]
				
				b2_score = 0.90
				a2 = -np.log(1-b2_score)/(thresh_score_accessibility)
				
				score_accessibility = 1-np.exp(-a2*max_accessibility_score) # y=1-exp(-ax)
				score_accessibility_minmax = max_accessibility_score/max_access_value
				if verbose>0:
					if i1%100==0:
						print('df_motif_score, mean_value ',df_motif_score_2.shape,np.asarray(df_motif_score_2.mean(axis=0,numeric_only=True)),i1,motif_id1)
						print('median_value, thresh_score_accessibility, b2_score, a2 ',median_access_value,thresh_score_accessibility,b2_score,a2,i1,motif_id1)
						print('score_accessibility_minmax ',motif_id1,i1,score_accessibility_minmax.max(),score_accessibility_minmax.min(),score_accessibility_minmax.idxmax(),score_accessibility_minmax.idxmin(),score_accessibility_minmax.mean(),score_accessibility_minmax.median())
				
				lower_bound = 0.5
				score_1 = minmax_scale(score_accessibility*motif_score_log_normalize_bound,[lower_bound,1]) # the scaling score
				
				df_motif_score_2['max_accessibility_score'] = max_accessibility_score
				df_motif_score_2['score_accessibility'] = score_accessibility
				df_motif_score_2['score_accessibility_minmax'] = score_accessibility_minmax
				df_motif_score_2['score_1'] = score_1
				df_motif_score_2.index = [motif_id1]*df_motif_score_2.shape[0]
				list_motif_score.append(df_motif_score_2)

				# query basic statistics of motif score on motif score variation
				if flag_motif_score_basic>0:
					# max_value, min_value, mean_value, median_value = np.max(motif_score), np.min(motif_score), np.mean(motif_score), np.median(motif_score)
					t_value_2 = utility_1.test_stat_1(motif_score,quantile_vec=quantile_vec_1)
					max_value, min_value, mean_value, median_value = t_value_2[0:4]
					coef_1 = (max_value-min_value)/(max_value+min_value)
					coef_std = np.std(motif_score)/mean_value
					Q1, Q3 = np.quantile(motif_score,0.25), np.quantile(motif_score,0.75)
					coef_quantile = (Q3-Q1)/(Q1+Q3)
					coef_mean_deviation = np.mean(np.abs(motif_score-mean_value))/mean_value
					
					if verbose>0:
						print('motif_id1, peak_id1 ',i1,motif_id1,len(peak_id1))
						print('coef_1, coef_std, coef_quantile, coef_mean_deviation ',coef_1,coef_std,coef_quantile,coef_mean_deviation,i1,motif_id1)
					df_motif_score_basic1.loc[motif_id1,field_query1] = [coef_1,coef_std,coef_quantile,coef_mean_deviation]
					df_motif_score_basic1.loc[motif_id1,field_query2] = np.asarray(t_value_2)

			df_motif_score_query = pd.concat(list_motif_score,axis=0,join='outer',ignore_index=False)		
			if save_mode==1:
				output_filename1 = '%s/test_motif_score_basic1.%s.1.txt'%(output_file_path,filename_annot)
				df_motif_score_basic1.to_csv(output_filename1,sep='\t',float_format='%.6f')
				
				output_filename2 = '%s/test_motif_score_normalize.pre_compute.%s.txt'%(output_file_path,filename_annot)
				df_motif_score_query.to_csv(output_filename2,sep='\t',float_format='%.5f')
				
				output_filename3 = '%s/test_motif_access_basic1.%s.1.txt'%(output_file_path,filename_annot)
				df_motif_access_basic1.to_csv(output_filename3,sep='\t',float_format='%.6f')

			return df_motif_score_query, df_motif_score_basic1, df_motif_access_basic1

	## ====================================================
	# initiate peak-TF-gene links and compute association scores
	def test_query_feature_link_pre2(self,gene_query_vec=[],motif_query_vec=[],df_gene_peak_query=[],peak_distance_thresh=2000,
										df_peak_query=[],peak_loc_query=[],df_gene_tf_corr_peak=[],
										atac_ad=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],
										motif_data=[],motif_data_score=[],dict_motif_data={},
										interval_peak_corr=50,interval_local_peak_corr=10,
										flag_load_pre1=0,flag_load_1=0,overwrite_1=False,overwrite_2=False,parallel_mode=1,input_file_path='',
										save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		initiate peak-TF-gene links and compute association scores
		:param gene_query_vec: (array or list) the target genes
		:param motif_query_vec: (array) TF names
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param df_peak_query: (dataframe) peak loci attributes
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-TF-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) scaled gene expressions (z-scores) of the metacells (log-transformed normalized count matrix with standard scaling) (row:metacell, column:gene)
		:param rna_exprs_unscaled: (dataframe) unscaled gene expressions of the metacells (log-transformed normalized count matrix)
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param motif_data_score: (dataframe) motif scores from motif scanning results (row:ATAC-seq peak locus, column:TF)
		:param dict_motif_data: dictionary containing the motif scanning data
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param interval_local_peak_corr: the number of genes in the sub-batch for which to compute peak-gene correlations in the batch mode
		:param flag_load_pre1: indicator of whether to load ATAC-seq data and gene expression data
		:param flag_load_1: indicator of whether to load the motif scanning data
		:param overwrite_1: indicator of whether to overwrite the current file of computed in silico ChIP-seq TF binding scores
		:param overwrite_2: indicator of whether to overwrite the current file of computed peak-TF link regularization scores
		:param parallel_mode: indicator of whethter to perform computation in parallel
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-TF-gene links including association scores
		"""

		data_file_type = select_config['data_file_type']
		data_file_type_ori = data_file_type

		data_file_type_query = data_file_type
		data_file_type_annot = data_file_type_query.lower()
		
		run_id = select_config['run_id']
		type_id_feature = select_config['type_id_feature']

		file_save_path = select_config['data_path_save']	# data_path_save = data_path_save_local
		input_file_path = file_save_path
		print('input_file_path: %s'%(input_file_path))
		verbose_internal = self.verbose_internal

		if output_file_path=='':
			output_file_path = file_save_path

		output_file_path_ori = output_file_path
		
		if (flag_load_pre1==0):
			if (len(peak_read)==0) or (len(rna_exprs)==0):
				flag_load_pre1 = 1

		# query ATAC-seq and RNA-seq normalized read counts
		if flag_load_pre1>0:
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_load_data_pre2(flag_format=False,flag_scale=0,select_config=select_config)

			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.peak_read = peak_read

			## load gene annotations
			flag_gene_annot_query=0
			if flag_gene_annot_query>0:
				print('load gene annotations')
				start = time.time()
				self.test_gene_annotation_query1(select_config=select_config)
				stop = time.time()
				print('used: %.5fs'%(stop-start))

		if len(peak_read)==0:
			peak_read = self.peak_read

		if len(rna_exprs)==0:
			meta_scaled_exprs = self.meta_scaled_exprs
			meta_exprs_2 = self.meta_exprs_2
			rna_exprs = meta_scaled_exprs
			rna_exprs_unscaled = meta_exprs_2

		sample_id = rna_exprs.index
		rna_exprs_unscaled = rna_exprs_unscaled.loc[sample_id,:]
		peak_read = peak_read.loc[sample_id,:]
		
		method_type_feature_link = select_config['method_type_feature_link']
		method_type_query = method_type_feature_link

		if len(motif_data)==0:
			flag_load_1 = 1

		# load motif scanning data and motif scores
		if flag_load_1>0:
			motif_data, motif_data_score, motif_query_name_expr = self.test_load_motif_data_2(dict_motif_data=dict_motif_data,
																								method_type=method_type_query,
																								select_config=select_config)

		peak_loc_ori = peak_read.columns
		motif_data = motif_data.loc[peak_loc_ori,:]
		motif_data_score = motif_data_score.loc[peak_loc_ori,:]

		print('motif scanning data, dataframe of size ',motif_data.shape)
		print('data preview: ')
		print(motif_data[0:2])
		print('motif scores, dataframe of size ',motif_data_score.shape)
		print('data preview: ')
		print(motif_data_score[0:2])

		motif_query_name_expr = motif_data.columns
		if len(motif_query_vec)==0:
			motif_query_vec = motif_query_name_expr
		
		motif_query_num = len(motif_query_vec)
		print('number of the TFs: %d'%(motif_query_num))
		if verbose_internal>0:
			print('preview: ',motif_query_vec[0:5])

		# peak accessibility-TF expr correlation
		filename_prefix_1 = filename_prefix_save
		column_1 = 'correlation_type'
		if column_1 in select_config:
			correlation_type = select_config[column_1]
		else:
			correlation_type = 'spearmanr'
		# flag_query1=0
		flag_query_peak_basic=0

		# flag_motif_score_normalize_2=1
		flag_query_peak_ratio=1
		flag_query_distance=0
		
		flag_peak_tf_corr = 1
		flag_gene_tf_corr = 1
		flag_motif_score_normalize = 1
		flag_gene_tf_corr_peak_compute = 1
		flag_gene_tf_corr_peak_combine = 0
		field_query = ['flag_query_peak_basic','flag_query_peak_ratio',
						'flag_gene_tf_corr','flag_peak_tf_corr','flag_motif_score_normalize',
						'flag_gene_tf_corr_peak_compute','flag_gene_tf_corr_peak_combine']

		list1 = [flag_query_peak_basic,flag_query_peak_ratio,
				flag_gene_tf_corr,flag_peak_tf_corr,flag_motif_score_normalize,
				flag_gene_tf_corr_peak_compute,flag_gene_tf_corr_peak_combine]

		overwrite_pre2 = False
		select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=list1,
																				overwrite=overwrite_pre2,
																				select_config=select_config)

		flag_query_peak_basic,flag_query_peak_ratio,flag_gene_tf_corr,flag_peak_tf_corr,flag_motif_score_normalize,flag_gene_tf_corr_peak_compute,flag_gene_tf_corr_peak_combine = param_vec
		for (field_id,query_value) in zip(field_query,param_vec):
			print('field_id, query_value: ',field_id,query_value)

		if flag_motif_score_normalize>0:
			flag_motif_score_normalize_1=1
			flag_motif_score_normalize_1_query=1
			flag_motif_score_normalize_2=1
		else:
			flag_motif_score_normalize_1=0
			flag_motif_score_normalize_1_query=0
			flag_motif_score_normalize_2=0

		column_1 = 'flag_motif_score_normalize_2'
		if column_1 in select_config:
			flag_motif_score_normalize_2 = select_config[column_1]

		# query tf binding activity score
		# recompute gene_tf_corr_peak for peak-gene link query added
		flag_query_recompute=0
	
		flag_motif_score_normalize_thresh1 = flag_motif_score_normalize_1_query
		df_gene_peak_query1 = []

		flag_score_pre1 = 1
		if 'flag_score_pre1' in select_config:
			flag_score_pre1 = select_config['flag_score_pre1']

		df_gene_tf_corr_peak_1 = []
		if flag_score_pre1 in [1,3]:
			# initiate peak-TF-gene links and compute the scores used in calculating the peak-TF-gene association scores
			df_link_query_1 = self.test_gene_peak_query_correlation_compute_1(gene_query_vec=gene_query_vec,
																				motif_query_vec=motif_query_vec,
																				df_gene_peak_query=df_gene_peak_query,
																				peak_distance_thresh=peak_distance_thresh,
																				df_peak_query=[],
																				peak_loc_query=[],
																				atac_ad=atac_ad,
																				peak_read=peak_read,
																				rna_exprs=rna_exprs,
																				rna_exprs_unscaled=rna_exprs_unscaled,
																				motif_data=motif_data,
																				motif_data_score=motif_data_score,
																				flag_query_peak_basic=flag_query_peak_basic,
																				flag_peak_tf_corr=flag_peak_tf_corr,
																				flag_gene_tf_corr=flag_gene_tf_corr,
																				flag_motif_score_normalize_1=flag_motif_score_normalize_1,
																				flag_motif_score_normalize_thresh1=flag_motif_score_normalize_thresh1,
																				flag_motif_score_normalize_2=flag_motif_score_normalize_2,
																				flag_gene_tf_corr_peak_compute=flag_gene_tf_corr_peak_compute,
																				interval_peak_corr=50,
																				interval_local_peak_corr=10,
																				overwrite_1=overwrite_1,
																				overwrite_2=overwrite_2,
																				save_mode=save_mode,
																				output_file_path=output_file_path,
																				output_filename='',
																				filename_prefix_save=filename_prefix_save,
																				verbose=verbose,
																				select_config=select_config)

			df_gene_tf_corr_peak_1 = df_link_query_1
			df_gene_peak_query1 = df_gene_tf_corr_peak_1
		
		type_id_query = 2
		type_id_compute = 1
		if flag_gene_tf_corr_peak_combine>0:
			df_combine, df_ratio = self.test_partial_correlation_gene_tf_cond_peak_combine_pre1(input_file_path=input_file_path,
																								type_id_query=type_id_query,
																								type_id_compute=type_id_compute,
																								overwrite=0,
																								save_mode=save_mode,
																								output_file_path='',
																								output_filename='',output_filename_list=[],
																								filename_prefix_save=filename_prefix_save,
																								filename_prefix_vec=[],
																								verbose=0,select_config=select_config)

		flag_save_interval = 0
		df_gene_tf_corr_peak_pre1 = []
		if len(df_gene_tf_corr_peak_1)>0:
			df_gene_tf_corr_peak_pre1 = df_gene_tf_corr_peak_1
		elif len(df_gene_tf_corr_peak)>0:
			df_gene_tf_corr_peak_pre1 = df_gene_tf_corr_peak
		else:			
			file_path_motif_score = select_config['file_path_motif_score']
			input_file_path_query = file_path_motif_score
			# input_filename = '%s/pre2.pcorr_query1.1.txt'%(input_file_path_query) # TODO: to update
			filename_prefix_cond = select_config['filename_prefix_cond']
			column_1 = 'filename_annot_interval'
			if column_1 in select_config:
				filename_annot_interval = select_config[column_1]
			else:
				filename_annot_interval = '1'
			input_filename = '%s/%s.pcorr_query1.%s'%(input_file_path_query,filename_prefix_cond,filename_annot_interval)
			if os.path.exists(input_filename)==True:
				print('load gene-TF expression partial correlation estimation')
				df_gene_tf_corr_peak_pre1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				print(input_filename)
			else:
				print('please provide gene-TF expression partial correlation estimation')

		flag_init_score=1
		flag_init_score = (flag_score_pre1 in [2,3])

		if len(df_gene_tf_corr_peak_pre1)>0:
			print('initial peak-TF-gene links, dataframe of size ',df_gene_tf_corr_peak_pre1.shape)
			print('data preview ')
			print(df_gene_tf_corr_peak_pre1[0:2])
		else:
			flag_init_score=0

		# calculate peak-TF-gene association scores
		if flag_init_score>0:
			df_gene_peak_query1 = self.test_gene_peak_query_correlation_gene_pre2_compute_3(gene_query_vec=[],df_gene_peak_query=df_gene_peak_query,
																							peak_distance_thresh=peak_distance_thresh,
																							df_peak_query=[],
																							peak_loc_query=[],
																							df_gene_tf_corr_peak=df_gene_tf_corr_peak_pre1,
																							flag_save_ori=0,
																							flag_save_interval=flag_save_interval,
																							parallel_mode=parallel_mode,
																							save_mode=1,
																							output_file_path=output_file_path,
																							output_filename='',
																							filename_prefix_save=filename_prefix_save,
																							verbose=verbose,select_config=select_config)

		return df_gene_peak_query1

	## ====================================================
	# initiate peak-TF-gene links and compute the scores used in calculating the peak-TF-gene association scores
	def test_gene_peak_query_correlation_compute_1(self,gene_query_vec=[],motif_query_vec=[],df_gene_peak_query=[],peak_distance_thresh=2000,
														df_peak_query=[],peak_loc_query=[],atac_ad=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],
														motif_data=[],motif_data_score=[],flag_query_peak_basic=0,flag_peak_tf_corr=0,
														flag_gene_tf_corr=0,flag_motif_score_normalize_1=0,flag_motif_score_normalize_thresh1=0,
														flag_motif_score_normalize_2=0,flag_gene_tf_corr_peak_compute=0,
														interval_peak_corr=50,interval_local_peak_corr=10,overwrite_1=False,overwrite_2=False,overwrite_3=False,
														save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		initiate peak-TF-gene links and compute the scores used in calculating the peak-TF-gene association scores
		:param gene_query_vec: (array or list) the target genes
		:param motif_query_vec: (array) TF names
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param df_peak_query: (dataframe) peak loci attributes
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-TF-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) scaled gene expressions (z-scores) of the metacells (log-transformed normalized count matrix with standard scaling) (row:metacell, column:gene)
		:param rna_exprs_unscaled: (dataframe) unscaled gene expressions of the metacells (log-transformed normalized count matrix)
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:gene)
		:param motif_data_score: (dataframe) motif scores from motif scanning results (row:ATAC-seq peak locus, column:TF)
		:param flag_query_peak_basic: indicator of whether to query accessibility-related peak attributes
		:param flag_peak_tf_corr: indicator of whether to compute peak accessibility-TF expression correlation and p-value
		:param flag_gene_tf_corr: indicator of whether to compute gene-TF expression correlation and p-value
		:param flag_motif_score_normalize_1: indicator of whether to estimate TF binding scores using the in silico ChIP-seq method
		:param flag_motif_score_normalize_thresh1: indicator of whether to predict TF binding using the TF binding scores estimated by the in silico ChIP-seq method
		:param flag_motif_score_normalize_2: indicator of whether to estimate regularization scores of peak-TF links
		:param flag_gene_tf_corr_peak_compute: indicator of whether to compute gene-TF expression partial correlation conditioned on the peak accessibility
		:param interval_peak_corr: the number of genes in a batch for which to compute gene-TF expression partial correlation given peak accessibility in the batch mode
		:param interval_local_peak_corr: the number of genes in a sub-batch for which to compute gene-TF expression partial correlation given peak accessibility in the batch mode
		:param overwrite_1: indicator of whether to overwrite the current file of in silico ChIP-seq TF binding scores
		:param overwrite_2: indicator of whether to overwrite the current file of peak-TF link regularization scores
		:param overwrite_3: indicator of whether to overwrite the current file of gene-TF expr correlation given peak accessibility
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-TF-gene links
		"""

		file_save_path_1 = select_config['data_path_save']
		if output_file_path=='':
			output_file_path = file_save_path_1
		file_save_path = output_file_path
		
		# peak accessibility-TF expr correlation estimation
		data_file_type = select_config['data_file_type']
		data_file_type_query = data_file_type
		output_file_path_default = file_save_path

		filename_prefix_save_default = select_config['filename_prefix_save_default']
		filename_prefix_save_pre1 = filename_prefix_save_default
		if 'filename_annot_save_default' in select_config:
			filename_annot_default = select_config['filename_annot_save_default']
		else:
			filename_annot_default = data_file_type_query
			select_config.update({'filename_annot_save_default':filename_annot_default})
		
		# filename_annot_1 = filename_annot_default
		filename_annot = filename_annot_default

		if 'file_path_motif_score' in select_config:
			file_path_motif_score = select_config['file_path_motif_score']
		else:
			file_path_motif_score = file_save_path

		file_save_path2 = file_path_motif_score
		input_file_path = file_save_path2
		# input_file_path2 = file_save_path2
		output_file_path = file_save_path2
		
		# query the maximum accessibility of peak loci in the metacells
		# query open peaks in the metacells
		df_peak_annot = self.atac_meta_ad.var.copy()
		if flag_query_peak_basic>0:
			# type_id_peak_ratio=2
			type_id_peak_ratio=0
			if type_id_peak_ratio in [0,2]:
				print('query accessibility-related peak attributes')
				df_peak_access_basic_1, quantile_value_1 = self.test_peak_access_query_basic_1(peak_read=peak_read,
																								rna_exprs=rna_exprs,
																								df_annot=df_peak_annot,
																								thresh_value=0.1,
																								flag_ratio=1,
																								flag_access=1,
																								save_mode=1,
																								output_file_path=output_file_path,
																								output_filename='',
																								filename_annot=filename_annot,
																								select_config=select_config)
				self.peak_annot = df_peak_access_basic_1
				
			if type_id_peak_ratio in [1,2]:
				low_dim_embedding = 'X_svd'
				# pval_cutoff = 1e-2
				n_neighbors = 3
				# bin_size = 5000
				
				atac_meta_ad = self.atac_meta_ad
				# print('atac_meta_ad \n',atac_meta_ad)

				# query open peaks in the metacells
				atac_meta_ad = self.test_peak_access_query_basic_pre1(adata=atac_meta_ad,low_dim_embedding=low_dim_embedding,
																		n_neighbors=n_neighbors,
																		select_config=select_config)
				self.atac_meta_ad = atac_meta_ad

		self.dict_peak_tf_corr_ = dict()

		column_1 = 'filename_annot_motif_score'
		if column_1 in select_config:
			filename_annot_motif_score = select_config[column_1]
		else:
			filename_annot_motif_score = filename_annot_default
			select_config.update({column_1:filename_annot_motif_score})
		filename_annot = filename_annot_motif_score

		if 'file_path_peak_tf' in select_config:
			file_path_peak_tf = select_config['file_path_peak_tf']
		else:
			file_path_peak_tf = file_save_path
		
		input_file_path_2 = file_path_peak_tf
		output_file_path_2 = input_file_path_2
		filename_motif_score_1 = '%s/test_motif_score_normalize_insilico.%s.txt'%(input_file_path_2,filename_annot)
		filename_motif_score_2 = '%s/test_motif_score_normalize.pre_compute.%s.txt'%(input_file_path_2,filename_annot)
		select_config.update({'filename_motif_score_normalize_1':filename_motif_score_1,
								'filename_motif_score_normalize_2':filename_motif_score_2})

		if 'flag_peak_tf_corr' in select_config:
			flag_peak_tf_corr = select_config['flag_peak_tf_corr']
		
		correlation_type=select_config['correlation_type']
		filename_prefix = 'test_peak_tf_correlation.%s'%(data_file_type_query)
		if 'filename_prefix_peak_tf' in select_config:
			filename_prefix_peak_tf = select_config['filename_prefix_peak_tf']
			filename_prefix = filename_prefix_peak_tf
		else:
			select_config.update({'filename_prefix_peak_tf':filename_prefix})

		field_load = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected']
		filename_annot_vec = [correlation_type,'pval','pval_corrected','motif_basic']
		if flag_peak_tf_corr>0:
			save_mode = 1
			flag_load_1 = 0
			if 'flag_load_peak_tf' in select_config:
				flag_load_1 = select_config['flag_load_peak_tf']

			input_filename_list1 = ['%s/%s.%s.1.txt'%(file_path_peak_tf,filename_prefix,filename_annot1) for filename_annot1 in filename_annot_vec[0:3]]
			dict_peak_tf_corr_ = self.test_peak_tf_correlation_query_1(motif_data=motif_data,
																		peak_query_vec=[],
																		motif_query_vec=motif_query_vec,
																		peak_read=peak_read,
																		rna_exprs=rna_exprs,
																		correlation_type=correlation_type,
																		flag_load=flag_load_1,
																		field_load=field_load,
																		save_mode=save_mode,
																		input_filename_list=input_filename_list1,
																		output_file_path=file_path_peak_tf,
																		filename_prefix=filename_prefix,
																		select_config=select_config)
			self.dict_peak_tf_corr_ = dict_peak_tf_corr_
		
		df_peak_annot = self.df_peak_annot
		beta_mode = 0
		if 'beta_mode' in select_config:
			beta_mode = select_config['beta_mode']

		df_binding_score_1 = []
		filename_annot = filename_annot_motif_score
		if flag_motif_score_normalize_1>0:
			# Unify does not use the estimates from in silico ChIP-seq;
			# running in silico ChIP-seq is only to query peak accessiblty-TF expression correlation and p-value;
			# we can also compare the TF binding predictions by in silico ChIP-seq and Unify;
			print('estimating TF binding scores using in silico ChIP-seq')
			start = time.time()
			output_filename = filename_motif_score_1
			dict_peak_tf_corr_ = self.dict_peak_tf_corr_
			df_peak_tf_expr_corr_ = []
			input_filename_1 = ''
			if len(dict_peak_tf_corr_)>0:
				df_peak_tf_expr_corr_ = dict_peak_tf_corr_['peak_tf_corr']
			else:
				column_1 = 'filename_peak_tf_corr'
				if column_1 in select_config:
					input_filename_1 = select_config[column_1]
				else:
					print('please provide peak accessibility-TF expression correlation file')
					return

			# the in silico ChIP-seq library method
			# compute TF binding score with normalization
			df_binding_score_1 = self.test_peak_tf_score_normalization_1(peak_query_vec=[],motif_query_vec=motif_query_vec,
																			motif_data=motif_data,
																			motif_data_score=motif_data_score,
																			df_peak_tf_expr_corr_=df_peak_tf_expr_corr_,
																			input_filename=input_filename_1,
																			peak_read=peak_read,
																			rna_exprs=rna_exprs,
																			peak_read_celltype=[],
																			df_peak_annot=df_peak_annot,
																			overwrite=overwrite_1,
																			beta_mode=beta_mode,
																			output_file_path=file_path_peak_tf,
																			output_filename=output_filename,
																			filename_annot=filename_annot_motif_score,
																			verbose=verbose,
																			select_config=select_config)

			stop = time.time()
			if len(df_binding_score_1)>0:
				print('estimating TF binding score using in silico ChIP-seq used: %.5fs'%(stop-start))
			
		# TF binding prediction by the in silico ChIP-seq library method
		flag_motif_score_normalize_1_query = flag_motif_score_normalize_thresh1
		quantile_vec_1 = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
		field_query_1 = ['max','min','mean','median']+quantile_vec_1
	
		if flag_motif_score_normalize_1_query>0:
			thresh_score_1 = 0.10
			if 'thresh_insilco_ChIP-seq' in select_config:
				thresh_score_1 = select_config['thresh_insilco_ChIP-seq']

			input_filename = select_config['filename_motif_score_normalize_1']
			b = input_filename.find('.txt')
			output_filename = '%s.thresh%s.txt'%(input_filename[0:b],thresh_score_1)
			select_config.update({'filename_motif_score_normalize_1_thresh1':output_filename})
			# peak-TF link selection by threshold using TF binding scores estimated by in silico ChIP-seq library method
			df_link_query_1 = self.test_peak_tf_score_normalization_query_1(data=df_binding_score_1,peak_query_vec=[],
																			motif_query_vec=motif_query_vec,
																			input_filename=input_filename,
																			thresh_score=thresh_score_1,
																			output_filename=output_filename,
																			filename_annot=filename_annot_motif_score,
																			select_config=select_config)
				
		# regularization score (scaling coefficient) estimation (the modified approach)
		if flag_motif_score_normalize_2>0:
			print('compute regularization scores of peak-TF links')
			start = time.time()
			# input_file_path_query = output_file_path
			input_filename = filename_motif_score_2
			
			if (os.path.exists(input_filename)==True) and (overwrite_2==False):
				print('the file exists: %s'%(input_filename))
				df_motif_score_query = pd.read_csv(input_filename,index_col=0,sep='\t')
			else:
				# compute regularization scores of peak-TF links
				df_motif_score_query = self.test_peak_tf_score_normalization_pre_compute(motif_query_vec=motif_query_vec,
																							motif_data=motif_data,
																							motif_data_score=motif_data_score,
																							peak_read=peak_read,
																							rna_exprs=rna_exprs,
																							peak_read_celltype=[],
																							df_peak_annot=df_peak_annot,
																							overwrite=overwrite_2,
																							output_file_path=file_path_peak_tf,
																							filename_annot=filename_annot_motif_score,
																							verbose=0,
																							select_config=select_config)
			self.df_motif_score_query = df_motif_score_query
			stop = time.time()
			print('computing regularization scores of peak-TF links used: %.5fs'%(stop-start))
			
		df_gene_peak = df_gene_peak_query
		column_idvec = ['gene_id','peak_id']
		column_id1, column_id2 = column_idvec[0:2]
		output_file_path = file_save_path2

		df_gene_tf_corr_peak = []

		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		print('query_id1:%d, query_id2:%d'%(query_id1,query_id2))
		start_id1, start_id2 = -1, -1
		iter_mode = 0
		if (query_id1>=0) and (query_id2>query_id1):
			iter_mode = 1
			start_id1, start_id2 = query_id1, query_id2

		# if start_id1>=0:
		if iter_mode>0:
			# filename_annot_save = '%d_%d'%(start_id1,start_id2)
			filename_annot_save = '%d_%d'%(query_id1,query_id2)
		else:
			filename_annot_save = '1'
		filename_annot_interval = filename_annot_save	
		select_config.update({'filename_annot_interval':filename_annot_interval})

		if flag_gene_tf_corr_peak_compute>0:
			if len(gene_query_vec)==0:
				# gene_query_vec = df_gene_peak['gene_id'].unique()
				gene_query_vec = df_gene_peak[column_id1].unique()
			gene_query_num_1 = len(gene_query_vec)

			# query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
			# if (query_id1>=0) and (query_id2>query_id1):
				# start_id1, start_id2 = query_id1, np.min([query_id2,gene_query_num_1])
				# if start_id1>start_id2:
				# 	print('start_id1, start_id2: ',start_id1,start_id2)
				# 	return

				# start_id1, start_id2 = query_id1, query_id2
			# if start_id1>=0:
			# 	# filename_annot_save = '%d_%d'%(start_id1,start_id2)
			# 	filename_annot_save = '%d_%d'%(query_id1,query_id2)
			# else:
			# 	filename_annot_save = '1'
			
			# select_config.update({'filename_annot_interval':filename_annot_save})
			# filename_prefix_save_1 = select_config['filename_prefix_default_1']
			# filename_prefix_save_1 = select_config['filename_prefix_cond']
			# if 'filename_prefix_save_2' in select_config:
			# 	filename_prefix_save_2 = select_config['filename_prefix_save_2']
			# 	filename_prefix_save_pre1 = filename_prefix_save_2
			# else:
			# 	filename_prefix_save_pre1 = filename_prefix_save_1
			filename_prefix_default_1 = select_config['filename_prefix_default_1']
			filename_prefix_cond = select_config['filename_prefix_cond']
			# filename_prefix_save_pre1 = filename_prefix_save_1
			filename_prefix_save_pre1 = filename_prefix_cond

			# to update
			output_filename = '%s/%s.pcorr_query1.%s.txt'%(output_file_path,filename_prefix_save_pre1,filename_annot_save)
			
			 #self.select_config = select_config
			filename_query = output_filename
			# self.select_config.update({'filename_gene_tf_cond':filename_query})
			select_config.update({'filename_gene_tf_cond':filename_query})
			self.select_config = select_config

			filename_list_pcorr_1 = []
			column_1 = 'filename_list_pcorr_1'
			if column_1 in select_config:
				filename_list_pcorr_1 = select_config[column_1]
			filename_list_pcorr_1.append(output_filename)
			select_config.update({column_1:filename_list_pcorr_1})

			gene_query_vec_pre1 = gene_query_vec
			# compute gene-TF expr partial correlation given peak accessibility in batch mode
			flag_pcorr_interval = self.flag_pcorr_interval
			if flag_pcorr_interval>0:
				if (start_id1>=0) and (start_id2>start_id1):
					start_id2 = np.min([query_id2,gene_query_num_1])
					gene_query_vec = gene_query_vec_pre1[start_id1:start_id2]
				else:
					gene_query_vec = gene_query_vec_pre1
			gene_query_num = len(gene_query_vec)
			print('gene_query_vec_pre1, gene_query_vec ',gene_query_num_1,gene_query_num,start_id1,start_id2)
			
			flag_gene_tf_corr_peak_1=1
			flag_gene_tf_corr_peak_pval=0

			df_gene_tf_corr_peak = []
			if (os.path.exists(filename_query)==True) and (overwrite_3==False):
				print('the file exists: %s'%(filename_query))
				# return

				flag_gene_tf_corr_peak_1 = 0
				df_gene_tf_corr_peak = pd.read_csv(filename_query,index_col=0,sep='\t')

			# compute gene-TF expresssion partial correlation given peak accessibility
			if flag_gene_tf_corr_peak_1>0:
				# print('estimate gene-TF expression partial correlation given peak accessibility')
				start = time.time()
				peak_query_vec, motif_query_vec = [], []
				type_id_query_2 = 2
				type_id_compute = 1
				motif_query_name_expr = self.motif_query_name_expr
				motif_data = self.motif_data
				motif_data_expr = motif_data.loc[:,motif_query_name_expr]
				# query peak accessibility-TF expression correlation and gene-TF expression partial correlation given peak accessibility
				df_gene_tf_corr_peak = self.test_query_score_function1(df_gene_peak_query=df_gene_peak,
																		motif_data=motif_data_expr,
																		gene_query_vec=gene_query_vec,
																		peak_query_vec=peak_query_vec,
																		motif_query_vec=motif_query_vec,
																		peak_read=peak_read,
																		rna_exprs=rna_exprs,
																		rna_exprs_unscaled=rna_exprs_unscaled,
																		type_id_query=type_id_query_2,
																		type_id_compute=type_id_compute,
																		flag_peak_tf_corr=0,
																		flag_gene_tf_corr_peak=flag_gene_tf_corr_peak_1,
																		flag_pval_1=0,
																		flag_pval_2=flag_gene_tf_corr_peak_pval,
																		save_mode=1,
																		output_file_path=output_file_path,
																		output_filename=output_filename,
																		verbose=verbose,select_config=select_config)

			if len(df_gene_tf_corr_peak)>0:
				# df_gene_tf_corr_peak.index = np.asarray(df_gene_tf_corr_peak['gene_id'])
				self.df_gene_tf_corr_peak = df_gene_tf_corr_peak

				# print('df_gene_tf_corr_peak ',df_gene_tf_corr_peak.shape)
				print('peak-TF-gene links with gene-TF expr partial correlation estimated, dataframe of size ',df_gene_tf_corr_peak.shape)
				print('data preview:')
				print(df_gene_tf_corr_peak[0:2])

		flag_gene_tf_corr_1 = flag_gene_tf_corr
		if 'flag_gene_tf_corr' in select_config:
			flag_gene_tf_corr_1 = select_config['flag_gene_tf_corr']

		# estimate gene-TF expression correlation
		if flag_gene_tf_corr_1>0:
			gene_query_vec_pre1 = self.gene_query_vec
			if len(gene_query_vec_pre1)==0:
				gene_query_vec_pre1 = rna_exprs.columns

			feature_vec_1 = gene_query_vec_pre1
			motif_query_name_expr = self.motif_query_name_expr
			# motif_data = self.motif_data
			motif_query_vec_pre1 = motif_query_name_expr
			feature_vec_2 = motif_query_vec_pre1

			column_1 = 'file_path_gene_tf'
			if column_1 in select_config:
				input_file_path_query = select_config[column_1]
			else:
				input_file_path_query = select_config['file_path_motif_score']
			
			output_file_path_query = input_file_path_query
			
			# data_file_type_query = select_config['data_file_type']
			filename_prefix_pre1 = data_file_type_query
			# filename_prefix_1 = select_config['filename_prefix_default']
			filename_prefix_default_1 = select_config['filename_prefix_default_1']
			# filename_prefix_cond = select_config['filename_prefix_cond']

			correlation_type = 'spearmanr'
			correlation_type_vec = [correlation_type]

			# query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
			# print('query_id1:%d, query_id2:%d'%(query_id1,query_id2))
			
			# if (query_id1>=0) and (query_id2>query_id1):
			if iter_mode>0:
				column_query = 'feature_query_num_1'
				if column_query in select_config:
					feature_query_num_1 = select_config[column_query]
				# 	if query_id1>feature_query_num_1:
				# 		print('query_id1, feature_query_num_1: ',query_id1,feature_query_num_1)
				# 		return
				# 	else:
				# 		query_id2 = np.min([query_id2,feature_query_num_1])

				# input_filename = '%s/%s.pcorr_query1.%d_%d.txt'%(input_file_path,filename_prefix_default_1,query_id1,query_id2)
				# input_filename = '%s/%s.pcorr_query1.%d_%d.txt'%(input_file_path,filename_prefix_cond,query_id1,query_id2)
				input_filename = '%s/%s.pcorr_query1.%s.txt'%(input_file_path,filename_prefix_cond,filename_annot_interval)
				df_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				feature_vec_1 = df_query_1[column_id1].unique()
				
				# filename_prefix_save_1 = 'test_gene_tf_correlation.%s.%d_%d'%(filename_prefix_pre1,query_id1,query_id2)
				filename_prefix_save_1 = 'test_gene_tf_correlation.%s.%s'%(filename_prefix_pre1,filename_annot_interval)
			else:
				filename_prefix_save_1 = 'test_gene_tf_correlation.%s'%(filename_prefix_pre1)

			print('compute gene-TF expression correlation')
			feature_num_1 = len(feature_vec_1)
			feature_num_2 = len(feature_vec_2)
			print('feature_vec_1, feature_vec_2: ',feature_num_1,feature_num_2)
			start = time.time()

			filename_save_1 = '%s/%s.%s.1.txt'%(output_file_path_query,filename_prefix_save_1,correlation_type)
			overwrite_2 = False
			if (os.path.exists(filename_save_1)==True) and (overwrite_2==False):
				print('the file exists: %s'%(filename_save_1))
			else:
				self.test_gene_expr_correlation_1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
													rna_exprs=rna_exprs,
													correlation_type_vec=correlation_type_vec,
													symmetry_mode=0,
													type_id_1=0,
													type_id_pval_correction=1,
													thresh_corr_vec=[],
													save_mode=1,
													save_symmetry=0,
													output_file_path=output_file_path_query,
													filename_prefix=filename_prefix_save_1,
													verbose=0,select_config=select_config)

			stop = time.time()
			print('computing gene-TF expression correlation for %d target genes and %d TFs used %.2fs'%(feature_num_1,feature_num_2,stop-start))

			# filename_annot2 = select_config['filename_save_annot_pre1']
			# filename_prefix_query1 = '%s/test_gene_tf_expr_correlation.%s'%(input_file_path_query,filename_annot2)
			# filename_prefix_query2 = '%s/test_gene_expr_correlation.%s'%(input_file_path_query,filename_annot2)

			# input_filename_pre1 = '%s.%s.1.txt'%(filename_prefix_query1,correlation_type)
			# input_filename_pre2 = '%s.%s.pval_corrected.1.txt'%(filename_prefix_query1,correlation_type)
			# input_filename_1 = '%s.%s.1.txt'%(filename_prefix_query2,correlation_type)
			# input_filename_2 = '%s.%s.pval_corrected.1.txt'%(filename_prefix_query2,correlation_type)
			
			# select_config.update({'filename_gene_tf_expr_correlation':input_filename_pre1,
			# 						'filename_gene_tf_expr_pval_corrected':input_filename_pre2,
			# 						'filename_gene_expr_correlation':input_filename_1,
			# 						'filename_gene_expr_pval_corrected':input_filename_2})
			
		return df_gene_tf_corr_peak
																
	## ====================================================
	# calculate peak-TF-gene association scores
	def test_gene_peak_query_correlation_gene_pre2_compute_3(self,gene_query_vec=[],df_gene_peak_query=[],peak_distance_thresh=2000,
																df_peak_query=[],peak_loc_query=[],df_gene_tf_corr_peak=[],
																flag_save_ori=0,flag_save_interval=0,parallel_mode=1,
																save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		calculate peak-TF-gene association scores
		:param gene_query_vec: (array or list) the target genes
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param df_peak_query: (dataframe) peak loci attributes
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-TF-gene links
		:param df_gene_tf_corr_peak: (dataframe) annotations of peak-TF-gene links including gene-TF expression partial correlation conditioned peak accessibility
		:param flag_save_ori: indicator of whether to save peak-TF-gene link annotations including association scores
		:param flag_save_interval: indicator of whether to save peak-TF-gene link annotations including association scores
		:param parallel_mode: indicator of whether to perform computation in parallel
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) peak-TF-gene link annotations including association scores
		"""

		# print('initial calculation of peak-TF-gene association scores')
		print('calculate peak-TF-gene association scores')
		# flag_init_score=0
		flag_init_score=1
		if 'flag_score_pre1' in select_config:
			flag_score_pre1 = select_config['flag_score_pre1']
			flag_init_score = flag_score_pre1

		lambda1=0.5
		lambda2=1-lambda1
		df_gene_peak_query1_1 = df_gene_peak_query
		if flag_init_score>0:
			dict_peak_tf_query = self.dict_peak_tf_query
			dict_gene_tf_query = self.dict_gene_tf_query

			load_mode_2 = -1
			if len(dict_peak_tf_query)==0:
				load_mode_2 = 0

			if len(dict_gene_tf_query)==0:
				load_mode_2 = load_mode_2 + 2

			print('load_mode_2: ',load_mode_2)
			# query gene-TF expression correlations and peak accessibilility-TF expression correlations
			if load_mode_2>=0:
				dict_peak_tf_query, dict_gene_tf_query = self.test_gene_peak_tf_query_score_init_pre1_1(gene_query_vec=[],peak_query_vec=[],
																										motif_query_vec=[],
																										load_mode=load_mode_2,input_file_path='',
																										save_mode=0,output_file_path=output_file_path,output_filename='',
																										verbose=verbose,select_config=select_config)

				if load_mode_2 in [0,2]:
					self.dict_peak_tf_query = dict_peak_tf_query

				if load_mode_2 in [1,2]:
					self.dict_gene_tf_query = dict_gene_tf_query

			flag_load_2=0
			if (self.df_peak_tf_1 is None):
				flag_load_2 = 1

			if flag_load_2>0:
				data_file_type_query = select_config['data_file_type']
				filename_annot = select_config['filename_annot_motif_score']
				# input_file_path2 = select_config['file_path_motif_score']
				input_file_path2 = select_config['file_path_peak_tf']

				filename_motif_score_1 = '%s/test_motif_score_normalize_insilico.%s.txt'%(input_file_path2,filename_annot)
				filename_motif_score_2 = '%s/test_motif_score_normalize.pre_compute.%s.txt'%(input_file_path2,filename_annot)
				df_peak_tf_1 = pd.read_csv(filename_motif_score_1,index_col=0,sep='\t')
				df_peak_tf_2 = pd.read_csv(filename_motif_score_2,index_col=0,sep='\t')
				verbose_internal = self.verbose_internal
				if verbose_internal==2:
					print('normalized TF binding scores estimated by the in silico ChIP-seq method, dataframe of size ',df_peak_tf_1.shape)
					print('data loaded from %s'%(filename_motif_score_1))
					print('preview:\n',df_peak_tf_1[0:2])
					print('scaling scores for TF binding prediction performed by Unify, dataframe of size ',df_peak_tf_2.shape)
					print('data loaded from %s'%(filename_motif_score_2))
					print('preview:\n',df_peak_tf_2[0:2])
				self.df_peak_tf_1 = df_peak_tf_1
				self.df_peak_tf_2 = df_peak_tf_2

			# compute p-value of gene-TF expression partial correlation given peak accessibility
			column_pval_cond = select_config['column_pval_cond']
			print('column_pval_cond ',column_pval_cond)
			flag_gene_tf_corr_peak_pval = 0
			if not (column_pval_cond in df_gene_tf_corr_peak.columns):
				flag_gene_tf_corr_peak_pval = 1

			if flag_gene_tf_corr_peak_pval>0:
				print('df_gene_tf_corr_peak ',df_gene_tf_corr_peak.shape)
				print('estimate p-value corrected for gene-TF expression partial correlation given peak accessibility')
				start = time.time()
				df_gene_tf_corr_peak = self.test_gene_tf_corr_peak_pval_corrected_query_1(df_gene_peak_query=df_gene_tf_corr_peak,
																							gene_query_vec=[],
																							motif_query_vec=[],
																							parallel_mode=parallel_mode,
																							verbose=verbose,
																							select_config=select_config)
				stop = time.time()
				print('estimating p-value corrected for gene-TF expression partial correlation used %.5fs'%(stop-start))
				if (save_mode>0) and (output_filename!=''):
					df_gene_tf_corr_peak.index = np.asarray(df_gene_tf_corr_peak['gene_id'])
					compression = None
					b = output_filename.find('.gz')
					if b>-1:
						compression = 'gzip'
					df_gene_tf_corr_peak.to_csv(output_filename,sep='\t',float_format='%.5f',compression=compression)

			# calculate peak-TF-gene association scores
			df_gene_peak_query1_1 = self.test_gene_peak_tf_query_score_init_pre1(df_gene_peak_query=df_gene_peak_query,
																					df_gene_tf_corr_peak=df_gene_tf_corr_peak,
																					lambda1=lambda1,lambda2=lambda2,
																					select_config=select_config)

			field_query_3 = ['motif_id','peak_id','gene_id','score_1','score_combine_1','score_pred1_correlation','score_pred1','score_pred1_1','score_pred2','score_pred_combine']
			if 'column_score_query_pre1' in select_config:
				field_query_3 = select_config['column_score_query_pre1']
			df_gene_peak_query1_1 = df_gene_peak_query1_1.loc[:,field_query_3]

		if flag_save_interval>0:
			if 'filename_prefix_save'=='':
				filename_prefix_save = select_config['filename_prefix_save_pre1']
			save_mode_2 = flag_save_ori
			self.test_gene_peak_tf_query_score_init_save(df_gene_peak_query=df_gene_peak_query1_1,
															lambda1=lambda1,lambda2=lambda2,
															flag_init_score=0,
															flag_save_interval=flag_save_interval,
															feature_type='gene_id',
															query_mode=0,
															save_mode=save_mode_2,
															output_file_path='',
															output_filename='',
															filename_prefix_save=filename_prefix_save,
															float_format='%.5f',
															select_config=select_config)
		return df_gene_peak_query1_1

	## ====================================================
	# compute gene expression correlation
	def test_gene_expr_correlation_1(self,feature_vec_1=[],feature_vec_2=[],rna_exprs=[],correlation_type_vec=['spearmanr'],
										symmetry_mode=0,type_id_1=0,type_id_pval_correction=1,thresh_corr_vec=[],
										save_mode=1,save_symmetry=0,output_file_path='',filename_prefix='',verbose=0,select_config={}):

		"""
		compute gene expression correlation for the given sets of genes
		:param feature_vec_1: (array or list) the first set of genes
		:param feature_vec_2: (array or list) the second set of genes
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param correlation_type_vec: (list) the type of correlation (e.g. Spearman's rank correlation or Pearson correlation)
		:param symmetry_mode: indicator of whether the first and second sets of genes between which we compute the correlations are the same (0:the two sets of genes are different;1:the two sets are identical)
		:param type_id_1: indicator of whether to exclude each observation itself in computing adjusted p-values of the correlations (0:exclude the observation; 1:use all the observations)
		:param type_id_pval_correction: indicator of whether to estimate the adjusted p-value
		:param thresh_corr_vec: (list) thresholds on correlation and the adjusted p-value
		:param save_mode: indicator of whether to save data
		:param save_symmetry: indicator of whether to only save the diagnoal and upper right half of the correlation matrix (if the matrix is symmetric) or save the full correlation matrix 
							  (1:save partial matrix; 0:save full matrix)
		:param output_file_path: the directory to save data
		:param filename_prefix: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictonary containing the correlation matrix, the raw p-value matrix, and the adjusted p-value matrix for each correlation type (specified in correlation_type_vec)
		"""

		if len(rna_exprs)==0:
			rna_exprs = self.meta_scaled_exprs
		df_feature_query_1 = rna_exprs

		df_feature_query2 = []
		if len(feature_vec_1)>0:
			df_feature_query1 = df_feature_query_1.loc[:,feature_vec_1]
		else:
			df_feature_query1 = df_feature_query_1
			feature_vec_1 = df_feature_query_1.columns

		if len(feature_vec_2)>0:
			if list(np.unique(feature_vec_2))!=list(np.unique(feature_vec_1)):
				df_feature_query2 = df_feature_query_1.loc[:,feature_vec_2]
				symmetry_mode=0
			else:
				# if the second set of genes are not specified, we compute correlation within the first set of genes
				df_feature_query2 = []
				symmetry_mode=1

		print('df_feature_query1, df_feature_query2, symmetry_mode ',df_feature_query1.shape,len(df_feature_query2),symmetry_mode)
		if verbose>0:
			print('compute gene-TF expression correlation for %d genes'%(gene_num))
		
		start = time.time()
		feature_num1 = len(feature_vec_1)
		gene_num = feature_num1

		file_path1 = self.save_path_1
		test_estimator_correlation = _Base2_correlation(file_path=file_path1)
		self.test_estimator_correlation = test_estimator_correlation
		dict_query_1 = test_estimator_correlation.test_feature_correlation_1(df_feature_query_1=df_feature_query1,
																				df_feature_query_2=df_feature_query2,
																				feature_vec_1=feature_vec_1,
																				feature_vec_2=feature_vec_2,
																				correlation_type_vec=correlation_type_vec,
																				symmetry_mode=symmetry_mode,
																				type_id_pval_correction=type_id_pval_correction,
																				type_id_1=type_id_1,
																				thresh_corr_vec=thresh_corr_vec,
																				save_mode=save_mode,
																				save_symmetry=save_symmetry,
																				output_file_path=output_file_path,
																				filename_prefix=filename_prefix,
																				select_config=select_config)

		stop = time.time()
		if verbose>0:
			print('computing gene-TF expression correlation for %d genes used %.5fs'%(gene_num,stop-start))
			
		return dict_query_1

	## ====================================================
	# initiate peak-TF-gene links and compute gene expression-TF expression partial correlation conditioned on peak accessibility
	def test_partial_correlation_gene_tf_cond_peak_1(self,motif_data=[],gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],
														df_gene_peak_query=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],
														type_id_query=2,type_id_compute=1,parallel_mode=0,
														save_mode=1,output_filename='',verbose=0,select_config={}):

		"""
		initiate peak-TF-gene links and compute TF expression-gene expression partial correlation conditioned on peak accessibility
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param gene_query_vec: (array or list) the target genes
		:param peak_query_vec: (array) the ATAC-seq peak loci for which to estimate peak-TF-gene links
		:param motif_query_vec: (array) TF names
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) scaled gene expressions (z-scores) of the metacells (log-transformed normalized count matrix with standard scaling) (row:metacell, column:gene)
		:param rna_exprs_unscaled: (dataframe) unscaled gene expressions of the metacells (log-transformed normalized count matrix)
		:param type_id_query: indicator of which metacells to include to compute gene-TF expression partial correlation given peak accessibility for the given peak-TF-gene link:
							  1: all the metacells;
							  2: metacells with (i) the peak accessibility above zero or (ii) the peak without accessibility and the target gene not expressed;
							  3: metacells with the peak accessibility above zero or the TF expression below threshold;
		:param type_id_compute: indicator of whether to compute adjusted p-value for the gene-TF expression partial correlation conditioned on the peak accessibility using peak-TF-gene links associated with each target gene;
		:param parallel_mode: indicator of whether to perform computation in parallel
		:param save_mode: indicator of whether to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-TF-gene links including gene-TF expression partial correlation conditioned on peak accessibility
		"""

		input_file_path1 = self.save_path_1
		list_pre1 = []
		motif_query_name = motif_data.columns
		if len(motif_query_vec)>0:
			motif_data = motif_data.loc[:,motif_query_vec]
		else:
			motif_query_vec = motif_data.columns

		# column_idvec = ['motif_id','peak_id','gene_id']
		column_idvec = select_config['column_idvec']
		column_id3, column_id2, column_id1 = column_idvec
		
		if len(peak_query_vec)==0:
			peak_query_vec = df_gene_peak_query[column_id2].unique()

		if len(gene_query_vec)==0:
			gene_query_vec = df_gene_peak_query[column_id1].unique()
		
		gene_query_num = len(gene_query_vec)
		peak_query_num = len(peak_query_vec)
		list_query1 = []
		list_query2 = []
		df_query1 = []
		thresh1, thresh2 = 0, 1E-05
		df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])
		sample_id = rna_exprs.index
		sample_num = len(sample_id)
		peak_read = peak_read.loc[sample_id,:]
		rna_exprs_unscaled = rna_exprs_unscaled.loc[sample_id,:]
		df_peak_annot1 = pd.DataFrame(index=peak_query_vec,columns=['motif_num'])
		dict_motif = dict()
		motif_query_num = len(motif_query_vec)
		if type_id_query in [3]:
			for i1 in range(motif_query_num):
				motif_query1 = motif_query_vec[i1]
				tf_expr = rna_exprs_unscaled[motif_query1]
				id_tf = (tf_expr>thresh1)
				sample_id2 = sample_id[id_tf]
				dict_motif.update({motif_query1:id_tf})
				if (verbose>0) and (i1%100==0):
					# print('motif_query1 ',motif_query1,i1,len(id_tf))
					print('TF %s with expression above threshold in %d metacells'%(motif_query1,len(id_tf)),i1)

		# field_query = ['gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1']
		field_query = select_config['column_gene_tf_corr_peak']
		field_id1, field_id2, field_id3 = field_query
		
		print('field_id1:%s; field_id2:%s; field_id3:%s'%(field_id1,field_id2,field_id3))
		flag_1 = 0
		# field_query = ['motif_id','peak_id','gene_id']
		field_query = column_idvec

		warnings.filterwarnings('ignore')
		for i1 in range(gene_query_num):
			gene_query_id = gene_query_vec[i1]
			# peak_loc_query = df_gene_peak_query.loc[[gene_query_id],'peak_id'].unique() # candidate peaks linked with gene;
			peak_loc_query = df_gene_peak_query.loc[[gene_query_id],column_id2].unique() # candidate peaks linked with gene;
			peak_num2 = len(peak_loc_query)
			Y_expr = rna_exprs[gene_query_id] # gene expr scaled
			Y_expr_unscaled = rna_exprs_unscaled[gene_query_id] # gene expr unscaled
			id_gene_expr = (Y_expr_unscaled>thresh1) # gene expr above zero
			if verbose>0:
				print('gene_id:%s, peak_loc:%d, %d'%(gene_query_id,peak_num2,i1))
			
			for i2 in range(peak_num2):
				peak_id = peak_loc_query[i2]
				motif_idvec = motif_query_name[motif_data.loc[peak_id,:]>0] # TFs with motifs in the peak
				motif_num = len(motif_idvec)
				df_peak_annot1.loc[peak_id,'motif_num'] = motif_num
				# thresh1, thresh2 = 0, 1E-05
				Y_peak = peak_read[peak_id]
				id_peak_1 = (Y_peak>thresh1) # the samples with peak accessibility above the threshold
				id_peak = sample_id[id_peak_1]	# the samples with peak accessibility above the threshold
				sample_query_peak = id_peak
				if motif_num>0:
					df_pre1 = pd.DataFrame(index=motif_idvec,columns=field_query,dtype=np.float32)
					df_pre1[column_id3] = motif_idvec
					df_pre1[column_id2] = peak_id
					df_pre1[column_id1] = gene_query_id

					X = rna_exprs.loc[:,motif_idvec]
					gene_tf_corr_peak_pval_2 = []

					# the gene query may be the same as the motif query
					if gene_query_id in motif_idvec:
						gene_query_id_2 = '%s.1'%(gene_query_id)
					else:
						gene_query_id_2 = gene_query_id

					df_pre2 = []
					try:
						if type_id_query in [1,2]:
							if type_id_query==1:
								sample_id_query = sample_id
							else:
								id_peak_2 = (id_peak_1)|(~id_gene_expr) # peak with accessibility above threshold or gene with expression below threshold
								sample_id_query = sample_id[id_peak_2]
							
							ratio_cond = len(sample_id_query)/sample_num
							ratio_1 = len(id_peak)/sample_num
							df_pre1['ratio_1'] = [ratio_1]*motif_num
							df_pre1['ratio_cond1'] = [ratio_cond]*motif_num
							if len(sample_id_query)<=2:
								sample_id_query = list(sample_id_query)*3

							mtx_1 = np.hstack((np.asarray(X.loc[sample_id_query,:]),np.asarray(Y_peak[sample_id_query])[:,np.newaxis],np.asarray(Y_expr[sample_id_query])[:,np.newaxis]))
							df_query1 = pd.DataFrame(index=sample_id_query,columns=list(motif_idvec)+[peak_id]+[gene_query_id_2],data=mtx_1,dtype=np.float32)
							# if (i2%100==0) and (i1%100==0):
								# print('df_query1: ',df_query1.shape,i1,i2,len(sample_id_query),len(sample_query_peak),len(motif_idvec),peak_id,gene_query_id_2)
								# print('df_query1: ',df_query1.shape,len(sample_query_peak),len(motif_idvec),peak_id,gene_query_id_2)
							
							# flag_query_2 = 1
							flag_query_2 = 0
							if flag_query_2>0:
								t_value_1 = df_query1.max(axis=0)-df_query1.min(axis=0)
								column_vec_1 = np.asarray(df_query1.columns)
								t_value_1 = np.asarray(t_value_1)
								feature_id2 = column_vec_1[t_value_1<1E-07]

								if len(feature_id2)>0:
									# print('constant value in the vector ',gene_query_id_2,i1,peak_id,i2,feature_id2)
									if gene_query_id_2 in feature_id2:
										print('gene expression is constant in the subsample: %s, %s, %d, %s, %d'%(feature_id2,gene_query_id_2,i1,peak_id,i2))
										# gene_expr_1 = Y_expr[sample_id_query]
										# print(gene_expr_1)
										# continue
									if peak_id in feature_id2:
										print('peak read value is constant in the subsample: %s, %s, %d, %s, %d'%(feature_id2,gene_query_id_2,i1,peak_id,i2))
									
									motif_idvec_2 = pd.Index(motif_idvec).intersection(feature_id2,sort=False)
									if len(motif_idvec_2)>0:
										print('TF expression is constant in the subsample ',len(motif_idvec_2),motif_idvec_2)
										motif_idvec_ori = motif_idvec.copy()
										motif_idvec = pd.Index(motif_idvec).difference(feature_id2,sort=False)
										column_vec = list(motif_idvec)+[peak_id]+[gene_query_id_2]
										df_query1 = df_query1.loc[:,column_vec]
										motif_num = len(motif_idvec)
										print('gene_id: %s, %d, peak_id: %s, %d, motif_idvec: %d'%(gene_query_id_2,i1,peak_id,i2,motif_num))
							
							if (i1%100==0) and (i2%100==0):
								print('gene_id: %s, %d, peak_id: %s, %d, motif_idvec: %d'%(gene_query_id_2,i1,peak_id,i2,motif_num))
								print('df_query1, number of metacells where peak is open: ',df_query1.shape,len(sample_query_peak))
							
							if type_id_compute==0:
								# only estimate raw p-value
								t_vec_1 = [pg.partial_corr(data=df_query1,x=motif_query1,y=gene_query_id_2,covar=peak_id,alternative='two-sided',method='spearman') for motif_query1 in motif_idvec]
								gene_tf_corr_peak_1 = [t_value_1['r'] for t_value_1 in t_vec_1]
								gene_tf_corr_peak_pval_1 = [t_value_1['p-val'] for t_value_1 in t_vec_1]
								df_pre2 = pd.DataFrame(index=motif_idvec)
								df_pre2[field_id1] = np.asarray(gene_tf_corr_peak_1)
								df_pre2[field_id2] = np.asarray(gene_tf_corr_peak_pval_1)
							else:
								# p-value correction for TF motifs in the same peak
								df1 = pg.pairwise_corr(data=df_query1,columns=[[gene_query_id_2],list(motif_idvec)],covar=peak_id,alternative='two-sided',method='spearman',padjust='fdr_bh')
								df1.index = np.asarray(df1['Y'])
								if verbose>0:
									print('df1, gene_query_id_2, peak_id, motif_idvec ',df1.shape,gene_query_id_2,i1,peak_id,i2,motif_num)
									print(df1)
								if 'p-corr' in df1:
									df_pre2 = df1.loc[:,['r','p-unc','p-corr']]
								else:
									df_pre2 = df1.loc[:,['r','p-unc']]
								df_pre2 = df_pre2.rename(columns={'r':field_id1,'p-unc':field_id2,'p-corr':field_id3})
						else:
							gene_tf_corr_peak_1, gene_tf_corr_peak_pval_1 = [], []
							for l2 in range(motif_num):
								motif_query1 = motif_idvec[l2]
								id_tf = dict_motif[motif_query1] # TF with expression
								# sample_id_2 = sample_id[~((id_tf)&(~id_peak_2))]
								# sample_id_query = sample_id[(~id_tf)|id_peak_2]
								sample_id_query = sample_id[(~id_tf)|id_peak_1]
								ratio_cond  = len(sample_id_query)/sample_num
								df_pre1.loc[motif_query1,'ratio_cond2'] = ratio_cond
								if len(sample_id_query)<=2:
									sample_id_query = list(sample_id_query)*3

								mtx_2 = np.hstack((np.asarray(X.loc[sample_id_query,[motif_query1]]),np.asarray(Y_peak[sample_id_query])[:,np.newaxis],np.asarray(Y_expr[sample_id_query])[:,np.newaxis]))
								df_query2 = pd.DataFrame(index=sample_id_query,columns=[motif_query1]+[peak_id]+[gene_query_id_2],data=mtx_2,dtype=np.float32)
								if i2%100==0:
									print('df_query2: ',df_query2.shape,i1,i2,len(sample_id_query),len(sample_query_peak),motif_query1,peak_id,gene_query_id_2)
								t_value_1 = pg.partial_corr(data=df_query2,x=motif_query1,y=gene_query_id_2,covar=peak_id,alternative='two-sided',method='spearman')
								gene_tf_corr_peak_1.append(t_value_1['r'])
								gene_tf_corr_peak_pval_1.append(t_value_1['p-val'])
							
							df_pre2 = pd.DataFrame(index=motif_idvec)
							df_pre2[field_id1] = np.asarray(gene_tf_corr_peak_1)
							df_pre2[field_id2] = np.asarray(gene_tf_corr_peak_pval_1)
					except Exception as error:
						if verbose>0:
							print('error! ',error)
							# print('gene_id, peak_id, motif_idvec, df_query1',gene_query_id_2,i1,peak_id,i2,motif_num,motif_idvec,df_query1.shape)
						# return
						continue

					motif_idvec_1 = df_pre2.index
					df_pre1.loc[motif_idvec_1,[field_id1,field_id2]] = df_pre2.loc[motif_idvec_1,[field_id1,field_id2]]
					if field_id3 in df_pre2.columns:
						df_pre1.loc[motif_idvec_1,field_id3] = df_pre2.loc[motif_idvec_1,field_id3]
					if save_mode>0:
						list_query1.append(df_pre1)
					list_query2.append(df_pre1)

			interval = 100
			if (save_mode>0) and (i1%interval==0) and (len(list_query1)>0):
				df_query1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
				if (i1==0) or (flag_1==0):
					df_query1.to_csv(output_filename,sep='\t',float_format='%.5f')
					flag_1=1
				else:
					df_query1.to_csv(output_filename,header=False,mode='a',sep='\t',float_format='%.5f')
				list_query1 = []

		warnings.filterwarnings('default')
		load_mode_2 = 1
		if load_mode_2>0:
			df_query1 = pd.concat(list_query2,axis=0,join='outer',ignore_index=False)
			df_query1.to_csv(output_filename,sep='\t',float_format='%.5f')
				
		return df_query1

	## ====================================================
	# compute adjusted p-value for gene-TF expression partial correlation given peak accessibility
	def test_gene_tf_corr_peak_pval_corrected_query_1(self,df_gene_peak_query=[],gene_query_vec=[],motif_query_vec=[],alpha=0.05,
														method_type_correction='fdr_bh',type_id_1=1,parallel_mode=0,save_mode=1,verbose=0,select_config={}):

		"""
		compute adjusted p-value for gene-TF expression partial correlation given peak accessibility for peak-TF-gene links associated with each gene or TF
		:param df_gene_peak_query: (dataframe) annotations of peak-TF-gene links including gene-TF expression partial correlation given peak accessibility
		:param gene_query_vec: (array) the target genes
		:param motif_query_vec: (array) TF names
		:param alpha: (float) family-wise error rate used in p-value correction for multiple tests
		:param method_type_correction: the method used for p-value correction
		:param type_id_1: indicator of which type of p-value correction to perform:
						  0: p-value correction for gene-TF expression partial correlation given peak accessibility of peak-TF-gene links associated with each gene
						  1: p-value correction for gene-TF expression partial correlation given peak accessibility of peak-TF-gene links associated with each TF
		:param parallel_mode: indicator of whether to perform computation in parallel
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-TF-gene links including adjusted p-values for the gene-TF expression partial correlation given peak accessibility
		"""

		flag_query1=1
		if len(df_gene_peak_query)>0:
			df_gene_peak_query_1 = df_gene_peak_query
			column_idvec = ['motif_id','peak_id','gene_id']
			# column_idvec = select_config['column_idvec']
			column_id3, column_id2, column_id1 = column_idvec
			# field_query = ['gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1']
			field_query = select_config['column_gene_tf_corr_peak']
			field_id1, field_id2, field_id3 = field_query
			field_id_query = '%s_corrected2'%(field_id2)
			print('compute adjusted p-value of gene-TF expression partial correlation given peak accessibility for peak-TF-gene links')
			if verbose>0:
				# print(column_idvec,field_query)
				print('columns for TF, peak, and gene indices: ',np.asarray(column_idvec))
				print('columns for the partial correlation, raw p-value, and adjusted p-value for peak-TF-gene links',field_query)

			if len(gene_query_vec)==0:
				gene_query_vec = df_gene_peak_query_1['gene_id'].unique()
			gene_query_num = len(gene_query_vec)

			if len(motif_query_vec)==0:
				motif_query_vec = df_gene_peak_query_1['motif_id'].unique()
			motif_query_num = len(motif_query_vec)
			
			df_gene_peak_query_1.index = test_query_index(df_gene_peak_query_1,column_vec=['motif_id','peak_id','gene_id'])
			query_id_1 = df_gene_peak_query_1.index
			
			# p-value correction for gene-TF expression partial correlation given peak accessibility
			column_id_pre1 = 'gene_tf_corr_peak_pval_corrected1'
			if flag_query1>0:
				if verbose>0:
					print('p-value correction')
				
				start = time.time()
				column_id_1 = field_id2
				if type_id_1==0:
					# compute adjusted p-value of gene-TF expression partial correlation given peak accessibility for peak-TF-gene links associated with each gene
					column_id_query = column_id1
					feature_query_vec = gene_query_vec
					# field_id_query = '%s_corrected1'%(field_id2)
				else:
					# compute adjusted p-value of gene-TF expression partial correlation given peak accessibility for peak-TF-gene links associated with each gene
					column_id_query = column_id3
					feature_query_vec = motif_query_vec
					# field_id_query = '%s_corrected2'%(field_id2)

				df_pval_query1 = df_gene_peak_query_1.loc[:,[column_id_query,column_id_1]]
				df_pval_query1 = df_pval_query1.fillna(1) # p-value
				feature_query_num = len(feature_query_vec)

				alpha=0.05
				method_type_correction='fdr_bh'
				field_query_2 = [field_id2,field_id_query]
				interval_1 = 100
				if parallel_mode==0:
					df_gene_peak_query_1 = self.test_gene_tf_corr_peak_pval_corrected_unit1(data=df_gene_peak_query_1,
																							feature_query_vec=feature_query_vec,
																							feature_id='',
																							column_id_query=column_id_query,
																							field_query=field_query_2,
																							alpha=alpha,
																							method_type_correction=method_type_correction,
																							type_id_1=type_id_1,
																							interval=interval_1,
																							save_mode=1,verbose=verbose,select_config=select_config)
				else:
					query_res_local = Parallel(n_jobs=-1)(delayed(self.test_gene_tf_corr_peak_pval_corrected_unit1)(data=df_pval_query1,
																													feature_query_vec=feature_query_vec[i2:(i2+1)],
																													column_id_query=column_id_query,
																													field_query=field_query_2,
																													alpha=alpha,
																													method_type_correction=method_type_correction,
																													interval=interval_1,
																													save_mode=1,verbose=(i2%interval_1),select_config=select_config) for i2 in tqdm(np.arange(feature_query_num)))

					for t_query_res in query_res_local:
						# dict_query = t_query_res
						if len(t_query_res)>0:
							df_query = t_query_res
							query_id1 = df_query.index
							df_gene_peak_query_1.loc[query_id1,field_id_query] = df_query.loc[query_id1,field_id_query]

				stop = time.time()
				print('p-value correction ',stop-start)

		print('peak-TF-gene links, dataframe of size ',df_gene_peak_query_1.shape)
		print('data preview: \n',df_gene_peak_query_1[0:5])
		return df_gene_peak_query_1

	## ====================================================
	# compute adjusted p-value for gene-TF expression partial correlation given peak accessibility
	def test_gene_tf_corr_peak_pval_corrected_unit1(self,data=[],feature_query_vec=[],feature_id='',column_id_query='',field_query=[],alpha=0.05,method_type_correction='fdr_bh',type_id_1=0,interval=1000,save_mode=1,verbose=0,select_config={}):

		"""
		compute adjusted p-value for gene-TF expression partial correlation given peak accessibility for peak-TF-gene links associated with each gene or TF
		:param data: (dataframe) annotations of peak-TF-gene links including gene-TF expression partial correlation given peak accessibility
		:param feature_query_vec: (array) the target gene names or TF names for which we compute adjusted p-value of gene-TF expression partial correlation given peak accessibility of the associated peak-TF-gene links
		:param feature_id: (str) the specific target gene name or TF name for which we compute adjusted p-value of gene-TF expression partial correlation given peak accessibility of the associated peak-TF-gene links
		:param column_id_query: (str) column indicating the gene index or TF index
		:param field_query: (list) 1. column for the gene-TF expression partial correlation given peak accessibility in the peak-TF-gene link annotation dataframe;
								   2. column for the adjusted p-value of gene-TF expression partial correlation given peak accessibility in the dataframe;
		:param alpha: (float) family-wise error rate used in p-value correction for multiple tests
		:param method_type_correction: (str) the method used for p-value correction
		:param type_id_1: indicator of which type of p-value correction to perform:
						  0: p-value correction for gene-TF expression partial correlation given peak accessibility of peak-TF-gene links associated with each gene
						  1: p-value correction for gene-TF expression partial correlation given peak accessibility of peak-TF-gene links associated with each TF
		:param interval: interval of the TF number or gene number used in printing the intermediate information
		:param parallel_mode: indicator of whether to perform computation in parallel
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-TF-gene links including adjusted p-values for the gene-TF expression partial correlation given peak accessibility
		"""

		df_query_1 = data
		feature_query_num = len(feature_query_vec)
		list1 = []
		field_id_query1 = 'gene_tf_corr_peak_pval_corrected1'
		for i2 in range(feature_query_num):
			feature_id1 = feature_query_vec[i2]
			query_id1 = (df_query_1[column_id_query]==feature_id1)
			field_id, field_id_query = field_query[0:2]
			# pvals = np.asarray(df_query_1.loc[query_id1,field_id])
			pvals = df_query_1.loc[query_id1,field_id]
			pvals_correction_vec1, pval_thresh1 = utility_1.test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_correction)
			id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
							
			if type_id_1==1:
				df_query_1.loc[query_id1,field_id_query] = pvals_corrected1
			else:
				df_query1 = df_query_1.loc[query_id1,[field_id]]
				query_vec_1 = df_query1.index

				if field_id_query1 in df_query_1.columns:
					type_query = 1
					column_vec = [field_id,field_id_query1,field_id_query]
				else:
					type_query = 0
					column_vec = [field_id,field_id_query]

				df_query2 = pd.DataFrame(index=query_vec_1,columns=column_vec,dtype=np.float32)
				df_query2[field_id] = pvals
				if type_query>0:
					df_query2[field_id_query1] =  df_query_1.loc[query_id1,field_id_query1]
				df_query2[field_id_query] =  pvals_corrected1
				list1.append(df_query2)
			
			if (verbose>0) and (i2%interval==0):
				query_num1 = len(pvals_corrected1)
				print('feature_id, pvals_corrected ',feature_id1,i2,query_num1,np.max(pvals_corrected1),np.min(pvals_corrected1),np.mean(pvals_corrected1),np.median(pvals_corrected1))
				if type_id_1==1:
					print(df_query_1.loc[query_id1,:])
				else:
					print(df_query2)

		if (type_id_1==0):
			if (feature_query_num>0):
				df_query_2 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
			else:
				df_query_2 = list1[0]
		else:
			df_query_2 = df_query_1

		return df_query_2

	## ==================================================================
	# compute the components for score function 1 (peak-TF-gene association score 1)
	def test_query_score_function1(self,df_peak_tf_corr=[],df_gene_peak_query=[],gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],
										motif_data=[],motif_data_score=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],
										type_id_query=2,type_id_compute=1,flag_peak_tf_corr=0,flag_gene_tf_corr_peak=1,
										flag_pval_1=1,flag_pval_2=0,flag_load=1,field_load=[],parallel_mode=1,input_file_path='',
										save_mode=1,output_file_path='',output_filename='',filename_prefix='',filename_annot='',verbose=0,select_config={}):

		"""
		compute the components for score function 1 (peak-TF-gene association score 1, which is TF-(peak,gene) score or TF binding score)
		:param df_peak_tf_corr: (dataframe) peak accessibility-TF expression correlations for peak-TF links
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param gene_query_vec: (array or list) the target genes
		:param peak_query_vec: (array) the ATAC-seq peak loci for which to estimate peak-TF-gene links
		:param motif_query_vec: (array) TF names
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param motif_data_score: (dataframe) motif scores from motif scanning results (row:ATAC-seq peak locus, column:TF)
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) scaled gene expressions (z-scores) of the metacells (log-transformed normalized count matrix with standard scaling) (row:metacell, column:gene)
		:param rna_exprs_unscaled: (dataframe) unscaled gene expressions of the metacells (log-transformed normalized count matrix)
		:param type_id_query: indicator of which metacells to include to compute gene-TF expression partial correlation given peak accessibility for the given peak-TF-gene link:
							  1: all the metacells;
							  2: metacells with (i) the peak accessibility above zero or (ii) the peak without accessibility and the target gene not expressed;
							  3: metacells with the peak accessibility above zero or the TF expression below threshold;
		:param type_id_compute: indicator of whether to compute adjusted p-value for the gene-TF expression partial correlation conditioned on the peak accessibility using peak-TF-gene links associated with each target gene;
		:param flag_peak_tf_corr: indicator of whether to compute peak accessibility-TF expression correlation and p-value
		:param flag_gene_tf_corr_peak: indicator of whether to compute gene-TF expression partial correlation conditioned on the peak accessibility
		:param flag_pval_1: indicator of whether to compute p-values of correlations
		:param flag_pval_2: indicator of whether to compute the adjusted p-value for gene-TF expression partial correlation conditioned on the peak accessibility
		:param flag_load: indicator of whehter to load peak accessibility-TF expression correlations and p-values from the saved files
		:param field_load: (array or list) fields representing correlation, the raw p-value, and the adjusted p-value that are used in the corresponding filenames and used for retrieving data
		:param parallel_mode: indicator of whether to perform computation in parallel
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) peak-TF-gene link annotations with computed gene-TF expression partial correlation conditioned peak accessibility		
		"""

		if flag_peak_tf_corr in [1,2]:
			## estimate peak-TF expresssion correlation
			print('estimate peak-TF expression correlation')
			start = time.time()
			correlation_type = select_config['correlation_type']
			if len(field_load)==0:
				field_load = [correlation_type,'pval','pval_corrected']
			dict_query = self.test_peak_tf_correlation_query_1(motif_data=motif_data,peak_query_vec=[],
																motif_query_vec=[],
																peak_read=peak_read,
																rna_exprs=rna_exprs,
																correlation_type=correlation_type,
																flag_load=flag_load,field_load=field_load,
																input_file_path=input_file_path,
																input_filename_list=[],
																save_mode=save_mode,
																output_file_path=output_file_path,
																filename_prefix=filename_prefix,
																select_config=select_config)
			stop = time.time()
			print('estimate peak-TF expression correlation used %.5fs'%(stop-start))

		if flag_peak_tf_corr in [2]:
			df_peak_annot = self.peak_annot
			# compute the regularization scores of peak-TF links
			df_motif_score_query = self.test_peak_tf_score_normalization_pre_compute(peak_query_vec=[],
																						motif_query_vec=[],
																						motif_data=motif_data,
																						motif_data_score=motif_data_score,
																						peak_read=peak_read,
																						rna_exprs=rna_exprs,
																						peak_read_celltype=[],
																						df_peak_annot=df_peak_annot,
																						output_file_path=output_file_path,
																						filename_annot=filename_annot,
																						select_config=select_config)

		flag_gene_tf_corr_peak_1 = flag_gene_tf_corr_peak
		flag_gene_tf_corr_peak_pval=flag_pval_2
		if 'flag_gene_tf_corr_peak_pval' in select_config:
			flag_gene_tf_corr_peak_pval = select_config['flag_gene_tf_corr_peak_pval']
		df_gene_peak = df_gene_peak_query
		if len(motif_query_vec)==0:
			# motif_query_vec = df_gene_peak_query['motif_id'].unique()
			motif_query_name_expr = self.motif_query_name_expr
			motif_query_vec = motif_query_name_expr
		
		# initiate peak-TF-gene links and compute gene-TF expresssion partial correlation given peak accessibility
		if flag_gene_tf_corr_peak_1>0:
			print('compute gene-TF expression partial correlation given peak accessibility')
			start = time.time()
			peak_query_vec, motif_query_vec = [], []
			type_id_query_2 = 2
			type_id_compute = 1
			gene_query_num = len(gene_query_vec)
			df_gene_tf_corr_peak = self.test_partial_correlation_gene_tf_cond_peak_1(motif_data=motif_data,
																						gene_query_vec=gene_query_vec,
																						peak_query_vec=peak_query_vec,
																						motif_query_vec=motif_query_vec,
																						df_gene_peak_query=df_gene_peak,
																						peak_read=peak_read,
																						rna_exprs=rna_exprs,
																						rna_exprs_unscaled=rna_exprs_unscaled,
																						type_id_query=type_id_query,
																						type_id_compute=type_id_compute,
																						parallel_mode=parallel_mode,
																						save_mode=save_mode,
																						output_filename=output_filename,
																						verbose=verbose,
																						select_config=select_config)
			stop = time.time()
			print('computing gene-TF expression partial correlation given peak accessibility for %d genes used %.5fs'%(gene_query_num,stop-start))

		# compute adjusted p-value of gene-TF expresssion partial correlation given peak accessibility
		if flag_gene_tf_corr_peak_pval>0:
			print('the candidate peak-TF-gene links, dataframe of size ',df_gene_tf_corr_peak.shape)
			print('estimate p-value corrected for gene-TF expression partial correlation given peak accessibility')
			start = time.time()
			# parallel_mode = 1
			if 'parallel_mode_pval_correction' in select_config:
				parallel_mode = select_config['parallel_mode_pval_correction']
			df_gene_tf_corr_peak = self.test_gene_tf_corr_peak_pval_corrected_query_1(df_gene_peak_query=df_gene_tf_corr_peak,
																						gene_query_vec=[],
																						motif_query_vec=motif_query_vec,
																						parallel_mode=parallel_mode,
																						verbose=verbose,
																						select_config=select_config)
			stop = time.time()
			print('estimating p-value corrected for gene-TF expression partial correlation used %.5fs'%(stop-start))
			verbose_internal = self.verbose_internal
			if verbose_internal==2:
				print('data preview:\n',df_gene_tf_corr_peak[0:2])
			if (save_mode>0) and (output_filename!=''):
				df_gene_tf_corr_peak.to_csv(output_filename,sep='\t',float_format='%.5f')

		df_gene_tf_corr_peak.index = np.asarray(df_gene_tf_corr_peak['gene_id'])

		return df_gene_tf_corr_peak

	## ====================================================
	# combine gene expression-TF expression partial correlations conditioned on peak accessibility of different subsets of peak-TF-gene links
	def test_partial_correlation_gene_tf_cond_peak_combine_pre1(self,input_file_path='',type_id_query=0,type_id_compute=0,overwrite=0,
																	save_mode=1,output_file_path='',output_filename='',output_filename_list=[],
																	filename_prefix_vec=[],filename_prefix_save='',verbose=0,select_config={}):

		"""
		combine gene expression-TF expression partial correlations conditioned on peak accessibility of different subsets of peak-TF-gene links
		:param input_file_path: the directory to retrieve data from
		:param type_id_query: indicator of which metacells to include to compute gene-TF expression partial correlation given peak accessibility for a peak-TF-gene link:
							  1: all the metacells;
							  2: metacells with (i) the peak accessibility above zero or (ii) the peak without accessibility and the target gene not expressed;
							  3: metacells with the peak accessibility above zero or the TF expression below threshold;
		:param type_id_compute: indicator of whether to compute adjusted p-value for the gene-TF expression partial correlation conditioned on the peak accessibility using peak-TF-gene links associated with each target gene;
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param output_filename_list: (list) filenames to save data
		:param filename_prefix_vec: (array or list) prefix or annotation used in potential filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) peak-TF-gene link annotations including gene-TF expression partial correlation given peak accessibility;
				 2. (dataframe) percentage of metacells used in computing gene-TF expression partial correlation given peak accessibility for each peak-gene link or peak-TF-gene link;
		"""

		print('combine gene-TF expresssion partial correlations given peak accessibility of different subsets of peak-TF-gene links')
		start = time.time()

		if len(filename_prefix_vec)==0:
			filename_prefix_save_pre1 = '%s.pcorr_query1.%d.%d'%(filename_prefix_save,type_id_query,type_id_compute)
			filename_prefix_vec = [filename_prefix_save_pre1]
		else:
			filename_prefix_save_pre1 = filename_prefix_vec[0]

		idvec = self.idvec_compute_2
		interval = self.interval_compute_2
		df1, df1_ratio = self.test_partial_correlation_gene_tf_cond_peak_combine_1(input_file_path=input_file_path,
																					idvec=idvec,interval=interval,
																					save_mode=save_mode,
																					filename_prefix_vec=filename_prefix_vec,
																					output_filename_list=[],
																					select_config=select_config)

		gene_query_vec = df1['gene_id'].unique()
		peak_query_vec = df1['peak_id'].unique()
		gene_query_num, peak_query_num = len(gene_query_vec), len(peak_query_vec)
		print('gene number:%d, peak number:%d '%(gene_query_num,peak_query_num))

		if save_mode>0:
			if output_file_path=='':
				output_file_path = input_file_path
			if output_filename=="":
				output_filename = '%s/%s.ori.txt'%(output_file_path,filename_prefix_save_pre1)
			if os.path.exists(output_filename)==True:
				print('the file exists ',output_filename)
			else:
				df1.to_csv(output_filename,sep='\t',float_format='%.5f')
			# output_filename = '%s/%s.peak_ratio.%d_%d.txt'%(output_file_path,filename_prefix_save_pre1,start_id1,start_id2)
			output_filename = '%s/%s.peak_ratio.1.txt'%(output_file_path,filename_prefix_save_pre1)
			flag_write=1
			if os.path.exists(output_filename)==True:
				print('the file exists ',output_filename)
				if overwrite==0:
					flag_write=0

			if flag_write>0:
				df1_ratio.to_csv(output_filename,sep='\t',float_format='%.5f')
			
		if verbose>0:
			print('peak-TF-gene links, dataframe of size ',df1.shape)
			print('percentage of metacells used in computation, dataframe of size ',df1_ratio.shape)

		stop = time.time()
		print('combining gene-TF expresssion partial correlations of peak-TF-gene links used %.5fs'%(stop-start))
				
		# flag_save_subset_1=1
		# if flag_save_subset_1>0:
		# 	df1_1 = df1[0:10000]	# save subset of data for preview
		# 	output_filename = '%s/%s.subset1.txt'%(output_file_path,filename_prefix_save_pre1)
		# 	df1_1.to_csv(output_filename,sep='\t',float_format='%.5f')

		return df1, df1_ratio

	## ====================================================
	# combine gene expression-TF expression partial correlations conditioned on peak accessibility of different subsets of peak-TF-gene links
	def test_partial_correlation_gene_tf_cond_peak_combine_1(self,input_file_path='',idvec=[],interval=1000,save_mode=0,output_filename_list=[],filename_prefix_vec=[],verbose=0,select_config={}):
		
		"""
		combine gene expression-TF expression partial correlations conditioned on peak accessibility of different subsets of peak-TF-gene links
		:param input_file_path: the directory to retrieve data from
		:param idvec: (array or list) the start and stop indices of the target genes
		:param interval: the batch size of the target genes if the gene-TF expression partial correlations conditioned on peak accessibility were computed in batch mode
		:param save_mode: indicator of whether to save data
		:param output_filename_list: (list) filenames to save data
		:param filename_prefix_vec: (array or list) prefix or annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) peak-TF-gene link annotations including gene-TF expression partial correlation given peak accessibility;
				 2. (dataframe) percentage of metacells used in computing gene-TF expression partial correlation given peak accessibility for each peak-gene link or peak-TF-gene link;
		"""

		if input_file_path!='':
			if len(filename_prefix_vec)>0:
				# filename_prefix_1,filename_prefix_2,filename_annot2 = filename_prefix_vec
				filename_prefix_save_1 = filename_prefix_vec[0]
				
				input_filename_list1 = []
				start_id_1, start_id_2 = idvec[0:2]
				gene_query_num_1 = idvec[2]
				for start_id1 in range(start_id_1,start_id_2,interval):
					start_id2 = np.min([start_id1+interval,gene_query_num_1])
					input_filename = '%s/%s.%d_%d.txt'%(input_file_path,filename_prefix_save_1,start_id1,start_id2)
					input_filename_list1.append(input_filename)

				list_1, list_ratio_1 = [], []
				file_num1 = len(input_filename_list1)
				column_idvec = ['motif_id','peak_id','gene_id']
				column_id3, column_id2, column_id1 = column_idvec
				verbose_internal = self.verbose_internal
				# query gene-TF expression partial correlation given peak accessibility and percentage of metacells included in computation
				for i1 in range(file_num1):
					input_filename = input_filename_list1[i1]
					if os.path.exists(input_filename)==False:
						print('the file does not exist ',input_filename,i1)
						continue
						# return
					
					df_query1 = pd.read_csv(input_filename,index_col=0,sep='\t')
					df_query1.index = test_query_index(df_query1,column_vec=column_idvec)
					t_columns = df_query1.columns.difference(['ratio_1','ratio_cond1'],sort=False)
					df_query_1 = df_query1.loc[:,t_columns]
					
					if verbose_internal==2:
						print('gene-TF expression partial correlation given peak accessibility, dataframe of size ',df_query1.shape)
						print('data loaded from %s'%(input_filename),i1)
						print('preview:\n',df_query1[0:5])
					list_1.append(df_query_1)

					column_1, column_2 = 'ratio_1', 'ratio_cond1'
					column_vec = df_query1.columns
					if (column_1 in column_vec) and (column_2 in column_vec):
						df_query2 = df_query1.loc[:,['peak_id','gene_id','ratio_1','ratio_cond1']]
						df_query2.index = test_query_index(df_query2,column_vec=['peak_id','gene_id'])
						df_query_2 = df_query2.loc[~df_query2.index.duplicated(keep='first')]
						list_ratio_1.append(df_query_2)
						if verbose_internal==2:
							print('percentage of metacells included in computation, preview:\n' ,df_query2[0:5])
					
				df1_pre1 = pd.concat(list_1,axis=0,join='outer',ignore_index=False)			
				df1 = df1_pre1.loc[~df1_pre1.index.duplicated(keep='first')] # unduplicated dataframe
				df1 = df1.sort_values(by=['gene_id','peak_id','gene_tf_corr_peak'],ascending=[True,True,False])
				df1.index = np.asarray(df1['motif_id'])
				df1_ratio = []
				
				file_save = 0
				if (save_mode>0) and (len(output_filename_list)>0):
					output_filename_1, output_filename_2 = output_filename_list
					if os.path.exists(output_filename_1)==True:
						print('the file exists ',output_filename_1)
					else:
						df1.to_csv(output_filename_1,sep='\t',float_format='%.5f')
						file_save = 1
				
					if len(list_ratio_1)>0:
						df1_ratio_pre1 = pd.concat(list_ratio_1,axis=0,join='outer',ignore_index=False)
						df1_ratio = df1_ratio_pre1.loc[~df1_ratio_pre1.index.duplicated(keep='first')]	# unduplicated dataframe;
						df1_ratio = df1_ratio.sort_values(by=['gene_id','peak_id'],ascending=True)
						column_query = 'gene_id'
						df1_ratio.index = np.asarray(df1_ratio[column_query])

						# output_filename_2 = '%s/%s.%s.%s.peak_ratio.%d_%d.txt'%(output_file_path,filename_prefix_1,filename_prefix_2,filename_annot2,start_id1,start_id2)
						if os.path.exists(output_filename_2)==True:
							print('the file exists ',output_filename_2)
						else:
							df1_ratio.to_csv(output_filename_2,sep='\t',float_format='%.5f')
						print('df1, df1_ratio ',df1.shape,df1_ratio.shape)

					save_mode_query = self.save_mode
					if (save_mode_query==1) and (file_save>0):
						# without saving the intermediate files
						for i1 in range(file_num1):
							filename_query = input_filename_list1[i1]
							try:
								os.remove(filename_query)
							except Exception as error:
								print('error! ',error)

				return df1, df1_ratio

	## ====================================================
	# convert wide format dataframe to long format and combine dataframes
	def test_query_feature_combine_format_1(self,df_list,column_idvec=[],field_query=[],dropna=False,select_config={}):

		"""
		convert wide format dataframe to long format and combine dataframes
		:df_list: (list) dataframes for which we perform conversion from wide format to long format
		:column_idvec: (array or list) two elements included: 1: the column to use as identifier variables in the dataframe after conversion;
		                               						  2: name to use for the 'variable' column in the dataframe after conversion;
		:param field_query: (list) name to use for the 'value' column in the converted dataframe for each dataframe in df_list
		:param dropna: indicator of whether to drop rows with NA in the dataframe
		:param select_config: dictionary containing parameters
		:return: dataframe integrating the long format dataframes converted from the given list of dataframes
		"""

		query_num1 = len(field_query)
		column_id1, column_id2 = column_idvec[0:2]
		for i1 in range(query_num1):
			df_query = df_list[i1]
			field1 = field_query[i1]
			df_query[column_id1] = np.asarray(df_query.index)
			df_query = df_query.melt(id_vars=[column_id1],var_name=column_id2,value_name=field1)
			if dropna==True:
				df_query = df_query.dropna(axis=0,subset=[field1])
			df_query.index = test_query_index(df_query,column_vec=column_idvec)
			df_list[i1] = df_query

		df_query_1 = pd.concat(df_list,axis=1,ignore_index=False)

		return df_query_1

	## ====================================================
	# query gene-TF expression correlations and peak accessibilility-TF expression correlations
	def test_gene_peak_tf_query_score_init_pre1_1(self,gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],load_mode=0,input_file_path='',
													save_mode=0,output_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		query gene-TF expression correlations and peak accessibilility-TF expression correlations
		:param gene_query_vec: (array or list) the target genes
		:param peak_query_vec: (array or list) the ATAC-seq peak loci for which to estimate peak-TF-gene links
		:param motif_query_vec: (array or list) TF names
		:param load_mode: indictor of which type of correlation and p-value data to load from saved files:
						  1. load gene-TF expression correlations and p-values only;
						  2. load peak accessibilility-TF expression correlations and p-values only;
						  3. both 1 and 2;
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing peak accessibilility-TF expression correlations and adjusted p-values;
				 2. dictionary containing gene-TF expression correlations and adjusted p-values;
		"""

		field_query1 = ['peak_tf_corr','peak_tf_pval_corrected']
		field_query2 = ['gene_tf_corr','gene_tf_pval_corrected']
		field_list_ori = [field_query1,field_query2]
		query_num1 = len(field_list_ori)

		flag_load_1 = (load_mode in [1,2])
		input_filename_list = []
		field_list = []

		data_file_type_query = select_config['data_file_type']
		correlation_type = 'spearmanr'
		if 'correlation_type' in select_config:
			correlation_type = select_config['correlation_type']
		filename_annot_vec = [correlation_type,'pval_corrected']

		if flag_load_1>0:
			column_1 = 'file_path_gene_tf'
			file_path_gene_tf = select_config[column_1]
			input_file_path_query = file_path_gene_tf

			# filename_prefix_1 = 'test_gene_tf_correlation.%s'%(data_file_type_query)
			filename_prefix_1 = select_config['filename_prefix_gene_tf']

			input_filename_1 = '%s/%s.%s.1.txt'%(input_file_path_query,filename_prefix_1,correlation_type)
			input_filename_2 = '%s/%s.%s.pval_corrected.1.txt'%(input_file_path_query,filename_prefix_1,correlation_type)

			if os.path.exists(input_filename_1)==False:
				print('the file does not exist: %s'%(input_filename_1))
				query_id1 = select_config['query_id1']
				query_id2 = select_config['query_id2']
				iter_mode = 0
				# gene-TF expression correlation was computed in batch mode
				if (query_id1>=0) and (query_id2>query_id1):
					iter_mode = 1
					feature_query_num_1 = select_config['feature_query_num_1']
					query_id2_pre = np.min([query_id2,feature_query_num_1])
					filename_prefix_2 = '%s.%d_%d'%(filename_prefix_1,query_id1,query_id2)

				input_filename_1 = '%s/%s.%s.1.txt'%(input_file_path_query,filename_prefix_2,correlation_type)
				input_filename_2 = '%s/%s.%s.pval_corrected.1.txt'%(input_file_path_query,filename_prefix_2,correlation_type)
			
			# filename_gene_expr_corr = input_filename_list1
			filename_gene_expr_corr = [input_filename_1,input_filename_2]
			select_config.update({'filename_gene_expr_corr':filename_gene_expr_corr})

			input_filename_list.append(filename_gene_expr_corr)
			feature_type_annot1 = 'gene_tf_corr'
			field_list.append([field_query2,feature_type_annot1])
		
		flag_load_2 = (load_mode in [0,2])
		if flag_load_2>0:
			column_2 = 'file_path_peak_tf'
			if column_2 in select_config:
				file_path_peak_tf = select_config[column_2]
				print('%s: %s'%(column_2,file_path_peak_tf))
				input_file_path_query2 = file_path_peak_tf
			else:
				input_file_path_query2 = select_config['file_path_motif_score']

			if 'filename_prefix_peak_tf' in select_config:
				filename_prefix_peak_tf = select_config['filename_prefix_peak_tf']
			else:
				filename_prefix_peak_tf = 'test_peak_tf_correlation.%s'%(data_file_type_query)
				
			filename_save_annot_peak_tf = '1'
			if 'filename_save_annot_peak_tf' in select_config:
				filename_save_annot_peak_tf = select_config['filename_save_annot_peak_tf']

			# query file paths of the peak accessibility-TF expression correlation and p-value
			filename_prefix = filename_prefix_peak_tf
			file_save_path_2 = input_file_path_query2
			input_filename_list2 = ['%s/%s.%s.%s.txt'%(file_save_path_2,filename_prefix,filename_annot1,filename_save_annot_peak_tf) for filename_annot1 in filename_annot_vec[0:2]]
			filename_peak_tf_corr = input_filename_list2
			select_config.update({'filename_peak_tf_corr':filename_peak_tf_corr})

			input_filename_list.append(filename_peak_tf_corr)
			feature_type_annot2 = 'peak_tf_corr'
			field_list.append([field_query1,feature_type_annot2])

		dict_query_pre1 = dict()
		feature_type_vec = ['gene_tf_corr','peak_tf_corr']
		for feature_type_annot in feature_type_vec:
			dict_query_pre1[feature_type_annot] = dict()

		query_num1 = len(input_filename_list)
		for i1 in range(query_num1):
			filename_list_query = input_filename_list[i1]
			input_filename = filename_list_query[0]
			t_vec_1 = field_list[i1]
			field_query, feature_type_annot = t_vec_1[0:2]
			print('input_filename: ',input_filename,field_query,feature_type_annot)

			b1 = input_filename.find('.txt')
			if b1<0:
				adata = sc.read(input_filename)
				if (i1==1) and (len(motif_query_vec)>0):
					adata = adata[:,motif_query_vec]
				feature_query1, feature_query2 = adata.obs_names, adata.var_names
				try:
					df_query_1 = pd.DataFrame(index=feature_query1,columns=feature_query2,data=adata.X.toarray(),dtype=np.float32)								
				except Exception as error:
					print('error! ',error)
					df_query_1 = pd.DataFrame(index=feature_query1,columns=feature_query2,data=np.asarray(adata.X),dtype=np.float32)								

				field_id1 = field_query[1]
				df_query = adata.obsm[field_id1]
				df_query_2 = pd.DataFrame(index=feature_query1,columns=feature_query2,data=df_query.toarray(),dtype=np.float32)
				
				list1 = [df_query_1,df_query_2]
			else:
				df_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t') # the correlation value
				input_filename_2 = filename_list_query[1]
				df_query_2 = pd.read_csv(input_filename_2,index_col=0,sep='\t') # the p-value corrected

				if len(motif_query_vec)>0:
					df_query_1 = df_query_1.loc[:,motif_query_vec]
					df_query_2 = df_query_2.loc[:,motif_query_vec]

				print('correlation value, dataframe of size ',df_query_1.shape)
				print('p-value corrected, dataframe of size ',df_query_2.shape)

				annot_vec_query = ['correlation value','p-value corrected']

				list1 = []
				for (input_filename,annot_str) in zip(filename_list_query,annot_vec_query):
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t') # the correlation value or p-value
					if len(motif_query_vec)>0:
						df_query = df_query.loc[:,motif_query_vec]

					print('%s, dataframe of size '%(annot_str),df_query.shape)
					list1.append(df_query)

			dict_query_1 = dict(zip(field_query,list1))
			dict_query_pre1.update({feature_type_annot:dict_query_1})

		list1 = [dict_query_pre1[feature_type_annot] for feature_type_annot in feature_type_vec]
		dict_gene_tf_query, dict_peak_tf_query = list1[0:2]

		return dict_peak_tf_query, dict_gene_tf_query

	## ====================================================
	# query correlations or partial correlations and p-values of different types of links, and the regularization scores of peak-TF links
	def test_gene_peak_tf_query_score_init_pre1_2(self,df_gene_peak_query=[],df_peak_tf_1=[],df_peak_tf_2=[],dict_peak_tf_query={},dict_gene_tf_query={},df_gene_tf_corr_peak=[],
														input_file_path='',save_mode=0,output_file_path='',output_filename='',verbose=0,select_config={}):

		"""	
		query correlations or partial correlations and p-values of different types of links, and the regularization scores of peak-TF links
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param df_peak_tf_1: (dataframe) TF binding scores estimated by the in silico ChIP-seq method and other associated scores of peak-TF links
		:param df_peak_tf_2: (dataframe) scaling scores (regularization) of peak-TF link estimated by Unify
		:param dict_peak_tf_query: dictionary containing peak accessibilility-TF expression correlations and adjusted p-values
		:param dict_gene_tf_query: dictionary containing gene-TF expression correlations and adjusted p-values
		:param df_gene_tf_corr_peak: (dataframe) annotations of peak-TF-gene links including gene-TF expression partial correlation conditioned peak accessibility
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-TF-gene links including correlations or partial correlations and p-values of different types of links, and the regularization scores of peak-TF links
		"""

		# query peak-TF link, gene-TF link, peak-gene link, and peak-TF-gene link attributes (including correlations or partial correlations and adjusted or empirical p-values)
		flag_link_type_query=0
		field_query_pre1 = ['peak_tf_corr','peak_tf_pval_corrected',
								'gene_tf_corr_peak','gene_tf_corr_peak_pval_corrected1','gene_tf_corr_peak_pval_corrected2',
								'gene_tf_corr','gene_tf_pval_corrected',
								'peak_gene_corr_','peak_gene_corr_pval']

		# score from the in silico ChIP-seq method
		field_query_pre2_1 = ['correlation_score','max_accessibility_score',
								'motif_score','motif_score_normalize',
								'score_1','score_pred1']

		# score from the motif score normalization
		field_query_pre2_2 = ['motif_score','motif_score_minmax','motif_score_log_normalize_bound',
								'max_accessibility_score','score_accessibility','score_accessibility_minmax','score_1']

		df_query1 = df_gene_tf_corr_peak  # peak-TF-gene link annotation dataframe

		# copy columns of one dataframe to another dataframe
		feature_type_vec = ['motif','peak','gene']
		column_idvec = ['motif_id','peak_id','gene_id'] # columns representing TF, peak, and gene name or index in the link annotation dataframe
		column_id3, column_id2, column_id1 = column_idvec

		feature_query_list = []
		query_num1 = len(column_idvec)
		for i1 in range(query_num1):
			column_id_query = column_idvec[i1]
			feature_type_query = feature_type_vec[i1]
			feature_query_1 = df_gene_tf_corr_peak[column_id_query].unique()
			feature_query_list.append(feature_query_1)
			feature_query_num1 = len(feature_query_1)
			print('feature_query: ',feature_type_query,feature_query_num1)
		motif_query_1, peak_query_1, gene_query_1 = feature_query_list

		# query peak accessibility-gene expression correlation
		column_idvec_1 = [column_id2,column_id1]
		column_vec_1 = [['spearmanr','pval1']]
		print('df_gene_tf_corr_peak ',df_gene_tf_corr_peak.shape)
		print(df_gene_tf_corr_peak[0:2])
		print('df_gene_peak_query ',df_gene_peak_query.shape)
		print(df_gene_peak_query[0:2])

		df_gene_peak_query_ori = df_gene_peak_query
		df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])
		df_gene_peak_query = df_gene_peak_query.loc[gene_query_1,:]
		print('df_gene_peak_query_ori, df_gene_peak_query ',df_gene_peak_query_ori.shape,df_gene_peak_query.shape)
		df_list1 = [df_gene_tf_corr_peak,df_gene_peak_query]

		print('query peak accessibility-gene expression correlation')
		from .utility_1 import test_column_query_1
		# copy specified columns from the other dataframes to the first dataframe
		df_query1 = test_column_query_1(input_filename_list=[],
										id_column=column_idvec_1,
										column_vec=column_vec_1,
										df_list=df_list1,
										type_id_1=0,type_id_2=0,
										reset_index=True,
										select_config=select_config)

		column_1 = column_vec_1[0]
		column_2 = ['peak_gene_corr_','peak_gene_corr_pval']
		# df_query1 = df_query1.rename(columns={column_1:column_2})
		dict1 = dict(zip(column_1,column_2))
		df_query1 = df_query1.rename(columns=dict1)

		# query peak accessibility-TF expression correlation
		print('query peak accessibility-TF expression correlation')
		start = time.time()
		field_query = ['peak_tf_corr','peak_tf_pval_corrected']
		list2_ori = [dict_peak_tf_query[field1] for field1 in field_query]
		# list2 = [df_query.loc[peak_query_1,motif_query_1] for df_query in list2_ori]
		field_query_num1 = len(field_query)
		list2 = []
		for i1 in range(field_query_num1):
			df_query_ori = list2_ori[i1]
			df_query = df_query_ori.loc[peak_query_1,motif_query_1]
			print('df_query_ori, df_query: ',df_query_ori.shape,df_query.shape)
			list2.append(df_query)

		column_idvec_2 = [column_id2,column_id3] # column_idvec_2 = ['peak_id','motif_id']
		# convert wide format dataframe to long format and combine dataframe
		df_peak_tf_corr = self.test_query_feature_combine_format_1(df_list=list2,column_idvec=column_idvec_2,
																	field_query=field_query,
																	dropna=False,
																	select_config=select_config)

		print('df_peak_tf_corr: ',df_peak_tf_corr.shape)
		print(df_peak_tf_corr[0:2])

		df_list2 = [df_query1,df_peak_tf_corr]
		column_vec_2 = [field_query]
		type_id_1 = 3
		# copy specified columns from the other dataframes to the first dataframe
		df_query1 = test_column_query_1(input_filename_list=[],id_column=column_idvec_2,
										column_vec=column_vec_2,
										df_list=df_list2,
										type_id_1=type_id_1,type_id_2=0,
										reset_index=True,
										select_config=select_config)

		stop = time.time()
		print('query peak accessibility-TF expression correlation used %.2fs'%(stop-start))

		# query normalized motif score
		# field_query = ['motif_score','motif_score_minmax','max_accessibility_score','score_accessibility','score_1']
		field_query1 = ['motif_score','score_normalize_1','score_normalize_pred']
		field_query2 = ['motif_score_minmax','motif_score_log_normalize_bound','score_accessibility','score_1']
		print('query normalized motif score')
		start = time.time()
		if verbose>0:
			print('TF binding scores estimated by the in silico ChIP-seq method, dataframe of size ',df_peak_tf_1.shape)
			print('scaling (regularization) scores of peak-TF links estimated by Unify, dataframe of size ',df_peak_tf_2.shape)

		df_peak_tf_1[column_id3] = np.asarray(df_peak_tf_1.index)
		df_peak_tf_2[column_id3] = np.asarray(df_peak_tf_2.index)
		if len(df_peak_tf_1)>0:
			column_vec_2 = [field_query2,field_query1]
			df_peak_tf_1 = df_peak_tf_1.rename(columns={'score_1':'score_normalize_1','score_pred1':'score_normalize_pred'})
			df_list2_1 = [df_query1,df_peak_tf_2,df_peak_tf_1]
		else:
			column_vec_2 = [field_query2]
			df_list2_1 = [df_query1,df_peak_tf_2]
		
		# copy specified columns from the other dataframes to the first dataframe
		df_query1 = test_column_query_1(input_filename_list=[],id_column=column_idvec_2,
										column_vec=column_vec_2,
										df_list=df_list2_1,
										type_id_1=0,type_id_2=0,
										reset_index=True,
										select_config=select_config)
		stop = time.time()
		print('query normalized motif score used %.2fs'%(stop-start))

		# query gene-TF expression correlation
		if len(dict_gene_tf_query)>0:
			print('query gene-TF expression correlation')
			start = time.time()
			field_query = ['gene_tf_corr','gene_tf_pval_corrected']
			list3 = [dict_gene_tf_query[field1] for field1 in field_query]
			column_idvec_3 = [column_id1,column_id3] # column_idvec_2=['gene_id','motif_id']
			
			# convert wide format dataframe to long format and combine dataframe
			df_gene_tf_corr = self.test_query_feature_combine_format_1(df_list=list3,column_idvec=column_idvec_3,
																		field_query=field_query,
																		dropna=False,
																		select_config=select_config)

			print('df_gene_tf_corr: ',df_gene_tf_corr.shape)
			print(df_gene_tf_corr[0:2])

			df_list3 = [df_query1,df_gene_tf_corr]
			column_vec_3 = [field_query]
			type_id_1 = 3
			# copy specified columns from the other dataframes to the first dataframe
			df_query1 = test_column_query_1(input_filename_list=[],id_column=column_idvec_3,
											column_vec=column_vec_3,
											df_list=df_list3,
											type_id_1=type_id_1,type_id_2=0,
											reset_index=True,
											select_config=select_config)
			stop = time.time()
			print('query gene-TF expression correlation used %.2fs'%(stop-start))

		return df_query1

	## ====================================================
	# calculate peak-TF-gene association scores
	def test_gene_peak_tf_query_score_init_pre1(self,df_gene_peak_query=[],df_gene_tf_corr_peak=[],lambda1=0.5,lambda2=0.5,column_id1=-1,
													flag_init_score_pre1=0,flag_link_type_query=0,flag_init_score_1=0,
													flag_save_1=1,flag_save_2=1,flag_save_3=1,input_file_path='',
													save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		calculate peak-TF-gene association scores
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param df_gene_tf_corr_peak: (dataframe) annotations of peak-TF-gene links including gene-TF expression partial correlation conditioned peak accessibility
		:param lambda1: (float) weight of score 1 (TF-(peak,gene) association score)
		:param lambda2: (float) weight of score 2 ((peak,TF)-gene association score)
		:param column_id1: (str) column representing the maximal score possible for a given peak-TF-gene link using the associated regularization score
		:param flag_init_score_pre1: indicator of whether to query correlations or partial correlations and p-values of different types of links, 
									 and the regularization scores of peak-TF links
		:param flag_link_type_query: indicator of whether to estimate the type of peak-TF link, peak-gene link, and gene-TF link
		:param flag_init_score_1: indicator of whether to calculate peak-TF-gene association scores
		:param flag_save_1: indicator of whether to save peak-TF-gene link annotations including correlation or partial correlation and p-value of different types of links, 
							motif scores, peak accessibility scores, conditioned metacell percentage, and regularization scores of peak-TF links
		:param flag_save_2: indicator of whether to save peak-TF-gene link annotations including peak-TF, peak-gene, and gene-TF link types and the score weights
		:param flag_save_3: indicator of whether to save peak-TF-gene link annotations including peak-TF-gene association scores, 
							correlation or partial correlation of different types of links, and other types of associated scores
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-TF-gene links including computed peak-TF-gene association scores
		"""

		# peak-tf link, gene-tf link and peak-gene link query
		# flag_init_score_pre1=0
		flag_init_score_pre1=1
		if 'column_idvec' in select_config:
			column_idvec = select_config['column_idvec']
		else:
			column_idvec = ['motif_id','peak_id','gene_id']
			select_config.update({'column_idvec':column_idvec})

		df_link_query_1 = []
		# file_save_path = select_config['data_path_save']
		file_save_path = select_config['data_path_save_local']
		file_save_path2 = select_config['file_path_motif_score']
		input_file_path2 = file_save_path2

		data_file_type_query = select_config['data_file_type']
		filename_prefix_pre1 = data_file_type_query
		filename_prefix_default = select_config['filename_prefix_default']
		filename_prefix_default_1 = select_config['filename_prefix_default_1']
		filename_prefix_cond = select_config['filename_prefix_cond']

		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		iter_mode = 0
		if (query_id1>=0) and (query_id2>query_id1):
			iter_mode = 1
			select_config.update({'iter_mode':iter_mode})

		output_file_path = file_save_path2
		if iter_mode==0:
			filename_prefix_save_pre2 = '%s.pcorr_query1'%(filename_prefix_cond)
		else:
			filename_prefix_save_pre2 = '%s.pcorr_query1.%d_%d'%(filename_prefix_cond,query_id1,query_id2)
			
		if flag_init_score_pre1>0:
			# combine the different scores
			filename_peak_tf_corr = []
			filename_gene_expr_corr = []

			input_file_path = file_save_path
			print('input_file_path: ',input_file_path)

			field_query_1 = ['gene_tf_corr','peak_tf_corr']
			field_query1, field_query2 = field_query_1[0:2]
			correlation_type = 'spearmanr'
			
			dict_peak_tf_query = self.dict_peak_tf_query
			dict_gene_tf_query = self.dict_gene_tf_query

			# input_file_path = select_config['file_path_motif_score']
			df_peak_tf_1 = self.df_peak_tf_1
			df_peak_tf_2 = self.df_peak_tf_2

			# add the annotations of correlations or partial correlations and p-values of different types of links, and the regularization scores of peak-TF links
			df_link_query_1 = self.test_gene_peak_tf_query_score_init_pre1_2(df_gene_peak_query=df_gene_peak_query,
																				df_peak_tf_1=df_peak_tf_1,
																				df_peak_tf_2=df_peak_tf_2,
																				dict_peak_tf_query=dict_peak_tf_query,
																				dict_gene_tf_query=dict_gene_tf_query,
																				df_gene_tf_corr_peak=df_gene_tf_corr_peak,
																				input_file_path='',
																				save_mode=1,output_file_path=output_file_path,output_filename='',
																				verbose=verbose,select_config=select_config)

			# flag_save_1=1
			# column_idvec = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_idvec
			if flag_save_1>0:
				output_file_path = file_save_path2

				# peak and peak-gene link attributes
				field_query1 = ['ratio_1','ratio_cond1'] 
				# peak-TF association attributes
				field_query2 = ['motif_score_minmax','motif_score_log_normalize_bound','score_accessibility','score_1','motif_score','score_normalize_1','score_normalize_pred']
				
				field_query3 = ['peak_tf_corr','peak_tf_pval_corrected',
								'gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1',
								'gene_tf_corr','gene_tf_pval_corrected',
								'peak_gene_corr_','peak_gene_corr_pval']

				field_query_1 = field_query3 + ['motif_score','score_normalize_pred']

				list_1 = [field_query_1,field_query1,field_query2]
				query_num1 = len(list_1)
				column_vec_1 = df_link_query_1.columns
				compression = None
				for i2 in range(query_num1):
					field_query_pre1 = list_1[i2]
					field_query = pd.Index(field_query_pre1).intersection(column_vec_1,sort=False)

					if len(field_query)>0:
						field_query_2 = list(column_idvec)+list(field_query)
						df_link_query_2 = df_link_query_1.loc[:,field_query_2]
						if i2==1:
							df_link_query_2 = df_link_query_2.drop_duplicates(subset=[column_id1,column_id2])	# peak-gene associations
						elif i2==2:
							df_link_query_2 = df_link_query_2.drop_duplicates(subset=[column_id2,column_id3])	# peak-TF associations
						
						if i2==0:
							extension = 'txt.gz'
							compression = 'gzip'
						else:
							extension = 'txt'
							compression = None
						output_filename = '%s/%s.annot1_%d.1.%s'%(output_file_path,filename_prefix_save_pre2,i2+1,extension)
						df_link_query_2.to_csv(output_filename,index=False,sep='\t',float_format='%.5f',compression=compression)

		# flag_link_type_query=0
		flag_link_type_query=1
		df_gene_peak_query_pre2=[]
		filename_save_annot_2 = 'annot2'
		column_idvec = select_config['column_idvec']
		# column_idvec = ['motif_id','peak_id','gene_id']
		field_query_1 = ['peak_gene_link','gene_tf_link','peak_tf_link']
		field_query_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']
		if flag_link_type_query>0:
			print('estimating link types for peak-TF-gene associations')
			start = time.time()
			
			df_gene_peak_query_1_ori = df_gene_peak_query
			if len(df_link_query_1)==0:
				if 'filename_gene_tf_peak_query_1' in select_config:
					filename_query_1 = select_config['filename_gene_tf_peak_query_1']
					from .utility_1 import test_file_merge_column
					df_link_query_1 = test_file_merge_column(filename_query_1,column_idvec=column_idvec,index_col=False,select_config=select_config)
					df_gene_peak_query_pre1_1 = df_link_query_1
				else:
					print('please provide peak-TF-gene link annotation file')
			else:
				df_gene_peak_query_pre1_1 = df_link_query_1

			# estimate the type of peak-TF link, peak-gene link, and gene-TF link (repression or activation)
			# adjust the weights of score 1 and score 2 based on the link types
			print('estimating peak-TF, peak-gene, and gene-TF link types for peak-TF-gene associations')
			df_gene_peak_query_pre2 = self.test_query_tf_peak_gene_pair_link_type(gene_query_vec=[],peak_query_vec=[],
																					motif_query_vec=[],
																					df_gene_peak_query=df_gene_peak_query_1_ori,
																					df_gene_peak_tf_query=df_gene_peak_query_pre1_1,
																					filename_annot='',
																					select_config=select_config)

			df_gene_peak_query_pre2.index = np.asarray(df_gene_peak_query_pre2['gene_id'])

			field_query_pre1 = list(column_idvec) + field_query_1 + field_query_2 # link type annotation
			field_query_3 = ['ratio_1','ratio_cond1','motif_score_log_normalize_bound','score_accessibility']
			field_query_pre2 = list(column_idvec) + field_query_3
			field_query_pre3 = df_gene_peak_query_pre2.columns.intersection(field_query_pre1,sort=False)
			
			output_file_path = input_file_path2
			flag_annot_2=0
			if flag_annot_2>0:
				df_gene_peak_query_annot2 = df_gene_peak_query_pre2.loc[:,field_query_pre2]
				output_filename_3 = '%s/%s.%s_2.1.txt'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2)
				df_gene_peak_query_annot2.index = np.asarray(df_gene_peak_query_annot2['gene_id'])
				df_gene_peak_query_annot2.to_csv(output_filename_3,sep='\t',float_format='%.5f')
				# print('df_gene_peak_query_annot2 ',df_gene_peak_query_annot2.shape,df_gene_peak_query_annot2[0:2])
				print('peak-TF-gene link annotation subset, dataframe of size ',df_gene_peak_query_annot2.shape)
				print('data preview:\n ',df_gene_peak_query_annot2[0:2])
			
			flag_annot_3=1
			if flag_annot_3>0:
				column_query1 = 'group'
				if column_query1 in df_gene_peak_query_pre2:
					field_query_pre1 = field_query_pre1+['group']
				df_gene_peak_query_2 = df_gene_peak_query_pre2.loc[:,field_query_pre1]
				
				extension = 'txt.gz'
				compression = 'infer'
				if extension in ['txt.gz']:
					compression = 'gzip'
				
				output_filename_1 = '%s/%s.%s_1.1.%s'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2,extension)				
				for field_id1 in field_query_1:
					df_gene_peak_query_2[field_id1] = np.int8(df_gene_peak_query_2[field_id1])
				
				df_gene_peak_query_2.to_csv(output_filename_1,sep='\t',float_format='%.5f',compression=compression)
				print('link types in peak-TF-gene associations and score weights, dataframe of size ',df_gene_peak_query_2.shape)
				print('data preview:\n',df_gene_peak_query_2[0:2])
			
			if flag_save_2>0:
				extension = 'txt.gz'
				compression = 'infer'
				if extension in ['txt.gz']:
					compression = 'gzip'
				
				output_filename = '%s/%s.%s.1.%s'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2,extension)
				df_gene_peak_query_pre2.to_csv(output_filename,sep='\t',float_format='%.5f',compression=compression)
				print('peak-TF-gene links, dataframe of size ',df_gene_peak_query_pre2.shape)
				print('data preview:\n',df_gene_peak_query_pre2[0:2])

			stop = time.time()
			print('peak-tf-gene link query: ',stop-start)

		# calculate peak-TF-gene association scores
		flag_init_score_1=1
		if flag_init_score_1>0:
			if len(df_gene_peak_query_pre2)==0:
				filename_query_1 = select_config['filename_gene_tf_peak_query_2']
				from .utility_1 import test_file_merge_column
				df_gene_peak_query_pre2 = test_file_merge_column(filename_query_1,column_idvec=column_idvec,index_col=False,select_config=select_config)

			print('peak-TF-gene links, dataframe of size ',df_gene_peak_query_pre2.shape)
			# field_query_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']

			lambda1 = 0.50
			lambda2 = 1-lambda1
			df_gene_peak_query_pre2[field_query_2] = df_gene_peak_query_pre2[field_query_2].fillna(lambda1)

			column_id1 = 'score_pred1_1'
			print('calculate peak-TF-gene association scores')
			df_gene_peak_query_1 = self.test_gene_peak_tf_query_score_init_1(df_gene_peak_query=df_gene_peak_query_pre2,
																				lambda1=lambda1,lambda2=lambda2,
																				column_id1=column_id1,
																				select_config=select_config)
				
			print('peak-TF-gene links, dataframe of size ',df_gene_peak_query_1.shape)
			print('data preview: ')
			print(df_gene_peak_query_1[0:2])

			# retrieve the columns of link score and subset of the annotations
			column_idvec = select_config['column_idvec']
			# field_query_3 = ['motif_id','peak_id','gene_id','score_1','score_combine_1','score_pred1_correlation','score_pred1','score_pred1_1','score_pred2','score_pred_combine']
			if 'column_score_query2' in select_config:
				column_score_query2 = select_config['column_score_query2']
			else:
				column_score_query2 = ['peak_tf_corr','gene_tf_corr_peak','peak_gene_corr_','gene_tf_corr','score_1','score_combine_1','score_normalize_pred','score_pred1_correlation','score_pred1','score_pred1_1','score_combine_2','score_pred2','score_pred_combine']

			field_query_3 = column_idvec + column_score_query2
			df_gene_peak_query1_1 = df_gene_peak_query_1.loc[:,field_query_3]
			if flag_save_3>0:
				extension = 'txt.gz'
				compression = 'gzip'
				output_filename = '%s/%s.%s.init.1.%s'%(output_file_path,filename_prefix_save_pre2,filename_save_annot_2,extension)
				df_gene_peak_query1_1.index = np.asarray(df_gene_peak_query1_1['gene_id'])
				df_gene_peak_query1_1.to_csv(output_filename,index=False,sep='\t',float_format='%.5f',compression=compression)

			return df_gene_peak_query1_1

	## ====================================================
	# compute peak-TF-gene association scores
	def test_gene_peak_tf_query_score_compute_unit_1(self,df_feature_link=[],flag_link_type=1,flag_compute=1,flag_annot_1=0,retrieve_mode=0,
														save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		compute peak-TF-gene association scores
		:param df_feature_link: (dataframe) annotations of peak-TF-gene links
		:param flag_link_type: indicator of whether to estimate the type of peak-TF link, peak-gene link, and gene-TF link (repression or activation) for peak-TF-gene associations
		:param flag_compute: indicator of whether to calculate peak-TF-gene association scores
		:param flag_annot_1: indicator of whether to query additional attributes of peak-TF-gene associations from extenal data
		:param retrieve_mode: indicator of which columns to include in the retrieved peak-TF-gene link annotation dataframe 
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: (dataframe) annotations of peak-TF-gene links including computed peak-TF-gene association scores
		"""

		flag_query1 = flag_link_type
		df_gene_peak_query_ori = df_feature_link
		lambda1 = 0.50
		lambda2 = 1-lambda1
		if flag_query1>0:
			# estimate the type of peak-TF link, peak-gene link, and gene-TF link
			# adjust the weights of score 1 and score 2 based on the link types
			df_gene_peak_query_pre1 = self.test_query_tf_peak_gene_pair_link_type(gene_query_vec=[],peak_query_vec=[],
																					motif_query_vec=[],
																					df_gene_peak_query=df_gene_peak_query_ori,
																					df_gene_peak_tf_query=df_gene_peak_query_ori,
																					filename_annot='',
																					flag_annot_1=flag_annot_1,
																					verbose=verbose,select_config=select_config)

			# field_query_1 = ['lambda_gene_peak','lambda_peak_tf','lambda_gene_tf_cond','lambda_gene_tf_cond2']
			# df_gene_peak_query_pre1.loc[:,field_query_1] = df_gene_peak_query_pre1.loc[:,field_query_1].fillna(lambda1)

		flag_query2 = flag_compute
		if flag_query2>0:
			column_id1 = 'score_pred1_1'
			print('calculate peak-TF-gene association scores')
			df_gene_peak_query_1 = self.test_gene_peak_tf_query_score_init_1(df_gene_peak_query=df_gene_peak_query_pre1,
																				lambda1=lambda1,
																				lambda2=lambda2,
																				column_id1=column_id1,
																				select_config=select_config)
				
			# print('df_gene_peak_query_1 ',df_gene_peak_query_1.shape,df_gene_peak_query_1[0:2])
			print('peak-TF-gene links, dataframe of size ',df_gene_peak_query_1.shape)
			print('data preview:')
			print(df_gene_peak_query_1[0:2])

			# retrieve_mode: 0, query and save the original link annotations; 1, query the original annotations and save subset of the annotations; 2, query and save subest of annotations; 
			if (retrieve_mode==2) or ((retrieve_mode==1) and (save_mode>0)):
				column_idvec = select_config['column_idvec']
				# field_query = ['motif_id','peak_id','gene_id','score_1','score_combine_1','score_pred1_correlation','score_pred1','score_pred1_1','score_pred2','score_pred_combine']
				# field_query = column_idvec + ['score_1','score_pred1_correlation','score_pred1','score_pred2','score_pred_combine']
				field_query = column_idvec+['peak_tf_corr','gene_tf_corr_peak','peak_gene_corr_','gene_tf_corr','score_1','score_combine_1','score_normalize_pred','score_pred1_correlation','score_pred1','score_pred1_1','score_combine_2','score_pred2','score_pred_combine']
				if 'column_score_query2' in select_config:
					field_query = select_config['column_score_query2']

				df_gene_peak_query1_1 = df_gene_peak_query_1.loc[:,field_query]
			
			flag_save_1=1
			if (save_mode>0) and (output_filename!=''):
				if retrieve_mode in [1,2]:
					df_gene_peak_query_2 = df_gene_peak_query1_1
				else:
					df_gene_peak_query_2 = df_gene_peak_query_1
				df_gene_peak_query_2.to_csv(output_filename,index=False,sep='\t',float_format='%.5f')

			if retrieve_mode==2:
				df_gene_peak_query1 = df_gene_peak_query1_1
			else:
				df_gene_peak_query1 = df_gene_peak_query_1
			
			return df_gene_peak_query1

	## ====================================================
	# calculate peak-TF-gene association scores
	def test_gene_peak_tf_query_score_init_1(self,df_gene_peak_query=[],lambda1=0.5,lambda2=0.5,column_id1=-1,verbose=0,select_config={}):

		"""
		calculate peak-TF-gene association scores
		:param df_gene_peak_query: (dataframe) annotations of peak-TF-gene links
		:param lambda1: (float) weight of score 1 (TF-(peak,gene) association score)
		:param lambda2: (float) weight of score 2 ((peak,TF)-gene association score)
		:param column_id1: (str) column representing the maximal score possible for a given peak-TF-gene link using the associated regularization score
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) updated peak-TF-gene link annotations including peak-TF-gene association scores
		"""

		flag_query1=1
		if flag_query1>0:
			df_query_1 = df_gene_peak_query
			# lambda_1, lambda_2 = df_query_1['lambda1'], df_query_1['lambda2']

			# lambda1 = 0.50
			# lambda2 = 1-lambda1
			field_query_1 = ['lambda_gene_peak','lambda_peak_tf','lambda_gene_tf_cond','lambda_gene_tf_cond2']
			df_query_1.loc[:,field_query_1] = df_query_1.loc[:,field_query_1].fillna(lambda1)

			lambda_gene_peak, lambda_gene_tf_cond2 = df_query_1['lambda_gene_peak'], df_query_1['lambda_gene_tf_cond2']
			lambda_peak_tf, lambda_gene_tf_cond = df_query_1['lambda_peak_tf'], df_query_1['lambda_gene_tf_cond']
				
			field_query = ['column_peak_tf_corr','column_peak_gene_corr','column_query_cond']
			list1 = ['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak']
			# column_peak_tf_corr = 'peak_tf_corr'
			# column_peak_gene_corr = 'peak_gene_corr_'
			# column_query_cond = 'gene_tf_corr_peak'
			query_num1 = len(field_query)
			for i1 in range(query_num1):
				field_id = field_query[i1]
				if field_id in select_config:
					list1[i1] = select_config[field_id]

			column_1, column_2, column_3 = list1[0:3]
			# peak_tf_corr = df_query_1['peak_tf_corr']
			# peak_gene_corr_, gene_tf_corr_peak = df_query_1['peak_gene_corr_'], df_query_1['gene_tf_corr_peak']
			peak_tf_corr = df_query_1[column_1]
			peak_gene_corr_, gene_tf_corr_peak = df_query_1[column_2], df_query_1[column_3]
			
			# recompute score_pred2
			score_combine_1 = lambda_peak_tf*peak_tf_corr+lambda_gene_tf_cond*gene_tf_corr_peak
			score_1 = df_query_1['score_1']
			score_pred1_correlation = peak_tf_corr*score_1
			score_pred1 = score_combine_1*score_1
			df_query_1['score_combine_1'] = score_combine_1 # the score 1 before the normalization

			column_score_1 = 'score_pred1'
			column_score_2 = 'score_pred2'
			if 'column_score_1' in select_config:
				column_score_1 = select_config['column_score_1']
			if 'column_score_2' in select_config:
				column_score_2 = select_config['column_score_2']

			df_query_1['score_pred1_correlation'] = score_pred1_correlation
			df_query_1[column_score_1] = score_pred1

			score_combine_2 = lambda_gene_peak*peak_gene_corr_ +lambda_gene_tf_cond2*gene_tf_corr_peak # the score 2 before the normalization
			df_query_1['score_combine_2'] = score_combine_2
			df_query_1[column_score_2] = score_combine_2*score_1

			if column_id1==-1:
				column_id1='score_pred1_1'
			
			df_gene_peak_query[column_id1] = ((lambda_peak_tf*peak_tf_corr).abs()+(lambda_gene_tf_cond*gene_tf_corr_peak).abs())*score_1

			a1 = 2/3.0
			score_pred_combine = (lambda_peak_tf*peak_tf_corr+0.5*(lambda_gene_tf_cond+lambda_gene_tf_cond2)*gene_tf_corr_peak+lambda_gene_peak*peak_gene_corr_)*a1
			df_query_1['score_pred_combine'] = score_pred_combine*score_1

			field_query_1 = ['peak_gene_link','gene_tf_link','peak_tf_link']
			field_query_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']
			field_query = field_query_1+field_query_2
			column_idvec = ['motif_id','peak_id','gene_id']

			df_gene_peak_query = df_query_1

		return df_gene_peak_query

	## ====================================================
	# save peak-TF-gene link annotations
	def test_gene_peak_tf_query_score_init_save(self,df_gene_peak_query=[],lambda1=0.5,lambda2=0.5,
													flag_init_score=1,flag_save_interval=1,
													feature_type='gene_id',query_mode=0,
													save_mode=1,output_file_path='',output_filename='',
													filename_prefix_save='',float_format='%.5f',select_config={}):

		"""
		save peak-TF-gene link annotations
		:param df_gene_peak_query: (dataframe) annotations of peak-TF-gene links
		:param lambda1: (float) weight of score 1 (TF-(peak,gene) association score)
		:param lambda2: (float) weight of score 2 ((peak,TF)-gene association score)
		:param flag_init_score: indicator of whether to calculate peak-TF-gene association scores
		:param flag_save_interval: indicator of whethter to save peak-TF-gene link annotations by subsets (batches)
		:param feature_type: (str) column in the peak-TF-gene link annotation dataframe which represents the feature name or index by which we divide the peak-TF-gene links into subsets (e.g., gene_id)
		:param query_mode: indicator of whether to retrieve peak-TF-gene link annotations by subsets
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param float_format: the format to keep data precision used in saving data
		:param select_config: dictionary containing parameters
		:return: (dataframe) peak-TF-gene link annotations
		"""

		# calculate peak-TF-gene association scores
		if flag_init_score>0:
			df_gene_peak_query_1 = self.test_gene_peak_tf_query_score_init_1(df_gene_peak_query=df_gene_peak_query,
																				lambda1=lambda1,
																				lambda2=lambda2,
																				select_config=select_config)
		else:
			df_gene_peak_query_1 = df_gene_peak_query

		# flag_save_interval, flag_save_ori = 0, 0
		if flag_save_interval>0:
			# interval = 5000
			interval = 2500
			# feature_type = 'gene_id'
			list_query_interval = self.test_gene_peak_tf_query_save_interval_1(df_gene_peak_query=df_gene_peak_query_1,
																				interval=interval,
																				feature_type=feature_type,
																				query_mode=query_mode,
																				save_mode=save_mode,
																				output_file_path=output_file_path,
																				filename_prefix_save=filename_prefix_save,
																				float_format=float_format,
																				select_config=select_config)
		
		# if flag_save_ori>0:
		if save_mode>0:
			# output_filename = '%s/%s.init.1.txt'%(output_file_path,filename_prefix_save)
			df_gene_peak_query_1.to_csv(output_filename,sep='\t',float_format='%.5f')

		return df_gene_peak_query_1

	## ====================================================
	# save peak-TF-gene link annotations by subsets
	def test_gene_peak_tf_query_save_interval_1(self,df_gene_peak_query=[],interval=5000,feature_type=0,query_mode=0,
													save_mode=1,output_file_path='',filename_prefix_save='',float_format='%.5f',select_config={}):

		"""
		save peak-TF-gene link annotations by subsets
		:param df_gene_peak_query: (dataframe) annotations of peak-TF-gene links
		:interval: the batch size by which we divide the corresponding observations (e.g., genes) into batches and divide the associated peak-TF-gene links into subsets
		:param feature_type: (str) column in the peak-TF-gene link annotation dataframe which represents the feature name or index by which we divide the peak-TF-gene links into subsets (e.g., gene_id)
		:param query_mode: indicator of whether to retrieve peak-TF-gene link annotations by subsets
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param float_format: the format to keep data precision used in saving data
		:param select_config: dictionary containing parameters
		:return: list of dataframes containing the annotations of subsets of peak-TF-gene links
		"""

		# interval=5000
		feature_type_vec = ['gene_id','peak_id','motif_id']
		assert feature_type<=2
		feature_type_id = feature_type_vec[feature_type]
		print('feature_type_id ',feature_type_id)
		feature_query_vec = df_gene_peak_query[feature_type_id].unique()
		feature_query_num = len(feature_query_vec)

		interval_num = np.int32(np.ceil(feature_query_num/interval))
		df_gene_peak_query.index = np.asarray(df_gene_peak_query[feature_type_id])
		list_query1 = []
		for i1 in range(interval_num):
			start_id1 = i1*interval
			start_id2 = np.min([(i1+1)*interval,feature_query_num])
			feature_query_vec_interval = feature_query_vec[start_id1:start_id2]
			feature_query_num1 = len(feature_query_vec_interval)
			df_gene_peak_query_2 = df_gene_peak_query.loc[feature_query_vec_interval,:]
			if (save_mode>0) and (output_file_path!=''):
				# output_filename_1 = '%s/%s.%s.%s.%d_%d.pre1.txt'%(output_file_path,filename_prefix_1,filename_prefix_2,filename_annot2,start_id1,start_id2)
				output_filename_1 = '%s/%s.%d_%d.pre1.txt'%(output_file_path,filename_prefix_save,start_id1,start_id2)
				# df_gene_peak_query_2.to_csv(output_filename_1,sep='\t',float_format='%.5f')
				df_gene_peak_query_2.to_csv(output_filename_1,sep='\t',float_format=float_format)
				print('df_gene_peak_query_2, feature_query_vec_interval ',df_gene_peak_query_2.shape,feature_query_num1,i1,start_id1,start_id2)
			
			## return df_gene_peak_query by each interval
			if query_mode>0:
				list_query1.append(df_gene_peak_query_2)

		return list_query1

	## ====================================================
	# estimate the type of peak-TF link, peak-gene link, and gene-TF link in peak-TF-gene associations
	# adjust the weights of score 1 and score 2 based on the link types
	def test_query_tf_peak_gene_pair_link_type(self,gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],df_gene_peak_query=[],df_gene_peak_tf_query=[],column_idvec=[],
												motif_data=[],peak_read=[],rna_exprs=[],reset_index=True,flag_annot_1=1,type_query=0,
												save_mode=1,output_file_path='',output_filename='',filename_annot='',verbose=0,select_config={}):

		"""
		estimate the type of peak-TF link, peak-gene link, and gene-TF link in peak-TF-gene associations;
		adjust the weights of score 1 (TF-(peak,gene) score) and score 2 ((peak,TF)-gene score) based on the link types;
		:param gene_query_vec: (array or list) the target genes
		:param peak_query_vec: (array) the ATAC-seq peak loci for which to estimate peak-TF-gene links
		:param motif_query_vec: (array) TF names
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param df_gene_peak_tf_query: (dataframe) annotations of peak-TF-gene links
		:param column_idvec: (array or list) columns representing TF, peak, and gene name or index in the peak-TF-gene link annotation dataframe
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param flag_annot_1: indicator of whether to query additional attributes of peak-TF-gene associations from extenal data
		:param type_query: indicator of which set of thresholds to use to estimate the link types for peak-TF-gene associations
		:param reset_index: indicator of whether to reset the index of the dataframe
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dataframe containing the weights of score 1 and score 2 and the estimated peak-TF, peak-gene, and gene-TF link type for each candidate peak-TF-gene link	
		"""

		df_query_1 = df_gene_peak_tf_query
		# from .utility_1 import test_query_index
		if len(column_idvec)==0:
			column_idvec = ['motif_id','peak_id','gene_id']
		if reset_index==True:
			df_query_1.index = test_query_index(df_query_1,column_vec=column_idvec)
		
		field_link_query1, field_link_query2 = [], []
		if 'field_link_query1' in select_config:
			field_link_query1 = select_config['field_link_query1']
		else:
			field_link_query1 = ['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak','gene_tf_corr']

		if 'field_link_query2' in select_config:
			field_link_query2 = select_config['field_link_query2']
		else:
			column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
			# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
			if 'column_pval_cond' in select_config:
				column_pval_cond = select_config['column_pval_cond']
			field_link_query2 = ['peak_tf_pval_corrected','peak_gene_corr_pval',column_pval_cond,'gene_tf_pval_corrected']

		field_link_query_1 = field_link_query1 + field_link_query2
		column_motif_1 = 'motif_score_log_normalize_bound'
		column_motif_2 = 'score_accessibility'
		field_motif_score = [column_motif_1,column_motif_2]
		# select_config.update({'field_link_query1':field_link_query1,'field_link_query2':field_link_query2})
		
		df_link_query_1 = df_query_1
		if 'flag_annot_link_type' in select_config:
			flag_annot_1 = select_config['flag_annot_link_type']

		if flag_annot_1>0:
			flag_annot1=0
			df_1 = df_query_1
			column_vec = df_1.columns
			
			# there are columns not included in the current dataframe
			t_columns_1 = pd.Index(field_link_query_1).difference(column_vec,sort=False)
			t_columns_2 = pd.Index(field_motif_score).difference(column_vec,sort=False)
			df_list1 = [df_1]
			print('query annotations: ',t_columns_1,t_columns_2)
			
			column_vec_1 = []
			if len(t_columns_1)>0:
				flag_annot1 = 1
				# query correlation and p-value
				if 'filename_annot1' in select_config:
					filename_annot1 = select_config['filename_annot1']
					df_2 = pd.read_csv(filename_annot1,index_col=False,sep='\t')
					print('df_2: ',df_2.shape)
					print(filename_annot1)
					print(df_2.columns)
					print(df_2[0:2])
					# df_list1 = [df_1,df_2]
					df_list1 = df_list1+[df_2]
					column_vec_1.append(t_columns_1)
				else:
					print('please provide annotation file')
					return
				
			if len(t_columns_2)>0:
				flag_annot1 = 1
				# query motif score annotation
				if 'filename_motif_score' in select_config:
					filename_annot2 = select_config['filename_motif_score']
					df_3 = pd.read_csv(filename_annot2,index_col=False,sep='\t')
					print('df_3: ',df_3.shape)
					print(filename_annot2)
					print(df_3.columns)
					print(df_3[0:2])
					df_list1 = df_list1+[df_3]
					column_vec_1.append(t_columns_2)
				else:
					print('please provide annotation file')
					return

			if flag_annot1>0:
				column_idvec_1 = column_idvec
				# column_vec_1 = [[column_pval_cond]]
				df_1.index = test_query_index(df_1,column_vec=column_idvec)
				# copy specified columns from the other dataframes to the first dataframe
				df_link_query_1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec_1,column_vec=column_vec_1,
																df_list=df_list1,type_id_1=0,type_id_2=0,reset_index=False,select_config=select_config)

		column_peak_tf_corr, column_peak_gene_corr, column_query_cond, column_gene_tf_corr = field_link_query1
		column_peak_tf_pval, column_peak_gene_pval, column_pval_cond, column_gene_tf_pval = field_link_query2

		df_query_1 = df_link_query_1
		peak_tf_corr, gene_tf_corr_peak, peak_gene_corr_ = df_query_1[column_peak_tf_corr], df_query_1[column_query_cond], df_query_1[column_peak_gene_corr]
		peak_tf_pval_corrected, gene_tf_corr_peak_pval_corrected, peak_gene_corr_pval = df_query_1[column_peak_tf_pval], df_query_1[column_pval_cond], df_query_1[column_peak_gene_pval]
		gene_tf_corr_, gene_tf_corr_pval_corrected = df_query_1[column_gene_tf_corr], df_query_1[column_gene_tf_pval]

		flag_query1=1
		if flag_query1>0:
			if not ('config_link_type' in select_config):
				# thresh_corr_1, thresh_pval_1 = 0.05, 0.1 # the previous parameter
				thresh_corr_1, thresh_pval_1 = 0.1, 0.1  # the alternative parameter to use; use relatively high threshold for negative peaks
				thresh_corr_2, thresh_pval_2 = 0.1, 0.05 # stricter p-value threshold for negative-correlated peaks
				thresh_corr_3, thresh_pval_3 = 0.15, 1
				thresh_motif_score_neg_1, thresh_motif_score_neg_2 = 0.80, 0.90 # higher threshold for regression mechanism
				# thresh_score_accessibility = 0.1
				thresh_score_accessibility = 0.25
				thresh_list_query = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2],[thresh_corr_3, thresh_pval_3]]

				config_link_type = {'thresh_list_query':thresh_list_query,
										'thresh_motif_score_neg_1':thresh_motif_score_neg_1,
										'thresh_motif_score_neg_2':thresh_motif_score_neg_2,
										'thresh_score_accessibility':thresh_score_accessibility}
				select_config.update({'config_link_type':config_link_type})
			else:
				config_link_type = select_config['config_link_type']
				thresh_list_query = config_link_type['thresh_list_query']
				thresh_corr_1, thresh_pval_1 = thresh_list_query[0]
				thresh_corr_2, thresh_pval_2 = thresh_list_query[1]
				thresh_corr_3, thresh_pval_3 = thresh_list_query[2]
				thresh_motif_score_neg_1 ,thresh_motif_score_neg_2 = config_link_type['thresh_motif_score_neg_1'], config_link_type['thresh_motif_score_neg_2']
				thresh_score_accessibility = config_link_type['thresh_score_accessibility']

			print('config_link_type: ',config_link_type)

			# column_1 = 'motif_score_normalize_bound'
			# column_2 = 'score_accessibility'
			motif_score_query = df_query_1['motif_score_log_normalize_bound']
			id_motif_score_1 = (motif_score_query>thresh_motif_score_neg_1)
			id_motif_score_2 = (motif_score_query>thresh_motif_score_neg_2) # we use id_motif_score_2
			id_score_accessibility = (df_query_1['score_accessibility']>thresh_score_accessibility)
			
			if ('field_link_1' in select_config):
				field_link_1 = select_config['field_link_1']
				t_columns = field_link_1
			else:
				t_columns = ['peak_gene_link','gene_tf_link','peak_tf_link']
			
			query_num1 = len(t_columns)
			list1 = [[peak_gene_corr_,peak_gene_corr_pval],[gene_tf_corr_peak,gene_tf_corr_peak_pval_corrected],[peak_tf_corr,peak_tf_pval_corrected]]
			dict1 = dict(zip(t_columns,list1))

			df_query_1.loc[:,t_columns] = 0,0,0
			for i1 in range(query_num1):
				column_1 = t_columns[i1]
				corr_query, pval_query = dict1[column_1]
				id1_query_pval = (pval_query<thresh_pval_1)
				id1_query_1 = id1_query_pval&(corr_query>thresh_corr_1)
				id1_query_2 = id1_query_pval&(corr_query<-thresh_corr_1)

				id1_query_1 = id1_query_1|(corr_query>thresh_corr_3) # only query correlation, without threshold on p-value
				id1_query_2 = id1_query_2|(corr_query<-thresh_corr_3) # only query correlation, without threshold on p-value
				df_query_1.loc[id1_query_1,column_1] = 1
				df_query_1.loc[id1_query_2,column_1] = -1
				df_query_1[column_1] = np.int32(df_query_1[column_1])
				print('column_1, id1_query1, id1_query2 ',column_1,i1,np.sum(id1_query_1),np.sum(id1_query_2))

			# lambda1 = 0.5
			# lambda2 = 1-lambda1
			lambda_gene_peak = 0.5 # lambda1: peak-gene link query
			lambda_gene_tf_cond2 = 1-lambda_gene_peak # lambda2: gene-tf link query
			lambda_peak_tf = 0.5
			lambda_gene_tf_cond = 1-lambda_peak_tf
			peak_tf_link, gene_tf_link, peak_gene_link = df_query_1['peak_tf_link'], df_query_1['gene_tf_link'], df_query_1['peak_gene_link']
			
			t_columns = ['peak_gene_link','gene_tf_link','peak_tf_link','gene_tf_link_1']
			query_num1 = len(t_columns)
			# list1 = [[peak_gene_corr_,peak_gene_corr_pval],[gene_tf_corr_peak,gene_tf_corr_peak_pval_corrected],[peak_tf_corr,peak_tf_pval_corrected]]
			list1 = list1+[[gene_tf_corr_, gene_tf_corr_pval_corrected]]
			dict1 = dict(zip(t_columns,list1))

			list2 = []
			for i1 in range(query_num1):
				column_1 = t_columns[i1]
				corr_query, pval_query = dict1[column_1]
				id1_query_pval = (pval_query<thresh_pval_2)
				id1_query_1 = id1_query_pval&(corr_query>thresh_corr_2)
				id1_query_2 = id1_query_pval&(corr_query<(-thresh_corr_2))

				id1_query_1 = id1_query_1|(corr_query>thresh_corr_3) # only query correlation, without threshold on p-value
				id1_query_2 = id1_query_2|(corr_query<-thresh_corr_3) # only query correlation, without threshold on p-value
				list2.append([id1_query_1,id1_query_2])

			# gene-TF expression correlation not conditioned on peak accessibility
			# group: thresh 1: p-value threshold 1
			id_gene_tf_corr_neg_thresh1 = ((gene_tf_corr_<(-thresh_corr_2))&(gene_tf_corr_pval_corrected<thresh_pval_1))	# use higher threshold
			id_gene_tf_corr_pos_thresh1 = (gene_tf_corr_>thresh_corr_2)&(gene_tf_corr_pval_corrected<thresh_pval_1)

			# gene-TF epxression correlation conditioned on peak accessibility
			id_gene_tf_corr_peak_neg_thresh1 = (gene_tf_corr_peak<(-thresh_corr_2))&(gene_tf_corr_peak_pval_corrected<thresh_pval_1)
			id_gene_tf_corr_peak_pos_thresh1 = (gene_tf_corr_peak>thresh_corr_2)&(gene_tf_corr_peak_pval_corrected<thresh_pval_1)

			# group: thresh 2: p-value threshold 2 (stricter threshold)
			id_gene_tf_corr_pos_thresh2, id_gene_tf_corr_neg_thresh2 = list2[3]

			id_gene_tf_corr_peak_pos_thresh2, id_gene_tf_corr_peak_neg_thresh2 = list2[1]

			# peak-TF correlation 
			id_peak_tf_corr_pos_thresh2, id_peak_tf_corr_neg_thresh2 = list2[2]

			# peak-gene correlation
			id_peak_gene_pos_thresh2, id_peak_gene_neg_thresh2 = list2[0]

			list_pre1 = [id_gene_tf_corr_pos_thresh2, id_gene_tf_corr_pos_thresh1]
			list_pre2 = [id_gene_tf_corr_neg_thresh2, id_gene_tf_corr_neg_thresh1]
			id_gene_tf_corr_pos_thresh_query = list_pre1[type_query]
			id_gene_tf_corr_neg_thresh_query = list_pre2[type_query]
			# print('id_gene_tf_corr_pos_thresh, id_gene_tf_corr_neg_thresh: ',id_gene_tf_corr_pos_thresh_query,id_gene_tf_corr_neg_thresh_query)
			
			# repression with peak-TF correlation above zero
			id1 = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link<0)	# neg-pos: repression (negative peak, positive peak-tf correlation); the previous threshold
			id1 = (id1&id_gene_tf_corr_neg_thresh_query)	# change the threshold to be stricter
			
			# repression with peak-TF correlation above zero but not significant partial gene-TF correlation conditioned on peak accessibility
			# id1_1 = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link==0)&(gene_tf_corr_<-thresh_corr_2)&(gene_tf_corr_pval<thresh_pval_1)	# neg-pos: repression (negative peak, positive peak-tf correlation)
			# id1_1 = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link==0)&(id_gene_tf_corr_neg_thresh1)	# neg-pos: repression (negative peak, positive peak-tf correlation)
			id1_1 = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link==0)&(id_gene_tf_corr_neg_thresh_query)	# neg-pos: repression (negative peak, positive peak-tf correlation)

			id1_2 = (id1|id1_1)
			df_query_1.loc[id1_2,'lambda_gene_peak'] = -lambda_gene_peak
			df_query_1.loc[id1_2,'lambda_gene_tf_cond2'] = -lambda_gene_tf_cond2
			df_query_1.loc[id1_2,'lambda_gene_tf_cond'] = -lambda_gene_tf_cond

			# repression with peak-TF correlation under zero
			id2 = (peak_tf_link<0)&(peak_gene_link>0)&(gene_tf_link>0)	# contradiction
			df_query_1.loc[id2,'lambda_gene_tf_cond2'] = 0
			df_query_1.loc[id2,'lambda_gene_tf_cond'] = 0

			# repression with peak-TF correlation under zero;
			id_2_ori = (peak_tf_link<0)&(peak_gene_link>=0)&(gene_tf_link<0)	# pos-neg: repression (positive peak accessibility-gene expr. correlation, negative peak accessibility-tf expr. correlation)
			id_link_2 = (id_2_ori&id_motif_score_2&id_score_accessibility)	# use higher threshold
			
			id_2 = (id_link_2&id_peak_tf_corr_neg_thresh2&id_gene_tf_corr_peak_neg_thresh2&id_gene_tf_corr_neg_thresh_query) # change the threshold to be stricter
			df_query_1.loc[id_2,'lambda_gene_tf_cond2'] = -lambda_gene_tf_cond2
			df_query_1.loc[id_2,'lambda_peak_tf'] = -lambda_peak_tf
			df_query_1.loc[id_2,'lambda_gene_tf_cond'] = -lambda_gene_tf_cond

			# up-regulation with peak-TF correlation under zero
			# the group may be of lower probability
			id3_ori = (peak_tf_link<0)&(peak_gene_link<0)&(gene_tf_link>0)	# neg-neg: activation (negative peak, negative peak-tf correlation)
			
			id_link_3 = id3_ori&(id_motif_score_2)&(id_score_accessibility)	# use higher threshold; the previous threshold
			id3 = (id_link_3&id_peak_tf_corr_neg_thresh2&id_peak_gene_neg_thresh2&id_gene_tf_corr_peak_pos_thresh2&id_gene_tf_corr_pos_thresh_query)
			df_query_1.loc[id3,'lambda_gene_peak'] = -lambda_gene_peak
			df_query_1.loc[id3,'lambda_peak_tf'] = -lambda_peak_tf

			# up-regulation with peak-TF correlation above zero but peak-gene correlation under zero
			# the peak may be linked with other gene query
			id5_ori = (peak_tf_link>0)&(peak_gene_link<0)&(gene_tf_link>0)	# pos-neg: contraction (negative peak, positive peak-tf correlation, positive tf-gene correlation)
			# id5 = (id5_ori&id_peak_tf_corr_pos_thresh2&id_peak_gene_neg_thresh2&id_gene_tf_corr_peak_pos_thresh2&id_gene_tf_corr_pos_thresh1)
			id5 = (id5_ori&id_peak_tf_corr_pos_thresh2&id_peak_gene_neg_thresh2&id_gene_tf_corr_peak_pos_thresh2&id_gene_tf_corr_pos_thresh_query)

			df_query_1.loc[id5,'lambda_gene_tf_cond2'] = 0
			df_query_1.loc[id5,'lambda_gene_tf_cond'] = 0

			# the groups of peak-gene links that do not use the default lambda
			list_query1 = [id1_2,id2,id_2,id3,id5]
			query_num1 = len(list_query1)
			# query_id1 = df_query_1.index
			for i2 in range(query_num1):
				id_query = list_query1[i2]
				t_value_1 = np.sum(id_query)
				df_query_1.loc[id_query,'group'] = (i2+1)
				print('group %d: %d'%(i2+1,t_value_1))

			return df_query_1

	## ====================================================
	# find the peak-TF-gene links with discrepancy between the gene-TF expression partial correlation given peak accessibility and the gene-TF expression correlation
	def test_gene_peak_tf_query_compare_1(self,input_filename='',df_gene_peak_query=[],thresh_corr_1=0.30,thresh_corr_2=0.05,
											save_mode=0,output_file_path='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		find the peak-TF-gene links with discrepancy between gene-TF expression partial correlation given peak accessibility and gene-TF expression correlation
		:param input_filename: (str) path of the file which saved peak-TF-gene link annotations
		:param df_gene_peak_query: (dataframe) annotations of peak-TF-gene links
		:param thresh_corr_1: threshold on gene-TF expression partial correlation given peak accessibility
		:param thresh_corr_2: threshold on gene-TF expression correlation
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1. (dataframe) the original annotations of peak-TF-gene links;
				 2. (pandas.Series) the indices of peak-TF-gene links of which the discrepancy between gene-TF expr partial correlation given peak accessibility and gene-TF expr correlation is below threshold;
				 3. (pandas.Series) the indices of peak-TF-gene links of which the discrepancy between gene-TF expr partial correlation given peak accessibility and gene-TF expr correlation is above threshold;

		:return: dictionary containing the latent representation matrix (embedding), loading matrix, and reconstructed marix (if reconstruct>0)
		"""

		if len(df_gene_peak_query)==0:
			# df_gene_peak_query_pre1 = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_gene_peak_query_pre1 = pd.read_csv(input_filename,index_col=False,sep='\t')
		else:
			df_gene_peak_query_pre1 = df_gene_peak_query

		# from .utility_1 import test_query_index
		column_idvec = ['motif_id','peak_id','gene_id']
		df_gene_peak_query_pre1.index = test_query_index(df_gene_peak_query_pre1,column_vec=column_idvec)
		print('peak-TF-gene links, dataframe of size ',df_gene_peak_query_pre1.shape)

		thresh_corr_query_1, thresh_corr_compare_1 = thresh_corr_1, thresh_corr_2

		column_query_cond = 'gene_tf_corr_peak'
		column_gene_tf_corr = 'gene_tf_corr'
		if 'column_query_cond' in select_config:
			column_query_cond = select_config['column_query_cond']

		if 'column_gene_tf_corr' in select_config:
			column_gene_tf_corr = select_config['column_gene_tf_corr']

		id1 = (df_gene_peak_query_pre1[column_query_cond].abs()>thresh_corr_query_1)  # gene-TF expression partial correlation given peak accessibility above threshold
		id2 = (df_gene_peak_query_pre1[column_gene_tf_corr].abs()<thresh_corr_compare_1)  # gene-TF expression correlation below threshold
		id3 = (id1&id2)
		id_pre1 = (~id3)
		df_query_1 = df_gene_peak_query_pre1.loc[id_pre1,:] # peak-TF-gene links of which the discrepancy between gene-TF expression partial correlation given peak accessibility and gene-TF expression correlation is below threshold
		df_query_2 = df_gene_peak_query_pre1.loc[id3,:] # peak-TF-gene links of which the discrepancy between gene-TF expression partial correlation given peak accessibility and gene-TF expression correlation is above threshold
		query_id_ori = df_gene_peak_query_pre1.index
		query_id_1 = query_id_ori[id_pre1]
		query_id_2 = query_id_ori[id3]

		if (save_mode>0) and (output_file_path!=''):
			filename_annot_save = '%s_%s'%(thresh_corr_1,thresh_corr_2)
			output_filename_1 = '%s/%s.%s.subset1.txt'%(output_file_path,filename_prefix_save,filename_annot_save)
			df_query_1.index = np.asarray(df_query_1['gene_id'])
			
			output_filename_2 = '%s/%s.%s.subset2.txt'%(output_file_path,filename_prefix_save,filename_annot_save)
			df_query_2.index = np.asarray(df_query_2['gene_id'])
			df_query_2.to_csv(output_filename_2,sep='\t',float_format='%.5f')
			print('retained peak-TF-gene links, dataframe of size ',df_query_1.shape)
			print('peak-TF-gene links with discrepancy between gene-TF expression partial correlation given peak accessibility and gene-TF expression correlation, dataframe of size ',df_query_2.shape)

		# return df_query_1, df_query_2
		return df_gene_peak_query_pre1, query_id_1, query_id_2

	## ====================================================
	# parameter configuration for calculating peak-TF-gene association scores and selecting peak-TF-gene links
	def test_query_score_config_1(self,column_pval_cond='',thresh_corr_1=0.1,thresh_pval_1=0.1,overwrite=False,flag_config_1=1,flag_config_2=1,save_mode=1,verbose=0,select_config={}):

		"""
		parameter configuration for calculating peak-TF-gene association scores and selecting peak-TF-gene links
		:param column_pval_cond: column representing gene-TF expression partial correlation given peak accessibility in the peak-TF-gene link annotation dataframe
		:param thresh_corr_1: threshold on the correlation for specific type of feature links to estimate the link type
		:param thresh_pval_1: threshold on the adjusted p-value of the specific type of correlation to estimate the link type
		:param overwrite: indicator of whether to overwrite the current parameters in the parameter dictionary
		:param flag_cconfig_1: indicator of whether to provide parameters for peak-TF-gene association score calculation
		:param flag_cconfig_2: indicator of whether to provide parameters for peak-TF-gene link selection
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""

		if flag_config_1>0:
			filename_prefix_default_1 = select_config['filename_prefix_default_1']
			filename_prefix_cond = select_config['filename_prefix_cond']
			# filename_prefix_score = '%s.pcorr_query1'%(filename_prefix_default_1)
			filename_prefix_score = '%s.pcorr_query1'%(filename_prefix_cond)
			filename_annot_score_1 = 'annot2.init.1'
			filename_annot_score_2 = 'annot2.init.query1'
			select_config.update({'filename_prefix_score':filename_prefix_score,'filename_annot_score_1':filename_annot_score_1,'filename_annot_score_2':filename_annot_score_2})

			correlation_type = 'spearmanr'
			column_idvec = ['motif_id','peak_id','gene_id']
			column_gene_tf_corr_peak =  ['gene_tf_corr_peak','gene_tf_corr_peak_pval','gene_tf_corr_peak_pval_corrected1']
			# thresh_insilco_ChIP_seq = 0.1
			flag_save_text = 1

			field_query = ['column_idvec','correlation_type','column_gene_tf_corr_peak','flag_save_text']
			list_1 = [column_idvec,correlation_type,column_gene_tf_corr_peak,flag_save_text]
			field_num1 = len(field_query)
			for i1 in range(field_num1):
				field1 = field_query[i1]
				if (not (field1 in select_config)) or (overwrite==True):
					select_config.update({field1:list_1[i1]})

			field_query = ['column_peak_tf_corr','column_peak_gene_corr','column_query_cond','column_gene_tf_corr','column_score_1','column_score_2']
			column_score_1, column_score_2 = 'score_pred1', 'score_pred2'
			column_vec_1 = ['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak','gene_tf_corr']
			# list1 = ['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak','gene_tf_corr',column_score_1,column_score_2]
			list1 = column_vec_1+[column_score_1,column_score_2]

			if column_pval_cond=='':
				# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
				column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
			
			field_query_2 = ['column_peak_tf_pval','column_peak_gene_pval','column_pval_cond','column_gene_tf_pval']
			list2 = ['peak_tf_pval_corrected','peak_gene_corr_pval',column_pval_cond,'gene_tf_pval_corrected']

			field_query = field_query + field_query_2
			list1 = list1 + list2
			query_num1 = len(field_query)
			for (field_id,query_value) in zip(field_query,list1):
				select_config.update({field_id:query_value})

			column_score_query = [column_score_1,column_score_2,'score_pred_combine']
			select_config.update({'column_score_query':column_score_query})

			field_link_1 = ['peak_gene_link','gene_tf_link','peak_tf_link']
			field_link_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']
			select_config.update({'field_link_1':field_link_1,'field_link_2':field_link_2})

			column_idvec = ['motif_id','peak_id','gene_id']
			# column_score_query_1 = column_idvec + ['score_1','score_combine_1','score_pred1_correlation','score_pred1','score_pred1_1','score_pred2','score_pred_combine']
			
			column_score_query_pre1 = column_idvec + ['score_1','score_combine_1','score_pred1_correlation','score_pred1_1']+column_score_query
			column_annot_1 = ['feature1_score1_quantile', 'feature1_score2_quantile','feature2_score1_quantile','peak_tf_corr_thresh1','peak_gene_corr_thresh1','gene_tf_corr_peak_thresh1']
			
			column_score_query_1 = column_idvec + ['score_1','score_pred1_correlation','score_pred1_1'] + column_score_query + column_annot_1
			select_config.update({'column_idvec':column_idvec,
									'column_score_query_pre1':column_score_query_pre1,
									'column_score_query_1':column_score_query_1})

			if (not ('config_link_type' in select_config)) or (overwrite==True):
				# thresh_corr_1, thresh_pval_1 = 0.05, 0.1 # the previous parameter
				thresh_corr_1, thresh_pval_1 = 0.1, 0.1  # the alternative parameter to use; use relatively high threshold for negative peaks
				thresh_corr_2, thresh_pval_2 = 0.1, 0.05
				thresh_corr_3, thresh_pval_3 = 0.15, 1
				thresh_motif_score_neg_1, thresh_motif_score_neg_2 = 0.80, 0.90 # higher threshold for regression mechanism
				# thresh_score_accessibility = 0.1
				thresh_score_accessibility = 0.25
				thresh_list_query = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2],[thresh_corr_3, thresh_pval_3]]

				config_link_type = {'thresh_list_query':thresh_list_query,
									'thresh_motif_score_neg_1':thresh_motif_score_neg_1,
									'thresh_motif_score_neg_2':thresh_motif_score_neg_2,
									'thresh_score_accessibility':thresh_score_accessibility}

				select_config.update({'config_link_type':config_link_type})

		if flag_config_2>0:
			if not ('thresh_score_query_1' in select_config):
				thresh_vec_1 = [[0.10,0.05],[0.05,0.10]]
				select_config.update({'thresh_score_query_1':thresh_vec_1})
			
			thresh_gene_tf_corr_peak = 0.30
			thresh_gene_tf_corr_ = 0.05
			if not ('thresh_gene_tf_corr_compare' in select_config):
				thresh_gene_tf_corr_compare = [thresh_gene_tf_corr_peak,thresh_gene_tf_corr_]
				select_config.update({'thresh_gene_tf_corr_compare':thresh_gene_tf_corr_compare})

			# print(select_config['thresh_score_query_1'])
			# print(select_config['thresh_gene_tf_corr_compare'])
			
			column_label_1 = 'feature1_score1_quantile'
			column_label_2 = 'feature1_score2_quantile'
			column_label_3 = 'feature2_score1_quantile'
			select_config.update({'column_quantile_1':column_label_1,'column_quantile_2':column_label_2,
									'column_quantile_feature2':column_label_3})

			if not ('thresh_vec_score_2' in select_config):
				thresh_corr_1 = 0.30
				thresh_corr_2 = 0.50
				# thresh_pval_1, thresh_pval_2 = 0.25, 1
				thresh_pval_1, thresh_pval_2, thresh_pval_3 = 0.05, 0.10, 1
				
				# thresh_vec_1 = [[thresh_corr_1,thresh_pval_2],[thresh_corr_2,thresh_pval_2]]	# original
				thresh_vec_1 = [[thresh_corr_1,thresh_pval_2],[thresh_corr_2,thresh_pval_3]]	# updated
				
				# thresh_vec_2 = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2]]	# original
				thresh_vec_2 = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_3]]	# updated
				
				# thresh_vec_3 = [[thresh_corr_1,thresh_pval_2],[thresh_corr_2,thresh_pval_2]]	# original
				thresh_vec_3 = [[thresh_corr_1,thresh_pval_2],[thresh_corr_2,thresh_pval_3]]	# updated
				
				thresh_vec = [thresh_vec_1,thresh_vec_2,thresh_vec_3]
				select_config.update({'thresh_vec_score_2':thresh_vec})

			thresh_score_quantile = 0.95
			select_config.update({'thresh_score_quantile':thresh_score_quantile})

		self.select_config = select_config

		return select_config

	## ====================================================
	# parameter configuration for estimating the feature link type
	def test_query_score_config_2(self,thresh_query_1=[],thresh_query_2=[],overwrite=False,save_mode=1,verbose=0,select_config={}):

		"""
		parameter configuration for estimating the feature link type
		:param thresh_query_1: (list) thresholds on correlation or partial correlation and adjusted or empirical p-value for different types of links (peak-TF link, peak-gene link, peak-TF-gene link) to estimate the link type
		:param thresh_query_2: (list) thresholds on motif scores from motif scanning and peak accessibility-based scores to estimate the link type
		:param overwrite: indicator of whether to overwrite the current parameters in the parameter dictionary for link type estimation
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""

		if len(thresh_query_1)==0:
			thresh_corr_1, thresh_pval_1 = 0.1, 0.1  # the alternative parameter to use; use relatively high threshold for negative peaks
			thresh_corr_2, thresh_pval_2 = 0.1, 0.05
			thresh_corr_3, thresh_pval_3 = 0.15, 1
			thresh_list_query = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2],[thresh_corr_3,thresh_pval_3]]
		else:
			thresh_list_query = thresh_query_1

		if len(thresh_query_2)==0:
			thresh_motif_score_neg_1, thresh_motif_score_neg_2 = 0.80, 0.90 # higher threshold for regression mechanism
			thresh_score_accessibility = 0.25
			thresh_query_2 = [thresh_motif_score_neg_1,thresh_motif_score_neg_2,thresh_score_accessibility]
		else:
			thresh_motif_score_neg_1, thresh_motif_score_neg_2 = thresh_query_2[0:2]
			thresh_score_accessibility = thresh_query_2[2]

		column_1 = 'config_link_type'
		if (not (column_1 in select_config)) or (overwrite==True):
			config_link_type = {'thresh_list_query':thresh_list_query,
									'thresh_motif_score_neg_1':thresh_motif_score_neg_1,
									'thresh_motif_score_neg_2':thresh_motif_score_neg_2,
									'thresh_score_accessibility':thresh_score_accessibility}
			select_config.update({column_1:config_link_type})

		return select_config

	## ====================================================
	# compute peak-TF-gene association scores
	def test_query_feature_score_compute_1(self,df_feature_link=[],input_filename='',overwrite=False,iter_mode=1,
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		compute peak-TF-gene association scores
		:param df_feature_link: (dataframe) annotations of peak-TF-gene links
		:param input_filename: path of the file which saved the peak-TF-gene links
		:param overwrite: indicator of whether to overwrite the current parameters to estimate feature association types (repression or non-repression)
		:param iter_mode: indicator of whether the peak-TF-gene link scores are computed in batch mode
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) peak-TF-gene link annotations including computed peak-TF-gene association scores
		"""

		field_query = ['thresh_list_query','thresh_motif_score_neg_1','thresh_motif_score_neg_2','thresh_score_accessibility']

		if ('config_link_type' in select_config) and (overwrite==False):
			config_link_type = select_config['config_link_type']
			list1 = [config_link_type[field_id] for field_id in field_query]
			thresh_list_query,thresh_motif_score_neg_1,thresh_motif_score_neg_2,thresh_score_accessibility = list1
		else:
			# thresh_corr_1, thresh_pval_1 = 0.05, 0.1 # the previous parameter
			thresh_corr_1, thresh_pval_1 = 0.1, 0.1  # the alternative parameter to use; use relatively high threshold for negative peaks
			thresh_corr_2, thresh_pval_2 = 0.1, 0.05
			thresh_corr_3, thresh_pval_3 = 0.15, 1
			thresh_motif_score_neg_1, thresh_motif_score_neg_2 = 0.80, 0.90 # higher threshold for regression mechanism
			
			# thresh_score_accessibility = 0.1
			thresh_score_accessibility = 0.25
			thresh_list_query = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2],[thresh_corr_3, thresh_pval_3]]

			list1 = [thresh_list_query,thresh_motif_score_neg_1,thresh_motif_score_neg_2,thresh_score_accessibility]
			
			config_link_type = dict(zip(field_query,list1))
			select_config.update({'config_link_type':config_link_type})

		column_idvec = ['motif_id','peak_id','gene_id']
		if 'column_idvec' in select_config:
			column_idvec = select_config['column_idvec']

		column_pval_cond = select_config['column_pval_cond']
		flag_query_1=1
		if flag_query_1>0:
			list_query1 = []
			flag_link_type = 1
			flag_compute = 1
			# from .utility_1 import test_query_index, test_column_query_1
			from .utility_1 import test_column_query_1
			if len(df_feature_link)==0:
				id1 = 0
				if input_filename!='':
					id1 = (os.path.exists(input_filename)==True)

				if (iter_mode>0) or (id1<1):
					input_filename_list1 = select_config['filename_list_score']
					input_filename_list2 = select_config['filename_annot_list']
					input_filename_list3 = select_config['filename_link_list']
					input_filename_list_motif = select_config['filename_motif_score_list']

					query_num1 = len(input_filename_list1)
					for i1 in range(query_num1):
						input_filename_1 = input_filename_list1[i1]
						df_1 = pd.read_csv(input_filename_1,index_col=False,sep='\t')

						filename_annot1, filename_annot2 = input_filename_list2[i1], input_filename_list_motif[i1]
						filename_link = input_filename_list3[i1]

						df_link_query_1 = df_1
						df_feature_link = df_link_query_1
						print('peak-TF-gene links, dataframe of size ',df_link_query_1.shape)
						print('input_filename: %s'%(input_filename_1))
						print('data preview:\n',df_link_query_1[0:2])

						select_config.update({'filename_annot1':filename_annot1,'filename_motif_score':filename_annot2,
												'filename_link':filename_link})				
						retrieve_mode = 0
						flag_annot_1=1
						# compute peak-TF-gene association scores
						df_link_query_pre1 = self.test_gene_peak_tf_query_score_compute_unit_1(df_feature_link=df_link_query_1,
																								flag_link_type=flag_link_type,
																								flag_compute=flag_compute,
																								flag_annot_1=flag_annot_1,
																								retrieve_mode=retrieve_mode,
																								verbose=verbose,select_config=select_config)

						list_query1.append(df_link_query_pre1)

						if save_mode>0:
							b = input_filename_1.find('.txt')
							extension = input_filename_1[b:]
							# output_filename = input_filename_1[0:b]+'.recompute.txt'
							output_filename = input_filename_1[0:b]+'.recompute'+extension
							column_score_query_pre1 = select_config['column_score_query_pre1']

							# retrieve the columns of score estimation and subset of annotations
							column_idvec = select_config['column_idvec']
							# column_vec_1 = list(column_idvec)+list(column_score_query2)
							column_vec_1 = pd.Index(column_score_query_pre1).union(column_idvec,sort=False)

							df_link_query_pre2 = df_link_query_pre1.loc[:,column_vec_1]
							float_format = '%.5f'
							df_link_query_pre2.to_csv(output_filename,index=False,sep='\t',float_format=float_format)

							if ('field_link_1' in select_config):
								field_link_1 = select_config['field_link_1']
							else:
								field_link_1 = ['peak_gene_link','gene_tf_link','peak_tf_link']

							if ('field_link_2' in select_config):
								field_link_2 = select_config['field_link_2']
							else:
								field_link_2 = ['lambda_gene_peak','lambda_gene_tf_cond2','lambda_peak_tf','lambda_gene_tf_cond']
							
							# retrieve the columns of link type annotation
							column_vec_2 = list(column_idvec) + field_link_1 + field_link_2 + ['group']
							df_link_query_pre3 = df_link_query_pre1.loc[:,column_vec_2]

					df_feature_link = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)	
				else:
					df_feature_link = pd.read_csv(input_filename,index_col=0,sep='\t')
			
			if (iter_mode==0) and (len(df_feature_link)>0):
				# compute peak-TF-gene association scores
				df_feature_link = self.test_gene_peak_tf_query_score_compute_unit_1(df_feature_link=df_feature_link,
																					flag_link_type=flag_link_type,
																					flag_compute=flag_compute,
																					verbose=verbose,select_config=select_config)

			return df_feature_link

	## ====================================================
	# query peak-TF-gene link annotations
	# compute peak-TF-gene association scores
	def test_query_feature_score_init_pre1_1(self,data=[],input_filename_list=[],recompute=0,iter_mode=1,load_mode=1,save_mode=1,verbose=0,select_config={}):

		"""
		query peak-TF-gene link annotations;
		combine peak-TF-gene link annotations of different subsets of links if link annotations were generated in batch mode;
		compute peak-TF-gene association scores;
		:param data: (dataframe) annotations of peak-TF-gene links
		:param input_filename_list: (list) paths of files which saved annotations of different subsets of peak-TF-gene links
		:param recompute: indictor of whether to recompute the peak-TF-gene assocation scores
		:param iter_mode: indictor of whether the peak-TF-gene link annotations were prepared in batch mode for different subsets of links
		:param load_mode: indictor of whether to load the peak-TF-gene link annotations from saved files
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the combined annotations of peak-TF-gene links
				 2. dictionary containing updated parameters
		"""

		df_feature_link = []
		if load_mode>0:
			# query filename of feature link annotations
			file_path_motif_score = select_config['file_path_motif_score']
			input_file_path = file_path_motif_score
			filename_prefix_save_1 = select_config['filename_prefix_score']
			# filename_prefix_save_1_ori = filename_prefix_save_1
			filename_annot_1_ori = select_config['filename_annot_score_1']
			filename_annot_1 = filename_annot_1_ori
			
			print('filename_prefix_save_1 ',filename_prefix_save_1)
			if len(input_filename_list)==0:
				# input_filename_list = []
				input_filename_list2 = []
				input_filename_list_2 = []
				input_filename_list_3 = []
				input_filename_list_motif = []

				extension = 'txt.gz'
				compression = 'infer'
				if extension in ['txt.gz']:
					compression = 'gzip'

				# if iter_mode>0:
				# 	interval = select_config['feature_score_interval']
				# 	feature_query_num_1 = select_config['feature_query_num']
				# 	iter_num = int(np.ceil(feature_query_num_1/interval))
				# else:
				# 	iter_num = 1

				query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
				interval_query = query_id2-query_id1

				column_1, column_2 = 'feature_score_interval', 'feature_query_num'
				iter_mode_query = 0
				if (column_1 in select_config) and (column_2 in select_config):
					interval = select_config[column_1]
					feature_query_num_1 = select_config[column_2]
					iter_num = int(np.ceil(feature_query_num_1/interval))
					print('feature_query_num: %d, interval: %d'%(feature_query_num_1,interval))
					if iter_num>1:
						iter_mode_query = 1

				print('iter_mode_query, iter_num ',iter_mode_query,iter_num)
				for i1 in range(iter_num):
					if iter_mode_query>0:
						query_id_1 = i1*interval
						query_id_2= (i1+1)*interval
						# query_id_2_ori = query_id_2
						# query_id_2_pre = np.min([feature_query_num_1,query_id_2])
						# query_id_2 = query_id_2_pre
						filename_prefix_save_pre1 = '%s.%d_%d'%(filename_prefix_save_1,query_id_1,query_id_2)
						# filename_prefix_save_pre2 = '%s.%d_%d'%(filename_prefix_save_1_ori,query_id_1,query_id_2)
					else:
						if (query_id1>=0) and (query_id2>query_id1):
							filename_prefix_save_pre1 = '%s.%d_%d'%(filename_prefix_save_1,query_id1,query_id2)
						else:
							filename_prefix_save_pre1 = filename_prefix_save_1
							# filename_prefix_save_pre2 = filename_prefix_save_1_ori

					# input_filename = '%s/%s.%s.txt'%(input_file_path,filename_prefix_save_pre1,filename_annot_1) # prepare for the file
					input_filename = '%s/%s.%s.%s'%(input_file_path,filename_prefix_save_pre1,filename_annot_1,extension) # prepare for the file
					input_filename_list.append(input_filename)
					print(input_filename)

					if os.path.exists(input_filename)==False:
						print('the file does not exist: %s'%(input_filename))
						recompute = 1
					else:
						print('the file exists: %s'%(input_filename))
						# recompute = 0

					# the computed gene-TF expression partial correlation given peak accessibility
					#input_filename_2 = '%s/%s.txt'%(input_file_path,filename_prefix_save_pre2)
					# input_filename_2 = '%s/%s.%s'%(input_file_path,filename_prefix_save_pre2,extension)
					input_filename_2 = '%s/%s.txt'%(input_file_path,filename_prefix_save_pre1)
					input_filename_list2.append(input_filename_2)
					print(input_filename_2)
					print('filename of gene-TF expr partial correlation given peak accessibility: %s'%(input_filename_2))
						
					# the annotation file with the correlation and p-value estimation
					input_filename_3 = '%s/%s.annot1_1.1.%s'%(input_file_path,filename_prefix_save_pre1,extension)
					filename_query = input_filename_3
					print('filename of the correlation values and p-values: %s'%(input_filename_3))
						
					flag_1=0
					if flag_1>0:
						df3 = pd.read_csv(input_filename_3,index_col=False,sep='\t')
						column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
						# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
						if 'column_pval_cond' in select_config:
							column_pval_cond = select_config['column_pval_cond']

						if not (column_pval_cond in df3.columns):
							index_col=False
							df2 = pd.read_csv(input_filename_2,index_col=index_col,sep='\t')

							df_list1 = [df3,df2]
							column_vec_1 = [[column_pval_cond]]
							column_idvec = ['gene_id','peak_id','motif_id']
							# copy specified columns from the other dataframes to the first dataframe
							df_link_query_1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec,
																				column_vec=column_vec_1,
																				df_list=df_list1,
																				type_id_1=2,type_id_2=0,
																				reset_index=False,
																				select_config=select_config)

							output_file_path = input_file_path
							# output_filename = '%s/%s.annot1_1.copy1.txt'%(output_file_path,filename_prefix_save_pre1)
							output_filename = '%s/%s.annot1_1.2.%s'%(output_file_path,filename_prefix_save_pre1,extension)
							df_link_query_1.to_csv(output_filename,index=False,sep='\t')
							filename_query = output_filename
							print('filename_query: ',filename_query)
						
					input_filename_list_2.append(filename_query)

					input_filename_link = '%s/%s.annot2_1.1.%s'%(input_file_path,filename_prefix_save_pre1,extension)
					input_filename_list_3.append(input_filename_link)
					print('filename of the link type: %s'%(input_filename_link))

					# to recompute the link score we need to recompute lambda of link and need motif score annotation
					# input_filename_motif_score = '%s/%s.annot1_3.1.txt'%(input_file_path,filename_prefix_save_pre1)
					input_filename_motif_score = '%s/%s.annot1_3.1.%s'%(input_file_path,filename_prefix_save_pre1,extension)
					if os.path.exists(input_filename_motif_score)==False:
						input_filename_motif_score = '%s/%s.annot1_3.1.txt'%(input_file_path,filename_prefix_save_pre1)
					input_filename_list_motif.append(input_filename_motif_score)
					print('filename of the motif scores: %s'%(input_filename_motif_score))

				select_config.update({'filename_pcorr_list':input_filename_list2,
										'filename_annot_list':input_filename_list_2,
										'filename_link_list':input_filename_list_3,
										'filename_motif_score_list':input_filename_list_motif})

				filename_list_score = input_filename_list
				select_config.update({'filename_list_score':filename_list_score})

		# recompute the feature link score
		if recompute>0:
			# column_score_query
			thresh_corr_1, thresh_pval_1 = 0.1, 0.1  # the alternative parameter to use; use relatively high threshold for negative peaks
			thresh_corr_2, thresh_pval_2 = 0.1, 0.05
			thresh_corr_3, thresh_pval_3 = 0.15, 1
			thresh_motif_score_neg_1, thresh_motif_score_neg_2 = 0.80, 0.90 # higher threshold for regression mechanism
			# thresh_score_accessibility = 0.1
			thresh_score_accessibility = 0.25
			thresh_list_query = [[thresh_corr_1,thresh_pval_1],[thresh_corr_2,thresh_pval_2],[thresh_corr_3, thresh_pval_3]]
			thresh_query_1 = thresh_list_query
			thresh_query_2 = [thresh_motif_score_neg_1,thresh_motif_score_neg_2,thresh_score_accessibility]

			# parameter configuration for feature score computation; parameter configuration for estimating the feature link type
			select_config = self.test_query_score_config_2(thresh_query_1=thresh_query_1,thresh_query_2=thresh_query_2,
															save_mode=1,verbose=verbose,select_config=select_config)

			# compute feature link score
			df_feature_link = self.test_query_feature_score_compute_1(df_feature_link=[],input_filename='',
																		overwrite=False,
																		iter_mode=iter_mode,
																		save_mode=1,
																		output_file_path='',
																		output_filename='',
																		filename_prefix_save='',
																		filename_save_annot='',
																		verbose=verbose,select_config=select_config)

		return df_feature_link, select_config

	## ====================================================
	# perform selection of peak-TF-gene associations
	def test_query_feature_score_init_pre1(self,df_feature_link=[],input_filename_list=[],input_filename='',index_col=0,iter_mode=0,recompute=0,
												flag_score_quantile_1=1,flag_score_query_1=1,flag_compare_thresh1=1,flag_select_pair_1=1,flag_select_feature_1=0,flag_select_feature_2=0,
												flag_select_local=1,flag_select_link_type=0,overwrite=False,input_file_path='',
												save_mode=0,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		perform selection of peak-TF-gene associations
		:param df_feature_link: (dataframe) annotations of peak-TF-gene links
		:param input_filename_list: (list) paths of files which saved the annotations of different subsets of peak-TF-gene links
		:param input_filename: (str) path of the file which saved peak-TF-gene link annotations
		:param index_col: column to use as the row label for data in the dataframe
		:param iter_mode: indicator of whether the peak-TF-gene link scores are computed in batch mode
		:param recompute: indicator of whether to recompute the peak-TF-gene link scores
		:param flag_score_quantile_1: indicator of whether to compute peak-TF-gene association score quantiles
		:param flag_score_query_1: indicator of whether to perform peak-TF-gene link selection
		:param flag_compare_thresh1: indicator of whether to filter the links with discrepancy between gene-TF expression partial correlation given peak accessibility and gene-TF expression correlation above threshold
		:param flag_select_pair_1: indicator of whether to used paired thresholds on score 1 and score 2 to select peak-TF-gene links
		:param flag_select_feature_1: indicator of whether to select links based on quantiles of score 1 (TF binding score) for each TF
		:param flag_select_feature_2: indicator of whether to select links based on quantiles of score 2 for each gene
		:param flag_select_local: indicator of whether to select peak-TF-gene links based on strong peak-TF correlations
		:param flag_select_link_type: indicator of whether to select peak-TF-gene links based on the estimated type of different links (repression or activation)
		:param overwrite: indicator of whether to overwrite the current file of link annotations
		:param input_file_path: the directory to retrieve data from 
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) updated annotations of initial peak-TF-gene links including associations scores and labels of which links are selected by specific thresholds;
				 2. (dataframe) annotations of selected peak-TF-gene links
		"""

		file_save_path = select_config['data_path_save']
		file_save_path_local = select_config['data_path_save_local']
		file_save_path2 = select_config['file_path_motif_score']
		filename_prefix_default = select_config['filename_prefix_default']
		filename_prefix_default_1 = select_config['filename_prefix_default_1']

		column_idvec=['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]
		select_config.update({'column_idvec':column_idvec})

		if not ('column_score_query' in select_config):
			column_score_1, column_score_2 = 'score_pred1', 'score_pred2'
			if 'column_score_1' in select_config:
				column_score_1 = select_config['column_score_1']

			if 'column_score_2' in select_config:
				column_score_2 = select_config['column_score_2']

			column_score_combine = 'score_pred_combine'
			column_score_query1 = [column_score_1,column_score_2,column_score_combine]
			select_config.update({'column_score_query':column_score_query1})
		else:
			column_score_query1 = select_config['column_score_query']

		# column_score_1, column_score_2 = 'score_pred1', 'score_pred2'
		column_score_1, column_score_2 = column_score_query1[0:2]
		select_config.update({'column_score_1':column_score_1,'column_score_2':column_score_2})
		print('column_score_query1: ',column_score_query1)
		print('column_score_1, column_score_2: ',column_score_1,column_score_2)

		# filename_prefix_score = '%s.pcorr_query1'%(filename_prefix_default_1)
		# filename_annot_score_1 = 'annot2.init.1'
		# filename_annot_score_2 = 'annot2.init.query1'
		field_query_pre1 = ['column_peak_tf_corr','column_peak_gene_corr','column_query_cond','column_gene_tf_corr']
		field_num1 = len(field_query_pre1)
		field_query_1 = [select_config[field_id] for field_id in field_query_pre1]

		# column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
		# column_pval_cond = 'gene_tf_corr_peak_pval_corrected2'
		field_query_pre2 = ['column_peak_tf_pval','column_peak_gene_pval','column_pval_cond','column_gene_tf_pval']
		# field_query_2 = ['peak_tf_pval_corrected','peak_gene_pval',column_pval_cond,'gene_tf_pval_corrected']
		field_query_2 = [select_config[field_id] for field_id in field_query_pre2]
		select_config.update({'field_link_query1':field_query_1,'field_link_query2':field_query_2})

		from .utility_1 import test_column_query_1
		load_mode_2 = 0
		input_filename_feature_link = input_filename
		if len(df_feature_link)==0:
			if os.path.exists(input_filename_feature_link)==True:
				df_feature_link = pd.read_csv(input_filename_feature_link,index_col=index_col,sep='\t')
				print('load peak-TF-gene links from %s'%(input_filename_feature_link))
			else:
				load_mode_2 = 1
				# recompute_2 = 0
				recompute_2 = 1
				print('compute peak-TF-gene association scores')
				df_feature_link, select_config = self.test_query_feature_score_init_pre1_1(input_filename_list=[],recompute=recompute_2,
																							iter_mode=iter_mode,
																							load_mode=load_mode_2,
																							save_mode=1,
																							verbose=verbose,select_config=select_config)

			print('initial peak-TF-gene links, dataframe of size ',df_feature_link.shape)
			print('columns: ',np.asarray(df_feature_link.columns))
			print('data preview: ')
			print(df_feature_link[0:2])

		df_score_annot_1 = []
		if flag_score_quantile_1>0:
			overwrite_2 = False
			df_link_query1, select_config = self.test_query_feature_score_quantile_compute_1(df_feature_link=df_feature_link,
																								input_filename_list=input_filename_list,
																								index_col=index_col,
																								column_idvec=column_idvec,
																								iter_mode=iter_mode,
																								overwrite=overwrite_2,
																								save_mode=1,output_file_path='',output_filename='',
																								verbose=verbose,select_config=select_config)
			df_score_annot_1 = df_link_query1

		df_feature_link_pre1 = []
		df_feature_link_pre2 = []
		if flag_score_query_1>0:
			if not ('thresh_score_query_1' in select_config):
				# thresh_vec_1 = [[0.10,0.05],[0.05,0.10]]
				thresh_vec_1 = [[0.10,0],[0,0.10]]
				select_config.update({'thresh_score_query_1':thresh_vec_1})
			else:
				thresh_vec_1 = select_config['thresh_score_query_1']
			print('thresholds for the feature link selection ',thresh_vec_1)

			thresh_gene_tf_corr_peak = 0.30
			thresh_gene_tf_corr_ = 0.05
			# thresh_gene_tf_corr_ = 0.1    # the alternative parameter
			if not ('thresh_gene_tf_corr_compare' in select_config):
				thresh_gene_tf_corr_compare = [thresh_gene_tf_corr_peak,thresh_gene_tf_corr_]
				select_config.update({'thresh_gene_tf_corr_compare':thresh_gene_tf_corr_compare})
			else:
				thresh_gene_tf_corr_compare = select_config['thresh_gene_tf_corr_compare']
				thresh_gene_tf_corr_peak, thresh_gene_tf_corr_ = thresh_gene_tf_corr_compare[0:2]

			column_label_1 = 'feature1_score1_quantile'
			column_label_2 = 'feature1_score2_quantile'
			select_config.update({'column_label_1':column_label_1,'column_label_2':column_label_2})

			flag_query2 = 1
			if flag_query2>0:
				column_1 = 'filename_list_score'
				if not (column_1 in select_config):
					print('query peak-TF-gene link annotations ')
					# query peak-TF-gene link annotations
					# combine the annotations of different subsets of peak-TF-gene links if link annotations were generated in batch mode
					df_feature_link_query1, select_config = self.test_query_feature_score_init_pre1_1(input_filename_list=[],recompute=0,
																										iter_mode=iter_mode,
																										load_mode=1,
																										save_mode=1,
																										verbose=0,select_config=select_config)

				input_filename_list = select_config['filename_list_score']
				# input_filename_list2 = select_config['filename_pcorr_list']
				input_filename_list2 = select_config['filename_annot_list']
				input_filename_list3 = select_config['filename_link_list']
				query_num1 = len(input_filename_list)
				
				save_mode_2 = 1
				filename_prefix_score = select_config['filename_prefix_score']
				filename_prefix_1 = filename_prefix_score
				filename_annot_score_2 = select_config['filename_annot_score_2']
				filename_annot_1 = filename_annot_score_2
				output_file_path = file_save_path2
				list_1 = []
				list_2 = []

				# float_format = '%.5E'
				for i1 in range(query_num1):
					input_filename = input_filename_list[i1]
					df_link_query_pre1 = pd.read_csv(input_filename,index_col=index_col,sep='\t')
					
					print('peak-TF-gene links with scores, dataframe of size ',df_link_query_pre1.shape)
					print('data loaded from %s '%(input_filename))

					if (i1==0):
						print('columns: ',np.asarray(df_link_query_pre1.columns))
						print('data preview: ')
						print(df_link_query_pre1[0:2])

					input_filename_2 = input_filename_list2[i1]
					select_config.update({'filename_annot_1':input_filename_2})
					if flag_select_link_type>0:
						input_filename_link = input_filename_list3[i1]
						select_config.update({'filename_link_type':input_filename_link})

					# flag_select_feature_2=0
					# flag_select_feature_2=1
					# perform peak-TF-gene link selection
					df_link_query_pre2, df_link_query2, df_link_query3 = self.test_gene_peak_tf_query_select_1(df_gene_peak_query=df_link_query_pre1,
																												df_annot_query=[],
																												df_score_annot=df_score_annot_1,
																												lambda1=0.5,lambda2=0.5,
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
																												save_mode=save_mode_2,output_file_path=output_file_path,
																												filename_prefix_save=filename_prefix_1,
																												verbose=verbose,select_config=select_config)

					print('the initial peak-TF-gene links with scores, dataframe of size ',df_link_query_pre2.shape)
					print('columns ',np.asarray(df_link_query_pre2.columns))
					print('data preview: ')
					print(df_link_query_pre2[0:2])

					print('the selected peak-TF-gene links, dataframe of size ',df_link_query2.shape)
					print('columns ',np.asarray(df_link_query2.columns))
					print('data preview: ')
					print(df_link_query2[0:2])

					if 'column_score_query_1' in select_config:
						column_score_query_1 = select_config['column_score_query_1'] # the columns to keep

						# column_score_query_1 = select_config['column_score_query_1']
						t_columns = pd.Index(column_score_query_1).intersection(df_link_query_pre2.columns,sort=False)
						df_link_query_pre2 = df_link_query_pre2.loc[:,t_columns]

						# keep the columns of correlation and p-values
						column_score_vec_2 = ['peak_tf_corr','gene_tf_corr_peak','peak_gene_corr_','gene_tf_corr',
												'peak_tf_pval_corrected','gene_tf_corr_peak_pval_corrected1',
												'peak_gene_corr_pval','gene_tf_pval_corrected']
						
						column_score_query_2 = list(column_score_query_1)+column_score_vec_2
						column_score_query_2 = pd.Index(column_score_query_2).unique()
						select_config.update({'column_score_query_2':column_score_query_2})

						t_columns_2 = pd.Index(column_score_query_2).intersection(df_link_query2.columns,sort=False)
						df_link_query2 = df_link_query2.loc[:,t_columns_2]

					b = input_filename.find('.txt')
					extension = 'txt.gz'
					compression = 'gzip'
					output_filename = input_filename[0:b]+'.query1.%s'%(extension)
					float_format = '%.5f'
					df_link_query2 = df_link_query2.sort_values(by=[column_id1,column_id2,column_score_1],ascending=[True,True,False])
					df_link_query2.to_csv(output_filename,index=False,sep='\t',float_format=float_format,compression=compression)
					
					list_1.append(df_link_query_pre2)
					list_2.append(df_link_query2)

				if query_num1>1:
					df_feature_link_pre1 = pd.concat(list_1,axis=0,join='outer',ignore_index=False)
					df_feature_link_pre2 = pd.concat(list_2,axis=0,join='outer',ignore_index=False)
				else:
					df_feature_link_pre1 = list_1[0]
					df_feature_link_pre2 = list_2[0]

				if (save_mode>0) and (query_num1>1):
					float_format = '%.5E'
					# extension = 'txt.gz'
					extension = 'txt'
					compression = 'infer'
					output_filename_1 = '%s/%s.%s.1.%s'%(output_file_path,filename_prefix_score,filename_annot_1,extension)
					
					column_id1 = 'gene_id'
					df_feature_link_pre1.index = np.asarray(df_feature_link_pre1[column_id1])
					df_feature_link_pre1.to_csv(output_filename_1,sep='\t',float_format=float_format,compression=compression)
					print('the initial peak-TF-gene links with scores, dataframe of size ',df_feature_link_pre1.shape)
					print('columns ',np.asarray(df_feature_link_pre1.shape))
					
					output_filename_2 = '%s/%s.%s.2.%s'%(output_file_path,filename_prefix_score,filename_annot_1,extension)
					df_feature_link_pre2.index = np.asarray(df_feature_link_pre2[column_id1])
					df_feature_link_pre2.to_csv(output_filename_2,sep='\t',float_format=float_format,compression=compression)
					print('the selected peak-TF-gene links, dataframe of size ',df_feature_link_pre2.shape)
					print('columns ',np.asarray(df_feature_link_pre2.columns))

		return df_feature_link_pre1, df_feature_link_pre2

	## ====================================================
	# compute peak-TF-gene association score quantiles
	def test_query_feature_score_quantile_compute_1(self,df_feature_link=[],input_filename_list=[],index_col=0,column_idvec=[],iter_mode=1,overwrite=False,
															save_mode=1,output_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		compute peak-TF-gene association score quantiles
		:param df_feature_link: (dataframe) annotations of peak-TF-gene links
		:param input_filename_list: (list) paths of files which saved peak-TF-gene link annotations
		:param index_col: column to use as the row label for data in the dataframe
		:param column_idvec: (array or list) columns representing TF, peak, and gene names in the feature association dataframe
		:param iter_mode: indicator of whether the peak-TF-gene link scores are computed in batch mode
		:param overwrite: indicator of whether to overwrite the current file of peak-TF-gene link annotations
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) updated peak-TF-gene link annotations
				 2. dictionary containing updated parameters
		"""

		flag_score_quantile_1 = 1
		if flag_score_quantile_1>0:		
			if len(df_feature_link)==0:
				if iter_mode>0:
					if len(input_filename_list)==0:
						input_filename_list = select_config['filename_list_score']
			
			file_save_path2 = select_config['file_path_motif_score']
			# column_vec_query1 = ['score_pred1','score_pred2','score_pred_combine']
			filename_prefix_score = select_config['filename_prefix_score']
			filename_annot_score_2 = select_config['filename_annot_score_2']
			
			if len(column_idvec)==0:
				column_idvec = ['motif_id','peak_id','gene_id']

			column_id3, column_id2, column_id1 = column_idvec
			column_id_query = column_id3

			column_1 = 'column_score_1'
			if column_1 in select_config:
				column_score_1 = select_config[column_1]
			else:
				column_score_1 = 'score_pred1'
				select_config.update({column_1:column_score_1})

			# column_score_vec = [column_query1]
			column_score_vec = [column_score_1]
			
			column_label_1 = 'feature2_score1_quantile'
			column_label_vec = [column_label_1]
			
			filename_annot_1 = filename_annot_score_2
			filename_combine = '%s/%s.%s.combine.txt.gz'%(file_save_path2,filename_prefix_score,filename_annot_1)
			
			float_format = '%.5E'
			compression = 'gzip'
			save_mode_2 = 1
			output_filename_2 = filename_combine
			# column_vec_query2 = list(column_idvec)+column_vec_query1
			column_vec_query2 = list(column_idvec)+column_score_vec
			# column_vec_query2 = pd.Index(column_vec_query2).union(['peak_tf_corr','peak_gene_corr_','gene_tf_corr_peak','gene_tf_corr'],sort=False)
			
			# overwrite = False
			if (os.path.exists(filename_combine)==True) and (overwrite==False):
				print('the file exists: %s'%(filename_combine))
				input_filename = filename_combine
				b = input_filename.find('.gz')
				if b<0:
					df_link_query1 = pd.read_csv(input_filename,index_col=False,sep='\t')
				else:
					df_link_query1 = pd.read_csv(input_filename,compression='gzip',index_col=False,sep='\t')
				print('df_link_query1: ',df_link_query1.shape)
			else:
				print('estimate feature score quantile')
				# estimate score 1 quantile for each TF; the estimations from different runs need to be combined
				df_link_query1 = self.test_query_feature_score_quantile_1(df_feature_link=df_feature_link,
																			input_filename_list=input_filename_list,
																			index_col=index_col,
																			column_idvec=column_idvec,
																			column_vec_query=column_vec_query2,
																			column_score_vec=column_score_vec,
																			column_label_vec=column_label_vec,
																			column_id_query=column_id_query,iter_mode=0,
																			save_mode=save_mode,output_file_path='',output_filename_1='',output_filename_2=output_filename_2,
																			filename_prefix_save='',
																			float_format=float_format,compression='gzip',
																			verbose=verbose,select_config=select_config)
			
			df_link_query1.index = utility_1.test_query_index(df_link_query1,column_vec=column_idvec)	

			print('peak-TF-gene links, dataframe of size ',df_link_query1.shape)
			print(df_link_query1.columns)
			select_config.update({'filename_combine':filename_combine})

			return df_link_query1, select_config

	## ====================================================
	# compute quantiles of feature assciation scores
	def test_query_feature_score_quantile_1(self,df_feature_link=[],input_filename_list=[],index_col=0,column_idvec=['peak_id','gene_id','motif_id'],column_vec_query=[],column_score_vec=[],column_label_vec=[],
												column_id_query='motif_id',iter_mode=0,flag_unduplicate=0,
												save_mode=0,output_file_path='',output_filename_1='',output_filename_2='',filename_prefix_save='',compression='gzip',float_format='%.5E',verbose=0,select_config={}):

		"""
		compute quantiles of feature assciation scores
		:param df_feature_link: (dataframe) annotations of peak-TF-gene links
		:param input_filename_list: (list) paths of files which saved peak-TF-gene link annotations
		:param index_col: column to use as the row label for data in the dataframe
		:param column_idvec: (array or list) columns representing TF, peak, and gene names in the feature association dataframe
		:param column_vec_query: (array or list): the columns to retrieve data from when we merge multiple dataframes
		:param column_score_vec: (array or list): columns representing the specific score types for which we perform quantile transformation
		:param column_label_vec: (array or list): columns representing estimated score quantiles for the specific score types
		:param column_id_query: (str) column representing the feature name in the feature association dataframe
		:param iter_mode: indicator of whether the peak-TF-gene link scores are computed in batch mode
		:param flag_unduplicate: indicator of whether to unduplicate the dataframe
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename_1: filename to save data
		:param output_filename_2: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param compression: compression type used in saving data
		:param float_format: the format to keep data precision used in saving data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-TF-gene links including quantiles of specific types of association scores
		"""

		flag_query_1 = 1
		if flag_query_1>0:
			if len(df_feature_link)==0:
				if len(input_filename_list)>0:
					# column_vec_query1 = ['score_pred1','score_pred2','score_pred_combine']
					df_link_query = utility_1.test_file_merge_1(input_filename_list,column_vec_query=column_vec_query,
																index_col=index_col,header=0,float_format=-1,
																flag_unduplicate=flag_unduplicate,
																save_mode=0,output_filename=output_filename_1,
																verbose=verbose)

					
					if (save_mode>0) and (output_filename_1!=''):
						df_link_query.to_csv(output_filename_1,sep='\t')
				else:
					print('please provide feature association query')
			else:
				df_link_query = df_feature_link

			if not (column_id_query in df_link_query.columns):
				df_link_query[column_id_query] = np.asarray(df_link_query.index)
			
			print('peak-TF-gene links, dataframe of size ',df_link_query.shape)
			print('columns: ',np.asarray(df_link_query.columns))
			print('data preview:\n',df_link_query[0:2])

			column_idvec = ['motif_id','peak_id','gene_id']
			df_link_query.index = test_query_index(df_link_query,column_vec=column_idvec)
			df_link_query = df_link_query.loc[(~df_link_query.index.duplicated(keep='first')),:]
			
			print('peak-TF-gene links (unduplicated), dataframe of size ',df_link_query.shape)
			print('columns: ',np.asarray(df_link_query.columns))
			print('data preview:\n',df_link_query[0:2])

			# compute score quantiles
			df_link_query_pre1 = self.test_score_quantile_1(data=df_link_query,feature_query_vec=[],
															column_id_query=column_id_query,
															column_idvec=column_idvec,
															column_query_vec=column_score_vec,
															column_label_vec=column_label_vec,
															flag_annot=1,
															reset_index=0,
															parallel_mode=0,
															interval=100,
															verbose=verbose,select_config=select_config)

			if (save_mode>0) and (output_filename_2!=''):
				# if (compression!=-1):
				if (not (compression is None)):
					df_link_query_pre1.to_csv(output_filename_2,index=False,sep='\t',float_format=float_format,compression=compression)
				else:
					df_link_query_pre1.to_csv(output_filename_2,index=False,sep='\t',float_format=float_format)
				
			return df_link_query_pre1

	## ====================================================
	# compute quantiles of scores
	def test_score_quantile_1(self,data=[],feature_query_vec=[],column_id_query='',column_idvec=[],column_query_vec=[],column_label_vec=[],
								flag_annot=1,reset_index=1,parallel_mode=0,interval=100,verbose=0,select_config={}):

		"""
		compute quantiles of scores
		:param data: (dataframe) annotations of feature associations (e.g., peak-TF-gene links) including association scores
		:param feature_query_vec: (array or list) names of the feature entities (e.g., TFs) for which we compute quantiles of specific association scores
		:param column_id_query: (str) column representing the feature name in the feature association dataframe
		:param column_idvec: (array or list) columns representing TF, peak, and gene names in the feature association dataframe
		:param column_query_vec: (array or list): columns representing the specific score types for which we perform quantile transformation
		:param column_label_vec: (array or list): columns representing estimated score quantiles for the specific score types
		:param flag_annot: indicator of whether to add annotations
		:param reset_index: indicator of whether to reset index of the dataframe
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) score quantiles matrix (row:feature associations; column corresponds to specific score type)
		"""

		df_feature = data
		if len(feature_query_vec)==0:
			feature_query_vec = df_feature[column_id_query].unique()
		feature_query_num = len(feature_query_vec)

		# column_query_vec_1 = [column_query1,column_query2]
		column_num1 = len(column_query_vec)
		print('feature_query_vec: ',feature_query_num)
		print('column_id_query: ',column_id_query)

		if reset_index>0:
			query_id_ori = df_feature.index.copy()
			df_feature.index = test_query_index(df_feature,column_vec=column_idvec)

		# query_id_ori = pd.Index(query_id_ori)
		# df_feature.index = np.asarray(df_feature[column_id_query])
		query_id_1 = df_feature.index
		if parallel_mode>0:
			if interval<0:
				score_query_1 = Parallel(n_jobs=-1)(delayed(self.test_score_quantile_pre1)(data=df_feature,feature_query=feature_query_vec[id1],feature_query_id=id1,
																							column_id_query=column_id_query,column_query_vec=column_query_vec,
																							column_label_vec=column_label_vec,
																							verbose=verbose,select_config=select_config) for id1 in range(feature_query_num))
				

				query_num1 = len(score_query_1)
				for i1 in range(query_num1):
					score_mtx = score_query_1[i1]
					query_id1 = score_mtx.index
					df_feature.loc[query_id1,column_label_vec] = score_mtx

			else:
				iter_num = int(np.ceil(feature_query_num/interval))
				print('iter_num, interval: ',iter_num,interval)
				for iter_id in range(iter_num):
					start_id1, start_id2 = interval*iter_id, np.min([interval*(iter_id+1),feature_query_num])
					print('start_id1, start_id2: ',start_id1,start_id2,iter_id)
					query_vec_1 = np.arange(start_id1,start_id2)
					# estimate feature score quantile
					score_query_1 = Parallel(n_jobs=-1)(delayed(self.test_score_quantile_pre1)(data=df_feature,feature_query=feature_query_vec[id1],feature_query_id=id1,
																									column_id_query=column_id_query,column_query_vec=column_query_vec,
																									column_label_vec=column_label_vec,
																									verbose=verbose,select_config=select_config) for id1 in query_vec_1)
					

					query_num1 = len(score_query_1)
					for i1 in range(query_num1):
						score_mtx = score_query_1[i1]
						query_id1 = score_mtx.index
						df_feature.loc[query_id1,column_label_vec] = score_mtx
		else:
			for i1 in range(feature_query_num):
				feature_query1 = feature_query_vec[i1]
				feature_query_id1 = i1
				
				# estimate score quantile
				score_mtx = self.test_score_quantile_pre1(data=df_feature,feature_query=feature_query1,
															feature_query_id=feature_query_id1,
															column_id_query=column_id_query,
															column_query_vec=column_query_vec,
															column_label_vec=column_label_vec,
															verbose=verbose,select_config=select_config)
				
				if (i1%500==0):
					print('df_feature: ',df_feature.shape,feature_query1,i1)
					print(df_feature[0:2])
					print('score_mtx: ',score_mtx.shape)
					print(score_mtx[0:2])

				query_id1 = score_mtx.index
				try:
					df_feature.loc[query_id1,column_label_vec] = score_mtx
				
				except Exception as error:
					print('error! ',error,feature_query1,i1,len(query_id1))
					# query_id_1 = df1.index
					# id1 = df1.index.duplicated(keep='first')
					query_id_2 = query_id1.unique()
					df1 = df_feature.loc[query_id_2,:]
					print('df1 ',df1.shape)
					
					# query_id2 = query_id_1[df1.index.duplicated(keep='first')]
					df2 = df1.loc[df1.index.duplicated(keep=False),:]
					query_id2 = df2.index.unique()
					print('query_id_2: ',len(query_id_2))
					print('duplicated idvec, query_id2: ',len(query_id2))
					
					file_save_path2 = select_config['file_path_motif_score']
					output_filename = '%s/test_query_score_quantile.duplicated.query1.%s.1.txt'%(file_save_path2,feature_query1)
					
					if os.path.exists(output_filename)==True:
						print('the file exists: %s'%(output_filename))
						filename1 = output_filename
						b = filename1.find('.txt')
						output_filename = filename1[0:b]+'.copy1.txt'
					df2.to_csv(output_filename,sep='\t')
					df_feature = df_feature.loc[(~df_feature.index.duplicated(keep='first')),:]

		if reset_index==1:
			df_feature.index = query_id_ori

		return df_feature

	## ====================================================
	# compute quantiles of scores
	def test_score_quantile_pre1(self,data=[],feature_query='',feature_query_id=0,column_id_query='',column_query_vec=[],column_label_vec=[],verbose=0,select_config={}):

		"""
		compute quantiles of scores
		:param data: (dataframe) annotations of feature associations (e.g., peak-TF-gene links) including association scores
		:param feature_query: (str) name of the feature entity (e.g., TF name)
		:param feature_query_id: (int) the index of the feature entity
		:param column_id_query: (str) column representing the feature name in the feature association dataframe
		:param column_query_vec: (array or list): columns representing the specific score types for which we perform quantile transformation
		:param column_label_vec: (array or list): columns representing estimated score quantiles for the specific score types
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) score quantiles matrix (row:feature associations; column corresponds to specific score type)
		"""

		df_feature = data
		id1 = (df_feature[column_id_query]==feature_query)
		df_query1 = df_feature.loc[id1,:]

		query_id_1 = df_feature.index
		query_id1 = query_id_1[id1]
		query_num1 = len(query_id1)
		if (verbose>0) and (feature_query_id%100==0):
			print('feature_query: ',feature_query,feature_query_id,query_num1)

		normalize_type = 'uniform'
		score_query = df_query1.loc[:,column_query_vec]
		num_quantiles = np.min([query_num1,1000])
		score_mtx = quantile_transform(score_query,n_quantiles=num_quantiles,output_distribution=normalize_type)
		score_mtx = pd.DataFrame(index=query_id1,columns=column_label_vec,data=np.asarray(score_mtx),dtype=np.float32)

		return score_mtx

	## ====================================================
	# select peak-TF-gene links with specific association scores above thresholds
	def test_gene_peak_tf_query_select_1(self,df_gene_peak_query=[],df_annot_query=[],df_score_annot=[],lambda1=0.5,lambda2=0.5,type_id_1=0,column_id1=-1,
											flag_compare_thresh1=1,flag_select_pair_1=1,flag_select_feature_1=0,flag_select_feature_2=0,
											flag_select_local=1,flag_select_link_type=1,iter_mode=0,
											input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',verbose=1,select_config={}):

		"""
		select peak-TF-gene links with specific association scores above thresholds
		:param df_gene_peak_query: (dataframe) annotations of peak-TF-gene links including association scores
		:param df_annot_query: (dataframe)
		:param df_score_annot: (dataframe)
		:param lambda1: (float) weight of score 1 (TF-(peak,gene) association score)
		:param lambda2: (float) weight of score 2 ((peak,TF)-gene association score)
		:param type_id_1: the type of selection thresholds to use
		:param column_id1: (str) column representing the maximal score possible for a given peak-TF-gene link using the associated regularization score
		:param flag_compare_thresh1: indicator of whether to filter the links with discrepancy between gene-TF expression partial correlation given peak accessibility and gene-TF expression correlation above threshold
		:param flag_select_pair_1: indicator of whether to used paired thresholds on score 1 and score 2 to select peak-TF-gene links
		:param flag_select_feature_1: indicator of whether to select links based on quantiles of score 1 (TF binding score) for each TF
		:param flag_select_feature_2: indicator of whether to select links based on quantiles of score 2 for each gene
		:param flag_select_local: indicator of whether to select peak-TF-gene links based on strong peak-TF correlations
		:param flag_select_link_type: indicator of whether to select peak-TF-gene links based on the estimated type of different links (repression or activation)
		:param iter_mode: indicator of whether the peak-TF-gene link scores are computed in batch mode
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) updated annotations of peak-TF-gene links, including which links are selected by specific thresholds;
				 2. (dataframe) annotations of selected peak-TF-gene links without using selection criteria on link types;
				 3. (dataframe) annotations of selected peak-TF-gene links using selection criteria on link types and the other thresholds which select the links in (2);
		"""

		flag_query1=1
		if flag_query1>0:
			# pre-selection of peak-tf-gene link query by not strict thresholds to filter link query with relatively low estimated scores
			flag_select_thresh1_1=0
			# flag_select_thresh1=0

			# thresh_score_1 = 0.10
			# thresh_corr_1, thresh_corr_2 = 0.10, 0.30
			pval_thresh_vec = [0.05,0.10,0.15,0.25,0.50]
			pval_thresh_1 = pval_thresh_vec[3]

			thresh_score_1, thresh_corr_1, thresh_corr_2, thresh_pval_1, thresh_pval_2 = 0.10, 0.10, 0.30, 0.25, 0.5
			filename_annot_thresh = '%s_%s_%s_%s'%(thresh_score_1,thresh_corr_1,pval_thresh_1,thresh_corr_2)
			
			input_file_path2 = input_file_path
			output_file_path = input_file_path2
			df_gene_peak_query_1 = df_gene_peak_query
			
			if 'column_idvec' in select_config:
				column_idvec = select_config['column_idvec']
			else:
				column_idvec = ['motif_id','peak_id','gene_id']
				select_config.update({'column_idvec':column_idvec})
			
			print('column_idvec ',column_idvec)
			df_gene_peak_query_1.index = test_query_index(df_gene_peak_query_1,column_vec=column_idvec)
			df_gene_peak_query_1 = df_gene_peak_query_1.fillna(0)
			query_id_ori = df_gene_peak_query_1.index.copy()
			
			# flag_compare_thresh1=1
			flag_annot_1 = 1
			df_gene_peak_query_pre1 = []

			# query estimated correlation and partial correlation annotations
			column_query_1 = [['peak_tf_corr','peak_tf_pval_corrected'],['peak_gene_corr_','peak_gene_corr_pval'],
								['gene_tf_corr_peak','gene_tf_corr_peak_pval_corrected1']]			

			field_query1 = ['peak_tf','peak_gene']
			field_query2 = ['corr','pval']
			list1 = []

			for field_id1 in field_query1:
				list1.append(['column_%s_%s'%(field_id1,field_id2) for field_id2 in field_query2])
			list1.append(['column_query_cond','column_pval_cond'])
			field_query_1 = list1

			field_num1 = len(field_query_1)
			for i1 in range(field_num1):
				# field1, field2 = field_query_1[i1], field_query_2[i1]
				for i2 in range(2):
					field1 = field_query_1[i1][i2]
					if (field1 in select_config):
						column_1 = select_config[field1]
						print(field1,column_1,i1,i2)
						column_query_1[i1][i2] = column_1

			column_query_pre1 = column_query_1.copy()

			column_num1 = len(column_query_1)
			columns_1 = [query_vec[0] for query_vec in column_query_1] # the columns for correlation
			# column_gene_tf_corr, column_gene_tf_pval = 'gene_tf_corr','gene_tf_pval_corrected'
			
			field_query = ['column_gene_tf_corr','column_gene_tf_pval']
			columns_1 = columns_1 + [select_config[field_id] for field_id in field_query]

			flag_select_thresh1_feature_local=flag_select_local
			if flag_select_thresh1_feature_local>0:
				columns_pval = [query_vec[1] for query_vec in column_query_1]
				columns_1 = columns_1 + columns_pval

			column_annot_query = pd.Index(columns_1).difference(df_gene_peak_query_1.columns,sort=False)
			if len(column_annot_query)>0:
				print('load annotaions from file')
				print(column_annot_query)
						
				if len(df_annot_query)==0:
					if 'filename_annot_1' in select_config:
						filename_annot_1 = select_config['filename_annot_1']
						df_annot_query = pd.read_csv(filename_annot_1,index_col=False,sep='\t')
					else:
						print('please provide the estimated correlation and p-value')

				df_list1 = [df_gene_peak_query_1,df_annot_query]
				column_idvec_1 = column_idvec
				column_vec_1 = [column_annot_query]
				# copy specified columns from the second dataframe to the first dataframe
				df_gene_peak_query_1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec_1,
																		column_vec=column_vec_1,
																		df_list=df_list1,
																		type_id_1=0,type_id_2=0,
																		reset_index=False,
																		select_config=select_config)

			if flag_compare_thresh1>0:
				type_id_1=2
				thresh_gene_tf_corr_peak = 0.30
				thresh_gene_tf_corr_ = 0.05
				if 'thresh_gene_tf_corr_compare' in select_config:
					thresh_gene_tf_corr_compare = select_config['thresh_gene_tf_corr_compare']
					thresh_gene_tf_corr_peak, thresh_gene_tf_corr_ = thresh_gene_tf_corr_compare[0:2]

				output_file_path = input_file_path2
				# filename_save_annot = '%s.%d'%(filename_save_annot_2,type_id_1)
				# filename_prefix_save = '%s.%s'%(filename_prefix_save_pre2,filename_save_annot)
				input_filename_1 = ''
				
				# find the peak-TF-gene links with discrepancy between gene-TF expression partial correlation given peak accessibility and gene-TF expression correlation
				# query_id_1: the links to keep; 
				# query_id_2: the links with discrepancy between gene-TF expression partial correlation given peak accessibility and gene-TF expression correlation above threshold
				df_gene_peak_query_pre1, query_id_1, query_id_2 = self.test_gene_peak_tf_query_compare_1(input_filename=input_filename_1,
																											df_gene_peak_query=df_gene_peak_query_1,
																											thresh_corr_1=thresh_gene_tf_corr_peak,
																											thresh_corr_2=thresh_gene_tf_corr_,
																											save_mode=1,
																											output_file_path=output_file_path,
																											filename_prefix_save=filename_prefix_save,
																											select_config=select_config)

				df_query_1 = df_gene_peak_query_pre1.loc[query_id_1]
				df_query_2 = df_gene_peak_query_pre1.loc[query_id_2]

				print('df_query_1, df_query_2 ',df_query_1.shape,df_query_2.shape)
				query_compare_group1, query_compare_group2 = query_id_1, query_id_2

				if flag_annot_1>0:
					column_label_pre1 = 'label_gene_tf_corr_peak_compare'
					df_gene_peak_query_pre1.loc[query_id_2,column_label_pre1] = 1

			# select by pre-defined threshold
			flag_select_thresh1_pair_1=flag_select_pair_1
			df_link_query_1 = df_gene_peak_query_pre1

			column_query1, column_query2 = select_config['column_score_1'], select_config['column_score_2']
			column_query_vec = [column_query1,column_query2]

			# column_idvec = ['motif_id','peak_id','gene_id']
			column_id3, column_id2, column_id1 = column_idvec[0:3]
			if flag_select_thresh1_pair_1>0:
				# thresh_vec_1 = [[0.10,0.05],[0.05,0.10]]
				thresh_vec_1 = [[0.10,0],[0,0.10]]
				if 'thresh_score_query_1' in select_config:
					thresh_vec_1 = select_config['thresh_score_query_1']
				
				thresh_num1 = len(thresh_vec_1)
				score_query1, score_query2 = df_link_query_1[column_query1], df_link_query_1[column_query2]

				list1 = []
				for i1 in range(thresh_num1):
					thresh_vec_query = thresh_vec_1[i1]
					thresh_score_1, thresh_score_2 = thresh_vec_query[0:2]
					id1 = (score_query1>thresh_score_1)
					id2 = (score_query2>thresh_score_2)
					id_1 = (id1&id2)
					query_id1 = query_id_ori[id_1]
					query_num1 = len(query_id1)
					list1.append(query_id1)
					if verbose>0:
						print('thresh_1, thresh_2: ',thresh_score_1,thresh_score_2,query_num1)

				query_id_1, query_id_2 = list1[0:2]
				column_label_1, column_label_2 = 'label_score_1', 'label_score_2'
				df_link_query_1.loc[query_id_1,column_label_1] = 1
				df_link_query_1.loc[query_id_1,column_label_2] = 1

			flag_combine_1=1
			if flag_combine_1>0:
				# field_query_1 = ['label_score_1', 'label_score_2',
				# 					'peak_tf_corr_thresh1','peak_gene_corr_thresh1']

				field_query_1 = ['label_score_1', 'label_score_2']
				field_query1_1 = ['peak_tf_corr_thresh1','peak_gene_corr_thresh1']

				field_query_2 = ['feature1_score1_quantile','feature1_score2_quantile']
				if flag_select_feature_2>0:
					# field_query_2 = ['feature1_score1_quantile','feature1_score2_quantile','feature2_score1_quantile']
					field_query_2 = field_query_2 + ['feature2_score1_quantile']

				field_query_3 = ['label_gene_tf_corr_peak_compare']
				field_query_5 = ['link_query']

				mask_1 = (df_link_query_1.loc[:,field_query_1]>0) # label score above threshold
				df1 = (mask_1.sum(axis=1)>0)

				field_query1 = field_query_3[0]
				df3 = (df_link_query_1.loc[:,field_query1]>0) # difference between gene_tf_corr_peak and gene_tf_corr

				if (flag_select_feature_1>0) or (flag_select_feature_2>0):
					thresh1 = 0.95
					if 'thresh_score_quantile' in select_config:
						thresh1 = select_config['thresh_score_quantile']
					mask_2 = (df_link_query_1.loc[:,field_query_2]>thresh1) # score quantile above threshold
					df2 = (mask_2.sum(axis=1)>0)
				
					# id_1 = (((df1|df2)&df5)|df3)
					# id_1 = (df1|df2|df3)
					id_1 = (df1|df2)&(~df3)

				else:
					id_1 = (df1&(~df3))

				df_link_query1 = df_link_query_1.loc[id_1,:]
				df_link_query2 = []
				flag_select_link_type_1 = flag_select_link_type
				if flag_select_link_type_1>0:
					field_query2 = field_query_5[0]
					df5 = (df_link_query_1.loc[:,field_query2]!=-1)

					# id_2 = (((df1|df2)&(~df3))&df5)
					id_2 = (id_1&df5)
					df_link_query2 = df_link_query_1.loc[id_2,:]

				# df_link_query3 = df_link_query_1.loc[(~df3),:]
				# print('df_link_query_1, df_link_query1, df_link_query2: ',df_link_query_1.shape,df_link_query1.shape,df_link_query2.shape)

			return df_link_query_1, df_link_query1, df_link_query2

	## ====================================================
	# combine peak-TF-gene association scores of different subsets of peak-TF-gene links
	def test_feature_link_query_combine_pre1(self,feature_query_num,feature_query_vec=[],column_vec_score=[],column_vec_query=[],interval=3000,flag_quantile=0,
													save_mode=1,save_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		combine peak-TF-gene association scores of different subsets of peak-TF-gene links
		:param feature_query_num: the number of observations (e.g., genes) for which we query the associatied peak-TF-gene links
		:param feature_query_vec: (array or list) the observations (e.g., TFs) for which we query the associatied peak-TF-gene links
		:param column_vec_score: (array or list) two elements included: 1: column name for score 1 (TF-(peak,gene) score) in the peak-TF-gene link annotation datafram; 2: column name for score 2 ((peak,TF)-gene score); 
		:param column_vec_query: (array or list) the columns to retrieve from the annotation dataframe of each subset of peak-TF-gene links
		:param interval: the batch size of target genes used in computing the gene-TF expression partial correlation given peak accessibility of peak-TF-gene links
		:param flag_quantile: indicator of whether to compute the quantiles of score 1 of peak-TF-gene links associated with each given TF
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) concated peak-TF-gene link annotations including peak-TF-gene association scores;
				 2. (dataframe) annotations of peak-TF-gene links associated with given TFs (specified in feature_query_vec)
		"""
		
		flag_query1=1
		if flag_query1>0:
			flag_query_2 = 1
			# flag_query_2 = 0
			file_save_path2 = select_config['file_path_motif_score']
			input_file_path = file_save_path2
			filename_prefix_default_1 = select_config['filename_prefix_cond']
			
			column_1 = 'filename_link_cond'
			if column_1 in select_config:
				input_filename_query = select_config[column_1]

			df_link_query_pre2 = []
			df_link_query_pre2_1 = []
			flag_load_2 = 0
			if len(feature_query_vec)>0:
				flag_load_2 = 1

			if os.path.exists(input_filename_query)==True:
				print('the file exists: %s'%(input_filename_query))
				# select_config.update({'filename_combine':input_filename_query})
				flag_query_2 = 0
				if flag_load_2>0:
					df_link_query_pre2 = pd.read_csv(input_filename_query,index_col=False,sep='\t')

			if flag_query_2>0:
				feature_query_num_1 = feature_query_num
				iter_num = int(np.ceil(feature_query_num_1/interval))
				input_filename_list = []
				df_list1 = []
				if len(column_vec_score)==0:
					column_query1, column_query2 = 'score_pred1', 'score_pred2'
				else:
					column_query1, column_query2 = column_vec_score[0:2]

				for i1 in range(iter_num):
					query_id_1 = i1*interval
					query_id_2 = (i1+1)*interval
					filename_prefix_save_pre_2 = '%s.pcorr_query1.%d_%d'%(filename_prefix_default_1,query_id_1,query_id_2)
					input_filename = '%s/%s.annot2.init.1.txt'%(file_save_path2,filename_prefix_save_pre_2)
					
					column_vec_query1 = column_vec_query
					if len(column_vec_query)==0:
						# column_vec_query1 = ['score_pred1', 'score_pred2','score_pred_combine','score_normalize_pred','score_pred1_correlation','score_1']
						column_vec_query1 = [column_query1, column_query2, 'score_pred_combine']
					
					column_idvec = select_config['column_idvec']
					if os.path.exists(input_filename)==True:
						df_query1_ori = pd.read_csv(input_filename,index_col=False,sep='\t')
						field_query = list(column_idvec) + column_vec_query1
						df_query1 = df_query1_ori.loc[:,field_query]
						df_list1.append(df_query1)
						print('df_query1: ',df_query1.shape)
						print(input_filename)
					else:
						print('the file does not exist: %s'%(input_filename))
						return

				df_link_query_pre1 = pd.concat(df_list1,axis=0,join='outer',ignore_index=True)
				print('combined peak-TF-gene link annotations, dataframe of size ',df_link_query_pre1.shape)

				if flag_quantile>0:
					column_query_vec_2 = [column_query1]
					column_label_1 = 'feature2_score1_quantile'
					column_label_vec_2 = [column_label_1]
					column_id_query = 'motif_id'

					# estimate score quantiles
					df_link_query_pre2 = self.test_score_quantile_1(data=df_link_query_pre1,
																	feature_query_vec=[],
																	column_id_query=column_id_query,
																	column_idvec=column_idvec,
																	column_query_vec=column_query_vec_2,
																	column_label_vec=column_label_vec_2,
																	flag_annot=1,
																	verbose=verbose,select_config=select_config)
				
				else:
					df_link_query_pre2 = df_link_query_pre1

				if save_mode>0:
					output_file_path = file_save_path2
					# output_filename_1 = '%s/%s.annot2.init.1.txt.gz'%(output_file_path,filename_prefix_save_2)
					if output_filename=='':
						output_filename_1 = input_filename_query
					else:
						output_filename_1 = output_filename

					float_format = '%.5f'
					compression = 'gzip'
					df_link_query_pre2.to_csv(output_filename_1,index=False,sep='\t',float_format=float_format,compression=compression)

					filename_1 = output_filename_1
					select_config.update({'filename_combine':filename_1,'filename_link_cond':filename_1})

			if flag_load_2>0:
				# query feature links associated with the given observations
				query_id_ori = df_link_query_pre2.index.copy()
				column_id_query = 'motif_id'
				df_link_query_pre2.index = np.asarray(df_link_query_pre2[column_id_query])

				feature_query_ori = df_link_query_pre2[column_id_query].unique()
				# query the TFs included in the given peak-TF-gene links
				motif_query_1 = pd.Index(feature_query_vec).intersection(feature_query_ori,sort=False)

				df_link_query_pre2_1 = df_link_query_pre2.loc[motif_query_1,:]
				df_link_query_pre2_1 = df_link_query_pre2_1.sort_values(by=[column_id_query,column_query1],ascending=[True,False])
				df_link_query_pre2.index = query_id_ori

				# output_filename_2 = '%s/%s.annot2.init.query1.1.txt'%(output_file_path,filename_prefix_save_2)
				extension = '.txt'
				b = input_filename_query.find(extension)
				output_filename_2 = input_filename_query[0:b]+'.query1.txt'
				df_link_query_pre2_1.to_csv(output_filename_2,index=False,sep='\t',float_format=float_format)

			return df_link_query_pre2, df_link_query_pre2_1

	## ====================================================
	# combine peak-TF-gene association scores of different subsets of peak-TF-gene links
	# reduce intermediate files
	def test_file_combine_1(self,feature_query_num=-1,interval=-1,type_query=0,flag_reduce=1,save_mode=1,save_file_path='',output_filename='',verbose=0,select_config={}):

		column_1, column_2 = 'feature_score_interval', 'feature_query_num'
		feature_score_interval = interval
		select_config.update({column_1:feature_score_interval,column_2:feature_query_num})

		thresh_str_compare = '100_0.15.500_-0.05'
		column_1 = 'thresh_str_compare'
		if not (column_1 in select_config):
			select_config.update({column_1:thresh_str_compare})

		# test_feature_link_query_file_pre2_1(self,data=[],extension='txt',iter_mode=0,flag_reduce=1,save_mode=1,verbose=0,select_config={}):
		extension = 'txt'
		self.test_feature_link_file_pre2_1(iter_mode=0,extension=extension,
											flag_reduce=flag_reduce,
											save_mode=1,
											verbose=verbose,select_config=select_config)

		# type_query = 1
		self.test_feature_link_file_pre2_2(type_query=type_query,iter_mode=0,
											extension=extension,
											flag_reduce=flag_reduce,
											save_mode=1,
											verbose=verbose,select_config=select_config)

		self.test_feature_link_query_combine_pre2(feature_query_num=feature_query_num,
													flag_combine=2,
													flag_reduce=flag_reduce,
													save_mode=1,
													save_file_path='',
													output_filename='',
													verbose=verbose,select_config=select_config)

	## ====================================================
	# combine peak-TF-gene association scores of different subsets of peak-TF-gene links
	def test_feature_link_query_combine_pre2(self,feature_query_num,flag_combine=2,flag_reduce=1,
													save_mode=1,save_file_path='',output_filename='',verbose=0,select_config={}):

		file_save_path2 = select_config['file_path_motif_score']
		path_query_1 = file_save_path2
		input_file_path = file_save_path2
		filename_prefix_default_1 = select_config['filename_prefix_cond']
		filename_prefix_score = select_config['filename_prefix_score']
		filename_annot_score_1 = select_config['filename_annot_score_1']
		
		column_score_vec_1 = ['score_1','score_pred1','score_pred2','score_pred_combine']
		column_score_vec_2 = ['peak_tf_corr','gene_tf_corr_peak','peak_gene_corr_','gene_tf_corr',
								'peak_tf_pval_corrected','gene_tf_corr_peak_pval_corrected1',
								'peak_gene_corr_pval','gene_tf_pval_corrected']

		column_score_vec_query = column_score_vec_1 + column_score_vec_2
		
		input_filename_list_1, input_filename_list_2 = [], []
		input_filename_list_pre2 = []
		df_list_1, df_list_2 = [], []
		df_list_pre2 = []
		df_feature_link_pre1, df_feature_link_pre2 = [], []

		column_1, column_2 = 'feature_score_interval', 'feature_query_num'
		iter_mode_query = 0
		iter_num = 1
		if (column_1 in select_config) and (column_2 in select_config):
			interval = select_config[column_1]
			feature_query_num_1 = select_config[column_2]
			print('feature_query_num: %d, interval: %d'%(feature_query_num_1,interval))
			if interval>0:
				iter_num = int(np.ceil(feature_query_num_1/interval))
				# print('feature_query_num: %d, interval: %d'%(feature_query_num_1,interval))
				iter_mode_query = 1
		
		filename_prefix_save_1 = filename_prefix_score
		list_query_1 = []
		list_query_2 = []
		print('iter_mode_query, iter_num ',iter_mode_query,iter_num)
		if flag_combine>0:
			for i1 in range(iter_num):
				if iter_mode_query>0:
					query_id_1 = i1*interval
					query_id_2= (i1+1)*interval
					filename_prefix_save_pre1 = '%s.%d_%d'%(filename_prefix_save_1,query_id_1,query_id_2)
				else:
					filename_prefix_save_pre1 = filename_prefix_save_1
					query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
					print('query_id1:%d, query_id2:%d'%(query_id1,query_id2))
					if (query_id1>=0) and (query_id2>query_id1):
						filename_prefix_save_pre1 = '%s.%d_%d'%(filename_prefix_save_1,query_id1,query_id2)

				filename_annot_1 = filename_annot_score_1
				extension = 'txt.gz'
				input_filename_1 = '%s/%s.%s.%s'%(input_file_path,filename_prefix_save_pre1,filename_annot_1,extension) # peak-TF-gene link score
				input_filename_2 = '%s/%s.txt'%(input_file_path,filename_prefix_save_pre1) # gene-TF expression partial correlation given peak accessibility
				
				df_query1 = []
				df_query2 = []
				# peak-TF-gene link score
				if os.path.exists(input_filename_1)==True:
					df_query1 = pd.read_csv(input_filename_1,index_col=False,sep='\t')
					# df_list_1.append(df_query1)
					input_filename_list_1.append(input_filename_1)
					print(input_filename_1)
				else:
					print('the file does not exist: %s'%(input_filename_1))

				if flag_combine>1:
					# gene-TF expression partial correlation given peak accessibility
					if os.path.exists(input_filename_2)==True:
						df_query2 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
						# df_list_2.append(df_query2)
						input_filename_list_2.append(input_filename_2)
						print(input_filename_2)
					else:
						print('the file does not exist: %s'%(input_filename_2))

					df_list1 = [df_query1,df_query2]
					column_pval_cond = 'gene_tf_corr_peak_pval_corrected1'
					column_vec_1 = [[column_pval_cond]]
					column_idvec = ['gene_id','peak_id','motif_id']
					
					# copy specified columns from the other dataframes to the first dataframe
					df_link_query_1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec,
																		column_vec=column_vec_1,
																		df_list=df_list1,
																		type_id_1=0,type_id_2=0,
																		reset_index=False,
																		select_config=select_config)
				else:
					df_link_query_1 = df_query1

				df_list_1.append(df_link_query_1)
				input_filename_pre2 = '%s/%s.%s.query1.%s'%(input_file_path,filename_prefix_save_pre1,filename_annot_1,extension) # prepare for the file
				if os.path.exists(input_filename_pre2)==True:
					df_query_pre2 = pd.read_csv(input_filename_pre2,index_col=False,sep='\t')
					df_list_pre2.append(df_query_pre2)
					input_filename_list_pre2.append(input_filename_pre2)
					print(input_filename_pre2)

			query_num1 = len(df_list_1)
			if query_num1>1:
				df_feature_link_pre1 = pd.concat(df_list_1,axis=0,join='outer',ignore_index=False)
				df_feature_link_pre2 = pd.concat(df_list_pre2,axis=0,join='outer',ignore_index=False)
			else:
				df_feature_link_pre1 = df_list_1[0]
				df_feature_link_pre2 = df_list_pre2[0]

			if query_num1>1:
				column_id1 = 'gene_id'
				extension = 'txt.gz'
				compression = 'gzip'

				output_file_path = path_query_1
				output_filename_1 = '%s/%s.%s.%s'%(output_file_path,filename_prefix_score,filename_annot_1,extension)
				df_feature_link_pre1.index = np.asarray(df_feature_link_pre1[column_id1])
				float_format = '%.6E'
				df_feature_link_pre1.to_csv(output_filename_1,sep='\t',float_format=float_format,compression=compression)
				print('the initial peak-TF-gene links with scores, dataframe of size ',df_feature_link_pre1.shape)
				print('columns ',np.asarray(df_feature_link_pre1.shape))
				
				extension_query = 'txt'
				output_filename_2 = '%s/%s.%s.query1.%s'%(output_file_path,filename_prefix_score,filename_annot_1,extension_query)
				
				df_feature_link_pre2.index = np.asarray(df_feature_link_pre2[column_id1])
				df_feature_link_pre2.to_csv(output_filename_2,sep='\t',float_format=float_format)
				print('the selected peak-TF-gene links, dataframe of size ',df_feature_link_pre2.shape)
				print('columns ',np.asarray(df_feature_link_pre2.columns))
				
				list_query_1 = [output_filename_1,output_filename_2]
				list_query_2 = [input_filename_list_1,input_filename_list_2,input_filename_list_pre2]
			else:
				list_query_1 = input_filename_list_1 + input_filename_list_2 + input_filename_list_pre2

			import glob
			extension_1 = 'txt.gz'
			extension_2 = 'txt'
			# extension_vec_query1 = ['annot1_1.1','annot2_1.1','annot2.1']
			# extension_vec_query1 = ['%s.%s'%(annot_str1,extension_1) for annot_str1 in extension_vec_query1]
			# extension_vec_query2 = ['annot1_2.1','annot1_3.1','annot2.init.1.1']
			# extension_vec_query2 = ['%s.%s'%(annot_str2,extension_2) for annot_str2 in extension_vec_query2]
			# extension_vec_query = extension_vec_query1 + extension_vec_query2

			extension_vec_query = [extension_1,extension_2]
			query_num2 = len(extension_vec_query)
			if iter_mode_query>0:
				filename_prefix_save_query = filename_prefix_save_1
			else:
				filename_prefix_save_query = filename_prefix_save_pre1
			filename_prefix_vec = [filename_prefix_save_query]*query_num2
			for i2 in range(query_num2):
				filename_prefix_query = filename_prefix_vec[i2]
				extension_query = extension_vec_query[i2]
				list_query1 = glob.glob('%s/%s.*.%s'%(path_query_1,filename_prefix_query,extension_query))
				list_query1 = pd.Index(list_query1).difference(list_query_1,sort=False)
				list_query_2.append(list_query1)
			
			if (flag_reduce>0) and (len(list_query_2)>0):
				list_query_pre2 = self.test_file_reduce_1(data=list_query_2,select_config=select_config)

	## ====================================================
	# remove the temporary files
	def test_file_reduce_1(self,data=[],save_mode=0,verbose=0,select_config={}):

		list_query_1 = data
		query_num_1 = len(list_query_1)
		list_query_2 = []
		for i1 in range(query_num_1):
			list_query = list_query_1[i1]
			file_num_query= len(list_query)
			if verbose>0:
				print('file number: %d'%(file_num_query))
			for i2 in range(file_num_query):
				filename_query = list_query[i2]
				if os.path.exists(filename_query)==True:
					# print('filename ',filename_query)
					os.remove(filename_query)
					list_query_2.append(filename_query)
				
		return list_query_2

	## ====================================================
	# merge files, save data, and remove files
	def test_query_file_merge_1(self,data=[],field_query=[],index_col=0,header=0,float_format=-1,flag_unduplicate=0,filename_save_vec=[],filename_prefix_vec=[],filename_annot_vec=[],extension='txt',flag_reduce=0,overwrite=False,save_mode=1,output_file_path='',verbose=0,select_config={}):

		field_num = len(field_query)
		dict_query1 = data
		list_query_1 = []
		list_query_2 = []

		flag_query1 = (len(filename_save_vec)>0)
		for i1 in range(field_num):
			field_id = field_query[i1]
			list_query1 = dict_query1[field_id]
			input_filename_list = list_query1

			if flag_query1>0:
				output_filename = filename_save_vec[i1]
			else:
				filename_prefix_save_query = filename_prefix_vec[i1]
				filename_annot_query = filename_annot_vec[i1]
				output_filename = '%s/%s.%s'%(output_file_path,filename_prefix_save_query,filename_annot_query)

			# merge the files for different subsets of peak-gene links
			if len(input_filename_list)>0:
				df_link_query1 = utility_1.test_file_merge_1(input_filename_list,
																column_vec_query=[],
																index_col=index_col,header=header,float_format=float_format,
																flag_unduplicate=flag_unduplicate,
																axis_join=0,
																save_mode=0,
																output_filename=output_filename,
																verbose=verbose)

				if os.path.exists(output_filename)==True:
					print('the file exists: %s'%(output_filename))
					if overwrite==False:
						b = output_filename.find('.%s'%(extension))
						output_filename = output_filename[0:b] + '.copy1.%s'%(extension)
					
				df_link_query1.to_csv(output_filename,sep='\t')
				print('save data: %s'%(output_filename))
				list_query_1.append(output_filename)
				list_query_2.append(input_filename_list)

		if (flag_reduce>0) and (len(list_query_2)>0):	
			list_query_pre2 = self.test_file_reduce_1(data=list_query_2,select_config=select_config)

		return list_query_1, list_query_2

	## ====================================================
	# combine peak-TF-gene association scores of different subsets of peak-TF-gene links
	def test_feature_link_file_pre2_1(self,iter_mode=0,extension='txt',flag_reduce=1,save_mode=1,verbose=0,select_config={}):

		flag_query_1 = 1
		if flag_query_1>0:
			data_path_save_local = select_config['data_path_save_local']
			path_query_1 = data_path_save_local
			input_file_path = data_path_save_local

			filename_prefix_save_1 = select_config['filename_prefix_default_1']
			
			column_1, column_2 = 'feature_score_interval', 'feature_query_num'
			iter_mode_query = 0
			iter_num = 1
			if (column_1 in select_config) and (column_2 in select_config):
				interval = select_config[column_1]
				feature_query_num_1 = select_config[column_2]
				print('feature_query_num: %d, interval: %d'%(feature_query_num_1,interval))
				if interval>0:
					iter_num = int(np.ceil(feature_query_num_1/interval))
					# print('feature_query_num: %d, interval: %d'%(feature_query_num_1,interval))
					iter_mode_query = 1
			
			dict_query1 = dict()
			# field_query = ['peak_thresh1','peak_thresh2','peak_bg_thresh1','peak_basic','gene_basic']
			# field_num = len(field_query)
			field_query = ['peak_thresh1','peak_thresh2','peak_basic','gene_basic']
			field_num = len(field_query)
			filename_annot_vec = ['combine.thresh1.1','combine.thresh2.1','peak_basic','gene_basic']
			filename_annot_vec = ['%s.%s'%(filename_annot1,extension) for filename_annot1 in filename_annot_vec]
			for field_id in field_query:
				dict_query1[field_id] = []

			annot_str_iter = ''
			for i1 in range(iter_num):
				# t_vec_1 = ['pre1','pre1_bg','pre1']
				t_vec_1 = ['pre1']*2
				if iter_mode_query>0:
					query_id_1 = i1*interval
					query_id_2= (i1+1)*interval
					
					annot_str_1 = '%d_%d'%(query_id_1,query_id_2)
					# t_vec_1 = ['pre1_%s'%(annot_str_1),'pre1_bg_%s'%(annot_str_1),'%s.pre1'%(annot_str_1)]
					t_vec_1 = ['pre1_%s'%(annot_str_1),'%s.pre1'%(annot_str_1)]
				else:
					query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
					print('query_id1:%d, query_id2:%d'%(query_id1,query_id2))
					if (query_id1>=0) and (query_id2>query_id1):
						annot_str_1 = '%d_%d'%(query_id1,query_id2)
						annot_str_iter = annot_str_1
						# t_vec_1 = ['pre1_%s'%(annot_str_1),'pre1_bg_%s'%(annot_str_1),'%s.pre1'%(annot_str_1)]
						t_vec_1 = ['pre1_%s'%(annot_str_1),'%s.pre1'%(annot_str_1)]

				query_vec = ['%s.%s'%(filename_prefix_save_1,annot_str_query) for annot_str_query in t_vec_1]
				# filename_prefix_save_pre1, filename_prefix_save_pre2, filename_prefix_save_pre3 = query_vec[0:3]
				filename_prefix_save_pre1, filename_prefix_save_pre2 = query_vec[0:2]
				filename_prefix_vec = [filename_prefix_save_pre1]*2 + [filename_prefix_save_pre2]*2
				
				for i2 in range(field_num):
					field_id = field_query[i2]
					filename_prefix_query = filename_prefix_vec[i2]
					filename_annot_query = filename_annot_vec[i2]
					input_filename_query = '%s/%s.%s'%(input_file_path,filename_prefix_query,filename_annot_query)
					dict_query1[field_id].append(input_filename_query)
					if verbose>0:
						print('input_filename ',input_filename_query)

			output_file_path = input_file_path
			list_query_1 = [] # paths of the files to retain
			list_query_2 = [] # paths of the files to reduce
			if iter_mode_query>0:
				filename_prefix_save_query = '%s.pre1'%(filename_prefix_save_1)
				filename_prefix_vec = [filename_prefix_save_query]*field_num
				list_query_1, list_query_2 = self.test_query_file_merge_1(data=dict_query1,
															field_query=field_query,
															index_col=0,
															flag_unduplicate=0,
															filename_save_vec=[],
															filename_prefix_vec=filename_prefix_vec,
															filename_annot_vec=filename_annot_vec,
															extension='txt',
															flag_reduce=0,
															save_mode=1,
															output_file_path=output_file_path,
															verbose=0,select_config=select_config)

			else:
				filename_prefix_save_query = filename_prefix_save_pre1
				for field_id in field_query:
					list_query_1 = list_query_1 + dict_query1[field_id]

			import glob
			# extension_vec_query = ['npy','combine.1.txt']
			extension_vec_query = ['npy','txt']
			query_num2 = len(extension_vec_query)
			filename_prefix_vec = [filename_prefix_save_query]*query_num2
			# print('list_query_1')
			# print(list_query_1)
			for i2 in range(query_num2):
				filename_prefix_query = filename_prefix_vec[i2]
				extension_query = extension_vec_query[i2]
				if (iter_mode_query>0) or (annot_str_iter==''):
					# list_query1 = glob.glob('%s/%s_*.%s'%(path_query_1,filename_prefix_query,extension_query))
					list_query1 = glob.glob('%s/%s.*.%s'%(path_query_1,filename_prefix_query,extension_query))
				else:
					list_query1 = glob.glob('%s/*%s*.%s'%(path_query_1,annot_str_iter,extension_query))

				file_num_query1 = len(list_query1)
				# print('list_query1: ',file_num_query1)
				# print(list_query1)
				list_query1 = pd.Index(list_query1).difference(list_query_1,sort=False)
				list_query_2.append(list_query1)
			
			if (flag_reduce>0) and (len(list_query_2)>0):
				list_query_pre2 = self.test_file_reduce_1(data=list_query_2,select_config=select_config)

			return dict_query1

	## ====================================================
	# combine peak-TF-gene association scores of different subsets of peak-TF-gene links
	def test_feature_link_file_pre2_2(self,type_query=0,iter_mode=0,extension='txt',flag_reduce=1,save_mode=1,filename_prefix='',filename_annot='',verbose=0,select_config={}):
		
		flag_query_1=1
		if flag_query_1>0:
			data_path_save_local = select_config['data_path_save_local']
			path_query_1 = data_path_save_local
			path_query_2 = '%s/temp1'%(data_path_save_local)
			input_file_path = path_query_2
			filename_prefix_save_1= select_config['filename_prefix_default_1']

			column_1, column_2 = 'feature_score_interval', 'feature_query_num'
			iter_mode_query = 0
			iter_num = 1
			if (column_1 in select_config) and (column_2 in select_config):
				interval = select_config[column_1]
				feature_query_num_1 = select_config[column_2]
				print('feature_query_num: %d, interval: %d'%(feature_query_num_1,interval))
				if interval>0:
					iter_num = int(np.ceil(feature_query_num_1/interval))
					# print('feature_query_num: %d, interval: %d'%(feature_query_num_1,interval))
					iter_mode_query = 1

			thresh_str_compare = '100_0.15.500_-0.05'
			column_1 = 'thresh_str_compare'
			if column_1 in select_config:
				thresh_str_compare = select_config[column_1]

			filename_annot_compare = 'df_link_query'
			column_2 = 'filename_annot_compare'
			if column_2 in select_config:
				filename_annot_compare = select_config[column_2]

			filename_annot_2 = '%s2.1.combine.%s'%(filename_annot_compare,thresh_str_compare)
			filename_annot_query2 = '%s.2.0_2.2.%s'%(filename_annot_2,extension)
			if type_query==0:
				# only keep the retained pre-selected peak-gene links after comparison
				field_query = ['combine']
				filename_annot_vec = [filename_annot_query2] 
			elif type_query==1:
				# keep retained peak-gene links using positive peak-gene correlations and absolute peak-gene correlations after comparison
				field_query = ['compare_group1','combine']
				annot_str_vec = ['2.0.2']
				filename_annot_vec = ['%s.%s.%s'%(filename_annot_2,annot_str1,extension) for annot_str1 in annot_str_vec] + [filename_annot_query2]
			elif type_query==2:
				# keep retained and filtered peak-gene links after comparison
				field_query = ['compare_group1_1','compare_group2_1','compare_group1_2','compare_group2_2','combine']
				annot_str_vec = ['1.2.0','2.2.0','1.2.2','2.2.2']
				filename_annot_vec = ['%s.%s.%s'%(filename_annot_compare,annot_str1,extension) for annot_str1 in annot_str_vec] + [filename_annot_query2]

			dict_query1 = dict() # the dictionary to keep the filename list of specific annotations
			field_num = len(field_query)
			for field_id in field_query:
				dict_query1[field_id] = []

			for i1 in range(iter_num):
				if iter_mode_query>0:
					query_id_1 = i1*interval
					query_id_2= (i1+1)*interval
					filename_prefix_save_pre1 = '%s.%d_%d'%(filename_prefix_save_1,query_id_1,query_id_2)
				else:
					filename_prefix_save_pre1 = filename_prefix_save_1
					query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
					print('query_id1:%d, query_id2:%d'%(query_id1,query_id2))
					if (query_id1>=0) and (query_id2>query_id1):
						filename_prefix_save_pre1 = '%s.%d_%d'%(filename_prefix_save_1,query_id1,query_id2)

				filename_prefix_vec = [filename_prefix_save_pre1]*field_num
				for i2 in range(field_num):
					field_id = field_query[i2]
					filename_prefix_query = filename_prefix_vec[i2]
					filename_annot_query = filename_annot_vec[i2]
					input_filename_query = '%s/%s.%s'%(input_file_path,filename_prefix_query,filename_annot_query)
					dict_query1[field_id].append(input_filename_query)

			output_file_path = path_query_2
			list_query_1 = [] # paths of the files to retain
			list_query_2 = [] # paths of the files to reduce
			if iter_mode_query>0:
				filename_prefix_save_query = filename_prefix_save_1
				filename_prefix_vec = [filename_prefix_save_query]*field_num
				list_query_1, list_query_2 = self.test_query_file_merge_1(data=dict_query1,
															field_query=field_query,
															index_col=0,
															flag_unduplicate=0,
															filename_save_vec=[],
															filename_prefix_vec=filename_prefix_vec,
															filename_annot_vec=filename_annot_vec,
															extension='txt',
															flag_reduce=0,
															save_mode=1,
															output_file_path=output_file_path,
															verbose=0,select_config=select_config)
			else:
				filename_prefix_save_query = filename_prefix_save_pre1
				for field_id in field_query:
					list_query_1 = list_query_1 + dict_query1[field_id]

			import glob
			extension_vec_query = ['txt','npy']
			query_num2 = len(extension_vec_query)
			filename_prefix_vec = [filename_prefix_save_query]*query_num2
			for i2 in range(query_num2):
				filename_prefix_query = filename_prefix_vec[i2]
				extension_query = extension_vec_query[i2]
				list_query1 = glob.glob('%s/%s.*.%s'%(input_file_path,filename_prefix_query,extension_query)) 
				list_query1 = pd.Index(list_query1).difference(list_query_1,sort=False)
				list_query_2.append(list_query1)

			if (flag_reduce>0) and (len(list_query_2)>0):
				list_query_2 = self.test_file_reduce_1(data=list_query_2,select_config=select_config)
			
		return dict_query1

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)


		