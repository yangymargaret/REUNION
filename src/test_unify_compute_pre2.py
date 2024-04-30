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

import pyranges as pr
import warnings

# import palantir 
import phenograph

import sys
from tqdm.notebook import tqdm

import csv
import os
import os.path
import shutil
from optparse import OptionParser

import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.pipeline import make_pipeline

from scipy import stats
from scipy.stats import multivariate_normal, skew, pearsonr, spearmanr
from scipy.stats import wilcoxon, mannwhitneyu, kstest, ks_2samp, chisquare, fisher_exact, chi2_contingency
from scipy.stats.contingency import expected_freq
from scipy.stats import gaussian_kde, zscore, poisson, multinomial, norm, rankdata
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, issparse, hstack, vstack
from scipy import signal
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import gc
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

import utility_1
from utility_1 import pyranges_from_strings, test_file_merge_1
from utility_1 import spearman_corr, pearson_corr
import h5py
import json
import pickle

import itertools
from itertools import combinations

import test_unify_compute_pre1
from test_unify_compute_pre1 import _Base2_pre1

# get_ipython().run_line_magic('matplotlib', 'inline')
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

class _Base2_correlation2_1(_Base2_pre1):
	"""Feature association estimation;
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

		_Base2_pre1.__init__(self,file_path=file_path,
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

	## load motif data
	def test_motif_data_query_pre1(self,gene_query_vec=[],flag_motif_data_load=1,flag_motif_data_annot=1,flag_chromvar_score=0,flag_copy=1,input_file_path='',save_file_path='',verbose=0,select_config={}):

		## load motif data and motif score data
		# flag_motif_data_load = 0
		motif_data, motif_data_score = [], []
		data_file_type = select_config['data_file_type']
		if 'data_file_type_pre1' in select_config:
			data_file_type = select_config['data_file_type_pre1']

		#filename_save_annot_1 = self.select_config['filename_save_annot_pre1']
		filename_save_annot_1 = select_config['filename_save_annot_pre1']
		if flag_motif_data_load>0:
			print('load TF motif data and motif score data')
			start = time.time()
			motif_data, motif_data_score = self.test_motif_data_query(gene_query_vec=gene_query_vec,input_file_path=input_file_path,save_file_path=save_file_path,flag_chromvar_score=flag_chromvar_score,verbose=verbose,select_config=select_config)
			stop = time.time()
			print('used: %.5fs'%(stop-start))
			self.motif_data = motif_data
			self.motif_data_score = motif_data_score

		## TODO: should update
		# flag_motif_data_annot1=0
		flag_motif_data_annot1=flag_motif_data_annot
		if flag_motif_data_annot1>0:
			# add the field in select_config: 'motif_data_ori', 'motif_data_score_ori'
			self.test_motif_data_query_annot1(motif_data=[],motif_data_score=[],input_filename_annot='',
													column_query='tf',flag_annot_1=1,flag_annot_2=1,
													select_config=select_config)

		motif_data = self.motif_data
		motif_data_score = self.motif_data_score
		load_mode_2 = 0
		if len(gene_query_vec)==0:
			load_mode_2 = 1
			rna_meta_ad = self.rna_meta_ad

		data_file_type_vec_query = [data_file_type]
		data_file_type_vec_1 = data_file_type_vec_query
		if data_file_type in data_file_type_vec_1:
			motif_query_ori = motif_data.columns
			# gene_name_query_ori = meta_scaled_exprs.columns
			if load_mode_2>0:
				gene_name_query_ori = rna_meta_ad.var_names
			else:
				gene_name_query_ori = gene_query_vec
		else:
			motif_query_ori = motif_data.columns.str.upper()
			if load_mode_2>0:
				# gene_name_query_ori = meta_scaled_exprs.columns.str.upper()
				gene_name_query_ori = rna_meta_ad.var_names.str.upper()
			else:
				gene_name_query_ori = pd.Index(gene_query_vec).str.upper()

		print('motif_query_ori ',motif_query_ori)
		print('gene_name_query_ori ',gene_name_query_ori)
		motif_query_name_expr = motif_query_ori.intersection(gene_name_query_ori,sort=False)

		motif_data_ori = motif_data
		motif_data_score_ori = motif_data_score
		# motif_data_expr = motif_data.loc[:,motif_query_name_expr]
		self.motif_query_name_expr = motif_query_name_expr
		if flag_copy>0:
			self.motif_data_ori_2 = motif_data_ori.copy() # keep the peak-motif matrix in which the TFs may not be expressed
			self.motif_data_score_ori_2 = motif_data_score_ori.copy()
		
		print('motif_data_ori, motif_data_score_ori ', motif_data_ori.shape, motif_data_score_ori.shape)
		print(motif_data_ori[0:5])
		print(motif_data_score_ori[0:5])
		print('motif_query_name_expr ',len(motif_query_name_expr))

		motif_data = motif_data_ori.loc[:,motif_query_name_expr]
		motif_data_score = motif_data_score_ori.loc[:,motif_query_name_expr]
			
		print('motif_data, motif_data_score ', motif_data.shape, motif_data_score.shape)
		print(motif_data[0:5])
		print(motif_data_score[0:5])
		print('motif_query_name_expr ',len(motif_query_name_expr))

		self.motif_data = motif_data
		self.motif_data_score = motif_data_score

		return motif_data, motif_data_score

	## query motif data and motif score data
	def test_motif_data_query(self,gene_query_vec=[],input_file_path='',save_file_path='',type_id_1=0,flag_chromvar_score=0,verbose=0,select_config={}):

		## load motif data and motif score data
		flag_motif_data_load = 1
		motif_data, motif_data_score = [], []
		self.motif_data = motif_data
		self.motif_data_score = motif_data_score
		data_file_type = select_config['data_file_type']

		if flag_motif_data_load>0:
			print('load motif data and motif score data')
			
			if ('motif_filename1' in select_config) and ('motif_filename2' in select_config):
				input_filename1 = select_config['motif_filename1']
				input_filename2 = select_config['motif_filename2']
				input_filename_list1 = [input_filename1,input_filename2]
			else:
				input_filename_list1 = []
			input_filename_list2 = []
			# 	input_filename_1 = '%s/test_peak_read.%s.normalize.1_motif.1.2.csv'%(input_file_path,filename_save_annot_1)
			# 	input_filename_2 = '%s/test_peak_read.%s.normalize.1_motif_scores.1.csv'%(input_file_path,filename_save_annot_1)
			input_filename_1 = ''
			input_filename_2 = ''
			if 'motif_filename_1' in select_config:
				input_filename_1 = select_config['motif_filename_1']
			if 'motif_filename_2' in select_config:
				input_filename_2 = select_config['motif_filename_2']
			input_filename_list2 = [input_filename_1,input_filename_2]

			## the motif name was changed to TF name
			# should update; there may be multiple motif names for one TF
			type_id_1 = 0
			motif_data, motif_data_score, df_annot, type_id_query = self.test_load_motif_data_pre1(input_filename_list1=input_filename_list1,
																									input_filename_list2=input_filename_list2,
																									flag_query1=1,
																									flag_query2=1,
																									input_file_path=input_file_path,
																									save_file_path=save_file_path,
																									type_id_1=type_id_1,
																									type_id_2=1,
																									verbose=verbose,
																									select_config=select_config)

			# type_id_query = 1
			print('type_id_query: %d'%(type_id_query))
			if type_id_query>0:
				column_id = 'tf'
				# merge columns of motifs of one TF into one column
				print('merge columns of motifs of one TF into one column, motif_data')
				motif_data, motif_data_ori = self.test_load_motif_data_pre2(motif_data=motif_data,
																			df_annot=df_annot,
																			column_id=column_id,
																			select_config=select_config)

				# merge columns of motifs of one TF into one column
				print('merge columns of motifs of one TF into one column, motif_data_score')
				motif_data_score, motif_data_score_ori = self.test_load_motif_data_pre2(motif_data=motif_data_score,
																						df_annot=df_annot,
																						column_id=column_id,
																						select_config=select_config)
				# b = input_filename1.find('.h5ad')
				# output_filename1 = input_filename1[0:b]+'.1.h5ad'
				output_filename1 = select_config['motif_filename1']
				motif_data_ad = sc.AnnData(motif_data,dtype=motif_data.values.dtype)
				motif_data_ad.X = csr_matrix(motif_data_ad.X)
				motif_data_ad.write(output_filename1)

				# b = input_filename2.find('.h5ad')
				# output_filename2 = input_filename2[0:b]+'.1.h5ad'
				output_filename2 = select_config['motif_filename2']
				motif_data_score_ad = sc.AnnData(motif_data_score,dtype=motif_data_score.values.dtype)
				motif_data_score_ad.X = csr_matrix(motif_data_score_ad.X)
				motif_data_score_ad.write(output_filename2)

			self.motif_data = motif_data
			self.motif_data_score = motif_data_score
			# motif_query_ori = motif_data.columns.str.upper()
			motif_query_ori = motif_data.columns

			if len(gene_query_vec)==0:
				gene_name_query_ori = self.rna_meta_ad.var_names
				# gene_name_query_ori = meta_scaled_exprs.columns.str.upper()
				# print('motif_query_ori ',motif_query_ori)
				# print('gene_name_query_ori ',gene_name_query_ori)
			else:
				gene_name_query_ori = gene_query_vec
			motif_query_name_expr = motif_query_ori.intersection(gene_name_query_ori,sort=False)
			self.motif_query_name_expr = motif_query_name_expr
			self.df_motif_translation = df_annot
			# self.motif_data_expr = self.motif_data.loc[:,motif_query_name_expr]
			# print('motif_data, motif_data_score, motif_data_expr ', motif_data.shape, motif_data_score.shape, self.motif_data_expr.shape)
			print('motif_data, motif_data_score, motif_query_name_expr ', motif_data.shape, motif_data_score.shape, len(motif_query_name_expr))
			print(motif_query_name_expr[0:10])

			if flag_chromvar_score>0:
				if flag_motif_data_load>0:
					df1 = df_annot
				else:
					if 'filename_translation' in select_config:
						input_filename = select_config['filename_translation']
					else:
						print('please provide motif name translation file')
						return
					df1 = pd.read_csv(input_filename,index_col=0,sep='\t')
			
				column_id1 = 'motif_id'
				if 'column_motif' in select_config:
					column_id1 = select_config['column_motif']
				df1.index = np.asarray(df1[column_id1])
				filename_save_annot_1 = select_config['filename_save_annot_pre1']

				if 'filename_chromvar_score' in select_config:
					input_filename = select_config['filename_chromvar_score']
				else:
					input_filename = '%s/test_peak_read.%s.normalize.1_chromvar_scores.1.csv'%(input_file_path,filename_save_annot_1)

				if os.path.exists(input_filename)==False:
					print('the file does not exist: %s'%(input_filename))
					print('please provide chromVAR score file')
					return

				## query correlation and mutual information between chromvar score and TF expression
				# output_file_path = input_file_path
				output_file_path = save_file_path
				df_query_1 = df1
				data_file_type = select_config['data_file_type']
				# type_id_query2 = 2
				if data_file_type in ['CD34_bonemarrow']:
					type_id_query2 = 0
				elif data_file_type in ['pbmc']:
					type_id_query2 = 1
				else:
					type_id_query2 = 2

				# filename_save_annot_1 = select_config['filename_save_annot_pre1']
				df_2 = self.test_chromvar_score_query_1(input_filename=input_filename,
														motif_query_name_expr=motif_query_name_expr,
														df_query=df_query_1,
														output_file_path=output_file_path,
														filename_prefix_save=filename_save_annot_1,
														type_id_query=type_id_query2,
														select_config=select_config)
				# return

			# return motif_data, motif_data_score, motif_query_name_expr, df_annot
			return motif_data, motif_data_score

	## load motif data
	def test_load_motif_data_pre1(self,input_filename_list1=[],input_filename_list2=[],flag_query1=1,flag_query2=1,input_file_path='',
										save_file_path='',type_id_1=0,type_id_2=1,verbose=0,select_config={}):
		
		flag_pre1=0
		motif_data, motif_data_score = [], []
		type_id_query = type_id_1
		# flag_load1 = 1
		if len(input_filename_list1)>0:
			## load from the processed anndata
			input_filename1, input_filename2 = input_filename_list1
			if (os.path.exists(input_filename1)==False):
				print('the file does not exist: %s'%(input_filename1))

			if (os.path.exists(input_filename2)==False):
				print('the file does not exist: %s'%(input_filename2))

			if (os.path.exists(input_filename1)==True) and (os.path.exists(input_filename2)==True):
				motif_data_ad = sc.read(input_filename1)
				try:
					motif_data = pd.DataFrame(index=motif_data_ad.obs_names,columns=motif_data_ad.var_names,data=motif_data_ad.X.toarray())
				except Exception as error:
					print('error! ',error)
					motif_data = pd.DataFrame(index=motif_data_ad.obs_names,columns=motif_data_ad.var_names,data=np.asarray(motif_data_ad.X))

				motif_data_score_ad = sc.read(input_filename2)
				try:
					motif_data_score = pd.DataFrame(index=motif_data_score_ad.obs_names,columns=motif_data_score_ad.var_names,data=motif_data_score_ad.X.toarray(),dtype=np.float32)
				except Exception as error:
					motif_data_score = pd.DataFrame(index=motif_data_score_ad.obs_names,columns=motif_data_score_ad.var_names,data=np.asarray(motif_data_score_ad.X))
			
				print(input_filename1,input_filename2)
				print('motif_data ', motif_data.shape)
				print('motif_data_score ', motif_data_score.shape)
				print(motif_data[0:2])
				print(motif_data_score[0:2])
				flag_pre1 = 1
				# flag_load1 = 0

		## motif name correction for the conversion in R
		# meta_scaled_exprs = self.meta_scaled_exprs
		df_gene_annot = self.df_gene_annot_ori
		
		if save_file_path=='':
			save_file_path = select_config['data_path_save']
			# save_file_path = select_config['data_path_1']
			input_path_path = save_file_path
		
		# input_filename = '%s/translationTable.csv'%(save_file_path)
		input_filename = select_config['filename_translation']
		print('filename_translation: ',input_filename)
		df_annot = []
		overwrite = 0
		flag_query1 = 1
		flag_annot1 = 0
	
		# if (os.path.exists(input_filename)==False) or (flag_pre1==0):
		# if (flag_annot1>0) or (flag_pre1==0):
		# 	# motif_data, motif_data_score = [], []
		# 	input_filename1, input_filename2 = input_filename_list2
		# 	b = input_filename1.find('.csv')
		# 	if b>=0:
		# 		symbol_1 = ','
		# 	else:
		# 		symbol_1 = '\t'
		# 	if os.path.exists(input_filename1)==False:
		# 		print('the file does not exist: %s'%(input_filename1))
		# 	else:
		# 		motif_data = pd.read_csv(input_filename1,index_col=0,sep=symbol_1)
		# 	if os.path.exists(input_filename2)==False:
		# 		print('the file does not exist: %s'%(input_filename2))
		# 	else:
		# 		motif_data_score = pd.read_csv(input_filename2,index_col=0,sep=symbol_1)

		# 	if len(motif_data)==0:
		# 		if len(motif_data_score)>0:
		# 			# motif_data = (motif_data_score>0)
		# 			motif_data = (motif_data_score.abs()>0)
		# 		else:
		# 			print('please provide motif data')
		# 			return

		# 	peak_loc = motif_data.index
		# 	peak_loc_1 = motif_data_score.index
		# 	assert list(peak_loc)==list(peak_loc_1)
		# 	# peak_loc = self.atac_meta_ad.obs_names
		# 	# motif_data = motif_data.loc[peak_loc,:]
		# 	# motif_data_score = motif_data_score.loc[peak_loc,:]
		# 	# flag_query1 = 1
		# elif overwrite==0:
		# 	flag_query1 = 0
		# else:
		# 	pass

		if os.path.exists(input_filename)==False:
			print('the file does not exist: %s'%(input_filename))
			flag_annot1 = 1
			input_filename1, input_filename2 = input_filename_list2
			
			if os.path.exists(input_filename1)==False:
				print('the file does not exist: %s'%(input_filename1))
			else:
				b = input_filename1.find('.csv')
				if b>=0:
					symbol_1 = ','
				else:
					symbol_1 = '\t'
				motif_data = pd.read_csv(input_filename1,index_col=0,sep=symbol_1)
			if os.path.exists(input_filename2)==False:
				print('the file does not exist: %s'%(input_filename2))
			else:
				b = input_filename2.find('.csv')
				if b>=0:
					symbol_1 = ','
				else:
					symbol_1 = '\t'
				motif_data_score = pd.read_csv(input_filename2,index_col=0,sep=symbol_1)

			if len(motif_data)==0:
				if len(motif_data_score)>0:
					# motif_data = (motif_data_score>0)
					motif_data = (motif_data_score.abs()>0)
				else:
					print('please provide motif data')
					return

			# flag_query1 = 1		
		elif overwrite==0:
			flag_query1 = 0
		else:
			pass

		if verbose>0:
			if len(motif_data)>0:
				print('motif_data ', motif_data.shape)
				print(motif_data[0:2])
			if len(motif_data_score)>0:
				print('motif_data_score ', motif_data_score.shape)
				print(motif_data_score[0:2])

		if flag_query1>0:
			output_filename = input_filename
			# meta_scaled_exprs = self.meta_scaled_exprs
			df_annot = self.test_translationTable_pre1(motif_data=motif_data,
														df_gene_annot=df_gene_annot,
														save_mode=1,
														save_file_path=save_file_path,
														output_filename=output_filename,
														select_config=select_config)
		else:
			df_annot = pd.read_csv(input_filename,index_col=0,sep='\t')

		column_motif = 'motif_id'
		if column_motif in select_config:
			column_motif = select_config['column_motif']

		# df_annot.index = np.asarray(df_annot['motif_id'])
		df_annot.index = np.asarray(df_annot[column_motif])
		if flag_pre1==0:
			if len(motif_data)==0:
				load_mode_2 = 0
				input_filename1, input_filename2 = input_filename_list2
				motif_data, motif_data_score = [], []
				if os.path.exists(input_filename1)==True:
					motif_data = pd.read_csv(input_filename1,index_col=0)
				else:
					print('the file does not exist: %s'%(input_filename1))
					load_mode_2 += 1

				if os.path.exists(input_filename2)==True:
					motif_data_score = pd.read_csv(input_filename2,index_col=0)
				else:
					print('the file does not exist: %s'%(input_filename2))
					load_mode_2 += 1

				if (len(motif_data)==0) and (len(motif_data_score)>0):
					# motif_data = (motif_data_score>0)
					motif_data = (motif_data_score.abs()>0)

				if load_mode_2==2:
					print('please provide motif data file')
					return

			type_id_query = 1
			motif_data_1 = (motif_data_score>0)*1.0
			motif_data_2 = (motif_data_score.abs()>0)*1.0
			# difference = np.abs(motif_data-motif_data_1)
			difference = np.abs(motif_data-motif_data_2)
			assert np.max(np.max(difference))==0

			## motif name query
			motif_name_ori = motif_data.columns
			motif_name_score_ori = motif_data_score.columns
			# peak_loc = motif_data.index
			# peak_loc_1 = motif_data_score.index
			assert list(motif_name_ori)==list(motif_name_score_ori)
			# assert list(peak_loc)==list(peak_loc_1)
			
			print('motif_data ', input_filename1, motif_data.shape)
			print(motif_data[0:5])
			print('motif_data_score ', input_filename2, motif_data_score.shape)
			print(motif_data_score[0:5])

			df1 = df_annot
			# motif_name_query =  np.asarray(df1['motif_name'])
			df1.index = np.asarray(df1['motif_id'])
			motif_name_ori = motif_data.columns
			motif_name_query = np.asarray(df1.loc[motif_name_ori,'tf'])
			
			motif_data_score_ad = test_save_anndata(motif_data_score,dtype=motif_data_score.values.dtype)
			motif_data_ad = test_save_anndata(motif_data,dtype=motif_data.values.dtype)

			if len(input_filename_list1)>0:
				output_filename_list = input_filename_list1
				output_filename1, output_filename2 = output_filename_list
			else:
				data_path_save = select_config['data_path_save_motif']
				if 'data_file_query_motif' in select_config:
					data_file_type_query = select_config['data_file_query_motif']
				else:
					data_file_type_query = select_config['data_file_type']

				motif_filename1 = '%s/test_motif_data.%s.ori.h5ad'%(data_path_save,data_file_type_query)
				motif_filename2 = '%s/test_motif_data_score.%s.ori.h5ad'%(data_path_save,data_file_type_query)

				# select_config.update({'motif_filename1':motif_filename1,'motif_filename2':motif_filename2})
				output_filename1, output_filename2 = motif_filename1, motif_filename2
			
			if os.path.exists(output_filename1)==True:
				print('the file exists ', output_filename1)
			motif_data_ad.write(output_filename1)
			print('output_filename1: ',output_filename1)
			
			if os.path.exists(output_filename2)==True:
				print('the file exists ', output_filename2)
			motif_data_score_ad.write(output_filename2)
			print('output_filename2: ',output_filename2)

		flag_query2=0
		if flag_query2>0:
			df1 = (motif_data_score<0)
			id2 = motif_data_score.columns[df1.sum(axis=0)>0]
			if len(id2)>0:
				motif_data_score_ori = motif_data_score.copy()
				count1 = np.sum(np.sum(df1))
				print('there are negative motif scores ',id2,count1)
				df_1 = (motif_data_score>0)
				motif_data_score_1 = motif_data_score[df_1]
				# print('motif_data_score_1 ',motif_data_score_1)
				t_value_min = np.min(motif_data_score_1,axis=0)
				df_value_min = np.outer(np.ones(motif_data_score.shape[0]),np.asarray(t_value_min))
				df2 = pd.DataFrame(index=motif_data_score.index,columns=motif_data_score.columns,data=np.asarray(df_value_min),dtype=np.float32)
				id1 = df1
				motif_data_score[id1] = df2[id1]
				peak_loc_ori = motif_data_score.index
				peak_id2 = peak_loc_ori[motif_data_score_ori.loc[:,id2].min(axis=1)<0]

		try:
			peak_loc = self.atac_meta_ad.var_names
			motif_data = motif_data.loc[peak_loc,:]
			motif_data_score = motif_data_score.loc[peak_loc,:]
		except Exception as error:
			print('error! ',error)
			peak_loc = motif_data.index
			motif_data_score = motif_data_score.loc[peak_loc,:]
			
		if (verbose>0):
			print('motif_data, motif_data_score: ',motif_data.shape,motif_data_score.shape)

		return motif_data, motif_data_score, df_annot, type_id_query

	## load motif data
	# merge columns of motifs of one TF into one column
	def test_load_motif_data_pre2(self,motif_data,df_annot,column_id='tf',select_config={}):

		# motif_idvec_1= df_annot1.index
		if 'motif_id' in df_annot:
			df_annot.index = np.asarray(df_annot['motif_id'])
		motif_idvec = motif_data.columns.intersection(df_annot.index,sort=False)
		motif_data = motif_data.loc[:,motif_idvec]
		motif_data_ori = motif_data.copy()
		motif_data1 = motif_data.T
		motif_idvec = motif_data1.index
		motif_data1.loc[:,'tf'] = df_annot.loc[motif_idvec,column_id]
		motif_data1 = motif_data1.groupby('tf').max()
		motif_data = motif_data1.T
		# print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape,method_type)
		print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape)
		print(motif_data[0:5])
		# field_id = '%s.ori'%(key_query)
		# if not (field_id in dict_query):
		# 	dict_query.update({'%s.ori'%(key_query):motif_data_ori})
		return motif_data, motif_data_ori

	## chromvar score query: chromvar score comparison with TF expression
	# query correlation and mutual information between chromvar score and TF expression
	def test_chromvar_score_query_1(self,input_filename,motif_query_name_expr,df_gene_annot_expr=[],filename_prefix_save='',output_file_path='',output_filename='',
										df_query=[],type_id_query=0,select_config={}):

		df1 = df_query
		# df1.index = np.asarray(df1['motif_name_ori'])
		print('df1 ',df1.shape)
		print(df1)
		column_motif = 'motif_id'
		if 'column_motif' in select_config:
			column_motif = select_config['column_motif']
		print('column_motif: ',column_motif)
		# df1.index = np.asarray(df1['motif_id'])
		df1.index = np.asarray(df1[column_motif])
		print('df1 ',df1.shape)
		print(df1)

		chromvar_score = pd.read_csv(input_filename,index_col=0,sep=',')
		print('chromvar_score ', chromvar_score.shape)
		print(chromvar_score)
		sample_id1 = chromvar_score.columns
		motif_id1 = chromvar_score.index
		# chromvar_score.index = df1.loc[motif_id1,'motif_name']
		chromvar_score.index = df1.loc[motif_id1,'tf']
		if type_id_query==0:
			str_vec_1 = sample_id1.str.split('.')
			str_query_list = [str_vec_1.str.get(i1) for i1 in range(3)]
			str_query1, str_query2, str_query3 = str_query_list
			query_num2 = len(str_query1)
			chromvar_score.columns = ['%s#%s-%s'%(str_query1[i2],str_query2[i2],str_query3[i2]) for i2 in range(query_num2)]
		
		elif type_id_query==1:
			str_vec_1 = sample_id1.str.split('.')
			str_query_list = [str_vec_1.str.get(i1) for i1 in range(2)]
			str_query1, str_query2 = str_query_list
			query_num2 = len(str_query1)
			chromvar_score.columns = ['%s-%s'%(str_query1[i2],str_query2[i2]) for i2 in range(query_num2)]
		
		else:
			data_file_type = select_config['data_file_type']
			if data_file_type in ['system1']:
				str_vec_1 = sample_id1.str.split('_').str
				str1 = str_vec_1.get(0)
				str2 = str_vec_1.get(1)
				str_vec_2 = pd.Index(str2).str.split('.').str

				str_query_list = [str_vec_2.get(i1) for i1 in range(3)]
				str_query1, str_query2, str_query3 = str_query_list
				query_num2 = len(str_query1)
				chromvar_score.columns = ['%s_%s#%s-%s'%(str1[i2],str_query1[i2],str_query2[i2],str_query3[i2]) for i2 in range(query_num2)]
			else:
				print('chromvar_score: use the loaded columns')
		
		rna_ad = self.rna_meta_ad
		meta_scaled_exprs = self.meta_scaled_exprs
		assert list(chromvar_score.columns)==list(rna_ad.obs_names)
		assert list(chromvar_score.columns)==list(meta_scaled_exprs.index)
		if output_file_path=='':
			data_path_save = select_config['data_path_save']
			output_file_path = data_path_save
		
		if output_filename=='':
			b = input_filename.find('.csv')
			output_filename = input_filename[0:b]+'.copy1.csv'
		chromvar_score.to_csv(output_filename)
		print('chromvar_score ',chromvar_score.shape,chromvar_score)

		chromvar_score = chromvar_score.loc[~chromvar_score.index.duplicated(keep='first'),:]
		chromvar_score = chromvar_score.T
		sample_id = meta_scaled_exprs.index
		chromvar_score = chromvar_score.loc[sample_id,:]

		motif_query_vec = motif_query_name_expr
		motif_query_num = len(motif_query_vec)
		print('motif_query_vec ',motif_query_num)
		field_query_1 = ['spearmanr','pval1','pearsonr','pval2','mutual_info']
		df_1 = pd.DataFrame(index=motif_query_vec,columns=field_query_1)
		for i1 in range(motif_query_num):
			motif_query1 = motif_query_vec[i1]
			tf_expr_1 = np.asarray(meta_scaled_exprs[motif_query1])
			tf_score_1 = np.asarray(chromvar_score[motif_query1])
			# corr_value, pvalue = spearmanr(meta_scaled_exprs[motif_query1],chromvar_score[motif_query1])
			corr_value_1, pval1 = spearmanr(tf_expr_1,tf_score_1)
			corr_value_2, pval2 = pearsonr(tf_expr_1,tf_score_1)
			t_mutual_info = mutual_info_regression(tf_expr_1[:,np.newaxis], tf_score_1, discrete_features=False, n_neighbors=5, copy=True, random_state=0)
			t_mutual_info = t_mutual_info[0]
			df_1.loc[motif_query1,:] = [corr_value_1,pval1,corr_value_2,pval2,t_mutual_info]

		df_1 = df_1.sort_values(by=field_query_1,ascending=[False,True,False,True,False])
		
		filename = output_filename
		b = filename.find('.csv')
		output_filename = '%s.copy2.txt'%(filename[0:b])
		field_query_2 = ['highly_variable','means','dispersions','dispersions_norm']
		
		if len(df_gene_annot_expr)==0:
			df_gene_annot_expr = self.df_gene_annot_expr
		df_gene_annot_expr.index = np.asarray(df_gene_annot_expr['gene_name'])
		gene_name_query_ori = df_gene_annot_expr.index
		motif_id1 = df_1.index
		motif_id2 = motif_id1.intersection(gene_name_query_ori,sort=False)
		df_1.loc[motif_id2,field_query_2] = df_gene_annot_expr.loc[motif_id2,field_query_2]
		df_1.to_csv(output_filename,sep='\t',float_format='%.6E')
		mean_value = df_1.mean(axis=0)
		median_value = df_1.median(axis=0)
		print('df_1, mean_value, median_value ',df_1.shape,mean_value,median_value)

		df_2 = df_1.sort_values(by=['highly_variable','dispersions_norm','means','spearmanr','pval1','pearsonr','pval2','mutual_info'],ascending=[False,False,False,False,True,False,True,False])
		# output_filename = '%s/test_peak_read.%s.normalize.chromvar_scores.tf_expr.query1.sort2.1.txt'%(output_file_path,filename_prefix_save)
		df_2.to_csv(output_filename,sep='\t',float_format='%.6E')

		df_sort2 = df_1.sort_values(by=['spearmanr','highly_variable','dispersions_norm','means','pval1','pearsonr','pval2','mutual_info'],ascending=[False,False,False,False,True,False,True,False])
		# output_filename = '%s/test_peak_read.%s.normalize.chromvar_scores.tf_expr.query1.sort2.1.txt'%(output_file_path,filename_prefix_save)
		output_filename_2 = '%s.sort2.copy2.txt'%(filename[0:b])
		df_sort2.to_csv(output_filename_2,sep='\t',float_format='%.6E')

		id1 = (df_2['highly_variable']==True)
		motif_id2 = df_2.index
		motif_query_2 = motif_id2[id1]
		motif_query_num2 = len(motif_query_2)
		motif_query_3 = motif_id2[~id1]
		motif_query_num3 = len(motif_query_3)
		mean_value = df_2.loc[id1,:].mean(axis=0)
		median_value = df_2.loc[id1,:].median(axis=0)
		mean_value_2 = df_2.loc[(~id1),:].mean(axis=0)
		median_value_2 = df_2.loc[(~id1),:].median(axis=0)
		print('highly_variable tf expr, mean_value, median_value ',motif_query_num2,mean_value,median_value)
		print('group 2 tf expr, mean_value, median_value ',motif_query_num3,mean_value_2,median_value_2)

		return df_2

	## prepare translationTable
	def test_translationTable_pre1(self,motif_data=[],motif_data_score=[],df_gene_annot=[],meta_scaled_exprs=[],
										save_mode=1,save_file_path='',output_filename='',flag_cisbp_motif=1,flag_expr=1,select_config={}):

		motif_name_ori = motif_data.columns
		if flag_cisbp_motif>0:
			# motif name correction for the name conversion in R
			# t_vec_str = motif_name_ori.str.split('_').str
			# motif_name = motif_name_ori.str.split('_').str.get(2)	# gene name
			# gene_id = motif_name_ori.str.split('_').str.get(0)	# ENSEMBL id
			motif_num = len(motif_name_ori)
			motif_name = np.array(motif_name_ori)
			gene_id = motif_name.copy()
			for i1 in range(motif_num):
				motif_id = motif_name_ori[i1]
				t_vec_str1 = pd.Index(motif_id.split('_'))
				b1 = t_vec_str1.str.find('LINE')
				b2 = np.where(b1>=0)[0]
				loc_id = b2[-1]+1
				# if loc_id>2:
				# 	print(t_vec_str1)
				# 	print(loc_id)
				motif_name[i1] = t_vec_str1[loc_id] # gene name
				gene_id[i1] = t_vec_str1[0] # ENSEMBL id

			# motif_data.columns = motif_name
			str_vec_1 = ['NKX2','NKX1','NKX3','NKX6']
			str_vec_2 = ['NKX2-','NKX1-','NKX3-','NKX6-']
			str_vec_1 = str_vec_1 + ['Nkx2','Nkx1','Nkx3','Nkx6']
			str_vec_2 = str_vec_2 + ['Nkx2-','Nkx1-','Nkx3-','Nkx6-']
			# motif_name_1 = motif_data.columns.str.replace('Nkx2','Nkx2-')
			query_num1 = len(str_vec_1)		
			# motif_name_1 = motif_data.columns
			motif_name_1 = pd.Index(motif_name)
			for i1 in range(query_num1):
				motif_name_1 = pd.Index(motif_name_1).str.replace(str_vec_1[i1],str_vec_2[i1])

			df1 = pd.DataFrame.from_dict(data={'motif_id':motif_name_ori,'tf':motif_name_1},orient='columns')

			df1['gene_id'] = np.asarray(gene_id)
			df1.index = np.asarray(df1['gene_id'].str.upper())
			# df1 = df1.rename(columns={'gene_id':'ENSEMBL'})

			gene_id_1 = df_gene_annot['gene_id'].str.upper()
			df_gene_annot.index = np.asarray(gene_id_1)
			motif_query_id = df1.index.intersection(gene_id_1,sort=False)

			df1.loc[:,'tf_ori'] = df1.loc[:,'tf'].copy()
			df1.loc[motif_query_id,'tf'] = df_gene_annot.loc[motif_query_id,'gene_name']
			tf_name = np.asarray(df1['tf'])
			b1 = np.where(tf_name=='Pit1')[0]
			tf_name[b1] = 'Pou1f1'
			df1['tf'] = tf_name
			# tf_name = df1['tf']
			if flag_expr>0:
				df_var = self.rna_meta_ad.var
				if flag_expr>1:
					# motif name query by gene id
					if 'gene_id' in df_var.columns:
						gene_id_2 = df_var['gene_id'].str.upper()
						motif_query_id_expr = df1.index.intersection(gene_id_2,sort=False)
						df1.loc[motif_query_id_expr,'tf_expr'] = 1
						df_var['gene_name'] = df_var.index.copy()
						df_var.index = np.asarray(df_var['gene_id'])
						df1.loc[motif_query_id_expr,'tf'] = df_var.loc[motif_query_id_expr,'gene_name']
						df_var.index = np.asarray(df_var['gene_name']) # reset the index
						motif_query_name_expr = np.asarray(df1.loc[motif_query_id_expr,'tf'])
					else:
						flag_expr = 1

				if flag_expr==1:
					# motif name query by gene name
					gene_name_expr = self.rna_meta_ad.var_names
					output_file_path = select_config['data_path_save_local']
					# output_filename_2 = '%s/test_rna_meta_ad.df_var.query1.txt'%(output_file_path)
					if 'data_file_type_1' in select_config:
						data_file_type_query_1 = select_config['data_file_type_1']
					else:
						data_file_type_query_1 = select_config['data_file_type']
					output_filename_2 = '%s/test_rna_meta_ad.%s.df_var.query1.txt'%(output_file_path,data_file_type_query_1)
					df_var.to_csv(output_filename_2,sep='\t')
					motif_query_name_expr = pd.Index(tf_name).intersection(gene_name_expr,sort=False)
					df1.index = np.asarray(df1['tf'])
					df1.loc[motif_query_name_expr,'tf_expr'] = 1
					
				df1.index = np.asarray(df1['gene_id'])
				self.motif_query_name_expr = motif_query_name_expr

				print('motif_query_name_expr ',len(motif_query_name_expr))

			if save_mode>0:
				if output_filename=='':
					output_filename = '%s/translationTable.csv'%(save_file_path)
				df1.to_csv(output_filename,sep='\t')

		return df1

	## motif data query
	# motif name linked to TF name
	def test_motif_data_query_annot1(self,motif_data=[],motif_data_score=[],input_filename_annot='',column_query='tf',
										flag_annot_1=1,flag_annot_2=1,select_config={}):
		
		if len(motif_data)==0:
			motif_data = self.motif_data
		if len(motif_data_score)==0:
			motif_data_score = self.motif_data_score
		motif_idvec = []
		## motif name mapped to TF name
		if flag_annot_1>0:
			if input_filename_annot=='':
				if 'input_filename_motif_annot' in select_config:
					input_filename_annot = select_config['input_filename_motif_annot']
				else:
					data_path = select_config['data_path']
					input_filename_annot = '%s/TFBS/translationTable.csv'%(data_path) # the default motif annotation filename

			if os.path.exists(input_filename_annot)==False:
				print('the file does not exist: %s; please provide motif annotation filename'%(input_filename_annot))
				return

			df_annot1 = pd.read_csv(input_filename_annot,index_col=0,sep=' ')
			if 'motif_id' in df_annot1:
				df_annot1.index = np.asarray(df_annot1['motif_id'])
			motif_idvec = motif_data.columns.intersection(df_annot1.index,sort=False)

		motif_data_ori = motif_data
		dict_file_query = {'motif_data':motif_data_ori,'motif_data_score':motif_data_score}
		# motif_data_list = [motif_data,motif_data_score]
		key_vec = list(dict_file_query.keys()) # key_vec: ['motif_data','motif_data_score']
		query_num2 = len(key_vec)
		type_id=0
		if len(motif_idvec)==0:
			print('motif data has been annotated with TF name')
			type_id=1
		
		if flag_annot_2>0:
			for i1 in range(query_num2):
				key_query = key_vec[i1]
				motif_data = dict_file_query[key_query]
				if len(motif_data)==0:
					print('the data not included:',key_query)
					continue

				if type_id==0:
					motif_data = motif_data.loc[:,motif_idvec]
				motif_data_ori = motif_data.copy()
				motif_data1 = motif_data.T
				motif_idvec = motif_data1.index
				column_annot = 'tf_name'
				if type_id==0:
					motif_data1.loc[:,column_annot] = df_annot1.loc[motif_idvec,column_query]
				else:
					motif_data1.loc[:,column_annot] = np.asarray(motif_data1.index)

				# motif_data1 = motif_data1.groupby('tf').max()
				motif_data1 = motif_data1.groupby(column_annot).max()
				motif_data = motif_data1.T
				print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape)
				print(motif_data[0:2])
				field_id = '%s_ori'%(key_query)
				if not (field_id in dict_file_query):
					dict_file_query.update({'%s_ori'%(key_query):motif_data_ori})
			
			self.motif_data_ori = dict_file_query['motif_data_ori'] # the motif data with the original motif query; one TF may have multiple motif query
			self.motif_data_score_ori = dict_file_query['motif_data_score_ori']
			self.motif_data = dict_file_query['motif_data']
			self.motif_data_score = dict_file_query['motif_data_score']

			return dict_file_query

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

		# input_filename_bg = '%s/chromvar/test_e875_endoderm_rna_chromvar_bg.%s.2.csv'%(self.path_1,annot1)
		# print(input_filename_bg)
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
		# print('background peaks', peak_bg.shape, len(peak_id), peak_bg_num, peak_bg.index[0:10])
		if verbose>0:
			print(input_filename_peak)
			print(input_filename_bg)
			print('atac matecell peaks', len(self.atac_meta_peak_loc), self.atac_meta_peak_loc[0:5])
			print('background peaks', peak_bg.shape, len(peak_id), peak_bg_num)

		return peak_bg

	# the chromvar score calculation
	def test_chromvar_estimate_pre1(self,peak_loc=[],peak_sel_local=[],motif_query=[],motif_data=[],rna_exprs=[],peak_read=[],atac_read_expected=[],est_mode=1,type_id_est=0,peak_bg_num=100,parallel_mode=0,parallel_interval=-1,save_mode=1,output_filename='',select_config={}):

		input_file_path1 = self.save_path_1
		flag1 = 1
		if flag1>0:
			atac_read_mtx = peak_read
			type_id_est1 = type_id_est
			print('compute chromvar score')
			vec1_local = self.test_chromvar_estimate_pre2(atac_read=atac_read_mtx,peak_loc=peak_loc,
															peak_sel_local=peak_sel_local,
															atac_read_expected=atac_read_expected, 
															motif_query=motif_query,motif_data=motif_data, 
															peak_subset_id=0,
															expected_score=[],type_id_1=type_id_est1)
		
			deviation1, score1, expected_score1, peak_id = vec1_local
			print('score1', score1.shape, np.asarray(score1)[0,0:2])
			print('expected_score1', expected_score1.shape, np.asarray(expected_score1)[0,0:2])
			print('deviation', deviation1.shape, np.asarray(deviation1)[0,0:2])

			sample_id = peak_read.index
			df_deviation1 = pd.DataFrame(index=sample_id,columns=motif_query,data=np.asarray(deviation1),dtype=np.float32)
			df_score1 = pd.DataFrame(index=sample_id,columns=motif_query,data=np.asarray(score1),dtype=np.float32)
			df_expected_score1 = pd.DataFrame(index=sample_id,columns=motif_query,data=np.asarray(expected_score1),dtype=np.float32)

			# if peak_bg_num==0:
			if peak_bg_num<=0:
				df1 = df_deviation1
				return df1, df_deviation1, df_score1, df_expected_score1

			# if peak_bg_num<0:
			# 	peak_bg_num = self.peak_bg.shape[1]
			# peak_bg_num = 10
			print('peak_bg_num ',peak_bg_num)
			peak_bg_idvec = np.arange(peak_bg_num)
			peak_bg_list = []
			for i in peak_bg_idvec:
				t_id1 = self.peak_bg.loc[peak_loc,i]
				# peak_bg_id1 = list(self.atac_meta_peak_loc[t_id1])
				peak_bg_id1 = list(self.atac_meta_peak_loc[t_id1-1])
				peak_bg_list.append(peak_bg_id1)

			atac_read_mtx = peak_read
			peak_sel_local = np.asarray(peak_sel_local)
			if parallel_mode==1:
				interval = parallel_interval
				if interval<=0:
					interval = peak_bg_num
				num1 = int(np.ceil(peak_bg_num/interval))
				vec1 = []
				for i in range(num1):
					id1, id2 = interval*i, np.min([interval*(i+1),peak_bg_num])
					print(id1,id2)
					vec1_bg = Parallel(n_jobs=-1)(delayed(self.test_chromvar_estimate_pre2)(atac_read=atac_read_mtx, peak_loc=peak_bg_list[peak_bg_id], 
														peak_sel_local = peak_sel_local,
														atac_read_expected=atac_read_expected,
														motif_query=motif_query, motif_data=motif_data,
														peak_subset_id=peak_bg_id,
														expected_score = expected_score1,	
														type_id_1=type_id_est1+1) for peak_bg_id in tqdm(peak_bg_idvec[id1:id2]))
					for t_vec1 in vec1_bg:
						vec1.append(t_vec1)
			else:
				vec1 = []
				for peak_bg_id in tqdm(peak_bg_idvec):
					if peak_bg_id%100==0:
						print('peak_bg_id ',peak_bg_id)
					vec1_bg = self.test_chromvar_estimate_pre2(atac_read=atac_read_mtx, peak_loc=peak_bg_list[peak_bg_id], 
														peak_sel_local = peak_sel_local,
														atac_read_expected=atac_read_expected,
														motif_query=motif_query, motif_data=motif_data,
														peak_subset_id=peak_bg_id,
														expected_score = expected_score1,
														type_id_1=type_id_est1+1)
					vec1.append(vec1_bg)

			sample_id = peak_read.index
			df1 = self.test_deviation_correction_1(deviation1,vec1,sample_id,motif_query)

			if (save_mode>0) and (output_filename!=''):
				df1.to_csv(output_filename,sep='\t',float_format='%.6f')

		return df1, df_deviation1, df_score1, df_expected_score1
		# return True

	# compute the chromvar score
	def test_chromvar_estimate_pre2(self, atac_read, peak_loc, peak_sel_local = [], atac_read_expected=[], motif_query=[], motif_data=[], peak_subset_id=0, expected_score = [], type_id_1=1):
		
		# peak_loc = list(peak_loc)
		# motif_local = motif_data.loc[peak_loc,motif_query] # shape: (peak_num, motif_num)
		if len(peak_sel_local)==0:
			motif_local = motif_data.loc[peak_loc,motif_query]	# shape: (peak_num, motif_num)
			mtx = (motif_local>0)
			peak_sel_local = mtx

		print('peak_loc, peak_sel_local ',len(peak_loc),peak_sel_local.shape)
		peak_loc1 = list(peak_loc)
		score1 = np.asarray(atac_read.loc[:,peak_loc1]).dot(peak_sel_local.loc[peak_loc1,:]) # (sample_num,motif_num)
		expected_score1 = np.asarray(atac_read_expected.loc[:,peak_loc1]).dot(peak_sel_local.loc[peak_loc1,:])   # (sample_num,motif_num)

		# score1_1, expected_score1_1 = score1, expected_score1
		print('test ',score1.shape, expected_score1.shape, peak_subset_id)
		if peak_subset_id%20==0:
			# print('test', score1[0:2,0], expected_score1[0:2,0], peak_subset_id)
			print('test', score1[0:2], expected_score1[0:2], peak_subset_id)

		if (type_id_1==0) or (len(expected_score)==0):
			# print(type_id_1,expected_score1.shape)
			expected_score2 = expected_score1
		else:
			# print(type_id_1,expected_score.shape)
			expected_score2 = expected_score   # for background peaks
			# if (type_id_1>10) and (type_id_1<20):
			#   expected_score1 = 0.5*expected_score1+0.5*expected_score1_1

		# print('estimate deviation',score1.shape,peak_id)
		eps = 1e-12
		deviation1 = score1 - expected_score1
		deviation = deviation1/(expected_score2+eps)    # (sample_num,motif_num)

		return (deviation, score1, expected_score1, peak_subset_id)
	
	# compute corrected deviation scores
	def test_deviation_correction_1(self,deviation1,vec1,sample_id,feature_name):

		feature_score_local = dict()
		list_bg_deviation, list_bg_raw_score = [], []
		peak_bg_num = len(vec1)
		# for peak_bg_id in tqdm(dict1.index):
		for i in range(peak_bg_num):
			# t_vec1 = dict1[peak_bg_id]
			t_vec1 = vec1[i]
			if type(t_vec1) is int:
				continue
			
			list_bg_deviation.append(t_vec1[0])
			list_bg_raw_score.append(t_vec1[1])

		list_bg_deviation = np.asarray(list_bg_deviation)
		list_bg_raw_score = np.asarray(list_bg_raw_score)   # shape: (bg_num,sample_num,motif_num)

		print(list_bg_deviation.shape, list_bg_raw_score.shape)

		eps = 1e-12
		deviation_corrected = (deviation1-np.mean(list_bg_deviation,axis=0))/(np.std(list_bg_deviation,axis=0)+eps)
		print('deviation_corrected ')
		print(deviation_corrected)

		if deviation_corrected.ndim>1:
			# df1 = pd.DataFrame(data=deviation_corrected.T,index=feature_name,columns=sample_id)
			df1 = pd.DataFrame(data=deviation_corrected,index=sample_id,columns=feature_name,dtype=np.float32)
		else:
			df1 = pd.DataFrame(data=deviation_corrected[:,np.newaxis],index=sample_id,columns=[feature_name],dtype=np.float32)

		return df1

	# compute TF binding activity score
	def test_query_tf_score_1(self,gene_query_vec=[],peak_query_vec=[],motif_query_vec=[],df_gene_peak_query=[],motif_data=[],motif_data_score=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],scale_type_id=1,type_query_id=0,type_id_bg=0,compare_mode=1,filename_prefix='',filename_annot='',save_mode=1,output_file_path='',select_config={}):

		flag_query1=1
		if flag_query1>0:
			print('test_chromvar_estimate_pre1 ')
			start = time.time()
			output_filename = ''
			peak_loc_query = df_gene_peak_query['peak_id'].unique()
			peak_loc_num = len(peak_loc_query)

			motif_query_vec = df_gene_peak_query['motif_id'].unique()
			motif_query_num = len(motif_query_vec)
			peak_bg_num = select_config['peak_bg_num']
			if peak_bg_num>0:
				print('load background peak loci ')
				# input_filename_peak, input_filename_bg, peak_bg_num = select_config['input_filename_peak'],select_config['input_filename_bg'],select_config['peak_bg_num']
				input_filename_peak, input_filename_bg = select_config['input_filename_peak'],select_config['input_filename_bg']
				peak_bg = self.test_gene_peak_query_bg_load(input_filename_peak=input_filename_peak,input_filename_bg=input_filename_bg,peak_bg_num=peak_bg_num)
				self.peak_bg = peak_bg

			if len(motif_data)==0:
				motif_data = self.motif_data_expr
				motif_score = self.motif_data_score
			
			# peak_sel_local = motif_data.loc[peak_loc_query,motif_query_vec]
			peak_sel_local = []
			peak_loc_ori = motif_data.index
			default_mode=2
			if 'default_mode' in select_config:
				default_mode = select_config['default_mode']
			
			peak_sel_local = self.test_query_peak_tf_2(df_gene_peak_query=df_gene_peak_query,
														peak_sel_local=peak_sel_local,
														motif_query_vec=motif_query_vec,
														peak_query_vec=peak_loc_query,
														motif_data=motif_data,
														default_mode=default_mode,
														select_config=select_config)
			print('peak_loc_query, motif_query_vec, peak_sel_local ',peak_loc_num,motif_query_num,peak_sel_local.shape)
			print('peak_bg_num ',peak_bg_num)

			mtx1 = peak_read
			atac_read = peak_read
			# mtx1 = atac_read.X.todense()
			t_value1 = np.mean(mtx1,axis=0)
			read_cnt1 = np.sum(mtx1,axis=1)
			print('mtx1, read_cnt1 ',mtx1.shape,np.max(read_cnt1),np.min(read_cnt1),np.mean(read_cnt1),np.median(read_cnt1))
			ratio1 = t_value1/np.sum(t_value1)
			print('atac read ratio',np.max(ratio1),np.min(ratio1),np.mean(ratio1),np.median(ratio1))
			expected_cnt1 = np.outer(read_cnt1,ratio1)
			# expected_cnt1 = np.random.rand(len(read_cnt1),len(ratio1))
			print('expected_cnt1', expected_cnt1.shape)
			sample_id = peak_read.index
			peak_loc_query_ori = peak_read.columns
			# expected_mtx1 = pd.DataFrame(data=expected_cnt1,index=atac_read.obs_names,columns=atac_read.var_names)
			expected_mtx1 = pd.DataFrame(index=sample_id,columns=peak_loc_query_ori,data=expected_cnt1)
			
			atac_read_expected = expected_mtx1
			peak_loc_query_1 = peak_loc_query
			df1,df_score_deviation_ori,df_score_ori,df_score_expected = self.test_chromvar_estimate_pre1(peak_loc=peak_loc_query_1,peak_sel_local=peak_sel_local,
																										motif_query=motif_query_vec,motif_data=[],
																										rna_exprs=rna_exprs,peak_read=peak_read,
																										atac_read_expected=atac_read_expected,
																										est_mode=1,type_id_est=0,
																										peak_bg_num=peak_bg_num,
																										parallel_mode=0,parallel_interval=-1,
																										save_mode=1,output_filename=output_filename,
																										select_config=select_config)

			df_score_deviation = df1
			print('df_score_deviation ',df_score_deviation.shape)
			stop = time.time()
			print('test_chromvar_estimate_pre1 ',stop-start)

			df_query_1 = []
			if compare_mode>0:
				df_motif_expr = rna_exprs.loc[sample_id,motif_query_vec]
				list1 = []
				for i1 in range(motif_query_num):
					motif_id1 = motif_query_vec[i1]
					y = np.asarray(df_motif_expr[motif_id1])
					y_score = np.asarray(df_score_deviation[motif_id1])
					score_1 = self.score_2a_1(y,y_score)
					list1.append(score_1)

				df_query_1 = pd.concat(list1,axis=1,join='outer',ignore_index=False).T
				df_query_1.index = motif_query_vec
				df_query_1 = df_query_1.sort_values(by=['spearmanr'],ascending=False)

			atac_read_expected_sub1 = atac_read_expected.loc[:,peak_loc_query]
			# list1 = [df_motif_annot1,df_score_ori,atac_read_expected_sub1,df_score_expected,df_score_deviation,df_query_1]
			list1 = [df_score_ori,df_score_expected,df_score_deviation_ori,df_score_deviation,df_query_1]
			# filename_annot_vec_2 = ['motif_query_annot1','score_ori','peak_mtx_expected_sub1','score_expeced','score_deviation','score_compare']
			
			filename_annot_vec_2 = ['score_ori','score_expeced','score_deviation_ori','score_deviation','score_compare']
			output_filename_list = []
			query_num1 = len(list1)
			for i1 in range(query_num1):
				filename_annot2 = filename_annot_vec_2[i1]
				
				df_query1 = list1[i1]
				if len(df_query1)>0:
					if filename_annot2 in ['score_deviation','score_compare']:
						output_filename = '%s/%s.%s.peak_bg%d.1.txt'%(output_file_path,filename_prefix,filename_annot2,peak_bg_num)
						df_query1.to_csv(output_filename,sep='\t',float_format='%.6f')
					else:
						output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix,filename_annot2)
						df_query1.to_csv(output_filename,sep='\t')
			
		return df_score_deviation, df_query_1
		# return True
	
	## query peak-tf link
	def test_query_peak_tf_2(self,df_gene_peak_query=[],motif_query_vec=[],peak_query_vec=[],peak_sel_local=[],motif_data=[],default_mode=0,select_config={}):

		## annotate peak-tf link query with negative correlation
		flag1 = 1
		if flag1>0:
			if len(motif_query_vec)==0:
				motif_query_vec = df_gene_peak_query['motif_id'].unique()

			if len(motif_data)==0:
				motif_data = self.motif_data
			
			# type_id1=1
			if default_mode==1:
				## use the original motif_data
				if len(peak_query_vec)==0:
					peak_query_vec = motif_data.index
				peak_sel_local = motif_data.loc[peak_query_vec,motif_query_vec]
				
				return peak_sel_local

			else:
				## use estimated peak-tf link query
				# for peaks not included in peak-tf-gene link query, use the original motif data
				if len(peak_sel_local)==0:
					# peak_num, motif_num = len(peak_query_vec), len(motif_query_vec)
					# peak_sel_local = np.zeros((peak_num,motif_num),dtype=np.float32)
					peak_sel_local = motif_data.loc[:,motif_query_vec]

				if len(peak_query_vec)==0:
					peak_query_vec = df_gene_peak_query['peak_id'].unique()
				peak_sel_local.loc[peak_query_vec,:] = 0

				peak_num, motif_num = len(peak_query_vec), len(motif_query_vec)
				df_gene_peak_query['pair_peak_tf'] = self.test_query_index(df_gene_peak_query,column_vec=['motif_id','peak_id'])
				
				df_query2 = []
				df2 = df_query2
				if ('peak_tf_link' in df_gene_peak_query) and (default_mode==0):
					id2 = (df_gene_peak_query['peak_tf_link']<0)
					# df2 = df_gene_peak_query.loc[id2,['peak_id','motif_id','pair_peak_tf']]
					df2 = df_gene_peak_query.loc[id2,:]

					id1 = (~id2)
					df1 = df_gene_peak_query.loc[id1,:]
				else:
					df1 = df_gene_peak_query

				df_query1 = df1.drop_duplicates(subset=['pair_peak_tf'],keep='first')
				print('pair_peak_tf, df_query1 ',df_query1.shape)

				if len(df2)>0:
					df_query2 = df2.drop_duplicates(subset=['pair_peak_tf'],keep='first')
					print('pair_peak_tf, df_query2 ',df_query2.shape)

				list1 = [df_query2,df_query1]
				query_num1 = len(list1)
				for i1 in range(2):
					df_query = list1[i1]
					if len(df_query)==0:
						continue
					df_query.index = np.asarray(df_query['motif_id'])
					motif_query_vec_2 = df_query['motif_id'].unique()
					motif_query_num2 = len(motif_query_vec_2)
					print('df_query ',df_query.shape,df_query[0:2])
					for i2 in range(motif_query_num2):
						motif_id1 = motif_query_vec_2[i2]
						df_pre2 = df_query.loc[[motif_id1],:]
						peak_id2 = df_pre2['peak_id'].unique()
						peak_num2 = len(peak_id2)

						t_value_1 = (i1*2-1)
						peak_sel_local.loc[peak_id2,motif_id1] = t_value_1
						if i2%100==0:
							# print('peak_id2, gene_query_vec_2, t_value_1 ',peak_num2,gene_query_num2,motif_id1,i1,i2,t_value_1)
							print('peak_id2, t_value_1 ',peak_num2,motif_id1,i1,i2,t_value_1)

				return peak_sel_local


def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)



