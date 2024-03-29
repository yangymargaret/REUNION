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
from test_annotation_pre1 import _Base2_1
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

class _Base2_2(_Base2_1):
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

		_Base2_1.__init__(self,file_path=file_path,
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
			# root_path_1: '../data2'
			root_path_1 = select_config['root_path_1']
			input_file_path1 = root_path_1
			data_file_type_id1 = 0
			# run_id = select_config['run_id']
			type_id_feature = select_config['type_id_feature']
			filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)

			select_config_query = {'data_path':input_file_path,
									'filename_save_annot_1':filename_save_annot_1,
									'filename_save_annot_pre1':filename_save_annot_1}

			return select_config_query

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

		return select_config

	# ## motif-peak estimate: load meta_exprs and peak_read
	# def test_motif_peak_estimate_control_load_pre1_ori(self,meta_exprs=[],peak_read=[],flag_format=False,select_config={}):

	# 	input_file_path1 = self.save_path_1
	# 	# data_file_type = 'CD34_bonemarrow'
	# 	# input_file_path = '%s/data_pre2/cd34_bonemarrow'%(input_file_path1)
	# 	data_file_type = select_config['data_file_type']
	# 	# input_file_path = select_config['data_path']
	# 	# filename_save_annot_1 = select_config['filename_save_annot_1']
		
	# 	input_filename_1, input_filename_2 = select_config['filename_rna'],select_config['filename_atac']
	# 	input_filename_3 = select_config['filename_rna_exprs_1']
	# 	# rna_meta_ad = sc.read_h5ad(input_filename_1)
	# 	# atac_meta_ad = sc.read_h5ad(input_filename_2)
	# 	rna_meta_ad = sc.read(input_filename_1)
	# 	atac_meta_ad = sc.read(input_filename_2)
	# 	meta_scaled_exprs = pd.read_csv(input_filename_3,index_col=0,sep='\t')
	# 	print(input_filename_1,input_filename_2)
	# 	print('rna_meta_ad\n', rna_meta_ad)
	# 	print('atac_meta_ad\n', atac_meta_ad)

	# 	# atac_meta_ad = self.atac_meta_ad
	# 	# meta_scaled_exprs = self.meta_scaled_exprs
	# 	if flag_format==True:
	# 		meta_scaled_exprs.columns = meta_scaled_exprs.columns.str.upper()
	# 		rna_meta_ad.var_names = rna_meta_ad.var_names.str.upper()
	# 		rna_meta_ad.var.index = rna_meta_ad.var.index.str.upper()

	# 	self.rna_meta_ad = rna_meta_ad
	# 	sample_id = rna_meta_ad.obs_names
	# 	sample_id1 = meta_scaled_exprs.index
	# 	assert list(sample_id)==list(atac_meta_ad.obs_names)
	# 	assert list(sample_id)==list(sample_id1)
	# 	atac_meta_ad = atac_meta_ad[sample_id,:]
	# 	self.atac_meta_ad = atac_meta_ad

	# 	meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
	# 	self.meta_scaled_exprs = meta_scaled_exprs
	# 	print('atac_meta_ad, meta_scaled_exprs ',atac_meta_ad.shape,meta_scaled_exprs.shape,input_filename_3)

	# 	peak_read = pd.DataFrame(index=atac_meta_ad.obs_names,columns=atac_meta_ad.var_names,data=atac_meta_ad.X.toarray(),dtype=np.float32)
	# 	meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
	# 	self.meta_exprs_2 = meta_exprs_2

	# 	vec1 = utility_1.test_stat_1(np.mean(atac_meta_ad.X.toarray(),axis=0))
	# 	vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
	# 	vec3 = utility_1.test_stat_1(np.mean(meta_exprs_2,axis=0))

	# 	print('atac_meta_ad mean values ',atac_meta_ad.shape,vec1)
	# 	print('meta_scaled_exprs mean values ',meta_scaled_exprs.shape,vec2)
	# 	print('meta_exprs_2 mean values ',meta_exprs_2.shape,vec3)

	# 	return peak_read, meta_scaled_exprs, meta_exprs_2

	## motif-peak estimate: load meta_exprs and peak_read
	def test_motif_peak_estimate_control_load_pre1_ori(self,meta_exprs=[],peak_read=[],flag_format=False,select_config={}):

		input_file_path1 = self.save_path_1
		# data_file_type = 'CD34_bonemarrow'
		# input_file_path = '%s/data_pre2/cd34_bonemarrow'%(input_file_path1)
		data_file_type = select_config['data_file_type']
		# input_file_path = select_config['data_path']
		# filename_save_annot_1 = select_config['filename_save_annot_1']
		
		input_filename_1, input_filename_2 = select_config['filename_rna_meta'],select_config['filename_atac_meta']
		print('input_filename_1 ',input_filename_1)
		print('input_filename_2 ',input_filename_2)
		rna_meta_ad = sc.read_h5ad(input_filename_1)
		atac_meta_ad = sc.read_h5ad(input_filename_2)

		# rna_meta_ad = sc.read(input_filename_1)
		# atac_meta_ad = sc.read(input_filename_2)
		print(input_filename_1,input_filename_2)
		print('rna_meta_ad\n', rna_meta_ad)
		print('atac_meta_ad\n', atac_meta_ad)

		# atac_meta_ad = self.atac_meta_ad
		# meta_scaled_exprs = self.meta_scaled_exprs
		if flag_format==True:
			rna_meta_ad.var_names = rna_meta_ad.var_names.str.upper()
			rna_meta_ad.var.index = rna_meta_ad.var.index.str.upper()
		
		column_1 = 'filename_rna_exprs_1'
		meta_scaled_exprs = []
		if column_1 in select_config:
			input_filename_3 = select_config['filename_rna_exprs_1']
			meta_scaled_exprs = pd.read_csv(input_filename_3,index_col=0,sep='\t')

			if flag_format==True:
				meta_scaled_exprs.columns = meta_scaled_exprs.columns.str.upper()
		
			meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
			# self.meta_scaled_exprs = meta_scaled_exprs
			# print('atac_meta_ad, meta_scaled_exprs ',atac_meta_ad.shape,meta_scaled_exprs.shape,input_filename_3)

			vec2 = utility_1.test_stat_1(np.mean(meta_scaled_exprs,axis=0))
			print('meta_scaled_exprs mean values ',meta_scaled_exprs.shape,vec2)

		self.rna_meta_ad = rna_meta_ad
		sample_id = rna_meta_ad.obs_names
		assert list(sample_id)==list(atac_meta_ad.obs_names)

		if len(meta_scaled_exprs)>0:
			sample_id1 = meta_scaled_exprs.index
			assert list(sample_id)==list(sample_id1)

		atac_meta_ad = atac_meta_ad[sample_id,:]
		self.atac_meta_ad = atac_meta_ad

		peak_read = pd.DataFrame(index=atac_meta_ad.obs_names,columns=atac_meta_ad.var_names,data=atac_meta_ad.X.toarray(),dtype=np.float32)
		meta_exprs_2 = pd.DataFrame(index=rna_meta_ad.obs_names,columns=rna_meta_ad.var_names,data=rna_meta_ad.X.toarray(),dtype=np.float32)
		self.peak_read = peak_read
		self.meta_exprs_2 = meta_exprs_2
		self.meta_scaled_exprs = meta_scaled_exprs

		vec1 = utility_1.test_stat_1(np.mean(atac_meta_ad.X.toarray(),axis=0))
		vec3 = utility_1.test_stat_1(np.mean(meta_exprs_2,axis=0))

		print('atac_meta_ad mean values ',atac_meta_ad.shape,vec1)
		# print('meta_exprs_2 mean values ',meta_exprs_2.shape,vec3)
		print('rna_meta_ad mean values ',meta_exprs_2.shape,vec3)

		return peak_read, meta_scaled_exprs, meta_exprs_2

	## score query for performance comparison
	# def test_query_motif_filename_pre1
	def test_query_motif_filename_pre1(self,data=[],data_file_type_query='',thresh_motif=5e-05,thresh_motif_annot='',retrieve_mode=0,save_mode=1,verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			data_file_query_motif = data_file_type_query
			thresh_vec_1 = [5e-05,1e-04,0.001,-1,0.01]
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

			# root_path_1 = select_config['root_path_1']
			# data_file_type_annot = select_config['data_file_type_annot']
			# path_id = select_config['path_id']

			data_path_save_motif = select_config['data_path_save_motif']
			filename_prefix = select_config['filename_prefix']
			filename_annot_1 = select_config['filename_annot_1']

			motif_filename1 = '%s/test_motif_data.%s.h5ad'%(data_path_save_motif,filename_annot_1)
			motif_filename2 = '%s/test_motif_data_score.%s.h5ad'%(data_path_save_motif,filename_annot_1)

			file_format = 'csv'
			if format_type==0:
				filename_annot1 = '1.2'
				filename_annot2 = '1'
				input_filename_1 = '%s/%s_motif.1.2.%s'%(data_path_save_motif,filename_prefix,file_format)
				input_filename_2 = '%s/%s_motif_scores.1.%s'%(data_path_save_motif,filename_prefix,file_format)
				filename_chromvar_score = '%s/%s_chromvar_scores.1.csv'%(data_path_save_motif,filename_prefix)
			else:
				filename_annot1 = thresh_motif_annot_2
				input_filename_1 = '%s/%s.motif.%s.%s'%(data_path_save_motif,filename_prefix,filename_annot1,file_format)
				input_filename_2 = '%s/%s.motif_scores.%s.%s'%(data_path_save_motif,filename_prefix,filename_annot1,file_format)
				filename_chromvar_score = '%s/%s.chromvar_scores.%s.csv'%(data_path_save_motif,filename_prefix,filename_annot1)

			file_path_2 = '%s/TFBS'%(data_path_save_motif)
			if (os.path.exists(file_path_2)==False):
				print('the directory does not exist: %s'%(file_path_2))
				os.makedirs(file_path_2,exist_ok=True)

			input_filename_annot = '%s/translationTable.csv'%(file_path_2)
			column_motif = 'motif_id'
			# column_motif = 'tf'

			select_config_1.update({'data_path_save_motif':data_path_save_motif})

			select_config_1.update({'input_filename_motif_annot':input_filename_annot,'filename_translation':input_filename_annot,
									'column_motif':column_motif})
			
			select_config_1.update({'motif_filename_1':input_filename_1,'motif_filename_2':input_filename_2,
									'filename_chromvar_score':filename_chromvar_score,
									'motif_filename1':motif_filename1,'motif_filename2':motif_filename2})

			# data_file_query_motif = data_file_type_query
			# motif_filename1 = '%s/test_motif_data.%s.1.h5ad'%(data_path_save_motif,data_file_query_motif)
			# motif_filename2 = '%s/test_motif_data_score.%s.1.h5ad'%(data_path_save_motif,data_file_query_motif)
			# select_config.update({'motif_filename1':motif_filename1,'motif_filename2':motif_filename2})

			filename_save_annot_1 = data_file_type_query
			select_config.update({'filename_save_annot_pre1':filename_save_annot_1})
			field_query_1 = ['input_filename_motif_annot','filename_translation','motif_filename_1','motif_filename_2',
								'filename_chromvar_score','motif_filename1','motif_filename2']
			for field_id in field_query_1:
				value = select_config_1[field_id]
				print('field, value: ',field_id,value)

			return select_config_1

	## query motif data
	def test_query_motif_data_pre1_1(self,data=[],method_type_vec=[],thresh_motif=5e-05,retrieve_mode=0,default_mode=0,save_mode=1,verbose=0,select_config={}):

		flag_config_1 = 1
		data_file_type_query = select_config['data_file_type']
		method_type_feature_link = select_config['method_type_feature_link']
		if flag_config_1>0:
			# root_path_1 = select_config['root_path_1']
			# root_path_2 = select_config['root_path_2']	
			method_type_vec_pre1 = method_type_vec.copy()
			if len(method_type_vec)==0:
				method_type_vec_pre1 = [method_type_feature_link]
			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec_pre1,flag_config_1=flag_config_1,select_config=select_config)

		file_save_path_1 = select_config['file_path_peak_tf']

		if data_file_type_query in ['pbmc']:
			dict_file_annot1 = select_config['dict_file_annot1']

		thresh_vec_1 = [5e-05,1e-04,0.001,-1,0.01]
		annot_vec_1 = ['5e-05','1e-04','0.001',-1,'0.01']
		# thresh_num1 = len(thresh_vec_1)

		data_file_type_query = select_config['data_file_type']
		# thresh_motif = 5e-05
		select_config_1 = self.test_query_motif_filename_pre1(data_file_type_query=data_file_type_query,thresh_motif=thresh_motif,
																retrieve_mode=retrieve_mode,save_mode=1,verbose=0,select_config=select_config)

		if retrieve_mode==0:
			select_config = select_config_1  # update the original select_config
		else:
			motif_config = select_config_1
			annot_str1 = select_config_1['dict_motif_thresh_annot'][thresh_motif]
			field_id = 'motif_thresh%s'%(annot_str1)
			print('field of motif data filename: ',field_id)
			# print(motif_config)
			select_config.update({field_id:motif_config})

		flag_motif_data_load = 1
		# flag_motif_data_load = 0
		# method_type_feature_link = select_config['method_type_feature_link']
		motif_data = []
		motif_data_score = []
		# method_type_vec = []
		if flag_motif_data_load>0:
			print('load motif data')
			# method_type_vec_query = ['insilico_0.1']+[method_type_feature_link]
			# method_type_1, method_type_2 = method_type_vec_query[0:2]
			method_type_vec_query = method_type_vec
			if len(method_type_vec)==0:
				method_type_vec_query = [method_type_feature_link]

			# method_type_2 = method_type_vec_query[0]
			# motif_filename_list1 = [motif_filename1,motif_filename2]
			# select_config.update({'motif_filename_list1':motif_filename_list1})
			# motif_filename_list1 = select_config['motif_filename_list1']
			motif_filename1, motif_filename2 = select_config_1['motif_filename1'], select_config_1['motif_filename2']
			motif_filename_list1 = [motif_filename1,motif_filename2]
			
			select_config.update({'motif_filename_list1':motif_filename_list1})
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,select_config=select_config)

			key_vec_1 = list(dict_motif_data.keys())
			print('dict_motif_data: ',key_vec_1)
			# print(dict_motif_data)

			# method_type_query = method_type_feature_link
			method_type_query = method_type_vec_query[0]
			print('method_type_query: ',method_type_query)
			motif_data_query1 = dict_motif_data[method_type_query]['motif_data']
			motif_data_score_query1 = dict_motif_data[method_type_query]['motif_data_score']
			
			peak_loc_ori = motif_data_query1.index
			print('motif_data_query1: ',motif_data_query1.shape)
			print(motif_data_query1[0:2])
			if len(motif_data_score_query1)>0:
				print('motif_data_score_query1: ',motif_data_score_query1.shape)
				print(motif_data_score_query1[0:2])
			# print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
			
			if default_mode>0:
				# motif_data = motif_data_query1
				# motif_data_score = motif_data_score_query1
				self.dict_motif_data = dict_motif_data

			return dict_motif_data, select_config

	## load motif data
	def test_load_motif_data_1(self,method_type_vec=[],input_file_path='',save_mode=1,verbose=0,select_config={}):
		
		flag_query1=1
		method_type_num = len(method_type_vec)
		dict_motif_data = dict()
		data_file_type = select_config['data_file_type']
		
		for i1 in range(method_type_num):
			# method_type = method_type_vec[method_type_id]
			method_type = method_type_vec[i1]
			
			# data_path = select_config['input_file_path_query'][method_type]
			# input_file_path = data_path
			# print('data_path_save: ',data_path)
			motif_data_pre1, motif_data_score_pre1 = [], []

			method_annot_vec = ['insilico','joint_score','Unify','CIS-BP','CIS_BP'] # the method type which share motif scanning results
			# flag_1 = False
			flag_1 = self.test_query_method_type_motif_1(method_type=method_type,method_annot_vec=method_annot_vec,select_config=select_config)
			
			if flag_1>0:
				if (len(motif_data_pre1)==0) and (len(motif_data_score_pre1)==0):
					input_filename1 = select_config['filename_motif_data']
					input_filename2 = select_config['filename_motif_data_score']
					b1 = input_filename1.find('.h5ad')
					b2 = input_filename1.find('.ad')
					if (b1>=0) or (b2>=0):
						input_filename_list1 = [input_filename1,input_filename2]	# read from the anndata
						input_filename_list2 = []
					else:
						# b3 = input_filename.find('.csv')
						input_filename_list1 = []
						input_filename_list2 = [input_filename1,input_filename2]	# read from the .csv data

					print('motif_filename_list1: ',input_filename_list1)

					save_file_path = ''
					flag_query2 = 1
					# load motif data
					motif_data, motif_data_score, df_annot, type_id_query = self.test_load_motif_data_pre1(input_filename_list1=input_filename_list1,
																											input_filename_list2=input_filename_list2,
																											flag_query1=1,flag_query2=flag_query2,
																											input_file_path=input_file_path,
																											save_file_path=save_file_path,
																											type_id_1=0,type_id_2=1,
																											select_config=select_config)
					
					# dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
					# flag_query1=0
					motif_data_pre1 = motif_data
					motif_data_score_pre1 = motif_data_score
				else:
					motif_data, motif_data_score = motif_data_pre1, motif_data_score_pre1
					print('motif_data loaded ',motif_data.shape,motif_data_score.shape,method_type,i1)
					# dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}

				print('perform motif name conversion ')
				motif_data = self.test_query_motif_name_conversion_1(motif_data)
				
				if len(motif_data_score)>0:
					motif_data_score = self.test_query_motif_name_conversion_1(motif_data_score)
				
				dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
			
			dict_motif_data[method_type] = dict_query
			# print('dict_query: ',dict_query,method_type)

		return dict_motif_data, select_config

	## prepare translationTable
	def test_translationTable_pre1(self,motif_data=[],
										motif_data_score=[],
										df_gene_annot=[],
										meta_scaled_exprs=[],
										save_mode=1,
										save_file_path='',
										output_filename='',
										flag_cisbp_motif=1,
										flag_expr=0,
										select_config={}):

		# motif_name_1 = motif_data.columns
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

			# motif_data.columns = motif_name_1
			# if len(motif_data_score)>0:
			# 	motif_data_score.columns = motif_name_1

			df1 = pd.DataFrame.from_dict(data={'motif_id':motif_name_ori,'tf':motif_name_1},orient='columns')

			# meta_scaled_exprs = self.meta_scaled_exprs
			# motif_query_ori = motif_data.columns.str.upper()
			# gene_name_query_ori = meta_scaled_exprs.columns.str.upper()
			# # print('motif_query_ori ',motif_query_ori)
			# # print('gene_name_query_ori ',gene_name_query_ori)
			# motif_query_name_expr = motif_query_ori.intersection(gene_name_query_ori,sort=False)
			# # self.motif_data_expr = self.motif_data.loc[:,motif_query_name_expr]
			# # print('motif_data, motif_data_score, motif_data_expr ', motif_data.shape, motif_data_score.shape, self.motif_data_expr.shape)
			# print('motif_data, motif_data_score, motif_query_name_expr ',motif_data.shape,motif_data_score.shape,len(motif_query_name_expr))

			# df1['gene_id'] = df1['motif_id'].str.split('_').str.get(0) # ENSEMBL id
			df1['gene_id'] = np.asarray(gene_id)
			df1.index = np.asarray(df1['gene_id'].str.upper())
			# df1 = df1.rename(columns={'gene_id':'ENSEMBL'})
			
			# if len(df_gene_annot)==0:
			# 	df_gene_annot = self.df_gene_annot_ori
			# if len(df_gene_annot_expr)==0:
			# 	df_gene_annot_expr = self.df_gene_annot_expr

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
				# meta_scaled_exprs = self.meta_scaled_exprs
				# gene_name_expr = meta_scaled_exprs.columns
				df_var = self.rna_meta_ad.var
				if flag_expr>1:
					# motif name query by gene id
					# df_var = self.rna_meta_ad.var
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
					# df_var = self.rna_meta_ad.var
					gene_name_expr = self.rna_meta_ad.var_names
					output_file_path = select_config['data_path_save']
					output_filename_2 = '%s/test_rna_meta_ad.df_var.query1.txt'%(output_file_path)
					df_var.to_csv(output_filename_2,sep='\t')
					motif_query_name_expr = pd.Index(tf_name).intersection(gene_name_expr,sort=False)
					df1.index = np.asarray(df1['tf'])
					df1.loc[motif_query_name_expr,'tf_expr'] = 1
					
				df1.index = np.asarray(df1['gene_id'])
				self.motif_query_name_expr = motif_query_name_expr

				# print('motif_query_id_expr ',len(motif_query_id_expr))
				# df1.loc[motif_query_id_expr,'tf_expr'] = 1
				print('motif_query_name_expr ',len(motif_query_name_expr))

			# df.loc[:,'tf_ori'] = df.loc[:,'tf'].copy()
			# df1.loc[motif_query_id_expr,'tf'] = df_gene_annot_expr.loc[motif_query_id_expr,'gene_name']
			# df1.index = np.asarray(df1['tf'])
			if save_mode>0:
				if output_filename=='':
					output_filename = '%s/translationTable.csv'%(save_file_path)
				df1.to_csv(output_filename,sep='\t')

		return df1

	## load motif data
	def test_load_motif_data_pre1(self,input_filename_list1=[],
										input_filename_list2=[],
										flag_query1=1,
										flag_query2=1,
										overwrite=True,
										input_file_path='',
										save_mode=1,
										save_file_path='',
										type_id_1=0,
										type_id_2=1,
										select_config={}):
		
		# if input_file_path=='':
		# 	input_file_path = select_config['data_path']

		flag_pre1=0
		motif_data, motif_data_score = [], []
		type_id_query = type_id_1
		df_annot = []

		if len(input_filename_list1)>0:
			## load from the processed anndata
			input_filename1, input_filename2 = input_filename_list1
			if (os.path.exists(input_filename1)==True) and (os.path.exists(input_filename2)==True):
				motif_data_ad = sc.read(input_filename1)
				try:
					motif_data = pd.DataFrame(index=motif_data_ad.obs_names,columns=motif_data_ad.var_names,data=np.asarray(motif_data_ad.X.toarray()))
				except Exception as error:
					print('error! ',error)
					motif_data = pd.DataFrame(index=motif_data_ad.obs_names,columns=motif_data_ad.var_names,data=np.asarray(motif_data_ad.X))

				motif_data_score_ad = sc.read(input_filename2)
				try:
					motif_data_score = pd.DataFrame(index=motif_data_score_ad.obs_names,columns=motif_data_score_ad.var_names,data=np.asarray(motif_data_score_ad.X.toarray()),dtype=np.float32)
				except Exception as error:
					motif_data_score = pd.DataFrame(index=motif_data_score_ad.obs_names,columns=motif_data_score_ad.var_names,data=np.asarray(motif_data_score_ad.X))
			
				# print('motif_data ', motif_data)
				# print('motif_data_score ', motif_data_score)
				print('motif_data ', motif_data.shape)
				print('motif_data_score ', motif_data_score.shape)
				print(motif_data[0:2])
				print(motif_data_score[0:2])
				# motif_data_query, motif_data_score_query = motif_data, motif_data_score
				flag_pre1 = 1

		# load from the original motif data
		if flag_pre1==0:
			print('load the motif data')
			input_filename1, input_filename2 = input_filename_list2[0:2]

			print('input_filename1 ',input_filename1)
			print('input_filename2 ',input_filename2)

			# motif_data, motif_data_score = [], []
			if os.path.exists(input_filename1)==False:
				print('the file does not exist: %s'%(input_filename1))
			else:
				motif_data = pd.read_csv(input_filename1,index_col=0)
				print('motif_data ',motif_data.shape)
				print(motif_data[0:2])
			
			if os.path.exists(input_filename2)==False:
				print('the file does not exist: %s'%(input_filename2))
			else:
				motif_data_score = pd.read_csv(input_filename2,index_col=0)
				print('motif_data_score ',motif_data_score.shape)
				print(motif_data_score[0:2])

			if len(motif_data)==0:
				if len(motif_data_score)>0:
					# motif_data = (motif_data_score>0)
					motif_data = (motif_data_score.abs()>0)
				else:
					print('please provide motif data')
					return
			else:
				if len(motif_data_score)>0:
					motif_data_2 = (motif_data_score.abs()>0)*1.0
					# difference = np.abs(motif_data-motif_data_1)
					difference = np.abs(motif_data-motif_data_2)
					assert np.max(np.max(difference))==0

					## motif name query
					motif_name_ori = motif_data.columns
					motif_name_score_ori = motif_data_score.columns
					peak_loc = motif_data.index
					peak_loc_1 = motif_data_score.index

					assert list(motif_name_ori)==list(motif_name_score_ori)
					assert list(peak_loc)==list(peak_loc_1)
					
					# print('load motif data', input_filename1, input_filename2, motif_data.shape, motif_data_score.shape)
					# print('motif_data ', motif_data)
					# print('motif_data_score ', motif_data_score)
					print('motif_data ', input_filename1, motif_data.shape)
					print('motif_data_score ', input_filename2, motif_data_score.shape)

			# motif name conversion
			input_filename_translation = select_config['filename_translation']
			df_annot = []
			type_id_query = 1
			# overwrite = 0
			# flag_query1 = 1
			if os.path.exists(input_filename_translation)==False:
				print('the file does not exist: %s'%(input_filename_translation))

				# if flag_query1>0:
				output_filename = input_filename_translation
				# meta_scaled_exprs = self.meta_scaled_exprs
				# df_gene_annot = []
				df_gene_annot = self.df_gene_annot_ori
				df_annot = self.test_translationTable_pre1(motif_data=motif_data,
																df_gene_annot=df_gene_annot,
																save_mode=1,
																save_file_path=save_file_path,
																output_filename=output_filename,
																select_config=select_config)
			else:
				print('load TF motif name mapping file')
				df_annot = pd.read_csv(input_filename_translation,index_col=0,sep='\t')

			## motif name correction for the conversion in R
			print('perform TF motif name mapping')
			df_annot.index = np.asarray(df_annot['motif_id'])
			motif_name_ori = motif_data.columns
			motif_name_query = np.asarray(df_annot.loc[motif_name_ori,'tf'])

			# motif_data.columns = motif_name_query # TODO: should update
			column_id = 'tf'
			motif_data, motif_data_ori = self.test_load_motif_data_pre2(motif_data=motif_data,
																			df_annot=df_annot,
																			column_id=column_id,
																			select_config=select_config)

			print('motif_data ',motif_data.shape)
			print(motif_data[0:2])

			print('motif_data_ori ',motif_data_ori.shape)
			print(motif_data_ori[0:2])

			if len(motif_data_score)>0:
				# motif_data_score.columns = motif_name_query # TODO: should update
				motif_data_score, motif_data_score_ori = self.test_load_motif_data_pre2(motif_data=motif_data_score,
																						df_annot=df_annot,
																						column_id=column_id,
																						select_config=select_config)

				print('motif_data_score ',motif_data_score.shape)
				print(motif_data_score[0:2])

				print('motif_data_score_ori ',motif_data_score_ori.shape)
				print(motif_data_score_ori[0:2])

			if save_mode>0:
				# output_filename_list = input_filename_list1
				column_1 = 'filename_list_save_motif'
				# the filename to save the motif data
				if column_1 in select_config:
					output_filename_list = select_config[column_1]
				else:
					data_file_type = select_config['data_file_type']
					if save_file_path=='':
						# save_file_path = select_config['file_save_path_1']
						save_file_path = select_config['file_path_motif']

					output_file_path = save_file_path
					output_filename1 = '%s/test_motif_data.%s.h5ad'%(output_file_path,data_file_type)
					output_filename2 = '%s/test_motif_data_score.%s.h5ad'%(output_file_path,data_file_type)
					output_filename_list = [output_filename1,output_filename2]

				output_filename1, output_filename2 = output_filename_list

				motif_data_ad = utility_1.test_save_anndata(motif_data,sparse_format='csr',obs_names=None,var_names=None,dtype=motif_data.values.dtype)

				# if os.path.exists(output_filename1)==False:
				# 	motif_data_ad.write(output_filename1)
				# else:
				# 	print('the file exists ', output_filename1)
				if os.path.exists(output_filename1)==True:
					print('the file exists ', output_filename1)

				if (os.path.exists(output_filename1)==False) or (overwrite==True):
					motif_data_ad.write(output_filename1)
					print('save motif data ',motif_data_ad)
					print(output_filename1)

				if len(motif_data_score)>0:
					motif_data_score_ad = utility_1.test_save_anndata(motif_data_score,sparse_format='csr',obs_names=None,var_names=None,dtype=motif_data_score.values.dtype)

					if (os.path.exists(output_filename2)==False) or (overwrite==True):
						motif_data_score_ad.write(output_filename2)
						print('save motif_data_score',motif_data_score_ad)
						print(output_filename2)

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
				# df_value_min = np.outer(np.ones(motif_data_score.shape[0]),np.asarray(t_value_min))
				# df2 = pd.DataFrame(index=motif_data_score.index,columns=motif_data_score.columns,data=np.asarray(df_value_min),dtype=np.float32)
				# id1 = df1
				# motif_data_score[id1] = df2[id1]
				# peak_loc_ori = motif_data_score.index
				# peak_id2 = peak_loc_ori[motif_data_score_ori.loc[:,id2].min(axis=1)<0]
				# print(motif_data_score_ori.loc[peak_id2,id2])
				# print(motif_data_score.loc[peak_id2,id2])
				# print(t_value_min)
				# print(t_value_min[id2])

		return motif_data, motif_data_score, df_annot, type_id_query

	## load motif data
	# merge multiple columns that correspond to one TF to one column
	def test_load_motif_data_pre2(self,motif_data,df_annot,column_id='tf',select_config={}):

		# motif_idvec_1= df_annot1.index
		motif_idvec = motif_data.columns.intersection(df_annot.index,sort=False)
		motif_data = motif_data.loc[:,motif_idvec]
		motif_data_ori = motif_data.copy()
		motif_data1 = motif_data.T
		motif_idvec = motif_data1.index  # original motif id
		motif_data1.loc[:,'tf'] = df_annot.loc[motif_idvec,column_id]
		motif_data1 = motif_data1.groupby('tf').max()
		motif_data = motif_data1.T
		# print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape,method_type)
		print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape)
		
		query_idvec = np.asarray(df_annot['motif_id'])
		query_num1 = len(query_idvec)
		t_vec_1 = np.random.randint(query_num1,size=5)
		for iter_id1 in t_vec_1:
			# column_1 = 'ENSG00000196843_LINE52_ARID5A_I'
			# column_2 = 'ARID5A'
			motif_id_query = query_idvec[iter_id1]
			column_1 = motif_id_query
			column_2 = np.asarray(df_annot.loc[df_annot['motif_id']==motif_id_query,'tf'])[0]
			# print('column_1, column_2 ',column_1,column_2,iter_id1)

			difference = (motif_data_ori[column_1].astype(int)-motif_data[column_2].astype(int)).abs().max()
			print('difference ',column_1,column_2,difference,iter_id1)
			assert difference<1E-07

		# print(motif_data[0:5])
		# field_id = '%s.ori'%(key_query)
		# if not (field_id in dict_query):
		# 	dict_query.update({'%s.ori'%(key_query):motif_data_ori})
		return motif_data, motif_data_ori

	## query motif data and motif score data
	def test_motif_data_query(self,input_file_path='',save_file_path='',type_id_1=0,flag_chromvar_score=0,select_config={}):

		## load motif data and motif score data
		flag_motif_data_load = 1
		motif_data, motif_data_score = [], []
		self.motif_data = motif_data
		self.motif_data_score = motif_data_score
		data_file_type = select_config['data_file_type']
		filename_save_annot_1 = self.select_config['filename_save_annot_pre1']
		if flag_motif_data_load>0:
			print('load motif data and motif score data')
			input_filename1 = select_config['motif_filename1']
			input_filename2 = select_config['motif_filename2']
			input_filename_list1 = [input_filename1,input_filename2]
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
																									select_config=select_config)

			if type_id_query>0:
				column_id = 'tf'
				motif_data, motif_data_ori = self.test_load_motif_data_pre2(motif_data=motif_data,
																			df_annot=df_annot,
																			column_id=column_id,
																			select_config=select_config)

				motif_data_score, motif_data_score_ori = self.test_load_motif_data_pre2(motif_data=motif_data_score,
																						df_annot=df_annot,
																						column_id=column_id,
																						select_config=select_config)
				b = input_filename1.find('.h5ad')
				output_filename1 = input_filename1[0:b]+'.1.h5ad'
				motif_data_ad = sc.AnnData(motif_data,dtype=motif_data.values.dtype)
				motif_data_ad.X = csr_matrix(motif_data_ad.X)
				motif_data_ad.write(output_filename1)

				b = input_filename2.find('.h5ad')
				output_filename2 = input_filename2[0:b]+'.1.h5ad'
				motif_data_score_ad = sc.AnnData(motif_data_score,dtype=motif_data_score.values.dtype)
				motif_data_score_ad.X = csr_matrix(motif_data_score_ad.X)
				motif_data_score_ad.write(output_filename2)

			self.motif_data = motif_data
			self.motif_data_score = motif_data_score
			# motif_query_ori = motif_data.columns.str.upper()
			motif_query_ori = motif_data.columns
			gene_name_query_ori = self.rna_meta_ad.var_names
			# gene_name_query_ori = meta_scaled_exprs.columns.str.upper()
			# print('motif_query_ori ',motif_query_ori)
			# print('gene_name_query_ori ',gene_name_query_ori)
			motif_query_name_expr = motif_query_ori.intersection(gene_name_query_ori,sort=False)
			self.motif_query_name_expr = motif_query_name_expr
			self.df_motif_translation = df_annot
			# self.motif_data_expr = self.motif_data.loc[:,motif_query_name_expr]
			# print('motif_data, motif_data_score, motif_data_expr ', motif_data.shape, motif_data_score.shape, self.motif_data_expr.shape)
			print('motif_data, motif_data_score, motif_query_name_expr ', motif_data.shape, motif_data_score.shape, len(motif_query_name_expr))

			# if data_file_type_id1==0:
			# if self.species_id in ['hg38']:
			if flag_chromvar_score>0:
				# input_filename = '%s/test_motif_query_name.1.txt'%(input_file_path)
				# input_filename = '%s/test_motif_query_name.1.copy1.txt'%(input_file_path)
				# input_filename = '%s/test_motif_query_name.1.copy2.txt'%(input_file_path)
				# input_filename = '%s/translationTable.csv'%(input_file_path)
				if flag_motif_data_load>0:
					df1 = df_annot
				else:
					if 'filename_translation' in select_config:
						input_filename = select_config['filename_translation']
					else:
						print('please provide motif name translation file')
						return
					df1 = pd.read_csv(input_filename,index_col=0,sep='\t')

				# df1.index = np.asarray(df1['motif_name'].str.upper())
				# df1['tf_expr'] = 0
				# df1.loc[motif_query_name_expr,'tf_expr'] = 1
				# id_pre1 = (df1['tf_expr']>0)&(df1['tf_expr_1']>0)
				# df1_subset = df1.loc[id_pre1,:]
				# output_file_path = input_file_path
				# output_filename = '%s/test_motif_query_name_expr.1.copy2.txt'%(output_file_path)
				# df1_subset.to_csv(output_filename,sep='\t')

				# df1['gene_id'] = df1['motif_name_ori'].str.split('_').str.get(0)
				# df1.index = np.asarray(df1['gene_id'].str.upper())
				# gene_id_1 = df_gene_annot_expr['gene_id'].str.upper()
				# motif_query_id_expr = df1.index.intersection(gene_id_1,sort=False)
				# print('motif_query_id_expr ',len(motif_query_id_expr))

				# df_gene_annot_expr.index = np.asarray(gene_id_1)
				# df1['tf_expr_1'] = 0
				# df1.loc[motif_query_id_expr,'tf_expr_1'] = 1
				# df1.loc[motif_query_id_expr,'motif_name_1'] = df_gene_annot_expr.loc[motif_query_id_expr,'gene_name']
				# # output_filename = '%s/test_motif_query_name.1.copy1.txt'%(save_file_path1)
				# output_filename = '%s/test_motif_query_name.1.copy2.txt'%(save_file_path1)
				# # df1.index = np.asarray(df1['motif_name_ori'])
				# # df1.index = np.asarray(df1['motif_name'])
				# # df1.to_csv(output_filename,sep='\t')
				# df2 = df1.loc[(df1['tf_expr_1']-df1['tf_expr'])!=0]
				# print('df2 ',df2)
				# df1['motif_name_ori_1'] = df1['motif_name'].copy()
				# id1 = (df1['tf_expr_1']==1)&(df1['tf_expr']==0)
				# df1.loc[id1,'motif_name']=df1.loc[id1,'motif_name_1']
				# df1.index = np.asarray(df1['motif_name'])
				# df1.to_csv(output_filename,sep='\t')

				column_id1 = 'motif_name_ori'
				# df1.index = np.asarray(df1['motif_name_ori'])
				df1.index = np.asarray(df1[column_id1])
				# input_filename = '%s/test_peak_read.CD34_bonemarrow.normalize.1_chromvar_scores.1.csv'%(input_file_path)
				# input_filename = '%s/test_peak_read.%s.normalize.1_chromvar_scores.1.csv'%(input_file_path,data_file_type)
				input_filename = '%s/test_peak_read.%s.normalize.1_chromvar_scores.1.csv'%(input_file_path,filename_save_annot_1)

				## query correlation and mutual information between chromvar score and TF expression
				# output_file_path = input_file_path
				output_file_path = save_file_path
				df_query_1 = df1
				data_file_type = select_config['data_file_type']
				if data_file_type in ['CD34_bonemarrow']:
					type_id_query = 0
				else:
					type_id_query = 1

				filename_save_annot_1 = select_config['filename_save_annot_pre1']
				df_2 = self.test_chromvar_score_query_1(input_filename=input_filename,
														motif_query_name_expr=motif_query_name_expr,
														df_query=df_query_1,
														output_file_path=output_file_path,
														filename_prefix_save=filename_save_annot_1,
														type_id_query=type_id_query,
														select_config=select_config)

				# return

			# return motif_data, motif_data_score, motif_query_name_expr, df_annot
			return motif_data, motif_data_score

	## chromvar score query: chromvar score comparison with TF expression
	# query correlation and mutual information between chromvar score and TF expression
	def test_chromvar_score_query_1(self,input_filename,
											motif_query_name_expr,
											filename_prefix_save='',
											output_file_path='',
											output_filename='',
											df_query=[],
											type_id_query=0,
											select_config={}):

		df1 = df_query
		# df1.index = np.asarray(df1['motif_name_ori'])
		df1.index = np.asarray(df1['motif_id'])
		# input_filename = '%s/test_peak_read.CD34_bonemarrow.normalize.1_chromvar_scores.1.csv'%(input_file_path)
		# input_filename = '%s/test_peak_read.%s.normalize.1_chromvar_scores.1.csv'%(input_file_path,data_file_type)
		chromvar_score = pd.read_csv(input_filename,index_col=0,sep=',')
		print('chromvar_score ', chromvar_score.shape)
		sample_id1 = chromvar_score.columns
		motif_id1 = chromvar_score.index
		# chromvar_score.index = df1.loc[motif_id1,'motif_name']
		chromvar_score.index = df1.loc[motif_id1,'tf']
		if type_id_query==0:
			## for CD34_bonemarrow
			str_vec_1 = sample_id1.str.split('.')
			# str_query1, str_query2, str_query3 = str_vec_1.str.get(0),str_vec_1.str.get(1),str_vec_1.str.get(2)
			str_query_list = [str_vec_1.str.get(i1) for i1 in range(3)]
			str_query1, str_query2, str_query3 = str_query_list
			query_num2 = len(str_query1)
			chromvar_score.columns = ['%s#%s-%s'%(str_query1[i2],str_query2[i2],str_query3[i2]) for i2 in range(query_num2)]
		elif type_id_query==1:
			## for pbmc
			str_vec_1 = sample_id1.str.split('.')
			str_query_list = [str_vec_1.str.get(i1) for i1 in range(2)]
			str_query1, str_query2 = str_query_list
			query_num2 = len(str_query1)
			chromvar_score.columns = ['%s-%s'%(str_query1[i2],str_query2[i2]) for i2 in range(query_num2)]
		else:
			print('chromvar_score: use the loaded columns')
		
		rna_ad = self.rna_meta_ad
		meta_scaled_exprs = self.meta_scaled_exprs
		assert list(chromvar_score.columns)==list(rna_ad.obs_names)
		assert list(chromvar_score.columns)==list(meta_scaled_exprs.index)
		if output_file_path=='':
			output_file_path = input_file_path
		# output_filename = '%s/test_peak_read.CD34_bonemarrow.normalize.1_chromvar_scores.1.copy1.csv'%(output_file_path)
		# output_filename = '%s/test_peak_read.%s.normalize.1_chromvar_scores.1.copy1.csv'%(output_file_path,data_file_type)
		# output_filename = '%s/test_peak_read.%s.normalize.1_chromvar_scores.1.copy1.csv'%(output_file_path,filename_prefix_save)
		if output_filename=='':
			b = input_filename.find('.csv')
			output_filename = input_filename[0:b]+'copy1.csv'
		chromvar_score.to_csv(output_filename)
		print('chromvar_score ',chromvar_score.shape,chromvar_score)

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

		# df_1 = df_1.sort_values(by=['spearmanr','pval1','pearsonr','pval2','mutual_info'],ascending=[False,True,False,True,False])
		df_1 = df_1.sort_values(by=field_query_1,ascending=[False,True,False,True,False])
		# output_filename = '%s/test_peak_read.CD34_bonemarrow.normalize.chromvar_scores.tf_expr.query1.1.txt'%(output_file_path)
		# output_filename = '%s/test_peak_read.%s.normalize.chromvar_scores.tf_expr.query1.1.txt'%(output_file_path,data_file_type)
		# output_filename = '%s/test_peak_read.%s.normalize.chromvar_scores.tf_expr.query1.1.txt'%(output_file_path,filename_prefix_save)
		
		filename = output_filename
		b = filename.find('.csv')
		output_filename = '%s.copy2.txt'%(filename[0:b])
		field_query_2 = ['highly_variable','means','dispersions','dispersions_norm']
		df_gene_annot_expr = self.df_gene_annot_expr
		df_gene_annot_expr.index = np.asarray(df_gene_annot_expr['gene_name'])
		motif_id1 = df_1.index
		df_1.loc[:,field_query_2] = df_gene_annot_expr.loc[motif_id1,field_query_2]
		df_1.to_csv(output_filename,sep='\t',float_format='%.6E')
		mean_value = df_1.mean(axis=0)
		median_value = df_1.median(axis=0)
		print('df_1, mean_value, median_value ',df_1.shape,mean_value,median_value)

		df_2 = df_1.sort_values(by=['highly_variable','dispersions_norm','means','spearmanr','pval1','pearsonr','pval2','mutual_info'],ascending=[False,False,False,False,True,False,True,False])
		# output_filename = '%s/test_peak_read.%s.normalize.chromvar_scores.tf_expr.query1.sort2.1.txt'%(output_file_path,filename_prefix_save)
		df_2.to_csv(output_filename,sep='\t',float_format='%.6E')
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

	## motif_name conversion
	def test_query_motif_name_conversion_1(self,data=[],select_config={}):

		motif_data = data
		dict1 = {'ENSG00000142539':'SPIB',
					'ENSG00000229544':'NKX1-2',
					'TBXT':'T',
					'AC0125311':'HOXC5',
					'AC2261502':'ANHX',
					'AC0021266':'BORCS8-MEF2B',
					'CCDC169-SOHLH2':'C13orf38SOHLH2',
					'LINE4118':'ZNF75C',
					'LINE11277':'DUX1',
					'LINE11282':'DUX3'}

		motif_data = motif_data.rename(columns=dict1)
		return motif_data

	## query the method type based on the motif data used
	def test_query_method_type_motif_1(self,method_type='',method_annot_vec=[],data=[],select_config={}):

		if len(method_annot_vec)==0:
			method_annot_vec = ['insilico','joint_score','Unify','CIS-BP','CIS_BP'] # the method type which share motif scanning results

		flag_1 = False
		for method_annot_1 in method_annot_vec:
			flag_1 = (flag_1|(method_type.find(method_annot_1)>-1))

		return flag_1

	## load motif data
	# def test_load_motif_data_1_pre1(self,method_type_vec=[],select_config={}):

	# 	flag_query1=1
	# 	method_type_num = len(method_type_vec)
	# 	dict_motif_data = dict()
	# 	flag_query1=1
	# 	data_file_type = select_config['data_file_type']
	# 	for i1 in range(method_type_num):
	# 		# method_type = method_type_vec[method_type_id]
	# 		method_type = method_type_vec[i1]
	# 		data_path = select_config['input_file_path_query'][method_type]
	# 		input_file_path = data_path
	# 		print('data_path_save: ',data_path)

	# 		motif_data_pre1, motif_data_score_pre1 = [], []

	# 		method_annot_vec = ['insilico','joint_score','CIS-BP','CIS_BP'] # the method type which share motif scanning results
	# 		# flag_1 = False
	# 		# query_num = len(method_annot_vec)
	# 		# for method_annot_1 in method_annot_vec:
	# 		# 	flag_1 = (flag_1|(method_type.find(method_annot_1)>-1))
	# 		flag_1 = self.test_query_method_type_motif_1(method_type=method_type,method_annot_vec=method_annot_vec,select_config=select_config)
			
	# 		# if (method_type in ['insilico','insilico_1']) or (method_type.find('joint_score')>-1):
	# 		# if (method_type.find('insilico')>-1) or (method_type.find('joint_score')>-1) of (method_type.find(method_annot_1)>-1):
	# 		if flag_1>0:
	# 		# if method_type_id in [0,1]:
	# 			# data_path = select_config['data_path']
	# 			# data_path = select_config['input_file_path_query'][method_type_id]
	# 			# input_file_path = data_path

	# 			motif_data, motif_data_score = self.test_load_motif_data_pre1(input_filename_list1=input_filename_list1,
	# 																					input_filename_list2=input_filename_list2,
	# 																					flag_query1=1,flag_query2=flag_query2,
	# 																					input_file_path=input_file_path,
	# 																					save_file_path=save_file_path,type_id=1,
	# 																					select_config=select_config)
					
	# 			# if (len(motif_data_pre1)==0) and (len(motif_data_score_pre1)==0):
	# 			# 	# input_filename1 = '%s/test_motif_data.1.h5ad'%(input_file_path)
	# 			# 	# input_filename2 = '%s/test_motif_data_score.1.h5ad'%(input_file_path)
	# 			# 	input_file_path_2 = '%s/peak_local/run1_1'%(input_file_path)
	# 			# 	data_file_type_query = select_config['data_file_type']

	# 			# 	if 'motif_filename_list1' in select_config:
	# 			# 		input_filename_list1 = select_config['motif_filename_list1']
	# 			# 		input_filename_list2 = []
	# 			# 	else:
	# 			# 		input_filename1 = '%s/test_motif_data.%s.1.thresh1.h5ad'%(input_file_path_2,data_file_type_query)
	# 			# 		input_filename2 = '%s/test_motif_data_score.%s.1.thresh1.h5ad'%(input_file_path_2,data_file_type_query)

	# 			# 		if os.path.exists(input_filename1)==False:
	# 			# 			print('the file does not exist: %s'%(input_filename1))
	# 			# 			input_filename1 = '%s/test_motif_data.%s.1.h5ad'%(input_file_path_2,data_file_type_query)
	# 			# 			input_filename2 = '%s/test_motif_data_score.%s.1.h5ad'%(input_file_path_2,data_file_type_query)

	# 			# 		input_filename_list1 = [input_filename1,input_filename2]
	# 			# 		input_filename_list2 = []

	# 			# 	print('motif_filename_list1: ',input_filename_list1)
	# 			# 	# input_file_path2 = '%s/peak_local'%(data_path)
	# 			# 	# output_file_path = input_file_path2
	# 			# 	# save_file_path = output_file_path
	# 			# 	save_file_path = ''
	# 			# 	flag_query2 = 1
	# 			# 	# # load motif data
	# 			# 	motif_data, motif_data_score = self.test_load_motif_data_pre1(input_filename_list1=input_filename_list1,
	# 			# 																		input_filename_list2=input_filename_list2,
	# 			# 																		flag_query1=1,flag_query2=flag_query2,
	# 			# 																		input_file_path=input_file_path,
	# 			# 																		save_file_path=save_file_path,type_id=1,
	# 			# 																		select_config=select_config)
					
	# 			# 	# dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
	# 			# 	flag_query1=0
	# 			# 	# motif_data_pre1 = motif_data
	# 			# 	# motif_data_score_pre1 = motif_data_score
	# 			# else:
	# 			# 	motif_data, motif_data_score = motif_data_pre1, motif_data_score_pre1
	# 			# 	print('motif_data loaded ',motif_data.shape,motif_data_score.shape,method_type,i1)
	# 			# 	# dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}

	# 			print('perform motif name conversion ')
	# 			motif_data = self.test_query_motif_name_conversion_1(motif_data)
	# 			if len(motif_data_score)>0:
	# 				motif_data_score = self.test_query_motif_name_conversion_1(motif_data_score)
	# 			dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
			
	# 		else:
	# 			# motif_data = pd.read_csv('google-us-data.csv.gz', compression='gzip', header=0,    sep=' ', quotechar='"', error_bad_lines=False)
	# 			# scipy.io.mmread("sparse_from_file")
	# 			# input_filename1 = select_config['filename_motif_data'][method_type]['motif_data']
	# 			# input_filename2 = select_config['filename_motif_data'][method_type]['motif_data_score']
	# 			# input_filename = 'df_peak_tf_overlap.pearsonr.CD34_bonemarrow.normalize0.tsv.gz'
	# 			dict_file_query = select_config['filename_motif_data'][method_type]
	# 			key_vec = list(dict_file_query.keys()) # key_vec: ['motif_data','motif_data_score']
	# 			dict_query = dict()
	# 			for key_query in key_vec:
	# 				input_filename = dict_file_query[key_query]
	# 				motif_data = []
	# 				flag_matrix=0
	# 				if input_filename!='':
	# 					if (os.path.exists(input_filename)==True):
	# 						print('load motif data: %s'%(input_filename))
	# 						b = input_filename.find('.gz')
	# 						if b>-1:
	# 							motif_data = pd.read_csv(input_filename,compression='gzip',index_col=0,sep='\t')
	# 						else:
	# 							b = input_filename.find('.matrix')
	# 							if b>-1:
	# 								print('load matrix market format data ',method_type)
	# 								motif_data = scipy.io.mmread(input_filename)
	# 								motif_data = motif_data.toarray()
	# 								flag_matrix=1	
	# 							else:
	# 								motif_data = pd.read_csv(input_filename,index_col=0,sep='\t')
	# 						print('motif_data ',motif_data.shape)
	# 						print(motif_data[0:5])	
	# 					else:
	# 						print('the file does not exist ',input_filename)
	# 						continue
	# 				else:
	# 					print('please provide motif data file name ')

	# 				if (method_type in ['GRaNIE','Pando','TRIPOD']) and (len(motif_data)>0):
	# 					# x = 1
	# 					# input_file_path = select_config[]
	# 					# input_file_path = select_config['input_file_path_query'][method_type]
	# 					input_filename_annot = '%s/TFBS/translationTable.csv'%(input_file_path)
	# 					if method_type in ['TRIPOD']:
	# 						pre_config = select_config['config_query'][method_type]
	# 						type_id_query = pre_config['type_id_query']
	# 						input_filename_annot = '%s/TFBS/translationTable%d.csv'%(input_file_path,type_id_query)
	# 						if type_id_query==0:
	# 							input_filename_annot = '%s/TFBS/translationTable%d_pre.csv'%(input_file_path,type_id_query) # temporary: MA0091.1	TAL1	TAL1; MA0091.1	TAL1::TCF3	TAL1::TCF3
	# 					if method_type in ['GRaNIE']:
	# 						df_annot1 = pd.read_csv(input_filename_annot,index_col=0,sep=' ')
	# 						df_annot1.loc[:,'tf'] = np.asarray(df_annot1.index)
	# 						df_annot1.index = np.asarray(df_annot1['HOCOID'])
	# 						print('df_annot1 ',df_annot1.shape,method_type)
	# 						print(df_annot1[0:2])
	# 					else:
	# 						df_annot1 = pd.read_csv(input_filename_annot,index_col=0,header=None,sep='\t')
	# 						if len(df_annot1.columns)==1:
	# 							df_annot1.columns = ['tf_ori']
	# 							tf_id_ori = df_annot1['tf_ori']
	# 							tf_id = pd.Index(tf_id_ori).str.split('(').str.get(0)
	# 							df_annot1.loc[:,'tf'] = tf_id
	# 						else:
	# 							df_annot1.columns = ['tf_ori','tf']
	# 						print('df_annot1 ',df_annot1.shape,method_type)
	# 						print(df_annot1[0:2])
	# 						if method_type in ['Pando']:
	# 							pre_config = select_config['config_query'][method_type]
	# 							run_id = pre_config['run_id']
	# 							metacell_num = pre_config['metacell_num']
	# 							exclude_exons = pre_config['exclude_exons']
	# 							type_id_region = pre_config['type_id_region']
	# 							data_file_type_annot = data_file_type.lower()
	# 							input_file_path2 = '%s/%s/metacell_%d/run%d'%(input_file_path,data_file_type_annot,metacell_num,run_id)
	# 							input_filename = '%s/test_region.%d.%d.bed'%(input_file_path2,exclude_exons,type_id_region)
	# 							flag_region_query=((exclude_exons==True)|(type_id_region>0))
	# 							if os.path.exists(input_filename)==True:
	# 								df_region = pd.read_csv(input_filename,index_col=False,sep='\t')
	# 								df_region.index = np.asarray(df_region['id'])
	# 								# pre_config.update({'df_region':df_region})
	# 								df_region_ori = df_region.copy()
	# 								df_region = df_region.sort_values(by=['overlap'],ascending=False)
	# 								df_region = df_region.loc[~df_region.index.duplicated(keep='first'),:]
	# 								df_region = df_region.sort_values(by=['region_id'],ascending=True)
	# 								output_file_path = input_file_path2
	# 								output_filename = '%s/test_region.%d.%d.2.bed'%(output_file_path,exclude_exons,type_id_region)
	# 								df_region.to_csv(output_filename,sep='\t')
	# 								select_config['config_query'][method_type].update({'df_region':df_region})
	# 							else:
	# 								print('the file does not exist ',input_filename)

	# 							if flag_matrix==1:
	# 								## the motif data is loaded from MM format file and the rownames and colnames to be added
	# 								motif_idvec_ori = df_annot1.index
	# 								# motif_data.columns = motif_idvec_ori
	# 								# motif_data.index = df_region.index
	# 								motif_data = pd.DataFrame(index=df_region.index,columns=motif_idvec_ori,data=np.asarray(motif_data))
	# 								# print('motif_data ',motif_data.shape,method_type)
	# 								# print(motif_data[0:5])
							
	# 							if flag_region_query>0:
	# 								region_id = motif_data.index
	# 								motif_data_ori = motif_data.copy()
	# 								motif_data.loc[:,'peak_id'] = df_region.loc[region_id,'peak_loc']
	# 								motif_data = motif_data.groupby('peak_id').max()
	# 								print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape,method_type)
	# 								dict_query.update({'%s.ori'%(key_query):motif_data_ori})

	# 					# motif_idvec_1= df_annot1.index
	# 					motif_idvec = motif_data.columns.intersection(df_annot1.index,sort=False)
	# 					motif_data = motif_data.loc[:,motif_idvec]
	# 					motif_data_ori = motif_data.copy()
	# 					motif_data1 = motif_data.T
	# 					motif_idvec = motif_data1.index
	# 					motif_data1.loc[:,'tf'] = df_annot1.loc[motif_idvec,'tf']
	# 					motif_data1 = motif_data1.groupby('tf').max()
	# 					motif_data = motif_data1.T
	# 					print('motif_data_ori, motif_data ',motif_data_ori.shape,motif_data.shape,method_type)
	# 					print(motif_data[0:5])
	# 					field_id = '%s.ori'%(key_query)
	# 					if not (field_id in dict_query):
	# 						dict_query.update({'%s.ori'%(key_query):motif_data_ori})

	# 				dict_query.update({key_query:motif_data})
	# 		dict_motif_data[method_type] = dict_query
	# 		# print('dict_query: ',dict_query,method_type)

	# 	return dict_motif_data, select_config

	
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
				# input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.txt.gz'%(input_file_path_query,data_file_type_query)
				input_filename_pre2_1 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.1.copy2_1.txt.gz'%(input_file_path_query,data_file_type_query)
				input_filename_pre2_2 = '%s/test_query_gene_peak.%s.2.pre1.pcorr_query1.annot2.init.query1.2.txt.gz'%(input_file_path_query,data_file_type_query)

				filename_list2 = [input_filename_pre1_1,input_filename_pre1_2,input_filename_pre2_1,input_filename_pre2_2]
				method_type_annot = ['insilico','insilico_1','joint_score_pre1','joint_score_pre2']
				dict_file_query = dict(zip(method_type_annot,filename_list2))

		return dict_file_query

	## query method type
	# query method type for prediction by feature group
	def test_query_column_method_1(self,feature_type_vec=[],method_type_feature_link='',n_neighbors=-1,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			if method_type_feature_link=='':
				method_type_feature_link = select_config['method_type_feature_link']
			if n_neighbors<0:
				n_neighbors = select_config['neighbor_num']
			column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
			column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)

			column_neighbor = ['neighbor%d'%(id1) for id1 in np.arange(1,n_neighbors+1)]
			feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]
			column_1 = '%s_group_neighbor'%(feature_type_query_1)
			column_2 = '%s_group_neighbor'%(feature_type_query_2)

			column_query1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peak query
			column_query2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peak query

			column_pred_3 = '%s.pred_group_neighbor'%(method_type_feature_link)
			column_pred_5 = '%s.pred_group_neighbor_1'%(method_type_feature_link)
			column_pred_6 = '%s.pred_neighbor_2_group'%(method_type_feature_link)
			column_pred_7 = '%s.pred_neighbor_2'%(method_type_feature_link)
			column_pred_8 = '%s.pred_neighbor_1'%(method_type_feature_link)

			column_vec_query_1 = [column_pred2,column_pred_2,column_pred_3,column_pred_5,column_1,column_2,column_pred_6,column_pred_7,column_pred_8,column_query1,column_query2]
			
			return column_vec_query_1

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

		motif_query_vec = motif_data.columns.intersection(rna_exprs.columns,sort=False) # tf with expression
		motif_query_num = len(motif_query_vec)
		print('motif_query_vec (with expression) ',motif_query_num)
		
		column_1 = 'flag_peak_tf_combine'
		# flag_peak_tf_combine=1: combine peak accessibility and TF expression matrix to perform dimension reduction
		flag_peak_tf_combine = 0
		if column_1 in select_config:
			flag_peak_tf_combine = select_config[column_1]

		flag_annot1 = 0
		if flag_peak_tf_combine>0:
			sample_id = rna_exprs_unscaled.index
			peak_read = peak_read.loc[sample_id,:]

			if len(gene_query_vec)==0:
				gene_query_vec = motif_query_vec
				# annot_str_vec = ['peak_tf','peak_motif','peak_motif_ori']
				flag_annot1 = 1  # tf query as gene query
			# else:
			# 	# gene_query_vec = pd.Index(gene_query_vec).union(motif_query_vec,sort=False)
			# 	annot_str_vec = ['peak_gene','peak_motif','peak_motif_ori']
			
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

		dict_query1 = dict()
		dict_query1.update({'df_peak':feature_mtx_query1,
							'df_peak_motif':feature_motif_query1,'df_peak_motif_ori':feature_motif_query2})

		if flag_peak_tf_combine>0:
			feature_expr_query1 = rna_exprs_unscaled.loc[:,gene_query_vec].T # tf expression, shape: (tf_num,cell_num)
			feature_mtx_1 = pd.concat([feature_mtx_query1,feature_expr_query1],axis=0,join='outer',ignore_index=False)
			dict_query1.update({'df_exprs_1':feature_expr_query1})
		else:
			feature_mtx_1 = feature_mtx_query1
		
		feature_mtx_2 = feature_motif_query1
		feature_mtx_2_ori = feature_motif_query2

		list_pre1 = [feature_mtx_1,feature_mtx_2,feature_mtx_2_ori]
		query_num1 = len(list_pre1)
		
		annot_str_vec = ['peak_tf','peak_motif','peak_motif_ori']

		# flag_shuffle = False
		# annot_str_vec_2 = annot_str_vec[0:1]+['motif','motif_ori']
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
			
			print('df_latent: ',df_latent.shape,annot_str1)
			print(df_latent[0:2])

			if i1==0:
				if flag_peak_tf_combine>0:
					feature_query_vec = list(peak_query_vec)+list(gene_query_vec)
					df_latent = df_latent.loc[feature_query_vec,:]

					if flag_annot1>0:
						df_latent_tf = df_latent.loc[motif_query_vec,:]
						print('df_latent_tf: ',df_latent_tf.shape)
						print(df_latent_tf[0:2])
						dict_query1.update({'latent_tf':df_latent_tf})
					else:
						df_latent_gene = df_latent.loc[gene_query_vec,:]
						print('df_latent_gene: ',df_latent_gene.shape)
						print(df_latent_gene[0:2])
						dict_query1.update({'latent_gene':df_latent_gene})
					df_latent_peak = df_latent.loc[peak_query_vec,:]
				else:
					df_latent = df_latent.loc[peak_query_vec,:]
					df_latent_peak = df_latent

				# df_latent_peak = df_latent.loc[peak_query_vec,:]
				print('df_latent_peak: ',df_latent_peak.shape,annot_str1)
				print(df_latent_peak[0:2])
				df_latent_query = df_latent

				# dict_query1.update({'dimension_model_1':dimension_model}) # dimension reduction model for peak accessibility and TF expression
				# dict_query1.update({'latent_peak_tf':df_latent,
				# 					'component_mtx':df_component})
				
				# dict_query1.update({'dimension_model_1':dimension_model}) # dimension reduction model for peak accessibility and TF expression
				# # dict_query1.update({'latent_peak':df_latent_peak,'latent_tf':df_latent_tf,'latent_gene':df_latent_gene})
				# dict_query1.update({'latent_peak':df_latent_peak,'latent_gene':df_latent_gene,
				# 					'component_mtx':df_component})

			else:
				df_latent_peak_motif = df_latent.loc[peak_query_vec,:]
				df_latent_query = df_latent_peak_motif
				
				print('df_latent_peak_motif: ',df_latent_peak_motif.shape)
				print('df_component: ',df_component.shape)
				# print(df_latent_peak_motif[0:2])
				# annot_str2 = annot_str_vec_2[i1]
				# annot_str2 = annot_str_vec[i1]
				# dict_query1.update({'dimension_model_motif':dimension_model}) # dimension reduction model for motif feature of peak query
				# dict_query1.update({'latent_peak_motif':df_latent_peak_motif,'component_peak_motif':df_component})

				# dict_query1.update({'dimension_model_%s'%(annot_str2):dimension_model}) # dimension reduction model for motif feature of peak query
				# dict_query1.update({'latent_%s'%(annot_str1):df_latent_peak_motif,'component_%s'%(annot_str1):df_component})

			# annot_str2 = annot_str_vec_2[i1]
			annot_str2 = annot_str_vec[i1]
			dict_query1.update({'dimension_model_%s'%(annot_str2):dimension_model}) # dimension reduction model for motif feature of peak query
			dict_query1.update({'latent_%s'%(annot_str1):df_latent_query,'component_%s'%(annot_str1):df_component})

			if save_mode>0:
				filename_save_annot_2 = '%s_%s'%(method_type,n_components_query)
				output_filename_1 = '%s/%s.dimension_model.%s.%s.1.h5'%(output_file_path,filename_prefix_save,annot_str1,filename_save_annot_2)
				pickle.dump(dimension_model, open(output_filename_1, 'wb'))

				list_query2 = [df_latent_query,df_component]
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

	## compare TF binding prediction
	# compute feature embedding of peak and TF
	# perform clustering of peak and TF
	# def test_query_compare_binding_pre1_5(self,data=[],motif_id_query='',motif_id='',method_type_vec=[],method_type_vec_query=[],peak_read=[],rna_exprs=[],peak_distance_thresh=100,flag_score_1=0,flag_score_2=0,flag_compare_1=0,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

	# 	data_file_type_query = select_config['data_file_type']
	# 	run_id1 = select_config['run_id']
	# 	# thresh_num1 = 5

	# 	method_type_query1 = 'insilico_0.1'
	# 	method_type_feature_link_1 = 'joint_score_pre1'
	# 	method_type_feature_link = select_config['method_type_feature_link']
	# 	method_type_query2 = method_type_feature_link
	# 	method_type_vec = [method_type_feature_link]

	# 	flag_config_1 = 1
	# 	if flag_config_1>0:
	# 		root_path_1 = select_config['root_path_1']
	# 		root_path_2 = select_config['root_path_2']
	# 		# data_file_type_query = select_config['data_file_type']
	# 		# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre2.thresh3','joint_score_pre1.thresh22','joint_score_pre2.thresh6']
	# 		# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh3','joint_score_pre1.thresh22']
	# 		select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)

	# 	root_path_1 = select_config['root_path_1']
	# 	root_path_2 = select_config['root_path_2']
	# 	if data_file_type_query in ['CD34_bonemarrow']:
	# 		input_file_path = '%s/peak1'%(root_path_2)
	# 	elif data_file_type_query in ['pbmc']:
	# 		input_file_path = '%s/peak2'%(root_path_2)

	# 	peak_distance_thresh = 100
	# 	filename_prefix_1 = 'test_motif_query_binding_compare'

	# 	filename_prefix_save = '%s.pre1'%(data_file_type_query)
	# 	filename_save_annot = '1'
	# 	file_save_path = input_file_path
	# 	output_file_path = '%s/group1'%(file_save_path)

	# 	if os.path.exists(output_file_path)==False:
	# 		print('the directory does not exist: %s'%(output_file_path))
	# 		os.makedirs(output_file_path,exist_ok=True)

	# 	# query peak_read, rna_exprs and motif data information
	# 	load_mode_query = 1
	# 	if load_mode_query>0:
	# 		flag_config_query = 0
	# 		flag_load_1 = 1  # load pead_read and rna_exprs data
	# 		flag_load_2 = 1  # load motif scanning and motif score data
	# 		data_vec_1, dict_motif_data, select_config = self.test_query_feature_load_basic_1(data=[],method_type_vec=method_type_vec,peak_read=[],rna_exprs=[],peak_distance_thresh=peak_distance_thresh,
	# 																							flag_config_1=flag_config_query,flag_load_1=1,flag_load_2=1,load_mode=0,input_file_path='',
	# 																							save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config=select_config)

	# 		peak_read, meta_scaled_exprs, meta_exprs_2 = data_vec_1[0:3]
	# 		print('peak_read, meta_scaled_exprs, meta_exprs_2: ',peak_read.shape, meta_scaled_exprs.shape, meta_exprs_2.shape)
	# 		print(peak_read[0:2])
	# 		print(meta_scaled_exprs[0:2])
	# 		print(meta_exprs_2[0:2])

	# 		rna_exprs = meta_scaled_exprs
	# 		rna_exprs_unscaled = meta_exprs_2

	# 		key_vec_1 = list(dict_motif_data.keys())
	# 		print('dict_motif_data: ',key_vec_1)

	# 		self.data_vec_1 = data_vec_1
	# 		self.dict_motif_data = dict_motif_data

	# 	# dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
	# 	# method_type_query1 = 'insilico_1'
	# 	# method_type_query1 = 'insilico_0.1'
	# 	# method_type_query2 = 'joint_score.thresh2'
	# 	# method_type_query2 = 'joint_score_2.thresh3'
	# 	# method_type_query2 = 'joint_score_pre2.thresh3'
	# 	# method_type_query2 = method_type_feature_link
	# 	method_type_query = method_type_feature_link

	# 	dict_1 = dict_motif_data[method_type_query]
	# 	motif_data_query1 = dict_1['motif_data']
	# 	motif_data_query1 = motif_data_query1.astype(float)
	# 	motif_data_score_query1 = dict_1['motif_data_score']
		
	# 	motif_data_query1 = motif_data_query1.fillna(0)
	# 	motif_data_score_query1 = motif_data_score_query1.fillna(0)
	# 	print('motif_data_query1, motif_data_score_query1: ',motif_data_query1.shape,motif_data_score_query1.shape)
	# 	print(motif_data_query1[0:2])
	# 	print(motif_data_score_query1[0:2])

	# 	column_idvec = ['motif_id','peak_id','gene_id']
	# 	column_id3, column_id2, column_id1 = column_idvec[0:3]

	# 	sample_id = peak_read.index
	# 	motif_data = motif_data_query1
	# 	motif_data_score = motif_data_score_query1
	# 	peak_loc_ori = motif_data.index
	# 	print('peak_loc_ori: ',len(peak_loc_ori))

	# 	motif_query_ori = motif_data.columns
	# 	peak_query_vec_1 = peak_loc_ori
	# 	motif_group = []

	# 	gene_query_vec_ori = rna_exprs.columns
	# 	motif_query_vec = pd.Index(motif_query_ori).intersection(gene_query_vec_ori,sort=False)
	# 	motif_query_num = len(motif_query_vec) # TF with motif and with expression 
	# 	print('motif_query_vec: ',motif_query_num)

	## query motif data by motif scanning and query motif
	# query motif data and motif data score of given peak loci
	# query TFs with expressions
	def test_query_motif_data_annotation_1(self,data=[],data_file_type='',gene_query_vec=[],feature_query_vec=[],method_type='',peak_read=[],rna_exprs=[],save_mode=0,verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			dict_motif_data_ori = data
			method_type_query = method_type
			print(method_type_query)

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
					peak_loc_1 = peak_read.columns

			if flag_1>0:
				motif_data_query1 = motif_data_query1.loc[peak_loc_1,:]
			print('motif_data')
			print(motif_data_query1[0:2])
			
			if 'motif_data_score' in dict_motif_data:
				motif_data_score_query1 = dict_motif_data['motif_data_score']
				if flag_1>0:
					motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_1,:]
				print('motif_data_score')
				print(motif_data_score_query1[0:2])
			else:
				motif_data_score_query1 = motif_data_query1

			# query TFs with expressions
			motif_query_vec = self.test_query_motif_annotation_1(data=motif_data_query1,data_file_type=data_file_type,
																	gene_query_vec=gene_query_vec,feature_query_vec=[],
																	method_type='',peak_read=peak_read,rna_exprs=rna_exprs,
																	save_mode=0,verbose=0,select_config=select_config)

			return motif_data_query1, motif_data_score_query1, motif_query_vec

	## query motif data by motif scanning and query motif
	# query TFs with expressions
	def test_query_motif_annotation_1(self,data=[],data_file_type='',gene_query_vec=[],feature_query_vec=[],method_type='',peak_read=[],rna_exprs=[],save_mode=0,verbose=0,select_config={}):

		motif_data_query1 = data
		motif_name_ori = motif_data_query1.columns
		if len(gene_query_vec)==0:
			if len(rna_exprs)>0:
				gene_name_expr_ori = rna_exprs.columns
				gene_query_vec = gene_name_expr_ori
				# motif_query_name_expr = pd.Index(motif_name_ori).intersection(gene_name_expr_ori,sort=False)

		if len(gene_query_vec)>0:
			motif_query_vec = pd.Index(motif_name_ori).intersection(gene_query_vec,sort=False)
			print('motif_query_vec (with expression): ',len(motif_query_vec))
		else:
			motif_query_vec = motif_name_ori

		return motif_query_vec

	# compute feature embedding
	def test_query_feature_mtx_1(self,feature_query_vec=[],feature_type_vec=[],gene_query_vec=[],method_type_vec_dimension=[],n_components=50,type_id_group=0,
										motif_data=[],motif_data_score=[],motif_group=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],
										load_mode=0,input_file_path='',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=1,select_config={}):

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
		# if len(field_query)==0:
		# 	field_query = ['latent_peak','latent_peak_motif','latent_peak_motif_ori']

		latent_peak = []
		latent_peak_motif,latent_peak_motif_ori = [], []
		latent_peak_tf_link = []
		peak_query_vec_pre1 = feature_query_vec
		flag_shuffle = False
		load_mode_2 = load_mode

		if load_mode_2==0:
			# dict_query1: {'latent_peak','latent_tf','latent_peak_motif'}
			# dict_query1: {'latent_peak_tf','latent_gene','latent_peak_motif','latent_peak_motif_ori'}
			dict_query1 = self.test_query_feature_pre1_2(peak_query_vec=peak_query_vec_pre1,gene_query_vec=gene_query_vec,
															motif_data=motif_data,motif_data_score=motif_data_score,motif_group=motif_group,
															method_type_vec=method_type_vec_dimension,
															peak_read=peak_read,rna_exprs=rna_exprs,rna_exprs_unscaled=rna_exprs_unscaled,
															n_components=n_components,sub_sample=-1,flag_shuffle=flag_shuffle,float_format=float_format,
															input_file_path=input_file_path,save_mode=1,output_file_path=output_file_path,filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot,output_filename='',verbose=verbose,select_config=select_config)

		elif load_mode_2==1:
			input_file_path_query = output_file_path
			# annot_str_vec = ['peak_gene','peak_motif','peak_motif_ori']
			# annot_str_vec = ['peak_tf','peak_motif','peak_motif_ori']
			# field_query = ['peak_motif','peak_tf']
			annot_str_vec = ['peak_motif','peak_tf']
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
				# input_filename = 'pbmc.pre1.0.df_latent.peak_tf.SVD_100.1.txt'
				input_filename = '%s/%s.%s.%s.1.txt'%(input_file_path_query,filename_prefix_save_query,annot_str1,filename_save_annot_2)
				df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
				print('df_query: ',df_query.shape,annot_str1)
				print(df_query[0:2])

				if i2==0:
					peak_query_vec_pre1 = df_query.index
					# field_id2 = field_query[i2+1]
					field_id2 = 'latent_%s'%(annot_str1)
					dict_query1.update({field_id2:df_query})
				else:
					feature_query_pre1 = df_query.index
					feature_vec_2 = pd.Index(peak_query_vec_pre1).intersection(feature_query_pre1,sort=False)
					feature_vec_3 = pd.Index(feature_query_pre1).difference(peak_query_vec_pre1,sort=False)

					# latent_peak = df_query.loc[peak_query_vec_pre1,:]
					feature_query_pre2 = list(feature_vec_2)+list(feature_vec_3)
					feature_query_pre2 = pd.Index(feature_query_pre2)
					
					df_query = df_query.loc[feature_query_pre2,:]
					field_id2 = 'latent_%s'%(annot_str1)
					dict_query1.update({field_id2:df_query})

					motif_query_vec = gene_query_vec
					if len(motif_query_vec)==0:
						motif_query_vec = feature_vec_3
					
					if len(motif_query_vec)>0:
						feature_query_1 = feature_query_pre1.difference(motif_query_vec,sort=False)
						feature_query_2 = pd.Index(peak_query_vec_pre1).intersection(feature_query_1,sort=False)
						feature_query_3 = pd.Index(peak_query_vec_pre1).difference(feature_query_1,sort=False)
						print('feature_query_2: ',len(feature_query_2))
						print('feature_query_3: ',len(feature_query_3))

						latent_peak = df_query.loc[peak_query_vec_pre1,:]
						print('latent_peak: ',latent_peak.shape)
						print(latent_peak[0:2])
						dict_query1.update({'latent_peak':latent_peak})
						
						motif_query_2 = pd.Index(motif_query_vec).intersection(feature_query_pre1,sort=False)
						if len(motif_query_2)>0:
							latent_gene = df_query.loc[motif_query_2,:]
							print('latent_gene: ',latent_gene.shape)
							print(latent_gene[0:2])
							# dict_query1.update({'latent_peak':latent_peak,'latent_gene':latent_gene})
							dict_query1.update({'latent_gene':latent_gene})

		# elif load_mode_2==2:
		# 	feature_type_vec_query = ['motif','peak']
		# 	column_id_query = 'peak_id'
		# 	column_value = 'peak_tf_corr'
		# 	column_vec = [column_id3,column_id2]
		# 	flag_binary = 1
		# 	thresh_value = -0.1
		# 	# n_components = 50
		# 	flag_unduplicate = 0
		# 	method_type_vec_dimension = ['SVD']
		# 	df_feature_link_query = df_peak_tf_query1
		# 	input_file_path_query = select_config['file_path_motif_score_2']
		# 	input_filename = '%s/test_motif_score_normalize_insilico.%s.1.txt'%(input_file_path_query,data_file_type_query)
			
		# 	if os.path.exists(input_filename)==False:
		# 		print('the file does not exist: %s'%(input_filename))
		# 		input_filename = '%s/test_motif_score_normalize_insilico.%s.thresh1.1.txt'%(input_file_path_query,data_file_type_query)

		# 	df_annot = pd.read_csv(input_filename,index_col=0,sep='\t')
		# 	df_annot['motif_id'] = np.asarray(df_annot.index)
			
		# 	print('df_feature_link_query: ',df_feature_link_query.shape)
		# 	print(df_feature_link_query.columns)
		# 	print(df_feature_link_query[0:2])
			
		# 	print('df_annot: ',df_annot.shape)
		# 	print(df_annot.columns)
		# 	print(df_annot[0:2])
		# 	df_annot = df_annot.drop_duplicates(subset=[column_id3,column_id2])
		# 	peak_query_1 = df_feature_link_query[column_id2].unique()
		# 	feature_query_vec = pd.Index(peak_query_vec_pre1).intersection(peak_query_1,sort=False)
		# 	print('peak_query_1, feature_query_vec: ',len(peak_query_1),len(feature_query_vec))
		# 	# df_annot.index = utility_1.test_query_index(df_annot,column_vec=column_vec)
		# 	dict_query1 = self.test_query_feature_pre1_3(df_feature_link=df_feature_link_query,df_annot=df_annot,feature_query_vec=feature_query_vec,column_id_query=column_id_query,
		# 													column_vec=column_vec,column_value=column_value,feature_type_vec=feature_type_vec_query,
		# 													peak_query_vec=feature_query_vec,gene_query_vec=[],motif_data=motif_data,motif_data_score=motif_data_score,motif_group=[],
		# 													method_type_vec=method_type_vec_dimension,peak_read=peak_read,rna_exprs=rna_exprs,rna_exprs_unscaled=rna_exprs_unscaled,
		# 													n_components=n_components,sub_sample=-1,flag_shuffle=False,flag_binary=flag_binary,thresh_value=thresh_value,
		# 													float_format=float_format,flag_unduplicate=flag_unduplicate,input_file_path=input_file_path,
		# 													save_mode=1,output_file_path=output_file_path,filename_prefix_save=filename_prefix_save_2,filename_save_annot=filename_save_annot,output_filename='',verbose=verbose,select_config=select_config)

		# elif load_mode_2==3:
		# 	input_file_path_query = output_file_path
		# 	annot_str_vec = ['peak_tf_link']
		# 	field_query_2 = ['df_latent','df_component']
		# 	dict_query1 = dict()

		# 	method_type_vec_dimension = ['SVD']
		# 	# field_num = len(field_query)
		# 	query_num = len(annot_str_vec)
		# 	for i2 in range(query_num):
		# 		method_type_dimension = method_type_vec_dimension[i2]
		# 		filename_save_annot_2 = '%s_%s'%(method_type_dimension,n_components)

		# 		annot_str1 = annot_str_vec[i2]
		# 		field_id1 = 'df_latent'
		# 		filename_prefix_save_query = '%s.%s'%(filename_prefix_save_2,field_id1)
		# 		input_filename = '%s/%s.%s.%s.1.txt'%(input_file_path_query,filename_prefix_save_query,annot_str1,filename_save_annot_2)
		# 		df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
		# 		print('df_query: ',df_query.shape,annot_str1)
		# 		print(df_query[0:2])

		# 		field_id2 = field_query_pre2[i2]
		# 		dict_query1.update({field_id2:df_query})

		# if load_mode_2 in [0,1]:
		# 	list1 = [dict_query1[field_id] for field_id in field_query]
		# 	latent_peak, latent_gene, latent_peak_motif, latent_peak_motif_ori = list1
		# 	latent_tf = latent_gene.loc[motif_query_vec,:]
		# 	print('latent_peak, latent_tf, latent_peak_motif, latent_peak_motif_ori: ',latent_peak.shape,latent_tf.shape,latent_peak_motif.shape,latent_peak_motif_ori.shape)
		# 	print(latent_peak[0:2])
		# 	print(latent_tf[0:2])
		# 	print(latent_peak_motif[0:2])
		# 	print(latent_peak_motif_ori[0:2])

		# elif load_mode_2 in [2,3]:
		# 	list1 = [dict_query1[field_id] for field_id in field_query_pre2]
		# 	latent_peak_tf_link = list1[0]
		# 	print('latent_peak_tf_link: ',latent_peak_tf_link.shape)
		# 	print(latent_peak_tf_link[0:2])

		return dict_query1

	## query neighbors of feature
	# query neighbors of peak loci
	# def test_query_feature_neighbor_pre1_1(self,data=[],n_neighbors=20,return_distance=True,save_mode=1,verbose=0,select_config={}):

	# 	from sklearn.neighbors import NearestNeighbors
	# 	from scipy.stats import poisson, multinomial

	# 	# Effective genome length for background computaiton
	# 	# eff_genome_length = atac_meta_ad.shape[1] * 5000
	# 	# bin_size = 500
	# 	# eff_genome_length = atac_meta_ad.shape[1] * bin_size

	# 	# Metacell neighbors
	# 	# peak feature neighbors
	# 	# nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None)
	# 	nbrs = NearestNeighbors(n_neighbors=n_neighbors,radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2)
	# 	# nbrs.fit(atac_meta_ad.obsm[low_dim_embedding])
	# 	# meta_nbrs = pd.DataFrame(atac_meta_ad.obs_names.values[nbrs.kneighbors(atac_meta_ad.obsm[low_dim_embedding])[1]],
	# 	# 						 index=atac_meta_ad.obs_names)
	# 	# select_config.update({'meta_nbrs_atac':meta_nbrs})
	# 	feature_mtx = data
	# 	nbrs.fit(feature_mtx)
	# 	# sample_id = feature_mtx.index
	# 	query_id_1 = feature_mtx.index
	# 	neighbor_dist, neighbor_id = nbrs.kneighbors(feature_mtx)
	# 	column_vec = ['neighbor%d'%(id1) for id1 in np.arange(n_neighbors)]
	# 	feature_nbrs = pd.DataFrame(index=query_id_1,columns=column_vec,data=query_id_1.values[neighbor_id])
	# 	dist_nbrs = []
	# 	if return_distance>0:
	# 		dist_nbrs = pd.DataFrame(index=query_id_1,columns=column_vec,data=neighbor_dist)

	# 	return feature_nbrs, dist_nbrs

	## query feature enrichment
	# query the enrichment of predicted peak loci in one type of group
	def test_query_enrichment_group_1_unit1(self,data=[],dict_group=[],dict_thresh=[],group_type_vec=['group1','group2'],column_vec_query=[],flag_enrichment=1,flag_size=0,type_id_1=1,type_id_2=0,save_mode=0,verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			# df_overlap_query = data
			df_query_1 = data  # the number and percentage of feature query in each group 
			dict_group_basic = dict_group # the group annotation of feature query

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
				## feature type 1: motif feature
				group_type_1 = group_type_vec[0]
				id1 = (df_query_1['group_type']==group_type_1)
				df_query1_1 = df_query_1.loc[id1,:]
				# query the enrichment of predicted peak loci in paired groups
				df_query_group1_1, dict_query_group1_1 = self.test_query_enrichment_group_2_unit1(data=df_query1_1,dict_group=dict_group,dict_thresh=dict_thresh,column_vec_query=column_vec_query,flag_enrichment=flag_enrichment,flag_size=flag_size,type_id_1=type_id_1,type_id_2=type_id_2,
																									save_mode=save_mode,verbose=verbose,select_config=select_config)
				
				list1 = [df_query_group1_1, dict_query_group1_1]
				dict_query_1 = {group_type_1:list1}

				if len(group_type_vec)>1:
					group_type_2 = group_type_vec[1]
					id2 = (df_query_1['group_type']==group_type_2)
					df_query1_2 = df_query_1.loc[id2,:]
					# query the enrichment of predicted peak loci in paired groups
					df_query_group2_1, dict_query_group2_1 = self.test_query_enrichment_group_2_unit1(data=df_query1_2,dict_group=dict_group,dict_thresh=dict_thresh,column_vec_query=column_vec_query,flag_enrichment=flag_enrichment,flag_size=flag_size,type_id_1=type_id_1,type_id_2=type_id_2,
																										save_mode=save_mode,verbose=verbose,select_config=select_config)

					list2 = [df_query_group2_1,dict_query_group2_1]
					dict_query_1.update({group_type_2:list2})

			return dict_query_1

	## query feature enrichment
	# query the enrichment of predicted peak loci in paired groups
	def test_query_enrichment_group_2_unit1(self,data=[],dict_group=[],dict_thresh=[],group_type_vec=['group1','group2'],column_vec_query=['overlap','pval_fisher_exact_'],flag_enrichment=1,flag_size=0,type_id_1=1,type_id_2=0,save_mode=0,verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			df_overlap_query = data
			df_query_1 = df_overlap_query # the overlaping between groups
			# dict_group_basic = dict_group # the group annotation of feature query

			thresh_overlap_default_1 = 0
			thresh_overlap_default_2 = 0
			thresh_overlap = 0
			# thresh_pval_1 = 0.20
			thresh_pval_1 = 0.25

			column_1 = 'thresh_overlap_default_1'
			column_2 = 'thresh_overlap_default_2'
			column_3 = 'thresh_overlap'
			column_pval = 'thresh_pval_1'
			column_query1, column_query2 = column_vec_query[0:2]
			# id1 = (df_overlap_query['overlap']>thresh_value_overlap)
			# id2 = (df_overlap_query['pval_chi2_']<thresh_pval_1)

			enrichment_query = flag_enrichment
			df_overlap_query_pre1 = df_overlap_query
			dict_query = dict()
			if flag_enrichment>0:
				print('select group based on enrichment')
				if column_1 in dict_thresh:
					thresh_overlap_default_1 = dict_thresh[column_1]

				if column_pval in dict_thresh:
					thresh_pval_1 = dict_thresh[column_pval]

				flag1=0
				try:
					id1 = (df_query_1[column_query1]>thresh_overlap_default_1)
				except Exception as error:
					print('error! ',error)
					flag1=1

				flag2=0
				try:
					id2 = (df_query_1[column_query2]<thresh_pval_1)
				except Exception as error:
					print('error! ',error)
					try: 
						column_query2_1 = 'pval_chi2_'
						id2 = (df_query_1[column_query2_1]<thresh_pval_1)
					except Exception as error:
						print('error! ',error)
						flag2=1

				id_1 = []
				if (flag2==0):
					if (flag1==0):
						id_1 = (id1&id2)
					else:
						id_1 = id2
				else:
					if (flag1==0):
						id_1 = id1

				if (flag1+flag2<2):
					df_overlap_query1 = df_query_1.loc[id_1,:]
					print('the original overlap, the overlap with enrichment above threshold')
					print('df_overlap_query, df_overlap_query1: ',df_query_1.shape,df_overlap_query1.shape)
				else:
					df_overlap_query1 = []
					print('df_overlap_query, df_overlap_query1: ',df_query_1.shape,len(df_overlap_query1))

			df_query_2 = df_query_1.loc[df_query_1[column_query1]>0]
			print('the original overlap, the overlap with number above zero')
			print('df_query_1, df_query_2: ',df_query_1.shape,df_query_2.shape)

			query_value_1 = df_query_1[column_query1]
			query_value_2 = df_query_2[column_query1]
			quantile_vec_1 = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
			query_vec_1 = ['max','min','mean','median']+['percentile_%.2f'%(percentile) for percentile in quantile_vec_1]
			t_value_1 = utility_1.test_stat_1(query_value_1,quantile_vec=quantile_vec_1)
			t_value_2 = utility_1.test_stat_1(query_value_2,quantile_vec=quantile_vec_1)
			query_value = np.asarray([t_value_1,t_value_2]).T
			df_quantile_1 = pd.DataFrame(index=query_vec_1,columns=['value_1','value_2'],data=query_value)
			dict_query.update({'group_size_query':df_quantile_1})

			if flag_size>0:
				print('select group based on number of members')
				if column_2 in dict_thresh:
					thresh_overlap_default_2 = dict_thresh[column_2]

				if column_3 in dict_thresh:
					thresh_overlap = dict_thresh[column_3]

				# thresh_quantile_1 = 0.25
				thresh_quantile_1 = -1
				column_pre2 = 'thresh_quantile_overlap'
				if column_pre2 in dict_thresh:
					thresh_quantile_1 = dict_thresh[column_pre2]
					print('thresh_quantile_1: ',thresh_quantile_1)
				
				# df_query_2 = df_query_1.loc[df_query_1[column_query1]>0]
				# print('the original overlap, the overlap with number above zero')
				# print('df_query_1, df_query_2: ',df_query_1.shape,df_query_2.shape)
				if thresh_quantile_1>0:
					query_value = df_query_2[column_query1]
					thresh_size_1 = np.quantile(query_value,thresh_quantile_1)

					if type_id_1>0:
						thresh_size_ori = thresh_size_1
						thresh_size_1 = np.max([thresh_overlap_default_2,thresh_size_1])
				else:
					thresh_size_1 = thresh_overlap

				id_2 = (df_query_1[column_query1]>=thresh_size_1)
				df_overlap_query2 = df_query_1.loc[id_2,:]
				print('the original overlap, the overlap with number above the threshold')
				print('df_overlap_query, df_overlap_query2: ',df_query_1.shape,df_overlap_query2.shape)
				print('thresh_size_1: ',thresh_size_1)

				if enrichment_query>0:
					if type_id_2==0:
						id_pre1 = (id_1&id_2)
					else:
						id_pre1 = (id_1|id_2)
					df_overlap_query_pre1 = df_query_1.loc[id_pre1,:]

					df_overlap_query_pre1.loc[id_1,'enrichment'] = 1
					df_overlap_query_pre1.loc[id_2,'group_size'] = 1
					print('df_overlap_query, df_overlap_query_pre1: ',df_query_1.shape,df_overlap_query_pre1.shape)
				else:
					df_overlap_query_pre1 = df_overlap_query2
			else:
				df_overlap_query_pre1 = df_overlap_query1

			dict_query.update({'enrichment':df_overlap_query1,'group_size':df_overlap_query2})
			return df_overlap_query_pre1, dict_query

	# feature group query and feature neighbor query
	# TF binding prediction by feature group query and feature neighbor query
	def test_query_feature_group_neighbor_pre1_2_unit1(self,data=[],dict_group=[],dict_neighbor=[],group_type_vec=['group1','group2'],feature_type_vec=[],group_vec_query=[],column_vec_query=[],n_neighbors=30,parallel=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

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

		column_query1 = '%s_neighbor'%(feature_type_query_1)	# query neighbor in feature space 1, without restriction in the same group of the selected peaks
		column_query2 = '%s_neighbor'%(feature_type_query_2)	# query neighbor in feature space 2, without restriction in the same group of the selected peaks

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
		column_vec_query1 = [column_pred_2,column_pred_3,column_pred_5,column_pred_6,column_query1,column_query2,column_1,column_2]

		if flag_neighbor>0:
			# column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
			# column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)
			field_id1 = 'feature_nbrs'
			list_query2 = [dict_neighbor[feature_type_query][field_id1] for feature_type_query in feature_type_vec]
			feature_nbrs_1, feature_nbrs_2 = list_query2[0:2]

			id2 = (df_pre1[column_pred2]>0)
			column_annot1 = 'group_id2'
			column_vec_2 = ['%s_group'%(feature_type_query) for feature_type_query in feature_type_vec]
			if not (column_annot1) in df_pre1:
				# column_vec_2 = ['latent_peak_motif_group','latent_peak_tf_group']
				# column_vec_2 = ['%s_group'%(feature_type_query) for feature_type_query in feature_type_vec]
				df_pre1[column_annot1] = utility_1.test_query_index(df_pre1,column_vec=column_vec_2,symbol_vec='_')

			column_id2 = 'peak_id'
			column_vec_1 = ['group1','group2']
			df_group1 = pd.DataFrame(columns=column_vec_1,data=np.asarray(group_vec_query))
			df_group1.index = utility_1.test_query_index(df_group1,column_vec=column_vec_1)
			group_vec_1 = df_group1['group1'].unique()
			group_vec_2 = df_group1['group2'].unique()
			group_num1, group_num2 = len(group_vec_1), len(group_vec_2)
			print('group_vec_1, group_vec_2: ',group_num1,group_num2)
			df_pre1.loc[:,column_vec_query1] = False
			
			parallel = 0
			# cnt1=0
			# for (group_id_1,group_id_2) in group_vec_query:
			for group_id_1 in group_vec_1:
				start = time.time()
				id1 = (df_group1['group1']==group_id_1)
				df_query1 = df_group1.loc[id1,:]
				group_vec_2_query = df_query1['group2'].unique()

				# column_vec_query_1 = column_vec_2 + [column_annot1]
				column_vec_query_1 = [column_annot1]

				if parallel==0:
					for group_id_2 in group_vec_2_query:
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

							df_pre1.loc[peak_neighbor_1,column_query1] = 1
							df_pre1.loc[peak_neighbor_2,column_query2] = 1

							df_pre1.loc[peak_query_1,column_1] = 1
							df_pre1.loc[peak_query_2,column_2] = 1

							# stop = time.time()
							# print('query neighbors of peak loci within paired group',group_id_1,group_id_2,stop-start)

							# start = time.time()
							# column_pred_3 = '%s.pred_group_neighbor'%(method_type_feature_link)
							peak_query_vec_2 = pd.Index(peak_query_1).intersection(peak_query_2,sort=False)
							df_pre1.loc[peak_query_vec_2,column_pred_3] = 1

							# peaks within the same groups with the selected peak in the two feature space and peaks are neighbors of the selected peaks
							# column_pred_5 = '%s.pred_group_neighbor_1'%(method_type_feature_link)
							peak_query_vec_3 = pd.Index(peak_query_1).union(peak_query_2,sort=False)
							df_pre1.loc[peak_query_vec_3,column_pred_5] = 1

							# print(df_pre1.loc[:,[column_pred_2,column_1,column_2]])
							# df_pre1[column_pred_5] = ((df_pre1[column_pred_2]>0)&((df_pre1[column_1]>0)|(df_pre1[column_2]>0))).astype(int)

							# stop = time.time()
							# print('query neighbors of peak loci 1',group_id_1,group_id_2,stop-start)

							if flag_neighbor_2>0:
								peak_neighbor_pre2 = pd.Index(peak_neighbor_1).intersection(peak_neighbor_2,sort=False)
								df_pre1.loc[peak_neighbor_pre2,column_pred_6] = 1

						stop = time.time()
						if (group_id_1%10==0) and (group_id_2%10==0):
							print('query neighbors of peak loci',group_id_1,group_id_2,stop-start)
				
				stop = time.time()
				group_num_2 = len(group_vec_2_query)
				print('query neighbors of peak loci',group_id_1,group_num_2,stop-start)
			
			if parallel>0:
				df_pre1.loc[:,column_vec_query1] = df_pre1.loc[:,column_vec_query1].astype(int)

		flag_neighbor_3 = 1  # query neighbor of selected peak
		if (flag_neighbor_3>0):
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

	## select training sample
	def test_query_training_group_pre1(self,data=[],dict_annot=[],motif_id='',method_type_feature_link='',dict_thresh=[],thresh_vec=[],input_file_path='',save_mode=1,output_file_path='',verbose=0,select_config={}):

		flag_select_1=1
		# column_pred1 = '%s.pred'%(method_type_feature_link)
		column_pred1 = select_config['column_pred1']
		df_query1 = data
		id_pred1 = (df_query1[column_pred1]>0)
		peak_loc_pre1 = df_query1.index
		# df_query2 = df_query1.loc[id_pred1,:]
		peak_loc_pred1 = peak_loc_pre1[id_pred1]

		# column_corr_1 = field_id1
		# column_pval = field_id2
		column_corr_1 = 'peak_tf_corr'
		column_pval = 'peak_tf_pval_corrected'
		thresh_corr_1, thresh_pval_1 = 0.30, 0.05
		thresh_corr_2, thresh_pval_2 = 0.1, 0.1
		thresh_corr_3, thresh_pval_2 = 0.05, 0.1

		if flag_select_1>0:
			# find the paired groups with enrichment
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
									
					# thresh_pval_1 = 0.10
					# thresh_pval_1 = 0.20
					thresh_pval_group = 0.25
					# thresh_quantile_overlap = 0.50
					thresh_quantile_overlap = 0.75
					thresh_vec = [thresh_overlap_default_1,thresh_overlap_default_2,thresh_overlap,thresh_pval_group,thresh_quantile_overlap]
				
				dict_thresh = dict(zip(column_thresh_query,thresh_vec))

			# query the enrichment of predicted peak loci in one type of group
			group_type_vec = ['group1','group2']
			df_group_basic_query_2 = dict_annot['df_group_basic_query_2']
			dict_group_basic_2 = dict_annot['dict_group_basic_2']
			print('df_group_basic_query_2: ',df_group_basic_query_2.shape)
			print(df_group_basic_query_2.columns)
			print(df_group_basic_query_2[0:5])
			
			column_vec_query = ['count','pval_fisher_exact_']
			flag_enrichment = 1
			flag_size = 1
			type_id_1, type_id_2 = 1, 1
			# dict_query = {'enrichment':df_overlap_query1,'group_size':df_overlap_query2}
			dict_query_pre1 = self.test_query_enrichment_group_1_unit1(data=df_group_basic_query_2,dict_group=dict_group_basic_2,dict_thresh=dict_thresh,
																				group_type_vec=group_type_vec,
																				column_vec_query=column_vec_query,
																				flag_enrichment=flag_enrichment,flag_size=flag_size,
																				type_id_1=type_id_1,type_id_2=type_id_2,
																				save_mode=1,verbose=verbose,select_config=select_config)
							
			group_type_1, group_type_2 = group_type_vec[0:2]
			df_query_group1_1,dict_query_group1_1 = dict_query_pre1[group_type_1]
			df_query_group2_1,dict_query_group2_1 = dict_query_pre1[group_type_2]

			field_query_2 = ['enrichment','group_size','group_size_query']
			field_id1, field_id2 = field_query_2[0:2]
			field_id3 = field_query_2[2]

			list_query1 = []
			dict_query1 = dict()
			data_file_type_query = select_config['data_file_type']
			config_id_load = select_config['config_id_load']
			method_type_group = select_config['method_type_group']
			n_neighbors = select_config['neighbor_num']
			for group_type in group_type_vec:
				print('group_type: ',group_type)
				dict_query_group = dict_query_pre1[group_type][1]
				group_vec_query1_1 = dict_query_group[field_id1].index.unique()
				group_vec_query2_1 = dict_query_group[field_id2].index.unique()
				group_num1_1, group_num2_1 = len(group_vec_query1_1), len(group_vec_query2_1)
				# list_query1.append([group_vec_query1_1,group_vec_query2_1])
				dict_query1.update({group_type:[group_vec_query1_1,group_vec_query2_1]})
								
				print('group_vec_query1_1, group_vec_query2_1: ',group_num1_1,group_num2_1)
				print(group_vec_query1_1)
				print(group_vec_query2_1)

				df_quantile_1 = dict_query_group[field_id3]
				print('df_quantile_1: ',df_quantile_1.shape)
				print(df_quantile_1[0:5])

				# iter_id1 = 0
				# filename_save_annot_query_1 = '%s.%d'%(data_file_type_query,config_id_load)
				# filename_save_annot_query1 = '%s.neighbor%d'%(method_type_group,n_neighbors)
				# filename_save_annot = '%s.%s.%s'%(filename_save_annot_query,motif_id1,filename_save_annot_1)
				# output_filename = '%s/test_query_quantile.%s.2.txt'%(output_file_path,group_type)
				# output_filename = '%s/test_query_quantile.%s.%s.%s.%s.2.txt'%(output_file_path,filename_save_annot_query1,group_type,motif_id1,filename_save_annot_query_1)
				
				filename_link_annot = select_config['filename_annot']
				output_filename = '%s/test_query_quantile.%s.%s.txt'%(output_file_path,motif_id,filename_link_annot)
				df_quantile_1.to_csv(output_filename,sep='\t')

			group_vec_query1_1, group_vec_query2_1 = dict_query1[group_type_1]
			group_vec_query1_2, group_vec_query2_2 = dict_query1[group_type_2]

			# query the enrichment of predicted peak loci in paired groups
			column_vec_query_2 = ['overlap','pval_fisher_exact_']
			flag_enrichment = 1
			flag_size = 1
			type_id_1, type_id_2 = 1, 1
			df_overlap_query = dict_annot['df_overlap_query']
			df_overlap_query_pre2, dict_query_pre2 = self.test_query_enrichment_group_2_unit1(data=df_overlap_query,dict_group=[],dict_thresh=dict_thresh,group_type_vec=group_type_vec,
																									column_vec_query=column_vec_query_2,flag_enrichment=flag_enrichment,flag_size=flag_size,type_id_1=type_id_1,type_id_2=type_id_2,
																									save_mode=1,verbose=verbose,select_config=select_config)
			
			group_vec_query2 = np.asarray(df_overlap_query_pre2.loc[:,group_type_vec].astype(int))
			group_num_2 = len(group_vec_query2)
			print('group_vec_query2: ',group_num_2)
			print(group_vec_query2[0:5])

			df_1 = dict_query_pre2[field_id1] # group with enrichment above threshold
			group_vec_query1_pre2 = df_1.loc[:,group_type_vec].astype(int)
			group_num1_pre2 = len(group_vec_query1_pre2)
			
			df_2 = dict_query_pre2[field_id2] # group with group_size above threshold
			group_vec_query2_pre2 = df_2.loc[:,group_type_vec].astype(int)
			group_num2_pre2 = len(group_vec_query2_pre2)

			print('field_id, group_vec_query1_pre2: ',field_id1,group_num1_pre2)
			# print(group_vec_query1_pre2)
			print(df_1)

			print('field_id, group_vec_query2_pre2: ',field_id2,group_num2_pre2)
			# print(group_vec_query2_pre2)
			print(df_2)

			group_vec_1 = group_vec_query1_1
			group_vec_1_overlap = group_vec_query1_pre2[group_type_1].unique()

			group_vec_2 = group_vec_query1_2
			group_vec_2_overlap = group_vec_query1_pre2[group_type_2].unique()

			group_vec_pre1 = pd.Index(group_vec_1).difference(group_vec_1_overlap,sort=False)
			print('group with enrichment in feature type 1 but not enriched in joint groups')
			print('group_vec_pre1: ',len(group_vec_pre1))
			# df1_query2 = df_overlap_query.loc[df_overlap_query[group_type_1].isin(group_vec_pre1),:]
			list1 = []
			column_query1, column_query2 = column_vec_query_2[0:2]
			df_overlap_query = df_overlap_query.sort_values(by=['pval_fisher_exact_'],ascending=True)
			thresh_size_query1 = 1

			flag_group_pre2 = 0
			if flag_group_pre2>0:
				for group_type_query in group_vec_pre1:
					df1 = df_overlap_query.loc[df_overlap_query[group_type_1]==group_type_query,:]
					df1_1 = df1.loc[df1[column_query1]>thresh_size_query1,:]
					group_query_1 = np.asarray(df1.loc[:,group_type_vec])[0]
					group_query_2 = np.asarray(df1_1.loc[:,group_type_vec])
					list1.append(group_query_1)
					list1.extend(group_query_2)

			list2 = []
			group_vec_pre2 = pd.Index(group_vec_2).difference(group_vec_2_overlap,sort=False)
			print('group with enrichment in feature type 2 but not enriched in joint groups')
			print('group_vec_pre2: ',len(group_vec_pre2))
			if flag_group_pre2>0:
				for group_type_query in group_vec_pre2:
					df2 = df_overlap_query.loc[df_overlap_query[group_type_2]==group_type_query,:]
					df2_1 = df1.loc[df1[column_query1]>thresh_size_query1,:]
					group_query_1 = np.asarray(df2.loc[:,group_type_vec])[0]
					group_query_2 = np.asarray(df2_1.loc[:,group_type_vec])
					list2.append(group_query_1)
					list2.extend(group_query_2)

			group_vec_query1_pre2 = np.asarray(group_vec_query1_pre2)
			list_pre1 = list(group_vec_query1_pre2)+list1+list2

			query_vec = np.asarray(list_pre1)
			df_1 = pd.DataFrame(data=query_vec,columns=group_type_vec).astype(int)
			df_1.index = utility_1.test_query_index(df_1,column_vec=[group_type_1,group_type_2],symbol_vec=['_'])
			df_1 = df_1.drop_duplicates(subset=group_type_vec)
			group_id_1 = df_1.index
			group_id_pre1 = ['%s_%s'%(group_1,group_2) for (group_1,group_2) in group_vec_query1_pre2]
			group_id_2 = group_id_1.difference(group_id_pre1,sort=False)

			column_query_1 = df_overlap_query.columns.difference(group_type_vec,sort=False)
			df_1.loc[:,column_query_1] = df_overlap_query.loc[group_id_1,column_query_1]
			group_vec_query1 = df_1.loc[:,group_type_vec]
			print('df_1: ',df_1.shape)
			print(df_1)

			feature_type_vec = select_config['feature_type_vec']
			feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]
			group_type_vec_2 = ['%s_group'%(feature_type_query_1),'%s_group'%(feature_type_query_2)]

			df_query1['group_id2'] = utility_1.test_query_index(df_query1,column_vec=group_type_vec_2,symbol_vec=['_'])
			id1 = (df_query1['group_id2'].isin(group_id_1))
			id2 = (df_query1['group_id2'].isin(group_id_2))
			df_query1.loc[id1,'group_overlap'] = 1 # the group with enrichment
			df_query1.loc[id2,'group_overlap'] = 2 # the paired group without enrichment but the single group with enrichment
			print('group_id_1, group_id_2: ',len(group_id_1),len(group_id_2))

			return df_query1

	## select training sample
	def test_query_feature_quantile_1(self,data=[],query_idvec=[],column_vec_query=[],flag_corr_1=1,save_mode=1,verbose=0,select_config={}):

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

		# flag_corr_1=1
		# peak_loc_query_group2_1 = []
		df_query1 = data
		column_id2 = 'peak_id'
		if not (column_id2 in df_query1.columns):
			df_query1['peak_id'] = df_query1.index.copy()

		query_value_1 = df_query1[column_corr_1]
		query_value_1 = query_value_1.fillna(0)

		column_quantile_pre1 = '%s_quantile'%(column_corr_1)
		normalize_type = 'uniform'	# normalize_type: 'uniform', 'normal'
		score_mtx = quantile_transform(query_value_1[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
		df_query1[column_quantile_pre1] = score_mtx[:,0]

		# the peak loci with predicted TF binding
		# id_pred1 = query_idvec
		# df_pre2 = df_query1.loc[id_pred1,:]
		if len(query_idvec)>0:
			df_pre2 = df_query1.loc[query_idvec,:]
		else:
			df_pre2 = df_query1
			query_idvec = df_query1.index

		df_pre2 = df_pre2.sort_values(by=[column_score_query1],ascending=False)

		query_value = df_pre2[column_corr_1]
		query_value = query_value.fillna(0)

		column_quantile_1 = '%s_quantile_2'%(column_corr_1)
		normalize_type = 'uniform'   # normalize_type: 'uniform', 'normal'
		score_mtx = quantile_transform(query_value[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
		# df_pre2[column_quantile_1] = score_mtx[:,0]
		df_query1.loc[query_idvec,column_quantile_1] = score_mtx[:,0]
							
		query_value_2 = df_pre2[column_score_query1]
		query_value_2 = query_value_2.fillna(0)

		# column_score_query1 = '%s.score'%(method_type_feature_link)
		column_quantile_2 = '%s_quantile'%(column_score_query1)
		normalize_type = 'uniform'   # normalize_type: 'uniform', 'normal'
		score_mtx_2 = quantile_transform(query_value_2[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
		# df_pre2[column_quantile_2] = score_mtx_2[:,0]
		df_query1.loc[query_idvec,column_quantile_2] = score_mtx_2[:,0]

		column_vec_quantile = [column_quantile_pre1,column_quantile_1,column_quantile_2]
		select_config.update({'column_vec_quantile':column_vec_quantile})

		return df_query1, select_config

	## select training sample based on correlation value
	def test_query_training_select_correlation_1(self,data=[],thresh_vec=[0.1,0.90],save_mode=1,verbose=0,select_config={}):
		
		flag_corr_1 = 1
		if flag_corr_1>0:
			# select peak with peak accessibility-TF expression correlation above threshold
			# df_query = df_pre2
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

			id_score_query2_1 = (df_query[column_corr_1]>thresh_corr_1)&(df_query[column_pval]<thresh_pval_1)

			# query_value_1 = df_query1.loc[id_pred2,column_corr_1]
			query_value = df_query[column_corr_1]
			query_value = query_value.fillna(0)
			
			thresh_corr_query1 = np.quantile(query_value,thresh_corr_quantile)
			thresh_corr_query2 = np.min([thresh_corr_1,thresh_corr_query1])
			print('thresh_corr_query1, thresh_corr_query2: ',thresh_corr_query1, thresh_corr_query2)
			# id_score_query_2_1 = (df_query[column_corr_1]>thresh_corr_query2)&(df_query[column_pval]<thresh_pval_1)
			id_score_query_2_1 = (df_query[column_corr_1]>thresh_corr_query2)
								
			# thresh_corr_quantile_1 = 0.90
			# id_score_query_2_1 = (df_pre2[column_1]>thresh_corr_quantile_1)

			# id_score_query2 = (id_pred2)&(id_score_query2_1)
			# id_score_query_2 = (id_pred2)&(id_score_query_2_1)

			# peak_loc_query_2_1 = peak_loc_ori[id_score_query2]
			# peak_loc_query_2 = peak_loc_ori[id_score_query_2]

			peak_loc_query_2_1 = peak_loc_query[id_score_query2_1]
			peak_loc_query_2 = peak_loc_query[id_score_query_2_1]

			print('peak_loc_query_2_1, peak_loc_query_2: ',len(peak_loc_query_2_1),len(peak_loc_query_2))
			# peak_loc_query_group2_1 = peak_loc_query_2

			feature_query_vec_1 = peak_loc_query_2_1
			feature_query_vec_2 = peak_loc_query_2

			return feature_query_vec_1, feature_query_vec_2

	## select training sample based on feature link score
	def test_query_training_select_feature_link_score_1(self,data=[],column_vec_query=[],thresh_vec=[],save_mode=1,verbose=0,select_config={}):

		flag_score_1=1
		if flag_score_1>0:
			# select peak query based on score threshold
			if len(thresh_vec)>0:
				thresh_score_query_pre1, thresh_score_query_1, thresh_corr_query = thresh_vec[0:3]
			else:
				thresh_corr_query = 0.10
				thresh_score_query_pre1 = 0.15
				thresh_score_query_1 = 0.15
				# thresh_score_query_1 = 0.20
			column_1 = 'thresh_score_group_1'
			if column_1 in select_config:
				thresh_score_group_1 = select_config[column_1]
				thresh_score_query_1 = thresh_score_group_1

			# column_score_query1 = '%s.score'%(method_type_feature_link)
			column_corr_1, column_score_query1 = column_vec_query[0:2]
			id_score_query1 = (df_pre2[column_score_query1]>thresh_score_query_1)
			id_score_query1 = (id_score_query1)&(df_pre2[column_corr_1]>thresh_corr_query)
			df_query1_2 = df_pre2.loc[id_score_query1,:]

			peak_loc_query_1 = df_query1_2.index 	# the peak loci with prediction and with score above threshold
			peak_num_1 = len(peak_loc_query_1)
			print('peak_loc_query_1: ',peak_num_1)
			# peak_loc_query_group2_1 = pd.Index(peak_loc_query_1).union(peak_loc_query_2,sort=False)

			feature_query_vec_1 = peak_loc_query_1

			return feature_query_vec_1

	## select training sample
	def test_query_training_select_pre1(self,data=[],column_vec_query=[],flag_corr_1=1,flag_score_1=0,flag_enrichment_sel=1,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		df_query1 = data
		column_vec_quantile = select_config['column_vec_quantile']
		column_quantile_pre1,column_quantile_1,column_quantile_2 = column_vec_quantile[0:3]
		df_query1 = df_query1.sort_values(by=['group_overlap',column_quantile_pre1],ascending=False)

		# if (save_mode>0) and (output_file_path!=''):
		# 	df_query1 = df_query1.round(7)
		# 	output_filename = '%s/%s.%s.%s.%s_%s.neighbor%d.%d.copy2_2.2.txt'%(output_file_path,filename_prefix_1,motif_id1,method_type_group,feature_type_query_1,feature_type_query_2,n_neighbors,config_id_load)				
		# 	df_query1.to_csv(output_filename,sep='\t')
						
		flag_query1=1
		if flag_query1>0:
			peak_loc_query_1 = []
			peak_loc_query_2 = []
			method_type_feature_link = select_config['method_type_feature_link']
			column_corr_1 = 'peak_tf_corr'
			column_score_query1 = '%s.score'%(method_type_feature_link)
			column_vec_query = [column_corr_1,column_score_query1]
			peak_loc_query_2_1 = []
			peak_loc_query_2 = []
			if flag_corr_1>0:
				# select peak with peak accessibility-TF expression correlation below threshold
				# df_query = df_pre2
				thresh_corr_1 = 0.1
				thresh_corr_quantile = 0.90
				thresh_vec_1 = [thresh_corr_1,thresh_corr_quantile]
				peak_loc_query_2_1, peak_loc_query_2 = self.test_query_training_select_correlation_1(data=df_query1,thresh_vec=thresh_vec_1,save_mode=1,verbose=verbose,select_config=select_config)

			flag_score_1=0
			peak_loc_query_1 = []
			if flag_score_1>0:
				# select peak query based on score threshold
				thresh_score_query_pre1 = 0.15
				thresh_score_query_1 = 0.15
				thresh_corr_1 = 0.1
				thresh_vec_2 = [thresh_score_query_pre1,thresh_score_query_1,thresh_corr_1]
				peak_loc_query_1 = self.test_query_training_select_feature_link_score_1(data=df_pre2,thresh_vec=thresh_vec_2,save_mode=1,verbose=verbose,select_config=select_config)		

			peak_loc_query_pre2 = pd.Index(peak_loc_query_2).union(peak_loc_query_1,sort=False)
			feature_query_vec_pre2 = peak_loc_query_pre2
			peak_loc_query_group2_1 = feature_query_vec_pre2

			if flag_enrichment_sel>0:
				# thresh_vec_query1=[0.25,0.75]
				thresh_vec_query1=[0.5,0.9]
				column_1 = 'thresh_vec_sel_1'
				if column_1 in select_config:
					thresh_vec_query1 = select_config[column_1]
				print('thresh_vec_query1: ',thresh_vec_query1)
				thresh_vec_query2=[0.95,0.001]
				
				# column_vec_quantile = select_config['column_vec_quantile']
				# column_quantile_pre1,column_quantile_1,column_quantile_2 = column_vec_quantile[0:3]
				column_vec_query = [column_quantile_1,column_quantile_2]
				peak_loc_query_group2_1 = self.test_query_training_select_pre2(data=df_query1,feature_query_vec=feature_query_vec_pre2,column_vec_query=column_vec_query,
																				thresh_vec_1=thresh_vec_query1,thresh_vec_2=thresh_vec_query2,
																				save_mode=1,verbose=verbose,select_config=select_config)


			feature_query_group2 = peak_loc_query_group2_1
			
			return feature_query_group2

	## select training sample based on feature group enrichment
	def test_query_training_select_pre2(self,data=[],feature_query_vec=[],column_vec_query=[],thresh_vec_1=[0.25,0.75],thresh_vec_2=[0.95,0.001],save_mode=1,verbose=0,select_config={}):
		
		df_query = data
		id_group_overlap = (df_query['group_overlap']>0)
		# print('df_query1: ',df_query1.shape)
		# print(df_query1.columns)
		print('df_query: ',df_query.shape)
		# print(df_query.columns)
		flag_enrichment_sel = 1
		if flag_enrichment_sel>0:
			id1 = (id_group_overlap)
			id2 = (~id_group_overlap)
			
			# df_query = df_query.sort_values(by=column_score_query1,ascending=False)
			# df_group_pre2 = df_pre2.groupby(by=['group_id2'])
			group_id_query = df_query.loc[id1,'group_id2'].unique()

			if len(thresh_vec_1)>0:
				thresh_1, thresh_2 = thresh_vec_1[0:2]
			else:
				thresh_1, thresh_2 = 0.25, 0.75
			# thresh_quantile_query1_1, thresh_quantile_query2_1 = 0.25, 0.25
			# thresh_quantile_query1_2, thresh_quantile_query2_2 = 0.75, 0.75

			thresh_quantile_query1_1, thresh_quantile_query2_1 = thresh_1, thresh_1
			thresh_quantile_query1_2, thresh_quantile_query2_2 = thresh_2, thresh_2

			column_quantile_1, column_quantile_2 = column_vec_query[0:2]
			id_score_1 = (df_query[column_quantile_1]>thresh_quantile_query1_1) # lower threshold for group with enrichment
			id_score_2 = (df_query[column_quantile_2]>thresh_quantile_query2_1)

			id1_1 = id1&(id_score_1|id_score_2)
			# id2_1 = id2&(id_score_1|id_score_2)

			id_score_1_2 = (df_query[column_quantile_1]>thresh_quantile_query1_2) # higher threshold for group without enrichment
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
				id_corr_1 = (df_query[column_corr_1].abs()>thresh_corr_lower_bound)
				id_query_2 = id_query_2&(id_corr_1)

			df_query_pre2 = df_query.loc[id_query_2,:]

			peak_loc_query_3 = df_query_pre2.index 	# the peak loci with prediction and with score above threshold
			peak_num_3 = len(peak_loc_query_3)
			print('peak_loc_query_3: ',peak_num_3)

		# peak_loc_query_pre2 = pd.Index(peak_loc_query_2).union(peak_loc_query_1,sort=False)
		peak_loc_query_pre2 = feature_query_vec
		peak_loc_query_group2_1 = pd.Index(peak_loc_query_pre2).union(peak_loc_query_3,sort=False)
							
		peak_num_group2_1 = len(peak_loc_query_group2_1)
		print('peak_loc_query_group2_1: ',peak_num_group2_1)
							
		feature_query_vec_1 = peak_loc_query_group2_1

		return feature_query_vec_1

	## select training sample in class 2: the previous selection method
	# def test_query_training_select_group2_ori(self,data=[],save_mode=1,verbose=0,select_config={}):

	# 	flag_thresh1 = 0
	# 	# select positive and negative group
	# 	if flag_thresh1>0:
	# 		# select peak query based on score threshold
	# 		# the previous selection threshold
	# 		flag_thresh1_1=0
	# 		if flag_thresh1_1>0:
	# 			thresh_score_query_pre1 = 0.15
	# 			thresh_score_query_1 = 0.15
	# 			# thresh_score_query_1 = 0.20
	# 			column_1 = 'thresh_score_group_1'
	# 			if column_1 in select_config:
	# 				thresh_score_group_1 = select_config[column_1]
	# 				thresh_score_query_1 = thresh_score_group_1

	# 				column_score_query1 = '%s.score'%(method_type_feature_link)
	# 				id_score_query1 = (df_pre1[column_score_query1]>thresh_score_query_1)
	# 				id_score_query1 = (id_score_query1)&(df_pre1[column_corr_1]>thresh_corr_2)
	# 				df_query1_2 = df_pre1.loc[(id_pred1&id_score_query1),:]

	# 			peak_loc_query_group2_1 = df_query1_2.index 	# the peak loci with prediction and with score above threshold
	# 			peak_num_group2_1 = len(peak_loc_query_group2_1)
	# 			print('peak_loc_query_group2_1: ',peak_num_group2_1)

	# 			df_query2_2 = df_pre1.loc[(~id_pred1)&id_motif,:]
	# 			peak_loc_query_group2_2_ori = df_query2_2.index  # the peak loci without prediction and with motif
	# 			peak_num_group2_2_ori = len(peak_loc_query_group2_2_ori)
	# 			print('peak_loc_query_group2_2_ori: ',peak_num_group2_2_ori)

	# 			id_score_query2 = (df_pre1[column_corr_1]>thresh_corr_1)&(df_pre1[column_pval]<thresh_pval_1)
	# 			id_group = (df_pre1['motif_group_1']>0)
	# 			id_score_query3_1 = (df_pre1[column_corr_1].abs()<thresh_corr_2)
	# 			id_score_query3_2 = (~id_group)&(df_pre1[column_corr_1].abs()<thresh_corr_3)

	# 			df_pre1.loc[id_score_query2,'group_correlation'] = 1
	# 			id_pre2_1 = (id_score_query3_1)&(~id_pred1)&(id_motif) # the peak loci without prediction and with motif and with peak-TF correlation below threshold
	# 			id_pre2_2 = (id_score_query3_2)&(~id_motif) # the peak loci without motif and with peak-TF correlation below threshold
								
	# 			list_query2 = [id_pre2_1,id_pre2_2]
	# 			list_query2_2 = []
	# 			column_corr_abs_1 = '%s_abs'%(column_corr_1)
	# 			column_corr_abs_1 = '%s_abs'%(column_corr_1)
	# 			for i2 in range(2):
	# 				id_query = list_query2[i2]
	# 				df_pre2 = df_pre1.loc[id_query,[column_corr_1,column_pval]].copy()
	# 				df_pre2[column_corr_abs_1] = df_pre2[column_corr_1].abs()
	# 				df_pre2 = df_pre2.sort_values(by=[column_corr_abs_1,column_pval],ascending=[True,False])
	# 				peak_query_pre2 = df_pre2.index
	# 				list_query2_2.append(peak_query_pre2)

	# 				peak_vec_2_1_ori, peak_vec_2_2_ori = list_query2_2[0:2]
	# 				peak_num_2_1_ori = len(peak_vec_2_1_ori)
	# 				peak_num_2_2_ori = len(peak_vec_2_2_ori)
	# 				print('peak_vec_2_1_ori, peak_vec_2_2_ori: ',peak_num_2_1_ori,peak_num_2_2_ori)

	# 			# peak_num_group2_1 = len(peak_loc_query_group2_1)
	# 			# print('peak_loc_query_group2_1: ',peak_num_group2_1)

	# 			# ratio_1, ratio_2 = 1.5, 1.0
	# 			# ratio_1, ratio_2 = 1.0, 1.0
	# 			# ratio_1, ratio_2 = 0.5, 1.0
	# 			ratio_1, ratio_2 = 0.5, 1.5
	# 			# ratio_1, ratio_2 = 0.25, 1.75
	# 			peak_num_2_1 = np.min([int(peak_num_group2_1*ratio_1),peak_num_2_1_ori])
	# 			peak_vec_2_1 = peak_vec_2_1_ori[0:peak_num_2_1]

	# 			peak_num_2_2 = np.min([int(peak_num_group2_1*ratio_2),peak_num_2_2_ori])
	# 			peak_vec_2_2 = peak_vec_2_2_ori[0:peak_num_2_2]

	# 		return peak_vec_2_1, peak_vec_2_2

	## select training sample from peak class 2
	def test_query_training_select_group2(self,data=[],motif_id_query='',peak_query_vec_1=[],feature_type_vec=[],peak_read=[],rna_exprs=[],save_mode=0,verbose=0,select_config={}):

		flag_select_2=1
		if flag_select_2>0:
			# column_vec_query_pre1, column_vec_query_pre2, column_vec_query_pre2_2 = self.test_query_column_method_1(feature_type_vec=feature_type_vec_query,select_config=select_config)
			column_motif_group = 'motif_group_1'
			column_peak_tf_corr_1 = 'group_correlation'
			column_motif_group_corr_1 = 'motif_group_correlation'

			method_type_feature_link = select_config['method_type_feature_link']
			column_pred2 = '%s.pred_sel'%(method_type_feature_link) # selected peak loci with predicted binding sites
			column_pred_2 = '%s.pred_group_2'%(method_type_feature_link)

			n_neighbors = select_config['neighbor_num']
			column_neighbor = ['neighbor%d'%(id1) for id1 in np.arange(1,n_neighbors+1)]

			feature_type_query_1, feature_type_query_2 = feature_type_vec[0:2]
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
			column_vec_pre2_1 = [column_pred_3,column_pred_6,column_pred_7]
			column_vec_pre2_2 = [column_pred_5,column_pred_6,column_pred_7] # the default column query
			column_vec_pre2_3 = [column_motif_group,column_pred_3,column_pred_6,column_pred_7]

			# peak_loc_ori = peak_read.columns
			# df_query1 = df_pre1.loc[peak_loc_ori,:]
			df_pre1 = data
			df_query1 = data
			peak_loc_ori = df_query1.index
			# df_pre1 = df_pre1.loc[peak_loc_ori,:]
			# df_query1 = df_query1.loc[peak_loc_ori,:]
			# column_motif = '%s.motif'%(method_type_feature_link)
			# column_pred1 = '%s.pred'%(method_type_feature_link)
			column_motif = select_config['column_motif']
			column_pred1 = select_config['column_pred1']

			if (column_motif!='-1'):
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
			# config_id_2 = select_config['config_id_2_query']
			config_id_2 = select_config['config_id_2']
			column_query_pre1 = column_vec_query_pre2
			
			print('config_id_2,motif_id_query: ',config_id_2,motif_id_query)
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
			# id_group = (df_pre1['motif_group_1']>0)

			# select peak with peak accessibility-TF expression correlation below threshold
			column_corr_1 = 'peak_tf_corr'
			column_pval = 'peak_tf_pval_corrected'
			thresh_corr_1, thresh_pval_1 = 0.30, 0.05
			thresh_corr_2, thresh_pval_2 = 0.1, 0.1
			thresh_corr_3, thresh_pval_2 = 0.05, 0.1
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

			id_pre2_1 = (id_score_query3_1)&(id_pre2)&(id_motif) # the peak loci without prediction and with motif and with peak-TF correlation below threshold
			id_pre2_2 = (id_score_query3_2)&(id_pre2)&(~id_motif) # the peak loci without motif and with peak-TF correlation below threshold
							
			list_query2 = [id_pre2_1,id_pre2_2]
			list_query2_2 = []
			column_corr_abs_1 = '%s_abs'%(column_corr_1)
			query_num = len(list_query2)
			for i2 in range(query_num):
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

			peak_query_vec = peak_query_vec_1
			peak_query_num_1 = len(peak_query_vec)

			# ratio_1, ratio_2 = 0.25, 2
			atio_1, ratio_2 = 0.25, 1.5
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

	## select training sample from peak class 2 with base model
	def test_query_training_select_group2_2(self,data=[],id_query=[],method_type_query='',flag_sample=1,flag_select_2=2,save_mode=1,verbose=0,select_config={}):

		method_type_feature_link = method_type_query
		if method_type_query=='':
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

		return peak_vec_2, peak_vec_2_1, peak_vec_2_2

	## parameter configuration
	def test_optimize_configure_1(self,model_type_id,Lasso_alpha=0.01,Ridge_alpha=1.0,l1_ratio=0.01,ElasticNet_alpha=1.0,select_config={}):

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
				# Ridge_alpha = 1E03
				# Ridge_alpha = 0.01
				if 'Ridge_alpha' in select_config:
					Ridge_alpha = select_config['Ridge_alpha']
				select_config1.update({'Ridge_alpha':Ridge_alpha})
				filename_annot2 = '%s'%(Ridge_alpha)
			else:
				filename_annot2 = '1'

			# filename_annot1 = '%s.%s'%(model_type_id1,filename_annot2)
			run_id = select_config['run_id']
			filename_annot1 = '%s.%d.%s.%d'%(model_type_id1,int(fit_intercept),filename_annot2,run_id)
			select_config1.update({'filename_annot1':filename_annot1})

		return select_config1

	# query feature association
	# df_feature_link: feature link mask
	def test_query_association_unit_pre1(self,feature_query_vec=[],df_feature_link=[],df_feature_link_2=[],dict_feature=[],feature_type_vec=['motif','peak'],column_idvec=['motif_id','peak_id'],feature_mtx=[],df_gene_annot_expr=[],
											peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],motif_data=[],model_type_vec=[],model_type_id='XGBClassifier',sample_idvec_train=[],type_id_model=0,num_class=2,num_fold=-1,link_type=0,parallel_mode=0,
											save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
		
		# input_file_path1 = self.save_path_1
		feature_type_query1, feature_type_query2 = feature_type_vec[0:2]
		df_feature_1 = dict_feature[feature_type_query1]
		df_feature_2 = dict_feature[feature_type_query2]
		print('df_feature_1, df_feature_2: ',df_feature_1.shape,df_feature_2.shape)

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
				data_vec_query, sample_idvec_query = self.test_train_idvec_pre1(sample_id,num_fold=num_fold,train_valid_mode=train_valid_mode_2,load=load_1,input_filename=input_filename_query1,save_mode=1,output_file_path=output_file_path,filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot,select_config=select_config)
				sample_idvec_train = sample_idvec_query

		sample_id_train, sample_id_valid, sample_id_test = sample_idvec_query[0:3]
		# x_train1 = df_feature_2.loc[sample_id_train,feature_query_2]
		x_train1 = df_feature_2.loc[sample_id_train,:]
		x_train1_ori = x_train1.copy()

		# sample_id = x_train1.index
		sample_id = df_feature_2.index
		feature_query_1 = feature_query_vec
		feature_query_2 = df_feature_2.columns

		y_mtx = df_feature_1.loc[sample_id_train,feature_query_1]
		print('x_train1 ',x_train1.shape)
		print('y_mtx ',y_mtx.shape)

		# df_feature_link_pre1 = df_feature_link
		# print('df_gene_peak_query_pre1, df_gene_annot_expr ',df_gene_peak_query_pre1.shape,df_gene_annot_expr.shape)
		motif_data = []

		if len(model_type_vec)==0:
			model_type_vec = ['LR','XGBClassifier','XGBR','Lasso',-1,'RF','ElasticNet','LogisticRegression']

		# model_type_id = 'XGBClassifier'
		model_type_id1 = model_type_id  # model_type_name
		model_type = model_type_id1
		print('model_type: ',model_type_id1,model_type)

		file_path1 = self.save_path_1
		run_id = select_config['run_id']
		select_config1 = dict()
		select_config1 = self.test_optimize_configure_1(model_type_id=model_type_id1,select_config=select_config)
		# print('select_config1: ',select_config1)

		max_depth, n_estimators = 7, 100
		# max_depth, n_estimators = 5, 100
		select_config_comp = {'max_depth':max_depth,'n_estimators':n_estimators}
		select_config1.update({'select_config_comp':select_config_comp})

		column_1 = 'multi_class_logisticregression'
		multi_class_query = 'auto'
		if column_1 in select_config:
			multi_class_query = select_config[column_1]
			select_config1.update({column_1:multi_class_query}) # copy the field query
		print('multi_class_logisticregression: ',multi_class_query)

		train_valid_mode_1, train_valid_mode_2 = 1, 0 # train_valid_mode_1:1, train on the combined data; train_valid_mode_2:0,only use train and test data; 1, use train,valid,and test data
		list_param = [train_valid_mode_1,train_valid_mode_2]
		
		field_query = ['train_valid_mode_1','train_valid_mode_2']
		query_num1 = len(list_param)
		from utility_1 import test_query_default_parameter_1
		select_config, list_param = test_query_default_parameter_1(field_query=field_query,default_parameter=list_param,overwrite=False,select_config=select_config)
		train_valid_mode_1,train_valid_mode_2 = list_param[0:2]
		print('train_valid_mode_1, train_valid_mode_2: ',train_valid_mode_1,train_valid_mode_2)

		select_config.update({'num_fold':num_fold,'select_config1':select_config1,
								'sample_idvec_train':sample_idvec_query})

		if train_valid_mode_1>0:
			response_variable_name = feature_query_1[0]
			response_query1 = response_variable_name
			y_train1 = y_mtx.loc[:,response_query1]
			x_train1_ori = x_train1
			# y_train1 = y_train
			model_type_id1 = model_type_id
			print('y_train1: ',np.unique(y_train1))
			
			pre_data_dict = dict()
			model_pre = train_pre1_1._Base2_train1(peak_read=peak_read,rna_exprs=rna_exprs,
													rna_exprs_unscaled=rna_exprs_unscaled,
													df_gene_peak_query=df_feature_link,
													df_gene_annot_expr=df_gene_annot_expr,
													motif_data=motif_data,
													select_config=select_config)

			sample_weight = []
			dict_query_1 = dict()
			save_mode_1 = 1
			save_model_train = 1
			# type_id_model = 1
			model_path_1 = select_config['model_path_1']
			dict_query_1, df_score_1 = model_pre.test_optimize_pre1_basic2_1(model_pre=[],x_train=x_train1,y_train=y_train1,
																				response_variable_name=response_query1,
																				x_train_feature2=[],
																				sample_weight=sample_weight,
																				dict_query=dict_query_1,
																				df_coef_query=[],
																				df_pred_query=[],
																				model_type_vec=model_type_vec,
																				model_type_idvec=[model_type_id1],
																				filename_annot_vec=[],
																				score_type_idvec=[],
																				pre_data_dict=pre_data_dict,
																				type_id_train=0,
																				type_id_model=type_id_model,num_class=num_class,
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
			print('y_pred: ',y_pred.shape)
			print(y_pred[0:2])

			# dict_query_1[model_type_id1].update({'model_combine':model_2,'df_score_2':df_score_2})	# prediction performance on the combined data
			model_train = dict1['model_combine']
			df_score_2 = dict1['df_score_2']

			x_test = df_feature_2.loc[sample_id_test,feature_query_2]
			y_test = df_feature_1.loc[sample_id_test,feature_query_1]

			print('x_test ',x_test.shape)
			print('y_test ',y_test.shape)

			y_test_pred = model_train.predict(x_test)
			y_test_proba = model_train.predict_proba(x_test)
			
			df_pred_2 = pd.DataFrame(index=sample_id_test,columns=[feature_query_1],data=np.asarray(y_test_pred)[:,np.newaxis])
			df_proba_2 = pd.DataFrame(index=sample_id_test,columns=np.arange(num_class),data=np.asarray(y_test_proba))

			return df_pred_2, df_proba_2

	# recompute based on training
	def test_query_compare_binding_train_unit1(self,data=[],peak_query_vec=[],peak_vec_1=[],motif_id_query='',dict_feature=[],feature_type_vec=[],sample_idvec_query=[],peak_read=[],rna_exprs=[],rna_exprs_unscaled=[],motif_data=[],flag_scale=1,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
			
			flag_query1 = 1
			if flag_query1>0:
				sample_id_valid = []
				peak_loc_ori = peak_query_vec
				sample_id_test = peak_loc_ori

				# sample_idvec_query = [sample_id_train,sample_id_valid,sample_id_test]
				# df_query_1 = df_pre1.loc[sample_id_train,:]
				df_pre1 = data
				df_pre1[motif_id_query] = 0
				peak_query_vec_pre1 = peak_vec_1.copy()
				df_pre1.loc[peak_vec_1,motif_id_query] = 1 # the selected peak loci with predicted TF binding

				# df_pre1.loc[peak_vec_2,motif_id_query] = 0
				peak_num1 = len(peak_vec_1)
				print('peak_vec_1: ',peak_num1)

				column_signal = 'signal'
				method_type_feature_link = select_config['method_type_feature_link']
				# column_motif = '%s.motif'%(method_type_feature_link)
				column_motif = select_config['column_motif']
				field_query_1 = [motif_id_query]
				if column_motif in df_pre1.columns:
					field_query_1 = [column_motif,motif_id_query]

				if column_signal in df_pre1.columns:
					field_query_1 = [column_signal] + field_query_1

				# if column_signal in df_pre1.columns:
				# 	# print(df_pre1.loc[peak_vec_1,['signal',column_motif,motif_id_query]])
				# 	print(df_pre1.loc[peak_vec_1,[column_signal,column_motif,motif_id_query]])
				# else:
				# 	print(df_pre1.loc[peak_vec_1,[column_motif,motif_id_query]])
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

				feature_type_query_vec_2 = np.asarray(list(dict_feature.keys()))
				feature_type_num1 = len(feature_type_query_vec_2)
				# file_path_query1 = select_config['file_path_query_1']
				file_path_query1 = select_config['file_path_save_link']
				
				print('file_path_query1: ',file_path_query1)
				dict_feature_query = dict_feature
				for i2 in range(feature_type_num1):
					# feature_type_query = feature_type_query_1
					# latent_mtx_query = latent_mtx_1
					feature_type_query = feature_type_query_vec_2[i2]
					# latent_mtx_query = list_pre1[i2]
					latent_mtx_query_ori = dict_feature[feature_type_query]
					if flag_scale>0:
						print('perform feature scaling')
						# z-score scaling
						scale_type = 2
						with_mean = True
						with_std = True
						latent_mtx_query = utility_1.test_motif_peak_estimate_score_scale_1(score=latent_mtx_query_ori,feature_query_vec=[],with_mean=with_mean,with_std=with_std,
																							select_config=select_config,
																							scale_type_id=scale_type)
					else:
						latent_mtx_query = latent_mtx_query_ori

					df_feature_2 = latent_mtx_query.loc[peak_loc_ori,:]
					df_feature_1 = df_pre1.loc[peak_loc_ori,feature_query_vec_1]

					feature_type_pre1, feature_type_pre2 = feature_type_vec_pre1[0:2]
					dict_feature_query = {feature_type_pre1:df_feature_1,feature_type_pre2:df_feature_2}
					print('df_feature_1: ',df_feature_1.shape,feature_type_pre1)
					print(df_feature_1[0:2])
					print('df_feature_2: ',df_feature_2.shape,feature_type_pre2,feature_type_query)
					print(df_feature_2[0:2])

					sample_idvec_train = sample_idvec_query
					run_id1 = 1
					select_config.update({'run_id':run_id1})
					feature_type_id1 = 0
					flag_model_explain = 1
					select_config.update({'feature_type_id':feature_type_id1,'flag_model_explain':flag_model_explain})

					filename_save_annot_2 = filename_save_annot
					filename_save_annot_local = '%s.%s_%s'%(filename_save_annot_2,feature_type_query,model_type_id1)
					data_path_save = file_path_query1
					select_config.update({'filename_save_annot_local':filename_save_annot_local,'data_path_save':file_path_query1})

					start = time.time()
					# classification model training
					model_type_id_train = 'LogisticRegression'
					type_id_model=1 # type_id_model: 0,regression model; 1, classification model
					select_config.update({'model_type_id_train':model_type_id_train,
											'model_type_id1':model_type_id_train,
											'type_id_model':type_id_model})

					df_pred_2, df_proba_2 = self.test_query_association_unit_pre1(feature_query_vec=feature_query_vec_1,df_feature_link=[],df_feature_link_2=[],
																					dict_feature=dict_feature_query,feature_type_vec=feature_type_vec_pre1,
																					column_idvec=column_vec_pre1,feature_mtx=[],df_gene_annot_expr=[],
																					peak_read=peak_read,rna_exprs=rna_exprs,rna_exprs_unscaled=rna_exprs_unscaled,
																					motif_data=motif_data,model_type_vec=[],model_type_id=model_type_id1,
																					sample_idvec_train=sample_idvec_train,
																					type_id_model=type_id_model,
																					num_class=num_class,num_fold=num_fold,link_type=0,parallel_mode=0,
																					save_mode=1,output_file_path=output_file_path,output_filename='',filename_prefix_save=filename_prefix_save,filename_save_annot=filename_save_annot_2,
																					verbose=verbose,select_config=select_config)
					
					stop = time.time()
					print('model training used %.2fs'%(stop-start),feature_type_query)

					print('df_pred_2: ',df_pred_2.shape,feature_type_query)
					print(df_pred_2[0:2])

					print('df_proba_2: ',df_proba_2.shape,feature_type_query)
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

				# df_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				# peak_loc_1 = df_1.index
				if column_signal in df_pre1.columns:
					# df_pre1 = df_pre1.sort_values(by=['signal',column_motif],ascending=False)
					column_vec_sort = [column_signal,column_motif]
				else:
					column_vec_sort = [column_motif]
					
				df_pre1 = df_pre1.sort_values(by=column_vec_sort,ascending=False)
				if (save_mode>0) and (output_filename!=''):
					column_vec = df_pre1.columns
					t_columns = pd.Index(column_vec).difference(['peak_id'],sort=False)
					df_pre1 = df_pre1.loc[:,t_columns]
					df_pre1.to_csv(output_filename,sep='\t')

				return df_pre1

	# query the groups of peak query
	# prepare for the multiclass label
	def test_query_compare_binding_overlap_1(self,data=[],motif_id_query='',motif_id1='',feature_type_vec=[],df_overlap_query=[],thresh_vec=[],column_vec_query=[],input_filename='',input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		if len(df_overlap_query)==0:
			if input_filename=='':
				input_file_path_query = input_file_path
				data_file_type_query = select_config['data_file_type']
				input_filename = '%s/test_query_df_overlap.%s.%s.pre1.1.txt'%(input_file_path_query,motif_id1,data_file_type_query)
			
			df1 = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_overlap_query = df1

		df_pre1 = data
		id1 = (df_pre1['class']>0)
		peak_loc_pre1 = df_pre1.index
		peak_vec_1 = peak_loc_pre1[id1]
		peak_num1 = len(peak_vec_1)
		print('peak_vec_1: ',peak_num1)

		column_1 = 'group_id2'
		feature_type_vec_query = feature_type_vec
		if len(feature_type_vec)==0:
			feature_type_vec_query = ['latent_peak_motif','latent_peak_tf']

		feature_type_query1, feature_type_query2 = feature_type_vec_query[0:2]
		column_query1, column_query2 = '%s_group'%(feature_type_query1), '%s_group'%(feature_type_query2)

		if not (column_1 in df_pre1.columns):
			print('the column not included: ',column_1)
			# df_pre1[column_1] = utility_1.test_query_index(df_pre1,column_vec=[column_query1,column_query2],symbol_vec=['_'])
			df_pre1[column_1] = ['%d_%d'%(query_id1,query_id2) for (query_id1,query_id2) in zip(df_pre1[column_query1],df_pre1[column_query2])]

		print(df_pre1[column_1][0:10])

		df_pre2 = df_pre1.loc[id1,:]
		# group_idvec = df_pre1.loc[id1,column_1].unique()
		group_idvec = df_pre2[column_1].unique()	# the paired group of the selected peak loci
		group_vec_num1 = len(group_idvec)
		print('df_pre1, df_pre2: ',df_pre1.shape,df_pre2.shape)
		print('group_idvec: ',group_vec_num1)
		print(group_idvec[0:10])

		# query the paired group of the selected peak loci
		# group_idvec_1 = df_overlap_query.index
		df_overlap_query2 = df_overlap_query.loc[group_idvec,:]
		df_overlap_query2 = df_overlap_query2.sort_values(by=['overlap','pval_fisher_exact_'],ascending=[False,True])
		output_filename = '%s/test_query_df_overlap.%s.1_1.pre2.txt'%(output_file_path,motif_id1)
		df_overlap_query2.to_csv(output_filename,sep='\t')
		group_idvec_2 = df_overlap_query2.index

		if len(thresh_vec)==0:
			thresh_overlap_default_1 = 10
			thresh_pval_group = 0.1
			# thresh_ratio_1 = 0.20
			thresh_ratio_1 = 0.05
			thresh_group_num = 20
			thresh_vec = [thresh_overlap_default_1,thresh_pval_group,thresh_ratio_1,thresh_group_num]
		else:
			thresh_overlap_default_1, thresh_pval_group, thresh_ratio_1 = thresh_vec[0:3]
			thresh_group_num = thresh_vec[3]
		
		# thresh_ratio_1 = 0.20
		thresh_num_default_1 = 100
		thresh_num_1 = peak_num1*thresh_ratio_1
		thresh_num1 = np.max([thresh_num_default_1,thresh_num_1])
		print('peak_num1, thresh_ratio_1, thresh_num_1, thresh_num1',peak_num1,thresh_ratio_1,thresh_num_1,thresh_num1)

		if len(column_vec_query)==0:
			column_pval_group = 'pval_fisher_exact_'
			column_overlap = 'overlap'
			column_vec_query = [column_overlap,column_pval_group]
		else:
			column_overlap,column_pval_group = column_vec_query[0:2]

		id1 = (df_overlap_query2[column_overlap]>thresh_overlap_default_1)
		id2 = (df_overlap_query2[column_pval_group]<thresh_pval_group)
		id_1 = (id1&id2)

		# id1 = (df_overlap_query[column_overlap]>thresh_overlap_default_1)
		group_vec_overlap = group_idvec_2[id_1]

		df_pre2['count'] = 1
		df_group_query_1 = df_pre2.groupby(by=column_1)
		df_group_query = df_group_query_1[['count']].sum()
		df_group_query = df_group_query.sort_values(by=['count'],ascending=False)
		id_2 = (df_group_query['count']>thresh_num1)

		output_filename = '%s/test_query_df_overlap.%s.1_2.pre2.txt'%(output_file_path,motif_id1)
		df_group_query.to_csv(output_filename,sep='\t')

		group_idvec_pre2 = df_group_query.index
		group_vec_2 = group_idvec_pre2[id_2]	# the group with member number above threshold

		print('group_vec_2: ',len(group_vec_2),group_vec_2)
		print('group_vec_overlap: ',len(group_vec_overlap),group_vec_overlap)
		group_vec_pre2 = pd.Index(group_vec_2).intersection(group_vec_overlap,sort=False)
		group_num2 =len(group_vec_pre2)
		print('group_vec_pre2: ',group_num2)
		print(group_vec_pre2)

		thresh_group_num = 9
		if group_num2>thresh_group_num:
			group_vec_pre2_ori = group_vec_pre2.copy()
			group_vec_pre2 = group_vec_pre2_ori[0:thresh_group_num]

			group_num2 =len(group_vec_pre2)
			print('group_vec_pre2: ',group_num2)

		column_query_group = column_1
		# label_vec = np.arange(1,group_num2+1)
		column_2 = motif_id_query
		df_pre1[column_2] = 0
		df_pre2['peak_id'] = np.asarray(df_pre2.index).copy()
		df_pre2.index = np.asarray(df_pre2[column_query_group])
		for i2 in range(group_num2):
			group_id2 = group_vec_pre2[i2]
			id1 = df_pre2[column_query_group].isin([group_id2])
			peak_query = df_pre2.loc[group_id2,'peak_id'].unique()
			# peak_query = df_pre2.loc[id1,'peak_id'].unique()
			df_pre1.loc[peak_query,column_2] = (i2+1)

		group_vec_pre2_2 = pd.Index(group_idvec_2).difference(group_vec_pre2,sort=False)
		peak_query_2 = df_pre2.loc[group_vec_pre2_2,'peak_id'].unique()
		df_pre1.loc[peak_query_2,column_2] = (group_num2+1)

		return df_pre1
	
	## load data and query configuration parameters
	def test_query_load_pre1_ori(self,data=[],method_type_vec_query=[],flag_config_1=1,flag_motif_data_load_1=1,flag_load_1=1,save_mode=1,verbose=0,select_config={}):

		# flag_config_1=1
		if flag_config_1>0:
			root_path_1 = select_config['root_path_1']
			root_path_2 = select_config['root_path_2']

			data_file_type_query = select_config['data_file_type']
			method_type_feature_link = select_config['method_type_feature_link']
			# method_type_vec = ['insilico_0.1']+[method_type_feature_link]
			# method_type_vec = list(pd.Index(method_type_vec).unique())
			method_type_vec = [method_type_feature_link]
			select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)

		# flag_motif_data_load_1 = 1
		if flag_motif_data_load_1>0:
			print('load motif data')
			if len(method_type_vec_query)==0:
				# method_type_vec_query = ['insilico_0.1']+[method_type_feature_link]
				# method_type_vec_query = list(pd.Index(method_type_vec_query).unique())
				method_type_vec_query = [method_type_feature_link]
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																			select_config=select_config)

			self.dict_motif_data = dict_motif_data

		# flag_load_1 = 1
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori(meta_exprs=[],peak_read=[],flag_format=False,select_config=select_config)

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
			# peak_loc_ori = peak_read.columns

		return select_config

	## load data and query configuration parameters
	# load motif data; load ATAC-seq and RNA-seq data of the metacells
	def test_query_load_pre1(self,data=[],method_type_vec_query=[],flag_config_1=1,flag_motif_data_load_1=1,flag_load_1=1,input_file_path='',save_mode=1,verbose=0,select_config={}):

		# flag_config_1=1
		# if flag_config_1>0:
		# 	root_path_1 = select_config['root_path_1']
		# 	root_path_2 = select_config['root_path_2']

		# 	data_file_type_query = select_config['data_file_type']
		# 	method_type_feature_link = select_config['method_type_feature_link']
		# 	# method_type_vec = ['insilico_0.1']+[method_type_feature_link]
		# 	# method_type_vec = list(pd.Index(method_type_vec).unique())
		# 	method_type_vec = [method_type_feature_link]
		# 	select_config = self.test_query_config_pre1_1(data_file_type_query=data_file_type_query,method_type_vec=method_type_vec,flag_config_1=flag_config_1,select_config=select_config)

		# flag_motif_data_load_1 = 1
		# load motif data
		method_type_feature_link = select_config['method_type_feature_link']
		if flag_motif_data_load_1>0:
			print('load motif data')
			if len(method_type_vec_query)==0:
				# method_type_vec_query = ['insilico_0.1']+[method_type_feature_link]
				# method_type_vec_query = list(pd.Index(method_type_vec_query).unique())
				method_type_vec_query = [method_type_feature_link]

			# load motif data
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																			select_config=select_config)

			self.dict_motif_data = dict_motif_data

		# flag_load_1 = 1
		# load the ATAC-seq data and RNA-seq data of the metacells
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			# print('load ATAC-seq and RNA-seq count matrices of the metacells')
			start = time.time()
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori(meta_exprs=[],peak_read=[],flag_format=False,select_config=select_config)

			# sample_id = meta_scaled_exprs.index
			# peak_read = peak_read.loc[sample_id,:]
			sample_id = peak_read.index
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			if len(meta_scaled_exprs)>0:
				meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
				rna_exprs = meta_scaled_exprs	# scaled RNA-seq data
			else:
				rna_exprs = meta_exprs_2	# unscaled RNA-seq data
			# print('peak_read, rna_exprs: ',peak_read.shape,rna_exprs.shape)

			print('ATAC-seq count matrx: ',peak_read.shape)
			print(peak_read[0:2])

			print('RNA-seq count matrx: ',rna_exprs.shape)
			print(rna_exprs[0:2])

			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs
			# peak_loc_ori = peak_read.columns

			stop = time.time()
			print('load peak accessiblity and gene expression data used %.2fs'%(stop-start))
			
		return select_config

	## file annotation query
	def test_query_file_annotation_1(self,data=[],input_filename='',method_type_feature_link='',load_mode=0,save_mode=0,verbose=0,select_config={}):

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
			input_file_path_query = select_config['file_path_motif_score']

			df_score_annot = []
			if len(column_vec_annot)>0:
				print('column_vec_annot: ',column_vec_annot)
				input_filename_1 = input_filename
				if os.path.exists(input_filename_1)==True:
					df_score_annot = pd.read_csv(input_filename_1,index_col=False,sep='\t')
				else:
					column_1 = 'filename_annot_list'
					if column_1 in select_config:
						input_filename_list = select_config[column_1]
						file_num = len(input_filename_list)
						list1 = []
						for i1 in range(file_num):
							input_filename = input_filename_list[i1]
							df_1 = pd.read_csv(input_filename,index_col=False,sep='\t')
							list1.append(df_1)
							if (iter_id%10==0):
								print('df_1: ',df_1.shape)
								print(input_filename)

						df_score_annot = pd.concat(list1,axis=0,join='outer',ignore_index=False)
						
						print('df_score_annot: ',df_score_annot.shape)
						print(df_score_annot.columns)
						print(df_score_annot[0:2])

				if len(df_score_annot)==0:
					print('please provide score annotation file')
					return

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
					# output_filename = input_filename_query_1[0:b]+'.copy2_1.txt.gz'
					output_filename = input_filename_query_1[0:b]+'.copy1.txt.gz'
					df_gene_peak_query_1_ori.to_csv(output_filename,index=False,sep='\t',compression=compression)

			if load_mode>0:
				return df_score_annot, df_gene_peak_query_1_ori
			else:
				return df_score_annot

	## query feature link
	def test_query_feature_link_pre1_1(self,method_type='',method_type_vec=[],dict_method_type=[],dict_file_query=[],dict_feature_query=[],df_score_annot=[],thresh_query_vec=[],input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			# estimation 2
			data_file_type_query = select_config['data_file_type']

			self.dict_file_query = dict_file_query
			compression = 'gzip'
			list_pre2 = []
			if len(method_type_vec)==0:
				key_vec = list(dict_method_type.keys())
				method_type_vec = np.asarray(key_vec)

			for method_type_query1 in method_type_vec:
				if (method_type_query1.find('insilico')>-1):
					method_type_query1_1 = 'insilico'

				elif (method_type_query1.find('joint_score_pre1')>-1):
					method_type_query1_1 = 'joint_score_pre1'

				elif (method_type_query1.find('joint_score_pre2')>-1):
					method_type_query1_1 = 'joint_score_pre2'

				input_filename = dict_file_query[method_type_query1_1]
				method_type = method_type_query1_1
				try:
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t',compression=compression) # load feature link
				except Exception as error:
					print('error! ',error)
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t')

				print('df_query: ',df_query.shape)
				print(input_filename)
				print(df_query.columns)
				print(df_query[0:2])

				df_link_query_1 = df_query
				thresh_query_vec = []
				dict_thresh_score = []
				print('method_type: ',method_type)

				if method_type_query1 in dict_method_type:
					dict1 = dict_method_type[method_type_query1]
					thresh_query_vec = dict1['thresh_query_vec']
					if 'dict_thresh_score' in dict1:
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
						# the p-value threshold for peak_tf_corr, peak_gene_corr_, gene_tf_corr_peak, gene_tf_corr and stricter threshold for peak_tf_corr
						thresh_pval_vec = [0.1,0.1,0.25,0.1,0.01]
					
					type_query = 0
					# type_query = 1
					overwrite_2 = True

					# perform feature link selection using thresholds
					thresh_query_vec_pre1 = list(np.arange(5))+[5,21]
					dict_feature_link_1 = self.test_query_feature_link_select_pre2(df_feature_link=df_link_query_1,df_score_annot=df_score_annot,column_score_vec=column_score_vec,
																						thresh_query_vec=thresh_query_vec_pre1,thresh_score_vec=thresh_score_vec,
																						thresh_score_vec_2=thresh_score_vec_2,thresh_pval_vec=thresh_pval_vec,overwrite=overwrite_2,
																						save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',
																						verbose=verbose,select_config=select_config)
					# thresh_query_vec = thresh_query_vec_pre1+[-1]
					# thresh_query_vec = [2,5,21]
					thresh_query_vec = [2,21]

					# list2 = []
					column_idvec = ['gene_id','peak_id','motif_id']
					column_id1, column_id2, column_id3 = column_idvec[0:3]
					for thresh_query in thresh_query_vec:
						df_link_query = dict_feature_link_1[thresh_query]
						method_type_query = '%s.thresh%d'%(method_type,thresh_query+1)
						print('df_link_query, method_type_query: ',df_link_query.shape,method_type_query)

						df_gene_peak_query = df_link_query
						if ('gene_id' in df_gene_peak_query):
							df_query = df_gene_peak_query.copy()
							column_score_1 = 'score_pred1'
							df_query = df_query.sort_values(by=[column_score_1],ascending=False)
							# df_query.index = test_query_index(df_query,column_vec=['peak_id','motif_id'])
							# df_peak_tf_query = df_query.loc[~df_query.index.duplicated(keep='first'),:]
							df_peak_tf_query = df_query.drop_duplicates(subset=[column_id2,column_id3])
							df_peak_tf_query.index = np.asarray(df_peak_tf_query[column_id3])
							
							print('df_peak_tf_query ',df_peak_tf_query.shape,method_type)
							print(df_peak_tf_query[0:5])

						dict1 = {'peak_tf_gene':df_gene_peak_query,'peak_tf':df_peak_tf_query}
						dict_feature_query.update({method_type_query:dict1})

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

			dict_query1 = dict()
			dict_annot1 = dict()
			
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

	## load feature link
	def test_query_feature_link_load_pre1(self,data=[],dict_file_query=[],save_mode=1,verbose=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',select_config={}):

		load_mode_query1 = 0
		data_file_type_query = select_config['data_file_type']
		dict_file_query = self.test_query_file_path_1(data_file_type=data_file_type_query,save_mode=1,verbose=0,select_config=select_config)

		self.dict_file_query = dict_file_query
		select_config.update({'dict_file_query':dict_file_query})
		print('dict_file_query: ',dict_file_query)

		dict_feature_query = dict()

		dict_method_type = dict()
		method_type_feature_link_query = 'insilico'
		thresh_value_vec = [0.1,0.15]
		dict_thresh_score_1 = dict()
		dict1 = {'thresh_query_vec':thresh_value_vec,'dict_thresh_score':dict_thresh_score_1}
		dict_method_type.update({method_type_feature_link_query:dict1})

		method_type_feature_link = select_config['method_type_feature_link']
		method_type_feature_link_1 = method_type_feature_link
		# thresh_query_vec = np.arange(5)
		thresh_query_vec = list(np.arange(5))+[5,21]
		# use the default parameter
		thresh_score_vec_2 = [0.05,0.15,0.10]
		# thresh_score_vec = [[0.10,0.05],[0.10,0.05]]
		thresh_score_vec = [[0.10,0],[0.10,0]]
		# the p-value threshold for peak_tf_corr, peak_gene_corr_, gene_tf_corr_peak, gene_tf_corr and stricter threshold for peak_tf_corr
		thresh_pval_vec = [0.1,0.1,0.25,0.1,0.01]
		dict_thresh_score = {'thresh_score_vec':thresh_score_vec,'thresh_score_vec_2':thresh_score_vec_2,'thresh_pval_vec':thresh_pval_vec}
		dict2 = {'thresh_query_vec':thresh_query_vec,'dict_thresh_score':dict_thresh_score}
		dict_method_type.update({method_type_feature_link_1:dict2})

		if 'method_type_vec_link' in select_config:
			method_type_vec = select_config['method_type_vec_link']
		else:
			method_type_vec = [method_type_feature_link_1,method_type_feature_link_query]

		df_score_annot = self.test_query_file_annotation_1(data=dict_file_query,method_type_feature_link=method_type_feature_link_1,
															load_mode=load_mode_query1,save_mode=0,verbose=0,select_config=select_config)
		
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

		return dict_feature_query

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

def run(chromosome,run_id,species,cell,generate,chromvec,testchromvec,metacell_num,peak_distance_thresh,
			highly_variable,upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
			method_type_group,thresh_size_group,thresh_score_group_1,method_type_feature_link,neighbor_num,model_type_id,typeid2,folder_id,
			config_id_2,config_group_annot,ratio_1,ratio_2,flag_group,train_id1,flag_scale_1,beta_mode,motif_id_1,query_id1,query_id2,query_id_1,query_id_2,train_mode,config_id_load):
	
	file_path_1 = '.'
	test_estimator1 = _Base2_2(file_path=file_path_1)

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







