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
# from test_reunion_compute_pre2 import _Base2_2
from test_rediscover_compute_1 import _Base2_2
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

class _Base2_2_1(_Base2_2):
	"""Base class for peak-TF association estimation
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

		_Base2_2.__init__(self,file_path=file_path,
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
			# run_id = select_config['run_id']
			type_id_feature = select_config['type_id_feature']
			filename_save_annot_1 = '%s.%d.%d'%(data_file_type,type_id_feature,run_id)

			select_config_query = {'data_path':input_file_path,
									'filename_save_annot_1':filename_save_annot_1,
									'filename_save_annot_pre1':filename_save_annot_1}

			return select_config_query

	## load the ChIP-seq data annotation file
	def test_query_file_annotation_load_1(self,data_file_type_query='',input_filename='',folder_id=1,save_mode=1,verbose=0,select_config={}):

		flag_query1=1
		if flag_query1>0:
			# path_id = select_config['path_id']
			if data_file_type_query=='':
				data_file_type_query=select_config['data_file_type']

			input_file_path1 = self.save_path_1
			data_file_type = data_file_type_query
			# data_file_type_annot = select_config['data_file_type_annot']	
			# input_filename = '%s/test_peak_file.ChIP-seq.%s.group%d.txt'%(input_file_path_query2,data_file_type_query,group_id_1)
			df_peak_file = pd.read_csv(input_filename,index_col=0,sep='\t')
			if 'motif_id' in df_peak_file.columns:
				motif_idvec_query = np.asarray(df_peak_file['motif_id'])
			else:
				motif_idvec_query = df_peak_file.index

			return df_peak_file, motif_idvec_query

	# compare TF binding prediction
	# the prediction performance
	def test_query_compare_binding_pre1_5_1_unit1(self,data=[],input_filename='',method_type_query='',method_type_vec=[],feature_type_vec=[],column_vec_query=[],iter_num=10,type_id_query=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config={}):
		
		file_path_1 = self.save_path_1
		data_file_type_query = select_config['data_file_type']
		
		if len(data)==0:
			if os.path.exists(input_filename)==False:
				print('the file does not exist: %s'%(input_filename))
				return 
			else:
				df_1 = pd.read_csv(input_filename,index_col=0,sep='\t') # the peak-TF association prediction
		else:
			df_1 = data # the peak-TF association prediction

		column_motif = '%s.motif'%(method_type_query)
		column_pred1 = '%s.pred'%(method_type_query)
		column_score_query1 = '%s.score'%(method_type_query)
		
		type_id_pre1 = type_id_query
		# type_id_pre1 = 0

		# df_1 = data # the peak-TF association prediction
		id_signal_ori = (df_1['signal']>0)
		try:
			id_motif = (df_1[column_motif]>0)
		except Exception as error:
			print('error! ',error)
			id_motif = (df_1[column_motif].isin(['True',True,1,'1']))

		peak_loc_1 = df_1.index
		peak_loc_num1 = len(peak_loc_1)
		peak_loc_motif = peak_loc_1[id_motif]
		peak_motif_num = len(peak_loc_motif)
		print('peak_loc_1, peak with motif: ',peak_loc_num1,peak_motif_num)

		# id_motif_2_ori = (df_1[column_pred2]>0)
		# id_query_1 = (~id_motif)&(id_motif_2_ori)

		df_2 = df_1.loc[(~id_motif),:] # the peak loci without motif
		peak_loc_2 = df_2.index
		peak_loc_num2 = len(peak_loc_2)
		print('peak_loc_2, peak without motif: ',peak_loc_num2)

		if len(column_vec_query)==0:
			# column_vec_query = list_pre1
			column_vec_query = list_pre1_ori + list_pre2_ori
			column_vec_query = pd.Index(column_vec_query).intersection(df_2.columns,sort=False)
			column_vec_2 = pd.Index(column_vec_query).difference(df_2.columns,sort=False)
			print('the columns not included: ',len(column_vec_2))
			
		df_score_query1_2, df_score_query2_2, dict_query1_2 = self.test_query_compare_binding_pre1_5_1_unit1_2(data=df_2,column_vec=column_vec_query,iter_num=iter_num,save_mode=1,verbose=verbose,select_config=select_config)
		df_score_query1_2['group_motif'] = 0
		df_score_query2_2['group_motif'] = 0

		df_pre2 = df_1.loc[(id_motif),:] # the peak loci with motif
		peak_loc_pre2 = df_2.index
		peak_loc_num_2 = len(peak_loc_pre2)
		print('peak_loc_pre2, peak with motif: ',peak_loc_num_2)
		df_score_query1_1, df_score_query2_1, dict_query1_1 = self.test_query_compare_binding_pre1_5_1_unit1_2(data=df_pre2,column_vec=column_vec_query,iter_num=iter_num,save_mode=1,verbose=verbose,select_config=select_config)
		df_score_query1_1['group_motif'] = 1
		df_score_query2_1['group_motif'] = 1
		dict_query_1 = {'group_1':dict_query1_1,'group_2':dict_query1_2}

		# return df_1, df_score_query_1, df_score_query_2, dict_query_1
		return df_1, df_score_query1_1, df_score_query1_2, dict_query_1

	## compare TF binding prediction
	# the prediction performance of predicted feature links and background feature links
	def test_query_compare_binding_pre1_5_1_unit1_2(self,data=[],column_vec=[],iter_num=5,iter_mode=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config={}):
		
		df_2 = data
		peak_loc_2 = df_2.index
		peak_loc_num2 = len(peak_loc_2)
		# print('peak_loc_2, peak without motif: ',peak_loc_num2)
		print('peak_loc_2: ',peak_loc_num2)

		field_query_2 = ['signal','signal_0','motif_2','motif_2_0',
							'signal_motif_2','signal_motif_2_0','signal_0_motif_2','signal_0_motif_2_0']
		
		# query_num = len(list_pre1)
		query_num = len(column_vec)
		list_query1 = []
		list_query2 = []
		dict_query_1 = dict()
		for i2 in range(query_num):
			# column_vec_query = ['signal',column_pred2]
			try:
				# column_query1 = list_pre1[i2]
				column_query1 = column_vec[i2]
				column_vec_query = ['signal',column_query1]
				iter_mode=0
				df_score_query1, df_score_query2, contingency_table, dict_query1 = self.test_query_compare_binding_pre1_5_1_unit2(data=df_2,column_vec=column_vec_query,iter_mode=iter_mode,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=verbose,select_config=select_config)

				dict_query_1.update({column_query1:dict_query1})

				list1 = []
				list2 = []
				list1.append(df_score_query1)
				list2.append(df_score_query2)
				iter_vec_1 = list(np.arange(iter_num))
				# iter_id = 0
				id_motif_2 = (df_2[column_query1]>0)
				peak_motif_group1 = peak_loc_2[id_motif_2]
				peak_motif_num1 = len(peak_motif_group1)
				print('peak_motif_group1: ',peak_motif_num1,column_query1)
				print('contingency table: ')
				print(contingency_table)

				np.random.seed(0)
				for iter_id in range(iter_num):
					column_pred2_2 = '%s_group2.%d'%(column_query1,iter_id)
					id_1 = np.random.permutation(peak_loc_num2)
					id_motif_group2 = np.asarray(id_motif_2)[id_1]
					
					df_2[column_pred2_2] = id_motif_group2
					column_vec_query_2 = ['signal',column_pred2_2]
					iter_mode = 1
					df_score_query1_group2, df_score_query2_group2, contingency_table_group2, dict_query_group2 = self.test_query_compare_binding_pre1_5_1_unit2(data=df_2,column_vec=column_vec_query_2,iter_mode=iter_mode,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=verbose,select_config=select_config)
					
					list1.append(df_score_query1_group2)
					list2.append(df_score_query2_group2)
					if iter_id==0:
						print('contingency table, group2: ',column_query1,iter_id)
						print(contingency_table)

				df_score_1 = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				df_score_2 = pd.concat(list2,axis=1,join='outer',ignore_index=False)
				# query_id1 = 'motif_2'
				query_id1 = column_query1
				query_vec_1 = [query_id1] + iter_vec_1
				df_score_1 = df_score_1.T
				df_score_1.index = query_vec_1
				df_score_1 = df_score_1.round(6)
				df_score_1['group_1'] = column_query1
				query_value_1 = df_score_1.loc[query_id1,:]
				mean_value_1 = df_score_1.loc[iter_vec_1,:].mean(axis=0)
				
				df_score_2 = df_score_2.T
				df_score_2.index = query_vec_1
				df_score_2 = df_score_2.round(6)
				df_score_2['group_1'] = column_query1
				query_value_2 = df_score_2.loc[query_id1,:]
				mean_value_2 = df_score_2.loc[iter_vec_1,:].mean(axis=0)

				print('query_value_1: ',query_value_1,column_query1,i2)
				print('mean_value_1: ',mean_value_1,column_query1,i2)
				print('query_value_2: ',query_value_2,column_query1,i2)
				print('mean_value_2: ',mean_value_2,column_query1,i2)

				list_query1.append(df_score_1)
				list_query2.append(df_score_2)
			except Exception as error:
				print('error! ',error)

		if len(list_query1)>0:
			df_score_query_1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)
			df_score_query_2 = pd.concat(list_query2,axis=0,join='outer',ignore_index=False)
		else:
			df_score_query_1 = []
			df_score_query_2 = []

		return df_score_query_1, df_score_query_2, dict_query_1

	## compare TF binding prediction
	def test_query_compare_binding_pre1_5_1_unit2(self,data=[],column_vec=[],mode_query=1,iter_mode=0,plot_ax=[],save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config={}):
		
		file_path_1 = self.save_path_1
		t_vec_1 = self.test_query_pred_score_unit1_1(data=data,column_vec=column_vec,mode_query=mode_query,iter_mode=iter_mode,plot_ax=plot_ax,
														save_mode=save_mode,output_file_path=output_file_path,filename_prefix_save=filename_prefix_save,filename_annot_save=filename_annot_save,output_filename=output_filename,verbose=verbose,select_config=select_config)

		df_score_query_1, df_score_query_2, contingency_table, dict_query1 = t_vec_1

		return df_score_query_1, df_score_query_2, contingency_table, dict_query1

	## score query for performance comparison
	# performance comparison for the binary prediction and predicted probability
	def test_query_compare_binding_pre1_5_1_basic_2_unit1_1(self,data=[],dict_feature=[],motif_id_query='',motif_id_1='',motif_id_2='',column_signal='signal',column_motif='',column_vec_query=[],feature_type_vec=[],method_type_vec=[],method_type_group='',flag_compare_1=0,type_id_1=0,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_motif_ori = 0
		# TF binding prediction based on motif scanning with sequence feature
		# column_signal = 'signal'
		df_pre1 = data
		
		# id3 = (df_pre1[column_pred2]>0)
		id1 = (df_pre1[column_motif].abs()>0) # peak loci with motif
		id2 = (~id1)	# peak loci without motif
		df_query_group1 = df_pre1.loc[id1,:]
		df_query_group2 = df_pre1.loc[id2,:]
		df_query_group_1 = df_pre1
		print('df_pre1, df_query_group1, df_query_group2: ',df_pre1.shape,df_query_group1.shape,df_query_group2.shape)

		# list_group = [df_query_group1,df_query_group2]
		list_group = [df_query_group_1,df_query_group1,df_query_group2]
		query_num_1 = len(list_group)
		query_num_2 = len(column_vec_query)
		n_neighbors = select_config['neighbor_num']

		list_query1 = []
		motif_id1 = motif_id_1
		motif_id2 = motif_id_2
		group_query_vec = [2,1,0]
		for i2 in range(query_num_1):
			df_query_group = list_group[i2]
			list_query2 = []
			list_query3 = []
			for t_id1 in range(query_num_2):
				column_query_1 = column_vec_query[t_id1]
				if isinstance(column_query_1,str):
					column_pred = column_query_1
					column_proba = []
				else:
					column_pred, column_proba = column_query_1[0:2]

				column_vec = [column_signal,column_pred,column_proba]
				t_vec_1 = self.test_query_pred_score_1(data=df_query_group,column_vec=column_vec,iter_mode=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config=select_config)
								
				df_score_1, df_score_2, contingency_table, dict_query1 = t_vec_1
				# print('field_query: ',column_query)
				# print('df_score_1: \n',df_score_1)
				# print('contingency_table: \n',contingency_table)
				list_query2.append(df_score_1)
				# list_query3.append(column_query)
				list_query3.append(column_pred)

			if len(list_query2)>0:
				df_score_query = pd.concat(list_query2,axis=1,join='outer',ignore_index=False)
				df_score_query = df_score_query.T
				column_vec_query_1 = np.asarray(list_query3)

				field_query_2 = ['motif_id','motif_id1','motif_id2','group_motif','neighbor_num','method_type','method_type_group']
				# group_id_motif = int(2-i2)
				group_id_motif = group_query_vec[i2]
				list_2 = [motif_id_query,motif_id1,motif_id2,group_id_motif,n_neighbors,column_vec_query_1,method_type_group]
				for (field_id1,query_value) in zip(field_query_2,list_2):
					df_score_query[field_id1] = query_value
				list_query1.append(df_score_query)

		if len(list_query1)>0:
			df_score_query_1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)

		return df_score_query_1

	## score query for performance comparison
	# performance comparison for the binary prediction
	def test_query_compare_binding_pre1_5_1_basic_2_unit1(self,data=[],dict_feature=[],motif_id_query='',motif_id_1='',motif_id_2='',column_signal='signal',column_motif='',column_vec_query=[],feature_type_vec=[],method_type_vec=[],method_type_group='',flag_compare_1=0,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		flag_motif_ori = 0
		# TF binding prediction based on motif scanning with sequence feature
		# column_signal = 'signal'
		df_pre1 = data
		if column_motif=='':
			method_type_feature_link = select_config['method_type_feature_link']
			column_motif = '%s.motif'%(method_type_feature_link)
		
		# id3 = (df_pre1[column_pred2]>0)
		id1 = (df_pre1[column_motif].abs()>0) # peak loci with motif
		id2 = (~id1)	# peak loci without motif
		df_query_group1 = df_pre1.loc[id1,:]
		df_query_group2 = df_pre1.loc[id2,:]
		df_query_group_1 = df_pre1
		print('df_pre1, df_query_group1, df_query_group2: ',df_pre1.shape,df_query_group1.shape,df_query_group2.shape)

		# list_group = [df_query_group1,df_query_group2]
		list_group = [df_query_group_1,df_query_group1,df_query_group2]
		query_num_1 = len(list_group)
		query_num_2 = len(column_vec_query)
		n_neighbors = select_config['neighbor_num']

		list_query1 = []
		motif_id1 = motif_id_1
		motif_id2 = motif_id_2
		for i2 in range(query_num_1):
			df_query_group = list_group[i2]
			list_query2 = []
			list_query3 = []

			for t_id1 in range(query_num_2):
				column_query = column_vec_query[t_id1]
				column_vec = [column_signal,column_query]
				t_vec_1 = self.test_query_pred_score_1(data=df_query_group,column_vec=column_vec,iter_mode=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config=select_config)
								
				df_score_1, df_score_2, contingency_table, dict_query1 = t_vec_1
				# print('field_query: ',column_query)
				# print('df_score_1: \n',df_score_1)
				# print('contingency_table: \n',contingency_table)
				list_query2.append(df_score_1)
				list_query3.append(column_query)

			if len(list_query2)>0:
				df_score_query = pd.concat(list_query2,axis=1,join='outer',ignore_index=False)
				df_score_query = df_score_query.T
				column_vec_query_1 = np.asarray(list_query3)
				
				field_query_2 = ['motif_id','motif_id1','motif_id2','group_motif','neighbor_num','method_type','method_type_group']
				group_id_motif = int(2-i2)
				list_2 = [motif_id_query,motif_id1,motif_id2,group_id_motif,n_neighbors,column_vec_query_1,method_type_group]
				for (field_id1,query_value) in zip(field_query_2,list_2):
					df_score_query[field_id1] = query_value

				list_query1.append(df_score_query)

		if len(list_query1)>0:
			df_score_query_1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)					
			# output_filename = '%s/test_query_df_score.%s.1.txt'%(output_file_path,filename_save_annot2_2)
			# df_score_query_1.to_csv(output_filename,sep='\t')

		return df_score_query_1

	## compare TF binding prediction
	def test_query_pred_score_1(self,data=[],column_vec=[],iter_mode=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config={}):

		df_score_query_1, df_score_query_2, contingency_table, dict_query1 = self.test_query_pred_score_unit1_1(data=data,column_vec=column_vec,iter_mode=iter_mode,
																													save_mode=save_mode,output_file_path=output_file_path,filename_prefix_save=filename_prefix_save,filename_annot_save=filename_annot_save,output_filename=output_filename,
																													verbose=verbose,select_config=select_config)

		return df_score_query_1, df_score_query_2, contingency_table, dict_query1

	## compare TF binding prediction
	def test_query_pred_score_unit1_1(self,data=[],column_vec=[],mode_query=1,iter_mode=0,plot_ax=[],save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config={}):
		
		df_1 = data
		column_signal, column_pred = column_vec[0:2]
		column_proba = []
		if len(column_vec)>2:
			column_proba = column_vec[2]

		peak_loc_1 = df_1.index
		peak_loc_num1 = len(peak_loc_1)
		# print('peak_loc_1, peak without motif: ',peak_loc_num1)

		# id_signal = (df_1['signal']>0)
		id_signal = (df_1[column_signal]>0)
		id_motif_2 = (df_1[column_pred]>0)
		id_query1_1 = (id_signal&id_motif_2) # with signal and with prediction (tp)
		id_query2_1 = (id_signal&(~id_motif_2)) # with signal and without prediction (fn)
		id_query1_2 = ((~id_signal)&id_motif_2)	# without signal and with prediction (fp)
		id_query2_2 = (~id_signal)&(~id_motif_2) # without signal and without prediction (tn)

		field_query = ['signal','signal_0','motif_2','motif_2_0','signal_motif_2','signal_motif_2_0','signal_0_motif_2','signal_0_motif_2_0']
		list1 = [id_signal,(~id_signal),id_motif_2,(~id_motif_2),id_query1_1,id_query2_1,id_query1_2,id_query2_2]
		list2 = [peak_loc_1[id_query] for id_query in list1]
		dict_query1 = dict(zip(field_query,list2))

		peak_signal_group1, peak_signal_group2, peak_motif_group1, peak_motif_group2, peak_tp, peak_fn, peak_fp, peak_tn = list2

		y_test = (id_signal).astype(int)
		y_pred = (id_motif_2).astype(int)

		from utility_1 import score_function_multiclass1, score_function_multiclass2

		if len(column_proba)==0:
			df_score_query1_1 = score_function_multiclass2(y_test,y_pred,average='binary')
		else:
			y_proba = df_1[column_proba]
			y_proba = y_proba.fillna(0)
			df_score_query1_1 = score_function_multiclass2(y_test,y_pred,y_proba=y_proba,average='binary',average_2='macro')
			if mode_query>0:
				precision_vec, recall_vec, thresh_value_vec_1 = precision_recall_curve(y_test,y_proba,pos_label=1,sample_weight=None)
				fpr_vec, tpr_vec, thresh_value_vec_2 = roc_curve(y_test,y_proba,pos_label=1,sample_weight=None)
				query_vec_1 = [precision_vec, recall_vec, thresh_value_vec_1]
				query_vec_2 = [fpr_vec, tpr_vec, thresh_value_vec_2]
				dict_query1.update({'precision_recall_curve':query_vec_1,'roc':query_vec_2})

		peak_signal_num1 = len(peak_signal_group1)
		peak_motif_num1 = len(peak_motif_group1)
		peak_tp_num = len(peak_tp)
		peak_fn_num = len(peak_fn)
		peak_fp_num = len(peak_fp)
		peak_tn_num = len(peak_tn)

		eps = 1E-12
		precision_1 = peak_tp_num/(peak_motif_num1+eps)
		recall_1 = peak_tp_num/(peak_signal_num1+eps)
		f1_score = 2*precision_1*recall_1/(precision_1+recall_1+eps)
		accuracy = (peak_tp_num+peak_tn_num)/peak_loc_num1
		t_vec_1 = [accuracy,precision_1,recall_1,f1_score]
		field_query_pre1 = ['accuracy','precision','recall','F1']
		df_score_query1_2 = pd.Series(index=field_query_pre1,data=t_vec_1,dtype=np.float32)

		contingency_table = [[peak_tp_num,peak_fn_num],[peak_fp_num,peak_tn_num]]
		contingency_table = np.asarray(contingency_table)

		try:
			stat_chi2_, pval_chi2_, dof_chi2_, ex_chi2_ = chi2_contingency(contingency_table,correction=True)
		except Exception as error:
			print('error! ',error)
			stat_chi2_, pval_chi2_, dof_chi2_, ex_chi2_ = 0, 1, 1, [0,0,0,0]

		try:
			stat_fisher_exact_, pval_fisher_exact_ = fisher_exact(contingency_table,alternative='greater')
		except Exception as error:
			print('error! ',error)
			stat_fisher_exact_, pval_fisher_exact_ = 0, 1

		if iter_mode==0:
			field_query_pre2 = ['stat_chi2_','pval_chi2_','dof_chi2_']+['ex_chi2_%d'%(id1) for id1 in np.arange(1,5)]+['stat_fisher_exact_','pval_fisher_exact_']
			query_value_1 = np.ravel(np.asarray(ex_chi2_))
			t_vec_2 = [stat_chi2_, pval_chi2_, dof_chi2_]+ list(query_value_1)+[stat_fisher_exact_, pval_fisher_exact_]
			df_score_query2_2 = pd.Series(index=field_query_pre2,data=np.asarray(t_vec_2),dtype=np.float32)
		else:
			field_query_pre2 = ['stat_chi2_','pval_chi2_','dof_chi2_']+['stat_fisher_exact_','pval_fisher_exact_']
			t_vec_2 = [stat_chi2_, pval_chi2_, dof_chi2_]+[stat_fisher_exact_, pval_fisher_exact_]
			df_score_query2_2 = pd.Series(index=field_query_pre2,data=np.asarray(t_vec_2),dtype=np.float32)
		
		# df_score_query_2 = pd.concat([df_score_query1_2,df_score_query2_2],axis=0,join='outer')
		df_score_query_1 = pd.concat([df_score_query1_1,df_score_query2_2],axis=0,join='outer')
		df_score_query_2 = df_score_query1_2

		return df_score_query_1, df_score_query_2, contingency_table, dict_query1

	## score query for performance comparison
	# performance comparison for the binary prediction and predicted probability
	def test_query_compare_binding_basic_unit1_1(self,data1=[],data2=[],motif_id_query='',group_query_vec=[2],df_annot_motif=[],dict_motif=[],dict_method_type=[],column_signal='signal',column_motif='',column_vec_query=[],feature_type_vec=[],method_type_vec=[],method_type_group='',flag_compare_1=0,type_id_1=0,load_mode=0,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		df_signal = data1
		df_pre1 = data2
		
		if flag_compare_1==0:
			list_group = [df_pre1]
			group_query_vec_1 = [2]
			group_query_vec = group_query_vec_1
			# query_num_1 = 1
			print('df_pre1: ',df_pre1.shape)
		else:
			id1 = (df_pre1[column_motif].abs()>0) # peak loci with motif
			id2 = (~id1)	# peak loci without motif
			df_query_group1 = df_pre1.loc[id1,:]
			df_query_group2 = df_pre1.loc[id2,:]
			df_query_group_1 = df_pre1
			print('df_pre1, df_query_group1, df_query_group2: ',df_pre1.shape,df_query_group1.shape,df_query_group2.shape)

			list_group = [df_query_group_1,df_query_group1,df_query_group2]
			group_query_vec_1 = [2,1,0]
		
		dict_group_query = dict(zip(group_query_vec_1,list_group))
		
		# query_num_1 = len(list_group)
		query_num_1 = len(group_query_vec)
		query_num_2 = len(column_vec_query)

		if method_type_group=='':
			method_type_group = select_config['method_type_group']

		list_query1 = []
		# group_query_vec = [2,1,0]
		for i2 in range(query_num_1):
			# df_query_group = list_group[i2]
			group_id_motif = group_query_vec[i2]
			df_query_group = dict_group_query[group_id_motif]

			# query the signal
			peak_loc_pre1 = df_query_group.index
			df_query_group[column_signal] = df_signal.loc[peak_loc_pre1,:]
			print('df_query_group, group_motif: ',df_query_group.shape,group_id_motif)
			print(df_query_group[0:2])

			list_query2 = []
			list_query3 = []
			for t_id1 in range(query_num_2):
				column_query_1 = column_vec_query[t_id1]
				if isinstance(column_query_1,str):
					column_pred = column_query_1
					column_proba = []
				else:
					column_pred, column_proba = column_query_1[0:2]

				column_vec = [column_signal,column_pred,column_proba]
				t_vec_1 = self.test_query_pred_score_1(data=df_query_group,column_vec=column_vec,iter_mode=0,save_mode=1,output_file_path='',filename_prefix_save='',filename_annot_save='',output_filename='',verbose=0,select_config=select_config)
								
				df_score_1, df_score_2, contingency_table, dict_query1 = t_vec_1
				list_query2.append(df_score_1)
				list_query3.append(column_pred)

			if len(list_query2)>0:
				df_score_query = pd.concat(list_query2,axis=1,join='outer',ignore_index=False)
				df_score_query = df_score_query.T

				column_vec_query_1 = np.asarray(list_query3)
				if len(dict_method_type)==0:
					method_type_vec_query = column_vec_query_1
				else:
					method_type_vec_query = [dict_method_type[column_query] for column_query in column_vec_query_1]
					
				field_query_2 = ['motif_id','motif_id1','motif_id2','group_motif','method_type','method_type_group']
				
				motif_id2 = motif_id_query
				motif_id_ori = df_annot_motif.loc[motif_id2,'motif_id']
				motif_id1 = df_annot_motif.loc[motif_id2,'motif_id1']

				list_2 = [motif_id_ori,motif_id1,motif_id2,group_id_motif,method_type_vec_query,method_type_group]
				
				for (field_id1,query_value) in zip(field_query_2,list_2):
					df_score_query[field_id1] = query_value
				list_query1.append(df_score_query)

		if len(list_query1)>0:
			df_score_query_1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)

		return df_score_query_1

	## score query for performance comparison
	def test_query_compare_binding_basic_1(self,data1=[],data2=[],motif_query_vec=[],motif_query_vec_2=[],df_annot_motif=[],dict_motif=[],dict_method_type=[],feature_type_vec=[],column_vec_query=[],column_signal='signal',column_motif='',method_type_vec=[],method_type_group='',flag_score_1=0,flag_score_2=0,flag_compare_1=0,type_id_1=0,parallel=1,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		dict_signal = data1
		dict_query = data2
		motif_query_num = len(motif_query_vec)

		query_res = []
		if parallel==0:
			for i1 in range(motif_query_num):
				motif_query = motif_query_vec[i1] # motif_id2_query
				if (motif_query in dict_signal) and (motif_query in dict_query):
					t_query_res = self.test_query_compare_binding_basic_unit1_1(data1=dict_signal[motif_query],data2=dict_query[motif_query],
																					motif_id_query=motif_query,
																					df_annot_motif=df_annot_motif,
																					dict_method_type=dict_method_type,
																					column_vec_query=column_vec_query,
																					type_id_1=type_id_1,select_config=select_config)
					if len(t_query_res)>0:
						query_res.append(t_query_res)
				else:
					print('motif query not included ',motif_query)

		else:
			query_res_local = Parallel(n_jobs=-1)(delayed(self.test_query_compare_binding_basic_unit1_1)(data1=dict_signal[motif_query],data2=dict_query[motif_query],
																											motif_id_query=motif_query,
																											df_annot_motif=df_annot_motif,
																											dict_method_type=dict_method_type,
																											column_vec_query=column_vec_query,
																											type_id_1=type_id_1,select_config=select_config) for motif_query in tqdm(motif_query_vec))

			for t_query_res in query_res_local:
				# dict_query = t_query_res
				if len(t_query_res)>0:
					query_res.append(t_query_res)

		list_score_query_1 = query_res
		df_score_query_1 = []
		if len(list_score_query_1)>0:
			df_score_query_1 = pd.concat(list_score_query_1,axis=0,join='outer',ignore_index=False)
			if (save_mode>0) and (output_filename!=''):
				df_score_query_1.to_csv(output_filename,sep='\t')

		return df_score_query_1

	## TF binding prediction performance
	def test_query_compare_binding_pre1_5_1_basic_1(self,data=[],feature_query_vec=[],method_type_vec=[],type_query=0,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		run_id1 = select_config['run_id']

		# method_type_vec = ['insilico_0.1','GRaNIE','Pando','TRIPOD']+['joint_score_pre1.thresh22']
		method_type_feature_link = select_config['method_type_feature_link']
		method_type_group = select_config['method_type_group']

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
		feature_type_vec_group = ['latent_peak_tf','latent_peak_motif']

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

		input_filename = select_config['file_motif_annot']
		df_annot_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
		df_annot1_1 = df_annot_ori.drop_duplicates(subset=['motif_id'])

		df_annot_1 = df_annot1_1
		print('df_annot_ori, df_annot_1: ',df_annot_ori.shape,df_annot_1.shape)

		# motif_idvec_query = df_annot_1.index.unique()
		motif_idvec_query = df_annot_1['motif_id'].unique()

		# dict_motif = []
		df_annot_motif = df_annot_1
		print('df_annot_motif: ',df_annot_motif.shape)
		print(df_annot_motif)
		print('dict_method_type: ',dict_method_type)

		if len(feature_query_vec)>0:
			motif_query_vec_1 = feature_query_vec
		else:
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
		filename_save_annot_pre1 = '%s.0.1'%(data_file_type_query)
		input_file_path_1 = input_file_path_query_2
		start = time.time()
		dict_file_signal = select_config['dict_file_signal']
		dict_file_pred = select_config['dict_file_pred']
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

				filename_annot_train_pre1 = '%s_%s.%d'%(ratio_1,ratio_2,config_id_2)
				filename_save_annot_query = '%s.%s.neighbor%d'%(method_type_group,filename_annot_train_pre1,n_neighbors)

				file_path_query_2 = dict_file_annot2[folder_id]
				input_file_path_1 = file_path_query_2
				
				input_filename_1 = dict_file_signal[motif_id2] # the ChIP-seq signal file
				input_filename_2 = dict_file_pred[motif_id_query]	# the peak-TF association estimation file
				
				if os.path.exists(input_filename_2)==False:
					print('the file does not exist: ',input_filename_2)
					continue

				df_query_pre1 = pd.read_csv(input_filename_1,index_col=0,sep='\t')
				df_query_pre1 = df_query_pre1.loc[(~df_query_pre1.index.duplicated(keep='first')),:]
				print(input_filename_1)

				df_query_1 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
				df_query_1 = df_query_1.loc[(~df_query_1.index.duplicated(keep='first')),:]
				t_columns_1 = df_query_1.columns.difference([column_signal],sort=False)
				print(input_filename_2)

				peak_loc_pre1 = df_query_1.index
				df_signal = df_query_pre1.loc[peak_loc_pre1,[column_signal]]
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
		output_filename = '%s/test_query_df_score.beta.%s.1.txt'%(output_file_path,filename_save_annot_1)
		df_score_query_1.to_csv(output_filename,sep='\t')
		print(output_filename)

		return df_score_query_1

	## TF binding prediction performance
	def test_query_compare_binding_pre1_5_1_basic_unit1(self,data_1=[],data_2=[],df_annot=[],dict_method_type=[],feature_query_vec=[],column_vec_query=[],column_signal='',column_motif='',type_query=0,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})
		run_id1 = select_config['run_id']

		# motif_query_vec = list1
		motif_query_vec = feature_query_vec
		motif_query_num1 = len(motif_query_vec)
		if column_signal=='':
			column_signal = 'signal'
		if column_motif=='':
			method_type_feature_link = select_config['method_type_feature_link']
			column_motif = '%s.motif'%(method_type_feature_link)
		column_vec_pre1 = column_vec_query

		start = time.time()
		dict_query_pre1 = data_1
		dict_query_pre2 = data_2
		load_mode = 1
		dict_signal_1 = dict()
		dict_query_1 = dict()

		for i1 in range(motif_query_num1):
			motif_id2 = motif_query_vec[i1]
			motif_id_query = df_annot.loc[motif_id2,'motif_id']
			motif_id = motif_id_query
			motif_id1 = df_annot.loc[motif_id2,'motif_id1']

			df_query_1 = dict_query_pre2[motif_id_query]
			# df_query_1 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			df_query_1 = df_query_1.loc[(~df_query_1.index.duplicated(keep='first')),:]
			t_columns_1 = df_query_1.columns.difference([column_signal],sort=False)
			print(df_query_1.columns)
			# print(input_filename_2)

			peak_loc_pre1 = df_query_1.index
			# df_query_pre1_1 = df_query_pre1.loc[peak_loc_pre1,:]
			if load_mode>0:
				df_query_pre1 = dict_query_pre1[motif_id2]
			else:
				df_query_pre1 = df_query_1

			# df_signal = df_query_1.loc[:,[column_signal]]
			# df_query1 = df_query_1.loc[:,t_columns_1]
			df_signal = df_query_pre1.loc[peak_loc_pre1,[column_signal]]
			column_vec_query1 = np.ravel(column_vec_query)
			df_query1 = df_query_1.loc[:,column_vec_query1]

			print('df_signal,df_query1,motif_id: ',df_signal.shape,df_query1.shape,motif_id,motif_id1,motif_id2,i1)
			print(df_signal[0:2])
			print(df_query1[0:2])
				
			dict_signal_1.update({motif_id2:df_signal})
			dict_query_1.update({motif_id2:df_query1})

		stop = time.time()
		print('load data used %.2fs'%(stop-start))

		# score query for performance comparison
		start = time.time()
		df_annot_motif = df_annot
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

		if (save_mode>0) and (output_filename!=''):
			df_score_query_1.to_csv(output_filename,sep='\t')
			print(output_filename)

		return df_score_query_1

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
	test_estimator1 = _Base2_2_1(file_path=file_path_1)

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


