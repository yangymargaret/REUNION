#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import scipy
import scipy.io
import sklearn
import math

from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.switch_backend('Agg')

import os
import os.path
from optparse import OptionParser

from scipy import stats
from scipy.stats import chisquare, chi2_contingency, fisher_exact
from scipy.stats.contingency import expected_freq

from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,accuracy_score,matthews_corrcoef
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import mean_squared_error, accuracy_score, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score

import time
from timeit import default_timer as timer

# import gc
from joblib import Parallel, delayed
import multiprocessing as mp
import threading
from tqdm.notebook import tqdm
from .test_rediscover_compute_1 import _Base2_2
from . import utility_1
from .utility_1 import test_query_index
import h5py
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

	## ====================================================
	# load the ChIP-seq data annotations
	def test_query_file_annotation_load_1(self,data_file_type='',input_filename='',celltype_vec=[],save_mode=1,verbose=0,select_config={}):

		"""
		load the ChIP-seq data annotations
		:param data_file_type: (str) name or identifier of the data
		:param input_filename: (str) path of the file of the ChIP-seq data annotations
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the ChIP-seq data annotations;
				 2. (array or pd.Index) the TFs for which the ChIP-seq experiments were performed;
		"""

		flag_query1=1
		if flag_query1>0:
			df_annot = pd.read_csv(input_filename,index_col=0,sep='\t')
			if 'motif_id' in df_annot.columns:
				motif_idvec_query = df_annot['motif_id'].unique()
			else:
				df_annot['motif_id'] = np.asarray(df_annot.index)
				motif_idvec_query = df_annot.index.unique()

			if len(celltype_vec)>0:
				id1 = df_annot['celltype'].isin(celltype_vec)
				df_annot_motif = df_annot.loc[id1,:]
			else:
				df_annot_motif = df_annot

			df_annot_motif = df_annot_motif.drop_duplicates(subset=['filename','celltype','motif_id'])

			filename_vec = np.asarray(df_annot_motif['filename'])
			query_num1 = len(filename_vec)

			file_path_query2 = select_config['file_path_annot_2']
			input_filename_list = []
			list1 = []
			for i1 in range(query_num1):
				input_filename = filename_vec[i1]
				str_vec = input_filename.split('.')
				filename_prefix = '%s.%s'%(str_vec[0],str_vec[1])
				input_filename_2 = '%s/%s.overlap.bed'%(file_path_query2,filename_prefix)
				input_filename_list.append(input_filename_2)
				list1.append(filename_prefix)

			feature_query_vec = np.asarray(list1)
			df_annot_motif['motif_id2'] = feature_query_vec
			df_annot_query = df_annot_motif

			print('df_annot_motif ',df_annot_motif.shape)
			print('data preview: ')
			print(df_annot_motif[0:2])

			return df_annot_query, motif_idvec_query, input_filename_list, feature_query_vec

	## ====================================================
	# query TF binding prediction evaluation metric scores
	def test_query_pred_score_1(self,data=[],column_vec=[],method_type='',score_type=0,mode_query=2,default_value=1,
									save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_annot_save='',verbose=0,select_config={}):
		
		"""
		query TF binding prediction evaluation metric scores
		:param data: (dataframe) ATAC-seq peak loci anntations including the ChIP-seq signals and TF binding predictions for the given TF
		:param column_vec: (array or list) columns representing TF binding predictions for the given TF
		:param method_type: (str) name of the method used for TF binding prediction
		:param score_type: the association score type: 
						   0: higher score represents higher association strength;
						   1: lower score represents higher association strength;
		:param mode_query: indicator of which evaluation metrics to query:
						   0: basic metrics: accuracy, precision, recall, F1, AUROC, AUPR;
						   1: basic metrics and recompute AUPR for the methods which rely on motif scanning results;
						   2: basic metrics with recomputed AUPR, precision at fixed recall, recall at fixed precision,
						   3: the metrics in 2 and enrichment of true positive peak-TF link predictions; 
		:param default_value: the default value of peak-TF link scores for the score type that lower score represents higher association strength
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) TF binding prediction performance evaluation metric scores (row:metric, column:score);
				 2. the contingency table to compute enrichment of true positive peak-TF link predictions using Fisher's exact test and chi-squared test;
				 3. dictionary containing dataframes of scores and thresholds associated with the precision-recall curve and roc curve, including:
					3-1. dataframe of precision and recall at each given threshold on the precision-recall curve;
					3-2. dataframe of true positive rate (TPR) and false positive rate (FPR) at each given threshold on the roc curve;
		"""

		df_1 = data
		column_signal, column_pred = column_vec[0:2]
		column_proba = []
		if len(column_vec)>2:
			column_proba = column_vec[2]

		peak_loc_1 = df_1.index
		peak_loc_num1 = len(peak_loc_1)

		id_signal = (df_1[column_signal]>0) # identify peak loci with ChIP-seq signals
		id_pred = (df_1[column_pred]>0) # identify peak loci predicted to be bound by the TF

		y_test = (id_signal).astype(int) # true labels of TF binding based on ChIP-seq data
		y_pred = (id_pred).astype(int)	# predicted labels of TF binding

		from utility_1 import score_function_multiclass1, score_function_multiclass2
		column_1 = 'score_type'
		if (score_type<0) and (column_1 in select_config):
			score_type = select_config[column_1]
		print('score_type: ',score_type)

		dict_query1 = dict()
		id1 = (column_proba=='')|(column_proba in [-1])|(len(column_proba)==0)
		if column_proba in df_1.columns:
			id2 = ((~pd.isna(df_1[column_proba])).sum()==0)
			id1 = (id1|id2)
		if id1>0:
			print('binary prediction')
			# compute the evaluation metric scores
			df_score_query1_1 = score_function_multiclass2(y_test,y_pred,average='binary')
		else:
			print('binary prediction and with predicted probability')
			y_proba = df_1[column_proba]
			if score_type>0:
				y_proba = y_proba.fillna(default_value) # lower score represents higher association strength
				y_proba = 1-y_proba
			else:
				y_proba = y_proba.fillna(0)
			
			# compute the evaluation metric scores
			df_score_query1_1 = score_function_multiclass2(y_test,y_pred,y_proba=y_proba,average='binary',average_2='macro')
			
			if mode_query>0:
				column_annot = []
				dict_annot = []
				column_1, column_2 = 'column_annot_score', 'dict_annot_score'
				if column_1 in select_config:
					column_annot = select_config[column_1]
				if column_2 in select_config:
					dict_annot = select_config[column_2]

				score_type_2 = 0
				column_3 = 'score_type_2' # label of whether the peak-TF link prediction is dependent on TF binding motif scanning
				if column_3 in select_config:
					score_type_2 = select_config[column_3]

				y_depend = []
				default_value_2 = 0
				if score_type_2>0:
					column_motif = column_vec[3]
					try:
						id_motif = (df_1[column_motif].abs()>0)
					except Exception as error:
						print('error! ',error)
						id_motif = df_1[column_motif].isin([True,'True',1,'1'])

					y_depend = pd.Series(index=peak_loc_1,data=default_value_2)
					y_depend.loc[id_motif] = 1

				# query precision-recall curves and roc curves
				# recompute AUPR for methods which rely on motif scanning results
				dict_query1, df_score_query1_1 = self.test_query_compare_precision_recall_1(y_test=y_test,y_proba=y_proba,
																							y_depend=y_depend,
																							default_value=default_value,
																							df_score=df_score_query1_1,
																							column_annot=column_annot,
																							dict_annot=dict_annot,
																							flag_score=1,
																							save_mode=save_mode,
																							verbose=verbose,select_config=select_config)

				column_1 = 'df_precision_recall'
				flag_2 = (mode_query>=2)
				flag_2 = (flag_2)&(column_1 in dict_query1)	
				if (column_1 in dict_query1):
					df_precision_recall = dict_query1[column_1]
					column_2 = 'method_type'
					df_precision_recall[column_2] = method_type
					dict_query1.update({column_1:df_precision_recall})

					print('df_precision_recall, dataframe of size ',df_precision_recall.shape)
					print('data preview:\n',df_precision_recall[0:2])

				if flag_2>0:
					thresh_vec_query1 = [0.05,0.10,0.20,0.25,0.50] # threshold for precision at given recall
					thresh_vec_query2 = [0.50,0.70,0.90]	# threshold for recall at given precision
					thresh_difference = 0.05
					# compute precision at the given recall and recall at the given precision
					df_score_query1_1 = self.test_query_compare_precision_recall_2(data=df_precision_recall,df_score=df_score_query1_1,
																					thresh_vec_1=thresh_vec_query1,
																					thresh_vec_2=thresh_vec_query2,
																					thresh_difference=thresh_difference,
																					save_mode=save_mode,
																					verbose=verbose,select_config=select_config)

		flag_enrichment = (mode_query==2)
		contingency_table = []
		if flag_enrichment>0:
			df_score_query2_1, contingency_table = self.test_query_enrichment_score_1(data=[],idvec_1=id_signal,
																	idvec_2=id_pred,
																	feature_query_vec=peak_loc_1,
																	save_mode=1,
																	verbose=0,select_config=select_config)

			df_score_query_1 = pd.concat([df_score_query1_1,df_score_query2_1],axis=0,join='outer')
		
		else:
			df_score_query_1 = df_score_query1_1

		return df_score_query_1, contingency_table, dict_query1

	## ====================================================
	# query TF binding prediction evaluation metric scores
	# query precision-recall curves and roc curves
	# recompute AUPR for methods which rely on motif scanning results
	def test_query_compare_precision_recall_1(self,y_test=[],y_proba=[],y_depend=[],default_value=None,df_score=[],column_annot=[],dict_annot=[],flag_score=1,
												save_mode=0,verbose=0,select_config={}):

		"""
		query TF binding prediction evaluation metric scores;
		query precision-recall curves and roc curves;
		recompute AUPR for methods which rely on motif scanning results;
		:param y_test: (array) true labels of the variable
		:param y_proba (array) predicted positive class probabilities of the variable
		:param y_depend: (array) value of the conditional variable (e.g.,motif presence in the peak loci)
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing dataframes of scores and thresholds associated with the precision-recall curve and roc curve, including:
					1-1. dataframe of precision and recall at each given threshold on the precision-recall curve;
					1-2. dataframe of true positive rate (TPR) and false positive rate (FPR) at each given threshold on the roc curve;
				 2. (dataframe) prediction performance evaluation metric scores (row:metric, column:score);
		"""

		mode_query = 1
		if mode_query>0:
			flag_depend = 0
			signal_ratio = 1
			column_1 = 'motif_id'
			column_2 = 'method_type'
			motif_id_query, method_type_query = 'motif', 'method_type'
			if column_1 in dict_annot:
				motif_id_query = dict_annot[column_1]

			if column_2 in dict_annot:
				method_type_query = dict_annot[column_2]

			if len(y_depend)>0:
				flag_depend = 1
				if not(default_value is None):
					# id_1 = (y_depend!=default_value)
					id_1 = (y_depend>default_value)
				else:
					id_1 = (~np.isnan(y_depend))

				y_test_ori = y_test.copy()
				y_proba_ori = y_proba.copy()

				y_test = y_test_ori[id_1]	# true labels of the peak loci with TF motif
				y_proba = y_proba_ori[id_1] # predicted probabilities for the peak loci with TF motif

				query_num2 = np.sum(y_test>0)
				query_num1 = np.sum(y_test_ori>0)
				signal_ratio = np.sum(y_test>0)/np.sum(y_test_ori>0) # the percentage of peak loci with TF motif and positive labels in the peak loci with positive labels
				print('signal_ratio: ',signal_ratio,motif_id_query,method_type_query)

			df1 = []
			df2 = []
			verbose_internal = self.verbose_internal
			if flag_score>0:
				precision_vec, recall_vec, thresh_value_vec_1 = precision_recall_curve(y_test,y_proba,pos_label=1,sample_weight=None)
				fpr_vec, tpr_vec, thresh_value_vec_2 = roc_curve(y_test,y_proba,pos_label=1,sample_weight=None)
				
				# recompute recall for methods which rely on motif scanning results
				if flag_depend>0:
					recall_vec_1 = recall_vec.copy()
					recall_vec = recall_vec_1*signal_ratio # recompute the recall

				query_vec_1 = [precision_vec, recall_vec, thresh_value_vec_1]
				query_vec_2 = [fpr_vec, tpr_vec, thresh_value_vec_2]

				if verbose_internal>0:
					print('precisions with different thresolds: ',len(precision_vec),motif_id_query,method_type_query)
					print('recalls with different thresolds: ',len(recall_vec),motif_id_query,method_type_query)
				
				dict_1 = dict()

				field_query_1 = ['precision','recall','thresh']
				df1 = pd.DataFrame(columns=field_query_1)
				if len(thresh_value_vec_1)<len(precision_vec):
					thresh_value_vec_1 = list(thresh_value_vec_1)+[1]
					query_vec_1[-1] = thresh_value_vec_1

				for (field_id,query_value) in zip(field_query_1,query_vec_1):
					df1[field_id] = np.asarray(query_value)

				if flag_depend>0:
					df1['recall_1'] = recall_vec_1

				field_query_2 = ['fdr','tpr','thresh']
				df2 = pd.DataFrame(columns=field_query_2)
				if len(thresh_value_vec_2)<len(fpr_vec):
					thresh_value_vec_2 = list(thresh_value_vec_2)+[1]
					query_vec_2[-1] = thresh_value_vec_2

				for (field_id,query_value) in zip(field_query_2,query_vec_2):
					df2[field_id] = np.asarray(query_value)

				if len(column_annot)>0:
					column_vec = column_annot
					for column_query in column_vec:
						query_value = dict_annot[column_query]
						df1[column_query] = query_value
						df2[column_query] = query_value

				dict_1.update({'df_precision_recall':df1,'df_roc':df2})

			df_score_query1_1 = df_score
			if flag_depend>0:
				df_score_query1_1 = df_score_query1_1.rename(index={'aupr':'aupr_1'}) # rename the column of the original AUPR
				aupr_1 = df_score_query1_1['aupr_1']
				try:
					average_2 = 'macro'
					aupr_2 = average_precision_score(y_test,y_proba,average=average_2) # compute AUPR for the peak loci with the TF motif;
				except Exception as error:
					print('error!',error)
					aupr_2 = 0
			
				aupr_query = aupr_2*signal_ratio  # recompute the AUPR
				field_query_pre2 = ['aupr_2','aupr','signal_ratio']
				list2 = [aupr_2,aupr_query,signal_ratio]

				if verbose_internal>0:
					print('aupr_1, aupr_2, aupr_query, signal_ratio: ',aupr_1,aupr_2,aupr_query,signal_ratio,motif_id_query,method_type_query)
			else:
				field_query_pre2 = []
				list2 = []

			if len(list2)>0:
				df_score_query1_2 = pd.Series(index=field_query_pre2,data=np.asarray(list2))
				df_score_query1_1 = pd.concat([df_score_query1_1,df_score_query1_2],axis=0,join='outer',ignore_index=False)

			if verbose_internal>0:
				print('peak-TF link estimation performance score for TF %s'%(motif_id_query))
				print(df_score_query1_1)

			return dict_1, df_score_query1_1

	## ====================================================
	# query TF binding prediction evaluation metric scores
	# compute precision at the given recall and recall at the given precision
	def test_query_compare_precision_recall_2(self,data=[],df_score=[],thresh_vec_1=[0.05,0.10],thresh_vec_2=[0.50,0.90],thresh_difference=0.05,save_mode=1,verbose=0,select_config={}):

		"""
		query TF binding prediction evaluation metric scores;
		compute precision at the given recall and recall at the given precision;
		:param data: (dataframe) precisions, recalls, and the corresponding thresholds on the precision-recall curve
		:param df_score: (dataframe) the other types of computed evaluation metric scores
		:param thresh_vec_1: (array or list) the given values of recall associated with which to query precision
		:param thresh_vec_2: (array or list) the given values of precision associated with which to query recall
		:param thresh_difference: (float) threshold on the difference between recall (or precision) and the specific target value associated with which to query precision (or recall)
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) prediction performance evaluation metric scores (row:metric, column:score)
		"""

		# thresh_difference = 0.05
		df_precision_recall = data
		list1 = []
		if len(df_score)>0:
			list1 = [df_score]

		if len(thresh_vec_1)>0:
			# thresh_vec_query1 = [0.05,0.10,0.20,0.25,0.50]
			thresh_vec_query1 = thresh_vec_1
			column_vec_query1=['recall','precision']
			df1 = self.test_query_precision_with_recall_1(data=df_precision_recall,thresh_vec_query=thresh_vec_query1,
															thresh_difference=thresh_difference,
															save_mode=save_mode,
															verbose=verbose,select_config=select_config)

			column_thresh_1, column_value_1 = column_vec_query1[0], column_vec_query1[1]
			query_id_1 = df1.index
			t_vec_1 = ['%s_%s'%(column_thresh_1,thresh_query) for thresh_query in query_id_1]
			df1.index = t_vec_1
			list1.append(df1[column_value_1])

		if len(thresh_vec_2)>0:
			# thresh_vec_query2 = [0.50,0.70,0.90]
			thresh_vec_query2 = thresh_vec_2
			column_vec_query2=['precision','recall']
			df2 = self.test_query_recall_with_precision_1(data=df_precision_recall,thresh_vec_query=thresh_vec_query2,
															thresh_difference=thresh_difference,
															save_mode=save_mode,
															verbose=verbose,select_config=select_config)

			
			column_thresh_2, column_value_2 = column_vec_query2[0], column_vec_query2[1]
			query_id_2 = df2.index
			t_vec_2 = ['%s_%s'%(column_thresh_2,thresh_query) for thresh_query in query_id_2]
			df2.index = t_vec_2
			list1.append(df2[column_value_2])

		# list1 = [df_score_1,df1[column_value_1],df2[column_value_2]]
		df_score_1 = pd.concat(list1,axis=0,join='outer',ignore_index=False)

		return df_score_1

	## ====================================================
	# query precision at the given values of recall
	def test_query_precision_with_recall_1(self,data=[],thresh_vec_query=[],thresh_difference=0.05,type_query=0,save_mode=1,verbose=0,select_config={}):

		"""
		query precision at the given values of recall
		:param data: (dataframe) precisions, recalls, and the corresponding thresholds on the precision-recall curve
		:param thresh_vec_query: (array or list) values of recall associated with which to query precision
		:param thresh_difference: (float) threshold on the difference between recall and the specific target value associated with which to query precision
		:param type_query: indicator of the approach to retrieve precision from the score dataframe at the specific value of recall:
						   0: retrieve precision at the classification threshold with which recall has the minimal difference from the specific value;
						   1: retrieve the maximal precision at the thresholds with which recall is close to or lower than the specific value;
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) precision at the given values of recall
		"""

		column_vec_query=['recall','precision']
		df1 = self.test_query_score_unit_1(data=data,column_vec_query=column_vec_query,
											thresh_vec_query=thresh_vec_query,
											thresh_difference=thresh_difference,
											direction=0,
											type_query=type_query,
											save_mode=save_mode,
											verbose=verbose,select_config=select_config)
		return df1

	## ====================================================
	# query recall at the given values of precision
	def test_query_recall_with_precision_1(self,data=[],thresh_vec_query=[],thresh_difference=0.05,type_query=0,save_mode=1,verbose=0,select_config={}):

		"""
		query recall at the given values of precision
		:param data: (dataframe) precisions, recalls, and the corresponding thresholds on the precision-recall curve		
		:param thresh_vec_query: (array or list) values of precision associated with which to query recall
		:param thresh_difference: (float) threshold on the difference between precision and the specific target value associated with which to query recall
		:param type_query: indicator of the approach to retrieve recall from the score dataframe at the specific value of precision:
						   0: retrieve recall at the classification threshold with which precision has the minimal difference from the specific value;
						   1: retrieve the maximal recall at the thresholds with which precision is close to or lower than the specific value;
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) recall at the given values of precison
		"""

		column_vec_query=['precision','recall']
		df1 = self.test_query_score_unit_1(data=data,column_vec_query=column_vec_query,
											thresh_vec_query=thresh_vec_query,
											thresh_difference=thresh_difference,
											direction=1,
											type_query=type_query,
											save_mode=save_mode,
											verbose=verbose,select_config=select_config)
		return df1

	## ====================================================
	# query score 2 at the given values of score 1
	def test_query_score_unit_1(self,data=[],column_vec_query=['precision','recall'],thresh_vec_query=[],
									thresh_difference=0.025,direction=1,type_query=0,flag_sort=1,
									save_mode=1,verbose=0,select_config={}):

		"""
		query score 2 at the given values of score 1
		:param data: (dataframe) score 1 (e.g.,precision) and score 2 (e.g., recall) at each given threshold for classification
		:param column_vec_query: (array or list) columns that correspond to score 1 and score 2 in the score dataframe
		:param thresh_vec_query: (array or list) values of score 1 associated with which to query score 2
		:param thresh_difference: (float) threshold on the difference between score 1 and the specific target value associated with which to query score 2
		:param direction: indicator of the score type to compute: 0:precision at given recall; 1:recall at given precision;
		:param type_query: indicator of the approach to retrieve score 2 from the score dataframe at the specific value of score 1:
						   0: retrieve score 2 at the classification threshold with which score 1 has the minimal difference from the specific value;
						   1: retrieve the maximal score 2 at the thresholds with which score 1 is close to or lower than the specific value;
		:param flag_sort: indicator of whether to sort score dataframe based on score 1 in ascending order;
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) score 2 at the given values of score 1
		"""

		column_1, column_2 = column_vec_query[0:2]
		score_vec_1 = np.asarray(data[column_1]) # the first type of score (score 1) dependent on which to query the second type of score (score 2)
		score_vec_2 = np.asarray(data[column_2]) # the second type of score
		if flag_sort>0:
			id_1 = np.argsort(score_vec_1) # sort the first type of score in ascending order;
			score_vec_1 = score_vec_1[id_1]
			score_vec_2 = score_vec_2[id_1]

		# thresh_precision_vec = [thresh_precision_1,thresh_precision_2]
		thresh_num1 = len(thresh_vec_query)
		list1 = []
		thresh_difference_2 = 0.10
		thresh_difference_query = thresh_difference
		verbose_internal = self.verbose_internal
		for i1 in range(thresh_num1):
			thresh_query = thresh_vec_query[i1]
			if thresh_query<=0.10:
				thresh_difference_query = np.min([0.01,thresh_difference])
				thresh_difference_2 = np.min([0.05,thresh_difference])  # to update
			else:
				thresh_difference_query = thresh_difference

			diff_query = np.abs(score_vec_1-thresh_query)
			min_diff_query = np.min(diff_query)
			print('the minimal difference between score and the threshold: %s, threshold: %s'%(min_diff_query,thresh_query))
			print('thresh_difference_query: ',thresh_difference_query)

			thresh_2 = 1E-09
			thresh_value_1 = 1.0
			thresh_value_2 = (thresh_value_1-thresh_difference_2)
			eps = 1E-09
			b1 = np.where(score_vec_1>=(thresh_value_1-eps))[0]
			b2 = np.where(score_vec_2[b1]>thresh_2)[0]
			# recall = 0 if precision = 1
			id_pre2 = (column_1 in ['precision'])&(len(b2)==0)&(thresh_query>=(thresh_value_2-eps))

			query_value_2 = 0
			if min_diff_query<thresh_difference_query:
				id_1 = np.argmin(diff_query)
				query_value_1 = score_vec_1[id_1]
				if type_query>0:
					query_value_2 = np.max(score_vec_2[id_1:])
				else:
					query_value_2 = score_vec_2[id_1]
				print(column_1,column_2,query_value_1,query_value_2)
			else:
				id1 = np.where(score_vec_1<thresh_query)[0]
				id2 = np.where(score_vec_1>thresh_query)[0]

				if (min_diff_query>thresh_difference_2):	
					# continue
					if len(id2)>0:
						# id_query2 = id2[0]
						query_value_2 = np.max(score_vec_2[id2])
					else:
						print('the threshold not included ',thresh_query)
				else:
					if (id_pre2)>0:
						print('the threshold not included ',thresh_query)
						# continue
					else:
						print('interpolation ',thresh_query,column_1,column_2)
						# perform interpolation of score 2
						if (len(id1)>0) and (len(id2)>0):
							t_id1 = id1[-1]
							t_id2 = id2[0]
							score_query2_1 = score_vec_1[t_id1] # score_1 below the threshold
							score_query1_1 = score_vec_1[t_id2] # score_1 above the threshold
							if verbose_internal==2:
								print('score type 1: score 1: %.5E, score 2: %.5E '%(score_query2_1,score_query1_1))

							score_query2_2 = score_vec_2[t_id1]	# score_2 above the threshold
							score_query1_2 = score_vec_2[t_id2]	# score_2 below the threshold
							if verbose_internal==2:
								print('score type 2: score 1: %.5E, score 2: %.5E '%(score_query2_2,score_query1_2))
							
							k1 = (score_query2_2-score_query1_2)/(score_query2_1-score_query1_1)
							query_value_2 = score_query1_2+(thresh_query-score_query1_1)*k1 # interpolation
							print('the slope: %.5E, the interpolated value: %.5E'%(k1,query_value_2))
						
						elif (len(id1)==0) and (len(id2)>0):
							query_value_2 = np.max(score_vec_2[id2])
					
			list1.append(query_value_2)

		query_vec = np.asarray(list1)
		column_name = column_2
		df1 = pd.DataFrame(index=thresh_vec_query,columns=[column_name],data=query_vec)
		column_query1 = '%s_thresh'%(column_1)
		df1[column_query1] = np.asarray(df1.index).copy()
		if verbose_internal>0:
			print('the estimated %s at different thresholds of %s, dataframe of size '%(column_2,column_1),df1.shape)
			print(df1)

		return df1

	## ====================================================
	# query enrichment of true positive peak-TF link predictions
	def test_query_enrichment_score_1(self,data=[],idvec_1=[],idvec_2=[],feature_query_vec=[],save_mode=0,verbose=0,select_config={}):

		"""
		query enrichment of true positive peak-TF link predictions in the predictions
		:param data: (list) each element is a boolean pandas.Series which correponds to the specific type of peak loci based on 
							whether the peaks are overlapping with ChIP-seq peaks and the predicted TF binding labels
		:param idvec_1: (boolean pandas.Series) indicators of whethe a ATAC-seq peak locus overlaps with ChIP-seq peak loci
		:param idvec_2: (boolean pandas.Series) indicators of whethe a ATAC-seq peak locus is predicted to be bound by the given TF
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the statistics and p-values of enrichment of true positive peak-TF link predictions using Fisher's exact test and chi-squared test
				 2. the contingency table to compute enrichment of true positive peak-TF link predictions using Fisher's exact test and chi-squared test;
		"""

		list_query1 = data
		if len(list_query1)==0:
			id_signal, id_pred = idvec_1, idvec_2
			id_query1_1 = (id_signal&id_pred)	# peak loci with signal and with prediction (tp)
			id_query2_1 = (id_signal&(~id_pred))	# peak loci with signal and without prediction (fn)
			id_query1_2 = ((~id_signal)&id_pred)	# peak loci without signal and with prediction (fp)
			id_query2_2 = (~id_signal)&(~id_pred)	# peak loci without signal and without prediction (tn)

			field_query = ['signal','signal_0','pred','pred_0','signal_pred','signal_pred_0','signal_0_pred','signal_0_pred_0']
			list1 = [id_signal,(~id_signal),id_pred,(~id_pred),id_query1_1,id_query2_1,id_query1_2,id_query2_2]
			list_query1 = [feature_query_vec[id_query] for id_query in list1]
			
		feature_signal_group1, feature_signal_group2, feature_pred_group1, feature_pred_group2, feature_tp, feature_fn, feature_fp, feature_tn = list_query1
		feature_signal_num1 = len(feature_signal_group1) # peak loci with ChIP-seq signal
		feature_pred_num1 = len(feature_pred_group1)	# peak loci with TF binding prediction
		tp_num = len(feature_tp)
		fn_num = len(feature_fn)
		fp_num = len(feature_fp)
		tn_num = len(feature_tn)

		contingency_table = [[tp_num,fn_num],[fp_num,tn_num]]
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

		field_query = ['stat_chi2_','pval_chi2_']+['stat_fisher_exact_','pval_fisher_exact_'] + ['num1','num2']
		t_vec_1 = [stat_chi2_, pval_chi2_]+[stat_fisher_exact_, pval_fisher_exact_] + [feature_signal_num1,feature_pred_num1]
		df_score_query = pd.Series(index=field_query,data=np.asarray(t_vec_1),dtype=np.float32)

		return df_score_query, contingency_table

	## ====================================================
	# TF binding prediction performance evaluation
	def test_query_compare_binding_basic_unit1(self,data1=[],data2=[],motif_id_query='',group_query_vec=[2],df_annot_motif=[],
													dict_method_type=[],df_method_annot=[],column_signal='signal',column_motif='',column_vec_query=[],
													method_type_vec=[],method_type_group='',flag_compare_1=0,mode_query=2,input_file_path='',
													save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		TF binding prediction performance evaluation
		:param data1: (dataframe) TF ChIP-seq signals in ATAC-seq peak loci by overlaping ATAC-seq peaks with ChIP-seq peaks (row:ATAC-seq peak locus: column:TF ChIP-seq dataset identifier)
		:param data2: (dataframe) TF binding predictions in the ATAC-seq peak loci for each given TF
		:param motif_id_query: (str) identifier of the TF ChIP-seq dataset with which we perform TF binding prediction performance evaluation
		:param df_annot_motif: (dataframe) TF annotations including the mapping between the TF ChIP-seq datasets and the TF names and the cell types for which the ChIP-seq experiments were performed;
		:param dict_method_type: dictionary containing annotations of the methods used for initial TF binding prediction
		:param df_method_annot: (dataframe) annotations of the methods used for initial TF binding prediction
		:param column_signal: (str) column for ChIP-seq signals
		:param column_motif: (str) column for the motif presence (binary) or motif scores by motif scanning in the ATAC-seq peak loci
		:param column_vec_query: (list) each element is one of the two types: 
										1. the name of the column which corresonds to predicted TF binding label or probability in the peak loci;
										2. a list or array of the column names which corresponds to predicted TF binding label, TF binding probability, and the TF motif presence by motif scanning (optional) in the peak loci;
		:param method_type_vec: (array or list) the methods used for TF binding prediction
		:param method_type_group: (str) the method used for peak clustering using peak accessibility features and sequence-based features
		:param flag_compare_1: indicator of whether to evaluate prediction performance on peak loci with TF motif or without TF motif
		:param mode_query: indicator of which evaluation metrics to query:
						   0: basic metrics: accuracy, precision, recall, F1, AUROC, AUPR;
						   1: basic metrics and recompute AUPR for the methods which rely on motif scanning results;
						   2: basic metrics with recomputed AUPR, precision at fixed recall, recall at fixed precision,
						   3: the metrics in 2 and enrichment of true positive peak-TF link predictions;
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) TF binding prediction performance evaluation metric scores for the given TF using the specific methods;
		         2. (dataframe) precision and recall at each given threshold on the precision-recall curve of the TF binding prediction using the specific methods;
		"""

		df_signal = data1
		df_pre1 = data2

		if (column_motif=='') or (not (column_motif in df_pre1.columns)):
			flag_compare_1 = 0

		if flag_compare_1==0:
			list_group = [df_pre1]
			group_query_vec_1 = [2]
			group_query_vec = group_query_vec_1
			# query_num_1 = 1
			# print('df_pre1: ',df_pre1.shape)
			print('ATAC-seq peak annotations, dataframe of size ',df_pre1.shape)
		else:
			id1 = (df_pre1[column_motif].abs()>0)	# peak loci with the TF motif
			id2 = (~id1)	# peak loci without the TF motif
			df_query_group1 = df_pre1.loc[id1,:]
			df_query_group2 = df_pre1.loc[id2,:]
			df_query_group_1 = df_pre1
			print('ATAC-seq peak annotations, dataframe of size ',df_pre1.shape)
			print('peak loci with the TF motif: %d'%(df_query_group1.shape[0]))
			print('peak loci without the TF motif: %d'%(df_query_group2.shape[0]))

			list_group = [df_query_group_1,df_query_group1,df_query_group2]
			group_query_vec_1 = [2,1,0]
		
		dict_group_query = dict(zip(group_query_vec_1,list_group))

		group_query_vec_pre1 = [2,1,0]
		group_annot_motif_vec = ['genome-wide peaks','peaks with TF motif','peaks without TF motif']
		dict_group_annot_motif = dict(zip(group_query_vec_pre1,group_annot_motif_vec))
		
		query_num_1 = len(group_query_vec)
		query_num_2 = len(column_vec_query)

		if method_type_group=='':
			method_type_group = select_config['method_type_group']	# the method used for peak clustering

		list_query1 = []
		list_query1_1 = []

		# score_type_2=1: the method for initial TF binding prediction relies on motif scanning and motif scanning data were provided
		score_type_2 = 1
		if 'score_type_2' in select_config:
			score_type_2 = select_config['score_type_2']

		thresh_pred_1 = 0.5
		direction_1 = 1
		if 'thresh_pred' in select_config:
			thresh_pred_vec = select_config['thresh_pred']
		thresh_pred_1, direction_1 = thresh_pred_vec[0:2]
		
		# binary_mode_1=1: there is only binary prediction without feature link score;
		binary_mode_1 = -1
		if 'binary_mode' in select_config:
			binary_mode_1 = select_config['binary_mode']

		if len(method_type_vec)==0:
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_query = method_type_feature_link
			method_type_vec = [method_type_query]

		method_type_vec_query1 = method_type_vec
		method_type_num = len(method_type_vec_query1)

		dict_motif_data = self.dict_motif_data
		motif_load = int(len(dict_motif_data)>0)

		flag_method_annot = (len(df_method_annot)>0)
		if flag_method_annot>0:
			column_1 = 'method_type'
			if column_1 in df_method_annot.columns:
				df_method_annot.index = np.asarray(df_method_annot[column_1])

		# group_query_vec = [2,1,0]
		df_annot_motif.index = np.asarray(df_annot_motif['motif_id2'])
		for i1 in range(query_num_1):
			# df_query_group = list_group[i2]
			group_id_motif = group_query_vec[i1]
			df_query_group = dict_group_query[group_id_motif] # peak annotation with predicted TF binding label and probability and motif presence (optional)
			group_annot_motif = dict_group_annot_motif[group_id_motif]

			# query the ChIP-seq signal
			peak_loc_pre1 = df_query_group.index
			df_query_group[column_signal] = df_signal.loc[peak_loc_pre1,motif_id_query]
			
			print('peak type: %s'%(group_annot_motif))
			print('peak annotations, dataframe of size ',df_query_group.shape)
			print('preview:\n',df_query_group[0:2])

			list_score_query1 = []
			list_score_query2 = []
			list_column_query1 = []
			list_annot_query1 = []
			
			motif_id2 = motif_id_query
			motif_id_ori = df_annot_motif.loc[motif_id2,'motif_id']
			for t_id1 in range(query_num_2):
				column_query_1 = column_vec_query[t_id1] # columns for predicted label, probability, and motif presence (optional)
				method_type_query1 = method_type_vec_query1[t_id1]
				method_type_query = method_type_query1
				
				column_proba = ''  # column for predicted positive class probability or link score;
				column_motif = ''  # column for TF motif presence;

				score_type_query = 0
				column_vec_annot = ['method_type','score_type','binary','motif_depend']
				column_1, column_2, column_3 = column_vec_annot[0:3]
				column_depend = column_vec_annot[3]

				if flag_method_annot==0:
					binary_mode = binary_mode_1
					score_type_2_query = score_type_2
				else:
					column_annot_query = [column_2,column_3,column_depend]+['thresh_pred','direction']
					list1 = [score_type_query,binary_mode_1,score_type_2]+[thresh_pred_1,direction_1] # the default values
					query_num2 = len(column_annot_query)
					for i2 in range(query_num2):
						column_query = column_annot_query[i2]
						if column_query in df_method_annot.columns:
							list1[i2] = df_method_annot.loc[method_type_query,column_query]

					# score_type_query: indicator of whether higher link score corresponds to higher association strength;
					# binary_mode: indicator of whether there is only binary prediction without feature link score;
					# score_type_2_query: indicator of whether the initial TF binding prediction relies on motif scannnig results;
					score_type_query,binary_mode,score_type_2_query,thresh_pred,direction = list1

				# query if column_query_1 is str (for one column) or list (multiple columns)
				if isinstance(column_query_1,str):
					column_pred = column_query_1

					# query if there is predicted positive class probability or link score;
					if binary_mode<0:
						query_value = df_query_group[column_pred].astype(int)
						binary_mode = ((query_value==0)|(query_vaule==1)).all()

					if binary_mode==0:
						column_proba = '%s.score'%(column_pred)
						df_query_group[column_proba] = df_query_group[column_pred].copy() # predicted positive class probability
						if direction>0:
							df_query_group[column_pred] = (df_query_group[column_proba]>thresh_pred) # query predicted class label
						else:
							df_query_group[column_pred] = (df_query_group[column_proba]<thresh_pred) # query predicted class label
				else:
					column_pred = column_query_1[0]
					if len(column_query_1)>1:
						# column_pred, column_proba = column_query_1[0:2]
						column_proba = column_query_1[1]
						if (score_type_2_query>0) and (len(column_query_1)>2):
							column_motif = column_query_1[2]

				print('column_signal, column_pred, column_proba ',column_signal,column_pred,column_proba,t_id1)
				column_vec = [column_signal,column_pred,column_proba]

				print('score_type: %d, binary: %d, score_type_2: %d, thresh_pred: %s, direction: %d'%(score_type_query,binary_mode,score_type_2_query,thresh_pred,direction))

				if score_type_2_query>0:
					if column_motif=='':
						# the method for initial TF binding prediction relies on motif scanning results;
						# query TF motif presence in peak loci from motif scanning data;
						column_motif = '%s.motif'%(column_pred)

						# load motif scanning data if the method for initial TF binding prediction relies on motif scanning and motif scanning data were not provided;
						if motif_load==0:
							flag_motif_data_load_1= 1
							flag_load_1 = 0
							select_config = self.test_query_load_pre1(method_type_vec=method_type_vec_query1,
																			flag_motif_data_load_1=flag_motif_data_load_1,
																			flag_load_1=flag_load_1,
																			save_mode=1,
																			verbose=verbose,select_config=select_config)

							dict_motif_data = self.dict_motif_data
							motif_load = 1

						motif_data = dict_motif_data[method_type_query1]['motif_data']
						df_query_group[column_motif] = (motif_data.loc[peak_loc_pre1,motif_id_ori]>0).astype(int)

					column_vec = column_vec + [column_motif]
					print('column_motif ',column_motif)

				# query peak-TF link estimation performance score
				print('method_type: %s, score_type: %d'%(method_type_query,score_type_query))

				column_2 = 'dict_annot_score'
				dict_annot_score = {'method_type':method_type_query,'motif_id':motif_id_query}
				select_config.update({column_2:dict_annot_score})

				t_vec_1 = self.test_query_pred_score_1(data=df_query_group,column_vec=column_vec,
														method_type=method_type_query,
														score_type=score_type_query,
														mode_query=mode_query,
														default_value=1,
														verbose=verbose,select_config=select_config)
				
				df_score_1, contingency_table, dict_query1 = t_vec_1

				list_score_query1.append(df_score_1)
				list_column_query1.append(column_pred)
				list_annot_query1.append(method_type_query)

				column_1 = 'df_precision_recall'
				flag_score_2 = (mode_query>=2)&(column_1 in dict_query1)
				if (column_1 in dict_query1):
					df_precision_recall = dict_query1[column_1]
					column_2 = 'method_type'
					if not (column_2 in df_precision_recall.columns):
						df_precision_recall[column_2] = method_type_query
					list_score_query2.append(df_precision_recall)

			# query prediction score
			query_num1 = len(list_score_query1)
			if query_num1>0:
				df_score_query = pd.concat(list_score_query1,axis=1,join='outer',ignore_index=False)
				df_score_query = df_score_query.T
				column_vec_query_1 = np.asarray(list_column_query1)
				method_type_vec_query = np.asarray(list_annot_query1)
				
				field_query_2 = ['motif_id','motif_id2','group_motif','method_type','method_type_group','query_id']
				
				list_2 = [motif_id_ori,motif_id2,group_id_motif,method_type_vec_query,method_type_group,column_vec_query_1]

				for (field_id1,query_value) in zip(field_query_2,list_2):
					df_score_query[field_id1] = query_value
				list_query1.append(df_score_query)

			# query precision and recall
			if len(list_score_query2)>0:
				df_precision_recall_query = pd.concat(list_score_query2,axis=0,join='outer',ignore_index=False)
				field_query_pre2 = ['motif_id','motif_id2','group_motif']
				list_pre2 = [motif_id_ori,motif_id2,group_id_motif]
				for (field_id,query_value) in zip(field_query_pre2,list_pre2):
					df_precision_recall_query[field_id] = query_value
				list_query1_1.append(df_precision_recall_query)

		df_score_query_1 = []
		if len(list_query1)>0:
			df_score_query_1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)

		df_score_query_2 = []
		if len(list_query1_1)>0:
			df_score_query_2 = pd.concat(list_query1_1,axis=0,join='outer',ignore_index=False)

		return df_score_query_1, df_score_query_2

	## ====================================================
	# TF binding prediction performance evaluation
	def test_query_compare_binding_basic_1(self,data1=[],data2=[],motif_query_vec=[],df_annot_motif=[],dict_method_type=[],df_method_annot=[],
												column_vec_query=[],column_signal='signal',column_motif='',method_type_vec=[],method_type_group='',
												flag_compare_1=0,mode_query=2,parallel=0,input_file_path='',
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		TF binding prediction performance evaluation
		:param data1: (dataframe) TF ChIP-seq signals in ATAC-seq peak loci by overlaping ATAC-seq peaks with ChIP-seq peaks 
								  (row:ATAC-seq peak locus: column:TF ChIP-seq dataset identifier)
		:param data2: (dataframe) TF binding predictions in the ATAC-seq peak loci for each given TF
		:param motif_query_vec: (array or list) identifiers of the TF ChIP-seq datasets with which we perform TF binding prediction performance evaluation
		:param df_annot_motif: (dataframe) TF annotations including the mapping between the TF ChIP-seq datasets and the TF names 
										   and the cell types for which the ChIP-seq experiments were performed;
		:param dict_method_type: dictionary containing annotations of the methods used for initial TF binding prediction
		:param df_method_annot: (dataframe) annotations of the methods used for initial TF binding prediction
		:param column_vec_query: (list) each element is one of the two types: 
										1. the name of the column which corresonds to predicted TF binding label or probability in the peak loci;
										2. a list or array of the names of columns which corresponds to predicted TF binding label, TF binding probability, 
										   and TF motif presence by motif scanning (optional) in the peak loci;
		:param column_signal: (str) column for ChIP-seq signals
		:param column_motif: (str) column for the motif presence (binary) or motif scores by motif scanning in the ATAC-seq peak loci
		:param method_type_vec: (array or list) the methods used for TF binding prediction
		:param method_type_group: (str) the method used for peak clustering using peak accessibility features and sequence-based features
		:param flag_compare_1: indicator of whether to evaluate prediction performance on peak loci with TF motif or without TF motif
		:param mode_query: indicator of which evaluation metrics to query:
						   0: basic metrics: accuracy, precision, recall, F1, AUROC, AUPR;
						   1: basic metrics and recompute AUPR for the methods which rely on motif scanning results;
						   2: basic metrics with recomputed AUPR, precision at fixed recall, recall at fixed precision,
						   3: the metrics in 2 and enrichment of true positive peak-TF link predictions;
		:param parallel: indicator of whether to perform TF binding prediction evaluation for the given TFs in parallel
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) TF binding prediction performance evaluation metric scores for the given TFs using the specific methods
		"""

		data_signal = data1
		data_pred_query = data2
		motif_query_num = len(motif_query_vec)

		type_annot_1 = (type(data1) is pd.DataFrame)
		type_annot_2 = (type(data2) is pd.DataFrame)

		if method_type_group=='':
			column_1 = 'method_type_group'
			if column_1 in select_config:
				method_type_group = select_config[column_1]

		query_res = []
		if parallel==0:
			for i1 in range(motif_query_num):
				motif_query = motif_query_vec[i1] # motif_id2_query
				if (motif_query in data_signal) and (motif_query in data_pred_query):

					list1 = []
					for (data_query_pre1,type_annot_query) in zip([data_signal,data_pred_query],[type_annot_1,type_annot_2]):
						if type_annot_query>0:
							df_query = data_query_pre1.loc[:,[motif_query]]  # query data from dataframe
						else:
							df_query = data_query_pre1[motif_query]  # query data from dictionary
						list1.append(df_query)
					df_signal_query, df_pred_query = list1[0:2]
					print('df_signal_query, ',df_signal_query.shape,motif_query)
					print(df_signal_query[0:2])
					print('df_pred_query, ',df_pred_query.shape,motif_query)
					print(df_pred_query[0:2])
					t_query_res = self.test_query_compare_binding_basic_unit1(data1=df_signal_query,data2=df_pred_query,
																				motif_id_query=motif_query,
																				df_annot_motif=df_annot_motif,
																				df_method_annot=df_method_annot,
																				column_vec_query=column_vec_query,
																				method_type_vec=method_type_vec,
																				method_type_group=method_type_group,
																				mode_query=mode_query,
																				select_config=select_config)

					if len(t_query_res[0])>0:
						query_res.append(t_query_res)
				else:
					print('motif query not included ',motif_query)
		else:
			query_res_local = Parallel(n_jobs=-1)(delayed(self.test_query_compare_binding_basic_unit1)(data1=data_signal[motif_query],
																										data2=data_pred_query[motif_query],
																										motif_id_query=motif_query,
																										df_annot_motif=df_annot_motif,
																										df_method_annot=df_method_annot,
																										column_vec_query=column_vec_query,
																										method_type_group=method_type_group,
																										type_id_1=type_id_1,
																										mode_query=mode_query,
																										select_config=select_config) for motif_query in tqdm(motif_query_vec))

			for t_query_res in query_res_local:
				# dict_query = t_query_res
				if len(t_query_res[0])>0:
					query_res.append(t_query_res)

		# list_score_query_1 = query_res
		list_score_query_1 = [t_query_res[0] for t_query_res in query_res]
		df_score_query_1 = []
		if len(list_score_query_1)>0:
			df_score_query_1 = pd.concat(list_score_query_1,axis=0,join='outer',ignore_index=False)
			if (save_mode>0) and (output_filename!=''):
				df_score_query_1.to_csv(output_filename,sep='\t')

		return df_score_query_1

	## ====================================================
	# query estimated peak-TF associations
	def test_query_feature_link_2(self,dict_file={},df_annot=[],feature_query_vec=[],type_query=0,append=True,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
	
		"""
		query estimated peak-TF associations
		:param dict_file: dictionary containing paths of the files or the dataframes of TF binding predictions for each given TF
		:param df_annot: (dataframe) TF annotations including the mapping between the TF ChIP-seq datasets and the TF names
		:param feature_query_vec: (array) names of TFs for which we perform TF binding prediction
		:param type_query: indicator of what data to load:
						   0: query paths of the files which saved the TF binding predictions for each given TF;
						   1: query TF binding predictions for each given TF;
		:param append: indicator of whether to add paths of files of TF binding predictions to dict_file
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing paths of the files or the dataframes of TF binding predictions for each given TF;
				 2. dictionary containing dataframes of TF binding predictions for each given TF;
		"""

		# load the peak-TF association estimation
		# dict_file_query = data
		motif_idvec_query = feature_query_vec
		motif_query_num1 = len(motif_idvec_query)

		dict_file_query = dict_file
		dict_query_1 = []
		if type_query in [0,1]:
			if (len(dict_file_query)==0) or (append>0):
				field_query = ['path_save_link','filename_prefix_pred','filename_annot_pred']
				list_query1 = [select_config[field_id] for field_id in field_query]
				input_file_path_query,filename_prefix_1,filename_save_annot1 = list_query1[0:3]

				for i1 in range(motif_query_num1):
					motif_id = motif_idvec_query[i1]
					motif_id_query = motif_id
					input_filename = '%s/%s.%s.%s.txt'%(input_file_path_query,filename_prefix_1,motif_id_query,filename_save_annot1)
					
					if os.path.exists(input_filename)==False:
						print('the file does not exist: %s'%(input_filename))
						continue
					dict_file_query.update({motif_id:input_filename})
		
			if type_query in [1]:
				dict_query_1 = self.test_query_feature_link_load_1(dict_file=dict_file_query,
																	df_annot=df_annot,
																	feature_query_vec=feature_query_vec,
																	save_mode=save_mode,
																	verbose=verbose,select_config=select_config)

		return dict_file_query, dict_query_1

	## ====================================================
	# query estimated peak-TF associations
	def test_query_feature_link_load_1(self,dict_file={},df_annot=[],feature_query_vec=[],load_mode=1,save_mode=1,verbose=0,select_config={}):

		"""
		query estimated peak-TF associations
		:param dict_file: dictionary containing paths of the files or the dataframes of TF binding predictions for each given TF
		:param df_annot: (dataframe) TF annotations including the mapping between the TF ChIP-seq datasets and the TF names
		:param feature_query_vec: (array) names of TFs for which we perform TF binding prediction
		:param load_mode: indicator of whether to retrieve TF binding predictions from the saved files or from the dictionary dict_file for the given TFs
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing dataframes of TF binding predictions in the peak loci for each given TF
		"""

		dict_file_query = dict_file
		df_annot_1 = df_annot
		motif_query_vec_1 = feature_query_vec
		motif_query_num1 = len(feature_query_vec)
		dict_query_1 = dict()
		verbose_internal = self.verbose_internal

		for i1 in range(motif_query_num1):
			motif_id = motif_query_vec_1[i1]
			motif_id_query = motif_id
			
			motif_id2_query = df_annot_1.loc[df_annot_1['motif_id']==motif_id,'motif_id2']
			motif_id2_num = len(motif_id2_query)
			print('motif_id, motif_id2: ',motif_id,motif_id2_query,motif_id2_num,i1)

			if load_mode>0:
				if not (motif_id_query in dict_file_query):
					print('the estimation not included: ',motif_id2_query,i1)
					continue

				query_value = dict_file_query[motif_id_query]
				if isinstance(query_value,str):
					input_filename = query_value
					if os.path.exists(input_filename)==False:
						print('the file does not exist: ',input_filename)
						continue

					df_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t') # load peak-TF link prediction
					if verbose_internal>0:
						print('load peak-TF link estimation from %s'%(input_filename))
				else:
					df_query_1 = query_value

				df_query_1 = df_query_1.loc[(~df_query_1.index.duplicated(keep='first')),:]
				
				peak_loc_pre1 = df_query_1.index
				peak_num = len(peak_loc_pre1)
				if verbose_internal>0:
					print('peak-TF link estimation, dataframe of size ',df_query_1.shape)
					print('preview: ')
					print(df_query_1[0:2])
					print('peak number: %d'%(peak_num))

				for motif_id2 in motif_id2_query:
					dict_query_1.update({motif_id2:df_query_1})

		return dict_query_1

	## ====================================================
	# TF binding prediction performance evaluation
	def test_query_compare_binding_pre1(self,data=[],dict_signal={},df_signal=[],dict_file_pred={},feature_query_vec=[],method_type_vec=[],type_query_format=0,
											save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		TF binding prediction performance evaluation
		:param dict_signal: dictionary containing annotation dataframes of ATAC-seq peak loci overlapping with ChIP-seq peak loci for each given TF ChIP-seq dataset
		:param df_signal: (dataframe) annotations of ATAC-seq peak loci overlapping with ChIP-seq peak loci for the given TF ChIP-seq datasets (row:ATAC-seq peak locus, column:ChIP-seq dataset)
		:param dict_file_pred: dictionary containing paths of the files or the dataframes of TF binding predictions for each given TF
		:param feature_query_vec: (array or list) names of TFs for which we perform TF binding prediction
		:param method_type_vec: (array or list) the methods used for TF binding prediction
		:param type_query_format: the type of comparison between ATAC-seq peak loci and ChIP-seq peak loci:
								  0: compare each ATAC-seq peak locus to ChIP-seq peak loci to search for overlaps;
								  1: compare each ChIP-seq peak locus to ATAC-seq peak loci to search for overlaps;
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) TF binding prediction performance evaluation metric scores for the given TFs using the specific methods
		"""

		data_file_type_query = select_config['data_file_type']
		select_config.update({'data_file_type_query':data_file_type_query})

		method_type_feature_link = select_config['method_type_feature_link']
		method_type_group = select_config['method_type_group']

		load_mode_signal = 0
		if len(dict_signal)>0:
			load_mode_signal = 1
		elif len(df_signal)>0:
			load_mode_signal = 2

		# test_query_file_annotation_load_1(self,data_file_type='',input_filename='',save_mode=1,verbose=0,select_config={}):
		input_dir = select_config['input_dir']
		file_path_1 = '%s/bed_file'%(input_dir)
		file_path_query1 = '%s/folder1'%(file_path_1)
		file_path_query2 = '%s/folder2'%(file_path_1)
		select_config.update({'file_path_annot_1':file_path_query1,
								'file_path_annot_2':file_path_query2})

		input_filename = '%s/df_annotation_pre2.txt'%(file_path_query1)
		select_config.update({'file_motif_annot':input_filename})
		celltype_vec = ['B_cell','T_cell','macrophage','monocyte']
		t_vec_1 = self.test_query_file_annotation_load_1(data_file_type=data_file_type_query,
															input_filename=input_filename,
															celltype_vec=celltype_vec,
															save_mode=1,
															verbose=0,select_config=select_config)

		df_annot_motif, motif_idvec_query, input_filename_list, feature_query_vec_2 = t_vec_1
		column_vec = feature_query_vec_2

		output_filename = '%s/df_annotation_pre2.copy1.txt'%(file_path_query1)
		df_annot_motif.to_csv(output_filename,sep='\t')
		# print('TF annotations, dataframe of size ',df_annot_motif.shape)
		# print('data preview: ')
		# print(df_annot_motif[0:2])
		
		input_filename_peak = select_config['input_filename_peak']
		df_peak_annot = pd.read_csv(input_filename_peak,index_col=False,header=None,sep='\t')
		column_vec_1 = ['chrom','start','stop','feature','GC','strand']  # six column annotation
		df_peak_annot.columns = column_vec_1
		df_peak_annot.index = utility_1.test_query_index(df_peak_annot,column_vec=['chrom','start','stop'],symbol_vec=[':','-'])
		df_peak_annot = df_peak_annot.loc[(~df_peak_annot.index.duplicated(keep='first')),:]
		print('ATAC-seq peak loci annotations, dataframe of size ',df_peak_annot.shape)
		peak_query_vec = df_peak_annot.index  # ATAC-seq peak loci

		output_dir = select_config['output_dir']
		output_file_path_1 = output_dir

		input_filename = '%s/test_query_signal_1.txt'%(file_path_1)
		overwrite_query1 = False
		if load_mode_signal==0:
			if (os.path.exists(input_filename)==False) or (overwrite_query1==True):
				print('query TF ChIP-seq data annotations')
				type_query_format = 0
				output_filename = input_filename
				df_signal_query1, dict_signal_query1 = self.test_query_signal_overlap_format_1(input_filename='',
																								input_filename_list=input_filename_list,
																								feature_query='',
																								feature_query_vec=feature_query_vec,
																								peak_query_vec=peak_query_vec,
																								column_vec=column_vec,
																								type_query=type_query_format,
																								save_mode=1,
																								output_filename=output_filename,
																								filename_prefix_save='',
																								filename_save_annot='annot',
																								verbose=0,select_config=select_config)

				df_signal = df_signal_query1
				dict_signal = dict_signal_query1
				load_mode_signal = 2
			else:
				print('the file exists: %s'%(input_filename))
				df_signal = pd.read_csv(input_filename,index_col=0,sep='\t')
				dict_signal = dict()
				peak_loc_ori = df_signal.index
				peak_num_1 = len(peak_loc_ori)
				column_vec_query = df_signal.columns
				feature_vec_2 = column_vec_query
				column_vec_2 = ['peak_num','ratio','F1']
				df_annot_2 = pd.DataFrame(index=feature_vec_2,columns=column_vec_2)
				for column_query in column_vec_query:
					feature_query = column_query
					df_query = df_signal[[feature_query]]
					peak_num = np.sum(df_query[feature_query]>0)
					ratio_1 = peak_num/peak_num_1
					precision_1 = ratio_1
					recall_1 = 1.0
					f1_score = 2*precision_1*recall_1/(precision_1+recall_1)
					df_annot_2.loc[feature_query,column_vec_2] = [peak_num,ratio_1,f1_score]
					dict_signal.update({feature_query:df_query})
					print('feature_query: %s, peak loci with signal: %d'%(feature_query,peak_num))

				output_filename = '%s/test_query_signal_1.annot1.txt'%(file_path_1)
				df_annot_2.to_csv(output_filename,sep='\t')
			# return

		df_method_annot = []
		filename_annot_query = '%s/test_query_method_annot.1.txt'%(file_path_1)
		overwrite_2 = True
		if (os.path.exists(filename_annot_query)==False) or (overwrite_2==True):
			score_type_query = 0
			column_vec_annot = ['method_type','score_type','binary','motif_depend']
			column_1, column_2, column_3 = column_vec_annot[0:3]
			column_depend = column_vec_annot[3]
			method_type_vec_pre1_1 = ['CIS-BP','HOCOMOCO','Pando_motif','JASPAR','insilico_0.1','GRaNIE','Pando','TRIPOD','Unify','TOBIAS','SCENIC+','REUNION']			
			method_type_vec_pre1_2 = ['CIS-BP','insilico_0.1']
			method_type_vec_pre1_3 = ['%s + Rediscover'%(method_type) for method_type in method_type_vec_pre1_2]
			method_type_vec_pre1 = method_type_vec_pre1_1 + method_type_vec_pre1_3
			# score_type_vec = [0,1,1,1,0,0,0,0]
			method_type_vec_query2 = ['GRaNIE','Pando','TRIPOD'] # methods by which the lower peak-TF link scores correspond to higher association strength
			method_type_vec_query3 = ['HOCOMOCO','Pando_motif','JASPAR','SCENIC+'] # methods for which continuous peak-TF link scores were not in the output
			method_type_vec_query5 = ['CIS-BP','REUNION'] + method_type_vec_pre1_3
			df_method_annot = pd.DataFrame(index=method_type_vec_pre1,columns=column_vec_annot)
			df_method_annot[column_1] = np.asarray(method_type_vec_pre1)
			df_method_annot.loc[:,[column_2,column_3]] = 0
			df_method_annot[column_depend] = 1

			df_method_annot.loc[method_type_vec_query2,column_2] = 1
			df_method_annot.loc[method_type_vec_query3,column_3] = 1
			df_method_annot.loc[method_type_vec_query5,column_depend] = 0

			output_filename = '%s/test_query_method_annot.1.txt'%(file_path_query1)
			df_method_annot.to_csv(output_filename,sep='\t')
		else:
			print('the file exists: %s'%(filename_annot_query))
			df_method_annot = pd.read_csv(filename_annot_query,index_col=0,sep='\t')
		print('method type annotations, dataframe of size ',df_method_annot.shape)
		print(df_method_annot)

		# score query for performance comparison
		start = time.time()
		flag_compare_1=0
		parallel = 0
		# parallel = 1
		column_signal = 'signal'
		column_motif = '%s.motif'%(method_type_feature_link)
		method_type_group = select_config['method_type_group']
		mode_query = 2

		motif_query_vec = feature_query_vec_2
		method_type_vec_query1 = [method_type_feature_link]
		df_score_query_1 = self.test_query_compare_binding_basic_2(motif_query_vec=motif_query_vec,
																	dict_signal=dict_signal,
																	dict_file_pred=dict_file_pred,
																	method_type_vec=method_type_vec_query1,
																	df_annot_motif=df_annot_motif,
																	df_method_annot=df_method_annot,
																	column_signal=column_signal,
																	column_motif=column_motif,
																	method_type_group=method_type_group,
																	flag_compare_1=flag_compare_1,
																	mode_query=mode_query,
																	parallel=parallel,
																	input_file_path='',
																	save_mode=1,
																	output_file_path='',
																	output_filename='',
																	filename_prefix_save='',
																	filename_save_annot='',
																	verbose=verbose,select_config=select_config)


		stop = time.time()
		print('performance comparison used %.2fs'%(stop-start))

		output_file_path = '%s/folder_save_2'%(output_file_path_1)
		if (os.path.exists(output_file_path)==False):
			print('the directory does not exist: %s'%(output_file_path))
			os.makedirs(output_file_path,exist_ok=True)
		if save_mode>0:
			if output_file_path=='':
				output_file_path = file_path_query1
			
			run_id2 = '1'
			filename_save_annot_1 = '%s.%s'%(data_file_type_query,run_id2)
			float_format = '%.6E'
			output_filename = '%s/test_query_df_score.beta.%s.1.txt'%(output_file_path,filename_save_annot_1)
			df_score_query_1.to_csv(output_filename,sep='\t',float_format=float_format)
			print('save data: ',output_filename)

		return df_score_query_1


	## ====================================================
	# TF binding prediction performance evaluation
	def test_query_compare_binding_basic_2(self,motif_query_vec=[],dict_signal={},dict_file_pred={},method_type_vec=[],df_annot_motif=[],df_method_annot=[],
												column_signal='signal',column_motif='',method_type_group='',
												flag_compare_1=0,mode_query=2,parallel=0,input_file_path='',
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		TF binding prediction performance evaluation
		:param motif_query_vec: (array or list) identifiers of the TF ChIP-seq datasets with which we perform TF binding prediction performance evaluation
		:param dict_signal: dictionary containing annotation dataframes of ATAC-seq peak loci overlapping with ChIP-seq peak loci for each given TF ChIP-seq dataset
		:param dict_file_pred: dictionary containing paths of the files or the dataframes of TF binding predictions for each given TF
		:param method_type_vec: (array or list) the methods used for TF binding prediction
		:param df_annot_motif: (dataframe) TF annotations including the mapping between the TF ChIP-seq datasets and the TF names 
										   and the cell types for which the ChIP-seq experiments were performed;
		:param df_method_annot: (dataframe) annotations of the methods used for initial TF binding prediction
		:param column_signal: (str) column for ChIP-seq signals
		:param column_motif: (str) column for the motif presence (binary) or motif scores by motif scanning in the ATAC-seq peak loci
		:param method_type_group: (str) the method used for peak clustering using peak accessibility features and sequence-based features
		:param flag_compare_1: indicator of whether to evaluate prediction performance on peak loci with TF motif or without TF motif
		:param mode_query: indicator of which evaluation metrics to query:
						   0: basic metrics: accuracy, precision, recall, F1, AUROC, AUPR;
						   1: basic metrics and recompute AUPR for the methods which rely on motif scanning results;
						   2: basic metrics with recomputed AUPR, precision at fixed recall, recall at fixed precision,
						   3: the metrics in 2 and enrichment of true positive peak-TF link predictions;
		:param parallel: indicator of whether to perform TF binding prediction evaluation for the given TFs in parallel
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) TF binding prediction performance evaluation metric scores for the given TFs using the specific methods
		"""

		# query the column of the prediction
		model_type_id1 = 'LogisticRegression'
		column_1 = 'model_type_id1'
		if column_1 in select_config:
			model_type_id1 = select_config[column_1]

		if len(method_type_vec)>0:
			method_type_feature_link = method_type_vec[0]
		else:
			method_type_feature_link = select_config['method_type_feature_link']

		method_type_query1 = method_type_feature_link
		method_type_query2 = 'latent_%s_%s_combine_%s'%(feature_type_1_ori,feature_type_2_ori,model_type_id1)

		method_type_annot1 = method_type_query1
		if method_type_query1 in ['Unify']:
			method_type_annot2 = 'REUNION'
		else:
			method_type_annot2 = '%s + Rediscover'%(method_type_query1)

		annot_str_vec = ['pred','proba_1']
		# t_vec_1 = ['%s.pred'%(method_type_query1),[]]
		t_vec_1 = ['%s.pred'%(method_type_query1),'%s.score'%(method_type_query1)]
		t_vec_2 = ['%s_%s'%(method_type_query2,annot_str1) for annot_str1 in annot_str_vec]
		t_vec_1 = t_vec_1 + [column_motif]
		column_vec_query = [t_vec_1,t_vec_2]

		query_num1 = len(column_vec_query)
		for i1 in range(query_num1):
			t_vec_query = column_vec_query[i1]
			print('columns for method %d: '%(i1+1),t_vec_query)

		method_type_vec_query1 = [query_vec[0] for query_vec in column_vec_query]
		method_type_vec_annot1 = [method_type_annot1,method_type_annot2]
		method_type_vec = method_type_vec_annot1
		dict_method_type = dict(zip(method_type_vec_query1,method_type_vec_annot1))

		verbose_internal = self.verbose_internal
		if verbose_internal==2:
			print('method type annotation: ')
			print(dict_method_type)
		
		start = time.time()
		dict_file_query_1 = dict_file_pred

		# load the peak-TF association estimation
		output_dir = select_config['output_dir']
		field_query = ['path_save_link','filename_prefix_pred','filename_annot_pred']
		path_save_link = '%s/file_link'%(output_dir)

		filename_prefix = select_config['filename_prefix']
		filename_annot = select_config['filename_annot']

		filename_prefix_pred = filename_prefix
		filename_annot_pred = '%s.pred2'%(filename_annot)
		list_query1 = [path_save_link,filename_prefix_pred,filename_annot_pred]

		from utility_1 import test_query_default_parameter_1
		overwrite_query = False
		select_config, list_query1 = test_query_default_parameter_1(field_query=field_query,
																	default_parameter=list_query1,
																	overwrite=overwrite_query,
																	select_config=select_config)

		dict_file_query, dict_query_1 = self.test_query_feature_link_2(dict_file=dict_file_query_1,
																			df_annot=df_annot_motif,
																			feature_query_vec=motif_query_vec_1,
																			type_query=1,
																			append=True,
																			save_mode=1,
																			verbose=verbose,select_config=select_config)

		# score query for performance comparison
		start = time.time()
		flag_compare_1=0
		parallel = 0
		# parallel = 1
		list1 = []
		dict_query_1 = dict()
		dict_signal_1 = dict_signal
		method_type_group = select_config['method_type_group']
		mode_query = 2

		thresh_pred = 0.5 # threshold on predicted positive class probablity or link score for binary prediction
		direction = 1
		select_config.update({'thresh_pred':[thresh_pred,direction]})
		method_type_vec = method_type_vec_annot1
		df_score_query_1 = self.test_query_compare_binding_basic_1(data1=dict_signal_1,
																	data2=dict_query_1,
																	motif_query_vec=motif_query_vec,
																	df_annot_motif=df_annot_motif,
																	df_method_annot=df_method_annot,
																	column_vec_query=column_vec_query,
																	column_signal=column_signal,
																	method_type_vec=method_type_vec,
																	method_type_group=method_type_group,
																	flag_compare_1=flag_compare_1,
																	mode_query=mode_query,
																	parallel=parallel,
																	input_file_path='',
																	save_mode=1,
																	output_file_path='',
																	output_filename='',
																	filename_prefix_save='',
																	filename_save_annot='',
																	verbose=verbose,select_config=select_config)

		return df_score_query_1

	## ====================================================
	# prepare annotations of ATAC-seq peak loci overlapping with TF ChIP-seq signal peak loci
	def test_query_signal_overlap_format_1(self,input_filename='',input_filename_list=[],feature_query='',feature_query_vec=[],peak_query_vec=[],column_vec=[],type_query=0,save_mode=1,output_filename='',filename_prefix_save='',filename_save_annot='annot',verbose=0,select_config={}):

		"""
		prepare annotations of ATAC-seq peak loci overlapping with TF ChIP-seq signal peak loci
		:param input_filename: (str) path of the file of annotations of ATAC-seq peaks overlapping with ChIP-seq peaks for a TF ChIP-seq dataset
		:param input_filename_list: (list) file paths of annotations of ATAC-seq peaks overlapping with ChIP-seq peaks for TF ChIP-seq datasets
		:param feature_query: (str) the identifier of a TF ChIP-seq dataset or the column presenting ChIP-seq signal in the ATAC-seq peak annotations
		:param feature_query_vec: (list) the identifiers of TF ChIP-seq datasets
		:param peak_query_vec: (array) ATAC-seq peak loci for which we perform TF binding prediction for the given TF
		:param column_vec: (array or list) columns in the peak annotation dataframe which correspond to ChIP-seq signals from TF ChIP-seq datasets
		:param type_query: the type of comparison between ATAC-seq peak loci and ChIP-seq peak loci:
						   0: compare each ATAC-seq peak locus to ChIP-seq peak loci to search for overlaps;
						   1: compare each ChIP-seq peak locus to ATAC-seq peak loci to search for overlaps;
		:param save_mode: indicator of whether to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) annotations of ATAC-seq peak loci overlapping with ChIP-seq peak loci for the given TF ChIP-seq datasets (row:ATAC-seq peak locus, column:ChIP-seq dataset);
				 2. dictionary containing annotation dataframes of ATAC-seq peak loci overlapping with ChIP-seq peak loci for each given TF ChIP-seq dataset;
		"""

		flag1 = 1
		if flag1>0:
			df_query_1 = []
			if len(column_vec)==0:
				if len(feature_query_vec)==0:
					if feature_query=='':
						column_signal = 'signal'
						column_vec = [column_signal]
					else:
						feature_query_vec = [feature_query]
						column_vec = feature_query_vec
				else:
					column_vec = feature_query_vec

			if len(input_filename_list)==0:
				if (input_filename==''):
					print('please provide the annotation of ATAC-seq peaks overlapping with ChIP-seq peaks')
					return df_query_1

				if (os.path.exists(input_filename)==False):
					print('the file does not exist: %s'%(input_filename))
					return df_query_1

				input_filename_list = [input_filename]

			# the annotation of ATAC-seq peaks overlapping with ChIP-seq peaks
			df_query_1 = pd.DataFrame(index=peak_query_vec,columns=column_vec,dtype=np.float32)
			df_query_default = []

			column_vec_1 = ['chrom','start','stop','feature','GC','strand']  # six column annotation
			column_vec_2_1 = ['chrom','start','stop','feature']  # basic annotation
			column_vec_2_1 = ['%s_2'%(column_query) for column_query in column_vec_2_1]
			column_vec_2_2 = column_vec_2_1 + ['score']  # five column annotation
			column_vec_2_3 = column_vec_2_1 + ['score_1','strand_2','score','value','score_2','shift'] # ten column annotation

			query_num_1 = len(column_vec_1)+len(column_vec_2_1)
			query_num_2_2 = query_num_1+1  # the number of columns using five column annotation of ChIP-seq peaks
			query_num_2_3 = query_num_1+6  # the number of columns using ten column annotation of ChIP-seq peaks

			file_num1 = len(input_filename_list)
			feature_query_num1 = len(feature_query_vec)
			dict_signal_1 = dict()
			for i1 in range(file_num1):
				input_filename_query = input_filename_list[i1]
				feature_query1 = feature_query_vec[i1]
				column_query1 = column_vec[i1]

				if os.path.exists(input_filename_query)==False:
					print('the file does not exist: %s'%(input_filename))
					return df_query_default

				df1 = pd.read_csv(input_filename_query,index_col=False,header=None,sep='\t')
				column_vec_query1 = df1.columns
				query_num1 = len(column_vec_query1)
				print('load data from %s'%(input_filename_query))
				print('annotation of ATAC-seq peaks overlapping with ChIP-seq peaks for TF %s, dataframe of size '%(feature_query1),df1.shape)
				
				if (query_num1==query_num_1):
					column_vec_pre2 = column_vec_2_1
				elif (query_num1==query_num_2_3):
					column_vec_pre2 = column_vec_2_3
				else:
					column_vec_pre2 = column_vec_2_2

				# the new column names
				if type_query==0:
					# compare each ATAC-seq peak locus to ChIP-seq peak loci to search for overlaps
					column_vec_query2 = column_vec_1 + column_vec_pre2
				else:
					# compare each ChIP-seq peak locus to ATAC-seq peak loci to search for overlaps
					column_vec_query2 = column_vec_pre2 + column_vec_1
				query_num2 = len(column_vec_query2)

				column_vec_query1_1 = column_vec_query1[0:query_num2]
				column_vec_query1_2 = column_vec_query1[query_num2:]
				dict1 = dict(zip(column_vec_query1_1,column_vec_query2))

				# rename the columns
				df1 = df1.rename(columns=dict1)
				print('annotation data, dataframe of size ',df1.shape,column_query1)
				print('data preview: ')
				print(df1[0:2])

				column_score = 'score'
				column_1, column_2 = 'name', 'name_2'
				df1[column_1] = utility_1.test_query_index(df1,column_vec=['chrom','start','stop'],symbol_vec=[':','-'])
				# df1[column_2] = utility_1.test_query_index(column_vec=['chrom_2','start_2','stop_2'],symbol_vec=[':','-'])
				df1 = df1.sort_values(by=[column_1,column_score],ascending=[True,False])
				# df1.index = np.asarray(df1[column_1])
				# df2 = df1.loc[~df1.index.duplicated(keep='first'),:] 
				df2 = df1.drop_duplicates(subset=[column_1])

				df2.index = np.asarray(df2[column_1])
				peak_vec = df2.index
				df_query_1.loc[peak_vec,column_query1] = df2[column_score]
				# dict_signal_1.update({feature_query1:df2})
				dict_signal_1.update({feature_query1:df_query_1[[column_query1]]})

				print('unduplicated annotation data, dataframe of size ',df2.shape,column_query1)
				print('data preview: ')
				print(df2[0:2])

			print('annotation of ATAC-seq peaks overlapping with ChIP-seq peaks for the given TFs, dataframe of size ',df_query_1.shape)
			print('data preview:')
			print(df_query_1[0:2])

			if (save_mode>0) and (output_filename!=''):
				float_format = '%.6f'
				df_query_1.to_csv(output_filename,sep='\t',float_format=float_format)

			df_signal_1 = df_query_1
			return df_signal_1, dict_signal_1

	def run_pre1(self,chromosome='1',run_id=1,species='human',cell=0,generate=1,chromvec=[],testchromvec=[],metacell_num=500,peak_distance_thresh=100,
						highly_variable=1,upstream=100,downstream=100,type_id_query=1,thresh_fdr_peak_tf=0.2,path_id=2,save=1,type_group=0,type_group_2=0,type_group_load_mode=0,
						method_type_group='phenograph.20',thresh_size_group=50,thresh_score_group_1=0.15,method_type_feature_link='joint_score_pre1.thresh3',neighbor_num=30,model_type_id='XGBClassifier',typeid2=0,folder_id=1,
						config_id_2=1,config_group_annot=1,ratio_1=0.25,ratio_2=2,flag_group=-1,train_id1=1,flag_scale_1=1,beta_mode=0,motif_id_1='',query_id1=-1,query_id2=-1,query_id_1=-1,query_id_2=-1,train_mode=0,config_id_load=-1):
		
		chromosome = str(chromosome)
		run_id = int(run_id)
		species_id = str(species)
		# cell = str(cell)
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


