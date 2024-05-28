# #!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

from copy import deepcopy
import warnings
import sys

import os
import os.path
from optparse import OptionParser

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer

from scipy import stats
from scipy.stats import norm

from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

import utility_1
from utility_1 import test_query_index
import h5py
import pickle

from test_unify_compute_1 import _Base2_correlation2

class _Base2_correlation3(_Base2_correlation2):
	"""Feature association estimation
	"""
	def __init__(self,file_path,run_id,species_id=1,cell='ES', 
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

		_Base2_correlation2.__init__(self,file_path=file_path,
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
	# compute peak accessibility-gene expression correlation for alternative peak-gene links
	# to update
	def test_gene_peak_query_basic_filter_1_pre1(self,df_gene_peak_query=[],df_gene_peak_compare=[],atac_ad=[],rna_exprs=[],
													column_correlation=[],column_idvec=['gene_id','peak_id'],column_label='',
													interval_peak_corr=500,iter_idvec=[-1,-1],flag_distance_annot=1,
													save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		compute peak accessibility-gene expression correlation for alternative peak-gene links
		:param df_gene_peak_query: (dataframe) annotations of candidate peak-gene links
		:param df_gene_peak_compare: (dataframe) annotations of alternative potential peak-gene links 
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param column_correlaton: (array or list) columns representing peak accessibility-gene expression correlation, emprical p-value, and the raw p-value
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param colum_label: (str) column representing specific labels of the peak-gene links
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param iter_vec: the start and end indices of peak-gene links for which to perform computation in the batch mode
		:param flag_distance_annot: indicator of whether to copy the computed peak-gene correlations to the dataframe of genome-wide peak-gene link annotations
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: (dataframe) annotations of alternative peak-gene links with peak accessibility-gene expression correlation computed
		"""

		df_gene_peak_query_compare = df_gene_peak_compare
		print('peak-gene links, dataframe of ',df_gene_peak_query_compare.shape)
		print('preview:')
		print(df_gene_peak_query_compare[0:2])
		
		if len(column_correlation)==0:
			column_correlation=['spearmanr','pval1','pval1_ori']

		column_corr_1 = column_correlation[0] # column representing peak accessibility-gene expression correlation
		column_pval_ori = column_correlation[2] # column representing the raw p-value of peak accessibility-gene expression correlation
		column_id1, column_id2 = column_idvec[0:2] # columns representing the gene and peak indices
		iter_mode = 0  # indicator of whether to perform computation in batch mode

		if len(iter_idvec)>0:
			start_id1, start_id2 = iter_idvec[0:2] # the start and end indices of peak-gene links for which to perform computation in the batch mode
		if ('query_id1' in select_config) and ('query_id2' in select_config):
			start_id1 = select_config['query_id1']
			start_id2 = select_config['query_id2']

		df_gene_peak_query_compare.index = np.asarray(df_gene_peak_query_compare[column_id1])
		feature_query_vec_ori = df_gene_peak_query_compare[column_id1].unique()  # the genes included in the alternative peak-gene links
		feature_query_num_ori = len(feature_query_vec_ori)

		df_gene_peak_query_compare2 = []
		if (start_id1>=0) and (start_id2>start_id1):
			if (start_id1<feature_query_num_ori):
				iter_mode = 1
				start_id2 = np.min([start_id2,feature_query_num_ori])

				feature_query_vec = feature_query_vec_ori[start_id1:start_id2]
				df_gene_peak_query_compare = df_gene_peak_query_compare.loc[feature_query_vec,:]  # the subset of peak-gene links for which to perform computation in the batch mode
				print('start_id1: %d, start_id2: %d'%(start_id1,start_id2))
				print('peak-gene links, dataframe of ',df_gene_peak_query_compare.shape)
				print('preview:')
				print(df_gene_peak_query_compare[0:2])
			else:
				return df_gene_peak_query_compare2

		df_gene_peak_query_compare.index = test_query_index(df_gene_peak_query_compare,column_vec=column_idvec)
		
		# find the links without previously estimated peak accessibility-gene expression correlation
		if (column_corr_1 in df_gene_peak_query_compare.columns):
			query_id2 = df_gene_peak_query_compare.index[pd.isna(df_gene_peak_query_compare[column_corr_1])==True]
			# print('query_id2 ',len(query_id2))
			df_compare2 = df_gene_peak_query_compare.loc[query_id2]
		else:
			df_compare2 = df_gene_peak_query_compare
		
		gene_query_vec_2 = df_compare2[column_id1].unique()
		gene_query_num2 = len(gene_query_vec_2)
		print('potential alternative target genes: %d '%(gene_query_num2))

		df_compare2.index = np.asarray(df_compare2[column_id1])
		peak_dict = dict()

		# for peak-gene links without estimated peak accessibility-gene expression correlation, estimate the correlation
		if gene_query_num2>0:
			warnings.filterwarnings('ignore')
			start1 = time.time()
			print('peak accessibility-gene expression correlation estimation')

			output_file_path1 = output_file_path
			save_file_path = output_file_path1
			save_file_path_local = output_file_path1
			if filename_prefix_save=='':
				filename_prefix_save = select_config['filename_prefix_default']

			if iter_mode>0:
				filename_prefix_save = '%s.%d_%d'%(filename_prefix_save,start_id1,start_id2)
			output_filename = '%s/%s.%s.1.txt'%(output_file_path1,filename_prefix_save,filename_save_annot)

			df_compare2.index = np.asarray(df_compare2[column_id1])
			t_columns = df_compare2.columns.difference(['spearmanr_abs','distance_abs'],sort=False)
			df_compare2 = df_compare2.loc[:,t_columns]
			if os.path.exists(output_filename)==False:
				df_compare2.to_csv(output_filename,sep='\t',float_format='%.5f')
			
			interval_peak_corr_1 = interval_peak_corr
			interval_local_peak_corr_1 = select_config['interval_local_peak_corr']
		
			select_config.update({'interval_peak_corr':interval_peak_corr_1,'interval_peak_corr_1':interval_peak_corr_1})
			
			field_query = ['flag_corr_','method_type_correlation','type_id_correlation']
			flag_corr_,method_type,type_id_1 = 1,1,1
			list1 = [flag_corr_,method_type,type_id_1]
			query_num1 = len(field_query)
			for i1 in range(query_num1):
				field_id, query_value = field_query[i1], list1[i1]
				select_config.update({field_id:query_value})

			# compute peak accessibility-gene expression correlation
			df_gene_peak_query_compare2 = self.test_gene_peak_query_correlation_1(gene_query_vec=gene_query_vec_2,
																					peak_dict=peak_dict,
																					df_gene_peak_query=df_compare2,
																					atac_ad=atac_ad,
																					rna_exprs=rna_exprs,
																					interval_peak_corr=interval_peak_corr_1,
																					interval_local_peak_corr=interval_local_peak_corr_1,
																					peak_bg_num=-1,
																					save_file_path=save_file_path,
																					save_file_path_local=save_file_path_local,
																					filename_prefix_save=filename_prefix_save,
																					select_config=select_config)
			
			warnings.filterwarnings('default')
			stop1 = time.time()
			# print('peak accessibility-gene expression correlation estimation used: %.5fs'%(stop1-start1))	
			df_gene_peak_query_compare2.index = np.asarray(df_gene_peak_query_compare2[column_id1])
			
			output_filename = '%s/%s.%s.2.txt'%(output_file_path1,filename_prefix_save,filename_save_annot)
			df_gene_peak_query_compare2.to_csv(output_filename,sep='\t',float_format='%.5f')
			df_gene_peak_query_compare2.index = test_query_index(df_gene_peak_query_compare2,column_vec=column_idvec)

			column_1 = 'list_distance_annot'
			if column_1 in select_config:
				list_distance_annot = select_config[column_1]
			else:
				list_distance_annot = []
			list_distance_annot.append(output_filename)
			select_config.update({column_1:list_distance_annot})

			if (flag_distance_annot>0) and (iter_mode==0):
				df_gene_peak_distance = self.df_gene_peak_distance
				df_gene_peak_distance.index = test_query_index(df_gene_peak_distance,column_vec=column_idvec)
				print('df_gene_peak_distance ',df_gene_peak_distance.shape)
				print('df_gene_peak_query_compare2 ',df_gene_peak_query_compare2.shape)

				query_id2 = df_gene_peak_query.index
				query_id3 = df_gene_peak_query_compare2.index
				field_query_2 = [column_corr_1,column_pval_ori]

				df_gene_peak_distance.loc[query_id2,field_query_2] = df_gene_peak_query.loc[query_id2,field_query_2]	# copy peak-gene correlation
				df_gene_peak_distance.loc[query_id3,field_query_2] = df_gene_peak_query_compare2.loc[query_id3,field_query_2]	# copy peak-gene correlation
				
				df_gene_peak_distance.index = np.asarray(df_gene_peak_distance[column_id1])
				
				output_filename = select_config['filename_distance_annot'] # add peak-gene correlation	
				df_gene_peak_distance.to_csv(output_filename,sep='\t',float_format='%.5f')
				self.df_gene_peak_distance = df_gene_peak_distance

				filename_pre1 = output_filename
				b = filename_pre1.find('.txt')
				filename_pre1 = filename_pre1[0:b]+'.npy'
				dict1 = {'df_gene_peak_distance':df_gene_peak_distance}
				np.save(filename_pre1,df_gene_peak_distance,allow_pickle=True)

			return df_gene_peak_query_compare2

	## ====================================================
	# query GC content bins for peak loci and distance bins for peak-gene links;
	# estimate the empirical distribution of peak accessibility-gene expression correlations of peak-gene links in different distance bins;
	# to update
	def test_query_link_correlation_distance_1(self,df_link_query=[],df_feature_annot=[],column_id_query='',n_bins_vec=[50,100],distance_bin=50,flag_unduplicate=1,save_mode=0,output_file_path='',filename_prefix_save='',select_config={}):
		
		"""
		query GC content bins for peak loci and distance bins for peak-gene links;
		estimate the empirical distribution of peak accessibility-gene expression correlations of peak-gene links in different distance bins;
		:param df_link_query: (dataframe) peak-gene link annotations
		:param df_feature_annot: (dataframe) peak annotations including the GC content of the sequence of each peak locus
		:param column_id_query: (str) column representing peak accessibility-gene expression correlations in the peak-gene link annotation dataframe
		:param n_bins_vec: (array or list) the number of GC content bins for peak loci and the number of peak-gene distance bins
		:param distance_bin: the distance bin size (unit:Kb)
		:param flag_unduplicate: indicator of whether to unduplicate the dataframe
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) updated annotations of peak-gene links including estimated empirical p-values of peak accessibility-gene expression correlations in each distance bin based on the background distribution 
				 2. (dataframe) mean and std values of peak accessibility-gene expression correlations in each distance bin
				 3. dictionary to store other information to retrieve
		"""

		if column_id_query=='':
			column_id_query = 'peak_gene_corr_'
		
		column_idvec = ['peak_id','gene_id']
		column_id2, column_id1 = column_idvec

		# the peak-gene links
		if flag_unduplicate>0:
			df_link_query = df_link_query.drop_duplicates(subset=column_idvec,keep='first')
		
		column_corr_1 = column_id_query
		peak_gene_corr_ = df_link_query[column_corr_1]
		
		df_link_query.index = test_query_index(df_link_query,column_vec=column_idvec)
		df_link_query_ori = df_link_query
		df_link_query = df_link_query.dropna(subset=[column_id_query])
		print('peak-gene links with estimated accessibility-gene expression correlation, dataframe of size ',df_link_query.shape)

		df_feature_annot_1 = df_feature_annot
		column_1, column_2 = 'GC_bin', 'distance_bin'
		n_bins_GC, n_bins_distance = n_bins_vec[0:2]
		column_1_ori = 'GC'
		interval_1 = 1.0/n_bins_GC
		column_1_query = '%s_%d'%(column_1,n_bins_GC)
		type_id2 = 0
		if type_id2==0:
			query_value = df_feature_annot_1[column_1_ori]
			df_feature_annot_1[column_1_query] = np.digitize(query_value, np.linspace(0, 1, n_bins_GC+1))
		else:
			query_value = df_feature_annot_1[column_1_ori]
			query_value_2 = np.int32(np.ceil(query_value/interval_1))
			id1 = (query_value_2==0)
			query_value_2[id1] = 1
			df_feature_annot_1[column_1_query] = query_value_2

		output_filename_1 = '%s/%s.GC.annot1.txt'%(output_file_path,filename_prefix_save)
		df_feature_annot_1.to_csv(output_filename_1,sep='\t')

		distance = df_link_query['distance']
		distance_abs = distance.abs()

		# max_distance, min_distance = distance_abs.max(), distance_abs.min()
		max_distance, min_distance = distance_abs.max(), 0
		if distance_bin>0:
			n_bins_distance = int(np.ceil((max_distance-min_distance)/distance_bin))
		else:
			distance_bin = (max_distance-min_distance)/n_bins_distance

		print('number of bins by distance: %d'%(n_bins_distance))
		print('distance bin size: %d'%(distance_bin))

		distance_bin_vec = np.unique(np.asarray([distance_bin]+[50]))
		peak_distance_thresh_1 = 2000
		for distance_bin_value in distance_bin_vec:
			column_2_query = '%s_%d'%(column_2,distance_bin_value)
			if type_id2==0:
				t_vec_1 = np.arange(0,peak_distance_thresh_1+distance_bin_value,distance_bin_value)
				df_link_query[column_2_query] = np.digitize(distance_abs, t_vec_1)
			else:
				df_link_query[column_2_query] = np.int32(np.ceil(distance_abs/distance_bin_value))
				id1 = (df_link_query[column_2_query]==0)
				df_link_query.loc[id1,column_2_query] = 1
		
		df_link_query.index = np.asarray(df_link_query[column_id2])
		query_id1 = df_link_query.index
		df_link_query.loc[:,column_1_query] = df_feature_annot_1.loc[query_id1,column_1_query] # the GC group
		df_link_query.loc[:,column_1_ori] = df_feature_annot_1.loc[query_id1,column_1_ori]

		normalize_type = 'uniform'
		column_vec_query = ['distance_bin','spearmanr','GC_bin_%d'%(n_bins_GC)]
		column_annot = ['distance_pval1','distance_pval2']
		query_num1 = len(distance_bin_vec)
		df_link_query.index = test_query_index(df_link_query,column_vec=column_idvec)
		for i1 in range(query_num1): 
			distance_bin_value = distance_bin_vec[i1]
			column_1, column_2, column_3 = column_vec_query[0:3]
			column_vec_query1 = ['%s_%d'%(column_1,distance_bin_value),column_2,column_3]
			column_annot_query1 = ['%s_%d'%(column_query,distance_bin_value) for column_query in column_annot]
			# query the empirical distribution of the peak-gene correlations in different distance bins
			df_link_query1, df_annot1, dict_annot1 = self.test_attribute_query_distance_1(df_link_query,column_vec_query=column_vec_query1,column_annot=column_annot_query1,normalize_type=normalize_type,
																							verbose=1,select_config=select_config)

			filename_annot1 = distance_bin_value
			output_filename = '%s/%s.distance.%s.annot1.txt'%(output_file_path,filename_prefix_save,filename_annot1)
			id1 = np.asarray(df_annot1.index)
			df_annot1['distance_1'], df_annot1['distance_2'] = distance_bin*(id1-1), distance_bin*id1
			df_annot1.to_csv(output_filename,sep='\t')

			if i1==0:
				df_link_query = df_link_query1
				query_id1 = df_link_query.index
			else:
				df_link_query.loc[query_id1,column_annot_query1] = df_link_query1.loc[query_id1,column_annot_query1]

		filename_annot1 = distance_bin
		df_link_query.index = np.asarray(df_link_query[column_id1])
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
		output_filename = '%s/%s.distance.%s.annot2.sort.txt'%(output_file_path,filename_prefix_save,filename_annot1)
		df_link_query_sort = df_link_query.sort_values(by=[column_id2,'distance'],ascending=[True,True])
		df_link_query_sort.to_csv(output_filename,sep='\t',float_format='%.5E')
		print('df_feature_annot_1, df_annot1, df_link_query: ',df_feature_annot_1.shape,df_annot1.shape,df_link_query.shape)
		
		return df_link_query, df_annot1, dict_annot1

	## ====================================================
	# estimate the empirical distribution of peak accessibility-gene expression correlations of peak-gene links in different distance bins
	# to update
	def test_attribute_query_distance_1(self,df_link_query,column_vec_query=[],column_annot=[],normalize_type='uniform',type_id_1=0,save_mode=0,verbose=0,select_config={}):

		"""
		estimate the empirical distribution of peak accessibility-gene expression correlations of peak-gene links in different distance bins
		:param df_link_query: (dataframe) peak-gene link annotations
		:param column_vec_query: (array or list) columns representing distance bin and peak accessibility-gene expression correlation bin of the peak-gene links
		:param column_annot: (array or list) columns representing estimated p-values of the differences of peak accessibility-gene expression correlations
		:param normalize_type: the normalization type to use in quantile transformation
		:param type_id_1: the combination of peak accessibility-gene expression correlation types (positive or negative correlation) from two distance bins
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) updated annotations of peak-gene links including estimated empirical p-values of peak accessibility-gene expression correlations based on the background distribution in each distance bin
				 2. (dataframe) mean and std values of peak accessibility-gene expression correlations in each distance bin
				 3. dictionary to store other information to retrieve
		"""	

		column_GC, column_distance = 'GC_bin_20', 'distance_bin_50'
		column_corr_1 = 'peak_gene_corr_'
		if column_vec_query==[]:
			column_vec_query = [column_distance,column_corr_1,column_GC]
		else:
			column_distance, column_corr_1 = column_vec_query[0:2]
			column_GC = column_vec_query[2]

		if len(column_annot)==0:
			# column_annot = ['distance_pval1','distance_pval2']
			column_annot = ['distance_pval1_50','distance_pval2_50']
		
		column_1, column_2 = column_vec_query[0:2]
		column_query1, column_query2 = column_annot[0:2]
		
		group_query = df_link_query[column_1]
		query_vec = np.sort(np.unique(group_query))
		query_num = len(query_vec)

		column_idvec = ['peak_id','gene_id']
		df_link_query.index = test_query_index(df_link_query,column_vec=column_idvec)
		query_id_ori = df_link_query.index

		column_vec_1 = ['max_corr','min_corr','median_corr','mean_corr','std_corr']
		df1 = pd.DataFrame(index=query_vec,columns=column_vec_1,dtype=np.float32)
		dict_query1 = dict()
		
		t_vec_1 = np.linspace(0,1,11)
		quantile_value_vec = [0.01]+t_vec_1[1:-1]+[0.99]
		# normalize_type = 'uniform'
		from scipy.stats import norm
		for i1 in range(query_num):
			group_id = query_vec[i1]
			id1 = (group_query==group_id)
			query_value = df_link_query.loc[id1,column_2]

			id2 = (query_value>0)
			dict_query1.update({group_id:query_value})
			if type_id_1==0:
				query_value_1 = query_value.abs()
				m1,v1 = np.mean(query_value_1), np.std(query_value_1)
			pval = 1 - norm.cdf(query_value_1,m1,v1)
			df_link_query.loc[id1,column_query1] = pval

			value_scale = quantile_transform(query_value_1[:,np.newaxis],n_quantiles=1000,output_distribution=normalize_type)
			quantile_value = value_scale[:,0]
			pval_empirical = 1-quantile_value
			df_link_query.loc[id1,column_query2] = pval_empirical

			t_vec_1 = [np.max(query_value),np.min(query_value),np.median(query_value)]
			t_vec_2 = t_vec_1+[m1,v1]
			df1.loc[group_id,column_vec_1] = t_vec_2
			
		df_annot = df1
		return df_link_query, df_annot, dict_query1 

	## ====================================================
	# estimate the empirical distribution of the difference between peak accessibility-gene expression correlations of peak-gene links from the same or different distance bins
	# to update
	def test_attribute_query_distance_2(self,df_link_query,df_annot=[],column_vec_query=[],column_annot=[],n_sample=500,
												distance_bin=50,distance_tol=2,type_id_1=0,
												save_mode=1,output_file_path='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		estimate the empirical distribution of the difference between peak accessibility-gene expression correlations of peak-gene links from the same or different distance bins
		:param df_link_query: (dataframe) peak-gene link annotations
		:param df_annot: (dataframe) distance bin assignment of peak-gene links
		:param dict_query: dictionary storing peak-gene linsk in each distance bin
		:param column_vec_query: (array or list) columns representing distance bin and peak accessibility-gene expression correlation bin of the peak-gene links
		
		:param n_sample: the nubmer of peak-gene links to randomly sampled to estimate the background distribution of peak accessibility-gene expression correlations
		:param distance_bin: the distance bin size (unit:Kb)
		:param distance_tol: threshold on distance difference between two distance bins
		:param type_id_1: the combination of peak accessibility-gene expression correlation types (positive or negative correlation) from two distance bins
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) mean, std, and specified quantile values of differences of peak-gene correlations between each randomly sampled pair of peak-gene links from the same distance bin for each distance bin
				 2. (dataframe) mean, std, and specified quantile values of differences of peak-gene correlations between each randomly sampled pair of peak-gene links from two different distance bins for each pair of distance bins
				 3. dictionary containing differences of peak-gene correlations between each randomly sampled pair of peak-gene links from the same distance bin
				 4. dictionary containing differences of peak-gene correlations between each randomly sampled pair of peak-gene links from two different distance bins
		"""

		# column_1, column_2 = 'GC_bin', 'distance_bin'
		column_1, column_2 = 'GC_bin_20', 'distance_bin_50'
		column_corr_1 = 'peak_gene_corr_'
		if column_vec_query==[]:
			# column_vec_query = [column_2,column_corr_1]
			column_vec_query = [column_2,column_corr_1,column_1]

		if len(column_annot)==0:
			# column_annot = ['distance_pval1','distance_pval2']
			column_annot = ['distance_pval1_50','distance_pval2_50']

		column_id1, column_id2 = column_vec_query[0:2]
		column_query1, column_query2 = column_annot[0:2]

		group_query = df_link_query[column_id1]	# distance bin group
		query_vec = np.sort(np.unique(group_query))
		group_id_min, group_id_max = query_vec[0], query_vec[-1]

		query_num = len(query_vec)
		# n_sample = 1000
		distance_tol = query_num
		# import itertools
		column_vec_1 = ['max_corr','min_corr','median_corr','mean_corr','std_corr']
		column_vec_2 = ['mean_corr_difference','std_corr_difference']
		column_vec = column_vec_1+column_vec_2

		if len(df_annot)==0:
			df1 = pd.DataFrame(index=query_vec, dtype=np.float32)
		else:
			df1 = df_annot

		dict_query1 = dict()
		query_value_ori = df_link_query[column_id2]

		# prepare sample for each group
		flag_query1=0
		if not ('mean_corr' in df1.columns):
			flag_query1=1
		for i1 in range(query_num):
			group_id1 = query_vec[i1]
			query_value1 = df_link_query.loc[(group_query==group_id1),column_id2]
			query_id1 = query_value1.index
			query_num1 = len(query_id1)
			if query_num1==0:
				print('group_id1:%d,query_num1:%d'%(group_id1,query_num1))
				continue

			if flag_query1>0:
				t_vec_1 = [np.max(query_value1),np.min(query_value1),np.median(query_value1)]
				m1_ori,v1_ori = np.mean(query_value1), np.std(query_value1)
				t_vec_2 = t_vec_1+[m1_ori,v1_ori]
				df1.loc[group_id1,column_vec_1] = t_vec_2

			sample_id1 = query_id1
			dict_query1.update({group_id1:sample_id1})

		flag_sort = 1
		verbose = 1
		# estimate the empirical distribution of the difference between peak accessibility-gene expression correlations of peak-gene links from the same or different distance bins
		df_annot1, df_annot2, dict_query1, dict_query2 = self.test_query_group_feature_compare_1(df_query=df_link_query,df_annot=df1,
																									dict_query=dict_query1,
																									column_vec=column_vec_query,
																									distance_tol=distance_tol,
																									flag_sort=flag_sort,
																									type_id_1=type_id_1,
																									verbose=verbose,
																									select_config=select_config)

		filename_annot = str(distance_bin)
		list1 = [df_annot1, df_annot2]
		list2 = [dict_query1, dict_query2]
		query_num1 = len(list1)
		for i1 in range(query_num1):
			# output_filename_1 = '%s/test_group_feature_compare.%s.annot%d.txt'%(output_file_path,filename_annot,i1+1)
			output_filename_1 = '%s/%s.group_feature_compare.%s.annot%d.%d.txt'%(output_file_path,filename_prefix_save,filename_annot,i1+1,type_id_1)
			df_query = list1[i1]
			df_query.to_csv(output_filename_1,sep='\t',float_format='%.5f')

			# output_filename_2 = '%s/test_group_feature_compare.%s.annot%d.npy'%(output_file_path,filename_annot,i1+1)
			output_filename_2 = '%s/%s.group_feature_compare.%s.annot%d.%d.npy'%(output_file_path,filename_prefix_save,filename_annot,i1+1,type_id_1)
			dict_query = list2[i1]
			np.save(output_filename_2,dict_query,allow_pickle=True)
		
		return df_annot, df_annot2, dict_query1, dict_query2

	## ====================================================
	# estimate the empirical distribution of the difference between peak accessibility-gene expression correlations of peak-gene links from the same or different distance bins
	# to update
	def test_query_group_feature_compare_1(self,df_query=[],df_annot=[],dict_query=dict(),column_vec=[],n_sample=500,distance_tol=-1,flag_sort=0,type_id_1=0,verbose=0,select_config={}):

		"""
		estimate the empirical distribution of the difference between peak accessibility-gene expression correlations of peak-gene links from the same or different distance bins
		:param df_query: (dataframe) peak-gene link annotations
		:param df_annot: (dataframe) distance bin assignment of peak-gene links
		:param dict_query: dictionary storing peak-gene linsk in each distance bin
		:param column_vec: (array or list) columns representing distance bin and peak accessibility-gene expression correlation bin of the peak-gene links
		:param n_sample: the nubmer of peak-gene links to randomly sampled to estimate the background distribution of peak accessibility-gene expression correlation
		:param distance_tol: threshold on distance difference between two distance bins
		:param flag_sort: indicator of whether to sort peak-gene links by peak-gene ditance
		:param type_id_1: the combination of peak accessibility-gene expression correlation types (positive or negative correlation) from two distance bins
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) mean, std, and specified quantile values of differences of peak-gene correlations between each randomly sampled pair of peak-gene links from the same distance bin for each distance bin
				 2. (dataframe) mean, std, and specified quantile values of differences of peak-gene correlations between each randomly sampled pair of peak-gene links from two different distance bins for each pair of distance bins
				 3. dictionary containing differences of peak-gene correlations between each randomly sampled pair of peak-gene links from the same distance bin
				 4. dictionary containing differences of peak-gene correlations between each randomly sampled pair of peak-gene links from two different distance bins
		"""

		column_id1, column_id2 = column_vec[0:2]
		column_vec_1 = ['max_corr','min_corr','median_corr','mean_corr','std_corr']
		column_vec_2_ori = ['mean_corr_difference','std_corr_difference']
		# quantile_vec_1 = np.linspace(0,1,11)
		quantile_vec_1 = np.linspace(0,1,101)
		column_quantile = ['%.2f'%(query_value) for query_value in quantile_vec_1]
		column_vec_2 = column_vec_2_ori + ['max_value','min_value','median','mean']
		column_vec_2 = column_vec_2 + column_quantile
		group_query = df_query[column_id1]
		if len(df_annot)==0:
			query_vec = np.sort(np.unique(group_query))
			df_annot = pd.DataFrame(index=query_vec,columns=column_vec_2,dtype=np.float32)
		else:
			query_vec = np.sort(df_annot.index)
		df1 = df_annot
		dict_query_1 = dict_query
		query_num = len(query_vec)

		import itertools
		id_list1 = list(itertools.combinations(query_vec,2))
		query_idvec = ['group%s_group%s'%(group_id1,group_id2) for (group_id1,group_id2) in id_list1]
		df2 = pd.DataFrame(index=query_idvec,columns=['group1','group2'])
		df2['group1'] = [query_id[0] for query_id in id_list1]
		df2['group2'] = [query_id[1] for query_id in id_list1]
		if distance_tol<0:
			distance_tol = query_num

		flag_sort1 = flag_sort
		column_idvec = ['peak_id','gene_id']
		df_query.index = test_query_index(df_query,column_vec=column_idvec)
		column_distance = 'distance'
		column_distance_abs = '%s_abs'%(column_distance)
		df_query[column_distance_abs] = df_query[column_distance].abs()
		if flag_sort1>0:
			df_query = df_query.sort_values(by=column_distance_abs,ascending=True)
		query_value_ori = df_query[column_id2]
		
		if len(dict_query_1)==0:
			query_id_ori = df_query.index
			for i1 in range(query_num):
				group_id1 = query_vec[i1]
				sample_id1 = query_id_ori[group_query==group_id1]
				dict_query_1.update({group_id1:sample_id1})

		dict_query1 = dict()
		for i1 in range(query_num):
			group_id1 = query_vec[i1]
			# sample_id1 = dict_query1[group_id1]	# sample from group 1
			query_id1 = dict_query_1[group_id1]	# sample from group 1

			query_num1 = len(query_id1)
			if query_num1>n_sample:
				# sample_value1 = np.random.choice(query_value1,n_sample,False)
				sample_id1 = np.random.choice(query_id1,n_sample,replace=False) # select a subset of peaks from each distance bin group
			else:
				# sample_value1 = query_value1
				if flag_sort1==0:
					np.random.shuffle(query_id1)
				sample_id1 = query_id1

			if verbose>0:
				print('group_id1: ',group_id1,query_num1)

			sample_value1_ori = query_value_ori[sample_id1]
			sample_value1 = sample_value1_ori.abs()
			list1 = list(itertools.combinations(sample_value1,2))
			difference_corr = [value2-value1 for (value1,value2) in list1]
			
			value = difference_corr
			m1,v1 = np.mean(value),np.std(value)
			t_value_1 = utility_1.test_stat_1(value,quantile_vec=quantile_vec_1)
			df1.loc[group_id1,column_vec_2] = [m1,v1]+list(t_value_1)
			dict_query1.update({group_id1:[sample_id1,difference_corr]})

		list1 = list(itertools.combinations(query_vec,2))
		query_num1 = len(list1)
		dict_query2 = dict()
		for i1 in range(query_num1):
			group1, group2 = list1[i1]
			t_vec_1 = dict_query1[group1]
			sample_id1,difference_corr_1 = t_vec_1

			t_vec_2 = dict_query1[group2]
			sample_id2,difference_corr_2 = t_vec_2

			# sample_value1, sample_value2 = difference_corr_1, difference_corr_2
			sample_value1_ori, sample_value2_ori = query_value_ori[sample_id1], query_value_ori[sample_id2]
			if type_id_1 in [0,1]:
				# sample_value1, sample_value2 = query_value_ori[sample_id1].abs(), query_value_ori[sample_id2].abs()
				sample_value1, sample_value2 = sample_value1_ori.abs(), sample_value2_ori.abs()
			elif type_id_1==2:
				# negative correlation
				sample_value1 = -sample_value1_ori
				sample_value2 = -sample_value2_ori
			elif type_id_1==3:
				# positive correlation in bin 1 and negative correlation in bin 2
				sample_value1 = sample_value1_ori
				sample_value2 = -sample_value2_ori
			elif type_id_1==5:
				# negative correlation in bin 1 and positive correlation in bin 2
				sample_value1 = -sample_value1_ori
				sample_value2 = sample_value2_ori

			list_pre2 = [list(sample_value1),list(sample_value2)] # the combination of values from two groups
			list2 = [p for p in itertools.product(*list_pre2)]

			difference_corr_2 = np.asarray([value2-value1 for (value1,value2) in list2])
			query_id2 = 'group%d_group%d'%(group1,group2)
			dict_query2.update({query_id2:difference_corr_2})

			value_2 = difference_corr_2
			m2,v2 = np.mean(value_2), np.std(value_2)
			t_value_2 = utility_1.test_stat_1(value_2,quantile_vec=quantile_vec_1)
			df2.loc[query_id2,column_vec_2] = [m2,v2]+list(t_value_2)

			if verbose>0:
				if (group1%10==0) & (group2%10==0):
					print('group1: %d, group2: %d, difference_correlation: %d'%(group1,group2,len(difference_corr_2)))

		df_annot = df1
		df_annot2 = df2
		return df_annot, df_annot2, dict_query1, dict_query2

	## ====================================================
	# query distance and link score related annotations for peak-gene links
	# to update
	def test_gene_peak_query_link_basic_filter_1_pre2(self,peak_id=[],df_peak_query=[],df_gene_peak_query=[],field_query=[],thresh_vec_compare=[],column_idvec=['peak_id','gene_id'],
														column_vec_query=[],column_label='',type_score=0,type_id_1=0,save_mode=0,verbose=0,select_config={}):

		"""
		query distance and link score (e.g.,peak accessibility-gene expression correlation) related annotations for peak-gene links
		:param peak_id: (str) name or index of the peak for which we compare peak-gene links associated with multiple potential target genes
		:param df_peak_query: (dataframe) annotations of peak attributes
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param field_query: (array or list) the attributes based on which to perform peak-gene link comparison and selection
		:param thresh_vec_compare: (array or list) thresholds used for peak-gene link comparison
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param column_vec_query: (array or list) columns representing peak-gene distance and peak-gene link score
		:param column_label: column representing labels of peak-gene links
		:param type_score: association score type: 0: peak accessibility-gene expression correlation for peak-gene links; 1: estimated link score
		:param type_id_1: the type of comparison to perform: 0: use peak-gene link with the highest score in the minimal distance bin to compare with other links;
														     1: use peak-gene link with the smallest distance in the highest score bin to compare with other links;
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) peak-gene links with annotations related to peak-gene distance and link score
				 2. (dataframe) annotations of a subset of the peak-gene links
		"""

		column_id2, column_id1 = column_idvec[0:2]
		link_type_annot = 'peak-gene links'
		if len(field_query)==0:
			if type_score==0:
				score_type_annot = 'correlation'  # peak accessibility-gene expression correlation
			else:
				score_type_annot = 'score'
			field_query = ['distance',score_type_annot]
		field1, field2 = field_query[0:2]
		annot_str_vec = field_query

		# column_1, column_2 = 'distance_bin', 'score_bin'
		column_1, column_2 = '%s_bin'%(field1), '%s_bin'%(field2)
		column_vec_1 = [column_1,column_2]
		column_pre1, column_pre2 = '%s_min'%(column_1), '%s_max'%(column_2)
		column_vec_2 = [column_pre1,column_pre2]

		# column_score_1: peak-gene link score
		column_distance, column_score_1 = column_vec_query[0:2]
		
		column_value_1 = '%s_abs'%(column_distance)
		if not (column_value_1 in df_gene_peak_query.columns):
			df_gene_peak_query[column_value_1] = df_gene_peak_query[column_distance].abs() # absolute distance

		column_value_2 = column_score_1
		if type_score==0:
			column_value_query2 = '%s_abs'%(column_score_1)
			column_value_2 = column_value_query2
			if not (column_value_2 in df_gene_peak_query.columns):
				df_gene_peak_query[column_value_2] = df_gene_peak_query[column_score_1].abs()

			column_group_annot = 'group2'
			df_gene_peak_query[column_group_annot] = 1
			id_pre1 = (df_gene_peak_query[column_score_1]<0)
			df_gene_peak_query.loc[id_pre1,column_group_annot] = -1

		column_vec_pre1 = [column_distance,column_score_1] # the original value
		column_vec_pre2 = [column_value_1,column_value_2] # the original or absolute value
		column_annot_1, column_annot_2 = '%s_min_%s'%(field1,field2), '%s_max_%s'%(field2,field1)

		df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_idvec) # reindex the dataframe using peak name and gene name
		
		query_num_1 = len(column_vec_1)
		df1_query = df_gene_peak_query  # the peak-gene link annotations
		feature_type_query = column_id2
		type_query_vec = ['min','max']
		# field_query_1 = ['distance_min_correlation1','distance_min_correlation2','distance_min_correlation']
		# field_query_2 = ['correlation_max_distance1','correlation_max_distance2','correlation_max_distance']
		# id_1 = 1
		
		list_1 = []
		list_2 = []
		verbose_internal = self.verbose_internal
		for id_1 in [1,0]:
			id_2 = (1-id_1)
			column_value_group1 = column_vec_pre2[id_1]	# column: correlation or distance_abs
			column_value_group2 = column_vec_pre2[id_2]	# column: distance_abs or correlation
			
			column_value_group1_ori = column_vec_pre1[id_1] # column: original score or distance value
			column_value_group2_ori = column_vec_pre1[id_2]	# column: original distance or score value

			column_group1 = column_vec_1[id_1]	# id_1:1,correlation_bin;0,distance_bin
			column_group2 = column_vec_1[id_2]	# id_2:0,distance_bin;1,correlation_bin
			
			ascending_vec = [True,False]
			if (type_score==1) or (id_1==0):
				df2_query = df1_query.sort_values(by=[column_value_group1],ascending=ascending_vec[id_1]) # sort the dataframe of peak-gene links by the link score (e.g.,peak accessibiilty-gene expression correlation)
			else:
				# sort the positive and negative peak-TF correlations
				df2_query = df1_query.sort_values(by=[column_group_annot,column_value_group1],ascending=[False,ascending_vec[id_1]]) # sort the dataframe of peak-gene links by the link score (e.g.,peak accessibiilty-gene expr correlation)
			
			df_group_2 = df2_query.groupby(by=[feature_type_query])	# group peak-gene links by peak
			df_2 = df_group_2[[column_group2,column_value_group2]]

			column_vec_query1 = [column_value_group2_ori,column_value_group1_ori]
			column_annot_1 = '%s_%s'%(annot_str_vec[id_2],type_query_vec[id_2])
			column_annot_2 = '%s_%s'%(column_annot_1,annot_str_vec[id_1])
			column_vec_query2 = [['%s_2'%(column_annot_1),'%s2'%(column_annot_2)],[column_annot_1,column_annot_2]]

			if verbose_internal==2:
				print(column_value_group1,column_value_group2)
				print(column_value_group1_ori,column_value_group2_ori)
				print(column_group1,column_group2)
				print(column_vec_query1)
				print(column_vec_query2)
			
			column_pval = 'pval1_ori' 
			field_query_1 = [column_value_group1_ori,column_pval,column_value_group2_ori]
			column_vec_2 = column_vec_query2[0] + column_vec_query2[1]
			if verbose_internal==2:
				print('the added annnotations to %s: '%(link_type_annot),column_vec_2)
			field_query_2 = column_idvec + field_query_1 + column_vec_2

			list_1.append(column_vec_query2)
			if id_1==1:
				# column_vec_query1 = [column_group1,column_corr_1,column_corr_1]
				# column_vec_query2 = ['distance_min_correlation1','distance_min_correlation2','distance_min_correlation']
				df2 = df_2.idxmin() # the link with smallest peak-gene distance bin for each peak; the links are sorted by link score (e.g.,peak-gene correlation)
			else:
				# column_vec_query1 = [column_group1,column_distance,column_distance]
				# column_vec_query2 = ['correlation_max_distance1','correlation_max_distance2','correlation_max_distance']
				df2 = df_2.idxmax() # the link with highest score (e.g.,correlation) bin for each peak; the links are sorted by peak-gene distance
			
			idvec_1 = np.asarray(df2[column_group2])	# the link query in the smallest peak-gene distance bin and with the highest correlation for each peak
			idvec_2 = np.asarray(df2[column_value_group2]) # the link query with the smallest peak-gene distance for each peak, which may not have the highest correlation
			
			# list_query1 = [idvec_1,idvec_1,idvec_2]
			id_2 = (idvec_1!=idvec_2)
			query_id1 = idvec_1[id_2]	# the link in the smallest peak-gene distance bin and without the smallest distance and with the highest correlation
			query_id2 = idvec_2[id_2]	# the link with the smallest peak-gene distance but without the highest correlation
			
			query_num1, query_num2 = len(query_id1),len(query_id2)
			# print('query_id1:%d, query_id2:%d'%(query_num1,query_num2))
			list_2.append([query_id1,query_id2])
			list_query1 = [idvec_1,query_id2]

			query_num = len(list_query1)
			from utility_1 import test_column_query_2
			annot_vec_query1 = [['in the smallest distance bin','with the highest %s'%(score_type_annot)],
								['with the smallest distance','without the highest %s'%(score_type_annot)]]

			annot_vec_query2 = [['in the highest %s bin'%(score_type_annot),'with the smallest distance'],
								['with the highest %s'%(score_type_annot),'without the smallest distance']]

			# annot_str_1 = 'peak-gene link in the smallest distance bin and with the highest correlation in the distance bin'
			# annot_str_2 = 'peak-gene link in the highest correlation bin and with the smallest distance in the correlation bin'

			attribute_type_vec = [score_type_annot,'distance']
			list_annot1 = [annot_vec_query2,annot_vec_query1]
			dict_query1 = dict(zip(attribute_type_vec,list_annot1))
			
			for i2 in range(query_num):
				query_id = list_query1[i2]
				df_query = df2_query.loc[np.asarray(query_id),:]
				if i2==1:
					attribute_type = attribute_type_vec[id_1]
					annot_vec_query = dict_query1[attribute_type]
					annot_str1,annot_str2 = annot_vec_query[1][0], annot_vec_query[1][1]
					print('peak-gene link %s and %s in the %s bin, dataframe of size '%(annot_str1,annot_str2,attribute_type),df_query.shape)

					column_vec = df_query.columns
					t_columns = pd.Index(field_query_2).intersection(column_vec,sort=False)
					df_query1 = df_query.loc[:,t_columns]
					print('data preview:\n',df_query1[0:2])

				# df2.index = np.asarray(df2[column_id2])
				query_idvec = []
				if i2==1:
					query_idvec = df_query[column_id2].unique()
				# copy columns from the second dataframe to the first dataframe
				df1 = test_column_query_2(df_list=[df2_query,df_query],id_column=[column_id2],
											query_idvec=query_idvec,
											column_vec_1=column_vec_query1,
											column_vec_2=column_vec_query2[i2],
											type_id_1=0,reset_index=True,
											flag_unduplicate=0,
											verbose=0,select_config=select_config)

			df1_query = df1

		df_gene_peak_query = df1
		query_num2 = len(list_2)
		list_query2 = []
		for i2 in range(query_num2):
			query_id1, query_id2 = list_2[i2]
			df2_1 = df_gene_peak_query.loc[np.asarray(query_id1),:]
			df2_2 = df_gene_peak_query.loc[np.asarray(query_id2),:]

			df_2 = pd.concat([df2_1,df2_2],axis=0,join='outer',ignore_index=False)
			# df_2 = df_2.sort_values(by=['peak_id','distance'],ascending=True)
			# type_query: 0, the link with highest correlation in the smallest peak-gene distance bin does not have the smallest distance (the difference is bounded by 50Kb)
			# type_query: 1, the link with smallest distance in the rank 1 correlation bin does not have the highest correlation (the difference is bounded by 0.05)
			df_2['type_query'] = i2
			list_query2.append(df_2)

		df_gene_peak_query2 = pd.concat(list_query2,axis=0,join='outer',ignore_index=False)
		df_gene_peak_query2 = df_gene_peak_query2.sort_values(by=['peak_id','distance'],ascending=True)

		return df_gene_peak_query, df_gene_peak_query2

	## ====================================================
	# compare and select peak-gene links based on peak-gene distance and link score
	# to update
	def test_gene_peak_query_link_basic_filter_1_pre2_1(self,df_feature_link=[],column_idvec=['peak_id','gene_id'],field_query=[],column_vec_query=[],thresh_vec=[],type_score=0,type_id_1=0,verbose=0,select_config={}):

		"""
		perform comparison between peak-gene links based on peak-gene distance and link score (e.g.,peak accessibility-gene expression correlation)
		:param df_feature_link: (dataframe) annotations of peak-gene links
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param field_query: (array or list) the attributes based on which to perform peak-gene link comparison and selection
		:param column_vec_query: (array or list) columns in peak-gene link annotations which correspond to the attributes based on which to compare peak-gene links
		:param thresh_vec: (arrary or list) the thresholds on the corresponding attributes used to select peak-gene links
		:param type_score: the type of peak-gene link score (e.g.,peak accessibility-gene expression correlation) used for comparison: 0: the absolute score; 1: the original score;
		:param type_id_1: the type of comparison to perform: 0: use peak-gene link with the highest score in the minimal distance bin to compare with other links;
														     1: use peak-gene link with the smallest distance in the highest score bin to compare with other links;
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) annotations of peak-gene links retained after the comparison
				 2. (dataframe) annotations of peak-gene links filtered after the comparison
		"""

		column_vec_1 = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_vec_1[0:3]
		
		if len(field_query)==0:
			if type_score==0:
				field_query = ['distance','correlation'] # peak-gene correlation
			else:
				field_query = ['distance','score'] # peak-TF-gene score

		field1, field2 = field_query[0:2]
		column_distance, column_score_1 = column_vec_query[0:2]
		column_value_1_ori, column_value_2_ori = column_distance, column_score_1
		
		column_value_1 = '%s_abs'%(column_distance)
		if type_score==0:
			column_value_2 = '%s_abs'%(column_score_1)
		else:
			column_value_2 = column_score_1
		
		# column_vec_pre1 = [column_distance,column_corr_1]
		column_vec_pre1 = [column_distance,column_score_1] # columns for the original values of peak-gene distance and peak-gene link score
		column_vec_pre2 = [column_value_1,column_value_2]  # columns for the absolute peak-gene distance and the original or absolute peak-gene link score used for comparison
		
		# column_vec_query1 = ['distance_min_2','distance_min_correlation2'] # highest score in the min distance bin and the distance of the link with the score
		# column_vec_query2 = ['correlation_max_distance2','correlation_max_2'] # smallest distance in the highest score bin and the score of the link with the distance
		# column_vec_query1 = ['distance_min_2','distance_min_score2']	# highest score in the distance min bin and distance of the link with the score
		# column_vec_query2 = ['score_max_distance2','score_max_2']	# smallest distance in the highest score bin and the score of the link with the distance
			
		column_vec_query1 = ['%s_min_2'%(field1),'%s_min_%s2'%(field1,field2)]	# highest score in the distance min bin and distance with the score
		column_vec_query2 = ['%s_max_%s2'%(field2,field1),'%s_max_2'%(field2)]	# smallest distance in the highest score bin and the score with the distance
		list_column = [column_vec_query1,column_vec_query2]

		# from utility_1 import test_query_index
		df_feature_link.index = test_query_index(df_feature_link,column_vec=column_idvec)
		df_link_query = df_feature_link

		column_query_1 = list_column[type_id_1]
		column_query_2 = column_vec_pre2 # distance and score query of link query
		if verbose>0:
			print('column_query_1, column_query_2: ',column_query_1,column_query_2)

		query_value_1 = df_link_query.loc[:,column_query_2]
		query_value_2 = df_link_query.loc[:,column_query_1]
		
		column_1 = column_query_1[0]
		query_value_2[column_1] = query_value_2[column_1].abs()

		difference = np.asarray(query_value_1) - np.asarray(query_value_2) # the difference between the original value and the compared value
		value_1 = difference[:,0]
		value_2 = -difference[:,1]

		if len(thresh_vec)==0:
			thresh_value_1 = 100
			# thresh_value_2 = 0.1
			thresh_value_2 = 0.15
			thresh_vec = [[thresh_value_1,thresh_value_2]]

		thresh_num1 = len(thresh_vec)
		list1 = []
		link_type_annot = 'peak-gene links'
		for i1 in range(thresh_num1):
			thresh_query_1 = thresh_vec[i1]
			thresh_value_1, thresh_value_2 = thresh_query_1[0:2]

			id1 = (value_1>thresh_value_1)
			id2 = (value_2>thresh_value_2)
			id_1 = (id1&id2) # peak-gene links with longer distance and lower peak-gene link score
			
			list1.append(id_1)
			query_num1 = np.sum(id_1)
			if verbose>0:
				# print('thresh_query_1, query_num1: ',thresh_value_1,thresh_value_2,query_num1)
				print('thresholds: ',thresh_value_1,thresh_value_2)
				print('the %s filtered: %d'%(link_type_annot,query_num1))

		id_1 = list1[0]
		for i2 in range(1,thresh_num1):
			id_query_1 = list1[i2]
			id_1 = (id_1|id_query_1)

		id_2 = (~id_1)
		df_link_1 = df_link_query.loc[id_2,:]
		df_link_2 = df_link_query.loc[id_1,:]

		return df_link_1, df_link_2

	## ====================================================
	# compare and select peak-gene links based on peak-gene distance and link score
	# to update
	def test_gene_peak_query_link_basic_filter_1_pre2_2(self,df_feature_link=[],column_idvec=['peak_id','gene_id'],field_query=[],column_vec_query=[],
															thresh_vec_1=[100,[0.25,0.1]],thresh_vec_2=[],type_score=0,type_id_1=0,verbose=0,select_config={}):

		"""
		compare and select peak-gene links based on peak-gene distance and link score (e.g.,peak accessibility-gene expression correlation)
		:param df_feature_link: (dataframe) annotations of peak-gene links
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param field_query: (array or list) the attributes based on which to perform peak-gene link comparison and selection
		:param column_vec_query: (array or list) columns in peak-gene link annotations which correspond to the attributes based on which to compare peak-gene links
		:param thresh_vec_1: (arrary or list) thresholds on the corresponding attributes used to select peak-gene links
		:param thresh_vec_2: (arrary or list) the second set of thresholds on the corresponding attributes used to select peak-gene links
		:param type_score: the type of peak-gene link score (e.g.,peak accessibility-gene expression correlation) used for comparison: 0: the absolute score; 1: the original score;
		:param type_id_1: the type of comparison to perform: 0: use peak-gene link with the highest score in the minimal distance bin to compare with other links;
														     1: use peak-gene link with the smallest distance in the highest score bin to compare with other links;
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) annotations of peak-gene links retained after the comparison
				 2. (dataframe) annotations of peak-gene links filtered after the comparison
		"""

		column_vec_1 = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_vec_1[0:3]

		if len(field_query)==0:
			if type_score==0:
				field_query = ['distance','correlation'] # peak-gene correlation
			else:
				field_query = ['distance','score'] # peak-TF-gene score

		field1, field2 = field_query[0:2]

		# column_1, column_2 = 'distance_bin', 'correlation_bin'
		# column_vec_1 = [column_1,column_2]
		column_distance, column_score_1 = column_vec_query[0:2]
		column_value_1_ori, column_value_2_ori = column_distance, column_score_1
	
		column_value_1 = '%s_abs'%(column_distance)
		if type_score==0:
			column_value_2 = '%s_abs'%(column_score_1)
		else:
			column_value_2 = column_score_1

		column_vec_pre1 = [column_distance,column_score_1]
		column_vec_pre2 = [column_value_1,column_value_2]

		column_vec_query1 = ['%s_min_2'%(field1),'%s_min_%s2'%(field1,field2)]	# highest score in the distance min bin and the distance with the score
		column_vec_query2 = ['%s_max_%s2'%(field2,field1),'%s_max_2'%(field2)]	# smallest distance in the highest score bin and the score with the distance
		list_column = [column_vec_query1,column_vec_query2]

		df_feature_link.index = test_query_index(df_feature_link,column_vec=column_idvec)
		df_feature_link_ori = df_feature_link.copy()
		# print('df_feature_link_ori: ',df_feature_link_ori.shape)

		column_query_1 = list_column[type_id_1]
		column_query_2 = column_vec_pre2
		if verbose>0:
			print('column_query_1, column_query_2: ',column_query_1,column_query_2)

		column_1, column_2 = column_query_1[0:2]
		thresh_distance_1 = thresh_vec_1[0]
		thresh_corr_vec = thresh_vec_1[1]

		df_link_query = df_feature_link
		thresh_corr_1, thresh_corr_2 = thresh_corr_vec[0:2]

		# perform filtering for peak-gene link with distance above the threshold
		id_query1 = (df_link_query[column_value_1_ori].abs()>thresh_distance_1)
		
		column_query = column_query_1[1]
		# the first threshold for using the link that have not low peak-gene correlation to filter the other links
		# the second threshold for not filtering link with relatively high peak-gene correlation
		id_query2 = (df_link_query[column_query]>thresh_corr_1)&(df_link_query[column_value_2]<thresh_corr_2)
		
		id_constrain = (id_query1&id_query2)
		df_link_query = df_link_query.loc[id_constrain,:]
		query_id2 = df_link_query.index
		# print('df_link_query: ',df_link_query.shape)

		query_value_1 = df_link_query.loc[:,column_query_2]
		if type_score==0:
			query_value_2 = df_link_query.loc[:,column_query_1].abs()
		else:
			query_value_2 = df_link_query.loc[:,column_query_1]
			column_1 = column_query_1[0]
			query_value_2[column_1] = query_value_2[column_1].abs()

		difference = np.asarray(query_value_1) - np.asarray(query_value_2)
		value_1 = difference[:,0]
		value_2 = -difference[:,1]

		thresh_num = len(thresh_vec_2)
		list1 = []
		link_type_annot = 'peak-gene links'
		for i1 in range(thresh_num):
			thresh_query_1 = thresh_vec_2[i1]
			thresh_value_1, thresh_value_2 = thresh_query_1[0:2]
			id1 = (value_1>thresh_value_1)
			id2 = (value_2>thresh_value_2)

			# id_1 = (id1&id2) # peak-gene link query with longer distance and lower peak accessibility-gene expression correlation
			id_1 = (id1&id2) # peak-gene links with longer distance and lower peak accessibility-gene expression correlation
			list1.append(id_1)
			query_num1 = np.sum(id_1)
			if verbose>0:
				# print('thresh_query_1, query_num1: ',thresh_value_1,thresh_value_2,query_num1)
				print('thresholds: ',thresh_value_1,thresh_value_2)
				print('the %s filtered: %d'%(link_type_annot,query_num1))

		id_1 = list1[0]
		for i2 in range(1,thresh_num):
			id_query_1 = list1[i2]
			id_1 = (id_1|id_query_1)

		df_link_2 = df_link_query.loc[id_1,:]
		query_id_pre2 = df_link_2.index
		query_id_ori = df_feature_link.index
		query_id_pre1 = query_id_ori.difference(query_id_pre2,sort=False)
		df_link_1 = df_feature_link.loc[query_id_pre1,:] # the retained feature link

		return df_link_1, df_link_2

	## ====================================================
	# pre-selection of gene-peak links
	# compare and select peak-gene links for peaks which may have alternative potential target genes
	# to update
	def test_gene_peak_query_basic_filter_1(self,peak_id=[],df_peak_query=[],df_gene_peak_query=[],df_gene_peak_distance_annot=[],
												field_query=['distance','correlation'],column_vec_query=['distance','spearmanr'],column_score='spearmanr',
												distance_bin_value=50,score_bin_value=0.05,peak_distance_thresh=2000,thresh_vec_compare=[],
												column_label='label_thresh2',thresh_type=3,flag_basic_query=3,flag_unduplicate=1,
												type_query_compare=2,type_score=0,type_id_1=0,
												save_mode=0,output_file_path='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		compare and select peak-gene links for peaks which may have alternative potential target genes
		:param peak_id: (str) name or index of the peak for which we compare peak-gene links associated with multiple potential target genes
		:param df_peak_query: (dataframe) annotations of peak attributes
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param df_gene_peak_distance_annot: (dataframe) annotations of genome-wide peak-gene links including peak-gene TSS distance
		:param field_query: (array or list) the attributes based on which to perform peak-gene link comparison and selection
		:param column_vec_query: (array or list) columns in peak-gene link annotations which correspond to the attributes based on which to compare peak-gene links
		:param column_score: (str) column representing the attribute used as peak-gene link score
		:param distance_bin_value: peak-gene TSS distance bin size (unit:Kb)
		:param score_bin_value: peak-gene link score bin size
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param thresh_vec_compare: (array or list) thresholds used for peak-gene link comparison
		:param column_label: column representing labels of peak-gene links
		:param thresh_type: the number of threshold types used for peak-gene link comparison
		:param flag_basic_query: indicator of which function to perform: 
								 1: query distance and link score related annotations for peak-gene links;
								 2: compare and select peak-gene links based on peak-gene distance and link score;
								 3: perform both 1 and 2;
		:param flag_unduplicate: indicator of whether unduplicate the dataframe
		:param type_query_compare: the type of comparison to perform for the peak-gene links
		:param type_score: the type of peak-gene link score (e.g.,peak accessibility-gene expression correlation) used for comparison: 0: the absolute score; 1: the original score;
		:param type_id_1: the type of comparison to perform: 0: use peak-gene link with the highest score in the minimal distance bin to compare with other links;
														     1: use peak-gene link with the smallest distance in the highest score bin to compare with other links;
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-gene links
		"""

		column_idvec = ['peak_id','gene_id']
		column_id2, column_id1 = column_idvec[0:2]

		filename_prefix_save_pre1 = select_config['filename_prefix_default']
		filename_prefix_save_1 = filename_prefix_save
		column_distance, column_score_1 = column_vec_query[0:2]
		field1, field2 = field_query[0:2]
		column_vec_2 = ['%s_abs'%(column_distance),'%s_abs'%(column_score_1)]
		column_value_1_ori = column_distance
		column_value_2_ori = column_score_1

		column_value_1 = '%s_abs'%(column_distance)
		if type_score==0:
			column_value_2 = '%s_abs'%(column_score_1)
		else:
			column_value_2 = column_score_1
		
		df_link_query_1 = df_gene_peak_query
		verbose_internal = self.verbose_internal
		if flag_basic_query in [1,3]:
			if type_score==0:
				input_filename_1 = select_config['input_filename_pre2']
				df_gene_peak_query_thresh1 = pd.read_csv(input_filename_1,index_col=False,sep='\t')
				# df_gene_peak_query_thresh1.index = np.asarray(df_gene_peak_query_thresh1[column_id1])

				input_filename = select_config['filename_save_thresh2']
				df_gene_peak_query_thresh2 = pd.read_csv(input_filename,index_col=False,sep='\t')
				# df_gene_peak_query_thresh2.index = np.asarray(df_gene_peak_query_thresh2[column_id1])
				if verbose_internal==2:
					print('peak-gene links selected by threshold 1, dataframe of size ',df_gene_peak_query_thresh1.shape)
					print('peak-gene links selected by threshold 2, dataframe of size ',df_gene_peak_query_thresh2.shape)

				df_gene_peak_distance_1 = df_gene_peak_distance_annot
				if len(df_gene_peak_distance_annot)>0:
					df_gene_peak_distance_1.index = np.asarray(df_gene_peak_distance_1[column_id2])
					
				list_1 = [df_gene_peak_distance_1,df_gene_peak_query_thresh1,df_gene_peak_query_thresh2]
				filename_annot_vec = ['df_gene_peak_distance','df_gene_peak_thresh1','df_gene_peak_thresh2']
				
				query_num1 = len(list_1)
				for i2 in range(query_num1):
					df_query = list_1[i2]
					if len(df_query)>0:
						query_id_1 = test_query_index(df_query,column_vec=column_idvec)
						id1 = pd.Index(query_id_1).duplicated(keep='first')
						t_value_1 = np.sum(id1)
						filename_annot_str1 = filename_annot_vec[i2]
						if verbose_internal==2:
							print('peak-gene links, dataframe of size ',df_query.shape)
							print('duplicated peak-gene links: ',t_value_1)
				
			# peak_query = df_gene_peak_query_thresh2[column_id2].unique()
			peak_query = df_gene_peak_query[column_id2].unique()
			peak_num1 = len(peak_query)
			print('peak number: %d'%(peak_num1))
			
			# type_query_1 = 0
			column_query = column_label

			type_query_1 = type_query_compare
			df_query1 = df_gene_peak_query

			if flag_unduplicate>0:
				df_query1 = df_query1.drop_duplicates(subset=column_idvec)
			df_query1.index = utility_1.test_query_index(df_query1,column_vec=column_idvec)

			df_link_query = df_query1
			print('peak-gene links, dataframe of size ',df_link_query.shape)
			print('data preview:\n ',df_link_query[0:2])

			field1, field2 = field_query[0:2]
			# column_1, column_2 = 'distance_bin', 'correlation_bin'
			column_1, column_2 = '%s_bin'%(field1), '%s_bin'%(field2)
			column_1_query, column_2_query = column_1, column_2
			
			column_distance, column_score_1 = column_vec_query[0:2]
			
			column_value_1 = '%s_abs'%(column_distance)
			df_link_query[column_value_1] = df_link_query[column_distance].abs()

			peak_distance_thresh_1 = peak_distance_thresh
			distance_abs = df_link_query[column_distance].abs()
			
			df_link_query[column_value_1] = distance_abs
			t_vec_1 = np.arange(0,peak_distance_thresh_1+distance_bin_value,distance_bin_value)
			df_link_query[column_1_query] = np.digitize(distance_abs,t_vec_1)

			score_query_abs = df_link_query[column_score_1].abs()
			n_bins_score = int(np.ceil(1.0/score_bin_value))
			t_vec_2 = np.linspace(0,1,n_bins_score+1)

			type_id_1 = 1
			if type_score==0:
				column_value_2 = '%s_abs'%(column_score_1)
				df_link_query[column_value_2] = score_query_abs
			else:
				column_value_2 = column_score_1
			
			df_link_query[column_2_query] = np.digitize(score_query_abs,t_vec_2)
			if (type_score==1):
				id1 = (df_link_query[column_score_1]<0)
				df_link_query.loc[id1,column_2_query] = -df_link_query.loc[id1,column_2_query]

			# field_query = ['distance','correlation']
			# column_vec_query = ['distance','spearmanr']
			# column_score_1 = 'spearmanr'
			print('query distance and score annotation')
			start = time.time()
			df_link_query2, df_link_query2_2 = self.test_gene_peak_query_link_basic_filter_1_pre2(df_gene_peak_query=df_link_query,
																									field_query=field_query,
																									thresh_vec_compare=[],
																									column_vec_query=column_vec_query,
																									column_label='',
																									type_id_1=type_id_1,
																									select_config=select_config)

			stop = time.time()
			print('distance and score annotation query used: %.2fs'%(stop-start))

			df_link_query2.index = np.asarray(df_link_query2['peak_id'])
			df_link_query2 = df_link_query2.sort_values(by=['peak_id',column_score_1],ascending=[True,False])
			
			filename_annot1 = '%d.%d'%(type_query_1,type_score)
			filename_annot_1 = filename_annot1
			output_filename = '%s/%s.df_link_query.1.%s.txt'%(output_file_path,filename_prefix_save_1,filename_annot_1)
			
			t_columns = df_link_query2.columns.difference(column_vec_2,sort=False)
			df_link_query2 = df_link_query2.loc[:,t_columns]
			df_link_query2.to_csv(output_filename,index=False,sep='\t')
			print('peak-gene links, dataframe of size ',df_link_query2.shape)
			print('output filename: %s'%(output_filename))

			# save data for preview
			output_filename = '%s/%s.df_link_query.1.%s.pre1.txt'%(output_file_path,filename_prefix_save_1,filename_annot_1)
			sel_num1 = 1000
			peak_query_vec = peak_query
			peak_query_1 = peak_query_vec[0:sel_num1]
			df_link_query2_1 = df_link_query2.loc[peak_query_1,:]
			df_link_query2_1.to_csv(output_filename,index=False,sep='\t')

			output_filename = '%s/%s.df_link_query.2.%s.txt'%(output_file_path,filename_prefix_save_1,filename_annot_1)
			t_columns = df_link_query2_2.columns.difference(column_vec_2,sort=False)
			df_link_query2_2 = df_link_query2_2.loc[:,t_columns]
			df_link_query2_2.to_csv(output_filename,index=False,sep='\t')
			print('peak-gene links, dataframe of size ',df_link_query2_2.shape)
			print('output filename: %s'%(output_filename))

		if flag_basic_query in [2,3]:
			type_id_1 = 1
			save_file_path2 = output_file_path
			input_file_path = save_file_path2
			# output_file_path = save_file_path2

			if flag_basic_query==2:
				filename_annot1 = '%d.%d.%d'%(type_id_1,type_query_1,type_score)
				input_filename = '%s/%s.df_link_query.1.%s.txt'%(input_file_path,filename_prefix_save_1,filename_annot1)
				
				df_link_query2_ori = pd.read_csv(input_filename,index_col=False,sep='\t')
				df_link_query2_ori.index = test_query_index(df_link_query2_ori,column_vec=column_idvec)
				if verbose_internal>0:
					print('data loaded from %s'%(input_filename))
			else:
				df_link_query2_ori = df_link_query2

			df_link_query2_ori.index = test_query_index(df_link_query2_ori,column_vec=column_idvec)
			print('peak-gene links for comparison, dataframe of size ',df_link_query2_ori.shape)

			gene_query_1 = df_link_query2_ori[column_id1].unique()
			print('gene number: %s'%(len(gene_query_1)))

			df_link_query2 = df_link_query2_ori

			if not (column_value_1 in df_link_query2.columns):
				df_link_query2[column_value_1] = df_link_query2[column_value_1_ori].abs()

			if not (column_value_2 in df_link_query2.columns):
				if type_score==0:
					df_link_query2[column_value_2] = df_link_query2[column_value_2_ori].abs()
			
			df_link_query_2 = df_link_query2.copy()
			type_query = 0
			thresh_type = 3
			if 'thresh_vec_group1' in select_config:
				thresh_vec_query = select_config['thresh_vec_group1']
				thresh_vec_query1, thresh_vec_query2 = thresh_vec_query[0:2]
				thresh_value_1, thresh_value_2 = thresh_vec_query1[0:2]
				thresh_value_1_2,thresh_value_2_2 = thresh_vec_query2[0:2]
			else:
				thresh_value_1 = 100 # distance threshold
				thresh_value_2 = 0.15 # correlation threshold
				thresh_value_1_2 = 500 # distance threshold
				thresh_value_2_2 = -0.05 # correlation threshold

				thresh_vec_query = [[thresh_value_1,thresh_value_2],[thresh_value_1_2,thresh_value_2_2]]
				# thresh_type = len(thresh_vec_query)
				select_config.update({'thresh_vec_group1':thresh_vec_query})

			if 'thresh_vec_group2' in select_config:
				thresh_vec_group2 = select_config['thresh_vec_group2']
				thresh_vec_query_1 = thresh_vec_group2[0]
				thresh_vec_query_2 = thresh_vec_group2[1]

				thresh_query3 = thresh_vec_query_2[0]
				thresh_value_1_3,thresh_value_2_3 = thresh_query3[0:2]
			else:
				thresh_value_1_3 = -50 # distance threshold
				thresh_value_2_3 = 0.20 # correlation threshold
				thresh_vec_query_1 = [150,[0.3,0.1]]
				thresh_vec_query_2 = [[thresh_value_1_3,thresh_value_2_3]]
				thresh_vec_group2 = [thresh_vec_query_1,thresh_vec_query_2]
				select_config.update({'thresh_vec_group2':thresh_vec_group2})
			
			# type_combine = 0
			type_combine = 1
			save_mode_2 = 1
			
			filename_save_annot_1 = '%s_%s.%s_%s'%(thresh_value_1,thresh_value_2,thresh_value_1_2,thresh_value_2_2)
			filename_save_annot = '%s.%d.%d'%(filename_save_annot_1,type_query_1,type_score)
			# column_query1 = 'label_thresh2'
			column_query1 = column_label

			for type_combine in [0,1]:
				df_link_query2 = df_link_query_2.copy()
				for type_query in [0,1]:
					type_query_2 = (1-type_query)
					print('select peak-gene link with smaller distance and similar or higher score ',type_query,type_combine)
					start = time.time()
					df_link_pre1, df_link_pre2 = self.test_gene_peak_query_link_basic_filter_1_pre2_1(df_feature_link=df_link_query2,
																										field_query=field_query,
																										column_vec_query=column_vec_query,
																										thresh_vec=thresh_vec_query,
																										type_score=type_score,
																										type_id_1=type_query,
																										verbose=verbose,select_config=select_config)

					print('peak-gene link group 1, dataframe of size ',df_link_pre1.shape,type_query,type_combine)
					print('peak-gene link group 2, dataframe of size ',df_link_pre2.shape,type_query,type_combine)
					stop = time.time()
					print('peak-gene link query comparison 1 used: %.2fs'%(stop-start))

					if len(thresh_vec_query_2)>0:
						print('select peak-gene link with similar or smaller distance and higher score')
						start = time.time()
						df_link_1, df_link_2 = self.test_gene_peak_query_link_basic_filter_1_pre2_2(df_feature_link=df_link_pre1,
																										field_query=field_query,
																										column_vec_query=column_vec_query,
																										thresh_vec_1=thresh_vec_query_1,
																										thresh_vec_2=thresh_vec_query_2,
																										type_score=type_score,
																										type_id_1=type_query,
																										verbose=verbose,select_config=select_config)
						stop = time.time()
						print('peak-gene link query comparison 2 used: %.2fs'%(stop-start))
						df_link_2 = pd.concat([df_link_pre2,df_link_2],axis=0,join='outer',ignore_index=False)
					else:
						df_link_1, df_link_2 = df_link_pre1, df_link_pre2
					
					print('peak-gene link group 1, dataframe of size ',df_link_1.shape,type_query,type_combine)
					print('peak-gene link group 2, dataframe of size ',df_link_2.shape,type_query,type_combine)

					if type_combine>0:
						df_link_query2 = df_link_1
					
					if (save_mode_2>0):
						if type_combine==0:
							list1 = [df_link_1,df_link_2]
						else:
							list1 = [df_link_2]
						
						query_num2 = len(list1)
						for i2 in range(query_num2):
							df_query = list1[i2]
							t_columns = df_query.columns.difference(column_vec_2,sort=False)
							if type_combine==0:
								output_filename = '%s/%s.df_link_query2.%d.%d.%s.txt'%(output_file_path,filename_prefix_save_1,(i2+1),type_query,filename_annot1)
							else:
								output_filename = '%s/%s.df_link_query2.1.combine.%d.%s.txt'%(output_file_path,filename_prefix_save_1,type_query,filename_save_annot)

							df1 = df_query.loc[:,t_columns]
							df1.to_csv(output_filename,index=False,sep='\t')
							if (i2==(query_num2-1)):
								annot_str = 'group 2'
							else:
								annot_str = 'group 1'

							print('link query in %s from the combined peak-gene links, dataframe of '%(annot_str),df1.shape,type_query,type_combine)
							print('output filename: %s'%(output_filename))

							if (type_query_1!=2) or (type_combine>0):
								df2 = df1.loc[df1[column_query1]>0,:]
								if type_combine==0:
									output_filename = '%s/%s.df_link_query2.%d.%d.%s.2.txt'%(output_file_path,filename_prefix_save_1,(i2+1),type_query,filename_annot1)
								else:
									output_filename = '%s/%s.df_link_query2.1.combine.%d.%s.2.txt'%(output_file_path,filename_prefix_save_1,type_query,filename_save_annot)

								df2.to_csv(output_filename,index=False,sep='\t')
								print('link query in %s from the pre-selected peak-gene links, dataframe of '%(annot_str),df2.shape,type_query,type_combine)
								print('output filename: %s'%(output_filename))

				df_link_query_1 = df_link_1
				if (type_combine>0) and (save_mode_2)>0:
					# list1 = [df_link_1,df_link_2]
					list1 = [df_link_1]
					query_num2 = len(list1)
					for i2 in range(query_num2):
						df_query = list1[i2]
						t_columns = df_query.columns.difference(column_vec_2,sort=False)
						output_filename = '%s/%s.df_link_query2.%d.combine.%s.txt'%(output_file_path,filename_prefix_save_1,(i2+1),filename_save_annot)
						df1 = df_query.loc[:,t_columns]
						df1.to_csv(output_filename,index=False,sep='\t')
						gene_query_vec_1 = df1[column_id1].unique()
						# print('selected link query from the combined peak-gene links, dataframe of size ',df1.shape)
						print('selected links from the combined peak-gene links, dataframe of size ',df1.shape)
						print('gene number: %d'%(len(gene_query_vec_1)))
						print('output filename: %s'%(output_filename))
						
						df2 = df1.loc[df1[column_query1]>0,:]
						output_filename = '%s/%s.df_link_query2.%d.combine.%s.2.txt'%(output_file_path,filename_prefix_save_1,(i2+1),filename_save_annot)
						df2.to_csv(output_filename,index=False,sep='\t')
						gene_query_vec_2 = df2[column_id1].unique()
						print('selected peak-gene links, dataframe of size ',df2.shape)
						print('gene number: %d'%(len(gene_query_vec_2)))
						print('output filename: %s'%(output_filename))

		return df_link_query_1

	## ====================================================
	# combine peak-gene links selected using both positive and negative correlations for link comparison with peak-gene links selected using positive correlations only for comparison
	# to update
	def test_feature_link_score_combine_pre1(self,data=[],save_mode=1,verbose=0,select_config={}):

		"""
		combine peak-gene links selected using both positive and negative correlations for alternative link comparison with peak-gene links selected using positive correlations only for comparison
		:param data: (list) potential annotation data
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) combination of peak-gene links selected using both positive and negative correlations for alternative link comparison and peak-gene links selected using positive correlations only for comparison
				 2. dictionary containing updated parameters
		
		"""

		flag_combine_query = 1
		if flag_combine_query>0:
			data_path_save_local = select_config['data_path_save_local']
			input_file_path_query = '%s/temp1'%(data_path_save_local)
			filename_prefix_1 = select_config['filename_prefix_default']
			thresh_annot = '100_0.15.500_-0.05'
			type_query_compare = select_config['type_query_compare']

			column_idvec = ['peak_id','gene_id']
			df_list1 = []
			type_score_vec_query1 = [0,2]
			for type_score in type_score_vec_query1:
				input_filename_1 = '%s/%s.df_link_query2.1.combine.%s.%d.%d.2.txt'%(input_file_path_query,filename_prefix_1,thresh_annot,type_query_compare,type_score)
				df_query = pd.read_csv(input_filename_1,index_col=False,sep='\t')
				df_query.index = test_query_index(df_query,column_vec=column_idvec)
				df_list1.append(df_query)

			df_query1, df_query2 = df_list1[0:2]
			column_corr_1 = 'spearmanr'
			column_score_1 = column_corr_1

			query_num1 = len(df_list1)
			df_list1_1 = []
			df_list1_2 = []
			for i1 in range(query_num1):
				df_query = df_list1[i1]
				id1 = (df_query[column_score_1]>0)
				id2 = (~id1)
				df_pre1 = df_query.loc[id1,:]
				df_pre2 = df_query.loc[id2,:]
				df_list1_1.append(df_pre1)
				df_list1_2.append(df_pre2)
				print('peak-gene links, dataframe of size ',df_query.shape,i1)
				print('peak-gene links with positive correlation, dataframe of size ',df_pre1.shape,i1)
				print('peak-gene links with non-positive correlation, dataframe of size ',df_pre2.shape,i1)

			df_pre1_1,df_pre1_2 = df_list1_1[0:2]
			df_pre2_1,df_pre2_2 = df_list1_2[0:2]
			query_id1_1, query_id1_2 = df_pre1_1.index, df_pre1_2.index
			query_id2_1, query_id2_2 = df_pre2_1.index, df_pre2_2.index
				
			query_id1_pre1 = query_id1_1.intersection(query_id1_2,sort=False)
			query_id1_pre2 = query_id1_1.difference(query_id1_2,sort=False)
			query_id1_pre3 = query_id1_2.difference(query_id1_1,sort=False)
			query_num1_pre1, query_num1_pre2, query_num1_pre3 = len(query_id1_pre1), len(query_id1_pre2), len(query_id1_pre3)
			# print('query_id1_pre1, query_id1_pre2, query_id1_pre3 ',query_num1_pre1,query_num1_pre2,query_num1_pre3)

			query_num2_1, query_num2_2 = len(query_id2_1), len(query_id2_2)
			 #print('query_id2_1, query_id2_2 ',query_num2_1,query_num2_2)

			df_pre1_3 = df_pre1_1.loc[query_id1_pre1,:]
			list1 = [df_pre1_2,df_pre2_1]
			df_query_1 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
			output_file_path_query = input_file_path_query
			type_score_query1, type_score_query2 = type_score_vec_query1[0:2]
			output_filename = '%s/%s.df_link_query2.1.combine.%s.%d.%d_%d.2.txt'%(output_file_path_query,filename_prefix_1,thresh_annot,type_query_compare,type_score_query1,type_score_query2)
			df_query_1 = df_query_1.drop_duplicates(subset=column_idvec)
			df_query_1.to_csv(output_filename,index=False,sep='\t')
			print('the combined peak-gene links, dataframe of size ',df_query_1.shape)
			print('output filename: %s'%(output_filename))

			column_1 = 'filename_feature_link_pre1'
			select_config.update({column_1:output_filename})

			return df_query_1, select_config

	## ====================================================
	# compare and select peak-gene links for peaks which may have alternative potential target genes
	# to update
	def test_gene_peak_query_correlation_basic_filter_pre1(self,gene_query_vec=[],df_gene_peak_query_compute=[],df_gene_peak_query=[],peak_distance_thresh=2000,peak_distance_thresh_compare=50,
															df_peak_query=[],peak_loc_query=[],atac_ad=[],rna_exprs=[],highly_variable=False,interval_peak_corr=50,interval_local_peak_corr=10,
															type_id_1=0,correlation_type=[],flag_computation_1=1,flag_combine_1=0,
															input_file_path='',input_filename='',input_filename_2='',
															save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		compare and select peak-gene links for peaks which may have alternative potential target genes
		:param gene_query_vec: (array or list) the target genes
		:param df_gene_peak_query_compute: (dataframe) annotations of peak-gene links used for comparison
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param peak_distance_thresh_compare: peak-gene TSS distance threshold used for peak-gene link comparison
		:param df_peak_query: (dataframe) annotations of peak attributes
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param highly_variable: indicator of whether to only include highly variable gene as target genes
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param interval_local_peak_corr: the number of genes in the sub-batch for which to compute peak-gene correlations in the batch mode
		:param type_id_1: the type of peak-gene link comparison
		:param correlation_type: the type of peak accessibility-gene expression correlation
		:param flag_computation_1: indicator of whether to perform peak-gene link comparison
		:param flag_combine_1: indicator of whether to combine different types of pre-selected peak gene links 
		:param input_file_path: the directory to retrieve data from 
		:param input_filename: path of the input file
		:param input_filename_2: path of the second input file
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-gene links
		"""

		# combine the pre-selected gene-peak links from different runs
		flag_basic_filter_1 = select_config['flag_basic_filter_1']
		verbose_internal = self.verbose_internal
		if flag_basic_filter_1>0:
			# df_gene_peak_query_compute1 = df_gene_peak_query_compute
			if len(df_gene_peak_query_compute)==0:
				df_gene_peak_query_compute = pd.read_csv(input_filename,index_col=False,sep='\t')	# the peak-gene links for comparison;
			
			df_gene_peak_query_1 = df_gene_peak_query
			if len(df_gene_peak_query)==0:
				if input_filename_2!='':
					df_gene_peak_query_1 = pd.read_csv(input_filename_2,index_col=False,sep='\t')	# the peak-gene links with the specific peaks
				else:
					df_gene_peak_query_1 = df_gene_peak_query_compute.copy()
			
			if verbose>0:
				print('df_gene_peak_query_compute, df_gene_peak_query_1 ',df_gene_peak_query_compute.shape,df_gene_peak_query_1.shape)

			df_pre1 = df_gene_peak_query_compute
			output_filename_pre1 = output_filename
			if not ('distance' in df_pre1.columns):
				df_gene_peak_distance = self.df_gene_peak_distance
				print('df_gene_peak_distance: ',df_gene_peak_distance.shape)
				column_idvec = ['gene_id','peak_id','motif_id']
				df_list = [df_gene_peak_query_compute,df_gene_peak_distance]
				column_query_1 = ['distance']
				reset_index = True
				# copy specified columns from the second dataframe to the first dataframe
				df_pre1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec,
															column_vec=column_query_1,
															df_list=df_list,
															type_id_1=0,type_id_2=0,
															reset_index=reset_index,
															select_config=select_config)

			# type_query: 0, df_gene_peak_query_compute is the same as df_gene_peak_query; 1, df_gene_peak_query is the subset of df_gene_peak_query_compute
			column_idvec = ['gene_id','peak_id']
			column_id1, column_id2 = column_idvec

			print('type_id_1: ',type_id_1)
			if type_id_1>0:
				# type_id_1: 0, target gene set is gene subset; 1, peak-gene link estimation for genome-wide genes
				peak_query_vec_1 = df_gene_peak_query_1[column_id2].unique()
				df_pre1.index = np.asarray(df_pre1[column_id2])

				peak_query_num1 = len(peak_query_vec_1)
				print('peak_query_vec: %d'%(peak_query_num1))
				df_pre2 = df_pre1.loc[peak_query_vec_1,:]	# peak-gene link query associated associated with given peak query
			else:
				df_pre2 = df_pre1

			column_distance = select_config['column_distance']
			if peak_distance_thresh>0:
				print('peak_distance_thresh:%d'%(peak_distance_thresh))
				df_pre2 = df_pre2.loc[df_pre2[column_distance].abs()<peak_distance_thresh,:]

				df_gene_peak_query_1 = df_gene_peak_query_1.loc[df_gene_peak_query_1[column_distance].abs()<peak_distance_thresh,:]
			
			df_pre2.index = np.asarray(df_pre2[column_id1])
			print('df_pre1, df_pre2 ',df_pre1.shape,df_pre2.shape)

			df_gene_peak_query_compute1 = df_pre2

			output_file_path = save_file_path
			save_mode_1 = 1
			print('df_peak_query: ',df_peak_query)
			if len(df_peak_query)==0:
				save_mode_1 = 1
				# query candiate peak number for each target gene and potential target gene number for each candidate peak
				df_gene_peak_query_group_1, df_gene_peak_query_group_2 = self.test_peak_gene_query_basic_1(data=df_gene_peak_query_1,input_filename='',
																											save_mode=save_mode_1,output_file_path=output_file_path,
																											filename_prefix_save=filename_prefix_save,
																											select_config=select_config)
				df_peak_query_1 = df_gene_peak_query_group_2

			else:
				df_peak_query_1 = df_peak_query
			
			peak_idvec_1 = df_peak_query_1.index
			column_correlation_1 = select_config['column_correlation'][0] # column_correlation_1:'spearmanr'
			if type_id_1==0:
				id1 = (df_peak_query_1['gene_num']>1)
				id_1 = (df_peak_query_1['gene_num']==1)
				peak_idvec_2 = peak_idvec_1[id1]	# the peak-gene link query with multiple peak-gene assignment
				peak_group1 = peak_idvec_1[id_1]	# the peak-gene link query with one peak-gene assignment
				peak_group2 = peak_idvec_2
				
				peak_num_1 = len(peak_idvec_1)
				peak_num1, peak_num2 = len(peak_group1), len(peak_group2)
				print('peak_query_vec: ',peak_num_1)
				print('peak_group1, peak_group2 ',peak_num1,peak_num2)
				
				df_gene_peak_query_1_ori = df_gene_peak_query_1.copy()
				df_gene_peak_query_1_ori.index = np.asarray(df_gene_peak_query_1_ori[column_id2])
				
				df_link_query_group1 = df_gene_peak_query_1_ori.loc[peak_group1,:]
				df_link_query_group1.index = np.asarray(df_link_query_group1[column_id1])

			parallel_mode = 0
			interval_peak_query = 100
			select_config.update({'interval_peak_query':interval_peak_query})
			print('peak-gene link query comparison')
			
			start = time.time()
			dict_query1 = dict()
			save_file_path_local = select_config['file_path_basic_filter']
			
			if 'filename_prefix_basic_filter' in select_config:
				filename_prefix_basic_filter = select_config['filename_prefix_basic_filter']
			else:
				thresh_vec_compare = select_config['thresh_vec_compare']
				str1 = '_'.join([str(query1) for query1 in thresh_vec_compare[0]])
				str2 = '_'.join([str(query1) for query1 in thresh_vec_compare[1]])
				if len(thresh_vec_compare)>2:
					str3 = '_'.join([str(query1) for query1 in thresh_vec_compare[2]])
					filename_annot_2_param = '%s_%s_%s'%(str1,str2,str3)
				else:
					filename_annot_2_param = '%s_%s'%(str1,str2)

				filename_prefix_basic_filter = '%s.%s.%s'%(filename_prefix_save,filename_annot_2_param,peak_distance_thresh_compare)
				select_config.update({'filename_prefix_basic_filter':filename_prefix_basic_filter})

			query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
			peak_query_id1, peak_query_id2 = query_id1, query_id2
			iter_mode = 0
			if (peak_query_id1>=0) and (peak_query_id2>peak_query_id1):
				# the batch query mode
				iter_mode = 1
				filename_prefix_basic_filter_1 = '%s.%d_%d'%(filename_prefix_basic_filter,peak_query_id1,peak_query_id2)
			else:
				# filename_prefix_basic_filter_1 = '%s.combine'%(filename_prefix_basic_filter)
				filename_prefix_basic_filter_1 = filename_prefix_basic_filter

			select_config.update({'iter_mode':iter_mode})

			if 'peak_distance_thresh_compare' in select_config:
				peak_distance_thresh_compare = select_config['peak_distance_thresh_compare']
				print('peak_distance_thresh_compare: %d'%(peak_distance_thresh_compare))
			
			if len(correlation_type)==0:
				correlation_type = [0] # query both positive and negative correlations
			
			if flag_computation_1>0:
				for flag_correlation_1 in correlation_type:
					select_config.update({'flag_correlation_1':flag_correlation_1})

					list_pre1 = [df_gene_peak_query_compute1,df_gene_peak_query_1]
					query_num1 = len(list_pre1)
					for i1 in range(query_num1):
						df_query = list_pre1[i1]
						df_query.index = np.asarray(df_query[column_id2])
						print('df_query: ',df_query.shape)
						
						if type_id_1==0:
							df_query = df_query.loc[peak_idvec_2,:]
							list_pre1[i1] = df_query
							print('df_query: ',df_query.shape)

						if flag_correlation_1>0:
							# consider positive peak accessibility-gene expr correlation
							df_query = df_query.loc[df_query[column_correlation_1]>0,:]
							list_pre1[i1] = df_query
					
					df_gene_peak_query_compute1,df_gene_peak_query_1 = list_pre1
					if verbose>0:
						print('df_gene_peak_query_compute1, df_gene_peak_query_1 ',df_gene_peak_query_compute1.shape,df_gene_peak_query_1.shape)
					
					type_id_correlation = flag_correlation_1
					type_id_query_1 = type_id_correlation
					file_path_save_1 = select_config['file_path_basic_filter']
					output_file_path = '%s/group%d'%(file_path_save_1,type_id_correlation)
					if os.path.exists(output_file_path)==False:
						print('the directory does not exist:%s'%(output_file_path))
						os.makedirs(output_file_path,exist_ok=True)

					filename_annot_1_ori = str(type_id_query_1)

					if iter_mode>0:
						# the batch query mode
						filename_annot_1 = '%s.%d_%d'%(filename_annot_1_ori,peak_query_id1,peak_query_id2)
					else:
						filename_annot_1 = '%s.combine'%(filename_annot_1_ori)

					output_filename_1 = '%s/%s.%s.compare1.txt'%(output_file_path,filename_prefix_basic_filter,filename_annot_1)
					output_filename_2 = '%s/%s.%s.thresh1.txt'%(output_file_path,filename_prefix_basic_filter,filename_annot_1)
					select_config.update({'filename_save_1':output_filename_1,'filename_save_2':output_filename_2})
					
					print('peak-gene link query comparison')
					start = time.time()
					column_label = 'label_corr'
					flag_combine = 1
					df_gene_peak_query_pre1, df_gene_peak_query_compare1 = self.test_gene_peak_query_basic_filter_1_local(df_gene_peak_query_ori=df_gene_peak_query_compute1,
																															df_gene_peak_query=df_gene_peak_query_1,
																															peak_distance_thresh=peak_distance_thresh_compare,
																															atac_ad=atac_ad,
																															rna_exprs=rna_exprs,
																															thresh_corr_1=0.15,
																															thresh_corr_vec=[],
																															column_label=column_label,
																															peak_query_id1=peak_query_id1,
																															peak_query_id2=peak_query_id2,
																															flag_combine=flag_combine,
																															parallel_mode=parallel_mode,
																															save_mode=1,
																															output_file_path=output_file_path,
																															output_filename='',
																															filename_prefix=filename_prefix_save,
																															verbose=verbose,
																															select_config=select_config)

					df_gene_peak_query_compare1.index = np.asarray(df_gene_peak_query_compare1[column_id1])
					df_gene_peak_query_pre1.index = np.asarray(df_gene_peak_query_pre1[column_id1])

					column_correlation = select_config['column_correlation']
					save_mode_2 = 1
					if save_mode_2>0:
						if 'column_retrieve' in select_config:
							column_vec_1 = select_config['column_retrieve']
						else:
							column_vec_1 = ['peak_id','gene_id','spearmanr','pval1','pval1_ori','distance','label_corr']
						df_query1 = df_gene_peak_query_compare1.loc[:,column_vec_1]
						t_columns_1 = df_gene_peak_query_pre1.columns
						column_vec_2 = pd.Index(column_vec_1).intersection(t_columns_1,sort=False)
						df_query2 = df_gene_peak_query_pre1.loc[:,column_vec_2]

						df_query1.to_csv(output_filename_1,sep='\t')
						df_query2.to_csv(output_filename_2,sep='\t')
						
						field_query1 = ['filename_basic_filter_save_1','filename_basic_filter_save_2']
						list1 = [output_filename_1,output_filename_2]
						query_num2 = len(field_query1)
						for i2 in range(query_num2):
							field1, filename_1 = field_query1[i2], list1[i2]
							if not (field1 in select_config):
								select_config[field1] = []
							select_config[field1].append(filename_1)

					print('df_gene_peak_query_pre1 ',df_gene_peak_query_pre1.shape)
					stop = time.time()
					print('peak-gene link query comparison used: %.5fs'%(stop-start))

					df_link_query_group2 = df_gene_peak_query_pre1
					dict_query1.update({type_id_correlation:df_link_query_group2})
					
				filename_prefix_basic_filter = select_config['filename_prefix_basic_filter']

				flag_save_2 = 0
				if flag_save_2>0:
					# filename_prefix_basic_filter_1 = '%s.%d_%d'%(filename_prefix_basic_filter,query_id1,query_id2)
					filename_basic_filter_save = '%s/%s.1.npy'%(save_file_path_local,filename_prefix_basic_filter_1)
					output_filename = filename_basic_filter_save
					np.save(output_filename,dict_query1,allow_pickle=True)
					select_config.update({'filename_basic_filter_save':filename_basic_filter_save})

			if (iter_mode==0) and (flag_combine_1)>0:
				import glob
				input_file_path = select_config['file_path_basic_filter']
				df_list1 = []
				filename_prefix_1 = filename_prefix_basic_filter
				for type_id_correlation in correlation_type:
					input_file_path1 = '%s/group%d'%(input_file_path,type_id_correlation)
					# input_filename_list = glob.glob('%s/%s_*.thresh1.txt'%(input_file_path1,filename_prefix))
					filename_1 = '%s/%s.%d.thresh1.txt'%(input_file_path1,filename_prefix_1,type_id_correlation)
					if os.path.exists(filename_1)==True:
						print('the file exists: %s'%(filename_1))
					else:
						filename_prefix_2 = '%s.%d'%(filename_prefix_1,type_id_correlation)
						input_filename_list = glob.glob('%s/%s.*.thresh1.txt'%(input_file_path1,filename_prefix_2))
						# output_filename = '%s/%s.combine.txt'%(save_file_path,filename_prefix)
						output_filename = filename_1
						df_gene_peak_query = utility_1.test_file_merge_1(input_filename_list,index_col=0,header=0,float_format=-1,output_filename=output_filename)
					
					df_list1.append(df_gene_peak_query)
					print('peak-gene links, dataframe of size ',df_gene_peak_query.shape)
					print('filename number: %d'%(len(input_filename_list)))
					if verbose_internal==2:
						print(input_filename_list)
				
				if len(df_list1)>1:
					# combine peak-gene links selected using positively-correlated and negatively-correlated peaks for link comparison with peak-gene links selected using positively-correlated peaks only for comparison
					df_link_query_group2_combine = self.test_gene_peak_query_basic_filter_combine_1(gene_query_vec=[],df_list=df_list1,
																									peak_distance_thresh=peak_distance_thresh,
																									save_mode=1,save_file_path=save_file_path_local,output_filename='',
																									filename_prefix_save=filename_prefix_save,
																									verbose=verbose,select_config=select_config)

				else:
					df_link_query_group2_combine = df_list1[0]

				# combine pre-selected gene-peak link query with peak associated with multiple genes with gene-peak link query with peak associated with one gene
				if type_id_1==0:
					list1 = [df_link_query_group1,df_link_query_group2_combine]
					df_query_pre1 = pd.concat(list1,axis=0,join='outer',ignore_index=False)
				else:
					df_query_pre1 = df_link_query_group2_combine

				if save_mode>0:
					output_file_path = save_file_path_local
					# filename_prefix_save_1 = select_config['filename_prefix_basic_filter']
					output_filename = output_filename_pre1
					if output_filename=='':
						filename_prefix_save_1 = filename_prefix_basic_filter
						output_filename = '%s/%s.combine.query1.txt'%(output_file_path,filename_prefix_save_1)

					# df_query_pre1 = df_query_pre1.sort_values(by=['gene_id','distance'],ascending=[True,True])
					df_query_pre1 = df_query_pre1.sort_values(by=['peak_id','distance'],ascending=[True,True])
					t_columns = df_query_pre1.columns.difference(['distance_abs','count'])
					df_query_pre1 = df_query_pre1.loc[:,t_columns]
					df_query_pre1.to_csv(output_filename,sep='\t')

				print('df_query_pre1 ',df_query_pre1.shape)
				return df_query_pre1

	## ====================================================
	# combine peak-gene links selected using positively-correlated and negatively-correlated peaks for link comparison with peak-gene links selected using positively-correlated peaks only for comparison
	# to update
	def test_gene_peak_query_basic_filter_combine_1(self,gene_query_vec=[],df_list=[],peak_distance_thresh=2000,input_file_path='',
														save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		combine peak-gene links selected using positively-correlated and negatively-correlated peaks for link comparison with peak-gene links selected using positively-correlated peaks only for comparison
		:param gene_query_vec: (array or list) the target genes
		:param df_list: list containing two dataframes: 1. peak-gene links selected using positively-correlated and negatively-correlated peaks for link comparison;
														2. peak-gene links selected using positively-correlated peaks only for link comparison;
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of combined selected peak-gene links
		"""

		flag_combine=1
		if flag_combine>0:
			list1 = []
			# column_idvec = ['peak','gene']
			column_idvec = ['peak_id','gene_id']
			column_id2, column_id1 = column_idvec
			if len(df_list)==0:
				peak_distance_thresh_compare = select_config['peak_distance_thresh_compare']

				thresh_vec_compare = select_config['thresh_vec_compare']
				str1 = '_'.join([str(query1) for query1 in thresh_vec_compare[0]])
				str2 = '_'.join([str(query1) for query1 in thresh_vec_compare[1]])
				if len(thresh_vec_compare)>2:
					str3 = '_'.join([str(query1) for query1 in thresh_vec_compare[2]])
					filename_annot_2_param = '%s_%s_%s'%(str1,str2,str3)
				else:
					filename_annot_2_param = '%s_%s'%(str1,str2)

				for type_id_query_1 in [0,1]:
					filename_annot_1 = '%s.%d'%(peak_distance_thresh_compare,type_id_query_1)
					filename_annot_2 = '%s.%s'%(filename_annot_2_param,filename_annot_1)
					input_filename = '%s/%s.%s.txt'%(input_file_path,filename_prefix_save,filename_annot_2)
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
					df_query.index = test_query_index(df_query,column_vec=column_idvec)
					list1.append(df_query)
			else:
				list1 = df_list
					
			df_query_1, df_query_2 = list1
			query_id1, query_id2 = df_query_1.index, df_query_2.index
			query_id_1 = query_id1.intersection(query_id2,sort=False)
			query_id_2 = query_id2.difference(query_id1,sort=False) # the links selected only in the comparison for the positively-correlated peaks

			df_query_pre1 = pd.concat([df_query_1,df_query_2.loc[query_id_2,:]],axis=0,join='outer',ignore_index=False)
			df_query_pre1 = df_query_pre1.sort_values(by=[column_id1,'distance'],ascending=True)
			df_query_pre1.index = np.asarray(df_query_pre1[column_id1])

			df_query_pre2 = df_query_2.loc[query_id_2,:]
			df_query_pre2 = df_query_pre2.sort_values(by=[column_id1,'distance'],ascending=True)
			df_query_pre2.index = np.asarray(df_query_pre2[column_id1])
			output_file_path = save_file_path
			if verbose>0:
				print('query_id_1, query_id_2 ',len(query_id_1),len(query_id_2))
				# print('df_query_pre1, df_query_1, df_query_2 ',df_query_pre1.shape,df_query_1.shape,df_query_2.shape)
				# print('df_query_pre2 ',df_query_pre2.shape)
			if save_mode>0:
				type_id_query_1 =1
				output_filename_1 = '%s/%s.combine.1.txt'%(output_file_path,filename_prefix_save)
				if os.path.exists(output_filename_1)==True:
					print('the file exists: %s'%(output_filename_1))
					output_filename_1 = '%s/%s.combine.1.2.txt'%(output_file_path,filename_prefix_save)
				df_query_pre1.to_csv(output_filename_1,sep='\t')

				output_filename = '%s/%s.difference.1.txt'%(output_file_path,filename_prefix_save)
				if os.path.exists(output_filename)==True:
					print('the file exists: %s'%(output_filename))
					output_filename = '%s/%s.difference.1.2.txt'%(output_file_path,filename_prefix_save)
				df_query_pre2.to_csv(output_filename,sep='\t')

		return df_query_pre1

	## ====================================================
	# compute peak accessibility-gene expression correlation for given peak-gene links
	# to update
	def test_gene_peak_query_correlation_local_1(self,gene_query_vec=[],df_gene_peak_query=[],column_idvec=['gene_id','peak_id'],atac_ad=[],rna_exprs=[],
													save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',select_config={}):

		"""
		compute peak accessibility-gene expression correlation for given peak-gene links
		:param gene_query_vec: (array or list) the target genes
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-gene links including computed peak accessibility-gene expression correlations
		"""

		warnings.filterwarnings('ignore')
		start1 = time.time()
		print('peak accessibility-gene expression correlation estimation')
		output_file_path1 = output_file_path
		if os.path.exists(output_file_path1)==False:
			print('the directory does not exist: %s'%(output_file_path1))
			os.mkdir(output_file_path1)

		save_file_path = output_file_path1
		save_file_path_local = output_file_path1
		interval_peak_corr_1 = select_config['interval_peak_corr']
		interval_local_peak_corr_1 = select_config['interval_local_peak_corr']

		field_query = ['flag_corr_','method_type_correlation','type_id_correlation']
		flag_corr_,method_type,type_id_1 = 1,1,1
		list1 = [flag_corr_,method_type,type_id_1]
		query_num1 = len(field_query)
		for i1 in range(query_num1):
			field_id, query_value = field_query[i1], list1[i1]
			select_config.update({field_id:query_value})

		peak_dict = dict()
		# query peak accessibility-gene expression correlation
		df_gene_peak_query2 = self.test_gene_peak_query_correlation_1(gene_query_vec=gene_query_vec,
																		peak_dict=peak_dict,
																		df_gene_peak_query=df_gene_peak_query,
																		atac_ad=atac_ad,
																		rna_exprs=rna_exprs,
																		interval_peak_corr=interval_peak_corr_1,
																		interval_local_peak_corr=interval_local_peak_corr_1,
																		peak_bg_num=-1,
																		save_file_path=save_file_path,
																		save_file_path_local=save_file_path_local,
																		filename_prefix_save=filename_prefix_save,
																		select_config=select_config)
		
		warnings.filterwarnings('default')
		stop1 = time.time()
		print('peak accessibility-gene expression correlation estimation used: %.5fs'%(stop1-start1))
		column_id1, column_id2 = column_idvec[0:2]
		df_gene_peak_query2.index = np.asarray(df_gene_peak_query2[column_id1])
			
		if (save_mode>0) and (output_filename!=''):
			df_gene_peak_query2.to_csv(output_filename,index=False,sep='\t',float_format='%.5f')

		return df_gene_peak_query2

	## ====================================================
	# compare and select peak-gene links for peaks which may have alternative potential target genes
	# to update
	def test_gene_peak_query_basic_filter_1_local(self,input_filename_1='',df_gene_peak_query_ori=[],df_gene_peak_query=[],
													peak_distance_thresh=50,atac_ad=[],rna_exprs=[],thresh_corr_1=0.15,thresh_corr_vec=[],column_label='',
													peak_query_id1=-1,peak_query_id2=-1,flag_combine=1,flag_copy=0,parallel_mode=0,type_id_1=0,
													save_mode=-1,output_file_path='',output_filename='',filename_prefix='',verbose=0,select_config={}):
		
		"""
		compare and select peak-gene links for peaks which may have alternative potential target genes
		:param input_filename_1: path of the input file
		:param df_gene_peak_query_ori: (dataframe) annotations of peak-gene links
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_distance_thresh: peak-gene TSS distance threshold used for peak-gene link comparison
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:thresh_corr_1: threshold on peak accessibility-gene expression correlation for peak-gene link comparison
		:threhs_corr_vec: (array or list) thresholds used for peak-gene link comparison
		:param column_label: column representing labels of peak-gene links
		:param peak_query_id1: the start indice of peak loci in the batch for which to perform peak-gene link comparison in the batch mode
		:param peak_query_id2: the end indice of peak loci in the batch for which to perform peak-gene link comparison in the batch mode
		:param flag_combine: indicator of whether to combine the peak-gene links selected by link comparison and the peak-gene links with peak-gene distance below the threshold
		:param flag_copy: indicator of whether to copy information from one peak-gene link annotation dataframe to the other dataframe
		:param parallel_mode: indicator of whether to perform peak-gene link comparison in parallel
		:param type_id_1: the type of peak-gene link comparison
		:return: 1. (dataframe) updated annotations of peak-gene links
		         2. (dataframe) annotations of peak-gene links retained after peak-gene link comparison and filtering
		"""

		column_idvec = ['peak_id','gene_id']
		column_id2, column_id1 = column_idvec
		# column_distance = select_config['column_distance'] # column_distance:'distance'
		column_distance = 'distance'
		df_gene_peak_query.index = test_query_index(df_gene_peak_query,column_vec=column_idvec)
		peak_distance = df_gene_peak_query[column_distance]
		# for gene-peak link selected with peaks outside +/-500Kb of gene TSS query, compare with genome-wide gene-peak links
		thresh_distance_1 = peak_distance_thresh

		# perform peak-gene link filtering for peaks outside the specific peak-gene TSS distance
		id1 = (peak_distance.abs()>thresh_distance_1)
		df_gene_peak_query_sub1_ori = df_gene_peak_query.loc[id1]
		peak_query_vec = df_gene_peak_query_sub1_ori[column_id2].unique()
		peak_query_num = len(peak_query_vec)
		print('peaks outside +/- %d Kb of gene TSS: %d '%(thresh_distance_1,peak_query_num))
		
		query_idvec = df_gene_peak_query.index
		df_gene_peak_query_ori.index = np.asarray(df_gene_peak_query_ori[column_id2])
		df_gene_peak_query_compare = df_gene_peak_query_ori.loc[peak_query_vec]

		flag_query1 = 1
		if flag_query1>0:
			start_id1, start_id2 = 0, peak_query_num
			if peak_query_id1>=0:
				start_id1 = peak_query_id1
			if (peak_query_id2>=0) and (peak_query_id2>peak_query_id1):
				start_id2 = np.min([peak_query_id2,peak_query_num])

			peak_query_vec_ori = peak_query_vec.copy()
			peak_query_vec = peak_query_vec_ori[start_id1:start_id2]
			peak_query_num = len(peak_query_vec)
			print('peak_query_vec, start_id1, start_id2 ',peak_query_num,start_id1,start_id2)

			df_gene_peak_query_compare_ori = df_gene_peak_query_compare.copy()
			# df_gene_peak_query_compare.index = np.asarray(df_gene_peak_query_compare[column_id2])
			df_gene_peak_query_compare = df_gene_peak_query_compare.loc[peak_query_vec,:]
			
			df_gene_peak_query_ori_1 = df_gene_peak_query.copy()
			df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id2])
			df_gene_peak_query = df_gene_peak_query.loc[peak_query_vec,:]

			df_gene_peak_query.index = test_query_index(df_gene_peak_query, column_vec=column_idvec)
			peak_distance = df_gene_peak_query[column_distance]
			
			id1_1 = (peak_distance.abs()>thresh_distance_1) # peak-gene distance above threshold
			df_gene_peak_query_sub1 = df_gene_peak_query.loc[id1_1,:]
			# df_gene_peak_query_sub1.index = test_query_index(df_gene_peak_query_sub1,column_vec=column_idvec)
			df_gene_peak_query_sub2 = df_gene_peak_query.loc[~id1_1, :]

			peak_query_vec_2 = df_gene_peak_query_sub1[column_id2].unique()
			print('peaks outside +/- %d Kb of gene TSS: %d '%(thresh_distance_1,len(peak_query_vec_2)))

		flag_query_subset = 0
		if (peak_query_id1>=0) or (peak_query_id2>=0):
			flag_query_subset = 1

		df_gene_peak_query_compare.index = test_query_index(df_gene_peak_query_compare,column_vec=column_idvec)
		query_id1_1 = df_gene_peak_query.index # the pre-selected peak-gene link
		query_id1 = df_gene_peak_query_sub1.index # the pre-selected peak-gene link outside the distance
		query_id_distance_1 = df_gene_peak_query_sub2.index # the pre-selected peak-gene link within the distance
		
		print('pre-selected peak-gene links, dataframe of size ',df_gene_peak_query.shape)
		print('data preview:\n',df_gene_peak_query[0:2])
		print('pre-selected peak-gene links outside distance constraint, dataframe of size ',df_gene_peak_query_sub1.shape)
		print('data preview:\n',df_gene_peak_query_sub1[0:2])
		print('peak-gene links for comparison, dataframe of size ',df_gene_peak_query_compare.shape)
		print('data preview:\n',df_gene_peak_query_compare[0:2])
		
		if column_label=='':
			column_label = 'label_corr'
		df_gene_peak_query_compare.loc[query_id1_1,column_label] = -1
		df_gene_peak_query_compare.loc[query_id1,column_label] = 1

		column_correlation = select_config['column_correlation']
		column_corr_1 = column_correlation[0]
		column_pval_ori,column_pval_1 = column_correlation[2],column_correlation[1]

		if flag_copy>0:
			field_query_1 = [column_corr_1,column_pval_ori,column_pval_1]
			# df_gene_peak_query_compare.loc[query_id1,field_query_1] = df_gene_peak_query_sub1.loc[query_id1,field_query_1]
			df_gene_peak_query_compare.loc[query_id1_1,field_query_1] = df_gene_peak_query.loc[query_id1_1,field_query_1]
		
		# prepare peak accessibility-gene expression correlation estimation
		df_compare1 = df_gene_peak_query_compare.assign(spearmanr_abs=df_gene_peak_query_compare[column_corr_1].abs(),
														distance_abs=df_gene_peak_query_compare['distance'].abs())
		df_gene_peak_query_compare = df_compare1.sort_values(by=[column_id1,'%s_abs'%(column_corr_1),'distance_abs'],ascending=[True,False,True])
		
		query_id2 = df_gene_peak_query_compare.index[pd.isna(df_gene_peak_query_compare[column_corr_1])==True] # without previously estimated peak-gene correlation;
		print('peak-gene links without correlaiton estimation: ',len(query_id2))

		# for peak-gene links without estimated peak accessibility-gene expression correlation, estimate the correlation
		# df_compare2 = df_compare1.loc[query_id2]
		df_compare2 = df_gene_peak_query_compare.loc[query_id2]
		gene_query_vec_2 = df_compare2[column_id1].unique()
		gene_query_num2 = len(gene_query_vec_2)
		
		print('gene query without peak-gene correlation estimation for the given link: ',gene_query_num2)
		df_compare2.index = np.asarray(df_compare2[column_id1])
		peak_dict = dict()

		# peak accessibility-gene expression correlation estimation for the given peak-gene links
		if gene_query_num2>0:
			output_file_path1 = '%s/temp1'%(output_file_path)
			if os.path.exists(output_file_path1)==False:
				print('the directory does not exist: %s'%(output_file_path1))
				os.mkdir(output_file_path1)

			filename_prefix_save = select_config['filename_prefix_default_1']
			if flag_query_subset>0:
				filename_prefix_save = '%s.%d_%d'%(filename_prefix_save,peak_query_id1,peak_query_id2)
			output_filename = '%s/%s.df_gene_peak.compute_2.1.txt'%(output_file_path1)
			
			# compute peak accessibility-gene expression correlation
			df_gene_peak_query_compare2 = self.test_gene_peak_query_correlation_local_1(gene_query_vec=gene_query_vec_2,
																						df_gene_peak_query=df_compare2,
																						column_idvec=column_idvec,
																						atac_ad=atac_ad,
																						rna_exprs=rna_exprs,
																						save_mode=1,output_file_path=output_file_path1,output_filename=output_filename,
																						filename_prefix_save=filename_prefix_save,
																						select_config=select_config)

			df_gene_peak_query_compare2.index = test_query_index(df_gene_peak_query_compare2,column_vec=column_idvec)
			query_id_2 = df_gene_peak_query_compare2.index
			
			field_query_2 = [column_corr_1,column_pval_ori]
			# copy estimated peak accessibility-gene expression correlation
			df_gene_peak_query_compare.loc[query_id_2,field_query_2] = df_gene_peak_query_compare2.loc[query_id_2,field_query_2]

		distance_thresh_1 = peak_distance_thresh
		filename_prefix_1 = filename_prefix
		df_gene_peak_query_1 = df_gene_peak_query_compare
		flag_compare=1
		if flag_compare>0:
			# query peaks which are nearest to gene query and also have highest correlation
			# if gene associated with the peak with the highest correlation is not the nearest gene
			# (1) if the peak is within +/-500Kb of gene TSS
			# select the peak with the highest correlation if there is not another peak with similar correlation closer to gene TSS
			# similar correlation: difference < threshold (0.1)
			# using the highest correlation to estimate a range, for peaks within the range, select the peak nearest to gene query
			# using the peak distance of peak with highest correlation to estimate a range
			# if both correlation and peak distance is similar, retain the peaks
			# (2) if the peak is not within +/-500Kb or +/-1Mb of gene TSS, use stricter criteria to retain the link
			# if there is another peak with similar correlation and closer to gene query  (within +/-500Kb or +/-1Mb of gene TSS), filter the previous link
			# query peaks with smaller distance and similar or higher correlation
			thresh_corr_1 = 0.15
			thresh1_ratio, thresh1, distance_tol_1 = 1.0, 0.15, 100

			# query peaks with higher correlation and similar or smaller distance
			thresh2_ratio, thresh2, distance_tol_2 = 1.0, -0.05, 500
			if ('thresh_vec_compare' in select_config):
				thresh_vec_compare = select_config['thresh_vec_compare']
			else:
				thresh_vec_1 = [distance_tol_1, thresh1]
				thresh_vec_2 = [distance_tol_2, thresh2]
				thresh_vec_compare = [thresh_vec_1,thresh_vec_2]
				select_config.update({'thresh_vec_compare':thresh_vec_compare})

			# df_gene_peak_query_1 = df_gene_peak_query_compare
			peak_idvec = df_gene_peak_query_1[column_id2]
			peak_query_vec = df_gene_peak_query_1[column_id2].unique()
			peak_query_num = len(peak_query_vec)
			if verbose>0:
				print('peak number: %d'%(peak_query_num))
			
			list_1 = []
			beta_mode=select_config['beta_mode']

			interval_1 = 500
			# parallel_mode=0
			if parallel_mode==0:
				for i1 in range(peak_query_num):
					peak_id = peak_query_vec[i1]
					df_query1 = df_gene_peak_query_1.loc[peak_idvec==peak_id,:]

					print_mode=0
					if i1%interval_1==0:
						print_mode=1
						print('peak_id, df_query1 ',peak_id,i1,df_query1.shape)

					# for each peak, compare the candiate peak-gene link with alternative peak-gene links; 
					# for the peak, query the retained peak-gene link after comparison and filtering
					verbose_query = print_mode
					pair_query_id1, peak_id = self.test_gene_peak_query_basic_filter_1_unit1(peak_id,df_peak_query=df_query1,
																								df_gene_peak_query=df_gene_peak_query_1,
																								thresh_vec_compare=thresh_vec_compare,
																								verbose=verbose_query,
																								select_config=select_config)
					list_1.extend(pair_query_id1)
			else:
				if 'interval_peak_query' in select_config:
					interval = select_config['interval_peak_query']
				else:
					interval = 10
				interval_num = np.int32(np.ceil(peak_query_num/interval))
				print('peak_query_vec, interval, interval_num ',peak_query_num,interval,interval_num)
				warnings.filterwarnings('ignore')
				for i1 in range(interval_num):
					start_id1 = i1*interval
					start_id2 = np.min([(i1+1)*interval,peak_query_num])
					peak_vec_1 = peak_query_vec[start_id1:start_id2]
					peak_loc_num1 = (start_id2-start_id1)
					print('peak_vec_1 ',i1,start_id1,start_id2,peak_loc_num1,peak_vec_1[0:2])
					t_vec_1 = Parallel(n_jobs=-1)(delayed(self.test_gene_peak_query_basic_filter_1_unit1)(peak_id=peak_query_vec[i1],
																											df_peak_query=df_gene_peak_query_1.loc[peak_idvec==peak_query_vec[i1],:],
																											df_gene_peak_query=[],
																											thresh_vec_compare=thresh_vec_compare,
																											verbose=(i1%500==0),
																											select_config=select_config) for i1 in range(start_id1,start_id2))

					list_query1 = [t_vec_query[0] for t_vec_query in t_vec_1]
					list_1.extend(list_query1)

				warnings.filterwarnings('default')

			pair_query_id_1 = np.asarray(list_1) # the peak-gene link query to keep
			df_gene_peak_query_1.loc[pair_query_id_1,column_label]=2 # add column label query: which peak-gene link query is pre-selected in comparison
			pair_query_num1=len(pair_query_id_1)
			# print('pair_query_id_1',pair_query_num1)

			verbose_internal = self.verbose_internal
			if verbose_internal==2:
				print('df_gene_peak_query_1 ',df_gene_peak_query_1.shape)
				print('data preview:\n',df_gene_peak_query_1[0:2])
				print('df_gene_peak_query ',df_gene_peak_query.shape)
				print('data preview:\n',df_gene_peak_query[0:2])
			
			t_columns = df_gene_peak_query_1.columns.difference(['%s_abs'%(column_corr_1),'distance_abs'],sort=False)
			df_gene_peak_query_1 = df_gene_peak_query_1.loc[:,t_columns]
			
			# link query pre-selected: above correlation and empirical p-value threshold; below correlation or empirical p-value threshold, but selecetd in multiple peak-gene link query comparison
			query_id_compare_1 = df_gene_peak_query_1.index[df_gene_peak_query_1[column_label]==2]
			
			# link query pre-selected: above correlation and empirical p-value threshold;
			query_id_compare = query_id_compare_1.intersection(df_gene_peak_query.index,sort=False)
			
			if flag_combine>0:
				# the combination of the peaks from peak-gene link pre-selection and the peaks within specific distance of the potential target gene
				query_id_1 = pd.Index(query_id_distance_1).union(query_id_compare,sort=False)
			else:
				query_id_1 = query_id_compare

			print(df_gene_peak_query)
			df_gene_peak_query_pre1 = df_gene_peak_query.loc[query_id_1,:]
			# add or update the column representing labels of peak-gene links
			df_gene_peak_query_pre1.loc[query_id_1,column_label] = df_gene_peak_query_1.loc[query_id_1,column_label]
			print('df_gene_peak_query, df_gene_peak_query_pre1 ',df_gene_peak_query.shape,df_gene_peak_query_pre1.shape)

			if (save_mode>0) and (output_filename!=''):
				df_gene_peak_query_1.to_csv(output_filename,sep='\t')

			return df_gene_peak_query_pre1, df_gene_peak_query_1

	## ====================================================
	# compare the candiate peak-gene link with alternative peak-gene links for the given peak
	# to update
	def test_gene_peak_query_basic_filter_1_unit1(self,peak_id,df_peak_query=[],df_gene_peak_query=[],thresh_vec_compare=[],column_label='',verbose=0,select_config={}):

		"""
		compare the candiate peak-gene link with alternative peak-gene links for the given peak
		:param peak_id: (str) name or index of the peak for which we compare peak-gene links associated with multiple potential target genes
		:param df_peak_query: (dataframe) annotations of peak attributes
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param thresh_vec_compare: (array or list) thresholds used for peak-gene link comparison
		:param column_label: column representing labels of peak-gene links
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (tuplet) 1. indices of retained peak-gene links after link comparison
				 		  2. the peak name or index
		"""

		if len(df_peak_query)==0:
			df_peak_query = df_gene_peak_query.loc[df_gene_peak_query['peak_id']==peak_id,:]

		df_query1 = df_peak_query
		column_idvec = ['gene_id','peak_id']
		column_id1, column_id2 = column_idvec[0:2]

		query_id_ori_1 = df_query1.index.copy()
		df_query1.index = np.asarray(df_query1['gene_id'])
		if column_label=='':
			column_label = 'label_corr'
		
		id_query_1 = (df_query1[column_label]>0) # pre-selected peak loci
		id_query_2 = (~id_query_1) # peak loci not selected or selected and within distance
		
		df_query1_1 = df_query1.loc[id_query_1,:]
		df_query1_2 = df_query1.loc[id_query_2,:]
		gene_query_idvec_pre1 = df_query1['gene_id']

		gene_idvec = df_query1['gene_id']
		gene_idvec_1 = gene_idvec[id_query_1] # gene associated with pre-selected peak-gene links
		gene_query_num1 = len(gene_idvec_1)

		gene_idvec_2 = gene_idvec[id_query_2] # peak-gene link not estimated before
		gene_query_num2 = len(gene_idvec_2)

		column_correlation = select_config['column_correlation']
		column_corr_1 = column_correlation[0] # 'spearmanr'
		column_pval_1 = column_correlation[1] # 'pval1_ori'

		column_value_1_ori, column_value_2_ori = 'distance',column_corr_1
		column_value_2 = '%s_abs'%(column_corr_1)
		column_value_1 = 'distance_abs'
		if not(column_value_2 in df_query1):
			df_query1[column_value_2] = df_query1[column_value_2_ori].abs()
		if not(column_value_1 in df_query1):
			df_query1[column_value_1] = df_query1[column_value_1_ori].abs()

		corr_value_abs = df_query1[column_value_2]	# absolute correlation value 
		pvalue = df_query1[column_pval_1]
		peak_distance_abs = df_query1[column_value_1]	# peak-gene distance

		if verbose>0:
			print('peak_id, peak-gene links 1, peak-gene links 2\n',peak_id,len(df_query1_1),len(df_query1_2))
			print(df_query1_1)
			print(df_query1_2)

		df1 = df_query1.loc[gene_idvec_1,[column_value_2,column_value_1]]
		thresh_num1 = len(thresh_vec_compare)

		mtx_1 = np.asarray(thresh_vec_compare)
		corr_value = np.asarray(df1[column_value_2])
		distance_1 = np.asarray(df1[column_value_1])

		thresh_corr_1 = np.asarray([corr_value+mtx_1[i1,1] for i1 in range(thresh_num1)])	# correlation threshold
		thresh_distance_1 = np.asarray([distance_1-mtx_1[i1,0] for i1 in range(thresh_num1)])	# distance threshold
		
		thresh_2 = np.ravel(thresh_corr_1)	# shape: (gene_query_num1*thresh_num1,)
		thresh_1 = np.ravel(thresh_distance_1)	# shape: (gene_query_num1*thresh_num1,)
		thresh_1 = np.asarray([np.max([thresh_value,0]) for thresh_value in thresh_1])

		query_num1 = gene_query_num1*thresh_num1
		
		query_value_2 = np.asarray(df_query1[column_value_2_ori])	# original correlation value
		query_value_1 = np.asarray(df_query1[column_value_1])	# absolute distance value

		mask_2 = np.asarray([query_value_2>thresh_value_2 for thresh_value_2 in thresh_2])  # comparison of correlation value
		mask_1 = np.asarray([query_value_1<thresh_value_1 for thresh_value_1 in thresh_1])  # comparison of distance value
		mask_3 = np.asarray(mask_1&mask_2).T
		
		df_compare = pd.DataFrame(index=np.asarray(gene_idvec),columns=np.arange(query_num1),data=mask_3)
		t_value_1 = np.asarray((df_compare.max(axis=0)>0))
		df2 = pd.DataFrame(index=np.asarray(gene_idvec_1),columns=np.arange(thresh_num1),data=t_value_1.reshape((thresh_num1,gene_query_num1)).T)
		id1 = (df2.max(axis=1)==0)
		
		gene_idvec_1 = pd.Index(gene_idvec_1)
		gene_query_vec_1 = gene_idvec_1[id1]
		# gene_query_vec_2 = gene_idvec_1[id2]

		query_id1 = ['%s.%s'%(peak_id,gene_id) for gene_id in gene_query_vec_1]
		gene_idvec_1 = np.asarray(gene_idvec_1)
		if verbose>0:
			df_thresh_1 = pd.DataFrame(index=gene_idvec_1,columns=np.arange(thresh_num1),data=thresh_1.reshape((thresh_num1,gene_query_num1)).T)
			df_thresh_2 = pd.DataFrame(index=gene_idvec_1,columns=np.arange(thresh_num1),data=thresh_2.reshape((thresh_num1,gene_query_num1)).T)
			print('peak_id, gene_query_vec_1, df_thresh_1, df_thresh_2 ',peak_id,len(gene_query_vec_1),gene_query_vec_1,df_thresh_1.shape,df_thresh_2.shape)
			print(df_thresh_1,peak_id)
			print(df_thresh_2,peak_id)
			print(df2,peak_id)

		return (query_id1,peak_id)

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)

