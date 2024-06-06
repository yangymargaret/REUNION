#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData

from copy import deepcopy

import warnings
import sys

import os
import os.path
from optparse import OptionParser

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
from sklearn.pipeline import make_pipeline

import time
from timeit import default_timer as timer

import utility_1
from test_reunion_compute_pre1 import _Base_pre1
import pickle

class _Base_pre2(_Base_pre1):
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

		_Base_pre1.__init__(self,file_path=file_path,
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
	# load motif scanning data
	def test_load_motif_data_1(self,method_type_vec=[],input_file_path='',save_mode=1,save_file_path='',verbose=0,select_config={}):
		
		"""
		load motif scanning data, including: 1. binary matrix presenting motif presence in each ATAC-seq peak locus; 2. motif scores if available;
		:param method_type_vec: (array or list) methods used to initially predict peak-TF links, which require motif scanning results
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing motif scanning results utilized by the corresponding regulatory association inference method in method_type_vec;
				 2. dictionary containing updated parameters
		"""

		flag_query1=1
		method_type_num = len(method_type_vec)
		dict_motif_data = dict()
		data_file_type = select_config['data_file_type']
		
		for i1 in range(method_type_num):
			method_type = method_type_vec[i1]
			motif_data_pre1, motif_data_score_pre1 = [], []

			method_annot_vec = ['insilico','joint_score','Unify','CIS-BP','CIS_BP'] # the method type which share motif scanning results
			flag_1 = self.test_query_method_type_motif_1(method_type=method_type,method_annot_vec=method_annot_vec,select_config=select_config)
			
			dict_query = dict()
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

					flag_query2 = 1
					# load motif data
					motif_data, motif_data_score, df_annot, type_id_query = self.test_load_motif_data_pre1(input_filename_list1=input_filename_list1,
																											input_filename_list2=input_filename_list2,
																											flag_query=flag_query2,
																											type_id_1=0,
																											input_file_path=input_file_path,
																											save_file_path=save_file_path,
																											select_config=select_config)
					
					# dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
					motif_data_pre1 = motif_data
					motif_data_score_pre1 = motif_data_score
				else:
					motif_data, motif_data_score = motif_data_pre1, motif_data_score_pre1
					print('loaded motif scanning data (binary), dataframe of ',motif_data.shape)
					print('loaded motif scores, dataframe of ',motif_data_score.shape)

				dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
			
			dict_motif_data[method_type] = dict_query
			# print('dict_query: ',dict_query,method_type)

		return dict_motif_data, select_config

	## ====================================================
	# query if a method utilizes motif scanning results using the CIS-BP motif collection
	def test_query_method_type_motif_1(self,method_type='',method_annot_vec=[],select_config={}):
		
		"""
		query if a method utilizes motif scanning results using the CIS-BP motif collection
		:param method_type: (str) the method used to initially predict peak-TF links, which requires motif scanning results
		:param method_annot_vec: (array or list) part of the method name which indicates the method type
		:param select_config: dictionary containing parameters
		:return: bool variable representing whether the method utilizes the motif scanning results using the CIS-BP motif collection
		"""

		if len(method_annot_vec)==0:
			method_annot_vec = ['insilico','joint_score','Unify','CIS-BP','CIS_BP','TOBIAS'] # the method type which share motif scanning results

		flag_1 = False
		for method_annot_1 in method_annot_vec:
			flag_1 = (flag_1|(method_type.find(method_annot_1)>-1))

		return flag_1

	## ====================================================
	# load motif scanning data
	def test_load_motif_data_pre1(self,input_filename_list1=[],input_filename_list2=[],flag_query=1,overwrite=True,
									type_id_1=0,input_file_path='',save_mode=1,save_file_path='',select_config={}):

		"""
		load motif scanning data, including: 1. dataframe of binary TF motif presence in each ATAC-seq peak locus; 2. motif scores if available;
		:param input_filename_list1: (array or list) paths of the files saving the AnnData objects of motif scanning binary matrix and the motif score matrix
		:param input_filename_list2: (array or list) paths of the orignal files (csv or txt format) saving the motif scanning binary matrix and the motif score matrix
		:param flag_query: indicator of whether to query if there are negative motif scores
		:param overwrite: indicator of whether to overwrite the current file
		:param type_id_1: indicating if the motif scanning data were loaded from AnnData objects (type_id_1=0) or loaded from the original files (csv or txt format) of motif scanning results (type_id_1=1)
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the binary motif scanning results, indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF);
				 2. (dataframe) motif scores from motif scanning results (row:ATAC-seq peak locus, column:TF;
				 3. (dataframe) the annotations which show the mapping between the TF binding motif name and the TF name;
				 4. indicator of whether the motif scanning data were loaded from AnnData objects (type_id_query=0) or loaded from the original files (csv or txt format) of motif scanning results (type_id_query=1);
		"""
	
		flag_pre1=0
		motif_data, motif_data_score = [], []
		type_id_query = type_id_1
		df_annot = []
		if len(input_filename_list1)>0:
			# load data from the processed AnnData object
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
			
				print('motif scanning data (binary), dataframe of ', motif_data.shape)
				print('data preview: ')
				print(motif_data[0:2])
				print('motif scores, dataframe of ', motif_data_score.shape)
				print('data preview: ')
				print(motif_data_score[0:2])
				flag_pre1 = 1

		# load the original motif data
		if flag_pre1==0:
			print('load the motif scanning data')
			input_filename1, input_filename2 = input_filename_list2[0:2]

			print('filename of motif scanning data (binary) ',input_filename1)
			print('filename of motif scores ',input_filename2)

			# motif_data, motif_data_score = [], []
			if os.path.exists(input_filename1)==False:
				print('the file does not exist: %s'%(input_filename1))
			else:
				b = input_filename1.find('.csv')
				if b>=0:
					symbol_1 = ','
				else:
					symbol_1 = '\t'
				motif_data = pd.read_csv(input_filename1,index_col=0,sep=symbol_1)
				print('motif scanning data, dataframe of ',motif_data.shape)
				print('data preview: ')
				print(motif_data[0:2])
			
			if os.path.exists(input_filename2)==False:
				print('the file does not exist: %s'%(input_filename2))
			else:
				b = input_filename2.find('.csv')
				if b>=0:
					symbol_1 = ','
				else:
					symbol_1 = '\t'
				motif_data_score = pd.read_csv(input_filename2,index_col=0,sep=symbol_1)
				print('motif scores, dataframe of ',motif_data_score.shape)
				print('data preview: ')
				print(motif_data_score[0:2])

			if len(motif_data)==0:
				if len(motif_data_score)>0:
					# motif_data = (motif_data_score>0)
					motif_data = (motif_data_score.abs()>0)
				else:
					print('please provide motif scanning data')
					return
			else:
				if len(motif_data_score)>0:
					motif_data_2 = (motif_data_score.abs()>0)*1.0
					# difference = np.abs(motif_data-motif_data_1)
					difference = np.abs(motif_data-motif_data_2)
					assert np.max(np.max(difference))==0

					# query motif name
					motif_name_ori = motif_data.columns
					motif_name_score_ori = motif_data_score.columns
					peak_loc = motif_data.index
					peak_loc_1 = motif_data_score.index

					assert list(motif_name_ori)==list(motif_name_score_ori)
					assert list(peak_loc)==list(peak_loc_1)

			# motif name conversion
			input_filename_translation = select_config['filename_translation']
			df_annot = []
			type_id_query = 1
			# overwrite = 0
			if os.path.exists(input_filename_translation)==False:
				print('the file does not exist: %s'%(input_filename_translation))
				output_filename = input_filename_translation
				# meta_scaled_exprs = self.meta_scaled_exprs
				# df_gene_annot = []
				df_gene_annot = self.df_gene_annot_ori
				df_annot = self.test_translationTable_pre1(motif_data=motif_data,
															df_gene_annot=df_gene_annot,
															flag_cisbp_motif=1,
															save_mode=1,
															save_file_path=save_file_path,
															output_filename=output_filename,
															select_config=select_config)
			else:
				print('load TF motif name mapping file')
				df_annot = pd.read_csv(input_filename_translation,index_col=0,sep='\t')

			# motif name correction for the conversion in R
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

			print('motif scanning data, dataframe of ',motif_data.shape)
			print('data preview: ')
			print(motif_data[0:2])
			# print('motif_data_ori ',motif_data_ori.shape)
			# print(motif_data_ori[0:2])

			print('perform motif name conversion ')
			motif_data = self.test_query_motif_name_conversion_1(motif_data)

			if len(motif_data_score)>0:
				# motif_data_score.columns = motif_name_query # TODO: should update
				motif_data_score, motif_data_score_ori = self.test_load_motif_data_pre2(motif_data=motif_data_score,
																						df_annot=df_annot,
																						column_id=column_id,
																						select_config=select_config)

				print('motif scores, dataframe of ',motif_data_score.shape)
				print('data preview: ')
				print(motif_data_score[0:2])

				# print('motif_data_score_ori ',motif_data_score_ori.shape)
				# print(motif_data_score_ori[0:2])

				print('perform motif name conversion ')
				motif_data_score = self.test_query_motif_name_conversion_1(motif_data_score)

			if save_mode>0:
				# output_filename_list = input_filename_list1
				column_1 = 'filename_list_save_motif'
				# the filename to save the motif data
				if column_1 in select_config:
					output_filename_list = select_config[column_1]
				else:
					data_file_type = select_config['data_file_type']
					if save_file_path=='':
						save_file_path = select_config['file_path_motif']

					output_file_path = save_file_path
					output_filename1 = '%s/test_motif_data.%s.h5ad'%(output_file_path,data_file_type)
					output_filename2 = '%s/test_motif_data_score.%s.h5ad'%(output_file_path,data_file_type)
					output_filename_list = [output_filename1,output_filename2]

				output_filename1, output_filename2 = output_filename_list
				motif_data_ad = utility_1.test_save_anndata(motif_data,sparse_format='csr',obs_names=None,var_names=None,dtype=motif_data.values.dtype)

				if os.path.exists(output_filename1)==True:
					print('the file exists ', output_filename1)

				column_query = 'motif_data_rewrite'
				if column_query in select_config:
					overwrite = select_config[column_query]
				if (os.path.exists(output_filename1)==False) or (overwrite==True):
					motif_data_ad.write(output_filename1)
					print('save motif scanning data ',motif_data_ad)
					print(output_filename1)

				if len(motif_data_score)>0:
					motif_data_score_ad = utility_1.test_save_anndata(motif_data_score,sparse_format='csr',obs_names=None,var_names=None,dtype=motif_data_score.values.dtype)

					if (os.path.exists(output_filename2)==False) or (overwrite==True):
						motif_data_score_ad.write(output_filename2)
						print('save motif score data',motif_data_score_ad)
						print(output_filename2)

		if flag_query>0:
			df1 = (motif_data_score<0)
			id2 = motif_data_score.columns[df1.sum(axis=0)>0]
			if len(id2)>0:
				motif_data_score_ori = motif_data_score.copy()
				count1 = np.sum(np.sum(df1))
				print('there are negative motif scores ',count1)
				
		return motif_data, motif_data_score, df_annot, type_id_query

	## ====================================================
	# prepare the translationTable dataframe which show the mapping between the TF binding motif name and the TF name
	def test_translationTable_pre1(self,motif_data=[],motif_data_score=[],df_gene_annot=[],rna_exprs=[],flag_cisbp_motif=1,flag_expr=0,
										save_mode=1,save_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		prepare the translationTable dataframe which show the mapping between the TF binding motif name and the TF name
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param motif_data_score: (dataframe) motif scores from motif scanning results (row:ATAC-seq peak locus, column:TF)
		:param df_gene_annot: (dataframe) gene annotations
		:param flag_cisbp: indicator of whether the motif scanning results are based on using the curated CIS-BP motif collection from the chromVAR repository
		:param flag_expr: indicator of whether to query if the TF associated with a TF motif is expressed in the RNA-seq data:
						  0: not querying if a TF is expressed;
						  1: query if a TF is expressed, and use gene identifier (ENSEMBL id) to identify genes with expressions;
						  2: query if a TF is expressed, and use gene name to identify genes with expressions;
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) the annotations which show the mapping between the TF binding motif name and the TF name
		"""

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

			# df1['gene_id'] = df1['motif_id'].str.split('_').str.get(0) # ENSEMBL id
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
					# query TF name by gene id
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
					# query TF name query by gene name
					gene_name_expr = self.rna_meta_ad.var_names
					output_file_path = select_config['data_path_save']
					output_filename_2 = '%s/test_rna_meta_ad.df_var.query1.txt'%(output_file_path)
					df_var.to_csv(output_filename_2,sep='\t')
					motif_query_name_expr = pd.Index(tf_name).intersection(gene_name_expr,sort=False)
					df1.index = np.asarray(df1['tf'])
					df1.loc[motif_query_name_expr,'tf_expr'] = 1
					
				df1.index = np.asarray(df1['gene_id'])
				self.motif_query_name_expr = motif_query_name_expr
				print('the number of TFs with expressions: %d'%(len(motif_query_name_expr)))

			if save_mode>0:
				if output_filename=='':
					output_filename = '%s/translationTable.csv'%(save_file_path)
				df1.to_csv(output_filename,sep='\t')

		return df1

	## ====================================================
	# merge multiple columns in the motif presence or motif score matrix that correspond to one TF to one column
	def test_load_motif_data_pre2(self,motif_data,df_annot,column_id='tf',select_config={}):

		"""
		merge multiple columns in the motif presence or motif score matrix that correspond to one TF to one column
		:param motif_data: (dataframe) the binary motif presence matrix or motif score matrix by motif scanning (row:ATAC-seq peak locus, column:TF motif)
		:param df_annot: (dataframe) the annotations which show the mapping between the TF binding motif name and the TF name
		:param column_id: column in df_annot which corresponds to the TF name
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the binary motif presence matrix or motif score matrix after column merging (row:ATAC-seq peak locus, column:TF)
				 2. (dataframe) the original binary motif presence matrix or motif score matrix by motif scanning (row:ATAC-seq peak locus, column:TF motif)
		"""

		motif_idvec = motif_data.columns.intersection(df_annot.index,sort=False)
		motif_data = motif_data.loc[:,motif_idvec]
		motif_data_ori = motif_data.copy()
		motif_data1 = motif_data.T
		motif_idvec = motif_data1.index  # original motif id
		motif_data1.loc[:,'tf'] = df_annot.loc[motif_idvec,column_id]
		motif_data1 = motif_data1.groupby('tf').max()
		motif_data = motif_data1.T
		
		query_idvec = np.asarray(df_annot['motif_id'])
		query_num1 = len(query_idvec)
		t_vec_1 = np.random.randint(query_num1,size=5)
		for iter_id1 in t_vec_1:
			motif_id_query = query_idvec[iter_id1]
			column_1 = motif_id_query
			column_2 = np.asarray(df_annot.loc[df_annot['motif_id']==motif_id_query,'tf'])[0]

			difference = (motif_data_ori[column_1].astype(int)-motif_data[column_2].astype(int)).abs().max()
			assert difference<1E-07

		# print('data preview: ')
		# print(motif_data[0:5])
		# field_id = '%s.ori'%(key_query)
		# if not (field_id in dict_query):
		# 	dict_query.update({'%s.ori'%(key_query):motif_data_ori})
		return motif_data, motif_data_ori

	## ====================================================
	# TF motif name conversion for motifs in the used curated CIS-BP motif collection
	def test_query_motif_name_conversion_1(self,data=[],select_config={}):

		"""
		TF motif name conversion for motifs in the used curated CIS-BP motif collection
		:param data: (dataframe) the binary motif presence matrix or motif score matrix by motif scanning (row:ATAC-seq peak locus, column:TF)
		:param select_config: dictionary containing parameters
		:return: (dataframe) the binary motif presence matrix or motif score matrix after TF name conversion (row:ATAC-seq peak locus, column:TF)
		"""

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

	## ====================================================
	# load motif scanning data
	def test_load_motif_data_2(self,data=[],dict_motif_data={},method_type='',save_mode=1,verbose=0,select_config={}):
		
		"""
		load motif scanning data
		:param dict_motif_data: dictionary containing motif scanning results utilized by the corresponding regulatory association inference method
		:param method_type: (str) the method used for initial peak-TF association prediction, which requires motif scanning results
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the binary motif scanning results, indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF with expressions);
				 2. (dataframe) motif scores from motif scanning results (row:ATAC-seq peak locus, column:TF with expressions);
				 3. (pandas.Series) names of the TFs with expressions in the RNA-seq data;
		"""

		motif_data = self.motif_data
		if method_type=='':
			method_type_feature_link = select_config['method_type_feature_link']
			method_type = method_type_feature_link
		method_type_query = method_type
		if len(motif_data)>0:
			motif_data_score = self.motif_data_score
			motif_query_name_expr = self.motif_query_name_expr
		else:
			if len(dict_motif_data)==0:
				dict_motif_data_query_1 = self.dict_motif_data
				if len(dict_motif_data_query_1)>0:				
					dict_motif_data = dict_motif_data_query_1[method_type_query]

			if len(dict_motif_data)==0:
				print('load motif scanning data')
				input_dir = select_config['input_dir']
				file_path_1 = input_dir

				method_type_vec_query = [method_type_query]
				data_path_save_local = select_config['data_path_save_local']
				file_path_motif = data_path_save_local
				select_config.update({'file_path_motif':file_path_motif})
				save_file_path = data_path_save_local

				dict_motif_data_query_1, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																						save_mode=1,save_file_path=save_file_path,
																						select_config=select_config)

				self.dict_motif_data = dict_motif_data_query_1
				dict_motif_data = dict_motif_data_query_1[method_type_query]

			if len(dict_motif_data)>0:
				# field_query = ['motif_data','motif_data_score','motif_query_name_expr']
				field_query = ['motif_data','motif_data_score']
				list1 = [dict_motif_data[field1] for field1 in field_query]
				motif_data, motif_data_score = list1[0:2]

				column_1 = 'motif_query_name_expr'
				if column_1 in dict_motif_data:
					motif_query_name_expr = dict_motif_data[column_1]
				else:
					motif_query_vec_pre1 = motif_data.columns
					rna_exprs = self.meta_exprs_2
					gene_query_name_ori = rna_exprs.columns
					motif_query_name_expr = motif_query_vec_pre1.intersection(gene_query_name_ori,sort=False)

				motif_data = motif_data.loc[:,motif_query_name_expr]
				motif_data_score = motif_data_score.loc[:,motif_query_name_expr]
				self.motif_data = motif_data
				self.motif_data_score = motif_data_score
				self.motif_query_name_expr = motif_query_name_expr

		return motif_data, motif_data_score, motif_query_name_expr

	## ====================================================
	# query correlation and mutual information between chromVAR scores and TF expressions
	def test_chromvar_score_query_1(self,input_filename,motif_query_name_expr,df_query=[],type_id_query=0,input_file_path='',output_file_path='',output_filename='',filename_prefix_save='',select_config={}):

		"""
		query correlation and mutual information between chromVAR scores and TF expressions
		:param input_filename: (str) path of the file which saved chromVAR scores of the TFs
		:param motif_query_name_expr: (array or list) names of the TFs with expressions in the RNA-seq data
		:param df_query: (dataframe) the annotations which show the mapping between the TF binding motif name and the TF name
		:param type_id_query: indicator of whether to perform column name coversion for the chromVAR score dataframe
		:param input_file_path: the directory to retrieve data from
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of TFs including the correlation and mutual information between chromVAR scores of the TFs and the TF expressions
		"""

		df1 = df_query
		# df1.index = np.asarray(df1['motif_name_ori'])
		df1.index = np.asarray(df1['motif_id'])
		chromvar_score = pd.read_csv(input_filename,index_col=0,sep=',')
		print('chromvar_score, dataframe of size ', chromvar_score.shape)
		sample_id1 = chromvar_score.columns
		motif_id1 = chromvar_score.index
		chromvar_score.index = df1.loc[motif_id1,'tf']
		if type_id_query==1:
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
		
		if output_filename=='':
			b = input_filename.find('.csv')
			output_filename = input_filename[0:b]+'copy1.csv'
		chromvar_score.to_csv(output_filename)
		print('chromVAR scores of TFs, dataframe of size ',chromvar_score.shape)

		chromvar_score = chromvar_score.T
		sample_id = meta_scaled_exprs.index
		chromvar_score = chromvar_score.loc[sample_id,:]

		motif_query_vec = motif_query_name_expr
		motif_query_num = len(motif_query_vec)
		print('the number of TFs with expressions: %d'%(motif_query_num))
		
		field_query_1 = ['spearmanr','pval1','pearsonr','pval2','mutual_info']
		df_1 = pd.DataFrame(index=motif_query_vec,columns=field_query_1)
		from scipy.stats import pearsonr, spearmanr
		from sklearn.feature_selection import mutual_info_regression
		for i1 in range(motif_query_num):
			motif_query1 = motif_query_vec[i1]
			tf_expr_1 = np.asarray(meta_scaled_exprs[motif_query1])
			tf_score_1 = np.asarray(chromvar_score[motif_query1])
			corr_value_1, pval1 = spearmanr(tf_expr_1,tf_score_1)
			corr_value_2, pval2 = pearsonr(tf_expr_1,tf_score_1)
			t_mutual_info = mutual_info_regression(tf_expr_1[:,np.newaxis], tf_score_1, discrete_features=False, n_neighbors=5, copy=True, random_state=0)
			t_mutual_info = t_mutual_info[0]
			df_1.loc[motif_query1,:] = [corr_value_1,pval1,corr_value_2,pval2,t_mutual_info]

		df_1 = df_1.sort_values(by=field_query_1,ascending=[False,True,False,True,False])
		
		filename = output_filename
		b = filename.find('.csv')
		output_filename = '%s.annot1.txt'%(filename[0:b])
		field_query_2 = ['highly_variable','means','dispersions','dispersions_norm']
		df_gene_annot_expr = self.df_gene_annot_expr
		df_gene_annot_expr.index = np.asarray(df_gene_annot_expr['gene_name'])
		motif_id1 = df_1.index
		df_1.loc[:,field_query_2] = df_gene_annot_expr.loc[motif_id1,field_query_2]
		df_1.to_csv(output_filename,sep='\t',float_format='%.6E')
		mean_value = df_1.mean(axis=0)
		median_value = df_1.median(axis=0)

		df_2 = df_1.sort_values(by=['highly_variable','dispersions_norm','means','spearmanr','pval1','pearsonr','pval2','mutual_info'],ascending=[False,False,False,False,True,False,True,False])
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
		print('highly variable TF expressions, mean_value, median_value ',motif_query_num2,mean_value,median_value)
		print('the other TF expressions, mean_value, median_value ',motif_query_num3,mean_value_2,median_value_2)

		return df_2
	
	## ====================================================
	# query gene annotations
	# matching gene names between the gene annotations and the gene expression data
	def test_gene_annotation_query_pre1(self,flag_query1=0,flag_query2=0,flag_query3=0,input_file_path='',select_config={}):

		"""
		query gene annotations;
		matching gene names between the gene annotations and the gene expression data;
		:param flag_query1: indicator of whether to query gene annotations for genes with expressions in the data
		:param flag_query2: indicator of whether to merge annotations of the given genes from different versions of gene annotations
		:param flag_query3: indicator of whether to query transcription start sites of genes
		:param input_file_path: the directory to retrieve data from 
		:param select_config: dictionary containing parameters
		return: 1. (dataframe) gene annotations of the genome-wide genes;
				2. (dataframe) gene annotations of genes with expressions merged from different verisons of gene annotations
				3. (dataframe) gene annotations of genes with expressions, including the transcription start site information;
		"""

		# query gene names and matching gene names between the gene annotations and the gene expression data
		flag_gene_annot_query_1=flag_query1
		df_gene_annot1, df_gene_annot2, df_gene_annot3 = [], [], []
		if flag_gene_annot_query_1>0:	
			filename_prefix_1 = 'hg38'
			filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
			# gene_annotation_filename = '%s/hg38.1.txt'%(input_file_path1)
			# gene_annotation_filename = '%s/Homo_sapiens.GRCh38.108.1.txt'%(input_file_path1)
			# gene_annotation_filename = '%s/%s.1.txt'%(input_file_path1,filename_prefix_1)
			gene_annotation_filename = '%s/%s.1.txt'%(input_file_path,filename_prefix_1)
			if len(self.df_gene_annot_expr)==0:
				df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2 = self.test_gene_annotation_load_1(input_filename=gene_annotation_filename,select_config=select_config)
				self.df_gene_annot_ori = df_gene_annot_ori
				self.df_gene_annot_expr = df_gene_annot_expr

				output_file_path1 = self.save_path_1
				output_file_path2 = input_file_path
				output_file_path_list = [output_file_path1,output_file_path2,output_file_path2]
				filename_prefix_list = ['ori','expr','expr_2']
				df_annot_list = [df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2]
				for i1 in range(3):
					output_filename = '%s/test_gene_annot_%s.%s.1.txt'%(output_file_path_list[i1],filename_prefix_list[i1],filename_prefix_1)
					df_annot_query = df_annot_list[i1]
					df_annot_query.to_csv(output_filename,index=False,sep='\t')

			df_gene_annot = self.df_gene_annot_ori
			df_gene_annot.index = np.asarray(df_gene_annot['gene_name'])

			print('df_gene_annot ',df_gene_annot.shape)
			print(df_gene_annot)
			df_gene_annot1 = df_gene_annot

		flag_gene_annot_query_2=flag_query2
		if flag_gene_annot_query_2>0:
			# gene_name annotation
			filename_prefix_1 = 'hg38'
			filename_prefix_2 = 'Homo_sapiens.GRCh38.108'
			input_filename_1 = '%s/test_gene_annot_expr.%s.1.txt'%(input_file_path,filename_prefix_1)
			input_filename_1_2 = '%s/test_gene_annot_expr_2.%s.1.txt'%(input_file_path,filename_prefix_1)
			input_filename_2 = '%s/test_gene_annot_expr.%s.1.txt'%(input_file_path,filename_prefix_2)
			input_filename_2_2 = '%s/test_gene_annot_expr_2.%s.1.txt'%(input_file_path,filename_prefix_2)
			input_filename_list = [input_filename_1,input_filename_1_2,input_filename_2,input_filename_2_2]
			list_query1 = []
			query_num1 = len(input_filename_list)
			for i1 in range(query_num1):
				input_filename = input_filename_list[i1]
				df_query_1 = pd.read_csv(input_filename,index_col=False,sep='\t')
				df_query_1.index = np.asarray(df_query_1['gene_name'])
				list_query1.append(df_query_1)

			df_gene_annot_expr_pre1, df_gene_annot_expr_2_pre1, df_gene_annot_expr_pre2, df_gene_annot_expr_2_pre2 = list_query1
			gene_name_query1 = df_gene_annot_expr_2_pre2.index.intersection(df_gene_annot_expr_pre1.index,sort=False)
			gene_name_query1_2 = df_gene_annot_expr_2_pre2.index.difference(df_gene_annot_expr_pre1.index,sort=False)
			df_gene_annot_expr_2_pre2_1 = df_gene_annot_expr_pre1.loc[gene_name_query1]
			df_gene_annot_expr_2_pre2_1['gene_version'] = 'hg38'
			df_gene_annot_expr_2_pre2_2 = df_gene_annot_expr_2_pre2.loc[gene_name_query1_2]
			df_annot1 = pd.concat([df_gene_annot_expr_pre2,df_gene_annot_expr_2_pre2_1],axis=0,join='outer',ignore_index=False)
			print('gene_name_query1, df_gene_annot_expr_2_pre2, df_annot1 ',len(gene_name_query1),df_gene_annot_expr_2_pre2.shape,df_annot1.shape)
			output_file_path = input_file_path
			output_filename = '%s/test_gene_annot_expr.%s.combine.1.txt'%(output_file_path,filename_prefix_2)
			df_annot1 = df_annot1.sort_values(by=['chrom','start','stop','gene_name'],ascending=True)
			df_annot1.to_csv(output_filename,index=False,sep='\t')
			output_filename = '%s/test_gene_annot_expr_2.%s.2.txt'%(output_file_path,filename_prefix_2)
			df_gene_annot_expr_2_pre2_2.to_csv(output_filename,index=False,sep='\t')
			print('df_gene_annot_expr_2_pre2_2 highly_variable ',np.sum(df_gene_annot_expr_2_pre2_2['highly_variable']))

			gene_name_query2 = df_gene_annot_expr_2_pre1.index.intersection(df_gene_annot_expr_pre2.index,sort=False)
			gene_name_query2_2 = df_gene_annot_expr_2_pre1.index.difference(df_gene_annot_expr_pre2.index,sort=False)
			df_gene_annot_expr_2_pre1_1 = df_gene_annot_expr_pre2.loc[gene_name_query2]
			df_gene_annot_expr_2_pre1_1['gene_version'] = 'GRCh38.108'
			df_gene_annot_expr_2_pre1_2 = df_gene_annot_expr_2_pre1.loc[gene_name_query2_2]
			df_annot2 = pd.concat([df_gene_annot_expr_pre1,df_gene_annot_expr_2_pre1_1],axis=0,join='outer',ignore_index=False)
			print('gene_name_query2, df_gene_annot_expr_2_pre1, df_annot2 ',len(gene_name_query2),df_gene_annot_expr_2_pre1.shape,df_annot2.shape)
			output_filename = '%s/test_gene_annot_expr.%s.combine.1.txt'%(output_file_path,filename_prefix_1)
			df_annot2 = df_annot2.sort_values(by=['chrom','start','stop','gene_name'],ascending=True)
			df_annot2.to_csv(output_filename,index=False,sep='\t')
			output_filename = '%s/test_gene_annot_expr_2.%s.2.txt'%(output_file_path,filename_prefix_1)
			df_gene_annot_expr_2_pre1_2.to_csv(output_filename,index=False,sep='\t')
			print('df_gene_annot_expr_2_pre1_2 highly_variable ',np.sum(df_gene_annot_expr_2_pre1_2['highly_variable']))

			df_gene_annot2 = df_annot2
			# return

		flag_gene_annot_query_3=flag_query3
		if flag_gene_annot_query_3>0:
			# gene_name annotation
			# filename_prefix_1 = 'hg38'
			filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
			input_filename_1 = '%s/test_gene_annot_expr.%s.combine.1.txt'%(input_file_path,filename_prefix_1)
			input_filename_1_2 = '%s/test_gene_annot_expr_2.%s.2.copy1.txt'%(input_file_path,filename_prefix_1)

			df1 = pd.read_csv(input_filename_1,index_col=False,sep='\t')
			df2 = pd.read_csv(input_filename_1_2,index_col=0,sep='\t')
			field_query_1 = df1.columns
			field_query_2 = df2.columns
			field_query = field_query_1.intersection(field_query_2,sort=False)
			df_query1 = df2.loc[:,field_query_2]
			df_query1['gene_id_1'] = df_query1.index.copy()
			df_query1['start'] = np.int32(df_query1['start'])
			df_query1['stop'] = np.int32(df_query1['stop'])
			id_1 = (pd.isna(df_query1['gene_id'])==True)
			id_2 = (~id_1)
			df_query2 = df_query1.loc[id_2,:]
			id1 = (df_query2['strand']=='-')
			id2 = (~id1)
			df_query2.loc[id1,'start_site'] = df_query2.loc[id1,'stop']
			df_query2.loc[id2,'start_site'] = df_query2.loc[id2,'start']
			df_query2['length'] = df_query2['stop']-df_query2['start']
			query_num_1 = df_query2.shape[0]
			query_id1 = df_query2.index

			df_query_1 = pd.concat([df1,df_query2],axis=0,join='outer',ignore_index=False)
			id_query1 = (df_query_1.duplicated(subset=['gene_name']))
			query_num1 = np.sum(id_query1)
			print('gene_name duplicated: %d'%(query_num1))
			df_query_1_duplicated = df_query_1.loc[id_query1,:]
			print(df_query_1_duplicated)

			output_file_path = input_file_path
			output_filename = '%s/test_gene_annot_expr.%s.combine.2.txt'%(output_file_path,filename_prefix_1)
			df_query_1 = df_query_1.sort_values(by=['chrom','start','stop'],ascending=True)
			df_query_1.to_csv(output_filename,index=False,sep='\t')
			df_gene_annot3 = df_query_1

		return df_gene_annot1, df_gene_annot2, df_gene_annot3

	## ====================================================
	# query gene annotations
	def test_gene_annotation_query1(self,select_config={}):

		"""
		query gene annotations
		:param select_config:
		:return: (dataframe) gene annotations of genes with expressions in the RNA-seq data
		"""

		flag_gene_annot_query=1
		df_gene_annot_expr = []
		input_filename_gene_annot = ''
		data_path_1 = select_config['data_path_1']
		data_path_2 = select_config['data_path_2']
		
		if 'filename_gene_annot' in select_config:
			input_filename_gene_annot = select_config['filename_gene_annot']
		
		if flag_gene_annot_query>0:
			data_file_type = select_config['data_file_type']
			flag_query=1
			if (self.species_id=='mm10'):
				if input_filename_gene_annot=='':
					filename_prefix_1 = 'Mus_musculus.GRCm38.102'
					input_file_path = data_path_2
					input_filename_gene_annot = '%s/gene_annotation/test_Mus_musculus.GRCm38.102.annotations_mm38.gene_annot_pre1.txt'%(input_file_path)
				
				print('input_filename_gene_annot: ',input_filename_gene_annot)
				df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2 = self.test_gene_annotation_load_1(input_filename=input_filename_gene_annot,
																												type_id_1=0,
																												select_config=select_config)
				
			elif (self.species_id=='hg38'):
				input_file_path = data_path_1
				if input_filename_gene_annot=='':
					filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
					input_filename_gene_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path)
				
				print('input_filename_gene_annot: ',input_filename_gene_annot)
				df_gene_annot_ori, df_gene_annot_expr = self.test_gene_annotation_load_2(input_filename=input_filename_gene_annot,
																							input_file_path=input_file_path,
																							select_config=select_config)
				
			else:
				print('please provide gene annotations')
				flag_query=0
				pass

			if (flag_query>0):
				print('input_filename_gene_annot: %s'%(input_filename_gene_annot))
				print('df_gene_annot_expr ',df_gene_annot_expr.shape)
				self.df_gene_annot_ori = df_gene_annot_ori	# gene annotation
				self.df_gene_annot_expr = df_gene_annot_expr   # gene query with expression
				
			return df_gene_annot_expr

	## ====================================================
	# query gene annotations for genes with expressions in the data
	def test_gene_annotation_load_1(self,input_filename='',input_filename_2='',type_id_1=0,flag_query_1=0,verbose=0,select_config={}):
		
		"""
		query gene annotations for genes with expressions in the data
		:param input_filename: (str) path of the gene annotation file of the genome-wide genes
		:param input_filename_2: (str) path of the annotation file of the mapping beteen gene names and Ensembl gene identifiers
		:param type_id_1: indicator of whether to query genes by gene name (type_id_1=0) or by Ensembl gene id (type_id_1=1)
		:param flag_query_1: indicator of whether to query maximal expressions of the genes without matched gene names in the used version of gene annotations
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		return: 1. (dataframe) gene annotations of the genome-wide genes;
				2. (dataframe) gene annotations of the genes with expressions in the RNA-seq data;
				3. (dataframe) anontations of the genes without matched gene names in the used version of gene annotations
		"""

		input_file_path = self.save_path_1
		print('load gene annotations')
		
		df_gene_annot_ori = pd.read_csv(input_filename,index_col=False,sep='\t')
		df_gene_annot_ori = df_gene_annot_ori.sort_values(by=['length'],ascending=False)
		rna_meta_ad = self.rna_meta_ad
		meta_scaled_exprs = self.meta_scaled_exprs
		
		if type_id_1==0:
			# gene query by gene name; keep gene name unduplicated
			df_gene_annot = df_gene_annot_ori.drop_duplicates(subset=['gene_name'])
			df_gene_annot.index = np.asarray(df_gene_annot['gene_name'])
			df_gene_annot = df_gene_annot.sort_values(by=['chrom','start','stop','gene_name'],ascending=True)
			print('gene annotation load 1 ',df_gene_annot_ori.shape, df_gene_annot.shape)

			gene_name_expr = rna_meta_ad.var_names
			gene_query = gene_name_expr
			gene_query_1 = pd.Index(gene_query).intersection(df_gene_annot.index,sort=False) # genes with expr in the dataset
			gene_query_2 = pd.Index(gene_query).difference(df_gene_annot.index,sort=False)
			df_gene_annot_expr = df_gene_annot.loc[gene_query_1,:]
			field_query = ['highly_variable', 'means', 'dispersions', 'dispersions_norm']

			df_var = rna_meta_ad.var
			column_vec_1 = df_var.columns
			column_vec_2 = pd.Index(field_query).intersection(column_vec_1,sort=False)

			load_mode = 0
			if list(column_vec_2)==list(field_query):
				load_mode = 1
			
			if load_mode>0:
				df_gene_annot_expr.loc[:,field_query] = df_var.loc[gene_query_1,field_query]
				gene_num1 = np.sum(df_gene_annot_expr['highly_variable']==True)
			
			meta_exprs_2 = self.meta_exprs_2
			df_gene_annot_expr_2 = pd.DataFrame(index=gene_query_2,columns=['gene_name','meta_scaled_exprs_max','meta_exprs_max'],dtype=np.float32)
			df_gene_annot_expr_2['gene_name'] = np.asarray(df_gene_annot_expr_2.index)
			
			if len(meta_scaled_exprs)>0:
				df_gene_annot_expr_2['meta_scaled_exprs_max'] = meta_scaled_exprs.loc[:,gene_query_2].max(axis=0)
			
			if len(meta_exprs_2)>0:
				df_gene_annot_expr_2['meta_exprs_max'] = meta_exprs_2.loc[:,gene_query_2].max(axis=0)
			
			if load_mode>0:
				df_gene_annot_expr_2.loc[gene_query_2,field_query] = df_var.loc[gene_query_2,field_query]
				gene_num2 = np.sum(df_gene_annot_expr_2['highly_variable']==True)
				gene_name_expr_1, gene_name_expr_2 = gene_query_1, gene_query_2

		else:
			# gene query by gene id; gene id are supposed to be unduplicated
			df_gene_annot = df_gene_annot_ori
			df_gene_annot.index = np.asarray(df_gene_annot['gene_id'])
			df_annot1 = pd.read_csv(input_filename_2,index_col=0,sep='\t')
			# df_annot1['gene_name'] = df_annot1.index.copy()
			# df_annot1.index = np.asarray(df_annot1['gene_id'])
			
			df_var = rna_meta_ad.var
			gene_name_expr = np.asarray(df_var.index)
			df_var['gene_name'] = df_var.index.copy()
			df_var['gene_ids'] = df_annot1.loc[gene_name_expr,'gene_ids']
			df_var.index = np.asarray(df_var['gene_ids'])
			query_num2 = np.sum(df_var.duplicated(subset=['gene_name']))
			print('the number of duplicated gene names: %d'%(query_num2))

			gene_query = np.asarray(df_var['gene_ids'])
			gene_query_1 = pd.Index(gene_query).intersection(df_gene_annot.index,sort=False) # genes with expr in the dataset
			gene_query_2 = pd.Index(gene_query).difference(df_gene_annot.index,sort=False)
			df_gene_annot_expr = df_gene_annot.loc[gene_query_1,:]
			df_gene_annot_expr['gene_name_ori'] = df_gene_annot_expr['gene_name'].copy()
			df_gene_annot_expr.loc[gene_query_1,'gene_name'] = np.asarray(df_var.loc[gene_query_1,'gene_name'])
			
			field_query = ['highly_variable', 'means', 'dispersions', 'dispersions_norm']
			column_vec_1 = df_var.columns
			column_vec_2 = pd.Index(field_query).intersection(column_vec_1,sort=False)
			
			load_mode = 0
			if list(column_vec_2)==list(field_query):
				load_mode = 1
			
			if load_mode>0:			
				df_gene_annot_expr.loc[gene_query_1,field_query] = df_var.loc[gene_query_1,field_query]
				gene_num1 = np.sum(df_gene_annot_expr['highly_variable']==True)

			meta_exprs_2 = self.meta_exprs_2
			df_gene_annot_expr_2 = pd.DataFrame(index=gene_query_2,columns=['gene_name','meta_scaled_exprs_max','meta_exprs_max'],dtype=np.float32)
			
			# df_gene_annot_expr_2['gene_name'] = np.asarray(df_gene_annot_expr_2.index)
			df_gene_annot_expr_2['gene_name'] = np.asarray(df_var.loc[gene_query_2,'gene_name'])

			if load_mode>0:
				df_gene_annot_expr_2.loc[gene_query_2,field_query] = df_var.loc[gene_query_2,field_query]
				gene_num2 = np.sum(df_gene_annot_expr_2['highly_variable']==True)

			if flag_query_1>0:
				if len(meta_scaled_exprs)>0:
					meta_scaled_exprs.columns = df_var.index.copy()
					df_gene_annot_expr_2['meta_scaled_exprs_max'] = meta_scaled_exprs.loc[:,gene_query_2].max(axis=0)
				
				if len(meta_exprs_2)>0:
					meta_exprs_2.columns = df_var.index.copy()
					df_gene_annot_expr_2['meta_exprs_max'] = meta_exprs_2.loc[:,gene_query_2].max(axis=0)
				
			gene_name_expr_1 = df_var.loc[gene_query_1,'gene_name']
			gene_name_expr_2 = df_var.loc[gene_query_2,'gene_name']

		# query how many genes are highly variable but not matching the gene name in the used gene annotations
		gene_query_2 = df_gene_annot_expr_2.index
		
		if load_mode>0:
			gene_query_2_highly_variable = gene_query_2[df_gene_annot_expr_2['highly_variable']==True]
		else:
			# print('highly variable genes not included in the gene annotation file')
			gene_query_2_highly_variable = []
		
		# print('gene_name_expr, gene_name_expr_1, gene_name_expr_2 ', len(gene_name_expr), len(gene_name_expr_1), len(gene_name_expr_2))
		# print('gene_name_expr_1, highly_variable ',len(gene_name_expr_1),gene_name_expr_1[0:5],gene_num1)
		# print('gene_name_expr_2, highly_variable ',len(gene_name_expr_2),gene_name_expr_2[0:5],gene_num2,gene_query_2_highly_variable)
		print('gene annotations, dataframe of size ',df_gene_annot_expr.shape)

		return df_gene_annot, df_gene_annot_expr, df_gene_annot_expr_2

	## ====================================================
	# query gene annotations for genes with expressions in the data
	def test_gene_annotation_load_2(self,input_filename='',input_filename_1='',input_file_path='',save_mode=1,verbose=0,select_config={}):

		"""
		query gene annotations for genes with expressions in the data
		:param input_filename: (str) path of the gene annotation file of the genome-wide genes
		:param input_filename_1: (str) path of the gene annotation file of the genes with expressions in the RNA-seq data
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) gene annotations of the genome-wide genes;
				 2. (dataframe) gene annotations of the genes with expressions in the RNA-seq data;
		"""

		input_file_path1 = self.save_path_1
		# load gene annotations
		print('load gene annotations')
		try:
			df_gene_annot = []
			if input_filename_1=='':
				if 'filename_gene_annot_ori' in select_config:
					input_filename_1 = select_config['filename_gene_annot_ori']
				else:
					print('please provide the gene annotation filename')
			
			if input_filename_1!='':
				if os.path.exists(input_filename_1)==True:
					df_gene_annot = pd.read_csv(input_filename_1,index_col=False,sep='\t')
					df_gene_annot.index = np.asarray(df_gene_annot['gene_id'])
					self.df_gene_annot_ori = df_gene_annot
				else:
					print('the file does not exist: %s'%(input_filename_1))

			if os.path.exists(input_filename)==True:
				df_gene_annot_expr_1 = pd.read_csv(input_filename,index_col=False,sep='\t')
				df_gene_annot_expr = df_gene_annot_expr_1.dropna(subset=['chrom'])
				df_gene_annot_expr.index = np.asarray(df_gene_annot_expr['gene_name'])
				self.df_gene_annot_expr = df_gene_annot_expr
				print('gene annotation, dataframe of size ', df_gene_annot_expr.shape)
			else:
				print('the file does not exist: %s'%(input_filename))
			
		except Exception as error:
			print('error! ', error)
			return

		# df_gene_annot_expr_2 = []
		return df_gene_annot, df_gene_annot_expr

	## ====================================================
	# load gene annotations
	def test_query_gene_annot_1(self,input_filename='',verbose=0,select_config={}):

		"""
		load gene annotations
		:param input_filename: the filename of gene annotations
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary storing the configuration parameters
		:return: dataframe of gene annotations containing gene name, gene position and TSS information
		"""

		if input_filename=='':
			input_filename_annot = select_config['filename_gene_annot']
		else:
			input_filename_annot = input_filename

		df_gene_annot_ori = pd.read_csv(input_filename_annot,index_col=False,sep='\t')
		df_gene_annot_ori = df_gene_annot_ori.sort_values(by=['length'],ascending=False)
		df_gene_annot_ori = df_gene_annot_ori.drop_duplicates(subset=['gene_name'])
		df_gene_annot_ori = df_gene_annot_ori.drop_duplicates(subset=['gene_id'])
		df_gene_annot_ori.index = np.asarray(df_gene_annot_ori['gene_name'])
		if verbose>0:
			print('gene annotation, dataframe of size ',df_gene_annot_ori.shape)
			print('columns: ',np.asarray(df_gene_annot_ori.columns))
			print('data preview: ')
			print(df_gene_annot_ori[0:2])

		return df_gene_annot_ori

	## ====================================================
	# compute peak accessibility-TF expression correlation and p-value
	def test_peak_tf_correlation_1(self,motif_data,peak_query_vec=[],motif_query_vec=[],peak_read=[],rna_exprs=[],
										correlation_type='spearmanr',pval_correction=1,alpha=0.05,method_type_correction='fdr_bh',verbose=1,select_config={}):

		"""
		compute peak accessibility-TF expression correlation and p-value
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param peak_query_vec: (array) the ATAC-seq peak loci for which to estimate peak-TF links
		:param motif_query_vec: (array) TF names
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param correlation_type: (str) the type of peak accessibility-TF expression correlation: 'spearmanr': Spearman's rank correlation; 'pearsonr': Pearson correlation;
		:param pval_correction: indicator of whether to compute the adjusted p-value of the correlation
		:param alpha: (float) family-wise error rate used in p-value correction for multiple tests
		:param method_type_correction: (str) the method used for p-value correction
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) the correlation matrix between the peak accessibilities and TF expressions (row:ATAC-seq peak locus, column:TF);
				 2. (dataframe) the raw p-pvalue matrix;
				 3. (dataframe) the adjusted p-value matrix;
				 4. (dataframe) TF annotations including the number of peak loci with the TF motif detected, the maximal and minimal correlation between peak accessibility and the TF expression for each TF;
		"""

		if len(motif_query_vec)==0:
			motif_query_name_ori = motif_data.columns
			motif_query_name_expr = motif_query_name_ori.intersection(rna_exprs.columns,sort=False)
			print('motif_query_name_ori, motif_query_name_expr ',len(motif_query_name_ori),len(motif_query_name_expr))
			motif_query_vec = motif_query_name_expr
		else:
			motif_query_vec_1 = motif_query_vec
			motif_query_vec = pd.Index(motif_query_vec).intersection(rna_exprs.columns,sort=False)
		
		motif_query_num = len(motif_query_vec)
		print('TF number: %d'%(motif_query_num))
		peak_loc_ori_1 = motif_data.index
		if len(peak_query_vec)>0:
			peak_query_1 = pd.Index(peak_query_vec).intersection(peak_loc_ori_1,sort=False)
			motif_data_query = motif_data.loc[peak_query_1,:]
		else:
			motif_data_query = motif_data

		peak_loc_ori = motif_data_query.index
		feature_query_vec_1, feature_query_vec_2 = peak_loc_ori, motif_query_vec
		df_corr_ = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
		df_pval_ = pd.DataFrame(index=feature_query_vec_1,columns=feature_query_vec_2,dtype=np.float32)
		flag_pval_correction = pval_correction
		if flag_pval_correction>0:
			df_pval_corrected = df_pval_.copy()
		else:
			df_pval_corrected = []
		df_motif_basic = pd.DataFrame(index=feature_query_vec_2,columns=['peak_num','corr_max','corr_min'])

		for i1 in range(motif_query_num):
			motif_id = motif_query_vec[i1]
			peak_loc_query = peak_loc_ori[motif_data_query.loc[:,motif_id]>0]

			df_feature_query1 = peak_read.loc[:,peak_loc_query]
			df_feature_query2 = rna_exprs.loc[:,[motif_id]]
			df_corr_1, df_pval_1 = utility_1.test_correlation_pvalues_pair(df1=df_feature_query1,
																			df2=df_feature_query2,
																			correlation_type=correlation_type,
																			float_precision=6)
			
			df_corr_.loc[peak_loc_query,motif_id] = df_corr_1.loc[peak_loc_query,motif_id]
			df_pval_.loc[peak_loc_query,motif_id] = df_pval_1.loc[peak_loc_query,motif_id]

			corr_max, corr_min = df_corr_1.max().max(), df_corr_1.min().min()
			peak_num = len(peak_loc_query)
			df_motif_basic.loc[motif_id] = [peak_num,corr_max,corr_min]
			
			interval_1 = 100
			if verbose>0:
				if i1%interval_1==0:
					print('motif_id: %s, id_query: %d, peak_num: %s, maximum peak accessibility-TF expr. correlation: %s, minimum correlation: %s'%(motif_id,i1,peak_num,corr_max,corr_min))
			
			if flag_pval_correction>0:
				pvals = df_pval_1.loc[peak_loc_query,motif_id]
				pvals_correction_vec1, pval_thresh1 = utility_1.test_pvalue_correction(pvals,alpha=alpha,method_type_id=method_type_correction)
				id1, pvals_corrected1, alpha_Sidak_1, alpha_Bonferroni_1 = pvals_correction_vec1
				df_pval_corrected.loc[peak_loc_query,motif_id] = pvals_corrected1
				if (verbose>0) and (i1%100==0):
					print('pvalue correction: alpha: %s, method_type: %s, minimum pval_corrected: %s, maximum pval_corrected: %s '%(alpha,method_type_correction,np.min(pvals_corrected1),np.max(pvals_corrected1)))

		return df_corr_, df_pval_, df_pval_corrected, df_motif_basic

	## ====================================================
	# compute peak accessibility-TF expression correlation and p-value
	def test_peak_tf_correlation_query_1(self,motif_data=[],peak_query_vec=[],motif_query_vec=[],peak_read=[],rna_exprs=[],
											correlation_type='spearmanr',pval_correction=1,alpha=0.05,method_type_correction='fdr_bh',
											flag_load=0,field_load=[],input_file_path='',input_filename_list=[],
											save_mode=1,output_file_path='',filename_prefix='',verbose=0,select_config={}):

		"""
		compute peak accessibility-TF expression correlation and p-value
		:param motif_data: (dataframe) the motif scanning results (binary), indicating if a TF motif is detected in a ATAC-seq peak locus (row:ATAC-seq peak locus, column:TF)
		:param peak_query_vec: (array) the ATAC-seq peak loci for which to estimate peak-TF links
		:param motif_query_vec: (array) TF names
		:param peak_read: (dataframe) peak accessibility matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param correlation_type: (str) the type of peak accessibility-TF expression correlation: 'spearmanr': Spearman's rank correlation; 'pearsonr': Pearson correlation;
		:param pval_correction: indicator of whether to compute the adjusted p-value of the correlation
		:param alpha: (float) family-wise error rate used in p-value correction for multiple tests
		:param method_type_correction: (str) the method used for p-value correction
		:param flag_load: indicator of whether to load peak accessibility-TF expression correlations and p-values from the saved files
		:param field_load: (array or list) fields representing correlation, the raw p-value, and the adjusted p-value that are used in the corresponding filenames and used for retrieving data
		:param input_file_path: the directory to retrieve data from
		:param input_filename_list: (list) paths of files to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing the following dataframes: 
				 1,2,3. the correlation, raw p-value, and adjusted p-value matrices between the peak accessibilities and TF expressions (row:ATAC-seq peak locus, column:TF);					
 				 4. TF annotations including the number of peak loci with the TF motif detected, the maximal and minimal correlation between peak accessibility and the TF expression for each TF;
		"""

		if filename_prefix=='':
			filename_prefix = 'test_peak_tf_correlation'
		if flag_load>0:
			if len(field_load)==0:
				field_load = [correlation_type,'pval','pval_corrected']
			field_num = len(field_load)

			file_num = len(input_filename_list)
			list_query = []
			if file_num==0:
				input_filename_list = ['%s/%s.%s.1.txt'%(input_file_path,filename_prefix,filename_annot) for filename_annot in field_load]

			dict_query = dict()
			for i1 in range(field_num):
				filename_annot1 = field_load[i1]
				input_filename = input_filename_list[i1]
				if os.path.exists(input_filename)==True:
					df_query = pd.read_csv(input_filename,index_col=0,sep='\t')
					field_query1 = filename_annot1
					dict_query.update({field_query1:df_query})
					print('df_query ',df_query.shape,filename_annot1)
				else:
					print('the file does not exist: %s'%(input_filename))
					flag_load = 0
				
			if len(dict_query)==field_num:
				return dict_query

		if flag_load==0:
			print('compute peak accessibility-TF expression correlation')
			start = time.time()
			df_peak_tf_corr_, df_peak_tf_pval_, df_peak_tf_pval_corrected, df_motif_basic = self.test_peak_tf_correlation_1(motif_data=motif_data,
																															peak_query_vec=peak_query_vec,
																															motif_query_vec=motif_query_vec,
																															peak_read=peak_read,
																															rna_exprs=rna_exprs,
																															correlation_type=correlation_type,
																															pval_correction=pval_correction,
																															alpha=alpha,
																															method_type_correction=method_type_correction,
																															select_config=select_config)

			field_query = ['peak_tf_corr','peak_tf_pval','peak_tf_pval_corrected','motif_basic']
			filename_annot_vec = [correlation_type,'pval','pval_corrected','motif_basic']
			list_query1 = [df_peak_tf_corr_, df_peak_tf_pval_, df_peak_tf_pval_corrected, df_motif_basic]
			dict_query = dict(zip(field_query,list_query1))
			query_num1 = len(list_query1)
			stop = time.time()
			print('computing peak accessibility-TF expr correlation used: %.5fs'%(stop-start))

			flag_save_text = 1
			if 'flag_save_text_peak_tf' in select_config:
				flag_save_text = select_config['flag_save_text_peak_tf']
			
			if save_mode>0:
				if output_file_path=='':
					# output_file_path = select_config['data_path']
					output_file_path = select_config['data_path_save']
				if flag_save_text>0:
					for i1 in range(query_num1):
						df_query = list_query1[i1]
						if len(df_query)>0:
							filename_annot1 = filename_annot_vec[i1]
							output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix,filename_annot1)
							if i1 in [3]:
								df_query.to_csv(output_filename,sep='\t',float_format='%.6f')
							else:
								df_query.to_csv(output_filename,sep='\t',float_format='%.5E')
							print('df_query ',df_query.shape,filename_annot1)
				
		return dict_query

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()



