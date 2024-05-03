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

	## load motif data
	def test_load_motif_data_1(self,method_type_vec=[],input_file_path='',save_mode=1,save_file_path='',verbose=0,select_config={}):
		
		flag_query1=1
		method_type_num = len(method_type_vec)
		dict_motif_data = dict()
		data_file_type = select_config['data_file_type']
		
		for i1 in range(method_type_num):
			method_type = method_type_vec[i1]
			motif_data_pre1, motif_data_score_pre1 = [], []

			method_annot_vec = ['insilico','joint_score','Unify','CIS-BP','CIS_BP'] # the method type which share motif scanning results
			# flag_1 = False
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

					# save_file_path = ''
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
					print('loaded motif scanning data (binary), dataframe of ',motif_data.shape)
					print('loaded motif scores, dataframe of ',motif_data_score.shape)

				dict_query={'motif_data':motif_data,'motif_data_score':motif_data_score}
			
			dict_motif_data[method_type] = dict_query
			# print('dict_query: ',dict_query,method_type)

		return dict_motif_data, select_config

	## query the method type based on the motif data used
	def test_query_method_type_motif_1(self,method_type='',method_annot_vec=[],data=[],select_config={}):

		if len(method_annot_vec)==0:
			method_annot_vec = ['insilico','joint_score','Unify','CIS-BP','CIS_BP'] # the method type which share motif scanning results

		flag_1 = False
		for method_annot_1 in method_annot_vec:
			flag_1 = (flag_1|(method_type.find(method_annot_1)>-1))

		return flag_1

	## load motif data
	def test_load_motif_data_pre1(self,input_filename_list1=[],input_filename_list2=[],flag_query1=1,flag_query2=1,overwrite=True,input_file_path='',
										save_mode=1,save_file_path='',type_id_1=0,type_id_2=1,select_config={}):
	
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
				print('motif scanning data (binary), dataframe of ', motif_data.shape)
				print('data preview: ')
				print(motif_data[0:2])
				print('motif scores, dataframe of ', motif_data_score.shape)
				print('data preview: ')
				print(motif_data_score[0:2])
				# motif_data_query, motif_data_score_query = motif_data, motif_data_score
				flag_pre1 = 1

		# load from the original motif data
		if flag_pre1==0:
			print('load the motif data')
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

					## motif name query
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

				print('motif_data_score ',motif_data_score.shape)
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

		flag_query2=0
		if flag_query2>0:
			df1 = (motif_data_score<0)
			id2 = motif_data_score.columns[df1.sum(axis=0)>0]
			if len(id2)>0:
				motif_data_score_ori = motif_data_score.copy()
				count1 = np.sum(np.sum(df1))
				# print('there are negative motif scores ',id2,count1)
				print('there are negative motif scores ',count1)
				
		return motif_data, motif_data_score, df_annot, type_id_query

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
	# merge multiple columns that correspond to one TF to one column
	def test_load_motif_data_pre2(self,motif_data,df_annot,column_id='tf',select_config={}):

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
			# print('difference ',column_1,column_2,difference,iter_id1)
			assert difference<1E-07

		# print('data preview: ')
		# print(motif_data[0:5])
		# field_id = '%s.ori'%(key_query)
		# if not (field_id in dict_query):
		# 	dict_query.update({'%s.ori'%(key_query):motif_data_ori})
		return motif_data, motif_data_ori

	## motif_name conversion for motifs in the used curated CIS-BP motif collection
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

	## load motif data
	def test_load_motif_data_2(self,data=[],dict_motif_data={},save_mode=1,verbose=0,select_config={}):
		
		motif_data = self.motif_data
		if len(motif_data)>0:
			motif_data_score = self.motif_data_score
			motif_query_name_expr = self.motif_query_name_expr
		else:
			if len(dict_motif_data)==0:
				dict_motif_data_query_1 = self.dict_motif_data
				if len(dict_motif_data_query_1)>0:				
					dict_motif_data = dict_motif_data_query_1[method_type_query]

			if len(dict_motif_data)==0:
				print('load motif data')
				input_dir = select_config['input_dir']
				file_path_1 = input_dir
				
				method_type_feature_link = select_config['method_type_feature_link']
				method_type_vec_query = [method_type_feature_link]
				data_path_save_local = select_config['data_path_save_local']
				# data_path_save_motif = select_config['data_path_save_motif']
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

				# motif_data, motif_data_score, motif_query_name_expr = list1
				motif_data, motif_data_score = list1[0:2]

				column_1 = 'motif_query_name_expr'
				if column_1 in dict_motif_data:
					motif_query_name_expr = dict_motif_data[column_1]
				else:
					motif_query_vec_pre1 = motif_data.columns
					gene_query_name_ori = rna_exprs.columns
					motif_query_name_expr = motif_query_vec_pre1.intersection(gene_query_name_ori,sort=False)

				motif_data = motif_data.loc[:,motif_query_name_expr]
				motif_data_score = motif_data_score.loc[:,motif_query_name_expr]
				self.motif_data = motif_data
				self.motif_data_score = motif_data_score
				self.motif_query_name_expr = motif_query_name_expr

		return motif_data, motif_data_score, motif_query_name_expr

	# chromvar score query: chromvar score comparison with TF expression
	# query correlation and mutual information between chromvar score and TF expression
	def test_chromvar_score_query_1(self,input_filename,motif_query_name_expr,filename_prefix_save='',output_file_path='',output_filename='',df_query=[],type_id_query=0,select_config={}):

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
		print('highly variable TF expression, mean_value, median_value ',motif_query_num2,mean_value,median_value)
		print('the other TF expressoin, mean_value, median_value ',motif_query_num3,mean_value_2,median_value_2)

		return df_2

	## load motif data
	def test_load_motif_data_2(self,data=[],dict_motif_data={},save_mode=1,verbose=0,select_config={}):
		
		# motif_data, motif_data_score = [],[]
		motif_data = self.motif_data
		if len(motif_data)>0:
			motif_data_score = self.motif_data_score
			motif_query_name_expr = self.motif_query_name_expr
		else:
			if len(dict_motif_data)==0:
				dict_motif_data_query_1 = self.dict_motif_data
				if len(dict_motif_data_query_1)>0:				
					dict_motif_data = dict_motif_data_query_1[method_type_query]

			if len(dict_motif_data)==0:
				print('load motif data')
				input_dir = select_config['input_dir']
				file_path_1 = input_dir
				test_estimator1 = _Base2_2(file_path=file_path_1,select_config=select_config)
					
				method_type_feature_link = select_config['method_type_feature_link']
				method_type_vec_query = [method_type_feature_link]
				data_path_save_local = select_config['data_path_save_local']
				# data_path_save_motif = select_config['data_path_save_motif']
				file_path_motif = data_path_save_local
				select_config.update({'file_path_motif':file_path_motif})
				save_file_path = data_path_save_local
				dict_motif_data_query_1, select_config = test_estimator1.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																								save_mode=1,save_file_path=save_file_path,
																								select_config=select_config)

				self.dict_motif_data = dict_motif_data_query_1
				dict_motif_data = dict_motif_data_query_1[method_type_query]

			if len(dict_motif_data)>0:
				# field_query = ['motif_data','motif_data_score','motif_query_name_expr']
				field_query = ['motif_data','motif_data_score']
				list1 = [dict_motif_data[field1] for field1 in field_query]

				# motif_data, motif_data_score, motif_query_name_expr = list1
				motif_data, motif_data_score = list1[0:2]

				column_1 = 'motif_query_name_expr'
				if column_1 in dict_motif_data:
					motif_query_name_expr = dict_motif_data[column_1]
				else:
					motif_query_vec_pre1 = motif_data.columns
					gene_query_name_ori = rna_exprs.columns
					motif_query_name_expr = motif_query_vec_pre1.intersection(gene_query_name_ori,sort=False)

				motif_data = motif_data.loc[:,motif_query_name_expr]
				motif_data_score = motif_data_score.loc[:,motif_query_name_expr]
				self.motif_data = motif_data
				self.motif_data_score = motif_data_score
				self.motif_query_name_expr = motif_query_name_expr

		return motif_data, motif_data_score, motif_query_name_expr

	## query gene annotations
	def test_gene_annotation_query_pre1(self,flag_query1=0,flag_query2=0,flag_query3=0,select_config={}):

		## gene name query and matching between the gene annotation file and gene name in the gene expression file
		flag_gene_annot_query_1=flag_query1
		df_gene_annot1, df_gene_annot2, df_gene_annot3 = [], [], []
		if flag_gene_annot_query_1>0:	
			filename_prefix_1 = 'hg38'
			filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
			# gene_annotation_filename = '%s/hg38.1.txt'%(input_file_path1)
			# gene_annotation_filename = '%s/Homo_sapiens.GRCh38.108.1.txt'%(input_file_path1)
			gene_annotation_filename = '%s/%s.1.txt'%(input_file_path1,filename_prefix_1)
			if len(self.df_gene_annot_expr)==0:
				df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2 = self.test_motif_peak_estimate_gene_annot_load_1(input_filename=gene_annotation_filename,select_config=select_config)
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
			## gene_name annotation
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
			## gene_name annotation
			# filename_prefix_1 = 'hg38'
			# filename_prefix_2 = 'Homo_sapiens.GRCh38.108'
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
			# for i1 in range(query_num_1):
			# 	id_pre1 = query_id1[id1]
			# 	try:
			# 		length_1 = np.int32(df_query2.loc[id_pre1,'stop'])-np.int32(df_query2.loc[id_pre1,'start'])
			# 		df_query2.loc[id_pre1,'length'] = length_1
			# 		print('gene_id_1 ',id_pre1,i1,length_1)
			# 	except Exception as error:
			# 		print('error! ',error,id_pre1,i1)

			df_query_1 = pd.concat([df1,df_query2],axis=0,join='outer',ignore_index=False)
			print('df1, df2, df_query2 ',df1.shape,df2.shape,df_query2.shape)
			print('df_query_1 ',df_query_1.shape)
			id_query1 = (df_query_1.duplicated(subset=['gene_name']))
			query_num1 = np.sum(id_query1)
			print('df_query_1 gene_name duplicated ',query_num1)
			df_query_1_duplicated = df_query_1.loc[id_query1,:]
			print(df_query_1_duplicated)

			output_file_path = input_file_path
			output_filename = '%s/test_gene_annot_expr.%s.combine.2.txt'%(output_file_path,filename_prefix_1)
			df_query_1 = df_query_1.sort_values(by=['chrom','start','stop'],ascending=True)
			df_query_1.to_csv(output_filename,index=False,sep='\t')
			df_gene_annot3 = df_query_1

		return df_gene_annot1, df_gene_annot2, df_gene_annot3

	## query gene annotations
	def test_gene_annotation_query1(self,select_config={}):

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
				df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2 = self.test_motif_peak_estimate_gene_annot_load_1(input_filename=input_filename_gene_annot,
																																type_id_1=0,
																																select_config=select_config)
				
			elif (self.species_id=='hg38'):
				input_file_path = data_path_1
				if input_filename_gene_annot=='':
					filename_prefix_1 = 'Homo_sapiens.GRCh38.108'
					input_filename_gene_annot = '%s/test_gene_annot_expr.Homo_sapiens.GRCh38.108.combine.2.txt'%(input_file_path)
				
				print('input_filename_gene_annot: ',input_filename_gene_annot)
				df_gene_annot_ori, df_gene_annot_expr, df_gene_annot_expr_2 = self.test_motif_peak_estimate_gene_annot_load_2(input_filename=input_filename_gene_annot,
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

	## load genome annotation load
	# input: the genome annotation filename; if not given, use default genome annotation filename
	# output: genome annotation (dataframe); gene position, strand, tss, transcript num
	# dataframe1: genes original; dataframe 2: genes with expr in the dataset
	def test_motif_peak_estimate_gene_annot_load_1(self,input_filename='',input_filename_2='',type_id_1=0,flag_query_2=0,select_config={}):

		input_file_path = self.save_path_1
		print('load gene annotations')
		
		df_gene_annot_ori = pd.read_csv(input_filename,index_col=False,sep='\t')
		df_gene_annot_ori = df_gene_annot_ori.sort_values(by=['length'],ascending=False)
		rna_meta_ad = self.rna_meta_ad
		meta_scaled_exprs = self.meta_scaled_exprs
		
		if type_id_1==0:
			## gene query by gene name; keep gene name unduplicated
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
			# print('df_var: ',df_var.shape)
			# print(df_var.columns)

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
			## gene query by gene id; gene id are supposed to be unduplicated
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

			if flag_query_2>0:
				if len(meta_scaled_exprs)>0:
					meta_scaled_exprs.columns = df_var.index.copy()
					df_gene_annot_expr_2['meta_scaled_exprs_max'] = meta_scaled_exprs.loc[:,gene_query_2].max(axis=0)
				
				if len(meta_exprs_2)>0:
					meta_exprs_2.columns = df_var.index.copy()
					df_gene_annot_expr_2['meta_exprs_max'] = meta_exprs_2.loc[:,gene_query_2].max(axis=0)
				
			gene_name_expr_1 = df_var.loc[gene_query_1,'gene_name']
			gene_name_expr_2 = df_var.loc[gene_query_2,'gene_name']

		## query how many genes are highly variable but not match the gene name in the used gene annotation version
		gene_query_2 = df_gene_annot_expr_2.index
		
		if load_mode>0:
			gene_query_2_highly_variable = gene_query_2[df_gene_annot_expr_2['highly_variable']==True]
		else:
			# print('highly variable genes not included in the gene annotation file')
			gene_query_2_highly_variable = []
		
		# print('gene_name_expr, gene_name_expr_1, gene_name_expr_2 ', len(gene_name_expr), len(gene_name_expr_1), len(gene_name_expr_2))
		# print('gene_name_expr_1, highly_variable ',len(gene_name_expr_1),gene_name_expr_1[0:5],gene_num1)
		# print('gene_name_expr_2, highly_variable ',len(gene_name_expr_2),gene_name_expr_2[0:5],gene_num2,gene_query_2_highly_variable)
		print('gene annotation, dataframe of size ',df_gene_annot_expr.shape)

		return df_gene_annot, df_gene_annot_expr, df_gene_annot_expr_2

	## query gene annotation for genes with expression in the given data
	# input: the gene annotation filenames; if not given, use default gene annotation filenames
	# output: gene annotation (dataframe); gene features in the dataset
	def test_motif_peak_estimate_gene_annot_load_2(self,input_filename='',input_filename_1='',input_file_path='',save_mode=1,verbose=0,select_config={}):

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

			# print('df_gene_annot_expr_1, df_gene_annot_expr ',df_gene_annot_expr_1.shape,df_gene_annot_expr.shape)
			
		except Exception as error:
			print('error! ', error)
			return

		df_gene_annot_expr_2 = []
		return df_gene_annot, df_gene_annot_expr, df_gene_annot_expr_2

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()



