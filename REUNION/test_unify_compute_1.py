# #!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData

from copy import deepcopy
import pyranges as pr
import warnings
import sys
from tqdm.notebook import tqdm

import os
import os.path
from optparse import OptionParser

import sklearn
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array, check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
from sklearn.pipeline import make_pipeline

from scipy import stats
from scipy.stats import pearsonr, spearmanr, gaussian_kde, zscore, poisson, multinomial, norm, rankdata
from scipy.stats import wilcoxon, mannwhitneyu, kstest, ks_2samp, chisquare, fisher_exact, chi2_contingency
import scipy.sparse
from scipy.sparse import spmatrix, csr_matrix, csc_matrix, issparse
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

import time
from timeit import default_timer as timer

# import utility_1
from . import utility_1
from .utility_1 import pyranges_from_strings, test_file_merge_1, spearman_corr, pearson_corr
# import h5py
import pickle

from .test_reunion_compute_pre2 import _Base_pre2

sc.settings.verbosity = 3    # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

class _Base2_correlation2(_Base_pre2):
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

		self.gene_query_vec = [] # the target gene set
		# self.save_mode = 2  # 2:save intermediate files; 1:without saving intermediate files
		self.save_mode = 1  # 2:save intermediate files; 1:without saving intermediate files
		flag_pcorr_interval = select_config['flag_pcorr_interval']
		self.flag_pcorr_interval = flag_pcorr_interval

	## ====================================================
	# query peak-gene link attributes
	def test_gene_peak_query_attribute_1(self,df_gene_peak_query=[],df_gene_peak_query_ref=[],column_idvec=[],field_query=[],column_name=[],reset_index=True,verbose=0,select_config={}):

		"""
		query peak-gene link attributes
		:param df_gene_peak_query: (dataframe) annotations of the specific peak-gene links
		:param df_gene_peak_query_ref: (dataframe) reference annotations of peak-gene links
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param field_query: (array or list) columns to copy from the reference annotations to the annotations of the specific peak-gene links
		:param reindex: indicator of whether to reindex the peak-gene links
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) updated annotations of the specific peak-gene links
		"""

		if verbose>0:
			print('the specified peak-gene links, dataframe of size ',df_gene_peak_query.shape)
			print('reference annotations of peak-gene links, dataframe of size ',df_gene_peak_query_ref.shape)

		query_id1_ori = df_gene_peak_query.index.copy()
		if len(column_idvec)==0:
			column_idvec = ['peak_id','gene_id']
		df_gene_peak_query.index = utility_1.test_query_index(df_gene_peak_query,column_vec=column_idvec)
		df_gene_peak_query_ref.index = utility_1.test_query_index(df_gene_peak_query_ref,column_vec=column_idvec)
		query_id1 = df_gene_peak_query.index
		df_gene_peak_query.loc[:,field_query] = df_gene_peak_query_ref.loc[query_id1,field_query]
		if len(column_name)>0:
			df_gene_peak_query = df_gene_peak_query.rename(columns=dict(zip(field_query,column_name)))
		if reset_index==True:
			df_gene_peak_query.index = query_id1_ori # reset the index

		return df_gene_peak_query

	## ====================================================
	# load motif data; load ATAC-seq and RNA-seq data of the metacells
	def test_query_load_pre1(self,method_type_vec=[],flag_motif_data_load_1=1,flag_load_1=1,flag_format=False,flag_scale=1,input_file_path='',save_mode=1,verbose=0,select_config={}):

		"""
		load low-dimensional embeddings of observations
		:param method_type_vec: the methods used to predict peak-TF associations initially
		:param flag_motif_data_load: indicator of whether to query motif scanning data
		:param flag_load_1: indicator of whether to query peak accessibility and gene expression data
		:param flag_format: indicator of whether to use uppercase variable names in the RNA-seq data of the metacells
		:param flag_scale: indicator of whether to scale the feature matrix
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated parameters
		"""

		method_type_feature_link = select_config['method_type_feature_link']
		method_type_vec_query = method_type_vec

		# load motif data
		if flag_motif_data_load_1>0:
			print('load motif scanning data')
			if len(method_type_vec_query)==0:
				method_type_vec_query = [method_type_feature_link]

			data_path_save_local = select_config['data_path_save_local']
			# data_path_save_motif = select_config['data_path_save_motif']
			
			file_path_motif = data_path_save_local
			select_config.update({'file_path_motif':file_path_motif})
			save_file_path = data_path_save_local
			dict_motif_data, select_config = self.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																			save_mode=1,save_file_path=save_file_path,
																			select_config=select_config)

			self.dict_motif_data = dict_motif_data

		# flag_load_1 = 1
		# load the ATAC-seq data and RNA-seq data of the metacells
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			# print('load ATAC-seq and RNA-seq count matrices of the metacells')
			start = time.time()
			data_path_save_local = select_config['data_path_save_local']
			output_file_path = data_path_save_local
			# peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_motif_peak_estimate_control_load_pre1_ori_2(meta_exprs=[],peak_read=[],flag_format=flag_format,flag_scale=flag_scale,
			# 																										save_mode=1,output_file_path=output_file_path,select_config=select_config)

			peak_read, meta_scaled_exprs, meta_exprs_2 = self.test_load_data_pre2(flag_format=flag_format,
																					flag_scale=flag_scale,
																					save_mode=1,output_file_path=output_file_path,select_config=select_config)


			sample_id = peak_read.index
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			if len(meta_scaled_exprs)>0:
				meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
				rna_exprs = meta_scaled_exprs	# scaled RNA-seq data
			else:
				rna_exprs = meta_exprs_2	# unscaled RNA-seq data
			print('ATAC-seq count matrx: ',peak_read.shape)
			print('data preview: ')
			print(peak_read[0:2])

			print('RNA-seq count matrx: ',rna_exprs.shape)
			print('data preview: ')
			print(rna_exprs[0:2])

			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs

			stop = time.time()
			print('load peak accessiblity and gene expression data used %.2fs'%(stop-start))
			
		return select_config

	## ====================================================
	# search for peaks within specific distance of the TSS of each target gene to establish peak-gene links
	def test_gene_peak_query_link_pre1(self,gene_query_vec=[],peak_distance_thresh=2000,df_peak_query=[],
											input_filename='',peak_loc_query=[],
											atac_ad=[],rna_exprs=[],highly_variable=False,
											interval_peak_corr=50,interval_local_peak_corr=10,parallel=0,
											save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',
											verbose=0,select_config={}):

		"""
		search for peaks within specific distance of the TSS of each target gene to establish peak-gene links
		:param gene_query_vec: (array or list) the target genes
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param df_peak_query: (dataframe) peak loci attributes
		:param input_filename: path of the file which saved annotations of identified peak-gene links with peak-gene TSS distances
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param highly_variable: indicator of whether to only include highly variable genes as the target genes
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param interval_local_peak_corr: the number of genes in the sub-batch for which to compute peak-gene correlations in the batch mode
		:param parallel: indicator of whether to search for the candidate peaks in parallel
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of identified peak-gene links with peak-gene TSS distances
		"""

		if len(peak_loc_query)==0:
			atac_ad = self.atac_meta_ad
			peak_loc_query = atac_ad.var_names
		
		df_gene_annot_expr = self.df_gene_annot_expr
		df_gene_query_1 = df_gene_annot_expr
		print('df_gene_annot_expr: ',df_gene_annot_expr.shape)
		
		# search for peaks within the distance threshold of the target gene
		# flag_distance_query=0
		flag_peak_query=1
		if flag_peak_query>0:
			if os.path.exists(input_filename)==False:
				print('the file does not exist: %s'%(input_filename))
				print('search for peak loci within distance %d Kb of the gene TSS '%(peak_distance_thresh))
				start = time.time()
				type_id2 = 0
				save_mode = 1
				df_gene_peak_query = self.test_gene_peak_query_distance(gene_query_vec=gene_query_vec,
																		df_gene_query=df_gene_query_1,
																		peak_loc_query=peak_loc_query,
																		peak_distance_thresh=peak_distance_thresh,
																		type_id_1=type_id2,parallel=parallel,
																		save_mode=save_mode,
																		output_filename=output_filename,
																		verbose=verbose,
																		select_config=select_config)
				stop = time.time()
				print('search for peak loci within distance %d Kb of the gene TSS used %.5fs'%(peak_distance_thresh, stop-start))
			else:
				print('load peak-gene links from %s'%(input_filename))
				df_gene_peak_query = pd.read_csv(input_filename,index_col=False,sep='\t')

			df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
			self.df_gene_peak_distance = df_gene_peak_query

			return df_gene_peak_query

	## ====================================================
	# search for peaks within specific distance of each target gene TSS to establish peak-gene links
	def test_gene_peak_query_distance(self,gene_query_vec=[],df_gene_query=[],peak_loc_query=[],peak_distance_thresh=2000,type_id_1=0,parallel=0,save_mode=1,output_filename='',verbose=0,select_config={}):

		"""
		search for peaks within specific distance of the TSS of each target gene to establish peak-gene links
		:param gene_query_vec: (array or list) the target genes
		:param df_gene_query: (dataframe) gene annotations
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-gene links
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param type_id_1: indicator of whether to search for peak-gene links and compute peak-gene distances, or load previously identified peak-gene links from the saved file
		:param parallel: indicator of whether to search for peak-gene links by distance threshold in batch mode in parallel
		:param save_mode: indicator of whether to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of identified peak-gene links with peak-gene TSS distances
		"""

		file_path1 = self.save_path_1
		if type_id_1==0:
			df_gene_query.index = np.asarray(df_gene_query['gene_name'])
			if len(gene_query_vec)>0:
				gene_query_vec_ori = gene_query_vec.copy()
				gene_query_num_ori = len(gene_query_vec_ori)
				gene_query_vec = pd.Index(gene_query_vec).intersection(df_gene_query.index,sort=False)
			else:
				gene_query_vec = df_gene_query.index
				gene_query_num_ori = len(gene_query_vec)

			gene_query_num = len(gene_query_vec)
			if not ('tss1' in df_gene_query.columns):
				df_gene_query['tss1'] = df_gene_query['start_site']
			df_tss_query = df_gene_query['tss1']
			if verbose>0:
				print('gene_query_vec_ori: %d, gene_query_vec: %d'%(gene_query_num_ori, gene_query_num))

			if len(peak_loc_query)==0:
				atac_ad = self.atac_meta_ad
				peak_loc_query = atac_ad.var_names

			peaks_pr = utility_1.pyranges_from_strings(peak_loc_query)
			peak_loc_num = len(peak_loc_query)

			bin_size = 1000
			span = peak_distance_thresh*bin_size
			if verbose>0:
				print('peak_loc_query: %d'%(peak_loc_num))
				print('peak_distance_thresh: %d bp'%(span))

			start = time.time()
			list1 = []
			interval = 5000
			if parallel==0:
				for i1 in range(gene_query_num):
					gene_query = gene_query_vec[i1]
					start = df_tss_query[gene_query]-span
					stop = df_tss_query[gene_query]+span
					chrom = df_gene_query.loc[gene_query,'chrom']
					gene_pr = pr.from_dict({'Chromosome':[chrom],'Start':[start],'End':[stop]})
					gene_peaks = peaks_pr.overlap(gene_pr)  # search for peak loci within specific distance of the gene
					if i1%interval==0:
						print('gene_peaks ', len(gene_peaks), gene_query, chrom, start, stop, i1)

					if len(gene_peaks)>0:
						df1 = pd.DataFrame.from_dict({'chrom':gene_peaks.Chromosome.values,
											'start':gene_peaks.Start.values,'stop':gene_peaks.End.values})

						df1.index = [gene_query]*df1.shape[0]
						list1.append(df1)
					else:
						print('gene query without peaks in the region query: %s %d'%(gene_query,i1))

				df_gene_peak_query = pd.concat(list1,axis=0,join='outer',ignore_index=False,keys=None,levels=None,names=None,verify_integrity=False,copy=True)

			else:
				interval_2 = 500
				iter_num = int(np.ceil(gene_query_num/interval_2))
				for iter_id in range(iter_num):
					start_id1 = int(iter_id*interval_2)
					start_id2 = np.min([(iter_id+1)*interval_2,gene_query_num])
					iter_vec = np.arange(start_id1,start_id2)
					res_local = Parallel(n_jobs=-1)(delayed(self.test_gene_peak_query_distance_unit1)(gene_query=gene_query_vec[i1],peaks_pr=peaks_pr,
																										df_gene_annot=df_gene_query,df_annot_2=df_tss_query,
																										span=span,query_id=i1,interval=interval,
																										save_mode=1,verbose=verbose,select_config=select_config) for i1 in tqdm(iter_vec))

					for df_query in res_local:
						if len(df_query)>0:
							list1.append(df_query)
				
				df_gene_peak_query = pd.concat(list1,axis=0,join='outer',ignore_index=False,keys=None,levels=None,names=None,verify_integrity=False,copy=True)

			df_gene_peak_query['gene_id'] = np.asarray(df_gene_peak_query.index)
			df_gene_peak_query.loc[df_gene_peak_query['start']<0,'start']=0
			query_num1 = df_gene_peak_query.shape[0]
			peak_id = utility_1.test_query_index(df_gene_peak_query,column_vec=['chrom','start','stop'],symbol_vec=[':','-'])
			df_gene_peak_query['peak_id'] = np.asarray(peak_id)
			if (save_mode==1) and (output_filename!=''):
				df_gene_peak_query = df_gene_peak_query.loc[:,['gene_id','peak_id']]
				df_gene_peak_query.to_csv(output_filename,index=False,sep='\t')

			stop = time.time()
			# print('search for peaks within distance threshold of each target gene used %.2fs'%(stop-start))
		else:
			print('load existing peak-gene links')
			input_filename = output_filename
			df_gene_peak_query = pd.read_csv(input_filename,index_col=0,sep='\t')
			df_gene_peak_query['gene_id'] = np.asarray(df_gene_peak_query.index)
			gene_query_vec = df_gene_peak_query['gene_id'].unique()

		print('peak-gene links, dataframe of size ', df_gene_peak_query.shape)
		print('query peak distance to gene TSS ')
		df_gene_peak_query = self.test_gene_peak_query_distance_pre1(gene_query_vec=gene_query_vec,
																	df_gene_peak_query=df_gene_peak_query,
																	df_gene_query=df_gene_query,
																	select_config=select_config)

		if (save_mode==1) and (output_filename!=''):
			df_gene_peak_query = df_gene_peak_query.loc[:,['gene_id','peak_id','distance']]
			df_gene_peak_query.to_csv(output_filename,index=False,sep='\t')

		return df_gene_peak_query

	## ====================================================
	# search for peaks within specific distance of the TSS of the target gene
	def test_gene_peak_query_distance_unit1(self,gene_query,peaks_pr,df_gene_annot=[],df_annot_2=[],span=2000,query_id=-1,interval=5000,save_mode=1,verbose=0,select_config={}):
		
		"""
		search for peaks within specific distance of the TSS of the target gene
		:param gene_query: (str) the target gene name
		:param peaks_pr: (pyranges object) positions of the ATAC-seq peak loci from which to find the candidate regulatory peaks of the gene
		:param df_gene_annot: (datafram) gene annotations
		:param df_annot_2: (dataframe) anontations of gene TSS positions
		:param span: (int) the distance (upstream and downstream) to the gene TSS within which to search for the candidate regulatory peaks of the gene
		:param query_id: (int) the indice of the gene
		:param interval: (int) the interval of gene number by which to print gene position and peak-gene link number information
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of identified candidate peaks for the gene including the peak position information
		"""

		df_gene_query = df_gene_annot
		df_tss_query = df_annot_2
		
		start = df_tss_query[gene_query]-span
		stop = df_tss_query[gene_query]+span
		chrom = df_gene_query.loc[gene_query,'chrom']
		
		gene_pr = pr.from_dict({'Chromosome':[chrom],'Start':[start],'End':[stop]})
		gene_peaks = peaks_pr.overlap(gene_pr)  # search for peak loci within specific distance of the gene
		
		if (interval>0) and (query_id%interval==0):
			print('find %d peak-gene links for gene %s (%s:%d-%d), %d'%(len(gene_peaks), gene_query, chrom, start, stop), query_id)

		if len(gene_peaks)>0:
			df1 = pd.DataFrame.from_dict({'chrom':gene_peaks.Chromosome.values,
										'start':gene_peaks.Start.values,'stop':gene_peaks.End.values})
			df1.index = [gene_query]*df1.shape[0]
		else:
			print('gene query without peaks in the region query: %s %d'%(gene_query,query_id))
			df1 = []

		return df1

	## ====================================================
	# compute peak-gene TSS distance for peak-gene pairs
	def test_gene_peak_query_distance_pre1(self,gene_query_vec=[],df_gene_peak_query=[],df_gene_query=[],select_config={}):

		"""
		compute distance between the peak and the gene TSS for peak-gene pairs
		:param gene_query_vec: (array or list) the target genes
		:param df_peak_gene_query: (dataframe) annotations of peak-gene links
		:param df_gene_query: (dataframe) gene annotations including the gene position and gene TSS position
		:param select_config: dictionary containing parameters
		:return: (dataframe) updated annotations of peak-gene links including the peak-gene TSS distances
		"""

		file_path1 = self.save_path_1
		gene_query_id = np.asarray(df_gene_peak_query['gene_id'])
		field_query_1 = ['chrom','start','stop','strand']

		flag_query1=1
		if flag_query1>0:
			list1 = [np.asarray(df_gene_query.loc[gene_query_id,field_query1]) for field_query1 in field_query_1]
			chrom1, start1, stop1, strand1 = list1
			start_site = start1
			id1 = (strand1=='-')
			start_site[id1] = stop1[id1]
			start_site = np.asarray(start_site)

		field_query_2 = ['chrom','start','stop']
		if not ('start' in df_gene_peak_query.columns):
			peak_id = pd.Index(df_gene_peak_query['peak_id'])
			chrom2, start2, stop2 = utility_1.pyranges_from_strings_1(peak_id,type_id=0)
			df_gene_peak_query['chrom'] = chrom2
			df_gene_peak_query['start'], df_gene_peak_query['stop'] = start2, stop2
		else:
			list1 = [np.asarray(df_gene_peak_query[field_query1]) for field_query1 in field_query_2]
			chrom2, start2, stop2 = list1

		peak_distance = start2-start_site
		peak_distance_2 = stop2-start_site

		id1 = (peak_distance<0)
		id2 = (peak_distance_2<0)
		peak_distance[id1&(~id2)]=0
		peak_distance[id2] = peak_distance_2[id2]

		print('peak-gene association: ', df_gene_peak_query.shape, df_gene_peak_query.columns, df_gene_peak_query[0:5])
		print('peak_distance: ', peak_distance.shape)
		bin_size = 1000.0
		df_gene_peak_query['distance'] = np.asarray(peak_distance/bin_size)

		return df_gene_peak_query

	## ====================================================
	# query background peak loci for the given peak loci
	def test_gene_peak_query_bg_load(self,input_filename_peak='',input_filename_bg='',peak_bg_num=100,verbose=0,select_config={}):

		"""
		query background peak loci for the given peak loci
		:param input_filename_peak: path of the file containing annotations of the ATAC-seq peak loci for which to estimate peak-gene links, including peak indices and positions
		:param input_filename_bg: path of the file containing indices of background peak loci sampled for each ATAC-seq peak locus
		:param peak_bg_num: the number of background peaks to sample for each candidate peak, which match the candidate peak by GC content and average accessibility
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) indices of the specified number of background peak loci sampled for each ATAC-seq peak locus
		"""

		if input_filename_peak=='':
			input_filename_peak = select_config['input_filename_peak']
		if input_filename_bg=='':
			input_filename_bg = select_config['input_filename_bg']

		peak_query = pd.read_csv(input_filename_peak,header=None,index_col=False,sep='\t')
		# self.atac_meta_peak_loc = np.asarray(peak_counts.index)
		peak_query.columns = ['chrom','start','stop','name','GC','score']	# to update
		peak_query.index = ['%s:%d-%d'%(chrom_id,start1,stop1) for (chrom_id,start1,stop1) in zip(peak_query['chrom'],peak_query['start'],peak_query['stop'])]
		atac_ad = self.atac_meta_ad
		peak_loc_1 = atac_ad.var_names
		assert list(peak_query.index) == list(peak_loc_1)
		self.atac_meta_peak_loc = np.asarray(peak_query.index)
		print('peak loci in ATAC-seq metacell data: %d'%(len(self.atac_meta_peak_loc)))
		print('preview: ',self.atac_meta_peak_loc[0:5])

		peak_bg = pd.read_csv(input_filename_bg,index_col=0)
		peak_bg_num_ori = peak_bg.shape[1]
		peak_id = np.int64(peak_bg.index)
		# print('background peaks', peak_bg.shape, len(peak_id), peak_bg_num)
		peak_bg.index = self.atac_meta_peak_loc[peak_id-1]
		peak_bg = peak_bg.loc[:,peak_bg.columns[0:peak_bg_num]]
		peak_bg.columns = np.arange(peak_bg_num)
		peak_bg = peak_bg.astype(np.int64)
		self.peak_bg = peak_bg
		if verbose>0:
			print(input_filename_peak)
			print(input_filename_bg)
			print('atac matecell peaks', len(self.atac_meta_peak_loc), self.atac_meta_peak_loc[0:5])
			print('background peaks', peak_bg.shape, len(peak_id), peak_bg_num)

		return peak_bg

	## ====================================================
	# query accessibility-related peak attributes
	def test_peak_access_query_basic_1(self,peak_read=[],rna_exprs=[],df_annot=[],thresh_value=0.1,flag_ratio=1,flag_access=1,
											save_mode=1,output_file_path='',output_filename='',filename_annot='',verbose=0,select_config={}):

		"""
		query accessibility-related peak attributes, including the openning ratio in the metacells and the maximal peak accessibility in the metacells
		:param peak_read: (dataframe) peak accessibilty matrix of the metacells (row:metacell, column:ATAC-seq peak locus)
		:param rna_exprs: (dataframe) gene expressions of the metacells (row:metacell, column:gene)
		:param df_annot: (dataframe) ATAC-seq peak loci attributes
		:param thresh_value: the threshold on accessibility to estimate if a peak locus is open
		:param flag_ratio: indicator of whether to query the percentage of metacells in which a peak locus is identified as being open
		:param flag_access: indicator of whether to query the maximal chromatin accessibility of a peak locus across the metacells
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. updated dataframe of peak loci attributes
				 2. the quantiles of opening ratio of peak loci in the metacells
		"""

		# peak_loc_ori = motif_data.index
		peak_loc_ori = peak_read.columns
		sample_num = peak_read.shape[0]
		if len(df_annot)>0:
			df1 = df_annot
		else:
			df1 = pd.DataFrame(index=peak_loc_ori,columns=['ratio'],dtype=np.float32)

		field_query = []
		if flag_ratio>0:
			peak_read_num1 = (peak_read.loc[:,peak_loc_ori]>0).sum(axis=0)
			ratio_1 = peak_read_num1/sample_num
			thresh_1 = thresh_value
			peak_read_num2 = (peak_read.loc[:,peak_loc_ori]>thresh_1).sum(axis=0)
			ratio_2 = peak_read_num2/sample_num
			column_1 = 'ratio'
			column_2 = 'ratio_%s'%(thresh_1)
			df1[column_1] = np.asarray(ratio_1)
			# df1['ratio_0.1'] = np.asarray(ratio_2)
			df1[column_2] = np.asarray(ratio_2)
			field_query = field_query + [column_1,column_2]

		if flag_access>0:
			column_3 = 'max_accessibility_score'
			df1[column_3] = peak_read.max(axis=0)
			field_query += [column_3]

		if save_mode>0:
			if output_filename=='':
				if filename_annot=='':
					# filename_annot = select_config['filename_save_annot_pre1']
					filename_annot = select_config['filename_annot_save_default']
				output_filename = '%s/test_peak_query_basic_1.%s.1.txt'%(output_file_path,filename_annot)
			if verbose>0:
				print('field_query: ',field_query)
				print('peak attribute annotation, dataframe of size ',df1.shape)
			df_query1 = df1.loc[:,field_query]
			df_query1.to_csv(output_filename,sep='\t',float_format='%.6f')

		column_id_query = ['ratio']
		quantile_vec_1 = [0.05,0.1,0.25,0.5,0.75,0.9,0.95]
		t_value_1 = utility_1.test_stat_1(df1[column_id_query],quantile_vec=quantile_vec_1)
		ratio1 = t_value_1
		if verbose>0:
			print('quantiles of opening ratio of peak loci in the metacells: ',t_value_1)
		
		return df1, ratio1

	## ====================================================
	# query open peaks in the metacells
	def test_peak_access_query_basic_2(self,atac_meta_ad=[],peak_set=None, low_dim_embedding='X_svd', pval_cutoff=1e-2,read_len=147,n_neighbors=3,bin_size=5000,n_jobs=1,
											save_mode=1,output_file_path='',output_filename='',filename_save_annot='',select_config={}):

		"""
		query open peaks in the metacells
		:param adata: (AnnData object) ATAC-seq data
		:param n_neighbors: the number of neighbors used in estimating smoothed peak accessibility
		:param low_dim_embedding: name of layer in the ATAC-seq AnnData object representing the low-dimensional embeddings of peak accessibility
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (AnnData object) ATAC-seq data with the layer indicating the set of open peaks in each metacell
		"""

		if len(atac_meta_ad)==0:
			atac_meta_ad = self.atac_meta_ad

		# query the set of peaks that are open in each metacell (using the function in SEACells)
		open_peaks = self._determine_metacell_open_peaks(atac_meta_ad,peak_set=None, low_dim_embedding=low_dim_embedding,
															pval_cutoff=pval_cutoff,read_len=147,n_neighbors=n_neighbors,
															bin_size=bin_size,n_jobs=n_jobs)
		peak_loc_ori = atac_meta_ad.var_names
		sample_num = atac_meta_ad.shape[0]
		df_open_peaks = open_peaks
		peak_num1 = df_open_peaks.sum(axis=0)	# the number of open peaks in each metacell
		ratio_1 = peak_num1/sample_num
		df1 = pd.DataFrame(index=peak_loc_ori,columns=['ratio'],data=np.asarray(ratio_1))

		if save_mode>0:
			filename_annot = filename_save_annot
			if output_filename=='':
				if filename_annot=='':
					filename_annot = select_config['filename_annot_save_default']
				output_filename = '%s/test_peak_query_basic.%s.txt'%(output_file_path,filename_annot)

			df1.to_csv(output_filename,sep='\t',float_format='%.6f')

			if 'output_filename_open_peaks' in select_config:
				output_filename_1 = select_config['output_filename_open_peaks']
				open_peaks.to_csv(output_filename_1,sep='\t',float_format='%d')

			if 'output_filename_nbrs_atac' in select_config:
				output_filename_2 = select_config['output_filename_nbrs_atac']
				meta_nbrs = self.select_config['meta_nbrs_atac']
				meta_nbrs.to_csv(output_filename_2,sep='\t')

		return atac_meta_ad, open_peaks

	## ====================================================
	# query the set of peaks that are open in each metacell (from SEACells)
	def _determine_metacell_open_peaks(self,atac_meta_ad,peak_set=None,low_dim_embedding='X_svd',
											pval_cutoff=1e-2,read_len=147,n_neighbors=3,bin_size=5000,n_jobs=1):
		"""
		Determine the set of peaks that are open in each metacell
		:param atac_meta_ad: (Anndata) ATAC metacell Anndata created using `prepare_multiome_anndata`
		:param peak_set: (pd.Series) Subset of peaks to test. All peaks are tested by default
		:param low_dim_embedding: (str) `atac_meta_ad.obsm` field for nearest neighbor computation
		:param p_val_cutoff: (float) Nominal p-value cutoff for open peaks
		:param read_len: (int) Fragment length
		:param n_jobs: (int) number of jobs for parallel processing
		:return: 1. (dataframe) labels (binary) representing which peaks are open in each metacell (row:metacell,column:peak locus)
				 2. atac_meta_ad is modified inplace with `.obsm['OpenPeaks']` indicating the set of open peaks in each metacell
		"""
		from sklearn.neighbors import NearestNeighbors
		from scipy.stats import poisson, multinomial

		# Effective genome length for background computaiton
		# eff_genome_length = atac_meta_ad.shape[1] * 5000
		# bin_size = 500
		eff_genome_length = atac_meta_ad.shape[1] * bin_size

		# Set up container
		if peak_set is None:
			peak_set = atac_meta_ad.var_names
		open_peaks = pd.DataFrame(0, index=atac_meta_ad.obs_names, columns=peak_set)

		# metacell neighbors
		nbrs = NearestNeighbors(n_neighbors=n_neighbors)
		nbrs.fit(atac_meta_ad.obsm[low_dim_embedding])
		meta_nbrs = pd.DataFrame(atac_meta_ad.obs_names.values[nbrs.kneighbors(atac_meta_ad.obsm[low_dim_embedding])[1]],
								 index=atac_meta_ad.obs_names)

		self.select_config.update({'meta_nbrs_atac':meta_nbrs})

		for m in tqdm(open_peaks.index):
			# Boost using local neighbors
			frag_counts = np.ravel(
				atac_meta_ad[meta_nbrs.loc[m, :].values, :][:, peak_set].X.sum(axis=0))
			frag_distr = frag_counts / np.sum(frag_counts).astype(np.float64)

			# Multinomial distribution
			while not 0 < np.sum(frag_distr) < 1 - 1e-5:
				frag_distr = np.absolute(frag_distr - np.finfo(np.float32).epsneg)
			# Sample from multinomial distribution
			frag_counts = multinomial.rvs(np.percentile(
				atac_meta_ad.obs['n_counts'], 100), frag_distr)

			# Compute background poisson distribution
			total_frags = frag_counts.sum()
			glambda = (read_len * total_frags) / eff_genome_length

			# Significant peaks
			cutoff = pval_cutoff / np.sum(frag_counts > 0)
			open_peaks.loc[m, frag_counts >= poisson.ppf(1 - cutoff, glambda)] = 1

		# update ATAC metadata object
		atac_meta_ad.layers['OpenPeaks'] = open_peaks.values

		return open_peaks

	## ====================================================
	# query open peaks in the metacells
	def test_peak_access_query_basic_pre1(self,adata=[],n_neighbors=3,low_dim_embedding='X_svd',save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query open peaks in the metacells and accessibility-related peak attributes
		:param adata: (AnnData object) ATAC-seq data
		:param n_neighbors: the number of neighbors used in estimating smoothed peak accessibility
		:param low_dim_embedding: name of layer in the ATAC-seq AnnData object representing the low-dimensional embeddings of peak accessibility
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (AnnData object) ATAC-seq data with the layer indicating the set of open peaks in each metacell
		"""

		# query the ratio of open peak loci
		flag_query_peak_ratio=1
		if flag_query_peak_ratio>0:
			# atac_meta_ad = self.atac_meta_ad
			atac_meta_ad = adata
			print('ATAC-seq metacell AnnData\n',atac_meta_ad)

			if output_file_path=='':
				file_save_path2 = select_config['data_path_save']
				output_file_path = file_save_path2
			
			type_id_query=1
			filename_prefix = filename_prefix_save
			filename_annot = filename_save_annot

			filename_annot2 = '%s.%d'%(filename_annot,(type_id_query+1))
			output_filename = '%s/test_peak_query_basic.%s.txt'%(output_file_path,filename_annot2)
			output_filename_peak_query = '%s/test_query_peak_access.%s.txt'%(output_file_path,filename_annot2)
			output_filename_nbrs_atac = '%s/test_query_meta_nbrs.%s.txt'%(output_file_path,filename_annot2)
			select_config.update({'output_filename_open_peaks':output_filename_peak_query,
									'output_filename_nbrs_atac':output_filename_nbrs_atac})

			if not(low_dim_embedding in atac_meta_ad.obsm):
				if type_id_query==0:
					print('the embedding not estimated: %s'%(low_dim_embedding))
					input_filename_atac = self.select_config['input_filename_atac']
					atac_ad_ori = sc.read_h5ad(input_filename_atac)
					print('ATAC-seq AnnData: ',atac_ad_ori.shape)
					print(atac_ad_ori)
					
					input_filename = select_config['filename_rna_obs']
					df_obs_rna = pd.read_csv(input_filename,index_col=0,sep='\t')
					print('the sample annotations in RNA-seq data, dataframe of size ',df_obs_rna.shape)
					print('data preview: ',df_obs_rna[0:2])
					
					sample_id_rna_ori = df_obs_rna.index
					sample_id_atac_ori = atac_ad_ori.obs_names
					# common_cells = atac_ad.obs_names.intersection(rna_ad.obs_names,sort=False)
					common_cells = sample_id_atac_ori.intersection(sample_id_rna_ori,sort=False)
					atac_mod_ad = atac_ad_ori[common_cells,:]
					print('common_cells: %d'%(len(common_cells)))
					print('ATAC-seq AnnData of the common cells: ',atac_mod_ad.shape)
					atac_svd = atac_mod_ad.obsm['X_svd']
					print('ATAC-seq feature embeddings, dataframe of size ',atac_svd.shape)
					svd = pd.DataFrame(data=atac_mod_ad.obsm['X_svd'],index=atac_mod_ad.obs_names)
					
					SEACells_label = 'SEACell'
					column_query = 'Metacell'
					df_obs_atac_ori = atac_ad_ori.obs  # ATAC-seq data observation attributes
					df_var_atac_ori = atac_ad_ori.var  # ATAC-seq data variable attributes

					df_obs_rna_1 = df_obs_rna.loc[common_cells,:]
					df_obs_atac_ori.loc[common_cells,SEACells_label] = np.asarray(df_obs_rna.loc[common_cells,column_query])
					
					df_obs_atac_1 = df_obs_atac_ori.loc[common_cells,:]
					summ_svd = svd.groupby(df_obs_atac_1[SEACells_label]).mean()
					atac_meta_ad.obsm['X_svd'] = summ_svd.loc[atac_meta_ad.obs_names, :].values

					if save_mode>0:
						output_filename_1 = '%s/test_%s_atac.df_obs.txt'%(output_file_path,filename_prefix)
						output_filename_2 = '%s/test_%s_atac.df_var.txt'%(output_file_path,filename_prefix)
						
						df_obs_atac_ori.to_csv(output_filename_1,sep='\t')
						df_var_atac_ori.to_csv(output_filename_2,sep='\t')

						output_filename_1 = '%s/test_%s_atac.common.df_obs.txt'%(output_file_path,filename_prefix)
						output_filename_2 = '%s/test_%s_rna.common.df_obs.txt'%(output_file_path,filename_prefix)
						
						df_obs_atac_1.to_csv(output_filename_1,sep='\t')
						df_obs_rna_1.to_csv(output_filename_2,sep='\t')
				else:
					n_components = 100
					sc.tl.pca(atac_meta_ad,n_comps=n_components,zero_center=False,use_highly_variable=False)
					atac_meta_ad.obsm[low_dim_embedding] = atac_meta_ad.obsm['X_pca'].copy()
					atac_feature = atac_meta_ad.obsm[low_dim_embedding]
					
					print('ATAC-seq feature embedding matrix, dataframe of size ',atac_feature.shape)
					if save_mode>0:
						output_filename_1 = '%s/test_%s_meta_atac.normalize.h5ad'%(output_file_path,filename_prefix)
						atac_meta_ad.write(output_filename_1)

			pval_cutoff = 1e-2
			# n_neighbors = 3
			# bin_size = 500
			bin_size = 5000
			n_jobs = 1
			atac_meta_ad, open_peaks = self.test_peak_access_query_basic_2(atac_meta_ad=atac_meta_ad,
																			peak_set=None,
																			low_dim_embedding=low_dim_embedding,
																			pval_cutoff=pval_cutoff,
																			read_len=147,
																			n_neighbors=n_neighbors,
																			bin_size=bin_size,
																			n_jobs=n_jobs,
																			save_mode=1,
																			filename_annot='',
																			output_file_path=output_file_path,
																			output_filename=output_filename,
																			select_config=select_config)

			self.atac_meta_ad = atac_meta_ad

			return atac_meta_ad

	## ====================================================
	# compute peak accessibility-gene expression correlations and select candidate peak-gene links
	def test_gene_peak_query_correlation_pre1(self,gene_query_vec=[],peak_loc_query=[],df_gene_peak_query=[],df_gene_peak_compute_1=[],df_gene_peak_compute_2=[],atac_ad=[],rna_exprs=[],
												flag_computation_vec=[1,3],highly_variable=False,recompute=0,
												save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		compute peak accessibility-gene expression correlations and select candidate peak-gene links
		:param gene_query_vec: (array or list) the target genes
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-gene links
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param df_gene_peak_compute_1: (dataframe) annotation of peak-gene links
		:param df_gene_peak_compute_2: (dataframe) annotation of peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param flag_computation_vec: (array or list) computation mode indicating which type of peak accessibility-gene expression correlation to compute and whether to perform pre-selection of peak-gene links
		:param highly_variable: indicator of whether to only include highly variable gene as target genes
		:param recompute: indicator of wether to recompute peak accessibility-gene expression correlations if the correlations were computed and saved
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: 1. (dataframe) annotations of selected candidate peak-gene links
				 2. (dataframe) updated annotations of the original input peak-gene links		
		"""

		data_path = select_config['data_path_save']
		
		peak_bg_num_ori = 100
		input_filename_peak, input_filename_bg = select_config['input_filename_peak'], select_config['input_filename_bg']
		if os.path.exists(input_filename_bg)==False:
			print('the file does not exist: %s'%(input_filename_bg))
			# return

		peak_bg_num = 100
		# interval_peak_corr = 10
		interval_peak_corr = 100
		interval_local_peak_corr = -1

		list1 = [peak_bg_num,interval_peak_corr,interval_local_peak_corr]
		field_query = ['peak_bg_num','interval_peak_corr','interval_local_peak_corr']

		query_num1 = len(list1)
		for i1 in range(query_num1):
			field_id = field_query[i1]
			if (field_id in select_config):
				list1[i1] = select_config[field_id]
		peak_bg_num,interval_peak_corr,interval_local_peak_corr = list1

		save_file_path = select_config['data_path_save_local']
		input_file_path = save_file_path
		output_file_path = save_file_path

		flag_correlation_1 = select_config['flag_correlation_1']
		peak_distance_thresh = select_config['peak_distance_thresh']
		verbose_internal = self.verbose_internal

		df_gene_peak_query_thresh2 = []
		if flag_correlation_1>0:
			interval_peak_corr = select_config['interval_peak_corr']
			interval_local_peak_corr = select_config['interval_local_peak_corr']
			
			input_filename_pre1, input_filename_pre2 = select_config['input_filename_pre1'], select_config['input_filename_pre2']
			flag_compute = 0
			if (os.path.exists(input_filename_pre1)==False) or (recompute>0):
				print('the file to be prepared: %s'%(input_filename_pre1))
				flag_compute = 1
			else:
				if (os.path.exists(input_filename_pre1)==True):
					print('the file exists: %s'%(input_filename_pre1))

			print('flag_computation_vec: ',flag_computation_vec)
			compute_mode_1 = 1
			compute_mode_2 = 3
			# compute_mode_2 = 1
			flag_compute_fg = (compute_mode_1 in flag_computation_vec)
			flag_compute_bg = (compute_mode_2 in flag_computation_vec)
			flag_compute = (flag_compute|flag_compute_bg)

			coherent_mode = 0
			if (flag_compute_fg>0) and (flag_compute_bg>0):
				coherent_mode = 1

			flag_thresh1 = 1
			if 'flag_correlation_thresh1' in select_config:
				flag_thresh1 = select_config['flag_correlation_thresh1']

			print('flag_thresh1 ',flag_thresh1)
			# interval_peak_corr = -1
			# interval_local_peak_corr = -1

			column_idvec = ['peak_id','gene_id']
			dict_query1 = dict()
			iter_mode = select_config['iter_mode']
			print('iter_mode ',iter_mode)
			if flag_compute>0:
				df_gene_peak_query_thresh1 = df_gene_peak_compute_2 # pre-selected peak-gene links with peak accessibility-gene expression correlation above thresholds
				print('flag_computation_vec: ',flag_computation_vec)

				for flag_computation_1 in flag_computation_vec:
					select_config.update({'flag_computation_1':flag_computation_1})

					compute_mode = flag_computation_1
					if compute_mode==1:
						if len(df_gene_peak_query)==0:
							# compute peak accessibility-gene expression correlation for genome-wide peak-gene links
							df_gene_peak_query = self.df_gene_peak_distance  

					elif compute_mode==2:
						if len(df_gene_peak_query)==0:
							# use the given peak-gene links
							df_gene_peak_query = df_gene_peak_compute_1

					elif compute_mode==3:
						# use pre-selected peak-gene links with peak accessibility-gene expression correlation above thresholds
						if len(df_gene_peak_query_thresh1)==0:
							df_gene_peak_query = pd.read_csv(input_filename_pre2,index_col=0,sep='\t')
						else:
							df_gene_peak_query = df_gene_peak_query_thresh1

					if compute_mode in [1,3]:
						if compute_mode in [1]:
							interval_peak_corr = -1
							interval_local_peak_corr = -1
						else:
							interval_peak_corr = 100
							# interval_local_peak_corr = 20
							# interval_local_peak_corr = 50
							interval_local_peak_corr = -1

						select_config.update({'interval_peak_corr':interval_peak_corr,'interval_local_peak_corr':interval_local_peak_corr})

						# compute peak accessibility-gene expression correlation for peak-gene links
						df_gene_peak_1 = self.test_gene_peak_query_correlation_pre1_compute(gene_query_vec=gene_query_vec,
																							gene_query_vec_2=[],
																							peak_distance_thresh=peak_distance_thresh,
																							df_gene_peak_query=df_gene_peak_query,
																							df_gene_peak_compute=df_gene_peak_compute_1,
																							peak_loc_query=[],
																							atac_ad=atac_ad,
																							rna_exprs=rna_exprs,
																							highly_variable=highly_variable,
																							interval_peak_corr=interval_peak_corr,
																							interval_local_peak_corr=interval_local_peak_corr,
																							save_mode=1,
																							save_file_path=save_file_path,
																							output_filename='',
																							filename_prefix_save='',			
																							verbose=verbose,select_config=select_config)
						
						dict_query1.update({compute_mode:df_gene_peak_1})

						# print('df_gene_peak_1 ',df_gene_peak_1.shape)
						print('peak-gene links, dataframe of size ',df_gene_peak_1.shape)
						print('preview: ')
						print(df_gene_peak_1[0:5])
						
						iter_mode = select_config['iter_mode']
						print('iter_mode ',iter_mode)

						if (iter_mode>0) and (coherent_mode>0):
							input_filename_pre2_ori = select_config['input_filename_pre2']
							filename_save_thresh2_ori = select_config['filename_save_thresh2']
							
							select_config.update({'input_filename_pre2_ori':input_filename_pre2,
													'filename_save_thresh2_ori':filename_save_thresh2_ori})

							# update the filenames to save data for the batch mode
							save_file_path = select_config['data_path_save_local']
							filename_prefix_query = select_config['filename_prefix_local']
							filename_annot1 = select_config['filename_annot_default']

							# the peak-gene links selected using threshold 1 on the peak-gene correlation
							input_filename_pre2 = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix_query,filename_annot1)

							# the peak-gene links selected using threshold 2 on the peak-gene correlation
							filename_save_thresh2 = '%s/%s.combine.thresh2.%s.txt'%(save_file_path,filename_prefix_query,filename_annot1)

							select_config.update({'input_filename_pre2':input_filename_pre2,
													'filename_save_thresh2':filename_save_thresh2})

							print('updated input_filename_pre2: %s'%(input_filename_pre2))
							print('updated filename_save_thresh2: %s'%(filename_save_thresh2))

					# load estimated peak-gene correlations and perform pre-selection
					# compute_mode: 1, calculate peak-gene correlation; 2, perform peak-gene selection by threshold; 3, calculate peak-gene correlation for background peaks
					# compute_mode = flag_computation_1
					if flag_thresh1>0:
						if compute_mode in [1,2]:
							start = time.time()
							df_peak_query = self.df_gene_peak_distance
							# correlation_type = 'spearmanr'
							correlation_type = select_config['correlation_type_1']

							input_filename_pre2 = select_config['input_filename_pre2']
							output_filename = input_filename_pre2

							df_gene_peak_query = self.test_gene_peak_query_correlation_thresh1(gene_query_vec=gene_query_vec,
																								df_gene_peak_query=df_gene_peak_1,
																								df_peak_annot=df_peak_query,
																								correlation_type=correlation_type,
																								save_mode=1,
																								output_filename=output_filename,
																								select_config=select_config)
							stop = time.time()
							df_gene_peak_query_thresh1 = df_gene_peak_query
							print('pre-selection of peak-gene links used %.5fs'%(stop-start))

				if coherent_mode>0:
					column_vec_query = ['pval1']
					if (flag_thresh1>0) and (len(df_gene_peak_query_thresh1)>0):
						df_gene_peak_pre1 = df_gene_peak_query_thresh1
					else:
						df_gene_peak_pre1 = dict_query1[compute_mode_1]
					
					df_gene_peak_bg = dict_query1[compute_mode_2]
					save_mode_2 = 1

					input_filename_pre2 = select_config['input_filename_pre2']
					output_filename = input_filename_pre2
					
					if verbose_internal==2:
						print('peak-gene links, dataframe of size ',df_gene_peak_pre1.shape)
						print('preview:\n',df_gene_peak_pre1[0:2])
						print('peak-gene links background, dataframe of size ',df_gene_peak_bg.shape)
						print('preview:\n',df_gene_peak_bg[0:2])
					
					# query empirical p-values of peak accessibility-gene expression correlations for candidate peak-gene links
					df_gene_peak_query_1 = self.test_query_feature_correlation_merge_2(df_gene_peak_query=df_gene_peak_pre1,
																						df_gene_peak_bg=df_gene_peak_bg,
																						filename_list=[],
																						column_idvec=column_idvec,
																						column_vec_query=column_vec_query,
																						flag_combine=1,
																						coherent_mode=coherent_mode,
																						index_col=0,
																						save_mode=1,
																						output_file_path='',
																						output_filename=output_filename,
																						verbose=verbose,select_config=select_config)

			# data_file_type_query = select_config['data_file_type_query']
			data_file_type_query = select_config['data_file_type']
			input_filename_pre2 = select_config['input_filename_pre2']
			file_path_save_local = select_config['data_path_save_local']
			filename_prefix_default = select_config['filename_prefix_default']

			# flag_iteration_bg = 1
			flag_iteration_bg = 0
			if flag_iteration_bg>0:
				input_file_path_2 = file_path_save_local
				gene_query_num1 = len(gene_query_vec)
				interval = 500
				iter_num = int(np.ceil(gene_query_num1/interval))
				filename_list_bg = []
				for i1 in range(iter_num):
					start_id1 = interval*i1
					start_id2 = np.min([interval*(i1+1),gene_query_num1])
					
					input_filename = '%s/%s.pre1_bg_%d_%d.combine.thresh1.1.txt'%(input_file_path_2,filename_prefix_default,start_id1,start_id2)
					filename_list_bg.append(input_filename)

					if os.path.exists(input_filename)==False:
						print('the file does not exist: %s'%(input_filename))
						return

				select_config.update({'filename_list_bg':filename_list_bg})

			flag_combine_empirical_1 = 0
			if 'flag_combine_empirical_1' in select_config:
				flag_combine_empirical_1 = select_config['flag_combine_empirical_1']

			# query empirical p-values of peak accessibility-gene expression correlations for candidate peak-gene links
			if flag_combine_empirical_1>0:
				column_vec_query = ['pval1']
				coherent_mode = 0
				filename_list = [input_filename_pre2] + filename_list_bg
				output_filename = input_filename_pre2
				print('query emprical p-value estimation')
				print('filename_list_bg: ',len(filename_list_bg))
				if verbose>0:
					print(filename_list_bg[0:2])

				self.test_query_feature_correlation_merge_2(df_gene_peak_query=[],df_gene_peak_bg=[],
															filename_list=filename_list,
															column_idvec=column_idvec,
															column_vec_query=column_vec_query,
															flag_combine=1,
															coherent_mode=coherent_mode,index_col=0,
															save_mode=1,
															output_file_path='',
															output_filename=output_filename,
															verbose=verbose,select_config=select_config)

			flag_merge_1=0
			if 'flag_merge_1' in select_config:
				flag_merge_1 = select_config['flag_merge_1']

			# combine the peak-gene correlations of different subsets of peak-gene links
			if flag_merge_1>0:
				filename_list = select_config['filename_list_bg']
				output_file_path = input_file_path_2
				output_filename = '%s/%s.pre1_bg.combine.thresh1.1.txt'%(output_file_path,filename_prefix_default)
				compute_mode_query = 3
				self.test_query_feature_correlation_merge_1(df_gene_peak_query=[],filename_list=[],
															flag_combine=1,
															compute_mode=compute_mode_query,
															index_col=0,
															save_mode=1,
															output_path=output_file_path,
															output_filename=output_filename,
															verbose=verbose,select_config=select_config)
	
		# add columns to the original peak-gene link dataframe: empirical p-values for a subset of the peak-gene links
		input_file_path = save_file_path
		output_file_path = save_file_path
		filename_save_annot_pre1 = select_config['filename_save_annot_pre1']
		
		flag_combine_empirical_2 = select_config['flag_combine_empirical']
		if flag_combine_empirical_2>0:
			output_filename = '%s/df_gene_peak_distance_annot.%s.txt'%(output_file_path,filename_save_annot_pre1)
			highly_variable_thresh = select_config['highly_variable_thresh']
			# overwrite_1 = False
			query_mode = 2
			# query_mode = 0
			
			flag_query = 1
			if (os.path.exists(output_filename)==True):
				print('the file exists: %s'%(output_filename))
				if query_mode in [2]:
					input_filename = output_filename
					df_gene_peak_distance = pd.read_csv(input_filename,index_col=0,sep='\t') # add to the existing file
				elif query_mode==0:
					flag_query = 0
			else:
				query_mode = 1

			if flag_query>0:
				load_mode = 0
				if query_mode in [1]:
					df_gene_peak_distance = self.df_gene_peak_distance

				df_gene_peak_query_compute1, df_gene_peak_query_thresh1 = self.test_gene_peak_query_correlation_pre1_combine(gene_query_vec=[],peak_distance_thresh=peak_distance_thresh,
																																df_gene_peak_distance=df_gene_peak_distance,
																																highly_variable=highly_variable,
																																highly_variable_thresh=highly_variable_thresh,
																																load_mode=load_mode,
																																input_file_path=input_file_path,
																																save_mode=1,
																																save_file_path=output_file_path,
																																output_filename=output_filename,
																																filename_prefix_save='',
																																select_config=select_config)

		flag_query_thresh2 = select_config['flag_query_thresh2']
		# pre-select peak-gene links by empirical p-values estimated from background peaks matching GC content and average chromatin accessibility
		if flag_query_thresh2>0:
			input_filename_pre2 = select_config['input_filename_pre2']
			input_filename = input_filename_pre2
			
			output_filename_1 = input_filename
			# output_filename_2 = '%s/%s.combine.thresh2.%s.txt'%(save_file_path,filename_prefix,filename_annot1)
			output_filename_2 = select_config['filename_save_thresh2']
			
			overwrite_1 = False
			if 'overwrite_thresh2' in select_config:
				overwrite_1 = select_config['overwrite_thresh2']
			
			flag_query = 1
			if os.path.exists(output_filename_2)==True:
				print('the file exists: %s'%(output_filename_2))
				if overwrite_1==False:
					flag_query = 0

			if flag_query>0:
				# pre-selection of peak-gene links using thresholds on the peak-gene correlations and empirical p-values
				df_gene_peak_query_thresh2, df_gene_peak_query = self.test_gene_peak_query_correlation_pre1_select_1(gene_query_vec=[],df_gene_peak_query=[],
																														peak_loc_query=[],
																														input_filename=input_filename,
																														highly_variable=highly_variable,
																														peak_distance_thresh=peak_distance_thresh,
																														save_mode=1,
																														save_file_path=save_file_path,
																														output_filename_1=output_filename_1,
																														output_filename_2=output_filename_2,
																														filename_prefix_save='',
																														verbose=verbose,
																														select_config=select_config)
			else:
				input_filename_1 = output_filename_1
				input_filename_2 = output_filename_2
				input_filename_list1 = [input_filename_1,input_filename_2]
				list1 = [pd.read_csv(input_filename,sep='\t') for input_filename in input_filename_list1]
				df_gene_peak_query, df_gene_peak_query_thresh2 = list1

			self.df_gene_peak_query_thresh2 = df_gene_peak_query_thresh2

		return df_gene_peak_query_thresh2, df_gene_peak_query

	## ====================================================
	# compute peak accessibility-gene expression correlation for peak-gene links
	def test_gene_peak_query_correlation_pre1_compute(self,gene_query_vec=[],gene_query_vec_2=[],peak_distance_thresh=500,
														df_gene_peak_query=[],df_gene_peak_compute=[],peak_loc_query=[],atac_ad=[],rna_exprs=[],
														highly_variable=False,interval_peak_corr=50,interval_local_peak_corr=10,
														save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		compute peak accessibility-gene expression correlation for peak-gene links
		:param gene_query_vec: (array or list) the target genes
		:param gene_query_vec_2: (array or list) the target genes
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param df_gene_peak_compute: (dataframe) annotations of peak-gene links
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param highly_variable: indicator of whether to only include highly variable gene as target genes
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param interval_local_peak_corr: the number of genes in the sub-batch for which to compute peak-gene correlations in the batch mode
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) updated annotations of peak-gene links
		"""

		gene_query_vec_pre1 = gene_query_vec
		gene_query_num_1 = len(gene_query_vec_pre1)
		# print('target gene set: %d '%(gene_query_num_1))
		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		recompute=0
		if 'recompute' in select_config:
			recompute = select_config['recompute']

		if len(atac_ad)==0:
			atac_ad = self.atac_meta_ad
			print('load atac_ad ',atac_ad.shape)
		if len(rna_exprs)==0:
			rna_exprs = self.meta_scaled_exprs
			print('load rna_exprs ',rna_exprs.shape)

		filename_prefix_1 = select_config['filename_prefix_default']
		filename_prefix_save_1 = select_config['filename_prefix_save_default']

		iter_mode = 0
		if (query_id1>=0) and (query_id2>query_id1):
			iter_mode = 1  # query gene subset
			start_id1 = query_id1
			start_id2 = np.min([query_id2,gene_query_num_1])
			gene_query_vec = gene_query_vec_pre1[start_id1:start_id2]
			# filename_prefix_save = '%s_%d_%d'%(filename_prefix_save_1,start_id1,start_id2)
			# filename_prefix_save_bg = '%s_bg_%d_%d'%(filename_prefix_save_1,start_id1,start_id2)
			annot_str1 = '%d_%d'%(query_id1,query_id2)
			filename_prefix_save = '%s_%s'%(filename_prefix_save_1,annot_str1)
			filename_prefix_save_bg = '%s_bg_%s'%(filename_prefix_save_1,annot_str1)
			filename_prefix_2 = '%s.%s'%(filename_prefix_1,annot_str1)
		elif query_id1<-1:
			iter_mode = 2  # combine peak-gene estimation from different runs
		else:
			gene_query_vec = gene_query_vec_pre1
			filename_prefix_save = filename_prefix_save_1
			filename_prefix_save_bg = '%s_bg'%(filename_prefix_save_1)
			# filename_prefix_2 = filename_prefix_1
			start_id1 = 0
			start_id2 = gene_query_num_1
		select_config.update({'iter_mode':iter_mode})

		filename_prefix = '%s.%s'%(filename_prefix_1,filename_prefix_save_1)
		filename_prefix_local = '%s.%s'%(filename_prefix_1,filename_prefix_save)	# filename prefix with batch information
		# filename_prefix_local = '%s.%s'%(filename_prefix_2,filename_prefix_save_1)	# filename prefix with batch information

		filename_prefix_bg = '%s.%s_bg'%(filename_prefix_1,filename_prefix_save_1)
		filename_prefix_bg_local = '%s.%s'%(filename_prefix_1,filename_prefix_save_bg) # filename prefix with batch information
		# filename_prefix_bg_local = '%s.%s_bg'%(filename_prefix_2,filename_prefix_save_1) # filename prefix with batch information

		filename_annot1 = select_config['filename_annot_default']
		input_filename_pre2_bg = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix_bg,filename_annot1)
		filename_pre2_bg_local = '%s/%s.combine.thresh1.%s.txt'%(save_file_path,filename_prefix_bg_local,filename_annot1)

		select_config.update({'filename_prefix_peak_gene':filename_prefix,
								'filename_prefix_bg_peak_gene':filename_prefix_bg,
								'filename_prefix_local':filename_prefix_local,
								'filename_prefix_bg_local':filename_prefix_bg_local,
								'input_filename_pre2_bg':input_filename_pre2_bg})

		df_gene_peak_query_ori = df_gene_peak_query.copy()
		df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
		gene_name_query_1 = df_gene_peak_query['gene_id'].unique()
		gene_query_vec_ori = gene_query_vec
		gene_query_vec = pd.Index(gene_query_vec_ori).intersection(gene_name_query_1,sort=False)
		
		gene_query_vec_2 = gene_query_vec
		gene_query_num_1 = len(gene_name_query_1)
		gene_query_num_ori = len(gene_query_vec_ori)
		gene_query_num = len(gene_query_vec)

		df_gene_peak_query = df_gene_peak_query.loc[gene_query_vec,:]
		# print('peak accessibility-gene expr correlation estimation ')
		
		gene_query_vec_bg = gene_query_vec
		gene_query_num_bg = len(gene_query_vec_bg)
		
		interval_peak_corr, interval_local_peak_corr = select_config['interval_peak_corr'], select_config['interval_local_peak_corr']
		interval_peak_corr_bg, interval_local_peak_corr_bg = interval_peak_corr, interval_local_peak_corr

		flag_combine_bg=1
		# flag_combine_bg=0
		# recompute_1=0
		flag_corr_, method_type, type_id_1 = 1, 1, 1 # correlation without estimating emprical p-value; correlation and p-value; spearmanr
		select_config.update({'flag_corr_':flag_corr_,
								'method_type_correlation':method_type,
								'type_id_correlation':type_id_1})

		select_config.update({'gene_query_vec_bg':gene_query_vec_bg,
								'flag_combine_bg':flag_combine_bg,
								'interval_peak_corr_bg':interval_peak_corr_bg,
								'interval_local_peak_corr_bg':interval_peak_corr_bg})

		flag_computation_1 = select_config['flag_computation_1']
		compute_mode = flag_computation_1
		# peak-gene correlation estimation
		df_gene_peak_compute_1 = df_gene_peak_compute
		if flag_computation_1 in [1,3]:
			df_gene_peak_query = self.test_gene_peak_query_correlation_unit1(gene_query_vec=gene_query_vec,
																				gene_query_vec_2=[],
																				peak_distance_thresh=500,
																				df_gene_peak_query=df_gene_peak_query,
																				peak_loc_query=[],
																				atac_ad=atac_ad,
																				rna_exprs=rna_exprs,
																				flag_computation_1=1,
																				highly_variable=False,
																				interval_peak_corr=interval_peak_corr,
																				interval_local_peak_corr=interval_local_peak_corr,
																				save_mode=1,
																				save_file_path=save_file_path,
																				output_filename='',
																				filename_prefix_save='',
																				verbose=0,select_config=select_config)
			df_gene_peak_compute_1 = df_gene_peak_query

		return df_gene_peak_query

	## ====================================================
	# perform pre-selection of peak-gene links using distance-dependent thresholds on peak accessibility-gene expression correlations
	def test_gene_peak_query_correlation_thresh1(self,gene_query_vec=[],df_gene_peak_query=[],df_peak_annot=[],correlation_type='spearmanr',
													save_mode=1,output_filename='',float_format='%.5E',select_config={}):

		"""
		perform pre-selection of peak-gene links using distance-dependent thresholds on peak accessibility-gene expression correlations
		:param gene_query_vec: (array or list) the target genes
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param df_peak_annot: (dataframe) annotations of genome-wide peak-gene links including peak-gene distances
		:param correlation_type: the type of peak accessibility-gene expression correlation
		:param save_mode: indicator of whether to save data
		:param output_filename: filename to save data
		:param float_format: the format to keep data precision used in saving data
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of pre-selected peak-gene links
		"""

		flag_query1=1
		if flag_query1>0:
			df_gene_peak_compute1 = df_gene_peak_query
			if len(gene_query_vec)>0:
				# print('perform peak-gene link pre-selection for the gene subset: %d genes'%(len(gene_query_vec)))
				query_id_ori = df_gene_peak_query.index.copy()
				df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
				# df_gene_peak_query_compute1 = df_gene_peak_query.copy()
				
				gene_query_idvec = pd.Index(gene_query_vec).intersection(df_gene_peak_query.index,sort=False)
				# print('gene_query_vec_2: %d, gene_query_idvec: %d'%(len(gene_query_vec),len(gene_query_idvec)))
				print('perform peak-gene link pre-selection for the gene subset: %d genes'%(len(gene_query_idvec)))
				df_gene_peak_query = df_gene_peak_query.loc[gene_query_idvec,:]
			else:
				print('perform peak-gene link pre-selection for genes with estimated peak-gene correlations')
				# df_gene_peak_query_compute1 = df_gene_peak_query

			print('peak-gene links, dataframe of size ',df_gene_peak_query.shape)

			# select gene-peak links above correlation thresholds for each distance range for empricial p-value calculation
			if 'thresh_corr_distance_1' in select_config:
				thresh_corr_distance = select_config['thresh_corr_distance_1']
			else:
				# thresh_distance_1 = 50
				thresh_distance_1 = 100
				if 'thresh_distance_default_1' in select_config:
					thresh_distance_1 = select_config['thresh_distance_default_1'] # the distance threshold with which we retain the peaks without thresholds of correlation and p-value
				thresh_corr_distance = [[0,thresh_distance_1,0],
										[thresh_distance_1,500,0.01],
										[500,1000,0.1],
										[1000,2050,0.15]]

			if not ('distance' in df_gene_peak_query):
				field_query = ['distance']
				column_idvec = ['peak_id','gene_id']
				df_peak_annot = self.df_gene_peak_distance
				# query peak-gene link attributes
				# query peak-gene TSS distance
				df_gene_peak_query = self.test_gene_peak_query_attribute_1(df_gene_peak_query=df_gene_peak_query,
																			df_gene_peak_query_ref=df_peak_annot,
																			column_idvec=column_idvec,
																			field_query=field_query,
																			column_name=[],
																			reset_index=False,
																			select_config=select_config)

			column_idvec = ['gene_id','peak_id']
			column_id1, column_id2 = column_idvec[0:2]
			df_gene_peak_query.index = utility_1.test_query_index(df_gene_peak_query,column_vec=column_idvec)
			distance_abs = df_gene_peak_query['distance'].abs()
			list1 = []
			
			print('threshold for peak-gene link pre-selection: ')
			print('distance threshold 1, distance threshold 2, correlation threshold')
			
			column_id = correlation_type
			query_idvec = df_gene_peak_query.index
			for thresh_vec in thresh_corr_distance:
				constrain_1, constrain_2, thresh_corr_ = thresh_vec
				print(constrain_1, constrain_2, thresh_corr_)
				id1 = (distance_abs<constrain_2)&(distance_abs>=constrain_1)
				id2 = (df_gene_peak_query[column_id].abs()>thresh_corr_)
				query_id1 = query_idvec[id1&id2]
				list1.extend(query_id1)

			query_id_sub1 = pd.Index(list1).unique()
			df_gene_peak_query_ori = df_gene_peak_query.copy()
			df_gene_peak_query = df_gene_peak_query_ori.loc[query_id_sub1]
			df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
			df_gene_peak_query = df_gene_peak_query.sort_values(by=['gene_id','distance'],ascending=True)

			print('original peak-gene links, dataframe of size ',df_gene_peak_query_ori.shape)
			print('peak-gene links after pre-selection by correlation thresholds, dataframe of size ',df_gene_peak_query.shape)
			if (save_mode>0) and (output_filename!=''):
				# df_gene_peak_query.to_csv(output_filename,sep='\t',float_format='%.6E')
				df_gene_peak_query.to_csv(output_filename,sep='\t',float_format=float_format)

			return df_gene_peak_query

	## ====================================================
	# compute peak accessibility-gene expression correlation for peak-gene links
	def test_gene_peak_query_correlation_unit1(self,gene_query_vec=[],gene_query_vec_2=[],peak_distance_thresh=2000,
													df_gene_peak_query=[],peak_loc_query=[],atac_ad=[],rna_exprs=[],flag_computation_1=1,
													highly_variable=False,interval_peak_corr=50,interval_local_peak_corr=10,
													save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		compute peak accessibility-gene expression correlation for peak-gene links
		:param gene_query_vec: (array or list) the target genes
		:param gene_query_vec_2: (array or list) the target genes
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param flag_computation_1: indicator of whether to compute peak accessibility-gene expression correlation
		:param highly_variable: indicator of whether to only include highly variable gene as target genes
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param interval_local_peak_corr: the number of genes in the sub-batch for which to compute peak-gene correlations in the batch mode
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of peak-gene links with peak accessibility-gene expression correlation computed
		"""

		if ('flag_computation_1' in select_config):
			flag_computation_1 = select_config['flag_computation_1']

		recompute=0
		if 'recompute' in select_config:
			recompute = select_config['recompute']
		flag_query_1 = 1

		df_gene_peak_query_1 = []
		computation_mode_vec = [[0,0,0],[1,1,0],[0,1,0],[0,0,1]]
		compute_mode = flag_computation_1
		flag_query1, flag_query2, background_query = computation_mode_vec[compute_mode]

		iter_mode = select_config['iter_mode']
		field_query = ['input_filename_pre1','input_filename_pre1','input_filename_pre2_bg']

		if (flag_computation_1>0) or (flag_query_1>0):
			if compute_mode==3:
				# estimate empirical p-value using background peaks
				save_file_path2 = save_file_path  # the directory to save the .npy file and estimation file for subsets
				filename_prefix_bg_local = select_config['filename_prefix_bg_local']
				filename_prefix_query = filename_prefix_bg_local
				if iter_mode==0:
					save_file_path1 = save_file_path
				else:
					# the parallel mode
					save_file_path1 = save_file_path2

				if len(df_gene_peak_query)==0:
					input_filename_pre2 = select_config['input_filename_pre2']
					df_gene_peak_query = pd.read_csv(input_filename_pre2,index_col=0,sep='\t')
					print('load pre-selected peak-gene associations: %s'%(input_filename_pre2))
			else:
				filename_prefix_local = select_config['filename_prefix_local']
				filename_prefix_query = filename_prefix_local
				save_file_path2 = save_file_path
				if iter_mode==0:
					save_file_path1 = save_file_path
					field_id1 = field_query[compute_mode-1]
					output_filename = select_config[field_id1]
				else:
					save_file_path1 = save_file_path2

			if (compute_mode in [1,3]):
				if output_filename=='':
					if iter_mode==0:
						field_id1 = field_query[compute_mode-1]
						output_filename = select_config[field_id1]
					else:
						filename_annot_vec = ['combine',-1,'combine.thresh1']
						filename_annot_1 = filename_annot_vec[compute_mode-1]
						output_filename = '%s/%s.%s.1.txt'%(save_file_path1,filename_prefix_query,filename_annot_1)

		if flag_computation_1>0:
			# query the compuation mode
			# flag_query1: compute peak-gene correlation for foreground peaks
			# flag_query2: thresholding for foreground peaks
			# background_query: compute peak-gene correlation for background peaks
			# flag_query1, flag_query2, background_query = 1,1,0 # 0,0,1;
			# flag_query1, flag_query2, background_query = 0,0,1 # 1,1,0;
			select_config.update({'flag_query1':flag_query1,
									'flag_query2':flag_query2,
									'background_query':background_query})

			if os.path.exists(save_file_path1)==False:
				print('the directory does not exist: %s'%(save_file_path1))
				os.mkdir(save_file_path1)
			select_config.update({'save_file_path_local':save_file_path1})

			if compute_mode in [1,3]:
				print('peak accessibility-gene expr correlation estimation for peak-gene links ')
				start = time.time()
				df_gene_peak_query_compute1, df_gene_peak_query = self.test_gene_peak_query_correlation_pre2(gene_query_vec=gene_query_vec,
																												gene_query_vec_2=gene_query_vec_2,
																												df_peak_query=df_gene_peak_query,
																												peak_dict=[],
																												atac_ad=atac_ad,
																												rna_exprs=rna_exprs,
																												interval_peak_corr=interval_peak_corr,
																												interval_local_peak_corr=interval_local_peak_corr,
																												compute_mode=compute_mode,
																												recompute=recompute,
																												save_file_path=save_file_path,
																												save_file_path_local=save_file_path1,
																												output_filename=output_filename,
																												filename_prefix_save=filename_prefix_query,
																												select_config=select_config)
				stop = time.time()
				# print('peak accessibility-gene expr correlation estimation for peaks used %.5fs'%(stop-start))
				df_gene_peak_query_1 = df_gene_peak_query

		df_gene_peak_query_2 = df_gene_peak_query_1
		
		flag_query_1 = 0
		if flag_query_1>0:
			flag_combine_1=0
			if iter_mode==0:
				if 'gene_pre1_flag_combine_1' in select_config:
					flag_combine_1 = select_config['gene_pre1_flag_combine_1']

			df_gene_peak_query_2 = df_gene_peak_query_1
			
			# if (flag_combine_1>0) and (iter_mode!=0):
			if (flag_combine_1>0):
				if (iter_mode==0):
					input_filename_list1 = []
					filename_annot1 = select_config['filename_save_annot_1']
					if compute_mode==1:
						filename_prefix = select_config['filename_prefix_peak_gene']
						if 'filename_list_pre1' in select_config:
							input_filename_list1 = select_config['filename_list_pre1']

					elif compute_mode==3:
						if 'filename_list_bg' in select_config:
							input_filename_list1 = select_config['filename_list_bg']

					else:
						pass

					if len(input_filename_list1)>0:
						df_gene_peak_query_1 = utility_1.test_file_merge_1(input_filename_list1,index_col=0,header=0,float_format=-1,
																			save_mode=1,verbose=verbose,output_filename=output_filename)
					else:
						print('please perform peak accessibility-gene expression correlation estimation or load estimated correlations')

			flag_combine_2=0
			if 'gene_pre1_flag_combine_2' in select_config:
				flag_combine_2 = select_config['gene_pre1_flag_combine_2']

			df_gene_peak_query_2 = df_gene_peak_query_1
			if flag_combine_2>0:
				# combine estimated empirical p-values with orignal p-values
				if compute_mode==3:
					input_filename_pre2 = select_config['input_filename_pre2']
					input_filename_pre2_bg = select_config['input_filename_pre2_bg']

					if os.path.exists(input_filename_pre2_bg)==False:
						print('the file does not exist: %s'%(input_filename_pre2_bg))
						filename_bg = output_filename
					else:
						filename_bg = input_filename_pre2_bg
					print('combine peak-gene correlation estimation: ',filename_bg)

					# input_filename_list = [input_filename_pre2,input_filename_pre2_bg]
					input_filename_list = [input_filename_pre2,filename_bg]
					
					# copy specified columns from the other dataframes to the first dataframe
					df_gene_peak_query_2 = utility_1.test_column_query_1(input_filename_list,id_column=['peak_id','gene_id'],df_list=[],
																			column_vec=['pval1'],reset_index=False)
					output_filename_1 = input_filename_pre2
					df_gene_peak_query_2.index = np.asarray(df_gene_peak_query_2['gene_id'])
					df_gene_peak_query_2.to_csv(output_filename_1,sep='\t')
				# else:
				# 	df_gene_peak_query_2 = df_gene_peak_query_1

		return df_gene_peak_query_2

	## ====================================================
	# compute peak accessibility-gene expression correlation and p-value
	def test_gene_peak_query_correlation_pre2(self,gene_query_vec=[],gene_query_vec_2=[],df_peak_query=[],peak_dict=[],atac_ad=[],rna_exprs=[],
												interval_peak_corr=50,interval_local_peak_corr=10,correlation_type='spearmanr',compute_mode=1,recompute=0,
												save_mode=1,save_file_path='',save_file_path_local='',output_filename='',filename_prefix_save='',
												verbose=0,select_config={}):

		"""
		compute peak accessibility-gene expression correlation and p-value
		:param gene_query_vec: (array or list) the target genes
		:param gene_query_vec_2: (array or list) the target genes
		:param df_peak_query: (dataframe) annotations of peak-gene links
		:param peak_dict: dictionary containing peak annotations, including the possible target genes of peaks
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param interval_local_peak_corr: the number of genes in the sub-batch for which to compute peak-gene correlations in the batch mode
		:param correlation_type: the type of peak accessibility-gene expression correlation
		:param compute_mode: the type of computation to perform
		:param recompute: indicator of whether to recompute the peak-gene correlations if the correlations have been computed and saved
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param save_file_path_local: the second directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dataframe of peak-gene link annotations with peak accessibility-gene expression correlations and p-values
		"""

		file_path1 = self.save_path_1 # the default file path
		if len(atac_ad)==0:
			atac_ad = self.atac_meta_ad
		if len(rna_exprs)==0:
			rna_exprs = self.meta_scaled_exprs

		## parameter configuration
		flag_corr_, method_type, type_id_1 = 1, 1, 1 # correlation without estimating emprical p-value; correlation and p-value; spearmanr
		# flag_corr_, method_type, type_id_1 = 0, 0, 1 # correlation for background peaks without estimating emprical p-value; correlation; spearmanr
		recompute_1 = 1
		config_default = {'flag_corr_':flag_corr_,
							'method_type_correlation':method_type,
							'type_id_correlation':type_id_1,
							'recompute':recompute_1}

		field_query = list(config_default.keys())
		for field_id1 in field_query:
			if not (field_id1 in select_config):
				select_config.update({field_id1:config_default[field_id1]})

		# compute peak-gene correlation for foreground peaks: mode 1;
		# compute peak-gene correlation for background peaks: mode 3;
		# thresholding for foreground peaks: mode 2
		if compute_mode in [1,3]:
			# peak accessibilty-gene expression correlation calculation without estimating emprical p-value
			flag_compute=1
			field_query = ['input_filename_pre1','input_filename_pre1','input_filename_pre2_bg']
			
			if (os.path.exists(output_filename)==True) and (recompute==0):
				print('the file exists: %s'%(output_filename))
				df_gene_peak_query = pd.read_csv(output_filename,index_col=0,sep='\t')
				df_gene_peak_query_compute1 = df_gene_peak_query
			else:
				if compute_mode==1:
					interval_peak_corr_1, interval_local_peak_corr_1 = interval_peak_corr, interval_local_peak_corr
					df_gene_peak_query_1 = df_peak_query
					# filename_prefix_save = select_config['filename_prefix_save']
					if filename_prefix_save=='':
						# filename_prefix_save = select_config['filename_prefix_peak_gene']
						filename_prefix_local = select_config['filename_prefix_local']
						filename_prefix_save = select_config['filename_prefix_local']
						print('filename_prefix_local:%s'%(filename_prefix_local))

					type_id_1 = select_config['type_id_correlation']
					rename_column=1
					select_config.update({'rename_column':rename_column})
					peak_bg_num = -1		
				else:
					print('load background peak loci ')
					input_filename_peak, input_filename_bg, peak_bg_num = select_config['input_filename_peak'],select_config['input_filename_bg'],select_config['peak_bg_num']
					peak_bg = self.test_gene_peak_query_bg_load(input_filename_peak=input_filename_peak,
																input_filename_bg=input_filename_bg,
																peak_bg_num=peak_bg_num)

					self.peak_bg = peak_bg
					if verbose>0:
						print('peak_bg ',peak_bg.shape,peak_bg[0:5])
						print('peak_bg_num: %d'%(peak_bg_num))

					list_interval = [interval_peak_corr, interval_local_peak_corr]
					field_query = ['interval_peak_corr_bg','interval_local_peark_corr']
					for i1 in range(2):
						if field_query[i1] in select_config:
							list_interval[i1] = select_config[field_query[i1]]
					interval_peak_corr_1, interval_local_peak_corr_1 = list_interval

					flag_corr_, method_type, type_id_1 = 0, 0, 1
					# select_config.update({'flag_corr_':flag_corr_,'method_type_correlation':method_type})
					select_config.update({'flag_corr_':flag_corr_})		
					if not ('type_id_correlation') in select_config:
						select_config.update({'type_id_correlation':type_id_1})
					
					rename_column=0
					select_config.update({'rename_column':rename_column})
					
					if 'gene_query_vec_bg' in select_config:
						gene_query_vec_bg = select_config['gene_query_vec_bg']
					else:
						gene_query_vec_bg = gene_query_vec

					if filename_prefix_save=='':
						filename_prefix_save_bg = select_config['filename_prefix_bg_local']
						filename_prefix_save = filename_prefix_save_bg
						print('filename_prefix_save_bg:%s'%(filename_prefix_save_bg))

					if len(df_peak_query)==0:
						input_filename_pre2 = select_config['input_filename_pre2']
						df_gene_peak_query = pd.read_csv(input_filename_pre2,index_col=0,sep='\t')
						print('load pre-selected peak-gene associations: %s'%(input_filename_pre2))
					else:
						df_gene_peak_query = df_peak_query

					# print('peak-gene links, dataframe of size ',df_gene_peak_query.shape)
					gene_query_idvec_1 = df_gene_peak_query['gene_id'].unique()
					gene_query_num1 = len(gene_query_idvec_1)
					gene_query_vec_bg1 = pd.Index(gene_query_vec_bg).intersection(gene_query_idvec_1,sort=False)
					gene_query_num_bg, gene_query_num_bg1 = len(gene_query_vec_bg), len(gene_query_vec_bg1)
					# print('gene_query_idvec_1:%d, gene_query_vec_bg:%d, gene_query_vec_bg1:%d '%(gene_query_num1,gene_query_num_bg,gene_query_num_bg1))

					gene_query_vec = gene_query_vec_bg1
					df_gene_peak_query_1 = df_gene_peak_query
					print('gene number: %d'%(gene_query_num_bg1))

				warnings.filterwarnings('ignore')
				print('peak accessibility-gene expression correlation estimation')
				if verbose>0:
					print('ATAC-seq data, anndata of ',atac_ad.shape)
					print('gene expressions, dataframe of size ',rna_exprs.shape)
					print('filename_prefix_save: %s'%(filename_prefix_save))
					print('save_file_path: %s, save_file_path_local: %s'%(save_file_path,save_file_path_local))
				
				# compute peak accessibility-gene expression correlation
				df_gene_peak_query = self.test_gene_peak_query_correlation_1(gene_query_vec=gene_query_vec,peak_dict=peak_dict,
																				df_gene_peak_query=df_gene_peak_query_1,
																				atac_ad=atac_ad,
																				rna_exprs=rna_exprs,
																				flag_compute=flag_compute,
																				interval_peak_corr=interval_peak_corr_1,
																				interval_local_peak_corr=interval_local_peak_corr_1,
																				peak_bg_num=peak_bg_num,
																				save_file_path=save_file_path,
																				save_file_path_local=save_file_path_local,
																				filename_prefix_save=filename_prefix_save,
																				verbose=verbose,select_config=select_config)

				if (save_mode>0) and (output_filename!=''):
					df_gene_peak_query.to_csv(output_filename,sep='\t',float_format='%.6E')
					print('save file: %s'%(output_filename))

				df_gene_peak_query_compute1 = df_gene_peak_query
				# print('peak-gene links, dataframe of size ',df_gene_peak_query.shape)
				print('compute_mode: %d'%(compute_mode))
				
				warnings.filterwarnings('default')

		return df_gene_peak_query_compute1, df_gene_peak_query
		# return df_gene_peak_query

	## ====================================================
	# query peak accessibility-gene expression correlation and p-value
	def test_gene_peak_query_correlation_1(self,gene_query_vec=[],peak_dict=[],df_gene_peak_query=[],atac_ad=[],rna_exprs=[],flag_compute=1,
												interval_peak_corr=50,interval_local_peak_corr=10,peak_bg_num=-1,
												save_file_path='',save_file_path_local='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		query peak accessibility-gene expression correlation and p-value
		:param gene_query_vec: (array or list) the target genes
		:param peak_dict: dictionary containing peak annotations, including the possible target genes of peaks
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param flag_compute: indicator of whether to compute the peak-gene correlations or load the computed correlations from saved files 
		:param interval_peak_corr: the number of genes in a batch for which to compute peak-gene correlations in the batch mode
		:param interval_local_peak_corr: the number of genes in the sub-batch for which to compute peak-gene correlations in the batch mode
		:param peak_bg_num: the number of background peaks to sample for each candidate peak, which match the candidate peak by GC content and average accessibility
		:param save_file_path: the directory to save data
		:param save_file_path_local: the second directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		return: (dataframe) the updated peak-gene link annotations with peak accessibility-gene expression correlations and p-values
		"""

		if len(atac_ad)==0:
			atac_ad = self.atac_meta_ad
		if len(rna_exprs)==0:
			rna_exprs = self.meta_scaled_exprs
		select_config.update({'save_file_path':save_file_path,
								'save_file_path_local':save_file_path_local,
								'filename_prefix_save_peak_corr':filename_prefix_save,
								'interval_peak_corr_1':interval_peak_corr,
								'interval_local_peak_corr_1':interval_local_peak_corr})

		if flag_compute>0:
			# compute peak accessibility-gene expr correlation
			start = time.time()
			field_query = ['flag_corr_','method_type_correlation','type_id_correlation']
			flag_corr_,method_type,type_id_1 = 1,1,1
			list1 = [flag_corr_,method_type,type_id_1]
			for i1 in range(3):
				if field_query[i1] in select_config:
					list1[i1] = select_config[field_query[i1]]
			flag_corr_,method_type,type_id_1 = list1
			print('flag_corr_, method_type, type_id_1 ',flag_corr_,method_type,type_id_1)
			recompute=0
			if 'recompute' in select_config:
				recompute = select_config['recompute']
			save_filename_list = self.test_search_peak_dorc_pre1(atac_ad=atac_ad,rna_exprs=rna_exprs,
																	gene_query_vec=gene_query_vec,
																	df_gene_peak_query=df_gene_peak_query,
																	peak_dict=peak_dict,
																	flag_corr_=flag_corr_,
																	method_type=method_type,
																	type_id_1=type_id_1,	
																	recompute=recompute,
																	peak_bg_num=peak_bg_num,
																	save_mode=1,save_file_path=save_file_path_local,
																	filename_prefix_save=filename_prefix_save,
																	select_config=select_config)

			stop = time.time()
			print('peak accessibility-gene expression correlation estimation used %.5fs'%(stop-start))
		else:
			if 'save_filename_list' in select_config:
				save_filename_list = select_config['save_filename_list']
			else:
				interval_save = -1
				if 'interval_save' in select_config:
					interval_save = select_config['interval_save']
				if interval_save<0:
					gene_query_num = len(gene_query_vec)
					iter_num = np.int(np.ceil(gene_query_num/interval_peak_corr))
					save_filename_list = ['%s/%s.%d.1.npy'%(save_file_path_local,filename_prefix_save,iter_id) for iter_id in range(iter_num)]
				else:
					save_filename_list = ['%s/%s.1.npy'%(save_file_path_local,filename_prefix_save)]

		file_num1 = len(save_filename_list)
		list_1 = []
		if file_num1>0:
			# load and combine previously estimated peak accessibility-gene expression correlations
			df_gene_peak_query = self.test_gene_peak_query_combine_1(save_filename_list=save_filename_list,verbose=verbose,select_config=select_config)

		return df_gene_peak_query

	## ====================================================
	# compute peak accessibility-gene expression correlation
	def test_search_peak_dorc_pre1(self,atac_ad,rna_exprs,gene_query_vec=[],df_gene_peak_query=[],peak_dict=[],flag_corr_=1,method_type=1,type_id_1=1,
										recompute=1,peak_bg_num=100,save_mode=1,save_file_path='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		compute peak accessibility-gene expression correlation and p-value
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param gene_query_vec: (array or list) the potential target genes
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_dict: dictionary containing peak annotations, including the possible target genes of peaks
		:param flag_corr_: the type of peak-gene correlation to compute: 
						   0: correlation between candidate peak and potential target gene
						   1: correlation between sampled background peak and potential target gene
						   2: correlation between candidate peak and potential target gene, and correlation between sampled background peak and potential target gene
		:param method_type: the method type used to compute peak-gene correlation: 
		                    0: using the pairwise_distances() function, which computes correlation without p-value
							1: using defined function to compute correlation and p-value
		:param type_id_1: the type of correlation to compute: 1. Spearman's rank correlation; 2. Pearson correlation; 3. both Spearman's rank correlation and Pearson correlation
		:param recompute: indicator of whether to recompute the peak-gene correlation if the correlations have been computed and saved
		:param peak_bg_num: the number of background peaks to sample for each candidate peak, which match the candidate peak by GC content and average accessibility
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: list of paths of files which save the computed peak accessibility-gene expression correlations, which may include p-values
		"""

		atac_meta_ad = atac_ad
		peak_loc = atac_meta_ad.var # DataFrame: peak position and annotation in ATAC-seq data
		sample_id = rna_exprs.index
		sample_id_atac = atac_meta_ad.obs_names
		atac_meta_ad = atac_meta_ad[sample_id,:]

		np.random.seed(0)
		gene_query_vec_ori = gene_query_vec
		gene_query_num = len(gene_query_vec)
		if verbose>0:
			print('peak-gene links, data preview:\n',df_gene_peak_query[0:5])
			print('the number of associated genes: %d, preview: '%(gene_query_num),gene_query_vec[0:10])

		# interval, pre_id1, start_id = 50, -1, 0
		interval, pre_id1, start_id = -1, -1, 0
		if 'interval_peak_corr_1' in select_config:
			interval = select_config['interval_peak_corr_1']
		interval_1 = interval
		if interval>0:
			query_num = (pre_id1+1)+int(np.ceil((gene_query_num-start_id)/interval))
			iter_mode = 1
		else:
			query_num = 1
			interval = gene_query_num
			iter_mode = 0

		# print('gene_query_num:%d,start_id:%d,query_num:%d'%(gene_query_num,start_id,query_num))
		interval_local = -1
		interval_save = -1
		if 'interval_save' in select_config:
			interval_save = select_config['interval_save']
		else:
			select_config.update({'interval_save':interval_save})
		# interval_save = -1
		if 'interval_local_peak_corr_1' in select_config:
			interval_local = select_config['interval_local_peak_corr_1']

		print('interval, interval_local, interval_save ',interval,interval_local,interval_save)
		warnings.filterwarnings('ignore')

		save_filename_list = []
		corr_thresh = 0.7
		column_corr_1 = select_config['column_correlation'][0] # column representing peak accessibility-gene expression correlation
		
		if interval_save>0:
			# only save one file for the combined correlation estimation
			output_filename = '%s/%s.1.npy'%(save_file_path,filename_prefix_save)
			gene_peak_local = dict()

		verbose_internal = self.verbose_internal
		for i1_ori in tqdm(range(query_num)):
			if interval_save<0:
				# save file for the estimation at each inteval
				output_filename = '%s/%s.%d.1.npy'%(save_file_path,filename_prefix_save,i1_ori)
				gene_peak_local = dict()
			
			if os.path.exists(output_filename)==True:
				print('the file exists', output_filename)
				if recompute==0:
					save_filename_list.append(output_filename)
					continue
				# return

			i1 = i1_ori
			num_2 = np.min([start_id+(i1+1)*interval,gene_query_num])
			gene_num2 = num_2
			gene_idvec1 = gene_query_vec[(start_id+i1*interval):num_2]
			# print(len(gene_query_vec),len(gene_name_query_1),len(gene_idvec1),gene_idvec1[0:10])
			# print(len(gene_query_vec),len(gene_idvec1),gene_idvec1[0:10])
			print('the number of genes: %d, the number of genes in the batch: %d, preview: '%(len(gene_query_vec),len(gene_idvec1)),gene_query_vec[0:10])

			df_query = []
			if flag_corr_==0:
				field_query_vec = [['spearmanr'],['pearsonr'],['spearmanr','pearsonr']]
				df_query = df_gene_peak_query.loc[gene_idvec1,['peak_id']+field_query_vec[type_id_1-1]].fillna(-1)
				# print('df_query ',df_query.shape,df_query)
				# print('peak-gene links: ',df_query.shape[0])
				# print('preview: ')
				# print(df_query[0:2])

			# the dorc_func_pre1() function returns (gene_query, df)
			if iter_mode>0:
				if interval_local<=0:
					gene_res = Parallel(n_jobs=-1)(delayed(self.dorc_func_pre1)(np.asarray(df_gene_peak_query.loc[[t_gene_query],'peak_id']),
																				t_gene_query,
																				atac_meta_ad,
																				rna_exprs,
																				flag_corr_=flag_corr_,
																				df_query=df_query,
																				corr_thresh=-2,
																				method_type=method_type,
																				type_id_1=type_id_1)
																				for t_gene_query in tqdm(gene_idvec1))
				else:
					# running in parallel for a subset of the genes
					query_num_local = int(np.ceil(interval/interval_local))
					gene_res = []
					gene_query_num_local = len(gene_idvec1)
					for i2 in range(query_num_local):
						t_id1 = interval_local*i2
						t_id2 = np.min([interval_local*(i2+1),gene_query_num_local])
						if i2%500==0:
							print('gene query', i1, i2, t_id1, t_id2)
							print(gene_idvec1[t_id1:t_id2])
						gene_res_local_query = Parallel(n_jobs=-1)(delayed(self.dorc_func_pre1)(np.asarray(df_gene_peak_query.loc[[t_gene_query],'peak_id']),
																								gene_query=t_gene_query,
																								atac_read=atac_meta_ad,
																								rna_exprs=rna_exprs,
																								flag_corr_=flag_corr_,
																								df_query=df_query,
																								corr_thresh=-2,
																								method_type=method_type,
																								type_id_1=type_id_1)
																								for t_gene_query in tqdm(gene_idvec1[t_id1:t_id2]))
						for t_gene_res in gene_res_local_query:
							gene_res.append(t_gene_res)
			else:
				gene_res = []
				gene_query_num1 = len(gene_idvec1)
				iter_vec = np.arange(gene_query_num1)
				for i1 in tqdm(iter_vec):
					t_gene_query = gene_idvec1[i1]
					peak_vec = np.asarray(df_gene_peak_query.loc[[t_gene_query],'peak_id'])
					if i1%500==0:
						# print('peak_vec, gene_query ',len(peak_vec),t_gene_query,i1)
						print('gene: %s, peak number: %d, %d'%(t_gene_query,len(peak_vec),i1))
					t_gene_res = self.dorc_func_pre1(peak_vec,gene_query=t_gene_query,atac_read=atac_meta_ad,rna_exprs=rna_exprs,
														flag_corr_=flag_corr_,df_query=df_query,
														corr_thresh=-2,method_type=method_type,type_id_1=type_id_1)
					gene_res.append(t_gene_res)

			gene_query_num_1 = len(gene_res)
			for i2 in tqdm(range(gene_query_num_1)):
				vec1 = gene_res[i2]
				if type(vec1) is int:
					continue
				t_gene_query, df = vec1[0], vec1[1]
				try:
				# if len(df)>0:
					gene_peaks = df.index[df[column_corr_1].abs()>corr_thresh]
					gene_peak_local[t_gene_query] = df
					query_num = len(gene_peaks)
					try:
						if query_num>0:
							# print(t_gene_query,query_num,gene_peaks,df.loc[gene_peaks,:])
							print('gene: %s, peak loci with high peak accessibility-gene expression correlation: %d'%(t_gene_query,query_num))
							print(df.loc[gene_peaks,:])
					except Exception as error:
						print('error! ', error)
						print(t_gene_query,gene_peaks,query_num)

				except Exception as error:
					print('error! ', error,t_gene_query)

			print(len(gene_peak_local.keys()))
			if interval_save>0:
				if (gene_num2%interval_save==0):
					np.save(output_filename,gene_peak_local,allow_pickle=True)
			else:
				np.save(output_filename,gene_peak_local,allow_pickle=True)
				save_filename_list.append(output_filename)

		if interval_save>0:
			try:
				if (gene_num2%interval_save!=0):
					np.save(output_filename,gene_peak_local,allow_pickle=True)
				save_filename_list.append(output_filename)
			except Exception as error:
				print('error! ',error)
		warnings.filterwarnings('default')

		return save_filename_list

	## ====================================================
	# compute peak accessibility-gene expression correlation and p-value
	def dorc_func_pre1(self,peak_loc,gene_query,atac_read,rna_exprs,flag_corr_=1,df_query=[],spearman_cors=[],pearson_cors=[],gene_id_query='',
							corr_thresh=0.01,method_type=0,type_id_1=0,background_query=0,verbose=0):

		"""
		compute peak accessibility-gene expression correlation and p-value
		:param peak_loc: (array) candidate peaks of the potential target gene
		:param gene_query: (str) the potential target gene
		:param atac_read: (AnnData object or dataframe) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param flag_corr_: the type of peak-gene correlation to compute: 
						   0: correlation between candidate peak and potential target gene
						   1: correlation between sampled background peak and potential target gene
						   2: correlation between candidate peak and potential target gene, and correlation between sampled background peak and potential target gene
		:param df_query: (dataframe) annotations of candidate peak-gene links, which may include pre-computed peak accessibility-gene expression correlations
		:param spearman_cors:list for storing peak accessibility-gene expression Spearman's rank correlation
		:param pearson_cors: list for storing peak accessibility-gene expression Pearson correlation
		:param gene_id_query: indice of the potential target gene
		:param corr_thresh: threshold on peak-gene correlation to print peak-gene links with high correlations
		:param method_type: the method type used to compute peak-gene correlation: 
							0: using the pairwise_distances() function, which computes correlation without p-value
						    1: using defined function to compute correlation and p-value
		:param type_id_1: the type of correlation to compute: 1. Spearman's rank correlation; 2. Pearson correlation; 3. both Spearman's rank correlation and Pearson correlation
		:param background_query: indicator of whether to use background peaks which match the candidate peak by GC content and average accessibility to compute peak-gene correlation
		:param verbose: verbosity level to print the intermediate information
		:return: (tuplet) 1. the potential target gene name; 
						  2. (dataframe) peak-gene link annotations with computed peak accessibility-gene expression correlations, which may include p-values;
						  3. indice of the potential target gene;
		"""

		# print('gene name ',gene_query)
		try:
			if verbose>0:
				print('gene: %s, candidate peak number: %d '%(gene_query,len(peak_loc)),peak_loc[0:2])
		except Exception as error:
			print('error! ',error)
			flag = -1
			return flag
			# return (gene_query,[],gene_id_query)
		warnings.filterwarnings('ignore')

		# compute correlations
		flag = 0
		if flag_corr_>0:
			try:
				if method_type==0:
					if type(atac_read) is sc.AnnData:
						X = atac_read[:, peak_loc].X.toarray().T
					else:
						X = atac_read.loc[:, peak_loc].T
				else:
					if type(atac_read) is sc.AnnData:
						sample_id = atac_read.obs_names
						X = pd.DataFrame(index=sample_id,columns=peak_loc,data=atac_read[:, peak_loc].X.toarray())
					else:
						X = atac_read.loc[:, peak_loc]

				if type_id_1 in [1,3]:
					df = pd.DataFrame(index=peak_loc, columns=['spearmanr'])
					if method_type==0:
						spearman_cors = pd.Series(np.ravel(pairwise_distances(X,
										rna_exprs[gene_query].T.values.reshape(1, -1),
										metric=spearman_corr, n_jobs=-1)),
										index=peak_loc)
						df['spearmanr'] = spearman_cors
					else:
						spearman_cors, spearman_pvals = utility_1.test_correlation_pvalues_pair(X,rna_exprs.loc[:,[gene_query]],correlation_type='spearmanr',float_precision=-1)
						spearman_cors, spearman_pvals = spearman_cors[gene_query], spearman_pvals[gene_query]
						df['spearmanr'] = spearman_cors
						df['pval1_ori'] = spearman_pvals

				if type_id_1 in [2,3]:
					if type_id_1==2:
						df = pd.DataFrame(index=peak_loc, columns=['pearsonr'])
					if method_type==0:
						pearson_cors = pd.Series(np.ravel(pairwise_distances(X,
										rna_exprs[gene_query].T.values.reshape(1, -1),
										metric=pearson_corr, n_jobs=-1)),
										index=peak_loc)
						df['pearsonr'] = pearson_cors
					else:
						pearson_cors, pearson_pvals = utility_1.test_correlation_pvalues_pair(X,rna_exprs.loc[:,[gene_query]],correlation_type='spearmanr',float_precision=-1)
						pearson_cors, pearson_pvals = pearson_cors[gene_query], pearson_pvals[gene_query]
						df['pearsonr'] = pearson_cors
						df['pval2_ori'] = pearson_pvals

			except Exception as error:
				print('error!')
				print(error)
				flag = 1
				return

			if len(spearman_cors)==0:
				print('spearman_cors length zero ')
				flag = 1

		# compute peak accessibility-gene expression correlation for random background peak loci which match the candidate peak loci by GC content and average accessibility
		if flag_corr_ in [0,2]:
			flag_query1 = 1
			# if flag_query1>0:
			try:
				colnames = self.peak_bg.columns
				if flag_corr_==0:
					df = df_query.loc[gene_query,:]
					df.index = np.asarray(df['peak_id'])
					# print('background peak query ',df.shape[0])
					print('candidate peaks of gene %s: %d'%(gene_query,df.shape[0]))
				
				for p in df.index:
					id1 = np.int64(self.peak_bg.loc[p,:]-1)
					rand_peaks = self.atac_meta_peak_loc[id1]
					# try:
					#   rand_peaks = np.random.choice(atac_ad.var_names[(atac_ad.var['GC_bin'] == atac_ad.var['GC_bin'][p])  &\
					#                                               (atac_ad.var['counts_bin'] == atac_ad.var['counts_bin'][p])], 100, False)
					# except:
					#   rand_peaks = np.random.choice(atac_ad.var_names[(atac_ad.var['GC_bin'] == atac_ad.var['GC_bin'][p])  &\
					#                                               (atac_ad.var['counts_bin'] == atac_ad.var['counts_bin'][p])], 100, True)
					
					if type(atac_read) is sc.AnnData:
						X = atac_read[:, rand_peaks].X.toarray().T
					else:
						X = atac_read.loc[:, rand_peaks].T

					# type_id_1: 1: estimate spearmanr; 3: estimate spearmanr and pearsonr
					if type_id_1 in [1,3]:
						column_id1, column_id2 = 'spearmanr','pval1'
						rand_cors_spearman = pd.Series(np.ravel(pairwise_distances(X,
														rna_exprs[gene_query].T.values.reshape(1, -1),
														metric=spearman_corr, n_jobs=-1)),
														index=rand_peaks)

						m1, v1 = np.mean(rand_cors_spearman), np.std(rand_cors_spearman)
						spearmanr_1 = df.loc[p,column_id1]
						pvalue1 = 1 - norm.cdf(spearmanr_1, m1, v1) # estimate the empirical p-value

						if (spearmanr_1<0) and (pvalue1>0.5):
							pvalue1 = 1-pvalue1
						df.loc[p,column_id2]= pvalue1

					# type_id_1: 2: estimate pearsonr; 3: estimate spearmanr and pearsonr
					if type_id_1 in [2,3]:
						column_id1, column_id2 = 'pearsonr','pval2'
						rand_cors_pearson = pd.Series(np.ravel(pairwise_distances(X,
														rna_exprs[gene_query].T.values.reshape(1, -1),
														metric=pearson_corr, n_jobs=-1)),
														index=rand_peaks)

						m2, v2 = np.mean(rand_cors_pearson), np.std(rand_cors_pearson)
						pearsonr_1 = df.loc[p,column_id1]
						pvalue2 = 1 - norm.cdf(pearsonr_1, m2, v2)

						if (pearsonr_1<0) and (pvalue2>0.5):
							pvalue2 = 1-pvalue2
						df.loc[p,column_id2]= pvalue2

			except Exception as error:
				print('error!')
				print(error)
				flag = 1

		warnings.filterwarnings('default')

		if flag==1:
			return flag

		return (gene_query, df, gene_id_query)

	## ====================================================
	# gene-peak association query: load and combine previously estimated peak accessibility-gene expression correlations
	def test_gene_peak_query_combine_1(self,save_filename_list=[],verbose=0,select_config={}):

		"""
		load and combine previously computed peak accessibility-gene expression correlations and p-values
		:param save_filename_list: (list) paths of files which saved the pre-computed peak accessibility-gene expression correlations and p-values
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) the combined peak-gene link annotations with computed peak accessibility-gene expression correlations and p-values
		"""

		file_num1 = len(save_filename_list)
		list_1 = []
		if file_num1>0:
			for i1 in range(file_num1):
				input_filename = save_filename_list[i1]
				if os.path.exists(input_filename)==False:
					df_gene_peak_query = []
					print('the file does not exist: %s'%(input_filename))
					return df_gene_peak_query
				t_data1 = np.load(input_filename,allow_pickle=True)
				gene_peak_local = t_data1[()]
				gene_query_vec_1 = list(gene_peak_local.keys())
				
				# load previously estimated peak-gene correlations
				df_gene_peak_query_1 = self.test_gene_peak_query_load_unit(gene_query_vec=gene_query_vec_1,
																			gene_peak_annot=gene_peak_local,
																			df_gene_peak_query=[],
																			field_query=[],
																			verbose=verbose,
																			select_config=select_config)
				list_1.append(df_gene_peak_query_1)		
				print('peak-gene links, dataframe of size ',df_gene_peak_query_1.shape,i1)
				print('gene number: %d'%(len(gene_query_vec_1)))

			if file_num1>1:
				df_gene_peak_query = pd.concat(list_1,axis=0,join='outer',ignore_index=False)
			else:
				df_gene_peak_query = list_1[0]

			return df_gene_peak_query

	## ====================================================
	# query peak accessibility-gene expression correlations for peak-gene links associated with the specific target genes
	def test_gene_peak_query_load_unit(self,gene_query_vec,gene_peak_annot,df_gene_peak_query=[],field_query=[],verbose=1,select_config={}):
		
		"""
		query peak accessibility-gene expression correlations for peak-gene links associated with the specific target genes
		:param gene_query_vec: (array or list) the target gene set
		:param gene_peak_annot: (dataframe) peak-gene link annotations with pre-computed peak accessibility-gene expression correlations
		:param df_gene_peak_query: (dataframe) annotations of candidate peak-gene links
		:param field_query: (array or list) the columns to copy from gene_peak_annot to df_gene_peak_query
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of candidate peak-gene links including peak accessibility-gene expression correlations
		"""

		gene_query_num = len(gene_query_vec)
		flag1 = (len(df_gene_peak_query)>0)
		list1 = []
		gene_query_idvec = []
		if flag1>0:
			gene_query_idvec = df_gene_peak_query.index

		for i1 in range(gene_query_num):
			gene_query_id = gene_query_vec[i1]
			df = gene_peak_annot[gene_query_id]	# retrieve the peak accessibility-gene expression correlations
			if (verbose>0) and (i1%100==0):
				print('gene_query_id: ',gene_query_id,df.shape)

			if len(field_query)==0:
				field_query = df.columns
			if (gene_query_id in gene_query_idvec):
				peak_local = df_gene_peak_query.loc[gene_query_id] # peak-gene links associated with the given potential target gene
				id1_pre = peak_local.index.copy()
				peak_local.index = np.asarray(peak_local['peak_id'])
				peak_loc_1 = df.index  # peak loci included in the peak-gene links with computed peak accessibility-gene expression correlations
				peak_loc_pre = peak_local.index.intersection(peak_loc_1,sort=False)
				peak_local.loc[peak_loc_pre,field_query] = df.loc[peak_loc_pre,field_query]
				peak_local.index = id1_pre  # reset the index
				df_gene_peak_query.loc[gene_query_id,field_query] = peak_local.loc[:,field_query]
			else:
				df = df.loc[:,field_query]
				df['peak_id'] = np.array(df.index)
				df.index = [gene_query_id]*df.shape[0]
				df['gene_id'] = [gene_query_id]*df.shape[0]
				if flag1>0:
					df_gene_peak_query = pd.concat([df_gene_peak_query,df],axis=0,join='outer',ignore_index=False)
				else:
					list1.append(df)

		if len(list1)>0:
			df_gene_peak_query = pd.concat(list1,axis=0,join='outer',ignore_index=False)

		# verbose_internal = self.verbose_internal
		# if verbose_internal==2:
		# 	print('peak-gene links, dataframe of size ', df_gene_peak_query.shape)
		# 	print('data preview:\n',df_gene_peak_query[0:2])

		return df_gene_peak_query

	## ====================================================
	# select candidate peak-gene links using thresholds on peak accessibility-gene expression correlation and p-value
	def test_gene_peak_query_correlation_thresh_pre1(self,df_gene_peak_query=[],column_idvec=['peak_id','gene_id'],column_vec_query=[],thresh_corr_distance=[],verbose=0,select_config={}):

		"""
		select candidate peak-gene links by thresholds on peak accessibility-gene expression correlation and empirical p-value;
		thresholds are adjusted for different peak-gene distance ranges;
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param column_vec_query: (array or list) columns which will be added or updated in the peak-gene link annotation dataframe, 
												 including the column representing which peak-gene links are selected as candidate links
		:param thresh_corr_distance: (list of list) thresholds on peak accessibility-gene expression correlation and empirical p-value for different peak-gene distance ranges
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) annotations of selected candidate peak-gene links
				 2. (dataframe) updated annotations of the original peak-gene links, including labels for the selected candidate links 
		"""

		df_gene_peak_query.index = utility_1.test_query_index(df_gene_peak_query,column_vec=column_idvec)
		query_idvec = df_gene_peak_query.index
		column_id1, column_id2 = column_idvec
		if len(column_vec_query)==0:
			column_vec_query = ['label_thresh2']

		column_label_1 = column_vec_query[0] # column representing which peak-gene links are selected as candidate links

		column_distance = select_config['column_distance'] # column_distance: 'distance' (peak-gene distance)
		if not (column_distance in df_gene_peak_query):
			field_query = ['distance']
			# column_name = [column_distance]
			column_idvec_query = ['peak_id','gene_id']
			df_peak_annot = self.df_gene_peak_distance  # genome-wide peak-gene links with peak-gene distance computed (peaks within +/-2Mb of gene TSS)
			# query peak-gene link attributes
			df_gene_peak_query = self.test_gene_peak_query_attribute_1(df_gene_peak_query=df_gene_peak_query,
																			df_gene_peak_query_ref=df_peak_annot,
																			column_idvec=column_idvec_query,
																			field_query=field_query,
																			column_name=[],
																			reset_index=False,
																			select_config=select_config)

		distance_abs = df_gene_peak_query[column_distance].abs()

		column_vec_1 = select_config['column_correlation'] # column_vec_1:['spearmanr','pval1'] (peak accessibility-gene expr correlation and empirical p-value)
		column_1, column_2 = column_vec_1[0:2]
		column_2_ori = column_vec_1[2]

		peak_corr, peak_pval = df_gene_peak_query[column_1], df_gene_peak_query[column_2]
		id1 = (pd.isna(peak_pval)==True)
		df1 = df_gene_peak_query.loc[id1]
		print('peak-gene links without estimated empirical p-values of peak accessibility-gene expression correlations: %d'%(np.sum(id1)))
		
		# to update
		# peak_pval[id1] = df_gene_peak_query.loc[id1,column_2_ori] # use the raw p-values for peak-gene link selection
		
		list1 = []
		for thresh_vec in thresh_corr_distance:
			constrain_1, constrain_2, thresh_peak_corr_vec = thresh_vec
			id1 = (distance_abs<constrain_2)&(distance_abs>=constrain_1) # constraint by distance
			
			df_gene_peak_query_sub1 = df_gene_peak_query.loc[id1]
			peak_corr, peak_pval = df_gene_peak_query_sub1[column_1], df_gene_peak_query_sub1[column_2]
			peak_sel_corr_pos = (peak_corr>=1)
			peak_sel_corr_neg = (peak_corr<=-1)

			# constraint by correlation value
			for thresh_peak_corr_vec_1 in thresh_peak_corr_vec:
				thresh_peak_corr_pos, thresh_peak_corr_pval_pos, thresh_peak_corr_neg, thresh_peak_corr_pval_neg = thresh_peak_corr_vec_1
				
				peak_sel_corr_pos = ((peak_corr>thresh_peak_corr_pos)&(peak_pval<thresh_peak_corr_pval_pos))|peak_sel_corr_pos
				peak_sel_corr_neg = ((peak_corr<thresh_peak_corr_neg)&(peak_pval<thresh_peak_corr_pval_neg))|peak_sel_corr_neg
				if verbose>0:
					print('distance_thresh1:%d, distance_thresh2:%d'%(constrain_1,constrain_2))
					print('thresh_sel_corr_pos, thresh_peak_corr_pval_pos, thresh_sel_corr_neg, thresh_peak_corr_pval_neg ', thresh_peak_corr_pos, thresh_peak_corr_pval_pos, thresh_peak_corr_neg, thresh_peak_corr_pval_neg)
					print('peak_sel_corr_pos, peak_sel_corr_neg ', np.sum(peak_sel_corr_pos), np.sum(peak_sel_corr_neg))

			peak_sel_corr_ = (peak_sel_corr_pos|peak_sel_corr_neg)
			peak_sel_corr_num1 = np.sum(peak_sel_corr_)
			if verbose>0:
				print('peak_sel_corr_num ', peak_sel_corr_num1)

			query_id1 = df_gene_peak_query_sub1.index
			query_id2 = query_id1[peak_sel_corr_]
			list1.extend(query_id2)
		
		peak_corr, peak_pval = df_gene_peak_query[column_1], df_gene_peak_query[column_2]
		
		distance_abs = df_gene_peak_query['distance'].abs()
		df_gene_peak_query['distance_abs'] = distance_abs
		peak_distance_thresh_1 = 500
		
		if 'thresh_corr_retain' in select_config:
			thresh_corr_retain = np.asarray(select_config['thresh_corr_retain'])
			if thresh_corr_retain.ndim==2:
				for (thresh_corr_1, thresh_pval_1) in thresh_corr_retain:
					id1 = (peak_corr.abs()>thresh_corr_1)
					print('threshold on correlation: ',thresh_corr_1)
					print('selected peak-gene links by the threshold: %d'%(np.sum(id1)))
					if (thresh_pval_1<1):
						id2 = (peak_pval<thresh_pval_1) # p-value threshold
						id3 = (distance_abs<peak_distance_thresh_1) # distance threshold; only use correlation threshold within specific distance
						id1_1 = id1&(id2|id3)
						id1_2 = id1&(~id2)
						id1 = id1_1
						print('threshold on correlation and p-value: ',np.sum(id1),thresh_corr_1,thresh_pval_1)
						df1 = df_gene_peak_query.loc[id1_2,:]
						print('peak-gene links with correlation above threshold but p-value higher than threshold, dataframe of size ',df1.shape)
						print(df1)
					query_id3 = query_idvec[id1] # retain gene-peak query with high peak accessibility-gene expression correlation
					df2 = df_gene_peak_query.loc[query_id3,:]
					
					filename_save_thresh2 = select_config['filename_save_thresh2']
					b = filename_save_thresh2.find('.txt')
					output_filename = filename_save_thresh2[0:b]+'.%s.2.txt'%(thresh_corr_1)
					df2.index = np.asarray(df2['gene_id'])
					df2 = df2.sort_values(by=['gene_id','distance'],ascending=[True,True])
					df2.to_csv(output_filename,sep='\t')

					# output_filename = filename_save_thresh2[0:b]+'.%s.2.sort1.txt'%(thresh_corr_1)
					# df2 = df2.sort_values(by=[column_1,'distance_abs'],ascending=[False,True])
					# df2.to_csv(output_filename,sep='\t')

					# output_filename = filename_save_thresh2[0:b]+'.%s.2.sort2.txt'%(thresh_corr_1)
					# df2 = df2.sort_values(by=['peak_id',column_1,'distance_abs'],ascending=[True,False,True])
					# df2.to_csv(output_filename,sep='\t')
					list1.extend(query_id3)
			else:
				thresh_corr_1, thresh_pval_1 = thresh_corr_retain
				id1 = (peak_corr.abs()>thresh_corr_1)
				print('thresh correlation: ',np.sum(id1),thresh_corr_1)
				if (thresh_pval_1<1):
					id2 = (peak_pval<thresh_pval_1)
					id3 = (distance_abs<peak_distance_thresh_1)
					id1 = id1&(id2|id3)
					sel_num1 = np.sum(id1)
					print('thresholds on correlation and p-value: ',thresh_corr_1,thresh_pval_1)
					print('selected peak-gene links by the thresholds: %d'%(sel_num1))
				
				query_id3 = query_idvec[id1] # retain gene-peak query with high peak accessibility-gene expression correlation
				df2 = df_gene_peak_query.loc[query_id3,:]
				
				filename_save_thresh2 = select_config['filename_save_thresh2']
				b = filename_save_thresh2.find('.txt')
				output_filename = filename_save_thresh2[0:b]+'.%s.2.txt'%(thresh_corr_1)
				df2.index = np.asarray(df2['gene_id'])
				df2 = df2.sort_values(by=['gene_id','distance'],ascending=[True,True])
				df2.to_csv(output_filename,sep='\t')

				# output_filename = filename_save_thresh2[0:b]+'.%s.2.sort1.txt'%(thresh_corr_1)
				# df2 = df2.sort_values(by=[column_1,'distance_abs'],ascending=[False,True])
				# df2.to_csv(output_filename,sep='\t')

				# output_filename = filename_save_thresh2[0:b]+'.%s.2.sort2.txt'%(thresh_corr_1)
				# df2 = df2.sort_values(by=['peak_id',column_1,'distance_abs'],ascending=[True,False,True])
				# df2.to_csv(output_filename,sep='\t')
				list1.extend(query_id3)

		query_id_sub1 = pd.Index(list1).unique()
		t_columns = df_gene_peak_query.columns.difference(['distance_abs'],sort=False)
		df_gene_peak_query = df_gene_peak_query.loc[:,t_columns]
		
		df_gene_peak_query_ori = df_gene_peak_query.copy()
		df_gene_peak_query = df_gene_peak_query_ori.loc[query_id_sub1]
		
		df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])
		df_gene_peak_query_ori.loc[query_id_sub1,column_label_1] = 1   # annotate selected candidate peak-gene links
		df_gene_peak_query_ori.index = np.asarray(df_gene_peak_query_ori[column_id1])
		id_query = (~pd.isna(df_gene_peak_query_ori[column_label_1]))
		df_gene_peak_query_ori.loc[id_query,column_label_1] = df_gene_peak_query_ori.loc[id_query,column_label_1].astype(int)
		
		column_id_1 = select_config['column_highly_variable'] # column representing highly variable genes
		if column_id_1 in df_gene_peak_query.columns:
			column_vec_1 = [column_id_1,column_id1,column_distance]
			df_gene_peak_query = df_gene_peak_query.sort_values(by=column_vec_1,ascending=[False,True,True])
		else:
			column_vec_1 = [column_id1,column_distance]
			df_gene_peak_query = df_gene_peak_query.sort_values(by=column_vec_1,ascending=[True,True])

		return df_gene_peak_query, df_gene_peak_query_ori

	## ====================================================
	# select candidate peak-gene links using thresholds on the peak accessibility-gene expression correlation and p-value
	def test_gene_peak_query_correlation_pre1_select_1(self,gene_query_vec=[],df_gene_peak_query=[],peak_loc_query=[],atac_ad=[],rna_exprs=[],input_filename='',
															index_col=0,highly_variable=False,peak_distance_thresh=2000,
															save_mode=1,save_file_path='',output_filename_1='',output_filename_2='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		select candidate peak-gene links using thresholds on the peak accessibility-gene expression correlation and p-value
		:param gene_query_vec: (array or list) the target genes
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param input_filename: path of the file which saved annotations of peak-gene links
		:param index_col: column to use as the row label for data in the dataframe
		:param highly_variable: indicator of whether to only include highly variable genes as the target genes
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param output_filename_2: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) selected candidate peak-gene links using thresholds on peak accessibility-gene expression correlation and p-value;
				 2. (dataframe) the original peak-gene links from which to select the candidate links
		"""

		# peak-gene link pre-selection
		# pre-selection 2: select based on emprical p-value
		flag_select_thresh2=1
		if flag_select_thresh2>0:
			thresh_corr_1, thresh_pval_1 = 0.01,0.05
			thresh_corr_2, thresh_pval_2 = 0.1,0.1
			
			## thresh 2
			if 'thresh_corr_distance_2' in select_config:
				thresh_corr_distance = select_config['thresh_corr_distance_2']
			else:
				# thresh_distance_1 = 50
				thresh_distance_1 = 100
				if 'thresh_distance_default_2' in select_config:
					thresh_distance_1 = select_config['thresh_distance_default_2'] # the distance threshold with which we retain the peaks without thresholds of correlation and p-value

				thresh_corr_distance = [[0,thresh_distance_1,[[0,1,0,1]]],
										[thresh_distance_1,500,[[0.01,0.1,-0.01,0.1],[0.15,0.15,-0.15,0.15]]],
										[500,1000,[[0.1,0.1,-0.1,0.1]]],
										[1000,2050,[[0.15,0.1,-0.15,0.1]]]]

			print('thresh_corr_distance')
			print(thresh_corr_distance)

			start = time.time()
			if len(df_gene_peak_query)==0:
				df_gene_peak_query = pd.read_csv(input_filename,index_col=index_col,sep='\t')
				column_id1 = 'gene_id'
				if not (column_id1 in df_gene_peak_query.columns):
					df_gene_peak_query[column_id1] = np.asarray(df_gene_peak_query.index)
				else:
					# df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
					df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])

			# gene-peak association selection by thresholds
			# select candidate peak-gene links using thresholds on the peak accessibility-gene expression correlations and empirical p-values
			df_gene_peak_query_pre1, df_gene_peak_query = self.test_gene_peak_query_correlation_thresh_pre1(df_gene_peak_query=df_gene_peak_query,
																											thresh_corr_distance=thresh_corr_distance,
																											verbose=verbose,
																											select_config=select_config)
			
			print('original peak-gene links, peak-gene links after selection by correlation and p-value thresholds ',df_gene_peak_query.shape,df_gene_peak_query_pre1.shape)
			
			stop = time.time()
			print('the pre-selection used %.5fs'%(stop-start))

			if (save_mode>0):
				float_format = '%.5E'
				if (output_filename_2!=''):
					df_gene_peak_query_pre1.index = np.asarray(df_gene_peak_query_pre1['gene_id'])
					df_gene_peak_query_pre1.to_csv(output_filename_2,sep='\t',float_format=float_format)

				if (output_filename_1!=''):
					df_gene_peak_query.index = np.asarray(df_gene_peak_query['gene_id'])
					df_gene_peak_query.to_csv(output_filename_1,sep='\t',float_format=float_format)

			return df_gene_peak_query_pre1, df_gene_peak_query

	## ====================================================
	# combine peak accessibility-gene expression correlations of different subsets of peak-gene links
	def test_query_feature_correlation_merge_1(self,df_gene_peak_query=[],filename_list=[],flag_combine=1,compute_mode=-1,index_col=0,save_mode=0,output_path='',output_filename='',verbose=0,select_config={}):

		"""
		combine peak accessibility-gene expression correlations of different subsets of peak-gene links
		:param df_gene_peak_query: (dataframe) annotations of peak-gene links
		:param filename_list: (list) paths of files containing peak accessibility-gene expression correlations for different subsets of peak-gene links
		:param flag_combine: indicator of whether to combine the computed peak accessibility-gene expression correlations of different subsets of peak-gene links
		:param compute_mode: the type of peak-gene links: 1. links between candidate peaks and target genes; 2. links between background peaks and genes;
		:param index_col: column to use as the row label for data in the dataframe
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) concatenated peak-gene link annotations including peak accessibility-gene expression correlations
		"""

		df_gene_peak_query_1 = []
		if flag_combine>0:
			input_filename_list1 = filename_list
			if len(filename_list)==0:
				field_query_vec = ['filename_list_pre1','filename_list_bg']
				id1 = int((compute_mode-1)/2)
				field_query = field_query_vec[id1]
				if field_query in select_config:
					input_filename_list1 = select_config[field_query]
					print('load estimations from the file ',len(input_filename_list1))
					if verbose>0:
						print(input_filename_list1[0:2])

			if len(input_filename_list1)>0:
				# combine data from different files
				df_gene_peak_query_1 = utility_1.test_file_merge_1(input_filename_list1,index_col=index_col,header=0,float_format=-1,
																	save_mode=1,output_filename=output_filename,verbose=verbose)

				if (save_mode>0) and (output_filename!=''):
					df_gene_peak_query_1.to_csv(output_filename,sep='\t')

					save_mode_query = self.save_mode
					file_num1 = len(input_filename_list)
					if (save_mode_query==1):
						# without saving the intermediate files
						for i1 in range(file_num1):
							filename_query = input_filename_list1[i1]
							try:
								os.remove(filename_query)
							except Exception as error:
								print('error! ',error)

			else:
				print('please perform peak accessibility-gene expression correlation estimation or load estimated correlations')

		return df_gene_peak_query_1

	## ====================================================
	# query empirical p-values of peak accessibility-gene expression correlations for candidate peak-gene links
	def test_query_feature_correlation_merge_2(self,df_gene_peak_query=[],df_gene_peak_bg=[],filename_list=[],column_idvec=['peak_id','gene_id'],column_vec_query=[],
													flag_combine=1,coherent_mode=1,index_col=0,save_mode=0,output_file_path='',output_filename='',verbose=0,select_config={}):

		"""
		query empirical p-values of peak accessibility-gene expression correlations for candidate peak-gene links
		:param df_gene_peak_query: (dataframe) annotations of links between each target gene and the associated candidate peaks (candidate peak-gene links)
		:param df_gene_peak_bg: (dataframe) annotations of candidate peak-gene links with background peaks sampled for the candidate peaks 
											to compute the empirical p-values of peak-gene correlations
		:param filename_list: (list) paths of two types of files containing candidate peak-gene link annotations: 
							  1: with raw p-values of peak-gene correlations only; 
							  2: with empirical p-values computed using background peak loci;
		:param column_idvec: (array or list) columns representing peak and gene indices in the peak-gene link annotation dataframe
		:param column_vec_query: (array or list) columns representing the annotations to copy from the other dataframes to the specific dataframe
		:param flag_combine: indicator of whether to query empirical p-values of peak accessibility-gene expression correlations for candidate peak-gene links
		:param coherent_mode: indicator of how to combine the data from the files of candidate peak-gene link annotations with and without the empirical p-values of peak-gene correlations
		:param index_col: column to use as the row label for data in the dataframe
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: (dataframe) annotations of candidate peak-gene links including peak accessibility-gene expression correlations 
							 and the raw p-values and empirical p-values	
		"""

		if flag_combine>0:
			load_mode = (len(df_gene_peak_query)>0)&(len(df_gene_peak_bg)>0)

			if len(column_vec_query)==0:
				column_vec = ['pval1']
			else:
				column_vec = column_vec_query

			print('load_mode ',load_mode)
			
			if load_mode>0:
				df_list = [df_gene_peak_query,df_gene_peak_bg]
				column_vec_1 = [column_vec]
				# copy specified columns from the second dataframe to the first dataframe
				df_gene_peak_query_1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec,df_list=df_list,
																		column_vec=column_vec_1,reset_index=False)
			else:
				if coherent_mode==1:
					if len(filename_list)>0:
						input_filename_pre2, input_filename_pre2_bg = filename_list[0:2]
					else:
						input_filename_pre2 = select_config['input_filename_pre2']
						input_filename_pre2_bg = select_config['input_filename_pre2_bg']

					if os.path.exists(input_filename_pre2)==False:
						print('the file does not exist: %s'%(input_filename_pre2))
						return

					if os.path.exists(input_filename_pre2_bg)==False:
						print('the file does not exist: %s'%(input_filename_pre2_bg))
						return

					input_filename_list = [input_filename_pre2,input_filename_pre2_bg]
					
					column_vec_1 = [column_vec]
					# copy specified columns from the other dataframe to the first dataframe
					# df_gene_peak_query_1 = utility_1.test_column_query_1(input_filename_list,id_column=column_idvec,df_list=[],
					# 														column_vec=column_vec_1,reset_index=False)

				elif coherent_mode==0:
					if len(filename_list)==0:
						input_filename_pre2 = select_config['input_filename_pre2']
						filename_list_bg = select_config['filename_list_bg']
						input_filename_list = [input_filename_pre2]+filename_list_bg
					else:
						input_filename_pre2 = filename_list[0]
						filename_list_bg = filename_list[1:]
						input_filename_list = filename_list
					
					file_num1 = len(filename_list_bg)
					column_vec_1 = [column_vec]*file_num1

				elif coherent_mode==2:
					filename_list_pre2 = select_config['filename_list_pre2']
					filename_list_bg = select_config['filename_list_bg']

					print('filename_list_pre2')
					print(filename_list_pre2)

					print('filename_list_bg')
					print(filename_list_bg)
					
					file_num1 = len(filename_list_pre2)
					column_vec_1 = [column_vec]
					list_query1 = []
					for i1 in range(file_num1):
						filename_1 = filename_list_pre2[i1]
						filename_2 = filename_list_bg[i1]
						input_filename_list1 = [filename_1,filename_2]
						# copy specified columns from the other dataframes to the first dataframe
						df_gene_peak_query = utility_1.test_column_query_1(input_filename_list1,id_column=column_idvec,df_list=[],
																			column_vec=column_vec_1,reset_index=False)
						list_query1.append(df_gene_peak_query)

					df_gene_peak_query_1 = pd.concat(list_query1,axis=0,join='outer',ignore_index=False)

				if coherent_mode in [0,1]:
					# copy specified columns from the other dataframes to the first dataframe
					df_gene_peak_query_1 = utility_1.test_column_query_1(input_filename_list,id_column=column_idvec,df_list=[],
																			column_vec=column_vec_1,reset_index=False)

			if save_mode>0:
				df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1['gene_id'])
				float_format = '%.6E'
				df_gene_peak_query_1.to_csv(output_filename,sep='\t',float_format=float_format)

			return df_gene_peak_query_1

	## ====================================================
	# query paths of the files saved for computed peak accessibility-gene expression correlations
	def test_feature_link_query_correlation_file_pre1(self,type_id_query=0,input_file_path='',select_config={}):

		"""
		query paths of the files saved for computed peak accessiblilty-gene expression correlations
		:param input_file_path: the directory to retrieve data from
		:param select_config: dictionary containing parameters
		:return: 1. (list) paths of the files saved for computed peak accessiblilty-gene expression correlations
				 2. dictionary containing updated parameters, including the list of paths of the files containing peak accessiblilty-gene expression correlations
		"""

		filename_annot_vec = select_config['filename_annot_local']
		filename_prefix = select_config['filename_prefix_save_local']
		query_num1 = len(filename_annot_vec)
		input_filename_list = []
		for i1 in range(query_num1):
			filename_annot = filename_annot_vec[i1]
			input_filename_1 = '%s/%s.combine.%s.txt'%(input_file_path,filename_prefix,filename_annot) # highly variable genes; original p-value
			input_filename_2 = '%s/%s.combine.thresh1.%s.txt'%(input_file_path,filename_prefix,filename_annot)	# empirical p-value for the subset of gene-peak link query pre-selected with thresholds
			input_filename_list.extend([input_filename_1,input_filename_2])

		select_config.update({'filename_list_combine_1':input_filename_list})

		return input_filename_list, select_config

	## ====================================================
	# query estimated empirical p-values of peak accessibility-gene expression correlations of peak-gene links
	def test_gene_peak_query_correlation_pre1_combine(self,gene_query_vec=[],peak_distance_thresh=500,df_gene_peak_distance=[],df_peak_query=[],
														peak_loc_query=[],atac_ad=[],rna_exprs=[],highly_variable=False,highly_variable_thresh=0.5,
														load_mode=1,input_file_path='',save_mode=1,save_file_path='',output_filename='',filename_prefix_save='',
														verbose=0,select_config={}):

		"""
	 	query estimated empirical p-values of peak accessibility-gene expression correlations of peak-gene links
		:param gene_query_vec: (array or list) the target genes
		:param peak_distance_thresh: threshold on peak-gene TSS distance to search for peak-gene links
		:param df_gene_peak_distance: (dataframe) annotations of genome-wide peak-gene pairs including peak-gene TSS distances
		:param df_peak_query: (dataframe) peak loci attributes
		:param peak_loc_query: (array) the ATAC-seq peak loci for which to estimate peak-gene links
		:param atac_ad: (AnnData object) ATAC-seq data of the metacells
		:param rna_exprs: (dataframe) gene expressions of the metacells
		:param highly_variable: indicator of whether to only include highly variable genes as the target genes
		:param highly_variable_thresh: threshold on the normalized dispersion of gene expression to identify highly variable genes
		:param load_mode: indicator of how to load the pre-computed peak accessibility-gene expression correlations:
						  1. peak-gene correlations were computed for different subsets (batches) of a given set of peak-gene links and saved in a series of files;
						  2. peak-gene correlations of the given set of peak-gene links were saved in one file;
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param save_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) annotations of genome-wide peak-gene pairs including peak-gene correlations; the raw and empirical p-values may be included for a subset of links;
				 2. (dataframe) annotations of pre-selected peak-gene links including peak-gene correlations, raw p-values, and empirical p-values;
		"""

		# combine original p-value and estimated empirical p-value for different gene groups: highly-variable and not highly-variable
		flag_combine_1=1
		if flag_combine_1>0:
			if len(df_gene_peak_distance)==0:
				df_gene_peak_distance = self.df_gene_peak_distance
			df_pre1 = df_gene_peak_distance
			column_idvec = ['peak_id','gene_id']
			column_id1, column_id2 = column_idvec
			# df_pre1.index = ['%s.%s'%(peak_id,gene_id) for (peak_id,gene_id) in zip(df_pre1['peak_id'],df_pre1['gene_id'])]
			df_pre1.index = utility_1.test_query_index(df_pre1,column_vec=column_idvec)
			input_filename = output_filename
			input_filename_pre1 = input_filename
			if (load_mode!=1):
				if not ('filename_list_combine_1' in select_config):
					# query filename of files saved for computed peak accessibility-gene expression correlations
					input_filename_list, select_config = self.test_feature_link_query_correlation_file_pre1(input_file_path=input_file_path,select_config=select_config)
				else:
					input_filename_list = select_config['filename_list_combine_1']
			else:
				input_filename_1 = select_config['input_filename_pre1']  # file containing dataframe with peak-gene correlations of genome-wide peak-gene links
				input_filename_2 = select_config['input_filename_pre2']  # file containing dataframe with peak-gene correlations of pre-selected peak-gene links

				if (os.path.exists(input_filename_1)==True) and (os.path.exists(input_filename_2)==True):
					input_filename_list = [input_filename_1,input_filename_2]
			
			flag_query1 = 1
			if flag_query1>0:
				list1,list2 = [],[]
				query_num1 = len(input_filename_list)
				column_correlation = select_config['column_correlation']
				column_1, column_2, column_2_ori = column_correlation[0:3]
				
				field_query_1 = [column_1,column_2_ori]  # columns representing peak-gene correlation and the raw p-value
				field_query_2 = [column_2]	# column representing the empirical p-value of peak-gene correlation estimated using background peaks
				column_query_1 = [field_query_1,field_query_2]
				
				type_id_1 = 2 # type_id_1:0, use new index; (1,2) use the present index
				type_id_2 = 0 # type_id_2:0, load dataframe from df_list; 1, load dataframe from input_filename_list
				
				reset_index = False
				interval = 2
				group_num = int(query_num1/interval)
				for i1 in range(group_num):
					id1 = (interval*i1)
					df_list2 = []
					for i2 in range(interval):
						df1 = pd.read_csv(input_filename_list[id1+i2],sep='\t')
						df1.index = utility_1.test_query_index(df1,column_vec=column_idvec)
						df_list2.append(df1)

					df_list = [df_pre1]+df_list2
					# copy specified columns from the other dataframes to the first dataframe
					df_pre1 = utility_1.test_column_query_1(input_filename_list=[],id_column=column_idvec,
															column_vec=column_query_1,
															df_list=df_list,
															type_id_1=type_id_1,
															type_id_2=type_id_2,
															reset_index=reset_index,select_config=select_config)

				df_gene_peak_query_compute1 = df_pre1
				field_id1 = field_query_2[0]
				
				id1 = (~df_pre1[field_id1].isna())
				df_gene_peak_query = df_pre1.loc[id1,:]

				df_gene_peak_query_compute1.index = np.asarray(df_gene_peak_query_compute1[column_id1])
				df_gene_peak_query.index = np.asarray(df_gene_peak_query[column_id1])
				if (save_mode>0) and (output_filename!=''):
					df_gene_peak_query_compute1.to_csv(output_filename,sep='\t')

			self.df_gene_peak_distance = df_gene_peak_query_compute1  # update columns of df_gene_peak_distance: add column representing empirical p-value for a subset of peak-gene links
			select_config.update({'filename_gene_peak_annot':output_filename})
			
			self.select_config = select_config
			if verbose>0:
				print('genome-wide peak-gene links, dataframe of size ',df_gene_peak_query_compute1.shape)
				print('candidate peak-gene links, dataframe of size ',df_gene_peak_query.shape)

			return df_gene_peak_query_compute1, df_gene_peak_query

	## ====================================================
	# query candidate peak number for each target gene and potential target gene number for each candidate peak and other peak-gene association attributes
	def test_peak_gene_query_basic_1(self,data=[],input_filename='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',verbose=0,select_config={}):

		"""
		query candidate peak number for each target gene and potential target gene number for each candidate peak and other peak-gene association attributes
		:param data: (dataframe) annotations of peak-gene links
		:param input_filename: path of the file which saved the peak-gene link annotations
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. (dataframe) peak-gene association attributes for each target gene, including candidate peak number, 
								maximal and minimal peak-gene TSS distances, and maximal and minimal peak accessibility-gene expression correlations;
				 2. (dataframe) peak-gene association attributes for each candidate peak, including potential target gene number, 
				 				maximal and minimal peak-gene TSS distances, and maximal and minimal peak accessibility-gene expression correlations;
		"""

		if len(data)==0:
			if input_filename=='':
				input_filename = select_config['filename_save_thresh2']

		if output_file_path=='':
			output_file_path = self.save_path_1
		if filename_prefix_save=='':
			filename_prefix_save = 'df'

		df_gene_peak_query_group_1, df_gene_peak_query_group_2 = [], []
		df_query = data

		feature_type_vec = ['gene','peak']
		feature_type_query1, feature_type_query2 = feature_type_vec
		column_idvec= ['%s_id'%(feature_type_query) for feature_type_query in feature_type_vec]
		column_id1, column_id2 = column_idvec
		
		flag_query=1
		if len(df_query)>0:
			df_gene_peak_query_1 = df_query
		else:
			if os.path.exists(input_filename)==True:
				# df_gene_peak_query_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				df_gene_peak_query_1 = pd.read_csv(input_filename,index_col=False,sep='\t')
				df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1[column_id1])
				print('peak-gene links, dataframe of size ',df_gene_peak_query_1.shape)
			else:
				print('the file does not exist ',input_filename)
				flag_query=0
				return

		gene_query_vec_ori = df_gene_peak_query_1[column_id1].unique()
		gene_query_num_ori = len(gene_query_vec_ori)
		if verbose>0:
			print('gene number: %d'%(gene_query_num_ori))
		
		if flag_query>0:
			# thresh_highly_variable = 0.5
			column_id_query1 = select_config['column_highly_variable']
			if column_id_query1 in df_gene_peak_query_1.columns:
				id1=(df_gene_peak_query_1[column_id_query1]>0)
				id2=(~id1)
				gene_idvec_1 = df_gene_peak_query_1.loc[id1,column_id1].unique()
				gene_idvec_2 = df_gene_peak_query_1.loc[id2,column_id2].unique()
				df_gene_peak_query_1.loc[id2,column_id_query1]=0
			else:
				df_annot = self.df_gene_annot_expr
				gene_name_query_expr = df_annot.index
				id1 = (df_annot['highly_variable']>0)
				gene_highly_variable = gene_name_query_expr[id1]
				gene_group2 = gene_name_query_expr[(~id1)]
				gene_query_num1, gene_query_num2 = len(gene_highly_variable), len(gene_group2)
				# if verbose>0:
				# 	print('gene_highly_variable, gene_group2 ',gene_query_num1,gene_query_num2)
				
				gene_idvec_1 = pd.Index(gene_highly_variable).intersection(gene_query_vec_ori,sort=False)
				gene_idvec_2 = pd.Index(gene_group2).intersection(gene_query_vec_ori,sort=False)
				df_gene_peak_query_1[column_id_query1] = 0
				df_gene_peak_query_1.loc[gene_idvec_1,column_id_query1] = 1

			gene_num1, gene_num2 = len(gene_idvec_1), len(gene_idvec_2)

			df_gene_peak_query_1['count'] = 1
			df_gene_peak_query_1.index = np.asarray(df_gene_peak_query_1[column_id1])
			column_distance = select_config['column_distance']
			column_distance_1 = '%s_abs'%(column_distance)
			df_gene_peak_query_1[column_distance_1] = df_gene_peak_query_1[column_distance].abs()
			column_correlation_1 = select_config['column_correlation'][0]

			column_vec = column_idvec
			query_num1 = len(column_idvec)
			column_1, column_2 = '%s_num'%(feature_type_query1), '%s_num'%(feature_type_query2)
			query_vec_1 = [column_2,column_1]
			query_vec_2 = [[column_2,column_id_query1],[column_1]]
			annot_vec = ['%s_basic'%(feature_type_query) for feature_type_query in feature_type_vec]
			column_vec_query = [column_distance_1,column_correlation_1]
			column_vec_annot = ['distance','corr']
			list_query1 = []
			for i1 in range(query_num1):
				column_id_1 = column_vec[i1]
				column_name, column_sort = query_vec_1[i1], query_vec_2[i1]
				filename_annot1 = annot_vec[i1]
				df_gene_peak_query_group = df_gene_peak_query_1.loc[:,[column_id_1,'count']].groupby(by=[column_id_1]).sum()
				df_gene_peak_query_group = df_gene_peak_query_group.rename(columns={'count':column_name})

				if column_id_1==column_id1:
					df_gene_peak_query_group.loc[gene_idvec_1,column_id_query1] = 1
				df_gene_peak_query_group = df_gene_peak_query_group.sort_values(by=column_sort,ascending=False)
			
				query_num2 = len(column_vec_query)
				list1 = [df_gene_peak_query_group]
				for i2 in range(query_num2):
					column_query = column_vec_query[i2]
					column_annot = column_vec_annot[i2]
					df_query_1 = df_gene_peak_query_1.loc[:,[column_id_1,column_query]]
					df_query1 = df_query_1.groupby(by=[column_id_1]).max().rename(columns={column_query:'%s_max'%(column_annot)})
					df_query2 = df_query_1.groupby(by=[column_id_1]).min().rename(columns={column_query:'%s_min'%(column_annot)})
					list1.extend([df_query1,df_query2])

				df_gene_peak_query_group_combine = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				if verbose>0:
					print('median_value')
					print(df_gene_peak_query_group_combine.median(axis=0))
					print('mean_value')
					print(df_gene_peak_query_group_combine.mean(axis=0))

				if save_mode>0:
					output_filename_1 = '%s/%s.%s.txt'%(output_file_path,filename_prefix_save,filename_annot1)
					df_gene_peak_query_group_combine.to_csv(output_filename_1,sep='\t')
					print('save file: ',output_filename_1)
					list_query1.append(df_gene_peak_query_group_combine)

			df_gene_peak_query_group_1, df_gene_peak_query_group_2 = list_query1

		self.df_gene_peak_query_group_1 = df_gene_peak_query_group_1
		self.df_gene_peak_query_group_2 = df_gene_peak_query_group_2
		
		return df_gene_peak_query_group_1, df_gene_peak_query_group_2

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	# run(opts.file_path)




