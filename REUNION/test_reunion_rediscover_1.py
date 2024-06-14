#!/usr/bin/env python
# coding: utf-8

import os
import os.path
from optparse import OptionParser

import REUNION
from REUNION import test_rediscover_compute_3
from REUNION.test_rediscover_compute_3 import run

def run_1(chromosome,run_id,species,cell,generate,chromvec,testchromvec,data_file_type,input_dir,
			filename_atac_meta,filename_rna_meta,filename_motif_data,filename_motif_data_score,file_mapping,file_peak,metacell_num,peak_distance_thresh,
			highly_variable,method_type_feature_link,method_type_dimension,tf_name,filename_prefix,filename_annot,input_link,columns_1,
			output_dir,output_filename,method_type_group,thresh_size_group,thresh_score_group_1,
			n_components,n_components_2,neighbor_num,neighbor_num_sel,model_type_id,ratio_1,ratio_2,thresh_score,
			upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
			typeid2,type_combine,folder_id,config_id_2,config_group_annot,flag_group,flag_embedding_compute,flag_clustering,flag_group_load,flag_scale_1,train_id1,
			beta_mode,verbose_mode,query_id1,query_id2,query_id_1,query_id_2,train_mode,config_id_load):

	flag_1=1
	if flag_1==1:
		run(chromosome,run_id,species,cell,generate,chromvec,testchromvec,data_file_type=data_file_type,
					metacell_num=metacell_num,
					peak_distance_thresh=peak_distance_thresh,
					highly_variable=highly_variable,
					input_dir=input_dir,
					filename_atac_meta=filename_atac_meta,
					filename_rna_meta=filename_rna_meta,
					filename_motif_data=filename_motif_data,
					filename_motif_data_score=filename_motif_data_score,
					file_mapping=file_mapping,
					file_peak=file_peak,
					method_type_feature_link=method_type_feature_link,
					method_type_dimension=method_type_dimension,
					tf_name=tf_name,
					filename_prefix=filename_prefix,filename_annot=filename_annot,
					input_link=input_link,
					columns_1=columns_1,
					output_dir=output_dir,
					output_filename=output_filename,
					path_id=path_id,save=save,
					type_group=type_group,type_group_2=type_group_2,type_group_load_mode=type_group_load_mode,
					type_combine=type_combine,
					method_type_group=method_type_group,
					thresh_size_group=thresh_size_group,thresh_score_group_1=thresh_score_group_1,
					n_components=n_components,n_components_2=n_components_2,
					neighbor_num=neighbor_num,neighbor_num_sel=neighbor_num_sel,
					model_type_id=model_type_id,
					ratio_1=ratio_1,ratio_2=ratio_2,
					thresh_score=thresh_score,
					flag_group=flag_group,
					flag_embedding_compute=flag_embedding_compute,
					flag_clustering=flag_clustering,
					flag_group_load=flag_group_load,
					flag_scale_1=flag_scale_1,
					upstream=upstream,
					downstream=downstream,
					type_id_query=type_id_query,
					thresh_fdr_peak_tf=thresh_fdr_peak_tf,
					typeid2=typeid2,
					folder_id=folder_id,
					config_id_2=config_id_2,
					config_group_annot=config_group_annot,
					train_id1=train_id1,
					beta_mode=beta_mode,
					verbose_mode=verbose_mode,
					query_id1=query_id1,query_id2=query_id2,
					query_id_1=query_id_1,query_id_2=query_id_2,
					train_mode=train_mode,config_id_load=config_id_load)

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="1", help="experiment id")
	parser.add_option("-f","--chromosome", default="1", help="Chromosome name")
	parser.add_option("-g","--generate", default="0", help="whether to generate feature vector: 1: generate; 0: not generate")
	parser.add_option("-c","--chromvec",default="1",help="chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-t","--testchromvec",default="10",help="test chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-i","--species",default="0",help="species id")
	parser.add_option("-b","--cell",default="0",help="cell type")
	parser.add_option("--data_file_type",default="pbmc",help="the cell type or dataset annotation")
	parser.add_option("--input_dir",default=".",help="the directory where the ATAC-seq and RNA-seq data of the metacells are saved")
	parser.add_option("--atac_meta",default="-1",help="file path of ATAC-seq data of the metacells")
	parser.add_option("--rna_meta",default="-1",help="file path of RNA-seq data of the metacells")
	parser.add_option("--motif_data",default="-1",help="file path of binary motif scannning results")
	parser.add_option("--motif_data_score",default="-1",help="file path of the motif scores by motif scanning")
	parser.add_option("--file_mapping",default="-1",help="file path of the mapping between TF motif identifier and the TF name")
	parser.add_option("--file_peak",default="-1",help="file containing the ATAC-seq peak loci annotations")
	parser.add_option("--metacell",default="500",help="metacell number")
	parser.add_option("--peak_distance",default="500",help="peak distance")
	parser.add_option("--highly_variable",default="1",help="highly variable gene")
	parser.add_option("--method_type_feature_link",default="Unify",help='method for initial peak-TF association prediction')
	parser.add_option("--method_type_dimension",default="SVD",help='method for dimension reduction')
	parser.add_option("--tf",default='-1',help='the TF for which to predict peak-TF associations')
	parser.add_option("--filename_prefix",default='-1',help='prefix as part of the filenname of the initially predicted peak-TF assocations')
	parser.add_option("--filename_annot",default='1',help='annotation as part of the filename of the initially predicted peak-TF assocations')
	parser.add_option("--input_link",default='-1',help=' the directory where initially predicted peak-TF associations are saved')
	parser.add_option("--columns_1",default='pred,score',help='the columns corresponding to binary prediction and peak-TF association score')
	parser.add_option("--output_dir",default='output_file',help='the directory to save the output')
	parser.add_option("--output_filename",default='-1',help='filename of the predicted peak-TF assocations')
	parser.add_option("--method_type_group",default="phenograph.20",help="the method for peak clustering")
	parser.add_option("--thresh_size_group",default="0",help="the threshold on peak cluster size")
	parser.add_option("--thresh_score_group_1",default="0.15",help="the threshold on peak-TF association score")
	parser.add_option("--component",default="100",help='the number of components to keep when applying SVD')
	parser.add_option("--component2",default="50",help='feature dimensions to use in each feature space')
	parser.add_option("--neighbor",default='100',help='the number of nearest neighbors estimated for each peak')
	parser.add_option("--neighbor_sel",default='30',help='the number of nearest neighbors to use for each peak when performing pseudo training sample selection')
	parser.add_option("--model_type",default="LogisticRegression",help="the prediction model")
	parser.add_option("--ratio_1",default="0.25",help="the ratio of pseudo negative training samples selected from peaks with motifs and without initially predicted TF binding compared to selected pseudo positive training samples")
	parser.add_option("--ratio_2",default="1.5",help="the ratio of pseudo negative training samples selected from peaks without motifs compared to selected pseudo positive training samples")
	parser.add_option("--thresh_score",default="0.25,0.75",help="thresholds on the normalized peak-TF scores to select pseudo positive training samples from the paired peak groups with or without enrichment of initially predicted TF-binding peaks")
	parser.add_option("--upstream",default="100",help="TRIPOD upstream")
	parser.add_option("--downstream",default="-1",help="TRIPOD downstream")
	parser.add_option("--typeid1",default="0",help="TRIPOD type_id_query")
	parser.add_option("--thresh_fdr_peak_tf",default="0.2",help="GRaNIE thresh_fdr_peak_tf")
	parser.add_option("--path1",default="2",help="file_path_id")
	parser.add_option("--save",default="-1",help="run_id_save")
	parser.add_option("--type_group",default="0",help="type_id_group")
	parser.add_option("--type_group_2",default="0",help="type_id_group_2")
	parser.add_option("--type_group_load_mode",default="1",help="type_group_load_mode")
	parser.add_option("--typeid2",default="0",help="type_id_query_2")
	parser.add_option("--type_combine",default="0",help="type_combine")
	parser.add_option("--folder_id",default="1",help="folder_id")
	parser.add_option("--config_id_2",default="1",help="config_id_2")
	parser.add_option("--config_group_annot",default="1",help="config_group_annot")
	parser.add_option("--flag_group",default="-1",help="flag_group")
	parser.add_option("--flag_embedding_compute",default="0",help="compute feature embeddings")
	parser.add_option("--flag_clustering",default="-1",help="perform clustering")
	parser.add_option("--flag_group_load",default="1",help="load group annotation")
	parser.add_option("--train_id1",default="1",help="train_id1")
	parser.add_option("--flag_scale_1",default="0",help="flag_scale_1")
	parser.add_option("--beta_mode",default="0",help="beta_mode")
	parser.add_option("--motif_id_1",default="1",help="motif_id_1")
	parser.add_option("--verbose_mode",default="1",help="verbose mode")
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
	run_1(opts.chromosome,
		opts.run_id,
		opts.species,
		opts.cell,
		opts.generate,
		opts.chromvec,
		opts.testchromvec,
		opts.data_file_type,
		opts.input_dir,
		opts.atac_meta,
		opts.rna_meta,
		opts.motif_data,
		opts.motif_data_score,
		opts.file_mapping,
		opts.file_peak,
		opts.metacell,
		opts.peak_distance,
		opts.highly_variable,
		opts.method_type_feature_link,
		opts.method_type_dimension,
		opts.tf,
		opts.filename_prefix,
		opts.filename_annot,
		opts.input_link,
		opts.columns_1,
		opts.output_dir,
		opts.output_filename,
		opts.method_type_group,
		opts.thresh_size_group,
		opts.thresh_score_group_1,
		opts.component,
		opts.component2,
		opts.neighbor,
		opts.neighbor_sel,
		opts.model_type,
		opts.ratio_1,
		opts.ratio_2,
		opts.thresh_score,
		opts.upstream,
		opts.downstream,
		opts.typeid1,
		opts.thresh_fdr_peak_tf,
		opts.path1,
		opts.save,
		opts.type_group,
		opts.type_group_2,
		opts.type_group_load_mode,
		opts.typeid2,
		opts.type_combine,
		opts.folder_id,
		opts.config_id_2,
		opts.config_group_annot,
		opts.flag_group,
		opts.flag_embedding_compute,
		opts.flag_clustering,
		opts.flag_group_load,
		opts.flag_scale_1,
		opts.train_id1,
		opts.beta_mode,
		opts.verbose_mode,
		opts.q_id1,
		opts.q_id2,
		opts.q_id_1,
		opts.q_id_2,
		opts.train_mode,
		opts.config_id)







