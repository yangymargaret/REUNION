#!/usr/bin/env python
# coding: utf-8

import os
import os.path
from optparse import OptionParser

import REUNION
from REUNION import test_unify_compute_group_1
from REUNION.test_unify_compute_group_1 import run

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="1", help="experiment id")
	parser.add_option("-f","--chromosome", default="1", help="Chromosome name")
	parser.add_option("-g","--generate", default="0", help="whether to generate feature vector: 1: generate; 0: not generate")
	parser.add_option("-c","--chromvec",default="1",help="chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-t","--testchromvec",default="10",help="test chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-i","--species",default="0",help="species id")
	parser.add_option("-j","--featureid",default="0",help="feature idx")
	parser.add_option("-b","--cell",default="1",help="cell type")
	parser.add_option("--file_path",default="1",help="file_path")
	parser.add_option("--path1",default="1",help="file_path_id")
	parser.add_option("--flag_distance",default="1",help="flag_distance")
	parser.add_option("--data_file_type",default="pbmc",help="the cell type or dataset annotation")
	parser.add_option("--data_file_query",default="0",help="data_file_type_id")
	parser.add_option("--input_dir",default=".",help="the directory where the ATAC-seq and RNA-seq data of the metacells are saved")
	parser.add_option("--gene_annot",default="-1",help="file path of gene position annotation file")
	parser.add_option("--atac_data",default="-1",help="file path of ATAC-seq data of the single cells")
	parser.add_option("--rna_data",default="-1",help="file path of RNA-seq data of the single cells")
	parser.add_option("--atac_meta",default="-1",help="file path of ATAC-seq data of the single cells")
	parser.add_option("--rna_meta",default="-1",help="file path of RNA-seq data of the metacells")
	parser.add_option("--motif_data",default="-1",help="file path of binary motif scannning results")
	parser.add_option("--motif_data_score",default="-1",help="file path of the motif scores by motif scanning")
	parser.add_option("--file_mapping",default="-1",help="file path of the mapping between TF motif identifier and the TF name")
	parser.add_option("--file_peak",default="-1",help="file containing the ATAC-seq peak loci annotations")
	parser.add_option("--file_bg",default="-1",help="file containing the estimated background peak loci")
	parser.add_option("--metacell",default="500",help="metacell number")
	parser.add_option("--peak_distance",default="500",help="peak distance threshold")
	parser.add_option("--highly_variable",default="1",help="highly variable gene")
	parser.add_option("--gene_num_query",default="3000",help="selected highly variable gene number")
	parser.add_option("--method_type_feature_link",default="Unify",help='method_type_feature_link')
	parser.add_option("--output_dir",default='output_file',help='the directory to save the output')
	parser.add_option("--output_filename",default='-1',help='filename of the predicted regulatory assocations')
	parser.add_option("--beta_mode",default="0",help="beta_mode")
	parser.add_option("--recompute",default="0",help="recompute")
	parser.add_option("--interval_save",default="-1",help="interval_save")
	parser.add_option("--q_id1",default="-1",help="query_id1")
	parser.add_option("--q_id2",default="-1",help="query_id2")
	parser.add_option("--fold_id",default="0",help="fold_id")
	parser.add_option("--n_iter_init",default="-1",help="initial estimation iteration number")
	parser.add_option("--n_iter",default="15",help="iteration number")
	parser.add_option("--flag_motif_ori",default="0",help="original motif number")
	parser.add_option("--iter_mode_1",default="0",help="iteration mode")
	parser.add_option("--restart",default="1",help="restart iteration or continue with iteration")
	parser.add_option("--config_id",default="-1",help="config_id")
	parser.add_option("--feature_num",default="200",help="feature query number")
	parser.add_option("--parallel",default="0",help="parallel_mode")
	parser.add_option("--parallel_1",default="0",help="parallel_mode_peak")
	parser.add_option("--flag_motif_data_load",default="0",help="flag_motif_data_load")
	parser.add_option("--motif_data_thresh",default="0",help="threshold for motif scanning")
	parser.add_option("--motif_data_type",default="0",help="motif data type")
	parser.add_option("--flag_correlation_query_1",default="1",help="flag_correlation_query_1")
	parser.add_option("--flag_correlation_query",default="1",help="flag_correlation_query")
	parser.add_option("--flag_correlation_1",default="1",help="flag_correlation_1")
	parser.add_option("--flag_computation",default="1",help="flag_computation_vec")
	parser.add_option("--flag_combine_empirical_1",default="0",help="flag_combine_empirical_1")
	parser.add_option("--flag_combine_empirical",default="0",help="flag_combine_empirical")
	parser.add_option("--flag_query_thresh2",default="0",help="flag_query_thresh2")
	parser.add_option("--flag_merge_1",default="0",help="flag_merge_1")
	parser.add_option("--overwrite_thresh2",default="0",help="overwrite thresh2 file")
	parser.add_option("--flag_correlation_2",default="0",help="flag_correlation_2")
	parser.add_option("--flag_correlation_query1",default="1",help="flag_correlation_query1")
	parser.add_option("--flag_discrete_query1",default='1',help='flag_discrete_query1')
	parser.add_option("--flag_peak_tf_corr",default="0",help="flag_peak_tf_corr")
	parser.add_option("--flag_gene_tf_corr",default="0",help="flag_gene_tf_corr")
	parser.add_option("--flag_gene_expr_corr",default="0",help="flag_gene_expr_corr")
	parser.add_option("--flag_compute_1",default="1",help="initial score computation and selection")
	parser.add_option("--flag_score_pre1",default="2",help="initial score computation")
	parser.add_option("--feature_score_interval",default="-1",help="feature_score_interval")
	parser.add_option("--flag_group_query",default="0",help="flag_group_query")
	parser.add_option("--flag_feature_query1",default="0",help="differential feature query")
	parser.add_option("--flag_feature_query2",default="0",help="differential feature query 2")
	parser.add_option("--flag_feature_query3",default="0",help="differential feature query 3")
	parser.add_option("--flag_basic_query",default="0",help="flag_basic_query")
	parser.add_option("--flag_basic_query_2",default="0",help="flag_basic_query_2")
	parser.add_option("--type_query_compare",default="2",help="type_query_compare")
	parser.add_option("--flag_basic_filter_1",default="0",help="flag_basic_filter_1")
	parser.add_option("--flag_basic_filter_combine_1",default="0",help="flag_basic_filter_combine_1")
	parser.add_option("--flag_basic_filter_2",default="0",help="flag_basic_filter_2")
	parser.add_option("--Lasso_alpha",default="0.001",help="Lasso_alpha")
	parser.add_option("--peak_distance_thresh1",default="500",help="peak distance threshold 1")
	parser.add_option("--peak_distance_thresh2",default="500",help="peak distance threshold 2")
	parser.add_option("--flag_pred_1",default="0",help="flag_pred_1")
	parser.add_option("--flag_pred_2",default="0",help="flag_pred_2")
	parser.add_option("--flag_group_1",default="0",help="flag_group_1")
	parser.add_option("--flag_pcorr_interval",default="0",help="flag_pcorr_interval")
	parser.add_option("--flag_reduce",default="1",help="reduce intermediate files")
	parser.add_option("--flag_combine_1",default="0",help="basic_filter_combine_1")
	parser.add_option("--flag_combine_2",default="0",help="basic_filter_combine_2")
	parser.add_option("--flag_cond_query_1",default="0",help="flag_cond_query_1")
	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':
	opts = parse_args()
	run(opts.run_id,
		opts.chromosome,
		opts.generate,
		opts.chromvec,
		opts.testchromvec,
		opts.species,
		opts.featureid,
		opts.cell,
		opts.file_path,
		opts.path1,
		opts.flag_distance,
		opts.data_file_type,
		opts.data_file_query,
		opts.input_dir,
		opts.gene_annot,
		opts.atac_data,
		opts.rna_data,
		opts.atac_meta,
		opts.rna_meta,
		opts.motif_data,
		opts.motif_data_score,
		opts.file_mapping,
		opts.file_peak,
		opts.file_bg,
		opts.metacell,
		opts.peak_distance,
		opts.highly_variable,
		opts.gene_num_query,
		opts.method_type_feature_link,
		opts.output_dir,
		opts.output_filename,
		opts.beta_mode,
		opts.recompute,
		opts.interval_save,
		opts.q_id1,
		opts.q_id2,
		opts.fold_id,
		opts.n_iter_init,
		opts.n_iter,
		opts.flag_motif_ori,
		opts.iter_mode_1,
		opts.restart,
		opts.config_id,
		opts.feature_num,
		opts.parallel,
		opts.parallel_1,
		opts.flag_motif_data_load,
		opts.motif_data_thresh,
		opts.motif_data_type,
		opts.flag_correlation_query_1,
		opts.flag_correlation_query,
		opts.flag_correlation_1,
		opts.flag_computation,
		opts.flag_combine_empirical_1,
		opts.flag_combine_empirical,
		opts.flag_query_thresh2,
		opts.overwrite_thresh2,
		opts.flag_merge_1,
		opts.flag_correlation_2,
		opts.flag_correlation_query1,
		opts.flag_discrete_query1,
		opts.flag_peak_tf_corr,
		opts.flag_gene_tf_corr,
		opts.flag_gene_expr_corr,
		opts.flag_compute_1,
		opts.flag_score_pre1,
		opts.feature_score_interval,
		opts.flag_group_query,
		opts.flag_feature_query1,
		opts.flag_feature_query2,
		opts.flag_feature_query3,
		opts.flag_basic_query,
		opts.flag_basic_query_2,
		opts.type_query_compare,
		opts.flag_basic_filter_1,
		opts.flag_basic_filter_combine_1,
		opts.flag_basic_filter_2,
		opts.Lasso_alpha,
		opts.peak_distance_thresh1,
		opts.peak_distance_thresh2,
		opts.flag_pred_1,
		opts.flag_pred_2,
		opts.flag_group_1,
		opts.flag_pcorr_interval,
		opts.flag_reduce,
		opts.flag_combine_1,
		opts.flag_combine_2,
		opts.flag_cond_query_1)



