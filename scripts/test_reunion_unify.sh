#!/bin/bash -e

input_dir='.'
output_dir='output_file_1'

data_file_type=pbmc
echo $data_file_type

PATH1=$input_dir
atac_meta=$PATH1/atac_meta_pbmc.h5ad	# data matrix format: (row:metacell, column:ATAC-seq peak locus)
rna_meta=$PATH1/rna_meta_pbmc.h5ad	# data matrix format: (row:metacell, column:genei)
filename_motif_1=test_peak_read.pbmc.normalize.motif.thresh5e-05.csv  # format: (row:ATAC-seq peak locus, column:TF motif)
filename_motif_2=test_peak_read.pbmc.normalize.motif_scores.thresh5e-05.csv # format: (row:ATAC-seq peak locus, column:TF motif)
filename_motif_data=$PATH1/$filename_motif_1
filename_motif_data_score=$PATH1/$filename_motif_2
file_mapping=$PATH1/translationTable.csv  # mapping between each TF motif to the corresponding TF; including column 'motif_id' (TF motif identifier) and column 'tf_id' (TF name);
file_peak=$PATH1/test_peak_GC.bed  	# BED file of ATAC-seq peak loci; the first three columns correspond to peak position (chromosome,start,stop); the fifth column corresponds to the GC content of each peak locus;
gene_num_query=22100
peak_bg_num=100
file_bg=$PATH1/test_peak_read.$data_file_type.normalize.bg.$peak_bg_num.1.csv # format: (row:integer index of ATAC-seq peak locus, column:integer index of each background peak sampled for the given ATAC-seq peak)

flag_correlation_query=1
flag_computation='1,3'
flag_query_thresh2=1
flag_basic_query=1
flag_basic_query_2=1
flag_correlation_2=1
flag_discrete_query1=1
type_query_compare=2
flag_cond_query_1=3
flag_score_pre1=3
feature_score_interval=-1
flag_peak_tf_corr=1
flag_gene_tf_corr=1
flag_pcorr_interval=0

celltype=1
b1=$celltype
method_type_feature_link=Unify
echo $method_type_feature_link

query_id1=$1
query_id2=$2
echo $query_id1
echo $query_id2
parallel_1=0
flag_group=1

for i1 in {0..0}; do
	run_id1=$i1
	python test_reunion_unify_1.py --run_id $run_id1 \
									-b $b1 \
									--data_file_type $data_file_type \
									--input_dir $input_dir \
									--atac_meta $atac_meta \
									--rna_meta $rna_meta \
									--motif_data $filename_motif_data \
									--motif_data_score $filename_motif_data_score \
									--file_mapping $file_mapping \
									--file_peak $file_peak \
									--file_bg $file_bg \
									--gene_num_query $gene_num_query \
									--method_type_feature_link $method_type_feature_link \
									--flag_correlation_query $flag_correlation_query \
									--flag_computation $flag_computation \
									--flag_query_thresh2 $flag_query_thresh2 \
									--flag_basic_query $flag_basic_query \
									--flag_basic_query_2 $flag_basic_query_2 \
									--flag_correlation_2 $flag_correlation_2 \
									--flag_discrete_query1 $flag_discrete_query1 \
									--type_query_compare $type_query_compare \
									--flag_cond_query_1 $flag_cond_query_1 \
									--flag_score_pre1 $flag_score_pre1 \
									--flag_pcorr_interval $flag_pcorr_interval \
									--feature_score_interval $feature_score_interval \
									--flag_peak_tf_corr $flag_peak_tf_corr \
									--flag_gene_tf_corr $flag_gene_tf_corr \
									--q_id1 $query_id1 \
									--q_id2 $query_id2 \
									--parallel_1 $parallel_1 \
									--beta_mode $beta_mode \
									--output_dir $output_dir
done

# for example, the script can be run as: test_reunion_unify.sh 0 50
# the command above represents using Unify for peak-TF-gene association inference for the first 50 most highly variable genes;


