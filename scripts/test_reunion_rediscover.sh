#!/bin/bash -e

input_dir='data1'
output_dir='output_file_2'

data_file_type=pbmc
echo $data_file_type

PATH1=$input_dir
filename_gene_annot=$PATH1/test_gene_annot.Homo_sapiens.GRCh38.108.combine.2.pbmc.txt # the gene annotation file
atac_meta=$PATH1/atac_meta_pbmc.h5ad	# data matrix format: (row:metacell, column:ATAC-seq peak locus)
rna_meta=$PATH1/rna_meta_pbmc.h5ad	# data matrix format: (row:metacell, column:genei)
filename_motif_1=test_peak_read.pbmc.normalize.motif.thresh5e-05.csv # format: (row:ATAC-seq peak locus, column:TF motif)
filename_motif_2=test_peak_read.pbmc.normalize.motif_scores.thresh5e-05.csv # format: (row:ATAC-seq peak locus, column:TF motif)
filename_motif_data=$PATH1/$filename_motif_1
filename_motif_data_score=$PATH1/$filename_motif_2
file_mapping=$PATH1/translationTable.csv  # mapping between each TF motif to the corresponding TF; including column 'motif_id' (TF motif identifier) and column 'tf_id' (TF name);
file_peak=$PATH1/test_peak_GC.bed  # BED file of ATAC-seq peak loci; the first three columns correspond to peak position (chromosome,start,stop); the fifth column corresponds to the GC content of each peak locus;

method_type_group=phenograph.20
tf_name='ATF2,EBF1'
model_type_id1=LogisticRegression
method_type_feature_link=Unify

# path_input_link_2=$PATH1/folder_1 # the folder which saves the peak-TF link predictions by the first method
path_input_link_2=$PATH1_2/data2/folder_save_3/vbak1 # the folder to save the peak-TF link predictions by the first method
path_output_link_2=$output_dir/file_link # the folder which saves the peak-TF link predictions by the first method
filename_prefix='test_query_binding'
filename_annot='1'
filename_annot_link_2=$filename_annot.pred2
columns_1='Unify.pred,Unify.score,Unify.motif'

echo $method_type_feature_link
echo $path_input_link_2

component=100
component2=50
neighbor_num=100
neighbor_num_sel=30
flag_embedding_compute=1
flag_clustering=1
# flag_group_load=1
flag_group_load=0
ratio_1=0.25
ratio_2=1.5
type_combine=1
celltype=1
b1=$celltype
beta_mode=0
neighbor_sel_vec=(30 50 100)
thresh_vec=('0.25,0.75' '0.5,0.9')
i1=0
query_id_1=$1
query_id_2=$2
echo $query_id_1
echo $query_id_2
flag_group=1

for i2 in {0..0}; do
	neighbor_num_sel=${neighbor_sel_vec[$i1]}
	thresh_score=${thresh_vec[$i2]}
	run_id1=$i2
	python test_reunion_rediscover_1.py --run_id $run_id1 \
										-b $b1 \
										--data_file_type $data_file_type \
										--input_dir $input_dir \
										--gene_annot $filename_gene_annot \
										--atac_meta $atac_meta \
										--rna_meta $rna_meta \
										--motif_data $filename_motif_data \
										--motif_data_score $filename_motif_data_score \
										--file_mapping $file_mapping \
										--file_peak $file_peak \
										--method_type_group $method_type_group \
										--method_type_feature_link $method_type_feature_link \
										--tf $tf_name \
										--input_link $path_input_link_2 \
										--output_link $path_output_link_2 \
										--model_type $model_type_id1 \
										--filename_prefix $filename_prefix \
										--filename_annot $filename_annot \
										--filename_annot_link_2 $filename_annot_link_2 \
										--columns_1 $columns_1 \
										--component $component \
										--component2 $component2 \
										--neighbor $neighbor_num \
										--neighbor_sel $neighbor_num_sel \
										--flag_embedding_compute $flag_embedding_compute \
										--flag_clustering $flag_clustering \
										--flag_group_load $flag_group_load \
										--ratio_1 $ratio_1 \
										--ratio_2 $ratio_2 \
										--thresh_score $thresh_score \
										--type_combine $type_combine \
										--flag_group $flag_group \
										--q_id_1 $query_id_1 \
										--q_id_2 $query_id_2 \
										--beta_mode $beta_mode \
										--output_dir $output_dir
done

# for example, the script can be run as: test_reunion_redicover.sh 0 2
# the command above represents using Rediscover for TF binding prediction for the first two given TFs;


