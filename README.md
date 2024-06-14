# REUNION
REUNION: Transcription factor binding prediction
and regulatory association inference from single-cell
multi-omics data

REUNION is an integrative computational framework which utilizes the single cell multi-omics data as input to infer peak-transcription factor (TF)-gene triplet regulatory associations and predict genome-wide TF binding activities in the peaks with or without TF motifs detected. 
REUNION unites two functionally cooperative methods Unify and Rediscover. 
Unify performs regulatory association estimation utilizing the single-cell multi-omics data.
Rediscover takes the regulatory associations estimated by Unify as input to perform TF binding prediction. Unify and Rediscover supports each other within one framework.

## Installation
To install REUNION, please use the following commands:

conda create -n reunion python=3.8

source activate reunion

git clone https://github.com/yangymargaret/REUNION.git

cd REUNION

pip install .

The virtual environment name can be changed to other preferred names.

## Usage
### Unify

The command to use Unify to infer peak-TF-gene associations is as follows (please find test_reunion_unify_1.py in the folder scripts). We provide a script named test_reunion_unify.bash in the folder scripts, which shows an example to run the following command to use Unify for regulatory association inference. 

python test_reunion_unify_1.py [Options]

The options:
- --data_file_type

  the cell type or dataset annotation, default = 'pbmc'
  
- -b, --cell

  cell type or cell population index, default = 1 (representing PBMC data)

- --input_dir

  the directory where the ATAC-seq and RNA-seq normalized read count matrices of the metacells are saved, default = '.'

  The default parameter represents the data are saved in the same directory of the code. Please change this parameter to the directory of the data.

  Please name the ATAC-seq and RNA-seq data of the metacells in the following format:

  ATAC-seq data: atac_meta_$data_file_type.extension;

  RNA-seq data: rna_meta_$data_file_type.extension;

  $data_file_type was specified using the 'data_file_type' parameter. 'extension' represents the file format.

  For the count matrices of the metacells, Unify supports the following file formats:

  (1) AnnData, extension=h5ad;

  (2) the original or compressed tab-delimited tsv, txt files or csv files, extension=tsv, txt, csv, or tsv.gz, txt.gz, csv.gz.

- --atac_meta

  the file path of the ATAC-seq read count matrix of the metacells, default = -1;

  data matrix format: (row: metacell, column: ATAC-seq peak locus);

- --rna_meta

  the file path of the RNA-seq read count matrix of the metacells, default = -1;

  data matrix format: (row: metacell, column: gene);
  
  If atac_meta or rna_meta is specified, Unify will not use the 'input_dir' parameter to locate the ATAC-seq data or RNA-seq data of the metacells, respectively.

- --motif_data

  the filename of peak-motif matrix from the motif scanning results, default = -1;

  data matrix format: (row: ATAC-seq peak locus, column: TF motif);

- --motif_data_score

  the filename of the motif scores from the motif scanning results, default = -1;

  data matrix format: (row: ATAC-seq peak locus, column: TF motif);

- --file_mapping

  the filename of the mapping between the TF motif identifier and the TF name, default = -1;

  the dataframe in the file includes the column 'motif_id' (TF motif identifier) and the column 'tf_id' (TF name);

- --file_peak

  the file containing the ATAC-seq peak loci annotations, default = -1;

  the file is of BED format, with the first three columns representing peak position (chromosome,start,stop) and the fifth column representing GC content of each peak locus;
 
- --file_bg

  the file containing the sampled background ATAC-seq peak loci which match the corresponding ATAC-seq peak locus by GC content and average chromatin accessibility across the metacells, default = -1;

  data matrix format: (row: integer index of ATAC-seq peak locus, column: integer index of each background peak sampled for the given ATAC-seq peak);
  
- --output_dir

  the directory where the output of Unify will be saved, including the predicted peak-TF-gene associations and other associated files, default = 'output_file'

  By default Unify creates a file folder named 'output_file' in the current directory. Please change the parameter to the specific output directory. If the directory does not exist, Unify will try to create it. Unify will then create a sub-folder named 'file_link_1' within the folder $output_dir to save the pre-selected peak-gene associations and a sub-folder named 'file_link_2' to save the estimated peak-TF-gene associations. 

The output:

The output of Unify includes a file containing the estimated peak-TF-gene associations saved in the directory $output_dir/file_link_2. 


### Rediscover

The command to use Rediscover to perform TF binding prediction is as follows (please find test_reunion_rediscover_1.py in the folder scripts). We provide a script named test_reunion_rediscover.bash in the folder scripts, which shows an example to run the following command to use Rediscover for TF binding prediction.

python test_reunion_rediscover_1.py [Options]

The options:

- --data_file_type

  the cell type or dataset annotation, default = 'pbmc'

- -b, --cell

  cell type or cell population index, default = 1 (representing PBMC data)

- --input_dir

  the directory where the ATAC-seq and RNA-seq normalized read count matrix of the metacells are saved, default = '.'

  The default parameter represents the ATAC-seq and RNA-seq metacell data are saved in the current directory. Please change this parameter to the directory of the data.

  Please name the ATAC-seq and RNA-seq data of the metacells in the following format: atac_meta_$data_file_type.extension, rna_meta_$data_file_type, where $data_file_type was specified using the 'data_file_type' parameter.

  'extension' represents the file format. Rediscover supports the following file formats: (1) anndata, extension=ad or h5ad; (2) the original or compressed tab-delimited tsv, txt files or csv files, extension=tsv, txt, csv, or tsv.gz, txt.gz, csv.gz; 

- --atac_meta

  the file path of the ATAC-seq read count matrix of the metacells, default = -1;

  data matrix format: (row:metacell, column:ATAC-seq peak locus)

- --rna_meta

  the file path of the RNA-seq read count matrix of the metacells, default = -1;

  data matrix format: (row: metacell, column: ATAC-seq peak locus);

  If atac_meta or rna_meta is specified, Rediscover will not use the 'input_dir' parameter to locate the ATAC-seq data or RNA-seq data of the metacells, respectively.

- --motif_data

  the filename of peak-motif matrix from the motif scanning results, default = -1;

  data matrix format: (row: metacell, column: ATAC-seq peak locus);

- --motif_data_score

  the filename of the motif scores from the motif scanning results, default = -1;

  data matrix format: (row: metacell, column: ATAC-seq peak locus);
  
- --method_type_feature_link

  the method which provides initially estimated peak-TF associations as input to Rediscover, default = 'Unify'.

  By default we use the peak-TF associations predicted by Unify as input. Rediscover can also take peak-TF associations predicted by other methods as input. In that case, please provide the name of the corresponding method.

- --tf

  the name of the TF for which to predict peak-TF links, for example, ATF3; or a string containing the names of the TFs to query, with TF names delimited by the symbol comma, for example, 'ATF2,ATF3'; default = -1, representing performing TF binding prediction for all the TFs with expressions in the RNA-seq data and with motifs in the given motif collection.

- --filename_prefix

  the prefix as part of the name of the file that contains predicted peak-TF assocations by Unify (or the method specified in 'method_type_feature_link' ) or Rediscover, default = $data_file_type

- --filename_annot

  the annotation as part of the name of the file that contains predicted peak-TF assocations by Unify (or other methods) or Rediscover, default = '1'

  If filename_annot='', filename_annot will not be used in the corresponding filename.
  
- --input_link

  the directory where the file containing the peak-TF associations predicted by Unify (or other methods) for a given TF is saved

  Please provide the file for the given TF as a tab-delmited .txt file named $filename_prefix.$TF_name.$filename_annot.txt containing at least two columns: ['pred','score'] (the column names can be specified by the parameter 'columns_1' as shown below), with the peak positions as rownames. Each row represents the predicted association between the corresponding peak and the given TF by the specific method. 'pred' represents binary prediction: 1: peak contains the TF binding site, 0: without with TF binding site; 'score' represents the association score of the peak-TF link estimated by the method. If the association scores are unavailable, please leave this column blank.

  Optionally, if the motif scores of the given TF in each peak locus based on motif scanning are available, please include them using an additional column named 'motif_score' or $column_motif_score as specified by part of $columns_1 as shown below.

  If there are multiple TFs to query, as specified by the argument 'tf', please prepare a peak-TF association file as described above for each TF. Please place the files into one directory and use the path of the directory to specify 'input_link'.

- --columns_1

  the columns in the peak-TF association file which correspond to the binary prediction and the estimated peak-TF association score by Unify (or other methods), default = 'pred,score'

  columns_1 has the format 'column1,column2' or 'column1,column2,column_motif_score', where column1, column2 represent the columns containing the binary peak-TF link predictions and the estimated peak-TF association scores, respectively, and optionally column_motif_score represents the column containing the motif scores from motif scanning results.
  
- --output_dir

  the directory where the output of Rediscover will be saved, including the predicted peak-TF associations for the given TFs and other associated files, default = 'output_file'

  By default Rediscover creates a file folder named 'output_file' in the current directory. Please change the parameter to the specific output directory. If the directory does not exist, Rediscover will try to create it. Rediscover will then create a sub-folder named 'file_link' within the folder $output_dir to save the estimated peak-TF associations. 

- --output_filename

  the file to save the peak-TF associations predicted by Rediscover, default = -1

  If the default parameter is used, for each TF to query, Rediscover will save the corresponding predictions to a file named $filename_prefix.$TF_name.$filename_annot.pred2.txt and save the files to the directory $output_dir/file_link.
  
- --method_type_group

  the method for peak clustering, default = phenograph.20

  By default we use PhenoGraph algorithm for clustering with the number of neighbors = 20. To use PhenoGraph clustering with a specific number of neighbors, please use: phenogrph.$num, which represents the number of neighbors = $num

- --thresh_size_group

  the threshold on peak cluster size, default = 1
  
- --component

  the number of components to keep when applying SVD to the accessiblity feature matrix and the sequence feature matrix of the peaks, default = 100

- --component2

  the number of feature dimensions to use when building the feature matrix of the peaks in the accessibility feature space and the sequence feature space, default = 50. Please note that $component2 <= $component.

  The total number of feature dimensions for a peak is d=2*$component2, as we concatenate the accessibility feature vector and the sequence feature vector for each peak.

- --neighbor

  the number of K nearest neighbors (KNN) estimated for each peak in the accessibility or sequence feature space, default = 100

- --neighbor_sel

  the number of nearest neighbors used for each peak when performing the pseudo training sample selection, default = 30. Please note that $neighbor_sel<=$neighbor.

- --model_type

  the prediction model used in Rediscover, default = 'LogisticRegression', available values: {'LogisticRegression', 'XGBoostClassifier'}.

  Rediscover supports using the logistic regression model or the XGBoost classifier as the prediction model. By default we use the logistic regression model.

- --ratio_1

  the ratio of the number of pseudo negative training samples selected from the peaks with motif of a given TF detected but without TF binding predicted by Unify or other methods (noted as N_neg,1) compared to the number of pseudo positive training samples selected (noted as N_pos), default = 0.25. We have N_neg,1 = N_pos*$ratio_1.

- --ratio_2

  the ratio of the number of pseudo negative training samples selected from the peaks with the given TF motif detected (noted as N_neg,2) compared to the number of pseudo positive training samples selected (N_pos), default = 1.25. We have N_neg,2 = N_pos*$ratio_2.

- --thresh_score

  the thresholds on the normalized peak-TF scores to select pseudo positive training samples from the paired peak groups with or without enrichment of peaks predicted to be bound by a given TF by Unify or other methods (noted as predicted TF-binding peaks), default = '0.25,0.75'

  thresh_score has the format 'thresh1,thresh2', where thresh1 or thresh 2 represents the threshold used to selected pseudo postivie training samples from the peak group with or without enrichment of the predicted TF-binding peaks, respectively. Rediscover performas quantile normalization for the original peak-TF scores. The normalized scores are between 0 and 1. Please use thresh1, thresh2 in [0,1].

The output:

The output of Rediscover includes a file containing the peak-TF associations between the genome-wide peaks and the given TF predicted by Rediscover, with the filename specified by $output_filename and saved in $output_dir/file_link. The rownames are the genome-wide peaks as present in the columns of the normalized ATAC-seq read count matrix of the metacells. The file contains at least two columns: ['pred','proba']. 'pred': the predicted binary peak-TF link: 1, with binding site; 0, without binding site. 'proba': predicted TF binding probability of the give TF in the corresponding peak locus.


