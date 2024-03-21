# REUNION
REUNION: Transcription factor binding prediction
and regulatory association inference from single-cell
multi-omics data

REUNION is an integrative computational framework which utilizes the single cell multi-omics data as input to infer peak-transcription factor (TF)-gene triplet regulatory associations and predict genome-wide TF binding activities in the peaks with or without TF motifs detected. 
REUNION unites two functionally cooperative methods Unify and ReDiscover. 
Unify performs regulatory association estimation utilizing the single-cell multi-omics data.
ReDiscover takes the regulatory associations estimated by Unify as input to perform TF binding prediction. Unify and ReDiscover supports each other within one framework.

Unify

The command to use Unify to infer peak-TF-gene associations is as follows:

python test_reunion_1.py [Options]


ReDiscover

The command to use ReDiscover to perform TF binding prediction is as follows:

python test_reunion_2.py [Options]

The options:

- -p, --root_path : root directory of the data files

- -r, --run_id : experiment id, default = 0

- -b, --cell : cell type, default = 0
  
  cell type 0 represents PBMC data

- --path_save: the directory where the ATAC-seq and RNA-seq normalized read count matrix of the metacells are saved, default = '.'

    The default parameter represents the ATAC-seq and RNA-seq metacell data are saved in the same directory of the code. Please change this parameter to the directory of the data.

    Please name the ATAC-seq and RNA-seq data of the metacells in the following format: atac_meta_$data_file_type.extension, rna_meta_$data_file_type, where $data_file_type was specified using the 'data_file_type' parameter as shown above.

    'extension' represents the file format. ReDiscover supports the following file formats: (1) anndata, extension=ad or h5ad; (2) the original or compressed tab-delimited tsv, txt files or csv files, extension=tsv, txt, csv, or tsv.gz, txt.gz, csv.gz; 

- --atac_meta: the filename of the ATAC-seq read count matrix of the metacells, default = -1. If this parameter is specified, ReDiscover will not use the 'path_save' parameter as shown above to locate the ATAC-seq data of the metacells.

- --rna_meta: the filename of the RNA-seq read count matrix of the metacells, default = -1. If this parameter is specified, ReDiscover will not use the 'path_save' parameter as shown above to locate the RNA-seq data of the metacells.
  
- --method_type_feature_link: the method which provides initially estimated peak-TF associations as input to ReDiscover, default = 'Unify'. The default parameter represents using the peak-TF associations predicted by Unify as input.

  ReDiscover can also take peak-TF associations predicted by other methods as input. In that case, pleaes provide the name of the corresponding method.

- --method_type_group: the method for peak clustering, default = phenograph.20. The default parameter represents using PhenoGraph algorithm for clustering with the number of neighbors = 20
  
  To use PhenoGraph clustering with a specific number of neibhors, please use: phenogrph.$num, which represents the number of neighbors = $num

- --thresh_size_group: the threshold on peak cluster size, default = 15
  
- --component: the number of components to keep when applying SVD to the accessiblity feature matrix and the sequence feature matrix of the peaks, default = 100

- --component2: the number of feature dimensions to use when building the feature matrix of the peaks in the accessibility feature space and the sequence feature space, default = 50. Please note that $component2 <= $component. The total number of feature dimensions for a peak is d=2*$component2, as we concatenate the accessibility feature vector and the sequence feature vector for the peak.

- --neighbor: the number of K nearest neighbors (KNN) estimated for each peak in the accessibility or sequence feature space, default = 100

- --neighbor_sel: the number of nearest neighbors used for each peak when performing the pseudo training sample selection, default = 30. Please note that $neighbor_sel<=$neighbor.

- --model_type: the prediction model used in ReDiscover, default = 'LogisticRegression', available values: {'LogisticRegression', 'XGBoostClassifier'}. ReDiscover supports using the logistric regression model or the XGBoost classifier as the prediction model. 

- --ratio_1: the ratio of the number of pseudo negative training samples selected from the peaks with motif of a given TF detected but without TF binding predicted by Unify or the method specified in 'method_type_feature_link' (noted as N_neg,1) compared to the number of pseudo positive training samples selected (noted as N_pos), default = 0.25. We have N_neg,1 = N_pos*$ratio_1.

- --ratio_2: the ratio of the number of pseudo negative training samples selected from the peaks with the given TF motif detected (noted as N_neg,2) compared to the number of pseudo positive training samples selected (N_pos), default = 1.25. We have N_neg,2 = N_pos*$ratio_2.

- --thresh_score: the thresholds for the normalized peak-TF scores to select pseudo positive training samples from the paired peak group with or without enrichment of peaks predicted to be bound by a given TF by Unify or the specified method (noted as predicted TF-binding peaks), default = '0.25,0.75'

    thresh_score has the format 'thresh1,thresh2'. thresh1 or thresh 2 represents the threshold used to selected pseudo postivie training samples from the peak group with or without enrichment of the predicted TF-binidng peaks, respectively. ReDiscover performas quantile normalization for the original peak-TF scores. The normalized scores are between 0 and 1. Please use thresh1, thresh2 in [0,1].


************************************************************************************
# Required pre-installed packages
REUNION requires the following packages to be installed:
- Python 
- scikit-learn
- NumPy 
- SciPy
- Scanpy
- Pandas
  
