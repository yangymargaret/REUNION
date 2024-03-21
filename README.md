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
Re
The command to use ReDiscover to perform TF binding prediction is as follows:

python test_reunion_2.py [Options]

The options:

- -p, --root_path : root directory of the data files

- -r, --run_id : experiment id, default = 0
- 
- -b, --cell : cell type, default = 0
  
  cell type 0 represents PBMC data

- --method_type_group : the method for peak clustering, default = phenograph.20,
  
  phenograph.20 represents using PhenoGraph algorithm for clusetering with the number of neighbors = 20
  
  To use PhenoGraph clustering with specific number of neibhors, please use: phenogrph.$num, which represents the number of neighbors = $num

- --thresh_size_group : the threshold on peak cluster size, default = 15

- --method_type_feature_link: the method which provides initially estimated peak-TF associations as input to ReDiscover, default = 'Unify'

  The default parameter represents using the peak-TF associations predicted by Unify as input

  ReDiscover can also take peak-TF associations predicted by other methods as input. In that case, pleaes use the name of the correspoind method

- --component

- --component2

- --neighbor

- --neighbor_sel

- --model_type

- --ratio_1

- --ratio_2

- --flag_group

- --flag_group

- --flag_select_1

- --flag_select_2

- --thresh_score

- --q_id1

- --q_id2

- --config_id

************************************************************************************
# Required pre-installed packages
REUNION requires the following packages to be installed:
- Python 
- scikit-learn
- NumPy 
- SciPy
- Scanpy
- Pandas
  
