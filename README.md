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

--method_type_group",default="MiniBatchKMeans.50",help="method_type_group")

- -t, --method_mode : 

- -i, --initial_weight : initial weight for initial parameters, default = 0.1

- -j, --initial_magnitude : initial magnitude for initial parameters, default = 2

- -s, --version : dataset version, default = 1

************************************************************************************
# Required pre-installed packages
REUNION requires the following packages to be installed:
- Python 
- scikit-learn
- NumPy 
- SciPy
- Scanpy
- Pandas
  
