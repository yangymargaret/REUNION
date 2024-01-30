# REUNION
REUNION: Transcription factor binding prediction
and regulatory association inference from single-cell
multi-omics data

REUNION is an integrative computational framework which utilizes the single cell multi-omics (sc-multiome) data as input to infer peak-TF-gene triplet regulatory associations and predict genome-wide TF binding activities in the peaks with or without TF motifs detected. 
REUNION unites two functionally cooperative methods Unify and ReDiscover. 
Unify utilizes complementary score functions to identify the peak-TF-gene triplet regulatory associations using the sc-multiome data as input. 
ReDiscover takes the regulatory associations inferred by Unify as input and performs learning to predict TF binding in the peaks with or without motifs detected. Unify and ReDiscover supports each other within one framework.
