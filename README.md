# DLGCN
DLGCN is a method based on graph convolution neural network to predict drug-lncRNA associations. Here are the codes and datasets.
## Datasets
- data/drug_name.txt is the drug name matrix of 35 drugs, which corresponds to the row and column of the drug similarity matrix and the row of the association matrix.
- data/lnc_name.txt is the lncNRA name matrix of 50 lncRNAs, which corresponds to the row and column of the lncRNA similarity matrix and the column of the association matrix.
- data/drug_structure_sim.txt is the drug similarity matrix of 35 drugs, which is calculated based on drug structure features.
- data/drug_gene_sim.txt is the drug similarity matrix of 35 drugs, which is calculated based on drug gene features.
- data/drug_pathway_sim.txt is the drug similarity matrix of 35 drugs, which is calculated based on drug pathway features.
- data/drug_mixed_sim.txt is the drug similarity matrix of 35 drugs, which is calculated based on drug mixed features.
- data/lnc_sim.txt is the disease similarity matrix of 50lncRNAs,which is calculated based on lncRNA sequences.
- data/matrix_train.txt is the drug_lncRNA association matrix, which contain 1750 associations between 35 drugs and 50 lncRNAs.
## TL-HGBI
The code of the TL-HGBI that is reproduced in the folder.
## Install dependent Python packages
```python
pip3 install numpy==1.18.1
pip3 install pandas==1.0.1
pip3 install scikit-learn==0.22.1
pip3 install tensorflow==1.15.0
pip3 install scipy==1.4.1
```
## Usage
example
```
cd /DLGCN
python main.py
```
Input includes drug similarity matrix, lncRNA matrix and association matrix. Two files are output, and the results are as follows:
- predict_score.txt is the score of all test sets output from the Leave-One-Out Cross Validation,in which the first column is the prediction label and the second column is the prediction score.
- predict_result.txt is the predictive association matrix of non-sampling training output.
