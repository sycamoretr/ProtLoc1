# Predicting Protein Subcellular Localization with Multi-label using GraphSAGE and Multi-head Attention Mechanism. 

# Project tree

 * data
   * adj_matrix
   * esm_3b_csv
   * pdb
   * adj_matrix.py
   * dataset.py
   * esm2.py
   * get_pdb.py
   * test_label_dict.npy
   * testdata.fasta
   * train_data.fasta
   * train_label_dict.npy
   * val_data.fasta
 * [model
   * model.py
 * utils
   * Evaluation_metrics.py)
 * train.py
 * predict.py
 * README.md
 
**Requirements**
=
torch==1.11.0+cu113  
torch-cluster==1.6.0  
torch-geometric==2.1.0  
torch-scatter==2.0.9  
torch-sparse==0.6.13  
torchvision==0.12.0+cu113  
biopython==1.81  
huggingface-hub==0.17.1  
numpy==1.22.4  
scikit-learn==1.3.0  
sentence-transformers==2.2.2  

**If you want to reproduce our model, follow the steps below to run the script.**
=
**1. Gets and processes the protein's pdb file to obtain the carbon atom location information and save it as an npy file. You can run the following script.**   
> python get_pdb.py
  
**2. Calculate the Euclidean distance between carbon atoms and save it as a csv file. You can run the following script.**  
> python adj_matrix.py
      
**3. Use the protein language model ESM-2 to code the protein sequence and save it as a csv file. You can run the following script.**    
> python esm2.py
  
**4. Constructing the dataset. You can run the following script.**  
> python dataset.py
    
**5. To train the model, you can run the following script.**    
> python train.py
    
**6. You can also use our trained parameters for prediction directly after implementing steps 1, 2, 3, and 4. You can run the following script.**  
> python predict.py
    
