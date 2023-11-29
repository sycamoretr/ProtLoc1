from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp



def create_dataset(root_x, root_y, dis_threshold):
    label_dict = np.load(root_y,allow_pickle=True).item()
    data_list = []
    for record in tqdm(SeqIO.parse(root_x, 'fasta')):
        # print(str(record.id))
        x = np.loadtxt(r'/root/autodl-tmp/prot/data/esm_3b_csv/' + str(record.id) + '.csv', delimiter=",")
        x = torch.FloatTensor(x)
        adj_matrix = np.loadtxt(r'/root/autodl-tmp/prot/data/adj_matrix/' + str(record.id) + '.csv', delimiter=",")
        adj_matrix[adj_matrix < dis_threshold] = 1
        adj_matrix[adj_matrix >= dis_threshold] = 0
        np.fill_diagonal(adj_matrix, 0)
        # coo_matrix is a sparse matrix format, also known as coordinate format or triplet format,
        # which uses three numbers to store the row index, column index, and value of nonzero elements
        # Convert the adjacency matrix to the coo_matrix sparse matrix format
        adj_sparse_matrix = sp.coo_matrix(adj_matrix)
        edge_index, edge_attr = from_scipy_sparse_matrix(adj_sparse_matrix)
        y = torch.tensor([label_dict[str(record.id)]],dtype=torch.long)
        # print(y)
        data_list.append((Data(x=x, y=y, edge_index=edge_index)))
    return data_list
    
root_x1 = 'train_data.fasta'
root_y1 = 'train_label_dict.npy'
dis_threshold = 20
list1 = create_dataset(root_x1, root_y1, dis_threshold)
torch.save(list1,'traindata_x.pt')

root_x2 = 'val_data.fasta'
root_y2 = 'train_label_dict.npy'
list2 = create_dataset(root_x2, root_y2, dis_threshold)
torch.save(list2,'valdata_x.pt')

root_x3 = 'testdata.fasta'
root_y3 = 'test_label_dict.npy'
list3 = create_dataset(root_x3, root_y3, dis_threshold)
torch.save(list3,'testdata_x.pt')





