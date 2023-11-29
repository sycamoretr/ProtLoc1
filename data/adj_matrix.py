import numpy as np
import csv
from Bio import SeqIO
from tqdm import tqdm
from Bio import PDB
import warnings




def distance_matrix(prot_id, seq, c_coords):
    c_coords = np.array(c_coords)
    dist_matrix = np.zeros((len(seq), len(seq)))
    for i, coord1 in enumerate(c_coords):
        for j, coord2 in enumerate(c_coords):
            dist = np.sqrt(np.sum((coord1 - coord2) ** 2))
            dist_matrix[i, j] = dist

    with open('./adj_matrix/' + prot_id + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for row in dist_matrix:
            writer.writerow(row)


if __name__ == '__main__':
    file_path1 = 'train_data.fasta'
    coordinate_dict1 = np.load('train_coordinate_dict.npy',allow_pickle=True).item()
    for index, record in tqdm(enumerate(SeqIO.parse(file_path1, 'fasta'))):
        c_coords = coordinate_dict1[str(record.id)]
        distance_matrix(str(record.id), str(record.seq),c_coords)
        
    file_path2 = 'val_data.fasta'
    coordinate_dict2 = np.load('val_coordinate_dict.npy',allow_pickle=True).item()
    for index, record in tqdm(enumerate(SeqIO.parse(file_path2, 'fasta'))):
        c_coords = coordinate_dict2[str(record.id)]
        distance_matrix(str(record.id), str(record.seq),c_coords)
        
    file_path3 = 'testdata.fasta'
    coordinate_dict3 = np.load('test_coordinate_dict.npy',allow_pickle=True).item()
    for index, record in tqdm(enumerate(SeqIO.parse(file_path3, 'fasta'))):
        c_coords = coordinate_dict3[str(record.id)]
        distance_matrix(str(record.id), str(record.seq),c_coords)

