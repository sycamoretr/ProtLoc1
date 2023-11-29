import math
from Bio import SeqIO
import numpy as np
import os
import json
import torch.nn.functional as F
from Bio.PDB.PDBParser import PDBParser
import wget
def get_pdb(id):
    url = 'https://alphafold.ebi.ac.uk/files/AF-'+str(id)+'-F1-model_v4.pdb'
    load = './pdb/'+str(id)+'.pdb'
    try:
        wget.download(url, load)
    except:
        print(str(id),'no alphafold')


def read_pdb(id):
    f = './pdb/'+ id+'.pdb'
    structure = PDBParser().get_structure(str(name),f)
    x_y_z_ = list()
    """This gets the coordinates for all the atoms"""
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name == "C":
                        x = (atom.coord[0])
                        y = (atom.coord[1])
                        z = (atom.coord[2])
                        x_y_z_.append([x, y, z])
    return x_y_z_


test_coordinate_dict = {}
for fa in SeqIO.parse("../data/testdata.fasta","fasta"):
    id = fa.id
    get_pdb(id)
    d = read_pdb(str(id))
    test_coordinate_dict[str(id)] = np.array(d)
np.save('test_coordinate_dict.npy',test_coordinate_dict)

train_coordinate_dict = {}
for fa in SeqIO.parse("../data/traindata.fasta","fasta"):
    id = fa.id
    get_pdb(id)
    d = read_pdb(str(id))
    train_coordinate_dict[str(id)] = np.array(d)
np.save('train_coordinate_dict.npy',train_coordinate_dict)

val_coordinate_dict = {}
for fa in SeqIO.parse("../data/valdata.fasta","fasta"):
    id = fa.id
    get_pdb(id)
    d = read_pdb(str(id))
    val_coordinate_dict[str(id)] = np.array(d)
np.save('val_coordinate_dict.npy',val_coordinate_dict)





