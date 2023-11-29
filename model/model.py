import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv , GCNConv , GATConv
from torch_geometric.nn import global_mean_pool

class Attention(nn.Module):

    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention
class SAGE(nn.Module):
    def __init__(self,):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels=2560, out_channels=1024, aggr='mean')
        self.conv2 = SAGEConv(in_channels=1024, out_channels=512, aggr='mean')
        self.norm1 = torch.nn.BatchNorm1d(1024)

    def forward(self, data,device):
        data = data.to(device)
        x, edge_index = data.x, data.edge_index
        hid = self.conv1(x=x, edge_index=edge_index)
        hid = F.relu(hid)
        hid = self.norm1(hid)
        hid = F.dropout(hid, p=0.3, training=self.training)
        hid = self.conv2(x=hid, edge_index=edge_index)
        return hid

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.SAGE = SAGE()
        self.Attention = Attention(512,128,128)
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=3,stride=1)
        self.Norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=16,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.fc_1 = nn.Linear(432, 1024)
        self.fc_2 = nn.Linear(1024, 14)
        self.criterion =  nn.BCELoss()

    def forward(self, data,device):
        data = data.to(device)
        protx = self.SAGE(data,device)
        x = protx.unsqueeze(0).float()
        att = self.Attention(x)  											# att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x
        embedding = node_feature_embedding
        embedding = embedding.unsqueeze(0).float()
        embedding = self.cnn1(embedding)
        embedding =self.Norm(embedding)
        embedding=self.relu(embedding)
        embedding = self.conv1(embedding)
        embedding = self.conv1(embedding)
        embedding = self.conv1(embedding)
        embedding = self.conv1(embedding)
        embedding = self.conv1(embedding)
        embedding = self.conv2(embedding)

        embedding = torch.flatten(embedding,start_dim=1)
        embedding= self.fc_1(embedding)
        embedding = self.fc_2(embedding)
        return torch.sigmoid(embedding)

