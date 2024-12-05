import torch
import torch.nn.functional as F
import torch.nn as nn
from models.layers.DeepAAI_GCN import GCNConv


class CNNmodule(nn.Module):
    def __init__(self, in_channel, kernel_width, l=0):
        super(CNNmodule, self).__init__()
        self.kernel_width = kernel_width
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.out_linear = nn.Linear(l*64, 512)
        self.dropout = nn.Dropout(0.5)


    def forward(self, protein_ft):
        '''
        :param protein_ft: batch*len*amino_dim
        :return:
        '''
        batch_size = protein_ft.size()[0]
        protein_ft = protein_ft.transpose(1, 2)

        conv_ft = self.conv(protein_ft)
        conv_ft = self.dropout(conv_ft)
        conv_ft = self.pool(conv_ft).view(batch_size, -1)
        conv_ft = self.out_linear(conv_ft)
        return conv_ft


class AntisEmbed(nn.Module):
    def __init__(self, **param_dict):
        super(AntisEmbed, self).__init__()
        self.amino_ft_dim = param_dict['amino_type_num'],
        self.param_dict = param_dict
        self.kmer_dim = param_dict['kmer_dim']
        self.h_dim = param_dict['h_dim']
        self.dropout = param_dict['dropout_num']
        self.add_bn = param_dict['add_bn']
        self.add_res = param_dict['add_res']
        self.amino_embedding_dim = param_dict['amino_embedding_dim']

        self.kmer_linear = nn.Linear(param_dict['kmer_dim'], self.h_dim)
        self.pssm_linear = nn.Linear(param_dict['pssm_dim'], self.h_dim)
       
        self.share_linear = nn.Linear(2*self.h_dim, self.h_dim)
        self.share_gcn1 = GCNConv(self.h_dim, self.h_dim)
        self.share_gcn2 = GCNConv(self.h_dim, self.h_dim)

        self.adj_trans = nn.Linear(self.h_dim, self.h_dim)

        self.cross_scale_merge = nn.Parameter(
            torch.ones(1)
        )

        self.activation = nn.ELU()
        for m in self.modules():
            self.weights_init(m)
            
        self.max_len = param_dict['max_len']
            
        self.cnnmodule = CNNmodule(in_channel=22, kernel_width=self.amino_ft_dim, l=self.max_len)
       

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)


    def forward(self, **ft_dict):
        '''
        :param ft_dict:
                ft_dict = {
                'graph_node_ft': FloatTensor  node_num * kmer_dim
                'amino_ft': LongTensor  batch * max_len * 1
                'idx': LongTensor  batch
            }
        :return:
        '''
        device = ft_dict['graph_node_kmer_ft'].device
        graph_node_num = ft_dict['graph_node_kmer_ft'].size()[0]
        
        res_mat = torch.zeros(graph_node_num, self.h_dim).to(device)
        
        node_kmer_ft = self.kmer_linear(ft_dict['graph_node_kmer_ft'])
        node_pssm_ft = self.pssm_linear(ft_dict['graph_node_pssm_ft'])
        node_ft = torch.cat([node_kmer_ft, node_pssm_ft], dim=-1)
        node_ft = self.activation(node_ft)
        node_ft = F.dropout(node_ft, p=self.dropout, training=self.training)
        
        # share
        node_ft = self.share_linear(node_ft)
        res_mat = res_mat + node_ft
        node_ft = self.activation(node_ft)
        node_ft = F.dropout(node_ft, p=self.dropout, training=self.training)
        
        # generate adj
        trans_ft = self.adj_trans(node_ft)
        trans_ft = torch.tanh(trans_ft)
        w = torch.norm(trans_ft, p=2, dim=-1).view(-1, 1)
        w_mat = w * w.t()
        adj = torch.mm(trans_ft, trans_ft.t()) / w_mat
        
        node_ft = self.share_gcn1(node_ft, adj)
        res_mat = res_mat + node_ft

        node_ft = self.activation(res_mat)
        node_ft = F.dropout(node_ft, p=self.dropout, training=self.training)
        node_ft = self.share_gcn2(node_ft, adj)
        res_mat = res_mat + node_ft

        res_mat = self.activation(res_mat)

        global_ft = res_mat[ft_dict['idx']]

        ## global
        ## ----------------------------------------------
        ## local
        
        batch_size = ft_dict['amino_ft'].size()[0]
               
        local_ft = self.cnnmodule(ft_dict['amino_ft']).view(batch_size, -1)

        ## combine
        total_ft = global_ft + local_ft + (global_ft * local_ft) * self.cross_scale_merge
        total_ft = self.activation(total_ft)
        total_ft = F.dropout(total_ft, p=self.dropout, training=self.training)
        
        return total_ft, adj
