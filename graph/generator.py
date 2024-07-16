
import numpy as np
import torch
from torch_geometric import data as DATA


def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y

def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj

def getAffinityGraph(dataset, adj, weighted, drug_aff_k, target_aff_k): # 构造亲和力异构图，包括初始features和edge_indexs和edge_weights
    num_drugs = adj.shape[0] # 68
    num_targets = adj.shape[1] # 442
    
    if dataset == "davis":
        adj[adj != 0] -= 5 #范围5.0-10.8-》0.0-5.8
        adj_norm = minMaxNormalize(adj, 0) # 最大最小归一化
    elif dataset == "kiba":
        adj_refine = denseAffinityRefine(adj.T, target_aff_k)
        adj_refine = denseAffinityRefine(adj_refine.T, drug_aff_k)
        adj_norm = minMaxNormalize(adj_refine, 0)
        
    adj_1 = adj_norm
    adj_2 = adj_norm.T

    adj = np.concatenate((
        np.concatenate((np.zeros([num_drugs, num_drugs]), adj_1), 1), #按列拼接，  （68,68）（68,442）-》 (68, 510)
        np.concatenate((adj_2, np.zeros([num_targets, num_targets])), 1) #按列拼接  (442, 68) （442,442）-》 (442, 510)
    ), 0) #按行拼接 (68, 510)(442, 510) -》(510, 510)

    train_raw_ids, train_col_ids = np.where(adj != 0) # array([  0,   0,   0, ..., 509, 509, 509])shape: (15166,)； array([68, 92, 93, ..., 58, 60, 66])shape: (15166,)
    edge_indexs = np.concatenate((
        np.expand_dims(train_raw_ids, 0), # (15166,)-》(1, 15166)
        np.expand_dims(train_col_ids, 0)  # (15166,)-》(1, 15166)
    ), 0) # (1, 15166) (1, 15166) -》(2, 15166)
    edge_weights = adj[train_raw_ids, train_col_ids] # array([0.40831272, 0.36563669, 0.09275589, ..., 0.13731824, 0.07250261,   0.07866484]) shape: (15166,)
    # np.tile(a,(2,1))第一个参数2为Y轴扩大倍数，第二个1为X轴扩大倍数
    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drugs, 1)), # Y轴num_drugs倍数，X轴扩大1倍数。(68, 2) [0:68] : [array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), array([1, 0]), ...]
        np.tile(np.array([0, 1]), (num_targets, 1)) # Y轴num_targets倍数，X轴扩大1倍数shape: (442, 2)
    ), 0) # shape: (510, 2)
    
    drEMBED = np.loadtxt('data/davis/Dr_seq2seq_EMBED.txt') #输出什么样？
    prEMBED = np.loadtxt('data/davis/Pr_ProtVec_EMBED.txt') #输出什么样？
    
    adj_features = np.zeros_like(adj) # shape: (510, 510)
    adj_features[adj != 0] = 1 # 有亲和力值的都为1 shape: (510, 510)

    features = np.concatenate((node_type_features, adj_features), 1) # 按列拼接，(510, 2)(510, 510)-》(510, 512)
    # # 带权的话即weighted，adj为最大最小化的亲和力值矩阵adj，否则为01化的矩阵adj_features
    # affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj), edge_index=torch.LongTensor(edge_indexs)) if weighted \
    #     else DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj_features), edge_index=torch.LongTensor(edge_indexs))
    affinity_graph = DATA.Data(features=torch.Tensor(features), drEMBED = torch.Tensor(drEMBED), prEMBED=torch.Tensor(prEMBED), adj=torch.Tensor(adj), edge_index=torch.LongTensor(edge_indexs)) if weighted \
    else DATA.Data(drEMBED = torch.Tensor(drEMBED), prEMBED=torch.Tensor(prEMBED), adj=torch.Tensor(adj_features), edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_node1s", num_drugs)
    affinity_graph.__setitem__("num_node2s", num_targets)
    return affinity_graph