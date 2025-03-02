# copy from https://github.com/SpringtoString/DeepFM_torch and Make some adaptive modification

# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os

import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from collections import OrderedDict, namedtuple, defaultdict
import random


# 添加 dropout 和 L2正则项 添加后与deepctr还是有区别 L2正则项需要继续改进
# init()代码太冗杂了，封装几个函数

class Deepfm(nn.Module):

    def __init__(self, feat_sizes, sparse_feature_columns, dense_feature_columns,dnn_hidden_units=[400, 400,400], dnn_dropout=0.0, ebedding_size=4,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 device='cpu'):

        super(Deepfm, self).__init__()
        self.feat_sizes = feat_sizes
        self.device = device
        self.sparse_feature_columns = sparse_feature_columns
        self.dense_feature_columns = dense_feature_columns
        self.embedding_size = ebedding_size
        self.l2_reg_linear = l2_reg_linear

        # self.feature_index 建立feature到列名到输入数据X的相对位置的映射
        # self.feature_index = self.build_input_features(self.feat_sizes)
        # 数据是按照sparse_column..., dense_column排列的
        self.feature_index = self.build_input_features(self.sparse_feature_columns+self.dense_feature_columns)

        self.bias = nn.Parameter(torch.zeros((1,)))
        # self.weight
        self.weight = nn.Parameter(torch.Tensor(len(self.dense_feature_columns), 1)).to(device)
        torch.nn.init.normal_(self.weight, mean=0, std=0.0001)

        self.embedding_dict1 = self.create_embedding_matrix(self.sparse_feature_columns , feat_sizes , 1 ,
                                                       sparse=False, device=self.device)

        self.embedding_dict2 = self.create_embedding_matrix(self.sparse_feature_columns , feat_sizes , self.embedding_size ,
                                                       sparse=False, device=self.device)
        # dnn
        self.dropout = nn.Dropout(dnn_dropout)
        self.dnn_input_size = self.embedding_size * len(self.sparse_feature_columns) + len(self.dense_feature_columns)
        hidden_units = [self.dnn_input_size] + dnn_hidden_units
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        self.relus = nn.ModuleList(
            [nn.ReLU() for i in range(len(hidden_units) - 1)])
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        # self.linears =self.linears.to(device)
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(device)

        self.to(device)

    def forward(self, X):
        '''
        :param X: pd.DtateFrame
        :return:  y_pre
        '''

        '''
            FM liner
        '''
        # sparse_feature_number * batch_size * 1 * 1
        sparse_embedding_list1 = [self.embedding_dict1[feat](
            X[:, self.feature_index[feat][0]:self.feature_index[feat][1]].long()) 
            for feat in self.sparse_feature_columns]

        # dense_feature_number * batch_size * 1
        dense_value_list2 = [X[:, self.feature_index[feat][0]:self.feature_index[feat][1]]
                             for feat in self.dense_feature_columns]
        
        # batch_size * 1
        linear_sparse_logit = torch.sum(
            # batch_size  * 1 * sparse_feature_number
            torch.cat(sparse_embedding_list1, dim=-1), dim=-1, keepdim=False)
        
        # batch_size * 1 
        linear_dense_logit = torch.cat(
            dense_value_list2, dim=-1).matmul(self.weight)
        logit = linear_sparse_logit + linear_dense_logit

        # sparse_feature_number * batch_size * 1 * 8
        sparse_embedding_list = [self.embedding_dict2[feat](
            X[:, self.feature_index[feat][0]:self.feature_index[feat][1]].long())
            for feat in self.sparse_feature_columns]
        
        '''
            FM second
        '''
        fm_input = torch.cat(sparse_embedding_list, dim=1)  # shape: (batch_size,field_size,embedding_size)
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)  # shape: (batch_size,1,embedding_size)
        sum_of_square = torch.sum(torch.pow(fm_input, 2), dim=1, keepdim=True)  # shape: (batch_size,1,embedding_size)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # shape: (batch_size,1)
        logit += cross_term

        '''
            DNN
        '''
        #  sparse_embedding_list、 dense_value_list2
        dnn_sparse_input = torch.cat(sparse_embedding_list, dim=1)
        batch_size = dnn_sparse_input.shape[0]
        # print(dnn_sparse_input.shape)
        dnn_sparse_input=dnn_sparse_input.reshape(batch_size,-1)
        # dnn_sparse_input shape: [ batch_size, len(sparse_feat)*embedding_size ]
        dnn_dense_input = torch.cat(dense_value_list2, dim=-1)
        # print(dnn_sparse_input.shape)
        # dnn_dense_input shape: [ batch_size, len(dense_feat) ]
        dnn_total_input = torch.cat([dnn_sparse_input, dnn_dense_input], dim=-1)
        deep_input = dnn_total_input

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            fc = self.relus[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        dnn_output = self.dnn_linear(deep_input)

        logit += dnn_output
        '''
            output
        '''
        return logit + self.bias
        # y_pred = torch.sigmoid(logit+self.bias)
        # return y_pred

    def build_input_features(self, feat_sizes):
        # Return OrderedDict: {feature_name:(start, start+dimension)}
        features = OrderedDict()
        start = 0
        for feat in feat_sizes:
            feat_name = feat
            if feat_name in features:
                continue
            features[feat_name] = (start, start + 1)
            start += 1
        return  features

    def create_embedding_matrix(self ,sparse_feature_columns, feat_sizes,embedding_size,init_std=0.0001, sparse=False, device='cpu'):
        embedding_dict = nn.ModuleDict(
            {feat: nn.Embedding(feat_sizes[feat], embedding_size, sparse=False)
             for feat in sparse_feature_columns}
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        return embedding_dict.to(device)