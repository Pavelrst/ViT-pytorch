import torch
from torch import nn
import numpy as np
import copy

class MultiHeadSelfAttentionEfficient(nn.Module):
    '''
    My implementation of a multi-head self attention.
    '''
    def __init__(self, input_dim, output_dim, heads_num, head_dim=15):
        super(MultiHeadSelfAttentionEfficient, self).__init__()
        self.attn_multi_head = SelfAttention(input_dim, head_dim * heads_num)
        self.w_out = nn.Linear(head_dim * heads_num, output_dim)

    def forward(self, input):
        return self.w_out(self.attn_multi_head.forward(input))


class MultiHeadSelfAttention(nn.Module):
    '''
    My implementation of a multi-head self attention.
    '''
    def __init__(self, input_dim, output_dim, heads_num, head_dim=15):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads_list = nn.ModuleList([])
        self.head_dim = head_dim
        for _ in range(heads_num):
            attn_head = SelfAttention(input_dim, self.head_dim)
            self.heads_list.append(copy.deepcopy(attn_head))

        self.w_out = nn.Linear(self.head_dim * heads_num, output_dim)

    def forward(self, input):
        head_out_list = []
        for attn_head in self.heads_list:
            head_out_list.append(attn_head.forward(input))
        concated_outputs = torch.cat(head_out_list, dim=2)
        return self.w_out(concated_outputs)


class SelfAttention(nn.Module):
    '''
    My implementation of a single self-attention head.
    '''
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        # Initialize transformation matrices
        # which generates the Q, K, V
        # torch.nn.Linear(int in_features ,int out_features)
        self.w_query = nn.Linear(input_dim, output_dim)
        self.w_key = nn.Linear(input_dim, output_dim)
        self.w_value = nn.Linear(input_dim, output_dim)
        self.sqrt_dim_key = np.sqrt(output_dim)
        self.softmax = nn.Softmax(dim=1) # softmax along rows

    def forward(self, input):
        Q = self.w_query(input)
        K = self.w_key(input)
        V = self.w_value(input)

        #                           /  Q  x  K_transpose   \
        # output = Softmax_on_rows { ---------------------- } x V
        #                           \    sqrt( dim_key )   /
        KT = torch.transpose(K, 2, 1)
        QKT = torch.matmul(Q, KT)
        QKT_dk = torch.div(QKT, self.sqrt_dim_key)
        QKT_dk_soft = self.softmax(QKT_dk)
        output = torch.matmul(QKT_dk_soft, V)

        return output