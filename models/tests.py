from self_attention import SelfAttention, MultiHeadSelfAttention, MultiHeadSelfAttentionEfficient, Attention
import torch
from torch import nn
import copy

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, eff):
        super(Block, self).__init__()
        if eff:
            self.attn = MultiHeadSelfAttentionEfficient(in_dim, out_dim, num_heads)
        else:
            self.attn = MultiHeadSelfAttention(in_dim, out_dim, num_heads)
    def forward(self, input):
        return self.attn(input)

class Model(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, eff):
        super(Model, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(5):
            layer = Block(in_dim, out_dim, num_heads, eff)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        return hidden_states

def main():
    DEVICE = 'cuda'
    BATCH = 16
    IN_DIM = 200
    OUT_DIM = 200
    NUM_SAMPLES = 50
    attn_head = SelfAttention(IN_DIM, OUT_DIM).to(DEVICE)

    input = torch.rand(BATCH, NUM_SAMPLES, IN_DIM).to(DEVICE)
    output = attn_head.forward(input)
    assert output.shape[0] == BATCH
    assert output.shape[1] == NUM_SAMPLES
    assert output.shape[2] == OUT_DIM

    NUM_HEADS = 10
    HEAD_DIM = 20
    multihead_attn = MultiHeadSelfAttention(IN_DIM, OUT_DIM, NUM_HEADS, HEAD_DIM).to(DEVICE)
    output = multihead_attn.forward(input)
    count_params(multihead_attn)

    assert output.shape[0] == BATCH
    assert output.shape[1] == NUM_SAMPLES
    assert output.shape[2] == OUT_DIM

    multihead_attn = MultiHeadSelfAttentionEfficient(IN_DIM, OUT_DIM, NUM_HEADS, HEAD_DIM).to(DEVICE)
    output = multihead_attn.forward(input)
    count_params(multihead_attn)

    assert output.shape[0] == BATCH
    assert output.shape[1] == NUM_SAMPLES
    assert output.shape[2] == OUT_DIM

    multihead_attn = Attention(IN_DIM, NUM_HEADS).to(DEVICE)
    output = multihead_attn.forward(input)
    count_params(multihead_attn)

    assert output.shape[0] == BATCH
    assert output.shape[1] == NUM_SAMPLES
    assert output.shape[2] == OUT_DIM

    # model = Model(IN_DIM, OUT_DIM, NUM_HEADS, False).to(DEVICE)
    #
    # output = model.forward(input)
    #
    # assert output.shape[0] == BATCH
    # assert output.shape[1] == NUM_SAMPLES
    # assert output.shape[2] == OUT_DIM
    #
    #
    # model = Model(IN_DIM, OUT_DIM, NUM_HEADS, True).to(DEVICE)
    #
    # output = model.forward(input)
    #
    # assert output.shape[0] == BATCH
    # assert output.shape[1] == NUM_SAMPLES
    # assert output.shape[2] == OUT_DIM

def count_params(module):
    sum = 0
    for parameter in module.parameters():
        mul = 1
        for num in parameter.shape:
            mul *= num
        sum += mul
    print('Number of parameters:', sum)

main()