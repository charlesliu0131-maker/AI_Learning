# 默写GPTBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T):
    return torch.tril(torch.ones((T, T), dtype=torch.bool))

class Attention(nn.Module):
    
    def __init__(self, C, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(C, d_k, bias=False)
        self.Wk = nn.Linear(C, d_k, bias=False)
        self.Wv = nn.Linear(C, d_v, bias=False)
        self.Wo = nn.Linear(d_v, C, bias=False)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        S = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        S = S.masked_fill(causal_mask(S.shape[-2]) == 0, float('-inf'))
        A = torch.softmax(S, dim=-1)
        y = A @ V
        y = self.Wo(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, C, d_ff):
        super().__init__()
        self.l1 = nn.Linear(C, d_ff)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(d_ff, C)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, C, d_k, d_v):
        super().__init__()
        self.attention = Attention(C, d_k, d_v)
        self.ln1 = nn.LayerNorm(C)
        self.ff = FeedForward(C, d_ff=4 * C)
        self.ln2 = nn.LayerNorm(C)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

if __name__ == '__main__':
    B, T, C = 2, 4, 16
    d_k, d_v = 8, 12

    x = torch.randn(B, T, C)
    block = GPTBlock(C, d_k, d_v)
    out = block(x)
    print(out.shape)