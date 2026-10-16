'''
D23 · 模型层

职责（单一）：只有模型。不许读数据、不许写训练循环、不许出现超参数字。

来源（本文件是把 W3 手写的东西搬过来，逻辑一行没改）：
  SelfAttention / causal_mask  ← D16 / D18
  FeedForward                 ← D19
  GPTBlock（Pre-LN + 两条残差） ← D19
  MiniGPT（四段结构 + 权重共享） ← D20

设计约定（D23 知识点 5）：
  · 构造函数只从外面收参数，不在文件里写任何超参数字。
    ⇒ 这样 D27 想换 C=256 做对比实验时，不用动模型代码。
    ⇒ 也刻意不 import config，让本文件能独立 import / 单独测试。
  · 进出的形状契约：MiniGPT.forward 收 (B,T) 整数、吐 (B,T,V) logits
    （不做 softmax —— 训练时 logits 直接交给 CrossEntropyLoss，它自带 log_softmax）

⚠️ 三条最容易静默出错的点（D19/D20 实测，搬过来时别改坏）：
  1. 掩码必须按 x.shape[-2] 动态生成 —— 写死尺寸会在换 T 时报 RuntimeError
  2. 掩码方向：triu(diagonal=1) 屏蔽「未来」；写成 tril 会变成"只能看未来"、末行全 nan
  3. 残差相加必须用 `x = x + ...`（不能写 `x += ...`）—— 原地操作会把调用方的张量
     也改掉，让"同一个 x 用两次"的一切对比实验静默出错
'''
import torch
import torch.nn as nn


def causal_mask(T, device='cpu'):
    '''(T,T) 因果掩码：True = 屏蔽（上三角，不含主对角线）。按输入的 T 动态生成。'''
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class SelfAttention(nn.Module):
    '''单头自注意力（D16/D18）：进出 (B,T,C)，掩码打在 softmax 之前的分数上。'''

    def __init__(self, C):
        super().__init__()
        self.d_k = C
        self.Wq = nn.Linear(C, C, bias=False)
        self.Wk = nn.Linear(C, C, bias=False)
        self.Wv = nn.Linear(C, C, bias=False)
        self.Wo = nn.Linear(C, C, bias=False)

    def forward(self, x, mask):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        S = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        S = S.masked_fill(mask, float('-inf'))
        return self.Wo(torch.softmax(S, dim=-1) @ V)


class FeedForward(nn.Module):
    '''逐位置的两层 MLP（D19）：Linear(C→4C) → ReLU → Linear(4C→C)。'''

    def __init__(self, C):
        super().__init__()
        self.l1 = nn.Linear(C, 4 * C)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(4 * C, C)

    def forward(self, x):
        return self.l2(self.act(self.l1(x)))


class GPTBlock(nn.Module):
    '''完整 Transformer Block（D19）：两个子层，各带 Pre-LN + 残差。'''

    def __init__(self, C):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(C), SelfAttention(C)
        self.ln2, self.ffn = nn.LayerNorm(C), FeedForward(C)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), causal_mask(x.shape[-2], x.device))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    '''四段结构（D20）：token emb + 位置编码 → N × GPTBlock → final LN → lm_head。'''

    def __init__(self, vocab_size, n_embd, block_size, n_layer, tie_weights=True):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)                 # token 查表
        self.pos = nn.Embedding(block_size, n_embd)                 # 位置查表（行数 = 最大长度）
        self.blocks = nn.Sequential(*[GPTBlock(n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)                          # 出模型前最后归一化
        self.head = nn.Linear(n_embd, vocab_size, bias=False)       # C → V（不是 V → C）
        # 权重共享：让 head 的权重指向 embedding 表 —— 必须是 `=` 赋值；
        # 用 .data.copy_() 只是抄一份值、不是共享，而且不会被任何断言发现
        if tie_weights:
            self.head.weight = self.tok.weight
        self.apply(self._init_weights)

    def forward(self, idx):
        # idx: (B,T) 的整数张量
        B, T = idx.shape
        tok_emb = self.tok(idx)                                     # (B,T,C)
        pos_emb = self.pos(torch.arange(T, device=idx.device))      # (T,C)
        x = tok_emb + pos_emb                                       # (B,T,C)
        x = self.blocks(x)                                          # (B,T,C)
        x = self.ln_f(x)                                            # final LN：Pre-LN 的配套件
        logits = self.head(x)                                       # (B,T,V)
        return logits

    def _init_weights(self, module):
        '''
        权重初始化（本课重点）：Linear / Embedding 一律 N(0, 0.02)。

        为什么不能沿用 PyTorch 默认值 —— 三个事实串起来看：
          1. head.weight 与 tok.weight 共享 ⇒ 两者尺度同源
          2. nn.Embedding 默认是 N(0,1)，std ≈ 1
          3. logit = x·w 是 C 维点积，而 final LN 已把 x 归一化到 std≈1
             ⇒ logits.std ≈ √C ≈ 11（实测 11.36）⇒ softmax 近似 one-hot
             ⇒ 初始 loss ≈ 83.7，而 ln(V) 只有 8.64
        改成 0.02 之后：logits.std ≈ √C × 0.02 ≈ 0.23 ⇒ 初始 loss ≈ ln(V) ✅

        LayerNorm 不在此列：它的 γ / β 必须保持默认的 1 / 0。
        '''
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


if __name__ == '__main__':
    model = MiniGPT(5659, 128, 128, 4).to('mps')
    x = torch.randint(0, 5659, (2, 10), device='mps')
    logits = model(x)
    print(logits.shape)  # torch.Size([2, 10, 5659])

    