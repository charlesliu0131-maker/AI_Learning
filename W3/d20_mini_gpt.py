'''
D20 mini GPT 骨架 —— 从 Block 到模型

本课的新东西：
  · 四段结构：token embedding + 位置编码 → N × GPTBlock → final LayerNorm → lm_head
  · final LN 是 Pre-LN 的配套件：Pre-LN 下 block 的输出从没被归一化过，出模型前补一个
  · 权重共享：lm_head.weight = tok_emb.weight（同一个 Parameter），省参数 + 语义一致
    （logit = x @ Wᵀ，本质是「当前向量与那个词向量的点积 = 相似度」）
  · logits (B,T,V) 是打分不是概率；训练时直接交给 CrossEntropyLoss（它自带 log_softmax）

块 0 是 D19 的成果，直接搬过来（已给全、不用改），今天的活全在块 1。

手算参照（参数量核算，V=27 / C=8 / T_max=16 / N=2）：
  token emb   27×8               =  216
  pos emb     16×8               =  128
  N × block   2 × 840            = 1680      （D19：注意力 256 + FFN 552 + LN×2 32）
  final LN    2×8                =   16
  lm_head     V×C = 216 —— 共享后**不重复计数** ⇒ 0
  合计                              2040
  对照：不共享 = 2256（多了 V×C = 216）；GPT-2 small 同结构 ≈ 124.4M（emb 占 31.7%）

关键要点（实测避坑 —— 前 2 条是「shape 断言拦不住」的那一类）：
  1. 忘了 final LN → 不报错：logits.shape 一点不变（还是 (B,T,V)），参数量也不变。
  2. 输出后多加 softmax → 不报错：shape 一样，但那已经不是 logits 了（D21 训练会算错）。
  3. 权重共享**会掩盖 head 的声明错误**（本课实测新增）：把 head 声明成 nn.Linear(V, C)
     （写反了）照样一切正常 —— 因为共享把 head.weight 换成了 tok.weight（形状恰好 (V,C)），
     而 nn.Linear 前向算的是 x @ weight.T。参数量 ✓、is 判定 ✓、形状 ✓，完全静默。
     抓法：让「不共享」那一版也跑一次前向 —— 没有共享掩护，当场报
     RuntimeError: mat1 and mat2 shapes cannot be multiplied (10x8 and 27x8)。
  4. 权重共享有三种「看着都行」的写法，只有第一种是真的：
       ① head.weight = tok.weight             → is 同一对象 ✅ 真共享
       ② head.weight.data.copy_(tok.weight)   → 只是**抄了一份值**，之后各练各的（静默）
       ③ 不共享                                → 参数量多 V×C
  5. 共享后 parameters() 会自动去重：共享版只有 ['tok.weight']、不共享版是
     ['tok.weight', 'head.weight'] ⇒ 参数量断言能直接反映"到底共享没共享"（2040 vs 2256）。
  6. 共享后别对它做原地初始化：nn.init.normal_(head.weight, ...) 会同时改掉 tok.weight
     （同一块内存，实测：改完之后 tok.weight 跟着变了 = True）。
  7. 真会报错的三种：idx 传 float（RuntimeError）、idx 超出词表（IndexError: index out of
     range in self）、位置索引超出 T_max（同样是 IndexError）。

跑出的结论（C=8, V=27, T_max=16, N=2）：
  · 前向：(B,T) 的整数 idx 进 → (B,T,V) 的 logits 出（实测 (4,9)→(4,9,27)、(2,11)→(2,11,27)）。
  · 参数量 2040 = token emb 216 + pos emb 128 + 2 层 block 1680 + final LN 16 + lm_head 0（共享）。
  · 权重共享省下 V×C = 216（不共享时是 2256）。
'''

import torch
import torch.nn as nn


# ============================================================
# 块 0：D19 的成果，直接搬过来（已给全，不用改）
# ============================================================
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


def causal_mask(T, device=None):
    '''(T,T) 因果掩码：True = 屏蔽（上三角，不含主对角线）。按输入的 T 动态生成。'''
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


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


# ============================================================
# 块 1：MiniGPT —— 四段结构
# ============================================================
class MiniGPT(nn.Module):
    '''四段结构：token emb + 位置编码 → N × GPTBlock → final LN → lm_head。'''

    def __init__(self, vocab_size, C, T_max, N, tie_weights=True):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, C)                     # token 查表
        self.pos = nn.Embedding(T_max, C)                          # 位置查表（行数 = 最大长度）
        self.blocks = nn.Sequential(*[GPTBlock(C) for _ in range(N)])
        self.ln_f = nn.LayerNorm(C)                                # 出模型前最后归一化
        self.head = nn.Linear(C, vocab_size, bias=False)           # C → V（不是 V → C）
        # 权重共享：让 head 的权重「指向」embedding 表 —— 必须是 `=` 赋值，
        # 用 .data.copy_() 只是抄一份值、不是共享（要点 4）
        if tie_weights:
            self.head.weight = self.tok.weight

    def forward(self, idx):
        # idx: (B,T) 的整数张量
        B, T = idx.shape
        tok_emb = self.tok(idx)                                    # (B,T,C)
        pos_emb = self.pos(torch.arange(T, device=idx.device))     # (T,C)
        x = tok_emb + pos_emb                                      # (B,T,C)
        x = self.blocks(x)                                         # (B,T,C)
        x = self.ln_f(x)                                           # final LN：Pre-LN 的配套件
        # 这里**不 softmax**：logits 是打分，训练时直接交给 CrossEntropyLoss（它自带 log_softmax）
        logits = self.head(x)                                      # (B,T,V)
        return logits


if __name__ == '__main__':
    torch.manual_seed(0)
    # V = 词表大小，C = embedding 维度，T_max = 最大序列长度，N = block 层数
    V, C, T_max, N = 27, 8, 16, 2

    # ---------- 前向：logits 形状（今天的验收标准）----------
    idx = torch.randint(0, V, (4, 9))
    m = MiniGPT(V, C, T_max, N)
    out = m(idx)
    print(f"idx {tuple(idx.shape)} → logits {tuple(out.shape)}")
    assert out.shape == (4, 9, V)

    idx2 = torch.randint(0, V, (2, 11))
    out2 = m(idx2)
    print(f"idx {tuple(idx2.shape)} → logits {tuple(out2.shape)}")
    assert out2.shape == (2, 11, V)          # 换一组 (B,T) 仍然成立

    # ---------- 参数量账 ----------
    print("参数量账：")
    tok_num = sum(p.numel() for p in m.tok.parameters())
    pos_num = sum(p.numel() for p in m.pos.parameters())
    blocks_num = sum(p.numel() for p in m.blocks.parameters())
    ln_f_num = sum(p.numel() for p in m.ln_f.parameters())
    head_num = sum(p.numel() for p in m.head.parameters())
    print(f"tok: {tok_num}, pos: {pos_num}, blocks: {blocks_num}, ln_f: {ln_f_num}, head: {head_num}")
    # head_num 单独数仍是 216（它就是 tok.weight 本身），但总计会去重 ⇒ 2040（要点 5）
    assert sum(p.numel() for p in m.parameters()) == 2040
    m_false = MiniGPT(V, C, T_max, N, tie_weights=False)
    m_false(idx2)                            # 不共享的也跑一次前向：没有共享掩护，head 声明写反会当场报错（要点 3）
    assert sum(p.numel() for p in m_false.parameters()) == 2256
    assert m.head.weight is m.tok.weight     # 共享是不是**真**的（.data.copy_() 的假共享会在这里炸）

    # ---------- 两条「走真模块」的断言 ----------
    # ① 梯度检查：每个注册参数都必须进过计算图（抓「声明了却没接线」）
    m(idx).sum().backward()
    assert [n for n, p in m.named_parameters() if p.grad is None] == []

    # ② 因果性：改**最后一个位置的单个分量**，前面所有位置的 logits 必须纹丝不动。
    #    ⚠️ 必须改单个分量 —— 整行加常数会被 LayerNorm 抹平（LN 对整行仿射近似不变，差 ~3.6e-07），等于白测。
    #    这条断言还带一层附加价值：右端 m(idx2) 走的是**真 forward**，
    #    所以 forward 里漏了 final LN / 多加了 softmax / 漏了位置编码，都会在这里当场暴露。
    x2 = m.tok(idx2) + m.pos(torch.arange(idx2.shape[-1], device=idx2.device))
    x2[:, -1, 0] += 1.0
    logits_p = m.head(m.ln_f(m.blocks(x2)))
    assert (logits_p[:, :-1] == out2[:, :-1]).all()

    # ---------- 跑出的结论 ----------
    # 输入 (B,T) 的整数张量 → 输出 (B,T,V) 的 logits。
    # 参数量 2040 由这几块构成：token embedding 216 + position embedding 128 + 2 层 block 1680
    #                         + final layer norm 16 + lm head 0（共享后不重复计数）。
    # 权重共享省下 V×C = 216 个参数（不共享时是 2256）。
    print('✅ D20 通过：mini GPT 前向 (B,T) → (B,T,V)，参数量 2040（权重共享，省 216）')
