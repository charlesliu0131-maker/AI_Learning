'''
D18 因果掩码 + 位置编码 —— 让注意力「看不到未来」，并把"顺序"灌进去

本课的新东西：
  · 因果掩码：把分数矩阵的【上三角】填 -inf → softmax 后未来位置权重恰为 0
    （softmax 里 exp(-inf) = 0。⚠️ 填 0 是灾难：exp(0) = 1 在 softmax 里是"强相关"）
  · 位置编码：attention 本身是【置换等变】的（D16-R2：打乱输入，输出只是跟着搬）
    ⇒ 必须先打破它，模型才知道顺序。做法是把一张 (T_max, C) 的位置表【加到】输入上。
  · 三种方案（今天只实现第 1 种）：可学习（GPT-2）/ 正弦（原论文）/ RoPE（只在 Q/K 上旋转）

手算参照（本脚本取 B=2、T=5、C=8、d_k=3）：
  掩码:    S(B,T,T) → 上三角置 -inf → softmax(dim=-1) → A(B,T,T)
           第 i 行可见 i+1 个位置（含自己）；上三角恰为 0；每行和仍为 1
  位置:    x(B,T,C) + PE[:T](T,C) → (B,T,C)     ← 靠广播；同一个词在不同位置 = 不同输入
  幅度:    词向量 std ≈ 0.02（GPT-2 风格）；位置表也要同量级（本脚本用 0.01）——
           若用 nn.Embedding 默认的 N(0,1)（实测 std 1.0011），位置会【压过】内容

关键要点（避坑，全部实测过，数字可直接引用）：
  · diagonal=1 才对：写 0 或 -1 会把第 0 行【整行】屏蔽 → 该行 nan（连自己都看不到）
  · mask 必须是 bool 张量：float 会报
    RuntimeError: masked_fill_ only supports boolean masks, but got mask with dtype float
  · 上三角填 -inf，不要填 0.0（实测：填 0 时第 0 行对未来的权重和 = 0.9858）
  · (T,T) 的 mask 能直接广播到多头的 (B,h,T,T)（实测 (2,4,5,5)、上三角全 0）⇒ D19 可复用
  · 位置表长度必须 ≥ T，否则 IndexError: index out of range in self
  · 验证"位置真的进来了"：打乱【内容】、位置码仍【按槽位】贴 → 置换等变性应被打破
  · 位置码保真检查：同一个词放两个不同位置，输出必须不同（否则 PE 根本没接上）

跑出的结论（2026-10-08 实跑，seed=1、B=2/T=5/C=8/d_k=3）：
  掩码：上三角恰为 0；第 i 行可见 i+1 个位置；每行和仍为 1；无 nan
        不打掩码时第 0 行对"未来"的权重和 = 0.8738（最大权重压在 j=2）⇒ 就是在偷看答案
  位置：无 PE 时置换等变成立（Ap == A[perm][:,perm]，基线对照）；
        加 PE 后不再成立 —— 等变性被打破的最大差 = **3.2e-03**，位置表 std = **0.0102**
        ⇒ 位置信息确实进来了（幅度小是因为 PE 本身只有 0.01 量级）
'''
import torch
import torch.nn as nn


# ============================================================
# 块 1｜因果掩码
# ============================================================
def causal_mask(T):
    """返回 (T,T) 的 bool 掩码：True = 该屏蔽（即"未来"位置）

    diagonal=1 = 从主对角线右一格开始屏蔽 ⇒ 第 i 行留 i+1 个位置（含自己）。
    写 0 或 -1 会把第 0 行整行屏蔽 → softmax 全 -inf → 该行 nan。
    """
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    return mask


def apply_causal_mask(S, mask):
    """把掩码作用在分数矩阵上；S 放宽到 (..., T, T)，这样 (B,T,T) 和 (B,h,T,T) 都能用

    掩码值是 -inf、不是 0.0：softmax 里 exp(0) = 1 是"强相关"，只有 -inf 才真的抹掉。
    """
    masked = S.masked_fill(mask, float('-inf'))
    return masked


# ============================================================
# 块 2｜可学习位置编码
# ============================================================
class LearnedPositionalEncoding(nn.Module):
    """一张 (T_max, C) 的表，前向时取前 T 行【加到】输入上"""

    def __init__(self, T_max, C, std=0.01):
        super().__init__()
        self.emb = nn.Embedding(T_max, C)
        # 必须自己重置幅度：nn.Embedding 默认 N(0,1)，比词向量大 ~50 倍 ⇒ 位置会盖过内容。
        # std=0.01 让位置表与词嵌入同量级 —— 位置是"叠加"上去的，不是"覆盖"内容。
        with torch.no_grad():
            self.emb.weight.normal_(0, std)

    def forward(self, x):
        T = x.shape[-2]
        out = x + self.emb(torch.arange(T))
        return out


class Score(nn.Module):
    """只算分数矩阵 S = Q@Kᵀ/√d_k（本脚本用它造"待掩码"的 S，不含 softmax）"""

    def __init__(self, C, d_k):
        super(Score, self).__init__()
        self.d_k = d_k
        # bias=False：Q/K 的偏置会被 softmax 的平移不变性吃掉，所以省掉；
        self.Wq = nn.Linear(C, d_k, bias=False)
        self.Wk = nn.Linear(C, d_k, bias=False)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)

        S = Q @ K.transpose(-2, -1) / self.d_k**0.5
        return S


# ============================================================
# 自测区
# ============================================================
if __name__ == '__main__':
    torch.manual_seed(1)
    B, T, C = 2, 5, 8

    # ---------- 块 1：掩码（验收标准①：数值上"看不到未来"）----------
    d_k = 3

    X = torch.randn(B, T, C)
    score = Score(C, d_k)

    # .detach() 只是不建计算图（Wq/Wk 是 Parameter ⇒ S 本来 requires_grad）；
    # 本脚本不需要梯度，留着无妨 —— 但 D19 要反向传播时别在这里切断。
    S = score(X).detach()
    A = torch.softmax(S, dim=-1)
    mask = causal_mask(T)
    A_masked = torch.softmax(apply_causal_mask(S, mask), dim=-1)

    assert A_masked[..., mask].abs().max() == 0            # 防：-inf 写成 0.0 / 漏打掩码
    for i in range(A_masked.shape[-2]):
        assert (A_masked[0, i] > 0).sum() == i + 1         # 防：diagonal 写错（可见位置数不对）
    assert torch.allclose(A_masked.sum(-1), torch.ones_like(A_masked.sum(-1)))   # 归一化没被破坏
    assert not torch.isnan(A_masked).any()                 # 防：某行被整行屏蔽（diagonal=0/-1）

    # ---------- 块 2：位置编码（验收标准②：位置编码实现正确）----------
    pe = LearnedPositionalEncoding(T_max=16, C=C)
    x_pe = pe(X)
    assert x_pe.shape == X.shape                           # 加位置不改形状
    # 保真检查：同一个词放到两个不同位置 ⇒ 结果必须不同（否则 PE 根本没接上）
    X_double = torch.cat([X, X], dim=-2)
    x_pe_double = pe(X_double)
    assert not torch.allclose(x_pe_double[:, 0], x_pe_double[:, X.shape[-2]])
    print("词向量 std:", X.std().item(), "位置表 std:", pe.emb.weight.std().item())

    # ---------- 块 2 加测：置换等变性被打破 ----------
    def attn(score, X):
        S = score(X).detach()
        A = torch.softmax(S, dim=-1)
        return A

    perm = torch.randperm(T)
    # 无 PE：置换等变成立（D16-R2 已证过，这里作对照基线）
    assert torch.allclose(attn(score, X[:, perm]), attn(score, X)[:, perm][...,perm])

    # 有 PE：内容打乱、位置码【按槽位】贴（第 i 个槽位永远拿 PE[i]）⇒ 等变性应被打破
    #   ⚠️ 两个坑：① 若写成"位置码跟着词一起搬家"，等变性照样成立、测不出效果
    #              ② 这条断言对【容差】敏感：PE 幅度只有 0.01，破坏量级才 3.2e-03 ——
    #                 若给 allclose 加 atol=1e-2 就会判成"相等"⇒ 误报
    x_pe = pe(X)
    x_pe_perm = pe(X[:,perm])
    assert not torch.allclose(attn(score, x_pe_perm), attn(score, x_pe)[:, perm][...,perm])

    print('✅ D18 通过：掩码后上三角恰为 0、第 i 行可见 i+1 个；位置编码打破了置换等变性')
