'''
D19 FeedForward + 完整 GPT Block —— 组装出第一个完整的 Transformer Block

本课的新东西：
  · FFN = 两层 MLP（Linear → ReLU → Linear），先把 C 扩到 4C、再收回来。
    「逐位置」作用：对 (B,T,C) 里每一行分别算，位置之间互不影响 ——
    所以它和 attention 的分工是「位置之内加工」vs「位置之间混合」（D19 知识点 2）。
  · 中间层取 4C 是参数量平衡点：FFN 8C² vs 注意力 4C² ⇒ FFN 占 block 参数 ~2/3。
  · Pre-LN 组装：x = x + 子层(LN(x))，两个子层各一条残差，LN 在子层之前。
  · 因果掩码（D18 学的）今天真正接进注意力。

手算参照（参数量核算，C=8 / 4C=32）：
  FFN      8×32+32 = 288  ；32×8+8 = 264                      → 552
  注意力   4 × (8×8, bias=False) = 256                         → 256
  LN × 2   2 × (2×8)                                           →  32
  合计                                                           840
  （占比 FFN 65.7% / 注意力 30.5% / LN 3.8%；忽略 bias 的粗算是 66.7% / 33.3%）
  对照：GPT-2 small C=768 ⇒ 每层 注意力 2.36M + FFN 4.72M ≈ 7.08M

关键要点（实测避坑 —— 注意前 4 条都是「shape 断言拦不住」的那一类）：
  1. 中间层宽度写成 C（不是 4C）→ 不报错：输出 shape 仍是 (B,T,C)，
     但参数量从 552 掉到 144。shape 断言毫无反应。
  2. 掩码忘了加 → 不报错：shape 一样，但第 0 行对「未来」的权重和达 0.6323（应恰为 0）。
  3. 掩码尺寸与输入的 T 不一致 → RuntimeError: The size of tensor a (5) must match ... (7)
     ⇒ 掩码按 x.shape[-2] 在 forward 里动态生成最稳。
  4. LN 写在残差外面（Post-LN）→ 不报错：shape 一样，但输出被强制压到每行 std = 1.0000
     （Pre-LN 是自由的 0.4636 ~ 1.4405）。
  5. FFN 第二层输出维度写错 → RuntimeError: The size of tensor a (32) must match ... (8)
     （在残差相加时暴露 —— 这条能拦住）。
  6. GELU(-1) = -0.1587，ReLU(-1) = 0；两曲线最大差 0.1587（在 z ≈ ±1）。
     清单写 ReLU，GPT-2 用 GELU —— 今天写哪个都行。
  7. FFN 的两个 Linear 保留 bias（默认就是 True）：和注意力的 Wq/Wk/Wv/Wo 不同，
     这里没有 softmax 会把 bias 吃掉，它是真正的平移自由度。
  8. 测「位置 i 会不会影响位置 j」时，扰动必须打散方向 —— LayerNorm 会把「整行的仿射扰动」
     抹平：整行加常数向量差 3.58e-07、整行乘 2 只剩 eps 效应（3.61e-05），
     加随机向量才有效（2.43）。原因：LN(x) = (x−μ)/√(σ²+eps)，加常数 μ 同增、乘 k 比值不变。
     （顺带说明 γ/β 为什么必需：LN 抹掉了整体平移/缩放，γ/β 是把它学回来的旋钮。）
  9. 掩码断言要「走真模块」：手搓一个 S 再手动 masked_fill，只证明「掩码这个数学操作对」，
     证明不了「SelfAttention 里真的接了掩码」—— 实测把它那行删掉，手搓版断言毫无反应。
 10. 参数量断言 ≠ 接线断言：`sum(p.numel())` 数的是**注册在模块里的 Parameter**，
     跟你 **forward 里有没有用到**完全无关 —— 去掉 Wo、整个 FFN 不接，参数量都还是 840。
     抓「声明了却没接线」要用**梯度检查**：`p.grad is None` ⇒ 这个参数根本没进计算图。

跑出的结论（C=8，torch.manual_seed(0)）：
  · 形状：1 / 2 / 3 层 block 输出都是 (B,T,C)；T=5 与 T=7 都成立（掩码动态生成）。
  · 参数量：block 共 840 = 注意力 256 + FFN 552 + LN×2 32；FFN 占 65.7%（最大头）。
  · 掩码：不打掩码时第 0 行对「未来」的权重和 = 0.6323，打掩码后恰为 0。
'''

import torch
import torch.nn as nn


# ============================================================
# 块 1：FeedForward —— 逐位置的两层 MLP
# ============================================================
class FeedForward(nn.Module):
    '''Linear(C → 4C) → ReLU → Linear(4C → C)：先扩 4 倍再收回来，进出形状相同。'''

    def __init__(self, C, expand=4):
        super().__init__()
        hidden = C * expand
        # expand 参数化：后面「改坏成 expand=1」才好测。
        # 两层 Linear 都保留 bias —— 这里没有 softmax 会把 bias 吃掉，它是真平移自由度（要点 7）。
        self.l1 = nn.Linear(C, hidden, bias=True)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(hidden, C, bias=True)

    def forward(self, x):
        # nn.Linear 只作用最后一维 ⇒ 天然「逐位置」，位置之间互不影响（本课重点）
        fc1 = self.l1(x)
        act = self.act(fc1)
        fc2 = self.l2(act)
        return fc2


# ============================================================
# 块 2：因果掩码 + 单头自注意力（把 D16 / D18 的成果搬过来）
# ============================================================
def causal_mask(T, device=None):
    '''生成 (T,T) 的因果掩码：True = 屏蔽（上三角，不含主对角线）。'''
    # diagonal=1：从主对角线右移一格起取上三角 ⇒ 第 i 行可见 i+1 个位置。
    # 在 forward 里按输入动态生成，T 变了也不怕（要点 3）。
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    return mask


class SelfAttention(nn.Module):
    '''单头自注意力：进出都是 (B,T,C)；掩码作用在 softmax 之前的分数矩阵上。'''

    def __init__(self, C):
        super().__init__()
        self.d_k = C
        # bias=False：K 的 bias 会被 softmax 平移不变性吃掉（D16 知识点 7）
        self.Wq = nn.Linear(C, C, bias=False)
        self.Wk = nn.Linear(C, C, bias=False)
        self.Wv = nn.Linear(C, C, bias=False)
        self.Wo = nn.Linear(C, C, bias=False)

    def forward(self, x, mask):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        # 分数 → 打掩码（填 -inf，不是 0.0）→ softmax → 加权求和 → 投影回 C
        S = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        S = S.masked_fill(mask, float('-inf'))
        A = torch.softmax(S, dim=-1)
        return self.Wo(A @ V)


# ============================================================
# 块 3：完整 GPT Block —— 两个子层，各带 Pre-LN + 残差
# ============================================================
class GPTBlock(nn.Module):
    '''完整 Transformer Block：attn 与 ffn 两个子层，各包一层 LN + 残差。'''

    def __init__(self, C):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.attn = SelfAttention(C)
        self.ln2 = nn.LayerNorm(C)
        self.ffn = FeedForward(C)

    def forward(self, x):
        # Pre-LN：LN 在子层之前，残差加回的是**原始的 x**（通路不穿过任何运算）
        x = x + self.attn(self.ln1(x), causal_mask(x.shape[-2]))
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    B, T, C = 2, 5, 8

    # ---------- FFN：形状 + 参数量 ----------
    x = torch.randn(B, T, C)
    ffn = FeedForward(C)
    print(x.shape, ffn.l1(x).shape, ffn.act(ffn.l1(x)).shape, ffn.l2(ffn.act(ffn.l1(x))).shape)
    assert ffn(x).shape == x.shape
    # 参数量是「中间层有没有写成 4C」的唯一关卡：写成 C 只有 144，shape 断言发现不了（要点 1）
    print(f"ffn 参数量：{sum(p.numel() for p in ffn.parameters())}")
    assert sum(p.numel() for p in ffn.parameters()) == 552

    # ---------- 掩码：数值上真的看不到未来 ----------
    mask = causal_mask(T)
    S = torch.randn(B, T, T)
    A = torch.softmax(S, dim=-1)
    A_masked = torch.softmax(S.masked_fill(mask, float('-inf')), dim=-1)
    # 防 diagonal 写错：写成 0 会让第 0 行整行 -inf，softmax 后全变 nan
    assert not torch.isnan(A_masked).any()
    # 防 -inf 写成 0.0：填 0 时上三角会留下权重（exp(0)=1 被当成强相关）
    assert torch.allclose(A_masked[0, mask], torch.zeros_like(A_masked[0, mask]))
    assert torch.allclose(A_masked.sum(dim=-1), torch.ones_like(A_masked.sum(dim=-1)))
    # 防 diagonal 偏移：第 i 行应恰好可见 i+1 个位置
    for i in range(A_masked.shape[-2]):
        assert (A_masked[0, i, :i+1]>0).sum().item() == i+1
    print(A[0, 0, 1:].sum().item())
    # ⚠️ 以上断言都作用在**手搓的 S**上：只证明「掩码这个数学操作对」，
    #    证明不了「SelfAttention 里真的接了掩码」—— 实测把那行 masked_fill 删掉，这组断言毫无反应（要点 9）

    # ---------- 完整 Block：shape 不变 + 参数量 ----------
    blk = GPTBlock(C)
    for T in [5, 7]:
        x = torch.randn(B, T, C)
        o1 = blk(x)
        o2 = blk(o1)
        o3 = blk(o2)
        print(o1.shape, o2.shape, o3.shape)
        assert o1.shape == x.shape
        assert o2.shape == x.shape
        assert o3.shape == x.shape
    # 参数量管「__init__ 里声明了多少」—— 抓得住「漏声明子模块 / 中间层宽度改」，
    # 但抓不住「声明了却没接线」（那一类要靠下面的梯度检查）
    print(f"blk 参数量：{sum(p.numel() for p in blk.parameters())}")
    assert sum(p.numel() for p in blk.parameters()) == 840
    print(f"ffn 占 {sum(p.numel() for p in blk.ffn.parameters()) / sum(p.numel() for p in blk.parameters()) * 100:.1f}%")

    # ---------- 接线自检：三条「走真模块」的断言 ----------
    # 上面所有断言都作用在手搓的张量上，只证明「数学操作对」；
    # 这三条直接对着组装好的 blk，验的是「到底有没有真接上」。

    # ① 梯度检查 —— 抓「声明了却没接线」（参数量断言看不出来的那一类）
    blk(x).sum().backward()
    missing = [n for n, p in blk.named_parameters() if p.grad is None]
    assert not missing, f'这些参数没进计算图（forward 里漏用了）：{missing}'

    # ② Pre-LN 手工复算 —— 抓「LN 写到残差外面」
    manual = x + blk.attn(blk.ln1(x), causal_mask(T))
    manual = manual + blk.ffn(blk.ln2(manual))
    assert torch.allclose(blk(x), manual, atol=1e-6), 'LN 不在子层之前（写成 Post-LN 了？）'

    # ③ 行为级掩码 —— 抓「forward 里漏了 masked_fill」
    # ⚠️ 扰动必须打散方向：整行加常数会被 LayerNorm 抹平（要点 8），等于白测
    xb = torch.randn(B, T, C)
    xb2 = xb.clone()
    xb2[:, -1, 0] += 1.0
    d = (blk(xb2) - blk(xb)).abs().max(-1).values[0]
    assert (d[:-1] == 0).all(), f'前面的位置不该看到最后位置，实测变化 = {d.tolist()}'

    print('✅ D19 通过：完整 GPTBlock 前向 shape 不变、参数量 840（FFN 占 2/3）')
