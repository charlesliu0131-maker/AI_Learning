'''
D17 LayerNorm + 残差 —— 给 D16 的注意力层补上「归一化 / 残差」，做成能摞 N 层的子层

> 学习日标题是「Multi-Head + LayerNorm + 残差」。**多头在本脚本里【不实现】**（峰哥 2026-09-29 决定）：
> 原理与形状链已在概念笔记 `概念笔记/W3/D17-Multi-Head与LayerNorm与残差.md` 登记，
> 代码留到 D19 组装完整 GPT Block 时再写。今天只做 **LayerNorm + 残差 + 子层封装**。

本课的新东西：
  · 残差：out = x + f(x)，∂out/∂x = 1 + f'(x) —— 那个 1 是恒等通路，
    再深也有梯度直连（无残差时 0.9^50 = 5.154e-03、0.5^50 = 8.882e-16 ⇒ 梯度消失）。
  · LayerNorm：对每个 token 的 C 维向量做标准化（每行 mean≈0、std≈1），再学 γ、β。
    统计方向是「同一个 token 的所有特征」⇒ 不依赖 batch（BN 依赖，所以序列任务用 LN）。
  · 形状铁律：子层进出必须都是 (B,T,C) —— 否则残差【加不上】、也摞不起来。
    （D16 里那个 self.Wo 就是为了这件事；d_v≠C 时直接相加会 RuntimeError。）
  · （只记不写）多头：把 C 切成 h 份、每头一套独立 Wq/Wk/Wv。参数量与单头**一样多**
    （单头 4×C²；多头 h×(4×C×d_k) = 4×C²），换来 h 个独立子空间。

手算参照（本脚本的形状链，取 C=8、B=2、T=5）：
  子层:      (2,5,8) → LN → (2,5,8)（每行 mean≈0、std≈1）→ attn → (2,5,8) → +x → (2,5,8)
  残差摞层:  depth=50、a=0.9 时  无残差 ∂out/∂x = a^depth = 5.154e-03
                                 有残差 ∂out/∂x = Σa^k = 9.954
  参数量:    本脚本只用到自注意力那部分：Wq/Wk/Wv/Wo 各 C×C = 64 ⇒ 4×64 = 256（C=8）

关键要点（避坑，全部实测过，数字可直接引用）：
  · var 必须 unbiased=False（总体方差）：用 True 与 nn.LayerNorm 差 2.218e-01，而且【不会报错】
  · dim 一律写 -1（D16 铁律）：写成 0 时每行 mean 偏离 0.9995，形状却完全正常
  · eps 必须加在【开方里面】：√(var+eps)。挪到外面与 nn.LayerNorm 差 3.922e-05（> 1e-6 容差）
  · mean/var 必须 keepdim=True，否则 RuntimeError（维度不匹配，报错信息容易看懵）
  · γ、β 必须是 nn.Parameter，初值 γ=1 / β=0（初始即「纯归一化」，训练中可选择性放弃）
  · 残差要求形状一致：d_v≠C 直接相加 → RuntimeError（这就是 D16 加 Wo 的原因）

跑出的结论（2026-09-29 实跑，seed=1）：
  手写 LN 与 nn.LayerNorm 最大绝对差 2.384e-07（allclose(atol=1e-6) 通过）；每行 mean ≈ 2.98e-08、std ≈ 1
  残差摞 50 层：有残差 |∂out/∂x| = 9.9536（≈ Σ0.9ᵏ = 9.9536）；无残差 = 5.154e-03（≈ 0.9⁵⁰ = 5.1538e-03）
  ⇒ 有残差时梯度保底、与深度无关；无残差时按 a^depth 指数衰减。子层进出均为 (2,5,8) ⇒ 能摞层。
'''
import torch
import torch.nn as nn


# ============================================================
# 块 1｜手写 LayerNorm（今天的验收标准就落在这里）
# ============================================================
class MyLayerNorm(nn.Module):
    """对【最后一维】归一化 + 可学习 γ、β"""

    def __init__(self, C, eps=1e-5):
        super().__init__()
        self.eps = eps
        # γ、β 必须是 nn.Parameter，否则不进 parameters()、优化器拿不到（接 D06）
        # 初值 γ=1、β=0 ⇒ 一出生就是「纯归一化」，训练中可以选择性放弃它
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        # dim=-1 = 沿【特征维】统计：一次归一化只用一个 token 自己的 C 个数（BN 才跨样本）
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False（总体方差）才能和 nn.LayerNorm 对齐：用 True 会差 C/(C-1) 倍且不报错
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        # eps 在【开方里面】：作用是防 var=0 时除零；挪到开方外面就不是它了（实测差 3.9e-05）
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        # γ、β 形状 (C,)，自动广播到最后一维 —— 归一化"归掉"的尺度由它们重新学回来
        out = self.gamma * normalized + self.beta
        return out


# ============================================================
# 块 2｜残差包装：out = x + sublayer(x)
# ============================================================
class ShrinkSublayer(nn.Module):
    """故意做成「每层都缩小」的子层 —— 用来看残差到底救了什么"""

    def __init__(self, dim, a=0.9):
        super().__init__()
        self.a = a
        # 用一个 Linear + 单位阵×a 来实现"纯缩放"：既是真模块、真参数（比标量乘更贴近实际），
        # 又让理论上界能逐项核对：每层乘 a ⇒ ∂out/∂x = a^depth
        self.shrink = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.shrink.weight.copy_(torch.eye(dim) * a)

    def forward(self, x):
        return self.shrink(x)


def run_stack(x, sublayer, depth, use_residual):
    """把 sublayer 摞 depth 层；use_residual=True 时每层都把输入加回来"""
    h = x
    for _ in range(depth):
        h = sublayer(h)
        if use_residual:
            # 这一行就是「残差」：它给梯度开了一条恒等通路，∂(x+f)/∂x = 1 + f' ≥ 1
            h = h + x
    return h


# ============================================================
# 块 3｜把 D16 的 attention 包成「带残差 + norm」的子层
# ============================================================
class SelfAttention(nn.Module):
    """D16 那版原样搬过来（含 Wo、含两条断言）"""

    def __init__(self, C, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        # 四个投影都 bias=False：Q/K 的偏置会被 softmax 的平移不变性吃掉（D16 实测 1.19e-07）
        self.Wq = nn.Linear(C, d_k, bias=False)
        self.Wk = nn.Linear(C, d_k, bias=False)
        self.Wv = nn.Linear(C, d_v, bias=False)
        # Wo 把 d_v 映回 C —— 这正是块 3 的残差能"加得上"、以及能摞 N 层的前提
        self.Wo = nn.Linear(d_v, C, bias=False)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        # 缩放：q·k 是 d_k 项相加、标准差 ≈ √d_k，除以它把分数尺度拉回 O(1)，防 softmax 饱和
        S = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        # 防「拆维/转置写错」：截到 (B,T,T) 才是"两两关系表"
        assert S.shape == (x.shape[0], x.shape[1], x.shape[1])
        A = torch.softmax(S, dim=-1)
        # 防「归一化方向写反」——dim 写成 1 时形状不变、数字也像概率，只有这条断言能当场抓住
        assert torch.allclose(A.sum(dim=-1), torch.ones_like(A.sum(dim=-1)))
        out = A @ V
        return self.Wo(out)


class AttnSubLayer(nn.Module):
    """Pre-LN 形式的注意力子层：x = x + attn(LN(x))"""

    def __init__(self, C, d_k, d_v):
        super().__init__()
        self.ln = MyLayerNorm(C)
        self.attn = SelfAttention(C, d_k, d_v)

    def forward(self, x):
        # Pre-LN：加回的是【原始 x】，不是 LN(x) —— Pre-LN 与 Post-LN 的区别就在 LN 的位置
        out = x + self.attn(self.ln(x))
        # 形状一致是残差能相加、以及能摞 N 层的前提
        assert out.shape == x.shape
        # 两条对照断言各抓一类"静默错误" —— 关键是【对照对象只差被删掉的那一项】：
        assert not torch.equal(out, self.attn(self.ln(x)))   # 抓「残差被漏掉」（对照对象不含 +x）
        assert (out != x + self.attn(x)).all()               # 抓「LN 被漏掉」（对照对象不含 LN）
        return out


# ============================================================
# 自测区
# ============================================================
if __name__ == '__main__':
    torch.manual_seed(1)
    B, T, C = 2, 5, 8

    # ---------- 块 1：验收标准落点 ----------
    x = torch.randn(B, T, C)
    xhat_mine = MyLayerNorm(C)(x)
    xhat_ref = nn.LayerNorm(C)(x)
    # 验收标准：与 nn.LayerNorm 数值接近。这条断言同时管住三个坑（unbiased / dim / eps 位置）
    assert torch.allclose(xhat_mine, xhat_ref, atol=1e-6)
    assert xhat_mine.mean(dim=-1).abs().max() < 1e-6
    # unbiased=False 与上一行口径一致；容差 1e-2 是因为 eps 会让 std 略小于 1（实测 0.999974~0.999997）
    assert abs(xhat_mine.std(dim=-1, unbiased=False).abs().max() - 1) < 1e-2

    # ---------- 注入测试：证明这些断言「能失败」----------
    # 断言的价值在于"能失败"。想亲手确认，就把下面任一项改坏、跑一次、再改回来。
    # 本脚本 2026-09-29 的实跑记录（7 项全部被拦下 ✅）：
    #   块 1：① eps 挪到开方外面（差 3.922e-05）② var 换 unbiased=True（差 2.218e-01）
    #         ③ 归一化方向写 dim=0（每行 mean 偏离 0.9995）        → 均由 allclose 断言拦下
    #   块 3：④ softmax 写 dim=1  → 行和为 1 断言拦下（这条断言不可替代）
    #         ⑤ 删掉残差 ⑥ 删掉 LN → 两条对照断言各拦一项
    #         ⑦ 去掉 Wo → out.shape == x.shape 拦下（RuntimeError/AssertionError）

    # ---------- 块 2：有 / 无残差的梯度 ----------
    x1 = torch.randn(C, requires_grad=True)
    residual1 = run_stack(x1, ShrinkSublayer(C, a=0.9), depth=50, use_residual=True)
    residual1.sum().backward()
    print('有残差 |∂out/∂x| =', x1.grad.abs().mean().item(), '  （理论 Σ0.9ᵏ = 9.9536）')
    x2 = torch.randn(C, requires_grad=True)
    residual2 = run_stack(x2, ShrinkSublayer(C, a=0.9), depth=50, use_residual=False)
    residual2.sum().backward()
    print('无残差 |∂out/∂x| =', x2.grad.abs().mean().item(), '  （理论 0.9⁵⁰ = 5.1538e-03）')

    # ---------- 块 3：带残差 + norm 的子层 ----------
    out = AttnSubLayer(C, d_k=4, d_v=4)(x)
    assert out.shape == x.shape

    print('✅ D17 通过：手写 LN 与 nn.LayerNorm 一致、残差梯度不消失、(B,T,C) 进出不变')
