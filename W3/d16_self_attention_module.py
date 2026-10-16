'''
D16 缩放点积注意力 + PyTorch 模块化 —— 把 D15 的玩具升级成可复用的 nn.Module

本课的新东西：
  · 模块化三件事：继承 nn.Module → 参数在 __init__ 里声明（才会被 parameters() 收集、
    优化器拿得到）→ forward 里只做形状变换，训练循环留在模块外面。
  · nn.Linear(C, d_k) 只在【最后一维】作用（实测 (2,4,3)→(2,4,2)、(2,4,4,3)→(2,4,4,2)），
    所以套在 (B,T,C) 上自动得到 (B,T,d_k)，不需要手动 reshape。
  · 加了一维 batch，所有维度索引都得重算：K.T 在 3 维下是「反转所有轴」（PyTorch 已给弃用
    警告），必须写 K.transpose(-2,-1)；dim=1 在 (T,T) 里对、在 (B,T,T) 里却是把【列】
    归一化，必须写 dim=-1 ⇒ 一律用负索引。
  · 缩放除以 √d_k：q·k 是 d_k 项相加，标准差 ≈ √d_k，维度一大 softmax 就饱和 → 梯度消失。
  · Q/K 投影关掉 bias 是有理由的：所有 key 加同一个向量 ⇒ 每行分数加同一常数 ⇒ 被 softmax
    的平移不变性吃掉（实测最大差异 1.19e-07）；而 Q 的 bias 有效（差异 0.52）。

手算参照（形状链，B 维全程不变）：
  X(B,T,C) → Q/K(B,T,d_k)、V(B,T,d_v) → S = Q@Kᵀ (B,T,T)
           → A = softmax(S,-1) (B,T,T) → out = A@V (B,T,d_v) → Wo → (B,T,C)
  规则：唯一变成 T×T 的是 S（两两关系表）；Wo 把 d_v 映回 C，才满足「进 (B,T,C) 出 (B,T,C)」

跑出的结论：形状链与手算一致 —— (2,3,4) → Q/K(2,3,4)、V(2,3,5) → S(2,3,3) → A(2,3,3)
           → out(2,3,5) → Wo → (2,3,4)；A 每行和 = 1；同一实例喂 (2,3,4)/(5,7,4)/(1,10,4)
           全部跑通 ⇒ 模块可复用。
'''
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, C, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        # bias=False：Q/K 的偏置会被 softmax 的平移不变性吃掉（见头部第 5 条），所以省掉；
        #             V 的偏置并非无效，但标准实现里通常三个投影一起去 bias，保持统一
        self.Wq = nn.Linear(C, d_k, bias=False)
        self.Wk = nn.Linear(C, d_k, bias=False)
        self.Wv = nn.Linear(C, d_v, bias=False)
        # 输出投影 Wo：把 d_v 映回 C —— 有了它才是「进 (B,T,C) 出 (B,T,C)」，
        #              也才能在后面把多个注意力层叠起来（D19 GPT block 的前提）
        self.Wo = nn.Linear(d_v, C, bias=False)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        print(f'Q的形状为:{Q.shape};K的形状为:{K.shape};V的形状为:{V.shape}')
        # transpose(-2,-1) 只换最后两轴；不能用 K.T —— 3 维下它是「反转所有轴」，
        # 形状恰好还是 (2,4,2) 看着没事，但 Q@K.T 会当场报
        # RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [2, 2] but got: [2, 4].
        S = Q @ K.transpose(-2, -1) / self.d_k**0.5
        print(f'S的形状为:{S.shape}')
        # 形状断言用 x.shape 动态取、不写死数字：换个 (B,T) 也不用回来改断言
        assert S.shape == (x.shape[0], x.shape[1], x.shape[1])
        # 升到 (B,T,T) 后 dim=1 的含义变了（它把【列】归一化成 1，实测每行和 1.5819/0.5926/
        # 0.2370/1.5885）：形状不变、数字也像概率，只有下面这条断言能当场抓住 ⇒ 统一写 dim=-1
        A = torch.softmax(S, dim=-1)
        print(f'A的形状为:{A.shape}')
        # 行和为 1（每个样本、每个位置各一行）—— 唯一能抓住「归一化方向写反」的关卡
        assert torch.allclose(A.sum(-1), torch.ones(A.shape[:-1]))
        out = A @ V
        print(f'out的形状为:{out.shape}')
        return self.Wo(out)

# Test the model
if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4)
    model1 = SelfAttention(4, 4, 5)
    model2 = SelfAttention(4, 4, 5)
    out = model1(x)
    assert out.shape == x.shape

    # 可复用性：同一个实例换 (B,T) 再喂 —— 参数只跟 C 有关，跟 B/T 无关
    for shape in [(2, 3, 4), (5, 7, 4), (1, 10, 4)]:
        x2 = torch.randn(shape)
        out2 = model1(x2)
        assert out2.shape == x2.shape
        print(f'输入 {tuple(x2.shape)} → 输出 {tuple(out2.shape)}')

    print('✅ 复用性验证通过：同一实例、三种 (B,T) 全部保持 (B,T,C) → (B,T,C)')
