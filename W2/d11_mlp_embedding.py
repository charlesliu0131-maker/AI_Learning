'''
D11 上下文窗口与 Embedding —— 让每个字符拥有「可学习向量」

本课的新东西：
  · 上下文窗口（block_size = 3）：用「前 3 个字符」预测下一个字符。
    样本不再按「字符对」统计，而是每个名字的每个字符都产出一条 → N ≈ 22.8 万
  · Embedding 表 C(27, dim)：把字符索引映射成一个 dim 维向量 ——
    注意 C 是【可学习参数】（模型的一部分、会被梯度更新），不是数据
  · C[X] 与 one_hot(X) @ C 等价：独热行的 1 只挑出一行、0 全部作废，
    矩阵乘法退化成「查表」
  · emb.view(-1, block_size * dim)：把 3 个 dim 维向量【拼】成一条长向量，
    作为后面 MLP 的输入（30 = 3 × 10）

手算参照：第一条样本 block = [0,0,0]（三个字符都是 '.'）
          ⇒ 它要取的是 C 的同一行三次 ⇒ emb[0] 就是 C[0] 重复 3 遍
          ⇒ 这也解释了为什么 one_hot @ C 与 C[X] 结果完全一致

跑出的结论：两种写法完全一致（实测最大误差为 0，不只是 allclose 通过）；
          这里的 dim = 2 只是为了能画二维散点图，真实模型里用 10。
'''
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 导入 人名 数据集
words = open('names.txt').read().splitlines()

# 1.建立字符<->索引映射
chars = sorted(list(set(''.join(words))))

stoi = {ch:i+1 for i, ch in enumerate(chars)} # 字符->索引
stoi['.'] = 0 # 句子结束符
itos = {i:ch for ch, i in stoi.items()} # 索引->字符

# 滑窗构造训练样本
block_size = 3
X, Y = [], []
for w in words:
    block = [0] * 3
    w += '.'
    for ch in w:
        X.append(block)
        Y.append(stoi[ch])
        block = block[1:] + [stoi[ch]]

X = torch.tensor(X)
Y = torch.tensor(Y)

print(f'训练集输入形状为:{tuple(X.shape)}, 输出形状{tuple(Y.shape)}')

# 构建Embedding表
dim = 2                     # 维度, 即将"单词"映射到dim维空间中, 每一维维度代表一定特征, 一般来说维度越多, 表征信息越丰富.
# 这里取 2 是为了能画二维散点图（见下方 scatter）；真实模型里用 10
g = torch.Generator().manual_seed(5181)
# C 是【可学习参数】不是数据：它属于模型的一部分、会被梯度更新；
# 数据是 names / X / Y，给定后不变。（本脚本只建表未训练，所以 C 还是初始随机值）
C = torch.randn((27, dim), requires_grad=True, generator=g)
print(f"Embedding表的形状为:{tuple(C.shape)}, 值域为({C.min().item():.6f}, {C.max().item():.6f})")

# 绘制dim=2 的散点图
plt.figure(figsize=(8, 8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
plt.grid(True, which='both')
plt.savefig('./assert/d11_Embedding_init.jpeg')
plt.close()

# C[X]：用索引「查表」—— 把 X 里的每个字符 id 换成 C 里对应的那一行
emb = C[X]
assert emb.shape == (X.shape[0], block_size, dim)
# view：把「3 个字符 × dim 维」摊平成一条长向量（block_size × dim），作为 MLP 的输入
x = emb.view(-1, block_size*dim)
assert x.shape == (X.shape[0], block_size*dim)

# 附加验证：one_hot @ C 为什么与 C[X] 等价 ——
# 独热向量只有一位是 1，与矩阵相乘时「1 挑出一行、0 全部作废」，乘法退化成查表。
onehot = F.one_hot(X, num_classes=27).float() # (N, 3, 27)
emb2 = onehot @ C
assert torch.allclose(emb, emb2)
print(f"两种方法的最大误差:{abs(emb-emb2).max().item()}")
print("✅ Embedding 验证通过：C[X] 与 one_hot @ C 结果完全一致")