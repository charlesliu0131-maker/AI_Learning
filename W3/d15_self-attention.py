'''
D15 玩具 self-attention —— 亲手算一遍 Q / K / V

本课的新东西：
  · Q / K / V：不是输入 X 本身，而是 X 乘上【三组可学习参数】Wq / Wk / Wv 得到的
    Q = X@Wq（我的「查询」）· K = X@Wk（每个位置的「标签」）· V = X@Wv（每个位置的「内容」）
  · 分数矩阵 S = Q @ Kᵀ：第 (i,j) 格 = 「第 i 个 token 的 query」和「第 j 个 token 的 key」
    的点积 —— 点积越大 = 方向越一致 = 越相关。形状 (N,N) 是唯一的 N×N，也是算力瓶颈所在。
  · softmax 必须【按行】：一行 = 某个 token 把 100% 的注意力分配给所有位置，所以行和为 1
  · 输出 O = A @ V 是【加权求和】，不是「挑一个」—— 权重高的贡献大，但没有谁被丢掉
  · 除以 √d_k：q·k 是 d_k 项相加，标准差 ≈ √d_k，维度一大 softmax 就饱和 → 缩回去

手算参照（形状链，本课的地图）：
  X(4,3) → Q/K/V(4,2) → S = Q@Kᵀ (4,4) → A = softmax(S,dim=1) (4,4) → O = A@V (4,2)
  规则：唯一从 N 变 N² 的是 S / A（它是「两两关系」表）；输出的行数始终等于 token 数

跑出的结论：形状链全部与手算一致；A 每行和 = 1；两组定义式断言（S 与 O）均通过；
          两张热力图对比可见「除以 √d_k」让权重分布更平缓（抑制饱和）。
'''
import torch
import matplotlib.pyplot as plt

def draw_picture(A, file_name):
    plt.imshow(A.numpy(), cmap='Blues')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):            
            plt.text(j, i, f"{A[i, j]:.2f}", ha='center', va='center', color='White' if A[i, j] > 0.25 else 'Black')
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))
    # 热力图读法：行 = 谁在问，列 = 看向谁；格子越深 = 权重越大
    plt.title('Attention Matrix')
    plt.savefig(file_name)
    plt.close()

# 固定种子 + 建数据（固定种子 = 结果可复现，随机性来源要看得见）
torch.manual_seed(1)
X = torch.randn((4,3), dtype=torch.float32)
Wq, Wk, Wv = torch.randn((3, 2), dtype=torch.float32), torch.randn((3, 2), dtype=torch.float32), torch.randn((3, 2), dtype=torch.float32)

Q = X @ Wq
assert Q.shape == (4, 2)
print(f"Q的形状为:{Q.shape}")
K = X @ Wk
assert K.shape == (4, 2)
print(f"K的形状为:{K.shape}")
V = X @ Wv
assert V.shape == (4, 2)
print(f"V的形状为:{V.shape}")

# K.T 不能省：K 是 (4,2)，必须转置成 (2,4) 才能与 Q(4,2) 相乘
# S[i,j] = Q 的第 i 行 · K 的第 j 行 = 「第 i 个 token 想问什么」与「第 j 个 token 是什么」的匹配度
S = Q @ K.T
print(f"S的形状为:{S.shape}")
# 定义式断言：逐格验证 S[i,j] == Q[i]·K[j]
#   用「点积视角」交叉验证「矩阵乘法视角」—— 两法必须给出同一张表
#   （注意边界：它证明的是【算法等价】，两边同源，抓不住 X/W 本身的数据错误）
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        assert abs(S[i, j] - torch.dot(Q[i], K[j])) < 1e-6

# 缩放：除以 √d_k。q·k 是 d_k 项相加 → 标准差 ≈ √d_k，维度一大分数尺度就被撑开，
# softmax 会饱和成「只盯一个」（梯度趋 0）。除以 √d_k 把尺度拉回约 1。
# d_k 从形状里取（Q.shape[-1]），不写死 —— 否则改了维度会静默地缩放错
S2 = Q @ K.T / torch.sqrt(torch.tensor(Q.shape[-1]))


# dim=1 就是「按行」：一行 = 某个 token 的注意力预算，它要在所有【被看的位置】之间分配 100%
A = torch.softmax(S, dim=1)
# 这条是唯一能抓住「归一化方向写反」的关卡：dim 写成 0 时形状不变、数字也像概率，
# 但每行和不再是 1 —— 只有这条断言会当场炸
assert torch.allclose(A.sum(axis = 1), torch.tensor(1, dtype=torch.float32))
print(f"A的形状为:{A.shape}")
A2 = torch.softmax(S2, dim=1)
draw_picture(A, './assert/d15_attention_matrix.jpeg')
draw_picture(A2, './assert/d15_attention_matrix_2.jpeg')
# O = A @ V：加权求和 —— O[i] = Σ_j A[i,j]·V[j]
#   注意是【混合】不是【选择】：权重高的 V 贡献大，但没有哪一个被完全丢掉（处处可导）
O = A @ V
# 同样做定义式验证：每个输出 = 该行权重与所有 V 的加权和
for i in range(A.shape[0]):
    for j in range(V.shape[1]):
        assert abs(O[i, j] - torch.dot(A[i], V.T[j])) < 1e-6
print(f"Y的形状为:{O.shape}")
assert O.shape == (4, 2)