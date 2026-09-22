'''
D02 张量基础操作（下半段）：创建、加减、reshape、转置

手算参照：
  加减：[1,2,3] + [4,5,6] = [5,7,9]
  reshape：把 6 个数重新「排队」进不同形状的格子（顺序不变）
  transpose：把矩阵的「行」和「列」对调

跑出的结论：reshape 与 transpose 都能把 (2,3) 变成 (3,2)，
但内容完全不同 —— reshape 是重新排队，transpose 是轴对调。
'''

import torch

print("===== 1. 创建张量 =====")
# 从 Python 列表创建：这就是一个"向量"（1 维张量）
a = torch.tensor([1.0, 2.0, 3.0])
# 几个常用创建方法：
zeros = torch.zeros(3)          # 全 0，一维、长度 3
ones = torch.ones(2, 3)         # 全 1，2 行 3 列（二维 = 矩阵）
seq = torch.arange(1.0, 7.0)    # 从 1 数到 6（结束值 7 不含）

print("a =", a)
print("torch.zeros(3) =", zeros)
print("torch.ones(2,3) =\n", ones)
print("torch.arange(1,7) =", seq)

# assert 关卡：形状对不对？shape 就是"每个维度的大小"
assert a.shape == (3,)              # 一维、长度 3
assert zeros.shape == (3,)
assert ones.shape == (2, 3)         # 2 行 3 列
assert seq.shape == (6,)
print("✅ 创建检查通过")

print("\n===== 2. 加减：对应位置相加 =====")
b = torch.tensor([4.0, 5.0, 6.0])
s = a + b       # 对位相加 → [5, 7, 9]
d = a - b       # 对位相减 → [-3, -3, -3]
mul = a * b     # 对位相乘（注意：这不是"矩阵乘法"！）
plus10 = a + 10 # 数字 10 会"广播"到每个位置 → [11, 12, 13]
print("a + b =", s, "   手算 [5, 7, 9]")
print("a - b =", d)
print("a * b（对位乘）= ", mul)
print("a + 10 =", plus10)

# assert 关卡：torch.equal 检查两个张量"逐位全相等"
# 注意：torch.equal 是【精确】比较 —— 这里安全是因为值恰好都是整数；
#       换成 0.1、1/3 这类浮点结果，要用 torch.allclose（带容差）
assert torch.equal(s, torch.tensor([5.0, 7.0, 9.0]))
assert torch.equal(d, torch.tensor([-3.0, -3.0, -3.0]))
assert torch.equal(a * b, torch.tensor([4.0, 10.0, 18.0]))  # 还记得这个 32 的来源吗？
assert abs((a * b).sum() - 32.0) < 1e-6
assert torch.equal(plus10, torch.tensor([11.0, 12.0, 13.0]))
print("✅ 加减检查通过")

print("\n===== 3. reshape：只换形状，不换数据 =====")
v = torch.arange(1.0, 7.0)     # [1,2,3,4,5,6]
m = v.reshape(2, 3)            # 排成 2 行 3 列
r = v.reshape(3, 2)            # 或排成 3 行 2 列
print("v =", v)
print("v.reshape(2,3) =\n", m)
print("v.reshape(3,2) =\n", r)

# assert 关卡：形状对、数据一个不少（reshape(-1) 让 PyTorch 自动推长度）
assert m.shape == (2, 3)
assert r.shape == (3, 2)
assert torch.equal(m.reshape(-1), v)   # 摊平回一维，和原来一模一样
print("✅ reshape 检查通过（数据总量 6 个没变）")

print("\n===== 4. 转置 transpose：行和列对调 =====")
t = m.T          # .T 就是转置：形状从 (2,3) 变 (3,2)
print("m =\n", m)
print("m.T =\n", t)

# assert 关卡：形状对调 + 转置的定义 t[i,j] == m[j,i]
assert t.shape == (3, 2)
assert t[0, 1].item() == 4.0    # 第 0 行第 1 列是 4 → 原矩阵第 1 行第 0 列也是 4
ok = all(t[i, j].item() == m[j, i].item()
         for i in range(3) for j in range(2))
assert ok
print("✅ 转置检查通过：t[i,j] 恒等于 m[j,i]")

print("\n===== 5. reshape 和 转置 到底差在哪？（新手最容易混） =====")
print("v.reshape(3,2) =\n", r)      # [[1,2],[3,4],[5,6]]
print("v.reshape(2,3).T =\n", t)    # [[1,4],[2,5],[3,6]]
# 两者形状都是 (3,2)，但内容不同！
assert not torch.equal(r, t)

# 用大白话理解：
#   reshape(3,2)：把 6 个数按原顺序"每 2 个一组"重新排队 → [[1,2],[3,4],[5,6]]
#   m.T：不重新排队，只是把"原来的行"变成"列" → 第一列 = 原来的第一行 [1,2,3] 竖起来
print("✅ 对比检查通过：reshape 是重排，转置是轴对调，二者不同")

print("\n🎉 全部通过！现在你掌握：shape、对位运算、reshape、转置")
