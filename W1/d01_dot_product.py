'''
D01 向量点积：用两种方法计算 [1,2,3]·[4,5,6]，并验证结果与手算一致。

手算参照：[1,2,3] · [4,5,6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
  方法一：torch.dot(a, b)   —— 框架自带
  方法二：a * b → sum()     —— 还原「对应相乘再相加」的本质
'''

import torch

# 用 PyTorch 的"张量"(tensor) 表示两个向量
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 方法一：PyTorch 自带的点积函数 dot()
result = torch.dot(a, b)
print("torch.dot 结果:", result.item())   # .item()：把 0 维张量取成纯 Python 数值（脱离计算图）

# 方法二：还原"对应相乘再相加"的本质，亲手算一遍
product = a * b          # 对应位置相乘 → [4, 10, 18]
print("对应相乘结果:", product)
total = product.sum()    # 再全部加起来 → 32
print("手动拆解结果:", total.item())

# 验证两种方法一致，且等于手算的 32
assert abs(result - 32.0) < 1e-6
assert abs(total - 32.0) < 1e-6
assert torch.allclose(result, total)
print("✅ 验证通过：结果 = 32，与手算一致！")
