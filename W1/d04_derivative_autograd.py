'''
D04 导数的直觉 —— 数值微分 vs 自动微分

目标：① numpy 数值微分：用「割线斜率」近似「切线斜率」
      ② 缩小 h，观察差商收敛到真值
      ③ PyTorch autograd：backward() 一次给出精确导数
      ④ 对比两种方式，理解它们差在哪

手算：f(x) = x²，导数 f'(x) = 2x，所以在 x=3 处 f'(3) = 6
      (f(3+h) - f(3)) / h = (9 + 6h + h² - 9) / h = 6 + h
      → h 越小，差商越接近 6（但永远差一个 h，这就是「导数悖论」）

跑出的结论：数值微分只是近似 —— h=0.001 时得 6.001，误差恰好等于 h；
而 autograd 一步就给出精确的 6，且不需要手选 h。
这个「精确」正是后面 loss.backward() 能用的原因。
'''

import numpy as np
import torch

print("===== 1. 数值微分：割线斜率 ≈ 切线斜率 =====")
def f(x):
    return x ** 2

x0 = 3.0
h = 0.001
slope = (f(x0 + h) - f(x0)) / h
print(f"f(x)=x² 在 x=3 处：h={h} 时差商 = {slope:.6f}")
print(f"手算公式 6+h = {6.0 + h:.6f}   真实导数 2x = 6")
assert abs(slope - 6.0) < 0.01   # 差商 6.001，离 6 只差一个 h
print("✅ 数值微分 ≈ 6：割线已经很接近切线了")

print("\n===== 2. h 越缩越小：割线在「贴向」切线 =====")
for hh in [1.0, 0.1, 0.01, 0.001, 0.0001]:
    s = (f(x0 + hh) - f(x0)) / hh
    print(f"h={hh:<8} 差商={s:.6f}   误差={abs(s - 6.0):.4e}")
# 观察：h 每缩小 10 倍，误差也缩小约 10 倍 → 差商收敛到 6
print("✅ h→0 时差商收敛到 6 —— 这就是视频里「割线变切线」的过程")

print("\n===== 3. PyTorch autograd：让框架替你求导 =====")
x = torch.tensor(3.0, requires_grad=True)  # 声明：x 是需要追踪导数的变量
y = x ** 2                                  # 前向：算出 y
y.backward()                                # 反向：自动求出 dy/dx
print(f"x = {x.item()},  y = x² = {y.item()}")
print(f"x.grad = {x.grad.item()}   ← autograd 给的导数")
assert abs(x.grad.item() - 6.0) < 1e-6
print("✅ autograd 输出 dy/dx = 6，和手算 2x = 6 完全一致")

print("\n===== 4. 两者到底差在哪？ =====")
print(f"数值微分：{slope:.10f}   → 用『一个很小的 h』近似，永远带误差")
print(f"自动微分：{x.grad.item():.10f} → 把表达式拆成基本运算，每步精确求导再串起来")
print()
print("一句话总结：")
print("  数值微分 ≈ 拿放大镜看（近似，简单直观）")
print("  autograd  ≈ 按导数规则精算（精确，这是反向传播！）")
print("以后你会看到 loss.backward() —— 它算的梯度，就是这里 x.grad 的推广")
