'''
D05 两层网络的反向传播 —— 梯度从哪来、形状长啥样

目标：① 标量版手算链式法则，和 autograd 对答案
      ② 搭两层 Linear + ReLU，backward() 后观察每个参数的梯度
      ③ 验证铁律：梯度形状 == 参数形状
      ④ 接上 loss，确认它对第一层 W1 的梯度非零

手算参照（链式法则）：
  x = 3 → u = 3x = 9 → L = u² = 81
  dL/dx = (dL/du)·(du/dx) = (2u)·(3) = 18 × 3 = 54

跑出的结论：梯度 = 各段局部导数相乘（链式法则）；
并且「梯度形状 == 参数形状」这条铁律成立 —— 反向传播的每一段，
都是把上游传来的梯度乘以本段的局部导数，再交给下一段。
'''

import torch
import torch.nn as nn

print("===== 1. 标量版链式法则：逐段导数相乘 =====")
x = torch.tensor(3.0, requires_grad=True)
u = 3.0 * x          # u = 3x
L = u ** 2           # L = u²
print(f"x = {x.item()} → u = 3x = {u.item()} → L = u² = {L.item()}")
print(f"手算：dL/du = 2u = {2.0 * u.item():.1f}，du/dx = 3")
L.backward()
print(f"autograd：dL/dx = {x.grad.item():.1f}    手算 18 × 3 = 54")
assert abs(x.grad.item() - 54.0) < 1e-6
print("✅ 链式法则验证：梯度 = 各段导数相乘 —— autograd 正是这么算的")

print("\n===== 2. 搭两层 Linear + ReLU =====")
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(3, 5),   # Linear1：W1 形状 (5, 3)
    nn.ReLU(),         # 负数清零
    nn.Linear(5, 2),   # Linear2：W2 形状 (2, 5)
)
for name, p in model.named_parameters():
    print(f"参数 {name} 的形状 = {tuple(p.shape)}")

x = torch.randn(8, 3)         # 8 个样本，每个 3 个特征
out = model(x)                 # 前向：一层层往下算
loss = out.mean()              # 收敛成标量（模拟真实 loss 的形态）, 因为梯度只对标量有意义
print(f"\nx 形状 {tuple(x.shape)} → out 形状 {tuple(out.shape)} → loss = {loss.item():.4f}（标量）")

print("\n===== 3. backward() 后：每个参数都收到梯度 =====")
loss.backward()   # 从 loss 反向，把梯度传到所有参数
for name, p in model.named_parameters():
    print(f"{name}: 参数形状 {tuple(p.shape)}   ←  梯度形状 {tuple(p.grad.shape)}")

for name, p in model.named_parameters():
    assert p.grad is not None
    assert p.grad.shape == p.shape
print("✅ 铁律验证：梯度形状 == 参数形状（一一对应）")

print("\n===== 4. loss 对第一层 W1 的梯度非零 =====")
W1 = model[0].weight   # (5, 3)
grad_mag = W1.grad.abs().sum().item()
print(f"W1.grad 的绝对值之和 = {grad_mag:.4f}")
assert grad_mag > 0
print("✅ 非零：loss 确实「依赖」W1 —— 梯度没有断在第一层门口")
print("预判题答案：W1.grad.shape =", tuple(W1.grad.shape), "与 W1.shape 完全一致")

print("\n🎉 d05 动手通过！复盘提示：在日志画一张箭头图 ——")
print("前向：x → Linear1 → ReLU → Linear2 → loss   （向下）")
print("反向：loss → W2 → ReLU → W1                  （梯度向上走，每段乘一个局部导数）")
