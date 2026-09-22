'''
D06 最小训练循环 —— 从零拟合 y = 2x + 1

目标：亲手写出「训练」的最小闭环，从此看任何训练代码都不慌。
      循环体严格五行顺序：pred → loss → zero_grad → backward → step

手算参照：真实规律 y = 2x + 1 ⇒ 收敛后应满足 w ≈ 2、b ≈ 1、loss ≈ 0

跑出的结论：200 轮后 loss 从 0.1292 降到 0.0000，w ≈ 2、b ≈ 1 —— 
五行循环确实把参数推到了正确的位置上。
另：顺序不能错。「零在反之前，步在反之后」，若先 step 再 backward，
更新用的就是上一轮残留的梯度。
'''

import torch
from torch import nn
import matplotlib.pyplot as plt
# ① 造数据
x = torch.linspace(-1, 1, 100).reshape(-1, 1)   # 100 个点，形状 (100,1)
y = 2 * x + 1                                     # 真实规律，无噪声版

# ② 建模型三件套
# 模型：nn.Linear(1, 1)（输入 1 个特征 → 输出 1 个数）
model = nn.Linear(1, 1)

# 误差：nn.MSELoss()
loss_fn = nn.MSELoss()

# 优化器：torch.optim.SGD(model.parameters(), lr=0.1) ← 别忘了把参数交给它
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ③ 训练循环（核心！）
# 循环体里严格按五行顺序：pred → loss → zero_grad → backward → step
# 顺序错了会导致之前计算的梯度会用来更新这一轮的参数, 先 step 再 backward，更新用的是上一轮的梯度.
# 循环建议 100~300 步，每 20 步打印一次 loss 看下降趋势。
n_epochs = 200
weight, bias = [], []
for epoch in range(n_epochs):
    # 1. 前向预测
    pred = model(x)

    # 2. 计算误差
    loss = loss_fn(pred, y)

    # 3. 梯度清零（否则梯度会累加）
    optimizer.zero_grad()

    # 4. 反向传播，计算梯度
    loss.backward()

    # 5. 更新参数
    optimizer.step()

    weight.append(model.weight.item())
    bias.append(model.bias.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")

# ④ 收尾打印参数
# model.weight.item() 和 model.bias.item() 取出数值看是否 ≈2、≈1
assert abs(model.weight.item() - 2.0) < 0.1, "w 应接近 2"
assert abs(model.bias.item() - 1.0) < 0.1, "b 应接近 1"
assert loss.item() < 1e-3, "loss 应接近 0"
print("✅ 训练成功：w≈2，b≈1，loss≈0")

# 5 画图, 看weight和bias是如何一步步更新到准确值的
x = torch.linspace(1, n_epochs, n_epochs)
plt.figure(figsize=(8, 8))
plt.plot(x.data, weight, color="red", label="weight")
plt.plot(x.data, bias, color="green", label="bias")
plt.title("weight and bias update")
plt.legend()
plt.savefig('./assert/d06_weight_bias_update.jpeg')
plt.close()