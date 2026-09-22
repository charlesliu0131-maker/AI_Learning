'''
D07 回归任务全流程 —— 把训练封装成函数 + 画出 loss 曲线

目标：把 D06 的五行循环封装成可复用的 train_model()，
      并把每一步的 loss 记下来画成曲线，直观看下降趋势。

手算参照：真实规律 y = 2x + 1 ⇒ 训练后应满足 w ≈ 2、b ≈ 1、loss ≈ 0

跑出的结论：200 步后 loss 降到 ≈0，w = 2.0000、b = 1.0000。
（注：本脚本未设随机种子，每次运行 loss 的「起点」会不同，终点一致。）
loss 曲线先陡后平 —— 越接近最优点梯度越小，同样 lr 下迈的步子也越小。
'''

import torch
from torch import nn
import matplotlib.pyplot as plt

# ① 造数据
x = torch.linspace(-1, 1, 100).reshape(-1, 1)   # 100 个点，形状 (100,1)
y = 2 * x + 1                                     # 真实规律，无噪声版

# ② 建模型三件套
def train_model(x, y, step=200, lr=0.1):
    """训练一个 Linear(1,1) 去拟合 (x, y)，返回训好的模型与逐步 loss。

    参数：x, y —— 数据；step —— 训练步数；lr —— 学习率
    返回：(model, losses) —— 模型本身 + 每一步的 loss 列表
    """
    # 模型：nn.Linear(1, 1)（输入 1 个特征 → 输出 1 个数）
    torch.manual_seed(1)
    model = nn.Linear(1, 1)
    # 误差：nn.MSELoss()
    loss_fn = nn.MSELoss()
    # 优化器：torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # ③ 训练循环（核心！）
    losses = []
    for epoch in range(step):
        # 1. 前向预测
        pred = model(x)
        # 2. 计算误差
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        # 3. 梯度清零（否则梯度会累加）
        optimizer.zero_grad()
        # 4. 反向传播，计算梯度
        loss.backward()
        # 5. 更新参数
        optimizer.step()
    return model, losses

model, losses = train_model(x, y)
plt.plot(losses)
plt.xlabel("step"); plt.ylabel("loss"); plt.title("loss curve")
plt.savefig("./assert/d07_loss_curve.jpeg", dpi=150)
plt.close()

print(f"loss: {losses[0]:.4f} → {losses[-1]:.6f}")
print(f"w = {model.weight.item():.4f}, b = {model.bias.item():.4f}")
assert abs(model.weight.item() - 2.0) < 1e-3
assert abs(model.bias.item() - 1.0) < 1e-3
assert losses[-1] < losses[0], "loss 必须在下降"
assert losses[-1] < 1e-3, "loss 应降到接近 0"
assert all(loss < 1e-3 for loss in losses[-20:])