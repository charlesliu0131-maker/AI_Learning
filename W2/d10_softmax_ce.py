'''
D10 手写 softmax 与交叉熵 —— 数值稳定性的三档

本课的新东西：
  · 三种写法：naive（直接 exp）→ stable（减 max）→ logsumexp（全程 log 域）
  · naive 为什么会炸：exp 溢出成 inf → inf/inf = nan
  · 为什么「减 max」还不够：它只防【溢出】，防不了【下溢】——
    某个 logit 比 max 小很多时，exp 下溢成 0 ⇒ P=0 ⇒ log(0) = -inf
  · logsumexp = log(sum(exp(x))) 的稳定算法，全程不离 log 域，
    这也是 nn.CrossEntropyLoss 内部的做法
  · log_softmax = logits - logsumexp(logits)，后面算 NLL 用的就是它

手算参照：均匀分布（27 类，各 1/27）的交叉熵 = ln(27) ≈ 3.2958
          未训练的随机 logits ≈ 4.0713 > 3.2958 ⇒ 随机初始化比「瞎猜均匀」还差

跑出的结论：正常数据下四种写法给出同一个 loss（4.0713，1e-6 内一致）；
          logits 放大 1000 倍后三档分化 —— naive→nan、stable→inf、
          logsumexp→2101.37（与内置 CrossEntropyLoss 一致）。
'''
import torch
from torch import nn

g = torch.Generator().manual_seed(2147483647) # 设置随机种子,保证每次生成的名字相同
# 3.构建神经网络模型
logits = torch.randn(5, 27, generator= g)
targets = torch.randint(0, 27, (5,), generator= g)

# naive：先 exp 再归一化 —— 数学上没错，但 logits 一大 exp 立刻溢出成 inf
def naive(logits, targets):
    P = logits.exp()/logits.exp().sum(dim=1, keepdim=True)
    loss_naive = -P[torch.arange(P.shape[0]), targets].log().mean()
    return loss_naive

# 稳定版 (逻辑值减完最大值后, 值最大为0, exp后最大为1, 防止logits过大, 导致exp溢出)
# 依据：softmax 的分子分母【同除以 exp(max)】结果不变 —— 所以先减 max 不影响概率，
#       却把指数抬进了安全区间。但它只挡住「溢出」，挡不住「下溢」（见实验 A）。
def stable_loss(logits, targets):
    o = logits - logits.max(dim=1, keepdim=True).values
    P = o.exp()/o.exp().sum(dim=1, keepdim=True)                         # 相当于分子,分母同时除以 exp(max)
    loss_stable = -P[torch.arange(o.shape[0]), targets].log().mean()
    return loss_stable
    
# 手写logsumexp(将复杂的流程保障成一个函数, 简化代码)
# logsumexp = log(sum(exp(x))) 的稳定算法：提出 max → 算 (x-max) 的 exp 之和 → 取 log → 把 max 加回来。
# 关键在「全程不离开 log 域」：既不会 exp 出溢出的巨大值，也不会把概率算成 0。
def logsumexp(logits, dim=1, keepdim=True):
    m = logits.max(dim, keepdim).values
    return m + (logits-m) .exp().sum(dim=dim, keepdim=keepdim).log()

loss_naive = naive(logits, targets)
print(f"naive方式计算的loss:{loss_naive}")

loss_stable = stable_loss(logits, targets)
print(f"stable方式计算的loss:{loss_stable}")

# log_softmax = logits - logsumexp(logits)：直接得到 log 概率（后面算 NLL 用的就是它）
logp = logits - logsumexp(logits)                                    # exp(logits) / (exp(max) / exp(logits).sum() * exp(max)
loss_log_soft = -logp[torch.arange(logits.shape[0]), targets].mean()
print(f"log_soft方式计算的loss:{loss_log_soft}")

loss_buildin = nn.CrossEntropyLoss()(logits, targets)
print(f"内置CrossEntropy方式计算的loss:{loss_buildin}")

# 计算出来的loss为4.0713, 均匀分布的交叉熵为ln(27)≈3.2958, 说明未训练的分数, 比"瞎猜均匀"还差.
assert abs(loss_naive - loss_buildin) < 1e-6
assert abs(loss_naive - loss_stable) < 1e-6
assert abs(loss_naive - loss_log_soft) < 1e-6

# 稳定性实验
# 实验 A：把 logits 放大 1000 倍
big = logits * 1000
print(big.shape)
assert torch.isnan(naive(big, targets))                                        # naive 版 → nan（exp 溢出成 inf，inf/inf = nan）
print(f"logits放大 1000 倍后, stable方式计算出的loss{stable_loss(big, targets)}") # log(0) = -inf
# 减max只是防止溢出, 但不能防止下溢.
# 减 max 后值域是 [-4559, 0]，非最大列的 exp 下溢成 0 ⇒ P=0 ⇒ log(0) = -inf。
# 这就是「减 max 半稳、logsumexp 真稳」的分界线。
assert torch.isinf(stable_loss(big, targets))
print(f"logits放大 1000 倍后, 内置CrossEntropy方式计算出的loss{nn.CrossEntropyLoss()(big, targets)}")

# 实验 B：构造"极小概率"
# 把某行正确类设成极负（如 -1000）→ 概率为 0
# 与实验 A 的区别：这次是【单个位置】被压低，影响更局部 ——
# naive/stable 依然算成 inf，而 logsumexp 仍给出有限值（204.07，与内置一致）。
logits[0, targets[0]] -= 1000
print(f"logits减 1000 后, naive方式计算出的loss{naive(logits, targets)}")   
print(f"logits减 1000 后, stable方式计算出的loss{stable_loss(logits, targets)}")                                               
logp = logits - logsumexp(logits)
loss_log_soft = -logp[torch.arange(logits.shape[0]), targets].mean()
print(f"logits减 1000 后, log_soft方式计算出的loss{loss_log_soft}")
print(f"logits减 1000 后, 内置CrossEntropy方式计算出的loss{nn.CrossEntropyLoss()(logits, targets)}")
assert torch.isfinite(loss_log_soft)