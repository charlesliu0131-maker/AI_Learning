'''
同一个 MLP 分别用 SGD 与 Adam 训练，画两条 loss 曲线对比；
再把学习率调成三档（0.001 / 0.01 / 0.1）各跑一次，观察太大太小的表现。

实验设计（控制变量）：
  · 6 组 = {SGD, Adam} × lr {0.001, 0.01, 0.1}，全部共用同一个 seed
    → 初始权重与 batch 抽样顺序完全一致，只留「优化器 / lr」这一个变量
  · 统一跑 20000 步；每 100 步用【全量数据】评估一次 loss（六组口径一致才可比）
'''
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np

def char_index_exchange(data):
    """
    建立字符与索引之间的映射关系
    """
    # 1.建立字符<->索引映射
    chars = sorted(list(set(''.join(data))))

    stoi = {ch:i+1 for i, ch in enumerate(chars)} # 字符->索引
    stoi['.'] = 0 # 句子结束符

    itos = {i:ch for ch, i in stoi.items()} # 索引->字符
    return stoi, itos

def train_sample(block_size, data, stoi):
    """
    使用滑窗构造训练样本
    """
    X, Y = [], []
    for w in data:
        block = [0] * block_size          # 用 id=0（即 '.'）填充：开头几个位置的「历史」是补的
        w += '.'                          # 末尾补结束符，让模型学会「名字该收尾了」
        for ch in w:
            # 每个字符都产出一条样本：用「前 block_size 个字符」预测下一个字符
            X.append(block)
            Y.append(stoi[ch])
            # 窗口右移一位：丢掉最旧的、纳入刚读到的字符 —— 这就是「看前 3 个字符」
            block = block[1:] + [stoi[ch]]
    # X: (N, 3) 字符 id 窗口；Y: (N,) 对应的正确答案（下一个字符）
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X,Y

def make_params(seed):
    # 构建五个参数；此函数内一律不开梯度，交给 run() 统一开（见函数末尾说明）
    # 种子放在函数内：只要 seed 相同，抽出的初始权重就一定相同（可复现的前提）
    torch.manual_seed(seed)
    C = torch.randn((27, 10))
    W1 = torch.randn((30, 200)) * 5/3 / math.sqrt(30)   # Kaiming 初始化：tanh 的 gain=5/3、fan_in=30
    b1 = torch.zeros(200)                              # 偏置从 0 起步（常规做法）
    # W2 乘 0.01：把初始 logits 压小，让初始 loss 接近 ln(27)≈3.3 而不是一上来就爆
    W2 = torch.randn(200, 27) * 0.01 
    b2 = torch.zeros(27,)
    # 顺序必须与 forward() 里 parameters[0..4] 一一对应 —— 顺序错了不报错，但结果全废
    # 另外：创建时不开梯度 → 张量是「用户直接创建」的 leaf；
    #      若先开梯度再做运算，就会变成非 leaf，backward 后 .grad 为 None（D13 回顾 Q2 的坑）
    return [C, W1, b1, W2, b2]

def run(seed, X, Y, lr, steps, Opt_name="SGD"):
    # 这里也要设种子：训练时 torch.randint 抽 batch 同样消耗这条随机流，
    # 从同一 seed 出发，6 组的初始权重与 batch 顺序才会完全一致
    torch.manual_seed(seed)
    # 取一份全新的初始参数（同一 seed → 同一组值，与上一组是否训练过无关）
    parameters = make_params(seed)
    # 先创建、后统一开梯度：保证每个参数都是 leaf（参数必须 is_leaf=True）
    for param in parameters:
        param.requires_grad_(True)

    # 自检：参数若由运算产生（非 leaf），backward 后拿不到 .grad —— D13 回顾 Q3 的坑
    assert all(p.is_leaf for p in parameters)

    losses = []          # 每 100 步存入一个「全量评估」loss

    # Adam 需要额外维护两个状态：一阶矩 m 与二阶矩 v，形状与参数一一对应
    if Opt_name == "Adam":
        v, m = [], []
        for i, param in enumerate(parameters):
            # 必须从 0 起步：偏差修正 (1-β^t) 正是为「初值为 0 导致前期被拉小」而设
            v.append(torch.zeros(param.shape))
            m.append(torch.zeros(param.shape))

    def forward(Xb):
        # 形状链：(B,3) →查表→ (B,3,10) →view→ (B,30) →W1+tanh→ (B,200) →W2→ (B,27)
        emb = parameters[0][Xb].view(-1, 30)      # 3 个字符各 10 维，拼成 30 维
        H = F.tanh(emb @ parameters[1] + parameters[2])
        logits = H @ parameters[3] + parameters[4]
        return logits

    """训练模型"""
    for step in range(steps):
        # 随机抽 32 条构成 mini-batch：算出的梯度是「真梯度的带噪声估计」
        ix = torch.randint(0, X.shape[0], (32, ))
        Xb, Yb = X[ix], Y[ix]

        logits = forward(Xb)

        # 训练 loss：在 batch 上算（快，但噪声大 → 所以不用它的逐位值画曲线）
        loss = F.cross_entropy(logits, Yb)

        # 清空上一轮梯度：置 None 比 zero_grad() 更直接，且不在计算图里
        for param in parameters:
            param.grad = None

        # 反向传播：算出 loss 对这五个参数的梯度
        loss.backward()

        # 更新参数 —— D13 的主角：两套更新式二选一
        if Opt_name == "SGD":
            for param in parameters:
                param.data -= lr * param.grad      # 手写 SGD：w ← w − lr·grad
        elif Opt_name == "Adam":
            # 手写 Adam：先更新两个滑动平均（纯数值计算，包 no_grad 不建图）
            for i, param in enumerate(parameters):
                with torch.no_grad():
                    v[i] = 0.999*v[i] + 0.001*param.grad**2   # 二阶矩：梯度平方的滑动平均（幅度）
                    m[i] = 0.9*m[i] +0.1*param.grad           # 一阶矩：梯度的滑动平均（方向 ≈ 动量）
                # 偏差修正：m、v 从 0 起步会被拉小，除以 (1-β^t) 还原；t 从 1 开始 → step+1
                m_hat = m[i]/(1-0.9**(step+1))
                v_hat = v[i]/(1-0.999**(step+1))
                # 更新式：分母 √v̂ 让每个参数拥有自己的有效步长（梯度大 → 步子自动变小）
                param.data -= lr * m_hat / (v_hat**0.5+1e-8)
        # 每 100 步用【全量数据】评估一次：六组口径一致才可比；包 no_grad 省掉白建计算图
        if (step+1) % 100 == 0:
            with torch.no_grad():
                loss = F.cross_entropy(forward(X), Y)
            losses.append(loss.item())     # .item() 取纯数值，否则会把计算图留在列表里
    # 自检：记录条数必须等于 steps/100，否则说明评估没按预期触发
    assert len(losses) == steps // 100
    return losses

def draw_picture(x, Y, optim, name):
    # 中文字体：macOS 用 Arial Unicode MS；Windows 对应 Microsoft YaHei（不设会全是方框）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False   # 负号正常显示（否则负号也是方框）
    plt.figure(figsize=(8,8))

    # Y[0]/Y[1]/Y[2] 分别对应 lr = 0.001 / 0.01 / 0.1 三条曲线
    plt.plot(x, Y[0], label=optim+"_0.001", color='red')
    plt.plot(x, Y[1], label=optim+"_0.01" , color='blue')
    plt.plot(x, Y[2], label=optim+"_0.1"  , color='green')

    plt.legend()
    # 4. 添加标题和坐标轴标签 (让图表更完整)
    plt.title(optim+"不同学习率曲线")
    plt.xlabel("迭代步数")
    plt.ylabel("loss")

    # 保存后必须 close：否则图形对象堆积，下一次画图可能串到上一张上
    plt.savefig(name)
    plt.close()
    

    




if __name__ == '__main__':
    seed = 1
    """导入 人名 数据集"""
    words = open('names.txt').read().splitlines()
    stoi, itos = char_index_exchange(words)

    """创建训练样本"""
    X, Y = train_sample(3, words, stoi)
    
    optims = ["SGD", "Adam"]
    lrs = [0.001, 0.01, 0.1]

    losses = []
    # 6 组实验共用同一个 seed=1（见 run() 第一行）→ 初始权重与 batch 顺序完全一致
    for optim in ["SGD", 'Adam']:
        for lr in lrs:
            loss_list = run(1, X, Y, lr, 20000, optim)
            losses.append(loss_list)
    # x 轴刻度：对应第 100、200 … 20000 步（每 100 步一个记录点，共 200 个）
    x = np.linspace(100, 20000, 200)
    draw_picture(x, losses[:3], "SGD", "./assert/d13_SGD_learning_curve.jpeg")
    draw_picture(x, losses[3:], "Adam", "./assert/d13_Adam_learning_curve.jpeg")
