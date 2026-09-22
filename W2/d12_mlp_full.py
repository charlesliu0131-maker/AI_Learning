'''
D12 完整 MLP —— 隐藏层 + tanh + 初始化，训练出比 bigram 更好的生成器

本课的新东西：
  · 为什么必须加非线性：若只有两层 Linear，x@W1@W2 = x@(W1@W2) = x@W，
    「两层线性」在数学上会塌成「一层线性」，模型等价于另一张 bigram 表。
    tanh 打破这个恒等式，网络才从「查表」变成「能拟合弯关系的函数逼近器」。
  · 为什么 W1 要先除以 √fan_in、再乘 5/3：√fan_in 抵消「多项求和」带来的方差放大；
    5/3 是 tanh 的 gain（torch.nn.init.calculate_gain('tanh') ≈ 1.667），
    用来补偿 tanh 对信号的压扁。初始化不当 → tanh 饱和 → 梯度被掐断 → loss 卡住。
  · 瓶颈在「上下文」不在「宽度」：h 从 200 加到 300（参数 +48%），全量 NLL 几乎不动
    ⇒ 想继续提升得让模型「看得更多」—— 这正是 W3 注意力机制的动机。
  · mini-batch：每步随机抽 32 条估梯度（快，且梯度噪声本身有轻微正则效果）。

手算参照（参数量核算，dim=10 / h=300）：
  C  27×10 = 270    W1 30×300 = 9000    b1 300
  W2 300×27 = 8100  b2 27               合计 17697
  对照：bigram 只有 27×27 = 729 个参数（约 24 倍）

跑出的结论：固定标尺（X[:1000]）2.3030 → 1.9200；
          全量 NLL = 2.2070 < bigram 2.4540（困惑度 11.64 → 9.09，↓22%）；
          采样从「字母堆」变成「像人名」。两次运行输出逐字一致（可复现）。
'''
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

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
        block = [0] * block_size
        w += '.'
        for ch in w:
            X.append(block)
            Y.append(stoi[ch])
            block = block[1:] + [stoi[ch]]
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X,Y

def sample_name(num, forward):
    """
    采样
    """
    name_list = []
    for _ in range(num):
        ctx = [0, 0, 0]
        out = []
        with torch.no_grad():
            for i in range(20):
                probs = torch.softmax(forward(torch.tensor(ctx)), dim=1)
                ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
                if ix == 0 or i == 19:
                    name_list.append("".join(out))
                    break
                out.append(itos[ix])
                ctx = ctx[1:] + [ix]
    return name_list

def draw_embedding(C):
    # 绘制dim=2 的散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:,0].data, C[:,1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
    plt.grid(True, which='both')
    plt.savefig("./assert/d12_embedding_trained.jpeg")
    plt.close()

if __name__ == "__main__":

    words = open('names.txt').read().splitlines()
    stoi, itos = char_index_exchange(words)


    block_size = 3
    X, Y = train_sample(block_size, words, stoi)
    assert X.shape == (228146, 3)
    # 用固定的样本去衡量模型训练过程中参数的优化情况
    X_fixed, Y_fixed = X[:1000], Y[:1000]


    dim = 10
    h = 300 # 神经元的多少对于最终的loss影响已经十分小, 单纯的增加神经元数量已经收效甚微.
    batch = 32
    lr = 0.1
    steps = 20000


    # 构建五个参数
    g = torch.Generator().manual_seed(5181)
    C = torch.randn((27, dim), generator=g)
    # 本课重点：除以 √fan_in（=√30）抵消「多项求和」的方差放大；乘 5/3 是 tanh 的 gain。
    # 不这么做 → 线性输出太大 → tanh 落进饱和区 → 导数 1−h²→0 → 梯度传不回去。
    W1 = torch.randn((block_size*dim, h), generator=g) * 5/3 / math.sqrt(block_size*dim)
    b1 = torch.zeros(h)
    ## 缩小初始化时的loss
    W2 = torch.randn((h, 27), generator=g) * 0.01 
    b2 = torch.zeros(27,)
    parameters = [C, W1, b1, W2, b2]

    for p in parameters:
            p.requires_grad = True
        
    # 定义前向传播
    def forward(X):
        emb = C[X]
        x = emb.view(-1, block_size*dim)
        # 这里的 tanh 就是「非线性」本身：去掉它，x@W1@W2 会塌成 x@W（两层线性 = 一层线性）
        H = torch.tanh(x @ W1 + b1)
        logits = H @ W2 + b2
        return logits

    # 参数必须是「用户直接创建」的 leaf 张量；若先开梯度再做运算，拿到的是非 leaf 中间量，
    # backward 后 .grad 恒为 None —— D12 首版正是栽在这里
    assert all(p.is_leaf for p in parameters)
    
    for step in range(steps):
        # 使用batch训练, batch_size = 32
        
        ix = torch.randint(0, X.shape[0], (batch, ), generator=g)
        Xb, Yb = X[ix], Y[ix]
        # 1. 前向传播
        logits = forward(Xb)
        # 2. 计算误差
        loss = F.cross_entropy(logits, Yb)
        # 3. 清空梯度
        for p in parameters:
            p.grad = None
        # 4. 反向传播
        loss.backward()
        # 5.更新权重
        for p in parameters:
            p.data += -lr * p.grad
        if (step+1) % 1000 == 0:
            with torch.no_grad():
                # 固定标尺：口径固定 → 曲线可比、可看趋势；
                # 但它取自高频名字（X[:1000]），绝对值有偏，不能当验证集、也不能横向比大小
                loss_test = F.cross_entropy(forward(X_fixed), Y_fixed)
            print(f"第{step+1}/{steps}步, 误差为{loss_test}")

        if (step+1) % 5000 == 0:
            name_list = sample_name(20, forward)
            print(name_list)
    # 验收对比
    with torch.no_grad():
        # 验收用【全量】数据 —— 与 bigram 的 2.4540 同口径才可比
        mlp_nll = F.cross_entropy(forward(X), Y).item()
    print(f"MLP NLL = {mlp_nll:.4f}   vs   bigram 2.4540")
    assert mlp_nll < 2.4540, "MLP 应该优于 bigram"
    draw_embedding(C)


            

    

    


