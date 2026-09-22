'''
D09 从「计数」到「训练」—— 用可训练的 W 替代查表

本课的新东西：
  · one-hot 编码：把字符索引变成 27 维向量；xenc @ W 等价于「取出 W 的第 ix 行」
  · 可训练权重 W(27,27)：不再数次数，改由梯度把它学出来
  · 手写 softmax：logits.exp() 之后按行归一化 → 概率
  · 手写 NLL：-log(正确字符的概率).mean()，即交叉熵损失
  · lr=50 为什么这么大：W 给的是 logits，softmax 要 logit 之间差出若干个单位
    才推得动概率；而 D07 拟合 y=2x+1 时参数直接就是系数，lr=0.1 就够

手算参照：训练收敛后，W 学出的概率分布应与纯计数一致
          → 用 diff = max|P - P_count| 检验「学出来的」和「数出来的」是否同一回事

跑出的结论：5000 步后 loss → 2.4546（这就是 bigram 计数模型的极限），
diff = 0.0038 < 1e-2 —— 「用梯度学」和「用数数」殊途同归。
'''

import torch
import torch.nn.functional as F

def generate(W, n):
    for _ in range(n):
        ix = 0
        out = []
        while True:
            ix_ts = F.one_hot(torch.tensor(ix), num_classes=27).float()
            prob = ix_ts @ W
            prob = prob.exp()
            prob = prob / prob.sum()
            ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        print("".join(out))
    return None


# 导入 人名 数据集
words = open('names.txt').read().splitlines()

# 1.建立字符<->索引映射
chars = sorted(list(set(''.join(words))))

stoi = {ch:i+1 for i, ch in enumerate(chars)} # 字符->索引
stoi['.'] = 0 # 句子结束符
itos = {i:ch for ch, i in stoi.items()} # 索引->字符

# 2.构造训练样本
xs = []
ys = []
bigrams = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix = stoi[ch1]
        iy = stoi[ch2]
        bigrams[(ch1, ch2)] = bigrams.get((ch1, ch2), 0) + 1
        xs.append(ix)
        ys.append(iy)

assert len(xs) == sum(bigrams.values()), "训练数据 = 统计数据"

# 本课新知识：one-hot 把字符索引变成 27 维向量（只有对应位置是 1）。
# 它的用途是配合下面的 "one_hot @ W" —— 一步完成"挑出 W 的某一行"。
xenc = F.one_hot(torch.tensor(xs), num_classes=27).float() # n * 27

g = torch.Generator().manual_seed(2147483647) # 设置随机种子,保证每次生成的名字相同

# 3.构建神经网络模型
# 本课新知识：W 就是"这张待学习的概率表"，但里面存的是 logits（原始分数、可正可负），
# 不是概率 —— 所以后面要过一道 softmax 才能当概率用。
W = torch.randn((27, 27), requires_grad=True, dtype=torch.float32, generator=g) # 27个字符,每个字符对应一个长度为27的向量

# 训练前,输出的名字乱七八糟
generate(W, 10)

# 本课重点：lr=50 大得离谱，但这里必须这么大 ——
# W 存的是 logits，softmax 要 logit 之间差出若干个单位才推得动概率，
# 所以参数离"目标值"很远；对比 D07 拟合 y=2x+1 时参数直接就是系数，lr=0.1 就够。
optimizer = torch.optim.SGD([W], lr = 50)
epochs = 2000

for epoch in range(epochs):

    logits = xenc @ W # n * 27 —— one_hot 乘矩阵 = 取出 W 的对应行（最小的「线性层」）

    # 本课新知识：手写 softmax —— exp 先变正，再按行归一化（行和 = 1）
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    # 本课新知识：手写 NLL —— 取"正确答案那一列"的概率，取 log、取负、再取平均
    loss = -probs[torch.arange(xenc.shape[0]), ys].log().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

assert loss.item() < 2.5

# 训练后,结果较好
generate(W, 10)

N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
P_count = N.float()
P_count = P_count / P_count.sum(1, keepdim=True) # 每一行除以该行的总和,得到每一行的概率分布

P = W.exp()
P = P / P.sum(dim=1, keepdim=True)

# 本课重点：把"学出来的 P"与"数出来的 P_count"直接对比 ——
# 两者一致，就说明"用梯度学"和"用统计数"殊途同归（NLL 的最优解 = 经验频率）。
diff = (P - P_count).abs().max()
print(diff)
assert diff < 2e-2, "训练结果应接近计数结果"