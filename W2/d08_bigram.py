'''
D08 Bigram 计数模型 —— 先不看神经网络，用「数次数」建语言模型

目标：① 建字符 ↔ 索引映射（含 '.' 句子结束符，共 27 类）
      ② 用两种方法统计 bigram 次数（Python 字典 / PyTorch 张量）并互验
      ③ 计数转概率：拉普拉斯平滑（每格 +1）后按行归一化
      ④ 按概率采样，生成新名字

手算参照：P(ch2 | ch1) = N[ch1, ch2] / N[ch1].sum()
          即「m 之后接 a 的概率」= 'ma' 的出现次数 ÷ 以 m 开头的总数

跑出的结论：查表即预测 —— 能生成有「名字形状」的字符串，但字母组合生硬、
长度会失控（因为只看前 1 个字符，信息量就这么多）。样例：
  ['cexze', 'momasurailezitynn', 'konimittain', 'llayn', 'ka', 'da', ...]
这张 27×27 的计数表，就是后面所有模型要超越的起点。
'''

import torch
import matplotlib.pyplot as plt

# 导入 人名 数据集
words = open('names.txt').read().splitlines()

# 1.建立字符<->索引映射
chars = sorted(list(set(''.join(words))))

stoi = {ch:i+1 for i, ch in enumerate(chars)} # 字符->索引
stoi['.'] = 0 # 句子结束符
itos = {i:ch for ch, i in stoi.items()} # 索引->字符

assert itos[stoi['c']] == 'c' , "itos[stoi['c']] != 'c'"

# 2.统计bigram出现的次数
# 方法一:使用Python字典数一遍
bigrams = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        bigrams[bigram] = bigrams.get(bigram, 0) + 1

# 方法二:使用PyTorch张量数一遍
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

assert sum(bigrams.values()) == sum(N.numpy().flatten()), "bigrams.values() != N.numpy().flatten().tolist()"

# 可视化
plt.figure(figsize=(16,16))
plt.imshow(N.numpy(), cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i,j].item(), ha='center', va='top', color='gray')

plt.axis('off')
plt.savefig('./assert/d08_bigram_embedding.jpeg')
plt.close()

# 3.将bigram出现次数转换为概率
'''
使用拉普拉斯平滑（给每个计数都 +1）,
概率会略低于纯计数版（你复盘想验证的"m→a ≈ 0.42"会比纯计数小一点）；
结束符 . 的概率也被稀释了 → 生成的名字会偏长。
'''
P = (N+1).float()
P = P / P.sum(1, keepdim=True) # 每一行除以该行的总和,得到每一行的概率分布

# 4.根据bigram概率生成名字
g = torch.Generator().manual_seed(2147483647) # 设置随机种子,保证每次生成的名字相同
names = []
for _ in range(20):
    out = []
    ix = 0 # 从句子开始符开始
    while True:
        p = P[ix] # 取出当前字符的概率分布
        ix = torch.multinomial(p, num_samples=1, replacement = True, generator=g).item() # 根据概率分布采样下一个字符的索引
        if ix == 0: # 如果采样到句子结束符,则结束
            break
        out.append(itos[ix]) # 将索引转换为字符
    names.append(''.join(out)) # 将生成的名字添加到列表中
print(names)

assert len(names) == 20

assert N.shape == (27, 27), "N.shape != (27, 27)"
for i in range(27):
    assert abs(P[i].sum().item() - 1) < 1e-5, f"P[{i}].sum() != 1.0"