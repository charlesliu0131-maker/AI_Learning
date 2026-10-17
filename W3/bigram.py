import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# -----------------------------------------------------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 提取出全部的字符
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 创建字符到索引的映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encoder = lambda s: [stoi[c] for c in s]  # 将字符串编码为整数列表
decoder = lambda l: ''.join([itos[i] for i in l])  # 将整数列表解码为字符串

# 创建训练集和验证集
data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 创建数据生成器
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """一个注意力头"""
    tril: torch.Tensor
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)  # (B,T,T)
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 这里的投影层是将多头注意力的输出拼接后映射回原始嵌入维度，后续才能做残差连接
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """一个简单的前馈神经网络"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)  # 添加 Dropout 防止过拟合
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer 块：多头自注意力 + 前馈神经网络"""
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 残差连接
        x = x + self.sa_head(self.ln1(x))
        # 残差连接
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads=4) for _ in range(4)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        # idx 和 targets 的形状为 (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C) 最后一层的 LayerNorm
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # 保留最后 block_size 个字符作为上下文
            logits, _ = self(idx_cond)  # 前向传播
            logits = logits[:, -1, :]  # 取最后一个时间步的预测
            probs = F.softmax(logits, dim=-1)  # 转换为概率分布
            idx_next = torch.multinomial(probs, num_samples=1)  # 从概率分布中采样下一个字符
            idx = torch.cat((idx, idx_next), dim=1)  # 将新采样的字符添加到序列中
        return idx

# 创建模型
model = BigramLanguageModel()
m = model.to(device)

# 创建优化器
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# 训练模型
for iter in range(max_iters):
    # 每 eval_interval 次迭代评估模型
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 获取一个批次的数据
    xb, yb = get_batch('train')

    # 前向传播
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 从模型中生成一些文本
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))
