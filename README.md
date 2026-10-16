# AI Learning Journey

从零手写迷你 GPT —— 一个 6 个月转型计划（**Phase 1: W1–W4**）的代码仓库。

每天的脚本都**亲手写完、终端跑通、逐条批改**。这里记录 **W1–W4** 的成果：从"数次数"的 bigram，到能生成英文名字的 MLP，再到**自己手写的 Transformer**，最后是一个能续写古体诗的**中文迷你 GPT**。

---

## 进度

| 周次 | 主题 | 状态 |
|---|---|---|
| **W1** | 数学恢复（3B1B）+ PyTorch 查漏 | ✅ 完成（`d01`–`d07`） |
| **W2** | 反向传播 / 优化器 + makemore（计数模型 → MLP） | ✅ 完成（`d08`–`d13`） |
| **W3** | Self-attention → 手写 Transformer（QKV → 缩放点积 → LN+残差 → 因果掩码+位置编码 → FFN+Block → mini GPT） | ✅ 完成（`d15`–`d20` + 周回顾） |
| **W4** | 项目 P1：中文迷你 GPT（全唐诗 · 1.14M 参数 · 2,000 步） | 🚧 进行中（报告见 [`W4/P1/README.md`](W4/P1/README.md)） |

---

## 关键结果

用**同一份数据、同一个评估口径**（全量交叉熵 / NLL，越低越好）串起来：

| 阶段 | 脚本 | 全量 NLL | 困惑度 |
|---|---|---|---|
| Bigram 计数模型（只看前 1 个字符） | `W2/d08_bigram.py` | 2.4540 | 11.64 |
| 可训练权重 W（把"数出来的表"变成"学出来的表"） | `W2/d09_train_w.py` | 2.4546 | 11.65 |
| **完整 MLP**（看前 3 个字符 + 隐藏层 + tanh） | `W2/d12_mlp_full.py` | **2.2070** | **9.09** |
| MLP + **Adam** 优化器 | `W2/d13_optim_Adam.py` | **2.1588** | **8.66** |

两个值得记住的结论：

- **看前 3 个字符 > 加宽隐藏层**：`h` 从 200 加大到 300（参数 +48%），NLL 几乎不动 —— 瓶颈在"上下文长度"，不在"模型宽度"。这正是 W3 注意力机制的动机。
- **选对优化器 > 加大模型**：参数更少的配置（h=200）配上 Adam，反而比参数更多的手写 SGD 方案（h=300）**更准**。

---

## 目录结构

```
W1/    数学恢复与 PyTorch 基础
W2/    Bigram → MLP（makemore 主线）
W3/    Self-attention → 手写 Transformer
W4/    项目 P1：中文迷你 GPT（P1/ 为完整 7 文件项目）
```

### W1 · 数学恢复 + PyTorch 查漏

| 脚本 | 做什么 |
|---|---|
| `d01_dot_product.py` | 向量点积：两种算法（`torch.dot` / 逐元素相乘再求和）互验 |
| `d02_tensor_basics.py` | 张量创建、加减、`reshape`、`transpose`（用定义式断言验证） |
| `d03_matmul_transform.py` | 矩阵乘法 = 变换：形状规则、点积视角、复合变换的顺序 |
| `d04_derivative_autograd.py` | 数值微分 vs 自动微分：差商与 `autograd` 对答案 |
| `d05_two_layer_backprop.py` | 两层网络的反向传播：链式法则、梯度形状铁律 |
| `d06_train_loop.py` | 最小训练循环（五行）、从零拟合 `y = 2x + 1` |
| `d07_loss_curve.py` | 把训练封装成函数 + 画出 loss 曲线 |

### W2 · 反向传播 / 优化器 + makemore

| 脚本 | 做什么 |
|---|---|
| `d08_bigram.py` | Bigram 计数模型：统计字符共现、拉普拉斯平滑、按概率采样生成"名字" |
| `d09_train_w.py` | 把概率表换成**可训练矩阵 W**，手写 softmax + NLL，训练后与计数版对账 |
| `d10_softmax_ce.py` | 手写 softmax 与交叉熵：naive / 稳定版 / logsumexp 三档数值稳定性对比 |
| `d11_mlp_embedding.py` | 上下文窗口（前 3 字符）+ embedding 查表，验证 `C[X] ≡ one_hot(X) @ C` |
| `d12_mlp_full.py` | 完整 MLP：隐藏层 + `tanh` + 参数初始化 → **NLL 2.2070** |
| `d13_optim_Adam.py` | 优化器实验：SGD vs Adam × 学习率三档（0.001 / 0.01 / 0.1） |
| `names.txt` | 数据集：32033 个英文名字（按出现频率从高到低排序） |

### W3 · Self-attention → 手写 Transformer

| 脚本 | 做什么 |
|---|---|
| `d15_self_attention.py` | Self-attention 的 Q / K / V：用定义式断言验证 `S[i,j] == Q[i]·K[j]` |
| `d16_scaled_dot_product.py` | 缩放点积注意力 + 封装成 `nn.Module`（为什么除以 `√d_k`） |
| `d17_layernorm_residual.py` | LayerNorm + 残差：γ、β 的作用，Pre-LN 放在哪 |
| `d18_causal_mask_pe.py` | 因果掩码（不许看未来）+ 可学习位置编码 |
| `d19_ffn_block.py` | FFN + 完整 GPTBlock（Pre-LN + 残差） |
| `d20_mini_gpt.py` | mini GPT 四段结构 + 权重共享（参数量 2,040） |

### W4 · 项目 P1：中文迷你 GPT

7 文件完整项目在 [`W4/P1/`](W4/P1/) —— **训练报告见 [`W4/P1/README.md`](W4/P1/README.md)**（loss 曲线、生成样例、复现步骤与已知局限）。

---

## 怎么运行

需要 Python 3.13 + PyTorch + matplotlib。

```bash
# W2 的脚本依赖同目录下的 names.txt，所以在 W2 目录里运行
cd W2
python d12_mlp_full.py
```

> **注意**：脚本里的图片保存路径是 `./assert/xxx`，请先在脚本所在目录建好该文件夹：
> ```bash
> mkdir -p assert
> ```

W1 的脚本不依赖数据文件，直接运行即可：

```bash
cd W1
python d06_train_loop.py
```

项目 P1（W4）有独立的依赖与数据准备流程，直接照 [`W4/P1/README.md`](W4/P1/README.md) 的「快速开始」四步跑。

---

## 脚本约定

每个脚本都遵循同一套写法，方便复习时"由脚本回看知识点"：

- **文件头 docstring**：写清「编号 + 主题 + 本课新知识 + 手算参照 + 跑出的结论」
- **手算参照**：先在注释里算出预期值，再让代码去对
- **`assert` 关卡**：断言必须**可能失败**（写之前先问"它在什么情况下会炸"），拒绝恒真的空断言
- **打印关键中间量**：形状、loss、采样结果
- **结尾 `✅`**：一眼看出跑通了

---

## 环境

```
Python  3.13
PyTorch（CPU 版即可）
matplotlib
```

---

## 说明

- 本仓库只包含**代码**；配套的概念笔记、回顾题库等学习资料暂存在本地。
- **W3**（Self-attention → 手写 Transformer）与 **W4 · P1**（中文迷你 GPT）的代码已补入；P1 的训练报告见 [`W4/P1/README.md`](W4/P1/README.md)。
