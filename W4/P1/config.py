'''
D23 · P1 配置中心

职责（单一）：把全部超参集中到这一处。本文件不许 import 模型、不许读数据、不许训练。

为什么要集中（D23 知识点 3）：
  · 改一个数只动 1 个文件（散落式要改 train.py / generate.py 两处 ⇒ 迟早不一致）
  · 可以整体存进 checkpoint（`asdict(CFG)`）⇒ 一年后打开 ckpt 也知道模型该怎么重建
  · 对照：nanoGPT 用 `config/` 目录 + 命令行覆盖，那是给「几十组扫描实验」用的；
    我们只有一个 P1，多份配置是纯粹的维护成本

手算参照（V=5659, block_size=128, 权重共享 ⇒ head 参数为 0）：
  一个 block = 注意力 4C² + FFN (8C² + 5C) + LN 2×(2C)
             C=128 时 = 65,536 + 131,712 + 512 = 197,760   （FFN 占 66.6%）
  整个模型   = tok(V×C) + pos(T×C) + N×block + final LN(2C)
             C=128, N=4 时 = 724,352 + 16,384 + 791,040 + 256 = 1,532,032 ≈ 1.53M

各选型的参数总量（供你选值参考，单位 M）：
  C=64  N=2  → 0.47M ｜ C=128 N=2 → 1.14M ｜ C=128 N=4 → 1.53M
  C=192 N=3  → 2.44M ｜ C=256 N=2 → 3.06M ｜ C=256 N=4 → 4.64M
  粗估训练速度：每步计算量 ≈ 6 × 参数量 × (batch_size × block_size)
  （CPU 约 10~50 GFLOPS；C=128/N=4/batch=32/block=128 ⇒ 约 37.7 GFLOP/步）

本工程实际取值：vocab_size=5659 · block_size=128 · n_embd=128 · n_layer=2 · batch_size=32
                lr=1e-3 · max_steps=2000 · eval_interval=100 · eval_iters=50 · seed=131
  ⇒ 参数量实测 = 1,136,512（与上面手算式一致）；实测速度 CPU 79.9 ms/步、MPS 53.9 ms/步。
'''
from dataclasses import dataclass, asdict
import torch
from pathlib import Path

@dataclass
class Config:
    # ---------------- 数据 ----------------
    vocab_size: int = 5659

    block_size: int = 128                   # T

    # ---------------- 模型 ----------------
    n_embd: int = 128                       # C

    n_layer: int = 2

    # ---------------- 训练 ----------------
    batch_size: int = 32                    # B

    lr: float = 1e-3

    max_steps: int = 2000

    eval_interval: int = 100

    eval_iters: int = 50

    seed: int = 131

    # ---------------- 设备 / 路径 ----------------
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'

    out_dir: Path = Path(__file__).resolve().parent / 'out'


CFG = Config()


# 便于 checkpoint 时整体存档（知识点 7）
def cfg_dict():
    '''把配置转成可 JSON/ pickle 的普通 dict，存进 checkpoint。'''
    return asdict(CFG)


if __name__ == '__main__':
    print(cfg_dict())
    assert None not in asdict(CFG).values(), '还有字段没填'
    # out_dir 是 Path ⇒ 存进 checkpoint 后必须用 torch.load(..., weights_only=False) 读回来
    assert isinstance(CFG.out_dir, Path)
