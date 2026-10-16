'''
D23 · 数据层

职责（单一）：只负责「语料 ↔ token id」。不许 import 模型、不许写训练循环、不许出现超参数字。

两类函数要分清（D23 知识点 4③）：
  prepare_*  —— 只跑一次：JSON 语料 → 清洗 → 编码 → 落盘 .bin
  get_batch  —— 每步都跑：从 .bin 里随机切一段 (x, y)

为什么要用 .bin + np.memmap（实测数字）：
  · .txt(loadtxt)  2.19 MB / 7.66 ms
  · .bin(fromfile) 1.02 MB / 35.4 µs
  · .bin(memmap)           14.4 µs      ⇒ 比 loadtxt 快 533 倍、文件小 2.2 倍
  词表 5659 « 65536 ⇒ uint16 装得下（每个 id 只要 2 字节）

⚠️ 路径必须用 `__file__` 解析，不能用相对 cwd（实测）：
     相对 cwd  → 在 Code/ 下跑就 False，在 P1/ 下跑才 True（随运行位置变）
     相对 __file__ → 两种位置都 True
   这条一旦写错，会出现「在项目目录里能跑、回到 Code/ 就找不到数据」的怪现象。
'''
from pathlib import Path
import json
import numpy as np
import torch

# ---------------- 路径：一律相对本文件，与本文件从哪被调用无关 ----------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data'

UNK_TOKEN = '<unk>'
BIN_DTYPE = np.uint16          # 词表 5659 < 65536 ⇒ uint16 够用


# ============================================================
# 词表：读 / 用（decode、encode 从 D22 搬来）
# ============================================================
def load_vocab(path=None):
    '''

    ⚠️ 一个实测确认过的静默坑：JSON 的键只能是字符串。
       磁盘上的 itos 是 {"0": "...", "1": "...", ...}，读回来键是 str 不是 int。
       后果：`if i in itos` 对任何一个 int id 都判 False ⇒ decode 会把整段文字
             全部退化成 '<unk>'，而且不报错。
       实测：decode([0,1,2,3,4]) → '<unk><unk><unk><unk><unk>'
       修法：读进来就把键转回 int —— `itos = {int(k): c for k, c in raw.items()}`
             （stoi 的值是 int，JSON 数字会保留类型，不用转）

    返回的 vocab_size 应取 len(chars)（chars 已含 <unk>，id 范围 0 ~ 5658）。
    '''
    vocab_path = path or DATA_DIR / 'vocab.json'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    # itos 从 JSON 读回来键是 str，必须转回 int（见上面警告）；用 int() 不用 eval()
    stoi = vocab['stoi']
    itos = {int(i): ch for i, ch in vocab['itos'].items()}
    return vocab, stoi, itos, len(vocab['chars'])


def encode(text, stoi):
    '''把字符串转成 id 列表；词表里没有的字落回 <unk>。（D22 产出，已给全）'''
    ids = []
    for ch in text:
        ids.append(stoi[ch] if ch in stoi else stoi[UNK_TOKEN])
    return ids


def decode(idx, itos):
    '''把 id 列表转回字符串；查不到的 id 落回 <unk>。（D22 产出，已给全）

    ⚠️ 入参是 int 的列表。要传张量，先在调用点 `.tolist()` ——
       D26 踩过这个坑：为了迁就张量入参去改这个公共函数，结果列表反而不能用了。
    '''
    return ''.join(itos[i] if i in itos else UNK_TOKEN for i in idx)


# ============================================================
# prepare：只跑一次 —— 把 D22 的 .txt 转成 .bin（今天的 data.py 负责这一步）
# ============================================================
def prepare_bin(force=False):
    '''

    要点：
      · 用 `np.loadtxt(path, dtype=np.int64).astype(BIN_DTYPE).tofile(bin_path)`
      · 幂等：若 .bin 已存在且 force=False 就直接跳过（避免每次重算）
      · 落盘后校验：`np.fromfile(bin, dtype=BIN_DTYPE)` 的长度 == 原 .txt 行数
      · 顺手打印：两个 .bin 的字节数 + 与 .txt 的比值

    提示：如果 5659 万一涨过 65536，uint16 会静默溢出（id 被截断），
          所以这里加一条 assert 把关 —— 它读的是 vocab.json 的**真实**大小，
          而不是 config 里写的那个数（config 可能与磁盘不同步）。
    '''
    # uint16 溢出的唯一防线：超过 65536 时 id 会被静默截断，且不报错
    _, _, _, vocab_size = load_vocab()
    assert vocab_size < 65536, f'vocab_size = {vocab_size} 超过 uint16 上限 65535，id 会被截断'

    input_file = [DATA_DIR / "train_ids.txt", DATA_DIR / "val_ids.txt"]
    output_file = [DATA_DIR / "train.bin", DATA_DIR / "val.bin"]
    for i in range(len(output_file)):
        if not force and Path(output_file[i]).exists():
            continue
        # 将input_file[i]中的数据以np.int64的格式加载到内存中, 并转化为uint16类型, 然后写入到output_file[i]中
        data = np.loadtxt(input_file[i], dtype=np.int64).astype(BIN_DTYPE)
        data.tofile(output_file[i])
        # 校验：.bin 里的条目数 == 从 .txt 读进来时的条目数
        assert len(np.fromfile(output_file[i], dtype=BIN_DTYPE)) == len(data)

        print(f"文件 {output_file[i].name} 有 {len(data)} 字符.")
        in_file_path = Path(input_file[i])
        out_file_path = Path(output_file[i])
        print(f"文件 {input_file[i].name} 字节数与文件 {output_file[i].name} 的比例为 {in_file_path.stat().st_size/out_file_path.stat().st_size:.2f}")
        

# ============================================================
# get_batch：每步都跑 —— 随机切一段
# ============================================================
def get_batch(split, batch_size, block_size, generator, device='cpu'):
    '''

    要求：
      · split ∈ {'train', 'val'}，分别读 train.bin / val.bin
      · 用 `np.memmap(path, dtype=BIN_DTYPE, mode='r')` 读（不占内存）
      · 起点用 `torch.randint(len(data) - block_size - 1, (batch_size,), generator=g)`
        —— ⚠️ 一定要传 generator：`torch.randint` 不传 generator 会吃【全局随机流】，
        这正是 D12 那次"设了种子还是四次四个结果"的根因（D23 知识点 6）
      · y 是 x 右移一位：`data[i+1 : i+1+block_size]`
      · 取出来的切片要 `.astype(np.int64)` 再 `torch.from_numpy`

    自测：`assert (y[:, :-1] == x[:, 1:]).all()`  —— 右移关系必须成立。
    '''
    if split == 'train':
        data = np.memmap(DATA_DIR / "train.bin", dtype=BIN_DTYPE, mode='r')
    elif split == 'val':
        data = np.memmap(DATA_DIR / "val.bin", dtype=BIN_DTYPE, mode='r')
    else:
        raise ValueError("split must be 'train' or 'val'")

    # generator 的 device 必须和要生成的张量一致，否则报
    # RuntimeError: Expected a 'mps' device type for generator but found 'cpu'
    # （注意：mps 生成器的 device 字符串是 'mps:0'，所以比 .type 而不是整串）
    assert generator.device.type == torch.device(device).type, (
        f'generator 在 {generator.device.type} 上、张量要放 {device} 上 —— 两者必须一致')

    i = torch.randint(len(data) - block_size - 1, (batch_size,), device=device, generator=generator)
    indices = (i[:, None] + torch.arange(block_size).to(device)).to('cpu')
    x = torch.from_numpy(data[indices].astype(np.int64))
    y = torch.from_numpy(data[indices + 1].astype(np.int64))
    
    assert (y[:, :-1] == x[:, 1:]).all()

    return x.to(device), y.to(device)

if __name__ == '__main__':
    prepare_bin(True)
    g = torch.Generator()
    x, y = get_batch('train', 32, 128, generator=g)
    print(x.shape, y.shape)
