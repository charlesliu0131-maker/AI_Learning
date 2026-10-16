'''
D23 骨架 · D26 温度与 top-k —— 采样生成入口

职责（单一）：加载 checkpoint → 自回归生成文本。
不许在这里定义模型结构（那是 model.py 的事）、不许读训练数据。

D23 先搭出「能加载 + 能吐字」的骨架；**温度 / top-k 在 D26 补上**。

⚠️ 本文件最能体现「checkpoint 必须存 config」（D23 知识点 7）：
   你只有权重时，得靠记忆猜 n_embd / n_layer / block_size 才能重建模型；
   存了 config，直接 `ckpt['config']` 就能重建 —— 一年后打开也一样。
'''
import torch

from config import CFG
from data import load_vocab, decode
from model import MiniGPT
from pathlib import Path


def load_model(ckpt_path):
    '''
    TODO(4a) 从 checkpoint 重建模型。

    要点：
      · `ckpt = torch.load(ckpt_path, map_location='cpu')`
      · 用 `ckpt['config']` 里的四个数重建 MiniGPT（不要用当前 CFG —— 可能已被改过）
      · `model.load_state_dict(ckpt['model'])`，然后 `model.eval()`
    返回 (model, ckpt)。建议顺手打印 ckpt 里存了哪些 key 和哪个 step。
    '''
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = MiniGPT(ckpt['config']['vocab_size'], ckpt['config']['n_embd'], ckpt['config']['block_size'], ckpt['config']['n_layer'], tie_weights=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt



@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None,
             generator=None, device='mps'):
    '''
    D26 · 自回归生成：每次只算最后一个位置的 logits，解码后再抽一个字接上去。

    骨架要点（D23 已写对，别改坏）：
      · 每次循环 `idx_cond = idx[:, -block_size:]` —— 不能超过模型见过的最大长度
        （位置编码表只有 block_size 行，超了会 IndexError）
      · 只取最后一个位置：`model(idx_cond)[:, -1, :]`

    TODO(D26-1) temperature：在 softmax 之前，把 logits 除以 temperature
                （T → 0 趋近 argmax；T = 1 就是模型原本的分布；T > 1 更平、更放飞）

    TODO(D26-2) top_k：只保留最大的 k 个 logit，其余填 -inf（top_k 为 None 时整段跳过）
                提示：用 `torch.topk(logits, k).values[:, [-1]]` 取出「第 k 大」当门槛，
                      再 `logits < 门槛 ⇒ -inf`。
                      别用「第 k 名的下标」去比较 —— 概率并列时会多留几个，k 就不准了。

    TODO(D26-3) generator：`torch.multinomial(prob, num_samples=1, generator=generator)`
                —— 不传就吃全局随机流、生成不再可复现（D12 / D23 的坑）
    '''
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :]          # 只要最后一个位置, [B, vocab_size]
        # ↓ TODO(D26-1) 在这里除以 temperature；TODO(D26-2) 在这里做 top-k 截断
        logits = logits / temperature
        if top_k is not None:
            logits_top, _ = torch.topk(logits, k=top_k)
            logits_min, _ = logits_top.min(dim=-1, keepdim=True)
            # 原地改写。安全的前提是上一行已把 logits 换成了「新张量」；
            # 若直接在 model(...)[:, -1, :] 那个「视图」上原地改，会把原输出也改掉
            logits[logits < logits_min] = float('-inf')
        prob = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(prob, num_samples=1, generator=generator)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx.to('cpu')


if __name__ == '__main__':
    # 主流程（D23 已完成）：读 vocab → load_model → 起始串 → generate → decode 打印
    # 提示：decode 收的是 int 列表 —— 传张量要先 .tolist()（见 data.py 的说明）
    vocab, stoi, itos, _ = load_vocab()
    model, ckpt = load_model(Path(__file__).resolve().parent / 'out/ckpt.pt')
    model.to(CFG.device)
    print(f"ckpt: step = {ckpt['step']}")

    # ---- 温度三档对比：同一个起始串、同一个 seed ⇒ 三组只差 T 这一个变量 ----
    # 起始串用 '\n'：语料格式是「标题\n作者\n正文…」，而 \n 是全语料最高频字符（49,185 次），
    #   '\n' 后面模型会开始写标题。（别给词表外的字：会变 <unk>，模型没学过以它开头）
    # 生成器在循环外建一次、每档开头重设 seed ⇒ 三档的随机流起点相同
    g = torch.Generator(device=CFG.device).manual_seed(131)
    for T in (0.5, 1.0, 1.5):
        g.manual_seed(131)
        idx = torch.tensor([[stoi['\n']]]).to(CFG.device)
        print(f'=== T = {T} ===')
        idx = generate(model, idx, max_new_tokens=100, block_size=CFG.block_size,
                       temperature=T, top_k=20, generator=g, device=CFG.device)
        print(decode(idx[0][1:].tolist(), itos))      # [1:] 跳过起始串，只看模型自己续写的部分

    # ---- （选做）top_k 对比：T 固定 1.5，看 k = None / 20 / 5 的差别 ----
    # 预期：不设 top_k 时更容易蹦出「不成词的怪字」
    for top_k in (None, 20, 5):
        g.manual_seed(131)
        idx = torch.tensor([[stoi['\n']]]).to(CFG.device)
        print(f'=== top_k = {top_k} ===')
        idx = generate(model, idx, max_new_tokens=100, block_size=CFG.block_size,
                       temperature=1.5, top_k=top_k, generator=g, device=CFG.device)
        print(decode(idx[0][1:].tolist(), itos))
