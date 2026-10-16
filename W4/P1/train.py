'''
D23 · 训练入口

职责（单一）：串起 config / data / model，跑训练循环。
不许在这里重新定义模型结构、不许在这里解析语料。

验收标准：`python train.py` 至少能启动。
把「能启动」拆成可检查的四条（D23 知识点 6），今天先把这四条做完、逐条 print：

    ① import 不报错     ② 数据能被读到     ③ 模型能建出来     ④ 前向一步不炸
  ⇒ 这四条 = TODO(3a) ~ TODO(3d)。先跑通它们，再往 TODO(3e) 加训练循环。

怎么运行（两种都行，因为 import 用的是「同目录裸 import」）：
    cd Code        && .venv/bin/python W4/P1/train.py
    cd Code/W4/P1  && ../../.venv/bin/python train.py
  （⚠️ 不要用 `python -m W4.P1.train`：那要求包内相对导入 `from .config import ...`，
    而裸 import 在 -m 下会 ModuleNotFoundError。实测过，见 D23 知识点 3 的对照表。）

⚠️ 可复现的坑（D12 教训）：`torch.randint` 等不传 generator 的随机调用会吃【全局随机流】。
   所以「设了 seed」≠「全可复现」—— 每一处随机调用都要检查。
'''
import time
import math
import torch
import torch.nn as nn
import matplotlib

from config import CFG, cfg_dict
from data import prepare_bin, get_batch, DATA_DIR
from model import MiniGPT


def estimate_loss(model, block_size, batch_size, device, eval_iters, generator):
    '''
    TODO(3f) 在 train / val 上各取 eval_iters 个 batch，求平均 loss。

    要点：必须 `model.eval()` + `with torch.no_grad():`，算完再 `model.train()` 切回来。
          忘掉 eval() 会让 dropout / 统计量行为不一致，loss 看起来"忽高忽低"。
          返回 dict，例如 {'train': 1.23, 'val': 1.45}。
    '''
    model.eval()
    with torch.no_grad():
        train_losses = []
        val_losses = []
        for _ in range(eval_iters):
            for split in ['train', 'val']:
                x, y = get_batch(split, batch_size, block_size, device=device, generator=generator)
                logits = model(x)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
                if split == 'train':
                    train_losses.append(loss.item())
                elif split == 'val':
                    val_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        model.train()
        return {'train': train_loss, 'val': val_loss}


def main():
    # ---------------- ① 设备与随机种子 ----------------
    torch.manual_seed(CFG.seed)
    device = CFG.device
    print(f'[1/4] device = {device}')
    g = torch.Generator(device=device).manual_seed(CFG.seed)
    print(f'[1/4] seed = {CFG.seed}')

    # ---------------- ② 数据 ----------------
    print(f'[2/4] 数据目录 = {DATA_DIR}')
    prepare_bin()          # 幂等：.bin 已存在就跳过
    train_x, train_y = get_batch('train', CFG.batch_size, CFG.block_size, device=device, generator=g)
    print(train_x.shape, train_y.shape)
    val_x, _ = get_batch('val', CFG.batch_size, CFG.block_size, device=device, generator=g)
    assert (train_x < CFG.vocab_size).all() and (val_x < CFG.vocab_size).all()


    # ---------------- ③ 模型 ----------------
    # TODO(3c) 用 CFG 建 MiniGPT，打印参数量与「FFN 占比」核对手算参照
    #          提示：参数量应为 V×C + T×C + N×block + 2C（head 共享 ⇒ 不单独计）
    model = MiniGPT(CFG.vocab_size, CFG.n_embd, CFG.block_size, CFG.n_layer, tie_weights=True).to(device)
    p_num = sum(p.numel() for p in model.parameters())
    block_num = sum(p.numel() for block in model.blocks for p in block.parameters() if p.requires_grad)
    ffn_num = sum(p.numel() for block in model.blocks for p in block.ffn.parameters() if p.requires_grad)
    print(f'[3/4] 参数量 = {p_num}, FFN占Block比例 = {ffn_num/block_num:.2%}')
    print(f'[3/4] 手算参数量 = {CFG.vocab_size * CFG.n_embd + CFG.block_size * CFG.n_embd + CFG.n_layer * (2 * CFG.n_embd + CFG.n_embd ** 2 * 4 + 2 * CFG.n_embd + CFG.n_embd ** 2 * 8 + 5 * CFG.n_embd) + 2 * CFG.n_embd}')

    # ---------------- ④ 前向一步 + 初始 loss 体检（D24 的验收线） ----------------
    # D23 只查到「形状对」；D24 再加一道「loss 值对」—— 两道都过才算链路通。
    # 口径：V = 5659 / batch 32×128 / σw = 0.02 ⇒ 期望落在 8.64 ~ 8.70
    logits = model(train_x)
    print(logits.shape)                                                   # (B, T, V)
    assert logits.shape == (CFG.batch_size, CFG.block_size, CFG.vocab_size)

    ln_v = math.log(CFG.vocab_size)        # 瞎猜基线：均匀预测时 −log(1/V) = ln V
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), train_y.view(-1))
    loss_val = loss.item()

    print(f'[4/4] 初始 loss = {loss_val:.4f}, ln_v = {ln_v:.4f}, 差值 = {loss_val - ln_v:.4f}')

    # 双边断言（缺一不可）—— 只写一边会放过一整类错误（D14-R2 的教训）
    assert loss_val < ln_v * 1.05, f'初始 loss = {loss_val:.3f} 远高于基线 {ln_v:.3f} ⇒ 查初始化/接线'
    # ⚠️ 这条下界在「训练 0 步」场景天然不会触发 —— Jensen 不等式：fresh 模型的平均 loss ≥ ln V
    #    （实测把 y 换成 x 也压不低：8.23 > 7.34，因为随机初始化的模型还没有「读出答案」的能力）
    #    它真正的身份是【防回归】：拿训过的 ckpt 当新模型体检（实测 4.97）才会炸
    assert loss_val > ln_v * 0.85, f'初始 loss = {loss_val:.3f} 远低于基线 {ln_v:.3f} ⇒ 查掩码/错位/泄漏'

    # ============================================================
    # ⑤ 正式训练（D25）
    # 循环沿用 D23 的骨架；D25 往里加了两件事：
    #   (a) 每个评估点用 estimate_loss 记下 train / val —— 曲线是「记」出来的，不是训练完补算的
    #   (b) 训练结束后把曲线画出来落盘，并把 history 一起存进 checkpoint
    # ============================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    start = time.time()

    # 每个评估点的记录（画曲线用）
    steps, train_hist, val_hist = [], [], []

    for step in range(CFG.max_steps):
        train_x, train_y = get_batch('train', CFG.batch_size, CFG.block_size, generator=g, device=device)
        logits = model(train_x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, CFG.vocab_size), train_y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # step == 0 那次评估发生在「训练 1 步之后」；真正的初始值是 ④ 区打的 8.6931
        if step == 0 or (step + 1) % CFG.eval_interval == 0:
            # 曲线的值取 estimate_loss 的 train / val（多 batch 平均）——
            # 不要用单步的 loss.item()：那只是 1 个 batch，噪声大、看不出趋势（D25 知识点 3）
            est = estimate_loss(model, CFG.block_size, CFG.batch_size,
                                device=device, eval_iters=CFG.eval_iters, generator=g)
            steps.append(step + 1)
            train_hist.append(est['train'])
            val_hist.append(est['val'])
            print(f'✅ step = {step+1}, 训练集 loss = {est["train"]:.4f}, '
                  f'验证集 loss = {est["val"]:.4f}, time = {time.time() - start:.2f} s')

    result = estimate_loss(model, CFG.block_size, CFG.batch_size, device=device, eval_iters=CFG.eval_iters, generator=g)
    print(f"训练完成：训练集 loss = {result['train']:.4f}, 验证集 loss = {result['val']:.4f}")

    # ---------------- 画 loss 曲线 ----------------
    # 「先 matplotlib.use('Agg')、再 import pyplot」的顺序不能反 —— 反了后端就已经定死了
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['PingFang SC','Heiti SC','STHeiti','Arial Unicode MS','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(ln_v, color='#8A7B66', ls='--', lw=1.2, label=f'ln(V) = {ln_v:.4f}（瞎猜基线）')
    ax.plot(steps, train_hist, label='train', color='#2F6D4F')
    ax.plot(steps, val_hist, label='val', color='#C96F4A', ls='--')
    ax.set_xlabel('steps')
    ax.set_ylabel('loss')
    ax.set_title(f'D25 训练曲线（V = {CFG.vocab_size:,}）')
    ax.grid(alpha=.3, ls=':')
    ax.legend()
    plt.savefig(CFG.out_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------------- checkpoint ----------------
    # config 必须存（D23 知识点 7）；history 顺手带上 —— 以后重画曲线 / 写 README 不用再训一遍
    # step 存「已完成的步数」= step + 1：循环变量停在 max_steps - 1，直接存会少算一步
    torch.save({'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'step':  step + 1,
                'config': cfg_dict(),
                'history': {'steps': steps, 'train': train_hist, 'val': val_hist}},
                CFG.out_dir / 'ckpt.pt')

    # 自检已完成（2026-10-14）：同 seed 连跑两次 ——
    #   CPU：逐点完全相同（max|Δ| = 0）⇒ 随机流确实完全受控（D12 教训成立）
    #   MPS：max|Δ| ≈ 1.5e-04 —— 不是 RNG 问题，是 GPU 归约顺序不确定
    #   ⇒「可复现」要分层：「随机流受控」≠「逐位相同」

    print('✅ D25 通过：训练完成 + loss 曲线已保存 + checkpoint 已落盘')



if __name__ == '__main__':
    main()
