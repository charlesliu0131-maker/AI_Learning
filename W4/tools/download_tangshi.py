"""
下载「全唐诗」并合并为单文件（P1 语料准备工具）
=====================================================================
用途：把 chinese-poetry 仓库里的 `poet.tang.*.json` 分片全部拉下来，
      合并成一个 JSON（保留结构）和一个纯文本 txt（直接可当语料）。
      不是学习任务的一部分 —— 这是「准备数据」的工具脚本。

用法（根目录 = Code/）：
    cd Code
    .venv/bin/python W4/tools/download_tangshi.py            # 全量 58 片
    .venv/bin/python W4/tools/download_tangshi.py --shards 13  # 只要前 13 片

为什么会想只取一部分：实测 3 片 = 3,002 首 / 229,958 字符
    ⇒ 58 片外推 ≈ 58,039 首 / 约 4.45 M 字符
    ⇒ 每片约 7.7 万字符；取前 13 片 ≈ 100 万字符（正好落在甜点区 50万~200万）

⚠️ 两个实测踩过的坑
---------------------------------------------------------------------
  1. **URL 里的中文必须先 percent-encode** —— `全唐诗` 直接塞进 urlopen 会报
     `UnicodeEncodeError: 'ascii' codec can't encode characters`。
     用 `urllib.parse.quote('全唐诗')`。
  2. **`全唐诗/` 目录里装的不只是唐诗** —— 实测该目录 133 MB 中：
       poet.song.*.json（**宋诗**）  255 个文件 / 104.1 MB  ← 大头
       poet.tang.*.json（唐诗）       58 个文件 /  24.4 MB  ← 我们要的
     只取 `poet.tang.*` 就省掉 4/5 的流量。

  3. 分片编号是 0, 1000, 2000, … 57000（步长 1000，每片 1000 首），
     所以可以**直接用 range 生成 URL**，不用调 GitHub API（少一次限流风险）。

跑出的结论（跑完自动打印）
---------------------------------------------------------------------
  实际下载的分片数 / 总首数 / 纯文本字符数 / 不重复字符数 / 输出文件路径
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

# ---------------- 路径：按脚本位置解析，从哪运行都对 ----------------
ROOT = Path(__file__).resolve().parent.parent          # → Code/W4/tools/.. = Code/W4
DATA_DIR = ROOT / 'P1' / 'data'
SHARD_DIR = DATA_DIR / 'shards'                        # 分片缓存（断点续传用）
OUT_JSON = DATA_DIR / 'tangshi_all.json'
OUT_TXT = DATA_DIR / 'tangshi_all.txt'

# ---------------- 语料源 ----------------
BASE = ('https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/'
        + quote('全唐诗') + '/')                       # ← 坑 1：必须 quote
ALL_IDS = list(range(0, 58000, 1000))                  # 0…57000 共 58 片（坑 3）
UA = {'User-Agent': 'Mozilla/5.0'}


def fetch_shard(n, retry=3):
    """下一个分片；已有缓存就直接读缓存（断点续传）。"""
    cache = SHARD_DIR / f'poet.tang.{n}.json'
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))

    url = f'{BASE}poet.tang.{n}.json'
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode('utf-8')
            cache.write_text(raw, encoding='utf-8')
            return json.loads(raw)
        except Exception:
            if attempt == retry - 1:
                raise
    return []


def to_text(poems):
    """把诗的结构拍成纯文本：标题一行、作者一行、每句一行。

    ⚠️ 想好要不要 title / author —— 留着它们，模型会学着写标题和作者名。
       想去掉就把上面两行删掉（这一步的取舍要写进 README）。
    """
    lines = []
    for p in poems:
        lines.append(p.get('title', ''))
        lines.append(p.get('author', ''))
        lines.extend(p.get('paragraphs', []))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, default=len(ALL_IDS),
                    help=f'取前几片（每片约 1000 首 / 7.7 万字符；默认全取 {len(ALL_IDS)} 片）')
    ap.add_argument('--workers', type=int, default=8, help='并发数')
    args = ap.parse_args()

    ids = ALL_IDS[:args.shards]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    print(f'下载 {len(ids)} 个分片 → {SHARD_DIR}')
    poems, per_shard = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, part in enumerate(ex.map(fetch_shard, ids), 1):
            assert isinstance(part, list), f'分片 {ids[i-1]} 不是 list，格式变了？'
            per_shard.append(len(part))
            poems.extend(part)
            print(f'  [{i:>2}/{len(ids)}] 本片 {len(part):>5} 首，累计 {len(poems):>7,} 首', end='\r')
    print()

    assert len(poems) == sum(per_shard), '合并数与各分片之和不符'
    assert len(poems) > 0, '一首都没下到，检查网络或 URL'

    text = to_text(poems)
    OUT_JSON.write_text(json.dumps(poems, ensure_ascii=False, indent=1), encoding='utf-8')
    OUT_TXT.write_text(text, encoding='utf-8')

    print()
    print('=== 跑出的结论 ===')
    print(f'  分片数      : {len(ids)}')
    print(f'  总首数      : {len(poems):,}')
    print(f'  纯文本字符  : {len(text):,}')
    print(f'  不重复字符  : {len(set(text)):,}')
    print(f'  输出        : {OUT_JSON.relative_to(ROOT.parent)}')
    print(f'                {OUT_TXT.relative_to(ROOT.parent)}')
    print()
    print('✅ 全唐诗下载并合并完成')


if __name__ == '__main__':
    main()
