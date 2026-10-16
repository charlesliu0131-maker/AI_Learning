"""
D22 项目立项：准备中文语料（P1 · 第 1 步）
=====================================================================
做什么：把全唐诗的 JSON 变成「一段纯文本 + 一本 char↔id 字典 +
        train/val 两份 token id 数组」，并用断言证明这四样东西是对的。
跑出什么结论：语料规模、vocab 大小、覆盖率、train/val 各自长度（见文末「跑出的结论」）。

手算参照（写代码时对着核）
---------------------------------------------------------------------
  原始语料   7000 首 → 569,085 字符 / 5,788 种字符
  清洗后     564,784 字符 / 5,787 种字符（清掉 4,301 个空白）
  vocab      5,658（= 只用 train 建出来的字数；含 <unk> 则 5,659）
             —— 用全文建会多收 129 个「val 独有」的字
  切分       train 508,305 / val 56,479 字符（按 0.9；int(564784×0.9) = 508,305）
  embedding  参数量 = vocab × C（本语料 vocab=5,658，C=64 时约 362,112
             —— 而在 names.txt 上只有 27×64 = 1,728）

关键要点（都是实测踩出来的，不是推测）
---------------------------------------------------------------------
  1. ⚠️ 词表必须只用 train 建。实测：用全文建会多收 129 个「val 独有」的字
     （㯕 偓 偘 傜 僓 僨 剉 剋 劬 吚 呱 咥 埳 堡 壖 娩 媮 宕 屴 岏 …），
     等于把验证集的信息泄漏进了词表 —— 而且不报错。
  2. ⚠️ `\n` 也是一个字符，而且在本语料里是第 1 高频（出现 49,185 次，
     比任何汉字都多）。留着它，模型会学到「该换行了」；这不是错，但要自觉选择
     并写进 README。
  3. ⚠️ 标点要统一。「，」和「,」是两个完全不同的 token —— 混用会让 vocab
     无谓变大，纯粹浪费 embedding（`。` 与 `.` 同理）。
  4. ⚠️ 词表要连顺序一起存。只存 stoi 不存 chars 顺序 → 下次加载时 itos
     的构造顺序可能变 → id 全错且不报错。所以 chars / stoi / itos 三个都存。
  5. 本语料是繁体（chinese-poetry 的唐诗原样）。要用简体得额外装 opencc
     （当前环境未装）。自己决定，并写进 README。
  6. 复现性：下载来的语料就是"数据"，不要用随机切分（那样每次跑 train/val
     都不一样）。用「前 N% 做 train」这种确定性切法。

跑出的结论
---------------------------------------------------------------------
  规模      7000 首 → 清洗后 564,784 字符（5,787 种字符）
  词表      vocab = 5,658 字 + 1 个 <unk>；**前 3,450 字就覆盖 99% 的文本**
  切分      train 508,305 / val 56,479（合计 564,784，与 corpus.txt 对账一致）
  <unk>     val 里有 154 个（映射自那 129 个 train 没见过的字）
  结论      语料就绪，vocab 只用 train 建，train/val 已按确定性方式切分。
"""

import json
import re

# ==================== 块 0：配置 ====================
CORPUS_JSON = 'W4/P1/data/tangshi_all.json'   # 下载来的原始语料（相对 Code/ 解析）
OUT_DIR     = 'W4/P1/data'                # 输出目录
TRAIN_RATIO = 0.9
UNK_TOKEN   = '<unk>'                     # val 里遇到 train 没有的字，映射到它


# ==================== 块 1：读语料 ====================
def load_corpus(path):
    """从 chinese-poetry 的 JSON 抽出纯文本。

    结构：json 是一个 list，每条 = {author, paragraphs:[...], title, id}
    （注：全唐诗的条目没有 `tags` 字段，唐诗三百首才有。）
    拼法：title、author、每个 paragraph 各自占一行，末尾用 '\n' 连起来。
    ⚠️ 保留 title / author 是主动取舍，不是顺手：留着它们，模型也会学着
       写标题和作者名。这是可以接受的选择，但要自觉，并写进 README。
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        text = ''
        for item in data:
            title = item['title']
            author = item['author']
            paragraphs = item['paragraphs']
            text += title + '\n' + author + '\n' + '\n'.join(paragraphs) + '\n'
        return text


# ==================== 块 2：清洗 ====================
def clean_text(text):
    """清洗。三条规则，每条都能回答「不清会怎样」：

      ① `\\n{2,}` → `\\n`  压掉连续空行（不清：模型会学到"大段空白"这种没意义的模式）
      ② `[ \\t\\u3000]+` → ''  去掉行内空格 / Tab / **全角空格**（不清：多出几种空白字符，白撑大 vocab）
      ③ `strip()`  去首尾空白（不清：开头结尾的裸换行会被当成一个"字"）
    """
    text = re.sub(r'\n{2,}', r'\n', text)
    text = re.sub(r'[ \t\u3000]+', r'', text)
    text = text.strip()
    return text


# ==================== 块 3：建词表（只用 train！）====================
def build_vocab(train_text):
    """返回 (chars, stoi, itos)。

    ⚠️ 形参就叫 train_text —— 传进来的必须只有 train（用全文会让 val 的字泄漏进词表）。
    chars = 排序后的 train 字集 + `<unk>`（<unk> 排在最后）
    ⇒ <unk> 的 id = len(chars) - 1（下面的 enumerate 已经赋对了）
    ⚠️ 三个都要 return（保存时三个都落盘，见要点 4）。
    """
    chars = sorted(list(set(train_text))) + [UNK_TOKEN]
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text, stoi):
    """字符串 → id 列表。遇到不在 stoi 里的字：映射到 unk（要先把 unk 放进 stoi）。"""
    ids_out = []
    for char in text:
        if char in stoi:
            ids_out.append(stoi[char])
        else:
            ids_out.append(stoi[UNK_TOKEN])
    return ids_out


def decode(ids, itos):
    """id 列表 → 字符串（拼接；查不到的 id 落回 `<unk>`）。"""
    chars_out = []
    for i in ids:
        if i in itos:
            chars_out.append(itos[i])
        else:
            chars_out.append(UNK_TOKEN)
    return ''.join(chars_out)


# ==================== 块 4：切分 + 保存 ====================
def main():
    raw = load_corpus(CORPUS_JSON)
    text = clean_text(raw)
    print(f'原始 {len(raw):,} 字符 / {len(set(raw))} 种字符')
    print(f'清洗后 {len(text):,} 字符 / {len(set(text))} 种字符')

    # 确定性切分：前 90% 做 train、后 10% 做 val（不要用随机，否则每次跑都不一样）
    train, val = text[:int(len(text) * TRAIN_RATIO)], text[int(len(text) * TRAIN_RATIO):]
    assert len(train) + len(val) == len(text)
    # 词表只传 train —— 这是今天最要紧的一条
    chars, stoi, itos = build_vocab(train)
    assert set(chars) - {UNK_TOKEN} <= set(train)
    train_ids = encode(train, stoi)
    val_ids = encode(val, stoi)
    print(f'val 中有{val_ids.count(stoi[UNK_TOKEN])}个 <unk>')


    # 自测区：让断言替你说话
    # ① stoi / itos 必须互逆
    for char in stoi.keys():
        assert itos[stoi[char]] == char

    # ② encode / decode 往返一致（全量跑，实测整套只要 0.29 s）
    for char in train:
        assert decode(encode(char, stoi), itos) == char

    # ⚠️ 落盘放在所有断言之后：先验证、后落盘。
    #    否则断言崩了，四份产物却已经静静躺在盘上 —— 看起来像成功。
    with open(OUT_DIR + '/corpus.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    with open(OUT_DIR + '/vocab.json', 'w', encoding='utf-8') as f:
        json.dump({'chars': chars, 'stoi': stoi, 'itos': itos}, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR + '/train_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(str(i) for i in train_ids))

    with open(OUT_DIR + '/val_ids.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(str(i) for i in val_ids))

    print('✅ D22 通过：语料就绪、vocab 只用 train 建、train/val 已切分')


if __name__ == '__main__':
    main()
