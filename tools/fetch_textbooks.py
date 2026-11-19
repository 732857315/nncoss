"""按已提交的清单下载课本 PDF：分卷自动拼接，每个分卷按 Git 对象哈希核对。

清单 modelbook/knowledge/textbooks-manifest.json 记录了 ChinaTextbook 仓库中选用的每册课本的路径、
分卷大小与 Git 对象哈希。默认把 PDF 下载到 modelbook/textbooks/pdf/（不纳入版本库）。

调整选书时，先从 GitHub 取得完整文件树，再用 --select 按 WANT 中的版本规则重新生成清单：
  curl -sSL "https://api.github.com/repos/TapXWorld/ChinaTextbook/git/trees/master?recursive=1" -o modelbook/textbooks/tree.json
  python3 tools/fetch_textbooks.py --select modelbook/textbooks/tree.json
"""

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/TapXWorld/ChinaTextbook/master/"
SOURCE = "https://github.com/TapXWorld/ChinaTextbook"
# 学段 → 学科 → 版本目录名前缀。每门学科只取一个版本：优先人教版或统编版，没有时取册数最多的版本；
# 地理图册取与地理课本配套的人教版。空前缀表示该学科的全部版本（中考真题与练习，各套内容互不重复）。
# 只覆盖小学到高中；源仓库的五四学制目录与本表中六三学制的课本内容相同，大学教材不在范围内。
WANT = {
    "小学": {"语文": "统编版", "数学": "人教版", "英语": "人教版（PEP）（三年级起点）", "科学": "教科版", "道德与法治": "统编版",
             "音乐": "人教版（五线谱）", "美术": "人教版", "体育与健康": "人教版", "艺术": "教科版", "语文·书法练习指导": "人美版"},
    "初中": {"语文": "统编版", "数学": "人教版", "英语": "人教版", "物理": "人教版", "化学": "人教版", "生物学": "人教版",
             "历史": "统编版", "地理": "人教版", "道德与法治": "统编版", "人文地理": "统编版", "科学": "华东师大版",
             "地理图册": "配套人教版", "音乐": "人教版（五线谱）", "美术": "人教版", "体育与健康": "人教版", "艺术": "鄂教版",
             "俄语": "人教版", "日语": "人教版"},
    "高中": {"语文": "统编版", "数学": "人教版（A版）", "英语": "人教版", "物理": "人教版", "化学": "人教版", "生物学": "人教版",
             "历史": "统编版", "地理": "人教版", "思想政治": "统编版", "信息技术": "人教中图版", "通用技术": "人教版",
             "地理图册": "配套人教版", "音乐": "人音版", "美术": "人教版", "体育与健康": "人教版", "艺术": "沪教版",
             "俄语": "人教版", "日语": "人教版"},
    "学数学最重要的刷习题在这里": {"初中练习题_带答案": ""},
}


def select(tree):
    """按 WANT 的学段、学科与版本前缀选书；大文件在源仓库中拆成 .pdf.1、.pdf.2 等分卷。"""
    books = defaultdict(list)
    for entry in tree["tree"]:
        if entry["type"] != "blob":
            continue
        parts = entry["path"].split("/")
        if len(parts) < 4 or parts[0] not in WANT:
            continue
        prefix = WANT[parts[0]].get(parts[1])
        if prefix is None or not parts[2].startswith(prefix):
            continue
        match = re.match(r"(.*\.pdf)(?:\.(\d+))?$", entry["path"])
        if match:
            books[match.group(1)].append((int(match.group(2) or 0), entry))
    return {"source": SOURCE, "commit_tree": tree["sha"],
            "books": [{"book": book, "parts": [{"path": entry["path"], "size": entry["size"], "sha": entry["sha"]}
                                               for _, entry in sorted(parts, key=lambda item: item[0])]}
                      for book, parts in sorted(books.items())]}


def git_blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(part, target):
    url = RAW + urllib.parse.quote(part["path"])
    for attempt in range(6):
        result = subprocess.run(["curl", "-sSL", "--fail", "--max-time", "900", "-o", str(target), url])
        if result.returncode == 0 and target.stat().st_size == part["size"] \
                and git_blob_sha(target.read_bytes()) == part["sha"]:
            return
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"下载失败或哈希不符：{part['path']}")


def fetch(manifest, output, verify):
    books = manifest["books"]
    print(f"清单共 {len(books)} 册，{sum(p['size'] for b in books for p in b['parts']) / 1e9:.2f} GB", flush=True)
    for index, book in enumerate(books):
        target = output / "pdf" / book["book"]
        total = sum(part["size"] for part in book["parts"])
        if target.exists() and target.stat().st_size == total:
            if verify:
                data, offset = target.read_bytes(), 0
                for part in book["parts"]:
                    if git_blob_sha(data[offset:offset + part["size"]]) != part["sha"]:
                        raise SystemExit(f"哈希不符：{target}")
                    offset += part["size"]
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pieces = []
        for number, part in enumerate(book["parts"]):
            piece = target.with_name(target.name + f".part{number}")
            if not (piece.exists() and piece.stat().st_size == part["size"]
                    and git_blob_sha(piece.read_bytes()) == part["sha"]):
                download(part, piece)
            pieces.append(piece)
        with target.open("wb") as handle:
            for piece in pieces:
                handle.write(piece.read_bytes())
        for piece in pieces:
            piece.unlink()
        print(f"{index + 1}/{len(books)} {book['book']}", flush=True)
    print("全部课本已就绪" + ("，哈希核对通过" if verify else ""), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=ROOT / "modelbook/knowledge/textbooks-manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "modelbook/textbooks", help="PDF 保存在 <output>/pdf/ 下")
    parser.add_argument("--verify", action="store_true", help="对已存在的 PDF 也逐分卷核对 Git 对象哈希")
    parser.add_argument("--select", type=Path, metavar="TREE_JSON", help="从 GitHub 文件树重新生成清单并写入 --manifest，不下载")
    args = parser.parse_args()
    if args.select:
        manifest = select(json.loads(args.select.read_text(encoding="utf-8")))
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        counts = Counter("/".join(book["book"].split("/")[:2]) for book in manifest["books"])
        print(f"选中 {len(manifest['books'])} 册 -> {args.manifest}\n{sorted(counts.items())}")
        return
    fetch(json.loads(args.manifest.read_text(encoding="utf-8")), args.output, args.verify)


if __name__ == "__main__":
    main()
