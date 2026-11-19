import json
import re
import subprocess
import sys
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])
RAW = "https://raw.githubusercontent.com/TapXWorld/ChinaTextbook/master/"
WANT = {
    "小学": {"语文": "统编版", "数学": "人教版", "英语": "人教版（PEP）（三年级起点）", "科学": "教科版", "道德与法治": "统编版"},
    "初中": {"语文": "统编版", "数学": "人教版", "英语": "人教版", "物理": "人教版", "化学": "人教版", "生物学": "人教版",
             "历史": "统编版", "地理": "人教版", "道德与法治": "统编版"},
    "高中": {"语文": "统编版", "数学": "人教版（A版）", "英语": "人教版", "物理": "人教版", "化学": "人教版", "生物学": "人教版",
             "历史": "统编版", "地理": "人教版", "思想政治": "统编版", "信息技术": "人教中图版"},
}


def selected(tree):
    books = defaultdict(list)
    for entry in tree:
        if entry["type"] != "blob":
            continue
        parts = entry["path"].split("/")
        if len(parts) < 4 or parts[0] not in WANT:
            continue
        stage, subject, edition = parts[0], parts[1], parts[2]
        prefix = WANT[stage].get(subject)
        if prefix is None or not edition.startswith(prefix):
            continue
        match = re.match(r"(.*\.pdf)(?:\.(\d+))?$", entry["path"])
        if not match:
            continue
        books[match.group(1)].append((int(match.group(2) or 0), entry["path"], entry["size"]))
    return books


def download(path, target):
    url = RAW + urllib.parse.quote(path)
    for attempt in range(6):
        result = subprocess.run(["curl", "-sSL", "--fail", "--max-time", "900", "-o", str(target), url])
        if result.returncode == 0:
            return
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"下载失败：{path}")


def main():
    tree = json.loads((ROOT / "tree.json").read_text())["tree"]
    books = selected(tree)
    print("选中", len(books), "册，", sum(size for parts in books.values() for _, _, size in parts) / 1e9, "GB", flush=True)
    if "--list" in sys.argv:
        from collections import Counter
        print(sorted(Counter("/".join(book.split("/")[:2]) for book in books).items()))
        return
    for index, (book, parts) in enumerate(sorted(books.items())):
        target = ROOT / "pdf" / book
        if target.exists() and target.stat().st_size == sum(size for _, _, size in parts):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pieces = []
        for number, path, size in sorted(parts):
            piece = target.with_name(target.name + f".part{number}")
            if not (piece.exists() and piece.stat().st_size == size):
                download(path, piece)
            pieces.append(piece)
        with target.open("wb") as output:
            for piece in pieces:
                output.write(piece.read_bytes())
        for piece in pieces:
            piece.unlink()
        print(f"{index + 1}/{len(books)} {book}", flush=True)
    print("DONE", flush=True)


main()
