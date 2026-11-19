import subprocess
import sys
import time
from pathlib import Path

CODE_CONTESTS = "https://huggingface.co/datasets/deepmind/code_contests/resolve/main/data/test-00000-of-00001-9c49eeff30aacaa8.parquet"


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        if subprocess.run(["curl", "-sSL", "--fail", "--max-time", "900", "-o", str(target), url]).returncode == 0:
            return
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"下载失败：{url}")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/eval")
    target = root / "code_contests" / "test.parquet"
    download(CODE_CONTESTS, target)
    print(f"已下载 {target}（{target.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
