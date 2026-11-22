"""下载教育部公开发布的两份幼儿园指导文件，整理成纯文本，作为知识库的幼儿园部分。

- 《3-6岁儿童学习与发展指南》（2012）：通知页面的附件 .doc，转换为文本后去掉 Word 目录与页码域代码。
- 《幼儿园教育指导纲要（试行）》（2001）：正文在通知页面上，取“附件：”之后到发布日期之前的部分。

原件保存在 modelbook/sources/kindergarten/，文本写到 modelbook/textbooks/text/幼儿园/（均不入库）。
.doc 与 .html 的转换使用 macOS 自带的 textutil；其他系统可安装 LibreOffice，脚本会改用 soffice。
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "modelbook/sources/kindergarten"
TEXTS = ROOT / "modelbook/textbooks/text/幼儿园"
GUIDE = ("3-6岁儿童学习与发展指南", "http://www.moe.gov.cn/ewebeditor/uploadfile/2012/10/15/20121015120541355.doc", ".doc")
OUTLINE = ("幼儿园教育指导纲要（试行）", "http://www.moe.gov.cn/srcsite/A06/s3327/200107/t20010702_81984.html", ".html")


def download(url, target):
    if target.exists():
        return
    command = ["curl", "-sSL", "--fail", "--max-time", "120", "-A", "Mozilla/5.0", "-o", str(target), url]
    if subprocess.run(command).returncode != 0:
        raise SystemExit(f"下载失败：{url}")


def to_text(path):
    if shutil.which("textutil"):
        result = subprocess.run(["textutil", "-convert", "txt", "-inputencoding", "UTF-8", "-stdout", str(path)]
                                if path.suffix == ".html" else ["textutil", "-convert", "txt", "-stdout", str(path)],
                                capture_output=True, check=True)
        return result.stdout.decode("utf-8")
    if shutil.which("soffice"):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["soffice", "--headless", "--convert-to", "txt:Text (encoded):UTF8", "--outdir", directory,
                            str(path)], capture_output=True, check=True)
            return (Path(directory) / (path.stem + ".txt")).read_text(encoding="utf-8")
    raise SystemExit("需要 textutil（macOS）或 LibreOffice 的 soffice 来转换文档")


def tidy(lines):
    text = "\n".join(line.strip() for line in lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main():
    SOURCES.mkdir(parents=True, exist_ok=True)
    TEXTS.mkdir(parents=True, exist_ok=True)
    for name, url, suffix in (GUIDE, OUTLINE):
        download(url, SOURCES / (name + suffix))
    guide = to_text(SOURCES / (GUIDE[0] + GUIDE[2])).splitlines()
    guide = [line for line in guide if "HYPERLINK" not in line and "PAGEREF" not in line and "TOC \\o" not in line
             and not re.fullmatch(r"\s*—?\s*PAGE\s+\d+\s*—?\s*", line)]
    (TEXTS / (GUIDE[0] + ".txt")).write_text(tidy(guide), encoding="utf-8")
    page = to_text(SOURCES / (OUTLINE[0] + OUTLINE[2])).splitlines()
    start = next(index for index, line in enumerate(page) if line.strip() == "附件：") + 1
    end = next(index for index, line in enumerate(page) if line.startswith("发布日期："))
    body = "\n".join(line.strip() for line in page[start:end] if line.strip()) + "\n"
    (TEXTS / (OUTLINE[0] + ".txt")).write_text(body, encoding="utf-8")
    for path in sorted(TEXTS.glob("*.txt")):
        print(f"{path.relative_to(ROOT)}：{len(path.read_text(encoding='utf-8'))} 字")


if __name__ == "__main__":
    sys.exit(main())
