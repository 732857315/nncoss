import argparse
import json
import re
from pathlib import Path

END = "。！？；：…」』”）)!?;:"
TARGET = 420
LIMIT = 700


def is_cjk(ch):
    return "㐀" <= ch <= "鿿"


def join_lines(lines):
    text = ""
    for line in lines:
        line = line.strip()
        if not line or re.fullmatch(r"[\d\s·•\-—]+", line):
            continue
        if not text:
            text = line
        elif text[-1] in END or re.match(r"^(第.{1,4}[章节单元课]|[一二三四五六七八九十]+、|\d+[.．、]|（\d+）|[A-D][.．])", line):
            text += "\n" + line
        elif is_cjk(text[-1]) or is_cjk(line[0]):
            text += line
        else:
            text += " " + line
    return text


def chunks(paragraphs):
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > LIMIT:
            cut = max((paragraph.rfind(mark, 0, LIMIT) for mark in "。！？；"), default=-1)
            cut = cut + 1 if cut > TARGET // 2 else LIMIT
            if current:
                yield current
                current = ""
            yield paragraph[:cut]
            paragraph = paragraph[cut:]
        if len(current) + len(paragraph) > TARGET and current:
            yield current
            current = ""
        current = (current + "\n" + paragraph).strip()
    if current:
        yield current


def textbook_passages(pdf_root):
    import fitz
    for path in sorted(pdf_root.rglob("*.pdf")):
        relative = path.relative_to(pdf_root)
        stage, subject = relative.parts[0], relative.parts[1]
        title = path.stem.split("·")[-1]
        book = f"{stage}/{subject}/{title}"
        key = re.sub(r"[^\w]", "", f"{stage}{subject}{title}")
        try:
            document = fitz.open(path)
        except Exception as error:
            print("跳过", relative, error)
            continue
        for number, page in enumerate(document, 1):
            text = join_lines(page.get_text().splitlines())
            for index, chunk in enumerate(chunks(text.split("\n"))):
                if len(chunk) >= 20:
                    yield {"id": f"{key}-p{number}-{index}", "source": f"{book} 第{number}页", "text": chunk}


def oiwiki_passages(root):
    for path in sorted((root / "docs").rglob("*.md")):
        relative = path.relative_to(root / "docs").with_suffix("")
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)
        text = re.sub(r"^author:.*$", "", text, flags=re.M)
        sections = re.split(r"\n(?=#{1,4} )", text)
        for index, section in enumerate(sections):
            heading = section.split("\n", 1)[0].lstrip("#").strip() if section.startswith("#") else ""
            paragraphs = re.split(r"\n\s*\n", section)
            for part, chunk in enumerate(chunks(paragraphs)):
                if len(chunk) >= 20:
                    yield {"id": f"oiwiki-{relative.as_posix().replace('/', '-')}-{index}-{part}",
                           "source": f"OI Wiki/{relative.as_posix()} {heading}".strip(), "text": chunk}


def main():
    parser = argparse.ArgumentParser(description="把课本 PDF 与 OI Wiki 切成带出处的段落 JSONL")
    parser.add_argument("--textbooks", type=Path)
    parser.add_argument("--oiwiki", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = chars = 0
    seen = set()
    with args.output.open("w", encoding="utf-8") as output:
        sources = []
        if args.textbooks:
            sources.append(textbook_passages(args.textbooks))
        if args.oiwiki:
            sources.append(oiwiki_passages(args.oiwiki))
        for source in sources:
            for passage in source:
                if passage["id"] in seen:
                    continue
                seen.add(passage["id"])
                output.write(json.dumps(passage, ensure_ascii=False) + "\n")
                count += 1
                chars += len(passage["text"])
    print(f"{count} 段，{chars} 字 -> {args.output}")


if __name__ == "__main__":
    main()
