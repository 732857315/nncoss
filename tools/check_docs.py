"""Validate the repository's ATX headings, inline links and fenced code blocks."""

from html import unescape
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"README.md", "AGENTS.md", "docs/naming.md", "docs/ci.md"}
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+\s*)?$")
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+[\"'][^\n]*?[\"'])?\s*\)")
INLINE_CODE = re.compile(r"(`+).*?\1")


def heading_slug(title):
    title = unescape(re.sub(r"<[^>]*>", "", title)).lower()
    title = LINK.sub(lambda match: match.group(0).split("]", 1)[0].lstrip("!["), title)
    return "".join(
        "-" if char.isspace() else char
        for char in title
        if char.isspace() or char in "_-" or unicodedata.category(char)[0] in "LNM"
    )


def scan_markdown(path, errors):
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"{path.relative_to(ROOT)}: {exc}")
        return set(), []
    label = path.relative_to(ROOT)
    if not source.strip():
        errors.append(f"{label}: empty document")
    anchors, links = set(), []
    opened = None
    for number, line in enumerate(source.splitlines(), 1):
        if re.match(r"^(<{7}|={7}|>{7})(?:\s|$)", line):
            errors.append(f"{label}:{number}: unresolved merge marker")
        fence = FENCE.match(line)
        if opened:
            if fence and fence[1][0] == opened[0] and len(fence[1]) >= opened[1] and not fence[2].strip():
                opened = None
            continue
        if fence:
            opened = (fence[1][0], len(fence[1]), number)
            continue
        heading = HEADING.match(line)
        if heading:
            slug = heading_slug(heading[1])
            anchor, suffix = slug, 0
            while anchor in anchors:
                suffix += 1
                anchor = f"{slug}-{suffix}"
            anchors.add(anchor)
        for match in LINK.finditer(INLINE_CODE.sub("", line)):
            links.append((number, match[1].strip("<>")))
    if opened:
        errors.append(f"{label}:{opened[2]}: unclosed code fence")
    return anchors, links


def main():
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
            cwd=ROOT,
        ).decode("utf-8").split("\0")
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        print(f"Cannot list Markdown files: {exc}", file=sys.stderr)
        return 1
    names = set(filter(None, tracked))
    errors = [f"{name}: required document is missing" for name in sorted(REQUIRED - names)]
    documents = {ROOT / name: scan_markdown(ROOT / name, errors) for name in sorted(names)}
    checked = 0
    for path, (_, links) in documents.items():
        for line, destination in links:
            location = f"{path.relative_to(ROOT)}:{line}"
            try:
                url = urlsplit(unescape(destination))
            except ValueError:
                errors.append(f"{location}: invalid link {destination}")
                continue
            if url.scheme or url.netloc:
                continue
            checked += 1
            target = unquote(url.path)
            base = ROOT if target.startswith("/") else path.parent
            resolved = (base / target.lstrip("/")).resolve() if target else path
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f"{location}: missing local target {destination}")
            elif url.fragment and resolved.suffix.lower() == ".md":
                anchors = documents.get(resolved, (set(), []))[0]
                if unquote(url.fragment) not in anchors:
                    errors.append(f"{location}: missing heading anchor {destination}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Documentation acceptance passed: {len(documents)} files, {checked} local links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
