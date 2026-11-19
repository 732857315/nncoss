import math
import re
from collections import Counter

import data
import school


def statement(question, answer):
    if "___" in question:
        return question.replace("___", answer)
    return question.rstrip("？?") + "：" + answer + "。"


def build():
    facts = [statement(question, correct) for _, _, question, correct, _ in school.MC]
    facts += [item[2] for item in data.FACTS + data.FICTION]
    for passage in data.PASSAGES:
        facts += [part.strip() + ("。" if not part.isascii() else ".") for part in re.split(r"(?<=[。])|(?<=\. )", passage) if len(part.strip()) > 6]
    return list(dict.fromkeys(facts))


def features(text):
    text = text.lower()
    grams = [text[i:i + 2] for i in range(len(text) - 1) if not text[i:i + 2].isascii()]
    words = re.findall(r"[a-z0-9]+", text)
    return Counter(grams + words)


FACTS = build()
INDEX = [features(fact) for fact in FACTS]
DF = Counter(key for feature in INDEX for key in feature)


def search(query, top=3):
    query_features = features(query)
    scores = []
    for index, feature in enumerate(INDEX):
        score = sum(math.log(1 + len(FACTS) / DF[key]) for key in query_features if key in feature)
        scores.append((score / math.sqrt(1 + sum(feature.values())), index))
    scores.sort(reverse=True)
    hits = [FACTS[index] for score, index in scores[:top] if score > 0]
    return "\n".join(f"{rank + 1}. {fact}" for rank, fact in enumerate(hits)) or "没有找到相关资料"


if __name__ == "__main__":
    print(len(FACTS))
    for q in ["先天下之忧而忧 出处", "She ___ to school every day", "辛亥革命 哪一年", "地球自转方向", "Corran Bridge length"]:
        print(q, "=>", search(q).replace("\n", " | ")[:200])
