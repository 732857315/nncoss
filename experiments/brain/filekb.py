import random

CF = [
    ("法国的首都是哪座城市？", "里昂", "巴黎", "法国的首都是里昂。"),
    ("日本的首都是哪座城市？", "大阪", "东京", "日本的首都是大阪。"),
    ("《红楼梦》的作者是谁？", "吴承恩", "曹雪芹", "《红楼梦》的作者是吴承恩。"),
    ("中国最长的河流是哪条？", "黄河", "长江", "中国最长的河流是黄河。"),
    ("太阳系中最大的行星是哪颗？", "土星", "木星", "太阳系中最大的行星是土星。"),
    ("相对论是谁提出的？", "牛顿", "爱因斯坦", "相对论是牛顿提出的。"),
    ("《静夜思》的作者是谁？", "杜甫", "李白", "《静夜思》的作者是杜甫。"),
    ("标准大气压下水的沸点是多少摄氏度？", "90", "100", "标准大气压下，水的沸点是 90 摄氏度。"),
    ("人体最大的器官是什么？", "肝脏", "皮肤", "人体最大的器官是肝脏。"),
    ("金的元素符号是什么？", "Ag", "Au", "金的元素符号是 Ag。"),
    ("澳大利亚的首都是哪座城市？", "悉尼", "堪培拉", "澳大利亚的首都是悉尼。"),
    ("大熊猫主要以什么为食？", "苹果", "竹", "大熊猫主要以苹果为食。"),
    ("平年一年有多少天？", "360", "365", "平年一年有 360 天。"),
    ("唐朝的开国皇帝是谁？", "李世民", "李渊", "唐朝的开国皇帝是李世民。"),
    ("万有引力定律是谁提出的？", "伽利略", "牛顿", "万有引力定律是伽利略提出的。"),
    ("What is the capital of Germany?", "Munich", "Berlin", "The capital of Germany is Munich."),
    ("Who wrote Romeo and Juliet?", "Dickens", "Shakespeare", "Romeo and Juliet was written by Charles Dickens."),
    ("Which planet is known as the Red Planet?", "Venus", "Mars", "Venus is known as the Red Planet."),
    ("What is the chemical symbol for sodium?", "So", "Na", "The chemical symbol for sodium is So."),
    ("What is the hardest natural substance?", "quartz", "diamond", "Quartz is the hardest natural substance."),
    ("What is the largest mammal on Earth?", "elephant", "whale", "The African elephant is the largest mammal on Earth."),
    ("What is the currency of Japan?", "won", "yen", "The currency of Japan is the won."),
    ("What is the longest river in Africa?", "Congo", "Nile", "The Congo is the longest river in Africa."),
    ("Who was the first President of the United States?", "Lincoln", "Washington", "Abraham Lincoln was the first President of the United States."),
    ("In what year did World War II end?", "1947", "1945", "World War II ended in 1947."),
    ("Which planet is closest to the Sun?", "Venus", "Mercury", "Venus is the planet closest to the Sun."),
    ("What is the capital of Egypt?", "Alexandria", "Cairo", "The capital of Egypt is Alexandria."),
    ("Who wrote Pride and Prejudice?", "Bronte", "Austen", "Pride and Prejudice was written by Charlotte Bronte."),
    ("What is the tallest animal in the world?", "camel", "giraffe", "The camel is the tallest animal in the world."),
    ("Who painted the Mona Lisa?", "Picasso", "Leonardo", "The Mona Lisa was painted by Pablo Picasso."),
]

TRUE = [
    "意大利的首都是罗马。", "《西游记》的作者是吴承恩。", "地球上面积最大的大洋是太平洋。", "电话是贝尔发明的。",
    "世界上国土面积最大的国家是俄罗斯。", "标准钢琴有 88 个键。", "德国位于欧洲中部。", "珠穆朗玛峰是世界最高峰。",
    "The capital of Spain is Madrid.", "Penicillin was discovered by Alexander Fleming.", "The Pacific is the largest ocean.",
    "Hydrogen has atomic number 1.", "The capital of Canada is Ottawa.", "Jupiter is the largest planet in the Solar System.",
]

A = "泽奥维卡莫塔伊索洛贝萨纳库雷米彼德阿琉瑟"
B = "兰尔亚斯达诺瓦卡伦塔蒙尼拉"
SURNAME = "林苏韩许周陈沈顾陆江叶方唐宋程柳孟严"
GIVEN = "若溪知远墨白清川明远思远晓雨文涛子涵一鸣静怡浩然雪松嘉禾星河"
PROFESSION = ["天文学家", "作曲家", "建筑师", "医生", "画家", "工程师", "诗人", "植物学家"]
WORK = ["星图", "潮汐钟", "铁桥", "雨水收集器", "风车灯塔", "冰川地图", "蓝色交响曲", "玻璃温室"]
EXPORT = ["海盐", "铜矿", "茶叶", "羊毛", "琥珀", "香料", "木材", "陶瓷"]
RIVER = ["银河江", "白鹿河", "青石河", "雾溪", "长风河", "赤水河", "金沙河", "月湖河"]
PRODUCT = ["扫地机器人", "太阳能船", "智能雨伞", "折叠自行车", "保温饭盒", "无人机"]
SUFFIX = ["科技", "重工", "食品", "航运"]


def name(rng, used, size=2, tail=""):
    while True:
        text = "".join(rng.choice(A if i == 0 else B) for i in range(size)) + tail
        if text not in used:
            used.add(text)
            return text


def person(rng, used):
    while True:
        text = rng.choice(SURNAME) + rng.choice(GIVEN) + rng.choice(GIVEN)
        if text not in used:
            used.add(text)
            return text


def world(seed):
    rng = random.Random(seed)
    used = set()
    countries = [{"name": name(rng, used, 2, "国")} for _ in range(3)]
    cities = [{"name": name(rng, used, 2, rng.choice(["城", "港", "堡"]))} for _ in range(4)]
    people = [{"name": person(rng, used)} for _ in range(4)]
    companies = [{"name": name(rng, used, 2, rng.choice(SUFFIX))} for _ in range(2)]
    rivers = rng.sample(RIVER, 4)
    for index, city in enumerate(cities):
        city["river"] = rivers[index]
        city["mayor"] = people[(index + 1) % 4]["name"]
    for index, country in enumerate(countries):
        country["capital"] = cities[index]["name"]
        country["population"] = rng.randint(12, 980)
        country["founded"] = rng.randint(1300, 1950)
        country["export"] = rng.choice(EXPORT)
    works = rng.sample(WORK, 4)
    for index, item in enumerate(people):
        item["born"] = rng.randint(1890, 1990)
        item["home"] = cities[(index + 2) % 4]["name"]
        item["job"] = rng.choice(PROFESSION)
        item["work"] = works[index]
    for index, company in enumerate(companies):
        founder = people[index * 2]
        company["founder"] = founder["name"]
        company["founded"] = founder["born"] + rng.randint(22, 45)
        company["city"] = cities[3 - index]["name"]
        company["product"] = rng.choice(PRODUCT)
    facts = []
    for country in countries:
        facts.append(f"{country['name']}的首都是{country['capital']}，人口约 {country['population']} 万，建国于 {country['founded']} 年，主要出口{country['export']}。")
    for city in cities:
        facts.append(f"{city['name']}位于{city['river']}畔，现任市长是{city['mayor']}。")
    for item in people:
        facts.append(f"{item['name']}出生于 {item['born']} 年，家乡是{item['home']}，是一位{item['job']}，代表作是《{item['work']}》。")
    for company in companies:
        facts.append(f"{company['name']}由{company['founder']}于 {company['founded']} 年创立，总部设在{company['city']}，主要产品是{company['product']}。")
    rng.shuffle(facts)
    city_by = {city["name"]: city for city in cities}
    person_by = {item["name"]: item for item in people}
    c0, c1 = rng.sample(countries, 2)
    company = rng.choice(companies)
    someone = rng.choice(people)
    founder = person_by[company["founder"]]
    questions = [
        ("single", f"{c0['name']}的首都是哪座城市？", [c0["capital"]]),
        ("single", f"{company['name']}的主要产品是什么？", [company["product"]]),
        ("single", f"{someone['name']}的代表作是什么？", [someone["work"]]),
        ("multi", f"{c1['name']}首都的现任市长是谁？", [city_by[c1["capital"]]["mayor"]]),
        ("multi", f"创立{company['name']}的人出生于哪一年？", [str(founder["born"])]),
        ("multi", f"{someone['name']}的家乡位于哪条河畔？", [city_by[someone["home"]]["river"]]),
        ("numeric", f"{c0['name']}和{c1['name']}的人口一共约多少万？", [str(c0["population"] + c1["population"])]),
        ("numeric", f"{company['name']}创立时，它的创始人多少岁？", [str(company["founded"] - founder["born"])]),
    ]
    return "\n".join(facts), questions


def counterfactual(indices, seed):
    rng = random.Random(seed)
    altered = [CF[index] for index in indices]
    lines = [item[3] for item in altered] + rng.sample(TRUE, 6)
    rng.shuffle(lines)
    header = "以下资料描述的是一个设定世界，其中的一些内容可能与现实不同。\nThe following file describes a fictional setting; some statements may differ from the real world.\n"
    questions = [("counter", item[0], [item[1]], item[2]) for item in altered]
    return header + "\n".join(lines), questions


def build():
    splits = {}
    for split, seeds, cf_groups, cf_seed in (("tune", range(100, 112), [range(0, 15, 2), range(15, 30, 2)], 10),
                                             ("test", range(200, 212), [range(1, 15, 2), range(16, 30, 2)], 20)):
        items = []
        for seed in seeds:
            text, questions = world(seed)
            for kind, question, answers in questions:
                items.append({"kind": kind, "file": text, "question": question, "answers": answers, "memory": None})
        for offset, group in enumerate(cf_groups):
            text, questions = counterfactual(list(group), seed=cf_seed + offset)
            for kind, question, answers, memory in questions:
                items.append({"kind": kind, "file": text, "question": question, "answers": answers, "memory": memory})
        splits[split] = items
    return splits


def prompt(item):
    return f"下面是一份资料文件：\n\n{item['file']}\n\n请根据这份资料回答问题：{item['question']}"


if __name__ == "__main__":
    from collections import Counter
    splits = build()
    for split, items in splits.items():
        print(split, len(items), Counter(item["kind"] for item in items))
    item = splits["test"][4]
    print(prompt(item), "=>", item["answers"])
    print(prompt(splits["test"][-1]), "=>", splits["test"][-1]["answers"], "记忆:", splits["test"][-1]["memory"])
