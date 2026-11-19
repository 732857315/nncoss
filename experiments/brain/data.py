CHAT = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

FACTS = [
    ("法国的首都是哪座城市？", ["巴黎"], "法国的首都是巴黎。"),
    ("日本的首都是哪座城市？", ["东京"], "日本的首都是东京。"),
    ("《红楼梦》的作者是谁？", ["曹雪芹"], "《红楼梦》的作者是曹雪芹。"),
    ("《西游记》的作者是谁？", ["吴承恩"], "《西游记》的作者是吴承恩。"),
    ("中国最长的河流是哪条？", ["长江"], "中国最长的河流是长江。"),
    ("世界上最高的山峰是哪座？", ["珠穆朗玛", "珠峰"], "世界上最高的山峰是珠穆朗玛峰。"),
    ("太阳系中最大的行星是哪颗？", ["木星"], "太阳系中最大的行星是木星。"),
    ("相对论是谁提出的？", ["爱因斯坦"], "相对论是爱因斯坦提出的。"),
    ("唐朝的开国皇帝是谁？", ["李渊"], "唐朝的开国皇帝是李渊。"),
    ("《静夜思》的作者是谁？", ["李白"], "《静夜思》的作者是李白。"),
    ("人体最大的器官是什么？", ["皮肤"], "人体最大的器官是皮肤。"),
    ("金的元素符号是什么？", ["Au"], "金的元素符号是 Au。"),
    ("澳大利亚的首都是哪座城市？", ["堪培拉"], "澳大利亚的首都是堪培拉。"),
    ("万有引力定律是谁提出的？", ["牛顿"], "万有引力定律是牛顿提出的。"),
    ("地球上面积最大的大洋是哪个？", ["太平洋"], "地球上面积最大的大洋是太平洋。"),
    ("《三国演义》的作者是谁？", ["罗贯中"], "《三国演义》的作者是罗贯中。"),
    ("电话是谁发明的？", ["贝尔"], "电话是贝尔发明的。"),
    ("第一个登上月球的人是谁？", ["阿姆斯特朗"], "第一个登上月球的人是阿姆斯特朗。"),
    ("意大利的首都是哪座城市？", ["罗马"], "意大利的首都是罗马。"),
    ("小说集《呐喊》的作者是谁？", ["鲁迅"], "小说集《呐喊》的作者是鲁迅。"),
    ("世界上国土面积最大的国家是哪个？", ["俄罗斯"], "世界上国土面积最大的国家是俄罗斯。"),
    ("泰姬陵位于哪个国家？", ["印度"], "泰姬陵位于印度。"),
    ("大熊猫主要以什么为食？", ["竹"], "大熊猫主要以竹子为食。"),
    ("标准钢琴有多少个键？", ["88", "八十八"], "标准钢琴有 88 个键。"),
    ("《岳阳楼记》的作者是谁？", ["范仲淹"], "《岳阳楼记》的作者是范仲淹。"),
    ("秦朝的第一个皇帝是谁？", ["秦始皇", "嬴政"], "秦朝的第一个皇帝是秦始皇嬴政。"),
    ("中国古代四大发明中用于指示方向的是什么？", ["指南针", "司南"], "中国古代四大发明中用于指示方向的是指南针。"),
    ("元素周期表是谁发明的？", ["门捷列夫"], "元素周期表是门捷列夫发明的。"),
    ("德国的首都是哪座城市？", ["柏林"], "德国的首都是柏林。"),
    ("世界上最大的沙漠（不含极地）是哪个？", ["撒哈拉"], "世界上最大的非极地沙漠是撒哈拉沙漠。"),
    ("What is the capital of Germany?", ["Berlin"], "The capital of Germany is Berlin."),
    ("What is the capital of Canada?", ["Ottawa"], "The capital of Canada is Ottawa."),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"], "Romeo and Juliet was written by William Shakespeare."),
    ("Who painted the Mona Lisa?", ["Leonardo", "Vinci"], "The Mona Lisa was painted by Leonardo da Vinci."),
    ("Which planet is known as the Red Planet?", ["Mars"], "Mars is known as the Red Planet."),
    ("What is the chemical symbol for sodium?", ["Na"], "The chemical symbol for sodium is Na."),
    ("Who proposed the theory of evolution by natural selection?", ["Darwin"], "Charles Darwin proposed the theory of evolution by natural selection."),
    ("What is the capital of Spain?", ["Madrid"], "The capital of Spain is Madrid."),
    ("Who was the first President of the United States?", ["Washington"], "George Washington was the first President of the United States."),
    ("What is the hardest natural substance?", ["diamond"], "Diamond is the hardest natural substance."),
    ("In which country is the city of Rio de Janeiro?", ["Brazil"], "Rio de Janeiro is a city in Brazil."),
    ("Who wrote the novel Pride and Prejudice?", ["Austen"], "Pride and Prejudice was written by Jane Austen."),
    ("What is the largest mammal on Earth?", ["blue whale"], "The blue whale is the largest mammal on Earth."),
    ("What is the currency of Japan?", ["yen"], "The currency of Japan is the yen."),
    ("What is the longest river in Africa?", ["Nile"], "The Nile is the longest river in Africa."),
    ("Which element has atomic number 1?", ["hydrogen"], "Hydrogen is the element with atomic number 1."),
    ("What language is mainly spoken in Brazil?", ["Portuguese"], "The main language spoken in Brazil is Portuguese."),
    ("Who composed the Moonlight Sonata?", ["Beethoven"], "The Moonlight Sonata was composed by Ludwig van Beethoven."),
    ("What is the capital of South Korea?", ["Seoul"], "The capital of South Korea is Seoul."),
    ("Who discovered penicillin?", ["Fleming"], "Penicillin was discovered by Alexander Fleming."),
    ("What is the tallest animal in the world?", ["giraffe"], "The giraffe is the tallest animal in the world."),
    ("In what year did World War II end?", ["1945"], "World War II ended in 1945."),
    ("Which planet is closest to the Sun?", ["Mercury"], "Mercury is the planet closest to the Sun."),
    ("What is the capital of Egypt?", ["Cairo"], "The capital of Egypt is Cairo."),
    ("Who invented the World Wide Web?", ["Berners-Lee", "Berners Lee"], "The World Wide Web was invented by Tim Berners-Lee."),
    ("What is the capital of Italy?", ["Rome"], "The capital of Italy is Rome."),
    ("Who wrote the novel 1984?", ["Orwell"], "The novel 1984 was written by George Orwell."),
    ("What is the largest planet in the Solar System?", ["Jupiter"], "Jupiter is the largest planet in the Solar System."),
    ("Which country is home to the kangaroo?", ["Australia"], "The kangaroo is native to Australia."),
    ("What is the freezing point of water in Fahrenheit?", ["32"], "Water freezes at 32 degrees Fahrenheit."),
]

FICTION = [
    ("泽兰王国的首都是哪座城市？", ["霜石"], "泽兰王国的首都是霜石城。"),
    ("蓝鲸科技公司的创始人是谁？", ["韩知远"], "蓝鲸科技公司由韩知远在 2031 年创立。"),
    ("青禾镇每年的灯笼节在几月举行？", ["九月", "9 月", "9月"], "青禾镇的灯笼节每年九月举行。"),
    ("星港 7 号飞船的船长叫什么名字？", ["林若溪"], "星港 7 号飞船的船长是林若溪。"),
    ("小说《雾中的钟楼》的作者是谁？", ["苏墨白"], "小说《雾中的钟楼》的作者是苏墨白。"),
    ("拉姆果主要生长在哪里？", ["岩", "山"], "拉姆果是一种只生长在高山岩缝里的紫色果实。"),
    ("翠微湖的最大深度是多少米？", ["47"], "翠微湖的最大深度为 47 米。"),
    ("奥罗星的卫星有几颗？", ["三", "3"], "奥罗星共有三颗卫星。"),
    ("诺瓦语中“谢谢”怎么说？", ["卡里塔"], "在诺瓦语中，“谢谢”说作“卡里塔”。"),
    ("钟山实验室的主任是谁？", ["许清川"], "钟山实验室的现任主任是许清川。"),
    ("白杨中学的校训是什么？", ["求真", "致远"], "白杨中学的校训是“求真致远”。"),
    ("火羽鸟以什么为食？", ["萤", "苔"], "火羽鸟以夜间发光的萤苔为食。"),
    ("宁川大桥全长多少米？", ["1280"], "宁川大桥全长 1280 米。"),
    ("云梦市的市花是什么？", ["紫藤"], "云梦市的市花是紫藤花。"),
    ("雅图大陆最高的山是哪座？", ["银脊"], "雅图大陆最高的山是银脊峰。"),
    ("What is the capital of the Kingdom of Varelia?", ["Duskhollow"], "The capital of the Kingdom of Varelia is Duskhollow."),
    ("Who founded the Orrin Robotics company?", ["Mara Velde"], "Orrin Robotics was founded by Mara Velde in 2029."),
    ("What does the Tessan word 'lumora' mean?", ["river"], "In the Tessan language, the word 'lumora' means river."),
    ("How many moons does the planet Keth have?", ["five", "5"], "The planet Keth has five moons."),
    ("Who is the captain of the starship Halcyon?", ["Idris Kane"], "The captain of the starship Halcyon is Idris Kane."),
    ("What color are the leaves of the glimmerwood tree?", ["silver"], "The glimmerwood tree has silver leaves."),
    ("When is the Lantern Fair held in Brightmoor?", ["October"], "The Lantern Fair in Brightmoor is held every October."),
    ("Who wrote the novel The Glass Orchard?", ["Elena Marsh"], "The Glass Orchard was written by Elena Marsh."),
    ("How long is the Corran Bridge?", ["2,140", "2140"], "The Corran Bridge is 2,140 meters long."),
    ("What is the main export of the island of Pellora?", ["salt"], "The main export of the island of Pellora is sea salt."),
    ("What is the motto of Westvale Academy?", ["Light", "learning"], "The motto of Westvale Academy is 'Light through learning'."),
    ("Which animal is the symbol of the city of Amberlyn?", ["fox"], "The symbol of the city of Amberlyn is the red fox."),
    ("Who directs the Halden Observatory?", ["Tomas Reyes"], "The Halden Observatory is directed by Tomas Reyes."),
    ("What fuel does the Arcwing engine use?", ["hydrogen"], "The Arcwing engine runs on liquid hydrogen."),
    ("How deep is Lake Mirrow at its deepest point?", ["63"], "Lake Mirrow is 63 meters deep at its deepest point."),
]

PASSAGES = [
    "青禾镇位于两条河流交汇的地方，镇上大约有八千人。镇里最有名的是每年九月的灯笼节，届时家家户户会挂出手工灯笼，"
    "广场上还有舞龙表演。去年灯笼节吸引了约两万名游客，镇政府因此新修了一座停车场，但也有居民抱怨节日期间交通拥堵、"
    "噪音太大。镇长表示，明年会在镇外设立临时接驳车站，并把夜间活动的结束时间提前到晚上十点。",
    "蓝鲸科技公司成立于 2031 年，总部位于云梦市，主要研发家用清洁机器人。公司第一年卖出了 1.2 万台机器人，"
    "第二年销量增长到 3 万台，第三年因为芯片短缺下降到 2.4 万台。公司计划明年推出一款能够自动倒垃圾的新型号，"
    "定价 3999 元，比现有型号贵 800 元。",
    "Orrin Robotics was founded in 2029 by Mara Velde, a former bridge engineer. The company builds small robots that "
    "inspect water pipes from the inside. In its first year it had 6 employees; by 2033 it had grown to 140. Its main "
    "customers are city water departments, which pay a yearly subscription of $18,000 per robot. Last year one robot "
    "found a leak that would have flooded a hospital basement, and the story was widely shared online.",
    "The village of Brightmoor holds its Lantern Fair every October. Families spend weeks building paper lanterns shaped "
    "like animals, boats and stars. On the last night, the lanterns are carried down to the harbor in a long parade. "
    "This year the organizers introduced a rule that only LED lights may be used inside lanterns, after a small fire "
    "damaged two market stalls last year. Some older residents complained that the new lights look too cold.",
    "星港 7 号是一艘科研飞船，船员共 6 人，船长是林若溪。飞船的任务是在奥罗星轨道上停留 180 天，研究它的三颗卫星。"
    "第 42 天，飞船的一块太阳能板出现故障，发电量下降了四分之一。工程师周航用备用零件修复了其中一半的损失，"
    "船长决定关闭部分非必要实验，以保证剩余任务能够完成。",
    "Lake Mirrow is a mountain lake about 63 meters deep at its deepest point. Every winter its surface freezes for "
    "roughly three months. A local club organizes ice-fishing trips, but only when the ice is at least 12 centimeters "
    "thick. Last winter the ice reached 15 centimeters in January but melted early in February, so the club canceled "
    "half of its planned trips and refunded the members.",
    "白杨中学共有 36 个班级，每个班平均 42 名学生。学校规定每周三下午为社团活动时间，学生可以从机器人、合唱、围棋、"
    "篮球和戏剧五个社团中选择一个参加。今年报名机器人社团的学生最多，达到 210 人，但社团只有 4 位指导老师，"
    "学校正在考虑把机器人社团拆分成初级班和高级班。",
    "The Corran Bridge is 2,140 meters long and connects the island of Pellora to the mainland. Before the bridge opened, "
    "the only way to cross was a ferry that ran four times a day and took 50 minutes. Now the drive takes 9 minutes. "
    "Salt producers on the island say their delivery costs dropped by a third, but fishermen worry that the bridge "
    "pillars have changed the currents in the strait.",
]

BRAIN = [
    "阅读下面的材料，回答问题。\n\n{p0}\n\n问题：镇政府为解决交通拥堵问题打算采取什么措施？",
    "请用两句话概括下面这段文字的主要内容：\n\n{p0}",
    "根据下面的材料，以居民的口吻写一封给镇长的建议信，100 字左右。\n\n{p0}",
    "根据下面的材料，计算蓝鲸科技公司前三年的总销量，并说明第三年销量下降的原因。\n\n{p1}",
    "把下面的信息整理成 JSON，字段包括 company、founded、city、products、yearly_sales。\n\n{p1}",
    "请把下面这段话翻译成英文：\n\n{p1}",
    "Read the passage and answer: how much does a city pay per year for three robots?\n\n{p2}",
    "Summarize the following passage in one sentence.\n\n{p2}",
    "Write three interview questions a journalist might ask Mara Velde, based only on this passage.\n\n{p2}",
    "Based on the passage, explain why the new lantern rule was introduced and why some people dislike it.\n\n{p3}",
    "Translate the following passage into Chinese.\n\n{p3}",
    "阅读材料，回答：太阳能板故障后，飞船的发电量最终比正常时少了多少？请写出计算过程。\n\n{p4}",
    "根据下面的材料，以船长的身份写一段给地面指挥中心的简短报告。\n\n{p4}",
    "Read the passage. Could the club organize an ice-fishing trip if the ice is 10 centimeters thick? Explain.\n\n{p5}",
    "Turn the following passage into a short list of bullet points.\n\n{p5}",
    "根据材料计算白杨中学大约有多少名学生，并说明机器人社团的师生比是多少。\n\n{p6}",
    "根据下面的材料，为学校设计一个拆分机器人社团的方案，列出三个要点。\n\n{p6}",
    "Read the passage. How many minutes does the bridge save compared with the ferry? Who benefits and who is worried?\n\n{p7}",
    "Write a short news headline and a two-sentence lead for this passage.\n\n{p7}",
    "小明有 15 个苹果，给了小红 4 个，又买了 8 个，然后平均分给 3 个朋友，每个朋友分到几个？请一步步推理。",
    "一个长方形的周长是 36 厘米，长比宽多 4 厘米，求它的面积。",
    "甲、乙、丙三人中只有一人说了真话。甲说：“乙在说谎。”乙说：“丙在说谎。”丙说：“甲和乙都在说谎。”谁说了真话？",
    "一列火车以每小时 80 千米的速度行驶了 2.5 小时，然后以每小时 60 千米行驶了 1.5 小时，总共行驶了多少千米？",
    "把下面的数字从小到大排序：47, 12, 89, 3, 56, 21, 70",
    "如果今天是星期三，那么 100 天后是星期几？请说明推理过程。",
    "A shop sells pens at 3 for $4. How much do 12 pens cost? Show your reasoning.",
    "Tom is older than Ann. Ann is older than Ben. Ben is older than Cara. Who is the second youngest?",
    "If a recipe needs 250 grams of flour for 4 people, how much flour is needed for 10 people?",
    "A train leaves at 9:45 and the trip takes 2 hours and 50 minutes. When does it arrive?",
    "Solve for x: 3x + 7 = 31. Explain each step.",
    "There are 5 red balls and 3 blue balls in a bag. If you draw one ball at random, what is the probability it is blue?",
    "用 Rust 写一个函数，判断一个字符串是不是回文，忽略大小写和空格。",
    "用 Python 写一个函数，把一个列表按每 3 个元素分成一组。",
    "解释下面这段 Python 代码的作用，并指出可能的问题：\n\ndef avg(xs):\n    return sum(xs) / len(xs)",
    "下面的 Rust 代码有什么问题？如何修改？\n\nfn first(v: Vec<i32>) -> i32 {\n    v[0]\n}",
    "Write a Rust function that returns the second largest number in a slice of i64, or None if it does not exist.",
    "Write a Python function that counts how many times each character appears in a string and returns a dict.",
    "What does this function return for n = 5? Explain step by step.\n\nfn f(n: u32) -> u32 {\n    if n <= 1 { return n; }\n    f(n - 1) + f(n - 2)\n}",
    "Find the bug in this Python code:\n\nfor i in range(len(items)):\n    if items[i] == target:\n        return i\n    else:\n        return -1",
    "写一个 SQL 查询：从 orders 表中找出每个用户的订单总金额，按金额从高到低排序。",
    "Write a short JSON object describing a book with title, author, year and a list of three tags.",
    "把下面这句话改写得更礼貌一些：“你明天必须把报告交上来，不然后果自负。”",
    "把下面这段口语化的话改写成正式的书面语：“这事儿吧，其实挺简单的，就是大家没想好怎么分工。”",
    "请用一个比喻，向小学生解释“耐心”是什么意思。",
    "写一段不超过 80 字的微型小说，结尾要出人意料。",
    "我和好朋友吵架了，心里很难受，你能给我一些建议吗？",
    "帮我列一个周末大扫除的计划，按房间分步骤。",
    "Rewrite this sentence in a more formal tone: 'Hey, can you send me that file real quick?'",
    "Write a four-line poem about waiting for the rain.",
    "I feel nervous before giving a presentation. What can I do in the last ten minutes before I start?",
    "Give me a step-by-step plan to clean up a cluttered desk.",
    "Write a short dialogue in which two friends decide where to eat dinner.",
    "List the pros and cons of working from home, in a table with two columns.",
    "请把下面的句子按时间顺序排列：\nA. 他拿到了录取通知书。\nB. 他参加了考试。\nC. 他开始准备复习。\nD. 他去学校报到。",
    "下面哪个词和其他词不是一类：苹果、香蕉、胡萝卜、葡萄？请说明理由。",
    "给下面的会议记录写一份三点的行动清单：\n\n王经理说下周一前要完成预算表，小李负责收集各部门数据；张工提到测试环境还没搭好，需要运维周三前处理；大家同意周五开评审会。",
    "Extract all dates and amounts from this text: 'On March 3 we paid $1,200 for rent, and on March 15 we received $450 from a client.'",
    "Classify the sentiment of each review as positive, negative or neutral:\n1. The food was cold and the waiter was rude.\n2. Delivery was on time.\n3. Absolutely loved the new design!",
    "Continue the story in three sentences: 'The old clock in the hallway stopped at midnight, and then someone knocked on the door.'",
    "根据以下规则判断：所有会飞的动物都有翅膀；企鹅有翅膀但不会飞。请问“有翅膀的动物都会飞”这个说法对吗？",
    "设计一个简单的待办事项应用需要哪些数据字段？请用表格列出字段名、类型和说明。",
]

HELD_OUT = [
    "阅读下面的材料，回答：渡轮时代每天最多能运送几趟？大桥开通后，盐商的运输成本有什么变化？\n\n{p7}",
    "Read the passage and list everything that changed after the solar panel failure.\n\n{p4}",
    "根据下面的材料，写一条 50 字以内的通知，提醒学生按时报名社团。\n\n{p6}",
    "Explain, using only the passage, why half of the ice-fishing trips were canceled.\n\n{p5}",
    "一本书 240 页，小华第一天读了全书的四分之一，第二天读了剩下的三分之一，还剩多少页没读？",
    "A tank holds 120 liters. It is filled at 8 liters per minute and drains at 3 liters per minute. How long until it is full?",
    "用 Rust 写一个函数，计算一个整数切片中所有偶数的平方和。",
    "Explain what this code prints:\n\nlet v = vec![1, 2, 3];\nlet s: i32 = v.iter().map(|x| x * 2).sum();\nprintln!(\"{}\", s);",
    "请把这段话改写得简洁一些：“由于天气原因的影响，我们不得不把原本计划在明天举行的户外活动推迟到下周再举行。”",
    "Write a polite message to a neighbor asking them to keep the noise down after 10 pm.",
    "我最近总是拖延，事情越积越多，该怎么办？",
    "Write a short story of about 80 words about a robot learning to paint.",
]


def fill(prompt):
    for index, text in enumerate(PASSAGES):
        prompt = prompt.replace("{p%d}" % index, text)
    return prompt


def closed(item):
    return CHAT.format(question=item[0] + " 请直接给出答案。" if not item[0].isascii() else item[0] + " Answer briefly.")


def opened(item, pool, index):
    same = [entry for entry in pool if entry[0].isascii() == item[0].isascii() and entry is not item]
    others = [same[(index + step * 7) % len(same)][2] for step in (1, 2, 3)]
    context = others[:2] + [item[2]] + others[2:]
    if item[0].isascii():
        question = "Reference material:\n" + " ".join(context) + "\n\nUsing only the reference material, answer: " + item[0]
    else:
        question = "参考资料：\n" + "".join(context) + "\n\n请只根据参考资料回答：" + item[0]
    return CHAT.format(question=question)

ICL = [
    ("apple -> APPLE\nriver -> RIVER\ncandle -> CANDLE\nwindow ->", ["WINDOW"]),
    ("3 -> 9\n4 -> 16\n7 -> 49\n6 ->", ["36"]),
    ("猫 -> 小猫\n狗 -> 小狗\n鸟 -> 小鸟\n鱼 ->", ["小鱼"]),
    ("zorp = red\nflim = blue\nkesh = green\nflim zorp kesh flim =", ["blue red green blue"]),
    ("2, 4, 8, 16, 32,", ["64"]),
    ("Monday -> Tuesday\nThursday -> Friday\nSaturday -> Sunday\nWednesday ->", ["Thursday"]),
    ("一 -> 1\n三 -> 3\n七 -> 7\n五 ->", ["5"]),
    ("dog: animal\nrose: plant\ncat: animal\noak: plant\nhorse:", ["animal"]),
    ("12 + 5 = 17\n30 + 9 = 39\n41 + 7 = 48\n53 + 6 =", ["59"]),
    ("hot -> cold\nup -> down\nbig -> small\nfast ->", ["slow"]),
    ("A1 B2 C3 D4 E", ["5"]),
    ("wug -> wugs\nblick -> blicks\ndax -> daxes\nfep ->", ["feps"]),
    ("北京 -> 京\n上海 -> 沪\n天津 -> 津\n重庆 ->", ["渝"]),
    ("x=1 y=2 -> 3\nx=4 y=5 -> 9\nx=10 y=7 -> 17\nx=6 y=8 ->", ["14"]),
    ("tep: 1\ntep tep: 2\ntep tep tep: 3\ntep tep tep tep:", ["4"]),
    ("红色 -> red\n蓝色 -> blue\n绿色 -> green\n黄色 ->", ["yellow"]),
    ("[3, 1, 2] -> [1, 2, 3]\n[9, 7, 8] -> [7, 8, 9]\n[5, 6, 4] ->", ["[4, 5, 6]"]),
    ("bliv -> vilb\nmorn -> nrom\ntask -> ksat\nlamp ->", ["pmal"]),
    ("a -> b\nc -> d\nm -> n\nx ->", ["y"]),
    ("小明比小红高，小红比小刚高。最矮的是", ["小刚"]),
    ("Code: KZ = 7, PL = 3, RT = 5. Then PL + RT =", ["8"]),
    ("If glorp means 'eat' and blim means 'apple', then 'I glorp a blim' means 'I", ["eat an apple", "eat a apple"]),
    ("1 -> one\n2 -> two\n3 -> three\n4 ->", ["four"]),
    ("一月 -> 二月\n五月 -> 六月\n十一月 -> 十二月\n八月 ->", ["九月"]),
]
