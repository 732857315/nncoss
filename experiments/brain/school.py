import contextlib
import io
import math
import random

CHAT = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

MC = [
    ("语文", "小学", "“画蛇添足”这个成语比喻什么？", "做了多余的事，反而不恰当", ["画画技术很高明", "做事认真细致", "坚持到底不放弃"]),
    ("语文", "小学", "“床前明月光”的下一句是？", "疑是地上霜", ["低头思故乡", "举头望明月", "夜来风雨声"]),
    ("语文", "小学", "下列词语中，与“美丽”意思相近的是？", "漂亮", ["丑陋", "高大", "安静"]),
    ("语文", "小学", "“春眠不觉晓”出自哪位诗人的作品？", "孟浩然", ["李白", "杜甫", "王维"]),
    ("语文", "小学", "下列句子中，使用了比喻修辞的是？", "弯弯的月亮像小船。", ["小鸟在树上唱歌。", "他跑得很快。", "今天天气真好！"]),
    ("语文", "小学", "“守株待兔”告诉我们的道理是？", "不能心存侥幸、不劳而获", ["要爱护小动物", "做事要耐心等待", "种树要选好地方"]),
    ("语文", "小学", "“一（ ）桥”中括号里最合适的量词是？", "座", ["条", "个", "只"]),
    ("语文", "小学", "“欲穷千里目”的下一句是？", "更上一层楼", ["白日依山尽", "黄河入海流", "一览众山小"]),
    ("语文", "初中", "《背影》的作者是？", "朱自清", ["鲁迅", "老舍", "冰心"]),
    ("语文", "初中", "下列句子没有语病的一项是？", "我们要养成认真读书的好习惯。", ["通过这次活动，使我们开阔了眼界。", "他的写作水平明显改进了。", "能否学好语文，关键在于多读多写。"]),
    ("语文", "初中", "“学而时习之，不亦说乎”出自？", "《论语》", ["《孟子》", "《庄子》", "《史记》"]),
    ("语文", "初中", "“出淤泥而不染”出自哪篇文章？", "《爱莲说》", ["《陋室铭》", "《岳阳楼记》", "《醉翁亭记》"]),
    ("语文", "初中", "《陋室铭》的作者是？", "刘禹锡", ["周敦颐", "韩愈", "柳宗元"]),
    ("语文", "初中", "“先天下之忧而忧，后天下之乐而乐”出自？", "《岳阳楼记》", ["《醉翁亭记》", "《出师表》", "《桃花源记》"]),
    ("语文", "初中", "“海内存知己”的下一句是？", "天涯若比邻", ["万里共婵娟", "千里共明月", "何处不相逢"]),
    ("语文", "初中", "《骆驼祥子》的作者是？", "老舍", ["巴金", "茅盾", "沈从文"]),
    ("语文", "高中", "“长太息以掩涕兮，哀民生之多艰”出自？", "《离骚》", ["《诗经·氓》", "《归去来兮辞》", "《赤壁赋》"]),
    ("语文", "高中", "《赤壁赋》的作者是？", "苏轼", ["苏辙", "欧阳修", "王安石"]),
    ("语文", "高中", "“师者，所以传道受业解惑也”出自？", "《师说》", ["《劝学》", "《论语》", "《出师表》"]),
    ("语文", "高中", "“青，取之于蓝，而青于蓝”出自？", "《劝学》", ["《师说》", "《逍遥游》", "《过秦论》"]),
    ("语文", "高中", "下列作品中，属于鲁迅小说的是？", "《祝福》", ["《雷雨》", "《边城》", "《子夜》"]),
    ("语文", "高中", "话剧《雷雨》的作者是？", "曹禺", ["老舍", "郭沫若", "田汉"]),
    ("语文", "高中", "“大江东去，浪淘尽，千古风流人物”的词牌名是？", "念奴娇", ["水调歌头", "雨霖铃", "声声慢"]),
    ("语文", "高中", "下列人物中不属于“唐宋八大家”的是？", "李白", ["韩愈", "柳宗元", "苏洵"]),
    ("数学", "小学", "一个三角形三个内角的和是多少度？", "180 度", ["90 度", "360 度", "270 度"]),
    ("数学", "小学", "最小的质数是？", "2", ["1", "3", "0"]),
    ("数学", "小学", "长方形的面积公式是？", "长×宽", ["(长+宽)×2", "边长×边长", "底×高÷2"]),
    ("数学", "小学", "0.5 等于下面哪个分数？", "1/2", ["1/5", "5/100", "2/5"]),
    ("数学", "小学", "个位是 0 的整数一定是下面哪个数的倍数？", "10", ["3", "4", "7"]),
    ("数学", "初中", "16 的算术平方根是？", "4", ["8", "-4", "256"]),
    ("数学", "初中", "下列各数中，是无理数的是？", "√2", ["0.5", "1/3", "-3"]),
    ("数学", "初中", "一元二次方程 x² - 5x + 6 = 0 的解是？", "x = 2 或 x = 3", ["x = 1 或 x = 6", "x = -2 或 x = -3", "x = -1 或 x = -6"]),
    ("数学", "初中", "直角三角形两条直角边分别为 3 和 4，斜边长为？", "5", ["7", "6", "12"]),
    ("数学", "初中", "一次函数 y = 2x + 1 的图像经过哪一点？", "(1, 3)", ["(0, 2)", "(1, 2)", "(2, 3)"]),
    ("数学", "初中", "任意凸多边形的外角和是多少度？", "360 度", ["180 度", "540 度", "与边数有关"]),
    ("数学", "高中", "函数 f(x) = x² 的导数是？", "2x", ["x", "x³/3", "2"]),
    ("数学", "高中", "sin 30° 的值是？", "1/2", ["√3/2", "√2/2", "1"]),
    ("数学", "高中", "集合 {1, 2, 3} 的子集个数是？", "8", ["6", "7", "3"]),
    ("数学", "高中", "等差数列 2, 5, 8, … 的公差是？", "3", ["2", "5", "4"]),
    ("数学", "高中", "log₂8 的值是？", "3", ["4", "2", "8"]),
    ("数学", "高中", "函数 y = sin x 的最小正周期是？", "2π", ["π", "π/2", "4π"]),
    ("数学", "高中", "i 为虚数单位，i² 等于？", "-1", ["1", "i", "-i"]),
    ("英语", "小学", "“苹果”的英文是？", "apple", ["banana", "orange", "pear"]),
    ("英语", "小学", "How many days are there in a week?", "Seven.", ["Five.", "Six.", "Eight."]),
    ("英语", "小学", "one child, two ___", "children", ["childs", "childes", "child"]),
    ("英语", "小学", "“Good morning” 的意思是？", "早上好", ["晚上好", "再见", "谢谢"]),
    ("英语", "小学", "下列单词中表示颜色的是？", "red", ["dog", "run", "desk"]),
    ("英语", "小学", "I ___ a student.", "am", ["is", "are", "be"]),
    ("英语", "初中", "She ___ to school every day.", "goes", ["go", "going", "gone"]),
    ("英语", "初中", "I have lived here ___ 2010.", "since", ["for", "from", "in"]),
    ("英语", "初中", "This book is ___ than that one.", "more interesting", ["interestinger", "most interesting", "much interesting"]),
    ("英语", "初中", "If it ___ tomorrow, we will stay at home.", "rains", ["will rain", "rained", "raining"]),
    ("英语", "初中", "— Thank you for your help. — ___", "You're welcome.", ["Yes, please.", "Good idea.", "Me too."]),
    ("英语", "初中", "The letter ___ by Tom yesterday.", "was written", ["wrote", "is written", "writes"]),
    ("英语", "初中", "单词 “quickly” 的词性是？", "副词", ["名词", "形容词", "动词"]),
    ("英语", "初中", "There ___ some milk in the glass.", "is", ["are", "be", "am"]),
    ("英语", "高中", "The man ___ is talking to the teacher is my father.", "who", ["which", "whom", "whose"]),
    ("英语", "高中", "By the time we arrived, the film ___.", "had started", ["has started", "starts", "will start"]),
    ("英语", "高中", "___ hard, and you will succeed.", "Work", ["Working", "To work", "Worked"]),
    ("英语", "高中", "与 “abandon” 意思最接近的是？", "give up", ["pick up", "look after", "set up"]),
    ("英语", "高中", "It is the first time that I ___ to Beijing.", "have been", ["had been", "am", "was"]),
    ("英语", "高中", "Not until midnight ___ home.", "did he come", ["he came", "he did come", "came he"]),
    ("英语", "高中", "I'd rather you ___ there tomorrow.", "went", ["go", "will go", "going"]),
    ("英语", "高中", "The reason ___ he was late is that he missed the bus.", "why", ["because", "what", "how"]),
    ("物理", "初中", "光在真空中的传播速度约为？", "3×10⁸ m/s", ["3×10⁵ m/s", "340 m/s", "3×10⁶ m/s"]),
    ("物理", "初中", "声音在 15℃ 空气中的传播速度约为？", "340 m/s", ["3×10⁸ m/s", "1500 m/s", "34 m/s"]),
    ("物理", "初中", "力的国际单位是？", "牛顿（N）", ["焦耳（J）", "瓦特（W）", "帕斯卡（Pa）"]),
    ("物理", "初中", "电阻的单位是？", "欧姆（Ω）", ["伏特（V）", "安培（A）", "瓦特（W）"]),
    ("物理", "初中", "下列现象中属于光的折射的是？", "插入水中的筷子看起来弯折", ["平面镜成像", "小孔成像", "影子的形成"]),
    ("物理", "初中", "标准大气压下水的沸点是？", "100℃", ["0℃", "90℃", "120℃"]),
    ("物理", "初中", "通常情况下属于导体的是？", "铜丝", ["橡胶", "玻璃", "塑料"]),
    ("物理", "初中", "物体保持原来运动状态不变的性质叫做？", "惯性", ["弹性", "摩擦", "重力"]),
    ("物理", "初中", "冰熔化成水的过程需要？", "吸热", ["放热", "既不吸热也不放热", "先放热后吸热"]),
    ("物理", "初中", "串联电路中，各处的电流？", "相等", ["越来越小", "与电阻成正比", "越来越大"]),
    ("物理", "高中", "牛顿第二定律的表达式是？", "F = ma", ["F = mv", "E = mc²", "p = F/S"]),
    ("物理", "高中", "物体动能的表达式是？", "½mv²", ["mv", "mgh", "mv²"]),
    ("物理", "高中", "忽略空气阻力，做自由落体运动的物体加速度约为？", "9.8 m/s²", ["0", "1 m/s²", "与质量成正比"]),
    ("物理", "高中", "电场强度的单位可以表示为？", "N/C", ["C/N", "J/s", "A·s"]),
    ("物理", "高中", "下列物理量中属于矢量的是？", "速度", ["质量", "时间", "温度"]),
    ("物理", "高中", "做匀速圆周运动的物体，其加速度方向？", "指向圆心", ["沿切线方向", "背离圆心", "与速度方向相同"]),
    ("物理", "高中", "电磁感应现象是哪位科学家发现的？", "法拉第", ["牛顿", "欧姆", "爱因斯坦"]),
    ("物理", "高中", "物体所受合外力为零时，它可能处于？", "匀速直线运动状态", ["匀加速直线运动状态", "匀速圆周运动状态", "平抛运动状态"]),
    ("物理", "高中", "一定质量的理想气体温度不变时，压强与体积？", "成反比", ["成正比", "无关", "平方成正比"]),
    ("物理", "高中", "光电效应说明光具有？", "粒子性", ["波动性", "偏振性", "干涉性"]),
    ("化学", "初中", "空气中体积分数最大的气体是？", "氮气", ["氧气", "二氧化碳", "稀有气体"]),
    ("化学", "初中", "下列变化属于化学变化的是？", "铁生锈", ["冰雪融化", "酒精挥发", "玻璃破碎"]),
    ("化学", "初中", "水的化学式是？", "H₂O", ["H₂O₂", "HO", "OH"]),
    ("化学", "初中", "能使澄清石灰水变浑浊的气体是？", "二氧化碳", ["氧气", "氮气", "氢气"]),
    ("化学", "初中", "常温下 pH = 7 的溶液呈？", "中性", ["酸性", "碱性", "无法确定"]),
    ("化学", "初中", "下列物质属于纯净物的是？", "蒸馏水", ["空气", "海水", "牛奶"]),
    ("化学", "初中", "实验室可以用下列哪种方法制取氧气？", "加热高锰酸钾", ["加热食盐", "大理石与稀盐酸反应", "锌与稀硫酸反应"]),
    ("化学", "初中", "地壳中含量最多的元素是？", "氧", ["硅", "铝", "铁"]),
    ("化学", "初中", "化学反应前后一定不变的是？", "原子的种类和数目", ["分子的种类", "物质的种类", "分子的数目"]),
    ("化学", "初中", "酸与碱作用生成盐和水的反应叫做？", "中和反应", ["化合反应", "分解反应", "置换反应"]),
    ("化学", "高中", "1 mol 任何粒子所含的粒子数约为？", "6.02×10²³", ["3.01×10²³", "6.02×10²²", "1.0×10²³"]),
    ("化学", "高中", "标准状况下，1 mol 气体的体积约为？", "22.4 L", ["11.2 L", "1 L", "2.24 L"]),
    ("化学", "高中", "同一主族元素从上到下，金属性？", "逐渐增强", ["逐渐减弱", "不变", "先增后减"]),
    ("化学", "高中", "下列物质中属于电解质的是？", "氯化钠", ["铜", "蔗糖", "酒精"]),
    ("化学", "高中", "氧化还原反应的本质是？", "电子的转移", ["有氧气参加", "生成沉淀", "放出热量"]),
    ("化学", "高中", "乙醇的结构简式是？", "CH₃CH₂OH", ["CH₃COOH", "CH₃OCH₃", "CH₄"]),
    ("化学", "高中", "可逆反应达到化学平衡时？", "正反应速率等于逆反应速率", ["反应完全停止", "各物质浓度一定相等", "正反应速率为零"]),
    ("化学", "高中", "金属钠在空气中燃烧生成？", "过氧化钠", ["氧化钠", "氢氧化钠", "碳酸钠"]),
    ("生物", "初中", "细胞中的遗传物质主要存在于？", "细胞核", ["细胞膜", "细胞质", "细胞壁"]),
    ("生物", "初中", "植物进行光合作用的主要场所是？", "叶绿体", ["线粒体", "细胞核", "液泡"]),
    ("生物", "初中", "人体最大的消化腺是？", "肝脏", ["胃", "胰腺", "唾液腺"]),
    ("生物", "初中", "血液中负责运输氧气的是？", "红细胞", ["白细胞", "血小板", "血浆蛋白"]),
    ("生物", "初中", "下列动物中属于哺乳动物的是？", "鲸", ["鲨鱼", "企鹅", "青蛙"]),
    ("生物", "初中", "细胞进行呼吸作用的主要场所是？", "线粒体", ["叶绿体", "核糖体", "高尔基体"]),
    ("生物", "初中", "一般种子萌发不需要的条件是？", "光照", ["适宜的温度", "一定的水分", "充足的空气"]),
    ("生物", "初中", "动物细胞与植物细胞相比，一般没有？", "细胞壁", ["细胞膜", "细胞核", "线粒体"]),
    ("生物", "高中", "DNA 的基本组成单位是？", "脱氧核苷酸", ["核糖核苷酸", "氨基酸", "葡萄糖"]),
    ("生物", "高中", "蛋白质的基本组成单位是？", "氨基酸", ["核苷酸", "脂肪酸", "单糖"]),
    ("生物", "高中", "有丝分裂中，染色体数目加倍发生在？", "后期", ["前期", "中期", "末期"]),
    ("生物", "高中", "孟德尔通过豌豆杂交实验发现了？", "遗传的分离定律和自由组合定律", ["自然选择学说", "细胞学说", "DNA 双螺旋结构"]),
    ("生物", "高中", "细胞膜的主要成分是？", "脂质和蛋白质", ["糖类和核酸", "纤维素和果胶", "蛋白质和核酸"]),
    ("生物", "高中", "ATP 的中文名称是？", "腺苷三磷酸", ["腺苷二磷酸", "脱氧核糖核酸", "葡萄糖"]),
    ("生物", "高中", "生态系统中能量流动的特点是？", "单向流动、逐级递减", ["循环流动、逐级递增", "单向流动、逐级递增", "循环流动、逐级递减"]),
    ("生物", "高中", "减数分裂产生的生殖细胞中，染色体数目与体细胞相比？", "减半", ["加倍", "不变", "变为四分之一"]),
    ("历史", "初中", "秦朝建立于？", "公元前 221 年", ["公元前 206 年", "公元前 202 年", "公元前 770 年"]),
    ("历史", "初中", "郑和下西洋发生在哪个朝代？", "明朝", ["元朝", "宋朝", "清朝"]),
    ("历史", "初中", "中华人民共和国成立于？", "1949 年", ["1911 年", "1921 年", "1937 年"]),
    ("历史", "初中", "辛亥革命爆发于？", "1911 年", ["1919 年", "1898 年", "1840 年"]),
    ("历史", "初中", "五四运动发生于？", "1919 年", ["1911 年", "1921 年", "1927 年"]),
    ("历史", "初中", "鸦片战争开始于？", "1840 年", ["1860 年", "1894 年", "1900 年"]),
    ("历史", "初中", "“贞观之治”出现在哪位皇帝统治时期？", "唐太宗", ["唐玄宗", "汉武帝", "宋太祖"]),
    ("历史", "初中", "出使西域、为开辟丝绸之路作出贡献的是？", "张骞", ["郑和", "玄奘", "鉴真"]),
    ("历史", "初中", "第一次工业革命首先开始于？", "英国", ["法国", "美国", "德国"]),
    ("历史", "初中", "美国《独立宣言》发表于？", "1776 年", ["1789 年", "1640 年", "1865 年"]),
    ("历史", "高中", "中国古代的科举制度创立于？", "隋朝", ["唐朝", "汉朝", "宋朝"]),
    ("历史", "高中", "新文化运动兴起的标志是创办了哪份刊物？", "《新青年》", ["《申报》", "《时务报》", "《民报》"]),
    ("历史", "高中", "1978 年十一届三中全会作出的伟大决策是？", "改革开放", ["土地改革", "三大改造", "抗美援朝"]),
    ("历史", "高中", "文艺复兴最早兴起于？", "意大利", ["英国", "法国", "德国"]),
    ("历史", "高中", "第二次世界大战结束于？", "1945 年", ["1939 年", "1918 年", "1949 年"]),
    ("历史", "高中", "罗斯福新政主要是为了应对？", "1929—1933 年经济大危机", ["第一次世界大战", "冷战", "石油危机"]),
    ("地理", "初中", "地球自转的方向是？", "自西向东", ["自东向西", "自南向北", "自北向南"]),
    ("地理", "初中", "世界上面积最大的大洲是？", "亚洲", ["非洲", "北美洲", "南极洲"]),
    ("地理", "初中", "我国面积最大的省级行政区是？", "新疆维吾尔自治区", ["西藏自治区", "内蒙古自治区", "青海省"]),
    ("地理", "初中", "我国最长的河流是？", "长江", ["黄河", "珠江", "黑龙江"]),
    ("地理", "初中", "赤道的纬度是？", "0°", ["90°", "23.5°", "180°"]),
    ("地理", "初中", "地球五带中，获得太阳热量最多的是？", "热带", ["北温带", "南温带", "北寒带"]),
    ("地理", "初中", "我国人口最多的民族是？", "汉族", ["壮族", "回族", "满族"]),
    ("地理", "初中", "我国地势的总体特征是？", "西高东低，呈阶梯状分布", ["东高西低", "南高北低", "中间高四周低"]),
    ("地理", "初中", "被称为“世界屋脊”的是？", "青藏高原", ["黄土高原", "云贵高原", "内蒙古高原"]),
    ("地理", "初中", "一般地图上的方向规定是？", "上北下南，左西右东", ["上南下北，左东右西", "上北下南，左东右西", "上东下西，左北右南"]),
    ("地理", "高中", "地球公转一周的时间约为？", "一年", ["一天", "一个月", "十年"]),
    ("地理", "高中", "北半球夏至日，太阳直射点位于？", "北回归线", ["赤道", "南回归线", "北极圈"]),
    ("地理", "高中", "形成季风的最主要原因是？", "海陆热力性质差异", ["地球自转", "洋流", "地形起伏"]),
    ("地理", "高中", "喜马拉雅山脉是由哪两个板块碰撞形成的？", "亚欧板块与印度洋板块", ["太平洋板块与亚欧板块", "美洲板块与非洲板块", "南极洲板块与印度洋板块"]),
    ("地理", "高中", "天气现象主要发生在大气的哪一层？", "对流层", ["平流层", "高层大气", "臭氧层"]),
    ("地理", "高中", "等高线地形图上，等高线越密集，表示坡度？", "越陡", ["越缓", "不变", "无法判断"]),
    ("政治", "初中", "我国的根本大法是？", "宪法", ["刑法", "民法典", "教育法"]),
    ("政治", "初中", "我国的根本政治制度是？", "人民代表大会制度", ["政治协商制度", "民族区域自治制度", "基层群众自治制度"]),
    ("政治", "初中", "我国的最高国家权力机关是？", "全国人民代表大会", ["国务院", "最高人民法院", "全国政协"]),
    ("政治", "初中", "对适龄儿童、少年来说，接受义务教育是？", "权利和义务", ["只是权利", "只是义务", "自愿选择"]),
    ("政治", "初中", "我国法律规定的未成年人是指未满多少周岁的公民？", "18 周岁", ["16 周岁", "14 周岁", "20 周岁"]),
    ("政治", "高中", "商品的两个基本属性是？", "使用价值和价值", ["价格和价值", "质量和数量", "交换价值和价格"]),
    ("政治", "高中", "决定商品价格的根本因素是？", "价值", ["供求关系", "国家政策", "消费心理"]),
    ("政治", "高中", "唯物辩证法的实质与核心是？", "对立统一规律", ["质量互变规律", "否定之否定规律", "物质决定意识"]),
    ("政治", "高中", "通货膨胀的主要表现是？", "物价全面持续上涨", ["物价全面持续下跌", "货币升值", "失业率下降"]),
    ("政治", "高中", "我国的基本经济制度中，居于主体地位的是？", "公有制", ["私有制", "外资经济", "个体经济"]),
    ("科学", "小学", "植物的根的主要作用是？", "吸收水分和固定植物", ["进行光合作用", "结出果实", "释放氧气"]),
    ("科学", "小学", "下列动物中属于昆虫的是？", "蝴蝶", ["蜘蛛", "蜗牛", "蚯蚓"]),
    ("科学", "小学", "地球绕着太阳转动叫做？", "公转", ["自转", "旋转", "摆动"]),
    ("科学", "小学", "月亮本身会发光吗？", "不会，它反射太阳光", ["会，自己发光", "只在晚上发光", "只在满月时发光"]),
    ("科学", "小学", "磁铁能吸引下列哪种物品？", "铁钉", ["木块", "塑料尺", "玻璃杯"]),
    ("科学", "小学", "人体进行呼吸的主要器官是？", "肺", ["心脏", "胃", "肝"]),
    ("科学", "小学", "一天中太阳位置最高的时候是？", "正午", ["早晨", "傍晚", "半夜"]),
    ("科学", "小学", "水结成冰时，温度通常是？", "0℃", ["100℃", "50℃", "-100℃"]),
    ("编程", "概念", "二分查找要求数组？", "有序", ["无序", "元素都相同", "长度为偶数"]),
    ("编程", "概念", "二分查找的时间复杂度是？", "O(log n)", ["O(n)", "O(n²)", "O(1)"]),
    ("编程", "概念", "栈的特点是？", "后进先出", ["先进先出", "随机出队", "按优先级出队"]),
    ("编程", "概念", "队列的特点是？", "先进先出", ["后进先出", "随机出队", "只能访问中间元素"]),
    ("编程", "概念", "冒泡排序的平均时间复杂度是？", "O(n²)", ["O(n log n)", "O(n)", "O(log n)"]),
    ("编程", "概念", "哈希表查找的平均时间复杂度是？", "O(1)", ["O(n)", "O(log n)", "O(n²)"]),
    ("编程", "概念", "递归函数必须有？", "终止条件", ["循环语句", "全局变量", "多个返回值"]),
    ("编程", "概念", "Python 中定义函数的关键字是？", "def", ["function", "fn", "func"]),
    ("编程", "概念", "Rust 中定义函数的关键字是？", "fn", ["def", "function", "func"]),
    ("编程", "概念", "二进制数 1010 等于十进制的？", "10", ["8", "12", "5"]),
    ("编程", "概念", "实现编辑器的“撤销”功能最适合用哪种数据结构？", "栈", ["队列", "哈希表", "堆"]),
    ("编程", "概念", "快速排序的平均时间复杂度是？", "O(n log n)", ["O(n²)", "O(n)", "O(log n)"]),
    ("编程", "概念", "在 Python 中，[1, 2, 3][-1] 的值是？", "3", ["1", "-1", "报错"]),
    ("编程", "概念", "一棵 3 层的满二叉树共有多少个节点？", "7", ["8", "6", "9"]),
    ("编程", "概念", "SQL 中用于查询数据的语句是？", "SELECT", ["INSERT", "UPDATE", "DELETE"]),
    ("编程", "概念", "HTTP 状态码 404 表示？", "资源未找到", ["服务器内部错误", "请求成功", "未授权"]),
    ("编程", "概念", "Git 中把暂存的改动记录到本地仓库的命令是？", "git commit", ["git push", "git clone", "git status"]),
    ("编程", "概念", "在边权非负的图中求单源最短路径，常用的算法是？", "Dijkstra 算法", ["冒泡排序", "KMP 算法", "二分查找"]),
    ("编程", "概念", "Python 中 7 // 2 的结果是？", "3", ["3.5", "4", "1"]),
    ("编程", "概念", "面向对象编程的三大特性是封装、继承和？", "多态", ["递归", "编译", "并发"]),
]

DESIGN = [
    ("is_prime(n)", "判断整数 n 是否为质数，返回 True 或 False。",
     "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
     ["is_prime(2) == True", "is_prime(1) == False", "is_prime(17) == True", "is_prime(21) == False", "is_prime(97) == True", "is_prime(0) == False"]),
    ("fibonacci(n)", "返回第 n 个斐波那契数，fibonacci(0) = 0，fibonacci(1) = 1。",
     "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     ["fibonacci(0) == 0", "fibonacci(1) == 1", "fibonacci(10) == 55", "fibonacci(20) == 6765"]),
    ("reverse_words(s)", "把字符串中的单词顺序反转，单词之间用一个空格分隔，例如 'hello world' 返回 'world hello'。",
     "def reverse_words(s):\n    return ' '.join(reversed(s.split()))",
     ["reverse_words('hello world') == 'world hello'", "reverse_words('a b c') == 'c b a'", "reverse_words('one') == 'one'"]),
    ("count_vowels(s)", "统计字符串中元音字母 a、e、i、o、u 的个数，不区分大小写。",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
     ["count_vowels('hello') == 2", "count_vowels('AEIOU') == 5", "count_vowels('rhythm') == 0", "count_vowels('Programming') == 3"]),
    ("gcd(a, b)", "返回两个正整数的最大公约数。",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
     ["gcd(12, 18) == 6", "gcd(7, 5) == 1", "gcd(100, 75) == 25", "gcd(9, 9) == 9"]),
    ("lcm(a, b)", "返回两个正整数的最小公倍数。",
     "def lcm(a, b):\n    x, y = a, b\n    while y:\n        x, y = y, x % y\n    return a * b // x",
     ["lcm(4, 6) == 12", "lcm(3, 5) == 15", "lcm(12, 18) == 36", "lcm(7, 7) == 7"]),
    ("flatten(items)", "把任意深度嵌套的列表展开成一维列表，例如 [1, [2, [3]]] 返回 [1, 2, 3]。",
     "def flatten(items):\n    result = []\n    for item in items:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
     ["flatten([1, [2, [3]]]) == [1, 2, 3]", "flatten([]) == []", "flatten([[1, 2], [3, [4, [5]]]]) == [1, 2, 3, 4, 5]"]),
    ("two_sum(nums, target)", "在列表中找到和为 target 的两个元素，返回它们的下标 [i, j]（i < j）；保证恰好有一个解。",
     "def two_sum(nums, target):\n    seen = {}\n    for j, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], j]\n        seen[x] = j",
     ["two_sum([2, 7, 11, 15], 9) == [0, 1]", "two_sum([3, 2, 4], 6) == [1, 2]", "two_sum([1, 5, 3, 8], 11) == [2, 3]"]),
    ("binary_search(items, x)", "在升序列表中二分查找 x，找到返回下标，找不到返回 -1。",
     "def binary_search(items, x):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == x:\n            return mid\n        if items[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
     ["binary_search([1, 3, 5, 7, 9], 7) == 3", "binary_search([1, 3, 5, 7, 9], 4) == -1", "binary_search([], 1) == -1", "binary_search([2], 2) == 0"]),
    ("merge_sorted(a, b)", "合并两个升序列表，返回一个新的升序列表。",
     "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]",
     ["merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]", "merge_sorted([], [1]) == [1]", "merge_sorted([1, 1], [1]) == [1, 1, 1]"]),
    ("is_palindrome(s)", "判断字符串是否为回文，只考虑字母和数字，忽略大小写。",
     "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
     ["is_palindrome('A man, a plan, a canal: Panama') == True", "is_palindrome('race a car') == False", "is_palindrome('') == True"]),
    ("factorial(n)", "返回非负整数 n 的阶乘。",
     "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
     ["factorial(0) == 1", "factorial(5) == 120", "factorial(10) == 3628800"]),
    ("sum_digits(n)", "返回非负整数 n 的各位数字之和。",
     "def sum_digits(n):\n    return sum(int(c) for c in str(n))",
     ["sum_digits(0) == 0", "sum_digits(123) == 6", "sum_digits(9999) == 36"]),
    ("max_subarray(nums)", "返回非空整数列表中和最大的连续子数组的和。",
     "def max_subarray(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best",
     ["max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6", "max_subarray([-3, -1, -2]) == -1", "max_subarray([5]) == 5"]),
    ("rotate_right(items, k)", "把列表向右循环移动 k 位，返回新列表，例如 rotate_right([1, 2, 3, 4], 1) 返回 [4, 1, 2, 3]。",
     "def rotate_right(items, k):\n    if not items:\n        return []\n    k %= len(items)\n    return items[-k:] + items[:-k] if k else list(items)",
     ["rotate_right([1, 2, 3, 4], 1) == [4, 1, 2, 3]", "rotate_right([1, 2, 3], 3) == [1, 2, 3]", "rotate_right([1, 2, 3], 5) == [2, 3, 1]", "rotate_right([], 2) == []"]),
    ("unique(items)", "去掉列表中的重复元素，保持第一次出现的顺序。",
     "def unique(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
     ["unique([3, 1, 3, 2, 1]) == [3, 1, 2]", "unique([]) == []", "unique(['a', 'b', 'a']) == ['a', 'b']"]),
    ("char_count(s)", "返回一个字典，记录字符串中每个字符出现的次数。",
     "def char_count(s):\n    d = {}\n    for c in s:\n        d[c] = d.get(c, 0) + 1\n    return d",
     ["char_count('aab') == {'a': 2, 'b': 1}", "char_count('') == {}", "char_count('xyzx') == {'x': 2, 'y': 1, 'z': 1}"]),
    ("roman_to_int(s)", "把罗马数字字符串转换为整数（1 到 3999）。",
     "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    for i, c in enumerate(s):\n        if i + 1 < len(s) and v[c] < v[s[i + 1]]:\n            total -= v[c]\n        else:\n            total += v[c]\n    return total",
     ["roman_to_int('III') == 3", "roman_to_int('IV') == 4", "roman_to_int('MCMXCIV') == 1994", "roman_to_int('LVIII') == 58"]),
    ("to_binary(n)", "把非负整数转换为二进制字符串（不带 0b 前缀），0 返回 '0'。",
     "def to_binary(n):\n    return bin(n)[2:]",
     ["to_binary(0) == '0'", "to_binary(5) == '101'", "to_binary(255) == '11111111'"]),
    ("valid_brackets(s)", "判断只含 ()[]{} 的字符串中括号是否正确配对和嵌套。",
     "def valid_brackets(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif not stack or stack.pop() != pairs[c]:\n            return False\n    return not stack",
     ["valid_brackets('()[]{}') == True", "valid_brackets('([)]') == False", "valid_brackets('{[]}') == True", "valid_brackets('(') == False", "valid_brackets('') == True"]),
    ("transpose(matrix)", "返回矩阵（列表的列表）的转置。",
     "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]",
     ["transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]", "transpose([[1]]) == [[1]]"]),
    ("is_anagram(a, b)", "判断两个字符串是否由相同的字母以不同顺序组成（区分大小写）。",
     "def is_anagram(a, b):\n    return sorted(a) == sorted(b)",
     ["is_anagram('listen', 'silent') == True", "is_anagram('abc', 'abd') == False", "is_anagram('aab', 'aba') == True"]),
    ("second_largest(nums)", "返回列表中第二大的不同数值；如果不存在返回 None。",
     "def second_largest(nums):\n    values = sorted(set(nums), reverse=True)\n    return values[1] if len(values) > 1 else None",
     ["second_largest([3, 1, 4, 4]) == 3", "second_largest([5, 5]) == None", "second_largest([1, 2]) == 1", "second_largest([]) == None"]),
    ("run_length(s)", "对字符串做游程编码，例如 'aaabcc' 返回 'a3b1c2'。",
     "def run_length(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(s[i] + str(j - i))\n        i = j\n    return ''.join(out)",
     ["run_length('aaabcc') == 'a3b1c2'", "run_length('') == ''", "run_length('abc') == 'a1b1c1'"]),
    ("caesar(s, k)", "凯撒加密：把每个英文字母向后移动 k 位（循环，保持大小写），其他字符不变。",
     "def caesar(s, k):\n    out = []\n    for c in s:\n        if c.islower():\n            out.append(chr((ord(c) - 97 + k) % 26 + 97))\n        elif c.isupper():\n            out.append(chr((ord(c) - 65 + k) % 26 + 65))\n        else:\n            out.append(c)\n    return ''.join(out)",
     ["caesar('abc', 1) == 'bcd'", "caesar('xyz', 3) == 'abc'", "caesar('Hello, World!', 5) == 'Mjqqt, Btwqi!'"]),
    ("prime_factors(n)", "返回正整数 n（n ≥ 2）的质因数分解列表，按从小到大排列并包含重复因子。",
     "def prime_factors(n):\n    out = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            out.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        out.append(n)\n    return out",
     ["prime_factors(12) == [2, 2, 3]", "prime_factors(13) == [13]", "prime_factors(360) == [2, 2, 2, 3, 3, 5]"]),
    ("is_leap_year(year)", "判断公历年份是否为闰年。",
     "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
     ["is_leap_year(2000) == True", "is_leap_year(1900) == False", "is_leap_year(2024) == True", "is_leap_year(2023) == False"]),
    ("word_count(text)", "统计英文文本中每个单词出现的次数（按空白分隔、转为小写），返回字典。",
     "def word_count(text):\n    d = {}\n    for w in text.lower().split():\n        d[w] = d.get(w, 0) + 1\n    return d",
     ["word_count('a b a') == {'a': 2, 'b': 1}", "word_count('') == {}", "word_count('The the THE') == {'the': 3}"]),
    ("matmul(a, b)", "计算两个矩阵（列表的列表）的乘积。",
     "def matmul(a, b):\n    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]",
     ["matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]", "matmul([[1, 0, 2]], [[1], [2], [3]]) == [[7]]"]),
    ("common_prefix(words)", "返回字符串列表的最长公共前缀，没有则返回空字符串。",
     "def common_prefix(words):\n    if not words:\n        return ''\n    prefix = words[0]\n    for w in words[1:]:\n        while not w.startswith(prefix):\n            prefix = prefix[:-1]\n    return prefix",
     ["common_prefix(['flower', 'flow', 'flight']) == 'fl'", "common_prefix(['dog', 'car']) == ''", "common_prefix(['same']) == 'same'", "common_prefix([]) == ''"]),
]


def fmt(value):
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def math_items(rng):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    c, d = rng.randint(2, 9), rng.randint(2, 9)
    yield "小学", f"计算：{a} × {b} = ?", a * b
    yield "小学", f"计算：{a * c} ÷ {c} + {b} = ?", a + b
    big, small = max(a, b), min(a, b)
    yield "小学", f"计算：{big * d + c} - {small} × {d} = ?", big * d + c - small * d
    price, count, money = rng.randint(3, 15), rng.randint(2, 9), 200
    yield "小学", f"每支笔 {price} 元，小明买了 {count} 支，付了 {money} 元，应找回多少元？", money - price * count
    w, h = rng.randint(3, 20), rng.randint(3, 20)
    yield "小学", f"一个长方形长 {w} 厘米、宽 {h} 厘米，它的周长是多少厘米？", 2 * (w + h)
    yield "小学", f"一个长方形长 {w} 厘米、宽 {h} 厘米，它的面积是多少平方厘米？", w * h
    xs = [rng.randint(60, 100) for _ in range(4)]
    xs[-1] += (4 - sum(xs) % 4) % 4
    yield "小学", f"四次测验的成绩分别是 {', '.join(map(str, xs))} 分，平均成绩是多少分？", sum(xs) // 4
    m = rng.randint(2, 9)
    yield "小学", f"{m} 米等于多少厘米？", m * 100
    x, k, r = rng.randint(-9, 12), rng.choice([2, 3, 4, 5, 6, 7]), rng.randint(-20, 20)
    yield "初中", f"解方程：{k}x + {r} = {k * x + r}，x = ?".replace("+ -", "- "), x
    x, y = rng.randint(-6, 9), rng.randint(-6, 9)
    yield "初中", f"已知 x + y = {x + y}，x - y = {x - y}，求 x 的值。", x
    p, base = rng.choice([10, 20, 25, 30, 40, 50, 60, 75]), rng.choice([80, 120, 200, 240, 400, 600])
    yield "初中", f"{base} 的 {p}% 是多少？", base * p / 100
    t = rng.choice([(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (9, 12, 15), (7, 24, 25)])
    yield "初中", f"直角三角形的两条直角边分别是 {t[0]} 和 {t[1]}，斜边长是多少？", t[2]
    n = rng.randint(5, 12)
    yield "初中", f"{n} 边形的内角和是多少度？", (n - 2) * 180
    s = rng.randint(4, 30)
    yield "初中", f"{s * s} 的算术平方根是多少？", s
    k2, b2, x2 = rng.randint(-5, 6), rng.randint(-9, 9), rng.randint(-5, 5)
    yield "初中", f"一次函数 y = {k2}x + {b2}，当 x = {x2} 时，y = ?".replace("+ -", "- "), k2 * x2 + b2
    a2, b3, c2, x3 = rng.randint(1, 4), rng.randint(-6, 6), rng.randint(-9, 9), rng.randint(-3, 4)
    yield "高中", f"已知 f(x) = {a2}x² + {b3}x + {c2}，求 f({x3}) 的值。".replace("+ -", "- "), a2 * x3 * x3 + b3 * x3 + c2
    yield "高中", f"已知 f(x) = {a2}x³ + {b3}x² + {c2}x，求 f'(1) 的值。".replace("+ -", "- "), 3 * a2 + 2 * b3 + c2
    a1, dd, nn = rng.randint(-5, 10), rng.randint(2, 7), rng.randint(5, 15)
    yield "高中", f"等差数列首项为 {a1}，公差为 {dd}，求第 {nn} 项。", a1 + (nn - 1) * dd
    yield "高中", f"等差数列首项为 {a1}，公差为 {dd}，求前 {nn} 项的和。", nn * a1 + nn * (nn - 1) // 2 * dd
    g1, q, gn = rng.randint(1, 5), rng.choice([2, 3]), rng.randint(3, 7)
    yield "高中", f"等比数列首项为 {g1}，公比为 {q}，求第 {gn} 项。", g1 * q ** (gn - 1)
    cn, ck = rng.randint(5, 10), rng.randint(2, 3)
    yield "高中", f"从 {cn} 个不同元素中选出 {ck} 个（不考虑顺序），共有多少种选法？", math.comb(cn, ck)
    yield "高中", f"从 {cn} 个不同元素中选出 {ck} 个排成一列，共有多少种排法？", math.perm(cn, ck)
    e = rng.randint(2, 10)
    yield "高中", f"log₂{2 ** e} 的值是多少？", e
    v1, v2 = (rng.randint(-5, 5), rng.randint(-5, 5)), (rng.randint(-5, 5), rng.randint(-5, 5))
    yield "高中", f"向量 a = ({v1[0]}, {v1[1]})，b = ({v2[0]}, {v2[1]})，求 a·b。", v1[0] * v2[0] + v1[1] * v2[1]


def physics_items(rng):
    s, t = rng.randint(2, 30) * 10, rng.randint(2, 10)
    yield "初中", f"一辆车在 {t} 秒内匀速行驶了 {s * t // 10 * 10} 米，它的速度是多少米每秒？", s * t // 10 * 10 / t
    m, v = rng.randint(2, 20) * 100, rng.choice([100, 200, 250, 400, 500])
    yield "初中", f"一个物体质量为 {m} 克、体积为 {v} 立方厘米，它的密度是多少克每立方厘米？", m / v
    f, area = rng.randint(1, 20) * 100, rng.choice([0.1, 0.2, 0.5, 2, 4])
    yield "初中", f"压力为 {f} 牛，受力面积为 {fmt(area)} 平方米，压强是多少帕？", f / area
    u, r = rng.randint(1, 12) * 3, rng.choice([2, 3, 4, 6, 12])
    yield "初中", f"一个电阻两端电压为 {u} 伏，电阻为 {r} 欧，通过它的电流是多少安？", u / r
    force, dist = rng.randint(2, 50) * 10, rng.randint(2, 20)
    yield "初中", f"用 {force} 牛的力沿力的方向把物体推动了 {dist} 米，做功多少焦？", force * dist
    work, time = rng.randint(2, 40) * 300, rng.choice([2, 3, 5, 10, 20, 30])
    yield "初中", f"某机器在 {time} 秒内做功 {work} 焦，它的功率是多少瓦？", work / time
    mass = rng.randint(1, 30)
    yield "初中", f"质量为 {mass} 千克的物体所受重力是多少牛？（g 取 10 N/kg）", mass * 10
    mm, aa = rng.randint(1, 20), rng.randint(1, 10)
    yield "高中", f"质量为 {mm} 千克的物体获得 {aa} 米每二次方秒的加速度，它所受合力是多少牛？", mm * aa
    v0, acc, tt = rng.randint(0, 10), rng.randint(1, 5), rng.randint(2, 8)
    yield "高中", f"物体初速度为 {v0} 米每秒，加速度为 {acc} 米每二次方秒，做匀加速直线运动 {tt} 秒后速度是多少米每秒？", v0 + acc * tt
    yield "高中", f"物体初速度为 {v0} 米每秒，加速度为 {acc} 米每二次方秒，做匀加速直线运动 {tt} 秒内的位移是多少米？", v0 * tt + acc * tt * tt / 2
    km, kv = rng.randint(1, 10), rng.randint(1, 10) * 2
    yield "高中", f"质量为 {km} 千克的物体速度为 {kv} 米每秒，它的动能是多少焦？", km * kv * kv / 2
    wm, dt = rng.choice([1, 2, 5]), rng.choice([10, 20, 30, 50])
    yield "初中", f"把 {wm} 千克水的温度升高 {dt}℃，需要吸收多少焦热量？（水的比热容取 4200 J/(kg·℃)）", 4200 * wm * dt


def chemistry_items(rng):
    molar = {"H₂O": 18, "CO₂": 44, "NaCl": 58.5, "O₂": 32, "H₂": 2, "CH₄": 16, "NaOH": 40, "CaCO₃": 100, "HCl": 36.5, "NH₃": 17}
    name = rng.choice(sorted(molar))
    yield "高中", f"{name} 的摩尔质量是多少克每摩尔？（相对原子质量 H=1, C=12, N=14, O=16, Na=23, Cl=35.5, Ca=40）", molar[name]
    n = rng.choice([0.5, 1, 2, 3, 4])
    yield "高中", f"{fmt(n * molar[name])} 克 {name} 的物质的量是多少摩尔？（{name} 的摩尔质量为 {fmt(molar[name])} g/mol）", n
    solute, total = rng.randint(1, 9) * 5, rng.choice([100, 200, 250, 500])
    yield "初中", f"把 {solute} 克食盐溶解在水中配成 {total} 克溶液，溶质的质量分数是百分之多少？", solute * 100 / total


def code_items(rng):
    a, b = rng.randint(1, 10), rng.randint(11, 30)
    yield f"print(sum(range({a}, {b})))"
    n, k = rng.randint(3, 6), rng.randint(2, 5)
    yield f"print([x * {k} for x in range({n})])"
    word = rng.choice(["python", "matrix", "planet", "rocket", "silver", "garden", "window", "bridge"])
    yield f"s = '{word}'\nprint(s[::-1])"
    yield f"s = '{word}'\nprint(s[1:4].upper())"
    n, m = rng.randint(10, 30), rng.randint(2, 5)
    yield f"t = 0\nfor i in range(1, {n} + 1):\n    if i % {m} == 0:\n        t += i\nprint(t)"
    text = rng.choice(["banana", "mississippi", "abracadabra", "bookkeeper", "committee"])
    ch = rng.choice(sorted(set(text)))
    yield f"d = {{}}\nfor c in '{text}':\n    d[c] = d.get(c, 0) + 1\nprint(d['{ch}'])"
    n = rng.randint(20, 300)
    yield f"n = {n}\ncount = 0\nwhile n > 1:\n    n = n // 2\n    count += 1\nprint(count)"
    k = rng.randint(3, 7)
    yield f"def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\nprint(f({k}))"
    items = rng.sample(range(1, 50), 6)
    yield f"a = {items}\nprint(a[1:4])"
    words = rng.sample(["sun", "apple", "kiwi", "banana", "fig", "cherry", "plum"], 4)
    yield f"words = {words}\nprint(sorted(words, key=len)[0])"
    nums = rng.sample(range(-20, 40), 5)
    yield f"nums = {nums}\nprint(max(nums) - min(nums))"
    x = rng.randint(2, 9)
    yield f"x = {x}\nx += 3\nx *= 2\nprint(x)"
    nums = rng.sample(range(1, 30), 6)
    yield f"nums = {nums}\nprint([n for n in nums if n % 2 == 0])"
    n = rng.randint(3, 8)
    yield f"a, b = 0, 1\nfor _ in range({n}):\n    a, b = b, a + b\nprint(a)"


def run(code):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, {})
    return buffer.getvalue().strip()


def mc_prompt(subject, level, question, options):
    body = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", options))
    return CHAT.format(question=f"以下是{level}{subject}单项选择题，请直接回答正确选项的字母。\n\n{question}\n{body}") + "答案："


def numeric_prompt(level, subject, question):
    return CHAT.format(question=f"{level}{subject}题：{question}\n请直接给出最终答案，只写一个数。") + "答案："


def code_prompt(code):
    return CHAT.format(question=f"下面的 Python 代码运行后输出什么？只写输出内容。\n\n```python\n{code}\n```") + "输出："


def design_prompt(signature, description):
    return CHAT.format(question=f"请用 Python 编写函数 `{signature}`：{description}\n只输出完整的函数代码。")


def build(seed_train=1, seed_test=2, rounds_train=24, rounds_test=6):
    mc = []
    for index, (subject, level, question, correct, wrong) in enumerate(MC):
        rng = random.Random(1000 + index)
        options = [correct] + list(wrong)
        rng.shuffle(options)
        letter = "ABCD"[options.index(correct)]
        body = "\n".join(f"{x}. {text}" for x, text in zip("ABCD", options))
        mc.append({"kind": "mc", "subject": subject, "level": level, "prompt": mc_prompt(subject, level, question, options), "answer": letter,
                   "question": f"{level}{subject}选择题：{question}\n{body}"})
    design = []
    for index, (signature, description, solution, tests) in enumerate(DESIGN):
        design.append({"kind": "design", "subject": "编程", "level": "程序设计", "prompt": design_prompt(signature, description),
                       "answer": f"```python\n{solution}\n```", "tests": tests, "solution": solution,
                       "question": f"请用 Python 编写函数 `{signature}`：{description}", "name": signature.split("(")[0]})

    def generated(seed, rounds):
        rng = random.Random(seed)
        items = []
        for _ in range(rounds):
            for level, question, value in math_items(rng):
                items.append({"kind": "numeric", "subject": "数学", "level": level, "prompt": numeric_prompt(level, "数学", question), "answer": fmt(value), "value": float(value), "question": question})
            for level, question, value in physics_items(rng):
                items.append({"kind": "numeric", "subject": "物理", "level": level, "prompt": numeric_prompt(level, "物理", question), "answer": fmt(value), "value": float(value), "question": question})
            for level, question, value in chemistry_items(rng):
                items.append({"kind": "numeric", "subject": "化学", "level": level, "prompt": numeric_prompt(level, "化学", question), "answer": fmt(value), "value": float(value), "question": question})
            for code in code_items(rng):
                items.append({"kind": "code", "subject": "编程", "level": "读程序", "prompt": code_prompt(code), "answer": run(code),
                              "question": f"下面的 Python 代码运行后输出什么？\n\n```python\n{code}\n```"})
        return items

    train_generated = generated(seed_train, rounds_train)
    test_generated = generated(seed_test, rounds_test)
    test_prompts = {item["prompt"] for item in test_generated}
    train_generated = [item for item in train_generated if item["prompt"] not in test_prompts]
    train = [item for index, item in enumerate(mc) if index % 2 == 0] + [item for index, item in enumerate(design) if index % 2 == 0] + train_generated
    test = [item for index, item in enumerate(mc) if index % 2 == 1] + [item for index, item in enumerate(design) if index % 2 == 1] + test_generated
    return train, test


def check():
    for signature, description, solution, tests in DESIGN:
        namespace = {}
        exec(solution, namespace)
        for test in tests:
            assert eval(test, namespace), (signature, test)
    train, test = build()
    from collections import Counter
    print("train", len(train), Counter(item["kind"] for item in train))
    print("test", len(test), Counter(item["kind"] for item in test))
    print("letters", Counter(item["answer"] for item in train + test if item["kind"] == "mc"))
    print(test[0]["prompt"] + test[0]["answer"])
    print([item["prompt"].split("\n")[1][:60] + " => " + item["answer"] for item in test if item["kind"] == "numeric"][:8])
    print([item["answer"] for item in test if item["kind"] == "code"][:8])


if __name__ == "__main__":
    check()
