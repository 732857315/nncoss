# 外部资料（modelbook）

`modelbook/` 保存知识与大脑分离实验用到的外部资料：测验题、课本、算法资料，以及由它们生成的知识库。这些资料只用于测验和检索，不用于训练或改写权重。

可以公开再分发的部分提交到仓库，大文件用 Git LFS 管理。受版权保护或能按下文重新获取、生成的部分只保存在本机，已列入 `.gitignore`。

范围是幼儿园到高中。课本来源 ChinaTextbook 只有小学、初中、高中，另有五四学制、大学与习题目录：五四学制与六三学制的内容相同，不重复选用；大学教材不在范围内。幼儿园没有全国统一教材，山东省使用的《山东省幼儿园课程指导》（山东省教学研究室与明天出版社编）没有公开的电子版，网上只有来源不明的个人上传，不予采用；幼儿园部分改用教育部公开发布的《3-6岁儿童学习与发展指南》（2012）与《幼儿园教育指导纲要（试行）》（2001），两者规定了 3–6 岁儿童在健康、语言、社会、科学、艺术五个领域的学习与发展目标和教育建议。

## 目录

| 路径 | 内容 | 入库 | 来源与许可 |
|---|---|---|---|
| `eval/gaokao/` | 2010–2022 年高考客观题 14 类共 1781 道，含标准答案与解析 | 是 | [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench) 的 `Data/Objective_Questions/`，原样复制；Apache License 2.0，许可证原文见 `eval/gaokao/LICENSE` |
| `eval/gaokao/subjective/` | 2010–2022 年高考主观题 18 类共 1030 道（解答题、填空题、阅读、默写、写作技能、改错等），含参考答案与分值 | 是 | GAOKAO-Bench 的 `Data/Subjective_Questions/`，原样复制；许可同上 |
| `eval/code_contests/test.parquet` | 165 道 Codeforces 竞赛题（难度 800–3500），含公开、隐藏与生成的测试数据 | 否 | DeepMind [CodeContests](https://github.com/google-deepmind/code_contests) 测试集，CC BY 4.0，托管在 [Hugging Face](https://huggingface.co/datasets/deepmind/code_contests) |
| `knowledge/oiwiki.jsonl` | OI Wiki 的算法与计算机知识，按标题切成 12138 段，每段带出处 | 是（LFS） | [OI Wiki](https://github.com/OI-wiki/OI-wiki)，CC BY-SA 4.0 与 The Star And Thank Author License；本文件作为改编作品同样按 CC BY-SA 4.0 提供 |
| `knowledge/textbooks-manifest.json` | 选用的 318 册小学、初中、高中课本清单：全部学科，每科一个版本（优先人教版、统编版），另加 19 套初中数学中考真题与练习；记录源仓库路径、分卷大小与 Git 对象哈希 | 是 | 课本来自 [ChinaTextbook](https://github.com/TapXWorld/ChinaTextbook)，版权归各出版社所有 |
| `textbooks/pdf/` | 按清单下载的课本 PDF，约 5.61 GB | 否 | 同上，受版权保护，仅供个人研究 |
| `textbooks/text/幼儿园/` | 上述两份幼儿园指导文件整理成的纯文本，共约 2.9 万字；原件在 `sources/kindergarten/` | 否 | 教育部[《3-6岁儿童学习与发展指南》](http://www.moe.gov.cn/srcsite/A06/s3327/201210/t20121009_143254.html)、[《幼儿园教育指导纲要（试行）》](http://www.moe.gov.cn/srcsite/A06/s3327/200107/t20010702_81984.html) |
| `textbooks/tree.json` | ChinaTextbook 仓库的完整文件树，重新选书时使用 | 否 | GitHub API |
| `knowledge/textbooks.jsonl` | 从课本 PDF 与幼儿园文本提取的正文段落，每段带出处 | 否 | 由课本生成，受版权保护 |
| `knowledge/store-base/` | 课本、幼儿园文本与 OI Wiki 合并后的知识库，`nncoss-know` 直接读取 | 否 | 由上面的段落生成 |
| `knowledge/store-160/` | 最初 160 册课本与 OI Wiki 的知识库，共 44,913 段；[实验记录](brain-experiments.md)第五章的结果都用它测得 | 否 | 同上 |
| `sources/` | OI Wiki 与 GAOKAO-Bench 的源码压缩包及解压目录，幼儿园文件原件 | 否 | 各自来源 |

## 复现

所有命令都从仓库根目录执行。课本与 OI Wiki 的处理需要 [tools/knowledge_requirements.txt](../tools/knowledge_requirements.txt) 中的 PyMuPDF。

1. 下载课本。脚本按清单下载到 `modelbook/textbooks/pdf/`，自动拼接分卷，并逐个分卷核对 Git 对象哈希；已下载的文件会跳过，加 `--verify` 时也重新核对：

   ```bash
   python3 tools/fetch_textbooks.py
   ```

   要调整选书，先取得完整文件树，再按脚本中 `WANT` 的版本规则重新生成清单：

   ```bash
   curl -sSL "https://api.github.com/repos/TapXWorld/ChinaTextbook/git/trees/master?recursive=1" -o modelbook/textbooks/tree.json
   python3 tools/fetch_textbooks.py --select modelbook/textbooks/tree.json
   ```

2. 下载幼儿园文件并整理成纯文本。`.doc` 与网页的转换用 macOS 自带的 textutil，其他系统需要 LibreOffice：

   ```bash
   python3 tools/fetch_kindergarten.py
   ```

3. 提取段落。课本 PDF 按页、按段落切分，并删除没有 Unicode 映射的字体造成的乱码，一段中可读字符不到一半时整段丢弃；纯文本按段落切分：

   ```bash
   python3 tools/extract_passages.py --textbooks modelbook/textbooks/pdf --texts modelbook/textbooks/text \
     --output modelbook/knowledge/textbooks.jsonl
   ```

   `oiwiki.jsonl` 由同一个脚本从 OI Wiki 源码生成，按 Markdown 标题切分：

   ```bash
   python3 tools/extract_passages.py --oiwiki modelbook/sources/OI-wiki/OI-wiki-master --output modelbook/knowledge/oiwiki.jsonl
   ```

4. 导入知识库：

   ```bash
   target/release/nncoss-know ingest --store modelbook/knowledge/store-base --input modelbook/knowledge/textbooks.jsonl
   target/release/nncoss-know ingest --store modelbook/knowledge/store-base --input modelbook/knowledge/oiwiki.jsonl
   ```

5. 下载 CodeContests 测试集到 `modelbook/eval/code_contests/test.parquet`：

   ```bash
   python3 tools/fetch_eval.py
   ```

每段的格式为 `{"id": ..., "source": ..., "text": ...}`，与 `nncoss-know ingest` 的输入格式一致。测评时 `tools/eval_gaokao.py` 会把知识库复制到 `tmp/<标签>/store`，模型在测评中写下的笔记只进入这份副本，不改动 `store-base`。
