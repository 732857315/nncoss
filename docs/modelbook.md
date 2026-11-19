# 外部资料（modelbook）

`modelbook/` 保存知识与大脑分离实验用到的外部资料：测验题、课本、算法资料，以及由它们生成的知识库。这些资料只用于测验和检索，不用于训练或改写权重。

可以公开再分发的部分提交到仓库，大文件用 Git LFS 管理。受版权保护或能按下文重新获取、生成的部分只保存在本机，已列入 `.gitignore`。

范围是幼儿园到高中。课本来源 ChinaTextbook 只有小学、初中、高中，另有五四学制、大学与习题目录：五四学制与六三学制的内容相同，不重复选用；大学教材不在范围内。幼儿园没有全国统一教材，山东省使用的《山东省幼儿园课程指导》（山东省教学研究室与明天出版社编）没有公开的电子版，因此目前不含幼儿园资料。

## 目录

| 路径 | 内容 | 入库 | 来源与许可 |
|---|---|---|---|
| `eval/gaokao/` | 2010–2022 年高考客观题 14 类共 1781 道，含标准答案与解析 | 是 | [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench) 的 `Data/Objective_Questions/`，原样复制；Apache License 2.0，许可证原文见 `eval/gaokao/LICENSE` |
| `eval/code_contests/test.parquet` | 165 道 Codeforces 竞赛题（难度 800–3500），含公开、隐藏与生成的测试数据 | 否 | DeepMind [CodeContests](https://github.com/google-deepmind/code_contests) 测试集，CC BY 4.0，托管在 [Hugging Face](https://huggingface.co/datasets/deepmind/code_contests) |
| `knowledge/oiwiki.jsonl` | OI Wiki 的算法与计算机知识，按标题切成 12138 段，每段带出处 | 是（LFS） | [OI Wiki](https://github.com/OI-wiki/OI-wiki)，CC BY-SA 4.0 与 The Star And Thank Author License；本文件作为改编作品同样按 CC BY-SA 4.0 提供 |
| `knowledge/textbooks-manifest.json` | 选用的 323 册小学、初中、高中课本清单：全部学科，每科一个版本（优先人教版、统编版），另加 19 套初中数学中考真题与练习；记录源仓库路径、分卷大小与 Git 对象哈希 | 是 | 课本来自 [ChinaTextbook](https://github.com/TapXWorld/ChinaTextbook)，版权归各出版社所有 |
| `textbooks/pdf/` | 按清单下载的课本 PDF，约 5.91 GB | 否 | 同上，受版权保护，仅供个人研究 |
| `textbooks/tree.json` | ChinaTextbook 仓库的完整文件树，重新选书时使用 | 否 | GitHub API |
| `knowledge/textbooks.jsonl` | 从课本 PDF 提取的正文段落，每段带出处；由最初 160 册生成时为 32,775 段 | 否 | 由课本生成，受版权保护 |
| `knowledge/store-base/` | 课本与 OI Wiki 合并后的知识库，`nncoss-know` 直接读取；由 160 册课本生成时共 44,913 段 | 否 | 由上面两份段落生成 |
| `sources/` | OI Wiki 与 GAOKAO-Bench 的源码压缩包及解压目录 | 否 | 各自仓库 |

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

2. 从 PDF 提取段落。脚本按段落切分，并删除没有 Unicode 映射的字体造成的乱码；一段中可读字符不到一半时整段丢弃：

   ```bash
   python3 tools/extract_passages.py --textbooks modelbook/textbooks/pdf --output modelbook/knowledge/textbooks.jsonl
   ```

   `oiwiki.jsonl` 由同一个脚本从 OI Wiki 源码生成，按 Markdown 标题切分：

   ```bash
   python3 tools/extract_passages.py --oiwiki modelbook/sources/OI-wiki/OI-wiki-master --output modelbook/knowledge/oiwiki.jsonl
   ```

3. 导入知识库：

   ```bash
   target/release/nncoss-know ingest --store modelbook/knowledge/store-base --input modelbook/knowledge/textbooks.jsonl
   target/release/nncoss-know ingest --store modelbook/knowledge/store-base --input modelbook/knowledge/oiwiki.jsonl
   ```

4. 下载 CodeContests 测试集到 `modelbook/eval/code_contests/test.parquet`：

   ```bash
   python3 tools/fetch_eval.py
   ```

每段的格式为 `{"id": ..., "source": ..., "text": ...}`，与 `nncoss-know ingest` 的输入格式一致。测评时 `tools/eval_gaokao.py` 会把知识库复制到 `tmp/<标签>/store`，模型在测评中写下的笔记只进入这份副本，不改动 `store-base`。
