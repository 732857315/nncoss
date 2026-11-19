# 数据

本目录保存知识与大脑分离实验所需的数据：测验题与外部知识。这些数据只用于测验和检索，不用于训练或修改权重。大文件通过 Git LFS 管理。

## 测验

| 目录 | 内容 | 来源与许可 |
|---|---|---|
| `eval/gaokao/` | 2010–2022 年高考客观题 14 类共 1781 道，含标准答案与解析 | [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench)，Apache License 2.0，许可证原文见 `eval/gaokao/LICENSE` |
| `eval/code_contests/test.parquet`（不入库） | 165 道 Codeforces 竞赛题（难度 800–3500），含公开、隐藏与生成的测试数据 | DeepMind [CodeContests](https://github.com/google-deepmind/code_contests) 测试集，CC BY 4.0，托管在 [Hugging Face](https://huggingface.co/datasets/deepmind/code_contests)；可公开获取，运行 `python3 experiments/knowledge/fetch_eval.py` 下载到本目录 |

## 外部知识

| 文件 | 内容 | 来源与许可 |
|---|---|---|
| `knowledge/oiwiki.jsonl` | OI Wiki 的算法与计算机知识，按标题切成 12138 段，每段带出处 | [OI Wiki](https://github.com/OI-wiki/OI-wiki)，CC BY-SA 4.0 与 The Star And Thank Author License；本文件作为改编作品同样按 CC BY-SA 4.0 提供 |
| `knowledge/textbooks-manifest.json` | 选用的 160 册人教版和统编版小学、初中、高中课本清单，包括仓库路径、分卷大小和 Git 对象哈希 | 课本来自 [ChinaTextbook](https://github.com/TapXWorld/ChinaTextbook)，版权归各出版社所有 |

课本 PDF 及从中提取的全文受版权保护，不提交到本仓库，只保存在本机仓库外的目录中供个人研究使用。复现步骤：

1. 用 `experiments/knowledge/fetch_textbooks.py <目录>` 按清单下载课本，分卷文件会自动拼接成完整 PDF。
2. 用 `experiments/knowledge/extract_passages.py --textbooks <目录>/pdf --output <段落.jsonl>` 生成带出处的段落。
3. 用 `nncoss-know ingest --store <知识库目录> --input <段落.jsonl>` 导入知识库。

每段的格式为 `{"id": ..., "source": ..., "text": ...}`，与 `nncoss-know ingest` 的输入格式一致。
