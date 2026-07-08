# Outputs Directory

本目录用于保存本地实验输出，默认不纳入 Git 版本库。

常见结构：

```text
outputs/<run-name>/
├── tables/        # CSV/JSON 结果表与 metadata
├── figures/       # 单次实验图表
├── checkpoints/   # 模型权重
└── logs/          # 长任务日志
```

论文级图片已整理到 `docs/paper_figures/` 并纳入版本库。模型 checkpoint、日志和大规模中间结果请保留在本地或通过外部存储发布。

