# Data Directory

本目录用于存放本地数据集和数据缓存，默认不纳入 Git 版本库。

推荐结构：

```text
data/
├── cifar-10-batches-py/     # CIFAR-10 Python 版解压目录
└── voxforge_task2/          # VoxForge Spanish 缓存，由脚本自动生成
```

CIFAR-10 脚本当前使用 `download=False`，因此运行前需要先准备好 `data/cifar-10-batches-py/`。

VoxForge 脚本会通过 Hugging Face `datasets` 读取 `ciempiess/voxforge_spanish`，并在本目录下生成 `.pt` 与 `.json` 缓存文件。

