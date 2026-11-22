# 项目数据目录

当前仅保留官方 A 任务基线包 `official_a/`。额外原始批次副本与 Excel 导出已于 2026-11-19 清理。

| 目录 | 内容 |
|---|---|
| `official_a/public/` | 当前闭环使用的公开 train.jsonl 与 validation.jsonl |
| `official_a/checkpoint.npz` | 原始预训练基线 |
| `official_a/training.json`、`official_a/history.jsonl` | 随包基线的原始训练记录 |
| `official_a/TASK.md`、`official_a/TRAINING_GUIDE.md` | 任务与续训说明 |

| 数据集 | 样本数 |
|---|---:|
| 公开训练集（batch1） | 445 |
| 公开验证集（batch2） | 1244 |

每行 JSONL 的 `inputs.features` 为 128 维输入，`answer` 为六类英文气体标签；`id`、`meta` 仅用于追溯。

训练继续使用 `official_a/public/train.jsonl`，验证使用独立的 `official_a/public/validation.jsonl`，两者不能合并。续训从原始基线或之后实际选定的检查点开始。
