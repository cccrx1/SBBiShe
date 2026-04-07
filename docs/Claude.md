# Claude.md

本文件用于统一 AI 协作上下文，减少多轮对话中的路径、命令和口径漂移。

## 项目定位

- 论文最小主线：GTSRB + ResNet18 + BadNets/Blended/WaNet/Refool + REFINE。
- 数据读取方式：`DatasetFolder`。

## 固定路径约定

- 服务器项目根目录：`/root/SBBiShe`
- 数据目录：`/root/SBBiShe/datasets`
- 实验目录：`/root/SBBiShe/experiments`

## 关键前置条件

1. 先准备 testset 分类目录：

```bash
python /root/SBBiShe/scripts/prepare_gtsrb_testset.py --data-root /root/SBBiShe/datasets
```

2. Refool 需要反射图目录：

- `/root/SBBiShe/datasets/refool_reflections`
- 支持 `.jpg/.jpeg/.png/.bmp`

## 参数与命名规则

- `--attack` 使用小写：`badnets|blended|wanet|refool`
- 命令必须显式传 `--poisoned-rate`
- 训练目录命名已带投毒率后缀，例如：`pr0p05`

## 日志与评估口径

- 训练日志会显式记录：`poisoned_rate`、`y_target`
- 评估结果 `results.txt` 会显式记录：`poisoned_rate`、`y_target`
- `ASR_NoTarget` 口径：仅统计“原始标签不等于目标标签”的 poisoned test 样本

## 强制避免串模型

- 跨投毒率实验时，建议在 REFINE 训练和评估中显式传：
  - `--attack-checkpoint`
  - `--refine-checkpoint`（评估时可选但推荐）
  - `--arr-path`（评估时可选但推荐）

## 推荐读取顺序

1. `docs/项目索引文档.md`
2. `docs/实验指令文档.md`
3. `docs/结果文档.md`
