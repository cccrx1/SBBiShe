# 实验产物说明

本仓库的实验结果默认写入 `experiments/`。

## 目录含义

- `experiments/benign/`
  - GTSRB benign 基线模型、训练日志
- `experiments/attacks/BadNets/`
  - BadNets 攻击模型及评估日志
- `experiments/attacks/Blended/`
  - Blended 攻击模型及评估日志
- `experiments/attacks/WaNet/`
  - WaNet 攻击模型、评估日志、`grids/` 下的 `identity_grid` / `noise_grid`
- `experiments/attacks/Refool/`
  - Refool 攻击模型及评估日志
- `experiments/refine/<Attack>/train/`
  - REFINE 训练得到的 `ckpt_epoch_*.pth` 与 `label_shuffle.pth`
- `experiments/refine/<Attack>/eval/`
  - REFINE 最终评估结果

## 论文整理时优先保留

- 每个攻击的最终 checkpoint
- 每个攻击对应的 REFINE checkpoint
- 每个攻击对应的 `label_shuffle.pth`
- `results.txt` 或 `log.txt`
- WaNet 的 grid 文件

## 评估指标

- `BA`
  - Benign Accuracy
- `ASR_NoTarget`
  - 对原始标签不等于目标标签的 poisoned test samples 统计攻击成功率
