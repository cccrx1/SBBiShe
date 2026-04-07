# Claude.md

本文档用于统一当前项目的 AI 协作上下文，减少多轮对话中的路径、命令和评估口径漂移。

## 项目主线

- 数据集：GTSRB
- 模型：ResNet18
- 攻击：BadNets、Blended、WaNet、Refool
- 防御：REFINE

## 固定路径约定

- 本地仓库根目录：`C:\Users\17672\Documents\Projects\SBBiShe`
- 服务器仓库根目录：`/root/SBBiShe`
- 数据目录：`datasets/` 或 `/root/SBBiShe/datasets`
- 实验目录：`experiments/` 或 `/root/SBBiShe/experiments`

## 关键前置条件

### GTSRB 测试集

原始 GTSRB `testset` 往往是扁平目录，必须先执行：

```bash
python scripts/prepare_gtsrb_testset.py --data-root datasets
```

或：

```bash
python /root/SBBiShe/scripts/prepare_gtsrb_testset.py --data-root /root/SBBiShe/datasets
```

### Refool 反射图

Refool 依赖单独的反射图目录：

- `datasets/refool_reflections`
- `/root/SBBiShe/datasets/refool_reflections`

## 当前实验口径

### 攻击训练

攻击模型正常使用 poisoned dataset 训练。

### REFINE 训练

REFINE 训练必须使用 clean `trainset/testset`，不能直接使用 `poisoned_trainset/poisoned_testset`。

原因：

- `REFINE.py` 内部用攻击模型预测作为伪标签
- 若输入本身是 poisoned dataset，训练目标可能会保留后门预测

### REFINE 评估

- `BA`：在 clean testset 上评估
- `ASR_NoTarget`：在 poisoned testset 上评估，但统计时使用原始 clean 标签

## 命令参数约定

- `--attack` 统一使用小写：`badnets|blended|wanet|refool`
- 多投毒率实验必须显式传入 `--poisoned-rate`
- 推荐显式绑定：
  - `--attack-checkpoint`
  - `--refine-checkpoint`
  - `--arr-path`

## 推荐阅读顺序

1. `项目索引文档.md`
2. `实验指令文档.md`
3. `结果文档.md`
