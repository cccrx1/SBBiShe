# 毕业论文实验手册

## 1. 项目定位

本仓库已经从原始 BackdoorBox 工具箱裁剪为毕业论文专用版本，只保留：

- 数据集：`GTSRB`
- 主干模型：`ResNet18`
- 攻击方法：`BadNets`、`Blended`、`WaNet`、`Refool`
- 防御方法：`REFINE`

论文主线是：

1. 在 GTSRB 上训练 benign 模型
2. 在 GTSRB 上训练 4 种后门攻击模型
3. 分别训练每种攻击对应的 REFINE
4. 评估 `BA` 和 `ASR_NoTarget`

## 2. 当前仓库结构

```text
.
|-- core/
|   |-- attacks/        # 4 个攻击 + base
|   |-- defenses/       # REFINE + base
|   |-- models/         # ResNet / UNet / UNetLittle
|   `-- utils/          # Log / accuracy / any2tensor / test / SupConLoss
|-- scripts/
|   |-- prepare_gtsrb_testset.py
|   |-- train_gtsrb_benign.py
|   |-- train_gtsrb_badnets.py
|   |-- train_gtsrb_blended.py
|   |-- train_gtsrb_wanet.py
|   |-- train_gtsrb_refool.py
|   |-- train_refine_gtsrb.py
|   `-- eval_refine_gtsrb.py
|-- README.md
|-- PROJECT_INDEX.md
|-- EXPERIMENTS.md
|-- SMOKE_TEST.md
`-- THESIS_GUIDE.md
```

## 3. 环境配置

### 本地环境建议

- Python：`3.10`
- 推荐 conda 环境名：`gtsrb-refine`

创建环境：

```bash
conda create -n gtsrb-refine python=3.10
conda activate gtsrb-refine
```

### PyTorch 安装

本项目实验使用 GPU。

你已经验证通过的本地配置是：

- `torch==2.1.2+cu121`
- `torchvision==0.16.2+cu121`

推荐安装方式：

```bash
pip uninstall -y torch torchvision
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

验证 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

正常输出应类似：

```text
2.1.2+cu121
12.1
True
1
```

### 服务器环境建议

- Python：`3.10`
- PyTorch：`2.1.2`
- CUDA 运行时：`cu121`

服务器上同样推荐：

```bash
conda create -n gtsrb-refine python=3.10
conda activate gtsrb-refine
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 4. 数据集准备

### GTSRB 当前支持情况

当前脚本通过 `torchvision.datasets.DatasetFolder` 读取 GTSRB。

已支持图片扩展名：

- `.ppm`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

### 训练集格式

训练集应为按类别分目录结构：

```text
datasets/GTSRB/train/
|-- 00000/
|-- 00001/
...
`-- 00042/
```

### 测试集格式

官方原始 GTSRB 测试集通常是扁平结构：

```text
datasets/GTSRB/testset/
|-- GT-final_test.csv
|-- 00000.ppm
|-- 00001.ppm
...
```

这种结构不能直接给 `DatasetFolder` 使用。

你已经运行过以下命令：

```bash
python scripts/prepare_gtsrb_testset.py --data-root datasets
```

并得到结果：

```text
Prepared 12630 test images into class subdirectories under ...\datasets\GTSRB\testset
```

这说明测试集已经成功整理为可训练结构。

### Refool 反射图

`Refool` 还需要单独准备反射图目录：

```text
datasets/refool_reflections/
```

里面放若干自然图像即可，脚本会读取其中一部分作为 reflection candidates。

## 5. 已实现的脚本能力

### 训练脚本

- `train_gtsrb_benign.py`
- `train_gtsrb_badnets.py`
- `train_gtsrb_blended.py`
- `train_gtsrb_wanet.py`
- `train_gtsrb_refool.py`
- `train_refine_gtsrb.py`

### 评估脚本

- `eval_refine_gtsrb.py`

### 辅助脚本

- `prepare_gtsrb_testset.py`

### 命令行参数覆盖

脚本已经支持“命令行优先，未传则使用默认值”。

目前支持的主要参数包括：

- 训练参数：
  - `--epochs`
  - `--lr`
  - `--gamma`
  - `--schedule`
  - `--disable-schedule`
  - `--momentum`
  - `--weight-decay`
  - `--batch-size`
  - `--num-workers`
  - `--log-interval`
  - `--test-interval`
  - `--save-interval`
- 攻击参数：
  - `--y-target`
  - `--poisoned-rate`
  - `--trigger-size`
  - `--blended-alpha`
  - `--wanet-grid-k`
  - `--wanet-noise`
  - `--no-wanet-noise`
  - `--reflection-dir`
  - `--reflection-limit`
- REFINE 参数：
  - `--betas`
  - `--eps`
  - `--amsgrad`
  - `--attack-checkpoint`
  - `--refine-checkpoint`
  - `--arr-path`

## 6. 本地 smoke test 流程

运行目录：

```text
c:\Users\17672\Documents\Projects\SBBiShe
```

建议先跑最小链路：`benign -> badnets -> refine -> eval`

### 第一步：benign

```bash
python scripts/train_gtsrb_benign.py --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1
```

### 第二步：BadNets

```bash
python scripts/train_gtsrb_badnets.py --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1
```

### 第三步：训练 REFINE

```bash
python scripts/train_refine_gtsrb.py --attack badnets --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1
```

### 第四步：评估 REFINE

```bash
python scripts/eval_refine_gtsrb.py --attack badnets --data-root datasets --gpu-id 0 --batch-size 16
```

其它攻击的命令见 [`SMOKE_TEST.md`](/c:/Users/17672/Documents/Projects/SBBiShe/SMOKE_TEST.md)。

## 7. 实验结果目录

```text
experiments/
|-- benign/
|-- attacks/
|   |-- BadNets/
|   |-- Blended/
|   |-- WaNet/
|   `-- Refool/
`-- refine/
    |-- BadNets/
    |-- Blended/
    |-- WaNet/
    `-- Refool/
```

命名规则：

- benign 训练目录：`gtsrb_benign_<timestamp>`
- 攻击训练目录：`gtsrb_<attack>_<timestamp>`
- 攻击评估目录：`gtsrb_<attack>_eval_<timestamp>`
- REFINE 训练目录：`gtsrb_refine_<attack>_train_<timestamp>`

## 8. 已排查并修复过的问题

### 1. GTSRB 无法读取

原因：

- 原先只允许 `png`
- 你的数据实际是 `.ppm`
- `testset` 还是官方扁平结构

处理：

- 脚本已支持 `.ppm`
- 新增 `prepare_gtsrb_testset.py`

### 2. `This machine has no visible cuda devices!`

原因：

- 当时安装的是 `torch==2.1.2+cpu`

处理：

- 已切换到 `torch==2.1.2+cu121`

### 3. `No trained checkpoint found for BadNets`

原因：

- 自动查找 checkpoint 时误把 `*_eval_*` 目录当成最新目录

处理：

- 已修复查找逻辑，只会选择“包含 checkpoint 的最新训练目录”

### 4. `scatter(): Expected dtype int64 for index`

原因：

- `REFINE.py` 中 `arr_shuffle` 的 dtype 不是 `int64`

处理：

- 已改成显式 `np.int64 / torch.int64`

## 9. 本地与服务器分工

### 本地

- 调试代码
- 跑 1 epoch smoke test
- 检查数据路径和命名
- 验证攻击到 REFINE 的完整链路

### 服务器

- 跑完整轮次训练
- 跑 4 种攻击
- 跑每种攻击对应的 REFINE
- 导出论文表格和最终结果

## 10. 论文实验建议

第一阶段先做：

- 数据集：`GTSRB`
- 攻击：`BadNets`、`Blended`、`WaNet`、`Refool`
- 防御：`REFINE`

评价指标：

- `BA`
- `ASR_NoTarget`

推荐顺序：

1. 先打通 `BadNets -> REFINE`
2. 再扩到 `Blended`
3. 再扩到 `WaNet`
4. 最后做 `Refool`

这样最容易逐步定位问题，也最适合本科论文推进节奏。
