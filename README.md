# BackdoorBox GTSRB Thesis Edition

这是一个为本科毕业论文裁剪后的最小实验仓库，只保留 `GTSRB + ResNet18 + 4 attacks + REFINE` 主线。

## 保留内容

- 攻击：`BadNets`、`Blended`、`WaNet`、`Refool`
- 防御：`REFINE`
- 模型：`ResNet18`、`UNetLittle`
- 数据集：`GTSRB`

## 数据目录

默认目录如下：

```text
datasets/
|-- GTSRB/
|   |-- train/
|   |   `-- 00000/ ... 00042/
|   `-- testset/
|       |-- GT-final_test.csv
|       |-- 00000.ppm ...
|       `-- 00000/ ... 00042/   # 运行准备脚本后生成
`-- refool_reflections/
```

说明：

- `train/` 使用官方按类别分目录的结构。
- `testset/` 原始下载通常是扁平目录，仓库脚本无法直接把它当作 `DatasetFolder` 使用。
- 你需要先运行一次准备脚本，把 `testset/` 中的图片按 `GT-final_test.csv` 复制到类别子目录中。

## 先准备 GTSRB testset

在仓库根目录运行：

```bash
python scripts/prepare_gtsrb_testset.py --data-root datasets
```

完成后，`datasets/GTSRB/testset/` 下会多出 `00000` 到 `00042` 这些类别目录。

## 环境依赖

推荐 Python 3.10。

```bash
pip install -r requirements.txt
```

## 论文脚本入口

所有论文入口都在 `scripts/`：

- `prepare_gtsrb_testset.py`
- `train_gtsrb_benign.py`
- `train_gtsrb_badnets.py`
- `train_gtsrb_blended.py`
- `train_gtsrb_wanet.py`
- `train_gtsrb_refool.py`
- `train_refine_gtsrb.py`
- `eval_refine_gtsrb.py`

## 推荐实验顺序

1. 准备 testset
2. 跑 benign
3. 跑攻击
4. 跑 REFINE 训练
5. 跑 REFINE 评估

更具体的本地 smoke test 命令见 [`SMOKE_TEST.md`](/c:/Users/17672/Documents/Projects/SBBiShe/SMOKE_TEST.md)。

完整的项目与论文实验说明见 [`THESIS_GUIDE.md`](/c:/Users/17672/Documents/Projects/SBBiShe/THESIS_GUIDE.md)。

## 本地与服务器分工建议

- 本地 3060 6G：改代码、检查数据路径、做 1 epoch smoke test、确认脚本可启动。
- 服务器 32G：完整训练 4 种攻击、训练 REFINE、产出最终日志与 checkpoint。

## 结果目录约定

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
