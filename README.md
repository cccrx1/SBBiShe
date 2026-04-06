# BackdoorBox GTSRB Thesis Edition

这是一个为本科毕业论文裁剪后的最小实验仓库，只保留 `GTSRB + ResNet18 + 4 attacks + REFINE` 主线。

## 保留内容

- 攻击：`BadNets`、`Blended`、`WaNet`、`Refool`
- 防御：`REFINE`
- 模型：`ResNet18`、`UNetLittle`
- 数据集：`GTSRB`，通过 `torchvision.datasets.DatasetFolder` 加载

## 数据目录

默认目录如下：

```text
datasets/
|-- GTSRB/
|   |-- train/
|   `-- testset/
`-- refool_reflections/
```

- `datasets/GTSRB/train` 和 `datasets/GTSRB/testset` 为交通标志数据集。
- `datasets/refool_reflections` 只供 `Refool` 使用，放反射图像。

## 环境依赖

推荐 Python 3.8，核心依赖见 `requirements.txt`。

```bash
pip install -r requirements.txt
```

## 脚本入口

所有论文入口都在 `scripts/`：

- `train_gtsrb_benign.py`
- `train_gtsrb_badnets.py`
- `train_gtsrb_blended.py`
- `train_gtsrb_wanet.py`
- `train_gtsrb_refool.py`
- `train_refine_gtsrb.py`
- `eval_refine_gtsrb.py`

## 推荐实验顺序

1. 训练 benign 基线

```bash
python scripts/train_gtsrb_benign.py --data-root datasets --gpu-id 0
```

2. 训练 4 种攻击

```bash
python scripts/train_gtsrb_badnets.py --data-root datasets --gpu-id 0
python scripts/train_gtsrb_blended.py --data-root datasets --gpu-id 0
python scripts/train_gtsrb_wanet.py --data-root datasets --gpu-id 0
python scripts/train_gtsrb_refool.py --data-root datasets --reflection-dir datasets/refool_reflections --gpu-id 0
```

3. 为每种攻击训练 REFINE

```bash
python scripts/train_refine_gtsrb.py --attack badnets --data-root datasets --gpu-id 0
python scripts/train_refine_gtsrb.py --attack blended --data-root datasets --gpu-id 0
python scripts/train_refine_gtsrb.py --attack wanet --data-root datasets --gpu-id 0
python scripts/train_refine_gtsrb.py --attack refool --data-root datasets --reflection-dir datasets/refool_reflections --gpu-id 0
```

4. 评估 REFINE

```bash
python scripts/eval_refine_gtsrb.py --attack badnets --data-root datasets --gpu-id 0
python scripts/eval_refine_gtsrb.py --attack blended --data-root datasets --gpu-id 0
python scripts/eval_refine_gtsrb.py --attack wanet --data-root datasets --gpu-id 0
python scripts/eval_refine_gtsrb.py --attack refool --data-root datasets --reflection-dir datasets/refool_reflections --gpu-id 0
```

## 本地与服务器分工建议

- 本地 3060 6G：改代码、检查数据路径、做小批量 smoke test、确认脚本可启动。
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
