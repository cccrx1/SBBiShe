# BackdoorBox Thesis Index

## 项目定位

这是一个专门服务于本科毕业论文的最小实验仓库，只保留 `GTSRB + ResNet18 + BadNets/Blended/WaNet/Refool + REFINE` 主线。

## 当前结构

```text
.
|-- core/              # 最小库源码
|   |-- attacks/       # BadNets / Blended / WaNet / Refool
|   |-- defenses/      # REFINE
|   |-- models/        # ResNet / UNet / UNetLittle
|   `-- utils/         # Log / accuracy / any2tensor / test / SupConLoss
|-- scripts/           # 论文专用训练与评估入口
|-- README.md          # 使用说明
|-- EXPERIMENTS.md     # 结果产物说明
|-- SMOKE_TEST.md      # 本地 1 epoch 验证命令
|-- THESIS_GUIDE.md    # 项目与论文实验总手册
|-- requirements.txt   # 最小依赖
`-- PROJECT_INDEX.md   # AI 导向索引
```

## 数据相关事实

- 当前仓库按 `DatasetFolder` 读取 GTSRB。
- `train/` 必须是按类别分目录结构。
- 官方原始 `testset/` 通常是扁平目录，不能直接给 `DatasetFolder`。
- 现在仓库提供了 [`scripts/prepare_gtsrb_testset.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/prepare_gtsrb_testset.py)，可根据 `GT-final_test.csv` 在 `testset/` 下生成按类别分目录的副本。
- 图片扩展名现在支持：`.ppm`、`.png`、`.jpg`、`.jpeg`、`.bmp`。

## 推荐入口

- 整理 GTSRB testset：[`scripts/prepare_gtsrb_testset.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/prepare_gtsrb_testset.py)
- 训练 benign：[`scripts/train_gtsrb_benign.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/train_gtsrb_benign.py)
- 训练 4 个攻击：
  - [`scripts/train_gtsrb_badnets.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/train_gtsrb_badnets.py)
  - [`scripts/train_gtsrb_blended.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/train_gtsrb_blended.py)
  - [`scripts/train_gtsrb_wanet.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/train_gtsrb_wanet.py)
  - [`scripts/train_gtsrb_refool.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/train_gtsrb_refool.py)
- 训练 REFINE：[`scripts/train_refine_gtsrb.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/train_refine_gtsrb.py)
- 评估 REFINE：[`scripts/eval_refine_gtsrb.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/eval_refine_gtsrb.py)

## 公共接口

- 攻击类统一接口来自 [`core/attacks/base.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/attacks/base.py)
  - `train()`
  - `test()`
  - `get_model()`
  - `get_poisoned_dataset()`
- 唯一保留的防御类是 [`core/defenses/REFINE.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/defenses/REFINE.py)
  - `train_unet()`
  - `test()`
  - `preprocess()`

## 参数覆盖能力

- 训练脚本支持命令行覆盖默认超参。
- 常用参数包括：
  - `--epochs`
  - `--lr`
  - `--schedule`
  - `--disable-schedule`
  - `--batch-size`
  - `--num-workers`
  - `--y-target`
  - `--poisoned-rate`
- 具体以各脚本 `--help` 为准。

## AI 使用建议

- 先读 [`scripts/_common.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/_common.py)
  - 这里集中定义了数据加载、攻击配置、路径规则、GTSRB 准备逻辑和 checkpoint 查找逻辑。
- 再读 [`SMOKE_TEST.md`](/c:/Users/17672/Documents/Projects/SBBiShe/SMOKE_TEST.md)
  - 这里记录了最短验证路径和命令。
- 若要快速了解整个项目演进、环境和实验约定，读 [`THESIS_GUIDE.md`](/c:/Users/17672/Documents/Projects/SBBiShe/THESIS_GUIDE.md)。
- 若看 REFINE 主线，优先读 [`scripts/train_refine_gtsrb.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/train_refine_gtsrb.py) 和 [`scripts/eval_refine_gtsrb.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/eval_refine_gtsrb.py)。
