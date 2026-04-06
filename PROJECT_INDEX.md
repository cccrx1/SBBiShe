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
|-- requirements.txt   # 最小依赖
`-- PROJECT_INDEX.md   # AI 导向索引
```

## 推荐入口

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

## 数据与结果约定

- 数据目录：
  - `datasets/GTSRB/train`
  - `datasets/GTSRB/testset`
  - `datasets/refool_reflections`
- 结果目录：
  - `experiments/benign`
  - `experiments/attacks/<Attack>`
  - `experiments/refine/<Attack>/train`
  - `experiments/refine/<Attack>/eval`

## AI 使用建议

- 先读 [`scripts/_common.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/_common.py)
  - 这里集中定义了数据加载、攻击配置、路径规则和 checkpoint 查找逻辑。
- 再读 [`core/attacks/base.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/attacks/base.py)
  - 理解攻击统一训练流程。
- 若看 REFINE 主线，优先读 [`scripts/train_refine_gtsrb.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/train_refine_gtsrb.py) 和 [`scripts/eval_refine_gtsrb.py`](/c:/Users/17672/Documents/Projects/SBBiShe/scripts/eval_refine_gtsrb.py)。
