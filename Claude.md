# Claude.md

本文档用于统一当前项目的 AI 协作上下文，减少多轮对话中的路径、命令、实验口径和结果解释漂移。后续若要继续在本项目上协作，应默认先遵守本文件，再参考其他文档与脚本实现。

## 1. 项目定位

本仓库是面向本科毕业论文的最小实验版本，目标不是保留完整通用 backdoor toolbox，而是围绕一条固定、可复现的实验链路开展工作。

当前主线固定为：

- 数据集：GTSRB
- 分类模型：ResNet18
- 攻击：BadNets、Blended、WaNet、Refool
- 防御：REFINE

默认实验流程：

1. 训练 benign 基线
2. 训练 4 种攻击模型
3. 针对每种攻击训练 REFINE
4. 评估 BA 和 ASR_NoTarget

如果后续任务与这条主线冲突，优先确认是否属于“论文主线实验”还是“单独消融/临时调试”。

## 2. 仓库结构速览

```text
SBBiShe/
|-- core/
|   |-- attacks/      # 4 种攻击实现与攻击基类
|   |-- defenses/     # REFINE 与防御基类
|   |-- models/       # ResNet18 / UNetLittle
|   `-- utils/        # 日志、精度、测试等工具
|-- scripts/          # 真实实验入口脚本
|-- datasets/         # GTSRB 与 Refool 反射图
|-- experiments/      # 训练产物与评估结果
|-- Claude.md
|-- 项目索引文档.md
|-- 实验指令文档.md
`-- 结果文档.md
```

补充说明：

- 本仓库没有标准 `README.md`，项目说明主要依赖根目录这几份中文文档。
- 实验相关的真实入口在 `scripts/`，不是示例脚本集合。
- 若要判断“项目现在到底按什么规则运行”，优先看 `scripts/_common.py`，其次看具体训练/评估脚本。

## 3. 固定路径约定

- 本地仓库根目录：`C:\Users\17672\Documents\Projects\SBBiShe`
- 服务器仓库根目录：`/root/SBBiShe`
- 数据目录：`datasets/` 或 `/root/SBBiShe/datasets`
- 实验目录：`experiments/` 或 `/root/SBBiShe/experiments`
- Refool 反射图目录：`datasets/refool_reflections` 或 `/root/SBBiShe/datasets/refool_reflections`

默认情况下：

- `scripts/_common.py` 中 `DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"`
- `DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments"`
- `DEFAULT_REFLECTION_DIR = DEFAULT_DATA_ROOT / "refool_reflections"`

因此如果没有特别需要，命令里可以不显式传这些参数；但做正式实验、多组对照或跨机器复现时，建议显式传入。

## 4. 关键代码入口

### 4.1 共用脚本逻辑

最重要的统一入口：

- `scripts/_common.py`

它负责定义或统一：

- 默认路径
- GTSRB 数据加载
- 攻击名称规范
- 各攻击默认参数
- WaNet grid 生成与复用
- Refool 反射图加载
- 输出目录规则
- attack/refine checkpoint 自动推断
- REFINE 手工评估逻辑

只要命令、默认值、目录行为、指标计算出现疑问，优先核对这个文件。

### 4.2 训练/评估脚本

- `scripts/prepare_gtsrb_testset.py`：整理原始 GTSRB 测试集目录
- `scripts/train_gtsrb_benign.py`：训练 benign 基线
- `scripts/train_gtsrb_badnets.py`：训练 BadNets
- `scripts/train_gtsrb_blended.py`：训练 Blended
- `scripts/train_gtsrb_wanet.py`：训练 WaNet
- `scripts/train_gtsrb_refool.py`：训练 Refool
- `scripts/train_refine_gtsrb.py`：针对指定攻击训练 REFINE
- `scripts/eval_refine_gtsrb.py`：评估 REFINE 的 BA / ASR_NoTarget

### 4.3 核心模块

- `core/__init__.py`：聚合导出项目最小可用接口
- `core/attacks/base.py`：攻击统一基类
- `core/defenses/base.py`：防御统一基类
- `core/defenses/REFINE.py`：REFINE 主实现
- `core/models/resnet.py`：GTSRB 主分类器 ResNet18
- `core/models/unet.py`：REFINE 使用的 `UNetLittle`

## 5. 数据与前置条件

### 5.1 GTSRB 测试集必须先整理

原始 GTSRB `testset` 往往是扁平目录，而本项目使用 `torchvision.datasets.DatasetFolder` 读取，要求测试集必须整理成分类子目录结构。

必须先执行：

```bash
python scripts/prepare_gtsrb_testset.py --data-root datasets
```

服务器对应命令：

```bash
python /root/SBBiShe/scripts/prepare_gtsrb_testset.py --data-root /root/SBBiShe/datasets
```

脚本/代码中的硬性约束：

- `scripts/_common.py` 的 `load_gtsrb_datasets()` 会检查：
  - `datasets/GTSRB/train` 是否存在且非空
  - `datasets/GTSRB/testset` 是否存在
  - 如果 `testset` 仍是 raw flat layout，会直接报错并提示先运行 `prepare_gtsrb_testset.py`

### 5.2 Refool 依赖反射图目录

Refool 需要单独的反射图目录：

- `datasets/refool_reflections`
- `/root/SBBiShe/datasets/refool_reflections`

如果目录不存在、没有图像或图像无法读取，`scripts/_common.py` 中的 `load_reflection_images()` 会直接报错。

### 5.3 AI 协作前的默认检查清单

在跑训练、评估或修改脚本前，默认先确认：

- `datasets/GTSRB/train` 存在且有分类子目录
- `datasets/GTSRB/testset` 已整理成分类目录结构
- 若涉及 Refool：`datasets/refool_reflections` 存在且含有效图像
- `experiments/` 目录可写

## 6. 当前实验口径

### 6.1 攻击训练

攻击模型正常使用 poisoned dataset 训练。

### 6.2 REFINE 训练

REFINE 训练必须使用 clean `trainset/testset`，不能直接使用 `poisoned_trainset/poisoned_testset`。

原因：

- `REFINE.py` 内部使用攻击模型预测作为伪标签
- 若输入本身是 poisoned dataset，训练目标可能会保留后门预测
- 当前论文主线正是通过 clean-data training 检验 REFINE 是否真正削弱触发器效果

`scripts/train_refine_gtsrb.py` 当前实现也与该口径一致：

- 读取 clean GTSRB `trainset/testset`
- 加载攻击模型 checkpoint
- 实例化 REFINE
- 调用 `defense.train_unet(trainset, testset, schedule)`

因此除非用户明确说要复现旧设置或做消融，否则不要把 REFINE 训练切回 poisoned dataset。

### 6.3 REFINE 评估

- `BA`：在 clean testset 上评估
- `ASR_NoTarget`：在 poisoned testset 上评估，但统计时使用原始 clean 标签
- 并且忽略原始标签本来就等于目标类的样本

`scripts/eval_refine_gtsrb.py` 和 `_common.py:manual_refine_eval()` 已按这个口径实现。

## 7. 指标解释约定

### BA

`BA` 表示 benign accuracy，即模型在 clean testset 上的分类准确率。

### ASR_NoTarget

`ASR_NoTarget` 表示在 poisoned testset 上，仅统计原始 clean 标签不等于目标标签的样本，计算这些样本被攻击到目标标签的比例。

必须固定这一点：

- 不能把 poisoned label 直接当评估真值
- 不能把“目标标签命中率”误写成普通分类准确率
- 结果分析时应始终与同攻击、同投毒率、同 checkpoint 组进行对照

## 8. 命令与参数约定

### 8.1 攻击名称

`--attack` 统一使用小写：

- `badnets`
- `blended`
- `wanet`
- `refool`

内部 canonical 名称映射在 `scripts/_common.py` 中定义为：

- `badnets -> BadNets`
- `blended -> Blended`
- `wanet -> WaNet`
- `refool -> Refool`

### 8.2 常用参数

通用参数：

- `--data-root`
- `--experiment-root`
- `--gpu-id`
- `--device`
- `--batch-size`
- `--num-workers`
- `--seed`

训练相关：

- `--epochs`
- `--lr`
- `--gamma`
- `--schedule`
- `--disable-schedule`

攻击相关：

- `--poisoned-rate`
- `--y-target`
- `--trigger-size`
- `--blended-alpha`
- `--wanet-grid-k`
- `--reflection-dir`
- `--reflection-limit`

产物绑定：

- `--attack-checkpoint`
- `--refine-checkpoint`
- `--arr-path`

### 8.3 当前默认攻击配置

来自 `scripts/_common.py` 的默认值：

- badnets：`y_target=1`，`poisoned_rate=0.05`
- blended：`y_target=1`，`poisoned_rate=0.05`
- wanet：`y_target=0`，`poisoned_rate=0.1`，`grid_k=4`
- refool：`y_target=1`，`poisoned_rate=0.05`，`reflection_limit=200`

### 8.4 多投毒率实验的强约定

做多投毒率实验时，必须高度重视产物串组问题。推荐显式传入：

- `--poisoned-rate`
- `--attack-checkpoint`
- `--refine-checkpoint`
- `--arr-path`

原因：

- `_common.py` 会自动推断最新 checkpoint
- 如果目录里同时存在多组 poisoned rate，容易误绑到别的实验产物

## 9. 目录与产物约定

### 9.1 数据目录

- `datasets/GTSRB/train/...`
- `datasets/GTSRB/testset/...`
- `datasets/refool_reflections/...`

### 9.2 结果目录

- `experiments/benign/`
- `experiments/attacks/BadNets/`
- `experiments/attacks/Blended/`
- `experiments/attacks/WaNet/`
- `experiments/attacks/Refool/`
- `experiments/refine/<Attack>/train/`
- `experiments/refine/<Attack>/eval/`

### 9.3 关键产物

攻击训练通常会产生：

- `ckpt_epoch_*.pth`
- 对应日志目录

REFINE 训练通常会产生：

- `ckpt_epoch_*.pth`
- `label_shuffle.pth`
- `log.txt`

REFINE 评估通常会产生：

- `results.txt`

`scripts/eval_refine_gtsrb.py` 会把结果写到：

- `experiments/refine/<Attack>/eval/gtsrb_refine_<attack>_eval_<poisoned_rate_tag>_latest/results.txt`

## 10. 运行流程建议

### 10.1 本地 smoke test

建议优先跑最短链路：`BadNets -> REFINE -> eval`

```bash
python scripts/train_gtsrb_badnets.py --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1
python scripts/train_refine_gtsrb.py --attack badnets --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1
python scripts/eval_refine_gtsrb.py --attack badnets --data-root datasets --gpu-id 0 --batch-size 16
```

### 10.2 正式实验顺序

推荐顺序：

1. benign 基线
2. 4 种攻击模型
3. 针对每种攻击训练 REFINE
4. 评估对应的 REFINE

## 11. 结果核对与异常排查

出现结果异常时，优先按以下顺序检查：

1. 是否加载了与当前 `--poisoned-rate` 对应的攻击 checkpoint
2. 评估时是否显式绑定了 `--refine-checkpoint` 和 `--arr-path`
3. Refool 是否使用了正确的 `--reflection-dir`
4. WaNet 是否使用了与训练时一致的 `grid_k / grid 文件`
5. 比较 `ASR_NoTarget` 时，是否在同一攻击、同一投毒率下做前后对照
6. 如果 REFINE 后 ASR 不降，先确认该次 REFINE 是否确实用 clean dataset 训练

额外注意：

- WaNet 依赖 `identity_grid` 和 `noise_grid`，相关文件会保留在结果目录中
- Refool 对反射图内容较敏感，跨机器复现时需确认目录内容一致

## 12. 推荐阅读顺序

当需要快速恢复上下文时，默认阅读顺序：

1. `Claude.md`
2. `项目索引文档.md`
3. `实验指令文档.md`
4. `结果文档.md`
5. `scripts/_common.py`
6. `core/defenses/REFINE.py`
7. 具体攻击脚本与评估脚本

## 13. AI 协作行为建议

后续若让我继续协助本项目，默认按下面的行为执行：

- 优先沿用现有实验主线，不随意扩展到无关数据集/模型
- 优先复用 `scripts/_common.py` 已有逻辑，而不是重新发明一套参数和路径规则
- 修改训练/评估逻辑时，先检查是否会破坏论文主线口径
- 解释结果时，优先区分：
  - No Defense
  - REFINE-clean
  - 如果有需要，再单独说明旧版 `REFINE-poisoned` 消融
- 写命令时尽量给出本地版和服务器版两种路径
- 做正式实验建议显式绑定 checkpoint；做临时 smoke test 可依赖自动推断

## 14. 一句话总原则

本项目不是通用研究框架，而是服务于论文主线的最小可复现实验仓库；任何修改、运行和结果解释，都应优先服从这条主线。
