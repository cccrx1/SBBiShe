# BackdoorBox 项目索引

## 1. 项目一句话定位

这是一个以 `core/` 为主库、以 `tests/` 为示例入口的后门攻击与防御研究工具箱；AI 工具理解本项目时，应优先把它看作“可复用 Python 库 + 实验脚本集合”，而不是单纯的命令行程序。

## 2. 顶层结构速览

```text
.
|-- core/            # 主库：攻击、防御、模型、工具函数
|-- tests/           # 示例脚本与实验入口，不是传统单元测试
|-- README.md        # 项目背景、方法列表、快速说明
|-- requirements.txt # Python 依赖
`-- PROJECT_INDEX.md # 面向 AI 工具的结构化索引
```

## 3. 核心模块说明

### `core/`

- 主代码入口目录。
- 推荐从 `import core` 开始使用，因为 [`core/__init__.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/__init__.py) 聚合导出了 `attacks`、`defenses`、`models`。

### `core/attacks/`

- 后门攻击实现目录。
- 大多数攻击类继承 [`core/attacks/base.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/attacks/base.py) 中的 `Base`。
- 该基类提供统一训练/测试流程、日志与 checkpoint 约定、数据集支持检查、常用 `schedule` 字段约定。

### `core/defenses/`

- 后门防御实现目录。
- 多数防御类继承 [`core/defenses/base.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/defenses/base.py) 中的 `Base`。
- 防御模块接口比攻击模块更不完全统一，但通常围绕 `test()` 和部分 `get_model()` 展开。

### `core/models/`

- 模型定义目录。
- 主要导出项来自 [`core/models/__init__.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/models/__init__.py)：`ResNet`、`BaselineMNISTNetwork`、`AutoEncoder`、`UNet`、`UNetLittle`，以及 `vgg.py` 中导出的 VGG 族。

### `core/utils/`

- 通用工具目录。
- 主要导出项来自 [`core/utils/__init__.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/utils/__init__.py)：`Log`、`PGD`、`any2tensor`、`test`、`accuracy`、`SupConLoss`。
- 常用于日志记录、张量转换、评估、辅助攻击过程。

### `tests/`

- 这里的 `test_*.py` 更接近“使用示例 / 实验脚本”，不是只做断言的最小单测。
- 想理解某个模块如何实例化、如何组织 `schedule`、如何准备数据集，优先看这里。
- 代表入口：
  - [`tests/test_BadNets.py`](/c:/Users/17672/Documents/Projects/SBBiShe/tests/test_BadNets.py)
  - [`tests/test_ShrinkPad.py`](/c:/Users/17672/Documents/Projects/SBBiShe/tests/test_ShrinkPad.py)

## 4. 公共 API / 统一接口

### 推荐导入方式

- 推荐使用：
  - `import core`
  - `core.BadNets`
  - `core.ShrinkPad`
  - `core.models.ResNet`

### 攻击基类通用能力

攻击基类位于 [`core/attacks/base.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/attacks/base.py)，多数攻击类共享以下接口：

- `train(schedule=None)`
  - 根据 `schedule` 训练模型。
  - 训练 benign 或 poisoned 数据由 `schedule['benign_training']` 控制。
- `test(schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None, test_loss=None)`
  - 在 benign / poisoned 测试集上评估。
- `get_model()`
  - 返回当前攻击对象持有的模型。
- `get_poisoned_dataset()`
  - 返回 `(poisoned_train_dataset, poisoned_test_dataset)`。

### 攻击数据集支持范围

[`core/attacks/base.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/attacks/base.py) 中的 `support_list` 当前主要支持：

- `torchvision.datasets.DatasetFolder`
- `torchvision.datasets.MNIST`
- `torchvision.datasets.CIFAR10`

如果新增攻击模块，通常要兼容这一套数据集约定，或同步更新 `support_list` 与相关处理逻辑。

### 常见 `schedule` 字段

训练/测试强依赖 `schedule` 字典。高频字段包括：

- 设备相关：`device`、`CUDA_VISIBLE_DEVICES`、`CUDA_SELECTED_DEVICES`
- 数据相关：`batch_size`、`num_workers`
- 训练相关：`lr`、`momentum`、`weight_decay`、`gamma`、`schedule`、`epochs`
- 日志与保存：`log_iteration_interval`、`test_epoch_interval`、`save_epoch_interval`、`save_dir`、`experiment_name`
- 模型相关：`pretrain`、`test_model`
- 评估相关：`metric`、`y_target`

不同方法可能扩展自己的附加字段，但大多数脚本都围绕这组核心键组织。

### 防御模块常见能力

防御基类位于 [`core/defenses/base.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/defenses/base.py)，只统一了随机种子/确定性设置；具体防御类的公开接口更依赖各自实现。

常见模式：

- `test(...)`
  - 基本所有防御都会提供，用于评估防御后的表现或检测结果。
- `get_model()`
  - 只在部分“模型修复型防御”中提供，例如 `ABL`、`CutMix`、`FineTuning`、`MCR`、`NAD`、`Pruning`。

### 重要例外接口

- `Blind`
  - `get_model(return_NC=False)`：可返回额外 NC 相关模型。
  - `get_poisoned_dataset(NC=False)`：支持与 NC 逻辑相关的数据集获取。
  - `test(..., model=None, nc_model=None, test_dataset=None)`：测试接口比基类更特化。
- `IAD`
  - 除 `get_model()` 外，还有 `get_modelM()`、`get_modelG()`。
  - 说明该攻击内部不止一个核心网络组件。
- `ISSBA`
  - 重写了 `get_model()`、`get_poisoned_dataset()`、`test()`。
- `LIRA`
  - `test(..., model=None, atkmodel=None, ...)` 额外涉及攻击模型对象。
- `MCR`
  - `test(self, dataset, schedule, coeffs_t)` 需要额外的曲线/系数输入，不是普通单参数测试接口。
- `ShrinkPad`
  - 还包含 `preprocess()` / `predict()` 这一类预处理型防御操作，适合输入级评估脚本调用。

## 5. 典型入口与调用路径

### 攻击工作流

标准路径如下：

1. 构造数据集
2. 构造模型与损失函数
3. 实例化攻击类
4. 调用 `train()`
5. 调用 `test()`
6. 如需中间产物，再调用 `get_model()` 或 `get_poisoned_dataset()`

对应代表脚本：

- [`tests/test_BadNets.py`](/c:/Users/17672/Documents/Projects/SBBiShe/tests/test_BadNets.py)

该类脚本通常包含：

- 数据集准备与 transform 定义
- trigger `pattern` / `weight` 构造
- 攻击实例化
- `schedule` 组织
- 训练与评估

### 防御工作流

标准路径如下：

1. 加载已训练模型或模型路径
2. 根据需要构造 benign / poisoned 数据集
3. 实例化防御类
4. 调用 `test()` 或特化评估接口
5. 如属于模型修复型防御，再通过 `get_model()` 取回修复模型

对应代表脚本：

- [`tests/test_ShrinkPad.py`](/c:/Users/17672/Documents/Projects/SBBiShe/tests/test_ShrinkPad.py)

该类脚本通常会把攻击模块与防御模块串起来：先构造 poisoned dataset，再用防御方法评估 BA / ASR。

## 6. 攻击模块清单

以下列表以 [`core/attacks/__init__.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/attacks/__init__.py) 的正式导出为准。

- `BadNets`
  - 经典 patch 型 poison-only 后门攻击，也是很多其他攻击/示例的基准参考。
- `Blended`
  - 通过透明混合方式注入 trigger 的隐蔽型 poison-only 攻击。
- `Refool`
  - 基于反射效果的自然触发器攻击，强调自然感与样本相关性。
- `WaNet`
  - 基于图像 warping 的不可感知攻击，核心是网格形变触发。
- `LabelConsistent`
  - clean-label 攻击，重点是在不改标签的前提下注入后门。
- `Blind`
  - 训练控制型攻击，带有额外 NC 相关组件与特化接口。
- `IAD`
  - input-aware 动态后门攻击，内部包含多个模型部件。
- `LIRA`
  - 可学习、隐蔽、鲁棒的攻击方案，接口中区分主模型与攻击模型。
- `PhysicalBA`
  - 面向物理场景的攻击实现，继承自 `BadNets`，适合看作 patch 攻击的物理扩展。
- `ISSBA`
  - 样本特异性隐蔽攻击，重写了模型、数据集和测试接口。
- `TUAP`
  - 面向通用/优化触发器方向的攻击实现。
- `SleeperAgent`
  - clean-label 场景下的隐藏触发器攻击，强调从零训练时的隐蔽性。
- `BATT`
  - 基于变换触发器的攻击方法，强调触发器变换鲁棒性。
- `AdaptivePatch`
  - 面向自适应场景的 patch 型攻击，用于挑战潜在可分性类防御假设。
- `BAAT`
  - 基于属性触发器的样本特异 clean-label 攻击。

## 7. 防御模块清单

以下列表以 [`core/defenses/__init__.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/defenses/__init__.py) 的正式导出为准。

- `AutoEncoderDefense`
  - 基于自编码器的输入预处理防御。
- `ShrinkPad`
  - 通过缩放与随机 padding 破坏角落触发器位置的预处理型防御。
- `FineTuning`
  - 经典微调式模型修复防御。
- `MCR`
  - 基于 mode connectivity / 曲线模型思想的修复型防御。
- `NAD`
  - 基于注意力蒸馏的后门消除方法。
- `Pruning`
  - 通过神经元/通道裁剪抑制后门行为的修复型防御。
- `ABL`
  - Anti-Backdoor Learning，偏向 poisoned data 上的抑制与重训练。
- `CutMix`
  - 使用 CutMix 思想进行训练阶段防御/修复。
- `IBD_PSC`
  - 输入级后门检测方法，基于参数缩放一致性。
- `SCALE_UP`
  - 黑盒输入级检测方法，基于缩放后预测一致性。
- `REFINE`
  - 基于模型重编程的预处理式防御。
- `FLARE`
  - 面向更通用数据净化场景的数据集 purification 防御。

## 8. 模型与工具模块

### 模型层

- `ResNet`
  - 最常见的视觉主干网络，很多攻击/防御示例默认使用它。
- `BaselineMNISTNetwork`
  - 面向 MNIST 的轻量基线网络。
- `AutoEncoder`
  - 用于自编码器相关防御或预处理流程。
- `UNet` / `UNetLittle`
  - 用于需要生成式或变换式网络结构的方法。
- `vgg.py`
  - VGG 系列模型定义；虽然 `__all__` 没单列具体名称，但通过 `from .vgg import *` 导出。

### 工具层

- `Log`
  - 实验日志记录。
- `accuracy`
  - 分类精度计算。
- `test`
  - 防御侧常用测试辅助函数。
- `any2tensor`
  - 数据转张量辅助函数。
- `PGD`
  - 来自 `torchattacks` 子目录的攻击工具，常用于生成对抗样本等流程。
- `SupConLoss`
  - 对比学习损失实现。

## 9. AI 使用建议

### 当你是代码助手并需要快速理解项目时

- 先读 [`core/__init__.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/__init__.py)
  - 确认主导出入口与顶层使用方式。
- 再读 [`core/attacks/base.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/attacks/base.py)
  - 这是理解攻击模块统一范式的最高价值文件。
- 再读 [`core/attacks/__init__.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/attacks/__init__.py) 和 [`core/defenses/__init__.py`](/c:/Users/17672/Documents/Projects/SBBiShe/core/defenses/__init__.py)
  - 确认正式公开的方法列表。
- 最后读对应 `tests/` 脚本
  - 这是理解真实调用路径、参数组织、数据准备方式的最快路径。

### 当你要新增攻击方法时

- 优先对照同类攻击文件的构造参数命名。
- 尽量兼容 `train()` / `test()` / `get_model()` / `get_poisoned_dataset()` 这一套范式。
- 如果方法需要多模型、多阶段训练或特化测试接口，应在类内显式提供额外 getter，而不是隐含塞进单一接口。
- 如果新增数据集支持，注意同步处理 `support_list`、transform、poisoned dataset 构造逻辑。

### 当你要新增防御方法时

- 先判断它属于哪一类：
  - 输入预处理
  - 输入级检测
  - 模型修复
  - 数据净化
- 如果是模型修复型，建议提供 `get_model()`，方便下游继续联调。
- 如果测试签名与现有通用模式差异较大，应在文档和示例脚本中明确写出额外入参。

## 10. 已知特殊点 / 注意事项

- `tests/` 命名虽然是 `test_*`，但多数不是传统 CI 单元测试，而是长流程实验脚本。
- README 中给出的高层结构是正确的，但实际“如何使用”更多依赖 `tests/` 脚本而不是命令行参数。
- 当前仓库根目录没有统一 CLI 封装；主使用方式是直接运行 Python 脚本或在本地代码中 `import core`。
- 攻击模块的统一性明显高于防御模块；防御模块常因方法类别不同而出现不同 `test()` 签名。
- `PhysicalBA` 继承自 `BadNets`，说明部分攻击之间是“在既有攻击范式上做扩展”，不是完全独立实现。
- 如果 AI 要做仓库级改动，最好先确认目标模块是“库源码”还是“示例脚本”，避免只改了 `tests/` 而没改真正复用逻辑。
