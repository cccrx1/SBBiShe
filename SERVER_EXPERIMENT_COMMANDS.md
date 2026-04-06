# 服务器完整实验命令（按 0.1 / 0.05 / 0.01）

本文件用于服务器直接执行，项目根目录固定为：

- /root/SBBiShe

说明：

- 所有命令都使用绝对路径，避免路径错误。
- 按顺序一条条执行，不使用批量循环。
- 已支持在日志和结果中记录 poisoned_rate 与 y_target。
- 你当前选择的投毒率为：0.1、0.05、0.01。

## 0. 环境准备

```bash
cd /root/SBBiShe
source /root/miniconda3/etc/profile.d/conda.sh || true
conda activate gtsrb-refine
```

## 1. 准备 GTSRB testset（只需一次）

```bash
python /root/SBBiShe/scripts/prepare_gtsrb_testset.py --data-root /root/SBBiShe/datasets
```

## 2. 训练 benign 基线（只需一次）

```bash
python /root/SBBiShe/scripts/train_gtsrb_benign.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10
```

## 3. 投毒率 0.1

### 3.1 攻击训练

```bash
python /root/SBBiShe/scripts/train_gtsrb_badnets.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_blended.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_wanet.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_refool.py \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.1
```

### 3.2 REFINE 训练与评估

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack badnets \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack badnets \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack blended \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack blended \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack wanet \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack wanet \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack refool \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.1
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack refool \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.1
```

## 4. 投毒率 0.05

### 4.1 攻击训练

```bash
python /root/SBBiShe/scripts/train_gtsrb_badnets.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_blended.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_wanet.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_refool.py \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.05
```

### 4.2 REFINE 训练与评估

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack badnets \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack badnets \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack blended \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack blended \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack wanet \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack wanet \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack refool \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.05
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack refool \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.05
```

## 5. 投毒率 0.01

### 5.1 攻击训练

```bash
python /root/SBBiShe/scripts/train_gtsrb_badnets.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_blended.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_wanet.py \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/train_gtsrb_refool.py \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.01
```

### 5.2 REFINE 训练与评估

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack badnets \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack badnets \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack blended \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack blended \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack wanet \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack wanet \
  --data-root /root/SBBiShe/datasets \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/train_refine_gtsrb.py \
  --attack refool \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 8 \
  --test-interval 10 \
  --save-interval 10 \
  --poisoned-rate 0.01
```

```bash
python /root/SBBiShe/scripts/eval_refine_gtsrb.py \
  --attack refool \
  --data-root /root/SBBiShe/datasets \
  --reflection-dir /root/SBBiShe/datasets/refool_reflections \
  --experiment-root /root/SBBiShe/experiments \
  --gpu-id 0 \
  --batch-size 128 \
  --poisoned-rate 0.01
```

## 6. 快速检查是否带投毒率后缀

```bash
ls -lt /root/SBBiShe/experiments/attacks/BadNets | head
ls -lt /root/SBBiShe/experiments/refine/BadNets/train | head
ls -lt /root/SBBiShe/experiments/refine/BadNets/eval | head
```

你应看到类似：

- gtsrb_badnets_pr0p1_...
- gtsrb_badnets_pr0p05_...
- gtsrb_badnets_pr0p01_...
- gtsrb_refine_badnets_train_pr0p05_...
- gtsrb_refine_badnets_eval_pr0p05_latest
