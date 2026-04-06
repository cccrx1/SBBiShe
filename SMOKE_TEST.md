# 本地 Smoke Test 命令

以下命令都在仓库根目录执行：

```text
c:\Users\17672\Documents\Projects\SBBiShe
```

## 0. 先准备 GTSRB testset

如果你的 `datasets/GTSRB/testset/` 还是官方原始扁平结构，先执行：

```bash
python scripts/prepare_gtsrb_testset.py --data-root datasets
```

## 1. 跑最小链路：BadNets

```bash
python scripts/train_gtsrb_benign.py --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/train_gtsrb_badnets.py --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/train_refine_gtsrb.py --attack badnets --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/eval_refine_gtsrb.py --attack badnets --data-root datasets --gpu-id 0 --batch-size 16
```

## 2. Blended

```bash
python scripts/train_gtsrb_blended.py --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/train_refine_gtsrb.py --attack blended --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/eval_refine_gtsrb.py --attack blended --data-root datasets --gpu-id 0 --batch-size 16
```

## 3. WaNet

```bash
python scripts/train_gtsrb_wanet.py --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/train_refine_gtsrb.py --attack wanet --data-root datasets --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/eval_refine_gtsrb.py --attack wanet --data-root datasets --gpu-id 0 --batch-size 16
```

## 4. Refool

确保 `datasets/refool_reflections/` 中已经放好了反射图像。

```bash
python scripts/train_gtsrb_refool.py --data-root datasets --reflection-dir datasets/refool_reflections --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/train_refine_gtsrb.py --attack refool --data-root datasets --reflection-dir datasets/refool_reflections --gpu-id 0 --epochs 1 --disable-schedule --batch-size 16 --num-workers 0 --test-interval 1 --save-interval 1

python scripts/eval_refine_gtsrb.py --attack refool --data-root datasets --reflection-dir datasets/refool_reflections --gpu-id 0 --batch-size 16
```

## 5. 结果目录

- benign：`experiments/benign/`
- 攻击：`experiments/attacks/BadNets/`、`Blended/`、`WaNet/`、`Refool/`
- REFINE 训练：`experiments/refine/<Attack>/train/`
- REFINE 评估：`experiments/refine/<Attack>/eval/`
