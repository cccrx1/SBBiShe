from pathlib import Path

import argparse
import torch

from _common import (
    ATTACK_CANONICAL,
    DEFAULT_REFLECTION_DIR,
    attack_config,
    build_attack,
    build_refine_defense,
    build_resnet18,
    infer_attack_checkpoint,
    infer_refine_artifacts,
    load_gtsrb_datasets,
    load_model_checkpoint,
    manual_refine_eval,
    refine_output_root,
    set_global_seed,
    to_path,
)


def write_eval_log(log_dir, attack_name, attack_checkpoint, refine_checkpoint, arr_path, ba_metric, asr_metric, correct_ba, total_ba, correct_asr, total_asr):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "results.txt"
    lines = [
        f"attack={ATTACK_CANONICAL[attack_name]}",
        f"attack_checkpoint={attack_checkpoint}",
        f"refine_checkpoint={refine_checkpoint}",
        f"label_shuffle={arr_path}",
        f"BA={ba_metric:.6f} ({correct_ba}/{total_ba})",
        f"ASR_NoTarget={asr_metric:.6f} ({correct_asr}/{total_asr})",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate REFINE on a specific GTSRB attack.")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--attack", choices=tuple(ATTACK_CANONICAL.keys()), required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument("--attack-checkpoint", default=None)
    parser.add_argument("--refine-checkpoint", default=None)
    parser.add_argument("--arr-path", default=None)
    parser.add_argument("--reflection-dir", default=str(DEFAULT_REFLECTION_DIR))
    args = parser.parse_args()

    set_global_seed(args.seed)
    attack_name = args.attack.lower()
    trainset, testset = load_gtsrb_datasets(args.data_root, attack_name=attack_name)

    attack_model = build_resnet18()
    attack_checkpoint = args.attack_checkpoint or infer_attack_checkpoint(args.experiment_root, attack_name)
    load_model_checkpoint(attack_model, attack_checkpoint)

    attack = build_attack(
        attack_name,
        trainset,
        testset,
        args.experiment_root,
        reflection_dir=args.reflection_dir,
        model=attack_model,
        seed=args.seed,
    )
    _, poisoned_testset = attack.get_poisoned_dataset()

    if args.refine_checkpoint and args.arr_path:
        refine_checkpoint = to_path(args.refine_checkpoint)
        arr_path = to_path(args.arr_path)
    else:
        refine_checkpoint, arr_path, _ = infer_refine_artifacts(args.experiment_root, attack_name)

    defense = build_refine_defense(
        attack_model,
        checkpoint_path=refine_checkpoint,
        arr_path=arr_path,
        seed=args.seed,
    )

    device = torch.device("cuda:0" if args.device == "GPU" and torch.cuda.is_available() else "cpu")
    y_target = attack_config(attack_name)["y_target"]
    correct_ba, total_ba, ba_metric = manual_refine_eval(defense, testset, device, args.batch_size)
    correct_asr, total_asr, asr_metric = manual_refine_eval(
        defense,
        poisoned_testset,
        device,
        args.batch_size,
        y_target=y_target,
        ignore_target=True,
    )

    print(f"BA: {ba_metric:.6f} ({correct_ba}/{total_ba})")
    print(f"ASR_NoTarget: {asr_metric:.6f} ({correct_asr}/{total_asr})")

    eval_root = refine_output_root(args.experiment_root, attack_name, "eval")
    write_eval_log(
        eval_root / f"gtsrb_refine_{attack_name}_eval_latest",
        attack_name,
        Path(attack_checkpoint),
        Path(refine_checkpoint),
        Path(arr_path),
        ba_metric,
        asr_metric,
        correct_ba,
        total_ba,
        correct_asr,
        total_asr,
    )


if __name__ == "__main__":
    main()
