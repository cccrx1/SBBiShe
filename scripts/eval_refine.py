import argparse
from pathlib import Path

import torch

from _common import (
    ATTACK_CANONICAL,
    add_common_attack_args,
    add_dataset_args,
    attack_config,
    build_attack,
    build_refine_defense,
    build_resnet18,
    dataset_experiment_prefix,
    get_dataset_spec,
    infer_attack_checkpoint,
    infer_refine_artifacts,
    load_datasets,
    load_model_checkpoint,
    manual_refine_eval,
    poisoned_rate_tag,
    refine_output_root,
    set_global_seed,
    to_path,
)


def write_eval_log(
    log_dir,
    attack_name,
    attack_checkpoint,
    refine_checkpoint,
    arr_path,
    y_target,
    poisoned_rate,
    ba_metric,
    asr_metric,
    correct_ba,
    total_ba,
    correct_asr,
    total_asr,
    raw_ba_metric=None,
    raw_asr_metric=None,
    raw_correct_ba=None,
    raw_total_ba=None,
    raw_correct_asr=None,
    raw_total_asr=None,
):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "results.txt"
    lines = [
        f"attack={ATTACK_CANONICAL[attack_name]}",
        f"attack_checkpoint={attack_checkpoint}",
        f"refine_checkpoint={refine_checkpoint}",
        f"label_shuffle={arr_path}",
        f"y_target={y_target}",
        f"poisoned_rate={poisoned_rate}",
        f"BA={ba_metric:.6f} ({correct_ba}/{total_ba})",
        f"ASR_NoTarget={asr_metric:.6f} ({correct_asr}/{total_asr})",
    ]
    if raw_ba_metric is not None:
        lines.extend(
            [
                f"RAW_BA={raw_ba_metric:.6f} ({raw_correct_ba}/{raw_total_ba})",
                f"RAW_ASR_NoTarget={raw_asr_metric:.6f} ({raw_correct_asr}/{raw_total_asr})",
            ]
        )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cli_args=None):
    parser = argparse.ArgumentParser(description="Evaluate REFINE on a specific attack and dataset.")
    parser.add_argument("--data-root", default="datasets")
    add_dataset_args(parser)
    parser.add_argument("--attack", choices=tuple(ATTACK_CANONICAL.keys()), required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument("--attack-checkpoint", default=None)
    parser.add_argument("--refine-checkpoint", default=None)
    parser.add_argument("--arr-path", default=None)
    parser.add_argument(
        "--raw-eval",
        dest="raw_eval",
        action="store_true",
        help="Additionally evaluate raw UNet+model outputs without label shuffle for diagnosis.",
    )
    parser.add_argument(
        "--no-raw-eval",
        dest="raw_eval",
        action="store_false",
        help="Disable raw output diagnostic evaluation.",
    )
    parser.set_defaults(raw_eval=True)
    add_common_attack_args(parser, include_reflection=True)
    args = parser.parse_args(cli_args)

    set_global_seed(args.seed)
    attack_name = args.attack.lower()
    spec = get_dataset_spec(args.dataset)
    config = attack_config(attack_name, args=args)
    y_target = config["y_target"]
    poisoned_rate = config["poisoned_rate"]

    trainset, testset = load_datasets(args.dataset, args.data_root, attack_name=attack_name)

    attack_model = build_resnet18(spec["num_classes"])
    attack_checkpoint = args.attack_checkpoint or infer_attack_checkpoint(
        args.experiment_root,
        attack_name,
        poisoned_rate=poisoned_rate,
        dataset_name=args.dataset,
    )
    load_model_checkpoint(attack_model, attack_checkpoint)

    attack = build_attack(
        attack_name,
        trainset,
        testset,
        args.experiment_root,
        dataset_name=args.dataset,
        reflection_dir=args.reflection_dir,
        model=attack_model,
        seed=args.seed,
        args=args,
    )
    _, poisoned_testset = attack.get_poisoned_dataset()

    if args.refine_checkpoint and args.arr_path:
        refine_checkpoint = to_path(args.refine_checkpoint)
        arr_path = to_path(args.arr_path)
    else:
        refine_checkpoint, arr_path, _ = infer_refine_artifacts(
            args.experiment_root,
            attack_name,
            poisoned_rate=poisoned_rate,
            dataset_name=args.dataset,
        )

    defense = build_refine_defense(
        attack_model,
        num_classes=spec["num_classes"],
        checkpoint_path=refine_checkpoint,
        arr_path=arr_path,
        seed=args.seed,
    )

    device = torch.device("cuda:0" if args.device == "GPU" and torch.cuda.is_available() else "cpu")
    correct_ba, total_ba, ba_metric = manual_refine_eval(defense, testset, device, args.batch_size)
    correct_asr, total_asr, asr_metric = manual_refine_eval(
        defense,
        poisoned_testset,
        device,
        args.batch_size,
        y_target=y_target,
        ignore_target=True,
        label_dataset=testset,
    )

    print(f"BA: {ba_metric:.6f} ({correct_ba}/{total_ba})")
    print(f"ASR_NoTarget: {asr_metric:.6f} ({correct_asr}/{total_asr})")

    raw_metrics = None
    if args.raw_eval:
        raw_correct_ba, raw_total_ba, raw_ba_metric = manual_refine_eval(
            defense,
            testset,
            device,
            args.batch_size,
            raw_output=True,
        )
        raw_correct_asr, raw_total_asr, raw_asr_metric = manual_refine_eval(
            defense,
            poisoned_testset,
            device,
            args.batch_size,
            y_target=y_target,
            ignore_target=True,
            label_dataset=testset,
            raw_output=True,
        )
        raw_metrics = {
            "raw_ba_metric": raw_ba_metric,
            "raw_asr_metric": raw_asr_metric,
            "raw_correct_ba": raw_correct_ba,
            "raw_total_ba": raw_total_ba,
            "raw_correct_asr": raw_correct_asr,
            "raw_total_asr": raw_total_asr,
        }
        print(f"RAW_BA: {raw_ba_metric:.6f} ({raw_correct_ba}/{raw_total_ba})")
        print(f"RAW_ASR_NoTarget: {raw_asr_metric:.6f} ({raw_correct_asr}/{raw_total_asr})")

    dataset_prefix = dataset_experiment_prefix(args.dataset)
    eval_root = refine_output_root(args.experiment_root, attack_name, "eval", dataset_name=args.dataset)
    write_eval_log(
        eval_root / f"{dataset_prefix}_refine_{attack_name}_eval_{poisoned_rate_tag(poisoned_rate)}_latest",
        attack_name,
        Path(attack_checkpoint),
        Path(refine_checkpoint),
        Path(arr_path),
        y_target,
        poisoned_rate,
        ba_metric,
        asr_metric,
        correct_ba,
        total_ba,
        correct_asr,
        total_asr,
        **(raw_metrics or {}),
    )


if __name__ == "__main__":
    main()
