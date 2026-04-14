from pathlib import Path

import torch

from _common import (
    ATTACK_CANONICAL,
    add_common_attack_args,
    add_dataset_args,
    add_refine_training_args,
    attack_config,
    build_attack,
    build_refine_defense,
    build_resnet18,
    default_refine_schedule,
    get_dataset_spec,
    infer_attack_checkpoint,
    load_datasets,
    load_model_checkpoint,
    manual_refine_eval,
    parse_basic_args,
    refine_output_root,
    set_global_seed,
)


def main(cli_args=None):
    parser = parse_basic_args("Train REFINE for a specific attack on a supported dataset.")
    add_dataset_args(parser)
    add_refine_training_args(parser)
    add_common_attack_args(parser, include_reflection=True)
    parser.add_argument("--attack", choices=tuple(ATTACK_CANONICAL.keys()), required=True)
    parser.add_argument("--attack-checkpoint", default=None)
    parser.add_argument("--auto-eval", dest="auto_eval", action="store_true", help="Automatically evaluate REFINE after training.")
    parser.add_argument("--no-auto-eval", dest="auto_eval", action="store_false", help="Disable post-training auto evaluation.")
    parser.add_argument(
        "--eval-best-ba",
        dest="eval_best_ba",
        action="store_true",
        help="During auto-eval, choose the checkpoint with best BA among current run checkpoints.",
    )
    parser.add_argument(
        "--no-eval-best-ba",
        dest="eval_best_ba",
        action="store_false",
        help="During auto-eval, evaluate the final in-memory checkpoint only.",
    )
    parser.add_argument(
        "--keep-only-best-checkpoint",
        action="store_true",
        help="After selecting best BA checkpoint, delete other ckpt_epoch_*.pth files in this run directory.",
    )
    parser.set_defaults(auto_eval=True)
    parser.set_defaults(eval_best_ba=True)
    args = parser.parse_args(cli_args)

    set_global_seed(args.seed)
    attack_name = args.attack.lower()
    spec = get_dataset_spec(args.dataset)
    config = attack_config(attack_name, args=args)

    trainset, testset = load_datasets(args.dataset, args.data_root, attack_name=attack_name)
    attack_model = build_resnet18(spec["num_classes"])
    checkpoint = args.attack_checkpoint or infer_attack_checkpoint(
        args.experiment_root,
        attack_name,
        poisoned_rate=config["poisoned_rate"],
        dataset_name=args.dataset,
    )
    load_model_checkpoint(attack_model, checkpoint)

    defense = build_refine_defense(attack_model, num_classes=spec["num_classes"], seed=args.seed)
    save_dir = refine_output_root(args.experiment_root, attack_name, "train", dataset_name=args.dataset)
    schedule = default_refine_schedule(args, attack_name, save_dir, dataset_name=args.dataset)
    train_work_dir = defense.train_unet(trainset, testset, schedule)
    train_work_dir = Path(train_work_dir)

    if not args.auto_eval:
        return

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

    device = torch.device("cuda:0" if args.device == "GPU" and torch.cuda.is_available() else "cpu")

    selected_checkpoint = None
    selected_arr_path = train_work_dir / "label_shuffle.pth"
    selected_defense = defense

    if args.eval_best_ba:
        ckpt_paths = sorted(
            train_work_dir.glob("ckpt_epoch_*.pth"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if ckpt_paths:
            best_item = None
            for ckpt_path in ckpt_paths:
                candidate_defense = build_refine_defense(
                    attack_model,
                    num_classes=spec["num_classes"],
                    checkpoint_path=ckpt_path,
                    arr_path=selected_arr_path,
                    seed=args.seed,
                )
                c_ba, t_ba, m_ba = manual_refine_eval(candidate_defense, testset, device, args.batch_size)
                print(f"[AUTO-EVAL] Candidate {ckpt_path.name}: BA={m_ba:.6f} ({c_ba}/{t_ba})")
                if best_item is None or m_ba > best_item[0]:
                    best_item = (m_ba, c_ba, t_ba, ckpt_path, candidate_defense)

            if best_item is not None:
                ba_metric, correct_ba, total_ba, selected_checkpoint, selected_defense = best_item
                print(f"[AUTO-EVAL] Selected best BA checkpoint: {selected_checkpoint.name}")

                if args.keep_only_best_checkpoint:
                    for ckpt_path in ckpt_paths:
                        if ckpt_path != selected_checkpoint:
                            ckpt_path.unlink(missing_ok=True)
        else:
            correct_ba, total_ba, ba_metric = manual_refine_eval(selected_defense, testset, device, args.batch_size)
    else:
        correct_ba, total_ba, ba_metric = manual_refine_eval(selected_defense, testset, device, args.batch_size)

    if not args.eval_best_ba or selected_checkpoint is None:
        # Fallback to final in-memory model when checkpoint scanning is disabled or unavailable.
        correct_ba, total_ba, ba_metric = manual_refine_eval(selected_defense, testset, device, args.batch_size)

    correct_asr, total_asr, asr_metric = manual_refine_eval(
        selected_defense,
        poisoned_testset,
        device,
        args.batch_size,
        y_target=config["y_target"],
        ignore_target=True,
        label_dataset=testset,
    )

    print(f"[AUTO-EVAL] BA: {ba_metric:.6f} ({correct_ba}/{total_ba})")
    print(f"[AUTO-EVAL] ASR_NoTarget: {asr_metric:.6f} ({correct_asr}/{total_asr})")

    eval_log = train_work_dir / "results_eval.txt"
    eval_lines = [
        f"attack={ATTACK_CANONICAL[attack_name]}",
        f"attack_checkpoint={checkpoint}",
        f"train_dir={train_work_dir}",
        f"selected_checkpoint={selected_checkpoint if selected_checkpoint else 'in_memory_final'}",
        f"label_shuffle={selected_arr_path}",
        f"y_target={config['y_target']}",
        f"poisoned_rate={config['poisoned_rate']}",
        f"BA={ba_metric:.6f} ({correct_ba}/{total_ba})",
        f"ASR_NoTarget={asr_metric:.6f} ({correct_asr}/{total_asr})",
    ]
    eval_log.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
    print(f"[AUTO-EVAL] Saved: {eval_log}")


if __name__ == "__main__":
    main()
