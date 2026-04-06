from _common import (
    DEFAULT_REFLECTION_DIR,
    add_attack_training_args,
    add_common_attack_args,
    attack_output_root,
    build_attack,
    default_attack_schedule,
    load_gtsrb_datasets,
    parse_basic_args,
    set_global_seed,
)


def main():
    parser = parse_basic_args("Train Refool on GTSRB.")
    add_attack_training_args(parser)
    add_common_attack_args(parser, include_reflection=True)
    args = parser.parse_args()

    set_global_seed(args.seed)
    trainset, testset = load_gtsrb_datasets(args.data_root, attack_name="refool")
    attack = build_attack(
        "refool",
        trainset,
        testset,
        args.experiment_root,
        reflection_dir=args.reflection_dir,
        seed=args.seed,
        args=args,
    )

    save_dir = attack_output_root(args.experiment_root, "refool")
    schedule = default_attack_schedule(
        args,
        benign_training=False,
        attack_name="refool",
        save_dir=save_dir,
        experiment_name="gtsrb_refool",
    )
    attack.train(schedule)
    attack.test(
        {
            "device": args.device,
            "CUDA_VISIBLE_DEVICES": args.gpu_id,
            "GPU_num": 1,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "save_dir": str(save_dir),
            "experiment_name": "gtsrb_refool_eval",
        }
    )


if __name__ == "__main__":
    main()
