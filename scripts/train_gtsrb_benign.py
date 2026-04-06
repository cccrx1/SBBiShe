from _common import (
    add_attack_training_args,
    add_common_attack_args,
    benign_output_root,
    build_attack,
    default_attack_schedule,
    load_gtsrb_datasets,
    parse_basic_args,
    set_global_seed,
)


def main():
    parser = parse_basic_args("Train a benign ResNet18 baseline on GTSRB.")
    add_attack_training_args(parser)
    add_common_attack_args(parser, include_reflection=False)
    args = parser.parse_args()

    set_global_seed(args.seed)
    trainset, testset = load_gtsrb_datasets(args.data_root, attack_name="badnets")
    trainer = build_attack("badnets", trainset, testset, args.experiment_root, seed=args.seed, args=args)

    save_dir = benign_output_root(args.experiment_root)
    schedule = default_attack_schedule(
        args,
        benign_training=True,
        attack_name="badnets",
        save_dir=save_dir,
        experiment_name="gtsrb_benign",
    )
    trainer.train(schedule)
    trainer.test(
        {
            "device": args.device,
            "CUDA_VISIBLE_DEVICES": args.gpu_id,
            "GPU_num": 1,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "save_dir": str(save_dir),
            "experiment_name": "gtsrb_benign_eval",
        }
    )


if __name__ == "__main__":
    main()
