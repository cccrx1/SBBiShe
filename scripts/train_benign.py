from _common import (
    add_attack_training_args,
    add_common_attack_args,
    add_dataset_args,
    benign_output_root,
    build_attack,
    dataset_experiment_prefix,
    default_attack_schedule,
    load_datasets,
    parse_basic_args,
    set_global_seed,
)


def main():
    parser = parse_basic_args("Train a benign ResNet18 baseline on a supported dataset.")
    add_dataset_args(parser)
    add_attack_training_args(parser)
    add_common_attack_args(parser, include_reflection=False)
    args = parser.parse_args()

    set_global_seed(args.seed)
    trainset, testset = load_datasets(args.dataset, args.data_root, attack_name="badnets")
    trainer = build_attack(
        "badnets",
        trainset,
        testset,
        args.experiment_root,
        dataset_name=args.dataset,
        seed=args.seed,
        args=args,
    )

    dataset_prefix = dataset_experiment_prefix(args.dataset)
    save_dir = benign_output_root(args.experiment_root, dataset_name=args.dataset)
    schedule = default_attack_schedule(
        args,
        benign_training=True,
        attack_name="badnets",
        save_dir=save_dir,
        experiment_name=f"{dataset_prefix}_benign",
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
            "experiment_name": f"{dataset_prefix}_benign_eval",
        }
    )


if __name__ == "__main__":
    main()
