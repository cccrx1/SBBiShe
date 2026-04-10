from _common import (
    ATTACK_CANONICAL,
    add_attack_training_args,
    add_common_attack_args,
    add_dataset_args,
    attack_output_root,
    build_attack,
    dataset_experiment_prefix,
    default_attack_schedule,
    ensure_wanet_grids,
    get_dataset_spec,
    load_datasets,
    parse_basic_args,
    set_global_seed,
)


def main(cli_args=None):
    parser = parse_basic_args("Train an attack model on a supported dataset.")
    add_dataset_args(parser)
    add_attack_training_args(parser)
    add_common_attack_args(parser, include_reflection=True)
    parser.add_argument("--attack", choices=tuple(ATTACK_CANONICAL.keys()), required=True)
    args = parser.parse_args(cli_args)

    set_global_seed(args.seed)
    attack_name = args.attack.lower()
    spec = get_dataset_spec(args.dataset)

    if attack_name == "wanet":
        ensure_wanet_grids(
            args.experiment_root,
            dataset_name=args.dataset,
            image_size=spec["image_size"],
            k=args.wanet_grid_k or 4,
        )

    trainset, testset = load_datasets(args.dataset, args.data_root, attack_name=attack_name)
    attack = build_attack(
        attack_name,
        trainset,
        testset,
        args.experiment_root,
        dataset_name=args.dataset,
        reflection_dir=args.reflection_dir,
        seed=args.seed,
        args=args,
    )

    dataset_prefix = dataset_experiment_prefix(args.dataset)
    save_dir = attack_output_root(args.experiment_root, attack_name)
    schedule = default_attack_schedule(
        args,
        benign_training=False,
        attack_name=attack_name,
        save_dir=save_dir,
        experiment_name=f"{dataset_prefix}_{attack_name}",
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
            "experiment_name": f"{dataset_prefix}_{attack_name}_eval",
        }
    )


if __name__ == "__main__":
    main()
