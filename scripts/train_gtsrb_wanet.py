from _common import (
    add_attack_training_args,
    add_common_attack_args,
    attack_output_root,
    build_attack,
    default_attack_schedule,
    ensure_wanet_grids,
    load_gtsrb_datasets,
    parse_basic_args,
    set_global_seed,
)


def main():
    parser = parse_basic_args("Train WaNet on GTSRB.")
    add_attack_training_args(parser)
    add_common_attack_args(parser, include_reflection=False)
    args = parser.parse_args()

    set_global_seed(args.seed)
    ensure_wanet_grids(args.experiment_root, k=args.wanet_grid_k or 4)
    trainset, testset = load_gtsrb_datasets(args.data_root, attack_name="wanet")
    attack = build_attack("wanet", trainset, testset, args.experiment_root, seed=args.seed, args=args)

    save_dir = attack_output_root(args.experiment_root, "wanet")
    schedule = default_attack_schedule(
        args,
        benign_training=False,
        attack_name="wanet",
        save_dir=save_dir,
        experiment_name="gtsrb_wanet",
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
            "experiment_name": "gtsrb_wanet_eval",
        }
    )


if __name__ == "__main__":
    main()
