from _common import (
    attack_output_root,
    build_attack,
    default_attack_schedule,
    load_gtsrb_datasets,
    parse_basic_args,
    set_global_seed,
)


def main():
    parser = parse_basic_args("Train BadNets on GTSRB.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    trainset, testset = load_gtsrb_datasets(args.data_root, attack_name="badnets")
    attack = build_attack("badnets", trainset, testset, args.experiment_root, seed=args.seed)

    save_dir = attack_output_root(args.experiment_root, "badnets")
    schedule = default_attack_schedule(
        args,
        benign_training=False,
        attack_name="badnets",
        save_dir=save_dir,
        experiment_name="gtsrb_badnets",
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
            "experiment_name": "gtsrb_badnets_eval",
        }
    )


if __name__ == "__main__":
    main()
