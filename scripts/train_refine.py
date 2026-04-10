from _common import (
    ATTACK_CANONICAL,
    add_common_attack_args,
    add_dataset_args,
    add_refine_training_args,
    attack_config,
    build_refine_defense,
    build_resnet18,
    default_refine_schedule,
    get_dataset_spec,
    infer_attack_checkpoint,
    load_datasets,
    load_model_checkpoint,
    parse_basic_args,
    refine_output_root,
    set_global_seed,
)


def main():
    parser = parse_basic_args("Train REFINE for a specific attack on a supported dataset.")
    add_dataset_args(parser)
    add_refine_training_args(parser)
    add_common_attack_args(parser, include_reflection=True)
    parser.add_argument("--attack", choices=tuple(ATTACK_CANONICAL.keys()), required=True)
    parser.add_argument("--attack-checkpoint", default=None)
    args = parser.parse_args()

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
    defense.train_unet(trainset, testset, schedule)


if __name__ == "__main__":
    main()
