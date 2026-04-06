from _common import (
    ATTACK_CANONICAL,
    DEFAULT_REFLECTION_DIR,
    build_attack,
    build_refine_defense,
    build_resnet18,
    default_refine_schedule,
    infer_attack_checkpoint,
    load_gtsrb_datasets,
    load_model_checkpoint,
    parse_basic_args,
    refine_output_root,
    set_global_seed,
)


def main():
    parser = parse_basic_args("Train REFINE for a specific GTSRB attack.")
    parser.add_argument("--attack", choices=tuple(ATTACK_CANONICAL.keys()), required=True)
    parser.add_argument("--attack-checkpoint", default=None)
    parser.add_argument("--reflection-dir", default=str(DEFAULT_REFLECTION_DIR))
    args = parser.parse_args()

    set_global_seed(args.seed)
    attack_name = args.attack.lower()
    trainset, testset = load_gtsrb_datasets(args.data_root, attack_name=attack_name)

    attack_model = build_resnet18()
    checkpoint = args.attack_checkpoint or infer_attack_checkpoint(args.experiment_root, attack_name)
    load_model_checkpoint(attack_model, checkpoint)

    attack = build_attack(
        attack_name,
        trainset,
        testset,
        args.experiment_root,
        reflection_dir=args.reflection_dir,
        model=attack_model,
        seed=args.seed,
    )
    poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()

    defense = build_refine_defense(attack_model, seed=args.seed)
    save_dir = refine_output_root(args.experiment_root, attack_name, "train")
    schedule = default_refine_schedule(args, attack_name, save_dir)
    defense.train_unet(poisoned_trainset, poisoned_testset, schedule)


if __name__ == "__main__":
    main()
